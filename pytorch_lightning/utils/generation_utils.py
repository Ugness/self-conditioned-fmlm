"""Generation helpers — text post-processing + sampling loop + decoding."""

from typing import Optional

import torch

from utils.sampling_utils import (
    cdeq_step, ctm_step, flow_map_step,
    ode_step, sde_step, get_sampling_steps, restore_cond,
)


def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int,
                    pad_token_id: int) -> torch.Tensor:
    """Mask everything at/after the first EOS per row (keeping EOS itself)."""
    eos_mask = predicted_ids == eos_token_id
    keep_mask = torch.cumsum(eos_mask.long(), dim=1) == 0
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor,
               pad_value: int = 0, axis: int = 1) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("x must have at least batch + sequence dims")
    if axis == 0:
        raise ValueError("axis=0 is the batch axis")
    axis = axis if axis >= 0 else x.ndim + axis
    if axis != 1:
        x = x.transpose(1, axis)
    seq_len = x.size(1)
    shift = shift_per_sample.long().to(x.device)
    base_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)
    gather_idx = shift.unsqueeze(1) + base_idx
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.ndim == 2:
        shifted = torch.gather(x, 1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        expand_shape = [-1] * x.ndim
        expand_shape[1] = -1
        gather_idx_expanded = gather_idx
        for _ in range(2, x.ndim):
            gather_idx_expanded = gather_idx_expanded.unsqueeze(-1)
        gather_idx_expanded = gather_idx_expanded.expand_as(x)
        shifted = torch.gather(x, 1, gather_idx_expanded)
        for _ in range(2, x.ndim):
            valid = valid.unsqueeze(-1)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.transpose(1, axis)
    return shifted


@torch.no_grad()
def generate_samples(
    model, z: torch.Tensor, t_steps: torch.Tensor,
    cond_seq: Optional[torch.Tensor], cond_seq_mask: Optional[torch.Tensor],
    *, config, sampling_config, cfg_scale: float, self_cond_cfg_scale: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Run a full sampling trajectory (SDE or ODE inner steps + final ODE step at t=1)."""
    method = sampling_config.sampling_method
    if cond_seq is None:
        cond_seq = torch.zeros_like(z)
        cond_seq_mask = torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    if method == "flowmap":
        # Two flow-map sampler regimes, selected by how the student was trained:
        #   * SC-free student (config.elfmap_student_sc_free): trained with the
        #     self-cond slot h=0 everywhere. The faithful, in-distribution
        #     sampler feeds h=0 at every step -- no SC pre-pass, no chaining.
        #     Total forwards: N for an N-step sampler.
        #   * Legacy hybrid-SC student (DEPRECATED, kept only for old ckpts):
        #     step 0 runs an SC pre-pass; subsequent steps chain the previous
        #     step's x_pred as the SC input. Total forwards: N + 1.
        # Either way the final step lands at t=1 exactly, so we do NOT append an
        # extra ODE step.
        sc_free = getattr(config, "elfmap_student_sc_free", False)
        sc_signal = (restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
                     if sc_free else None)
        for k in range(len(t_steps) - 1):
            z, x_pred = flow_map_step(
                model, z, sc_signal,
                float(t_steps[k].item()), float(t_steps[k + 1].item()),
                config=config, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                self_cond_cfg_scale=self_cond_cfg_scale,
                do_sc_prepass=(False if sc_free else (k == 0)),
            )
            if not sc_free:
                sc_signal = x_pred
        return z

    if method == "ctm":
        # CTM (arXiv:2310.02279) gamma-sampling: each step predicts the clean
        # (t=1) target, mixes the implied current noise with fresh noise by gamma
        # (sqrt(1-g^2)*old + g*fresh), then re-places at t_next. gamma from
        # sampling_config.sde_gamma (0=deterministic ... 1=fully stochastic). SC
        # handling matches flowmap. Final step lands at t=1 exactly.
        gamma = float(getattr(sampling_config, "sde_gamma", 1.0))
        sc_free = getattr(config, "elfmap_student_sc_free", False)
        sc_signal = (restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
                     if sc_free else None)
        for k in range(len(t_steps) - 1):
            z, x_pred = ctm_step(
                model, z, sc_signal,
                float(t_steps[k].item()), float(t_steps[k + 1].item()),
                config=config, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                self_cond_cfg_scale=self_cond_cfg_scale, gamma=gamma,
                generator=generator,
                do_sc_prepass=(False if sc_free else (k == 0)),
            )
            if not sc_free:
                sc_signal = x_pred
        return z

    if method == "cdeq":
        # CDEQ Stage-2 sampler: one student forward per step with
        # a zero SC slot at the cold-start phase, no cross-timestep carry. Inner
        # steps use the SDE churn (sde_gamma); the final step lands at t=1 (no
        # churn) like the sde/ode path.
        gamma = float(getattr(sampling_config, "sde_gamma", 0.0))
        phase = float(getattr(config, "cdeq_inference_phase", 0.002))
        for k in range(len(t_steps) - 2):
            z, _ = cdeq_step(
                model, z, float(t_steps[k].item()), float(t_steps[k + 1].item()),
                config=config, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                self_cond_cfg_scale=self_cond_cfg_scale, phase=phase,
                gamma=gamma, generator=generator,
            )
        z, _ = cdeq_step(
            model, z, float(t_steps[-2].item()), float(t_steps[-1].item()),
            config=config, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
            self_cond_cfg_scale=self_cond_cfg_scale, phase=phase,
            gamma=0.0, generator=generator,
        )
        return z

    inner = len(t_steps) - 2  # last step always ODE
    gamma = float(getattr(sampling_config, "sde_gamma", 0.0))
    for k in range(inner):
        t = float(t_steps[k].item())
        t_next = float(t_steps[k + 1].item())
        if method == "sde":
            z, x_pred = sde_step(
                model, z, t, t_next, x_pred,
                config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                gamma=gamma, generator=generator,
            )
        elif method == "ode":
            z, x_pred = ode_step(
                model, z, t, t_next, x_pred, config=config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
            )
        else:
            raise ValueError(f"Invalid sampling method: {method}")

    t_last = float(t_steps[-2].item())
    t_final = float(t_steps[-1].item())
    z, x_pred = ode_step(
        model, z, t_last, t_final, x_pred, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z


@torch.no_grad()
def dlm_decode_batch(model, z: torch.Tensor, *, config,
                    self_cond_cfg_scale: float, t_final_val: float = 1.0) -> torch.Tensor:
    """Decode z (at t=1) to token ids via the decoder head.

    Under `double_time_emb`, the decoder branch passes s=t=t_final_val so every
    time-token slot receives emb(t_final_val) — identical to the vanilla
    single-time block.
    """
    B = z.size(0)
    t_final = torch.full((B,), float(t_final_val), device=z.device, dtype=z.dtype)
    if config.num_self_cond_cfg_tokens > 0:
        sc_scale = torch.full((B,), float(self_cond_cfg_scale), device=z.device, dtype=z.dtype)
    else:
        sc_scale = None
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    s_val = t_final if getattr(config, "double_time_emb", False) else None
    _, decoder_logits = model(
        z_input, t_final, self_cond_cfg_scale=sc_scale, decoder_step_active=True,
        s=s_val,
    )
    return torch.argmax(decoder_logits, dim=-1)


def build_run_name(sampling_method: str, num_sampling_steps: int, cfg_scale: float,
                   self_cond_cfg_scale: float, time_schedule: str, sde_gamma: float,
                   suffix: str) -> str:
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    # gamma is load-bearing for every stochastic/flow-map sampler (churn for
    # sde/cdeq, mix ratio for ctm); emit it for all so gamma sweeps get distinct
    # run names (previously only "sde" did, collapsing ctm/cdeq sweeps).
    gamma_str = f"-gamma{sde_gamma}" if sampling_method in ("sde", "ctm", "cdeq") else ""
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{gamma_str}-{suffix}"

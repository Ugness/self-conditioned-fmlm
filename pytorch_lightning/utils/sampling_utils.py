"""Sampling utilities for ELF flow-matching generation.

Core formulas:
- add_noise:        z = t*x0 + (1-t)*noise*denoiser_noise_scale  (cond preserved)
- sample_timesteps: sigmoid(N(P_mean, P_std)) or uniform
- sample_cfg_scale: log-uniform in [cfg_min, cfg_max]
- net_out_to_v_x:   v = (x - z) / max(1 - t, t_eps)
- ode_step:         z' = z + (t_next - t) * v
- sde_step:         z_back = alpha*z + (1-alpha)*eps (cond restored); ODE from t_back
"""

import math
from typing import Optional, Tuple

import torch


def add_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, config,
              cond_seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    t_expanded = t.view(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


def sample_timesteps(generator: Optional[torch.Generator], batch_size: int, device,
                     P_mean: float = -0.8, P_std: float = 0.8,
                     time_schedule: str = "logit_normal") -> torch.Tensor:
    if time_schedule == "logit_normal":
        z = torch.randn(batch_size, generator=generator, device=device) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == "uniform":
        return torch.rand(batch_size, generator=generator, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(generator: Optional[torch.Generator], n_steps: int, device,
                       time_schedule: str = "logit_normal",
                       P_mean: float = -0.8, P_std: float = 0.8) -> torch.Tensor:
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if time_schedule == "logit_normal":
        if n_steps < 2:
            return torch.tensor([0.0, 1.0], device=device)
        steps = sample_timesteps(generator, n_steps - 1, device,
                                 P_mean=P_mean, P_std=P_std, time_schedule="logit_normal")
        steps, _ = torch.sort(steps)
        return torch.cat([torch.zeros(1, device=device), steps, torch.ones(1, device=device)])
    if time_schedule == "logit_normal_uniform":
        if n_steps < 2:
            return torch.tensor([0.0, 1.0], device=device)
        K = n_steps
        # 1/K, 2/K, ..., (K-1)/K 의 deterministic percentile 위치
        percentiles = torch.arange(1, K, device=device, dtype=torch.float32) / K
        # standard normal inverse CDF -> shift/scale 로 logit-normal 분포의 quantile
        z = torch.special.ndtri(percentiles) * P_std + P_mean
        # sigmoid는 단조증가이므로 이미 정렬되어 있음
        steps = torch.sigmoid(z)
        return torch.cat([torch.zeros(1, device=device), steps, torch.ones(1, device=device)])
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def sample_cfg_scale(generator: Optional[torch.Generator], batch_size: int, device,
                     cfg_min: float = 0.0, cfg_max: float = 3.0) -> torch.Tensor:
    u = torch.rand(batch_size, generator=generator, device=device)
    a, b = 1.0 + cfg_min, 1.0 + cfg_max
    return a * torch.exp(u * math.log(b / a)) - 1.0


def restore_cond(z_updated: torch.Tensor, cond_seq: torch.Tensor,
                 cond_seq_mask: torch.Tensor) -> torch.Tensor:
    mask = cond_seq_mask
    target_ndim = max(z_updated.ndim, cond_seq.ndim)
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def _restore_vx(v: torch.Tensor, x: torch.Tensor,
                cond_seq: Optional[torch.Tensor],
                cond_seq_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if cond_seq is None:
        return v, x
    return restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask), restore_cond(x, cond_seq, cond_seq_mask)


def _zero_cond(z: torch.Tensor, cond_seq: Optional[torch.Tensor],
               cond_seq_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Build a zero tensor with cond positions restored from `cond_seq`."""
    if cond_seq is None:
        return torch.zeros_like(z)
    return restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)


def net_out_to_v_x(net_out, z: torch.Tensor, t: torch.Tensor,
                   t_eps: float = 5e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert x_pred network output to (v, x). Drops decoder_logits if present."""
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    x = net_out
    v = (x - z) / torch.clamp(1.0 - t.view(-1, 1, 1), min=t_eps)
    return v, x


def _forward_sample_self_cond(
    model, z: torch.Tensor, t_batch: torch.Tensor, x_pred_prev: Optional[torch.Tensor],
    config, self_cond_cfg_scale: float,
    cond_seq: Optional[torch.Tensor], cond_seq_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_eps = config.t_eps

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = _zero_cond(z, cond_seq, cond_seq_mask)
        sc_scale_batch = torch.full((z.size(0),), float(self_cond_cfg_scale),
                                    device=z.device, dtype=z.dtype)
        net_out_cond = model(torch.cat([z, x_pred_prev], dim=-1), t_batch,
                             self_cond_cfg_scale=sc_scale_batch)
        return _restore_vx(*net_out_to_v_x(net_out_cond, z, t_batch, t_eps),
                           cond_seq, cond_seq_mask)

    if config.self_cond_prob == 0:
        return _restore_vx(*net_out_to_v_x(model(z, t_batch), z, t_batch, t_eps),
                           cond_seq, cond_seq_mask)

    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        net_out_uncond = model(torch.cat([z, _zero_cond(z, cond_seq, cond_seq_mask)], dim=-1), t_batch)
        v_uncond, x_uncond = _restore_vx(*net_out_to_v_x(net_out_uncond, z, t_batch, t_eps),
                                         cond_seq, cond_seq_mask)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    net_out_cond = model(torch.cat([z, x_pred_prev], dim=-1), t_batch)
    v_cond, x_cond = _restore_vx(*net_out_to_v_x(net_out_cond, z, t_batch, t_eps),
                                 cond_seq, cond_seq_mask)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return _restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


def _forward_sample(
    model, z: torch.Tensor, t_batch: torch.Tensor, x_pred_prev: Optional[torch.Tensor],
    config, cfg_scale: float, self_cond_cfg_scale: float,
    cond_seq: Optional[torch.Tensor], cond_seq_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    v_cond, x_cond = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (None if x_pred_prev is None
                          else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask))
    v_uncond, x_uncond = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq) if cond_seq is not None else None,
        cond_seq_mask=cond_seq_mask,
    )
    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return _restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


def ode_step(model, z, t, t_next, x_pred_prev, config,
             cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask):
    t_batch = torch.full((z.size(0),), float(t), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model, z, t_batch, x_pred_prev, config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred


@torch.no_grad()
def flow_map_step(model, z, sc_signal, t_curr, t_next, *, config,
                  cond_seq, cond_seq_mask, self_cond_cfg_scale=1.0,
                  do_sc_prepass=False, pred_time=None):
    """One FMLM flow-map step.

    z_{t_next} = (1 - t_next)/(1 - t_curr) · z_{t_curr}
               + (t_next - t_curr)/(1 - t_curr) · x_pred
    where x_pred = net(z, s=t_curr, t=pred_time, sc=sc_signal, w=self_cond_cfg_scale).

    pred_time = the model's target-time argument; None -> t_next (the usual two-time
    flow map). Set pred_time=1.0 for the "consistency" variant (flowmap_churn_CM):
    the model always predicts the clean endpoint at t=1, while the interpolation
    weights below still use t_next, so the update is the Euler / FMLM step toward
    t_next driven by the clean-endpoint estimate.

    Hybrid SC scheme:
      - do_sc_prepass=True  (step 0): compute sc_signal = net(z, s=t=t_curr, sc=zeros, w).
      - do_sc_prepass=False (n >= 1): caller passes the previous step's x_pred as sc_signal.
    """
    B = z.size(0)
    t_curr_b = torch.full((B,), float(t_curr), device=z.device, dtype=z.dtype)
    t_next_b = torch.full((B,), float(t_next), device=z.device, dtype=z.dtype)
    pred_t = t_next if pred_time is None else pred_time
    pred_time_b = torch.full((B,), float(pred_t), device=z.device, dtype=z.dtype)
    sc_scale = (torch.full((B,), float(self_cond_cfg_scale), device=z.device, dtype=z.dtype)
                if config.num_self_cond_cfg_tokens > 0 else None)

    if do_sc_prepass:
        z_in0 = (torch.cat([z, torch.zeros_like(z)], dim=-1)
                 if config.self_cond_prob > 0 else z)
        sc_out, _ = model(z_in0, t_curr_b, self_cond_cfg_scale=sc_scale,
                          decoder_step_active=False, s=t_curr_b)
        sc_signal = restore_cond(sc_out, cond_seq, cond_seq_mask)

    z_in = (torch.cat([z, sc_signal], dim=-1)
            if config.self_cond_prob > 0 else z)
    x_pred, _ = model(z_in, pred_time_b, self_cond_cfg_scale=sc_scale,
                      decoder_step_active=False, s=t_curr_b)
    x_pred = restore_cond(x_pred, cond_seq, cond_seq_mask)

    eps = 1e-7  # tiny divide-by-zero guard ONLY (not config.t_eps=0.05): on the
                # final step t_next=1 the FMLM map needs w_d = (1-t_curr)/(1-t_curr)
                # = 1 exactly (z_next = x_pred). The logit grid puts t_curr > 0.95,
                # so a 0.05 clamp would shrink the clean latent fed to the decoder.
    one_minus_curr = (1.0 - t_curr_b).clamp(min=eps).view(-1, 1, 1)
    w_z = (1.0 - t_next_b).view(-1, 1, 1) / one_minus_curr
    w_d = (t_next_b - t_curr_b).view(-1, 1, 1) / one_minus_curr
    z_next = w_z * z + w_d * x_pred
    z_next = restore_cond(z_next, cond_seq, cond_seq_mask)
    return z_next, x_pred


@torch.no_grad()
def ctm_step(model, z, sc_signal, t_curr, t_next, *, config,
             cond_seq, cond_seq_mask, self_cond_cfg_scale,
             gamma, generator, do_sc_prepass=False):
    """CTM (arXiv:2310.02279) gamma-sampling step, adapted to ELF's linear
    interpolation z_t = t*x0 + (1-t)*eps*scale (t=1 clean, t=0 noise).

    CTM (VE): denoise to noise level sqrt(1-gamma^2)*sigma_{t_next}, then forward-
    diffuse up to sigma_{t_next}; G_theta(x,t,0) is the clean (data) estimate.
    ELF is NOT VE -- its noise level is sigma ∝ (1-t) and the signal coefficient
    shrinks with noise -- so CTM's sqrt(1-gamma^2)/gamma variance split becomes a
    NOISE MIX on the interpolant rather than additive VE noise:

        x_pred  = net(z, s=t_curr, t=1)                      # clean (endpoint) est
        eps_old = (z - t_curr*x_pred) / ((1-t_curr)*scale)   # implied current noise
        eps_mix = sqrt(1-gamma^2)*eps_old + gamma*eps_fresh  # var (1-g^2)+g^2 = 1
        z_next  = t_next*x_pred + (1-t_next)*eps_mix*scale    # re-place at t_next

    gamma=0 -> keep eps_old == deterministic PF-ODE flow using the clean estimate.
    gamma=1 -> fully fresh noise == multistep CM-style stochastic sampling.
    The final step (t_next=1) lands on x_pred exactly (noise coeff 0)."""
    scale = config.denoiser_noise_scale
    if t_next >= 1.0:
        _, x_pred = flow_map_step(
            model, z, sc_signal, t_curr, 1.0, config=config,
            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
            self_cond_cfg_scale=self_cond_cfg_scale, do_sc_prepass=do_sc_prepass,
        )
        return restore_cond(x_pred, cond_seq, cond_seq_mask), x_pred
    
    sigma_target = 1.0 - t_next
    sigma_tilde = sigma_target * torch.sqrt(torch.tensor(1.0 - gamma**2))
    t_tilde = 1.0 - sigma_tilde
    _, x_pred = flow_map_step(
        model, z, sc_signal, t_curr, t_tilde, config=config,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        self_cond_cfg_scale=self_cond_cfg_scale, do_sc_prepass=do_sc_prepass,
    )
    
    g = max(0.0, min(1.0, float(gamma)))
    if g >= 1.0:
        eps_mix = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
    else:
        denom = max(1.0 - t_curr, 1e-6) * scale
        eps_old = (z - t_curr * x_pred) / denom
        if g <= 0.0:
            eps_mix = eps_old
        else:
            eps_fresh = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
            eps_mix = (1.0 - g * g) ** 0.5 * eps_old + g * eps_fresh
    z_next = t_next * x_pred + (1.0 - t_next) * eps_mix * scale
    return restore_cond(z_next, cond_seq, cond_seq_mask), x_pred


def get_flowmap_time_steps(num_steps: int, *, tau_kind: str,
                            P_mean: float, P_std: float,
                            device=None, dtype=torch.float32) -> torch.Tensor:
    """Deterministic time grid 0 = t_0 < ... < t_N = 1.

    Endpoints 0 and 1 are pinned; the interior is reparametrised so the marginal
    time distribution matches the teacher's training-time distribution
    (sigmoid(P_std·logit(tau) + P_mean) for the default logit_normal tau_kind).
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    tau = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    if tau_kind == "identity":
        return tau
    if tau_kind == "logit_normal":
        out = tau.clone()
        if num_steps >= 2:
            interior = tau[1:-1]
            logit = torch.log(interior / (1 - interior))
            out[1:-1] = torch.sigmoid(P_std * logit + P_mean)
        out[0] = 0.0
        out[-1] = 1.0
        return out
    if tau_kind == "logit_normal_uniform":
        # Exact counterpart of "logit_normal" above: probit (inverse normal CDF)
        # in place of logit. Interior points 1/N..(N-1)/N are strictly in (0, 1),
        # so ndtri is finite without clamping.
        out = tau.clone()
        if num_steps >= 2:
            interior = tau[1:-1]
            z = torch.special.ndtri(interior) * P_std + P_mean
            out[1:-1] = torch.sigmoid(z)
        out[0] = 0.0
        out[-1] = 1.0
        return out
    raise ValueError(f"Unknown tau_kind: {tau_kind}")


def sde_step(model, z, t, t_next, x_pred_prev, config,
             cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
             gamma, generator):
    alpha = max(0.0, min(1.0, 1.0 - gamma * (t_next - t)))
    t_back = alpha * t
    eps = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype) * config.denoiser_noise_scale
    z_back = alpha * z + (1.0 - alpha) * eps
    if cond_seq is not None:
        z_back = restore_cond(z_back, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.size(0),), float(t_back), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model, z_back, t_batch, x_pred_prev, config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z_back + (t_next - t_back) * v_pred, x_pred


@torch.no_grad()
def cdeq_step(model, z, t, t_next, *, config, cond_seq, cond_seq_mask,
              self_cond_cfg_scale, phase, gamma, generator):
    """One CDEQ Stage-2 denoising step.

    A SINGLE student forward with a ZERO self-cond slot at the cold-start phase
    `tau_cold` (=config.cdeq_inference_phase): there c_skip=0 and the degenerate
    student Anderson step reduce g(z,sc=0,tau_0) to the network's x-prediction.
    No cross-timestep SC carry (x_pred is NOT fed into the next step). The SDE
    churn arithmetic is duplicated here (NOT routed through sde_step, which would
    rebuild the SC input); gamma=0 → pure Euler/ODE landing step."""
    B = z.size(0)
    alpha = max(0.0, min(1.0, 1.0 - gamma * (t_next - t)))
    t_back = alpha * t
    if alpha < 1.0:
        eps = torch.randn(z.shape, generator=generator, device=z.device,
                          dtype=z.dtype) * config.denoiser_noise_scale
        z_back = alpha * z + (1.0 - alpha) * eps
        if cond_seq is not None:
            z_back = restore_cond(z_back, cond_seq, cond_seq_mask)
    else:
        z_back = z
    t_b = torch.full((B,), float(t_back), device=z.device, dtype=z.dtype)
    phase_b = torch.full((B,), float(phase), device=z.device, dtype=z.dtype)
    sc_scale = (torch.full((B,), float(self_cond_cfg_scale), device=z.device, dtype=z.dtype)
                if config.num_self_cond_cfg_tokens > 0 else None)
    z_in = (torch.cat([z_back, torch.zeros_like(z_back)], dim=-1)
            if config.self_cond_prob > 0 else z_back)
    x_pred, _ = model(z_in, t_b, self_cond_cfg_scale=sc_scale,
                      decoder_step_active=False, s=t_b, phase=phase_b)
    x_pred = restore_cond(x_pred, cond_seq, cond_seq_mask)
    v_pred, _ = net_out_to_v_x(x_pred, z_back, t_b, config.t_eps)
    return z_back + (t_next - t_back) * v_pred, x_pred

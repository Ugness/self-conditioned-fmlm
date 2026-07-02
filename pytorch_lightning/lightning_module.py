"""ELF as a `pl.LightningModule`.

The training step runs the two-objective ELF update: a Bernoulli(decoder_prob)
branch select, denoiser L2 with optional self-cond CFG guidance, decoder-branch
CE on token ids, EMA updated only on real optimizer steps. DDP / grad
accumulation / AMP / checkpointing are delegated to `pl.Trainer` — see
`train_lightning.py`.
"""

import math
from typing import Any, Dict, List

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT

from configs.config import Config
from encoders import build_encoder
from modules.model import ELF_models
from utils.data_utils import get_pad_token_id, load_dataset, make_dataloader
from utils.encoder_utils import encode_text
from utils.logging_utils import log_for_0
from utils.muon import Muon, build_muon_param_groups
from utils.sampling_utils import (
    add_noise, net_out_to_v_x, restore_cond, sample_cfg_scale, sample_timesteps,
)


# EMA — updated only on real optimizer steps (matches JAX `is_optimizer_step`).
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def _aligned(self, name: str, p: torch.Tensor) -> torch.Tensor:
        """Return the shadow tensor for `name` on `p`'s device, migrating lazily.
        Shadow is built before Lightning moves the model; first call after the
        move copies each entry across."""
        buf = self.shadow[name]
        if buf.device != p.device:
            buf = buf.to(p.device)
            self.shadow[name] = buf
        return buf

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self._aligned(n, p).mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> dict:
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self._aligned(n, p))
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self) -> dict:
        return {"decay": self.decay,
                "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict, device):
        self.decay = state.get("decay", self.decay)
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}


# -----------------------------------------------------------------------------
# LightningModule
# -----------------------------------------------------------------------------
class ELFLitModule(L.LightningModule):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.cfg = config
        self.vocab_size = vocab_size
        self.automatic_optimization = False  # we manage Muon + AdamW + EMA manually
        self._train_generator = None
        self._ema = None
        self._loss_running: Dict[str, List[float]] = {"loss": [], "l2": [], "ce": []}

        # Frozen pretrained encoder.
        self.encoder = build_encoder(
            config.encoder_model_name,
            dtype=torch.float32,
            feature_layer=config.feature_layer,
        )
        encoder_dim = self.encoder.d_model

        # ELF transformer.
        model_fn = ELF_models[config.model]
        self.model = model_fn(
            text_encoder_dim=encoder_dim, max_length=config.max_length,
            attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
            num_time_tokens=config.num_time_tokens,
            num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
            vocab_size=vocab_size,
            num_model_mode_tokens=config.num_model_mode_tokens,
            bottleneck_dim=config.bottleneck_dim,
            self_cond_input=(config.self_cond_prob > 0),
            use_flash=config.use_flash,
            double_time_emb=config.double_time_emb,
            # Phase bank is active for the real CDEQ student (cdeq_enabled) AND for
            # a Stage-3 flow-map student distilling a CDEQ teacher
            # (elfmap_teacher_is_cdeq): the latter mirrors the teacher's phase
            # tokens but pins them to a CONSTANT phase = tau_0 (cdeq_inference_phase)
            # so the warm-started student reproduces the CDEQ teacher's 1-forward
            # diagonal target exactly. The real CDEQ student leaves const_phase
            # None (it conditions on a varying phase during distillation).
            num_phase_tokens=(config.cdeq_num_phase_tokens
                              if (getattr(config, "cdeq_enabled", False)
                                  or getattr(config, "elfmap_teacher_is_cdeq", False))
                              else 0),
            const_phase=(config.cdeq_inference_phase
                         if (getattr(config, "elfmap_teacher_is_cdeq", False)
                             and not getattr(config, "cdeq_enabled", False))
                         else None),
        )

    # --- lifecycle hooks --------------------------------------------------
    def setup(self, stage=None):
        if self._ema is None:
            self._ema = EMA(self.model, decay=self.cfg.ema_decay1)
        if self._train_generator is None:
            seed = self.cfg.seed * 100003 + self.global_rank
            self._train_generator = torch.Generator(device=self.device).manual_seed(seed)
        # Self-maintained step counter: manual-optim mode's `self.global_step`
        # semantics with multiple optimizers (Muon + AdamW) are opaque enough
        # that gating on it can silently miss. We increment this every is_opt_step
        # and use it for the LR schedule and log throttling.
        if not hasattr(self, "_my_opt_step"):
            self._my_opt_step = 0

    # --- optimizer & schedule --------------------------------------------
    def configure_optimizers(self):
        cfg = self.cfg
        if cfg.optimizer == "muon":
            muon_groups, adamw_groups = build_muon_param_groups(
                self.model,
                adamw_betas=(cfg.adam_b1, cfg.adam_b2), weight_decay=cfg.weight_decay,
            )
            muon = Muon(muon_groups, lr=1e-12)
            adamw = torch.optim.AdamW(adamw_groups, lr=1e-12,
                                       betas=(cfg.adam_b1, cfg.adam_b2),
                                       weight_decay=cfg.weight_decay)
            return [muon, adamw]
        elif cfg.optimizer == "adamw":
            return [torch.optim.AdamW(self.model.parameters(), lr=1e-12,
                                       betas=(cfg.adam_b1, cfg.adam_b2),
                                       weight_decay=cfg.weight_decay)]
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # --- training step ----------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        cfg = self.cfg
        gen = self._train_generator
        device = batch["input_ids"].device

        # Clamp ids to model vocab range. Some prepped OWT-GPT2 datasets contain
        # a tiny number of stray pad-token ids equal to len(tokenizer-with-pad)
        # that exceed the model's vocab_size; without this clamp those rows
        # crash the CE-branch logit gather (and the encoder, which has its own
        # clamp). No-op for clean datasets.
        batch["input_ids"] = batch["input_ids"].clamp_max(self.vocab_size - 1)

        # Use our own opt-step counter: with manual optimization + multiple
        # optimizers (Muon + AdamW), Lightning's `global_step` semantics are
        # opaque enough that gating on it can silently miss.
        is_opt_step = (batch_idx + 1) % cfg.grad_accum_steps == 0
        lr = self._lr_at_step(self._my_opt_step)
        for opt in self.optimizers():
            for g in opt.param_groups:
                g["lr"] = lr

        encoder_attention_mask = batch["encoder_attention_mask"]
        if cfg.label_drop_prob > 0:
            B = encoder_attention_mask.size(0)
            drop = (torch.rand(B, generator=gen, device=device)
                    < cfg.label_drop_prob).float().view(B, 1, 1)
            cm = batch["cond_seq_mask"]
            block_mask = (1 - cm).unsqueeze(2) * cm.unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)
            label_drop_mask = drop.squeeze(-1).squeeze(-1).bool()
        else:
            label_drop_mask = torch.zeros(encoder_attention_mask.size(0),
                                          dtype=torch.bool, device=device)

        with torch.no_grad():
            x0 = encode_text(batch["input_ids"], encoder_attention_mask,
                             encoder=self.encoder, latent_mean=cfg.latent_mean,
                             latent_std=cfg.latent_std)
        B, S, _ = x0.shape

        t = sample_timesteps(gen, B, device=device,
                             P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
                             time_schedule=cfg.time_schedule)
        noise = torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)

        cond_seq_mask = batch["cond_seq_mask"].unsqueeze(-1)
        loss_mask = (batch["attention_mask"] if cfg.pad_token == "pad"
                     else torch.ones_like(batch["attention_mask"]))
        loss_mask = loss_mask * (1 - batch["cond_seq_mask"])

        denoiser_z = add_noise(x0, noise, t, cfg, cond_seq_mask=cond_seq_mask)
        if cfg.label_drop_prob > 0:
            drop_zero = ((label_drop_mask.view(-1, 1, 1).float() * (cond_seq_mask > 0).float()) > 0)
            zero = torch.zeros_like(denoiser_z)
            denoiser_z = torch.where(drop_zero, zero, denoiser_z)
            x0 = torch.where(drop_zero, zero, x0)

        decoder_z_vals = (torch.randn(B * S, generator=gen, device=device)
                          * cfg.decoder_p_std + cfg.decoder_p_mean)
        decoder_lambda_t = torch.sigmoid(decoder_z_vals).view(B, S, 1).to(x0.dtype)
        decoder_noise = (torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)
                         * cfg.decoder_noise_scale)
        decoder_z = decoder_lambda_t * x0 + (1.0 - decoder_lambda_t) * decoder_noise

        t_eps = cfg.t_eps
        v_target = (x0 - denoiser_z) / torch.clamp(1.0 - t.view(-1, 1, 1), min=t_eps)

        use_self_cond_mask = None
        if cfg.self_cond_prob > 0:
            use_self_cond_mask = ((torch.rand(B, generator=gen, device=device) < cfg.self_cond_prob)
                                  .view(-1, 1, 1).to(x0.dtype))
        sc_cfg_scale = None
        if cfg.num_self_cond_cfg_tokens > 0:
            sc_cfg_scale = sample_cfg_scale(gen, B, device=device,
                                            cfg_min=cfg.self_cond_cfg_min,
                                            cfg_max=cfg.self_cond_cfg_max).to(x0.dtype)

        decoder_step_active = bool(
            (torch.rand((), generator=gen, device=device) < cfg.decoder_prob).item()
        )

        if decoder_step_active:
            decoder_input = (torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
                             if cfg.self_cond_prob > 0 else decoder_z)
            _, decoder_logits = self.model(decoder_input, torch.ones_like(t),
                                           self_cond_cfg_scale=sc_cfg_scale,
                                           decoder_step_active=True)
            log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
            ce = -log_probs.gather(-1, batch["input_ids"].unsqueeze(-1)).squeeze(-1)
            ce_loss = (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = ce_loss
            l2_loss = torch.zeros((), device=device)
        else:
            if cfg.self_cond_prob > 0:
                with torch.no_grad():
                    z_uncond = restore_cond(torch.zeros_like(denoiser_z), x0, cond_seq_mask)
                    net_out_init = self.model(torch.cat([denoiser_z, z_uncond], dim=-1),
                                              t, self_cond_cfg_scale=sc_cfg_scale)
                    _, x_pred_init = net_out_to_v_x(net_out_init, denoiser_z, t, t_eps)
                    x_pred_init = restore_cond(x_pred_init, x0, cond_seq_mask)
                    x_pred_cond = x_pred_init * use_self_cond_mask.to(denoiser_z.dtype)
                    x_pred_cond = restore_cond(x_pred_cond, x0, cond_seq_mask)
                denoiser_input = torch.cat([denoiser_z, x_pred_cond], dim=-1)
            else:
                denoiser_input = denoiser_z

            net_out = self.model(denoiser_input, t, self_cond_cfg_scale=sc_cfg_scale,
                                 decoder_step_active=False)
            v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, t_eps)

            if cfg.num_self_cond_cfg_tokens > 0:
                with torch.no_grad():
                    z_uncond = restore_cond(torch.zeros_like(denoiser_z), x0, cond_seq_mask)
                    net_out_uncond = self.model(torch.cat([denoiser_z, z_uncond], dim=-1),
                                                t, self_cond_cfg_scale=sc_cfg_scale)
                    v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, denoiser_z, t, t_eps)
                    x_uncond = restore_cond(x_uncond, x0, cond_seq_mask)
                    net_out_cond = self.model(torch.cat([denoiser_z, x_uncond], dim=-1),
                                              t, self_cond_cfg_scale=sc_cfg_scale)
                    v_cond, _ = net_out_to_v_x(net_out_cond, denoiser_z, t, t_eps)
                    sc_w = sc_cfg_scale.view(B, 1, 1).to(v_target.dtype)
                    sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
                    if use_self_cond_mask is not None:
                        sc_guidance = torch.where(use_self_cond_mask > 0, sc_guidance,
                                                  torch.zeros_like(sc_guidance))
                    v_final_target = (v_target + sc_guidance).detach()
            else:
                v_final_target = v_target

            per_token_loss = ((v_pred - v_final_target) ** 2).mean(dim=-1)
            safe = torch.where(loss_mask > 0, per_token_loss, torch.zeros_like(per_token_loss))
            l2_loss = (safe * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = l2_loss
            ce_loss = torch.zeros((), device=device)

        self.manual_backward(loss / cfg.grad_accum_steps)
        self._loss_running["loss"].append(loss.detach())
        self._loss_running["l2"].append(l2_loss.detach())
        self._loss_running["ce"].append(ce_loss.detach())

        if is_opt_step:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            for opt in self.optimizers():
                opt.step()
                opt.zero_grad(set_to_none=True)
            self._ema.update(self.model)
            self._my_opt_step += 1
            if self._my_opt_step % cfg.log_freq == 0:
                self._log_running(lr)

        return loss.detach()

    # --- helpers ---------------------------------------------------------
    def _lr_at_step(self, opt_step: int) -> float:
        cfg = self.cfg
        base_lr = cfg.lr if cfg.lr else cfg.blr * cfg.global_batch_size * cfg.grad_accum_steps / 256
        if cfg.warmup_steps is not None and cfg.warmup_steps > 0:
            num_warmup = cfg.warmup_steps
        elif cfg.warmup_epochs is not None:
            num_warmup = int(cfg.warmup_epochs * getattr(self, "_steps_per_epoch", 1))
        else:
            num_warmup = 0
        if opt_step < num_warmup:
            return base_lr * (opt_step / max(1, num_warmup))
        if cfg.lr_schedule == "cosine":
            total = getattr(self, "_num_optimizer_steps", 1)
            t = (opt_step - num_warmup) / max(1, total - num_warmup)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
            return cfg.min_lr + (base_lr - cfg.min_lr) * cos
        return base_lr

    def _log_running(self, lr: float):
        if not self._loss_running["loss"]: return
        dp = max(1e-8, self.cfg.decoder_prob)
        np_ = max(1e-8, 1.0 - self.cfg.decoder_prob)
        loss = torch.stack(self._loss_running["loss"]).float().mean()
        l2 = torch.stack(self._loss_running["l2"]).float().mean() / np_
        ce = torch.stack(self._loss_running["ce"]).float().mean() / dp
        # Lightning auto-syncs across DDP ranks when sync_dist=True. In manual
        # optimization mode, `self.log` defaults to on_epoch=True/on_step=False,
        # which queues metrics until epoch end — explicitly setting on_step=True
        # makes them flow to wandb every `log_every_n_steps` iterations.
        self.log("train/loss", loss, sync_dist=True, prog_bar=True,
                  on_step=True, on_epoch=False)
        self.log("train/l2_loss", l2, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/ce_loss", ce, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/lr", lr, sync_dist=False,
                  on_step=True, on_epoch=False)
        for k in self._loss_running: self._loss_running[k].clear()

    # --- checkpoint plumbing --------------------------------------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["ema"] = self._ema.state_dict()
        # Persist our own opt-step counter so the LR schedule (warmup + cosine)
        # is continuous across resumes. Without this, _my_opt_step would reset
        # to 0 in setup() and re-trigger warmup from base_lr=0.
        checkpoint["_my_opt_step"] = getattr(self, "_my_opt_step", 0)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self._ema is None:
            self._ema = EMA(self.model, decay=self.cfg.ema_decay1)
        if "ema" in checkpoint:
            self._ema.load_state_dict(checkpoint["ema"], device=self.device)
        if "_my_opt_step" in checkpoint:
            self._my_opt_step = int(checkpoint["_my_opt_step"])
        elif "global_step" in checkpoint:
            # Fallback for checkpoints saved before _my_opt_step was persisted.
            # In manual-optim mode, Lightning's `global_step` increments once
            # per `LightningOptimizer.step()`, so it counts opt steps × #
            # optimizers (Muon + AdamW → 2; AdamW alone → 1).
            num_opts = 2 if self.cfg.optimizer == "muon" else 1
            self._my_opt_step = int(checkpoint["global_step"]) // num_opts


# -----------------------------------------------------------------------------
# FPFlowMap — FMLM-style flow-map training (distill-only).
# -----------------------------------------------------------------------------
class _RunningMean:
    """Sum/count accumulator: ensures zero-contribution steps don't dilute the active branch's mean."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, v, active=True):
        if active:
            self.sum += float(v)
            self.count += 1

    def flush(self):
        v = self.sum / max(self.count, 1)
        n = self.count
        self.sum = 0.0
        self.count = 0
        return v, n


class FPFlowMap(ELFLitModule):
    """ELFLitModule subclass that lifts FMLM flow-map distillation onto ELF.

    Inherits EMA, optimisers, LR schedule, DataModule wiring, and checkpoint
    hooks. Overrides training_step and the running-loss accumulator. Adds a
    frozen teacher (loaded in setup) used for the diagonal target. Default: a
    single-time vanilla ELF. Canonical stage-3 setup (elfmap_teacher_is_cdeq=True,
    e.g. train_owt_FMLM_star.yml) instead uses a two-time CDEQ teacher, i.e. a
    frozen Stage-2 FPFlow checkpoint.

    NOTE: FPFlow (the Stage-2 CDEQ module) subclasses THIS class for code reuse,
    so the class tree is inverted vs the training pipeline order
    (Stage-1 ELFLitModule -> Stage-2 FPFlow -> Stage-3 FPFlowMap).
    """

    def __init__(self, config: Config, vocab_size: int):
        super().__init__(config, vocab_size)
        # Replace the flat lists with sum/count semantics.
        self._loss_running = {
            "loss": _RunningMean(),
            "diag": _RunningMean(),
            "off":  _RunningMean(),
            "ce":   _RunningMean(),
        }
        self._branch_running = {
            "decoder": _RunningMean(),
            "fm_diag": _RunningMean(),
            "fm_off":  _RunningMean(),
            "fm_sc":   _RunningMean(),
        }
        self.teacher_model = None

    # ---------- lifecycle ---------------------------------------------
    def setup(self, stage=None):
        super().setup(stage=stage)
        if stage != "fit":
            return
        if self.teacher_model is not None:
            return  # idempotent
        cfg = self.cfg
        if not cfg.teacher_ckpt_path:
            raise ValueError("FPFlowMap requires cfg.teacher_ckpt_path (distill-only).")
        if not cfg.double_time_emb:
            raise ValueError("FPFlowMap requires cfg.double_time_emb=True.")
        if getattr(cfg, "elfmap_teacher_is_cdeq", False):
            # Stage-3: the teacher is a Stage-2 FPFlow ckpt. It carries the phase
            # bank and a SC slot; the diagonal target uses the CDEQ 1-forward
            # contract (see _teacher_x_pred), so these must hold to build/load it.
            if cfg.cdeq_num_phase_tokens <= 0:
                raise ValueError(
                    "elfmap_teacher_is_cdeq requires cdeq_num_phase_tokens > 0 "
                    "(must match the Stage-2 FPFlow teacher checkpoint).")
            if cfg.self_cond_prob <= 0:
                raise ValueError(
                    "elfmap_teacher_is_cdeq feeds the CDEQ teacher's SC slot zeros; "
                    "requires self_cond_prob > 0 (the SC input slot must exist).")
        if cfg.elfmap_sc_refine_steps < 1:
            raise ValueError(
                "FPFlowMap requires cfg.elfmap_sc_refine_steps >= 1. "
                "This field is the paper's cold-start fixed-point iteration "
                "count N (# FPIs): N=1 is the bare cold-start prediction "
                "(one teacher forward); the paper has no N=0.")
        if cfg.elfmap_student_sc_free:
            # Regime is encoded in w (a self-cond-CFG token) and the teacher needs
            # the SC input slot.
            if cfg.num_self_cond_cfg_tokens <= 0:
                raise ValueError(
                    "elfmap_student_sc_free encodes the no-SC/SC regime in the "
                    "self-cond-CFG scale w; requires num_self_cond_cfg_tokens > 0.")
            if cfg.self_cond_prob <= 0:
                raise ValueError(
                    "elfmap_student_sc_free needs the SC input slot "
                    "(self_cond_prob > 0) for the teacher; the student feeds zeros.")
            if cfg.elfmap_scfree_offdiag_mode not in ("free", "chained", "rerun"):
                raise ValueError(
                    "elfmap_scfree_offdiag_mode must be free|chained|rerun, got "
                    f"{cfg.elfmap_scfree_offdiag_mode!r}.")
        encoder_dim = self.encoder.d_model
        # Teacher arch. Default: single-time vanilla ELF (double_time_emb=False,
        # no phase tokens). CDEQ-teacher mode (Stage 3): the teacher is a Stage-2
        # FPFlow ckpt, so it must be built two-time + with the phase-token bank to
        # match the checkpoint under strict load (the student stays phase-free).
        teacher_is_cdeq = getattr(cfg, "elfmap_teacher_is_cdeq", False)
        self.teacher_model = ELF_models[cfg.model](
            text_encoder_dim=encoder_dim, max_length=cfg.max_length,
            attn_drop=0.0, proj_drop=0.0,
            num_time_tokens=cfg.num_time_tokens,
            num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
            vocab_size=self.vocab_size,
            num_model_mode_tokens=cfg.num_model_mode_tokens,
            bottleneck_dim=cfg.bottleneck_dim,
            self_cond_input=(cfg.self_cond_prob > 0),
            use_flash=cfg.use_flash,
            double_time_emb=teacher_is_cdeq,
            num_phase_tokens=(cfg.cdeq_num_phase_tokens if teacher_is_cdeq else 0),
        ).to(self.device)

        ckpt = torch.load(cfg.teacher_ckpt_path, map_location=self.device,
                          weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        teacher_state = {k[len("model."):]: v
                         for k, v in state.items() if k.startswith("model.")}
        if not teacher_state:
            # Direct model state_dict (not a Lightning checkpoint).
            teacher_state = state
        self.teacher_model.load_state_dict(teacher_state, strict=True)

        # Prefer EMA weights for the teacher (matches eval_lightning.py:124-127).
        if "ema" in ckpt:
            t_ema = EMA(self.teacher_model, decay=cfg.ema_decay1)
            t_ema.load_state_dict(ckpt["ema"], device=self.device)
            t_ema.swap_in(self.teacher_model)

        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        self.teacher_model.eval()

        # Student warm-start from the teacher's EMA weights. teacher_model now
        # holds the EMA weights swapped in above — the
        # same weights used for the diagonal distillation target — so the
        # student starts consistent with the target it is trained against.
        # The student and teacher have identical parameter sets — the two-time
        # conditioning adds no new parameters (build_context partitions the
        # existing t_emb_tokens bank by role), so this is a complete 1:1 copy.
        teacher_ema_state = self.teacher_model.state_dict()
        student_sd = self.model.state_dict()
        skipped = []
        for k, v in teacher_ema_state.items():
            if k in student_sd and student_sd[k].shape == v.shape:
                student_sd[k] = v
            else:
                tgt_shape = tuple(student_sd[k].shape) if k in student_sd else None
                skipped.append((k, tuple(v.shape), tgt_shape))
        self.model.load_state_dict(student_sd, strict=False)
        if skipped:
            log_for_0(f"[ELFMap] warm-start: {len(skipped)} teacher key(s) skipped: {skipped}")
        else:
            log_for_0(f"[ELFMap] warm-start: copied {len(teacher_ema_state)} teacher tensor(s) into student.")

        # Re-seed the EMA shadow from the warm-started weights. super().setup()
        # built self._ema from the random-init model (before this warm-start),
        # so without this the shadow lags the entire teacher->student jump and
        # corrupts every EMA-based eval until the 0.9999 decay washes it out.
        # On resume, on_load_checkpoint overwrites self._ema afterwards — same
        # contract as the base class's setup-time EMA construction.
        self._ema = EMA(self.model, decay=cfg.ema_decay1)

    # ---------- helpers ------------------------------------------------
    def _tau_to_t(self, tau: torch.Tensor) -> torch.Tensor:
        kind = self.cfg.elfmap_tau_kind
        if kind == "identity":
            return tau
        if kind == "logit_normal":
            eps = 1e-7
            tc = tau.clamp(eps, 1 - eps)
            logit = torch.log(tc / (1 - tc))
            return torch.sigmoid(self.cfg.denoiser_p_std * logit
                                  + self.cfg.denoiser_p_mean)
        if kind == "logit_normal_uniform":
            # Exact logit-normal quantile transform: the standard-normal inverse
            # CDF (probit) in place of logit, so a uniform percentile tau maps to
            # the teacher's exact logit-normal time marginal. Mirrors the
            # "logit_normal_uniform" branch of get_sampling_steps.
            eps = 1e-7
            tc = tau.clamp(eps, 1 - eps)
            z = torch.special.ndtri(tc) * self.cfg.denoiser_p_std + self.cfg.denoiser_p_mean
            return torch.sigmoid(z)
        if kind == "vocab_pe":
            # FMLM Pe-based tau; weak motivation in embedding space — kept as a stub.
            return tau
        raise ValueError(f"Unknown elfmap_tau_kind: {kind}")

    def _sample_two_time(self, B: int, gen, device):
        cfg = self.cfg
        is_diag = torch.rand(B, generator=gen, device=device) < cfg.diagonal_fraction
        tau_d = torch.rand(B, generator=gen, device=device)
        tau_s = torch.rand(B, generator=gen, device=device) * (1 - tau_d)
        tau_t = tau_s + tau_d
        tau_u = 0.5 * (tau_s + tau_t)
        is_off = ~is_diag
        # Boundary injection — only among off-diag rows.
        is_bndry = is_off & (torch.rand(B, generator=gen, device=device)
                              < (1.0 / max(1, cfg.boundary_prob)))
        tau_s = torch.where(is_bndry, torch.zeros_like(tau_s), tau_s)
        tau_t = torch.where(is_bndry, torch.ones_like(tau_t),  tau_t)
        tau_u = torch.where(is_bndry, 0.5 * torch.ones_like(tau_u), tau_u)
        # Diagonal rows: s = t.
        tau_s = torch.where(is_diag, tau_t, tau_s)
        tau_u = torch.where(is_diag, tau_t, tau_u)
        # τ → t reparam (logit-normal by default).
        s = self._tau_to_t(tau_s)
        u = self._tau_to_t(tau_u)
        t = self._tau_to_t(tau_t)
        # Pin boundary endpoints exactly post-reparam (avoid eps drift).
        s = torch.where(is_bndry, torch.zeros_like(s), s)
        t = torch.where(is_bndry, torch.ones_like(t), t)
        kind = torch.where(
            is_diag, torch.zeros(B, dtype=torch.long, device=device),
            torch.where(is_bndry,
                        torch.full((B,), 2, dtype=torch.long, device=device),
                        torch.ones(B, dtype=torch.long, device=device)),
        )
        return s, u, t, kind

    def _sample_sc_cfg_w(self, B: int, gen, device, dtype):
        """SC-CFG scale w per row.
        Reuses the existing log-uniform `sample_cfg_scale` so the student's SC-CFG
        conditioning matches vanilla ELF's training distribution exactly."""
        return sample_cfg_scale(
            gen, B, device=device,
            cfg_min=self.cfg.self_cond_cfg_min, cfg_max=self.cfg.self_cond_cfg_max,
        ).to(dtype)

    def _refine_sc(self, net, z, t, s, w, n_steps, cond_seq, cond_mask,
                   *, return_trajectory=False, method=None,
                   aa_m=None, aa_beta=None, aa_lam=None):
        """Iterated self-conditioning refinement:
        hold the latent `z` (and times `t`, `s`) fixed and refine the SC slot for
        the fixed-point map F(sc) = restore_cond(net(z, t, sc, w)) from a zero
        slot. Returns (sc_N, sc_1): the depth-`n_steps` result and the depth-1
        (zero-SC) result, both restore_cond-clamped, NOT detached. Draws zero RNG.
        The caller owns the no_grad context (or relies on a frozen net +
        .detach()). `s=None` omits the s= kwarg (single-time teacher). `n_steps`>=1.

        cfg.elfmap_sc_refine_method selects how the fixed point is iterated:
          "picard"   — sc_{k+1} = F(sc_k)              (n_steps forwards)
          "anderson" — Anderson acceleration of F, matched budget so the total
                       number of teacher forwards is still n_steps (bootstrap
                       F(0) + (n_steps-1) accelerated updates). sc_1 is the same
                       bootstrap F(0) in both methods. n_steps=1 is bit-identical.

        CDEQ extension: `method`/`aa_m`/`aa_beta`/`aa_lam` override
        the cfg.elfmap_* knobs (None = fall back to cfg). When `return_trajectory`
        is True, returns (traj, sc_star) instead, where `traj` is the FULL
        un-popped list of SC iterate states [sc_0=0, sc_1, ..., sc_K] (length
        n_steps+1) and sc_star = traj[-1] is the converged endpoint. The default
        (return_trajectory=False) path returns the (sc_star, sc_1) 2-tuple."""
        sc_scale = w if self.cfg.num_self_cond_cfg_tokens > 0 else None
        has_slot = self.cfg.self_cond_prob > 0
        kw = {} if s is None else {"s": s}

        def F(sc):
            z_in = torch.cat([z, sc], dim=-1) if has_slot else z
            pred, _ = net(z_in, t, self_cond_cfg_scale=sc_scale,
                          decoder_step_active=False, **kw)
            return restore_cond(pred, cond_seq, cond_mask)

        method = method if method is not None else getattr(self.cfg, "elfmap_sc_refine_method", "picard")
        if method == "picard":
            traj = [torch.zeros_like(z)] if return_trajectory else None
            sc_k, sc_1 = torch.zeros_like(z), None
            for k in range(n_steps):
                sc_k = F(sc_k)
                if k == 0:
                    sc_1 = sc_k
                if return_trajectory:
                    traj.append(sc_k)
            # N=0 ablation: no refinement forward at all -> teacher reads a pure-zero
            # SC slot (sc_k stays zeros) and the depth-1 no-SC target sc_1 has no pass
            # to take, so it falls back to the same zeros (never selected when
            # use_sc is all-True, i.e. self_cond_prob>=1 / scp1.0 runs).
            if sc_1 is None:
                sc_1 = sc_k
            if return_trajectory:
                return traj, traj[-1]
            return sc_k, sc_1

        if method == "anderson":
            m = int(aa_m if aa_m is not None else getattr(self.cfg, "elfmap_anderson_m", 3))
            beta = float(aa_beta if aa_beta is not None else getattr(self.cfg, "elfmap_anderson_beta", 1.0))
            lam = float(aa_lam if aa_lam is not None else getattr(self.cfg, "elfmap_anderson_lam", 1e-4))
            B = z.size(0)
            sc = torch.zeros_like(z)
            fx = F(sc)                       # forward 1 (== Picard depth-1)
            sc_1 = fx
            # Un-popped trajectory accumulator: fx_hist below is
            # capped at m, so the full K-point trajectory must be collected here.
            traj = [torch.zeros_like(z), fx] if return_trajectory else None
            sc_hist, fx_hist, g_hist = [sc], [fx], [fx - sc]
            for _ in range(n_steps - 1):     # matched budget: total forwards == n_steps
                m_k = min(m, len(g_hist))
                if m_k < 2:
                    sc_new = beta * fx_hist[-1] + (1.0 - beta) * sc_hist[-1]
                else:
                    # Constrained LS for the mixing weights, done in fp32 so the
                    # GGᵀ reduction is stable under bf16-mixed autocast.
                    G = torch.stack([g_hist[-m_k + i].reshape(B, -1).float()
                                     for i in range(m_k)], dim=1)        # (B, m_k, n)
                    GGT = torch.bmm(G, G.transpose(1, 2))                # (B, m_k, m_k)
                    reg = lam * torch.eye(m_k, device=G.device, dtype=G.dtype).unsqueeze(0)
                    ones_vec = torch.ones(B, m_k, 1, device=G.device, dtype=G.dtype)
                    try:
                        inv_ones = torch.linalg.solve(GGT + reg, ones_vec)
                    except RuntimeError:
                        inv_ones = ones_vec / float(m_k)
                    denom = inv_ones.sum(dim=1, keepdim=True).clamp(min=1e-12)
                    alpha = (inv_ones / denom).to(sc.dtype)              # (B, m_k, 1)
                    Fm = torch.stack([fx_hist[-m_k + i].reshape(B, -1)
                                      for i in range(m_k)], dim=1)
                    X = torch.stack([sc_hist[-m_k + i].reshape(B, -1)
                                     for i in range(m_k)], dim=1)
                    mix = beta * Fm + (1.0 - beta) * X
                    sc_new = (alpha * mix).sum(dim=1).view_as(sc)
                fx = F(sc_new)
                sc = sc_new
                sc_hist.append(sc); fx_hist.append(fx); g_hist.append(fx - sc)
                if return_trajectory:
                    traj.append(fx)
                if len(sc_hist) > m:
                    sc_hist.pop(0); fx_hist.pop(0); g_hist.pop(0)
            if return_trajectory:
                return traj, traj[-1]
            return fx, sc_1

        raise ValueError(
            f"Unknown elfmap_sc_refine_method: {method!r} (picard|anderson)")

    def _teacher_x_pred(self, z, t, w, cond_seq_mask_expanded, cond_seq,
                        use_sc=None):
        """External teacher's single-time x-prediction at (z, t).
        Builds the SC by refining a zero slot for (N-1) iterations, then a with-SC
        main pass produces `out` -- so the readout sees N total fixed-point
        iterations, where N == cfg.elfmap_sc_refine_steps is the paper's # FPIs
        (>=1). N=1 is the bare cold-start (one forward). `use_sc`
        (per-row bool) picks the with-SC target (`out`) for SC rows and the no-SC
        target (`teacher_sc`, the depth-1 zero-SC pass) for the rest; None keeps
        the with-SC target for every row. Output is clamped to cond_seq on cond
        positions. Grad isolation: teacher params are frozen + the return is
        .detach()'d.

        ELF* teacher mode (cfg.elfmap_teacher_is_cdeq, Stage 3): the teacher is a
        Stage-2 FPFlow student that already collapses the SC fixed point in ONE
        forward. Reproduce its inference contract for the diagonal
        target — zero SC slot, phase=cdeq_inference_phase, s=t — instead of
        _refine_sc/Anderson. No use_sc dual-target: the no-SC/SC regime is carried
        by w exactly as for the SC-free student. (z, t are the diagonal z_s, s.)"""
        cfg = self.cfg
        if getattr(cfg, "elfmap_teacher_is_cdeq", False):
            sc_scale = w if cfg.num_self_cond_cfg_tokens > 0 else None
            phase_b = z.new_full((z.shape[0],), float(cfg.cdeq_inference_phase))
            z_in = (torch.cat([z, torch.zeros_like(z)], dim=-1)
                    if cfg.self_cond_prob > 0 else z)
            out, _ = self.teacher_model(
                z_in, t, self_cond_cfg_scale=sc_scale,
                decoder_step_active=False, s=t, phase=phase_b,
            )
            out = restore_cond(out, cond_seq, cond_seq_mask_expanded)
            return out.detach()
        sc_scale = w if self.cfg.num_self_cond_cfg_tokens > 0 else None
        # elfmap_sc_refine_steps is the paper's # FPIs N (>=1). The fixed point is
        # read out by the with-SC forward below, so N FPIs == (N-1) slot refinements
        # here + that 1 readout forward. N=1 -> 0 refinements -> bare cold-start.
        sc_N, teacher_sc = self._refine_sc(
            self.teacher_model, z, t, None, w,
            self.cfg.elfmap_sc_refine_steps - 1, cond_seq, cond_seq_mask_expanded,
        )
        z_in = (torch.cat([z, sc_N], dim=-1)
                if self.cfg.self_cond_prob > 0 else z)
        out, _ = self.teacher_model(
            z_in, t, self_cond_cfg_scale=sc_scale, decoder_step_active=False,
        )
        out = restore_cond(out, cond_seq, cond_seq_mask_expanded)
        if use_sc is None:
            return out.detach()
        return torch.where(use_sc.view(-1, 1, 1), out, teacher_sc).detach()

    @staticmethod
    def _masked_mse(pred, target, loss_mask):
        """Per-token MSE (mean over D) with sequence mask. pred/target: (B, S, D),
        loss_mask: (B, S). Returns scalar."""
        per_token = ((pred - target) ** 2).mean(dim=-1)  # (B, S)
        safe = torch.where(loss_mask > 0, per_token, torch.zeros_like(per_token))
        return (safe * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    # ---------- training step ------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        cfg = self.cfg
        gen = self._train_generator
        device = batch["input_ids"].device

        batch["input_ids"] = batch["input_ids"].clamp_max(self.vocab_size - 1)
        is_opt_step = (batch_idx + 1) % cfg.grad_accum_steps == 0
        lr = self._lr_at_step(self._my_opt_step)
        for opt in self.optimizers():
            for g in opt.param_groups:
                g["lr"] = lr

        # Identical prelude to ELFLitModule (label drop preserved verbatim).
        encoder_attention_mask = batch["encoder_attention_mask"]
        if cfg.label_drop_prob > 0:
            B0 = encoder_attention_mask.size(0)
            drop = (torch.rand(B0, generator=gen, device=device)
                    < cfg.label_drop_prob).float().view(B0, 1, 1)
            cm = batch["cond_seq_mask"]
            block_mask = (1 - cm).unsqueeze(2) * cm.unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

        with torch.no_grad():
            x0 = encode_text(batch["input_ids"], encoder_attention_mask,
                             encoder=self.encoder, latent_mean=cfg.latent_mean,
                             latent_std=cfg.latent_std)
        B, S, D = x0.shape
        cond_seq_mask = batch["cond_seq_mask"].unsqueeze(-1)  # (B, S, 1)
        loss_mask = (batch["attention_mask"] if cfg.pad_token == "pad"
                     else torch.ones_like(batch["attention_mask"]))
        loss_mask = loss_mask * (1 - batch["cond_seq_mask"])  # (B, S)

        decoder_step_active = bool(
            (torch.rand((), generator=gen, device=device) < cfg.decoder_prob).item()
        )

        if decoder_step_active:
            # === Decode CE branch (matches ELFLitModule.training_step) ===
            ones = torch.ones(B, device=device, dtype=x0.dtype)
            decoder_z_vals = (torch.randn(B * S, generator=gen, device=device)
                              * cfg.decoder_p_std + cfg.decoder_p_mean)
            decoder_lambda_t = torch.sigmoid(decoder_z_vals).view(B, S, 1).to(x0.dtype)
            decoder_noise = (torch.randn(x0.shape, generator=gen, device=device,
                                          dtype=x0.dtype) * cfg.decoder_noise_scale)
            decoder_z = decoder_lambda_t * x0 + (1.0 - decoder_lambda_t) * decoder_noise

            sc_w = (self._sample_sc_cfg_w(B, gen, device, x0.dtype)
                    if cfg.num_self_cond_cfg_tokens > 0 else None)
            decoder_input = (torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
                             if cfg.self_cond_prob > 0 else decoder_z)
            _, decoder_logits = self.model(
                decoder_input, ones, self_cond_cfg_scale=sc_w,
                decoder_step_active=True, s=ones,
            )
            log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
            ce = -log_probs.gather(-1, batch["input_ids"].unsqueeze(-1)).squeeze(-1)
            loss_ce = (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = loss_ce
            loss_diag = torch.zeros((), device=device)
            loss_off = torch.zeros((), device=device)
            n_diag = 0
            n_off = 0
            frac_sc = 0.0
        else:
            # === Flow-map MSE branch ===
            s, u, t, kind = self._sample_two_time(B, gen, device)
            s = s.to(x0.dtype); u = u.to(x0.dtype); t = t.to(x0.dtype)
            # Per-row self-conditioning mask — Bernoulli(self_cond_prob), the
            # flow-map analogue of vanilla ELF's use_self_cond_mask. No-SC rows
            # feed a zero SC slot into every model call and take the teacher's
            # zero-SC (single-pass) diagonal target. self_cond_prob>=1 draws no
            # RNG and is bit-identical to the always-on path; ==0 -> no SC slot
            # anywhere (use_sc all-False, never consumed).
            if cfg.self_cond_prob >= 1.0:
                use_sc = torch.ones(B, dtype=torch.bool, device=device)
            else:
                use_sc = (torch.rand(B, generator=gen, device=device)
                          < cfg.self_cond_prob)
            sc_w = (self._sample_sc_cfg_w(B, gen, device, x0.dtype)
                    if cfg.num_self_cond_cfg_tokens > 0 else None)
            if cfg.elfmap_student_sc_free and sc_w is not None:
                # Encode the no-SC / SC regime in w: no-SC rows
                # -> w=0 (scale 1+w=1.0, no guidance), SC rows keep the sampled w.
                # The SC-free student reads the regime from w (its SC slot is
                # always zeros). No RNG drawn (torch.where is deterministic).
                sc_w = torch.where(use_sc, sc_w, torch.zeros_like(sc_w))

            noise = torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)
            z_s = add_noise(x0, noise, s, cfg, cond_seq_mask=cond_seq_mask)
            ZEROS = torch.zeros_like(z_s)

            # 1) Student SC pre-pass: sc_signal = student(z_s, s=s, t=s, sc=zeros, w).
            #    Skipped in the SC-free regime (the student never consumes SC).
            sc_signal = None
            if not cfg.elfmap_student_sc_free:
                with torch.no_grad():
                    z_for_sc = (torch.cat([z_s, ZEROS], dim=-1)
                                if cfg.self_cond_prob > 0 else z_s)
                    sc_pred, _ = self.model(
                        z_for_sc, s, self_cond_cfg_scale=sc_w,
                        decoder_step_active=False, s=s,
                    )
                    sc_signal = restore_cond(sc_pred, x0, cond_seq_mask).detach()

            # 2) Off-diagonal PSD targets — live student under no_grad.
            idx_off  = (kind > 0).nonzero(as_tuple=True)[0]
            idx_diag = (kind == 0).nonzero(as_tuple=True)[0]
            off_target = None
            if idx_off.numel() > 0:
                with torch.no_grad(): # od: off_diagonal
                    z_od = z_s[idx_off]
                    s_od, u_od, t_od = s[idx_off], u[idx_off], t[idx_off]
                    w_od = sc_w[idx_off] if sc_w is not None else None
                    cm_od = cond_seq_mask[idx_off]
                    x0_od = x0[idx_off]
                    eps = cfg.t_eps
                    if cfg.elfmap_student_sc_free:
                        # SC-free regime: the GRADED student is
                        # SC-free; the off-diag *target* is built from the live
                        # student (the single-time teacher cannot do s!=t). The
                        # first half-step A is always SC-free; B's SC slot at the
                        # midpoint is chosen by elfmap_scfree_offdiag_mode.
                        in_A = (torch.cat([z_od, torch.zeros_like(z_od)], dim=-1)
                                if cfg.self_cond_prob > 0 else z_od)
                        A_pred, _ = self.model(
                            in_A, u_od, self_cond_cfg_scale=w_od,
                            decoder_step_active=False, s=s_od,
                        )
                        A_pred = restore_cond(A_pred, x0_od, cm_od)
                        # Midpoint state (FMLM Eq 19): X_{s,u} along the Euler step.
                        one_minus_s = (1 - s_od).clamp(min=eps).view(-1, 1, 1)
                        X_su = ((1 - u_od).view(-1, 1, 1) / one_minus_s) * z_od \
                             + ((u_od - s_od).view(-1, 1, 1) / one_minus_s) * A_pred
                        X_su = restore_cond(X_su, x0_od, cm_od)
                        # B's self-condition slot:
                        #   free -> zeros; chained -> A_pred; rerun -> N-step
                        #   refinement at the midpoint X_su.
                        od_mode = cfg.elfmap_scfree_offdiag_mode
                        if od_mode == "free":
                            B_sc = torch.zeros_like(X_su)
                        elif od_mode == "chained":
                            B_sc = A_pred
                        elif od_mode == "rerun":
                            # N-1 refinements + the with-SC readout in in_B == N FPIs
                            # (elfmap_sc_refine_steps is the paper's # FPIs N >= 1).
                            B_sc, _ = self._refine_sc(
                                self.model, X_su, u_od, u_od, w_od,
                                cfg.elfmap_sc_refine_steps - 1, x0_od, cm_od,
                            )
                        else:
                            raise ValueError(
                                f"Unknown elfmap_scfree_offdiag_mode: {od_mode}")
                        in_B = (torch.cat([X_su, B_sc], dim=-1)
                                if cfg.self_cond_prob > 0 else X_su)
                        B_pred, _ = self.model(
                            in_B, t_od, self_cond_cfg_scale=w_od,
                            decoder_step_active=False, s=u_od,
                        )
                        B_pred = restore_cond(B_pred, x0_od, cm_od)
                        # Convex combo weight γ (FMLM Eq 22).
                        denom = ((1 - u_od) * (t_od - s_od)).clamp(min=eps).view(-1, 1, 1)
                        gamma = (((1 - t_od) * (u_od - s_od)).view(-1, 1, 1) / denom)
                        off_target = (gamma * A_pred + (1 - gamma) * B_pred).detach()
                    else:
                        sc_od = sc_signal[idx_off]
                        # Case B: no-SC off-diag rows feed a zero SC slot into both
                        # target-building passes (A and B). use_sc_od masks both.
                        use_sc_od = use_sc[idx_off].view(-1, 1, 1).to(z_od.dtype)
                        # A = net(z_s, s=s, t=u, sc=sc_signal, w)
                        in_A = (torch.cat([z_od, sc_od * use_sc_od], dim=-1)
                                if cfg.self_cond_prob > 0 else z_od)
                        A_pred, _ = self.model(
                            in_A, u_od, self_cond_cfg_scale=w_od,
                            decoder_step_active=False, s=s_od,
                        )
                        A_pred = restore_cond(A_pred, x0_od, cm_od)
                        # Midpoint state (FMLM Eq 19): X_{s,u} along the Euler step.
                        one_minus_s = (1 - s_od).clamp(min=eps).view(-1, 1, 1)
                        X_su = ((1 - u_od).view(-1, 1, 1) / one_minus_s) * z_od \
                             + ((u_od - s_od).view(-1, 1, 1) / one_minus_s) * A_pred
                        X_su = restore_cond(X_su, x0_od, cm_od)

                        # SC source for B_pred.
                        b_mode = cfg.elfmap_offdiag_b_sc_mode
                        if b_mode == "anchored":
                            B_sc = sc_od                              # diagonal pre-pass at z_s
                        elif b_mode == "chained":
                            B_sc = A_pred                              # A_pred already detached + restored
                        elif b_mode == "rerun":
                            # c-u: fresh diagonal pre-pass at X_{s,u} with time u
                            z_B0 = (torch.cat([X_su, torch.zeros_like(X_su)], dim=-1)
                                    if cfg.self_cond_prob > 0 else X_su)
                            sc_at_B, _ = self.model(
                                z_B0, u_od, self_cond_cfg_scale=w_od,
                                decoder_step_active=False, s=u_od,
                            )
                            B_sc = restore_cond(sc_at_B, x0_od, cm_od)
                        else:
                            raise ValueError(
                                f"Unknown elfmap_offdiag_b_sc_mode: {b_mode}"
                            )

                        # B = net(X_{s,u}, s=u, t=t, sc=B_sc, w). No-SC rows get a
                        # zero SC slot regardless of elfmap_offdiag_b_sc_mode.
                        in_B = (torch.cat([X_su, B_sc * use_sc_od], dim=-1)
                                if cfg.self_cond_prob > 0 else X_su)
                        B_pred, _ = self.model(
                            in_B, t_od, self_cond_cfg_scale=w_od,
                            decoder_step_active=False, s=u_od,
                        )
                        B_pred = restore_cond(B_pred, x0_od, cm_od)
                        # Convex combo weight γ (FMLM Eq 22).
                        denom = ((1 - u_od) * (t_od - s_od)).clamp(min=eps).view(-1, 1, 1)
                        gamma = (((1 - t_od) * (u_od - s_od)).view(-1, 1, 1) / denom)
                        off_target = (gamma * A_pred + (1 - gamma) * B_pred).detach()
                    if cfg.rescale_offdiag_loss_psd:
                        # FMLM flm/algo.py:1539-1541 — scale by 1/(t-s) so all step
                        # widths contribute comparably (off by default).
                        # Note: applied to loss in step 4 below.
                        pass

            # 3) Diagonal target from external teacher. Case A: SC
            #    rows get the 2-pass with-SC target, no-SC rows the zero-SC pass.
            diag_target = None
            if idx_diag.numel() > 0:
                w_dd = sc_w[idx_diag] if sc_w is not None else None
                diag_target = self._teacher_x_pred(
                    z_s[idx_diag], s[idx_diag], w_dd,
                    cond_seq_mask[idx_diag], x0[idx_diag],
                    use_sc=use_sc[idx_diag],
                )

            # 4) Student grad pass: a single (s, t) forward.
            if cfg.elfmap_student_sc_free:
                # SC-free student: zero SC slot + w. The no-SC/SC
                # regime is carried by sc_w; no sc_signal, no off-diag SC mode.
                z_for_stu = (torch.cat([z_s, ZEROS], dim=-1)
                             if cfg.self_cond_prob > 0 else z_s)
            else:
                # SC source for off-diag rows is configurable.
                stu_mode = cfg.elfmap_student_offdiag_sc_mode
                if stu_mode == "sc_signal":
                    student_sc_input = sc_signal
                elif stu_mode == "zeros":
                    # Diag rows keep sc_signal; off-diag rows get zero SC.
                    if idx_off.numel() > 0:
                        student_sc_input = sc_signal.clone()
                        student_sc_input[idx_off] = 0
                    else:
                        student_sc_input = sc_signal
                else:
                    raise ValueError(
                        f"Unknown elfmap_student_offdiag_sc_mode: {stu_mode}"
                    )
                # Cases A & B: no-SC rows feed a zero SC slot to the student too.
                student_sc_input = (student_sc_input
                                    * use_sc.view(-1, 1, 1).to(z_s.dtype))
                z_for_stu = (torch.cat([z_s, student_sc_input], dim=-1)
                             if cfg.self_cond_prob > 0 else z_s)
            student_pred, _ = self.model(
                z_for_stu, t, self_cond_cfg_scale=sc_w,
                decoder_step_active=False, s=s,
            )

            # 5) Per-branch masked MSE.
            n_diag = int(idx_diag.numel())
            n_off = int(idx_off.numel())
            frac_sc = int(use_sc.sum()) / B
            if n_diag > 0:
                loss_diag = self._masked_mse(
                    student_pred[idx_diag], diag_target, loss_mask[idx_diag],
                )
            else:
                loss_diag = torch.zeros((), device=device)
            if n_off > 0:
                loss_off = self._masked_mse(
                    student_pred[idx_off], off_target, loss_mask[idx_off],
                )
                if cfg.rescale_offdiag_loss_psd:
                    # Scale by mean inverse width 1/(t-s) across off-diag rows.
                    width = (t[idx_off] - s[idx_off]).clamp(min=cfg.t_eps)
                    loss_off = loss_off * (1.0 / width).mean()
            else:
                loss_off = torch.zeros((), device=device)
            loss = cfg.diag_weight * loss_diag + cfg.off_weight * loss_off
            loss_ce = torch.zeros((), device=device)

        # Backward + book-keeping.
        self.manual_backward(loss / cfg.grad_accum_steps)
        self._loss_running["loss"].add(loss.detach(), active=True)
        self._loss_running["ce"].add(loss_ce.detach(), active=decoder_step_active)
        self._loss_running["diag"].add(
            loss_diag.detach(), active=(not decoder_step_active and n_diag > 0)
        )
        self._loss_running["off"].add(
            loss_off.detach(), active=(not decoder_step_active and n_off > 0)
        )
        self._branch_running["decoder"].add(1.0 if decoder_step_active else 0.0, active=True)
        # Mean per-row diag/off fraction over flow-map steps (tracks
        # diagonal_fraction). A per-step ">=1 diag row" indicator is degenerate:
        # every batch is split per-row, so it would be ~always 1.
        self._branch_running["fm_diag"].add(n_diag / B, active=(not decoder_step_active))
        self._branch_running["fm_off"].add(n_off / B, active=(not decoder_step_active))
        self._branch_running["fm_sc"].add(frac_sc, active=(not decoder_step_active))

        if is_opt_step:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            for opt in self.optimizers():
                opt.step()
                opt.zero_grad(set_to_none=True)
            self._ema.update(self.model)
            self._my_opt_step += 1
            if self._my_opt_step % cfg.log_freq == 0:
                self._log_running(lr)

        return loss.detach()

    # ---------- logging ------------------------------------------------
    def _log_running(self, lr: float):
        if self._loss_running["loss"].count == 0:
            return
        loss_v, _   = self._loss_running["loss"].flush()
        diag_v, _   = self._loss_running["diag"].flush()
        off_v,  _   = self._loss_running["off"].flush()
        ce_v,   _   = self._loss_running["ce"].flush()
        f_dec,  _   = self._branch_running["decoder"].flush()
        f_diag, _   = self._branch_running["fm_diag"].flush()
        f_off,  _   = self._branch_running["fm_off"].flush()
        f_sc,   _   = self._branch_running["fm_sc"].flush()

        self.log("train/loss", loss_v, sync_dist=True, prog_bar=True,
                  on_step=True, on_epoch=False)
        self.log("train/loss_diag", diag_v, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/loss_off", off_v, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/loss_ce", ce_v, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/frac_decoder", f_dec, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/frac_flowmap_diag", f_diag, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/frac_flowmap_off", f_off, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/frac_flowmap_sc", f_sc, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/lr", lr, sync_dist=False,
                  on_step=True, on_epoch=False)

    # ---------- checkpoint plumbing ------------------------------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        # Strip teacher_model.* from state_dict (regenerated in setup from
        # cfg.teacher_ckpt_path on resume).
        sd = checkpoint.get("state_dict")
        if sd is not None:
            for k in list(sd.keys()):
                if k.startswith("teacher_model."):
                    del sd[k]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        # Lightning calls self.load_state_dict(checkpoint["state_dict"], strict=True)
        # next. Re-inject the (in-memory) teacher_model.* keys so the strict load
        # passes. The teacher was loaded from cfg.teacher_ckpt_path in setup().
        sd = checkpoint.get("state_dict")
        if sd is None or self.teacher_model is None:
            return
        for k, v in self.teacher_model.state_dict().items():
            sd[f"teacher_model.{k}"] = v


# -----------------------------------------------------------------------------
# FPFlow — Stage-2 CDEQ consistency distillation of the SC refinement axis.
# Subclasses FPFlowMap to reuse the frozen-teacher
# loader + student warm-start + EMA reseed + checkpoint hooks + _refine_sc.
# -----------------------------------------------------------------------------
class FPFlow(FPFlowMap):
    """Distill the within-timestep self-conditioning fixed point.

    Per non-decoder step: build an online teacher SC-refinement trajectory
    [sc_0=0, .., sc_K=sc*] (Anderson m/beta, no_grad), then train a learned
    one-Anderson-step student g(sc_a, sc_b, tau) toward sc* (global) and toward
    its own earlier-phase, stop-grad output (local). Phase tau_k indexes the
    refinement axis and conditions the student via the dedicated phase tokens.
    Inference (sampling_method="cdeq") feeds sc=0 at the cold-start phase tau_0
    in ONE forward per denoising step (no cross-timestep SC carry).

    Subclasses FPFlowMap purely to reuse its teacher-load / EMA-swap / warm-start
    / checkpoint machinery. Despite being an EARLIER pipeline stage than the
    flow-map (Stage-2 vs Stage-3), it therefore sits BELOW FPFlowMap in the
    class tree -- see the FPFlowMap docstring note on the inverted hierarchy."""

    def __init__(self, config: Config, vocab_size: int):
        super().__init__(config, vocab_size)
        self._loss_running = {
            "loss":   _RunningMean(),
            "global": _RunningMean(),
            "local":  _RunningMean(),
            "ce":     _RunningMean(),
        }
        self._branch_running = {
            "decoder": _RunningMean(),
            "cdeq":    _RunningMean(),
            "cold":    _RunningMean(),
        }
        # Per-group grad-norm probe: split the student into the denoiser CORE vs
        # the phase pathway (phase_embedder/phase_emb_tokens). Measures whether
        # the core actually receives gradient under the c_skip/c_out gating, or
        # only the phase tokens move.
        self._grad_running = {
            "core":  _RunningMean(),
            "phase": _RunningMean(),
        }

    # ---------- lifecycle ---------------------------------------------
    def setup(self, stage=None):
        cfg = self.cfg
        if stage == "fit":
            if cfg.self_cond_prob <= 0:
                raise ValueError("FPFlow requires self_cond_prob>0 (the SC input slot).")
            if cfg.num_self_cond_cfg_tokens <= 0:
                raise ValueError("FPFlow requires num_self_cond_cfg_tokens>0 (w conditioning).")
            if cfg.cdeq_num_phase_tokens <= 0:
                raise ValueError("FPFlow requires cdeq_num_phase_tokens>0 (phase conditioning).")
        # FPFlowMap.setup loads the frozen teacher (num_phase_tokens=0),
        # warm-starts the student (phase params absent in teacher -> stay zero-
        # init), and re-seeds the EMA shadow (phase params included at zero-init).
        super().setup(stage=stage)

    # ---------- helpers ------------------------------------------------
    def _tau_of_k(self, k_idx):
        """Consistency-time map, paper Eq.6 verbatim: tau_k = eps + (1 - e^{-rho k})(T - eps).
        tau_0 = eps EXACTLY (cold start, c_skip=0, pure-net — the inference point).
        tau_K < T (the exponential never reaches T → soft equilibrium boundary, as
        in the paper; the equilibrium is anchored by L_global, not the c_skip
        boundary). (B,) float."""
        cfg = self.cfg
        eps, T, rho = cfg.cdeq_tau_eps, cfg.cdeq_tau_T, cfg.cdeq_rho
        return eps + (1.0 - torch.exp(-rho * k_idx.float())) * (T - eps)

    def _c_skip(self, tau):
        """c_skip(tau) = ((tau-eps)/(T-eps))^gamma, clamped to [0,1]. (B,) -> (B,1,1)."""
        cfg = self.cfg
        eps, T, gamma = cfg.cdeq_tau_eps, cfg.cdeq_tau_T, cfg.cdeq_gamma
        cs = ((tau - eps) / (T - eps)).clamp(0.0, 1.0) ** gamma
        return cs.view(-1, 1, 1)

    @staticmethod
    def _anderson_combine(sc_a, sc_b, f_a, f_b, beta):
        """Closed-form 2-point Anderson step (matches official anderson_step).
        Degenerate when sc_a==sc_b: dr=0 -> a1=0,a0=1 -> beta*f_b (=f at the state)."""
        r_a = f_a - sc_a
        r_b = f_b - sc_b
        dr = r_a - r_b
        dot = (r_b * dr).sum(dim=(-2, -1), keepdim=True)
        norm_sq = (dr * dr).sum(dim=(-2, -1), keepdim=True)
        a1 = -dot / (norm_sq + 1e-9)
        a0 = 1.0 - a1
        return beta * (a0 * f_b + a1 * f_a) + (1.0 - beta) * (a0 * sc_b + a1 * sc_a)

    def _h(self, z_t, sc, tau, t, w, cond_seq, cond_mask):
        """Phase-conditioned student forward (the learnable map). x-prediction,
        cond-clamped. s=t (diagonal denoising time); phase=tau (refinement axis)."""
        sc_scale = w if self.cfg.num_self_cond_cfg_tokens > 0 else None
        z_in = torch.cat([z_t, sc], dim=-1)
        pred, _ = self.model(z_in, t, self_cond_cfg_scale=sc_scale,
                             decoder_step_active=False, s=t, phase=tau)
        return restore_cond(pred, cond_seq, cond_mask)

    def _student_g(self, z_t, sc_a, sc_b, tau_a, tau_b, t, w, cond_seq, cond_mask):
        """g = c_skip(tau_a)*sc_a + c_out(tau_a)*P, P = one learned Anderson step
        over (sc_a, sc_b). Two student forwards."""
        f_a = self._h(z_t, sc_a, tau_a, t, w, cond_seq, cond_mask)
        f_b = self._h(z_t, sc_b, tau_b, t, w, cond_seq, cond_mask)
        P = self._anderson_combine(sc_a, sc_b, f_a, f_b, self.cfg.cdeq_student_beta)
        cs = self._c_skip(tau_a)
        g = cs * sc_a + (1.0 - cs) * P
        return restore_cond(g, cond_seq, cond_mask)

    def _masked_pseudo_huber(self, pred, target, loss_mask):
        """Per-token Pseudo-Huber sqrt(mean_D(diff^2)+c^2)-c with sequence mask.
        c = cfg.cdeq_huber_c or 0.00054*sqrt(D)."""
        c = self.cfg.cdeq_huber_c
        if c is None:
            c = 0.00054 * math.sqrt(pred.shape[-1])
        per_token = torch.sqrt(((pred - target) ** 2).mean(dim=-1) + c * c) - c
        safe = torch.where(loss_mask > 0, per_token, torch.zeros_like(per_token))
        return (safe * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    # ---------- training step ------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        cfg = self.cfg
        gen = self._train_generator
        device = batch["input_ids"].device

        batch["input_ids"] = batch["input_ids"].clamp_max(self.vocab_size - 1)
        is_opt_step = (batch_idx + 1) % cfg.grad_accum_steps == 0
        lr = self._lr_at_step(self._my_opt_step)
        for opt in self.optimizers():
            for g in opt.param_groups:
                g["lr"] = lr

        encoder_attention_mask = batch["encoder_attention_mask"]
        if cfg.label_drop_prob > 0:
            B0 = encoder_attention_mask.size(0)
            drop = (torch.rand(B0, generator=gen, device=device)
                    < cfg.label_drop_prob).float().view(B0, 1, 1)
            cm = batch["cond_seq_mask"]
            block_mask = (1 - cm).unsqueeze(2) * cm.unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

        with torch.no_grad():
            x0 = encode_text(batch["input_ids"], encoder_attention_mask,
                             encoder=self.encoder, latent_mean=cfg.latent_mean,
                             latent_std=cfg.latent_std)
        B, S, D = x0.shape
        cond_seq_mask = batch["cond_seq_mask"].unsqueeze(-1)  # (B, S, 1)
        loss_mask = (batch["attention_mask"] if cfg.pad_token == "pad"
                     else torch.ones_like(batch["attention_mask"]))
        loss_mask = loss_mask * (1 - batch["cond_seq_mask"])  # (B, S)

        decoder_step_active = bool(
            (torch.rand((), generator=gen, device=device) < cfg.decoder_prob).item()
        )

        loss_global = torch.zeros((), device=device)
        loss_local = torch.zeros((), device=device)
        loss_ce = torch.zeros((), device=device)
        cold_frac = 0.0

        if decoder_step_active:
            # === Decode CE branch ===
            ones = torch.ones(B, device=device, dtype=x0.dtype)
            decoder_z_vals = (torch.randn(B * S, generator=gen, device=device)
                              * cfg.decoder_p_std + cfg.decoder_p_mean)
            decoder_lambda_t = torch.sigmoid(decoder_z_vals).view(B, S, 1).to(x0.dtype)
            decoder_noise = (torch.randn(x0.shape, generator=gen, device=device,
                                          dtype=x0.dtype) * cfg.decoder_noise_scale)
            decoder_z = decoder_lambda_t * x0 + (1.0 - decoder_lambda_t) * decoder_noise
            sc_w = (self._sample_sc_cfg_w(B, gen, device, x0.dtype)
                    if cfg.num_self_cond_cfg_tokens > 0 else None)
            decoder_input = (torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
                             if cfg.self_cond_prob > 0 else decoder_z)
            _, decoder_logits = self.model(
                decoder_input, ones, self_cond_cfg_scale=sc_w,
                decoder_step_active=True, s=ones,
            )
            log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
            ce = -log_probs.gather(-1, batch["input_ids"].unsqueeze(-1)).squeeze(-1)
            loss_ce = (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = loss_ce
        else:
            # === CDEQ consistency-distillation branch ===
            t = sample_timesteps(gen, B, device=device,
                                 P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
                                 time_schedule=cfg.time_schedule).to(x0.dtype)
            noise = torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)
            z_t = add_noise(x0, noise, t, cfg, cond_seq_mask=cond_seq_mask)
            w = (self._sample_sc_cfg_w(B, gen, device, x0.dtype)
                 if cfg.num_self_cond_cfg_tokens > 0 else None)

            # --- online teacher SC-refinement trajectory (frozen, no grad) ---
            K = cfg.cdeq_K
            with torch.no_grad():
                traj, sc_star = self._refine_sc(
                    self.teacher_model, z_t, t, None, w, K, x0, cond_seq_mask,
                    return_trajectory=True, method="anderson",
                    aa_m=cfg.cdeq_anderson_m, aa_beta=cfg.cdeq_anderson_beta,
                    aa_lam=cfg.cdeq_anderson_lam,
                )
            traj = torch.stack([s.detach() for s in traj], dim=0)  # (K+1, B, S, D)
            sc_star = sc_star.detach()
            arangeB = torch.arange(B, device=device)

            def gather_k(idx):  # idx: (B,) long -> (B, S, D)
                return traj[idx.clamp(0, K), arangeB]

            # --- sample the phase index k per row (cold-start coverage) ---
            cold = torch.rand(B, generator=gen, device=device) < cfg.cdeq_p_coldstart
            # non-cold rows: k ~ U{2..K} (needs k-2>=0 for the local target pair)
            k_rand = torch.randint(2, K + 1, (B,), generator=gen, device=device)
            k_idx = torch.where(cold, torch.zeros_like(k_rand), k_rand)
            ka = k_idx
            kb = (k_idx - 1).clamp(min=0)
            kc = (k_idx - 2).clamp(min=0)
            sc_a, sc_b, sc_c = gather_k(ka), gather_k(kb), gather_k(kc)
            tau_a, tau_b, tau_c = self._tau_of_k(ka), self._tau_of_k(kb), self._tau_of_k(kc)

            # --- snap-to-equilibrium augmentation ---
            # Replace the iterate VALUES with sc* but KEEP the sampled phases tau
            # (matches the official code: it snaps the value at a mid-trajectory
            # position, NOT at the identity boundary). Overriding tau->tau_K would
            # make c_skip(tau_K)=1 trivially satisfy g=sc* with zero gradient,
            # neutralizing the augmentation.
            aug = ((torch.rand(B, generator=gen, device=device) < cfg.cdeq_p_aug)
                   & (k_idx >= cfg.cdeq_k_min) & (k_idx <= K - cfg.cdeq_k_tail))
            if aug.any():
                aug3 = aug.view(-1, 1, 1)
                sc_a = torch.where(aug3, sc_star, sc_a)
                sc_b = torch.where(aug3, sc_star, sc_b)
                sc_c = torch.where(aug3, sc_star, sc_c)

            # --- student: graded g_k, stop-grad earlier-phase local target ---
            g_k = self._student_g(z_t, sc_a, sc_b, tau_a, tau_b, t, w,
                                  x0, cond_seq_mask)            # grad ON
            with torch.no_grad():
                g_km1 = self._student_g(z_t, sc_b, sc_c, tau_b, tau_c, t, w,
                                        x0, cond_seq_mask)      # sg(phi) target

            loss_global = self._masked_mse(g_k, sc_star, loss_mask)
            if cfg.cdeq_loss_metric == "pseudo_huber":
                loss_local = self._masked_pseudo_huber(g_k, g_km1.detach(), loss_mask)
            else:
                loss_local = self._masked_mse(g_k, g_km1.detach(), loss_mask)
            loss = cfg.cdeq_lambda1 * loss_global + (1.0 - cfg.cdeq_lambda1) * loss_local
            cold_frac = float(cold.float().mean().item())

        self.manual_backward(loss / cfg.grad_accum_steps)
        self._loss_running["loss"].add(loss.detach(), active=True)
        self._loss_running["global"].add(loss_global.detach(), active=not decoder_step_active)
        self._loss_running["local"].add(loss_local.detach(), active=not decoder_step_active)
        self._loss_running["ce"].add(loss_ce.detach(), active=decoder_step_active)
        self._branch_running["decoder"].add(1.0, active=decoder_step_active)
        self._branch_running["cdeq"].add(1.0, active=not decoder_step_active)
        self._branch_running["cold"].add(cold_frac, active=not decoder_step_active)

        if is_opt_step:
            # Per-group grad norm BEFORE clipping (raw learning signal). Buckets
            # by name: phase pathway vs everything else (core denoiser).
            core_sq = torch.zeros((), device=device)
            phase_sq = torch.zeros((), device=device)
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                g2 = p.grad.detach().pow(2).sum()
                if "phase" in n:
                    phase_sq = phase_sq + g2
                else:
                    core_sq = core_sq + g2
            self._grad_running["core"].add(core_sq.sqrt().item(), active=True)
            self._grad_running["phase"].add(phase_sq.sqrt().item(), active=True)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            for opt in self.optimizers():
                opt.step()
                opt.zero_grad(set_to_none=True)
            self._ema.update(self.model)
            self._my_opt_step += 1
            if self._my_opt_step % cfg.log_freq == 0:
                self._log_running(lr)

        return loss.detach()

    # ---------- logging ------------------------------------------------
    def _log_running(self, lr: float):
        if self._loss_running["loss"].count == 0:
            return
        loss_v, _   = self._loss_running["loss"].flush()
        glob_v, _   = self._loss_running["global"].flush()
        loc_v,  _   = self._loss_running["local"].flush()
        ce_v,   _   = self._loss_running["ce"].flush()
        f_dec,  _   = self._branch_running["decoder"].flush()
        f_cdeq, _   = self._branch_running["cdeq"].flush()
        f_cold, _   = self._branch_running["cold"].flush()
        self.log("train/loss", loss_v, sync_dist=True, prog_bar=True,
                  on_step=True, on_epoch=False)
        self.log("train/loss_global", glob_v, sync_dist=True, on_step=True, on_epoch=False)
        self.log("train/loss_local", loc_v, sync_dist=True, on_step=True, on_epoch=False)
        self.log("train/loss_ce", ce_v, sync_dist=True, on_step=True, on_epoch=False)
        self.log("train/frac_decoder", f_dec, sync_dist=True, on_step=True, on_epoch=False)
        self.log("train/frac_cdeq", f_cdeq, sync_dist=True, on_step=True, on_epoch=False)
        self.log("train/frac_coldstart", f_cold, sync_dist=True, on_step=True, on_epoch=False)
        gn_core, _  = self._grad_running["core"].flush()
        gn_phase, _ = self._grad_running["phase"].flush()
        self.log("train/gradnorm_core", gn_core, sync_dist=True, on_step=True, on_epoch=False)
        self.log("train/gradnorm_phase", gn_phase, sync_dist=True, on_step=True, on_epoch=False)
        self.log("train/lr", lr, sync_dist=False, on_step=True, on_epoch=False)


# -----------------------------------------------------------------------------
# DataModule
# -----------------------------------------------------------------------------
class ELFDataModule(L.LightningDataModule):
    def __init__(self, config: Config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        self._train_dataset = None
        self._eval_dataset = None

    def setup(self, stage=None):
        if self._train_dataset is None:
            self._train_dataset, self._eval_dataset = load_dataset(self.cfg)

    def train_dataloader(self):
        # Lightning attaches the DistributedSampler when strategy=ddp.
        return make_dataloader(
            self._train_dataset,
            batch_size=self.cfg.global_batch_size // self.trainer.world_size,
            shuffle=True,
            max_seq_length=self.cfg.max_length,
            pad_token_id=get_pad_token_id(self.tokenizer, self.cfg.pad_token),
            max_input_seq_length=self.cfg.max_input_length,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            drop_last=True,
        )

"""Config class — all default hyperparameters plus the optimization knobs
(FlashAttention, precision, DataLoader tuning)."""

import os
from typing import List, Optional, Union

import yaml


class SamplingConfig:
    sampling_method: str = "ode"
    num_sampling_steps: list = [50]
    cfgs: list = [1]
    self_cond_cfg_scales: list = [1.0]
    time_schedule: str = "logit_normal"
    sde_gamma: float = 0.0   # churn for sde/cdeq; gamma for the flow-map ctm sampler

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        fields = {k: getattr(self, k, None) for k in self.__class__.__annotations__}
        fields.update({k: v for k, v in vars(self).items() if not k.startswith("_")})
        return f"SamplingConfig({', '.join(f'{k}={v!r}' for k,v in fields.items())})"


class Config:
    # Dataset
    data_path: str = None
    eval_data_path: str = None
    max_length: int = 128
    max_input_length: int = None
    pad_token: str = "pad"
    tokenizer_name: str = None

    # Encoder
    encoder_model_name: str = "gpt2-large"
    encoder_checkpoint: str = None
    # "last" (final hidden state, with ln_f for GPT-2) or an int index into
    # `hidden_states[i]`. Only GPT2Encoder consumes this today.
    feature_layer: Union[str, int] = "last"
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Model architecture
    model: str = "ELF-B"
    bottleneck_dim: int = 128
    num_time_tokens: int = 4
    num_self_cond_cfg_tokens: int = 4
    num_model_mode_tokens: int = 4
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Denoiser objective
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"

    # Decoder objective
    decoder_prob: float = 0.5
    decoder_noise_scale: float = 1.0
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8

    # Conditioning / CFG
    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training
    epochs: int = 200
    warmup_epochs: float = None
    warmup_steps: int = 5000
    batch_size: int = None
    global_batch_size: int = 512
    lr: float = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1

    # EMA
    ema_decay1: float = 0.9999

    # Sampling
    sampling_configs_path: str = None
    sampling_configs: list = [SamplingConfig()]
    num_samples: int = 100

    # PPL eval
    online_eval: bool = True
    eval_ppl_model: str = "gpt2-large"
    eval_ppl_batch_size: int = 64
    eval_ppl_max_length: int = 1024

    # Logging / checkpointing
    log_freq: int = 100
    eval_freq: int = 10
    save_freq: float = 100
    # Optional step-based overrides — used by smoke runs that need to bound a
    # training launch by step count and produce a checkpoint without running a
    # full epoch. 0 / -1 keep the existing epoch-based behavior.
    max_train_steps: int = -1
    ckpt_every_n_train_steps: int = 0

    # Output
    output_dir: str = "./output_dir"
    resume: str = None

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "ELF"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tag: str = None

    # Misc
    seed: int = 0
    num_workers: int = 8           # 8 per rank by default.
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # Optimization knobs
    use_flash: bool = True         # FlashAttention path in ELF.
    precision: str = "fp32"        # Lightning precision: "fp32" | "bf16-mixed" | "16-mixed"

    # Per-epoch eval callback
    epoch_eval_num_samples: int = 256
    epoch_eval_num_sampling_steps: int = 32
    epoch_eval_sde_gamma: float = 1.5
    epoch_eval_self_cond_cfg_scale: float = 3.0

    # Disable when bos_token_id == eos_token_id (T5/flm_t5): otherwise the
    # first generated token (= BOS) is treated as a terminal EOS and the whole
    # row gets pad-masked, producing empty text and NaN gen_ppl.
    mask_after_eos: bool = True

    # Flow-map (FMLM-style) training — distill-only. Off by default.
    elfmap_enabled: bool = False
    elfmap_tau_kind: str = "logit_normal"   # "logit_normal" | "logit_normal_uniform" | "vocab_pe" | "identity"
    diagonal_fraction: float = 0.5
    boundary_prob: int = 32                  # 1/32 chance of (s,t)=(0,1) among off-diag rows
    diag_weight: float = 1.0
    off_weight: float = 1.0
    rescale_offdiag_loss_psd: bool = False
    teacher_ckpt_path: str = None            # REQUIRED when elfmap_enabled=True
    double_time_emb: bool = False            # ELFMap toggles this on; default off keeps parity
    # SC source for B_pred inside the off-diag PSD chain:
    #   "anchored" — B_sc = sc_signal (the diagonal pre-pass at z_s; current default)
    #   "chained"  — B_sc = A_pred.detach()       (use A's prediction as B's SC)
    #   "rerun"    — B_sc = student(X_{s,u}, s=u, t=u, sc=zeros, w)  (+1 student fwd per off-diag row)
    elfmap_offdiag_b_sc_mode: str = "anchored"
    # SC source for the student grad pass on off-diagonal rows:
    #   "sc_signal" — use sc_signal (anchored to z_s; current default)
    #   "zeros"     — feed zeros as SC for off-diag rows (diag rows still get sc_signal)
    elfmap_student_offdiag_sc_mode: str = "sc_signal"
    # Cold-start fixed-point iteration count N for the teacher diagonal target,
    # i.e. the paper's "# FPIs" (>=1). The fixed point
    # sc_0=zeros; sc_{k+1}=restore_cond(teacher(z, t, sc_k, w)) is read out by the
    # with-SC forward, so N FPIs == (N-1) refinements + 1 readout forward. N=1
    # (default) is the bare cold-start prediction (a single teacher forward).
    elfmap_sc_refine_steps: int = 1
    # Method used for the teacher diagonal SC refinement (_refine_sc):
    #   "picard"   — plain fixed-point chain sc_{k+1}=F(sc_k) (default, unchanged)
    #   "anderson" — Anderson acceleration of the same fixed point, matched budget
    #                (total teacher forwards == elfmap_sc_refine_steps). Uses the
    #                m/beta/lam knobs below. m_k<2 falls back to a Picard step.
    elfmap_sc_refine_method: str = "picard"
    elfmap_anderson_m: int = 3          # Anderson history size
    elfmap_anderson_beta: float = 1.0   # mixing (1.0 full step; <1 damps; >1 over-relax)
    elfmap_anderson_lam: float = 1e-4   # Tikhonov reg on the LS solve
    # SC-free student regime. When True, every student forward
    # (graded loss pass + off-diag A/B target passes) feeds a ZERO SC slot and is
    # conditioned only on (z, s, t, w); only the frozen teacher self-conditions
    # (diagonal target, refined N steps). The no-SC/SC regime is encoded in w
    # (no-SC rows -> w=0). Skips sc_signal and the off-diag b_sc_mode chain; the
    # elfmap_offdiag_b_sc_mode / elfmap_student_offdiag_sc_mode knobs are inert.
    # Requires num_self_cond_cfg_tokens > 0 and self_cond_prob > 0.
    elfmap_student_sc_free: bool = False
    # Off-diagonal target SC source in the SC-free regime. The
    # off-diag target is built from the live student (the single-time teacher
    # cannot do s!=t); the GRADED student stays SC-free. The first half-step A is
    # always SC-free; this knob picks B's SC slot at the midpoint X_su:
    #   "free"    — zeros   (no off-diag self-conditioning)
    #   "chained" — A_pred  (chain the s->u half-step prediction)
    #   "rerun"   — an elfmap_sc_refine_steps-step refinement at X_su
    # Ignored unless elfmap_student_sc_free.
    elfmap_scfree_offdiag_mode: str = "free"
    # CDEQ-teacher mode (Stage 3): the frozen flow-map teacher is a
    # Stage-2 FPFlow checkpoint instead of a single-time vanilla ELF. When True,
    # the teacher is built two-time + with the phase-token bank (to match the ckpt
    # under strict load) and the diagonal target is produced by the CDEQ "1
    # forward/step" inference contract (zero SC slot, phase=cdeq_inference_phase,
    # s=t) — NOT by _refine_sc/Anderson iteration. Requires cdeq_num_phase_tokens>0
    # and self_cond_prob>0; reuses cdeq_num_phase_tokens / cdeq_inference_phase to
    # match the teacher. The STUDENT also gets the phase bank but PINNED to a
    # constant phase = cdeq_inference_phase (tau_0) via ELF.const_phase, so the
    # warm-started student reproduces the teacher's 1-forward diagonal target
    # bit-exactly (diag loss = 0 at init). The student never varies phase.
    elfmap_teacher_is_cdeq: bool = False

    # CDEQ (Stage 2) — consistency distillation of the SC refinement axis.
    # Off by default. Selected by cdeq_enabled in train_lightning.py (cdeq > elfmap > base).
    # Subclasses FPFlowMap: reuses the frozen-teacher loader, warm-start, EMA,
    # and checkpoint hooks; requires double_time_emb=True, self_cond_prob>0,
    # num_self_cond_cfg_tokens>0, and a Stage-1 teacher_ckpt_path.
    cdeq_enabled: bool = False
    cdeq_num_phase_tokens: int = 4         # dedicated phase/consistency-time prefix tokens (NON-resumable if changed)
    cdeq_tau_eps: float = 0.002            # consistency-time map lower bound (official EPSILON)
    cdeq_tau_T: float = 5.0                # consistency-time map upper bound (official T)
    cdeq_rho: float = 0.3                  # tau_k = eps+(1-e^{-rho k})(T-eps); scale so (1-e^{-rho*K})~=1. Unsourced/tunable.
    cdeq_gamma: float = 1.0                # c_skip exponent ((tau-eps)/(T-eps))^gamma. 1.0 = LINEAR (code-faithful); tunable.
    cdeq_K: int = 20                       # online teacher trajectory length (free choice; cost driver)
    cdeq_anderson_m: int = 3               # teacher-trajectory Anderson history (ELF choice, not a CDEQ value)
    cdeq_anderson_beta: float = 0.9        # teacher-trajectory Anderson mixing (ELF choice)
    cdeq_anderson_lam: float = 1e-4        # teacher-trajectory Anderson Tikhonov reg
    cdeq_student_beta: float = 1.0         # student structural-AA-step mixing (code-backed; 1.0 → clean degenerate inference)
    cdeq_lambda1: float = 0.8              # L = lambda1*L_global(anchor sc*) + (1-lambda1)*L_local(consistency). Paper Table 5.
    cdeq_loss_metric: str = "mse"          # L_local metric: "mse" (CDEQ paper+code use MSE) | "pseudo_huber" (optional, NOT a CDEQ reference)
    cdeq_huber_c: float = None             # Pseudo-Huber c; None -> 0.00054*sqrt(D) at runtime
    cdeq_p_aug: float = 0.1                # snap-to-equilibrium augmentation prob (code value)
    cdeq_k_min: int = 1                    # aug window low (code range(1, n-3) => 1)
    cdeq_k_tail: int = 3                   # aug window high (code => 3)
    cdeq_p_coldstart: float = 0.0          # ABLATION knob. Default 0 = paper/code fidelity: k~U{2..K}, rely on consistency-generalization for the cold (0,0,tau_0) inference input. >0 force-trains that exact one-shot point -> sc* (non-CDEQ addition).
    cdeq_inference_phase: float = 0.002    # tau fed at inference (= tau_0 = tau_eps; c_skip=0 -> pure network)

    # Per-epoch eval — flowmap variant
    epoch_eval_method_kind: str = "sde"      # "sde" | "flowmap"
    flow_map_num_steps_list: list = [1, 2, 4, 32]
    flow_map_self_cond_cfg_scales: list = [1.0, 1.5, 2.0, 3.0]
    # Fire the per-epoch eval callback every N `trainer.global_step` ticks
    # instead of at epoch end (0 = disabled → fall back to on_train_epoch_end).
    # Under manual optim with Muon+AdamW, global_step increments 2x per real
    # opt step; e.g., 5000 here means eval every 2500 real optimizer steps.
    epoch_eval_step_freq: int = 0


def load_config_from_yaml(path: Optional[str]) -> Config:
    cfg = Config()
    if not path or not os.path.isfile(path):
        return cfg
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}
    for k, v in d.items():
        if k == "sampling_configs":
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _coerce(value: str, target_type) -> object:
    if target_type is bool:
        return value.lower() in ("true", "1", "yes")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def apply_config_overrides(config: Config, overrides: List[str]) -> Config:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override: '{override}'")
        name, val = override.split("=", 1)
        name, val = name.strip(), val.strip()
        if not hasattr(config, name):
            raise ValueError(f"No config field '{name}'")
        if val.lower() == "none":
            setattr(config, name, None)
            continue
        cur = getattr(config, name)
        target_type = type(cur) if cur is not None else config.__annotations__.get(name, str)
        setattr(config, name, _coerce(val, target_type))
    return config


def load_sampling_configs(path: str) -> List[SamplingConfig]:
    with open(path, "r") as f:
        entries = yaml.safe_load(f)
    return [SamplingConfig(**e) for e in entries]


def load_config(path: Optional[str], overrides: Optional[List[str]] = None) -> Config:
    """Single entry point used by train/eval: YAML defaults -> CLI overrides ->
    sampling configs. Sampling configs resolve last so a
    `--config_override sampling_configs_path=...` points them at the final path."""
    cfg = load_config_from_yaml(path)
    if overrides:
        cfg = apply_config_overrides(cfg, overrides)
    if cfg.sampling_configs_path:
        cfg.sampling_configs = load_sampling_configs(cfg.sampling_configs_path)
    return cfg

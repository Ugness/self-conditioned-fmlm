#!/usr/bin/env python
"""Canonical 1000-sample eval against a Lightning checkpoint.

Usage:
  cd pytorch_lightning/
  torchrun --nproc_per_node=8 --master_port=29510 eval_lightning.py \
      --config configs/training_configs/train_owt_ELF-B_gpt2.yml \
      --checkpoint_path outputs/elf_b-owt-lightning/last.ckpt \
      --num_samples 1000

Mirrors `PerEpochGenEvalCallback` (32-step SDE γ=1.5 SC-CFG=3 by default) but
takes `num_samples` as a CLI arg and runs standalone (no Trainer loop). Writes
the generated samples + metrics into outputs/<output_dir>/eval_<run_name>/.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from configs.config import SamplingConfig, load_config
from encoders import build_encoder
from lightning_module import EMA
from modules.model import ELF_models
from utils.generation_utils import (
    build_run_name, dlm_decode_batch, generate_samples, mask_after_eos,
)
from utils.metrics_utils import Metrics as PPLMetrics
from utils.sampling_utils import get_flowmap_time_steps, get_sampling_steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--eval_batch_size", type=int, default=None,
                   help="Per-rank generation/decode batch size. Default: "
                        "global_batch_size // world_size. Lower it to bound peak "
                        "memory (decode logits ~0.19 GiB/sample) without changing "
                        "num_samples or the metric.")
    p.add_argument("--num_sampling_steps", type=int, default=32)
    p.add_argument("--sde_gamma", type=float, default=1.5,
                   help="churn for sde/cdeq; also the gamma of the flow-map 'ctm' sampler")
    p.add_argument("--self_cond_cfg_scale", type=float, default=3.0)
    p.add_argument("--sampling_method", choices=["sde", "ode", "flowmap", "ctm", "cdeq"], default="sde")
    p.add_argument("--time_schedule", choices=["logit_normal", "logit_normal_uniform", "uniform"], default="logit_normal")
    p.add_argument("--decode_checkpoint_path", type=str, default=None,
                   help="If set, ALSO decode the trajectory model's final embedding with this "
                        "model (e.g. the flow teacher) -> setup C, alongside the trajectory "
                        "model's own decode (setup B), from the SAME latent.")
    p.add_argument("--decode_double_time_emb", action="store_true",
                   help="double_time_emb for the decode model (a single-time teacher = False; "
                        "omit for the teacher).")
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


def _all_gather_ids(local: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return local
    sizes = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(sizes, local)
    return torch.cat(sizes, dim=0)


def main():
    args = parse_args()

    # Distributed init (torchrun sets LOCAL_RANK / RANK / WORLD_SIZE / MASTER_*).
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cfg = load_config(args.config, args.config_override)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size

    encoder = build_encoder(
        cfg.encoder_model_name,
        dtype=torch.float32,
        feature_layer=cfg.feature_layer,
    ).to(device)
    encoder.eval()
    encoder_dim = encoder.d_model

    model = ELF_models[cfg.model](
        text_encoder_dim=encoder_dim, max_length=cfg.max_length,
        attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
        num_time_tokens=cfg.num_time_tokens,
        num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=cfg.num_model_mode_tokens,
        bottleneck_dim=cfg.bottleneck_dim,
        self_cond_input=(cfg.self_cond_prob > 0),
        use_flash=cfg.use_flash,
        double_time_emb=cfg.double_time_emb,
        # Stage-3 flow-map students (elfmap_teacher_is_cdeq) carry the phase bank
        # pinned to a constant tau_0; mirror the training-time build so the ckpt
        # loads strictly and eval runs at the same phase the student trained on.
        num_phase_tokens=(cfg.cdeq_num_phase_tokens
                          if (getattr(cfg, "cdeq_enabled", False)
                              or getattr(cfg, "elfmap_teacher_is_cdeq", False))
                          else 0),
        const_phase=(cfg.cdeq_inference_phase
                     if (getattr(cfg, "elfmap_teacher_is_cdeq", False)
                         and not getattr(cfg, "cdeq_enabled", False))
                     else None),
    ).to(device)

    # Load Lightning checkpoint. The Lightning state_dict has top-level
    # `state_dict` containing keys like "encoder.<...>" and "model.<...>".
    if rank == 0:
        print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    # Pull out the `model.*` submodule weights (the ELF transformer).
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if rank == 0:
        if missing:    print(f"WARNING missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected: print(f"WARNING unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # EMA: restore from checkpoint and swap into the model.
    ema = EMA(model, decay=cfg.ema_decay1)
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"], device=device)
        ema.swap_in(model)
        if rank == 0:
            print(f"EMA weights swapped in (decay={ema.decay}).")
    elif rank == 0:
        print("WARNING: no EMA in checkpoint; using raw weights.")

    model.eval()

    # Optional second model that performs ONLY the final embedding->token decode
    # (setup C: flow-map trajectory decoded by the flow teacher). The trajectory
    # always comes from `model`; `decode_model` decodes the same latent. Built by
    # the same load+EMA pattern as the primary model, but with its own
    # double_time_emb (a single-time teacher uses double_time_emb=False).
    decode_model = None
    if args.decode_checkpoint_path:
        decode_model = ELF_models[cfg.model](
            text_encoder_dim=encoder_dim, max_length=cfg.max_length,
            attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
            num_time_tokens=cfg.num_time_tokens,
            num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
            vocab_size=vocab_size,
            num_model_mode_tokens=cfg.num_model_mode_tokens,
            bottleneck_dim=cfg.bottleneck_dim,
            self_cond_input=(cfg.self_cond_prob > 0),
            use_flash=cfg.use_flash,
            double_time_emb=args.decode_double_time_emb,
            num_phase_tokens=(cfg.cdeq_num_phase_tokens
                              if getattr(cfg, "cdeq_enabled", False) else 0),
        ).to(device)
        if rank == 0:
            print(f"Loading decode checkpoint: {args.decode_checkpoint_path}")
        dck = torch.load(args.decode_checkpoint_path, map_location=device, weights_only=False)
        dstate = dck.get("state_dict", dck)
        dmodel_state = {k[len("model."):]: v for k, v in dstate.items() if k.startswith("model.")}
        dmissing, dunexpected = decode_model.load_state_dict(dmodel_state, strict=False)
        if rank == 0:
            if dmissing:    print(f"WARNING decode missing keys ({len(dmissing)}): {dmissing[:5]}...")
            if dunexpected: print(f"WARNING decode unexpected keys ({len(dunexpected)}): {dunexpected[:5]}...")
        dema = EMA(decode_model, decay=cfg.ema_decay1)
        if "ema" in dck:
            dema.load_state_dict(dck["ema"], device=device)
            dema.swap_in(decode_model)
            if rank == 0:
                print(f"Decode-model EMA weights swapped in (decay={dema.decay}).")
        elif rank == 0:
            print("WARNING: no EMA in decode checkpoint; using raw weights.")
        for p in decode_model.parameters():
            p.requires_grad_(False)
        decode_model.eval()

    # Sample.
    sc = SamplingConfig(
        sampling_method=args.sampling_method, num_sampling_steps=[args.num_sampling_steps],
        cfgs=[1], self_cond_cfg_scales=[args.self_cond_cfg_scale],
        sde_gamma=args.sde_gamma,
        time_schedule=args.time_schedule,
    )
    pad_token_id = (tokenizer.eos_token_id if cfg.pad_token == "eos" else tokenizer.pad_token_id)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    samples_per_rank = (args.num_samples + world_size - 1) // world_size
    per_rank_batch = (args.eval_batch_size if args.eval_batch_size
                      else cfg.global_batch_size // max(1, world_size))
    per_rank_batch = max(1, min(per_rank_batch, samples_per_rank))
    num_batches = (samples_per_rank + per_rank_batch - 1) // per_rank_batch

    def _decode_to_texts(decode_m, latent, t_final_val):
        """Decode a clean latent -> token ids -> gathered text list (one call per model)."""
        predicted_ids = dlm_decode_batch(
            decode_m, latent, config=cfg,
            self_cond_cfg_scale=args.self_cond_cfg_scale,
            t_final_val=t_final_val,
        )
        if getattr(cfg, "mask_after_eos", True):
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id,
                                           pad_token_id=pad_token_id)
        gathered = _all_gather_ids(predicted_ids, world_size)
        return [tokenizer.decode(row.cpu().tolist(), skip_special_tokens=True)
                for row in gathered]

    # generated_texts: trajectory model's own decode (setup A or B).
    # generated_texts_x: decode_model's decode of the SAME latent (setup C).
    generated_texts: list = []
    generated_texts_x: list = []
    for batch_idx in tqdm(range(num_batches), desc="generate", disable=(rank != 0)):
        cur = min(per_rank_batch, samples_per_rank - batch_idx * per_rank_batch)
        if cur <= 0: break
        # Stable per-rank seed (same form as the callback).
        seed = cfg.seed * 1000003 + 991 + batch_idx * 97 + rank
        gen = torch.Generator(device=device).manual_seed(seed)

        if args.sampling_method in ("flowmap", "ctm"):
            t_steps = get_flowmap_time_steps(
                args.num_sampling_steps, tau_kind=cfg.elfmap_tau_kind,
                P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
                device=device, dtype=torch.float32,
            )
        else:
            t_steps = get_sampling_steps(
                gen, n_steps=args.num_sampling_steps, device=device,
                time_schedule=args.time_schedule,
                P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
            )
        z = torch.randn(
            (cur, cfg.max_length, encoder_dim),
            generator=gen, device=device,
        ) * cfg.denoiser_noise_scale

        with torch.no_grad():
            latent = generate_samples(
                model, z, t_steps, cond_seq=None, cond_seq_mask=None,
                config=cfg, sampling_config=sc,
                cfg_scale=1.0, self_cond_cfg_scale=args.self_cond_cfg_scale,
                generator=gen,
            )
            t_final_val = float(t_steps[-1].item())
            generated_texts.extend(_decode_to_texts(model, latent, t_final_val))
            if decode_model is not None:
                generated_texts_x.extend(_decode_to_texts(decode_model, latent, t_final_val))

    # Rank-0 writes JSONL + computes PPL on the consolidated set(s).
    if rank == 0:
        run_name = build_run_name(
            args.sampling_method, args.num_sampling_steps, 1.0, args.self_cond_cfg_scale,
            args.time_schedule, args.sde_gamma, suffix=f"eval{args.num_samples}",
        )
        ppl_eval = PPLMetrics(
            gen_ppl_eval_model_name_or_path=cfg.eval_ppl_model,
            eval_ppl_batch_size=cfg.eval_ppl_batch_size,
            eval_context_size=cfg.eval_ppl_max_length,
            device=str(device),
        )

        def _score_and_write(texts, name, decode_checkpoint):
            # Trim over-generation from the gather (samples_per_rank * world_size
            # can exceed num_samples by up to world_size-1).
            texts = texts[: args.num_samples]
            out_dir = os.path.join(cfg.output_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "all_generated.jsonl"), "w", encoding="utf-8") as f:
                for i, t in enumerate(texts):
                    f.write(json.dumps({"id": i, "generated": t}, ensure_ascii=False) + "\n")
            nonempty = [s for s in texts if isinstance(s, str) and s.strip()]
            print(f"[{name}] Generated {len(texts)} samples; {len(nonempty)} non-empty.")
            metrics_path = os.path.join(out_dir, "metrics.jsonl")
            if not nonempty:
                with open(metrics_path, "w") as f:
                    f.write(json.dumps({"ppl": None, "mean_entropy": None,
                                        "checkpoint": args.checkpoint_path,
                                        "decode_checkpoint": decode_checkpoint,
                                        "num_samples": args.num_samples,
                                        "note": "all generations empty"}) + "\n")
                return
            res = ppl_eval.record_generative_perplexity(
                text_samples=nonempty, max_length=cfg.eval_ppl_max_length, retokenize=True,
            )
            row = {
                "ppl": float(res["ppl"]), "mean_entropy": float(res["mean_entropy"]),
                "checkpoint": args.checkpoint_path,
                "decode_checkpoint": decode_checkpoint,
                "num_samples": args.num_samples,
                "num_nonempty": len(nonempty),
                "num_sampling_steps": args.num_sampling_steps,
                "sde_gamma": args.sde_gamma,
                "self_cond_cfg_scale": args.self_cond_cfg_scale,
            }
            with open(metrics_path, "w") as f:
                f.write(json.dumps(row) + "\n")
            print(f"==> [{name}] gen_ppl={row['ppl']:.4f}  sample_entropy={row['mean_entropy']:.4f}")
            print(f"==> wrote {metrics_path}")

        # Setup A/B: the trajectory model decodes its own latent.
        _score_and_write(generated_texts, run_name, None)
        # Setup C: the decode model (flow teacher) decodes the SAME latent.
        if decode_model is not None:
            _score_and_write(generated_texts_x, run_name + "-xdecode",
                             args.decode_checkpoint_path)

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Compute scalar latent_mean / latent_std for the GPT-2 encoder over a random
subset of OWT rows.

The ELF training loop normalizes encoder latents with two scalars
(`latent_mean`, `latent_std`) — see `utils/encoder_utils.py::encode_text`.
This script estimates those scalars by running the frozen encoder over a random
subset of prepared rows in bf16 (matching training-time precision) and
accumulating sum, sumsq, and count over all valid scalar entries
(B · L · d_model).

Usage:
    python pytorch_lightning/scripts/compute_gpt2_latent_stats.py \\
        --data_path ../dataset/openwebtext-gpt2-flm/train \\
        --num_samples 50000 \\
        --batch_size 16 \\
        --out_json gpt2_latent_stats.json
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from datasets import load_from_disk

# Allow running as `python pytorch_lightning/scripts/compute_gpt2_latent_stats.py`
# from the repo root: ensure pytorch_lightning/ is on sys.path so `encoders`
# resolves the same way training does.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYL_DIR = os.path.dirname(_THIS_DIR)
if _PYL_DIR not in sys.path:
    sys.path.insert(0, _PYL_DIR)

from encoders import build_encoder  # noqa: E402


_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True,
                    help="HF dataset on disk (e.g. ../dataset/openwebtext-gpt2-flm/train)")
    ap.add_argument("--encoder_model_name", default="gpt2-large")
    ap.add_argument("--feature_layer", default="last",
                    help='"last" (default) or an int hidden-state index')
    ap.add_argument("--num_samples", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="gpt2_latent_stats.json")
    args = ap.parse_args()

    feature_layer = int(args.feature_layer) if args.feature_layer.lstrip("-").isdigit() else args.feature_layer
    dtype = _DTYPES[args.dtype]

    print(f"[LOAD] dataset={args.data_path}")
    ds = load_from_disk(args.data_path)
    n_total = len(ds)
    n_take = min(args.num_samples, n_total)
    print(f"[DATA] total_rows={n_total}  num_samples={n_take}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g)[:n_take].tolist()

    print(f"[ENCODER] {args.encoder_model_name}  dtype={args.dtype}  feature_layer={feature_layer}")
    encoder = build_encoder(
        args.encoder_model_name,
        dtype=dtype,
        feature_layer=feature_layer,
    ).to(args.device).eval()

    # Online accumulators on CPU as float64 to keep precision over many
    # billions of scalar entries (50k × 1024 × 1280 ≈ 6.6e10).
    running_sum = 0.0
    running_sumsq = 0.0
    running_count = 0

    t0 = time.time()
    n_batches = math.ceil(n_take / args.batch_size)
    print(f"[RUN] batches={n_batches}  batch_size={args.batch_size}")

    for bi in range(n_batches):
        idx = perm[bi * args.batch_size : (bi + 1) * args.batch_size]
        rows = ds[idx]

        input_ids = torch.tensor(rows["input_ids"], dtype=torch.long, device=args.device)
        # Each prepared row already has attention_mask = ones(block_size); use the
        # same 2D mask shape the row carries so we exercise the same code path.
        attn = torch.tensor(rows["attention_mask"], dtype=torch.float32, device=args.device)
        if attn.dim() == 3 and attn.shape[1] == 1:
            attn = attn.squeeze(1)

        with torch.no_grad():
            h = encoder(input_ids=input_ids, attention_mask=attn)  # (B, L, d_model)

        valid = (attn > 0).unsqueeze(-1)  # (B, L, 1)
        h_f32 = h.float()
        # Sum/sumsq only over valid scalar entries (every d_model entry inherits
        # the row-position validity).
        h_valid_sum = (h_f32 * valid).sum().item()
        h_valid_sumsq = ((h_f32 * h_f32) * valid).sum().item()
        n_valid = int(valid.sum().item()) * h_f32.shape[-1]

        running_sum += h_valid_sum
        running_sumsq += h_valid_sumsq
        running_count += n_valid

        if (bi + 1) % 50 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            rate = (bi + 1) * args.batch_size / max(elapsed, 1e-9)
            mean_so_far = running_sum / running_count
            var_so_far = max(running_sumsq / running_count - mean_so_far ** 2, 0.0)
            std_so_far = math.sqrt(var_so_far)
            print(f"[STEP] {bi+1}/{n_batches}  elapsed={elapsed:.1f}s  "
                  f"rate={rate:.1f} rows/s  mean={mean_so_far:.6f}  std={std_so_far:.6f}")

    mean = running_sum / running_count
    var = max(running_sumsq / running_count - mean ** 2, 0.0)
    std = math.sqrt(var)

    print(f"[RESULT] num_samples={n_take}  scalar_count={running_count}  "
          f"mean={mean:.6f}  std={std:.6f}")

    out = {
        "encoder_model_name": args.encoder_model_name,
        "feature_layer": args.feature_layer,
        "dtype": args.dtype,
        "data_path": args.data_path,
        "num_samples": n_take,
        "scalar_count": running_count,
        "latent_mean": mean,
        "latent_std": std,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVE] {args.out_json}")


if __name__ == "__main__":
    main()

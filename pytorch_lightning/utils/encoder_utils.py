"""Encoder utilities."""

from typing import Tuple

import numpy as np
import torch


def encode_text(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                encoder: torch.nn.Module, latent_mean: float, latent_std: float) -> torch.Tensor:
    """Run the frozen encoder and normalize the latents."""
    with torch.no_grad():
        latents = encoder(input_ids=input_ids, attention_mask=attention_mask)
    return (latents - latent_mean) / latent_std


def build_self_attn_cond_masks(is_cond, is_valid, xp=np) -> Tuple:
    """Return (encoder_attention_mask, attention_mask, cond_seq_mask) as float32 arrays.

    encoder_attention_mask: (B, L, L) — cond rows attend only to cond cols;
    non-cond rows attend to all valid cols.
    """
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :])
        | ((~is_cond[:, :, None]) & is_valid[:, None, :])
    ).astype(xp.float32)
    return encoder_attention_mask, is_valid.astype(xp.float32), is_cond.astype(xp.float32)

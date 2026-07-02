"""Encoder interface.

The ELF transformer talks to its frozen text encoder ONLY through this interface.
To add a future LLaMA / GPT-2-large adapter, subclass `EncoderInterface`, implement
`forward()` so it produces last-hidden-state in the exact dtype/shape the original
pretrained model emits, register it in `pytorch_lightning/encoders/__init__.py`,
and add a parity test under `tests/test_encoder_parity_<name>.py`.

Hard rule: each adapter must reproduce its reference implementation **bit-for-bit
or to within fp32 round-off** on a fixed input batch. We never re-implement,
fuse, or apply FlashAttention inside an encoder — pretrained-encoder exact
reproduction supersedes any speedup.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class EncoderInterface(nn.Module, ABC):
    """Frozen text encoder contract for ELF.

    The ELF transformer needs only three things from an encoder:
      1. `d_model` — the hidden size of the last-hidden-state.
      2. `attention_mask_convention` — `1=valid, 0=masked` (current standard).
      3. `forward(input_ids, attention_mask) -> last_hidden_state`.

    Adapters are free to use either a 2D (B, L) or 3D (B, L, L) attention mask —
    HF stacks support both. The collator emits a (B, L, L) mask for ELF's
    structured cond/target attention.

    All adapters are frozen at construction (`requires_grad=False`).
    """

    #: Last-hidden-state width. Sub-classes must set this in `__init__`.
    d_model: int

    #: For documentation; do not change without updating the rest of the model.
    attention_mask_convention: str = "1=valid, 0=masked"

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return last_hidden_state of shape (B, L, d_model)."""
        raise NotImplementedError

    @torch.no_grad()
    def parity_check(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        reference_outputs: torch.Tensor,
        *,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> dict:
        """Run the adapter on `(input_ids, attention_mask)` and compare against
        `reference_outputs`. Returns a dict with `max_abs_diff`, `mean_abs_diff`,
        and `passed`. Concrete encoder tests call this with reference outputs
        captured from the original (e.g., HF or JAX) implementation."""
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        max_abs = (out - reference_outputs).abs().max().item()
        mean_abs = (out - reference_outputs).abs().mean().item()
        passed = torch.allclose(out, reference_outputs, atol=atol, rtol=rtol)
        return {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs, "passed": passed}

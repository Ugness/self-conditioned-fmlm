"""GPT-2 encoder adapter.

Wraps `transformers.GPT2Model` so the ELF model can talk to it through
`EncoderInterface`. We deliberately do NOT touch any internal op of the GPT-2
encoder — no FlashAttention swap, no fused norm, no precision change inside it.
The encoder is a frozen, externally-defined component.

GPT-2 is a causal decoder-only stack: every token only
attends to its left context. We do not override that — `GPT2Model` runs causal
natively. The DiT denoiser remains bidirectional; only the latent that the
denoiser learns to produce is left-context-only.

`feature_layer` must be `"last"` (return `last_hidden_state`); this is the
encoder output ELF is trained on.
"""

import torch
from transformers import GPT2Model

from .base import EncoderInterface


_GPT2_ALIASES = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
}


class GPT2Encoder(EncoderInterface):
    """Frozen GPT-2 encoder. Causal attention, no internal op changes.

    Notes:
      * Defaults to bf16 weights; the call-site can override via the `dtype`
        kwarg passed through `build_encoder`.
      * Supports 2D `(B, L)` or 3D `(B, L, L)` `attention_mask` — HF's
        `get_extended_attention_mask` handles both. The collator emits 3D
        for the cond/target attention layout; GPT-2 still applies its own
        causal mask on top.
    """

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        feature_layer: str = "last",
    ):
        super().__init__(model_name=model_name)
        if feature_layer != "last":
            raise ValueError(
                f"GPT2Encoder only supports feature_layer='last', got "
                f"{feature_layer!r}.")
        resolved = _GPT2_ALIASES.get(model_name, model_name)
        self.model = GPT2Model.from_pretrained(resolved, torch_dtype=dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.d_model = self.model.config.n_embd
        self.feature_layer = feature_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Defensive clamp: some prepped OWT datasets contain a small number of
        # stray ids equal to len(tokenizer) (= vocab_size + extra special tokens)
        # because the upstream preprocess added pad_token before grouping. GPT-2
        # `wte` only has rows [0, vocab_size); a single stray id triggers an
        # async vectorized_gather_kernel index-out-of-bounds assert that kills
        # the training step. Clamping is a no-op for clean datasets.
        input_ids = input_ids.clamp_max(self.model.config.vocab_size - 1)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

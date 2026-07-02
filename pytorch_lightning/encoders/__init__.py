"""Encoder registry. Future LLaMA adapters register here."""

from typing import Dict, Type

from .base import EncoderInterface
from .gpt2 import GPT2Encoder


_REGISTRY: Dict[str, Type[EncoderInterface]] = {
    "gpt2": GPT2Encoder,
    "gpt2-medium": GPT2Encoder,
    "gpt2-large": GPT2Encoder,
    "gpt2-xl": GPT2Encoder,
}


def build_encoder(name: str, **kwargs) -> EncoderInterface:
    """Factory: resolve `name` to a registered encoder class and construct it."""
    if name in _REGISTRY:
        cls = _REGISTRY[name]
    elif name.startswith("openai-community/gpt2"):
        cls = GPT2Encoder
    else:
        raise ValueError(
            f"No encoder registered for {name!r}. "
            f"Registered: {sorted(_REGISTRY)}. "
            f"To add a new pretrained encoder, implement EncoderInterface in "
            f"pytorch_lightning/encoders/<name>.py and register it here."
        )
    return cls(name, **kwargs)


__all__ = ["EncoderInterface", "GPT2Encoder", "build_encoder"]

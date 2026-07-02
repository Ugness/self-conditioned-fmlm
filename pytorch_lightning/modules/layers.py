"""ELF layers with optional FlashAttention path.

When `use_flash=True`, attention runs `F.scaled_dot_product_attention` under
the FA/EFFICIENT SDPA backend with Q/K/V cast to bf16 at the boundary and back
to the input dtype after — the only precision change in the model path.

The frozen pretrained encoder is NOT routed through this — it has its own
attention path (see `pytorch_lightning/encoders/t5.py`). Exact encoder
reproduction supersedes any speedup.
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Init defaults match JAX: xavier_uniform for kernels, zeros for biases.
def _make_linear(in_features: int, out_features: int, *, bias: bool = True,
                 kernel_init: Callable = nn.init.xavier_uniform_,
                 bias_init: Optional[Callable] = nn.init.zeros_) -> nn.Linear:
    lin = nn.Linear(in_features, out_features, bias=bias)
    kernel_init(lin.weight)
    if bias and bias_init is not None:
        bias_init(lin.bias)
    return lin


def _normal_002_(t: torch.Tensor) -> torch.Tensor:
    return nn.init.normal_(t, mean=0.0, std=0.02)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight.to(torch.float32) * x_fp32).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class TextRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim: int, pt_seq_len: int = 512, ft_seq_len: Optional[int] = None,
                 theta: float = 10000.0, num_empty_token: int = 0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.ft_seq_len = ft_seq_len if ft_seq_len is not None else pt_seq_len
        self.theta = theta
        self.num_empty_token = num_empty_token

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        pos = torch.arange(self.ft_seq_len).float() / self.ft_seq_len * self.pt_seq_len
        freqs_main = torch.einsum("..., f -> ... f", pos, freqs)
        freqs_main = freqs_main.unsqueeze(-1).expand(*freqs_main.shape, 2).flatten(-2)

        D = freqs_main.shape[-1]
        cos_parts, sin_parts = [], []
        if num_empty_token > 0:
            cos_parts.append(torch.ones(num_empty_token, D))
            sin_parts.append(torch.zeros(num_empty_token, D))
        cos_parts.append(torch.cos(freqs_main))
        sin_parts.append(torch.sin(freqs_main))
        self.register_buffer("freqs_cos", torch.cat(cos_parts, dim=0), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(sin_parts, dim=0), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        cos = self.freqs_cos.to(dtype=t.dtype, device=t.device).view(1, 1, *self.freqs_cos.shape)
        sin = self.freqs_sin.to(dtype=t.dtype, device=t.device).view(1, 1, *self.freqs_sin.shape)
        return t * cos + _rotate_half(t) * sin


class BottleneckTextProj(nn.Module):
    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.proj1 = _make_linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = _make_linear(bottleneck_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


def _broadcast_attn_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    """Reshape attn_mask to (B, 1, Lq_or_1, Ls) for SDPA broadcasting.
    2D (B, L) -> (B, 1, 1, L); 3D (B, L, L) -> (B, 1, L, L)."""
    if attn_mask.dim() == 2:
        return attn_mask[:, None, None, :]
    if attn_mask.dim() == 3:
        return attn_mask[:, None, :, :]
    return attn_mask


def _manual_sdpa_fp32(q, k, v, attn_mask=None):
    scale = 1.0 / math.sqrt(q.size(-1))
    attn_weight = torch.einsum("bhld,bhsd->bhls", q.float(), k.float()) * scale
    if attn_mask is not None:
        mask = _broadcast_attn_mask(attn_mask)
        attn_weight = torch.where(mask == 0, torch.full_like(attn_weight, -1e9), attn_weight)
    attn_weight = torch.softmax(attn_weight, dim=-1).to(v.dtype)
    return torch.einsum("bhls,bhsd->bhld", attn_weight, v)


def _flash_sdpa_bf16(q, k, v, attn_mask=None):
    """PyTorch SDPA in bf16 — kept as the masked-attention fallback because
    FA4's `flash_attn_func` does not accept an arbitrary dense (B, L, L) mask
    (it supports `causal`, `window_size`, `mask_mod` callable, or block-sparse).
    Q/K/V cast to bf16 at the boundary, output cast back to input dtype for the
    residual add — the only precision change in the model.

    Mask is additive: 1=valid -> 0, 0=masked -> finfo.min (bf16-safe; -inf can
    NaN through softmax).
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    target_dtype = torch.bfloat16
    out_dtype = q.dtype
    qb, kb, vb = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)

    additive_mask = None
    if attn_mask is not None:
        m = _broadcast_attn_mask(attn_mask).to(target_dtype)
        additive_mask = (m == 0).to(target_dtype) * torch.finfo(target_dtype).min

    backend = SDPBackend.EFFICIENT_ATTENTION if additive_mask is not None else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=additive_mask)
    return out.to(out_dtype)


def _flash4_bf16(q, k, v):
    """FlashAttention-4 path via `flash_attn.cute.flash_attn_func` (CuTeDSL,
    optimized for Hopper/Blackwell). Q/K/V cast to bf16 at the boundary; output
    cast back to the input dtype. The ELF model never passes a dense attention
    mask, so this no-mask variant covers every actual call site.

    Layout dance: SDPA uses (B, H, N, D); FA4 expects (B, N, H, D) with only
    the last dim (head_dim) contiguous and 16-byte aligned — overall contiguity
    is not required. After `transpose(1, 2)`, head_dim is still stride-1, so we
    pass the transposed view directly. The `.to(bf16)` cast preserves strides.
    """
    from flash_attn.cute import flash_attn_func

    target_dtype = torch.bfloat16
    out_dtype = q.dtype
    qb = q.transpose(1, 2).to(target_dtype)
    kb = k.transpose(1, 2).to(target_dtype)
    vb = v.transpose(1, 2).to(target_dtype)
    out, _ = flash_attn_func(qb, kb, vb)  # (output, lse|None)
    return out.transpose(1, 2).to(out_dtype)


class Attention(nn.Module):
    """Self-attention with optional RoPE, head-wise RMSNorm on Q/K, and an
    optional FlashAttention path."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 qk_norm: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 use_flash: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.use_flash = use_flash
        self.qkv = _make_linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = _make_linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope_fn=None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)
        if self.use_flash:
            # FA4 doesn't accept an arbitrary dense mask. Route to PyTorch SDPA
            # (EFFICIENT_ATTENTION) when a mask is present; FA4 otherwise.
            if attention_mask is None:
                out = _flash4_bf16(q, k, v)
            else:
                out = _flash_sdpa_bf16(q, k, v, attn_mask=attention_mask)
        else:
            out = _manual_sdpa_fp32(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        hidden = int(hidden_dim * 2 / 3)
        self.w12 = _make_linear(dim, 2 * hidden, bias=bias)
        self.w3 = _make_linear(hidden, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = _make_linear(frequency_embedding_size, hidden_size, bias=True,
                                   kernel_init=_normal_002_, bias_init=nn.init.zeros_)
        self.mlp_2 = _make_linear(hidden_size, hidden_size, bias=True,
                                   kernel_init=_normal_002_, bias_init=nn.init.zeros_)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.mlp_0(self.timestep_embedding(t, self.frequency_embedding_size))
        return self.mlp_2(F.silu(t_emb))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        out = patch_size * patch_size * out_channels
        self.linear = _make_linear(hidden_size, out, bias=True,
                                   kernel_init=nn.init.zeros_, bias_init=nn.init.zeros_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))

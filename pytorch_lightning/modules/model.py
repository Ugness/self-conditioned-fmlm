"""ELF transformer with FlashAttention switch on the model only (NOT the encoder).

Every `Attention` constructor receives `use_flash=<config>`.  The encoder is
wired in through `EncoderInterface` externally; nothing in this file touches it.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder, _normal_002_,
)


class ELFBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 use_flash: bool = False):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop,
                              use_flash=use_flash)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden, drop=proj_drop)

    def forward(self, x, rope_fn=None, attention_mask=None):
        x = x + self.attn(self.norm1(x), rope_fn=rope_fn, attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ELF(nn.Module):
    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        self_cond_input: bool = True,
        use_flash: bool = False,
        double_time_emb: bool = False,
        num_phase_tokens: int = 0,
        const_phase: float = None,
    ):
        super().__init__()
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")

        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.self_cond_input = self_cond_input
        self.use_flash = use_flash
        self.double_time_emb = double_time_emb
        self.num_phase_tokens = num_phase_tokens
        # Optional fixed consistency-time fed to the phase bank when a caller does
        # not pass `phase` on a denoiser forward (Stage-3 CDEQ-teacher mode: the
        # student carries the phase tokens but sits at a CONSTANT phase = tau_0,
        # the CDEQ cold-start inference point, so it reproduces the CDEQ teacher's
        # 1-forward diagonal target bit-exactly at warm-start). None = disabled
        # (real CDEQ student / plain models pass a varying or absent phase).
        self.const_phase = const_phase

        if self_cond_input:
            self.self_cond_proj = nn.Linear(2 * text_encoder_dim, text_encoder_dim, bias=True)
            nn.init.xavier_uniform_(self.self_cond_proj.weight)
            nn.init.zeros_(self.self_cond_proj.bias)

        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        _normal_002_(self.t_emb_tokens)
        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.empty(1, num_self_cond_cfg_tokens, hidden_size)
            )
            _normal_002_(self.self_cond_cfg_tokens)
        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            _normal_002_(self.mode_tokens)
        if num_phase_tokens > 0:
            # CDEQ Stage-2 phase / consistency-time conditioning.
            # Zero-init both the token bank AND the embedder's output projection so
            # the phase prefix contributes EXACTLY zero at warm-start init (the
            # phase pathway is a no-op until trained). NB: the extra zero prefix
            # tokens still participate in attention, so this is near- (not bit-)
            # identical to a num_phase_tokens=0 model — expected.
            self.phase_embedder = TimestepEmbedder(hidden_size)
            self.phase_emb_tokens = nn.Parameter(torch.zeros(1, num_phase_tokens, hidden_size))
            nn.init.zeros_(self.phase_embedder.mlp_2.weight)
            nn.init.zeros_(self.phase_embedder.mlp_2.bias)

        head_dim = hidden_size // num_heads
        prefix_len = (num_time_tokens
                      + (num_self_cond_cfg_tokens if num_self_cond_cfg_tokens > 0 else 0)
                      + (num_phase_tokens if num_phase_tokens > 0 else 0))
        empty_offset = prefix_len + (num_model_mode_tokens if num_model_mode_tokens > 0 else 0)
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=empty_offset,
        )

        q1, q3 = depth // 4, depth // 4 * 3
        self.blocks = nn.ModuleList([
            ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=(attn_drop if q3 > i >= q1 else 0.0),
                proj_drop=(proj_drop if q3 > i >= q1 else 0.0),
                use_flash=use_flash,
            )
            for i in range(depth)
        ])

        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        nn.init.xavier_uniform_(self.proj_kernel); nn.init.zeros_(self.proj_bias)
        nn.init.xavier_uniform_(self.unembed_kernel); nn.init.zeros_(self.unembed_bias)

        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=text_encoder_dim)

    def build_context(self, t, self_cond_cfg_scale=None, s=None, phase=None):
        B = t.shape[0]
        t_emb = self.t_embedder(t).unsqueeze(1)  # (B, 1, H)
        if self.double_time_emb:
            # Two-time conditioning: the first num_time_tokens // 2 prefix slots
            # carry the target time t, the rest carry the source time s. At
            # s == t this collapses to the vanilla single-time block.
            s_emb = self.t_embedder(t if s is None else s).unsqueeze(1)
            n_t = self.num_time_tokens // 2
            t_emb = torch.cat(
                [t_emb.expand(B, n_t, -1),
                 s_emb.expand(B, self.num_time_tokens - n_t, -1)], dim=1)
        out = [self.t_emb_tokens.expand(B, -1, -1) + t_emb]
        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            out.append(self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1))
        if self.num_phase_tokens > 0:
            # CDEQ phase prefix. ALWAYS emitted when the model has
            # phase tokens so the sequence length (and hence the RoPE empty_offset
            # alignment) is constant across calls — branches that don't pass a
            # phase (CE/decoder, dlm_decode) default to phase 0. Appended after the
            # SC-CFG block; all prefix tokens sit in the RoPE-identity region so
            # intra-prefix order does not affect text-token positions.
            ph = phase if phase is not None else t.new_zeros(B)
            ph_emb = self.phase_embedder(ph)
            out.append(self.phase_emb_tokens.expand(B, -1, -1) + ph_emb.unsqueeze(1))
        return out

    def forward(self, x, t, attention_mask=None,
                self_cond_cfg_scale=None, decoder_step_active: bool = False,
                s=None, phase=None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = x.shape[0]
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x)
        x = self.text_proj(x)

        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if not decoder_step_active:
                mode_tokens = torch.zeros_like(mode_tokens)
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones(B, self.num_model_mode_tokens,
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        # Constant-phase fallback (Stage-3): a phase-equipped student that did not
        # receive an explicit `phase` on a denoiser pass sits at self.const_phase
        # (= tau_0). Decoder/CE passes keep the default phase=0 to match the
        # teacher's decode regime.
        if (phase is None and not decoder_step_active
                and self.num_phase_tokens > 0 and self.const_phase is not None):
            phase = torch.full((B,), float(self.const_phase),
                               device=t.device, dtype=t.dtype)

        prefix_len = 0
        context = self.build_context(t, self_cond_cfg_scale=self_cond_cfg_scale, s=s, phase=phase)
        if context:
            prefix = torch.cat(context, dim=1)
            prefix_len = prefix.shape[1]
            x = torch.cat([prefix, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(B, prefix_len, dtype=attention_mask.dtype,
                                          device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        for block in self.blocks:
            x = block(x, rope_fn=self.feat_rope, attention_mask=attention_mask)
        x = x[:, prefix_len + model_mode_offset:]

        decoder_logits = None
        if decoder_step_active:
            decoder_logits = F.gelu(x @ self.proj_kernel + self.proj_bias) @ self.unembed_kernel + self.unembed_bias

        return self.final_layer(x), decoder_logits


def ELF_B(**kw): return ELF(depth=12, hidden_size=768, num_heads=12, **kw)
def ELF_M(**kw): return ELF(depth=24, hidden_size=1056, num_heads=16, **kw)
def ELF_L(**kw): return ELF(depth=32, hidden_size=1280, num_heads=16, **kw)
ELF_models = {"ELF-B": ELF_B, "ELF-M": ELF_M, "ELF-L": ELF_L}

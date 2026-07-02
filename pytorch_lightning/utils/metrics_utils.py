"""Generative perplexity + unigram entropy metrics (1:1 port of metrics_utils.py).

PPL: frozen HF causal LM (`config.eval_ppl_model`). Token mask matches JAX:
`(first_eos[:, 1:] | (input_ids != eos)[:, 1:])` — index 0 (post-BOS) excluded,
first EOS included. Entropy is per-sample unigram entropy in nats, then averaged.
"""

import math
import os
from typing import Dict, List

import numpy as np
import torch
import transformers
from tqdm import tqdm

from utils.logging_utils import log_for_0


class Metrics:
    """Generative-PPL + unigram-entropy evaluator (single-process)."""

    def __init__(self, gen_ppl_eval_model_name_or_path: str,
                 eval_ppl_batch_size: int = 64, eval_context_size: int = 1024,
                 device: str = "cuda"):
        self.model_name = gen_ppl_eval_model_name_or_path
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.eval_context_size = eval_context_size
        self.device = device

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        use_fast = "mt5" not in gen_ppl_eval_model_name_or_path.lower()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            gen_ppl_eval_model_name_or_path, use_fast=use_fast,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._model = None

    def _load_model(self):
        if self._model is None:
            import transformers
            log_for_0(f"Loading PPL model: {self.model_name}")
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.float32,
            ).to(self.device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model

    def _retokenize(self, text_samples: List[str], max_length: int):
        out = self.tokenizer(
            text_samples, return_tensors="np", return_token_type_ids=False,
            return_attention_mask=True, truncation=True, padding=True,
            max_length=max_length,
        )
        return out["input_ids"], out["attention_mask"]

    @torch.no_grad()
    def record_generative_perplexity(self, text_samples: List[str], max_length: int,
                                     retokenize: bool = True) -> Dict:
        model = self._load_model()
        eos_token_id = self.tokenizer.eos_token_id

        if retokenize:
            samples, attn_mask = self._retokenize(text_samples, max_length=max_length)
        else:
            samples = np.asarray(text_samples)
            attn_mask = np.ones_like(samples)

        N = samples.shape[0]
        batch_size = self.eval_ppl_batch_size or N
        batch_size = max(1, min(batch_size, N))
        num_batches = (N + batch_size - 1) // batch_size
        log_for_0(f"PPL: batch_size={batch_size}, {num_batches} batches")

        per_sample_nll_sum = np.zeros(N, dtype=np.float64)
        per_sample_token_count = np.zeros(N, dtype=np.float64)

        for i in tqdm(range(num_batches), desc="Evaluating perplexity", disable=False):
            s = i * batch_size
            e = min((i + 1) * batch_size, N)

            input_ids = torch.from_numpy(samples[s:e]).long().to(self.device)
            am = torch.from_numpy(attn_mask[s:e]).long().to(self.device)

            for chunk_start in range(0, input_ids.size(1), self.eval_context_size):
                chunk_end = min(chunk_start + self.eval_context_size, input_ids.size(1))
                ids = input_ids[:, chunk_start:chunk_end]
                am_chunk = am[:, chunk_start:chunk_end]

                logits = model(ids, attention_mask=am_chunk).logits  # (B, L, V)
                targets = ids[:, 1:]
                logits_pred = logits[:, :-1, :]
                # NLL = logsumexp(logits) - logits[target]
                gathered = logits_pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                log_norm = torch.logsumexp(logits_pred, dim=-1)
                nlls = (log_norm - gathered).float()  # (B, L-1)

                is_eos = (ids == eos_token_id)
                first_eos = (torch.cumsum(is_eos.long(), dim=-1) == 1)
                token_mask = (ids != eos_token_id)
                valid_tokens = (first_eos[:, 1:] | token_mask[:, 1:]).float()

                weighted = (nlls * valid_tokens).detach().cpu().numpy()
                vt = valid_tokens.detach().cpu().numpy()
                per_sample_nll_sum[s:e] += weighted.sum(axis=-1)
                per_sample_token_count[s:e] += vt.sum(axis=-1)

        # PPL across all valid tokens.
        total_nll = float(per_sample_nll_sum.sum())
        total_tokens = float(per_sample_token_count.sum())
        ppl = math.exp(total_nll / max(total_tokens, 1e-8))

        # Per-sample PPL (NaN if no tokens).
        with np.errstate(divide="ignore", invalid="ignore"):
            per_sample_ppl = np.exp(per_sample_nll_sum / np.maximum(per_sample_token_count, 1e-8))
        per_sample_ppl = np.where(per_sample_token_count > 0, per_sample_ppl, np.nan).tolist()

        # Unigram entropy per sample (over tokens up to the first padded position).
        per_sample_entropy = []
        for i in range(N):
            valid_len = int(attn_mask[i].sum())
            toks = samples[i, :valid_len]
            if valid_len == 0:
                per_sample_entropy.append(0.0)
                continue
            _, counts = np.unique(toks, return_counts=True)
            probs = counts.astype(np.float64) / counts.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            per_sample_entropy.append(entropy)
        mean_entropy = sum(per_sample_entropy) / max(1, len(per_sample_entropy))

        return {
            "ppl": ppl,
            "per_sample_ppl": per_sample_ppl,
            "mean_entropy": mean_entropy,
        }

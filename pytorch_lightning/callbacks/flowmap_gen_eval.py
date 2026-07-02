"""Per-epoch flowmap gen-PPL + unigram-entropy callback.

Sweeps `num_sampling_steps_list × self_cond_cfg_scales` and logs one
`eval/flowmap/gen_ppl/n{n}_w{w}` + `eval/flowmap/entropy/n{n}_w{w}`
scalar per cell, plus aggregate `eval/flowmap/gen_ppl_best/n{n}` and
`eval/flowmap/best_w/n{n}` summaries.
"""

import json
import os

import lightning as L
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from tqdm import tqdm

from configs.config import Config, SamplingConfig
from utils.generation_utils import (
    build_run_name, dlm_decode_batch, generate_samples, mask_after_eos,
)
from utils.metrics_utils import Metrics as PPLMetrics
from utils.sampling_utils import get_flowmap_time_steps


class PerEpochFlowMapGenEvalCallback(Callback):
    """At every epoch end: sweep (num_steps, self_cond_cfg) for the flowmap sampler."""

    def __init__(self, *, tokenizer, output_dir: str,
                  num_samples: int, num_sampling_steps_list: list,
                  self_cond_cfg_scales: list, tau_kind: str,
                  eval_ppl_model: str, eval_ppl_batch_size: int,
                  eval_ppl_max_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.num_sampling_steps_list = list(num_sampling_steps_list)
        self.self_cond_cfg_scales = list(self_cond_cfg_scales)
        self.tau_kind = tau_kind
        self.eval_ppl_model = eval_ppl_model
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.eval_ppl_max_length = eval_ppl_max_length
        self._ppl_metrics = None
        self._last_fired_step = -1

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if int(getattr(pl_module.cfg, "epoch_eval_step_freq", 0)) > 0:
            return  # step-based eval owns firing
        self._run_eval(trainer, pl_module)

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule,
                            outputs, batch, batch_idx):
        freq = int(getattr(pl_module.cfg, "epoch_eval_step_freq", 0))
        if freq <= 0:
            return
        cur = int(trainer.global_step)
        if cur > 0 and cur % freq == 0 and cur != self._last_fired_step:
            self._last_fired_step = cur
            self._run_eval(trainer, pl_module)

    @torch.no_grad()
    def _run_eval(self, trainer: L.Trainer, pl_module: L.LightningModule):
        cfg: Config = pl_module.cfg
        device = pl_module.device
        rank = trainer.global_rank
        world_size = trainer.world_size

        backup = pl_module._ema.swap_in(pl_module.model)
        try:
            pl_module.model.eval()
            pad_token_id = (self.tokenizer.eos_token_id if cfg.pad_token == "eos"
                            else self.tokenizer.pad_token_id)
            eos_token_id = (self.tokenizer.eos_token_id
                            if self.tokenizer.eos_token_id is not None else 1)
            text_encoder_dim = pl_module.model.text_encoder_dim

            results: dict = {}  # {(n, w): {"gen_ppl": .., "entropy": ..}}
            for n in self.num_sampling_steps_list:
                for w in self.self_cond_cfg_scales:
                    metric = self._run_cell(
                        trainer, pl_module, cfg, device, rank, world_size,
                        text_encoder_dim, pad_token_id, eos_token_id,
                        num_sampling_steps=int(n), self_cond_cfg_scale=float(w),
                    )
                    if rank == 0:
                        ppl, ent = metric
                        results[(int(n), float(w))] = {"gen_ppl": ppl, "entropy": ent}
                        pl_module.log(f"eval/flowmap/gen_ppl/n{int(n)}_w{float(w):.1f}",
                                       float("nan") if ppl is None else float(ppl),
                                       rank_zero_only=True)
                        pl_module.log(f"eval/flowmap/entropy/n{int(n)}_w{float(w):.1f}",
                                       float("nan") if ent is None else float(ent),
                                       rank_zero_only=True)

            if rank == 0:
                self._log_aggregates(pl_module, results)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        finally:
            pl_module._ema.restore(pl_module.model, backup)
            pl_module.model.train()

    def _run_cell(self, trainer, pl_module, cfg, device, rank, world_size,
                  text_encoder_dim, pad_token_id, eos_token_id,
                  *, num_sampling_steps: int, self_cond_cfg_scale: float):
        samples_per_rank_total = (self.num_samples + world_size - 1) // world_size
        per_rank_batch = cfg.global_batch_size // max(1, world_size)
        num_batches = (samples_per_rank_total + per_rank_batch - 1) // per_rank_batch

        sc = SamplingConfig(
            sampling_method="flowmap",
            num_sampling_steps=[num_sampling_steps],
            cfgs=[1],
            self_cond_cfg_scales=[self_cond_cfg_scale],
            time_schedule="logit_normal",
        )
        generated_texts = []
        desc = f"[ep{trainer.current_epoch+1}] fm n={num_sampling_steps} w={self_cond_cfg_scale}"
        for batch_idx in tqdm(range(num_batches), desc=desc, disable=(rank != 0)):
            cur = min(per_rank_batch, samples_per_rank_total - batch_idx * per_rank_batch)
            if cur <= 0:
                break
            seed = (cfg.seed * 1000003 + trainer.current_epoch * 991
                    + batch_idx * 97 + rank + num_sampling_steps * 31
                    + int(self_cond_cfg_scale * 100))
            gen = torch.Generator(device=device).manual_seed(seed)

            t_steps = get_flowmap_time_steps(
                num_sampling_steps, tau_kind=self.tau_kind,
                P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
                device=device, dtype=torch.float32,
            )
            z = torch.randn(
                (cur, cfg.max_length, text_encoder_dim),
                generator=gen, device=device,
            ) * cfg.denoiser_noise_scale

            latent = generate_samples(
                pl_module.model, z, t_steps,
                cond_seq=None, cond_seq_mask=None,
                config=cfg, sampling_config=sc,
                cfg_scale=1.0, self_cond_cfg_scale=self_cond_cfg_scale,
                generator=gen,
            )
            predicted_ids = dlm_decode_batch(
                pl_module.model, latent, config=cfg,
                self_cond_cfg_scale=self_cond_cfg_scale,
                t_final_val=float(t_steps[-1].item()),
            )
            if getattr(cfg, "mask_after_eos", True):
                predicted_ids = mask_after_eos(
                    predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
                )
            gathered = self._all_gather_ids(predicted_ids, world_size)
            for row in gathered:
                text = self.tokenizer.decode(row.cpu().tolist(), skip_special_tokens=True)
                generated_texts.append(text)

        if rank != 0:
            return (None, None)

        # Persist + PPL on rank 0.
        run_name = build_run_name(
            "flowmap", num_sampling_steps, 1.0, self_cond_cfg_scale,
            "logit_normal", 0.0, suffix="uncond_epoch",
        )
        out_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"epoch_{trainer.current_epoch+1:03d}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(generated_texts):
                f.write(json.dumps({"id": i, "generated": t}, ensure_ascii=False) + "\n")

        nonempty = [s for s in generated_texts if isinstance(s, str) and s.strip()]
        if not nonempty:
            return (None, None)
        if self._ppl_metrics is None:
            self._ppl_metrics = PPLMetrics(
                gen_ppl_eval_model_name_or_path=self.eval_ppl_model,
                eval_ppl_batch_size=self.eval_ppl_batch_size,
                eval_context_size=self.eval_ppl_max_length,
                device=str(pl_module.device),
            )
        res = self._ppl_metrics.record_generative_perplexity(
            text_samples=nonempty, max_length=self.eval_ppl_max_length,
            retokenize=True,
        )
        with open(os.path.join(out_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": trainer.current_epoch + 1,
                "step": trainer.global_step,
                "gen_ppl": float(res["ppl"]),
                "sample_entropy": float(res["mean_entropy"]),
                "num_sampling_steps": num_sampling_steps,
                "self_cond_cfg_scale": self_cond_cfg_scale,
            }, ensure_ascii=False) + "\n")
        return (float(res["ppl"]), float(res["mean_entropy"]))

    @staticmethod
    def _log_aggregates(pl_module, results: dict) -> None:
        if not results:
            return
        ns = sorted({n for (n, _) in results.keys()})
        best_overall = None
        for n in ns:
            ppls = {w: results[(n, w)]["gen_ppl"]
                    for (m, w) in results.keys() if m == n
                    and results[(n, w)]["gen_ppl"] is not None}
            if not ppls:
                continue
            best_w = min(ppls, key=ppls.get)
            pl_module.log(f"eval/flowmap/gen_ppl_best/n{n}", ppls[best_w],
                           rank_zero_only=True)
            pl_module.log(f"eval/flowmap/best_w/n{n}", best_w, rank_zero_only=True)
            if best_overall is None or ppls[best_w] < best_overall:
                best_overall = ppls[best_w]
        if best_overall is not None:
            pl_module.log("eval/flowmap/gen_ppl_best_overall", best_overall,
                           rank_zero_only=True)

    @staticmethod
    def _all_gather_ids(local: torch.Tensor, world_size: int) -> torch.Tensor:
        if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
            return local
        sizes = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(sizes, local)
        return torch.cat(sizes, dim=0)

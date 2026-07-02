"""DataLoader helpers. Lightning owns the DistributedSampler; we expose plain
DataLoaders."""

import json
from typing import Optional

import numpy as np
import torch
from datasets import DatasetDict, load_dataset as hf_load_dataset, load_from_disk
from torch.utils.data import DataLoader

from utils.encoder_utils import build_self_attn_cond_masks
from utils.logging_utils import log_for_0


def get_pad_token_id(tokenizer, pad_token: str = "pad") -> int:
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
    return token_id


def _pad_and_truncate(ids_list, target_len: int, pad_token_id: int):
    padded, lengths = [], []
    for ids in ids_list:
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate([ids, np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)])
        padded.append(ids); lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


def make_collate_fn(*, max_seq_length: int, pad_token_id: int,
                    max_input_seq_length: Optional[int] = None):
    def collate(batch_list):
        input_ids_list = [np.asarray(item["input_ids"]) for item in batch_list]
        if "condition_input_ids" in batch_list[0]:
            seq_list, cond_lens = [], []
            for item in batch_list:
                cond = np.asarray(item["condition_input_ids"])[:max_input_seq_length]
                inp = np.asarray(item["input_ids"])
                seq_list.append(np.concatenate([cond, inp])); cond_lens.append(len(cond))
            cond_lens = np.asarray(cond_lens)
        else:
            seq_list = input_ids_list
            cond_lens = np.zeros(len(input_ids_list), dtype=np.int32)
        ids, total_lens = _pad_and_truncate(seq_list, max_seq_length, pad_token_id)
        pos = np.arange(max_seq_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        encoder_attn, attn, pred = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
        result = {
            "input_ids": torch.from_numpy(ids).long(),
            "encoder_attention_mask": torch.from_numpy(encoder_attn).float(),
            "attention_mask": torch.from_numpy(attn).float(),
            "cond_seq_mask": torch.from_numpy(pred).float(),
        }
        for key in ("index", "input", "target"):
            if key in batch_list[0]:
                result[key] = [item[key] for item in batch_list]
        return result
    return collate


def make_dataloader(
    dataset, *, batch_size: int, shuffle: bool = True, max_seq_length: int = 512,
    pad_token_id: int = 0, max_input_seq_length: Optional[int] = None,
    num_workers: int = 8, prefetch_factor: int = 4,
    pin_memory: bool = True, persistent_workers: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    collate = make_collate_fn(
        max_seq_length=max_seq_length, pad_token_id=pad_token_id,
        max_input_seq_length=max_input_seq_length,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
    )


def load_jsonl_dataset(path, tokenizer, input_key="input", output_key="output"):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            examples.append({
                "index": i, "input": data[input_key], "target": data[output_key],
                "condition_input_ids": tokenizer(data[input_key], add_special_tokens=False)["input_ids"],
                "input_ids": tokenizer(data[output_key], add_special_tokens=False)["input_ids"],
            })
    return examples


def _looks_like_save_to_disk_arrow(ds) -> bool:
    return (
        len(ds) == 1
        and any(c.startswith("_") for c in ds.column_names)
        and not any(not c.startswith("_") for c in ds.column_names)
    )


def load_dataset_split(path: str, dataset_cache_dir=None):
    ds = None
    try:
        ds = hf_load_dataset(path, cache_dir=dataset_cache_dir)
    except Exception:
        ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        if len(splits) != 1:
            raise ValueError(f"Expected single split at {path!r}, got {splits}.")
        ds = ds[splits[0]]
    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download
        log_for_0(f"Dataset {path!r} is save_to_disk format; re-downloading.")
        local_dir = snapshot_download(repo_id=path, repo_type="dataset", cache_dir=dataset_cache_dir)
        ds = load_from_disk(local_dir)
        if isinstance(ds, DatasetDict):
            splits = list(ds.keys())
            ds = ds[splits[0]]
    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_dataset(config, dataset_cache_dir=None):
    log_for_0(f"Loading dataset from {config.data_path}...")
    train_dataset = load_dataset_split(config.data_path, dataset_cache_dir)
    log_for_0(f"Train size: {len(train_dataset)}")
    eval_dataset = None
    if config.eval_data_path:
        eval_dataset = load_dataset_split(config.eval_data_path, dataset_cache_dir)
        log_for_0(f"Eval size: {len(eval_dataset)}")
    else:
        log_for_0("No eval dataset")
    return train_dataset, eval_dataset

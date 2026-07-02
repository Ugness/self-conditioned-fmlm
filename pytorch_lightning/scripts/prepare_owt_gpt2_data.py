"""Preprocess OpenWebText for ELF training with the GPT-2 tokenizer.

Uses GPT-2 BPE. GPT-2 has
`bos_token == eos_token == "<|endoftext|>" == 50256`, so each output row is
`[50256] + 1022 packed gpt2 tokens + [50256]` with `attention_mask = ones(1024)`.

Usage:
    python pytorch_lightning/scripts/prepare_owt_gpt2_data.py \\
        --out_dir ../dataset/openwebtext-gpt2-flm \\
        --num_proc 4
"""
import argparse
import functools
import itertools
import os

import torch
import transformers
from datasets import load_dataset, load_from_disk


def _group_texts(examples, block_size, bos, eos):
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    total_length = (total_length // new_block_size) * new_block_size
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        _values.append([bos] + concatenated_examples[i : i + new_block_size] + [eos])
        _attn_masks.append(torch.ones(block_size))
    result["input_ids"] = _values
    result["attention_mask"] = _attn_masks
    return result


def _build_tokenizer():
    """GPT-2 BPE. bos_token == eos_token == '<|endoftext|>' (id 50256)."""
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def _prepare_split(split_name, hf_split, out_dir, hf_cache_dir, num_proc, block_size):
    out_path = os.path.join(out_dir, split_name)
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists.")
        ds = load_from_disk(out_path)
        print(f"       rows={len(ds)} columns={ds.column_names}")
        return

    print(f"[LOAD] openwebtext split={hf_split!r} cache_dir={hf_cache_dir}")
    raw = load_dataset(
        "openwebtext",
        split=hf_split,
        cache_dir=hf_cache_dir,
        streaming=False,
        num_proc=num_proc,
        trust_remote_code=True,
    )

    tokenizer = _build_tokenizer()
    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    BOS = tokenizer.encode(tokenizer.bos_token)[0]
    print(f"[TOKENIZER] gpt2  BOS={BOS}  EOS={EOS}  vocab_size={tokenizer.vocab_size}")
    assert BOS == EOS == 50256, f"Expected BOS == EOS == 50256 for gpt2, got BOS={BOS} EOS={EOS}"

    def preprocess_and_tokenize(example):
        text = example["text"]
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
        return tokens

    print(f"[TOKENIZE] num_proc={num_proc}")
    tokenized = raw.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Tokenizing",
    )
    tokenized = tokenized.remove_columns("text")

    print(f"[GROUP] block_size={block_size}")
    group_texts = functools.partial(_group_texts, block_size=block_size, bos=BOS, eos=EOS)
    chunked = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Grouping",
    )

    print(f"[SAVE] {out_path}  rows={len(chunked)}")
    os.makedirs(out_dir, exist_ok=True)
    chunked.save_to_disk(out_path)
    print(f"[DONE] {split_name}: {len(chunked)} rows × {block_size} tokens.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        default="../dataset/openwebtext-gpt2-flm",
        help="Directory to write {out_dir}/train and {out_dir}/valid",
    )
    ap.add_argument(
        "--hf_cache_dir",
        default=None,
        help="HF datasets cache (default: HF_HOME or ~/.cache/huggingface)",
    )
    ap.add_argument("--num_proc", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=1024)
    args = ap.parse_args()

    if args.hf_cache_dir is None:
        args.hf_cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    print(f"[CONFIG] out_dir={args.out_dir}  hf_cache_dir={args.hf_cache_dir}  num_proc={args.num_proc}")
    os.makedirs(args.out_dir, exist_ok=True)

    _prepare_split(
        split_name="train",
        hf_split="train[:-100000]",
        out_dir=args.out_dir,
        hf_cache_dir=args.hf_cache_dir,
        num_proc=args.num_proc,
        block_size=args.block_size,
    )
    _prepare_split(
        split_name="valid",
        hf_split="train[-100000:]",
        out_dir=args.out_dir,
        hf_cache_dir=args.hf_cache_dir,
        num_proc=args.num_proc,
        block_size=args.block_size,
    )
    print("[ALL DONE]")


if __name__ == "__main__":
    main()

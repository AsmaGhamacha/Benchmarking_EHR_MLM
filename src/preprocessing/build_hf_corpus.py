#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create HF DatasetDict for DAPT from cleaned .txt notes.
- Reads UTF-8 text files from input_dir
- Tokenizes with specified tokenizer (BERT-style or GPT-style)
- Chunks into fixed token windows with optional overlap
- Writes DatasetDict(train/validation) with a 'text' column
"""

import os
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict

from datasets import Dataset, DatasetDict, Features, Value
from transformers import AutoTokenizer

def read_all_texts(input_dir: Path) -> List[str]:
    texts = []
    for p in sorted(input_dir.glob("*.txt")):
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = p.read_text(encoding="latin-1")
        # keep file-level separators to help the model learn section boundaries
        texts.append(txt.strip())
    return texts

def chunk_token_ids(token_ids: List[int], max_len: int, overlap: int) -> List[List[int]]:
    chunks = []
    i = 0
    step = max_len - overlap if max_len > overlap else max_len
    while i < len(token_ids):
        window = token_ids[i:i+max_len]
        if not window: break
        chunks.append(window)
        if len(window) < max_len: break
        i += step
    return chunks

def main():
    ap = argparse.ArgumentParser("Build HF corpus for DAPT")
    ap.add_argument("--input_dir", default="data/processed/cleaned_txt", type=str)
    ap.add_argument("--model_name", required=True, type=str,
                    help="Tokenizer to use, e.g., emilyalsentzer/Bio_ClinicalBERT, microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract, microsoft/BioGPT-Large")
    ap.add_argument("--max_tokens", type=int, default=512)      # 512 for BERT; 1024 for BioGPT if VRAM allows
    ap.add_argument("--overlap", type=int, default=64)          # 64 is a good default
    ap.add_argument("--val_ratio", type=float, default=0.02)    # 2% validation split
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=None)        # if None: data/hf_corpora/<safe_model_name>
    args = ap.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"Input dir not found: {input_dir}"

    # resolve output dir
    safe_model = args.model_name.replace("/", "__")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/hf_corpora/{safe_model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading cleaned texts from: {input_dir}")
    texts = read_all_texts(input_dir)
    print(f"[INFO] Loaded {len(texts)} documents")

    print(f"[INFO] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # concatenate docs and then re-split by token windows for stable sequences
    print("[INFO] Tokenizing and chunking …")
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else None
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None

    all_chunks_str: List[str] = []
    for doc in texts:
        # tokenize per-document to avoid cross-doc leakage if you prefer
        ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_tokens: #SAFEGUARD truncate if longer than model max length
            ids = ids[:args.max_tokens]

        # optional BOS/EOS for decoder-only (BioGPT)
        if bos is not None:
            ids = [bos] + ids
        if eos is not None:
            ids = ids + [eos]

        chunks = chunk_token_ids(ids, args.max_tokens, args.overlap)
        # decode back to text chunks; HF Trainer data collators will re-tokenize anyway with dynamic masking (BERT) or causal (GPT)
        for ch in chunks:
            all_chunks_str.append(tokenizer.decode(ch, skip_special_tokens=True))

    print(f"[INFO] Total chunks: {len(all_chunks_str)}")

    # train/val split
    random.shuffle(all_chunks_str)
    n = len(all_chunks_str)
    n_val = max(1, int(n * args.val_ratio)) if n > 10 else 1
    val_texts = all_chunks_str[:n_val]
    train_texts = all_chunks_str[n_val:]

    features = Features({"text": Value("string")})
    ds_train = Dataset.from_dict({"text": train_texts}, features=features)
    ds_val   = Dataset.from_dict({"text": val_texts},  features=features)
    dset = DatasetDict({"train": ds_train, "validation": ds_val})

    print(dset)
    save_path = out_dir / "dataset"
    dset.save_to_disk(str(save_path))
    # small manifest for convenience
    (out_dir / "manifest.json").write_text(
        f'{{"model_name":"{args.model_name}","max_tokens":{args.max_tokens},"overlap":{args.overlap},"val_ratio":{args.val_ratio},"num_chunks":{len(all_chunks_str)}}}',
        encoding="utf-8"
    )
    print(f"[OK] Saved to: {save_path}")

if __name__ == "__main__":
    main()

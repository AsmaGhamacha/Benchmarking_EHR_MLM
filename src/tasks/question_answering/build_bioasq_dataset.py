#!/usr/bin/env python3
# build_bioasq_dataset.py

import os
import json
import argparse
from pathlib import Path
import random
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict, Features, Value

def normalize_text(s):
    """Standard text normalization (lowercase, strip punctuation, etc)."""
    import re
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def align_answer_span(context: str, answers: List[str]) -> Tuple[int, str]:
    """Try to find one gold alias in context (case-insensitive)."""
    norm_context = normalize_text(context)
    for ans in answers:
        norm_ans = normalize_text(ans)
        start_idx = norm_context.find(norm_ans)
        if start_idx != -1:
            raw_start = context.lower().find(ans.lower())
            if raw_start != -1:
                return raw_start, ans
    return -1, None

def build_dataset(json_path: Path, out_dir: Path, seed=42):
    print(f"🔹 Loading raw BioASQ from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    factoid_data = [q for q in data["questions"] if q["type"] == "factoid"]
    print(f"✅ Found {len(factoid_data)} factoid questions")

    random.seed(seed)
    random.shuffle(factoid_data)

    all_rows = []
    skipped = 0

    for q in factoid_data:
        question = q["body"].strip()
        snippets = q.get("snippets", [])
        if not snippets:
            skipped += 1
            continue

        context = snippets[0]["text"].strip()
        answers = q.get("exact_answer", [])
        if isinstance(answers[0], list):
            answers = answers[0]

        if not answers or not context:
            skipped += 1
            continue

        # Align span (for BERT-style QA training)
        start_char, matched = align_answer_span(context, answers)

        all_rows.append({
            "id": q["id"],
            "question": question,
            "context": context,
            "answers": {"text": answers},
            "answer_text": matched if matched else "",
            "answer_start": start_char if start_char != -1 else -1,
        })

    print(f"🟢 Prepared {len(all_rows)} QA examples")
    print(f"⚠️ Skipped {skipped} due to missing context/answer")

    # Split into train/dev/test
    N = len(all_rows)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)

    d_train = all_rows[:n_train]
    d_val   = all_rows[n_train:n_train + n_val]
    d_test  = all_rows[n_train + n_val:]

    features = Features({
        "id": Value("string"),
        "question": Value("string"),
        "context": Value("string"),
        "answers": {
            "text": [Value("string")]
        },
        "answer_text": Value("string"),
        "answer_start": Value("int32")
    })

    ds = DatasetDict({
        "train": Dataset.from_list(d_train, features=features),
        "validation": Dataset.from_list(d_val, features=features),
        "test": Dataset.from_list(d_test, features=features)
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"📦 Saved HF dataset to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, type=str, help="Path to BioASQ JSON file")
    ap.add_argument("--output_dir", required=True, type=str, help="Where to save the HF DatasetDict")
    args = ap.parse_args()

    build_dataset(Path(args.input_path), Path(args.output_dir))

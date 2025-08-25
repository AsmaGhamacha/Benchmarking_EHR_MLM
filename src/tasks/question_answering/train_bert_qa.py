#!/usr/bin/env python3
# train_bert_qa.py

import os
import time
import json
import argparse
import numpy as np
import collections
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from tqdm import tqdm
from qa_metrics import compute_em_f1


def prepare_features(examples, tokenizer, max_length, doc_stride, is_training=True):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]

    if is_training:
        tokenized_examples.pop("offset_mapping")  # don't keep offsets in training

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_idx = sample_mapping[i]

        answers = examples["answers"][sample_idx]["text"]
        answer_start = examples["answer_start"][sample_idx]

        if answer_start == -1 or not answers:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answer_start
        end_char = start_char + len(answers)

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        start_token = end_token = cls_index
        for idx in range(token_start_index, token_end_index + 1):
            if offsets[idx][0] <= start_char < offsets[idx][1]:
                start_token = idx
            if offsets[idx][0] < end_char <= offsets[idx][1]:
                end_token = idx
                break

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    predictions = collections.OrderedDict()

    for i, example in enumerate(examples):
        offsets = features[i]["offset_mapping"]
        start_logits = all_start_logits[i]
        end_logits = all_end_logits[i]

        start_indexes = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
        end_indexes = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()

        context = example["context"]
        best_answer = ""
        max_score = -float("inf")

        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(offsets) or end_index >= len(offsets):
                    continue
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue

                start_char = offsets[start_index][0]
                end_char = offsets[end_index][1]
                score = start_logits[start_index] + end_logits[end_index]

                if score > max_score:
                    max_score = score
                    best_answer = context[start_char:end_char]

        predictions[example["id"]] = best_answer

    return predictions


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"🔹 Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    print(f"🔹 Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    print("🔹 Preprocessing dataset...")
    tokenized_datasets = dataset.map(
        lambda x: prepare_features(x, tokenizer, args.max_length, args.doc_stride, is_training=True),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )

    print("🔹 Setting up training...")
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=args.fp16,
        logging_dir=os.path.join(args.out_dir, "logs"),
        logging_steps=10
    )

    print("🔹 Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print("🔹 Final evaluation on test set...")
    raw_test_set = dataset["test"]
    tokenized_test = raw_test_set.map(
        lambda x: prepare_features(x, tokenizer, args.max_length, args.doc_stride, is_training=False),
        batched=True,
        remove_columns=raw_test_set.column_names
    )


    raw_predictions = trainer.predict(tokenized_test)
    final_predictions = postprocess_qa_predictions(
        raw_test_set,
        tokenized_test,
        raw_predictions.predictions,
        tokenizer
    )

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_test_set]
# Get HF eval metrics (like loss, runtime, throughput)
    built_in_metrics = trainer.evaluate(tokenized_test)

    # Get EM/F1 from custom QA logic
    qa_metrics = compute_em_f1(final_predictions, references)

    # Combine both into a single dictionary
    all_metrics = {**built_in_metrics, **qa_metrics}

    # Save merged metrics to disk
    metrics_path = Path(args.out_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    duration = time.time() - start_time
    print(f"✅ Done! Training + eval took {duration:.2f} seconds")
    print(f"📊 Metrics saved to: {metrics_path}")
    print(f"🔍 Sample prediction: {list(final_predictions.items())[:3]}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# train_biogpt_qa.py

import os
import time
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from qa_metrics import compute_em_f1

def postprocess_biogpt_predictions(predictions, dataset):
    return {ex["id"]: pred.strip() for ex, pred in zip(dataset, predictions)}

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"🔹 Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    print(f"🔹 Loading BioGPT model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    def preprocess(example):
        prompt = f"question: {example['question']} context: {example['context']}"
        input_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt").input_ids[0]
        return {"input_ids": input_ids, "labels": input_ids}

    print("🔹 Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

    print("🔹 Setting up training...")
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_dir=os.path.join(args.out_dir, "logs"),
        logging_steps=10,
        fp16=args.fp16,
        dataloader_pin_memory=False
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if args.fp16 else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("🔹 Starting training...")
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print("🔹 Final evaluation on test set (manual batch prediction)...")
    test_dataset = dataset["test"]
    eval_dataset = test_dataset.map(preprocess, remove_columns=test_dataset.column_names)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=32)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_preds.extend(decoded)

    final_preds = postprocess_biogpt_predictions(all_preds, test_dataset)
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in test_dataset]

    print("🔹 Computing metrics...")
    qa_metrics = compute_em_f1(final_preds, references)

    built_in_metrics = trainer.evaluate()
    all_metrics = {**built_in_metrics, **qa_metrics}

    metrics_path = Path(args.out_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"✅ Done! Training + eval took {time.time() - start_time:.2f} seconds")
    print(f"📊 Metrics saved to: {metrics_path}")
    print(f"🔍 Sample predictions: {list(final_preds.items())[:3]}")

if __name__ == "__main__":
    main()
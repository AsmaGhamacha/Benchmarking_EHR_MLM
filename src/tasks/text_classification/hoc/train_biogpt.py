# src/tasks/text_classification/hoc/train_biogpt.py

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk
from transformers import GPT2Tokenizer, GPT2Config, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from src.modeling.biogpt_classifier import BioGptForSequenceClassification


def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    roc_auc = roc_auc_score(labels, probs, average="macro")
    pr_auc = average_precision_score(labels, probs, average="macro")

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():
    print("\n Starting BioGPT training on HoC task")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    print(f"\n Step 1: Loading dataset from {args.dataset_path}/dataset")
    ds = load_from_disk(os.path.join(args.dataset_path, "dataset"))

    print(f" Step 2: Loading tokenizer from: {args.model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token  # BioGPT has no pad_token

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=args.max_len)

    print(f" Step 3: Tokenizing dataset")
    ds_tok = ds.map(tokenize_fn, batched=True)
    print(f"    ➤ Tokenized keys: {list(ds_tok['train'].features.keys())}")

    print(f" Step 4: Loading BioGPT classifier model")
    model = BioGptForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=10,
        problem_type="multi_label_classification"
    )

    print(" Step 5: Setting training args")
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.out_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=True,
        save_total_limit=2
    )

    print(" Step 6: Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    print(" Step 7: Launching training... 🧠")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.2f}s ({elapsed / 60:.2f} minutes)")

    print(" Step 8: Saving model and tokenizer")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f" Model saved to {args.out_dir}")


if __name__ == "__main__":
    main()
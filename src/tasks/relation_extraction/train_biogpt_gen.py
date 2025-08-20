#!/usr/bin/env python3
import argparse
import os
import json
import time
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)

def format_examples(data):
    return [
        f"Relation between {ex['head']} and {ex['tail']}: {ex['label']}"
        for ex in data
    ]

def encode_dataset(data, tokenizer, max_length):
    texts = format_examples(data)
    print(f"🔹 Tokenizing {len(texts)} samples...")
    
    # Fix: force generator → list[str]
    tokenized = tokenizer(
        list(tqdm(texts, desc="Tokenizing")),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    
    # Replace padding with -100 to ignore in loss
    labels = tokenized["input_ids"].copy()
    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in example]
        for example in labels
    ]
    tokenized["labels"] = labels
    return Dataset.from_dict(tokenized)



def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--labels_path", required=True)  # not used in generation
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(" Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))

    print(" Loading and encoding datasets...")
    train_data = load_json(args.data_path)
    val_data = load_json(args.val_path)

    train_dataset = encode_dataset(train_data, tokenizer, args.max_length)
    val_dataset = encode_dataset(val_data, tokenizer, args.max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    print(" Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        fp16=args.fp16,
        report_to="none",
        dataloader_pin_memory=False,
    )

    print(" Starting training...")
    epoch_start = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    epoch_end = time.time()

    print(f" Training complete in {(epoch_end - start)/60:.2f} min")
    print(f" Time spent in training loop only: {(epoch_end - epoch_start)/60:.2f} min")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

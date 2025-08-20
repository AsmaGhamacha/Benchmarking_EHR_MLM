#!/usr/bin/env python3
import os
import json
import argparse
import time
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
from src.tasks.relation_extraction.dataset import load_chemprot_for_bert, get_chemprot_labels
from src.tasks.relation_extraction.metrics import compute_metrics


def encode_dataset(data, tokenizer, label_encoder, desc="Encoding"):
    texts = []
    labels = []

    for ex in tqdm(data, desc=desc):
        text = f"{ex['head']} [SEP] {ex['tail']} [SEP] {ex['text']}"
        label = label_encoder.transform([ex["label"]])[0]
        texts.append(text)
        labels.append(label)

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    encodings["labels"] = labels
    return Dataset.from_dict(encodings)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n Step 1: Load tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=len(get_chemprot_labels())
    )

    print("\n Step 2: Load and parse JSON datasets...")
    with open(args.data_path) as f:
        train_data = json.load(f)
    with open(args.val_path) as f:
        val_data = json.load(f)

    print("\n Step 3: Encode labels...")
    label_list = get_chemprot_labels()
    le = LabelEncoder()
    le.fit(label_list)

    print("\nStep 4: Encode training and validation datasets...")
    train_dataset = encode_dataset(train_data, tokenizer, le, desc="Encoding Train")
    val_dataset = encode_dataset(val_data, tokenizer, le, desc="Encoding Validation")

    print("\n Step 5: Set training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        report_to="none",
    )

    print("\n Step 6: Initialize Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("\n Step 7: Start training...")
    trainer.train()

    print("\n Step 8: Save model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n Step 9: Evaluate model on validation set...")
    preds = trainer.predict(val_dataset)

    print("\n Step 10: Save metrics...")

    os.makedirs(os.path.join(args.output_dir, "metrics"), exist_ok=True)

    # Save evaluation metrics (loss, runtime, speed, etc.)
    eval_metrics = trainer.evaluate(val_dataset)
    with open(os.path.join(args.output_dir, "metrics", "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # Save training summary (last epoch)
    if trainer.state.log_history:
        train_metrics = trainer.state.log_history[-1]
        with open(os.path.join(args.output_dir, "metrics", "training_summary.json"), "w") as f:
            json.dump(train_metrics, f, indent=2)

    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4)
    print(report)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTraining complete in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()
# src/tasks/relation_extraction/train_bert.py
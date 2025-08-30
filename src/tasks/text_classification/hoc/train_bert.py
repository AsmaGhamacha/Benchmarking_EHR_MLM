import argparse
import os
import time
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch
from tqdm import tqdm


def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)

    try:
        roc_auc = roc_auc_score(labels, probs, average="macro")
    except:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels, probs, average="macro")
    except:
        pr_auc = float("nan")

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


# 🔧 Subclass Trainer to override compute_loss (to cast labels to float)
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # <-- add **kwargs
        labels = inputs.pop("labels").float()  # Cast labels to float for BCE loss
        outputs = model(**inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    print("🧪 Starting HoC training pipeline...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    print(f"📂 Step 1: Loading dataset from {args.dataset_path}/dataset")
    ds = load_from_disk(os.path.join(args.dataset_path, "dataset"))

    print(f"🔤 Step 2: Loading tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print(f"✂️ Step 3: Tokenizing dataset with max length {args.max_len}")
    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, max_length=args.max_len)

    ds_tok = ds.map(tokenize_fn, batched=True)
    print(f"✅ Tokenization complete. Sample keys: {list(ds_tok['train'].features.keys())}")

    print(f"🧠 Step 4: Loading model with multi-label classification head")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=10,
        problem_type="multi_label_classification"
    )

    print(f"⚙️ Step 5: Setting up training arguments")
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

    print(f"🚀 Step 6: Initializing Trainer")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    print("🏁 Step 7: Starting training… (this may take a while)")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"⏱️ Training time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    print("💾 Step 8: Saving model and tokenizer")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"✅ Training complete. Model saved to {args.out_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--fp16", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    args = ap.parse_args()

    ds = load_from_disk(args.dataset_path)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)

    def tok_fn(ex): return tok(ex["text"], truncation=True, max_length=512)
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=2000,
        save_steps=2000,
        logging_steps=200,
        save_total_limit=3,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, data_collator=collator,
                      train_dataset=ds_tok["train"], eval_dataset=ds_tok["validation"], tokenizer=tok)
    trainer.train()
    trainer.save_model(args.out_dir)

if __name__ == "__main__":
    main()
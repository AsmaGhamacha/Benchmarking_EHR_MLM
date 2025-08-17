#!/usr/bin/env python3
import argparse
import os
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bsz", type=int, default=4)  # Smaller batch size for GPT
    ap.add_argument("--grad_accum", type=int, default=4)  # Higher accumulation
    ap.add_argument("--fp16", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--warmup_steps", type=int, default=500)
    args = ap.parse_args()

    print(f"🔹 Loading dataset from: {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    print(f"🔹 Dataset contains: {len(ds['train'])} training samples, {len(ds['validation'])} validation samples")
    
    print(f"🔹 Loading BioGPT model and tokenizer: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Set pad token for GPT models
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # For causal LM, we don't use MLM (no masking)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok, 
        mlm=False,  # Key difference: no masking for causal LM
    )

    def tok_fn(ex):
        # Add EOS token for proper causal LM training
        texts = [text + tok.eos_token for text in ex["text"]]
        return tok(texts, truncation=True, max_length=args.max_length, padding=False)

    print("🔹 Tokenizing dataset...")
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["text"])

    total_steps = (len(ds_tok["train"]) // (args.bsz * args.grad_accum)) * args.epochs
    print(f"🔹 Total training steps estimated: {total_steps}")

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=125  ,
        save_steps=250,
        logging_steps=100,
        logging_dir=os.path.join(args.out_dir, "logs"),  
        report_to="tensorboard",                         
        save_total_limit=3,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        # Important for causal LM
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok
    )

    print(f"Starting BioGPT training for {args.epochs} epochs...")
    print(f"🔹 Batch size: {args.bsz} (grad_accum: {args.grad_accum})")
    print(f"🔹 Learning rate: {args.lr}")
    print(f"🔹 Max length: {args.max_length}")
    
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"BioGPT model and tokenizer saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
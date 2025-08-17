#!/usr/bin/env python3
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_from_disk
import math
import argparse

def compute_perplexity(model_name, dataset_path):
    print(f"Evaluating: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    dataset = load_from_disk(dataset_path)
    val_dataset = dataset['validation']

    total_loss = 0.0
    total_tokens = 0

    for item in val_dataset:
        inputs = tokenizer(item['text'], return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f" Perplexity: {perplexity:.4f} | Avg Loss: {avg_loss:.6f}")
    return perplexity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dapt_model", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()

    print("\n=== Intrinsic Evaluation: Base vs DAPT ===")
    base_perplexity = compute_perplexity(args.base_model, args.dataset_path)
    dapt_perplexity = compute_perplexity(args.dapt_model, args.dataset_path)

    delta = base_perplexity - dapt_perplexity
    print(f" Δ Perplexity improvement: {delta:.4f}")

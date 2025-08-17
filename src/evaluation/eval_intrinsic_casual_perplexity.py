#!/usr/bin/env python3
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def compute_perplexity(model_name_or_path, dataset_path):
    print(f"\nEvaluating: {model_name_or_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    dataset = load_from_disk(dataset_path)["validation"]

    losses = []

    for example in tqdm(dataset):
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()
            losses.append(loss)

    avg_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dapt_model", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()

    print("\n=== Intrinsic Evaluation (Causal Model): Base vs DAPT ===")
    
    base_perplexity, base_loss = compute_perplexity(args.base_model, args.dataset_path)
    dapt_perplexity, dapt_loss = compute_perplexity(args.dapt_model, args.dataset_path)

    print(f"\nBase Perplexity: {base_perplexity:.4f} | Avg Loss: {base_loss:.6f}")
    print(f"DAPT Perplexity: {dapt_perplexity:.4f} | Avg Loss: {dapt_loss:.6f}")
    print(f"\u0394 Perplexity improvement: {base_perplexity - dapt_perplexity:.4f}\n")


if __name__ == "__main__":
    main()
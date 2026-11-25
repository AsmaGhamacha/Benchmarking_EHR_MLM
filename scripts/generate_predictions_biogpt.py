import argparse
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_data(path):
    with open(path) as f:
        return json.load(f)

def format_prompt(head, tail):
    return f"Relation between {head} and {tail}:"

def generate_predictions(model, tokenizer, data, max_length=64, device="cuda"):
    model.to(device)
    model.eval()

    predictions = []
    for example in tqdm(data, desc="Generating predictions"):
        prompt = format_prompt(example['head'], example['tail'])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        label_part = decoded.replace(prompt, "").strip()

        predictions.append({
            "head": example["head"],
            "tail": example["tail"],
            "label": example["label"],
            "prediction": label_part
        })
    return predictions

def save_predictions(preds, output_path):
    with open(output_path, "w") as f:
        json.dump(preds, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)

    print("Loading validation data...")
    val_data = load_data(args.val_path)

    preds = generate_predictions(model, tokenizer, val_data, max_length=args.max_length)

    print(f"Saving predictions to {args.output_path}...")
    save_predictions(preds, args.output_path)
    print("Done ")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
import json
from datasets import load_dataset

os.makedirs("data/chemprot_json", exist_ok=True)

def load_chemprot_for_bert(split="train"):
    ds = load_dataset("bigbio/chemprot", name="chemprot_bigbio_kb", split=split)
    processed = []
    for example in ds:
        text = example["passages"][0]["text"][0]  # full sentence

        # Build ID → entity text lookup
        entity_map = {e["id"]: e["text"][0] for e in example["entities"]}

        for rel in example["relations"]:
            head_id = rel["arg1_id"]
            tail_id = rel["arg2_id"]
            if head_id in entity_map and tail_id in entity_map:
                processed.append({
                    "text": text,
                    "head": entity_map[head_id],
                    "tail": entity_map[tail_id],
                    "label": rel["type"]
                })
    return processed


def get_chemprot_labels():
    train_set = load_chemprot_for_bert("train")
    return sorted(set(ex["label"] for ex in train_set))


# Save train set
with open("data/chemprot_json/train.json", "w") as f:
    json.dump(load_chemprot_for_bert("train"), f, indent=2)

# Save validation set
with open("data/chemprot_json/validation.json", "w") as f:
    json.dump(load_chemprot_for_bert("validation"), f, indent=2)

# Save label list
with open("data/chemprot_json/labels.txt", "w") as f:
    for label in get_chemprot_labels():
        f.write(label + "\n")

print("Saved train.json, validation.json, and labels.txt in data/chemprot_json/")
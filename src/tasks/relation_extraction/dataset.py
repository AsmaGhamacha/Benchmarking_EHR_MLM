from datasets import load_dataset

def load_chemprot_for_bert(split="train"):
    ds = load_dataset("bigbio/chemprot", name="chemprot_bigbio_kb", split=split)
    processed = []
    for example in ds:
        for rel in example['relations']:
            processed.append({
                "text": example["passages"][0]["text"][0],
                "head": rel["arg1_id"],
                "tail": rel["arg2_id"],
                "label": rel["type"]
            })
    return processed

def get_chemprot_labels():
    return sorted(set(r["label"] for r in load_chemprot_for_bert("train")))

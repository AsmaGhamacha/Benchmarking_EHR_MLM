import os
import json
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

LABELS = [
    "Agonist", "Antagonist", "Cofactor", "Downregulator", "Modulator",
    "Not", "Part_of", "Regulator", "Substrate", "Undefined", "Upregulator"
]

def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    preds, refs = [], []
    for item in data:
        pred = item.get("prediction", "").strip()
        ref = item.get("reference", "").strip()
        preds.append(pred)
        refs.append(ref)
    return preds, refs

def compute_metrics(preds, refs):
    filtered = [(p, r) for p, r in zip(preds, refs) if p in LABELS and r in LABELS]
    if not filtered:
        return {m: None for m in [
            "eval_accuracy", "eval_macro_f1", "eval_micro_f1", "eval_weighted_f1",
            "eval_macro_precision", "eval_macro_recall",
            "eval_micro_precision", "eval_micro_recall"
        ]} | {
            "num_filtered_out": len(preds),
            "num_total": len(preds)
        }
    filtered_preds, filtered_refs = zip(*filtered)
    return {
        "eval_accuracy": accuracy_score(filtered_refs, filtered_preds),
        "eval_macro_f1": f1_score(filtered_refs, filtered_preds, average="macro"),
        "eval_micro_f1": f1_score(filtered_refs, filtered_preds, average="micro"),
        "eval_weighted_f1": f1_score(filtered_refs, filtered_preds, average="weighted"),
        "eval_macro_precision": precision_score(filtered_refs, filtered_preds, average="macro", zero_division=0),
        "eval_macro_recall": recall_score(filtered_refs, filtered_preds, average="macro", zero_division=0),
        "eval_micro_precision": precision_score(filtered_refs, filtered_preds, average="micro", zero_division=0),
        "eval_micro_recall": recall_score(filtered_refs, filtered_preds, average="micro", zero_division=0),
        "num_filtered_out": len(preds) - len(filtered),
        "num_total": len(preds)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    preds, refs = load_predictions(args.predictions_path)
    metrics = compute_metrics(preds, refs)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to:", args.output_path)

if __name__ == "__main__":
    main()

import os
import json
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Updated label set from your fixed_predictions.json
LABELS = [
    "Agonist", "Antagonist", "Cofactor", "Downregulator", "Modulator",
    "Not", "Part_of", "Regulator", "Substrate", "Undefined", "Upregulator"
]

def extract_predictions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    predictions = []
    references = []

    for ex in data:
        pred_text = ex.get("prediction", "").strip()
        ref = ex.get("reference", "").strip()

        # Try to match any label inside the prediction string
        matched_label = None
        for label in LABELS:
            if label.lower() in pred_text.lower():  # case-insensitive match
                matched_label = label
                break

        # If no label found, set to UNKNOWN
        predictions.append(matched_label if matched_label else "UNKNOWN")
        references.append(ref)

    return predictions, references


def compute_metrics(preds, refs):
    # Keep only examples with known valid labels
    filtered = [(p, r) for p, r in zip(preds, refs) if p in LABELS and r in LABELS]
    
    if not filtered:
        print("⚠️ No valid predictions found.")
        return {
            "eval_accuracy": None,
            "eval_macro_f1": None,
            "eval_micro_f1": None,
            "eval_weighted_f1": None,
            "eval_macro_precision": None,
            "eval_macro_recall": None,
            "eval_micro_precision": None,
            "eval_micro_recall": None,
            "num_filtered_out": len(preds),
            "num_total": len(preds),
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
        "num_total": len(preds),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True, help="Path to fixed predictions JSON")
    parser.add_argument("--output_path", required=True, help="Where to save the evaluation results")
    args = parser.parse_args()

    preds, refs = extract_predictions(args.predictions_path)
    metrics = compute_metrics(preds, refs)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation complete.")
    print(" Saved metrics to:", args.output_path)

if __name__ == "__main__":
    main()

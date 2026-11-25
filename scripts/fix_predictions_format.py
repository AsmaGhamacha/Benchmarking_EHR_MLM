import json
import argparse
import os

# Define ChemProt labels
LABELS = [
    "Agonist", "Antagonist", "Cofactor", "Downregulator", "Modulator",
    "Not", "Part_of", "Regulator", "Substrate", "Undefined", "Upregulator"
]

def extract_label_from_text(text):
    for label in LABELS:
        if label.lower() in text.lower():
            return label
    return "UNKNOWN"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to generated_predictions.json")
    parser.add_argument("--output_path", required=True, help="Path to save fixed_predictions.json")
    args = parser.parse_args()

    # Load raw predictions
    with open(args.input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    formatted = []
    for entry in raw_data:
        raw_pred = entry.get("prediction", "").strip()
        ref = entry.get("label", "").strip()

        if raw_pred and ref:
            # Extract clean prediction label
            clean_pred = extract_label_from_text(raw_pred)
            formatted.append({
                "prediction": clean_pred,
                "reference": ref
            })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2)

    print(f"Fixed {len(formatted)} prediction entries.")
    print(f"Saved to: {args.output_path}")

if __name__ == "__main__":
    main()
import os
import json
from datasets import Dataset, DatasetDict, load_dataset, Features, Value
from sklearn.model_selection import StratifiedKFold
import numpy as np
from pathlib import Path
from datasets import Sequence

# Define the 10 top-level hallmarks (ordered)
TOP10 = [
    "angiogenesis",
    "apoptosis",
    "cellular_energetics",
    "cell_proliferation",
    "genome_instability",
    "growth_suppressors",
    "immune",
    "invasion_metastasis",
    "replicative_imortality",
    "tumor_promoting_inflammation"
]

LABEL2IDX = {label: i for i, label in enumerate(TOP10)}

BIGBIO_LABEL_NAMES = [
    'evading growth suppressors',
    'tumor promoting inflammation',
    'enabling replicative immortality',
    'cellular energetics',
    'resisting cell death',
    'activating invasion and metastasis',
    'genomic instability and mutation',
    'none',
    'inducing angiogenesis',
    'sustaining proliferative signaling',
    'avoiding immune destruction'
]

BIGBIO2TOP10 = {
    0: "growth_suppressors",
    1: "tumor_promoting_inflammation",
    2: "replicative_imortality",
    3: "cellular_energetics",
    4: "apoptosis",
    5: "invasion_metastasis",
    6: "genome_instability",
    8: "angiogenesis",
    9: "cell_proliferation",
    10: "immune"
    # 7 = 'none' is excluded
}

def encode_labels(label_ids):
    vec = [0] * len(TOP10)
    for idx in label_ids:
        if idx in BIGBIO2TOP10:
            mapped = BIGBIO2TOP10[idx]
            vec[LABEL2IDX[mapped]] = 1
    return vec

def main():
    # Load the raw HoC dataset
    raw = load_dataset("bigbio/hallmarks_of_cancer", "hallmarks_of_cancer_source")
    ds = raw["train"]  # All data is under 'train'

    # Extract text and mapped labels
    data = []
    for ex in ds:
        text = ex["text"]
        label_ids = ex["label"]  # Corrected field name
        vec = encode_labels(label_ids)
        if sum(vec) == 0:
            continue  # skip examples with no mapped top-10 labels
        data.append({"text": text, "labels": vec})

    print(f"✔ Loaded {len(data)} examples with at least one top-10 label")

    # Create HuggingFace Dataset
    features = Features({
        "text": Value("string"),
        "labels": Sequence(Value("int32"))
    })
    full_dataset = Dataset.from_list(data).cast(features)

    # 5-fold stratified CV (flattened label indices)
    print("✔ Generating 5 stratified folds...")
    multilabel_y = np.array([ex["labels"] for ex in data])
    y_counts = multilabel_y.sum(axis=1)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(multilabel_y, y_counts)):
        splits.append({"fold": fold, "train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()})

    # Save splits to disk
    out_dir = Path("data/hoc")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits_5fold.json").write_text(json.dumps(splits, indent=2))

    # Optionally save a default DatasetDict (e.g., fold 0)
    train_idx = splits[0]["train_idx"]
    val_idx = splits[0]["val_idx"]
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_data).cast(features),
        "validation": Dataset.from_list(val_data).cast(features)
    })
    ds_dict.save_to_disk(str(out_dir / "dataset"))
    print(f"✔ Saved DatasetDict and splits to: {out_dir}")

if __name__ == "__main__":
    main()
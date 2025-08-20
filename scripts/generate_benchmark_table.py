#!/usr/bin/env python3
import os
import json
import pandas as pd

def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def collect_metrics(base_dir):
    rows = []
    for model_name in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_name, "metrics")
        training_path = os.path.join(model_path, "training_summary.json")
        eval_path = os.path.join(model_path, "eval_metrics.json")

        training_metrics = load_json_if_exists(training_path)
        eval_metrics = load_json_if_exists(eval_path)

        row = {"model": model_name}

        # Add training metrics
        for key in ["train_loss", "train_runtime", "train_samples_per_second", "train_steps_per_second", "epoch", "step"]:
            row[key] = training_metrics.get(key)

        # Add evaluation metrics
        for key in ["eval_loss", "eval_accuracy", "eval_macro_f1", "eval_micro_f1", "eval_weighted_f1",
                    "eval_macro_precision", "eval_macro_recall", "eval_micro_precision", "eval_micro_recall", "eval_runtime"]:
            row[key] = eval_metrics.get(key)

        rows.append(row)
    return rows

def main():
    base_dir = "reports/relation_extraction"
    output_file = os.path.join(base_dir, "benchmarking", "benchmark_results.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f" Collecting metrics from: {base_dir}")
    rows = collect_metrics(base_dir)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f" Benchmark table saved to: {output_file}")

if __name__ == "__main__":
    main()

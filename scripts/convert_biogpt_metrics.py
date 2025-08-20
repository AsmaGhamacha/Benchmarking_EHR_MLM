import os
import json

def convert_metrics(model_dir):
    metrics_path = os.path.join(model_dir, "training_metrics.json")
    output_dir = os.path.join(model_dir, "metrics")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_summary.json")

    if not os.path.exists(metrics_path):
        print(f" {metrics_path} not found.")
        return

    with open(metrics_path, "r") as f:
        data = json.load(f)

    converted = {
        "train_loss": data.get("train_loss", None),
        "train_runtime": round(data.get("train_runtime_sec", data.get("train_runtime", 0)), 2),
        "train_samples_per_second": data.get("train_samples_per_second", None),
        "train_steps_per_second": data.get("train_steps_per_second", None),
        "epoch": data.get("epoch", None),
        "step": data.get("step", None)
    }

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f" Converted metrics saved to: {output_path}")


if __name__ == "__main__":
    BASE_DIR = "reports/relation_extraction"
    biogpt_models = ["biogpt_base", "biogpt_dapt"]

    for model in biogpt_models:
        model_path = os.path.join(BASE_DIR, model)
        convert_metrics(model_path)

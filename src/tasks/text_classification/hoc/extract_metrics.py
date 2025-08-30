import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Base model path
BASE_DIR = "models/hoc"
RUNS = ["clinicalbert_base", "clinicalbert_dapt", "pubmedbert_base", "pubmedbert_dapt", "biogpt_base", "biogpt_dapt"]

# Correct tag names based on your logs
SCALARS = ["eval/macro_f1", "eval/micro_f1", "eval/roc_auc", "eval/pr_auc"]

all_metrics = []

for run in RUNS:
    logdir = os.path.join(BASE_DIR, run, "logs")
    event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    if not event_files:
        print(f" No event files found for {run}")
        continue

    ea = EventAccumulator(logdir)
    ea.Reload()

    for tag in SCALARS:
        if tag not in ea.Tags()["scalars"]:
            print(f"⚠ {tag} not found in {run}")
            continue
        for event in ea.Scalars(tag):
            all_metrics.append({
                "model": run,
                "step": event.step,
                "metric": tag.replace("eval/", ""),  # optional: simplify names
                "value": event.value
            })

if not all_metrics:
    print(" No metrics found. Check your training logs.")
else:
    df = pd.DataFrame(all_metrics)
    df = df.sort_values(by=["model", "step", "metric"])
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/hoc_metrics.csv", index=False)
    print("Metrics extracted and saved to reports/hoc_metrics.csv")

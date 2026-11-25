from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Input CSV files
FILES = {
    "BioGPT": "figures/data/dapt_biogpt_ep3_logs.csv",
    "PubMedBERT": "figures/data/dapt_pubmedbert_ep5_logs.csv",
    "ClinicalBERT": "figures/data/dapt_clinicalbert_ep5_logs.csv",
}

# Output folder
OUT_DIR = Path("figures/generated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Springer friendly style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

for model_name, file_path in FILES.items():
    df = pd.read_csv(file_path)

    # TensorBoard CSV columns
    x = df["Step"]
    y = df["Value"]

    plt.figure(figsize=(4.8, 3.6))
    plt.plot(x, y, linewidth=2)

    plt.xlabel("Training Step")
    plt.ylabel("Training Loss")
    plt.title(f"{model_name} DAPT Training Loss")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    pdf_path = OUT_DIR / f"{model_name.lower()}_dapt_loss.pdf"
    png_path = OUT_DIR / f"{model_name.lower()}_dapt_loss.png"

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
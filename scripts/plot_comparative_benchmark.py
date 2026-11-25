import matplotlib.pyplot as plt
import numpy as np
import os

models = [
    "ClinicalBERT\n(base)",
    "ClinicalBERT\n(DAPT)",
    "PubMedBERT\n(base)",
    "PubMedBERT\n(DAPT)",
    "BioGPT\n(base)",
    "BioGPT\n(DAPT)"
]

chemprot = [0.4401648243, 0.4660850529, 0.5655931323, 0.5683341239, 0.141, 0.070]
bioasq   = [0.899101, 0.699301, 0.040093, 0.056410, 0.092786, 0.091488]
hoc      = [0.858, 0.879, 0.903, 0.903, 0.822, 0.813]

x = np.arange(len(models))
width = 0.24

plt.figure(figsize=(10, 5.8))

# COLORS
color_chemprot = "#7b6bd6"   # purple
color_bioasq   = "#6aaed6"   # blue/gray
color_hoc      = "#f4a3b5"   # pink

plt.bar(x - width, chemprot, width,
        label="ChemProt (Macro-F1)",
        color=color_chemprot)

plt.bar(x, bioasq, width,
        label="BioASQ (F1 normalized)",
        color=color_bioasq)

plt.bar(x + width, hoc, width,
        label="HoC (Macro-F1)",
        color=color_hoc)

plt.xticks(x, models, rotation=15)
plt.ylabel("Score (0–1 scale)")
plt.ylim(0, 1.0)
plt.title("Comparative Benchmarking – Summary of Model Performance")

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

out_dir = "figures/generated"
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, "comparative_benchmark_summary.pdf")
png_path = os.path.join(out_dir, "comparative_benchmark_summary.png")

plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")

print("Saved to:", pdf_path)

plt.show()
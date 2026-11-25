import time
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
NUM_SAMPLES = 256
MAX_LENGTH = 256
WARMUP_STEPS = 10

MODELS = {
    "clinicalbert_base": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert_base": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "biogpt_base": "microsoft/BioGPT",
}

def benchmark_model(model_name: str, checkpoint: str):
    print(f"\n=== {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModel.from_pretrained(checkpoint)
    model.to(DEVICE)
    model.eval()

    texts = ["Synthetic clinical note example for benchmarking."] * NUM_SAMPLES
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(**{k: v[:BATCH_SIZE] for k, v in enc.items()})
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for i in range(0, NUM_SAMPLES, BATCH_SIZE):
            batch = {k: v[i:i + BATCH_SIZE] for k, v in enc.items()}
            _ = model(**batch)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total_time = end - start
    latency_ms = (total_time / NUM_SAMPLES) * 1000.0
    throughput = NUM_SAMPLES / total_time

    peak_mem_mb = None
    if DEVICE == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"device: {DEVICE}")
    print(f"batch_size: {BATCH_SIZE}, max_length: {MAX_LENGTH}, num_samples: {NUM_SAMPLES}")
    print(f"latency_ms_per_sample: {latency_ms:.2f}")
    print(f"throughput_samples_per_s: {throughput:.2f}")
    if peak_mem_mb is not None:
        print(f"peak_gpu_mem_mb: {peak_mem_mb:.1f}")
    else:
        print("peak_gpu_mem_mb: NA")

def main():
    for name, ckpt in MODELS.items():
        benchmark_model(name, ckpt)

if __name__ == "__main__":
    main()

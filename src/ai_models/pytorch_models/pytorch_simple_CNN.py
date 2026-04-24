import time
import math
import random
import statistics
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 128
NUM_CLASSES = 10
IMAGE_SHAPE = (3, 32, 32)
WARMUP_STEPS = 100
TIMED_STEPS = 200
LR = 1e-3

DEVICE = "cpu"
USE_COMPILE = True  # set True to compare torch.compile
TORCH_NUM_THREADS = 4

NUM_RUNS = 100



# ---- Reproducibility ----
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---- Model Class ----
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --- Helper Functions ----
def summarize_times(times: list[float], batch_size: int) -> dict:
    mean_s = statistics.mean(times)
    median_s = statistics.median(times)
    p95_s = np.percentile(times, 95)
    return {
        "mean_ms": mean_s * 1000,
        "median_ms": median_s * 1000,
        "p95_ms": p95_s * 1000,
        "samples_per_sec": batch_size / mean_s,
    }

# --- inference benchmarking ---
def benchmark_inference(model: nn.Module, x: torch.Tensor) -> dict:
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(x)
        # sync_if_needed(DEVICE)

        times = []
        for _ in range(TIMED_STEPS):
            start = time.perf_counter()
            _ = model(x)
            # sync_if_needed(DEVICE)
            end = time.perf_counter()
            times.append(end - start)

    return summarize_times(times, x.shape[0])

def benchmark_inference_repeated(
    model: nn.Module,
    x: torch.Tensor,
    num_runs: int = NUM_RUNS,
) -> dict:
    """Run the inference benchmark num_runs times and return per-run mean and statistics across runs."""
    per_run_mean_ms = []

    for _ in range(num_runs):
        stats_one_run = benchmark_inference(model, x)
        per_run_mean_ms.append(stats_one_run["mean_ms"])

    per_run_mean_ms = np.array(per_run_mean_ms)
    n = len(per_run_mean_ms)
    mean = per_run_mean_ms.mean()
    std = per_run_mean_ms.std(ddof=1)  # sample std, not population. N-1 denominator for sample std.

    # 95% CI via t-distribution (for small n)
    t_critical = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_critical * std / math.sqrt(n)
    ci_high = mean + t_critical * std / math.sqrt(n)

    return {
        "per_run_mean_ms": per_run_mean_ms,
        "mean_ms": mean,
        "std_ms": std,
        "ci95_low_ms": ci_low,
        "ci95_high_ms": ci_high,
        "n_runs": n,
    }

def plot_benchmark_histogram(
    results: dict,
    title: str = "Inference latency across runs",
    save_path: str | None = None,
) -> None:
    data = results["per_run_mean_ms"]
    mean = results["mean_ms"]
    std = results["std_ms"]
    ci_low = results["ci95_low_ms"]
    ci_high = results["ci95_high_ms"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=20, edgecolor="black", alpha=0.75)

    # Mean line
    ax.axvline(mean, color="red", linestyle="-", linewidth=2,
               label=f"mean = {mean:.3f} ms")
    # +/- 1 std lines
    ax.axvline(mean - std, color="red", linestyle="--", linewidth=1.5,
               label=f"±1 std ({std:.3f} ms)")
    ax.axvline(mean + std, color="red", linestyle="--", linewidth=1.5)
    # 95% CI lines
    ax.axvline(ci_low, color="green", linestyle=":", linewidth=1.5,
               label=f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    ax.axvline(ci_high, color="green", linestyle=":", linewidth=1.5)

    ax.set_xlabel("Mean per-batch inference time (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}  (n = {results['n_runs']} runs)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved histogram to {save_path}")

# --- train-step benchmarking (forward pass + backward pass + update) ---
def benchmark_train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    model.train()

    # Warmup steps
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


    times = []
    for _ in range(TIMED_STEPS):
        start = time.perf_counter()
        
        # zero out gradients
        optimizer.zero_grad(set_to_none=True)
        # forward pass
        logits = model(x)
        # compute loss
        loss = criterion(logits, y)
        # backward pass
        loss.backward()
        # update model parameters
        optimizer.step()
        # sync_if_needed(DEVICE)

        end = time.perf_counter()
        times.append(end - start)

    return summarize_times(times, x.shape[0])

def benchmark_train_step_repeated(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_runs: int = NUM_RUNS,
) -> dict:
    """Run the train-step benchmark num_runs times and return per-run mean and statistics across runs."""
    per_run_mean_ms = []

    for _ in range(num_runs):
        stats_one_run = benchmark_train_step(model, x, y, criterion, optimizer)
        print(stats_one_run['mean_ms'])
        print()
        per_run_mean_ms.append(stats_one_run["mean_ms"])

    per_run_mean_ms = np.array(per_run_mean_ms)
    n = len(per_run_mean_ms)
    mean = per_run_mean_ms.mean()
    std = per_run_mean_ms.std(ddof=1)  # sample std, not population

    # 95% CI via t-distribution (correct for small n)
    t_critical = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_critical * std / math.sqrt(n)
    ci_high = mean + t_critical * std / math.sqrt(n)

    return {
        "per_run_mean_ms": per_run_mean_ms,
        "mean_ms": mean,
        "std_ms": std,
        "ci95_low_ms": ci_low,
        "ci95_high_ms": ci_high,
        "n_runs": n,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    # If --feature is used, args.feature is True; otherwise, it is False
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    USE_COMPILE = args.compile
    # set number of cpu threads for torch 
    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(TORCH_NUM_THREADS)
    
    # set seed for reproducibility
    set_seed(42)

    model = SmallCNN(NUM_CLASSES).to(DEVICE)
    if USE_COMPILE:
        model = torch.compile(model)

    # create random input tensor for inference and training-step benchmarks
    x = torch.randn(BATCH_SIZE, *IMAGE_SHAPE, device=DEVICE)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)

    print(f"Device: {DEVICE}")
    print(f"torch.compile enabled: {USE_COMPILE}")

    # ---- Inference Benchmark ----
    inference_results = benchmark_inference_repeated(model, x, num_runs=NUM_RUNS)
    print(
        f"Inference: {inference_results['mean_ms']:.3f} ± {inference_results['std_ms']:.3f} ms "
        f"(95% CI [{inference_results['ci95_low_ms']:.3f}, {inference_results['ci95_high_ms']:.3f}], "
        f"n={inference_results['n_runs']})"
    )
    inference_results_save_path = './results/pytorch/inference_results_compiled.pkl' if USE_COMPILE else './results/pytorch/inference_results_eager.pkl'
    with open(inference_results_save_path, 'wb') as f:
        pickle.dump(inference_results, f)
    inference_histogram_save_path = './results/pytorch/inference_histogram_compiled.png' if USE_COMPILE else './results/pytorch/inference_histogram_eager.png'
    plot_benchmark_histogram(
        inference_results,
        title="PyTorch compiled inference - mean per-batch time" if USE_COMPILE else "PyTorch eager inference - mean per-batch time",
        save_path=inference_histogram_save_path,
    )

    # ---- Train Step Benchmark ---- 
    model = SmallCNN(NUM_CLASSES).to(DEVICE)
    if USE_COMPILE:
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_step_results = benchmark_train_step_repeated(model, x, y, criterion, optimizer, num_runs=NUM_RUNS)
    print(
        f"Training-step: {train_step_results['mean_ms']:.3f} ± {train_step_results['std_ms']:.3f} ms "
        f"(95% CI [{train_step_results['ci95_low_ms']:.3f}, {train_step_results['ci95_high_ms']:.3f}], "
        f"n={train_step_results['n_runs']})"
    )
    train_step_results_save_path = './results/pytorch/train_step_results_compiled.pkl' if USE_COMPILE else './results/pytorch/train_step_results_eager.pkl'
    with open(train_step_results_save_path, 'wb') as f:
        pickle.dump(train_step_results, f)
    histogram_save_path = './results/pytorch/train_step_histogram_compiled.png' if USE_COMPILE else './results/pytorch/train_step_histogram_eager.png'
    plot_benchmark_histogram(
        train_step_results,
        title="PyTorch compiled training-step" if USE_COMPILE else "PyTorch eager training-step",
        save_path=histogram_save_path,
    )

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import time
import statistics
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats 
import argparse

SAMPLE_HIDDEN_LAYER_DIM = 16
NUM_CLASSES = 2
WARMUP_STEPS = 100
TIMED_STEPS = 200
BATCH_SIZE = 64
NUM_RUNS = 100
USE_COMPILE = True
DEVICE = "cpu"

BUILD_TEST_LAYER_NUMS = [10, 100, 500]
FORWARD_TEST_LAYER_NUMS = [10, 100, 500]


class SimpleNN(nn.Module):
    def __init__(self, base_hidden_layer_dim: int, num_classes: int, num_layers: int) -> None:
        super().__init__()
        self.base_hidden_layer_dim = base_hidden_layer_dim

        layers, layer_size = self.add_layers(num_layers)
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Linear(layer_size[-1][1], num_classes)

    def add_layers(self, num_layers: int) -> tuple[list, list]:
        layer_size = [(self.base_hidden_layer_dim, self.base_hidden_layer_dim)]
        for a in range(num_layers):
            noise = random.randint(0, 2) # add noise to avoid layer fusion
            noise = noise if a % 2 == 1 else -noise
            layer_size.append((layer_size[-1][1],self.base_hidden_layer_dim+noise))

        layers = [nn.Linear(in_size, out_size) for (in_size, out_size) in layer_size]
        return layers, layer_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def summarize_results(times: list[float]) -> dict:
    mean_s = statistics.mean(times)
    median_s = statistics.median(times)
    p95_s = np.percentile(times, 95)

    return {
        "mean_ms": mean_s * 1000,
        "median_ms": median_s * 1000,
        "p95_ms": p95_s * 1000,
        }


def benchmark_build_times(num_layers: int, x: torch.Tensor, num_steps: int=TIMED_STEPS, compile: bool=USE_COMPILE) -> dict:
    times = []
    for _ in range (num_steps):
        model = SimpleNN(SAMPLE_HIDDEN_LAYER_DIM, NUM_CLASSES, num_layers)
        if compile:
            model = torch.compile(model)
        start = time.perf_counter()
        _ = model(x) # time through first pass
        end = time.perf_counter()
        times.append(end-start)

    return summarize_results(times)


def benchmark_forward_times(num_layers: int, x: torch.Tensor, timed_steps: int=TIMED_STEPS, warmup_steps: int=WARMUP_STEPS, compile: bool=USE_COMPILE) -> dict:
    times = []
    model = SimpleNN(SAMPLE_HIDDEN_LAYER_DIM, NUM_CLASSES, num_layers)
    if compile:
        model = torch.compile(model)

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(x)

    with torch.no_grad():
        for _ in range (timed_steps):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append((end-start) * 1000)

    return summarize_results(times)


def summarize_results_repeated(per_run_mean_ms):
    n = len(per_run_mean_ms)
    mean = per_run_mean_ms.mean()
    std = per_run_mean_ms.std(ddof=1)  # sample std, not population. N-1 denominator for sample std.

    # 95% CI via t-distribution (for small n)
    t_critical = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_critical * std / np.sqrt(n)
    ci_high = mean + t_critical * std / np.sqrt(n)

    return {
        "per_run_mean_ms": per_run_mean_ms,
        "mean_ms": mean,
        "std_ms": std,
        "ci95_low_ms": ci_low,
        "ci95_high_ms": ci_high,
        "n_runs": n,
    }


def benchmark_build_times_repeated(num_layers: int, num_steps_per_run: int=TIMED_STEPS, num_runs: int=NUM_RUNS, compile=USE_COMPILE):
    per_run_mean_ms = []

    x = torch.randn(BATCH_SIZE, SAMPLE_HIDDEN_LAYER_DIM)
    for _ in range(num_runs):
        run_results = benchmark_build_times(num_layers, x, num_steps_per_run, compile)
        per_run_mean_ms.append(run_results["mean_ms"])

    per_run_mean_ms = np.array(per_run_mean_ms)
    return summarize_results_repeated(per_run_mean_ms)


def benchmark_forward_times_repeated(num_layers: int, num_steps_per_run: int=TIMED_STEPS, num_warmup_steps_per_run: int=WARMUP_STEPS, num_runs: int=NUM_RUNS, compile=USE_COMPILE):
    per_run_mean_ms = []

    x = torch.randn(BATCH_SIZE, SAMPLE_HIDDEN_LAYER_DIM)
    for _ in range(num_runs):
        run_results = benchmark_forward_times(num_layers, x, num_steps_per_run, num_warmup_steps_per_run, compile)
        per_run_mean_ms.append(run_results["mean_ms"])

    per_run_mean_ms = np.array(per_run_mean_ms)
    return summarize_results_repeated(per_run_mean_ms)


def plot_benchmark_histogram(
    results: dict,
    title: str,
    save_path: str | None = None,
) -> None:
    data = results["per_run_mean_ms"]
    mean = results["mean_ms"]
    std = results["std_ms"]
    ci_low = results["ci95_low_ms"]
    ci_high = results["ci95_high_ms"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=max(len(data)//5, 10), edgecolor="black", alpha=0.75)

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

    ax.set_xlabel(f"Time (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}  (n = {results['n_runs']} runs)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"Saved histogram to {save_path}")

    plt.show()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    set_seed(42)

    parser = argparse.ArgumentParser()
    # If --feature is used, args.feature is True; otherwise, it is False
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    USE_COMPILE = args.compile

    for num_layers in BUILD_TEST_LAYER_NUMS:
        build_results = benchmark_build_times_repeated(num_layers, compile=USE_COMPILE)
        print(f"Build results for {num_layers} layers: ")
        print(build_results)
        tag = "Compiled" if USE_COMPILE else "Eager"
        plot_benchmark_histogram(build_results, f"Pytorch Sequential Average Cold Start Time, {num_layers} Layers" + ", " + tag)

    for num_layers in FORWARD_TEST_LAYER_NUMS:
        forward_results = benchmark_forward_times_repeated(num_layers, compile=USE_COMPILE)
        print(f"Forward results for {num_layers} layers: ")
        print(forward_results)
        tag = "Compiled" if USE_COMPILE else "Eager"
        plot_benchmark_histogram(forward_results, f"Pytorch Sequential Average Steady State Forward Times, {num_layers} Layers" + ", " + tag)


if __name__ == "__main__":
    main()

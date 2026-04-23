import torch
import torch.nn as nn
import time
import statistics
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats 

SAMPLE_HIDDEN_LAYER_DIM = 16
NUM_CLASSES = 2
WARMUP_STEPS = 100
TIMED_STEPS = 200
BATCH_SIZE = 64

TEST_LAYER_NUMS = [10, 100, 1000]


class SimpleNN(nn.Module):
    def __init__(self, hidden_layer_dim: int, num_classes: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_layer_dim = hidden_layer_dim

        layers = self.add_layers(num_layers)
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_layer_dim, num_classes)

    def add_layers(self, num_layers: int) -> list:
        return [nn.Linear(self.hidden_layer_dim,self.hidden_layer_dim) for _ in range(num_layers)]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def summarize_results(times: list[float], num_layers) -> dict:
    times_arr = np.array(times)
    mean = statistics.mean(times)
    std = times_arr.std(ddof=1)  # sample std, not population. N-1 denominator for sample std.
    n = len(times_arr)

    t_critical = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_critical * std / np.sqrt(n)
    ci_high = mean + t_critical * std / np.sqrt(n)

    return {
        "times": times,
        "num_layers": num_layers,
        "mean_ms": mean,
        "std_ms": std,
        "ci95_low_ms": ci_low,
        "ci95_high_ms": ci_high,
        "n_runs": n
        }


def benchmark_build_times(num_layers: int, num_runs: int=TIMED_STEPS) -> dict:
    times = []
    for _ in range (num_runs):
        start = time.perf_counter()
        _ = SimpleNN(SAMPLE_HIDDEN_LAYER_DIM, NUM_CLASSES, num_layers)
        end = time.perf_counter()
        times.append((end-start) * 1000)

    return summarize_results(times, num_layers)


def benchmark_forward_times(num_layers: int, x: torch.Tensor, timed_runs: int=TIMED_STEPS, warmup_runs: int=WARMUP_STEPS) -> dict:
    times = []
    model = SimpleNN(SAMPLE_HIDDEN_LAYER_DIM, NUM_CLASSES, num_layers)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)

    with torch.no_grad():
        for _ in range (timed_runs):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append((end-start) * 1000)

    return summarize_results(times, num_layers)


def plot_benchmark_histogram(
    results: dict,
    title: str,
    save_path: str | None = None,
) -> None:
    data = results["times"]
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

    x = torch.randn(BATCH_SIZE, SAMPLE_HIDDEN_LAYER_DIM)

    for num_layers in TEST_LAYER_NUMS:
        build_results = benchmark_build_times(num_layers)
        print(f"Build results for {num_layers} layers: ")
        print(build_results)

        forward_results = benchmark_forward_times(num_layers, x)
        print(f"Forward results for {num_layers} layers: ")
        print(forward_results)

        plot_benchmark_histogram(build_results, f"Pytorch Sequential Average Build Times, {num_layers} Layers")
        plot_benchmark_histogram(forward_results, f"Pytorch Sequential Average Forward Times, {num_layers} Layers")


if __name__ == "__main__":
    main()

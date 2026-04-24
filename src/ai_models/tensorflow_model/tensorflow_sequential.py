import tensorflow as tf
import random
import numpy as np
import time
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import pickle

SAMPLE_HIDDEN_LAYER_DIM = 16
NUM_CLASSES = 2
WARMUP_STEPS = 20
TIMED_STEPS = 50
BATCH_SIZE = 64
NUM_RUNS = 50
NUM_BUILDS = 100
USE_COMPILE = True

BUILD_TEST_LAYER_NUMS = [10, 100, 500]
FORWARD_TEST_LAYER_NUMS = [10, 100, 500]

def build_sequential_model(num_layers, num_classes: int=NUM_CLASSES) -> tf.keras.Model:
    layers = []
    out_layer_size = [SAMPLE_HIDDEN_LAYER_DIM]
    for a in range(num_layers):
        noise = random.randint(0,2)
        noise = noise if a % 2 == 1 else -noise
        out_layer_size.append(SAMPLE_HIDDEN_LAYER_DIM+noise)
    
    out_layer_size.append(num_classes)
    layers = [tf.keras.layers.Dense(out_size) for out_size in out_layer_size]

    return tf.keras.models.Sequential(layers)


def make_inference_fn(model: tf.keras.Model, use_compile: bool):
    def _infer(x):
        return model(x, training=False)

    if use_compile:
        # jit_compile=True turns on XLA; leaving it False matches default graph mode.
        return tf.function(_infer, jit_compile=True)
    return _infer


def benchmark_cold_start_time(num_layers: int, x: tf.Tensor, compile: bool=USE_COMPILE) -> dict:
    model = build_sequential_model(num_layers, NUM_CLASSES)
    infer_fn = make_inference_fn(model, compile)

    start = time.perf_counter()
    _ = infer_fn(x) # time through first pass
    end = time.perf_counter()

    return (end-start) * 1000


def benchmark_forward_times_avg(infer_fn, x: tf.Tensor, timed_steps: int=TIMED_STEPS) -> dict:
    start = time.perf_counter()
    for _ in range (timed_steps):
        _ = infer_fn(x)
    end = time.perf_counter()

    return ((end-start) / timed_steps) * 1000


def summarize_results_repeated(results):
    n = len(results)
    mean = results.mean()
    std = results.std(ddof=1)  # sample std, not population. N-1 denominator for sample std.

    # 95% CI via t-distribution (for small n)
    t_critical = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_critical * std / np.sqrt(n)
    ci_high = mean + t_critical * std / np.sqrt(n)

    return {
        "results": results,
        "mean_ms": mean,
        "std_ms": std,
        "ci95_low_ms": ci_low,
        "ci95_high_ms": ci_high,
        "n_runs": n,
    }


def benchmark_cold_start_times_repeated(num_layers: int, num_builds: int=NUM_BUILDS, compile=USE_COMPILE):
    per_run_ms = []

    x = tf.random.normal([BATCH_SIZE,SAMPLE_HIDDEN_LAYER_DIM])
    for _ in range(num_builds):
        run_results = benchmark_cold_start_time(num_layers, x, compile)
        per_run_ms.append(run_results)

    per_run_ms = np.array(per_run_ms)
    return summarize_results_repeated(per_run_ms)


def benchmark_forward_times_repeated(num_layers: int, num_steps_per_run: int=TIMED_STEPS, warmup_steps: int=WARMUP_STEPS, num_runs: int=NUM_RUNS, compile=USE_COMPILE):
    per_run_mean_ms = []

    model = build_sequential_model(num_layers, NUM_CLASSES)
    infer_fn = make_inference_fn(model, compile)

    x = tf.random.normal([BATCH_SIZE,SAMPLE_HIDDEN_LAYER_DIM])
    for _ in range(warmup_steps):
        _ = infer_fn(x)

    for _ in range(num_runs):
        run_results = benchmark_forward_times_avg(infer_fn, x, num_steps_per_run)
        per_run_mean_ms.append(run_results)

    per_run_mean_ms = np.array(per_run_mean_ms)
    return summarize_results_repeated(per_run_mean_ms)


def plot_benchmark_histogram(
    results: dict,
    title: str,
    save_path: str | None = None,
) -> None:
    data = results["results"]
    mean = results["mean_ms"]
    std = results["std_ms"]
    ci_low = results["ci95_low_ms"]
    ci_high = results["ci95_high_ms"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=80, edgecolor="black", alpha=0.75)

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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    USE_COMPILE = args.compile

    for num_layers in BUILD_TEST_LAYER_NUMS:
        build_results = benchmark_cold_start_times_repeated(num_layers, compile=USE_COMPILE)
        print(f"Cold start results for {num_layers} layers: ")
        print(build_results)

        dir = './results/tensorflow/cold_start_results/'
        results_save_path = f'cold_start_results_compiled_{num_layers}.pkl' if USE_COMPILE else f'cold_start_results_eager_{num_layers}.pkl'
        with open(dir + results_save_path, 'wb') as f:
            pickle.dump(build_results, f)

        hist_results_save_path = f'cold_start_histogram_compiled_{num_layers}.png' if USE_COMPILE else f'cold_start_histogram_eager_{num_layers}.png'
        tag = "Compiled" if USE_COMPILE else "Eager"
        plot_benchmark_histogram(build_results, 
                                 f"TF Sequential Cold Start Time, {num_layers} Layers" + ", " + tag,
                                 dir + hist_results_save_path)


    for num_layers in FORWARD_TEST_LAYER_NUMS:
        forward_results = benchmark_forward_times_repeated(num_layers, compile=USE_COMPILE)
        print(f"Forward results for {num_layers} layers: ")
        print(forward_results)

        dir = './results/tensorflow/forward_results/'
        results_save_path = f'forward_results_compiled_{num_layers}.pkl' if USE_COMPILE else f'forward_results_eager_{num_layers}.pkl'
        with open(dir + results_save_path, 'wb') as f:
            pickle.dump(forward_results, f)

        hist_results_save_path = f'forward_histogram_compiled_{num_layers}.png' if USE_COMPILE else f'forward_histogram_eager_{num_layers}.png'
        tag = "Compiled" if USE_COMPILE else "Eager"
        plot_benchmark_histogram(forward_results, 
                                 f"TF Sequential Average Forward Time, {num_layers} Layers" + ", " + tag,
                                 dir + hist_results_save_path)
    

if __name__ == "__main__":
    main()

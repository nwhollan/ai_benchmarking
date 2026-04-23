import time
import math
import random
import statistics
import pickle
import argparse
import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Silence some TF logs; set before importing tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 128
NUM_CLASSES = 10
# TF uses channels-last by default: (H, W, C)
IMAGE_SHAPE = (32, 32, 3)
WARMUP_STEPS = 100
TIMED_STEPS = 200
LR = 1e-3

DEVICE = "/CPU:0"
#USE_COMPILE = True  # whether to wrap steps in tf.function (graph mode)
TF_NUM_THREADS = 4

NUM_RUNS = 100


# ---- Reproducibility ----
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---- Model ----
def build_small_cnn(num_classes: int = 10) -> tf.keras.Model:
    """Keras equivalent of the PyTorch SmallCNN.

    PyTorch layout used Conv2d -> ReLU -> MaxPool2d pairs, followed by a
    final Conv2d -> ReLU -> AdaptiveAvgPool2d((1,1)) and a Linear head.
    AdaptiveAvgPool2d((1,1)) is equivalent to GlobalAveragePooling2D in Keras.
    """
    inputs = tf.keras.Input(shape=IMAGE_SHAPE)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same")(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)  # 16x16

    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)  # 8x8

    x = tf.keras.layers.Conv2D(128, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # (B, 128)

    outputs = tf.keras.layers.Dense(num_classes)(x)  # logits
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="SmallCNN")


# --- Helper Functions ----
def summarize_times(times: list, batch_size: int) -> dict:
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
def make_inference_fn(model: tf.keras.Model, use_compile: bool):
    """Return a callable that runs a forward pass.

    If use_compile is True, wrap in tf.function (graph mode, analogous to
    torch.compile in spirit — both trade warmup cost for faster steady-state).
    """
    def _infer(x):
        return model(x, training=False)

    if use_compile:
        # jit_compile=True turns on XLA; leaving it False matches default graph mode.
        return tf.function(_infer, jit_compile=True)
    return _infer


def benchmark_inference(infer_fn, x: tf.Tensor) -> dict:
    # Warmup
    for _ in range(WARMUP_STEPS):
        _ = infer_fn(x)

    times = []
    for _ in range(TIMED_STEPS):
        start = time.perf_counter()
        out = infer_fn(x)
        # Force materialization so we measure real compute, not just op dispatch.
        # .numpy() blocks until the op is done; for CPU this is effectively a sync.
        _ = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        end = time.perf_counter()
        times.append(end - start)

    return summarize_times(times, x.shape[0])


def benchmark_inference_repeated(
    infer_fn,
    x: tf.Tensor,
    num_runs: int = NUM_RUNS,
) -> dict:
    """Run the inference benchmark num_runs times and return per-run mean and statistics across runs."""
    per_run_mean_ms = []

    for _ in range(num_runs):
        stats_one_run = benchmark_inference(infer_fn, x)
        per_run_mean_ms.append(stats_one_run["mean_ms"])

    per_run_mean_ms = np.array(per_run_mean_ms)
    n = len(per_run_mean_ms)
    mean = per_run_mean_ms.mean()
    std = per_run_mean_ms.std(ddof=1)  # sample std, N-1 denominator

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


# --- train-step benchmarking (forward + backward + update) ---
def make_train_step_fn(
    model: tf.keras.Model,
    loss_fn,
    optimizer: tf.keras.optimizers.Optimizer,
    use_compile: bool,
):
    """Build a single train-step function, optionally graph-compiled."""
    def _train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    if use_compile:
        return tf.function(_train_step, jit_compile=True)
    return _train_step


def benchmark_train_step(train_step_fn, x: tf.Tensor, y: tf.Tensor) -> dict:
    # Warmup
    for _ in range(WARMUP_STEPS):
        _ = train_step_fn(x, y)

    times = []
    for _ in range(TIMED_STEPS):
        start = time.perf_counter()
        loss = train_step_fn(x, y)
        # Force sync so we measure actual compute time.
        _ = loss.numpy() if hasattr(loss, "numpy") else float(loss)
        end = time.perf_counter()
        times.append(end - start)

    return summarize_times(times, x.shape[0])


def benchmark_train_step_repeated(
    train_step_fn,
    x: tf.Tensor,
    y: tf.Tensor,
    num_runs: int = NUM_RUNS,
) -> dict:
    """Run the train-step benchmark num_runs times and return per-run mean and statistics across runs."""
    per_run_mean_ms = []

    for _ in range(num_runs):
        stats_one_run = benchmark_train_step(train_step_fn, x, y)
        print(stats_one_run['mean_ms'])
        print()
        per_run_mean_ms.append(stats_one_run["mean_ms"])

    per_run_mean_ms = np.array(per_run_mean_ms)
    n = len(per_run_mean_ms)
    mean = per_run_mean_ms.mean()
    std = per_run_mean_ms.std(ddof=1)  # sample std

    # 95% CI via t-distribution
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
    parser.add_argument("--compile", action="store_true",
                        help="Wrap step functions in tf.function (graph mode).")
    args = parser.parse_args()
    use_compile = args.compile

    # Set CPU thread counts (TF equivalent of torch.set_num_threads /
    # set_num_interop_threads). Must be set before any ops run.
    tf.config.threading.set_intra_op_parallelism_threads(TF_NUM_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(TF_NUM_THREADS)

    # Set visible GPU devices to empty list.
    tf.config.set_visible_devices([], "GPU")

    set_seed(42)

    with tf.device(DEVICE):
        model = build_small_cnn(NUM_CLASSES)

        # Random input tensors for inference and training-step benchmarks.
        x = tf.random.normal((BATCH_SIZE,) + IMAGE_SHAPE)
        y = tf.random.uniform(
            (BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32
        )

        print(f"Device: {DEVICE}")
        print(f"tf.function (graph) enabled: {use_compile}")

        # ---- Inference Benchmark ----
        infer_fn = make_inference_fn(model, use_compile)
        inference_results = benchmark_inference_repeated(infer_fn, x, num_runs=NUM_RUNS)
        print(
            f"Inference: {inference_results['mean_ms']:.3f} ± {inference_results['std_ms']:.3f} ms "
            f"(95% CI [{inference_results['ci95_low_ms']:.3f}, {inference_results['ci95_high_ms']:.3f}], "
            f"n={inference_results['n_runs']})"
        )
        inference_results_save_path = (
            "inference_results_compiled.pkl" if use_compile else "inference_results_eager.pkl"
        )
        with open(inference_results_save_path, "wb") as f:
            pickle.dump(inference_results, f)
        plot_benchmark_histogram(
            inference_results,
            title="TF graph-mode inference - mean per-batch time" if use_compile
            else "TF eager inference - mean per-batch time",
            save_path="inference_histogram.png",
        )

        # ---- Train Step Benchmark ----
        # Rebuild model + optimizer so train benchmark starts fresh, matching the
        # original script.
        model = build_small_cnn(NUM_CLASSES)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        train_step_fn = make_train_step_fn(model, loss_fn, optimizer, use_compile)

        train_step_results = benchmark_train_step_repeated(
            train_step_fn, x, y, num_runs=NUM_RUNS
        )
        print(
            f"Training-step: {train_step_results['mean_ms']:.3f} ± {train_step_results['std_ms']:.3f} ms "
            f"(95% CI [{train_step_results['ci95_low_ms']:.3f}, {train_step_results['ci95_high_ms']:.3f}], "
            f"n={train_step_results['n_runs']})"
        )
        train_step_results_save_path = (
            "./results/tensorflow/train_step_results_compiled.pkl" if use_compile else "./results/tensorflow/train_step_results_eager.pkl"
        )
        with open(train_step_results_save_path, "wb") as f:
            pickle.dump(train_step_results, f)
        histogram_save_path = './results/tensorflow/train_step_histogram_compiled.png' if use_compile else './results/tensorflow/train_step_histogram_eager.png'
        plot_benchmark_histogram(
            train_step_results,
            title="TF graph-mode training-step" if use_compile else "TF eager training-step",
            save_path=histogram_save_path,
        )


if __name__ == "__main__":
    main()
import time
import random
import statistics

import numpy as np
import tensorflow as tf


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 128
NUM_CLASSES = 10
IMAGE_SHAPE = (32, 32, 3)
WARMUP_STEPS = 50
TIMED_STEPS = 200
EPOCHS = 10
LR = 1e-3
TARGET_VAL_ACC = 0.80

# For clean comparison on Mac, use CPU first
USE_CPU_ONLY = True
USE_TF_FUNCTION = False


# -----------------------------
# Device setup
# -----------------------------
if USE_CPU_ONLY:
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Model
# -----------------------------
def make_model(num_classes: int = 10) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=IMAGE_SHAPE)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs, outputs)


# -----------------------------
# Utilities
# -----------------------------
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


def benchmark_inference(model: tf.keras.Model, x: tf.Tensor) -> dict:
    def step(inp):
        return model(inp, training=False)

    if USE_TF_FUNCTION:
        step_fn = tf.function(step)
    else:
        step_fn = step

    for _ in range(WARMUP_STEPS):
        _ = step_fn(x)

    times = []
    for _ in range(TIMED_STEPS):
        start = time.perf_counter()
        _ = step_fn(x)
        end = time.perf_counter()
        times.append(end - start)

    return summarize_times(times, int(x.shape[0]))


def benchmark_train_step(
    model: tf.keras.Model,
    x: tf.Tensor,
    y: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn,
) -> dict:
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            logits = model(inp, training=True)
            loss = loss_fn(target, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    if USE_TF_FUNCTION:
        train_step_fn = tf.function(train_step)
    else:
        train_step_fn = train_step

    for _ in range(WARMUP_STEPS):
        _ = train_step_fn(x, y)

    times = []
    for _ in range(TIMED_STEPS):
        start = time.perf_counter()
        _ = train_step_fn(x, y)
        end = time.perf_counter()
        times.append(end - start)

    return summarize_times(times, int(x.shape[0]))


def get_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = (x_train - 0.5) / 0.5
    x_test = (x_test - 0.5) / 0.5

    y_train = y_train.squeeze().astype("int32")
    y_test = y_test.squeeze().astype("int32")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000, seed=42).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


def evaluate(model: tf.keras.Model, dataset) -> float:
    correct = 0
    total = 0
    for xb, yb in dataset:
        logits = model(xb, training=False)
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, yb), tf.int32)))
        total += int(tf.size(yb))
    return correct / total


def time_to_quality() -> None:
    train_ds, test_ds = get_cifar10()

    model = make_model(NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def train_step(inp, target):
        with tf.GradientTape() as tape:
            logits = model(inp, training=True)
            loss = loss_fn(target, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    if USE_TF_FUNCTION:
        train_step_fn = tf.function(train_step)
    else:
        train_step_fn = train_step

    start = time.perf_counter()
    total_steps = 0

    for epoch in range(1, EPOCHS + 1):
        for xb, yb in train_ds:
            _ = train_step_fn(xb, yb)
            total_steps += 1

        val_acc = evaluate(model, test_ds)
        elapsed = time.perf_counter() - start
        print(f"[epoch {epoch}] val_acc={val_acc:.4f} elapsed_s={elapsed:.2f} steps={total_steps}")

        if val_acc >= TARGET_VAL_ACC:
            print(
                f"Reached target accuracy {TARGET_VAL_ACC:.2%} "
                f"in {elapsed:.2f}s over {total_steps} steps."
            )
            return

    elapsed = time.perf_counter() - start
    print(
        f"Did not reach target accuracy {TARGET_VAL_ACC:.2%} "
        f"within {EPOCHS} epochs. Final elapsed_s={elapsed:.2f}, steps={total_steps}"
    )


def main() -> None:
    set_seed(42)

    model = make_model(NUM_CLASSES)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    x = tf.random.normal((BATCH_SIZE, *IMAGE_SHAPE))
    y = tf.random.uniform((BATCH_SIZE,), minval=0, maxval=NUM_CLASSES, dtype=tf.int32)

    print("CPU only:" if USE_CPU_ONLY else "Default device selection")
    print(f"tf.function enabled: {USE_TF_FUNCTION}")

    inference_stats = benchmark_inference(model, x)
    print("Inference benchmark:", inference_stats)

    model2 = make_model(NUM_CLASSES)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=LR)
    train_stats = benchmark_train_step(model2, x, y, optimizer2, loss_fn)
    print("Training-step benchmark:", train_stats)

    time_to_quality()


if __name__ == "__main__":
    main()
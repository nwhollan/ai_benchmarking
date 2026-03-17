import time
import math
import random
import statistics
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 128
NUM_CLASSES = 10
IMAGE_SHAPE = (3, 32, 32)
WARMUP_STEPS = 50
TIMED_STEPS = 200
EPOCHS = 10
LR = 1e-3
TARGET_VAL_ACC = 0.80

DEVICE = "cpu"
USE_COMPILE = False  # set True to compare torch.compile
TORCH_NUM_THREADS = 4


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------
# Model
# -----------------------------
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


# -----------------------------
# Utilities
# -----------------------------
def sync_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        # MPS timing can still benefit from sync via mps.synchronize when available
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


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


def benchmark_inference(model: nn.Module, x: torch.Tensor) -> dict:
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(x)
        sync_if_needed(DEVICE)

        times = []
        for _ in range(TIMED_STEPS):
            start = time.perf_counter()
            _ = model(x)
            sync_if_needed(DEVICE)
            end = time.perf_counter()
            times.append(end - start)

    return summarize_times(times, x.shape[0])


def benchmark_train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    model.train()

    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    sync_if_needed(DEVICE)

    times = []
    for _ in range(TIMED_STEPS):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        sync_if_needed(DEVICE)
        end = time.perf_counter()
        times.append(end - start)

    return summarize_times(times, x.shape[0])


def get_cifar10_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if DEVICE == "cpu" else 0,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if DEVICE == "cpu" else 0,
    )
    return trainloader, testloader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.numel()
    return correct / total


def time_to_quality() -> None:
    trainloader, testloader = get_cifar10_loaders(BATCH_SIZE)

    model = SmallCNN(NUM_CLASSES).to(DEVICE)
    if USE_COMPILE:
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start = time.perf_counter()
    total_steps = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in trainloader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_steps += 1

        val_acc = evaluate(model, testloader)
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
    torch.set_num_threads(TORCH_NUM_THREADS)

    model = SmallCNN(NUM_CLASSES).to(DEVICE)
    if USE_COMPILE:
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    x = torch.randn(BATCH_SIZE, *IMAGE_SHAPE, device=DEVICE)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=DEVICE)

    print(f"Device: {DEVICE}")
    print(f"torch.compile enabled: {USE_COMPILE}")

    inf_stats = benchmark_inference(model, x)
    print("Inference benchmark:", inf_stats)

    # fresh model for train-step benchmark
    model2 = SmallCNN(NUM_CLASSES).to(DEVICE)
    if USE_COMPILE:
        model2 = torch.compile(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR)
    train_stats = benchmark_train_step(model2, x, y, criterion, optimizer2)
    print("Training-step benchmark:", train_stats)

    time_to_quality()


if __name__ == "__main__":
    main()
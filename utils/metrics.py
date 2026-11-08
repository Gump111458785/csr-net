"""Evaluation metrics suite for NLP classification and calibration benchmarking.
"""

import time
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece * 100.0)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro") * 100.0
    ece = compute_ece(all_probs, all_targets, n_bins=15)

    return {
        "accuracy": round(float(acc), 2),
        "macro_f1": round(float(macro_f1), 2),
        "ece": round(float(ece), 2),
        "total_samples": len(all_targets),
    }


def measure_gpu_efficiency(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int = 64,
    warmup_iters: int = 30,
    measure_iters: int = 100,
) -> Dict[str, float]:
    if device.type != "cuda":
        raise ValueError("CUDA GPU is required for accurate efficiency profiling.")

    model.eval()
    dummy_ids = torch.randint(2, 500, (1, seq_len), device=device)
    dummy_mask = torch.ones(1, seq_len, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_ids, dummy_mask)
        torch.cuda.synchronize()

    # Latency Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(dummy_ids, dummy_mask)
    end_event.record()
    torch.cuda.synchronize()

    avg_latency_ms = start_event.elapsed_time(end_event) / measure_iters

    # Throughput benchmark with batch_size = 64
    batch_size = 64
    batch_ids = torch.randint(2, 500, (batch_size, seq_len), device=device)
    batch_mask = torch.ones(batch_size, seq_len, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch_ids, batch_mask)
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = model(batch_ids, batch_mask)
        torch.cuda.synchronize()
    t1 = time.time()
    throughput = (batch_size * 50) / (t1 - t0)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        "params_m": round(float(total_params), 2),
        "latency_ms": round(float(avg_latency_ms), 2),
        "throughput": round(float(throughput), 1),
    }

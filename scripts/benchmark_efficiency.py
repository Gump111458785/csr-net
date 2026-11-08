"""Hardware efficiency profiling script for Table 2.
Measures Parameters (M), Estimated FLOPs (G), GPU Latency (ms), and Throughput (samples/s).
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.csr_net import build_model
from utils.metrics import measure_gpu_efficiency
from utils.logger import ExperimentLogger


def estimate_flops(model: torch.nn.Module, seq_len=64) -> float:
    total_flops = 0
    def linear_hook(self, input, output):
        batch_size = input[0].size(0)
        weight_ops = self.in_features * self.out_features * 2
        nonlocal total_flops
        total_flops += batch_size * weight_ops

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    dummy_ids = torch.randint(2, 500, (1, seq_len))
    dummy_mask = torch.ones(1, seq_len)
    with torch.no_grad():
        _ = model(dummy_ids, dummy_mask)

    for h in hooks:
        h.remove()
    return round(total_flops / 1e9, 2)


def main():
    parser = argparse.ArgumentParser(description="Profile model efficiency on GPU")
    parser.add_argument("--cpu_debug", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n[CRITICAL] GPU is required for efficiency measurement.\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running hardware profiler on: {device}")

    models = ["textcnn", "bilstm_attn", "transformer", "csr_net"]
    results = []

    for name in models:
        model = build_model(name, vocab_size=10000, num_classes=4)
        flops_g = estimate_flops(model, seq_len=64)
        model = model.to(device)

        if device.type == "cuda":
            eff = measure_gpu_efficiency(model, device, seq_len=64)
            lat = eff["latency_ms"]
            thr = eff["throughput"]
            param = eff["params_m"]
        else:
            param = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
            lat = "N/A (CPU)"
            thr = "N/A (CPU)"

        results.append({
            "Architecture": name.upper(),
            "Params (M)": param,
            "FLOPs (G)": flops_g,
            "Latency (ms)": lat,
            "Throughput (samples/s)": thr,
        })

    ExperimentLogger.print_comparison_table(results, title="Hardware Complexity and Inference Efficiency (Table 2)")


if __name__ == "__main__":
    main()

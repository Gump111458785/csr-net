"""Model evaluation and inference testing script for NLP benchmarks.
Requires GPU by default.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_dataloaders, get_dataset_info, SUPPORTED_DATASETS
from models.csr_net import build_model
from utils.metrics import evaluate_model
from utils.logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLP model on benchmark dataset")
    parser.add_argument("--model", type=str, default="csr_net", choices=["csr_net", "textcnn", "bilstm_attn", "transformer"])
    parser.add_argument("--dataset", type=str, default="sst2", choices=SUPPORTED_DATASETS)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_csr", action="store_true", default=True)
    parser.add_argument("--no_csr", dest="use_csr", action="store_false")
    parser.add_argument("--use_temp_scale", action="store_true", default=True)
    parser.add_argument("--no_temp_scale", dest="use_temp_scale", action="store_false")
    parser.add_argument("--cpu_debug", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n" + "!" * 70, file=sys.stderr)
        print(" [CRITICAL HARDWARE REQUIREMENT] CUDA GPU is required for evaluation.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_dataset_info(args.dataset)

    _, test_loader = get_dataloaders(args.dataset, data_dir=args.data_dir, batch_size=args.batch_size)
    model = build_model(
        args.model,
        vocab_size=cfg["vocab_size"],
        num_classes=cfg["num_classes"],
        use_csr=args.use_csr,
        use_temp_scale=args.use_temp_scale,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    model = model.to(device)
    metrics = evaluate_model(model, test_loader, device)

    print("\n" + "=" * 55)
    print(f" BENCHMARK EVALUATION: {args.model.upper()} on {args.dataset.upper()}")
    print("=" * 55)
    print(f"  • Accuracy:   {metrics['accuracy']:.2f} %")
    print(f"  • Macro-F1:   {metrics['macro_f1']:.2f} %")
    print(f"  • ECE (15-b): {metrics['ece']:.2f} %")
    print(f"  • Samples:    {metrics['total_samples']}")
    print("=" * 55 + "\n")

    logger = ExperimentLogger()
    run_name = f"eval_{args.model}_{args.dataset}"
    logger.log_metrics(run_name, metrics)


if __name__ == "__main__":
    main()

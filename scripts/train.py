"""Training script for NLP classification and calibration experiments.
Requires CUDA GPU by default.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import get_dataloaders, get_dataset_info, SUPPORTED_DATASETS
from models.csr_net import build_model
from utils.loss import MarginConstrainedCalibrationLoss
from utils.metrics import evaluate_model
from utils.logger import ExperimentLogger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Train text classification models on GPU")
    parser.add_argument("--model", type=str, default="csr_net", choices=["csr_net", "textcnn", "bilstm_attn", "transformer"])
    parser.add_argument("--dataset", type=str, default="sst2", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_csr", action="store_true", default=True)
    parser.add_argument("--no_csr", dest="use_csr", action="store_false")
    parser.add_argument("--use_mcc", action="store_true", default=True)
    parser.add_argument("--no_mcc", dest="use_mcc", action="store_false")
    parser.add_argument("--use_temp_scale", action="store_true", default=True)
    parser.add_argument("--no_temp_scale", dest="use_temp_scale", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_debug", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.cpu_debug:
        print("\n" + "!" * 70, file=sys.stderr)
        print(" [CRITICAL HARDWARE REQUIREMENT] ", file=sys.stderr)
        print(" PyTorch CUDA acceleration is mandatory for this experiment.", file=sys.stderr)
        print(" No CUDA GPU detected. Please execute on an NVIDIA GPU environment.", file=sys.stderr)
        print("!" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running on: {device}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = get_dataset_info(args.dataset)
    num_classes = cfg["num_classes"]
    vocab_size = cfg["vocab_size"]

    print(f"[Data] Loading dataset '{args.dataset}' (Classes={num_classes}, Vocab={vocab_size})...")
    train_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    print(f"[Model] Building '{args.model}' (CSR={args.use_csr}, TempScale={args.use_temp_scale})...")
    model = build_model(
        model_name=args.model,
        vocab_size=vocab_size,
        num_classes=num_classes,
        use_csr=args.use_csr,
        use_temp_scale=args.use_temp_scale,
    ).to(device)

    criterion = MarginConstrainedCalibrationLoss(smoothing=0.05, cal_weight=0.15, use_mcc=args.use_mcc).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    run_name = f"{args.model}_{args.dataset}_seed{args.seed}"
    best_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, f"{run_name}_best.pth")

    print(f"\n--- Starting Training: {run_name} ({args.epochs} epochs) ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        test_metrics = evaluate_model(model, test_loader, device)
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {avg_loss:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.2f}% | "
            f"Test ECE: {test_metrics['ece']:.2f}%"
        )

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_metrics": test_metrics,
            }, best_ckpt_path)

    print("\nTraining complete. Evaluating best checkpoint...")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    final_metrics = evaluate_model(model, test_loader, device)
    print("=" * 60)
    print(f" FINAL BENCHMARK METRICS FOR: {run_name}")
    print(f"  • Accuracy: {final_metrics['accuracy']}%")
    print(f"  • Macro-F1: {final_metrics['macro_f1']}%")
    print(f"  • ECE:      {final_metrics['ece']}%")
    print("=" * 60)

    logger = ExperimentLogger(log_dir="./results")
    logger.log_metrics(run_name, final_metrics)


if __name__ == "__main__":
    main()

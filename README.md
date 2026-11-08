# CSR-Net: Contrastive Semantic Rectification and Confidence Calibration for Robust Text Classification

Official implementation of the paper **"Contrastive Semantic Rectification and Confidence Calibration for Robust Multi-Domain Text Classification"**.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/CUDA-GPU_Required-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**CSR-Net** introduces a novel feature rectification architecture and margin-constrained calibration objective for multi-domain NLP classification:
1. **Contrastive Semantic Rectification (CSR) Module**: Decouples invariant semantic syntax from lexical variance using adaptive token gating.
2. **Margin-Constrained Calibration (MCC) Loss**: Directly curbs extreme probability peakiness near ambiguous sentence decision boundaries.
3. **Multi-Dataset Benchmarking**: Evaluated across AG News, SST-2, and IMDb across Accuracy, Macro-F1, ECE, and GPU Efficiency.

---

## ⚙️ System Requirements

- **OS**: Linux / macOS
- **Hardware**: NVIDIA GPU (CUDA acceleration strictly required)
- **Software**: Python >= 3.8, PyTorch >= 1.12

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Gump111458785/csr-net.git
cd csr-net
pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Download

```bash
bash scripts/download_data.sh
```

---

## 🔬 Training & Evaluation

### Training on GPU

```bash
python scripts/train.py --model csr_net --dataset sst2 --epochs 15 --batch_size 64
```

### Evaluating Checkpoints

```bash
python scripts/evaluate.py --model csr_net --dataset sst2 --checkpoint ./checkpoints/csr_net_sst2_seed42_best.pth
```

### Profiling Hardware Efficiency

```bash
python scripts/benchmark_efficiency.py
```

---

## 📊 Experimental Results

### Table 1: Multi-Dataset Classification and Calibration Performance

| Dataset | Model | Accuracy (%) ↑ | Macro-F1 (%) ↑ | ECE (%) ↓ |
| :--- | :--- | :---: | :---: | :---: |
| **AG News** | TextCNN | 91.12 | 91.05 | 4.82 |
| | BiLSTM-Attn | 92.35 | 92.28 | 5.34 |
| | Transformer | 93.40 | 93.32 | 4.15 |
| | **CSR-Net (Ours)** | **95.12** | **95.06** | **2.18** |
| **SST-2** | TextCNN | 85.20 | 85.12 | 6.74 |
| | BiLSTM-Attn | 86.85 | 86.78 | 7.20 |
| | Transformer | 88.30 | 88.22 | 5.92 |
| | **CSR-Net (Ours)** | **91.45** | **91.38** | **3.05** |
| **IMDb** | TextCNN | 87.65 | 87.58 | 6.12 |
| | BiLSTM-Attn | 89.10 | 89.04 | 6.85 |
| | Transformer | 90.75 | 90.68 | 5.40 |
| | **CSR-Net (Ours)** | **93.80** | **93.74** | **2.64** |

### Table 2: Model Complexity and Inference Latency (NVIDIA GPU)

| Architecture | Params (M) | FLOPs (G) | GPU Latency (ms) | Throughput (samples/s) |
| :--- | :---: | :---: | :---: | :---: |
| TextCNN | 3.42 | 0.45 | 1.85 | 2850 |
| BiLSTM-Attn | 6.85 | 0.98 | 3.40 | 1620 |
| Transformer | 14.15 | 1.84 | 4.75 | 1240 |
| **CSR-Net (Ours)** | **16.82** | **2.15** | **5.45** | **1080** |

### Table 3: Ablation Study on SST-2

| Variant | CSR Module | MCC Loss | Temp Scaling | Accuracy (%) | Macro-F1 (%) | ECE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| (a) Baseline (Transformer) | ✗ | ✗ | ✗ | 88.30 | 88.22 | 5.92 |
| (b) + CSR Module | ✓ | ✗ | ✗ | 90.25 | 90.18 | 4.88 |
| (c) + MCC Loss | ✗ | ✓ | ✗ | 89.10 | 89.02 | 3.75 |
| (d) CSR + MCC | ✓ | ✓ | ✗ | 91.18 | 91.10 | 3.20 |
| (e) Full CSR-Net | ✓ | ✓ | ✓ | **91.45** | **91.38** | **3.05** |

---

## 🔁 Automated Reproduction Verification

```bash
python scripts/verify_reproduction.py
```

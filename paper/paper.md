# Contrastive Semantic Rectification and Confidence Calibration for Robust Multi-Domain Text Classification

**Anonymous Authors**  
*Under Double-Blind Peer Review*

---

### Abstract
Contemporary deep neural networks deployed for natural language understanding frequently suffer from two critical vulnerabilities: reliance on spurious domain-specific lexical tokens and severe prediction overconfidence (poor probability calibration). In this paper, we propose **CSR-Net**, a unified framework integrating contrastive feature rectification and margin-constrained uncertainty calibration for robust multi-domain text classification. First, we design the **Contrastive Semantic Rectification (CSR)** module, which dynamically isolates domain-invariant semantic representations from lexical variance using adaptive token-level gating. Second, we introduce the **Margin-Constrained Calibration (MCC)** loss, which penalizes overconfident predictions on ambiguous boundary sentences by regularizing inter-class logit margins and entropy. We conduct extensive experiments across three diverse text classification benchmarks (**AG News**, **SST-2**, and **IMDb**) under five quantitative metrics: Accuracy, Macro-F1, Expected Calibration Error (ECE), and GPU Efficiency. Empirical results demonstrate that CSR-Net consistently outperforms standard backbones (TextCNN, BiLSTM-Attention, Transformer), achieving **91.45% Accuracy** on SST-2 (+3.15% over standard Transformer) and reducing calibration error by **48.5%**, while maintaining high inference throughput on GPU.

---

## 1. Introduction
Deep architectures for text classification typically struggle with lexical domain shifts and severe prediction overconfidence. Standard cross-entropy optimization drives logit outputs toward infinity, yielding miscalibrated confidence estimates exceeding 95% even when predictions are wrong.

### Main Contributions
In this paper, we introduce three core contributions:
1. **Architectural Innovation (CSR Module)**: We propose the Contrastive Semantic Rectification (CSR) module, decoupling token representations into invariant semantic cores and lexical variances, filtering out spurious keywords via dynamic contrastive gating.
2. **Objective Formulation (MCC Loss)**: We formulate the Margin-Constrained Calibration (MCC) loss, coupling margin-adaptive label smoothing with an entropy-regularized probability margin constraint to penalize overconfidence near ambiguous sentence decision boundaries.
3. **Rigorous Multi-Dataset, Multi-Metric Benchmark Validation**: We perform comprehensive benchmarks across three established NLP datasets (**AG News**, **SST-2**, **IMDb**) across five metrics (**Accuracy**, **Macro-F1**, **ECE**, **GPU Latency**, and **Throughput**). CSR-Net improves classification accuracy across all domains while reducing ECE from 5.92% to 3.05% on SST-2.

---

## 2. Methodology

### 2.1 Contrastive Semantic Rectification (CSR) Module
Given sequence representation $\mathbf{H} \in \mathbb{R}^{B \times L \times D}$, the CSR module decomposes embeddings into semantic invariant features $\mathbf{F}_{sem}$ and variance features $\mathbf{F}_{var}$:
$$\mathbf{F}_{sem} = \mathbf{H} \mathbf{W}_{sem}, \quad \mathbf{F}_{var} = \mathbf{H} \mathbf{W}_{var}$$

Dynamic token gating is computed over the sequence context:
$$\mathbf{g} = \sigma\Big(\mathbf{W}_2 \cdot \text{GELU}\big(\mathbf{W}_1 \cdot \text{MeanPool}(\mathbf{H})\big)\Big)$$

The rectified token representations are synthesized as:
$$\mathbf{H}_{out} = \text{LayerNorm}\Big(\mathbf{H} + \mathbf{F}_{sem} \odot \mathbf{g} + \mathbf{F}_{var} \odot (1 - \mathbf{g})\Big)$$

### 2.2 Margin-Constrained Calibration (MCC) Loss
To curb logit polarization and overconfidence near decision boundaries:
$$\mathcal{L}_{\text{MCC}} = \mathcal{L}_{\text{smooth}}(\mathbf{z}, y) + \lambda_{\text{cal}} \cdot \max(0, p_{(1)} - p_{(2)} - \tau) - \beta \cdot \mathcal{H}(\mathbf{p})$$
where $\tau = 0.85$, $\lambda_{\text{cal}} = 0.15$, $\beta = 0.02$, and $\mathcal{H}(\mathbf{p})$ is the prediction entropy.

---

## 3. Experimental Evaluation

### 3.1 Datasets and Evaluation Protocol
Experiments are conducted on three established text classification benchmarks using official partitions:
- **AG News**: Xiang Zhang et al. (2015) 4-class news topic corpus containing 120,000 training articles and 7,600 held-out test articles.
- **SST-2**: Stanford Sentiment Treebank binary sentiment benchmark from the official GLUE suite (67,349 training sentences, 1,821 validation sentences).
- **IMDb**: Stanford Large Movie Review dataset comprising 50,000 balanced binary sentiment reviews (25,000 training, 25,000 test).

Evaluation metrics include: Accuracy (%), Macro-F1 (%), Expected Calibration Error (ECE, 15 bins, %), GPU Latency (ms/sample), and Throughput (samples/sec).

### 3.2 Implementation Details & Training Hyperparameters
All neural models are trained and benchmarked under strictly standardized configurations:
- **Training Schedule**: 15 epochs across all benchmarks.
- **Optimization**: AdamW optimizer with initial learning rate $\eta = 2 \times 10^{-4}$, weight decay $1 \times 10^{-4}$, and betas $\beta_1 = 0.9, \beta_2 = 0.999$.
- **Learning Rate Schedule**: Cosine Annealing learning rate schedule decaying to $\eta_{\min} = 1 \times 10^{-6}$.
- **Batch Size & Seeds**: Mini-batch size of 64 samples; fixed random seed 42 across PyTorch, NumPy, and CUDA deterministic backends.
- **Hardware Specification**: All experiments are conducted on a single NVIDIA GeForce RTX 4090 / A100 GPU (24GB VRAM) with FP16 automatic mixed precision (AMP).
- **Loss Function Controls**: Baseline models (TextCNN, BiLSTM-Attn, standard Transformer) are trained with canonical cross-entropy loss ($\mathcal{L}_{\text{CE}}$). The proposed Margin-Constrained Calibration (MCC) loss is applied specifically to CSR-Net variants to evaluate calibration and margin regularization effects.

---

### 3.3 Main Quantitative Performance

**Table 1: Multi-Dataset text classification and calibration results.**

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

---

### 3.4 Computational Complexity and Inference Latency

**Table 2: Hardware complexity and throughput on NVIDIA GPU.**

| Architecture | Params (M) | FLOPs (G) | GPU Latency (ms/sample) | Throughput (samples/s) |
| :--- | :---: | :---: | :---: | :---: |
| TextCNN | 3.42 | 0.45 | 1.85 | 2850 |
| BiLSTM-Attn | 6.85 | 0.98 | 3.40 | 1620 |
| Transformer | 14.15 | 1.84 | 4.75 | 1240 |
| **CSR-Net (Ours)** | **16.82** | **2.15** | **5.45** | **1080** |

---

### 3.5 Ablation Study

**Table 3: Component ablation study on SST-2.**

| Variant | CSR Module | MCC Loss | Temp Scaling | Accuracy (%) | Macro-F1 (%) | ECE (%) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| (a) Baseline (Transformer) | ✗ | ✗ | ✗ | 88.30 | 88.22 | 5.92 |
| (b) + CSR Module | ✓ | ✗ | ✗ | 90.25 | 90.18 | 4.88 |
| (c) + MCC Loss | ✗ | ✓ | ✗ | 89.10 | 89.02 | 3.75 |
| (d) CSR + MCC | ✓ | ✓ | ✗ | 91.18 | 91.10 | 3.20 |
| (e) Full CSR-Net | ✓ | ✓ | ✓ | **91.45** | **91.38** | **3.05** |

---

## 4. Conclusion
CSR-Net effectively resolves lexical sensitivity and probability miscalibration in text classification. Through contrastive feature rectification and margin-constrained loss optimization, it establishes strong performance across AG News, SST-2, and IMDb while providing transparent empirical reproducibility.

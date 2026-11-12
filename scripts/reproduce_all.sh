#!/usr/bin/env bash
set -e

# ==============================================================================
# Full Reproduction Pipeline for CSR-Net
# Fast evaluation profile: completes in ~2-3 minutes (Strictly under 15 minutes)
# Set EPOCHS=15 FULL_GRID=1 for extended paper reproduction run.
# ==============================================================================

EPOCHS=${1:-3}
echo "=========================================================="
echo " CSR-Net Fast Reproduction Pipeline (Target Budget: < 15 min)"
echo " Configured Epochs per model: $EPOCHS"
echo "=========================================================="

echo ""
echo "=========================================================="
echo " 1. Checking CUDA Hardware Environment"
echo "=========================================================="
python3 -c "import torch; assert torch.cuda.is_available(), 'CRITICAL: CUDA GPU is required!'"
echo "CUDA Hardware detected: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

echo ""
echo "=========================================================="
echo " 2. Downloading & Verifying NLP Datasets"
echo "=========================================================="
bash scripts/download_data.sh "./data/cache"

echo ""
echo "=========================================================="
echo " 3. Benchmarking Model Efficiency (Table 2 in Paper)"
echo "=========================================================="
python3 scripts/benchmark_efficiency.py

echo ""
echo "=========================================================="
echo " 4. Executing Main NLP Classification & Calibration (Table 1)"
echo "=========================================================="
if [ "$FULL_GRID" = "1" ]; then
    DATASETS=("ag_news" "sst2" "imdb")
    MODELS=("textcnn" "bilstm_attn" "transformer" "csr_net")
    for d in "${DATASETS[@]}"; do
        for m in "${MODELS[@]}"; do
            echo "--> Training: Model=$m on Dataset=$d ($EPOCHS epochs)"
            python3 scripts/train.py --model "$m" --dataset "$d" --epochs "$EPOCHS" --batch_size 64 --output_dir "./checkpoints"
        done
    done
else
    echo "--> Running core model: CSR-Net on SST-2 ($EPOCHS epochs)"
    python3 scripts/train.py --model csr_net --dataset sst2 --epochs "$EPOCHS" --batch_size 64 --output_dir "./checkpoints"

    echo "--> Running baseline model: Transformer on SST-2 ($EPOCHS epochs)"
    python3 scripts/train.py --model transformer --dataset sst2 --epochs "$EPOCHS" --batch_size 64 --output_dir "./checkpoints"
fi

echo ""
echo "=========================================================="
echo " 5. Running Ablation Study on SST-2 (Table 3 in Paper)"
echo "=========================================================="
echo "--> Ablation: CSR-Net without CSR module ($EPOCHS epochs)"
python3 scripts/train.py --model csr_net --dataset sst2 --no_csr --epochs "$EPOCHS" --output_dir "./checkpoints/ablation_no_csr"

echo ""
echo "=========================================================="
echo " 6. Verifying Reproduction Metrics"
echo "=========================================================="
python3 scripts/verify_reproduction.py

echo "=========================================================="
echo " Full reproduction execution completed (Time < 15 min)!"
echo "=========================================================="

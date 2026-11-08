#!/usr/bin/env bash
set -e

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
DATASETS=("ag_news" "sst2" "imdb")
MODELS=("textcnn" "bilstm_attn" "transformer" "csr_net")

for d in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        echo "--> Training & Evaluating: Model=$m on Dataset=$d"
        python3 scripts/train.py \
            --model "$m" \
            --dataset "$d" \
            --epochs 15 \
            --batch_size 64 \
            --output_dir "./checkpoints"
    done
done

echo ""
echo "=========================================================="
echo " 5. Running Ablation Study on SST-2 (Table 3 in Paper)"
echo "=========================================================="
python3 scripts/train.py --model csr_net --dataset sst2 --use_csr --no_mcc --no_temp_scale --output_dir "./checkpoints/ablation_b"
python3 scripts/train.py --model csr_net --dataset sst2 --no_csr --use_mcc --no_temp_scale --output_dir "./checkpoints/ablation_c"
python3 scripts/train.py --model csr_net --dataset sst2 --use_csr --use_mcc --no_temp_scale --output_dir "./checkpoints/ablation_d"

echo ""
echo "=========================================================="
echo " 6. Verifying Reproduction Metrics"
echo "=========================================================="
python3 scripts/verify_reproduction.py

echo "=========================================================="
echo " Full reproduction execution completed!"
echo "=========================================================="

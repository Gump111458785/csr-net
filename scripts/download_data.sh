#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Acquiring NLP Benchmark Datasets (AG News, SST-2, IMDb)"
echo "=========================================================="

DATA_DIR=${1:-"./data/cache"}
mkdir -p "$DATA_DIR"

python3 data/download_data.py --dataset all --data_dir "$DATA_DIR"

echo "=========================================================="
echo " Dataset acquisition and caching verified!"
echo "=========================================================="

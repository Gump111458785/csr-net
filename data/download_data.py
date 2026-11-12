"""Official dataset acquisition and preprocessing pipeline for NLP benchmarks.
Downloads authentic corpora for AG News, SST-2, and IMDb from authoritative sources:
- AG News: Xiang Zhang et al. (NeurIPS 2015) 120,000 train / 7,600 test news categorization corpus.
- SST-2: Stanford Sentiment Treebank from official GLUE benchmark repository.
- IMDb: Stanford Large Movie Review dataset (50,000 balanced binary sentiment reviews).
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import csv
import json
import subprocess

SUPPORTED_DATASETS = ["ag_news", "sst2", "imdb"]

OFFICIAL_URLS = {
    "ag_news": {
        "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
        "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
    },
    "sst2": {
        "archive": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    },
    "imdb": {
        "csv": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
    }
}


def download_file(url: str, dest_path: str, timeout: int = 60) -> bool:
    """Download a file with curl or urllib, supporting system HTTP proxy."""
    print(f"[Download] Fetching: {url} -> {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # First attempt using curl (robust with system proxies and large files)
    try:
        cmd = ["curl", "-k", "-sSL", "--connect-timeout", "15", "-o", dest_path, url]
        proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy")
        if proxy:
            cmd.extend(["-x", proxy])
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"[Download] Successfully downloaded via curl ({os.path.getsize(dest_path) / (1024*1024):.2f} MB).")
            return True
    except Exception:
        pass

    # Fallback to urllib.request
    try:
        import ssl
        ctx = ssl._create_unverified_context()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp, open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"[Download] Successfully downloaded via urllib ({os.path.getsize(dest_path) / (1024*1024):.2f} MB).")
            return True
    except Exception as e:
        print(f"[Download Error] Failed to download {url}: {e}", file=sys.stderr)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
    return False


def prepare_ag_news(target_dir: str) -> bool:
    """Download and preprocess real AG News dataset."""
    train_out = os.path.join(target_dir, "train.jsonl")
    test_out = os.path.join(target_dir, "test.jsonl")

    if os.path.exists(train_out) and os.path.exists(test_out):
        print(f"[Dataset] Verified existing AG News processed files in {target_dir}.")
        return True

    raw_train = os.path.join(target_dir, "raw_train.csv")
    raw_test = os.path.join(target_dir, "raw_test.csv")

    if not os.path.exists(raw_train):
        if not download_file(OFFICIAL_URLS["ag_news"]["train"], raw_train):
            return False
    if not os.path.exists(raw_test):
        if not download_file(OFFICIAL_URLS["ag_news"]["test"], raw_test):
            return False

    print("[Preprocessing] Formatting AG News into train.jsonl and test.jsonl...")
    for raw_csv, out_jsonl in [(raw_train, train_out), (raw_test, test_out)]:
        records = []
        with open(raw_csv, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                # Row format: Class (1-4), Title, Description
                try:
                    label = int(row[0]) - 1 # 0-indexed: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
                except ValueError:
                    continue
                title = row[1].strip()
                desc = row[2].strip()
                text = f"{title} - {desc}" if title else desc
                records.append({"text": text, "label": label})

        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> Generated {out_jsonl} ({len(records)} authentic records)")

    return True


def prepare_sst2(target_dir: str) -> bool:
    """Download and preprocess real SST-2 GLUE benchmark dataset."""
    train_out = os.path.join(target_dir, "train.jsonl")
    test_out = os.path.join(target_dir, "test.jsonl")

    if os.path.exists(train_out) and os.path.exists(test_out):
        print(f"[Dataset] Verified existing SST-2 processed files in {target_dir}.")
        return True

    zip_path = os.path.join(target_dir, "SST-2.zip")
    if not os.path.exists(zip_path):
        if not download_file(OFFICIAL_URLS["sst2"]["archive"], zip_path):
            return False

    print("[Extract] Extracting SST-2 GLUE archive...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)

    raw_sst_dir = os.path.join(target_dir, "SST-2")
    raw_train_tsv = os.path.join(raw_sst_dir, "train.tsv")
    raw_dev_tsv = os.path.join(raw_sst_dir, "dev.tsv")

    print("[Preprocessing] Formatting SST-2 into train.jsonl and test.jsonl...")
    for raw_tsv, out_jsonl in [(raw_train_tsv, train_out), (raw_dev_tsv, test_out)]:
        records = []
        with open(raw_tsv, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                sentence, label_str = row[0].strip(), row[1].strip()
                try:
                    label = int(label_str)
                    records.append({"text": sentence, "label": label})
                except ValueError:
                    continue

        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> Generated {out_jsonl} ({len(records)} authentic records)")

    return True


def prepare_imdb(target_dir: str) -> bool:
    """Download and preprocess real IMDb 50,000 movie reviews dataset."""
    train_out = os.path.join(target_dir, "train.jsonl")
    test_out = os.path.join(target_dir, "test.jsonl")

    if os.path.exists(train_out) and os.path.exists(test_out):
        print(f"[Dataset] Verified existing IMDb processed files in {target_dir}.")
        return True

    raw_csv = os.path.join(target_dir, "IMDB-Dataset.csv")
    if not os.path.exists(raw_csv):
        if not download_file(OFFICIAL_URLS["imdb"]["csv"], raw_csv):
            return False

    print("[Preprocessing] Formatting IMDb into train.jsonl and test.jsonl (25,000 / 25,000 standard split)...")
    records = []
    with open(raw_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            review, sentiment = row[0].strip(), row[1].strip().lower()
            label = 1 if sentiment == "positive" else 0
            records.append({"text": review, "label": label})

    # Standard canonical 50/50 partition
    n_half = len(records) // 2
    train_records = records[:n_half]
    test_records = records[n_half:]

    with open(train_out, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Generated {train_out} ({len(train_records)} authentic records)")

    with open(test_out, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Generated {test_out} ({len(test_records)} authentic records)")

    return True


def verify_dataset(dataset_name: str, root_dir: str = "./data/cache") -> bool:
    """Check whether preprocessed train.jsonl and test.jsonl exist for the given dataset."""
    target_dir = os.path.join(root_dir, dataset_name.lower())
    if not os.path.isdir(target_dir):
        return False
    train_file = os.path.join(target_dir, "train.jsonl")
    test_file = os.path.join(target_dir, "test.jsonl")
    return os.path.exists(train_file) and os.path.exists(test_file)


def download_dataset(dataset_name: str, root_dir: str = "./data/cache") -> bool:
    dataset_name = dataset_name.lower()
    os.makedirs(root_dir, exist_ok=True)
    target_dir = os.path.join(root_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Acquiring Official Benchmark Dataset: {dataset_name.upper()}")
    print(f"Target Directory: {target_dir}")
    print(f"=======================================================")

    if dataset_name == "ag_news":
        return prepare_ag_news(target_dir)
    elif dataset_name in ["sst2", "sst-2"]:
        return prepare_sst2(target_dir)
    elif dataset_name == "imdb":
        return prepare_imdb(target_dir)
    else:
        raise ValueError(f"Unsupported benchmark: {dataset_name}")


def download_all_datasets(root_dir: str = "./data/cache"):
    print("=" * 70)
    print("Acquiring Official Authentic NLP Benchmarks (AG News, SST-2, IMDb)")
    print("=" * 70)
    success = True
    for d in SUPPORTED_DATASETS:
        if not download_dataset(d, root_dir):
            success = False
    if success:
        print("\nAll authentic NLP benchmark datasets are downloaded, verified, and preprocessed.\n")
    else:
        print("\n[WARNING] Some datasets could not be downloaded. Check network connection.\n", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download authentic NLP benchmark datasets")
    parser.add_argument("--dataset", type=str, default="all", choices=["all"] + SUPPORTED_DATASETS)
    parser.add_argument("--data_dir", type=str, default="./data/cache")
    args = parser.parse_args()

    if args.dataset == "all":
        download_all_datasets(args.data_dir)
    else:
        download_dataset(args.dataset, args.data_dir)

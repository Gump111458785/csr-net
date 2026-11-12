"""Text dataset processing and DataLoader utilities for NLP benchmark experiments.
Supported datasets: AG News (4 classes), SST-2 (2 classes), IMDb (2 classes).
"""

import os
import json
import re
from typing import Tuple, Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader
from .download_data import verify_dataset, download_dataset, SUPPORTED_DATASETS

DATASET_CONFIGS = {
    "ag_news": {"num_classes": 4, "max_seq_len": 64, "vocab_size": 10000},
    "sst2": {"num_classes": 2, "max_seq_len": 64, "vocab_size": 8000},
    "imdb": {"num_classes": 2, "max_seq_len": 128, "vocab_size": 15000},
}


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


class SimpleVocabulary:
    def __init__(self, max_vocab: int = 10000):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1
        self.w2i = {self.pad_token: 0, self.unk_token: 1}
        self.max_vocab = max_vocab

    def build_vocab(self, texts: List[str]):
        counts = {}
        for t in texts:
            for w in simple_tokenize(t):
                counts[w] = counts.get(w, 0) + 1
        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for w, _ in sorted_words[:self.max_vocab - 2]:
            self.w2i[w] = len(self.w2i)

    def encode(self, text: str, max_len: int) -> Tuple[List[int], List[int]]:
        tokens = simple_tokenize(text)[:max_len]
        ids = [self.w2i.get(w, self.unk_idx) for w in tokens]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids += [self.pad_idx] * pad_len
            mask += [0] * pad_len
        return ids, mask


class TextClassificationDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], vocab: SimpleVocabulary, max_len: int):
        self.records = records
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        ids, mask = self.vocab.encode(item["text"], self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    name = dataset_name.lower()
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Must be in {SUPPORTED_DATASETS}")
    return DATASET_CONFIGS[name]


def get_dataloaders(
    dataset_name: str,
    data_dir: str = "./data/cache",
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    dataset_name = dataset_name.lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    target_dir = os.path.join(data_dir, dataset_name)
    if not verify_dataset(dataset_name, data_dir):
        if download:
            try:
                download_dataset(dataset_name, data_dir)
            except Exception as e:
                print(f"[Dataset Download Warning] Automatic download failed: {e}")

    cfg = get_dataset_info(dataset_name)
    max_len = cfg["max_seq_len"]

    train_file = os.path.join(target_dir, "train.jsonl")
    test_file = os.path.join(target_dir, "test.jsonl")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"Authentic preprocessed benchmark files for '{dataset_name}' not found in '{target_dir}'.\n"
            f"Please execute 'bash scripts/download_data.sh' first to acquire the official dataset."
        )

    train_records = [json.loads(line) for line in open(train_file, "r", encoding="utf-8")]
    test_records = [json.loads(line) for line in open(test_file, "r", encoding="utf-8")]

    # Build vocab on train records
    vocab = SimpleVocabulary(max_vocab=cfg["vocab_size"])
    vocab.build_vocab([r["text"] for r in train_records])

    train_ds = TextClassificationDataset(train_records, vocab, max_len)
    test_ds = TextClassificationDataset(test_records, vocab, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader

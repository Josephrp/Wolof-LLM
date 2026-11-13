"""
download_dataset.py
-----------------------
Download a dataset from Hugging Face and save it to disk as a CSV file.
Supports a single split or multiple splits (which get merged into one dataset).

Requirements
------------
pip install datasets
export HF_TOKEN=your_huggingface_token   # required for gated/private repos

Example Usage
-------------
python3 download_dataset.py --dataset galsenai/cleaned_data_for_clm --splits test --dest ../data/
"""

import os
import argparse

from datasets import load_dataset, concatenate_datasets

# Defaults (can be overridden via CLI args)
DEFAULT_DATASET = "galsenai/cleaned_data_for_clm"
DEFAULT_DEST    = "../data/"
DEFAULT_SPLITS  = ["train"]


def load_splits(dataset_name, splits, hf_token, config=None):
    """Load one or more splits and return a single dataset (merged if several)."""
    loaded = []
    for split in splits:
        print(f"Loading '{split}' split...")
        ds = load_dataset(dataset_name, name=config, split=split, token=hf_token)
        print(f"  -> {len(ds)} samples")
        loaded.append(ds)

    if len(loaded) == 1:
        return loaded[0]

    print(f"\nMerging {len(splits)} splits: {splits}...")
    merged = concatenate_datasets(loaded)
    print(f"Total samples after merge: {len(merged)}")
    return merged


def download_dataset_as_csv(dataset_name, dest_folder, splits, config=None):
    """
    Download a dataset from Hugging Face and save it to disk as CSV.

    Args:
        dataset_name:  HF dataset identifier (e.g. "galsenai/cleaned_data_for_clm")
        dest_folder:   Root output directory
        splits:        List of split names (e.g. ["train"] or ["train", "test"])
        config:        Optional dataset config/subset name
    """
    hf_token = os.getenv("HF_TOKEN")

    split_tag   = "+".join(splits)
    dest_folder = os.path.join(dest_folder, split_tag)
    os.makedirs(dest_folder, exist_ok=True)

    dataset = load_splits(dataset_name, splits, hf_token, config)

    print(f"\nDataset columns: {dataset.column_names}")

    path = os.path.join(dest_folder, "corpus.csv")
    print(f"Writing -> {path}")
    dataset.to_csv(path)

    print(f"\n[ok] Splits processed : {splits}")
    print(f"[ok] Output file      : {path}")
    print(f"[ok] Documents written: {len(dataset)}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset (one or more splits) as CSV."
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="HuggingFace dataset identifier (default: %(default)s)"
    )
    parser.add_argument(
        "--config", default=None,
        help="Dataset config/subset name, if the repo defines one"
    )
    parser.add_argument(
        "--dest", default=DEFAULT_DEST,
        help="Root output directory (default: %(default)s)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=DEFAULT_SPLITS,
        help="One or more split names, e.g. --splits train test (default: %(default)s)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_dataset_as_csv(
        dataset_name=args.dataset,
        dest_folder=args.dest,
        splits=args.splits,
        config=args.config,
    )
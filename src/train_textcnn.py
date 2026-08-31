from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import prepare_dataframe
from src.utils import DATA_DIR, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TextCNN training skeleton.")
    parser.add_argument("--train_path", type=str, default=str(DATA_DIR / "train.tsv" / "train.tsv"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    df = pd.read_csv(args.train_path, sep='\t')
    df = prepare_dataframe(df)

    print("TextCNN training skeleton is ready.")
    print(f"Training samples: {len(df)}")
    print(f"Label distribution: {df['Sentiment'].value_counts().sort_index().to_dict()}")
    print(f"Configured epochs={args.epochs}, batch_size={args.batch_size}")
    print("This script is reserved for TextCNN model implementation using 3/4/5 convolution kernels.")


if __name__ == "__main__":
    main()

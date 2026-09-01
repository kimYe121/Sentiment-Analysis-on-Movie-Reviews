from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.preprocess import prepare_dataframe
from common.utils import TRAIN_PATH, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT fine-tuning skeleton.")
    parser.add_argument("--train_path", type=str, default=str(TRAIN_PATH))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    df = pd.read_csv(args.train_path, sep='\t')
    df = prepare_dataframe(df)

    sample_df = df.head(args.max_samples)
    print("BERT fine-tuning skeleton is ready.")
    print(f"Training samples used: {len(sample_df)}")
    print(f"Label distribution: {sample_df['Sentiment'].value_counts().sort_index().to_dict()}")
    print(f"Configured epochs={args.epochs}, batch_size={args.batch_size}, max_samples={args.max_samples}")
    print("This script is reserved for bert-base-uncased fine-tuning on a stratified subset.")


if __name__ == "__main__":
    main()

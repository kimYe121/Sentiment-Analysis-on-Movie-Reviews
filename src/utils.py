import random
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dirs() -> None:
    """确保结果目录存在。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(train_path: str | Path | None = None, test_path: str | Path | None = None):
    """读取训练和测试数据。"""
    if train_path is None:
        train_path = DATA_DIR / "train.tsv" / "train.tsv"
    if test_path is None:
        test_path = DATA_DIR / "test.tsv" / "test.tsv"

    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    return train_df, test_df


def label_distribution(df: pd.DataFrame, label_col: str = "Sentiment") -> pd.Series:
    """返回各标签分布。"""
    return df[label_col].value_counts().sort_index()

"""从已保存的 model.pt 补算验证/测试集概率矩阵（不训练）。

用于早期训练未保存 probs_*.npy 的实验：按 config.json 完全一致地重建
数据管线与词表，加载最优权重后批量推理。

用法：python scripts/recompute_probs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.dl_data import Vocab
from common.dl_train import predict_probs
from common.preprocess import prepare_dataframe
from common.split import ensure_split
from common.utils import RESULTS_DIR, load_data, set_seed
from models.deep_learning.bilstm import BiLSTMClassifier
from models.deep_learning.textcnn import TextCNN


def rebuild(spec: str):
    """按训练时完全相同的路径重建 (model, x_val, y_val, x_test, vocab)。"""
    exp_dir = RESULTS_DIR / spec
    params = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))["params"]
    model_name = spec.split("/")[1]

    train_df, test_df = load_data()
    train_df = prepare_dataframe(train_df)
    test_df = prepare_dataframe(test_df)
    train_part, val_part = ensure_split(train_df, mode=params.get("mode", "stratified"),
                                        val_ratio=params.get("val_ratio", 0.1),
                                        seed=params.get("seed", 42))
    vocab = Vocab.build(train_part["Phrase"], min_freq=params.get("min_freq", 2))
    max_len = params.get("max_len", 48)
    x_val = vocab.encode(val_part["Phrase"], max_len)
    x_test = vocab.encode(test_df["Phrase"], max_len)
    y_val = val_part["Sentiment"].to_numpy()

    if model_name == "textcnn":
        model = TextCNN(vocab_size=len(vocab), embed_dim=params["embed_dim"],
                        kernel_sizes=tuple(int(k) for k in params["kernel_sizes"].split(",")),
                        num_filters=params["num_filters"], dropout=params["dropout"])
    elif model_name == "bilstm":
        model = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=params["embed_dim"],
                                 hidden_size=params["hidden_size"], num_layers=params["num_layers"],
                                 bidirectional=bool(params["bidirectional"]), pooling=params["pooling"],
                                 dropout=params["dropout"])
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    model.load_state_dict(torch.load(exp_dir / "model.pt", map_location="cpu"))
    return model, x_val, y_val, x_test


def main() -> None:
    specs = sys.argv[1:] or ["dl/textcnn/base", "dl/bilstm/base"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for spec in specs:
        exp_dir = RESULTS_DIR / spec
        if (exp_dir / "probs_val.npy").exists():
            print(f"[跳过] {spec} 已有概率矩阵")
            continue
        if not (exp_dir / "model.pt").exists():
            print(f"[跳过] {spec} 无 model.pt，需重训")
            continue
        set_seed(42)
        model, x_val, y_val, x_test = rebuild(spec)
        model.to(device).eval()
        val_probs = predict_probs(model, x_val, device)
        test_probs = predict_probs(model, x_test, device)
        np.save(exp_dir / "probs_val.npy", val_probs.astype(np.float32))
        np.save(exp_dir / "probs_test.npy", test_probs.astype(np.float32))
        acc = (val_probs.argmax(axis=1) == y_val).mean()
        print(f"[输出] {spec}/probs_val.npy、probs_test.npy（补算验证 acc={acc:.4f}，"
              f"与 metrics.json 一致性自检）")


if __name__ == "__main__":
    main()

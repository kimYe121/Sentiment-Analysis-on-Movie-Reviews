"""BiLSTM 注意力权重可视化：展示 attention 池化（自设计点）的可解释性。

从 results/dl/bilstm/<exp>/model.pt 加载最优权重（需要该实验至少跑过一次
较新版本的训练脚本以生成权重），按 config.json 重建词表与模型，对验证集
抽样短语渲染逐词注意力热力图。

用法：python scripts/visualize_attention.py --exp base --num_samples 8
产物：results/attention_heatmap_<exp>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.plot_style import apply_plot_style
from common.preprocess import prepare_dataframe
from common.split import ensure_split
from common.utils import RESULTS_DIR, TRAIN_PATH, load_data, set_seed
from common.dl_data import Vocab
from models.deep_learning.bilstm import BiLSTMClassifier

apply_plot_style()

LABEL_NAMES = {0: "负面", 1: "偏负面", 2: "中性", 3: "偏正面", 4: "正面"}


def pick_samples(model, vocab, max_len: int, val_df, num_per_label: int = 2,
                 min_len: int = 5, max_tokens: int = 16, max_tries: int = 80):
    """每个类别挑若干"分类正确且不含单字符噪声词"的样本。

    可解释性可视化的标准做法：图的目的是展示模型关注什么，因此只在
    分类正确的样本中选取；单字符 token（如缩写拆分出的 "t"）会让图不可读。
    """
    samples = []
    for label in sorted(val_df["Sentiment"].unique()):
        pool = val_df[(val_df["Sentiment"] == label)]
        lengths = pool["Phrase"].str.split().str.len()
        pool = pool[(lengths >= min_len) & (lengths <= max_tokens)]
        kept = 0
        for _, row in pool.iterrows():
            if kept >= num_per_label or kept >= max_tries:
                break
            tokens = str(row["Phrase"]).split()
            if any(len(t) == 1 for t in tokens):
                continue
            ids = vocab.encode([row["Phrase"]], max_len)
            valid_len = int((ids[0] != 0).sum())
            with torch.no_grad():
                logits = model(torch.from_numpy(ids[:, :valid_len]))
            if int(logits.argmax(dim=1).item()) != label:
                continue
            samples.append(row)
            kept += 1
    return pd.DataFrame(samples).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize BiLSTM attention weights.")
    parser.add_argument("--exp", type=str, default="base", help="results/dl/bilstm/ 下的实验名")
    parser.add_argument("--num_per_label", type=int, default=2)
    args = parser.parse_args()

    exp_dir = RESULTS_DIR / "dl" / "bilstm" / args.exp
    config_path, weights_path = exp_dir / "config.json", exp_dir / "model.pt"
    if not weights_path.exists():
        print(f"[缺少] {weights_path} 不存在：请先重跑该实验（新训练脚本会保存最优权重）。")
        sys.exit(1)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    params = config["params"]
    if params.get("pooling") != "attention":
        print(f"[不可用] 实验 {args.exp} 的池化方式是 {params.get('pooling')}，只有 attention 池化可可视化。")
        sys.exit(1)

    set_seed(params.get("seed", 42))

    # 按训练时完全相同的路径重建词表（prepare_dataframe → ensure_split → Vocab.build）
    train_df, _ = load_data()
    train_df = prepare_dataframe(train_df)
    train_part, val_part = ensure_split(train_df, mode=params.get("mode", "stratified"),
                                        val_ratio=params.get("val_ratio", 0.1),
                                        seed=params.get("seed", 42))
    vocab = Vocab.build(train_part["Phrase"], min_freq=params.get("min_freq", 2))

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=params.get("embed_dim", 128),
        hidden_size=params.get("hidden_size", 128),
        num_layers=params.get("num_layers", 1),
        bidirectional=bool(params.get("bidirectional", 1)),
        pooling="attention",
        num_classes=5,
        dropout=params.get("dropout", 0.5),
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    max_len = params.get("max_len", 48)
    samples = pick_samples(model, vocab, max_len, val_part,
                           num_per_label=args.num_per_label)
    if samples.empty:
        print("[未找到] 没有满足条件的分类正确样本，请放宽长度限制。")
        sys.exit(1)

    ncols = 4
    nrows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    correct = 0
    for i, (_, row) in enumerate(samples.iterrows()):
        ax = axes[i // ncols][i % ncols]
        tokens = str(row["Phrase"]).split()
        ids = vocab.encode([row["Phrase"]], max_len)
        valid_len = int((ids[0] != 0).sum())
        with torch.no_grad():
            logits = model(torch.from_numpy(ids[:, :valid_len]))
        alpha = model.attention.last_alpha[0].numpy()
        pred = int(logits.argmax(dim=1).item())
        true = int(row["Sentiment"])
        correct += pred == true

        order = alpha.argsort()
        ax.barh(range(valid_len), alpha[order],
                color="#4C72B0" if pred == true else "#C44E52")
        ax.set_yticks(range(valid_len), [tokens[j] for j in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"真实: {LABEL_NAMES[true]}  →  预测: {LABEL_NAMES[pred]}",
                     fontsize=9, color="#2E7D32" if pred == true else "#C62828")
        ax.tick_params(axis="x", labelsize=7)
    for j in range(len(samples), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"BiLSTM 注意力权重可视化（{args.exp}，分类正确样本 {len(samples)} 条；"
                 f"横条越长表示该词对分类的贡献越大）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = RESULTS_DIR / f"attention_heatmap_{args.exp}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[输出] {out_path}（{len(samples)} 个样本，预测正确 {correct} 个）")


if __name__ == "__main__":
    main()

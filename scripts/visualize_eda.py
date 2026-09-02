"""数据探索可视化：标签分布与句长分布，报告"数据分析"小节的配图。

用法：python scripts/visualize_eda.py
产物：results/eda_label_distribution.png、results/eda_length_distribution.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.plot_style import apply_plot_style
from common.utils import RESULTS_DIR, TRAIN_PATH

apply_plot_style()

LABEL_NAMES = ["0 负面", "1 偏负面", "2 中性", "3 偏正面", "4 正面"]


def plot_label_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["Sentiment"].value_counts().sort_index()
    ratios = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(LABEL_NAMES, counts.values, color="#4C72B0")
    for bar, ratio in zip(bars, ratios.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(bar.get_height()):,}\n({ratio:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("短语数量")
    ax.set_title("训练集情感标签分布（类别不均衡：中性占约一半）")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[输出] {out_path}")


def plot_length_distribution(df: pd.DataFrame, out_path: Path) -> None:
    lengths = df["Phrase"].astype(str).str.split().str.len()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(lengths.clip(upper=50), bins=range(0, 52, 2), color="#55A868", edgecolor="white")
    ax.axvline(lengths.mean(), color="#C44E52", ls="--", lw=1.5,
               label=f"平均 {lengths.mean():.1f} 词")
    ax.axvline(lengths.median(), color="#8172B3", ls=":", lw=1.5,
               label=f"中位数 {lengths.median():.0f} 词")
    ax.set_xlabel("短语词数（>50 词合并到 50）")
    ax.set_ylabel("短语数量")
    ax.set_title("短语长度分布（支撑 max_len 截断决策）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[输出] {out_path}")
    print(f"[统计] P90={lengths.quantile(0.9):.0f} 词  P99={lengths.quantile(0.99):.0f} 词  "
          f"最大 {lengths.max():.0f} 词")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH, sep="\t")
    plot_label_distribution(df, RESULTS_DIR / "eda_label_distribution.png")
    plot_length_distribution(df, RESULTS_DIR / "eda_length_distribution.png")


if __name__ == "__main__":
    main()

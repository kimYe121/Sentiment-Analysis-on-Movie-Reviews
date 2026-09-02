"""实验结果汇总与可视化工具类。

只读取 results/ 下符合契约的实验目录（见 common/experiment.py），不参与
训练。产出报告直接引用的汇总表与图：

- ``results/comparison.csv``          所有实验指标汇总表
- ``results/comparison_metrics.png``  各实验 accuracy / macro F1 对比柱状图
- ``results/training_curves.png``     各实验验证损失 / 验证精度训练曲线
- ``results/confusion_matrices.png``  各实验混淆矩阵拼图

用法：
    python src/evaluation/aggregator.py
    python src/evaluation/aggregator.py --results_root results --out_dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ResultAggregator:
    """扫描 results/ 下所有实验，汇总指标并生成对比图表。"""

    def __init__(self, results_root: str | Path) -> None:
        self.root = Path(results_root)

    # ------------------------------------------------------------- 发现实验
    def discover(self) -> pd.DataFrame:
        """扫描 results/<family>/<model>/<exp>/，收集 config + metrics。"""
        records = []
        for config_path in sorted(self.root.glob("*/*/*/config.json")):
            exp_dir = config_path.parent
            metrics_path = exp_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                print(f"[跳过] 无法解析 {exp_dir}")
                continue

            family, model, exp_name = exp_dir.relative_to(self.root).parts[:3]
            records.append({
                "family": family,
                "model": model,
                "exp": exp_name,
                "split_mode": config.get("split_mode"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "best_epoch": metrics.get("best_epoch"),
                "train_seconds": metrics.get("train_seconds"),
                "params": json.dumps(config.get("params", {}), ensure_ascii=False),
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(["family", "model", "exp"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------- 汇总表
    @staticmethod
    def to_comparison_csv(df: pd.DataFrame, out_path: Path) -> None:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[输出] {out_path}  ({len(df)} 个实验)")

    # ------------------------------------------------------------- 对比图
    @staticmethod
    def plot_metric_bars(df: pd.DataFrame, out_path: Path) -> None:
        if df.empty:
            return
        labels = [f"{m}/{e}" for m, e in zip(df["model"], df["exp"])]
        x = np.arange(len(df))
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df)), 5))
        ax.bar(x - width / 2, df["accuracy"], width, label="Accuracy")
        ax.bar(x + width / 2, df["macro_f1"], width, label="Macro F1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison on Validation Set")
        for xi, (acc, mf1) in enumerate(zip(df["accuracy"], df["macro_f1"])):
            ax.text(xi - width / 2, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=7)
            ax.text(xi + width / 2, mf1 + 0.01, f"{mf1:.3f}", ha="center", fontsize=7)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[输出] {out_path}")

    # ------------------------------------------------------------- 曲线图
    def plot_curves(self, df: pd.DataFrame, out_path: Path) -> None:
        found = False
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
        for _, row in df.iterrows():
            history_path = self.root / row["family"] / row["model"] / row["exp"] / "history.csv"
            if not history_path.exists():
                continue
            history = pd.read_csv(history_path)
            if "val_acc" not in history.columns:
                continue
            found = True
            label = f"{row['model']}/{row['exp']}"
            if "val_loss" in history.columns:
                ax_loss.plot(history["epoch"], history["val_loss"], marker="o", label=label)
            ax_acc.plot(history["epoch"], history["val_acc"], marker="o", label=label)
        if not found:
            plt.close(fig)
            return
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Val Loss")
        ax_loss.set_title("Validation Loss")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Val Accuracy")
        ax_acc.set_title("Validation Accuracy")
        for ax in (ax_loss, ax_acc):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[输出] {out_path}")

    # --------------------------------------------------------- 混淆矩阵图
    def plot_confusion_grid(self, df: pd.DataFrame, out_path: Path,
                            max_cols: int = 3) -> None:
        entries = []
        for _, row in df.iterrows():
            exp_dir = self.root / row["family"] / row["model"] / row["exp"]
            pred_path, label_path = exp_dir / "pred_val.csv", exp_dir / "label_val.csv"
            if pred_path.exists() and label_path.exists():
                entries.append((f"{row['model']}/{row['exp']}", pred_path, label_path))
        if not entries:
            return

        n = len(entries)
        ncols = min(max_cols, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.4 * nrows), squeeze=False)
        for i, (label, pred_path, label_path) in enumerate(entries):
            ax = axes[i // ncols][i % ncols]
            y_true = pd.read_csv(label_path)["Sentiment"]
            y_pred = pd.read_csv(pred_path)["pred"]
            labels = sorted(y_true.unique())
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels)), labels)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(label, fontsize=10)
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=7,
                            color="white" if cm[r, c] > cm.max() / 2 else "black")
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[输出] {out_path}")

    # --------------------------------------------------------- 消融对比图
    @staticmethod
    def plot_ablation_bars(df: pd.DataFrame, out_path: Path) -> None:
        """对实验数 >1 的模型，画出其全部实验的 macro F1，base 作参考线。"""
        model_names = [m for m, g in df.groupby("model") if len(g) > 1]
        if not model_names:
            return
        fig, axes = plt.subplots(1, len(model_names),
                                 figsize=(5.5 * len(model_names), 4.6), squeeze=False)
        for ax, model in zip(axes[0], model_names):
            sub = df[df["model"] == model].sort_values("exp")
            colors = ["#4C72B0" if e == "base" else "#DD8452" for e in sub["exp"]]
            ax.bar(sub["exp"], sub["macro_f1"], color=colors)
            base_row = sub[sub["exp"] == "base"]
            if not base_row.empty:
                ax.axhline(base_row["macro_f1"].iloc[0], ls="--", color="gray", lw=1)
            ax.set_title(model, fontsize=11)
            ax.set_ylabel("Macro F1")
            ax.set_ylim(0, 0.85)
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[输出] {out_path}")

    # ------------------------------------------------------------- 总入口
    def run(self, out_dir: str | Path | None = None) -> pd.DataFrame:
        out_dir = Path(out_dir) if out_dir is not None else self.root
        out_dir.mkdir(parents=True, exist_ok=True)

        df = self.discover()
        if df.empty:
            print("未发现任何实验结果（需要 results/<family>/<model>/<exp>/config.json "
                  "与 metrics.json）。请先运行训练脚本。")
            return df

        self.to_comparison_csv(df, out_dir / "comparison.csv")
        self.plot_metric_bars(df, out_dir / "comparison_metrics.png")
        self.plot_curves(df, out_dir / "training_curves.png")
        self.plot_confusion_grid(df, out_dir / "confusion_matrices.png")
        self.plot_ablation_bars(df, out_dir / "ablation_comparison.png")

        print("\n[汇总预览]")
        preview = df[["family", "model", "exp", "split_mode", "accuracy", "macro_f1", "weighted_f1"]]
        print(preview.to_string(index=False))
        return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate all experiment results and plot comparisons.")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default=None, help="汇总产物目录，默认与 results_root 相同")
    args = parser.parse_args()

    aggregator = ResultAggregator(args.results_root)
    aggregator.run(out_dir=args.out_dir)


if __name__ == "__main__":
    main()

"""报告图表统一出口：一条命令生成全部报告配图与汇总表。

包含：
1. eda_label_distribution      标签分布（类别不均衡证据）
2. eda_length_distribution     短语长度分布（max_len 决策依据）
3. comparison.csv              全部实验的数值汇总表
4. model_comparison            主对比柱状图（accuracy / macro F1 + 多数类基线）
5. training_curves             训练曲线（loss 与 acc 双面板）
6. per_class_f1                各类别 F1 分组柱状图（少数类短板可视化）
7. error_structure             误差结构分析（混淆占比：邻近 vs 远距错误）
8. confusion_matrices          混淆矩阵拼图（原始计数）
9. attention_heatmap           BiLSTM 注意力可视化（需 model.pt）

用法：
    python scripts/make_figures.py                # 全部
    python scripts/make_figures.py --only eda     # 只生成 eda_* 图
    python scripts/make_figures.py --only main    # 汇总类图与 comparison.csv
    python scripts/make_figures.py --only attention
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
sys.path.insert(0, str(ROOT / "src"))

from common.utils import RESULTS_DIR, TRAIN_PATH

# 中文字体统一样式（Windows 自带微软雅黑，其他系统自动回退）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

LABEL_NAMES = ["负面", "偏负面", "中性", "偏正面", "正面"]
MODEL_COLORS = {"textcnn": "#4C72B0", "bilstm": "#55A868", "bert": "#C44E52",
                "ensemble": "#8172B3"}


# ---------------------------------------------------------------- 数据分析图
def plot_eda(results_dir: Path) -> None:
    df = pd.read_csv(TRAIN_PATH, sep="\t")

    counts = df["Sentiment"].value_counts().sort_index()
    ratios = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(LABEL_NAMES, counts.values, color="#4C72B0")
    for bar, ratio in zip(bars, ratios.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(bar.get_height()):,}\n({ratio:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("短语数量")
    ax.set_title("训练集情感标签分布（类别不均衡：中性约占一半）")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "eda_label_distribution.png", dpi=150)
    plt.close(fig)

    lengths = df["Phrase"].astype(str).str.split().str.len()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(lengths.clip(upper=50), bins=range(0, 52, 2), color="#55A868", edgecolor="white")
    ax.axvline(lengths.mean(), color="#C44E52", ls="--", lw=1.5,
               label=f"平均 {lengths.mean():.1f} 词")
    ax.axvline(lengths.quantile(0.9), color="#8172B3", ls=":", lw=1.5,
               label=f"P90 = {lengths.quantile(0.9):.0f} 词")
    ax.set_xlabel("短语词数（>50 词合并）")
    ax.set_ylabel("短语数量")
    ax.set_title("短语长度分布（支撑 max_len 截断决策）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "eda_length_distribution.png", dpi=150)
    plt.close(fig)
    print("[输出] eda_label_distribution.png / eda_length_distribution.png")


# ---------------------------------------------------------------- 实验数据读取
def load_experiments(results_dir: Path) -> list[dict]:
    """读取全部实验的 metrics 与 config，附带实验显示名。"""
    experiments = []
    for config_path in sorted(results_dir.glob("*/*/*/config.json")):
        exp_dir = config_path.parent
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model, exp = exp_dir.parts[-2], exp_dir.parts[-1]
        display = f"{model}/{exp}"
        experiments.append({"dir": exp_dir, "model": model, "exp": exp,
                            "display": display, "config": config, "metrics": metrics})
    return experiments


def write_comparison_csv(experiments: list[dict], results_dir: Path) -> None:
    """全部实验的数值汇总表（comparison.csv）。"""
    rows = []
    for e in experiments:
        m, c = e["metrics"], e["config"]
        rows.append({
            "family": e["dir"].parts[-3], "model": e["model"], "exp": e["exp"],
            "split_mode": c.get("split_mode"),
            "accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
            "best_epoch": m.get("best_epoch"), "train_seconds": m.get("train_seconds"),
            "params": json.dumps(c.get("params", {}), ensure_ascii=False),
        })
    df = pd.DataFrame(rows).sort_values(["family", "model", "exp"]).reset_index(drop=True)
    df.to_csv(results_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    print(f"[输出] comparison.csv（{len(df)} 个实验）")


def plot_confusion_grid(experiments: list[dict], results_dir: Path,
                        max_cols: int = 3) -> None:
    """各实验混淆矩阵拼图（原始计数）。"""
    show = [e for e in experiments
            if (e["dir"] / "pred_val.csv").exists() and (e["dir"] / "label_val.csv").exists()]
    if not show:
        return
    n = len(show)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.4 * nrows), squeeze=False)
    for i, e in enumerate(show):
        ax = axes[i // ncols][i % ncols]
        y_true = pd.read_csv(e["dir"] / "label_val.csv")["Sentiment"]
        y_pred = pd.read_csv(e["dir"] / "pred_val.csv")["pred"]
        labels = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        label = f"{e['model']}{'(+ctx)' if e['exp'] == 'ctx' else ''}"
        ax.set_title(f"{label}/{e['exp']}", fontsize=10)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=7,
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    print("[输出] confusion_matrices.png")


def pick(experiments: list[dict], model: str, exp: str) -> dict | None:
    for e in experiments:
        if e["model"] == model and e["exp"] == exp:
            return e
    return None


# ---------------------------------------------------------------- 主对比图
def plot_model_comparison(experiments: list[dict], results_dir: Path) -> None:
    show = [e for e in experiments if e["exp"] in ("base", "ctx", "base_grouped", "ensemble")]
    if not show:
        return
    names = [e["display"] for e in show]
    x = np.arange(len(show))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(show)), 5))
    ax.bar(x - width / 2, [e["metrics"]["accuracy"] for e in show], width,
           label="Accuracy", color="#4C72B0")
    ax.bar(x + width / 2, [e["metrics"]["macro_f1"] for e in show], width,
           label="Macro F1", color="#DD8452")
    ax.axhline(0.512, ls="--", color="gray", lw=1, label="多数类基线 0.512")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score")
    ax.set_title("模型对比（stratified 验证集）")
    for xi, e in enumerate(show):
        ax.text(xi - width / 2, e["metrics"]["accuracy"] + 0.01,
                f"{e['metrics']['accuracy']:.3f}", ha="center", fontsize=7)
        ax.text(xi + width / 2, e["metrics"]["macro_f1"] + 0.01,
                f"{e['metrics']['macro_f1']:.3f}", ha="center", fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=150)
    plt.close(fig)
    print("[输出] model_comparison.png")


# ---------------------------------------------------------------- 训练曲线
def plot_training_curves(experiments: list[dict], results_dir: Path) -> None:
    curves = []
    for spec in (("textcnn", "base"), ("bilstm", "base"), ("bert", "base"),
                 ("bilstm", "ctx"), ("bert", "ctx")):
        e = pick(experiments, *spec)
        if e and (e["dir"] / "history.csv").exists():
            curves.append((spec, pd.read_csv(e["dir"] / "history.csv")))
    if not curves:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (model, exp), history in curves:
        label = f"{model}{'(+ctx)' if exp == 'ctx' else ''}"
        style = "--" if exp == "ctx" else "-"
        axes[0].plot(history["epoch"], history["train_loss"], style, marker="o", ms=3,
                     color=MODEL_COLORS.get(model), label=f"{label} train")
        if "val_loss" in history.columns:
            axes[0].plot(history["epoch"], history["val_loss"], style, marker="s", ms=3,
                         color=MODEL_COLORS.get(model), alpha=0.55, label=f"{label} val")
        axes[1].plot(history["epoch"], history["val_acc"], style, marker="o", ms=3,
                     color=MODEL_COLORS.get(model), label=label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("训练 / 验证损失")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("验证精度曲线")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    print("[输出] training_curves.png")


# ---------------------------------------------------------------- 各类别 F1
def plot_per_class_f1(experiments: list[dict], results_dir: Path) -> None:
    show = [e for e in experiments
            if e["exp"] in ("base", "ctx") and "per_class_f1" in e["metrics"]]
    if not show:
        return
    n = len(show)
    width = 0.8 / n
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, e in enumerate(show):
        label = f"{e['model']}{'(+ctx)' if e['exp'] == 'ctx' else ''}"
        ax.bar(x + (i - n / 2 + 0.5) * width, e["metrics"]["per_class_f1"], width,
               label=label, color=MODEL_COLORS.get(e["model"]), alpha=0.85)
    ax.set_xticks(x, LABEL_NAMES)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.95)
    ax.set_title("各类别 F1 对比（两端类别为共同短板——类别不均衡的直接后果）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "per_class_f1.png", dpi=150)
    plt.close(fig)
    print("[输出] per_class_f1.png")


# ---------------------------------------------------------------- 误差结构
def plot_error_structure(experiments: list[dict], results_dir: Path) -> None:
    show = [e for e in experiments if e["exp"] in ("base", "ctx")
            and (e["dir"] / "pred_val.csv").exists()]
    if not show:
        return
    n = len(show)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False)
    distance_names = {1: "邻近类(±1)", 2: "相隔类(±2)", 3: "远距类(±3)", 4: "极远类(±4)"}
    for ax, e in zip(axes[0], show):
        y_true = pd.read_csv(e["dir"] / "label_val.csv")["Sentiment"]
        y_pred = pd.read_csv(e["dir"] / "pred_val.csv")["pred"]
        errors = (y_true - y_pred).abs()
        errors = errors[errors > 0]
        dist_counts = errors.value_counts().sort_index()
        dists = [dist_counts.get(d, 0) for d in (1, 2, 3, 4)]
        colors = ["#C44E52", "#DD8452", "#BCBD22", "#8172B3"]
        bars = ax.bar([distance_names[d] for d in (1, 2, 3, 4)], dists, color=colors)
        for bar, v in zip(bars, dists):
            pct = v / max(len(errors), 1) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{pct:.0f}%",
                    ha="center", va="bottom", fontsize=9)
        label = f"{e['model']}{'(+ctx)' if e['exp'] == 'ctx' else ''}"
        ax.set_title(f"{label}\n错误样本 {len(errors)} 条", fontsize=10)
        ax.set_ylabel("错误条数")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("误差结构：错误集中于相邻情感等级（序数性质），极少跨极混淆", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(results_dir / "error_structure.png", dpi=150)
    plt.close(fig)
    print("[输出] error_structure.png")


# ---------------------------------------------------------------- 注意力图
def plot_attention(results_dir: Path, exp: str = "base", num_per_label: int = 2) -> None:
    import json

    import torch

    from common.dl_data import Vocab
    from common.preprocess import prepare_dataframe
    from common.split import ensure_split
    from common.utils import load_data, set_seed
    from models.deep_learning.bilstm import BiLSTMClassifier

    exp_dir = results_dir / "dl" / "bilstm" / exp
    weights_path = exp_dir / "model.pt"
    if not weights_path.exists():
        print(f"[跳过] {weights_path} 不存在（该实验未保存权重）")
        return
    config = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    params = config["params"]
    if params.get("pooling") != "attention" or params.get("use_context"):
        print("[跳过] 仅支持普通单通道 attention 模型的权重")
        return
    set_seed(params.get("seed", 42))

    train_df, _ = load_data()
    train_df = prepare_dataframe(train_df)
    train_part, val_part = ensure_split(train_df, mode=params.get("mode", "stratified"),
                                        val_ratio=params.get("val_ratio", 0.1),
                                        seed=params.get("seed", 42))
    vocab = Vocab.build(train_part["Phrase"], min_freq=params.get("min_freq", 2))
    model = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=params.get("embed_dim", 128),
        hidden_size=params.get("hidden_size", 128), num_layers=params.get("num_layers", 1),
        bidirectional=bool(params.get("bidirectional", 1)), pooling="attention",
        dropout=params.get("dropout", 0.5),
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    max_len = params.get("max_len", 48)

    samples = []
    for label in sorted(val_part["Sentiment"].unique()):
        pool = val_part[val_part["Sentiment"] == label]
        lens = pool["Phrase"].str.split().str.len()
        pool = pool[(lens >= 5) & (lens <= 16)]
        kept = 0
        for _, row in pool.iterrows():
            if kept >= num_per_label:
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
            samples.append((row["Phrase"], label, ids, valid_len))
            kept += 1
    if not samples:
        print("[跳过] 未找到分类正确的干净样本")
        return

    ncols = 4
    nrows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    for i, (phrase, label, ids, valid_len) in enumerate(samples):
        ax = axes[i // ncols][i % ncols]
        with torch.no_grad():
            model(torch.from_numpy(ids[:, :valid_len]))
        alpha = model.attention.last_alpha[0].numpy()
        tokens = phrase.split()
        order = alpha.argsort()
        ax.barh(range(valid_len), alpha[order], color="#4C72B0")
        ax.set_yticks(range(valid_len), [tokens[j] for j in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"真实/预测一致: {LABEL_NAMES[label]}", fontsize=9, color="#2E7D32")
        ax.tick_params(axis="x", labelsize=7)
    for j in range(len(samples), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("BiLSTM 注意力权重（分类正确样本；横条越长该词贡献越大）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(results_dir / f"attention_heatmap_{exp}.png", dpi=150)
    plt.close(fig)
    print(f"[输出] attention_heatmap_{exp}.png")


# --------------------------------------------------------------------- 入口
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all report figures.")
    parser.add_argument("--only", type=str, default="all",
                        choices=("all", "eda", "main", "attention"),
                        help="eda=数据图；main=依赖实验结果的汇总图；attention=注意力图")
    parser.add_argument("--attention_exp", type=str, default="base")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "eda"):
        plot_eda(RESULTS_DIR)
    if args.only in ("all", "main"):
        experiments = load_experiments(RESULTS_DIR)
        write_comparison_csv(experiments, RESULTS_DIR)
        plot_model_comparison(experiments, RESULTS_DIR)
        plot_training_curves(experiments, RESULTS_DIR)
        plot_per_class_f1(experiments, RESULTS_DIR)
        plot_error_structure(experiments, RESULTS_DIR)
        plot_confusion_grid(experiments, RESULTS_DIR)
    if args.only in ("all", "attention"):
        plot_attention(RESULTS_DIR, exp=args.attention_exp)


if __name__ == "__main__":
    main()

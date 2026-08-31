from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation and visualization entrypoint.")
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series):
    """统一评估指标。"""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=sorted(y_true.unique()))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
    }


def plot_confusion_matrix(cm: pd.DataFrame, labels: list[int], save_path: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = pd.read_csv(args.label_path)
    y_pred = pd.read_csv(args.pred_path)

    if "Sentiment" in y_true.columns:
        y_true = y_true["Sentiment"]
    if "pred" in y_pred.columns:
        y_pred = y_pred["pred"]

    metrics = evaluate_predictions(y_true, y_pred)
    print(metrics)

    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    plot_confusion_matrix(pd.DataFrame(cm), sorted(y_true.unique()), str(output_dir / "confusion_matrix.png"))
    print("Evaluation report generated.")


if __name__ == "__main__":
    main()

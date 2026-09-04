"""统一评估指标：所有训练脚本与集成脚本共用。"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def evaluate_predictions(y_true, y_pred):
    """统一评估指标，输出完整的多分类指标字典。"""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    labels = sorted(y_true.unique())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                                labels=labels, zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": [int(v) for v in labels],
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }

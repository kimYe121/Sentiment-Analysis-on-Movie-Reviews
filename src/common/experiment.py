"""实验产物统一读写模块（所有训练脚本的落盘契约）。

每个实验在 ``results/<family>/<model>/<exp_name>/`` 下产出固定文件：

- ``config.json``      实验全部超参与环境信息，报告"实验设置"直接引用
- ``metrics.json``     验证集最终指标（accuracy / macro_f1 / weighted_f1 / 各类别 P/R/F1）
- ``history.csv``      逐轮训练记录（经典模型没有轮次概念时可省略）
- ``pred_val.csv``     验证集预测，列: PhraseId, pred
- ``label_val.csv``    验证集真实标签，列: PhraseId, Sentiment
- ``submission.csv``   Kaggle 测试集提交文件，列: PhraseId, Sentiment

``src/evaluation/aggregator.py`` 依据该契约自动扫描并汇总所有实验，
训练脚本因此不需要关心画图与对比逻辑。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from common.utils import RESULTS_DIR, ensure_dirs


def _library_versions() -> dict:
    """记录关键库版本，写入 config.json 便于复现。"""
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "sklearn", "torch", "transformers"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions


class ExperimentLogger:
    """负责一个实验目录的创建与全部产物写入。"""

    def __init__(
        self,
        family: str,
        model: str,
        exp_name: str,
        params: dict | None = None,
        split_mode: str | None = None,
        seed: int | None = None,
        results_root: str | Path | None = None,
    ) -> None:
        ensure_dirs()
        root = Path(results_root) if results_root is not None else RESULTS_DIR
        self.exp_dir = root / family / model / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            "family": family,
            "model": model,
            "exp_name": exp_name,
            "split_mode": split_mode,
            "seed": seed,
            "params": params or {},
            "library_versions": _library_versions(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._write_config()

    # ------------------------------------------------------------------ config
    def set_data_info(self, train_size: int, val_size: int, test_size: int | None = None) -> None:
        self.config["train_size"] = train_size
        self.config["val_size"] = val_size
        if test_size is not None:
            self.config["test_size"] = test_size
        self._write_config()

    def _write_config(self) -> None:
        (self.exp_dir / "config.json").write_text(
            json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ----------------------------------------------------------------- history
    def save_history(self, history: pd.DataFrame | list[dict]) -> None:
        """逐轮训练记录；经典模型无轮次记录时不调用即可。"""
        df = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
        df.to_csv(self.exp_dir / "history.csv", index=False)

    # ----------------------------------------------------------------- metrics
    def save_metrics(self, metrics: dict) -> None:
        (self.exp_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------ predictions
    def save_predictions(self, phrase_ids, y_true, y_pred) -> None:
        pd.DataFrame({"PhraseId": phrase_ids, "pred": y_pred}).to_csv(
            self.exp_dir / "pred_val.csv", index=False
        )
        pd.DataFrame({"PhraseId": phrase_ids, "Sentiment": y_true}).to_csv(
            self.exp_dir / "label_val.csv", index=False
        )

    def save_submission(self, phrase_ids, y_pred) -> None:
        pd.DataFrame({"PhraseId": phrase_ids, "Sentiment": y_pred}).to_csv(
            self.exp_dir / "submission.csv", index=False
        )

    def save_model(self, state_dict) -> None:
        """保存最优模型权重，用于推理复现与可解释性分析（如注意力可视化）。"""
        torch.save(state_dict, self.exp_dir / "model.pt")

    def print_summary(self, metrics: dict) -> None:
        print(f"\n[实验目录] {self.exp_dir}")
        print(f"[最终指标] accuracy={metrics['accuracy']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  weighted_f1={metrics['weighted_f1']:.4f}")

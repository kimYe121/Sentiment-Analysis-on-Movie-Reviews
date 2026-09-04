"""概率平均集成：把多个实验保存的 softmax 概率矩阵平均，生成一个新实验。

集成学习（Ensemble）原理：不同模型犯的错误不同，对预测概率取平均可以
互相纠正，通常优于任何单个模型（参考文献：Zhou Z H. Ensemble Methods:
Foundations and Algorithms[M]. Boca Raton: CRC Press, 2012.）。

用法：
    python scripts/ensemble.py                                    # 默认：bert/base + bilstm/base + textcnn/base
    python scripts/ensemble.py dl/bert/ctx dl/bilstm/ctx dl/textcnn/base
产物：results/dl/ensemble/<exp_name>/（与训练实验同契约，自动进汇总表）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.experiment import ExperimentLogger
from common.utils import RESULTS_DIR
from common.metrics import evaluate_predictions


def load_experiment(results_dir: Path, spec: str) -> dict:
    """spec 形如 dl/bert/base，读取其概率矩阵与配置。"""
    exp_dir = results_dir / spec
    probs_path = exp_dir / "probs_val.npy"
    if not probs_path.exists():
        raise FileNotFoundError(f"{probs_path} 不存在（该实验未保存概率矩阵）")
    config = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    return {
        "spec": spec,
        "dir": exp_dir,
        "probs_val": np.load(probs_path),
        "probs_test": (np.load(exp_dir / "probs_test.npy")
                       if (exp_dir / "probs_test.npy").exists() else None),
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probability-averaging ensemble.")
    parser.add_argument("specs", nargs="*", default=["dl/bert/base", "dl/bilstm/base", "dl/textcnn/base"],
                        help="参与集成的实验（family/model/exp），默认三个 base")
    parser.add_argument("--exp_name", type=str, default="avg",
                        help="集成实验名，目录为 results/dl/ensemble/<exp_name>")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    members = [load_experiment(RESULTS_DIR, spec) for spec in args.specs]
    print(f"[集成] 成员: {[m['spec'] for m in members]}")

    label_ref = None
    for m in members:
        label_path = m["dir"] / "label_val.csv"
        y = pd.read_csv(label_path)["Sentiment"].to_numpy()
        if label_ref is None:
            label_ref = y
        elif not np.array_equal(label_ref, y):
            raise ValueError(f"{m['spec']} 的验证集标签与第一个成员不一致，无法集成")
    phrase_ids = pd.read_csv(members[0]["dir"] / "label_val.csv")["PhraseId"]

    probs_avg = np.mean([m["probs_val"] for m in members], axis=0)
    val_pred = probs_avg.argmax(axis=1)
    metrics = evaluate_predictions(pd.Series(label_ref), pd.Series(val_pred))

    # 与单模型的对比
    print("[对比]")
    for m in members:
        m_metrics = json.loads((m["dir"] / "metrics.json").read_text(encoding="utf-8"))
        print(f"  {m['spec']}: acc={m_metrics['accuracy']:.4f}  macro_f1={m_metrics['macro_f1']:.4f}")
    print(f"  集成   : acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")

    logger = ExperimentLogger(
        family="dl", model="ensemble", exp_name=args.exp_name,
        params={"members": args.specs, "method": "probability_averaging"},
        split_mode=members[0]["config"].get("split_mode"), seed=args.seed,
    )
    logger.set_data_info(members[0]["config"].get("train_size", 0),
                         len(label_ref),
                         members[0]["config"].get("test_size", 0))
    logger.save_metrics(metrics)
    logger.save_predictions(phrase_ids, label_ref, val_pred)

    test_probs = [m["probs_test"] for m in members if m["probs_test"] is not None]
    if test_probs:
        probs_test_avg = np.mean(test_probs, axis=0)
        test_pred = probs_test_avg.argmax(axis=1)
        # 用第一个有 test 概率的成员的 submission 对齐 PhraseId
        for m in members:
            if m["probs_test"] is not None:
                sub = pd.read_csv(m["dir"] / "submission.csv")
                logger.save_submission(sub["PhraseId"], test_pred)
                break
    logger.print_summary(metrics)


if __name__ == "__main__":
    main()

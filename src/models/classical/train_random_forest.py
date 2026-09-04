"""TF-IDF + Random Forest —— 经典机器学习模型。

参照 train_logistic_regression.py 模板实现，保证与其余经典/深度学习模型可比。

⚠️ 注意：TF-IDF 特征维度高（默认 5 万），Random Forest 在稀疏高维特征上训练较慢，
建议先用 ``--max_samples 20000`` 做快速调试，再在全量数据上跑最终实验。

关键超参：
    --n_estimators      森林中树的数量（默认 100，增大可提分但更慢）
    --max_depth         单棵树的最大深度（默认 None 不限制，设小可加速+防过拟合）
    --min_samples_split 内部节点再划分所需最小样本数（默认 2）
    --rf_max_features   每棵树随机采样的特征比例（默认 "sqrt"，可选 "log2" 或 0~1 浮点数）
    --n_jobs            并行训练线程数（默认 -1 利用全部 CPU 核心）

运行示例：
    # 全量训练
    python src/models/classical/train_random_forest.py --exp_name base
    # 快速调试（~1/7 数据）
    python src/models/classical/train_random_forest.py --max_samples 20000 --n_estimators 50 --exp_name debug
    # 限制深度加速
    python src/models/classical/train_random_forest.py --max_depth 30 --exp_name d30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment import ExperimentLogger
from common.preprocess import prepare_dataframe
from common.split import ensure_split
from common.utils import ensure_dirs, load_data, set_seed
from common.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF + Random Forest.")
    parser.add_argument("--exp_name", type=str, default="base")
    parser.add_argument("--mode", type=str, default="stratified", choices=("stratified", "grouped"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # -- RF 模型超参 --
    parser.add_argument("--n_estimators", type=int, default=100, help="森林中树的数量")
    parser.add_argument("--max_depth", type=int, default=0,
                        help="单棵树最大深度，0 表示不限制 (None)")
    parser.add_argument("--min_samples_split", type=int, default=2,
                        help="内部节点再划分所需最小样本数")
    parser.add_argument("--rf_max_features", type=str, default="sqrt",
                        help="每棵树随机采样的特征数 (sqrt/log2/数字/None)")
    parser.add_argument("--n_jobs", type=int, default=-1, help="并行训练线程数，-1 表示全部核心")
    # -- TF-IDF 超参 --
    parser.add_argument("--max_features", type=int, default=50000, help="TF-IDF 最大特征数")
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--min_df", type=int, default=2, help="最小文档频率")
    parser.add_argument("--max_samples", type=int, default=0, help="调试用：>0 时只抽取训练子集")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()

    # -------------------------------------------------- 数据读取与统一划分
    train_df, test_df = load_data()
    train_df = prepare_dataframe(train_df)      # 统一文本清洗
    test_df = prepare_dataframe(test_df)
    train_part, val_part = ensure_split(train_df, mode=args.mode,
                                        val_ratio=args.val_ratio, seed=args.seed)
    if args.max_samples > 0:
        train_part = train_part.sample(n=min(args.max_samples, len(train_part)),
                                       random_state=args.seed).reset_index(drop=True)
    print(f"[数据] train={len(train_part)}  val={len(val_part)}  test={len(test_df)}")

    logger = ExperimentLogger(family="classical", model="random_forest",
                              exp_name=args.exp_name, params=vars(args),
                              split_mode=args.mode, seed=args.seed)
    logger.set_data_info(len(train_part), len(val_part), len(test_df))

    # -------------------------------------------------- 特征与训练
    t0 = time.time()
    vectorizer = TfidfVectorizer(ngram_range=(args.ngram_min, args.ngram_max),
                                 max_features=args.max_features, min_df=args.min_df)
    x_train = vectorizer.fit_transform(train_part["Phrase"])
    x_val = vectorizer.transform(val_part["Phrase"])

    # RF 的 max_depth=None 表示不限制；用 0 作为 "未设置" 哨兵，避免 argparse 传 None 麻烦
    max_depth = None if args.max_depth == 0 else args.max_depth
    # max_features 的字符串参数保持原样传；sklearn 会自行解析 "sqrt" / "log2" / "None"
    mf_raw = args.rf_max_features
    if mf_raw.lower() in ("none", "null"):
        mf = None
    else:
        try:
            mf = float(mf_raw)
        except ValueError:
            mf = mf_raw   # "sqrt" 或 "log2"

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_split=args.min_samples_split,
        max_features=mf,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    model.fit(x_train, train_part["Sentiment"])
    train_seconds = round(time.time() - t0, 1)

    # -------------------------------------------------- 评估与落盘
    val_pred = model.predict(x_val)
    metrics = evaluate_predictions(pd.Series(val_part["Sentiment"]), pd.Series(val_pred))
    metrics.update({"train_seconds": train_seconds})
    logger.save_metrics(metrics)
    logger.save_predictions(val_part["PhraseId"], val_part["Sentiment"], val_pred)

    test_pred = model.predict(vectorizer.transform(test_df["Phrase"]))
    logger.save_submission(test_df["PhraseId"], test_pred)
    logger.print_summary(metrics)


if __name__ == "__main__":
    main()

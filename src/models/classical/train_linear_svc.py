"""TF-IDF + Linear SVC —— 经典机器学习模型。

参照 train_logistic_regression.py 模板实现，保证与其余经典/深度学习模型可比。

关键超参：
    --C        正则化强度的倒数（越小正则越强）
    --dual     是否求解对偶问题（样本数 > 特征数时建议 False，本任务默认 False）
    --max_iter 迭代次数上限（SVC 收敛较慢，默认 20000）

运行示例：
    python src/models/classical/train_linear_svc.py --exp_name base
    python src/models/classical/train_linear_svc.py --C 0.5 --exp_name C0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment import ExperimentLogger
from common.preprocess import prepare_dataframe
from common.split import ensure_split
from common.utils import ensure_dirs, load_data, set_seed
from common.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF + Linear SVC.")
    parser.add_argument("--exp_name", type=str, default="base")
    parser.add_argument("--mode", type=str, default="stratified", choices=("stratified", "grouped"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="正则化强度的倒数")
    parser.add_argument("--dual", type=lambda x: x.lower() != "false", default=False,
                        help="是否求解对偶问题 (True/False)")
    parser.add_argument("--max_iter", type=int, default=20000, help="迭代次数上限")
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

    logger = ExperimentLogger(family="classical", model="linear_svc",
                              exp_name=args.exp_name, params=vars(args),
                              split_mode=args.mode, seed=args.seed)
    logger.set_data_info(len(train_part), len(val_part), len(test_df))

    # -------------------------------------------------- 特征与训练
    t0 = time.time()
    vectorizer = TfidfVectorizer(ngram_range=(args.ngram_min, args.ngram_max),
                                 max_features=args.max_features, min_df=args.min_df)
    x_train = vectorizer.fit_transform(train_part["Phrase"])
    x_val = vectorizer.transform(val_part["Phrase"])

    model = LinearSVC(C=args.C, dual=args.dual, max_iter=args.max_iter)
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

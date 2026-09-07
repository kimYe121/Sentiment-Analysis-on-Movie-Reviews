"""TF-IDF + Naive Bayes —— 经典机器学习模型（Multinomial / Complement 对照）。

按照 train_logistic_regression.py 模板的实验契约实现：

1. 从 common.split.ensure_split 获取统一训练/验证划分（不要自己切分！）
2. 用 common.experiment.ExperimentLogger 落盘实验产物（契约见该文件 docstring）
3. 用 common.metrics.evaluate_predictions 计算统一指标
4. 生成 Kaggle 提交文件 submission.csv
 
ComplementNB 是 MultinomialNB 的不平衡变体：本数据集极端情感（0/4）样本
远少于中性（2），ComplementNB 通常对稀少类的 macro-F1 更好。

运行示例：
    python src/models/classical/train_multinomial_nb.py --exp_name base
    python src/models/classical/train_multinomial_nb.py --alpha 0.3 --exp_name a03
    python src/models/classical/train_multinomial_nb.py --variant complement --exp_name cnb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment import ExperimentLogger
from common.preprocess import combine_phrase_and_sentence, prepare_dataframe
from common.split import ensure_split
from common.utils import ensure_dirs, load_data, set_seed
from common.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF + Naive Bayes (Multinomial / Complement).")
    parser.add_argument("--exp_name", type=str, default="base")
    parser.add_argument("--mode", type=str, default="stratified", choices=("stratified", "grouped"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5, help="拉普拉斯平滑系数，NB 最关键超参")
    parser.add_argument("--variant", type=str, default="multinomial",
                        choices=("multinomial", "complement"),
                        help="multinomial=标准多项式NB；complement=类别不平衡变体")
    parser.add_argument("--vectorizer", type=str, default="tfidf",
                        choices=("tfidf", "counts"),
                        help="特征表示：tfidf 与 LR 可比；counts 为原始词频，NB 理论上更匹配"
                             "（tf-idf 值远小于 1，alpha 平滑相对过强）")
    parser.add_argument("--max_features", type=int, default=100000, help="TF-IDF 最大特征数")
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--min_df", type=int, default=2, help="最小文档频率")
    parser.add_argument("--text_field", type=str, default="phrase",
                        choices=("phrase", "phrase_sentence"),
                        help="特征文本：phrase=仅短语（真实数据上更优）；"
                             "phrase_sentence=拼接句子上下文，用作消融对照")
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
    if args.text_field == "phrase_sentence":
        # 消融对照：同句短语共享相同的句子上下文（无泄漏，但真实数据上弱于纯短语）
        train_df = combine_phrase_and_sentence(train_df)
        test_df = combine_phrase_and_sentence(test_df)
        text_col = "phrase_sentence_text"
    else:
        text_col = "Phrase"
    train_part, val_part = ensure_split(train_df, mode=args.mode,
                                        val_ratio=args.val_ratio, seed=args.seed)
    if args.max_samples > 0:
        train_part = train_part.sample(n=min(args.max_samples, len(train_part)),
                                       random_state=args.seed).reset_index(drop=True)
    print(f"[数据] train={len(train_part)}  val={len(val_part)}  test={len(test_df)}  "
          f"特征文本={text_col}")

    logger = ExperimentLogger(family="classical", model=f"{args.variant}_nb",
                              exp_name=args.exp_name, params=vars(args),
                              split_mode=args.mode, seed=args.seed)
    logger.set_data_info(len(train_part), len(val_part), len(test_df))

    # -------------------------------------------------- 特征与训练
    t0 = time.time()
    if args.vectorizer == "counts":
        vectorizer = CountVectorizer(ngram_range=(args.ngram_min, args.ngram_max),
                                     max_features=args.max_features, min_df=args.min_df,
                                     dtype=np.float32)
    else:
        vectorizer = TfidfVectorizer(ngram_range=(args.ngram_min, args.ngram_max),
                                     max_features=args.max_features, min_df=args.min_df,
                                     sublinear_tf=True, dtype=np.float32)
    x_train = vectorizer.fit_transform(train_part[text_col])
    x_val = vectorizer.transform(val_part[text_col])
    x_test = vectorizer.transform(test_df[text_col])

    nb_cls = MultinomialNB if args.variant == "multinomial" else ComplementNB
    model = nb_cls(alpha=args.alpha)
    model.fit(x_train, train_part["Sentiment"])
    train_seconds = round(time.time() - t0, 1)

    # -------------------------------------------------- 评估与落盘
    val_pred = model.predict(x_val)
    metrics = evaluate_predictions(pd.Series(val_part["Sentiment"]), pd.Series(val_pred))
    metrics.update({"train_seconds": train_seconds})
    logger.save_metrics(metrics)
    logger.save_predictions(val_part["PhraseId"], val_part["Sentiment"], val_pred)
    # 保存概率矩阵供 scripts/ensemble.py 概率平均集成（列顺序 = model.classes_ = [0..4]）
    logger.save_probs(model.predict_proba(x_val), model.predict_proba(x_test))

    test_pred = model.predict(x_test)
    logger.save_submission(test_df["PhraseId"], test_pred)
    logger.print_summary(metrics)


if __name__ == "__main__":
    main()

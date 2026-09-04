"""TextCNN 训练脚本：模型与数据管线均为手写实现（复现 Kim, 2014, EMNLP）。

运行方式（二选一）：
    # 编排脚本批量跑（推荐，自动跳过已完成实验）
    python scripts/run_experiments.py --group base
    # 单独运行本脚本
    python src/models/deep_learning/train_textcnn.py --exp_name base

其他参数示例：
    python src/models/deep_learning/train_textcnn.py --num_filters 64 --exp_name filters64
    python src/models/deep_learning/train_textcnn.py --max_samples 20000 --epochs 2   # 快速调试
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dl_data import BatchIterator, Vocab
from common.dl_train import predict_probs, run_training
from common.experiment import ExperimentLogger
from common.preprocess import prepare_dataframe
from common.split import ensure_split
from common.utils import ensure_dirs, load_data, set_seed
from common.metrics import evaluate_predictions
from models.deep_learning.layers import run_component_checks
from models.deep_learning.textcnn import TextCNN, verify_manual_conv1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hand-written TextCNN model.")
    parser.add_argument("--exp_name", type=str, default="base", help="实验名，对应结果子目录")
    parser.add_argument("--mode", type=str, default="stratified", choices=("stratified", "grouped"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--kernel_sizes", type=str, default="3,4,5", help="逗号分隔的卷积核宽度")
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW 解耦权重衰减")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑系数")
    parser.add_argument("--min_freq", type=int, default=2, help="词表最小词频")
    parser.add_argument("--max_len", type=int, default=48)
    parser.add_argument("--max_samples", type=int, default=0, help="调试用：>0 时只抽取训练子集")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=6, help="早停轮数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[环境] device={device}")

    run_component_checks()
    print(f"[组件验证] ManualConv1d vs nn.Conv1d   max|diff| = {verify_manual_conv1d():.2e}")

    # -------------------------------------------------- 数据读取与统一划分
    train_df, test_df = load_data()
    train_df = prepare_dataframe(train_df)
    test_df = prepare_dataframe(test_df)
    train_part, val_part = ensure_split(train_df, mode=args.mode,
                                        val_ratio=args.val_ratio, seed=args.seed)
    if args.max_samples > 0:
        train_part = train_part.sample(n=min(args.max_samples, len(train_part)),
                                       random_state=args.seed).reset_index(drop=True)
    print(f"[数据] train={len(train_part)}  val={len(val_part)}  test={len(test_df)}")

    # -------------------------------------------------- 手写词表与编码
    vocab = Vocab.build(train_part["Phrase"], min_freq=args.min_freq)
    print(f"[词表] 词表大小={len(vocab)}  min_freq={args.min_freq}")
    x_train = vocab.encode(train_part["Phrase"], args.max_len)
    x_val = vocab.encode(val_part["Phrase"], args.max_len)
    y_train = train_part["Sentiment"].to_numpy()
    y_val = val_part["Sentiment"].to_numpy()

    logger = ExperimentLogger(family="dl", model="textcnn", exp_name=args.exp_name,
                              params=vars(args), split_mode=args.mode, seed=args.seed)
    logger.set_data_info(len(train_part), len(val_part), len(test_df))

    # -------------------------------------------------- 训练
    model = TextCNN(vocab_size=len(vocab), embed_dim=args.embed_dim,
                    kernel_sizes=tuple(int(k) for k in args.kernel_sizes.split(",")),
                    num_filters=args.num_filters, dropout=args.dropout).to(device)
    # AdamW：权重衰减与梯度自适应尺度解耦，正则化作用更可控
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    train_iter = BatchIterator(x_train, y_train, args.batch_size, shuffle=True, seed=args.seed)

    t0 = time.time()
    history, best_epoch = run_training(model, optimizer, train_iter, x_val, y_val,
                                       device, epochs=args.epochs,
                                       grad_clip=args.grad_clip, patience=args.patience,
                                       label_smoothing=args.label_smoothing,
                                       lr_schedule="cosine")
    train_seconds = round(time.time() - t0, 1)
    logger.save_history(history)
    logger.save_model(model.state_dict())

    # -------------------------------------------------- 评估与落盘
    val_probs = predict_probs(model, x_val, device)
    val_pred = val_probs.argmax(axis=1)
    metrics = evaluate_predictions(pd.Series(y_val), pd.Series(val_pred))
    metrics.update({"best_epoch": best_epoch, "train_seconds": train_seconds})
    logger.save_metrics(metrics)
    logger.save_predictions(val_part["PhraseId"], y_val, val_pred)

    x_test = vocab.encode(test_df["Phrase"], args.max_len)
    test_probs = predict_probs(model, x_test, device)
    logger.save_probs(val_probs, test_probs)
    logger.save_submission(test_df["PhraseId"], test_probs.argmax(axis=1))
    logger.print_summary(metrics)


if __name__ == "__main__":
    main()

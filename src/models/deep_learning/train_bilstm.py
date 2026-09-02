"""BiLSTM 训练脚本：LSTM 单元、双向展开与注意力池化均为手写实现。

示例：
    python src/models/deep_learning/train_bilstm.py --exp_name base
    python src/models/deep_learning/train_bilstm.py --pooling mean --exp_name pool_mean
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
from common.dl_train import predict, run_training
from common.experiment import ExperimentLogger
from common.preprocess import prepare_dataframe
from common.split import ensure_split
from common.utils import ensure_dirs, load_data, set_seed
from evaluation.evaluate import evaluate_predictions
from models.deep_learning.bilstm import BiLSTMClassifier, benchmark_vs_nn_lstm, verify_manual_bilstm
from models.deep_learning.layers import run_component_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hand-written BiLSTM model.")
    parser.add_argument("--exp_name", type=str, default="base")
    parser.add_argument("--mode", type=str, default="stratified", choices=("stratified", "grouped"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=int, default=1, help="1=双向，0=单向")
    parser.add_argument("--pooling", type=str, default="attention",
                        choices=("attention", "mean", "last"))
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=48)
    parser.add_argument("--max_samples", type=int, default=0, help="调试用：>0 时只抽取训练子集")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[环境] device={device}")

    run_component_checks()
    print(f"[组件验证] 手写BiLSTM vs nn.LSTM   max|diff| = {verify_manual_bilstm():.2e}")
    bench = benchmark_vs_nn_lstm()
    print(f"[性能对比] 单批前向耗时(64×48): 手写BiLSTM {bench['manual_ms']:.2f} ms  "
          f"vs nn.LSTM(cuDNN) {bench['nn_ms']:.2f} ms  ({bench['ratio']:.1f}x)")

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

    logger = ExperimentLogger(family="dl", model="bilstm", exp_name=args.exp_name,
                              params=vars(args), split_mode=args.mode, seed=args.seed)
    logger.set_data_info(len(train_part), len(val_part), len(test_df))

    # -------------------------------------------------- 训练
    model = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=args.embed_dim,
                             hidden_size=args.hidden_size, num_layers=args.num_layers,
                             bidirectional=bool(args.bidirectional), pooling=args.pooling,
                             dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_iter = BatchIterator(x_train, y_train, args.batch_size, shuffle=True, seed=args.seed)

    t0 = time.time()
    history, best_epoch = run_training(model, optimizer, train_iter, x_val, y_val,
                                       device, epochs=args.epochs,
                                       grad_clip=args.grad_clip, patience=args.patience)
    train_seconds = round(time.time() - t0, 1)
    logger.save_history(history)

    # -------------------------------------------------- 评估与落盘
    _, val_pred = predict(model, x_val, device)
    metrics = evaluate_predictions(pd.Series(y_val), pd.Series(val_pred))
    metrics.update({"best_epoch": best_epoch, "train_seconds": train_seconds})
    logger.save_metrics(metrics)
    logger.save_predictions(val_part["PhraseId"], y_val, val_pred)

    x_test = vocab.encode(test_df["Phrase"], args.max_len)
    _, test_pred = predict(model, x_test, device)
    logger.save_submission(test_df["PhraseId"], test_pred)
    logger.print_summary(metrics)


if __name__ == "__main__":
    main()

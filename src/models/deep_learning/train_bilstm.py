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
from common.dl_train import predict, predict_probs, run_training
from common.experiment import ExperimentLogger
from common.preprocess import prepare_dataframe
from common.split import ensure_split
from common.utils import ensure_dirs, load_data, set_seed
from common.metrics import evaluate_predictions
from models.deep_learning.bilstm import BiLSTMClassifier, ContextBiLSTMClassifier, benchmark_vs_nn_lstm, verify_manual_bilstm
from models.deep_learning.layers import run_component_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hand-written BiLSTM model.")
    parser.add_argument("--exp_name", type=str, default="base")
    parser.add_argument("--mode", type=str, default="stratified", choices=("stratified", "grouped"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=int, default=1, help="1=双向，0=单向")
    parser.add_argument("--pooling", type=str, default="attention",
                        choices=("attention", "mean", "last"))
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW 解耦权重衰减")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑系数")
    parser.add_argument("--impl", type=str, default="manual", choices=("manual", "nn"),
                        help="循环核心实现：manual=手写（正式结果），nn=nn.LSTM（性能对照实验）")
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=48)
    parser.add_argument("--max_samples", type=int, default=0, help="调试用：>0 时只抽取训练子集")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_context", action="store_true",
                        help="启用双通道上下文融合（短语 + 所在完整句子）")
    parser.add_argument("--ctx_max_len", type=int, default=48, help="上下文序列最大长度")
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
    # 上下文融合模式：词表需覆盖上下文中的词（句子全集大于短语词集）
    if args.use_context:
        vocab_text = pd.concat([train_part["Phrase"], train_part["sentence_context"]],
                               ignore_index=True)
    else:
        vocab_text = train_part["Phrase"]
    vocab = Vocab.build(vocab_text, min_freq=args.min_freq)
    print(f"[词表] 词表大小={len(vocab)}  min_freq={args.min_freq}")
    x_train = vocab.encode(train_part["Phrase"], args.max_len)
    x_val = vocab.encode(val_part["Phrase"], args.max_len)
    y_train = train_part["Sentiment"].to_numpy()
    y_val = val_part["Sentiment"].to_numpy()
    if args.use_context:
        # 干净上下文：每句最长短语重建完整句子（覆盖 prepare_dataframe 的拼接版）
        from common.preprocess import build_sentence_context
        train_part = build_sentence_context(train_part)
        val_part = build_sentence_context(val_part)
        test_df = build_sentence_context(test_df)
        x_ctx_train = vocab.encode(train_part["sentence_context"], args.ctx_max_len)
        x_ctx_val = vocab.encode(val_part["sentence_context"], args.ctx_max_len)
        print(f"[上下文] 双通道融合（最长短语重建）：短语 max_len={args.max_len}，上下文 max_len={args.ctx_max_len}")
    else:
        x_ctx_train = x_ctx_val = None

    logger = ExperimentLogger(family="dl", model="bilstm", exp_name=args.exp_name,
                              params=vars(args), split_mode=args.mode, seed=args.seed)
    logger.set_data_info(len(train_part), len(val_part), len(test_df))

    # -------------------------------------------------- 训练
    print(f"[实现] 循环核心 = {args.impl}" + ("（性能对照实验，正式结果请用 manual）" if args.impl == "nn" else ""))
    if args.use_context:
        model = ContextBiLSTMClassifier(vocab_size=len(vocab), embed_dim=args.embed_dim,
                                        hidden_size=args.hidden_size,
                                        dropout=args.dropout).to(device)
    else:
        model = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=args.embed_dim,
                                 hidden_size=args.hidden_size, num_layers=args.num_layers,
                                 bidirectional=bool(args.bidirectional), pooling=args.pooling,
                                 dropout=args.dropout, recurrent_impl=args.impl).to(device)
    # AdamW：权重衰减与梯度自适应尺度解耦，正则化作用更可控
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    train_iter = BatchIterator(x_train, y_train, args.batch_size, shuffle=True, seed=args.seed,
                               context_ids=x_ctx_train)

    t0 = time.time()
    history, best_epoch = run_training(model, optimizer, train_iter, x_val, y_val,
                                       device, epochs=args.epochs,
                                       grad_clip=args.grad_clip, patience=args.patience,
                                       label_smoothing=args.label_smoothing,
                                       lr_schedule="cosine",
                                       context_val_ids=x_ctx_val)
    train_seconds = round(time.time() - t0, 1)
    logger.save_history(history)
    if not args.use_context:
        logger.save_model(model.state_dict())

    # -------------------------------------------------- 评估与落盘
    val_probs = predict_probs(model, x_val, device, context_ids=x_ctx_val)
    val_pred = val_probs.argmax(axis=1)
    metrics = evaluate_predictions(pd.Series(y_val), pd.Series(val_pred))
    metrics.update({"best_epoch": best_epoch, "train_seconds": train_seconds})
    logger.save_metrics(metrics)
    logger.save_predictions(val_part["PhraseId"], y_val, val_pred)

    x_test = vocab.encode(test_df["Phrase"], args.max_len)
    x_ctx_test = vocab.encode(test_df["sentence_context"], args.ctx_max_len) if args.use_context else None
    test_probs = predict_probs(model, x_test, device, context_ids=x_ctx_test)
    logger.save_probs(val_probs, test_probs)
    logger.save_submission(test_df["PhraseId"], test_probs.argmax(axis=1))
    logger.print_summary(metrics)


if __name__ == "__main__":
    main()

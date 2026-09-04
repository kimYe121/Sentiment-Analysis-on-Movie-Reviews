"""BERT 微调脚本（复现: Devlin et al., 2019, NAACL 的微调协议）。

定位说明：BERT 预训练权重与 tokenizer 来自 HuggingFace transformers，
是项目中的"复现发表方法"对照项；但训练循环、warmup + 线性衰减学习率
调度、混合精度、梯度裁剪与评估流程均为手写实现，不使用 Trainer 等高层
封装。

注意：BERT 输入直接使用原始 Phrase 文本，不做 clean_text 清洗（小写化
与去标点会破坏预训练时的词分布）。国内下载预训练权重先设置镜像：
    $env:HF_ENDPOINT = "https://hf-mirror.com"

运行方式（二选一）：
    # 编排脚本批量跑（推荐，自动跳过已完成实验）
    python scripts/run_experiments.py --group base        # 或 --group context
    # 单独运行本脚本
    python src/models/deep_learning/train_bert.py --exp_name base

其他参数示例：
    python src/models/deep_learning/train_bert.py --exp_name ctx --use_context --max_len 96   # 句对融合
    python src/models/deep_learning/train_bert.py --max_samples 5000 --epochs 1               # 冒烟
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dl_train import clip_grad_norm
from common.experiment import ExperimentLogger
from common.split import ensure_split
from common.utils import ensure_dirs, load_data, set_seed
from common.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT with a hand-written loop.")
    parser.add_argument("--exp_name", type=str, default="base")
    parser.add_argument("--mode", type=str, default="stratified", choices=("stratified", "grouped"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=48, help="短语任务 48 已覆盖 99.98%% 样本；ctx 句对模式在编排脚本中显式传 96")
    parser.add_argument("--use_context", action="store_true",
                        help="启用句对输入 (完整句子, 短语) 的上下文融合")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0, help="调试用：>0 时只抽取训练子集")
    parser.add_argument("--fp16", type=int, default=1, help="1=启用混合精度")
    return parser.parse_args()


def warmup_linear_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    """手写 warmup + 线性衰减调度，等价于 transformers 的 get_linear_schedule_with_warmup。"""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - progress)


def set_lr(optimizer: torch.optim.Optimizer, base_lrs: list[float], scale: float) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


def add_raw_context(df: pd.DataFrame) -> pd.DataFrame:
    """按句重建原始完整句子作为上下文（BERT 用原始文本，不做清洗）。"""
    from common.preprocess import build_sentence_context
    return build_sentence_context(df)


def encode_texts(tokenizer, texts, max_len: int, batch_size: int = 1024,
                 contexts=None):
    """批量 tokenize 成定长 id / mask 矩阵。

    传入 ``contexts`` 时使用句对编码 (context, phrase)：``longest_first``
    逐 token 截断较长序列——上下文几乎总是更长，因此实际被截的几乎都是
    上下文；该策略同时保证任何超长短语都不会放不进 max_length。
    """
    all_ids, all_mask = [], []
    for start in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[start:start + batch_size]]
        if contexts is not None:
            cchunk = [str(t) for t in contexts[start:start + batch_size]]
            encoded = tokenizer(cchunk, chunk, truncation="longest_first",
                                max_length=max_len, padding="max_length",
                                return_tensors="np")
        else:
            encoded = tokenizer(chunk, truncation=True, max_length=max_len,
                                padding="max_length", return_tensors="np")
        all_ids.append(encoded["input_ids"].astype(np.int64))
        all_mask.append(encoded["attention_mask"].astype(np.int64))
    return np.concatenate(all_ids), np.concatenate(all_mask)


def batch_predict(model, ids, mask, device, batch_size: int, use_amp: bool,
                  labels=None) -> tuple[float | None, np.ndarray, np.ndarray]:
    """批量推理，返回 (平均交叉熵损失或 None, 预测类别, softmax 概率)。"""
    model.eval()
    preds, all_probs, total_loss, seen = [], [], 0.0, 0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            sl = slice(start, start + batch_size)
            # 批内动态长度裁剪（与训练一致）
            valid = max(int(mask[sl].sum(axis=1).max()), 4)
            xb = torch.from_numpy(np.ascontiguousarray(ids[sl][:, :valid])).to(device)
            mb = torch.from_numpy(np.ascontiguousarray(mask[sl][:, :valid])).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids=xb, attention_mask=mb).logits
                if labels is not None:
                    yb = torch.from_numpy(np.array(labels[sl])).to(device)
                    total_loss += F.cross_entropy(logits.float(), yb).item() * len(xb)
                    seen += len(xb)
            probs = torch.softmax(logits.float(), dim=1)
            preds.append(probs.argmax(dim=1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    mean_loss = total_loss / seen if seen else None
    return mean_loss, np.concatenate(preds), np.concatenate(all_probs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16) and device.type == "cuda"
    print(f"[环境] device={device}  fp16={use_amp}")

    # -------------------------------------------------- 数据读取与统一划分
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    train_df, test_df = load_data()  # BERT 使用原始文本，不做 clean_text
    if args.use_context:
        # 在切分前按句子聚合上下文，保证每条短语拿到完整句子
        train_df = add_raw_context(train_df)
        test_df = add_raw_context(test_df)
        print(f"[上下文] 句对输入 (context, phrase)，max_len={args.max_len}")
    train_part, val_part = ensure_split(train_df, mode=args.mode,
                                        val_ratio=args.val_ratio, seed=args.seed)
    if args.max_samples > 0:
        train_part = train_part.sample(n=min(args.max_samples, len(train_part)),
                                       random_state=args.seed).reset_index(drop=True)
    print(f"[数据] train={len(train_part)}  val={len(val_part)}  test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ctx_train = train_part["sentence_context"] if args.use_context else None
    ctx_val = val_part["sentence_context"] if args.use_context else None
    x_train, m_train = encode_texts(tokenizer, train_part["Phrase"], args.max_len, contexts=ctx_train)
    x_val, m_val = encode_texts(tokenizer, val_part["Phrase"], args.max_len, contexts=ctx_val)
    y_train = train_part["Sentiment"].to_numpy()
    y_val = val_part["Sentiment"].to_numpy()

    logger = ExperimentLogger(family="dl", model="bert", exp_name=args.exp_name,
                              params=vars(args), split_mode=args.mode, seed=args.seed)
    logger.set_data_info(len(train_part), len(val_part), len(test_df))

    # -------------------------------------------------- 模型与优化器
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5).to(device)
    no_decay = ("bias", "LayerNorm.weight")
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr, eps=1e-8)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    steps_per_epoch = math.ceil(len(x_train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(args.seed)

    # -------------------------------------------------- 手写训练循环
    history: list[dict] = []
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    bad_epochs = 0
    global_step = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, seen = 0.0, 0
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            # 批内动态长度：截去批内全为 0 的尾部。短语平均仅约 7 词，
            # 固定 max_len 会让大半计算花在 padding 上（与 BiLSTM 同款优化）。
            valid = max(int(m_train[index].sum(axis=1).max()), 4)
            xb = torch.from_numpy(np.ascontiguousarray(x_train[index][:, :valid])).to(device)
            mb = torch.from_numpy(np.ascontiguousarray(m_train[index][:, :valid])).to(device)
            yb = torch.from_numpy(np.array(y_train[index])).to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids=xb, attention_mask=mb).logits
                loss = F.cross_entropy(logits.float(), yb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            set_lr(optimizer, base_lrs, warmup_linear_scale(global_step, total_steps, warmup_steps))

            total_loss += loss.item() * len(xb)
            seen += len(xb)

        val_loss, val_pred, _ = batch_predict(model, x_val, m_val, device, args.batch_size * 2, use_amp,
                                              labels=y_val)
        metrics_epoch = evaluate_predictions(pd.Series(y_val), pd.Series(val_pred))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_acc": metrics_epoch["accuracy"],
            "val_macro_f1": metrics_epoch["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"[epoch {epoch:02d}] train_loss={row['train_loss']:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={row['val_acc']:.4f}  val_macro_f1={row['val_macro_f1']:.4f}  "
              f"lr={row['lr']:.2e}  耗时={row['time_s']}s")

        if row["val_acc"] > best["val_acc"]:
            best = {"val_acc": row["val_acc"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[早停] 连续 {args.patience} 轮验证精度未提升，在第 {epoch} 轮停止。")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    train_seconds = round(time.time() - t_start, 1)
    logger.save_history(history)

    # -------------------------------------------------- 评估与落盘
    _, val_pred, val_probs = batch_predict(model, x_val, m_val, device, args.batch_size * 2, use_amp)
    metrics = evaluate_predictions(pd.Series(y_val), pd.Series(val_pred))
    metrics.update({"best_epoch": best["epoch"], "train_seconds": train_seconds})
    logger.save_metrics(metrics)
    logger.save_predictions(val_part["PhraseId"], y_val, val_pred)

    x_test, m_test = encode_texts(tokenizer, test_df["Phrase"], args.max_len,
                                  contexts=test_df["sentence_context"] if args.use_context else None)
    _, test_pred, test_probs = batch_predict(model, x_test, m_test, device, args.batch_size * 2, use_amp)
    logger.save_probs(val_probs, test_probs)
    logger.save_submission(test_df["PhraseId"], test_pred)
    logger.print_summary(metrics)


if __name__ == "__main__":
    main()

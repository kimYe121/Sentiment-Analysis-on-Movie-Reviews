"""
手写训练循环：前向 / 反向 / 参数更新、梯度裁剪、余弦学习率衰减、
逐轮验证、最优权重保留。

TextCNN 与 BiLSTM 共用本模块；BERT 因使用混合精度与独立的学习率调度，
在其训练脚本中另有一份针对预训练模型的手写循环。这里同样不依赖任何
高层训练框架（Lightning / Trainer 等）。
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from common.dl_data import PAD_ID
from models.deep_learning.layers import manual_cross_entropy


def clip_grad_norm(params, max_norm: float) -> torch.Tensor:
    """手写全局梯度范数裁剪，等价于 torch.nn.utils.clip_grad_norm_。"""
    grads = [p.grad for p in params if p.grad is not None]
    total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    if max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(scale)
    return total_norm


def _forward_batch(model, batch, device):
    """把批张量搬到设备并前向，兼容 (ids, labels) 与 (ids, context, labels)。"""
    if len(batch) == 3:
        xb, xc, yb = batch
        return model(xb.to(device), xc.to(device)), yb.to(device)
    xb, yb = batch
    return model(xb.to(device)), yb.to(device)


def predict(model, ids: np.ndarray, device, batch_size: int = 512,
            labels: np.ndarray | None = None,
            label_smoothing: float = 0.0,
            context_ids: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray]:
    """对给定数据批量推理，返回 (平均交叉熵损失或 None, 预测类别)。"""
    model.eval()
    all_logits = []
    total_loss, seen = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            sl = slice(start, start + batch_size)
            batch_np = ids[sl]
            # 与 BatchIterator 相同的批内动态长度截断
            max_len = max(int((batch_np != PAD_ID).sum(axis=1).max()), 12)
            batch = [torch.from_numpy(np.ascontiguousarray(batch_np[:, :max_len])).to(device)]
            if context_ids is not None:
                ctx_np = context_ids[sl]
                c_len = max(int((ctx_np != PAD_ID).sum(axis=1).max()), 12)
                batch.append(torch.from_numpy(np.ascontiguousarray(ctx_np[:, :c_len])).to(device))
            logits = model(*batch)
            all_logits.append(logits.float().cpu())
            if labels is not None:
                # pandas 2.x 的 to_numpy() 可能返回只读数组，np.array 强制拷贝保证可写
                yb = torch.from_numpy(np.array(labels[sl])).to(device)
                total_loss += manual_cross_entropy(logits, yb, label_smoothing).item() * len(batch[0])
                seen += len(batch[0])
    logits = torch.cat(all_logits)
    mean_loss = total_loss / seen if seen else None
    return mean_loss, logits.argmax(dim=1).numpy()


def predict_probs(model, ids: np.ndarray, device, batch_size: int = 512,
                  context_ids: np.ndarray | None = None) -> np.ndarray:
    """批量推理并返回 softmax 概率矩阵 (N, num_classes)，供模型集成使用。"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            sl = slice(start, start + batch_size)
            batch_np = ids[sl]
            max_len = max(int((batch_np != PAD_ID).sum(axis=1).max()), 12)
            batch = [torch.from_numpy(np.ascontiguousarray(batch_np[:, :max_len])).to(device)]
            if context_ids is not None:
                ctx_np = context_ids[sl]
                c_len = max(int((ctx_np != PAD_ID).sum(axis=1).max()), 12)
                batch.append(torch.from_numpy(np.ascontiguousarray(ctx_np[:, :c_len])).to(device))
            probs = torch.softmax(model(*batch).float(), dim=1)
            all_probs.append(probs.cpu())
    return torch.cat(all_probs).numpy()


def run_training(model, optimizer, train_iter, val_ids: np.ndarray, val_labels: np.ndarray,
                 device, epochs: int, grad_clip: float = 0.0, patience: int = 3,
                 label_smoothing: float = 0.0, lr_schedule: str = "none",
                 context_val_ids: np.ndarray | None = None):
    """标准训练循环，按 val_acc 保留最优权重（原地加载回模型）。

    - ``label_smoothing``：标签平滑系数，训练与验证损失使用同一口径；
    - ``lr_schedule="cosine"``：手写余弦退火，第 e 轮学习率为
      lr * 0.5 * (1 + cos(pi * (e-1) / (epochs-1)))，由初始值平滑衰减到 0，
      后期小学习率减轻对训练集的过度拟合，让验证指标能在更多轮次上缓慢爬升。

    返回 (history DataFrame, best_epoch)。
    """
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    history: list[dict] = []
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        if lr_schedule == "cosine":
            scale = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(1, epochs - 1)))
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * scale

        model.train()
        t0 = time.time()
        total_loss, seen = 0.0, 0
        for batch in train_iter:
            logits, yb = _forward_batch(model, batch, device)
            loss = manual_cross_entropy(logits, yb, label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * len(yb)
            seen += len(yb)

        val_loss, val_pred = predict(model, val_ids, device, labels=val_labels,
                                     label_smoothing=label_smoothing,
                                     context_ids=context_val_ids)
        val_acc = accuracy_score(val_labels, val_pred)
        val_macro_f1 = f1_score(val_labels, val_pred, average="macro")
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "time_s": round(time.time() - t0, 1),
        }
        history.append(row)
        print(f"[epoch {epoch:02d}] lr={row['lr']:.2e}  train_loss={row['train_loss']:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"val_macro_f1={val_macro_f1:.4f}  耗时={row['time_s']}s")

        if val_acc > best["val_acc"]:
            best = {
                "val_acc": val_acc,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[早停] 连续 {patience} 轮验证精度未提升，在第 {epoch} 轮停止。")
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return pd.DataFrame(history), best["epoch"]

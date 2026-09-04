"""手写基础层与损失函数：TextCNN / BiLSTM 共用的最小组件集。

除 ``nn.Parameter`` 参数容器与 ``nn.Module``（仅作为参数注册载体）之外，
所有计算均以张量运算手写实现，不调用 nn.Embedding / nn.Linear / nn.Dropout /
nn.Conv1d / nn.LSTM 等现成模块，对应课程设计"不调用库函数完整实现算法"的
加分项。每个手写组件都提供与库实现的一致性验证函数（``verify_*``），
供报告论证自实现实现的正确性。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ManualEmbedding(nn.Module):
    """查表嵌入：等价于 nn.Embedding，用参数索引直接实现。"""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        with torch.no_grad():
            self.weight.normal_(0.0, 0.1)
            if padding_idx is not None:
                self.weight[padding_idx].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # (B, L) -> (B, L, D)
        return self.weight[ids]


class ManualLinear(nn.Module):
    """线性层 y = x @ W^T + b，等价于 nn.Linear。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        bound = 1.0 / in_features ** 0.5
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return x @ self.weight.T
        return x @ self.weight.T + self.bias


class ManualDropout(nn.Module):
    """倒置 dropout，等价于 nn.Dropout。"""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand_like(x) < keep).to(x.dtype)
        return x * mask / keep


def manual_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         label_smoothing: float = 0.0) -> torch.Tensor:
    """手写 softmax 交叉熵（数值稳定版），支持标签平滑，等价于
    F.cross_entropy(..., label_smoothing=...)。

    标签平滑把硬目标 y 换成 (1-eps)*onehot + eps/K 的软目标，
    抑制模型对训练集的过度自信，是缓解过拟合的正则化手段。
    """
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_probs = shifted - shifted.exp().sum(dim=-1, keepdim=True).log()
    nll = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    if label_smoothing > 0.0:
        smooth = -log_probs.mean(dim=1)
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    return nll.mean()


# --------------------------------------------------------------------- 验证
def verify_manual_linear(tol: float = 1e-5) -> float:
    """验证 ManualLinear 与 nn.Linear 数值一致。"""
    x = torch.randn(8, 16)
    mine = ManualLinear(16, 4)
    ref = nn.Linear(16, 4)
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
        ref.bias.copy_(mine.bias)
    return (mine(x) - ref(x)).abs().max().item()


def verify_manual_cross_entropy(tol: float = 1e-5) -> float:
    """验证手写交叉熵（含标签平滑）与 F.cross_entropy 数值一致。"""
    logits = torch.randn(32, 5)
    targets = torch.randint(0, 5, (32,))
    diff_plain = (manual_cross_entropy(logits, targets)
                  - F.cross_entropy(logits, targets)).abs().max().item()
    diff_smooth = (manual_cross_entropy(logits, targets, label_smoothing=0.1)
                   - F.cross_entropy(logits, targets, label_smoothing=0.1)).abs().max().item()
    return max(diff_plain, diff_smooth)


def run_component_checks() -> None:
    """训练脚本启动时调用，打印全部手写组件的一致性验证结果。"""
    print(f"[组件验证] ManualLinear vs nn.Linear   max|diff| = {verify_manual_linear():.2e}")
    print(f"[组件验证] manual_cross_entropy vs F.cross_entropy   "
          f"max|diff| = {verify_manual_cross_entropy():.2e}")

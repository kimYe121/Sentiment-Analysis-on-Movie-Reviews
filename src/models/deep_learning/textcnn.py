"""TextCNN 手写实现（复现: Kim, 2014, EMNLP）。

网络结构：嵌入查表 -> 多尺寸一维卷积(k=3,4,5) -> ReLU -> 时间维最大池化
-> 拼接 -> dropout -> 全连接分类。

卷积层不使用 ``nn.Conv1d``，而是用 unfold 滑窗 + einsum 矩阵乘法手写，
``verify_manual_conv1d`` 会把相同权重拷贝进 nn.Conv1d 对比输出，
验证自实现与库实现的数值一致性。
"""

from __future__ import annotations

import torch
from torch import nn

from models.deep_learning.layers import ManualDropout, ManualEmbedding, ManualLinear


class ManualConv1d(nn.Module):
    """一维卷积手写实现，等价于 nn.Conv1d(padding=0)。

    对输入 (B, C_in, L) 沿时间维取滑窗得到 (B, C_in, L', k)，再与权重
    (C_out, C_in, k) 做 einsum 求和，数学上等价于标准卷积。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        bound = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = x.unfold(2, self.kernel_size, 1)  # (B, C_in, L-k+1, k)
        return torch.einsum("bclk,ock->bol", windows, self.weight) + self.bias[:, None]


def verify_manual_conv1d(in_channels: int = 32, out_channels: int = 8,
                         kernel_size: int = 3, seq_len: int = 20) -> float:
    """把权重拷贝进 nn.Conv1d，返回两者的最大输出误差。"""
    x = torch.randn(4, in_channels, seq_len)
    mine = ManualConv1d(in_channels, out_channels, kernel_size)
    ref = nn.Conv1d(in_channels, out_channels, kernel_size)
    with torch.no_grad():
        ref.weight.copy_(mine.weight)
        ref.bias.copy_(mine.bias)
    return (mine(x) - ref(x)).abs().max().item()


class TextCNN(nn.Module):
    """TextCNN 分类器。模型组件全部来自 layers.py 的手写实现。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 128,
        num_classes: int = 5,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = ManualEmbedding(vocab_size, embed_dim, padding_idx)
        self.convs = nn.ModuleList(
            [ManualConv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = ManualDropout(dropout)
        self.fc = ManualLinear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(ids).transpose(1, 2)              # (B, D, L)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        hidden = torch.cat(pooled, dim=1)                      # (B, F * len(kernels))
        return self.fc(self.dropout(hidden))

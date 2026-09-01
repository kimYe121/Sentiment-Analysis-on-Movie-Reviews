"""BiLSTM 手写实现 + 注意力池化。

- LSTM 单元完全手写（输入/遗忘/输出门 + 候选记忆），沿时间步逐步展开，
  对应复现: Hochreiter & Schmidhuber, 1997, Long Short-Term Memory。
- 双向结构：正向与反向各用一个独立单元，输出按时间对齐后拼接。
- 注意力池化对应复现: Zhou et al., 2016, ACL（tanh 注意力对 BiLSTM
  隐藏状态做加权和），同时提供 last / mean 池化作为消融对照。

``verify_manual_bilstm`` 会把权重拷贝进 nn.LSTM 对比输出，验证自实现
与库实现的数值一致性。
"""

from __future__ import annotations

import torch
from torch import nn

from models.deep_learning.layers import ManualDropout, ManualEmbedding, ManualLinear


class ManualLSTMCell(nn.Module):
    """单步 LSTM 单元手写实现，等价于 nn.LSTMCell（门顺序 i, f, g, o）。"""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(
            torch.empty(4 * hidden_size, input_size).uniform_(-1.0 / input_size ** 0.5, 1.0 / input_size ** 0.5)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(4 * hidden_size, hidden_size).uniform_(-1.0 / hidden_size ** 0.5, 1.0 / hidden_size ** 0.5)
        )
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        # 遗忘门偏置置 1：标准工程技巧，缓解训练初期记忆快速丢失
        with torch.no_grad():
            self.bias_ih[hidden_size:2 * hidden_size].fill_(1.0)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        gates = x @ self.weight_ih.T + self.bias_ih + h_prev @ self.weight_hh.T + self.bias_hh
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ManualBiLSTM(nn.Module):
    """双向 LSTM：逐步展开循环，正向/反向各自维护 (h, c) 状态。"""

    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.forward_cell = ManualLSTMCell(input_size, hidden_size)
        self.backward_cell = ManualLSTMCell(input_size, hidden_size) if bidirectional else None

    def output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in) -> (B, T, out)
        batch, seq_len = x.shape[0], x.shape[1]

        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        forward_states = []
        for t in range(seq_len):
            h, c = self.forward_cell(x[:, t], (h, c))
            forward_states.append(h)

        if not self.bidirectional:
            return torch.stack(forward_states, dim=1)

        h = torch.zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        backward_states = [None] * seq_len
        for t in range(seq_len - 1, -1, -1):
            h, c = self.backward_cell(x[:, t], (h, c))
            backward_states[t] = h

        return torch.cat([torch.stack(forward_states, dim=1),
                          torch.stack(backward_states, dim=1)], dim=2)


class ManualAttention(nn.Module):
    """加性注意力池化（tanh attention）：score = v^T tanh(W h)。

    对 padding 位置掩码后做 softmax，再加权求和得到句子表示；
    注意力权重 alpha 保留在 ``last_alpha`` 中，可用于可视化。
    """

    def __init__(self, hidden_size: int, attn_dim: int = 64) -> None:
        super().__init__()
        self.proj = ManualLinear(hidden_size, attn_dim)
        self.vector = nn.Parameter(torch.empty(attn_dim).normal_(0.0, 0.1))
        self.last_alpha: torch.Tensor | None = None

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # states: (B, T, H)，mask: (B, T) bool，True 表示有效位置
        energy = torch.tanh(self.proj(states)) @ self.vector      # (B, T)
        energy = energy.masked_fill(~mask, -1e9)
        alpha = torch.softmax(energy, dim=1)
        self.last_alpha = alpha.detach()
        return (alpha.unsqueeze(-1) * states).sum(dim=1)          # (B, H)


class BiLSTMClassifier(nn.Module):
    """BiLSTM 分类器：嵌入 -> (多层)BiLSTM -> 池化(last/mean/attention) -> 分类。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        pooling: str = "attention",
        num_classes: int = 5,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if pooling not in ("last", "mean", "attention"):
            raise ValueError(f"不支持的池化方式: {pooling}")

        self.padding_idx = padding_idx
        self.pooling = pooling
        self.bidirectional = bidirectional

        self.embedding = ManualEmbedding(vocab_size, embed_dim, padding_idx)
        layers: list[ManualBiLSTM] = []
        in_size = embed_dim
        for _ in range(num_layers):
            layer = ManualBiLSTM(in_size, hidden_size, bidirectional)
            layers.append(layer)
            in_size = layer.output_size()
        self.bilstm_layers = nn.ModuleList(layers)

        out_size = in_size
        if pooling == "attention":
            self.attention = ManualAttention(out_size)
        else:
            self.attention = None
        self.dropout = ManualDropout(dropout)
        self.fc = ManualLinear(out_size, num_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids != self.padding_idx                            # (B, L)
        hidden = self.embedding(ids)                              # (B, L, D)
        for layer in self.bilstm_layers:
            hidden = layer(hidden)

        if self.pooling == "attention":
            pooled = self.attention(hidden, mask)
        elif self.pooling == "mean":
            summed = hidden.masked_fill(~mask.unsqueeze(-1), 0.0).sum(dim=1)
            pooled = summed / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:  # last：取每个序列最后一个有效时间步的隐状态
            lengths = mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]

        return self.fc(self.dropout(pooled))


# --------------------------------------------------------------------- 验证
def _copy_cell_into(cell: ManualLSTMCell, prefix: str, lstm: nn.LSTM) -> None:
    with torch.no_grad():
        getattr(lstm, f"weight_ih_{prefix}").copy_(cell.weight_ih)
        getattr(lstm, f"weight_hh_{prefix}").copy_(cell.weight_hh)
        getattr(lstm, f"bias_ih_{prefix}").copy_(cell.bias_ih)
        getattr(lstm, f"bias_hh_{prefix}").copy_(cell.bias_hh)


def verify_manual_bilstm(vocab_size: int = 50, embed_dim: int = 16,
                         hidden_size: int = 12, seq_len: int = 9) -> float:
    """把权重拷贝进 nn.Embedding + nn.LSTM，返回最大输出误差。"""
    torch.manual_seed(0)
    ids = torch.randint(2, vocab_size, (6, seq_len))
    mine = BiLSTMClassifier(vocab_size, embed_dim, hidden_size, num_layers=1,
                            bidirectional=True, pooling="mean", num_classes=5,
                            dropout=0.0)

    ref = nn.Sequential()
    ref_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    ref_lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1,
                       batch_first=True, bidirectional=True)
    ref_fc = nn.Linear(hidden_size * 2, 5)
    with torch.no_grad():
        ref_embedding.weight.copy_(mine.embedding.weight)
        _copy_cell_into(mine.bilstm_layers[0].forward_cell, "l0", ref_lstm)
        _copy_cell_into(mine.bilstm_layers[0].backward_cell, "l0_reverse", ref_lstm)
        ref_fc.weight.copy_(mine.fc.weight)
        ref_fc.bias.copy_(mine.fc.bias)

    mine.eval()
    with torch.no_grad():
        mine_logits = mine(ids)
        ref_out, _ = ref_lstm(ref_embedding(ids))
        # mean 池化（padding 位置恒为 pad 嵌入非零，这里所有位置都视为有效，
        # 与 mask 无关的纯数值一致性检查）
        ref_logits = ref_fc(ref_out.mean(dim=1))
    return (mine_logits - ref_logits).abs().max().item()

import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_index=0):
        super().__init__()
        # 词嵌入层，把 [B, T] 的 token 索引转成 [B, T, E] 的实数向量。
        # padding_idx=pad_index：该索引位置的嵌入向量会固定为 0，反向传播时不更新
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, # 每个时间步输入向量维度就是 E。
            hidden_size=hidden_dim,   # 每层每方向的隐状态维度是 H。
            num_layers=n_layers,      # 隐状态层数
            bidirectional=bidirectional, # 若为 True，每层有前向和后向两个方向；输出/隐状态通道翻倍。
            dropout=dropout if n_layers > 1 else 0.0, # PyTorch 语义是层间dropout，仅当 n_layers > 1 时才生效（单层无层间过渡，自然不生效）。
            batch_first=True # 输入输出的 batch 维在最前，即 [B, T, *]（比默认 [T, B, *] 更直观）。
        )
        # LSTM 输出通道 C = H(单向) 或 2H(双向)。
        # 前向里做了 mean-pooling 和 max-pooling 并拼接 -> 特征维度为 2*C。
        # 因此全连接层输入应为 2*C。
        C = hidden_dim * (2 if bidirectional else 1)
        fc_in = C * 2
        self.fc = nn.Linear(fc_in, output_dim) # 全连接层，做最终分类
        self.dropout = nn.Dropout(dropout) # dropout 层
        self.bidirectional = bidirectional # 记录是否双向 LSTM

    def forward(self, ids, lengths):
        # ids: [B, T], lengths: [B]
        emb = self.dropout(self.embedding(ids))  # [B, T, E]

        # pack 需要 CPU 的 lengths（兼容性更好）
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)

        # 解包回 [B, T, C]，C = H 或 2H（双向）
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, T, C]
        B, T, C = out.size()

        # ---- 构造 mask（True=有效 token）----
        lengths_dev = lengths.to(out.device)                  # 关键：放到同一设备
        time_idx = torch.arange(T, device=out.device).unsqueeze(0)  # [1, T]
        mask = time_idx < lengths_dev.unsqueeze(1)            # [B, T] bool

        # ---- mean pooling（按有效长度）----
        out_masked = out * mask.unsqueeze(-1)                 # [B, T, C]
        sum_pool = out_masked.sum(dim=1)                      # [B, C]
        len_clamped = lengths_dev.clamp_min(1).unsqueeze(1)   # [B, 1]
        mean_pool = sum_pool / len_clamped                    # [B, C]

        # ---- max pooling（pad 位置置 -inf）----
        out_neg_inf = out.masked_fill(~mask.unsqueeze(-1), float('-inf'))  # [B, T, C]
        max_pool = out_neg_inf.max(dim=1).values                            # [B, C]
        # 极端：若某些样本 length=0（基本不会发生，但保险起见）
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        # ---- 拼接 & 分类 ----
        feat = torch.cat([mean_pool, max_pool], dim=1)        # [B, 2C]
        feat = self.dropout(feat)
        logits = self.fc(feat)                                 # [B, output_dim]
        return logits
    

    def forward_old(self, ids, lengths):
        # ids: [B, T], lengths: [B]
        emb = self.dropout(self.embedding(ids))  # [B, T, E]

        # pack 需要 CPU 上的 lengths（不同 PyTorch 版本可能有要求）
        # packed 是一个 PackedSequence 对象，包含压缩后的序列数据和批次信息
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, (h_n, c_n) = self.lstm(packed)
        # h_n: [num_layers * num_directions, B, H]

        if self.bidirectional:
            # 取最后一层的正向和反向隐状态拼接
            # 层索引：最后一层正向 = -2，最后一层反向 = -1
            # -2 是最后一层正向隐状态，-1 是最后一层反向隐状态 为什么是这样？
            # 因为在双向 LSTM 中，正向和反向的隐状态是分开存储的，最后一层的正向隐状态在 -2 位置，反向隐状态在 -1 位置
            # 只关心最后一层的隐状态
            h_fwd = h_n[-2, :, :]
            h_bwd = h_n[-1, :, :]
            # 通过最后一层的正向和反向隐状态拼接作为特征
            feat = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 2H]
        else:
            feat = h_n[-1, :, :]  # [B, H]

        feat = self.dropout(feat)
        # 通过 dropout 防止过拟合
        logits = self.fc(feat)  # [B, 2]
        # 返回 logits fc 是最终分类结果 linear 层的输出
        return logits
    
# 假设：

# num_layers = L Layer 数量 指的是 LSTM 的层数
# bidirectional = True
# hidden_dim = H
# batch_size = B


# 那么：

# h_n.shape = [num_layers * num_directions, batch_size, hidden_dim]


# 其中：

# num_directions = 2（因为双向）

# 所以 h_n 的第 0 维长度 = L * 2

# 这第 0 维的每个 index 对应「第几层的哪一方向」的最后 hidden
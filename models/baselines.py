"""Baseline text classification architectures.
Includes TextCNN, BiLSTM-Attention, and TransformerClassifier.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, num_classes: int = 4, num_filters: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in [3, 4, 5]
        ])
        self.fc = nn.Linear(num_filters * 3, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # [B, L, D] -> [B, D, L]
        emb = self.embedding(input_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)
        return self.fc(out)


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)  # [B, L, 2H]
        attn_weights = F.softmax(self.attn_proj(lstm_out), dim=1)  # [B, L, 1]
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [B, 2H]
        context = self.dropout(context)
        return self.fc(context)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, num_heads: int = 4, num_layers: int = 2, num_classes: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        emb = self.embedding(input_ids) * math.sqrt(emb.size(-1) if "emb" in locals() else 256)
        x = self.pos_encoder(emb)
        mask = None
        if attention_mask is not None:
            mask = (attention_mask == 0)
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        # Average pooling over non-padding tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            sum_emb = torch.sum(encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_emb / sum_mask
        else:
            pooled = encoded.mean(dim=1)
        return self.fc(pooled)

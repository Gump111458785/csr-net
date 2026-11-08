"""Proposed Architecture: Contrastive Semantic Rectification Network (CSR-Net).
Incorporates Contrastive Semantic Rectification (CSR) module and calibrated output heads.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .baselines import TextCNN, BiLSTMAttention, TransformerClassifier, PositionalEncoding


class ContrastiveSemanticRectificationModule(nn.Module):
    """Contrastive Semantic Rectification (CSR) Module.
    
    Decouples token representations into invariant semantic core and lexical variance,
    applying dynamic contrastive gating to filter out spurious lexical cues.
    """
    def __init__(self, embed_dim: int = 256, reduction: int = 4):
        super().__init__()
        mid_dim = max(32, embed_dim // reduction)
        self.semantic_proj = nn.Linear(embed_dim, embed_dim)
        self.variance_proj = nn.Linear(embed_dim, embed_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, embed_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        b, l, d = x.shape
        semantic_feat = self.semantic_proj(x)
        variance_feat = self.variance_proj(x)

        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).expand_as(x)
            pooled = (x * mask_exp).sum(dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        else:
            pooled = x.mean(dim=1)

        # Dynamic rectification gate
        gate_weights = self.gate(pooled).unsqueeze(1)  # [B, 1, D]
        rectified = semantic_feat * gate_weights + variance_feat * (1.0 - gate_weights)
        return self.norm(x + rectified)


class CSRNet(nn.Module):
    """Contrastive Semantic Rectification Network (CSR-Net).
    
    A robust transformer-based text classifier with contrastive feature rectification
    and uncertainty calibration.
    """
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 4,
        use_csr: bool = True,
        use_temp_scale: bool = True,
    ):
        super().__init__()
        self.use_csr = use_csr
        self.use_temp_scale = use_temp_scale
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.2, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_csr:
            self.csr_module = ContrastiveSemanticRectificationModule(embed_dim=embed_dim)
        else:
            self.csr_module = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        if use_temp_scale:
            self.temperature = nn.Parameter(torch.ones(1) * 1.15)
        else:
            self.register_buffer("temperature", torch.ones(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, apply_temp: bool = True) -> torch.Tensor:
        emb = self.embedding(input_ids) * math.sqrt(256)
        x = self.pos_encoder(emb)

        mask = None
        if attention_mask is not None:
            mask = (attention_mask == 0)

        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Apply CSR Module
        if self.use_csr:
            encoded = self.csr_module(encoded, attention_mask=attention_mask)

        # Masked mean pooling
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).expand_as(encoded)
            pooled = (encoded * mask_exp).sum(dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        else:
            pooled = encoded.mean(dim=1)

        logits = self.classifier(pooled)

        if apply_temp and self.use_temp_scale:
            temp = torch.clamp(self.temperature, min=0.1, max=5.0)
            return logits / temp
        return logits


def build_model(
    model_name: str,
    vocab_size: int,
    num_classes: int,
    use_csr: bool = True,
    use_temp_scale: bool = True,
) -> nn.Module:
    model_name = model_name.lower().replace("-", "_")
    if model_name in ["csr_net", "csrnet", "ours"]:
        return CSRNet(
            vocab_size=vocab_size,
            num_classes=num_classes,
            use_csr=use_csr,
            use_temp_scale=use_temp_scale,
        )
    elif model_name == "textcnn":
        return TextCNN(vocab_size=vocab_size, num_classes=num_classes)
    elif model_name in ["bilstm_attn", "bilstm"]:
        return BiLSTMAttention(vocab_size=vocab_size, num_classes=num_classes)
    elif model_name == "transformer":
        return TransformerClassifier(vocab_size=vocab_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name '{model_name}'.")

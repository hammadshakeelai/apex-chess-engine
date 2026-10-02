"""
Spatial 64-Square Attention Transformer Policy-Value Network.

Architecture:
- 64 Chess squares as spatial tokens + global game state embedding.
- Multi-Head Self-Attention (MHSA) capturing long-range piece batteries (diagonals, files).
- Dual Output Heads:
  1. Policy Head: Move prior probabilities over 4,096 action transitions.
  2. Value Head: Position evaluation & Win/Draw/Loss probabilities.
"""

from typing import Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax implementation in NumPy."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class ChessTransformerNumpy:
    """Lightweight pure NumPy spatial transformer for inference."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        rng = np.random.RandomState(42)

        # Projection from 14 piece channels to embed_dim
        self.patch_proj = (rng.randn(14, embed_dim) / np.sqrt(14)).astype(np.float32)
        self.pos_embed = (rng.randn(64, embed_dim) * 0.02).astype(np.float32)

        # Multi-Head Attention weights
        self.w_q = (rng.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)).astype(np.float32)
        self.w_k = (rng.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)).astype(np.float32)
        self.w_v = (rng.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)).astype(np.float32)
        self.w_o = (rng.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)).astype(np.float32)

        # Feedforward MLP
        self.ffn_w1 = (rng.randn(embed_dim, embed_dim * 2) / np.sqrt(embed_dim)).astype(np.float32)
        self.ffn_w2 = (rng.randn(embed_dim * 2, embed_dim) / np.sqrt(embed_dim * 2)).astype(np.float32)

        # Policy Head: [64, embed_dim] -> from_sq and to_sq bilinear scoring
        self.policy_from = (rng.randn(embed_dim, 32) / np.sqrt(embed_dim)).astype(np.float32)
        self.policy_to = (rng.randn(embed_dim, 32) / np.sqrt(embed_dim)).astype(np.float32)

        # Value Head: Pooled features -> Scalar centipawns
        self.val_fc = (rng.randn(embed_dim, 1) / np.sqrt(embed_dim)).astype(np.float32)

    def forward(self, board_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Input: [8, 8, 14] spatial tensor
        Returns:
            policy_logits: [4096] unnormalized move logits (from_sq * 64 + to_sq)
            value: scalar evaluation in centipawns
        """
        # Reshape to [64, 14] tokens
        tokens = board_tensor.reshape(64, 14)

        # Project and add positional embeddings
        x = np.dot(tokens, self.patch_proj) + self.pos_embed

        # Multi-Head Self-Attention
        q = np.dot(x, self.w_q)
        k = np.dot(x, self.w_k)
        v = np.dot(x, self.w_v)

        d_k = self.embed_dim // self.num_heads
        scores = np.dot(q, k.T) / np.sqrt(d_k)
        attn = softmax_np(scores, axis=-1)
        mha_out = np.dot(np.dot(attn, v), self.w_o)

        # Residual connection
        x = x + mha_out

        # Feedforward
        ffn = np.maximum(0, np.dot(x, self.ffn_w1))
        x = x + np.dot(ffn, self.ffn_w2)

        # Policy Logits: Outer dot product of from-square features and to-square features
        from_feats = np.dot(x, self.policy_from)  # [64, 32]
        to_feats = np.dot(x, self.policy_to)      # [64, 32]
        # Logits matrix: [64, 64] -> flattened to [4096]
        policy_matrix = np.dot(from_feats, to_feats.T)  # [64, 64]
        policy_logits = policy_matrix.flatten()

        # Value evaluation: Global average pool -> Linear
        global_pool = np.mean(x, axis=0)  # [embed_dim]
        value = float(np.dot(global_pool, self.val_fc)[0]) * 350.0

        return policy_logits, value


if HAS_TORCH:
    class ChessTransformer(nn.Module):
        """PyTorch Spatial Transformer Policy-Value Network."""

        def __init__(self, embed_dim: int = 128, num_heads: int = 8, num_layers: int = 4):
            super().__init__()
            self.embed_dim = embed_dim
            self.patch_proj = nn.Linear(14, embed_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Bilinear move policy heads
            self.policy_from = nn.Linear(embed_dim, 64)
            self.policy_to = nn.Linear(embed_dim, 64)

            # Value head
            self.val_head = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            x: [Batch, 8, 8, 14]
            Returns:
                policy_logits: [Batch, 4096]
                value: [Batch, 1]
            """
            B = x.shape[0]
            tokens = x.view(B, 64, 14)
            h = self.patch_proj(tokens) + self.pos_embed
            h = self.transformer(h)  # [B, 64, embed_dim]

            # Compute from-to logits
            from_proj = self.policy_from(h)  # [B, 64, 64]
            to_proj = self.policy_to(h)      # [B, 64, 64]
            logits_matrix = torch.bmm(from_proj, to_proj.transpose(1, 2))  # [B, 64, 64]
            policy_logits = logits_matrix.view(B, 4096)

            # Value prediction
            pooled = torch.mean(h, dim=1)
            val = self.val_head(pooled)

            return policy_logits, val

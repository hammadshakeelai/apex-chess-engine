"""
NNUE (Efficiently Updatable Neural Network) Architecture.

Implements:
- HalfKP sparse feature accumulator with incremental updates.
- SCReLU (Squared Clipped ReLU) and ClippedReLU activations.
- Quantized-ready forward pass.
- Pure NumPy inference mode for zero-dependency portability and PyTorch Module for training.
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data.tokenizer import HALFKP_TOTAL_FEATURES


class NNUEWeights:
    """In-memory quantized or float weights container for NNUE."""

    def __init__(self, feature_dim: int = HALFKP_TOTAL_FEATURES, hidden_dim: int = 256):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Accumulator weights: [feature_dim, hidden_dim]
        # In production int16 scaled; here initialized with normalized gaussian
        rng = np.random.RandomState(42)
        scale = 1.0 / np.sqrt(30)  # ~30 active features per board
        self.feature_weights = (rng.randn(feature_dim, hidden_dim) * scale).astype(np.float32)
        self.feature_bias = (rng.randn(hidden_dim) * 0.05).astype(np.float32)

        # Dense Layer 1: [2 * hidden_dim, 32]
        self.w1 = (rng.randn(hidden_dim * 2, 32) / np.sqrt(hidden_dim * 2)).astype(np.float32)
        self.b1 = np.zeros(32, dtype=np.float32)

        # Dense Layer 2: [32, 32]
        self.w2 = (rng.randn(32, 32) / np.sqrt(32)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)

        # Output Layer: [32, 1]
        self.w_out = (rng.randn(32, 1) / np.sqrt(32)).astype(np.float32)
        self.b_out = np.zeros(1, dtype=np.float32)

    @classmethod
    def from_file(cls, path: str) -> "NNUEWeights":
        """Loads weights from .npz or .pt file."""
        weights = cls()
        weights.load(path)
        return weights

    def load(self, path: str):
        """Loads weights from .npz or .pt checkpoint file."""
        if path.endswith(".npz"):
            data = np.load(path)
            # High-fidelity FP32 format
            if "feature_weights" in data:
                self.feature_weights = data["feature_weights"].astype(np.float32)
                self.feature_bias = data["feature_bias"].astype(np.float32)
                self.hidden_dim = self.feature_bias.shape[0]
                self.w1 = data["fc1_w"].astype(np.float32)
                self.b1 = data["fc1_b"].astype(np.float32)
                self.w2 = data["fc2_w"].astype(np.float32)
                self.b2 = data["fc2_b"].astype(np.float32)
                self.w_out = data["out_w"].astype(np.float32)
                self.b_out = data["out_b"].astype(np.float32)
            elif "w_feat" in data:
                # Quantized export format
                self.feature_weights = data["w_feat"].astype(np.float32) / 255.0
                self.feature_bias = data["b_feat"].astype(np.float32) / 255.0
                self.hidden_dim = self.feature_bias.shape[0]

                w_fc1 = data["w_fc1"].astype(np.float32) / 64.0
                self.w1 = w_fc1.T if w_fc1.shape[0] == 32 else w_fc1
                b_fc1 = data["b_fc1"].astype(np.float32)
                if np.max(np.abs(b_fc1)) > 10.0:
                    b_fc1 = b_fc1 / (64.0 * 255.0)
                self.b1 = b_fc1

                w_fc2 = data["w_fc2"].astype(np.float32) / 64.0
                self.w2 = w_fc2.T if w_fc2.shape == (32, 32) else w_fc2
                b_fc2 = data["b_fc2"].astype(np.float32)
                if np.max(np.abs(b_fc2)) > 10.0:
                    b_fc2 = b_fc2 / 64.0
                self.b2 = b_fc2

                w_out = data["w_out"].astype(np.float32)
                self.w_out = (w_out.T if w_out.shape == (1, 32) else w_out) / 400.0
                b_out = data["b_out"].astype(np.float32)
                self.b_out = b_out / 400.0 if np.max(np.abs(b_out)) > 1.0 else b_out

        elif path.endswith(".pt") and HAS_TORCH:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            self.feature_weights = state["feature_embed.weight"].numpy().astype(np.float32)
            self.feature_bias = state["feature_bias"].numpy().astype(np.float32)
            self.w1 = state["fc1.weight"].numpy().astype(np.float32).T
            self.b1 = state["fc1.bias"].numpy().astype(np.float32)
            self.w2 = state["fc2.weight"].numpy().astype(np.float32).T
            self.b2 = state["fc2.bias"].numpy().astype(np.float32)
            self.w_out = state["out.weight"].numpy().astype(np.float32).T
            self.b_out = state["out.bias"].numpy().astype(np.float32)


def screlu(x: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """Squared Clipped ReLU: (clamp(x, 0, max_val))^2."""
    clamped = np.clip(x, 0.0, max_val)
    return clamped * clamped


class NNUEInference:
    """Fast NumPy-based NNUE inference engine with accumulator caching."""

    def __init__(self, weights: Optional[NNUEWeights] = None):
        self.weights = weights or NNUEWeights()

    def compute_accumulator(self, active_indices: List[int]) -> np.ndarray:
        """
        Compute accumulator from scratch given active HalfKP indices.
        A = bias + sum(weights[idx])
        """
        if len(active_indices) == 0:
            return self.weights.feature_bias.copy()

        # Sum active feature weight rows
        accum = self.weights.feature_bias + np.sum(self.weights.feature_weights[active_indices], axis=0)
        return accum

    def forward(
        self,
        white_indices: List[int],
        black_indices: List[int],
        turn_is_white: bool,
    ) -> float:
        """
        Evaluate position given sparse HalfKP feature indices.
        Returns evaluation in centipawns.
        """
        acc_w = self.compute_accumulator(white_indices)
        acc_b = self.compute_accumulator(black_indices)

        # Side to move ordering: [Friendly Accumulator, Enemy Accumulator]
        if turn_is_white:
            hidden = np.concatenate([screlu(acc_w), screlu(acc_b)])
        else:
            hidden = np.concatenate([screlu(acc_b), screlu(acc_w)])

        # Dense Layer 1
        h1 = screlu(np.dot(hidden, self.weights.w1) + self.weights.b1)

        # Dense Layer 2
        h2 = screlu(np.dot(h1, self.weights.w2) + self.weights.b2)

        # Output Centipawns (scaled by 400.0)
        res = np.dot(h2, self.weights.w_out) + self.weights.b_out
        out_eval = float(res[0])
        return out_eval * 400.0


if HAS_TORCH:
    class PyTorchNNUE(nn.Module):
        """PyTorch NNUE Module for GPU/CPU training with Backpropagation."""

        def __init__(self, feature_dim: int = HALFKP_TOTAL_FEATURES, hidden_dim: int = 256):
            super().__init__()
            self.hidden_dim = hidden_dim
            # Sparse embedding layer acts as the feature transformer table
            self.feature_embed = nn.EmbeddingBag(
                num_embeddings=feature_dim,
                embedding_dim=hidden_dim,
                mode="sum",
            )
            self.feature_bias = nn.Parameter(torch.zeros(hidden_dim))

            self.fc1 = nn.Linear(hidden_dim * 2, 32)
            self.fc2 = nn.Linear(32, 32)
            self.out = nn.Linear(32, 1)

        def screlu(self, x: torch.Tensor) -> torch.Tensor:
            clamped = torch.clamp(x, min=0.0, max=1.0)
            return clamped * clamped

        def forward(
            self,
            w_indices: torch.Tensor,
            w_offsets: torch.Tensor,
            b_indices: torch.Tensor,
            b_offsets: torch.Tensor,
            turn_is_white: torch.Tensor,
        ) -> torch.Tensor:
            acc_w = self.feature_embed(w_indices, w_offsets) + self.feature_bias
            acc_b = self.feature_embed(b_indices, b_offsets) + self.feature_bias

            act_w = self.screlu(acc_w)
            act_b = self.screlu(acc_b)

            # Stack according to turn
            # turn_is_white: [B, 1] boolean
            h_white_stm = torch.cat([act_w, act_b], dim=-1)
            h_black_stm = torch.cat([act_b, act_w], dim=-1)
            hidden = torch.where(turn_is_white, h_white_stm, h_black_stm)

            x = self.screlu(self.fc1(hidden))
            x = self.screlu(self.fc2(x))
            val = self.out(x)
            return val

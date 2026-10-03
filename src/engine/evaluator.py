"""
Unified Neural Evaluator for ApexChess.

Coordinates NNUE evaluation, Chess Transformer policy priors, and positional analysis.
"""

import os
from typing import Tuple, List, Optional, Dict
import chess
import numpy as np

from src.data.tokenizer import extract_halfkp_indices, board_to_spatial_tensor, move_to_action_index
from src.models.nnue import NNUEInference, NNUEWeights
from src.models.transformer_policy import ChessTransformerNumpy


class ApexEvaluator:
    """Unified evaluator integrating NNUE fast inference and Transformer policy priors."""

    def __init__(
        self,
        use_nnue: bool = True,
        use_policy_prior: bool = True,
        weights_path: Optional[str] = None,
    ):
        self.use_nnue = use_nnue
        self.use_policy_prior = use_policy_prior

        # Initialize neural models
        if weights_path and os.path.exists(weights_path):
            self.nnue = NNUEInference(weights=NNUEWeights.from_file(weights_path))
        elif os.path.exists("weights/apex_v1_quant.npz"):
            self.nnue = NNUEInference(weights=NNUEWeights.from_file("weights/apex_v1_quant.npz"))
        else:
            self.nnue = NNUEInference()

        self.transformer = ChessTransformerNumpy()

    def evaluate_nnue(self, board: chess.Board) -> int:
        """
        Evaluate position using NNUE from side-to-move's perspective.
        Returns integer centipawns.
        """
        w_idx, b_idx = extract_halfkp_indices(board)
        turn_is_white = (board.turn == chess.WHITE)
        score_cp = self.nnue.forward(w_idx, b_idx, turn_is_white)
        return int(round(score_cp))

    def evaluate_static(self, board: chess.Board) -> int:
        """
        Static evaluation of board position.
        Handles terminal conditions (checkmate, stalemate), then falls back to NNUE.
        """
        if board.is_checkmate():
            return -29000  # Losing player receives -29,000 centipawns

        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0  # Draw

        if self.use_nnue:
            return self.evaluate_nnue(board)

        # Basic material fallback
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }
        score = 0
        for sq, piece in board.piece_map().items():
            val = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += val
            else:
                score -= val

        return score if board.turn == chess.WHITE else -score

    def get_policy_prior(self, board: chess.Board) -> Dict[chess.Move, float]:
        """
        Compute move prior probabilities for all legal moves in the position.
        Uses the Spatial Transformer.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return {}

        if not self.use_policy_prior:
            # Uniform prior fallback
            uniform_prob = 1.0 / len(legal_moves)
            return {m: uniform_prob for m in legal_moves}

        # Run forward pass through Spatial Transformer
        tensor = board_to_spatial_tensor(board)
        logits, _ = self.transformer.forward(tensor)

        # Extract logits for legal moves
        move_logits = []
        for m in legal_moves:
            idx = move_to_action_index(m)
            move_logits.append(logits[idx] if idx < len(logits) else 0.0)

        # Softmax over legal moves
        arr = np.array(move_logits, dtype=np.float32)
        exp_arr = np.exp(arr - np.max(arr))
        probs = exp_arr / np.sum(exp_arr)

        return {m: float(p) for m, p in zip(legal_moves, probs)}


# Backward compatibility alias
UnifiedEvaluator = ApexEvaluator

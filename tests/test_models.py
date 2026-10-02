"""Unit tests for NNUE and Transformer Policy-Value models."""

import unittest
import chess
import numpy as np
from src.data.tokenizer import extract_halfkp_indices, board_to_spatial_tensor
from src.models.nnue import NNUEInference, NNUEWeights
from src.models.transformer_policy import ChessTransformerNumpy


class TestNeuralModels(unittest.TestCase):

    def setUp(self):
        self.board = chess.Board()

    def test_nnue_forward_pass(self):
        nnue = NNUEInference()
        w_idx, b_idx = extract_halfkp_indices(self.board)
        score_cp = nnue.forward(w_idx, b_idx, turn_is_white=True)

        self.assertIsInstance(score_cp, float)
        # Random initial weights should produce reasonable bound within [-2000, 2000]
        self.assertTrue(-2000 < score_cp < 2000)

    def test_transformer_forward_pass(self):
        model = ChessTransformerNumpy(embed_dim=32, num_heads=2)
        tensor = board_to_spatial_tensor(self.board)
        policy_logits, val = model.forward(tensor)

        # Policy logits must match 4096 square-transition space
        self.assertEqual(policy_logits.shape, (4096,))
        self.assertIsInstance(val, float)
        self.assertFalse(np.isnan(policy_logits).any())
        self.assertFalse(np.isnan(val))


if __name__ == "__main__":
    unittest.main()

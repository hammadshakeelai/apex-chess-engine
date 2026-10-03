"""
Unit tests for the Prototype Chess Neural Pipeline.
"""

import os
import unittest
import chess
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.nnue import PyTorchNNUE, NNUEWeights, NNUEInference
from src.engine.evaluator import ApexEvaluator
from src.data.tokenizer import extract_halfkp_indices


class TestPrototypePipeline(unittest.TestCase):

    def setUp(self):
        self.board = chess.Board()

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for neural module tests")
    def test_prototype_model_custom_hidden(self):
        model = PyTorchNNUE(hidden_dim=128)
        self.assertEqual(model.hidden_dim, 128)
        self.assertEqual(model.feature_embed.embedding_dim, 128)
        self.assertEqual(model.fc1.in_features, 256)

        w_idx, b_idx = extract_halfkp_indices(self.board)
        w_t = torch.tensor(w_idx, dtype=torch.long)
        b_t = torch.tensor(b_idx, dtype=torch.long)
        w_off = torch.tensor([0], dtype=torch.long)
        b_off = torch.tensor([0], dtype=torch.long)
        turn = torch.tensor([[True]], dtype=torch.bool)

        out = model(w_t, w_off, b_t, b_off, turn)
        self.assertEqual(out.shape, (1, 1))

    def test_prototype_evaluator_sanity_checks(self):
        evaluator = ApexEvaluator(use_nnue=True)
        # 1. Starting board
        score_start = evaluator.evaluate_static(self.board)
        self.assertIsInstance(score_start, int)

        # 2. Checkmate
        mate_board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertEqual(evaluator.evaluate_static(mate_board), -29000)

        # 3. Stalemate
        stalemate_board = chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
        self.assertEqual(evaluator.evaluate_static(stalemate_board), 0)


if __name__ == "__main__":
    unittest.main()

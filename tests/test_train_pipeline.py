"""
Unit tests for ApexChess Neural Training and Quantization Pipeline.
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

from src.models.nnue import NNUEWeights, NNUEInference, PyTorchNNUE
from src.engine.evaluator import ApexEvaluator
from src.data.tokenizer import extract_halfkp_indices


class TestTrainPipeline(unittest.TestCase):

    def setUp(self):
        self.board = chess.Board()

    def test_nnue_weights_instantiation(self):
        weights = NNUEWeights()
        self.assertEqual(weights.feature_weights.shape, (40960, 256))
        self.assertEqual(weights.w1.shape, (512, 32))
        self.assertEqual(weights.w2.shape, (32, 32))
        self.assertEqual(weights.w_out.shape, (32, 1))

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for neural module tests")
    def test_pytorch_nnue_forward(self):
        model = PyTorchNNUE()
        w_idx, b_idx = extract_halfkp_indices(self.board)

        w_tensor = torch.tensor(w_idx, dtype=torch.long)
        b_tensor = torch.tensor(b_idx, dtype=torch.long)
        w_offsets = torch.tensor([0], dtype=torch.long)
        b_offsets = torch.tensor([0], dtype=torch.long)
        turn = torch.tensor([[True]], dtype=torch.bool)

        out = model(w_tensor, w_offsets, b_tensor, b_offsets, turn)
        self.assertEqual(out.shape, (1, 1))
        self.assertFalse(torch.isnan(out).any())

    def test_evaluator_with_nnue(self):
        evaluator = ApexEvaluator(use_nnue=True)
        score = evaluator.evaluate_static(self.board)
        self.assertIsInstance(score, int)

    def test_evaluator_terminal_states(self):
        evaluator = ApexEvaluator()
        # Fool's mate
        mate_board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(mate_board.is_checkmate())
        score = evaluator.evaluate_static(mate_board)
        self.assertEqual(score, -29000)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for board encoding, tokenization, and Static Exchange Evaluation."""

import unittest
import chess
from src.data.tokenizer import (
    extract_halfkp_indices,
    board_to_spatial_tensor,
    move_to_action_index,
    action_index_to_move,
    HALFKP_TOTAL_FEATURES,
)
from src.engine.board import mvv_lva_score, calculate_game_phase


class TestBoardAndTokenizers(unittest.TestCase):

    def setUp(self):
        self.board = chess.Board()

    def test_halfkp_extraction(self):
        w_indices, b_indices = extract_halfkp_indices(self.board)
        # Starting position has 16 white pieces - 1 king = 15 pieces
        # 16 black pieces - 1 king = 15 pieces. Total = 30 active features.
        self.assertEqual(len(w_indices), 30)
        self.assertEqual(len(b_indices), 30)

        # Indices must fall within feature bounds
        for idx in w_indices:
            self.assertTrue(0 <= idx < HALFKP_TOTAL_FEATURES)
        for idx in b_indices:
            self.assertTrue(0 <= idx < HALFKP_TOTAL_FEATURES)

    def test_spatial_tensor_shape(self):
        tensor = board_to_spatial_tensor(self.board)
        self.assertEqual(tensor.shape, (8, 8, 14))
        # White pawns on rank 1 (index 1)
        self.assertEqual(tensor[1, 0, 0], 1.0)
        # Side to move plane (channel 12) must be 1.0 for White
        self.assertEqual(tensor[0, 0, 12], 1.0)

    def test_move_action_conversion(self):
        move = chess.Move.from_uci("e2e4")
        action_idx = move_to_action_index(move)
        decoded_move = action_index_to_move(action_idx, self.board)
        self.assertEqual(move, decoded_move)

    def test_mvv_lva(self):
        # Position where Pawn captures Queen
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/5P1q/8/PPPPP1PP/RNBQKBNR w KQkq - 1 3")
        # g2g3 attacked, suppose pawn captures queen
        test_board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4q3/4P3/PPPP2PP/RNBQKBNR w KQkq - 0 4")
        # Move d2d3 attacks queen, or pawn captures queen if on d4
        cap_board = chess.Board("rnb1kbnr/pppp1ppp/8/8/3qP3/8/PPPP2PP/RNBQKBNR w KQkq - 0 4")
        move = chess.Move.from_uci("e4d4")  # pawn takes queen
        score = mvv_lva_score(cap_board, move)
        self.assertGreater(score, 8000)

    def test_game_phase(self):
        # Starting position is 1.0 (Full middlegame/opening)
        self.assertAlmostEqual(calculate_game_phase(self.board), 1.0)
        # Bare kings endgame is 0.0
        bare_kings = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
        self.assertAlmostEqual(calculate_game_phase(bare_kings), 0.0)


if __name__ == "__main__":
    unittest.main()

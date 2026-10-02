"""Unit tests for Search Engine, Transposition Table, and UCI protocol."""

import unittest
import chess
from src.engine.search import SearchEngine, SearchLimits, TranspositionTable, FLAG_EXACT
from src.engine.evaluator import ApexEvaluator
from src.engine.uci import UCIEngine


class TestSearchAndUCI(unittest.TestCase):

    def test_transposition_table(self):
        tt = TranspositionTable(size_mb=1)
        key = 123456789
        move = chess.Move.from_uci("e2e4")
        tt.store(key, depth=4, score=150, flag=FLAG_EXACT, best_move=move)

        entry = tt.get(key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.depth, 4)
        self.assertEqual(entry.score, 150)
        self.assertEqual(entry.best_move, move)

    def test_mate_in_one_tactical_search(self):
        # Scholar's mate delivery position: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
        # White to move: Qxf7# is immediate mate-in-1
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        evaluator = ApexEvaluator()
        searcher = SearchEngine(evaluator=evaluator)

        limits = SearchLimits(depth=3)
        best_move, score = searcher.search(board, limits)

        expected_mate = chess.Move.from_uci("h5f7")
        self.assertEqual(best_move, expected_mate)
        self.assertGreater(score, 25000)  # Mate score is near 30,000

    def test_uci_protocol_initialization(self):
        engine = UCIEngine()
        # Test basic handlers without crashing
        engine.handle_uci()
        engine.handle_isready()
        engine.handle_position(["position", "startpos", "moves", "e2e4", "e7e5"])
        self.assertEqual(len(engine.board.move_stack), 2)
        engine.handle_eval()


if __name__ == "__main__":
    unittest.main()

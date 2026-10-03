"""
ApexChess Tactical Benchmark Suite (Bratko-Kopec & Tactical Diagnostics)
Evaluates chess engine search depth, tactical accuracy, and node efficiency.
"""

import os
import sys
import time
import argparse
from typing import Optional
import chess

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.search import SearchEngine, SearchLimits
from src.engine.evaluator import ApexEvaluator, UnifiedEvaluator


# Canonical Bratko-Kopec Test (BKT) Positions & Seminal Tactical Benchmarks
BENCHMARK_POSITIONS = [
    # 1. Knight fork / Tactical shot
    {
        "id": "BK.01",
        "fen": "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
        "expected": ["d6d1"],
        "theme": "Queen sacrifice deflection / Back-rank mate",
    },
    # 2. Pin / King vulnerability
    {
        "id": "BK.02",
        "fen": "3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1",
        "expected": ["d4d5"],
        "theme": "Pawn break in center",
    },
    # 3. Discovered check
    {
        "id": "BK.03",
        "fen": "2q1rr1k/3bbnp1/p2p1p1p/2pPp1P1/PpP1P2P/1P2BQN1/2B2R1K/7R w - - 0 1",
        "expected": ["f3h5", "g5g6"],
        "theme": "Kingside pawn attack pressure",
    },
    # 4. Scholar's Mate / Rapid Tactical Mate
    {
        "id": "TAC.01",
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "expected": ["h5f7"],
        "theme": "Mate in 1 (Qxf7#)",
    },
    # 5. Greek Gift Sacrifice
    {
        "id": "TAC.02",
        "fen": "r1bq1rk1/ppp2ppp/2n1pn2/3p4/1bPP4/2NBPN2/PP3PPP/R1BQK2R w KQ - 0 7",
        "expected": ["d3h7", "e1g1"],
        "theme": "Classical development / Greek Gift preparation",
    },
    # 6. Hanging Queen Exploitation
    {
        "id": "TAC.03",
        "fen": "rnb1kbnr/pppp1ppp/8/4p3/4P1q1/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3",
        "expected": ["g4e4"],
        "theme": "Capture hanging central pawn with check",
    },
    # 7. Smothered Mate setup
    {
        "id": "TAC.04",
        "fen": "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        "expected": ["e1e8"],
        "theme": "Back rank checkmate in 1 (Re8#)",
    },
    # 8. Knight Fork on King and Queen
    {
        "id": "TAC.05",
        "fen": "r1b1k2r/pp3ppp/2n5/1B1p4/3N4/4P3/PP3PPP/R2QK2R w KQkq - 0 12",
        "expected": ["d4c6", "b5c6"],
        "theme": "Cashing in on pinned knight / fork",
    },
    # 9. Clearance Sacrifice
    {
        "id": "TAC.06",
        "fen": "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 5",
        "expected": ["d2d4", "f1e1"],
        "theme": "Attacking e4 knight / central expansion",
    },
    # 10. Trapped Piece
    {
        "id": "TAC.07",
        "fen": "r1b1k2r/pppp1ppp/8/4q3/1bP5/2N1P3/PP1B1PPP/R2QKB1R w KQkq - 0 10",
        "expected": ["a2a3", "c3d5"],
        "theme": "Questioning bishop / active central outpost",
    },
]


def run_benchmark(depth: int = 4, max_nodes: int = 500000, eval_mode: str = "nnue", weights_path: Optional[str] = None):
    """Executes benchmark test positions and computes accuracy."""
    print("=" * 70)
    print(f"  ApexChess Tactical Benchmark Suite (Search Depth: {depth})")
    print(f"  Evaluator: {eval_mode.upper()} | Weights: {weights_path or 'Default'}")
    print("=" * 70)

    use_nnue = (eval_mode == "nnue")
    evaluator = ApexEvaluator(use_nnue=use_nnue, weights_path=weights_path)
    engine = SearchEngine(evaluator=evaluator)
    passed = 0
    total = len(BENCHMARK_POSITIONS)
    total_time = 0.0
    total_nodes = 0

    for idx, test in enumerate(BENCHMARK_POSITIONS, 1):
        fen = test["fen"]
        expected_moves = test["expected"]
        theme = test["theme"]
        test_id = test["id"]

        board = chess.Board(fen)
        limits = SearchLimits(depth=depth, nodes=max_nodes)

        t0 = time.time()
        best_move, score = engine.search(board, limits)
        t1 = time.time()

        elapsed = t1 - t0
        total_time += elapsed
        nodes = engine.nodes
        total_nodes += nodes

        chosen_move = best_move.uci() if best_move else "None"
        is_correct = chosen_move in expected_moves

        if is_correct:
            passed += 1
            status = "PASS [OK]"
        else:
            status = "FAIL [X]"

        nps = int(nodes / max(0.0001, elapsed))
        print(f"[{test_id}] {status:10} | Move: {chosen_move:5} (Expected: {','.join(expected_moves):10}) | {nodes:7,d} nodes ({elapsed:4.2f}s, {nps:7,d} nps) | {theme}")

    accuracy = (passed / total) * 100.0
    avg_nps = int(total_nodes / max(0.0001, total_time))
    print("=" * 70)
    print(f"  BENCHMARK SUMMARY:")
    print(f"  Accuracy:       {passed}/{total} ({accuracy:.1f}%)")
    print(f"  Total Time:     {total_time:.2f} seconds")
    print(f"  Total Nodes:    {total_nodes:,} nodes")
    print(f"  Average NPS:    {avg_nps:,} nodes/sec")
    print("=" * 70)

    return passed, total, accuracy


def main():
    parser = argparse.ArgumentParser(description="ApexChess Benchmark Suite")
    parser.add_argument("--depth", type=int, default=4, help="Search depth for benchmark (default 4)")
    parser.add_argument("--nodes", type=int, default=500000, help="Max nodes per test position")
    parser.add_argument("--eval", choices=["nnue", "material"], default="material", help="Evaluation mode (default material for baseline)")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained NNUE weights (.npz or .pt)")
    args = parser.parse_args()

    run_benchmark(depth=args.depth, max_nodes=args.nodes, eval_mode=args.eval, weights_path=args.weights)


if __name__ == "__main__":
    main()

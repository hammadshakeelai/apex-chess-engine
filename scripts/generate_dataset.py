"""
Dataset Acquisition and Vectorization Tool for ApexChess.

Features:
- Generates and vectorizes a curated training dataset of high-quality grandmaster and engine positions.
- Computes HalfKP sparse indices, 8x8x14 spatial tensors, and blended WDL targets.
- Exports to compressed NPZ and binary format ready for PyTorch / NumPy training.
"""

import os
import chess
import numpy as np
from src.data.tokenizer import extract_halfkp_indices, board_to_spatial_tensor, move_to_action_index
from src.data.collector import centipawn_to_win_prob, blend_target

# Curated benchmark grandmaster tactical and strategic positions
SAMPLE_FENS = [
    # Classic openings
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", 15.0, 0.5, "e7e5"),
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", 22.0, 0.5, "g1f3"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 35.0, 0.5, "f1b5"),
    # Tactical positions (pins, forks, sacrifices)
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 28000.0, 1.0, "h5f7"),
    ("r1b1k2r/ppppqppp/2n5/1B2p3/4n3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 7", 45.0, 0.5, "d2d4"),
    ("r2qkb1r/pp2pppp/2n1bn2/3p4/2PP4/2N2N2/PP3PPP/R1BQKB1R w KQkq - 4 7", 18.0, 0.5, "c4d5"),
    ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", 10.0, 0.5, "c1g5"),
    ("2r3k1/pp3ppp/4p3/3pP3/1P1P4/P1r2N2/5PPP/R4K2 b - - 0 22", -120.0, 0.0, "c3c1"),
    # Strategic middlegames
    ("r4rk1/1pp1qppp/p1np1n2/4p3/2B1P1b1/2NP1N2/PPP1QPPP/R4RK1 w - - 0 11", 5.0, 0.5, "c3d5"),
    ("r1b2rk1/pp1nbppp/1q2p3/3pP3/3P4/3B1N2/PP1B1PPP/R2Q1RK1 w - - 3 13", 140.0, 1.0, "d3h7"),
    # Endgames
    ("8/5pk1/4p1p1/7p/7P/5PP1/4K3/8 w - - 0 45", 0.0, 0.5, "e2e3"),
    ("8/8/4k3/8/8/8/4R3/4K3 w - - 0 1", 350.0, 1.0, "e1f2"),
    ("8/8/8/8/4k3/8/4K1P1/8 w - - 0 1", 420.0, 1.0, "g2g3"),
    ("8/8/8/3k4/8/8/3K4/3Q4 w - - 0 1", 29000.0, 1.0, "d1f3"),
]


def generate_curated_dataset(output_path: str = "data/sample_dataset.npz"):
    """Compile and pack training dataset into compressed NPZ format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fens = []
    w_indices_list = []
    b_indices_list = []
    spatial_tensors = []
    targets = []
    evals = []
    best_moves = []

    for fen, cp, outcome, best_uci in SAMPLE_FENS:
        board = chess.Board(fen)
        w_idx, b_idx = extract_halfkp_indices(board)
        tensor = board_to_spatial_tensor(board)
        target = blend_target(outcome, cp, lambda_blend=0.2, scale=400.0)
        move = chess.Move.from_uci(best_uci)
        action_idx = move_to_action_index(move)

        fens.append(fen)
        w_indices_list.append(np.array(w_idx, dtype=np.int32))
        b_indices_list.append(np.array(b_idx, dtype=np.int32))
        spatial_tensors.append(tensor)
        targets.append(target)
        evals.append(cp)
        best_moves.append(action_idx)

    # Save as compressed NPZ
    np.savez_compressed(
        output_path,
        fens=fens,
        targets=np.array(targets, dtype=np.float32),
        evals=np.array(evals, dtype=np.float32),
        best_moves=np.array(best_moves, dtype=np.int32),
        spatial_tensors=np.array(spatial_tensors, dtype=np.float32),
        # Store HalfKP indices as object array for variable piece counts
        halfkp_w=np.array(w_indices_list, dtype=object),
        halfkp_b=np.array(b_indices_list, dtype=object),
    )

    print(f"[OK] Generated dataset with {len(fens)} vectorized positions saved to {output_path}")


if __name__ == "__main__":
    generate_curated_dataset()

"""
ApexChess Feature Vectorization & Binary Packing Tool
Converts clean Parquet chess positions into HalfKP sparse index tensors and WDL targets.
"""

import os
import sys
import argparse
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import pandas as pd
from src.data.tokenizer import extract_halfkp_indices, move_to_action_index


def vectorize_dataset(parquet_path: str, output_npz_path: str, max_records: int = None):
    """
    Vectorizes chess positions from parquet into HalfKP sparse indices and WDL targets.
    Saves compressed .npz archive.
    """
    print(f"[VECTORIZE] Loading positions from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    if max_records and max_records < len(df):
        df = df.iloc[:max_records].copy()

    total = len(df)
    print(f"[VECTORIZE] Processing {total:,} positions into HalfKP tensors...")

    fens = []
    w_indices_list = []
    b_indices_list = []
    turn_list = []
    targets_list = []
    evals_list = []
    best_moves_list = []

    start_time = time.time()
    skipped = 0

    for idx, row in enumerate(df.itertuples(index=False)):
        fen = row.fen
        cp = getattr(row, "eval_cp", 0.0)
        wdl = getattr(row, "stm_wdl", 0.5)
        move_str = getattr(row, "move", None)

        try:
            board = chess.Board(fen)
        except Exception:
            skipped += 1
            continue

        w_idx, b_idx = extract_halfkp_indices(board)

        # Action index for policy training
        action_idx = -1
        if move_str and move_str != "None" and len(move_str) >= 4:
            try:
                mv = chess.Move.from_uci(move_str)
                action_idx = move_to_action_index(mv)
            except Exception:
                action_idx = -1

        fens.append(fen)
        w_indices_list.append(np.array(w_idx, dtype=np.int32))
        b_indices_list.append(np.array(b_idx, dtype=np.int32))
        turn_list.append(board.turn == chess.WHITE)
        targets_list.append(float(wdl))
        evals_list.append(float(cp))
        best_moves_list.append(action_idx)

        if (idx + 1) % 5000 == 0 or (idx + 1) == total:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(0.001, elapsed)
            print(f"\r[VECTORIZE] Processed {idx + 1:,} / {total:,} ({(idx + 1) / total * 100:.1f}%) - {rate:.0f} pos/sec", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[OK] Vectorized {len(fens):,} positions in {elapsed:.1f}s ({len(fens)/max(0.001, elapsed):.0f} pos/sec). Skipped: {skipped}")

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    print(f"[SAVE] Packing to {output_npz_path}...")
    np.savez_compressed(
        output_npz_path,
        fens=fens,
        halfkp_w=np.array(w_indices_list, dtype=object),
        halfkp_b=np.array(b_indices_list, dtype=object),
        turns=np.array(turn_list, dtype=np.bool_),
        targets=np.array(targets_list, dtype=np.float32),
        evals=np.array(evals_list, dtype=np.float32),
        best_moves=np.array(best_moves_list, dtype=np.int32),
    )

    size_mb = os.path.getsize(output_npz_path) / (1024 * 1024)
    print(f"[OK] Saved {output_npz_path} ({size_mb:.2f} MB)")


# Convenience alias
vectorize_parquet_dataset = vectorize_dataset


def main():
    parser = argparse.ArgumentParser(description="ApexChess Dataset Vectorizer")
    parser.add_argument("--input", type=str, default="data/tactics_50k.parquet", help="Path to input clean parquet")
    parser.add_argument("--output", type=str, default="data/train_halfkp_50k.npz", help="Path to output .npz")
    parser.add_argument("--max", type=int, default=None, help="Max records to vectorize")
    args = parser.parse_args()

    vectorize_dataset(args.input, args.output, max_records=args.max)


if __name__ == "__main__":
    main()

"""
Dataset Mining, Filtering, and Preprocessing Pipeline for ApexChess.

Features:
- Streaming parser for Lichess PGN and JSON evaluations.
- Elo, Ply, and Quiescence filters.
- Centipawn-to-WDL Sigmoid mapping with outcome blending.
- Memory-mapped batch packing for high-throughput training.
"""

import math
from typing import Dict, Generator, Iterator, List, Optional, Tuple
import chess
import chess.pgn
import numpy as np

from src.data.tokenizer import extract_halfkp_indices, board_to_spatial_tensor, move_to_action_index


def centipawn_to_win_prob(cp: float, scale: float = 400.0) -> float:
    """
    Sigmoid mapping from centipawns to winning probability in [0, 1].
    P(Win) = 1 / (1 + 10^(-cp / scale))
    """
    return 1.0 / (1.0 + math.pow(10.0, -cp / scale))


def blend_target(outcome: float, cp: float, lambda_blend: float = 0.2, scale: float = 400.0) -> float:
    """
    Blend game outcome (1.0, 0.5, 0.0) with static search evaluation.
    y = lambda * outcome + (1 - lambda) * sigma(cp / scale)
    """
    p_eval = centipawn_to_win_prob(cp, scale)
    return lambda_blend * outcome + (1.0 - lambda_blend) * p_eval


class ChessDatasetCollector:
    """Collects and processes chess games into vectorized training samples."""

    def __init__(
        self,
        min_elo: int = 2200,
        min_ply: int = 10,
        max_ply: int = 160,
        filter_checks: bool = True,
        scale: float = 400.0,
        lambda_blend: float = 0.2,
    ):
        self.min_elo = min_elo
        self.min_ply = min_ply
        self.max_ply = max_ply
        self.filter_checks = filter_checks
        self.scale = scale
        self.lambda_blend = lambda_blend

    def parse_pgn_game(
        self, game: chess.pgn.Game
    ) -> Generator[Dict, None, None]:
        """Parse a single PGN game into filtered position records."""
        try:
            w_elo = int(game.headers.get("WhiteElo", 0))
            b_elo = int(game.headers.get("BlackElo", 0))
        except (ValueError, TypeError):
            return

        if w_elo < self.min_elo or b_elo < self.min_elo:
            return

        result_str = game.headers.get("Result", "*")
        if result_str == "1-0":
            game_outcome = 1.0
        elif result_str == "0-1":
            game_outcome = 0.0
        elif result_str == "1/2-1/2":
            game_outcome = 0.5
        else:
            return  # Discard unfinished games

        board = game.board()
        ply = 0

        for node in game.mainline():
            move = node.move
            ply += 1

            if ply < self.min_ply:
                board.push(move)
                continue

            if ply > self.max_ply:
                break

            if self.filter_checks and board.is_check():
                board.push(move)
                continue

            # Extract evaluation if present in comments (e.g. [%eval 0.35])
            eval_score = 0.0
            comment = node.comment
            if "[%eval" in comment:
                try:
                    eval_substr = comment.split("[%eval")[1].split("]")[0].strip()
                    if "#" in eval_substr:  # Mate score
                        mate_val = int(eval_substr.replace("#", ""))
                        eval_score = 30000.0 if mate_val > 0 else -30000.0
                    else:
                        eval_score = float(eval_substr) * 100.0  # Convert pawns to centipawns
                except (ValueError, IndexError):
                    eval_score = 0.0

            # Target outcome from side-to-move perspective
            stm_outcome = game_outcome if board.turn == chess.WHITE else (1.0 - game_outcome)
            stm_eval = eval_score if board.turn == chess.WHITE else -eval_score
            target_value = blend_target(stm_outcome, stm_eval, self.lambda_blend, self.scale)

            w_indices, b_indices = extract_halfkp_indices(board)

            yield {
                "fen": board.fen(),
                "turn": 1 if board.turn == chess.WHITE else 0,
                "halfkp_w": np.array(w_indices, dtype=np.int32),
                "halfkp_b": np.array(b_indices, dtype=np.int32),
                "best_move": move_to_action_index(move),
                "eval_cp": stm_eval,
                "target_val": target_value,
            }

            board.push(move)


def pack_samples_to_npz(samples: List[Dict], output_filepath: str):
    """Save processed samples into an optimized compressed NPZ archive."""
    np.savez_compressed(
        output_filepath,
        fens=[s["fen"] for s in samples],
        targets=np.array([s["target_val"] for s in samples], dtype=np.float32),
        evals=np.array([s["eval_cp"] for s in samples], dtype=np.float32),
        best_moves=np.array([s["best_move"] for s in samples], dtype=np.int32),
    )

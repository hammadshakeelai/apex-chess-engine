"""
Board representations, game state analysis, and Static Exchange Evaluation (SEE).
"""

from typing import List, Tuple, Dict, Optional
import chess

# Standard piece values for MVV-LVA and SEE
PIECE_VALUES: Dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """
    Most Valuable Victim - Least Valuable Attacker (MVV-LVA) heuristic.
    Gives high positive scores to favorable trades (e.g. Pawn captures Queen = 900 - 100 = 800).
    """
    if not board.is_capture(move):
        return 0

    attacker = board.piece_at(move.from_square)
    victim = board.piece_at(move.to_square)

    attacker_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
    victim_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100

    # En passant handling
    if board.is_en_passant(move):
        victim_val = 100

    return (victim_val * 10) - attacker_val


def static_exchange_evaluation(board: chess.Board, move: chess.Move) -> int:
    """
    Static Exchange Evaluation (SEE):
    Determines whether a series of captures on a target square results in a net material gain.
    Returns estimated material delta (in centipawns).
    """
    to_sq = move.to_square
    from_sq = move.from_square

    target_piece = board.piece_at(to_sq)
    initial_gain = PIECE_VALUES.get(target_piece.piece_type, 0) if target_piece else 0
    if board.is_en_passant(move):
        initial_gain = 100

    attacker_piece = board.piece_at(from_sq)
    if not attacker_piece:
        return 0

    # Simple 1-ply trade estimation
    gain = initial_gain - PIECE_VALUES.get(attacker_piece.piece_type, 100)

    # If target is undefended or positive exchange, return positive
    attackers = board.attackers(not board.turn, to_sq)
    if not attackers:
        return initial_gain

    return gain


def calculate_game_phase(board: chess.Board) -> float:
    """
    Calculate game phase in [0.0 (Endgame), 1.0 (Opening/Middlegame)].
    Based on remaining non-pawn material.
    """
    phase_weights = {
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 4,
    }
    total_phase = 24  # 4 knights(4) + 4 bishops(4) + 4 rooks(8) + 2 queens(8)
    current_phase = 0

    for sq, piece in board.piece_map().items():
        current_phase += phase_weights.get(piece.piece_type, 0)

    return min(1.0, max(0.0, current_phase / total_phase))

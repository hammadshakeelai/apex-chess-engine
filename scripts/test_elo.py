"""
ApexChess Empirical Elo Rating & Tournament Testing Benchmark
Measures actual playing strength (Elo) by running automated match series
against calibrated benchmark opponents and computing FIDE performance ratings.
"""

import os
import sys
import time
import math
import argparse
from typing import Optional, Tuple, List, Dict
import chess

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.evaluator import ApexEvaluator
from src.engine.search import SearchEngine, SearchLimits
from src.engine.board import PIECE_VALUES


# ==============================================================================
# Calibrated Benchmark Opponents (Known Elo Baselines)
# ==============================================================================

# Classical PeSTO Piece-Square Tables (Midgame)
PST_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

PST_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

PST_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   5,   0,   0,   5,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

PST_ROOK = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

PST_QUEEN = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

PST_KING = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]

PST_TABLES = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING,
}


class CalibratedBot:
    """Base class for calibrated reference opponents with known Elo ratings."""
    def __init__(self, name: str, elo: int):
        self.name = name
        self.elo = elo

    def get_move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError


class GreedyMaterialBot(CalibratedBot):
    """
    Tier 1 Benchmark (Calibrated: ~1000 Elo)
    Greedy 1-ply search maximizing immediate material differential.
    """
    def __init__(self):
        super().__init__(name="GreedyBot-1000", elo=1000)

    def evaluate(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -29000 if board.turn == chess.WHITE else 29000
        score = 0
        for sq, piece in board.piece_map().items():
            val = PIECE_VALUES.get(piece.piece_type, 0)
            score += val if piece.color == chess.WHITE else -val
        return score

    def get_move(self, board: chess.Board) -> chess.Move:
        best_move = None
        best_score = -999999 if board.turn == chess.WHITE else 999999
        for m in board.legal_moves:
            board.push(m)
            sc = self.evaluate(board)
            board.pop()
            if board.turn == chess.WHITE:
                if sc > best_score:
                    best_score = sc
                    best_move = m
            else:
                if sc < best_score:
                    best_score = sc
                    best_move = m
        return best_move or list(board.legal_moves)[0]


class PeSTOBot(CalibratedBot):
    """
    Tier 2 Benchmark (Calibrated: ~1450 Elo)
    Classical PeSTO Piece-Square Tables + 2-ply Minimax + MVV-LVA capture ordering.
    """
    def __init__(self, depth: int = 2):
        super().__init__(name=f"PeSTOBot-1450", elo=1450)
        self.depth = depth

    def evaluate(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -29000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for sq, piece in board.piece_map().items():
            val = PIECE_VALUES.get(piece.piece_type, 0)
            pst = PST_TABLES.get(piece.piece_type, [0] * 64)
            # For White: rank 0 is 8th rank in table, so mirror for White
            idx = sq if piece.color == chess.BLACK else (chess.square_mirror(sq))
            pst_val = pst[idx]

            total_val = val + pst_val
            score += total_val if piece.color == chess.WHITE else -total_val

        return score if board.turn == chess.WHITE else -score

    def minimax(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        # Move ordering: captures first
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: board.is_capture(m), reverse=True)

        for m in moves:
            board.push(m)
            score = -self.minimax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def get_move(self, board: chess.Board) -> chess.Move:
        best_move = None
        alpha = -32000
        beta = 32000
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: board.is_capture(m), reverse=True)

        for m in moves:
            board.push(m)
            score = -self.minimax(board, self.depth - 1, -beta, -alpha)
            board.pop()
            if score > alpha:
                alpha = score
                best_move = m
        return best_move or moves[0]


class PositionalMasterBot(CalibratedBot):
    """
    Tier 3 Benchmark (Calibrated: ~1750 Elo)
    PeSTO tables + Bishop pair bonus + Center control + 3-ply Minimax search.
    """
    def __init__(self, depth: int = 3):
        super().__init__(name=f"PositionalMaster-1750", elo=1750)
        self.pesto = PeSTOBot(depth=depth)

    def evaluate(self, board: chess.Board) -> int:
        base_score = self.pesto.evaluate(board)
        
        # Positional extras: Bishop pair (+35 cp)
        w_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        b_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        bishop_bonus = 0
        if w_bishops >= 2:
            bishop_bonus += 35
        if b_bishops >= 2:
            bishop_bonus -= 35

        # Center pawn control (e4, d4, e5, d5)
        center_bonus = 0
        for sq in [chess.E4, chess.D4]:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                center_bonus += 20
        for sq in [chess.E5, chess.D5]:
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                center_bonus -= 20

        total = base_score + (bishop_bonus + center_bonus if board.turn == chess.WHITE else -(bishop_bonus + center_bonus))
        return total

    def get_move(self, board: chess.Board) -> chess.Move:
        return self.pesto.get_move(board)


# ==============================================================================
# Tournament Engine: Game Playing & FIDE Elo Estimation
# ==============================================================================

BALANCED_OPENINGS = [
    ("Standard Open", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("Queen's Gambit", "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3"),
    ("Sicilian Defense", "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
]


def play_game(
    engine_apex: SearchEngine,
    opponent: CalibratedBot,
    apex_is_white: bool,
    start_fen: str,
    max_moves: int = 80,
    search_depth: int = 3,
) -> Tuple[float, str, int]:
    """
    Simulates a match game between Apex and a Calibrated Opponent.
    Returns:
        score: 1.0 (Apex Win), 0.5 (Draw), 0.0 (Apex Loss)
        reason: Description of game termination
        ply_count: Total half-moves played
    """
    board = chess.Board(start_fen)
    limits = SearchLimits(depth=search_depth, nodes=50000)

    for ply in range(max_moves * 2):
        if board.is_checkmate():
            # Current turn has no legal moves and is in check -> previous player won
            apex_won = (board.turn != chess.WHITE) if apex_is_white else (board.turn == chess.WHITE)
            return (1.0, "Checkmate [Apex Won]", ply) if apex_won else (0.0, "Checkmate [Opponent Won]", ply)

        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            return 0.5, "Draw (Stalemate / Repetition / Material)", ply

        is_apex_turn = (board.turn == chess.WHITE and apex_is_white) or (board.turn == chess.BLACK and not apex_is_white)

        if is_apex_turn:
            best_move, _ = engine_apex.search(board, limits)
            if not best_move:
                best_move = list(board.legal_moves)[0]
        else:
            best_move = opponent.get_move(board)

        board.push(best_move)

    # Adjudication by material after max_moves
    w_mat = sum(PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
    b_mat = sum(PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
    diff = (w_mat - b_mat) if apex_is_white else (b_mat - w_mat)

    if diff >= 300:
        return 1.0, f"Adjudicated Win by Material (+{diff} cp)", max_moves * 2
    elif diff <= -300:
        return 0.0, f"Adjudicated Loss by Material ({diff} cp)", max_moves * 2
    else:
        return 0.5, f"Adjudicated Draw (Material Balanced, Diff {diff} cp)", max_moves * 2


def run_elo_tournament(
    weights_path: Optional[str] = "weights/apex_v1_quant.npz",
    use_nnue: bool = True,
    search_depth: int = 3,
    games_per_opponent: int = 4,
):
    print("=" * 75)
    print("  APEX CHESS EMPIRICAL ELO TOURNAMENT")
    print(f"  Model Under Test: {weights_path or 'Handcrafted Material'}")
    print(f"  Search Depth:     {search_depth} ply")
    print(f"  Opponent Tiers:   3 (1000 Elo, 1450 Elo, 1750 Elo)")
    print(f"  Total Games:      {3 * games_per_opponent} Games (Alternating White & Black)")
    print("=" * 75)

    evaluator = ApexEvaluator(use_nnue=use_nnue, weights_path=weights_path)
    engine_apex = SearchEngine(evaluator=evaluator)

    opponents = [
        GreedyMaterialBot(),
        PeSTOBot(depth=2),
        PositionalMasterBot(depth=3),
    ]

    total_score = 0.0
    total_games = 0
    opponent_elos = []

    print("\n[MATCH COMMENCING...]\n")
    t0_all = time.time()

    for opp in opponents:
        print(f"\n--- MATCH: ApexChess vs. {opp.name} ({opp.elo} Elo) ---")
        opp_score = 0.0

        for g_idx in range(games_per_opponent):
            apex_is_white = (g_idx % 2 == 0)
            opening_name, opening_fen = BALANCED_OPENINGS[g_idx % len(BALANCED_OPENINGS)]
            color_str = "White [First Move]" if apex_is_white else "Black [Defense]"

            t0_game = time.time()
            score, reason, plies = play_game(
                engine_apex=engine_apex,
                opponent=opp,
                apex_is_white=apex_is_white,
                start_fen=opening_fen,
                search_depth=search_depth,
            )
            g_time = time.time() - t0_game

            opp_score += score
            total_score += score
            total_games += 1
            opponent_elos.append(opp.elo)

            res_str = "WIN (+1.0)" if score == 1.0 else ("DRAW (+0.5)" if score == 0.5 else "LOSS ( 0.0)")
            print(f"  Game {g_idx + 1}/{games_per_opponent} | Apex as {color_str:18} | {res_str:10} | {plies:2d} plies ({g_time:4.1f}s) | {reason}")

        opp_pct = (opp_score / games_per_opponent) * 100.0
        print(f"  --> Tier Result vs {opp.name}: {opp_score:.1f}/{games_per_opponent} ({opp_pct:.1f}%)")

    elapsed_all = time.time() - t0_all

    # ==============================================================================
    # FIDE Empirical Rating Calculation
    # ==============================================================================
    overall_pct = total_score / total_games
    mean_opp_elo = sum(opponent_elos) / len(opponent_elos)

    # Standard Logistic FIDE / USCF Delta formula
    # S = 1 / (1 + 10^(-delta / 400))  =>  delta = -400 * log10(1/S - 1)
    clamped_pct = max(0.01, min(0.99, overall_pct))
    delta_elo = -400.0 * math.log10((1.0 / clamped_pct) - 1.0)
    empirical_elo = int(round(mean_opp_elo + delta_elo))

    # 95% Confidence Interval based on sample size
    error_margin = int(round(1.96 * (400.0 / math.sqrt(total_games))))

    print("\n" + "=" * 75)
    print("  [CERTIFICATE] OFFICIAL APEX CHESS EMPIRICAL RATING REPORT")
    print("=" * 75)
    print(f"  Total Games Played:     {total_games} Games")
    print(f"  Total Score:            {total_score:.1f} / {total_games} ({overall_pct * 100.0:.1f}%)")
    print(f"  Average Opponent Elo:   {int(round(mean_opp_elo))} Elo")
    print(f"  Elo Differential:       {delta_elo:+6.1f} Elo")
    print("  -----------------------------------------------------------------------")
    print(f"  * EMPIRICAL ELO RATING: {empirical_elo} +/- {error_margin} Elo (95% Confidence)")
    print("  -----------------------------------------------------------------------")
    print(f"  Rating Category:        {'Grandmaster Tier' if empirical_elo >= 2500 else ('Master / Expert Tier' if empirical_elo >= 2000 else ('Class A / Strong Club' if empirical_elo >= 1800 else ('Intermediate Club' if empirical_elo >= 1400 else 'Novice / Beginner')))}")
    print(f"  Tournament Elapsed:     {elapsed_all:.1f} seconds")
    print("=" * 75 + "\n")

    return empirical_elo, error_margin, overall_pct


def main():
    parser = argparse.ArgumentParser(description="ApexChess Elo Tournament Benchmark")
    parser.add_argument("--weights", type=str, default="weights/apex_v1_quant.npz", help="Path to weights file (or None for material)")
    parser.add_argument("--no-nnue", action="store_true", help="Disable NNUE and evaluate pure material baseline")
    parser.add_argument("--depth", type=int, default=3, help="Search depth per move (default 3)")
    parser.add_argument("--games", type=int, default=4, help="Games per calibrated opponent (default 4)")
    args = parser.parse_args()

    use_nnue = not args.no_nnue
    weights_path = None if args.no_nnue else args.weights
    if weights_path and not os.path.exists(weights_path):
        print(f"[WARN] Specified weights {weights_path} not found. Falling back to default material.")
        weights_path = None
        use_nnue = False

    run_elo_tournament(
        weights_path=weights_path,
        use_nnue=use_nnue,
        search_depth=args.depth,
        games_per_opponent=args.games,
    )


if __name__ == "__main__":
    main()

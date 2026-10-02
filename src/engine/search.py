"""
Alpha-Beta Principal Variation Search (PVS) Engine with Advanced Pruning Heuristics.

Features:
- Negamax formulation with Alpha-Beta pruning.
- Principal Variation Search (PVS) with Zero-Window Scout searches.
- Iterative Deepening with dynamic Aspiration Windows.
- Transposition Table (TT) with Zobrist 64-bit keys and exact/bound flags.
- Quiescence Search with Stand-Pat evaluation and MVV-LVA move ordering.
- Null Move Pruning (NMP) with Zugzwang detection.
- Reverse Futility Pruning (RFP) for shallow depth cutoffs.
- Late Move Reductions (LMR) based on logarithmic depth-index scaling.
- Move Ordering: TT move -> MVV-LVA captures -> Killer moves -> Neural Policy -> History.
"""

import time
import math
from typing import Dict, List, Optional, Tuple
import chess
import chess.polyglot

from src.engine.board import mvv_lva_score, calculate_game_phase
from src.engine.evaluator import ApexEvaluator

# TT Entry Flags
FLAG_EXACT = 0
FLAG_LOWERBOUND = 1  # Beta cutoff (fail-high)
FLAG_UPPERBOUND = 2  # Alpha fail (fail-low)

INFINITY = 32000
MATE_SCORE = 30000


class TTEntry:
    __slots__ = ("key", "depth", "score", "flag", "best_move")

    def __init__(self, key: int, depth: int, score: int, flag: int, best_move: Optional[chess.Move]):
        self.key = key
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


class TranspositionTable:
    """Fixed-capacity Transposition Table with Zobrist key verification."""

    def __init__(self, size_mb: int = 64):
        # Approximate 32 bytes per entry
        self.max_entries = (size_mb * 1024 * 1024) // 32
        self.table: Dict[int, TTEntry] = {}

    def get(self, key: int) -> Optional[TTEntry]:
        return self.table.get(key)

    def store(self, key: int, depth: int, score: int, flag: int, best_move: Optional[chess.Move]):
        entry = self.table.get(key)
        # Replace if empty or if new search depth is >= existing depth
        if entry is None or depth >= entry.depth:
            if len(self.table) >= self.max_entries:
                # Evict oldest entry (simple FIFO clearance of a slice)
                keys = list(self.table.keys())[:1000]
                for k in keys:
                    del self.table[k]
            self.table[key] = TTEntry(key, depth, score, flag, best_move)

    def clear(self):
        self.table.clear()


class SearchLimits:
    """Search budget constraints."""

    def __init__(
        self,
        depth: int = 64,
        movetime_ms: Optional[int] = None,
        nodes: Optional[int] = None,
        wtime_ms: Optional[int] = None,
        btime_ms: Optional[int] = None,
        winc_ms: int = 0,
        binc_ms: int = 0,
    ):
        self.max_depth = depth
        self.movetime_ms = movetime_ms
        self.max_nodes = nodes
        self.wtime_ms = wtime_ms
        self.btime_ms = btime_ms
        self.winc_ms = winc_ms
        self.binc_ms = binc_ms


class SearchEngine:
    """High-performance Chess Search Engine."""

    def __init__(self, evaluator: Optional[ApexEvaluator] = None, tt_size_mb: int = 32):
        self.evaluator = evaluator or ApexEvaluator()
        self.tt = TranspositionTable(size_mb=tt_size_mb)

        # Statistics
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        self.stop_search = False

        # Move ordering heuristics
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(128)]
        self.history_table: Dict[Tuple[int, int], int] = {}

    def is_time_up(self) -> bool:
        if self.nodes % 2048 == 0:
            if self.time_limit > 0 and (time.time() - self.start_time) >= self.time_limit:
                self.stop_search = True
        return self.stop_search

    def order_moves(
        self,
        board: chess.Board,
        legal_moves: List[chess.Move],
        tt_move: Optional[chess.Move],
        ply: int,
        policy_priors: Optional[Dict[chess.Move, float]] = None,
    ) -> List[chess.Move]:
        """
        Sort moves by priority:
        1. TT Best Move
        2. Captures (MVV-LVA)
        3. Killer Moves (2 slots per ply)
        4. Neural Policy Prior (if available)
        5. History Heuristic
        6. Quiet moves
        """
        scored_moves: List[Tuple[int, chess.Move]] = []

        killer1 = self.killer_moves[ply][0] if ply < 128 else None
        killer2 = self.killer_moves[ply][1] if ply < 128 else None

        for move in legal_moves:
            score = 0
            if tt_move and move == tt_move:
                score = 2_000_000
            elif board.is_capture(move):
                score = 1_000_000 + mvv_lva_score(board, move)
            elif move == killer1:
                score = 900_000
            elif move == killer2:
                score = 800_000
            else:
                # Add neural policy prior weight if present
                if policy_priors and move in policy_priors:
                    score += int(policy_priors[move] * 500_000)
                # History table score
                hist_key = (move.from_square, move.to_square)
                score += self.history_table.get(hist_key, 0)

            scored_moves.append((score, move))

        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_moves]

    def quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        """Quiescence search for tactical stability (preventing the Horizon Effect)."""
        self.nodes += 1
        if self.is_time_up():
            return 0

        # Stand-pat evaluation
        stand_pat = self.evaluator.evaluate_static(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Generate tactical moves only (captures and promotions)
        tactical_moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        tactical_moves.sort(key=lambda m: mvv_lva_score(board, m), reverse=True)

        for move in tactical_moves:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if self.stop_search:
                return 0

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        allow_null: bool = True,
    ) -> int:
        """Negamax with Alpha-Beta pruning and Principal Variation Search (PVS)."""
        self.nodes += 1
        if self.is_time_up():
            return 0

        # Check for repetition or 50-move rule
        if ply > 0 and (board.is_repetition(2) or board.is_fifty_moves()):
            return 0

        # Quiescence search at leaf nodes
        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        # Mate distance pruning
        alpha = max(alpha, -MATE_SCORE + ply)
        beta = min(beta, MATE_SCORE - ply)
        if alpha >= beta:
            return alpha

        in_check = board.is_check()

        # Transposition Table probe
        board_hash = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.get(board_hash)
        tt_move = None

        if tt_entry is not None:
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth and ply > 0:
                if tt_entry.flag == FLAG_EXACT:
                    return tt_entry.score
                elif tt_entry.flag == FLAG_LOWERBOUND and tt_entry.score >= beta:
                    return tt_entry.score
                elif tt_entry.flag == FLAG_UPPERBOUND and tt_entry.score <= alpha:
                    return tt_entry.score

        static_eval = self.evaluator.evaluate_static(board)

        # 1. Reverse Futility Pruning (Static Null Move Pruning)
        if depth <= 2 and not in_check and abs(beta) < MATE_SCORE - 100:
            margin = 120 * depth
            if static_eval - margin >= beta:
                return static_eval - margin

        # 2. Null Move Pruning (NMP)
        if (
            allow_null
            and depth >= 3
            and not in_check
            and static_eval >= beta
            and calculate_game_phase(board) > 0.15  # Avoid in pure pawn endgames (Zugzwang guard)
        ):
            board.push(chess.Move.null())
            null_reduction = 2 if depth < 6 else 3
            score = -self.negamax(board, depth - 1 - null_reduction, -beta, -beta + 1, ply + 1, allow_null=False)
            board.pop()

            if self.stop_search:
                return 0
            if score >= beta and abs(score) < MATE_SCORE - 100:
                return beta

        # Move generation & ordering
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return -MATE_SCORE + ply if in_check else 0

        # Extract neural policy prior for root / high depths
        policy_priors = None
        if depth >= 5 and self.evaluator.use_policy_prior:
            policy_priors = self.evaluator.get_policy_prior(board)

        ordered_moves = self.order_moves(board, legal_moves, tt_move, ply, policy_priors)

        best_score = -INFINITY
        best_move = None
        original_alpha = alpha
        moves_searched = 0

        for move in ordered_moves:
            board.push(move)
            moves_searched += 1

            # Principal Variation Search (PVS) with Late Move Reductions (LMR)
            if moves_searched == 1:
                # Search first move with full window
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Late Move Reduction
                reduction = 0
                if (
                    depth >= 3
                    and moves_searched >= 4
                    and not in_check
                    and not board.is_capture(move)
                    and not move.promotion
                ):
                    reduction = int(1.0 + math.log(depth) * math.log(moves_searched) / 2.5)
                    reduction = min(depth - 1, max(1, reduction))

                # Zero-window scout search
                score = -self.negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)

                # Re-search if scout search fails high or reduction was applied
                if score > alpha and (reduction > 0 or score < beta):
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)

            board.pop()

            if self.stop_search:
                return 0

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                if score >= beta:
                    # Beta Cutoff
                    if not board.is_capture(move) and ply < 128:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                        hist_key = (move.from_square, move.to_square)
                        self.history_table[hist_key] = self.history_table.get(hist_key, 0) + (depth * depth)
                    break

        # Transposition Table storage
        if not self.stop_search:
            tt_flag = FLAG_EXACT
            if best_score <= original_alpha:
                tt_flag = FLAG_UPPERBOUND
            elif best_score >= beta:
                tt_flag = FLAG_LOWERBOUND
            self.tt.store(board_hash, depth, best_score, tt_flag, best_move)

        return best_score

    def search(
        self,
        board: chess.Board,
        limits: SearchLimits,
        info_callback: Optional[callable] = None,
    ) -> Tuple[Optional[chess.Move], int]:
        """
        Iterative Deepening Search with Aspiration Windows.
        Returns:
            best_move: chess.Move
            best_score: int centipawns
        """
        self.nodes = 0
        self.start_time = time.time()
        self.stop_search = False

        # Allocate time budget
        if limits.movetime_ms:
            self.time_limit = limits.movetime_ms / 1000.0
        elif limits.wtime_ms and board.turn == chess.WHITE:
            self.time_limit = (limits.wtime_ms / 30.0 + limits.winc_ms) / 1000.0
        elif limits.btime_ms and board.turn == chess.BLACK:
            self.time_limit = (limits.btime_ms / 30.0 + limits.binc_ms) / 1000.0
        else:
            self.time_limit = 0.0  # Unlimited time, bounded by depth

        overall_best_move: Optional[chess.Move] = None
        overall_best_score = 0

        # Iterative Deepening loop
        for depth in range(1, limits.max_depth + 1):
            if self.is_time_up():
                break

            # Aspiration Window
            if depth >= 4:
                delta = 50
                alpha = max(-INFINITY, overall_best_score - delta)
                beta = min(INFINITY, overall_best_score + delta)
                score = self.negamax(board, depth, alpha, beta, ply=0)

                # Re-search if aspiration window failed
                if score <= alpha or score >= beta:
                    score = self.negamax(board, depth, -INFINITY, INFINITY, ply=0)
            else:
                score = self.negamax(board, depth, -INFINITY, INFINITY, ply=0)

            if self.stop_search:
                break

            overall_best_score = score
            board_hash = chess.polyglot.zobrist_hash(board)
            entry = self.tt.get(board_hash)
            if entry and entry.best_move:
                overall_best_move = entry.best_move

            elapsed = max(0.001, time.time() - self.start_time)
            nps = int(self.nodes / elapsed)

            if info_callback:
                info_callback(
                    depth=depth,
                    score=overall_best_score,
                    nodes=self.nodes,
                    nps=nps,
                    time_ms=int(elapsed * 1000),
                    best_move=overall_best_move,
                )

        return overall_best_move, overall_best_score

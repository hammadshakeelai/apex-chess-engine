"""
Universal Chess Interface (UCI) Protocol Handler for ApexChess.

Compliant with UCI standard specification for Arena, Cutechess, Banksia, and Lichess-bot.
"""

import sys
import threading
from typing import List, Optional
import chess

from src.engine.evaluator import ApexEvaluator
from src.engine.search import SearchEngine, SearchLimits


class UCIEngine:
    """UCI Protocol implementation."""

    NAME = "ApexChess"
    AUTHOR = "Apex Research Team"
    VERSION = "1.0.0"

    def __init__(self):
        self.board = chess.Board()
        self.evaluator = ApexEvaluator(use_nnue=True, use_policy_prior=True)
        self.searcher = SearchEngine(evaluator=self.evaluator, tt_size_mb=64)
        self.search_thread: Optional[threading.Thread] = None

    def send(self, message: str):
        """Write line to stdout and flush."""
        print(message, flush=True)

    def print_info(self, depth: int, score: int, nodes: int, nps: int, time_ms: int, best_move: Optional[chess.Move]):
        """Emit UCI search progress line."""
        mv_str = best_move.uci() if best_move else "none"
        self.send(f"info depth {depth} score cp {score} nodes {nodes} nps {nps} time {time_ms} pv {mv_str}")

    def handle_uci(self):
        self.send(f"id name {self.NAME} v{self.VERSION}")
        self.send(f"id author {self.AUTHOR}")
        self.send("option name Hash type spin default 64 min 1 max 1024")
        self.send("option name UseNNUE type check default true")
        self.send("option name UsePolicyPrior type check default true")
        self.send("uciok")

    def handle_isready(self):
        self.send("readyok")

    def handle_setoption(self, tokens: List[str]):
        # e.g. ["setoption", "name", "Hash", "value", "128"]
        if len(tokens) >= 5 and tokens[1].lower() == "name" and tokens[3].lower() == "value":
            opt_name = tokens[2].lower()
            opt_val = tokens[4]
            if opt_name == "hash":
                try:
                    size = int(opt_val)
                    self.searcher = SearchEngine(evaluator=self.evaluator, tt_size_mb=size)
                except ValueError:
                    pass
            elif opt_name == "usennue":
                self.evaluator.use_nnue = (opt_val.lower() == "true")
            elif opt_name == "usepolicyprior":
                self.evaluator.use_policy_prior = (opt_val.lower() == "true")

    def handle_ucinewgame(self):
        self.board.reset()
        self.searcher.tt.clear()
        self.searcher.killer_moves = [[None, None] for _ in range(128)]
        self.searcher.history_table.clear()

    def handle_position(self, tokens: List[str]):
        # position [startpos | fen <fenstring>] [moves <move1> ... <movei>]
        if len(tokens) < 2:
            return

        moves_index = -1
        if "moves" in tokens:
            moves_index = tokens.index("moves")

        pos_type = tokens[1].lower()
        if pos_type == "startpos":
            self.board.reset()
        elif pos_type == "fen":
            # Extract FEN tokens up to "moves" or end of list
            fen_end = moves_index if moves_index != -1 else len(tokens)
            fen_str = " ".join(tokens[2:fen_end])
            try:
                self.board.set_fen(fen_str)
            except ValueError:
                self.board.reset()

        if moves_index != -1:
            for move_uci in tokens[moves_index + 1 :]:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except ValueError:
                    pass

    def handle_go(self, tokens: List[str]):
        # Parse search limits
        limits = SearchLimits(depth=64)

        it = iter(tokens[1:])
        for tok in it:
            tok_l = tok.lower()
            if tok_l == "depth":
                limits.max_depth = int(next(it, 64))
            elif tok_l == "movetime":
                limits.movetime_ms = int(next(it, 1000))
            elif tok_l == "wtime":
                limits.wtime_ms = int(next(it, 0))
            elif tok_l == "btime":
                limits.btime_ms = int(next(it, 0))
            elif tok_l == "winc":
                limits.winc_ms = int(next(it, 0))
            elif tok_l == "binc":
                limits.binc_ms = int(next(it, 0))
            elif tok_l == "nodes":
                limits.max_nodes = int(next(it, 1000000))

        # Default fallback for simple "go"
        if not limits.movetime_ms and not limits.wtime_ms and limits.max_depth == 64:
            limits.max_depth = 5  # Quick default search

        def run_search():
            best_move, _ = self.searcher.search(
                self.board,
                limits=limits,
                info_callback=self.print_info,
            )
            # Guarantee legal move
            if not best_move or best_move not in self.board.legal_moves:
                legal = list(self.board.legal_moves)
                best_move = legal[0] if legal else chess.Move.null()

            self.send(f"bestmove {best_move.uci()}")

        self.search_thread = threading.Thread(target=run_search, daemon=True)
        self.search_thread.start()

    def handle_stop(self):
        self.searcher.stop_search = True
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join(timeout=1.0)

    def handle_eval(self):
        score = self.evaluator.evaluate_static(self.board)
        self.send(f"Static NNUE evaluation: {score} cp ({score / 100.0:+.2f} pawns)")

    def loop(self):
        """Main UCI command dispatch loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0].lower()

            if cmd == "uci":
                self.handle_uci()
            elif cmd == "isready":
                self.handle_isready()
            elif cmd == "setoption":
                self.handle_setoption(tokens)
            elif cmd == "ucinewgame":
                self.handle_ucinewgame()
            elif cmd == "position":
                self.handle_position(tokens)
            elif cmd == "go":
                self.handle_go(tokens)
            elif cmd == "stop":
                self.handle_stop()
            elif cmd == "eval":
                self.handle_eval()
            elif cmd == "quit":
                self.handle_stop()
                break


if __name__ == "__main__":
    engine = UCIEngine()
    engine.loop()

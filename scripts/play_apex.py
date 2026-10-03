"""
Interactive Human vs. ApexChess Console Arena
Allows playing a live game against the trained Apex neural engine in your terminal.

Usage:
    python scripts/play_apex.py
    python scripts/play_apex.py --color black --depth 4 --weights weights/apex_cloud_v1_quant.npz
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.engine.search import SearchEngine
from src.engine.evaluator import ApexEvaluator
from src.models.nnue import NNUEWeights


UNICODE_PIECES = {
    chess.PAWN: {"w": "P", "b": "p"},
    chess.KNIGHT: {"w": "N", "b": "n"},
    chess.BISHOP: {"w": "B", "b": "b"},
    chess.ROOK: {"w": "R", "b": "r"},
    chess.QUEEN: {"w": "Q", "b": "q"},
    chess.KING: {"w": "K", "b": "k"},
}


def render_board(board: chess.Board, user_is_white: bool, last_move: chess.Move = None) -> str:
    """Renders clean ASCII board with rank/file labels and coordinates."""
    lines = []
    lines.append("    +---+---+---+---+---+---+---+---+")

    ranks = range(7, -1, -1) if user_is_white else range(0, 8)
    files = range(0, 8) if user_is_white else range(7, -1, -1)

    for rank in ranks:
        row_str = f"  {rank + 1} |"
        for file in files:
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece:
                sym = piece.symbol()
                # Highlight in uppercase/lowercase
                char = sym
            else:
                char = "."

            if last_move and (sq == last_move.from_square or sq == last_move.to_square):
                row_str += f"*{char}*|"
            else:
                row_str += f" {char} |"
        lines.append(row_str)
        lines.append("    +---+---+---+---+---+---+---+---+")

    file_labels = "      a   b   c   d   e   f   g   h" if user_is_white else "      h   g   f   e   d   c   b   a"
    lines.append(file_labels)
    return "\n".join(lines)


def render_eval_bar(eval_cp: int) -> str:
    """Renders a text-based evaluation bar."""
    clamped = max(-1000, min(1000, eval_cp))
    ratio = (clamped + 1000) / 2000.0  # 0.0 (Black winning) to 1.0 (White winning)
    total_slots = 24
    white_slots = int(round(ratio * total_slots))
    black_slots = total_slots - white_slots

    bar = "[" + "#" * white_slots + "-" * black_slots + "]"
    sign = "+" if eval_cp > 0 else ""
    return f"{bar} {sign}{eval_cp / 100.0:.2f} pawns"


def main():
    parser = argparse.ArgumentParser(description="Play against ApexChess in terminal")
    parser.add_argument("--weights", type=str, default="weights/apex_cloud_v1_quant.npz", help="Path to trained NNUE weights")
    parser.add_argument("--color", type=str, default="white", choices=["white", "black"], help="Your playing color")
    parser.add_argument("--depth", type=int, default=3, help="Engine search depth (default 3)")
    args = parser.parse_args()

    user_is_white = args.color.lower() == "white"

    print("=" * 65)
    print("       APEX CHESS - INTERACTIVE HUMAN VS. AI ARENA")
    print("=" * 65)

    # Initialize Engine
    if os.path.exists(args.weights):
        print(f"[ENGINE] Loading Neural Weights: {args.weights}")
        weights = NNUEWeights.from_file(args.weights)
        evaluator = ApexEvaluator(weights=weights, use_nnue=True)
    else:
        print("[ENGINE] Warning: Weights file not found. Falling back to classical evaluator.")
        evaluator = ApexEvaluator(use_nnue=False)

    searcher = SearchEngine(evaluator=evaluator)
    board = chess.Board()

    print(f"[MATCH] You: {'White' if user_is_white else 'Black'}  |  ApexChess: {'Black' if user_is_white else 'White'} (Depth {args.depth})")
    print("[RULES] Enter moves in UCI or SAN format (e.g., 'e2e4', 'e4', 'Nf3', 'O-O').")
    print("[RULES] Type 'quit', 'resign', or 'eval' at any prompt.")
    print("=" * 65)

    last_move = None

    while not board.is_game_over():
        print("\n" + render_board(board, user_is_white, last_move))
        current_eval = evaluator.evaluate(board)
        # Perspective of white
        white_eval = current_eval if board.turn == chess.WHITE else -current_eval
        print(f"Eval: {render_eval_bar(white_eval)}  (Turn: {'White' if board.turn == chess.WHITE else 'Black'})")

        if (board.turn == chess.WHITE and user_is_white) or (board.turn == chess.BLACK and not user_is_white):
            # Human move
            while True:
                try:
                    move_input = input("\nYour move > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGame aborted by user.")
                    return

                if move_input.lower() in ["quit", "exit"]:
                    print("Exiting match.")
                    return
                elif move_input.lower() in ["resign"]:
                    print("You resigned. ApexChess wins!")
                    return
                elif move_input.lower() in ["eval"]:
                    print(f"Current Raw Centipawn Score (Side to Move): {evaluator.evaluate(board)} cp")
                    continue

                # Parse move as SAN or UCI
                move = None
                try:
                    move = board.parse_san(move_input)
                except Exception:
                    try:
                        move = chess.Move.from_uci(move_input)
                        if move not in board.legal_moves:
                            move = None
                    except Exception:
                        move = None

                if move and move in board.legal_moves:
                    last_move = move
                    board.push(move)
                    break
                else:
                    print(f"Illegal move: '{move_input}'. Try again (e.g. 'e4', 'Nf3', 'e2e4').")
        else:
            # Engine move
            print(f"\nApexChess is calculating (depth {args.depth})...", end="", flush=True)
            t0 = time.time()
            best_move, score = searcher.search(board, depth=args.depth)
            dt = time.time() - t0
            nps = searcher.nodes_evaluated / max(0.001, dt)

            if not best_move:
                legal = list(board.legal_moves)
                best_move = legal[0] if legal else None

            if best_move:
                san_move = board.san(best_move)
                last_move = best_move
                board.push(best_move)
                print(f" done! [{dt:.2f}s | {searcher.nodes_evaluated} nodes | {nps:.0f} NPS]")
                print(f"Apex plays: {san_move} ({best_move.uci()})  [Score: {score:+} cp]")
            else:
                break

    # Game Over
    print("\n" + "=" * 65)
    print("                         GAME OVER")
    print("=" * 65)
    print(render_board(board, user_is_white, last_move))
    outcome = board.outcome()
    if outcome:
        print(f"Result: {board.result()} ({outcome.termination.name})")
        if outcome.winner is not None:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            is_user_winner = (outcome.winner == chess.WHITE and user_is_white) or (outcome.winner == chess.BLACK and not user_is_white)
            print(f"Winner: {winner} ({'You won!' if is_user_winner else 'ApexChess won!'})")
        else:
            print("Match ended in a Draw!")


if __name__ == "__main__":
    main()

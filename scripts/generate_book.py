#!/usr/bin/env python3
"""
ApexChess Polyglot Opening Book Generator
Creates a high-quality Grandmaster Polyglot opening book (apex_book.bin)
covering core GM repertoires across e4, d4, c4, and Nf3 openings.
"""

import os
import struct
import chess
import chess.polyglot
from collections import defaultdict
from typing import Dict, List, Tuple

def encode_polyglot_move(board: chess.Board, move: chess.Move) -> int:
    m960 = board._to_chess960(move)
    to_sq = m960.to_square
    from_sq = m960.from_square
    promo = 0
    if move.promotion == chess.KNIGHT: promo = 1
    elif move.promotion == chess.BISHOP: promo = 2
    elif move.promotion == chess.ROOK: promo = 3
    elif move.promotion == chess.QUEEN: promo = 4
    return (to_sq & 0x3F) | ((from_sq & 0x3F) << 6) | ((promo & 0x07) << 12)

# Grandmaster Repertoire Lines (UCI sequences)
GM_LINES = [
    # --- 1. e4 Open Games (1... e5) ---
    # Ruy Lopez (Berlin)
    ("e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4 d2d4 e4d6 b5c6 d7c6 d4e5 d6f5 d1d8 e8d8", 100),
    # Ruy Lopez (Closed / Marshall)
    ("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6a5 b3c2 c7c5 d2d4 d8c7", 100),
    ("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 e8g8 c2c3 d7d5 e4d5 f6d5 f3e5 c6e5 e1e5 c7c6", 90),
    # Italian Game (Giuoco Piano / Evans)
    ("e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6 e1g1 a7a6 a2a4 c5a7 f1e1 e8g8", 100),
    ("e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4 c5b4 c2c3 b4a5 d2d4 e5d4 e1g1 g8e7 c3d4 d7d5 e4d5 e7d5", 70),
    # Italian Two Knights
    ("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6 e1g1 e8g8 b1d2 a7a5 f1e1 c8e6 c4b5", 85),
    # Scotch Game
    ("e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5 c1e3 d8f6 c2c3 g8e7 f1c4 e8g8 e1g1 c6e5 c4e2", 80),
    # Vienna Game
    ("e2e4 e7e5 b1c3 g8f6 f2f4 d7d5 f4e5 f6e4 d2d3 e4c3 b2c3 d5d4 g1f3 b8c6 f1e2 f8c5", 60),

    # --- 1. e4 Sicilian Defense (1... c5) ---
    # Sicilian Najdorf (6. Bg5, 6. Be3, 6. Be2, 6. h3)
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1g5 e7e6 f2f4 f8e7 d1f3 d8c7 e1c1 b8d7", 100),
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 f8e7 d1d2 e8g8 e1c1 b8d7", 100),
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 f8e7 e1g1 e8g8 c1e3 c8e6", 90),
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 h2h3 e7e5 d4e2 h7h5 g2g3 c8e6 f1g2", 80),
    # Sicilian Dragon (Yugoslav Attack)
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 b8c6 d1d2 e8g8 f1c4 c8d7 e1c1 a8c8 c4b3 c6e5", 95),
    # Sicilian Classical / Richter-Rauzer
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 b8c6 c1g5 e7e6 d1d2 a7a6 e1c1 c8d7 f2f4 b7b5", 85),
    # Sicilian Scheveningen / Keres Attack
    ("e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e6 g2g4 h7h6 h2h4 b8c6 h1g1 d6d5", 80),
    # Sicilian Alapin (2. c3)
    ("e2e4 c7c5 c2c3 d7d5 e4d5 d8d5 d2d4 g8f6 g1f3 e7e6 f1e2 b8c6 e1g1 f8e7 c1e3 c5d4 c3d4 e8g8", 85),
    # Sicilian Rossolimo & Moscow (3. Bb5)
    ("e2e4 c7c5 g1f3 b8c6 f1b5 g7g6 b5c6 d7c6 d2d3 f8g7 h2h3 g8f6 b1c3 e8g8 c1e3 b7b6", 85),
    ("e2e4 c7c5 g1f3 d7d6 f1b5 c8d7 b5d7 d8d7 e1g1 g8f6 f1e1 b8c6 c2c3 e7e6 d2d4 c5d4 c3d4 d6d5", 85),
    # Sicilian Kan / Taimanov
    ("e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 f1d3 f8c5 d4b3 c5a7 d1e2 b8c6 c1e3 d7d6", 75),
    ("e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 b1c3 d8c7 c1e3 a7a6 d1d2 g8f6 e1c1 f8e7 f2f3 b7b5", 80),

    # --- 1. e4 French Defense (1... e6) ---
    # French Winawer
    ("e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3 g8e7 d1g4 e8g8 f1d3 b8c6 g4h5 e7g6", 90),
    # French Classical & Steinitz
    ("e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 e4e5 f6d7 f2f4 c7c5 g1f3 b8c6 c1e3 a7a6 d1d2 b7b5 a2a3", 85),
    # French Tarrasch (3. Nd2)
    ("e2e4 e7e6 d2d4 d7d5 b1d2 c7c5 e4d5 d8d5 g1f3 c5d4 f1c4 d5d6 e1g1 g8f6 d2b3 b8c6 b3d4 c6d4 f3d4", 80),
    # French Advance
    ("e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 c8d7 f1e2 g8e7 b1a3 c5d4 c3d4 e7f5 a3c2", 80),

    # --- 1. e4 Caro-Kann Defense (1... c6) ---
    # Caro-Kann Classical (4... Bf5)
    ("e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6 g1f3 b8d7 h4h5 g6h7 f1d3 h7d3 d1d3 e7e6", 95),
    # Caro-Kann Advance (3... Bf5 & 3... c5)
    ("e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2 c6c5 c1e3 b8c6 e1g1 g8e7 c2c4 d5c4 b1a3", 90),
    # Caro-Kann Tartakower / Korchnoi
    ("e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 g8f6 e4f6 e7f6 c2c3 f8d6 f1d3 e8g8 d1c2 f8e8 g1e2", 80),

    # --- 1. e4 Scandinavian & Pirc ---
    ("e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6 g1f3 c7c6 f1c4 c8f5 c1d2 e7e6 d1e2 f8b4 e1c1", 80),
    ("e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4 f8g7 g1f3 e8g8 c1e3 b7b6 e4e5 f6g4 e3g1 c7c5", 75),

    # --- 1. d4 Queen's Gambit & Indian Defenses ---
    # QGD Tartakower / Orthodox
    ("d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 h7h6 g5h4 e8g8 g1f3 b7b6 c4d5 f6d5 h4e7 d8e7 c3d5 e6d5", 95),
    ("d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 g1f3 f8e7 c1f4 e8g8 e2e3 c7c5 d4c5 e7c5 d1c2 b8c6 a2a3 d8a5", 90),
    # Slav Defense (4... dxc4 & Semi-Slav)
    ("d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5 e2e3 e7e6 f1c4 f8b4 e1g1 e8g8 d1e2 b8d7 e3e4 f5g6", 95),
    ("d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 e2e3 b8d7 f1d3 d5c4 d3c4 b7b5 c4d3 c8b7 e1g1 a7a6 e3e4 c6c5", 95),
    # Catalan Opening
    ("d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 g1f3 e8g8 e1g1 d5c4 d1c2 a7a6 a2a4 c8d7 c2c4 d7c6 c1g5", 90),
    # King's Indian Defense (Mar del Plata & Classical)
    ("d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5 c6e7 f3e1 f6d7 c1e3 f7f5 f2f3 f5f4 e3f2 g6g5", 95),
    # Grünfeld Defense (Exchange)
    ("d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7 g1f3 c7c5 a1b1 e8g8 f1e2 b8c6 d4d5 c6e5 f3e5 g7e5", 90),
    # Nimzo-Indian Defense (Rubinstein & Classical)
    ("d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5 g1f3 c7c5 e1g1 d5c4 d3c4 b8d7 a2a3 c5d4 e3d4 b4c3 b2c3 d8c7", 90),
    ("d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8 a2a3 b4c3 c2c3 b7b6 c1g5 c8b7 f2f3 d7d5 e2e3 b8d7 c4d5 e6d5", 90),
    # Queen's Indian Defense
    ("d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8a6 b2b3 f8b4 c1d2 b4e7 f1g2 c7c6 d2c3 d7d5 f3e5 f6d7 e5d7 b8d7", 80),
    # London System
    ("d2d4 d7d5 c1f4 g8f6 e2e3 c7c5 g1f3 b8c6 c2c3 d8b6 d1b3 c5c4 b3c2 c8f5 c2c1 e7e6 b1d2", 85),

    # --- 1. c4 English Opening ---
    ("c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 f8b4 f1g2 e8g8 e1g1 e5e4 f3g5 b4c3 b2c3 f8e8 f2f3 e4f3 g5f3 d7d5", 85),
    ("c2c4 c7c5 g1f3 g8f6 b1c3 b8c6 d2d4 c5d4 f3d4 e7e6 g2g3 d8b6 d4b3 c6e5 e2e4 f8b4 d1e2 d7d6 f2f4 e5c6", 85),

    # --- 1. Nf3 Réti Opening ---
    ("g1f3 d7d5 g2g3 g8f6 f1g2 c7c6 e1g1 c8g4 d2d3 b8d7 b1d2 e7e5 e2e4 f8d6 h2h3 g4h5 d1e1 e8g8 f3h4 f8e8", 85),
    ("g1f3 d7d5 g2g3 c7c6 f1g2 c8g4 e1g1 b8d7 d2d4 e7e6 b1d2 f8e7 c2c4 g8f6 b2b3 e8g8 c1b2", 85),

    # --- Extended Classical Repertoires ---
    # Ruy Lopez Breyer & Chigorin
    ("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6b8 d2d4 b8d7 b1d2 c8b7", 90),
    ("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6a5 b3c2 c7c5 d2d4 f6d7", 90),
    # Italian Game Giuoco Piano Evans Extended
    ("e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4 e5d4 c3d4 c5b4 c1d2 b4d2 b1d2 d7d5 e4d5 f6d5 d1b3 c6e7", 85),
    # Sicilian Accelerated Dragon & Maroczy
    ("e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6 c2c4 f8g7 c1e3 g8f6 b1c3 e8g8 f1e2 d7d6 e1g1 c8d7 d1d2 c6d4 e3d4 d7c6", 85),
    # French Burn & Advance Extended
    ("e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 d5e4 c3e4 f8e7 g5f6 e7f6 g1f3 e8g8 d1d2 b8d7 e1c1 f6e7", 85),
    ("e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 d8b6 a2a3 g8h6 b2b4 c5d4 c3d4 h6f5 c1b2 c8d7", 85),
    # Caro-Kann Advance Short System
    ("e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2 c6c5 c1e3 d8b6 b1c3 b8c6 e1g1 b6b2 d1e1 c5d4 e3d4 c6d4 f3d4 f8b4", 85),
    ("e2e4 c7c6 d2d4 d7d5 b1d2 d5e4 d2e4 b8d7 e4g5 g8f6 f1d3 e7e6 g1f3 f8d6 d1e2 h7h6 g5e4 f6e4 e2e4", 85),
    # Cambridge Springs QGD
    ("d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 b8d7 e2e3 c7c6 g1f3 d8a5 f3d2 f8b4 d1c2 e8g8 f1e2 d5c4 g5f6 d7f6 d2c4 a5c7", 85),
    # Semi-Slav Botvinnik & Meran
    ("d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 c1g5 d5c4 e2e4 b7b5 e4e5 h7h6 g5h4 g7g5 f3g5 h6g5 h4g5 b8d7", 85),
    # King's Indian Sämisch
    ("d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3 e8g8 c1e3 e7e5 d4d5 c7c6 d1d2 c6d5 c4d5 a7a6 e1c1 b8d7", 85),
    # Grünfeld Russian
    ("d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 g1f3 f8g7 d1b3 d5c4 b3c4 e8g8 e2e4 c8g4 c1e3 f6d7 c4b3 b8c6", 85),
    # Catalan Closed
    ("d2d4 d7d5 c2c4 e7e6 g1f3 g8f6 g2g3 f8e7 f1g2 e8g8 e1g1 d5c4 d1c2 a7a6 a2a4 c8d7 c2c4 d7c6 c1g5 c6d5 c4c2 d5e4", 85),
    # Dutch Defense Leningrad
    ("d2d4 f7f5 g2g3 g8f6 f1g2 g7g6 g1f3 f8g7 e1g1 e8g8 c2c4 d7d6 b1c3 c7c6 d4d5 e7e5 d5e6 c8e6", 85),
    # London System Extended
    ("d2d4 g8f6 c1f4 d7d5 e2e3 c7c5 c2c3 b8c6 b1d2 e7e6 g1f3 f8d6 f4g3 e8g8 f1d3 b7b6 f3e5 c8b7 f2f4 c6e7", 85),
]

def build_polyglot_book(output_path: str):
    print(f"[BOOK] Compiling Grandmaster Opening Book to {output_path}...")
    book_entries: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    total_positions = 0
    total_lines = len(GM_LINES)

    for line_str, base_weight in GM_LINES:
        board = chess.Board()
        moves = line_str.split()
        for i, move_str in enumerate(moves):
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    break
                poly_key = chess.polyglot.zobrist_hash(board)
                poly_move = encode_polyglot_move(board, move)

                # Decay weight gently with move depth
                weight = max(5, int(base_weight * (0.95 ** i)))
                book_entries[poly_key][poly_move] = max(book_entries[poly_key][poly_move], weight)
                board.push(move)
                total_positions += 1
            except Exception as e:
                print(f"Error on move {move_str}: {e}")
                break

    # Flatten and sort entries by key (Polyglot binary search requirement)
    flattened_entries: List[Tuple[int, int, int]] = []
    for key, move_dict in book_entries.items():
        for raw_move, weight in move_dict.items():
            flattened_entries.append((key, raw_move, weight))

    # Sort strictly by key ascending
    flattened_entries.sort(key=lambda x: x[0])

    # Write binary file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        for key, raw_move, weight in flattened_entries:
            # Polyglot format: >QHHI (uint64 key, uint16 move, uint16 weight, uint32 learn)
            f.write(struct.pack(">QHHI", key, raw_move, weight, 0))

    file_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"[OK] Successfully built {output_path}:")
    print(f"     Unique Keys: {len(book_entries):,}")
    print(f"     Total Book Moves: {len(flattened_entries):,}")
    print(f"     File Size: {file_size_kb:.1f} KB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Polyglot Opening Book")
    parser.add_argument("--output", type=str, default="weights/apex_book.bin")
    args = parser.parse_args()
    build_polyglot_book(args.output)

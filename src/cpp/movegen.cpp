#include "movegen.h"
#include <chrono>
#include <iomanip>

namespace Apex {

static void generate_pseudolegal_moves(const Position& pos, MoveList& moves, bool captures_only) {
    Color us = pos.turn();
    Color them = ~us;
    Bitboard our_pieces = pos.pieces(us);
    Bitboard their_pieces = pos.pieces(them);
    Bitboard occ = pos.occupied();
    Bitboard empty = ~occ;
    Bitboard target_mask = captures_only ? their_pieces : ~our_pieces;

    // 1. Pawns
    Bitboard pawns = pos.pieces(us, PAWN);
    while (pawns) {
        Square sq = pop_lsb(pawns);
        int f = square_file(sq);
        int r = square_rank(sq);

        if (us == WHITE) {
            // Single push
            Square up = make_square(f, r + 1);
            if (!captures_only && (square_bb(up) & empty)) {
                if (r == 6) { // Promotion
                    moves.add(make_move(sq, up, FLAG_PROMOTION, QUEEN));
                    moves.add(make_move(sq, up, FLAG_PROMOTION, ROOK));
                    moves.add(make_move(sq, up, FLAG_PROMOTION, BISHOP));
                    moves.add(make_move(sq, up, FLAG_PROMOTION, KNIGHT));
                } else {
                    moves.add(make_move(sq, up, FLAG_NORMAL));
                    // Double push
                    Square up2 = make_square(f, r + 2);
                    if (r == 1 && (square_bb(up2) & empty)) {
                        moves.add(make_move(sq, up2, FLAG_NORMAL));
                    }
                }
            }

            // Captures
            Bitboard atks = pawn_attacks(WHITE, sq) & their_pieces;
            while (atks) {
                Square to_sq = pop_lsb(atks);
                if (r == 6) {
                    moves.add(make_move(sq, to_sq, FLAG_PROMOTION, QUEEN));
                    moves.add(make_move(sq, to_sq, FLAG_PROMOTION, ROOK));
                    moves.add(make_move(sq, to_sq, FLAG_PROMOTION, BISHOP));
                    moves.add(make_move(sq, to_sq, FLAG_PROMOTION, KNIGHT));
                } else {
                    moves.add(make_move(sq, to_sq, FLAG_NORMAL));
                }
            }

            // En Passant
            if (pos.en_passant() != SQ_NONE) {
                if (pawn_attacks(WHITE, sq) & square_bb(pos.en_passant())) {
                    moves.add(make_move(sq, pos.en_passant(), FLAG_EN_PASSANT));
                }
            }
        } else {
            // Black pawns
            // Single push
            Square down = make_square(f, r - 1);
            if (!captures_only && (square_bb(down) & empty)) {
                if (r == 1) { // Promotion
                    moves.add(make_move(sq, down, FLAG_PROMOTION, QUEEN));
                    moves.add(make_move(sq, down, FLAG_PROMOTION, ROOK));
                    moves.add(make_move(sq, down, FLAG_PROMOTION, BISHOP));
                    moves.add(make_move(sq, down, FLAG_PROMOTION, KNIGHT));
                } else {
                    moves.add(make_move(sq, down, FLAG_NORMAL));
                    // Double push
                    Square down2 = make_square(f, r - 2);
                    if (r == 6 && (square_bb(down2) & empty)) {
                        moves.add(make_move(sq, down2, FLAG_NORMAL));
                    }
                }
            }

            // Captures
            Bitboard atks = pawn_attacks(BLACK, sq) & their_pieces;
            while (atks) {
                Square to_sq = pop_lsb(atks);
                if (r == 1) {
                    moves.add(make_move(sq, to_sq, FLAG_PROMOTION, QUEEN));
                    moves.add(make_move(sq, to_sq, FLAG_PROMOTION, ROOK));
                    moves.add(make_move(sq, to_sq, FLAG_PROMOTION, BISHOP));
                    moves.add(make_move(sq, to_sq, FLAG_PROMOTION, KNIGHT));
                } else {
                    moves.add(make_move(sq, to_sq, FLAG_NORMAL));
                }
            }

            // En Passant
            if (pos.en_passant() != SQ_NONE) {
                if (pawn_attacks(BLACK, sq) & square_bb(pos.en_passant())) {
                    moves.add(make_move(sq, pos.en_passant(), FLAG_EN_PASSANT));
                }
            }
        }
    }

    // 2. Knights
    Bitboard knights = pos.pieces(us, KNIGHT);
    while (knights) {
        Square sq = pop_lsb(knights);
        Bitboard targets = knight_attacks(sq) & target_mask;
        while (targets) {
            moves.add(make_move(sq, pop_lsb(targets), FLAG_NORMAL));
        }
    }

    // 3. Bishops
    Bitboard bishops = pos.pieces(us, BISHOP);
    while (bishops) {
        Square sq = pop_lsb(bishops);
        Bitboard targets = bishop_attacks(sq, occ) & target_mask;
        while (targets) {
            moves.add(make_move(sq, pop_lsb(targets), FLAG_NORMAL));
        }
    }

    // 4. Rooks
    Bitboard rooks = pos.pieces(us, ROOK);
    while (rooks) {
        Square sq = pop_lsb(rooks);
        Bitboard targets = rook_attacks(sq, occ) & target_mask;
        while (targets) {
            moves.add(make_move(sq, pop_lsb(targets), FLAG_NORMAL));
        }
    }

    // 5. Queens
    Bitboard queens = pos.pieces(us, QUEEN);
    while (queens) {
        Square sq = pop_lsb(queens);
        Bitboard targets = queen_attacks(sq, occ) & target_mask;
        while (targets) {
            moves.add(make_move(sq, pop_lsb(targets), FLAG_NORMAL));
        }
    }

    // 6. King moves
    Square ksq = pos.king_square(us);
    Bitboard king_targets = king_attacks(ksq) & target_mask;
    while (king_targets) {
        moves.add(make_move(ksq, pop_lsb(king_targets), FLAG_NORMAL));
    }

    // 7. Castling (not in captures_only)
    if (!captures_only && !pos.in_check()) {
        if (us == WHITE) {
            // White O-O
            if (pos.castling() & WHITE_OO) {
                if (!(occ & (SquareBB[SQ_F1] | SquareBB[SQ_G1]))) {
                    if (!pos.is_square_attacked(SQ_F1, BLACK) && !pos.is_square_attacked(SQ_G1, BLACK)) {
                        moves.add(make_move(SQ_E1, SQ_G1, FLAG_CASTLING));
                    }
                }
            }
            // White O-O-O
            if (pos.castling() & WHITE_OOO) {
                if (!(occ & (SquareBB[SQ_D1] | SquareBB[SQ_C1] | SquareBB[SQ_B1]))) {
                    if (!pos.is_square_attacked(SQ_D1, BLACK) && !pos.is_square_attacked(SQ_C1, BLACK)) {
                        moves.add(make_move(SQ_E1, SQ_C1, FLAG_CASTLING));
                    }
                }
            }
        } else {
            // Black O-O
            if (pos.castling() & BLACK_OO) {
                if (!(occ & (SquareBB[SQ_F8] | SquareBB[SQ_G8]))) {
                    if (!pos.is_square_attacked(SQ_F8, WHITE) && !pos.is_square_attacked(SQ_G8, WHITE)) {
                        moves.add(make_move(SQ_E8, SQ_G8, FLAG_CASTLING));
                    }
                }
            }
            // Black O-O-O
            if (pos.castling() & BLACK_OOO) {
                if (!(occ & (SquareBB[SQ_D8] | SquareBB[SQ_C8] | SquareBB[SQ_B8]))) {
                    if (!pos.is_square_attacked(SQ_D8, WHITE) && !pos.is_square_attacked(SQ_C8, WHITE)) {
                        moves.add(make_move(SQ_E8, SQ_C8, FLAG_CASTLING));
                    }
                }
            }
        }
    }
}

void generate_legal_moves(Position& pos, MoveList& legal_moves) {
    MoveList pseudo;
    generate_pseudolegal_moves(pos, pseudo, false);

    legal_moves.count = 0;
    StateInfo state;
    for (int i = 0; i < pseudo.count; ++i) {
        Move m = pseudo[i];
        pos.make_move(m, state);
        // After make_move, side_to_move has toggled, so we check if the player who just moved is in check
        Color mover = ~pos.turn();
        Square ksq = pos.king_square(mover);
        if (!pos.is_square_attacked(ksq, pos.turn())) {
            legal_moves.add(m);
        }
        pos.unmake_move(m, state);
    }
}

void generate_captures_only(Position& pos, MoveList& capture_moves) {
    MoveList pseudo;
    generate_pseudolegal_moves(pos, pseudo, true);

    capture_moves.count = 0;
    StateInfo state;
    for (int i = 0; i < pseudo.count; ++i) {
        Move m = pseudo[i];
        pos.make_move(m, state);
        Color mover = ~pos.turn();
        Square ksq = pos.king_square(mover);
        if (!pos.is_square_attacked(ksq, pos.turn())) {
            capture_moves.add(m);
        }
        pos.unmake_move(m, state);
    }
}

uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1ULL;

    MoveList legal_moves;
    generate_legal_moves(pos, legal_moves);

    if (depth == 1) return legal_moves.count;

    uint64_t nodes = 0;
    StateInfo state;

    for (int i = 0; i < legal_moves.count; ++i) {
        Move m = legal_moves[i];
        pos.make_move(m, state);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(m, state);
    }

    return nodes;
}

void run_perft_suite(Position& pos, int depth) {
    std::cout << "\n======================================================\n";
    std::cout << "  APEX CHESS HIGH-SPEED C++ PERFT BENCHMARK (Depth " << depth << ")\n";
    std::cout << "======================================================\n";

    MoveList moves;
    generate_legal_moves(pos, moves);

    uint64_t total_nodes = 0;
    StateInfo state;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < moves.count; ++i) {
        Move m = moves[i];
        pos.make_move(m, state);
        uint64_t branch_nodes = (depth > 1) ? perft(pos, depth - 1) : 1;
        pos.unmake_move(m, state);

        total_nodes += branch_nodes;
        std::cout << "  " << move_to_uci(m) << " : " << branch_nodes << "\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double nps = total_nodes / std::max(0.0001, elapsed_sec);

    std::cout << "------------------------------------------------------\n";
    std::cout << "  Total Nodes Evaluated: " << total_nodes << "\n";
    std::cout << "  Time Elapsed:          " << std::fixed << std::setprecision(3) << elapsed_sec << " seconds\n";
    std::cout << "  Node Throughput:       " << std::fixed << std::setprecision(0) << nps << " NPS ("
              << (nps / 1e6) << " MNPS)\n";
    std::cout << "======================================================\n\n";
}

} // namespace Apex

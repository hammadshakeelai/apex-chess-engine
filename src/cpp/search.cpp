#include "search.h"
#include "evaluate.h"
#include "book.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>

namespace Apex {

// MVV-LVA victim values
static const int PieceVictimValue[6] = { 100, 300, 310, 500, 900, 10000 };

TranspositionTable::TranspositionTable(size_t size_mb) {
    resize(size_mb);
}

TranspositionTable::~TranspositionTable() {
    delete[] table;
}

void TranspositionTable::resize(size_t size_mb) {
    delete[] table;
    size_t bytes = size_mb * 1024 * 1024;
    count = bytes / sizeof(TTEntry);
    table = new TTEntry[count]();
}

void TranspositionTable::clear() {
    if (table) {
        std::memset(table, 0, count * sizeof(TTEntry));
    }
}

bool TranspositionTable::probe(uint64_t key, int depth, int alpha, int beta, int ply, int& out_score, Move& out_move) {
    size_t idx = key % count;
    const TTEntry& entry = table[idx];

    if (entry.key == key) {
        out_move = entry.move;
        if (entry.depth >= depth) {
            int score = entry.score;
            // Adjust mate score for ply
            if (score > MATE_SCORE - 100) score -= ply;
            else if (score < -MATE_SCORE + 100) score += ply;

            if (entry.flag == TT_EXACT) {
                out_score = score;
                return true;
            }
            if (entry.flag == TT_LOWERBOUND && score >= beta) {
                out_score = score;
                return true;
            }
            if (entry.flag == TT_UPPERBOUND && score <= alpha) {
                out_score = score;
                return true;
            }
        }
    }
    return false;
}

void TranspositionTable::store(uint64_t key, int depth, int score, TTFlag flag, Move best_move, int ply) {
    size_t idx = key % count;
    TTEntry& entry = table[idx];

    // Normalize mate score to root ply
    if (score > MATE_SCORE - 100) score += ply;
    else if (score < -MATE_SCORE + 100) score -= ply;

    // Always overwrite if deeper or new key
    if (entry.key != key || depth >= entry.depth) {
        entry.key = key;
        entry.move = best_move;
        entry.score = static_cast<int16_t>(score);
        entry.depth = static_cast<int8_t>(depth);
        entry.flag = flag;
    }
}

Searcher::Searcher() : tt(32), nodes_evaluated(0), stop_requested(false), allocated_time_ms(0) {
    std::memset(killer_moves, 0, sizeof(killer_moves));
    std::memset(history_table, 0, sizeof(history_table));

    for (int d = 0; d < 64; ++d) {
        for (int m = 0; m < 64; ++m) {
            if (d == 0 || m == 0) {
                lmr_table[d][m] = 0;
            } else {
                lmr_table[d][m] = 0.75 + std::log(d) * std::log(m) / 2.25;
            }
        }
    }
}

void Searcher::stop() {
    stop_requested = true;
}

void Searcher::score_moves(const Position& pos, MoveList& moves, Move tt_move, int ply) {
    Color us = pos.turn();
    for (int i = 0; i < moves.count; ++i) {
        Move m = moves[i];
        int score = 0;

        if (m == tt_move) {
            score = 10000000;
        } else {
            Square from = move_from(m);
            Square to = move_to(m);
            Piece cap = pos.get_piece_at(to);

            if (cap != NO_PIECE) {
                PieceType attacker = piece_type(pos.get_piece_at(from));
                PieceType victim = piece_type(cap);
                score = 1000000 + PieceVictimValue[victim] * 10 - PieceVictimValue[attacker];
            } else if (move_flag(m) == FLAG_PROMOTION) {
                score = 900000;
            } else if (ply < MAX_PLY && m == killer_moves[ply][0]) {
                score = 800000;
            } else if (ply < MAX_PLY && m == killer_moves[ply][1]) {
                score = 700000;
            } else {
                score = history_table[us][from][to];
            }
        }

        // Encode score into upper bits or use parallel array
        // For simplicity: store parallel scores via a simple in-place bubble
    }
}

int Searcher::quiescence(Position& pos, int alpha, int beta, int ply) {
    nodes_evaluated++;

    // Check time limit every 2048 nodes
    if ((nodes_evaluated & 2047) == 0 && allocated_time_ms > 0) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
        if (elapsed_ms >= allocated_time_ms) {
            stop_requested = true;
        }
    }

    if (stop_requested) return 0;
    if (ply >= MAX_PLY - 1) return evaluate(pos);

    int stand_pat = evaluate(pos);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    MoveList captures;
    generate_captures_only(pos, captures);

    // Score and order captures MVV-LVA
    int scores[256];
    for (int i = 0; i < captures.count; ++i) {
        Square from = move_from(captures[i]);
        Square to = move_to(captures[i]);
        Piece cap = pos.get_piece_at(to);
        int victim_val = (cap != NO_PIECE) ? PieceVictimValue[piece_type(cap)] : 100;
        int attacker_val = PieceVictimValue[piece_type(pos.get_piece_at(from))];
        scores[i] = victim_val * 10 - attacker_val;
    }

    StateInfo state;
    for (int i = 0; i < captures.count; ++i) {
        // Pick best
        int best_idx = i;
        for (int j = i + 1; j < captures.count; ++j) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        std::swap(captures.moves[i], captures.moves[best_idx]);
        std::swap(scores[i], scores[best_idx]);

        Move m = captures[i];
        pos.make_move(m, state);
        int score = -quiescence(pos, -beta, -alpha, ply + 1);
        pos.unmake_move(m, state);

        if (stop_requested) return 0;

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

int Searcher::pvs(Position& pos, int depth, int alpha, int beta, int ply, bool is_pv) {
    nodes_evaluated++;

    // Check time limit
    if ((nodes_evaluated & 2047) == 0 && allocated_time_ms > 0) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
        if (elapsed_ms >= allocated_time_ms) {
            stop_requested = true;
        }
    }

    if (stop_requested) return 0;
    if (ply >= MAX_PLY - 1) return evaluate(pos);

    // 1. Repetition and 50-move rule detection
    if (ply > 0 && pos.is_draw(ply)) {
        return 0;
    }

    bool in_check = pos.in_check();
    if (in_check) depth++; // Check extension

    if (depth <= 0) {
        return quiescence(pos, alpha, beta, ply);
    }

    int orig_alpha = alpha;
    Move tt_move = MOVE_NONE;
    int tt_score = 0;

    if (tt.probe(pos.hash(), depth, alpha, beta, ply, tt_score, tt_move)) {
        if (!is_pv) return tt_score;
    }

    Color us = pos.turn();

    // Null Move Pruning (NMP) for non-PV nodes
    Bitboard non_pawns = pos.pieces(us, KNIGHT) | pos.pieces(us, BISHOP) | pos.pieces(us, ROOK) | pos.pieces(us, QUEEN);
    if (!is_pv && !in_check && depth >= 3 && non_pawns) {
        int static_eval = evaluate(pos);
        if (static_eval >= beta) {
            StateInfo null_state;
            pos.make_null_move(null_state);
            int R = 2 + depth / 6;
            int null_score = -pvs(pos, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
            pos.unmake_null_move(null_state);

            if (stop_requested) return 0;
            if (null_score >= beta) {
                return (null_score >= MATE_SCORE - 100) ? beta : null_score;
            }
        }
    }

    MoveList moves;
    generate_legal_moves(pos, moves);

    if (moves.count == 0) {
        if (in_check) return -MATE_SCORE + ply; // Checkmate
        return 0; // Stalemate
    }

    // Score moves
    int scores[256];
    for (int i = 0; i < moves.count; ++i) {
        Move m = moves[i];
        if (m == tt_move) {
            scores[i] = 10000000;
        } else {
            Square from = move_from(m);
            Square to = move_to(m);
            Piece cap = pos.get_piece_at(to);

            if (cap != NO_PIECE) {
                PieceType attacker = piece_type(pos.get_piece_at(from));
                PieceType victim = piece_type(cap);
                scores[i] = 1000000 + PieceVictimValue[victim] * 10 - PieceVictimValue[attacker];
            } else if (move_flag(m) == FLAG_PROMOTION) {
                scores[i] = 900000;
            } else if (ply < MAX_PLY && m == killer_moves[ply][0]) {
                scores[i] = 800000;
            } else if (ply < MAX_PLY && m == killer_moves[ply][1]) {
                scores[i] = 700000;
            } else {
                scores[i] = history_table[us][from][to];
            }
        }
    }

    Move best_move = MOVE_NONE;
    int best_score = -INFINITY_SCORE;
    StateInfo state;

    for (int i = 0; i < moves.count; ++i) {
        // Selection sort
        int best_idx = i;
        for (int j = i + 1; j < moves.count; ++j) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        std::swap(moves.moves[i], moves.moves[best_idx]);
        std::swap(scores[i], scores[best_idx]);

        Move m = moves[i];
        pos.make_move(m, state);

        int score = 0;
        if (i == 0) {
            // PV move: full window search
            score = -pvs(pos, depth - 1, -beta, -alpha, ply + 1, is_pv);
        } else {
            // Late Move Reductions (LMR) for quiet moves
            int reduction = 0;
            if (depth >= 3 && i >= 4 && pos.get_piece_at(move_to(m)) == NO_PIECE && !in_check) {
                reduction = 1;
            }

            // Zero-window search
            score = -pvs(pos, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, false);

            // Re-search if reduced move failed high
            if (reduction > 0 && score > alpha) {
                score = -pvs(pos, depth - 1, -alpha - 1, -alpha, ply + 1, false);
            }

            // Re-search full window if scout search failed high
            if (score > alpha && score < beta) {
                score = -pvs(pos, depth - 1, -beta, -alpha, ply + 1, true);
            }
        }

        pos.unmake_move(m, state);

        if (stop_requested) return 0;

        if (score > best_score) {
            best_score = score;
            best_move = m;
        }

        if (score > alpha) {
            alpha = score;
            if (score >= beta) {
                // Beta cutoff
                if (pos.get_piece_at(move_to(m)) == NO_PIECE && ply < MAX_PLY) {
                    killer_moves[ply][1] = killer_moves[ply][0];
                    killer_moves[ply][0] = m;
                    history_table[us][move_from(m)][move_to(m)] += depth * depth;
                }
                break;
            }
        }
    }

    TTFlag flag = TT_EXACT;
    if (best_score <= orig_alpha) flag = TT_UPPERBOUND;
    else if (best_score >= beta) flag = TT_LOWERBOUND;

    tt.store(pos.hash(), depth, best_score, flag, best_move, ply);
    return best_score;
}

Move Searcher::search(Position& pos, const SearchLimits& limits) {
    nodes_evaluated = 0;
    stop_requested = false;
    start_time = std::chrono::high_resolution_clock::now();

    // 1. Probe Opening Book (instant 0ms Grandmaster response)
    Move book_move = GlobalBook.probe(pos);
    if (book_move != MOVE_NONE) {
        std::cout << "info string book move " << move_to_uci(book_move) << std::endl;
        std::cout << "bestmove " << move_to_uci(book_move) << std::endl;
        return book_move;
    }

    // Time allocation
    Color us = pos.turn();
    if (limits.movetime > 0) {
        allocated_time_ms = limits.movetime;
    } else if (limits.time[us] > 0) {
        allocated_time_ms = limits.time[us] / 30 + limits.inc[us] / 2;
    } else {
        allocated_time_ms = 0;
    }

    int max_depth = (limits.depth > 0) ? limits.depth : 64;
    Move best_move = MOVE_NONE;
    int best_score = 0;

    MoveList root_moves;
    generate_legal_moves(pos, root_moves);
    if (root_moves.count == 0) return MOVE_NONE;
    best_move = root_moves[0];

    for (int d = 1; d <= max_depth; ++d) {
        int score = 0;
        if (d >= 4) {
            int delta = 50;
            int alpha = std::max(-INFINITY_SCORE, best_score - delta);
            int beta = std::min(INFINITY_SCORE, best_score + delta);
            score = pvs(pos, d, alpha, beta, 0, true);

            // If score falls outside window, re-search with full window
            if (score <= alpha || score >= beta) {
                score = pvs(pos, d, -INFINITY_SCORE, INFINITY_SCORE, 0, true);
            }
        } else {
            score = pvs(pos, d, -INFINITY_SCORE, INFINITY_SCORE, 0, true);
        }

        if (stop_requested && d > 1) {
            break;
        }

        best_score = score;
        Move current_best = MOVE_NONE;
        int dummy_score = 0;
        tt.probe(pos.hash(), d, -INFINITY_SCORE, INFINITY_SCORE, 0, dummy_score, current_best);
        if (current_best != MOVE_NONE) {
            best_move = current_best;
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
        uint64_t nps = static_cast<uint64_t>(nodes_evaluated / std::max(0.001, elapsed_ms / 1000.0));

        std::cout << "info depth " << d
                  << " score cp " << best_score
                  << " nodes " << nodes_evaluated
                  << " time " << static_cast<int>(elapsed_ms)
                  << " nps " << nps
                  << " pv " << move_to_uci(best_move)
                  << std::endl;

        if (limits.movetime > 0 && elapsed_ms * 2 >= limits.movetime) {
            break;
        }
    }

    std::cout << "bestmove " << move_to_uci(best_move) << std::endl;
    return best_move;
}

} // namespace Apex

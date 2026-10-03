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

// Futility pruning margins indexed by depth
static const int FUTILITY_MARGIN[4] = { 0, 200, 300, 500 };

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
    current_generation = 0;
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

    // Generation-based replacement: prefer replacing stale entries
    // Replace if: different key, or same key with deeper search, or stale generation
    bool replace = (entry.key != key) ||
                   (depth >= entry.depth) ||
                   (entry.generation != current_generation);

    if (replace) {
        entry.key = key;
        entry.move = best_move;
        entry.score = static_cast<int16_t>(score);
        entry.depth = static_cast<int8_t>(depth);
        entry.flag = flag;
        entry.generation = current_generation;
    }
}

// ============================================================================
// Static Exchange Evaluation (SEE)
// ============================================================================

Bitboard Searcher::attackers_to(const Position& pos, Square sq, Bitboard occ) const {
    return (pawn_attacks(BLACK, sq) & pos.pieces(WHITE, PAWN))
         | (pawn_attacks(WHITE, sq) & pos.pieces(BLACK, PAWN))
         | (knight_attacks(sq)      & (pos.pieces(WHITE, KNIGHT) | pos.pieces(BLACK, KNIGHT)))
         | (bishop_attacks(sq, occ) & (pos.pieces(WHITE, BISHOP) | pos.pieces(BLACK, BISHOP)
                                     | pos.pieces(WHITE, QUEEN) | pos.pieces(BLACK, QUEEN)))
         | (rook_attacks(sq, occ)   & (pos.pieces(WHITE, ROOK) | pos.pieces(BLACK, ROOK)
                                     | pos.pieces(WHITE, QUEEN) | pos.pieces(BLACK, QUEEN)))
         | (king_attacks(sq)        & (pos.pieces(WHITE, KING) | pos.pieces(BLACK, KING)));
}

Bitboard Searcher::least_valuable_attacker(const Position& pos, Bitboard attacker_bb, Color side, PieceType& pt) const {
    for (int p = PAWN; p <= KING; ++p) {
        pt = static_cast<PieceType>(p);
        Bitboard pieces = attacker_bb & pos.pieces(side, pt);
        if (pieces) {
            return pieces & (~pieces + 1);  // isolate LSB
        }
    }
    return 0;
}

bool Searcher::see_ge(const Position& pos, Move m, int threshold) const {
    if (move_flag(m) == FLAG_CASTLING) return true;

    Square from = move_from(m);
    Square to = move_to(m);

    int swap_value = 0;

    // Initial capture value
    Piece cap = pos.get_piece_at(to);
    if (move_flag(m) == FLAG_EN_PASSANT) {
        swap_value = SEE_VALUE[PAWN] - threshold;
    } else if (cap != NO_PIECE) {
        swap_value = SEE_VALUE[piece_type(cap)] - threshold;
    } else {
        swap_value = -threshold;
    }

    // If even capturing for free doesn't meet threshold, fail
    if (swap_value < 0) return false;

    // Value of the piece making the first capture
    Piece moving = pos.get_piece_at(from);
    PieceType moving_pt = piece_type(moving);

    // If we lose the capturing piece and still meet threshold, succeed
    swap_value -= SEE_VALUE[moving_pt];
    if (swap_value >= 0) return true;

    // Swap list algorithm
    Bitboard occ = pos.occupied() ^ square_bb(from) ^ square_bb(to);
    if (move_flag(m) == FLAG_EN_PASSANT) {
        Color us = piece_color(moving);
        Square ep_cap_sq = static_cast<Square>(to + (us == WHITE ? -8 : 8));
        occ ^= square_bb(ep_cap_sq);
    }

    Bitboard attackers = attackers_to(pos, to, occ);
    Color stm = ~piece_color(moving);  // Side to move in the exchange

    while (true) {
        // Remove used pieces
        Bitboard stm_attackers = attackers & pos.pieces(stm);
        if (!stm_attackers) break;

        // Find least valuable attacker
        PieceType pt;
        Bitboard lva = least_valuable_attacker(pos, stm_attackers, stm, pt);
        if (!lva) break;

        // Remove attacker from occupancy
        occ ^= lva;

        // Discover new sliding attackers through the removed piece
        if (pt == PAWN || pt == BISHOP || pt == QUEEN) {
            attackers |= bishop_attacks(to, occ) & (pos.pieces(WHITE, BISHOP) | pos.pieces(BLACK, BISHOP)
                                                   | pos.pieces(WHITE, QUEEN)  | pos.pieces(BLACK, QUEEN));
        }
        if (pt == ROOK || pt == QUEEN) {
            attackers |= rook_attacks(to, occ) & (pos.pieces(WHITE, ROOK) | pos.pieces(BLACK, ROOK)
                                                 | pos.pieces(WHITE, QUEEN) | pos.pieces(BLACK, QUEEN));
        }

        attackers &= occ;  // Remove pieces no longer on the board

        swap_value = -swap_value - 1 - SEE_VALUE[pt];
        stm = ~stm;

        if (swap_value >= 0) {
            // If the king captured and opponent still has attackers, king capture loses
            if (pt == KING && (attackers & pos.pieces(stm))) {
                stm = ~stm;  // Switch back
            }
            break;
        }
    }

    // The side that needs to move is the one that just failed
    return stm != piece_color(moving);
}

// ============================================================================
// Searcher
// ============================================================================

Searcher::Searcher() : tt(64), nodes_evaluated(0), stop_requested(false), allocated_time_ms(0) {
    std::memset(killer_moves, 0, sizeof(killer_moves));
    std::memset(history_table, 0, sizeof(history_table));
    std::memset(counter_moves, 0, sizeof(counter_moves));

    for (int d = 0; d < MAX_PLY; ++d) {
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

    // Delta pruning: if we're so far behind that even capturing a queen won't help
    const int DELTA_MARGIN = 1000;  // ~Queen value
    if (stand_pat + DELTA_MARGIN < alpha) return alpha;

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

        // SEE pruning: skip captures that lose material
        if (!see_ge(pos, m, 0)) continue;

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
        // Contempt: penalize draws when we're the engine (assume we're stronger)
        return -CONTEMPT;
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
    int static_eval = evaluate(pos);

    // 2. Null Move Pruning (NMP) for non-PV nodes
    Bitboard non_pawns = pos.pieces(us, KNIGHT) | pos.pieces(us, BISHOP) | pos.pieces(us, ROOK) | pos.pieces(us, QUEEN);
    if (!is_pv && !in_check && depth >= 3 && non_pawns) {
        if (static_eval >= beta) {
            StateInfo null_state;
            pos.make_null_move(null_state);
            int R = 2 + depth / 6;  // Conservative reduction
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

    // Score moves with TT, MVV-LVA, killers, counter moves, and history
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
                // Good captures scored high, bad captures scored low
                if (see_ge(pos, m, 0)) {
                    scores[i] = 1000000 + PieceVictimValue[victim] * 10 - PieceVictimValue[attacker];
                } else {
                    scores[i] = -1000000 + PieceVictimValue[victim] * 10 - PieceVictimValue[attacker];
                }
            } else if (move_flag(m) == FLAG_EN_PASSANT) {
                scores[i] = 1000500;  // EP captures are always good
            } else if (move_flag(m) == FLAG_PROMOTION) {
                scores[i] = 900000 + SEE_VALUE[move_promo_piece(m)];
            } else if (ply < MAX_PLY && m == killer_moves[ply][0]) {
                scores[i] = 800000;
            } else if (ply < MAX_PLY && m == killer_moves[ply][1]) {
                scores[i] = 700000;
            } else {
                // Counter move bonus
                scores[i] = history_table[us][from][to];
            }
        }
    }

    Move best_move = MOVE_NONE;
    int best_score = -INFINITY_SCORE;
    StateInfo state;
    int moves_searched = 0;

    for (int i = 0; i < moves.count; ++i) {
        // Selection sort: pick highest scored move
        int best_idx = i;
        for (int j = i + 1; j < moves.count; ++j) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        std::swap(moves.moves[i], moves.moves[best_idx]);
        std::swap(scores[i], scores[best_idx]);

        Move m = moves[i];
        Square from = move_from(m);
        Square to = move_to(m);
        Piece cap = pos.get_piece_at(to);
        bool is_capture = (cap != NO_PIECE) || (move_flag(m) == FLAG_EN_PASSANT);
        bool is_promotion = (move_flag(m) == FLAG_PROMOTION);
        bool is_quiet = !is_capture && !is_promotion;

        // Pruning: skip bad captures (negative SEE) at low depths for non-PV
        if (!is_pv && depth <= 3 && is_capture && moves_searched > 0 && !see_ge(pos, m, 0)) {
            if (scores[i] < 0) continue;  // Only skip clearly losing captures
        }

        pos.make_move(m, state);

        int score = 0;
        if (moves_searched == 0) {
            // PV move: full window search
            score = -pvs(pos, depth - 1, -beta, -alpha, ply + 1, is_pv);
        } else {
            // 8. Late Move Reductions (LMR) — using the precomputed table
            int reduction = 0;
            if (depth >= 3 && moves_searched >= 3 && is_quiet && !in_check) {
                reduction = static_cast<int>(lmr_table[std::min(depth, MAX_PLY - 1)][std::min(moves_searched, 63)]);

                // Reduce less for PV nodes
                if (is_pv) reduction -= 1;

                // Reduce less for killer moves
                if (ply < MAX_PLY && (m == killer_moves[ply][0] || m == killer_moves[ply][1])) {
                    reduction -= 1;
                }

                // Reduce more for moves with bad history
                if (history_table[us][from][to] < 0) {
                    reduction += 1;
                }

                // Clamp reduction
                reduction = std::clamp(reduction, 0, depth - 2);
            }

            // Zero-window search with reduction
            score = -pvs(pos, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, false);

            // Re-search if reduced move failed high
            if (reduction > 0 && score > alpha) {
                score = -pvs(pos, depth - 1, -alpha - 1, -alpha, ply + 1, false);
            }

            // Re-search full window if scout search failed high in PV
            if (score > alpha && score < beta) {
                score = -pvs(pos, depth - 1, -beta, -alpha, ply + 1, true);
            }
        }

        pos.unmake_move(m, state);
        moves_searched++;

        if (stop_requested) return 0;

        if (score > best_score) {
            best_score = score;
            best_move = m;
        }

        if (score > alpha) {
            alpha = score;
            if (score >= beta) {
                // Beta cutoff — update heuristics for quiet moves
                if (is_quiet && ply < MAX_PLY) {
                    // Killer moves
                    if (m != killer_moves[ply][0]) {
                        killer_moves[ply][1] = killer_moves[ply][0];
                        killer_moves[ply][0] = m;
                    }

                    // History heuristic with gravity (cap at ±16384)
                    int bonus = depth * depth;
                    int& h = history_table[us][from][to];
                    h += bonus - h * std::abs(bonus) / 16384;

                    // Penalize all previously searched quiet moves (history malus)
                    for (int k = 0; k < i; ++k) {
                        Move prev = moves[k];
                        Piece prev_cap = pos.get_piece_at(move_to(prev));
                        if (prev_cap == NO_PIECE && move_flag(prev) != FLAG_PROMOTION) {
                            int& ph = history_table[us][move_from(prev)][move_to(prev)];
                            ph += -bonus - ph * std::abs(bonus) / 16384;
                        }
                    }

                    // Counter move heuristic
                    // (Would need previous move info passed to pvs — simplified version)
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

    // New search generation for TT aging
    tt.new_search();

    // Age history table (divide by 2) to prevent overflow and encourage fresh data
    for (int c = 0; c < 2; ++c)
        for (int f = 0; f < 64; ++f)
            for (int t = 0; t < 64; ++t)
                history_table[c][f][t] /= 2;

    // Clear killers
    std::memset(killer_moves, 0, sizeof(killer_moves));

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
            // Gradual aspiration window widening
            int delta = 30;
            int alpha = std::max(-INFINITY_SCORE, best_score - delta);
            int beta = std::min(INFINITY_SCORE, best_score + delta);

            while (true) {
                score = pvs(pos, d, alpha, beta, 0, true);

                if (stop_requested) break;

                if (score <= alpha) {
                    // Fail low — widen alpha
                    beta = (alpha + beta) / 2;
                    alpha = std::max(-INFINITY_SCORE, alpha - delta);
                } else if (score >= beta) {
                    // Fail high — widen beta
                    beta = std::min(INFINITY_SCORE, beta + delta);
                } else {
                    break;  // Score within window
                }

                delta += delta / 2;  // Exponential widening
                if (delta > 500) {
                    // Fall back to full window
                    score = pvs(pos, d, -INFINITY_SCORE, INFINITY_SCORE, 0, true);
                    break;
                }
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

        // Don't start a new iteration if we've used more than half the time
        if (limits.movetime > 0 && elapsed_ms * 2 >= limits.movetime) {
            break;
        }
        // For time-controlled games: be smarter about when to start new iteration
        if (limits.time[us] > 0 && elapsed_ms * 2.5 >= allocated_time_ms) {
            break;
        }
    }

    std::cout << "bestmove " << move_to_uci(best_move) << std::endl;
    return best_move;
}

} // namespace Apex

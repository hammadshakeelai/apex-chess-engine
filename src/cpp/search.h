#pragma once

#include "types.h"
#include "position.h"
#include "movegen.h"
#include "bitboard.h"
#include <chrono>

namespace Apex {

constexpr int INFINITY_SCORE = 32000;
constexpr int MATE_SCORE = 30000;
constexpr int MAX_PLY = 128;

// Contempt: positive = engine avoids draws (aggressive), 0 = neutral
constexpr int CONTEMPT = 50;  // 50 centipawns — aggressively penalize draws

enum TTFlag : uint8_t {
    TT_NONE = 0,
    TT_EXACT = 1,
    TT_LOWERBOUND = 2,
    TT_UPPERBOUND = 3
};

struct TTEntry {
    uint64_t key = 0;
    Move move = MOVE_NONE;
    int16_t score = 0;
    int8_t depth = 0;
    TTFlag flag = TT_NONE;
    uint8_t generation = 0;  // Age tracking for replacement
};

class TranspositionTable {
public:
    TranspositionTable(size_t size_mb = 64);
    ~TranspositionTable();

    void resize(size_t size_mb);
    void clear();
    void new_search() { current_generation++; }

    bool probe(uint64_t key, int depth, int alpha, int beta, int ply, int& out_score, Move& out_move);
    void store(uint64_t key, int depth, int score, TTFlag flag, Move best_move, int ply);

private:
    TTEntry* table = nullptr;
    size_t count = 0;
    uint8_t current_generation = 0;
};

struct SearchLimits {
    int depth = 0;
    uint64_t nodes = 0;
    int movetime = 0;       // in milliseconds
    int time[2] = {0, 0};   // wtime, btime in ms
    int inc[2] = {0, 0};    // winc, binc in ms
    bool infinite = false;
};

// SEE piece values (same scale as MVV-LVA)
static const int SEE_VALUE[7] = { 100, 320, 330, 500, 900, 20000, 0 };

class Searcher {
public:
    Searcher();

    Move search(Position& pos, const SearchLimits& limits);
    void stop();

    uint64_t get_nodes() const { return nodes_evaluated; }

private:
    int pvs(Position& pos, int depth, int alpha, int beta, int ply, bool is_pv);
    int quiescence(Position& pos, int alpha, int beta, int ply);
    void score_moves(const Position& pos, MoveList& moves, Move tt_move, int ply);

    // Static Exchange Evaluation
    Bitboard attackers_to(const Position& pos, Square sq, Bitboard occ) const;
    Bitboard least_valuable_attacker(const Position& pos, Bitboard attacker_bb, Color side, PieceType& pt) const;
    bool see_ge(const Position& pos, Move m, int threshold) const;

    TranspositionTable tt;
    uint64_t nodes_evaluated;
    bool stop_requested;

    Move killer_moves[MAX_PLY][2];
    int history_table[2][64][64];
    Move counter_moves[12][64];  // Indexed by [prev_piece][prev_to]
    double lmr_table[MAX_PLY][64];

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    int allocated_time_ms;
};

} // namespace Apex

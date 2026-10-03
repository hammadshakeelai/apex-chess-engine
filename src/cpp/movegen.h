#pragma once

#include "types.h"
#include "position.h"

namespace Apex {

struct MoveList {
    Move moves[256];
    int count = 0;

    void add(Move m) {
        moves[count++] = m;
    }

    Move operator[](int idx) const {
        return moves[idx];
    }
};

void generate_legal_moves(Position& pos, MoveList& move_list);
void generate_captures_only(Position& pos, MoveList& move_list);

uint64_t perft(Position& pos, int depth);
void run_perft_suite(Position& pos, int depth);

} // namespace Apex

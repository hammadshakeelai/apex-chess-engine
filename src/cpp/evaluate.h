#pragma once

#include "position.h"

namespace Apex {

void init_evaluation();
bool load_nnue(const std::string& path);
bool is_nnue_enabled();
void set_nnue_enabled(bool enabled);

// Returns evaluation from perspective of side to move in centipawns
int evaluate(const Position& pos);

} // namespace Apex

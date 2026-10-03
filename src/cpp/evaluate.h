#pragma once

#include "position.h"

namespace Apex {

void init_evaluation();

// Returns evaluation from perspective of side to move in centipawns
int evaluate(const Position& pos);

} // namespace Apex

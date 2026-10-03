#pragma once

#include "position.h"
#include "search.h"

namespace Apex {

class UCIEngine {
public:
    UCIEngine();
    void loop();

private:
    void handle_position(std::istringstream& iss);
    void handle_go(std::istringstream& iss);

    Position pos;
    Searcher searcher;
};

} // namespace Apex

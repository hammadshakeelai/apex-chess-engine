#include "types.h"
#include "bitboard.h"
#include "position.h"
#include "movegen.h"
#include "uci.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    Apex::init_bitboards();
    Apex::Position::init_zobrist();

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--perft" || arg == "perft") {
            int depth = (argc > 2) ? std::stoi(argv[2]) : 5;
            Apex::Position pos;
            Apex::run_perft_suite(pos, depth);
            return 0;
        } else if (arg == "--version" || arg == "-v") {
            std::cout << "ApexChess 1.0 (High-Performance C++ Core Engine)\n";
            std::cout << "Target: AMD Zen 3 / x86_64 AVX2 BMI2\n";
            return 0;
        }
    }

    Apex::UCIEngine engine;
    engine.loop();

    return 0;
}

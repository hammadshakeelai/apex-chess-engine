#include "uci.h"
#include "movegen.h"
#include "evaluate.h"
#include "book.h"
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

namespace Apex {

UCIEngine::UCIEngine() {
    pos.set_startpos();
    // Auto-load NNUE binary if present
    std::ifstream test_f("weights/apex_nnue.bin", std::ios::binary);
    if (test_f.good()) {
        test_f.close();
        load_nnue("weights/apex_nnue.bin");
    }

    // Auto-load Polyglot Opening Book if present
    std::ifstream test_b("weights/apex_book.bin", std::ios::binary);
    if (test_b.good()) {
        test_b.close();
        GlobalBook.load("weights/apex_book.bin");
    }
}

void UCIEngine::handle_position(std::istringstream& iss) {
    std::string token;
    iss >> token;

    if (token == "startpos") {
        pos.set_startpos();
        iss >> token; // Check if next token is "moves"
    } else if (token == "fen") {
        std::string fen_part, fen;
        while (iss >> fen_part && fen_part != "moves") {
            if (!fen.empty()) fen += " ";
            fen += fen_part;
        }
        pos.set_fen(fen);
        token = fen_part; // might be "moves"
    }

    if (token == "moves") {
        std::string move_str;
        StateInfo state;
        while (iss >> move_str) {
            MoveList legal_moves;
            generate_legal_moves(pos, legal_moves);

            for (int i = 0; i < legal_moves.count; ++i) {
                if (move_to_uci(legal_moves[i]) == move_str) {
                    pos.make_move(legal_moves[i], state);
                    break;
                }
            }
        }
    }
}

void UCIEngine::handle_go(std::istringstream& iss) {
    SearchLimits limits;
    std::string token;

    while (iss >> token) {
        if (token == "depth") iss >> limits.depth;
        else if (token == "movetime") iss >> limits.movetime;
        else if (token == "wtime") iss >> limits.time[WHITE];
        else if (token == "btime") iss >> limits.time[BLACK];
        else if (token == "winc") iss >> limits.inc[WHITE];
        else if (token == "binc") iss >> limits.inc[BLACK];
        else if (token == "nodes") iss >> limits.nodes;
        else if (token == "infinite") limits.infinite = true;
    }

    if (limits.depth == 0 && limits.movetime == 0 && limits.time[WHITE] == 0 && !limits.infinite) {
        limits.depth = 6; // Default search depth
    }

    searcher.search(pos, limits);
}

void UCIEngine::loop() {
    std::string line;

    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "uci") {
            std::cout << "id name ApexChess 1.0 (C++ Core + AVX2 NNUE)" << std::endl;
            std::cout << "id author Apex Deep Neural Research" << std::endl;
            std::cout << "option name Hash type spin default 32 min 1 max 1024" << std::endl;
            std::cout << "option name EvalFile type string default weights/apex_nnue.bin" << std::endl;
            std::cout << "option name UseNNUE type check default true" << std::endl;
            std::cout << "option name BookFile type string default weights/apex_book.bin" << std::endl;
            std::cout << "option name OwnBook type check default true" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (cmd == "setoption") {
            std::string name_tok, opt_name, val_tok, opt_val;
            iss >> name_tok >> opt_name;
            if (iss >> val_tok >> opt_val) {
                if (opt_name == "EvalFile") {
                    load_nnue(opt_val);
                } else if (opt_name == "UseNNUE") {
                    set_nnue_enabled(opt_val == "true");
                } else if (opt_name == "BookFile") {
                    GlobalBook.load(opt_val);
                } else if (opt_name == "OwnBook") {
                    if (opt_val == "false") GlobalBook.close();
                    else GlobalBook.load("weights/apex_book.bin");
                }
            }
        } else if (cmd == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (cmd == "ucinewgame") {
            pos.set_startpos();
        } else if (cmd == "position") {
            handle_position(iss);
        } else if (cmd == "go") {
            handle_go(iss);
        } else if (cmd == "perft") {
            int depth = 5;
            iss >> depth;
            run_perft_suite(pos, depth);
        } else if (cmd == "d" || cmd == "print") {
            pos.print();
        } else if (cmd == "stop") {
            searcher.stop();
        } else if (cmd == "quit") {
            break;
        }
    }
}

} // namespace Apex

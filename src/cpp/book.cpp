#include "book.h"
#include "polyglot_keys.h"
#include "movegen.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

namespace Apex {

OpeningBook GlobalBook;

OpeningBook::OpeningBook() : entries(nullptr), count(0) {}

OpeningBook::~OpeningBook() {
    close();
}

void OpeningBook::close() {
    delete[] entries;
    entries = nullptr;
    count = 0;
}

static inline uint64_t swap_u64(uint64_t val) {
    return __builtin_bswap64(val);
}

static inline uint16_t swap_u16(uint16_t val) {
    return __builtin_bswap16(val);
}

bool OpeningBook::load(const std::string& filepath) {
    close();
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size <= 0 || (file_size % sizeof(PolyglotEntry)) != 0) {
        return false;
    }

    count = file_size / sizeof(PolyglotEntry);
    entries = new PolyglotEntry[count];

    file.read(reinterpret_cast<char*>(entries), file_size);
    if (!file) {
        close();
        return false;
    }

    // Convert from big-endian (Polyglot standard) to host endian
    for (size_t i = 0; i < count; ++i) {
        entries[i].key = swap_u64(entries[i].key);
        entries[i].move = swap_u16(entries[i].move);
        entries[i].weight = swap_u16(entries[i].weight);
        entries[i].learn = __builtin_bswap32(entries[i].learn);
    }

    // Ensure sorted by key
    std::sort(entries, entries + count, [](const PolyglotEntry& a, const PolyglotEntry& b) {
        return a.key < b.key;
    });

    std::cout << "info string Loaded opening book: " << filepath << " (" << count << " entries)" << std::endl;
    return true;
}

uint64_t OpeningBook::compute_key(const Position& pos) {
    uint64_t key = 0;

    // 1. Pieces
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = pos.get_piece_at(static_cast<Square>(sq));
        if (p != NO_PIECE) {
            PieceType pt = piece_type(p);
            Color c = piece_color(p);
            // Polyglot piece indexing: pt * 2 + (c == WHITE ? 1 : 0)
            int piece_idx = pt * 2 + (c == WHITE ? 1 : 0);
            key ^= PolyglotRandom[64 * piece_idx + sq];
        }
    }

    // 2. Castling
    uint8_t cr = pos.castling();
    if (cr & WHITE_OO)  key ^= PolyglotRandom[768];
    if (cr & WHITE_OOO) key ^= PolyglotRandom[769];
    if (cr & BLACK_OO)  key ^= PolyglotRandom[770];
    if (cr & BLACK_OOO) key ^= PolyglotRandom[771];

    // 3. En passant target square (only if capturable by pawn)
    Square ep = pos.en_passant();
    if (ep != SQ_NONE) {
        Color us = pos.turn();
        int ep_file = square_file(ep);
        int ep_rank = square_rank(ep);
        Bitboard pawns = pos.pieces(us, PAWN);
        bool can_cap = false;
        if (us == WHITE && ep_rank == 5) {
            Square p1 = (ep_file > 0) ? make_square(ep_file - 1, 4) : SQ_NONE;
            Square p2 = (ep_file < 7) ? make_square(ep_file + 1, 4) : SQ_NONE;
            if ((p1 != SQ_NONE && (pawns & SquareBB[p1])) || (p2 != SQ_NONE && (pawns & SquareBB[p2]))) {
                can_cap = true;
            }
        } else if (us == BLACK && ep_rank == 2) {
            Square p1 = (ep_file > 0) ? make_square(ep_file - 1, 3) : SQ_NONE;
            Square p2 = (ep_file < 7) ? make_square(ep_file + 1, 3) : SQ_NONE;
            if ((p1 != SQ_NONE && (pawns & SquareBB[p1])) || (p2 != SQ_NONE && (pawns & SquareBB[p2]))) {
                can_cap = true;
            }
        }
        if (can_cap) {
            key ^= PolyglotRandom[772 + ep_file];
        }
    }

    // 4. Active color (Polyglot hashes 780 if WHITE)
    if (pos.turn() == WHITE) {
        key ^= PolyglotRandom[780];
    }

    return key;
}

Move OpeningBook::probe(Position& pos) {
    if (!is_loaded()) return MOVE_NONE;

    uint64_t key = compute_key(pos);

    // Binary search for key
    auto cmp = [](const PolyglotEntry& entry, uint64_t target_key) {
        return entry.key < target_key;
    };
    const PolyglotEntry* it = std::lower_bound(entries, entries + count, key, cmp);

    if (it == entries + count || it->key != key) {
        return MOVE_NONE;
    }

    // Collect all candidate moves for this position
    MoveList legal_moves;
    generate_legal_moves(pos, legal_moves);

    struct Candidate {
        Move move;
        uint16_t weight;
    };
    std::vector<Candidate> candidates;
    uint32_t total_weight = 0;

    while (it != entries + count && it->key == key) {
        uint16_t raw_move = it->move;
        Square to = static_cast<Square>(raw_move & 0x3F);
        Square from = static_cast<Square>((raw_move >> 6) & 0x3F);
        uint16_t promo_code = (raw_move >> 12) & 0x07;

        // Castling translation (Polyglot encodes e1h1 as e1g1, e1a1 as e1c1)
        if (pos.get_piece_at(from) == W_KING && from == SQ_E1) {
            if (to == SQ_H1) to = SQ_G1;
            else if (to == SQ_A1) to = SQ_C1;
        } else if (pos.get_piece_at(from) == B_KING && from == SQ_E8) {
            if (to == SQ_H8) to = SQ_G8;
            else if (to == SQ_A8) to = SQ_C8;
        }

        PieceType promo = NO_PIECE_TYPE;
        if (promo_code == 1) promo = KNIGHT;
        else if (promo_code == 2) promo = BISHOP;
        else if (promo_code == 3) promo = ROOK;
        else if (promo_code == 4) promo = QUEEN;

        // Find match in legal moves
        for (int i = 0; i < legal_moves.count; ++i) {
            Move m = legal_moves[i];
            if (move_from(m) == from && move_to(m) == to) {
                if (promo != NO_PIECE_TYPE && move_flag(m) == FLAG_PROMOTION) {
                    if (move_promo_piece(m) != promo) continue;
                }
                candidates.push_back({m, it->weight});
                total_weight += it->weight;
                break;
            }
        }
        ++it;
    }

    if (candidates.empty() || total_weight == 0) {
        return MOVE_NONE;
    }

    // Weighted random selection
    static std::mt19937 rng(1337);
    std::uniform_int_distribution<uint32_t> dist(1, total_weight);
    uint32_t roll = dist(rng);
    uint32_t acc = 0;

    for (const auto& c : candidates) {
        acc += c.weight;
        if (acc >= roll) {
            return c.move;
        }
    }

    return candidates.front().move;
}

} // namespace Apex

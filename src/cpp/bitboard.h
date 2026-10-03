#pragma once

#include "types.h"

namespace Apex {

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_B = 0x0202020202020202ULL;
constexpr Bitboard FILE_C = 0x0404040404040404ULL;
constexpr Bitboard FILE_D = 0x0808080808080808ULL;
constexpr Bitboard FILE_E = 0x1010101010101010ULL;
constexpr Bitboard FILE_F = 0x2020202020202020ULL;
constexpr Bitboard FILE_G = 0x4040404040404040ULL;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;

constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
constexpr Bitboard RANK_3 = 0x0000000000FF0000ULL;
constexpr Bitboard RANK_4 = 0x00000000FF000000ULL;
constexpr Bitboard RANK_5 = 0x000000FF00000000ULL;
constexpr Bitboard RANK_6 = 0x0000FF0000000000ULL;
constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

extern Bitboard SquareBB[64];
extern Bitboard PawnAttacks[2][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];

void init_bitboards();

inline Bitboard square_bb(Square sq) {
    return SquareBB[sq];
}

inline Bitboard pawn_attacks(Color c, Square sq) {
    return PawnAttacks[c][sq];
}

inline Bitboard knight_attacks(Square sq) {
    return KnightAttacks[sq];
}

inline Bitboard king_attacks(Square sq) {
    return KingAttacks[sq];
}

Bitboard bishop_attacks(Square sq, Bitboard occ);
Bitboard rook_attacks(Square sq, Bitboard occ);

inline Bitboard queen_attacks(Square sq, Bitboard occ) {
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ);
}

void print_bitboard(Bitboard bb);

} // namespace Apex

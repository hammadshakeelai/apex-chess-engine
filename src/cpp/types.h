#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <iostream>

namespace Apex {

typedef uint64_t Bitboard;

enum Color : uint8_t {
    WHITE = 0,
    BLACK = 1,
    NO_COLOR = 2
};

inline Color operator~(Color c) {
    return static_cast<Color>(c ^ 1);
}

enum PieceType : uint8_t {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    NO_PIECE_TYPE = 6
};

enum Piece : uint8_t {
    W_PAWN = 0, W_KNIGHT = 1, W_BISHOP = 2, W_ROOK = 3, W_QUEEN = 4, W_KING = 5,
    B_PAWN = 6, B_KNIGHT = 7, B_BISHOP = 8, B_ROOK = 9, B_QUEEN = 10, B_KING = 11,
    NO_PIECE = 12
};

inline Piece make_piece(Color c, PieceType pt) {
    return static_cast<Piece>((c * 6) + pt);
}

inline PieceType piece_type(Piece p) {
    return static_cast<PieceType>(p % 6);
}

inline Color piece_color(Piece p) {
    return static_cast<Color>(p / 6);
}

enum Square : uint8_t {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE = 64
};

inline Square make_square(int file, int rank) {
    return static_cast<Square>((rank << 3) | file);
}

inline int square_file(Square sq) {
    return sq & 7;
}

inline int square_rank(Square sq) {
    return sq >> 3;
}

// Castling Rights bitmask
enum CastlingRights : uint8_t {
    NO_CASTLING = 0,
    WHITE_OO = 1,
    WHITE_OOO = 2,
    BLACK_OO = 4,
    BLACK_OOO = 8,
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
    ALL_CASTLING = WHITE_CASTLING | BLACK_CASTLING
};

// 16-bit packed move:
// bits 0-5:   from square (0-63)
// bits 6-11:  to square (0-63)
// bits 12-13: promotion piece type (0: KNIGHT, 1: BISHOP, 2: ROOK, 3: QUEEN)
// bits 14-15: move flag (0: Normal, 1: Promotion, 2: En Passant, 3: Castling)
typedef uint16_t Move;

constexpr Move MOVE_NONE = 0;

enum MoveFlag : uint8_t {
    FLAG_NORMAL = 0,
    FLAG_PROMOTION = 1,
    FLAG_EN_PASSANT = 2,
    FLAG_CASTLING = 3
};

inline Move make_move(Square from, Square to, MoveFlag flag = FLAG_NORMAL, PieceType promo = KNIGHT) {
    uint16_t promo_bits = 0;
    if (flag == FLAG_PROMOTION) {
        promo_bits = (promo - KNIGHT) & 0x3;
    }
    return static_cast<Move>(from | (to << 6) | (promo_bits << 12) | (flag << 14));
}

inline Square move_from(Move m) {
    return static_cast<Square>(m & 0x3F);
}

inline Square move_to(Move m) {
    return static_cast<Square>((m >> 6) & 0x3F);
}

inline MoveFlag move_flag(Move m) {
    return static_cast<MoveFlag>((m >> 14) & 0x3);
}

inline PieceType move_promo_piece(Move m) {
    return static_cast<PieceType>((((m >> 12) & 0x3) + KNIGHT));
}

inline std::string square_to_string(Square sq) {
    if (sq >= SQ_NONE) return "-";
    char f = 'a' + square_file(sq);
    char r = '1' + square_rank(sq);
    return std::string{f, r};
}

inline Square string_to_square(const std::string& str) {
    if (str.length() < 2 || str[0] < 'a' || str[0] > 'h' || str[1] < '1' || str[1] > '8') {
        return SQ_NONE;
    }
    return make_square(str[0] - 'a', str[1] - '1');
}

inline std::string move_to_uci(Move m) {
    if (m == MOVE_NONE) return "0000";
    std::string s = square_to_string(move_from(m)) + square_to_string(move_to(m));
    if (move_flag(m) == FLAG_PROMOTION) {
        switch (move_promo_piece(m)) {
            case QUEEN: s += 'q'; break;
            case ROOK: s += 'r'; break;
            case BISHOP: s += 'b'; break;
            case KNIGHT: s += 'n'; break;
            default: break;
        }
    }
    return s;
}

// Bit manipulation intrinsics
inline int popcount(Bitboard b) {
    return __builtin_popcountll(b);
}

inline Square lsb(Bitboard b) {
    return static_cast<Square>(__builtin_ctzll(b));
}

inline Square pop_lsb(Bitboard& b) {
    Square sq = lsb(b);
    b &= b - 1;
    return sq;
}

} // namespace Apex

#pragma once

#include "types.h"
#include "bitboard.h"
#include <string>
#include <vector>

namespace Apex {

struct StateInfo {
    uint8_t castling_rights = 0;
    Square en_passant_sq = SQ_NONE;
    int halfmove_clock = 0;
    Piece captured_piece = NO_PIECE;
    uint64_t zobrist_key = 0;
};

class Position {
public:
    Position();

    void set_startpos();
    bool set_fen(const std::string& fen);
    std::string to_fen() const;

    void make_move(Move m, StateInfo& state);
    void unmake_move(Move m, const StateInfo& state);

    bool is_square_attacked(Square sq, Color by_color) const;
    bool in_check() const;

    Bitboard occupied() const { return color_bb[WHITE] | color_bb[BLACK]; }
    Bitboard pieces(Color c) const { return color_bb[c]; }
    Bitboard pieces(Color c, PieceType pt) const { return piece_bb[make_piece(c, pt)]; }
    Piece get_piece_at(Square sq) const { return piece_at[sq]; }
    Color turn() const { return side_to_move; }
    Square king_square(Color c) const { return lsb(piece_bb[make_piece(c, KING)]); }
    uint8_t castling() const { return castling_rights; }
    Square en_passant() const { return en_passant_sq; }
    uint64_t hash() const { return zobrist_key; }

    void print() const;

    static void init_zobrist();

private:
    void clear();
    void put_piece(Piece p, Square sq);
    void remove_piece(Square sq);
    void move_piece(Square from, Square to);
    uint64_t compute_zobrist() const;

    Bitboard piece_bb[12];
    Bitboard color_bb[2];
    Piece piece_at[64];

    Color side_to_move;
    uint8_t castling_rights;
    Square en_passant_sq;
    int halfmove_clock;
    int fullmove_number;
    uint64_t zobrist_key;

    static uint64_t ZobristPieces[12][64];
    static uint64_t ZobristTurn;
    static uint64_t ZobristCastling[16];
    static uint64_t ZobristEnPassant[65];
};

} // namespace Apex

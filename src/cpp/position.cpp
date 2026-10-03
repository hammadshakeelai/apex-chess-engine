#include "position.h"
#include <sstream>
#include <iostream>
#include <cctype>

namespace Apex {

uint64_t Position::ZobristPieces[12][64];
uint64_t Position::ZobristTurn;
uint64_t Position::ZobristCastling[16];
uint64_t Position::ZobristEnPassant[65];

// Castling rights update masks
// When piece on square moves or is captured, bitwise AND with mask
static uint8_t CastlingMask[64];

static uint64_t xorshift64_state = 1070372;
static uint64_t xorshift64() {
    uint64_t x = xorshift64_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return xorshift64_state = x;
}

void Position::init_zobrist() {
    xorshift64_state = 88172645463325252ULL;
    for (int p = 0; p < 12; ++p) {
        for (int sq = 0; sq < 64; ++sq) {
            ZobristPieces[p][sq] = xorshift64();
        }
    }
    ZobristTurn = xorshift64();
    for (int i = 0; i < 16; ++i) {
        ZobristCastling[i] = xorshift64();
    }
    for (int i = 0; i < 65; ++i) {
        ZobristEnPassant[i] = xorshift64();
    }

    // Castling masks
    for (int sq = 0; sq < 64; ++sq) CastlingMask[sq] = ALL_CASTLING;
    CastlingMask[SQ_E1] &= ~WHITE_CASTLING;
    CastlingMask[SQ_A1] &= ~WHITE_OOO;
    CastlingMask[SQ_H1] &= ~WHITE_OO;

    CastlingMask[SQ_E8] &= ~BLACK_CASTLING;
    CastlingMask[SQ_A8] &= ~BLACK_OOO;
    CastlingMask[SQ_H8] &= ~BLACK_OO;
}

Position::Position() {
    clear();
    set_startpos();
}

void Position::clear() {
    for (int i = 0; i < 12; ++i) piece_bb[i] = 0;
    color_bb[WHITE] = 0;
    color_bb[BLACK] = 0;
    for (int i = 0; i < 64; ++i) piece_at[i] = NO_PIECE;

    side_to_move = WHITE;
    castling_rights = NO_CASTLING;
    en_passant_sq = SQ_NONE;
    halfmove_clock = 0;
    fullmove_number = 1;
    zobrist_key = 0;
}

void Position::put_piece(Piece p, Square sq) {
    Color c = piece_color(p);
    piece_bb[p] |= SquareBB[sq];
    color_bb[c] |= SquareBB[sq];
    piece_at[sq] = p;
    zobrist_key ^= ZobristPieces[p][sq];
}

void Position::remove_piece(Square sq) {
    Piece p = piece_at[sq];
    if (p == NO_PIECE) return;
    Color c = piece_color(p);
    piece_bb[p] &= ~SquareBB[sq];
    color_bb[c] &= ~SquareBB[sq];
    piece_at[sq] = NO_PIECE;
    zobrist_key ^= ZobristPieces[p][sq];
}

void Position::move_piece(Square from, Square to) {
    Piece p = piece_at[from];
    Color c = piece_color(p);
    Bitboard from_to = SquareBB[from] | SquareBB[to];
    piece_bb[p] ^= from_to;
    color_bb[c] ^= from_to;
    piece_at[from] = NO_PIECE;
    piece_at[to] = p;

    zobrist_key ^= ZobristPieces[p][from];
    zobrist_key ^= ZobristPieces[p][to];
}

uint64_t Position::compute_zobrist() const {
    uint64_t k = 0;
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = piece_at[sq];
        if (p != NO_PIECE) {
            k ^= ZobristPieces[p][sq];
        }
    }
    if (side_to_move == BLACK) k ^= ZobristTurn;
    k ^= ZobristCastling[castling_rights];
    k ^= ZobristEnPassant[en_passant_sq];
    return k;
}

void Position::set_startpos() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool Position::set_fen(const std::string& fen) {
    clear();
    std::istringstream iss(fen);
    std::string pieces_str, turn_str, castling_str, ep_str;
    int halfmove = 0, fullmove = 1;

    if (!(iss >> pieces_str >> turn_str >> castling_str >> ep_str)) {
        return false;
    }
    iss >> halfmove >> fullmove;

    // 1. Piece placement
    int rank = 7, file = 0;
    for (char c : pieces_str) {
        if (c == '/') {
            rank--;
            file = 0;
        } else if (std::isdigit(c)) {
            file += (c - '0');
        } else {
            Square sq = make_square(file, rank);
            switch (c) {
                case 'P': put_piece(W_PAWN, sq); break;
                case 'N': put_piece(W_KNIGHT, sq); break;
                case 'B': put_piece(W_BISHOP, sq); break;
                case 'R': put_piece(W_ROOK, sq); break;
                case 'Q': put_piece(W_QUEEN, sq); break;
                case 'K': put_piece(W_KING, sq); break;

                case 'p': put_piece(B_PAWN, sq); break;
                case 'n': put_piece(B_KNIGHT, sq); break;
                case 'b': put_piece(B_BISHOP, sq); break;
                case 'r': put_piece(B_ROOK, sq); break;
                case 'q': put_piece(B_QUEEN, sq); break;
                case 'k': put_piece(B_KING, sq); break;
                default: break;
            }
            file++;
        }
    }

    // 2. Active color
    side_to_move = (turn_str == "w") ? WHITE : BLACK;

    // 3. Castling rights
    castling_rights = NO_CASTLING;
    for (char c : castling_str) {
        if (c == 'K') castling_rights |= WHITE_OO;
        else if (c == 'Q') castling_rights |= WHITE_OOO;
        else if (c == 'k') castling_rights |= BLACK_OO;
        else if (c == 'q') castling_rights |= BLACK_OOO;
    }

    // 4. En passant target
    if (ep_str != "-") {
        en_passant_sq = string_to_square(ep_str);
    } else {
        en_passant_sq = SQ_NONE;
    }

    halfmove_clock = halfmove;
    fullmove_number = fullmove;
    zobrist_key = compute_zobrist();

    return true;
}

std::string Position::to_fen() const {
    std::string fen = "";
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(f, r);
            Piece p = piece_at[sq];
            if (p == NO_PIECE) {
                empty++;
            } else {
                if (empty > 0) {
                    fen += std::to_string(empty);
                    empty = 0;
                }
                static const char symbols[] = "PNBRQKpnbrqk";
                fen += symbols[p];
            }
        }
        if (empty > 0) fen += std::to_string(empty);
        if (r > 0) fen += '/';
    }

    fen += (side_to_move == WHITE) ? " w " : " b ";

    std::string castling = "";
    if (castling_rights & WHITE_OO) castling += 'K';
    if (castling_rights & WHITE_OOO) castling += 'Q';
    if (castling_rights & BLACK_OO) castling += 'k';
    if (castling_rights & BLACK_OOO) castling += 'q';
    if (castling.empty()) castling = "-";
    fen += castling + " ";

    fen += (en_passant_sq != SQ_NONE) ? square_to_string(en_passant_sq) : "-";
    fen += " " + std::to_string(halfmove_clock) + " " + std::to_string(fullmove_number);
    return fen;
}

bool Position::is_square_attacked(Square sq, Color by_color) const {
    // 1. Pawn attacks
    // White pawn attacks sq from sq - 7 or sq - 9 (i.e. pawn on black_pawn_attack(sq))
    Color defender = ~by_color;
    if (pawn_attacks(defender, sq) & pieces(by_color, PAWN)) return true;

    // 2. Knight attacks
    if (knight_attacks(sq) & pieces(by_color, KNIGHT)) return true;

    // 3. King attacks
    if (king_attacks(sq) & pieces(by_color, KING)) return true;

    // 4. Bishop/Queen attacks
    Bitboard occ = occupied();
    Bitboard bishops = pieces(by_color, BISHOP) | pieces(by_color, QUEEN);
    if (bishops && (bishop_attacks(sq, occ) & bishops)) return true;

    // 5. Rook/Queen attacks
    Bitboard rooks = pieces(by_color, ROOK) | pieces(by_color, QUEEN);
    if (rooks && (rook_attacks(sq, occ) & rooks)) return true;

    return false;
}

bool Position::in_check() const {
    Square ksq = king_square(side_to_move);
    return is_square_attacked(ksq, ~side_to_move);
}

void Position::make_move(Move m, StateInfo& state) {
    Square from = move_from(m);
    Square to = move_to(m);
    MoveFlag flag = move_flag(m);

    // Save previous state
    state.castling_rights = castling_rights;
    state.en_passant_sq = en_passant_sq;
    state.halfmove_clock = halfmove_clock;
    state.captured_piece = piece_at[to];
    state.zobrist_key = zobrist_key;

    // Remove en passant square from zobrist
    zobrist_key ^= ZobristEnPassant[en_passant_sq];
    en_passant_sq = SQ_NONE;

    halfmove_clock++;
    if (side_to_move == BLACK) fullmove_number++;

    Piece moving_piece = piece_at[from];
    PieceType pt = piece_type(moving_piece);

    // Reset 50-move clock on pawn move or capture
    if (pt == PAWN || state.captured_piece != NO_PIECE) {
        halfmove_clock = 0;
    }

    // Normal capture
    if (state.captured_piece != NO_PIECE) {
        remove_piece(to);
    }

    // Handle special moves
    if (flag == FLAG_EN_PASSANT) {
        Square cap_sq = (side_to_move == WHITE) ? make_square(square_file(to), square_rank(to) - 1)
                                                : make_square(square_file(to), square_rank(to) + 1);
        state.captured_piece = piece_at[cap_sq];
        remove_piece(cap_sq);
        move_piece(from, to);
    } else if (flag == FLAG_CASTLING) {
        move_piece(from, to);
        if (to == SQ_G1) move_piece(SQ_H1, SQ_F1);
        else if (to == SQ_C1) move_piece(SQ_A1, SQ_D1);
        else if (to == SQ_G8) move_piece(SQ_H8, SQ_F8);
        else if (to == SQ_C8) move_piece(SQ_A8, SQ_D8);
    } else if (flag == FLAG_PROMOTION) {
        remove_piece(from);
        PieceType promo = move_promo_piece(m);
        put_piece(make_piece(side_to_move, promo), to);
    } else {
        // Normal move
        move_piece(from, to);

        // Check for double pawn push (set en passant target)
        if (pt == PAWN) {
            if (side_to_move == WHITE && square_rank(to) - square_rank(from) == 2) {
                en_passant_sq = make_square(square_file(from), square_rank(from) + 1);
                zobrist_key ^= ZobristEnPassant[en_passant_sq];
            } else if (side_to_move == BLACK && square_rank(from) - square_rank(to) == 2) {
                en_passant_sq = make_square(square_file(from), square_rank(from) - 1);
                zobrist_key ^= ZobristEnPassant[en_passant_sq];
            }
        }
    }

    // Update castling rights
    zobrist_key ^= ZobristCastling[castling_rights];
    castling_rights &= CastlingMask[from];
    castling_rights &= CastlingMask[to];
    zobrist_key ^= ZobristCastling[castling_rights];

    // Toggle turn
    side_to_move = ~side_to_move;
    zobrist_key ^= ZobristTurn;
}

void Position::unmake_move(Move m, const StateInfo& state) {
    Square from = move_from(m);
    Square to = move_to(m);
    MoveFlag flag = move_flag(m);

    side_to_move = ~side_to_move;
    if (side_to_move == BLACK) fullmove_number--;

    if (flag == FLAG_CASTLING) {
        move_piece(to, from);
        if (to == SQ_G1) move_piece(SQ_F1, SQ_H1);
        else if (to == SQ_C1) move_piece(SQ_D1, SQ_A1);
        else if (to == SQ_G8) move_piece(SQ_F8, SQ_H8);
        else if (to == SQ_C8) move_piece(SQ_D8, SQ_A8);
    } else if (flag == FLAG_PROMOTION) {
        remove_piece(to);
        put_piece(make_piece(side_to_move, PAWN), from);
        if (state.captured_piece != NO_PIECE) {
            put_piece(state.captured_piece, to);
        }
    } else if (flag == FLAG_EN_PASSANT) {
        move_piece(to, from);
        Square cap_sq = (side_to_move == WHITE) ? make_square(square_file(to), square_rank(to) - 1)
                                                : make_square(square_file(to), square_rank(to) + 1);
        put_piece(state.captured_piece, cap_sq);
    } else {
        move_piece(to, from);
        if (state.captured_piece != NO_PIECE) {
            put_piece(state.captured_piece, to);
        }
    }

    castling_rights = state.castling_rights;
    en_passant_sq = state.en_passant_sq;
    halfmove_clock = state.halfmove_clock;
    zobrist_key = state.zobrist_key;
}

void Position::print() const {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        std::cout << (r + 1) << " |";
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(f, r);
            Piece p = piece_at[sq];
            if (p == NO_PIECE) {
                std::cout << " . |";
            } else {
                static const char symbols[] = "PNBRQKpnbrqk";
                std::cout << " " << symbols[p] << " |";
            }
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n";
    std::cout << "  FEN: " << to_fen() << "\n";
    std::cout << "  Turn: " << (side_to_move == WHITE ? "White" : "Black") << "\n\n";
}

} // namespace Apex

#include "bitboard.h"
#include <iostream>

namespace Apex {

Bitboard SquareBB[64];
Bitboard PawnAttacks[2][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];

static const int KnightDeltas[8][2] = {
    {1, 2}, {2, 1}, {2, -1}, {1, -2},
    {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}
};

static const int KingDeltas[8][2] = {
    {0, 1}, {1, 1}, {1, 0}, {1, -1},
    {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}
};

void init_bitboards() {
    for (int sq = 0; sq < 64; ++sq) {
        SquareBB[sq] = 1ULL << sq;
    }

    // Pawn attacks
    for (int sq = 0; sq < 64; ++sq) {
        int f = square_file(static_cast<Square>(sq));
        int r = square_rank(static_cast<Square>(sq));

        // White pawns move up (rank + 1)
        Bitboard w_atk = 0;
        if (r < 7) {
            if (f > 0) w_atk |= SquareBB[make_square(f - 1, r + 1)];
            if (f < 7) w_atk |= SquareBB[make_square(f + 1, r + 1)];
        }
        PawnAttacks[WHITE][sq] = w_atk;

        // Black pawns move down (rank - 1)
        Bitboard b_atk = 0;
        if (r > 0) {
            if (f > 0) b_atk |= SquareBB[make_square(f - 1, r - 1)];
            if (f < 7) b_atk |= SquareBB[make_square(f + 1, r - 1)];
        }
        PawnAttacks[BLACK][sq] = b_atk;
    }

    // Knight attacks
    for (int sq = 0; sq < 64; ++sq) {
        int f = square_file(static_cast<Square>(sq));
        int r = square_rank(static_cast<Square>(sq));
        Bitboard atk = 0;

        for (int i = 0; i < 8; ++i) {
            int nf = f + KnightDeltas[i][0];
            int nr = r + KnightDeltas[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                atk |= SquareBB[make_square(nf, nr)];
            }
        }
        KnightAttacks[sq] = atk;
    }

    // King attacks
    for (int sq = 0; sq < 64; ++sq) {
        int f = square_file(static_cast<Square>(sq));
        int r = square_rank(static_cast<Square>(sq));
        Bitboard atk = 0;

        for (int i = 0; i < 8; ++i) {
            int nf = f + KingDeltas[i][0];
            int nr = r + KingDeltas[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                atk |= SquareBB[make_square(nf, nr)];
            }
        }
        KingAttacks[sq] = atk;
    }
}

Bitboard bishop_attacks(Square sq, Bitboard occ) {
    Bitboard attacks = 0;
    int f = square_file(sq);
    int r = square_rank(sq);

    // NE
    for (int df = 1, dr = 1; f + df < 8 && r + dr < 8; ++df, ++dr) {
        Square target = make_square(f + df, r + dr);
        attacks |= SquareBB[target];
        if (occ & SquareBB[target]) break;
    }
    // NW
    for (int df = -1, dr = 1; f + df >= 0 && r + dr < 8; --df, ++dr) {
        Square target = make_square(f + df, r + dr);
        attacks |= SquareBB[target];
        if (occ & SquareBB[target]) break;
    }
    // SE
    for (int df = 1, dr = -1; f + df < 8 && r + dr >= 0; ++df, --dr) {
        Square target = make_square(f + df, r + dr);
        attacks |= SquareBB[target];
        if (occ & SquareBB[target]) break;
    }
    // SW
    for (int df = -1, dr = -1; f + df >= 0 && r + dr >= 0; --df, --dr) {
        Square target = make_square(f + df, r + dr);
        attacks |= SquareBB[target];
        if (occ & SquareBB[target]) break;
    }

    return attacks;
}

Bitboard rook_attacks(Square sq, Bitboard occ) {
    Bitboard attacks = 0;
    int f = square_file(sq);
    int r = square_rank(sq);

    // North
    for (int nr = r + 1; nr < 8; ++nr) {
        Square target = make_square(f, nr);
        attacks |= SquareBB[target];
        if (occ & SquareBB[target]) break;
    }
    // South
    for (int nr = r - 1; nr >= 0; --nr) {
        Square target = make_square(f, nr);
        attacks |= SquareBB[target];
        if (occ & SquareBB[target]) break;
    }
    // East
    for (int nf = f + 1; nf < 8; ++nf) {
        Square target = make_square(nf, r);
        attacks |= SquareBB[target];
        if (occ & SquareBB[target]) break;
    }
    // West
    for (int nf = f - 1; nf >= 0; --nf) {
        Square target = make_square(nf, r);
        attacks |= SquareBB[target];
        if (occ & SquareBB[target]) break;
    }

    return attacks;
}

void print_bitboard(Bitboard bb) {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        std::cout << (r + 1) << " |";
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(f, r);
            std::cout << ((bb & SquareBB[sq]) ? " X |" : " . |");
        }
        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "    a   b   c   d   e   f   g   h\n\n";
}

} // namespace Apex

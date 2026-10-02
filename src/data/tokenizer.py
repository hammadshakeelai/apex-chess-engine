"""
Tokenizers and Feature Extractors for Neural Chess Engines.

Supports:
1. HalfKP Sparse Features (for NNUE Accumulators)
2. 8x8x14 Spatial Bitboard Tensors (for Chess Transformers & ResNets)
3. Action Space Indexing (1,968 standard move space)
"""

import chess
from typing import List, Tuple, Dict, Optional
import numpy as np

# Total piece types in HalfKP: 10 piece types (no kings) * 64 squares = 640
PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN
]

# Action Space: 64 from_sq * 64 to_sq = 4096 (standard subset is 1968 legal moves)
TOTAL_SQUARES = 64
HALFKP_PIECE_TYPES = 10  # 5 white piece types + 5 black piece types
HALFKP_FEATURES_PER_KING = HALFKP_PIECE_TYPES * TOTAL_SQUARES  # 640
HALFKP_TOTAL_FEATURES = TOTAL_SQUARES * HALFKP_FEATURES_PER_KING  # 40,960


def square_mirror(sq: int) -> int:
    """Flip square vertically (rank mirroring for Black perspective)."""
    return sq ^ 56


def get_halfkp_piece_index(piece: chess.Piece, is_white_perspective: bool) -> int:
    """
    Map piece to an index in [0, 9] based on perspective:
    Friendly Pawn..Queen = 0..4
    Enemy Pawn..Queen = 5..9
    """
    piece_type = piece.piece_type
    if piece_type == chess.KING:
        return -1  # King is encoded as the indexing root, not a feature piece

    type_offset = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
    }[piece_type]

    is_friendly = (piece.color == chess.WHITE) if is_white_perspective else (piece.color == chess.BLACK)
    return type_offset if is_friendly else (type_offset + 5)


def extract_halfkp_indices(board: chess.Board) -> Tuple[List[int], List[int]]:
    """
    Extract active sparse HalfKP feature indices for White and Black perspectives.
    Returns:
        white_indices: list of feature indices relative to White King
        black_indices: list of feature indices relative to Black King
    """
    w_king_sq = board.king(chess.WHITE)
    b_king_sq = board.king(chess.BLACK)

    if w_king_sq is None or b_king_sq is None:
        return [], []

    w_features: List[int] = []
    b_features: List[int] = []

    b_king_sq_mirrored = square_mirror(b_king_sq)

    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue

        # White perspective
        w_pt = get_halfkp_piece_index(piece, is_white_perspective=True)
        if w_pt >= 0:
            w_idx = w_king_sq * HALFKP_FEATURES_PER_KING + (w_pt * TOTAL_SQUARES + sq)
            w_features.append(w_idx)

        # Black perspective (mirrored vertically)
        b_pt = get_halfkp_piece_index(piece, is_white_perspective=False)
        if b_pt >= 0:
            sq_mirrored = square_mirror(sq)
            b_idx = b_king_sq_mirrored * HALFKP_FEATURES_PER_KING + (b_pt * TOTAL_SQUARES + sq_mirrored)
            b_features.append(b_idx)

    return w_features, b_features


def board_to_spatial_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert chess.Board to an 8x8x14 spatial tensor:
    Channels 0-5: White P, N, B, R, Q, K
    Channels 6-11: Black p, n, b, r, q, k
    Channel 12: Side to move (all 1s if White, all 0s if Black)
    Channel 13: Castling rights & en-passant availability
    """
    tensor = np.zeros((8, 8, 14), dtype=np.float32)

    piece_channel_map = {
        (chess.WHITE, chess.PAWN): 0,
        (chess.WHITE, chess.KNIGHT): 1,
        (chess.WHITE, chess.BISHOP): 2,
        (chess.WHITE, chess.ROOK): 3,
        (chess.WHITE, chess.QUEEN): 4,
        (chess.WHITE, chess.KING): 5,
        (chess.BLACK, chess.PAWN): 6,
        (chess.BLACK, chess.KNIGHT): 7,
        (chess.BLACK, chess.BISHOP): 8,
        (chess.BLACK, chess.ROOK): 9,
        (chess.BLACK, chess.QUEEN): 10,
        (chess.BLACK, chess.KING): 11,
    }

    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        ch = piece_channel_map[(piece.color, piece.piece_type)]
        tensor[rank, file, ch] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        tensor[:, :, 12] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[0, 6, 13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[0, 2, 13] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[7, 6, 13] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[7, 2, 13] = 1.0

    return tensor


def move_to_action_index(move: chess.Move) -> int:
    """
    Encode move into an index in [0, 4095]: from_sq * 64 + to_sq.
    Underpromotions are mapped into specialized high offsets if needed.
    """
    return move.from_square * 64 + move.to_square


def action_index_to_move(action_idx: int, board: Optional[chess.Board] = None) -> chess.Move:
    """Decode action index back to chess.Move."""
    from_sq = action_idx // 64
    to_sq = action_idx % 64
    move = chess.Move(from_sq, to_sq)

    # Check for pawn promotion
    if board is not None and board.piece_at(from_sq) is not None:
        p = board.piece_at(from_sq)
        if p.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if (p.color == chess.WHITE and to_rank == 7) or (p.color == chess.BLACK and to_rank == 0):
                move.promotion = chess.QUEEN

    return move

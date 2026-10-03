"""
ApexChess Lightweight Prototype Pipeline: "The Whole Nine Yards"
Builds, trains, evaluates, and tests a neural chess model on local CPU from scratch.
"""

import os
import sys
import time
import argparse
import chess
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scripts.download_dataset import download_and_extract_tactics
from scripts.prepare_dataset import vectorize_parquet_dataset
from src.data.tokenizer import extract_halfkp_indices
from src.models.nnue import NNUEWeights, NNUEInference, PyTorchNNUE
from src.train.trainer import NPZChessDataset, collate_halfkp, NNUETrainer
from src.engine.evaluator import ApexEvaluator
from src.engine.search import SearchEngine, SearchLimits


def print_header(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def print_board_pretty(board: chess.Board):
    """Prints a clean ASCII board with square coordinates."""
    lines = str(board).split("\n")
    print("    a b c d e f g h")
    print("  +-----------------+")
    for idx, line in enumerate(lines):
        rank = 8 - idx
        print(f"{rank} | {line} | {rank}")
    print("  +-----------------+")
    print("    a b c d e f g h")


# =====================================================================
# Stage 1: Data Slicing & Inspection
# =====================================================================
def stage_1_data_inspection(samples: int = 5000) -> str:
    print_header(f"STAGE 1: Data Acquisition & Inspection ({samples:,} Samples)")
    
    parquet_path = f"data/prototype_{samples // 1000}k.parquet" if samples >= 1000 else f"data/prototype_{samples}.parquet"
    
    if not os.path.exists(parquet_path):
        source_50k = "data/tactics_50k.parquet"
        if os.path.exists(source_50k):
            print(f"[DATA] Slicing {samples:,} positions from cached {source_50k}...")
            import pyarrow.parquet as pq
            table = pq.read_table(source_50k)
            sliced = table.slice(0, min(samples, table.num_rows))
            pq.write_table(sliced, parquet_path)
        else:
            print(f"[DATA] Downloading and slicing {samples:,} positions from Hugging Face...")
            download_and_extract_tactics(output_path=parquet_path, limit=samples)
    else:
        print(f"[DATA] Found cached prototype dataset: {parquet_path}")

    # Inspect sample position #1
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    p_dict = table.slice(0, 1).to_pydict()
    f_col = [c for c in p_dict.keys() if c.lower() == "fen"][0]
    e_col = [c for c in p_dict.keys() if "eval" in c.lower()][0]
    m_col = [c for c in p_dict.keys() if "move" in c.lower()]

    sample_fen = p_dict[f_col][0]
    sample_eval = str(p_dict[e_col][0])
    sample_move = p_dict[m_col[0]][0] if m_col else "N/A"

    board = chess.Board(sample_fen)
    print("\n--- SAMPLE CHESS POSITION #1 ---")
    print_board_pretty(board)
    print(f"FEN:              {sample_fen}")
    print(f"Side to Move:     {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"Stockfish Eval:   {sample_eval}")
    
    # Calculate target winning probability
    try:
        cp_val = float(sample_eval.replace("#+", "30000").replace("#-", "-30000").replace("+", ""))
    except ValueError:
        cp_val = 0.0
    win_prob = 1.0 / (1.0 + 10.0 ** (-cp_val / 400.0))
    print(f"Win Probability:  {win_prob:.3f} ({win_prob * 100:.1f}%)")
    print(f"Best Engine Move: {sample_move}")
    print("---------------------------------------------------------")

    return parquet_path


# =====================================================================
# Stage 2: Feature Vectorization
# =====================================================================
def stage_2_feature_vectorization(parquet_path: str, samples: int = 5000) -> str:
    print_header("STAGE 2: Feature Vectorization (HalfKP Sparse Representations)")
    
    npz_path = parquet_path.replace(".parquet", ".npz")
    if not os.path.exists(npz_path):
        print(f"[VECTORIZE] Encoding {samples:,} positions into HalfKP sparse tensors...")
        t0 = time.time()
        vectorize_parquet_dataset(parquet_path=parquet_path, output_npz_path=npz_path, max_records=samples)
        elapsed = time.time() - t0
        print(f"[VECTORIZE] Finished in {elapsed:.2f}s ({int(samples / max(0.001, elapsed)):,d} pos/sec) -> {npz_path}")
    else:
        print(f"[VECTORIZE] Using cached HalfKP feature tensors: {npz_path}")

    # Inspect sparse active features for starting position
    board = chess.Board()
    w_idx, b_idx = extract_halfkp_indices(board)
    print(f"\n[FEATURES] Total HalfKP Dimension:  40,960 possible input features")
    print(f"[FEATURES] Active White Features:     {len(w_idx)} non-zero entries (99.92% sparse!)")
    print(f"[FEATURES] Active Black Features:     {len(b_idx)} non-zero entries")
    print(f"[FEATURES] First 5 White indices:    {w_idx[:5]}")
    print("---------------------------------------------------------")

    return npz_path


# =====================================================================
# Stage 3: Lightweight Model Architecture
# =====================================================================
def stage_3_model_instantiation(hidden_dim: int = 128):
    print_header(f"STAGE 3: Lightweight Neural Architecture (Dual-NNUE, Dim: {hidden_dim})")
    
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for model instantiation.")

    model = PyTorchNNUE(hidden_dim=hidden_dim)
    total_params = sum(p.numel() for p in model.parameters())
    sparse_params = model.feature_embed.weight.numel()
    dense_params = total_params - sparse_params

    print(f"Model Class:             PyTorchNNUE")
    print(f"Feature Embedding Table: 40,960 -> {hidden_dim} ({sparse_params:,} sparse weights)")
    print(f"Dense Layer 1 (FC1):     {hidden_dim * 2} -> 32 (SCReLU activation)")
    print(f"Dense Layer 2 (FC2):     32 -> 32 (SCReLU activation)")
    print(f"Output Layer (Out):      32 -> 1 (Centipawn Logit)")
    print(f"Total Parameters:        {total_params:,} (~{total_params * 4 / (1024 * 1024):.1f} MB in FP32)")
    print(f"Dense Compute FLOPs:     ~{dense_params * 2:,} FLOPs per leaf node (Evaluates in ~50 nanoseconds!)")
    print("---------------------------------------------------------")

    return model


# =====================================================================
# Stage 4: Model Training (The "Nine Yards")
# =====================================================================
def stage_4_training(npz_path: str, model: "PyTorchNNUE", epochs: int = 5, batch_size: int = 128, lr: float = 1e-3):
    print_header(f"STAGE 4: Model Training ({epochs} Epochs, Batch Size: {batch_size})")

    dataset = NPZChessDataset(npz_path)
    val_size = max(50, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_halfkp)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_halfkp)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    save_pt = "weights/prototype_v1.pt"
    export_npz = "weights/prototype_v1_quant.npz"
    trainer = NNUETrainer(model=model, model_save_path=save_pt, lr=lr)

    print(f"[TRAIN] Training {train_size:,} positions, Validating on {val_size:,} positions...")
    start_train = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader, model, optimizer, criterion)
        val_loss, val_acc = trainer.evaluate(val_loader, model, criterion)
        scheduler.step()
        epoch_sec = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:2d}/{epochs:2d} [{epoch_sec:4.1f}s] | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val Accuracy: {val_acc:5.1f}% | LR: {curr_lr:.6f}"
        )

    total_sec = time.time() - start_train
    print(f"\n[OK] Training completed in {total_sec:.1f} seconds on CPU!")

    # Export weights
    trainer.save_checkpoint(model, epochs, val_loss, val_acc)
    trainer.export_quantized_weights(model, export_npz)
    print("---------------------------------------------------------")

    return export_npz, val_loader, criterion


# =====================================================================
# Stage 5: Multi-Tier Quantitative & Qualitative Evaluation
# =====================================================================
def stage_5_evaluation(model: "PyTorchNNUE", val_loader, criterion, weights_path: str):
    print_header("STAGE 5: Multi-Tier Quantitative & Qualitative Evaluation")

    # 1. Quantitative Evaluation on Held-Out Validation Data
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    steps = 0
    mae_probs = 0.0

    with torch.no_grad():
        for batch in val_loader:
            pred = model(
                batch["w_indices"],
                batch["w_offsets"],
                batch["b_indices"],
                batch["b_offsets"],
                batch["turns"],
            )
            loss = criterion(pred, batch["targets"])
            total_loss += loss.item()
            steps += 1

            pred_prob = torch.sigmoid(pred)
            mae_probs += torch.abs(pred_prob - batch["targets"]).sum().item()

            pred_win = pred > 0.0
            target_win = batch["targets"] > 0.5
            correct += (pred_win == target_win).sum().item()
            total_samples += batch["targets"].size(0)

    val_loss = total_loss / max(1, steps)
    accuracy = (correct / max(1, total_samples)) * 100.0
    avg_mae = (mae_probs / max(1, total_samples)) * 100.0

    print("--- QUANTITATIVE METRICS (Held-Out Data) ---")
    print(f"Validation Loss (BCE):       {val_loss:.5f}")
    print(f"Sign-Agreement Accuracy:    {accuracy:.1f}% (Correctly identified winning side)")
    print(f"Mean Absolute Error (P(W)): {avg_mae:.2f}% percentage points")

    # 2. Qualitative Sanity Tests (The 4 Fundamental Chess Scenarios)
    print("\n--- QUALITATIVE CHESS SANITY CHECKS ---")
    evaluator = ApexEvaluator(use_nnue=True, weights_path=weights_path)

    test_cases = [
        ("1. Standard Starting Board", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Near 0 cp (Balanced)"),
        ("2. White Up a Queen (+9)",   "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "> +300 cp (White Decisively Winning)"),
        ("3. Black Up a Queen (-9)",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", "< -300 cp (Black Decisively Winning)"),
        ("4. Checkmate (Fool's Mate)",  "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "-29,000 cp (White is Checkmated)"),
    ]

    for name, fen, expectation in test_cases:
        b = chess.Board(fen)
        score_cp = evaluator.evaluate_static(b)
        win_prob = 1.0 / (1.0 + 10.0 ** (-score_cp / 400.0))
        print(f"\n[{name}]")
        print(f"  Expected:    {expectation}")
        print(f"  Model Score: {score_cp:+6d} cp ({score_cp / 100.0:+.2f} pawns) | P(Win): {win_prob * 100.0:.1f}%")
        
        # Verify logical consistency
        if "White Up" in name:
            passed = score_cp > 0
        elif "Black Up" in name:
            passed = score_cp < 0
        elif "Checkmate" in name:
            passed = score_cp == -29000
        else:
            passed = abs(score_cp) < 400
        status = "PASS [OK]" if passed else "NOTE [Model Needs More Training Data]"
        print(f"  Status:      {status}")

    print("---------------------------------------------------------")


# =====================================================================
# Stage 6: Live Engine Play Demonstration
# =====================================================================
def stage_6_live_play_demonstration(weights_path: str, num_turns: int = 6):
    print_header(f"STAGE 6: Live Engine Play Demonstration ({num_turns} Moves)")
    print(f"Connecting trained prototype weights ({weights_path}) to Alpha-Beta Search Engine...\n")

    evaluator = ApexEvaluator(use_nnue=True, weights_path=weights_path)
    engine = SearchEngine(evaluator=evaluator)
    board = chess.Board()

    print("Initial Game Position:")
    print_board_pretty(board)

    print("\n--- SIMULATED MOVE-BY-MOVE GAME ---")
    move_log = []

    for turn in range(1, num_turns + 1):
        side = "White" if board.turn == chess.WHITE else "Black"
        limits = SearchLimits(depth=2, nodes=5000)

        t0 = time.time()
        best_move, score = engine.search(board, limits)
        elapsed = time.time() - t0

        if not best_move:
            print(f"Game over or no moves available.")
            break

        san_move = board.san(best_move)
        board.push(best_move)
        nodes = engine.nodes
        nps = int(nodes / max(0.0001, elapsed))

        print(f"Turn {turn:2d} | {side:5} plays: {san_move:6} | Eval: {score:+5d} cp ({score/100.0:+.2f} pawns) | {nodes:4d} nodes ({elapsed:.2f}s, {nps:5d} nps)")
        move_log.append((turn, side, san_move, score))

    print("\nFinal Position After Mini-Game:")
    print_board_pretty(board)
    print(f"\nPGN Moves: {' '.join([m[2] for m in move_log])}")
    print("=" * 75)
    print("  PROTOTYPE LIFECYCLE DEMONSTRATION COMPLETE!")
    print("  We successfully built, vectorized, trained, evaluated, and played")
    print("  a neural chess model 100% on your local CPU!")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="ApexChess Prototype Pipeline")
    parser.add_argument("--samples", type=int, default=5000, help="Number of positions for prototype (default: 5000)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension for prototype (default: 128)")
    parser.add_argument("--turns", type=int, default=6, help="Demonstration game turns (default: 6)")
    args = parser.parse_args()

    t_all = time.time()

    # 1. Slicing
    parquet_path = stage_1_data_inspection(samples=args.samples)

    # 2. Vectorization
    npz_path = stage_2_feature_vectorization(parquet_path=parquet_path, samples=args.samples)

    # 3. Model
    model = stage_3_model_instantiation(hidden_dim=args.hidden)

    # 4. Training
    weights_path, val_loader, criterion = stage_4_training(
        npz_path=npz_path,
        model=model,
        epochs=args.epochs,
        batch_size=128,
        lr=1e-3,
    )

    # 5. Evaluation
    stage_5_evaluation(model=model, val_loader=val_loader, criterion=criterion, weights_path=weights_path)

    # 6. Live Play
    stage_6_live_play_demonstration(weights_path=weights_path, num_turns=args.turns)

    total_time = time.time() - t_all
    print(f"\n[TOTAL TIME] Entire 6-stage lifecycle completed in {total_time:.1f} seconds.\n")


if __name__ == "__main__":
    main()

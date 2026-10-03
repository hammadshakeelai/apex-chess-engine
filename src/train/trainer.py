"""
PyTorch and NumPy Training Pipeline for ApexChess Neural Models.

Implements:
- Direct NPZ Dataset loading with train/val split.
- Numerically stable BCEWithLogitsLoss on blended sigmoid targets.
- Cosine Annealing learning rate schedule.
- Checkpoint persistence and quantized NumPy/binary export.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models.nnue import NNUEWeights
if HAS_TORCH:
    from src.models.nnue import PyTorchNNUE
else:
    PyTorchNNUE = None


if HAS_TORCH:
    class NPZChessDataset(Dataset):
        """Torch Dataset for pre-vectorized HalfKP sparse indices and target evals from NPZ."""

        def __init__(self, npz_path: str):
            print(f"[DATA] Loading dataset from {npz_path}...")
            data = np.load(npz_path, allow_pickle=True)
            self.w_indices = data["halfkp_w"]
            self.b_indices = data["halfkp_b"]
            self.turns = data["turns"]
            self.targets = data["targets"]
            self.evals = data.get("evals", np.zeros_like(self.targets))
            self.best_moves = data.get("best_moves", np.full_like(self.targets, -1, dtype=np.int32))
            self.length = len(self.targets)
            print(f"[DATA] Loaded {self.length:,} records from {npz_path}.")

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            w_idx = self.w_indices[idx]
            b_idx = self.b_indices[idx]
            turn = bool(self.turns[idx])
            target = float(self.targets[idx])
            move = int(self.best_moves[idx])

            return {
                "w_idx": torch.tensor(w_idx, dtype=torch.long),
                "b_idx": torch.tensor(b_idx, dtype=torch.long),
                "turn": torch.tensor([turn], dtype=torch.bool),
                "target": torch.tensor([target], dtype=torch.float32),
                "best_move": torch.tensor(move, dtype=torch.long),
            }

    def collate_halfkp(batch):
        """Collate variable-length HalfKP sparse feature indices for EmbeddingBag."""
        w_indices = []
        w_offsets = [0]
        b_indices = []
        b_offsets = [0]
        turns = []
        targets = []
        moves = []

        for item in batch:
            w_idx = item["w_idx"]
            b_idx = item["b_idx"]

            w_indices.append(w_idx)
            w_offsets.append(w_offsets[-1] + len(w_idx))

            b_indices.append(b_idx)
            b_offsets.append(b_offsets[-1] + len(b_idx))

            turns.append(item["turn"])
            targets.append(item["target"])
            moves.append(item["best_move"])

        return {
            "w_indices": torch.cat(w_indices) if w_indices else torch.empty(0, dtype=torch.long),
            "w_offsets": torch.tensor(w_offsets[:-1], dtype=torch.long),
            "b_indices": torch.cat(b_indices) if b_indices else torch.empty(0, dtype=torch.long),
            "b_offsets": torch.tensor(b_offsets[:-1], dtype=torch.long),
            "turns": torch.stack(turns),
            "targets": torch.stack(targets),
            "moves": torch.stack(moves),
        }


class NNUETrainer:
    """Training manager for NNUE."""

    def __init__(
        self,
        model: Optional["PyTorchNNUE"] = None,
        model_save_path: str = "weights/apex_v1.pt",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model_save_path = model_save_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = model

    def train_epoch(self, dataloader, model, optimizer, criterion) -> float:
        if not HAS_TORCH:
            print("[NNUETrainer] PyTorch is required for backpropagation training.")
            return 0.0

        model.train()
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            optimizer.zero_grad()
            # Raw output from model is in logits scale (pred corresponds to eval / 400.0)
            pred = model(
                batch["w_indices"],
                batch["w_offsets"],
                batch["b_indices"],
                batch["b_offsets"],
                batch["turns"],
            )

            # BCE with logits on target probability in [0, 1]
            loss = criterion(pred, batch["targets"])
            loss.backward()

            # Gradient clipping to prevent exploding updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            steps += 1

        return total_loss / max(1, steps)

    def evaluate(self, dataloader, model, criterion) -> Tuple[float, float]:
        """Evaluates loss and accuracy (sign agreement with winning probability)."""
        if not HAS_TORCH:
            return 0.0, 0.0

        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        steps = 0

        with torch.no_grad():
            for batch in dataloader:
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

                # Sign agreement: predicted > 0 corresponds to win prob > 0.5
                pred_win = pred > 0.0
                target_win = batch["targets"] > 0.5
                correct += (pred_win == target_win).sum().item()
                total_samples += batch["targets"].size(0)

        val_loss = total_loss / max(1, steps)
        accuracy = (correct / max(1, total_samples)) * 100.0
        return val_loss, accuracy

    def save_checkpoint(self, model, epoch: int, loss: float, val_acc: float = 0.0):
        if not HAS_TORCH:
            return
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "val_acc": val_acc,
            "state_dict": model.state_dict(),
        }, self.model_save_path)
        print(f"[CHECKPOINT] Saved PyTorch model to {self.model_save_path}")

    @staticmethod
    def export_quantized_weights(model: "PyTorchNNUE", output_path: str):
        """
        Quantizes FP32 trained weights into standard integer weights and exports
        to a portable .npz or binary format usable by the C++/NumPy engine evaluator.
        """
        if not HAS_TORCH:
            return

        model.eval()
        state = model.state_dict()

        # Extract weights
        feature_weights = state["feature_embed.weight"].cpu().numpy() # [40960, 256]
        feature_bias = state["feature_bias"].cpu().numpy()            # [256]
        fc1_w = state["fc1.weight"].cpu().numpy()                     # [32, 512]
        fc1_b = state["fc1.bias"].cpu().numpy()                       # [32]
        fc2_w = state["fc2.weight"].cpu().numpy()                     # [32, 32]
        fc2_b = state["fc2.bias"].cpu().numpy()                       # [32]
        out_w = state["out.weight"].cpu().numpy()                     # [1, 32]
        out_b = state["out.bias"].cpu().numpy()                       # [1]

        # Quantize Feature Transformer (Scale = 255.0 -> INT16)
        SCALE_FEATURE = 255.0
        w_feat_quant = np.clip(np.round(feature_weights * SCALE_FEATURE), -32768, 32767).astype(np.int16)
        b_feat_quant = np.clip(np.round(feature_bias * SCALE_FEATURE), -32768, 32767).astype(np.int16)

        # Quantize Dense Layers (Scale = 64.0 -> INT8)
        SCALE_DENSE = 64.0
        w_fc1_quant = np.clip(np.round(fc1_w * SCALE_DENSE), -128, 127).astype(np.int8)
        b_fc1_quant = np.round(fc1_b * SCALE_DENSE * SCALE_FEATURE).astype(np.int32)

        w_fc2_quant = np.clip(np.round(fc2_w * SCALE_DENSE), -128, 127).astype(np.int8)
        b_fc2_quant = np.round(fc2_b * SCALE_DENSE).astype(np.int32)

        w_out_quant = (out_w * 400.0).astype(np.float32)
        b_out_quant = (out_b * 400.0).astype(np.float32)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(
            output_path,
            # FP32 weights for native high-precision evaluation
            feature_weights=feature_weights,
            feature_bias=feature_bias,
            fc1_w=fc1_w.T,
            fc1_b=fc1_b,
            fc2_w=fc2_w.T,
            fc2_b=fc2_b,
            out_w=out_w.T,
            out_b=out_b,
            # Quantized integer weights for AVX2 / SIMD inference
            w_feat=w_feat_quant,
            b_feat=b_feat_quant,
            w_fc1=w_fc1_quant,
            b_fc1=b_fc1_quant,
            w_fc2=w_fc2_quant,
            b_fc2=b_fc2_quant,
            w_out=w_out_quant,
            b_out=b_out_quant,
        )

        size_kb = os.path.getsize(output_path) / 1024
        print(f"[EXPORT] Quantized NNUE weights successfully exported to {output_path} ({size_kb:.1f} KB)")


def run_training_pipeline(
    data_path: str,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    save_pt: str = "weights/apex_v1.pt",
    export_npz: str = "weights/apex_v1_quant.npz",
):
    if not HAS_TORCH:
        print("[ERROR] PyTorch is required to run the neural training pipeline.")
        return

    # 1. Dataset & Split
    dataset = NPZChessDataset(data_path)
    val_size = max(100, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_halfkp)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_halfkp)

    # 2. Model & Optimizer
    model = PyTorchNNUE()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    trainer = NNUETrainer(model=model, model_save_path=save_pt, lr=lr)

    print(f"\n[TRAIN] Beginning training for {epochs} epochs (Train: {train_size:,} | Val: {val_size:,})...")
    print(f"        Batch Size: {batch_size} | Initial LR: {lr}")

    best_val_loss = float("inf")
    start_all = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader, model, optimizer, criterion)
        val_loss, val_acc = trainer.evaluate(val_loader, model, criterion)
        scheduler.step()
        epoch_time = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:2d}/{epochs:2d} [{epoch_time:4.1f}s] | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val Acc: {val_acc:5.1f}% | LR: {curr_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(model, epoch, val_loss, val_acc)

    total_time = time.time() - start_all
    print(f"\n[OK] Training completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.5f}")

    # 3. Export Quantized Weights
    trainer.export_quantized_weights(model, export_npz)


def main():
    parser = argparse.ArgumentParser(description="ApexChess PyTorch Neural Training Pipeline")
    parser.add_argument("--data", type=str, default="data/train_halfkp_50k.npz", help="Path to pre-vectorized .npz dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save", type=str, default="weights/apex_v1.pt", help="Path to save PyTorch checkpoint")
    parser.add_argument("--export", type=str, default="weights/apex_v1_quant.npz", help="Path to export quantized weights")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_pt=args.save,
        export_npz=args.export,
    )


if __name__ == "__main__":
    main()

"""
One-Click Cloud GPU Training Launcher for Google Colab / Kaggle.
Enables training ApexChess NNUE models on free NVIDIA T4/A100 GPUs (16GB-40GB VRAM).

Usage in Colab / Kaggle:
    !git clone https://github.com/hammadshakeelai/apex-chess-engine.git
    %cd apex-chess-engine
    !pip install -r requirements.txt
    !python scripts/cloud_train_colab.py --positions 500000 --epochs 15
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scripts.download_dataset import download_and_extract_tactics
from scripts.prepare_dataset import vectorize_parquet_dataset
from src.train.trainer import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="ApexChess Cloud GPU Training Pipeline")
    parser.add_argument("--positions", type=int, default=100000, help="Number of positions to train on (e.g. 100000, 500000, 1000000)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default 10)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (default 1024 for GPU)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default 1e-3)")
    parser.add_argument("--save", type=str, default="weights/apex_cloud_v1.pt", help="Path to save PyTorch weights")
    parser.add_argument("--export", type=str, default="weights/apex_cloud_v1_quant.npz", help="Path to export quantized weights")
    args = parser.parse_args()

    print("=" * 70)
    print("  ApexChess Cloud GPU Training Pipeline")
    print("=" * 70)

    # 1. Hardware Detection
    if HAS_TORCH and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[HARDWARE] Detected NVIDIA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("[HARDWARE] No GPU detected. Falling back to multi-core CPU.")

    os.makedirs("data", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    parquet_path = f"data/tactics_{args.positions // 1000}k.parquet" if args.positions >= 1000 else f"data/tactics_{args.positions}.parquet"
    npz_path = parquet_path.replace(".parquet", ".npz")

    # 2. Acquire Dataset
    if not os.path.exists(parquet_path):
        print(f"\n[STEP 1/3] Downloading and slicing {args.positions:,} positions...")
        download_and_extract_tactics(output_path=parquet_path, limit=args.positions)
    else:
        print(f"\n[STEP 1/3] Using cached Parquet dataset: {parquet_path}")

    # 3. Vectorize Features
    if not os.path.exists(npz_path):
        print(f"\n[STEP 2/3] Vectorizing HalfKP sparse feature tensors...")
        vectorize_parquet_dataset(parquet_path=parquet_path, output_npz_path=npz_path, max_records=args.positions)
    else:
        print(f"\n[STEP 2/3] Using cached NPZ feature tensors: {npz_path}")

    # 4. Train with CUDA Acceleration
    print(f"\n[STEP 3/3] Training NNUE on {args.positions:,} positions ({args.epochs} epochs, batch size {args.batch_size})...")
    t0 = time.time()
    run_training_pipeline(
        data_path=npz_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_pt=args.save,
        export_npz=args.export,
    )
    elapsed = time.time() - t0

    print("=" * 70)
    print(f"  CLOUD TRAINING COMPLETE in {elapsed:.1f} seconds!")
    print(f"  Exported Weights: {args.export}")
    print(f"  PyTorch Checkpoint: {args.save}")
    print("=" * 70)

    # 5. Colab Direct Download Helper
    try:
        from google.colab import files
        print("[COLAB] Downloading weights to local machine...")
        files.download(args.export)
    except ImportError:
        pass


if __name__ == "__main__":
    main()

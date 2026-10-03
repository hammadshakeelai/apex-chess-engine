"""
ApexChess NNUE C++ Binary Weight Exporter
Converts .npz quantized/float weights into an exact binary layout for C++ AVX2 SIMD inference.
"""

import os
import sys
import struct
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nnue import NNUEWeights

MAGIC = b"APEXNNUE"
VERSION = 2


def export_npz_to_bin(npz_path: str, bin_path: str):
    print("=" * 65)
    print(f"  EXPORTING NNUE TO C++ BINARY FORMAT (V2)")
    print(f"  Source: {npz_path}")
    print(f"  Target: {bin_path}")
    print("=" * 65)

    weights = NNUEWeights.from_file(npz_path)

    # 1. Feature Transformer: [40960, 256] int16
    w_feat = np.clip(np.round(weights.feature_weights * 255.0), -32768, 32767).astype(np.int16)
    b_feat = np.clip(np.round(weights.feature_bias * 255.0), -32768, 32767).astype(np.int16)

    # 2. Dense Layer 1: weights.w1 is [512, 32] float32
    w1 = weights.w1.astype(np.float32)
    b1 = weights.b1.astype(np.float32)

    # 3. Dense Layer 2: weights.w2 is [32, 32] float32
    w2 = weights.w2.astype(np.float32)
    b2 = weights.b2.astype(np.float32)

    # 4. Output Layer: weights.w_out is [32, 1] -> [32] float32, b_out is float32
    w_out = weights.w_out.flatten().astype(np.float32)
    b_out = float(weights.b_out.flatten()[0])

    feature_dim, hidden_dim = w_feat.shape
    print(f"[SHAPES] Feature Weights: [{feature_dim}, {hidden_dim}] int16")
    print(f"[SHAPES] Dense 1 Weights: {w1.shape} float32 | Biases: {b1.shape} float32")
    print(f"[SHAPES] Dense 2 Weights: {w2.shape} float32 | Biases: {b2.shape} float32")
    print(f"[SHAPES] Output Weights:  {w_out.shape} float32 | Bias: {b_out:.4f}")

    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    with open(bin_path, "wb") as f:
        # Header: MAGIC (8B), VERSION (4B), FEATURE_DIM (4B), HIDDEN_DIM (4B)
        header = struct.pack("<8sIII", MAGIC, VERSION, feature_dim, hidden_dim)
        f.write(header)

        # Contiguous binary payloads
        f.write(w_feat.tobytes())
        f.write(b_feat.tobytes())
        f.write(w1.tobytes())
        f.write(b1.tobytes())
        f.write(w2.tobytes())
        f.write(b2.tobytes())
        f.write(w_out.tobytes())
        f.write(struct.pack("<f", b_out))

    size_mb = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"[OK] Successfully packed NNUE weights to {bin_path} ({size_mb:.2f} MB)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Export NNUE weights to C++ binary format")
    parser.add_argument("--input", type=str, default="weights/apex_cloud_1m_quant.npz", help="Input .npz weights")
    parser.add_argument("--output", type=str, default="weights/apex_nnue.bin", help="Output .bin file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    export_npz_to_bin(args.input, args.output)


if __name__ == "__main__":
    main()

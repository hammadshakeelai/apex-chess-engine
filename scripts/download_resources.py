"""
Automated Downloader for Pretrained Model Weights and Research Papers.

Supported downloads:
1. Stockfish Official NNUE Weights (Small Net & Big Net)
2. Seminal Research Papers from arXiv (PDF & metadata)
3. Lc0 & Maia Model Checkpoints
"""

import os
import sys
import urllib.request
import json

# Official Pretrained Weights Registry
WEIGHTS_REGISTRY = {
    "sf17_small": {
        "name": "Stockfish 17 Small Net (~1.5 MB)",
        "url": "https://tests.stockfishchess.org/api/nn/nn-baff1ede1f0f.nnue",
        "filename": "models/sf17_small.nnue",
    },
    "sf17_big": {
        "name": "Stockfish 17 Big Net (~60 MB)",
        "url": "https://tests.stockfishchess.org/api/nn/nn-b1a57edbea57.nnue",
        "filename": "models/sf17_big.nnue",
    },
    "sf16_default": {
        "name": "Stockfish 16 Default Net (~45 MB)",
        "url": "https://tests.stockfishchess.org/api/nn/nn-e8bac1c074c4.nnue",
        "filename": "models/sf16_default.nnue",
    },
    "maia_1500": {
        "name": "Maia Chess 1500 Elo Model (~15 MB)",
        "url": "https://github.com/CSSLab/maia-chess/raw/master/model_files/maia-1500.pb.gz",
        "filename": "models/maia-1500.pb.gz",
    }
}

# Curated Research Papers on arXiv
PAPERS_REGISTRY = {
    "searchless_chess": {
        "title": "Amortized Planning with Large-Scale Transformers (DeepMind 2024)",
        "arxiv_id": "2402.04494",
        "pdf_url": "https://arxiv.org/pdf/2402.04494",
        "filename": "papers/deepmind_searchless_chess_2402.04494.pdf",
    },
    "efficient_selfplay": {
        "title": "Engineering Efficient Self-Play Chess: Search, Replay & Throughput (2026)",
        "arxiv_id": "2609.37447",
        "pdf_url": "https://arxiv.org/pdf/2609.37447",
        "filename": "papers/efficient_selfplay_2609.37447.pdf",
    },
    "alphazero": {
        "title": "Mastering Chess and Shogi by Self-Play with General RL (DeepMind 2017)",
        "arxiv_id": "1712.01815",
        "pdf_url": "https://arxiv.org/pdf/1712.01815",
        "filename": "papers/alphazero_1712.01815.pdf",
    },
    "maia_chess": {
        "title": "Aligning Superhuman AI with Human Behavior: Chess as a Model System (NeurIPS 2020)",
        "arxiv_id": "2006.01855",
        "pdf_url": "https://arxiv.org/pdf/2006.01855",
        "filename": "papers/maia_chess_2006.01855.pdf",
    }
}


def download_file(url: str, dest_path: str):
    """Download a file with browser headers and progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[EXISTS] {dest_path} already exists. Skipping download.")
        return

    print(f"[DOWNLOADING] {url} -> {dest_path} ...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
        print(f"[OK] Successfully saved to {dest_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")


def download_all_papers():
    """Download all curated research papers in PDF format."""
    print("=== Downloading Seminal Research Papers ===")
    for key, info in PAPERS_REGISTRY.items():
        print(f"\nProcessing: {info['title']}")
        download_file(info["pdf_url"], info["filename"])


def download_all_weights():
    """Download representative open-source neural weights."""
    print("=== Downloading Pretrained Model Weights ===")
    for key, info in WEIGHTS_REGISTRY.items():
        print(f"\nProcessing: {info['name']}")
        download_file(info["url"], info["filename"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("papers", "all"):
        download_all_papers()
    if mode in ("weights", "all"):
        download_all_weights()

"""
ApexChess Dataset Downloader & Preprocessing Pipeline
Downloads and prepares evaluated chess positions from Hugging Face (ssingh22/chess-evaluations).
"""

import os
import sys
import argparse
import urllib.request
import time
import re
import pandas as pd
import pyarrow.parquet as pq


TACTICS_PARQUET_URL = (
    "https://huggingface.co/datasets/ssingh22/chess-evaluations/resolve/main/"
    "tactics/train-00000-of-00001-87e7d058f638f8f3.parquet"
)


def download_with_progress(url: str, output_path: str):
    """Download a remote file with progress reporting."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[CACHE] Found existing file at {output_path} ({size_mb:.2f} MB)")
        return

    print(f"[DOWNLOAD] Fetching from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ApexChess/1.0)"})
    
    start_time = time.time()
    with urllib.request.urlopen(req) as resp, open(output_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunk

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r[DOWNLOAD] {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[OK] Downloaded successfully in {elapsed:.1f}s -> {output_path}")


def parse_evaluation(eval_str) -> tuple[float, float]:
    """
    Parse evaluation string (e.g. '+248', '-50', '#+2', '#-1') into:
    (centipawn_eval, win_probability_wdl)
    """
    if eval_str is None or pd.isna(eval_str):
        return 0.0, 0.5

    s = str(eval_str).strip()

    # Mate score handling: '#+3' or '#-1' or '#0'
    if s.startswith("#") or "#" in s:
        # Check if mate for white or black
        if "-" in s:
            return -30000.0, 0.0
        else:
            return 30000.0, 1.0

    try:
        cp = float(s.replace("+", ""))
        # Clamp extreme values
        cp = max(-30000.0, min(30000.0, cp))
        # Logistic win probability W = 1 / (1 + 10^(-cp / 400))
        wdl = 1.0 / (1.0 + 10.0 ** (-cp / 400.0))
        return cp, wdl
    except (ValueError, TypeError):
        return 0.0, 0.5


def clean_and_slice_dataset(
    raw_parquet_path: str,
    output_path: str,
    limit: int = 100000
) -> pd.DataFrame:
    """
    Reads Parquet dataset, parses evaluations, filters valid positions,
    and saves clean dataset.
    """
    print(f"[PROCESS] Reading raw parquet from {raw_parquet_path}...")
    table = pq.read_table(raw_parquet_path)
    df = table.to_pandas()
    print(f"[INFO] Raw dataset contains {len(df):,} total positions.")

    if limit and limit < len(df):
        print(f"[SLICE] Selecting first {limit:,} positions for fast prototyping...")
        df = df.iloc[:limit].copy()
    else:
        df = df.copy()

    # Standardize columns: expected 'FEN', 'Evaluation', 'Move'
    col_map = {c: c.lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    print("[PROCESS] Parsing evaluations and computing win probabilities...")
    parsed = [parse_evaluation(ev) for ev in df["evaluation"]]
    df["eval_cp"] = [p[0] for p in parsed]
    df["wdl_target"] = [p[1] for p in parsed]

    # Side to move extraction from FEN (second field in FEN)
    def extract_stm(fen):
        parts = str(fen).split()
        return parts[1] if len(parts) > 1 else "w"

    df["turn"] = df["fen"].apply(extract_stm)
    
    # Side-to-move win probability: from the perspective of the player to move
    df["stm_wdl"] = df.apply(
        lambda row: row["wdl_target"] if row["turn"] == "w" else (1.0 - row["wdl_target"]),
        axis=1
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[OK] Clean dataset saved to {output_path} with {len(df):,} positions.")
    print(f"[SAMPLE] FEN: {df['fen'].iloc[0]}")
    print(f"         Eval: {df['eval_cp'].iloc[0]} cp | WDL: {df['wdl_target'].iloc[0]:.4f} | Move: {df.get('move', pd.Series(['?'])).iloc[0]}")

    return df


def download_and_extract_tactics(output_path: str, limit: int = 100000, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Download tactics parquet if missing and slice clean records."""
    os.makedirs(raw_dir, exist_ok=True)
    raw_file = os.path.join(raw_dir, "tactics_ssingh22.parquet")
    download_with_progress(TACTICS_PARQUET_URL, raw_file)
    lim = None if limit <= 0 else limit
    return clean_and_slice_dataset(raw_file, output_path, limit=lim)

    parser = argparse.ArgumentParser(description="ApexChess Dataset Acquisition Pipeline")
    parser.add_argument("--limit", type=int, default=100000, help="Number of positions to slice (default 100k, 0 for all 2.6M)")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory for raw downloaded parquet")
    parser.add_argument("--output", type=str, default="data/tactics_clean.parquet", help="Path for clean processed parquet")
    args = parser.parse_args()

    raw_file = os.path.join(args.raw_dir, "tactics_ssingh22.parquet")
    download_with_progress(TACTICS_PARQUET_URL, raw_file)

    limit = None if args.limit <= 0 else args.limit
    clean_and_slice_dataset(raw_file, args.output, limit=limit)


if __name__ == "__main__":
    main()

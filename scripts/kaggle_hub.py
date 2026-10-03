"""
ApexChess Kaggle Cloud GPU Automation Hub
Manages remote GPU training jobs, status monitoring, log streaming,
and weight synchronization via the official Kaggle CLI.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KAGGLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kaggle")
METADATA_FILE = os.path.join(KAGGLE_DIR, "kernel-metadata.json")


def get_kaggle_executable() -> str:
    """Finds the kaggle executable in Anaconda or system PATH."""
    python_dir = os.path.dirname(sys.executable)
    # Check Scripts/kaggle.exe (Windows Anaconda)
    candidate_win = os.path.join(python_dir, "Scripts", "kaggle.exe")
    if os.path.exists(candidate_win):
        return candidate_win
    # Fallback to sys.executable -m kaggle
    return f'"{sys.executable}" -m kaggle'


def run_kaggle_command(cmd_args: list[str]) -> subprocess.CompletedProcess:
    """Executes a Kaggle CLI command safely."""
    kaggle_exe = get_kaggle_executable()
    if kaggle_exe.startswith('"'):
        # Module invocation: python -m kaggle ...
        cmd = [sys.executable, "-m", "kaggle"] + cmd_args
    else:
        cmd = [kaggle_exe] + cmd_args

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    res.stdout = res.stdout.encode("ascii", errors="replace").decode("ascii")
    res.stderr = res.stderr.encode("ascii", errors="replace").decode("ascii")
    return res


def load_metadata() -> Dict[str, Any]:
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Kaggle metadata file not found at {METADATA_FILE}")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def check_auth() -> Optional[str]:
    """Verifies Kaggle credentials and API connectivity."""
    print("=" * 70)
    print("  KAGGLE CLI AUTHENTICATION & ENVIRONMENT CHECK")
    print("=" * 70)

    token_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(token_path):
        print(f"[ERROR] Kaggle token not found at {token_path}")
        print("        Please place your kaggle.json file in ~/.kaggle/kaggle.json")
        return None

    try:
        with open(token_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            username = data.get("username", "Unknown")
        print(f"[OK] Token Found: {token_path} (User: {username})")
    except Exception as e:
        print(f"[ERROR] Failed to read kaggle.json: {e}")
        return None

    # Test API connection
    print("[API] Pinging Kaggle API...")
    res = run_kaggle_command(["datasets", "list", "--max-size", "100", "-v"])
    if res.returncode == 0:
        print("[OK] Successfully authenticated with Kaggle Cloud API!")
    else:
        print(f"[WARN] API check returned code {res.returncode}: {res.stderr.strip()}")

    # Check local kernel metadata
    try:
        meta = load_metadata()
        print(f"[METADATA] Kernel ID:       {meta.get('id')}")
        print(f"[METADATA] Accelerator:     {'GPU Enabled (T4/P100)' if meta.get('enable_gpu') else 'CPU Only'}")
        print(f"[METADATA] Cloud Internet:  {'Enabled' if meta.get('enable_internet') else 'Disabled'}")
    except Exception as e:
        print(f"[WARN] Metadata check: {e}")

    print("=" * 70)
    return username


def push_kernel():
    """Pushes the kernel notebook to Kaggle to trigger GPU execution."""
    print("=" * 70)
    print("  PUSHING APEX CHESS KERNEL TO KAGGLE CLOUD GPU")
    print("=" * 70)

    meta = load_metadata()
    kernel_id = meta.get("id")
    print(f"[PUSH] Target Kernel: {kernel_id}")
    print(f"[PUSH] Directory:     {KAGGLE_DIR}")

    res = run_kaggle_command(["kernels", "push", "-p", KAGGLE_DIR])
    print(res.stdout.strip())
    if res.stderr and "Warning" not in res.stderr:
        print(f"[STDERR] {res.stderr.strip()}")

    if res.returncode == 0:
        print("\n[OK] Kernel pushed successfully!")
        print(f"     Dashboard: https://www.kaggle.com/code/{kernel_id}")
        print(f"     Track status locally: python scripts/kaggle_hub.py --status")
    else:
        print(f"\n[ERROR] Failed to push kernel (code {res.returncode})")

    print("=" * 70)


def check_status(kernel_id: Optional[str] = None):
    """Checks the status of the remote Kaggle job."""
    if not kernel_id:
        meta = load_metadata()
        kernel_id = meta.get("id")

    print(f"[STATUS] Checking Kaggle kernel: {kernel_id}...")
    res = run_kaggle_command(["kernels", "status", kernel_id])
    output = res.stdout.strip()
    print(f"\n  Current Cloud Status: {output}")
    if "complete" in output.lower():
        print("  --> Job Finished! You can pull weights via: python scripts/kaggle_hub.py --pull\n")
    elif "running" in output.lower():
        print("  --> Job is actively running on Kaggle GPU! Check back in a few minutes.\n")
    elif "queued" in output.lower():
        print("  --> Job is queued waiting for an available Kaggle GPU.\n")
    else:
        if res.stderr and "Warning" not in res.stderr:
            print(f"  --> Notice: {res.stderr.strip()}")


def pull_outputs(kernel_id: Optional[str] = None, dest_dir: str = "weights"):
    """Downloads the trained model weights from the completed Kaggle run."""
    if not kernel_id:
        meta = load_metadata()
        kernel_id = meta.get("id")

    os.makedirs(dest_dir, exist_ok=True)
    print("=" * 70)
    print(f"  PULLING TRAINED WEIGHTS FROM KAGGLE ({kernel_id})")
    print(f"  Target Destination: {dest_dir}/")
    print("=" * 70)

    res = run_kaggle_command(["kernels", "output", kernel_id, "-p", dest_dir])
    lines = [l for l in res.stdout.strip().splitlines() if any(k in l for k in [".pt", ".npz", ".nnue", "apex", "log"])]
    for l in lines[:10]:
        print(f"  --> {l}")
    if res.stderr and "Warning" not in res.stderr:
        print(res.stderr.strip())

    if res.returncode == 0:
        print(f"\n[OK] Weights successfully pulled to {dest_dir}/!")
        print("     Local Weights Directory Contents:")
        for f in os.listdir(dest_dir):
            p = os.path.join(dest_dir, f)
            if os.path.isfile(p):
                size_mb = os.path.getsize(p) / (1024 * 1024)
                print(f"     - {f:30} ({size_mb:.2f} MB)")
    else:
        print(f"\n[ERROR] Failed to pull outputs (code {res.returncode})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ApexChess Kaggle Cloud GPU Automation Hub")
    parser.add_argument("--check", action="store_true", help="Check Kaggle credentials and API connection")
    parser.add_argument("--push", action="store_true", help="Push kernel and start cloud GPU training")
    parser.add_argument("--status", action="store_true", help="Check remote cloud training status")
    parser.add_argument("--pull", action="store_true", help="Download trained model weights to local weights/ directory")
    parser.add_argument("--kernel", type=str, default=None, help="Custom Kaggle kernel ID (e.g. username/slug)")
    args = parser.parse_args()

    # Default to --check if no actions specified
    if not (args.check or args.push or args.status or args.pull):
        check_auth()
        print("\nCommands available:")
        print("  python scripts/kaggle_hub.py --push    # Push & start cloud GPU training")
        print("  python scripts/kaggle_hub.py --status  # Check if training is running/complete")
        print("  python scripts/kaggle_hub.py --pull    # Download weights when finished")
        return

    if args.check:
        check_auth()
    if args.push:
        push_kernel()
    if args.status:
        check_status(args.kernel)
    if args.pull:
        pull_outputs(args.kernel)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "multicontrol_baseline_200.h5"))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "python_impl" / "python_scripts" / "train_anc_multi_control_canonical_q.py"),
        "--h5-path",
        str(args.h5_path),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--device",
        str(args.device),
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

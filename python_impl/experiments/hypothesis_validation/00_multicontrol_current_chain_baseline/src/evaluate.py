from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "multicontrol_baseline_200.h5"))
    parser.add_argument("--checkpoint-path", type=str, default=str(Path(__file__).resolve().parents[1] / "results" / "best_replay.pt"))
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "results" / "eval"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "python_impl" / "python_scripts" / "evaluate_anc_multi_control_canonical_q.py"),
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--h5-path",
        str(args.h5_path),
        "--output-dir",
        str(args.output_dir),
        "--device",
        str(args.device),
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

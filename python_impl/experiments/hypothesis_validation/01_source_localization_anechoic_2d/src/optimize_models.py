from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_geometry_pipeline import (
    evaluate_stage01_stable_candidates,
    train_stage01_stable_models,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--live-plot", action="store_true")
    parser.add_argument("--candidate-id", action="append", default=None)
    args = parser.parse_args()

    train_summary = train_stage01_stable_models(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device=str(args.device),
        seed=int(args.seed),
        live_plot=bool(args.live_plot),
        candidate_ids=list(args.candidate_id or []),
    )
    print(train_summary)
    eval_summary = evaluate_stage01_stable_candidates(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        device=str(args.device),
    )
    print(eval_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

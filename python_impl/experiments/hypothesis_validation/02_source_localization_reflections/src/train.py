from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.hypothesis_validation_common import resolve_stage_level, train_localization_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--level", type=str, default="L1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--live-plot", action="store_true")
    args = parser.parse_args()
    stage_cfg = resolve_stage_level("02", args.level, epochs=args.epochs)
    h5_path = args.h5_path or str(
        Path(__file__).resolve().parents[1]
        / "data"
        / (
            f"source_localization_single_reflection_{stage_cfg['level'].lower()}_stable_v3_w2.h5"
            if str(stage_cfg["level"]).upper() == "L1"
            else f"source_localization_single_reflection_{stage_cfg['level'].lower()}_stable_v3_w2_50k.h5"
        )
    )
    output_dir = args.output_dir or str(Path(__file__).resolve().parents[1] / "results" / stage_cfg["level"])
    print(
        train_localization_suite(
            h5_path,
            output_dir,
            epochs=int(stage_cfg["epochs"]),
            batch_size=int(args.batch_size),
            device=str(args.device),
            live_plot=bool(args.live_plot),
            stage_id="02",
            level=str(stage_cfg["level"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

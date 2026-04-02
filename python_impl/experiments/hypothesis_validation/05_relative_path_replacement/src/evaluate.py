from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.hypothesis_validation_common import evaluate_single_control_suite, resolve_stage_level


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--level", type=str, default="L1")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--feature-kind", choices=("relative", "relative_plus_r2r"), default="relative")
    parser.add_argument("--reference-summary", type=str, default=str(ROOT / "python_impl" / "experiments" / "hypothesis_validation" / "03_single_control_hyperplane_anechoic" / "results" / "L1" / "summary.json"))
    args = parser.parse_args()
    stage_cfg = resolve_stage_level("05", args.level)
    h5_path = args.h5_path or str(Path(__file__).resolve().parents[1] / "data" / f"single_control_relative_anechoic_{stage_cfg['level'].lower()}.h5")
    output_dir = args.output_dir or str(Path(__file__).resolve().parents[1] / "results" / stage_cfg["level"] / str(args.feature_kind))
    print(
        evaluate_single_control_suite(
            h5_path,
            output_dir,
            feature_kind=str(args.feature_kind),
            reference_summary_path=args.reference_summary,
            device=str(args.device),
            stage_id="05",
            level=str(stage_cfg["level"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

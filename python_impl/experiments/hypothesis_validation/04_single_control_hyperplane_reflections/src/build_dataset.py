from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.hypothesis_validation_common import SingleControlValidationConfig, build_single_control_dataset, resolve_stage_level


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-h5", type=str, default=None)
    parser.add_argument("--level", type=str, default="L1")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260401)
    args = parser.parse_args()
    stage_cfg = resolve_stage_level("04", args.level, num_samples=args.num_samples)
    output_h5 = args.output_h5 or str(Path(__file__).resolve().parents[1] / "data" / f"single_control_single_reflection_{stage_cfg['level'].lower()}.h5")
    cfg = SingleControlValidationConfig(num_samples=int(stage_cfg["num_samples"]), seed=int(args.seed), profile="single_reflection")
    print(build_single_control_dataset(output_h5, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

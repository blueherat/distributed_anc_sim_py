from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.hypothesis_validation_common import subset_multicontrol_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-h5", type=str, default=str(ROOT / "python_impl" / "python_scripts" / "cfxlms_qc_dataset_multicontrol.h5"))
    parser.add_argument("--dst-h5", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "multicontrol_baseline_200.h5"))
    parser.add_argument("--num-rooms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260401)
    args = parser.parse_args()
    summary = subset_multicontrol_dataset(args.src_h5, args.dst_h5, num_rooms=int(args.num_rooms), seed=int(args.seed))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

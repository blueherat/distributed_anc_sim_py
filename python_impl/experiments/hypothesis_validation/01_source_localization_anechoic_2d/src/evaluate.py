from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.hypothesis_validation_common import evaluate_localization_suite, resolve_stage_level


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--level", type=str, default="L1")
    parser.add_argument("--window-preset", type=str, default="W1")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--geometry-filter-mode", type=str, default="none", choices=["none", "stable"])
    args = parser.parse_args()
    stage_cfg = resolve_stage_level("01", args.level)
    window_suffix = "" if str(args.window_preset).upper() == "W1" else f"_{str(args.window_preset).lower()}"
    geometry_suffix = "" if str(args.geometry_filter_mode).lower() == "none" else "_stable"
    h5_path = args.h5_path or str(Path(__file__).resolve().parents[1] / "data" / f"source_localization_anechoic_2d_{stage_cfg['level'].lower()}{window_suffix}{geometry_suffix}.h5")
    result_dir_name = str(stage_cfg["level"]) if str(args.window_preset).upper() == "W1" else f"{stage_cfg['level']}_{str(args.window_preset).upper()}"
    if str(args.geometry_filter_mode).lower() == "stable":
        result_dir_name = f"{result_dir_name}_stable"
    output_dir = args.output_dir or str(Path(__file__).resolve().parents[1] / "results" / result_dir_name)
    summary = evaluate_localization_suite(h5_path, output_dir, device=str(args.device), stage_id="01", level=str(stage_cfg["level"]))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

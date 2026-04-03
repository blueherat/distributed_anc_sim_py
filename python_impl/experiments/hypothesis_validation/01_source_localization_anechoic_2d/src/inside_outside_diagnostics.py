from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from relative_distance_notebook_utils import export_inside_outside_diagnostics


def _parse_bins(text: str, name: str) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError(f"{name} requires at least two comma-separated values")
    for left, right in zip(values[:-1], values[1:]):
        if right <= left:
            raise ValueError(f"{name} must be strictly increasing: {values}")
    return tuple(values)


def _default_checkpoint_path() -> Path:
    step_dir = Path(__file__).resolve().parents[1]
    return step_dir / "results" / "relative_distance_l2_stable_v3_30k_true_tdoa" / "best_model.pt"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Stage01 inside/outside diagnostics for relative-distance model predictions."
    )
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--use-true-tdoa", action="store_true")
    parser.add_argument("--use-gcc-tdoa", action="store_true")
    parser.add_argument("--success-threshold-m", type=float, default=0.10)
    parser.add_argument("--cond-bins", type=str, default="0,10,20,30,40,1000000000")
    parser.add_argument("--area-bins", type=str, default="0,0.15,0.25,0.4,0.7,10")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if bool(args.use_true_tdoa) and bool(args.use_gcc_tdoa):
        raise ValueError("--use-true-tdoa and --use-gcc-tdoa cannot be enabled together")

    checkpoint_path = Path(args.checkpoint_path).resolve() if args.checkpoint_path else _default_checkpoint_path().resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else checkpoint_path.parent / "inside_outside_diagnostics.csv"
    output_json = Path(args.output_json).resolve() if args.output_json else checkpoint_path.parent / "inside_outside_summary.json"

    use_true_tdoa: bool | None = None
    if bool(args.use_true_tdoa):
        use_true_tdoa = True
    elif bool(args.use_gcc_tdoa):
        use_true_tdoa = False

    summary = export_inside_outside_diagnostics(
        checkpoint_path=checkpoint_path,
        output_csv=output_csv,
        output_json=output_json,
        h5_path=args.h5_path,
        use_true_tdoa=use_true_tdoa,
        success_threshold_m=float(args.success_threshold_m),
        cond_bins=_parse_bins(args.cond_bins, "cond_bins"),
        area_bins=_parse_bins(args.area_bins, "area_bins"),
        device=str(args.device),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

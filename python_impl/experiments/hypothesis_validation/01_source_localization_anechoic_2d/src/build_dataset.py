from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.hypothesis_validation_common import (
    LocalizationConfig,
    build_localization_dataset,
    resolve_localization_window_preset,
    resolve_stage_level,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-h5", type=str, default=None)
    parser.add_argument("--level", type=str, default="L1")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--window-preset", type=str, default="W1")
    parser.add_argument("--signal-len", type=int, default=None)
    parser.add_argument("--ref-window-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--geometry-filter-mode", type=str, default="stable", choices=["none", "stable"])
    parser.add_argument("--min-triangle-area", type=float, default=0.25)
    parser.add_argument("--max-jacobian-condition", type=float, default=30.0)
    parser.add_argument("--max-triangle-angle-deg", type=float, default=120.0)
    parser.add_argument("--near-ref-inside-threshold-m", type=float, default=0.50)
    parser.add_argument("--require-source-inside-if-obtuse", action="store_true", default=True)
    parser.add_argument("--small-angle-threshold-deg", type=float, default=30.0)
    parser.add_argument("--small-angle-vertex-clearance-ratio", type=float, default=0.35)
    parser.add_argument("--small-angle-opposite-edge-clearance-ratio", type=float, default=0.20)
    parser.add_argument("--small-angle-prefer-long-edges-ratio", type=float, default=0.85)
    parser.add_argument("--store-geometry-metrics", action="store_true")
    args = parser.parse_args()
    stage_cfg = resolve_stage_level("01", args.level, num_samples=args.num_samples)
    window_cfg = resolve_localization_window_preset(args.window_preset, signal_len=args.signal_len, ref_window_len=args.ref_window_len)
    window_suffix = "" if str(window_cfg["window_preset"]) == "W1" else f"_{str(window_cfg['window_preset']).lower()}"
    geometry_suffix = "" if str(args.geometry_filter_mode).lower() == "none" else "_stable_v3"
    sample_suffix = f"_{int(stage_cfg['num_samples']) // 1000}k" if int(stage_cfg["num_samples"]) % 1000 == 0 else f"_{int(stage_cfg['num_samples'])}"
    output_h5 = args.output_h5 or str(
        Path(__file__).resolve().parents[1] / "data" / f"source_localization_anechoic_2d_{stage_cfg['level'].lower()}{window_suffix}{geometry_suffix}{sample_suffix}.h5"
    )
    cfg = LocalizationConfig(
        num_samples=int(stage_cfg["num_samples"]),
        seed=int(args.seed),
        profile="anechoic",
        signal_len=int(window_cfg["signal_len"]),
        ref_window_len=int(window_cfg["ref_window_len"]),
        window_preset=str(window_cfg["window_preset"]),
        geometry_filter_mode=str(args.geometry_filter_mode),
        min_triangle_area=float(args.min_triangle_area),
        max_jacobian_condition=float(args.max_jacobian_condition),
        max_triangle_angle_deg=float(args.max_triangle_angle_deg),
        near_ref_inside_threshold_m=float(args.near_ref_inside_threshold_m),
        require_source_inside_if_obtuse=bool(args.require_source_inside_if_obtuse),
        small_angle_threshold_deg=float(args.small_angle_threshold_deg),
        small_angle_vertex_clearance_ratio=float(args.small_angle_vertex_clearance_ratio),
        small_angle_opposite_edge_clearance_ratio=float(args.small_angle_opposite_edge_clearance_ratio),
        small_angle_prefer_long_edges_ratio=float(args.small_angle_prefer_long_edges_ratio),
        store_geometry_metrics=bool(args.store_geometry_metrics or str(args.geometry_filter_mode).lower() == "stable"),
        audit_rule_version="stage01_stable_v3" if str(args.geometry_filter_mode).lower() == "stable" else "none",
        source_region_rule_version="stage01_stable_v3" if str(args.geometry_filter_mode).lower() == "stable" else "none",
    )
    summary = build_localization_dataset(output_h5, cfg)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

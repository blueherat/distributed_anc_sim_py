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
    parser.add_argument("--window-preset", type=str, default="W2")
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
    parser.add_argument("--reflection-gain", type=float, default=0.65)
    parser.add_argument("--air-attenuation-alpha-per-m", type=float, default=0.03)
    parser.add_argument("--disable-air-attenuation", action="store_true")
    parser.add_argument("--anechoic-mix-frac", type=float, default=0.10)
    parser.add_argument("--store-geometry-metrics", action="store_true")
    args = parser.parse_args()
    stage_cfg = resolve_stage_level("02", args.level, num_samples=args.num_samples)
    window_cfg = resolve_localization_window_preset(args.window_preset, signal_len=args.signal_len, ref_window_len=args.ref_window_len)
    geometry_suffix = "" if str(args.geometry_filter_mode).lower() == "none" else "_stable_v3"
    window_suffix = f"_{str(window_cfg['window_preset']).lower()}"
    if str(stage_cfg["level"]).upper() == "L1" and int(stage_cfg["num_samples"]) == 6000:
        default_name = f"source_localization_single_reflection_{stage_cfg['level'].lower()}{geometry_suffix}{window_suffix}.h5"
    elif str(stage_cfg["level"]).upper() == "L2" and int(stage_cfg["num_samples"]) == 50000:
        default_name = f"source_localization_single_reflection_{stage_cfg['level'].lower()}{geometry_suffix}{window_suffix}_50k.h5"
    else:
        sample_suffix = (
            f"_{int(stage_cfg['num_samples']) // 1000}k"
            if int(stage_cfg["num_samples"]) % 1000 == 0
            else f"_{int(stage_cfg['num_samples'])}"
        )
        default_name = f"source_localization_single_reflection_{stage_cfg['level'].lower()}{geometry_suffix}{window_suffix}{sample_suffix}.h5"
    output_h5 = args.output_h5 or str(Path(__file__).resolve().parents[1] / "data" / default_name)
    cfg = LocalizationConfig(
        num_samples=int(stage_cfg["num_samples"]),
        seed=int(args.seed),
        profile="single_reflection",
        signal_len=int(window_cfg["signal_len"]),
        ref_window_len=int(window_cfg["ref_window_len"]),
        window_preset=str(window_cfg["window_preset"]),
        rir_model="manual_2d_image_source_air",
        reflection_gain=float(args.reflection_gain),
        air_attenuation_enabled=not bool(args.disable_air_attenuation),
        air_attenuation_alpha_per_m=float(args.air_attenuation_alpha_per_m),
        reflection_profile_mix_anechoic_frac=float(args.anechoic_mix_frac),
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
        audit_rule_version="stage02_reflection_w2_air_v1" if str(args.geometry_filter_mode).lower() == "stable" else "none",
        source_region_rule_version="stage01_stable_v3" if str(args.geometry_filter_mode).lower() == "stable" else "none",
    )
    print(build_localization_dataset(output_h5, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

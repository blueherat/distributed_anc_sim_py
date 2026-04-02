from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.hypothesis_validation_common import (
    LOCALIZATION_THRESHOLDS,
    LocalizationConfig,
    resolve_localization_window_preset,
    run_localization_geometry_stability_audit,
)


def _build_cfg(window_preset: str, signal_len: int | None, ref_window_len: int | None, seed: int) -> LocalizationConfig:
    window_cfg = resolve_localization_window_preset(window_preset, signal_len=signal_len, ref_window_len=ref_window_len)
    return LocalizationConfig(
        num_samples=1,
        seed=int(seed),
        profile="anechoic",
        signal_len=int(window_cfg["signal_len"]),
        ref_window_len=int(window_cfg["ref_window_len"]),
        geometry_filter_mode="none",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--window-preset", type=str, default="W1")
    parser.add_argument("--signal-len", type=int, default=None)
    parser.add_argument("--ref-window-len", type=int, default=None)
    parser.add_argument("--fallback-window-preset", type=str, default="W2")
    parser.add_argument("--seed", type=int, default=20260401)
    args = parser.parse_args()

    base_output = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[1] / "results" / "geometry_stability_audit"
    base_output.mkdir(parents=True, exist_ok=True)
    primary_cfg = _build_cfg(args.window_preset, args.signal_len, args.ref_window_len, args.seed)
    primary_output = base_output / str(args.window_preset).upper()
    primary_summary = run_localization_geometry_stability_audit(
        primary_output,
        primary_cfg,
        num_samples=int(args.num_samples),
        stage_id="01",
        level="audit",
    )
    final_summary = {
        "attempted_window_presets": [str(args.window_preset).upper()],
        "selected_window_preset": primary_summary["window_preset"],
        "selected_thresholds": primary_summary.get("selected_thresholds"),
        "runs": {str(args.window_preset).upper(): primary_summary},
        "gate_thresholds": LOCALIZATION_THRESHOLDS["01"],
    }
    if primary_summary.get("selected_thresholds") is None and args.fallback_window_preset:
        fallback_cfg = _build_cfg(args.fallback_window_preset, None, None, args.seed)
        fallback_output = base_output / str(args.fallback_window_preset).upper()
        fallback_summary = run_localization_geometry_stability_audit(
            fallback_output,
            fallback_cfg,
            num_samples=int(args.num_samples),
            stage_id="01",
            level="audit",
        )
        final_summary["attempted_window_presets"].append(str(args.fallback_window_preset).upper())
        final_summary["runs"][str(args.fallback_window_preset).upper()] = fallback_summary
        if fallback_summary.get("selected_thresholds") is not None:
            final_summary["selected_window_preset"] = fallback_summary["window_preset"]
            final_summary["selected_thresholds"] = fallback_summary["selected_thresholds"]
    (base_output / "summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(final_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

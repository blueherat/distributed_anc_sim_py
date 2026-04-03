from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from gcc_reflection_notebook_utils import build_gcc_reflection_bundle, train_gcc_to_tdoa_model


def _default_h5_path() -> Path:
    step_dir = Path(__file__).resolve().parents[1]
    return step_dir / "data" / "source_localization_single_reflection_l1_stable_v3_w2.h5"


def _default_results_root() -> Path:
    step_dir = Path(__file__).resolve().parents[1]
    return step_dir / "results" / "_target_push_geom_val_1e4"


def _resolve_device(text: str) -> str:
    value = str(text).strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    def _read(split: str, key: str) -> float:
        return float(summary.get(split, {}).get(key, float("inf")))

    return {
        "best_epoch": int(summary.get("best_epoch", -1)),
        "epochs_ran": int(summary.get("epochs_ran", summary.get("epochs_requested", -1))),
        "iid_val_tdoa_mae_s": _read("iid_val", "tdoa_mae_s"),
        "geom_val_tdoa_mae_s": _read("geom_val", "tdoa_mae_s"),
        "iid_test_tdoa_mae_s": _read("iid_test", "tdoa_mae_s"),
        "geom_test_tdoa_mae_s": _read("geom_test", "tdoa_mae_s"),
        "iid_test_consistency_mae_samples": _read("iid_test", "consistency_mae_samples"),
        "geom_test_consistency_mae_samples": _read("geom_test", "consistency_mae_samples"),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage02 GCC->TDOA short ablation and long target push.")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--short-epochs", type=int, default=16)
    parser.add_argument("--long-epochs", type=int, default=96)
    parser.add_argument("--target-geom-val-mae-s", type=float, default=1.0e-4)
    parser.add_argument("--skip-long-run", action="store_true")
    parser.add_argument("--aux-feature-mode", type=str, default="pair_distance_norm")
    parser.add_argument("--dropout-p", type=float, default=0.10)
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1.0e-6)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    args = parser.parse_args()

    h5_path = Path(args.h5_path).resolve() if args.h5_path else _default_h5_path().resolve()
    results_root = Path(args.results_root).resolve() if args.results_root else _default_results_root().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    bundle = build_gcc_reflection_bundle(h5_path)

    common = {
        "bundle": bundle,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "live_plot": False,
        "aux_feature_mode": str(args.aux_feature_mode),
        "dropout_p": float(args.dropout_p),
        "scheduler_patience": int(args.scheduler_patience),
        "scheduler_factor": float(args.scheduler_factor),
        "scheduler_min_lr": float(args.scheduler_min_lr),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
    }

    short_specs: list[dict[str, Any]] = [
        {
            "name": "short_nocurr_default",
            "params": {
                "lr": 1.0e-3,
                "epochs": int(args.short_epochs),
                "huber_delta_norm": 0.05,
                "bound_penalty_weight": 0.05,
                "curriculum_mode": "none",
            },
        },
        {
            "name": "short_curr_default",
            "params": {
                "lr": 1.0e-3,
                "epochs": int(args.short_epochs),
                "huber_delta_norm": 0.05,
                "bound_penalty_weight": 0.05,
                "curriculum_mode": "profile_two_stage",
                "curriculum_stage1_ratio": 0.5,
                "curriculum_stage1_anechoic_boost": 8.0,
            },
        },
        {
            "name": "short_curr_tight_huber",
            "params": {
                "lr": 7.0e-4,
                "epochs": int(args.short_epochs),
                "huber_delta_norm": 0.02,
                "bound_penalty_weight": 0.10,
                "curriculum_mode": "profile_two_stage",
                "curriculum_stage1_ratio": 0.6,
                "curriculum_stage1_anechoic_boost": 10.0,
            },
        },
    ]

    short_reports: list[dict[str, Any]] = []
    print(f"[info] device={device}, h5={h5_path}")
    for spec in short_specs:
        run_dir = results_root / spec["name"]
        kwargs = dict(common)
        kwargs.update(spec["params"])
        kwargs["result_dir"] = run_dir
        summary = train_gcc_to_tdoa_model(**kwargs)
        metrics = _extract_metrics(summary)
        report = {
            "run": str(spec["name"]),
            "result_dir": str(run_dir),
            "params": spec["params"],
            **metrics,
        }
        short_reports.append(report)
        print(json.dumps(report, ensure_ascii=False))

    short_reports_sorted = sorted(short_reports, key=lambda x: (float(x["geom_val_tdoa_mae_s"]), float(x["iid_val_tdoa_mae_s"])))
    best_short = short_reports_sorted[0]

    final_report: dict[str, Any] = {
        "h5_path": str(h5_path),
        "device": str(device),
        "target_geom_val_mae_s": float(args.target_geom_val_mae_s),
        "short_runs": short_reports,
        "best_short": best_short,
    }

    if not bool(args.skip_long_run):
        best_params = dict(best_short["params"])
        best_params["epochs"] = int(args.long_epochs)
        long_dir = results_root / "long_best_from_short"
        long_kwargs = dict(common)
        long_kwargs.update(best_params)
        long_kwargs["result_dir"] = long_dir
        long_summary = train_gcc_to_tdoa_model(**long_kwargs)
        long_metrics = _extract_metrics(long_summary)
        final_report["long_run"] = {
            "run": "long_best_from_short",
            "result_dir": str(long_dir),
            "params": best_params,
            **long_metrics,
            "pass_target": bool(float(long_metrics["geom_val_tdoa_mae_s"]) < float(args.target_geom_val_mae_s)),
        }
        print(json.dumps(final_report["long_run"], ensure_ascii=False))

    _save_json(results_root / "ablation_report.json", final_report)
    print(f"[done] report={results_root / 'ablation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

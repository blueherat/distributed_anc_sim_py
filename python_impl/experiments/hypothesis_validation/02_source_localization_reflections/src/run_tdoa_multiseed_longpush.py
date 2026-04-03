from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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


def _default_init_checkpoint() -> Path:
    step_dir = Path(__file__).resolve().parents[1]
    return step_dir / "results" / "_target_push_geom_val_1e4" / "finetune_from_long_nocurr_lr2e4" / "best_model.pt"


def _default_results_root() -> Path:
    step_dir = Path(__file__).resolve().parents[1]
    return step_dir / "results" / "_target_push_geom_val_1e4" / "multiseed_longpush"


def _resolve_device(text: str) -> str:
    value = str(text).strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _parse_seeds(text: str) -> list[int]:
    tokens = [item.strip() for item in str(text).split(",") if item.strip()]
    seeds = [int(item) for item in tokens]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    return seeds


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


def _aggregate_reports(seed_reports: list[dict[str, Any]], target: float) -> dict[str, Any]:
    if not seed_reports:
        return {
            "num_runs": 0,
            "pass_count": 0,
            "pass_rate": 0.0,
        }

    geom_vals = np.asarray([float(item["geom_val_tdoa_mae_s"]) for item in seed_reports], dtype=np.float64)
    geom_tests = np.asarray([float(item["geom_test_tdoa_mae_s"]) for item in seed_reports], dtype=np.float64)
    iid_vals = np.asarray([float(item["iid_val_tdoa_mae_s"]) for item in seed_reports], dtype=np.float64)
    pass_flags = geom_vals < float(target)
    best_idx = int(np.argmin(geom_vals))

    return {
        "num_runs": int(len(seed_reports)),
        "best_seed": int(seed_reports[best_idx]["seed"]),
        "best_result_dir": str(seed_reports[best_idx]["result_dir"]),
        "best_epoch": int(seed_reports[best_idx]["best_epoch"]),
        "geom_val_min": float(np.min(geom_vals)),
        "geom_val_mean": float(np.mean(geom_vals)),
        "geom_val_std": float(np.std(geom_vals)),
        "geom_test_at_best_seed": float(seed_reports[best_idx]["geom_test_tdoa_mae_s"]),
        "geom_test_mean": float(np.mean(geom_tests)),
        "geom_test_std": float(np.std(geom_tests)),
        "iid_val_mean": float(np.mean(iid_vals)),
        "iid_val_std": float(np.std(iid_vals)),
        "pass_count": int(np.sum(pass_flags)),
        "pass_rate": float(np.mean(pass_flags.astype(np.float64))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage02 multi-seed long-push finetuning and aggregate results.")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--no-init-checkpoint", action="store_true")
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="7,42,123,999,2024")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--huber-delta-norm", type=float, default=0.05)
    parser.add_argument("--bound-penalty-weight", type=float, default=0.0)
    parser.add_argument("--aux-feature-mode", type=str, default="pair_distance_norm")
    parser.add_argument("--dropout-p", type=float, default=0.10)
    parser.add_argument("--model-width-mult", type=float, default=1.0)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1.0e-7)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--target-geom-val-mae-s", type=float, default=1.0e-4)
    args = parser.parse_args()

    h5_path = Path(args.h5_path).resolve() if args.h5_path else _default_h5_path().resolve()
    init_checkpoint: Path | None
    if bool(args.no_init_checkpoint):
        init_checkpoint = None
    else:
        init_checkpoint = Path(args.init_checkpoint).resolve() if args.init_checkpoint else _default_init_checkpoint().resolve()
    results_root = Path(args.results_root).resolve() if args.results_root else _default_results_root().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    device = _resolve_device(args.device)

    bundle = build_gcc_reflection_bundle(h5_path)

    common = {
        "bundle": bundle,
        "device": str(device),
        "live_plot": False,
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "huber_delta_norm": float(args.huber_delta_norm),
        "bound_penalty_weight": float(args.bound_penalty_weight),
        "aux_feature_mode": str(args.aux_feature_mode),
        "dropout_p": float(args.dropout_p),
        "model_width_mult": float(args.model_width_mult),
        "scheduler_patience": int(args.scheduler_patience),
        "scheduler_factor": float(args.scheduler_factor),
        "scheduler_min_lr": float(args.scheduler_min_lr),
        "early_stop_patience": None if int(args.early_stop_patience) <= 0 else int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "curriculum_mode": "none",
        "init_checkpoint_path": init_checkpoint,
    }

    print(f"[info] device={device}, h5={h5_path}")
    print(f"[info] init_checkpoint={init_checkpoint}")
    print(f"[info] seeds={seeds}")

    seed_reports: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = results_root / f"seed_{int(seed)}"
        kwargs = dict(common)
        kwargs["seed"] = int(seed)
        kwargs["result_dir"] = run_dir
        summary = train_gcc_to_tdoa_model(**kwargs)
        metrics = _extract_metrics(summary)
        report = {
            "seed": int(seed),
            "result_dir": str(run_dir),
            **metrics,
            "pass_target": bool(float(metrics["geom_val_tdoa_mae_s"]) < float(args.target_geom_val_mae_s)),
        }
        seed_reports.append(report)
        print(json.dumps(report, ensure_ascii=False))

    final_report = {
        "h5_path": str(h5_path),
        "init_checkpoint_path": None if init_checkpoint is None else str(init_checkpoint),
        "device": str(device),
        "target_geom_val_mae_s": float(args.target_geom_val_mae_s),
        "config": {
            "lr": float(args.lr),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "huber_delta_norm": float(args.huber_delta_norm),
            "bound_penalty_weight": float(args.bound_penalty_weight),
            "aux_feature_mode": str(args.aux_feature_mode),
            "dropout_p": float(args.dropout_p),
            "model_width_mult": float(args.model_width_mult),
            "scheduler_patience": int(args.scheduler_patience),
            "scheduler_factor": float(args.scheduler_factor),
            "scheduler_min_lr": float(args.scheduler_min_lr),
            "early_stop_patience": None if int(args.early_stop_patience) <= 0 else int(args.early_stop_patience),
            "early_stop_min_delta": float(args.early_stop_min_delta),
        },
        "seed_runs": seed_reports,
        "aggregate": _aggregate_reports(seed_reports, target=float(args.target_geom_val_mae_s)),
    }

    report_path = results_root / "multiseed_longpush_report.json"
    _save_json(report_path, final_report)
    print(f"[done] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

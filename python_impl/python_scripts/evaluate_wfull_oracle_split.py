from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.evaluate_hybrid_deep_fxlms_single_control import (
    load_target_metric_values,
    warmstart_metrics,
)
from python_scripts.train_hybrid_deep_fxlms_single_control import (
    level_mask,
    split_indices_train_val_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate oracle W_full init on strict split.")
    parser.add_argument(
        "--h5-path",
        type=str,
        default=str(ROOT / "python_impl" / "python_scripts" / "cfxlms_qc_dataset_single_control.h5"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--eval-split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--warmstart-level", type=int, default=3)
    parser.add_argument("--warmstart-cases", type=int, default=8)
    parser.add_argument("--early-window-s", type=float, default=0.25)
    parser.add_argument("--half-target-ratio", type=float, default=0.5)
    parser.add_argument("--min-improvement-db", type=float, default=6.0)
    parser.add_argument("--target-metric", choices=("nr_last_db",), default="nr_last_db")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def _resolve_eval_indices(
    image_order: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    warmstart_level: int,
    eval_split: str,
) -> np.ndarray:
    level_idx = np.where(level_mask(image_order, int(warmstart_level)))[0].astype(np.int64)
    if eval_split == "train":
        base = train_idx
    elif eval_split == "val":
        base = val_idx
    else:
        base = test_idx
    return np.intersect1d(level_idx, base)


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5_path)

    with h5py.File(str(h5_path), "r") as h5:
        raw = h5["raw"]
        image_order = np.asarray(raw["room_params/image_source_order"], dtype=np.int64)
        if "W_full" not in raw:
            raise RuntimeError("raw/W_full missing in dataset.")
        w_full = np.asarray(raw["W_full"], dtype=np.float32)

    n_rooms = int(image_order.shape[0])
    all_idx = np.arange(n_rooms, dtype=np.int64)
    train_idx, val_idx, test_idx = split_indices_train_val_test(
        indices=all_idx,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    eval_idx = _resolve_eval_indices(
        image_order=image_order,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        warmstart_level=int(args.warmstart_level),
        eval_split=str(args.eval_split),
    )
    if eval_idx.size == 0:
        raise RuntimeError(f"No eval samples in split={args.eval_split} for warmstart_level={args.warmstart_level}.")

    probe = eval_idx[: min(int(args.warmstart_cases), int(eval_idx.size))]
    probe_list = [int(v) for v in probe.tolist()]

    w_probe = np.asarray(w_full[probe, 0, :, :], dtype=np.float32)
    target_vals = load_target_metric_values(
        h5_path=h5_path,
        room_indices=probe_list,
        metric_key=str(args.target_metric),
    )

    warm = warmstart_metrics(
        h5_path=h5_path,
        room_indices=probe_list,
        w_pred=w_probe,
        early_window_s=float(args.early_window_s),
        target_nr_db=target_vals,
        half_target_ratio=float(args.half_target_ratio),
        target_metric=str(args.target_metric),
        min_improvement_db=float(args.min_improvement_db),
    )

    report: dict[str, Any] = {
        "h5_path": str(h5_path),
        "seed": int(args.seed),
        "eval_split": str(args.eval_split),
        "warmstart_level": int(args.warmstart_level),
        "warmstart_cases": int(args.warmstart_cases),
        "warmstart_metrics": warm,
        "improvement_gate": {
            "enabled": True,
            "threshold_db": float(args.min_improvement_db),
            "early_gain_db_mean": float(warm.get("early_gain_db_mean", float("nan"))),
            "gap_db": float(warm.get("improvement_gap_db", float("nan"))),
            "sample_pass_rate": float(warm.get("sample_improvement_pass_rate", float("nan"))),
            "sample_6db_pass_rate": float(warm.get("sample_6db_pass_rate", float("nan"))),
            "num_samples": int(warm.get("num_samples", 0)),
            "pass": bool(warm.get("improvement_pass", False)),
        },
        "half_target_gate": {
            "legacy": True,
            "enabled": True,
            "ratio": float(args.half_target_ratio),
            "target_metric": str(args.target_metric),
            "target_nr_db_mean": float(warm.get("target_nr_db_mean", float("nan"))),
            "init_nr_db_mean": float(warm.get("init_nr_db_mean", float("nan"))),
            "threshold_db": float(warm.get("half_target_threshold_db", float("nan"))),
            "gap_db": float(warm.get("half_target_gap_db", float("nan"))),
            "sample_pass_rate": float(warm.get("sample_pass_rate", float("nan"))),
            "num_samples": int(warm.get("num_samples", 0)),
            "pass": bool(warm.get("half_target_pass", False)),
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_json = str(args.output_json).strip()
    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

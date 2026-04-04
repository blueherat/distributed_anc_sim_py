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

from python_scripts.cfxlms_single_control_dataset_impl import (
    DatasetBuildConfig,
    _canonical_q_from_paths,
    _solve_w_canonical_from_q,
)
from python_scripts.evaluate_hybrid_deep_fxlms_single_control import (
    load_target_metric_values,
    warmstart_metrics,
)
from python_scripts.train_hybrid_deep_fxlms_single_control import (
    level_mask,
    split_indices_train_val_test,
)


def parse_float_list(text: str) -> list[float]:
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("Empty float list.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical path-derived W init on strict holdout (no W_opt supervision)."
    )
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
    parser.add_argument("--calibration-cases", type=int, default=64)
    parser.add_argument("--lambda-q-grid", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--lambda-w-grid", type=str, default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3")
    parser.add_argument("--gain-grid", type=str, default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--tune-objective", choices=("mse", "half_target_gap"), default="mse")
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


def _predict_w_from_paths(
    p_ref_paths: np.ndarray,
    d_paths: np.ndarray,
    s_paths: np.ndarray,
    indices: np.ndarray,
    filter_len: int,
    q_full_len: int,
    lambda_q_scale: float,
    lambda_w: float,
) -> np.ndarray:
    out = np.zeros((indices.size, p_ref_paths.shape[1], filter_len), dtype=np.float32)
    for j, idx in enumerate(indices.tolist()):
        q = _canonical_q_from_paths(
            d_path=d_paths[idx],
            s_path=s_paths[idx],
            q_full_len=int(q_full_len),
            lambda_q_scale=float(lambda_q_scale),
        )
        w = _solve_w_canonical_from_q(
            q_full=q,
            p_ref_paths=p_ref_paths[idx],
            filter_len=int(filter_len),
            lambda_w=float(lambda_w),
            q_full_len=int(q_full_len),
        )
        out[j] = w[0]
    return out


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5_path)

    lambda_q_grid = parse_float_list(args.lambda_q_grid)
    lambda_w_grid = parse_float_list(args.lambda_w_grid)
    gain_grid = parse_float_list(args.gain_grid)

    with h5py.File(str(h5_path), "r") as h5:
        cfg = DatasetBuildConfig(**json.loads(h5.attrs["config_json"]))
        raw = h5["raw"]
        image_order = np.asarray(raw["room_params/image_source_order"], dtype=np.int64)
        p_ref_paths = np.asarray(raw["P_ref_paths"], dtype=np.float32)
        d_paths = np.asarray(raw["D_path"], dtype=np.float32)
        s_paths = np.asarray(raw["S_paths"], dtype=np.float32)
        if "W_full" not in raw:
            raise RuntimeError("raw/W_full missing; canonical path init evaluation requires W_full for val calibration.")
        w_full = np.asarray(raw["W_full"], dtype=np.float32)

    n_rooms = int(image_order.shape[0])
    all_indices = np.arange(n_rooms, dtype=np.int64)
    train_idx, val_idx, test_idx = split_indices_train_val_test(
        indices=all_indices,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    calib_pool = _resolve_eval_indices(
        image_order=image_order,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        warmstart_level=int(args.warmstart_level),
        eval_split="val",
    )
    if calib_pool.size == 0:
        calib_pool = _resolve_eval_indices(
            image_order=image_order,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            warmstart_level=int(args.warmstart_level),
            eval_split="train",
        )
    if calib_pool.size == 0:
        raise RuntimeError("No calibration samples available in train/val split for selected warmstart level.")

    calib_idx = calib_pool[: min(int(args.calibration_cases), int(calib_pool.size))]

    filter_len = int(cfg.filter_len)
    q_full_len = int(cfg.filter_len + cfg.rir_store_len - 1)

    best_mse = float("inf")
    best_gap = float("-inf")
    best_pass_rate = float("-inf")
    best_lambda_q = float(lambda_q_grid[0])
    best_lambda_w = float(lambda_w_grid[0])
    best_gain = float(gain_grid[0])

    target_calib = np.asarray(w_full[calib_idx, 0, :, :], dtype=np.float32)
    calib_list = [int(v) for v in calib_idx.tolist()]
    calib_target_vals = load_target_metric_values(
        h5_path=h5_path,
        room_indices=calib_list,
        metric_key=str(args.target_metric),
    )

    for lambda_q in lambda_q_grid:
        for lambda_w in lambda_w_grid:
            pred = _predict_w_from_paths(
                p_ref_paths=p_ref_paths,
                d_paths=d_paths,
                s_paths=s_paths,
                indices=calib_idx,
                filter_len=filter_len,
                q_full_len=q_full_len,
                lambda_q_scale=float(lambda_q),
                lambda_w=float(lambda_w),
            )
            for gain in gain_grid:
                pred_scaled = pred * float(gain)
                mse = float(np.mean((pred_scaled - target_calib) ** 2))
                if str(args.tune_objective) == "mse":
                    if mse < best_mse:
                        best_mse = mse
                        best_lambda_q = float(lambda_q)
                        best_lambda_w = float(lambda_w)
                        best_gain = float(gain)
                else:
                    calib_warm = warmstart_metrics(
                        h5_path=h5_path,
                        room_indices=calib_list,
                        w_pred=pred_scaled,
                        early_window_s=float(args.early_window_s),
                        target_nr_db=calib_target_vals,
                        half_target_ratio=float(args.half_target_ratio),
                        target_metric=str(args.target_metric),
                        min_improvement_db=float(args.min_improvement_db),
                    )
                    gap = float(calib_warm.get("half_target_gap_db", float("-inf")))
                    pass_rate = float(calib_warm.get("sample_pass_rate", float("-inf")))
                    if (
                        gap > best_gap + 1.0e-12
                        or (np.isclose(gap, best_gap) and pass_rate > best_pass_rate + 1.0e-12)
                    ):
                        best_gap = gap
                        best_pass_rate = pass_rate
                        best_mse = mse
                        best_lambda_q = float(lambda_q)
                        best_lambda_w = float(lambda_w)
                        best_gain = float(gain)

    eval_pool = _resolve_eval_indices(
        image_order=image_order,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        warmstart_level=int(args.warmstart_level),
        eval_split=str(args.eval_split),
    )
    if eval_pool.size == 0:
        raise RuntimeError(f"No eval samples in split={args.eval_split} for warmstart_level={args.warmstart_level}.")

    probe = eval_pool[: min(int(args.warmstart_cases), int(eval_pool.size))]
    probe_list = [int(v) for v in probe.tolist()]

    w_probe = _predict_w_from_paths(
        p_ref_paths=p_ref_paths,
        d_paths=d_paths,
        s_paths=s_paths,
        indices=probe,
        filter_len=filter_len,
        q_full_len=q_full_len,
        lambda_q_scale=float(best_lambda_q),
        lambda_w=float(best_lambda_w),
    ) * float(best_gain)

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
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "eval_split": str(args.eval_split),
        "warmstart_level": int(args.warmstart_level),
        "warmstart_cases": int(args.warmstart_cases),
        "calibration_cases": int(calib_idx.size),
        "tune_objective": str(args.tune_objective),
        "calibration_mse": float(best_mse),
        "best_lambda_q_scale": float(best_lambda_q),
        "best_lambda_w": float(best_lambda_w),
        "best_gain": float(best_gain),
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

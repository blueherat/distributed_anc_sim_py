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


def parse_int_list(text: str) -> list[int]:
    vals = [int(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("Empty int list.")
    return vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KNN path-based W init on strict holdout.")
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
    parser.add_argument("--k-grid", type=str, default="1,3,5,9,17")
    parser.add_argument("--feature-p-ref-len", type=int, default=256)
    parser.add_argument("--feature-d-len", type=int, default=256)
    parser.add_argument("--feature-s-len", type=int, default=256)
    parser.add_argument("--calibration-cases", type=int, default=128)
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


def _build_features(
    p_ref_paths: np.ndarray,
    d_paths: np.ndarray,
    s_paths: np.ndarray,
    p_ref_len: int,
    d_len: int,
    s_len: int,
) -> np.ndarray:
    p = p_ref_paths[:, :, : int(p_ref_len)].reshape(p_ref_paths.shape[0], -1)
    d = d_paths[:, : int(d_len)]
    s = s_paths[:, : int(s_len)]
    return np.concatenate([p, d, s], axis=1).astype(np.float32)


def _standardize(train_feat: np.ndarray, query_feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_feat, axis=0, keepdims=True)
    std = np.std(train_feat, axis=0, keepdims=True)
    std = np.where(std < 1.0e-8, 1.0, std)
    return (train_feat - mean) / std, (query_feat - mean) / std


def _row_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1.0e-12, 1.0, n)
    return x / n


def _knn_predict_w(
    train_feat: np.ndarray,
    train_w: np.ndarray,
    query_feat: np.ndarray,
    k: int,
) -> np.ndarray:
    k = int(max(1, min(int(k), int(train_feat.shape[0]))))

    train_norm = _row_normalize(train_feat.astype(np.float64))
    query_norm = _row_normalize(query_feat.astype(np.float64))
    sims = query_norm @ train_norm.T

    topk_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    pred = np.zeros((query_feat.shape[0], train_w.shape[1], train_w.shape[2]), dtype=np.float32)
    for i in range(query_feat.shape[0]):
        pred[i] = np.mean(train_w[topk_idx[i]], axis=0)
    return pred


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5_path)
    k_grid = parse_int_list(args.k_grid)

    with h5py.File(str(h5_path), "r") as h5:
        raw = h5["raw"]
        image_order = np.asarray(raw["room_params/image_source_order"], dtype=np.int64)
        p_ref_paths = np.asarray(raw["P_ref_paths"], dtype=np.float32)
        d_paths = np.asarray(raw["D_path"], dtype=np.float32)
        s_paths = np.asarray(raw["S_paths"], dtype=np.float32)
        if "W_full" not in raw:
            raise RuntimeError("raw/W_full missing for KNN calibration/evaluation.")
        w_full = np.asarray(raw["W_full"], dtype=np.float32)

    features = _build_features(
        p_ref_paths=p_ref_paths,
        d_paths=d_paths,
        s_paths=s_paths,
        p_ref_len=int(args.feature_p_ref_len),
        d_len=int(args.feature_d_len),
        s_len=int(args.feature_s_len),
    )

    n_rooms = int(features.shape[0])
    all_idx = np.arange(n_rooms, dtype=np.int64)
    train_idx, val_idx, test_idx = split_indices_train_val_test(
        indices=all_idx,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    train_level_idx = _resolve_eval_indices(
        image_order=image_order,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        warmstart_level=int(args.warmstart_level),
        eval_split="train",
    )
    val_level_idx = _resolve_eval_indices(
        image_order=image_order,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        warmstart_level=int(args.warmstart_level),
        eval_split="val",
    )

    if train_level_idx.size == 0 or val_level_idx.size == 0:
        raise RuntimeError("Insufficient train/val level samples for KNN tuning.")

    val_tune_idx = val_level_idx[: min(int(args.calibration_cases), int(val_level_idx.size))]

    train_feat_raw = features[train_level_idx]
    train_w = np.asarray(w_full[train_level_idx, 0, :, :], dtype=np.float32)
    val_feat_raw = features[val_tune_idx]
    val_w = np.asarray(w_full[val_tune_idx, 0, :, :], dtype=np.float32)

    train_feat, val_feat = _standardize(train_feat_raw, val_feat_raw)

    best_k = int(k_grid[0])
    best_mse = float("inf")
    for k in k_grid:
        pred_val = _knn_predict_w(train_feat=train_feat, train_w=train_w, query_feat=val_feat, k=int(k))
        mse = float(np.mean((pred_val - val_w) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_k = int(k)

    eval_level_idx = _resolve_eval_indices(
        image_order=image_order,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        warmstart_level=int(args.warmstart_level),
        eval_split=str(args.eval_split),
    )
    if eval_level_idx.size == 0:
        raise RuntimeError(f"No eval samples in split={args.eval_split} for warmstart_level={args.warmstart_level}.")

    probe_idx = eval_level_idx[: min(int(args.warmstart_cases), int(eval_level_idx.size))]
    probe_list = [int(v) for v in probe_idx.tolist()]

    _, probe_feat = _standardize(train_feat_raw, features[probe_idx])
    pred_probe = _knn_predict_w(
        train_feat=train_feat,
        train_w=train_w,
        query_feat=probe_feat,
        k=int(best_k),
    )

    target_vals = load_target_metric_values(
        h5_path=h5_path,
        room_indices=probe_list,
        metric_key=str(args.target_metric),
    )
    warm = warmstart_metrics(
        h5_path=h5_path,
        room_indices=probe_list,
        w_pred=pred_probe,
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
        "calibration_cases": int(val_tune_idx.size),
        "feature_dims": {
            "p_ref_len": int(args.feature_p_ref_len),
            "d_len": int(args.feature_d_len),
            "s_len": int(args.feature_s_len),
            "total": int(features.shape[1]),
        },
        "best_k": int(best_k),
        "calibration_mse": float(best_mse),
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

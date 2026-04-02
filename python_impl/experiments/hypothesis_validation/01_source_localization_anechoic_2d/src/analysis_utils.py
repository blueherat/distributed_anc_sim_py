from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


WINDOW_KEYS = ("W1", "W2", "W3")


def get_step_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def get_run_paths(window_key: str) -> dict[str, Path]:
    key = str(window_key).upper()
    if key not in WINDOW_KEYS:
        raise KeyError(f"Unknown window key: {window_key}")
    step_dir = get_step_dir()
    return {
        "window_key": Path(key),
        "h5_path": step_dir / "data" / f"source_localization_anechoic_2d_l1_{key.lower()}_stage01.h5",
        "summary_path": step_dir / "results" / f"L1_{key}_stage01" / "summary.json",
        "csv_path": step_dir / "results" / f"L1_{key}_stage01" / "analytic_sample_errors.csv",
        "plots_dir": step_dir / "results" / f"L1_{key}_stage01" / "plots",
    }


def load_run_bundle(window_key: str) -> dict[str, Any]:
    paths = get_run_paths(window_key)
    summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
    return {
        "window_key": str(window_key).upper(),
        "paths": paths,
        "summary": summary,
    }


def load_all_run_bundles() -> dict[str, dict[str, Any]]:
    return {key: load_run_bundle(key) for key in WINDOW_KEYS}


def extract_position_metric_rows(runs: dict[str, dict[str, Any]], method: str = "gcc_phat") -> list[dict[str, Any]]:
    summary_key = "analytic_gcc_phat" if str(method) == "gcc_phat" else "analytic_plain_gcc"
    rows: list[dict[str, Any]] = []
    for key in WINDOW_KEYS:
        metrics = runs[key]["summary"][summary_key]
        rows.extend(
            [
                {
                    "window_key": key,
                    "split": "iid_test",
                    "method": str(method),
                    "median_m": float(metrics["iid_test"]["median_m"]),
                    "p90_m": float(metrics["iid_test"]["p90_m"]),
                    "mean_m": float(metrics["iid_test"]["mean_m"]),
                    "max_m": float(metrics["iid_test"]["max_m"]),
                },
                {
                    "window_key": key,
                    "split": "geom_test",
                    "method": str(method),
                    "median_m": float(metrics["geom_test"]["median_m"]),
                    "p90_m": float(metrics["geom_test"]["p90_m"]),
                    "mean_m": float(metrics["geom_test"]["mean_m"]),
                    "max_m": float(metrics["geom_test"]["max_m"]),
                },
            ]
        )
    return rows


def extract_tdoa_metric_rows(
    runs: dict[str, dict[str, Any]],
    method: str = "gcc_phat",
    scope: str = "overall",
) -> list[dict[str, Any]]:
    summary_key = "analytic_gcc_phat" if str(method) == "gcc_phat" else "analytic_plain_gcc"
    rows: list[dict[str, Any]] = []
    for key in WINDOW_KEYS:
        metrics = runs[key]["summary"][summary_key]
        for split in ("iid_test", "geom_test"):
            delay = metrics[split]["delay_error"][scope]
            rows.append(
                {
                    "window_key": key,
                    "split": split,
                    "scope": scope,
                    "method": str(method),
                    "median_samples": float(delay["median_samples"]),
                    "p90_samples": float(delay["p90_samples"]),
                    "mean_samples": float(delay["mean_samples"]),
                    "max_samples": float(delay["max_samples"]),
                    "median_m": float(delay["median_m"]),
                    "p90_m": float(delay["p90_m"]),
                    "mean_m": float(delay["mean_m"]),
                    "max_m": float(delay["max_m"]),
                }
            )
    return rows


def _triangle_area(tri: np.ndarray) -> float:
    a, b, c = np.asarray(tri, dtype=np.float64)
    return float(abs(0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))))


def _barycentric_inside(p: np.ndarray, tri: np.ndarray) -> bool:
    a, b, c = np.asarray(tri, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(float(den)) < 1.0e-12:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    return bool((u >= 0.0) and (v >= 0.0) and (u + v <= 1.0))


def _jacobian_condition_number(source_pos: np.ndarray, ref_positions: np.ndarray) -> float:
    pairs = ((0, 1), (0, 2), (1, 2))
    src = np.asarray(source_pos, dtype=np.float64)
    refs = np.asarray(ref_positions, dtype=np.float64)
    jac_rows: list[np.ndarray] = []
    for i, j in pairs:
        di = np.linalg.norm(src - refs[i])
        dj = np.linalg.norm(src - refs[j])
        gi = (src - refs[i]) / max(float(di), 1.0e-12)
        gj = (src - refs[j]) / max(float(dj), 1.0e-12)
        jac_rows.append(gi - gj)
    jac = np.asarray(jac_rows, dtype=np.float64)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    if singular_values[-1] < 1.0e-12:
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def _mean_of(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean(np.asarray([row[key] for row in records], dtype=np.float64)))


def compute_sensitivity_stats(window_key: str = "W3", method: str = "gcc_phat") -> dict[str, Any]:
    bundle = load_run_bundle(window_key)
    h5_path = bundle["paths"]["h5_path"]
    csv_path = bundle["paths"]["csv_path"]
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["method"] == str(method):
                rows.append(row)
    with h5py.File(str(h5_path), "r") as h5:
        ref_positions = np.asarray(h5["raw/ref_positions"], dtype=np.float32)
        source_positions = np.asarray(h5["raw/source_position"], dtype=np.float32)
        cfg = json.loads(h5.attrs["config_json"])
    records: list[dict[str, Any]] = []
    for row in rows:
        sample_index = int(row["sample_index"])
        tri = ref_positions[sample_index]
        src = source_positions[sample_index]
        records.append(
            {
                "split": str(row["split"]),
                "sample_index": sample_index,
                "position_error_m": float(row["position_error_m"]),
                "triangle_area": _triangle_area(tri),
                "inside_array": _barycentric_inside(src, tri),
                "jacobian_condition": _jacobian_condition_number(src, tri),
                "min_ref_dist": float(np.min(np.linalg.norm(tri - src[None, :], axis=1))),
                "max_ref_dist": float(np.max(np.linalg.norm(tri - src[None, :], axis=1))),
            }
        )

    def summarize_split(split_name: str) -> dict[str, Any]:
        split_records = [row for row in records if row["split"] == split_name]
        ordered = sorted(split_records, key=lambda row: row["position_error_m"])
        cut = int(math.floor(0.9 * len(ordered)))
        rest = ordered[:cut]
        worst = ordered[cut:]
        return {
            "count": int(len(ordered)),
            "worst10_pos_err_mean": _mean_of(worst, "position_error_m"),
            "rest90_pos_err_mean": _mean_of(rest, "position_error_m"),
            "worst10_area_mean": _mean_of(worst, "triangle_area"),
            "rest90_area_mean": _mean_of(rest, "triangle_area"),
            "worst10_cond_mean": _mean_of(worst, "jacobian_condition"),
            "rest90_cond_mean": _mean_of(rest, "jacobian_condition"),
            "worst10_inside_frac": float(np.mean([1.0 if row["inside_array"] else 0.0 for row in worst])),
            "rest90_inside_frac": float(np.mean([1.0 if row["inside_array"] else 0.0 for row in rest])),
            "worst10_min_ref_dist_mean": _mean_of(worst, "min_ref_dist"),
            "rest90_min_ref_dist_mean": _mean_of(rest, "min_ref_dist"),
            "worst_examples": worst[-5:],
        }

    return {
        "window_key": str(window_key).upper(),
        "method": str(method),
        "cfg": {
            "fs": int(cfg["fs"]),
            "c": float(cfg["c"]),
            "signal_len": int(cfg["signal_len"]),
            "ref_window_len": int(cfg["ref_window_len"]),
        },
        "iid_test": summarize_split("iid_test"),
        "geom_test": summarize_split("geom_test"),
    }


def make_verdict(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    w3 = runs["W3"]["summary"]
    analytic_gate = w3["analytic_gate"]
    return {
        "window_key": "W3",
        "analytic_gate_passed": bool(analytic_gate["passed"]),
        "checks": dict(analytic_gate["checks"]),
        "verdict": "failed" if not bool(analytic_gate["passed"]) else "passed",
        "reason": (
            "W3 下 GCC-PHAT 解析仍未通过 iid/geom 的 p90 门槛，因此 01 当前前提不成立。"
            if not bool(analytic_gate["passed"])
            else "W3 下 GCC-PHAT 解析已通过既定门槛。"
        ),
    }

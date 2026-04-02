from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import convolution_matrix, svd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw-W identifiability for the single-control dataset.")
    parser.add_argument("--h5-path", type=str, default="python_impl/python_scripts/cfxlms_qc_dataset_single_control.h5")
    parser.add_argument("--room-index", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1.0e-8)
    return parser.parse_args()


def build_equivalent_q_matrix(p_ref_paths: np.ndarray, filter_len: int, q_len: int) -> np.ndarray:
    mats = [convolution_matrix(np.asarray(p_i, dtype=np.float64), int(filter_len))[: int(q_len), :] for p_i in np.asarray(p_ref_paths, dtype=np.float64)]
    return np.concatenate(mats, axis=1)


def main() -> int:
    args = parse_args()
    with h5py.File(str(Path(args.h5_path)), "r") as h5:
        cfg = json.loads(h5.attrs["config_json"])
        p_ref = np.asarray(h5["raw/P_ref_paths"][int(args.room_index)] if "P_ref_paths" in h5["raw"] else h5["raw/E2R_paths"][int(args.room_index)], dtype=np.float64)
        w_h5 = np.asarray(h5["raw/W_full"][int(args.room_index), 0], dtype=np.float64)
        filter_len = int(cfg["filter_len"])
        q_len = int(filter_len + cfg["rir_store_len"] - 1 if "rir_store_len" in cfg else filter_len + 512 - 1)

    a_mat = build_equivalent_q_matrix(p_ref, filter_len=filter_len, q_len=q_len)
    svals = svd(a_mat, compute_uv=False)
    rank = int(np.sum(svals > float(args.tol) * svals[0]))
    nullity_lb = int(a_mat.shape[1] - rank)
    w_flat = w_h5.reshape(-1)
    q_equiv = a_mat @ w_flat
    alt_flat = w_flat.copy()
    if nullity_lb > 0:
        _, _, vh = np.linalg.svd(a_mat, full_matrices=True)
        null_vec = vh[-1]
        alt_flat = w_flat + 0.1 * null_vec / (np.linalg.norm(null_vec) + 1.0e-12)
    alt_q = a_mat @ alt_flat
    rel_err = float(np.linalg.norm(q_equiv - alt_q) / (np.linalg.norm(q_equiv) + 1.0e-12))
    summary = {
        "room_index": int(args.room_index),
        "a_shape": [int(v) for v in a_mat.shape],
        "rank": rank,
        "nullity_lower_bound": nullity_lb,
        "equiv_q_rel_error": rel_err,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

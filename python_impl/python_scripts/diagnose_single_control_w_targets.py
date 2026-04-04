from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def resolve_h5_path(explicit_path: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(ROOT / "python_impl" / "python_scripts" / "cfxlms_qc_dataset_single_control.h5")
    candidates.append(ROOT / "python_impl" / "python_scripts" / "cfxlms_qc_dataset_single_control_exp500.h5")

    for cand in candidates:
        p = cand if cand.is_absolute() else (ROOT / cand).resolve()
        if p.exists():
            return p

    checked = [str((c if c.is_absolute() else (ROOT / c).resolve())) for c in candidates]
    raise FileNotFoundError(f"No dataset found. Checked: {checked}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose W_opt/W_full consistency in single-control H5 datasets.")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    h5_path = resolve_h5_path(args.h5_path)

    with h5py.File(str(h5_path), "r") as h5:
        if "raw/W_opt" not in h5 or "raw/W_full" not in h5:
            raise KeyError("Missing raw/W_opt or raw/W_full in dataset.")

        w_opt = np.asarray(h5["raw/W_opt"], dtype=np.float32)
        w_full = np.asarray(h5["raw/W_full"], dtype=np.float32)

    if w_opt.ndim != 3:
        raise ValueError(f"raw/W_opt expected 3D [N,R,L], got shape {w_opt.shape}")
    if w_full.ndim != 4:
        raise ValueError(f"raw/W_full expected 4D [N,S,R,L], got shape {w_full.shape}")
    if w_full.shape[1] < 1:
        raise ValueError(f"raw/W_full second dim must be >=1, got shape {w_full.shape}")

    w_full_slice = w_full[:, 0, :, :]
    if w_full_slice.shape != w_opt.shape:
        raise ValueError(
            f"Shape mismatch: W_full[:,0,:,:]={w_full_slice.shape} vs W_opt={w_opt.shape}."
        )

    diff = np.abs(w_full_slice.astype(np.float64) - w_opt.astype(np.float64))
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))
    p99_abs_diff = float(np.percentile(diff, 99.0))
    allclose = bool(np.allclose(w_full_slice, w_opt, atol=float(args.atol), rtol=0.0))

    mismatch_idx = np.argwhere(diff > float(args.atol))
    first_mismatch: dict[str, int | float] | None = None
    if mismatch_idx.size > 0:
        i, r, l = mismatch_idx[0].tolist()
        first_mismatch = {
            "sample": int(i),
            "ref": int(r),
            "tap": int(l),
            "w_full": float(w_full_slice[i, r, l]),
            "w_opt": float(w_opt[i, r, l]),
            "abs_diff": float(diff[i, r, l]),
        }

    report = {
        "h5_path": str(h5_path),
        "w_opt_shape": [int(v) for v in w_opt.shape],
        "w_full_shape": [int(v) for v in w_full.shape],
        "slice_shape": [int(v) for v in w_full_slice.shape],
        "atol": float(args.atol),
        "allclose": allclose,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "p99_abs_diff": p99_abs_diff,
        "first_mismatch": first_mismatch,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if str(args.save_json).strip():
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

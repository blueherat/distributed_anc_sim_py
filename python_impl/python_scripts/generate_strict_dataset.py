from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from py_anc.acoustics import RIRManager
from py_anc.utils import wn_gen


DEFAULT_DATASET = ROOT_DIR / "python_scripts" / "strict_equiv_dataset.mat"


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(x), axis=0, keepdims=True)
    denom = np.where(denom < np.finfo(float).eps, 1.0, denom)
    return x / denom


def _build_manager(
    fs: int,
    compensate_fractional_delay: bool,
    fractional_delay_shift: int | None,
) -> tuple[RIRManager, np.ndarray, np.ndarray, np.ndarray]:
    mgr = RIRManager()
    mgr.fs = fs
    mgr.room = np.array([5.0, 5.0, 5.0], dtype=float)
    mgr.image_source_order = 2
    mgr.material_absorption = 0.5
    mgr.compensate_fractional_delay = bool(compensate_fractional_delay)
    mgr.fractional_delay_shift = None if fractional_delay_shift is None else int(fractional_delay_shift)

    center = mgr.room / 2.0

    mgr.add_primary_speaker(101, center + np.array([1.0, 0.0, 0.0]))

    ref_ids = np.array([401, 402, 403, 404], dtype=int)
    sec_ids = np.array([201, 202, 203, 204], dtype=int)
    err_ids = np.array([301, 302, 303, 304], dtype=int)

    r_ref = 0.9
    mgr.add_reference_microphone(int(ref_ids[0]), center + np.array([r_ref, 0.0, 0.0]))
    mgr.add_reference_microphone(int(ref_ids[1]), center - np.array([r_ref, 0.0, 0.0]))
    mgr.add_reference_microphone(int(ref_ids[2]), center + np.array([0.0, r_ref, 0.0]))
    mgr.add_reference_microphone(int(ref_ids[3]), center - np.array([0.0, r_ref, 0.0]))

    r1 = 0.6
    mgr.add_secondary_speaker(int(sec_ids[0]), center + np.array([r1, 0.0, 0.0]))
    mgr.add_secondary_speaker(int(sec_ids[1]), center - np.array([r1, 0.0, 0.0]))
    mgr.add_secondary_speaker(int(sec_ids[2]), center + np.array([0.0, r1, 0.0]))
    mgr.add_secondary_speaker(int(sec_ids[3]), center - np.array([0.0, r1, 0.0]))

    r2 = 0.3
    mgr.add_error_microphone(int(err_ids[0]), center + np.array([r2, 0.0, 0.0]))
    mgr.add_error_microphone(int(err_ids[1]), center - np.array([r2, 0.0, 0.0]))
    mgr.add_error_microphone(int(err_ids[2]), center + np.array([0.0, r2, 0.0]))
    mgr.add_error_microphone(int(err_ids[3]), center - np.array([0.0, r2, 0.0]))

    return mgr, ref_ids, sec_ids, err_ids


def _pack_secondary_rirs(mgr: RIRManager, sec_ids: np.ndarray, err_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_sec = len(sec_ids)
    n_err = len(err_ids)

    lengths = np.zeros((n_sec, n_err), dtype=np.int32)
    max_len = 0

    for i, sec_id in enumerate(sec_ids):
        for j, err_id in enumerate(err_ids):
            rir = np.asarray(mgr.get_secondary_rir(int(sec_id), int(err_id)), dtype=float)
            lengths[i, j] = int(len(rir))
            max_len = max(max_len, int(len(rir)))

    packed = np.zeros((n_sec, n_err, max_len), dtype=float)
    for i, sec_id in enumerate(sec_ids):
        for j, err_id in enumerate(err_ids):
            rir = np.asarray(mgr.get_secondary_rir(int(sec_id), int(err_id)), dtype=float)
            packed[i, j, : len(rir)] = rir

    return packed, lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared strict-equivalence dataset for MATLAB/Python.")
    parser.add_argument("--dataset-out", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fs", type=int, default=4000)
    parser.add_argument("--f-low", type=float, default=100.0)
    parser.add_argument("--f-high", type=float, default=1500.0)
    parser.add_argument("--L", type=int, default=1024)
    parser.add_argument("--mu", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-compensation", action="store_true")
    parser.add_argument("--fractional-delay-shift", type=int, default=None)
    args = parser.parse_args()

    mgr, ref_ids, sec_ids, err_ids = _build_manager(
        fs=args.fs,
        compensate_fractional_delay=not args.disable_compensation,
        fractional_delay_shift=args.fractional_delay_shift,
    )

    print("Building room impulse responses for strict dataset...")
    mgr.build(verbose=False)

    rng = np.random.default_rng(args.seed)
    noise, time_axis = wn_gen(args.fs, args.duration, args.f_low, args.f_high, rng=rng)
    source_signal = _normalize_columns(noise)

    d = mgr.calculate_desired_signal(source_signal, len(time_axis))
    x = mgr.calculate_reference_signal(source_signal, len(time_axis))
    x = _normalize_columns(x)

    sec_rirs, sec_rir_lengths = _pack_secondary_rirs(mgr, sec_ids, err_ids)

    out_path = Path(args.dataset_out)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "fs": np.array([[int(args.fs)]], dtype=np.int32),
        "duration_s": np.array([[float(args.duration)]], dtype=float),
        "f_low": np.array([[float(args.f_low)]], dtype=float),
        "f_high": np.array([[float(args.f_high)]], dtype=float),
        "L": np.array([[int(args.L)]], dtype=np.int32),
        "mu": np.array([[float(args.mu)]], dtype=float),
        "seed": np.array([[int(args.seed)]], dtype=np.int32),
        "compensate_fractional_delay": np.array([[int(not args.disable_compensation)]], dtype=np.int32),
        "fractional_delay_shift": np.array(
            [[-1 if args.fractional_delay_shift is None else int(args.fractional_delay_shift)]], dtype=np.int32
        ),
        "time": np.asarray(time_axis, dtype=float).reshape(-1, 1),
        "source_signal": np.asarray(source_signal, dtype=float),
        "reference_signal": np.asarray(x, dtype=float),
        "desired_signal": np.asarray(d, dtype=float),
        "ref_ids": np.asarray(ref_ids, dtype=np.int32).reshape(1, -1),
        "sec_ids": np.asarray(sec_ids, dtype=np.int32).reshape(1, -1),
        "err_ids": np.asarray(err_ids, dtype=np.int32).reshape(1, -1),
        "sec_rirs": np.asarray(sec_rirs, dtype=float),
        "sec_rir_lengths": np.asarray(sec_rir_lengths, dtype=np.int32),
    }

    savemat(str(out_path), payload, do_compression=True)
    print(f"Strict dataset saved to: {out_path}")


if __name__ == "__main__":
    main()

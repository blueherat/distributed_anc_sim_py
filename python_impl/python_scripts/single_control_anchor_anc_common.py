from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_repo_root() -> Path:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "python_impl").exists() and (parent / "README.md").exists():
            return parent
    raise FileNotFoundError("Could not locate repository root.")


ROOT = find_repo_root()
os.chdir(ROOT)
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from py_anc.algorithms import cfxlms as cfxlms_fn
from py_anc.utils import wn_gen
from python_scripts.cfxlms_single_control_dataset_impl import AcousticScenarioSampler, DatasetBuildConfig


def resolve_h5_path(explicit_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    env_path = os.environ.get("ANC_H5_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("python_impl") / "python_scripts" / "cfxlms_qc_dataset_single_control.h5")
    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
        checked.append(str(resolved))
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Single-control dataset HDF5 was not found. Checked: {checked}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(arr, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)).astype(np.float32)


def stats_general(arr: np.ndarray, train_idx: np.ndarray, reduce_axes: tuple[int, ...], eps: float = 1.0e-6) -> dict[str, np.ndarray]:
    train_arr = np.asarray(arr[train_idx], dtype=np.float32)
    mean = train_arr.mean(axis=reduce_axes, dtype=np.float64).astype(np.float32)
    std = train_arr.std(axis=reduce_axes, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return {"mean": mean, "std": std}


def normalize_columns(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    denom = np.max(np.abs(arr), axis=0, keepdims=True)
    denom = np.where(denom < np.finfo(float).eps, 1.0, denom)
    return arr / denom


def rolling_mse_db(sig: np.ndarray, fs: int, window_samples: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(sig, dtype=float).reshape(-1)
    if arr.size < int(window_samples):
        mse = float(np.mean(arr**2))
        return np.array([0.0], dtype=float), np.array([10.0 * np.log10(mse + np.finfo(float).eps)], dtype=float)
    kernel = np.ones((int(window_samples),), dtype=float) / float(window_samples)
    pow_s = np.convolve(arr**2, kernel, mode="valid")
    t = (np.arange(len(pow_s), dtype=float) + float(window_samples - 1)) / float(fs)
    return t, 10.0 * np.log10(pow_s + np.finfo(float).eps)


def build_laguerre_basis(tap_len: int, basis_dim: int, pole: float) -> np.ndarray:
    tap_len = int(tap_len)
    basis_dim = int(basis_dim)
    pole = float(pole)
    if tap_len <= 0:
        raise ValueError("tap_len must be positive.")
    if basis_dim <= 0:
        raise ValueError("basis_dim must be positive.")
    if not (0.0 < pole < 1.0):
        raise ValueError("pole must lie in (0, 1).")
    n = np.arange(tap_len, dtype=np.float64)
    basis = np.zeros((tap_len, basis_dim), dtype=np.float64)
    basis[:, 0] = np.sqrt(1.0 - pole * pole) * (pole**n)
    for k in range(1, basis_dim):
        basis[0, k] = np.sqrt(1.0 - pole * pole) * ((-pole) ** k)
        for t in range(1, tap_len):
            basis[t, k] = pole * basis[t - 1, k] + basis[t - 1, k - 1] - pole * basis[t, k - 1]
    q, _ = np.linalg.qr(basis)
    return q.astype(np.float32)


def compute_reference_covariance_features(x_ref: np.ndarray, n_fft: int) -> np.ndarray:
    arr = np.asarray(x_ref, dtype=np.float32)
    if arr.shape[0] != 3:
        raise ValueError(f"x_ref must have shape (3, T), got {arr.shape}.")
    spec = np.fft.rfft(arr, n=int(n_fft), axis=-1)
    scale = max(int(arr.shape[-1]), 1)
    auto = (spec * np.conj(spec)).real / float(scale)
    cross01 = spec[0] * np.conj(spec[1]) / float(scale)
    cross02 = spec[0] * np.conj(spec[2]) / float(scale)
    cross12 = spec[1] * np.conj(spec[2]) / float(scale)
    channels = [
        auto[0].real,
        auto[1].real,
        auto[2].real,
        cross01.real,
        cross02.real,
        cross12.real,
        cross01.imag,
        cross02.imag,
        cross12.imag,
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def compute_reference_covariance_batch(x_ref: np.ndarray, n_fft: int) -> np.ndarray:
    arr = np.asarray(x_ref, dtype=np.float32)
    out = np.zeros((arr.shape[0], 9, int(n_fft // 2 + 1)), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = compute_reference_covariance_features(arr[i], n_fft=int(n_fft))
    return out


def signed_log1p(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)


def compute_delay_summary_features(s_err: np.ndarray, e2r: np.ndarray, gcc_phat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s_err, dtype=np.float32)
    e = np.asarray(e2r, dtype=np.float32)
    gcc = np.asarray(gcc_phat, dtype=np.float32)
    s_peak_idx = np.argmax(np.abs(s[:, 0, :]), axis=1).astype(np.float32)
    s_peak_val = np.max(np.abs(s[:, 0, :]), axis=1).astype(np.float32)
    e_peak_idx = np.argmax(np.abs(e), axis=2).astype(np.float32)
    e_peak_val = np.max(np.abs(e), axis=2).astype(np.float32)
    e_early_energy = np.sqrt(np.mean(e[:, :, :64] ** 2, axis=2) + 1.0e-8).astype(np.float32)
    gcc_abs = np.abs(gcc)
    gcc_peak_idx = np.argmax(gcc_abs, axis=2).astype(np.float32)
    gcc_peak_val = np.max(gcc_abs, axis=2).astype(np.float32)
    gcc_center = float(gcc.shape[-1] // 2)
    gcc_peak_lag = (gcc_peak_idx - gcc_center).astype(np.float32)

    shared = np.concatenate(
        [
            s_peak_idx[:, None],
            s_peak_val[:, None],
            gcc_peak_lag,
            gcc_peak_val,
        ],
        axis=1,
    ).astype(np.float32)
    ref = np.stack(
        [
            e_peak_idx,
            e_peak_val,
            e_early_energy,
        ],
        axis=-1,
    ).astype(np.float32)
    return shared, ref


def fit_anchor_prior_ridge(
    delay_shared: np.ndarray,
    delay_ref: np.ndarray,
    anchors: np.ndarray,
    train_idx: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, np.ndarray | float]:
    shared = np.asarray(delay_shared, dtype=np.float32)
    ref = np.asarray(delay_ref, dtype=np.float32)
    y = np.asarray(anchors, dtype=np.float32)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    shared_mean = np.mean(shared[train_idx], axis=0, dtype=np.float64).astype(np.float32)
    shared_std = np.maximum(np.std(shared[train_idx], axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0e-6))
    ref_mean = np.mean(ref[train_idx], axis=(0, 1), dtype=np.float64).astype(np.float32)
    ref_std = np.maximum(np.std(ref[train_idx], axis=(0, 1), dtype=np.float64).astype(np.float32), np.float32(1.0e-6))
    shared_z = ((shared - shared_mean) / shared_std).astype(np.float32)
    ref_z = ((ref - ref_mean[None, None, :]) / ref_std[None, None, :]).astype(np.float32)
    weights = []
    biases = []
    n_feat = int(shared_z.shape[1] + ref_z.shape[2])
    eye = np.eye(n_feat + 1, dtype=np.float64)
    eye[0, 0] = 0.0
    for ref_idx in range(y.shape[1]):
        x_train = np.concatenate([shared_z[train_idx], ref_z[train_idx, ref_idx, :]], axis=1).astype(np.float64)
        x_aug = np.concatenate([np.ones((x_train.shape[0], 1), dtype=np.float64), x_train], axis=1)
        y_train = y[train_idx, ref_idx].astype(np.float64)
        beta = np.linalg.solve(x_aug.T @ x_aug + float(alpha) * eye, x_aug.T @ y_train)
        biases.append(np.float32(beta[0]))
        weights.append(beta[1:].astype(np.float32))
    return {
        "shared_mean": shared_mean,
        "shared_std": shared_std,
        "ref_mean": ref_mean,
        "ref_std": ref_std,
        "weights": np.stack(weights, axis=0).astype(np.float32),
        "bias": np.asarray(biases, dtype=np.float32),
        "alpha": float(alpha),
    }


def predict_anchor_prior_ridge(
    delay_shared: np.ndarray,
    delay_ref: np.ndarray,
    prior_fit: dict[str, np.ndarray | float],
    tap_len: int,
) -> np.ndarray:
    shared = np.asarray(delay_shared, dtype=np.float32)
    ref = np.asarray(delay_ref, dtype=np.float32)
    shared_z = ((shared - np.asarray(prior_fit["shared_mean"], dtype=np.float32)) / np.asarray(prior_fit["shared_std"], dtype=np.float32)).astype(np.float32)
    ref_z = (
        (ref - np.asarray(prior_fit["ref_mean"], dtype=np.float32)[None, None, :])
        / np.asarray(prior_fit["ref_std"], dtype=np.float32)[None, None, :]
    ).astype(np.float32)
    pred = np.zeros((shared.shape[0], ref.shape[1]), dtype=np.float32)
    for ref_idx in range(ref.shape[1]):
        x = np.concatenate([shared_z, ref_z[:, ref_idx, :]], axis=1).astype(np.float32)
        pred[:, ref_idx] = np.asarray(prior_fit["bias"], dtype=np.float32)[ref_idx] + x @ np.asarray(prior_fit["weights"], dtype=np.float32)[ref_idx]
    return np.clip(np.rint(pred).astype(np.int64), 0, int(tap_len) - 1)


def compute_geometry_anchor_indices(
    source_position: np.ndarray,
    ref_positions: np.ndarray,
    sec_positions: np.ndarray,
    err_positions: np.ndarray,
    sound_speed: np.ndarray,
    fs: int,
    tap_len: int,
) -> np.ndarray:
    src = np.asarray(source_position, dtype=np.float64)
    refs = np.asarray(ref_positions, dtype=np.float64)
    sec = np.asarray(sec_positions, dtype=np.float64)[:, 0, :]
    err = np.asarray(err_positions, dtype=np.float64)[:, 0, :]
    c = np.asarray(sound_speed, dtype=np.float64).reshape(-1, 1)
    source_err = np.linalg.norm(src - err, axis=1, keepdims=True)
    source_ref = np.linalg.norm(refs - src[:, None, :], axis=2)
    sec_err = np.linalg.norm(sec - err, axis=1, keepdims=True)
    delay = (source_err - source_ref - sec_err) / np.maximum(c, np.finfo(np.float64).eps)
    anchor = np.rint(delay * float(fs)).astype(np.int64)
    return np.clip(anchor, 0, int(tap_len) - 1)


def build_window_mask_np(anchors: np.ndarray, tap_len: int, half_width: int) -> np.ndarray:
    anc = np.asarray(anchors, dtype=np.int64)
    mask = np.zeros((anc.shape[0], anc.shape[1], int(tap_len)), dtype=bool)
    for offset in range(-int(half_width), int(half_width) + 1):
        idx = anc + int(offset)
        valid = (idx >= 0) & (idx < int(tap_len))
        batch_idx = np.broadcast_to(np.arange(anc.shape[0])[:, None], anc.shape)
        ref_idx = np.broadcast_to(np.arange(anc.shape[1])[None, :], anc.shape)
        mask[batch_idx[valid], ref_idx[valid], idx[valid]] = True
    return mask


def build_window_mask_torch(anchors: torch.Tensor, tap_len: int, half_width: int) -> torch.Tensor:
    anc = anchors.to(dtype=torch.long)
    mask = torch.zeros((anc.shape[0], anc.shape[1], int(tap_len)), dtype=torch.bool, device=anc.device)
    batch_idx = torch.arange(anc.shape[0], device=anc.device).view(-1, 1).expand_as(anc)
    ref_idx = torch.arange(anc.shape[1], device=anc.device).view(1, -1).expand_as(anc)
    for offset in range(-int(half_width), int(half_width) + 1):
        idx = anc + int(offset)
        valid = (idx >= 0) & (idx < int(tap_len))
        mask[batch_idx[valid], ref_idx[valid], idx[valid]] = True
    return mask


def extract_local_windows_np(w: np.ndarray, anchors: np.ndarray, half_width: int) -> np.ndarray:
    arr = np.asarray(w, dtype=np.float32)
    anc = np.asarray(anchors, dtype=np.int64)
    win_len = int(2 * int(half_width) + 1)
    out = np.zeros((arr.shape[0], arr.shape[1], win_len), dtype=np.float32)
    tap_len = int(arr.shape[-1])
    for offset, delta in enumerate(range(-int(half_width), int(half_width) + 1)):
        idx = anc + int(delta)
        valid = (idx >= 0) & (idx < tap_len)
        batch_idx = np.broadcast_to(np.arange(arr.shape[0])[:, None], anc.shape)
        ref_idx = np.broadcast_to(np.arange(arr.shape[1])[None, :], anc.shape)
        out[batch_idx[valid], ref_idx[valid], offset] = arr[batch_idx[valid], ref_idx[valid], idx[valid]]
    return out


def scatter_local_windows_torch(local_kernel: torch.Tensor, anchors: torch.Tensor, tap_len: int, half_width: int) -> torch.Tensor:
    out = torch.zeros((local_kernel.shape[0], local_kernel.shape[1], int(tap_len)), dtype=local_kernel.dtype, device=local_kernel.device)
    anc = anchors.to(dtype=torch.long)
    batch_idx = torch.arange(anc.shape[0], device=anc.device).view(-1, 1).expand_as(anc)
    ref_idx = torch.arange(anc.shape[1], device=anc.device).view(1, -1).expand_as(anc)
    for offset, delta in enumerate(range(-int(half_width), int(half_width) + 1)):
        idx = anc + int(delta)
        valid = (idx >= 0) & (idx < int(tap_len))
        out[batch_idx[valid], ref_idx[valid], idx[valid]] += local_kernel[:, :, offset][valid]
    return out


def project_tail_coeffs_np(w: np.ndarray, anchors: np.ndarray, basis: np.ndarray, half_width: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(w, dtype=np.float32)
    mask = build_window_mask_np(anchors, tap_len=arr.shape[-1], half_width=int(half_width))
    tail_target = np.where(mask, 0.0, arr).astype(np.float32)
    coeffs = np.einsum("nrt,td->nrd", tail_target, np.asarray(basis, dtype=np.float32), optimize=True).astype(np.float32)
    return tail_target, coeffs


def reconstruct_w_np(anchor_idx: np.ndarray, local_kernel: np.ndarray, tail_coeffs: np.ndarray, basis: np.ndarray, tap_len: int, half_width: int) -> np.ndarray:
    anc = np.asarray(anchor_idx, dtype=np.int64)
    local = np.asarray(local_kernel, dtype=np.float32)
    coeffs = np.asarray(tail_coeffs, dtype=np.float32)
    tail = np.einsum("nrd,td->nrt", coeffs, np.asarray(basis, dtype=np.float32), optimize=True).astype(np.float32)
    mask = build_window_mask_np(anc, tap_len=int(tap_len), half_width=int(half_width))
    tail = np.where(mask, 0.0, tail).astype(np.float32)
    full = np.zeros_like(tail)
    for offset, delta in enumerate(range(-int(half_width), int(half_width) + 1)):
        idx = anc + int(delta)
        valid = (idx >= 0) & (idx < int(tap_len))
        batch_idx = np.broadcast_to(np.arange(full.shape[0])[:, None], anc.shape)
        ref_idx = np.broadcast_to(np.arange(full.shape[1])[None, :], anc.shape)
        full[batch_idx[valid], ref_idx[valid], idx[valid]] += local[batch_idx[valid], ref_idx[valid], offset]
    return (full + tail).astype(np.float32)


def reconstruct_w_torch(anchor_idx: torch.Tensor, local_kernel: torch.Tensor, tail_coeffs: torch.Tensor, basis_t: torch.Tensor, tap_len: int, half_width: int) -> torch.Tensor:
    tail = torch.einsum("brd,td->brt", tail_coeffs, basis_t)
    mask = build_window_mask_torch(anchor_idx, tap_len=int(tap_len), half_width=int(half_width))
    tail = tail * (~mask).to(dtype=tail.dtype)
    local = scatter_local_windows_torch(local_kernel, anchor_idx, tap_len=int(tap_len), half_width=int(half_width))
    return local + tail


def anchor_metrics(pred_anchor: np.ndarray, true_anchor: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_anchor, dtype=np.int64)
    true = np.asarray(true_anchor, dtype=np.int64)
    abs_err = np.abs(pred - true)
    return {
        "anchor_exact": float(np.mean(abs_err == 0)),
        "anchor_within_1": float(np.mean(abs_err <= 1)),
        "anchor_mae": float(np.mean(abs_err)),
    }


def local_energy_capture(true_w: np.ndarray, pred_anchor: np.ndarray, half_width: int) -> float:
    arr = np.asarray(true_w, dtype=np.float32)
    mask = build_window_mask_np(pred_anchor, tap_len=arr.shape[-1], half_width=int(half_width))
    numerator = np.sum((arr**2) * mask.astype(np.float32), axis=-1)
    denominator = np.sum(arr**2, axis=-1)
    frac = numerator / np.maximum(denominator, np.finfo(np.float32).eps)
    return float(np.mean(frac))


def weighted_tap_mse(pred: torch.Tensor, target: torch.Tensor, tap_weights: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    return torch.mean(diff2 * tap_weights[None, None, :])


@dataclass
class AnchorDatasetBundle:
    ref_cov: np.ndarray
    s_err: np.ndarray
    e2r: np.ndarray
    s2r: np.ndarray
    gcc_phat: np.ndarray
    delay_shared: np.ndarray
    delay_ref: np.ndarray
    acoustic_shared: np.ndarray
    acoustic_ref: np.ndarray
    s2r_ref: np.ndarray
    oracle_shared: np.ndarray
    oracle_ref: np.ndarray
    w_target: np.ndarray
    w_full: np.ndarray
    anchors: np.ndarray
    local_target: np.ndarray
    tail_target: np.ndarray
    tail_coeffs: np.ndarray
    basis: np.ndarray
    room_data: dict[str, np.ndarray]
    source_seeds: np.ndarray
    cfg: DatasetBuildConfig
    meta: dict[str, Any]


def load_anchor_dataset(h5_path: Path, tail_basis_dim: int, local_half_width: int) -> AnchorDatasetBundle:
    with h5py.File(str(h5_path), "r") as h5:
        cfg = DatasetBuildConfig(**json.loads(h5.attrs["config_json"]))
        processed = h5["processed"]
        raw = h5["raw"]
        room = raw["room_params"]
        qc = raw["qc_metrics"]

        w_keep_len = int(processed.attrs["w_keep_len"])
        num_refs = int(cfg.num_reference_mics)
        ref_cov = compute_reference_covariance_batch(np.asarray(raw["x_ref"], dtype=np.float32), n_fft=int(cfg.psd_nfft))
        s_err = np.asarray(raw["S_paths"], dtype=np.float32)[:, None, :]
        e2r = np.asarray(raw["E2R_paths"], dtype=np.float32)
        s2r = np.asarray(raw["S2R_paths"], dtype=np.float32)
        gcc_phat = np.asarray(processed["gcc_phat"], dtype=np.float32)
        w_target = np.asarray(processed["w_targets"], dtype=np.float32).reshape(-1, num_refs, w_keep_len)
        w_full = np.asarray(raw["W_full"], dtype=np.float32)

        room_data = {
            "room_size": np.asarray(room["room_size"], dtype=np.float32),
            "source_position": np.asarray(room["source_position"], dtype=np.float32),
            "ref_positions": np.asarray(room["ref_positions"], dtype=np.float32),
            "sec_positions": np.asarray(room["sec_positions"], dtype=np.float32),
            "err_positions": np.asarray(room["err_positions"], dtype=np.float32),
            "sound_speed": np.asarray(room["sound_speed"], dtype=np.float32),
            "material_absorption": np.asarray(room["material_absorption"], dtype=np.float32),
            "image_source_order": np.asarray(room["image_source_order"], dtype=np.float32),
            "ref_radii": np.asarray(room["ref_radii"], dtype=np.float32),
            "ref_azimuth_deg": np.asarray(room["ref_azimuth_deg"], dtype=np.float32),
        }
        source_seeds = np.asarray(qc["source_seed"], dtype=np.int64)

    delay_shared, delay_ref = compute_delay_summary_features(s_err=s_err, e2r=e2r, gcc_phat=gcc_phat)
    anchors = compute_geometry_anchor_indices(
        source_position=room_data["source_position"],
        ref_positions=room_data["ref_positions"],
        sec_positions=room_data["sec_positions"],
        err_positions=room_data["err_positions"],
        sound_speed=room_data["sound_speed"],
        fs=int(cfg.fs),
        tap_len=int(w_keep_len),
    ).astype(np.int64)
    basis = build_laguerre_basis(tap_len=int(w_keep_len), basis_dim=int(tail_basis_dim), pole=0.55)
    local_target = extract_local_windows_np(w_target, anchors, half_width=int(local_half_width))
    tail_target, tail_coeffs = project_tail_coeffs_np(w_target, anchors, basis=basis, half_width=int(local_half_width))

    source_err = np.linalg.norm(room_data["source_position"] - room_data["err_positions"][:, 0, :], axis=1, keepdims=True).astype(np.float32)
    source_sec = np.linalg.norm(room_data["source_position"] - room_data["sec_positions"][:, 0, :], axis=1, keepdims=True).astype(np.float32)
    sec_err = np.linalg.norm(room_data["sec_positions"][:, 0, :] - room_data["err_positions"][:, 0, :], axis=1, keepdims=True).astype(np.float32)
    source_ref = np.linalg.norm(room_data["ref_positions"] - room_data["source_position"][:, None, :], axis=2).astype(np.float32)
    err_ref = np.linalg.norm(room_data["ref_positions"] - room_data["err_positions"], axis=2).astype(np.float32)
    sec_ref = np.linalg.norm(room_data["ref_positions"] - room_data["sec_positions"], axis=2).astype(np.float32)
    s_crop_len = int(min(96, s_err.shape[-1]))
    e_crop_len = int(min(96, e2r.shape[-1]))
    s2r_crop_len = int(min(96, s2r.shape[-1]))
    acoustic_shared = np.concatenate(
        [
            signed_log1p(ref_cov.reshape(ref_cov.shape[0], -1)),
            s_err[:, 0, :s_crop_len].astype(np.float32),
            delay_shared.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    acoustic_ref = np.concatenate(
        [
            e2r[:, :, :e_crop_len].astype(np.float32),
            delay_ref.astype(np.float32),
        ],
        axis=2,
    ).astype(np.float32)
    s2r_peak_idx = np.argmax(np.abs(s2r), axis=2).astype(np.float32)
    s2r_peak_val = np.max(np.abs(s2r), axis=2).astype(np.float32)
    s2r_ref = np.concatenate(
        [
            s2r[:, :, :s2r_crop_len].astype(np.float32),
            np.stack([s2r_peak_idx, s2r_peak_val], axis=-1).astype(np.float32),
        ],
        axis=2,
    ).astype(np.float32)

    oracle_shared = np.concatenate(
        [
            source_err,
            source_sec,
            sec_err,
            room_data["room_size"],
            room_data["material_absorption"][:, None],
            room_data["image_source_order"][:, None],
            room_data["sound_speed"][:, None],
        ],
        axis=1,
    ).astype(np.float32)
    oracle_ref = np.stack([source_ref, err_ref, sec_ref], axis=-1).astype(np.float32)

    meta = {
        "h5_path": str(h5_path),
        "num_refs": num_refs,
        "tap_len": int(w_keep_len),
        "local_half_width": int(local_half_width),
        "local_len": int(2 * local_half_width + 1),
        "tail_basis_dim": int(tail_basis_dim),
        "ref_cov_bins": int(ref_cov.shape[-1]),
        "rir_len": int(s_err.shape[-1]),
        "s_crop_len": int(s_crop_len),
        "e_crop_len": int(e_crop_len),
        "s2r_crop_len": int(s2r_crop_len),
        "acoustic_shared_dim": int(acoustic_shared.shape[-1]),
        "acoustic_ref_dim": int(acoustic_ref.shape[-1]),
    }
    return AnchorDatasetBundle(
        ref_cov=ref_cov,
        s_err=s_err,
        e2r=e2r,
        s2r=s2r,
        gcc_phat=gcc_phat,
        delay_shared=delay_shared,
        delay_ref=delay_ref,
        acoustic_shared=acoustic_shared,
        acoustic_ref=acoustic_ref,
        s2r_ref=s2r_ref,
        oracle_shared=oracle_shared,
        oracle_ref=oracle_ref,
        w_target=w_target,
        w_full=w_full,
        anchors=anchors,
        local_target=local_target,
        tail_target=tail_target,
        tail_coeffs=tail_coeffs,
        basis=basis,
        room_data=room_data,
        source_seeds=source_seeds,
        cfg=cfg,
        meta=meta,
    )


class PathEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.gn1 = nn.GroupNorm(1, 16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.gn2 = nn.GroupNorm(1, 32)
        self.conv3 = nn.Conv1d(32, 48, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(1, 48)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, int(out_dim))
        self.ln = nn.LayerNorm(int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.gn1(self.conv1(x)))
        z = F.gelu(self.gn2(self.conv2(z)))
        z = F.gelu(self.gn3(self.conv3(z)))
        z = self.pool(z).squeeze(-1)
        return F.gelu(self.ln(self.fc(z)))


class CovarianceEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(9, 32, kernel_size=5, stride=1, padding=2)
        self.gn1 = nn.GroupNorm(1, 32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.gn2 = nn.GroupNorm(1, 64)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(1, 96)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(96, int(out_dim))
        self.ln = nn.LayerNorm(int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.gn1(self.conv1(x)))
        z = F.gelu(self.gn2(self.conv2(z)))
        z = F.gelu(self.gn3(self.conv3(z)))
        z = self.pool(z).squeeze(-1)
        return F.gelu(self.ln(self.fc(z)))


class GeometrySharedEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, int(out_dim)),
            nn.LayerNorm(int(out_dim)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeometryRefEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, int(out_dim)),
            nn.LayerNorm(int(out_dim)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RefPredictionHeads(nn.Module):
    def __init__(self, in_dim: int, tap_len: int, local_len: int, tail_basis_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.anchor_head = nn.Linear(int(hidden_dim), int(tap_len))
        self.local_head = nn.Linear(int(hidden_dim), int(local_len))
        self.tail_head = nn.Linear(int(hidden_dim), int(tail_basis_dim))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk(x)
        return {
            "anchor_logits": self.anchor_head(z),
            "local_kernel": self.local_head(z),
            "tail_coeffs": self.tail_head(z),
        }


class AcousticDelayPriorANCNet(nn.Module):
    def __init__(
        self,
        tap_len: int,
        num_refs: int,
        local_len: int,
        tail_basis_dim: int,
        shared_in_dim: int,
        ref_in_dim: int,
        prior_radius: int = 12,
        shared_dim: int = 192,
        ref_dim: int = 96,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.tap_len = int(tap_len)
        self.num_refs = int(num_refs)
        self.prior_radius = int(prior_radius)
        self.shared_encoder = nn.Sequential(
            nn.Linear(int(shared_in_dim), int(shared_dim)),
            nn.LayerNorm(int(shared_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(shared_dim), int(shared_dim)),
            nn.LayerNorm(int(shared_dim)),
            nn.GELU(),
        )
        self.ref_encoder = nn.Sequential(
            nn.Linear(int(ref_in_dim) + 1, int(ref_dim)),
            nn.LayerNorm(int(ref_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ref_dim), int(ref_dim)),
            nn.LayerNorm(int(ref_dim)),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(int(shared_dim + ref_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.anchor_residual_head = nn.Linear(int(hidden_dim), int(2 * self.prior_radius + 1))
        self.local_head = nn.Linear(int(hidden_dim), int(local_len))
        self.tail_head = nn.Linear(int(hidden_dim), int(tail_basis_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                try:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                except Exception:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _scatter_prior_logits(self, prior_anchor: torch.Tensor, residual_logits: torch.Tensor) -> torch.Tensor:
        prior = prior_anchor.to(dtype=torch.long)
        full = torch.full(
            (prior.shape[0], prior.shape[1], int(self.tap_len)),
            fill_value=-12.0,
            dtype=residual_logits.dtype,
            device=residual_logits.device,
        )
        batch_idx = torch.arange(prior.shape[0], device=prior.device).view(-1, 1).expand_as(prior)
        ref_idx = torch.arange(prior.shape[1], device=prior.device).view(1, -1).expand_as(prior)
        for offset in range(-self.prior_radius, self.prior_radius + 1):
            cls_idx = int(offset + self.prior_radius)
            tap_idx = prior + int(offset)
            valid = (tap_idx >= 0) & (tap_idx < int(self.tap_len))
            full[batch_idx[valid], ref_idx[valid], tap_idx[valid]] = residual_logits[:, :, cls_idx][valid]
        return full

    def forward(self, acoustic_shared: torch.Tensor, acoustic_ref: torch.Tensor, anchor_prior: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = int(acoustic_shared.shape[0])
        z_shared = self.shared_encoder(acoustic_shared)[:, None, :].expand(-1, self.num_refs, -1)
        prior_norm = (2.0 * anchor_prior.to(dtype=acoustic_ref.dtype) / max(float(self.tap_len - 1), 1.0)) - 1.0
        ref_input = torch.cat([acoustic_ref, prior_norm.unsqueeze(-1)], dim=-1)
        z_ref = self.ref_encoder(ref_input.reshape(batch * self.num_refs, -1)).reshape(batch, self.num_refs, -1)
        z = self.trunk(torch.cat([z_shared, z_ref], dim=-1).reshape(batch * self.num_refs, -1)).reshape(batch, self.num_refs, -1)
        residual_logits = self.anchor_residual_head(z)
        return {
            "anchor_logits": self._scatter_prior_logits(prior_anchor=anchor_prior, residual_logits=residual_logits),
            "local_kernel": self.local_head(z),
            "tail_coeffs": self.tail_head(z),
        }


class AcousticPeakANCNet(nn.Module):
    def __init__(
        self,
        tap_len: int,
        num_refs: int,
        local_len: int,
        tail_basis_dim: int,
        include_s2r: bool = False,
        shared_dim: int = 128,
        path_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.tap_len = int(tap_len)
        self.num_refs = int(num_refs)
        self.include_s2r = bool(include_s2r)
        self.cov_encoder = CovarianceEncoder(out_dim=int(shared_dim))
        self.s_err_encoder = PathEncoder(out_dim=int(path_dim))
        self.e2r_encoder = PathEncoder(out_dim=int(path_dim))
        self.s2r_encoder = PathEncoder(out_dim=int(path_dim)) if self.include_s2r else None
        ref_in_dim = int(shared_dim + path_dim + path_dim + (path_dim if self.include_s2r else 0))
        self.heads = RefPredictionHeads(
            in_dim=ref_in_dim,
            tap_len=int(tap_len),
            local_len=int(local_len),
            tail_basis_dim=int(tail_basis_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                try:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                except Exception:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, ref_cov: torch.Tensor, s_err: torch.Tensor, e2r: torch.Tensor, s2r: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch = int(ref_cov.shape[0])
        z_cov = self.cov_encoder(ref_cov)
        z_s = self.s_err_encoder(s_err)
        z_cov_rep = z_cov[:, None, :].expand(-1, self.num_refs, -1)
        z_s_rep = z_s[:, None, :].expand(-1, self.num_refs, -1)
        e2r_flat = e2r.reshape(batch * self.num_refs, 1, e2r.shape[-1])
        z_e = self.e2r_encoder(e2r_flat).reshape(batch, self.num_refs, -1)
        ref_parts = [z_cov_rep, z_s_rep, z_e]
        if self.include_s2r and self.s2r_encoder is not None and s2r is not None:
            s2r_flat = s2r.reshape(batch * self.num_refs, 1, s2r.shape[-1])
            z_s2r = self.s2r_encoder(s2r_flat).reshape(batch, self.num_refs, -1)
            ref_parts.append(z_s2r)
        ref_context = torch.cat(ref_parts, dim=-1)
        out = self.heads(ref_context.reshape(batch * self.num_refs, -1))
        return {key: value.reshape(batch, self.num_refs, -1) for key, value in out.items()}


class GeometryOracleANCNet(nn.Module):
    def __init__(
        self,
        tap_len: int,
        num_refs: int,
        local_len: int,
        tail_basis_dim: int,
        shared_in_dim: int,
        ref_in_dim: int,
        shared_dim: int = 64,
        ref_dim: int = 32,
        hidden_dim: int = 192,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.tap_len = int(tap_len)
        self.num_refs = int(num_refs)
        self.shared_encoder = GeometrySharedEncoder(in_dim=int(shared_in_dim), out_dim=int(shared_dim))
        self.ref_encoder = GeometryRefEncoder(in_dim=int(ref_in_dim), out_dim=int(ref_dim))
        self.heads = RefPredictionHeads(
            in_dim=int(shared_dim + ref_dim),
            tap_len=int(tap_len),
            local_len=int(local_len),
            tail_basis_dim=int(tail_basis_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                try:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                except Exception:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, oracle_shared: torch.Tensor, oracle_ref: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = int(oracle_shared.shape[0])
        z_shared = self.shared_encoder(oracle_shared)[:, None, :].expand(-1, self.num_refs, -1)
        z_ref = self.ref_encoder(oracle_ref.reshape(batch * self.num_refs, -1)).reshape(batch, self.num_refs, -1)
        out = self.heads(torch.cat([z_shared, z_ref], dim=-1).reshape(batch * self.num_refs, -1))
        return {key: value.reshape(batch, self.num_refs, -1) for key, value in out.items()}


def build_model(
    model_kind: str,
    bundle: AnchorDatasetBundle,
    include_s2r: bool,
    dropout: float,
    acoustic_shared_dim: int | None = None,
    acoustic_ref_dim: int | None = None,
) -> nn.Module:
    if str(model_kind) == "acoustic":
        return AcousticDelayPriorANCNet(
            tap_len=int(bundle.meta["tap_len"]),
            num_refs=int(bundle.meta["num_refs"]),
            local_len=int(bundle.meta["local_len"]),
            tail_basis_dim=int(bundle.meta["tail_basis_dim"]),
            shared_in_dim=int(acoustic_shared_dim if acoustic_shared_dim is not None else bundle.meta["acoustic_shared_dim"]),
            ref_in_dim=int(acoustic_ref_dim if acoustic_ref_dim is not None else bundle.meta["acoustic_ref_dim"]),
            dropout=float(dropout),
        )
    if str(model_kind) == "oracle":
        return GeometryOracleANCNet(
            tap_len=int(bundle.meta["tap_len"]),
            num_refs=int(bundle.meta["num_refs"]),
            local_len=int(bundle.meta["local_len"]),
            tail_basis_dim=int(bundle.meta["tail_basis_dim"]),
            shared_in_dim=int(bundle.oracle_shared.shape[-1]),
            ref_in_dim=int(bundle.oracle_ref.shape[-1]),
            dropout=float(dropout),
        )
    raise ValueError(f"Unknown model_kind: {model_kind}")


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(n), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    split = int(int(n) * (1.0 - float(val_frac)))
    return np.sort(indices[:split]), np.sort(indices[split:])


@dataclass
class ReplayCase:
    idx: int
    manager: Any
    time_axis: np.ndarray
    reference_signal: np.ndarray
    desired_signal: np.ndarray
    w_h5: np.ndarray


def room_sample_from_bundle(bundle: AnchorDatasetBundle, idx: int) -> dict[str, Any]:
    return {
        "room_size": np.asarray(bundle.room_data["room_size"][idx], dtype=float),
        "source_pos": np.asarray(bundle.room_data["source_position"][idx], dtype=float),
        "ref_positions": np.asarray(bundle.room_data["ref_positions"][idx], dtype=float),
        "sec_positions": np.asarray(bundle.room_data["sec_positions"][idx], dtype=float),
        "err_positions": np.asarray(bundle.room_data["err_positions"][idx], dtype=float),
        "sound_speed": float(bundle.room_data["sound_speed"][idx]),
        "absorption": float(bundle.room_data["material_absorption"][idx]),
        "image_order": int(bundle.room_data["image_source_order"][idx]),
    }


def build_replay_cases(bundle: AnchorDatasetBundle, room_indices: list[int]) -> list[ReplayCase]:
    sampler = AcousticScenarioSampler(bundle.cfg, np.random.default_rng(int(bundle.cfg.random_seed)))
    cases: list[ReplayCase] = []
    for idx in room_indices:
        sampled = room_sample_from_bundle(bundle, int(idx))
        manager = sampler.build_manager(sampled)
        manager.build(verbose=False)
        source_seed = int(bundle.source_seeds[int(idx)])
        noise, t = wn_gen(
            fs=bundle.cfg.fs,
            duration=bundle.cfg.noise_duration_s,
            f_low=bundle.cfg.f_low,
            f_high=bundle.cfg.f_high,
            rng=np.random.default_rng(source_seed),
        )
        source_signal = normalize_columns(noise)
        time_axis = np.asarray(t[:, 0], dtype=float)
        desired_signal = manager.calculate_desired_signal(source_signal, len(time_axis))
        reference_signal = normalize_columns(manager.calculate_reference_signal(source_signal, len(time_axis)))
        cases.append(
            ReplayCase(
                idx=int(idx),
                manager=manager,
                time_axis=time_axis,
                reference_signal=reference_signal,
                desired_signal=desired_signal,
                w_h5=np.asarray(bundle.w_full[int(idx)], dtype=np.float32),
            )
        )
    return cases


def build_weight_tensor_from_w_full(w_full: np.ndarray, filter_len: int, n_ref: int, n_sec: int) -> np.ndarray:
    w = np.zeros((filter_len, n_ref, n_sec), dtype=float)
    arr = np.asarray(w_full, dtype=float)
    for k in range(min(n_sec, arr.shape[0])):
        keep_ref = min(n_ref, arr.shape[1])
        keep_len = min(filter_len, arr.shape[2])
        w[:keep_len, :keep_ref, k] = arr[k, :keep_ref, :keep_len].T
    return w


def cfxlms_with_init(time: np.ndarray, rir_manager: Any, filter_len: int, mu: float, reference_signal: np.ndarray, desired_signal: np.ndarray, w_init: np.ndarray | None = None, normalized_update: bool = False, norm_epsilon: float = 1.0e-8) -> dict[str, Any]:
    x = np.asarray(reference_signal, dtype=float)
    d = np.asarray(desired_signal, dtype=float)
    key_sec = list(rir_manager.secondary_speakers.keys())
    key_err = list(rir_manager.error_microphones.keys())
    n_ref = len(rir_manager.reference_microphones)
    n_sec = len(key_sec)
    n_err = len(key_err)
    n_samples = len(time)

    s_paths = [[None for _ in range(n_err)] for _ in range(n_sec)]
    s_lens = np.zeros((n_sec, n_err), dtype=np.int32)
    max_ls = 0
    for k, sec_id in enumerate(key_sec):
        for m, err_id in enumerate(key_err):
            s = np.asarray(rir_manager.get_secondary_rir(sec_id, err_id), dtype=float)
            s_paths[k][m] = s
            s_lens[k, m] = int(s.size)
            max_ls = max(max_ls, int(s.size))

    w = np.zeros((int(filter_len), n_ref, n_sec), dtype=float) if w_init is None else build_weight_tensor_from_w_full(w_init, int(filter_len), n_ref, n_sec)
    x_taps = np.zeros((max(int(filter_len), max_ls), n_ref), dtype=float)
    xf_taps = np.zeros((int(filter_len), n_ref, n_sec, n_err), dtype=float)
    y_taps = [np.zeros(int(np.max(s_lens[k])), dtype=float) for k in range(n_sec)]
    e = np.zeros((n_samples, n_err), dtype=float)

    for n in range(n_samples):
        x_taps[1:, :] = x_taps[:-1, :]
        x_taps[0, :] = x[n, :]
        for k in range(n_sec):
            y = np.sum(w[:, :, k] * x_taps[: int(filter_len), :])
            y_taps[k][1:] = y_taps[k][:-1]
            y_taps[k][0] = y
        for m in range(n_err):
            yf = 0.0
            for k in range(n_sec):
                s = s_paths[k][m]
                yf += float(np.dot(s, y_taps[k][: int(s_lens[k, m])]))
            e[n, m] = d[n, m] + yf
        xf_taps[1:, :, :, :] = xf_taps[:-1, :, :, :]
        for k in range(n_sec):
            for m in range(n_err):
                s = s_paths[k][m]
                xf_taps[0, :, k, m] = s @ x_taps[: int(s_lens[k, m]), :]
        for k in range(n_sec):
            grad_k = np.zeros((int(filter_len), n_ref), dtype=float)
            for m in range(n_err):
                phi = xf_taps[:, :, k, m]
                if normalized_update:
                    grad_k += (phi * e[n, m]) / (float(np.sum(phi * phi)) + float(norm_epsilon))
                else:
                    grad_k += phi * e[n, m]
            w[:, :, k] = w[:, :, k] - float(mu) * grad_k

    return {"err_hist": e}


def replay_metrics_for_case(case: ReplayCase, w_ai: np.ndarray, cfg: DatasetBuildConfig, early_window_s: float) -> dict[str, float]:
    mu = float(cfg.mu_candidates[0])
    params = {
        "time": case.time_axis,
        "rir_manager": case.manager,
        "L": int(cfg.filter_len),
        "mu": mu,
        "reference_signal": case.reference_signal,
        "desired_signal": case.desired_signal,
        "verbose": False,
        "normalized_update": bool(cfg.anc_normalized_update),
        "norm_epsilon": float(cfg.anc_norm_epsilon),
    }
    e_zero = np.asarray(cfxlms_fn(params)["err_hist"], dtype=float)[:, 0]
    e_ai = np.asarray(
        cfxlms_with_init(
            case.time_axis,
            case.manager,
            int(cfg.filter_len),
            mu,
            case.reference_signal,
            case.desired_signal,
            w_init=w_ai,
            normalized_update=bool(cfg.anc_normalized_update),
            norm_epsilon=float(cfg.anc_norm_epsilon),
        )["err_hist"],
        dtype=float,
    )[:, 0]
    e_h5 = np.asarray(
        cfxlms_with_init(
            case.time_axis,
            case.manager,
            int(cfg.filter_len),
            mu,
            case.reference_signal,
            case.desired_signal,
            w_init=case.w_h5,
            normalized_update=bool(cfg.anc_normalized_update),
            norm_epsilon=float(cfg.anc_norm_epsilon),
        )["err_hist"],
        dtype=float,
    )[:, 0]
    window_samples = min(max(32, int(round(float(early_window_s) * float(cfg.fs)))), max(int(len(case.time_axis) // 2), 32))
    t_db, db_zero = rolling_mse_db(e_zero, int(cfg.fs), window_samples=window_samples)
    _, db_ai = rolling_mse_db(e_ai, int(cfg.fs), window_samples=window_samples)
    _, db_h5 = rolling_mse_db(e_h5, int(cfg.fs), window_samples=window_samples)
    early_mask = t_db <= float(early_window_s)
    if not np.any(early_mask):
        early_mask = np.ones_like(t_db, dtype=bool)
    ai_vs_zero = float(np.mean(db_zero[early_mask] - db_ai[early_mask]))
    h5_vs_zero = float(np.mean(db_zero[early_mask] - db_h5[early_mask]))
    ai_to_h5_gap = float(np.mean(np.abs(db_ai[early_mask] - db_h5[early_mask])))
    return {
        "ai_vs_zero_db": ai_vs_zero,
        "h5_vs_zero_db": h5_vs_zero,
        "ai_to_h5_gap_db": ai_to_h5_gap,
    }

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.linalg import convolution_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from py_anc.acoustics import RIRManager
from py_anc.algorithms import cfxlms
from py_anc.utils import wn_gen


class QCError(RuntimeError):
    """Raised when a sampled room fails geometry / RIR / ANC quality control."""


@dataclass
class DatasetBuildConfig:
    target_rooms: int = 1000
    max_total_attempts: int = 30000

    fs: int = 4000
    noise_duration_s: float = 7.0
    f_low: float = 100.0
    f_high: float = 1500.0

    filter_len: int = 512
    ref_window_ms: float = 50.0
    warmup_start_s_min: float = 0.5
    warmup_start_s_max: float = 2.0

    room_size_min: float = 3.0
    room_size_max: float = 5.0
    wall_margin: float = 0.25
    source_wall_margin: float = 0.85

    sound_speed_min: float = 338.0
    sound_speed_max: float = 346.0
    reflection_room_probability: float = 0.55
    direct_room_absorption_range: tuple[float, float] = (0.22, 0.32)
    reflection_room_absorption_range: tuple[float, float] = (0.20, 0.30)
    direct_room_image_order_choices: tuple[int, ...] = (0, 1)
    direct_room_image_order_probs: tuple[float, ...] = (0.25, 0.75)
    reflection_room_image_order_choices: tuple[int, ...] = (1, 2)
    reflection_room_image_order_probs: tuple[float, ...] = (0.80, 0.20)
    enable_air_absorption: bool = True

    num_reference_mics: int = 3
    num_secondary_speakers: int = 1
    num_error_mics: int = 1

    min_reference_pair_distance: float = 0.22
    min_source_reference_distance: float = 0.28
    min_source_device_distance: float = 0.55
    min_reference_device_distance: float = 0.35
    min_primary_advance_margin_m: float = 0.35
    min_secondary_feedback_margin_m: float = 0.08
    min_noise_source_ref_center_distance: float = 0.50

    ref_center_jitter_xy: float = 0.45
    ref_center_height_range: tuple[float, float] = (1.05, 1.70)
    ref_triangle_min_angle_deg: float = 26.0
    ref_triangle_max_angle_deg: float = 88.0
    sec_center_distance_range: tuple[float, float] = (0.90, 1.70)
    err_center_distance_range: tuple[float, float] = (1.15, 2.20)
    min_err_sec_gap: float = 0.12
    collinearity_max_sine: float = 0.08
    line_to_ref_min_distance: float = 0.18
    triangle_outside_margin: float = 0.04

    layout_mode_choices: tuple[str, ...] = ("free_far", "wall_side", "corner_far")
    layout_mode_probs: tuple[float, ...] = (0.45, 0.30, 0.25)

    reference_radius_range: tuple[float, float] = (0.30, 0.75)
    reference_azimuth_base: tuple[float, ...] = (0.0, 120.0, 240.0)
    reference_azimuth_jitter: float = 24.0
    reference_z_offset_range: tuple[float, float] = (-0.10, 0.10)
    reference_height_range: tuple[float, float] = (0.95, 1.65)

    secondary_height_range: tuple[float, float] = (0.95, 1.75)
    secondary_source_distance_range: tuple[float, float] = (1.00, 3.40)
    sec_err_distance_range: tuple[float, float] = (0.12, 0.35)

    mu_candidates: tuple[float, ...] = (1.0e-4,)
    min_nr_last_db: float = 15.0
    min_nr_gain_db: float = 15.0
    anc_normalized_update: bool = False
    anc_norm_epsilon: float = 1.0e-8

    direct_energy_half_window: int = 2
    min_direct_ratio: float = 0.08
    min_control_ratio: float = 0.10

    rir_store_len: int = 512
    gcc_truncated_len: int = 129
    gcc_delay_padding: int = 5
    gcc_use_geometry_window: bool = True
    psd_nfft: int = 256

    acoustic_feature_nfft: int = 512
    acoustic_feature_low_hz: float = 0.0
    acoustic_feature_high_hz: float = 1000.0

    truncate_energy_ratio: float = 0.90
    truncate_length_quantile: float = 0.95
    min_path_keep_len: int = 48
    max_path_keep_len: int = 256
    w_truncate_energy_ratio: float = 0.98
    min_w_keep_len: int = 96
    max_w_keep_len: int = 384

    canonical_q_energy_ratio: float = 0.995
    canonical_q_length_quantile: float = 0.95
    canonical_q_min_keep_len: int = 64
    canonical_q_max_keep_len: int = 512
    canonical_q_local_half_width: int = 7
    canonical_q_tail_basis_dim: int = 12
    canonical_lambda_q_scale_candidates: tuple[float, ...] = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
    canonical_lambda_w_candidates: tuple[float, ...] = (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
    canonical_calibration_rooms: int = 64
    canonical_replay_early_window_s: float = 0.25
    canonical_r2r_pair_order: tuple[str, ...] = ("01", "02", "12")

    random_seed: int = 20260331
    progress_interval: int = 20
    attempt_log_interval: int = 10
    layout_preview: bool = True
    layout_preview_interval: int = 1

    output_h5: str = str(ROOT_DIR / "python_scripts" / "cfxlms_qc_dataset_single_control.h5")

    @property
    def ref_window_samples(self) -> int:
        return int(round(self.fs * self.ref_window_ms / 1000.0))


@dataclass
class RoomSample:
    room_params: dict[str, Any]
    x_ref: np.ndarray
    secondary_path: np.ndarray
    secondary_path_length: int
    primary_to_ref_paths: np.ndarray
    primary_to_ref_lengths: np.ndarray
    primary_to_error_path: np.ndarray
    primary_to_error_length: int
    error_to_ref_paths: np.ndarray
    error_to_ref_lengths: np.ndarray
    secondary_to_ref_paths: np.ndarray
    secondary_to_ref_lengths: np.ndarray
    ref_to_ref_paths: np.ndarray
    ref_to_ref_lengths: np.ndarray
    w_opt: np.ndarray
    w_full: np.ndarray
    qc_metrics: dict[str, float]


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(x), axis=0, keepdims=True)
    denom = np.where(denom < np.finfo(float).eps, 1.0, denom)
    return x / denom


def _min_pairwise_distance(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return float("inf")
    min_dist = float("inf")
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            min_dist = min(min_dist, float(np.linalg.norm(pts[i] - pts[j])))
    return min_dist


def _triangle_angles_deg(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(3, 3)
    out = np.zeros((3,), dtype=float)
    for i in range(3):
        a = pts[(i + 1) % 3] - pts[i]
        b = pts[(i + 2) % 3] - pts[i]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= np.finfo(float).eps:
            out[i] = 180.0
            continue
        cosine = float(np.dot(a, b) / denom)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        out[i] = float(np.degrees(np.arccos(cosine)))
    return out


def _point_in_triangle_xy(point: np.ndarray, triangle: np.ndarray) -> bool:
    p = np.asarray(point, dtype=float).reshape(2)
    tri = np.asarray(triangle, dtype=float).reshape(3, 2)
    a = tri[0]
    b = tri[1]
    c = tri[2]
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(den) <= np.finfo(float).eps:
        return False
    inv_den = 1.0 / den
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) * inv_den)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) * inv_den)
    return bool(u >= 0.0 and v >= 0.0 and (u + v) <= 1.0)


def _point_to_segment_distance_xy(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    p = np.asarray(point, dtype=float).reshape(2)
    a = np.asarray(seg_a, dtype=float).reshape(2)
    b = np.asarray(seg_b, dtype=float).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= np.finfo(float).eps:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _point_to_line_distance(point: np.ndarray, line_a: np.ndarray, line_b: np.ndarray) -> float:
    p = np.asarray(point, dtype=float).reshape(3)
    a = np.asarray(line_a, dtype=float).reshape(3)
    b = np.asarray(line_b, dtype=float).reshape(3)
    ab = b - a
    norm_ab = float(np.linalg.norm(ab))
    if norm_ab <= np.finfo(float).eps:
        return float(np.linalg.norm(p - a))
    cross = np.cross(p - a, ab)
    return float(np.linalg.norm(cross) / norm_ab)


def _crop_rir(rir: np.ndarray, keep_len: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(rir, dtype=float).reshape(-1)
    keep = min(int(keep_len), arr.size)
    out = np.zeros((int(keep_len),), dtype=np.float32)
    if keep > 0:
        out[:keep] = arr[:keep].astype(np.float32)
    return out, int(keep)


def _energy_keep_lengths(arr: np.ndarray, energy_ratio: float) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    lengths = np.ones((x.shape[0],), dtype=np.int32)
    for i, row in enumerate(x):
        energy = np.cumsum(row**2)
        total = float(energy[-1]) if energy.size else 0.0
        if total <= np.finfo(float).eps:
            lengths[i] = 1
            continue
        threshold = float(energy_ratio) * total
        lengths[i] = int(np.searchsorted(energy, threshold, side="left") + 1)
    return lengths


def _choose_keep_len(
    arr: np.ndarray,
    energy_ratio: float,
    quantile: float,
    min_keep: int,
    max_keep: int,
) -> int:
    lengths = _energy_keep_lengths(arr, energy_ratio)
    keep = int(np.ceil(np.quantile(lengths.astype(np.float64), float(quantile))))
    keep = max(int(min_keep), min(int(max_keep), int(keep)))
    return keep


def _next_pow2(n: int) -> int:
    return 1 if int(n) <= 1 else 1 << (int(n) - 1).bit_length()


def _decode_layout_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _window_mask_1d_np(anchors: np.ndarray, tap_len: int, half_width: int) -> np.ndarray:
    anc = np.asarray(anchors, dtype=np.int64).reshape(-1)
    mask = np.zeros((anc.shape[0], int(tap_len)), dtype=bool)
    row_idx = np.arange(anc.shape[0], dtype=np.int64)
    for offset in range(-int(half_width), int(half_width) + 1):
        idx = anc + int(offset)
        valid = (idx >= 0) & (idx < int(tap_len))
        mask[row_idx[valid], idx[valid]] = True
    return mask


def _extract_local_windows_1d_np(q: np.ndarray, anchors: np.ndarray, half_width: int) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float32)
    anc = np.asarray(anchors, dtype=np.int64).reshape(-1)
    win_len = int(2 * int(half_width) + 1)
    out = np.zeros((arr.shape[0], win_len), dtype=np.float32)
    tap_len = int(arr.shape[-1])
    row_idx = np.arange(arr.shape[0], dtype=np.int64)
    for offset, delta in enumerate(range(-int(half_width), int(half_width) + 1)):
        idx = anc + int(delta)
        valid = (idx >= 0) & (idx < tap_len)
        out[row_idx[valid], offset] = arr[row_idx[valid], idx[valid]]
    return out


def _build_laguerre_basis(tap_len: int, basis_dim: int, pole: float) -> np.ndarray:
    tap_len = int(tap_len)
    basis_dim = int(basis_dim)
    pole = float(pole)
    if tap_len <= 0 or basis_dim <= 0:
        raise ValueError("tap_len and basis_dim must be positive.")
    if not (0.0 < pole < 1.0):
        raise ValueError("pole must be in (0, 1).")
    n = np.arange(tap_len, dtype=np.float64)
    basis = np.zeros((tap_len, basis_dim), dtype=np.float64)
    basis[:, 0] = np.sqrt(1.0 - pole * pole) * (pole**n)
    for k in range(1, basis_dim):
        basis[0, k] = np.sqrt(1.0 - pole * pole) * ((-pole) ** k)
        for t in range(1, tap_len):
            basis[t, k] = pole * basis[t - 1, k] + basis[t - 1, k - 1] - pole * basis[t, k - 1]
    q, _ = np.linalg.qr(basis)
    return q.astype(np.float32)


def _project_tail_coeffs_1d_np(q: np.ndarray, anchors: np.ndarray, basis: np.ndarray, half_width: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(q, dtype=np.float32)
    mask = _window_mask_1d_np(anchors, tap_len=arr.shape[-1], half_width=int(half_width))
    tail_target = np.where(mask, 0.0, arr).astype(np.float32)
    coeffs = np.einsum("nt,td->nd", tail_target, np.asarray(basis, dtype=np.float32), optimize=True).astype(np.float32)
    return tail_target, coeffs


def _rolling_mse_db(sig: np.ndarray, fs: int, window_samples: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(sig, dtype=float).reshape(-1)
    if arr.size < int(window_samples):
        mse = float(np.mean(arr**2))
        return np.array([0.0], dtype=float), np.array([10.0 * np.log10(mse + np.finfo(float).eps)], dtype=float)
    kernel = np.ones((int(window_samples),), dtype=float) / float(window_samples)
    pow_s = np.convolve(arr**2, kernel, mode="valid")
    t = (np.arange(len(pow_s), dtype=float) + float(window_samples - 1)) / float(fs)
    return t, 10.0 * np.log10(pow_s + np.finfo(float).eps)


def _build_weight_tensor_from_w(w_init: np.ndarray | None, filter_len: int, num_refs: int, num_secs: int) -> np.ndarray:
    w = np.zeros((int(filter_len), int(num_refs), int(num_secs)), dtype=float)
    if w_init is None:
        return w
    arr = np.asarray(w_init, dtype=float)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == int(num_secs):
            secs = min(int(num_secs), int(arr.shape[0]))
            refs = min(int(num_refs), int(arr.shape[1]))
            taps = min(int(filter_len), int(arr.shape[2]))
            for k in range(secs):
                w[:taps, :refs, k] = arr[k, :refs, :taps].T
            return w
        if arr.shape[-1] == int(num_secs):
            refs = min(int(num_refs), int(arr.shape[0]))
            taps = min(int(filter_len), int(arr.shape[1]))
            secs = min(int(num_secs), int(arr.shape[2]))
            for k in range(secs):
                w[:taps, :refs, k] = arr[:refs, :taps, k].T
            return w
        raise ValueError(f"Unexpected 3D init weight shape: {arr.shape}")
    if arr.ndim == 2:
        refs = min(int(num_refs), int(arr.shape[0]))
        taps = min(int(filter_len), int(arr.shape[1]))
        w[:taps, :refs, 0] = arr[:refs, :taps].T
        return w
    raise ValueError(f"Unexpected init weight shape: {arr.shape}")
    return w


def _cfxlms_with_init(
    time_axis: np.ndarray,
    rir_manager: RIRManager,
    filter_len: int,
    mu: float,
    reference_signal: np.ndarray,
    desired_signal: np.ndarray,
    w_init: np.ndarray | None = None,
    normalized_update: bool = False,
    norm_epsilon: float = 1.0e-8,
) -> dict[str, Any]:
    x = np.asarray(reference_signal, dtype=float)
    d = np.asarray(desired_signal, dtype=float)
    key_sec = list(rir_manager.secondary_speakers.keys())
    key_err = list(rir_manager.error_microphones.keys())
    num_refs = len(rir_manager.reference_microphones)
    num_secs = len(key_sec)
    num_errs = len(key_err)
    n_samples = len(time_axis)

    s_paths = [[None for _ in range(num_errs)] for _ in range(num_secs)]
    s_lens = np.zeros((num_secs, num_errs), dtype=np.int32)
    max_ls = 0
    for k, sec_id in enumerate(key_sec):
        for m, err_id in enumerate(key_err):
            s = np.asarray(rir_manager.get_secondary_rir(sec_id, err_id), dtype=float)
            s_paths[k][m] = s
            s_lens[k, m] = int(s.size)
            max_ls = max(max_ls, int(s.size))

    w = _build_weight_tensor_from_w(w_init, int(filter_len), num_refs, num_secs)
    x_taps = np.zeros((max(int(filter_len), max_ls), num_refs), dtype=float)
    xf_taps = np.zeros((int(filter_len), num_refs, num_secs, num_errs), dtype=float)
    y_taps = [np.zeros(int(np.max(s_lens[k])), dtype=float) for k in range(num_secs)]
    e = np.zeros((n_samples, num_errs), dtype=float)

    for n in range(n_samples):
        x_taps[1:, :] = x_taps[:-1, :]
        x_taps[0, :] = x[n, :]
        for k in range(num_secs):
            y = np.sum(w[:, :, k] * x_taps[: int(filter_len), :])
            y_taps[k][1:] = y_taps[k][:-1]
            y_taps[k][0] = y
        for m in range(num_errs):
            yf = 0.0
            for k in range(num_secs):
                s = s_paths[k][m]
                yf += float(np.dot(s, y_taps[k][: int(s_lens[k, m])]))
            e[n, m] = d[n, m] + yf
        xf_taps[1:, :, :, :] = xf_taps[:-1, :, :, :]
        for k in range(num_secs):
            for m in range(num_errs):
                s = s_paths[k][m]
                xf_taps[0, :, k, m] = s @ x_taps[: int(s_lens[k, m]), :]
        for k in range(num_secs):
            grad_k = np.zeros((int(filter_len), num_refs), dtype=float)
            for m in range(num_errs):
                phi = xf_taps[:, :, k, m]
                if normalized_update:
                    denom = float(np.sum(phi * phi)) + float(norm_epsilon)
                    grad_k += (phi * e[n, m]) / denom
                else:
                    grad_k += phi * e[n, m]
            w[:, :, k] = w[:, :, k] - float(mu) * grad_k
    return {"err_hist": e}


def _canonical_q_from_paths(d_path: np.ndarray, s_path: np.ndarray, q_full_len: int, lambda_q_scale: float) -> np.ndarray:
    d = np.asarray(d_path, dtype=np.float64).reshape(-1)
    s = np.asarray(s_path, dtype=np.float64).reshape(-1)
    n_fft = _next_pow2(int(q_full_len) + max(len(s), len(d)) - 1)
    d_f = np.fft.rfft(d, n=n_fft)
    s_f = np.fft.rfft(s, n=n_fft)
    lam_q = float(lambda_q_scale) * max(float(np.max(np.abs(s_f) ** 2)), 1.0e-12)
    q_f = -(np.conj(s_f) * d_f) / (np.abs(s_f) ** 2 + lam_q)
    q = np.fft.irfft(q_f, n=n_fft)[: int(q_full_len)]
    return q.astype(np.float32)


def _canonical_conv_matrix(p_ref_paths: np.ndarray, filter_len: int, q_full_len: int) -> np.ndarray:
    p = np.asarray(p_ref_paths, dtype=np.float64)
    mats = [convolution_matrix(p_i, int(filter_len))[: int(q_full_len), :] for p_i in p]
    return np.concatenate(mats, axis=1)


def _equivalent_q_from_w_full(w_full: np.ndarray, p_ref_paths: np.ndarray, q_full_len: int, filter_len: int) -> np.ndarray:
    arr = np.asarray(w_full, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Unexpected 3D W shape for equivalent Q construction: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Unexpected W shape for equivalent Q construction: {arr.shape}")
    a_mat = _canonical_conv_matrix(p_ref_paths, int(filter_len), int(q_full_len))
    return (a_mat @ arr.reshape(-1)).astype(np.float32)


def _solve_w_canonical_from_q(q_full: np.ndarray, p_ref_paths: np.ndarray, filter_len: int, lambda_w: float, q_full_len: int) -> np.ndarray:
    q = np.asarray(q_full, dtype=np.float64).reshape(-1)[: int(q_full_len)]
    a_mat = _canonical_conv_matrix(p_ref_paths, int(filter_len), int(q_full_len))
    ata = a_mat.T @ a_mat
    rhs = a_mat.T @ q
    w_flat = np.linalg.solve(ata + float(lambda_w) * np.eye(ata.shape[0], dtype=np.float64), rhs)
    w = np.zeros((1, p_ref_paths.shape[0], int(filter_len)), dtype=np.float32)
    w[0] = w_flat.reshape(p_ref_paths.shape[0], int(filter_len)).astype(np.float32)
    return w


class FeatureProcessor:
    def __init__(self, cfg: DatasetBuildConfig):
        self.cfg = cfg
        self.pairs = [(0, 1), (0, 2), (1, 2)]
        self.acoustic_nfft = int(cfg.acoustic_feature_nfft)
        self.acoustic_freq = np.fft.rfftfreq(self.acoustic_nfft, d=1.0 / float(cfg.fs))
        self.acoustic_mask = (self.acoustic_freq >= float(cfg.acoustic_feature_low_hz)) & (
            self.acoustic_freq <= float(cfg.acoustic_feature_high_hz)
        )
        if not np.any(self.acoustic_mask):
            raise ValueError("acoustic feature mask is empty. Please adjust acoustic_feature_low_hz/high_hz.")

    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 if n <= 1 else 1 << (n - 1).bit_length()

    @staticmethod
    def _central_crop(x: np.ndarray, target_len: int) -> np.ndarray:
        if x.size == target_len:
            return x
        if x.size < target_len:
            pad_total = target_len - x.size
            left = pad_total // 2
            right = pad_total - left
            return np.pad(x, (left, right), mode="constant")
        mid = x.size // 2
        half = target_len // 2
        if target_len % 2:
            return x[mid - half : mid + half + 1]
        return x[mid - half : mid + half]

    @staticmethod
    def _resolve_max_delay_samples(
        ref_a: np.ndarray,
        ref_b: np.ndarray,
        sound_speed: float,
        fs: int,
    ) -> int:
        dist = float(np.linalg.norm(np.asarray(ref_a, dtype=float) - np.asarray(ref_b, dtype=float)))
        delay_s = dist / max(float(sound_speed), 1.0e-6)
        return int(np.ceil(delay_s * float(fs)))

    def gcc_phat_pair(self, sig_a: np.ndarray, sig_b: np.ndarray, max_delay_samples: int | None = None) -> np.ndarray:
        n = int(sig_a.size + sig_b.size - 1)
        n_fft = self._next_pow2(n)
        sig_a_f = np.fft.rfft(sig_a, n=n_fft)
        sig_b_f = np.fft.rfft(sig_b, n=n_fft)
        cross = sig_a_f * np.conj(sig_b_f)
        cross /= np.maximum(np.abs(cross), np.finfo(float).eps)
        corr = np.fft.irfft(cross, n=n_fft)
        corr = np.concatenate([corr[-(n_fft // 2) :], corr[: n_fft // 2]])

        if bool(self.cfg.gcc_use_geometry_window) and max_delay_samples is not None:
            half_span = int(max(0, int(max_delay_samples)) + int(self.cfg.gcc_delay_padding))
            geom_len = int(2 * half_span + 1)
            corr = self._central_crop(corr.astype(np.float64, copy=False), geom_len)
        return self._central_crop(corr.astype(np.float64, copy=False), self.cfg.gcc_truncated_len)

    def compute_gcc_phat(
        self,
        x_ref: np.ndarray,
        ref_positions: np.ndarray | None = None,
        sound_speed: float | None = None,
    ) -> np.ndarray:
        arr = np.asarray(x_ref, dtype=float)
        if arr.shape[0] != 3:
            raise ValueError(f"x_ref first dimension must be 3, got {arr.shape[0]}.")
        out = np.zeros((3, self.cfg.gcc_truncated_len), dtype=np.float32)
        for k, (i, j) in enumerate(self.pairs):
            max_delay_samples = None
            if ref_positions is not None and sound_speed is not None:
                max_delay_samples = self._resolve_max_delay_samples(
                    np.asarray(ref_positions, dtype=float)[i],
                    np.asarray(ref_positions, dtype=float)[j],
                    sound_speed=float(sound_speed),
                    fs=int(self.cfg.fs),
                )
            out[k] = self.gcc_phat_pair(arr[i], arr[j], max_delay_samples=max_delay_samples).astype(np.float32)
        return out

    def compute_psd_features(self, sig: np.ndarray) -> np.ndarray:
        n_fft = int(self.cfg.psd_nfft)
        arr = np.asarray(sig, dtype=float).reshape(-1)
        if arr.size < n_fft:
            arr = np.pad(arr, (0, n_fft - arr.size), mode="constant")
        else:
            arr = arr[:n_fft]
        spec = np.fft.rfft(arr, n=n_fft)
        psd = (np.abs(spec) ** 2) / max(n_fft, 1)
        psd = np.log10(psd + np.finfo(float).eps)
        return psd.astype(np.float32)

    def lowband_complex_spectrum(self, sig: np.ndarray) -> np.ndarray:
        arr = np.asarray(sig, dtype=float).reshape(-1)
        n_fft = int(self.acoustic_nfft)
        if arr.size < n_fft:
            arr = np.pad(arr, (0, n_fft - arr.size), mode="constant")
        else:
            arr = arr[:n_fft]
        spec = np.fft.rfft(arr, n=n_fft)
        return np.asarray(spec[self.acoustic_mask], dtype=np.complex64)

    def encode_complex_ri(self, spec: np.ndarray) -> np.ndarray:
        c = np.asarray(spec, dtype=np.complex64)
        return np.stack([c.real.astype(np.float32), c.imag.astype(np.float32)], axis=0)

    def encode_complex_mp(self, spec: np.ndarray) -> np.ndarray:
        c = np.asarray(spec, dtype=np.complex64)
        mag = np.log1p(np.abs(c)).astype(np.float32)
        phase = np.angle(c).astype(np.float32)
        return np.stack([mag, phase], axis=0)


class LayoutPreviewer:
    def __init__(self, cfg: DatasetBuildConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.layout_preview)
        self.plt = None
        self.fig = None
        self.ax = None
        if not self.enabled:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - preview is optional
            print(f"[Preview] disabled: {type(exc).__name__}: {exc}")
            self.enabled = False
            return
        self.plt = plt
        try:
            self.plt.ion()
            self.fig, self.ax = self.plt.subplots(figsize=(8, 6))
        except Exception as exc:  # pragma: no cover - preview is optional
            print(f"[Preview] disabled: {type(exc).__name__}: {exc}")
            self.enabled = False

    def update(self, mgr: RIRManager, sampled: dict[str, Any], accepted: int, attempts: int) -> None:
        if not self.enabled:
            return
        if accepted % max(int(self.cfg.layout_preview_interval), 1) != 0:
            return
        self.ax.clear()
        mgr.plot_layout_2d(ax=self.ax)
        src = np.asarray(sampled["source_pos"], dtype=float)
        refs = np.asarray(sampled["ref_positions"], dtype=float)
        sec = np.asarray(sampled["sec_positions"], dtype=float)[0]
        err = np.asarray(sampled["err_positions"], dtype=float)[0]
        self.ax.plot([src[0], sec[0]], [src[1], sec[1]], color="tab:red", alpha=0.25, linestyle="--")
        self.ax.plot([sec[0], err[0]], [sec[1], err[1]], color="tab:green", alpha=0.50, linestyle=":")
        for ref in refs:
            self.ax.plot([src[0], ref[0]], [src[1], ref[1]], color="tab:purple", alpha=0.20, linestyle="--")
        self.ax.set_title(f"Accepted {accepted} / attempts {attempts} | {sampled['layout_mode']}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self) -> None:
        if self.enabled and self.fig is not None:  # pragma: no cover - preview is optional
            self.plt.close(self.fig)


class AcousticScenarioSampler:
    def __init__(self, cfg: DatasetBuildConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

        self.source_id = 101
        self.ref_ids = np.array([401, 402, 403], dtype=int)
        self.sec_ids = np.array([201], dtype=int)
        self.err_ids = np.array([301], dtype=int)

    def _all_inside_room(self, points: np.ndarray, room_size: np.ndarray) -> bool:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        low = np.full(3, float(self.cfg.wall_margin), dtype=float)
        high = np.asarray(room_size, dtype=float) - low
        return bool(np.all(pts >= low) and np.all(pts <= high))

    def _sample_room_size(self) -> np.ndarray:
        return self.rng.uniform(self.cfg.room_size_min, self.cfg.room_size_max, size=3).astype(float)

    def _sample_room_acoustics(self) -> tuple[float, int]:
        if float(self.rng.random()) < float(self.cfg.reflection_room_probability):
            absorption = self.rng.uniform(*self.cfg.reflection_room_absorption_range)
            image_order = int(
                self.rng.choice(
                    self.cfg.reflection_room_image_order_choices,
                    p=self.cfg.reflection_room_image_order_probs,
                )
            )
        else:
            absorption = self.rng.uniform(*self.cfg.direct_room_absorption_range)
            image_order = int(
                self.rng.choice(
                    self.cfg.direct_room_image_order_choices,
                    p=self.cfg.direct_room_image_order_probs,
                )
            )
        return float(absorption), int(image_order)

    def _sample_reference_center(self, room_size: np.ndarray) -> np.ndarray:
        room_mid = 0.5 * np.asarray(room_size, dtype=float)
        max_ref_radius = float(self.cfg.reference_radius_range[1]) + 0.08
        x_span = min(float(self.cfg.ref_center_jitter_xy), float(room_mid[0] - self.cfg.wall_margin - max_ref_radius))
        y_span = min(float(self.cfg.ref_center_jitter_xy), float(room_mid[1] - self.cfg.wall_margin - max_ref_radius))
        if x_span <= 0.0 or y_span <= 0.0:
            raise QCError("geometry:room_too_small_for_reference_center")
        z_low = max(float(self.cfg.wall_margin + 0.10), float(self.cfg.ref_center_height_range[0]))
        z_high = min(float(room_size[2] - self.cfg.wall_margin - 0.10), float(self.cfg.ref_center_height_range[1]))
        if z_high <= z_low:
            raise QCError("geometry:room_too_small_for_reference_center_height")
        return np.array(
            [
                self.rng.uniform(room_mid[0] - x_span, room_mid[0] + x_span),
                self.rng.uniform(room_mid[1] - y_span, room_mid[1] + y_span),
                self.rng.uniform(z_low, z_high),
            ],
            dtype=float,
        )

    def _sample_source_position(self, room_size: np.ndarray) -> np.ndarray:
        xy_margin = min(
            float(self.cfg.source_wall_margin),
            float(np.min(room_size[:2]) / 2.0 - 0.20),
        )
        if xy_margin <= float(self.cfg.wall_margin + 0.10):
            raise QCError("geometry:room_too_small_for_source")
        z_low = max(float(self.cfg.wall_margin + 0.20), 0.90)
        z_high = float(room_size[2] - self.cfg.wall_margin - 0.20)
        if z_high <= z_low:
            raise QCError("geometry:room_too_small_for_source_height")
        return np.array(
            [
                self.rng.uniform(xy_margin, room_size[0] - xy_margin),
                self.rng.uniform(xy_margin, room_size[1] - xy_margin),
                self.rng.uniform(z_low, z_high),
            ],
            dtype=float,
        )

    def _sample_reference_positions(self, source_pos: np.ndarray, room_size: np.ndarray) -> np.ndarray:
        for _ in range(100):
            base = float(self.rng.uniform(0.0, 360.0))
            angles = (
                base
                + np.asarray(self.cfg.reference_azimuth_base)
                + self.rng.uniform(
                    -self.cfg.reference_azimuth_jitter,
                    self.cfg.reference_azimuth_jitter,
                    size=3,
                )
            ) % 360.0
            refs = np.zeros((3, 3), dtype=float)
            for i, angle in enumerate(angles):
                radius = float(self.rng.uniform(*self.cfg.reference_radius_range))
                theta = np.deg2rad(angle)
                z_off = float(self.rng.uniform(*self.cfg.reference_z_offset_range))
                refs[i] = np.array(
                    [
                        source_pos[0] + radius * np.cos(theta),
                        source_pos[1] + radius * np.sin(theta),
                        np.clip(
                            source_pos[2] + z_off,
                            max(float(self.cfg.wall_margin + 0.05), float(self.cfg.reference_height_range[0])),
                            min(float(room_size[2] - self.cfg.wall_margin - 0.05), float(self.cfg.reference_height_range[1])),
                        ),
                    ],
                    dtype=float,
                )
            if not self._all_inside_room(refs, room_size):
                continue
            if _min_pairwise_distance(refs) < float(self.cfg.min_reference_pair_distance):
                continue
            if float(np.min(np.linalg.norm(refs - source_pos[None, :], axis=1))) < float(self.cfg.min_source_reference_distance):
                continue
            tri_angles = _triangle_angles_deg(refs)
            if float(np.min(tri_angles)) < float(self.cfg.ref_triangle_min_angle_deg):
                continue
            if float(np.max(tri_angles)) > float(self.cfg.ref_triangle_max_angle_deg):
                continue
            return refs
        raise QCError("geometry:failed_reference_layout")

    def _is_outside_reference_triangle(self, point: np.ndarray, ref_positions: np.ndarray) -> bool:
        p_xy = np.asarray(point, dtype=float)[:2]
        tri_xy = np.asarray(ref_positions, dtype=float)[:, :2]
        if _point_in_triangle_xy(p_xy, tri_xy):
            return False
        edge_dist = min(
            _point_to_segment_distance_xy(p_xy, tri_xy[0], tri_xy[1]),
            _point_to_segment_distance_xy(p_xy, tri_xy[1], tri_xy[2]),
            _point_to_segment_distance_xy(p_xy, tri_xy[2], tri_xy[0]),
        )
        return bool(edge_dist >= float(self.cfg.triangle_outside_margin))

    def _sample_aligned_secondary_and_error(
        self,
        ref_center: np.ndarray,
        ref_positions: np.ndarray,
        room_size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        z_low = max(float(self.cfg.wall_margin + 0.05), float(self.cfg.secondary_height_range[0]))
        z_high = min(float(room_size[2] - self.cfg.wall_margin - 0.05), float(self.cfg.secondary_height_range[1]))
        if z_high <= z_low:
            raise QCError("geometry:room_too_small_for_secondary")

        for _ in range(220):
            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            sec_radius = float(self.rng.uniform(*self.cfg.sec_center_distance_range))
            err_low = max(float(self.cfg.err_center_distance_range[0]), sec_radius + float(self.cfg.min_err_sec_gap))
            err_high = float(self.cfg.err_center_distance_range[1])
            if err_high <= err_low:
                continue
            err_radius = float(self.rng.uniform(err_low, err_high))

            z_base = float(np.clip(ref_center[2] + self.rng.uniform(-0.05, 0.05), z_low, z_high))
            z_err = float(np.clip(z_base + self.rng.uniform(-0.02, 0.02), z_low, z_high))
            sec = np.array(
                [
                    ref_center[0] + sec_radius * np.cos(theta),
                    ref_center[1] + sec_radius * np.sin(theta),
                    z_base,
                ],
                dtype=float,
            )
            err = np.array(
                [
                    ref_center[0] + err_radius * np.cos(theta),
                    ref_center[1] + err_radius * np.sin(theta),
                    z_err,
                ],
                dtype=float,
            )
            if not self._all_inside_room(np.vstack([sec, err]), room_size):
                continue
            if not self._is_outside_reference_triangle(sec, ref_positions):
                continue
            if not self._is_outside_reference_triangle(err, ref_positions):
                continue
            if float(np.min(np.linalg.norm(ref_positions - sec[None, :], axis=1))) < float(self.cfg.min_reference_device_distance):
                continue
            if float(np.min(np.linalg.norm(ref_positions - err[None, :], axis=1))) < float(self.cfg.min_reference_device_distance):
                continue

            line_clearance = min(_point_to_line_distance(ref_pos, sec, err) for ref_pos in np.asarray(ref_positions, dtype=float))
            if line_clearance < float(self.cfg.line_to_ref_min_distance):
                continue

            v_center = np.asarray(ref_center, dtype=float) - sec
            v_err = err - sec
            if float(np.dot(v_center, v_err)) >= 0.0:
                continue
            denom = float(np.linalg.norm(v_center) * np.linalg.norm(v_err))
            if denom <= np.finfo(float).eps:
                continue
            col_sine = float(np.linalg.norm(np.cross(v_center, v_err)) / denom)
            if col_sine > float(self.cfg.collinearity_max_sine):
                continue
            return sec, err
        raise QCError("geometry:failed_secondary_error_collinear_layout")

    def _sample_source_position_farfield(
        self,
        room_size: np.ndarray,
        ref_center: np.ndarray,
        ref_positions: np.ndarray,
        sec_pos: np.ndarray,
        err_pos: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        for _ in range(220):
            source_pos = self._sample_source_position(room_size)
            if float(np.linalg.norm(source_pos - ref_center)) < float(self.cfg.min_noise_source_ref_center_distance):
                continue
            if float(np.min(np.linalg.norm(ref_positions - source_pos[None, :], axis=1))) < float(self.cfg.min_source_reference_distance):
                continue
            if float(np.linalg.norm(source_pos - sec_pos)) < float(self.cfg.min_source_device_distance):
                continue
            if float(np.linalg.norm(source_pos - err_pos)) < float(self.cfg.min_source_device_distance):
                continue
            ok, primary_margin, secondary_margin = self._causality_status(source_pos, ref_positions, sec_pos, err_pos)
            if not ok:
                continue
            return source_pos, primary_margin, secondary_margin
        raise QCError("geometry:failed_farfield_source_layout")

    def _sample_secondary_height(self, room_size: np.ndarray) -> float:
        z_low = max(float(self.cfg.wall_margin + 0.05), float(self.cfg.secondary_height_range[0]))
        z_high = min(float(room_size[2] - self.cfg.wall_margin - 0.05), float(self.cfg.secondary_height_range[1]))
        if z_high <= z_low:
            raise QCError("geometry:room_too_small_for_secondary")
        return float(self.rng.uniform(z_low, z_high))

    def _sample_layout_mode(self) -> str:
        return str(self.rng.choice(self.cfg.layout_mode_choices, p=self.cfg.layout_mode_probs))

    def _sample_secondary_position(
        self,
        source_pos: np.ndarray,
        ref_positions: np.ndarray,
        room_size: np.ndarray,
        layout_mode: str,
    ) -> np.ndarray:
        for _ in range(120):
            if layout_mode == "wall_side":
                axis = int(self.rng.integers(0, 2))
                low_side = bool(self.rng.integers(0, 2) == 0)
                inward = float(self.rng.uniform(0.45, 1.25))
                z = self._sample_secondary_height(room_size)
                if axis == 0:
                    x = inward if low_side else float(room_size[0] - inward)
                    y = float(self.rng.uniform(self.cfg.wall_margin + 0.40, room_size[1] - self.cfg.wall_margin - 0.40))
                else:
                    y = inward if low_side else float(room_size[1] - inward)
                    x = float(self.rng.uniform(self.cfg.wall_margin + 0.40, room_size[0] - self.cfg.wall_margin - 0.40))
                sec = np.array([x, y, z], dtype=float)
            elif layout_mode == "corner_far":
                x_low = bool(self.rng.integers(0, 2) == 0)
                y_low = bool(self.rng.integers(0, 2) == 0)
                x = float(self.rng.uniform(0.45, 1.35))
                y = float(self.rng.uniform(0.45, 1.35))
                z = self._sample_secondary_height(room_size)
                sec = np.array(
                    [
                        x if x_low else float(room_size[0] - x),
                        y if y_low else float(room_size[1] - y),
                        z,
                    ],
                    dtype=float,
                )
            else:
                z = self._sample_secondary_height(room_size)
                sec = np.array(
                    [
                        self.rng.uniform(self.cfg.wall_margin + 0.40, room_size[0] - self.cfg.wall_margin - 0.40),
                        self.rng.uniform(self.cfg.wall_margin + 0.40, room_size[1] - self.cfg.wall_margin - 0.40),
                        z,
                    ],
                    dtype=float,
                )

            source_dist = float(np.linalg.norm(sec - source_pos))
            if source_dist < float(self.cfg.secondary_source_distance_range[0]) or source_dist > float(self.cfg.secondary_source_distance_range[1]):
                continue
            if float(np.min(np.linalg.norm(ref_positions - sec[None, :], axis=1))) < float(self.cfg.min_reference_device_distance):
                continue
            if source_dist < float(self.cfg.min_source_device_distance):
                continue
            return sec
        raise QCError(f"geometry:failed_secondary_layout:{layout_mode}")

    def _causality_status(
        self,
        source_pos: np.ndarray,
        ref_positions: np.ndarray,
        sec_pos: np.ndarray,
        err_pos: np.ndarray,
    ) -> tuple[bool, float, float]:
        source_ref = np.linalg.norm(ref_positions - source_pos[None, :], axis=1)
        source_err = float(np.linalg.norm(err_pos - source_pos))
        primary_margin = float(np.min(source_err - source_ref))

        sec_ref = np.linalg.norm(ref_positions - sec_pos[None, :], axis=1)
        sec_err = float(np.linalg.norm(err_pos - sec_pos))
        secondary_margin = float(np.min(sec_ref - sec_err))

        ok = primary_margin >= float(self.cfg.min_primary_advance_margin_m) and secondary_margin >= float(self.cfg.min_secondary_feedback_margin_m)
        return bool(ok), primary_margin, secondary_margin

    def _sample_error_position(
        self,
        source_pos: np.ndarray,
        ref_positions: np.ndarray,
        sec_pos: np.ndarray,
        room_size: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        for _ in range(160):
            radius = float(self.rng.uniform(*self.cfg.sec_err_distance_range))
            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            z_delta = float(self.rng.uniform(-0.08, 0.08))
            err = sec_pos + np.array([radius * np.cos(theta), radius * np.sin(theta), z_delta], dtype=float)
            if not self._all_inside_room(err[None, :], room_size):
                continue
            if float(np.linalg.norm(err - source_pos)) < float(self.cfg.min_source_device_distance):
                continue
            if float(np.min(np.linalg.norm(ref_positions - err[None, :], axis=1))) < float(self.cfg.min_reference_device_distance):
                continue
            ok, primary_margin, secondary_margin = self._causality_status(source_pos, ref_positions, sec_pos, err)
            if not ok:
                continue
            return err, primary_margin, secondary_margin
        raise QCError("geometry:failed_error_layout")

    def _layout_metadata(
        self,
        source_pos: np.ndarray,
        ref_positions: np.ndarray,
        sec_pos: np.ndarray,
        err_pos: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        ref_vec = np.asarray(ref_positions, dtype=float) - np.asarray(source_pos, dtype=float)[None, :]
        ref_azimuth_deg = (np.degrees(np.arctan2(ref_vec[:, 1], ref_vec[:, 0])) + 360.0) % 360.0
        ref_radii = np.linalg.norm(ref_vec, axis=1)
        return {
            "ref_azimuth_deg": ref_azimuth_deg.astype(float),
            "ref_radii": ref_radii.astype(float),
            "sec_source_distance": float(np.linalg.norm(np.asarray(sec_pos, dtype=float) - np.asarray(source_pos, dtype=float))),
            "err_source_distance": float(np.linalg.norm(np.asarray(err_pos, dtype=float) - np.asarray(source_pos, dtype=float))),
            "sec_err_distance": float(np.linalg.norm(np.asarray(err_pos, dtype=float) - np.asarray(sec_pos, dtype=float))),
        }

    def sample(self) -> dict[str, Any]:
        for _ in range(1500):
            room_size = self._sample_room_size()
            ref_center = self._sample_reference_center(room_size)
            ref_positions = self._sample_reference_positions(ref_center, room_size)
            layout_mode = self._sample_layout_mode()
            try:
                sec_pos, err_pos = self._sample_aligned_secondary_and_error(ref_center, ref_positions, room_size)
                source_pos, primary_margin, secondary_margin = self._sample_source_position_farfield(
                    room_size,
                    ref_center,
                    ref_positions,
                    sec_pos,
                    err_pos,
                )
            except QCError:
                continue

            absorption, image_order = self._sample_room_acoustics()
            sound_speed = float(self.rng.uniform(self.cfg.sound_speed_min, self.cfg.sound_speed_max))
            layout_meta = self._layout_metadata(source_pos, ref_positions, sec_pos, err_pos)
            return {
                "room_size": room_size,
                "source_pos": source_pos,
                "ref_positions": ref_positions,
                "ref_center": np.asarray(ref_center, dtype=float),
                "sec_positions": np.asarray(sec_pos[None, :], dtype=float),
                "err_positions": np.asarray(err_pos[None, :], dtype=float),
                **layout_meta,
                "primary_advance_margin_min": float(primary_margin),
                "secondary_feedback_margin_min": float(secondary_margin),
                "sound_speed": sound_speed,
                "absorption": float(absorption),
                "image_order": int(image_order),
                "layout_mode": layout_mode,
            }
        raise QCError("geometry:failed_to_sample_valid_layout")

    def build_manager(self, sampled: dict[str, Any]) -> RIRManager:
        mgr = RIRManager()
        mgr.room = np.asarray(sampled["room_size"], dtype=float)
        mgr.fs = int(self.cfg.fs)
        mgr.sound_speed = float(sampled["sound_speed"])
        mgr.image_source_order = int(sampled["image_order"])
        mgr.material_absorption = float(sampled["absorption"])
        mgr.air_absorption = bool(self.cfg.enable_air_absorption)
        mgr.compensate_fractional_delay = True
        mgr.fractional_delay_shift = None
        mgr.add_primary_speaker(int(self.source_id), sampled["source_pos"])
        for i, ref_id in enumerate(self.ref_ids):
            mgr.add_reference_microphone(int(ref_id), sampled["ref_positions"][i])
        mgr.add_secondary_speaker(int(self.sec_ids[0]), sampled["sec_positions"][0])
        mgr.add_error_microphone(int(self.err_ids[0]), sampled["err_positions"][0])
        return mgr


class ANCQualityController:
    def __init__(self, cfg: DatasetBuildConfig):
        self.cfg = cfg

    @staticmethod
    def _path_is_legal(path: np.ndarray) -> bool:
        arr = np.asarray(path, dtype=float).reshape(-1)
        if arr.size < 8:
            return False
        if not np.all(np.isfinite(arr)):
            return False
        if float(np.sum(arr**2)) <= np.finfo(float).eps:
            return False
        return True

    def _direct_energy_ratio(
        self,
        rir: np.ndarray,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
        fs: int,
        sound_speed: float,
    ) -> float:
        distance = float(np.linalg.norm(np.asarray(tx_pos, dtype=float) - np.asarray(rx_pos, dtype=float)))
        expected_idx = int(round(distance / max(sound_speed, 1.0e-6) * fs))
        rir = np.asarray(rir, dtype=float).reshape(-1)
        if expected_idx >= rir.size:
            return 0.0
        half_w = int(self.cfg.direct_energy_half_window)
        left = max(0, expected_idx - half_w)
        right = min(rir.size, expected_idx + half_w + 1)
        direct_e = float(np.sum(rir[left:right] ** 2))
        total_e = float(np.sum(rir**2)) + np.finfo(float).eps
        return direct_e / total_e

    def validate_rirs(
        self,
        mgr: RIRManager,
        sampled: dict[str, Any],
        source_id: int,
        ref_ids: np.ndarray,
        sec_ids: np.ndarray,
        err_ids: np.ndarray,
    ) -> tuple[
        dict[str, float],
        np.ndarray,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        sec_id = int(sec_ids[0])
        err_id = int(err_ids[0])
        source_pos = np.asarray(sampled["source_pos"], dtype=float)
        ref_positions = np.asarray(sampled["ref_positions"], dtype=float)
        sec_pos = np.asarray(sampled["sec_positions"][0], dtype=float)
        err_pos = np.asarray(sampled["err_positions"][0], dtype=float)

        ratios: list[float] = []

        s_err = np.asarray(mgr.get_secondary_rir(sec_id, err_id), dtype=float)
        if not self._path_is_legal(s_err):
            raise QCError("rir:illegal_secondary_path")
        ratios.append(self._direct_energy_ratio(s_err, sec_pos, err_pos, int(mgr.fs), float(mgr.sound_speed)))
        secondary_path, secondary_len = _crop_rir(s_err, self.cfg.rir_store_len)

        p_err = np.asarray(mgr.get_primary_rir(source_id, err_id), dtype=float)
        if not self._path_is_legal(p_err):
            raise QCError("rir:illegal_primary_error_path")
        ratios.append(self._direct_energy_ratio(p_err, source_pos, err_pos, int(mgr.fs), float(mgr.sound_speed)))
        primary_to_error_path, primary_to_error_len = _crop_rir(p_err, self.cfg.rir_store_len)

        e2r_full = np.asarray(mgr.compute_transfer_rirs(err_pos, ref_positions), dtype=float)
        if e2r_full.shape[0] != int(self.cfg.num_reference_mics):
            raise QCError("rir:unexpected_error_to_ref_shape")
        e2r_paths = np.zeros((self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype=np.float32)
        e2r_lengths = np.zeros((self.cfg.num_reference_mics,), dtype=np.int32)
        for i in range(self.cfg.num_reference_mics):
            if not self._path_is_legal(e2r_full[i]):
                raise QCError("rir:illegal_error_to_reference_path")
            e2r_paths[i], e2r_lengths[i] = _crop_rir(e2r_full[i], self.cfg.rir_store_len)

        s2r_full = np.asarray(mgr.compute_transfer_rirs(sec_pos, ref_positions), dtype=float)
        if s2r_full.shape[0] != int(self.cfg.num_reference_mics):
            raise QCError("rir:unexpected_secondary_to_ref_shape")
        s2r_paths = np.zeros((self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype=np.float32)
        s2r_lengths = np.zeros((self.cfg.num_reference_mics,), dtype=np.int32)
        feedback_norm_max = 0.0
        for i in range(self.cfg.num_reference_mics):
            if not self._path_is_legal(s2r_full[i]):
                raise QCError("rir:illegal_secondary_to_reference_path")
            s2r_paths[i], s2r_lengths[i] = _crop_rir(s2r_full[i], self.cfg.rir_store_len)
            feedback_norm_max = max(feedback_norm_max, float(np.linalg.norm(s2r_full[i])))

        p_ref_paths = np.zeros((self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype=np.float32)
        p_ref_lengths = np.zeros((self.cfg.num_reference_mics,), dtype=np.int32)
        r2r_paths = np.zeros((len(self.cfg.canonical_r2r_pair_order), self.cfg.rir_store_len), dtype=np.float32)
        r2r_lengths = np.zeros((len(self.cfg.canonical_r2r_pair_order),), dtype=np.int32)

        p_ref_norm_mean = 0.0
        for i, ref_id in enumerate(ref_ids):
            p_ref = np.asarray(mgr.get_reference_rir(source_id, int(ref_id)), dtype=float)
            if not self._path_is_legal(p_ref):
                raise QCError("rir:illegal_primary_reference_path")
            ratios.append(self._direct_energy_ratio(p_ref, source_pos, ref_positions[i], int(mgr.fs), float(mgr.sound_speed)))
            p_ref_norm_mean += float(np.linalg.norm(p_ref))
            p_ref_paths[i], p_ref_lengths[i] = _crop_rir(p_ref, self.cfg.rir_store_len)
        p_ref_norm_mean /= float(max(len(ref_ids), 1))

        r2r_pairs = ((0, 1), (0, 2), (1, 2))
        for pair_idx, (i, j) in enumerate(r2r_pairs):
            r2r = np.asarray(mgr.compute_transfer_rirs(ref_positions[i], ref_positions[j][None, :]), dtype=float).reshape(-1)
            if not self._path_is_legal(r2r):
                raise QCError("rir:illegal_reference_to_reference_path")
            r2r_paths[pair_idx], r2r_lengths[pair_idx] = _crop_rir(r2r, self.cfg.rir_store_len)

        ratio_min = float(np.min(ratios))
        if ratio_min < float(self.cfg.min_direct_ratio):
            raise QCError("rir:direct_path_not_dominant_enough")

        control_ratio = float(np.linalg.norm(s_err)) / (float(np.linalg.norm(p_err)) + np.finfo(float).eps)
        if control_ratio < float(self.cfg.min_control_ratio):
            raise QCError("rir:insufficient_secondary_control_energy")

        feedback_ratio_max = feedback_norm_max / max(p_ref_norm_mean, np.finfo(float).eps)

        return (
            {
                "direct_ratio_min": ratio_min,
                "control_ratio_min": control_ratio,
                "feedback_ratio_max": float(feedback_ratio_max),
            },
            secondary_path,
            int(secondary_len),
            primary_to_error_path,
            int(primary_to_error_len),
            p_ref_paths,
            p_ref_lengths,
            e2r_paths,
            e2r_lengths,
            s2r_paths,
            s2r_lengths,
            r2r_paths,
            r2r_lengths,
        )

    @staticmethod
    def _nr_db(d_seg: np.ndarray, e_seg: np.ndarray) -> float:
        d_pow = float(np.mean(np.asarray(d_seg, dtype=float) ** 2)) + np.finfo(float).eps
        e_pow = float(np.mean(np.asarray(e_seg, dtype=float) ** 2)) + np.finfo(float).eps
        return float(10.0 * np.log10(d_pow / e_pow))

    def _nr_metrics(self, d: np.ndarray, e: np.ndarray, fs: int) -> dict[str, float]:
        d_arr = np.asarray(d, dtype=float).reshape(d.shape[0], -1)
        e_arr = np.asarray(e, dtype=float).reshape(e.shape[0], -1)
        n = int(d_arr.shape[0])
        if n < 16:
            raise QCError("anc:signal_too_short")
        win = min(max(int(round(0.5 * fs)), 8), n // 2)
        nr_first = self._nr_db(d_arr[:win], e_arr[:win])
        nr_last = self._nr_db(d_arr[-win:], e_arr[-win:])
        return {
            "nr_first_db": float(nr_first),
            "nr_last_db": float(nr_last),
            "nr_gain_db": float(nr_last - nr_first),
        }

    def evaluate_anc(
        self,
        mgr: RIRManager,
        source_signal: np.ndarray,
        time_axis: np.ndarray,
        sec_ids: np.ndarray,
    ) -> dict[str, Any]:
        d = mgr.calculate_desired_signal(source_signal, len(time_axis))
        x = mgr.calculate_reference_signal(source_signal, len(time_axis))
        x = _normalize_columns(x)

        best: dict[str, Any] | None = None
        for mu in self.cfg.mu_candidates:
            params = {
                "time": time_axis,
                "rir_manager": mgr,
                "L": int(self.cfg.filter_len),
                "mu": float(mu),
                "reference_signal": x,
                "desired_signal": d,
                "verbose": False,
                "normalized_update": bool(self.cfg.anc_normalized_update),
                "norm_epsilon": float(self.cfg.anc_norm_epsilon),
            }
            try:
                results = cfxlms(params)
            except Exception:
                continue

            e = np.asarray(results.get("err_hist"), dtype=float)
            if e.ndim != 2 or e.shape[1] != int(self.cfg.num_error_mics) or not np.all(np.isfinite(e)):
                continue

            sec_id = int(sec_ids[0])
            w_mat = np.asarray(results.get("filter_coeffs", {}).get(sec_id, []), dtype=float)
            if w_mat.ndim != 2 or w_mat.shape[0] < int(self.cfg.filter_len) or w_mat.shape[1] < int(self.cfg.num_reference_mics):
                continue
            if not np.all(np.isfinite(w_mat)):
                continue

            metrics = self._nr_metrics(d, e, int(mgr.fs))
            passed = metrics["nr_last_db"] >= float(self.cfg.min_nr_last_db) and metrics["nr_gain_db"] >= float(self.cfg.min_nr_gain_db)
            if not passed:
                continue
            if best is None or metrics["nr_last_db"] > best["metrics"]["nr_last_db"]:
                best = {
                    "mu": float(mu),
                    "results": results,
                    "metrics": metrics,
                    "x": x,
                    "d": d,
                }

        if best is None:
            raise QCError("anc:noise_reduction_qc_failed")

        sec_id = int(sec_ids[0])
        w_mat = np.asarray(best["results"]["filter_coeffs"][sec_id], dtype=float)
        w_full = np.zeros((1, self.cfg.num_reference_mics, self.cfg.filter_len), dtype=np.float32)
        w_full[0] = w_mat[: self.cfg.filter_len, : self.cfg.num_reference_mics].T.astype(np.float32)
        best["w_full"] = w_full
        best["w_opt"] = w_full[0].astype(np.float32)
        return best


class HDF5DatasetWriter:
    def __init__(self, cfg: DatasetBuildConfig, output_path: Path):
        self.cfg = cfg
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(str(self.output_path), "w")
        self._build_layout()

    def _build_layout(self) -> None:
        n = int(self.cfg.target_rooms)
        self.h5.attrs["schema"] = "CFxLMS_ANC_QC_Dataset_v4_single_control_canonical_q"
        self.h5.attrs["config_json"] = json.dumps(asdict(self.cfg), ensure_ascii=False)

        raw = self.h5.create_group("raw")
        room = raw.create_group("room_params")
        qc = raw.create_group("qc_metrics")
        raw.attrs["r2r_pair_order_json"] = json.dumps(list(self.cfg.canonical_r2r_pair_order), ensure_ascii=False)

        room.create_dataset("room_size", shape=(n, 3), dtype="f4")
        room.create_dataset("source_position", shape=(n, 3), dtype="f4")
        room.create_dataset("ref_positions", shape=(n, self.cfg.num_reference_mics, 3), dtype="f4")
        room.create_dataset("sec_positions", shape=(n, self.cfg.num_secondary_speakers, 3), dtype="f4")
        room.create_dataset("err_positions", shape=(n, self.cfg.num_error_mics, 3), dtype="f4")
        room.create_dataset("ref_azimuth_deg", shape=(n, self.cfg.num_reference_mics), dtype="f4")
        room.create_dataset("ref_radii", shape=(n, self.cfg.num_reference_mics), dtype="f4")
        room.create_dataset("sec_source_distance", shape=(n,), dtype="f4")
        room.create_dataset("err_source_distance", shape=(n,), dtype="f4")
        room.create_dataset("sec_err_distance", shape=(n,), dtype="f4")
        room.create_dataset("primary_advance_margin_min", shape=(n,), dtype="f4")
        room.create_dataset("secondary_feedback_margin_min", shape=(n,), dtype="f4")
        room.create_dataset("sound_speed", shape=(n,), dtype="f4")
        room.create_dataset("material_absorption", shape=(n,), dtype="f4")
        room.create_dataset("image_source_order", shape=(n,), dtype="i4")
        room.create_dataset("layout_mode", shape=(n,), dtype=h5py.string_dtype(encoding="utf-8"))

        raw.create_dataset("x_ref", shape=(n, self.cfg.num_reference_mics, self.cfg.ref_window_samples), dtype="f4")
        raw.create_dataset("S_paths", shape=(n, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("S_path_lengths", shape=(n,), dtype="i4")
        raw.create_dataset("P_ref_paths", shape=(n, self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("P_ref_path_lengths", shape=(n, self.cfg.num_reference_mics), dtype="i4")
        raw.create_dataset("D_path", shape=(n, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("D_path_length", shape=(n,), dtype="i4")
        raw.create_dataset("E2R_paths", shape=(n, self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("E2R_path_lengths", shape=(n, self.cfg.num_reference_mics), dtype="i4")
        raw.create_dataset("S2R_paths", shape=(n, self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("S2R_path_lengths", shape=(n, self.cfg.num_reference_mics), dtype="i4")
        raw.create_dataset("R2R_paths", shape=(n, len(self.cfg.canonical_r2r_pair_order), self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("R2R_path_lengths", shape=(n, len(self.cfg.canonical_r2r_pair_order)), dtype="i4")
        raw.create_dataset("W_opt", shape=(n, self.cfg.num_reference_mics, self.cfg.filter_len), dtype="f4")
        raw.create_dataset("W_full", shape=(n, 1, self.cfg.num_reference_mics, self.cfg.filter_len), dtype="f4")

        qc.create_dataset("nr_first_db", shape=(n,), dtype="f4")
        qc.create_dataset("nr_last_db", shape=(n,), dtype="f4")
        qc.create_dataset("nr_gain_db", shape=(n,), dtype="f4")
        qc.create_dataset("direct_ratio_min", shape=(n,), dtype="f4")
        qc.create_dataset("control_ratio_min", shape=(n,), dtype="f4")
        qc.create_dataset("feedback_ratio_max", shape=(n,), dtype="f4")
        qc.create_dataset("mu_used", shape=(n,), dtype="f4")
        qc.create_dataset("source_seed", shape=(n,), dtype="i8")
        qc.create_dataset("warmup_start_s", shape=(n,), dtype="f4")
        qc.create_dataset("warmup_start_index", shape=(n,), dtype="i4")

    def write(self, idx: int, sample: RoomSample) -> None:
        room = self.h5["raw/room_params"]
        qc = self.h5["raw/qc_metrics"]

        room["room_size"][idx] = sample.room_params["room_size"].astype(np.float32)
        room["source_position"][idx] = sample.room_params["source_pos"].astype(np.float32)
        room["ref_positions"][idx] = sample.room_params["ref_positions"].astype(np.float32)
        room["sec_positions"][idx] = sample.room_params["sec_positions"].astype(np.float32)
        room["err_positions"][idx] = sample.room_params["err_positions"].astype(np.float32)
        room["ref_azimuth_deg"][idx] = sample.room_params["ref_azimuth_deg"].astype(np.float32)
        room["ref_radii"][idx] = sample.room_params["ref_radii"].astype(np.float32)
        room["sec_source_distance"][idx] = float(sample.room_params["sec_source_distance"])
        room["err_source_distance"][idx] = float(sample.room_params["err_source_distance"])
        room["sec_err_distance"][idx] = float(sample.room_params["sec_err_distance"])
        room["primary_advance_margin_min"][idx] = float(sample.room_params["primary_advance_margin_min"])
        room["secondary_feedback_margin_min"][idx] = float(sample.room_params["secondary_feedback_margin_min"])
        room["sound_speed"][idx] = float(sample.room_params["sound_speed"])
        room["material_absorption"][idx] = float(sample.room_params["absorption"])
        room["image_source_order"][idx] = int(sample.room_params["image_order"])
        room["layout_mode"][idx] = str(sample.room_params["layout_mode"])

        self.h5["raw/x_ref"][idx] = sample.x_ref.astype(np.float32)
        self.h5["raw/S_paths"][idx] = sample.secondary_path.astype(np.float32)
        self.h5["raw/S_path_lengths"][idx] = int(sample.secondary_path_length)
        self.h5["raw/P_ref_paths"][idx] = sample.primary_to_ref_paths.astype(np.float32)
        self.h5["raw/P_ref_path_lengths"][idx] = sample.primary_to_ref_lengths.astype(np.int32)
        self.h5["raw/D_path"][idx] = sample.primary_to_error_path.astype(np.float32)
        self.h5["raw/D_path_length"][idx] = int(sample.primary_to_error_length)
        self.h5["raw/E2R_paths"][idx] = sample.error_to_ref_paths.astype(np.float32)
        self.h5["raw/E2R_path_lengths"][idx] = sample.error_to_ref_lengths.astype(np.int32)
        self.h5["raw/S2R_paths"][idx] = sample.secondary_to_ref_paths.astype(np.float32)
        self.h5["raw/S2R_path_lengths"][idx] = sample.secondary_to_ref_lengths.astype(np.int32)
        self.h5["raw/R2R_paths"][idx] = sample.ref_to_ref_paths.astype(np.float32)
        self.h5["raw/R2R_path_lengths"][idx] = sample.ref_to_ref_lengths.astype(np.int32)
        self.h5["raw/W_opt"][idx] = sample.w_opt.astype(np.float32)
        self.h5["raw/W_full"][idx] = sample.w_full.astype(np.float32)

        qc["nr_first_db"][idx] = float(sample.qc_metrics["nr_first_db"])
        qc["nr_last_db"][idx] = float(sample.qc_metrics["nr_last_db"])
        qc["nr_gain_db"][idx] = float(sample.qc_metrics["nr_gain_db"])
        qc["direct_ratio_min"][idx] = float(sample.qc_metrics["direct_ratio_min"])
        qc["control_ratio_min"][idx] = float(sample.qc_metrics["control_ratio_min"])
        qc["feedback_ratio_max"][idx] = float(sample.qc_metrics["feedback_ratio_max"])
        qc["mu_used"][idx] = float(sample.qc_metrics["mu_used"])
        qc["source_seed"][idx] = int(sample.qc_metrics["source_seed"])
        qc["warmup_start_s"][idx] = float(sample.qc_metrics["warmup_start_s"])
        qc["warmup_start_index"][idx] = int(sample.qc_metrics["warmup_start_index"])

    def close(self) -> None:
        if self.h5:
            self.h5.flush()
            self.h5.close()


class ANCDatasetBuilder:
    def __init__(self, cfg: DatasetBuildConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.py_random = random.Random(cfg.random_seed)
        self.sampler = AcousticScenarioSampler(cfg, self.rng)
        self.qc = ANCQualityController(cfg)
        self.layout_previewer = LayoutPreviewer(cfg)
        self.failure_stats: dict[str, int] = {}

    def _count_fail(self, reason: str) -> None:
        self.failure_stats[reason] = self.failure_stats.get(reason, 0) + 1

    def _sample_warmup_start(self, total_samples: int) -> tuple[int, float]:
        window = int(self.cfg.ref_window_samples)
        if total_samples < window:
            raise QCError("warmup:signal_shorter_than_slice")
        fs = float(self.cfg.fs)
        latest_start_s = (total_samples - window) / fs
        start_min_s = float(self.cfg.warmup_start_s_min)
        start_max_s = min(float(self.cfg.warmup_start_s_max), latest_start_s)
        if start_max_s < start_min_s:
            raise QCError("warmup:invalid_start_interval")
        start_s = float(self.py_random.uniform(start_min_s, start_max_s))
        start_idx = int(round(start_s * fs))
        start_idx = max(0, min(start_idx, total_samples - window))
        return start_idx, start_s

    def _build_single_sample(self) -> RoomSample:
        sampled = self.sampler.sample()
        mgr = self.sampler.build_manager(sampled)
        mgr.build(verbose=False)

        (
            rir_metrics,
            secondary_path,
            secondary_len,
            primary_to_error_path,
            primary_to_error_len,
            primary_to_ref_paths,
            primary_to_ref_lengths,
            e2r_paths,
            e2r_lengths,
            s2r_paths,
            s2r_lengths,
            r2r_paths,
            r2r_lengths,
        ) = self.qc.validate_rirs(
            mgr=mgr,
            sampled=sampled,
            source_id=int(self.sampler.source_id),
            ref_ids=self.sampler.ref_ids,
            sec_ids=self.sampler.sec_ids,
            err_ids=self.sampler.err_ids,
        )

        source_seed = int(self.py_random.randrange(1, 2**31 - 1))
        noise, t = wn_gen(
            fs=int(self.cfg.fs),
            duration=float(self.cfg.noise_duration_s),
            f_low=float(self.cfg.f_low),
            f_high=float(self.cfg.f_high),
            rng=np.random.default_rng(source_seed),
        )
        source_signal = _normalize_columns(noise)
        time_axis = np.asarray(t[:, 0], dtype=float)

        anc_result = self.qc.evaluate_anc(
            mgr=mgr,
            source_signal=source_signal,
            time_axis=time_axis,
            sec_ids=self.sampler.sec_ids,
        )

        x = np.asarray(anc_result["x"], dtype=float)
        if x.ndim != 2 or x.shape[1] < int(self.cfg.num_reference_mics):
            raise QCError("anc:unexpected_reference_shape")
        start_idx, start_s = self._sample_warmup_start(x.shape[0])
        end_idx = start_idx + self.cfg.ref_window_samples
        x_ref = x[start_idx:end_idx, : self.cfg.num_reference_mics].T.astype(np.float32)

        qc_metrics = {
            "nr_first_db": float(anc_result["metrics"]["nr_first_db"]),
            "nr_last_db": float(anc_result["metrics"]["nr_last_db"]),
            "nr_gain_db": float(anc_result["metrics"]["nr_gain_db"]),
            "direct_ratio_min": float(rir_metrics["direct_ratio_min"]),
            "control_ratio_min": float(rir_metrics["control_ratio_min"]),
            "feedback_ratio_max": float(rir_metrics["feedback_ratio_max"]),
            "mu_used": float(anc_result["mu"]),
            "source_seed": int(source_seed),
            "warmup_start_s": float(start_s),
            "warmup_start_index": int(start_idx),
        }

        return RoomSample(
            room_params=sampled,
            x_ref=x_ref,
            secondary_path=secondary_path,
            secondary_path_length=secondary_len,
            primary_to_ref_paths=primary_to_ref_paths,
            primary_to_ref_lengths=primary_to_ref_lengths,
            primary_to_error_path=primary_to_error_path,
            primary_to_error_length=primary_to_error_len,
            error_to_ref_paths=e2r_paths,
            error_to_ref_lengths=e2r_lengths,
            secondary_to_ref_paths=s2r_paths,
            secondary_to_ref_lengths=s2r_lengths,
            ref_to_ref_paths=r2r_paths,
            ref_to_ref_lengths=r2r_lengths,
            w_opt=np.asarray(anc_result["w_opt"], dtype=np.float32),
            w_full=np.asarray(anc_result["w_full"], dtype=np.float32),
            qc_metrics=qc_metrics,
        )

    def build_dataset(self) -> Path:
        output_path = Path(self.cfg.output_h5)
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()

        writer = HDF5DatasetWriter(self.cfg, output_path)
        accepted = 0
        attempts = 0
        try:
            while accepted < self.cfg.target_rooms and attempts < self.cfg.max_total_attempts:
                attempts += 1
                try:
                    sample = self._build_single_sample()
                    writer.write(accepted, sample)
                    accepted += 1
                    preview_mgr = self.sampler.build_manager(sample.room_params)
                    self.layout_previewer.update(preview_mgr, sample.room_params, accepted, attempts)
                    if accepted % max(int(self.cfg.progress_interval), 1) == 0:
                        print(
                            f"[Progress] accepted={accepted}/{self.cfg.target_rooms}, attempts={attempts}, "
                            f"pass_rate={accepted / max(attempts, 1):.3f}"
                        )
                except QCError as exc:
                    self._count_fail(str(exc))
                    if attempts % max(int(self.cfg.attempt_log_interval), 1) == 0:
                        top = sorted(self.failure_stats.items(), key=lambda kv: kv[1], reverse=True)[:1]
                        top_text = ", ".join(f"{k}:{v}" for k, v in top) if top else "none"
                        print(
                            f"[Attempt] attempts={attempts}, accepted={accepted}/{self.cfg.target_rooms}, "
                            f"pass_rate={accepted / max(attempts, 1):.3f}, top_fail={top_text}"
                        )
        finally:
            writer.close()
            self.layout_previewer.close()

        if accepted < self.cfg.target_rooms:
            raise RuntimeError(
                f"Only accepted {accepted} rooms before reaching max_total_attempts={self.cfg.max_total_attempts}. "
                f"Please increase attempts or relax geometry/QC thresholds."
            )

        compute_processed_features(output_path)

        print("Common failure reasons (Top8):")
        for reason, count in sorted(self.failure_stats.items(), key=lambda kv: kv[1], reverse=True)[:8]:
            print(f"  - {reason}: {count}")
        print(f"All done.\nOutput file: {output_path}")
        return output_path


@dataclass
class _CalibrationReplayCase:
    idx: int
    manager: RIRManager
    time_axis: np.ndarray
    reference_signal: np.ndarray
    desired_signal: np.ndarray
    w_h5: np.ndarray
    p_ref_paths: np.ndarray
    d_path: np.ndarray
    s_path: np.ndarray


def _sampled_room_from_h5(h5: h5py.File, idx: int) -> dict[str, Any]:
    room = h5["raw/room_params"]
    return {
        "room_size": np.asarray(room["room_size"][idx], dtype=float),
        "source_pos": np.asarray(room["source_position"][idx], dtype=float),
        "ref_positions": np.asarray(room["ref_positions"][idx], dtype=float),
        "sec_positions": np.asarray(room["sec_positions"][idx], dtype=float),
        "err_positions": np.asarray(room["err_positions"][idx], dtype=float),
        "ref_azimuth_deg": np.asarray(room["ref_azimuth_deg"][idx], dtype=float),
        "ref_radii": np.asarray(room["ref_radii"][idx], dtype=float),
        "sec_source_distance": float(room["sec_source_distance"][idx]),
        "err_source_distance": float(room["err_source_distance"][idx]),
        "sec_err_distance": float(room["sec_err_distance"][idx]),
        "primary_advance_margin_min": float(room["primary_advance_margin_min"][idx]),
        "secondary_feedback_margin_min": float(room["secondary_feedback_margin_min"][idx]),
        "sound_speed": float(room["sound_speed"][idx]),
        "absorption": float(room["material_absorption"][idx]),
        "image_order": int(room["image_source_order"][idx]),
        "layout_mode": _decode_layout_value(room["layout_mode"][idx]),
    }


def _extract_canonical_raw_paths(
    cfg: DatasetBuildConfig,
    mgr: RIRManager,
    sampled: dict[str, Any],
    source_id: int,
    ref_ids: np.ndarray,
    err_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    ref_positions = np.asarray(sampled["ref_positions"], dtype=float)
    err_id = int(err_ids[0])

    p_ref_paths = np.zeros((cfg.num_reference_mics, cfg.rir_store_len), dtype=np.float32)
    p_ref_lengths = np.zeros((cfg.num_reference_mics,), dtype=np.int32)
    for i, ref_id in enumerate(ref_ids):
        p_ref = np.asarray(mgr.get_reference_rir(source_id, int(ref_id)), dtype=float)
        p_ref_paths[i], p_ref_lengths[i] = _crop_rir(p_ref, cfg.rir_store_len)

    d_path = np.asarray(mgr.get_primary_rir(source_id, err_id), dtype=float)
    d_path_cropped, d_len = _crop_rir(d_path, cfg.rir_store_len)

    r2r_paths = np.zeros((len(cfg.canonical_r2r_pair_order), cfg.rir_store_len), dtype=np.float32)
    r2r_lengths = np.zeros((len(cfg.canonical_r2r_pair_order),), dtype=np.int32)
    for pair_idx, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        r2r = np.asarray(mgr.compute_transfer_rirs(ref_positions[i], ref_positions[j][None, :]), dtype=float).reshape(-1)
        r2r_paths[pair_idx], r2r_lengths[pair_idx] = _crop_rir(r2r, cfg.rir_store_len)
    return p_ref_paths, p_ref_lengths, d_path_cropped, int(d_len), r2r_paths, r2r_lengths


def _ensure_canonical_raw_inputs(h5: h5py.File, cfg: DatasetBuildConfig) -> None:
    raw = h5["raw"]
    n_rooms = int(raw["W_full"].shape[0])
    required = {
        "P_ref_paths": (n_rooms, cfg.num_reference_mics, cfg.rir_store_len),
        "P_ref_path_lengths": (n_rooms, cfg.num_reference_mics),
        "D_path": (n_rooms, cfg.rir_store_len),
        "D_path_length": (n_rooms,),
        "R2R_paths": (n_rooms, len(cfg.canonical_r2r_pair_order), cfg.rir_store_len),
        "R2R_path_lengths": (n_rooms, len(cfg.canonical_r2r_pair_order)),
    }
    missing = [name for name in required if name not in raw]
    if not missing:
        raw.attrs["r2r_pair_order_json"] = json.dumps(list(cfg.canonical_r2r_pair_order), ensure_ascii=False)
        return

    dtype_map = {
        "P_ref_paths": "f4",
        "P_ref_path_lengths": "i4",
        "D_path": "f4",
        "D_path_length": "i4",
        "R2R_paths": "f4",
        "R2R_path_lengths": "i4",
    }
    for name in missing:
        raw.create_dataset(name, shape=required[name], dtype=dtype_map[name])
    raw.attrs["r2r_pair_order_json"] = json.dumps(list(cfg.canonical_r2r_pair_order), ensure_ascii=False)

    sampler = AcousticScenarioSampler(cfg, np.random.default_rng(int(cfg.random_seed)))
    for idx in range(n_rooms):
        sampled = _sampled_room_from_h5(h5, idx)
        mgr = sampler.build_manager(sampled)
        mgr.build(verbose=False)
        p_ref_paths, p_ref_lengths, d_path, d_len, r2r_paths, r2r_lengths = _extract_canonical_raw_paths(
            cfg=cfg,
            mgr=mgr,
            sampled=sampled,
            source_id=int(sampler.source_id),
            ref_ids=sampler.ref_ids,
            err_ids=sampler.err_ids,
        )
        raw["P_ref_paths"][idx] = p_ref_paths
        raw["P_ref_path_lengths"][idx] = p_ref_lengths
        raw["D_path"][idx] = d_path
        raw["D_path_length"][idx] = int(d_len)
        raw["R2R_paths"][idx] = r2r_paths
        raw["R2R_path_lengths"][idx] = r2r_lengths
        if (idx + 1) % 50 == 0 or idx + 1 == n_rooms:
            print(f"[Canonical Raw] restored {idx + 1}/{n_rooms} rooms")


def _build_calibration_replay_cases(h5: h5py.File, cfg: DatasetBuildConfig, room_indices: np.ndarray) -> list[_CalibrationReplayCase]:
    raw = h5["raw"]
    qc = raw["qc_metrics"]
    sampler = AcousticScenarioSampler(cfg, np.random.default_rng(int(cfg.random_seed)))
    cases: list[_CalibrationReplayCase] = []
    for idx in np.asarray(room_indices, dtype=np.int64):
        sampled = _sampled_room_from_h5(h5, int(idx))
        mgr = sampler.build_manager(sampled)
        mgr.build(verbose=False)
        source_seed = int(qc["source_seed"][idx])
        noise, t = wn_gen(
            fs=int(cfg.fs),
            duration=float(cfg.noise_duration_s),
            f_low=float(cfg.f_low),
            f_high=float(cfg.f_high),
            rng=np.random.default_rng(source_seed),
        )
        source_signal = _normalize_columns(noise)
        time_axis = np.asarray(t[:, 0], dtype=float)
        reference_signal = _normalize_columns(mgr.calculate_reference_signal(source_signal, len(time_axis)))
        desired_signal = mgr.calculate_desired_signal(source_signal, len(time_axis))
        cases.append(
            _CalibrationReplayCase(
                idx=int(idx),
                manager=mgr,
                time_axis=time_axis,
                reference_signal=reference_signal,
                desired_signal=desired_signal,
                w_h5=np.asarray(raw["W_full"][idx], dtype=np.float32),
                p_ref_paths=np.asarray(raw["P_ref_paths"][idx], dtype=np.float32),
                d_path=np.asarray(raw["D_path"][idx], dtype=np.float32),
                s_path=np.asarray(raw["S_paths"][idx], dtype=np.float32),
            )
        )
    return cases


def _replay_metrics_for_case(case: _CalibrationReplayCase, w_ai: np.ndarray, cfg: DatasetBuildConfig, early_window_s: float) -> dict[str, float]:
    params = {
        "time": case.time_axis,
        "rir_manager": case.manager,
        "L": int(cfg.filter_len),
        "mu": float(cfg.mu_candidates[0]),
        "reference_signal": case.reference_signal,
        "desired_signal": case.desired_signal,
        "verbose": False,
        "normalized_update": bool(cfg.anc_normalized_update),
        "norm_epsilon": float(cfg.anc_norm_epsilon),
    }
    e_zero = np.asarray(cfxlms(params)["err_hist"], dtype=float)[:, 0]
    e_ai = np.asarray(
        _cfxlms_with_init(
            case.time_axis,
            case.manager,
            int(cfg.filter_len),
            float(cfg.mu_candidates[0]),
            case.reference_signal,
            case.desired_signal,
            w_init=w_ai,
            normalized_update=bool(cfg.anc_normalized_update),
            norm_epsilon=float(cfg.anc_norm_epsilon),
        )["err_hist"],
        dtype=float,
    )[:, 0]
    e_h5 = np.asarray(
        _cfxlms_with_init(
            case.time_axis,
            case.manager,
            int(cfg.filter_len),
            float(cfg.mu_candidates[0]),
            case.reference_signal,
            case.desired_signal,
            w_init=case.w_h5,
            normalized_update=bool(cfg.anc_normalized_update),
            norm_epsilon=float(cfg.anc_norm_epsilon),
        )["err_hist"],
        dtype=float,
    )[:, 0]
    window_samples = min(max(32, int(round(float(early_window_s) * float(cfg.fs)))), max(int(len(case.time_axis) // 2), 32))
    t_db, db_zero = _rolling_mse_db(e_zero, int(cfg.fs), window_samples=window_samples)
    _, db_ai = _rolling_mse_db(e_ai, int(cfg.fs), window_samples=window_samples)
    _, db_h5 = _rolling_mse_db(e_h5, int(cfg.fs), window_samples=window_samples)
    early_mask = t_db <= float(early_window_s)
    if not np.any(early_mask):
        early_mask = np.ones_like(t_db, dtype=bool)
    return {
        "ai_vs_zero_db": float(np.mean(db_zero[early_mask] - db_ai[early_mask])),
        "h5_vs_zero_db": float(np.mean(db_zero[early_mask] - db_h5[early_mask])),
        "ai_to_h5_gap_db": float(np.mean(np.abs(db_ai[early_mask] - db_h5[early_mask]))),
    }


def _calibrate_canonical_regularization(
    h5: h5py.File,
    cfg: DatasetBuildConfig,
    q_full_len: int,
) -> dict[str, Any]:
    n_rooms = int(h5["raw/W_full"].shape[0])
    indices = np.arange(n_rooms, dtype=np.int64)
    rng = np.random.default_rng(int(cfg.random_seed))
    rng.shuffle(indices)
    split = int(n_rooms * 0.8)
    train_idx = np.sort(indices[:split])
    calib_idx = np.asarray(train_idx[: min(int(cfg.canonical_calibration_rooms), len(train_idx))], dtype=np.int64)
    cases = _build_calibration_replay_cases(h5, cfg, calib_idx)
    best: dict[str, Any] | None = None
    for lambda_w in cfg.canonical_lambda_w_candidates:
        gains = []
        gaps = []
        for case in cases:
            q_full = _equivalent_q_from_w_full(case.w_h5, case.p_ref_paths, int(q_full_len), int(cfg.filter_len))
            w_canon = _solve_w_canonical_from_q(q_full, case.p_ref_paths, int(cfg.filter_len), float(lambda_w), int(q_full_len))
            metrics = _replay_metrics_for_case(case, w_canon, cfg, float(cfg.canonical_replay_early_window_s))
            gains.append(float(metrics["ai_vs_zero_db"]))
            gaps.append(float(metrics["ai_to_h5_gap_db"]))
        summary = {
            "q_target_source": "h5_equivalent",
            "lambda_q_scale": 0.0,
            "lambda_w": float(lambda_w),
            "ai_vs_zero_db_mean": float(np.mean(gains)),
            "ai_to_h5_gap_db_mean": float(np.mean(gaps)),
            "calibration_indices": calib_idx.tolist(),
        }
        if (
            best is None
            or summary["ai_to_h5_gap_db_mean"] < best["ai_to_h5_gap_db_mean"] - 1.0e-9
            or (
                np.isclose(summary["ai_to_h5_gap_db_mean"], best["ai_to_h5_gap_db_mean"])
                and summary["ai_vs_zero_db_mean"] > best["ai_vs_zero_db_mean"] + 1.0e-9
            )
        ):
            best = summary
        print(
            "[Canonical Calib] "
            f"q_target=h5_equivalent, lambda_w={lambda_w:.1e}, "
            f"gain={summary['ai_vs_zero_db_mean']:.3f} dB, gap={summary['ai_to_h5_gap_db_mean']:.3f} dB"
        )
    if best is None:
        raise RuntimeError("Canonical calibration did not evaluate any candidate.")
    if best["ai_vs_zero_db_mean"] <= 0.0:
        raise RuntimeError(
            "Exact canonical Q->W replay did not improve over zero-init on the calibration rooms. "
            "Please revisit the canonical construction before training."
        )
    return best


def compute_processed_features(h5_path: str | Path) -> dict[str, Any]:
    h5_path = Path(h5_path)
    with h5py.File(str(h5_path), "a") as h5:
        cfg = DatasetBuildConfig(**json.loads(h5.attrs["config_json"]))
        feature_proc = FeatureProcessor(cfg)
        raw = h5["raw"]
        _ensure_canonical_raw_inputs(h5, cfg)

        x_ref = np.asarray(raw["x_ref"], dtype=np.float32)
        s_paths = np.asarray(raw["S_paths"], dtype=np.float32)
        p_ref_paths = np.asarray(raw["P_ref_paths"], dtype=np.float32)
        d_paths = np.asarray(raw["D_path"], dtype=np.float32)
        e2r = np.asarray(raw["E2R_paths"], dtype=np.float32)
        s2r = np.asarray(raw["S2R_paths"], dtype=np.float32)
        r2r = np.asarray(raw["R2R_paths"], dtype=np.float32)
        room = raw["room_params"]
        room_ref_positions = np.asarray(room["ref_positions"], dtype=np.float32)
        room_sound_speed = np.asarray(room["sound_speed"], dtype=np.float32)
        w_opt = np.asarray(raw["W_opt"], dtype=np.float32)
        n_rooms = int(x_ref.shape[0])
        q_full_len = int(cfg.filter_len + cfg.rir_store_len - 1)

        calibration = _calibrate_canonical_regularization(h5, cfg, q_full_len=q_full_len)
        best_lambda_q_scale = float(calibration["lambda_q_scale"])
        best_lambda_w = float(calibration["lambda_w"])

        q_full = np.zeros((n_rooms, q_full_len), dtype=np.float32)
        w_canon = np.zeros((n_rooms, 1, cfg.num_reference_mics, cfg.filter_len), dtype=np.float32)
        for i in range(n_rooms):
            q_full[i] = _equivalent_q_from_w_full(
                raw["W_full"][i],
                p_ref_paths[i],
                q_full_len=q_full_len,
                filter_len=int(cfg.filter_len),
            )
            w_canon[i] = _solve_w_canonical_from_q(
                q_full[i],
                p_ref_paths[i],
                filter_len=int(cfg.filter_len),
                lambda_w=best_lambda_w,
                q_full_len=q_full_len,
            )

        q_keep = _choose_keep_len(
            q_full,
            cfg.canonical_q_energy_ratio,
            cfg.canonical_q_length_quantile,
            cfg.canonical_q_min_keep_len,
            min(cfg.canonical_q_max_keep_len, q_full_len),
        )

        s_keep = _choose_keep_len(
            s_paths,
            cfg.truncate_energy_ratio,
            cfg.truncate_length_quantile,
            cfg.min_path_keep_len,
            cfg.max_path_keep_len,
        )
        e2r_keep = _choose_keep_len(
            e2r,
            cfg.truncate_energy_ratio,
            cfg.truncate_length_quantile,
            cfg.min_path_keep_len,
            cfg.max_path_keep_len,
        )
        s2r_keep = _choose_keep_len(
            s2r,
            cfg.truncate_energy_ratio,
            cfg.truncate_length_quantile,
            cfg.min_path_keep_len,
            cfg.max_path_keep_len,
        )
        w_keep = _choose_keep_len(
            w_opt,
            cfg.w_truncate_energy_ratio,
            cfg.truncate_length_quantile,
            cfg.min_w_keep_len,
            cfg.max_w_keep_len,
        )

        gcc = np.zeros((n_rooms, 3, cfg.gcc_truncated_len), dtype=np.float32)
        psd = np.zeros((n_rooms, cfg.psd_nfft // 2 + 1), dtype=np.float32)
        for i in range(n_rooms):
            gcc[i] = feature_proc.compute_gcc_phat(
                x_ref[i],
                ref_positions=room_ref_positions[i],
                sound_speed=float(room_sound_speed[i]),
            )
            psd[i] = feature_proc.compute_psd_features(x_ref[i, 0])

        acoustic_channel_names = [
            "S_err",
            "D_err",
            "P_ref_0",
            "P_ref_1",
            "P_ref_2",
            "E2R_0",
            "E2R_1",
            "E2R_2",
            "S2R_0",
            "S2R_1",
            "S2R_2",
            "R2R_01",
            "R2R_02",
            "R2R_12",
        ]
        lowband_bins = int(np.sum(feature_proc.acoustic_mask))
        acoustic_ri = np.zeros((n_rooms, 2 * len(acoustic_channel_names), lowband_bins), dtype=np.float32)
        acoustic_mp = np.zeros((n_rooms, 2 * len(acoustic_channel_names), lowband_bins), dtype=np.float32)
        for i in range(n_rooms):
            traces = [
                np.asarray(s_paths[i], dtype=np.float32),
                np.asarray(d_paths[i], dtype=np.float32),
                np.asarray(p_ref_paths[i, 0], dtype=np.float32),
                np.asarray(p_ref_paths[i, 1], dtype=np.float32),
                np.asarray(p_ref_paths[i, 2], dtype=np.float32),
                np.asarray(e2r[i, 0], dtype=np.float32),
                np.asarray(e2r[i, 1], dtype=np.float32),
                np.asarray(e2r[i, 2], dtype=np.float32),
                np.asarray(s2r[i, 0], dtype=np.float32),
                np.asarray(s2r[i, 1], dtype=np.float32),
                np.asarray(s2r[i, 2], dtype=np.float32),
                np.asarray(r2r[i, 0], dtype=np.float32),
                np.asarray(r2r[i, 1], dtype=np.float32),
                np.asarray(r2r[i, 2], dtype=np.float32),
            ]
            for ch_idx, trace in enumerate(traces):
                spec = feature_proc.lowband_complex_spectrum(trace)
                acoustic_ri[i, 2 * ch_idx : 2 * ch_idx + 2] = feature_proc.encode_complex_ri(spec)
                acoustic_mp[i, 2 * ch_idx : 2 * ch_idx + 2] = feature_proc.encode_complex_mp(spec)

        path_features = np.concatenate(
            [
                s_paths[:, :s_keep],
                e2r[:, :, :e2r_keep].reshape(n_rooms, -1),
                s2r[:, :, :s2r_keep].reshape(n_rooms, -1),
            ],
            axis=1,
        ).astype(np.float32)
        w_targets = w_opt[:, :, :w_keep].reshape(n_rooms, -1).astype(np.float32)
        q_target = q_full[:, :q_keep].astype(np.float32)

        if "processed" in h5:
            del h5["processed"]
        processed = h5.create_group("processed")
        processed.create_dataset("gcc_phat", data=gcc, dtype="f4")
        processed.create_dataset("psd_features", data=psd, dtype="f4")
        processed.create_dataset("acoustic_feature_ri", data=acoustic_ri, dtype="f4")
        processed.create_dataset("acoustic_feature_mp", data=acoustic_mp, dtype="f4")
        processed.create_dataset("path_features", data=path_features, dtype="f4")
        processed.create_dataset("w_targets", data=w_targets, dtype="f4")
        processed.create_dataset("q_target", data=q_target, dtype="f4")
        processed.create_dataset("w_canon", data=w_canon, dtype="f4")

        slices = {
            "secondary_to_error": [0, int(s_keep)],
            "error_to_reference": [int(s_keep), int(s_keep + cfg.num_reference_mics * e2r_keep)],
            "secondary_to_reference": [
                int(s_keep + cfg.num_reference_mics * e2r_keep),
                int(path_features.shape[1]),
            ],
        }
        target_slices = {
            f"ref_{idx}": [int(idx * w_keep), int((idx + 1) * w_keep)]
            for idx in range(cfg.num_reference_mics)
        }
        processed.attrs["s_keep_len"] = int(s_keep)
        processed.attrs["e2r_keep_len"] = int(e2r_keep)
        processed.attrs["s2r_keep_len"] = int(s2r_keep)
        processed.attrs["w_keep_len"] = int(w_keep)
        processed.attrs["path_feature_dim"] = int(path_features.shape[1])
        processed.attrs["w_target_dim"] = int(w_targets.shape[1])
        processed.attrs["path_feature_slices_json"] = json.dumps(slices, ensure_ascii=False)
        processed.attrs["w_target_slices_json"] = json.dumps(target_slices, ensure_ascii=False)
        processed.attrs["q_keep_len"] = int(q_keep)
        processed.attrs["q_full_len"] = int(q_full_len)
        processed.attrs["q_target_dim"] = int(q_target.shape[1])
        processed.attrs["lambda_q_scale"] = float(best_lambda_q_scale)
        processed.attrs["lambda_w"] = float(best_lambda_w)
        processed.attrs["q_target_source"] = str(calibration.get("q_target_source", "h5_equivalent"))
        processed.attrs["q_local_half_width"] = int(cfg.canonical_q_local_half_width)
        processed.attrs["q_tail_basis_dim"] = int(cfg.canonical_q_tail_basis_dim)
        processed.attrs["r2r_pair_order_json"] = json.dumps(list(cfg.canonical_r2r_pair_order), ensure_ascii=False)
        processed.attrs["canonical_calibration_json"] = json.dumps(calibration, ensure_ascii=False)
        processed.attrs["acoustic_feature_nfft"] = int(cfg.acoustic_feature_nfft)
        processed.attrs["acoustic_feature_low_hz"] = float(cfg.acoustic_feature_low_hz)
        processed.attrs["acoustic_feature_high_hz"] = float(cfg.acoustic_feature_high_hz)
        processed.attrs["acoustic_feature_bins"] = int(lowband_bins)
        processed.attrs["acoustic_feature_channel_names_json"] = json.dumps(acoustic_channel_names, ensure_ascii=False)
        processed.attrs["acoustic_feature_ri_channels"] = int(acoustic_ri.shape[1])
        processed.attrs["acoustic_feature_mp_channels"] = int(acoustic_mp.shape[1])

    keep_summary = {
        "s_keep_len": int(s_keep),
        "e2r_keep_len": int(e2r_keep),
        "s2r_keep_len": int(s2r_keep),
        "w_keep_len": int(w_keep),
        "q_keep_len": int(q_keep),
        "q_full_len": int(q_full_len),
        "path_feature_dim": int(path_features.shape[1]),
        "w_target_dim": int(w_targets.shape[1]),
        "q_target_dim": int(q_target.shape[1]),
        "lambda_q_scale": float(best_lambda_q_scale),
        "lambda_w": float(best_lambda_w),
        "q_target_source": str(calibration.get("q_target_source", "h5_equivalent")),
        "acoustic_feature_bins": int(lowband_bins),
        "acoustic_feature_ri_channels": int(acoustic_ri.shape[1]),
        "acoustic_feature_mp_channels": int(acoustic_mp.shape[1]),
    }
    print("Processed features updated:", keep_summary)
    return keep_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build simplified CFxLMS ANC dataset (3 refs, 1 speaker, 1 error mic).")
    parser.add_argument("--num-rooms", type=int, default=1000, help="Number of accepted rooms to generate.")
    parser.add_argument("--max-attempts", type=int, default=30000, help="Maximum total sampling attempts.")
    parser.add_argument(
        "--output-h5",
        type=str,
        default=str(ROOT_DIR / "python_scripts" / "cfxlms_qc_dataset_single_control.h5"),
        help="Output HDF5 path.",
    )
    parser.add_argument("--seed", type=int, default=20260331, help="Random seed.")
    parser.add_argument("--progress-interval", type=int, default=20, help="Progress print interval.")
    parser.add_argument("--attempt-log-interval", type=int, default=10, help="Attempt log interval.")
    parser.add_argument("--preview-interval", type=int, default=1, help="Layout preview refresh interval.")
    parser.add_argument("--no-preview-layouts", action="store_true", help="Disable dynamic layout preview.")
    parser.add_argument("--min-nr-last-db", type=float, default=15.0, help="ANC QC threshold for final NR dB.")
    parser.add_argument("--min-nr-gain-db", type=float, default=15.0, help="ANC QC threshold for NR gain dB.")
    parser.add_argument("--process-only", action="store_true", help="Only rebuild processed features for an existing HDF5 file.")
    return parser


def config_from_args(args: argparse.Namespace) -> DatasetBuildConfig:
    return DatasetBuildConfig(
        target_rooms=int(args.num_rooms),
        max_total_attempts=int(args.max_attempts),
        random_seed=int(args.seed),
        output_h5=str(args.output_h5),
        progress_interval=int(args.progress_interval),
        attempt_log_interval=int(args.attempt_log_interval),
        layout_preview=not bool(args.no_preview_layouts),
        layout_preview_interval=int(args.preview_interval),
        min_nr_last_db=float(args.min_nr_last_db),
        min_nr_gain_db=float(args.min_nr_gain_db),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.process_only:
        compute_processed_features(args.output_h5)
        return 0

    cfg = config_from_args(args)
    print("Starting simplified CFxLMS dataset build...")
    print("========== Run Config ==========")
    print(f"Reference mics: {cfg.num_reference_mics}")
    print(f"Secondary speakers: {cfg.num_secondary_speakers}")
    print(f"Error microphones: {cfg.num_error_mics}")
    print(f"fs: {cfg.fs} Hz")
    print(f"duration: {cfg.noise_duration_s:.2f} s")
    print(f"filter_len: {cfg.filter_len}")
    print(f"mu_candidates: {cfg.mu_candidates}")
    print(f"ANC thresholds: min_nr_last_db={cfg.min_nr_last_db:.2f}, min_nr_gain_db={cfg.min_nr_gain_db:.2f}")
    print(
        "Causality margins: "
        f"primary>={cfg.min_primary_advance_margin_m:.2f} m, "
        f"secondary_feedback>={cfg.min_secondary_feedback_margin_m:.2f} m"
    )
    print(f"target rooms: {cfg.target_rooms}")
    print(f"max attempts: {cfg.max_total_attempts}")
    print(f"output: {cfg.output_h5}")
    print("================================")

    builder = ANCDatasetBuilder(cfg)
    builder.build_dataset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

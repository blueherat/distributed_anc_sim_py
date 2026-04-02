from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from py_anc.acoustics import RIRManager
from py_anc.algorithms import cfxlms
from py_anc.utils import wn_gen
from python_scripts.cfxlms_single_control_dataset_impl import (
    FeatureProcessor,
    QCError,
    _build_laguerre_basis,
    _cfxlms_with_init,
    _choose_keep_len,
    _crop_rir,
    _normalize_columns,
    _project_tail_coeffs_1d_np,
    _rolling_mse_db,
)


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
    room_size_max: float = 6.0
    wall_margin: float = 0.25
    source_wall_margin: float = 0.45

    sound_speed_min: float = 338.0
    sound_speed_max: float = 346.0
    reflection_room_probability: float = 0.55
    direct_room_absorption_range: tuple[float, float] = (0.68, 0.88)
    reflection_room_absorption_range: tuple[float, float] = (0.30, 0.58)
    direct_room_image_order_choices: tuple[int, ...] = (0, 1)
    direct_room_image_order_probs: tuple[float, ...] = (0.25, 0.75)
    reflection_room_image_order_choices: tuple[int, ...] = (1, 2)
    reflection_room_image_order_probs: tuple[float, ...] = (0.80, 0.20)

    num_reference_mics: int = 3
    num_secondary_speakers: int = 3
    num_error_mics: int = 3

    min_secondary_node_distance: float = 0.40
    min_device_distance: float = 0.10
    min_source_device_distance: float = 0.35
    neighbor_causality_tolerance: float = 0.0
    max_causality_violations: int = 0

    layout_mode_choices: tuple[str, ...] = ("source_radial", "wall_strip", "corner_cluster", "free_scatter")
    layout_mode_probs: tuple[float, ...] = (0.15, 0.30, 0.10, 0.45)
    azimuth_spacing_base: tuple[float, ...] = (0.0, 120.0, 240.0)
    azimuth_spacing_jitter: float = 22.0

    ref_radius_range: tuple[float, float] = (0.35, 0.55)
    sec_delta_range: tuple[float, float] = (0.18, 0.38)
    err_delta_range: tuple[float, float] = (0.18, 0.55)
    z_offset_range: tuple[float, float] = (-0.10, 0.10)
    node_ref_to_sec_range: tuple[float, float] = (0.10, 0.32)
    node_err_extra_range: tuple[float, float] = (0.10, 0.45)
    secondary_height_range: tuple[float, float] = (0.95, 1.75)
    node_direction_tilt_range: tuple[float, float] = (-0.04, 0.04)
    source_barycentric_alpha: float = 2.8
    min_secondary_triangle_area: float = 0.18
    min_source_angle_separation_deg: float = 22.0

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
    psd_nfft: int = 256

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

    secondary_condition_fft: int = 2048
    secondary_condition_p95_max: float = 1.0e3
    secondary_condition_median_max: float = 1.0e2

    random_seed: int = 20260401
    progress_interval: int = 20
    attempt_log_interval: int = 10
    layout_preview: bool = True
    layout_preview_interval: int = 1

    output_h5: str = str(ROOT_DIR / "python_scripts" / "cfxlms_qc_dataset_multicontrol.h5")

    @property
    def ref_window_samples(self) -> int:
        return int(round(self.fs * self.ref_window_ms / 1000.0))


@dataclass
class RoomSample:
    room_params: dict[str, Any]
    x_ref: np.ndarray
    p_ref_paths: np.ndarray
    p_ref_lengths: np.ndarray
    d_paths: np.ndarray
    d_lengths: np.ndarray
    s_matrix_paths: np.ndarray
    s_matrix_lengths: np.ndarray
    s2r_paths: np.ndarray
    s2r_lengths: np.ndarray
    r2r_paths: np.ndarray
    r2r_lengths: np.ndarray
    w_opt: np.ndarray
    w_full: np.ndarray
    qc_metrics: dict[str, float]


def _min_pairwise_distance(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return float("inf")
    min_dist = float("inf")
    for i, j in combinations(range(pts.shape[0]), 2):
        min_dist = min(min_dist, float(np.linalg.norm(pts[i] - pts[j])))
    return min_dist


def _rolling_mse_db_multichannel(sig: np.ndarray, fs: int, window_samples: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(sig, dtype=float)
    if arr.ndim == 1:
        return _rolling_mse_db(arr, fs=int(fs), window_samples=int(window_samples))
    mean_power = np.mean(arr**2, axis=1)
    return _rolling_mse_db(np.sqrt(np.maximum(mean_power, 0.0)), fs=int(fs), window_samples=int(window_samples))


def _canonical_q_from_matrix_paths(d_paths: np.ndarray, s_matrix_paths: np.ndarray, q_full_len: int, lambda_q_scale: float) -> np.ndarray:
    d = np.asarray(d_paths, dtype=np.float64)
    s = np.asarray(s_matrix_paths, dtype=np.float64)
    if d.shape[0] != s.shape[1]:
        raise ValueError(f"d_paths shape {d.shape} does not match s_matrix_paths shape {s.shape}.")
    n_fft = 1 << (int(q_full_len) + int(max(d.shape[-1], s.shape[-1])) - 1).bit_length()
    d_f = np.fft.rfft(d, n=n_fft, axis=-1)
    s_f = np.fft.rfft(s, n=n_fft, axis=-1)
    q_f = np.zeros((s.shape[0], d_f.shape[-1]), dtype=np.complex128)
    for fi in range(d_f.shape[-1]):
        h = s_f[:, :, fi].T
        sigma = np.linalg.svd(h, compute_uv=False)
        sigma_max_sq = float(np.max(np.abs(sigma) ** 2)) if sigma.size else 0.0
        lam = float(lambda_q_scale) * max(sigma_max_sq, 1.0e-12)
        lhs = h.conj().T @ h + lam * np.eye(h.shape[1], dtype=np.complex128)
        rhs = -(h.conj().T @ d_f[:, fi])
        try:
            q_f[:, fi] = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            q_f[:, fi] = np.linalg.pinv(lhs) @ rhs
    q = np.fft.irfft(q_f, n=n_fft, axis=-1)[:, : int(q_full_len)]
    return q.astype(np.float32)


def _solve_w_canonical_from_q(q_full: np.ndarray, p_ref_paths: np.ndarray, filter_len: int, lambda_w: float, q_full_len: int) -> np.ndarray:
    q = np.asarray(q_full, dtype=np.float64)
    p = np.asarray(p_ref_paths, dtype=np.float64)
    n_fft = 1 << (int(q_full_len) + int(p.shape[-1]) - 1).bit_length()
    q_f = np.fft.rfft(q[:, : int(q_full_len)], n=n_fft, axis=-1)
    p_f = np.fft.rfft(p, n=n_fft, axis=-1)
    denom = np.sum(np.abs(p_f) ** 2, axis=0) + float(lambda_w)
    w_f = np.zeros((q.shape[0], p.shape[0], p_f.shape[-1]), dtype=np.complex128)
    for sec_idx in range(q.shape[0]):
        w_f[sec_idx] = (q_f[sec_idx][None, :] * np.conj(p_f)) / denom[None, :]
    w_time = np.fft.irfft(w_f, n=n_fft, axis=-1)[..., : int(filter_len)]
    return w_time.astype(np.float32)


def _plant_residual_ratio(q_full: np.ndarray, d_paths: np.ndarray, s_matrix_paths: np.ndarray) -> float:
    q = np.asarray(q_full, dtype=np.float64)
    d = np.asarray(d_paths, dtype=np.float64)
    s = np.asarray(s_matrix_paths, dtype=np.float64)
    out_len = int(q.shape[-1] + s.shape[-1] - 1)
    residual = np.zeros((d.shape[0], out_len), dtype=np.float64)
    residual[:, : d.shape[-1]] = d
    for err_idx in range(d.shape[0]):
        for sec_idx in range(q.shape[0]):
            residual[err_idx] += np.convolve(q[sec_idx], s[sec_idx, err_idx], mode="full")[:out_len]
    denom = float(np.sum(d**2)) + np.finfo(float).eps
    return float(np.sqrt(np.sum(residual**2) / denom))


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
        except Exception as exc:  # pragma: no cover
            print(f"[Preview] disabled: {type(exc).__name__}: {exc}")
            self.enabled = False
            return
        self.plt = plt
        try:
            self.plt.ion()
            self.fig, self.ax = self.plt.subplots(figsize=(9, 6))
        except Exception as exc:  # pragma: no cover
            print(f"[Preview] disabled: {type(exc).__name__}: {exc}")
            self.enabled = False

    def _annotate_devices(self, mgr: RIRManager) -> None:
        groups = (
            ("P", mgr.primary_speakers, "darkred"),
            ("R", mgr.reference_microphones, "purple"),
            ("S", mgr.secondary_speakers, "darkgreen"),
            ("E", mgr.error_microphones, "navy"),
        )
        for prefix, table, color in groups:
            for did, pos in table.items():
                xy = np.asarray(pos, dtype=float)
                self.ax.text(xy[0], xy[1], f"{prefix}{int(did)}", color=color, fontsize=8)

    def _draw_node_links(self, sampled: dict[str, Any]) -> None:
        ref_positions = np.asarray(sampled["ref_positions"], dtype=float)
        sec_positions = np.asarray(sampled["sec_positions"], dtype=float)
        err_positions = np.asarray(sampled["err_positions"], dtype=float)
        for idx in range(min(len(ref_positions), len(sec_positions), len(err_positions))):
            color = f"C{idx}"
            self.ax.plot(
                [ref_positions[idx, 0], sec_positions[idx, 0], err_positions[idx, 0]],
                [ref_positions[idx, 1], sec_positions[idx, 1], err_positions[idx, 1]],
                color=color,
                alpha=0.45,
                linewidth=1.2,
            )

    def update(self, mgr: RIRManager, sampled: dict[str, Any], accepted: int, attempts: int) -> None:
        if not self.enabled or accepted <= 0:
            return
        if accepted % max(int(self.cfg.layout_preview_interval), 1) != 0:
            return
        self.ax.clear()
        mgr.plot_layout_2d(ax=self.ax)
        self._annotate_devices(mgr)
        self._draw_node_links(sampled)
        source_pos = np.asarray(sampled["source_pos"], dtype=float)
        sec_positions = np.asarray(sampled["sec_positions"], dtype=float)
        src_sec_dist = np.linalg.norm(sec_positions - source_pos[None, :], axis=1)
        info_lines = [
            f"accepted={accepted}/{self.cfg.target_rooms}",
            f"attempt={attempts}",
            f"layout={sampled.get('layout_mode', '-')}",
            f"src-sec={float(np.min(src_sec_dist)):.2f}-{float(np.max(src_sec_dist)):.2f} m",
            f"abs={float(sampled['absorption']):.2f}, order={int(sampled['image_order'])}",
        ]
        self.ax.text(
            1.02,
            0.98,
            "\n".join(info_lines),
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.90},
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self) -> None:
        if self.enabled and self.fig is not None:  # pragma: no cover
            self.plt.close(self.fig)


class AcousticScenarioSampler:
    def __init__(self, cfg: DatasetBuildConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.source_id = 101
        self.ref_ids = np.array([401, 402, 403], dtype=int)
        self.sec_ids = np.array([201, 202, 203], dtype=int)
        self.err_ids = np.array([301, 302, 303], dtype=int)

    def _all_inside_room(self, points: np.ndarray, room_size: np.ndarray) -> bool:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        low = np.full(3, float(self.cfg.wall_margin), dtype=float)
        high = np.asarray(room_size, dtype=float) - low
        return bool(np.all(pts >= low) and np.all(pts <= high))

    @staticmethod
    def _triangle_area_xy(points: np.ndarray) -> float:
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 3:
            return 0.0
        a, b, c = pts[0, :2], pts[1, :2], pts[2, :2]
        return float(0.5 * abs(np.cross(b - a, c - a)))

    def _source_angle_separation_ok(self, source_pos: np.ndarray, sec_positions: np.ndarray) -> bool:
        src_xy = np.asarray(source_pos, dtype=float)[:2]
        sec_xy = np.asarray(sec_positions, dtype=float)[:, :2]
        vec = sec_xy - src_xy[None, :]
        norms = np.linalg.norm(vec, axis=1)
        if np.any(norms <= 1.0e-6):
            return False
        angles = (np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) + 360.0) % 360.0
        min_sep = 180.0
        for i, j in combinations(range(len(angles)), 2):
            diff = abs(float(angles[i] - angles[j]))
            min_sep = min(min_sep, min(diff, 360.0 - diff))
        return bool(min_sep >= float(self.cfg.min_source_angle_separation_deg))

    def _sample_room_size(self) -> np.ndarray:
        return self.rng.uniform(self.cfg.room_size_min, self.cfg.room_size_max, size=3).astype(float)

    def _sample_room_acoustics(self) -> tuple[float, int]:
        if float(self.rng.random()) < float(self.cfg.reflection_room_probability):
            absorption = self.rng.uniform(*self.cfg.reflection_room_absorption_range)
            image_order = int(self.rng.choice(self.cfg.reflection_room_image_order_choices, p=self.cfg.reflection_room_image_order_probs))
        else:
            absorption = self.rng.uniform(*self.cfg.direct_room_absorption_range)
            image_order = int(self.rng.choice(self.cfg.direct_room_image_order_choices, p=self.cfg.direct_room_image_order_probs))
        return float(absorption), int(image_order)

    def _sample_layout_mode(self) -> str:
        return str(self.rng.choice(self.cfg.layout_mode_choices, p=self.cfg.layout_mode_probs))

    def _sample_source_position(self, room_size: np.ndarray, xy_margin: float | None = None) -> np.ndarray:
        margin_xy = float(self.cfg.source_wall_margin if xy_margin is None else xy_margin)
        margin_xy = max(margin_xy, float(self.cfg.wall_margin))
        z_margin = max(float(self.cfg.wall_margin + 0.10), 0.55)
        low = np.array([margin_xy, margin_xy, z_margin], dtype=float)
        high = room_size - low
        if np.any(high <= low):
            raise QCError("geometry:room_too_small_for_source_sampling")
        return self.rng.uniform(low, high).astype(float)

    def _sample_source_inside_secondary_triangle(self, sec_positions: np.ndarray, room_size: np.ndarray) -> np.ndarray:
        sec_positions = np.asarray(sec_positions, dtype=float)
        low = np.array(
            [
                float(self.cfg.source_wall_margin),
                float(self.cfg.source_wall_margin),
                max(float(self.cfg.wall_margin + 0.10), 0.55),
            ],
            dtype=float,
        )
        high = room_size - low
        for _ in range(80):
            weights = self.rng.dirichlet(np.full(self.cfg.num_secondary_speakers, float(self.cfg.source_barycentric_alpha), dtype=float))
            xy = weights @ sec_positions[:, :2]
            z = float(np.clip(float(weights @ sec_positions[:, 2]) + self.rng.uniform(-0.10, 0.10), low[2], high[2]))
            source_pos = np.array([xy[0], xy[1], z], dtype=float)
            if np.any(source_pos < low) or np.any(source_pos > high):
                continue
            if float(np.min(np.linalg.norm(sec_positions - source_pos[None, :], axis=1))) < float(self.cfg.min_source_device_distance + 0.10):
                continue
            if not self._source_angle_separation_ok(source_pos, sec_positions):
                continue
            return source_pos
        raise QCError("geometry:failed_to_place_source_inside_secondary_triangle")

    def _sample_secondary_height(self, room_size: np.ndarray) -> float:
        z_low = max(float(self.cfg.wall_margin + 0.05), float(self.cfg.secondary_height_range[0]))
        z_high = min(float(room_size[2] - self.cfg.wall_margin - 0.05), float(self.cfg.secondary_height_range[1]))
        if z_high <= z_low:
            raise QCError("geometry:room_height_too_small_for_devices")
        return float(self.rng.uniform(z_low, z_high))

    def _sample_unit_xy(self) -> np.ndarray:
        theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
        return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)

    def _layout_metadata(self, source_pos: np.ndarray, ref_positions: np.ndarray, sec_positions: np.ndarray, err_positions: np.ndarray) -> dict[str, np.ndarray]:
        source_pos = np.asarray(source_pos, dtype=float)
        sec_vec = np.asarray(sec_positions, dtype=float) - source_pos[None, :]
        azimuth_deg = (np.degrees(np.arctan2(sec_vec[:, 1], sec_vec[:, 0])) + 360.0) % 360.0
        return {
            "azimuth_deg": azimuth_deg.astype(float),
            "ref_radii": np.linalg.norm(np.asarray(ref_positions, dtype=float) - source_pos[None, :], axis=1).astype(float),
            "sec_radii": np.linalg.norm(np.asarray(sec_positions, dtype=float) - source_pos[None, :], axis=1).astype(float),
            "err_radii": np.linalg.norm(np.asarray(err_positions, dtype=float) - source_pos[None, :], axis=1).astype(float),
            "z_offsets": (np.asarray(sec_positions, dtype=float)[:, 2] - source_pos[2]).astype(float),
        }

    def _sample_local_direction(self, sec_pos: np.ndarray, focus_point: np.ndarray) -> np.ndarray:
        direction = np.asarray(sec_pos, dtype=float) - np.asarray(focus_point, dtype=float)
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            direction = self._sample_unit_xy()
        else:
            direction = direction / norm
        direction[2] = float(self.rng.uniform(*self.cfg.node_direction_tilt_range))
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        return direction

    def _max_forward_distance_inside_room(self, point: np.ndarray, direction: np.ndarray, room_size: np.ndarray) -> float:
        low = np.full(3, float(self.cfg.wall_margin), dtype=float)
        high = np.asarray(room_size, dtype=float) - low
        t_max = float("inf")
        for dim in range(3):
            d = float(direction[dim])
            if abs(d) <= 1.0e-9:
                continue
            if d > 0.0:
                limit = (float(high[dim]) - float(point[dim])) / d
            else:
                limit = (float(low[dim]) - float(point[dim])) / d
            if limit < 0.0:
                return -1.0
            t_max = min(t_max, float(limit))
        return float(t_max)

    def _derive_triplets_from_source(self, source_pos: np.ndarray, sec_positions: np.ndarray, room_size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source_pos = np.asarray(source_pos, dtype=float)
        sec_positions = np.asarray(sec_positions, dtype=float)
        for _ in range(40):
            ref_positions = np.zeros_like(sec_positions, dtype=float)
            err_positions = np.zeros_like(sec_positions, dtype=float)
            valid = True
            for i, sec_pos in enumerate(sec_positions):
                axis = sec_pos - source_pos
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm <= float(self.cfg.min_source_device_distance + self.cfg.node_ref_to_sec_range[0] + 0.02):
                    valid = False
                    break
                direction = axis / axis_norm
                ref_gap_lo = float(self.cfg.node_ref_to_sec_range[0])
                ref_gap_hi = min(float(self.cfg.node_ref_to_sec_range[1]), axis_norm - float(self.cfg.min_source_device_distance) - 0.02)
                if ref_gap_hi <= ref_gap_lo:
                    valid = False
                    break
                max_err_gap = self._max_forward_distance_inside_room(sec_pos, direction, room_size) - 0.02
                err_gap_lo = float(self.cfg.node_err_extra_range[0])
                err_gap_hi = min(float(self.cfg.node_err_extra_range[1]), float(max_err_gap))
                if err_gap_hi <= err_gap_lo:
                    valid = False
                    break
                ref_gap = float(self.rng.uniform(ref_gap_lo, ref_gap_hi))
                err_gap = float(self.rng.uniform(err_gap_lo, err_gap_hi))
                ref_dist = axis_norm - ref_gap
                if ref_dist <= float(self.cfg.min_source_device_distance):
                    valid = False
                    break
                ref_positions[i] = source_pos + direction * ref_dist
                err_positions[i] = sec_pos + direction * err_gap
            if valid and self._all_inside_room(ref_positions, room_size) and self._all_inside_room(err_positions, room_size):
                return ref_positions, err_positions
        raise QCError("geometry:failed_to_place_triplets_from_source")

    def _neighbor_causality_status(self, ref_positions: np.ndarray, sec_positions: np.ndarray, err_positions: np.ndarray) -> tuple[bool, float]:
        violations = 0
        min_margin = float("inf")
        for i in range(self.cfg.num_secondary_speakers):
            for j in range(self.cfg.num_secondary_speakers):
                if i == j:
                    continue
                d_ref = float(np.linalg.norm(sec_positions[j] - ref_positions[i]))
                d_err = float(np.linalg.norm(sec_positions[j] - err_positions[i]))
                min_margin = min(min_margin, d_err - d_ref)
                if d_ref > d_err + float(self.cfg.neighbor_causality_tolerance):
                    violations += 1
                    if violations > int(self.cfg.max_causality_violations):
                        return False, float(min_margin)
        if not np.isfinite(min_margin):
            min_margin = 0.0
        return True, float(min_margin)

    def _sample_source_radial_layout(self, room_size: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(80):
            base_start = float(self.rng.uniform(0.0, 360.0))
            spacing = np.array(self.cfg.azimuth_spacing_base, dtype=float)
            spacing += self.rng.uniform(-self.cfg.azimuth_spacing_jitter, self.cfg.azimuth_spacing_jitter, size=self.cfg.num_secondary_speakers)
            azimuths = (base_start + spacing) % 360.0
            per_node = []
            max_err_radius = 0.0
            for az in azimuths:
                ref_r = float(self.rng.uniform(*self.cfg.ref_radius_range))
                sec_r = ref_r + float(self.rng.uniform(*self.cfg.sec_delta_range))
                err_r = sec_r + float(self.rng.uniform(*self.cfg.err_delta_range))
                z_off = float(self.rng.uniform(*self.cfg.z_offset_range))
                per_node.append((float(az), ref_r, sec_r, err_r, z_off))
                max_err_radius = max(max_err_radius, err_r)
            side_margin_xy = min(float(np.min(room_size[:2]) / 2.0 - 0.05), float(self.cfg.wall_margin + max_err_radius + 0.05))
            if side_margin_xy <= float(self.cfg.wall_margin):
                continue
            source_pos = self._sample_source_position(room_size, xy_margin=side_margin_xy)
            ref_positions = []
            sec_positions = []
            err_positions = []
            for az, ref_r, sec_r, err_r, z_off in per_node:
                theta = np.deg2rad(az)
                direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
                z_vec = np.array([0.0, 0.0, z_off], dtype=float)
                ref_positions.append(source_pos + direction * ref_r + z_vec)
                sec_positions.append(source_pos + direction * sec_r + z_vec)
                err_positions.append(source_pos + direction * err_r + z_vec)
            ref_positions = np.asarray(ref_positions, dtype=float)
            sec_positions = np.asarray(sec_positions, dtype=float)
            err_positions = np.asarray(err_positions, dtype=float)
            if self._all_inside_room(ref_positions, room_size) and self._all_inside_room(sec_positions, room_size) and self._all_inside_room(err_positions, room_size):
                return source_pos, ref_positions, sec_positions, err_positions
        raise QCError("geometry:failed_source_radial_layout")

    def _sample_wall_strip_layout(self, room_size: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outward_gap = float(self.cfg.node_ref_to_sec_range[1] + self.cfg.node_err_extra_range[1])
        base_margin = float(self.cfg.wall_margin + outward_gap + 0.05)
        for _ in range(80):
            axis = int(self.rng.integers(0, 2))
            low_side = bool(self.rng.integers(0, 2) == 0)
            sec_positions = np.zeros((self.cfg.num_secondary_speakers, 3), dtype=float)
            if axis == 0:
                x_base = base_margin + float(self.rng.uniform(0.05, 0.35))
                x_base = x_base if low_side else float(room_size[0] - x_base)
                y_low = float(self.cfg.wall_margin + 0.20)
                y_high = float(room_size[1] - self.cfg.wall_margin - 0.20)
                if y_high <= y_low:
                    continue
                y_center = float(self.rng.uniform(y_low, y_high))
                half_span = float(self.rng.uniform(0.20, min(1.40, max((y_high - y_low) / 2.0, 0.20))))
                span_low = max(y_low, y_center - half_span)
                span_high = min(y_high, y_center + half_span)
                sec_positions[:, 0] = x_base + self.rng.uniform(-0.18, 0.18, size=self.cfg.num_secondary_speakers)
                sec_positions[:, 1] = np.sort(self.rng.uniform(span_low, span_high, size=self.cfg.num_secondary_speakers))
            else:
                y_base = base_margin + float(self.rng.uniform(0.05, 0.35))
                y_base = y_base if low_side else float(room_size[1] - y_base)
                x_low = float(self.cfg.wall_margin + 0.20)
                x_high = float(room_size[0] - self.cfg.wall_margin - 0.20)
                if x_high <= x_low:
                    continue
                x_center = float(self.rng.uniform(x_low, x_high))
                half_span = float(self.rng.uniform(0.20, min(1.40, max((x_high - x_low) / 2.0, 0.20))))
                span_low = max(x_low, x_center - half_span)
                span_high = min(x_high, x_center + half_span)
                sec_positions[:, 0] = np.sort(self.rng.uniform(span_low, span_high, size=self.cfg.num_secondary_speakers))
                sec_positions[:, 1] = y_base + self.rng.uniform(-0.18, 0.18, size=self.cfg.num_secondary_speakers)
            for i in range(self.cfg.num_secondary_speakers):
                sec_positions[i, 2] = self._sample_secondary_height(room_size)
            if self._triangle_area_xy(sec_positions) < float(self.cfg.min_secondary_triangle_area):
                continue
            source_pos = self._sample_source_inside_secondary_triangle(sec_positions, room_size)
            ref_positions, err_positions = self._derive_triplets_from_source(source_pos, sec_positions, room_size)
            return source_pos, ref_positions, sec_positions, err_positions
        raise QCError("geometry:failed_wall_strip_layout")

    def _sample_corner_cluster_layout(self, room_size: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outward_gap = float(self.cfg.node_ref_to_sec_range[1] + self.cfg.node_err_extra_range[1])
        inner_margin = float(self.cfg.wall_margin + outward_gap + 0.05)
        for _ in range(80):
            x_low = bool(self.rng.integers(0, 2) == 0)
            y_low = bool(self.rng.integers(0, 2) == 0)
            sign_x = 1.0 if x_low else -1.0
            sign_y = 1.0 if y_low else -1.0
            max_span_x = float(room_size[0] - 2.0 * inner_margin - 0.10)
            max_span_y = float(room_size[1] - 2.0 * inner_margin - 0.10)
            if max_span_x <= 0.15 or max_span_y <= 0.15:
                continue
            span_x = float(self.rng.uniform(0.15, min(1.25, max_span_x)))
            span_y = float(self.rng.uniform(0.15, min(1.25, max_span_y)))
            base_x = inner_margin + float(self.rng.uniform(0.05, 0.35))
            base_y = inner_margin + float(self.rng.uniform(0.05, 0.35))
            base_x = base_x if x_low else float(room_size[0] - base_x)
            base_y = base_y if y_low else float(room_size[1] - base_y)
            sec_positions = np.zeros((self.cfg.num_secondary_speakers, 3), dtype=float)
            for i in range(self.cfg.num_secondary_speakers):
                sec_positions[i, 0] = base_x + sign_x * float(self.rng.uniform(0.0, span_x))
                sec_positions[i, 1] = base_y + sign_y * float(self.rng.uniform(0.0, span_y))
                sec_positions[i, 2] = self._sample_secondary_height(room_size)
            if self._triangle_area_xy(sec_positions) < float(self.cfg.min_secondary_triangle_area):
                continue
            source_pos = self._sample_source_inside_secondary_triangle(sec_positions, room_size)
            ref_positions, err_positions = self._derive_triplets_from_source(source_pos, sec_positions, room_size)
            return source_pos, ref_positions, sec_positions, err_positions
        raise QCError("geometry:failed_corner_cluster_layout")

    def _sample_free_scatter_layout(self, room_size: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        outward_gap = float(self.cfg.node_ref_to_sec_range[1] + self.cfg.node_err_extra_range[1])
        xy_margin = float(self.cfg.wall_margin + outward_gap + 0.08)
        for _ in range(120):
            if room_size[0] <= 2.0 * xy_margin or room_size[1] <= 2.0 * xy_margin:
                continue
            sec_positions = np.zeros((self.cfg.num_secondary_speakers, 3), dtype=float)
            for i in range(self.cfg.num_secondary_speakers):
                sec_positions[i] = np.array(
                    [
                        self.rng.uniform(xy_margin, room_size[0] - xy_margin),
                        self.rng.uniform(xy_margin, room_size[1] - xy_margin),
                        self._sample_secondary_height(room_size),
                    ],
                    dtype=float,
                )
            if _min_pairwise_distance(sec_positions) < float(self.cfg.min_secondary_node_distance * 1.05):
                continue
            if self._triangle_area_xy(sec_positions) < float(self.cfg.min_secondary_triangle_area):
                continue
            source_pos = self._sample_source_inside_secondary_triangle(sec_positions, room_size)
            ref_positions, err_positions = self._derive_triplets_from_source(source_pos, sec_positions, room_size)
            return source_pos, ref_positions, sec_positions, err_positions
        raise QCError("geometry:failed_free_scatter_layout")

    def sample(self) -> dict[str, Any]:
        for _ in range(1200):
            room_size = self._sample_room_size()
            layout_mode = self._sample_layout_mode()
            try:
                if layout_mode == "source_radial":
                    source_pos, ref_positions, sec_positions, err_positions = self._sample_source_radial_layout(room_size)
                elif layout_mode == "wall_strip":
                    source_pos, ref_positions, sec_positions, err_positions = self._sample_wall_strip_layout(room_size)
                elif layout_mode == "corner_cluster":
                    source_pos, ref_positions, sec_positions, err_positions = self._sample_corner_cluster_layout(room_size)
                else:
                    source_pos, ref_positions, sec_positions, err_positions = self._sample_free_scatter_layout(room_size)
            except QCError:
                continue
            if not self._all_inside_room(ref_positions, room_size) or not self._all_inside_room(sec_positions, room_size) or not self._all_inside_room(err_positions, room_size):
                continue
            if _min_pairwise_distance(sec_positions) < float(self.cfg.min_secondary_node_distance):
                continue
            if self._triangle_area_xy(sec_positions) < float(self.cfg.min_secondary_triangle_area):
                continue
            if not self._source_angle_separation_ok(source_pos, sec_positions):
                continue
            all_devices = np.vstack([ref_positions, sec_positions, err_positions])
            if _min_pairwise_distance(all_devices) < float(self.cfg.min_device_distance):
                continue
            if float(np.min(np.linalg.norm(all_devices - np.asarray(source_pos, dtype=float)[None, :], axis=1))) < float(self.cfg.min_source_device_distance):
                continue
            causality_ok, causality_margin_min = self._neighbor_causality_status(ref_positions, sec_positions, err_positions)
            if not causality_ok:
                continue
            sound_speed = float(self.rng.uniform(self.cfg.sound_speed_min, self.cfg.sound_speed_max))
            absorption, image_order = self._sample_room_acoustics()
            layout_meta = self._layout_metadata(source_pos, ref_positions, sec_positions, err_positions)
            return {
                "room_size": room_size,
                "source_pos": source_pos,
                "ref_positions": ref_positions,
                "sec_positions": sec_positions,
                "err_positions": err_positions,
                **layout_meta,
                "sound_speed": float(sound_speed),
                "absorption": float(absorption),
                "image_order": int(image_order),
                "layout_mode": layout_mode,
                "causality_margin_min": float(causality_margin_min),
            }
        raise QCError("geometry:failed_to_sample_valid_layout")

    def build_manager(self, sampled: dict[str, Any]) -> RIRManager:
        mgr = RIRManager()
        mgr.room = np.asarray(sampled["room_size"], dtype=float)
        mgr.fs = int(self.cfg.fs)
        mgr.sound_speed = float(sampled["sound_speed"])
        mgr.image_source_order = int(sampled["image_order"])
        mgr.material_absorption = float(sampled["absorption"])
        mgr.compensate_fractional_delay = True
        mgr.fractional_delay_shift = None
        mgr.add_primary_speaker(int(self.source_id), sampled["source_pos"])
        for i in range(self.cfg.num_reference_mics):
            mgr.add_reference_microphone(int(self.ref_ids[i]), sampled["ref_positions"][i])
        for i in range(self.cfg.num_secondary_speakers):
            mgr.add_secondary_speaker(int(self.sec_ids[i]), sampled["sec_positions"][i])
        for i in range(self.cfg.num_error_mics):
            mgr.add_error_microphone(int(self.err_ids[i]), sampled["err_positions"][i])
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

    def _direct_energy_ratio(self, rir: np.ndarray, tx_pos: np.ndarray, rx_pos: np.ndarray, fs: int, sound_speed: float) -> float:
        distance = float(np.linalg.norm(np.asarray(tx_pos, dtype=float) - np.asarray(rx_pos, dtype=float)))
        expected_idx = int(round(distance / max(float(sound_speed), 1.0e-6) * float(fs)))
        if expected_idx >= int(np.asarray(rir).size):
            return 0.0
        half_w = int(self.cfg.direct_energy_half_window)
        left = max(0, expected_idx - half_w)
        right = min(int(np.asarray(rir).size), expected_idx + half_w + 1)
        direct_e = float(np.sum(np.asarray(rir, dtype=float)[left:right] ** 2))
        total_e = float(np.sum(np.asarray(rir, dtype=float) ** 2)) + np.finfo(float).eps
        return direct_e / total_e

    def _secondary_condition_summary(self, s_matrix_paths: np.ndarray) -> dict[str, float]:
        s = np.asarray(s_matrix_paths, dtype=np.float64)
        n_fft = int(self.cfg.secondary_condition_fft)
        spec = np.fft.rfft(s, n=n_fft, axis=-1)
        freq = np.fft.rfftfreq(n_fft, d=1.0 / float(self.cfg.fs))
        band = (freq >= float(self.cfg.f_low)) & (freq <= float(self.cfg.f_high))
        conds: list[float] = []
        for fi in np.where(band)[0]:
            h = spec[:, :, fi].T
            try:
                sigma = np.linalg.svd(h, compute_uv=False)
            except np.linalg.LinAlgError:
                conds.append(float("inf"))
                continue
            if sigma.size == 0:
                conds.append(float("inf"))
                continue
            sigma_max = float(np.max(np.abs(sigma)))
            sigma_min = float(np.min(np.abs(sigma)))
            conds.append(float("inf") if sigma_min <= 1.0e-12 else sigma_max / sigma_min)
        if not conds:
            raise QCError("rir:empty_secondary_condition_band")
        cond_arr = np.asarray(conds, dtype=np.float64)
        return {
            "secondary_condition_median": float(np.median(cond_arr)),
            "secondary_condition_p95": float(np.quantile(cond_arr, 0.95)),
            "secondary_condition_max": float(np.max(cond_arr)),
        }

    def validate_rirs(
        self,
        mgr: RIRManager,
        sampled: dict[str, Any],
        source_id: int,
        ref_ids: np.ndarray,
        sec_ids: np.ndarray,
        err_ids: np.ndarray,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ratios: list[float] = []
        s_matrix_paths = np.zeros((self.cfg.num_secondary_speakers, self.cfg.num_error_mics, self.cfg.rir_store_len), dtype=np.float32)
        s_matrix_lengths = np.zeros((self.cfg.num_secondary_speakers, self.cfg.num_error_mics), dtype=np.int32)
        for sec_idx, sec_id in enumerate(sec_ids):
            for err_idx, err_id in enumerate(err_ids):
                s_path = np.asarray(mgr.get_secondary_rir(int(sec_id), int(err_id)), dtype=float)
                if not self._path_is_legal(s_path):
                    raise QCError("rir:illegal_secondary_path")
                s_matrix_paths[sec_idx, err_idx], s_matrix_lengths[sec_idx, err_idx] = _crop_rir(s_path, self.cfg.rir_store_len)
                ratios.append(
                    self._direct_energy_ratio(
                        s_path,
                        np.asarray(sampled["sec_positions"][sec_idx], dtype=float),
                        np.asarray(sampled["err_positions"][err_idx], dtype=float),
                        int(mgr.fs),
                        float(mgr.sound_speed),
                    )
                )

        p_ref_paths = np.zeros((self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype=np.float32)
        p_ref_lengths = np.zeros((self.cfg.num_reference_mics,), dtype=np.int32)
        for ref_idx, ref_id in enumerate(ref_ids):
            p_ref = np.asarray(mgr.get_reference_rir(int(source_id), int(ref_id)), dtype=float)
            if not self._path_is_legal(p_ref):
                raise QCError("rir:illegal_primary_to_reference_path")
            p_ref_paths[ref_idx], p_ref_lengths[ref_idx] = _crop_rir(p_ref, self.cfg.rir_store_len)
            ratios.append(
                self._direct_energy_ratio(
                    p_ref,
                    np.asarray(sampled["source_pos"], dtype=float),
                    np.asarray(sampled["ref_positions"][ref_idx], dtype=float),
                    int(mgr.fs),
                    float(mgr.sound_speed),
                )
            )

        d_paths = np.zeros((self.cfg.num_error_mics, self.cfg.rir_store_len), dtype=np.float32)
        d_lengths = np.zeros((self.cfg.num_error_mics,), dtype=np.int32)
        for err_idx, err_id in enumerate(err_ids):
            d_path = np.asarray(mgr.get_primary_rir(int(source_id), int(err_id)), dtype=float)
            if not self._path_is_legal(d_path):
                raise QCError("rir:illegal_primary_to_error_path")
            d_paths[err_idx], d_lengths[err_idx] = _crop_rir(d_path, self.cfg.rir_store_len)
            ratios.append(
                self._direct_energy_ratio(
                    d_path,
                    np.asarray(sampled["source_pos"], dtype=float),
                    np.asarray(sampled["err_positions"][err_idx], dtype=float),
                    int(mgr.fs),
                    float(mgr.sound_speed),
                )
            )

        s2r_paths = np.zeros((self.cfg.num_secondary_speakers, self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype=np.float32)
        s2r_lengths = np.zeros((self.cfg.num_secondary_speakers, self.cfg.num_reference_mics), dtype=np.int32)
        for sec_idx in range(self.cfg.num_secondary_speakers):
            s2r = np.asarray(mgr.compute_transfer_rirs(sampled["sec_positions"][sec_idx], sampled["ref_positions"]), dtype=float)
            for ref_idx in range(self.cfg.num_reference_mics):
                s2r_paths[sec_idx, ref_idx], s2r_lengths[sec_idx, ref_idx] = _crop_rir(s2r[ref_idx], self.cfg.rir_store_len)

        r2r_paths = np.zeros((len(self.cfg.canonical_r2r_pair_order), self.cfg.rir_store_len), dtype=np.float32)
        r2r_lengths = np.zeros((len(self.cfg.canonical_r2r_pair_order),), dtype=np.int32)
        for pair_idx, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            r2r = np.asarray(mgr.compute_transfer_rirs(sampled["ref_positions"][i], sampled["ref_positions"][j][None, :]), dtype=float).reshape(-1)
            r2r_paths[pair_idx], r2r_lengths[pair_idx] = _crop_rir(r2r, self.cfg.rir_store_len)

        ratio_min = float(np.min(ratios))
        if ratio_min < float(self.cfg.min_direct_ratio):
            raise QCError("rir:direct_path_not_dominant_enough")

        control_ratios = []
        for err_idx in range(self.cfg.num_error_mics):
            p_norm = float(np.linalg.norm(d_paths[err_idx])) + np.finfo(float).eps
            s_norm_best = float(np.max([np.linalg.norm(s_matrix_paths[sec_idx, err_idx]) for sec_idx in range(self.cfg.num_secondary_speakers)]))
            control_ratios.append(s_norm_best / p_norm)
        control_ratio_min = float(np.min(control_ratios))
        if control_ratio_min < float(self.cfg.min_control_ratio):
            raise QCError("rir:insufficient_secondary_control_energy")

        condition = self._secondary_condition_summary(s_matrix_paths)
        if condition["secondary_condition_p95"] > float(self.cfg.secondary_condition_p95_max):
            raise QCError("rir:secondary_condition_p95_too_high")
        if condition["secondary_condition_median"] > float(self.cfg.secondary_condition_median_max):
            raise QCError("rir:secondary_condition_median_too_high")

        metrics = {
            "direct_ratio_min": ratio_min,
            "control_ratio_min": control_ratio_min,
            "secondary_condition_median": float(condition["secondary_condition_median"]),
            "secondary_condition_p95": float(condition["secondary_condition_p95"]),
            "secondary_condition_max": float(condition["secondary_condition_max"]),
        }
        return metrics, p_ref_paths, p_ref_lengths, d_paths, d_lengths, s_matrix_paths, s_matrix_lengths, s2r_paths, s2r_lengths, r2r_paths, r2r_lengths

    @staticmethod
    def _nr_db(d_seg: np.ndarray, e_seg: np.ndarray) -> float:
        d_pow = float(np.mean(np.asarray(d_seg, dtype=float) ** 2)) + np.finfo(float).eps
        e_pow = float(np.mean(np.asarray(e_seg, dtype=float) ** 2)) + np.finfo(float).eps
        return float(10.0 * np.log10(d_pow / e_pow))

    def _nr_metrics(self, d: np.ndarray, e: np.ndarray, fs: int) -> dict[str, float]:
        n = int(np.asarray(d).shape[0])
        if n < 16:
            raise QCError("anc:signal_too_short")
        win = min(max(int(round(0.5 * fs)), 8), n // 2)
        nr_first = self._nr_db(np.asarray(d)[:win], np.asarray(e)[:win])
        nr_last = self._nr_db(np.asarray(d)[-win:], np.asarray(e)[-win:])
        return {
            "nr_first_db": float(nr_first),
            "nr_last_db": float(nr_last),
            "nr_gain_db": float(nr_last - nr_first),
        }

    def evaluate_anc(self, mgr: RIRManager, source_signal: np.ndarray, time_axis: np.ndarray, sec_ids: np.ndarray) -> dict[str, Any]:
        d = mgr.calculate_desired_signal(source_signal, len(time_axis))
        x = _normalize_columns(mgr.calculate_reference_signal(source_signal, len(time_axis)))
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
            e = np.asarray(results["err_hist"], dtype=float)
            if e.ndim != 2 or e.shape[1] != int(self.cfg.num_error_mics) or not np.all(np.isfinite(e)):
                continue
            filters_are_finite = True
            filter_coeffs = results.get("filter_coeffs", {})
            for sec_id in sec_ids:
                w_mat = np.asarray(filter_coeffs.get(int(sec_id), []), dtype=float)
                if w_mat.ndim != 2 or not np.all(np.isfinite(w_mat)):
                    filters_are_finite = False
                    break
            if not filters_are_finite:
                continue
            metrics = self._nr_metrics(d, e, int(mgr.fs))
            if metrics["nr_last_db"] < float(self.cfg.min_nr_last_db) or metrics["nr_gain_db"] < float(self.cfg.min_nr_gain_db):
                continue
            if best is None or metrics["nr_last_db"] > best["metrics"]["nr_last_db"]:
                best = {"mu": float(mu), "results": results, "metrics": metrics, "x": x, "d": d}
        if best is None:
            raise QCError("anc:noise_reduction_qc_failed")

        w_opt = np.zeros((self.cfg.num_secondary_speakers, self.cfg.filter_len), dtype=np.float32)
        w_full = np.zeros((self.cfg.num_secondary_speakers, self.cfg.num_reference_mics, self.cfg.filter_len), dtype=np.float32)
        filter_coeffs = best["results"]["filter_coeffs"]
        for sec_idx, sec_id in enumerate(sec_ids):
            w_mat = np.asarray(filter_coeffs[int(sec_id)], dtype=float)
            if w_mat.ndim != 2 or w_mat.shape[0] < int(self.cfg.filter_len):
                raise QCError("anc:unexpected_filter_shape")
            keep_l = min(int(self.cfg.filter_len), int(w_mat.shape[0]))
            keep_r = min(int(self.cfg.num_reference_mics), int(w_mat.shape[1]))
            w_full[sec_idx, :keep_r, :keep_l] = w_mat[:keep_l, :keep_r].T.astype(np.float32)
            diag_ref = min(sec_idx, keep_r - 1)
            w_opt[sec_idx] = w_full[sec_idx, diag_ref, :].astype(np.float32)
        best["w_opt"] = w_opt
        best["w_full"] = w_full
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
        self.h5.attrs["schema"] = "CFxLMS_ANC_QC_Dataset_v5_multicontrol_canonical_q"
        self.h5.attrs["config_json"] = json.dumps(asdict(self.cfg), ensure_ascii=False)
        raw = self.h5.create_group("raw")
        room = raw.create_group("room_params")
        room.create_dataset("room_size", shape=(n, 3), dtype="f4")
        room.create_dataset("source_position", shape=(n, 3), dtype="f4")
        room.create_dataset("ref_positions", shape=(n, self.cfg.num_reference_mics, 3), dtype="f4")
        room.create_dataset("sec_positions", shape=(n, self.cfg.num_secondary_speakers, 3), dtype="f4")
        room.create_dataset("err_positions", shape=(n, self.cfg.num_error_mics, 3), dtype="f4")
        room.create_dataset("azimuth_deg", shape=(n, self.cfg.num_secondary_speakers), dtype="f4")
        room.create_dataset("ref_radii", shape=(n, self.cfg.num_reference_mics), dtype="f4")
        room.create_dataset("sec_radii", shape=(n, self.cfg.num_secondary_speakers), dtype="f4")
        room.create_dataset("err_radii", shape=(n, self.cfg.num_error_mics), dtype="f4")
        room.create_dataset("z_offsets", shape=(n, self.cfg.num_secondary_speakers), dtype="f4")
        room.create_dataset("sound_speed", shape=(n,), dtype="f4")
        room.create_dataset("material_absorption", shape=(n,), dtype="f4")
        room.create_dataset("image_source_order", shape=(n,), dtype="i4")
        room.create_dataset("causality_margin_min", shape=(n,), dtype="f4")
        room.create_dataset("layout_mode", shape=(n,), dtype=h5py.string_dtype(encoding="utf-8"))

        raw.create_dataset("x_ref", shape=(n, self.cfg.num_reference_mics, self.cfg.ref_window_samples), dtype="f4")
        raw.create_dataset("P_ref_paths", shape=(n, self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("P_ref_path_lengths", shape=(n, self.cfg.num_reference_mics), dtype="i4")
        raw.create_dataset("D_paths", shape=(n, self.cfg.num_error_mics, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("D_path_lengths", shape=(n, self.cfg.num_error_mics), dtype="i4")
        raw.create_dataset("S_matrix_paths", shape=(n, self.cfg.num_secondary_speakers, self.cfg.num_error_mics, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("S_matrix_path_lengths", shape=(n, self.cfg.num_secondary_speakers, self.cfg.num_error_mics), dtype="i4")
        raw.create_dataset("S2R_paths", shape=(n, self.cfg.num_secondary_speakers, self.cfg.num_reference_mics, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("S2R_path_lengths", shape=(n, self.cfg.num_secondary_speakers, self.cfg.num_reference_mics), dtype="i4")
        raw.create_dataset("R2R_paths", shape=(n, len(self.cfg.canonical_r2r_pair_order), self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("R2R_path_lengths", shape=(n, len(self.cfg.canonical_r2r_pair_order)), dtype="i4")
        raw.create_dataset("W_opt", shape=(n, self.cfg.num_secondary_speakers, self.cfg.filter_len), dtype="f4")
        raw.create_dataset("W_full", shape=(n, self.cfg.num_secondary_speakers, self.cfg.num_reference_mics, self.cfg.filter_len), dtype="f4")
        raw.attrs["r2r_pair_order_json"] = json.dumps(list(self.cfg.canonical_r2r_pair_order), ensure_ascii=False)

        qc = raw.create_group("qc_metrics")
        for name in (
            "nr_first_db",
            "nr_last_db",
            "nr_gain_db",
            "direct_ratio_min",
            "control_ratio_min",
            "secondary_condition_median",
            "secondary_condition_p95",
            "secondary_condition_max",
            "mu_used",
            "warmup_start_s",
        ):
            qc.create_dataset(name, shape=(n,), dtype="f4")
        qc.create_dataset("source_seed", shape=(n,), dtype="i8")
        qc.create_dataset("warmup_start_index", shape=(n,), dtype="i4")

    def write(self, idx: int, sample: RoomSample) -> None:
        self.h5["raw/room_params/room_size"][idx] = sample.room_params["room_size"].astype(np.float32)
        self.h5["raw/room_params/source_position"][idx] = sample.room_params["source_pos"].astype(np.float32)
        self.h5["raw/room_params/ref_positions"][idx] = sample.room_params["ref_positions"].astype(np.float32)
        self.h5["raw/room_params/sec_positions"][idx] = sample.room_params["sec_positions"].astype(np.float32)
        self.h5["raw/room_params/err_positions"][idx] = sample.room_params["err_positions"].astype(np.float32)
        self.h5["raw/room_params/azimuth_deg"][idx] = sample.room_params["azimuth_deg"].astype(np.float32)
        self.h5["raw/room_params/ref_radii"][idx] = sample.room_params["ref_radii"].astype(np.float32)
        self.h5["raw/room_params/sec_radii"][idx] = sample.room_params["sec_radii"].astype(np.float32)
        self.h5["raw/room_params/err_radii"][idx] = sample.room_params["err_radii"].astype(np.float32)
        self.h5["raw/room_params/z_offsets"][idx] = sample.room_params["z_offsets"].astype(np.float32)
        self.h5["raw/room_params/sound_speed"][idx] = float(sample.room_params["sound_speed"])
        self.h5["raw/room_params/material_absorption"][idx] = float(sample.room_params["absorption"])
        self.h5["raw/room_params/image_source_order"][idx] = int(sample.room_params["image_order"])
        self.h5["raw/room_params/causality_margin_min"][idx] = float(sample.room_params["causality_margin_min"])
        self.h5["raw/room_params/layout_mode"][idx] = str(sample.room_params["layout_mode"])

        self.h5["raw/x_ref"][idx] = sample.x_ref.astype(np.float32)
        self.h5["raw/P_ref_paths"][idx] = sample.p_ref_paths.astype(np.float32)
        self.h5["raw/P_ref_path_lengths"][idx] = sample.p_ref_lengths.astype(np.int32)
        self.h5["raw/D_paths"][idx] = sample.d_paths.astype(np.float32)
        self.h5["raw/D_path_lengths"][idx] = sample.d_lengths.astype(np.int32)
        self.h5["raw/S_matrix_paths"][idx] = sample.s_matrix_paths.astype(np.float32)
        self.h5["raw/S_matrix_path_lengths"][idx] = sample.s_matrix_lengths.astype(np.int32)
        self.h5["raw/S2R_paths"][idx] = sample.s2r_paths.astype(np.float32)
        self.h5["raw/S2R_path_lengths"][idx] = sample.s2r_lengths.astype(np.int32)
        self.h5["raw/R2R_paths"][idx] = sample.r2r_paths.astype(np.float32)
        self.h5["raw/R2R_path_lengths"][idx] = sample.r2r_lengths.astype(np.int32)
        self.h5["raw/W_opt"][idx] = sample.w_opt.astype(np.float32)
        self.h5["raw/W_full"][idx] = sample.w_full.astype(np.float32)
        for key, value in sample.qc_metrics.items():
            self.h5[f"raw/qc_metrics/{key}"][idx] = value

    def close(self) -> None:
        if self.h5:
            self.h5.flush()
            self.h5.close()


class ANCDatasetBuilder:
    def __init__(self, cfg: DatasetBuildConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.random_seed))
        self.py_random = random.Random(int(cfg.random_seed))
        self.sampler = AcousticScenarioSampler(cfg, self.rng)
        self.qc = ANCQualityController(cfg)
        self.feature_processor = FeatureProcessor(cfg)
        self.layout_previewer = LayoutPreviewer(cfg)
        self.failure_stats: dict[str, int] = {}

    def _count_fail(self, reason: str) -> None:
        self.failure_stats[reason] = self.failure_stats.get(reason, 0) + 1

    def _sample_warmup_start(self, total_samples: int) -> tuple[int, float]:
        window = int(self.cfg.ref_window_samples)
        if total_samples < window:
            raise QCError("warmup:signal_shorter_than_slice")
        latest_start_s = (total_samples - window) / float(self.cfg.fs)
        start_min_s = float(self.cfg.warmup_start_s_min)
        start_max_s = min(float(self.cfg.warmup_start_s_max), float(latest_start_s))
        if start_max_s < start_min_s:
            raise QCError("warmup:invalid_start_interval")
        start_s = float(self.py_random.uniform(start_min_s, start_max_s))
        start_idx = int(round(start_s * float(self.cfg.fs)))
        start_idx = max(0, min(start_idx, total_samples - window))
        return start_idx, start_s

    def _build_single_sample(self) -> RoomSample:
        sampled = self.sampler.sample()
        mgr = self.sampler.build_manager(sampled)
        mgr.build(verbose=False)
        (
            rir_metrics,
            p_ref_paths,
            p_ref_lengths,
            d_paths,
            d_lengths,
            s_matrix_paths,
            s_matrix_lengths,
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
        anc_result = self.qc.evaluate_anc(mgr=mgr, source_signal=source_signal, time_axis=time_axis, sec_ids=self.sampler.sec_ids)

        x = np.asarray(anc_result["x"], dtype=float)
        if x.ndim != 2 or x.shape[1] < int(self.cfg.num_reference_mics):
            raise QCError("anc:unexpected_reference_shape")
        start_idx, start_s = self._sample_warmup_start(x.shape[0])
        end_idx = start_idx + int(self.cfg.ref_window_samples)
        x_ref = x[start_idx:end_idx, : int(self.cfg.num_reference_mics)].T.astype(np.float32)

        qc_metrics = {
            "nr_first_db": float(anc_result["metrics"]["nr_first_db"]),
            "nr_last_db": float(anc_result["metrics"]["nr_last_db"]),
            "nr_gain_db": float(anc_result["metrics"]["nr_gain_db"]),
            "direct_ratio_min": float(rir_metrics["direct_ratio_min"]),
            "control_ratio_min": float(rir_metrics["control_ratio_min"]),
            "secondary_condition_median": float(rir_metrics["secondary_condition_median"]),
            "secondary_condition_p95": float(rir_metrics["secondary_condition_p95"]),
            "secondary_condition_max": float(rir_metrics["secondary_condition_max"]),
            "mu_used": float(anc_result["mu"]),
            "source_seed": int(source_seed),
            "warmup_start_s": float(start_s),
            "warmup_start_index": int(start_idx),
        }
        return RoomSample(
            room_params=sampled,
            x_ref=x_ref,
            p_ref_paths=p_ref_paths,
            p_ref_lengths=p_ref_lengths,
            d_paths=d_paths,
            d_lengths=d_lengths,
            s_matrix_paths=s_matrix_paths,
            s_matrix_lengths=s_matrix_lengths,
            s2r_paths=s2r_paths,
            s2r_lengths=s2r_lengths,
            r2r_paths=r2r_paths,
            r2r_lengths=r2r_lengths,
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
            while accepted < int(self.cfg.target_rooms) and attempts < int(self.cfg.max_total_attempts):
                attempts += 1
                try:
                    sample = self._build_single_sample()
                    writer.write(accepted, sample)
                    accepted += 1
                    preview_mgr = self.sampler.build_manager(sample.room_params)
                    self.layout_previewer.update(preview_mgr, sample.room_params, accepted, attempts)
                    if accepted % max(int(self.cfg.progress_interval), 1) == 0 or accepted == int(self.cfg.target_rooms):
                        print(
                            f"[Progress] accepted={accepted}/{self.cfg.target_rooms}, attempts={attempts}, "
                            f"pass_rate={accepted / max(attempts, 1):.3f}"
                        )
                        writer.h5.attrs["accepted_so_far"] = int(accepted)
                        writer.h5.attrs["attempts_so_far"] = int(attempts)
                        writer.h5.flush()
                except QCError as exc:
                    self._count_fail(str(exc))
                except Exception as exc:
                    self._count_fail(f"unexpected:{type(exc).__name__}")
                if attempts % max(int(self.cfg.attempt_log_interval), 1) == 0:
                    top = sorted(self.failure_stats.items(), key=lambda kv: kv[1], reverse=True)[:1]
                    top_text = ", ".join(f"{k}:{v}" for k, v in top) if top else "none"
                    print(
                        f"[Attempt] attempts={attempts}, accepted={accepted}/{self.cfg.target_rooms}, "
                        f"pass_rate={accepted / max(attempts, 1):.3f}, top_fail={top_text}"
                    )
            if accepted < int(self.cfg.target_rooms):
                raise RuntimeError(
                    f"Only accepted {accepted} rooms before reaching max_total_attempts={self.cfg.max_total_attempts}. "
                    "Please increase attempts or relax geometry/QC thresholds."
                )
            writer.h5.attrs["accepted_rooms"] = int(accepted)
            writer.h5.attrs["attempts"] = int(attempts)
            writer.h5.attrs["failure_stats_json"] = json.dumps(self.failure_stats, ensure_ascii=False)
            writer.h5.flush()
        finally:
            writer.close()
            self.layout_previewer.close()

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
    d_paths: np.ndarray
    s_matrix_paths: np.ndarray
    t_db: np.ndarray
    db_zero: np.ndarray
    db_h5: np.ndarray
    early_mask: np.ndarray


def _sampled_room_from_h5(h5: h5py.File, idx: int) -> dict[str, Any]:
    room = h5["raw/room_params"]
    layout_value = room["layout_mode"][idx]
    if isinstance(layout_value, bytes):
        layout_value = layout_value.decode("utf-8")
    return {
        "room_size": np.asarray(room["room_size"][idx], dtype=float),
        "source_pos": np.asarray(room["source_position"][idx], dtype=float),
        "ref_positions": np.asarray(room["ref_positions"][idx], dtype=float),
        "sec_positions": np.asarray(room["sec_positions"][idx], dtype=float),
        "err_positions": np.asarray(room["err_positions"][idx], dtype=float),
        "azimuth_deg": np.asarray(room["azimuth_deg"][idx], dtype=float),
        "ref_radii": np.asarray(room["ref_radii"][idx], dtype=float),
        "sec_radii": np.asarray(room["sec_radii"][idx], dtype=float),
        "err_radii": np.asarray(room["err_radii"][idx], dtype=float),
        "z_offsets": np.asarray(room["z_offsets"][idx], dtype=float),
        "sound_speed": float(room["sound_speed"][idx]),
        "absorption": float(room["material_absorption"][idx]),
        "image_order": int(room["image_source_order"][idx]),
        "layout_mode": str(layout_value),
        "causality_margin_min": float(room["causality_margin_min"][idx]),
    }


def _replay_metrics_for_case(case: _CalibrationReplayCase, w_ai: np.ndarray, cfg: DatasetBuildConfig, early_window_s: float) -> dict[str, float]:
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
    )
    window_samples = min(max(32, int(round(float(early_window_s) * float(cfg.fs)))), max(int(len(case.time_axis) // 2), 32))
    t_db = case.t_db
    db_zero = case.db_zero
    db_h5 = case.db_h5
    _, db_ai = _rolling_mse_db_multichannel(e_ai, int(cfg.fs), window_samples=window_samples)
    early_mask = case.early_mask
    return {
        "ai_vs_zero_db": float(np.mean(db_zero[early_mask] - db_ai[early_mask])),
        "h5_vs_zero_db": float(np.mean(db_zero[early_mask] - db_h5[early_mask])),
        "ai_to_h5_gap_db": float(np.mean(np.abs(db_ai[early_mask] - db_h5[early_mask]))),
    }


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
        params = {
            "time": time_axis,
            "rir_manager": mgr,
            "L": int(cfg.filter_len),
            "mu": float(cfg.mu_candidates[0]),
            "reference_signal": reference_signal,
            "desired_signal": desired_signal,
            "verbose": False,
            "normalized_update": bool(cfg.anc_normalized_update),
            "norm_epsilon": float(cfg.anc_norm_epsilon),
        }
        w_h5 = np.asarray(raw["W_full"][idx], dtype=np.float32)
        e_zero = np.asarray(cfxlms(params)["err_hist"], dtype=float)
        e_h5 = np.asarray(
            _cfxlms_with_init(
                time_axis,
                mgr,
                int(cfg.filter_len),
                float(cfg.mu_candidates[0]),
                reference_signal,
                desired_signal,
                w_init=w_h5,
                normalized_update=bool(cfg.anc_normalized_update),
                norm_epsilon=float(cfg.anc_norm_epsilon),
            )["err_hist"],
            dtype=float,
        )
        window_samples = min(max(32, int(round(float(cfg.canonical_replay_early_window_s) * float(cfg.fs)))), max(int(len(time_axis) // 2), 32))
        t_db, db_zero = _rolling_mse_db_multichannel(e_zero, int(cfg.fs), window_samples=window_samples)
        _, db_h5 = _rolling_mse_db_multichannel(e_h5, int(cfg.fs), window_samples=window_samples)
        early_mask = t_db <= float(cfg.canonical_replay_early_window_s)
        if not np.any(early_mask):
            early_mask = np.ones_like(t_db, dtype=bool)
        cases.append(
            _CalibrationReplayCase(
                idx=int(idx),
                manager=mgr,
                time_axis=time_axis,
                reference_signal=reference_signal,
                desired_signal=desired_signal,
                w_h5=w_h5,
                p_ref_paths=np.asarray(raw["P_ref_paths"][idx], dtype=np.float32),
                d_paths=np.asarray(raw["D_paths"][idx], dtype=np.float32),
                s_matrix_paths=np.asarray(raw["S_matrix_paths"][idx], dtype=np.float32),
                t_db=t_db,
                db_zero=db_zero,
                db_h5=db_h5,
                early_mask=early_mask,
            )
        )
    return cases


def _calibrate_canonical_regularization(h5: h5py.File, cfg: DatasetBuildConfig, q_full_len: int) -> dict[str, Any]:
    n_rooms = int(h5["raw/W_full"].shape[0])
    indices = np.arange(n_rooms, dtype=np.int64)
    rng = np.random.default_rng(int(cfg.random_seed))
    rng.shuffle(indices)
    split = max(1, min(n_rooms - 1, int(n_rooms * 0.8)))
    train_idx = np.sort(indices[:split])
    calib_idx = np.asarray(train_idx[: min(int(cfg.canonical_calibration_rooms), len(train_idx))], dtype=np.int64)
    cases = _build_calibration_replay_cases(h5, cfg, calib_idx)
    best: dict[str, Any] | None = None
    for lambda_q_scale in cfg.canonical_lambda_q_scale_candidates:
        for lambda_w in cfg.canonical_lambda_w_candidates:
            gains = []
            gaps = []
            residuals = []
            for case in cases:
                q_full = _canonical_q_from_matrix_paths(case.d_paths, case.s_matrix_paths, int(q_full_len), float(lambda_q_scale))
                w_canon = _solve_w_canonical_from_q(q_full, case.p_ref_paths, int(cfg.filter_len), float(lambda_w), int(q_full_len))
                metrics = _replay_metrics_for_case(case, w_canon, cfg, float(cfg.canonical_replay_early_window_s))
                gains.append(float(metrics["ai_vs_zero_db"]))
                gaps.append(float(metrics["ai_to_h5_gap_db"]))
                residuals.append(float(_plant_residual_ratio(q_full, case.d_paths, case.s_matrix_paths)))
            summary = {
                "q_target_source": "regularized_secondary_inverse",
                "lambda_q_scale": float(lambda_q_scale),
                "lambda_w": float(lambda_w),
                "ai_vs_zero_db_mean": float(np.mean(gains)),
                "ai_to_h5_gap_db_mean": float(np.mean(gaps)),
                "plant_residual_mean": float(np.mean(residuals)),
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
                f"lambda_q_scale={lambda_q_scale:.1e}, lambda_w={lambda_w:.1e}, "
                f"gain={summary['ai_vs_zero_db_mean']:.3f} dB, gap={summary['ai_to_h5_gap_db_mean']:.3f} dB, "
                f"residual={summary['plant_residual_mean']:.4f}"
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
        raw = h5["raw"]
        feature_proc = FeatureProcessor(cfg)
        x_ref = np.asarray(raw["x_ref"], dtype=np.float32)
        p_ref_paths = np.asarray(raw["P_ref_paths"], dtype=np.float32)
        d_paths = np.asarray(raw["D_paths"], dtype=np.float32)
        s_matrix_paths = np.asarray(raw["S_matrix_paths"], dtype=np.float32)
        n_rooms = int(x_ref.shape[0])
        q_full_len = int(cfg.filter_len + cfg.rir_store_len - 1)

        calibration = _calibrate_canonical_regularization(h5, cfg, q_full_len=q_full_len)
        best_lambda_q_scale = float(calibration["lambda_q_scale"])
        best_lambda_w = float(calibration["lambda_w"])

        q_full = np.zeros((n_rooms, cfg.num_secondary_speakers, q_full_len), dtype=np.float32)
        w_canon = np.zeros((n_rooms, cfg.num_secondary_speakers, cfg.num_reference_mics, cfg.filter_len), dtype=np.float32)
        plant_residual = np.zeros((n_rooms,), dtype=np.float32)
        for i in range(n_rooms):
            q_full[i] = _canonical_q_from_matrix_paths(d_paths[i], s_matrix_paths[i], q_full_len=q_full_len, lambda_q_scale=best_lambda_q_scale)
            w_canon[i] = _solve_w_canonical_from_q(q_full[i], p_ref_paths[i], filter_len=int(cfg.filter_len), lambda_w=best_lambda_w, q_full_len=q_full_len)
            plant_residual[i] = float(_plant_residual_ratio(q_full[i], d_paths[i], s_matrix_paths[i]))

        q_keep = _choose_keep_len(
            q_full.reshape(-1, q_full_len),
            cfg.canonical_q_energy_ratio,
            cfg.canonical_q_length_quantile,
            cfg.canonical_q_min_keep_len,
            min(cfg.canonical_q_max_keep_len, q_full_len),
        )

        gcc = np.zeros((n_rooms, 3, cfg.gcc_truncated_len), dtype=np.float32)
        psd = np.zeros((n_rooms, cfg.psd_nfft // 2 + 1), dtype=np.float32)
        for i in range(n_rooms):
            gcc[i] = feature_proc.compute_gcc_phat(x_ref[i])
            psd[i] = feature_proc.compute_psd_features(x_ref[i, 0])

        if "processed" in h5:
            del h5["processed"]
        processed = h5.create_group("processed")
        processed.create_dataset("gcc_phat", data=gcc, dtype="f4")
        processed.create_dataset("psd_features", data=psd, dtype="f4")
        processed.create_dataset("q_target", data=q_full[:, :, :q_keep], dtype="f4")
        processed.create_dataset("w_canon", data=w_canon, dtype="f4")
        processed.create_dataset("plant_residual", data=plant_residual, dtype="f4")
        processed.attrs["q_keep_len"] = int(q_keep)
        processed.attrs["q_full_len"] = int(q_full_len)
        processed.attrs["q_target_dim"] = int(cfg.num_secondary_speakers * q_keep)
        processed.attrs["q_num_speakers"] = int(cfg.num_secondary_speakers)
        processed.attrs["lambda_q_scale"] = float(best_lambda_q_scale)
        processed.attrs["lambda_w"] = float(best_lambda_w)
        processed.attrs["q_target_source"] = str(calibration.get("q_target_source", "regularized_secondary_inverse"))
        processed.attrs["q_local_half_width"] = int(cfg.canonical_q_local_half_width)
        processed.attrs["q_tail_basis_dim"] = int(cfg.canonical_q_tail_basis_dim)
        processed.attrs["r2r_pair_order_json"] = json.dumps(list(cfg.canonical_r2r_pair_order), ensure_ascii=False)
        processed.attrs["canonical_calibration_json"] = json.dumps(calibration, ensure_ascii=False)
        processed.attrs["secondary_condition_p95_max"] = float(cfg.secondary_condition_p95_max)
        processed.attrs["secondary_condition_median_max"] = float(cfg.secondary_condition_median_max)

    keep_summary = {
        "q_keep_len": int(q_keep),
        "q_full_len": int(q_full_len),
        "q_target_dim": int(cfg.num_secondary_speakers * q_keep),
        "lambda_q_scale": float(best_lambda_q_scale),
        "lambda_w": float(best_lambda_w),
        "plant_residual_mean": float(np.mean(plant_residual)),
    }
    print("Processed features updated:", keep_summary)
    return keep_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build multi-control CFxLMS ANC dataset (3 refs, 3 speakers, 3 error mics).")
    parser.add_argument("--num-rooms", type=int, default=1000, help="Number of accepted rooms to generate.")
    parser.add_argument("--max-attempts", type=int, default=30000, help="Maximum total sampling attempts.")
    parser.add_argument("--output-h5", type=str, default=str(ROOT_DIR / "python_scripts" / "cfxlms_qc_dataset_multicontrol.h5"), help="Output HDF5 path.")
    parser.add_argument("--seed", type=int, default=20260401, help="Random seed.")
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
    print("Starting multi-control CFxLMS dataset build...")
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
        "Secondary conditioning: "
        f"median<={cfg.secondary_condition_median_max:.1f}, "
        f"p95<={cfg.secondary_condition_p95_max:.1f}"
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

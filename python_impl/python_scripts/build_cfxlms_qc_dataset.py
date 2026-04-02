from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cfxlms_multi_control_dataset_impl import *

if __name__ == "__main__":
    raise SystemExit(main())

'''

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


class QCError(RuntimeError):
    """质量控制失败异常，用于单条样本失败时跳过并继续下一条。"""


@dataclass
class DatasetBuildConfig:
    """CFxLMS 数据集构建配置。"""

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
    absorption_min: float = 0.30
    absorption_max: float = 0.88
    image_order_choices: tuple[int, ...] = (0, 1, 2)
    image_order_probs: tuple[float, ...] = (0.20, 0.60, 0.20)
    reflection_room_probability: float = 0.55
    direct_room_absorption_range: tuple[float, float] = (0.68, 0.88)
    direct_room_image_order_choices: tuple[int, ...] = (0, 1)
    direct_room_image_order_probs: tuple[float, ...] = (0.25, 0.75)
    reflection_room_absorption_range: tuple[float, float] = (0.30, 0.58)
    reflection_room_image_order_choices: tuple[int, ...] = (1, 2)
    reflection_room_image_order_probs: tuple[float, ...] = (0.80, 0.20)

    num_nodes: int = 3
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
    min_nr_last_db: float = 15
    min_nr_gain_db: float = 15
    anc_normalized_update: bool = False
    anc_norm_epsilon: float = 1.0e-8

    direct_energy_half_window: int = 2
    min_direct_ratio: float = 0.08

    rir_store_len: int = 512
    gcc_truncated_len: int = 129
    psd_nfft: int = 256
    svd_components: int = 32

    random_seed: int = 20260329
    progress_interval: int = 20
    attempt_log_interval: int = 10
    layout_preview: bool = True
    layout_preview_interval: int = 1

    output_h5: str = str(ROOT_DIR / "python_scripts" / "cfxlms_qc_dataset.h5")

    @property
    def ref_window_samples(self) -> int:
        return int(round(self.fs * self.ref_window_ms / 1000.0))

    @property
    def s_feature_dim(self) -> int:
        # 采用完整次级通路矩阵: [sec, err, rir_len]
        return int(self.num_nodes * self.num_nodes * self.rir_store_len)

    @property
    def w_feature_dim(self) -> int:
        # 采用完整控制滤波器矩阵: [sec, ref, filter_len]
        return int(self.num_nodes * self.num_nodes * self.filter_len)


@dataclass
class RoomSample:
    """单条通过 QC 的样本数据容器。"""

    room_params: dict[str, Any]
    x_ref: np.ndarray
    s_paths: np.ndarray
    s_path_lengths: np.ndarray
    w_opt: np.ndarray
    w_full: np.ndarray
    gcc_phat: np.ndarray
    psd_features: np.ndarray
    qc_metrics: dict[str, float]


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(x), axis=0, keepdims=True)
    denom = np.where(denom < np.finfo(float).eps, 1.0, denom)
    return x / denom


class FeatureProcessor:
    """原始信号预处理特征提取器。"""

    def __init__(self, cfg: DatasetBuildConfig):
        self.cfg = cfg
        self.pairs = [(0, 1), (0, 2), (1, 2)]

    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 if n <= 1 else 1 << (n - 1).bit_length()

    @staticmethod
    def _central_crop(x: np.ndarray, target_len: int) -> np.ndarray:
        if target_len <= 0:
            raise ValueError("target_len 必须为正数。")

        if x.size == target_len:
            return x

        if x.size < target_len:
            pad_total = target_len - x.size
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(x, (pad_left, pad_right), mode="constant")

        mid = x.size // 2
        half = target_len // 2
        if target_len % 2 == 1:
            return x[mid - half : mid + half + 1]
        return x[mid - half : mid + half]

    def gcc_phat_pair(self, sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray:
        n = int(sig_a.size + sig_b.size - 1)
        n_fft = self._next_pow2(n)

        sig_a_f = np.fft.rfft(sig_a, n=n_fft)
        sig_b_f = np.fft.rfft(sig_b, n=n_fft)
        cross = sig_a_f * np.conj(sig_b_f)

        cross /= np.maximum(np.abs(cross), np.finfo(float).eps)
        corr = np.fft.irfft(cross, n=n_fft)

        # 将负延迟部分平移到前面，便于截取中心区域。
        corr = np.concatenate([corr[-(n_fft // 2) :], corr[: n_fft // 2]])
        corr = corr.astype(np.float64, copy=False)
        return self._central_crop(corr, self.cfg.gcc_truncated_len)

    def compute_gcc_phat(self, x_ref: np.ndarray) -> np.ndarray:
        if x_ref.shape[0] != 3:
            raise ValueError(f"x_ref 第一维应为 3，实际为 {x_ref.shape[0]}。")

        gcc = np.zeros((len(self.pairs), self.cfg.gcc_truncated_len), dtype=np.float32)
        for k, (i, j) in enumerate(self.pairs):
            gcc[k] = self.gcc_phat_pair(x_ref[i], x_ref[j]).astype(np.float32)
        return gcc

    def compute_psd_features(self, sig: np.ndarray) -> np.ndarray:
        n_fft = int(self.cfg.psd_nfft)
        if sig.size < n_fft:
            padded = np.pad(sig, (0, n_fft - sig.size), mode="constant")
        else:
            padded = sig[:n_fft]

        spec = np.fft.rfft(padded, n=n_fft)
        psd = (np.abs(spec) ** 2) / max(n_fft, 1)
        # 使用对数功率提升动态范围稳定性。
        psd = np.log10(psd + np.finfo(float).eps)
        return psd.astype(np.float32)


class LayoutPreviewer:
    """实时布局预览器：复用同一窗口，不保存图片。"""

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
        except Exception as exc:
            print(f"[Preview] layout preview disabled: {type(exc).__name__}: {exc}")
            self.enabled = False
            return

        self.plt = plt
        try:
            self.plt.ion()
            self.fig, self.ax = self.plt.subplots(figsize=(9, 6))
            manager = getattr(self.fig.canvas, "manager", None)
            if manager is not None and hasattr(manager, "set_window_title"):
                manager.set_window_title("CFxLMS Layout Preview")
            self.plt.show(block=False)
        except Exception as exc:
            print(f"[Preview] layout preview disabled during initialization: {type(exc).__name__}: {exc}")
            self.enabled = False
            self.plt = None
            self.fig = None
            self.ax = None

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
                linewidth=1.3,
            )

    def update(self, mgr: RIRManager, sampled: dict[str, Any], accepted: int, attempts: int) -> None:
        if not self.enabled:
            return
        if accepted <= 0:
            return
        if accepted % max(int(self.cfg.layout_preview_interval), 1) != 0:
            return
        if self.plt is None or self.fig is None or self.ax is None:
            return

        try:
            if hasattr(self.plt, "fignum_exists") and not self.plt.fignum_exists(self.fig.number):
                self.enabled = False
                return

            self.ax.clear()
            mgr.plot_layout_2d(ax=self.ax)
            self._annotate_devices(mgr)
            self._draw_node_links(sampled)

            source_pos = np.asarray(sampled["source_pos"], dtype=float)
            sec_positions = np.asarray(sampled["sec_positions"], dtype=float)
            src_sec_dist = np.linalg.norm(sec_positions - source_pos, axis=1)
            info_lines = [
                f"accepted={accepted}/{self.cfg.target_rooms}",
                f"attempt={attempts}",
                f"layout={sampled.get('layout_mode', '-')}",
                f"src-sec range={float(np.min(src_sec_dist)):.2f}-{float(np.max(src_sec_dist)):.2f} m",
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
            self.ax.set_title("Accepted Room Layout Preview")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)
        except Exception as exc:
            print(f"[Preview] layout preview disabled during update: {type(exc).__name__}: {exc}")
            self.enabled = False

    def finalize(self) -> None:
        if self.plt is None:
            return
        try:
            self.plt.ioff()
            self.plt.pause(0.001)
        except Exception:
            pass


class AcousticScenarioSampler:
    """随机场景采样器：负责房间参数与三节点几何布局生成。"""

    def __init__(self, cfg: DatasetBuildConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

        self.source_id = 101
        self.ref_ids = np.array([401, 402, 403], dtype=int)
        self.sec_ids = np.array([201, 202, 203], dtype=int)
        self.err_ids = np.array([301, 302, 303], dtype=int)

    def _min_pairwise_distance(self, points: np.ndarray) -> float:
        if points.shape[0] < 2:
            return float("inf")

        min_dist = float("inf")
        for i, j in combinations(range(points.shape[0]), 2):
            d = float(np.linalg.norm(points[i] - points[j]))
            min_dist = min(min_dist, d)
        return min_dist

    def _all_inside_room(self, points: np.ndarray, room_size: np.ndarray) -> bool:
        m = float(self.cfg.wall_margin)
        low = np.array([m, m, m], dtype=float)
        high = room_size - m
        return bool(np.all(points >= low) and np.all(points <= high))

    @staticmethod
    def _triangle_area_xy(points: np.ndarray) -> float:
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 3:
            return 0.0
        a = pts[0, :2]
        b = pts[1, :2]
        c = pts[2, :2]
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
            diff = min(diff, 360.0 - diff)
            min_sep = min(min_sep, diff)
        return bool(min_sep >= float(self.cfg.min_source_angle_separation_deg))

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
            weights = self.rng.dirichlet(np.full(self.cfg.num_nodes, float(self.cfg.source_barycentric_alpha), dtype=float))
            xy = weights @ sec_positions[:, :2]
            z_center = float(weights @ sec_positions[:, 2])
            z = float(np.clip(z_center + self.rng.uniform(-0.10, 0.10), low[2], high[2]))
            source_pos = np.array([xy[0], xy[1], z], dtype=float)
            if np.any(source_pos < low) or np.any(source_pos > high):
                continue
            if float(np.min(np.linalg.norm(sec_positions - source_pos, axis=1))) < float(self.cfg.min_source_device_distance + 0.10):
                continue
            if not self._source_angle_separation_ok(source_pos, sec_positions):
                continue
            return source_pos

        raise QCError("geometry:failed_to_place_source_inside_secondary_triangle")

    def _check_neighbor_speaker_causality(
        self,
        ref_positions: np.ndarray,
        sec_positions: np.ndarray,
        err_positions: np.ndarray,
    ) -> bool:
        # 非因果约束: 不允许邻居扬声器到本节点参考麦克风距离大于到本节点误差麦克风距离。
        # 即要求 d(sec_j, ref_i) <= d(sec_j, err_i)。
        violations = 0
        for i in range(self.cfg.num_nodes):
            for j in range(self.cfg.num_nodes):
                if i == j:
                    continue
                d_ref = float(np.linalg.norm(sec_positions[j] - ref_positions[i]))
                d_err = float(np.linalg.norm(sec_positions[j] - err_positions[i]))
                if d_ref > d_err + float(self.cfg.neighbor_causality_tolerance):
                    violations += 1
                    if violations > int(self.cfg.max_causality_violations):
                        return False
        return True

    def _neighbor_causality_status(
        self,
        ref_positions: np.ndarray,
        sec_positions: np.ndarray,
        err_positions: np.ndarray,
    ) -> tuple[bool, float]:
        violations = 0
        min_margin = float("inf")
        for i in range(self.cfg.num_nodes):
            for j in range(self.cfg.num_nodes):
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

    def _sample_room_size(self) -> np.ndarray:
        return self.rng.uniform(self.cfg.room_size_min, self.cfg.room_size_max, size=3).astype(float)

    def _sample_source_position(self, room_size: np.ndarray, xy_margin: float | None = None) -> np.ndarray:
        margin_xy = float(self.cfg.source_wall_margin if xy_margin is None else xy_margin)
        margin_xy = max(margin_xy, float(self.cfg.wall_margin))
        z_margin = max(float(self.cfg.wall_margin + 0.10), 0.55)

        low = np.array([margin_xy, margin_xy, z_margin], dtype=float)
        high = room_size - low
        if np.any(high <= low):
            raise QCError("geometry:room_too_small_for_source_sampling")

        return self.rng.uniform(low, high).astype(float)

    def _sample_source_in_ranges(
        self,
        room_size: np.ndarray,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> np.ndarray:
        z_low = max(float(self.cfg.wall_margin + 0.10), 0.55)
        z_high = float(room_size[2] - z_low)
        if x_range[1] <= x_range[0] or y_range[1] <= y_range[0] or z_high <= z_low:
            return self._sample_source_position(room_size)

        return np.array(
            [
                self.rng.uniform(*x_range),
                self.rng.uniform(*y_range),
                self.rng.uniform(z_low, z_high),
            ],
            dtype=float,
        )

    def _sample_secondary_height(self, room_size: np.ndarray) -> float:
        z_low = max(float(self.cfg.wall_margin + 0.05), float(self.cfg.secondary_height_range[0]))
        z_high = min(float(room_size[2] - self.cfg.wall_margin - 0.05), float(self.cfg.secondary_height_range[1]))
        if z_high <= z_low:
            raise QCError("geometry:room_height_too_small_for_devices")
        return float(self.rng.uniform(z_low, z_high))

    def _sample_unit_xy(self) -> np.ndarray:
        theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
        return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)

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

    def _sample_layout_mode(self) -> str:
        return str(self.rng.choice(self.cfg.layout_mode_choices, p=self.cfg.layout_mode_probs))

    def _layout_metadata(
        self,
        source_pos: np.ndarray,
        ref_positions: np.ndarray,
        sec_positions: np.ndarray,
        err_positions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        source_pos = np.asarray(source_pos, dtype=float)
        sec_vec = np.asarray(sec_positions, dtype=float) - source_pos
        azimuth_deg = (np.degrees(np.arctan2(sec_vec[:, 1], sec_vec[:, 0])) + 360.0) % 360.0

        return {
            "azimuth_deg": azimuth_deg.astype(float),
            "ref_radii": np.linalg.norm(np.asarray(ref_positions, dtype=float) - source_pos, axis=1).astype(float),
            "sec_radii": np.linalg.norm(np.asarray(sec_positions, dtype=float) - source_pos, axis=1).astype(float),
            "err_radii": np.linalg.norm(np.asarray(err_positions, dtype=float) - source_pos, axis=1).astype(float),
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

    def _max_forward_distance_inside_room(
        self,
        point: np.ndarray,
        direction: np.ndarray,
        room_size: np.ndarray,
    ) -> float:
        point = np.asarray(point, dtype=float)
        direction = np.asarray(direction, dtype=float)
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

    def _derive_triplets_from_secondary(
        self,
        sec_positions: np.ndarray,
        focus_point: np.ndarray,
        room_size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sec_positions = np.asarray(sec_positions, dtype=float)

        for _ in range(40):
            ref_positions = np.zeros_like(sec_positions, dtype=float)
            err_positions = np.zeros_like(sec_positions, dtype=float)
            for i, sec_pos in enumerate(sec_positions):
                direction = self._sample_local_direction(sec_pos, focus_point)
                ref_gap = float(self.rng.uniform(*self.cfg.node_ref_to_sec_range))
                err_gap = ref_gap + float(self.rng.uniform(*self.cfg.node_err_extra_range))
                ref_positions[i] = sec_pos - direction * ref_gap
                err_positions[i] = sec_pos + direction * err_gap

            if self._all_inside_room(ref_positions, room_size) and self._all_inside_room(err_positions, room_size):
                return ref_positions, err_positions

        raise QCError("geometry:failed_to_place_triplets")

    def _derive_triplets_from_source(
        self,
        source_pos: np.ndarray,
        sec_positions: np.ndarray,
        room_size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_pos = np.asarray(source_pos, dtype=float)
        sec_positions = np.asarray(sec_positions, dtype=float)

        for _ in range(40):
            ref_positions = np.zeros_like(sec_positions, dtype=float)
            err_positions = np.zeros_like(sec_positions, dtype=float)
            valid = True
            for i, sec_pos in enumerate(sec_positions):
                axis = np.asarray(sec_pos, dtype=float) - source_pos
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm <= float(self.cfg.min_source_device_distance + self.cfg.node_ref_to_sec_range[0] + 0.02):
                    valid = False
                    break

                direction = axis / axis_norm
                ref_gap_lo = float(self.cfg.node_ref_to_sec_range[0])
                ref_gap_hi = min(
                    float(self.cfg.node_ref_to_sec_range[1]),
                    axis_norm - float(self.cfg.min_source_device_distance) - 0.02,
                )
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

    def _sample_source_radial_layout(self, room_size: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(80):
            base_start = float(self.rng.uniform(0.0, 360.0))
            spacing = np.array(self.cfg.azimuth_spacing_base, dtype=float)
            spacing += self.rng.uniform(
                -self.cfg.azimuth_spacing_jitter,
                self.cfg.azimuth_spacing_jitter,
                size=self.cfg.num_nodes,
            )
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

            side_margin_xy = min(
                float(np.min(room_size[:2]) / 2.0 - 0.05),
                float(self.cfg.wall_margin + max_err_radius + 0.05),
            )
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
            sec_positions = np.zeros((self.cfg.num_nodes, 3), dtype=float)

            if axis == 0:
                x_base = base_margin + float(self.rng.uniform(0.05, 0.35))
                x_base = x_base if low_side else float(room_size[0] - x_base)
                y_low = float(self.cfg.wall_margin + 0.20)
                y_high = float(room_size[1] - self.cfg.wall_margin - 0.20)
                if y_high <= y_low:
                    continue
                y_center = float(self.rng.uniform(y_low, y_high))
                max_half_span = max((y_high - y_low) / 2.0, 0.20)
                half_span_hi = min(1.40, max_half_span)
                if half_span_hi <= 0.20:
                    continue
                half_span = float(self.rng.uniform(0.20, half_span_hi))
                span_low = max(y_low, y_center - half_span)
                span_high = min(y_high, y_center + half_span)
                if span_high <= span_low:
                    continue

                sec_positions[:, 0] = x_base + self.rng.uniform(-0.18, 0.18, size=self.cfg.num_nodes)
                sec_positions[:, 1] = np.sort(self.rng.uniform(span_low, span_high, size=self.cfg.num_nodes))

                if low_side:
                    source_x = (max(float(room_size[0] * 0.45), self.cfg.source_wall_margin), float(room_size[0] - self.cfg.source_wall_margin))
                else:
                    source_x = (float(self.cfg.source_wall_margin), min(float(room_size[0] * 0.55), float(room_size[0] - self.cfg.source_wall_margin)))
                source_y = (float(self.cfg.source_wall_margin), float(room_size[1] - self.cfg.source_wall_margin))
            else:
                y_base = base_margin + float(self.rng.uniform(0.05, 0.35))
                y_base = y_base if low_side else float(room_size[1] - y_base)
                x_low = float(self.cfg.wall_margin + 0.20)
                x_high = float(room_size[0] - self.cfg.wall_margin - 0.20)
                if x_high <= x_low:
                    continue
                x_center = float(self.rng.uniform(x_low, x_high))
                max_half_span = max((x_high - x_low) / 2.0, 0.20)
                half_span_hi = min(1.40, max_half_span)
                if half_span_hi <= 0.20:
                    continue
                half_span = float(self.rng.uniform(0.20, half_span_hi))
                span_low = max(x_low, x_center - half_span)
                span_high = min(x_high, x_center + half_span)
                if span_high <= span_low:
                    continue

                sec_positions[:, 0] = np.sort(self.rng.uniform(span_low, span_high, size=self.cfg.num_nodes))
                sec_positions[:, 1] = y_base + self.rng.uniform(-0.18, 0.18, size=self.cfg.num_nodes)

                source_x = (float(self.cfg.source_wall_margin), float(room_size[0] - self.cfg.source_wall_margin))
                if low_side:
                    source_y = (max(float(room_size[1] * 0.45), self.cfg.source_wall_margin), float(room_size[1] - self.cfg.source_wall_margin))
                else:
                    source_y = (float(self.cfg.source_wall_margin), min(float(room_size[1] * 0.55), float(room_size[1] - self.cfg.source_wall_margin)))

            for i in range(self.cfg.num_nodes):
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

            sec_positions = np.zeros((self.cfg.num_nodes, 3), dtype=float)
            for i in range(self.cfg.num_nodes):
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

            sec_positions = np.zeros((self.cfg.num_nodes, 3), dtype=float)
            for i in range(self.cfg.num_nodes):
                sec_positions[i] = np.array(
                    [
                        self.rng.uniform(xy_margin, room_size[0] - xy_margin),
                        self.rng.uniform(xy_margin, room_size[1] - xy_margin),
                        self._sample_secondary_height(room_size),
                    ],
                    dtype=float,
                )

            if self._min_pairwise_distance(sec_positions) < float(self.cfg.min_secondary_node_distance * 1.05):
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

            if not self._all_inside_room(ref_positions, room_size):
                continue
            if not self._all_inside_room(sec_positions, room_size):
                continue
            if not self._all_inside_room(err_positions, room_size):
                continue

            if self._min_pairwise_distance(sec_positions) < self.cfg.min_secondary_node_distance:
                continue
            if self._triangle_area_xy(sec_positions) < float(self.cfg.min_secondary_triangle_area):
                continue
            if not self._source_angle_separation_ok(source_pos, sec_positions):
                continue

            all_devices = np.vstack([ref_positions, sec_positions, err_positions])
            if self._min_pairwise_distance(all_devices) < self.cfg.min_device_distance:
                continue
            if float(np.min(np.linalg.norm(all_devices - np.asarray(source_pos, dtype=float), axis=1))) < float(self.cfg.min_source_device_distance):
                continue

            causality_ok, causality_margin_min = self._neighbor_causality_status(ref_positions, sec_positions, err_positions)
            if not causality_ok:
                continue

            sound_speed = self.rng.uniform(self.cfg.sound_speed_min, self.cfg.sound_speed_max)
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
                "image_order": image_order,
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

        for i in range(self.cfg.num_nodes):
            mgr.add_reference_microphone(int(self.ref_ids[i]), sampled["ref_positions"][i])
            mgr.add_secondary_speaker(int(self.sec_ids[i]), sampled["sec_positions"][i])
            mgr.add_error_microphone(int(self.err_ids[i]), sampled["err_positions"][i])

        return mgr


class ANCQualityController:
    """RIR 合法性与 ANC 效果质量控制。"""

    def __init__(self, cfg: DatasetBuildConfig):
        self.cfg = cfg

    @staticmethod
    def _path_is_legal(path: np.ndarray) -> bool:
        if path.size < 8:
            return False
        if not np.all(np.isfinite(path)):
            return False
        if float(np.sum(path**2)) <= np.finfo(float).eps:
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
        distance = float(np.linalg.norm(tx_pos - rx_pos))
        expected_idx = int(round(distance / max(sound_speed, 1e-6) * fs))
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
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        ratios: list[float] = []

        # 完整次级通路张量: [sec_idx, err_idx, rir_tap]
        s_paths = np.zeros((self.cfg.num_nodes, self.cfg.num_nodes, self.cfg.rir_store_len), dtype=np.float32)
        s_lengths = np.zeros((self.cfg.num_nodes, self.cfg.num_nodes), dtype=np.int32)

        for k in range(self.cfg.num_nodes):
            sec_id = int(sec_ids[k])
            for m in range(self.cfg.num_nodes):
                err_id = int(err_ids[m])

                s_km = np.asarray(mgr.get_secondary_rir(sec_id, err_id), dtype=float)
                if not self._path_is_legal(s_km):
                    raise QCError("rir:illegal_secondary_path")

                keep = min(self.cfg.rir_store_len, s_km.size)
                s_paths[k, m, :keep] = s_km[:keep].astype(np.float32)
                s_lengths[k, m] = int(keep)

                ratio_s = self._direct_energy_ratio(
                    s_km,
                    np.asarray(sampled["sec_positions"][k], dtype=float),
                    np.asarray(sampled["err_positions"][m], dtype=float),
                    int(mgr.fs),
                    float(mgr.sound_speed),
                )
                ratios.append(ratio_s)

        for i in range(self.cfg.num_nodes):
            ref_id = int(ref_ids[i])
            err_id = int(err_ids[i])

            p_ref = np.asarray(mgr.get_reference_rir(source_id, ref_id), dtype=float)
            p_err = np.asarray(mgr.get_primary_rir(source_id, err_id), dtype=float)

            if not self._path_is_legal(p_ref) or not self._path_is_legal(p_err):
                raise QCError("rir:illegal_primary_or_reference_path")

            ratio_ref = self._direct_energy_ratio(
                p_ref,
                np.asarray(sampled["source_pos"], dtype=float),
                np.asarray(sampled["ref_positions"][i], dtype=float),
                int(mgr.fs),
                float(mgr.sound_speed),
            )
            ratio_err = self._direct_energy_ratio(
                p_err,
                np.asarray(sampled["source_pos"], dtype=float),
                np.asarray(sampled["err_positions"][i], dtype=float),
                int(mgr.fs),
                float(mgr.sound_speed),
            )

            ratios.append(ratio_ref)
            ratios.append(ratio_err)

        ratio_min = float(np.min(ratios))
        if ratio_min < self.cfg.min_direct_ratio:
            raise QCError("rir:direct_path_not_dominant_enough")

        # ANC 可控性预筛：若次级路径能量相对主路径过低，先淘汰，避免进入昂贵的 CFxLMS 迭代。
        control_ratios = []
        for m in range(self.cfg.num_nodes):
            p_err = np.asarray(mgr.get_primary_rir(source_id, int(err_ids[m])), dtype=float)
            p_norm = float(np.linalg.norm(p_err)) + np.finfo(float).eps
            s_norm_best = 0.0
            for k in range(self.cfg.num_nodes):
                s_km = np.asarray(mgr.get_secondary_rir(int(sec_ids[k]), int(err_ids[m])), dtype=float)
                s_norm_best = max(s_norm_best, float(np.linalg.norm(s_km)))
            control_ratios.append(s_norm_best / p_norm)
        control_ratio_min = float(np.min(control_ratios))
        if control_ratio_min < 0.10:
            raise QCError("rir:insufficient_secondary_control_energy")

        return {"direct_ratio_min": ratio_min, "control_ratio_min": control_ratio_min}, s_paths, s_lengths

    @staticmethod
    def _nr_db(d_seg: np.ndarray, e_seg: np.ndarray) -> float:
        d_pow = float(np.mean(d_seg**2)) + np.finfo(float).eps
        e_pow = float(np.mean(e_seg**2)) + np.finfo(float).eps
        return float(10.0 * np.log10(d_pow / e_pow))

    def _nr_metrics(self, d: np.ndarray, e: np.ndarray, fs: int) -> dict[str, float]:
        n = int(d.shape[0])
        if n < 16:
            raise QCError("anc:signal_too_short")

        win = min(max(int(round(0.5 * fs)), 8), n // 2)

        nr_first = self._nr_db(d[:win], e[:win])
        nr_last = self._nr_db(d[-win:], e[-win:])
        nr_gain = nr_last - nr_first

        return {
            "nr_first_db": float(nr_first),
            "nr_last_db": float(nr_last),
            "nr_gain_db": float(nr_gain),
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
            except Exception as exc:
                continue

            e = np.asarray(results["err_hist"], dtype=float)
            if e.ndim != 2 or not np.all(np.isfinite(e)):
                continue

            filter_coeffs = results.get("filter_coeffs", {})
            filters_are_finite = True
            for sec_id in sec_ids:
                w_mat = np.asarray(filter_coeffs.get(int(sec_id), []), dtype=float)
                if w_mat.ndim != 2 or not np.all(np.isfinite(w_mat)):
                    filters_are_finite = False
                    break
            if not filters_are_finite:
                continue

            metrics = self._nr_metrics(d, e, int(mgr.fs))

            passed = metrics["nr_last_db"] >= self.cfg.min_nr_last_db and metrics["nr_gain_db"] >= self.cfg.min_nr_gain_db
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

        # 提取完整控制滤波器张量 [sec, ref, L]，并保留对角滤波器兼容旧流程。
        w_opt = np.zeros((self.cfg.num_nodes, self.cfg.filter_len), dtype=np.float32)
        w_full = np.zeros((self.cfg.num_nodes, self.cfg.num_nodes, self.cfg.filter_len), dtype=np.float32)
        filter_coeffs = best["results"]["filter_coeffs"]
        for sec_idx, sec_id in enumerate(sec_ids):
            w_mat = np.asarray(filter_coeffs[int(sec_id)], dtype=float)
            if w_mat.ndim != 2 or w_mat.shape[0] < self.cfg.filter_len:
                raise QCError("anc:unexpected_filter_shape")

            keep_l = min(self.cfg.filter_len, w_mat.shape[0])
            keep_r = min(self.cfg.num_nodes, w_mat.shape[1])
            if keep_r <= 0:
                raise QCError("anc:unexpected_filter_shape")
            w_full[sec_idx, :keep_r, :keep_l] = w_mat[:keep_l, :keep_r].T.astype(np.float32)

            ref_idx = min(sec_idx, keep_r - 1)
            w_opt[sec_idx] = w_full[sec_idx, ref_idx, : self.cfg.filter_len].astype(np.float32)

        best["w_opt"] = w_opt
        best["w_full"] = w_full
        return best


class HDF5DatasetWriter:
    """HDF5 双轨存储器：/raw 与 /processed。"""

    def __init__(self, cfg: DatasetBuildConfig, output_path: Path):
        self.cfg = cfg
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.h5 = h5py.File(str(self.output_path), "w")
        self._build_layout()

    def _build_layout(self) -> None:
        n = int(self.cfg.target_rooms)

        self.h5.attrs["schema"] = "CFxLMS_ANC_QC_Dataset_v2_cross_secondary"
        self.h5.attrs["config_json"] = json.dumps(asdict(self.cfg), ensure_ascii=False)

        raw = self.h5.create_group("raw")
        room = raw.create_group("room_params")

        room.create_dataset("room_size", shape=(n, 3), dtype="f4")
        room.create_dataset("source_position", shape=(n, 3), dtype="f4")
        room.create_dataset("ref_positions", shape=(n, self.cfg.num_nodes, 3), dtype="f4")
        room.create_dataset("sec_positions", shape=(n, self.cfg.num_nodes, 3), dtype="f4")
        room.create_dataset("err_positions", shape=(n, self.cfg.num_nodes, 3), dtype="f4")

        room.create_dataset("azimuth_deg", shape=(n, self.cfg.num_nodes), dtype="f4")
        room.create_dataset("ref_radii", shape=(n, self.cfg.num_nodes), dtype="f4")
        room.create_dataset("sec_radii", shape=(n, self.cfg.num_nodes), dtype="f4")
        room.create_dataset("err_radii", shape=(n, self.cfg.num_nodes), dtype="f4")
        room.create_dataset("z_offsets", shape=(n, self.cfg.num_nodes), dtype="f4")

        room.create_dataset("sound_speed", shape=(n,), dtype="f4")
        room.create_dataset("material_absorption", shape=(n,), dtype="f4")
        room.create_dataset("image_source_order", shape=(n,), dtype="i4")
        room.create_dataset("causality_margin_min", shape=(n,), dtype="f4")
        room.create_dataset("layout_mode", shape=(n,), dtype=h5py.string_dtype(encoding="utf-8"))

        raw.create_dataset("x_ref", shape=(n, self.cfg.num_nodes, self.cfg.ref_window_samples), dtype="f4")
        # 兼容旧流程: 仅保留对角线(本节点)次级通路。
        raw.create_dataset("S_paths", shape=(n, self.cfg.num_nodes, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("S_path_lengths", shape=(n, self.cfg.num_nodes), dtype="i4")
        # 新流程: 保存完整次级通路矩阵 [sec, err, rir]。
        raw.create_dataset("S_paths_full", shape=(n, self.cfg.num_nodes, self.cfg.num_nodes, self.cfg.rir_store_len), dtype="f4")
        raw.create_dataset("S_path_lengths_full", shape=(n, self.cfg.num_nodes, self.cfg.num_nodes), dtype="i4")

        # 兼容旧流程: 对角线控制滤波器 [node, L]。
        raw.create_dataset("W_opt", shape=(n, self.cfg.num_nodes, self.cfg.filter_len), dtype="f4")
        # 新流程: 完整控制滤波器矩阵 [sec, ref, L]。
        raw.create_dataset("W_full", shape=(n, self.cfg.num_nodes, self.cfg.num_nodes, self.cfg.filter_len), dtype="f4")

        qc = raw.create_group("qc_metrics")
        qc.create_dataset("nr_first_db", shape=(n,), dtype="f4")
        qc.create_dataset("nr_last_db", shape=(n,), dtype="f4")
        qc.create_dataset("nr_gain_db", shape=(n,), dtype="f4")
        qc.create_dataset("direct_ratio_min", shape=(n,), dtype="f4")
        qc.create_dataset("control_ratio_min", shape=(n,), dtype="f4")
        qc.create_dataset("mu_used", shape=(n,), dtype="f4")
        qc.create_dataset("source_seed", shape=(n,), dtype="i8")
        qc.create_dataset("warmup_start_s", shape=(n,), dtype="f4")
        qc.create_dataset("warmup_start_index", shape=(n,), dtype="i4")

        processed = self.h5.create_group("processed")
        processed.create_dataset("gcc_phat", shape=(n, 3, self.cfg.gcc_truncated_len), dtype="f4")
        processed.create_dataset("psd_features", shape=(n, self.cfg.psd_nfft // 2 + 1), dtype="f4")
        processed.create_dataset("S_pca_coeffs", shape=(n, self.cfg.svd_components), dtype="f4")
        processed.create_dataset("W_pca_coeffs", shape=(n, self.cfg.svd_components), dtype="f4")
        processed.create_dataset(
            "V_w",
            shape=(self.cfg.svd_components, self.cfg.w_feature_dim),
            dtype="f4",
        )

        svd_group = processed.create_group("global_svd")
        svd_group.create_dataset("S_mean", shape=(self.cfg.s_feature_dim,), dtype="f4")
        svd_group.create_dataset(
            "S_components",
            shape=(self.cfg.svd_components, self.cfg.s_feature_dim),
            dtype="f4",
        )
        svd_group.create_dataset("W_mean", shape=(self.cfg.w_feature_dim,), dtype="f4")
        svd_group.create_dataset(
            "W_components",
            shape=(self.cfg.svd_components, self.cfg.w_feature_dim),
            dtype="f4",
        )

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
        diag_idx = np.arange(self.cfg.num_nodes, dtype=int)
        self.h5["raw/S_paths"][idx] = sample.s_paths[diag_idx, diag_idx, :].astype(np.float32)
        self.h5["raw/S_path_lengths"][idx] = sample.s_path_lengths[diag_idx, diag_idx].astype(np.int32)
        self.h5["raw/S_paths_full"][idx] = sample.s_paths.astype(np.float32)
        self.h5["raw/S_path_lengths_full"][idx] = sample.s_path_lengths.astype(np.int32)
        self.h5["raw/W_opt"][idx] = sample.w_opt.astype(np.float32)
        self.h5["raw/W_full"][idx] = sample.w_full.astype(np.float32)

        self.h5["raw/qc_metrics/nr_first_db"][idx] = float(sample.qc_metrics["nr_first_db"])
        self.h5["raw/qc_metrics/nr_last_db"][idx] = float(sample.qc_metrics["nr_last_db"])
        self.h5["raw/qc_metrics/nr_gain_db"][idx] = float(sample.qc_metrics["nr_gain_db"])
        self.h5["raw/qc_metrics/direct_ratio_min"][idx] = float(sample.qc_metrics["direct_ratio_min"])
        self.h5["raw/qc_metrics/control_ratio_min"][idx] = float(sample.qc_metrics["control_ratio_min"])
        self.h5["raw/qc_metrics/mu_used"][idx] = float(sample.qc_metrics["mu_used"])
        self.h5["raw/qc_metrics/source_seed"][idx] = int(sample.qc_metrics["source_seed"])
        self.h5["raw/qc_metrics/warmup_start_s"][idx] = float(sample.qc_metrics["warmup_start_s"])
        self.h5["raw/qc_metrics/warmup_start_index"][idx] = int(sample.qc_metrics["warmup_start_index"])

        self.h5["processed/gcc_phat"][idx] = sample.gcc_phat.astype(np.float32)
        self.h5["processed/psd_features"][idx] = sample.psd_features.astype(np.float32)

    def close(self) -> None:
        if self.h5:
            self.h5.flush()
            self.h5.close()


class ANCDatasetBuilder:
    """CFxLMS 高保真数据集构建器。"""

    def __init__(self, cfg: DatasetBuildConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.py_random = random.Random(cfg.random_seed)

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

        fs = float(self.cfg.fs)
        latest_start_s = (total_samples - window) / fs
        start_min_s = float(self.cfg.warmup_start_s_min)
        start_max_s = min(float(self.cfg.warmup_start_s_max), latest_start_s)
        if start_max_s < start_min_s:
            raise QCError("warmup:invalid_start_interval")

        # 使用 Python random 在稳定区间随机采样起点，避免固定起点伪影。
        start_s = float(self.py_random.uniform(start_min_s, start_max_s))
        start_idx = int(round(start_s * fs))
        start_idx = max(0, min(start_idx, total_samples - window))
        return start_idx, start_s

    def _build_single_sample(self) -> RoomSample:
        sampled = self.sampler.sample()
        mgr = self.sampler.build_manager(sampled)
        mgr.build(verbose=False)

        rir_metrics, s_paths, s_lengths = self.qc.validate_rirs(
            mgr=mgr,
            sampled=sampled,
            source_id=int(self.sampler.source_id),
            ref_ids=self.sampler.ref_ids,
            sec_ids=self.sampler.sec_ids,
            err_ids=self.sampler.err_ids,
        )

        # 每条样本单独记录源噪声种子，便于抽查时严格复现同一激励。
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
        if x.ndim != 2 or x.shape[1] < self.cfg.num_nodes:
            raise QCError("anc:unexpected_reference_shape")
        start_idx, start_s = self._sample_warmup_start(x.shape[0])
        end_idx = start_idx + self.cfg.ref_window_samples
        x_ref = x[start_idx:end_idx, : self.cfg.num_nodes].T.astype(np.float32)

        gcc = self.feature_processor.compute_gcc_phat(x_ref)
        psd = self.feature_processor.compute_psd_features(x_ref[0])

        qc_metrics = {
            "nr_first_db": float(anc_result["metrics"]["nr_first_db"]),
            "nr_last_db": float(anc_result["metrics"]["nr_last_db"]),
            "nr_gain_db": float(anc_result["metrics"]["nr_gain_db"]),
            "direct_ratio_min": float(rir_metrics["direct_ratio_min"]),
            "control_ratio_min": float(rir_metrics["control_ratio_min"]),
            "mu_used": float(anc_result["mu"]),
            "source_seed": int(source_seed),
            "warmup_start_s": float(start_s),
            "warmup_start_index": int(start_idx),
        }

        return RoomSample(
            room_params=sampled,
            x_ref=x_ref,
            s_paths=s_paths,
            s_path_lengths=s_lengths,
            w_opt=np.asarray(anc_result["w_opt"], dtype=np.float32),
            w_full=np.asarray(anc_result["w_full"], dtype=np.float32),
            gcc_phat=gcc,
            psd_features=psd,
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

                    if accepted % self.cfg.progress_interval == 0 or accepted == self.cfg.target_rooms:
                        print(
                            f"[Progress] accepted={accepted}/{self.cfg.target_rooms}, "
                            f"attempts={attempts}, pass_rate={accepted / max(attempts, 1):.3f}"
                        )
                        writer.h5.attrs["accepted_so_far"] = int(accepted)
                        writer.h5.attrs["attempts_so_far"] = int(attempts)
                        writer.h5.flush()
                except QCError as exc:
                    self._count_fail(str(exc))
                except Exception as exc:
                    self._count_fail(f"unexpected:{type(exc).__name__}")

                if attempts % self.cfg.attempt_log_interval == 0:
                    top_reason = "-"
                    top_count = 0
                    if self.failure_stats:
                        top_reason, top_count = max(self.failure_stats.items(), key=lambda kv: kv[1])
                    print(
                        f"[Attempt] attempts={attempts}, accepted={accepted}/{self.cfg.target_rooms}, "
                        f"pass_rate={accepted / max(attempts, 1):.3f}, top_fail={top_reason}:{top_count}"
                    )

            if accepted < self.cfg.target_rooms:
                raise RuntimeError(
                    f"数据集构建失败：在 {attempts} 次尝试后仅通过 {accepted} 条。"
                    f"请放宽 QC 阈值或提高 max_total_attempts。"
                )

            writer.h5.attrs["accepted_rooms"] = int(accepted)
            writer.h5.attrs["attempts"] = int(attempts)
            writer.h5.attrs["failure_stats_json"] = json.dumps(self.failure_stats, ensure_ascii=False)
            writer.h5.flush()
        finally:
            self.layout_previewer.finalize()
            writer.close()

        print("数据集原始与局部特征已写入完成。")
        if self.failure_stats:
            top_fail = sorted(self.failure_stats.items(), key=lambda kv: kv[1], reverse=True)[:8]
            print("常见失败原因（Top8）:")
            for reason, cnt in top_fail:
                print(f"  - {reason}: {cnt}")

        return output_path


def _svd_project(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("输入矩阵必须为二维。")

    mean = np.mean(x, axis=0, keepdims=True)
    centered = x - mean

    # 通过紧致 SVD 提取主方向。
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_comp = min(k, vt.shape[0])
    comp = vt[:n_comp]
    coeff = centered @ comp.T

    return coeff.astype(np.float32), mean.reshape(-1).astype(np.float32), comp.astype(np.float32)


def _ensure_dataset(group: h5py.Group, name: str, shape: tuple[int, ...], dtype: str):
    """确保数据集存在且形状匹配，不匹配时删除并重建。"""
    if name in group:
        ds = group[name]
        if tuple(ds.shape) == tuple(shape) and str(ds.dtype) == str(np.dtype(dtype)):
            return ds
        del group[name]
    return group.create_dataset(name, shape=shape, dtype=dtype)


def compute_global_svd(h5_path: str | Path, n_components: int = 32) -> None:
    """全局聚合函数：读取全部 raw 数据并写入 PCA/SVD 降维系数。"""

    h5_file = Path(h5_path)
    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5 文件不存在: {h5_file}")

    with h5py.File(str(h5_file), "r+") as h5:
        if "raw/S_paths_full" in h5:
            s_raw = np.asarray(h5["raw/S_paths_full"], dtype=np.float64)
        else:
            s_raw = np.asarray(h5["raw/S_paths"], dtype=np.float64)

        if "raw/W_full" in h5:
            w_raw = np.asarray(h5["raw/W_full"], dtype=np.float64)
        else:
            w_raw = np.asarray(h5["raw/W_opt"], dtype=np.float64)

        n_rooms = s_raw.shape[0]
        s_mat = s_raw.reshape(n_rooms, -1)
        w_mat = w_raw.reshape(n_rooms, -1)

        s_coeff, s_mean, s_comp = _svd_project(s_mat, int(n_components))
        w_coeff, w_mean, w_comp = _svd_project(w_mat, int(n_components))

        processed = h5["processed"] if "processed" in h5 else h5.create_group("processed")
        svd_group = processed["global_svd"] if "global_svd" in processed else processed.create_group("global_svd")

        s_pca_ds = _ensure_dataset(processed, "S_pca_coeffs", (n_rooms, int(n_components)), "f4")
        w_pca_ds = _ensure_dataset(processed, "W_pca_coeffs", (n_rooms, int(n_components)), "f4")
        v_w_ds = _ensure_dataset(processed, "V_w", (int(n_components), w_mat.shape[1]), "f4")

        s_mean_ds = _ensure_dataset(svd_group, "S_mean", (s_mat.shape[1],), "f4")
        w_mean_ds = _ensure_dataset(svd_group, "W_mean", (w_mat.shape[1],), "f4")
        s_comp_ds = _ensure_dataset(svd_group, "S_components", (int(n_components), s_mat.shape[1]), "f4")
        w_comp_ds = _ensure_dataset(svd_group, "W_components", (int(n_components), w_mat.shape[1]), "f4")

        # 若请求维度大于可用秩，按实际维度写入前段，其余保持 0。
        s_pca_ds[:] = 0.0
        w_pca_ds[:] = 0.0
        s_pca_ds[:, : s_coeff.shape[1]] = s_coeff
        w_pca_ds[:, : w_coeff.shape[1]] = w_coeff

        s_mean_ds[:] = s_mean
        w_mean_ds[:] = w_mean

        s_comp_ds[:] = 0.0
        w_comp_ds[:] = 0.0
        s_comp_ds[: s_comp.shape[0], :] = s_comp
        w_comp_ds[: w_comp.shape[0], :] = w_comp

        # 训练端统一使用 /processed/V_w 作为全局物理基底矩阵。
        v_w_ds[:] = 0.0
        v_w_ds[: w_comp.shape[0], :] = w_comp

        h5.attrs["svd_components_requested"] = int(n_components)
        h5.attrs["svd_components_effective_s"] = int(s_comp.shape[0])
        h5.attrs["svd_components_effective_w"] = int(w_comp.shape[0])

    print(f"全局 SVD 聚合完成: {h5_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 CFxLMS 高质量 ANC 多通道数据集（含 QC + HDF5 双轨存储 + 全局 SVD）")
    parser.add_argument("--num-rooms", type=int, default=1000, help="目标通过 QC 的房间样本数。")
    parser.add_argument("--max-attempts", type=int, default=30000, help="总尝试上限（失败会自动重采样）。")
    parser.add_argument("--seed", type=int, default=20260329, help="随机种子。")
    parser.add_argument(
        "--output-h5",
        type=str,
        default=str(ROOT_DIR / "python_scripts" / "cfxlms_qc_dataset.h5"),
        help="输出 HDF5 文件路径。",
    )
    parser.add_argument("--svd-components", type=int, default=32, help="全局 SVD 主成分数。")
    return parser.parse_args()


# Override the legacy argparse block above with clean, UTF-8 help text.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建带 QC 与全局 SVD 的 CFxLMS 多通道 ANC 数据集。")
    parser.add_argument("--num-rooms", type=int, default=1000, help="目标通过 QC 的房间样本数。")
    parser.add_argument("--max-attempts", type=int, default=30000, help="总尝试上限，失败会自动重采样。")
    parser.add_argument("--seed", type=int, default=20260329, help="随机种子。")
    parser.add_argument(
        "--output-h5",
        type=str,
        default=str(ROOT_DIR / "python_scripts" / "cfxlms_qc_dataset.h5"),
        help="输出 HDF5 文件路径。",
    )
    parser.add_argument("--svd-components", type=int, default=32, help="全局 SVD 主成分数。")
    parser.add_argument("--no-preview-layouts", action="store_true", help="关闭实时布局预览窗口。")
    parser.add_argument("--preview-interval", type=int, default=1, help="每通过多少个样本刷新一次布局预览。")
    return parser.parse_args()


def _print_run_parameters(cfg: DatasetBuildConfig) -> None:
    print("========== 数据集运行参数 ==========")
    print("参数对齐说明: 除三节点几何布局外，核心仿真参数与 test.py 对齐。")
    print(f"节点数: {cfg.num_nodes} (按数据集要求固定)")
    print(f"采样率 fs: {cfg.fs} Hz")
    print(f"仿真时长: {cfg.noise_duration_s:.2f} s")
    print(f"白噪声频段: [{cfg.f_low:.1f}, {cfg.f_high:.1f}] Hz")
    print(f"滤波器阶数 L: {cfg.filter_len}")
    print(f"步长候选 mu: {cfg.mu_candidates}")
    print(f"CFxLMS 归一化更新: {cfg.anc_normalized_update}, norm_eps={cfg.anc_norm_epsilon:.1e}")
    print(f"ANC 阈值: min_nr_last_db={cfg.min_nr_last_db:.2f}, min_nr_gain_db={cfg.min_nr_gain_db:.2f}")
    print(
        "邻居因果约束: "
        f"tol={cfg.neighbor_causality_tolerance:.2f}m, "
        f"max_violations={cfg.max_causality_violations}"
    )
    print(f"x_ref 长度: {cfg.ref_window_ms:.1f} ms ({cfg.ref_window_samples} samples)")
    print(
        "预热截取法: "
        f"在 [{cfg.warmup_start_s_min:.2f}, {cfg.warmup_start_s_max:.2f}] s 随机起点截取 x_ref "
        "(使用 Python random.uniform)"
    )
    print("次级通路保存: 对角自通路 + 全交叉矩阵 S_paths_full[sec,err,:]")
    print("滤波器保存: 兼容对角 W_opt + 全矩阵 W_full[sec,ref,:]")
    print(f"目标样本数: {cfg.target_rooms}")
    print(f"最大尝试数: {cfg.max_total_attempts}")
    print(f"尝试日志间隔: 每 {cfg.attempt_log_interval} 次")
    print(f"输出文件: {cfg.output_h5}")
    print("====================================")


def main() -> None:
    args = parse_args()

    cfg = DatasetBuildConfig(
        target_rooms=int(args.num_rooms),
        max_total_attempts=int(args.max_attempts),
        random_seed=int(args.seed),
        output_h5=str(args.output_h5),
        svd_components=int(args.svd_components),
        layout_preview=not bool(args.no_preview_layouts),
        layout_preview_interval=max(int(args.preview_interval), 1),
    )

    print("开始构建 CFxLMS 高质量数据集...")
    _print_run_parameters(cfg)
    builder = ANCDatasetBuilder(cfg)

    output_path = builder.build_dataset()

    # 按要求在全部样本完成后执行全局 SVD 聚合。
    compute_global_svd(output_path, n_components=cfg.svd_components)

    print("全部完成。")
    print(f"输出文件: {output_path}")


if __name__ == "__main__":
    main()
'''

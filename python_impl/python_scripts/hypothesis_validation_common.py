from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, TensorDataset


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

from python_scripts.cfxlms_single_control_dataset_impl import (
    _canonical_q_from_paths,
    _normalize_columns,
    _rolling_mse_db,
    _solve_w_canonical_from_q,
)
from python_scripts.multi_control_canonical_q_common import set_seed, split_indices
from python_scripts.single_control_canonical_q_common import compute_reference_covariance_batch


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def next_pow2(n: int) -> int:
    return 1 if int(n) <= 1 else 1 << (int(n) - 1).bit_length()


def central_crop(x: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == int(target_len):
        return arr.astype(np.float32)
    if arr.size < int(target_len):
        pad_total = int(target_len) - arr.size
        left = pad_total // 2
        right = pad_total - left
        return np.pad(arr, (left, right), mode="constant").astype(np.float32)
    mid = arr.size // 2
    half = int(target_len) // 2
    if int(target_len) % 2 == 1:
        return arr[mid - half : mid + half + 1].astype(np.float32)
    return arr[mid - half : mid + half].astype(np.float32)


LOCALIZATION_WINDOW_PRESETS: dict[str, dict[str, int]] = {
    "W1": {"signal_len": 4096, "ref_window_len": 2048},
    "W2": {"signal_len": 8192, "ref_window_len": 4096},
    "W3": {"signal_len": 16384, "ref_window_len": 8192},
}


def resolve_localization_window_preset(
    window_preset: str,
    signal_len: int | None = None,
    ref_window_len: int | None = None,
) -> dict[str, int | str]:
    preset = str(window_preset).upper()
    if preset not in LOCALIZATION_WINDOW_PRESETS:
        raise KeyError(f"Unknown localization window preset: {preset}")
    cfg = dict(LOCALIZATION_WINDOW_PRESETS[preset])
    if signal_len is not None:
        cfg["signal_len"] = int(signal_len)
    if ref_window_len is not None:
        cfg["ref_window_len"] = int(ref_window_len)
    cfg["window_preset"] = preset
    return cfg


def gcc_pair(sig_a: np.ndarray, sig_b: np.ndarray, out_len: int, phat: bool = True) -> np.ndarray:
    a = np.asarray(sig_a, dtype=np.float64).reshape(-1)
    b = np.asarray(sig_b, dtype=np.float64).reshape(-1)
    n = int(a.size + b.size - 1)
    n_fft = next_pow2(n)
    a_f = np.fft.rfft(a, n=n_fft)
    b_f = np.fft.rfft(b, n=n_fft)
    cross = a_f * np.conj(b_f)
    if bool(phat):
        cross /= np.maximum(np.abs(cross), np.finfo(np.float64).eps)
    corr = np.fft.irfft(cross, n=n_fft)
    corr = np.concatenate([corr[-(n_fft // 2) :], corr[: n_fft // 2]])
    return central_crop(corr, int(out_len))


def gcc_phat_pair(sig_a: np.ndarray, sig_b: np.ndarray, out_len: int) -> np.ndarray:
    return gcc_pair(sig_a, sig_b, out_len=out_len, phat=True)


def plain_gcc_pair(sig_a: np.ndarray, sig_b: np.ndarray, out_len: int) -> np.ndarray:
    return gcc_pair(sig_a, sig_b, out_len=out_len, phat=False)


def compute_gcc_triplet(x_ref: np.ndarray, out_len: int, phat: bool = True) -> np.ndarray:
    arr = np.asarray(x_ref, dtype=np.float64)
    return np.stack(
        [
            gcc_pair(arr[0], arr[1], out_len, phat=phat),
            gcc_pair(arr[0], arr[2], out_len, phat=phat),
            gcc_pair(arr[1], arr[2], out_len, phat=phat),
        ],
        axis=0,
    ).astype(np.float32)


def compute_reference_covariance_features(x_ref: np.ndarray, n_fft: int) -> np.ndarray:
    return compute_reference_covariance_batch(np.asarray(x_ref, dtype=np.float32)[None, ...], n_fft=int(n_fft))[0]


def rolling_mse_db_multichannel(sig: np.ndarray, fs: int, window_samples: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(sig, dtype=float)
    if arr.ndim == 1:
        return _rolling_mse_db(arr, fs=int(fs), window_samples=int(window_samples))
    mean_power = np.mean(arr**2, axis=1)
    return _rolling_mse_db(np.sqrt(np.maximum(mean_power, 0.0)), fs=int(fs), window_samples=int(window_samples))


def fractional_delay_rir(delay_samples: float, rir_len: int, amplitude: float = 1.0, kernel_radius: int = 24) -> np.ndarray:
    taps = np.arange(int(rir_len), dtype=np.float64)
    center = float(delay_samples)
    kernel = np.sinc(taps - center)
    window = np.hanning(2 * int(kernel_radius) + 1)
    mask = np.abs(taps - center) <= int(kernel_radius)
    full_window = np.zeros_like(kernel)
    full_window[mask] = window[: int(np.sum(mask))]
    kernel = kernel * full_window
    if np.sum(np.abs(kernel)) <= np.finfo(np.float64).eps:
        idx = int(np.clip(round(center), 0, int(rir_len) - 1))
        kernel[idx] = 1.0
    kernel /= np.sqrt(np.sum(kernel**2)) + np.finfo(np.float64).eps
    return (float(amplitude) * kernel).astype(np.float32)


def direct_path_rir(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    fs: int,
    c: float,
    rir_len: int,
    amplitude_power: float = 1.0,
    air_alpha_per_m: float = 0.0,
) -> np.ndarray:
    distance = float(np.linalg.norm(np.asarray(tx_pos, dtype=float) - np.asarray(rx_pos, dtype=float)))
    delay = distance / float(c) * float(fs)
    geometric_decay = 1.0 / max(distance**float(amplitude_power), 1.0e-3)
    air_decay = math.exp(-max(float(air_alpha_per_m), 0.0) * distance)
    amplitude = geometric_decay * air_decay
    return fractional_delay_rir(delay, rir_len=int(rir_len), amplitude=amplitude)


def one_reflection_rir(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    room_size: np.ndarray,
    fs: int,
    c: float,
    rir_len: int,
    reflection_gain: float = 0.65,
    amplitude_power: float = 1.0,
    air_alpha_per_m: float = 0.0,
) -> np.ndarray:
    tx = np.asarray(tx_pos, dtype=float)
    rx = np.asarray(rx_pos, dtype=float)
    room = np.asarray(room_size, dtype=float)
    rir = direct_path_rir(
        tx,
        rx,
        fs=int(fs),
        c=float(c),
        rir_len=int(rir_len),
        amplitude_power=float(amplitude_power),
        air_alpha_per_m=float(air_alpha_per_m),
    ).astype(np.float64)
    image_sources = [
        np.array([-tx[0], tx[1]], dtype=float),
        np.array([2.0 * room[0] - tx[0], tx[1]], dtype=float),
        np.array([tx[0], -tx[1]], dtype=float),
        np.array([tx[0], 2.0 * room[1] - tx[1]], dtype=float),
    ]
    for image in image_sources:
        distance = float(np.linalg.norm(image - rx))
        delay = distance / float(c) * float(fs)
        geometric_decay = 1.0 / max(distance**float(amplitude_power), 1.0e-3)
        air_decay = math.exp(-max(float(air_alpha_per_m), 0.0) * distance)
        amplitude = float(reflection_gain) * geometric_decay * air_decay
        rir += fractional_delay_rir(delay, rir_len=int(rir_len), amplitude=amplitude).astype(np.float64)
    return rir.astype(np.float32)


def convolve_and_crop(sig: np.ndarray, rir: np.ndarray, out_len: int) -> np.ndarray:
    x = np.convolve(np.asarray(sig, dtype=np.float64), np.asarray(rir, dtype=np.float64), mode="full")
    return x[: int(out_len)].astype(np.float32)


def analytic_ref_triangle_signature(ref_positions: np.ndarray) -> tuple[float, float, float, float]:
    ref = np.asarray(ref_positions, dtype=np.float64)
    edges = []
    for i in range(ref.shape[0]):
        for j in range(i + 1, ref.shape[0]):
            edges.append(float(np.linalg.norm(ref[i] - ref[j])))
    edges = sorted(edges)
    a, b, c = edges
    s = 0.5 * (a + b + c)
    area = max(s * (s - a) * (s - b) * (s - c), 0.0) ** 0.5
    return (round(a, 2), round(b, 2), round(c, 2), round(area, 2))


def build_geometry_holdout_split(ref_positions: np.ndarray, seed: int, holdout_frac: float = 0.2) -> dict[str, np.ndarray]:
    signatures = [analytic_ref_triangle_signature(ref_positions[i]) for i in range(ref_positions.shape[0])]
    unique = sorted(set(signatures))
    rng = np.random.default_rng(int(seed))
    perm = np.arange(len(unique), dtype=np.int64)
    rng.shuffle(perm)
    holdout_count = max(1, int(round(len(unique) * float(holdout_frac))))
    holdout_groups = {unique[int(i)] for i in perm[:holdout_count]}
    train_candidates = np.array([i for i, sig in enumerate(signatures) if sig not in holdout_groups], dtype=np.int64)
    test_idx = np.array([i for i, sig in enumerate(signatures) if sig in holdout_groups], dtype=np.int64)
    if train_candidates.size < 2 or test_idx.size < 1:
        raise RuntimeError("Geometry holdout split is degenerate.")
    rng.shuffle(train_candidates)
    val_count = max(1, int(round(train_candidates.size * 0.1)))
    val_idx = np.sort(train_candidates[:val_count])
    train_idx = np.sort(train_candidates[val_count:])
    return {
        "geom_train": train_idx.astype(np.int64),
        "geom_val": val_idx.astype(np.int64),
        "geom_test": np.sort(test_idx).astype(np.int64),
    }


def fit_standardizer(arr: np.ndarray, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    train_arr = np.asarray(arr[train_idx], dtype=np.float32)
    mean = train_arr.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_arr.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1.0e-6))
    return {"mean": mean, "std": std}


def apply_standardizer(arr: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((np.asarray(arr, dtype=np.float32) - stats["mean"]) / stats["std"]).astype(np.float32)


def signed_log1p_np(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)


def signed_expm1_np(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    return (np.sign(x) * np.expm1(np.abs(x))).astype(np.float32)


def signed_expm1_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


STAGE_LEVELS: dict[str, dict[str, dict[str, int]]] = {
    "01": {"L1": {"num_samples": 6000, "epochs": 40}, "L2": {"num_samples": 30000, "epochs": 80}, "L3": {"num_samples": 60000, "epochs": 120}},
    "02": {"L1": {"num_samples": 6000, "epochs": 40}, "L2": {"num_samples": 50000, "epochs": 80}, "L3": {"num_samples": 60000, "epochs": 120}},
    "03": {"L1": {"num_samples": 4000, "epochs": 40}, "L2": {"num_samples": 12000, "epochs": 80}, "L3": {"num_samples": 30000, "epochs": 120}},
    "04": {"L1": {"num_samples": 4000, "epochs": 40}, "L2": {"num_samples": 12000, "epochs": 80}, "L3": {"num_samples": 30000, "epochs": 120}},
    "05": {"L1": {"num_samples": 4000, "epochs": 40}, "L2": {"num_samples": 12000, "epochs": 80}, "L3": {"num_samples": 30000, "epochs": 120}},
}


LOCALIZATION_THRESHOLDS: dict[str, dict[str, float]] = {
    "01": {
        "analytic_iid_median_max": 0.05,
        "analytic_iid_p90_max": 0.15,
        "analytic_geom_median_max": 0.10,
        "analytic_geom_p90_max": 0.25,
        "learned_iid_median_max": 0.05,
        "learned_iid_p90_max": 0.15,
        "learned_geom_median_max": 0.10,
        "learned_geom_p90_max": 0.25,
    },
    "02": {
        "learned_iid_median_max": 0.15,
        "learned_iid_p90_max": 0.35,
        "learned_geom_median_max": 0.25,
        "learned_geom_p90_max": 0.45,
    },
}


LOCALIZATION_GEOMETRY_METRIC_NAMES: tuple[str, ...] = (
    "triangle_area",
    "edge_len_min",
    "edge_len_mid",
    "edge_len_max",
    "edge_len_ratio",
    "triangle_angle_min_deg",
    "triangle_angle_mid_deg",
    "triangle_angle_max_deg",
    "obtuse_triangle_flag",
    "small_angle_flag",
    "small_angle_vertex_dist_norm",
    "small_angle_opposite_edge_dist_norm",
    "small_angle_long_edge_pref_ratio",
    "centroid_source_dist",
    "centroid_source_dist_norm",
    "inside_convex_hull",
    "min_source_ref_dist",
    "near_ref_inside_trigger_flag",
    "max_source_ref_dist",
    "jacobian_condition",
    "jacobian_sigma_min",
)


LOCALIZATION_STABLE_GEOMETRY_GRID: tuple[tuple[float, float], ...] = tuple(
    (float(area_min), float(cond_max))
    for area_min in (0.10, 0.15, 0.20, 0.25, 0.30)
    for cond_max in (15.0, 20.0, 25.0, 30.0, 40.0)
)


CONTROL_THRESHOLDS: dict[str, dict[str, float]] = {
    "03": {
        "hyperplane_vs_w_mse_test_gain_margin_db": 3.0,
        "hyperplane_test_gap_max_db": 1.0,
    },
    "04": {
        "hyperplane_vs_w_mse_test_gain_margin_db": 2.0,
        "hyperplane_test_gap_max_db": 2.0,
    },
    "05": {
        "relative_to_stage03_test_gap_abs_max_db": 0.5,
    },
}


def resolve_stage_level(stage_id: str, level: str, num_samples: int | None = None, epochs: int | None = None) -> dict[str, int | str]:
    stage = str(stage_id)
    lvl = str(level).upper()
    if stage not in STAGE_LEVELS:
        raise KeyError(f"Unknown stage: {stage}")
    if lvl not in STAGE_LEVELS[stage]:
        raise KeyError(f"Unknown level {lvl} for stage {stage}")
    cfg = dict(STAGE_LEVELS[stage][lvl])
    if num_samples is not None:
        cfg["num_samples"] = int(num_samples)
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    cfg["stage_id"] = stage
    cfg["level"] = lvl
    return cfg


def save_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_training_curves(
    history_rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    metric_keys: list[str],
    metric_labels: list[str],
    live_plot: bool = False,
) -> None:
    if not history_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg" if not live_plot else matplotlib.get_backend())
        import matplotlib.pyplot as plt
    except Exception:
        return
    epochs = [int(row["epoch"]) for row in history_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history_rows], label="train loss", color="tab:blue")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in history_rows], label="val loss", color="tab:orange")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[0].set_title("Loss")
    for key, label in zip(metric_keys, metric_labels):
        axes[1].plot(epochs, [float(row[key]) for row in history_rows], label=label)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[1].set_title("Primary Metric")
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    if live_plot:
        try:
            from IPython.display import clear_output, display

            clear_output(wait=True)
            display(fig)
        except Exception:
            try:
                plt.show(block=False)
                plt.pause(0.001)
            except Exception:
                pass
    plt.close(fig)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.10):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [int(in_dim)] + [int(hidden)] * int(depth) + [int(out_dim)]
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.GELU(), nn.Dropout(p=float(dropout))])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 4, dropout: float = 0.10):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(int(in_dim), int(hidden)), nn.GELU(), nn.Dropout(p=float(dropout)))
        blocks: list[nn.Module] = []
        for _ in range(int(depth)):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(int(hidden)),
                    nn.Linear(int(hidden), int(hidden)),
                    nn.GELU(),
                    nn.Dropout(p=float(dropout)),
                    nn.Linear(int(hidden), int(hidden)),
                    nn.Dropout(p=float(dropout)),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Linear(int(hidden), int(out_dim))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


def build_localization_model(in_dim: int, out_dim: int, model_kind: str) -> nn.Module:
    kind = str(model_kind).lower()
    if kind == "mlp":
        return SimpleMLP(int(in_dim), int(out_dim), hidden=256, depth=3, dropout=0.10)
    if kind == "resmlp":
        return ResidualMLP(int(in_dim), int(out_dim), hidden=256, depth=4, dropout=0.10)
    raise KeyError(f"Unknown localization model kind: {model_kind}")


@dataclass
class LocalizationConfig:
    num_samples: int = 6000
    fs: int = 16000
    signal_len: int = 4096
    ref_window_len: int = 2048
    gcc_len: int = 257
    psd_nfft: int = 256
    rir_len: int = 192
    c: float = 343.0
    window_preset: str = "W1"
    plane_room_size: tuple[float, float] = (4.0, 4.0)
    reflection_gain: float = 0.65
    amplitude_power: float = 1.0
    rir_model: str = "manual_2d_image_source"
    air_attenuation_enabled: bool = False
    air_attenuation_alpha_per_m: float = 0.03
    reflection_profile_mix_anechoic_frac: float = 0.0
    ref_margin: float = 0.45
    source_margin: float = 0.35
    min_ref_pair_dist: float = 0.25
    min_source_ref_dist: float = 0.30
    near_ref_inside_threshold_m: float = 0.0
    seed: int = 20260401
    profile: str = "anechoic"
    geometry_filter_mode: str = "none"
    min_triangle_area: float = 0.10
    max_jacobian_condition: float = 25.0
    max_triangle_angle_deg: float = 180.0
    require_source_inside_if_obtuse: bool = False
    small_angle_threshold_deg: float = 30.0
    small_angle_vertex_clearance_ratio: float = 0.35
    small_angle_opposite_edge_clearance_ratio: float = 0.20
    small_angle_prefer_long_edges_ratio: float = 0.85
    store_geometry_metrics: bool = True
    audit_rule_version: str = "stage01_stable_v1"
    source_region_rule_version: str = "none"


def localization_triangle_area(ref_positions: np.ndarray) -> float:
    ref = np.asarray(ref_positions, dtype=np.float64)
    a, b, c = ref
    return float(abs(0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))))


def localization_triangle_edge_lengths(ref_positions: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref_positions, dtype=np.float64)
    edges = []
    for i in range(ref.shape[0]):
        for j in range(i + 1, ref.shape[0]):
            edges.append(float(np.linalg.norm(ref[i] - ref[j])))
    return np.sort(np.asarray(edges, dtype=np.float64))


def localization_triangle_angles_deg(ref_positions: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref_positions, dtype=np.float64)
    angles: list[float] = []
    for vertex_idx in range(3):
        p = ref[vertex_idx]
        q = ref[(vertex_idx + 1) % 3]
        r = ref[(vertex_idx + 2) % 3]
        v1 = q - p
        v2 = r - p
        n1 = max(float(np.linalg.norm(v1)), 1.0e-12)
        n2 = max(float(np.linalg.norm(v2)), 1.0e-12)
        cos_theta = float(np.dot(v1, v2) / (n1 * n2))
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos_theta))))
    return np.sort(np.asarray(angles, dtype=np.float64))


def localization_triangle_angles_deg_vertexwise(ref_positions: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref_positions, dtype=np.float64)
    angles: list[float] = []
    for vertex_idx in range(3):
        p = ref[vertex_idx]
        other = [idx for idx in range(3) if idx != vertex_idx]
        q = ref[other[0]]
        r = ref[other[1]]
        v1 = q - p
        v2 = r - p
        n1 = max(float(np.linalg.norm(v1)), 1.0e-12)
        n2 = max(float(np.linalg.norm(v2)), 1.0e-12)
        cos_theta = float(np.dot(v1, v2) / (n1 * n2))
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cos_theta))))
    return np.asarray(angles, dtype=np.float64)


def point_to_segment_distance(point_xy: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    p = np.asarray(point_xy, dtype=np.float64)
    a = np.asarray(seg_a, dtype=np.float64)
    b = np.asarray(seg_b, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def localization_source_region_metrics(source_position: np.ndarray, ref_positions: np.ndarray, cfg: LocalizationConfig | None = None) -> dict[str, float]:
    cfg_obj = cfg or LocalizationConfig()
    ref = np.asarray(ref_positions, dtype=np.float64)
    src = np.asarray(source_position, dtype=np.float64)
    edge_lengths = localization_triangle_edge_lengths(ref)
    mean_edge = max(float(np.mean(edge_lengths)), 1.0e-12)
    angles_vertex = localization_triangle_angles_deg_vertexwise(ref)
    min_idx = int(np.argmin(angles_vertex))
    max_angle = float(np.max(angles_vertex))
    min_angle = float(np.min(angles_vertex))
    opposite = [idx for idx in range(3) if idx != min_idx]
    opp_len = float(np.linalg.norm(ref[opposite[0]] - ref[opposite[1]]))
    area = localization_triangle_area(ref)
    altitude = float(2.0 * area / max(opp_len, 1.0e-12))
    vertex_dist = float(np.linalg.norm(src - ref[min_idx]))
    opposite_edge_dist = point_to_segment_distance(src, ref[opposite[0]], ref[opposite[1]])
    adjacent_edge_dist_a = point_to_segment_distance(src, ref[min_idx], ref[opposite[0]])
    adjacent_edge_dist_b = point_to_segment_distance(src, ref[min_idx], ref[opposite[1]])
    long_edge_pref_ratio = float(min(adjacent_edge_dist_a, adjacent_edge_dist_b) / max(opposite_edge_dist, 1.0e-12))
    return {
        "obtuse_triangle_flag": float(1.0 if max_angle > 90.0 else 0.0),
        "small_angle_flag": float(1.0 if min_angle < float(cfg_obj.small_angle_threshold_deg) else 0.0),
        "small_angle_vertex_dist_norm": float(vertex_dist / mean_edge),
        "small_angle_opposite_edge_dist_norm": float(opposite_edge_dist / max(altitude, 1.0e-12)),
        "small_angle_long_edge_pref_ratio": float(long_edge_pref_ratio),
    }


def localization_inside_convex_hull(point_xy: np.ndarray, ref_positions: np.ndarray) -> bool:
    tri = np.asarray(ref_positions, dtype=np.float64)
    p = np.asarray(point_xy, dtype=np.float64)
    a, b, c = tri
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(float(den)) <= 1.0e-12:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    return bool((u >= 0.0) and (v >= 0.0) and (u + v <= 1.0))


def localization_tdoa_jacobian(source_position: np.ndarray, ref_positions: np.ndarray) -> np.ndarray:
    src = np.asarray(source_position, dtype=np.float64)
    refs = np.asarray(ref_positions, dtype=np.float64)
    rows: list[np.ndarray] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        di = np.linalg.norm(src - refs[i])
        dj = np.linalg.norm(src - refs[j])
        gi = (src - refs[i]) / max(float(di), 1.0e-12)
        gj = (src - refs[j]) / max(float(dj), 1.0e-12)
        rows.append(gi - gj)
    return np.asarray(rows, dtype=np.float64)


def localization_geometry_metrics(source_position: np.ndarray, ref_positions: np.ndarray, cfg: LocalizationConfig | None = None) -> dict[str, float]:
    cfg_obj = cfg or LocalizationConfig()
    ref = np.asarray(ref_positions, dtype=np.float64)
    src = np.asarray(source_position, dtype=np.float64)
    edge_lengths = localization_triangle_edge_lengths(ref)
    angle_deg = localization_triangle_angles_deg(ref)
    region_metrics = localization_source_region_metrics(src, ref, cfg=cfg)
    area = localization_triangle_area(ref)
    centroid = np.mean(ref, axis=0)
    mean_edge = max(float(np.mean(edge_lengths)), 1.0e-12)
    src_ref_dists = np.linalg.norm(ref - src[None, :], axis=1)
    jac = localization_tdoa_jacobian(src, ref)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
    cond = float("inf") if sigma_min <= 1.0e-12 else float(singular_values[0] / sigma_min)
    return {
        "triangle_area": float(area),
        "edge_len_min": float(edge_lengths[0]),
        "edge_len_mid": float(edge_lengths[1]),
        "edge_len_max": float(edge_lengths[2]),
        "edge_len_ratio": float(edge_lengths[2] / max(edge_lengths[0], 1.0e-12)),
        "triangle_angle_min_deg": float(angle_deg[0]),
        "triangle_angle_mid_deg": float(angle_deg[1]),
        "triangle_angle_max_deg": float(angle_deg[2]),
        "obtuse_triangle_flag": float(region_metrics["obtuse_triangle_flag"]),
        "small_angle_flag": float(region_metrics["small_angle_flag"]),
        "small_angle_vertex_dist_norm": float(region_metrics["small_angle_vertex_dist_norm"]),
        "small_angle_opposite_edge_dist_norm": float(region_metrics["small_angle_opposite_edge_dist_norm"]),
        "small_angle_long_edge_pref_ratio": float(region_metrics["small_angle_long_edge_pref_ratio"]),
        "centroid_source_dist": float(np.linalg.norm(src - centroid)),
        "centroid_source_dist_norm": float(np.linalg.norm(src - centroid) / mean_edge),
        "inside_convex_hull": float(1.0 if localization_inside_convex_hull(src, ref) else 0.0),
        "min_source_ref_dist": float(np.min(src_ref_dists)),
        "near_ref_inside_trigger_flag": float(
            1.0
            if float(np.min(src_ref_dists)) < max(float(cfg_obj.near_ref_inside_threshold_m), 0.0)
            and float(cfg_obj.near_ref_inside_threshold_m) > 0.0
            else 0.0
        ),
        "max_source_ref_dist": float(np.max(src_ref_dists)),
        "jacobian_condition": float(cond),
        "jacobian_sigma_min": float(sigma_min),
    }


def localization_geometry_metric_vector(metric_dict: dict[str, float]) -> np.ndarray:
    return np.asarray([float(metric_dict[name]) for name in LOCALIZATION_GEOMETRY_METRIC_NAMES], dtype=np.float32)


def localization_geometry_passes_filter(metric_dict: dict[str, float], cfg: LocalizationConfig) -> bool:
    mode = str(cfg.geometry_filter_mode).lower()
    if mode == "none":
        return True
    if mode != "stable":
        raise KeyError(f"Unknown geometry_filter_mode: {cfg.geometry_filter_mode}")
    basic_ok = bool(
        float(metric_dict["triangle_area"]) >= float(cfg.min_triangle_area)
        and float(metric_dict["jacobian_condition"]) <= float(cfg.max_jacobian_condition)
        and float(metric_dict["triangle_angle_max_deg"]) <= float(cfg.max_triangle_angle_deg)
    )
    if not basic_ok:
        return False
    if bool(cfg.require_source_inside_if_obtuse) and float(metric_dict["obtuse_triangle_flag"]) > 0.5:
        if float(metric_dict["inside_convex_hull"]) < 0.5:
            return False
    if float(metric_dict.get("near_ref_inside_trigger_flag", 0.0)) > 0.5:
        if float(metric_dict["inside_convex_hull"]) < 0.5:
            return False
    if float(metric_dict["small_angle_flag"]) > 0.5:
        if float(metric_dict["small_angle_vertex_dist_norm"]) < float(cfg.small_angle_vertex_clearance_ratio):
            return False
        if float(metric_dict["small_angle_opposite_edge_dist_norm"]) < float(cfg.small_angle_opposite_edge_clearance_ratio):
            return False
        if float(metric_dict["small_angle_long_edge_pref_ratio"]) > float(cfg.small_angle_prefer_long_edges_ratio):
            return False
    return True


def sample_ref_positions_2d(rng: np.random.Generator, room_size: tuple[float, float], margin: float, min_pair: float) -> np.ndarray:
    width, height = float(room_size[0]), float(room_size[1])
    for _ in range(200):
        ref = rng.uniform([margin, margin], [width - margin, height - margin], size=(3, 2))
        if analytic_ref_triangle_signature(ref)[-1] <= 0.03:
            continue
        ok = True
        for i in range(3):
            for j in range(i + 1, 3):
                if float(np.linalg.norm(ref[i] - ref[j])) < float(min_pair):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return ref.astype(np.float32)
    raise RuntimeError("Failed to sample reference triangle.")


def sample_source_position_2d(rng: np.random.Generator, room_size: tuple[float, float], margin: float, ref_positions: np.ndarray, min_ref_dist: float) -> np.ndarray:
    width, height = float(room_size[0]), float(room_size[1])
    for _ in range(200):
        src = rng.uniform([margin, margin], [width - margin, height - margin], size=(2,))
        if float(np.min(np.linalg.norm(ref_positions - src[None, :], axis=1))) >= float(min_ref_dist):
            return src.astype(np.float32)
    raise RuntimeError("Failed to sample source position.")


def localization_profile_id(profile_name: str) -> int:
    lowered = str(profile_name).lower()
    if lowered == "anechoic":
        return 0
    if lowered == "single_reflection":
        return 1
    raise KeyError(f"Unknown localization profile: {profile_name}")


def sample_localization_profile_name(rng: np.random.Generator, cfg: LocalizationConfig) -> str:
    base_profile = str(cfg.profile).lower()
    if base_profile == "anechoic":
        return "anechoic"
    if base_profile != "single_reflection":
        raise KeyError(f"Unsupported localization profile: {cfg.profile}")
    mix_frac = float(np.clip(float(cfg.reflection_profile_mix_anechoic_frac), 0.0, 1.0))
    if mix_frac > 0.0 and float(rng.random()) < mix_frac:
        return "anechoic"
    return "single_reflection"


def sample_localization_example(
    rng: np.random.Generator,
    cfg: LocalizationConfig,
    compute_covariance: bool = True,
) -> dict[str, np.ndarray | int | dict[str, float] | None]:
    for attempt_idx in range(2000):
        ref = sample_ref_positions_2d(rng, cfg.plane_room_size, cfg.ref_margin, cfg.min_ref_pair_dist)
        src = sample_source_position_2d(rng, cfg.plane_room_size, cfg.source_margin, ref, cfg.min_source_ref_dist)
        geom = localization_geometry_metrics(src, ref, cfg=cfg)
        if not localization_geometry_passes_filter(geom, cfg):
            continue
        noise_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        local_rng = np.random.default_rng(noise_seed)
        noise = local_rng.standard_normal(int(cfg.signal_len)).astype(np.float32)
        sample_profile = sample_localization_profile_name(rng, cfg)
        window = np.zeros((3, int(cfg.ref_window_len)), dtype=np.float32)
        for mic_idx in range(3):
            rir = rir_for_profile(src, ref[mic_idx], cfg, profile_name=sample_profile)
            window[mic_idx] = convolve_and_crop(noise, rir, int(cfg.ref_window_len))
        true_tdoa = true_tdoa_triplet_from_geometry(src, ref, cfg.fs, cfg.c).astype(np.float32)
        return {
            "ref_positions": ref.astype(np.float32),
            "source_position": src.astype(np.float32),
            "x_ref": window.astype(np.float32),
            "gcc_phat": compute_gcc_triplet(window, out_len=int(cfg.gcc_len), phat=True).astype(np.float32),
            "reference_covariance_features": (
                compute_reference_covariance_features(window, n_fft=int(cfg.psd_nfft)).astype(np.float32)
                if bool(compute_covariance)
                else None
            ),
            "source_seed": int(noise_seed),
            "geometry_metrics": geom,
            "true_tdoa": true_tdoa,
            "sample_profile_name": sample_profile,
            "sample_profile_id": int(localization_profile_id(sample_profile)),
            "sample_attempts": int(attempt_idx + 1),
        }
    raise RuntimeError("Failed to sample localization example under the current geometry filter.")


def split_train_val_test(n: int, val_frac: float, test_frac: float, seed: int) -> dict[str, np.ndarray]:
    if float(val_frac) + float(test_frac) >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.")
    idx = np.arange(int(n), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    test_count = max(1, int(round(int(n) * float(test_frac))))
    val_count = max(1, int(round(int(n) * float(val_frac))))
    train_count = int(n) - test_count - val_count
    if train_count < 1:
        raise RuntimeError("Degenerate split.")
    return {
        "train": np.sort(idx[:train_count]).astype(np.int64),
        "val": np.sort(idx[train_count : train_count + val_count]).astype(np.int64),
        "test": np.sort(idx[train_count + val_count :]).astype(np.int64),
    }


def rir_for_profile(tx_pos: np.ndarray, rx_pos: np.ndarray, cfg: LocalizationConfig, profile_name: str | None = None) -> np.ndarray:
    resolved_profile = str(profile_name or cfg.profile).lower()
    air_alpha = float(cfg.air_attenuation_alpha_per_m) if bool(cfg.air_attenuation_enabled) else 0.0
    if resolved_profile == "anechoic":
        return direct_path_rir(
            tx_pos,
            rx_pos,
            fs=int(cfg.fs),
            c=float(cfg.c),
            rir_len=int(cfg.rir_len),
            amplitude_power=float(cfg.amplitude_power),
            air_alpha_per_m=air_alpha,
        )
    if resolved_profile != "single_reflection":
        raise KeyError(f"Unsupported localization RIR profile: {resolved_profile}")
    return one_reflection_rir(
        tx_pos,
        rx_pos,
        room_size=np.asarray(cfg.plane_room_size, dtype=np.float32),
        fs=int(cfg.fs),
        c=float(cfg.c),
        rir_len=int(cfg.rir_len),
        reflection_gain=float(cfg.reflection_gain),
        amplitude_power=float(cfg.amplitude_power),
        air_alpha_per_m=air_alpha,
    )


def build_localization_dataset(output_h5: Path | str, cfg: LocalizationConfig) -> dict[str, Any]:
    path = Path(output_h5)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.seed))
    n_bins = int(cfg.psd_nfft // 2 + 1)
    ref_positions = np.zeros((cfg.num_samples, 3, 2), dtype=np.float32)
    source_positions = np.zeros((cfg.num_samples, 2), dtype=np.float32)
    x_ref = np.zeros((cfg.num_samples, 3, cfg.ref_window_len), dtype=np.float32)
    gcc = np.zeros((cfg.num_samples, 3, cfg.gcc_len), dtype=np.float32)
    cov = np.zeros((cfg.num_samples, 9, n_bins), dtype=np.float32)
    source_seeds = np.zeros((cfg.num_samples,), dtype=np.int64)
    true_tdoa = np.zeros((cfg.num_samples, 3), dtype=np.float32)
    geometry_metrics = np.zeros((cfg.num_samples, len(LOCALIZATION_GEOMETRY_METRIC_NAMES)), dtype=np.float32)
    profile_ids = np.zeros((cfg.num_samples,), dtype=np.int64)
    accepted = 0
    attempts = 0
    max_attempts = max(int(cfg.num_samples) * 200, 2000)
    while accepted < int(cfg.num_samples):
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to build localization dataset with geometry_filter_mode={cfg.geometry_filter_mode!r} "
                f"after {attempts} attempts for {cfg.num_samples} accepted samples."
            )
        example = sample_localization_example(rng, cfg)
        attempts += int(example.get("sample_attempts", 1))
        ref_positions[accepted] = np.asarray(example["ref_positions"], dtype=np.float32)
        source_positions[accepted] = np.asarray(example["source_position"], dtype=np.float32)
        x_ref[accepted] = np.asarray(example["x_ref"], dtype=np.float32)
        gcc[accepted] = np.asarray(example["gcc_phat"], dtype=np.float32)
        cov[accepted] = np.asarray(example["reference_covariance_features"], dtype=np.float32)
        source_seeds[accepted] = int(example["source_seed"])
        true_tdoa[accepted] = np.asarray(example["true_tdoa"], dtype=np.float32)
        geometry_metrics[accepted] = localization_geometry_metric_vector(example["geometry_metrics"])
        profile_ids[accepted] = int(example.get("sample_profile_id", localization_profile_id(cfg.profile)))
        accepted += 1
    iid_split = split_train_val_test(int(cfg.num_samples), val_frac=0.15, test_frac=0.15, seed=int(cfg.seed))
    geom_split = build_geometry_holdout_split(ref_positions, seed=int(cfg.seed), holdout_frac=0.2)
    with h5py.File(str(path), "w") as h5:
        h5.attrs["config_json"] = json.dumps(asdict(cfg), ensure_ascii=False)
        h5.attrs["geometry_metric_names_json"] = json.dumps(list(LOCALIZATION_GEOMETRY_METRIC_NAMES), ensure_ascii=False)
        h5.attrs["geometry_filter_mode"] = str(cfg.geometry_filter_mode)
        h5.attrs["min_triangle_area"] = float(cfg.min_triangle_area)
        h5.attrs["max_jacobian_condition"] = float(cfg.max_jacobian_condition)
        h5.attrs["max_triangle_angle_deg"] = float(cfg.max_triangle_angle_deg)
        h5.attrs["near_ref_inside_threshold_m"] = float(cfg.near_ref_inside_threshold_m)
        h5.attrs["audit_rule_version"] = str(cfg.audit_rule_version)
        h5.attrs["source_region_rule_version"] = str(cfg.source_region_rule_version)
        h5.attrs["rir_model"] = str(cfg.rir_model)
        h5.attrs["air_attenuation_enabled"] = bool(cfg.air_attenuation_enabled)
        h5.attrs["air_attenuation_alpha_per_m"] = float(cfg.air_attenuation_alpha_per_m)
        h5.attrs["reflection_profile_mix_json"] = json.dumps(
            {
                "single_reflection": float(max(0.0, 1.0 - float(cfg.reflection_profile_mix_anechoic_frac)))
                if str(cfg.profile).lower() == "single_reflection"
                else 0.0,
                "anechoic": float(cfg.reflection_profile_mix_anechoic_frac)
                if str(cfg.profile).lower() == "single_reflection"
                else 1.0,
            },
            ensure_ascii=False,
        )
        h5.attrs["window_preset"] = str(cfg.window_preset)
        h5.attrs["profile_id_map_json"] = json.dumps({"anechoic": 0, "single_reflection": 1}, ensure_ascii=False)
        raw = h5.create_group("raw")
        raw.create_dataset("ref_positions", data=ref_positions)
        raw.create_dataset("source_position", data=source_positions)
        raw.create_dataset("x_ref", data=x_ref)
        raw.create_dataset("gcc_phat", data=gcc)
        raw.create_dataset("reference_covariance_features", data=cov)
        raw.create_dataset("source_seed", data=source_seeds)
        raw.create_dataset("true_tdoa", data=true_tdoa)
        raw.create_dataset("profile_id", data=profile_ids)
        if bool(cfg.store_geometry_metrics):
            raw.create_dataset("geometry_metrics", data=geometry_metrics)
        splits = h5.create_group("splits")
        for name, arr in {**iid_split, **geom_split}.items():
            splits.create_dataset(name, data=np.asarray(arr, dtype=np.int64))
    summary = {
        "h5_path": str(path),
        "profile": cfg.profile,
        "num_samples": int(cfg.num_samples),
        "sampling_attempts": int(attempts),
        "acceptance_rate": float(int(cfg.num_samples) / max(int(attempts), 1)),
        "geometry_filter_mode": str(cfg.geometry_filter_mode),
        "min_triangle_area": float(cfg.min_triangle_area),
        "max_jacobian_condition": float(cfg.max_jacobian_condition),
        "max_triangle_angle_deg": float(cfg.max_triangle_angle_deg),
        "near_ref_inside_threshold_m": float(cfg.near_ref_inside_threshold_m),
        "audit_rule_version": str(cfg.audit_rule_version),
        "source_region_rule_version": str(cfg.source_region_rule_version),
        "rir_model": str(cfg.rir_model),
        "air_attenuation_enabled": bool(cfg.air_attenuation_enabled),
        "air_attenuation_alpha_per_m": float(cfg.air_attenuation_alpha_per_m),
        "reflection_profile_mix": {
            "single_reflection": float(max(0.0, 1.0 - float(cfg.reflection_profile_mix_anechoic_frac)))
            if str(cfg.profile).lower() == "single_reflection"
            else 0.0,
            "anechoic": float(cfg.reflection_profile_mix_anechoic_frac)
            if str(cfg.profile).lower() == "single_reflection"
            else 1.0,
        },
        "window_preset": str(cfg.window_preset),
        "profile_counts": {
            "anechoic": int(np.sum(profile_ids == 0)),
            "single_reflection": int(np.sum(profile_ids == 1)),
        },
        "iid_train": int(iid_split["train"].size),
        "iid_val": int(iid_split["val"].size),
        "iid_test": int(iid_split["test"].size),
        "geom_train": int(geom_split["geom_train"].size),
        "geom_val": int(geom_split["geom_val"].size),
        "geom_test": int(geom_split["geom_test"].size),
    }
    save_json(path.with_suffix(".manifest.json"), summary)
    return summary


def load_localization_dataset(h5_path: Path | str) -> tuple[LocalizationConfig, dict[str, np.ndarray], dict[str, np.ndarray]]:
    with h5py.File(str(h5_path), "r") as h5:
        cfg = LocalizationConfig(**json.loads(h5.attrs["config_json"]))
        raw = h5["raw"]
        splits = h5["splits"]
        if "geometry_metric_names_json" in h5.attrs:
            geometry_metric_names = tuple(json.loads(h5.attrs["geometry_metric_names_json"]))
        else:
            geometry_metric_names = LOCALIZATION_GEOMETRY_METRIC_NAMES
        data = {
            "ref_positions": np.asarray(raw["ref_positions"], dtype=np.float32),
            "source_position": np.asarray(raw["source_position"], dtype=np.float32),
            "x_ref": np.asarray(raw["x_ref"], dtype=np.float32),
            "gcc_phat": np.asarray(raw["gcc_phat"], dtype=np.float32),
            "reference_covariance_features": np.asarray(raw["reference_covariance_features"], dtype=np.float32),
            "true_tdoa": np.asarray(raw["true_tdoa"], dtype=np.float32)
            if "true_tdoa" in raw
            else np.zeros((raw["ref_positions"].shape[0], 3), dtype=np.float32),
            "profile_id": np.asarray(raw["profile_id"], dtype=np.int64)
            if "profile_id" in raw
            else np.full((raw["ref_positions"].shape[0],), localization_profile_id(cfg.profile), dtype=np.int64),
            "geometry_metric_names": np.asarray(geometry_metric_names, dtype=object),
        }
        if "geometry_metrics" in raw:
            data["geometry_metrics"] = np.asarray(raw["geometry_metrics"], dtype=np.float32)
        else:
            geom = np.zeros((data["ref_positions"].shape[0], len(LOCALIZATION_GEOMETRY_METRIC_NAMES)), dtype=np.float32)
            true_tdoa = np.zeros((data["ref_positions"].shape[0], 3), dtype=np.float32)
            for idx in range(data["ref_positions"].shape[0]):
                geom[idx] = localization_geometry_metric_vector(
                    localization_geometry_metrics(data["source_position"][idx], data["ref_positions"][idx], cfg=cfg)
                )
                true_tdoa[idx] = true_tdoa_triplet_from_geometry(
                    data["source_position"][idx], data["ref_positions"][idx], cfg.fs, cfg.c
                ).astype(np.float32)
            data["geometry_metrics"] = geom
            data["true_tdoa"] = true_tdoa
        split_map = {name: np.asarray(splits[name], dtype=np.int64) for name in splits.keys()}
    return cfg, data, split_map


def peak_index_subsample(vec: np.ndarray) -> float:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    idx = int(np.argmax(arr))
    if idx <= 0 or idx >= arr.size - 1:
        return float(idx)
    left = float(arr[idx - 1])
    center = float(arr[idx])
    right = float(arr[idx + 1])
    denom = left - 2.0 * center + right
    if abs(denom) <= 1.0e-12:
        return float(idx)
    return float(idx) + 0.5 * (left - right) / denom


def estimate_tdoa_from_gcc_triplet(gcc_triplet: np.ndarray) -> np.ndarray:
    arr = np.asarray(gcc_triplet, dtype=np.float64)
    center = float((arr.shape[-1] - 1) // 2)
    return np.array([peak_index_subsample(row) - center for row in arr], dtype=np.float64)


def estimate_source_position_from_tdoa(lag_samples: np.ndarray, ref_positions: np.ndarray, room_size: tuple[float, float], fs: int, c: float) -> np.ndarray:
    pair_indices = [(0, 1), (0, 2), (1, 2)]
    delta_dist = np.asarray(lag_samples, dtype=np.float64) / float(fs) * float(c)
    refs = np.asarray(ref_positions, dtype=np.float64)
    room = np.asarray(room_size, dtype=np.float64)

    def residual(src_xy: np.ndarray) -> np.ndarray:
        src = np.asarray(src_xy, dtype=np.float64)
        return np.asarray(
            [
                np.linalg.norm(src - refs[i]) - np.linalg.norm(src - refs[j]) - delta
                for (i, j), delta in zip(pair_indices, delta_dist)
            ],
            dtype=np.float64,
        )

    grid_x = np.linspace(0.15, float(room[0]) - 0.15, 31)
    grid_y = np.linspace(0.15, float(room[1]) - 0.15, 31)
    best = np.array([room[0] * 0.5, room[1] * 0.5], dtype=np.float64)
    best_cost = float("inf")
    for x in grid_x:
        for y in grid_y:
            r = residual(np.array([x, y], dtype=np.float64))
            cost = float(np.sum(r * r))
            if cost < best_cost:
                best_cost = cost
                best = np.array([x, y], dtype=np.float64)
    result = least_squares(
        residual,
        x0=best,
        bounds=([0.10, 0.10], [float(room[0]) - 0.10, float(room[1]) - 0.10]),
        method="trf",
        max_nfev=200,
    )
    return np.asarray(result.x, dtype=np.float32)


def estimate_source_position_analytic(gcc_triplet: np.ndarray, ref_positions: np.ndarray, room_size: tuple[float, float], fs: int, c: float) -> np.ndarray:
    lag_samples = estimate_tdoa_from_gcc_triplet(gcc_triplet)
    return estimate_source_position_from_tdoa(lag_samples, ref_positions, room_size, fs, c)


def localization_error_stats(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64), axis=1)
    return {
        "median_m": float(np.median(err)),
        "p90_m": float(np.quantile(err, 0.90)),
        "mean_m": float(np.mean(err)),
        "max_m": float(np.max(err)),
    }


def localization_error_vector(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64), axis=1)


def true_tdoa_triplet_from_geometry(source_position: np.ndarray, ref_positions: np.ndarray, fs: int, c: float) -> np.ndarray:
    src = np.asarray(source_position, dtype=np.float64)
    refs = np.asarray(ref_positions, dtype=np.float64)
    pair_indices = [(0, 1), (0, 2), (1, 2)]
    delays_sec = np.asarray(
        [
            (np.linalg.norm(src - refs[i]) - np.linalg.norm(src - refs[j])) / float(c)
            for i, j in pair_indices
        ],
        dtype=np.float64,
    )
    return delays_sec * float(fs)


def delay_error_stats(est_lags: np.ndarray, true_lags: np.ndarray, fs: int, c: float) -> dict[str, Any]:
    est = np.asarray(est_lags, dtype=np.float64)
    true = np.asarray(true_lags, dtype=np.float64)
    if est.shape != true.shape:
        raise ValueError(f"Mismatched delay shapes: {est.shape} vs {true.shape}")
    pair_labels = ("01", "02", "12")
    out: dict[str, Any] = {}
    all_abs = np.abs(est - true).reshape(-1)
    out["overall"] = {
        "median_samples": float(np.median(all_abs)),
        "p90_samples": float(np.quantile(all_abs, 0.90)),
        "mean_samples": float(np.mean(all_abs)),
        "max_samples": float(np.max(all_abs)),
        "median_m": float(np.median(all_abs) / float(fs) * float(c)),
        "p90_m": float(np.quantile(all_abs, 0.90) / float(fs) * float(c)),
        "mean_m": float(np.mean(all_abs) / float(fs) * float(c)),
        "max_m": float(np.max(all_abs) / float(fs) * float(c)),
    }
    for pair_idx, pair_label in enumerate(pair_labels):
        pair_abs = np.abs(est[:, pair_idx] - true[:, pair_idx])
        out[pair_label] = {
            "median_samples": float(np.median(pair_abs)),
            "p90_samples": float(np.quantile(pair_abs, 0.90)),
            "mean_samples": float(np.mean(pair_abs)),
            "max_samples": float(np.max(pair_abs)),
            "median_m": float(np.median(pair_abs) / float(fs) * float(c)),
            "p90_m": float(np.quantile(pair_abs, 0.90) / float(fs) * float(c)),
            "mean_m": float(np.mean(pair_abs) / float(fs) * float(c)),
            "max_m": float(np.max(pair_abs) / float(fs) * float(c)),
        }
    return out


def _build_localization_features(data: dict[str, np.ndarray], kind: str) -> np.ndarray:
    ref_flat = data["ref_positions"].reshape(data["ref_positions"].shape[0], -1)
    if kind == "tdoa":
        feat = np.concatenate([ref_flat, np.asarray(data["true_tdoa"], dtype=np.float32)], axis=1)
    elif kind == "gcc":
        feat = np.concatenate([ref_flat, data["gcc_phat"].reshape(data["gcc_phat"].shape[0], -1)], axis=1)
    elif kind == "cov":
        feat = np.concatenate([ref_flat, data["reference_covariance_features"].reshape(data["reference_covariance_features"].shape[0], -1)], axis=1)
    elif kind == "gcc_cov":
        feat = np.concatenate(
            [
                ref_flat,
                data["gcc_phat"].reshape(data["gcc_phat"].shape[0], -1),
                data["reference_covariance_features"].reshape(data["reference_covariance_features"].shape[0], -1),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown localization feature kind: {kind}")
    return feat.astype(np.float32)


def localization_model_id(feature_kind: str, model_kind: str) -> str:
    if str(model_kind) == "mlp":
        return str(feature_kind)
    return f"{feature_kind}_{model_kind}"


def localization_candidate_specs(candidate_mode: str) -> list[tuple[str, str]]:
    mode = str(candidate_mode).lower()
    base = [("gcc", "mlp"), ("cov", "mlp")]
    if mode == "base":
        return list(base)
    if mode == "fusion":
        return list(base) + [("gcc_cov", "mlp")]
    if mode == "residual":
        return list(base) + [("gcc_cov", "mlp"), ("gcc_cov", "resmlp")]
    if mode == "stable":
        return [("tdoa", "mlp"), ("tdoa", "resmlp"), ("gcc", "mlp"), ("gcc_cov", "mlp")]
    if mode == "auto":
        return [("tdoa", "mlp"), ("tdoa", "resmlp"), ("gcc", "mlp"), ("gcc_cov", "mlp")]
    raise KeyError(f"Unknown localization candidate mode: {candidate_mode}")


def infer_localization_window_preset(cfg: LocalizationConfig) -> str:
    for preset, lengths in LOCALIZATION_WINDOW_PRESETS.items():
        if int(cfg.signal_len) == int(lengths["signal_len"]) and int(cfg.ref_window_len) == int(lengths["ref_window_len"]):
            return preset
    return "CUSTOM"


def evaluate_stage01_analytic_gate(analytic_gcc_phat: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    checks = {
        "iid_test_median": float(analytic_gcc_phat["iid_test"]["median_m"]) < float(thresholds["analytic_iid_median_max"]),
        "iid_test_p90": float(analytic_gcc_phat["iid_test"]["p90_m"]) < float(thresholds["analytic_iid_p90_max"]),
        "geom_test_median": float(analytic_gcc_phat["geom_test"]["median_m"]) < float(thresholds["analytic_geom_median_max"]),
        "geom_test_p90": float(analytic_gcc_phat["geom_test"]["p90_m"]) < float(thresholds["analytic_geom_p90_max"]),
    }
    return {"checks": checks, "passed": bool(all(bool(v) for v in checks.values()))}


def _analytic_gcc_triplet_for_method(data: dict[str, np.ndarray], sample_index: int, cfg: LocalizationConfig, method_name: str) -> np.ndarray:
    method = str(method_name)
    if method == "gcc_phat":
        return np.asarray(data["gcc_phat"][int(sample_index)], dtype=np.float32)
    if method == "plain_gcc":
        return compute_gcc_triplet(data["x_ref"][int(sample_index)], out_len=int(cfg.gcc_len), phat=False)
    raise KeyError(f"Unknown analytic localization method: {method_name}")


def _evaluate_analytic_localization_method(
    cfg: LocalizationConfig,
    data: dict[str, np.ndarray],
    idx: np.ndarray,
    split_name: str,
    method_name: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    pred = np.zeros((int(idx.size), 2), dtype=np.float32)
    est_lags = np.zeros((int(idx.size), 3), dtype=np.float64)
    true_lags = np.zeros((int(idx.size), 3), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    pair_labels = ("01", "02", "12")
    for row_idx, sample_index in enumerate(np.asarray(idx, dtype=np.int64)):
        gcc_triplet = _analytic_gcc_triplet_for_method(data, int(sample_index), cfg, method_name)
        pred[row_idx] = estimate_source_position_analytic(
            gcc_triplet,
            data["ref_positions"][int(sample_index)],
            cfg.plane_room_size,
            cfg.fs,
            cfg.c,
        )
        est_lags[row_idx] = estimate_tdoa_from_gcc_triplet(gcc_triplet)
        true_lags[row_idx] = true_tdoa_triplet_from_geometry(
            data["source_position"][int(sample_index)],
            data["ref_positions"][int(sample_index)],
            cfg.fs,
            cfg.c,
        )
        position_error_m = float(np.linalg.norm(pred[row_idx].astype(np.float64) - data["source_position"][int(sample_index)].astype(np.float64)))
        row = {
            "split": str(split_name),
            "method": str(method_name),
            "sample_index": int(sample_index),
            "position_error_m": position_error_m,
        }
        for pair_idx, pair_label in enumerate(pair_labels):
            abs_err_samples = abs(float(est_lags[row_idx, pair_idx] - true_lags[row_idx, pair_idx]))
            row[f"est_tdoa_samples_{pair_label}"] = float(est_lags[row_idx, pair_idx])
            row[f"true_tdoa_samples_{pair_label}"] = float(true_lags[row_idx, pair_idx])
            row[f"abs_tdoa_error_samples_{pair_label}"] = abs_err_samples
            row[f"abs_tdoa_error_m_{pair_label}"] = float(abs_err_samples / float(cfg.fs) * float(cfg.c))
        rows.append(row)
    stats = localization_error_stats(pred, data["source_position"][idx])
    stats["delay_error"] = delay_error_stats(est_lags, true_lags, cfg.fs, cfg.c)
    return stats, pred, est_lags, true_lags, rows


def _plot_localization_prediction_set(
    output_path: Path,
    cfg: LocalizationConfig,
    true_pos: np.ndarray,
    pred_sets: list[tuple[str, np.ndarray, str]],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.scatter(true_pos[:, 0], true_pos[:, 1], s=14, alpha=0.50, label="True", color="tab:blue")
    for label, pred, color in pred_sets:
        ax.scatter(pred[:, 0], pred[:, 1], s=13, alpha=0.40, label=label, color=color)
    ax.set_xlim(0.0, float(cfg.plane_room_size[0]))
    ax.set_ylim(0.0, float(cfg.plane_room_size[1]))
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_localization_sample(
    output_path: Path,
    cfg: LocalizationConfig,
    refs: np.ndarray,
    true_pos: np.ndarray,
    pred_pos: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    ax.scatter(refs[:, 0], refs[:, 1], s=90, color="tab:green", label="Refs")
    ax.scatter(float(true_pos[0]), float(true_pos[1]), s=90, color="tab:blue", label="True")
    ax.scatter(float(pred_pos[0]), float(pred_pos[1]), s=90, color="tab:orange", label="Pred")
    ax.plot([float(true_pos[0]), float(pred_pos[0])], [float(true_pos[1]), float(pred_pos[1])], linestyle="--", color="tab:red", alpha=0.6)
    ax.set_xlim(0.0, float(cfg.plane_room_size[0]))
    ax.set_ylim(0.0, float(cfg.plane_room_size[1]))
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def evaluate_localization_analytic_only(
    cfg: LocalizationConfig,
    data: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    output_dir: Path | str,
    stage_id: str,
    level: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    analytic_plain_gcc: dict[str, Any] = {}
    analytic_gcc_phat: dict[str, Any] = {}
    analytic_predictions: dict[str, dict[str, np.ndarray]] = {"plain_gcc": {}, "gcc_phat": {}}
    sample_rows: list[dict[str, Any]] = []
    for method_name, target in [("plain_gcc", analytic_plain_gcc), ("gcc_phat", analytic_gcc_phat)]:
        for split_name, idx in [("iid_test", splits["test"]), ("geom_test", splits["geom_test"])]:
            stats, pred, _, _, rows = _evaluate_analytic_localization_method(cfg, data, idx, split_name, method_name)
            target[split_name] = stats
            analytic_predictions[method_name][split_name] = pred
            sample_rows.extend(rows)
    save_history_csv(out_dir / "analytic_sample_errors.csv", sample_rows)
    for split_name, idx in [("iid_test", splits["test"]), ("geom_test", splits["geom_test"])]:
        _plot_localization_prediction_set(
            plots_dir / f"analytic_compare_{split_name}.png",
            cfg,
            data["source_position"][idx],
            [
                ("plain GCC", analytic_predictions["plain_gcc"][split_name], "tab:purple"),
                ("GCC-PHAT", analytic_predictions["gcc_phat"][split_name], "tab:orange"),
            ],
            title=f"Analytic compare | {split_name}",
        )
    phat_errors = localization_error_vector(analytic_predictions["gcc_phat"]["iid_test"], data["source_position"][splits["test"]])
    success_idx = int(np.argmin(phat_errors))
    failure_idx = int(np.argmax(phat_errors))
    _plot_localization_sample(
        plots_dir / "analytic_gcc_phat_success.png",
        cfg,
        data["ref_positions"][splits["test"][success_idx]],
        data["source_position"][splits["test"][success_idx]],
        analytic_predictions["gcc_phat"]["iid_test"][success_idx],
        title=f"Analytic GCC-PHAT success | error={phat_errors[success_idx]:.4f} m",
    )
    _plot_localization_sample(
        plots_dir / "analytic_gcc_phat_failure.png",
        cfg,
        data["ref_positions"][splits["test"][failure_idx]],
        data["source_position"][splits["test"][failure_idx]],
        analytic_predictions["gcc_phat"]["iid_test"][failure_idx],
        title=f"Analytic GCC-PHAT failure | error={phat_errors[failure_idx]:.4f} m",
    )
    gate = evaluate_stage01_analytic_gate(analytic_gcc_phat, LOCALIZATION_THRESHOLDS[str(stage_id)]) if str(stage_id) == "01" else {"checks": {}, "passed": True}
    return {
        "level": level,
        "window_preset": infer_localization_window_preset(cfg),
        "window_config": {
            "signal_len": int(cfg.signal_len),
            "ref_window_len": int(cfg.ref_window_len),
            "fs": int(cfg.fs),
            "c": float(cfg.c),
        },
        "analytic_plain_gcc": analytic_plain_gcc,
        "analytic_gcc_phat": analytic_gcc_phat,
        "analytic": analytic_gcc_phat,
        "analytic_gate": gate,
        "analytic_sample_errors_csv": str(out_dir / "analytic_sample_errors.csv"),
    }


def _error_vector_stats(err: np.ndarray) -> dict[str, float]:
    arr = np.asarray(err, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"median_m": float("inf"), "p90_m": float("inf"), "mean_m": float("inf"), "max_m": float("inf")}
    return {
        "median_m": float(np.median(arr)),
        "p90_m": float(np.quantile(arr, 0.90)),
        "mean_m": float(np.mean(arr)),
        "max_m": float(np.max(arr)),
    }


def _metrics_from_geometry_array(geometry_metrics: np.ndarray, metric_name: str) -> np.ndarray:
    metric_idx = LOCALIZATION_GEOMETRY_METRIC_NAMES.index(str(metric_name))
    return np.asarray(geometry_metrics[:, metric_idx], dtype=np.float64)


def _write_sample_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_geometry_scatter(path: Path, x: np.ndarray, y: np.ndarray, x_label: str, title: str, log_x: bool = False, color_by_inside: np.ndarray | None = None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    if color_by_inside is None:
        ax.scatter(x, y, s=8, alpha=0.18, color="tab:orange")
    else:
        inside_mask = np.asarray(color_by_inside, dtype=np.float64) > 0.5
        ax.scatter(np.asarray(x)[~inside_mask], np.asarray(y)[~inside_mask], s=8, alpha=0.16, color="tab:orange", label="outside")
        ax.scatter(np.asarray(x)[inside_mask], np.asarray(y)[inside_mask], s=9, alpha=0.25, color="tab:blue", label="inside")
        ax.legend()
    if bool(log_x):
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("GCC-PHAT position error (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_threshold_heatmaps(path: Path, grid_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    area_values = sorted({float(row["area_min"]) for row in grid_rows})
    cond_values = sorted({float(row["cond_max"]) for row in grid_rows})
    retention = np.full((len(area_values), len(cond_values)), np.nan, dtype=np.float64)
    iid_p90 = np.full_like(retention, np.nan)
    geom_p90 = np.full_like(retention, np.nan)
    passed = np.full_like(retention, np.nan)
    area_to_idx = {val: idx for idx, val in enumerate(area_values)}
    cond_to_idx = {val: idx for idx, val in enumerate(cond_values)}
    for row in grid_rows:
        i = area_to_idx[float(row["area_min"])]
        j = cond_to_idx[float(row["cond_max"])]
        retention[i, j] = float(row["retention_overall"])
        iid_p90[i, j] = float(row["gcc_phat_iid_test_p90_m"])
        geom_p90[i, j] = float(row["gcc_phat_geom_test_p90_m"])
        passed[i, j] = 1.0 if bool(row["passed"]) else 0.0
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    for ax, mat, title in zip(
        axes,
        (retention, iid_p90, geom_p90, passed),
        ("Retention", "iid p90", "geom p90", "Passed"),
    ):
        im = ax.imshow(mat, origin="lower", aspect="auto")
        ax.set_xticks(np.arange(len(cond_values)))
        ax.set_xticklabels([str(int(v)) for v in cond_values])
        ax.set_yticks(np.arange(len(area_values)))
        ax.set_yticklabels([f"{v:.2f}" for v in area_values])
        ax.set_xlabel("cond_max")
        ax.set_ylabel("area_min")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_geometry_failure_examples(path: Path, ref_positions: np.ndarray, source_positions: np.ndarray, pred_positions: np.ndarray, errors: np.ndarray, triangle_area: np.ndarray, cond_j: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    worst_idx = np.argsort(np.asarray(errors, dtype=np.float64))[-4:]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, idx in zip(axes, worst_idx[::-1]):
        tri = np.asarray(ref_positions[int(idx)], dtype=np.float64)
        true_pos = np.asarray(source_positions[int(idx)], dtype=np.float64)
        pred_pos = np.asarray(pred_positions[int(idx)], dtype=np.float64)
        ax.scatter(tri[:, 0], tri[:, 1], s=80, color="tab:green", label="refs")
        ax.scatter(true_pos[0], true_pos[1], s=80, color="tab:blue", label="true")
        ax.scatter(pred_pos[0], pred_pos[1], s=80, color="tab:orange", label="pred")
        ax.plot([true_pos[0], pred_pos[0]], [true_pos[1], pred_pos[1]], linestyle="--", color="tab:red", alpha=0.55)
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 4.0)
        ax.set_title(
            f"err={float(errors[int(idx)]):.3f}m | area={float(triangle_area[int(idx)]):.3f} | cond={float(cond_j[int(idx)]):.1f}"
        )
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _select_stable_geometry_candidate(grid_rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any] | None:
    passing = [
        row
        for row in grid_rows
        if bool(row["passed"])
        and float(row["gcc_phat_iid_test_median_m"]) < float(thresholds["analytic_iid_median_max"])
        and float(row["gcc_phat_iid_test_p90_m"]) < float(thresholds["analytic_iid_p90_max"])
        and float(row["gcc_phat_geom_test_median_m"]) < float(thresholds["analytic_geom_median_max"])
        and float(row["gcc_phat_geom_test_p90_m"]) < float(thresholds["analytic_geom_p90_max"])
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            -float(row["retention_overall"]),
            float(row["gcc_phat_iid_test_p90_m"]) + float(row["gcc_phat_geom_test_p90_m"]),
            -float(row["area_min"]),
            float(row["cond_max"]),
        ),
    )


def run_localization_geometry_stability_audit(
    output_dir: Path | str,
    cfg: LocalizationConfig,
    num_samples: int = 100000,
    stage_id: str = "01",
    level: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    audit_cfg = LocalizationConfig(**asdict(cfg))
    audit_cfg.num_samples = int(num_samples)
    audit_cfg.geometry_filter_mode = "none"
    rng = np.random.default_rng(int(audit_cfg.seed))
    ref_positions = np.zeros((int(num_samples), 3, 2), dtype=np.float32)
    source_positions = np.zeros((int(num_samples), 2), dtype=np.float32)
    true_tdoa = np.zeros((int(num_samples), 3), dtype=np.float32)
    gcc_tdoa = np.zeros((int(num_samples), 3), dtype=np.float32)
    geometry_metrics = np.zeros((int(num_samples), len(LOCALIZATION_GEOMETRY_METRIC_NAMES)), dtype=np.float32)
    oracle_pred = np.zeros((int(num_samples), 2), dtype=np.float32)
    gcc_pred = np.zeros((int(num_samples), 2), dtype=np.float32)
    for idx in range(int(num_samples)):
        example = sample_localization_example(rng, audit_cfg, compute_covariance=False)
        ref = np.asarray(example["ref_positions"], dtype=np.float32)
        src = np.asarray(example["source_position"], dtype=np.float32)
        gcc_triplet = np.asarray(example["gcc_phat"], dtype=np.float32)
        lag_true = np.asarray(example["true_tdoa"], dtype=np.float32)
        lag_gcc = estimate_tdoa_from_gcc_triplet(gcc_triplet).astype(np.float32)
        ref_positions[idx] = ref
        source_positions[idx] = src
        true_tdoa[idx] = lag_true
        gcc_tdoa[idx] = lag_gcc
        geometry_metrics[idx] = localization_geometry_metric_vector(example["geometry_metrics"])
        oracle_pred[idx] = estimate_source_position_from_tdoa(lag_true, ref, audit_cfg.plane_room_size, audit_cfg.fs, audit_cfg.c)
        gcc_pred[idx] = estimate_source_position_from_tdoa(lag_gcc, ref, audit_cfg.plane_room_size, audit_cfg.fs, audit_cfg.c)
    iid_split = split_train_val_test(int(num_samples), val_frac=0.15, test_frac=0.15, seed=int(audit_cfg.seed))
    geom_split = build_geometry_holdout_split(ref_positions, seed=int(audit_cfg.seed), holdout_frac=0.2)
    iid_labels = np.full((int(num_samples),), "train", dtype=object)
    iid_labels[np.asarray(iid_split["val"], dtype=np.int64)] = "val"
    iid_labels[np.asarray(iid_split["test"], dtype=np.int64)] = "test"
    geom_labels = np.full((int(num_samples),), "geom_train", dtype=object)
    geom_labels[np.asarray(geom_split["geom_val"], dtype=np.int64)] = "geom_val"
    geom_labels[np.asarray(geom_split["geom_test"], dtype=np.int64)] = "geom_test"
    oracle_error = localization_error_vector(oracle_pred, source_positions)
    gcc_error = localization_error_vector(gcc_pred, source_positions)
    gcc_tdoa_abs = np.abs(np.asarray(gcc_tdoa, dtype=np.float64) - np.asarray(true_tdoa, dtype=np.float64))
    sample_metrics_path = out_dir / "sample_metrics.csv"
    fieldnames = [
        "sample_index",
        "iid_split",
        "geom_split",
        "oracle_position_error_m",
        "gcc_phat_position_error_m",
        *LOCALIZATION_GEOMETRY_METRIC_NAMES,
        "true_tdoa_samples_01",
        "true_tdoa_samples_02",
        "true_tdoa_samples_12",
        "gcc_phat_tdoa_samples_01",
        "gcc_phat_tdoa_samples_02",
        "gcc_phat_tdoa_samples_12",
        "gcc_phat_tdoa_abs_error_samples_01",
        "gcc_phat_tdoa_abs_error_samples_02",
        "gcc_phat_tdoa_abs_error_samples_12",
        "gcc_phat_tdoa_abs_error_m_01",
        "gcc_phat_tdoa_abs_error_m_02",
        "gcc_phat_tdoa_abs_error_m_12",
    ]
    sample_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(int(num_samples)):
            row = {
                "sample_index": int(idx),
                "iid_split": str(iid_labels[idx]),
                "geom_split": str(geom_labels[idx]),
                "oracle_position_error_m": float(oracle_error[idx]),
                "gcc_phat_position_error_m": float(gcc_error[idx]),
            }
            for metric_idx, metric_name in enumerate(LOCALIZATION_GEOMETRY_METRIC_NAMES):
                row[metric_name] = float(geometry_metrics[idx, metric_idx])
            for pair_idx, pair_label in enumerate(("01", "02", "12")):
                row[f"true_tdoa_samples_{pair_label}"] = float(true_tdoa[idx, pair_idx])
                row[f"gcc_phat_tdoa_samples_{pair_label}"] = float(gcc_tdoa[idx, pair_idx])
                row[f"gcc_phat_tdoa_abs_error_samples_{pair_label}"] = float(gcc_tdoa_abs[idx, pair_idx])
                row[f"gcc_phat_tdoa_abs_error_m_{pair_label}"] = float(gcc_tdoa_abs[idx, pair_idx] / float(audit_cfg.fs) * float(audit_cfg.c))
            writer.writerow(row)
    iid_test_mask = np.zeros((int(num_samples),), dtype=bool)
    iid_test_mask[np.asarray(iid_split["test"], dtype=np.int64)] = True
    geom_test_mask = np.zeros((int(num_samples),), dtype=bool)
    geom_test_mask[np.asarray(geom_split["geom_test"], dtype=np.int64)] = True
    area_arr = _metrics_from_geometry_array(geometry_metrics, "triangle_area")
    cond_arr = _metrics_from_geometry_array(geometry_metrics, "jacobian_condition")
    inside_arr = _metrics_from_geometry_array(geometry_metrics, "inside_convex_hull")
    norm_radius_arr = _metrics_from_geometry_array(geometry_metrics, "centroid_source_dist_norm")
    grid_rows: list[dict[str, Any]] = []
    thresholds = LOCALIZATION_THRESHOLDS[str(stage_id)]
    for area_min, cond_max in LOCALIZATION_STABLE_GEOMETRY_GRID:
        keep_mask = (area_arr >= float(area_min)) & (cond_arr <= float(cond_max))
        iid_err = gcc_error[iid_test_mask & keep_mask]
        geom_err = gcc_error[geom_test_mask & keep_mask]
        iid_stats = _error_vector_stats(iid_err)
        geom_stats = _error_vector_stats(geom_err)
        oracle_iid_stats = _error_vector_stats(oracle_error[iid_test_mask & keep_mask])
        oracle_geom_stats = _error_vector_stats(oracle_error[geom_test_mask & keep_mask])
        row = {
            "area_min": float(area_min),
            "cond_max": float(cond_max),
            "retention_overall": float(np.mean(keep_mask)),
            "retention_iid_test": float(np.mean(keep_mask[iid_test_mask])) if np.any(iid_test_mask) else 0.0,
            "retention_geom_test": float(np.mean(keep_mask[geom_test_mask])) if np.any(geom_test_mask) else 0.0,
            "gcc_phat_iid_test_median_m": float(iid_stats["median_m"]),
            "gcc_phat_iid_test_p90_m": float(iid_stats["p90_m"]),
            "gcc_phat_geom_test_median_m": float(geom_stats["median_m"]),
            "gcc_phat_geom_test_p90_m": float(geom_stats["p90_m"]),
            "oracle_iid_test_median_m": float(oracle_iid_stats["median_m"]),
            "oracle_iid_test_p90_m": float(oracle_iid_stats["p90_m"]),
            "oracle_geom_test_median_m": float(oracle_geom_stats["median_m"]),
            "oracle_geom_test_p90_m": float(oracle_geom_stats["p90_m"]),
        }
        row["passed"] = bool(
            float(row["gcc_phat_iid_test_median_m"]) < float(thresholds["analytic_iid_median_max"])
            and float(row["gcc_phat_iid_test_p90_m"]) < float(thresholds["analytic_iid_p90_max"])
            and float(row["gcc_phat_geom_test_median_m"]) < float(thresholds["analytic_geom_median_max"])
            and float(row["gcc_phat_geom_test_p90_m"]) < float(thresholds["analytic_geom_p90_max"])
        )
        grid_rows.append(row)
    save_history_csv(out_dir / "threshold_grid.csv", grid_rows)
    selected = _select_stable_geometry_candidate(grid_rows, thresholds)
    _plot_geometry_scatter(
        plots_dir / "gcc_phat_error_vs_triangle_area.png",
        area_arr,
        gcc_error,
        x_label="triangle_area",
        title="GCC-PHAT position error vs triangle area",
    )
    _plot_geometry_scatter(
        plots_dir / "gcc_phat_error_vs_jacobian_condition.png",
        cond_arr,
        gcc_error,
        x_label="jacobian_condition",
        title="GCC-PHAT position error vs Jacobian condition",
        log_x=True,
    )
    _plot_geometry_scatter(
        plots_dir / "gcc_phat_error_vs_normalized_radius.png",
        norm_radius_arr,
        gcc_error,
        x_label="centroid_source_dist_norm",
        title="GCC-PHAT position error vs normalized source radius",
        color_by_inside=inside_arr,
    )
    _plot_threshold_heatmaps(plots_dir / "threshold_grid_heatmaps.png", grid_rows)
    _plot_geometry_failure_examples(
        plots_dir / "typical_failure_geometries.png",
        ref_positions[iid_test_mask],
        source_positions[iid_test_mask],
        gcc_pred[iid_test_mask],
        gcc_error[iid_test_mask],
        area_arr[iid_test_mask],
        cond_arr[iid_test_mask],
    )
    summary = {
        "stage_id": str(stage_id),
        "level": level,
        "num_samples": int(num_samples),
        "window_preset": infer_localization_window_preset(audit_cfg),
        "window_config": {"signal_len": int(audit_cfg.signal_len), "ref_window_len": int(audit_cfg.ref_window_len), "fs": int(audit_cfg.fs), "c": float(audit_cfg.c)},
        "oracle_iid_test": _error_vector_stats(oracle_error[iid_test_mask]),
        "oracle_geom_test": _error_vector_stats(oracle_error[geom_test_mask]),
        "gcc_phat_iid_test": _error_vector_stats(gcc_error[iid_test_mask]),
        "gcc_phat_geom_test": _error_vector_stats(gcc_error[geom_test_mask]),
        "gcc_phat_delay_error": delay_error_stats(gcc_tdoa, true_tdoa, audit_cfg.fs, audit_cfg.c),
        "selected_thresholds": selected,
        "num_passing_candidates": int(sum(1 for row in grid_rows if bool(row["passed"]))),
        "threshold_grid_csv": str(out_dir / "threshold_grid.csv"),
        "sample_metrics_csv": str(sample_metrics_path),
    }
    save_json(out_dir / "summary.json", summary)
    return summary


def _train_simple_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    plot_title: str,
    plot_path: Path,
    csv_path: Path,
    model_kind: str = "mlp",
    live_plot: bool = False,
) -> tuple[nn.Module, dict[str, np.ndarray], dict[str, Any]]:
    set_seed(int(seed))
    x_stats = fit_standardizer(train_x, np.arange(train_x.shape[0], dtype=np.int64))
    train_x_std = apply_standardizer(train_x, x_stats)
    val_x_std = apply_standardizer(val_x, x_stats)
    y_mean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_std = np.maximum(train_y.std(axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0e-6))
    train_y_std = ((train_y - y_mean) / y_std).astype(np.float32)
    val_y_std = ((val_y - y_mean) / y_std).astype(np.float32)
    model = build_localization_model(int(train_x.shape[1]), int(train_y.shape[1]), model_kind=model_kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_std), torch.from_numpy(train_y_std)),
        batch_size=int(batch_size),
        shuffle=True,
    )
    val_x_t = torch.from_numpy(val_x_std).to(device=device, dtype=torch.float32)
    val_y_t = torch.from_numpy(val_y_std).to(device=device, dtype=torch.float32)
    best_state = None
    best_score = (float("inf"), float("inf"), float("inf"))
    history = {"train_loss": [], "val_loss": []}
    epoch_rows: list[dict[str, Any]] = []
    for _ in range(int(epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * int(xb.shape[0])
            count += int(xb.shape[0])
        model.eval()
        with torch.no_grad():
            val_pred_std = model(val_x_t)
            val_loss = float(F.mse_loss(val_pred_std, val_y_t).detach().cpu())
            val_pred = (val_pred_std.cpu().numpy() * y_std[None, :] + y_mean[None, :]).astype(np.float32)
        val_metrics = localization_error_stats(val_pred, val_y)
        history["train_loss"].append(total / max(count, 1))
        history["val_loss"].append(val_loss)
        epoch_row = {
            "epoch": len(history["train_loss"]),
            "train_loss": history["train_loss"][-1],
            "val_loss": val_loss,
            "val_median_error_m": float(val_metrics["median_m"]),
            "val_p90_error_m": float(val_metrics["p90_m"]),
            "best_so_far_m": float(min([row["val_median_error_m"] for row in epoch_rows] + [float(val_metrics["median_m"])])),
        }
        epoch_rows.append(epoch_row)
        save_history_csv(csv_path, epoch_rows)
        render_training_curves(
            epoch_rows,
            output_path=plot_path,
            title=plot_title,
            metric_keys=["val_median_error_m", "best_so_far_m"],
            metric_labels=["val median error", "best-so-far"],
            live_plot=bool(live_plot),
        )
        score = (float(val_metrics["median_m"]), float(val_metrics["p90_m"]), float(val_loss))
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, x_stats, {"target_mean": y_mean, "target_std": y_std, "best_val_median_error_m": best_score[0], "best_val_p90_error_m": best_score[1], "history": history, "epoch_rows": epoch_rows}


def _predict_regressor(model: nn.Module, x: np.ndarray, x_stats: dict[str, np.ndarray], y_mean: np.ndarray, y_std: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        pred = model(torch.from_numpy(apply_standardizer(x, x_stats)).to(device=device, dtype=torch.float32)).cpu().numpy()
    return (pred * y_std[None, :] + y_mean[None, :]).astype(np.float32)


def train_localization_suite(
    h5_path: Path | str,
    output_dir: Path | str,
    epochs: int,
    batch_size: int,
    device: str = "auto",
    seed: int = 20260401,
    live_plot: bool = False,
    stage_id: str | None = None,
    level: str | None = None,
    candidate_mode: str = "base",
    require_analytic_gate: bool = True,
) -> dict[str, Any]:
    cfg, data, splits = load_localization_dataset(h5_path)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    stage = str(stage_id) if stage_id is not None else ("01" if str(cfg.profile).lower() == "anechoic" else "02")
    resolved_candidate_mode = str(candidate_mode).lower()
    if resolved_candidate_mode == "auto":
        resolved_candidate_mode = "stable" if str(cfg.geometry_filter_mode).lower() == "stable" else "base"
    analytic_bundle = evaluate_localization_analytic_only(cfg, data, splits, out_dir, stage_id=stage, level=level)
    summary: dict[str, Any] = {
        "h5_path": str(h5_path),
        "stage_id": stage,
        "level": level,
        "candidate_mode": str(resolved_candidate_mode),
        "geometry_filter_mode": str(cfg.geometry_filter_mode),
        "min_triangle_area": float(cfg.min_triangle_area),
        "max_jacobian_condition": float(cfg.max_jacobian_condition),
        "window_preset": analytic_bundle["window_preset"],
        "window_config": analytic_bundle["window_config"],
        "analytic_plain_gcc": analytic_bundle["analytic_plain_gcc"],
        "analytic_gcc_phat": analytic_bundle["analytic_gcc_phat"],
        "analytic_gate": analytic_bundle["analytic_gate"],
        "analytic_sample_errors_csv": analytic_bundle["analytic_sample_errors_csv"],
        "models": {},
    }
    if str(stage) == "01" and bool(require_analytic_gate) and not bool(analytic_bundle["analytic_gate"]["passed"]):
        summary["skipped_due_to_analytic_gate"] = True
        summary["train_recommendation"] = "increase_xref_window"
        save_json(out_dir / "train_summary.json", summary)
        return summary
    feature_cache: dict[str, np.ndarray] = {}
    candidate_specs = localization_candidate_specs(resolved_candidate_mode)
    for split_kind, train_key, val_key in [("iid", "train", "val"), ("geom", "geom_train", "geom_val")]:
        train_idx = splits[train_key]
        val_idx = splits[val_key]
        for feat_kind, model_kind in candidate_specs:
            if feat_kind not in feature_cache:
                feature_cache[feat_kind] = _build_localization_features(data, feat_kind)
            feats = feature_cache[feat_kind]
            model_id = localization_model_id(feat_kind, model_kind)
            model, x_stats, extra = _train_simple_regressor(
                feats[train_idx],
                data["source_position"][train_idx],
                feats[val_idx],
                data["source_position"][val_idx],
                epochs=int(epochs),
                batch_size=int(batch_size),
                device=dev,
                seed=int(seed),
                plot_title=f"Stage {stage} {split_kind} {model_id}",
                plot_path=out_dir / f"{split_kind}_{model_id}_loss_curves.png",
                csv_path=out_dir / f"{split_kind}_{model_id}_epoch_metrics.csv",
                model_kind=model_kind,
                live_plot=bool(live_plot),
            )
            ckpt_path = out_dir / f"{split_kind}_{model_id}_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_kind": feat_kind,
                    "model_kind": model_kind,
                    "split_kind": split_kind,
                    "feature_stats": x_stats,
                    "target_mean": extra["target_mean"],
                    "target_std": extra["target_std"],
                    "best_val_median_error_m": extra["best_val_median_error_m"],
                    "best_val_p90_error_m": extra["best_val_p90_error_m"],
                },
                str(ckpt_path),
            )
            save_json(out_dir / f"{split_kind}_{model_id}_history.json", extra["history"])
            summary["models"][f"{split_kind}_{model_id}"] = {
                "checkpoint_path": str(ckpt_path),
                "feature_kind": str(feat_kind),
                "model_kind": str(model_kind),
                "split_kind": str(split_kind),
                "best_val_median_error_m": float(extra["best_val_median_error_m"]),
                "best_val_p90_error_m": float(extra["best_val_p90_error_m"]),
                "loss_curve_path": str(out_dir / f"{split_kind}_{model_id}_loss_curves.png"),
                "epoch_metrics_csv": str(out_dir / f"{split_kind}_{model_id}_epoch_metrics.csv"),
            }
    save_json(out_dir / "train_summary.json", summary)
    return summary


def evaluate_localization_suite(h5_path: Path | str, output_dir: Path | str, device: str = "auto", stage_id: str | None = None, level: str | None = None) -> dict[str, Any]:
    cfg, data, splits = load_localization_dataset(h5_path)
    out_dir = Path(output_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    stage = str(stage_id) if stage_id is not None else ("01" if str(cfg.profile).lower() == "anechoic" else "02")
    analytic_bundle = evaluate_localization_analytic_only(cfg, data, splits, out_dir, stage_id=stage, level=level)
    analytic_plain_gcc = analytic_bundle["analytic_plain_gcc"]
    analytic_gcc_phat = analytic_bundle["analytic_gcc_phat"]
    learned = {}
    plot_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    train_summary_path = out_dir / "train_summary.json"
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8")) if train_summary_path.exists() else {}
    feature_cache: dict[str, np.ndarray] = {}
    for model_key, meta in train_summary.get("models", {}).items():
        ckpt_path = Path(meta["checkpoint_path"])
        if not ckpt_path.exists():
            continue
        split_kind = str(meta["split_kind"])
        test_key = "test" if split_kind == "iid" else "geom_test"
        idx = splits[test_key]
        feature_kind = str(meta["feature_kind"])
        model_kind = str(meta.get("model_kind", "mlp"))
        if feature_kind not in feature_cache:
            feature_cache[feature_kind] = _build_localization_features(data, feature_kind)
        feats = feature_cache[feature_kind]
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = build_localization_model(int(feats.shape[1]), 2, model_kind=model_kind).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        pred = _predict_regressor(
            model,
            feats[idx],
            ckpt["feature_stats"],
            np.asarray(ckpt["target_mean"], dtype=np.float32),
            np.asarray(ckpt["target_std"], dtype=np.float32),
            dev,
        )
        learned[model_key] = localization_error_stats(pred, data["source_position"][idx])
        plot_cache[model_key] = (pred, data["source_position"][idx])
    thresholds = LOCALIZATION_THRESHOLDS[stage]
    iid_keys = sorted([key for key in learned.keys() if key.startswith("iid_")])
    geom_keys = sorted([key for key in learned.keys() if key.startswith("geom_")])
    best_iid_key = min(iid_keys, key=lambda key: (learned[key]["median_m"], learned[key]["p90_m"], key)) if iid_keys else None
    best_geom_key = min(geom_keys, key=lambda key: (learned[key]["median_m"], learned[key]["p90_m"], key)) if geom_keys else None
    if stage == "01":
        passed = {
            "analytic_gcc_phat_iid": float(analytic_gcc_phat["iid_test"]["median_m"]) < thresholds["analytic_iid_median_max"]
            and float(analytic_gcc_phat["iid_test"]["p90_m"]) < thresholds["analytic_iid_p90_max"],
            "analytic_gcc_phat_geom": float(analytic_gcc_phat["geom_test"]["median_m"]) < thresholds["analytic_geom_median_max"]
            and float(analytic_gcc_phat["geom_test"]["p90_m"]) < thresholds["analytic_geom_p90_max"],
            "best_iid": bool(best_iid_key)
            and float(learned[best_iid_key]["median_m"]) < thresholds["learned_iid_median_max"]
            and float(learned[best_iid_key]["p90_m"]) < thresholds["learned_iid_p90_max"],
            "best_geom": bool(best_geom_key)
            and float(learned[best_geom_key]["median_m"]) < thresholds["learned_geom_median_max"]
            and float(learned[best_geom_key]["p90_m"]) < thresholds["learned_geom_p90_max"],
        }
    else:
        passed = {
            "best_iid": bool(best_iid_key)
            and float(learned[best_iid_key]["median_m"]) < thresholds["learned_iid_median_max"]
            and float(learned[best_iid_key]["p90_m"]) < thresholds["learned_iid_p90_max"],
            "best_geom": bool(best_geom_key)
            and float(learned[best_geom_key]["median_m"]) < thresholds["learned_geom_median_max"]
            and float(learned[best_geom_key]["p90_m"]) < thresholds["learned_geom_p90_max"],
        }
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for label, key in [("best_iid_localization.png", best_iid_key), ("best_geom_localization.png", best_geom_key)]:
            if key is None:
                continue
            pred_plot, true_plot = plot_cache[key]
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
            ax.scatter(true_plot[:, 0], true_plot[:, 1], s=14, alpha=0.55, label="True", color="tab:blue")
            ax.scatter(pred_plot[:, 0], pred_plot[:, 1], s=14, alpha=0.55, label="Pred", color="tab:orange")
            ax.set_xlim(0.0, float(cfg.plane_room_size[0]))
            ax.set_ylim(0.0, float(cfg.plane_room_size[1]))
            ax.grid(True, alpha=0.25)
            ax.legend()
            ax.set_title(key)
            fig.savefig(plots_dir / label, dpi=140, bbox_inches="tight")
            plt.close(fig)
        if best_iid_key is not None:
            split_name = "test"
            idx = splits[split_name]
            pred_plot, true_plot = plot_cache[best_iid_key]
            ref_plot = data["ref_positions"][idx]
            err = localization_error_vector(pred_plot, true_plot)
            for label, sample_idx in [("success_sample.png", int(np.argmin(err))), ("failure_sample.png", int(np.argmax(err)))]:
                fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
                tri = ref_plot[sample_idx]
                ax.scatter(tri[:, 0], tri[:, 1], s=90, color="tab:green", label="Refs")
                ax.scatter(true_plot[sample_idx, 0], true_plot[sample_idx, 1], s=90, color="tab:blue", label="True")
                ax.scatter(pred_plot[sample_idx, 0], pred_plot[sample_idx, 1], s=90, color="tab:orange", label="Pred")
                ax.plot(
                    [true_plot[sample_idx, 0], pred_plot[sample_idx, 0]],
                    [true_plot[sample_idx, 1], pred_plot[sample_idx, 1]],
                    linestyle="--",
                    color="tab:red",
                    alpha=0.6,
                )
                ax.set_xlim(0.0, float(cfg.plane_room_size[0]))
                ax.set_ylim(0.0, float(cfg.plane_room_size[1]))
                ax.grid(True, alpha=0.25)
                ax.legend()
                ax.set_title(f"{label} | error={err[sample_idx]:.3f} m")
                fig.savefig(plots_dir / label, dpi=140, bbox_inches="tight")
                plt.close(fig)
    except Exception:
        pass
    tdoa_iid_keys = sorted([key for key in learned.keys() if key.startswith("iid_tdoa")])
    tdoa_geom_keys = sorted([key for key in learned.keys() if key.startswith("geom_tdoa")])
    best_tdoa_iid_key = min(tdoa_iid_keys, key=lambda key: (learned[key]["median_m"], learned[key]["p90_m"], key)) if tdoa_iid_keys else None
    best_tdoa_geom_key = min(tdoa_geom_keys, key=lambda key: (learned[key]["median_m"], learned[key]["p90_m"], key)) if tdoa_geom_keys else None
    network_feasibility = None
    if stage == "01" and best_tdoa_iid_key is not None and best_tdoa_geom_key is not None:
        iid_pass = (
            float(learned[best_tdoa_iid_key]["median_m"]) < thresholds["learned_iid_median_max"]
            and float(learned[best_tdoa_iid_key]["p90_m"]) < thresholds["learned_iid_p90_max"]
            and float(learned[best_tdoa_iid_key]["p90_m"]) <= 2.0 * float(analytic_gcc_phat["iid_test"]["p90_m"])
        )
        geom_pass = (
            float(learned[best_tdoa_geom_key]["median_m"]) < thresholds["learned_geom_median_max"]
            and float(learned[best_tdoa_geom_key]["p90_m"]) < thresholds["learned_geom_p90_max"]
            and float(learned[best_tdoa_geom_key]["p90_m"]) <= 2.0 * float(analytic_gcc_phat["geom_test"]["p90_m"])
        )
        passed_flag = bool(iid_pass and geom_pass)
        if bool(analytic_bundle["analytic_gate"]["passed"]) and passed_flag:
            conclusion = "几何稳定 + 解析通过 + 网络通过"
        elif bool(analytic_bundle["analytic_gate"]["passed"]):
            conclusion = "几何稳定 + 解析通过 + 网络失败"
        else:
            conclusion = "几何稳定策略本身失败"
        network_feasibility = {
            "best_tdoa_iid_model_key": best_tdoa_iid_key,
            "best_tdoa_geom_model_key": best_tdoa_geom_key,
            "iid_passed": bool(iid_pass),
            "geom_passed": bool(geom_pass),
            "passed": passed_flag,
            "conclusion": conclusion,
        }
        valid_conclusions = {
            "几何稳定 + 解析通过 + 网络通过",
            "几何稳定 + 解析通过 + 网络失败",
            "几何稳定策略本身失败",
        }
        if str(network_feasibility["conclusion"]) not in valid_conclusions:
            if bool(analytic_bundle["analytic_gate"]["passed"]) and passed_flag:
                network_feasibility["conclusion"] = "几何稳定 + 解析通过 + 网络通过"
            elif bool(analytic_bundle["analytic_gate"]["passed"]):
                network_feasibility["conclusion"] = "几何稳定 + 解析通过 + 网络失败"
            else:
                network_feasibility["conclusion"] = "几何稳定策略本身失败"
    summary = {
        "h5_path": str(h5_path),
        "stage_id": stage,
        "level": level,
        "profile": cfg.profile,
        "geometry_filter_mode": str(cfg.geometry_filter_mode),
        "min_triangle_area": float(cfg.min_triangle_area),
        "max_jacobian_condition": float(cfg.max_jacobian_condition),
        "window_preset": analytic_bundle["window_preset"],
        "window_config": analytic_bundle["window_config"],
        "analytic_plain_gcc": analytic_plain_gcc,
        "analytic_gcc_phat": analytic_gcc_phat,
        "analytic": analytic_gcc_phat,
        "analytic_gate": analytic_bundle["analytic_gate"],
        "analytic_sample_errors_csv": analytic_bundle["analytic_sample_errors_csv"],
        "learned": learned,
        "best_iid_model_key": best_iid_key,
        "best_geom_model_key": best_geom_key,
        "best_tdoa_iid_model_key": best_tdoa_iid_key,
        "best_tdoa_geom_model_key": best_tdoa_geom_key,
        "gate_thresholds": thresholds,
        "passed": passed,
        "ready_for_training": bool(analytic_bundle["analytic_gate"]["passed"]) if str(stage) == "01" else True,
        "next_stage_unlocked": bool(all(bool(v) for v in passed.values())),
        "network_feasibility": network_feasibility,
    }
    save_json(out_dir / "summary.json", summary)
    return summary


@dataclass
class SingleControlValidationConfig:
    num_samples: int = 4000
    fs: int = 8000
    signal_len: int = 4096
    ref_window_len: int = 2048
    gcc_len: int = 257
    psd_nfft: int = 256
    rir_len: int = 96
    filter_len: int = 96
    q_full_len: int = 191
    c: float = 343.0
    room_size: tuple[float, float] = (4.0, 4.0)
    profile: str = "anechoic"
    reflection_gain: float = 0.65
    margin: float = 0.35
    ref_margin: float = 0.45
    min_ref_pair_dist: float = 0.25
    min_source_ref_dist: float = 0.30
    min_device_dist: float = 0.30
    seed: int = 20260401
    lambda_q_scale: float = 1.0e-6
    lambda_w: float = 1.0e-8
    replay_early_window_s: float = 0.25


def control_rir(tx_pos: np.ndarray, rx_pos: np.ndarray, cfg: SingleControlValidationConfig) -> np.ndarray:
    loc_cfg = LocalizationConfig(
        fs=int(cfg.fs),
        rir_len=int(cfg.rir_len),
        c=float(cfg.c),
        plane_room_size=tuple(cfg.room_size),
        reflection_gain=float(cfg.reflection_gain),
        profile=str(cfg.profile),
    )
    return rir_for_profile(tx_pos, rx_pos, loc_cfg)


def sample_aux_position_2d(
    rng: np.random.Generator,
    room_size: tuple[float, float],
    margin: float,
    min_dist_to: list[np.ndarray],
    min_dist: float,
) -> np.ndarray:
    width, height = float(room_size[0]), float(room_size[1])
    for _ in range(200):
        pos = rng.uniform([margin, margin], [width - margin, height - margin], size=(2,))
        if all(float(np.linalg.norm(pos - np.asarray(p, dtype=float))) >= float(min_dist) for p in min_dist_to):
            return pos.astype(np.float32)
    raise RuntimeError("Failed to sample device position.")


def pairwise_ref_paths(ref_positions: np.ndarray, cfg: SingleControlValidationConfig) -> np.ndarray:
    pairs = [(0, 1), (0, 2), (1, 2)]
    out = np.zeros((len(pairs), int(cfg.rir_len)), dtype=np.float32)
    for idx, (i, j) in enumerate(pairs):
        out[idx] = control_rir(ref_positions[i], ref_positions[j], cfg)
    return out


def build_single_control_dataset(output_h5: Path | str, cfg: SingleControlValidationConfig) -> dict[str, Any]:
    path = Path(output_h5)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.seed))
    n_bins = int(cfg.psd_nfft // 2 + 1)
    ref_positions = np.zeros((cfg.num_samples, 3, 2), dtype=np.float32)
    source_positions = np.zeros((cfg.num_samples, 2), dtype=np.float32)
    speaker_positions = np.zeros((cfg.num_samples, 2), dtype=np.float32)
    error_positions = np.zeros((cfg.num_samples, 2), dtype=np.float32)
    x_ref = np.zeros((cfg.num_samples, 3, cfg.signal_len), dtype=np.float32)
    d_signal = np.zeros((cfg.num_samples, cfg.signal_len), dtype=np.float32)
    gcc = np.zeros((cfg.num_samples, 3, cfg.gcc_len), dtype=np.float32)
    cov = np.zeros((cfg.num_samples, 9, n_bins), dtype=np.float32)
    p_ref = np.zeros((cfg.num_samples, 3, cfg.rir_len), dtype=np.float32)
    d_path = np.zeros((cfg.num_samples, cfg.rir_len), dtype=np.float32)
    s_path = np.zeros((cfg.num_samples, cfg.rir_len), dtype=np.float32)
    e2r = np.zeros((cfg.num_samples, 3, cfg.rir_len), dtype=np.float32)
    s2r = np.zeros((cfg.num_samples, 3, cfg.rir_len), dtype=np.float32)
    r2r = np.zeros((cfg.num_samples, 3, cfg.rir_len), dtype=np.float32)
    q_target = np.zeros((cfg.num_samples, cfg.q_full_len), dtype=np.float32)
    w_canon = np.zeros((cfg.num_samples, 3, cfg.filter_len), dtype=np.float32)
    for idx in range(int(cfg.num_samples)):
        ref = sample_ref_positions_2d(rng, cfg.room_size, cfg.ref_margin, cfg.min_ref_pair_dist)
        src = sample_source_position_2d(rng, cfg.room_size, cfg.margin, ref, cfg.min_source_ref_dist)
        spk = sample_aux_position_2d(rng, cfg.room_size, cfg.margin, [src] + [ref[i] for i in range(3)], cfg.min_device_dist)
        err = sample_aux_position_2d(rng, cfg.room_size, cfg.margin, [src, spk] + [ref[i] for i in range(3)], cfg.min_device_dist)
        noise = rng.standard_normal(int(cfg.signal_len)).astype(np.float32)
        ref_rirs = np.zeros((3, int(cfg.rir_len)), dtype=np.float32)
        ref_sig = np.zeros((3, int(cfg.signal_len)), dtype=np.float32)
        for mic_idx in range(3):
            ref_rirs[mic_idx] = control_rir(src, ref[mic_idx], cfg)
            ref_sig[mic_idx] = convolve_and_crop(noise, ref_rirs[mic_idx], int(cfg.signal_len))
            e2r[idx, mic_idx] = control_rir(err, ref[mic_idx], cfg)
            s2r[idx, mic_idx] = control_rir(spk, ref[mic_idx], cfg)
        d_rir = control_rir(src, err, cfg)
        s_rir = control_rir(spk, err, cfg)
        d_sig = convolve_and_crop(noise, d_rir, int(cfg.signal_len))
        q = _canonical_q_from_paths(d_rir, s_rir, int(cfg.q_full_len), float(cfg.lambda_q_scale))
        w = _solve_w_canonical_from_q(q, ref_rirs, int(cfg.filter_len), float(cfg.lambda_w), int(cfg.q_full_len))[0]
        ref_positions[idx] = ref
        source_positions[idx] = src
        speaker_positions[idx] = spk
        error_positions[idx] = err
        x_ref[idx] = ref_sig
        d_signal[idx] = d_sig
        gcc[idx] = compute_gcc_triplet(ref_sig[:, : int(cfg.ref_window_len)], out_len=int(cfg.gcc_len))
        cov[idx] = compute_reference_covariance_features(ref_sig[:, : int(cfg.ref_window_len)], n_fft=int(cfg.psd_nfft))
        p_ref[idx] = ref_rirs
        d_path[idx] = d_rir
        s_path[idx] = s_rir
        r2r[idx] = pairwise_ref_paths(ref, cfg)
        q_target[idx] = q
        w_canon[idx] = w
    splits = split_train_val_test(int(cfg.num_samples), val_frac=0.15, test_frac=0.15, seed=int(cfg.seed))
    with h5py.File(str(path), "w") as h5:
        h5.attrs["config_json"] = json.dumps(asdict(cfg), ensure_ascii=False)
        raw = h5.create_group("raw")
        for name, value in {
            "ref_positions": ref_positions,
            "source_position": source_positions,
            "speaker_position": speaker_positions,
            "error_position": error_positions,
            "x_ref": x_ref,
            "d_signal": d_signal,
            "gcc_phat": gcc,
            "reference_covariance_features": cov,
            "P_ref_paths": p_ref,
            "D_path": d_path,
            "S_path": s_path,
            "E2R_paths": e2r,
            "S2R_paths": s2r,
            "R2R_paths": r2r,
            "q_target": q_target,
            "W_canon": w_canon,
        }.items():
            raw.create_dataset(name, data=value)
        split_group = h5.create_group("splits")
        for name, arr in splits.items():
            split_group.create_dataset(name, data=np.asarray(arr, dtype=np.int64))
    summary = {"h5_path": str(path), "profile": cfg.profile, "num_samples": int(cfg.num_samples)}
    save_json(path.with_suffix(".manifest.json"), summary)
    return summary


def load_single_control_dataset(h5_path: Path | str) -> tuple[SingleControlValidationConfig, dict[str, np.ndarray], dict[str, np.ndarray]]:
    with h5py.File(str(h5_path), "r") as h5:
        cfg = SingleControlValidationConfig(**json.loads(h5.attrs["config_json"]))
        raw = h5["raw"]
        data = {name: np.asarray(raw[name]) for name in raw.keys()}
        splits = {name: np.asarray(h5["splits"][name], dtype=np.int64) for name in h5["splits"].keys()}
    for key in data:
        if data[key].dtype.kind != "i":
            data[key] = np.asarray(data[key], dtype=np.float32)
    return cfg, data, splits


def equivalent_q_from_w_np(w: np.ndarray, p_ref_paths: np.ndarray, q_len: int) -> np.ndarray:
    w_arr = np.asarray(w, dtype=np.float64)
    p_arr = np.asarray(p_ref_paths, dtype=np.float64)
    n_fft = next_pow2(int(q_len) + int(w_arr.shape[-1]) + int(p_arr.shape[-1]) - 1)
    w_f = np.fft.rfft(w_arr, n=n_fft, axis=-1)
    p_f = np.fft.rfft(p_arr, n=n_fft, axis=-1)
    q_f = np.sum(w_f * p_f, axis=-2)
    q = np.fft.irfft(q_f, n=n_fft, axis=-1)[..., : int(q_len)]
    return q.astype(np.float32)


def equivalent_q_from_w_torch(w: torch.Tensor, p_ref_paths: torch.Tensor, q_len: int) -> torch.Tensor:
    n_fft = next_pow2(int(q_len) + int(w.shape[-1]) + int(p_ref_paths.shape[-1]) - 1)
    w_f = torch.fft.rfft(w, n=n_fft, dim=-1)
    p_f = torch.fft.rfft(p_ref_paths, n=n_fft, dim=-1)
    q_f = torch.sum(w_f * p_f, dim=1)
    return torch.fft.irfft(q_f, n=n_fft, dim=-1)[..., : int(q_len)]


def static_error_signal_from_w(x_ref: np.ndarray, d_signal: np.ndarray, s_path: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.asarray(x_ref, dtype=np.float64)
    d = np.asarray(d_signal, dtype=np.float64).reshape(-1)
    s = np.asarray(s_path, dtype=np.float64).reshape(-1)
    filt = np.asarray(w, dtype=np.float64)
    y = np.zeros_like(d)
    for ref_idx in range(x.shape[0]):
        y += np.convolve(x[ref_idx], filt[ref_idx], mode="full")[: d.size]
    sec = np.convolve(y, s, mode="full")[: d.size]
    return (d + sec).astype(np.float32)


def replay_gain_db(zero_err: np.ndarray, pred_err: np.ndarray, early_samples: int) -> float:
    z = np.asarray(zero_err, dtype=np.float64)[: int(early_samples)]
    p = np.asarray(pred_err, dtype=np.float64)[: int(early_samples)]
    return float(10.0 * np.log10((np.mean(z**2) + 1.0e-12) / (np.mean(p**2) + 1.0e-12)))


def select_metric_probe(indices: np.ndarray, max_count: int = 128) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size <= int(max_count):
        return idx
    pos = np.linspace(0, idx.size - 1, int(max_count), dtype=np.int64)
    return np.sort(idx[pos])


def summarize_control_predictions(
    pred_w: np.ndarray,
    data: dict[str, np.ndarray],
    room_idx: np.ndarray,
    cfg: SingleControlValidationConfig,
) -> tuple[dict[str, float], np.ndarray]:
    idx = np.asarray(room_idx, dtype=np.int64)
    q_pred = equivalent_q_from_w_np(pred_w, data["P_ref_paths"][idx], int(cfg.q_full_len))
    q_true = data["q_target"][idx]
    q_residual = np.sqrt(np.sum((q_pred - q_true) ** 2, axis=1) / (np.sum(q_true**2, axis=1) + 1.0e-12))
    early_samples = min(int(round(float(cfg.replay_early_window_s) * float(cfg.fs))), int(cfg.signal_len))
    pred_gain = []
    exact_gain = []
    per_room = np.zeros((idx.size, 4), dtype=np.float32)
    for local_idx, room_id in enumerate(idx):
        zero_err = data["d_signal"][int(room_id)]
        pred_err = static_error_signal_from_w(data["x_ref"][int(room_id)], data["d_signal"][int(room_id)], data["S_path"][int(room_id)], pred_w[local_idx])
        exact_err = static_error_signal_from_w(data["x_ref"][int(room_id)], data["d_signal"][int(room_id)], data["S_path"][int(room_id)], data["W_canon"][int(room_id)])
        pred_gain.append(replay_gain_db(zero_err, pred_err, early_samples))
        exact_gain.append(replay_gain_db(zero_err, exact_err, early_samples))
        per_room[local_idx] = np.array([float(room_id), pred_gain[-1], exact_gain[-1], float(q_residual[local_idx])], dtype=np.float32)
    summary = {
        "q_residual_mean": float(np.mean(q_residual)),
        "replay_gain_mean_db": float(np.mean(pred_gain)),
        "replay_gain_median_db": float(np.median(pred_gain)),
        "exact_gain_mean_db": float(np.mean(exact_gain)),
        "replay_gap_mean_db": float(np.mean(np.asarray(exact_gain) - np.asarray(pred_gain))),
        "num_rooms": int(idx.size),
    }
    return summary, per_room


def _build_control_features(data: dict[str, np.ndarray], feature_kind: str) -> np.ndarray:
    gcc = data["gcc_phat"].reshape(data["gcc_phat"].shape[0], -1)
    cov = data["reference_covariance_features"].reshape(data["reference_covariance_features"].shape[0], -1)
    if feature_kind == "positions":
        pos = np.concatenate(
            [
                data["ref_positions"].reshape(data["ref_positions"].shape[0], -1),
                data["speaker_position"].reshape(data["speaker_position"].shape[0], -1),
                data["error_position"].reshape(data["error_position"].shape[0], -1),
            ],
            axis=1,
        )
        feat = np.concatenate([pos, gcc, cov], axis=1)
    elif feature_kind == "relative":
        feat = np.concatenate([gcc, cov, data["E2R_paths"].reshape(data["E2R_paths"].shape[0], -1), data["S2R_paths"].reshape(data["S2R_paths"].shape[0], -1)], axis=1)
    elif feature_kind == "relative_plus_r2r":
        feat = np.concatenate(
            [
                gcc,
                cov,
                data["E2R_paths"].reshape(data["E2R_paths"].shape[0], -1),
                data["S2R_paths"].reshape(data["S2R_paths"].shape[0], -1),
                data["R2R_paths"].reshape(data["R2R_paths"].shape[0], -1),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown control feature kind: {feature_kind}")
    return feat.astype(np.float32)


def _train_control_model(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_w: np.ndarray,
    val_w: np.ndarray,
    train_p: np.ndarray,
    val_p: np.ndarray,
    train_q: np.ndarray,
    val_q: np.ndarray,
    loss_kind: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    val_probe_p: np.ndarray,
    val_probe_q: np.ndarray,
    val_probe_w: np.ndarray,
    val_probe_x_ref: np.ndarray,
    val_probe_d_signal: np.ndarray,
    val_probe_s_path: np.ndarray,
    val_probe_x: np.ndarray,
    cfg: SingleControlValidationConfig,
    plot_title: str,
    plot_path: Path,
    csv_path: Path,
    live_plot: bool = False,
) -> tuple[SimpleMLP, dict[str, np.ndarray], dict[str, Any]]:
    set_seed(int(seed))
    x_stats = fit_standardizer(train_x, np.arange(train_x.shape[0], dtype=np.int64))
    train_x_std = apply_standardizer(train_x, x_stats)
    val_x_std = apply_standardizer(val_x, x_stats)
    train_w_log = signed_log1p_np(train_w)
    val_w_log = signed_log1p_np(val_w)
    w_mean = train_w_log.mean(axis=0, dtype=np.float64).astype(np.float32)
    w_std = np.maximum(train_w_log.std(axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0e-6))
    train_w_std = ((train_w_log - w_mean[None, :]) / w_std[None, :]).astype(np.float32)
    val_w_std = ((val_w_log - w_mean[None, :]) / w_std[None, :]).astype(np.float32)
    model = SimpleMLP(int(train_x.shape[1]), int(train_w.shape[1]), hidden=384, depth=4, dropout=0.12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x_std),
            torch.from_numpy(train_w_std),
            torch.from_numpy(train_p.astype(np.float32)),
            torch.from_numpy(train_q.astype(np.float32)),
        ),
        batch_size=int(batch_size),
        shuffle=True,
    )
    val_x_t = torch.from_numpy(val_x_std).to(device=device, dtype=torch.float32)
    val_w_t = torch.from_numpy(val_w_std).to(device=device, dtype=torch.float32)
    val_p_t = torch.from_numpy(val_p.astype(np.float32)).to(device=device, dtype=torch.float32)
    val_q_t = torch.from_numpy(val_q.astype(np.float32)).to(device=device, dtype=torch.float32)
    val_probe_x_t = torch.from_numpy(apply_standardizer(val_probe_x, x_stats)).to(device=device, dtype=torch.float32)
    w_mean_t = torch.from_numpy(w_mean).to(device=device, dtype=torch.float32)[None, :]
    w_std_t = torch.from_numpy(w_std).to(device=device, dtype=torch.float32)[None, :]
    best_val = float("inf")
    best_gain = -float("inf")
    best_gap = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    epoch_rows: list[dict[str, Any]] = []
    probe_data = {
        "P_ref_paths": np.asarray(val_probe_p, dtype=np.float32),
        "q_target": np.asarray(val_probe_q, dtype=np.float32),
        "W_canon": np.asarray(val_probe_w, dtype=np.float32),
        "x_ref": np.asarray(val_probe_x_ref, dtype=np.float32),
        "d_signal": np.asarray(val_probe_d_signal, dtype=np.float32),
        "S_path": np.asarray(val_probe_s_path, dtype=np.float32),
    }
    for _ in range(int(epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, wb_std, pb, qb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            wb_std = wb_std.to(device=device, dtype=torch.float32)
            pb = pb.to(device=device, dtype=torch.float32)
            qb = qb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred_std = torch.clamp(model(xb), min=-6.0, max=6.0)
            pred_w_log = pred_std * w_std_t + w_mean_t
            pred_w = signed_expm1_torch(pred_w_log)
            pred_w_3d = pred_w.reshape(pred_w.shape[0], 3, -1)
            if str(loss_kind) == "hyperplane":
                q_pred = equivalent_q_from_w_torch(pred_w_3d, pb, qb.shape[-1])
                numer = torch.mean((q_pred - qb) ** 2, dim=1)
                denom = torch.mean(qb**2, dim=1) + 1.0e-6
                loss = torch.mean(numer / denom) + 1.0e-4 * torch.mean(pred_w_3d**2)
            else:
                loss = F.mse_loss(pred_std, wb_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * int(xb.shape[0])
            count += int(xb.shape[0])
        model.eval()
        with torch.no_grad():
            pred_std = torch.clamp(model(val_x_t), min=-6.0, max=6.0)
            pred_w_log = pred_std * w_std_t + w_mean_t
            pred_w = signed_expm1_torch(pred_w_log)
            pred_w_3d = pred_w.reshape(pred_w.shape[0], 3, -1)
            if str(loss_kind) == "hyperplane":
                q_pred = equivalent_q_from_w_torch(pred_w_3d, val_p_t, val_q_t.shape[-1])
                numer = torch.mean((q_pred - val_q_t) ** 2, dim=1)
                denom = torch.mean(val_q_t**2, dim=1) + 1.0e-6
                val_loss = float(torch.mean(numer / denom).detach().cpu())
            else:
                val_loss = float(F.mse_loss(pred_std, val_w_t).detach().cpu())
            probe_std = torch.clamp(model(val_probe_x_t), min=-6.0, max=6.0).cpu().numpy()
            probe_local = signed_expm1_np((probe_std * w_std[None, :]) + w_mean[None, :]).reshape(val_probe_x.shape[0], 3, -1)
        probe_summary, _ = summarize_control_predictions(probe_local, probe_data, np.arange(val_probe_x.shape[0], dtype=np.int64), cfg)
        history["train_loss"].append(total / max(count, 1))
        history["val_loss"].append(val_loss)
        epoch_row = {
            "epoch": len(history["train_loss"]),
            "train_loss": history["train_loss"][-1],
            "val_loss": val_loss,
            "val_replay_gain_mean_db": float(probe_summary["replay_gain_mean_db"]),
            "val_replay_gap_mean_db": float(probe_summary["replay_gap_mean_db"]),
            "best_val_replay_gain_mean_db": float(max([row["val_replay_gain_mean_db"] for row in epoch_rows] + [float(probe_summary["replay_gain_mean_db"])])),
        }
        epoch_rows.append(epoch_row)
        save_history_csv(csv_path, epoch_rows)
        render_training_curves(
            epoch_rows,
            output_path=plot_path,
            title=plot_title,
            metric_keys=["val_replay_gain_mean_db", "val_replay_gap_mean_db"],
            metric_labels=["val replay gain", "val replay gap"],
            live_plot=bool(live_plot),
        )
        is_better = float(probe_summary["replay_gain_mean_db"]) > best_gain + 1.0e-9 or (
            np.isclose(float(probe_summary["replay_gain_mean_db"]), best_gain) and float(probe_summary["replay_gap_mean_db"]) < best_gap - 1.0e-9
        )
        if is_better:
            best_gain = float(probe_summary["replay_gain_mean_db"])
            best_gap = float(probe_summary["replay_gap_mean_db"])
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, x_stats, {
        "w_mean": w_mean,
        "w_std": w_std,
        "best_val_loss": best_val,
        "best_val_replay_gain_mean_db": best_gain,
        "best_val_replay_gap_mean_db": best_gap,
        "history": history,
        "epoch_rows": epoch_rows,
    }


def train_single_control_suite(
    h5_path: Path | str,
    output_dir: Path | str,
    feature_kind: str,
    epochs: int,
    batch_size: int,
    device: str = "auto",
    seed: int = 20260401,
    live_plot: bool = False,
    stage_id: str | None = None,
    level: str | None = None,
) -> dict[str, Any]:
    cfg, data, splits = load_single_control_dataset(h5_path)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    stage = str(stage_id) if stage_id is not None else ("03" if str(cfg.profile).lower() == "anechoic" else "04")
    feats = _build_control_features(data, feature_kind)
    train_idx = splits["train"]
    val_idx = splits["val"]
    val_probe_idx = select_metric_probe(val_idx, max_count=128)
    summary = {"h5_path": str(h5_path), "stage_id": stage, "level": level, "feature_kind": feature_kind, "val_metric_room_count": int(val_probe_idx.size), "models": {}}
    for loss_kind in ("hyperplane", "w_mse"):
        model, x_stats, extra = _train_control_model(
            feats[train_idx],
            feats[val_idx],
            data["W_canon"][train_idx].reshape(train_idx.size, -1),
            data["W_canon"][val_idx].reshape(val_idx.size, -1),
            data["P_ref_paths"][train_idx],
            data["P_ref_paths"][val_idx],
            data["q_target"][train_idx],
            data["q_target"][val_idx],
            loss_kind=loss_kind,
            epochs=int(epochs),
            batch_size=int(batch_size),
            device=dev,
            seed=int(seed),
            val_probe_p=data["P_ref_paths"][val_probe_idx],
            val_probe_q=data["q_target"][val_probe_idx],
            val_probe_w=data["W_canon"][val_probe_idx],
            val_probe_x_ref=data["x_ref"][val_probe_idx],
            val_probe_d_signal=data["d_signal"][val_probe_idx],
            val_probe_s_path=data["S_path"][val_probe_idx],
            val_probe_x=feats[val_probe_idx],
            cfg=cfg,
            plot_title=f"Stage {stage} {feature_kind} {loss_kind}",
            plot_path=out_dir / f"{feature_kind}_{loss_kind}_loss_curves.png",
            csv_path=out_dir / f"{feature_kind}_{loss_kind}_epoch_metrics.csv",
            live_plot=bool(live_plot),
        )
        ckpt_path = out_dir / f"{feature_kind}_{loss_kind}_best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_kind": feature_kind,
                "loss_kind": loss_kind,
                "feature_stats": x_stats,
                "w_mean": extra["w_mean"],
                "w_std": extra["w_std"],
                "best_val_replay_gain_mean_db": extra["best_val_replay_gain_mean_db"],
                "best_val_replay_gap_mean_db": extra["best_val_replay_gap_mean_db"],
            },
            str(ckpt_path),
        )
        save_json(out_dir / f"{feature_kind}_{loss_kind}_history.json", extra["history"])
        summary["models"][loss_kind] = {
            "checkpoint_path": str(ckpt_path),
            "best_val_loss": float(extra["best_val_loss"]),
            "best_val_replay_gain_mean_db": float(extra["best_val_replay_gain_mean_db"]),
            "best_val_replay_gap_mean_db": float(extra["best_val_replay_gap_mean_db"]),
            "loss_curve_path": str(out_dir / f"{feature_kind}_{loss_kind}_loss_curves.png"),
            "epoch_metrics_csv": str(out_dir / f"{feature_kind}_{loss_kind}_epoch_metrics.csv"),
        }
    save_json(out_dir / "train_summary.json", summary)
    return summary


def _predict_control(model: SimpleMLP, feats: np.ndarray, stats: dict[str, np.ndarray], w_mean: np.ndarray, w_std: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        pred_std = torch.clamp(model(torch.from_numpy(apply_standardizer(feats, stats)).to(device=device, dtype=torch.float32)), min=-6.0, max=6.0).cpu().numpy()
    pred = signed_expm1_np(pred_std * np.asarray(w_std, dtype=np.float32)[None, :] + np.asarray(w_mean, dtype=np.float32)[None, :])
    return pred.reshape(pred.shape[0], 3, -1).astype(np.float32)


def _evaluate_control_checkpoint(ckpt_path: Path, h5_path: Path | str, split_key: str, device: torch.device) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    cfg, data, splits = load_single_control_dataset(h5_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    feats = _build_control_features(data, str(ckpt["feature_kind"]))
    idx = np.asarray(splits[split_key], dtype=np.int64)
    model = SimpleMLP(int(feats.shape[1]), int(data["W_canon"].reshape(data["W_canon"].shape[0], -1).shape[1]), hidden=384, depth=4, dropout=0.12).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    pred_w = _predict_control(model, feats[idx], ckpt["feature_stats"], ckpt["w_mean"], ckpt["w_std"], device)
    summary, per_room = summarize_control_predictions(pred_w, data, idx, cfg)
    return summary, pred_w, per_room


def evaluate_single_control_suite(
    h5_path: Path | str,
    output_dir: Path | str,
    feature_kind: str,
    reference_summary_path: Path | str | None = None,
    device: str = "auto",
    stage_id: str | None = None,
    level: str | None = None,
) -> dict[str, Any]:
    cfg, data, splits = load_single_control_dataset(h5_path)
    out_dir = Path(output_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    stage = str(stage_id) if stage_id is not None else ("03" if str(cfg.profile).lower() == "anechoic" else "04")
    models: dict[str, Any] = {}
    pred_store: dict[str, np.ndarray] = {}
    per_room_store: dict[str, np.ndarray] = {}
    for loss_kind in ("hyperplane", "w_mse"):
        ckpt_path = out_dir / f"{feature_kind}_{loss_kind}_best.pt"
        val_summary, _, _ = _evaluate_control_checkpoint(ckpt_path, h5_path, "val", dev)
        test_summary, pred_w_test, per_room_test = _evaluate_control_checkpoint(ckpt_path, h5_path, "test", dev)
        models[loss_kind] = {"val": val_summary, "test": test_summary}
        pred_store[loss_kind] = pred_w_test
        per_room_store[loss_kind] = per_room_test
    exact_upper_bound = {
        "test_replay_mean_db": float(
            np.mean(
                [
                    replay_gain_db(
                        data["d_signal"][int(i)],
                        static_error_signal_from_w(data["x_ref"][int(i)], data["d_signal"][int(i)], data["S_path"][int(i)], data["W_canon"][int(i)]),
                        min(int(round(float(cfg.replay_early_window_s) * float(cfg.fs))), int(cfg.signal_len)),
                    )
                    for i in splits["test"]
                ]
            )
        )
    }
    threshold_key = "05" if str(stage) == "05" else stage
    gate_thresholds = CONTROL_THRESHOLDS[threshold_key]
    passed = {}
    if str(stage) in {"03", "04"}:
        passed = {
            "hyperplane_positive_val_replay": models["hyperplane"]["val"]["replay_gain_mean_db"] > 0.0,
            "hyperplane_positive_test_replay": models["hyperplane"]["test"]["replay_gain_mean_db"] > 0.0,
            "hyperplane_beats_w_mse_test": models["hyperplane"]["test"]["replay_gain_mean_db"] >= models["w_mse"]["test"]["replay_gain_mean_db"] + gate_thresholds["hyperplane_vs_w_mse_test_gain_margin_db"],
            "hyperplane_test_gap_threshold": models["hyperplane"]["test"]["replay_gap_mean_db"] < gate_thresholds["hyperplane_test_gap_max_db"],
        }
    if reference_summary_path is not None and Path(reference_summary_path).exists():
        ref = json.loads(Path(reference_summary_path).read_text(encoding="utf-8"))
        ref_gain = float(ref["models"]["hyperplane"]["test"]["replay_gain_mean_db"])
        models["relative_to_reference_test_db"] = models["hyperplane"]["test"]["replay_gain_mean_db"] - ref_gain
        if str(stage) == "05":
            passed["relative_gap_within_0p5db"] = abs(float(models["relative_to_reference_test_db"])) <= gate_thresholds["relative_to_stage03_test_gap_abs_max_db"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        test_rooms = np.asarray(splits["test"], dtype=np.int64)
        hyper_room = per_room_store["hyperplane"]
        gains = hyper_room[:, 1]
        for label, local_idx in [("success_room_compare.png", int(np.argmax(gains))), ("failure_room_compare.png", int(np.argmin(gains)))]:
            room_idx = int(test_rooms[local_idx])
            pred_h = pred_store["hyperplane"][local_idx]
            pred_b = pred_store["w_mse"][local_idx]
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
            for ref_idx, ax in enumerate(np.atleast_1d(axes)):
                ax.plot(data["W_canon"][room_idx, ref_idx], color="tab:orange", label="W canon")
                ax.plot(pred_h[ref_idx], color="tab:blue", label="hyperplane")
                ax.plot(pred_b[ref_idx], color="tab:green", alpha=0.7, label="W-MSE")
                ax.grid(True, alpha=0.25)
                ax.set_title(f"Room {room_idx} ref {ref_idx}")
                if ref_idx == 0:
                    ax.legend()
            fig.savefig(plots_dir / label, dpi=140, bbox_inches="tight")
            plt.close(fig)
            zero_err = data["d_signal"][room_idx]
            pred_err = static_error_signal_from_w(data["x_ref"][room_idx], data["d_signal"][room_idx], data["S_path"][room_idx], pred_h)
            exact_err = static_error_signal_from_w(data["x_ref"][room_idx], data["d_signal"][room_idx], data["S_path"][room_idx], data["W_canon"][room_idx])
            t0, zero_curve = _rolling_mse_db(zero_err, int(cfg.fs), max(32, int(cfg.fs * 0.02)))
            _, pred_curve = _rolling_mse_db(pred_err, int(cfg.fs), max(32, int(cfg.fs * 0.02)))
            _, exact_curve = _rolling_mse_db(exact_err, int(cfg.fs), max(32, int(cfg.fs * 0.02)))
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            ax.plot(t0, zero_curve, color="tab:gray", label="zero-init")
            ax.plot(t0, pred_curve, color="tab:blue", label="hyperplane")
            ax.plot(t0, exact_curve, color="tab:orange", label="exact")
            ax.grid(True, alpha=0.25)
            ax.legend()
            ax.set_title(f"{label} replay")
            fig.savefig(plots_dir / label.replace("_compare.png", "_replay.png"), dpi=140, bbox_inches="tight")
            plt.close(fig)
    except Exception:
        pass
    summary = {
        "h5_path": str(h5_path),
        "stage_id": stage,
        "level": level,
        "feature_kind": feature_kind,
        "profile": cfg.profile,
        "models": models,
        "exact_upper_bound": exact_upper_bound,
        "gate_thresholds": gate_thresholds,
        "passed": passed,
        "next_stage_unlocked": bool(all(bool(v) for v in passed.values())) if passed else False,
    }
    save_json(out_dir / "summary.json", summary)
    return summary


def _copy_h5_item_subset(src_item: h5py.Dataset | h5py.Group, dst_parent: h5py.Group, name: str, indices: np.ndarray, root_len: int) -> None:
    if isinstance(src_item, h5py.Group):
        dst_group = dst_parent.create_group(name)
        for key, value in src_item.attrs.items():
            dst_group.attrs[key] = value
        for child_name, child_item in src_item.items():
            _copy_h5_item_subset(child_item, dst_group, child_name, indices, root_len)
        return
    arr = src_item[...]
    if arr.shape and int(arr.shape[0]) == int(root_len):
        arr = arr[np.asarray(indices, dtype=np.int64)]
    dataset = dst_parent.create_dataset(name, data=arr)
    for key, value in src_item.attrs.items():
        dataset.attrs[key] = value


def subset_multicontrol_dataset(src_h5: Path | str, dst_h5: Path | str, num_rooms: int, seed: int = 20260401) -> dict[str, Any]:
    src = Path(src_h5)
    dst = Path(dst_h5)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(src), "r") as h5_src:
        root_len = int(h5_src["raw"]["W_full"].shape[0])
        rng = np.random.default_rng(int(seed))
        indices = np.arange(root_len, dtype=np.int64)
        rng.shuffle(indices)
        keep = np.sort(indices[: min(int(num_rooms), root_len)])
        with h5py.File(str(dst), "w") as h5_dst:
            for key, value in h5_src.attrs.items():
                h5_dst.attrs[key] = value
            h5_dst.attrs["subset_indices_json"] = json.dumps(keep.tolist(), ensure_ascii=False)
            for name, item in h5_src.items():
                _copy_h5_item_subset(item, h5_dst, name, keep, root_len)
    summary = {"src_h5": str(src), "dst_h5": str(dst), "num_rooms": int(keep.size)}
    save_json(dst.with_suffix(".manifest.json"), summary)
    return summary

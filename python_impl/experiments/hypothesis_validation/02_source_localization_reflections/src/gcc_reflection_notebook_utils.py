from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from python_impl.python_scripts.hypothesis_validation_common import (
    estimate_source_position_from_tdoa,
    load_localization_dataset,
)


PAIR_REORDER_FROM_H5 = np.asarray([0, 2, 1], dtype=np.int64)  # [01,02,12] -> [01,12,02]


@dataclass
class GCCReflectionBundle:
    h5_path: str
    fs: int
    c: float
    room_size: tuple[float, float]
    geometry_filter_mode: str
    min_triangle_area: float
    max_jacobian_condition: float
    max_triangle_angle_deg: float
    near_ref_inside_threshold_m: float
    source_region_rule_version: str
    rir_model: str
    air_attenuation_enabled: bool
    air_attenuation_alpha_per_m: float
    reflection_profile_mix: dict[str, float]
    profile_id: np.ndarray
    profile_counts: dict[str, int]
    gcc_phat: np.ndarray
    pair_distances: np.ndarray
    pair_distances_norm: np.ndarray
    pair_distance_mean: np.ndarray
    pair_distance_std: np.ndarray
    gcc_peak_lag_samples: np.ndarray
    gcc_peak_lag_norm: np.ndarray
    ref_positions_flat: np.ndarray
    ref_positions_flat_norm: np.ndarray
    ref_position_mean: np.ndarray
    ref_position_std: np.ndarray
    pair_plus_peak_features: np.ndarray
    pair_plus_ref_features: np.ndarray
    pair_plus_ref_plus_peak_features: np.ndarray
    pair_lag_bounds: np.ndarray
    tdoa_lag_samples: np.ndarray
    tdoa_lag_norm: np.ndarray
    tdoa_seconds: np.ndarray
    target_r: np.ndarray
    target_xy: np.ndarray
    ref_positions: np.ndarray
    split_indices: dict[str, np.ndarray]
    split_sizes: dict[str, int]
    train_overlap_removed: int


def _to_sorted_unique(indices: np.ndarray | list[int]) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(arr)


def _pair_distances(ref_positions: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref_positions, dtype=np.float32)
    out = np.zeros((ref.shape[0], 3), dtype=np.float32)
    pairs = ((0, 1), (1, 2), (0, 2))
    for pair_idx, (i, j) in enumerate(pairs):
        out[:, pair_idx] = np.linalg.norm(ref[:, i, :] - ref[:, j, :], axis=1).astype(np.float32)
    return out


def _radii(source_position: np.ndarray, ref_positions: np.ndarray) -> np.ndarray:
    src = np.asarray(source_position, dtype=np.float32)
    ref = np.asarray(ref_positions, dtype=np.float32)
    return np.linalg.norm(src[:, None, :] - ref, axis=2).astype(np.float32)


def _standardize(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(np.mean(train_values, axis=0), dtype=np.float32)
    std = np.asarray(np.std(train_values, axis=0), dtype=np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    norm = ((np.asarray(values, dtype=np.float32) - mean) / std).astype(np.float32)
    return norm, mean, std


def _pair_lag_bounds(pair_distances: np.ndarray, fs: int, c: float) -> np.ndarray:
    dist = np.asarray(pair_distances, dtype=np.float32)
    return (dist / float(c) * float(fs)).astype(np.float32)


def _reorder_tdoa_from_h5(lag_triplets: np.ndarray) -> np.ndarray:
    arr = np.asarray(lag_triplets, dtype=np.float32)
    return arr[:, PAIR_REORDER_FROM_H5].astype(np.float32)


def build_gcc_reflection_bundle(h5_path: str | Path) -> GCCReflectionBundle:
    cfg, data, splits = load_localization_dataset(Path(h5_path))
    ref_positions = np.asarray(data["ref_positions"], dtype=np.float32)
    source_position = np.asarray(data["source_position"], dtype=np.float32)
    pair_dist = _pair_distances(ref_positions)
    target_r = _radii(source_position, ref_positions)
    tdoa_lag_samples = _reorder_tdoa_from_h5(np.asarray(data["true_tdoa"], dtype=np.float32)).astype(np.float32)
    tdoa_seconds = (tdoa_lag_samples / float(cfg.fs)).astype(np.float32)
    gcc_phat = np.asarray(data["gcc_phat"], dtype=np.float32)
    profile_id = np.asarray(data.get("profile_id", np.zeros((ref_positions.shape[0],), dtype=np.int64)), dtype=np.int64)

    geom_holdout = np.union1d(_to_sorted_unique(splits["geom_val"]), _to_sorted_unique(splits["geom_test"]))
    train_base = _to_sorted_unique(splits["train"])
    train_idx = np.setdiff1d(train_base, geom_holdout).astype(np.int64)
    removed = int(train_base.size - train_idx.size)

    split_indices = {
        "train": train_idx,
        "iid_val": _to_sorted_unique(splits["val"]),
        "iid_test": _to_sorted_unique(splits["test"]),
        "geom_val": _to_sorted_unique(splits["geom_val"]),
        "geom_test": _to_sorted_unique(splits["geom_test"]),
    }
    split_sizes = {key: int(idx.size) for key, idx in split_indices.items()}

    pair_dist_norm, pair_mean, pair_std = _standardize(pair_dist[train_idx], pair_dist)
    ref_flat = ref_positions.reshape(ref_positions.shape[0], -1).astype(np.float32)
    ref_flat_norm, ref_mean, ref_std = _standardize(ref_flat[train_idx], ref_flat)
    lag_bounds = _pair_lag_bounds(pair_dist, fs=int(cfg.fs), c=float(cfg.c))
    lag_bounds_safe = np.maximum(lag_bounds, 1.0e-6).astype(np.float32)
    tdoa_lag_norm = np.clip(tdoa_lag_samples / lag_bounds_safe, -1.0, 1.0).astype(np.float32)
    lag_center = 0.5 * float(int(gcc_phat.shape[-1]) - 1)
    gcc_peak_idx = np.argmax(gcc_phat, axis=2).astype(np.float32)
    gcc_peak_lag_samples = (gcc_peak_idx - lag_center).astype(np.float32)
    gcc_peak_lag_norm = np.clip(gcc_peak_lag_samples / lag_bounds_safe, -1.0, 1.0).astype(np.float32)
    pair_plus_peak = np.concatenate([pair_dist_norm, gcc_peak_lag_norm], axis=1).astype(np.float32)
    pair_plus_ref = np.concatenate([pair_dist_norm, ref_flat_norm], axis=1).astype(np.float32)
    pair_plus_ref_plus_peak = np.concatenate([pair_dist_norm, ref_flat_norm, gcc_peak_lag_norm], axis=1).astype(np.float32)

    return GCCReflectionBundle(
        h5_path=str(Path(h5_path)),
        fs=int(cfg.fs),
        c=float(cfg.c),
        room_size=(float(cfg.plane_room_size[0]), float(cfg.plane_room_size[1])),
        geometry_filter_mode=str(getattr(cfg, "geometry_filter_mode", "none")),
        min_triangle_area=float(getattr(cfg, "min_triangle_area", 0.0)),
        max_jacobian_condition=float(getattr(cfg, "max_jacobian_condition", float("inf"))),
        max_triangle_angle_deg=float(getattr(cfg, "max_triangle_angle_deg", 180.0)),
        near_ref_inside_threshold_m=float(getattr(cfg, "near_ref_inside_threshold_m", 0.0)),
        source_region_rule_version=str(getattr(cfg, "source_region_rule_version", "none")),
        rir_model=str(getattr(cfg, "rir_model", "manual_2d_image_source")),
        air_attenuation_enabled=bool(getattr(cfg, "air_attenuation_enabled", False)),
        air_attenuation_alpha_per_m=float(getattr(cfg, "air_attenuation_alpha_per_m", 0.0)),
        reflection_profile_mix={
            "single_reflection": float(max(0.0, 1.0 - float(getattr(cfg, "reflection_profile_mix_anechoic_frac", 0.0))))
            if str(getattr(cfg, "profile", "single_reflection")).lower() == "single_reflection"
            else 0.0,
            "anechoic": float(getattr(cfg, "reflection_profile_mix_anechoic_frac", 0.0))
            if str(getattr(cfg, "profile", "single_reflection")).lower() == "single_reflection"
            else 1.0,
        },
        profile_id=profile_id,
        profile_counts={
            "anechoic": int(np.sum(profile_id == 0)),
            "single_reflection": int(np.sum(profile_id == 1)),
        },
        gcc_phat=gcc_phat,
        pair_distances=pair_dist,
        pair_distances_norm=pair_dist_norm,
        pair_distance_mean=pair_mean,
        pair_distance_std=pair_std,
        gcc_peak_lag_samples=gcc_peak_lag_samples,
        gcc_peak_lag_norm=gcc_peak_lag_norm,
        ref_positions_flat=ref_flat,
        ref_positions_flat_norm=ref_flat_norm,
        ref_position_mean=ref_mean,
        ref_position_std=ref_std,
        pair_plus_peak_features=pair_plus_peak,
        pair_plus_ref_features=pair_plus_ref,
        pair_plus_ref_plus_peak_features=pair_plus_ref_plus_peak,
        pair_lag_bounds=lag_bounds,
        tdoa_lag_samples=tdoa_lag_samples,
        tdoa_lag_norm=tdoa_lag_norm,
        tdoa_seconds=tdoa_seconds,
        target_r=target_r,
        target_xy=source_position,
        ref_positions=ref_positions,
        split_indices=split_indices,
        split_sizes=split_sizes,
        train_overlap_removed=removed,
    )


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _select_aux_features(bundle: GCCReflectionBundle, aux_feature_mode: str) -> tuple[np.ndarray, str]:
    mode = str(aux_feature_mode).strip().lower()
    if mode in {"pair_distance_norm", "pair_dist_norm", "pair"}:
        return np.asarray(bundle.pair_distances_norm, dtype=np.float32), "pair_distance_norm"
    if mode in {"pair_plus_peak_norm", "pair_plus_peak", "pair_peak", "pair_plus_gcc_peak_norm"}:
        return np.asarray(bundle.pair_plus_peak_features, dtype=np.float32), "pair_plus_peak_norm"
    if mode in {"pair_plus_ref_position_norm", "pair_plus_ref", "pair_ref"}:
        return np.asarray(bundle.pair_plus_ref_features, dtype=np.float32), "pair_plus_ref_position_norm"
    if mode in {"pair_plus_ref_plus_peak_norm", "pair_ref_peak", "pair_plus_ref_peak"}:
        return np.asarray(bundle.pair_plus_ref_plus_peak_features, dtype=np.float32), "pair_plus_ref_plus_peak_norm"
    raise ValueError(f"Unsupported aux_feature_mode: {aux_feature_mode}")


def _scaled_width(base: int, width_mult: float) -> int:
    return int(max(8, round(int(base) * float(width_mult))))


class GCCEncoder(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, gcc: torch.Tensor) -> torch.Tensor:
        x = self.net(gcc)
        return x.squeeze(-1)


class GCCPositionalEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        dropout_p: float = 0.10,
        channels: tuple[int, int, int] = (32, 64, 96),
        proj_hidden_dim: int = 256,
        out_dim: int = 128,
    ):
        super().__init__()
        c1, c2, c3 = (int(channels[0]), int(channels[1]), int(channels[2]))
        proj_h = int(proj_hidden_dim)
        out_h = int(out_dim)
        self.backbone = nn.Sequential(
            nn.Conv1d(3, c1, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            nn.Conv1d(c3, c3, kernel_size=3, padding=1),
            nn.GELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, int(seq_len), dtype=torch.float32)
            reduced_len = int(self.backbone(dummy).shape[-1])
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * reduced_len, proj_h),
            nn.GELU(),
            nn.Dropout(p=float(dropout_p)),
            nn.Linear(proj_h, out_h),
            nn.GELU(),
        )
        self.out_dim = out_h

    def forward(self, gcc: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(gcc))


class GCCToTDOAModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        aux_dim: int = 3,
        dropout_p: float = 0.10,
        model_width_mult: float = 1.0,
    ):
        super().__init__()
        width_mult = float(max(model_width_mult, 0.5))
        enc_channels = (
            _scaled_width(32, width_mult),
            _scaled_width(64, width_mult),
            _scaled_width(96, width_mult),
        )
        enc_proj_dim = _scaled_width(256, width_mult)
        enc_out_dim = _scaled_width(128, width_mult)
        head_h1 = _scaled_width(128, width_mult)
        head_h2 = _scaled_width(64, width_mult)
        self.encoder = GCCPositionalEncoder(
            seq_len,
            dropout_p=dropout_p,
            channels=enc_channels,
            proj_hidden_dim=enc_proj_dim,
            out_dim=enc_out_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(enc_out_dim + int(aux_dim), head_h1),
            nn.GELU(),
            nn.Linear(head_h1, head_h2),
            nn.GELU(),
            nn.Linear(head_h2, 2),
            nn.Tanh(),
        )
        self.output_dim = 2
        self.model_width_mult = width_mult
        self.encoder_channels = tuple(int(v) for v in enc_channels)
        self.encoder_proj_dim = int(enc_proj_dim)
        self.encoder_out_dim = int(enc_out_dim)
        self.head_hidden_dims = (int(head_h1), int(head_h2))

    def forward(self, gcc: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(gcc)
        return self.head(torch.cat([feat, aux_feat], dim=1))


class GCCToRelativeDistanceModel(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.encoder = GCCEncoder(seq_len)
        self.head = nn.Sequential(
            nn.Linear(64 + 3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Softplus(),
        )
        self.output_dim = 3

    def forward(self, gcc: torch.Tensor, pair_dist: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(gcc)
        return self.head(torch.cat([feat, pair_dist], dim=1))


def _reconstruct_xy_from_radii(ref_positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
    refs = np.asarray(ref_positions, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    out = np.zeros((refs.shape[0], 2), dtype=np.float64)
    for idx in range(refs.shape[0]):
        ref = refs[idx]
        ri = r[idx]
        x0, y0 = ref[0]
        a_rows = []
        b_rows = []
        for j in (1, 2):
            xj, yj = ref[j]
            a_rows.append([2.0 * (x0 - xj), 2.0 * (y0 - yj)])
            b_rows.append((ri[j] ** 2 - ri[0] ** 2) - (xj**2 + yj**2) + (x0**2 + y0**2))
        a = np.asarray(a_rows, dtype=np.float64)
        b = np.asarray(b_rows, dtype=np.float64)
        sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        out[idx] = sol
    return out.astype(np.float32)


def _xy_error_stats(pred_xy: np.ndarray, true_xy: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(np.asarray(pred_xy, dtype=np.float64) - np.asarray(true_xy, dtype=np.float64), axis=1)
    return {
        "median_m": float(np.median(err)),
        "p90_m": float(np.quantile(err, 0.90)),
        "mean_m": float(np.mean(err)),
        "max_m": float(np.max(err)),
    }


def _decode_lag_triplet_from_two_np(pred_two_norm: np.ndarray, lag_bounds: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_two_norm, dtype=np.float32)
    bounds = np.maximum(np.asarray(lag_bounds, dtype=np.float32), 1.0e-6)
    lag01 = pred[:, 0] * bounds[:, 0]
    lag12 = pred[:, 1] * bounds[:, 1]
    lag02 = lag01 + lag12
    return np.stack([lag01, lag12, lag02], axis=1).astype(np.float32)


def _decode_norm_triplet_from_two_np(pred_two_norm: np.ndarray, lag_bounds: np.ndarray) -> np.ndarray:
    lag_triplet = _decode_lag_triplet_from_two_np(pred_two_norm, lag_bounds)
    bounds = np.maximum(np.asarray(lag_bounds, dtype=np.float32), 1.0e-6)
    return (lag_triplet / bounds).astype(np.float32)


def _decode_norm_triplet_from_two_torch(pred_two_norm: torch.Tensor, lag_bounds: torch.Tensor) -> torch.Tensor:
    bounds = torch.clamp(lag_bounds, min=1.0e-6)
    lag01 = pred_two_norm[:, 0] * bounds[:, 0]
    lag12 = pred_two_norm[:, 1] * bounds[:, 1]
    lag02 = lag01 + lag12
    norm02 = lag02 / bounds[:, 2]
    return torch.stack([pred_two_norm[:, 0], pred_two_norm[:, 1], norm02], dim=1)


def _huber_loss_np(error: np.ndarray, delta: float) -> np.ndarray:
    abs_err = np.abs(np.asarray(error, dtype=np.float32))
    quad = np.minimum(abs_err, float(delta)).astype(np.float32)
    lin = abs_err - quad
    return 0.5 * (quad**2) + float(delta) * lin


def _huber_loss_torch(error: torch.Tensor, delta: float) -> torch.Tensor:
    abs_err = torch.abs(error)
    quad = torch.minimum(abs_err, torch.full_like(abs_err, float(delta)))
    lin = abs_err - quad
    return 0.5 * (quad**2) + float(delta) * lin


def _stage1_epoch_count(total_epochs: int, stage1_ratio: float) -> int:
    total = max(int(total_epochs), 1)
    ratio = float(np.clip(float(stage1_ratio), 0.0, 1.0))
    count = int(round(total * ratio))
    return int(min(max(count, 1), total))


def _curriculum_sample_weights(
    profile_id: np.ndarray,
    epoch: int,
    total_epochs: int,
    mode: str,
    stage1_ratio: float,
    stage1_anechoic_boost: float,
) -> np.ndarray | None:
    if str(mode) != "profile_two_stage":
        return None
    stage1_epochs = _stage1_epoch_count(total_epochs=total_epochs, stage1_ratio=stage1_ratio)
    if int(epoch) > int(stage1_epochs):
        return None
    boost = float(max(stage1_anechoic_boost, 1.0))
    weights = np.ones((int(profile_id.size),), dtype=np.float64)
    # profile_id==0 means anechoic in this dataset; stage1 oversamples easier samples.
    weights[np.asarray(profile_id, dtype=np.int64) == 0] = boost
    return weights


def _predict(model: nn.Module, gcc: np.ndarray, aux_features: np.ndarray, indices: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, idx.size, int(batch_size)):
            sl = idx[start : start + int(batch_size)]
            gcc_batch = torch.from_numpy(np.asarray(gcc[sl], dtype=np.float32)).to(device)
            aux_batch = torch.from_numpy(np.asarray(aux_features[sl], dtype=np.float32)).to(device)
            outputs.append(model(gcc_batch, aux_batch).cpu().numpy().astype(np.float32))
    if not outputs:
        out_dim = int(getattr(model, "output_dim", 3))
        return np.zeros((0, out_dim), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _xy_from_tdoa(bundle: GCCReflectionBundle, pred_tdoa_seconds: np.ndarray, indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    pred_rel = np.asarray(pred_tdoa_seconds, dtype=np.float32)
    pred_h5_order = pred_rel[:, PAIR_REORDER_FROM_H5]
    lag_samples = pred_h5_order * float(bundle.fs)
    out = np.zeros((idx.size, 2), dtype=np.float32)
    for row_i, sample_idx in enumerate(idx):
        out[row_i] = estimate_source_position_from_tdoa(
            lag_samples[row_i],
            bundle.ref_positions[sample_idx],
            bundle.room_size,
            bundle.fs,
            bundle.c,
        )
    return out


def _plot_history(history: list[dict[str, Any]], metric_keys: list[str], metric_labels: list[str], path: Path | None = None) -> plt.Figure:
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    axes[0].plot(epochs, [float(row["iid_val_loss"]) for row in history], label="iid val")
    axes[0].plot(epochs, [float(row["geom_val_loss"]) for row in history], label="geom val")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for key, label in zip(metric_keys, metric_labels):
        axes[1].plot(epochs, [float(row[key]) for row in history], label=label)
    axes[1].set_title("Validation Metric")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def _maybe_live_plot(history: list[dict[str, Any]], metric_keys: list[str], metric_labels: list[str], enabled: bool) -> None:
    if not enabled:
        return
    try:
        from IPython.display import clear_output, display
    except Exception:
        return
    clear_output(wait=True)
    fig = _plot_history(history, metric_keys, metric_labels, path=None)
    display(fig)
    plt.close(fig)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_scatter(true_xy: np.ndarray, pred_xy: np.ndarray, title: str, path: Path) -> None:
    err = np.linalg.norm(np.asarray(pred_xy, dtype=np.float32) - np.asarray(true_xy, dtype=np.float32), axis=1)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    sc = ax.scatter(true_xy[:, 0], true_xy[:, 1], c=err, cmap="viridis", s=12, alpha=0.65, label="true")
    ax.scatter(pred_xy[:, 0], pred_xy[:, 1], c="tab:red", s=10, alpha=0.35, label="pred")
    fig.colorbar(sc, ax=ax, label="|pred-true| (m)")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_gcc_to_tdoa_model(
    bundle: GCCReflectionBundle,
    result_dir: str | Path,
    lr: float = 1.0e-3,
    batch_size: int = 128,
    epochs: int = 20,
    seed: int = 0,
    device: str = "cpu",
    live_plot: bool = False,
    huber_delta_norm: float = 0.05,
    bound_penalty_weight: float = 0.05,
    scheduler_patience: int = 4,
    scheduler_factor: float = 0.5,
    scheduler_min_lr: float = 1.0e-6,
    early_stop_patience: int | None = 12,
    early_stop_min_delta: float = 0.0,
    curriculum_mode: str = "none",
    curriculum_stage1_ratio: float = 0.4,
    curriculum_stage1_anechoic_boost: float = 6.0,
    aux_feature_mode: str = "pair_distance_norm",
    dropout_p: float = 0.10,
    model_width_mult: float = 1.0,
    init_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(seed))
    dev = torch.device(device)
    huber_delta = float(huber_delta_norm)
    bound_weight = float(bound_penalty_weight)
    aux_features, aux_mode = _select_aux_features(bundle, aux_feature_mode)
    model = GCCToTDOAModel(
        seq_len=int(bundle.gcc_phat.shape[-1]),
        aux_dim=int(aux_features.shape[1]),
        dropout_p=float(dropout_p),
        model_width_mult=float(model_width_mult),
    ).to(dev)
    init_ckpt_resolved: str | None = None
    if init_checkpoint_path is not None:
        ckpt_path = Path(init_checkpoint_path).resolve()
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load init checkpoint '{ckpt_path}' with aux_feature_mode='{aux_mode}'. "
                f"Current model_width_mult={float(model_width_mult):.3f}. "
                "Try aux_feature_mode='pair_distance_norm' or retrain without init_checkpoint_path."
            ) from exc
        init_ckpt_resolved = str(ckpt_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_factor),
        patience=max(int(scheduler_patience), 0),
        min_lr=float(scheduler_min_lr),
    )

    train_idx = bundle.split_indices["train"]
    train_profile_id = np.asarray(bundle.profile_id[train_idx], dtype=np.int64)
    curr_mode = str(curriculum_mode).strip().lower()
    if curr_mode not in {"none", "profile_two_stage"}:
        raise ValueError(f"Unsupported curriculum_mode: {curriculum_mode}")
    train_ds = TensorDataset(
        torch.from_numpy(np.asarray(bundle.gcc_phat[train_idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(aux_features[train_idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.tdoa_lag_norm[train_idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.pair_lag_bounds[train_idx], dtype=np.float32)),
    )

    def _build_train_loader(epoch: int) -> DataLoader:
        sample_weights = _curriculum_sample_weights(
            profile_id=train_profile_id,
            epoch=int(epoch),
            total_epochs=int(epochs),
            mode=curr_mode,
            stage1_ratio=float(curriculum_stage1_ratio),
            stage1_anechoic_boost=float(curriculum_stage1_anechoic_boost),
        )
        if sample_weights is None:
            return DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(np.asarray(sample_weights, dtype=np.float64)),
            num_samples=int(train_idx.size),
            replacement=True,
        )
        return DataLoader(train_ds, batch_size=int(batch_size), sampler=sampler, drop_last=False)

    stage1_epochs = _stage1_epoch_count(total_epochs=int(epochs), stage1_ratio=float(curriculum_stage1_ratio))

    history: list[dict[str, Any]] = []
    best_selector: tuple[float, float] | None = None
    best_checkpoint_path = result_path / "best_model.pt"
    best_epoch = 0
    epochs_ran = 0
    best_geom_for_early_stop: float | None = None
    no_improve_epochs = 0
    stop_patience = None if early_stop_patience is None else int(early_stop_patience)
    if stop_patience is not None and stop_patience <= 0:
        stop_patience = None
    stop_min_delta = float(max(0.0, early_stop_min_delta))

    def _split_loss_and_mae(split_key: str) -> tuple[float, float]:
        idx = bundle.split_indices[split_key]
        pred_two_norm = _predict(model, bundle.gcc_phat, aux_features, idx, dev)
        bounds = bundle.pair_lag_bounds[idx]
        pred_norm = _decode_norm_triplet_from_two_np(pred_two_norm, bounds)
        target_norm = bundle.tdoa_lag_norm[idx]
        pred_tdoa = _decode_lag_triplet_from_two_np(pred_two_norm, bounds) / float(bundle.fs)
        target_tdoa = bundle.tdoa_seconds[idx]
        loss = float(np.mean(_huber_loss_np(pred_norm - target_norm, huber_delta)))
        mae = float(np.mean(np.abs(pred_tdoa - target_tdoa)))
        return loss, mae

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loader = _build_train_loader(epoch=epoch)
        train_loss_sum = 0.0
        train_huber_sum = 0.0
        train_bound_sum = 0.0
        batch_count = 0
        for gcc_batch, dist_batch, target_norm_batch, lag_bounds_batch in train_loader:
            gcc_batch = gcc_batch.to(dev)
            dist_batch = dist_batch.to(dev)
            target_norm_batch = target_norm_batch.to(dev)
            lag_bounds_batch = lag_bounds_batch.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred_two_norm = model(gcc_batch, dist_batch)
            pred_norm_triplet = _decode_norm_triplet_from_two_torch(pred_two_norm, lag_bounds_batch)
            huber_term = _huber_loss_torch(pred_norm_triplet - target_norm_batch, huber_delta)
            loss_huber = torch.mean(huber_term)
            bound_violation = F.relu(torch.abs(pred_norm_triplet) - 1.0)
            loss_bound = torch.mean(bound_violation**2)
            loss = loss_huber + bound_weight * loss_bound
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_huber_sum += float(loss_huber.item())
            train_bound_sum += float(loss_bound.item())
            batch_count += 1

        iid_val_loss, iid_val_mae = _split_loss_and_mae("iid_val")
        geom_val_loss, geom_val_mae = _split_loss_and_mae("geom_val")
        scheduler.step(float(geom_val_mae))
        current_lr = float(optimizer.param_groups[0]["lr"])
        curriculum_stage = "stage1" if (curr_mode == "profile_two_stage" and int(epoch) <= int(stage1_epochs)) else "stage2"
        row = {
            "epoch": int(epoch),
            "lr": current_lr,
            "train_loss": train_loss_sum / max(batch_count, 1),
            "train_huber_loss": train_huber_sum / max(batch_count, 1),
            "train_bound_loss": train_bound_sum / max(batch_count, 1),
            "iid_val_loss": iid_val_loss,
            "geom_val_loss": geom_val_loss,
            "iid_val_tdoa_mae_s": iid_val_mae,
            "geom_val_tdoa_mae_s": geom_val_mae,
            "curriculum_stage": curriculum_stage,
        }
        history.append(row)
        _maybe_live_plot(history, ["iid_val_tdoa_mae_s", "geom_val_tdoa_mae_s"], ["iid val tdoa mae", "geom val tdoa mae"], bool(live_plot))
        selector = (geom_val_mae, iid_val_mae)
        if best_selector is None or selector < best_selector:
            best_selector = selector
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": int(seed),
                    "target_parameterization": "normalized_lag_samples_2dof_reconstruct_3rd",
                    "huber_delta_norm": huber_delta,
                    "bound_penalty_weight": bound_weight,
                    "scheduler_patience": int(scheduler_patience),
                    "scheduler_factor": float(scheduler_factor),
                    "scheduler_min_lr": float(scheduler_min_lr),
                    "early_stop_patience": stop_patience,
                    "early_stop_min_delta": float(stop_min_delta),
                    "curriculum_mode": curr_mode,
                    "curriculum_stage1_ratio": float(curriculum_stage1_ratio),
                    "curriculum_stage1_anechoic_boost": float(curriculum_stage1_anechoic_boost),
                    "aux_feature_mode": aux_mode,
                    "aux_feature_dim": int(aux_features.shape[1]),
                    "dropout_p": float(dropout_p),
                    "model_width_mult": float(model.model_width_mult),
                    "encoder_channels": list(model.encoder_channels),
                    "encoder_proj_dim": int(model.encoder_proj_dim),
                    "encoder_out_dim": int(model.encoder_out_dim),
                    "head_hidden_dims": list(model.head_hidden_dims),
                    "init_checkpoint_path": init_ckpt_resolved,
                },
                best_checkpoint_path,
            )
        geom_metric = float(geom_val_mae)
        if best_geom_for_early_stop is None or geom_metric < (best_geom_for_early_stop - stop_min_delta):
            best_geom_for_early_stop = geom_metric
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        epochs_ran = int(epoch)
        if stop_patience is not None and no_improve_epochs >= stop_patience:
            break

    _write_history_csv(result_path / "train_history.csv", history)
    fig = _plot_history(history, ["iid_val_tdoa_mae_s", "geom_val_tdoa_mae_s"], ["iid val tdoa mae", "geom val tdoa mae"], result_path / "loss_curves.png")
    plt.close(fig)

    ckpt = torch.load(best_checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    summary: dict[str, Any] = {
        "h5_path": bundle.h5_path,
        "best_epoch": int(best_epoch),
        "epochs_requested": int(epochs),
        "epochs_ran": int(epochs_ran),
        "checkpoint_path": str(best_checkpoint_path),
        "target_parameterization": "normalized_lag_samples_2dof_reconstruct_3rd",
        "huber_delta_norm": huber_delta,
        "bound_penalty_weight": bound_weight,
        "scheduler_patience": int(scheduler_patience),
        "scheduler_factor": float(scheduler_factor),
        "scheduler_min_lr": float(scheduler_min_lr),
        "early_stop_patience": stop_patience,
        "early_stop_min_delta": float(stop_min_delta),
        "curriculum_mode": curr_mode,
        "curriculum_stage1_ratio": float(curriculum_stage1_ratio),
        "curriculum_stage1_anechoic_boost": float(curriculum_stage1_anechoic_boost),
        "curriculum_stage1_epochs": int(stage1_epochs),
        "aux_feature_mode": aux_mode,
        "aux_feature_dim": int(aux_features.shape[1]),
        "dropout_p": float(dropout_p),
        "model_width_mult": float(model.model_width_mult),
        "encoder_channels": list(model.encoder_channels),
        "encoder_proj_dim": int(model.encoder_proj_dim),
        "encoder_out_dim": int(model.encoder_out_dim),
        "head_hidden_dims": list(model.head_hidden_dims),
        "init_checkpoint_path": init_ckpt_resolved,
        "geometry_filter_mode": bundle.geometry_filter_mode,
        "min_triangle_area": float(bundle.min_triangle_area),
        "max_jacobian_condition": float(bundle.max_jacobian_condition),
        "max_triangle_angle_deg": float(bundle.max_triangle_angle_deg),
        "near_ref_inside_threshold_m": float(bundle.near_ref_inside_threshold_m),
        "source_region_rule_version": bundle.source_region_rule_version,
        "rir_model": bundle.rir_model,
        "air_attenuation_enabled": bool(bundle.air_attenuation_enabled),
        "air_attenuation_alpha_per_m": float(bundle.air_attenuation_alpha_per_m),
        "reflection_profile_mix": bundle.reflection_profile_mix,
        "profile_counts": bundle.profile_counts,
        "split_sizes": bundle.split_sizes,
        "train_overlap_removed": int(bundle.train_overlap_removed),
    }
    for split_key in ("iid_val", "geom_val", "iid_test", "geom_test"):
        idx = bundle.split_indices[split_key]
        pred_two_norm = _predict(model, bundle.gcc_phat, aux_features, idx, dev)
        bounds = bundle.pair_lag_bounds[idx]
        pred_lag = _decode_lag_triplet_from_two_np(pred_two_norm, bounds)
        pred_norm = _decode_norm_triplet_from_two_np(pred_two_norm, bounds)
        pred_tdoa = pred_lag / float(bundle.fs)
        true_tdoa = bundle.tdoa_seconds[idx]
        pred_xy = _xy_from_tdoa(bundle, pred_tdoa, idx)
        xy_stats = _xy_error_stats(pred_xy, bundle.target_xy[idx])
        summary[split_key] = {
            "norm_huber_loss": float(np.mean(_huber_loss_np(pred_norm - bundle.tdoa_lag_norm[idx], huber_delta))),
            "tdoa_mse_s2": float(np.mean((pred_tdoa - true_tdoa) ** 2)),
            "tdoa_mae_s": float(np.mean(np.abs(pred_tdoa - true_tdoa))),
            "tdoa_mae_m": float(np.mean(np.abs(pred_tdoa - true_tdoa)) * bundle.c),
            "consistency_mae_samples": float(np.mean(np.abs(pred_lag[:, 2] - (pred_lag[:, 0] + pred_lag[:, 1])))),
            **xy_stats,
        }
        if str(split_key).endswith("test"):
            _plot_scatter(bundle.target_xy[idx], pred_xy, f"{split_key}: GCC -> TDOA -> XY", result_path / f"scatter_{split_key}.png")
    _save_json(result_path / "summary.json", summary)
    return summary


def train_gcc_to_relative_distance_model(
    bundle: GCCReflectionBundle,
    result_dir: str | Path,
    alpha: float = 1.0,
    beta: float = 1.0,
    lr: float = 1.0e-3,
    batch_size: int = 128,
    epochs: int = 20,
    seed: int = 0,
    device: str = "cpu",
    live_plot: bool = False,
) -> dict[str, Any]:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(seed))
    dev = torch.device(device)
    model = GCCToRelativeDistanceModel(seq_len=int(bundle.gcc_phat.shape[-1])).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    train_idx = bundle.split_indices["train"]
    train_ds = TensorDataset(
        torch.from_numpy(np.asarray(bundle.gcc_phat[train_idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.pair_distances_norm[train_idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.target_r[train_idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.tdoa_seconds[train_idx], dtype=np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)

    history: list[dict[str, Any]] = []
    best_selector: tuple[float, float, float, float] | None = None
    best_checkpoint_path = result_path / "best_model.pt"
    best_epoch = 0

    def _eval_split(split_key: str) -> dict[str, Any]:
        idx = bundle.split_indices[split_key]
        pred_r = _predict(model, bundle.gcc_phat, bundle.pair_distances_norm, idx, dev)
        pred_diff = np.stack(
            [pred_r[:, 0] - pred_r[:, 1], pred_r[:, 1] - pred_r[:, 2], pred_r[:, 0] - pred_r[:, 2]],
            axis=1,
        )
        target_diff = bundle.c * bundle.tdoa_seconds[idx]
        pred_xy = _reconstruct_xy_from_radii(bundle.ref_positions[idx], pred_r)
        return {
            "distance_mse": float(np.mean((pred_r - bundle.target_r[idx]) ** 2)),
            "distance_mae": float(np.mean(np.abs(pred_r - bundle.target_r[idx]))),
            "geo_mse": float(np.mean((pred_diff - target_diff) ** 2)),
            "geo_mae": float(np.mean(np.abs(pred_diff - target_diff))),
            "pred_xy": pred_xy,
            **_xy_error_stats(pred_xy, bundle.target_xy[idx]),
        }

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        batch_count = 0
        for gcc_batch, dist_batch, r_batch, tdoa_batch in train_loader:
            gcc_batch = gcc_batch.to(dev)
            dist_batch = dist_batch.to(dev)
            r_batch = r_batch.to(dev)
            tdoa_batch = tdoa_batch.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred_r = model(gcc_batch, dist_batch)
            pred_diff = torch.stack(
                [pred_r[:, 0] - pred_r[:, 1], pred_r[:, 1] - pred_r[:, 2], pred_r[:, 0] - pred_r[:, 2]],
                dim=1,
            )
            target_diff = float(bundle.c) * tdoa_batch
            loss_r = torch.mean((pred_r - r_batch) ** 2)
            loss_geo = torch.mean((pred_diff - target_diff) ** 2)
            loss = float(alpha) * loss_r + float(beta) * loss_geo
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            batch_count += 1

        iid_val = _eval_split("iid_val")
        geom_val = _eval_split("geom_val")
        row = {
            "epoch": int(epoch),
            "train_loss": train_loss_sum / max(batch_count, 1),
            "iid_val_loss": float(iid_val["distance_mse"]),
            "geom_val_loss": float(geom_val["distance_mse"]),
            "iid_val_median_m": float(iid_val["median_m"]),
            "iid_val_p90_m": float(iid_val["p90_m"]),
            "geom_val_median_m": float(geom_val["median_m"]),
            "geom_val_p90_m": float(geom_val["p90_m"]),
        }
        history.append(row)
        _maybe_live_plot(history, ["iid_val_p90_m", "geom_val_p90_m"], ["iid val p90", "geom val p90"], bool(live_plot))
        selector = (float(geom_val["p90_m"]), float(geom_val["median_m"]), float(iid_val["p90_m"]), float(iid_val["median_m"]))
        if best_selector is None or selector < best_selector:
            best_selector = selector
            best_epoch = int(epoch)
            torch.save({"model_state": model.state_dict(), "seed": int(seed)}, best_checkpoint_path)

    _write_history_csv(result_path / "train_history.csv", history)
    fig = _plot_history(history, ["iid_val_p90_m", "geom_val_p90_m"], ["iid val p90", "geom val p90"], result_path / "loss_curves.png")
    plt.close(fig)

    ckpt = torch.load(best_checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    summary: dict[str, Any] = {
        "h5_path": bundle.h5_path,
        "best_epoch": int(best_epoch),
        "checkpoint_path": str(best_checkpoint_path),
        "geometry_filter_mode": bundle.geometry_filter_mode,
        "min_triangle_area": float(bundle.min_triangle_area),
        "max_jacobian_condition": float(bundle.max_jacobian_condition),
        "max_triangle_angle_deg": float(bundle.max_triangle_angle_deg),
        "near_ref_inside_threshold_m": float(bundle.near_ref_inside_threshold_m),
        "source_region_rule_version": bundle.source_region_rule_version,
        "rir_model": bundle.rir_model,
        "air_attenuation_enabled": bool(bundle.air_attenuation_enabled),
        "air_attenuation_alpha_per_m": float(bundle.air_attenuation_alpha_per_m),
        "reflection_profile_mix": bundle.reflection_profile_mix,
        "profile_counts": bundle.profile_counts,
        "split_sizes": bundle.split_sizes,
        "train_overlap_removed": int(bundle.train_overlap_removed),
    }
    for split_key in ("iid_test", "geom_test"):
        idx = bundle.split_indices[split_key]
        pred_r = _predict(model, bundle.gcc_phat, bundle.pair_distances_norm, idx, dev)
        pred_diff = np.stack(
            [pred_r[:, 0] - pred_r[:, 1], pred_r[:, 1] - pred_r[:, 2], pred_r[:, 0] - pred_r[:, 2]],
            axis=1,
        )
        target_diff = bundle.c * bundle.tdoa_seconds[idx]
        pred_xy = _reconstruct_xy_from_radii(bundle.ref_positions[idx], pred_r)
        summary[split_key] = {
            "distance_mse": float(np.mean((pred_r - bundle.target_r[idx]) ** 2)),
            "distance_mae": float(np.mean(np.abs(pred_r - bundle.target_r[idx]))),
            "geo_mse": float(np.mean((pred_diff - target_diff) ** 2)),
            "geo_mae": float(np.mean(np.abs(pred_diff - target_diff))),
            **_xy_error_stats(pred_xy, bundle.target_xy[idx]),
        }
        _plot_scatter(bundle.target_xy[idx], pred_xy, f"{split_key}: GCC -> Relative Distance -> XY", result_path / f"scatter_{split_key}.png")
    _save_json(result_path / "summary.json", summary)
    return summary

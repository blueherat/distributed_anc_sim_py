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
from torch.utils.data import DataLoader, TensorDataset

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


def _reorder_tdoa_from_h5(lag_triplets: np.ndarray) -> np.ndarray:
    arr = np.asarray(lag_triplets, dtype=np.float32)
    return arr[:, PAIR_REORDER_FROM_H5].astype(np.float32)


def build_gcc_reflection_bundle(h5_path: str | Path) -> GCCReflectionBundle:
    cfg, data, splits = load_localization_dataset(Path(h5_path))
    ref_positions = np.asarray(data["ref_positions"], dtype=np.float32)
    source_position = np.asarray(data["source_position"], dtype=np.float32)
    pair_dist = _pair_distances(ref_positions)
    target_r = _radii(source_position, ref_positions)
    tdoa_seconds = (_reorder_tdoa_from_h5(np.asarray(data["true_tdoa"], dtype=np.float32)) / float(cfg.fs)).astype(np.float32)
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


class GCCToTDOAModel(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.encoder = GCCEncoder(seq_len)
        self.head = nn.Sequential(
            nn.Linear(64 + 3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, gcc: torch.Tensor, pair_dist: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(gcc)
        return self.head(torch.cat([feat, pair_dist], dim=1))


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


def _predict(model: nn.Module, gcc: np.ndarray, pair_dist: np.ndarray, indices: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, idx.size, int(batch_size)):
            sl = idx[start : start + int(batch_size)]
            gcc_batch = torch.from_numpy(np.asarray(gcc[sl], dtype=np.float32)).to(device)
            dist_batch = torch.from_numpy(np.asarray(pair_dist[sl], dtype=np.float32)).to(device)
            outputs.append(model(gcc_batch, dist_batch).cpu().numpy().astype(np.float32))
    if not outputs:
        return np.zeros((0, 3), dtype=np.float32)
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
) -> dict[str, Any]:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(seed))
    dev = torch.device(device)
    model = GCCToTDOAModel(seq_len=int(bundle.gcc_phat.shape[-1])).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    train_idx = bundle.split_indices["train"]
    train_ds = TensorDataset(
        torch.from_numpy(np.asarray(bundle.gcc_phat[train_idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.pair_distances_norm[train_idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.tdoa_seconds[train_idx], dtype=np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=False)

    history: list[dict[str, Any]] = []
    best_selector: tuple[float, float] | None = None
    best_checkpoint_path = result_path / "best_model.pt"
    best_epoch = 0

    def _split_loss_and_mae(split_key: str) -> tuple[float, float]:
        idx = bundle.split_indices[split_key]
        pred = _predict(model, bundle.gcc_phat, bundle.pair_distances_norm, idx, dev)
        target = bundle.tdoa_seconds[idx]
        loss = float(np.mean((pred - target) ** 2))
        mae = float(np.mean(np.abs(pred - target)))
        return loss, mae

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        batch_count = 0
        for gcc_batch, dist_batch, tdoa_batch in train_loader:
            gcc_batch = gcc_batch.to(dev)
            dist_batch = dist_batch.to(dev)
            tdoa_batch = tdoa_batch.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred = model(gcc_batch, dist_batch)
            loss = torch.mean((pred - tdoa_batch) ** 2)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            batch_count += 1

        iid_val_loss, iid_val_mae = _split_loss_and_mae("iid_val")
        geom_val_loss, geom_val_mae = _split_loss_and_mae("geom_val")
        row = {
            "epoch": int(epoch),
            "train_loss": train_loss_sum / max(batch_count, 1),
            "iid_val_loss": iid_val_loss,
            "geom_val_loss": geom_val_loss,
            "iid_val_tdoa_mae_s": iid_val_mae,
            "geom_val_tdoa_mae_s": geom_val_mae,
        }
        history.append(row)
        _maybe_live_plot(history, ["iid_val_tdoa_mae_s", "geom_val_tdoa_mae_s"], ["iid val tdoa mae", "geom val tdoa mae"], bool(live_plot))
        selector = (geom_val_mae, iid_val_mae)
        if best_selector is None or selector < best_selector:
            best_selector = selector
            best_epoch = int(epoch)
            torch.save({"model_state": model.state_dict(), "seed": int(seed)}, best_checkpoint_path)

    _write_history_csv(result_path / "train_history.csv", history)
    fig = _plot_history(history, ["iid_val_tdoa_mae_s", "geom_val_tdoa_mae_s"], ["iid val tdoa mae", "geom val tdoa mae"], result_path / "loss_curves.png")
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
        pred_tdoa = _predict(model, bundle.gcc_phat, bundle.pair_distances_norm, idx, dev)
        true_tdoa = bundle.tdoa_seconds[idx]
        pred_xy = _xy_from_tdoa(bundle, pred_tdoa, idx)
        xy_stats = _xy_error_stats(pred_xy, bundle.target_xy[idx])
        summary[split_key] = {
            "tdoa_mse_s2": float(np.mean((pred_tdoa - true_tdoa) ** 2)),
            "tdoa_mae_s": float(np.mean(np.abs(pred_tdoa - true_tdoa))),
            "tdoa_mae_m": float(np.mean(np.abs(pred_tdoa - true_tdoa)) * bundle.c),
            **xy_stats,
        }
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

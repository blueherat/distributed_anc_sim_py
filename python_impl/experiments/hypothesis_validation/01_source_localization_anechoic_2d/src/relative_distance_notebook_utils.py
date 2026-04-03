from __future__ import annotations

import csv
import json
import math
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
    load_localization_dataset,
    estimate_tdoa_from_gcc_triplet,
    localization_geometry_metrics,
)


PAIR_LABELS = ("12", "23", "13")
PAIR_INDICES_REL = ((0, 1), (1, 2), (0, 2))
PAIR_REORDER_FROM_H5 = np.asarray([0, 2, 1], dtype=np.int64)


@dataclass
class RelativeDistanceBundle:
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
    use_true_tdoa: bool
    features_raw: np.ndarray
    features_norm: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_r: np.ndarray
    target_xy: np.ndarray
    ref_positions: np.ndarray
    tdoa_seconds: np.ndarray
    raw_pair_distances: np.ndarray
    split_indices: dict[str, np.ndarray]
    split_sizes: dict[str, int]
    train_overlap_removed: int


def _to_sorted_unique(indices: np.ndarray | list[int]) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(arr)


def _standardize(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(np.mean(train_values, axis=0), dtype=np.float32)
    std = np.asarray(np.std(train_values, axis=0), dtype=np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    norm = ((np.asarray(values, dtype=np.float32) - mean) / std).astype(np.float32)
    return norm, mean, std


def _pair_distances(ref_positions: np.ndarray) -> np.ndarray:
    ref = np.asarray(ref_positions, dtype=np.float32)
    out = np.zeros((ref.shape[0], 3), dtype=np.float32)
    for pair_idx, (i, j) in enumerate(PAIR_INDICES_REL):
        out[:, pair_idx] = np.linalg.norm(ref[:, i, :] - ref[:, j, :], axis=1).astype(np.float32)
    return out


def _radii(source_position: np.ndarray, ref_positions: np.ndarray) -> np.ndarray:
    src = np.asarray(source_position, dtype=np.float32)
    ref = np.asarray(ref_positions, dtype=np.float32)
    return np.linalg.norm(src[:, None, :] - ref, axis=2).astype(np.float32)


def _reorder_tdoa_from_h5(lag_triplets: np.ndarray) -> np.ndarray:
    arr = np.asarray(lag_triplets, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected TDOA shape (N, 3), got {arr.shape}")
    return arr[:, PAIR_REORDER_FROM_H5].astype(np.float32)


def _estimate_tdoa_seconds_from_gcc(gcc_phat: np.ndarray, fs: int) -> np.ndarray:
    est = np.stack([estimate_tdoa_from_gcc_triplet(row) for row in np.asarray(gcc_phat, dtype=np.float32)], axis=0)
    est = _reorder_tdoa_from_h5(est)
    return (est / float(fs)).astype(np.float32)


def _true_tdoa_seconds(true_tdoa_samples: np.ndarray, fs: int) -> np.ndarray:
    return (_reorder_tdoa_from_h5(true_tdoa_samples) / float(fs)).astype(np.float32)


def build_relative_distance_bundle(h5_path: str | Path, use_true_tdoa: bool = True) -> RelativeDistanceBundle:
    cfg, data, splits = load_localization_dataset(Path(h5_path))

    ref_positions = np.asarray(data["ref_positions"], dtype=np.float32)
    source_position = np.asarray(data["source_position"], dtype=np.float32)
    pair_dist = _pair_distances(ref_positions)
    true_r = _radii(source_position, ref_positions)

    if use_true_tdoa:
        tdoa_seconds = _true_tdoa_seconds(np.asarray(data["true_tdoa"], dtype=np.float32), int(cfg.fs))
    else:
        tdoa_seconds = _estimate_tdoa_seconds_from_gcc(np.asarray(data["gcc_phat"], dtype=np.float32), int(cfg.fs))

    features_raw = np.concatenate([pair_dist, tdoa_seconds], axis=1).astype(np.float32)

    geom_holdout = np.union1d(
        _to_sorted_unique(splits["geom_val"]),
        _to_sorted_unique(splits["geom_test"]),
    )
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

    features_norm, feature_mean, feature_std = _standardize(features_raw[train_idx], features_raw)

    return RelativeDistanceBundle(
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
        use_true_tdoa=bool(use_true_tdoa),
        features_raw=features_raw,
        features_norm=features_norm,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_r=true_r.astype(np.float32),
        target_xy=source_position.astype(np.float32),
        ref_positions=ref_positions.astype(np.float32),
        tdoa_seconds=tdoa_seconds.astype(np.float32),
        raw_pair_distances=pair_dist.astype(np.float32),
        split_indices=split_indices,
        split_sizes=split_sizes,
        train_overlap_removed=removed,
    )


def random_consistency_check(bundle: RelativeDistanceBundle, num_samples: int = 16, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    n = bundle.target_r.shape[0]
    choose = rng.choice(n, size=min(int(num_samples), n), replace=False)
    r = bundle.target_r[choose]
    tdoa = bundle.tdoa_seconds[choose]
    rhs = np.stack(
        [
            r[:, 0] - r[:, 1],
            r[:, 1] - r[:, 2],
            r[:, 0] - r[:, 2],
        ],
        axis=1,
    ).astype(np.float32)
    lhs = bundle.c * tdoa
    abs_err = np.abs(rhs - lhs)
    return {
        "num_samples": int(choose.size),
        "mean_abs_err_m": float(np.mean(abs_err)),
        "max_abs_err_m": float(np.max(abs_err)),
    }


def reconstruct_xy_from_radii(ref_positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
    refs = np.asarray(ref_positions, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    if refs.ndim != 3 or refs.shape[1:] != (3, 2):
        raise ValueError(f"Expected ref_positions shape (N, 3, 2), got {refs.shape}")
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"Expected radii shape (N, 3), got {r.shape}")

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


def _geo_error(pred_r: np.ndarray, tdoa_seconds: np.ndarray, c: float) -> np.ndarray:
    pred_r = np.asarray(pred_r, dtype=np.float32)
    pred_diff = np.stack(
        [
            pred_r[:, 0] - pred_r[:, 1],
            pred_r[:, 1] - pred_r[:, 2],
            pred_r[:, 0] - pred_r[:, 2],
        ],
        axis=1,
    )
    target_diff = float(c) * np.asarray(tdoa_seconds, dtype=np.float32)
    return pred_diff - target_diff


def evaluate_prediction_block(
    pred_r: np.ndarray,
    true_r: np.ndarray,
    ref_positions: np.ndarray,
    true_xy: np.ndarray,
    tdoa_seconds: np.ndarray,
    c: float,
) -> dict[str, Any]:
    pred_r = np.asarray(pred_r, dtype=np.float32)
    true_r = np.asarray(true_r, dtype=np.float32)
    pred_xy = reconstruct_xy_from_radii(ref_positions, pred_r)
    xy_stats = _xy_error_stats(pred_xy, true_xy)
    geo_res = _geo_error(pred_r, tdoa_seconds, c)
    return {
        "pred_xy": pred_xy,
        "distance_mse": float(np.mean((pred_r - true_r) ** 2)),
        "distance_mae": float(np.mean(np.abs(pred_r - true_r))),
        "geo_mse": float(np.mean(geo_res**2)),
        "geo_mae": float(np.mean(np.abs(geo_res))),
        **xy_stats,
    }


class RelativeDistanceMLP(nn.Module):
    def __init__(self, in_dim: int = 6, hidden_dims: tuple[int, int, int] = (64, 128, 64), out_dim: int = 3):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, out_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def hybrid_distance_geo_loss(
    pred_r: torch.Tensor,
    true_r: torch.Tensor,
    tdoa_seconds: torch.Tensor,
    c: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_diff = torch.stack(
        [
            pred_r[:, 0] - pred_r[:, 1],
            pred_r[:, 1] - pred_r[:, 2],
            pred_r[:, 0] - pred_r[:, 2],
        ],
        dim=1,
    )
    target_diff = float(c) * tdoa_seconds
    loss_r = torch.mean((pred_r - true_r) ** 2)
    loss_geo = torch.mean((pred_diff - target_diff) ** 2)
    total = float(alpha) * loss_r + float(beta) * loss_geo
    return total, loss_r, loss_geo


def _make_loader(features: np.ndarray, target_r: np.ndarray, tdoa_seconds: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    idx = np.asarray(indices, dtype=np.int64)
    ds = TensorDataset(
        torch.from_numpy(np.asarray(features[idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(target_r[idx], dtype=np.float32)),
        torch.from_numpy(np.asarray(tdoa_seconds[idx], dtype=np.float32)),
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=False)


def _predict_radii(model: nn.Module, features: np.ndarray, indices: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    idx = np.asarray(indices, dtype=np.int64)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, idx.size, int(batch_size)):
            sl = idx[start : start + int(batch_size)]
            x = torch.from_numpy(np.asarray(features[sl], dtype=np.float32)).to(device)
            y = model(x).cpu().numpy()
            outputs.append(np.asarray(y, dtype=np.float32))
    if not outputs:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _selector_tuple(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["geom_val"]["p90_m"]),
        float(metrics["geom_val"]["median_m"]),
        float(metrics["iid_val"]["p90_m"]),
        float(metrics["iid_val"]["median_m"]),
    )


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_training_curves(history: list[dict[str, Any]], path: Path | None = None) -> plt.Figure:
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [float(row["train_total_loss"]) for row in history], label="train total")
    axes[0].plot(epochs, [float(row["train_r_loss"]) for row in history], label="train r-loss")
    axes[0].plot(epochs, [float(row["train_geo_loss"]) for row in history], label="train geo-loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [float(row["iid_val_median_m"]) for row in history], label="iid val median")
    axes[1].plot(epochs, [float(row["geom_val_median_m"]) for row in history], label="geom val median")
    axes[1].plot(epochs, [float(row["iid_val_p90_m"]) for row in history], label="iid val p90")
    axes[1].plot(epochs, [float(row["geom_val_p90_m"]) for row in history], label="geom val p90")
    axes[1].set_title("Validation Localization Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Meters")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def _maybe_live_plot(history: list[dict[str, Any]], enabled: bool) -> None:
    if not enabled:
        return
    try:
        from IPython.display import clear_output, display
    except Exception:
        return
    clear_output(wait=True)
    fig = _plot_training_curves(history, path=None)
    display(fig)
    plt.close(fig)


def _plot_scatter(true_xy: np.ndarray, pred_xy: np.ndarray, title: str, path: Path) -> None:
    true_xy = np.asarray(true_xy, dtype=np.float32)
    pred_xy = np.asarray(pred_xy, dtype=np.float32)
    err = np.linalg.norm(pred_xy - true_xy, axis=1)
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


def _plot_success_failure_samples(
    ref_positions: np.ndarray,
    true_xy: np.ndarray,
    pred_xy: np.ndarray,
    title: str,
    path: Path,
) -> None:
    err = np.linalg.norm(np.asarray(pred_xy, dtype=np.float32) - np.asarray(true_xy, dtype=np.float32), axis=1)
    if err.size == 0:
        return
    best_idx = int(np.argmin(err))
    worst_idx = int(np.argmax(err))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax, sample_idx, label in zip(axes, [best_idx, worst_idx], ["Success Sample", "Failure Sample"]):
        ref = np.asarray(ref_positions[sample_idx], dtype=np.float32)
        tgt = np.asarray(true_xy[sample_idx], dtype=np.float32)
        pred = np.asarray(pred_xy[sample_idx], dtype=np.float32)
        ax.scatter(ref[:, 0], ref[:, 1], c="tab:blue", s=50, label="refs")
        ax.scatter([tgt[0]], [tgt[1]], c="tab:green", s=70, label="true")
        ax.scatter([pred[0]], [pred[1]], c="tab:red", s=70, label="pred")
        for ridx, point in enumerate(ref):
            ax.annotate(f"ref{ridx+1}", (float(point[0]), float(point[1])), textcoords="offset points", xytext=(5, 5))
        ax.set_title(f"{label}\nerror={err[sample_idx]:.4f} m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def train_relative_distance_model(
    bundle: RelativeDistanceBundle,
    result_dir: str | Path,
    alpha: float = 1.0,
    beta: float = 1.0,
    lr: float = 1.0e-3,
    batch_size: int = 128,
    epochs: int = 40,
    seed: int = 0,
    device: str = "cpu",
    live_plot: bool = False,
) -> dict[str, Any]:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    set_global_seed(int(seed))

    dev = torch.device(device)
    model = RelativeDistanceMLP().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    train_loader = _make_loader(
        bundle.features_norm,
        bundle.target_r,
        bundle.tdoa_seconds,
        bundle.split_indices["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    history: list[dict[str, Any]] = []
    best_selector: tuple[float, float, float, float] | None = None
    best_epoch = 0
    best_checkpoint_path = result_path / "best_model.pt"

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss_sum = 0.0
        r_loss_sum = 0.0
        geo_loss_sum = 0.0
        batch_count = 0
        for feat_batch, r_batch, tdoa_batch in train_loader:
            feat_batch = feat_batch.to(dev)
            r_batch = r_batch.to(dev)
            tdoa_batch = tdoa_batch.to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred_r = model(feat_batch)
            total_loss, loss_r, loss_geo = hybrid_distance_geo_loss(
                pred_r,
                r_batch,
                tdoa_batch,
                c=bundle.c,
                alpha=alpha,
                beta=beta,
            )
            total_loss.backward()
            optimizer.step()

            total_loss_sum += float(total_loss.item())
            r_loss_sum += float(loss_r.item())
            geo_loss_sum += float(loss_geo.item())
            batch_count += 1

        def _eval_split(split_key: str) -> dict[str, Any]:
            idx = bundle.split_indices[split_key]
            pred_r = _predict_radii(model, bundle.features_norm, idx, dev)
            return evaluate_prediction_block(
                pred_r,
                bundle.target_r[idx],
                bundle.ref_positions[idx],
                bundle.target_xy[idx],
                bundle.tdoa_seconds[idx],
                bundle.c,
            )

        iid_val_metrics = _eval_split("iid_val")
        geom_val_metrics = _eval_split("geom_val")
        row = {
            "epoch": int(epoch),
            "train_total_loss": total_loss_sum / max(batch_count, 1),
            "train_r_loss": r_loss_sum / max(batch_count, 1),
            "train_geo_loss": geo_loss_sum / max(batch_count, 1),
            "iid_val_median_m": float(iid_val_metrics["median_m"]),
            "iid_val_p90_m": float(iid_val_metrics["p90_m"]),
            "geom_val_median_m": float(geom_val_metrics["median_m"]),
            "geom_val_p90_m": float(geom_val_metrics["p90_m"]),
            "iid_val_distance_mse": float(iid_val_metrics["distance_mse"]),
            "geom_val_distance_mse": float(geom_val_metrics["distance_mse"]),
        }
        history.append(row)
        _maybe_live_plot(history, enabled=bool(live_plot))

        selector = _selector_tuple({"iid_val": iid_val_metrics, "geom_val": geom_val_metrics})
        if best_selector is None or selector < best_selector:
            best_selector = selector
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": bundle.feature_mean,
                    "feature_std": bundle.feature_std,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "lr": float(lr),
                    "batch_size": int(batch_size),
                    "epochs": int(epochs),
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "selector": [float(x) for x in selector],
                    "use_true_tdoa": bool(bundle.use_true_tdoa),
                    "h5_path": bundle.h5_path,
                },
                best_checkpoint_path,
            )

    _write_history_csv(history, result_path / "train_history.csv")
    final_fig = _plot_training_curves(history, path=result_path / "loss_curves.png")
    plt.close(final_fig)

    ckpt = torch.load(best_checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    split_metrics: dict[str, Any] = {}
    pred_cache: dict[str, np.ndarray] = {}
    for split_key in ("iid_val", "iid_test", "geom_val", "geom_test"):
        idx = bundle.split_indices[split_key]
        pred_r = _predict_radii(model, bundle.features_norm, idx, dev)
        pred_cache[split_key] = pred_r
        split_metrics[split_key] = evaluate_prediction_block(
            pred_r,
            bundle.target_r[idx],
            bundle.ref_positions[idx],
            bundle.target_xy[idx],
            bundle.tdoa_seconds[idx],
            bundle.c,
        )

    _plot_scatter(
        bundle.target_xy[bundle.split_indices["iid_test"]],
        split_metrics["iid_test"]["pred_xy"],
        "IID Test: Predicted vs True Source Positions",
        result_path / "scatter_iid_test.png",
    )
    _plot_scatter(
        bundle.target_xy[bundle.split_indices["geom_test"]],
        split_metrics["geom_test"]["pred_xy"],
        "Geom Test: Predicted vs True Source Positions",
        result_path / "scatter_geom_test.png",
    )
    _plot_success_failure_samples(
        bundle.ref_positions[bundle.split_indices["iid_test"]],
        bundle.target_xy[bundle.split_indices["iid_test"]],
        split_metrics["iid_test"]["pred_xy"],
        "IID Test Samples",
        result_path / "samples_iid_test.png",
    )
    _plot_success_failure_samples(
        bundle.ref_positions[bundle.split_indices["geom_test"]],
        bundle.target_xy[bundle.split_indices["geom_test"]],
        split_metrics["geom_test"]["pred_xy"],
        "Geom Test Samples",
        result_path / "samples_geom_test.png",
    )

    summary = {
        "h5_path": bundle.h5_path,
        "use_true_tdoa": bool(bundle.use_true_tdoa),
        "fs": int(bundle.fs),
        "c": float(bundle.c),
        "room_size": [float(bundle.room_size[0]), float(bundle.room_size[1])],
        "geometry_filter_mode": str(bundle.geometry_filter_mode),
        "min_triangle_area": float(bundle.min_triangle_area),
        "max_jacobian_condition": float(bundle.max_jacobian_condition),
        "max_triangle_angle_deg": float(bundle.max_triangle_angle_deg),
        "near_ref_inside_threshold_m": float(bundle.near_ref_inside_threshold_m),
        "source_region_rule_version": str(bundle.source_region_rule_version),
        "alpha": float(alpha),
        "beta": float(beta),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "seed": int(seed),
        "split_sizes": bundle.split_sizes,
        "train_overlap_removed": int(bundle.train_overlap_removed),
        "best_epoch": int(best_epoch),
        "best_selector": [float(x) for x in best_selector] if best_selector is not None else None,
        "checkpoint_path": str(best_checkpoint_path),
        "iid_val": {k: float(v) for k, v in split_metrics["iid_val"].items() if k != "pred_xy"},
        "iid_test": {k: float(v) for k, v in split_metrics["iid_test"].items() if k != "pred_xy"},
        "geom_val": {k: float(v) for k, v in split_metrics["geom_val"].items() if k != "pred_xy"},
        "geom_test": {k: float(v) for k, v in split_metrics["geom_test"].items() if k != "pred_xy"},
    }
    _save_json(result_path / "summary.json", summary)
    return summary


def load_summary(result_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(result_dir) / "summary.json").read_text(encoding="utf-8"))


def _describe_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "position_error_m_median": None,
            "position_error_m_p90": None,
            "position_error_m_mean": None,
            "tdoa_abs_err_s_median": None,
            "tdoa_abs_err_s_p90": None,
            "tdoa_abs_err_s_mean": None,
            "success_rate": None,
        }
    pos = np.asarray([float(row["position_error_m"]) for row in rows], dtype=np.float64)
    tdoa = np.asarray([float(row["tdoa_abs_err_s"]) for row in rows], dtype=np.float64)
    success = np.asarray([1.0 if bool(row["success_flag"]) else 0.0 for row in rows], dtype=np.float64)
    return {
        "count": int(pos.size),
        "position_error_m_median": float(np.median(pos)),
        "position_error_m_p90": float(np.quantile(pos, 0.90)),
        "position_error_m_mean": float(np.mean(pos)),
        "tdoa_abs_err_s_median": float(np.median(tdoa)),
        "tdoa_abs_err_s_p90": float(np.quantile(tdoa, 0.90)),
        "tdoa_abs_err_s_mean": float(np.mean(tdoa)),
        "success_rate": float(np.mean(success)),
    }


def _bin_label(left: float, right: float, right_inclusive: bool) -> str:
    suffix = "]" if right_inclusive else ")"
    return f"[{left:.3f}, {right:.3f}{suffix}"


def _binned_stats(
    rows: list[dict[str, Any]],
    metric_key: str,
    bins: tuple[float, ...],
) -> list[dict[str, Any]]:
    if len(bins) < 2:
        raise ValueError("bins must contain at least two values")
    out: list[dict[str, Any]] = []
    for bin_idx in range(len(bins) - 1):
        left = float(bins[bin_idx])
        right = float(bins[bin_idx + 1])
        right_inclusive = bool(bin_idx == len(bins) - 2)
        if right_inclusive:
            selected = [
                row
                for row in rows
                if float(row[metric_key]) >= left and float(row[metric_key]) <= right
            ]
        else:
            selected = [
                row
                for row in rows
                if float(row[metric_key]) >= left and float(row[metric_key]) < right
            ]
        stats = _describe_records(selected)
        stats.update(
            {
                "metric": str(metric_key),
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_label": _bin_label(left, right, right_inclusive),
            }
        )
        out.append(stats)
    return out


def export_inside_outside_diagnostics(
    checkpoint_path: str | Path,
    output_csv: str | Path,
    output_json: str | Path | None = None,
    h5_path: str | Path | None = None,
    use_true_tdoa: bool | None = None,
    success_threshold_m: float = 0.10,
    cond_bins: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 40.0, 1.0e9),
    area_bins: tuple[float, ...] = (0.0, 0.15, 0.25, 0.40, 0.70, 10.0),
    device: str = "cpu",
) -> dict[str, Any]:
    ckpt_path = Path(checkpoint_path).resolve()
    dev = torch.device(device)
    checkpoint = torch.load(str(ckpt_path), map_location=dev, weights_only=False)

    resolved_h5 = Path(h5_path) if h5_path is not None else Path(str(checkpoint.get("h5_path", "")))
    if not resolved_h5.exists():
        raise FileNotFoundError(f"Failed to resolve h5 path: {resolved_h5}")
    resolved_use_true_tdoa = bool(checkpoint.get("use_true_tdoa", True)) if use_true_tdoa is None else bool(use_true_tdoa)
    bundle = build_relative_distance_bundle(resolved_h5, use_true_tdoa=resolved_use_true_tdoa)

    model = RelativeDistanceMLP().to(dev)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    split_order = ("iid_val", "iid_test", "geom_val", "geom_test")
    for split_name in split_order:
        split_indices = np.asarray(bundle.split_indices[split_name], dtype=np.int64)
        pred_r = _predict_radii(model, bundle.features_norm, split_indices, dev)
        pred_xy = reconstruct_xy_from_radii(bundle.ref_positions[split_indices], pred_r)
        true_xy = np.asarray(bundle.target_xy[split_indices], dtype=np.float32)
        position_error = np.linalg.norm(pred_xy - true_xy, axis=1)

        pred_tdoa_seconds = np.stack(
            [
                (pred_r[:, 0] - pred_r[:, 1]) / float(bundle.c),
                (pred_r[:, 1] - pred_r[:, 2]) / float(bundle.c),
                (pred_r[:, 0] - pred_r[:, 2]) / float(bundle.c),
            ],
            axis=1,
        ).astype(np.float32)
        true_tdoa_seconds = np.asarray(bundle.tdoa_seconds[split_indices], dtype=np.float32)
        tdoa_abs_err_s = np.mean(np.abs(pred_tdoa_seconds - true_tdoa_seconds), axis=1)

        split_rows: list[dict[str, Any]] = []
        for local_idx, sample_index in enumerate(split_indices.tolist()):
            tri = np.asarray(bundle.ref_positions[sample_index], dtype=np.float32)
            src = np.asarray(bundle.target_xy[sample_index], dtype=np.float32)
            geom = localization_geometry_metrics(src, tri)
            inside_triangle = bool(float(geom["inside_convex_hull"]) > 0.5)
            row = {
                "split": str(split_name),
                "sample_index": int(sample_index),
                "position_error_m": float(position_error[local_idx]),
                "tdoa_abs_err_s": float(tdoa_abs_err_s[local_idx]),
                "success_flag": bool(float(position_error[local_idx]) <= float(success_threshold_m)),
                "inside_triangle": bool(inside_triangle),
                "outside_triangle": bool(not inside_triangle),
                "triangle_area": float(geom["triangle_area"]),
                "jacobian_condition": float(geom["jacobian_condition"]),
                "small_angle_flag": bool(float(geom["small_angle_flag"]) > 0.5),
                "obtuse_triangle_flag": bool(float(geom["obtuse_triangle_flag"]) > 0.5),
                "min_source_ref_dist": float(geom["min_source_ref_dist"]),
                "max_source_ref_dist": float(geom["max_source_ref_dist"]),
                "centroid_source_dist_norm": float(geom["centroid_source_dist_norm"]),
            }
            split_rows.append(row)

        all_rows.extend(split_rows)

        inside_rows = [row for row in split_rows if bool(row["inside_triangle"])]
        outside_rows = [row for row in split_rows if bool(row["outside_triangle"])]
        outside_success = [row for row in outside_rows if bool(row["success_flag"])]
        outside_failure = [row for row in outside_rows if not bool(row["success_flag"])]

        split_summaries[str(split_name)] = {
            "total": _describe_records(split_rows),
            "inside": _describe_records(inside_rows),
            "outside": _describe_records(outside_rows),
            "outside_success": _describe_records(outside_success),
            "outside_failure": _describe_records(outside_failure),
            "outside_success_count": int(len(outside_success)),
            "outside_failure_count": int(len(outside_failure)),
            "by_jacobian_condition": _binned_stats(split_rows, "jacobian_condition", cond_bins),
            "by_triangle_area": _binned_stats(split_rows, "triangle_area", area_bins),
        }

    fieldnames = [
        "split",
        "sample_index",
        "position_error_m",
        "tdoa_abs_err_s",
        "success_flag",
        "inside_triangle",
        "outside_triangle",
        "triangle_area",
        "jacobian_condition",
        "small_angle_flag",
        "obtuse_triangle_flag",
        "min_source_ref_dist",
        "max_source_ref_dist",
        "centroid_source_dist_norm",
    ]
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    summary = {
        "checkpoint_path": str(ckpt_path),
        "h5_path": str(resolved_h5),
        "use_true_tdoa": bool(resolved_use_true_tdoa),
        "success_threshold_m": float(success_threshold_m),
        "output_csv": str(output_csv_path),
        "split_order": list(split_order),
        "cond_bins": [float(v) for v in cond_bins],
        "area_bins": [float(v) for v in area_bins],
        "splits": split_summaries,
    }

    resolved_json_path = Path(output_json) if output_json is not None else output_csv_path.with_suffix(".summary.json")
    summary["output_json"] = str(resolved_json_path)
    _save_json(resolved_json_path, summary)
    return summary

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def resolve_h5_path(explicit_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    env_path = os.environ.get("ANC_H5_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("python_impl") / "python_scripts" / "cfxlms_qc_dataset_single_control.h5",
            Path("python_impl") / "python_scripts" / "cfxlms_qc_dataset_single_control_500_seeded.h5",
            Path("python_impl") / "python_scripts" / "_tmp_single_control_smoke.h5",
        ]
    )

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
        checked.append(str(resolved))
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Single-control dataset HDF5 was not found. "
        f"Checked: {checked}"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stats_from_train(arr: np.ndarray, train_idx: np.ndarray, eps: float = 1.0e-6) -> dict[str, np.ndarray]:
    train_arr = np.asarray(arr[train_idx], dtype=np.float32)
    mean = train_arr.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_arr.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return {"mean": mean, "std": std}


def standardize(arr: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return ((np.asarray(arr, dtype=np.float32) - stats["mean"]) / stats["std"]).astype(np.float32)


def as_device_stats(stats: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "mean": torch.as_tensor(stats["mean"], dtype=torch.float32, device=device),
        "std": torch.as_tensor(stats["std"], dtype=torch.float32, device=device),
    }


@dataclass
class DatasetBundle:
    x_p: np.ndarray
    x_s: np.ndarray
    y_raw: np.ndarray
    meta: dict[str, Any]


def load_dataset(h5_path: Path) -> DatasetBundle:
    with h5py.File(str(h5_path), "r") as h5:
        gcc = np.asarray(h5["processed/gcc_phat"], dtype=np.float32)
        psd = np.asarray(h5["processed/psd_features"], dtype=np.float32)
        x_p = np.concatenate([gcc, psd[:, None, :]], axis=1).astype(np.float32)
        x_s = np.asarray(h5["processed/path_features"], dtype=np.float32)
        y_raw = np.asarray(h5["processed/w_targets"], dtype=np.float32)
        meta = {
            "h5_path": str(h5_path),
            "config_json": str(h5.attrs["config_json"]),
            "path_feature_dim": int(h5["processed"].attrs["path_feature_dim"]),
            "w_target_dim": int(h5["processed"].attrs["w_target_dim"]),
            "w_keep_len": int(h5["processed"].attrs["w_keep_len"]),
            "path_feature_slices_json": str(h5["processed"].attrs["path_feature_slices_json"]),
            "w_target_slices_json": str(h5["processed"].attrs["w_target_slices_json"]),
            "num_rooms": int(gcc.shape[0]),
            "x_p_shape": tuple(x_p.shape[1:]),
            "x_s_dim": int(x_s.shape[1]),
            "y_dim": int(y_raw.shape[1]),
        }
    return DatasetBundle(x_p=x_p, x_s=x_s, y_raw=y_raw, meta=meta)


class IdentityTargetTransform:
    name = "full"

    def __init__(self, y_train_z: np.ndarray):
        self.output_dim = int(y_train_z.shape[1])

    def transform(self, y_z: np.ndarray) -> np.ndarray:
        return np.asarray(y_z, dtype=np.float32)

    def inverse_to_yz(self, pred: torch.Tensor) -> torch.Tensor:
        return pred

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "output_dim": self.output_dim}


class PCATargetTransform:
    name = "pca"

    def __init__(self, y_train_z: np.ndarray, latent_dim: int):
        y = np.asarray(y_train_z, dtype=np.float32)
        _, _, vt = np.linalg.svd(y, full_matrices=False)
        self.components = vt[:latent_dim].astype(np.float32)
        self.output_dim = int(self.components.shape[0])

    def transform(self, y_z: np.ndarray) -> np.ndarray:
        return np.asarray(y_z, dtype=np.float32) @ self.components.T

    def inverse_to_yz(self, pred: torch.Tensor) -> torch.Tensor:
        comp = torch.as_tensor(self.components, dtype=pred.dtype, device=pred.device)
        return pred @ comp

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_dim": self.output_dim,
            "components": self.components.tolist(),
        }


class PCAFeatureTransform:
    def __init__(self, x_train: np.ndarray, latent_dim: int):
        x = np.asarray(x_train, dtype=np.float32)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        self.components = vt[:latent_dim].astype(np.float32)
        self.output_dim = int(self.components.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) @ self.components.T

    def state_dict(self) -> dict[str, Any]:
        return {
            "output_dim": self.output_dim,
            "components": self.components.tolist(),
        }


class ConvXPEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 24, kernel_size=5, stride=2, padding=2)
        self.gn1 = nn.GroupNorm(1, 24)
        self.conv2 = nn.Conv1d(24, 48, kernel_size=5, stride=2, padding=2)
        self.gn2 = nn.GroupNorm(1, 48)
        self.conv3 = nn.Conv1d(48, 96, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(1, 96)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(96, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.gn1(self.conv1(x)))
        z = F.gelu(self.gn2(self.conv2(z)))
        z = F.gelu(self.gn3(self.conv3(z)))
        z = self.pool(z).squeeze(-1)
        return F.gelu(self.ln(self.fc(z)))


class FlatXPEncoder(nn.Module):
    def __init__(self, width: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(width, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.flatten(start_dim=1)
        z = F.gelu(self.ln1(self.fc1(z)))
        z = self.drop(z)
        z = F.gelu(self.ln2(self.fc2(z)))
        return z


class XSBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ANCRegressor(nn.Module):
    def __init__(
        self,
        x_p_shape: tuple[int, int],
        x_s_dim: int,
        out_dim: int,
        model_type: str,
        feature_mode: str,
        dropout: float,
    ):
        super().__init__()
        self.feature_mode = feature_mode
        xp_dim = 0
        xs_dim = 0

        if feature_mode in ("xp_xs", "xp_only"):
            if model_type == "conv":
                self.xp_branch = ConvXPEncoder(out_dim=128)
            elif model_type == "flat":
                self.xp_branch = FlatXPEncoder(width=int(np.prod(x_p_shape)), out_dim=128, dropout=dropout)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            xp_dim = 128
        else:
            self.xp_branch = None

        if feature_mode in ("xp_xs", "xs_only"):
            self.xs_branch = XSBranch(x_s_dim, hidden_dims=(256, 128), dropout=dropout)
            xs_dim = self.xs_branch.out_dim
        else:
            self.xs_branch = None

        fusion_in = xp_dim + xs_dim
        if fusion_in <= 0:
            raise ValueError("At least one feature branch must be active.")
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
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

    def forward(self, x_p: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.xp_branch is not None:
            parts.append(self.xp_branch(x_p))
        if self.xs_branch is not None:
            parts.append(self.xs_branch(x_s))
        z = torch.cat(parts, dim=1)
        return self.fusion(z)


def build_target_transform(name: str, y_train_z: np.ndarray, latent_dim: int) -> IdentityTargetTransform | PCATargetTransform:
    if name == "full":
        return IdentityTargetTransform(y_train_z)
    if name == "pca":
        return PCATargetTransform(y_train_z, latent_dim=min(latent_dim, y_train_z.shape[1]))
    raise ValueError(f"Unknown target_mode: {name}")


def make_dataloaders(
    x_p: np.ndarray,
    x_s: np.ndarray,
    y_t: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(x_p[train_idx]),
        torch.from_numpy(x_s[train_idx]),
        torch.from_numpy(y_t[train_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_p[val_idx]),
        torch.from_numpy(x_s[val_idx]),
        torch.from_numpy(y_t[val_idx]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def mse_mean_predictor(y_target: np.ndarray, idx: np.ndarray) -> float:
    arr = np.asarray(y_target[idx], dtype=np.float32)
    return float(np.mean(arr**2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-control ANC experiments from shell.")
    parser.add_argument("--h5-path", type=str, default=None, help="Path to single-control dataset HDF5.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save logs and checkpoints.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--model-type", choices=("conv", "flat"), default="conv")
    parser.add_argument("--feature-mode", choices=("xp_xs", "xs_only", "xp_only"), default="xp_xs")
    parser.add_argument("--target-mode", choices=("full", "pca"), default="full")
    parser.add_argument("--target-latent-dim", type=int, default=64)
    parser.add_argument("--x-s-pca-dim", type=int, default=0)
    parser.add_argument("--main-loss-weight", type=float, default=1.0)
    parser.add_argument("--raw-loss-weight", type=float, default=0.10)
    parser.add_argument("--monitor", choices=("loss", "raw"), default="loss")
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--note", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))

    h5_path = resolve_h5_path(args.h5_path)
    bundle = load_dataset(h5_path)
    n = int(bundle.x_p.shape[0])
    indices = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(indices)
    split = int(n * (1.0 - float(args.val_frac)))
    train_idx = np.sort(indices[:split])
    val_idx = np.sort(indices[split:])

    x_p_stats = stats_from_train(bundle.x_p, train_idx)
    x_s_stats = stats_from_train(bundle.x_s, train_idx)
    y_stats = stats_from_train(bundle.y_raw, train_idx)
    x_p_z = standardize(bundle.x_p, x_p_stats)
    x_s_z = standardize(bundle.x_s, x_s_stats)
    x_s_transform = None
    if int(args.x_s_pca_dim) > 0:
        x_s_transform = PCAFeatureTransform(x_s_z[train_idx], latent_dim=min(int(args.x_s_pca_dim), x_s_z.shape[1]))
        x_s_z = x_s_transform.transform(x_s_z)
    y_z = standardize(bundle.y_raw, y_stats)
    target_transform = build_target_transform(str(args.target_mode), y_z[train_idx], int(args.target_latent_dim))
    y_t = target_transform.transform(y_z)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"Using device: {device}")
    print(f"HDF5: {h5_path}")
    print(f"Samples: {n}, train={len(train_idx)}, val={len(val_idx)}")
    print(f"x_p shape={bundle.x_p.shape}, x_s shape={bundle.x_s.shape}, y_raw shape={bundle.y_raw.shape}")
    if x_s_transform is not None:
        print(f"x_s PCA dim={x_s_transform.output_dim}")
    print(f"target mode={target_transform.name}, output_dim={target_transform.output_dim}")

    train_loader, val_loader = make_dataloaders(
        x_p=x_p_z,
        x_s=x_s_z,
        y_t=y_t,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    model = ANCRegressor(
        x_p_shape=tuple(bundle.meta["x_p_shape"]),
        x_s_dim=int(x_s_z.shape[1]),
        out_dim=int(target_transform.output_dim),
        model_type=str(args.model_type),
        feature_mode=str(args.feature_mode),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=8, factor=0.5)

    y_stats_t = as_device_stats(y_stats, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    best_ckpt = out_dir / "best.pt"
    summary_path = out_dir / "summary.json"

    val_target_baseline = mse_mean_predictor(y_t, val_idx)
    best_metric = float("inf")
    history: list[dict[str, Any]] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "train_raw_mse",
                "val_raw_mse",
                "lr",
                "seconds",
            ],
        )
        writer.writeheader()

        for epoch in range(1, int(args.epochs) + 1):
            t0 = time.time()
            model.train()
            train_loss_sum = 0.0
            train_raw_sum = 0.0
            train_count = 0
            for x_p_b, x_s_b, y_t_b in train_loader:
                x_p_b = x_p_b.to(device=device, dtype=torch.float32, non_blocking=True)
                x_s_b = x_s_b.to(device=device, dtype=torch.float32, non_blocking=True)
                y_t_b = y_t_b.to(device=device, dtype=torch.float32, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model(x_p_b, x_s_b)
                pred_yz = target_transform.inverse_to_yz(pred)
                true_yz = target_transform.inverse_to_yz(y_t_b)
                pred_raw = pred_yz * y_stats_t["std"] + y_stats_t["mean"]
                true_raw = true_yz * y_stats_t["std"] + y_stats_t["mean"]
                loss_main = F.mse_loss(pred, y_t_b)
                loss_raw = F.mse_loss(pred_raw, true_raw)
                loss = float(args.main_loss_weight) * loss_main + float(args.raw_loss_weight) * loss_raw
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.clip_grad))
                optimizer.step()

                bs = int(x_p_b.shape[0])
                train_loss_sum += float(loss.detach().cpu()) * bs
                train_raw_sum += float(loss_raw.detach().cpu()) * bs
                train_count += bs

            train_loss = train_loss_sum / max(train_count, 1)
            train_raw_mse = train_raw_sum / max(train_count, 1)

            model.eval()
            val_loss_sum = 0.0
            val_raw_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for x_p_b, x_s_b, y_t_b in val_loader:
                    x_p_b = x_p_b.to(device=device, dtype=torch.float32, non_blocking=True)
                    x_s_b = x_s_b.to(device=device, dtype=torch.float32, non_blocking=True)
                    y_t_b = y_t_b.to(device=device, dtype=torch.float32, non_blocking=True)
                    pred = model(x_p_b, x_s_b)
                    pred_yz = target_transform.inverse_to_yz(pred)
                    true_yz = target_transform.inverse_to_yz(y_t_b)
                    pred_raw = pred_yz * y_stats_t["std"] + y_stats_t["mean"]
                    true_raw = true_yz * y_stats_t["std"] + y_stats_t["mean"]
                    loss_main = F.mse_loss(pred, y_t_b)
                    loss_raw = F.mse_loss(pred_raw, true_raw)
                    loss = float(args.main_loss_weight) * loss_main + float(args.raw_loss_weight) * loss_raw
                    bs = int(x_p_b.shape[0])
                    val_loss_sum += float(loss.detach().cpu()) * bs
                    val_raw_sum += float(loss_raw.detach().cpu()) * bs
                    val_count += bs

            val_loss = val_loss_sum / max(val_count, 1)
            val_raw_mse = val_raw_sum / max(val_count, 1)
            scheduler.step(val_loss)
            monitor_value = val_loss if str(args.monitor) == "loss" else val_raw_mse

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_raw_mse": train_raw_mse,
                "val_raw_mse": val_raw_mse,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.time() - t0,
            }
            writer.writerow(row)
            f_csv.flush()
            history.append(row)
            print(
                f"[Epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} "
                f"train_raw={train_raw_mse:.6f} val_raw={val_raw_mse:.6f} "
                f"lr={row['lr']:.3e} t={row['seconds']:.1f}s"
            )

            if monitor_value < best_metric:
                best_metric = monitor_value
                torch.save(
                    {
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "args": vars(args),
                        "dataset_meta": bundle.meta,
                        "train_indices": train_idx.tolist(),
                        "val_indices": val_idx.tolist(),
                        "x_p_stats": x_p_stats,
                        "x_s_stats": x_s_stats,
                        "y_stats": y_stats,
                        "x_s_transform": None if x_s_transform is None else x_s_transform.state_dict(),
                        "target_transform": target_transform.state_dict(),
                        "best_metric": best_metric,
                        "monitor": str(args.monitor),
                        "history_tail": history[-10:],
                    },
                    best_ckpt,
                )

    summary = {
        "args": vars(args),
        "dataset_meta": bundle.meta,
        "best_metric": best_metric,
        "final_val": history[-1]["val_loss"] if history else None,
        "final_val_raw_mse": history[-1]["val_raw_mse"] if history else None,
        "val_target_baseline": val_target_baseline,
        "best_epoch": int(min(history, key=lambda x: x["val_loss"] if str(args.monitor) == "loss" else x["val_raw_mse"])["epoch"]) if history else None,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics to {csv_path}")
    print(f"Saved best checkpoint to {best_ckpt}")
    print(f"Saved summary to {summary_path}")
    print(f"Val mean-predictor baseline (target space MSE): {val_target_baseline:.6f}")
    print(f"Best monitored metric ({args.monitor}): {best_metric:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

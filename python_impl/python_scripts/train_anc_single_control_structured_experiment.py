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


def build_direct_laguerre_basis(tap_len: int, direct_taps: int, laguerre_dim: int, pole: float) -> np.ndarray:
    direct_taps = int(direct_taps)
    laguerre_dim = int(laguerre_dim)
    tap_len = int(tap_len)
    pole = float(pole)
    if tap_len <= 0:
        raise ValueError("tap_len must be positive.")
    if direct_taps < 0 or laguerre_dim <= 0:
        raise ValueError("direct_taps must be >= 0 and laguerre_dim must be > 0.")
    if direct_taps >= tap_len:
        raise ValueError("direct_taps must be smaller than tap_len.")
    if not (0.0 < pole < 1.0):
        raise ValueError("Laguerre pole must lie in (0, 1).")

    n = np.arange(tap_len, dtype=np.float64)
    cols: list[np.ndarray] = []
    for idx in range(direct_taps):
        e = np.zeros(tap_len, dtype=np.float64)
        e[idx] = 1.0
        cols.append(e)

    lag = np.zeros((tap_len, laguerre_dim), dtype=np.float64)
    lag[:, 0] = np.sqrt(1.0 - pole * pole) * (pole ** n)
    for k in range(1, laguerre_dim):
        lag[0, k] = np.sqrt(1.0 - pole * pole) * ((-pole) ** k)
        for t in range(1, tap_len):
            lag[t, k] = pole * lag[t - 1, k] + lag[t - 1, k - 1] - pole * lag[t, k - 1]
    for k in range(laguerre_dim):
        cols.append(lag[:, k])

    basis = np.stack(cols, axis=1)
    q, _ = np.linalg.qr(basis)
    return q.astype(np.float32)


@dataclass
class StructuredBundle:
    x_p: np.ndarray
    s_err: np.ndarray
    e2r: np.ndarray
    s2r: np.ndarray
    y: np.ndarray
    meta: dict[str, Any]


def load_structured_dataset(h5_path: Path, input_path_mode: str) -> StructuredBundle:
    with h5py.File(str(h5_path), "r") as h5:
        gcc = np.asarray(h5["processed/gcc_phat"], dtype=np.float32)
        psd = np.asarray(h5["processed/psd_features"], dtype=np.float32)
        x_p = np.concatenate([gcc, psd[:, None, :]], axis=1).astype(np.float32)
        y = np.asarray(h5["processed/w_targets"], dtype=np.float32)

        s_keep = int(h5["processed"].attrs["s_keep_len"])
        e2r_keep = int(h5["processed"].attrs["e2r_keep_len"])
        s2r_keep = int(h5["processed"].attrs["s2r_keep_len"])
        w_keep = int(h5["processed"].attrs["w_keep_len"])
        num_refs = int(json.loads(h5.attrs["config_json"])["num_reference_mics"])

        if input_path_mode == "processed":
            path = np.asarray(h5["processed/path_features"], dtype=np.float32)
            s_err = path[:, :s_keep][:, None, :]
            e2r = path[:, s_keep : s_keep + num_refs * e2r_keep].reshape(-1, num_refs, e2r_keep)
            s2r = path[:, s_keep + num_refs * e2r_keep :].reshape(-1, num_refs, s2r_keep)
        elif input_path_mode == "raw":
            s_err = np.asarray(h5["raw/S_paths"], dtype=np.float32)[:, None, :]
            e2r = np.asarray(h5["raw/E2R_paths"], dtype=np.float32)
            s2r = np.asarray(h5["raw/S2R_paths"], dtype=np.float32)
        else:
            raise ValueError(f"Unknown input_path_mode: {input_path_mode}")
        y = y.reshape(-1, num_refs, w_keep)

        meta = {
            "h5_path": str(h5_path),
            "config_json": str(h5.attrs["config_json"]),
            "num_refs": num_refs,
            "input_path_mode": input_path_mode,
            "s_keep_len": s_keep,
            "e2r_keep_len": e2r_keep,
            "s2r_keep_len": s2r_keep,
            "w_keep_len": w_keep,
        }
    return StructuredBundle(x_p=x_p, s_err=s_err, e2r=e2r, s2r=s2r, y=y, meta=meta)


class TraceEncoder(nn.Module):
    def __init__(self, out_dim: int = 48):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.gn1 = nn.GroupNorm(1, 16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.gn2 = nn.GroupNorm(1, 32)
        self.conv3 = nn.Conv1d(32, 48, kernel_size=3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(1, 48)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.gn1(self.conv1(x)))
        z = F.gelu(self.gn2(self.conv2(z)))
        z = F.gelu(self.gn3(self.conv3(z)))
        z = self.pool(z).squeeze(-1)
        return F.gelu(self.ln(self.fc(z)))


class PrimaryFeatureEncoder(nn.Module):
    def __init__(self, num_refs: int, gcc_dim: int = 48, psd_dim: int = 32, global_dim: int = 64):
        super().__init__()
        self.num_refs = int(num_refs)
        self.gcc_encoder = TraceEncoder(out_dim=gcc_dim)
        self.psd_encoder = TraceEncoder(out_dim=psd_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(gcc_dim + psd_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.GELU(),
        )
        self.out_dim = int(gcc_dim + global_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = int(x.shape[0])
        gcc = x[:, : self.num_refs, :]
        psd = x[:, self.num_refs : self.num_refs + 1, :]
        z_gcc = self.gcc_encoder(gcc.reshape(batch * self.num_refs, 1, gcc.shape[-1])).reshape(batch, self.num_refs, -1)
        z_psd = self.psd_encoder(psd)
        z_global = self.global_mlp(torch.cat([z_gcc.mean(dim=1), z_psd], dim=-1))
        z_global = z_global[:, None, :].expand(-1, self.num_refs, -1)
        return torch.cat([z_gcc, z_global], dim=-1)


class RefDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuredANCNet(nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_refs: int,
        dropout: float = 0.15,
        use_e2r: bool = True,
        use_s2r: bool = True,
        primary_gcc_dim: int = 48,
        primary_psd_dim: int = 32,
        primary_global_dim: int = 64,
        path_embed_dim: int = 32,
        decoder_hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_refs = int(num_refs)
        self.out_dim = int(out_dim)
        self.use_e2r = bool(use_e2r)
        self.use_s2r = bool(use_s2r)
        self.primary_encoder = PrimaryFeatureEncoder(
            num_refs=self.num_refs,
            gcc_dim=int(primary_gcc_dim),
            psd_dim=int(primary_psd_dim),
            global_dim=int(primary_global_dim),
        )
        self.s_err_encoder = TraceEncoder(out_dim=int(path_embed_dim))
        self.e2r_encoder = TraceEncoder(out_dim=int(path_embed_dim)) if self.use_e2r else None
        self.s2r_encoder = TraceEncoder(out_dim=int(path_embed_dim)) if self.use_s2r else None
        ref_in_dim = int(
            self.primary_encoder.out_dim
            + int(path_embed_dim)
            + (int(path_embed_dim) if self.use_e2r else 0)
            + (int(path_embed_dim) if self.use_s2r else 0)
        )
        self.ref_decoder = RefDecoder(
            in_dim=ref_in_dim,
            out_dim=out_dim,
            hidden_dim=int(decoder_hidden_dim),
            dropout=dropout,
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

    def forward(self, x_p: torch.Tensor, s_err: torch.Tensor, e2r: torch.Tensor, s2r: torch.Tensor) -> torch.Tensor:
        batch = int(x_p.shape[0])
        z_primary = self.primary_encoder(x_p)
        z_s_err = self.s_err_encoder(s_err)

        z_s_err_rep = z_s_err[:, None, :].expand(-1, self.num_refs, -1)
        ref_parts = [z_primary, z_s_err_rep]
        if self.use_e2r and self.e2r_encoder is not None:
            e2r_flat = e2r.reshape(batch * self.num_refs, 1, e2r.shape[-1])
            z_e2r = self.e2r_encoder(e2r_flat).reshape(batch, self.num_refs, -1)
            ref_parts.append(z_e2r)
        if self.use_s2r and self.s2r_encoder is not None:
            s2r_flat = s2r.reshape(batch * self.num_refs, 1, s2r.shape[-1])
            z_s2r = self.s2r_encoder(s2r_flat).reshape(batch, self.num_refs, -1)
            ref_parts.append(z_s2r)
        ref_context = torch.cat(ref_parts, dim=-1)
        pred = self.ref_decoder(ref_context.reshape(batch * self.num_refs, -1))
        return pred.reshape(batch, self.num_refs, self.out_dim)


def weighted_tap_mse(pred: torch.Tensor, target: torch.Tensor, tap_weights: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    return torch.mean(diff2 * tap_weights[None, None, :])


def raw_mean_predictor_mse(y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> float:
    y_mean = np.asarray(y[train_idx], dtype=np.float32).mean(axis=0, dtype=np.float64).astype(np.float32)
    return float(np.mean((np.asarray(y[val_idx], dtype=np.float32) - y_mean[None, :, :]) ** 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured non-PCA single-control ANC experiment.")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--input-path-mode", choices=("processed", "raw"), default="processed")
    parser.add_argument("--target-mode", choices=("direct", "direct_factorized", "direct_laguerre"), default="direct")
    parser.add_argument("--basis-direct-taps", type=int, default=16)
    parser.add_argument("--basis-laguerre-dim", type=int, default=16)
    parser.add_argument("--basis-pole", type=float, default=0.55)
    parser.add_argument("--disable-e2r", action="store_true")
    parser.add_argument("--disable-s2r", action="store_true")
    parser.add_argument("--coeff-loss-weight", type=float, default=1.0)
    parser.add_argument("--shape-loss-weight", type=float, default=1.0)
    parser.add_argument("--gain-loss-weight", type=float, default=0.25)
    parser.add_argument("--raw-loss-weight", type=float, default=0.0)
    parser.add_argument("--weighted-loss", action="store_true", help="Use tap-energy-weighted raw MSE.")
    parser.add_argument("--weight-power", type=float, default=0.5)
    parser.add_argument("--weight-min", type=float, default=0.5)
    parser.add_argument("--weight-max", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--primary-gcc-dim", type=int, default=48)
    parser.add_argument("--primary-psd-dim", type=int, default=32)
    parser.add_argument("--primary-global-dim", type=int, default=64)
    parser.add_argument("--path-embed-dim", type=int, default=32)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    h5_path = resolve_h5_path(args.h5_path)
    bundle = load_structured_dataset(h5_path, input_path_mode=str(args.input_path_mode))
    n = int(bundle.x_p.shape[0])
    indices = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(args.seed))
    rng.shuffle(indices)
    split = int(n * (1.0 - float(args.val_frac)))
    train_idx = np.sort(indices[:split])
    val_idx = np.sort(indices[split:])

    x_p_stats = stats_general(bundle.x_p, train_idx, reduce_axes=(0,))
    s_err_stats = stats_general(bundle.s_err, train_idx, reduce_axes=(0,))
    e2r_stats = stats_general(bundle.e2r, train_idx, reduce_axes=(0, 1))
    s2r_stats = stats_general(bundle.s2r, train_idx, reduce_axes=(0, 1))
    x_p_z = standardize(bundle.x_p, x_p_stats["mean"], x_p_stats["std"])
    s_err_z = standardize(bundle.s_err, s_err_stats["mean"], s_err_stats["std"])
    e2r_z = standardize(bundle.e2r, e2r_stats["mean"][None, :], e2r_stats["std"][None, :])
    s2r_z = standardize(bundle.s2r, s2r_stats["mean"][None, :], s2r_stats["std"][None, :])

    basis_matrix: np.ndarray | None = None
    factorized_output = str(args.target_mode) == "direct_factorized"
    target_is_basis = str(args.target_mode) == "direct_laguerre"
    if target_is_basis:
        basis_matrix = build_direct_laguerre_basis(
            tap_len=int(bundle.meta["w_keep_len"]),
            direct_taps=int(args.basis_direct_taps),
            laguerre_dim=int(args.basis_laguerre_dim),
            pole=float(args.basis_pole),
        )
        coeff_targets = np.einsum("brt,tk->brk", np.asarray(bundle.y, dtype=np.float32), basis_matrix, optimize=True).astype(np.float32)
        y_stats = stats_general(coeff_targets, train_idx, reduce_axes=(0, 1))
        y_z = standardize(coeff_targets, y_stats["mean"][None, :], y_stats["std"][None, :])
        target_dim = int(coeff_targets.shape[-1])
    elif factorized_output:
        y_scale = np.sqrt(np.mean(np.asarray(bundle.y, dtype=np.float32) ** 2, axis=-1, keepdims=True))
        y_scale = np.maximum(y_scale, np.float32(1.0e-6))
        y_shape = (np.asarray(bundle.y, dtype=np.float32) / y_scale).astype(np.float32)
        y_gain = np.log(y_scale).astype(np.float32)
        shape_stats = stats_general(y_shape, train_idx, reduce_axes=(0, 1))
        gain_stats = stats_general(y_gain, train_idx, reduce_axes=(0, 1))
        y_shape_z = standardize(y_shape, shape_stats["mean"][None, :], shape_stats["std"][None, :])
        y_gain_z = standardize(y_gain, gain_stats["mean"], gain_stats["std"])
        y_z = np.concatenate([y_shape_z, y_gain_z], axis=-1).astype(np.float32)
        y_stats = {
            "shape_mean": shape_stats["mean"],
            "shape_std": shape_stats["std"],
            "gain_mean": gain_stats["mean"],
            "gain_std": gain_stats["std"],
        }
        target_dim = int(bundle.meta["w_keep_len"] + 1)
    else:
        y_stats = stats_general(bundle.y, train_idx, reduce_axes=(0, 1))
        y_z = standardize(bundle.y, y_stats["mean"][None, :], y_stats["std"][None, :])
        target_dim = int(bundle.meta["w_keep_len"])

    tap_rms = np.sqrt(np.mean(np.asarray(bundle.y[train_idx], dtype=np.float32) ** 2, axis=(0, 1)))
    tap_weights = np.power(np.maximum(tap_rms, np.finfo(np.float32).eps), float(args.weight_power))
    tap_weights = tap_weights / float(np.mean(tap_weights))
    tap_weights = np.clip(tap_weights, float(args.weight_min), float(args.weight_max)).astype(np.float32)
    baseline_raw = raw_mean_predictor_mse(bundle.y, train_idx, val_idx)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    print(f"Using device: {device}")
    print(f"HDF5: {h5_path}")
    print(f"Samples: {n}, train={len(train_idx)}, val={len(val_idx)}")
    print(f"x_p={bundle.x_p.shape}, s_err={bundle.s_err.shape}, e2r={bundle.e2r.shape}, s2r={bundle.s2r.shape}, y={bundle.y.shape}")
    print(
        f"target_mode={args.target_mode}, target_dim={target_dim}, "
        f"use_e2r={not bool(args.disable_e2r)}, use_s2r={not bool(args.disable_s2r)}, "
        f"weighted_loss={bool(args.weighted_loss)}, baseline_raw_mse={baseline_raw:.6e}"
    )

    train_ds = TensorDataset(
        torch.from_numpy(x_p_z[train_idx]),
        torch.from_numpy(s_err_z[train_idx]),
        torch.from_numpy(e2r_z[train_idx]),
        torch.from_numpy(s2r_z[train_idx]),
        torch.from_numpy(y_z[train_idx]),
        torch.from_numpy(bundle.y[train_idx]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_p_z[val_idx]),
        torch.from_numpy(s_err_z[val_idx]),
        torch.from_numpy(e2r_z[val_idx]),
        torch.from_numpy(s2r_z[val_idx]),
        torch.from_numpy(y_z[val_idx]),
        torch.from_numpy(bundle.y[val_idx]),
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = StructuredANCNet(
        out_dim=target_dim,
        num_refs=int(bundle.meta["num_refs"]),
        dropout=float(args.dropout),
        use_e2r=not bool(args.disable_e2r),
        use_s2r=not bool(args.disable_s2r),
        primary_gcc_dim=int(args.primary_gcc_dim),
        primary_psd_dim=int(args.primary_psd_dim),
        primary_global_dim=int(args.primary_global_dim),
        path_embed_dim=int(args.path_embed_dim),
        decoder_hidden_dim=int(args.decoder_hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=6, factor=0.5)
    if factorized_output:
        shape_mean_t = torch.as_tensor(y_stats["shape_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
        shape_std_t = torch.as_tensor(y_stats["shape_std"], dtype=torch.float32, device=device).view(1, 1, -1)
        gain_mean_t = torch.as_tensor(y_stats["gain_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
        gain_std_t = torch.as_tensor(y_stats["gain_std"], dtype=torch.float32, device=device).view(1, 1, -1)
        y_mean_t = None
        y_std_t = None
    else:
        y_mean_t = torch.as_tensor(y_stats["mean"], dtype=torch.float32, device=device).view(1, 1, -1)
        y_std_t = torch.as_tensor(y_stats["std"], dtype=torch.float32, device=device).view(1, 1, -1)
    tap_weights_t = torch.as_tensor(tap_weights, dtype=torch.float32, device=device)
    basis_t = torch.as_tensor(basis_matrix, dtype=torch.float32, device=device) if basis_matrix is not None else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    best_ckpt = out_dir / "best.pt"
    summary_path = out_dir / "summary.json"

    best_val_raw = float("inf")
    history: list[dict[str, Any]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["epoch", "train_loss", "val_loss", "train_raw_mse", "val_raw_mse", "lr", "seconds"],
        )
        writer.writeheader()

        for epoch in range(1, int(args.epochs) + 1):
            t0 = time.time()
            model.train()
            train_loss_sum = 0.0
            train_raw_sum = 0.0
            train_count = 0
            for x_p_b, s_err_b, e2r_b, s2r_b, y_z_b, y_raw_b in train_loader:
                x_p_b = x_p_b.to(device=device, dtype=torch.float32, non_blocking=True)
                s_err_b = s_err_b.to(device=device, dtype=torch.float32, non_blocking=True)
                e2r_b = e2r_b.to(device=device, dtype=torch.float32, non_blocking=True)
                s2r_b = s2r_b.to(device=device, dtype=torch.float32, non_blocking=True)
                y_z_b = y_z_b.to(device=device, dtype=torch.float32, non_blocking=True)
                y_raw_b = y_raw_b.to(device=device, dtype=torch.float32, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred_z = model(x_p_b, s_err_b, e2r_b, s2r_b)
                if factorized_output:
                    pred_shape_z = pred_z[:, :, :-1]
                    pred_gain_z = pred_z[:, :, -1:]
                    tgt_shape_z = y_z_b[:, :, :-1]
                    tgt_gain_z = y_z_b[:, :, -1:]
                    pred_shape = pred_shape_z * shape_std_t + shape_mean_t
                    pred_gain = pred_gain_z * gain_std_t + gain_mean_t
                    pred_raw = torch.exp(pred_gain) * pred_shape
                    shape_loss = F.mse_loss(pred_shape_z, tgt_shape_z)
                    gain_loss = F.mse_loss(pred_gain_z, tgt_gain_z)
                    raw_loss = weighted_tap_mse(pred_raw, y_raw_b, tap_weights_t) if args.weighted_loss else F.mse_loss(pred_raw, y_raw_b)
                    loss = float(args.shape_loss_weight) * shape_loss + float(args.gain_loss_weight) * gain_loss + float(args.raw_loss_weight) * raw_loss
                else:
                    pred_target = pred_z * y_std_t + y_mean_t
                    if basis_t is not None:
                        pred_raw = torch.einsum("brk,tk->brt", pred_target, basis_t)
                    else:
                        pred_raw = pred_target
                    raw_loss = weighted_tap_mse(pred_raw, y_raw_b, tap_weights_t) if args.weighted_loss else F.mse_loss(pred_raw, y_raw_b)
                    if basis_t is not None:
                        coeff_loss = F.mse_loss(pred_z, y_z_b)
                        loss = float(args.coeff_loss_weight) * coeff_loss + float(args.raw_loss_weight) * raw_loss
                    else:
                        loss = raw_loss
                raw_mse = F.mse_loss(pred_raw, y_raw_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = int(x_p_b.shape[0])
                train_loss_sum += float(loss.detach().cpu()) * bs
                train_raw_sum += float(raw_mse.detach().cpu()) * bs
                train_count += bs

            train_loss = train_loss_sum / max(train_count, 1)
            train_raw_mse = train_raw_sum / max(train_count, 1)

            model.eval()
            val_loss_sum = 0.0
            val_raw_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for x_p_b, s_err_b, e2r_b, s2r_b, y_z_b, y_raw_b in val_loader:
                    x_p_b = x_p_b.to(device=device, dtype=torch.float32, non_blocking=True)
                    s_err_b = s_err_b.to(device=device, dtype=torch.float32, non_blocking=True)
                    e2r_b = e2r_b.to(device=device, dtype=torch.float32, non_blocking=True)
                    s2r_b = s2r_b.to(device=device, dtype=torch.float32, non_blocking=True)
                    y_z_b = y_z_b.to(device=device, dtype=torch.float32, non_blocking=True)
                    y_raw_b = y_raw_b.to(device=device, dtype=torch.float32, non_blocking=True)
                    pred_z = model(x_p_b, s_err_b, e2r_b, s2r_b)
                    if factorized_output:
                        pred_shape_z = pred_z[:, :, :-1]
                        pred_gain_z = pred_z[:, :, -1:]
                        tgt_shape_z = y_z_b[:, :, :-1]
                        tgt_gain_z = y_z_b[:, :, -1:]
                        pred_shape = pred_shape_z * shape_std_t + shape_mean_t
                        pred_gain = pred_gain_z * gain_std_t + gain_mean_t
                        pred_raw = torch.exp(pred_gain) * pred_shape
                        shape_loss = F.mse_loss(pred_shape_z, tgt_shape_z)
                        gain_loss = F.mse_loss(pred_gain_z, tgt_gain_z)
                        raw_loss = weighted_tap_mse(pred_raw, y_raw_b, tap_weights_t) if args.weighted_loss else F.mse_loss(pred_raw, y_raw_b)
                        loss = float(args.shape_loss_weight) * shape_loss + float(args.gain_loss_weight) * gain_loss + float(args.raw_loss_weight) * raw_loss
                    else:
                        pred_target = pred_z * y_std_t + y_mean_t
                        if basis_t is not None:
                            pred_raw = torch.einsum("brk,tk->brt", pred_target, basis_t)
                        else:
                            pred_raw = pred_target
                        raw_loss = weighted_tap_mse(pred_raw, y_raw_b, tap_weights_t) if args.weighted_loss else F.mse_loss(pred_raw, y_raw_b)
                        if basis_t is not None:
                            coeff_loss = F.mse_loss(pred_z, y_z_b)
                            loss = float(args.coeff_loss_weight) * coeff_loss + float(args.raw_loss_weight) * raw_loss
                        else:
                            loss = raw_loss
                    raw_mse = F.mse_loss(pred_raw, y_raw_b)
                    bs = int(x_p_b.shape[0])
                    val_loss_sum += float(loss.detach().cpu()) * bs
                    val_raw_sum += float(raw_mse.detach().cpu()) * bs
                    val_count += bs

            val_loss = val_loss_sum / max(val_count, 1)
            val_raw_mse = val_raw_sum / max(val_count, 1)
            scheduler.step(val_raw_mse)

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
                f"[Epoch {epoch:03d}] train_loss={train_loss:.6e} val_loss={val_loss:.6e} "
                f"train_raw={train_raw_mse:.6e} val_raw={val_raw_mse:.6e} lr={row['lr']:.3e} t={row['seconds']:.1f}s"
            )

            if val_raw_mse < best_val_raw:
                best_val_raw = val_raw_mse
                torch.save(
                    {
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "args": vars(args),
                        "dataset_meta": bundle.meta,
                        "train_indices": train_idx.tolist(),
                        "val_indices": val_idx.tolist(),
                        "x_p_stats": x_p_stats,
                        "s_err_stats": s_err_stats,
                        "e2r_stats": e2r_stats,
                        "s2r_stats": s2r_stats,
                        "y_stats": y_stats,
                        "tap_weights": tap_weights.tolist(),
                        "target_mode": str(args.target_mode),
                        "basis_matrix": basis_matrix,
                        "best_val_raw_mse": best_val_raw,
                    },
                    best_ckpt,
                )

    summary = {
        "args": vars(args),
        "dataset_meta": bundle.meta,
        "baseline_raw_mse": baseline_raw,
        "best_val_raw_mse": best_val_raw,
        "best_epoch": int(min(history, key=lambda x: x["val_raw_mse"])["epoch"]) if history else None,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics to {csv_path}")
    print(f"Saved best checkpoint to {best_ckpt}")
    print(f"Saved summary to {summary_path}")
    print(f"Best val_raw_mse: {best_val_raw:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

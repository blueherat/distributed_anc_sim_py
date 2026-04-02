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
from python_scripts.cfxlms_single_control_dataset_impl import (
    AcousticScenarioSampler,
    DatasetBuildConfig,
    _canonical_q_from_paths,
    _cfxlms_with_init,
    _normalize_columns,
    _rolling_mse_db,
    _solve_w_canonical_from_q,
    compute_processed_features,
)


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


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(n), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    split = int(round(int(n) * (1.0 - float(val_frac))))
    split = min(max(split, 1), int(n) - 1)
    return np.sort(indices[:split]), np.sort(indices[split:])


def standardize(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(arr, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)).astype(np.float32)


def stats_general(arr: np.ndarray, train_idx: np.ndarray, reduce_axes: tuple[int, ...], eps: float = 1.0e-6) -> dict[str, np.ndarray]:
    train_arr = np.asarray(arr[train_idx], dtype=np.float32)
    mean = train_arr.mean(axis=reduce_axes, dtype=np.float64).astype(np.float32)
    std = train_arr.std(axis=reduce_axes, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return {"mean": mean, "std": std}


def signed_log1p(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)


def build_laguerre_basis(tap_len: int, basis_dim: int, pole: float) -> np.ndarray:
    tap_len = int(tap_len)
    basis_dim = int(basis_dim)
    pole = float(pole)
    if tap_len <= 0 or basis_dim <= 0:
        raise ValueError("tap_len and basis_dim must both be positive.")
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


def window_mask_1d(anchor_idx: np.ndarray, tap_len: int, half_width: int) -> np.ndarray:
    anchor = np.asarray(anchor_idx, dtype=np.int64).reshape(-1)
    mask = np.zeros((anchor.shape[0], int(tap_len)), dtype=bool)
    rows = np.arange(anchor.shape[0], dtype=np.int64)
    for delta in range(-int(half_width), int(half_width) + 1):
        cols = anchor + int(delta)
        valid = (cols >= 0) & (cols < int(tap_len))
        mask[rows[valid], cols[valid]] = True
    return mask


def extract_local_windows_1d(q: np.ndarray, anchor_idx: np.ndarray, half_width: int) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float32)
    anchor = np.asarray(anchor_idx, dtype=np.int64).reshape(-1)
    win_len = 2 * int(half_width) + 1
    out = np.zeros((arr.shape[0], win_len), dtype=np.float32)
    rows = np.arange(arr.shape[0], dtype=np.int64)
    for col, delta in enumerate(range(-int(half_width), int(half_width) + 1)):
        taps = anchor + int(delta)
        valid = (taps >= 0) & (taps < int(arr.shape[-1]))
        out[rows[valid], col] = arr[rows[valid], taps[valid]]
    return out


def project_tail_coeffs(q: np.ndarray, anchor_idx: np.ndarray, basis: np.ndarray, half_width: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(q, dtype=np.float32)
    mask = window_mask_1d(anchor_idx, tap_len=arr.shape[-1], half_width=int(half_width))
    tail_target = np.where(mask, 0.0, arr).astype(np.float32)
    coeffs = np.einsum("nt,td->nd", tail_target, np.asarray(basis, dtype=np.float32), optimize=True).astype(np.float32)
    return tail_target, coeffs


def reconstruct_q_np(anchor_idx: np.ndarray, local_kernel: np.ndarray, tail_coeffs: np.ndarray, basis: np.ndarray, tap_len: int, half_width: int) -> np.ndarray:
    anchor = np.asarray(anchor_idx, dtype=np.int64).reshape(-1)
    local = np.asarray(local_kernel, dtype=np.float32)
    tail = np.asarray(tail_coeffs, dtype=np.float32)
    q = tail @ np.asarray(basis, dtype=np.float32).T
    mask = window_mask_1d(anchor, tap_len=int(tap_len), half_width=int(half_width))
    q[mask] = 0.0
    rows = np.arange(anchor.shape[0], dtype=np.int64)
    for col, delta in enumerate(range(-int(half_width), int(half_width) + 1)):
        taps = anchor + int(delta)
        valid = (taps >= 0) & (taps < int(tap_len))
        q[rows[valid], taps[valid]] += local[rows[valid], col]
    return q.astype(np.float32)


def reconstruct_q_torch(anchor_idx: torch.Tensor, local_kernel: torch.Tensor, tail_coeffs: torch.Tensor, basis_t: torch.Tensor, tap_len: int, half_width: int) -> torch.Tensor:
    q = tail_coeffs @ basis_t.transpose(0, 1)
    device = q.device
    anchor = anchor_idx.to(device=device, dtype=torch.long).reshape(-1)
    rows = torch.arange(anchor.shape[0], device=device)
    for delta_idx, delta in enumerate(range(-int(half_width), int(half_width) + 1)):
        taps = anchor + int(delta)
        valid = (taps >= 0) & (taps < int(tap_len))
        if torch.any(valid):
            q[rows[valid], taps[valid]] = 0.0
    for delta_idx, delta in enumerate(range(-int(half_width), int(half_width) + 1)):
        taps = anchor + int(delta)
        valid = (taps >= 0) & (taps < int(tap_len))
        if torch.any(valid):
            q[rows[valid], taps[valid]] += local_kernel[rows[valid], delta_idx]
    return q


def anchor_metrics(pred_anchor: np.ndarray, true_anchor: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_anchor, dtype=np.int64).reshape(-1)
    true = np.asarray(true_anchor, dtype=np.int64).reshape(-1)
    delta = np.abs(pred - true)
    return {
        "anchor_exact": float(np.mean(delta == 0)),
        "anchor_within1": float(np.mean(delta <= 1)),
        "anchor_mae": float(np.mean(delta.astype(np.float64))),
    }


def local_energy_capture(q_true: np.ndarray, pred_anchor: np.ndarray, half_width: int) -> float:
    q = np.asarray(q_true, dtype=np.float32)
    mask = window_mask_1d(pred_anchor, tap_len=q.shape[-1], half_width=int(half_width))
    num = np.sum(np.where(mask, q, 0.0) ** 2, axis=1)
    den = np.sum(q**2, axis=1) + np.finfo(np.float32).eps
    return float(np.mean(num / den))


def compute_reference_covariance_features(x_ref: np.ndarray, n_fft: int) -> np.ndarray:
    arr = np.asarray(x_ref, dtype=np.float32)
    spec = np.fft.rfft(arr, n=int(n_fft), axis=-1)
    scale = max(int(arr.shape[-1]), 1)
    auto = (spec * np.conj(spec)).real / float(scale)
    cross01 = spec[0] * np.conj(spec[1]) / float(scale)
    cross02 = spec[0] * np.conj(spec[2]) / float(scale)
    cross12 = spec[1] * np.conj(spec[2]) / float(scale)
    channels = [
        np.log1p(auto[0].real).astype(np.float32),
        np.log1p(auto[1].real).astype(np.float32),
        np.log1p(auto[2].real).astype(np.float32),
        signed_log1p(cross01.real),
        signed_log1p(cross02.real),
        signed_log1p(cross12.real),
        signed_log1p(cross01.imag),
        signed_log1p(cross02.imag),
        signed_log1p(cross12.imag),
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def compute_reference_covariance_batch(x_ref: np.ndarray, n_fft: int) -> np.ndarray:
    arr = np.asarray(x_ref, dtype=np.float32)
    out = np.zeros((arr.shape[0], 9, int(n_fft // 2 + 1)), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = compute_reference_covariance_features(arr[i], int(n_fft))
    return out


def compute_anchor_prior_batch(d_path: np.ndarray, s_path: np.ndarray, q_full_len: int, q_keep_len: int, lambda_q_scale: float = 1.0e-3) -> np.ndarray:
    d_arr = np.asarray(d_path, dtype=np.float32)
    s_arr = np.asarray(s_path, dtype=np.float32)
    prior = np.zeros((d_arr.shape[0],), dtype=np.int64)
    for i in range(d_arr.shape[0]):
        q_full = _canonical_q_from_paths(d_arr[i], s_arr[i], int(q_full_len), float(lambda_q_scale))
        prior[i] = int(np.argmax(np.abs(q_full[: int(q_keep_len)])))
    return prior


@dataclass
class CanonicalDatasetBundle:
    h5_path: Path
    cfg: DatasetBuildConfig
    meta: dict[str, Any]
    p_ref_paths: np.ndarray
    d_path: np.ndarray
    s_path: np.ndarray
    r2r_paths: np.ndarray
    x_ref: np.ndarray
    q_target: np.ndarray
    w_canon: np.ndarray
    w_h5: np.ndarray
    anchors: np.ndarray
    local_target: np.ndarray
    tail_coeffs: np.ndarray
    tail_target: np.ndarray
    basis: np.ndarray
    room_data: dict[str, np.ndarray]
    source_seeds: np.ndarray
    anchor_prior: np.ndarray


def ensure_canonical_processed(h5_path: Path) -> None:
    needs_process = False
    with h5py.File(str(h5_path), "r") as h5:
        raw = h5["raw"]
        processed = h5.get("processed")
        required_raw = {"P_ref_paths", "D_path", "R2R_paths"}
        required_proc = {"q_target", "w_canon"}
        needs_process = any(name not in raw for name in required_raw) or processed is None or any(name not in processed for name in required_proc)
    if needs_process:
        print(f"[Canonical-Q] rebuilding processed features for {h5_path}")
        compute_processed_features(h5_path)


def load_canonical_q_dataset(h5_path: str | Path, laguerre_pole: float = 0.55) -> CanonicalDatasetBundle:
    path = Path(h5_path)
    ensure_canonical_processed(path)
    with h5py.File(str(path), "r") as h5:
        cfg = DatasetBuildConfig(**json.loads(h5.attrs["config_json"]))
        raw = h5["raw"]
        processed = h5["processed"]
        q_target = np.asarray(processed["q_target"], dtype=np.float32)
        q_keep_len = int(processed.attrs["q_keep_len"])
        q_full_len = int(processed.attrs["q_full_len"])
        local_half_width = int(processed.attrs["q_local_half_width"])
        tail_basis_dim = int(processed.attrs["q_tail_basis_dim"])
        basis = build_laguerre_basis(q_keep_len, tail_basis_dim, pole=float(laguerre_pole))
        anchors = np.argmax(np.abs(q_target), axis=1).astype(np.int64)
        local_target = extract_local_windows_1d(q_target, anchors, half_width=local_half_width)
        tail_target, tail_coeffs = project_tail_coeffs(q_target, anchors, basis=basis, half_width=local_half_width)
        room = raw["room_params"]
        room_data = {
            "room_size": np.asarray(room["room_size"], dtype=np.float32),
            "source_position": np.asarray(room["source_position"], dtype=np.float32),
            "ref_positions": np.asarray(room["ref_positions"], dtype=np.float32),
            "sec_positions": np.asarray(room["sec_positions"], dtype=np.float32),
            "err_positions": np.asarray(room["err_positions"], dtype=np.float32),
            "sound_speed": np.asarray(room["sound_speed"], dtype=np.float32),
            "material_absorption": np.asarray(room["material_absorption"], dtype=np.float32),
            "image_source_order": np.asarray(room["image_source_order"], dtype=np.int32),
            "layout_mode": np.asarray(room["layout_mode"]),
        }
        meta = {
            "q_keep_len": q_keep_len,
            "q_full_len": q_full_len,
            "q_target_dim": int(processed.attrs["q_target_dim"]),
            "local_half_width": local_half_width,
            "local_len": int(2 * local_half_width + 1),
            "tail_basis_dim": tail_basis_dim,
            "laguerre_pole": float(laguerre_pole),
            "lambda_q_scale": float(processed.attrs["lambda_q_scale"]),
            "lambda_w": float(processed.attrs["lambda_w"]),
            "q_target_source": str(processed.attrs.get("q_target_source", "h5_equivalent")),
            "r2r_pair_order_json": str(processed.attrs["r2r_pair_order_json"]),
        }
        return CanonicalDatasetBundle(
            h5_path=path,
            cfg=cfg,
            meta=meta,
            p_ref_paths=np.asarray(raw["P_ref_paths"], dtype=np.float32),
            d_path=np.asarray(raw["D_path"], dtype=np.float32),
            s_path=np.asarray(raw["S_paths"], dtype=np.float32),
            r2r_paths=np.asarray(raw["R2R_paths"], dtype=np.float32),
            x_ref=np.asarray(raw["x_ref"], dtype=np.float32),
            q_target=q_target,
            w_canon=np.asarray(processed["w_canon"], dtype=np.float32),
            w_h5=np.asarray(raw["W_full"], dtype=np.float32),
            anchors=anchors,
            local_target=local_target,
            tail_coeffs=tail_coeffs,
            tail_target=tail_target,
            basis=basis,
            room_data=room_data,
            source_seeds=np.asarray(raw["qc_metrics"]["source_seed"], dtype=np.int64),
            anchor_prior=compute_anchor_prior_batch(
                np.asarray(raw["D_path"], dtype=np.float32),
                np.asarray(raw["S_paths"], dtype=np.float32),
                q_full_len=q_full_len,
                q_keep_len=q_keep_len,
                lambda_q_scale=1.0e-3,
            ),
        )


def fit_feature_stats(bundle: CanonicalDatasetBundle, train_idx: np.ndarray, input_variant: str) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    stats["p_ref"] = stats_general(bundle.p_ref_paths, train_idx, reduce_axes=(0,))
    stats["d_path"] = stats_general(bundle.d_path, train_idx, reduce_axes=(0,))
    stats["s_path"] = stats_general(bundle.s_path, train_idx, reduce_axes=(0,))
    if "xref" in str(input_variant):
        xref_cov = compute_reference_covariance_batch(bundle.x_ref, n_fft=int(bundle.cfg.psd_nfft))
        stats["xref_cov"] = stats_general(xref_cov, train_idx, reduce_axes=(0,))
    if "r2r" in str(input_variant):
        stats["r2r"] = stats_general(bundle.r2r_paths, train_idx, reduce_axes=(0,))
    return stats


def build_feature_cache(bundle: CanonicalDatasetBundle, feature_stats: dict[str, Any], input_variant: str) -> dict[str, np.ndarray]:
    feats: dict[str, np.ndarray] = {}
    feats["p_ref"] = standardize(bundle.p_ref_paths, feature_stats["p_ref"]["mean"], feature_stats["p_ref"]["std"])
    feats["d_path"] = standardize(bundle.d_path, feature_stats["d_path"]["mean"], feature_stats["d_path"]["std"])
    feats["s_path"] = standardize(bundle.s_path, feature_stats["s_path"]["mean"], feature_stats["s_path"]["std"])
    feats["anchor_prior"] = np.asarray(bundle.anchor_prior, dtype=np.int64)
    if "xref" in str(input_variant):
        xref_cov = compute_reference_covariance_batch(bundle.x_ref, n_fft=int(bundle.cfg.psd_nfft))
        feats["xref_cov"] = standardize(xref_cov, feature_stats["xref_cov"]["mean"], feature_stats["xref_cov"]["std"])
    if "r2r" in str(input_variant):
        feats["r2r"] = standardize(bundle.r2r_paths, feature_stats["r2r"]["mean"], feature_stats["r2r"]["std"])
    return feats


def fit_target_stats(bundle: CanonicalDatasetBundle, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    q_scale = float(np.mean(bundle.q_target[train_idx] ** 2) + 1.0e-8)
    local_mean = np.mean(bundle.local_target[train_idx], axis=0, dtype=np.float64).astype(np.float32)
    local_std = np.maximum(np.std(bundle.local_target[train_idx], axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0e-6))
    tail_mean = np.mean(bundle.tail_coeffs[train_idx], axis=0, dtype=np.float64).astype(np.float32)
    tail_std = np.maximum(np.std(bundle.tail_coeffs[train_idx], axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0e-6))
    return {
        "q_scale": np.asarray(q_scale, dtype=np.float32),
        "local_mean": local_mean,
        "local_std": local_std,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
    }


class ConvNormAct1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 8):
        super().__init__()
        padding = int(kernel_size // 2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        gn_groups = max(1, min(int(groups), int(out_channels)))
        while out_channels % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.norm = nn.GroupNorm(gn_groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, groups: int = 8):
        super().__init__()
        padding = int(kernel_size // 2)
        gn_groups = max(1, min(int(groups), int(channels)))
        while channels % gn_groups != 0 and gn_groups > 1:
            gn_groups -= 1
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(gn_groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(gn_groups, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class PathEncoder1d(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, out_dim: int = 96, groups: int = 8):
        super().__init__()
        c1 = int(base_channels)
        c2 = int(base_channels * 2)
        c3 = int(base_channels * 3)
        self.stem = ConvNormAct1d(in_channels, c1, kernel_size=7, stride=1, groups=groups)
        self.block1 = ResidualBlock1d(c1, kernel_size=5, groups=groups)
        self.down1 = ConvNormAct1d(c1, c2, kernel_size=5, stride=2, groups=groups)
        self.block2 = ResidualBlock1d(c2, kernel_size=5, groups=groups)
        self.down2 = ConvNormAct1d(c2, c3, kernel_size=5, stride=2, groups=groups)
        self.block3 = ResidualBlock1d(c3, kernel_size=3, groups=groups)
        self.proj = nn.Sequential(
            nn.Linear(c3 * 2, out_dim),
            nn.GELU(),
            nn.Dropout(p=0.10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(self.stem(x))
        x = self.block2(self.down1(x))
        x = self.block3(self.down2(x))
        avg = x.mean(dim=-1)
        mx = x.amax(dim=-1)
        return self.proj(torch.cat([avg, mx], dim=1))


class CanonicalQPeakAnchorNet(nn.Module):
    def __init__(
        self,
        q_keep_len: int,
        local_len: int,
        tail_basis_dim: int,
        dropout: float = 0.10,
        include_xref: bool = False,
        include_r2r: bool = False,
    ):
        super().__init__()
        self.include_xref = bool(include_xref)
        self.include_r2r = bool(include_r2r)
        self.p_ref_encoder = PathEncoder1d(in_channels=1, base_channels=24, out_dim=64)
        self.d_encoder = PathEncoder1d(in_channels=1, base_channels=24, out_dim=64)
        self.s_encoder = PathEncoder1d(in_channels=1, base_channels=24, out_dim=64)
        if self.include_r2r:
            self.r2r_encoder = PathEncoder1d(in_channels=1, base_channels=16, out_dim=48)
        else:
            self.r2r_encoder = None
        if self.include_xref:
            self.xref_encoder = PathEncoder1d(in_channels=9, base_channels=24, out_dim=64)
        else:
            self.xref_encoder = None
        fusion_in = 64 * 3 + 64 + 64
        if self.include_r2r:
            fusion_in += 48 * 3
        if self.include_xref:
            fusion_in += 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.anchor_embed = nn.Embedding(int(q_keep_len), 24)
        self.prior_sigma = 2.5
        self.prior_strength = 3.0
        self.anchor_head = nn.Linear(192 + 24, int(q_keep_len))
        self.local_head = nn.Linear(192, int(local_len))
        self.tail_head = nn.Linear(192, int(tail_basis_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        p_ref: torch.Tensor,
        d_path: torch.Tensor,
        s_path: torch.Tensor,
        anchor_prior: torch.Tensor,
        xref_cov: torch.Tensor | None = None,
        r2r: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = int(p_ref.shape[0])
        p_feat = self.p_ref_encoder(p_ref.reshape(batch * 3, 1, -1)).reshape(batch, 3, -1).reshape(batch, -1)
        d_feat = self.d_encoder(d_path[:, None, :])
        s_feat = self.s_encoder(s_path[:, None, :])
        feats = [p_feat, d_feat, s_feat]
        if self.include_r2r:
            if r2r is None:
                raise ValueError("r2r input is required when include_r2r=True.")
            r2r_feat = self.r2r_encoder(r2r.reshape(batch * r2r.shape[1], 1, -1)).reshape(batch, r2r.shape[1], -1).reshape(batch, -1)
            feats.append(r2r_feat)
        if self.include_xref:
            if xref_cov is None:
                raise ValueError("xref_cov input is required when include_xref=True.")
            feats.append(self.xref_encoder(xref_cov))
        fused = self.fusion(torch.cat(feats, dim=1))
        prior_idx = anchor_prior.to(device=fused.device, dtype=torch.long).clamp_min(0).clamp_max(self.anchor_embed.num_embeddings - 1)
        prior_feat = self.anchor_embed(prior_idx)
        anchor_logits = self.anchor_head(torch.cat([fused, prior_feat], dim=1))
        grid = torch.arange(self.anchor_embed.num_embeddings, device=fused.device, dtype=torch.float32)[None, :]
        prior_bias = -((grid - prior_idx[:, None].to(dtype=torch.float32)) / float(self.prior_sigma)) ** 2
        anchor_logits = anchor_logits + float(self.prior_strength) * prior_bias
        return {
            "anchor_logits": anchor_logits,
            "local_kernel": self.local_head(fused),
            "tail_coeffs": self.tail_head(fused),
        }


def build_model(bundle: CanonicalDatasetBundle, input_variant: str, dropout: float) -> CanonicalQPeakAnchorNet:
    variant = str(input_variant)
    return CanonicalQPeakAnchorNet(
        q_keep_len=int(bundle.meta["q_keep_len"]),
        local_len=int(bundle.meta["local_len"]),
        tail_basis_dim=int(bundle.meta["tail_basis_dim"]),
        dropout=float(dropout),
        include_xref=("xref" in variant),
        include_r2r=("r2r" in variant),
    )


@dataclass
class ReplayCase:
    idx: int
    manager: Any
    time_axis: np.ndarray
    reference_signal: np.ndarray
    desired_signal: np.ndarray
    w_h5: np.ndarray
    w_exact: np.ndarray
    p_ref_paths: np.ndarray


def room_sample_from_bundle(bundle: CanonicalDatasetBundle, idx: int) -> dict[str, Any]:
    return {
        "room_size": np.asarray(bundle.room_data["room_size"][idx], dtype=float),
        "source_pos": np.asarray(bundle.room_data["source_position"][idx], dtype=float),
        "ref_positions": np.asarray(bundle.room_data["ref_positions"][idx], dtype=float),
        "sec_positions": np.asarray(bundle.room_data["sec_positions"][idx], dtype=float),
        "err_positions": np.asarray(bundle.room_data["err_positions"][idx], dtype=float),
        "sound_speed": float(bundle.room_data["sound_speed"][idx]),
        "absorption": float(bundle.room_data["material_absorption"][idx]),
        "image_order": int(bundle.room_data["image_source_order"][idx]),
        "layout_mode": (
            bundle.room_data["layout_mode"][idx].decode("utf-8")
            if isinstance(bundle.room_data["layout_mode"][idx], bytes)
            else str(bundle.room_data["layout_mode"][idx])
        ),
    }


def build_replay_cases(bundle: CanonicalDatasetBundle, room_indices: list[int]) -> list[ReplayCase]:
    sampler = AcousticScenarioSampler(bundle.cfg, np.random.default_rng(int(bundle.cfg.random_seed)))
    cases: list[ReplayCase] = []
    for idx in room_indices:
        sampled = room_sample_from_bundle(bundle, int(idx))
        manager = sampler.build_manager(sampled)
        manager.build(verbose=False)
        source_seed = int(bundle.source_seeds[int(idx)])
        noise, t = wn_gen(
            fs=int(bundle.cfg.fs),
            duration=float(bundle.cfg.noise_duration_s),
            f_low=float(bundle.cfg.f_low),
            f_high=float(bundle.cfg.f_high),
            rng=np.random.default_rng(source_seed),
        )
        source_signal = _normalize_columns(noise)
        time_axis = np.asarray(t[:, 0], dtype=float)
        desired_signal = manager.calculate_desired_signal(source_signal, len(time_axis))
        reference_signal = _normalize_columns(manager.calculate_reference_signal(source_signal, len(time_axis)))
        cases.append(
            ReplayCase(
                idx=int(idx),
                manager=manager,
                time_axis=time_axis,
                reference_signal=reference_signal,
                desired_signal=desired_signal,
                w_h5=np.asarray(bundle.w_h5[int(idx)], dtype=np.float32),
                w_exact=np.asarray(bundle.w_canon[int(idx)], dtype=np.float32),
                p_ref_paths=np.asarray(bundle.p_ref_paths[int(idx)], dtype=np.float32),
            )
        )
    return cases


def q_to_w_canon_batch(bundle: CanonicalDatasetBundle, q_truncated: np.ndarray, room_indices: list[int] | np.ndarray | None = None) -> np.ndarray:
    q = np.asarray(q_truncated, dtype=np.float32)
    if room_indices is None:
        room_indices_arr = np.arange(q.shape[0], dtype=np.int64)
    else:
        room_indices_arr = np.asarray(room_indices, dtype=np.int64)
        if room_indices_arr.shape[0] != q.shape[0]:
            raise ValueError(f"room_indices length {room_indices_arr.shape[0]} does not match q batch {q.shape[0]}.")
    q_full = np.zeros((q.shape[0], int(bundle.meta["q_full_len"])), dtype=np.float32)
    keep = min(int(bundle.meta["q_keep_len"]), q.shape[-1])
    q_full[:, :keep] = q[:, :keep]
    out = np.zeros((q.shape[0], 1, bundle.cfg.num_reference_mics, bundle.cfg.filter_len), dtype=np.float32)
    for i in range(q.shape[0]):
        out[i] = _solve_w_canonical_from_q(
            q_full[i],
            bundle.p_ref_paths[int(room_indices_arr[i])],
            filter_len=int(bundle.cfg.filter_len),
            lambda_w=float(bundle.meta["lambda_w"]),
            q_full_len=int(bundle.meta["q_full_len"]),
        )
    return out


def replay_metrics_for_case(case: ReplayCase, w_ai: np.ndarray, cfg: DatasetBuildConfig, early_window_s: float) -> dict[str, float]:
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
    e_zero = np.asarray(cfxlms_fn(params)["err_hist"], dtype=float)[:, 0]
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
    e_exact = np.asarray(
        _cfxlms_with_init(
            case.time_axis,
            case.manager,
            int(cfg.filter_len),
            float(cfg.mu_candidates[0]),
            case.reference_signal,
            case.desired_signal,
            w_init=case.w_exact,
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
    _, db_exact = _rolling_mse_db(e_exact, int(cfg.fs), window_samples=window_samples)
    _, db_h5 = _rolling_mse_db(e_h5, int(cfg.fs), window_samples=window_samples)
    early_mask = t_db <= float(early_window_s)
    if not np.any(early_mask):
        early_mask = np.ones_like(t_db, dtype=bool)
    return {
        "ai_vs_zero_db": float(np.mean(db_zero[early_mask] - db_ai[early_mask])),
        "exact_vs_zero_db": float(np.mean(db_zero[early_mask] - db_exact[early_mask])),
        "h5_vs_zero_db": float(np.mean(db_zero[early_mask] - db_h5[early_mask])),
        "ai_to_exact_gap_db": float(np.mean(np.abs(db_ai[early_mask] - db_exact[early_mask]))),
        "ai_to_h5_gap_db": float(np.mean(np.abs(db_ai[early_mask] - db_h5[early_mask]))),
    }


def summarize_replay(bundle: CanonicalDatasetBundle, room_indices: list[int], w_ai_batch: np.ndarray, early_window_s: float) -> dict[str, Any]:
    cases = build_replay_cases(bundle, room_indices)
    rows = []
    for case, w_ai in zip(cases, np.asarray(w_ai_batch, dtype=np.float32)):
        row = replay_metrics_for_case(case, w_ai=w_ai, cfg=bundle.cfg, early_window_s=float(early_window_s))
        row["room_idx"] = int(case.idx)
        rows.append(row)
    if not rows:
        return {"room_indices": [], "per_room": []}
    return {
        "room_indices": [int(v) for v in room_indices],
        "ai_vs_zero_db_mean": float(np.mean([row["ai_vs_zero_db"] for row in rows])),
        "exact_vs_zero_db_mean": float(np.mean([row["exact_vs_zero_db"] for row in rows])),
        "h5_vs_zero_db_mean": float(np.mean([row["h5_vs_zero_db"] for row in rows])),
        "ai_to_exact_gap_db_mean": float(np.mean([row["ai_to_exact_gap_db"] for row in rows])),
        "ai_to_h5_gap_db_mean": float(np.mean([row["ai_to_h5_gap_db"] for row in rows])),
        "per_room": rows,
    }


def exact_canonical_summary(bundle: CanonicalDatasetBundle, room_indices: list[int], early_window_s: float) -> dict[str, Any]:
    w_exact = np.asarray(bundle.w_canon[room_indices], dtype=np.float32)
    return summarize_replay(bundle, room_indices=room_indices, w_ai_batch=w_exact, early_window_s=float(early_window_s))


def build_probe_indices(train_idx: np.ndarray, val_idx: np.ndarray, probe_count: int, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(int(seed))
    train_sel = np.asarray(train_idx, dtype=np.int64).copy()
    val_sel = np.asarray(val_idx, dtype=np.int64).copy()
    rng.shuffle(train_sel)
    rng.shuffle(val_sel)
    return train_sel[: min(int(probe_count), train_sel.size)].tolist(), val_sel[: min(int(probe_count), val_sel.size)].tolist()


def exact_q_from_paths(bundle: CanonicalDatasetBundle, room_indices: np.ndarray) -> np.ndarray:
    q_out = np.zeros((len(room_indices), int(bundle.meta["q_keep_len"])), dtype=np.float32)
    for out_i, room_idx in enumerate(np.asarray(room_indices, dtype=np.int64)):
        q_full = _canonical_q_from_paths(
            bundle.d_path[room_idx],
            bundle.s_path[room_idx],
            q_full_len=int(bundle.meta["q_full_len"]),
            lambda_q_scale=float(bundle.meta["lambda_q_scale"]),
        )
        q_out[out_i] = q_full[: int(bundle.meta["q_keep_len"])]
    return q_out

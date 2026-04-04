from __future__ import annotations

import argparse
import csv
import json
import math
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from py_anc.algorithms.hybrid_loss import HybridAcousticLoss


def resolve_h5_path(explicit_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(ROOT / "python_impl" / "python_scripts" / "cfxlms_qc_dataset_single_control.h5")
    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
        checked.append(str(resolved))
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Single-control dataset HDF5 was not found. Checked: {checked}")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in str(text).split(",") if v.strip()]


def level_mask(image_order: np.ndarray, level: int) -> np.ndarray:
    arr = np.asarray(image_order, dtype=np.int64)
    if int(level) == 1:
        return arr == 0
    if int(level) in (2, 3):
        return np.isin(arr, np.asarray([1, 2], dtype=np.int64))
    raise ValueError(f"Unsupported level: {level}")


def build_dct_basis(filter_len: int, basis_dim: int) -> torch.Tensor:
    l = int(filter_len)
    k = int(basis_dim)
    if l <= 0 or k <= 0:
        raise ValueError("filter_len and basis_dim must be positive.")
    n = torch.arange(l, dtype=torch.float32)
    basis = torch.zeros((k, l), dtype=torch.float32)
    scale0 = math.sqrt(1.0 / float(l))
    scale = math.sqrt(2.0 / float(l))
    for i in range(k):
        alpha = scale0 if i == 0 else scale
        basis[i] = alpha * torch.cos((math.pi / float(l)) * (n + 0.5) * float(i))
    return basis


@dataclass
class HybridBundle:
    gcc: np.ndarray
    acoustic: np.ndarray | None
    p_ref: np.ndarray
    d_path: np.ndarray
    s_path: np.ndarray
    target_nr_db: np.ndarray | None
    w_opt: np.ndarray | None
    w_full: np.ndarray | None
    w_canon: np.ndarray | None
    image_order: np.ndarray
    source_position: np.ndarray


class HybridAncDataset(Dataset):
    def __init__(
        self,
        bundle: HybridBundle,
        indices: np.ndarray,
    ):
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        ridx = int(self.indices[int(idx)])
        gcc = torch.from_numpy(self.bundle.gcc[ridx]).to(dtype=torch.float32)
        if self.bundle.acoustic is not None:
            acoustic = torch.from_numpy(self.bundle.acoustic[ridx]).to(dtype=torch.float32)
        else:
            acoustic = torch.zeros((1, gcc.shape[-1]), dtype=torch.float32)
        p_ref = torch.from_numpy(self.bundle.p_ref[ridx]).to(dtype=torch.float32)
        d_path = torch.from_numpy(self.bundle.d_path[ridx]).to(dtype=torch.float32)
        s_path = torch.from_numpy(self.bundle.s_path[ridx]).to(dtype=torch.float32)
        if self.bundle.target_nr_db is None:
            target_nr_db = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            target_nr_db = torch.tensor(float(self.bundle.target_nr_db[ridx]), dtype=torch.float32)
        if self.bundle.w_opt is None:
            w_opt = torch.full_like(p_ref, float("nan"), dtype=torch.float32)
        else:
            w_opt = torch.from_numpy(self.bundle.w_opt[ridx]).to(dtype=torch.float32)
        if self.bundle.w_full is None:
            w_full = torch.full((1, p_ref.shape[0], p_ref.shape[1]), float("nan"), dtype=torch.float32)
        else:
            w_full = torch.from_numpy(self.bundle.w_full[ridx]).to(dtype=torch.float32)
        sample_idx = torch.tensor(ridx, dtype=torch.int64)
        return gcc, acoustic, p_ref, d_path, s_path, target_nr_db, w_opt, w_full, sample_idx


class ConvTokenEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(96, int(embed_dim), kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(8, 64)
        self.norm2 = nn.GroupNorm(8, 96)
        self.norm3 = nn.GroupNorm(8, int(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.norm1(self.conv1(x)))
        z = F.gelu(self.norm2(self.conv2(z)))
        z = F.gelu(self.norm3(self.conv3(z)))
        return z.transpose(1, 2)


class HybridDeepFxLMSNet(nn.Module):
    def __init__(
        self,
        acoustic_in_channels: int,
        filter_len: int,
        num_refs: int,
        basis_dim: int,
        embed_dim: int = 128,
        fusion_mode: str = "cross",
        num_heads: int = 4,
        disable_feature_b: bool = False,
        use_path_features: bool = False,
        use_index_embedding: bool = False,
        index_direct_lookup: bool = False,
        num_samples: int = 0,
        use_canonical_prior: bool = False,
        canonical_prior_lookup: np.ndarray | torch.Tensor | None = None,
        canonical_prior_scale: float = 1.0,
        residual_head_zero_init: bool = False,
    ):
        super().__init__()
        self.disable_feature_b = bool(disable_feature_b)
        self.fusion_mode = str(fusion_mode)
        self.num_refs = int(num_refs)
        self.basis_dim = int(basis_dim)
        self.use_path_features = bool(use_path_features)
        self.use_index_embedding = bool(use_index_embedding)
        self.index_direct_lookup = bool(index_direct_lookup)
        self.num_samples = int(num_samples)
        self.filter_len = int(filter_len)
        self.use_canonical_prior = bool(use_canonical_prior)
        self.canonical_prior_scale = float(canonical_prior_scale)

        if self.use_canonical_prior:
            if canonical_prior_lookup is None:
                raise ValueError("use_canonical_prior=True but canonical_prior_lookup was not provided.")
            prior_t = torch.as_tensor(canonical_prior_lookup, dtype=torch.float32)
            if prior_t.ndim == 4:
                prior_t = prior_t[:, 0, :, :]
            if prior_t.ndim != 3:
                raise ValueError(
                    "canonical_prior_lookup must be [N,R,L] (or [N,1,R,L]); "
                    f"got shape={tuple(prior_t.shape)}"
                )
            if prior_t.shape[1] != self.num_refs or prior_t.shape[2] != self.filter_len:
                raise ValueError(
                    "canonical_prior_lookup shape mismatch with model geometry: "
                    f"prior={tuple(prior_t.shape[1:])}, model={(self.num_refs, self.filter_len)}"
                )
            if self.num_samples > 0 and prior_t.shape[0] < self.num_samples:
                raise ValueError(
                    f"canonical_prior_lookup num_samples={prior_t.shape[0]} is smaller than model num_samples={self.num_samples}"
                )
            if self.num_samples <= 0:
                self.num_samples = int(prior_t.shape[0])
            self.register_buffer("canonical_prior_lookup", prior_t)
        else:
            self.canonical_prior_lookup = None

        if self.index_direct_lookup and self.num_samples <= 0:
            raise ValueError("num_samples must be positive when index_direct_lookup is enabled.")

        self.spatial_encoder = ConvTokenEncoder(in_channels=3, embed_dim=int(embed_dim))
        self.acoustic_encoder = ConvTokenEncoder(in_channels=int(max(acoustic_in_channels, 1)), embed_dim=int(embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=int(embed_dim), num_heads=int(num_heads), batch_first=True)

        if self.fusion_mode == "cross":
            fusion_in = int(embed_dim)
        elif self.fusion_mode == "cat":
            fusion_in = int(embed_dim * 2)
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        if self.use_path_features:
            path_in = int(self.num_refs * self.filter_len + self.filter_len + self.filter_len)
            self.path_encoder = nn.Sequential(
                nn.Linear(path_in, 256),
                nn.GELU(),
                nn.Linear(256, int(embed_dim)),
                nn.GELU(),
            )
            fusion_in += int(embed_dim)
        else:
            self.path_encoder = None

        if self.use_index_embedding:
            if self.num_samples <= 0:
                raise ValueError("num_samples must be positive when use_index_embedding is enabled.")
            self.index_embedding = nn.Embedding(self.num_samples, int(embed_dim))
            fusion_in += int(embed_dim)
        else:
            self.index_embedding = None

        if self.index_direct_lookup:
            self.index_direct_head = nn.Embedding(self.num_samples, int(self.num_refs * self.basis_dim))
            # Start from near-zero taps for stable direct-lookup optimization.
            nn.init.zeros_(self.index_direct_head.weight)
        else:
            self.index_direct_head = None

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.coeff_head = nn.Linear(192, int(self.num_refs * self.basis_dim))

        if bool(residual_head_zero_init):
            nn.init.zeros_(self.coeff_head.weight)
            nn.init.zeros_(self.coeff_head.bias)

        basis = build_dct_basis(filter_len=int(filter_len), basis_dim=int(self.basis_dim))
        self.register_buffer("dct_basis", basis)

    def _canonical_prior_for_batch(self, sample_idx: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        if not self.use_canonical_prior:
            return torch.zeros_like(ref)
        if sample_idx is None:
            raise ValueError("use_canonical_prior enabled but sample_idx was not provided.")
        if self.canonical_prior_lookup is None:
            raise RuntimeError("canonical_prior_lookup buffer is missing while use_canonical_prior=True.")
        idx = sample_idx.reshape(-1).to(device=ref.device, dtype=torch.long)
        idx = idx.clamp(min=0, max=int(self.canonical_prior_lookup.shape[0] - 1))
        prior = self.canonical_prior_lookup[idx].to(dtype=ref.dtype)
        return float(self.canonical_prior_scale) * prior

    def forward(
        self,
        gcc: torch.Tensor,
        acoustic: torch.Tensor,
        p_ref: torch.Tensor | None = None,
        d_path: torch.Tensor | None = None,
        s_path: torch.Tensor | None = None,
        sample_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.index_direct_lookup:
            if sample_idx is None:
                raise ValueError("index_direct_lookup enabled but sample_idx was not provided.")
            idx = sample_idx.reshape(-1).to(device=gcc.device, dtype=torch.long)
            idx = idx.clamp(min=0, max=self.num_samples - 1)
            coeffs = self.index_direct_head(idx).reshape(-1, self.num_refs, self.basis_dim)
            w_delta = torch.einsum("brk,kl->brl", coeffs, self.dct_basis)
            w_prior = self._canonical_prior_for_batch(sample_idx=sample_idx, ref=w_delta)
            w_pred = w_delta + w_prior
            return {
                "coeffs": coeffs,
                "w_pred": w_pred,
                "w_delta": w_delta,
                "w_prior": w_prior,
            }

        spatial_tokens = self.spatial_encoder(gcc)
        spatial_pool = torch.mean(spatial_tokens, dim=1)

        if self.disable_feature_b:
            acoustic_tokens = None
            acoustic_pool = torch.zeros_like(spatial_pool)
        else:
            acoustic_tokens = self.acoustic_encoder(acoustic)
            acoustic_pool = torch.mean(acoustic_tokens, dim=1)

        if self.fusion_mode == "cross":
            if acoustic_tokens is None:
                fused = spatial_pool
            else:
                attn_out, _ = self.cross_attn(query=spatial_tokens, key=acoustic_tokens, value=acoustic_tokens)
                fused = torch.mean(attn_out + spatial_tokens, dim=1)
        else:
            fused = torch.cat([spatial_pool, acoustic_pool], dim=1)

        if self.use_path_features:
            if p_ref is None or d_path is None or s_path is None:
                raise ValueError("Path features enabled but p_ref/d_path/s_path were not provided.")
            bsz = int(gcc.shape[0])
            path_flat = torch.cat(
                [
                    p_ref.reshape(bsz, -1),
                    d_path.reshape(bsz, -1),
                    s_path.reshape(bsz, -1),
                ],
                dim=1,
            )
            path_embed = self.path_encoder(path_flat)
            fused = torch.cat([fused, path_embed], dim=1)

        if self.use_index_embedding:
            if sample_idx is None:
                raise ValueError("Index embedding enabled but sample_idx was not provided.")
            idx = sample_idx.reshape(-1).to(device=gcc.device, dtype=torch.long)
            idx = idx.clamp(min=0, max=self.num_samples - 1)
            idx_embed = self.index_embedding(idx)
            fused = torch.cat([fused, idx_embed], dim=1)

        latent = self.fusion_mlp(fused)
        coeffs = self.coeff_head(latent).reshape(-1, self.num_refs, self.basis_dim)
        w_delta = torch.einsum("brk,kl->brl", coeffs, self.dct_basis)
        w_prior = self._canonical_prior_for_batch(sample_idx=sample_idx, ref=w_delta)
        w_pred = w_delta + w_prior
        return {
            "coeffs": coeffs,
            "w_pred": w_pred,
            "w_delta": w_delta,
            "w_prior": w_prior,
        }


def split_indices_train_val_test(
    indices: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64).copy()
    if idx.size < 3:
        raise ValueError(f"Need at least 3 samples for train/val/test split, got {idx.size}.")

    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)

    test_n = int(round(float(idx.size) * float(test_frac)))
    test_n = min(max(test_n, 1), idx.size - 2)

    rem_n = int(idx.size - test_n)
    val_n = int(round(float(rem_n) * float(val_frac)))
    val_n = min(max(val_n, 1), rem_n - 1)

    train_end = rem_n - val_n
    val_end = rem_n

    train_idx = np.sort(idx[:train_end])
    val_idx = np.sort(idx[train_end:val_end])
    test_idx = np.sort(idx[val_end:])
    return train_idx, val_idx, test_idx


def load_bundle(h5_path: Path, encoding: str, disable_feature_b: bool) -> HybridBundle:
    with h5py.File(str(h5_path), "r") as h5:
        raw = h5["raw"]
        processed = h5["processed"]
        gcc = np.asarray(processed["gcc_phat"], dtype=np.float32)
        p_ref = np.asarray(raw["P_ref_paths"], dtype=np.float32)
        d_path = np.asarray(raw["D_path"], dtype=np.float32)
        s_path = np.asarray(raw["S_paths"], dtype=np.float32)
        if disable_feature_b:
            acoustic = None
        else:
            key = "acoustic_feature_ri" if str(encoding) == "ri" else "acoustic_feature_mp"
            if key not in processed:
                raise KeyError(
                    f"Missing {key} in processed group. Run dataset processing with updated feature pipeline first."
                )
            acoustic = np.asarray(processed[key], dtype=np.float32)

        target_nr_db: np.ndarray | None = None
        if "qc_metrics" in raw and "nr_last_db" in raw["qc_metrics"]:
            target_nr_db = np.asarray(raw["qc_metrics"]["nr_last_db"], dtype=np.float32)

        w_opt: np.ndarray | None = None
        if "W_opt" in raw:
            w_opt = np.asarray(raw["W_opt"], dtype=np.float32)

        w_full: np.ndarray | None = None
        if "W_full" in raw:
            w_full = np.asarray(raw["W_full"], dtype=np.float32)

        w_canon: np.ndarray | None = None
        if "w_canon" in processed:
            w_canon_raw = np.asarray(processed["w_canon"], dtype=np.float32)
            if w_canon_raw.ndim == 4:
                w_canon_raw = w_canon_raw[:, 0, :, :]
            if w_canon_raw.ndim != 3:
                raise ValueError(f"processed/w_canon must be rank-3 or rank-4, got shape={w_canon_raw.shape}")
            if w_canon_raw.shape[0] != gcc.shape[0]:
                raise ValueError(
                    f"processed/w_canon room count mismatch with gcc_phat: {w_canon_raw.shape[0]} vs {gcc.shape[0]}"
                )
            if w_canon_raw.shape[1] != p_ref.shape[1] or w_canon_raw.shape[2] != p_ref.shape[2]:
                raise ValueError(
                    "processed/w_canon shape mismatch with P_ref_paths-derived [num_refs, filter_len]: "
                    f"{w_canon_raw.shape[1:]} vs {(p_ref.shape[1], p_ref.shape[2])}"
                )
            w_canon = w_canon_raw

        return HybridBundle(
            gcc=gcc,
            acoustic=acoustic,
            p_ref=p_ref,
            d_path=d_path,
            s_path=s_path,
            target_nr_db=target_nr_db,
            w_opt=w_opt,
            w_full=w_full,
            w_canon=w_canon,
            image_order=np.asarray(raw["room_params/image_source_order"], dtype=np.int64),
            source_position=np.asarray(raw["room_params/source_position"], dtype=np.float32),
        )


def dataset_quality_gate(
    h5_path: Path,
    bundle: HybridBundle,
    min_level1_samples: int,
    min_level23_samples: int,
    min_qc_nr_last_p10_db: float,
    min_qc_nr_gain_p10_db: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    failures: list[str] = []

    with h5py.File(str(h5_path), "r") as h5:
        raw = h5["raw"]
        qc = raw["qc_metrics"]
        image_order = np.asarray(raw["room_params/image_source_order"], dtype=np.int64)

        level1_count = int(np.sum(image_order == 0))
        level23_count = int(np.sum(np.isin(image_order, np.asarray([1, 2], dtype=np.int64))))
        report["level1_samples"] = level1_count
        report["level23_samples"] = level23_count

        if level1_count < int(min_level1_samples):
            failures.append(
                f"level1_samples_below_threshold:{level1_count}<{int(min_level1_samples)}"
            )
        if level23_count < int(min_level23_samples):
            failures.append(
                f"level23_samples_below_threshold:{level23_count}<{int(min_level23_samples)}"
            )

        if "nr_last_db" not in qc:
            failures.append("qc_missing:nr_last_db")
        else:
            nr_last = np.asarray(qc["nr_last_db"], dtype=np.float64)
            if not np.all(np.isfinite(nr_last)):
                failures.append("qc_non_finite:nr_last_db")
            nr_last_p10 = float(np.percentile(nr_last, 10.0))
            report["nr_last_db_p10"] = nr_last_p10
            if nr_last_p10 < float(min_qc_nr_last_p10_db):
                failures.append(
                    f"nr_last_db_p10_below_threshold:{nr_last_p10:.3f}<{float(min_qc_nr_last_p10_db):.3f}"
                )

        if "nr_gain_db" not in qc:
            failures.append("qc_missing:nr_gain_db")
        else:
            nr_gain = np.asarray(qc["nr_gain_db"], dtype=np.float64)
            if not np.all(np.isfinite(nr_gain)):
                failures.append("qc_non_finite:nr_gain_db")
            nr_gain_p10 = float(np.percentile(nr_gain, 10.0))
            report["nr_gain_db_p10"] = nr_gain_p10
            if nr_gain_p10 < float(min_qc_nr_gain_p10_db):
                failures.append(
                    f"nr_gain_db_p10_below_threshold:{nr_gain_p10:.3f}<{float(min_qc_nr_gain_p10_db):.3f}"
                )

    finite_checks: dict[str, np.ndarray | None] = {
        "gcc": bundle.gcc,
        "acoustic": bundle.acoustic,
        "p_ref": bundle.p_ref,
        "d_path": bundle.d_path,
        "s_path": bundle.s_path,
    }
    for name, arr in finite_checks.items():
        if arr is None:
            continue
        finite_ratio = float(np.mean(np.isfinite(arr)))
        report[f"{name}_finite_ratio"] = finite_ratio
        if finite_ratio < 1.0:
            failures.append(f"non_finite_values:{name}:{finite_ratio:.6f}")

    report["failures"] = failures
    if failures:
        raise RuntimeError(
            "Dataset quality gate failed. "
            + json.dumps(report, ensure_ascii=False, indent=2)
        )
    return report


def run_epoch(
    loader: DataLoader,
    model: HybridDeepFxLMSNet,
    device: torch.device,
    loss_module: HybridAcousticLoss,
    optimizer: torch.optim.Optimizer | None,
    margin_weight: float,
    supervision_weight: float,
    supervision_source: str,
    acoustic_loss_weight: float,
    canonical_residual_l2_weight: float = 0.0,
    grad_clip_norm: float | None = 5.0,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_sum = 0.0
    acoustic_sum = 0.0
    reg_sum = 0.0
    margin_sum = 0.0
    supervision_sum = 0.0
    residual_l2_sum = 0.0
    nr_sum = 0.0
    count = 0
    margin_weight_last = float(margin_weight)
    supervision_weight_last = float(supervision_weight)
    acoustic_weight_last = float(acoustic_loss_weight)
    residual_l2_weight_last = float(canonical_residual_l2_weight)
    supervision_source = str(supervision_source)

    for batch in loader:
        gcc, acoustic, p_ref, d_path, s_path, target_nr_db, w_opt, w_full, sample_idx = batch
        gcc = gcc.to(device=device, dtype=torch.float32, non_blocking=True)
        acoustic = acoustic.to(device=device, dtype=torch.float32, non_blocking=True)
        p_ref = p_ref.to(device=device, dtype=torch.float32, non_blocking=True)
        d_path = d_path.to(device=device, dtype=torch.float32, non_blocking=True)
        s_path = s_path.to(device=device, dtype=torch.float32, non_blocking=True)
        target_nr_db = target_nr_db.to(device=device, dtype=torch.float32, non_blocking=True)
        w_opt = w_opt.to(device=device, dtype=torch.float32, non_blocking=True)
        w_full = w_full.to(device=device, dtype=torch.float32, non_blocking=True)
        sample_idx = sample_idx.to(device=device, dtype=torch.long, non_blocking=True)

        out = model(gcc=gcc, acoustic=acoustic, p_ref=p_ref, d_path=d_path, s_path=s_path, sample_idx=sample_idx)
        losses = loss_module(
            w_pred=out["w_pred"],
            p_ref=p_ref,
            p_true=d_path,
            s_true=s_path,
            target_nr_db=target_nr_db,
            margin_weight=float(margin_weight),
        )

        supervision_weight_t = torch.as_tensor(float(supervision_weight), device=device, dtype=out["w_pred"].dtype)
        acoustic_weight_t = torch.as_tensor(float(acoustic_loss_weight), device=device, dtype=out["w_pred"].dtype)
        residual_l2_weight_t = torch.as_tensor(float(canonical_residual_l2_weight), device=device, dtype=out["w_pred"].dtype)
        supervision_loss = torch.zeros((), device=device, dtype=out["w_pred"].dtype)
        residual_l2_loss = torch.zeros((), device=device, dtype=out["w_pred"].dtype)
        supervision_target: torch.Tensor | None = None
        if supervision_source == "w_opt":
            supervision_target = w_opt
        elif supervision_source == "w_full":
            supervision_target = w_full[:, 0, :, :]

        if float(supervision_weight_t.detach().cpu()) > 0.0 and supervision_target is not None:
            valid_supervision = torch.isfinite(supervision_target).all(dim=(1, 2))
            if torch.any(valid_supervision):
                diff = out["w_pred"][valid_supervision] - supervision_target[valid_supervision]
                supervision_loss = torch.mean(diff.pow(2))

        if float(residual_l2_weight_t.detach().cpu()) > 0.0 and "w_delta" in out:
            w_delta = out["w_delta"]
            valid_delta = torch.isfinite(w_delta).all(dim=(1, 2))
            if torch.any(valid_delta):
                residual_l2_loss = torch.mean(w_delta[valid_delta].pow(2))

        total_loss = (
            acoustic_weight_t * losses["total"]
            + supervision_weight_t * supervision_loss
            + residual_l2_weight_t * residual_l2_loss
        )

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()

        bs = int(gcc.shape[0])
        total_sum += float(total_loss.detach().cpu()) * bs
        acoustic_sum += float(losses["acoustic"].detach().cpu()) * bs
        reg_sum += float(losses["reg"].detach().cpu()) * bs
        margin_sum += float(losses["margin"].detach().cpu()) * bs
        supervision_sum += float(supervision_loss.detach().cpu()) * bs
        residual_l2_sum += float(residual_l2_loss.detach().cpu()) * bs
        margin_weight_last = float(losses["margin_weight"].detach().cpu())
        supervision_weight_last = float(supervision_weight_t.detach().cpu())
        acoustic_weight_last = float(acoustic_weight_t.detach().cpu())
        residual_l2_weight_last = float(residual_l2_weight_t.detach().cpu())
        nr_sum += float(losses["nr_db"].detach().cpu()) * bs
        count += bs

    supervision_mean = supervision_sum / max(count, 1)
    return {
        "loss_total": total_sum / max(count, 1),
        "loss_acoustic": acoustic_sum / max(count, 1),
        "loss_reg": reg_sum / max(count, 1),
        "loss_margin": margin_sum / max(count, 1),
        "loss_supervision": supervision_mean,
        "loss_wopt": supervision_mean,
        "loss_residual_l2": residual_l2_sum / max(count, 1),
        "margin_weight": margin_weight_last,
        "supervision_weight": supervision_weight_last,
        "wopt_supervision_weight": supervision_weight_last,
        "supervision_source": supervision_source,
        "acoustic_loss_weight": acoustic_weight_last,
        "canonical_residual_l2_weight": residual_l2_weight_last,
        "nr_db": nr_sum / max(count, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-control Hybrid Deep-FxLMS with DCT synthesis and physics loss.")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--curriculum-levels", type=str, default="1,2,3")
    parser.add_argument("--epochs-per-level", type=str, default="8")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0e-3)
    parser.add_argument("--nr-margin-weight", type=float, default=0.0)
    parser.add_argument("--nr-target-ratio", type=float, default=0.5)
    parser.add_argument("--nr-margin-warmup-epochs", type=int, default=0)
    parser.add_argument("--nr-margin-mode", choices=("power", "db"), default="power")
    parser.add_argument("--nr-margin-focus-ratio", type=float, default=1.0)
    parser.add_argument("--wopt-supervision-weight", type=float, default=0.0)
    parser.add_argument("--supervision-weight", type=float, default=None)
    parser.add_argument("--supervision-source", choices=("none", "w_opt", "w_full"), default="w_opt")
    parser.add_argument("--acoustic-loss-weight", type=float, default=1.0)
    parser.add_argument("--loss-domain", choices=("freq", "time"), default="freq")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--basis-dim", type=int, default=32)
    parser.add_argument("--fusion-mode", choices=("cat", "cross"), default="cross")
    parser.add_argument("--feature-encoding", choices=("ri", "mp"), default="ri")
    parser.add_argument("--disable-feature-b", action="store_true")
    parser.add_argument("--use-path-features", action="store_true")
    parser.add_argument("--use-index-embedding", action="store_true")
    parser.add_argument("--index-direct-lookup", action="store_true")
    parser.add_argument("--index-direct-init-wopt", action="store_true")
    parser.add_argument("--index-direct-init-source", choices=("none", "w_opt", "w_full"), default="none")
    parser.add_argument("--index-direct-freeze", action="store_true")
    parser.add_argument("--use-canonical-prior", action="store_true")
    parser.add_argument("--canonical-prior-scale", type=float, default=1.0)
    parser.add_argument("--canonical-residual-l2-weight", type=float, default=0.0)
    parser.add_argument("--residual-head-zero-init", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--ablation-tag", type=str, default="")
    parser.add_argument("--skip-dataset-quality-gate", action="store_true")
    parser.add_argument("--min-level1-samples", type=int, default=128)
    parser.add_argument("--min-level23-samples", type=int, default=512)
    parser.add_argument("--min-qc-nr-last-p10-db", type=float, default=12.0)
    parser.add_argument("--min-qc-nr-gain-p10-db", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))

    if bool(args.disable_feature_b) and str(args.fusion_mode) == "cross":
        raise ValueError("disable-feature-b and fusion-mode=cross are incompatible. Use fusion-mode=cat for Feature-B ablation.")

    levels = parse_int_list(args.curriculum_levels)
    epoch_list = parse_int_list(args.epochs_per_level)
    if len(epoch_list) == 1 and len(levels) > 1:
        epoch_list = epoch_list * len(levels)
    if len(epoch_list) != len(levels):
        raise ValueError("epochs-per-level must provide one value or match curriculum-levels length.")

    h5_path = resolve_h5_path(args.h5_path)
    bundle = load_bundle(h5_path=h5_path, encoding=str(args.feature_encoding), disable_feature_b=bool(args.disable_feature_b))

    if bool(args.use_canonical_prior) and bundle.w_canon is None:
        raise RuntimeError(
            "use-canonical-prior was requested but processed/w_canon is missing in dataset. "
            "Please rebuild dataset with canonical prior cache."
        )

    if not (0.0 < float(args.val_frac) < 1.0):
        raise ValueError(f"val-frac must be in (0,1), got {args.val_frac}.")
    if not (0.0 < float(args.test_frac) < 1.0):
        raise ValueError(f"test-frac must be in (0,1), got {args.test_frac}.")

    supervision_weight = (
        float(args.supervision_weight)
        if args.supervision_weight is not None
        else float(args.wopt_supervision_weight)
    )
    supervision_source = str(args.supervision_source)
    if supervision_source == "none":
        supervision_weight = 0.0

    index_direct_init_source = str(args.index_direct_init_source)
    if bool(args.index_direct_init_wopt):
        index_direct_init_source = "w_opt"

    if bool(args.skip_dataset_quality_gate):
        quality_report: dict[str, Any] = {"skipped": True}
    else:
        quality_report = dataset_quality_gate(
            h5_path=h5_path,
            bundle=bundle,
            min_level1_samples=int(args.min_level1_samples),
            min_level23_samples=int(args.min_level23_samples),
            min_qc_nr_last_p10_db=float(args.min_qc_nr_last_p10_db),
            min_qc_nr_gain_p10_db=float(args.min_qc_nr_gain_p10_db),
        )
    print("Dataset quality gate:")
    print(json.dumps(quality_report, ensure_ascii=False, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if (str(args.device) == "auto" and torch.cuda.is_available()) else ("cpu" if str(args.device) == "auto" else str(args.device))
    )

    acoustic_in_channels = 1 if bundle.acoustic is None else int(bundle.acoustic.shape[1])
    model = HybridDeepFxLMSNet(
        acoustic_in_channels=acoustic_in_channels,
        filter_len=int(bundle.p_ref.shape[-1]),
        num_refs=int(bundle.p_ref.shape[1]),
        basis_dim=int(args.basis_dim),
        embed_dim=int(args.embed_dim),
        fusion_mode=str(args.fusion_mode),
        num_heads=int(args.num_heads),
        disable_feature_b=bool(args.disable_feature_b),
        use_path_features=bool(args.use_path_features),
        use_index_embedding=bool(args.use_index_embedding),
        index_direct_lookup=bool(args.index_direct_lookup),
        num_samples=int(bundle.gcc.shape[0]),
        use_canonical_prior=bool(args.use_canonical_prior),
        canonical_prior_lookup=bundle.w_canon if bool(args.use_canonical_prior) else None,
        canonical_prior_scale=float(args.canonical_prior_scale),
        residual_head_zero_init=bool(args.residual_head_zero_init),
    ).to(device)

    if bool(args.index_direct_lookup) and bool(args.index_direct_freeze):
        model.index_direct_head.weight.requires_grad_(False)
    loss_module = HybridAcousticLoss(
        lambda_reg=float(args.lambda_reg),
        conv_domain=str(args.loss_domain),
        nr_margin_weight=float(args.nr_margin_weight),
        nr_target_ratio=float(args.nr_target_ratio),
        nr_margin_mode=str(args.nr_margin_mode),
        nr_margin_focus_ratio=float(args.nr_margin_focus_ratio),
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying freeze options.")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    history_rows: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    global_epoch = 0
    t0 = time.time()

    all_indices = np.arange(int(bundle.gcc.shape[0]), dtype=np.int64)
    train_global_idx, val_global_idx, test_global_idx = split_indices_train_val_test(
        indices=all_indices,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    if bool(args.index_direct_lookup) and index_direct_init_source != "none":
        if index_direct_init_source == "w_opt":
            if bundle.w_opt is None:
                raise RuntimeError("index-direct-init-source=w_opt requested but raw/W_opt is missing in dataset.")
            init_w = np.asarray(bundle.w_opt, dtype=np.float32)
        elif index_direct_init_source == "w_full":
            if bundle.w_full is None:
                raise RuntimeError("index-direct-init-source=w_full requested but raw/W_full is missing in dataset.")
            init_w = np.asarray(bundle.w_full[:, 0, :, :], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported index-direct-init-source: {index_direct_init_source}")

        train_idx = np.asarray(train_global_idx, dtype=np.int64)
        init_w_train_t = torch.from_numpy(init_w[train_idx]).to(device=model.dct_basis.device)
        coeffs_train = torch.einsum("nrl,kl->nrk", init_w_train_t, model.dct_basis)
        with torch.no_grad():
            idx_t = torch.from_numpy(train_idx).to(device=model.dct_basis.device, dtype=torch.long)
            model.index_direct_head.weight[idx_t] = coeffs_train.reshape(int(coeffs_train.shape[0]), -1)

    if np.intersect1d(train_global_idx, val_global_idx).size > 0:
        raise RuntimeError("Global split leakage detected: train intersects val.")
    if np.intersect1d(train_global_idx, test_global_idx).size > 0:
        raise RuntimeError("Global split leakage detected: train intersects test.")
    if np.intersect1d(val_global_idx, test_global_idx).size > 0:
        raise RuntimeError("Global split leakage detected: val intersects test.")

    for stage_idx, (level, stage_epochs) in enumerate(zip(levels, epoch_list), start=1):
        mask = level_mask(bundle.image_order, int(level))
        stage_indices = np.where(mask)[0].astype(np.int64)
        if stage_indices.size < 8:
            raise RuntimeError(f"Level {level} has too few samples ({stage_indices.size}).")

        train_idx = np.intersect1d(stage_indices, train_global_idx)
        val_idx = np.intersect1d(stage_indices, val_global_idx)
        if int(args.max_train_samples) > 0:
            train_idx = train_idx[: int(args.max_train_samples)]

        if train_idx.size < 2 or val_idx.size < 2:
            raise RuntimeError(
                f"Level {level} has insufficient split samples: train={train_idx.size}, val={val_idx.size}. "
                "Adjust val-frac/test-frac or dataset size."
            )

        train_ds = HybridAncDataset(bundle=bundle, indices=train_idx)
        val_ds = HybridAncDataset(bundle=bundle, indices=val_idx)

        train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)

        best_val_nr = -1.0e9
        for _ in range(int(stage_epochs)):
            global_epoch += 1
            if float(args.nr_margin_weight) <= 0.0:
                current_margin_weight = 0.0
            elif int(args.nr_margin_warmup_epochs) <= 0:
                current_margin_weight = float(args.nr_margin_weight)
            else:
                ramp = min(1.0, float(global_epoch) / float(int(args.nr_margin_warmup_epochs)))
                current_margin_weight = float(args.nr_margin_weight) * ramp

            train_metrics = run_epoch(
                loader=train_loader,
                model=model,
                device=device,
                loss_module=loss_module,
                optimizer=optimizer,
                margin_weight=current_margin_weight,
                supervision_weight=float(supervision_weight),
                supervision_source=supervision_source,
                acoustic_loss_weight=float(args.acoustic_loss_weight),
                canonical_residual_l2_weight=float(args.canonical_residual_l2_weight),
                grad_clip_norm=float(args.grad_clip_norm),
            )
            val_metrics = run_epoch(
                loader=val_loader,
                model=model,
                device=device,
                loss_module=loss_module,
                optimizer=None,
                margin_weight=current_margin_weight,
                supervision_weight=float(supervision_weight),
                supervision_source=supervision_source,
                acoustic_loss_weight=float(args.acoustic_loss_weight),
                canonical_residual_l2_weight=float(args.canonical_residual_l2_weight),
                grad_clip_norm=float(args.grad_clip_norm),
            )

            if not np.isfinite(val_metrics["loss_total"]):
                raise RuntimeError(f"Validation loss became non-finite at epoch {global_epoch} (level {level}).")

            best_val_nr = max(best_val_nr, float(val_metrics["nr_db"]))
            row = {
                "stage": int(stage_idx),
                "level": int(level),
                "epoch": int(global_epoch),
                "train_loss_total": float(train_metrics["loss_total"]),
                "train_loss_acoustic": float(train_metrics["loss_acoustic"]),
                "train_loss_reg": float(train_metrics["loss_reg"]),
                "train_loss_margin": float(train_metrics["loss_margin"]),
                "train_loss_supervision": float(train_metrics["loss_supervision"]),
                "train_loss_wopt": float(train_metrics["loss_wopt"]),
                "train_loss_residual_l2": float(train_metrics["loss_residual_l2"]),
                "train_nr_db": float(train_metrics["nr_db"]),
                "val_loss_total": float(val_metrics["loss_total"]),
                "val_loss_acoustic": float(val_metrics["loss_acoustic"]),
                "val_loss_reg": float(val_metrics["loss_reg"]),
                "val_loss_margin": float(val_metrics["loss_margin"]),
                "val_loss_supervision": float(val_metrics["loss_supervision"]),
                "val_loss_wopt": float(val_metrics["loss_wopt"]),
                "val_loss_residual_l2": float(val_metrics["loss_residual_l2"]),
                "val_nr_db": float(val_metrics["nr_db"]),
                "margin_weight": float(current_margin_weight),
                "supervision_source": str(supervision_source),
                "supervision_weight": float(supervision_weight),
                "wopt_supervision_weight": float(supervision_weight),
                "acoustic_loss_weight": float(args.acoustic_loss_weight),
                "use_canonical_prior": bool(args.use_canonical_prior),
                "canonical_prior_scale": float(args.canonical_prior_scale),
                "canonical_residual_l2_weight": float(args.canonical_residual_l2_weight),
                "residual_head_zero_init": bool(args.residual_head_zero_init),
                "lambda_reg": float(args.lambda_reg),
                "loss_domain": str(args.loss_domain),
                "fusion_mode": str(args.fusion_mode),
                "feature_encoding": str(args.feature_encoding),
                "disable_feature_b": bool(args.disable_feature_b),
                "use_path_features": bool(args.use_path_features),
                "use_index_embedding": bool(args.use_index_embedding),
                "index_direct_lookup": bool(args.index_direct_lookup),
                "index_direct_init_source": str(index_direct_init_source),
                "index_direct_init_scope": "train_only" if (bool(args.index_direct_lookup) and index_direct_init_source != "none") else "none",
                "index_direct_init_wopt": bool(index_direct_init_source == "w_opt"),
                "index_direct_freeze": bool(args.index_direct_freeze),
            }
            history_rows.append(row)
            print(
                f"[Stage {stage_idx} L{level} | Epoch {global_epoch:03d}] "
                f"train_loss={train_metrics['loss_total']:.4e}, "
                f"val_loss={val_metrics['loss_total']:.4e}, "
                f"val_nr={val_metrics['nr_db']:.3f} dB"
            )

        stage_summaries.append(
            {
                "stage": int(stage_idx),
                "level": int(level),
                "train_samples": int(train_idx.size),
                "val_samples": int(val_idx.size),
                "test_samples": int(np.intersect1d(stage_indices, test_global_idx).size),
                "best_val_nr_db": float(best_val_nr),
            }
        )

    torch.save(
        {
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "dct_basis": model.dct_basis.detach().cpu(),
            "stage_summaries": stage_summaries,
            "split_indices": {
                "train": train_global_idx,
                "val": val_global_idx,
                "test": test_global_idx,
            },
            "h5_path": str(h5_path),
        },
        output_dir / "final_hybrid_deep_fxlms.pt",
    )

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()) if history_rows else [])
        if history_rows:
            writer.writeheader()
            writer.writerows(history_rows)

    summary = {
        "h5_path": str(h5_path),
        "elapsed_s": float(time.time() - t0),
        "seed": int(args.seed),
        "ablation_tag": str(args.ablation_tag),
        "levels": [int(v) for v in levels],
        "epochs_per_level": [int(v) for v in epoch_list],
        "fusion_mode": str(args.fusion_mode),
        "feature_encoding": str(args.feature_encoding),
        "disable_feature_b": bool(args.disable_feature_b),
        "use_path_features": bool(args.use_path_features),
        "use_index_embedding": bool(args.use_index_embedding),
        "index_direct_lookup": bool(args.index_direct_lookup),
        "index_direct_init_source": str(index_direct_init_source),
        "index_direct_init_scope": "train_only" if (bool(args.index_direct_lookup) and index_direct_init_source != "none") else "none",
        "index_direct_init_wopt": bool(index_direct_init_source == "w_opt"),
        "index_direct_freeze": bool(args.index_direct_freeze),
        "use_canonical_prior": bool(args.use_canonical_prior),
        "canonical_prior_scale": float(args.canonical_prior_scale),
        "canonical_residual_l2_weight": float(args.canonical_residual_l2_weight),
        "residual_head_zero_init": bool(args.residual_head_zero_init),
        "lambda_reg": float(args.lambda_reg),
        "nr_margin_weight": float(args.nr_margin_weight),
        "nr_target_ratio": float(args.nr_target_ratio),
        "nr_margin_warmup_epochs": int(args.nr_margin_warmup_epochs),
        "nr_margin_mode": str(args.nr_margin_mode),
        "nr_margin_focus_ratio": float(args.nr_margin_focus_ratio),
        "supervision_source": str(supervision_source),
        "supervision_weight": float(supervision_weight),
        "wopt_supervision_weight": float(supervision_weight),
        "acoustic_loss_weight": float(args.acoustic_loss_weight),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "global_split": {
            "seed": int(args.seed),
            "num_total": int(bundle.gcc.shape[0]),
            "num_train": int(train_global_idx.size),
            "num_val": int(val_global_idx.size),
            "num_test": int(test_global_idx.size),
        },
        "loss_domain": str(args.loss_domain),
        "dataset_quality_gate": quality_report,
        "stage_summaries": stage_summaries,
        "final_epoch": int(global_epoch),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

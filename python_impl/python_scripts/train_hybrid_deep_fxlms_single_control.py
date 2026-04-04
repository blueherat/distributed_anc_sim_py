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
        sample_idx = torch.tensor(ridx, dtype=torch.int64)
        return gcc, acoustic, p_ref, d_path, s_path, target_nr_db, w_opt, sample_idx


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

        basis = build_dct_basis(filter_len=int(filter_len), basis_dim=int(self.basis_dim))
        self.register_buffer("dct_basis", basis)

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
            w_pred = torch.einsum("brk,kl->brl", coeffs, self.dct_basis)
            return {
                "coeffs": coeffs,
                "w_pred": w_pred,
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
        w_pred = torch.einsum("brk,kl->brl", coeffs, self.dct_basis)
        return {
            "coeffs": coeffs,
            "w_pred": w_pred,
        }


def split_indices(indices: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64).copy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    split = int(round(idx.size * (1.0 - float(val_frac))))
    split = min(max(split, 1), idx.size - 1)
    return np.sort(idx[:split]), np.sort(idx[split:])


def load_bundle(h5_path: Path, encoding: str, disable_feature_b: bool) -> HybridBundle:
    with h5py.File(str(h5_path), "r") as h5:
        raw = h5["raw"]
        processed = h5["processed"]
        gcc = np.asarray(processed["gcc_phat"], dtype=np.float32)
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

        return HybridBundle(
            gcc=gcc,
            acoustic=acoustic,
            p_ref=np.asarray(raw["P_ref_paths"], dtype=np.float32),
            d_path=np.asarray(raw["D_path"], dtype=np.float32),
            s_path=np.asarray(raw["S_paths"], dtype=np.float32),
            target_nr_db=target_nr_db,
            w_opt=w_opt,
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
    wopt_supervision_weight: float,
    acoustic_loss_weight: float,
    grad_clip_norm: float | None = 5.0,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_sum = 0.0
    acoustic_sum = 0.0
    reg_sum = 0.0
    margin_sum = 0.0
    wopt_sum = 0.0
    nr_sum = 0.0
    count = 0
    margin_weight_last = float(margin_weight)
    wopt_weight_last = float(wopt_supervision_weight)
    acoustic_weight_last = float(acoustic_loss_weight)

    for batch in loader:
        gcc, acoustic, p_ref, d_path, s_path, target_nr_db, w_opt, sample_idx = batch
        gcc = gcc.to(device=device, dtype=torch.float32, non_blocking=True)
        acoustic = acoustic.to(device=device, dtype=torch.float32, non_blocking=True)
        p_ref = p_ref.to(device=device, dtype=torch.float32, non_blocking=True)
        d_path = d_path.to(device=device, dtype=torch.float32, non_blocking=True)
        s_path = s_path.to(device=device, dtype=torch.float32, non_blocking=True)
        target_nr_db = target_nr_db.to(device=device, dtype=torch.float32, non_blocking=True)
        w_opt = w_opt.to(device=device, dtype=torch.float32, non_blocking=True)
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

        wopt_weight_t = torch.as_tensor(float(wopt_supervision_weight), device=device, dtype=out["w_pred"].dtype)
        acoustic_weight_t = torch.as_tensor(float(acoustic_loss_weight), device=device, dtype=out["w_pred"].dtype)
        wopt_loss = torch.zeros((), device=device, dtype=out["w_pred"].dtype)
        if float(wopt_weight_t.detach().cpu()) > 0.0:
            valid_wopt = torch.isfinite(w_opt).all(dim=(1, 2))
            if torch.any(valid_wopt):
                diff = out["w_pred"][valid_wopt] - w_opt[valid_wopt]
                wopt_loss = torch.mean(diff.pow(2))

        total_loss = acoustic_weight_t * losses["total"] + wopt_weight_t * wopt_loss

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
        wopt_sum += float(wopt_loss.detach().cpu()) * bs
        margin_weight_last = float(losses["margin_weight"].detach().cpu())
        wopt_weight_last = float(wopt_weight_t.detach().cpu())
        acoustic_weight_last = float(acoustic_weight_t.detach().cpu())
        nr_sum += float(losses["nr_db"].detach().cpu()) * bs
        count += bs

    return {
        "loss_total": total_sum / max(count, 1),
        "loss_acoustic": acoustic_sum / max(count, 1),
        "loss_reg": reg_sum / max(count, 1),
        "loss_margin": margin_sum / max(count, 1),
        "loss_wopt": wopt_sum / max(count, 1),
        "margin_weight": margin_weight_last,
        "wopt_supervision_weight": wopt_weight_last,
        "acoustic_loss_weight": acoustic_weight_last,
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
    parser.add_argument("--index-direct-freeze", action="store_true")
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
    ).to(device)

    if bool(args.index_direct_lookup) and bool(args.index_direct_init_wopt):
        if bundle.w_opt is None:
            raise RuntimeError("index-direct-init-wopt requested but raw/W_opt is missing in dataset.")
        w_opt_t = torch.from_numpy(np.asarray(bundle.w_opt, dtype=np.float32)).to(device=model.dct_basis.device)
        coeffs = torch.einsum("nrl,kl->nrk", w_opt_t, model.dct_basis)
        with torch.no_grad():
            model.index_direct_head.weight.copy_(coeffs.reshape(int(coeffs.shape[0]), -1))

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

    for stage_idx, (level, stage_epochs) in enumerate(zip(levels, epoch_list), start=1):
        mask = level_mask(bundle.image_order, int(level))
        stage_indices = np.where(mask)[0].astype(np.int64)
        if int(args.max_train_samples) > 0:
            stage_indices = stage_indices[: int(args.max_train_samples)]
        if stage_indices.size < 8:
            raise RuntimeError(f"Level {level} has too few samples ({stage_indices.size}).")

        train_idx, val_idx = split_indices(stage_indices, val_frac=float(args.val_frac), seed=int(args.seed + stage_idx))
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
                wopt_supervision_weight=float(args.wopt_supervision_weight),
                acoustic_loss_weight=float(args.acoustic_loss_weight),
                grad_clip_norm=float(args.grad_clip_norm),
            )
            val_metrics = run_epoch(
                loader=val_loader,
                model=model,
                device=device,
                loss_module=loss_module,
                optimizer=None,
                margin_weight=current_margin_weight,
                wopt_supervision_weight=float(args.wopt_supervision_weight),
                acoustic_loss_weight=float(args.acoustic_loss_weight),
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
                "train_loss_wopt": float(train_metrics["loss_wopt"]),
                "train_nr_db": float(train_metrics["nr_db"]),
                "val_loss_total": float(val_metrics["loss_total"]),
                "val_loss_acoustic": float(val_metrics["loss_acoustic"]),
                "val_loss_reg": float(val_metrics["loss_reg"]),
                "val_loss_margin": float(val_metrics["loss_margin"]),
                "val_loss_wopt": float(val_metrics["loss_wopt"]),
                "val_nr_db": float(val_metrics["nr_db"]),
                "margin_weight": float(current_margin_weight),
                "wopt_supervision_weight": float(args.wopt_supervision_weight),
                "acoustic_loss_weight": float(args.acoustic_loss_weight),
                "lambda_reg": float(args.lambda_reg),
                "loss_domain": str(args.loss_domain),
                "fusion_mode": str(args.fusion_mode),
                "feature_encoding": str(args.feature_encoding),
                "disable_feature_b": bool(args.disable_feature_b),
                "use_path_features": bool(args.use_path_features),
                "use_index_embedding": bool(args.use_index_embedding),
                "index_direct_lookup": bool(args.index_direct_lookup),
                "index_direct_init_wopt": bool(args.index_direct_init_wopt),
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
                "best_val_nr_db": float(best_val_nr),
            }
        )

    torch.save(
        {
            "args": vars(args),
            "model_state_dict": model.state_dict(),
            "dct_basis": model.dct_basis.detach().cpu(),
            "stage_summaries": stage_summaries,
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
        "index_direct_init_wopt": bool(args.index_direct_init_wopt),
        "index_direct_freeze": bool(args.index_direct_freeze),
        "lambda_reg": float(args.lambda_reg),
        "nr_margin_weight": float(args.nr_margin_weight),
        "nr_target_ratio": float(args.nr_target_ratio),
        "nr_margin_warmup_epochs": int(args.nr_margin_warmup_epochs),
        "nr_margin_mode": str(args.nr_margin_mode),
        "nr_margin_focus_ratio": float(args.nr_margin_focus_ratio),
        "wopt_supervision_weight": float(args.wopt_supervision_weight),
        "acoustic_loss_weight": float(args.acoustic_loss_weight),
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

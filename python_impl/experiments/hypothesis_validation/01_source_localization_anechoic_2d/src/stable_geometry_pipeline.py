from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares
from torch.utils.data import DataLoader, TensorDataset

from python_scripts.hypothesis_validation_common import (
    LOCALIZATION_THRESHOLDS,
    ResidualMLP,
    SimpleMLP,
    compute_gcc_triplet,
    compute_reference_covariance_features,
    evaluate_localization_analytic_only,
    fit_standardizer,
    infer_localization_window_preset,
    load_localization_dataset,
    localization_error_stats,
    localization_error_vector,
    peak_index_subsample,
    render_training_curves,
    save_history_csv,
    save_json,
    set_seed,
)


PAIR_LABELS: tuple[str, str, str] = ("01", "02", "12")
PEAK_WINDOW_LEN = 65
PHYSICS_LOSS_WEIGHT = 0.25
GEOM_REF_FLAT_DIM = 6
GEOM_SHAPE_SLICE = slice(6, 8)
GEOM_AREA_IDX = 8
GEOM_COND_IDX = 9


@dataclass(frozen=True)
class Stage01CandidateSpec:
    candidate_id: str
    feature_kind: str
    model_kind: str
    target_kind: str
    loss_kind: str
    hidden: int = 256
    depth: int = 4
    dropout: float = 0.10
    physics_weight: float = PHYSICS_LOSS_WEIGHT
    solver_steps: int = 5


def stable_candidate_specs() -> list[Stage01CandidateSpec]:
    return [
        Stage01CandidateSpec(
            candidate_id="tdoa_global_baseline",
            feature_kind="tdoa_global",
            model_kind="mlp",
            target_kind="global_xy",
            loss_kind="mse",
            hidden=256,
            depth=3,
            dropout=0.10,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_canonical_mlp",
            feature_kind="tdoa_canonical",
            model_kind="mlp",
            target_kind="canonical_xy",
            loss_kind="huber",
            hidden=192,
            depth=3,
            dropout=0.08,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_canonical_resmlp",
            feature_kind="tdoa_canonical",
            model_kind="resmlp",
            target_kind="canonical_xy",
            loss_kind="huber",
            hidden=256,
            depth=4,
            dropout=0.10,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_canonical_resmlp_phys",
            feature_kind="tdoa_canonical",
            model_kind="resmlp",
            target_kind="canonical_xy",
            loss_kind="physics_huber",
            hidden=256,
            depth=4,
            dropout=0.10,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_anchor_range_mlp",
            feature_kind="tdoa_canonical",
            model_kind="mlp",
            target_kind="anchor_range",
            loss_kind="anchor_range_huber",
            hidden=192,
            depth=3,
            dropout=0.08,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_energy_canonical_mlp",
            feature_kind="tdoa_energy_canonical",
            model_kind="mlp",
            target_kind="canonical_xy",
            loss_kind="huber",
            hidden=256,
            depth=3,
            dropout=0.08,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_energy_abs_canonical_mlp",
            feature_kind="tdoa_energy_abs_canonical",
            model_kind="mlp",
            target_kind="canonical_xy",
            loss_kind="huber",
            hidden=256,
            depth=3,
            dropout=0.08,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_energy_abs_phys_mlp_l010",
            feature_kind="tdoa_energy_abs_canonical",
            model_kind="mlp",
            target_kind="canonical_xy",
            loss_kind="physics_huber",
            hidden=256,
            depth=3,
            dropout=0.08,
            physics_weight=0.10,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_energy_abs_phys_mlp_l025",
            feature_kind="tdoa_energy_abs_canonical",
            model_kind="mlp",
            target_kind="canonical_xy",
            loss_kind="physics_huber",
            hidden=256,
            depth=3,
            dropout=0.08,
            physics_weight=0.25,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_energy_abs_phys_mlp_l050",
            feature_kind="tdoa_energy_abs_canonical",
            model_kind="mlp",
            target_kind="canonical_xy",
            loss_kind="physics_huber",
            hidden=256,
            depth=3,
            dropout=0.08,
            physics_weight=0.50,
        ),
        Stage01CandidateSpec(
            candidate_id="tdoa_barycentric_resmlp",
            feature_kind="tdoa_canonical",
            model_kind="resmlp",
            target_kind="barycentric",
            loss_kind="barycentric_huber",
            hidden=256,
            depth=4,
            dropout=0.10,
        ),
        Stage01CandidateSpec(
            candidate_id="two_tower_tdoa_energy",
            feature_kind="two_tower_tdoa_energy",
            model_kind="two_tower_mlp",
            target_kind="canonical_xy",
            loss_kind="huber",
            hidden=224,
            depth=3,
            dropout=0.08,
        ),
        Stage01CandidateSpec(
            candidate_id="solver_tdoa_energy_anchor",
            feature_kind="two_tower_tdoa_energy_abs",
            model_kind="two_tower_mlp",
            target_kind="canonical_xy",
            loss_kind="solver_huber",
            hidden=224,
            depth=3,
            dropout=0.08,
            physics_weight=0.25,
            solver_steps=5,
        ),
        Stage01CandidateSpec(
            candidate_id="two_tower_tdoa_energy_phys",
            feature_kind="two_tower_tdoa_energy",
            model_kind="two_tower_mlp",
            target_kind="canonical_xy",
            loss_kind="physics_huber",
            hidden=224,
            depth=3,
            dropout=0.08,
        ),
        Stage01CandidateSpec(
            candidate_id="two_tower_tdoa_energy_film",
            feature_kind="two_tower_tdoa_energy",
            model_kind="two_tower_film",
            target_kind="canonical_xy",
            loss_kind="huber",
            hidden=224,
            depth=3,
            dropout=0.08,
        ),
        Stage01CandidateSpec(
            candidate_id="solver_gcc_peak_anchor",
            feature_kind="solver_gcc_peak_energy_abs",
            model_kind="two_tower_conv",
            target_kind="canonical_xy",
            loss_kind="solver_huber",
            hidden=224,
            depth=3,
            dropout=0.10,
            physics_weight=0.25,
            solver_steps=5,
        ),
        Stage01CandidateSpec(
            candidate_id="solver_gcc_full_anchor",
            feature_kind="solver_gcc_full_energy_abs",
            model_kind="two_tower_conv",
            target_kind="canonical_xy",
            loss_kind="solver_huber",
            hidden=256,
            depth=3,
            dropout=0.10,
            physics_weight=0.25,
            solver_steps=5,
        ),
        Stage01CandidateSpec(
            candidate_id="two_tower_gcc_peak_energy",
            feature_kind="two_tower_gcc_peak_energy",
            model_kind="two_tower_conv",
            target_kind="canonical_xy",
            loss_kind="huber",
            hidden=224,
            depth=3,
            dropout=0.10,
        ),
        Stage01CandidateSpec(
            candidate_id="two_tower_gcc_full_energy",
            feature_kind="two_tower_gcc_full_energy",
            model_kind="two_tower_conv",
            target_kind="canonical_xy",
            loss_kind="huber",
            hidden=256,
            depth=3,
            dropout=0.10,
        ),
    ]


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_geometry_audit_bundle(audit_dir: Path | str) -> dict[str, Any]:
    audit_path = Path(audit_dir)
    summary = json.loads((audit_path / "summary.json").read_text(encoding="utf-8"))
    selected_window = str(summary.get("selected_window_preset", "W1"))
    run_dir = audit_path / selected_window
    return {
        "audit_dir": audit_path,
        "summary": summary,
        "selected_window_preset": selected_window,
        "threshold_grid": _load_csv_rows(run_dir / "threshold_grid.csv"),
        "sample_metrics": _load_csv_rows(run_dir / "sample_metrics.csv"),
    }


def load_stable_localization_summary(result_dir: Path | str) -> dict[str, Any]:
    path = Path(result_dir) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def choose_sample_indices_from_summary(summary: dict[str, Any]) -> dict[str, int]:
    plots = summary.get("analytic_debug_samples", {})
    return {
        "success_iid": int(plots.get("success_iid_sample_index", 0)),
        "failure_iid": int(plots.get("failure_iid_sample_index", 0)),
        "success_geom": int(plots.get("success_geom_sample_index", 0)),
        "failure_geom": int(plots.get("failure_geom_sample_index", 0)),
    }


def sample_rows_to_dataframe(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = rows[0].keys() if rows else []
    out: dict[str, np.ndarray] = {}
    for key in keys:
        values = [row[key] for row in rows]
        try:
            out[key] = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            out[key] = np.asarray(values, dtype=object)
    return out


class PairwiseConvTower(nn.Module):
    def __init__(self, pair_count: int, seq_len: int, extra_dim: int, out_dim: int, dropout: float = 0.10):
        super().__init__()
        hidden = 48
        self.pair_count = int(pair_count)
        self.seq_len = int(seq_len)
        self.extra_dim = int(extra_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(self.pair_count, hidden, kernel_size=7, padding=3),
            nn.GroupNorm(6, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(6, hidden),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden * 2),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )
        self.proj = ResidualMLP(hidden * 2 + self.extra_dim, int(out_dim), hidden=max(int(out_dim), 96), depth=2, dropout=float(dropout))

    def forward(self, signal_flat: torch.Tensor) -> torch.Tensor:
        seq_flat_dim = int(self.pair_count * self.seq_len)
        seq = signal_flat[:, :seq_flat_dim].reshape(signal_flat.shape[0], self.pair_count, self.seq_len)
        extra = signal_flat[:, seq_flat_dim : seq_flat_dim + self.extra_dim]
        h = self.conv(seq)
        h = torch.mean(h, dim=-1)
        if self.extra_dim > 0:
            h = torch.cat([h, extra], dim=1)
        return self.proj(h)


class TwoTowerRegressor(nn.Module):
    def __init__(
        self,
        geometry_dim: int,
        signal_dim: int,
        out_dim: int,
        hidden: int = 224,
        depth: int = 3,
        dropout: float = 0.10,
        signal_mode: str = "mlp",
        pair_count: int = 0,
        seq_len: int = 0,
        extra_dim: int = 0,
    ):
        super().__init__()
        geom_hidden = max(int(hidden // 2), 96)
        sig_hidden = max(int(hidden // 2), 96)
        self.geometry_dim = int(geometry_dim)
        self.signal_dim = int(signal_dim)
        self.signal_mode = str(signal_mode)
        self.geometry_tower = ResidualMLP(int(geometry_dim), geom_hidden, hidden=geom_hidden, depth=max(int(depth) - 1, 2), dropout=float(dropout))
        if self.signal_mode == "mlp":
            self.signal_tower = ResidualMLP(int(signal_dim), sig_hidden, hidden=sig_hidden, depth=max(int(depth) - 1, 2), dropout=float(dropout))
        elif self.signal_mode == "pairwise_conv":
            self.signal_tower = PairwiseConvTower(int(pair_count), int(seq_len), int(extra_dim), sig_hidden, dropout=float(dropout))
        else:
            raise KeyError(f"Unsupported signal tower mode: {signal_mode}")
        self.fusion_head = ResidualMLP(geom_hidden + sig_hidden, int(out_dim), hidden=int(hidden), depth=int(depth), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        geom = x[:, : self.geometry_dim]
        sig = x[:, self.geometry_dim : self.geometry_dim + self.signal_dim]
        geom_h = self.geometry_tower(geom)
        sig_h = self.signal_tower(sig)
        return self.fusion_head(torch.cat([geom_h, sig_h], dim=1))


class TwoTowerFiLMRegressor(nn.Module):
    def __init__(
        self,
        geometry_dim: int,
        signal_dim: int,
        out_dim: int,
        hidden: int = 224,
        depth: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.geometry_dim = int(geometry_dim)
        self.signal_dim = int(signal_dim)
        self.hidden = int(hidden)
        film_depth = max(int(depth), 2)
        self.geometry_tower = ResidualMLP(int(geometry_dim), self.hidden, hidden=max(self.hidden, 128), depth=2, dropout=float(dropout))
        self.signal_in = nn.Sequential(
            nn.Linear(int(signal_dim), self.hidden),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.signal_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden),
                    nn.Linear(self.hidden, self.hidden),
                    nn.GELU(),
                    nn.Dropout(p=float(dropout)),
                    nn.Linear(self.hidden, self.hidden),
                    nn.Dropout(p=float(dropout)),
                )
                for _ in range(film_depth)
            ]
        )
        self.gamma_heads = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(film_depth)])
        self.beta_heads = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(film_depth)])
        self.out = ResidualMLP(self.hidden * 2, int(out_dim), hidden=self.hidden, depth=2, dropout=float(dropout))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        geom = x[:, : self.geometry_dim]
        sig = x[:, self.geometry_dim : self.geometry_dim + self.signal_dim]
        g = self.geometry_tower(geom)
        h = self.signal_in(sig)
        for block, gamma_head, beta_head in zip(self.signal_blocks, self.gamma_heads, self.beta_heads):
            gamma = 1.0 + 0.10 * torch.tanh(gamma_head(g))
            beta = 0.10 * beta_head(g)
            h = h + gamma * block(h) + beta
        return self.out(torch.cat([h, g], dim=1))


def _longest_edge_permutation(ref_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ref = np.asarray(ref_positions, dtype=np.float64)
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_lengths = np.asarray([np.linalg.norm(ref[i] - ref[j]) for i, j in pairs], dtype=np.float64)
    pair_idx = int(np.argmax(pair_lengths))
    i, j = pairs[pair_idx]
    k = int([idx for idx in (0, 1, 2) if idx not in (i, j)][0])
    origin = ref[i].copy()
    scale = float(max(pair_lengths[pair_idx], 1.0e-12))
    ex = (ref[j] - ref[i]) / scale
    ey = np.array([-ex[1], ex[0]], dtype=np.float64)
    third_local_y = float(np.dot(ref[k] - origin, ey) / scale)
    if third_local_y < 0.0:
        i, j = j, i
        origin = ref[i].copy()
        ex = (ref[j] - ref[i]) / scale
    perm = np.asarray([i, j, k], dtype=np.int64)
    return perm, origin, ex.astype(np.float64), scale


def _apply_canonical_transform(points: np.ndarray, origin: np.ndarray, ex: np.ndarray, scale: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ey = np.array([-ex[1], ex[0]], dtype=np.float64)
    rel = pts - origin[None, :]
    x = rel @ ex / float(scale)
    y = rel @ ey / float(scale)
    return np.stack([x, y], axis=-1).astype(np.float32)


def canonicalize_ref_geometry(ref_positions: np.ndarray, source_position: np.ndarray) -> dict[str, np.ndarray | float]:
    perm, origin, ex, scale = _longest_edge_permutation(ref_positions)
    refs_perm = np.asarray(ref_positions, dtype=np.float64)[perm]
    src = np.asarray(source_position, dtype=np.float64)
    refs_canonical = _apply_canonical_transform(refs_perm, origin, ex, scale)
    src_canonical = _apply_canonical_transform(src[None, :], origin, ex, scale)[0]
    return {
        "perm": perm.astype(np.int64),
        "origin": origin.astype(np.float32),
        "basis_x": ex.astype(np.float32),
        "scale": float(scale),
        "ref_canonical": refs_canonical.astype(np.float32),
        "source_canonical": src_canonical.astype(np.float32),
    }


def invert_canonical_points(points_canonical: np.ndarray, origin: np.ndarray, basis_x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_canonical, dtype=np.float64)
    org = np.asarray(origin, dtype=np.float64)
    ex = np.asarray(basis_x, dtype=np.float64)
    scl = np.asarray(scale, dtype=np.float64).reshape(-1, 1)
    ey = np.stack([-ex[:, 1], ex[:, 0]], axis=1)
    return (org + scl * (pts[:, [0]] * ex + pts[:, [1]] * ey)).astype(np.float32)


def canonical_pairwise_distance_deltas(source_canonical: np.ndarray, ref_canonical: np.ndarray) -> np.ndarray:
    src = np.asarray(source_canonical, dtype=np.float64)
    refs = np.asarray(ref_canonical, dtype=np.float64)
    dists = np.linalg.norm(src[None, :] - refs, axis=1)
    return np.asarray([dists[0] - dists[1], dists[0] - dists[2], dists[1] - dists[2]], dtype=np.float32)


def canonical_barycentric(source_canonical: np.ndarray, ref_canonical: np.ndarray) -> np.ndarray:
    tri = np.asarray(ref_canonical, dtype=np.float64)
    src = np.asarray(source_canonical, dtype=np.float64)
    mat = np.array(
        [
            [tri[0, 0], tri[1, 0], tri[2, 0]],
            [tri[0, 1], tri[1, 1], tri[2, 1]],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    rhs = np.array([src[0], src[1], 1.0], dtype=np.float64)
    return np.linalg.solve(mat, rhs).astype(np.float32)


def barycentric_to_canonical(barycentric: np.ndarray, ref_canonical: np.ndarray) -> np.ndarray:
    lam = np.asarray(barycentric, dtype=np.float64)
    tri = np.asarray(ref_canonical, dtype=np.float64)
    return (lam[:, :, None] * tri).sum(axis=1).astype(np.float32)


def _crop_peak_windows(gcc_triplet: np.ndarray, window_len: int = PEAK_WINDOW_LEN) -> np.ndarray:
    gcc = np.asarray(gcc_triplet, dtype=np.float32)
    half = int(window_len // 2)
    out = np.zeros((gcc.shape[0], int(window_len)), dtype=np.float32)
    for pair_idx in range(gcc.shape[0]):
        peak = int(round(float(peak_index_subsample(gcc[pair_idx]))))
        start = peak - half
        end = start + int(window_len)
        src_left = max(start, 0)
        src_right = min(end, gcc.shape[1])
        dst_left = src_left - start
        dst_right = dst_left + (src_right - src_left)
        out[pair_idx, dst_left:dst_right] = gcc[pair_idx, src_left:src_right]
    return out


def build_canonical_stage01_bundle(h5_path: Path | str) -> dict[str, Any]:
    cfg, data, splits = load_localization_dataset(h5_path)
    n = int(data["ref_positions"].shape[0])
    geom_name_to_idx = {str(name): idx for idx, name in enumerate(np.asarray(data["geometry_metric_names"]).tolist())}
    cond_idx = int(geom_name_to_idx["jacobian_condition"])
    ref_canonical = np.zeros((n, 3, 2), dtype=np.float32)
    source_canonical = np.zeros((n, 2), dtype=np.float32)
    origin = np.zeros((n, 2), dtype=np.float32)
    basis_x = np.zeros((n, 2), dtype=np.float32)
    scale = np.zeros((n,), dtype=np.float32)
    perm = np.zeros((n, 3), dtype=np.int64)
    tdoa_norm = np.zeros((n, 3), dtype=np.float32)
    barycentric = np.zeros((n, 3), dtype=np.float32)
    canonical_gcc = np.zeros_like(data["gcc_phat"], dtype=np.float32)
    canonical_gcc_peak = np.zeros((n, 3, PEAK_WINDOW_LEN), dtype=np.float32)
    canonical_cov = np.zeros_like(data["reference_covariance_features"], dtype=np.float32)
    canonical_ref_log_rms = np.zeros((n, 3), dtype=np.float32)
    canonical_ref_log_rms_abs = np.zeros((n, 3), dtype=np.float32)
    canonical_ref_log_rms_mean = np.zeros((n, 1), dtype=np.float32)
    for idx in range(n):
        geom = canonicalize_ref_geometry(data["ref_positions"][idx], data["source_position"][idx])
        perm[idx] = np.asarray(geom["perm"], dtype=np.int64)
        origin[idx] = np.asarray(geom["origin"], dtype=np.float32)
        basis_x[idx] = np.asarray(geom["basis_x"], dtype=np.float32)
        scale[idx] = float(geom["scale"])
        ref_canonical[idx] = np.asarray(geom["ref_canonical"], dtype=np.float32)
        source_canonical[idx] = np.asarray(geom["source_canonical"], dtype=np.float32)
        tdoa_norm[idx] = canonical_pairwise_distance_deltas(source_canonical[idx], ref_canonical[idx]).astype(np.float32)
        barycentric[idx] = canonical_barycentric(source_canonical[idx], ref_canonical[idx]).astype(np.float32)
        x_ref_perm = np.asarray(data["x_ref"][idx], dtype=np.float32)[perm[idx]]
        rms = np.sqrt(np.mean(np.square(x_ref_perm), axis=1) + 1.0e-8).astype(np.float32)
        log_rms = np.log(rms + 1.0e-8).astype(np.float32)
        canonical_ref_log_rms_abs[idx] = log_rms
        canonical_ref_log_rms_mean[idx, 0] = float(np.mean(log_rms))
        canonical_ref_log_rms[idx] = (log_rms - canonical_ref_log_rms_mean[idx, 0]).astype(np.float32)
        canonical_gcc[idx] = compute_gcc_triplet(x_ref_perm, out_len=int(cfg.gcc_len), phat=True).astype(np.float32)
        canonical_gcc_peak[idx] = _crop_peak_windows(canonical_gcc[idx], window_len=PEAK_WINDOW_LEN).astype(np.float32)
        canonical_cov[idx] = compute_reference_covariance_features(x_ref_perm, n_fft=int(cfg.psd_nfft)).astype(np.float32)
    shape_vec = ref_canonical[:, 2, :].astype(np.float32)
    area_norm = (0.5 * np.abs(shape_vec[:, 1])).astype(np.float32)
    jacobian_condition = np.asarray(data["geometry_metrics"][:, cond_idx], dtype=np.float32)
    geometry_tower_input = np.concatenate(
        [
            ref_canonical.reshape(n, -1).astype(np.float32),
            shape_vec.astype(np.float32),
            area_norm[:, None].astype(np.float32),
            np.log1p(np.maximum(jacobian_condition, 0.0)).astype(np.float32)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "cfg": cfg,
        "data": data,
        "splits": splits,
        "canonical": {
            "ref_canonical": ref_canonical,
            "source_canonical": source_canonical,
            "origin": origin,
            "basis_x": basis_x,
            "scale": scale,
            "perm": perm,
            "shape_vec": shape_vec,
            "area_norm": area_norm,
            "jacobian_condition": jacobian_condition.astype(np.float32),
            "geometry_tower_input": geometry_tower_input,
            "tdoa_norm": tdoa_norm,
            "ref_log_rms": canonical_ref_log_rms,
            "ref_log_rms_centered": canonical_ref_log_rms,
            "ref_log_rms_abs": canonical_ref_log_rms_abs,
            "ref_log_rms_mean": canonical_ref_log_rms_mean,
            "barycentric": barycentric,
            "gcc_phat": canonical_gcc,
            "gcc_peak": canonical_gcc_peak,
            "reference_covariance_features": canonical_cov,
        },
    }


def build_stage01_candidate_features(bundle: dict[str, Any], feature_kind: str) -> np.ndarray:
    data = bundle["data"]
    canonical = bundle["canonical"]
    ref_flat = data["ref_positions"].reshape(data["ref_positions"].shape[0], -1)
    if feature_kind == "tdoa_global":
        return np.concatenate([ref_flat, np.asarray(data["true_tdoa"], dtype=np.float32)], axis=1).astype(np.float32)
    if feature_kind == "tdoa_canonical":
        return np.concatenate(
            [
                canonical["shape_vec"].astype(np.float32),
                canonical["tdoa_norm"].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    if feature_kind == "tdoa_energy_canonical":
        return np.concatenate(
            [
                canonical["shape_vec"].astype(np.float32),
                canonical["tdoa_norm"].astype(np.float32),
                canonical["ref_log_rms"].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    if feature_kind == "tdoa_energy_abs_canonical":
        return np.concatenate(
            [
                canonical["shape_vec"].astype(np.float32),
                canonical["tdoa_norm"].astype(np.float32),
                canonical["ref_log_rms_centered"].astype(np.float32),
                canonical["ref_log_rms_abs"].astype(np.float32),
                canonical["ref_log_rms_mean"].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    if feature_kind == "two_tower_tdoa_energy":
        return np.concatenate(
            [
                canonical["geometry_tower_input"].astype(np.float32),
                canonical["tdoa_norm"].astype(np.float32),
                canonical["ref_log_rms"].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    if feature_kind == "two_tower_tdoa_energy_abs":
        return np.concatenate(
            [
                canonical["geometry_tower_input"].astype(np.float32),
                canonical["tdoa_norm"].astype(np.float32),
                canonical["ref_log_rms_centered"].astype(np.float32),
                canonical["ref_log_rms_abs"].astype(np.float32),
                canonical["ref_log_rms_mean"].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    if feature_kind == "two_tower_gcc_peak_energy":
        return np.concatenate(
            [
                canonical["geometry_tower_input"].astype(np.float32),
                canonical["gcc_peak"].reshape(canonical["gcc_peak"].shape[0], -1).astype(np.float32),
                canonical["ref_log_rms"].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    if feature_kind == "solver_gcc_peak_energy_abs":
        return np.concatenate(
            [
                canonical["geometry_tower_input"].astype(np.float32),
                canonical["gcc_peak"].reshape(canonical["gcc_peak"].shape[0], -1).astype(np.float32),
                canonical["tdoa_norm"].astype(np.float32),
                canonical["ref_log_rms_centered"].astype(np.float32),
                canonical["ref_log_rms_abs"].astype(np.float32),
                canonical["ref_log_rms_mean"].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    if feature_kind == "two_tower_gcc_full_energy":
        return np.concatenate(
            [
                canonical["geometry_tower_input"].astype(np.float32),
                canonical["gcc_phat"].reshape(canonical["gcc_phat"].shape[0], -1).astype(np.float32),
                canonical["ref_log_rms"].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    if feature_kind == "solver_gcc_full_energy_abs":
        return np.concatenate(
            [
                canonical["geometry_tower_input"].astype(np.float32),
                canonical["gcc_phat"].reshape(canonical["gcc_phat"].shape[0], -1).astype(np.float32),
                canonical["tdoa_norm"].astype(np.float32),
                canonical["ref_log_rms_centered"].astype(np.float32),
                canonical["ref_log_rms_abs"].astype(np.float32),
                canonical["ref_log_rms_mean"].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
    raise KeyError(f"Unsupported Stage 01 stable feature kind: {feature_kind}")


def build_stage01_feature_layout(bundle: dict[str, Any], feature_kind: str) -> dict[str, Any]:
    canonical = bundle["canonical"]
    geom_dim = int(canonical["geometry_tower_input"].shape[1])
    if feature_kind == "tdoa_global":
        return {"input_dim": 9, "geometry_dim": 0, "signal_dim": 9, "signal_mode": "mlp"}
    if feature_kind == "tdoa_canonical":
        return {
            "input_dim": 5,
            "geometry_dim": 0,
            "signal_dim": 5,
            "signal_mode": "mlp",
            "shape_slice": slice(0, 2),
            "tdoa_slice": slice(2, 5),
        }
    if feature_kind == "tdoa_energy_canonical":
        return {
            "input_dim": 8,
            "geometry_dim": 0,
            "signal_dim": 8,
            "signal_mode": "mlp",
            "shape_slice": slice(0, 2),
            "tdoa_slice": slice(2, 5),
        }
    if feature_kind == "tdoa_energy_abs_canonical":
        return {
            "input_dim": 12,
            "geometry_dim": 0,
            "signal_dim": 12,
            "signal_mode": "mlp",
            "shape_slice": slice(0, 2),
            "tdoa_slice": slice(2, 5),
        }
    if feature_kind == "two_tower_tdoa_energy":
        return {
            "input_dim": geom_dim + 6,
            "geometry_dim": geom_dim,
            "signal_dim": 6,
            "signal_mode": "mlp",
            "shape_slice": GEOM_SHAPE_SLICE,
            "tdoa_slice": slice(geom_dim, geom_dim + 3),
        }
    if feature_kind == "two_tower_tdoa_energy_abs":
        return {
            "input_dim": geom_dim + 10,
            "geometry_dim": geom_dim,
            "signal_dim": 10,
            "signal_mode": "mlp",
            "shape_slice": GEOM_SHAPE_SLICE,
            "tdoa_slice": slice(geom_dim, geom_dim + 3),
        }
    if feature_kind == "two_tower_gcc_peak_energy":
        signal_dim = int(3 * PEAK_WINDOW_LEN + 3)
        return {
            "input_dim": geom_dim + signal_dim,
            "geometry_dim": geom_dim,
            "signal_dim": signal_dim,
            "signal_mode": "pairwise_conv",
            "pair_count": 3,
            "seq_len": PEAK_WINDOW_LEN,
            "extra_dim": 3,
        }
    if feature_kind == "solver_gcc_peak_energy_abs":
        signal_dim = int(3 * PEAK_WINDOW_LEN + 10)
        return {
            "input_dim": geom_dim + signal_dim,
            "geometry_dim": geom_dim,
            "signal_dim": signal_dim,
            "signal_mode": "pairwise_conv",
            "pair_count": 3,
            "seq_len": PEAK_WINDOW_LEN,
            "extra_dim": 10,
            "shape_slice": GEOM_SHAPE_SLICE,
            "tdoa_slice": slice(geom_dim + (3 * PEAK_WINDOW_LEN), geom_dim + (3 * PEAK_WINDOW_LEN) + 3),
        }
    if feature_kind == "two_tower_gcc_full_energy":
        seq_len = int(canonical["gcc_phat"].shape[-1])
        signal_dim = int(3 * seq_len + 3)
        return {
            "input_dim": geom_dim + signal_dim,
            "geometry_dim": geom_dim,
            "signal_dim": signal_dim,
            "signal_mode": "pairwise_conv",
            "pair_count": 3,
            "seq_len": seq_len,
            "extra_dim": 3,
        }
    if feature_kind == "solver_gcc_full_energy_abs":
        seq_len = int(canonical["gcc_phat"].shape[-1])
        signal_dim = int(3 * seq_len + 10)
        return {
            "input_dim": geom_dim + signal_dim,
            "geometry_dim": geom_dim,
            "signal_dim": signal_dim,
            "signal_mode": "pairwise_conv",
            "pair_count": 3,
            "seq_len": seq_len,
            "extra_dim": 10,
            "shape_slice": GEOM_SHAPE_SLICE,
            "tdoa_slice": slice(geom_dim + (3 * seq_len), geom_dim + (3 * seq_len) + 3),
        }
    raise KeyError(f"Unsupported feature layout request: {feature_kind}")


def build_stage01_candidate_targets(bundle: dict[str, Any], target_kind: str) -> np.ndarray:
    data = bundle["data"]
    canonical = bundle["canonical"]
    if target_kind == "global_xy":
        return np.asarray(data["source_position"], dtype=np.float32)
    if target_kind == "canonical_xy":
        return np.asarray(canonical["source_canonical"], dtype=np.float32)
    if target_kind == "anchor_range":
        return np.linalg.norm(np.asarray(canonical["source_canonical"], dtype=np.float32), axis=1, keepdims=True).astype(np.float32)
    if target_kind == "barycentric":
        return np.asarray(canonical["barycentric"], dtype=np.float32)
    raise KeyError(f"Unsupported target kind: {target_kind}")


def _zero_last_linear_layer(model: nn.Module) -> None:
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            nn.init.zeros_(last_linear.bias)


def build_stage01_model(layout: dict[str, Any], out_dim: int, spec: Stage01CandidateSpec) -> nn.Module:
    in_dim = int(layout["input_dim"])
    effective_out_dim = 9 if spec.loss_kind == "solver_huber" else int(out_dim)
    if spec.model_kind == "mlp":
        model = SimpleMLP(int(in_dim), effective_out_dim, hidden=int(spec.hidden), depth=int(spec.depth), dropout=float(spec.dropout))
        if spec.loss_kind == "solver_huber":
            _zero_last_linear_layer(model)
        return model
    if spec.model_kind == "resmlp":
        model = ResidualMLP(int(in_dim), effective_out_dim, hidden=int(spec.hidden), depth=int(spec.depth), dropout=float(spec.dropout))
        if spec.loss_kind == "solver_huber":
            _zero_last_linear_layer(model)
        return model
    if spec.model_kind == "two_tower_mlp":
        model = TwoTowerRegressor(
            geometry_dim=int(layout["geometry_dim"]),
            signal_dim=int(layout["signal_dim"]),
            out_dim=effective_out_dim,
            hidden=int(spec.hidden),
            depth=int(spec.depth),
            dropout=float(spec.dropout),
            signal_mode="mlp",
        )
        if spec.loss_kind == "solver_huber":
            _zero_last_linear_layer(model)
        return model
    if spec.model_kind == "two_tower_film":
        model = TwoTowerFiLMRegressor(
            geometry_dim=int(layout["geometry_dim"]),
            signal_dim=int(layout["signal_dim"]),
            out_dim=effective_out_dim,
            hidden=int(spec.hidden),
            depth=int(spec.depth),
            dropout=float(spec.dropout),
        )
        if spec.loss_kind == "solver_huber":
            _zero_last_linear_layer(model)
        return model
    if spec.model_kind == "two_tower_conv":
        model = TwoTowerRegressor(
            geometry_dim=int(layout["geometry_dim"]),
            signal_dim=int(layout["signal_dim"]),
            out_dim=effective_out_dim,
            hidden=int(spec.hidden),
            depth=int(spec.depth),
            dropout=float(spec.dropout),
            signal_mode=str(layout["signal_mode"]),
            pair_count=int(layout["pair_count"]),
            seq_len=int(layout["seq_len"]),
            extra_dim=int(layout["extra_dim"]),
        )
        if spec.loss_kind == "solver_huber":
            _zero_last_linear_layer(model)
        return model
    raise KeyError(f"Unsupported model kind: {spec.model_kind}")


def decode_stage01_prediction(pred: torch.Tensor, spec: Stage01CandidateSpec, ref_shape: torch.Tensor) -> torch.Tensor:
    if spec.target_kind in {"global_xy", "canonical_xy"}:
        return pred
    if spec.target_kind == "anchor_range":
        raise RuntimeError("anchor_range decoding requires raw feature tensor; use decode_anchor_range_to_canonical_torch.")
    if spec.target_kind == "barycentric":
        r0 = torch.zeros((pred.shape[0], 2), device=pred.device, dtype=pred.dtype)
        r1 = torch.stack([torch.ones_like(ref_shape[:, 0]), torch.zeros_like(ref_shape[:, 0])], dim=1)
        r2 = ref_shape
        tri = torch.stack([r0, r1, r2], dim=1)
        return torch.sum(pred.unsqueeze(-1) * tri, dim=1)
    raise KeyError(f"Unsupported target kind: {spec.target_kind}")


def canonical_tdoa_from_xy_torch(xy: torch.Tensor, ref_shape: torch.Tensor) -> torch.Tensor:
    r0 = torch.zeros((xy.shape[0], 2), device=xy.device, dtype=xy.dtype)
    r1 = torch.stack([torch.ones_like(ref_shape[:, 0]), torch.zeros_like(ref_shape[:, 0])], dim=1)
    r2 = ref_shape
    refs = torch.stack([r0, r1, r2], dim=1)
    dist = torch.linalg.norm(xy[:, None, :] - refs, dim=-1)
    return torch.stack([dist[:, 0] - dist[:, 1], dist[:, 0] - dist[:, 2], dist[:, 1] - dist[:, 2]], dim=1)


def canonical_refs_from_shape_torch(ref_shape: torch.Tensor) -> torch.Tensor:
    r0 = torch.zeros((ref_shape.shape[0], 2), device=ref_shape.device, dtype=ref_shape.dtype)
    r1 = torch.stack([torch.ones_like(ref_shape[:, 0]), torch.zeros_like(ref_shape[:, 0])], dim=1)
    r2 = ref_shape
    return torch.stack([r0, r1, r2], dim=1)


def tdoa_residual_and_jacobian_torch(
    xy: torch.Tensor,
    ref_shape: torch.Tensor,
    target_tdoa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    refs = canonical_refs_from_shape_torch(ref_shape)
    delta = xy[:, None, :] - refs
    dist = torch.linalg.norm(delta, dim=-1).clamp_min(1.0e-6)
    unit = delta / dist[:, :, None]
    pred_tdoa = torch.stack([dist[:, 0] - dist[:, 1], dist[:, 0] - dist[:, 2], dist[:, 1] - dist[:, 2]], dim=1)
    residual = pred_tdoa - target_tdoa
    jac = torch.stack(
        [
            unit[:, 0, :] - unit[:, 1, :],
            unit[:, 0, :] - unit[:, 2, :],
            unit[:, 1, :] - unit[:, 2, :],
        ],
        dim=1,
    )
    return residual, jac


def differentiable_tdoa_solver_torch(
    anchor_xy: torch.Tensor,
    ref_shape: torch.Tensor,
    corrected_tdoa: torch.Tensor,
    weight_logits: torch.Tensor,
    raw_mu: torch.Tensor,
    steps: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = anchor_xy
    weights = F.softplus(weight_logits) + 1.0e-3
    mu = F.softplus(raw_mu) + 1.0e-3
    eye = torch.eye(2, device=x.device, dtype=x.dtype).unsqueeze(0)
    residual = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)
    for _ in range(int(steps)):
        residual, jac = tdoa_residual_and_jacobian_torch(x, ref_shape, corrected_tdoa)
        weighted_jac = weights[:, :, None] * jac
        h = torch.matmul(jac.transpose(1, 2), weighted_jac) + mu[:, None, :] * eye
        g = torch.matmul(jac.transpose(1, 2), (weights * residual)[:, :, None])
        step = torch.linalg.solve(h, g).squeeze(-1)
        step = torch.clamp(step, min=-1.5, max=1.5)
        x = x - step
    residual, _ = tdoa_residual_and_jacobian_torch(x, ref_shape, corrected_tdoa)
    return x, residual


def decode_anchor_range_to_canonical_torch(pred_range_raw: torch.Tensor, raw_feature_tensor: torch.Tensor) -> torch.Tensor:
    d0 = F.softplus(pred_range_raw[:, 0]) + 1.0e-6
    a = raw_feature_tensor[:, 0]
    b = torch.clamp(raw_feature_tensor[:, 1], min=1.0e-4)
    q01 = raw_feature_tensor[:, 2]
    q02 = raw_feature_tensor[:, 3]
    x = q01 * d0 + 0.5 * (1.0 - q01 * q01)
    y = (q02 * d0 + 0.5 * (a * a + b * b - q02 * q02) - a * x) / b
    return torch.stack([x, y], dim=1)


def decode_anchor_range_to_canonical_np(pred_range_raw: np.ndarray, raw_feature_array: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_range_raw, dtype=np.float32).reshape(-1)
    raw = np.asarray(raw_feature_array, dtype=np.float32)
    d0 = np.log1p(np.exp(pred)) + 1.0e-6
    a = raw[:, 0]
    b = np.maximum(raw[:, 1], 1.0e-4)
    q01 = raw[:, 2]
    q02 = raw[:, 3]
    x = q01 * d0 + 0.5 * (1.0 - q01 * q01)
    y = (q02 * d0 + 0.5 * (a * a + b * b - q02 * q02) - a * x) / b
    return np.stack([x, y], axis=1).astype(np.float32)


def _physics_shape_and_tdoa(raw_feature_tensor: torch.Tensor, spec: Stage01CandidateSpec, layout: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.feature_kind in {"tdoa_canonical", "tdoa_energy_canonical", "tdoa_energy_abs_canonical"}:
        shape = raw_feature_tensor[:, 0:2]
        tdoa = raw_feature_tensor[:, 2:5]
        return shape, tdoa
    if spec.feature_kind in {"two_tower_tdoa_energy", "two_tower_tdoa_energy_abs", "solver_gcc_peak_energy_abs", "solver_gcc_full_energy_abs"}:
        shape_slice = layout["shape_slice"]
        tdoa_slice = layout["tdoa_slice"]
        shape = raw_feature_tensor[:, shape_slice]
        tdoa = raw_feature_tensor[:, tdoa_slice]
        return shape, tdoa
    raise KeyError(f"Physics loss is not supported for feature kind: {spec.feature_kind}")


def _solver_decode_torch(
    pred: torch.Tensor,
    raw_feature_tensor: torch.Tensor,
    spec: Stage01CandidateSpec,
    layout: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    shape, base_tdoa = _physics_shape_and_tdoa(raw_feature_tensor, spec, layout)
    coarse_init = torch.stack(
        [
            (1.0 + shape[:, 0]) / 3.0,
            shape[:, 1] / 3.0,
        ],
        dim=1,
    )
    coarse_xy, _ = differentiable_tdoa_solver_torch(
        coarse_init,
        shape,
        base_tdoa,
        torch.zeros_like(base_tdoa),
        torch.full((pred.shape[0], 1), -2.0, device=pred.device, dtype=pred.dtype),
        steps=max(6, int(spec.solver_steps) + 1),
    )
    anchor_xy = coarse_xy + 0.35 * torch.tanh(pred[:, 0:2])
    delta_tdoa = 0.05 * torch.tanh(pred[:, 2:5])
    weight_logits = 0.25 * pred[:, 5:8]
    raw_mu = 0.25 * pred[:, 8:9]
    corrected_tdoa = base_tdoa + delta_tdoa
    solver_xy, residual = differentiable_tdoa_solver_torch(
        anchor_xy,
        shape,
        corrected_tdoa,
        weight_logits,
        raw_mu,
        steps=int(spec.solver_steps),
    )
    return solver_xy, {
        "coarse_xy": coarse_xy,
        "anchor_xy": anchor_xy,
        "corrected_tdoa": corrected_tdoa,
        "residual": residual,
    }


def _solver_decode_np(
    pred: np.ndarray,
    raw_feature_array: np.ndarray,
    spec: Stage01CandidateSpec,
    layout: dict[str, Any],
) -> np.ndarray:
    pred_t = torch.from_numpy(np.asarray(pred, dtype=np.float32))
    raw_t = torch.from_numpy(np.asarray(raw_feature_array, dtype=np.float32))
    with torch.no_grad():
        xy, _ = _solver_decode_torch(pred_t, raw_t, spec, layout)
    return xy.cpu().numpy().astype(np.float32)


def find_tdoa_room_solutions(
    lag_samples: np.ndarray,
    ref_positions: np.ndarray,
    room_size: tuple[float, float],
    fs: int,
    c: float,
    grid_size: int = 11,
    residual_tol: float = 1.0e-5,
    merge_tol: float = 1.0e-3,
) -> np.ndarray:
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

    xs = np.linspace(0.10, float(room[0]) - 0.10, int(grid_size))
    ys = np.linspace(0.10, float(room[1]) - 0.10, int(grid_size))
    sols: list[np.ndarray] = []
    for x0 in xs:
        for y0 in ys:
            res = least_squares(
                residual,
                x0=np.array([x0, y0], dtype=np.float64),
                bounds=([0.10, 0.10], [float(room[0]) - 0.10, float(room[1]) - 0.10]),
                method="trf",
                max_nfev=300,
            )
            if not bool(res.success):
                continue
            xy = np.asarray(res.x, dtype=np.float64)
            if float(np.linalg.norm(residual(xy))) > float(residual_tol):
                continue
            duplicate = False
            for prev in sols:
                if float(np.linalg.norm(prev - xy)) <= float(merge_tol):
                    duplicate = True
                    break
            if not duplicate:
                sols.append(xy)
    if not sols:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(sols, axis=0).astype(np.float32)


def _candidate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    raw_feature_tensor: torch.Tensor,
    spec: Stage01CandidateSpec,
    layout: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if spec.loss_kind == "mse":
        loss = F.mse_loss(pred, target)
        return loss, {"main_loss": float(loss.detach().cpu()), "physics_loss": 0.0}
    if spec.loss_kind == "huber":
        loss = F.smooth_l1_loss(pred, target)
        return loss, {"main_loss": float(loss.detach().cpu()), "physics_loss": 0.0}
    if spec.loss_kind == "barycentric_huber":
        loss = F.smooth_l1_loss(pred, target)
        return loss, {"main_loss": float(loss.detach().cpu()), "physics_loss": 0.0}
    if spec.loss_kind == "anchor_range_huber":
        shape, tdoa = _physics_shape_and_tdoa(raw_feature_tensor, spec, layout)
        compact = torch.cat([shape, tdoa[:, :2]], dim=1)
        pred_xy = decode_anchor_range_to_canonical_torch(pred, compact)
        target_xy = decode_anchor_range_to_canonical_torch(target, compact)
        main_loss = F.smooth_l1_loss(pred_xy, target_xy)
        physics_loss = F.smooth_l1_loss(canonical_tdoa_from_xy_torch(pred_xy, shape), tdoa)
        loss = main_loss + 0.10 * physics_loss
        return loss, {
            "main_loss": float(main_loss.detach().cpu()),
            "physics_loss": float(physics_loss.detach().cpu()),
        }
    if spec.loss_kind == "physics_huber":
        shape, tdoa = _physics_shape_and_tdoa(raw_feature_tensor, spec, layout)
        pred_xy = decode_stage01_prediction(pred, spec, shape)
        main_loss = F.smooth_l1_loss(pred_xy, target)
        physics_loss = F.smooth_l1_loss(canonical_tdoa_from_xy_torch(pred_xy, shape), tdoa)
        loss = main_loss + float(spec.physics_weight) * physics_loss
        return loss, {
            "main_loss": float(main_loss.detach().cpu()),
            "physics_loss": float(physics_loss.detach().cpu()),
        }
    if spec.loss_kind == "solver_huber":
        pred_xy, aux = _solver_decode_torch(pred, raw_feature_tensor, spec, layout)
        shape, tdoa = _physics_shape_and_tdoa(raw_feature_tensor, spec, layout)
        main_loss = F.smooth_l1_loss(pred_xy, target)
        tdoa_loss = F.smooth_l1_loss(aux["corrected_tdoa"], tdoa)
        anchor_loss = F.smooth_l1_loss(aux["anchor_xy"], target)
        residual_loss = torch.mean(aux["residual"] ** 2)
        physics_loss = float(spec.physics_weight) * tdoa_loss + 0.10 * anchor_loss + 0.10 * residual_loss
        loss = main_loss + physics_loss
        return loss, {
            "main_loss": float(main_loss.detach().cpu()),
            "physics_loss": float(physics_loss.detach().cpu()),
        }
    raise KeyError(f"Unsupported loss kind: {spec.loss_kind}")


def invert_predictions_to_global(pred_target: np.ndarray, bundle: dict[str, Any], spec: Stage01CandidateSpec, sample_idx: np.ndarray) -> np.ndarray:
    canonical = bundle["canonical"]
    idx = np.asarray(sample_idx, dtype=np.int64)
    if spec.target_kind == "global_xy":
        return np.asarray(pred_target, dtype=np.float32)
    if spec.loss_kind == "solver_huber":
        raw_feat = build_stage01_candidate_features(bundle, spec.feature_kind)[idx]
        layout = build_stage01_feature_layout(bundle, spec.feature_kind)
        canonical_xy = _solver_decode_np(np.asarray(pred_target, dtype=np.float32), raw_feat, spec, layout)
    elif spec.target_kind == "canonical_xy":
        canonical_xy = np.asarray(pred_target, dtype=np.float32)
    elif spec.target_kind == "anchor_range":
        raw_feat = build_stage01_candidate_features(bundle, spec.feature_kind)[idx]
        canonical_xy = decode_anchor_range_to_canonical_np(np.asarray(pred_target, dtype=np.float32), raw_feat)
    elif spec.target_kind == "barycentric":
        canonical_xy = barycentric_to_canonical(np.asarray(pred_target, dtype=np.float32), canonical["ref_canonical"][idx])
    else:
        raise KeyError(f"Unsupported target kind: {spec.target_kind}")
    return invert_canonical_points(
        canonical_xy,
        canonical["origin"][idx],
        canonical["basis_x"][idx],
        canonical["scale"][idx],
    )


def _predict_candidate(model: nn.Module, x: np.ndarray, x_stats: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x_std = ((np.asarray(x, dtype=np.float32) - x_stats["mean"]) / x_stats["std"]).astype(np.float32)
        pred = model(torch.from_numpy(x_std).to(device=device, dtype=torch.float32)).cpu().numpy()
    return pred.astype(np.float32)


def train_stage01_candidate(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_layout: dict[str, Any],
    spec: Stage01CandidateSpec,
    output_dir: Path,
    split_kind: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    live_plot: bool = False,
) -> dict[str, Any]:
    set_seed(int(seed))
    train_x = np.asarray(feature_array[train_idx], dtype=np.float32)
    val_x = np.asarray(feature_array[val_idx], dtype=np.float32)
    train_y = np.asarray(target_array[train_idx], dtype=np.float32)
    val_y = np.asarray(target_array[val_idx], dtype=np.float32)
    x_stats = fit_standardizer(train_x, np.arange(train_x.shape[0], dtype=np.int64))
    train_x_std = ((train_x - x_stats["mean"]) / x_stats["std"]).astype(np.float32)
    val_x_std = ((val_x - x_stats["mean"]) / x_stats["std"]).astype(np.float32)
    model = build_stage01_model(feature_layout, int(train_y.shape[1]), spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x_std), torch.from_numpy(train_y), torch.from_numpy(train_x)),
        batch_size=int(batch_size),
        shuffle=True,
    )
    val_x_t = torch.from_numpy(val_x_std).to(device=device, dtype=torch.float32)
    val_y_t = torch.from_numpy(val_y).to(device=device, dtype=torch.float32)
    val_x_raw_t = torch.from_numpy(val_x).to(device=device, dtype=torch.float32)
    best_state = None
    best_score = (float("inf"), float("inf"), float("inf"))
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_main = 0.0
        total_phys = 0.0
        total_count = 0
        for xb, yb, xraw in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            xraw = xraw.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, loss_terms = _candidate_loss(pred, yb, xraw, spec, feature_layout)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_n = int(xb.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_n
            total_main += float(loss_terms["main_loss"]) * batch_n
            total_phys += float(loss_terms["physics_loss"]) * batch_n
            total_count += batch_n
        model.eval()
        with torch.no_grad():
            pred_val = model(val_x_t)
            val_loss, loss_terms = _candidate_loss(pred_val, val_y_t, val_x_raw_t, spec, feature_layout)
        pred_val_np = pred_val.detach().cpu().numpy().astype(np.float32)
        target_metric = val_y
        if spec.target_kind == "anchor_range":
            pred_metric = decode_anchor_range_to_canonical_np(pred_val_np, val_x)
            target_metric = decode_anchor_range_to_canonical_np(val_y, val_x)
        elif spec.target_kind == "barycentric":
            ref2 = val_x[:, :2]
            tri = np.zeros((val_x.shape[0], 3, 2), dtype=np.float32)
            tri[:, 1, 0] = 1.0
            tri[:, 2, :] = ref2
            pred_metric = barycentric_to_canonical(pred_val_np, tri)
            target_metric = barycentric_to_canonical(val_y, tri)
        elif spec.loss_kind == "solver_huber":
            pred_metric = _solver_decode_np(pred_val_np, val_x, spec, feature_layout)
        else:
            pred_metric = pred_val_np
        val_metrics = localization_error_stats(pred_metric, target_metric)
        history_row = {
            "epoch": int(epoch),
            "train_loss": total_loss / max(total_count, 1),
            "train_main_loss": total_main / max(total_count, 1),
            "train_physics_loss": total_phys / max(total_count, 1),
            "val_loss": float(val_loss.detach().cpu()),
            "val_main_loss": float(loss_terms["main_loss"]),
            "val_physics_loss": float(loss_terms["physics_loss"]),
            "val_median_error_m": float(val_metrics["median_m"]),
            "val_p90_error_m": float(val_metrics["p90_m"]),
            "best_so_far_m": float(min([row["val_median_error_m"] for row in history_rows] + [float(val_metrics["median_m"])])),
        }
        history_rows.append(history_row)
        csv_path = output_dir / f"{split_kind}_{spec.candidate_id}_epoch_metrics.csv"
        save_history_csv(csv_path, history_rows)
        render_training_curves(
            history_rows,
            output_path=output_dir / f"{split_kind}_{spec.candidate_id}_loss_curves.png",
            title=f"Stage 01 stable | {split_kind} | {spec.candidate_id}",
            metric_keys=["val_median_error_m", "best_so_far_m"],
            metric_labels=["val median error", "best-so-far"],
            live_plot=bool(live_plot),
        )
        score = (float(val_metrics["median_m"]), float(val_metrics["p90_m"]), float(val_loss.detach().cpu()))
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = output_dir / f"{split_kind}_{spec.candidate_id}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "candidate_id": spec.candidate_id,
            "feature_kind": spec.feature_kind,
            "model_kind": spec.model_kind,
            "target_kind": spec.target_kind,
            "loss_kind": spec.loss_kind,
            "feature_stats": x_stats,
            "hidden": int(spec.hidden),
            "depth": int(spec.depth),
            "dropout": float(spec.dropout),
            "physics_weight": float(spec.physics_weight),
            "solver_steps": int(spec.solver_steps),
            "best_val_median_error_m": float(best_score[0]),
            "best_val_p90_error_m": float(best_score[1]),
        },
        str(ckpt_path),
    )
    return {
        "checkpoint_path": str(ckpt_path),
        "feature_kind": spec.feature_kind,
        "model_kind": spec.model_kind,
        "target_kind": spec.target_kind,
        "loss_kind": spec.loss_kind,
        "physics_weight": float(spec.physics_weight),
        "solver_steps": int(spec.solver_steps),
        "best_val_median_error_m": float(best_score[0]),
        "best_val_p90_error_m": float(best_score[1]),
        "epoch_metrics_csv": str(output_dir / f"{split_kind}_{spec.candidate_id}_epoch_metrics.csv"),
        "loss_curve_path": str(output_dir / f"{split_kind}_{spec.candidate_id}_loss_curves.png"),
    }


def _load_candidate_from_ckpt(ckpt_path: Path, feature_layout: dict[str, Any], output_dim: int, device: torch.device) -> tuple[nn.Module, dict[str, np.ndarray], dict[str, Any]]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    spec = Stage01CandidateSpec(
        candidate_id=str(ckpt["candidate_id"]),
        feature_kind=str(ckpt["feature_kind"]),
        model_kind=str(ckpt["model_kind"]),
        target_kind=str(ckpt["target_kind"]),
        loss_kind=str(ckpt["loss_kind"]),
        hidden=int(ckpt.get("hidden", 256)),
        depth=int(ckpt.get("depth", 4)),
        dropout=float(ckpt.get("dropout", 0.10)),
        physics_weight=float(ckpt.get("physics_weight", PHYSICS_LOSS_WEIGHT)),
        solver_steps=int(ckpt.get("solver_steps", 5)),
    )
    model = build_stage01_model(feature_layout, int(output_dim), spec).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt["feature_stats"], {"spec": spec}


def _stage01_gate(analytic_stats: dict[str, Any], learned_stats: dict[str, Any], thresholds: dict[str, float]) -> dict[str, bool]:
    return {
        "iid_passed": float(learned_stats["iid_test"]["median_m"]) < float(thresholds["learned_iid_median_max"])
        and float(learned_stats["iid_test"]["p90_m"]) < float(thresholds["learned_iid_p90_max"]),
        "geom_passed": float(learned_stats["geom_test"]["median_m"]) < float(thresholds["learned_geom_median_max"])
        and float(learned_stats["geom_test"]["p90_m"]) < float(thresholds["learned_geom_p90_max"]),
        "iid_within_2x_analytic_p90": float(learned_stats["iid_test"]["p90_m"]) <= 2.0 * float(analytic_stats["iid_test"]["p90_m"]),
        "geom_within_2x_analytic_p90": float(learned_stats["geom_test"]["p90_m"]) <= 2.0 * float(analytic_stats["geom_test"]["p90_m"]),
    }


def _is_stage01_primary_tdoa_candidate_meta(meta: dict[str, Any]) -> bool:
    candidate_id = str(meta.get("candidate_id", ""))
    feature_kind = str(meta.get("feature_kind", ""))
    if candidate_id.endswith("_tdoa_global_baseline"):
        return True
    if candidate_id.startswith("tdoa_") or candidate_id.startswith("solver_tdoa"):
        return True
    if "_tdoa_" in candidate_id:
        return True
    if feature_kind.startswith("tdoa"):
        return True
    if feature_kind.startswith("two_tower_tdoa"):
        return True
    return False


def _plot_stage01_sample(
    output_path: Path,
    room_bounds: tuple[float, float],
    ref_positions: np.ndarray,
    true_position: np.ndarray,
    pred_position: np.ndarray,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6.2, 6.0), constrained_layout=True)
    tri = np.asarray(ref_positions, dtype=np.float64)
    true_pos = np.asarray(true_position, dtype=np.float64)
    pred_pos = np.asarray(pred_position, dtype=np.float64)
    ax.scatter(tri[:, 0], tri[:, 1], s=90, color="tab:green", label="refs")
    ax.scatter(true_pos[0], true_pos[1], s=90, color="tab:blue", label="true")
    ax.scatter(pred_pos[0], pred_pos[1], s=90, color="tab:orange", label="pred")
    ax.plot([true_pos[0], pred_pos[0]], [true_pos[1], pred_pos[1]], linestyle="--", color="tab:red", alpha=0.60)
    ax.set_xlim(0.0, float(room_bounds[0]))
    ax.set_ylim(0.0, float(room_bounds[1]))
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def train_stage01_stable_models(
    h5_path: Path | str,
    output_dir: Path | str,
    epochs: int = 40,
    batch_size: int = 128,
    device: str = "auto",
    seed: int = 20260401,
    live_plot: bool = False,
    candidate_ids: list[str] | None = None,
) -> dict[str, Any]:
    bundle = build_canonical_stage01_bundle(h5_path)
    cfg = bundle["cfg"]
    data = bundle["data"]
    splits = bundle["splits"]
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    analytic_bundle = evaluate_localization_analytic_only(cfg, data, splits, out_dir, stage_id="01", level="L1")
    if not bool(analytic_bundle["analytic_gate"]["passed"]):
        summary = {
            "h5_path": str(h5_path),
            "skipped_due_to_analytic_gate": True,
            "analytic_gate": analytic_bundle["analytic_gate"],
            "models": {},
        }
        save_json(out_dir / "train_summary.json", summary)
        return summary
    summary: dict[str, Any] = {
        "h5_path": str(h5_path),
        "geometry_filter_mode": str(cfg.geometry_filter_mode),
        "min_triangle_area": float(cfg.min_triangle_area),
        "max_jacobian_condition": float(cfg.max_jacobian_condition),
        "window_preset": infer_localization_window_preset(cfg),
        "analytic_gcc_phat": analytic_bundle["analytic_gcc_phat"],
        "models": {},
        "stage": "01_stable_geometry",
    }
    candidate_specs = stable_candidate_specs()
    allowed = {str(item) for item in candidate_ids or []}
    if allowed:
        candidate_specs = [spec for spec in candidate_specs if spec.candidate_id in allowed]
    for split_kind, train_key, val_key in [("iid", "train", "val"), ("geom", "geom_train", "geom_val")]:
        train_idx = splits[train_key]
        val_idx = splits[val_key]
        for spec in candidate_specs:
            features = build_stage01_candidate_features(bundle, spec.feature_kind)
            feature_layout = build_stage01_feature_layout(bundle, spec.feature_kind)
            targets = build_stage01_candidate_targets(bundle, spec.target_kind)
            meta = train_stage01_candidate(
                features,
                targets,
                train_idx=train_idx,
                val_idx=val_idx,
                feature_layout=feature_layout,
                spec=spec,
                output_dir=out_dir,
                split_kind=split_kind,
                epochs=int(epochs),
                batch_size=int(batch_size),
                device=dev,
                seed=int(seed),
                live_plot=bool(live_plot),
            )
            meta.update(
                {
                    "candidate_id": spec.candidate_id,
                    "split_kind": split_kind,
                    "hidden": int(spec.hidden),
                    "depth": int(spec.depth),
                    "dropout": float(spec.dropout),
                    "physics_weight": float(spec.physics_weight),
                    "solver_steps": int(spec.solver_steps),
                }
            )
            summary["models"][f"{split_kind}_{spec.candidate_id}"] = meta
    save_json(out_dir / "train_summary.json", summary)
    return summary


def evaluate_stage01_stable_candidates(
    h5_path: Path | str,
    output_dir: Path | str,
    device: str = "auto",
) -> dict[str, Any]:
    bundle = build_canonical_stage01_bundle(h5_path)
    cfg = bundle["cfg"]
    data = bundle["data"]
    splits = bundle["splits"]
    out_dir = Path(output_dir).resolve()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    analytic_bundle = evaluate_localization_analytic_only(cfg, data, splits, out_dir, stage_id="01", level="L1")
    train_summary = json.loads((out_dir / "train_summary.json").read_text(encoding="utf-8"))
    learned: dict[str, dict[str, float]] = {}
    plot_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model_key, meta in train_summary.get("models", {}).items():
        ckpt_path = Path(meta["checkpoint_path"])
        if not ckpt_path.exists():
            continue
        split_kind = str(meta["split_kind"])
        test_key = "test" if split_kind == "iid" else "geom_test"
        idx = splits[test_key]
        spec = Stage01CandidateSpec(
            candidate_id=str(meta["candidate_id"]),
            feature_kind=str(meta["feature_kind"]),
            model_kind=str(meta["model_kind"]),
            target_kind=str(meta["target_kind"]),
            loss_kind=str(meta["loss_kind"]),
            hidden=int(meta.get("hidden", 256)),
            depth=int(meta.get("depth", 4)),
            dropout=float(meta.get("dropout", 0.10)),
            physics_weight=float(meta.get("physics_weight", PHYSICS_LOSS_WEIGHT)),
            solver_steps=int(meta.get("solver_steps", 5)),
        )
        feat = build_stage01_candidate_features(bundle, spec.feature_kind)
        feature_layout = build_stage01_feature_layout(bundle, spec.feature_kind)
        target = build_stage01_candidate_targets(bundle, spec.target_kind)
        model, x_stats, _ = _load_candidate_from_ckpt(ckpt_path, feature_layout, target.shape[1], dev)
        pred_target = _predict_candidate(model, feat[idx], x_stats, dev)
        pred_global = invert_predictions_to_global(pred_target, bundle, spec, idx)
        stats = localization_error_stats(pred_global, data["source_position"][idx])
        learned[model_key] = stats
        plot_cache[model_key] = (pred_global, data["source_position"][idx], data["ref_positions"][idx])
    tdoa_keys = sorted(
        [
            key
            for key, meta in train_summary.get("models", {}).items()
            if key in learned and _is_stage01_primary_tdoa_candidate_meta(meta)
        ]
    )
    iid_tdoa_keys = [key for key in tdoa_keys if key.startswith("iid_")]
    geom_tdoa_keys = [key for key in tdoa_keys if key.startswith("geom_")]
    best_iid_tdoa_key = (
        min(iid_tdoa_keys, key=lambda key: (learned[key]["p90_m"], learned[key]["median_m"], key))
        if iid_tdoa_keys
        else None
    )
    best_geom_tdoa_key = (
        min(geom_tdoa_keys, key=lambda key: (learned[key]["p90_m"], learned[key]["median_m"], key))
        if geom_tdoa_keys
        else None
    )
    thresholds = LOCALIZATION_THRESHOLDS["01"]
    if best_iid_tdoa_key is not None and best_geom_tdoa_key is not None:
        best_tdoa_stats = {
            "iid_test": learned[best_iid_tdoa_key],
            "geom_test": learned[best_geom_tdoa_key],
        }
        gate = _stage01_gate(analytic_bundle["analytic_gcc_phat"], best_tdoa_stats, thresholds)
        passed = bool(gate["iid_passed"] and gate["geom_passed"] and gate["iid_within_2x_analytic_p90"] and gate["geom_within_2x_analytic_p90"])
        conclusion = (
            "stable_geometry + analytic_passed + network_passed"
            if passed
            else "stable_geometry + analytic_passed + network_failed"
        )
        plot_key_pairs = [("best_iid_tdoa", best_iid_tdoa_key), ("best_geom_tdoa", best_geom_tdoa_key)]
    else:
        best_tdoa_stats = {}
        gate = {
            "iid_passed": False,
            "geom_passed": False,
            "iid_within_2x_analytic_p90": False,
            "geom_within_2x_analytic_p90": False,
        }
        passed = False
        conclusion = "stable_geometry + analytic_passed + no_tdoa_candidate_for_gate"
        plot_key_pairs = []
    for label, key in plot_key_pairs:
        pred_plot, true_plot, ref_plot = plot_cache[key]
        err = localization_error_vector(pred_plot, true_plot)
        success_idx = int(np.argmin(err))
        failure_idx = int(np.argmax(err))
        for suffix, sample_index in [("success", success_idx), ("failure", failure_idx)]:
            _plot_stage01_sample(
                plots_dir / f"{label}_{suffix}.png",
                room_bounds=(float(cfg.plane_room_size[0]), float(cfg.plane_room_size[1])),
                ref_positions=ref_plot[sample_index],
                true_position=true_plot[sample_index],
                pred_position=pred_plot[sample_index],
                title=f"{key} | {suffix} | error={float(err[sample_index]):.4f} m",
            )
    summary = {
        "h5_path": str(h5_path),
        "output_dir": str(out_dir),
        "geometry_filter_mode": str(cfg.geometry_filter_mode),
        "min_triangle_area": float(cfg.min_triangle_area),
        "max_jacobian_condition": float(cfg.max_jacobian_condition),
        "window_preset": infer_localization_window_preset(cfg),
        "analytic_gcc_phat": analytic_bundle["analytic_gcc_phat"],
        "learned": learned,
        "best_iid_tdoa_key": best_iid_tdoa_key,
        "best_geom_tdoa_key": best_geom_tdoa_key,
        "gate_thresholds": thresholds,
        "tdoa_gate": gate,
        "network_feasibility": {
            "passed": passed,
            "conclusion": conclusion,
        },
    }
    save_json(out_dir / "summary.json", summary)
    return summary


def load_stage01_candidate_prediction_bundle(
    h5_path: Path | str,
    output_dir: Path | str,
    model_key: str | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    bundle = build_canonical_stage01_bundle(h5_path)
    out_dir = Path(output_dir).resolve()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    train_summary = json.loads((out_dir / "train_summary.json").read_text(encoding="utf-8"))
    if model_key is None:
        learned = summary.get("learned", {})
        if not learned:
            raise RuntimeError("No learned candidate metrics were found in this result directory.")
        model_key = min(learned.keys(), key=lambda key: (learned[key]["p90_m"], learned[key]["median_m"], key))
    meta = train_summary["models"][str(model_key)]
    spec = Stage01CandidateSpec(
        candidate_id=str(meta["candidate_id"]),
        feature_kind=str(meta["feature_kind"]),
        model_kind=str(meta["model_kind"]),
        target_kind=str(meta["target_kind"]),
        loss_kind=str(meta["loss_kind"]),
        hidden=int(meta.get("hidden", 256)),
        depth=int(meta.get("depth", 4)),
        dropout=float(meta.get("dropout", 0.10)),
        physics_weight=float(meta.get("physics_weight", PHYSICS_LOSS_WEIGHT)),
        solver_steps=int(meta.get("solver_steps", 5)),
    )
    split_kind = str(meta["split_kind"])
    idx = bundle["splits"]["test" if split_kind == "iid" else "geom_test"]
    feat = build_stage01_candidate_features(bundle, spec.feature_kind)
    layout = build_stage01_feature_layout(bundle, spec.feature_kind)
    target = build_stage01_candidate_targets(bundle, spec.target_kind)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    model, x_stats, _ = _load_candidate_from_ckpt(Path(meta["checkpoint_path"]), layout, target.shape[1], dev)
    pred_target = _predict_candidate(model, feat[idx], x_stats, dev)
    pred_global = invert_predictions_to_global(pred_target, bundle, spec, idx)
    true_global = np.asarray(bundle["data"]["source_position"][idx], dtype=np.float32)
    ref_global = np.asarray(bundle["data"]["ref_positions"][idx], dtype=np.float32)
    err = localization_error_vector(pred_global, true_global)
    return {
        "model_key": str(model_key),
        "split_kind": split_kind,
        "indices": np.asarray(idx, dtype=np.int64),
        "pred_global": pred_global,
        "true_global": true_global,
        "ref_global": ref_global,
        "error_m": err.astype(np.float32),
    }

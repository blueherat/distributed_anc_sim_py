from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn


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
from python_scripts.cfxlms_multi_control_dataset_impl import (
    AcousticScenarioSampler,
    DatasetBuildConfig,
    _cfxlms_with_init,
    _normalize_columns,
    _plant_residual_ratio,
    _rolling_mse_db_multichannel,
    _solve_w_canonical_from_q,
    compute_processed_features,
)
from python_scripts.single_control_canonical_q_common import (
    PathEncoder1d,
    anchor_metrics,
    build_laguerre_basis,
    compute_reference_covariance_batch,
    extract_local_windows_1d,
    local_energy_capture,
    project_tail_coeffs,
    reconstruct_q_np,
    reconstruct_q_torch,
    set_seed,
    split_indices,
    standardize,
    stats_general,
)


def resolve_h5_path(explicit_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    env_path = os.environ.get("ANC_H5_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path("python_impl") / "python_scripts" / "cfxlms_qc_dataset_multicontrol.h5")
    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
        checked.append(str(resolved))
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Multi-control dataset HDF5 was not found. Checked: {checked}")


@dataclass
class MultiCanonicalDatasetBundle:
    h5_path: Path
    cfg: DatasetBuildConfig
    meta: dict[str, Any]
    p_ref_paths: np.ndarray
    d_paths: np.ndarray
    s_matrix_paths: np.ndarray
    r2r_paths: np.ndarray
    s2r_paths: np.ndarray
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


def ensure_canonical_processed(h5_path: Path) -> None:
    needs_process = False
    with h5py.File(str(h5_path), "r") as h5:
        raw = h5["raw"]
        processed = h5.get("processed")
        required_raw = {"P_ref_paths", "D_paths", "S_matrix_paths", "R2R_paths", "S2R_paths"}
        required_proc = {"q_target", "w_canon"}
        needs_process = any(name not in raw for name in required_raw) or processed is None or any(name not in processed for name in required_proc)
    if needs_process:
        print(f"[Canonical-Q 3x3] rebuilding processed features for {h5_path}")
        compute_processed_features(h5_path)


def load_canonical_q_dataset(h5_path: str | Path, laguerre_pole: float = 0.55) -> MultiCanonicalDatasetBundle:
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
        flat_q = q_target.reshape(-1, q_keep_len)
        flat_anchor = np.argmax(np.abs(flat_q), axis=1).astype(np.int64)
        flat_local = extract_local_windows_1d(flat_q, flat_anchor, half_width=local_half_width)
        flat_tail_target, flat_tail_coeffs = project_tail_coeffs(flat_q, flat_anchor, basis=basis, half_width=local_half_width)
        n_rooms = int(q_target.shape[0])
        n_sec = int(q_target.shape[1])
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
            "q_num_speakers": int(processed.attrs["q_num_speakers"]),
            "q_target_dim": int(processed.attrs["q_target_dim"]),
            "local_half_width": local_half_width,
            "local_len": int(2 * local_half_width + 1),
            "tail_basis_dim": tail_basis_dim,
            "laguerre_pole": float(laguerre_pole),
            "lambda_q_scale": float(processed.attrs["lambda_q_scale"]),
            "lambda_w": float(processed.attrs["lambda_w"]),
            "q_target_source": str(processed.attrs.get("q_target_source", "regularized_secondary_inverse")),
            "r2r_pair_order_json": str(processed.attrs["r2r_pair_order_json"]),
        }
        return MultiCanonicalDatasetBundle(
            h5_path=path,
            cfg=cfg,
            meta=meta,
            p_ref_paths=np.asarray(raw["P_ref_paths"], dtype=np.float32),
            d_paths=np.asarray(raw["D_paths"], dtype=np.float32),
            s_matrix_paths=np.asarray(raw["S_matrix_paths"], dtype=np.float32),
            r2r_paths=np.asarray(raw["R2R_paths"], dtype=np.float32),
            s2r_paths=np.asarray(raw["S2R_paths"], dtype=np.float32),
            x_ref=np.asarray(raw["x_ref"], dtype=np.float32),
            q_target=q_target,
            w_canon=np.asarray(processed["w_canon"], dtype=np.float32),
            w_h5=np.asarray(raw["W_full"], dtype=np.float32),
            anchors=flat_anchor.reshape(n_rooms, n_sec),
            local_target=flat_local.reshape(n_rooms, n_sec, -1),
            tail_coeffs=flat_tail_coeffs.reshape(n_rooms, n_sec, -1),
            tail_target=flat_tail_target.reshape(n_rooms, n_sec, -1),
            basis=basis,
            room_data=room_data,
            source_seeds=np.asarray(raw["qc_metrics"]["source_seed"], dtype=np.int64),
        )


def fit_feature_stats(bundle: MultiCanonicalDatasetBundle, train_idx: np.ndarray, input_variant: str) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    stats["p_ref"] = stats_general(bundle.p_ref_paths, train_idx, reduce_axes=(0,))
    stats["d_paths"] = stats_general(bundle.d_paths, train_idx, reduce_axes=(0,))
    stats["s_matrix"] = stats_general(bundle.s_matrix_paths, train_idx, reduce_axes=(0,))
    if "xref" in str(input_variant):
        xref_cov = compute_reference_covariance_batch(bundle.x_ref, n_fft=int(bundle.cfg.psd_nfft))
        stats["xref_cov"] = stats_general(xref_cov, train_idx, reduce_axes=(0,))
    if "r2r" in str(input_variant):
        stats["r2r"] = stats_general(bundle.r2r_paths, train_idx, reduce_axes=(0,))
    return stats


def build_feature_cache(bundle: MultiCanonicalDatasetBundle, feature_stats: dict[str, Any], input_variant: str) -> dict[str, np.ndarray]:
    feats: dict[str, np.ndarray] = {}
    feats["p_ref"] = standardize(bundle.p_ref_paths, feature_stats["p_ref"]["mean"], feature_stats["p_ref"]["std"])
    feats["d_paths"] = standardize(bundle.d_paths, feature_stats["d_paths"]["mean"], feature_stats["d_paths"]["std"])
    feats["s_matrix"] = standardize(bundle.s_matrix_paths, feature_stats["s_matrix"]["mean"], feature_stats["s_matrix"]["std"])
    if "xref" in str(input_variant):
        xref_cov = compute_reference_covariance_batch(bundle.x_ref, n_fft=int(bundle.cfg.psd_nfft))
        feats["xref_cov"] = standardize(xref_cov, feature_stats["xref_cov"]["mean"], feature_stats["xref_cov"]["std"])
    if "r2r" in str(input_variant):
        feats["r2r"] = standardize(bundle.r2r_paths, feature_stats["r2r"]["mean"], feature_stats["r2r"]["std"])
    return feats


def fit_target_stats(bundle: MultiCanonicalDatasetBundle, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    q_scale = float(np.mean(bundle.q_target[train_idx] ** 2) + 1.0e-8)
    local_flat = bundle.local_target[train_idx].reshape(-1, bundle.local_target.shape[-1])
    tail_flat = bundle.tail_coeffs[train_idx].reshape(-1, bundle.tail_coeffs.shape[-1])
    local_mean = np.mean(local_flat, axis=0, dtype=np.float64).astype(np.float32)
    local_std = np.maximum(np.std(local_flat, axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0e-6))
    tail_mean = np.mean(tail_flat, axis=0, dtype=np.float64).astype(np.float32)
    tail_std = np.maximum(np.std(tail_flat, axis=0, dtype=np.float64).astype(np.float32), np.float32(1.0e-6))
    return {
        "q_scale": np.asarray(q_scale, dtype=np.float32),
        "local_mean": local_mean,
        "local_std": local_std,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
    }


class MultiControlCanonicalQNet(nn.Module):
    def __init__(
        self,
        q_keep_len: int,
        num_speakers: int,
        local_len: int,
        tail_basis_dim: int,
        dropout: float = 0.10,
        include_xref: bool = False,
        include_r2r: bool = False,
    ):
        super().__init__()
        self.include_xref = bool(include_xref)
        self.include_r2r = bool(include_r2r)
        self.num_speakers = int(num_speakers)
        self.p_ref_encoder = PathEncoder1d(in_channels=1, base_channels=24, out_dim=48)
        self.d_encoder = PathEncoder1d(in_channels=1, base_channels=24, out_dim=48)
        self.s_encoder = PathEncoder1d(in_channels=1, base_channels=24, out_dim=48)
        self.r2r_encoder = PathEncoder1d(in_channels=1, base_channels=16, out_dim=32) if self.include_r2r else None
        self.xref_encoder = PathEncoder1d(in_channels=9, base_channels=24, out_dim=64) if self.include_xref else None
        fusion_in = 48 * 3 + 48 * 3 + 48 * 9
        if self.include_r2r:
            fusion_in += 32 * 3
        if self.include_xref:
            fusion_in += 64
        self.global_fusion = nn.Sequential(
            nn.Linear(fusion_in, 384),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.speaker_embed = nn.Embedding(self.num_speakers, 16)
        self.speaker_head = nn.Sequential(
            nn.Linear(256 + 48 * 3 + 16, 192),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
        )
        self.anchor_head = nn.Linear(128, int(q_keep_len))
        self.local_head = nn.Linear(128, int(local_len))
        self.tail_head = nn.Linear(128, int(tail_basis_dim))
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
        d_paths: torch.Tensor,
        s_matrix: torch.Tensor,
        xref_cov: torch.Tensor | None = None,
        r2r: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = int(p_ref.shape[0])
        p_feat = self.p_ref_encoder(p_ref.reshape(batch * p_ref.shape[1], 1, -1)).reshape(batch, p_ref.shape[1], -1)
        d_feat = self.d_encoder(d_paths.reshape(batch * d_paths.shape[1], 1, -1)).reshape(batch, d_paths.shape[1], -1)
        s_feat = self.s_encoder(s_matrix.reshape(batch * s_matrix.shape[1] * s_matrix.shape[2], 1, -1)).reshape(batch, s_matrix.shape[1], s_matrix.shape[2], -1)

        global_parts = [p_feat.reshape(batch, -1), d_feat.reshape(batch, -1), s_feat.reshape(batch, -1)]
        if self.include_r2r:
            if r2r is None:
                raise ValueError("r2r input is required when include_r2r=True.")
            r2r_feat = self.r2r_encoder(r2r.reshape(batch * r2r.shape[1], 1, -1)).reshape(batch, r2r.shape[1], -1)
            global_parts.append(r2r_feat.reshape(batch, -1))
        if self.include_xref:
            if xref_cov is None:
                raise ValueError("xref_cov input is required when include_xref=True.")
            global_parts.append(self.xref_encoder(xref_cov))
        global_feat = self.global_fusion(torch.cat(global_parts, dim=1))

        row_feat = s_feat.reshape(batch, self.num_speakers, -1)
        speaker_ids = torch.arange(self.num_speakers, device=global_feat.device, dtype=torch.long)[None, :].expand(batch, -1)
        speaker_embed = self.speaker_embed(speaker_ids)
        speaker_in = torch.cat([global_feat[:, None, :].expand(-1, self.num_speakers, -1), row_feat, speaker_embed], dim=-1)
        hidden = self.speaker_head(speaker_in.reshape(batch * self.num_speakers, -1)).reshape(batch, self.num_speakers, -1)
        return {
            "anchor_logits": self.anchor_head(hidden.reshape(batch * self.num_speakers, -1)).reshape(batch, self.num_speakers, -1),
            "local_kernel": self.local_head(hidden.reshape(batch * self.num_speakers, -1)).reshape(batch, self.num_speakers, -1),
            "tail_coeffs": self.tail_head(hidden.reshape(batch * self.num_speakers, -1)).reshape(batch, self.num_speakers, -1),
        }


def build_model(bundle: MultiCanonicalDatasetBundle, input_variant: str, dropout: float) -> MultiControlCanonicalQNet:
    variant = str(input_variant)
    return MultiControlCanonicalQNet(
        q_keep_len=int(bundle.meta["q_keep_len"]),
        num_speakers=int(bundle.meta["q_num_speakers"]),
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
    d_paths: np.ndarray
    s_matrix_paths: np.ndarray


def room_sample_from_bundle(bundle: MultiCanonicalDatasetBundle, idx: int) -> dict[str, Any]:
    layout_value = bundle.room_data["layout_mode"][idx]
    if isinstance(layout_value, bytes):
        layout_value = layout_value.decode("utf-8")
    return {
        "room_size": np.asarray(bundle.room_data["room_size"][idx], dtype=float),
        "source_pos": np.asarray(bundle.room_data["source_position"][idx], dtype=float),
        "ref_positions": np.asarray(bundle.room_data["ref_positions"][idx], dtype=float),
        "sec_positions": np.asarray(bundle.room_data["sec_positions"][idx], dtype=float),
        "err_positions": np.asarray(bundle.room_data["err_positions"][idx], dtype=float),
        "sound_speed": float(bundle.room_data["sound_speed"][idx]),
        "absorption": float(bundle.room_data["material_absorption"][idx]),
        "image_order": int(bundle.room_data["image_source_order"][idx]),
        "layout_mode": str(layout_value),
    }


def build_replay_cases(bundle: MultiCanonicalDatasetBundle, room_indices: list[int]) -> list[ReplayCase]:
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
                d_paths=np.asarray(bundle.d_paths[int(idx)], dtype=np.float32),
                s_matrix_paths=np.asarray(bundle.s_matrix_paths[int(idx)], dtype=np.float32),
            )
        )
    return cases


def q_to_full(bundle: MultiCanonicalDatasetBundle, q_truncated: np.ndarray) -> np.ndarray:
    q = np.asarray(q_truncated, dtype=np.float32)
    q_full = np.zeros((q.shape[0], q.shape[1], int(bundle.meta["q_full_len"])), dtype=np.float32)
    keep = min(int(bundle.meta["q_keep_len"]), q.shape[-1])
    q_full[:, :, :keep] = q[:, :, :keep]
    return q_full


def q_to_w_canon_batch(bundle: MultiCanonicalDatasetBundle, q_truncated: np.ndarray, room_indices: list[int] | np.ndarray | None = None) -> np.ndarray:
    q_full = q_to_full(bundle, q_truncated)
    if room_indices is None:
        room_indices_arr = np.arange(q_full.shape[0], dtype=np.int64)
    else:
        room_indices_arr = np.asarray(room_indices, dtype=np.int64)
        if room_indices_arr.shape[0] != q_full.shape[0]:
            raise ValueError(f"room_indices length {room_indices_arr.shape[0]} does not match q batch {q_full.shape[0]}.")
    out = np.zeros((q_full.shape[0], bundle.cfg.num_secondary_speakers, bundle.cfg.num_reference_mics, bundle.cfg.filter_len), dtype=np.float32)
    for i in range(q_full.shape[0]):
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
    e_zero = np.asarray(cfxlms_fn(params)["err_hist"], dtype=float)
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
    )
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
    )
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
    )
    window_samples = min(max(32, int(round(float(early_window_s) * float(cfg.fs)))), max(int(len(case.time_axis) // 2), 32))
    t_db, db_zero = _rolling_mse_db_multichannel(e_zero, int(cfg.fs), window_samples=window_samples)
    _, db_ai = _rolling_mse_db_multichannel(e_ai, int(cfg.fs), window_samples=window_samples)
    _, db_exact = _rolling_mse_db_multichannel(e_exact, int(cfg.fs), window_samples=window_samples)
    _, db_h5 = _rolling_mse_db_multichannel(e_h5, int(cfg.fs), window_samples=window_samples)
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


def summarize_replay(bundle: MultiCanonicalDatasetBundle, room_indices: list[int], q_truncated: np.ndarray, early_window_s: float) -> dict[str, Any]:
    q_arr = np.asarray(q_truncated, dtype=np.float32)
    q_full = q_to_full(bundle, q_arr)
    w_ai_batch = q_to_w_canon_batch(bundle, q_arr, room_indices=room_indices)
    cases = build_replay_cases(bundle, room_indices)
    rows = []
    residuals = []
    for i, (case, w_ai) in enumerate(zip(cases, np.asarray(w_ai_batch, dtype=np.float32))):
        row = replay_metrics_for_case(case, w_ai=w_ai, cfg=bundle.cfg, early_window_s=float(early_window_s))
        row["room_idx"] = int(case.idx)
        row["plant_residual"] = float(_plant_residual_ratio(q_full[i], case.d_paths, case.s_matrix_paths))
        residuals.append(float(row["plant_residual"]))
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
        "plant_residual_mean": float(np.mean(residuals)),
        "per_room": rows,
    }


def exact_canonical_summary(bundle: MultiCanonicalDatasetBundle, room_indices: list[int], early_window_s: float) -> dict[str, Any]:
    return summarize_replay(bundle, room_indices=room_indices, q_truncated=np.asarray(bundle.q_target[room_indices], dtype=np.float32), early_window_s=float(early_window_s))


def build_probe_indices(train_idx: np.ndarray, val_idx: np.ndarray, probe_count: int, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(int(seed))
    train_sel = np.asarray(train_idx, dtype=np.int64).copy()
    val_sel = np.asarray(val_idx, dtype=np.int64).copy()
    rng.shuffle(train_sel)
    rng.shuffle(val_sel)
    return train_sel[: min(int(probe_count), train_sel.size)].tolist(), val_sel[: min(int(probe_count), val_sel.size)].tolist()

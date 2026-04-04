from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from py_anc.algorithms.hybrid_loss import HybridAcousticLoss

from py_anc.algorithms import cfxlms
from py_anc.utils import wn_gen
from python_scripts.cfxlms_single_control_dataset_impl import (
    AcousticScenarioSampler,
    DatasetBuildConfig,
    _cfxlms_with_init,
    _normalize_columns,
    _rolling_mse_db,
)
from python_scripts.train_hybrid_deep_fxlms_single_control import (
    HybridAncDataset,
    HybridDeepFxLMSNet,
    level_mask,
    load_bundle,
    resolve_h5_path,
    run_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-control Hybrid Deep-FxLMS checkpoint and acceptance gates.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warmstart-cases", type=int, default=8)
    parser.add_argument("--warmstart-level", type=int, default=3)
    parser.add_argument("--early-window-s", type=float, default=0.25)
    parser.add_argument("--target-metric", choices=("nr_last_db",), default="nr_last_db")
    parser.add_argument("--half-target-ratio", type=float, default=0.5)
    parser.add_argument("--disable-half-target-gate", action="store_true")
    return parser.parse_args()


@dataclass
class ReplayCase:
    idx: int
    manager: Any
    time_axis: np.ndarray
    reference_signal: np.ndarray
    desired_signal: np.ndarray


def build_replay_cases(h5_path: Path, room_indices: list[int]) -> list[ReplayCase]:
    with h5py.File(str(h5_path), "r") as h5:
        cfg = DatasetBuildConfig(**json.loads(h5.attrs["config_json"]))
        room = h5["raw/room_params"]
        source_seeds = np.asarray(h5["raw/qc_metrics/source_seed"], dtype=np.int64)

    sampler = AcousticScenarioSampler(cfg, np.random.default_rng(int(cfg.random_seed)))
    out: list[ReplayCase] = []
    for idx in room_indices:
        with h5py.File(str(h5_path), "r") as h5:
            room = h5["raw/room_params"]
            sampled = {
                "room_size": np.asarray(room["room_size"][idx], dtype=float),
                "source_pos": np.asarray(room["source_position"][idx], dtype=float),
                "ref_positions": np.asarray(room["ref_positions"][idx], dtype=float),
                "sec_positions": np.asarray(room["sec_positions"][idx], dtype=float),
                "err_positions": np.asarray(room["err_positions"][idx], dtype=float),
                "ref_azimuth_deg": np.asarray(room["ref_azimuth_deg"][idx], dtype=float),
                "ref_radii": np.asarray(room["ref_radii"][idx], dtype=float),
                "sec_source_distance": float(room["sec_source_distance"][idx]),
                "err_source_distance": float(room["err_source_distance"][idx]),
                "sec_err_distance": float(room["sec_err_distance"][idx]),
                "primary_advance_margin_min": float(room["primary_advance_margin_min"][idx]),
                "secondary_feedback_margin_min": float(room["secondary_feedback_margin_min"][idx]),
                "sound_speed": float(room["sound_speed"][idx]),
                "absorption": float(room["material_absorption"][idx]),
                "image_order": int(room["image_source_order"][idx]),
                "layout_mode": room["layout_mode"][idx].decode("utf-8") if isinstance(room["layout_mode"][idx], bytes) else str(room["layout_mode"][idx]),
            }

        mgr = sampler.build_manager(sampled)
        mgr.build(verbose=False)

        source_seed = int(source_seeds[int(idx)])
        noise, t = wn_gen(
            fs=int(cfg.fs),
            duration=float(cfg.noise_duration_s),
            f_low=float(cfg.f_low),
            f_high=float(cfg.f_high),
            rng=np.random.default_rng(source_seed),
        )
        source_signal = _normalize_columns(noise)
        time_axis = np.asarray(t[:, 0], dtype=float)
        reference_signal = _normalize_columns(mgr.calculate_reference_signal(source_signal, len(time_axis)))
        desired_signal = mgr.calculate_desired_signal(source_signal, len(time_axis))
        out.append(
            ReplayCase(
                idx=int(idx),
                manager=mgr,
                time_axis=time_axis,
                reference_signal=reference_signal,
                desired_signal=desired_signal,
            )
        )
    return out


def predict_w_batch(
    model: HybridDeepFxLMSNet,
    bundle,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds = HybridAncDataset(bundle=bundle, indices=np.asarray(indices, dtype=np.int64))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    pred_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for gcc, acoustic, p_ref, d_path, s_path, _, _, sample_idx in loader:
            out = model(
                gcc=gcc.to(device=device, dtype=torch.float32),
                acoustic=acoustic.to(device=device, dtype=torch.float32),
                p_ref=p_ref.to(device=device, dtype=torch.float32),
                d_path=d_path.to(device=device, dtype=torch.float32),
                s_path=s_path.to(device=device, dtype=torch.float32),
                sample_idx=sample_idx.to(device=device, dtype=torch.long),
            )
            pred_list.append(out["w_pred"].detach().cpu().numpy())
    return np.concatenate(pred_list, axis=0)


def load_target_metric_values(
    h5_path: Path,
    room_indices: list[int],
    metric_key: str,
) -> np.ndarray:
    idx = np.asarray(room_indices, dtype=np.int64)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.float64)

    with h5py.File(str(h5_path), "r") as h5:
        qc = h5["raw/qc_metrics"]
        if str(metric_key) not in qc:
            raise KeyError(f"Missing target metric in raw/qc_metrics: {metric_key}")
        values = np.asarray(qc[str(metric_key)], dtype=np.float64)

    if np.any(idx < 0) or np.any(idx >= values.shape[0]):
        raise IndexError("room index out of range when loading target metric values")
    return values[idx]


def convergence_step(db_curve: np.ndarray) -> int:
    arr = np.asarray(db_curve, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0
    tail = max(4, int(arr.size * 0.1))
    steady = float(np.mean(arr[-tail:]))
    threshold = steady + 1.0
    hit = np.where(arr <= threshold)[0]
    if hit.size == 0:
        return int(arr.size)
    return int(hit[0])


def warmstart_metrics(
    h5_path: Path,
    room_indices: list[int],
    w_pred: np.ndarray,
    early_window_s: float,
    target_nr_db: np.ndarray | None,
    half_target_ratio: float,
    target_metric: str,
) -> dict[str, float]:
    with h5py.File(str(h5_path), "r") as h5:
        cfg = DatasetBuildConfig(**json.loads(h5.attrs["config_json"]))

    replay_cases = build_replay_cases(h5_path, room_indices)
    gains: list[float] = []
    step_ratios: list[float] = []
    init_nr_vals: list[float] = []
    target_vals: list[float] = []
    sample_pass_vals: list[float] = []

    for case_idx, (case, w_i) in enumerate(zip(replay_cases, np.asarray(w_pred, dtype=np.float32))):
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
        e_zero = np.asarray(cfxlms(params)["err_hist"], dtype=float)[:, 0]
        e_warm = np.asarray(
            _cfxlms_with_init(
                case.time_axis,
                case.manager,
                int(cfg.filter_len),
                float(cfg.mu_candidates[0]),
                case.reference_signal,
                case.desired_signal,
                w_init=w_i[None, :, :],
                normalized_update=bool(cfg.anc_normalized_update),
                norm_epsilon=float(cfg.anc_norm_epsilon),
            )["err_hist"],
            dtype=float,
        )[:, 0]

        window_samples = min(max(32, int(round(float(early_window_s) * float(cfg.fs)))), max(int(len(case.time_axis) // 2), 32))
        t_db, db_zero = _rolling_mse_db(e_zero, int(cfg.fs), window_samples=window_samples)
        _, db_warm = _rolling_mse_db(e_warm, int(cfg.fs), window_samples=window_samples)

        early_mask = t_db <= float(early_window_s)
        if not np.any(early_mask):
            early_mask = np.ones_like(t_db, dtype=bool)
        gains.append(float(np.mean(db_zero[early_mask] - db_warm[early_mask])))

        desired = np.asarray(case.desired_signal, dtype=float).reshape(-1)
        error = np.asarray(e_warm, dtype=float).reshape(-1)
        n_sig = int(min(desired.size, error.size))
        if n_sig > 0:
            early_samples = min(max(8, int(round(float(early_window_s) * float(cfg.fs)))), n_sig)
            d_seg = desired[:early_samples]
            e_seg = error[:early_samples]
            d_pow = float(np.mean(d_seg**2)) + np.finfo(float).eps
            e_pow = float(np.mean(e_seg**2)) + np.finfo(float).eps
            init_nr_db = float(10.0 * np.log10(d_pow / e_pow))
            init_nr_vals.append(init_nr_db)

            if target_nr_db is not None and case_idx < int(len(target_nr_db)):
                target_db = float(target_nr_db[case_idx])
                if np.isfinite(target_db):
                    target_vals.append(target_db)
                    sample_pass_vals.append(float(init_nr_db >= float(half_target_ratio) * target_db))

        step_zero = convergence_step(db_zero)
        step_warm = max(1, convergence_step(db_warm))
        step_ratios.append(float(step_zero / step_warm))

    init_nr_mean = float(np.mean(init_nr_vals)) if init_nr_vals else float("nan")
    target_nr_mean = float(np.mean(target_vals)) if target_vals else float("nan")
    half_target_threshold = (
        float(half_target_ratio) * target_nr_mean if np.isfinite(target_nr_mean) else float("nan")
    )
    half_target_gap = (
        init_nr_mean - half_target_threshold
        if np.isfinite(init_nr_mean) and np.isfinite(half_target_threshold)
        else float("nan")
    )
    half_target_pass = bool(np.isfinite(half_target_gap) and half_target_gap >= 0.0)

    return {
        "early_gain_db_mean": float(np.mean(gains)) if gains else 0.0,
        "convergence_step_ratio_mean": float(np.mean(step_ratios)) if step_ratios else 0.0,
        "init_nr_db_mean": init_nr_mean,
        "target_metric": str(target_metric),
        "target_nr_db_mean": target_nr_mean,
        "half_target_ratio": float(half_target_ratio),
        "half_target_threshold_db": half_target_threshold,
        "half_target_gap_db": half_target_gap,
        "half_target_pass": half_target_pass,
        "sample_pass_rate": float(np.mean(sample_pass_vals)) if sample_pass_vals else float("nan"),
        "num_samples": int(len(init_nr_vals)),
    }


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint_path)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    train_args = checkpoint["args"]

    h5_path = resolve_h5_path(args.h5_path if args.h5_path else checkpoint.get("h5_path"))
    bundle = load_bundle(
        h5_path=h5_path,
        encoding=str(train_args["feature_encoding"]),
        disable_feature_b=bool(train_args["disable_feature_b"]),
    )

    device = torch.device(
        "cuda" if (str(args.device) == "auto" and torch.cuda.is_available()) else ("cpu" if str(args.device) == "auto" else str(args.device))
    )

    acoustic_in_channels = 1 if bundle.acoustic is None else int(bundle.acoustic.shape[1])
    model = HybridDeepFxLMSNet(
        acoustic_in_channels=acoustic_in_channels,
        filter_len=int(bundle.p_ref.shape[-1]),
        num_refs=int(bundle.p_ref.shape[1]),
        basis_dim=int(train_args["basis_dim"]),
        embed_dim=int(train_args["embed_dim"]),
        fusion_mode=str(train_args["fusion_mode"]),
        num_heads=int(train_args["num_heads"]),
        disable_feature_b=bool(train_args["disable_feature_b"]),
        use_path_features=bool(train_args.get("use_path_features", False)),
        use_index_embedding=bool(train_args.get("use_index_embedding", False)),
        index_direct_lookup=bool(train_args.get("index_direct_lookup", False)),
        num_samples=int(bundle.gcc.shape[0]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_module = HybridAcousticLoss(
        lambda_reg=float(train_args.get("lambda_reg", 1.0e-3)),
        conv_domain=str(train_args.get("loss_domain", "freq")),
        nr_margin_mode=str(train_args.get("nr_margin_mode", "power")),
        nr_margin_focus_ratio=float(train_args.get("nr_margin_focus_ratio", 1.0)),
    ).to(device)

    level_results: dict[str, dict[str, float]] = {}
    fail_reasons: list[str] = []
    all_level_nrs: list[float] = []

    for level in (1, 2, 3):
        idx = np.where(level_mask(bundle.image_order, level))[0].astype(np.int64)
        if idx.size < 4:
            level_results[f"level_{level}"] = {"nr_db": float("nan"), "loss_total": float("nan")}
            fail_reasons.append(f"level_{level}:not_enough_samples")
            continue

        ds = HybridAncDataset(bundle=bundle, indices=idx)
        loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False)
        metrics = run_epoch(
            loader=loader,
            model=model,
            device=device,
            loss_module=loss_module,
            optimizer=None,
            margin_weight=0.0,
            wopt_supervision_weight=0.0,
            acoustic_loss_weight=1.0,
        )
        level_results[f"level_{level}"] = {
            "nr_db": float(metrics["nr_db"]),
            "loss_total": float(metrics["loss_total"]),
            "loss_acoustic": float(metrics["loss_acoustic"]),
            "loss_reg": float(metrics["loss_reg"]),
            "num_samples": int(idx.size),
        }

        if np.isfinite(metrics["nr_db"]):
            all_level_nrs.append(float(metrics["nr_db"]))
        else:
            fail_reasons.append(f"level_{level}:nr_non_finite")

    level1_nr = float(level_results.get("level_1", {}).get("nr_db", float("nan")))
    if not np.isfinite(level1_nr) or level1_nr < 15.0:
        fail_reasons.append("level_1_nr_below_15db")

    level2_loss = float(level_results.get("level_2", {}).get("loss_total", float("nan")))
    if not np.isfinite(level2_loss):
        fail_reasons.append("level_2_loss_nan")

    mean_nr = float(np.mean(all_level_nrs)) if all_level_nrs else float("nan")
    if not np.isfinite(mean_nr) or mean_nr <= 10.0:
        fail_reasons.append("mean_nr_not_above_10db")

    warm_level = int(args.warmstart_level)
    warm_idx = np.where(level_mask(bundle.image_order, warm_level))[0].astype(np.int64)
    if warm_idx.size > 0 and int(args.warmstart_cases) > 0:
        probe = warm_idx[: min(int(args.warmstart_cases), warm_idx.size)]
        probe_list = [int(v) for v in probe.tolist()]
        w_pred = predict_w_batch(
            model=model,
            bundle=bundle,
            indices=probe,
            batch_size=min(int(args.batch_size), int(probe.size)),
            device=device,
        )
        target_vals: np.ndarray | None
        try:
            target_vals = load_target_metric_values(
                h5_path=h5_path,
                room_indices=probe_list,
                metric_key=str(args.target_metric),
            )
        except Exception:
            target_vals = None
            fail_reasons.append("half_target_target_metric_missing")

        warm_metrics = warmstart_metrics(
            h5_path=h5_path,
            room_indices=probe_list,
            w_pred=w_pred,
            early_window_s=float(args.early_window_s),
            target_nr_db=target_vals,
            half_target_ratio=float(args.half_target_ratio),
            target_metric=str(args.target_metric),
        )
    else:
        warm_metrics = {
            "early_gain_db_mean": float("nan"),
            "convergence_step_ratio_mean": float("nan"),
            "init_nr_db_mean": float("nan"),
            "target_metric": str(args.target_metric),
            "target_nr_db_mean": float("nan"),
            "half_target_ratio": float(args.half_target_ratio),
            "half_target_threshold_db": float("nan"),
            "half_target_gap_db": float("nan"),
            "half_target_pass": False,
            "sample_pass_rate": float("nan"),
            "num_samples": 0,
        }

    ratio = float(warm_metrics.get("convergence_step_ratio_mean", float("nan")))
    if np.isfinite(ratio) and ratio < 10.0:
        fail_reasons.append("warmstart_convergence_ratio_below_10x")

    half_target_gate = {
        "enabled": not bool(args.disable_half_target_gate),
        "target_metric": str(args.target_metric),
        "ratio": float(args.half_target_ratio),
        "target_nr_db_mean": float(warm_metrics.get("target_nr_db_mean", float("nan"))),
        "init_nr_db_mean": float(warm_metrics.get("init_nr_db_mean", float("nan"))),
        "threshold_db": float(warm_metrics.get("half_target_threshold_db", float("nan"))),
        "gap_db": float(warm_metrics.get("half_target_gap_db", float("nan"))),
        "sample_pass_rate": float(warm_metrics.get("sample_pass_rate", float("nan"))),
        "num_samples": int(warm_metrics.get("num_samples", 0)),
        "pass": bool(warm_metrics.get("half_target_pass", False)),
    }

    if bool(half_target_gate["enabled"]):
        target_mean = float(half_target_gate["target_nr_db_mean"])
        init_mean = float(half_target_gate["init_nr_db_mean"])
        if not (np.isfinite(target_mean) and np.isfinite(init_mean)):
            fail_reasons.append("half_target_metric_non_finite")
        elif not bool(half_target_gate["pass"]):
            fail_reasons.append("init_nr_below_half_target_mean")

    gate_status = "passed" if not fail_reasons else "failed"

    summary = {
        "checkpoint": str(ckpt_path),
        "h5_path": str(h5_path),
        "seed": int(train_args.get("seed", -1)),
        "ablation_tag": str(train_args.get("ablation_tag", "")),
        "feature_encoding": str(train_args.get("feature_encoding", "ri")),
        "fusion_mode": str(train_args.get("fusion_mode", "cross")),
        "disable_feature_b": bool(train_args.get("disable_feature_b", False)),
        "use_path_features": bool(train_args.get("use_path_features", False)),
        "use_index_embedding": bool(train_args.get("use_index_embedding", False)),
        "index_direct_lookup": bool(train_args.get("index_direct_lookup", False)),
        "lambda_reg": float(train_args.get("lambda_reg", 1.0e-3)),
        "basis_dim": int(train_args.get("basis_dim", 32)),
        "loss_domain": str(train_args.get("loss_domain", "freq")),
        "gate_status": gate_status,
        "fail_reasons": fail_reasons,
        "level_results": level_results,
        "mean_nr_db": mean_nr,
        "warmstart_metrics": warm_metrics,
        "half_target_gate": half_target_gate,
    }

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "eval_hybrid_deep_fxlms"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

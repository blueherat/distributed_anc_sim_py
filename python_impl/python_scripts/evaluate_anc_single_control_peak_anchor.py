from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.single_control_anchor_anc_common import (
    anchor_metrics,
    build_model,
    build_replay_cases,
    load_anchor_dataset,
    local_energy_capture,
    predict_anchor_prior_ridge,
    reconstruct_w_np,
    replay_metrics_for_case,
    resolve_h5_path,
    standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-control ANC peak-anchor checkpoints.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot-room-count", type=int, default=3)
    parser.add_argument("--replay-rooms", type=str, default=None, help="Examples: first:4 or 1,3,5")
    parser.add_argument("--replay-early-window-s", type=float, default=0.25)
    return parser.parse_args()


def parse_indices(spec: str, default_indices: list[int], max_n: int) -> list[int]:
    if spec is None:
        return default_indices
    s = str(spec).strip()
    if not s or s.lower() == "none":
        return default_indices
    if s.startswith("first:"):
        k = int(s.split(":", 1)[1])
        return list(range(min(k, max_n)))
    out: list[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return [v for v in out if 0 <= v < max_n]


def build_feature_cache(bundle, feature_stats: dict[str, dict[str, np.ndarray]], model_kind: str, include_s2r: bool) -> dict[str, np.ndarray]:
    feats: dict[str, np.ndarray] = {}
    if str(model_kind) == "acoustic":
        acoustic_ref_raw = bundle.acoustic_ref if not include_s2r else np.concatenate([bundle.acoustic_ref, bundle.s2r_ref], axis=2).astype(np.float32)
        feats["acoustic_shared"] = standardize(bundle.acoustic_shared, feature_stats["acoustic_shared"]["mean"], feature_stats["acoustic_shared"]["std"])
        feats["acoustic_ref"] = standardize(acoustic_ref_raw, feature_stats["acoustic_ref"]["mean"][None, :], feature_stats["acoustic_ref"]["std"][None, :])
        feats["anchor_prior"] = predict_anchor_prior_ridge(
            bundle.delay_shared,
            bundle.delay_ref,
            prior_fit=feature_stats["anchor_prior_fit"],
            tap_len=int(bundle.meta["tap_len"]),
        ).astype(np.int64)
    else:
        feats["oracle_shared"] = standardize(bundle.oracle_shared, feature_stats["oracle_shared"]["mean"], feature_stats["oracle_shared"]["std"])
        feats["oracle_ref"] = standardize(bundle.oracle_ref, feature_stats["oracle_ref"]["mean"][None, :], feature_stats["oracle_ref"]["std"][None, :])
    return feats


def predict_truncated_w(
    model_kind: str,
    model: torch.nn.Module,
    bundle,
    feats: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    include_s2r: bool,
    target_stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        if str(model_kind) == "acoustic":
            outputs = model(
                torch.from_numpy(feats["acoustic_shared"][idx]).to(device=device, dtype=torch.float32),
                torch.from_numpy(feats["acoustic_ref"][idx]).to(device=device, dtype=torch.float32),
                torch.from_numpy(feats["anchor_prior"][idx]).to(device=device, dtype=torch.long),
            )
        else:
            outputs = model(
                torch.from_numpy(feats["oracle_shared"][idx]).to(device=device, dtype=torch.float32),
                torch.from_numpy(feats["oracle_ref"][idx]).to(device=device, dtype=torch.float32),
            )
    if str(model_kind) == "oracle":
        pred_anchor = bundle.anchors[idx].astype(np.int64)
    else:
        pred_anchor = torch.argmax(outputs["anchor_logits"], dim=-1).cpu().numpy()
    local_pred = outputs["local_kernel"].cpu().numpy() * np.asarray(target_stats["local_std"], dtype=np.float32)[None, None, :] + np.asarray(target_stats["local_mean"], dtype=np.float32)[None, None, :]
    tail_pred = outputs["tail_coeffs"].cpu().numpy() * np.asarray(target_stats["tail_std"], dtype=np.float32)[None, None, :] + np.asarray(target_stats["tail_mean"], dtype=np.float32)[None, None, :]
    pred_w = reconstruct_w_np(
        anchor_idx=pred_anchor,
        local_kernel=local_pred,
        tail_coeffs=tail_pred,
        basis=bundle.basis,
        tap_len=int(bundle.meta["tap_len"]),
        half_width=int(bundle.meta["local_half_width"]),
    )
    return pred_anchor, pred_w


def truncated_to_full_w(pred_w: np.ndarray, filter_len: int) -> np.ndarray:
    arr = np.asarray(pred_w, dtype=np.float32)
    out = np.zeros((1, arr.shape[0], int(filter_len)), dtype=np.float32)
    out[0, :, : arr.shape[-1]] = arr
    return out


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint_path)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    h5_path = resolve_h5_path(args.h5_path)
    bundle = load_anchor_dataset(h5_path, tail_basis_dim=int(checkpoint["bundle_meta"]["tail_basis_dim"]), local_half_width=int(checkpoint["bundle_meta"]["local_half_width"]))
    model_kind = str(checkpoint["model_kind"])
    include_s2r = bool(checkpoint["include_s2r"])
    feature_stats = checkpoint["feature_stats"]
    target_stats = checkpoint["target_stats"]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    feats = build_feature_cache(bundle, feature_stats=feature_stats, model_kind=model_kind, include_s2r=include_s2r)
    model = build_model(
        model_kind,
        bundle,
        include_s2r=include_s2r,
        dropout=float(checkpoint["args"]["dropout"]),
        acoustic_shared_dim=(int(feats["acoustic_shared"].shape[-1]) if model_kind == "acoustic" else None),
        acoustic_ref_dim=(int(feats["acoustic_ref"].shape[-1]) if model_kind == "acoustic" else None),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    train_idx = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx = np.asarray(checkpoint["val_indices"], dtype=np.int64)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "eval_peak_anchor"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict[str, float]] = {}
    for split_name, indices in [("train", train_idx), ("val", val_idx)]:
        pred_anchor, pred_w = predict_truncated_w(model_kind, model, bundle, feats, indices, device, include_s2r, target_stats)
        true_anchor = bundle.anchors[indices]
        true_w = bundle.w_target[indices]
        split_metrics = anchor_metrics(pred_anchor, true_anchor)
        split_metrics["local_energy_capture"] = local_energy_capture(true_w, pred_anchor, half_width=int(bundle.meta["local_half_width"]))
        split_metrics["full_raw_mse"] = float(np.mean((pred_w - true_w) ** 2))
        metrics[split_name] = split_metrics

    default_replay = [int(v) for v in checkpoint.get("replay_room_indices", [])]
    replay_indices = parse_indices(args.replay_rooms, default_indices=default_replay, max_n=int(bundle.w_target.shape[0]))
    replay_cases = build_replay_cases(bundle, replay_indices)
    _, pred_w_rooms = predict_truncated_w(model_kind, model, bundle, feats, replay_indices, device, include_s2r, target_stats)

    replay_rows = []
    for case, pred_w in zip(replay_cases, pred_w_rooms):
        replay_row = replay_metrics_for_case(
            case=case,
            w_ai=truncated_to_full_w(pred_w, filter_len=int(bundle.cfg.filter_len)),
            cfg=bundle.cfg,
            early_window_s=float(args.replay_early_window_s),
        )
        replay_row["room_idx"] = int(case.idx)
        replay_rows.append(replay_row)

    replay_summary = {
        "room_indices": replay_indices,
        "ai_vs_zero_db_mean": float(np.mean([row["ai_vs_zero_db"] for row in replay_rows])) if replay_rows else None,
        "h5_vs_zero_db_mean": float(np.mean([row["h5_vs_zero_db"] for row in replay_rows])) if replay_rows else None,
        "ai_to_h5_gap_db_mean": float(np.mean([row["ai_to_h5_gap_db"] for row in replay_rows])) if replay_rows else None,
        "per_room": replay_rows,
    }

    for case, pred_w in zip(replay_cases[: int(args.plot_room_count)], pred_w_rooms[: int(args.plot_room_count)]):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(bundle.w_full[case.idx][0, 0, :80], label="HDF5 ref0", color="tab:orange")
        ax.plot(pred_w[0, :80], label="Pred ref0", color="tab:blue")
        ax.set_title(f"Room {case.idx}: ref0 first 80 taps")
        ax.set_xlabel("Tap")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(out_dir / f"room_{case.idx:04d}_ref0_taps.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "checkpoint": str(ckpt_path),
        "h5_path": str(h5_path),
        "model_kind": model_kind,
        "include_s2r": include_s2r,
        "train_metrics": metrics["train"],
        "val_metrics": metrics["val"],
        "replay_summary": replay_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

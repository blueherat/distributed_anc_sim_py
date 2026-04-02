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

from python_scripts.multi_control_canonical_q_common import (
    anchor_metrics,
    build_feature_cache,
    build_model,
    exact_canonical_summary,
    fit_feature_stats,
    load_canonical_q_dataset,
    local_energy_capture,
    reconstruct_q_np,
    resolve_h5_path,
    summarize_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the canonical-Q multi-control ANC model.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot-room-count", type=int, default=3)
    return parser.parse_args()


def reconstruct_q_multi_np(anchor: np.ndarray, local: np.ndarray, tail: np.ndarray, basis: np.ndarray, tap_len: int, half_width: int) -> np.ndarray:
    flat = reconstruct_q_np(anchor.reshape(-1), local.reshape(-1, local.shape[-1]), tail.reshape(-1, tail.shape[-1]), basis=np.asarray(basis, dtype=np.float32), tap_len=int(tap_len), half_width=int(half_width))
    return flat.reshape(anchor.shape[0], anchor.shape[1], -1)


def predict_q(model: torch.nn.Module, variant: str, feats: dict[str, np.ndarray], idx: np.ndarray, device: torch.device, basis: np.ndarray, meta: dict[str, object], target_stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        outputs = model(
            p_ref=torch.from_numpy(feats["p_ref"][idx]).to(device=device, dtype=torch.float32),
            d_paths=torch.from_numpy(feats["d_paths"][idx]).to(device=device, dtype=torch.float32),
            s_matrix=torch.from_numpy(feats["s_matrix"][idx]).to(device=device, dtype=torch.float32),
            xref_cov=(torch.from_numpy(feats["xref_cov"][idx]).to(device=device, dtype=torch.float32) if "xref" in variant else None),
            r2r=(torch.from_numpy(feats["r2r"][idx]).to(device=device, dtype=torch.float32) if "r2r" in variant else None),
        )
    anchor = torch.argmax(outputs["anchor_logits"], dim=-1).cpu().numpy()
    local = outputs["local_kernel"].cpu().numpy() * np.asarray(target_stats["local_std"], dtype=np.float32)[None, None, :] + np.asarray(target_stats["local_mean"], dtype=np.float32)[None, None, :]
    tail = outputs["tail_coeffs"].cpu().numpy() * np.asarray(target_stats["tail_std"], dtype=np.float32)[None, None, :] + np.asarray(target_stats["tail_mean"], dtype=np.float32)[None, None, :]
    q_pred = reconstruct_q_multi_np(anchor, local, tail, basis=np.asarray(basis, dtype=np.float32), tap_len=int(meta["q_keep_len"]), half_width=int(meta["local_half_width"]))
    return anchor, q_pred


def main() -> int:
    args = parse_args()
    ckpt_path = Path(args.checkpoint_path)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    h5_path = resolve_h5_path(args.h5_path if args.h5_path else checkpoint.get("h5_path"))
    bundle = load_canonical_q_dataset(h5_path, laguerre_pole=float(checkpoint["canonical_meta"]["laguerre_pole"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = build_model(bundle, input_variant=str(checkpoint["input_variant"]), dropout=float(checkpoint["args"]["dropout"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    feats = build_feature_cache(bundle, feature_stats=checkpoint["feature_stats"], input_variant=str(checkpoint["input_variant"]))
    train_idx = np.asarray(checkpoint["train_indices"], dtype=np.int64)
    val_idx = np.asarray(checkpoint["val_indices"], dtype=np.int64)
    train_probe = [int(v) for v in checkpoint["train_probe_indices"]]
    val_probe = [int(v) for v in checkpoint["val_probe_indices"]]
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "eval_canonical_q"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict[str, float]] = {}
    for split_name, split_idx in [("train", train_idx), ("val", val_idx)]:
        pred_anchor, pred_q = predict_q(model, str(checkpoint["input_variant"]), feats, split_idx, device, bundle.basis, bundle.meta, checkpoint["target_stats"])
        true_anchor = bundle.anchors[split_idx].reshape(-1)
        pred_anchor_flat = pred_anchor.reshape(-1)
        true_q = bundle.q_target[split_idx].reshape(-1, bundle.q_target.shape[-1])
        pred_q_flat = pred_q.reshape(-1, pred_q.shape[-1])
        split_metrics = anchor_metrics(pred_anchor_flat, true_anchor)
        split_metrics["local_energy_capture"] = local_energy_capture(true_q, pred_anchor_flat, half_width=int(bundle.meta["local_half_width"]))
        split_metrics["q_raw_mse"] = float(np.mean((pred_q_flat - true_q) ** 2))
        metrics[split_name] = split_metrics

    _, train_probe_q = predict_q(model, str(checkpoint["input_variant"]), feats, np.asarray(train_probe, dtype=np.int64), device, bundle.basis, bundle.meta, checkpoint["target_stats"])
    _, val_probe_q = predict_q(model, str(checkpoint["input_variant"]), feats, np.asarray(val_probe, dtype=np.int64), device, bundle.basis, bundle.meta, checkpoint["target_stats"])
    train_replay = summarize_replay(bundle, train_probe, train_probe_q, early_window_s=float(checkpoint["args"]["replay_early_window_s"]))
    val_replay = summarize_replay(bundle, val_probe, val_probe_q, early_window_s=float(checkpoint["args"]["replay_early_window_s"]))
    exact_train = exact_canonical_summary(bundle, train_probe, early_window_s=float(checkpoint["args"]["replay_early_window_s"]))
    exact_val = exact_canonical_summary(bundle, val_probe, early_window_s=float(checkpoint["args"]["replay_early_window_s"]))

    plot_indices = val_probe[: int(args.plot_room_count)]
    pred_anchor_plot, pred_q_plot = predict_q(model, str(checkpoint["input_variant"]), feats, np.asarray(plot_indices, dtype=np.int64), device, bundle.basis, bundle.meta, checkpoint["target_stats"])
    for room_idx, anchor_arr, q_pred in zip(plot_indices, pred_anchor_plot, pred_q_plot):
        q_true = bundle.q_target[int(room_idx)]
        fig, axes = plt.subplots(bundle.cfg.num_secondary_speakers, 1, figsize=(10, 8), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for sec_idx, ax in enumerate(axes):
            ax.plot(q_true[sec_idx, :120], color="tab:orange", label="Q exact")
            ax.plot(q_pred[sec_idx, :120], color="tab:blue", label="Q pred")
            ax.axvline(int(anchor_arr[sec_idx]), color="tab:green", linestyle="--", alpha=0.6, label="pred anchor")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Room {room_idx}: speaker {sec_idx}")
            if sec_idx == 0:
                ax.legend()
        fig.savefig(out_dir / f"room_{int(room_idx):04d}_canonical_q.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "checkpoint": str(ckpt_path),
        "h5_path": str(h5_path),
        "input_variant": checkpoint["input_variant"],
        "train_metrics": metrics["train"],
        "val_metrics": metrics["val"],
        "train_probe_replay": train_replay,
        "val_probe_replay": val_replay,
        "exact_train_probe": exact_train,
        "exact_val_probe": exact_val,
        "best_replay_metrics": checkpoint["best_replay_metrics"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

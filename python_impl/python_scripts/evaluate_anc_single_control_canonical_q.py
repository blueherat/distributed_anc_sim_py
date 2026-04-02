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

from python_scripts.single_control_canonical_q_common import (
    anchor_metrics,
    build_feature_cache,
    build_model,
    exact_canonical_summary,
    fit_feature_stats,
    load_canonical_q_dataset,
    local_energy_capture,
    q_to_w_canon_batch,
    reconstruct_q_np,
    resolve_h5_path,
    summarize_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the canonical-Q single-control ANC model.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot-room-count", type=int, default=3)
    return parser.parse_args()


def predict_q(model: torch.nn.Module, variant: str, feats: dict[str, np.ndarray], idx: np.ndarray, device: torch.device, basis: np.ndarray, meta: dict[str, object], target_stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        outputs = model(
            p_ref=torch.from_numpy(feats["p_ref"][idx]).to(device=device, dtype=torch.float32),
            d_path=torch.from_numpy(feats["d_path"][idx]).to(device=device, dtype=torch.float32),
            s_path=torch.from_numpy(feats["s_path"][idx]).to(device=device, dtype=torch.float32),
            anchor_prior=torch.from_numpy(feats["anchor_prior"][idx].astype(np.int64)).to(device=device, dtype=torch.long),
            xref_cov=(torch.from_numpy(feats["xref_cov"][idx]).to(device=device, dtype=torch.float32) if "xref" in variant else None),
            r2r=(torch.from_numpy(feats["r2r"][idx]).to(device=device, dtype=torch.float32) if "r2r" in variant else None),
        )
    anchor = torch.argmax(outputs["anchor_logits"], dim=1).cpu().numpy()
    local = outputs["local_kernel"].cpu().numpy() * np.asarray(target_stats["local_std"], dtype=np.float32)[None, :] + np.asarray(target_stats["local_mean"], dtype=np.float32)[None, :]
    tail = outputs["tail_coeffs"].cpu().numpy() * np.asarray(target_stats["tail_std"], dtype=np.float32)[None, :] + np.asarray(target_stats["tail_mean"], dtype=np.float32)[None, :]
    q_pred = reconstruct_q_np(anchor, local, tail, basis=np.asarray(basis, dtype=np.float32), tap_len=int(meta["q_keep_len"]), half_width=int(meta["local_half_width"]))
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
    pred_cache: dict[str, np.ndarray] = {}
    for split_name, split_idx in [("train", train_idx), ("val", val_idx)]:
        pred_anchor, pred_q = predict_q(model, str(checkpoint["input_variant"]), feats, split_idx, device, bundle.basis, bundle.meta, checkpoint["target_stats"])
        true_anchor = bundle.anchors[split_idx]
        true_q = bundle.q_target[split_idx]
        split_metrics = anchor_metrics(pred_anchor, true_anchor)
        split_metrics["local_energy_capture"] = local_energy_capture(true_q, pred_anchor, half_width=int(bundle.meta["local_half_width"]))
        split_metrics["q_raw_mse"] = float(np.mean((pred_q - true_q) ** 2))
        metrics[split_name] = split_metrics
        pred_cache[split_name] = pred_q

    train_probe_anchor, train_probe_q = predict_q(model, str(checkpoint["input_variant"]), feats, np.asarray(train_probe, dtype=np.int64), device, bundle.basis, bundle.meta, checkpoint["target_stats"])
    val_probe_anchor, val_probe_q = predict_q(model, str(checkpoint["input_variant"]), feats, np.asarray(val_probe, dtype=np.int64), device, bundle.basis, bundle.meta, checkpoint["target_stats"])
    train_replay = summarize_replay(bundle, train_probe, q_to_w_canon_batch(bundle, train_probe_q, room_indices=train_probe), early_window_s=float(checkpoint["args"]["replay_early_window_s"]))
    val_replay = summarize_replay(bundle, val_probe, q_to_w_canon_batch(bundle, val_probe_q, room_indices=val_probe), early_window_s=float(checkpoint["args"]["replay_early_window_s"]))
    exact_train = exact_canonical_summary(bundle, train_probe, early_window_s=float(checkpoint["args"]["replay_early_window_s"]))
    exact_val = exact_canonical_summary(bundle, val_probe, early_window_s=float(checkpoint["args"]["replay_early_window_s"]))

    plot_indices = val_probe[: int(args.plot_room_count)]
    pred_anchor_plot, pred_q_plot = predict_q(model, str(checkpoint["input_variant"]), feats, np.asarray(plot_indices, dtype=np.int64), device, bundle.basis, bundle.meta, checkpoint["target_stats"])
    for room_idx, anchor_idx, q_pred in zip(plot_indices, pred_anchor_plot, pred_q_plot):
        q_true = bundle.q_target[int(room_idx)]
        w_pred = q_to_w_canon_batch(bundle, q_pred[None, :], room_indices=[int(room_idx)])[0]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        axes[0].plot(q_true[:120], color="tab:orange", label="Q exact")
        axes[0].plot(q_pred[:120], color="tab:blue", label="Q pred")
        axes[0].axvline(int(anchor_idx), color="tab:green", linestyle="--", alpha=0.6, label="pred anchor")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_title(f"Room {room_idx}: canonical Q")
        axes[1].plot(bundle.w_canon[int(room_idx), 0, 0, :120], color="tab:orange", label="W_canon ref0")
        axes[1].plot(w_pred[0, 0, :120], color="tab:blue", label="W_pred ref0")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_title(f"Room {room_idx}: canonical W ref0")
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

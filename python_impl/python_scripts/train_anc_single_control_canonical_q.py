from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.single_control_canonical_q_common import (
    anchor_metrics,
    build_feature_cache,
    build_model,
    build_probe_indices,
    build_replay_cases,
    exact_canonical_summary,
    fit_feature_stats,
    fit_target_stats,
    load_canonical_q_dataset,
    local_energy_capture,
    q_to_w_canon_batch,
    reconstruct_q_np,
    reconstruct_q_torch,
    resolve_h5_path,
    set_seed,
    split_indices,
    summarize_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the canonical-Q single-control ANC model.")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--input-variant", choices=("base", "plus_xref", "plus_r2r", "plus_both"), default="base")
    parser.add_argument("--anchor-loss-weight", type=float, default=1.0)
    parser.add_argument("--local-loss-weight", type=float, default=0.75)
    parser.add_argument("--tail-loss-weight", type=float, default=0.20)
    parser.add_argument("--q-loss-weight", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--probe-count", type=int, default=12)
    parser.add_argument("--replay-eval-interval", type=int, default=5)
    parser.add_argument("--replay-early-window-s", type=float, default=0.25)
    parser.add_argument("--laguerre-pole", type=float, default=0.55)
    return parser.parse_args()


def variant_name(raw: str) -> str:
    return {
        "base": "base",
        "plus_xref": "xref",
        "plus_r2r": "r2r",
        "plus_both": "xref_r2r",
    }[str(raw)]


def build_dataset(feats: dict[str, np.ndarray], bundle, idx: np.ndarray, variant: str) -> TensorDataset:
    tensors: list[torch.Tensor] = [
        torch.from_numpy(feats["p_ref"][idx]),
        torch.from_numpy(feats["d_path"][idx]),
        torch.from_numpy(feats["s_path"][idx]),
        torch.from_numpy(feats["anchor_prior"][idx].astype(np.int64)),
    ]
    if "xref" in variant:
        tensors.append(torch.from_numpy(feats["xref_cov"][idx]))
    if "r2r" in variant:
        tensors.append(torch.from_numpy(feats["r2r"][idx]))
    tensors.extend(
        [
            torch.from_numpy(bundle.anchors[idx].astype(np.int64)),
            torch.from_numpy(bundle.local_target[idx]),
            torch.from_numpy(bundle.tail_coeffs[idx]),
            torch.from_numpy(bundle.q_target[idx]),
        ]
    )
    return TensorDataset(*tensors)


def unpack_batch(batch: tuple[torch.Tensor, ...], variant: str, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pos = 0
    inputs = {
        "p_ref": batch[pos].to(device=device, dtype=torch.float32, non_blocking=True),
        "d_path": batch[pos + 1].to(device=device, dtype=torch.float32, non_blocking=True),
        "s_path": batch[pos + 2].to(device=device, dtype=torch.float32, non_blocking=True),
        "anchor_prior": batch[pos + 3].to(device=device, dtype=torch.long, non_blocking=True),
    }
    pos += 4
    if "xref" in variant:
        inputs["xref_cov"] = batch[pos].to(device=device, dtype=torch.float32, non_blocking=True)
        pos += 1
    else:
        inputs["xref_cov"] = None
    if "r2r" in variant:
        inputs["r2r"] = batch[pos].to(device=device, dtype=torch.float32, non_blocking=True)
        pos += 1
    else:
        inputs["r2r"] = None
    targets = {
        "anchor": batch[pos].to(device=device, dtype=torch.long, non_blocking=True),
        "local": batch[pos + 1].to(device=device, dtype=torch.float32, non_blocking=True),
        "tail": batch[pos + 2].to(device=device, dtype=torch.float32, non_blocking=True),
        "q": batch[pos + 3].to(device=device, dtype=torch.float32, non_blocking=True),
    }
    return inputs, targets


def evaluate_epoch(loader: DataLoader, model: torch.nn.Module, variant: str, device: torch.device, basis_t: torch.Tensor, meta: dict[str, Any], target_stats: dict[str, np.ndarray], loss_weights: dict[str, float], label_smoothing: float) -> dict[str, float]:
    model.eval()
    local_mean_t = torch.from_numpy(np.asarray(target_stats["local_mean"], dtype=np.float32)).to(device=device)[None, :]
    local_std_t = torch.from_numpy(np.asarray(target_stats["local_std"], dtype=np.float32)).to(device=device)[None, :]
    tail_mean_t = torch.from_numpy(np.asarray(target_stats["tail_mean"], dtype=np.float32)).to(device=device)[None, :]
    tail_std_t = torch.from_numpy(np.asarray(target_stats["tail_std"], dtype=np.float32)).to(device=device)[None, :]
    q_scale = float(target_stats["q_scale"])
    loss_sum = 0.0
    q_mse_sum = 0.0
    pred_anchor_all: list[np.ndarray] = []
    true_anchor_all: list[np.ndarray] = []
    pred_q_all: list[np.ndarray] = []
    true_q_all: list[np.ndarray] = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = unpack_batch(batch, variant, device)
            outputs = model(**inputs)
            local_pred = outputs["local_kernel"] * local_std_t + local_mean_t
            tail_pred = outputs["tail_coeffs"] * tail_std_t + tail_mean_t
            q_teacher = reconstruct_q_torch(targets["anchor"], local_pred, tail_pred, basis_t, tap_len=int(meta["q_keep_len"]), half_width=int(meta["local_half_width"]))
            local_true_norm = (targets["local"] - local_mean_t) / local_std_t
            tail_true_norm = (targets["tail"] - tail_mean_t) / tail_std_t
            anchor_loss = F.cross_entropy(outputs["anchor_logits"], targets["anchor"], label_smoothing=float(label_smoothing))
            local_loss = 0.5 * (F.l1_loss(outputs["local_kernel"], local_true_norm) + F.mse_loss(outputs["local_kernel"], local_true_norm))
            tail_loss = F.mse_loss(outputs["tail_coeffs"], tail_true_norm)
            q_loss = F.mse_loss(q_teacher, targets["q"]) / float(q_scale)
            total = (
                float(loss_weights["anchor"]) * anchor_loss
                + float(loss_weights["local"]) * local_loss
                + float(loss_weights["tail"]) * tail_loss
                + float(loss_weights["q"]) * q_loss
            )
            pred_anchor = torch.argmax(outputs["anchor_logits"], dim=1)
            pred_q = reconstruct_q_torch(pred_anchor, local_pred, tail_pred, basis_t, tap_len=int(meta["q_keep_len"]), half_width=int(meta["local_half_width"]))
            bs = int(targets["anchor"].shape[0])
            loss_sum += float(total.detach().cpu()) * bs
            q_mse_sum += float(F.mse_loss(pred_q, targets["q"]).detach().cpu()) * bs
            count += bs
            pred_anchor_all.append(pred_anchor.cpu().numpy())
            true_anchor_all.append(targets["anchor"].cpu().numpy())
            pred_q_all.append(pred_q.cpu().numpy())
            true_q_all.append(targets["q"].cpu().numpy())
    pred_anchor_np = np.concatenate(pred_anchor_all, axis=0)
    true_anchor_np = np.concatenate(true_anchor_all, axis=0)
    pred_q_np = np.concatenate(pred_q_all, axis=0)
    true_q_np = np.concatenate(true_q_all, axis=0)
    metrics = anchor_metrics(pred_anchor_np, true_anchor_np)
    metrics["local_energy_capture"] = local_energy_capture(true_q_np, pred_anchor_np, half_width=int(meta["local_half_width"]))
    metrics["loss"] = loss_sum / max(count, 1)
    metrics["q_raw_mse"] = q_mse_sum / max(count, 1)
    return metrics


def predict_q(model: torch.nn.Module, variant: str, feats: dict[str, np.ndarray], idx: list[int], device: torch.device, basis: np.ndarray, meta: dict[str, Any], target_stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    index = np.asarray(idx, dtype=np.int64)
    local_mean_t = torch.from_numpy(np.asarray(target_stats["local_mean"], dtype=np.float32)).to(device=device)[None, :]
    local_std_t = torch.from_numpy(np.asarray(target_stats["local_std"], dtype=np.float32)).to(device=device)[None, :]
    tail_mean_t = torch.from_numpy(np.asarray(target_stats["tail_mean"], dtype=np.float32)).to(device=device)[None, :]
    tail_std_t = torch.from_numpy(np.asarray(target_stats["tail_std"], dtype=np.float32)).to(device=device)[None, :]
    model.eval()
    with torch.no_grad():
        outputs = model(
            p_ref=torch.from_numpy(feats["p_ref"][index]).to(device=device, dtype=torch.float32),
            d_path=torch.from_numpy(feats["d_path"][index]).to(device=device, dtype=torch.float32),
            s_path=torch.from_numpy(feats["s_path"][index]).to(device=device, dtype=torch.float32),
            anchor_prior=torch.from_numpy(feats["anchor_prior"][index].astype(np.int64)).to(device=device, dtype=torch.long),
            xref_cov=(torch.from_numpy(feats["xref_cov"][index]).to(device=device, dtype=torch.float32) if "xref" in variant else None),
            r2r=(torch.from_numpy(feats["r2r"][index]).to(device=device, dtype=torch.float32) if "r2r" in variant else None),
        )
    anchor = torch.argmax(outputs["anchor_logits"], dim=1).cpu().numpy()
    local = outputs["local_kernel"].cpu().numpy() * np.asarray(target_stats["local_std"], dtype=np.float32)[None, :] + np.asarray(target_stats["local_mean"], dtype=np.float32)[None, :]
    tail = outputs["tail_coeffs"].cpu().numpy() * np.asarray(target_stats["tail_std"], dtype=np.float32)[None, :] + np.asarray(target_stats["tail_mean"], dtype=np.float32)[None, :]
    q_pred = reconstruct_q_np(anchor, local, tail, basis=np.asarray(basis, dtype=np.float32), tap_len=int(meta["q_keep_len"]), half_width=int(meta["local_half_width"]))
    return anchor, q_pred


def is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    cand_gain = float(candidate["ai_vs_zero_db_mean"])
    best_gain = float(best["ai_vs_zero_db_mean"])
    if cand_gain > best_gain + 1.0e-9:
        return True
    if np.isclose(cand_gain, best_gain):
        return float(candidate["ai_to_h5_gap_db_mean"]) < float(best["ai_to_h5_gap_db_mean"]) - 1.0e-9
    return False


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    variant = variant_name(args.input_variant)
    h5_path = resolve_h5_path(args.h5_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_canonical_q_dataset(h5_path, laguerre_pole=float(args.laguerre_pole))
    train_idx, val_idx = split_indices(bundle.q_target.shape[0], float(args.val_frac), int(args.seed))
    train_probe, val_probe = build_probe_indices(train_idx, val_idx, probe_count=int(args.probe_count), seed=int(args.seed))
    exact_train = exact_canonical_summary(bundle, room_indices=train_probe, early_window_s=float(args.replay_early_window_s))
    exact_val = exact_canonical_summary(bundle, room_indices=val_probe, early_window_s=float(args.replay_early_window_s))
    if float(exact_val.get("ai_vs_zero_db_mean", 0.0)) <= 0.0:
        raise RuntimeError("Exact canonical replay upper bound is not above zero-init on the validation probe.")

    feature_stats = fit_feature_stats(bundle, train_idx=train_idx, input_variant=variant)
    feats = build_feature_cache(bundle, feature_stats=feature_stats, input_variant=variant)
    target_stats = fit_target_stats(bundle, train_idx=train_idx)
    train_ds = build_dataset(feats, bundle, train_idx, variant)
    val_ds = build_dataset(feats, bundle, val_idx, variant)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = build_model(bundle, input_variant=variant, dropout=float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    basis_t = torch.from_numpy(bundle.basis).to(device=device, dtype=torch.float32)
    local_mean_t = torch.from_numpy(np.asarray(target_stats["local_mean"], dtype=np.float32)).to(device=device)[None, :]
    local_std_t = torch.from_numpy(np.asarray(target_stats["local_std"], dtype=np.float32)).to(device=device)[None, :]
    tail_mean_t = torch.from_numpy(np.asarray(target_stats["tail_mean"], dtype=np.float32)).to(device=device)[None, :]
    tail_std_t = torch.from_numpy(np.asarray(target_stats["tail_std"], dtype=np.float32)).to(device=device)[None, :]
    q_scale = float(target_stats["q_scale"])
    loss_weights = {
        "anchor": float(args.anchor_loss_weight),
        "local": float(args.local_loss_weight),
        "tail": float(args.tail_loss_weight),
        "q": float(args.q_loss_weight),
    }

    history_rows: list[dict[str, Any]] = []
    best_summary: dict[str, Any] | None = None
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            inputs, targets = unpack_batch(batch, variant, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs)
            local_pred = outputs["local_kernel"] * local_std_t + local_mean_t
            tail_pred = outputs["tail_coeffs"] * tail_std_t + tail_mean_t
            q_teacher = reconstruct_q_torch(targets["anchor"], local_pred, tail_pred, basis_t, tap_len=int(bundle.meta["q_keep_len"]), half_width=int(bundle.meta["local_half_width"]))
            local_true_norm = (targets["local"] - local_mean_t) / local_std_t
            tail_true_norm = (targets["tail"] - tail_mean_t) / tail_std_t
            anchor_loss = F.cross_entropy(outputs["anchor_logits"], targets["anchor"], label_smoothing=float(args.label_smoothing))
            local_loss = 0.5 * (F.l1_loss(outputs["local_kernel"], local_true_norm) + F.mse_loss(outputs["local_kernel"], local_true_norm))
            tail_loss = F.mse_loss(outputs["tail_coeffs"], tail_true_norm)
            q_loss = F.mse_loss(q_teacher, targets["q"]) / float(q_scale)
            total = (
                loss_weights["anchor"] * anchor_loss
                + loss_weights["local"] * local_loss
                + loss_weights["tail"] * tail_loss
                + loss_weights["q"] * q_loss
            )
            total.backward()
            optimizer.step()
            bs = int(targets["anchor"].shape[0])
            train_loss_sum += float(total.detach().cpu()) * bs
            train_count += bs

        train_metrics = evaluate_epoch(train_loader, model, variant, device, basis_t, bundle.meta, target_stats, loss_weights, float(args.label_smoothing))
        val_metrics = evaluate_epoch(val_loader, model, variant, device, basis_t, bundle.meta, target_stats, loss_weights, float(args.label_smoothing))
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train_anchor_exact": train_metrics["anchor_exact"],
            "train_q_raw_mse": train_metrics["q_raw_mse"],
            "val_anchor_exact": val_metrics["anchor_exact"],
            "val_q_raw_mse": val_metrics["q_raw_mse"],
        }
        if epoch == 1 or epoch % int(args.replay_eval_interval) == 0 or epoch == int(args.epochs):
            _, q_train_probe = predict_q(model, variant, feats, train_probe, device, bundle.basis, bundle.meta, target_stats)
            _, q_val_probe = predict_q(model, variant, feats, val_probe, device, bundle.basis, bundle.meta, target_stats)
            train_replay = summarize_replay(bundle, room_indices=train_probe, w_ai_batch=q_to_w_canon_batch(bundle, q_train_probe, room_indices=train_probe), early_window_s=float(args.replay_early_window_s))
            val_replay = summarize_replay(bundle, room_indices=val_probe, w_ai_batch=q_to_w_canon_batch(bundle, q_val_probe, room_indices=val_probe), early_window_s=float(args.replay_early_window_s))
            row["train_probe_ai_vs_zero_db"] = train_replay["ai_vs_zero_db_mean"]
            row["val_probe_ai_vs_zero_db"] = val_replay["ai_vs_zero_db_mean"]
            row["val_probe_ai_to_h5_gap_db"] = val_replay["ai_to_h5_gap_db_mean"]
            if is_better(val_replay, best_summary):
                best_summary = dict(val_replay)
                best_summary["epoch"] = int(epoch)
                checkpoint = {
                    "args": vars(args),
                    "h5_path": str(h5_path),
                    "input_variant": variant,
                    "model_state_dict": model.state_dict(),
                    "feature_stats": feature_stats,
                    "target_stats": target_stats,
                    "canonical_meta": bundle.meta,
                    "train_indices": train_idx.tolist(),
                    "val_indices": val_idx.tolist(),
                    "train_probe_indices": train_probe,
                    "val_probe_indices": val_probe,
                    "best_replay_metrics": best_summary,
                    "exact_train_probe": exact_train,
                    "exact_val_probe": exact_val,
                }
                torch.save(checkpoint, out_dir / "best_replay.pt")
        history_rows.append(row)
        print(
            f"[Epoch {epoch:03d}] train_q_mse={train_metrics['q_raw_mse']:.4e}, "
            f"val_q_mse={val_metrics['q_raw_mse']:.4e}, "
            f"val_anchor_exact={val_metrics['anchor_exact']:.3f}, "
            f"best_val_probe={(best_summary['ai_vs_zero_db_mean'] if best_summary else float('nan')):.3f} dB"
        )

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in history_rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(history_rows)
    summary = {
        "h5_path": str(h5_path),
        "input_variant": variant,
        "epochs": int(args.epochs),
        "elapsed_s": float(time.time() - t0),
        "train_probe_indices": train_probe,
        "val_probe_indices": val_probe,
        "exact_train_probe": exact_train,
        "exact_val_probe": exact_val,
        "best_replay_metrics": best_summary,
        "final_train_metrics": train_metrics,
        "final_val_metrics": val_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

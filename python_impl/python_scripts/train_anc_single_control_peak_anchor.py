from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.single_control_anchor_anc_common import (
    AnchorDatasetBundle,
    anchor_metrics,
    build_model,
    build_replay_cases,
    fit_anchor_prior_ridge,
    load_anchor_dataset,
    local_energy_capture,
    predict_anchor_prior_ridge,
    reconstruct_w_np,
    reconstruct_w_torch,
    replay_metrics_for_case,
    resolve_h5_path,
    set_seed,
    split_indices,
    standardize,
    stats_general,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-control ANC with geometric peak-anchor parameterization.")
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-kind", choices=("oracle", "acoustic"), default="acoustic")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tail-basis-dim", type=int, default=8)
    parser.add_argument("--local-half-width", type=int, default=5)
    parser.add_argument("--include-s2r", action="store_true")
    parser.add_argument("--anchor-loss-weight", type=float, default=1.0)
    parser.add_argument("--local-loss-weight", type=float, default=1.0)
    parser.add_argument("--tail-loss-weight", type=float, default=0.5)
    parser.add_argument("--full-loss-weight", type=float, default=0.25)
    parser.add_argument("--anchor-label-smoothing", type=float, default=0.05)
    parser.add_argument("--replay-room-count", type=int, default=4)
    parser.add_argument("--replay-eval-interval", type=int, default=5)
    parser.add_argument("--replay-early-window-s", type=float, default=0.25)
    return parser.parse_args()


def build_feature_cache(bundle: AnchorDatasetBundle, train_idx: np.ndarray, model_kind: str, include_s2r: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stats: dict[str, Any] = {}
    feats: dict[str, np.ndarray] = {}
    if str(model_kind) == "acoustic":
        acoustic_ref_raw = bundle.acoustic_ref if not include_s2r else np.concatenate([bundle.acoustic_ref, bundle.s2r_ref], axis=2).astype(np.float32)
        stats["acoustic_shared"] = stats_general(bundle.acoustic_shared, train_idx, reduce_axes=(0,))
        stats["acoustic_ref"] = stats_general(acoustic_ref_raw, train_idx, reduce_axes=(0, 1))
        feats["acoustic_shared"] = standardize(bundle.acoustic_shared, stats["acoustic_shared"]["mean"], stats["acoustic_shared"]["std"])
        feats["acoustic_ref"] = standardize(acoustic_ref_raw, stats["acoustic_ref"]["mean"][None, :], stats["acoustic_ref"]["std"][None, :])
        stats["anchor_prior_fit"] = fit_anchor_prior_ridge(bundle.delay_shared, bundle.delay_ref, bundle.anchors, train_idx=train_idx, alpha=1.0)
        feats["anchor_prior"] = predict_anchor_prior_ridge(
            bundle.delay_shared,
            bundle.delay_ref,
            prior_fit=stats["anchor_prior_fit"],
            tap_len=int(bundle.meta["tap_len"]),
        ).astype(np.int64)
        if include_s2r:
            stats["s2r_ref"] = stats_general(bundle.s2r_ref, train_idx, reduce_axes=(0, 1))
    elif str(model_kind) == "oracle":
        stats["oracle_shared"] = stats_general(bundle.oracle_shared, train_idx, reduce_axes=(0,))
        stats["oracle_ref"] = stats_general(bundle.oracle_ref, train_idx, reduce_axes=(0, 1))
        feats["oracle_shared"] = standardize(bundle.oracle_shared, stats["oracle_shared"]["mean"], stats["oracle_shared"]["std"])
        feats["oracle_ref"] = standardize(bundle.oracle_ref, stats["oracle_ref"]["mean"][None, :], stats["oracle_ref"]["std"][None, :])
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    return feats, stats


def build_tensor_dataset(bundle: AnchorDatasetBundle, feats: dict[str, np.ndarray], indices: np.ndarray, model_kind: str) -> TensorDataset:
    if str(model_kind) == "acoustic":
        return TensorDataset(
            torch.from_numpy(feats["acoustic_shared"][indices]),
            torch.from_numpy(feats["acoustic_ref"][indices]),
            torch.from_numpy(feats["anchor_prior"][indices].astype(np.int64)),
            torch.from_numpy(bundle.anchors[indices].astype(np.int64)),
            torch.from_numpy(bundle.local_target[indices]),
            torch.from_numpy(bundle.tail_coeffs[indices]),
            torch.from_numpy(bundle.w_target[indices]),
        )
    return TensorDataset(
        torch.from_numpy(feats["oracle_shared"][indices]),
        torch.from_numpy(feats["oracle_ref"][indices]),
        torch.from_numpy(bundle.anchors[indices].astype(np.int64)),
        torch.from_numpy(bundle.local_target[indices]),
        torch.from_numpy(bundle.tail_coeffs[indices]),
        torch.from_numpy(bundle.w_target[indices]),
    )


def forward_batch(model_kind: str, model: torch.nn.Module, batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if str(model_kind) == "acoustic":
        acoustic_shared_b, acoustic_ref_b, anchor_prior_b, anchors_b, local_b, tail_coeff_b, w_b = batch
        outputs = model(
            acoustic_shared_b.to(device=device, dtype=torch.float32, non_blocking=True),
            acoustic_ref_b.to(device=device, dtype=torch.float32, non_blocking=True),
            anchor_prior_b.to(device=device, dtype=torch.long, non_blocking=True),
        )
        targets = {
            "anchors": anchors_b.to(device=device, dtype=torch.long, non_blocking=True),
            "local": local_b.to(device=device, dtype=torch.float32, non_blocking=True),
            "tail_coeffs": tail_coeff_b.to(device=device, dtype=torch.float32, non_blocking=True),
            "w": w_b.to(device=device, dtype=torch.float32, non_blocking=True),
        }
        return outputs, targets
    oracle_shared_b, oracle_ref_b, anchors_b, local_b, tail_coeff_b, w_b = batch
    outputs = model(
        oracle_shared_b.to(device=device, dtype=torch.float32, non_blocking=True),
        oracle_ref_b.to(device=device, dtype=torch.float32, non_blocking=True),
    )
    targets = {
        "anchors": anchors_b.to(device=device, dtype=torch.long, non_blocking=True),
        "local": local_b.to(device=device, dtype=torch.float32, non_blocking=True),
        "tail_coeffs": tail_coeff_b.to(device=device, dtype=torch.float32, non_blocking=True),
        "w": w_b.to(device=device, dtype=torch.float32, non_blocking=True),
    }
    return outputs, targets


def evaluate_epoch(
    loader: DataLoader,
    model_kind: str,
    model: torch.nn.Module,
    device: torch.device,
    basis_t: torch.Tensor,
    tap_len: int,
    half_width: int,
    full_scale: float,
    local_mean_t: torch.Tensor,
    local_std_t: torch.Tensor,
    tail_mean_t: torch.Tensor,
    tail_std_t: torch.Tensor,
    label_smoothing: float,
    anchor_weight: float,
    local_weight: float,
    tail_weight: float,
    full_weight: float,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    full_sum = 0.0
    tail_sum = 0.0
    count = 0
    pred_anchor_all: list[np.ndarray] = []
    true_anchor_all: list[np.ndarray] = []
    pred_w_all: list[np.ndarray] = []
    true_w_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            outputs, targets = forward_batch(model_kind, model, batch, device)
            anchor_logits = outputs["anchor_logits"]
            local_pred_norm = outputs["local_kernel"]
            tail_pred_norm = outputs["tail_coeffs"]
            anchors = targets["anchors"]
            local_true = targets["local"]
            tail_true = targets["tail_coeffs"]
            w_true = targets["w"]

            local_true_norm = (local_true - local_mean_t) / local_std_t
            tail_true_norm = (tail_true - tail_mean_t) / tail_std_t
            local_pred = local_pred_norm * local_std_t + local_mean_t
            tail_pred = tail_pred_norm * tail_std_t + tail_mean_t

            pred_full_teacher = reconstruct_w_torch(anchors, local_pred, tail_pred, basis_t, tap_len=int(tap_len), half_width=int(half_width))

            anchor_loss = F.cross_entropy(anchor_logits.reshape(-1, int(tap_len)), anchors.reshape(-1), label_smoothing=float(label_smoothing))
            local_mse = F.mse_loss(local_pred_norm, local_true_norm)
            local_l1 = F.l1_loss(local_pred_norm, local_true_norm)
            local_loss = 0.5 * (local_mse + local_l1)
            tail_loss = F.mse_loss(tail_pred_norm, tail_true_norm)
            full_loss = F.mse_loss(pred_full_teacher, w_true) / float(full_scale)
            total = float(anchor_weight) * anchor_loss + float(local_weight) * local_loss + float(tail_weight) * tail_loss + float(full_weight) * full_loss

            pred_anchor = anchors if str(model_kind) == "oracle" else torch.argmax(anchor_logits, dim=-1)
            pred_full_eval = reconstruct_w_torch(pred_anchor, local_pred, tail_pred, basis_t, tap_len=int(tap_len), half_width=int(half_width))

            bs = int(anchors.shape[0])
            loss_sum += float(total.detach().cpu()) * bs
            full_sum += float(F.mse_loss(pred_full_eval, w_true).detach().cpu()) * bs
            tail_sum += float(tail_loss.detach().cpu()) * bs
            count += bs

            pred_anchor_all.append(pred_anchor.cpu().numpy())
            true_anchor_all.append(anchors.cpu().numpy())
            pred_w_all.append(pred_full_eval.cpu().numpy())
            true_w_all.append(w_true.cpu().numpy())

    pred_anchor_np = np.concatenate(pred_anchor_all, axis=0)
    true_anchor_np = np.concatenate(true_anchor_all, axis=0)
    pred_w_np = np.concatenate(pred_w_all, axis=0)
    true_w_np = np.concatenate(true_w_all, axis=0)
    metrics = anchor_metrics(pred_anchor_np, true_anchor_np)
    metrics["local_energy_capture"] = local_energy_capture(true_w_np, pred_anchor_np, half_width=int(half_width))
    metrics["loss"] = loss_sum / max(count, 1)
    metrics["full_raw_mse"] = full_sum / max(count, 1)
    metrics["tail_loss_norm"] = tail_sum / max(count, 1)
    return metrics


def predict_truncated_w(
    model_kind: str,
    model: torch.nn.Module,
    bundle: AnchorDatasetBundle,
    feats: dict[str, np.ndarray],
    indices: list[int],
    device: torch.device,
    include_s2r: bool,
    target_stats: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    idx = np.asarray(indices, dtype=np.int64)
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
    set_seed(int(args.seed))
    h5_path = resolve_h5_path(args.h5_path)
    bundle = load_anchor_dataset(h5_path, tail_basis_dim=int(args.tail_basis_dim), local_half_width=int(args.local_half_width))
    train_idx, val_idx = split_indices(bundle.w_target.shape[0], val_frac=float(args.val_frac), seed=int(args.seed))
    feats, feature_stats = build_feature_cache(bundle, train_idx, model_kind=str(args.model_kind), include_s2r=bool(args.include_s2r))

    train_ds = build_tensor_dataset(bundle, feats, train_idx, model_kind=str(args.model_kind))
    val_ds = build_tensor_dataset(bundle, feats, val_idx, model_kind=str(args.model_kind))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), generator=torch.Generator().manual_seed(int(args.seed)))
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = build_model(
        str(args.model_kind),
        bundle,
        include_s2r=bool(args.include_s2r),
        dropout=float(args.dropout),
        acoustic_shared_dim=(int(feats["acoustic_shared"].shape[-1]) if str(args.model_kind) == "acoustic" else None),
        acoustic_ref_dim=(int(feats["acoustic_ref"].shape[-1]) if str(args.model_kind) == "acoustic" else None),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.5)
    basis_t = torch.as_tensor(bundle.basis, dtype=torch.float32, device=device)

    target_stats = {
        "local_mean": np.mean(np.asarray(bundle.local_target[train_idx], dtype=np.float32), axis=(0, 1), dtype=np.float64).astype(np.float32),
        "local_std": np.maximum(np.std(np.asarray(bundle.local_target[train_idx], dtype=np.float32), axis=(0, 1), dtype=np.float64).astype(np.float32), np.float32(1.0e-6)),
        "tail_mean": np.mean(np.asarray(bundle.tail_coeffs[train_idx], dtype=np.float32), axis=(0, 1), dtype=np.float64).astype(np.float32),
        "tail_std": np.maximum(np.std(np.asarray(bundle.tail_coeffs[train_idx], dtype=np.float32), axis=(0, 1), dtype=np.float64).astype(np.float32), np.float32(1.0e-6)),
    }
    local_mean_t = torch.as_tensor(target_stats["local_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
    local_std_t = torch.as_tensor(target_stats["local_std"], dtype=torch.float32, device=device).view(1, 1, -1)
    tail_mean_t = torch.as_tensor(target_stats["tail_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
    tail_std_t = torch.as_tensor(target_stats["tail_std"], dtype=torch.float32, device=device).view(1, 1, -1)
    full_scale = float(np.mean(np.asarray(bundle.w_target[train_idx], dtype=np.float32) ** 2) + 1.0e-8)

    rng = np.random.default_rng(int(args.seed) + 17)
    replay_room_indices = sorted(int(v) for v in rng.choice(train_idx, size=min(int(args.replay_room_count), len(train_idx)), replace=False))
    replay_cases = build_replay_cases(bundle, replay_room_indices)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    best_ckpt = out_dir / "best_replay.pt"
    final_ckpt = out_dir / "final.pt"
    summary_path = out_dir / "summary.json"

    best_replay_gain = float("-inf")
    best_replay_gap = float("inf")
    history: list[dict[str, Any]] = []

    print(f"Using device: {device}")
    print(f"HDF5: {h5_path}")
    print(f"Model kind: {args.model_kind}")
    print(f"Train/val sizes: {len(train_idx)}/{len(val_idx)}")
    print(f"Replay rooms: {replay_room_indices}")

    with metrics_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_anchor_exact",
                "train_anchor_within_1",
                "train_local_capture",
                "train_full_raw_mse",
                "val_loss",
                "val_anchor_exact",
                "val_anchor_within_1",
                "val_local_capture",
                "val_full_raw_mse",
                "replay_ai_vs_zero_db",
                "replay_ai_to_h5_gap_db",
                "lr",
                "seconds",
            ],
        )
        writer.writeheader()

        for epoch in range(1, int(args.epochs) + 1):
            t0 = time.time()
            model.train()
            train_loss_sum = 0.0
            pred_anchor_all: list[np.ndarray] = []
            true_anchor_all: list[np.ndarray] = []
            pred_w_all: list[np.ndarray] = []
            true_w_all: list[np.ndarray] = []
            count = 0

            for batch in train_loader:
                outputs, targets = forward_batch(str(args.model_kind), model, batch, device)
                anchor_logits = outputs["anchor_logits"]
                local_pred_norm = outputs["local_kernel"]
                tail_pred_norm = outputs["tail_coeffs"]
                anchors = targets["anchors"]
                local_true = targets["local"]
                tail_true = targets["tail_coeffs"]
                w_true = targets["w"]

                local_true_norm = (local_true - local_mean_t) / local_std_t
                tail_true_norm = (tail_true - tail_mean_t) / tail_std_t
                local_pred = local_pred_norm * local_std_t + local_mean_t
                tail_pred = tail_pred_norm * tail_std_t + tail_mean_t

                pred_full_teacher = reconstruct_w_torch(anchors, local_pred, tail_pred, basis_t, tap_len=int(bundle.meta["tap_len"]), half_width=int(bundle.meta["local_half_width"]))

                anchor_loss = F.cross_entropy(anchor_logits.reshape(-1, int(bundle.meta["tap_len"])), anchors.reshape(-1), label_smoothing=float(args.anchor_label_smoothing))
                local_mse = F.mse_loss(local_pred_norm, local_true_norm)
                local_l1 = F.l1_loss(local_pred_norm, local_true_norm)
                local_loss = 0.5 * (local_mse + local_l1)
                tail_loss = F.mse_loss(tail_pred_norm, tail_true_norm)
                full_loss = F.mse_loss(pred_full_teacher, w_true) / float(full_scale)
                total = (
                    float(args.anchor_loss_weight) * anchor_loss
                    + float(args.local_loss_weight) * local_loss
                    + float(args.tail_loss_weight) * tail_loss
                    + float(args.full_loss_weight) * full_loss
                )

                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                pred_anchor = anchors if str(args.model_kind) == "oracle" else torch.argmax(anchor_logits, dim=-1)
                pred_full_eval = reconstruct_w_torch(pred_anchor, local_pred, tail_pred, basis_t, tap_len=int(bundle.meta["tap_len"]), half_width=int(bundle.meta["local_half_width"]))

                bs = int(anchors.shape[0])
                train_loss_sum += float(total.detach().cpu()) * bs
                count += bs
                pred_anchor_all.append(pred_anchor.detach().cpu().numpy())
                true_anchor_all.append(anchors.detach().cpu().numpy())
                pred_w_all.append(pred_full_eval.detach().cpu().numpy())
                true_w_all.append(w_true.detach().cpu().numpy())

            train_anchor_np = np.concatenate(pred_anchor_all, axis=0)
            train_true_anchor_np = np.concatenate(true_anchor_all, axis=0)
            train_pred_w_np = np.concatenate(pred_w_all, axis=0)
            train_true_w_np = np.concatenate(true_w_all, axis=0)
            train_metrics = anchor_metrics(train_anchor_np, train_true_anchor_np)
            train_metrics["local_energy_capture"] = local_energy_capture(train_true_w_np, train_anchor_np, half_width=int(bundle.meta["local_half_width"]))
            train_metrics["loss"] = train_loss_sum / max(count, 1)
            train_metrics["full_raw_mse"] = float(np.mean((train_pred_w_np - train_true_w_np) ** 2))

            val_metrics = evaluate_epoch(
                loader=val_loader,
                model_kind=str(args.model_kind),
                model=model,
                device=device,
                basis_t=basis_t,
                tap_len=int(bundle.meta["tap_len"]),
                half_width=int(bundle.meta["local_half_width"]),
                full_scale=float(full_scale),
                local_mean_t=local_mean_t,
                local_std_t=local_std_t,
                tail_mean_t=tail_mean_t,
                tail_std_t=tail_std_t,
                label_smoothing=float(args.anchor_label_smoothing),
                anchor_weight=float(args.anchor_loss_weight),
                local_weight=float(args.local_loss_weight),
                tail_weight=float(args.tail_loss_weight),
                full_weight=float(args.full_loss_weight),
            )

            replay_gain = None
            replay_gap = None
            if epoch % max(int(args.replay_eval_interval), 1) == 0 or epoch == int(args.epochs):
                _, pred_w_rooms = predict_truncated_w(
                    model_kind=str(args.model_kind),
                    model=model,
                    bundle=bundle,
                    feats=feats,
                    indices=replay_room_indices,
                    device=device,
                    include_s2r=bool(args.include_s2r),
                    target_stats=target_stats,
                )
                replay_rows = []
                for case, pred_w in zip(replay_cases, pred_w_rooms):
                    replay_rows.append(
                        replay_metrics_for_case(
                            case=case,
                            w_ai=truncated_to_full_w(pred_w, filter_len=int(bundle.cfg.filter_len)),
                            cfg=bundle.cfg,
                            early_window_s=float(args.replay_early_window_s),
                        )
                    )
                replay_gain = float(np.mean([row["ai_vs_zero_db"] for row in replay_rows]))
                replay_gap = float(np.mean([row["ai_to_h5_gap_db"] for row in replay_rows]))
                if replay_gain > best_replay_gain or (np.isclose(replay_gain, best_replay_gain) and replay_gap < best_replay_gap):
                    best_replay_gain = replay_gain
                    best_replay_gap = replay_gap
                    torch.save(
                        {
                            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "model_kind": str(args.model_kind),
                            "include_s2r": bool(args.include_s2r),
                            "args": vars(args),
                            "bundle_meta": bundle.meta,
                            "config_json": json.dumps(bundle.cfg.__dict__, ensure_ascii=False),
                            "feature_stats": feature_stats,
                            "target_stats": target_stats,
                            "basis": bundle.basis,
                            "train_indices": train_idx.tolist(),
                            "val_indices": val_idx.tolist(),
                            "replay_room_indices": replay_room_indices,
                            "best_replay_gain_db": best_replay_gain,
                            "best_replay_gap_db": best_replay_gap,
                        },
                        best_ckpt,
                    )
            scheduler.step(float(replay_gain if replay_gain is not None else val_metrics["anchor_exact"]))

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_anchor_exact": train_metrics["anchor_exact"],
                "train_anchor_within_1": train_metrics["anchor_within_1"],
                "train_local_capture": train_metrics["local_energy_capture"],
                "train_full_raw_mse": train_metrics["full_raw_mse"],
                "val_loss": val_metrics["loss"],
                "val_anchor_exact": val_metrics["anchor_exact"],
                "val_anchor_within_1": val_metrics["anchor_within_1"],
                "val_local_capture": val_metrics["local_energy_capture"],
                "val_full_raw_mse": val_metrics["full_raw_mse"],
                "replay_ai_vs_zero_db": replay_gain,
                "replay_ai_to_h5_gap_db": replay_gap,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.time() - t0,
            }
            writer.writerow(row)
            f_csv.flush()
            history.append(row)
            print(
                f"[Epoch {epoch:03d}] train_exact={row['train_anchor_exact']:.4f} val_exact={row['val_anchor_exact']:.4f} "
                f"train_capture={row['train_local_capture']:.4f} val_capture={row['val_local_capture']:.4f} "
                f"replay_gain={row['replay_ai_vs_zero_db']} gap={row['replay_ai_to_h5_gap_db']} lr={row['lr']:.3e} t={row['seconds']:.1f}s"
            )

    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_kind": str(args.model_kind),
            "include_s2r": bool(args.include_s2r),
            "args": vars(args),
            "bundle_meta": bundle.meta,
            "config_json": json.dumps(bundle.cfg.__dict__, ensure_ascii=False),
            "feature_stats": feature_stats,
            "target_stats": target_stats,
            "basis": bundle.basis,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "replay_room_indices": replay_room_indices,
            "best_replay_gain_db": best_replay_gain,
            "best_replay_gap_db": best_replay_gap,
        },
        final_ckpt,
    )
    summary = {
        "args": vars(args),
        "bundle_meta": bundle.meta,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "replay_room_indices": replay_room_indices,
        "best_replay_gain_db": best_replay_gain,
        "best_replay_gap_db": best_replay_gap,
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_csv}")
    print(f"Saved best checkpoint to {best_ckpt}")
    print(f"Saved final checkpoint to {final_ckpt}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

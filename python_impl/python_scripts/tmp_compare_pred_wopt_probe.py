from __future__ import annotations

import numpy as np
import torch
import sys

from evaluate_hybrid_deep_fxlms_single_control import (
    load_target_metric_values,
    predict_w_batch,
    warmstart_metrics,
)
from train_hybrid_deep_fxlms_single_control import HybridDeepFxLMSNet, load_bundle, resolve_h5_path, level_mask


def main() -> int:
    if len(sys.argv) > 1:
        ckpt_path = str(sys.argv[1])
    else:
        ckpt_path = "python_impl/experiments/anc_single_control/target_goal_pathfeat_woptonly_b128_s7/train/final_hybrid_deep_fxlms.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    h5_path = resolve_h5_path(None)
    bundle = load_bundle(h5_path=h5_path, encoding=args["feature_encoding"], disable_feature_b=bool(args["disable_feature_b"]))

    model = HybridDeepFxLMSNet(
        acoustic_in_channels=int(bundle.acoustic.shape[1]),
        filter_len=int(bundle.p_ref.shape[-1]),
        num_refs=int(bundle.p_ref.shape[1]),
        basis_dim=int(args["basis_dim"]),
        embed_dim=int(args["embed_dim"]),
        fusion_mode=str(args["fusion_mode"]),
        num_heads=int(args["num_heads"]),
        disable_feature_b=bool(args["disable_feature_b"]),
        use_path_features=bool(args.get("use_path_features", False)),
        use_index_embedding=bool(args.get("use_index_embedding", False)),
        num_samples=int(bundle.gcc.shape[0]),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    warm_idx = np.where(level_mask(bundle.image_order, 3))[0].astype(np.int64)
    probe = warm_idx[:8]
    probe_list = [int(v) for v in probe.tolist()]

    w_pred = predict_w_batch(model=model, bundle=bundle, indices=probe, batch_size=8, device=torch.device("cpu"))
    w_opt = bundle.w_opt[probe]

    diff = w_pred - w_opt
    rmse = float(np.sqrt(np.mean(diff**2)))
    opt_rms = float(np.sqrt(np.mean(w_opt**2)))
    rel = rmse / (opt_rms + 1e-12)

    target_vals = load_target_metric_values(h5_path=h5_path, room_indices=probe_list, metric_key="nr_last_db")
    m_pred = warmstart_metrics(
        h5_path=h5_path,
        room_indices=probe_list,
        w_pred=w_pred,
        early_window_s=0.25,
        target_nr_db=target_vals,
        half_target_ratio=0.5,
        target_metric="nr_last_db",
    )
    m_opt = warmstart_metrics(
        h5_path=h5_path,
        room_indices=probe_list,
        w_pred=w_opt,
        early_window_s=0.25,
        target_nr_db=target_vals,
        half_target_ratio=0.5,
        target_metric="nr_last_db",
    )

    print({
        "probe": probe_list,
        "rmse": rmse,
        "opt_rms": opt_rms,
        "relative_rmse": rel,
        "pred_init_nr": m_pred["init_nr_db_mean"],
        "opt_init_nr": m_opt["init_nr_db_mean"],
        "pred_gap": m_pred["half_target_gap_db"],
        "opt_gap": m_opt["half_target_gap_db"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import numpy as np

from evaluate_hybrid_deep_fxlms_single_control import (
    load_target_metric_values,
    warmstart_metrics,
)
from train_hybrid_deep_fxlms_single_control import load_bundle, resolve_h5_path, level_mask


def main() -> int:
    h5_path = resolve_h5_path(None)
    bundle = load_bundle(h5_path=h5_path, encoding="ri", disable_feature_b=False)

    warm_level = 3
    warm_cases = 8
    warm_idx = np.where(level_mask(bundle.image_order, warm_level))[0].astype(np.int64)
    probe = warm_idx[: min(warm_cases, warm_idx.size)]

    gcc_flat = bundle.gcc.reshape(bundle.gcc.shape[0], -1)
    ac_flat = bundle.acoustic.reshape(bundle.acoustic.shape[0], -1)
    feat = np.concatenate([gcc_flat, ac_flat], axis=1).astype(np.float32)

    # Leave-one-out 1-NN retrieval on full pool.
    w_pred = []
    for idx in probe:
        dist = np.sum((feat - feat[idx : idx + 1]) ** 2, axis=1)
        dist[idx] = np.inf
        j = int(np.argmin(dist))
        w_pred.append(bundle.w_opt[j])
    w_pred = np.asarray(w_pred, dtype=np.float32)

    target_vals = load_target_metric_values(
        h5_path=h5_path,
        room_indices=[int(v) for v in probe.tolist()],
        metric_key="nr_last_db",
    )

    metrics = warmstart_metrics(
        h5_path=h5_path,
        room_indices=[int(v) for v in probe.tolist()],
        w_pred=w_pred,
        early_window_s=0.25,
        target_nr_db=target_vals,
        half_target_ratio=0.5,
        target_metric="nr_last_db",
    )

    print("probe_indices:", probe.tolist())
    print("metrics:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

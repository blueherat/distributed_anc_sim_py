from __future__ import annotations

import numpy as np
import torch

from train_hybrid_deep_fxlms_single_control import load_bundle, resolve_h5_path
from py_anc.algorithms.hybrid_loss import HybridAcousticLoss


def main() -> int:
    h5 = resolve_h5_path(None)
    bundle = load_bundle(h5_path=h5, encoding="ri", disable_feature_b=False)

    idx = np.arange(0, 64)
    w_opt = torch.from_numpy(bundle.w_opt[idx]).float()
    p_ref = torch.from_numpy(bundle.p_ref[idx]).float()
    d_path = torch.from_numpy(bundle.d_path[idx]).float()
    s_path = torch.from_numpy(bundle.s_path[idx]).float()

    loss = HybridAcousticLoss(lambda_reg=1e-3, conv_domain="freq")
    out_opt = loss(w_opt, p_ref, d_path, s_path)
    out_zero = loss(torch.zeros_like(w_opt), p_ref, d_path, s_path)

    print(
        {
            "nr_opt": float(out_opt["nr_db"]),
            "nr_zero": float(out_zero["nr_db"]),
            "acoustic_opt": float(out_opt["acoustic"]),
            "acoustic_zero": float(out_zero["acoustic"]),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from typing import Dict

import numpy as np


def cfxlms(params: Dict):
    """Centralized FxLMS ANC algorithm."""
    time = params["time"]
    rir_manager = params["rir_manager"]
    l = int(params["L"])
    mu = float(params["mu"])
    verbose = bool(params.get("verbose", False))
    normalized_update = bool(params.get("normalized_update", False))
    norm_epsilon = float(params.get("norm_epsilon", 1.0e-8))
    x = np.asarray(params["reference_signal"], dtype=float)
    d = np.asarray(params["desired_signal"], dtype=float)

    key_sec_spks = list(rir_manager.secondary_speakers.keys())
    key_err_mics = list(rir_manager.error_microphones.keys())

    num_ref_mics = len(rir_manager.reference_microphones)
    num_sec_spks = len(key_sec_spks)
    num_err_mics = len(key_err_mics)

    if x.shape[1] != num_ref_mics:
        raise ValueError(
            f"reference_signal columns ({x.shape[1]}) must match number of reference microphones ({num_ref_mics})."
        )

    n_samples = len(time)

    # 缓存次级路径，避免在样本循环内重复字典查询与数组构造。
    s_paths = [[None for _ in range(num_err_mics)] for _ in range(num_sec_spks)]
    s_lens = np.zeros((num_sec_spks, num_err_mics), dtype=np.int32)
    max_ls_hat = 0
    for k, sec_id in enumerate(key_sec_spks):
        for m, err_id in enumerate(key_err_mics):
            s = np.asarray(rir_manager.get_secondary_rir(sec_id, err_id), dtype=float)
            s_paths[k][m] = s
            s_lens[k, m] = int(s.size)
            if s.size > max_ls_hat:
                max_ls_hat = int(s.size)

    w = np.zeros((l, num_ref_mics, num_sec_spks), dtype=float)
    x_taps = np.zeros((max(l, max_ls_hat), num_ref_mics), dtype=float)
    xf_taps = np.zeros((l, num_ref_mics, num_sec_spks, num_err_mics), dtype=float)

    e = np.zeros((n_samples, num_err_mics), dtype=float)

    y_taps = []
    for k in range(num_sec_spks):
        sec_len = int(np.max(s_lens[k]))
        y_taps.append(np.zeros(sec_len, dtype=float))

    if verbose:
        print("开始集中式FxLMS仿真...")
    for n in range(n_samples):
        x_taps[1:, :] = x_taps[:-1, :]
        x_taps[0, :] = x[n, :]

        for k in range(num_sec_spks):
            y = np.sum(w[:, :, k] * x_taps[:l, :])
            y_taps[k][1:] = y_taps[k][:-1]
            y_taps[k][0] = y

        for m in range(num_err_mics):
            yf = 0.0
            for k in range(num_sec_spks):
                s = s_paths[k][m]
                ls = int(s_lens[k, m])
                yf += float(np.dot(s, y_taps[k][:ls]))
            e[n, m] = d[n, m] + yf

        xf_taps[1:, :, :, :] = xf_taps[:-1, :, :, :]
        for k in range(num_sec_spks):
            for m in range(num_err_mics):
                s = s_paths[k][m]
                ls_hat = int(s_lens[k, m])
                xf_taps[0, :, k, m] = s @ x_taps[:ls_hat, :]

        for k in range(num_sec_spks):
            grad_k = np.zeros((l, num_ref_mics), dtype=float)
            for m in range(num_err_mics):
                phi = xf_taps[:, :, k, m]
                if normalized_update:
                    denom = float(np.sum(phi * phi)) + norm_epsilon
                    grad_k += (phi * e[n, m]) / denom
                else:
                    grad_k += phi * e[n, m]
            w[:, :, k] = w[:, :, k] - mu * grad_k

    filter_coeffs = {sec_id: w[:, :, k].copy() for k, sec_id in enumerate(key_sec_spks)}
    return {"err_hist": e, "filter_coeffs": filter_coeffs}

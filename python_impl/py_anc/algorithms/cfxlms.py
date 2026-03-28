from __future__ import annotations

from typing import Dict

import numpy as np


def cfxlms(params: Dict):
    """Centralized FxLMS ANC algorithm."""
    time = params["time"]
    rir_manager = params["rir_manager"]
    l = int(params["L"])
    mu = float(params["mu"])
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

    max_ls_hat = 0
    for sec_id in key_sec_spks:
        ls_hat = len(rir_manager.get_secondary_rir(sec_id, key_err_mics[0]))
        if ls_hat > max_ls_hat:
            max_ls_hat = ls_hat

    w = np.zeros((l, num_ref_mics, num_sec_spks), dtype=float)
    x_taps = np.zeros((max(l, max_ls_hat), num_ref_mics), dtype=float)
    xf_taps = np.zeros((l, num_ref_mics, num_sec_spks, num_err_mics), dtype=float)

    e = np.zeros((n_samples, num_err_mics), dtype=float)

    y_taps = []
    for sec_id in key_sec_spks:
        sec_len = len(rir_manager.get_secondary_rir(sec_id, key_err_mics[0]))
        y_taps.append(np.zeros(sec_len, dtype=float))

    print("开始集中式FxLMS仿真...")
    for n in range(n_samples):
        x_taps = np.vstack([x[n : n + 1, :], x_taps[:-1, :]])

        for k in range(num_sec_spks):
            y = np.sum(w[:, :, k] * x_taps[:l, :])
            y_taps[k] = np.concatenate(([y], y_taps[k][:-1]))

        for m, err_id in enumerate(key_err_mics):
            yf = 0.0
            for k, sec_id in enumerate(key_sec_spks):
                s = rir_manager.get_secondary_rir(sec_id, err_id)
                ls = len(s)
                yf += float(np.dot(s, y_taps[k][:ls]))
            e[n, m] = d[n, m] + yf

        xf = np.zeros((1, num_ref_mics, num_sec_spks, num_err_mics), dtype=float)
        for k, sec_id in enumerate(key_sec_spks):
            for m, err_id in enumerate(key_err_mics):
                s = rir_manager.get_secondary_rir(sec_id, err_id)
                ls_hat = len(s)
                xf[0, :, k, m] = s @ x_taps[:ls_hat, :]

        xf_taps = np.concatenate([xf, xf_taps[:-1, :, :, :]], axis=0)

        for k in range(num_sec_spks):
            for m in range(num_err_mics):
                w[:, :, k] = w[:, :, k] - mu * xf_taps[:, :, k, m] * e[n, m]

    filter_coeffs = {sec_id: w[:, :, k].copy() for k, sec_id in enumerate(key_sec_spks)}
    return {"err_hist": e, "filter_coeffs": filter_coeffs}

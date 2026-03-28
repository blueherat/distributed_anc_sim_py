from __future__ import annotations

from typing import Dict

import numpy as np


def dcfxlms(params: Dict):
    """Distributed decentralized FxLMS algorithm."""
    time = params["time"]
    rir_manager = params["rir_manager"]
    network = params["network"]
    l = int(params["L"])
    x = np.asarray(params["reference_signal"], dtype=float)
    d = np.asarray(params["desired_signal"], dtype=float)

    key_ref_mics = list(rir_manager.reference_microphones.keys())
    key_sec_spks = list(rir_manager.secondary_speakers.keys())
    key_err_mics = list(rir_manager.error_microphones.keys())

    ref_idx = {rid: i for i, rid in enumerate(key_ref_mics)}
    sec_idx = {sid: i for i, sid in enumerate(key_sec_spks)}
    err_idx = {eid: i for i, eid in enumerate(key_err_mics)}

    num_ref_mics = len(key_ref_mics)
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

    x_taps = np.zeros((max(l, max_ls_hat), num_ref_mics), dtype=float)
    e = np.zeros((n_samples, num_err_mics), dtype=float)

    y_taps = []
    for sec_id in key_sec_spks:
        sec_len = len(rir_manager.get_secondary_rir(sec_id, key_err_mics[0]))
        y_taps.append(np.zeros(sec_len, dtype=float))

    for node in network.nodes.values():
        if node.ref_mic_id not in ref_idx:
            raise ValueError(f"Node {node.node_id} has invalid ref_mic_id={node.ref_mic_id}")
        node.init(l)

    print("开始DCFxLMS仿真...")
    for n in range(n_samples):
        x_taps = np.vstack([x[n : n + 1, :], x_taps[:-1, :]])

        for node in network.nodes.values():
            ref_col = ref_idx[node.ref_mic_id]
            y = float(node.w @ x_taps[:l, ref_col])
            k = sec_idx[node.sec_spk_id]
            y_taps[k] = np.concatenate(([y], y_taps[k][:-1]))

        for m, err_id in enumerate(key_err_mics):
            yf = 0.0
            for k, sec_id in enumerate(key_sec_spks):
                s = rir_manager.get_secondary_rir(sec_id, err_id)
                ls = len(s)
                yf += float(np.dot(s, y_taps[k][:ls]))
            e[n, m] = d[n, m] + yf

        for node in network.nodes.values():
            ref_col = ref_idx[node.ref_mic_id]
            s_hat = rir_manager.get_secondary_rir(node.sec_spk_id, node.err_mic_id)
            ls_hat = len(s_hat)
            xf = float(s_hat @ x_taps[:ls_hat, ref_col])
            node.xf_taps = np.concatenate(([xf], node.xf_taps[:-1]))

        for node in network.nodes.values():
            node.w = node.w - node.step_size * e[n, err_idx[node.err_mic_id]] * node.xf_taps

    filter_coeffs = {node.sec_spk_id: node.w.copy() for node in network.nodes.values()}
    return {"err_hist": e, "filter_coeffs": filter_coeffs}

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.linalg import convolution_matrix


def mgdfxlms(params: Dict):
    """Distributed MGDFxLMS algorithm."""
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

    print("预计算交叉路径补偿滤波器 (ckm)...")
    for node in network.nodes.values():
        for neighbor_id in node.neighbor_ids:
            if neighbor_id == node.node_id:
                continue
            neighbor = network.nodes[neighbor_id]

            s_hat_km = np.asarray(rir_manager.get_secondary_rir(node.sec_spk_id, neighbor.err_mic_id), dtype=float)
            s_hat_mm = np.asarray(rir_manager.get_secondary_rir(neighbor.sec_spk_id, neighbor.err_mic_id), dtype=float)

            mtx = convolution_matrix(s_hat_mm, node.lc, mode="full")
            expected_len = mtx.shape[0]
            current_len = len(s_hat_km)

            if current_len < expected_len:
                b = np.concatenate([s_hat_km, np.zeros(expected_len - current_len, dtype=float)])
            elif current_len > expected_len:
                b = s_hat_km[:expected_len]
                print(
                    f"Warning: Truncating S_hat_km from length {current_len} to {expected_len} "
                    f"for node {node.node_id} -> neighbor {neighbor_id}"
                )
            else:
                b = s_hat_km

            ckm, *_ = np.linalg.lstsq(mtx, b, rcond=None)
            node.ckm_taps[neighbor_id] = ckm

            s_km_approx = np.convolve(s_hat_mm, ckm)[: len(s_hat_km)]
            rel_err = np.linalg.norm(s_km_approx - s_hat_km) / (np.linalg.norm(s_hat_km) + 1e-12)
            print(
                f"Node {node.node_id} -> Neighbor {neighbor_id}: 相对误差 = {rel_err:.4f} ({rel_err * 100:.1f}%)"
            )

    print("开始MGDFxLMS仿真...")
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
            node.gradient = e[n, err_idx[node.err_mic_id]] * node.xf_taps
            node.direction = node.gradient.copy()

        for node in network.nodes.values():
            for neighbor_id in node.neighbor_ids:
                if neighbor_id == node.node_id:
                    continue
                neighbor = network.nodes[neighbor_id]

                ckm = node.ckm_taps[neighbor.node_id]
                ckm_flipped = ckm[::-1]
                full_conv = np.convolve(neighbor.gradient, ckm_flipped)
                start = node.lc - 1
                correction = full_conv[start : start + l]

                if correction.shape[0] < l:
                    correction = np.pad(correction, (0, l - correction.shape[0]))

                node.direction = node.direction + correction

        for node in network.nodes.values():
            node.w = node.w - node.step_size * node.direction

    filter_coeffs = {node.sec_spk_id: node.w.copy() for node in network.nodes.values()}
    return {"err_hist": e, "filter_coeffs": filter_coeffs}

from __future__ import annotations

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_tap_weights(filter_coeffs_list: Sequence[Dict[int, np.ndarray]], alg_names: Sequence[str], sec_spk_ids: Sequence[int]):
    if len(filter_coeffs_list) != len(alg_names):
        raise ValueError("filter coeff lists and algorithm names must have the same length.")

    fig, ax = plt.subplots(figsize=(11, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(alg_names)))

    l = 0
    for idx, coeff_dict in enumerate(filter_coeffs_list):
        all_coeffs = []
        for spk_id in sec_spk_ids:
            w = coeff_dict.get(spk_id)
            if w is None:
                if l > 0:
                    all_coeffs.append(np.zeros(l, dtype=float))
                continue
            w_vec = np.asarray(w)
            if w_vec.ndim > 1:
                w_vec = w_vec.reshape(-1)
            if l == 0:
                l = len(w_vec)
            all_coeffs.append(w_vec)

        if all_coeffs:
            ax.plot(np.concatenate(all_coeffs), color=colors[idx], linewidth=1.2, label=alg_names[idx])

    if l > 0:
        num_spks = len(sec_spk_ids)
        for k in range(1, num_spks):
            ax.axvline(k * l, color=(0.8, 0.8, 0.8), linestyle="--", linewidth=1.0)
        ax.set_xlim([0, num_spks * l])

    ax.set_title("Comparison of Concatenated Tap Weights")
    ax.set_xlabel("Tap Index (Grouped by Speaker)")
    ax.set_ylabel("Tap Weight")
    ax.grid(True)
    ax.legend(loc="best")
    return fig

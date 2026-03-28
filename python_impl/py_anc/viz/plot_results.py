from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def plot_results(
    time: np.ndarray,
    d_ch: np.ndarray,
    error_signals: Sequence[np.ndarray],
    alg_names: Sequence[str],
    mic_id: int,
    fs: float,
):
    if len(error_signals) != len(alg_names):
        raise ValueError("The number of error signals and algorithm names must match.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(error_signals)))

    axes[0].plot(time, d_ch, "-", color=(0.3010, 0.7450, 0.9330), linewidth=1.0, label=f"期望信号 d_{mic_id}(n)")
    for idx, err in enumerate(error_signals):
        axes[0].plot(time, err, "-", color=colors[idx], linewidth=1.2, label=f"误差 ({alg_names[idx]})")
    axes[0].set_title(f"麦克风 {mic_id}: 时域信号对比")
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("幅值")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    duration = float(time[-1]) if len(time) > 0 else 0.0
    if duration > 1.0:
        start_idx = int(np.searchsorted(time, duration - 1.0))
    else:
        start_idx = 0

    d_segment = d_ch[start_idx:]
    f, p_d = welch(d_segment, fs=fs)
    axes[1].plot(f, 10.0 * np.log10(p_d + np.finfo(float).eps), "-", color=(0.3010, 0.7450, 0.9330), linewidth=1.0, label=f"PSD of d_{mic_id}")

    for idx, err in enumerate(error_signals):
        e_segment = err[start_idx:]
        f_e, p_e = welch(e_segment, fs=fs)
        axes[1].plot(f_e, 10.0 * np.log10(p_e + np.finfo(float).eps), "-", color=colors[idx], linewidth=1.2, label=f"PSD of error ({alg_names[idx]})")

    axes[1].set_title(f"麦克风 {mic_id}: 功率谱密度对比 (信号末段)")
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("功率/频率 (dB/Hz)")
    axes[1].set_xlim([0, fs / 2.0])
    axes[1].grid(True)
    axes[1].legend(loc="best")

    return fig

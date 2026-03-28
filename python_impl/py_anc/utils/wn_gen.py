from __future__ import annotations

import numpy as np


def wn_gen(fs: int, duration: float, f_low: float, f_high: float, rng: np.random.Generator | None = None):
    """Generate band-limited white noise with random phase in frequency domain."""
    if rng is None:
        rng = np.random.default_rng()

    n = int(round(fs * duration))
    if n % 2 != 0:
        n += 1

    x_freq = np.zeros(n, dtype=complex)
    df = fs / n

    idx_low = int(np.floor(f_low / df))
    idx_high = int(np.ceil(f_high / df))

    idx_low = max(idx_low, 0)
    idx_high = min(idx_high, n // 2)

    if idx_low <= idx_high:
        num_pts = idx_high - idx_low + 1
        rand_phase = 2.0 * np.pi * rng.random(num_pts)
        x_freq[idx_low : idx_high + 1] = np.exp(1j * rand_phase)

        # Mirror to negative frequencies for real-valued time-domain signal.
        if idx_low == 0:
            pos = np.arange(1, idx_high + 1)
        else:
            pos = np.arange(idx_low, idx_high + 1)

        neg = (-pos) % n
        x_freq[neg] = np.conj(x_freq[pos])

    signal = np.real(np.fft.ifft(x_freq)).reshape(-1, 1)
    t = np.arange(n, dtype=float).reshape(-1, 1) / float(fs)
    return signal, t

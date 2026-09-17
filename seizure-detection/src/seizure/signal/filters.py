"""Notch and bandpass. R21/R22: notch first, before windowing, always.

Boston mains is 60 Hz and the gamma band (30-80 Hz) sits directly on top of it.
Line-noise amplitude varies by recording session, so un-notched gamma power is
partly a session fingerprint -- the same leakage as a bad split in different
clothing (PRD §7.3). Measured on csv_scaled, un-notched 60 Hz power is ~1000x
its neighbours at 55 and 65 Hz.

Filtering happens on whole records, never on 10 s windows: PRD §14 names
post-windowing filtering as the cause of gamma dominance, because a short
segment gives the filter no room to settle and leaves edge transients in every
window.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig

NOTCH_Q = 30.0


def notch(x: np.ndarray, fs: float, freqs: list[float], q: float = NOTCH_Q) -> np.ndarray:
    """Zero-phase IIR notch at each frequency. x is (n_channels, n_samples)."""
    out = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    for f0 in freqs:
        if not (0 < f0 < nyq):
            continue  # 120 Hz is unusable at fs=256 only if nyq <= 120; it is not
        b, a = sig.iirnotch(f0, q, fs)
        out = sig.filtfilt(b, a, out, axis=-1)
    return out


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass."""
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.99)
    sos = sig.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sig.sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def preprocess(
    x: np.ndarray, fs: float, notch_hz: list[float], band: tuple[float, float]
) -> np.ndarray:
    """R22 order: notch, then bandpass. Returns float32 to halve table size."""
    y = notch(x, fs, notch_hz)
    y = bandpass(y, fs, band[0], band[1])
    return np.ascontiguousarray(y, dtype=np.float32)


def line_noise_ratio(x: np.ndarray, fs: float, f0: float = 60.0, delta: float = 5.0) -> float:
    """Power at f0 divided by the mean power at f0 +/- delta.

    The R21 enforcement test asserts this is near 1 after filtering. Before
    filtering it is roughly 1000 on this data, which is the whole point.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    nper = min(1024, x.shape[-1])
    f, p = sig.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)
    pm = p.mean(axis=0)

    def at(target: float) -> float:
        return float(pm[int(np.argmin(np.abs(f - target)))])

    side = 0.5 * (at(f0 - delta) + at(f0 + delta))
    return at(f0) / side if side > 0 else float("inf")

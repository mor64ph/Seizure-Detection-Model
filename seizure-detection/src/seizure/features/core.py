"""The 14 per-channel-per-window features (PRD §8.1). Pure numpy/scipy.

Absolute band powers are log-transformed because they are heavy-tailed and
unlogged values dominate any linear model. Relative band powers are kept
alongside them because absolute EEG amplitude varies with electrode impedance,
skull thickness, age and session gain -- it is close to a subject fingerprint,
so relative power should transfer better across subjects (PRD §8.2). Reporting
the comparison is the point of computing both.

R17: nothing here hardcodes a channel count or a window length.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig

TEMPORAL_NAMES = ("line_length", "variance", "hjorth_mobility", "hjorth_complexity")


def feature_names(bands: dict[str, tuple[float, float]]) -> list[str]:
    """Deterministic order, used for every column-name construction."""
    names = [f"bp_{b}_log" for b in bands] + [f"bp_{b}_rel" for b in bands]
    return names + list(TEMPORAL_NAMES)


# -- temporal ------------------------------------------------------------
def line_length(x: np.ndarray) -> np.ndarray:
    """sum |diff(x)| along time. Linear ramp of slope m, length n -> m*(n-1)."""
    return np.abs(np.diff(x, axis=-1)).sum(axis=-1)


def variance(x: np.ndarray) -> np.ndarray:
    return x.var(axis=-1)


def _mobility(x: np.ndarray) -> np.ndarray:
    v = x.var(axis=-1)
    dv = np.diff(x, axis=-1).var(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, np.sqrt(dv / np.where(v > 0, v, 1.0)), 0.0)


def hjorth_mobility(x: np.ndarray) -> np.ndarray:
    return _mobility(x)


def hjorth_complexity(x: np.ndarray) -> np.ndarray:
    """mobility(dx) / mobility(x). Equals ~1 for a pure sine."""
    m = _mobility(x)
    md = _mobility(np.diff(x, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m > 0, md / np.where(m > 0, m, 1.0), 0.0)


# -- spectral ------------------------------------------------------------
def band_powers(
    x: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    nperseg: int = 512,
    noverlap: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD integrated over band edges.

    Returns (absolute, relative), each (..., n_bands). Relative power divides by
    total power over the full analysed span, so the columns sum to ~1 when the
    bands tile the band-limited signal.
    """
    nper = int(min(nperseg, x.shape[-1]))
    nov = int(min(noverlap, max(0, nper - 1)))
    f, p = sig.welch(x, fs=fs, nperseg=nper, noverlap=nov, axis=-1)

    lo = min(b[0] for b in bands.values())
    hi = max(b[1] for b in bands.values())
    total = np.trapezoid(
        p[..., (f >= lo) & (f <= hi)], f[(f >= lo) & (f <= hi)], axis=-1
    )

    absol = []
    for b0, b1 in bands.values():
        m = (f >= b0) & (f <= b1)
        absol.append(np.trapezoid(p[..., m], f[m], axis=-1) if m.sum() > 1 else p[..., m].sum(-1))
    absol_arr = np.stack(absol, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = absol_arr / np.where(total > 0, total, 1.0)[..., None]
    return absol_arr, np.where(np.isfinite(rel), rel, 0.0)


def extract(
    x: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    nperseg: int = 512,
    noverlap: int = 256,
) -> np.ndarray:
    """All 14 features for x of shape (..., n_samples) -> (..., 14).

    Column order matches feature_names(bands).
    """
    absol, rel = band_powers(x, fs, bands, nperseg, noverlap)
    cols = [
        np.log1p(np.clip(absol, 0, None)),
        rel,
        line_length(x)[..., None],
        variance(x)[..., None],
        hjorth_mobility(x)[..., None],
        hjorth_complexity(x)[..., None],
    ]
    out = np.concatenate([c if c.ndim == absol.ndim else c for c in cols], axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

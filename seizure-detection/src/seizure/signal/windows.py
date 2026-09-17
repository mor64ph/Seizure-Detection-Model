"""Windowing and label assignment (PRD §6.2).

Windows are cut inside a single record, never across a file boundary (PRD §0
rule 2): consecutive .edf files are separated by hardware gaps of up to ~10 s,
so a window spanning the seam would contain a discontinuity that no filter can
repair and no label can describe.

Labelling, for a window [t, t+L):

    overlap_frac = seconds of window inside any seizure interval / L
    label = 1  if overlap_frac >= min_overlap_frac
    label = -1 if the window is within guard_sec of an interval boundary
    label = 0  otherwise

The guard band exists because onsets are human-annotated to the nearest second
and the ictal transition is physiologically gradual, so boundary-straddling
windows are genuinely ambiguous. Training on them as hard negatives injects
label noise. Label -1 rows are excluded from training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GUARD = -1
INTERICTAL = 0
ICTAL = 1


@dataclass(frozen=True)
class WindowSpec:
    length_sec: float
    stride_sec: float
    min_overlap_frac: float
    guard_sec: float
    sample_rate: int

    @property
    def length_samples(self) -> int:
        return int(round(self.length_sec * self.sample_rate))

    @property
    def stride_samples(self) -> int:
        return int(round(self.stride_sec * self.sample_rate))


def overlap_seconds(t0: float, t1: float, intervals: list[tuple[int, int]]) -> float:
    """Total seconds of [t0, t1) inside any interval. Intervals may touch."""
    return float(sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in intervals))


def near_boundary(t0: float, t1: float, intervals: list[tuple[int, int]], guard: float) -> bool:
    """True if the window comes within `guard` seconds of any onset or offset
    without being fully inside the interval."""
    for a, b in intervals:
        if t1 > a - guard and t0 < a + guard:
            return True
        if t1 > b - guard and t0 < b + guard:
            return True
    return False


def window_bounds(n_samples: int, spec: WindowSpec) -> list[tuple[int, int]]:
    """Sample index pairs. Trailing partial window is dropped, not zero-padded:
    a short window would have a different Welch resolution and a different
    line-length scale, silently changing the features."""
    L, S = spec.length_samples, spec.stride_samples
    if L <= 0 or S <= 0:
        raise ValueError("window length and stride must be positive")
    return [(s, s + L) for s in range(0, max(0, n_samples - L + 1), S)]


def label_window(
    t0: float,
    t1: float,
    ictal: list[tuple[int, int]],
    spec: WindowSpec,
    extra_guard: list[tuple[int, int]] | None = None,
) -> tuple[int, float]:
    """Returns (label, overlap_frac)."""
    length = t1 - t0
    ov = overlap_seconds(t0, t1, ictal)
    frac = ov / length if length > 0 else 0.0

    if frac >= spec.min_overlap_frac:
        return ICTAL, frac
    # Disputed spans (labels/disputed.py) are forced to guard before the
    # interictal decision, so a source conflict never becomes a hard negative.
    if extra_guard and overlap_seconds(t0, t1, extra_guard) > 0:
        return GUARD, frac
    if frac > 0 or near_boundary(t0, t1, ictal, spec.guard_sec):
        return GUARD, frac
    return INTERICTAL, frac


def build_table(
    record_id: str,
    n_samples: int,
    ictal: list[tuple[int, int]],
    spec: WindowSpec,
    extra_guard: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Key columns for every window in one record (PRD §5.3)."""
    rows = []
    for idx, (s0, s1) in enumerate(window_bounds(n_samples, spec)):
        t0, t1 = s0 / spec.sample_rate, s1 / spec.sample_rate
        lab, frac = label_window(t0, t1, ictal, spec, extra_guard)
        rows.append(
            {
                "window_uid": f"{record_id}_{idx:06d}",
                "record_id": record_id,
                "window_idx": idx,
                "t_start_sec": t0,
                "t_end_sec": t1,
                "label": lab,
                "overlap_frac": frac,
                "_s0": s0,
                "_s1": s1,
            }
        )
    return rows


def overlap_sensitivity(
    rows: list[dict], fracs: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75)
) -> list[dict]:
    """PRD §6.2 asks for positive counts at several thresholds -- a one-line
    finding, not a research project."""
    arr = np.array([r["overlap_frac"] for r in rows], dtype=float)
    return [
        {"min_overlap_frac": f, "n_positive": int((arr >= f).sum()), "n_windows": len(arr)}
        for f in fracs
    ]

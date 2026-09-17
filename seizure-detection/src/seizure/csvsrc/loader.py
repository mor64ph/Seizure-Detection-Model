"""Loader for the pre-derived ``EEG_Scaled_data.csv``.

R1/R3: this is the *other* dataset. It looks like CHB-MIT and is not. Measured
shape: 11,233 rows of 18 channels x 2048 samples, i.e. 8-second windows at
256 Hz, channel-major, already globally scaled, with **no subject, patient or
record identifier in any column**. The headline cross-subject claim is
therefore unobtainable from it (R3), and results derived from it never claim
scaler discipline (R4).

What it is good for is the leakage study, and that study needs code rather than
a notebook session, because the numbers it produces are cited in RULES.

Layout was derived, not assumed: all 17 boundaries at multiples of 2048 show
3.2-5.2x the mean absolute first difference while interior points sit at 0.998,
and lag-1 autocorrelation is 0.78. Assuming 256 Hz places a razor-sharp ~1000x
spectral peak at exactly 60.0 Hz with a 120 Hz harmonic, which self-confirms
the rate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

N_CHANNELS = 18
WINDOW_SAMPLES = 2048
SAMPLE_RATE = 256
N_FEATURES_EXPECTED = N_CHANNELS * WINDOW_SAMPLES  # 36,864


@dataclass
class CsvBlock:
    """One chunk of rows, reshaped."""
    data: np.ndarray      # (n, 18, 2048) float32
    labels: np.ndarray    # (n,) int8
    hashes: list[bytes]   # per-row digest of the feature text, for dedup
    row_index: np.ndarray  # original 0-based row numbers


def header_shape(path: Path) -> tuple[int, int]:
    """(n_feature_columns, has_target). Verified before any parsing."""
    with open(path, encoding="utf-8") as f:
        cols = f.readline().rstrip("\r\n").split(",")
    has_target = cols[-1].strip().lower() == "target"
    return len(cols) - (1 if has_target else 0), has_target


def iter_blocks(path: Path, chunk: int = 512) -> Iterator[CsvBlock]:
    """Stream the file in chunks, reshaping each row to (18, 2048).

    Streamed rather than loaded because the full array is 11,233 x 18 x 2048
    float32 = 1.65 GB, and nothing downstream needs it resident at once.

    The per-row hash is taken over the *feature text* before parsing, so it
    detects the exact duplicate rows the file actually contains without
    depending on float round-tripping.
    """
    n_feat, has_target = header_shape(path)
    if n_feat != N_FEATURES_EXPECTED:
        raise ValueError(
            f"expected {N_FEATURES_EXPECTED} feature columns "
            f"({N_CHANNELS} x {WINDOW_SAMPLES}), found {n_feat}"
        )
    if not has_target:
        raise ValueError("no 'target' column found")

    rows: list[np.ndarray] = []
    labs: list[int] = []
    hsh: list[bytes] = []
    idx: list[int] = []

    with open(path, encoding="utf-8") as f:
        f.readline()  # header
        for i, line in enumerate(f):
            s = line.rstrip("\r\n")
            cut = s.rindex(",")
            hsh.append(hashlib.blake2b(s[:cut].encode(), digest_size=16).digest())
            labs.append(int(s[cut + 1 :]))
            rows.append(np.fromstring(s[:cut], sep=",", dtype=np.float32))
            idx.append(i)
            if len(rows) >= chunk:
                yield CsvBlock(
                    np.stack(rows).reshape(-1, N_CHANNELS, WINDOW_SAMPLES),
                    np.array(labs, dtype=np.int8), hsh, np.array(idx),
                )
                rows, labs, hsh, idx = [], [], [], []
    if rows:
        yield CsvBlock(
            np.stack(rows).reshape(-1, N_CHANNELS, WINDOW_SAMPLES),
            np.array(labs, dtype=np.int8), hsh, np.array(idx),
        )


def pseudo_records(labels: np.ndarray) -> np.ndarray:
    """Group id per row, approximating a record boundary.

    The rows are temporally ordered, not shuffled -- 141 label runs observed
    against 2505 +- 24 under shuffling, which is 97 sigma. So contiguous runs
    are real seizures and the interictal between them is real recording time.

    A boundary is cut at the midpoint of every interictal run, yielding one
    group per seizure. That is the closest available stand-in for a record, and
    it is explicitly *not* a subject: multiple groups certainly come from the
    same patient, so grouping on it still leaks (R3).
    """
    n = len(labels)
    idx = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, idx]
    lens = np.diff(np.r_[0, idx, n])
    vals = labels[starts]
    cuts = [s + l // 2 for s, l, v in zip(starts, lens, vals)
            if v == 0 and s > 0 and s + l < n]
    grp = np.zeros(n, dtype=np.int32)
    for c in cuts:
        grp[c:] += 1
    return grp


def duplicate_mask(hashes: list[bytes]) -> np.ndarray:
    """True where a row repeats an earlier row (keep-first semantics).

    Measured on this file: 877 repeats across 463 groups of size 2-4, 86% of
    them seizure rows, at a modal stride of 3,154 -- i.e. minority-class
    oversampling by row duplication. Removing them alone moved HGB PR-AUC from
    0.992 to 0.899, which is why R6 exists.
    """
    seen: set[bytes] = set()
    out = np.zeros(len(hashes), dtype=bool)
    for i, h in enumerate(hashes):
        if h in seen:
            out[i] = True
        seen.add(h)
    return out

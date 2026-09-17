"""The two feature views (PRD §8.3).

per_channel  n_channels x 14   higher capacity, montage-fragile, more
                              subject-specific
aggregated   14 x 4            channel-order invariant, montage-robust,
                              expected to transfer better across subjects

If `aggregated` beats `per_channel` under LOSO while losing under a random
split, that is the most interesting result the project can produce, because it
says the extra capacity of per-channel features is being spent on subject
identity rather than on seizure morphology.

R16: per-channel columns are named {channel_label}__{feature}. Never an index.
"""

from __future__ import annotations

import numpy as np

STATS = ("mean", "max", "std", "median")


def per_channel_names(channels: list[str], feats: list[str]) -> list[str]:
    return [f"{c}__{f}" for c in channels for f in feats]


def aggregated_names(feats: list[str]) -> list[str]:
    return [f"{f}__{s}" for f in feats for s in STATS]


def per_channel(fmat: np.ndarray) -> np.ndarray:
    """(n_windows, n_channels, n_feats) -> (n_windows, n_channels*n_feats).

    Row-major over channels so the order matches per_channel_names.
    """
    n = fmat.shape[0]
    return fmat.reshape(n, -1)


def aggregated(fmat: np.ndarray) -> np.ndarray:
    """(n_windows, n_channels, n_feats) -> (n_windows, n_feats*4).

    Aggregation is across channels, which is what makes the view invariant to
    channel order and tolerant of a montage that differs between subjects.
    """
    stats = [
        fmat.mean(axis=1),
        fmat.max(axis=1),
        fmat.std(axis=1),
        np.median(fmat, axis=1),
    ]
    # Interleave so column order is feature-major: f0__mean, f0__max, ...
    out = np.stack(stats, axis=-1)  # (n_windows, n_feats, 4)
    return out.reshape(out.shape[0], -1).astype(np.float32)

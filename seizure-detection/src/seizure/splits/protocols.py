"""Fold assignment. This module is where leakage is either prevented or created.

Three protocols:

  loso            23 folds, one per subject_id. The honest protocol.
  random_window   80/20 over shuffled windows, subject ignored. This exists to
                  be wrong, so the inflation can be quantified (PRD §9.2).
  grouped_pseudo  group by contiguous pseudo-record, for a source with no
                  subject labels (R3).

R6 is enforced here rather than trusted: assign() refuses a frame containing
duplicate feature rows, because duplicate copies land in different groups and
group-aware splitting does not protect against them.

R7: nothing in this module looks at feature VALUES. It partitions on identity
columns only, so it cannot leak by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..signal.windows import GUARD


@dataclass(frozen=True)
class Fold:
    fold_idx: int
    test_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    train_subjects: tuple[str, ...]

    def role_of(self, subject: str) -> str:
        if subject in self.test_subjects:
            return "test"
        if subject in self.val_subjects:
            return "val"
        return "train"


class DuplicateRowsError(ValueError):
    """R6: raised when a frame still contains exact duplicate feature rows."""


def assert_deduplicated(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """R6 enforcement. Cheap hash check over the feature block."""
    if not feature_cols:
        return
    n_dup = int(df.duplicated(subset=feature_cols).sum())
    if n_dup:
        raise DuplicateRowsError(
            f"{n_dup} duplicate feature rows present; deduplicate before splitting "
            "(R6 -- grouping does not substitute for deduplication)"
        )


def drop_guard(df: pd.DataFrame) -> pd.DataFrame:
    """Label -1 rows are excluded from training and evaluation (PRD §6.2)."""
    return df.loc[df["label"] != GUARD].reset_index(drop=True)


def loso_folds(subjects: list[str], n_val: int, seed: int) -> list[Fold]:
    """One fold per subject. Validation subjects are drawn deterministically
    from the remaining subjects so the same fold is reproducible across runs."""
    subs = sorted(set(subjects))
    folds = []
    for i, test in enumerate(subs):
        pool = [s for s in subs if s != test]
        rng = np.random.default_rng([seed, i])  # per-fold stream, not shared
        k = min(n_val, max(0, len(pool) - 1))
        val = tuple(sorted(rng.choice(pool, size=k, replace=False).tolist())) if k else ()
        train = tuple(s for s in pool if s not in val)
        folds.append(Fold(i, (test,), val, train))
    return folds


def assign(
    df: pd.DataFrame,
    protocol: str,
    n_val: int = 4,
    seed: int = 42,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[Fold]]:
    """Return (fold_assignments, folds).

    fold_assignments has one row per (fold_idx, group_key, role).
    """
    if feature_cols:
        assert_deduplicated(df, feature_cols)

    if protocol == "loso":
        subs = sorted(df["subject_id"].unique())
        folds = loso_folds(subs, n_val, seed)
        rows = [
            {"fold_idx": f.fold_idx, "subject_id": s, "role": f.role_of(s)}
            for f in folds
            for s in subs
        ]
        return pd.DataFrame(rows), folds

    if protocol == "random_window":
        # Deliberately wrong. Subject is ignored entirely; a single 80/20 split.
        rng = np.random.default_rng(seed)
        n = len(df)
        idx = rng.permutation(n)
        cut = int(0.8 * n)
        role = np.empty(n, dtype=object)
        role[idx[:cut]] = "train"
        role[idx[cut:]] = "test"
        out = pd.DataFrame({"fold_idx": 0, "row": np.arange(n), "role": role})
        return out, [Fold(0, ("<random>",), (), ())]

    if protocol == "grouped_pseudo":
        # For a source with no subject labels (R3). Groups must already exist
        # as a 'pseudo_record' column.
        if "pseudo_record" not in df.columns:
            raise ValueError("grouped_pseudo requires a 'pseudo_record' column")
        groups = sorted(df["pseudo_record"].unique())
        folds = loso_folds([str(g) for g in groups], n_val, seed)
        rows = [
            {"fold_idx": f.fold_idx, "pseudo_record": g, "role": f.role_of(str(g))}
            for f in folds
            for g in groups
        ]
        return pd.DataFrame(rows), folds

    raise ValueError(f"unknown protocol: {protocol}")


def downsample_negatives(
    df: pd.DataFrame, ratio: int | None, seed: int = 0
) -> pd.DataFrame:
    """Training folds only. PRD §10.2: test and validation are never downsampled.

    Callers must pass only training rows. The function does not know the role,
    so the discipline lives at the call site and is asserted in tests.
    """
    if ratio is None:
        return df
    pos = df.loc[df["label"] == 1]
    neg = df.loc[df["label"] == 0]
    keep = min(len(neg), ratio * max(1, len(pos)))
    neg_s = neg.sample(n=keep, random_state=seed) if keep < len(neg) else neg
    return (
        pd.concat([pos, neg_s])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

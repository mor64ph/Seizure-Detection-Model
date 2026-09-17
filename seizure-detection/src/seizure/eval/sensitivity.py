"""PRD §6.2: how the ictal threshold changes the positive class, and the model.

The labelling rule marks a window ictal when at least ``min_overlap_frac`` of
it falls inside an annotated seizure. The default is 0.5, which is a judgement
call, so the PRD asks for the positive-class count and headline recall at
0.1 / 0.25 / 0.5 / 0.75 -- a one-line finding, not a research project.

This is cheap to answer honestly because ``overlap_frac`` is stored on every
window at extraction time. Relabelling is therefore a column operation, with no
re-extraction and no second pass over 45.76 GB.

The guard band needs care, and a first attempt got it wrong. Guard covers two
different situations: a window with partial overlap that failed the ictal
threshold, and a window near a seizure boundary with no overlap at all. Holding
all guard rows fixed made the sweep degenerate -- 0.1, 0.25 and 0.5 returned
*identical* positive counts, because every window with overlap between 0.1 and
0.5 had already been quarantined as guard at extraction and could never be
recovered by lowering the threshold.

The correct counterfactual is: a guard window whose ``overlap_frac`` clears the
new threshold would have been labelled ictal under that configuration, so it
becomes ictal here. A guard window that still fails the threshold stays
excluded, because its guard status then comes from boundary proximity, which no
overlap threshold changes.

Remaining honest limitation: ``guard_sec`` itself is held at its configured
value. Re-deriving the guard distance per threshold would need the seizure
intervals joined back to every window, which is a re-extraction. That is a
separate knob from the one PRD §6.2 asks about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..signal.windows import GUARD, ICTAL, INTERICTAL

FRACS = (0.1, 0.25, 0.5, 0.75)


def relabel(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Reassign label from the stored overlap_frac at a new threshold.

    A guard row that clears the threshold becomes ictal -- under that config it
    would have been ictal rather than quarantined. A guard row that does not
    clear it stays guard, since its exclusion is then boundary proximity.
    """
    out = df.copy()
    ov = out["overlap_frac"].to_numpy()
    was_guard = (out["label"] == GUARD).to_numpy()
    clears = ov >= frac
    label = np.where(clears, ICTAL, INTERICTAL)
    # guard AND still failing -> remains excluded
    out["label"] = np.where(was_guard & ~clears, GUARD, label)
    return out


def counts(df: pd.DataFrame, fracs=FRACS) -> list[dict]:
    """Positive-class size at each threshold, with the guard accounted for."""
    rows = []
    for f in fracs:
        r = relabel(df, f)
        usable = r.loc[r["label"] != GUARD]
        pos = int((usable["label"] == ICTAL).sum())
        rows.append({
            "min_overlap_frac": f,
            "n_positive": pos,
            "n_windows": int(len(usable)),
            "n_guard": int((r["label"] == GUARD).sum()),
            "positive_rate": pos / len(usable) if len(usable) else float("nan"),
        })
    return rows


def sweep(
    df: pd.DataFrame,
    cfg: Config,
    truth: dict,
    model: str = "logreg",
    fracs=FRACS,
) -> list[dict]:
    """Counts plus a LOSO run per threshold, so 'headline recall' is measured.

    Deliberately one cheap model on one view. The question is how the labelling
    threshold moves the answer, not which model wins -- running the full grid
    four times over would cost hours and answer a different question.
    """
    from .. import train as T

    single = cfg.model_copy(update={
        "training": cfg.training.model_copy(update={"models": [model], "search": False})
    })
    out = []
    base = {r["min_overlap_frac"]: r for r in counts(df, fracs)}
    for f in fracs:
        _, detail = T.run_loso(relabel(df, f), single, truth, None)
        rows = [d for d in detail if "pr_auc" in d]
        det = float(np.nansum([d.get("n_detected", 0) for d in rows]))
        tot = float(np.nansum([d.get("n_true_events", 0) for d in rows]))
        out.append({
            **base[f],
            "model": model,
            "folds": len(rows),
            "pr_auc": float(np.nanmean([d["pr_auc"] for d in rows])) if rows else float("nan"),
            "recall": float(np.nanmean([d["recall"] for d in rows])) if rows else float("nan"),
            "precision": float(np.nanmean([d["precision"] for d in rows])) if rows else float("nan"),
            "event_sensitivity": (det / tot) if tot else float("nan"),
        })
    return out

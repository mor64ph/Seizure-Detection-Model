"""R25: the three-protocol leakage study on ``csv_scaled``, as code.

This exists because the headline number in RULES Appendix A -- HGB PR-AUC
0.992 with duplicates against 0.899 without -- was originally measured in an
ad-hoc session. A cited number that no committed code can regenerate is a
liability, so the study is reproducible here.

Three protocols, deliberately ordered worst to best:

  random_with_duplicates   shuffle rows, 80/20, keep the 877 duplicate rows
  random_deduplicated      shuffle rows, 80/20, drop repeats (keep first)
  grouped_deduplicated     group by pseudo-record, drop repeats

The comparison that matters is the *third against the second* holding
deduplication fixed, and the *second against the first* holding the split
fixed. That decomposition is what shows duplication dominates split protocol --
and that grouping alone does not save you, because duplicate copies land in
different groups.

R21 is honoured here as on the raw data: the 60/120 Hz notch is applied before
featurising. The file arrives un-notched (measured: 60 Hz power ~1000x its
neighbours), so skipping it would let gamma features carry line noise.

R4: nothing here claims scaler discipline. The file was globally pre-scaled
upstream of us.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..features import core, views
from ..signal import filters
from . import loader

PROTOCOLS = (
    "random_with_duplicates",
    "grouped_with_duplicates",   # the protocol that proves grouping is not enough
    "random_deduplicated",
    "grouped_deduplicated",
)


def build_features(path: Path, cfg: Config, notch: bool = True) -> pd.DataFrame:
    """One pass over the CSV -> an aggregated feature table.

    Aggregated view only. The per-channel view would be 18 x 14 = 252 columns
    keyed by channel *label*, and this file's channels have no labels (R16
    cannot be satisfied), so naming them would invent provenance the data does
    not have.
    """
    feats = core.feature_names(cfg.features.bands)
    frames: list[pd.DataFrame] = []
    all_hashes: list[bytes] = []
    all_labels: list[int] = []
    all_idx: list[int] = []

    for block in loader.iter_blocks(path):
        x = block.data.astype(np.float64)
        if notch:
            # (n, 18, 2048) -> filter along the sample axis for the whole block
            x = filters.notch(x, loader.SAMPLE_RATE, cfg.signal.notch_hz)
            x = filters.bandpass(x, loader.SAMPLE_RATE, *cfg.signal.bandpass)
        fm = core.extract(
            x.astype(np.float32), loader.SAMPLE_RATE, cfg.features.bands,
            cfg.features.welch_nperseg, cfg.features.welch_noverlap,
        )
        frames.append(pd.DataFrame(views.aggregated(fm),
                                   columns=views.aggregated_names(feats)))
        all_hashes += block.hashes
        all_labels += block.labels.tolist()
        all_idx += block.row_index.tolist()

    df = pd.concat(frames, ignore_index=True)
    labels = np.array(all_labels, dtype=int)
    df.insert(0, "row_index", all_idx)
    df.insert(1, "label", labels)
    df.insert(2, "pseudo_record", loader.pseudo_records(labels))
    df.insert(3, "is_repeat", loader.duplicate_mask(all_hashes))
    df["source"] = "csv_scaled"            # R1
    df["config_hash"] = cfg.extraction_hash  # R19
    # There is no subject. Say so in the schema rather than inventing one (R3).
    df["subject_id"] = pd.NA
    return df


def _eval(X, y, groups, protocol: str, models: list[str], seed: int) -> list[dict]:
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

    from ..eval import metrics as M
    from ..models import zoo

    out = []
    for name in models:
        est = zoo.build(name, seed)
        if protocol.startswith("grouped"):
            cv, g = GroupKFold(n_splits=5), groups
        else:
            cv, g = KFold(5, shuffle=True, random_state=seed), None
        s = cross_val_predict(est, X, y, cv=cv, groups=g, method="predict_proba")
        s = s[:, 1] if s.ndim == 2 and s.shape[1] > 1 else np.zeros(len(y))
        thr, _ = zoo.pick_threshold(y, s, "f2")
        wm = M.window_metrics(y, s, thr)
        out.append({"protocol": protocol, "model": name,
                    "n_rows": int(len(y)), "n_pos": int(y.sum()),
                    "pos_rate": float(y.mean()), **wm})
    return out


def run(df: pd.DataFrame, cfg: Config, models: list[str] | None = None) -> list[dict]:
    """Evaluate all three protocols on a prebuilt feature table."""
    models = models or ["dummy", "logreg", "rf", "hgb"]
    key = {"row_index", "label", "pseudo_record", "is_repeat", "source",
           "config_hash", "subject_id"}
    cols = [c for c in df.columns if c not in key]
    rows: list[dict] = []

    for protocol in PROTOCOLS:
        sub = df if protocol.endswith("with_duplicates") else df.loc[~df.is_repeat]
        X = sub[cols].to_numpy(dtype=np.float32)
        y = sub["label"].to_numpy()
        g = sub["pseudo_record"].to_numpy()
        rows += _eval(X, y, g, protocol, models, cfg.splits.seed)
    return rows


def decompose(rows: list[dict]) -> list[dict]:
    """Attribute the inflation to duplication versus split protocol."""
    by = {(r["protocol"], r["model"]): r for r in rows}
    models = sorted({r["model"] for r in rows})
    out = []
    for m in models:
        a = by.get(("random_with_duplicates", m))
        gd = by.get(("grouped_with_duplicates", m))
        b = by.get(("random_deduplicated", m))
        c = by.get(("grouped_deduplicated", m))
        if not (a and gd and b and c):
            continue
        out.append({
            "model": m,
            "pr_dup_random": a["pr_auc"],
            "pr_dup_grouped": gd["pr_auc"],
            "pr_dedup_random": b["pr_auc"],
            "pr_dedup_grouped": c["pr_auc"],
            # Holding the split fixed, what does removing duplicates cost?
            "dedup_effect_random": a["pr_auc"] - b["pr_auc"],
            "dedup_effect_grouped": gd["pr_auc"] - c["pr_auc"],
            # Holding duplication fixed, what does grouping buy?
            "grouping_effect_with_dups": a["pr_auc"] - gd["pr_auc"],
            "grouping_effect_deduped": b["pr_auc"] - c["pr_auc"],
        })
    return out


def write(artifacts: Path, rows: list[dict], dec: list[dict]) -> Path:
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    p = artifacts / "csv_leakage_study.json"
    p.write_text(json.dumps({"protocols": rows, "decomposition": dec}, indent=2),
                 encoding="utf-8")
    return p

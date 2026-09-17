"""Per-record extraction: read -> filter -> window -> featurise.

Pure Python. No Spark imports anywhere in this module or its dependencies
(R20) -- that boundary is what makes the whole feature path unit-testable and
portable to serverless unchanged.

Order is fixed by R22: notch and bandpass run on the whole record, before
windowing. Filtering after windowing leaves edge transients in every window
and is what PRD §14 blames for gamma dominance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..labels import disputed
from ..labels.registry import CASE_TO_SUBJECT, case_of_record
from ..signal import filters, io, windows
from . import core, views


@dataclass
class ExtractResult:
    record_id: str
    per_channel: pd.DataFrame | None
    aggregated: pd.DataFrame | None
    n_windows: int
    label_counts: dict[int, int]
    duration_sec: float
    channels: list[str]
    dropped: dict[str, list[str]] = field(default_factory=dict)
    skipped_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None


def key_columns(record_id: str, rows: list[dict], cfg: Config,
                allow_unknown: bool = False) -> pd.DataFrame:
    case = case_of_record(record_id, strict=not allow_unknown)
    df = pd.DataFrame(rows).drop(columns=["_s0", "_s1"])
    df.insert(1, "case_id", case if case else pd.NA)
    df.insert(2, "subject_id", CASE_TO_SUBJECT.get(case) if case else pd.NA)
    df["config_hash"] = cfg.extraction_hash  # R19
    df["source"] = cfg.data.source  # R1
    return df


def extract_record(
    path: Path,
    ictal: list[tuple[int, int]],
    cfg: Config,
    allow_unknown_record: bool = False,
) -> ExtractResult:
    """Extract both feature views for one .edf file.

    ``allow_unknown_record`` is for scoring a recording from outside the study,
    where the filename carries no case or subject. Without it the id lookup
    raises, which made every arbitrary upload fail before reaching the model.
    """
    record_id = Path(path).stem
    canonical = list(cfg.signal.canonical_channels)

    try:
        rec = io.read(path, canonical, cfg.data.on_missing_channel)
    except io.SkipFile as e:
        return ExtractResult(
            record_id, None, None, 0, {}, 0.0, [], {}, f"missing_channels:{e.missing}"
        )

    if rec.data.shape[0] == 0:
        return ExtractResult(record_id, None, None, 0, {}, rec.duration_sec, [], rec.dropped,
                             "no_usable_channels")

    fs = rec.sample_rate
    if abs(fs - cfg.signal.sample_rate) > 1e-6:
        return ExtractResult(record_id, None, None, 0, {}, rec.duration_sec, rec.channels,
                             rec.dropped, f"unexpected_sample_rate:{fs}")

    # R21/R22 -- notch then bandpass, on the whole record.
    clean = filters.preprocess(rec.data, fs, cfg.signal.notch_hz, cfg.signal.bandpass)

    spec = windows.WindowSpec(
        length_sec=cfg.windowing.length_sec,
        stride_sec=cfg.windowing.stride_sec,
        min_overlap_frac=cfg.windowing.min_overlap_frac,
        guard_sec=cfg.windowing.guard_sec,
        sample_rate=int(fs),
    )
    ictal_resolved = disputed.resolve(record_id, ictal)
    rows = windows.build_table(
        record_id, rec.n_samples, ictal_resolved, spec, disputed.guard_spans(record_id)
    )
    if not rows:
        return ExtractResult(record_id, None, None, 0, {}, rec.duration_sec, rec.channels,
                             rec.dropped, "record_shorter_than_window")

    # (n_windows, n_channels, window_samples) via a strided view -- no copy.
    starts = np.array([r["_s0"] for r in rows])
    L = spec.length_samples
    seg = np.stack([clean[:, s : s + L] for s in starts])

    fmat = core.extract(
        seg, fs, cfg.features.bands, cfg.features.welch_nperseg, cfg.features.welch_noverlap
    )  # (n_windows, n_channels, 14)

    feats = core.feature_names(cfg.features.bands)
    keys = key_columns(record_id, rows, cfg, allow_unknown_record)

    pc = agg = None
    if "per_channel" in cfg.features.views:
        pc = pd.concat(
            [keys, pd.DataFrame(views.per_channel(fmat),
                                columns=views.per_channel_names(rec.channels, feats))],
            axis=1,
        )
    if "aggregated" in cfg.features.views:
        agg = pd.concat(
            [keys, pd.DataFrame(views.aggregated(fmat),
                                columns=views.aggregated_names(feats))],
            axis=1,
        )

    counts = keys["label"].value_counts().to_dict()
    return ExtractResult(
        record_id, pc, agg, len(rows), {int(k): int(v) for k, v in counts.items()},
        rec.duration_sec, rec.channels, rec.dropped,
    )

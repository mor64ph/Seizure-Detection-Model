"""EDF reading via mne, with the channel projection applied by label.

PRD §7.1 specifies mne.io.read_raw_edf(preload=True, verbose=False). preload
matters: without it every window slice re-reads from disk, which PRD §14 names
as a cause of unexpectedly slow extraction.

R16: channels are selected by label and returned in canonical order, so the
feature matrix has identical semantics for every subject. Nothing here indexes
by position.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ingest import edf_header as eh
from . import channels as ch

_MNE_CONFIGURED = False


def _configure_mne():
    """PRD §14: set the log level once at startup or mne floods the log."""
    global _MNE_CONFIGURED
    import mne

    if not _MNE_CONFIGURED:
        mne.set_log_level("ERROR")
        _MNE_CONFIGURED = True
    return mne


@dataclass
class Record:
    record_id: str
    data: np.ndarray  # (n_canonical_channels, n_samples), canonical order
    channels: list[str]
    sample_rate: float
    duration_sec: float
    dropped: dict[str, list[str]]
    missing: list[str]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]


class SkipFile(Exception):
    """Raised when on_missing_channel='skip_file' and a canonical channel is
    absent. Callers must count these, not swallow them (PRD §7.2)."""

    def __init__(self, record_id: str, missing: list[str]):
        super().__init__(f"{record_id}: missing canonical channels {missing}")
        self.record_id = record_id
        self.missing = missing


def read(
    path: Path,
    canonical: list[str],
    on_missing: str = "skip_file",
) -> Record:
    """Read one .edf and project onto the canonical channel set."""
    mne = _configure_mne()
    path = Path(path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    rate = float(raw.info["sfreq"])

    # Labels come from the EDF header, not from mne.ch_names.
    #
    # The 23-channel montage lists T8-P8 twice, and mne resolves that by
    # appending running numbers ("T8-P8-0", "T8-P8-1"), which no longer parse
    # as bipolar derivations -- so every file would be skipped for a "missing"
    # T8-P8. The header is the authoritative source and mne preserves channel
    # order, so position i in get_data() is position i in the header.
    with open(path, "rb") as f:
        probe = f.read(eh.PROBE_BYTES)
        f.seek(0)
        head = eh.parse(f.read(eh.header_size_from_probe(probe)))
    raw_labels = head.labels
    if len(raw_labels) != len(raw.ch_names):
        raise ValueError(
            f"{path.stem}: header lists {len(raw_labels)} signals, "
            f"mne reports {len(raw.ch_names)}"
        )
    cls = ch.classify(raw_labels)

    # Map canonical label -> the first raw channel bearing it. Duplicate labels
    # keep their first occurrence (PRD §2).
    by_label: dict[str, int] = {}
    for i, lab in enumerate(raw_labels):
        norm = ch.normalise(lab)
        if ch.is_scalp_eeg(norm) and norm not in by_label:
            by_label[norm] = i

    selected, missing = ch.project(list(by_label), canonical, on_missing)
    if on_missing == "skip_file" and missing:
        raise SkipFile(path.stem, missing)

    arr = raw.get_data()
    if selected:
        data = np.stack([arr[by_label[lab]] for lab in selected])
    else:
        data = np.empty((0, arr.shape[1]))

    if on_missing == "fill_nan" and missing:
        # Keep the column layout stable across files so the feature table has
        # one schema. HistGradientBoosting handles NaN natively (PRD §10.1).
        full = np.full((len(canonical), arr.shape[1]), np.nan)
        pos = {lab: k for k, lab in enumerate(canonical)}
        for lab in selected:
            full[pos[lab]] = data[selected.index(lab)]
        data, selected = full, list(canonical)

    return Record(
        record_id=path.stem,
        data=np.ascontiguousarray(data, dtype=np.float64),
        channels=list(selected),
        sample_rate=rate,
        duration_sec=arr.shape[1] / rate,
        dropped={
            "duplicate": cls["duplicate"],
            "non_eeg": cls["non_eeg"],
            "unrecognised": cls["unrecognised"],
        },
        missing=missing,
    )

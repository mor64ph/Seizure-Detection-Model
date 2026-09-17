"""Streaming per-file ingest: download, verify, extract, delete raw, checkpoint.

Disk is 26 GB free against a 45.76 GB dataset, so raw bytes cannot all be
resident. R14 makes the unit of work a single .edf (largest 0.18 GB) rather
than a case (largest 6.86 GB), which keeps peak transient disk near 0.2 GB.

R11 is the load-bearing rule and it is enforced mechanically, not by care:
``delete_raw`` takes the manifest row and refuses unless every
capture-before-delete field is present. Deletion is never a bare unlink in
this codebase.

R13 keeps chb01, chb12 and chb17 permanently, because they exercise the
montage and naming landmines and let signal code be iterated offline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..config import Config
from ..features import extract as fx
from . import fetch

KEEP_CASES = frozenset({"chb01", "chb12", "chb17"})  # R13

REQUIRED_BEFORE_DELETE = (
    "checksum_verified",
    "n_samples",
    "duration_sec",
    "channel_labels",
    "channel_stats",
    "features_path",
    "n_windows",
)


class CaptureIncomplete(RuntimeError):
    """R11: raw bytes may not be deleted until the manifest row is complete."""


@dataclass
class RecordState:
    record_id: str
    case_id: str
    rel: str
    state: str = "pending"  # pending|downloaded|extracted|raw_deleted|skipped|failed
    s3_size: int | None = None
    checksum_verified: bool | None = None
    n_samples: int | None = None
    duration_sec: float | None = None
    channel_labels: list[str] | None = None
    channel_stats: dict | None = None
    features_path: str | None = None
    n_windows: int | None = None
    label_counts: dict | None = None
    dropped: dict | None = None
    reason: str | None = None
    error: str | None = None


class Ledger:
    """Resume manifest keyed on record_id (R14). Re-running a completed record
    is a no-op that logs and returns."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: dict[str, RecordState] = {}
        if self.path.exists():
            for r in json.loads(self.path.read_text(encoding="utf-8")):
                self.rows[r["record_id"]] = RecordState(**r)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(v) for v in self.rows.values()], indent=1), encoding="utf-8"
        )

    def get(self, record_id: str) -> RecordState | None:
        return self.rows.get(record_id)

    def put(self, st: RecordState) -> None:
        self.rows[st.record_id] = st
        self.save()

    def done(self, record_id: str) -> bool:
        st = self.rows.get(record_id)
        return st is not None and st.state in {"extracted", "raw_deleted", "skipped"}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(v) for v in self.rows.values()])


def channel_stats(data, labels: list[str]) -> dict:
    """R12: per-channel summary statistics, captured while the file is resident.

    Stored per case and pooled per subject at use time, because chb01 and
    chb21 are one subject and per-case baselines would give them two.
    """
    import numpy as np

    return {
        lab: {
            "mean": float(data[i].mean()),
            "std": float(data[i].std()),
            "p05": float(np.percentile(data[i], 5)),
            "p50": float(np.percentile(data[i], 50)),
            "p95": float(np.percentile(data[i], 95)),
        }
        for i, lab in enumerate(labels)
    }


def delete_raw(path: Path, st: RecordState, force_keep: bool = False) -> bool:
    """R11 gate. Returns True if the raw file was deleted."""
    if force_keep or st.case_id in KEEP_CASES:
        return False
    missing = [f for f in REQUIRED_BEFORE_DELETE if getattr(st, f) in (None, [], {})]
    if missing:
        raise CaptureIncomplete(
            f"{st.record_id}: refusing to delete raw bytes; unrecorded fields {missing} "
            "(R11 -- these are unrecoverable without re-downloading)"
        )
    p = Path(path)
    if p.exists():
        p.unlink()
    return True


def ingest_record(
    rel: str,
    s3_size: int,
    ictal: list[tuple[int, int]],
    cfg: Config,
    out_dir: Path,
    checksums: dict[str, str],
    keep_raw: bool = False,
) -> RecordState:
    """One record, end to end. Raises only on programmer error; data problems
    are recorded in the returned state (R28)."""
    from ..signal import io

    record_id = Path(rel).stem
    st = RecordState(record_id=record_id, case_id=rel.split("/")[0], rel=rel, s3_size=s3_size)
    raw_path = Path(cfg.data.root) / rel

    try:
        fetch.download(rel, raw_path, expect_size=s3_size)
        st.state = "downloaded"

        want = checksums.get(rel)
        st.checksum_verified = (want is not None) and fetch.sha256_file(raw_path) == want
        if want is not None and not st.checksum_verified:
            st.state, st.reason = "failed", "checksum_mismatch"
            return st

        res = fx.extract_record(raw_path, ictal, cfg)
        if res.skipped:
            st.state, st.reason = "skipped", res.skipped_reason
            st.duration_sec = res.duration_sec
            st.dropped = res.dropped
            # A skipped file still had its raw bytes fetched; they are not
            # needed again, but R11's fields were never captured, so keep them
            # unless the caller is explicitly reclaiming space.
            return st

        rec = io.read(raw_path, list(cfg.signal.canonical_channels), cfg.data.on_missing_channel)
        st.n_samples = int(rec.n_samples)
        st.duration_sec = float(rec.duration_sec)
        st.channel_labels = list(rec.channels)
        st.channel_stats = channel_stats(rec.data, rec.channels)
        st.n_windows = res.n_windows
        st.label_counts = res.label_counts
        st.dropped = res.dropped

        out_dir = Path(out_dir)
        for view, frame in (("per_channel", res.per_channel), ("aggregated", res.aggregated)):
            if frame is None:
                continue
            d = out_dir / view / f"case_id={st.case_id}"
            d.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(d / f"{record_id}.parquet", index=False)
        st.features_path = str(out_dir)
        st.state = "extracted"

        if delete_raw(raw_path, st, force_keep=keep_raw):
            st.state = "raw_deleted"
        return st

    except Exception as e:  # noqa: BLE001 -- recorded, not swallowed
        st.state, st.error = "failed", f"{type(e).__name__}: {e}"
        return st

"""Parse chbNN-summary.txt into seizure intervals.

Format notes that drive the implementation (PRD §6.1):

  * Both ``Seizure Start Time:`` and ``Seizure 1 Start Time:`` occur. Single-
    seizure records use the unindexed form.
  * ``File Start Time`` / ``File End Time`` are ignored entirely. Some cases
    contain clock values past 24:00:00, so parsing them as times would raise;
    seizure offsets are relative seconds and wall-clock is never needed.
  * chb17 records are named chb17a_*, chb17b_*, so no chb\\d\\d_\\d\\d regex.
  * Some cases interleave a ``Channels changed`` block and re-list the montage
    mid-file. Those blocks are skipped, not treated as records.

The parser raises on an unrecognised seizure block rather than silently
returning empty -- a quiet zero here is indistinguishable from a real
zero-seizure record, and PRD §14 names that as the cause of a wrong total.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FILE_NAME = re.compile(r"^File\s*Name:\s*(?P<name>\S+)", re.I)
N_SEIZ = re.compile(r"^Number\s*of\s*Seizures\s*in\s*File:\s*(?P<n>\d+)", re.I)
# Group 1 = optional seizure index, group 2 = seconds.
SEIZ_TIME = re.compile(
    r"^Seizure\s*(?P<idx>\d+)?\s*(?P<kind>Start|End)\s*Time:\s*(?P<sec>[\d.]+)\s*(?:seconds)?",
    re.I,
)


@dataclass(frozen=True)
class Seizure:
    record_id: str
    seizure_idx: int
    start_sec: int
    end_sec: int

    @property
    def duration_sec(self) -> int:
        return self.end_sec - self.start_sec


@dataclass
class RecordSummary:
    record_id: str
    n_seizures_declared: int
    seizures: list[Seizure]
    anomalies: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.anomalies is None:
            self.anomalies = []


def _record_id(name: str) -> str:
    return name[:-4] if name.lower().endswith(".edf") else name


def parse(text: str) -> list[RecordSummary]:
    """Parse one chbNN-summary.txt. Returns one entry per File Name block."""
    records: list[RecordSummary] = []
    cur: str | None = None
    declared: int | None = None
    starts: list[tuple[int | None, float]] = []
    ends: list[tuple[int | None, float]] = []

    def flush() -> None:
        nonlocal cur, declared, starts, ends
        if cur is None:
            return
        n = declared if declared is not None else 0
        if len(starts) != len(ends):
            raise ValueError(
                f"{cur}: {len(starts)} start times vs {len(ends)} end times"
            )
        if n != len(starts):
            raise ValueError(
                f"{cur}: declares {n} seizures but {len(starts)} start times parsed"
            )
        seiz: list[Seizure] = []
        notes: list[str] = []
        for i, ((si, s), (ei, e)) in enumerate(zip(starts, ends), start=1):
            if si is not None and ei is not None and si != ei:
                # chb09_08 labels its second seizure's end as "Seizure 1 End
                # Time" -- an upstream typo. Pairing by position rather than by
                # index recovers the right interval; raising here would drop
                # all four of chb09's seizures and miss the 198 total.
                notes.append(
                    f"index mismatch: start labelled {si}, end labelled {ei}; "
                    f"paired positionally as seizure {i}"
                )
            idx = i  # position is authoritative, not the printed index
            if e <= s:
                raise ValueError(f"{cur}: seizure {idx} has end {e} <= start {s}")
            seiz.append(Seizure(cur, int(idx), int(s), int(e)))
        records.append(RecordSummary(cur, n, seiz, notes))
        cur, declared, starts, ends = None, None, [], []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := FILE_NAME.match(line):
            flush()
            cur = _record_id(m.group("name"))
            continue
        if cur is None:
            continue  # header / montage / "Channels changed" preamble
        if m := N_SEIZ.match(line):
            declared = int(m.group("n"))
            continue
        if m := SEIZ_TIME.match(line):
            idx = int(m.group("idx")) if m.group("idx") else None
            sec = float(m.group("sec"))
            (starts if m.group("kind").lower() == "start" else ends).append((idx, sec))
            continue
        if re.match(r"^(File\s*(Start|End)\s*Time|Channel\s|Channels\s)", line, re.I):
            continue
        if re.match(r"^Seizure", line, re.I):
            # A Seizure* line we failed to understand is a format variant we do
            # not handle. Raise (PRD §6.1) rather than undercount.
            raise ValueError(f"{cur}: unrecognised seizure line: {line!r}")

    flush()
    return records


def to_rows(records: list[RecordSummary], case_to_subject: dict[str, str]) -> list[dict]:
    """Flatten to the silver seizure_intervals schema (PRD §5.2)."""
    from .registry import case_of_record

    out = []
    for rec in records:
        for s in rec.seizures:
            case = case_of_record(s.record_id)
            out.append(
                {
                    "case_id": case,
                    "subject_id": case_to_subject[case],
                    "record_id": s.record_id,
                    "seizure_idx": s.seizure_idx,
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "duration_sec": s.duration_sec,
                }
            )
    return out

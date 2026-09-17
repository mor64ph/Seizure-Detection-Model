"""Reader for the WFDB-format ``.edf.seizures`` annotation files.

R29 requires the summary-text parser to be cross-validated against these,
because they are an independent second source for the same intervals. PRD §14
names "parser dropping a format variant" as the likely cause of a wrong seizure
count, and checking a parser against itself proves nothing.

Format, reverse-engineered and verified against chb01_03 (published seizure
2996-3036 s, decoded here to samples 766976 and 777216 at 256 Hz):

Each annotation is a little-endian 16-bit word, ``A = w >> 10``,
``I = w & 0x3FF``, where I advances the sample clock.

    A = 59  SKIP   I is unused; the next 4 bytes carry a 32-bit interval as
                   (signed high word << 16) | unsigned low word, each LE.
    A = 63  AUX    I bytes of text follow, padded to an even total.
    A = 32  VFON   '[' -- seizure onset, at the current clock
    A = 33  VFOFF  ']' -- seizure offset
    A = 22  NOTE   leading metadata marker
    A = 0, I = 0   end of file

Files open with SKIP(-1) followed by a NOTQRS of I=1, which nets the clock to
zero. The leading AUX text is "## time resolution: 256", so the clock is in
samples and seconds require dividing by that rate -- it is read from the file
rather than assumed (R32 in spirit).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

SKIP, AUX, NOTE, VFON, VFOFF, NOTQRS = 59, 63, 22, 32, 33, 0
SUB, CHN, NUM = 61, 62, 60
# Codes whose I field carries a payload rather than a time increment.
_NON_TEMPORAL = frozenset({SUB, CHN, NUM, AUX})
DEFAULT_RESOLUTION_HZ = 256.0
_RES_RE = re.compile(r"time\s+resolution:\s*([\d.]+)", re.I)


@dataclass(frozen=True)
class AnnotInterval:
    start_sample: int
    end_sample: int
    resolution_hz: float

    @property
    def start_sec(self) -> float:
        return self.start_sample / self.resolution_hz

    @property
    def end_sec(self) -> float:
        return self.end_sample / self.resolution_hz


def parse(buf: bytes) -> list[AnnotInterval]:
    """Decode one .edf.seizures file into seizure intervals."""
    t = 0
    i = 0
    n = len(buf)
    resolution = DEFAULT_RESOLUTION_HZ
    open_at: int | None = None
    out: list[AnnotInterval] = []

    while i + 1 < n:
        (w,) = struct.unpack_from("<H", buf, i)
        i += 2
        a, inc = w >> 10, w & 0x3FF

        if a == NOTQRS and inc == 0:
            break  # end of file

        if a == SKIP:
            if i + 4 > n:
                raise ValueError("truncated SKIP payload")
            (high,) = struct.unpack_from("<h", buf, i)
            (low,) = struct.unpack_from("<H", buf, i + 2)
            i += 4
            t += (high << 16) | low
            continue

        if a in _NON_TEMPORAL:
            # For SUB/CHN/NUM/AUX the I field is a payload value or byte
            # length, NOT a time increment. Advancing the clock by it puts
            # every subsequent timestamp out by the length of the leading
            # "## time resolution: 256" string -- 23 samples, which shows up as
            # a spurious 0.09 s on every interval.
            if a == AUX:
                text = buf[i : i + inc].decode("ascii", "replace")
                i += inc + (inc % 2)  # pad to even
                if m := _RES_RE.search(text):
                    resolution = float(m.group(1))
            continue

        t += inc

        if a == VFON:
            if open_at is not None:
                raise ValueError(f"nested seizure onset at sample {t}")
            open_at = t
        elif a == VFOFF:
            if open_at is None:
                raise ValueError(f"seizure offset without onset at sample {t}")
            if t <= open_at:
                raise ValueError(f"seizure offset {t} <= onset {open_at}")
            out.append(AnnotInterval(open_at, t, resolution))
            open_at = None

    if open_at is not None:
        raise ValueError(f"unterminated seizure onset at sample {open_at}")
    return out


def to_seconds(intervals: list[AnnotInterval]) -> list[tuple[int, int]]:
    """Round to whole seconds, matching the summary files' granularity."""
    return [(int(round(v.start_sec)), int(round(v.end_sec))) for v in intervals]


def cross_validate(
    summary: dict[str, list[tuple[int, int]]],
    annot: dict[str, list[tuple[int, int]]],
    tolerance_sec: int = 1,
) -> list[str]:
    """R29. Returns a list of disagreements; empty means the sources agree.

    A tolerance of 1 s absorbs rounding between the two representations (the
    annotations are in samples, the summaries in whole seconds).
    """
    problems: list[str] = []
    for rec in sorted(set(summary) | set(annot)):
        s = sorted(summary.get(rec, []))
        a = sorted(annot.get(rec, []))
        if len(s) != len(a):
            problems.append(
                f"{rec}: summary has {len(s)} seizures, annotations have {len(a)}"
            )
            continue
        for k, ((s0, s1), (a0, a1)) in enumerate(zip(s, a), start=1):
            if abs(s0 - a0) > tolerance_sec or abs(s1 - a1) > tolerance_sec:
                problems.append(
                    f"{rec} seizure {k}: summary {s0}-{s1}s vs annotation {a0}-{a1}s"
                )
    return problems

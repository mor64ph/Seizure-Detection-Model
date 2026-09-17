"""Channel classification and canonicalisation.

R31 requires an allowlist, not a denylist, and the reason is specific: the
distribution contains signals that are not scalp EEG and that PRD §2 does not
mention --

  * ECG in the last 36 files of chb04
  * a vagal nerve stimulus channel in the last 18 files of chb09
  * up to 5 dummy signals in some cases

VNS is a *treatment device* signal. A detector that reads the stimulator rather
than the brain would score well and mean nothing. A denylist fails the moment an
unanticipated label appears; an allowlist fails safe.

Rather than hardcode 18 or 23 labels (R17), a label is accepted iff it parses as
``E1-E2`` where both electrodes are recognised 10-20 positions. That admits every
legitimate bipolar derivation and rejects ECG, VNS, EOG and placeholders without
enumerating them.

R16: canonicalisation is a downstream projection keyed by label, never by
position. Nothing in this module reorders by index.
"""

from __future__ import annotations

import re
from collections import Counter

# 10-20 and 10-10 positions that appear in CHB-MIT, including the older
# T3/T4/T5/T6 naming used by some cases alongside modern T7/T8/P7/P8.
ELECTRODES: frozenset[str] = frozenset(
    """
    FP1 FP2 FPZ F1 F2 F3 F4 F5 F6 F7 F8 FZ
    FC1 FC2 FC3 FC4 FC5 FC6 FCZ FT9 FT10 FT7 FT8
    C1 C2 C3 C4 C5 C6 CZ CP1 CP2 CP3 CP4 CP5 CP6 CPZ
    T1 T2 T3 T4 T5 T6 T7 T8 TP7 TP8
    P1 P2 P3 P4 P5 P6 P7 P8 PZ PO3 PO4 POZ
    O1 O2 OZ A1 A2 M1 M2
    """.split()
)

# Labels seen in the wild that are explicitly not scalp EEG. Kept only so the
# run report can distinguish "known non-EEG" from "unrecognised" (R28); the
# allowlist above is what actually decides.
KNOWN_NON_EEG: frozenset[str] = frozenset(
    {"ECG", "EKG", "VNS", "EMG", "LOC", "ROC", "LUE", "RAE", "-", ".", ""}
)

_BIPOLAR = re.compile(r"^([A-Z0-9]+)\s*-\s*([A-Z0-9]+)$")


def normalise(label: str) -> str:
    """Upper-case, collapse whitespace. Does not alter electrode identity."""
    return re.sub(r"\s+", "", (label or "").strip().upper())


def is_scalp_eeg(label: str) -> bool:
    m = _BIPOLAR.match(normalise(label))
    if not m:
        return False
    a, b = m.group(1), m.group(2)
    return a in ELECTRODES and b in ELECTRODES and a != b


def classify(labels: list[str]) -> dict[str, list[str]]:
    """Split raw header labels into kept / non-EEG / unrecognised / duplicate.

    PRD §2: duplicate labels (T8-P8 appears twice in the 23-channel montage)
    are deduplicated keeping the first occurrence, and the drop is logged.
    """
    seen: set[str] = set()
    kept: list[str] = []
    dup: list[str] = []
    non_eeg: list[str] = []
    unknown: list[str] = []

    for raw in labels:
        lab = normalise(raw)
        if not is_scalp_eeg(lab):
            (non_eeg if lab in KNOWN_NON_EEG else unknown).append(raw)
            continue
        if lab in seen:
            dup.append(raw)
            continue
        seen.add(lab)
        kept.append(lab)

    return {"kept": kept, "duplicate": dup, "non_eeg": non_eeg, "unrecognised": unknown}


def canonical_from_manifest(
    per_file_labels: dict[str, list[str]], min_presence: float = 0.95
) -> list[str]:
    """R30: the canonical set is the labels present in >= min_presence of files.

    Returned in a deterministic order (descending presence, then alphabetical)
    so the frozen list is reproducible across runs.
    """
    if not per_file_labels:
        raise ValueError("empty channel manifest; cannot freeze a canonical set")
    n = len(per_file_labels)
    counts: Counter[str] = Counter()
    for labels in per_file_labels.values():
        counts.update(set(classify(labels)["kept"]))
    return sorted(
        (lab for lab, c in counts.items() if c / n >= min_presence),
        key=lambda lab: (-counts[lab], lab),
    )


def presence_table(per_file_labels: dict[str, list[str]]) -> list[tuple[str, int, float]]:
    """(label, n_files, fraction) descending. Feeds docs/data-notes.md."""
    n = len(per_file_labels)
    counts: Counter[str] = Counter()
    for labels in per_file_labels.values():
        counts.update(set(classify(labels)["kept"]))
    return [(lab, c, c / n) for lab, c in counts.most_common()]


def project(
    available: list[str], canonical: list[str], on_missing: str = "skip_file"
) -> tuple[list[str], list[str]]:
    """Select canonical labels from what a file actually has (R16).

    Returns (selected_labels, missing_labels). Selection is by label, so the
    output is always in canonical order regardless of the file's own ordering
    -- which is the failure PRD §14 warns about.
    """
    have = set(available)
    missing = [c for c in canonical if c not in have]
    if missing and on_missing == "skip_file":
        return [], missing
    return [c for c in canonical if c in have], missing

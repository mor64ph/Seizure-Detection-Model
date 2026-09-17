"""Records where the two independent label sources disagree.

R29's cross-validation leaves exactly one disagreement across 141 records:

    chb24_21  summary: 2804-2872 s   annotation: 2404-2872 s

The end times agree; the starts differ by exactly 400 s. One of the two is a
digit error and there is no third source to break the tie. The annotation file
implies a 468 s seizure, which is atypical for this database; the summary
implies 68 s, which is typical. The summary text is also demonstrably fallible
(chb09_08 mislabels its second seizure's end index), so neither source can be
declared clean.

Policy, rather than a guess: take the narrower interval as ictal and mark the
disputed 400 s as **guard** (label -1), so it trains and scores nothing. This
reuses the mechanism PRD §6.2 already provides for genuinely ambiguous windows
instead of inventing a special case, and it is symmetric -- whichever source is
right, we neither learn from nor are penalised on the contested span.

R28: this is recorded here and in docs/data-notes.md, not papered over.
"""

from __future__ import annotations

# record_id -> list of (start_sec, end_sec) spans to force to label -1
DISPUTED_GUARD: dict[str, list[tuple[int, int]]] = {
    "chb24_21": [(2404, 2804)],
}

# record_id -> the interval we treat as ictal, overriding source disagreement
RESOLVED_ICTAL: dict[str, list[tuple[int, int]]] = {
    "chb24_21": [(2804, 2872)],
}


def guard_spans(record_id: str) -> list[tuple[int, int]]:
    return DISPUTED_GUARD.get(record_id, [])


def resolve(record_id: str, summary: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the intervals to treat as ictal for a record."""
    return RESOLVED_ICTAL.get(record_id, summary)

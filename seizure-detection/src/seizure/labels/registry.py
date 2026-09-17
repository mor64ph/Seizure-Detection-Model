"""Case -> subject registry. R5: subject_id is the only grouping key.

chb21 is the same human as chb01, recorded 1.5 years later. SUBJECT-INFO
independently corroborates this: both are listed female, ages 11 and 13.

chb24 was added to the distribution in December 2010 and is absent from
SUBJECT-INFO, so its demographics are genuinely unavailable. The registry must
tolerate that rather than drop the case (PRD §2).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

N_CASES = 24
N_SUBJECTS = 23

CASE_TO_SUBJECT: dict[str, str] = {f"chb{i:02d}": f"sub{i:02d}" for i in range(1, N_CASES + 1)}
CASE_TO_SUBJECT["chb21"] = "sub01"  # same human as chb01, 1.5 years later

# PRD §2: chb17 files are named chb17a_*, chb17b_*, so the regex must not
# assume chb\d\d_\d\d. Record ids also appear as chb24_01 etc.
RECORD_RE = re.compile(r"^(?P<case>chb\d{2})[a-z]?_(?P<idx>\d+)(?:\+)?$")


def case_of_record(record_id: str, strict: bool = True) -> str | None:
    """'chb17a_03' -> 'chb17'. Raises on anything unrecognised (PRD §6.1).

    ``strict=False`` returns None instead, for **inference on a recording that
    is not part of the study**. It exists only for that: an uploaded file has no
    case and no subject, and inventing one would be worse than admitting it
    (R3). Every training and ingest path leaves strict on, so R5's guarantee
    that each row carries a real subject is untouched.
    """
    m = RECORD_RE.match(record_id)
    if not m:
        if strict:
            raise ValueError(f"unrecognised record id: {record_id!r}")
        return None
    return m.group("case")


def subject_of_record(record_id: str) -> str:
    return CASE_TO_SUBJECT[case_of_record(record_id)]


def parse_subject_info(text: str) -> dict[str, tuple[str | None, float | None]]:
    """Parse SUBJECT-INFO into {case_id: (gender, age)}. Tab separated, with a
    blank line after the header and space-padded ages."""
    out: dict[str, tuple[str | None, float | None]] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 3 or not parts[0].startswith("chb"):
            continue
        gender = parts[1] or None
        try:
            age = float(parts[2])
        except ValueError:
            age = None
        out[parts[0]] = (gender, age)
    return out


def build(subject_info_text: str | None = None) -> pd.DataFrame:
    """Emit the subject_registry table (PRD §5.1)."""
    info = parse_subject_info(subject_info_text) if subject_info_text else {}
    rows = []
    for case, subject in sorted(CASE_TO_SUBJECT.items()):
        gender, age = info.get(case, (None, None))
        rows.append(
            {"case_id": case, "subject_id": subject, "gender": gender, "age": age}
        )
    df = pd.DataFrame(rows)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    """R5 invariants. Called on every build; also exercised directly by tests."""
    assert len(df) == N_CASES, f"expected {N_CASES} cases, got {len(df)}"
    n = df["subject_id"].nunique()
    assert n == N_SUBJECTS, f"expected {N_SUBJECTS} subjects, got {n}"
    shared = set(df.loc[df["case_id"].isin(["chb01", "chb21"]), "subject_id"])
    assert shared == {"sub01"}, f"chb01/chb21 must share sub01, got {shared}"
    assert df.loc[df["case_id"] == "chb24", "gender"].isna().all(), (
        "chb24 is absent from SUBJECT-INFO; its demographics must stay null"
    )


def load(root: Path) -> pd.DataFrame:
    p = Path(root) / "SUBJECT-INFO"
    return build(p.read_text(encoding="utf-8") if p.exists() else None)

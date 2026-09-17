"""Build the seizure_intervals table from both sources, and cross-validate them.

R29: the summary-text parser is checked against the 141 .seizures binary
annotation files, and the total is asserted to be 198. Checking a parser
against itself proves nothing, and PRD §14 names a dropped format variant as
the likely cause of a wrong count.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..ingest import metadata
from . import registry, seizures_annot as sa, summary_parser as sp

EXPECTED_TOTAL = 198
EXPECTED_ORIGINAL_23 = 182
EXPECTED_RECORDS = 141


class LabelCountError(AssertionError):
    pass


def from_summaries(root: Path) -> tuple[dict[str, list[tuple[int, int]]], list[dict]]:
    """record_id -> [(start, end)], plus recorded parser anomalies."""
    out: dict[str, list[tuple[int, int]]] = {}
    anomalies: list[dict] = []
    for p in metadata.summary_paths(root):
        recs = sp.parse(p.read_text(encoding="utf-8", errors="replace"))
        for r in recs:
            if r.seizures:
                out[r.record_id] = [(s.start_sec, s.end_sec) for s in r.seizures]
            for a in r.anomalies:
                anomalies.append({"record_id": r.record_id, "anomaly": a})
    return out, anomalies


def from_annotations(root: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for p in metadata.seizure_annot_paths(root):
        rid = p.name.replace(".edf.seizures", "")
        out[rid] = sa.to_seconds(sa.parse(p.read_bytes()))
    return out


def build(root: Path, strict: bool = True) -> tuple[pd.DataFrame, dict]:
    """Returns (seizure_intervals, report)."""
    summ, anomalies = from_summaries(root)
    annot = from_annotations(root)
    disagreements = sa.cross_validate(summ, annot)

    n_summ = sum(len(v) for v in summ.values())
    n_annot = sum(len(v) for v in annot.values())

    report = {
        "n_from_summary": n_summ,
        "n_from_annotations": n_annot,
        "n_records_summary": len(summ),
        "n_records_annotations": len(annot),
        "parser_anomalies": anomalies,
        "source_disagreements": disagreements,
    }

    if strict:
        if n_annot != EXPECTED_TOTAL:
            raise LabelCountError(
                f"annotations give {n_annot} seizures, expected {EXPECTED_TOTAL}"
            )
        if n_summ != EXPECTED_TOTAL:
            raise LabelCountError(
                f"summary text gives {n_summ} seizures, expected {EXPECTED_TOTAL}"
            )
        if len(annot) != EXPECTED_RECORDS:
            raise LabelCountError(
                f"{len(annot)} annotated records, expected {EXPECTED_RECORDS}"
            )

    rows = []
    for rid in sorted(summ):
        case = registry.case_of_record(rid)
        for i, (a, b) in enumerate(sorted(summ[rid]), start=1):
            rows.append({
                "case_id": case,
                "subject_id": registry.CASE_TO_SUBJECT[case],
                "record_id": rid,
                "seizure_idx": i,
                "start_sec": a,
                "end_sec": b,
                "duration_sec": b - a,
            })
    df = pd.DataFrame(rows)
    assert (df["duration_sec"] > 0).all(), "non-positive seizure duration"

    by_case = df.groupby("case_id").size()
    orig = int(by_case.drop(index="chb24", errors="ignore").sum())
    report["n_original_23"] = orig
    if strict and orig != EXPECTED_ORIGINAL_23:
        raise LabelCountError(
            f"chb01-chb23 give {orig} seizures, expected {EXPECTED_ORIGINAL_23}"
        )
    return df, report

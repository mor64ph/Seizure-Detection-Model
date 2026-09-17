"""Fetch and cache the entire non-.edf payload.

R13: it is 2.19 MB for all 141 .seizures, 24 summaries and 6 root files, so
there is no reason to be selective. Fetch once, keep forever. These are the
only inputs to the label layer, and re-deriving them after a raw-file delete
would mean re-downloading.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import fetch

ROOT_FILES = [
    "RECORDS",
    "RECORDS-WITH-SEIZURES",
    "SUBJECT-INFO",
    "ANNOTATORS",
    "SHA256SUMS.txt",
]


def sync(root: Path, workers: int = 12, include_pdf: bool = False) -> dict[str, int]:
    """Mirror every non-.edf object into ``root``. Idempotent (R14)."""
    root = Path(root)
    objs = fetch.list_objects()
    wanted = [o for o in objs if not o.key.endswith(".edf")]
    if not include_pdf:
        wanted = [o for o in wanted if not o.key.endswith(".pdf")]

    def one(o) -> tuple[str, int]:
        dest = root / o.rel
        if dest.exists() and dest.stat().st_size == o.size:
            return o.rel, 0
        fetch.download(o.rel, dest, expect_size=o.size)
        return o.rel, o.size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, wanted))

    return {
        "objects": len(wanted),
        "fetched": sum(1 for _, n in results if n),
        "bytes": sum(n for _, n in results),
    }


def records(root: Path) -> list[str]:
    return [l.strip() for l in (Path(root) / "RECORDS").read_text().splitlines() if l.strip()]


def records_with_seizures(root: Path) -> list[str]:
    p = Path(root) / "RECORDS-WITH-SEIZURES"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def summary_paths(root: Path) -> list[Path]:
    return sorted(Path(root).glob("chb*/chb*-summary.txt"))


def seizure_annot_paths(root: Path) -> list[Path]:
    return sorted(Path(root).glob("chb*/*.edf.seizures"))

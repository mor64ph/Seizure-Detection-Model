"""R30: the header-only pre-pass, and the file inventory it produces.

Two range requests per file -- 256 bytes to learn the signal count, then the
exact header. About 4.2 MB for all 686 files, against 45.76 GB for the bulk
payload. This buys back PRD §7.2's global two-pass guarantee without violating
R15, so the canonical channel list is frozen before any bandwidth is committed.

Everything captured here is on R11's capture-before-delete list: duration,
sample counts, channel labels. Losing it means re-downloading.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..signal import channels as ch
from . import edf_header as eh
from . import fetch

MANIFEST_PARQUET = "file_manifest.parquet"
CHANNELS_JSON = "channel_manifest.json"


def probe_one(rel: str) -> dict:
    """Two range requests, returning a manifest row for one .edf file."""
    probe = fetch.get_range(rel, 0, eh.PROBE_BYTES - 1)
    size = eh.header_size_from_probe(probe)
    buf = probe if size <= len(probe) else probe + fetch.get_range(rel, len(probe), size - 1)
    h = eh.parse(buf)
    cls = ch.classify(h.labels)
    rates = set(h.sample_rates)
    return {
        "rel": rel,
        "record_id": Path(rel).stem,
        "case_id": rel.split("/")[0],
        "n_signals": h.n_signals,
        "n_records": h.n_records,
        "record_duration_sec": h.record_duration_sec,
        "duration_sec": h.duration_sec,
        "sample_rate": max(rates) if len(rates) == 1 else -1.0,
        "mixed_sample_rates": len(rates) > 1,
        "header_bytes": h.header_bytes,
        "expected_file_bytes": eh.expected_file_bytes(h),
        "start_date": h.start_date,
        "start_time": h.start_time,
        "labels_raw": h.labels,
        "labels_kept": cls["kept"],
        "n_kept": len(cls["kept"]),
        "dropped_duplicate": cls["duplicate"],
        "dropped_non_eeg": cls["non_eeg"],
        "dropped_unrecognised": cls["unrecognised"],
    }


def build(out_dir: Path, workers: int = 12, limit: int | None = None) -> pd.DataFrame:
    """Run the pre-pass over every .edf in the bucket."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    objs = fetch.list_objects()
    edfs = sorted(fetch.iter_edf_keys(objs), key=lambda o: o.key)
    if limit:
        edfs = edfs[:limit]
    sizes = {o.rel: o.size for o in edfs}

    rows: list[dict] = []
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(probe_one, o.rel): o.rel for o in edfs}
        for i, f in enumerate(as_completed(futs), 1):
            rel = futs[f]
            try:
                r = f.result()
                r["s3_size"] = sizes.get(rel)
                r["size_matches_header"] = r["s3_size"] == r["expected_file_bytes"]
                rows.append(r)
            except Exception as e:  # R28: record, never swallow
                errors.append({"rel": rel, "error": f"{type(e).__name__}: {e}"})
            if i % 100 == 0:
                print(f"  probed {i}/{len(edfs)}", flush=True)

    df = pd.DataFrame(rows).sort_values("rel").reset_index(drop=True)

    per_file = {r["rel"]: r["labels_raw"] for r in rows}
    canonical = ch.canonical_from_manifest(per_file)
    payload = {
        "n_files": len(rows),
        "canonical_channels": canonical,
        "presence": [
            {"label": lab, "n_files": n, "frac": round(fr, 4)}
            for lab, n, fr in ch.presence_table(per_file)
        ],
        "errors": errors,
    }
    (out_dir / CHANNELS_JSON).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    keep = [c for c in df.columns if not c.startswith("labels_") and not c.startswith("dropped_")]
    df[keep].to_parquet(out_dir / MANIFEST_PARQUET, index=False)
    df.to_json(out_dir / "file_manifest_full.json", orient="records", indent=1)
    return df


def load(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    out_dir = Path(out_dir)
    df = pd.read_parquet(out_dir / MANIFEST_PARQUET)
    payload = json.loads((out_dir / CHANNELS_JSON).read_text(encoding="utf-8"))
    return df, payload

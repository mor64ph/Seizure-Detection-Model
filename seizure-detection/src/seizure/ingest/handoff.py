"""Manual-download handoff: generate a work list, then extract what arrives.

Why this exists: extraction is 4% of per-record wall time and download is 96%,
and the download is limited by the link rather than by concurrency (measured:
4 parallel streams gave *less* aggregate throughput than 1). So the only real
speedup is a faster link, which means someone else fetching the bytes.

The R11 contract is unchanged. Files that arrive by hand are still SHA-256
verified against the distribution's own `SHA256SUMS.txt`, still have all seven
capture-before-delete fields recorded, and are still deleted only through
`stream.delete_raw`. A hand-delivered file gets no more trust than a
self-downloaded one -- arguably it needs more checking, not less, because a
truncated or resumed-wrong transfer is invisible until the checksum fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config
from ..labels.registry import CASE_TO_SUBJECT
from . import fetch, stream

BASE_URL = fetch.BUCKET + fetch.PREFIX


def remaining(cfg: Config, ledger: stream.Ledger) -> list[tuple[str, int]]:
    """(rel, size) for every .edf not yet completed in the ledger."""
    objs = {o.rel: o.size for o in fetch.iter_edf_keys(fetch.list_objects())}
    return sorted(
        (rel, size) for rel, size in objs.items()
        if not ledger.done(Path(rel).stem)
    )


def interictal_hours(ledger: stream.Ledger) -> dict[str, float]:
    """Interictal hours already held, per subject.

    This is the quantity that decides download priority. A subject with 2.5
    hours contributes a false-alarm-rate estimate with roughly four times the
    relative error of one with 40 hours, so bytes spent on the starved subject
    buy far more than bytes spent on the rich one.
    """
    out: dict[str, float] = {}
    for r in ledger.rows.values():
        if r.state not in ("extracted", "raw_deleted"):
            continue
        sub = CASE_TO_SUBJECT[r.case_id]
        n0 = (r.label_counts or {}).get("0", 0) or (r.label_counts or {}).get(0, 0)
        out[sub] = out.get(sub, 0.0) + n0 * 10 / 3600
    return out


def priority_tiers(
    cfg: Config,
    ledger: stream.Ledger,
    out_dir: Path,
    per_subject: int = 6,
    n_tiers: int = 4,
) -> dict:
    """Emit download sets ordered by which subject needs interictal data most.

    Unlike ``plan``, which batches by size to fit disk, this batches by *value*:
    tier 1 is the subjects with the least interictal coverage, capped at
    ``per_subject`` records each and spread evenly across the subject's
    remaining records so the sample is not all from one stretch of the
    admission.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = interictal_hours(ledger)
    rem = remaining(cfg, ledger)

    by_sub: dict[str, list[tuple[str, int]]] = {}
    for rel, size in rem:
        by_sub.setdefault(CASE_TO_SUBJECT[rel.split("/")[0]], []).append((rel, size))

    # Poorest coverage first.
    order = sorted(by_sub, key=lambda s: have.get(s, 0.0))
    per_tier = max(1, len(order) // n_tiers)
    tiers: list[dict] = []
    for t in range(n_tiers):
        subs = order[t * per_tier : (t + 1) * per_tier] if t < n_tiers - 1 else order[t * per_tier :]
        picked: list[tuple[str, int]] = []
        for sub in subs:
            lst = by_sub[sub]
            step = max(1, len(lst) // per_subject)
            picked += lst[::step][:per_subject]
        picked.sort()
        if not picked:
            continue
        name = f"tier{t + 1}"
        lines = []
        for rel, _ in picked:
            lines.append(f'url = "{BASE_URL + rel.replace("+", "%2B")}"')
            lines.append(f'output = "{rel}"')
        (out_dir / f"{name}.curlrc").write_text("\n".join(lines) + "\n", encoding="utf-8")
        tiers.append({
            "tier": t + 1,
            "subjects": [{"subject": s, "interictal_h": round(have.get(s, 0.0), 1)} for s in subs],
            "records": len(picked),
            "gb": round(sum(s for _, s in picked) / 1e9, 2),
            "file": str(out_dir / f"{name}.curlrc"),
        })

    summary = {"tiers": tiers, "per_subject": per_subject,
               "destination": str(Path(cfg.data.root).resolve())}
    (out_dir / "priority_tiers.json").write_text(json.dumps(summary, indent=2),
                                                 encoding="utf-8")
    return summary


def plan(
    cfg: Config,
    ledger: stream.Ledger,
    out_dir: Path,
    batch_gb: float = 4.0,
    per_subject: int | None = None,
) -> dict:
    """Write the work list, batched to fit a disk budget.

    ``per_subject`` restricts to a stratified sample spread evenly across each
    subject's records, which is the cheap way to fix peri-ictal bias without
    fetching everything.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rem = remaining(cfg, ledger)

    if per_subject:
        by_sub: dict[str, list[tuple[str, int]]] = {}
        for rel, size in rem:
            by_sub.setdefault(CASE_TO_SUBJECT[rel.split("/")[0]], []).append((rel, size))
        picked: list[tuple[str, int]] = []
        for lst in by_sub.values():
            step = max(1, len(lst) // per_subject)
            picked += lst[::step][:per_subject]
        rem = sorted(picked)

    sums = fetch.parse_sha256sums(
        (Path(cfg.data.root) / "SHA256SUMS.txt").read_text(encoding="utf-8")
    )

    batches: list[list[tuple[str, int]]] = [[]]
    acc = 0.0
    for rel, size in rem:
        if acc + size > batch_gb * 1e9 and batches[-1]:
            batches.append([])
            acc = 0.0
        batches[-1].append((rel, size))
        acc += size

    rows = [
        {"batch": i + 1, "rel": rel, "size_bytes": size,
         "url": BASE_URL + rel.replace("+", "%2B"),
         "sha256": sums.get(rel, "")}
        for i, b in enumerate(batches) for rel, size in b
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "download_worklist.csv", index=False)

    # One URL list per batch, for `curl -K` or any downloader.
    for i, b in enumerate(batches, 1):
        lines = []
        for rel, _ in b:
            lines.append(f'url = "{BASE_URL + rel.replace("+", "%2B")}"')
            lines.append(f'output = "{rel}"')
        (out_dir / f"batch{i:02d}.curlrc").write_text("\n".join(lines) + "\n",
                                                      encoding="utf-8")

    summary = {
        "records": len(rem),
        "total_bytes": int(sum(s for _, s in rem)),
        "total_gb": round(sum(s for _, s in rem) / 1e9, 2),
        "batches": [
            {"batch": i + 1, "records": len(b),
             "gb": round(sum(s for _, s in b) / 1e9, 2)}
            for i, b in enumerate(batches)
        ],
        "base_url": BASE_URL,
        "destination": str(Path(cfg.data.root).resolve()),
    }
    (out_dir / "download_plan.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    return summary


def extract_present(
    cfg: Config,
    ledger: stream.Ledger,
    out_dir: Path,
    delete: bool = True,
) -> dict:
    """Extract every .edf now on disk that the ledger has not completed.

    Same verification and same delete gate as the streaming path. Nothing here
    touches the network except the seizure intervals, which come from the
    already-local label artifacts.
    """
    from ..features import extract as fx
    from ..labels import build as lb
    from ..signal import io

    root = Path(cfg.data.root)
    intervals, _ = lb.build(root, strict=False)
    truth = {
        r: [(int(a), int(b)) for a, b in zip(g.start_sec, g.end_sec)]
        for r, g in intervals.groupby("record_id")
    }
    sums = fetch.parse_sha256sums((root / "SHA256SUMS.txt").read_text(encoding="utf-8"))

    found = sorted(p for p in root.rglob("*.edf") if "_meta" not in p.parts)
    todo = [p for p in found if not ledger.done(p.stem)]
    tally = {"found": len(found), "todo": len(todo), "extracted": 0,
             "skipped": 0, "failed": 0, "checksum_bad": 0, "deleted": 0}

    for i, path in enumerate(todo, 1):
        rel = f"{path.parent.name}/{path.name}"
        st = stream.RecordState(record_id=path.stem, case_id=path.parent.name,
                                rel=rel, s3_size=path.stat().st_size)
        try:
            want = sums.get(rel)
            st.checksum_verified = (want is not None) and fetch.sha256_file(path) == want
            if want is not None and not st.checksum_verified:
                st.state, st.reason = "failed", "checksum_mismatch"
                tally["checksum_bad"] += 1
                ledger.put(st)
                print(f"  [{i}/{len(todo)}] {path.stem:14s} CHECKSUM FAIL "
                      f"-- re-download this one", flush=True)
                continue

            res = fx.extract_record(path, truth.get(path.stem, []), cfg)
            if res.skipped:
                st.state, st.reason = "skipped", res.skipped_reason
                st.duration_sec = res.duration_sec
                tally["skipped"] += 1
                ledger.put(st)
                print(f"  [{i}/{len(todo)}] {path.stem:14s} SKIP  {res.skipped_reason}",
                      flush=True)
                continue

            rec = io.read(path, list(cfg.signal.canonical_channels),
                          cfg.data.on_missing_channel)
            st.n_samples = int(rec.n_samples)
            st.duration_sec = float(rec.duration_sec)
            st.channel_labels = list(rec.channels)
            st.channel_stats = stream.channel_stats(rec.data, rec.channels)
            st.n_windows = res.n_windows
            st.label_counts = res.label_counts
            st.dropped = res.dropped

            for view, frame in (("per_channel", res.per_channel),
                                ("aggregated", res.aggregated)):
                if frame is None:
                    continue
                d = Path(out_dir) / view / f"case_id={st.case_id}"
                d.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(d / f"{path.stem}.parquet", index=False)
            st.features_path = str(out_dir)
            st.state = "extracted"
            tally["extracted"] += 1

            if delete and stream.delete_raw(path, st):
                st.state = "raw_deleted"
                tally["deleted"] += 1
            ledger.put(st)
            print(f"  [{i}/{len(todo)}] {path.stem:14s} ok    "
                  f"windows={st.n_windows} labels={st.label_counts}", flush=True)

        except Exception as e:  # noqa: BLE001 -- recorded, not swallowed
            st.state, st.error = "failed", f"{type(e).__name__}: {e}"
            tally["failed"] += 1
            ledger.put(st)
            print(f"  [{i}/{len(todo)}] {path.stem:14s} FAIL  {st.error}", flush=True)

    return tally

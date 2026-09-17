"""Command line entry point.

PRD §4 specifies typer. typer is not installed in this environment and argparse
covers the small surface here without adding a dependency; the deviation is
recorded in docs/data-notes.md (R28).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import config as C


def _cfg(args) -> C.Config:
    return C.load(args.config)


def cmd_manifest(args) -> int:
    from .ingest import manifest
    cfg = _cfg(args)
    df = manifest.build(Path(cfg.data.root) / "_meta", workers=args.workers, limit=args.limit)
    print(f"probed {len(df)} files")
    return 0


def cmd_metadata(args) -> int:
    from .ingest import metadata
    cfg = _cfg(args)
    print(metadata.sync(cfg.data.root, workers=args.workers))
    return 0


def cmd_labels(args) -> int:
    from .labels import build as lb
    cfg = _cfg(args)
    df, rep = lb.build(Path(cfg.data.root), strict=not args.lenient)
    out = Path(cfg.tracking.local_artifacts)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "seizure_intervals.parquet", index=False)
    (out / "label_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"seizures: {len(df)} across {df.record_id.nunique()} records")
    print(f"  from summary: {rep['n_from_summary']}  from annotations: {rep['n_from_annotations']}")
    print(f"  chb01-chb23: {rep['n_original_23']}")
    print(f"  parser anomalies: {len(rep['parser_anomalies'])}")
    print(f"  source disagreements: {len(rep['source_disagreements'])}")
    for d in rep["source_disagreements"]:
        print(f"    {d}")
    return 0


def cmd_registry(args) -> int:
    from .labels import registry
    cfg = _cfg(args)
    df = registry.load(Path(cfg.data.root))
    print(df.to_string(index=False))
    print(f"\n{df.subject_id.nunique()} distinct subjects across {len(df)} cases")
    return 0


def cmd_ingest(args) -> int:
    from .ingest import fetch, stream
    from .labels import build as lb
    cfg = _cfg(args)
    root = Path(cfg.data.root)

    intervals, _ = lb.build(root, strict=False)
    truth = {
        r: [(int(a), int(b)) for a, b in zip(g.start_sec, g.end_sec)]
        for r, g in intervals.groupby("record_id")
    }
    checksums = fetch.parse_sha256sums((root / "SHA256SUMS.txt").read_text(encoding="utf-8"))
    objs = {o.rel: o.size for o in fetch.iter_edf_keys(fetch.list_objects())}

    cases = None if cfg.data.cases == "all" else set(cfg.data.cases)
    rels = sorted(r for r in objs if cases is None or r.split("/")[0] in cases)
    if args.seizures_only:
        rels = [r for r in rels if Path(r).stem in truth]
    if args.limit:
        rels = rels[: args.limit]

    out_dir = Path(cfg.tracking.local_artifacts) / "features" / cfg.extraction_hash
    ledger = stream.Ledger(out_dir / "ledger.json")
    print(f"{len(rels)} records targeted; ledger has {len(ledger.rows)} entries")

    for i, rel in enumerate(rels, 1):
        rid = Path(rel).stem
        if ledger.done(rid) and not args.force:
            continue
        st = stream.ingest_record(
            rel, objs[rel], truth.get(rid, []), cfg, out_dir, checksums,
            keep_raw=args.keep_raw,
        )
        ledger.put(st)
        flag = {"raw_deleted": "ok", "extracted": "kept", "skipped": "SKIP",
                "failed": "FAIL"}.get(st.state, st.state)
        print(f"  [{i}/{len(rels)}] {rid:14s} {flag:5s} "
              f"windows={st.n_windows} labels={st.label_counts} "
              f"{st.reason or st.error or ''}", flush=True)

    fr = ledger.to_frame()
    print("\nstate counts:", fr.state.value_counts().to_dict())
    return 0


def cmd_plan(args) -> int:
    """Generate the manual-download work list (handoff to a faster link)."""
    from .ingest import handoff, stream
    cfg = _cfg(args)
    out_dir = Path(cfg.tracking.local_artifacts) / "features" / cfg.extraction_hash
    ledger = stream.Ledger(out_dir / "ledger.json")
    plan_dir = Path(cfg.tracking.local_artifacts) / "handoff"
    summary = handoff.plan(cfg, ledger, plan_dir, batch_gb=args.batch_gb,
                           per_subject=args.per_subject)
    print(f"{summary['records']} records, {summary['total_gb']} GB, "
          f"{len(summary['batches'])} batches")
    for b in summary["batches"]:
        print(f"  batch {b['batch']:02d}: {b['records']:3d} records  {b['gb']:5.2f} GB")
    print(f"\nwork list : {plan_dir / 'download_worklist.csv'}")
    print(f"curl files: {plan_dir}/batch*.curlrc")
    print(f"destination: {summary['destination']}")
    return 0


def cmd_extract_local(args) -> int:
    """Extract .edf files already on disk (verify, extract, delete per R11)."""
    from .ingest import handoff, stream
    cfg = _cfg(args)
    out_dir = Path(cfg.tracking.local_artifacts) / "features" / cfg.extraction_hash
    ledger = stream.Ledger(out_dir / "ledger.json")
    tally = handoff.extract_present(cfg, ledger, out_dir, delete=not args.keep_raw)
    print(f"\n{tally}")
    if tally["checksum_bad"]:
        print(f"WARNING: {tally['checksum_bad']} file(s) failed checksum "
              f"-- re-download those before trusting any result")
    return 0


def cmd_train(args) -> int:
    from . import train as T
    from .labels import build as lb
    cfg = _cfg(args)
    root = Path(cfg.data.root)
    feat_dir = Path(cfg.tracking.local_artifacts) / "features" / cfg.extraction_hash / args.view
    files = sorted(feat_dir.glob("case_id=*/*.parquet"))
    if not files:
        print(f"no feature files under {feat_dir}", file=sys.stderr)
        return 1
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"loaded {len(df):,} windows x {df.shape[1]} cols from {len(files)} records")
    print("  label counts:", df.label.value_counts().to_dict())

    intervals, _ = lb.build(root, strict=False)
    truth = {
        r: [(int(a), int(b)) for a, b in zip(g.start_sec, g.end_sec)]
        for r, g in intervals.groupby("record_id")
    }
    ledger_p = Path(cfg.tracking.local_artifacts) / "features" / cfg.extraction_hash / "ledger.json"
    durations = {}
    if ledger_p.exists():
        durations = {
            r["record_id"]: r["duration_sec"]
            for r in json.loads(ledger_p.read_text(encoding="utf-8"))
            if r.get("duration_sec")
        }

    rows, detail = T.run_loso(df, cfg, truth, durations)
    rand = T.run_random_window(df, cfg) if args.contrast else []

    # Local JSON run log, flushed per fold so a kill loses at most one (PRD §3.2).
    from .tracking import sink as SK
    run = SK.RunSink(Path(cfg.tracking.local_artifacts), cfg.full_hash,
                     cfg.tracking.mlflow_experiment)
    run.log_params(view=args.view, protocol=cfg.splits.protocol,
                   models=",".join(cfg.training.models), seed=cfg.splits.seed,
                   extraction_hash=cfg.extraction_hash,
                   n_windows=len(df), source=cfg.data.source)
    for d in detail:
        if "pr_auc" in d:
            run.log_metrics(int(d["fold_idx"]), d["model"], d["feature_view"], d)
    print("  tracking:", run.path, "| mlflow:", run.mirror_to_mlflow())

    out = Path(cfg.tracking.local_artifacts)
    out.mkdir(parents=True, exist_ok=True)
    if len(rows):
        rows.to_parquet(out / f"run_metrics_{args.view}.parquet", index=False)
    payload = {"loso": detail, "random_window": rand,
               "config_hash": cfg.full_hash, "view": args.view}
    (out / f"detail_{args.view}.json").write_text(json.dumps(payload, indent=1, default=str),
                                                  encoding="utf-8")
    print(f"\nwrote {out/('detail_%s.json' % args.view)}")
    return 0


def cmd_csv_study(args) -> int:
    """R25: the four-protocol leakage study on csv_scaled."""
    from .csvsrc import study
    cfg = _cfg(args)
    art = Path(cfg.tracking.local_artifacts)
    cache = art / "csv_scaled_features.parquet"
    if cache.exists() and not args.rebuild:
        df = pd.read_parquet(cache)
        print(f"reusing cached features {df.shape} ({cache})")
    else:
        csv = cfg.data.csv_path or Path("../EEG_Scaled_data.csv")
        print(f"building features from {csv} (one full pass, a few minutes)")
        df = study.build_features(Path(csv), cfg, notch=not args.no_notch)
        df.to_parquet(cache, index=False)
    print(f"  rows {len(df)} | repeats {int(df.is_repeat.sum())} | "
          f"pseudo_records {df.pseudo_record.nunique()}")
    rows = study.run(df, cfg)
    dec = study.decompose(rows)
    out = study.write(art, rows, dec)
    print(pd.DataFrame(rows)[["protocol", "model", "n_rows", "n_pos",
                              "pr_auc", "recall", "precision"]].to_string(index=False))
    print()
    print(pd.DataFrame(dec).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


def cmd_sensitivity(args) -> int:
    """PRD §6.2 ictal-threshold sweep."""
    from .eval import sensitivity as S
    from .labels import build as lb
    cfg = _cfg(args)
    feat = Path(cfg.tracking.local_artifacts) / "features" / cfg.extraction_hash / args.view
    files = sorted(feat.glob("case_id=*/*.parquet"))
    if not files:
        print(f"no feature files under {feat}", file=sys.stderr)
        return 1
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    iv, _ = lb.build(Path(cfg.data.root), strict=False)
    truth = {r: [(int(a), int(b)) for a, b in zip(g.start_sec, g.end_sec)]
             for r, g in iv.groupby("record_id")}
    rows = S.sweep(df, cfg, truth, model=args.model)
    out = Path(cfg.tracking.local_artifacts) / "overlap_sensitivity.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


def cmd_export_scores(args) -> int:
    """Per-window held-out scores, so a viewer can redraw at any threshold."""
    from . import train as T
    from .labels import build as lb
    cfg = _cfg(args)
    feat_dir = Path(cfg.tracking.local_artifacts) / "features" / cfg.extraction_hash / args.view
    files = sorted(feat_dir.glob("case_id=*/*.parquet"))
    if not files:
        print(f"no feature files under {feat_dir}", file=sys.stderr)
        return 1
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    intervals, _ = lb.build(Path(cfg.data.root), strict=False)
    truth = {r: [(int(a), int(b)) for a, b in zip(g.start_sec, g.end_sec)]
             for r, g in intervals.groupby("record_id")}
    subs = [s.strip() for s in args.subjects.split(",") if s.strip()]
    out = T.export_scores(df, cfg, truth, subs, args.model)
    if not len(out):
        print(f"no folds matched {subs}", file=sys.stderr)
        return 1
    dest = Path(cfg.tracking.local_artifacts) / f"scores_{args.view}.parquet"
    out.to_parquet(dest, index=False)
    print(f"wrote {dest}: {len(out):,} windows, "
          f"{out.record_id.nunique()} records, subjects {sorted(out.subject_id.unique())}")
    return 0


def cmd_fit_final(args) -> int:
    """Fit the designated model on every subject and persist it for inference.

    The LOSO summary travels with the artifact. A model trained on all 23
    subjects has no held-out score of its own, so shipping it without the
    cross-subject numbers would invite someone to assume it is better than the
    evaluation says.
    """
    import joblib

    from . import train as T
    cfg = _cfg(args)
    art = Path(cfg.tracking.local_artifacts)
    feat_dir = art / "features" / cfg.extraction_hash / args.view
    files = sorted(feat_dir.glob("case_id=*/*.parquet"))
    if not files:
        print(f"no feature files under {feat_dir}", file=sys.stderr)
        return 1
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    model, cols, meta = T.fit_final(df, cfg, args.model)

    # Operating point and expected performance come from the LOSO run, not from
    # this fit, which has nothing held out.
    detail = Path(args.loso_detail) if args.loso_detail else art / f"detail_{args.view}.json"
    if detail.exists():
        d = pd.DataFrame(json.loads(detail.read_text(encoding="utf-8"))["loso"])
        g = d[d.model == meta["model"]]
        if len(g):
            det, true = int(g.n_detected.sum()), int(g.n_true_events.sum())
            meta["operating_point"] = {
                "threshold": float(g.threshold.median()),
                "k": int(g.consecutive_k.mode().iloc[0]),
                "threshold_range": [float(g.threshold.min()), float(g.threshold.max())],
                "basis": "median threshold and modal k across the LOSO folds",
            }
            meta["loso_expectation"] = {
                "pr_auc_mean": float(g.pr_auc.mean()),
                "seizures_detected": det, "seizures_total": true,
                "event_sensitivity": det / true if true else None,
                "fa_per_hour": float(g.false_alarms.sum() / g.interictal_hours.sum()),
                "subjects_with_zero_detections": int(
                    ((g.n_detected == 0) & (g.n_true_events > 0)).sum()),
                "n_subjects": int(g.test_subject.nunique()),
                "source": str(detail),
            }
    meta["feature_columns"] = cols
    dest = art / f"final_model_{args.view}.joblib"
    joblib.dump({"model": model, "columns": cols, "meta": meta}, dest)
    (art / f"final_model_{args.view}.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8")
    op = meta.get("operating_point", {})
    print(f"wrote {dest}")
    print(f"  {meta['model']} on {meta['n_subjects']} subjects, "
          f"{meta['n_rows_fitted']:,} rows, {meta['n_features']} features")
    print(f"  operating point: threshold {op.get('threshold')}, k {op.get('k')}")
    return 0


def cmd_report(args) -> int:
    from .eval import report as R
    cfg = _cfg(args)
    R.write(Path(cfg.tracking.local_artifacts), cfg)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="seizure")
    p.add_argument("--config", default="configs/base.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("manifest", help="R30 header-only pre-pass over all .edf files")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_manifest)

    s = sub.add_parser("metadata", help="fetch the non-.edf payload (R13)")
    s.add_argument("--workers", type=int, default=12)
    s.set_defaults(fn=cmd_metadata)

    s = sub.add_parser("registry", help="print the subject registry")
    s.set_defaults(fn=cmd_registry)

    s = sub.add_parser("labels", help="build seizure intervals, cross-validate (R29)")
    s.add_argument("--lenient", action="store_true", help="do not assert the 198 total")
    s.set_defaults(fn=cmd_labels)

    s = sub.add_parser("ingest", help="stream download -> extract -> delete raw (R11/R14)")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seizures-only", action="store_true",
                   help="only records containing an annotated seizure")
    s.add_argument("--keep-raw", action="store_true", help="do not delete raw .edf files")
    s.add_argument("--force", action="store_true", help="re-ingest completed records")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("plan", help="emit a manual-download work list for a faster link")
    s.add_argument("--batch-gb", type=float, default=4.0,
                   help="max GB per batch, so it fits free disk")
    s.add_argument("--per-subject", type=int, default=None,
                   help="stratified sample: N interictal records per subject")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("extract-local",
                       help="verify+extract .edf files already on disk, then delete (R11)")
    s.add_argument("--keep-raw", action="store_true")
    s.set_defaults(fn=cmd_extract_local)

    s = sub.add_parser("train", help="LOSO loop plus the random-split contrast")
    s.add_argument("--view", default="aggregated", choices=["aggregated", "per_channel"])
    s.add_argument("--contrast", action="store_true", help="also run the random-window split")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("csv-study", help="R25 four-protocol leakage study on csv_scaled")
    s.add_argument("--rebuild", action="store_true", help="re-extract features from the CSV")
    s.add_argument("--no-notch", action="store_true", help="skip the notch (violates R21)")
    s.set_defaults(fn=cmd_csv_study)

    s = sub.add_parser("sensitivity", help="PRD 6.2 ictal-threshold sweep")
    s.add_argument("--view", default="aggregated", choices=["aggregated", "per_channel"])
    s.add_argument("--model", default="logreg")
    s.set_defaults(fn=cmd_sensitivity)

    s = sub.add_parser("export-scores", help="per-window held-out scores for a viewer")
    s.add_argument("--view", default="aggregated", choices=["aggregated", "per_channel"])
    s.add_argument("--subjects", required=True, help="comma-separated, e.g. sub01,sub12")
    s.add_argument("--model", default=None, help="defaults to training.headline_model")
    s.set_defaults(fn=cmd_export_scores)

    s = sub.add_parser("fit-final", help="fit on all subjects and persist for inference")
    s.add_argument("--view", default="aggregated", choices=["aggregated", "per_channel"])
    s.add_argument("--model", default=None, help="defaults to training.headline_model")
    s.add_argument("--loso-detail", default=None,
                   help="detail json supplying the operating point and LOSO expectation")
    s.set_defaults(fn=cmd_fit_final)

    s = sub.add_parser("report", help="generate artifacts/report.md and report.json")
    s.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())

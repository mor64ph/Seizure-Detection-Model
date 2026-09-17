"""Thin Spark Connect driver over the pure-Python extraction path.

This is the ONLY module in the project permitted to import pyspark (R20). It
contains no signal processing and no feature logic of its own: it maps
``features.extract.extract_record`` over a DataFrame of file paths and unions
the results. If this file were deleted the pipeline would still run.

Its purpose is the portability claim in the README. PRD §3.3 requires all
extraction code to be Spark Connect-compatible so it runs unchanged on
serverless, which forbids RDD APIs, ``sc.parallelize``, ``sparkContext`` and
``.rdd``. Enforced by ``test_no_rdd_apis_under_src``.

Why the local pipeline does not use it: 686 files is a fan-out over paths, not
a big-data workload. A thread pool is the honest tool at this scale, and adding
a Spark session buys latency and a dependency for nothing. The driver exists so
the claim "this would run on a cluster unchanged" is checkable rather than
asserted.

Usage on serverless (see notebooks/00_verify_portability.py):

    df = spark.createDataFrame([(rel,) for rel in rels], "rel string")
    out = extract_fanout(spark, df, cfg, truth, volume_root)
    out.write.mode("overwrite").partitionBy("case_id").saveAsTable(...)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd


def _schema_for(cfg, view: str) -> str:
    """DDL string for the output frame. Derived from config, never hardcoded (R17)."""
    from ..features import core, views

    feats = core.feature_names(cfg.features.bands)
    if view == "per_channel":
        cols = views.per_channel_names(list(cfg.signal.canonical_channels), feats)
    else:
        cols = views.aggregated_names(feats)
    key = [
        "window_uid string", "case_id string", "subject_id string",
        "record_id string", "window_idx int", "t_start_sec double",
        "t_end_sec double", "label int", "overlap_frac double",
        "config_hash string", "source string",
    ]
    return ", ".join(key + [f"`{c}` float" for c in cols])


def extract_fanout(spark, paths_df, cfg, truth: dict, root: str, view: str = "aggregated"):
    """Map extraction over a DataFrame with a single 'rel' column.

    ``paths_df`` is a Spark DataFrame, not a collection — nothing is collected
    to the driver, and no RDD API is touched. ``truth`` and ``cfg`` are
    broadcast implicitly by closure capture, which Spark Connect supports
    because both are small and picklable.
    """
    cfg_json = cfg.model_dump_json()
    truth_json = json.dumps({k: [list(x) for x in v] for k, v in truth.items()})
    schema = _schema_for(cfg, view)

    def _work(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        # Imported inside the UDF so the driver does not need them resident,
        # and so a missing dependency fails on the executor with a clear error.
        from ..config import Config
        from ..features import extract as fx

        local_cfg = Config.model_validate_json(cfg_json)
        local_truth = {
            k: [(int(a), int(b)) for a, b in v] for k, v in json.loads(truth_json).items()
        }
        for batch in batches:
            frames = []
            for rel in batch["rel"]:
                rid = Path(str(rel)).stem
                res = fx.extract_record(
                    Path(root) / str(rel), local_truth.get(rid, []), local_cfg
                )
                if res.skipped:
                    continue
                frame = res.aggregated if view == "aggregated" else res.per_channel
                if frame is not None:
                    frames.append(frame)
            if frames:
                yield pd.concat(frames, ignore_index=True)

    # mapInPandas is Spark Connect-compatible; mapPartitions (RDD) is not.
    return paths_df.mapInPandas(_work, schema=schema)


BANNED_APIS = {
    "dot-rdd": r"\.rdd\b",
    "parallelize": r"sc\.parallelize",
    "spark-context": r"sparkContext",
    "rdd-type": r"\bRDD\b",
}

PURE_PACKAGES = ("features", "signal", "labels", "eval")


def code_only(source: str) -> str:
    """Strip comments and string literals, leaving executable code.

    Necessary because a naive grep cannot distinguish code that *uses* a
    forbidden API from prose that *documents* that it is forbidden. Without
    this, the portability test fails on its own explanatory docstring, which
    would pressure the next person to stop documenting the rule.
    """
    import io
    import tokenize

    lines = source.splitlines(keepends=True)
    grid = [list(line) for line in lines]
    drop = {tokenize.COMMENT, tokenize.STRING}
    fstring_mid = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fstring_mid is not None:
        drop.add(fstring_mid)

    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type not in drop:
                continue
            (r0, c0), (r1, c1) = tok.start, tok.end
            for r in range(r0 - 1, r1):
                if r >= len(grid):
                    break
                lo = c0 if r == r0 - 1 else 0
                hi = c1 if r == r1 - 1 else len(grid[r])
                for c in range(lo, min(hi, len(grid[r]))):
                    if grid[r][c] != "\n":
                        grid[r][c] = " "
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source  # fail closed: scan everything rather than nothing

    # Blanking in place, rather than re-joining tokens, keeps `df.rdd` contiguous.
    # Joining tokens with a separator split attribute access apart, so the
    # scanner reported every file as portable -- a false negative, which is
    # strictly worse than the docstring false positive it was meant to fix.
    return "".join("".join(row) for row in grid)


def portability_report(src_root: Path) -> dict:
    """Scan the source tree for the APIs serverless forbids (PRD §3.3).

    One implementation, called by both the notebook and the test suite.
    """
    import re

    hits: list[str] = []
    spark_in_pure: list[str] = []

    for p in sorted(Path(src_root).rglob("*.py")):
        raw = p.read_text(encoding="utf-8")
        code = code_only(raw)
        for name, pat in BANNED_APIS.items():
            if re.search(pat, code):
                hits.append(f"{p.name}: {name}")
        if any(d in p.parts for d in PURE_PACKAGES) and re.search(
            r"^\s*(import|from)\s+pyspark", code, re.M
        ):
            spark_in_pure.append(p.name)

    return {
        "files_scanned": len(list(Path(src_root).rglob("*.py"))),
        "rdd_api_usages": hits,
        "spark_imports_in_pure_modules": spark_in_pure,
        "portable": not hits and not spark_in_pure,
    }

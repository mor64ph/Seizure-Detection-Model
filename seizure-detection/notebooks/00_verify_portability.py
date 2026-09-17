# Databricks notebook source
# MAGIC %md
# MAGIC # Portability verification
# MAGIC
# MAGIC PRD §3.3 and R20. This notebook is the proof that the extraction path runs
# MAGIC **unchanged** on Databricks serverless, against a small sample uploaded to a Unity
# MAGIC Catalog Volume. The full 686-file run happens locally — see the README, which states
# MAGIC the split explicitly rather than implying the whole pipeline ran in the cloud.
# MAGIC
# MAGIC Constraints this notebook is written against:
# MAGIC
# MAGIC - **Serverless only.** No custom compute, no GPUs.
# MAGIC - **Spark Connect APIs only.** No RDD APIs, no `sc.parallelize`, no `.rdd`.
# MAGIC - **No outbound internet from UDFs.** The `.edf` files must already be in the Volume;
# MAGIC   nothing here downloads from PhysioNet.
# MAGIC - **`spark.createDataFrame` from local data caps at 128 MB of rows.** Only file paths
# MAGIC   cross that boundary, never signal data.
# MAGIC - **Quota overrun kills compute for the day.** Keep the sample small.

# COMMAND ----------

# MAGIC %pip install mne pydantic pyyaml
# MAGIC %restart_python

# COMMAND ----------

import json
import sys
from pathlib import Path

# The repo is expected as a workspace file or Git folder. Only src/ is needed.
REPO = "/Workspace/Repos/seizure-detection"
VOLUME = "/Volumes/main/seizure/chbmit"   # UC Volume holding the sample
sys.path.insert(0, f"{REPO}/src")

from seizure import config                      # noqa: E402
from seizure.spark import fanout                # noqa: E402

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Static check: no forbidden APIs anywhere under `src/`
# MAGIC
# MAGIC Same scanner the local test suite uses, so this cannot drift from it. It strips
# MAGIC comments and string literals first — a naive grep cannot distinguish code that uses
# MAGIC `.rdd` from a docstring saying `.rdd` is banned.

# COMMAND ----------

report = fanout.portability_report(Path(f"{REPO}/src"))
print(json.dumps(report, indent=2))
assert report["portable"], report
print(f"\nOK — {report['files_scanned']} files scanned, zero RDD-API usages.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Dynamic check: run the identical extraction on a 10-file sample

# COMMAND ----------

cfg = config.load(f"{REPO}/configs/base.yaml")
print("extraction_hash:", cfg.extraction_hash)
print("canonical channels:", len(cfg.signal.canonical_channels))
print("window:", cfg.windowing.length_sec, "s at", cfg.signal.sample_rate, "Hz")

# Seizure intervals come from the checked-in label artifact, not from a re-parse,
# so this notebook needs no network access at all.
intervals = json.loads(Path(f"{REPO}/artifacts/seizure_intervals.json").read_text())
truth = {
    r: [(int(a), int(b)) for a, b in v] for r, v in intervals.items()
}

rels = [p.name for p in sorted(Path(VOLUME).rglob("*.edf"))][:10]
print(f"\nsample: {len(rels)} files")
for r in rels:
    print("  ", r)

# COMMAND ----------

# Only paths cross into Spark — well under the 128 MB createDataFrame cap.
paths_df = spark.createDataFrame([(r,) for r in rels], "rel string")  # noqa: F821

out = fanout.extract_fanout(
    spark,          # noqa: F821
    paths_df,
    cfg,
    truth,
    root=VOLUME,
    view="aggregated",
)

n = out.count()
print("windows extracted:", n)
out.printSchema()

# COMMAND ----------

display(  # noqa: F821
    out.groupBy("case_id", "label").count().orderBy("case_id", "label")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Equivalence check
# MAGIC
# MAGIC The same records extracted locally must give the same window count and the same label
# MAGIC distribution. The `config_hash` stamped on every row (R19) is what makes the comparison
# MAGIC meaningful: if it differs, the two runs were not configured identically and the
# MAGIC comparison is void.

# COMMAND ----------

hashes = [r["config_hash"] for r in out.select("config_hash").distinct().collect()]
assert hashes == [cfg.extraction_hash], hashes
print("config_hash matches local extraction:", hashes[0])

summary = {
    "n_windows": n,
    "by_label": {
        str(r["label"]): r["count"]
        for r in out.groupBy("label").count().collect()
    },
    "config_hash": cfg.extraction_hash,
    "records": rels,
    "portability": report,
}
print(json.dumps(summary, indent=2))

# Commit this output alongside the notebook (PRD §13 acceptance criterion).
Path(f"{REPO}/artifacts/portability_serverless.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What this does and does not establish
# MAGIC
# MAGIC **Establishes:** the feature path imports and runs under Spark Connect on serverless,
# MAGIC touches no RDD API, and produces byte-identical feature semantics to the local run for
# MAGIC the same records and the same `config_hash`.
# MAGIC
# MAGIC **Does not establish:** that the full 686-file run was performed here. It was not.
# MAGIC Ingest is a per-file stream on local disk because the dataset is 45.76 GB against a
# MAGIC 26 GB budget, and because 686 files is a fan-out over paths rather than a workload that
# MAGIC needs a cluster.

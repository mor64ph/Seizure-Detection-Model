# PRD: Cross-Patient Seizure Detection on CHB-MIT

**Target executor:** Claude Code
**Status:** Ready to build
**Owner:** Hrisit

---

## 0. Read this first

This document is the spec. Build in the milestone order given in §12. Do not skip ahead to modelling.

Three rules that override anything else you infer:

1. **Grouping key is `subject_id`, never `case_id`.** `chb01` and `chb21` are the same human, recorded 1.5 years apart. Treating them as two patients is silent leakage and it defeats the entire purpose of this project.
2. **Never concatenate records.** There are hardware gaps between consecutively-numbered `.edf` files within a case. Windows are computed inside a single file, never across a file boundary.
3. **Nothing is fit on test data.** Scalers, thresholds, hyperparameters, feature selection — all fit inside the training fold only.

If any instruction here turns out to be factually wrong about the data (channel names, summary format, file counts), fix the code to handle reality and record the discrepancy in `docs/data-notes.md`. Do not silently paper over it.

---

## 1. What we are building

A binary classifier that takes a 10-second window of multi-channel scalp EEG and outputs a seizure probability, evaluated under **leave-one-subject-out cross-validation** on the full CHB-MIT database.

The deliverable is not the model. The deliverable is a pipeline plus an honest evaluation report that states:

- Window-level precision, recall, PR-AUC and ROC-AUC on held-out subjects
- Event-level sensitivity, mean detection latency, and false alarms per hour
- The same metrics computed under a naive random window split, side by side, so the inflation is visible and quantified
- A per-subject breakdown showing which subjects the model fails on

### Non-goals for v1

- Deep learning of any kind. No CNN, no LSTM, no transformer.
- Seizure *prediction* (pre-ictal forecasting). This is detection only.
- Real-time or streaming inference.
- Patient-specific models. The whole point is cross-subject generalisation.
- Model serving endpoints.

---

## 2. Dataset facts

Source: CHB-MIT Scalp EEG Database v1.0.0, PhysioNet. License: Open Data Commons Attribution. No account required.

| Fact | Value |
|---|---|
| Case directories | 24 (`chb01`–`chb24`) |
| Distinct subjects | 23 (chb01 and chb21 are the same subject) |
| Total `.edf` files | 664 (listed in the `RECORDS` file) |
| Files containing seizures | 129 (listed in `RECORDS-WITH-SEIZURES`) |
| Total annotated seizures | 198 (182 in the original 23 cases) |
| Sampling rate | 256 Hz |
| Resolution | 16-bit |
| Channels per file | 23–26, bipolar longitudinal montage, 10-20 system |
| Approx. total duration | ~915–980 hours |
| Approx. download size | ~43 GB |

Root-level files to use: `RECORDS`, `RECORDS-WITH-SEIZURES`, `SUBJECT-INFO`, `MD5SUMS`, `SHA256SUMS`.

### Known landmines

| Landmine | Handling |
|---|---|
| `chb21` is the same subject as `chb01` | Hardcode in the subject registry (§5.1). Assert on it in a test. |
| `chb24` is absent from `SUBJECT-INFO` | Registry must tolerate missing demographics. Do not drop the case. |
| `chb12_27/28/29` use a different montage | Canonicalisation (§7.2) drops them or fails loudly, per config. |
| `chb17` files are named `chb17a_*`, `chb17b_*` | Filename regex must not assume `chb\d\d_\d\d`. |
| Duplicate channel labels (e.g. `T8-P8` twice in some files) | Deduplicate by label, keep first occurrence, log it. |
| Placeholder channels (`-`, `.`) | Drop before canonicalisation. |
| Gaps between consecutive files | Never concatenate. Window within file only. |
| Summary clock times can exceed 24:00:00 | Do not parse with `datetime.strptime`. Seizure offsets are relative seconds; you do not need wall-clock time for labelling. |
| Multi-seizure files use `Seizure 1 Start Time:` etc. | Parser must handle both the indexed and non-indexed forms. |

---

## 3. Environment and where each stage runs

### 3.1 Local (the workhorse)

- Python 3.11
- `mne`, `numpy`, `scipy`, `pandas`, `pyarrow`, `pyspark`, `delta-spark`, `scikit-learn`, `typer`, `pyyaml`, `pytest`, `ruff`
- Spark runs as `local[*]`. This is a fan-out over file paths, not a big-data workload.
- Disk budget: ~45 GB raw + ~2 GB derived.

### 3.2 Databricks Free Edition (silver onward)

Hard constraints you must design around:

- **Serverless only.** No custom compute, no GPUs.
- **Outbound internet restricted to trusted domains.** You cannot download PhysioNet from inside. UDFs have no internet at all.
- **Spark Connect APIs only.** No RDD APIs, no `sc.parallelize`, no `.rdd` access anywhere.
- **`spark.createDataFrame` from local data caps at 128 MB rows.**
- **DBFS access limited.** Use Unity Catalog Volumes or workspace files.
- **Quota overrun kills compute for the remainder of the day.** Design every long-running job to checkpoint.
- Max 5 concurrent job tasks. SQL warehouse capped at 2X-Small. One workspace, one metastore. No Scala or R.

### 3.3 Spark portability rule

All extraction code must be Spark Connect-compatible so it runs unchanged on serverless. Enforced by:

- A lint rule / test that greps `src/` for `.rdd`, `sc.parallelize`, `sparkContext`, `RDD` and fails if found.
- A `notebooks/00_verify_portability.py` Databricks notebook that runs the identical extraction job against a 10-file sample uploaded to a UC Volume. This is the proof that the code is cluster-ready; the full 664-file run happens locally.

State this split explicitly in the README. Do not imply the full run happened in the cloud.

---

## 4. Repository layout

```
seizure-detection/
├── README.md                      # the honest report, written last
├── pyproject.toml
├── Makefile
├── configs/
│   ├── base.yaml
│   ├── fast.yaml                  # 5 subjects, aggregated features, LR only
│   └── full.yaml                  # all 23 subjects, both views, all models
├── src/seizure/
│   ├── cli.py                     # typer entrypoint
│   ├── config.py                  # pydantic config models, config hashing
│   ├── ingest/
│   │   ├── download.py            # resumable wget, checksum verify
│   │   └── manifest.py            # file inventory -> parquet
│   ├── labels/
│   │   ├── summary_parser.py      # chbNN-summary.txt -> intervals
│   │   └── registry.py            # case_id -> subject_id mapping
│   ├── signal/
│   │   ├── io.py                  # mne EDF reading
│   │   ├── channels.py            # canonicalisation
│   │   ├── filters.py             # bandpass + notch
│   │   └── windows.py             # windowing + label assignment
│   ├── features/
│   │   ├── spectral.py            # Welch PSD, band power
│   │   ├── temporal.py            # line length, variance, Hjorth
│   │   ├── aggregate.py           # channel-agnostic aggregation
│   │   └── extract.py             # per-file orchestration (pure, no Spark)
│   ├── spark/
│   │   └── fanout.py              # thin mapInPandas driver over extract.py
│   ├── splits/
│   │   └── lopo.py                # subject-level fold assignment
│   ├── models/
│   │   ├── train.py
│   │   └── threshold.py
│   ├── eval/
│   │   ├── window_metrics.py
│   │   ├── event_metrics.py       # sensitivity, latency, FA/hour
│   │   └── report.py              # markdown + json artifacts
│   └── tracking/
│       └── mlflow_sink.py         # dual-writes MLflow + local JSON
├── notebooks/
│   ├── 00_verify_portability.py
│   ├── 01_silver_load.py
│   ├── 02_gold_build.py
│   └── 03_train_eval.py
├── tests/
│   └── fixtures/                  # golden summary files, synthetic signals
├── artifacts/                     # gitignored, run outputs
└── docs/
    └── data-notes.md              # every discrepancy found in the wild
```

**Hard rule:** everything under `src/seizure/features/`, `src/seizure/signal/`, `src/seizure/labels/` and `src/seizure/eval/` is pure Python. No Spark imports. Spark exists only in `src/seizure/spark/`. This is what makes the code testable and portable.

---

## 5. Data model

### 5.1 Subject registry (`registry.py`)

```python
CASE_TO_SUBJECT: dict[str, str] = {
    f"chb{i:02d}": f"sub{i:02d}" for i in range(1, 25)
}
CASE_TO_SUBJECT["chb21"] = "sub01"   # same human as chb01, 1.5 years later
```

Emit a table `subject_registry`:

| column | type | note |
|---|---|---|
| `case_id` | string | `chb01` … `chb24` |
| `subject_id` | string | 23 distinct values |
| `gender` | string | nullable — chb24 absent from SUBJECT-INFO |
| `age` | float | nullable |

Test: `assert df.subject_id.nunique() == 23` and `assert set(df[df.case_id.isin(["chb01","chb21"])].subject_id) == {"sub01"}`.

### 5.2 Seizure intervals (silver)

| column | type |
|---|---|
| `case_id` | string |
| `subject_id` | string |
| `record_id` | string (e.g. `chb01_03`) |
| `seizure_idx` | int (1-based within record) |
| `start_sec` | int |
| `end_sec` | int |
| `duration_sec` | int |

Validation: `duration_sec > 0`, total row count must equal 198 across all cases. If it does not, stop and investigate before proceeding.

### 5.3 Window features (silver, the main table)

One row per window. Two physical tables, same key columns:

**Key columns (both tables):**

| column | type |
|---|---|
| `window_uid` | string, `{record_id}_{window_idx:06d}` |
| `case_id` | string |
| `subject_id` | string |
| `record_id` | string |
| `window_idx` | int |
| `t_start_sec` | float |
| `t_end_sec` | float |
| `label` | int, 0 = interictal, 1 = ictal, -1 = guard (excluded) |
| `overlap_frac` | float, fraction of window inside a seizure interval |
| `config_hash` | string, hash of the extraction config |

**Table A — `features_per_channel`:** plus 23 × 14 = 322 float32 columns named `{channel}__{feature}`, e.g. `FP1_F7__bp_delta_rel`.

**Table B — `features_aggregated`:** plus 14 × 4 = 56 float32 columns named `{feature}__{stat}` where stat ∈ {mean, max, std, median} across channels.

Partition both by `case_id`. Write as Delta.

### 5.4 Gold

`fold_assignments`: `subject_id`, `fold_idx`, `role` ∈ {train, val, test}.
`run_metrics`: one row per (run_id, fold_idx, model, feature_view, metric_name, metric_value).

---

## 6. Labelling

### 6.1 Summary parser

Parse `chbNN-summary.txt`. Formats vary between cases; the parser must be tolerant and must **raise** rather than silently return empty on an unrecognised block.

Extract per record: file name, number of seizures, and for each seizure the start and end second (relative to file start).

Handle both `Seizure Start Time:` and `Seizure 1 Start Time:` forms. Ignore `File Start Time` / `File End Time` entirely — you do not need wall-clock time, and those fields contain values past 24:00:00 in some cases.

**Tests required:** one golden fixture per distinct format variant encountered, checked into `tests/fixtures/`. At minimum: a zero-seizure record, a single-seizure record, a multi-seizure record, and the `chb17a`/`chb17b` naming variant.

### 6.2 Window labelling

For each window `[t, t+10)`:

```
overlap_frac = (seconds of window inside any seizure interval) / window_length

if overlap_frac >= min_overlap_frac:          label = 1
elif window is within guard_sec of any interval boundary:  label = -1
else:                                          label = 0
```

Defaults: `min_overlap_frac = 0.5`, `guard_sec = 30`.

Rationale for the guard band: seizure onsets are human-annotated to the nearest second and the ictal/interictal transition is physiologically gradual. Windows straddling the boundary are genuinely ambiguous and training on them as hard negatives injects label noise. Label `-1` rows are excluded from both training and evaluation by default.

Produce a sensitivity table showing positive-class count and headline recall at `min_overlap_frac` ∈ {0.1, 0.25, 0.5, 0.75}. This is a one-line finding for the README, not a research project.

---

## 7. Signal processing

### 7.1 Order of operations (per file, strictly)

1. Read EDF with `mne.io.read_raw_edf(preload=True, verbose=False)`
2. Drop placeholder and duplicate channels
3. Canonicalise channel set (§7.2)
4. Notch filter at 60 Hz and 120 Hz
5. Bandpass 0.5–80 Hz
6. Window into non-overlapping 10s segments
7. Featurise per window per channel

Do not resample. 256 Hz is fine and resampling costs you time for nothing.

### 7.2 Channel canonicalisation

This is the step that determines whether cross-subject transfer is even possible. Feature vectors must be in identical channel order for every subject.

**Procedure:**
1. Pass 1 over all 664 files: collect the channel label set of each file into a manifest.
2. Compute the intersection present in ≥ 95% of files. Expect ~18–23 channels. Log the exact list to `docs/data-notes.md`.
3. Freeze that list in config as `canonical_channels`.
4. Pass 2 (extraction): reorder each file's channels to the canonical order. If a file is missing a canonical channel, apply `on_missing_channel` policy: `"skip_file"` (default) or `"fill_nan"`.

Files skipped under this policy must be logged with the reason and counted in the run report. Do not let them disappear.

### 7.3 Why the 60 Hz notch is mandatory

Boston mains is 60 Hz. The gamma band (30–80 Hz) sits directly on top of it. Without the notch, gamma band power is substantially a measure of line noise, and line noise level varies by recording session. A model given that feature can partially identify the session, which is the same leakage as a bad split wearing a different hat.

---

## 8. Features

### 8.1 Per channel per window (14 features)

**Spectral** — compute PSD via `scipy.signal.welch(x, fs=256, nperseg=512, noverlap=256)`, integrate over band edges:

| Band | Range (Hz) |
|---|---|
| delta | 0.5–4 |
| theta | 4–8 |
| alpha | 8–13 |
| beta | 13–30 |
| gamma | 30–80 |

- `bp_{band}_log` — `log1p` of absolute band power (5 features)
- `bp_{band}_rel` — band power / total power across 0.5–80 Hz (5 features)

**Temporal:**

- `line_length` — `sum(abs(diff(x)))`
- `variance` — `var(x)`
- `hjorth_mobility` — `sqrt(var(dx) / var(x))`
- `hjorth_complexity` — `mobility(dx) / mobility(x)`

Log-transform absolute band powers because they are heavy-tailed and unlogged values dominate any linear model.

### 8.2 Why relative band power matters here

Absolute EEG amplitude varies with electrode impedance, skull thickness, subject age and session gain. It is close to a subject fingerprint. Relative band power normalises it away and should transfer better across subjects. Compute both; the comparison is a finding worth reporting.

### 8.3 Two feature views

| View | Dim | Property |
|---|---|---|
| `per_channel` | 322 | Higher capacity, montage-fragile, more subject-specific |
| `aggregated` | 56 | Channel-order invariant, montage-robust, expected to transfer better |

Run every experiment against both. If `aggregated` beats `per_channel` under LOSO while losing under random split, that is the single most interesting result in the project and it goes at the top of the README.

### 8.4 Feature function tests (required)

Test against synthetic signals with known answers:

- Pure 10 Hz sine → `bp_alpha_rel` > 0.9
- Pure 2 Hz sine → `bp_delta_rel` > 0.9
- Linear ramp of slope `m`, length `n` → `line_length ≈ m * (n-1)`
- White noise, unit variance → `variance ≈ 1.0` within tolerance
- Pure sine → `hjorth_complexity ≈ 1.0` within tolerance

---

## 9. Splitting protocol

### 9.1 Primary: leave-one-subject-out

23 folds, one per `subject_id`. In each fold:

- Test = all windows from the held-out subject
- From the remaining 22, hold out 4 subjects as validation (deterministic, seeded)
- Train on the remaining 18

Validation subjects are used for threshold selection and hyperparameter choice. The test subject is touched exactly once, at the end.

### 9.2 Contrast run: naive random window split

Same data, same features, same model. Shuffle windows, 80/20 random split, ignore subject entirely. Report identically.

This run exists to be wrong. Both sets of numbers go in the README under the heading **"The same model, two evaluation protocols."** Compute and state the delta explicitly.

### 9.3 Inner loop

Hyperparameter search inside a fold uses `GroupKFold(groups=subject_id)` over the training subjects. Never `KFold`. Never `StratifiedKFold` without groups.

### 9.4 Scaler discipline

`StandardScaler` fit on training-fold rows only, applied to val and test. Wrap in a `sklearn.pipeline.Pipeline` so this cannot be got wrong by accident.

---

## 10. Models, imbalance, thresholds

### 10.1 Model ladder

Build in this order. Do not skip step 0.

| # | Model | Purpose |
|---|---|---|
| 0 | `DummyClassifier(strategy="most_frequent")` | Prints ~99.7% accuracy. This is the proof that accuracy is meaningless here. It goes in the README. |
| 1 | `LogisticRegression(class_weight="balanced", max_iter=2000)` | Interpretable baseline. Inspect coefficients. |
| 2 | `RandomForestClassifier(class_weight="balanced_subsample")` | Non-linear baseline. |
| 3 | `HistGradientBoostingClassifier` | Best expected performer, handles NaN natively. |

### 10.2 Imbalance

Positive rate is roughly 0.3% of windows (~1,100 seizure windows out of ~340,000). Compute the exact figure and put it in the README.

- Use `class_weight="balanced"` where supported.
- **No SMOTE.** Synthesising EEG feature vectors by interpolating between real windows is hard to defend and you will be asked about it.
- For training only, optionally downsample negatives to a configurable ratio (default 10:1). This cuts training rows from ~300k to ~12k per fold and makes the 23-fold loop survivable on Free Edition compute quota. **Test and validation folds are never downsampled.** Note the calibration caveat in the report.

### 10.3 Threshold selection

Do not use 0.5. Select the operating point by maximising F2 (recall-weighted) on the validation subjects of that fold, then freeze it before scoring test. Log the chosen threshold per fold; variance across folds is itself a finding.

### 10.4 Event smoothing

Require `k` consecutive positive windows to declare an event. Tune `k` ∈ {1, 2, 3} on validation. This is the highest-leverage lever for false alarms per hour and costs a few seconds of detection latency.

---

## 11. Evaluation

### 11.1 Window-level

Precision, recall, F1, F2, PR-AUC, ROC-AUC, confusion matrix. **Lead with PR-AUC**, not ROC-AUC — at 0.3% prevalence ROC-AUC looks flattering and hides the precision problem.

Report pooled across folds *and* per subject. The per-subject table is what reveals that the model works on 15 subjects and fails completely on 6.

### 11.2 Event-level (this is what makes the project non-generic)

Define a detected event as: a ground-truth seizure interval that overlaps at least one predicted positive run of length ≥ `k`.

| Metric | Definition |
|---|---|
| Event sensitivity | detected seizures / total seizures |
| Detection latency | seconds from annotated onset to first window of the detecting run. Report median and IQR. |
| False alarms per hour | predicted positive runs with no overlapping ground-truth seizure, divided by total interictal hours |

Clinically, FA/hour is the metric that decides whether a detector is usable. A window-level recall of 0.62 reads as mediocre; "detected 141 of 198 seizures, median latency 6s, 1.9 false alarms per hour" reads as a real system with known limitations.

### 11.3 Cost framing (required in README)

Write a short paragraph on what a false negative costs versus a false positive in this setting: a missed seizure means an unrecorded event in a monitoring unit evaluating a child for neurosurgery; a false alarm means clinician review time and alarm fatigue. State which way you tuned the threshold and why. Two or three sentences, no padding.

---

## 12. Build order

Each milestone ends with something runnable and tested. Do not start the next until the current one passes.

| M | Deliverable | Done when |
|---|---|---|
| **M0** | Repo scaffold, config system, CLI skeleton, CI running ruff + pytest | `make test` passes on an empty test suite |
| **M1** | Download + manifest + checksum verification | 664 files present, all checksums match `MD5SUMS`, manifest parquet written |
| **M2** | Summary parser + subject registry | Total seizure count == 198; registry asserts chb01/chb21 share a subject; golden-file tests pass |
| **M3** | Channel manifest pass + canonical list frozen | `docs/data-notes.md` lists the canonical channels and every non-conforming file with a reason |
| **M4** | Signal pipeline + windowing + labelling, single file | Feature functions pass synthetic-signal tests; one record produces a labelled window table |
| **M5** | Spark fan-out, chb01 only | Feature parquet for chb01; portability grep test passes |
| **M6** | Full extraction, all 24 cases | Both feature tables written; row count and skipped-file count reported |
| **M7** | Upload to Databricks UC Volume, silver Delta tables | Tables queryable; `notebooks/00_verify_portability.py` runs the identical job on a 10-file sample on serverless |
| **M8** | LOSO fold assignment + Dummy + LR, aggregated view only | First honest recall number exists |
| **M9** | Full grid: 4 models × 2 views × 23 folds, MLflow tracked | `run_metrics` populated; per-fold checkpointing survives a compute kill |
| **M10** | Naive random-split contrast run | Both numbers in one table |
| **M11** | Event-level evaluation | Sensitivity, latency, FA/hour per subject |
| **M12** | README written | Acceptance criteria in §13 all satisfied |

Use `configs/fast.yaml` (5 subjects, aggregated features, LR only) for every iteration up to M8. Only run `configs/full.yaml` when the pipeline is stable. This is not an optimisation, it is quota management.

---

## 13. Acceptance criteria

The project is done when all of these are true.

- [ ] `make test` passes; test suite covers the summary parser (golden files), window labelling boundary conditions, all 14 feature functions against synthetic signals, and the subject registry invariants
- [ ] Extraction is idempotent and cached by `config_hash`; re-running costs seconds, not hours
- [ ] The portability grep test finds zero RDD-API usages under `src/`
- [ ] `notebooks/00_verify_portability.py` has been run on Databricks serverless and its output is committed
- [ ] MLflow experiment contains one run per (fold, model, feature_view) with params, metrics and the fitted threshold logged
- [ ] `artifacts/report.json` and `artifacts/report.md` are generated by code, not written by hand
- [ ] README states the exact list of subjects in train, validation and test for at least one named fold
- [ ] README reports window-level precision, recall, PR-AUC and ROC-AUC on held-out subjects
- [ ] README reports event sensitivity, median detection latency, and false alarms per hour
- [ ] README contains the DummyClassifier accuracy figure and one sentence on why it is meaningless
- [ ] README contains the LOSO-vs-random-split comparison table with the delta stated numerically
- [ ] README contains a per-subject results table identifying the worst subjects
- [ ] README contains a confusion matrix and the false-negative cost paragraph
- [ ] README states plainly which stages ran locally and which ran on Databricks, and why
- [ ] `docs/data-notes.md` documents every file skipped, every channel anomaly, and every summary-format variant encountered

---

## 14. Things that will go wrong

Anticipate these rather than debugging them from scratch.

| Symptom | Likely cause |
|---|---|
| Recall near 1.0 under LOSO | Subject leakage. Check chb21. Check that `groups=` is actually being passed. |
| Total seizure count ≠ 198 | Parser dropping a format variant. Check `chb17a`/`chb17b` and multi-seizure blocks. |
| Feature columns misaligned across cases | Canonicalisation not applied, or applied by position instead of by label. |
| Gamma band power dominates everything | Notch filter missing or applied after windowing. |
| Model performs worse than Dummy on PR-AUC | Scaler fit on the wrong split, or labels shifted by one window. |
| Databricks compute dies mid-run | Quota. Reduce to `fast.yaml`, verify per-fold checkpointing is writing. |
| `mne` warnings flood the log | `verbose=False` on read, and set `mne.set_log_level("ERROR")` once at startup. |
| Extraction slower than expected | Welch `nperseg` too large, or `preload=True` missing causing repeated disk reads. |

---

## 15. Config schema (`configs/base.yaml`)

```yaml
data:
  root: ./data/chbmit
  cases: all                    # or explicit list
  on_missing_channel: skip_file # skip_file | fill_nan

signal:
  sample_rate: 256
  notch_hz: [60, 120]
  bandpass: [0.5, 80.0]
  canonical_channels: []        # populated at M3, frozen thereafter

windowing:
  length_sec: 10.0
  stride_sec: 10.0
  min_overlap_frac: 0.5
  guard_sec: 30

features:
  bands:
    delta: [0.5, 4.0]
    theta: [4.0, 8.0]
    alpha: [8.0, 13.0]
    beta:  [13.0, 30.0]
    gamma: [30.0, 80.0]
  welch_nperseg: 512
  welch_noverlap: 256
  views: [per_channel, aggregated]

splits:
  protocol: loso                # loso | random_window
  n_val_subjects: 4
  seed: 42

training:
  models: [dummy, logreg, rf, hgb]
  negative_downsample_ratio: 10 # train fold only; null to disable
  threshold_metric: f2
  consecutive_k: [1, 2, 3]

tracking:
  mlflow_experiment: /Users/<you>/seizure-detection
  local_artifacts: ./artifacts
```

Config is hashed (stable JSON serialisation → sha256, first 12 chars) and the hash is written into every derived table so cached features are never silently mismatched with the config that would produce them.

---

## 16. Attribution

Cite per the Open Data Commons Attribution License:

> Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD thesis, MIT.
>
> Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220.

Include this in the README and in `docs/data-notes.md`.

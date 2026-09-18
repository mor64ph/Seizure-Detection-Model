# Cross-Patient Seizure Detection on CHB-MIT

Window-level seizure detection on scalp EEG, evaluated under leave-one-subject-out
cross-validation. Built to measure honestly what a classical ML pipeline can and cannot do on
patients it has never seen.

**[Live demo →](https://eeg-seizure-detection-model.streamlit.app)** · [Full evaluation report](artifacts/report.md) · [Project rules](../RULES.md)

> **Not a medical device.** Research code on a public research dataset. No regulatory clearance;
> not for use in patient care. On held-out patients it detected 71 of 185 seizures and found
> nothing at all for 7 of 23 patients.

---

## Results

Leave-one-subject-out, 23 folds, 979.9 hours, 352,742 windows at a 0.32% seizure rate.

| Model | PR-AUC | ROC-AUC | Precision (pooled) | Seizures | FA/h | Time in alarm |
|---|---|---|---|---|---|---|
| `dummy` | 0.004 | 0.500 | 0.0032 | 185 / 185 | 0.89 | **1.000** |
| `logreg` | 0.164 | 0.798 | 0.0043 | 55 / 185 | 4.08 | 0.243 |
| `rf` | 0.178 | 0.833 | 0.0047 | **71 / 185** | 4.36 | 0.261 |
| `hgb` | **0.205** | 0.843 | 0.0044 | 59 / 185 | **3.52** | 0.256 |

`rf` is the designated configuration, chosen on event-level results. Validation-only selection
would have picked `hgb`, so the `rf` figures are an optimistically biased estimate (see R44).

**Read the `dummy` row.** It predicts "no seizure" for every window, yet reports 185 of 185
seizures at a better false-alarm rate than any real model — because false alarms are counted
per *run*, and a detector that never stops alarming logs one long alarm. Its 100% time in alarm
is what exposes it. This is why a deliberately useless baseline is mandatory, and why FA/h is
never reported without time in alarm beside it.

Per-patient results are bimodal: reliable for 4 of 23 patients, no detections for 7, and 3 more
reach full sensitivity only by alarming almost continuously. Pooled percentages hide that — see
[`artifacts/report.md`](artifacts/report.md) for the per-patient table.

**Measured data leakage.** Same model and features, only the split protocol differs:

| Model | Grouped by patient | Windows shuffled | Inflation |
|---|---|---|---|
| `hgb` | 0.205 | 0.388 | **+0.182** |
| `rf` | 0.178 | 0.339 | +0.161 |
| `logreg` | 0.164 | 0.068 | −0.096 |

A random split inflates PR-AUC by up to 0.18 because adjacent windows from one patient are
near-duplicates. Logistic regression gains nothing, so inflation is a property of model
capacity, not of the dataset alone.

---

## Quick start

```bash
git clone https://github.com/mor64ph/Seizure-Detection-Model.git
cd Seizure-Detection-Model
pip install -r requirements.txt
```

Run the demo app — no dataset required, results and model ship with the repo:

```bash
cd seizure-detection
streamlit run app/streamlit_app.py
```

Reproduce the evaluation from the committed artifacts:

```bash
make report        # regenerates artifacts/report.md from run artifacts
make test          # 142 tests
```

Rebuild from raw data (downloads 45.76 GB from PhysioNet, streamed and deleted per file):

```bash
make metadata      # 2.19 MB of summaries, annotations and checksums
make manifest      # header-only pre-pass over 686 files (~4.2 MB)
make labels        # parse both label sources, cross-validate to 198 seizures
make ingest        # download -> verify SHA-256 -> featurise -> delete raw
make train         # LOSO plus the random-split contrast
make report
```

---

## Usage

```bash
python -m seizure.cli --config configs/base.yaml <command>
```

| Command | Purpose |
|---|---|
| `ingest` | streaming download, verify, featurise, delete raw |
| `train --view {aggregated,per_channel} [--contrast]` | LOSO evaluation |
| `report` | regenerate `artifacts/report.md` and `report.json` |
| `sensitivity` | ictal-threshold sweep (PRD §6.2) |
| `csv-study` | four-protocol leakage study on the companion CSV |
| `export-scores --subjects sub01,sub12` | per-window held-out scores |
| `fit-final --view aggregated` | fit on all subjects and persist for inference |

Configs in [`configs/`](configs/): `base.yaml` is the real run, `fast.yaml` is 5 cases for
iteration, and the `amplitude_*`/`joint_k`/`search_on`/`select_event` variants reproduce
specific experiments.

---

## How it works

| Stage | Detail |
|---|---|
| Data | CHB-MIT Scalp EEG v1.0.0 (PhysioNet) — 23 subjects, 24 cases, 686 EDF files, 198 seizures |
| Channels | 18 canonical bipolar derivations, frozen by a header-only pre-pass |
| Conditioning | 60/120 Hz notch, 0.5–80 Hz bandpass |
| Windowing | 10 s non-overlapping; 30 s guard band around seizure boundaries, excluded from training, scoring and the false-alarm denominator |
| Features | 14 per channel per window → 252 per-channel or 56 aggregated |
| Models | `DummyClassifier`, logistic regression, random forest, histogram gradient boosting |
| Splits | leave-one-subject-out grouped on `subject_id`; a random-window split as a deliberate contrast |
| Operating point | threshold and consecutive-window count chosen jointly on validation subjects |

Ingest streams one file at a time — download, verify, featurise, delete — so the 45.76 GB
dataset processes under a 26 GB disk ceiling. All 686 files passed SHA-256 with zero failures.

---

## Project structure

```
RULES.md                    44 binding rules; read before changing anything
PRD-seizure-detection.md    original specification
requirements.txt            app runtime dependencies
seizure-detection/
  src/seizure/              pipeline: ingest, labels, signal, features, splits, eval, models
  app/streamlit_app.py      three-page demo application
  configs/                  experiment configurations
  artifacts/                committed results, report, fitted model
  samples/                  bundled demo recording
  scripts/                  sample-generation utilities
  tests/                    142 tests
  docs/data-notes.md        every discrepancy found in the source data
```

---

## Limitations

- **Bimodal performance.** Reliable for 4 of 23 held-out patients, no detections for 7. The
  pooled 38% is an average over that, not a consistent hit rate.
- **Low window precision.** Pooled 0.0037 — roughly 1 flagged window in 270 is ictal. The
  consecutive-window rule suppresses this at the event level.
- **20% time in alarm.** Count-based false-alarm rates hide this; both are reported.
- **Fixed input format.** Requires 18 canonical bipolar channels at 256 Hz. Other montages are
  refused, not adapted.
- **Population.** Trained on paediatric presurgical monitoring. Behaviour on adults, other
  hardware or ambulatory recordings is unmeasured.
- **Not run in the cloud.** Databricks and MLflow code paths exist and have never executed
  against a live service.

---

## Development

```bash
make test          # pytest, 142 tests
make lint          # ruff
```

[`RULES.md`](../RULES.md) is binding and supersedes the PRD where they conflict. Each rule
records a measurement or a defect that motivated it; Appendix C records which results
overturned earlier conclusions.

---

## Attribution

Built on the CHB-MIT Scalp EEG Database, distributed by PhysioNet under the
[Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).
Cite all three:

> Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). *PhysioNet.*
> https://doi.org/10.13026/C2K01R
>
> Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and
> Treatment.* PhD thesis, MIT.
>
> Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*
> 101(23):e215–e220.

Code is [MIT licensed](../LICENSE); the licence covers the code only, not the dataset.

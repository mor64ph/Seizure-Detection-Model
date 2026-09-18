# Cross-Patient Seizure Detection on CHB-MIT

[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue)](seizure-detection/pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen)](#testing)

A seizure-detection pipeline for scalp EEG that measures how well it works on patients it has
never seen — built for ML practitioners and researchers who need an honest cross-subject
baseline rather than a headline number.

[Live demo](https://eeg-seizure-detection-model.streamlit.app) ·
[Full evaluation report](seizure-detection/artifacts/report.md) ·
[Project rules](RULES.md) ·
[Specification](PRD-seizure-detection.md)

> **Not a medical device.** Research code on a public research dataset, with no regulatory
> clearance, not for use in patient care. On held-out patients it detected 71 of 185 seizures
> and found nothing at all for 7 of 23 patients.

---

## Contents

- [Overview](#overview)
- [Results](#results)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

Most published seizure-detection results are reported on data splits that let the same patient
appear in both training and test. Because consecutive EEG windows from one patient are
near-duplicates, this inflates scores substantially without the model learning anything
transferable.

This project evaluates under leave-one-subject-out cross-validation and reports the gap
directly: the same model and features score PR-AUC 0.388 on a shuffled split and 0.205 when
every patient is confined to one fold. Alongside that, it reports per-patient outcomes,
event-level detection, and alarm duration — figures that pooled averages conceal.

## Results

Leave-one-subject-out, 23 folds, 979.9 hours, 352,742 windows at a 0.32% seizure rate.

| Model | PR-AUC | ROC-AUC | Precision (pooled) | Seizures | FA/h | Time in alarm |
|---|---|---|---|---|---|---|
| `dummy` | 0.004 | 0.500 | 0.0032 | 185 / 185 | 0.89 | **1.000** |
| `logreg` | 0.164 | 0.798 | 0.0043 | 55 / 185 | 4.08 | 0.243 |
| `rf` | 0.178 | 0.833 | 0.0047 | **71 / 185** | 4.36 | 0.261 |
| `hgb` | **0.205** | 0.843 | 0.0044 | 59 / 185 | **3.52** | 0.256 |

The `dummy` row predicts "no seizure" for every window, yet reports 185 of 185 seizures at a
better alarm rate than any real model — because false alarms are counted per run, and a
detector that never stops alarming logs one long alarm. Its 100% time in alarm is what exposes
it; FA/h is therefore never reported without it.

**Measured leakage.** Same model and features, only the split protocol differs:

| Model | Grouped by patient | Windows shuffled | Inflation |
|---|---|---|---|
| `hgb` | 0.205 | 0.388 | **+0.182** |
| `rf` | 0.178 | 0.339 | +0.161 |
| `logreg` | 0.164 | 0.068 | −0.096 |

Logistic regression gains nothing from the leaky split, so inflation is a property of model
capacity rather than of the dataset alone.

**Per-patient outcomes are bimodal:** reliable for 4 of 23 patients, no detections for 7, and 3
more reach full sensitivity only by alarming almost continuously. The per-patient table is in
[`artifacts/report.md`](seizure-detection/artifacts/report.md).

## Features

**Pipeline**

- Streaming ingest of 45.76 GB under a 26 GB disk ceiling — download, SHA-256 verify,
  featurise, delete raw, checkpoint, per file
- Labels parsed from two independent sources and cross-validated to 198 seizures
- Channel allowlist frozen by a header-only pre-pass over all 686 files
- Three-valued labels with a 30 s guard band excluded from training, scoring and the
  false-alarm denominator

**Evaluation**

- Leave-one-subject-out grouped on patient, plus a random-window split as a deliberate contrast
- Event-level metrics pooled rather than averaged as ratios
- Precision reported both pooled and per-fold, which differ by 62x
- Alarm duration reported beside alarm counts
- Threshold and consecutive-window count selected jointly on validation subjects

**Application**

- Upload an EDF and get a detection timeline, flagged segments, band powers and raw traces
- Operating-point simulation with sweep curves computed from the uploaded recording
- Recordings matching an annotated CHB-MIT record are scored against ground truth; others are
  labelled unverifiable
- Format gate refuses recordings that do not match the trained montage rather than adapting them

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Modelling | scikit-learn (random forest, histogram gradient boosting, logistic regression) |
| Signal processing | SciPy, MNE |
| Data | pandas, PyArrow, NumPy |
| Config | Pydantic, PyYAML |
| Application | Streamlit, Plotly |

## Getting started

### Prerequisites

| Requirement | Detail |
|---|---|
| Python | 3.11 or newer (`requires-python` in `seizure-detection/pyproject.toml`) |
| Disk, demo only | ~50 MB |
| Disk, full rebuild | 26 GB free or more; the pipeline streams 45.76 GB and deletes as it goes |
| Accounts or API keys | none |

### Installation

```bash
git clone https://github.com/mor64ph/Seizure-Detection-Model.git
cd Seizure-Detection-Model
pip install -r requirements.txt
```

### Configuration

Experiment settings live in [`seizure-detection/configs/`](seizure-detection/configs/);
`base.yaml` is the reference run. One environment variable is read at runtime:

| Variable | Default | Purpose |
|---|---|---|
| `SEIZURE_MAX_DECODE_MB` | `700` | Peak memory budget for decoding an uploaded EDF. Files whose estimated decode cost exceeds it are refused rather than risking an out-of-memory kill. |

### Running

The demo needs no dataset — results and the fitted model ship with the repository:

```bash
cd seizure-detection
streamlit run app/streamlit_app.py
```

## Usage

```bash
cd seizure-detection
python -m seizure.cli --config configs/base.yaml <command>
```

| Command | Purpose |
|---|---|
| `ingest` | stream download, verify, featurise, delete raw |
| `train --view {aggregated,per_channel} [--contrast]` | leave-one-subject-out evaluation |
| `report` | regenerate `artifacts/report.md` and `report.json` |
| `sensitivity` | ictal-threshold sweep |
| `csv-study` | four-protocol leakage study on the companion CSV |
| `export-scores --subjects sub01,sub12` | per-window held-out scores |
| `fit-final --view aggregated` | fit on all subjects and persist for inference |

Reproduce the published evaluation from committed artifacts:

```bash
make report
make test
```

Rebuild from raw data, which downloads 45.76 GB from PhysioNet:

```bash
make metadata      # 2.19 MB of summaries, annotations and checksums
make manifest      # header-only pre-pass over 686 files
make labels        # parse both label sources, cross-validate to 198 seizures
make ingest        # download, verify, featurise, delete raw
make train         # LOSO plus the random-split contrast
make report
```

## How it works

| Stage | Detail |
|---|---|
| Data | CHB-MIT Scalp EEG v1.0.0 — 23 subjects, 24 cases, 686 EDF files, 198 seizures |
| Channels | 18 canonical bipolar derivations |
| Conditioning | 60/120 Hz notch, 0.5–80 Hz bandpass |
| Windowing | 10 s non-overlapping, 30 s guard band |
| Features | 14 per channel per window, giving 252 per-channel or 56 aggregated |
| Models | `DummyClassifier`, logistic regression, random forest, histogram gradient boosting |
| Splits | leave-one-subject-out grouped on `subject_id`, plus a random-window contrast |
| Operating point | threshold and consecutive-window count chosen jointly on validation subjects |

All 686 files passed SHA-256 verification with zero failures.

## Project structure

```
RULES.md                    44 binding rules; read before changing anything
PRD-seizure-detection.md    original specification
requirements.txt            application runtime dependencies
seizure-detection/
  src/seizure/              ingest, labels, signal, features, splits, eval, models
  app/streamlit_app.py      three-page demo application
  configs/                  experiment configurations
  artifacts/                committed results, report, fitted model
  samples/                  bundled demo recording
  scripts/                  sample-generation utilities
  tests/                    142 tests
  docs/data-notes.md        discrepancies found in the source data
```

## Testing

```bash
cd seizure-detection
make test          # pytest, 142 tests
make lint          # ruff
```

The suite covers EDF header parsing against golden bytes, the deletion gate that blocks
raw-file removal until every ledger field is captured, label cross-validation, feature
extraction, window and event metrics, split protocols, and the Spark-portability scanner.
Several tests pin defects that were found and fixed, so they fail if a defect returns.

## Deployment

The application is deployed on Streamlit Community Cloud from `main`. Two files must sit at the
repository root, because Streamlit Cloud resolves both relative to the working directory rather
than to the entrypoint:

- `requirements.txt`
- `.streamlit/config.toml`

Entrypoint: `seizure-detection/app/streamlit_app.py`.

## Limitations

- **Bimodal performance.** Reliable for 4 of 23 held-out patients, no detections for 7. The
  pooled 38% is an average over that, not a consistent hit rate.
- **Low window precision.** Pooled 0.0037, so roughly 1 flagged window in 270 is ictal. The
  consecutive-window rule suppresses this at the event level.
- **20% time in alarm.** Count-based alarm rates hide this; both are reported.
- **Fixed input format.** Requires 18 canonical bipolar channels at 256 Hz. Other montages are
  refused, not adapted.
- **Population.** Trained on paediatric presurgical monitoring. Behaviour on adults, other
  hardware or ambulatory recordings is unmeasured.
- **No cloud execution.** Databricks and MLflow code paths exist and have never run against a
  live service.

## Roadmap

- [x] Streaming ingest of the full 686-file dataset
- [x] Leave-one-subject-out evaluation with a leakage contrast
- [x] Event-level metrics with pooled precision and alarm duration
- [x] Joint threshold and run-length selection
- [x] Interactive analysis application
- [ ] Validation-only selection across feature subsets, to test whether a reduced feature set
      can be promoted without selection bias
- [ ] Per-record baseline normalisation for cross-patient transfer
- [ ] Execute the portability notebook against a live Spark cluster

## Contributing

Issues and pull requests are welcome. Before changing pipeline behaviour, read
[`RULES.md`](RULES.md) — it is binding, supersedes the specification where they conflict, and
each rule records the measurement or defect that motivated it. Run `make test` and `make lint`
before opening a pull request.

## License

[MIT](LICENSE). The licence covers the code in this repository only. The CHB-MIT Scalp EEG
Database is not redistributed here; the pipeline downloads it from PhysioNet at run time under
its own terms.

## Acknowledgements

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

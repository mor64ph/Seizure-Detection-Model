# Seizure Detector Report Card

```bash
cd seizure-detection
pip install -r ../requirements.txt
streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

An upload-driven analysis tool. **Every figure on every page is computed from the recording
the user supplies** — nothing is precomputed from the study. One analysis is held in
`st.session_state["analysis"]` and shared by all three pages.

| Page | Content |
|---|---|
| Analyse | upload, headline figures for that file, model card |
| Signal and detections | score trace, flagged segments, band powers over time, raw montage |
| Operating point | threshold/k simulation plus sweep curves for that file |

## Accuracy needs labels

An arbitrary EDF has no annotation, so sensitivity, false alarms and latency are *not
computable* for it. Rather than invent them, the app resolves three states from the filename
against `artifacts/seizure_intervals.json` (committed) and the CHB-MIT record-id pattern:

| `kind` | Meaning | What is shown |
|---|---|---|
| `annotated` | matches a record with annotated seizures | full accuracy against ground truth |
| `annotated_seizure_free` | matches a CHB-MIT record with no seizures | every flagged segment is a false alarm; flagging none is the correct result |
| `unverifiable` | no pattern match | detection output only, labelled unverifiable |

The middle state matters: a seizure-free record is *absent* from the intervals file, so
membership alone cannot tell "no annotation" from "annotated as having no seizure". Gate
annotation-dependent output on `a["recognised"]`, never on a non-empty `truth` list —
`truth == []` is falsy but meaningful.

## Non-negotiable

- **The refusal gate.** Uploads pass through `features/extract.py`, so a wrong montage,
  wrong sample rate or too-short recording is rejected with the pipeline's own reason.
  Without it the model scores out-of-distribution input and returns confident nonsense.
  Read `res.skipped_reason` **directly**; a defaulted `getattr` silently disabled it once.
- **The one-line research-use footer.**

## Traps that cost real debugging

- **`icon=` must be a single emoji or `":material/name:"`.** `icon="✓"` and `icon="!"` both
  raise `StreamlitAPIException` — and only on the success path, so the app looked fine until
  a recording was actually loaded.
- **Never put two `st.*` calls in a multi-line ternary.** Streamlit introspects the caller's
  source line to name the element and the parse fails with a `SyntaxError` thrown from
  inside Streamlit.
- **`AppTest` cannot drive `file_uploader`.** Inject `at.session_state["analysis"]` with a
  real analysis dict instead; testing only the empty state misses every populated-path bug.
- KPI cards are hand-rolled HTML, so `AppTest` reports `metric=0`. Assert on `at.markdown`.
- `.streamlit/config.toml` and `requirements.txt` must both live at the **repo root** —
  Streamlit reads `$CWD/.streamlit/config.toml` and Cloud runs from the root regardless of
  where the entrypoint sits. Misplaced, they are ignored silently.

## Upload size is limited by RAM, not by the config number

`maxUploadSize = 200` in the root `.streamlit/config.toml`, but that number is not the real
constraint. EDF stores int16 and mne preloads float64, so decoding costs ~4x the file size
with the raw bytes resident too — about **5x peak**:

| file | hours (18 ch) | peak decode |
|---|---|---|
| 51 MB (one CHB-MIT record) | 1.6 | ~255 MB |
| 120 MB | 3.8 | ~600 MB |
| 200 MB | 6.3 | **~1000 MB** |

Streamlit Community Cloud's free tier is ~1 GB, with roughly 300 MB already taken by Python,
the model and Streamlit. So `analyse()` estimates the peak **before** decoding and refuses with
the arithmetic shown, because an OOM kill surfaces to a visitor as an unexplained connection
error. Raise `SEIZURE_MAX_DECODE_MB` (default 700) when hosting somewhere larger.

The stored montage for the raw-trace view is decimated to a **byte budget**
(`DISPLAY_BUDGET_MB = 40`), not to a fixed rate — at a fixed 64 Hz a six-hour upload would add
100 MB of its own. A one-hour file displays at 64 Hz, six hours at ~26 Hz, and below 16 Hz the
trace is dropped as too coarse to be worth drawing.

## Performance

`analyse()` is cached on the file bytes, so re-selecting a file is free and the result
survives navigation between pages. The model always scores the full 256 Hz signal; only the
stored copy used for drawing traces is decimated, to the byte budget described above.

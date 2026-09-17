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

## Performance

`analyse()` is cached on the file bytes, so re-selecting a file is free and the result
survives navigation. The decoded montage is stored at 64 Hz — a sixteenth of the memory of
the 256 Hz original, still ample for drawing traces. The model scores the full-rate signal.

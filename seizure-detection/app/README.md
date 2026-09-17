# The honest-evaluation explorer

```bash
cd seizure-detection
python -m pip install streamlit plotly
python -m seizure.cli --config configs/base.yaml export-scores \
    --view aggregated --subjects sub01,sub12,sub17     # once, for tab 2
streamlit run app/streamlit_app.py
```

## What this is

A four-tab demonstration whose subject is **the evaluation, not the patient**.

| Tab | The finding it makes visible |
|---|---|
| 1 · Who it works for | a 30–38% average hides a bimodal failure; four patients are genuine successes, seven get nothing, three only *look* perfect |
| 2 · One record, one operating point | an operating point is a trade — drag the threshold, drop `k`, watch false alarms multiply |
| 3 · Why the split matters | one toggle, same model, PR-AUC 0.388 → 0.205 |
| 4 · Try an unseen recording | upload an `.edf` and watch the detector run — or be refused |

Tab 1 is the centrepiece. The three red bars reach the top of the chart while detecting
nothing real: they alarm almost continuously and are credited for the overlap. That is why
event sensitivity is never shown here without precision beside it.

## Tab 4 — the upload path

It accepts a research `.edf`, runs the real pipeline over it, and shows the flagged segments.
Three properties make that defensible:

1. **The refusal gate is the pipeline's, not the app's.** The file goes through
   `features/extract.py`, so a recording lacking the 18 canonical channels, or not at 256 Hz,
   or shorter than one window, is rejected with the real reason. Verified against
   `chb12_27`/`_28`/`_29` — the three records the project genuinely could not read.
   Read `res.skipped_reason` **directly**; a `getattr(res, ..., None)` with a default silently
   disabled the entire gate once already.
2. **The output is "candidate segments", never "seizures".** An uploaded file has no expert
   annotation, so nothing in it can be confirmed or refuted, and the app says so.
3. **The LOSO expectation is printed above the uploader, not below the result** — 71 of 185
   seizures, 4.36 false alarms/hour, and seven patients with nothing detected at all. A reader
   meets the limitation before they meet the output.

The model is `artifacts/final_model_aggregated.joblib`, fitted on all 23 subjects by
`seizure fit-final`. It has **no held-out score of its own and never can** — no subject is
left to hold out — so the LOSO figures travel inside the artifact and are surfaced in the UI.
Its operating point is the median threshold and modal k across the LOSO folds, which is a
compromise, not a solution: the per-fold threshold ranged 0.000 to 0.994.

## What it will not do

- **No verdict about any person.**
- **No precautionary or medical guidance.** Seizure action plans come from a treating
  neurologist. The app links to the Epilepsy Foundation, NHS and ILAE instead.
- **No consumer-headset input.** A 2–8 channel wearable is out of distribution for a model
  trained on 18 canonical bipolar derivations at 256 Hz; the output would be confident noise.

The upload path in tab 4 is for **research recordings**. Do not upload identifiable patient
data to a demonstration app, and do not present its output to anyone as a finding about a
person. The refusal gate is deliberately wired to the pipeline's own channel and rate checks
rather than to a disclaimer, so an unreadable file is rejected by the same code that skipped
three `chb12` records during extraction.

Passing that gate is not the same as being in distribution: a different amplifier, referencing
scheme, age group or clinical context all shift the features, and the model cannot report that
it is out of its depth — it returns confident scores regardless. The UI states this.

## Notes for whoever maintains it

- It reads the frozen snapshot in `artifacts/rfjoint_base/`, **not** the live `artifacts/`
  directory. Training runs rewrite `detail_*.json` in place, and a reader should not have the
  numbers change under them mid-session. Re-point `SNAP` when you promote a new run.
- Tab 2 needs `artifacts/scores_aggregated.parquet` from `export-scores`. The pipeline keeps
  only aggregate metrics, so per-window scores have to be exported deliberately. Each subject
  is scored by the model fitted on **its own LOSO fold**, so these are held-out predictions and
  the slider's default position is the operating point the pipeline actually chose.
- Raw EEG traces only exist for `chb01`, `chb12` and `chb17` — the R13 fixtures kept on disk.
  Every other record's `.edf` was deleted after feature extraction, and the app says so rather
  than failing.
- The headline banner reprints the selection-bias caveat whenever
  `training.headline_model` overrides automatic selection (R44). Do not remove it while the
  override stands.

# Seizure Detector Report Card

*Does it work on a patient it has never seen?*

```bash
cd seizure-detection
pip install -r ../requirements.txt
streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Five sections, sidebar-navigated. Tabs 1–4 need no raw data; the small result files they
read are committed.

| Section | What it shows |
|---|---|
| Overview | KPI strip, four findings, and the full **model card** |
| Per-patient outcomes | 23 patients graded Reliable / Partial / Over-alarming / No detections |
| Record explorer | threshold and run-length sliders redrawing alarms live |
| Validation protocol | grouped-by-patient against shuffled-windows, same model |
| Analyse a recording | upload an EDF and run the real pipeline over it |

## Design rules

**The evaluation is the product.** Limitations are presented as model documentation, not as
warning boxes. A model card reads as engineering maturity; a wall of red alerts reads as
hedging. Same information, different register.

**Two things are functional, not cosmetic, and must survive any redesign:**

1. **The refusal gate** in `page_analyse`. Uploaded files pass through
   `features/extract.py`, so a recording lacking the 18 canonical channels, or not at 256 Hz,
   or shorter than one window, is rejected with the pipeline's own reason. Without it the
   model silently scores out-of-distribution input and returns confident nonsense — which is
   both unsafe and a worse product than one that says "unsupported format". Read
   `res.skipped_reason` **directly**; a `getattr(res, ..., None)` with a default silently
   disabled the whole gate once already.
2. **The one-line research-use footer.** Minimum for software that ingests EEG and talks
   about seizures. It is small print, not a banner, and that is the right size for it.

## Implementation notes

- Reads the frozen snapshot in `artifacts/rfjoint_base/`, **not** live `artifacts/`. Training
  runs rewrite `detail_*.json` in place and a reader should not have numbers move mid-session.
  Re-point `SNAP` when promoting a new run.
- KPI cards are hand-rolled HTML rather than `st.metric`, for control over typography and
  density. Consequence: `AppTest` reports `metric=0` — assert against `at.markdown` instead.
- The Record explorer needs `artifacts/scores_aggregated.parquet` from
  `seizure export-scores`. Each patient is scored by the model fitted on **its own LOSO
  fold**, so those are held-out predictions and the slider's default is the operating point
  the pipeline actually chose.
- Raw EEG traces exist only for `chb01`, `chb12` and `chb17`. Everywhere else the view
  explains that recordings are streamed and deleted after feature extraction.
- The Overview headline model follows `report.json`'s `selected_model`, which honours
  `training.headline_model` (R44).

## Testing

`AppTest` drives every section. Identify widgets **by label**, not index — the main-content
radio precedes the sidebar radio in the element tree, so positional access silently grabs the
wrong widget:

```python
radio = next(r for r in at.radio if "consecutive" in r.label.lower())
```

Interaction behaviour worth preserving, measured on `chb01_03`: at the tuned operating point
(threshold 0.125, k=3) the record shows 3 false alarms and 4.2% time in alarm; dropping k to 1
gives 22 and 11.7%; at threshold 0.02 it is 62 and 70.6%. Detection stays at 1 throughout.
That progression is the whole point of the section.

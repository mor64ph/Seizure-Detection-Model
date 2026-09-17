# Sample recordings

## `sample_chb01_03_2880-3180s.edf` — committed, and the one to demo with

Five minutes of real scalp EEG cut from CHB-MIT `chb01_03`, seconds 2880–3180 of the full
record, containing one expert-annotated seizure. 2.64 MB, 18 canonical bipolar channels,
256 Hz.

Regenerate with `python scripts/make_sample_excerpt.py` (needs `chb01` on disk — it is an R13
permanent fixture, so `make ingest` keeps it).

Measured at the shipped operating point (threshold 0.268, k=3):

| | |
|---|---|
| Seizures detected | **1 / 1** |
| False alarms | **0** |
| Latency | 4 s from annotated onset |
| Time in alarm | 16.7% |
| Mean score, seizure window | **0.637** (max 0.960) |
| Mean score, background | **0.028** |

The excerpt's clock restarts at zero, so the annotated span sits at **116–156 s**, not
2996–3036 s. The app cannot recover that from the filename, so it is registered in
`BUNDLED_SAMPLES` in `app/streamlit_app.py`. **Renaming this file without updating that entry
silently loses the ground truth** and the demo degrades to "unverifiable".

Redistribution is permitted: CHB-MIT is published by PhysioNet under the Open Data Commons
Attribution License, and the attribution is in `LICENSE` and the app footer.

## `synthetic_demo.edf` — not committed, and not a detection demo

`python scripts/make_sample_edf.py` writes a fully synthetic 300 s recording with the exact
montage and rate the refusal gate demands. It exists to exercise the **upload path** — it is
how the `allow_unknown_record` crash was found — and it is deliberately not named like a
CHB-MIT record, so it lands in the `unverifiable` branch.

**The model does not detect its injected event, and that is the honest result.** Measured:

| | mean score |
|---|---|
| Background | 0.259 |
| Injected "seizure", 182–218 s | **0.151** |

The event scores *below* background. A first attempt used a clean 3.4 Hz sinusoid; a second
added the properties real seizures have — frequency evolving 11 Hz → 3 Hz, spike-and-wave
morphology, amplitude several times background, postictal suppression — and the score got
slightly worse, not better.

It was not tuned further. Adjusting synthetic data until the classifier fires would
manufacture a demo that proves nothing, and the negative result is itself informative: a
signal that looks plausibly ictal to a human does not look ictal to a model trained on real
EEG. Plausible suspects are the absence of inter-channel correlation (synthetic channels are
independent; real EEG is strongly coupled) and power concentrated in one narrow evolving band
where real seizures are broadband. Neither was chased, because the goal was a test file, not
a rigged one.

## A trap this cost real time

`mne` returns EDF data in **volts**, not microvolts, while the writer declares a physical
dimension of µV. The first excerpt divided volts by 0.2 µV/LSB, rounded every sample to zero,
and produced a flat line that the model scored identically (0.112) across the whole file
including the seizure. `make_sample_excerpt.py` now asserts the round trip — correlation
> 0.999 and a standard-deviation ratio within 1% of the source — and deletes its own output
rather than shipping a silently destroyed signal.

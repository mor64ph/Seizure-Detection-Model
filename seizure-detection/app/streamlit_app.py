"""An honest-evaluation explorer for the CHB-MIT seizure detector.

The subject of this app is the *evaluation*, not the patient. It exists to make
three measured findings legible to someone who will not read the report:

  1. a 38% average hides a bimodal failure -- under the designated rf + joint
     configuration the detector genuinely works on four of twenty-three
     children, fails completely on seven, and only *looks* perfect on three
  2. an operating point is a trade, and you can feel it by dragging it
  3. the honest split costs half the headline score, and one toggle shows it

It deliberately does not accept patient data and does not produce a verdict
about any person. See app/README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seizure.eval import metrics as M  # noqa: E402

# The frozen snapshot, not the live artifacts directory. Training runs rewrite
# `detail_*.json` in place, so pointing the app at it would let the numbers
# change under a reader mid-session.
SNAP = ROOT / "artifacts" / "rfjoint_base"
SCORES = ROOT / "artifacts" / "scores_aggregated.parquet"
DATA = ROOT / "data" / "chbmit"

ZERO, SATURATED, GENUINE, PARTIAL = "zero", "saturated", "genuine", "partial"
COLOURS = {
    GENUINE: "#2e7d32",
    PARTIAL: "#f9a825",
    SATURATED: "#c62828",
    ZERO: "#9e9e9e",
}
# A subject whose every seizure is "detected" at this precision or below is not
# detecting, it is alarming continuously and being credited for the overlap.
SATURATION_PRECISION = 0.01

st.set_page_config(page_title="Seizure detection — honest evaluation",
                   layout="wide")


@st.cache_data
def load_detail() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = json.loads((SNAP / "detail_aggregated.json").read_text(encoding="utf-8"))
    rep = json.loads((SNAP / "report.json").read_text(encoding="utf-8"))
    return pd.DataFrame(d["loso"]), pd.DataFrame(d["random_window"]), rep


@st.cache_data
def load_scores() -> pd.DataFrame:
    if not SCORES.exists():
        return pd.DataFrame()
    return pd.read_parquet(SCORES)


@st.cache_data
def load_truth() -> dict[str, list[tuple[int, int]]]:
    p = ROOT / "artifacts" / "seizure_intervals.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {rec: [(int(a), int(b)) for a, b in spans] for rec, spans in raw.items()}


def classify(row) -> str:
    if row.n_true_events == 0:
        return PARTIAL
    if row.n_detected == 0:
        return ZERO
    if row.n_detected >= row.n_true_events:
        return GENUINE if row.precision > SATURATION_PRECISION else SATURATED
    return PARTIAL


# ---------------------------------------------------------------- screen one
def screen_bimodality(loso: pd.DataFrame, headline: str) -> None:
    st.header("Where it works, and where it does not")
    g = loso[loso.model == headline].copy()
    g = g[g.n_true_events > 0]
    g["category"] = g.apply(classify, axis=1)
    g["frac"] = g.n_detected / g.n_true_events
    g = g.sort_values(["frac", "test_subject"])

    counts = g.category.value_counts()
    c = st.columns(4)
    c[0].metric("Genuine detection", int(counts.get(GENUINE, 0)),
                help="every seizure found, and the alarms were mostly real")
    c[1].metric("Partial", int(counts.get(PARTIAL, 0)))
    c[2].metric("Saturated", int(counts.get(SATURATED, 0)),
                help=f"every seizure 'found' at precision <= {SATURATION_PRECISION}"
                     " — this is continuous alarming, not detection")
    c[3].metric("Zero detections", int(counts.get(ZERO, 0)))

    fig = go.Figure()
    for cat in (GENUINE, PARTIAL, SATURATED, ZERO):
        s = g[g.category == cat]
        if not len(s):
            continue
        fig.add_bar(
            y=s.test_subject, x=s.frac, orientation="h", name=cat,
            marker_color=COLOURS[cat],
            customdata=np.stack([s.n_detected, s.n_true_events,
                                 s.precision, s.fa_per_hour], axis=-1),
            hovertemplate=("<b>%{y}</b><br>detected %{customdata[0]} of "
                           "%{customdata[1]}<br>window precision "
                           "%{customdata[2]:.4f}<br>%{customdata[3]:.2f} "
                           "false alarms/hour<extra></extra>"),
        )
    fig.update_layout(height=620, barmode="stack",
                      xaxis_title="fraction of that patient's seizures detected",
                      xaxis_range=[0, 1.02], margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width='stretch')

    det, true = int(g.n_detected.sum()), int(g.n_true_events.sum())
    st.warning(
        f"Pooled, this is **{det} of {true} seizures ({det / true:.1%})**. Per patient it is "
        f"**{int(counts.get(GENUINE, 0))} genuine successes** out of {len(g)}. "
        f"The {int(counts.get(SATURATED, 0))} red bars reach the top of the chart by alarming "
        "almost continuously — they are the reason event sensitivity must never be read "
        "without precision beside it."
    )


# ---------------------------------------------------------------- screen two
def screen_record(scores: pd.DataFrame, truth: dict) -> None:
    st.header("One record, and the cost of an operating point")
    if not len(scores):
        st.info("No exported scores yet. Generate them with:\n\n"
                "`python -m seizure.cli --config configs/base.yaml export-scores "
                "--subjects sub01,sub12,sub17`")
        return

    subj = st.selectbox("Patient", sorted(scores.subject_id.unique()))
    sub = scores[scores.subject_id == subj]
    recs = sorted(sub.record_id.unique())
    default = next((i for i, r in enumerate(recs) if truth.get(r)), 0)
    rec = st.selectbox("Record", recs, index=default,
                       format_func=lambda r: f"{r}{'  ·  has seizure' if truth.get(r) else ''}")

    g = sub[sub.record_id == rec].sort_values("window_idx")
    chosen_thr = float(g.chosen_threshold.iloc[0])
    chosen_k = int(g.chosen_k.iloc[0])

    c1, c2 = st.columns([3, 1])
    thr = c1.slider("Alarm threshold", 0.0, 1.0, chosen_thr, 0.005,
                    help=f"the pipeline chose {chosen_thr:.3f} on validation subjects")
    k = c2.radio("Consecutive windows required (k)", [1, 2, 3],
                 index=[1, 2, 3].index(chosen_k) if chosen_k in (1, 2, 3) else 0,
                 horizontal=True)

    pred = (g.score.to_numpy() >= thr)
    runs = M.positive_runs(g.window_idx.to_numpy(), g.t_start_sec.to_numpy(),
                           g.t_end_sec.to_numpy(), pred, k)
    spans = truth.get(rec, [])
    matched = {i for i, r in enumerate(runs)
               if any(r.end_sec > a and r.start_sec < b for a, b in spans)}
    hit = sum(1 for a, b in spans
              if any(r.end_sec > a and r.start_sec < b for r in runs))

    m = st.columns(4)
    m[0].metric("Seizures in record", len(spans))
    m[1].metric("Detected", hit)
    m[2].metric("False alarms", len(runs) - len(matched))
    alarm_sec = sum(r.end_sec - r.start_sec for r in runs)
    total = float(g.t_end_sec.max() - g.t_start_sec.min()) or 1.0
    m[3].metric("Time in alarm", f"{alarm_sec / total:.1%}")

    fig = go.Figure()
    for a, b in spans:
        fig.add_vrect(x0=a, x1=b, fillcolor="#2e7d32", opacity=0.22,
                      line_width=0, layer="below")
    for i, r in enumerate(runs):
        fig.add_vrect(x0=r.start_sec, x1=r.end_sec,
                      fillcolor="#1565c0" if i in matched else "#c62828",
                      opacity=0.30, line_width=0, layer="below")
    fig.add_scatter(x=g.t_start_sec, y=g.score, mode="lines",
                    line=dict(width=1.2, color="#212121"), name="score")
    fig.add_hline(y=thr, line=dict(color="#c62828", dash="dash", width=1))
    fig.update_layout(height=340, xaxis_title="seconds into record",
                      yaxis_title="model score", showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
    st.caption("Green = annotated seizure · blue = alarm overlapping one · "
               "red = false alarm. Drag the threshold, or drop k to 1, and watch "
               "the red blocks multiply.")

    if st.checkbox("Show the raw EEG for this record (slower)"):
        draw_eeg(rec, spans)


@st.cache_data(show_spinner="Reading the .edf …")
def read_eeg(rec: str) -> tuple[np.ndarray, list[str], float] | None:
    import yaml

    from seizure.signal import io as SIO
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    canonical = cfg["signal"]["canonical_channels"]
    case = rec.split("_")[0]
    p = DATA / case / f"{rec}.edf"
    if not p.exists():
        return None
    r = SIO.read(p, canonical, on_missing="skip_file")
    return r.data, r.channels, r.sample_rate


def draw_eeg(rec: str, spans: list[tuple[int, int]]) -> None:
    got = read_eeg(rec)
    if got is None:
        st.info(f"`{rec}.edf` is not on disk. Only chb01, chb12 and chb17 are "
                "kept locally (R13 test fixtures); the rest were deleted after "
                "feature extraction.")
        return
    data, chans, rate = got
    mid = spans[0][0] if spans else 0
    t0 = st.number_input("Window start (s)", 0, int(data.shape[1] / rate) - 20,
                         max(0, int(mid) - 10), step=5)
    span = 30
    a, b = int(t0 * rate), int((t0 + span) * rate)
    seg = data[:, a:b]
    t = np.arange(seg.shape[1]) / rate + t0

    # Fixed offset per channel so the montage reads like clinical paper EEG.
    step = float(np.percentile(np.abs(seg), 99)) * 3 or 1.0
    fig = go.Figure()
    for i, ch in enumerate(chans):
        fig.add_scatter(x=t, y=seg[i] - i * step, mode="lines",
                        line=dict(width=0.7), name=ch, showlegend=False,
                        hoverinfo="skip")
    for s0, s1 in spans:
        if s1 > t0 and s0 < t0 + span:
            fig.add_vrect(x0=max(s0, t0), x1=min(s1, t0 + span),
                          fillcolor="#2e7d32", opacity=0.18, line_width=0,
                          layer="below")
    fig.update_layout(
        height=760, xaxis_title="seconds",
        yaxis=dict(tickmode="array", tickvals=[-i * step for i in range(len(chans))],
                   ticktext=chans, tickfont=dict(size=9)),
        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------- screen four
FINAL_MODEL = ROOT / "artifacts" / "final_model_aggregated.joblib"

# Refusal reasons from features/extract.py, rendered for a human. The gate is
# not app-level validation invented here -- it is the same code path the
# pipeline used to skip three chb12 records, so a file this app rejects is a
# file the project genuinely cannot read.
REFUSALS = {
    "missing_channels": (
        "This recording does not contain all 18 canonical bipolar channels the "
        "model was trained on. Projecting a different montage onto them would "
        "silently feed the model the wrong electrodes."),
    "no_usable_channels": (
        "No channel in this file parses as a scalp-EEG bipolar derivation "
        "between two 10-20 positions."),
    "unexpected_sample_rate": (
        "The model was trained at 256 Hz. A different rate shifts every "
        "frequency-band feature, so the scores would not mean what they say."),
    "record_shorter_than_window": (
        "The recording is shorter than one 10-second analysis window."),
}


@st.cache_resource
def load_final_model():
    if not FINAL_MODEL.exists():
        return None
    import joblib
    return joblib.load(FINAL_MODEL)


@st.cache_data
def load_cfg_obj():
    from seizure.config import load
    return load(ROOT / "configs" / "base.yaml")


def screen_upload() -> None:
    st.header("Run the detector on a recording it has never seen")
    st.error(
        "**This is a research demonstration, not a diagnostic tool, and not a medical "
        "device.** It produces no finding about any person. It cannot tell you whether "
        "someone had a seizure, is having one, or is at risk of one. Nothing here is "
        "medical advice, and it must not inform any decision about anyone's care. "
        "Seizure precautions and seizure action plans come from a treating neurologist — "
        "see the Epilepsy Foundation, NHS, or ILAE.", icon="⛔")

    bundle = load_final_model()
    if bundle is None:
        st.info("No fitted model on disk. Create one with:\n\n"
                "`python -m seizure.cli --config configs/base.yaml fit-final "
                "--view aggregated --loso-detail artifacts/rfjoint_base/detail_aggregated.json`")
        return

    meta = bundle["meta"]
    exp = meta.get("loso_expectation", {})
    op = meta.get("operating_point", {})
    if exp:
        st.warning(
            f"**Before you read anything below.** On the {exp.get('n_subjects')} patients this "
            f"model was evaluated against — each one held out of training entirely — it found "
            f"**{exp.get('seizures_detected')} of {exp.get('seizures_total')} seizures "
            f"({exp.get('event_sensitivity', 0):.0%})** at **{exp.get('fa_per_hour', 0):.2f} false "
            f"alarms per hour, and detected nothing at all for "
            f"{exp.get('subjects_with_zero_detections')} of them.** That is the honest "
            f"expectation for a new recording. Training on all 23 subjects did not make this "
            f"model better; it only removed our ability to measure it."
        )

    up = st.file_uploader(
        "European Data Format recording (.edf)", type=["edf"],
        help="18 canonical bipolar channels at 256 Hz. Research recordings only — "
             "do not upload identifiable patient data to a demo.")
    if up is None:
        st.caption(
            f"Expected input: the montage in `configs/base.yaml` at 256 Hz — the CHB-MIT "
            f"format. Files that do not match are refused rather than guessed at. "
            f"Operating point: threshold {op.get('threshold', 0):.3f}, k = {op.get('k')} "
            f"({op.get('basis', 'n/a')})."
        )
        return

    import tempfile

    from seizure.features import extract as EX
    cfg = load_cfg_obj()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / up.name
        p.write_bytes(up.getbuffer())
        with st.spinner(f"Reading, filtering and featurising {up.name} …"):
            try:
                # No annotations exist for an unseen file, so no window can be
                # labelled ictal or guarded -- every row is scored.
                res = EX.extract_record(p, [], cfg)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read this file: `{type(e).__name__}: {e}`")
                return

    # Read the attribute directly. A getattr with a default here silently
    # disabled the whole refusal gate when the field was named differently.
    reason = res.skipped_reason
    if reason:
        key = str(reason).split(":")[0]
        st.error(f"**Refused: `{reason}`**\n\n{REFUSALS.get(key, 'This file cannot be read.')}")
        st.caption("This refusal comes from the pipeline's own channel and rate checks, "
                   "not from a separate rule in this app — the same gate skipped three "
                   "chb12 records during extraction.")
        return
    if res.aggregated is None or not len(res.aggregated):
        st.error("No analysis windows could be built from this recording.")
        return

    agg = res.aggregated
    cols = bundle["columns"]
    missing = [c for c in cols if c not in agg.columns]
    if missing:
        st.error(f"Feature mismatch: {len(missing)} expected columns absent "
                 f"(first few: {missing[:4]}).")
        return

    scores = bundle["model"].predict_proba(agg[cols].to_numpy(dtype=np.float32))[:, 1]
    thr = float(op.get("threshold", 0.5))
    k = int(op.get("k", 1))

    c1, c2 = st.columns([3, 1])
    thr = c1.slider("Alarm threshold", 0.0, 1.0, thr, 0.005, key="up_thr")
    k = c2.radio("Consecutive windows (k)", [1, 2, 3],
                 index=[1, 2, 3].index(k) if k in (1, 2, 3) else 0,
                 horizontal=True, key="up_k")

    runs = M.positive_runs(agg.window_idx.to_numpy(), agg.t_start_sec.to_numpy(),
                           agg.t_end_sec.to_numpy(), (scores >= thr), k)
    dur = float(agg.t_end_sec.max())
    alarm_sec = sum(r.end_sec - r.start_sec for r in runs)

    m = st.columns(4)
    m[0].metric("Recording length", f"{dur / 60:.1f} min")
    m[1].metric("Analysis windows", f"{len(agg):,}")
    m[2].metric("Candidate segments flagged", len(runs))
    m[3].metric("Time in alarm", f"{alarm_sec / dur:.1%}" if dur else "—")

    fig = go.Figure()
    for r in runs:
        fig.add_vrect(x0=r.start_sec, x1=r.end_sec, fillcolor="#f9a825",
                      opacity=0.32, line_width=0, layer="below")
    fig.add_scatter(x=agg.t_start_sec, y=scores, mode="lines",
                    line=dict(width=1.1, color="#212121"), name="score")
    fig.add_hline(y=thr, line=dict(color="#c62828", dash="dash", width=1))
    fig.update_layout(height=320, xaxis_title="seconds into recording",
                      yaxis_title="model score", showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    if runs:
        st.dataframe(pd.DataFrame([{
            "segment": i + 1,
            "start": f"{int(r.start_sec // 60):02d}:{int(r.start_sec % 60):02d}",
            "end": f"{int(r.end_sec // 60):02d}:{int(r.end_sec % 60):02d}",
            "duration_sec": round(r.end_sec - r.start_sec, 1),
            "windows": r.n_windows,
        } for i, r in enumerate(runs)]), width="stretch", hide_index=True)

    st.info(
        "**These are flagged segments, not seizures.** This file has no expert annotation, so "
        "nothing here can be confirmed or refuted — the app cannot tell a seizure from chewing "
        "artifact, electrode movement, or ordinary drowsiness. On annotated data this model "
        "misses roughly two seizures in three and flags several segments per hour that are not "
        "seizures. A neurologist reading the trace is the only thing that resolves any of these.",
        icon="ℹ️")
    st.caption(
        "A further limit worth stating: passing the channel and rate gate does not make a "
        "recording similar to the training data. A different amplifier, referencing scheme, "
        "electrode paste, age group or clinical context all shift the features, and the model "
        "has no way to report that it is out of its depth — it will return confident scores "
        "regardless."
    )


# -------------------------------------------------------------- screen three
def screen_leak(loso: pd.DataFrame, rand: pd.DataFrame) -> None:
    st.header("The same model, two ways of splitting the data")
    protocol = st.radio(
        "Split protocol", ["group by patient (honest)", "shuffle windows (leaky)"],
        horizontal=True)
    leaky = protocol.startswith("shuffle")

    rows = []
    for m in ("logreg", "rf", "hgb"):
        lo = loso[loso.model == m]
        rd = rand[rand.model == m]
        if not len(lo) or not len(rd):
            continue
        rows.append({"model": m, "honest": lo.pr_auc.mean(),
                     "leaky": float(rd.pr_auc.iloc[0])})
    t = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_bar(x=t.model, y=t.leaky if leaky else t.honest,
                marker_color="#c62828" if leaky else "#2e7d32",
                text=[f"{v:.3f}" for v in (t.leaky if leaky else t.honest)],
                textposition="outside")
    fig.update_layout(height=380, yaxis_title="PR-AUC",
                      yaxis_range=[0, max(t.leaky.max(), t.honest.max()) * 1.25],
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width='stretch')

    t["leak"] = t.leaky - t.honest
    st.dataframe(t.rename(columns={
        "honest": "grouped by patient", "leaky": "windows shuffled",
        "leak": "inflation"}).style.format(precision=4), width='stretch')
    st.error(
        "Shuffling windows puts the same patient on both sides of the split. Two windows "
        "ten seconds apart are nearly identical, so the model is scored partly on data it "
        "trained on. Nothing about the model changes between these two bars — only the "
        "split does. **That difference is the single most important number in the project**, "
        "and a detector evaluated the leaky way would collapse on its first new patient."
    )


def main() -> None:
    loso, rand, rep = load_detail()
    sel = rep.get("selected_model", {}).get("aggregated", {})
    headline = sel.get("model", "rf")

    st.title("Cross-patient seizure detection — what it can and cannot do")
    st.caption(
        f"CHB-MIT Scalp EEG v1.0.0 · 23 patients · 979.9 hours · 352,742 windows at a "
        f"0.32% seizure rate · headline model `{headline}` · "
        f"extraction `{rep.get('extraction_hash')}`"
    )
    st.info(
        "**Not a medical device, and not a diagnostic tool.** This is a research "
        "demonstration built on a public research dataset. It accepts no patient data and "
        "produces no finding about any individual. Seizure precautions and seizure action "
        "plans come from a treating neurologist — see the Epilepsy Foundation, NHS, or ILAE.",
        icon="⚠️")
    if sel.get("owner_designated") and sel.get("auto_choice") != headline:
        st.caption(
            f"`{headline}` was designated by the project owner on event-level results. "
            f"Validation-only selection would have chosen `{sel.get('auto_choice')}`, so the "
            f"headline figures are an optimistically biased estimate rather than an unbiased one."
        )

    one, two, three, four = st.tabs([
        "1 · Who it works for", "2 · One record, one operating point",
        "3 · Why the split matters", "4 · Try an unseen recording"])
    with one:
        screen_bimodality(loso, headline)
    with two:
        screen_record(load_scores(), load_truth())
    with three:
        screen_leak(loso, rand)
    with four:
        screen_upload()


if __name__ == "__main__":
    main()

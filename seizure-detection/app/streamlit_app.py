"""Seizure Detector Report Card — an upload-driven analysis tool.

Every number on every page is computed from the recording the user supplies.
Nothing is precomputed from the study, and the three pages share one analysis
held in session state:

  1  Analyse      upload, headline figures for that file, model card
  2  Signal       score trace, flagged segments, raw montage, band powers
  3  Operating    threshold/k simulation and sweep curves for that file

Honest limit, and the reason for `truth` below: accuracy needs labels. An
arbitrary EDF has none, so sensitivity, false alarms and latency are simply not
computable for it. CHB-MIT records *do* have expert annotations, and
`artifacts/seizure_intervals.json` ships with the repo, so a recognised record
is scored against ground truth while an unknown file reports detection output
only -- labelled as unverifiable rather than dressed up as accuracy.

Two things here are functional, not decorative:
  * the channel/rate refusal gate, which stops the model scoring
    out-of-distribution input and returning confident nonsense
  * the one-line research-use footer
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seizure.eval import metrics as M  # noqa: E402

SNAP = ROOT / "artifacts" / "rfjoint_base"
FINAL_MODEL = ROOT / "artifacts" / "final_model_aggregated.joblib"

INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
ACCENT = "#1d4ed8"
GOOD = "#15803d"
WARN = "#b45309"
BAD = "#b91c1c"
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
BAND_COLOUR = ("#1d4ed8", "#0891b2", "#15803d", "#b45309", "#b91c1c")

st.set_page_config(page_title="Seizure Detector Report Card",
                   page_icon="◫", layout="wide",
                   initial_sidebar_state="auto")

CSS = f"""
<style>
#MainMenu, footer, header {{visibility: hidden;}}
div[data-testid="stToolbar"], div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"], .stAppDeployButton,
button[kind="header"] {{display: none !important;}}
.block-container {{padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1180px;}}
html, body, [class*="css"] {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
  color: {INK};
}}
h1, h2, h3 {{letter-spacing: -0.02em; font-weight: 650;}}
.hero {{border-bottom: 1px solid {LINE}; padding-bottom: 1.1rem; margin-bottom: 1.6rem;}}
.hero-title {{font-size: 1.85rem; font-weight: 680; margin: 0;}}
.kpi-grid {{display: flex; gap: .8rem; flex-wrap: wrap; margin: .4rem 0 1.5rem;}}
.kpi {{flex: 1 1 150px; border: 1px solid {LINE}; border-radius: 10px;
  padding: .85rem 1rem; background: #fff;}}
.kpi-label {{font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
  color: {MUTED}; font-weight: 600;}}
.kpi-value {{font-size: 1.6rem; font-weight: 680; margin-top: .15rem; line-height: 1.15;}}
.kpi-note {{font-size: .78rem; color: {MUTED}; margin-top: .1rem;}}
.card {{border: 1px solid {LINE}; border-left: 3px solid {ACCENT}; border-radius: 10px;
  padding: 1rem 1.15rem; background: #fff; margin-bottom: .85rem;}}
.card h4 {{margin: 0 0 .35rem; font-size: .97rem; font-weight: 650;}}
.card p {{margin: 0; color: #334154; font-size: .9rem; line-height: 1.5;}}
.section {{font-size: .74rem; text-transform: uppercase; letter-spacing: .09em;
  color: {MUTED}; font-weight: 650; margin: 1.7rem 0 .55rem;}}
.foot {{border-top: 1px solid {LINE}; margin-top: 2.6rem; padding-top: .9rem;
  color: {MUTED}; font-size: .78rem; line-height: 1.6;}}
div[data-testid="stSidebarNav"] {{display: none;}}
section[data-testid="stSidebar"] {{border-right: 1px solid {LINE};}}
.sb-brand {{font-weight: 680; font-size: 1.02rem; margin-bottom: .9rem;}}
.stPlotlyChart {{border: 1px solid {LINE}; border-radius: 10px; padding: .35rem;}}
div[data-testid="stMarkdownContainer"] table {{display: block; overflow-x: auto;
  max-width: 100%;}}

@media (max-width: 640px) {{
  .block-container {{padding-top: 1.1rem; padding-left: .85rem;
    padding-right: .85rem; padding-bottom: 1.6rem;}}
  .hero {{padding-bottom: .8rem; margin-bottom: 1.1rem;}}
  .hero-title {{font-size: 1.32rem;}}
  .kpi {{flex: 1 1 calc(50% - .4rem); min-width: 0; padding: .65rem .75rem;}}
  .kpi-value {{font-size: 1.24rem;}}
  .kpi-label {{font-size: .66rem;}}
  .kpi-note {{font-size: .72rem;}}
  .kpi-grid {{gap: .5rem; margin-bottom: 1.1rem;}}
  .card {{padding: .8rem .9rem; margin-bottom: .6rem;}}
  .card h4 {{font-size: .91rem;}}
  .card p {{font-size: .85rem; line-height: 1.45;}}
  .section {{margin: 1.2rem 0 .45rem;}}
  .foot {{margin-top: 1.8rem; font-size: .73rem;}}
  div[role="radiogroup"] label, section[data-testid="stSidebar"] label,
  div[data-testid="stCheckbox"] label {{
    min-height: 40px; display: flex; align-items: center;}}
  div[role="radiogroup"] {{gap: .1rem;}}
  section[data-testid="stSidebar"] button[kind="headerNoPadding"],
  div[data-testid="stElementToolbarButton"] button {{min-width: 34px;
    min-height: 34px;}}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# CHB-MIT record ids look like chb01_03, with an optional case-letter variant
# (chb17a_03). A seizure-free record is absent from seizure_intervals.json, so
# membership alone cannot distinguish "no annotation" from "annotated as having
# no seizure" -- and those are very different claims. Matching the id pattern
# separates them: pattern + in dict = annotated seizures, pattern + absent =
# annotated seizure-free, no pattern match = genuinely unverifiable.
RECORD_RE = re.compile(r"^chb\d{2}[a-z]?_\d+\+?$", re.I)

REFUSALS = {
    "missing_channels": "This recording does not contain the 18 canonical bipolar "
                        "channels the model requires.",
    "no_usable_channels": "No channel in this file parses as a bipolar derivation "
                          "between two 10-20 electrode positions.",
    "unexpected_sample_rate": "The model requires 256 Hz. A different rate shifts "
                              "every spectral feature.",
    "record_shorter_than_window": "The recording is shorter than one 10-second window.",
}


# ----------------------------------------------------------------- resources
@st.cache_resource
def load_model():
    if not FINAL_MODEL.exists():
        return None
    import joblib
    return joblib.load(FINAL_MODEL)


@st.cache_data
def load_cfg():
    from seizure.config import load
    return load(ROOT / "configs" / "base.yaml")


@st.cache_data
def load_truth() -> dict[str, list[tuple[int, int]]]:
    """Expert annotations, used only when the upload is a recognised record."""
    p = ROOT / "artifacts" / "seizure_intervals.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {r: [(int(a), int(b)) for a, b in s] for r, s in raw.items()}


@st.cache_data(show_spinner=False, max_entries=2)
def analyse(raw: bytes, filename: str) -> dict:
    """Decode, filter, window, featurise and score one uploaded recording.

    Keyed on the file bytes so re-selecting the same file is free and the
    result survives navigation between pages.
    """
    import tempfile

    from seizure.features import extract as EX
    cfg = load_cfg()
    bundle = load_model()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / filename
        p.write_bytes(raw)
        res = EX.extract_record(p, [], cfg)
        signal = None
        if not res.skipped_reason:
            from seizure.signal import io as SIO
            try:
                rec = SIO.read(p, list(cfg.signal.canonical_channels), "skip_file")
                # Store at 64 Hz: enough to draw a montage, a sixteenth of the
                # memory of the 256 Hz original.
                signal = (rec.data[:, ::4].astype(np.float32), rec.channels, 64.0)
            except Exception:  # noqa: BLE001
                signal = None

    if res.skipped_reason:
        return {"ok": False, "reason": str(res.skipped_reason), "name": filename}
    agg = res.aggregated
    if agg is None or not len(agg):
        return {"ok": False, "reason": "no_windows", "name": filename}

    cols = bundle["columns"]
    if [c for c in cols if c not in agg.columns]:
        return {"ok": False, "reason": "schema_mismatch", "name": filename}

    scores = bundle["model"].predict_proba(agg[cols].to_numpy(dtype=np.float32))[:, 1]
    stem = Path(filename).stem
    all_truth = load_truth()
    if stem in all_truth:
        truth, kind = all_truth[stem], "annotated"
    elif RECORD_RE.match(stem):
        truth, kind = [], "annotated_seizure_free"
    else:
        truth, kind = None, "unverifiable"
    return {
        "ok": True, "name": filename, "record_id": stem, "kind": kind,
        "frame": agg[["window_idx", "t_start_sec", "t_end_sec"]].copy(),
        "bands": {b: agg[f"bp_{b}_rel__mean"].to_numpy()
                  for b in BANDS if f"bp_{b}_rel__mean" in agg.columns},
        "scores": scores, "signal": signal,
        "truth": truth, "recognised": truth is not None,
        "duration": float(agg["t_end_sec"].max()),
        "op": bundle["meta"].get("operating_point", {}),
    }


# -------------------------------------------------------------------- chrome
def hero(title: str) -> None:
    st.markdown(f'<div class="hero"><div class="hero-title">{title}</div></div>',
                unsafe_allow_html=True)


def kpis(items) -> None:
    st.markdown('<div class="kpi-grid">' + "".join(
        f'<div class="kpi"><div class="kpi-label">{a}</div>'
        f'<div class="kpi-value">{b}</div><div class="kpi-note">{c}</div></div>'
        for a, b, c in items) + "</div>", unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="section">{label}</div>', unsafe_allow_html=True)


def plot(fig, height: int = 380, top: int = 18) -> None:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=top, b=8),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="-apple-system, Segoe UI, Inter, sans-serif", size=12,
                  color=INK),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def runs_for(a: dict, thr: float, k: int):
    f = a["frame"]
    return M.positive_runs(f.window_idx.to_numpy(), f.t_start_sec.to_numpy(),
                           f.t_end_sec.to_numpy(), (a["scores"] >= thr), k)


def controls(a: dict, key: str) -> tuple[float, int]:
    op = a["op"]
    c1, c2 = st.columns([3, 1])
    thr = c1.slider("Alarm threshold", 0.0, 1.0, float(op.get("threshold", 0.5)),
                    0.005, key=f"thr_{key}",
                    help="Default is the operating point selected on validation "
                         "patients during evaluation.")
    kk = int(op.get("k", 1))
    k = c2.radio("Consecutive windows", [1, 2, 3],
                 index=[1, 2, 3].index(kk) if kk in (1, 2, 3) else 0,
                 horizontal=True, key=f"k_{key}")
    return thr, k


def need_upload() -> None:
    st.info("No recording loaded. Upload an EDF file on **Analyse** to populate "
            "this page — every figure here is computed from that file.")


# ------------------------------------------------------------------- page 1
def page_analyse() -> None:
    hero("Analyse a recording")
    if load_model() is None:
        st.info("No fitted model found. Run `seizure fit-final --view aggregated`.")
        return

    st.caption("Requires 18 canonical bipolar channels at 256 Hz (CHB-MIT format). "
               "Files are processed in memory and discarded when the request ends. "
               "Use public research recordings — do not upload identifiable patient data.")
    up = st.file_uploader("EDF recording", type=["edf"], label_visibility="collapsed")

    if up is not None:
        with st.spinner("Decoding, filtering, windowing and scoring…"):
            st.session_state["analysis"] = analyse(up.getvalue(), up.name)

    a = st.session_state.get("analysis")
    if a is None:
        card("What this does",
             "Upload a scalp-EEG recording and the full pipeline runs over it: 60/120 Hz "
             "notch, 0.5–80 Hz bandpass, 10-second windowing, 56 features per window, then "
             "a random forest trained on 23 patients scores every window. You get the "
             "score trace, the flagged segments, and an operating-point simulation — all "
             "computed from your file.")
        card("Accuracy needs labels",
             "If the recording is a CHB-MIT record with expert annotations, detections are "
             "scored against ground truth. For any other file there is nothing to score "
             "against, so the output is flagged segments only and is labelled as "
             "unverifiable rather than presented as accuracy.")
        model_card()
        return

    if not a["ok"]:
        key = a["reason"].split(":")[0]
        st.error(f"**Unsupported recording** — "
                 f"{REFUSALS.get(key, 'this file cannot be read.')}")
        st.caption(f"Pipeline reason: `{a['reason']}`")
        model_card()
        return

    thr, k = controls(a, "p1")
    runs = runs_for(a, thr, k)
    dur = a["duration"] or 1.0
    alarm = sum(r.end_sec - r.start_sec for r in runs)

    items = [("Duration", f"{dur / 60:.1f} min", a["name"]),
             ("Windows scored", f"{len(a['frame']):,}", "10 s each"),
             ("Segments flagged", f"{len(runs)}", "candidate events"),
             ("Time in alarm", f"{alarm / dur:.1%}", "of this recording")]
    if a["recognised"]:
        spans = a["truth"]
        matched = {i for i, r in enumerate(runs)
                   if any(r.end_sec > s and r.start_sec < e for s, e in spans)}
        hit = sum(1 for s, e in spans
                  if any(r.end_sec > s and r.start_sec < e for r in runs))
        lat = [min((r.start_sec for r in runs if r.end_sec > s and r.start_sec < e),
                   default=np.nan) - s for s, e in spans]
        lat = [x for x in lat if x == x]
        items = [("Seizures detected",
                  f"{hit} / {len(spans)}" if spans else "0 / 0",
                  "against annotations" if spans else "record is seizure-free"),
                 ("False alarms", f"{len(runs) - len(matched)}", "no overlap with truth"),
                 ("Median latency", f"{np.median(lat):.1f}s" if lat else "—",
                  "from annotated onset"),
                 ("Time in alarm", f"{alarm / dur:.1%}", f"{dur / 60:.1f} min recording")]
    kpis(items)

    kind = a.get("kind")
    if kind == "annotated":
        st.success(f"`{a['record_id']}` matches a CHB-MIT record with "
                   f"{len(a['truth'])} expert-annotated seizure(s), so the figures above "
                   f"are scored against ground truth.", icon=":material/check_circle:")
    elif kind == "annotated_seizure_free":
        # Plain control flow, not a ternary over two st.* calls: Streamlit
        # introspects the caller's source line to name the element, and a
        # multi-line conditional expression makes that parse fail with a
        # SyntaxError raised from inside Streamlit.
        msg = (f"`{a['record_id']}` matches a CHB-MIT record annotated as containing **no "
               f"seizures**. Every segment flagged here is therefore a false alarm — ")
        if len(runs) == 0:
            st.success(msg + "and the model flagged none, which is the correct result.",
                       icon=":material/check_circle:")
        else:
            st.warning(msg + f"the model flagged {len(runs)}.",
                       icon=":material/warning:")
    else:
        st.warning("This file does not match a known CHB-MIT record, so there is no "
                   "annotation to score against. The figures describe what the model "
                   "flagged, not whether it was right — it cannot distinguish a seizure "
                   "from chewing artifact or electrode movement.", icon=":material/warning:")

    model_card()


def model_card() -> None:
    with st.expander("Model card — architecture, data, metrics and limitations"):
        st.markdown("""
**Task** Binary classification of 10-second scalp-EEG windows as ictal or interictal.

**Model** Random forest over 56 channel-aggregated features (5 log band powers, 5 relative
band powers, line length, variance, Hjorth mobility and complexity, each summarised
mean/max/std/median across 18 canonical bipolar derivations).

**Signal chain** 60/120 Hz notch, 0.5–80 Hz bandpass, 10 s non-overlapping windows,
Welch PSD (nperseg 512, noverlap 256).

**Training data** CHB-MIT Scalp EEG v1.0.0 (PhysioNet) — paediatric presurgical monitoring,
23 patients, 686 recordings, 198 annotated seizures. Ictal windows are 0.32% of the data.

**Evaluation** Leave-one-subject-out, 23 folds, grouped on patient so no patient appears in
both training and test. Windows within 30 s of a seizure boundary are excluded from
training, scoring and the false-alarm denominator.

**Operating point** Threshold and consecutive-window count were selected together on
validation patients and frozen before any held-out patient was scored.

**Known limitations**
- Cross-patient performance is bimodal: on held-out patients it worked reliably for a
  minority and found nothing at all for several. Expect wide variation between recordings.
- Window-level precision is low; most flagged windows are not ictal. The
  consecutive-window rule exists to suppress that at the event level.
- Requires 18 canonical bipolar channels at 256 Hz. Other montages are refused, not adapted.
- Trained on paediatric presurgical recordings. Behaviour on adults, other hardware or
  ambulatory data is unmeasured, and the model returns confident scores regardless.

**Intended use** Methodology demonstration and retrospective research. Not a medical device.
        """)


# ------------------------------------------------------------------- page 2
def page_signal() -> None:
    hero("Signal and detections")
    a = st.session_state.get("analysis")
    if a is None or not a.get("ok"):
        need_upload()
        return

    thr, k = controls(a, "p2")
    runs = runs_for(a, thr, k)
    spans = a["truth"] or []
    matched = {i for i, r in enumerate(runs)
               if any(r.end_sec > s and r.start_sec < e for s, e in spans)}

    section("Model score across the recording")
    fig = go.Figure()
    for s, e in spans:
        fig.add_vrect(x0=s, x1=e, fillcolor=GOOD, opacity=0.16, line_width=0,
                      layer="below")
    for i, r in enumerate(runs):
        fig.add_vrect(x0=r.start_sec, x1=r.end_sec,
                      fillcolor=ACCENT if i in matched else (BAD if spans else WARN),
                      opacity=0.26, line_width=0, layer="below")
    fig.add_scatter(x=a["frame"].t_start_sec, y=a["scores"], mode="lines",
                    line=dict(width=1.1, color=INK),
                    hovertemplate="%{x:.0f}s · %{y:.3f}<extra></extra>")
    fig.add_hline(y=thr, line=dict(color=BAD, dash="dot", width=1.2))
    fig.update_layout(xaxis_title="seconds into recording", yaxis_title="score",
                      showlegend=False)
    plot(fig, 320)
    st.caption(("Green: annotated seizure. Blue: alarm overlapping one. Red: false alarm."
                if spans else
                "Amber: flagged segment. No annotation exists for this file, so none of "
                "these can be confirmed or refuted."))

    if runs:
        section("Flagged segments")
        rows = []
        for i, r in enumerate(runs):
            sc = a["scores"][(a["frame"].t_start_sec >= r.start_sec)
                             & (a["frame"].t_end_sec <= r.end_sec)]
            rows.append({
                "#": i + 1,
                "start": f"{int(r.start_sec // 60):02d}:{int(r.start_sec % 60):02d}",
                "end": f"{int(r.end_sec // 60):02d}:{int(r.end_sec % 60):02d}",
                "seconds": round(r.end_sec - r.start_sec, 1),
                "windows": r.n_windows,
                "peak score": round(float(sc.max()), 3) if len(sc) else None,
                **({"overlaps annotation": "yes" if i in matched else "no"}
                   if spans else {}),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if a["bands"]:
        section("Relative band power over time")
        fig = go.Figure()
        for (b, v), col in zip(a["bands"].items(), BAND_COLOUR):
            fig.add_scatter(x=a["frame"].t_start_sec, y=v, mode="lines", name=b,
                            line=dict(width=1.1, color=col), stackgroup="one")
        for s, e in spans:
            fig.add_vrect(x0=s, x1=e, fillcolor="#000", opacity=0.10, line_width=0)
        fig.update_layout(xaxis_title="seconds into recording",
                          yaxis_title="share of total power",
                          legend=dict(orientation="h", y=1.10, x=0))
        plot(fig, 320, top=52)
        card("What to look for",
             "Seizure onset usually shows as a rapid shift in this mix — often a surge in "
             "theta and alpha at the expense of delta — and it is the same spectral change "
             "the model keys on, since ten of its fourteen per-channel features are band "
             "powers. A gradual drift instead suggests a state change such as drowsiness.")

    if a["signal"] is not None:
        section("Raw montage")
        data, chans, rate = a["signal"]
        default = int(spans[0][0]) - 10 if spans else (
            int(runs[0].start_sec) - 10 if runs else 0)
        maxt = max(0, int(data.shape[1] / rate) - 20)
        t0 = st.number_input("Window start (seconds)", 0, maxt,
                             min(max(0, default), maxt), step=5)
        span = 30
        seg = data[:, int(t0 * rate):int((t0 + span) * rate)]
        t = np.arange(seg.shape[1]) / rate + t0
        step = float(np.percentile(np.abs(seg), 99)) * 3 or 1.0
        fig = go.Figure()
        for i, ch in enumerate(chans):
            fig.add_scatter(x=t, y=seg[i] - i * step, mode="lines",
                            line=dict(width=0.65, color=INK), showlegend=False,
                            hoverinfo="skip")
        for s, e in spans:
            if e > t0 and s < t0 + span:
                fig.add_vrect(x0=max(s, t0), x1=min(e, t0 + span), fillcolor=GOOD,
                              opacity=0.14, line_width=0, layer="below")
        for r in runs:
            if r.end_sec > t0 and r.start_sec < t0 + span:
                fig.add_vrect(x0=max(r.start_sec, t0), x1=min(r.end_sec, t0 + span),
                              fillcolor=WARN, opacity=0.12, line_width=0, layer="below")
        fig.update_layout(
            xaxis_title="seconds",
            yaxis=dict(tickmode="array",
                       tickvals=[-i * step for i in range(len(chans))],
                       ticktext=chans, tickfont=dict(size=9)))
        plot(fig, 700)
        st.caption("Displayed at 64 Hz to keep the page responsive; the model scored the "
                   "full 256 Hz signal.")


# ------------------------------------------------------------------- page 3
def page_operating() -> None:
    hero("Operating point")
    a = st.session_state.get("analysis")
    if a is None or not a.get("ok"):
        need_upload()
        return

    thr, k = controls(a, "p3")
    runs = runs_for(a, thr, k)
    dur = a["duration"] or 1.0
    alarm = sum(r.end_sec - r.start_sec for r in runs)
    spans = a["truth"] or []
    matched = {i for i, r in enumerate(runs)
               if any(r.end_sec > s and r.start_sec < e for s, e in spans)}
    hit = sum(1 for s, e in spans
              if any(r.end_sec > s and r.start_sec < e for r in runs))

    base = [("Segments flagged", f"{len(runs)}", "at this setting"),
            ("Time in alarm", f"{alarm / dur:.1%}", "of the recording"),
            ("Alarm rate", f"{len(runs) / (dur / 3600):.2f}/h", "segments per hour"),
            ("Windows over threshold", f"{int((a['scores'] >= thr).sum()):,}",
             f"of {len(a['scores']):,}")]
    if spans:
        base[0] = ("Seizures detected", f"{hit} / {len(spans)}", "against annotations")
        base[2] = ("False alarms", f"{len(runs) - len(matched)}", "no overlap with truth")
    kpis(base)

    section("How the threshold changes the outcome, on this recording")
    grid = np.unique(np.quantile(a["scores"], np.linspace(0.50, 0.9995, 60)))
    rows = []
    for t in grid:
        rr = runs_for(a, float(t), k)
        al = sum(r.end_sec - r.start_sec for r in rr)
        row = {"threshold": float(t), "segments": len(rr),
               "time_in_alarm": al / dur}
        if spans:
            mt = {i for i, r in enumerate(rr)
                  if any(r.end_sec > s and r.start_sec < e for s, e in spans)}
            row["detected"] = sum(1 for s, e in spans
                                  if any(r.end_sec > s and r.start_sec < e for r in rr))
            row["false_alarms"] = len(rr) - len(mt)
        rows.append(row)
    sweep = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_scatter(x=sweep.threshold, y=sweep.time_in_alarm, mode="lines",
                    name="time in alarm", line=dict(width=2, color=BAD),
                    hovertemplate="thr %{x:.3f} · %{y:.1%}<extra></extra>")
    if spans:
        fig.add_scatter(x=sweep.threshold, y=sweep.detected / max(1, len(spans)),
                        mode="lines", name="fraction of seizures detected",
                        line=dict(width=2, color=GOOD))
    fig.add_vline(x=thr, line=dict(color=INK, dash="dot", width=1.2))
    fig.update_layout(xaxis_title="alarm threshold", yaxis_title="share",
                      yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=1.10, x=0))
    plot(fig, 340, top=52)

    fig2 = go.Figure()
    fig2.add_scatter(x=sweep.threshold, y=sweep.segments, mode="lines",
                     name="segments", line=dict(width=2, color=ACCENT))
    if spans:
        fig2.add_scatter(x=sweep.threshold, y=sweep.false_alarms, mode="lines",
                         name="false alarms", line=dict(width=2, color=BAD))
    fig2.add_vline(x=thr, line=dict(color=INK, dash="dot", width=1.2))
    fig2.update_layout(xaxis_title="alarm threshold", yaxis_title="count",
                       legend=dict(orientation="h", y=1.10, x=0))
    plot(fig2, 300, top=52)

    section("Consecutive-window rule at this threshold")
    krows = []
    for kk in (1, 2, 3, 4, 5):
        rr = runs_for(a, thr, kk)
        al = sum(r.end_sec - r.start_sec for r in rr)
        row = {"k": kk, "segments flagged": len(rr),
               "time in alarm": f"{al / dur:.1%}"}
        if spans:
            mt = {i for i, r in enumerate(rr)
                  if any(r.end_sec > s and r.start_sec < e for s, e in spans)}
            row["seizures detected"] = f"{sum(1 for s, e in spans if any(r.end_sec > s and r.start_sec < e for r in rr))} / {len(spans)}"
            row["false alarms"] = len(rr) - len(mt)
        krows.append(row)
    st.dataframe(pd.DataFrame(krows), width="stretch", hide_index=True)

    a1, a2 = st.columns(2)
    with a1:
        card("The two controls are not independent",
             "Raising the threshold shortens every run of positive windows, so a strict "
             "threshold and a long required run cancel each other out. During evaluation a "
             "fold that picked them sequentially ended up with correct windows, zero false "
             "positives, and zero detected seizures — nothing ever reached the required "
             "length. They are selected jointly for that reason.")
    with a2:
        card("Why time in alarm is shown beside the count",
             "Alarm counts treat one continuous alarm as a single event, so a detector that "
             "never stops alarming scores an excellent alarm rate. Duration cannot be gamed "
             "that way. Drag the threshold to zero and watch the count stay low while time "
             "in alarm goes to 100%.")

    if not a["recognised"]:
        st.warning("This file does not match a known CHB-MIT record, so the curves above show "
                   "what the model flags at each setting, not how often it is right.",
                   icon=":material/warning:")


# ---------------------------------------------------------------------- main
PAGES = {"Analyse": page_analyse, "Signal and detections": page_signal,
         "Operating point": page_operating}


def main() -> None:
    with st.sidebar:
        st.markdown('<div class="sb-brand">Seizure Detector Report Card</div>',
                    unsafe_allow_html=True)
        page = st.radio("Section", list(PAGES), label_visibility="collapsed")
        a = st.session_state.get("analysis")
        st.markdown("---")
        if a and a.get("ok"):
            st.markdown(
                f'<div style="font-size:.78rem;color:{MUTED};line-height:1.7">'
                f'<b>Loaded</b> {a["name"]}<br>'
                f'<b>Length</b> {a["duration"] / 60:.1f} min · '
                f'{len(a["frame"]):,} windows<br>'
                f'<b>Annotations</b> '
                f'{"available" if a["recognised"] else "none"}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:.78rem;color:{MUTED}">'
                        f'No recording loaded.</div>', unsafe_allow_html=True)

    PAGES[page]()

    st.markdown(
        '<div class="foot">Research software for methodology demonstration. '
        'Not a medical device, not clinically validated, and not for use in patient care. '
        'Model trained on the CHB-MIT Scalp EEG Database (Shoeb 2009; Goldberger et al. '
        '2000), distributed by PhysioNet under ODC-BY.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

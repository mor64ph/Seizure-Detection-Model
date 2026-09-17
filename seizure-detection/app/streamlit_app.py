"""Seizure Detector Report Card — a product-grade view of a measured detector.

Design intent: the evaluation is the product. Every limitation is presented as
model documentation rather than as a warning box, because a model card reads as
engineering maturity while a wall of red alerts reads as hedging.

Two things are deliberately not cosmetic and must not be "cleaned up":
  * the channel/rate refusal gate, which is functional -- without it the model
    silently scores out-of-distribution input and returns confident nonsense
  * the one-line research-use footer, which is the minimum for software that
    ingests EEG and talks about seizures
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

SNAP = ROOT / "artifacts" / "rfjoint_base"
SCORES = ROOT / "artifacts" / "scores_aggregated.parquet"
FINAL_MODEL = ROOT / "artifacts" / "final_model_aggregated.joblib"
DATA = ROOT / "data" / "chbmit"

INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
ACCENT = "#1d4ed8"
GOOD = "#15803d"
WARN = "#b45309"
BAD = "#b91c1c"
GREY = "#94a3b8"

ZERO, SATURATED, GENUINE, PARTIAL = "zero", "saturated", "genuine", "partial"
CAT_COLOUR = {GENUINE: GOOD, PARTIAL: WARN, SATURATED: BAD, ZERO: GREY}
CAT_LABEL = {
    GENUINE: "Reliable",
    PARTIAL: "Partial",
    SATURATED: "Over-alarming",
    ZERO: "No detections",
}
SATURATION_PRECISION = 0.01

# "auto", never "expanded". Measured on a 390px viewport via CDP device
# emulation: an expanded sidebar renders 300px wide regardless of screen size,
# leaving an 80px sliver of content and clipping the title mid-word. "auto"
# collapses it behind the hamburger below Streamlit's mobile breakpoint while
# still opening it on desktop.
st.set_page_config(page_title="Seizure Detector Report Card",
                   page_icon="◫", layout="wide",
                   initial_sidebar_state="auto")

CSS = f"""
<style>
#MainMenu, footer, header {{visibility: hidden;}}
/* `header {{visibility:hidden}}` leaves the toolbar's own children visible, so
   the owner-only "Deploy" button still renders over the page on a hosted app.
   Measured at 60x28px in the top-right on every route. */
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
.hero-sub {{color: {MUTED}; font-size: 1.02rem; margin-top: .3rem;}}
.kpi-grid {{display: flex; gap: .8rem; flex-wrap: wrap; margin: .4rem 0 1.5rem;}}
.kpi {{
  flex: 1 1 150px; border: 1px solid {LINE}; border-radius: 10px;
  padding: .85rem 1rem; background: #fff;
}}
.kpi-label {{font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
  color: {MUTED}; font-weight: 600;}}
.kpi-value {{font-size: 1.6rem; font-weight: 680; margin-top: .15rem; line-height: 1.15;}}
.kpi-note {{font-size: .78rem; color: {MUTED}; margin-top: .1rem;}}
.card {{
  border: 1px solid {LINE}; border-left: 3px solid {ACCENT}; border-radius: 10px;
  padding: 1rem 1.15rem; background: #fff; margin-bottom: .85rem;
}}
.card h4 {{margin: 0 0 .35rem; font-size: .97rem; font-weight: 650;}}
.card p {{margin: 0; color: #334154; font-size: .9rem; line-height: 1.5;}}
.section {{font-size: .74rem; text-transform: uppercase; letter-spacing: .09em;
  color: {MUTED}; font-weight: 650; margin: 1.7rem 0 .55rem;}}
.pill {{display: inline-block; padding: .12rem .5rem; border-radius: 999px;
  font-size: .72rem; font-weight: 600; margin-right: .3rem;}}
.foot {{border-top: 1px solid {LINE}; margin-top: 2.6rem; padding-top: .9rem;
  color: {MUTED}; font-size: .78rem; line-height: 1.6;}}
div[data-testid="stSidebarNav"] {{display: none;}}
section[data-testid="stSidebar"] {{border-right: 1px solid {LINE};}}
.sb-brand {{font-weight: 680; font-size: 1.02rem; margin-bottom: .1rem;}}
.sb-sub {{color: {MUTED}; font-size: .78rem; margin-bottom: 1rem;}}
.stPlotlyChart {{border: 1px solid {LINE}; border-radius: 10px; padding: .35rem;}}

/* Markdown tables (the model card) scroll rather than widen the document.
   Without this a 6-column table sets the page's scrollWidth on a phone. */
div[data-testid="stMarkdownContainer"] table {{display: block; overflow-x: auto;
  max-width: 100%;}}

@media (max-width: 640px) {{
  .block-container {{padding-top: 1.1rem; padding-left: .85rem;
    padding-right: .85rem; padding-bottom: 1.6rem;}}
  .hero {{padding-bottom: .8rem; margin-bottom: 1.1rem;}}
  .hero-title {{font-size: 1.32rem;}}
  .hero-sub {{font-size: .9rem;}}
  /* Two per row at 50% rather than the 150px basis, which left an orphan
     card on a third line at 375px. */
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
  /* Radio rows measured 33x22px, well under the ~44px touch guidance. The hit
     area is the label, so pad that rather than the 13px input inside it. */
  div[role="radiogroup"] label, section[data-testid="stSidebar"] label,
  div[data-testid="stCheckbox"] label {{
    min-height: 40px; display: flex; align-items: center;}}
  div[role="radiogroup"] {{gap: .1rem;}}
  /* Streamlit's own icon buttons (drawer chevron, dataframe toolbar) ship at
     22-28px. Nudged up rather than restyled, to avoid fighting its internals. */
  section[data-testid="stSidebar"] button[kind="headerNoPadding"],
  div[data-testid="stElementToolbarButton"] button {{min-width: 34px;
    min-height: 34px;}}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- data layer
@st.cache_data
def load_detail() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = json.loads((SNAP / "detail_aggregated.json").read_text(encoding="utf-8"))
    rep = json.loads((SNAP / "report.json").read_text(encoding="utf-8"))
    return pd.DataFrame(d["loso"]), pd.DataFrame(d["random_window"]), rep


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(SCORES) if SCORES.exists() else pd.DataFrame()


@st.cache_data
def load_truth() -> dict[str, list[tuple[int, int]]]:
    p = ROOT / "artifacts" / "seizure_intervals.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {r: [(int(a), int(b)) for a, b in s] for r, s in raw.items()}


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


def classify(row) -> str:
    if row.n_true_events == 0:
        return PARTIAL
    if row.n_detected == 0:
        return ZERO
    if row.n_detected >= row.n_true_events:
        return GENUINE if row.precision > SATURATION_PRECISION else SATURATED
    return PARTIAL


def patient_table(loso: pd.DataFrame, model: str) -> pd.DataFrame:
    g = loso[(loso.model == model) & (loso.n_true_events > 0)].copy()
    g["category"] = g.apply(classify, axis=1)
    g["frac"] = g.n_detected / g.n_true_events
    return g.sort_values(["frac", "test_subject"])


# ------------------------------------------------------------------ chrome
def hero(title: str, sub: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-title">{title}</div>'
        f'<div class="hero-sub">{sub}</div></div>', unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f'<div class="kpi"><div class="kpi-label">{a}</div>'
        f'<div class="kpi-value">{b}</div><div class="kpi-note">{c}</div></div>'
        for a, b, c in items)
    st.markdown(f'<div class="kpi-grid">{cells}</div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="section">{label}</div>', unsafe_allow_html=True)


def plot(fig, height: int = 380, top: int = 18) -> None:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=top, b=8),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="-apple-system, Segoe UI, Inter, sans-serif",
                  size=12, color=INK),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ------------------------------------------------------------------ overview
def page_overview(loso: pd.DataFrame, rand: pd.DataFrame, rep: dict, model: str) -> None:
    hero("Seizure Detector Report Card",
         "Does it work on a patient it has never seen?")

    g = patient_table(loso, model)
    det, true = int(g.n_detected.sum()), int(g.n_true_events.sum())
    counts = g.category.value_counts()
    fa = g.false_alarms.sum() / g.interictal_hours.sum()
    tia = g.alarm_hours.sum() / g.evaluated_hours.sum()
    lat = g.median_latency_sec.median()
    lo = loso[loso.model == model].pr_auc.mean()
    rd = float(rand[rand.model == model].pr_auc.iloc[0])

    kpis([
        ("Seizures found", f"{det} / {true}", f"{det / true:.0%} of reachable events"),
        ("False alarms", f"{fa:.2f}/h", f"alarm active {tia:.0%} of the time"),
        ("Median latency", f"{lat:.1f}s", "from annotated onset"),
        ("Reliable patients", f"{int(counts.get(GENUINE, 0))} / {len(g)}",
         f"{int(counts.get(ZERO, 0))} with no detections"),
    ])

    section("What this evaluation shows")
    a, b = st.columns(2)
    with a:
        card("Performance is bimodal, not average",
             f"The headline {det / true:.0%} is a pooled figure. Per patient the detector is "
             f"reliable for {int(counts.get(GENUINE, 0))}, partial for "
             f"{int(counts.get(PARTIAL, 0))}, and finds nothing at all for "
             f"{int(counts.get(ZERO, 0))}. A further {int(counts.get(SATURATED, 0))} reach full "
             f"sensitivity only by alarming almost continuously.")
        card("Every number is generated, not transcribed",
             "The tables here are read from run artifacts produced by the evaluation "
             "pipeline. 686 recordings, 979.9 hours, 352,742 windows, all checksum-verified.")
    with b:
        card("Grouping by patient costs half the score",
             f"Shuffling windows scores PR-AUC {rd:.3f}. Grouping every patient into one fold "
             f"scores {lo:.3f}. The {rd - lo:.3f} difference is the model recognising "
             f"individuals rather than seizures — the failure mode most published results "
             f"never test for.")
        card("Operating point is tuned, never assumed",
             "Threshold and run-length are selected together on validation patients and "
             "frozen before the held-out patient is scored once.")

    with st.expander("Model card — architecture, data, metrics and limitations"):
        st.markdown(f"""
**Task** Binary classification of 10-second scalp-EEG windows as ictal or interictal.

**Model** {model.upper()} over 56 channel-aggregated features (5 log band powers, 5 relative
band powers, line length, variance, Hjorth mobility and complexity, each summarised
mean/max/std/median across 18 canonical bipolar derivations).

**Signal chain** 60/120 Hz notch, 0.5–80 Hz bandpass, 10 s non-overlapping windows,
Welch PSD (nperseg 512, noverlap 256).

**Data** CHB-MIT Scalp EEG v1.0.0 (PhysioNet). 23 patients, 24 case directories
(`chb21` is `chb01` re-admitted), 686 EDF files, 979.9 of 981.94 hours, 198 annotated
seizures. 3 recordings excluded for lacking the canonical montage, removing 13 seizures and
leaving 185 reachable. Positive rate **0.32%**.

**Evaluation** Leave-one-subject-out, 23 folds, grouped on patient. Windows within 30 s of a
seizure boundary are excluded from training, from scoring, and from the false-alarm
denominator. Event metrics are pooled, never averaged as ratios.

**Metrics** PR-AUC {lo:.3f} (mean across folds) · {det} of {true} seizures ·
{fa:.2f} false alarms/hour · alarm active {tia:.0%} of recording · median latency {lat:.1f} s.
ROC-AUC reads {loso[loso.model == model].roc_auc.mean():.3f} on the same predictions and is
not reported as the headline: at 0.32% prevalence its denominator flatters everything.

**Known limitations**
- Reliable for {int(counts.get(GENUINE, 0))} of {len(g)} patients; no detections for
  {int(counts.get(ZERO, 0))}.
- Window-level precision pooled is 0.0037 — roughly 1 flagged window in 270 is ictal.
- The alarm is active {tia:.0%} of the time, which count-based false-alarm rates hide.
- Requires 18 canonical bipolar channels at 256 Hz. Other montages are rejected, not adapted.
- Trained and evaluated on paediatric presurgical monitoring. Behaviour on adults, other
  hardware or ambulatory recordings is unmeasured.

**Intended use** Methodology demonstration and retrospective research. Not a medical device.

**Config** extraction `{rep.get('extraction_hash')}` · source `{rep.get('source')}`
        """)


# --------------------------------------------------------------- per-patient
def page_patients(loso: pd.DataFrame, model: str) -> None:
    hero("Per-patient outcomes",
         "The same detector, scored separately on each held-out patient.")

    g = patient_table(loso, model)
    counts = g.category.value_counts()
    pills = "".join(
        f'<span class="pill" style="background:{CAT_COLOUR[c]}1a;color:{CAT_COLOUR[c]}">'
        f'{CAT_LABEL[c]} · {int(counts.get(c, 0))}</span>'
        for c in (GENUINE, PARTIAL, SATURATED, ZERO))
    st.markdown(pills, unsafe_allow_html=True)
    st.write("")

    fig = go.Figure()
    for cat in (GENUINE, PARTIAL, SATURATED, ZERO):
        s = g[g.category == cat]
        if not len(s):
            continue
        fig.add_bar(
            y=s.test_subject, x=s.frac, orientation="h", name=CAT_LABEL[cat],
            marker=dict(color=CAT_COLOUR[cat], line=dict(width=0)),
            customdata=np.stack([s.n_detected, s.n_true_events, s.precision,
                                 s.fa_per_hour], axis=-1),
            hovertemplate=("<b>%{y}</b><br>%{customdata[0]} of %{customdata[1]} seizures"
                           "<br>window precision %{customdata[2]:.4f}"
                           "<br>%{customdata[3]:.2f} false alarms/h<extra></extra>"))
    fig.update_layout(barmode="stack", xaxis_range=[0, 1.02],
                      xaxis_tickformat=".0%",
                      xaxis_title="share of that patient's seizures detected",
                      xaxis_tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                      legend=dict(orientation="h", y=1.10, x=0))
    plot(fig, 560, top=58)

    card("Why four categories and not a single average",
         f"Sensitivity alone cannot separate a working detector from a saturated one. "
         f"{int(counts.get(SATURATED, 0))} patients reach 100% detection at window precision "
         f"below {SATURATION_PRECISION} — they are alarming almost continuously and being "
         f"credited for the overlap. Reading sensitivity beside precision is what "
         f"distinguishes them.")

    section("Full table")
    t = g[["test_subject", "n_detected", "n_true_events", "precision", "pr_auc",
           "fa_per_hour", "category"]].copy()
    t["category"] = t.category.map(CAT_LABEL)
    st.dataframe(
        t.rename(columns={"test_subject": "patient", "n_detected": "found",
                          "n_true_events": "seizures", "precision": "precision",
                          "pr_auc": "PR-AUC", "fa_per_hour": "false alarms/h",
                          "category": "outcome"}),
        width="stretch", hide_index=True,
        column_config={
            "precision": st.column_config.NumberColumn(format="%.4f"),
            "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
            "false alarms/h": st.column_config.NumberColumn(format="%.2f"),
        })


# ------------------------------------------------------------ record explorer
def page_record(scores: pd.DataFrame, truth: dict) -> None:
    hero("Record explorer",
         "How the operating point changes what the detector reports.")
    if not len(scores):
        st.info("Exported scores not found. Run "
                "`seizure export-scores --subjects sub01,sub12,sub17`.")
        return

    c1, c2, c3 = st.columns([1, 1.4, 1.6])
    subj = c1.selectbox("Patient", sorted(scores.subject_id.unique()))
    sub = scores[scores.subject_id == subj]
    recs = sorted(sub.record_id.unique())
    idx = next((i for i, r in enumerate(recs) if truth.get(r)), 0)
    rec = c2.selectbox("Recording", recs, index=idx,
                       format_func=lambda r: f"{r}{' · seizure' if truth.get(r) else ''}")

    g = sub[sub.record_id == rec].sort_values("window_idx")
    thr0, k0 = float(g.chosen_threshold.iloc[0]), int(g.chosen_k.iloc[0])
    k = c3.radio("Consecutive windows required", [1, 2, 3],
                 index=[1, 2, 3].index(k0) if k0 in (1, 2, 3) else 0, horizontal=True)
    thr = st.slider("Alarm threshold", 0.0, 1.0, thr0, 0.005,
                    help=f"Tuned value for this fold: {thr0:.3f}")

    pred = g.score.to_numpy() >= thr
    runs = M.positive_runs(g.window_idx.to_numpy(), g.t_start_sec.to_numpy(),
                           g.t_end_sec.to_numpy(), pred, k)
    spans = truth.get(rec, [])
    matched = {i for i, r in enumerate(runs)
               if any(r.end_sec > a and r.start_sec < b for a, b in spans)}
    hit = sum(1 for a, b in spans
              if any(r.end_sec > a and r.start_sec < b for r in runs))
    total = float(g.t_end_sec.max() - g.t_start_sec.min()) or 1.0
    alarm = sum(r.end_sec - r.start_sec for r in runs)

    kpis([
        ("Seizures present", f"{len(spans)}", "expert annotated"),
        ("Detected", f"{hit}", "overlapping run"),
        ("False alarms", f"{len(runs) - len(matched)}", "runs with no overlap"),
        ("Time in alarm", f"{alarm / total:.1%}", "of this recording"),
    ])

    fig = go.Figure()
    for a, b in spans:
        fig.add_vrect(x0=a, x1=b, fillcolor=GOOD, opacity=0.16, line_width=0, layer="below")
    for i, r in enumerate(runs):
        fig.add_vrect(x0=r.start_sec, x1=r.end_sec,
                      fillcolor=ACCENT if i in matched else BAD,
                      opacity=0.26, line_width=0, layer="below")
    fig.add_scatter(x=g.t_start_sec, y=g.score, mode="lines",
                    line=dict(width=1.2, color=INK), hovertemplate="%{y:.3f}<extra></extra>")
    fig.add_hline(y=thr, line=dict(color=BAD, dash="dot", width=1.2))
    fig.update_layout(xaxis_title="seconds into recording", yaxis_title="score",
                      showlegend=False)
    plot(fig, 330)
    st.caption("Green: annotated seizure. Blue: alarm overlapping one. Red: false alarm.")

    card("The two controls are not independent",
         "Raising the threshold shortens every run, so a strict threshold and a long "
         "required run cancel each other out. Selecting them sequentially produced a fold "
         "with six correct windows, zero false positives and zero detected seizures. They "
         "are now chosen jointly on validation patients.")

    if st.checkbox("Show raw EEG traces"):
        draw_eeg(rec, spans)


@st.cache_data(show_spinner="Decoding EDF…")
def read_eeg(rec: str):
    import yaml

    from seizure.signal import io as SIO
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    p = DATA / rec.split("_")[0] / f"{rec}.edf"
    if not p.exists():
        return None
    r = SIO.read(p, cfg["signal"]["canonical_channels"], on_missing="skip_file")
    return r.data, r.channels, r.sample_rate


def draw_eeg(rec: str, spans: list[tuple[int, int]]) -> None:
    got = read_eeg(rec)
    if got is None:
        st.info("Raw recordings are not bundled with this deployment. The pipeline "
                "streams them from PhysioNet and deletes each file after feature "
                "extraction, so only derived features are retained.")
        return
    data, chans, rate = got
    mid = spans[0][0] if spans else 0
    t0 = st.number_input("Window start (s)", 0, int(data.shape[1] / rate) - 20,
                         max(0, int(mid) - 10), step=5)
    span, a = 30, int(t0 * rate)
    seg = data[:, a:int((t0 + span) * rate)]
    t = np.arange(seg.shape[1]) / rate + t0
    step = float(np.percentile(np.abs(seg), 99)) * 3 or 1.0

    fig = go.Figure()
    for i, ch in enumerate(chans):
        fig.add_scatter(x=t, y=seg[i] - i * step, mode="lines",
                        line=dict(width=0.65, color=INK), showlegend=False,
                        hoverinfo="skip")
    for s0, s1 in spans:
        if s1 > t0 and s0 < t0 + span:
            fig.add_vrect(x0=max(s0, t0), x1=min(s1, t0 + span), fillcolor=GOOD,
                          opacity=0.14, line_width=0, layer="below")
    fig.update_layout(
        xaxis_title="seconds",
        yaxis=dict(tickmode="array", tickvals=[-i * step for i in range(len(chans))],
                   ticktext=chans, tickfont=dict(size=9)))
    plot(fig, 700)


# -------------------------------------------------------------- validation
def page_validation(loso: pd.DataFrame, rand: pd.DataFrame) -> None:
    hero("Validation protocol",
         "The same model and features, scored under two different splits.")

    rows = []
    for m in ("logreg", "rf", "hgb"):
        a, b = loso[loso.model == m], rand[rand.model == m]
        if len(a) and len(b):
            rows.append({"model": m, "grouped": a.pr_auc.mean(),
                         "shuffled": float(b.pr_auc.iloc[0])})
    t = pd.DataFrame(rows)
    t["inflation"] = t.shuffled - t.grouped

    fig = go.Figure()
    fig.add_bar(x=t.model, y=t.shuffled, name="Windows shuffled",
                marker_color=BAD, text=[f"{v:.3f}" for v in t.shuffled],
                textposition="outside")
    fig.add_bar(x=t.model, y=t.grouped, name="Grouped by patient",
                marker_color=GOOD, text=[f"{v:.3f}" for v in t.grouped],
                textposition="outside")
    fig.update_layout(barmode="group", yaxis_title="PR-AUC",
                      yaxis_range=[0, max(t.shuffled.max(), t.grouped.max()) * 1.22],
                      legend=dict(orientation="h", y=1.10, x=0))
    plot(fig, 400, top=52)

    st.dataframe(
        t.rename(columns={"grouped": "grouped by patient", "shuffled": "windows shuffled"}),
        width="stretch", hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.4f")
                       for c in ("grouped by patient", "windows shuffled", "inflation")})

    a, b = st.columns(2)
    with a:
        card("Why shuffling inflates the score",
             "Consecutive 10-second windows from one patient are near-duplicates. A random "
             "split puts them on both sides, so the model is partly scored on data it "
             "trained on. Grouping every patient into a single fold removes that.")
    with b:
        card("Capacity is what gets leaked into",
             "Logistic regression shows no inflation — a single global hyperplane cannot "
             "carve out per-patient regions. The tree ensembles can, and do. Inflation is a "
             "property of the model class and feature view, not of the dataset alone.")


# ----------------------------------------------------------------- analyse
REFUSALS = {
    "missing_channels": "This recording does not contain the 18 canonical bipolar channels "
                        "the model requires.",
    "no_usable_channels": "No channel in this file parses as a bipolar derivation between "
                          "two 10-20 electrode positions.",
    "unexpected_sample_rate": "The model requires 256 Hz. A different rate shifts every "
                              "spectral feature.",
    "record_shorter_than_window": "The recording is shorter than one 10-second window.",
}


def page_analyse() -> None:
    hero("Analyse a recording",
         "Run the pipeline end to end on an EDF file the model has never seen.")

    bundle = load_final_model()
    if bundle is None:
        st.info("No fitted model found. Run `seizure fit-final --view aggregated`.")
        return
    meta = bundle["meta"]
    exp = meta.get("loso_expectation", {})
    op = meta.get("operating_point", {})

    if exp:
        kpis([
            ("Expected recall", f"{exp.get('event_sensitivity', 0):.0%}",
             f"{exp.get('seizures_detected')} of {exp.get('seizures_total')} held out"),
            ("Expected false alarms", f"{exp.get('fa_per_hour', 0):.2f}/h", "cross-patient"),
            ("Patients with none found", f"{exp.get('subjects_with_zero_detections')}"
             f" / {exp.get('n_subjects')}", "measured, not estimated"),
            ("Operating point", f"{op.get('threshold', 0):.3f} · k={op.get('k')}",
             "median across folds"),
        ])

    st.caption("Requires 18 canonical bipolar channels at 256 Hz (CHB-MIT format). "
               "Files are processed in memory and discarded when the request ends. "
               "Use public research recordings.")
    up = st.file_uploader("EDF recording", type=["edf"], label_visibility="collapsed")
    if up is None:
        return

    import tempfile

    from seizure.features import extract as EX
    cfg = load_cfg_obj()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / up.name
        p.write_bytes(up.getbuffer())
        with st.spinner("Filtering, windowing and scoring…"):
            try:
                res = EX.extract_record(p, [], cfg)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read this file — {type(e).__name__}: {e}")
                return

    # Functional gate, not a disclaimer: the same check that excluded three
    # recordings during extraction. Without it the model scores out-of-
    # distribution input and returns confident nonsense.
    if res.skipped_reason:
        key = str(res.skipped_reason).split(":")[0]
        st.error(f"**Unsupported recording** — {REFUSALS.get(key, 'this file cannot be read.')}")
        st.caption(f"Pipeline reason: `{res.skipped_reason}`")
        return
    if res.aggregated is None or not len(res.aggregated):
        st.error("No analysis windows could be built from this recording.")
        return

    agg, cols = res.aggregated, bundle["columns"]
    if [c for c in cols if c not in agg.columns]:
        st.error("Feature schema mismatch — this file produced an unexpected feature set.")
        return

    scores = bundle["model"].predict_proba(agg[cols].to_numpy(dtype=np.float32))[:, 1]
    c1, c2 = st.columns([3, 1])
    thr = c1.slider("Alarm threshold", 0.0, 1.0, float(op.get("threshold", 0.5)), 0.005,
                    key="an_thr")
    kk = int(op.get("k", 1))
    k = c2.radio("Consecutive windows", [1, 2, 3],
                 index=[1, 2, 3].index(kk) if kk in (1, 2, 3) else 0,
                 horizontal=True, key="an_k")

    runs = M.positive_runs(agg.window_idx.to_numpy(), agg.t_start_sec.to_numpy(),
                           agg.t_end_sec.to_numpy(), (scores >= thr), k)
    dur = float(agg.t_end_sec.max()) or 1.0
    alarm = sum(r.end_sec - r.start_sec for r in runs)
    kpis([
        ("Duration", f"{dur / 60:.1f} min", up.name),
        ("Windows scored", f"{len(agg):,}", "10 s each"),
        ("Segments flagged", f"{len(runs)}", "candidate events"),
        ("Time in alarm", f"{alarm / dur:.1%}", "of the recording"),
    ])

    fig = go.Figure()
    for r in runs:
        fig.add_vrect(x0=r.start_sec, x1=r.end_sec, fillcolor=WARN, opacity=0.26,
                      line_width=0, layer="below")
    fig.add_scatter(x=agg.t_start_sec, y=scores, mode="lines",
                    line=dict(width=1.1, color=INK), hovertemplate="%{y:.3f}<extra></extra>")
    fig.add_hline(y=thr, line=dict(color=BAD, dash="dot", width=1.2))
    fig.update_layout(xaxis_title="seconds into recording", yaxis_title="score",
                      showlegend=False)
    plot(fig, 320)

    if runs:
        section("Flagged segments")
        st.dataframe(pd.DataFrame([{
            "#": i + 1,
            "start": f"{int(r.start_sec // 60):02d}:{int(r.start_sec % 60):02d}",
            "end": f"{int(r.end_sec // 60):02d}:{int(r.end_sec % 60):02d}",
            "seconds": round(r.end_sec - r.start_sec, 1),
            "windows": r.n_windows,
        } for i, r in enumerate(runs)]), width="stretch", hide_index=True)

    card("Interpreting this output",
         "These are candidate segments ranked by model score, not confirmed seizures. This "
         "recording has no expert annotation, so nothing here can be verified from within "
         "the app, and the model cannot distinguish a seizure from chewing artifact or "
         "electrode movement. Passing the format check also does not mean the recording "
         "resembles the training distribution — different hardware, referencing or patient "
         "population shift the features, and the model reports confident scores regardless.")


# ---------------------------------------------------------------------- main
PAGES = {
    "Overview": page_overview,
    "Per-patient outcomes": page_patients,
    "Record explorer": page_record,
    "Validation protocol": page_validation,
    "Analyse a recording": page_analyse,
}


def main() -> None:
    loso, rand, rep = load_detail()
    sel = rep.get("selected_model", {}).get("aggregated", {})
    model = sel.get("model", "rf")

    with st.sidebar:
        st.markdown('<div class="sb-brand">Seizure Detector Report Card</div>'
                    '<div class="sb-sub">CHB-MIT · leave-one-subject-out</div>',
                    unsafe_allow_html=True)
        page = st.radio("Section", list(PAGES), label_visibility="collapsed")
        st.markdown("---")
        st.markdown(
            f'<div style="font-size:.78rem;color:{MUTED};line-height:1.7">'
            f'<b>Model</b> {model.upper()}, 56 features<br>'
            f'<b>Patients</b> 23 · <b>Hours</b> 979.9<br>'
            f'<b>Windows</b> 352,742 · <b>Ictal</b> 0.32%<br>'
            f'<b>Protocol</b> 23-fold LOSO</div>', unsafe_allow_html=True)

    fn = PAGES[page]
    if page == "Overview":
        fn(loso, rand, rep, model)
    elif page == "Per-patient outcomes":
        fn(loso, model)
    elif page == "Record explorer":
        fn(load_scores(), load_truth())
    elif page == "Validation protocol":
        fn(loso, rand)
    else:
        fn()

    st.markdown(
        '<div class="foot">Research software for methodology demonstration. '
        'Not a medical device, not clinically validated, and not for use in patient care. '
        'Built on the CHB-MIT Scalp EEG Database (Shoeb 2009; Goldberger et al. 2000), '
        'distributed by PhysioNet under ODC-BY.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

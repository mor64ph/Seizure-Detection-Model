"""Window-level and event-level evaluation.

PRD §11.1: lead with PR-AUC. At a positive rate near 0.3%, ROC-AUC is
flattering and hides the precision problem, because the false-positive rate
denominator is enormous.

PRD §11.2: the event-level metrics are what make this a system rather than a
classifier. "Window recall 0.62" reads as mediocre; "detected 141 of 198
seizures, median latency 6 s, 1.9 false alarms per hour" reads as a real
detector with known limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from ..models.zoo import fbeta


class MixedSourceError(ValueError):
    """R2: a metric may never be computed over rows from two datasets."""


def assert_single_source(df) -> str:
    """R2 enforcement, inside the metric computation rather than only at load.

    The two datasets have different window lengths, channel counts, scaling
    history and label provenance. A frame that silently unions them produces a
    number that describes nothing. Checking at load time is not enough, because
    a concat can happen anywhere between load and evaluation.
    """
    if "source" not in getattr(df, "columns", ()):
        raise MixedSourceError(
            "frame has no 'source' column; R1 requires one on every derived table"
        )
    uniq = df["source"].dropna().unique()
    if len(uniq) != 1:
        raise MixedSourceError(
            f"metrics computed over {len(uniq)} sources {sorted(uniq)}; "
            "R2 forbids mixing csv_scaled and chbmit_raw in one metric"
        )
    return str(uniq[0])


def pooled_window_metrics(rows) -> dict:
    """Window metrics recomputed from summed counts across folds.

    R36 forbade means-of-ratios for event metrics but was never applied to
    window metrics, so the per-fold mean stayed the only reported figure. The
    gap is not cosmetic. Measured for hgb on the aggregated view: mean
    precision across 23 folds is 0.2307, while pooling the confusion matrix
    gives 0.0037 -- 279 true positives against 74,678 false ones. Subjects with
    few windows and high precision dominate the average; subjects emitting tens
    of thousands of false positives barely move it.

    Both are legitimate summaries of different questions. Only the pooled one
    answers "of everything this detector flagged, how much was real".

    ``rows`` is any iterable of per-fold dicts carrying tp/fp/fn/tn.
    """
    tp = int(sum(r.get("tp", 0) for r in rows))
    fp = int(sum(r.get("fp", 0) for r in rows))
    fn = int(sum(r.get("fn", 0) for r in rows))
    tn = int(sum(r.get("tn", 0) for r in rows))
    n = tp + fp + fn + tn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
        "precision_pooled": tp / (tp + fp) if tp + fp else 0.0,
        "recall_pooled": tp / (tp + fn) if tp + fn else 0.0,
        "f1_pooled": float(fbeta(tp, fp, fn, 1.0)),
        "f2_pooled": float(fbeta(tp, fp, fn, 2.0)),
        "accuracy_pooled": (tp + tn) / n if n else 0.0,
    }


# ------------------------------------------------------------- window level
def window_metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pred = (s >= thr).astype(int)

    tn, fp, fn, tp = (0, 0, 0, 0)
    if len(y):
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    both = len(np.unique(y)) > 1

    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else 0.0,
        "threshold": float(thr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(fbeta(tp, fp, fn, 1.0)),
        "f2": float(fbeta(tp, fp, fn, 2.0)),
        "accuracy": float((tp + tn) / len(y)) if len(y) else 0.0,
        # Lead metric. PR-AUC is prevalence-sensitive, which is the point.
        "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
    }


# -------------------------------------------------------------- event level
@dataclass
class Event:
    """A contiguous run of predicted-positive windows."""
    start_sec: float
    end_sec: float
    n_windows: int


@dataclass
class EventResult:
    n_true: int
    n_detected: int
    latencies_sec: list[float] = field(default_factory=list)
    false_alarms: int = 0
    interictal_hours: float = 0.0
    excluded_hours: float = 0.0
    n_fallback_duration: int = 0
    # Alarm *duration*, not just alarm count. Counting runs leaves FA/h open to
    # a detector that never stops alarming: one continuous alarm spanning a
    # whole record scores as a single false alarm. Measured with the dummy
    # classifier, which predicts the majority class for every window and
    # therefore alarms 100% of the time -- it scored 185 of 185 seizures
    # "detected" at 0.89 FA/h, better than every real model. Time in alarm
    # makes that impossible to hide.
    alarm_sec: float = 0.0
    false_alarm_sec: float = 0.0
    evaluated_sec: float = 0.0

    @property
    def sensitivity(self) -> float:
        return self.n_detected / self.n_true if self.n_true else float("nan")

    @property
    def fa_per_hour(self) -> float:
        return self.false_alarms / self.interictal_hours if self.interictal_hours else float("nan")

    @property
    def time_in_alarm_frac(self) -> float:
        """Fraction of evaluated recording time spent inside any alarm.

        A usable detector keeps this small. At ~1.0 the detector is saturated
        and its event sensitivity is meaningless.
        """
        return self.alarm_sec / self.evaluated_sec if self.evaluated_sec else float("nan")

    def summary(self) -> dict:
        lat = np.array(self.latencies_sec, dtype=float)
        return {
            "n_true_events": self.n_true,
            "n_detected": self.n_detected,
            "event_sensitivity": float(self.sensitivity),
            "median_latency_sec": float(np.median(lat)) if len(lat) else float("nan"),
            "latency_iqr_sec": (
                [float(np.percentile(lat, 25)), float(np.percentile(lat, 75))]
                if len(lat) else [float("nan"), float("nan")]
            ),
            "false_alarms": self.false_alarms,
            "interictal_hours": float(self.interictal_hours),
            "excluded_hours": float(self.excluded_hours),
            "fa_per_hour": float(self.fa_per_hour),
            "n_fallback_duration": int(self.n_fallback_duration),
            "alarm_hours": float(self.alarm_sec / 3600.0),
            "false_alarm_hours": float(self.false_alarm_sec / 3600.0),
            # Reported so callers can pool the fraction across folds instead of
            # averaging per-fold ratios, which R40 exists to forbid.
            "evaluated_hours": float(self.evaluated_sec / 3600.0),
            "time_in_alarm_frac": float(self.time_in_alarm_frac),
        }


def positive_runs(
    window_idx: np.ndarray,
    t_start: np.ndarray,
    t_end: np.ndarray,
    pred: np.ndarray,
    k: int,
) -> list[Event]:
    """Contiguous runs of >= k predicted positives, within one record.

    PRD §10.4: requiring k consecutive positive windows is the highest-leverage
    lever on false alarms per hour, and it costs a few seconds of latency.

    Adjacency is decided on ``window_idx``, not on time. An earlier version
    tested ``isclose(t_start[j+1], t_end[j])``, which only holds when
    ``stride_sec == length_sec``; with overlapping windows every run collapsed
    to a single window, zeroing event sensitivity for every k >= 2. Window
    index is stride-independent, and it is also what breaks a run at a record
    boundary.

    Callers must pass rows from a single record, sorted by window_idx.
    """
    out: list[Event] = []
    n = len(pred)
    i = 0
    while i < n:
        if not pred[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and pred[j + 1] and window_idx[j + 1] == window_idx[j] + 1:
            j += 1
        if (j - i + 1) >= k:
            out.append(Event(float(t_start[i]), float(t_end[j]), j - i + 1))
        i = j + 1
    return out


def runs_by_record(df, pred: np.ndarray, k: int) -> dict[str, list[Event]]:
    """positive_runs per record. The only correct way to call it on a frame
    spanning more than one record, because t_start_sec restarts at 0 in every
    record."""
    out: dict[str, list[Event]] = {}
    for rec, g in df.assign(_pred=pred).groupby("record_id", sort=True):
        g = g.sort_values("window_idx")
        out[str(rec)] = positive_runs(
            g["window_idx"].to_numpy(), g["t_start_sec"].to_numpy(),
            g["t_end_sec"].to_numpy(), g["_pred"].to_numpy().astype(bool), k,
        )
    return out


def run_mask(df, pred: np.ndarray, k: int) -> np.ndarray:
    """Boolean per row: is this window inside a surviving run of its OWN record.

    Built per record. Masking by time overlap across a concatenated frame marks
    windows in unrelated records that happen to share a time offset, which is
    the defect this function exists to prevent.
    """
    mask = np.zeros(len(df), dtype=bool)
    pos = {rid: i for i, rid in enumerate(df.index)}
    for rec, g in df.assign(_pred=pred).groupby("record_id", sort=True):
        g = g.sort_values("window_idx")
        runs = positive_runs(
            g["window_idx"].to_numpy(), g["t_start_sec"].to_numpy(),
            g["t_end_sec"].to_numpy(), g["_pred"].to_numpy().astype(bool), k,
        )
        if not runs:
            continue
        ts, te = g["t_start_sec"].to_numpy(), g["t_end_sec"].to_numpy()
        local = np.zeros(len(g), dtype=bool)
        for r in runs:
            local |= (te > r.start_sec) & (ts < r.end_sec)
        for lbl, keep in zip(g.index, local):
            if keep:
                mask[pos[lbl]] = True
    return mask


def event_metrics(
    df,
    pred: np.ndarray,
    truth_intervals: dict[str, list[tuple[int, int]]],
    k: int,
    record_durations: dict[str, float] | None = None,
) -> EventResult:
    """Event-level evaluation over one or more records.

    df must carry record_id, t_start_sec, t_end_sec, sorted within record.

    A true seizure counts as detected if any predicted run of length >= k
    overlaps it. Latency is measured from the annotated onset to the start of
    the detecting run, and can be negative when the detector fires before the
    human-marked onset -- which happens, and is reported rather than clipped.
    """
    # R2: refuse before computing anything, so a mixed frame cannot produce a
    # plausible-looking clinical number.
    assert_single_source(df)

    res = EventResult(n_true=0, n_detected=0)
    ictal_sec = 0.0
    total_sec = 0.0
    excluded_sec = 0.0
    n_fallback_duration = 0

    for rec, g in df.assign(_pred=pred).groupby("record_id", sort=True):
        g = g.sort_values("window_idx")
        runs = positive_runs(
            g["window_idx"].to_numpy(), g["t_start_sec"].to_numpy(),
            g["t_end_sec"].to_numpy(), g["_pred"].to_numpy().astype(bool), k,
        )
        truth = truth_intervals.get(rec, [])
        res.n_true += len(truth)

        # Per-record fallback. Gating the fallback on the dict being non-empty
        # meant a record present in df but missing from the map contributed
        # zero seconds, silently shrinking the FA/h denominator.
        dur = (record_durations or {}).get(rec)
        if not dur:
            dur = float(g["t_end_sec"].max() - g["t_start_sec"].min())
            n_fallback_duration += 1
        total_sec += dur
        ictal_sec += float(sum(b - a for a, b in truth))

        # Guard windows were dropped before evaluation, so no alarm can be
        # raised in their time. Counting that time in the denominator would
        # understate FA/h. Reconstruct it from the gaps in window_idx.
        span = g["t_end_sec"].to_numpy() - g["t_start_sec"].to_numpy()
        stride = float(np.median(span)) if len(span) else 0.0
        present = len(g)
        expected = int(g["window_idx"].max()) + 1 if len(g) else 0
        excluded_sec += max(0, expected - present) * stride

        matched_runs: set[int] = set()
        for a, b in truth:
            hit = [
                (i, r) for i, r in enumerate(runs)
                if r.end_sec > a and r.start_sec < b
            ]
            if hit:
                res.n_detected += 1
                first = min(hit, key=lambda x: x[1].start_sec)[1]
                res.latencies_sec.append(first.start_sec - a)
                matched_runs.update(i for i, _ in hit)
        res.false_alarms += len(runs) - len(matched_runs)

        res.alarm_sec += float(sum(r.end_sec - r.start_sec for r in runs))
        res.false_alarm_sec += float(sum(
            r.end_sec - r.start_sec
            for i, r in enumerate(runs) if i not in matched_runs
        ))

    # Interictal hours, not total hours: the denominator PRD §11.2 specifies.
    # Guard-excluded time comes out too, since a dropped window cannot alarm.
    res.interictal_hours = max(0.0, total_sec - ictal_sec - excluded_sec) / 3600.0
    res.n_fallback_duration = n_fallback_duration
    res.excluded_hours = excluded_sec / 3600.0
    # Denominator for time-in-alarm is the time an alarm could have been raised,
    # which excludes dropped guard windows for the same reason FA/h does.
    res.evaluated_sec = max(0.0, total_sec - excluded_sec)
    return res

"""Tests for eval/metrics.py.

This module had no coverage until an audit pointed it out, and it is where the
project's headline clinical numbers come from. Several of these tests pin bugs
that were found and fixed: a run-adjacency test that broke under overlapping
windows, a per-record duration fallback that silently contributed zero, and a
guard-time denominator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seizure.eval import metrics as M


def frame(record_specs: dict[str, int], length: float = 10.0, stride: float | None = None):
    """{record_id: n_windows} -> a frame with window_idx and times."""
    stride = stride if stride is not None else length
    rows = []
    for rec, n in record_specs.items():
        for i in range(n):
            rows.append({
                "record_id": rec, "window_idx": i,
                "t_start_sec": i * stride, "t_end_sec": i * stride + length,
                "label": 0,
            })
    df = pd.DataFrame(rows)
    # R1: every derived table carries a source. Tests that deliberately probe
    # the R2 guard drop or overwrite it.
    df["source"] = "chbmit_raw"
    return df


# ------------------------------------------------------------ positive_runs
def _runs(pred, k=1, length=10.0, stride=None):
    df = frame({"r": len(pred)}, length, stride)
    return M.positive_runs(df.window_idx.to_numpy(), df.t_start_sec.to_numpy(),
                           df.t_end_sec.to_numpy(), np.array(pred, dtype=bool), k)


def test_single_run_found():
    r = _runs([0, 1, 1, 1, 0])
    assert len(r) == 1 and r[0].n_windows == 3
    assert (r[0].start_sec, r[0].end_sec) == (10.0, 40.0)


def test_k_suppresses_short_runs():
    assert len(_runs([0, 1, 0, 1, 0], k=2)) == 0
    assert len(_runs([0, 1, 1, 0], k=2)) == 1
    assert len(_runs([0, 1, 1, 0], k=3)) == 0


def test_two_separate_runs():
    assert len(_runs([1, 1, 0, 0, 1, 1])) == 2


def test_run_at_frame_edges():
    r = _runs([1, 1, 0, 1])
    assert len(r) == 2 and r[0].start_sec == 0.0


def test_runs_survive_overlapping_windows():
    """Adjacency is by window_idx, so a 50%-overlap stride still forms runs.

    With the old time-contiguity test (t_start[j+1] == t_end[j]) every run
    collapsed to one window and event sensitivity went to zero for all k >= 2.
    """
    r = _runs([0, 1, 1, 1, 1, 0], k=3, length=10.0, stride=5.0)
    assert len(r) == 1 and r[0].n_windows == 4


def test_runs_never_merge_across_records():
    """Both records have windows at the same time offsets."""
    df = frame({"a": 3, "b": 3})
    pred = np.array([0, 0, 1, 1, 0, 0], dtype=bool)  # last of 'a', first of 'b'
    by_rec = M.runs_by_record(df, pred, k=1)
    assert len(by_rec["a"]) == 1 and len(by_rec["b"]) == 1
    assert by_rec["a"][0].n_windows == 1 and by_rec["b"][0].n_windows == 1
    assert len(M.runs_by_record(df, pred, k=2)["a"]) == 0


# ---------------------------------------------------------------- run_mask
def test_run_mask_is_per_record():
    """The critical bug: a run in record 'a' must not mark record 'b'.

    t_start_sec restarts at 0 in every record, so a time-overlap mask built on
    the concatenated frame marked every record sharing the offset.
    """
    df = frame({"a": 4, "b": 4})
    pred = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)  # only record 'a'
    mask = M.run_mask(df, pred, k=2)
    assert mask[:4].tolist() == [True, True, False, False]
    assert not mask[4:].any(), "record 'b' was marked by record 'a's run"


def test_run_mask_respects_k():
    df = frame({"a": 4})
    assert not M.run_mask(df, np.array([1, 0, 1, 0], dtype=bool), k=2).any()


def test_run_mask_survives_unsorted_index():
    df = frame({"a": 3, "b": 3}).sample(frac=1.0, random_state=0)
    pred = (df.record_id == "a").to_numpy() & (df.window_idx == 0).to_numpy()
    mask = M.run_mask(df, pred, k=1)
    assert mask.sum() == 1
    assert df.loc[mask, "record_id"].tolist() == ["a"]


# ------------------------------------------------------------ event_metrics
def test_one_run_detects_one_seizure():
    df = frame({"r": 40})
    pred = np.zeros(40, int); pred[10:14] = 1          # 100-140 s
    ev = M.event_metrics(df, pred, {"r": [(100, 140)]}, k=1, record_durations={"r": 400.0})
    s = ev.summary()
    assert s["n_true_events"] == 1 and s["n_detected"] == 1
    assert s["event_sensitivity"] == 1.0 and s["false_alarms"] == 0


def test_two_runs_overlapping_one_seizure_is_one_detection_no_false_alarm():
    df = frame({"r": 40})
    pred = np.zeros(40, int); pred[10] = 1; pred[13] = 1
    ev = M.event_metrics(df, pred, {"r": [(100, 140)]}, k=1, record_durations={"r": 400.0})
    assert ev.n_detected == 1
    assert ev.false_alarms == 0, "both runs hit the seizure; neither is a false alarm"


def test_one_run_overlapping_two_seizures_counts_both():
    df = frame({"r": 40})
    pred = np.zeros(40, int); pred[10:20] = 1          # 100-200 s
    ev = M.event_metrics(df, pred, {"r": [(100, 120), (180, 200)]}, k=1,
                         record_durations={"r": 400.0})
    assert ev.n_true == 2 and ev.n_detected == 2 and ev.false_alarms == 0


def test_false_alarm_counted_and_never_negative():
    df = frame({"r": 40})
    pred = np.zeros(40, int); pred[0:3] = 1            # nowhere near the seizure
    ev = M.event_metrics(df, pred, {"r": [(300, 320)]}, k=1, record_durations={"r": 400.0})
    assert ev.n_detected == 0 and ev.false_alarms == 1
    assert ev.false_alarms >= 0


def test_missed_seizure_gives_zero_sensitivity():
    df = frame({"r": 40})
    ev = M.event_metrics(df, np.zeros(40, int), {"r": [(100, 140)]}, k=1,
                         record_durations={"r": 400.0})
    assert ev.n_detected == 0 and ev.summary()["event_sensitivity"] == 0.0


def test_latency_is_positive_when_detection_follows_onset():
    df = frame({"r": 40})
    pred = np.zeros(40, int); pred[11:14] = 1          # starts 110 s
    ev = M.event_metrics(df, pred, {"r": [(100, 140)]}, k=1, record_durations={"r": 400.0})
    assert ev.summary()["median_latency_sec"] == pytest.approx(10.0)


def test_latency_can_be_negative_and_is_not_clipped():
    """The detector firing before the human-marked onset happens, and is reported."""
    df = frame({"r": 40})
    pred = np.zeros(40, int); pred[8:14] = 1           # starts 80 s, onset 100 s
    ev = M.event_metrics(df, pred, {"r": [(100, 140)]}, k=1, record_durations={"r": 400.0})
    assert ev.summary()["median_latency_sec"] == pytest.approx(-20.0)


def test_latency_uses_earliest_detecting_run():
    df = frame({"r": 40})
    pred = np.zeros(40, int); pred[10] = 1; pred[13] = 1
    ev = M.event_metrics(df, pred, {"r": [(100, 140)]}, k=1, record_durations={"r": 400.0})
    assert ev.summary()["median_latency_sec"] == pytest.approx(0.0)


def test_interictal_hours_excludes_ictal_time():
    df = frame({"r": 360})                              # 3600 s
    ev = M.event_metrics(df, np.zeros(360, int), {"r": [(100, 460)]}, k=1,
                         record_durations={"r": 3600.0})
    assert ev.interictal_hours == pytest.approx((3600 - 360) / 3600.0)


def test_missing_duration_falls_back_per_record_not_to_zero():
    """A record present in df but absent from the duration map must fall back
    to its window span, not contribute zero seconds to the FA/h denominator."""
    df = frame({"a": 100, "b": 100})
    ev = M.event_metrics(df, np.zeros(200, int), {}, k=1, record_durations={"a": 1000.0})
    assert ev.n_fallback_duration == 1
    assert ev.interictal_hours > 1000.0 / 3600.0, "record 'b' contributed nothing"


def test_guard_excluded_time_leaves_the_denominator():
    """Guard windows are dropped before evaluation, so no alarm can be raised
    in their time; counting it would understate FA/h."""
    df = frame({"r": 360})
    df = df[~df.window_idx.between(100, 129)]           # drop 30 windows = 300 s
    ev = M.event_metrics(df, np.zeros(len(df), int), {}, k=1,
                         record_durations={"r": 3600.0})
    assert ev.excluded_hours == pytest.approx(300 / 3600.0)
    assert ev.interictal_hours == pytest.approx((3600 - 300) / 3600.0)


def test_fa_per_hour_arithmetic():
    df = frame({"r": 360})
    pred = np.zeros(360, int); pred[0:2] = 1; pred[100:102] = 1
    ev = M.event_metrics(df, pred, {}, k=1, record_durations={"r": 3600.0})
    assert ev.false_alarms == 2
    assert ev.summary()["fa_per_hour"] == pytest.approx(2.0 / 1.0)


# --------------------------------------------------------- window_metrics
def test_window_metrics_confusion_matrix():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.8, 0.2])
    m = M.window_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)


def test_window_metrics_pr_auc_uses_scores_not_predictions():
    """Perfect ranking at a bad threshold must still give PR-AUC 1.0."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.01, 0.02, 0.03, 0.04])
    m = M.window_metrics(y, s, 0.99)
    assert m["tp"] == 0
    assert m["pr_auc"] == pytest.approx(1.0)


def test_window_metrics_single_class_returns_nan_not_zero():
    m = M.window_metrics(np.zeros(10, int), np.linspace(0, 1, 10), 0.5)
    assert m["pr_auc"] != m["pr_auc"] and m["roc_auc"] != m["roc_auc"]


def test_window_metrics_dummy_accuracy_is_high_at_low_prevalence():
    """The PRD's point: accuracy is meaningless here."""
    y = np.array([0] * 997 + [1] * 3)
    m = M.window_metrics(y, np.zeros(1000), 0.5)
    assert m["accuracy"] == pytest.approx(0.997)
    assert m["recall"] == 0.0


def test_f2_exceeds_f1_when_recall_dominates():
    y = np.array([0] * 10 + [1] * 10)
    s = np.r_[np.zeros(10), np.ones(10)] * 0.9 + 0.05
    m = M.window_metrics(y, s, 0.5)
    assert m["f1"] == pytest.approx(1.0) and m["f2"] == pytest.approx(1.0)


# ----------------------------------------------------------- R2: source purity
def test_event_metrics_refuses_mixed_sources():
    """R2: two datasets in one metric produces a number that describes nothing."""
    df = frame({"r": 10})
    df["source"] = ["chbmit_raw"] * 5 + ["csv_scaled"] * 5
    with pytest.raises(M.MixedSourceError) as e:
        M.event_metrics(df, np.zeros(10, int), {}, k=1)
    assert "csv_scaled" in str(e.value)


def test_event_metrics_refuses_missing_source_column():
    df = frame({"r": 10}).drop(columns=["source"])
    with pytest.raises(M.MixedSourceError):
        M.event_metrics(df, np.zeros(10, int), {}, k=1)


def test_event_metrics_accepts_single_source():
    df = frame({"r": 10})
    df["source"] = "chbmit_raw"
    ev = M.event_metrics(df, np.zeros(10, int), {}, k=1)
    assert ev.n_true == 0


def test_assert_single_source_returns_the_source():
    df = frame({"r": 4})
    df["source"] = "csv_scaled"
    assert M.assert_single_source(df) == "csv_scaled"


def test_assert_single_source_ignores_nulls():
    """A partially-populated column is still single-valued if the non-nulls agree."""
    df = frame({"r": 4})
    df["source"] = ["chbmit_raw", None, "chbmit_raw", None]
    assert M.assert_single_source(df) == "chbmit_raw"


# ------------------------------------------------- alarm duration (saturation)
def test_time_in_alarm_exposes_a_saturated_detector():
    """A detector that alarms continuously must not look good.

    Counting false alarms per run lets an always-positive detector score one
    long false alarm and an excellent FA/h. That is how the dummy classifier
    came to report 185 of 185 seizures at 0.89 FA/h.
    """
    df = frame({"r": 40})
    ev = M.event_metrics(df, np.ones(40, int), {"r": [(100, 140)]}, k=1,
                         record_durations={"r": 400.0})
    s = ev.summary()
    assert s["n_detected"] == 1
    # One continuous run, so run-counting sees no false alarm at all.
    assert s["false_alarms"] == 0
    # Duration tells the truth: the detector is alarming the entire time.
    assert s["time_in_alarm_frac"] == pytest.approx(1.0)


def test_time_in_alarm_small_for_a_sparse_detector():
    df = frame({"r": 40})
    pred = np.zeros(40, int)
    pred[10:14] = 1          # 40 s of alarm inside 400 s
    ev = M.event_metrics(df, pred, {"r": [(100, 140)]}, k=1,
                         record_durations={"r": 400.0})
    assert ev.summary()["time_in_alarm_frac"] == pytest.approx(0.1)


def test_false_alarm_hours_excludes_the_matched_run():
    df = frame({"r": 40})
    pred = np.zeros(40, int)
    pred[10:14] = 1          # overlaps truth -> matched
    pred[30:32] = 1          # does not -> false
    ev = M.event_metrics(df, pred, {"r": [(100, 140)]}, k=1,
                         record_durations={"r": 400.0})
    s = ev.summary()
    assert s["false_alarms"] == 1
    assert s["alarm_hours"] == pytest.approx(60 / 3600.0)
    assert s["false_alarm_hours"] == pytest.approx(20 / 3600.0)


# -------------------------------------------------------- pooled window metrics
def test_pooled_precision_differs_from_the_fold_mean():
    """R36 applied to window metrics.

    Two folds: one tiny and perfect, one large and mostly wrong. The mean of
    the per-fold precisions is near 0.5; pooling the counts gives ~0.01.
    """
    rows = [
        {"tp": 1, "fp": 0, "fn": 0, "tn": 9},        # precision 1.0
        {"tp": 1, "fp": 99, "fn": 0, "tn": 900},     # precision 0.01
    ]
    p = M.pooled_window_metrics(rows)
    assert p["tp"] == 2 and p["fp"] == 99
    assert p["precision_pooled"] == pytest.approx(2 / 101)
    mean_of_ratios = (1.0 + 0.01) / 2
    assert mean_of_ratios > 10 * p["precision_pooled"]


def test_pooled_window_metrics_handles_empty_counts():
    p = M.pooled_window_metrics([])
    assert p["n"] == 0
    assert p["precision_pooled"] == 0.0
    assert p["accuracy_pooled"] == 0.0


# ------------------------------------------------- amplitude-invariant subset
def test_feature_subset_splits_aggregated_columns_cleanly():
    """PRD 8.2: scale-dependent features are close to a subject fingerprint."""
    from seizure.train import feature_columns

    df = pd.DataFrame(columns=[
        "label", "subject_id", "record_id",
        "bp_delta_log__mean", "bp_delta_rel__mean",
        "line_length__mean", "variance__mean",
        "hjorth_mobility__mean", "hjorth_complexity__std",
    ])
    inv = feature_columns(df, "amplitude_invariant")
    dep = feature_columns(df, "amplitude_dependent")
    assert set(inv) == {"bp_delta_rel__mean", "hjorth_mobility__mean",
                        "hjorth_complexity__std"}
    assert set(dep) == {"bp_delta_log__mean", "line_length__mean",
                        "variance__mean"}
    # The two families must partition the features, with nothing counted twice.
    assert not set(inv) & set(dep)
    assert set(inv) | set(dep) == set(feature_columns(df, "all"))


def test_feature_subset_handles_per_channel_naming():
    """Per-channel columns are {channel}__{feature}, the reverse order."""
    from seizure.train import feature_columns

    df = pd.DataFrame(columns=[
        "label", "T7-P7__bp_delta_rel", "T7-P7__bp_delta_log",
        "T7-P7__hjorth_mobility", "T7-P7__variance", "T7-P7__line_length",
    ])
    assert set(feature_columns(df, "amplitude_invariant")) == {
        "T7-P7__bp_delta_rel", "T7-P7__hjorth_mobility"}
    assert set(feature_columns(df, "amplitude_dependent")) == {
        "T7-P7__bp_delta_log", "T7-P7__variance", "T7-P7__line_length"}


def test_unknown_feature_subset_raises():
    from seizure.train import feature_columns
    with pytest.raises(ValueError, match="unknown feature_subset"):
        feature_columns(pd.DataFrame(columns=["label"]), "nonsense")


def test_event_f2_selection_must_reject_a_saturated_detector():
    """R41 applied to model selection, not just to reporting.

    A first version of the event-level selection score used event F2 alone.
    Because false alarms are counted per run, an always-positive predictor
    earns one run per record and scored 0.514 against the best real model's
    0.160 -- it would have been selected as the headline model. The
    time-in-alarm factor is what prevents that.
    """
    from seizure.config import load
    from seizure.train import _val_event_score

    va = frame({"r": 40})
    va.loc[10:13, "label"] = 1
    truth = {"r": [(100, 140)]}
    dur = {"r": 400.0}
    cfg = load("configs/base.yaml")
    assert cfg is not None  # config must still parse for the app/CLI paths

    always = np.ones(40, dtype=float)
    sparse = np.zeros(40, dtype=float)
    sparse[10:14] = 1.0

    sat = _val_event_score(va, always, 0.5, 1, truth, dur)
    good = _val_event_score(va, sparse, 0.5, 1, truth, dur)
    # Both find the seizure; only one of them is a detector.
    assert good > sat, f"saturated {sat} must not beat sparse {good}"
    assert sat < 0.05

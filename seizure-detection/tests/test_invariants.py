"""The invariant tests: registry, window labelling, splits, portability.

These are the tests that would have caught the bugs this project actually hit,
so each one names the rule it enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seizure.labels import registry, seizures_annot as sa, summary_parser as sp
from seizure.models import zoo
from seizure.signal import channels as ch, windows as W
from seizure.splits import protocols as P

SRC = Path(__file__).resolve().parents[1] / "src"


# ----------------------------------------------------------- R5: registry
def test_registry_has_23_subjects_across_24_cases():
    df = registry.build()
    assert len(df) == 24
    assert df.subject_id.nunique() == 23


def test_chb01_and_chb21_are_one_subject():
    df = registry.build()
    assert set(df.loc[df.case_id.isin(["chb01", "chb21"]), "subject_id"]) == {"sub01"}


def test_chb24_tolerates_missing_demographics():
    df = registry.build("Case\tGender\tAge (years)\n\nchb01\tF\t11\n")
    assert df.loc[df.case_id == "chb24", "gender"].isna().all()
    registry.validate(df)


def test_chb17_naming_variant_parses():
    assert registry.case_of_record("chb17a_03") == "chb17"
    assert registry.case_of_record("chb17b_69") == "chb17"
    assert registry.subject_of_record("chb21_01") == "sub01"


def test_plus_suffix_record_parses():
    """chb02_16+.edf is a real filename."""
    assert registry.case_of_record("chb02_16+") == "chb02"


def test_unrecognised_record_id_raises():
    with pytest.raises(ValueError):
        registry.case_of_record("not-a-record")


# --------------------------------------------------- R31: channel allowlist
def test_non_eeg_channels_are_rejected():
    for bad in ["ECG", "VNS", "EKG1-CHIN", "LOC-ROC", "LUE-RAE", "-", ".", ""]:
        assert not ch.is_scalp_eeg(bad), bad


def test_referential_and_bare_montages_are_rejected():
    """chb12_27 uses -CS2 derivations; chb12_28/29 use bare electrode names."""
    for bad in ["F7-CS2", "FP1-CS2", "F7", "T7", "FP1"]:
        assert not ch.is_scalp_eeg(bad), bad


def test_bipolar_labels_are_accepted():
    for good in ["FP1-F7", "FZ-CZ", "T8-P8", "FT9-FT10", "P7-T7", "T7-FT9"]:
        assert ch.is_scalp_eeg(good), good


def test_typo_channel_01_is_rejected():
    """'01' (zero-one) appears instead of 'O1' (letter O) in two files."""
    assert not ch.is_scalp_eeg("01-F7")


def test_duplicate_labels_keep_first_occurrence():
    out = ch.classify(["FP1-F7", "T8-P8", "ECG", "T8-P8", "-"])
    assert out["kept"] == ["FP1-F7", "T8-P8"]
    assert out["duplicate"] == ["T8-P8"]
    assert "ECG" in out["non_eeg"] and "-" in out["non_eeg"]


def test_projection_is_by_label_not_position():
    """R16. A file listing channels in a different order must still project
    onto the canonical order."""
    canonical = ["FP1-F7", "F7-T7", "T7-P7"]
    sel, missing = ch.project(["T7-P7", "FP1-F7", "F7-T7"], canonical)
    assert sel == canonical and missing == []


def test_projection_reports_missing_and_skips():
    sel, missing = ch.project(["FP1-F7"], ["FP1-F7", "FZ-CZ"], "skip_file")
    assert sel == [] and missing == ["FZ-CZ"]


# -------------------------------------------- PRD 6.2: window labelling
SPEC = W.WindowSpec(10.0, 10.0, 0.5, 30.0, 256)


def test_window_fully_inside_seizure_is_ictal():
    lab, frac = W.label_window(3000, 3010, [(2996, 3036)], SPEC)
    assert lab == W.ICTAL and frac == pytest.approx(1.0)


def test_window_at_exactly_half_overlap_is_ictal():
    """The boundary condition: >= min_overlap_frac, not >."""
    lab, frac = W.label_window(3030, 3040, [(2996, 3035)], SPEC)
    assert frac == pytest.approx(0.5) and lab == W.ICTAL


def test_window_just_below_half_overlap_is_guard_not_negative():
    lab, frac = W.label_window(3030, 3040, [(2996, 3034)], SPEC)
    assert frac == pytest.approx(0.4) and lab == W.GUARD


def test_window_far_away_is_interictal():
    lab, frac = W.label_window(100, 110, [(2996, 3036)], SPEC)
    assert lab == W.INTERICTAL and frac == 0.0


def test_window_inside_guard_band_is_excluded():
    lab, _ = W.label_window(2970, 2980, [(2996, 3036)], SPEC)
    assert lab == W.GUARD


def test_window_just_outside_guard_band_is_interictal():
    lab, _ = W.label_window(2950, 2960, [(2996, 3036)], SPEC)
    assert lab == W.INTERICTAL


def test_disputed_span_forced_to_guard():
    """chb24_21: the 400 s the two label sources disagree about trains nothing."""
    lab, _ = W.label_window(2500, 2510, [(2804, 2872)], SPEC, extra_guard=[(2404, 2804)])
    assert lab == W.GUARD


def test_windows_never_span_a_record_boundary():
    """A trailing partial window is dropped, not zero-padded."""
    b = W.window_bounds(2560 * 3 + 100, SPEC)
    assert len(b) == 3 and b[-1][1] <= 2560 * 3 + 100


def test_window_count_matches_one_hour_record():
    assert len(W.window_bounds(3600 * 256, SPEC)) == 360


def test_overlap_sensitivity_is_monotone():
    rows = [{"overlap_frac": f} for f in np.linspace(0, 1, 101)]
    tab = W.overlap_sensitivity(rows)
    counts = [t["n_positive"] for t in tab]
    assert counts == sorted(counts, reverse=True)


# ------------------------------------------------- R6/R7/R10: splitting
def _frame(n_subj=6, n_per=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        for i in range(n_per):
            rows.append({
                "subject_id": f"sub{s:02d}", "case_id": f"chb{s:02d}",
                "record_id": f"chb{s:02d}_01", "window_idx": i,
                "t_start_sec": i * 10.0, "t_end_sec": i * 10.0 + 10.0,
                "label": int(i % 10 == 0), "overlap_frac": 0.0,
                "f0": rng.normal(), "f1": rng.normal(),
            })
    return pd.DataFrame(rows)


def test_loso_gives_one_fold_per_subject_and_disjoint_roles():
    df = _frame()
    _, folds = P.assign(df, "loso", n_val=2, seed=42, feature_cols=["f0", "f1"])
    assert len(folds) == df.subject_id.nunique()
    for f in folds:
        sets = [set(f.test_subjects), set(f.val_subjects), set(f.train_subjects)]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_loso_is_deterministic_for_a_seed():
    df = _frame()
    a = P.assign(df, "loso", 2, 42)[1]
    b = P.assign(df, "loso", 2, 42)[1]
    assert a == b


def test_chb01_and_chb21_never_split_across_roles():
    """R5, the leakage this whole project is about."""
    rows = []
    for case, sub in [("chb01", "sub01"), ("chb21", "sub01"), ("chb02", "sub02"),
                      ("chb03", "sub03"), ("chb04", "sub04")]:
        for i in range(10):
            rows.append({"subject_id": sub, "case_id": case, "label": i % 5 == 0,
                         "f0": 0.0})
    df = pd.DataFrame(rows)
    assign, folds = P.assign(df, "loso", n_val=1, seed=1)
    for f in folds:
        roles = {f.role_of(s) for s in df.loc[df.case_id.isin(["chb01", "chb21"]), "subject_id"]}
        assert len(roles) == 1, "chb01 and chb21 landed in different roles"


def test_duplicate_rows_are_refused_before_splitting():
    """R6: measured to matter -- removing duplicates alone moved PR-AUC 0.992 -> 0.899."""
    df = _frame()
    dup = pd.concat([df, df.iloc[:5]], ignore_index=True)
    with pytest.raises(P.DuplicateRowsError):
        P.assign(dup, "loso", 2, 42, feature_cols=["f0", "f1"])


def test_guard_rows_are_dropped_from_evaluation():
    df = _frame()
    df.loc[:4, "label"] = W.GUARD
    assert (P.drop_guard(df)["label"] != W.GUARD).all()
    assert len(P.drop_guard(df)) == len(df) - 5


def test_downsampling_preserves_every_positive():
    df = _frame(n_subj=3, n_per=100)
    out = P.downsample_negatives(df, ratio=2)
    assert out.label.sum() == df.label.sum()
    assert (out.label == 0).sum() <= 2 * df.label.sum()


def test_downsampling_disabled_is_identity():
    df = _frame()
    assert len(P.downsample_negatives(df, None)) == len(df)


# --------------------------------------------------- PRD 10.3: thresholds
def test_threshold_is_not_half():
    y = np.array([0] * 95 + [1] * 5)
    s = np.concatenate([np.linspace(0, .3, 95), np.linspace(.31, .4, 5)])
    thr, f2 = zoo.pick_threshold(y, s, "f2")
    assert thr != 0.5 and f2 > 0.9


def test_threshold_degenerate_input_is_safe():
    thr, f2 = zoo.pick_threshold(np.zeros(10, int), np.zeros(10), "f2")
    assert thr == 0.5 and f2 == 0.0


def test_f2_weights_recall_above_precision():
    assert zoo.fbeta(tp=8, fp=8, fn=2, beta=2.0) > zoo.fbeta(tp=8, fp=2, fn=8, beta=2.0)


def test_dummy_scores_do_not_crash():
    m = zoo.build("dummy")
    X = np.random.default_rng(0).normal(size=(20, 3))
    m.fit(X, np.array([0] * 18 + [1] * 2))
    assert len(zoo.scores(m, X)) == 20


# ------------------------------------------------------ R20: portability
def test_no_rdd_apis_under_src():
    """PRD §3.3: the code must run unchanged on Spark Connect / serverless.

    Uses the shared scanner, which strips comments and string literals first.
    A naive grep cannot tell code that uses `.rdd` from a docstring explaining
    that `.rdd` is banned, and failing on the latter punishes documentation.
    """
    from seizure.spark.fanout import portability_report

    rep = portability_report(SRC)
    assert rep["rdd_api_usages"] == [], rep["rdd_api_usages"]
    assert rep["files_scanned"] > 20, "scanner found almost nothing; check the path"


def test_pure_python_modules_have_no_spark_imports():
    """R20: features/, signal/, labels/, eval/ stay Spark-free."""
    from seizure.spark.fanout import portability_report

    rep = portability_report(SRC)
    assert rep["spark_imports_in_pure_modules"] == [], rep["spark_imports_in_pure_modules"]
    assert rep["portable"] is True


def test_code_only_strips_prose_but_keeps_code():
    """The scanner's own contract."""
    from seizure.spark.fanout import code_only

    src = '"""A docstring mentioning sparkContext."""\n# and a comment with .rdd\nx = 1\n'
    stripped = code_only(src)
    assert "sparkContext" not in stripped and ".rdd" not in stripped
    assert "x" in stripped

    real = "y = df.rdd.map(f)\n"
    assert ".rdd" in code_only(real), "must still catch genuine usage"


def test_no_hardcoded_dimensions_in_src():
    """R17: no bare 23 / 18 / 322 / 2048 magic numbers driving shape."""
    offenders = []
    for p in SRC.rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            if re.search(r"(n_channels|n_feat\w*|width|shape)\s*=\s*(18|23|322|2048)\b", code):
                offenders.append(f"{p.name}:{i}")
    assert not offenders, offenders


# ------------------------------------------ R29: annotation cross-validation
def test_annotation_decoder_on_golden_bytes():
    """chb01_03.edf.seizures, verbatim. Published seizure 2996-3036 s."""
    raw = bytes.fromhex(
        "005817fc2323207469"
        "6d65207265736f6c7574696f6e3a203235360000"
        "ecffffffff010000ec0b000 0b400800 0ec0000002800840000".replace(" ", "")
    )
    got = sa.to_seconds(sa.parse(raw))
    assert got == [(2996, 3036)]


def test_cross_validate_flags_a_count_mismatch():
    probs = sa.cross_validate({"r": [(1, 2)]}, {"r": [(1, 2), (5, 6)]})
    assert len(probs) == 1 and "1 seizures" in probs[0]


def test_cross_validate_tolerates_one_second_rounding():
    assert sa.cross_validate({"r": [(100, 200)]}, {"r": [(101, 199)]}) == []


def test_cross_validate_flags_a_real_shift():
    probs = sa.cross_validate({"r": [(2804, 2872)]}, {"r": [(2404, 2872)]})
    assert len(probs) == 1


# ------------------------------------------- PRD 6.1: summary parser
def test_multi_seizure_block_with_index_typo_is_recovered():
    """chb09_08 labels its second seizure's end as 'Seizure 1 End Time'."""
    text = (
        "File Name: chb09_08.edf\n"
        "File Start Time: 20:22:17\n"
        "File End Time: 24:22:17\n"
        "Number of Seizures in File: 2\n"
        "Seizure 1 Start Time: 2951 seconds\n"
        "Seizure 1 End Time: 3030 seconds\n"
        "Seizure 2 Start Time: 9196 seconds\n"
        "Seizure 1 End Time: 9267 seconds\n"
    )
    recs = sp.parse(text)
    assert len(recs[0].seizures) == 2
    assert [(s.start_sec, s.end_sec) for s in recs[0].seizures] == [(2951, 3030), (9196, 9267)]
    assert recs[0].anomalies, "the index typo must be recorded, not silently accepted"


def test_clock_past_24h_does_not_break_parsing():
    """'24:22:17' would raise if parsed as a time; it must be ignored."""
    text = ("File Name: chb09_08.edf\nFile End Time: 24:22:17\n"
            "Number of Seizures in File: 0\n")
    assert sp.parse(text)[0].seizures == []


def test_unindexed_single_seizure_form():
    text = ("File Name: chb24_21.edf\nNumber of Seizures in File: 1\n"
            "Seizure Start Time: 2804 seconds\nSeizure End Time: 2872 seconds\n")
    s = sp.parse(text)[0].seizures
    assert [(x.start_sec, x.end_sec) for x in s] == [(2804, 2872)]


def test_zero_seizure_record():
    text = "File Name: chb01_01.edf\nNumber of Seizures in File: 0\n"
    assert sp.parse(text)[0].seizures == []


def test_chb17_naming_variant_in_summary():
    text = ("File Name: chb17a_03.edf\nNumber of Seizures in File: 1\n"
            "Seizure Start Time: 100 seconds\nSeizure End Time: 150 seconds\n")
    assert sp.parse(text)[0].record_id == "chb17a_03"


def test_declared_count_mismatch_raises():
    text = ("File Name: chb01_03.edf\nNumber of Seizures in File: 2\n"
            "Seizure Start Time: 10 seconds\nSeizure End Time: 20 seconds\n")
    with pytest.raises(ValueError):
        sp.parse(text)


def test_unrecognised_seizure_line_raises_not_returns_empty():
    text = ("File Name: chb01_03.edf\nNumber of Seizures in File: 1\n"
            "Seizure Onset Marker: 10\n")
    with pytest.raises(ValueError):
        sp.parse(text)


# ------------------------------- PRD 9.3 / R9: inner hyperparameter search
def _grouped(n_subj=6, n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        for i in range(n_per):
            y = int(i % 12 == 0)
            rows.append({
                "subject_id": f"sub{s:02d}",
                "label": y,
                # signal correlated with the label, plus a per-subject offset
                "f0": rng.normal(2.0 if y else 0.0) + s * 5.0,
                "f1": rng.normal(),
            })
    return pd.DataFrame(rows)


def test_search_returns_params_from_the_grid():
    df = _grouped()
    params, score = zoo.search(
        "logreg", df[["f0", "f1"]].to_numpy(), df.label.to_numpy(),
        df.subject_id.to_numpy(), n_splits=3)
    assert set(params) == {"clf__C"}
    assert params["clf__C"] in zoo.GRIDS["logreg"]["clf__C"]
    assert 0.0 <= score <= 1.0


def test_search_uses_groupkfold_not_kfold():
    """R9. GroupKFold must never place a subject on both sides of an inner
    split; a bare KFold would, and would tune on subject identity."""
    import inspect
    src = inspect.getsource(zoo.search)
    assert "GroupKFold" in src
    assert "StratifiedKFold" not in src
    assert "KFold(" not in src.replace("GroupKFold(", "")


def test_search_is_empty_for_dummy():
    df = _grouped()
    params, score = zoo.search(
        "dummy", df[["f0", "f1"]].to_numpy(), df.label.to_numpy(),
        df.subject_id.to_numpy())
    assert params == {} and score != score  # NaN


def test_search_degrades_gracefully_on_one_group():
    df = _grouped(n_subj=1)
    params, _ = zoo.search(
        "logreg", df[["f0", "f1"]].to_numpy(), df.label.to_numpy(),
        df.subject_id.to_numpy())
    assert params == {}, "a single group cannot support a grouped inner split"


def test_search_degrades_gracefully_on_one_class():
    df = _grouped()
    df["label"] = 0
    params, _ = zoo.search(
        "logreg", df[["f0", "f1"]].to_numpy(), df.label.to_numpy(),
        df.subject_id.to_numpy())
    assert params == {}


def test_build_tuned_applies_params():
    m = zoo.build_tuned("logreg", {"clf__C": 0.01})
    assert m.get_params()["clf__C"] == 0.01


def test_build_tuned_with_no_params_matches_default():
    a = zoo.build_tuned("hgb", {}).get_params()
    b = zoo.build("hgb").get_params()
    assert a["learning_rate"] == b["learning_rate"]


def test_every_model_has_a_grid_entry():
    """A model without a grid entry would silently skip the search."""
    assert set(zoo.GRIDS) == set(zoo.MODELS)


def test_grids_are_small_enough_to_run():
    """23 outer folds x 2 views x |grid| x inner_splits fits. Keep it bounded."""
    for name, grid in zoo.GRIDS.items():
        n = 1
        for vals in grid.values():
            n *= len(vals)
        assert n <= 8, f"{name} grid has {n} combinations; too many for 23 folds"

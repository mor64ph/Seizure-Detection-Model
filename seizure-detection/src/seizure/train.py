"""The LOSO training and evaluation loop.

R7/R10: the scaler lives inside the Pipeline so it is fit on training rows
only; the threshold and k are chosen on validation subjects; the test subject
is scored exactly once, at the end of its fold.

PRD §10.2: negatives are downsampled in the TRAINING fold only. Validation and
test are never downsampled, because their class balance is the thing being
measured.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config
from .eval import metrics as M
from .models import zoo
from .splits import protocols as P

KEY_COLS = {
    "window_uid", "case_id", "subject_id", "record_id", "window_idx",
    "t_start_sec", "t_end_sec", "label", "overlap_frac", "config_hash",
    "source", "pseudo_record",
}


# Which feature families survive a change of signal scale. Relative band power
# divides by total power; Hjorth mobility is sqrt(var(dx)/var(x)) and complexity
# a ratio of those, so both are unchanged if the whole trace is multiplied by a
# constant. Absolute band power, line length and variance all scale with it.
# The markers match both view namings: `bp_delta_rel__mean` (aggregated) and
# `T7-P7__bp_delta_rel` (per-channel).
_INVARIANT_MARKERS = ("_rel", "hjorth")
_DEPENDENT_MARKERS = ("_log", "line_length", "variance")


def _is_amplitude_invariant(col: str) -> bool:
    return any(m in col for m in _INVARIANT_MARKERS)


def apply_relabel(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Re-derive labels at a different ictal-overlap threshold, if configured.

    Reuses eval.sensitivity.relabel, which is already the tested definition of
    this counterfactual: a window clears the new threshold and becomes ictal, a
    guard window that still fails it stays excluded because its exclusion comes
    from boundary proximity rather than from overlap.
    """
    frac = cfg.training.relabel_overlap_min
    if frac is None:
        return df
    if "overlap_frac" not in df.columns:
        raise ValueError(
            "relabel_overlap_min is set but the feature table has no "
            "overlap_frac column; re-extract or unset it")
    from .eval.sensitivity import relabel
    return relabel(df, float(frac))


def feature_columns(df: pd.DataFrame, subset: str = "all") -> list[str]:
    cols = [c for c in df.columns if c not in KEY_COLS]
    if subset == "all":
        return cols
    if subset == "amplitude_invariant":
        return [c for c in cols if _is_amplitude_invariant(c)]
    if subset == "amplitude_dependent":
        return [c for c in cols
                if any(m in c for m in _DEPENDENT_MARKERS)
                and not _is_amplitude_invariant(c)]
    raise ValueError(f"unknown feature_subset: {subset}")


def _fit(name: str, tr: pd.DataFrame, cols: list[str], seed: int):
    model = zoo.build(name, seed)
    model.fit(tr[cols].to_numpy(dtype=np.float32), tr["label"].to_numpy())
    return model


def _score(model, ev: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not len(ev):
        return np.array([])
    return zoo.scores(model, ev[cols].to_numpy(dtype=np.float32))


def _pick_k(model_scores: np.ndarray, va: pd.DataFrame, thr: float, ks: list[int]) -> tuple[int, float]:
    """Tune k on validation only (PRD §10.4).

    The run mask is built per record by eval.metrics.run_mask. Building it from
    a time-overlap test over the whole concatenated validation frame marked
    windows in every record sharing a time offset, because t_start_sec restarts
    at 0 in each record -- so best_k was chosen from a meaningless vector.
    """
    pv = (model_scores >= thr).astype(bool)
    best_k, best_f2 = ks[0] if ks else 1, -1.0
    for k in ks:
        mask = M.run_mask(va, pv, k)
        wm = M.window_metrics(va["label"].to_numpy(), mask.astype(float), 0.5)
        if wm["f2"] > best_f2:
            best_f2, best_k = wm["f2"], k
    return best_k, best_f2


def _pick_thr_k(
    s_va: np.ndarray, va: pd.DataFrame, ks: list[int], n_grid: int = 24
) -> tuple[float, int, float]:
    """Choose threshold and k together, on validation only.

    Sequential choice -- best threshold first, then best k given it -- can land
    on a pair that detects nothing. Measured on sub07: threshold 0.9927 admitted
    6 windows with zero false positives, but k=3 required three consecutive, so
    no run formed and 0 of 3 seizures were detected despite perfect window
    precision. The two parameters are not separable, because raising the
    threshold shortens every run.

    Candidates are score quantiles rather than every unique score: the joint
    grid is |ks| x n_grid run-mask evaluations, and run_mask groups by record.
    """
    y = va["label"].to_numpy()
    if not len(y) or len(np.unique(y)) < 2 or not len(s_va):
        return 0.5, (ks[0] if ks else 1), float("nan")

    qs = np.linspace(0.0, 1.0, n_grid)
    cands = np.unique(np.quantile(s_va, qs))
    # A threshold at the maximum score admits nothing, so it can never win.
    cands = cands[cands < s_va.max()] if len(cands) > 1 else cands

    best = (0.5, ks[0] if ks else 1, -1.0)
    for k in ks:
        for thr in cands:
            mask = M.run_mask(va, (s_va >= thr).astype(bool), k)
            f2 = M.window_metrics(y, mask.astype(float), 0.5)["f2"]
            if f2 > best[2]:
                best = (float(thr), k, float(f2))
    return best


def _val_event_score(
    va: pd.DataFrame,
    s_va: np.ndarray,
    thr: float,
    k: int,
    truth: dict[str, list[tuple[int, int]]],
    durations: dict[str, float] | None,
) -> float:
    """Event-level F2 on the VALIDATION subjects, for model selection.

    Window F2 cannot express the trade that actually decides between these
    models: seizures found against false alarms raised. Measured on this data,
    window F2 ranked hgb above rf while rf detected 71 of 185 seizures against
    hgb's 56 -- so the two criteria genuinely disagree, and the window one had
    to be overridden by hand, which imports selection bias from the test set.

    Scoring the same preference on validation removes the bias: precision here
    is detected events over alarms raised, recall is detected over true, and F2
    weights recall double exactly as at the window level (R36 is satisfied
    because no test row is touched).

    The time-in-alarm factor is not decoration, it is load-bearing (R41). Event
    F2 alone counts false alarms per *run*, so a detector that never stops
    alarming earns one giant run per record. Measured: the dummy classifier
    scored event F2 **0.5140** against hgb's 0.1596 -- three times the best real
    model -- and would have been selected as the headline. Scaling by the
    fraction of time *not* spent in alarm collapses it to ~0.00005 while costing
    a real detector about a quarter of its score. The same loophole, caught
    twice in the same project, in FA/h and then here.
    """
    if not len(va):
        return float("nan")
    ev = M.event_metrics(va, (s_va >= thr).astype(int), truth, k, durations)
    if ev.n_true == 0:
        return float("nan")
    f2 = float(zoo.fbeta(ev.n_detected, ev.false_alarms,
                         ev.n_true - ev.n_detected, 2.0))
    quiet = 1.0 - ev.time_in_alarm_frac
    if not np.isfinite(quiet):
        return f2
    return f2 * max(0.0, quiet)


def export_scores(
    df: pd.DataFrame,
    cfg: Config,
    truth: dict[str, list[tuple[int, int]]],
    subjects: list[str],
    model_name: str | None = None,
    durations: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-window scores for named held-out subjects, under their own LOSO fold.

    The training loop keeps only aggregate metrics, so nothing downstream can
    redraw a detection at a different threshold. Anything that wants to show
    *why* an operating point behaves as it does -- a viewer with a threshold
    slider, a figure of the threshold/k interaction -- needs the raw scores.

    Each subject is scored by the model fitted on its own fold, so the exported
    scores are held-out predictions, not in-sample ones. The threshold and k of
    that fold travel with them, which is what makes a slider honest: the default
    position is the operating point the pipeline actually chose.
    """
    df = P.drop_guard(apply_relabel(df, cfg))
    cols = feature_columns(df, cfg.training.feature_subset)
    name = model_name or cfg.training.headline_model or "rf"
    _, folds = P.assign(df, "loso", cfg.splits.n_val_subjects, cfg.splits.seed)
    want = set(subjects)
    out: list[pd.DataFrame] = []

    for f in folds:
        if not want & set(f.test_subjects):
            continue
        te = df[df.subject_id.isin(f.test_subjects)]
        tr = df[df.subject_id.isin(f.train_subjects)]
        va = df[df.subject_id.isin(f.val_subjects)]
        if not len(te) or not len(tr):
            continue
        tr_ds = P.downsample_negatives(tr, cfg.training.negative_downsample_ratio,
                                       cfg.splits.seed)
        model = zoo.build(name, cfg.splits.seed)
        model.fit(tr_ds[cols].to_numpy(dtype=np.float32), tr_ds["label"].to_numpy())
        s_te = _score(model, te, cols)

        thr, best_k = 0.5, 1
        if len(va) and va["label"].nunique() > 1:
            s_va = _score(model, va, cols)
            if cfg.training.joint_threshold_k:
                thr, best_k, _ = _pick_thr_k(
                    s_va, va, list(cfg.training.consecutive_k), cfg.training.joint_grid)
            else:
                thr, _ = zoo.pick_threshold(
                    va["label"].to_numpy(), s_va, cfg.training.threshold_metric)
                best_k, _ = _pick_k(s_va, va, thr, list(cfg.training.consecutive_k))

        keep = ["subject_id", "record_id", "window_idx", "t_start_sec",
                "t_end_sec", "label"]
        g = te[keep].copy()
        g["score"] = s_te
        g["chosen_threshold"] = thr
        g["chosen_k"] = best_k
        g["model"] = name
        out.append(g)

    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).sort_values(["record_id", "window_idx"])


def fit_final(
    df: pd.DataFrame, cfg: Config, model_name: str | None = None
) -> tuple[object, list[str], dict]:
    """Fit the designated model on **every** subject, for use on new files.

    This model has no held-out estimate of its own and never can: there is no
    subject left to hold out. Its expected behaviour on an unseen patient is
    the LOSO result, which is exactly what LOSO was measuring, so the caller
    must carry those figures alongside the artifact rather than implying that
    training on more data made it better.

    The operating point cannot be tuned either -- there is no validation
    subject outside the training set. The median threshold and modal k across
    the LOSO folds are the honest default, and their spread (0.000 to 0.994
    across folds, measured) is the reason a single number here is a compromise
    and not a solution.
    """
    df = P.drop_guard(apply_relabel(df, cfg))
    cols = feature_columns(df, cfg.training.feature_subset)
    name = model_name or cfg.training.headline_model or "rf"
    ds = P.downsample_negatives(df, cfg.training.negative_downsample_ratio,
                                cfg.splits.seed)
    model = zoo.build(name, cfg.splits.seed)
    model.fit(ds[cols].to_numpy(dtype=np.float32), ds["label"].to_numpy())
    meta = {
        "model": name,
        "n_subjects": int(df.subject_id.nunique()),
        "n_rows_available": int(len(df)),
        "n_rows_fitted": int(len(ds)),
        "n_features": len(cols),
        "feature_subset": cfg.training.feature_subset,
        "extraction_hash": cfg.extraction_hash,
        "full_hash": cfg.full_hash,
    }
    return model, cols, meta


def run_loso(
    df: pd.DataFrame,
    cfg: Config,
    truth: dict[str, list[tuple[int, int]]],
    durations: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (run_metrics long-form, per-fold detail)."""
    df = P.drop_guard(apply_relabel(df, cfg))
    cols = feature_columns(df, cfg.training.feature_subset)
    P.assert_deduplicated(df, cols)  # R6, before any split exists

    _, folds = P.assign(df, "loso", cfg.splits.n_val_subjects, cfg.splits.seed)
    rows: list[dict] = []
    detail: list[dict] = []

    for f in folds:
        te = df[df.subject_id.isin(f.test_subjects)]
        va = df[df.subject_id.isin(f.val_subjects)]
        tr = df[df.subject_id.isin(f.train_subjects)]
        # Structural guard only. An earlier version also skipped folds where
        # te["label"].nunique() < 2, which filters folds by inspecting the
        # answer; single-class test folds now stay in the run and simply get
        # NaN PR-AUC from window_metrics.
        if not len(te) or not len(tr):
            detail.append({"fold_idx": f.fold_idx, "test_subject": f.test_subjects[0],
                           "skipped": "no_train_or_test_rows"})
            continue

        tr_ds = P.downsample_negatives(tr, cfg.training.negative_downsample_ratio,
                                       cfg.splits.seed)

        for name in cfg.training.models:
            # Inner hyperparameter search on TRAINING rows only, grouped by
            # subject (R9/PRD §9.3). Never sees val or test.
            params, search_score = {}, float("nan")
            if cfg.training.search:
                params, search_score = zoo.search(
                    name,
                    tr_ds[cols].to_numpy(dtype=np.float32),
                    tr_ds["label"].to_numpy(),
                    tr_ds["subject_id"].to_numpy(),
                    n_splits=cfg.training.inner_splits,
                    seed=cfg.splits.seed,
                )

            # Fit once, score both splits. The model is identical for val and
            # test scoring, so refitting was pure waste (and a refit on
            # train+val would have been a leak).
            model = zoo.build_tuned(name, params, cfg.splits.seed)
            model.fit(tr_ds[cols].to_numpy(dtype=np.float32), tr_ds["label"].to_numpy())
            s_va = _score(model, va, cols)
            s_te = _score(model, te, cols)

            val_score = float("nan")
            if len(va) and va["label"].nunique() > 1:
                if cfg.training.joint_threshold_k:
                    thr, best_k, val_score = _pick_thr_k(
                        s_va, va, list(cfg.training.consecutive_k),
                        cfg.training.joint_grid)
                else:
                    thr, val_score = zoo.pick_threshold(
                        va["label"].to_numpy(), s_va, cfg.training.threshold_metric)
                    best_k, _ = _pick_k(s_va, va, thr, list(cfg.training.consecutive_k))
            else:
                thr, best_k = 0.5, 1

            val_event = _val_event_score(va, s_va, thr, best_k, truth, durations)
            if cfg.training.select_on == "event_f2":
                val_score = val_event

            wm = M.window_metrics(te["label"].to_numpy(), s_te, thr)
            # event_metrics applies the k-of-consecutive rule itself, so it
            # takes raw thresholded predictions, not an already-masked vector.
            ev = M.event_metrics(te, (s_te >= thr).astype(int), truth, best_k, durations)
            es = ev.summary()

            base = {
                "fold_idx": f.fold_idx, "model": name,
                "feature_view": "per_channel" if len(cols) > 100 else "aggregated",
                "test_subject": f.test_subjects[0], "protocol": "loso",
                "threshold": thr, "consecutive_k": best_k,
                # Model selection must use this, never a test metric (R10).
                "val_score": val_score,
                "val_event_f2": val_event,
                "search_params": json.dumps(params) if params else "",
                "search_score": search_score,
                "n_train": len(tr_ds), "n_train_pre_downsample": len(tr),
                "n_val": len(va), "n_test": len(te),
            }
            detail.append({**base, **wm, **es})
            for mk, mv in {**wm, **es}.items():
                if isinstance(mv, (int, float)):
                    rows.append({**base, "metric_name": mk, "metric_value": float(mv)})

    return pd.DataFrame(rows), detail


def run_random_window(df: pd.DataFrame, cfg: Config) -> list[dict]:
    """PRD §9.2: the contrast run that exists to be wrong.

    It must be wrong in exactly ONE way -- ignoring subject identity -- so the
    delta against LOSO is attributable to the split protocol. An earlier
    version fitted the operating threshold on the test partition's own labels,
    which added oracle-threshold leakage on top and made the comparison
    uninterpretable. The threshold is now chosen on a validation slice carved
    out of the training partition, exactly as run_loso does.
    """
    df = P.drop_guard(apply_relabel(df, cfg)).reset_index(drop=True)
    cols = feature_columns(df, cfg.training.feature_subset)
    P.assert_deduplicated(df, cols)

    rng = np.random.default_rng(cfg.splits.seed)
    idx = rng.permutation(len(df))
    n_te = int(0.2 * len(df))
    n_va = int(0.1 * len(df))
    te = df.iloc[idx[:n_te]]
    va = df.iloc[idx[n_te : n_te + n_va]]
    tr = df.iloc[idx[n_te + n_va :]]
    tr_ds = P.downsample_negatives(tr, cfg.training.negative_downsample_ratio, cfg.splits.seed)

    out = []
    for name in cfg.training.models:
        model = _fit(name, tr_ds, cols, cfg.splits.seed)
        s_va, s_te = _score(model, va, cols), _score(model, te, cols)
        if len(va) and va["label"].nunique() > 1:
            thr, val_score = zoo.pick_threshold(
                va["label"].to_numpy(), s_va, cfg.training.threshold_metric)
        else:
            thr, val_score = 0.5, float("nan")
        wm = M.window_metrics(te["label"].to_numpy(), s_te, thr)
        out.append({
            "protocol": "random_window", "model": name,
            "feature_view": "per_channel" if len(cols) > 100 else "aggregated",
            "val_score": val_score,
            "n_train": len(tr_ds), "n_val": len(va), "n_test": len(te), **wm,
        })
    return out

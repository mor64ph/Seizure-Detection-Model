"""The model ladder (PRD §10.1) and threshold selection (PRD §10.3).

Step 0 is not optional. DummyClassifier(strategy="most_frequent") prints an
accuracy that looks excellent and is worthless, and that number belongs in the
report as the proof that accuracy is the wrong metric here.

R7: every model that needs scaling is wrapped in a Pipeline, so the scaler is
fit inside the training fold and cannot escape it by accident.

No SMOTE. Synthesising EEG feature vectors by interpolating between real
windows produces rows that correspond to no physiological state, and it is
indefensible in a clinical framing (PRD §10.2).
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS = ("dummy", "logreg", "rf", "hgb")


def build(name: str, seed: int = 42):
    """Construct an unfitted estimator. Scaling and imputation live inside."""
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")

    if name == "logreg":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ])

    if name == "rf":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                n_jobs=-1, random_state=seed)),
        ])

    if name == "hgb":
        # Handles NaN natively, so no imputer -- and fill_nan channels stay
        # informative rather than being median-filled into fake signal.
        return HistGradientBoostingClassifier(random_state=seed)

    raise ValueError(f"unknown model: {name}")


# -- hyperparameter grids (PRD §9.3) ------------------------------------
# Deliberately small. The inner loop runs |grid| x inner_splits fits inside
# each of 23 outer folds for each model and feature view, so a grid that looks
# harmless multiplies into thousands of fits. These cover the axis that
# actually matters per model -- regularisation strength for the linear model,
# tree depth/leaf size for the ensembles -- and nothing decorative.
GRIDS: dict[str, dict[str, list]] = {
    "dummy": {},
    "logreg": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "rf": {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 5]},
    "hgb": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
}


def search(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 3,
    scoring: str = "average_precision",
    seed: int = 42,
) -> tuple[dict, float]:
    """Grid search inside one training fold, grouped by subject.

    R9/PRD §9.3: the inner CV is ``GroupKFold(groups=subject_id)``. A bare
    ``KFold`` here would put the same subject on both sides of the inner split,
    so the chosen hyperparameters would be tuned partly on subject identity --
    the same leak as a bad outer split, one level down and much easier to miss.

    Returns ({} , nan) when the grid is empty or the fold cannot support the
    requested number of group splits, in which case the caller uses defaults.
    """
    from sklearn.model_selection import GridSearchCV, GroupKFold

    grid = GRIDS.get(name, {})
    n_groups = len(np.unique(groups))
    if not grid or n_groups < 2 or len(np.unique(y)) < 2:
        return {}, float("nan")

    cv = GroupKFold(n_splits=min(n_splits, n_groups))
    gs = GridSearchCV(
        build(name, seed), grid, scoring=scoring, cv=cv, n_jobs=1, refit=False,
        error_score=float("nan"),
    )
    gs.fit(X, y, groups=groups)
    return dict(gs.best_params_), float(gs.best_score_)


def build_tuned(name: str, params: dict, seed: int = 42):
    """Construct an estimator with searched parameters applied."""
    m = build(name, seed)
    if params:
        m.set_params(**params)
    return m


def scores(model, X: np.ndarray) -> np.ndarray:
    """Positive-class score, whatever the estimator exposes."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else np.zeros(len(X))
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


# -- thresholds ----------------------------------------------------------
def fbeta(tp: int, fp: int, fn: int, beta: float) -> float:
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    if prec == 0 and rec == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def pick_threshold(y: np.ndarray, s: np.ndarray, metric: str = "f2") -> tuple[float, float]:
    """Maximise F-beta on validation scores. Returns (threshold, achieved).

    Never 0.5 (PRD §10.3). F2 weights recall over precision, which matches the
    cost asymmetry: a missed seizure in a presurgical admission is worse than a
    review the technologist dismisses.

    Candidates are the unique scores themselves, so the search is exact rather
    than a grid approximation.
    """
    beta = 2.0 if metric == "f2" else 1.0
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if len(y) == 0 or y.sum() == 0 or len(np.unique(s)) < 2:
        return 0.5, 0.0

    order = np.argsort(-s)
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    total_pos = int(y.sum())
    fn = total_pos - tp

    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(total_pos, 1)
        f = (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-12)
    f = np.nan_to_num(f)

    # Candidates are restricted to tie-block boundaries -- indices where the
    # next score actually differs, plus the last. Taking argmax over every row
    # can land inside a block of equal scores, where the (tp, fp) prefix is not
    # realisable by any threshold: `s >= thr` admits the whole block at once.
    # That returned an F-beta no threshold could achieve, and a threshold that
    # scored worse than the reported value.
    cuts = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    k = int(cuts[int(np.argmax(f[cuts]))])
    # Threshold sits between this score and the next distinct one, so exactly
    # the first k+1 rows satisfy `s >= thr`.
    thr = float(ss[k]) if k + 1 >= len(ss) else float((ss[k] + ss[k + 1]) / 2)
    return thr, float(f[k])

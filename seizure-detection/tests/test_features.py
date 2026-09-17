"""PRD §8.4: every feature function against a synthetic signal with a known
answer. These are the tests that catch a band-edge off-by-one or a mis-scaled
PSD, which no amount of staring at real EEG will reveal."""

from __future__ import annotations

import numpy as np
import pytest

from seizure.features import core, views

FS = 256
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}
NAMES = core.feature_names(BANDS)


def sine(freq: float, n: int = 2560, fs: int = FS, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def feat(x: np.ndarray, name: str) -> float:
    return float(core.extract(x, FS, BANDS)[..., NAMES.index(name)])


def test_names_are_14():
    assert len(NAMES) == 2 * len(BANDS) + 4 == 14


def test_pure_10hz_is_alpha():
    assert feat(sine(10.0), "bp_alpha_rel") > 0.9


def test_pure_2hz_is_delta():
    assert feat(sine(2.0), "bp_delta_rel") > 0.9


def test_pure_40hz_is_gamma():
    assert feat(sine(40.0), "bp_gamma_rel") > 0.9


def test_relative_powers_sum_to_about_one():
    x = sine(10.0) + sine(2.0) + sine(40.0)
    total = sum(feat(x, f"bp_{b}_rel") for b in BANDS)
    assert 0.9 < total < 1.1


def test_line_length_of_ramp():
    """Ramp of slope m, length n -> m*(n-1)."""
    n, m = 1000, 0.25
    x = m * np.arange(n, dtype=float)
    assert core.line_length(x) == pytest.approx(m * (n - 1), rel=1e-9)


def test_variance_of_unit_noise():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000)
    assert core.variance(x) == pytest.approx(1.0, abs=0.02)


def test_hjorth_complexity_of_sine_is_one():
    assert core.hjorth_complexity(sine(10.0, n=10_240)) == pytest.approx(1.0, abs=0.05)


def test_hjorth_mobility_scales_with_frequency():
    """Mobility ~ 2*pi*f/fs for a sine, so doubling f doubles mobility."""
    m10 = core.hjorth_mobility(sine(10.0, n=10_240))
    m20 = core.hjorth_mobility(sine(20.0, n=10_240))
    assert m20 / m10 == pytest.approx(2.0, rel=0.05)


def test_variance_is_amplitude_squared():
    assert core.variance(sine(10.0, amp=3.0)) == pytest.approx(4.5, rel=0.02)


def test_absolute_power_is_logged_not_raw():
    """A 100x amplitude increase is a 10,000x power increase. The logged
    feature must compress that by at least two orders of magnitude, otherwise
    heavy-tailed absolute power dominates any linear model (PRD §8.1)."""
    small = feat(sine(10.0, amp=1.0), "bp_alpha_log")
    big = feat(sine(10.0, amp=100.0), "bp_alpha_log")
    raw_ratio = 100.0**2
    assert big / small < raw_ratio / 100  # measured ~21x against 10,000x raw


def test_absolute_power_log_is_exact():
    """A unit-amplitude sine carries power 1/2, so log1p gives log(1.5)."""
    assert feat(sine(10.0), "bp_alpha_log") == pytest.approx(np.log1p(0.5), rel=0.02)


def test_relative_power_is_amplitude_invariant():
    """PRD §8.2: this is the property that should transfer across subjects."""
    a = feat(sine(10.0, amp=1.0), "bp_alpha_rel")
    b = feat(sine(10.0, amp=250.0), "bp_alpha_rel")
    assert a == pytest.approx(b, abs=1e-3)


def test_nan_and_inf_never_escape():
    x = np.zeros(2560)  # zero variance -> division by zero internally
    out = core.extract(x, FS, BANDS)
    assert np.isfinite(out).all()


# -- views ---------------------------------------------------------------
def test_per_channel_column_names_use_labels_not_indices():
    names = views.per_channel_names(["FP1-F7", "FZ-CZ"], NAMES)
    assert names[0] == "FP1-F7__bp_delta_log"
    assert len(names) == 2 * 14
    assert not any(n.split("__")[0].isdigit() for n in names)


def test_aggregated_is_channel_order_invariant():
    """The property that makes the aggregated view montage-robust."""
    rng = np.random.default_rng(1)
    f = rng.standard_normal((5, 18, 14)).astype(np.float32)
    shuffled = f[:, rng.permutation(18), :]
    np.testing.assert_allclose(
        views.aggregated(f), views.aggregated(shuffled), rtol=1e-5, atol=1e-6
    )


def test_per_channel_is_not_channel_order_invariant():
    """The contrast: this view is montage-fragile, which is why R16 exists."""
    rng = np.random.default_rng(2)
    f = rng.standard_normal((5, 18, 14)).astype(np.float32)
    shuffled = f[:, rng.permutation(18), :]
    assert not np.allclose(views.per_channel(f), views.per_channel(shuffled))


def test_aggregated_width():
    f = np.zeros((3, 18, 14), dtype=np.float32)
    assert views.aggregated(f).shape == (3, 14 * 4)
    assert len(views.aggregated_names(NAMES)) == 14 * 4

"""Generate a synthetic EDF the app will accept, for demoing the upload path.

The live app asks for a recording and a visitor has none: CHB-MIT files are
~51 MB each and are not redistributed here. This writes a small synthetic file
with the exact montage and sample rate the refusal gate demands, so the upload
path can be exercised without clinical data.

It is deliberately *not* named like a CHB-MIT record. The app resolves ground
truth by matching the filename against known record ids, so calling this
`chb01_03.edf` would attach real annotations to fabricated signal and report
accuracy that means nothing. Under a neutral name it lands in the
`unverifiable` branch, which is the honest one for synthetic input.

MEASURED: the model does NOT detect the injected event -- mean score 0.151
inside it against 0.259 on background, i.e. below background. Not tuned
further on purpose; see samples/README.md. For a demo that shows detection
actually working, use the real excerpt from make_sample_excerpt.py.

No EDF writer is installed (no pyedflib, no edfio, and mne's EDF export needs
edfio), so the 256-byte general header plus 256 bytes per signal are written
directly. Verified by reading the result back through the project's own header
parser and full extraction path.
"""

from __future__ import annotations

import datetime as dt
import struct
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FS = 256
DURATION_SEC = 300
RECORD_SEC = 1
# 0.2 uV per least-significant bit, which comfortably covers scalp EEG.
PHYS_MIN, PHYS_MAX = -3277.0, 3276.0
DIG_MIN, DIG_MAX = -32768, 32767
SCALE = (PHYS_MAX - PHYS_MIN) / (DIG_MAX - DIG_MIN)

# One focal onset that spreads, which is what the model is trained to see.
SEIZURE = (182, 218)
ONSET_CHANNELS = ("F7-T7", "T7-P7", "FP1-F7")
SPREAD_CHANNELS = ("FP1-F3", "F3-C3", "C3-P3", "P7-O1")


def _fld(value: str, width: int) -> bytes:
    """ASCII, left-justified, space-padded. EDF is unforgiving about this."""
    b = str(value).encode("ascii", "replace")[:width]
    return b + b" " * (width - len(b))


def background(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f-ish noise band-limited to 0.5-40 Hz at roughly 25 uV RMS.

    Real EEG background is not white: power falls with frequency. Shaping the
    spectrum matters here because ten of the model's fourteen per-channel
    features are band powers, and white noise would put implausible energy in
    beta and gamma.
    """
    f = np.fft.rfftfreq(n, 1 / FS)
    mag = np.zeros_like(f)
    ok = f > 0
    mag[ok] = 1.0 / np.sqrt(f[ok])
    mag[(f < 0.5) | (f > 40)] = 0.0
    phase = rng.uniform(0, 2 * np.pi, len(f))
    x = np.fft.irfft(mag * np.exp(1j * phase), n)
    return x / (x.std() or 1.0) * 25.0


def alpha_bursts(n: int, rng: np.random.Generator, weight: float) -> np.ndarray:
    """Intermittent 10 Hz posterior rhythm, stronger occipitally."""
    t = np.arange(n) / FS
    env = np.zeros(n)
    for _ in range(rng.integers(6, 12)):
        start = rng.uniform(0, DURATION_SEC - 8)
        length = rng.uniform(2.0, 6.0)
        m = (t >= start) & (t < start + length)
        env[m] += np.hanning(m.sum()) if m.sum() else 0
    return env * np.sin(2 * np.pi * 10.2 * t) * 18.0 * weight


def seizure_component(n: int, involvement: float, delay: float) -> np.ndarray:
    """An electrographic seizure with the morphology real ones have.

    A first version used a clean 3.4 Hz sinusoid and the model scored it *below*
    background (0.188 against 0.259) -- a pure tone concentrates power in one
    narrow bin, which looks nothing like ictal EEG. The features it actually
    keys on come from four properties, all modelled here:

      * frequency evolution -- fast recruiting rhythm near 11 Hz slowing toward
        3 Hz as the seizure organises, which is the classic progression
      * spike-and-wave morphology -- a sharp transient followed by a slow
        component, so power spreads across harmonics instead of one peak
      * amplitude several times background, rising then falling
      * postictal suppression -- a depressed-amplitude tail afterwards

    Deliberately not tuned against the classifier's output. The signal is made
    physiologically plausible and whatever the model then reports is the honest
    result, including "missed it".
    """
    t = np.arange(n) / FS
    a, b = SEIZURE[0] + delay, SEIZURE[1]
    out = np.zeros(n)
    m = (t >= a) & (t <= b)
    if not m.any():
        return out

    local = (t[m] - a) / max(1e-9, (b - a))
    # 11 Hz -> 3 Hz. Integrating the instantaneous frequency keeps the phase
    # continuous; multiplying t by a varying f would introduce a phase jump.
    f_inst = 11.0 - 8.0 * local
    phase = 2 * np.pi * np.cumsum(f_inst) / FS
    spike = np.sign(np.sin(phase)) * np.abs(np.sin(phase)) ** 0.35
    slow = 0.6 * np.sin(phase - 0.9)
    ramp = np.clip(local / 0.2, 0, 1) * np.clip((1 - local) / 0.3, 0, 1)
    out[m] = ramp * (spike + slow) * 150.0 * involvement

    # Postictal suppression: 20 s of attenuation, strongest right after offset.
    post = (t > b) & (t <= b + 20)
    if post.any():
        out[post] = -0.0  # the attenuation is applied by the caller
    return out


def postictal_gain(n: int, delay: float, involvement: float) -> np.ndarray:
    """Multiplicative envelope that suppresses background after the seizure."""
    t = np.arange(n) / FS
    g = np.ones(n)
    b = SEIZURE[1]
    m = (t > b) & (t <= b + 20)
    if m.any():
        decay = (t[m] - b) / 20.0
        g[m] = 1.0 - 0.55 * involvement * (1.0 - decay)
    return g


def build(channels: list[str], seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = FS * DURATION_SEC
    out = np.zeros((len(channels), n), dtype=np.float64)
    for i, ch in enumerate(channels):
        posterior = 1.0 if ch.endswith(("O1", "O2", "PZ")) else 0.45
        if ch in ONSET_CHANNELS:
            involve, delay = 1.0, 0.0
        elif ch in SPREAD_CHANNELS:
            involve, delay = 0.55, 6.0
        else:
            involve, delay = 0.15, 11.0
        # Background is suppressed postictally, so the gain multiplies the
        # background only -- the discharge itself is already over by then.
        sig = ((background(n, rng) + alpha_bursts(n, rng, posterior))
               * postictal_gain(n, delay, involve))
        sig += seizure_component(n, involve, delay)
        # A little 60 Hz, because real recordings have it and the pipeline's
        # notch is supposed to remove it.
        sig += 4.0 * np.sin(2 * np.pi * 60 * np.arange(n) / FS + rng.uniform(0, 6))
        out[i] = sig
    return out


def write_edf(path: Path, data: np.ndarray, channels: list[str]) -> None:
    ns = len(channels)
    spr = FS * RECORD_SEC
    n_records = data.shape[1] // spr
    now = dt.datetime(2026, 1, 1, 9, 0, 0)

    head = b"".join([
        _fld("0", 8),
        _fld("X X X Synthetic_demo", 80),
        _fld("Startdate 01-JAN-2026 X X synthetic_18ch_256Hz", 80),
        _fld(now.strftime("%d.%m.%y"), 8),
        _fld(now.strftime("%H.%M.%S"), 8),
        _fld(256 + 256 * ns, 8),
        _fld("EDF", 44),
        _fld(n_records, 8),
        _fld(RECORD_SEC, 8),
        _fld(ns, 4),
    ])
    head += b"".join(_fld(c, 16) for c in channels)
    head += b"".join(_fld("AgAgCl electrode", 80) for _ in channels)
    head += b"".join(_fld("uV", 8) for _ in channels)
    head += b"".join(_fld(f"{PHYS_MIN:.0f}", 8) for _ in channels)
    head += b"".join(_fld(f"{PHYS_MAX:.0f}", 8) for _ in channels)
    head += b"".join(_fld(DIG_MIN, 8) for _ in channels)
    head += b"".join(_fld(DIG_MAX, 8) for _ in channels)
    head += b"".join(_fld("HP:0.5Hz LP:80Hz", 80) for _ in channels)
    head += b"".join(_fld(spr, 8) for _ in channels)
    head += b"".join(_fld("", 32) for _ in channels)
    assert len(head) == 256 + 256 * ns, len(head)

    digital = np.clip(np.round(data / SCALE), DIG_MIN, DIG_MAX).astype("<i2")
    with open(path, "wb") as f:
        f.write(head)
        for r in range(n_records):
            sl = slice(r * spr, (r + 1) * spr)
            f.write(digital[:, sl].tobytes(order="C"))


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    channels = list(cfg["signal"]["canonical_channels"])
    out = ROOT / "samples" / "synthetic_demo.edf"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_edf(out, build(channels), channels)
    mb = out.stat().st_size / 1024 / 1024
    print(f"wrote {out} ({mb:.2f} MB, {len(channels)} ch, {FS} Hz, "
          f"{DURATION_SEC}s, event at {SEIZURE[0]}-{SEIZURE[1]}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

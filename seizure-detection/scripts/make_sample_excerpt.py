"""Cut a short, real excerpt from a CHB-MIT record for the bundled demo.

The deployed app asks for a recording and a visitor has none. Full CHB-MIT
records are ~51 MB, too large to commit, and the synthetic generator in
make_sample_edf.py produces a file the model does *not* fire on -- useful for
testing the upload path, useless as a demonstration that detection works.

A five-minute excerpt around a real annotated seizure is ~2.6 MB, is genuinely
detected, and is redistributable: CHB-MIT is published under the Open Data
Commons Attribution License, and the attribution is in LICENSE and the app
footer.

The excerpt's clock restarts at zero, so the seizure interval shifts by the cut
offset. The app cannot recover that from the filename, so the offset seizure
span is registered in BUNDLED_SAMPLES in the app -- do not rename the output
without updating that entry, or the demo will silently lose its ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_sample_edf import FS, write_edf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "chbmit" / "chb01" / "chb01_03.edf"
SEIZURE = (2996, 3036)          # as annotated in the full record
CUT = (2880, 3180)              # 300 s, seizure sits 116-156 s into the excerpt


def main() -> int:
    if not SOURCE.exists():
        print(f"source not on disk: {SOURCE}\n"
              f"run `make ingest` or keep chb01 locally (R13)", file=sys.stderr)
        return 1
    from seizure.signal import io as SIO

    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    channels = list(cfg["signal"]["canonical_channels"])
    rec = SIO.read(SOURCE, channels, "skip_file")
    if abs(rec.sample_rate - FS) > 1e-6:
        print(f"unexpected source rate {rec.sample_rate}", file=sys.stderr)
        return 1

    a, b = int(CUT[0] * FS), int(CUT[1] * FS)
    # mne returns SI units, i.e. VOLTS, while the writer declares a physical
    # dimension of uV. Without this conversion every sample divides down to
    # zero and the excerpt is a flat line that the model scores identically
    # everywhere -- which is exactly what happened first time round.
    data = np.ascontiguousarray(rec.data[:, a:b]) * 1e6
    out = ROOT / "samples" / "sample_chb01_03_2880-3180s.edf"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_edf(out, data, rec.channels)

    # Round-trip assertion. A writer that silently destroys the signal is worse
    # than one that errors, so refuse to ship a file that does not read back as
    # the source slice.
    back = SIO.read(out, channels, "skip_file")
    src_v = rec.data[:, a:b]
    n = min(src_v.shape[1], back.data.shape[1])
    corr = float(np.corrcoef(src_v[0, :n], back.data[0, :n])[0, 1])
    ratio = float(back.data.std() / (src_v.std() or 1))
    if not (corr > 0.999 and 0.99 < ratio < 1.01):
        print(f"ROUND-TRIP FAILED: corr={corr:.4f} std_ratio={ratio:.4f}",
              file=sys.stderr)
        out.unlink(missing_ok=True)
        return 1
    print(f"  round-trip : corr={corr:.5f} std_ratio={ratio:.4f} OK")

    off = (SEIZURE[0] - CUT[0], SEIZURE[1] - CUT[0])
    print(f"wrote {out} ({out.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  source     : chb01_03.edf {CUT[0]}-{CUT[1]} s")
    print(f"  seizure at : {off[0]}-{off[1]} s within the excerpt")
    print(f"  register as: {out.name} -> {[off]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for the streaming ingest contract.

R11 is the rule with the worst failure mode in the project: deleting raw bytes
is a one-way door, and the fields that must be captured first are unrecoverable
without re-downloading. It is enforced by a function, so it gets tested like
one.
"""

from __future__ import annotations

import json

import pytest

from seizure.ingest import edf_header as eh
from seizure.ingest import fetch, stream


def complete_state(case="chb05", rid="chb05_13") -> stream.RecordState:
    """A state with every capture-before-delete field populated."""
    return stream.RecordState(
        record_id=rid, case_id=case, rel=f"{case}/{rid}.edf",
        state="extracted", s3_size=1_000,
        checksum_verified=True, n_samples=921_600, duration_sec=3600.0,
        channel_labels=["FP1-F7", "FZ-CZ"],
        channel_stats={"FP1-F7": {"std": 1.0}, "FZ-CZ": {"std": 1.0}},
        features_path="artifacts/features/abc", n_windows=360,
    )


# ------------------------------------------------------------ R11 the gate
def test_delete_raw_deletes_when_capture_is_complete(tmp_path):
    p = tmp_path / "chb05_13.edf"
    p.write_bytes(b"x")
    assert stream.delete_raw(p, complete_state()) is True
    assert not p.exists()


@pytest.mark.parametrize("field", list(stream.REQUIRED_BEFORE_DELETE))
def test_delete_raw_refuses_when_any_field_is_missing(tmp_path, field):
    """Every one of the seven fields must independently block deletion."""
    p = tmp_path / "chb05_13.edf"
    p.write_bytes(b"x")
    st = complete_state()
    setattr(st, field, None)
    with pytest.raises(stream.CaptureIncomplete) as e:
        stream.delete_raw(p, st)
    assert field in str(e.value)
    assert p.exists(), "raw file was deleted despite incomplete capture"


def test_delete_raw_refuses_on_empty_collections(tmp_path):
    """An empty label list is as unrecoverable as a null one."""
    p = tmp_path / "chb05_13.edf"
    p.write_bytes(b"x")
    st = complete_state()
    st.channel_labels = []
    with pytest.raises(stream.CaptureIncomplete):
        stream.delete_raw(p, st)
    assert p.exists()


def test_keep_cases_are_never_deleted(tmp_path):
    """R13: chb01, chb12 and chb17 are permanent fixtures."""
    for case in stream.KEEP_CASES:
        p = tmp_path / f"{case}_01.edf"
        p.write_bytes(b"x")
        st = complete_state(case=case, rid=f"{case}_01")
        assert stream.delete_raw(p, st) is False
        assert p.exists(), f"{case} is an R13 fixture and must survive"


def test_keep_cases_covers_the_landmine_cases():
    assert stream.KEEP_CASES == frozenset({"chb01", "chb12", "chb17"})


def test_force_keep_overrides_deletion(tmp_path):
    p = tmp_path / "chb05_13.edf"
    p.write_bytes(b"x")
    assert stream.delete_raw(p, complete_state(), force_keep=True) is False
    assert p.exists()


def test_required_fields_are_the_seven_named_in_the_rule():
    assert set(stream.REQUIRED_BEFORE_DELETE) == {
        "checksum_verified", "n_samples", "duration_sec", "channel_labels",
        "channel_stats", "features_path", "n_windows",
    }


# --------------------------------------------------------- R14 the ledger
def test_ledger_roundtrips(tmp_path):
    led = stream.Ledger(tmp_path / "ledger.json")
    led.put(complete_state())
    again = stream.Ledger(tmp_path / "ledger.json")
    assert again.get("chb05_13").duration_sec == 3600.0


def test_ledger_treats_completed_records_as_done(tmp_path):
    led = stream.Ledger(tmp_path / "ledger.json")
    for state, expected in [("extracted", True), ("raw_deleted", True),
                            ("skipped", True), ("failed", False),
                            ("downloaded", False), ("pending", False)]:
        st = complete_state(rid=f"r_{state}")
        st.state = state
        led.put(st)
        assert led.done(f"r_{state}") is expected, state


def test_ledger_reports_unknown_record_as_not_done(tmp_path):
    assert stream.Ledger(tmp_path / "ledger.json").done("nope") is False


def test_ledger_is_valid_json_on_disk(tmp_path):
    led = stream.Ledger(tmp_path / "ledger.json")
    led.put(complete_state())
    json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))


# ------------------------------------------------- checksums and URL keys
def test_sha256sums_parsing():
    txt = ("a" * 64 + "  chb01/chb01_01.edf\n"
           + "b" * 64 + " *chb02/chb02_16+.edf\n"
           "not-a-line\n")
    got = fetch.parse_sha256sums(txt)
    assert got["chb01/chb01_01.edf"] == "a" * 64
    assert got["chb02/chb02_16+.edf"] == "b" * 64


def test_plus_in_filename_is_percent_encoded():
    """An unencoded '+' in a URL path is read as a space by S3 and 404s."""
    url = fetch.url_for("chb02/chb02_16+.edf")
    assert "chb02_16%2B.edf" in url
    assert "+" not in url.rsplit("/", 1)[-1]


def test_url_keeps_path_separators():
    assert fetch.url_for("chb01/chb01_01.edf").endswith("chbmit/1.0.0/chb01/chb01_01.edf")


# ------------------------------------------------------- EDF header parse
def _synth_header(n_signals: int, n_records: int = 10, spr: int = 256) -> bytes:
    def f(s, w):
        return str(s).ljust(w)[:w].encode("ascii")
    head = (f(0, 8) + f("patient", 80) + f("recording", 80) + f("01.01.11", 8)
            + f("00.00.00", 8) + f(256 + 256 * n_signals, 8) + f("", 44)
            + f(n_records, 8) + f(1, 8) + f(n_signals, 4))
    labels = b"".join(f(lab, 16) for lab in ["FP1-F7", "FZ-CZ", "ECG", "-"][:n_signals])
    return (head + labels + b"".join(f("", 80) for _ in range(n_signals))
            + b"".join(f("uV", 8) for _ in range(n_signals))
            + b"".join(f(-100, 8) for _ in range(n_signals))
            + b"".join(f(100, 8) for _ in range(n_signals))
            + b"".join(f(-2048, 8) for _ in range(n_signals))
            + b"".join(f(2047, 8) for _ in range(n_signals))
            + b"".join(f("", 80) for _ in range(n_signals))
            + b"".join(f(spr, 8) for _ in range(n_signals))
            + b"".join(f("", 32) for _ in range(n_signals)))


def test_header_size_is_learnable_from_256_bytes():
    buf = _synth_header(3)
    assert eh.header_size_from_probe(buf[:256]) == 256 + 256 * 3 == len(buf)


def test_header_parses_duration_and_labels():
    h = eh.parse(_synth_header(3, n_records=3600, spr=256))
    assert h.n_signals == 3
    assert h.duration_sec == 3600.0
    assert h.sample_rates == [256.0, 256.0, 256.0]
    assert h.labels[:2] == ["FP1-F7", "FZ-CZ"]


def test_header_size_matches_chbmit_23_signal_files():
    """6,144 bytes, verified against chb01_01.edf."""
    assert eh.header_size_from_probe(_synth_header(23)[:256]) == 6144


def test_truncated_header_raises():
    with pytest.raises(ValueError):
        eh.parse(_synth_header(3)[:500])


def test_expected_file_bytes_arithmetic():
    h = eh.parse(_synth_header(2, n_records=10, spr=256))
    assert eh.expected_data_bytes(h) == 10 * (256 + 256) * 2
    assert eh.expected_file_bytes(h) == h.header_bytes + eh.expected_data_bytes(h)


# -------------------------------------------------------- channel_stats
def test_channel_stats_are_per_channel_and_keyed_by_label():
    import numpy as np
    data = np.array([[0.0, 1.0, 2.0], [10.0, 10.0, 10.0]])
    st = stream.channel_stats(data, ["FP1-F7", "FZ-CZ"])
    assert set(st) == {"FP1-F7", "FZ-CZ"}
    assert st["FZ-CZ"]["std"] == pytest.approx(0.0)
    assert st["FP1-F7"]["p50"] == pytest.approx(1.0)

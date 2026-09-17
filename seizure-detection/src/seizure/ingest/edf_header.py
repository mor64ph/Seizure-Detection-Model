"""Pure EDF header parser, readable from a byte prefix.

Why this exists alongside mne: R30 freezes the canonical channel list from a
header-only range-request pre-pass over all 686 files, for ~4.2 MB instead of
45.76 GB. mne cannot read a partial object over HTTP, so the header is parsed
directly. Full signal reads still go through mne (PRD §7.1).

EDF layout (European Data Format, Kemp et al.):

    offset  size  field
    0       8     version, "0       "
    8       80    patient id
    88      80    recording id
    168     8     start date, dd.mm.yy
    176     8     start time, hh.mm.ss
    184     8     header size in bytes  == 256 + 256*ns
    192     44    reserved
    236     8     number of data records, -1 if unknown
    244     8     duration of a data record, seconds
    252     4     number of signals, ns

then, each field repeated ns times, in this order:

    ns*16  labels          ns*80  transducer      ns*8   physical dimension
    ns*8   physical min    ns*8   physical max    ns*8   digital min
    ns*8   digital max     ns*80  prefiltering    ns*8   samples per record
    ns*32  reserved

Total header = 256 + 256*ns, so 6,144 B for the 23-signal CHB-MIT files
(verified against chb01_01.edf).
"""

from __future__ import annotations

from dataclasses import dataclass, field

FIXED = 256
PER_SIGNAL = 256
PROBE_BYTES = 256  # enough to learn ns, hence the exact header size


@dataclass(frozen=True)
class EdfHeader:
    version: str
    start_date: str
    start_time: str
    header_bytes: int
    n_records: int
    record_duration_sec: float
    n_signals: int
    labels: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    phys_min: list[float] = field(default_factory=list)
    phys_max: list[float] = field(default_factory=list)
    dig_min: list[float] = field(default_factory=list)
    dig_max: list[float] = field(default_factory=list)
    prefilter: list[str] = field(default_factory=list)
    samples_per_record: list[int] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        """R32: read, never assume. Durations are not uniform -- most files are
        1 h, chb10 is 2 h, and chb04/06/07/09/23 are 4 h."""
        return self.n_records * self.record_duration_sec

    @property
    def sample_rates(self) -> list[float]:
        return [n / self.record_duration_sec for n in self.samples_per_record]

    def rate_of(self, label: str) -> float:
        return self.sample_rates[self.labels.index(label)]


def header_size_from_probe(probe: bytes) -> int:
    """Exact header length from the first 256 bytes, so the caller can issue a
    second, exactly-sized range request."""
    if len(probe) < FIXED:
        raise ValueError(f"need {FIXED} bytes to size an EDF header, got {len(probe)}")
    ns = int(probe[252:256].decode("ascii", "replace").strip())
    return FIXED + PER_SIGNAL * ns


def parse(buf: bytes) -> EdfHeader:
    """Parse an EDF header from a buffer holding at least 256 + 256*ns bytes."""
    if len(buf) < FIXED:
        raise ValueError(f"buffer too short for an EDF header: {len(buf)} B")

    def s(a: int, b: int) -> str:
        return buf[a:b].decode("ascii", "replace").strip()

    ns = int(s(252, 256))
    need = FIXED + PER_SIGNAL * ns
    if len(buf) < need:
        raise ValueError(f"truncated EDF header: have {len(buf)} B, need {need} B")

    declared = int(s(184, 192))
    if declared != need:
        # Not fatal, but it means the file disagrees with the spec. R28.
        raise ValueError(
            f"EDF header size mismatch: declared {declared} B, computed {need} B"
        )

    o = FIXED

    def block(width: int, count: int = ns) -> list[str]:
        nonlocal o
        out = [
            buf[o + i * width : o + (i + 1) * width].decode("ascii", "replace").strip()
            for i in range(count)
        ]
        o += width * count
        return out

    labels = block(16)
    _transducer = block(80)
    units = block(8)
    phys_min = [float(x) for x in block(8)]
    phys_max = [float(x) for x in block(8)]
    dig_min = [float(x) for x in block(8)]
    dig_max = [float(x) for x in block(8)]
    prefilter = block(80)
    spr = [int(x) for x in block(8)]

    return EdfHeader(
        version=s(0, 8),
        start_date=s(168, 176),
        start_time=s(176, 184),
        header_bytes=need,
        n_records=int(s(236, 244)),
        record_duration_sec=float(s(244, 252)),
        n_signals=ns,
        labels=labels,
        units=units,
        phys_min=phys_min,
        phys_max=phys_max,
        dig_min=dig_min,
        dig_max=dig_max,
        prefilter=prefilter,
        samples_per_record=spr,
    )


def expected_data_bytes(h: EdfHeader) -> int:
    return h.n_records * sum(h.samples_per_record) * 2  # int16


def expected_file_bytes(h: EdfHeader) -> int:
    return h.header_bytes + expected_data_bytes(h)

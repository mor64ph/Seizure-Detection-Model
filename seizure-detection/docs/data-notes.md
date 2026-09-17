# Data notes

R28 / PRD §0: every discrepancy between the spec and reality, every skipped file, every
channel anomaly and every format variant, with the reason. Nothing here was papered over.

All counts were measured directly against the PhysioNet open S3 mirror
(`s3://physionet-open/chbmit/1.0.0/`, public and unsigned) on 2026-09-16.

---

## 1. Counts in the PRD and on the dataset page are stale

| Fact | PRD / page says | Measured | Resolution |
|---|---|---|---|
| `.edf` files | 664 | **686** | 664 is the pre-`chb24` count. `RECORDS` has 686 entries and matches S3 exactly — zero difference either way. **PRD M1's "664 files present" criterion can never pass.** |
| Records with seizures | 129 | **141** | also pre-`chb24`. `RECORDS-WITH-SEIZURES` has 141 entries and there are exactly 141 `.seizures` files. |
| Total size | 42.6 GB | **45.76 GB** | not a discrepancy: 45.76 × 10⁹ B = 42.6 GiB. Do not "fix" this. |
| Channels per file | 23–26 | **22, 23, 24, 25, 28, 29, 31, 38** | the spread is much wider than stated. Counts: 23→275 files, 28→275, 24→54, 38→39, 22→26, 29→14, 25→1, 31→1. |
| Distinct subjects | 23 | **23** | PRD is correct. The dataset page's "23 cases from 22 subjects" is the pre-`chb24` statement. |
| Checksums | `MD5SUMS` and `SHA256SUMS` | **`SHA256SUMS.txt` only** | there is no `MD5SUMS` in the distribution. PRD §2 and M1 are wrong. The pipeline verifies SHA-256. |

`SUBJECT-INFO` lists `chb01`–`chb23` only, confirming PRD §2: `chb24` has no demographics and
the registry keeps them null rather than dropping the case. It also independently
corroborates `chb01` ≡ `chb21` — both female, ages 11 and 13, consistent with the stated
1.5-year gap.

## 2. Durations are not uniform — R32

The dataset page says files are "mostly 1 hour", with `chb10` at 2 h and `chb04`, `chb06`,
`chb07`, `chb09`, `chb23` at 4 h. Measured from headers: **only 520 of 685 probed files are
exactly 3600 s**, and the range is **600 s to 14,427 s** across 41 distinct durations.

This matters because total interictal hours is the denominator for false-alarms-per-hour
(PRD §11.2). A uniform-duration assumption would corrupt the headline clinical metric. The
pipeline reads `duration_sec` from each EDF header and records it in the ledger before the
raw file can be deleted (R11).

## 3. Non-EEG signals the PRD does not mention — R31

Found by the header pre-pass, counted in file-instances:

| Label | Files | What it is |
|---|---|---|
| `-` | 1471 | placeholder / dummy |
| `.` | 320 | placeholder / dummy |
| `FC1-Ref` … `CP6-Ref` (8 labels) | 40 each | **referential** derivations, not bipolar |
| `ECG` | 36 | electrocardiogram (`chb04`, as documented) |
| `VNS` | 18 | **vagus nerve stimulus** (`chb09`, as documented) |
| `LOC-ROC` | 11 | electrooculogram — eye movement. Not documented anywhere. |
| bare electrode names (`F7`, `T7`, `FP1`, …) | 2 each | monopolar, no bipolar pairing |
| `*-CS2` (23 labels) | 1 each | another reference scheme |
| `EKG1-CHIN` | 2 | ECG referenced to chin — mixed modality |
| `EKG1-EKG2`, `LUE-RAE` | 1 each | ECG, and limb electrodes |
| `01` | 2 | **a typo in the source data**: digit zero-one instead of letter `O1` |

The VNS channel is the dangerous one. It is a treatment-device signal, and a detector that
learned to read the stimulator rather than the brain would score well and mean nothing.

This is why R31 mandates an **allowlist** (a label is accepted only if it parses as `E1-E2`
with both electrodes recognised 10-20 positions) rather than a denylist. A denylist would
have missed `LOC-ROC`, `EKG1-CHIN` and the `01` typo, none of which were anticipated.

## 4. Duplicate channel labels

`T8-P8` appears **twice** in the header of **654 files**. Deduplicated by label keeping the
first occurrence, and logged (PRD §2).

**mne mangles this.** `mne.io.read_raw_edf` renames duplicates with running numbers
(`T8-P8-0`, `T8-P8-1`), which no longer parse as bipolar derivations — so every one of those
654 files was initially skipped for a "missing" `T8-P8`. Fixed by taking channel labels from
our own EDF header parser, which is authoritative and unmangled, and using mne only for
decoded signal values. mne preserves channel order, so position *i* in `get_data()` is
position *i* in the header.

## 5. Files with no usable channels

Exactly three, and they are precisely the landmine PRD §2 predicted:

| File | Montage | Result |
|---|---|---|
| `chb12_27.edf` | `*-CS2` referential | 0 usable bipolar channels |
| `chb12_28.edf` | bare electrode names | 0 usable bipolar channels |
| `chb12_29.edf` | bare electrode names | 0 usable bipolar channels |

These fail loudly (`skipped_reason = no_usable_channels`) and are counted in the run report
rather than disappearing.

## 6. The canonical channel set — a deliberate deviation from PRD §7.2

The header pre-pass over all 686 files found **two tiers**, not one:

- **18 labels present in 99.6% of files** (682/685): `FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3,
  F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8, P8-O2, FZ-CZ, CZ-PZ`
- **4 more at 95.5%** (654/685): `T7-FT9, FT9-FT10, FT10-T8, P7-T7`
- `PZ-OZ` at 5.7% (39 files), well below any threshold.

PRD §7.2's "≥95% of files" rule would admit all 22. Measured against the `skip_file` policy:

| Canonical set | Files surviving | Files skipped |
|---|---|---|
| 18 channels | **682** | 3 |
| 22 channels | 654 | 31 |

**Frozen at 18.** Four extra channels are not worth 28 extra discarded files, and the
18-channel set is the standard CHB-MIT common montage. Because R16 makes canonicalisation a
downstream projection rather than a pre-extraction gate, revisiting this decision is a
re-projection, not a re-download.

## 7. URL encoding: `chb02_16+.edf`

One filename contains a `+`. Unencoded in a URL path, S3 reads it as a space and the object
404s — the single failure in the first full pre-pass. Fixed by percent-encoding the key path.
PRD §2 flags the `chb17a`/`chb17b` naming variant but not this one.

## 8. `chb17b_69.edf` is 256 bytes larger than its header arithmetic predicts

`n_records × Σ(samples per record) × 2 + header_bytes` gives 51,617,024 B; the object is
51,617,280 B. A 256-byte trailing excess, equal to one per-signal header block. The only such
file out of 685. Not fatal — mne reads it — but recorded because a strict size check would
reject it.

## 9. Summary-text parser: a genuine typo in the source data

`chb09-summary.txt`, record `chb09_08`, declares 2 seizures and prints:

```
Seizure 1 Start Time: 2951 seconds
Seizure 1 End Time:   3030 seconds
Seizure 2 Start Time: 9196 seconds
Seizure 1 End Time:   9267 seconds     <- should be "Seizure 2 End Time"
```

The second seizure's end is labelled with index 1. A parser that trusts the printed index
raises or mispairs, and the first implementation here **raised and dropped all four of
`chb09`'s seizures, giving a total of 194 instead of 198.**

Resolved by pairing starts to ends **by position** rather than by printed index, and
recording the anomaly on the record rather than silently accepting it. Covered by a test.

Also confirmed: `chb09_08` has `File End Time: 24:22:17`, past 24 hours. PRD §2 is right that
these fields must not be parsed as clock times. They are ignored entirely.

## 10. R29 cross-validation: one irreducible source conflict

The 141 `.seizures` binary annotation files are an independent second source. After the
fixes above, both sources give **exactly 198 seizures across 141 records**, with **182 in
`chb01`–`chb23`** — all three PRD assertions satisfied.

140 of 141 records agree exactly. One does not:

| Record | Summary text | `.seizures` annotation |
|---|---|---|
| `chb24_21` | 2804–2872 s (68 s) | 2404–2872 s (468 s) |

End times agree; starts differ by exactly 400 s. One is a digit error and there is no third
source. The annotation implies a 468 s seizure, atypical for this database; the summary
implies 68 s, typical. The summary text is demonstrably fallible (item 9), so neither source
is clean.

**Policy, not a guess:** take the narrower interval as ictal and force the disputed 400 s to
**guard** (label −1), so it trains and scores nothing. This reuses the mechanism PRD §6.2
already provides for ambiguous windows, and it is symmetric — whichever source is right, the
model neither learns from nor is penalised on the contested span. Implemented in
`labels/disputed.py`.

### A bug in our own annotation decoder, found by this check

The first decoder advanced the sample clock by the `I` field of every annotation, including
`AUX`. For `AUX` that field is the **byte length of the payload**, not a time increment, so
every timestamp was offset by the length of the leading `## time resolution: 256` string — 23
samples, showing up as a spurious 0.09 s on every interval (2996.1 s instead of 2996.0 s).
Fixed by excluding `SUB`/`CHN`/`NUM`/`AUX` from clock advancement. A golden-bytes test now
pins `chb01_03` to exactly `[(2996, 3036)]`.

## 11. Deviations from the PRD's stated toolchain

| PRD says | Used instead | Why |
|---|---|---|
| `typer` for the CLI | stdlib `argparse` | `typer` is not installed; the CLI surface is small and argparse adds no dependency. |
| one `config_hash` (§15) | `extraction_hash` **and** `full_hash` | a cache miss now costs bandwidth, not just CPU (R15/R19). Changing `training.models` must not invalidate 45.76 GB of extraction work, so the extraction hash covers only data/signal/windowing/features. |
| download everything, then extract (§3.1, ~45 GB budget) | per-**file** stream: download → verify → extract → delete raw → checkpoint | 26 GB free disk. Largest single `.edf` is 0.18 GB against a largest case of 6.86 GB, so file-level granularity keeps peak transient disk near 0.2 GB (R14). |
| two-pass channel canonicalisation (§7.2) | header-only range-request pre-pass, then downstream projection | the two-pass design cannot survive streaming ingest. Headers are 6,144 B for a 23-signal file and S3 honours range requests, so the full 686-file manifest costs ~4.2 MB (R30). |
| cite Shoeb 2009 + Goldberger 2000 (§16) | those **plus** Guttag 2010 and the DOI | PhysioNet requires citing both the resource and the platform paper; the PRD omits the former. |

## 12. Attribution

> Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). *PhysioNet*.
> https://doi.org/10.13026/C2K01R
>
> Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and
> Treatment.* PhD thesis, MIT.
>
> Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*
> 101(23):e215–e220.

Open Data Commons Attribution License v1.0.

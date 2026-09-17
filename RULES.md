# Project Rules — Seizure Detection

**Audience:** whoever executes this build (in practice Claude Code, per the PRD's `Target executor`) plus the project owner.

**Status:** binding. These rules amend [PRD-seizure-detection.md](PRD-seizure-detection.md). Where a rule here conflicts with the PRD, this file wins, and the PRD section is cited so the divergence is traceable.

**Why this file exists:** every rule below was written after a real discrepancy or a real leakage measurement, not in advance. The point is to spend each decision once. If you find yourself re-litigating one of these, the answer is already here.

---

## A. Dataset identity

There are **two** datasets in this project and they are not interchangeable.

| Handle | What it is | Windows | Channels |
|---|---|---|---|
| `csv_scaled` | `EEG_Scaled_data.csv`, pre-derived, pre-scaled | 8.0 s | 18, identity unknown |
| `chbmit_raw` | CHB-MIT v1.0.0 `.edf` from PhysioNet | ours to choose | 23–26, labelled |

**R1.** Every derived table carries a non-null `source` column, valued `csv_scaled` or `chbmit_raw`.
*Why:* the two have different window lengths, channel counts, scaling history and label provenance. Silently unioning them invalidates every downstream number.
*Enforce:* schema check rejects null or absent `source`; a test asserts exactly one distinct value per run unless the run is explicitly declared cross-source.

**R2.** No metric, table, or figure ever mixes rows from both sources.
*Enforce:* assert single-valued `source` inside the metric computation, not only at load time.

**R3.** `csv_scaled` is never used for the headline cross-subject claim.
*Why:* it has no subject, patient, or record identifier in any column. The PRD's central deliverable (§9.1, LOSO) is unobtainable from it. Its legitimate uses are pipeline development and the leakage study in R25.

**R4.** Results derived from `csv_scaled` never claim scaler discipline.
*Why:* the file arrived globally pre-scaled, so PRD §9.4 was already violated upstream of us. State this wherever those numbers appear.

---

## B. Grouping, splitting, leakage

**R5.** `subject_id` is the only grouping key. Never `case_id`, never `record_id`, never row index.
*Why:* PRD §0 rule 1. `chb21` and `chb01` are the same human on a later admission — the database says 1.5 years, `SUBJECT-INFO` lists ages 11 and 13.
*Enforce:* assert 23 distinct subject ids, and assert that `chb01` and `chb21` both map to `sub01`.

**R6. Deduplicate before splitting, always — grouping does not substitute for deduplication.**
*Why:* measured, and reproducible by `seizure csv-study` (`csvsrc/study.py`), which runs all
four protocol combinations on `csv_scaled`. PR-AUC:

| model | dup + random | dup + grouped | dedup + random | dedup + grouped |
|---|---|---|---|---|
| hgb | 0.9880 | **0.9841** | 0.8996 | 0.8051 |
| rf | 0.9864 | **0.9817** | 0.8817 | 0.7657 |
| logreg | 0.7636 | **0.7452** | 0.6358 | 0.5639 |

The decisive column is the second. **While duplicates remain, switching from a random split to
a grouped split changes PR-AUC by 0.004 (hgb), 0.005 (rf) and 0.018 (logreg) — nothing.**
Duplicate copies land in different groups, so grouping cannot separate them. A practitioner
who grouped conscientiously would read 0.984, believe it, and be wrong.

Once deduplicated, grouping earns its keep: 0.095 (hgb), 0.116 (rf), 0.072 (logreg). And
deduplication matters *more* under grouping (0.179–0.216) than under a random split
(0.088–0.128), because grouping removes the other leak and leaves duplication exposed.

*Supersedes an earlier citation of "0.992 to 0.899"*, which came from an ad-hoc session using
36 amplitude-only features on un-notched data. The direction and magnitude hold; the exact
figures above are the ones committed code reproduces, on the full 56-column aggregated view
with the mandatory 60/120 Hz notch applied (R21).
*Enforce:* the split function takes an already-deduplicated frame and asserts zero duplicate feature-row hashes before assigning folds.

**R7.** Global **metadata** may be computed across all files. Anything derived from signal **values** or **labels** must be fold-local.
*Why:* this distinction is the one that otherwise causes circular arguments. Channel labels, file durations and sample counts carry no target information, so computing them globally is not leakage. Scalers, thresholds, feature selection, imputation values and class priors do carry it, so they are fit inside the training fold only.
*Enforce:* scalers and thresholds live inside a `sklearn.pipeline.Pipeline` (PRD §9.4) so they cannot escape the fold by accident.

**R8.** Row order is never assumed shuffled, and never assumed contiguous across a record boundary.
*Why:* `csv_scaled` rows are strongly temporally ordered — 141 label runs observed against 2505 ± 24 under shuffling, 97σ. Treating them as i.i.d. leaks. Conversely, PRD §0 rule 2: real `.edf` files have hardware gaps between them, so windows never span a file boundary.

**R9.** Hyperparameter search inside a fold uses `GroupKFold(groups=subject_id)`. Never bare `KFold`, never `StratifiedKFold` without groups.

**R10.** Test data is touched exactly once, at the end of a fold. Validation subjects carry threshold and hyperparameter selection.

---

## C. Streaming ingest and irreversibility

Disk is 26 GB free against a measured 45.76 GB raw dataset, so ingest streams one **file** at a time (R14): download, verify, extract, **delete raw**, checkpoint. Deletion is a one-way door.

**R11.** A raw `.edf` file may be deleted **only** after its ledger row is written with all seven of: `checksum_verified`, `n_samples`, `duration_sec`, `channel_labels`, `channel_stats`, `features_path`, `n_windows`.
*Why:* each is unrecoverable after deletion without re-downloading. `duration_sec` in particular is the denominator for false-alarms-per-hour (PRD §11.2) and is trivially easy to forget.
*Enforce:* `stream.delete_raw` accepts the ledger row, asserts every field is present and non-empty, and raises `CaptureIncomplete` otherwise. Deletion is never a bare `unlink` anywhere in the codebase; `test_ingest.py` parametrises over all seven fields to prove each one independently blocks deletion.

**R12.** Per-channel summary statistics are captured per **record**, aggregated per case, and pooled **per subject** at use time.
*Why:* `chb01` and `chb21` are one subject. Per-case statistics would silently give that subject two baselines, which is R5 reappearing in a new costume.

**R13.** These are never deleted: the entire non-`.edf` payload (all 24 `chbNN-summary.txt`, all 141 `.seizures`, `RECORDS`, `RECORDS-WITH-SEIZURES`, `SUBJECT-INFO`, `ANNOTATORS`, `SHA256SUMS.txt`, `shoeb-icml-2010.pdf`) and the raw `.edf` files for `chb01`, `chb12`, `chb17`.
*Why:* the whole non-EDF payload is **2.19 MB measured** — there is no reason to be selective, and it is needed for event-level evaluation. Those three cases exercise the §2 landmines (an ordinary case, the different montage, the `chb17a`/`chb17b` naming variant) at 4.04 GB, letting signal and parser code be iterated without touching the network.
*Note:* there is **no `MD5SUMS`** in the distribution. Only `SHA256SUMS.txt` exists. PRD §2 and M1 are wrong on this point.

**R14.** Ingest is idempotent and restartable at **file** granularity, not case granularity.
*Why:* measured — the largest single `.edf` is 0.18 GB while the largest case (`chb04`) is 6.86 GB. Per-file streaming drops peak transient disk from ~6.9 GB to ~0.2 GB, and per-file is already the natural unit of work under PRD §0 rule 2.
*Enforce:* resume manifest keyed on `record_id` with an explicit state field; re-running a completed record is a no-op that logs and returns.

**R15.** Never download the full `.edf` payload twice. If a design requires two passes over bulk signal data, redesign it.
*Why:* bandwidth is the dominant cost of any mistake. This is what forces R16.
*Exempt:* HTTP range requests for EDF **headers** (6,144 bytes for a 23-signal file) are not a pass over bulk data. The full 686-file header manifest costs about 4.2 MB. See R30.

---

## D. Feature schema

**R16.** Per-channel features are keyed by channel **label**, never by position. Canonicalisation is a **downstream projection** on the feature table, not a pre-extraction gate.
*Why:* this replaces PRD §7.2's two-pass design, which cannot survive streaming ingest — pass 1 needs all files present, and streaming has deleted them. It also makes PRD §14's "canonicalisation applied by position instead of by label" failure impossible by construction, and it turns `on_missing_channel` from a re-download into a re-projection.
*Enforce:* column naming is `{channel_label}__{feature}`; a test asserts no feature column name contains a bare channel index.

**R17.** Extraction is parameterised on `(fs, n_channels, window_samples)`. No hardcoded `23`, `18`, `322`, `10.0`, `8.0`, or `2048` anywhere under `src/`.
*Why:* this is the single change that lets one body of modelling and evaluation code serve both datasets. PRD §5.3's `322` becomes `n_channels × 14`.
*Enforce:* grep test over `src/` for those literals, in the same spirit as the PRD §3.3 RDD grep.

**R18.** Be generous on the first extraction pass. Extra features are nearly free while the file is already in memory and cost a full re-download afterward.

**R19.** Every derived table carries `config_hash` (PRD §15), and caching is keyed on it.
*Why:* more load-bearing here than in the PRD, because a cache miss now costs bandwidth, not just CPU.

**R20.** The pure-Python boundary holds: no Spark imports under `features/`, `signal/`, `labels/`, `eval/`. Spark lives only in `spark/`, and no RDD APIs appear anywhere.
*Enforce:* PRD §3.3 grep test.

---

## E. Signal processing

**R21.** The 60/120 Hz notch is applied before featurising, on **both** datasets, without exception.
*Why:* measured on `csv_scaled` — power at 60.0 Hz is roughly 1000× its neighbours at 55 and 65 Hz, and gamma carries 8.5% of total power. Un-notched gamma is substantially a measure of line noise, line noise varies by session, and that is session identification wearing a disguise (PRD §7.3).
*Enforce:* a test asserts post-filter power at 60 Hz is within an order of magnitude of the 55 and 65 Hz bins.

**R22.** Operation order is fixed: read, drop placeholder and duplicate channels, notch, bandpass, window, featurise. Filtering never happens after windowing.
*Why:* PRD §7.1. PRD §14 lists post-windowing filtering as the cause of gamma dominance.

**R23.** No resampling. 256 Hz is the working rate.

---

## F. Reporting

**R24.** Lead with PR-AUC. Accuracy is never reported without the `DummyClassifier` figure beside it.
*Why:* PRD §10.1 and §11.1. On the deduplicated CSV, predicting the majority class alone gives 93.7% accuracy.

**R25.** Every headline metric is reported under **all** applicable split protocols side by side, with the delta stated numerically.
*Why:* PRD §9.2. For `csv_scaled` there are three protocols, not two: random with duplicates, random deduplicated, and grouped.

**R26.** A metric biased by how the data was constructed is labelled as such **at the point of reporting**, not in a footnote.
*Why:* `csv_scaled` retains roughly 25 h against CHB-MIT's ~950 h, at a 12.8% positive rate against ~0.3% real. Its interictal is peri-ictal, so false-alarms-per-hour is computable but not clinically meaningful.

**R27.** No number appears in any report that was not produced by code. `artifacts/report.json` and `artifacts/report.md` are generated, never hand-edited.

**R28.** Discrepancies are fixed in code and recorded in `docs/data-notes.md`, never silently papered over.
*Why:* PRD §0. Every skipped file, channel anomaly and summary-format variant gets an entry with a reason.

---

## G. CHB-MIT specifics

**R29.** The summary-text parser is cross-validated against the 141 `.seizures` binary annotation files. Disagreement is a hard failure, not a warning.
*Why:* PRD §14 names "parser dropping a format variant" as the likely cause of a wrong seizure count, and PRD §6.1 requires the parser to raise rather than return empty. The distribution ships a machine-readable second source, so the 198-seizure assertion can be checked against something other than the parser that produced it. Do not skip this because the text parse "looks right".

**R30.** The canonical channel list is frozen from a **header-only range-request pre-pass** over all 686 files, before any bulk download begins.
*Why:* this recovers PRD §7.2's global two-pass guarantee for ~4.2 MB, so the canonical set is known before 45.76 GB of bandwidth is committed. It does not replace R16 — features are still stored label-keyed and canonicalisation is still a downstream projection, because that is what keeps `on_missing_channel` a re-projection instead of a re-download.
*Enforce:* the pre-pass writes a per-file channel-label manifest; bulk ingest asserts the manifest exists and its `config_hash` matches before it fetches a single `.edf`.

**R31.** Non-EEG channels are dropped explicitly by name, and the drop is logged per file.
*Why:* the distribution contains signals that are not scalp EEG and that the PRD does not mention — an **ECG** channel in the last 36 files of `chb04`, a **vagal nerve stimulus** channel in the last 18 files of `chb09`, and up to 5 dummy signals in some cases. VNS is a treatment-device signal; leaving it in a feature vector is a direct path to a detector that reads the device rather than the brain.
*Enforce:* an allowlist of scalp-EEG bipolar labels, not a denylist. Anything unrecognised is dropped and counted in the run report.

**R32.** Per-file duration is read from the EDF header, never assumed.
*Why:* file lengths are not uniform — most are 1 hour, `chb10` is 2 hours, and `chb04`, `chb06`, `chb07`, `chb09`, `chb23` are 4 hours. A uniform-duration assumption silently corrupts the false-alarms-per-hour denominator (R11, PRD §11.2). Measured: only 520 of 685 files are exactly 3600 s; the range is 600 s to 14,427 s.

---

## H. Frame-level arithmetic

These five come from an adversarial audit of the finished pipeline. Each names a bug that was
present, produced plausible-looking numbers, and was caught only by reading the code.

**R33.** Any per-record computation performed on a frame spanning several records **must**
group by `record_id` first.
*Why:* `t_start_sec` restarts at 0 in every record. A time-overlap test applied to a
concatenated frame marks windows in every record that happens to share the offset. This was a
live bug in validation-side `k` selection: a predicted run in one record marked windows in
every other record and subject, so `k` was chosen from a meaningless vector.
*Enforce:* `eval/metrics.py` exposes `runs_by_record` and `run_mask` as the only sanctioned
ways to build a run mask; `test_run_mask_is_per_record` pins it on a frame where two records
deliberately share time offsets.

**R34.** Run adjacency is decided on `window_idx`, never on wall-clock contiguity.
*Why:* `stride_sec` is configurable independently of `length_sec`. Testing
`t_start[j+1] == t_end[j]` holds only when stride equals length; under any overlap every run
collapses to a single window and event sensitivity silently goes to zero for every `k >= 2`.
Window index is stride-independent and also breaks runs at a record boundary.

**R35.** Event-level metrics are reported as **pooled numerator over pooled denominator**,
never as a mean of per-fold ratios.
*Why:* subjects carry between 1 and 15+ seizures and their recording hours differ by an order
of magnitude, so a mean of ratios weights a one-seizure subject equally with a fifteen-seizure
one — and cannot be stated in the form PRD §11.2 asks for, "detected N of 198 seizures".
*Enforce:* the report prints `detected` and `of` as separate columns so the fraction is
visible and checkable.

**R36.** Model selection uses a **validation** score. No test-fold metric may influence which
model, threshold, `k`, or feature view is reported as the headline.
*Why:* choosing the best of four models by their test PR-AUC is a hyperparameter fitted on
test, which R10 forbids; it just hides one level up from the estimator. Every fold row
therefore carries `val_score`, and the report states the basis of its choice in words.

**R37.** The contrast run must be wrong in **exactly one** way.
*Why:* PRD §9.2's random-window split exists so the delta against LOSO is attributable to the
split protocol. An earlier version also fitted its operating threshold on the test partition's
own labels, so the reported delta mixed oracle-threshold leakage with protocol leakage and was
uninterpretable. The contrast run now carves a validation slice out of its own training
partition, exactly as the LOSO arm does.

**R39.** Per-fold hyperparameter tuning is off unless it is shown to help on held-out
subjects, and any claim that it helps must cite a measured delta **on full data**.
*Why:* measured twice (Appendix C). On the complete database the headline model `hgb` gets
**worse** with search on (−0.0095 PR-AUC, one seizure fewer) at four to nine times the compute,
while `logreg` gains (+0.0096, six seizures more). The inner CV optimises for training subjects
whose optimum does not transfer, and the chosen values swing three orders of magnitude between
folds. The search is implemented and tested (R9 governs it), it is simply not enabled.
*Also:* the first measurement of this rule was taken on a prevalence-skewed subset and got two
of three model signs wrong. A per-model tuning claim requires a full-data measurement.

**R38.** Guard-excluded time leaves the false-alarm denominator.
*Why:* label `-1` windows are dropped before evaluation, so no alarm can be raised in their
time. Leaving those seconds in the denominator understates FA/h — the one metric PRD §11.2
calls clinically decisive.

**R40.** Precision is reported **pooled** as well as averaged, and the pooled figure is the one
quoted.
*Why:* R36 extended to window metrics. Measured (Appendix C): the fold mean is 0.2307 and the
pooled value 0.0037 — a 62x gap, because averaging ratios weights a subject with a handful of
high-precision windows equally with one emitting tens of thousands of false positives. Reporting
only the mean overstated real-alarm rate by two orders of magnitude.
*Enforce:* `eval.metrics.pooled_window_metrics` recomputes from summed tp/fp/fn/tn, and the
report prints `prec (mean)` beside `prec (pooled)`.

**R41.** Every FA/h figure is reported beside **time in alarm**.
*Why:* false alarms are counted per run, so a detector that never stops alarming records one
long false alarm and an excellent FA/h. Measured (Appendix C): the dummy classifier scored 185 of
185 seizures at 0.89 FA/h while alarming continuously. FA/h without an alarm-duration companion
cannot distinguish a detector from a stuck buzzer.
*Enforce:* `EventResult.time_in_alarm_frac`, tested against an always-positive predictor that
must show 0 false alarms and a fraction of 1.0.
*Scope — this applies to any score built on run counts, not only to reporting.* The rule was
written for FA/h, then the identical loophole was rebuilt days later inside a new
**model-selection** criterion: event F2 on validation scored the dummy classifier **0.5140**
against the best real model's 0.1596, and would have crowned it as headline. Scaling by
`1 - time_in_alarm` collapses it to **0.00006** and costs a real detector about a quarter of its
score. Knowing a failure mode in one place demonstrably does not prevent rebuilding it in
another, so every new run-counting metric gets the always-positive test before it is trusted.

**R42.** Cloud execution is **out of scope**. Databricks and MLflow were removed from the
acceptance criteria by the project owner on 2026-09-17; do not re-add them, and do not treat
their absence as an outstanding defect.
*Why:* neither a workspace nor a tracking server was ever available, so the criteria could not
be met locally and were blocking a project that is otherwise complete. **Descoped is not
done.** `notebooks/00_verify_portability.py` and the MLflow mirror in `tracking/sink.py` are
written and unit-tested but have never executed against a live service. Any future claim that
either works needs a real run behind it, not this rule.

**R43.** `windowing.min_overlap_frac` stays at **0.5** until the owner decides otherwise.
*Why:* the sweep measures 0.75 as better on every window metric (PR-AUC 0.211 vs 0.206) with
event sensitivity slightly higher, and the confound — a smaller positive class makes PR-AUC
harder — works against 0.75, so the finding is real. But the threshold defines the positive
class, and redefining labels is an owner decision, not an optimisation. The measurement is
recorded in the report at every run; the default does not move on the strength of it alone.
This rule exists so the open decision is neither silently taken nor silently forgotten.

**R44.** The designated configuration is **`rf` + aggregated + joint (threshold, k)**, chosen by
the project owner on 2026-09-17. Any report of it must carry the selection-bias warning.
*Why:* it detects **71 of 185** seizures against the previous headline's 56, cuts median latency
from 14.0 s to **5.75 s**, and leaves **7** rather than 10 patients with zero detections — at a
cost of FA/h 2.61 → 4.36 and time in alarm 0.207 → 0.261. That is a defensible clinical trade
that validation *window* F2 cannot express, which is precisely why it required an owner call.
*But:* the choice was made on **event-level test metrics** across eight candidate configurations,
which R10/R36 forbid as a selection criterion. So `71 of 185` is an **optimistically biased**
estimate — part of its margin is the act of selecting it. Validation F2 would have chosen `hgb`.
*Enforce:* `training.headline_model` overrides automatic selection only while the report prints
the bias warning naming what validation would have chosen. Removing the warning while keeping
the override is forbidden.
*Path out — attempted and closed.* `training.select_on: event_f2` scores the same event-level
preference on **validation** subjects, which would have made the choice unbiased had it selected
`rf`. Measured: it selects **`hgb`** (0.1251 against `rf`'s 0.1059), as does the unpenalised
variant (0.1596 against 0.1386). **Every criterion that balances seizures found against false
alarms prefers `hgb`.** The designation therefore remains an owner judgement that weights
seizures found more heavily than any derivable criterion does — legitimate, but not something
the data supports on its own, and the warning stays until a *pre-committed* criterion picks `rf`.

---

## Appendix A: measured constants of the pre-derived CSV

Established by direct measurement on `EEG_Scaled_data.csv`. **Do not re-derive these; cite them.** If a future measurement disagrees, that is a finding for `docs/data-notes.md`, not a correction to make quietly.

| Fact | Value | How it was established |
|---|---|---|
| Shape | 11,233 × 36,865 (36,864 features plus `target`) | header parse, line count |
| Layout | 18 channels × 2048 samples, channel-major | all 17 boundaries at multiples of 2048 show 3.2–5.2× the mean absolute first difference; interior points 0.998; lag-1 autocorrelation 0.78 |
| Sample rate | 256 Hz | assuming it places a ~1000× spectral peak at exactly 60.0 Hz, plus a 120 Hz harmonic |
| Window length | 8.0 s | 2048 / 256 |
| Exact duplicate rows | 877 repeats across 463 groups of size 2–4 | full-row blake2b hashing |
| Duplicate character | 85.9% positive, all label-consistent, modal stride 3,154 | minority-class oversampling by row duplication |
| Positive rate | 12.77% raw, **6.30% deduplicated** | 54% of the seizure class is duplicated rows |
| Distinct rows | 10,356 | about 23.0 h, of which about 1.45 h is ictal |
| Pseudo-events | 70 contiguous positive runs, median 140 s | label run-length analysis |
| Subject recoverability | **none** | amplitude-fingerprint clustering: silhouette falls monotonically from 0.475 at k=2 to 0.192 at k=30, with no structure near k=23 |
| Leakage decomposition | HGB PR-AUC 0.9841 grouped with duplicates, 0.8051 grouped and deduplicated; grouping alone buys only 0.0039 while duplicates remain | duplication dominates split protocol — full four-protocol table in R6, reproducible via `seizure csv-study` |
| Health | no NaNs, no dead channels, mixed line endings (CRLF header, LF data) | full-pass scan |

---

## Appendix B: measured CHB-MIT constants

Enumerated directly from the public S3 bucket on 2026-09-16 (`https://physionet-open.s3.amazonaws.com/?list-type=2&prefix=chbmit/1.0.0/`, no credentials required). **These supersede PRD §2 where they differ.**

| Fact | Measured | PRD / dataset page says | Reconciliation |
|---|---|---|---|
| `.edf` files | **686** | 664 | 664 is the pre-`chb24` count. `RECORDS` has 686 entries and matches S3 exactly — zero difference in either direction. **PRD M1's "664 files present" criterion would never pass.** |
| Records with seizures | **141** | 129 | also pre-`chb24`. `RECORDS-WITH-SEIZURES` has 141 entries and there are exactly 141 `.seizures` files. |
| Total size | **45.76 GB** | 42.6 GB | same number, different units: 45.76 × 10⁹ B = 42.6 GiB. Not a discrepancy — do not "fix" it. |
| Case directories | 24 (`chb01`–`chb24`) | 24 | `RECORDS` includes `chb24`. |
| Distinct subjects | 23 | 23 | the page's "23 cases from 22 subjects" is the pre-`chb24` statement. `chb01`≡`chb21` gives 22 across `chb01`–`chb23`, plus `chb24` = 23. **The PRD is right here.** |
| `SUBJECT-INFO` rows | 23 (`chb01`–`chb23`) | `chb24` absent | confirmed. `chb24` demographics are genuinely unavailable. |
| Checksums | `SHA256SUMS.txt` only | `MD5SUMS` and `SHA256SUMS` | **no `MD5SUMS` exists.** See R13. |
| Total seizures | 198 (182 pre-`chb24`) | 198 | stands. |
| Largest case | `chb04`, 6.86 GB | — | 4-hour files. |
| Largest single `.edf` | 0.18 GB | — | drives R14's file-level granularity. |
| Non-`.edf` payload | 2.19 MB total | — | 141 `.seizures` + 24 summaries + 6 root files. Fetch all, keep forever (R13). |
| EDF header size | 6,144 B for 23 signals | — | 256 + 256 × n_signals. Range-readable (R30). |

Independent corroboration of `chb01`≡`chb21`: `SUBJECT-INFO` lists both as female, ages 11 and 13 — consistent with the stated 1.5-year gap.

### Per-case sizes (GB, decimal)

`chb01` 1.72 · `chb02` 1.50 · `chb03` 1.61 · **`chb04` 6.86** · `chb05` 1.65 · `chb06` 2.83 · `chb07` 2.84 · `chb08` 0.85 · `chb09` 3.00 · `chb10` 2.12 · `chb11` 1.79 · `chb12` 1.25 · `chb13` 1.47 · `chb14` 1.34 · `chb15` 2.79 · `chb16` 0.96 · `chb17` 1.07 · `chb18` 1.83 · `chb19` 1.53 · `chb20` 1.42 · `chb21` 1.69 · `chb22` 1.60 · `chb23` 1.13 · `chb24` 0.90

### Revised disk budget

| Item | Size |
|---|---|
| Permanent fixtures (`chb01`, `chb12`, `chb17`) | 4.04 GB |
| Non-`.edf` payload | 0.01 GB |
| Transient (one `.edf` in flight) | 0.18 GB |
| Derived features (PRD estimate) | ~2 GB |
| **Peak** | **~6.3 GB** against 26 GB free |

### Access

Public, unsigned. No AWS CLI or `wget` on this machine — `curl` only, which is sufficient.

- Bulk object: `https://physionet-open.s3.amazonaws.com/chbmit/1.0.0/{case}/{record}.edf`
- Listing: `?list-type=2&prefix=chbmit/1.0.0/` with `continuation-token` paging
- Header-only: `curl -r 0-8191` against an object (verified working)
- Resumable: `curl -C -`
- Official equivalents from the dataset page: `wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/` and `aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ DESTINATION`

---

## Appendix C: run results, and what they overturned

**Read this first.** Everything below was originally measured on a **138-record partial run**
(67,489 windows, positive rate 1.71%) — a seizure-enriched subset, because interictal-only
records had not been ingested yet. The full database (686 records, 352,742 windows, positive
rate **0.32%**) reversed one of those conclusions outright. Figures are now labelled
`[partial]` or `[full]`, and **no `[partial]` figure may be quoted as a result.** The general
lesson is the harder one: a conclusion drawn on a prevalence-skewed subset can invert when
prevalence becomes real. Full tables in `seizure-detection/artifacts/report.md` (R27).

**PRD §8.3 — reversed on full data.** `[partial]` suggested per-channel beat aggregated under
LOSO for every model, and the appendix recorded §8.3 as "half wrong". `[full]` shows that
holds only for the *window* metric and inverts where it matters:

| `hgb`, `[full]` | aggregated (56 cols) | per-channel (252 cols) |
|---|---|---|
| window PR-AUC (LOSO) | 0.205 | **0.308** |
| random-split PR-AUC | 0.388 | 0.712 |
| **leak gap** | **0.182** | **0.404** |
| seizures detected | **56 of 185** | 45 of 185 |
| FA/h | **2.61** | 5.86 |

Per-channel scores 50% higher on window PR-AUC while detecting **11 fewer seizures at more
than double the false-alarm rate**, and its leak gap is 2.2x larger. `rf` repeats the pattern
(window 0.178 → 0.301; FA/h 6.74 → **10.86**). **PRD §8.3 is therefore vindicated at the event
level**, and the `[partial]` reversal was an artifact of prevalence. The operative rule: judge a
feature view on event metrics, never on window PR-AUC alone.

**The inflation is not a property of the split alone.** Logistic regression shows −0.024 on the
56 aggregated features and +0.217 on the 252 per-channel ones. A hyperplane has identical
capacity in both cases, so subject leakage is carried by the **channel-specific features**, not
purely by model capacity. Corollary worth keeping: quoting a single "random-split inflation"
figure for a dataset is meaningless — it is a property of the feature view and the model class
together.

**Per-fold hyperparameter tuning does not help the model we report (R39).** PRD §9.3's inner
`GroupKFold` search, aggregated view, 23 folds. First measured `[partial]`, then re-measured
`[full]` — keep both, because the disagreement between them is the more useful result.

`[partial]` (138 records, 1.71% prevalence) — **superseded, do not quote:**

| model | search off | search on | delta | compute |
|---|---|---|---|---|
| logreg | 0.2902 | 0.2898 | −0.0004 | 4.0x |
| hgb | 0.3481 | 0.3510 | +0.0029 | 8.7x |
| rf | 0.3260 | 0.3143 | **−0.0117** | 9.1x |

**Re-measured `[full]`** (aggregated view, 23 folds, 0.32% prevalence, artifacts preserved in
`artifacts/search_on_full/`):

| model | search off | search on | delta | seizures off → on | FA/h off → on |
|---|---|---|---|---|---|
| logreg | 0.1635 | 0.1731 | **+0.0096** | 44 → **50** | 3.98 → **3.66** |
| rf | 0.1782 | 0.1699 | −0.0082 | 62 → 61 | 6.74 → 6.78 |
| hgb | 0.2054 | 0.1960 | **−0.0095** | 56 → 55 | 2.61 → **2.46** |

**The verdict holds; two of the three `[partial]` signs did not.** `logreg` went −0.0004 →
**+0.0096** and `hgb` went +0.0029 → **−0.0095** — both flipped. Only `rf` kept its sign. So the
`[partial]` table was directionally unreliable for two of three models, and any *per-model*
claim drawn from it was unsound; only the aggregate conclusion survived by luck.

The corrected reasoning is not "search never helps". It is: **search helps the weakest model and
hurts the strongest.** `logreg` improves on three independent axes at once (+0.0096 PR-AUC,
+6 seizures, lower FA/h), which is hard to dismiss as noise — regularisation strength genuinely
matters for a single global hyperplane. But `hgb`, the headline model, gets **worse**, and the
chosen parameters remain wildly unstable: `logreg`'s `C` picked 10.0 in 9 folds, 0.01 in 7 and
1.0 in 6 — a three-order-of-magnitude spread across folds of the same dataset. An optimum that
moves that much between folds is not an optimum.

`training.search` therefore stays **false**: it costs 4–9x the compute to make the reported model
worse. The `logreg` gain is recorded as a real but unused finding — it would only matter if the
linear model were ever promoted to headline.

No model gains beyond noise, one gets materially worse, at four to nine times the compute.
The mechanism is the same one behind the threshold spread: the inner CV maximises PR-AUC on
held-out *training* subjects, and that optimum does not transfer to the test subject. The
chosen values swing wildly between folds — logistic regression's `C` picked 0.01 in ten folds
and 10.0 in eight, a three-order-of-magnitude spread. So the search adds variance without
reducing bias. `training.search` therefore defaults to **false**, and the negative result is
the finding rather than a reason to keep tuning.

**Threshold variance is the readiness signal.** `[full]` The F2-selected threshold ranged
0.000–0.994 across the 23 folds and tuned `k` split 12/8/3 across {2,1,3}. `k` is therefore
**not a configured constant** and must never be quoted as one.

**A strong average hides a bimodal failure. `[full]`** `hgb` + aggregated pools to 30.3% event
sensitivity. Per subject that is **10 of 23 detecting zero seizures**, 7 detecting all, 6
partial — and of the 7 "perfect", only `sub05` (precision 0.640), `sub08` (0.947) and `sub19`
(0.846) are genuine. `sub04`, `sub09`, `sub10` and `sub16` reach sensitivity 1.000 at precision
0.001–0.009 by alarming almost continuously. The honest headline is **"works on 3 of 23
patients"**, not "30% sensitivity". Always publish the per-subject table.

**Pooled and averaged precision disagree by 62x. `[full]`** For `hgb` + aggregated the mean of
per-fold precisions is **0.2307**; pooling the confusion matrix gives **0.0037** (279 true
positives against 74,678 false). Roughly 1 flagged window in 270 is real. R36 forbade
means-of-ratios for event metrics but was never applied to window metrics, so for a long time
only the flattering figure was reported. Both are now printed, labelled.

**FA/h alone is gameable, and the dummy model proved it. `[full]`** False alarms are counted
per *run*, so a detector that never stops alarming scores one long false alarm. Under joint
(threshold, k) selection the `DummyClassifier` — which predicts the majority class for every
window — reported **185 of 185 seizures detected at 0.89 FA/h**, better than every real model,
at window precision 0.0024. This is why step 0 of the model ladder is mandatory: a deliberately
useless model is the only reliable detector of a broken metric.

**Feature-subset experiment: the amplitude hypothesis failed, and the mechanism is capacity.
`[full]`** PRD 8.2 argues absolute amplitude is close to a subject fingerprint, so the 56
aggregated columns were split in half -- 28 amplitude-invariant (relative band powers, Hjorth
mobility and complexity, all unchanged by scaling the trace) against 28 amplitude-dependent
(log absolute powers, line length, variance). Artifacts in `artifacts/experiments/`.

| features | model | LOSO PR-AUC | leak gap | seizures | FA/h | zero-detect subjects |
|---|---|---|---|---|---|---|
| all 56 | rf | 0.178 | 0.161 | 71 | 4.36 | 7 |
| all 56 | hgb | 0.205 | 0.182 | 59 | 3.52 | 8 |
| invariant 28 | rf | **0.017** | 0.094 | 116 | **25.32** | 3 |
| invariant 28 | hgb | **0.017** | 0.076 | 92 | 15.75 | 3 |
| dependent 28 | rf | 0.114 | **-0.091** | 44 | 4.10 | 8 |
| dependent 28 | hgb | **0.227** | 0.114 | 64 | 3.96 | 6 |

**Dropping amplitude destroys the model**, PR-AUC 0.178 -> 0.017. Its apparent gains -- 116
seizures, only 3 subjects at zero -- are the saturation pattern again, bought with 25.3 FA/h.
**But the leak shrank in *both* halves**, and the amplitude-dependent half leaks less than the
full set while scoring better. If amplitude-dependence caused the leak, that half should leak
like the whole. So the leak tracks **feature count, i.e. capacity**, more than it tracks scale,
and PRD 8.2's fingerprint argument is at best partial. Practical consequence: `hgb` +
amplitude-dependent-only is the strongest configuration yet measured (PR-AUC 0.227, leak 0.114,
64 of 185, FA/h 3.96, 6 zero-detect) on **half** the features -- but it was found by comparing
*test* results, so adopting it would repeat the R44 error. It may only be promoted if a
pre-committed validation criterion selects it.

**Threshold and `k` are not separable. `[full]`** Choosing the threshold first and `k` second can
select a pair that detects nothing: `sub07` drew threshold 0.9927 with `k=3`, producing 6 correct
windows, **zero** false positives, and **0 of 3** seizures detected — no 3 flagged windows were
ever consecutive. Raising a threshold shortens every run. Joint search over (threshold, `k`) on
validation recovered `sub23` (0 of 7 → 3 of 7) and made `rf` + aggregated better on both axes at
once (62 → **71 of 185**, FA/h 6.74 → **4.36**), by choosing `k=3` in 17 of 23 folds while
lowering the median threshold 0.480 → 0.268. It did **not** rescue `sub07`: threshold and `k` are
tuned on validation subjects, and no search fixes a held-out subject whose score distribution
differs. That residue is an irreducible LOSO limitation, not a tuning error.

---

## Attribution

Per the Open Data Commons Attribution License v1.0 (PRD §16). PhysioNet requires citing **both** the resource and the PhysioNet platform paper; the PRD omits the former, so all three go in the README and `docs/data-notes.md`:

> Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). *PhysioNet*. https://doi.org/10.13026/C2K01R
>
> Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD thesis, MIT.
>
> Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23):e215–e220.

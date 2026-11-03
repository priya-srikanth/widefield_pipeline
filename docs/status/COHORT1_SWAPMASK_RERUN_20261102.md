# Cohort 1 re-run on the swap-masked, 415-lag-fixed correction (opened 2026-11-02)

Priya 11-02, asking "did we end up checking whether LED-swapped or bleedthrough images are a problem for cohort 1
data?", agreed to the re-run below ("sure"), and asked for this record ("keep a record that we're doing this and why in
the repo"). Status: **RUNNING (started 11-02 evening, in parallel with the frozen-store pass 2 at Priya's request).**

## Why

The cohort 2 QC (docs/COHORT2_QC_2026-10-29.md Â§0b-0d, Â§0i, Â§0l) found two defects in the hemodynamic correction.
Both are also present in cohort 1 (PS92-95):

1. **Swapped frames.** A few frames per session carry the OTHER LED's image. On the DAQ the TTLs look clean at these
   frames; it is an occasional camera / LED timing slip.
   - Cohort 1 count (`swap_scan.py --cohort1`, 35 sessions): 10-171 frames per session (0.004-0.07% of pairs), and
     879 flagged pairs in PS92_0828.
   - In the adopted corrected signal each one is a single-frame spike of 2-11 robust SD. Together they hold 0.5-1.5%
     of corrected variance, and 10% in PS93_0826.
   - They pull the hemodynamic coefficient low in some sessions (PS93_0826 0.85 -> 1.36).
   - They appeared as one-frame "events" in the cohort 1 event videos (PS92 08-28 dips).
2. **The 16 ms 415 lag.** 415 is sampled one frame after 470. The new variant advances it before the fit.

**Calcium bleed-through into 415** (09-19, STATUS_2026-09-19) is a separate, known issue. It affected only the retired
"neurovascular coupling" measure and is not the subject of this re-run.

## What was already checked (10-30) and what was not

The new variant is `hemo_meegkit_hpfit_lag_refitT_swapmask` (swapped pairs interpolated, 415 advanced 16 ms, T
refitted). It was compared against the adopted `hemo_meegkit_hpfit`.

- **Â§0i** (`cohort1_impact.py`, 8 sessions, including the worst):
  - Trial-averaged maps r .97-.997; time courses r .98-.998.
  - Amplitudes -2 to -8%, and -14% in PS93_0826.
  - Residual heartbeat drops from 35-97% to 3-9%.
- **Â§0l** (`cohort1_subset_headline.py`, 24 sessions: 2 pre / 1 acute / 1 subacute / 2 chronic per animal):
  - The epoch change in spout decoding keeps every sign and size (largest shift .02).
  - Area-amplitude epoch changes are identical to 3 decimals.
  - Verdict then: no blanket re-analysis needed. The new correction (with the pre-SVD repair) is the default going
    forward.

**Not tested:**
- the LocaNMF / frozen-decoder / encoding pipeline;
- the RESTnl maps;
- the quiet-period events.

**The real exposure is any amplitude compared ACROSS sessions** (epoch ratios, stroke time courses). The shrink varies
by session (2-14%), so it does not cancel the way within-session contrasts do. Single-frame spikes also matter for any
event detector. The quiet-event v2 definition already excludes events within 0.5 s of a swapped pair; earlier event
results were not re-checked.

## Plan

1. Build the variant for every curated cohort 1 session. Use new `hemo_*` subdirectories only. The adopted product,
   `SVTcorr.npy` and every existing variant stay untouched.
2. Find out how each downstream family selects its hemo variant. Then re-render into NEW folders:
   - **(a) frozen decoders (imaging Figs 2.x).** Do NOT overwrite the stored frozen models. The variant changes the
     inputs, so it needs its own spec ids.
   - **(b) the encoding headline.**
   - **(c) the RESTnl maps (imaging section 7), including the PS94 chronic ratio (about .71).**
   - **(d) the quiet events (imaging section 15 / encoding page section 23).**
3. Report per animal first, then pooled. List only what moves. If anything changes, it is ADDED to the pages as a
   sensitivity check after discussion with Priya; nothing is replaced (Priya 11-01).

## Log

- 2026-11-02: queued. Running list RUNNING_LIST_2026-11-01.md.
- 2026-11-02 ~22:10: started in parallel with frozen-store pass 2 (Priya: "parallelize where possible"). Own worktree wf_swapmask, branch analysis/cohort1-swapmask; <=4 workers per job; stored frozen models untouched; frozen-decoder family (a) last.

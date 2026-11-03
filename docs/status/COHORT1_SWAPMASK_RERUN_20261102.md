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
- 2026-11-02 (helper): worktree `C:/Users/SabatiniLab/Github/wf_swapmask`, branch `analysis/cohort1-swapmask` off
  `analysis/frozen-default` 2f25bced. All 120 curated cohort-1 session dirs are under `M:/MICROSCOPE/Priya/...` (checked);
  25 already had the variant from 10-30 (manifests match: variant, refit, lag shift, func channel, interpolated-pair
  count). Build of the other 95 started (`scripts/cohort2_qc/build_swapmask_cohort1.py --jobs 4`, bad pairs from the
  10-30 `swapped_pairs.npz`, new `hemo_*` dirs only).
- 2026-11-02 (helper): how each family picks the variant: (c) RESTnl maps and (a) frozen decoders read
  `config.svtcorr_path` -> `WIDEFIELD_HEMO_VARIANT`; (b) encoder and (d) quiet events read
  `combined_encoding.load_imaging`, which HARD-CODED `hemo_{ADOPTED}` -> new `combined_encoding.hemo_dir()` honours the same
  env var (default unchanged, test added). Output-root overrides added: FROZEN_OUT_ROOT, RESTNL_OUT_ROOT, QS_OUT_ROOT.
  Commit b334316a. TRAP found: `session_cache.cached` PRUNES every other signature of the same label x kind, so a variant
  run against the shared session cache would delete the adopted prm / tf caches (and restnl_mask prunes its own dir the
  same way) -> every variant run uses `WIDEFIELD_SESSION_CACHE=C:/Users/SabatiniLab/.widefield_session_cache_swapmask`.
- 2026-11-02 (helper): (a) the epoch decoders do NOT read per-session LocaNMF: they use the per-animal JOINT basis
  (fixed footprints) and PROJECT each session's SVTcorr, so the variant enters without any LocaNMF re-run. Trap: in-fit
  (pre) sessions were served their stored C (adopted data) -> `joint_locanmf.project_all()` (commit 4813c3cf) projects
  them too under a non-default variant (or `WIDEFIELD_JOINT_PROJECT_ALL=1` for the adopted control). A per-session
  LocaNMF re-run is NOT needed for any family here and was NOT run: measured 63-103 s/session on the GPU (~2.5-3.5 h
  for 120), and a refit of the ADOPTED PS93_0826 on today's torch 2.11 / RTX 5060 gives 95 components vs the stored 93,
  i.e. a refit is not reproducible and would confound the hemo change. Fixed-footprint LS projection reproduces the
  stored per-session C at median r .9994 (min .977) on PS93_0826.
- 2026-11-02 (helper): (b) runs on a second worktree `wf_swapmask_enc` (branch `analysis/cohort1-swapmask-enc`,
  9f458196 = the commit of the published `frozen_encoder_gate-attempted_v7g3_t3d` run, + the same two env overrides,
  e80dbef7), so code drift since 10-24 cannot masquerade as a hemo effect. Outputs:
  `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/swapmask_rerun_20261102/{quiet_events,restnl,encoding,frozen}`.
  Store safety: `WIDEFIELD_EPOCH_FROZEN=0`, `WIDEFIELD_REST_FROZEN=0` AND `WIDEFIELD_FROZEN_MODEL_DIR` pointed at a
  separate empty dir.
- 2026-11-02 ~23:15 (helper): variant BUILD DONE for all 120 curated cohort-1 sessions: 95 built, 25 reused, 0 failed
  (log `C:/Users/SabatiniLab/swapmask_rerun_logs/build_summary.jsonl`). (d) quiet extraction running (3 workers);
  (c) RESTnl chain started; (b) waits for (d) to finish (memory).
- 2026-11-02 23:45 (helper): (d) QUIET EVENTS DONE (v2_quiet_lp definition, k = 4, 108 sessions, 0 failed; RESTnl
  baselines for (c) also done: frame counts 107/107 equal to restw_check, as expected since they do not depend on hemo).
  Orofacial (positive) per animal acute/pre adopted -> swapmask: PS92 4.8 -> 3.4, PS93 4.7 -> 5.0, PS94 3.1 -> 2.6,
  PS95 5.4 -> 6.1 (4/4 rise either way). LMM acute +.63 -> +.60, subacute +.37 -> +.37 (both q < .005 either way);
  chronic +.22 (q .08) -> +.13 (q .22). Both polarities: chronic q .039 -> .032. Absolute rates are higher on the
  variant (lower robust SD; PS92 pre 2.0 -> 4.1 /min); session-level r .86. Limb: both-polarity acute q .19 -> .046
  (moved; PS92 acute 8.3 -> 19.9 /min drives it). Outputs `.../swapmask_rerun_20261102/quiet_events/v2_quiet_lp/`
  (+ `compare_adopted/`). (b) frozen encoder started 23:44 (2 workers).
- 2026-11-03 02:45 (helper): (c) RESTNL MAPS DONE (contact precue / cue / lick, LP cue / lick; 15r + 15rpa + figdata
  bundles, `--plot-only` works via restnl_maps). Far-contra RESTnl ratio per animal (PS92/93/94/95) adopted -> swapmask:
  contact cue ACUTE .59/.52/.60/.43 -> .55/.52/.58/.44 (animal mean .535 -> .524); CHRONIC 1.01/1.20/.71/1.21 ->
  1.00/1.23/.71/1.20 -- **PS94 chronic .71 unchanged** (.710 -> .712); LP lick acute .50/.55/.82/1.07 -> .46/.54/.80/1.09,
  chronic PS94 .72 -> .72. Precue acute 2.00/1.17/1.18/1.63 -> 1.83/1.08/1.10/1.70 (still > 1 in 4/4). Median |change|
  over all cells .017-.028; outliers are PS92 subacute (2 sessions; close-ipsi .79 -> 3.19 precue) and PS93 subacute
  far-contra (.95 -> .83 cue). Outputs `.../swapmask_rerun_20261102/restnl/{epoch_restnl,LP_licks/epoch_restnl,compare_adopted}`.
- 2026-11-03 02:49 (helper): (b) FROZEN ENCODER DONE (107 sessions, at commit 9f458196 + overrides). Headline cells
  adopted -> swapmask (cue, frame R2, mean of animals' change from pre, nested 95% CI): target unique cortex chronic
  -.012 [-.021,-.004] 0/4 up -> -.013 [-.022,-.005] 0/4; spout unique SSp-m contra - ipsi chronic +.016 [.003,.029] 4/4 ->
  +.018 [.005,.031] 4/4 (per animal .016/.019/.016/.012 -> .013/.019/.020/.018). Joint R2 changes from pre move <= .016;
  pre R2 rises slightly (+.01-.02 per animal). Adopted numbers reproduce the page exactly. Nothing moves beyond noise.
  Outputs `.../swapmask_rerun_20261102/encoding/{frozen_encoder_gate-attempted_v7g3_t3d,compare_adopted}`.
- 2026-11-03 04:41 (helper): (a) FROZEN DECODERS (epoch acc, DAQ 2 s, 5 arms, in-memory, every session projected on the
  joint basis) DONE: swapmask `.../frozen/epoch`, adopted control of the same method `.../frozen_adopted_projall/epoch`,
  compared with the published in-memory render `frozen_default_20261102/inmemory/epoch`. Ran against
  analysis/frozen-default 2f25bced (unchanged during the run). Far-contra pooled bar, published / adopted-projall /
  swapmask: cue working acute .322 / .330 / .301, chronic .740 / .740 / .733; lick acute .500 / .516 / .516, chronic
  .750 / .777 / .772; ENL working acute .251 / .271 / .258, subacute .340 / .374 / .394, chronic .410* / .435 / .453.
  Hemo effect (swapmask vs control): bars |d| median .007-.012, max .043; method effect (control vs published) is of the
  SAME size (max .12 on deltas). Mark changes vs control: ENL subacute fC ** -> *, chronic fI * -> **; lick acute fC ** -> *,
  several near/mid lick cells lose a star. Per-animal far-contra acute drop stays in 4/4 (where computable).
- 2026-11-02 ~22:10: started in parallel with frozen-store pass 2 (Priya: "parallelize where possible"). Own worktree wf_swapmask, branch analysis/cohort1-swapmask; <=4 workers per job; stored frozen models untouched; frozen-decoder family (a) last.

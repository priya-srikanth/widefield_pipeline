# State-space recovery: handoff (2026-11-07, night)

**START HERE.** Exploratory arm asking, in BCI terms, whether post-stroke cortical activity is the pre-stroke activity
scaled, rotated, shifted, re-associated or new, and how it gets there through acute and subacute. Cohort 1 (PS92-95)
only. Everything is on branch `analysis/state-space-recovery-20261107` in worktree `C:/Users/SabatiniLab/Github/
wf_statespace` (NOT pushed, NOT merged; push only if Priya asks -- it is our own branch).

- Results page (draft, private, Priya's): https://claude.ai/artifact/9sGELLFEwADEt9k64fXTxe (v11). Source:
  `docs/results_pages/state_space/index.html` (+ backup `M:/MICROSCOPE/Priya/Widefield/state_space_recovery/page/`),
  figure registry `fig_ids.csv` (`make_fig_ids.py`), renumbering `renumber.py` (by position; ids `f-<image stem>`
  are permanent). Working copy of the page + images used for publishing: session scratchpad `state_space_page/`
  (re-copy PNGs from `state_space_recovery/page_figs/` if the scratchpad is gone).
- Plan, every number, decisions, retractions: `docs/STATE_SPACE_RECOVERY_PLAN.md` (read top to bottom; later sections
  supersede earlier ones). Literature: `docs/STATE_SPACE_RECOVERY_LITERATURE.md`.
- **11-08 (overnight, Priya asleep): read `docs/status/RUNNING_LIST_2026-11-08_STATE_SPACE.md` FIRST** -- decisions
  waiting for Priya (section A), new findings B1-B3 (detour axis vs the balance marker; within-session neural drift
  tracks landing drift post-stroke 4/4; the repertoire test has NO power -> slide-vs-search withdrawn), plans fixed
  before the LocaNMF / June data (C), status log (D). FINDINGS 0c is on main (not pushed); the O2 joint-LocaNMF job
  `rebuild_20261103/o2/ss_lnmf_20261108` was staged by Priya 11-08 (outputs not on the share yet).

## 11-08 DAY -- START HERE (latest state; supersedes the branch / worktree note above)

Code is on MAIN (`C:/Users/SabatiniLab/Github/widefield_pipeline`; the wf_statespace branch was merged 11-08). Read the
running list `docs/status/RUNNING_LIST_2026-11-08_STATE_SPACE.md` sections A (decisions) and B (B4-B14) first.
- **Bases** (`WF_SS_BASIS`): allen (default) | lnmf (per-animal joint LocaNMF, O2 `ss_lnmf_20261108`; glue / bulb
  components DROPPED at load by default -> outputs `_lnmfbm`; `WF_SS_LNMF_MASK=none` = the unmasked `_lnmf` runs) |
  svd (ONE cohort SVD on the brain mask, glue + bulbs removed, built LOCALLY: `scripts/o2_state_space_svd.py bundle
  --local` + `run-local`, bundle `rebuild_20261103/o2/ss_svdbm_20261108`; outputs `_svd`).
- **REBUILD MASK BUG (B12, decision A9 open):** under the rebuild profile `beta_maps.painted_exclusion()` is silently
  empty (masks only in the canonical grant_figures) -> every rebuild `brain_mask()` consumer (encoder, area masks of
  the Allen extract, maps) includes the glue. This arm resolves its masks WITHOUT the profile (guard in
  `o2_state_space_svd.analysis_mask`). Shared fix proposed, not applied.
- **Cortical change maps** (`scripts/state_space_chronic_maps.py`, svd only): per animal change vs own pre in units of
  its pre response, mean of 4; versions full / nocw (minus cortex-wide PC1) / spout (spout-specific); contours =
  `beta_maps.significance_contour` (nested bootstrap, as the 15r / residual maps); per-animal session-shuffle contours
  + k/4 row; split-basis area table (ALM / tjM1, ROI v8) + ALM / tjM1 outlines; grey = unanalyzed brain.
  `WF_SS_MAPBASE` lick (pre-lick window; F12 caveat, epoch-dependent) | rest (RESTnl on the drift-removed signal,
  `scripts/state_space_rest_baseline.py`) | none (session-mean reference) | enl (the trial's lick-free >= 1 s ENL window,
  `scripts/state_space_enl_baseline.py`, from the continuous signal); `WF_SS_MAPWIN=precue` = the pre-cue map (the ENL
  window itself, vs rest / none). Trials: working (camera gate) first-lick trials, misses excluded.
  Figures `page_figs/ss_cmap_<version>[_<base>][_precue]_<align>_svd.png`; tables `chronic_maps_<align>_svd[_<base>]...`.
  Final sweep of all 10 variants: scratchpad `maps_sweep.sh` (11-08 ~10:30).
- **11-08 late morning:** SVD state-space set complete (running list B15: the cohort SVD reads like Allen -- larger,
  no rotation -- so re-patterning is LocaNMF-only; affects A8). Sweep: RESTnl + ENL lick maps done (B16: PS94's
  negative no-subtraction map is mostly the session-mean reference). "Why cleaner": `scripts/state_space_why_cleaner.py`
  (plan B17; reads the 15r rebuild cache, refuses to recompute) -> `why_cleaner_lick/steps.csv`,
  `page_figs/ss_why_cleaner_lick.png`. Figure titles now wrapped (`_wrap`); re-render all variants with `--replot`
  after the sweep.
- **11-08 ~11:45:** page v15 has section 9 (maps, all baselines, why-cleaner). "Why cleaner" answered for the 15r
  activity maps (B19: window, not denoising); the residual-map half not done.
- **11-09 ~03:00 (running list B29):** `scripts/state_space_shift_recovery.py`: pre-lick orofacial rise grows with
  far-contra aim recovery (4/4); post-lick orofacial increase and posterior-medial transient do not track aim. Page
  v25 Fig 10.23 (page script `state_space_page/add_shift_recovery.py`). Next: pre-lick rise vs Fig 9.8's pre-cue rise.
- **11-09 ~02:30 (running list B28):** `scripts/state_space_shift_movement.py` (jaw + FaceRhythm + running on top of
  licking; no motion energy per Priya): orofacial increase survives (residual = observed, 4/4); part of the
  posterior-medial transient is movement. Page v24 Fig 10.22 (page script `state_space_page/add_shift_movement.py`).
- **11-09 ~01:45 (running list B27):** `scripts/state_space_shift_behavior.py`: the shared orofacial increase and the
  posterior-medial transient are NOT explained by licking (frozen pre-stroke activity-vs-licking fit; residual =
  observed). Page v23 Fig 10.21 (page script `state_space_page/add_shift_behavior.py` in the session scratchpad).
  Next: jaw / face / body around the first lick.
- **11-09 ~01:15 (running list B26):** `WF_SS_SHIFTBASE=rest|enl` anchors the shared change to RESTnl / the trial's
  ENL window: the shift's dips were the window-mean construction; it is an increase (post-lick posterior-medial
  transient + sustained anterolateral orofacial increase). Page v22 (Figs 10.17-10.20; page scripts in the session
  scratchpad `state_space_page/add_shift_ref.py`).
- **11-09 ~00:45 (running list B25):** `scripts/state_space_shift_carried.py` = where x when the shared shift is
  carried (area x bin heat maps, per-bin energy, per-area enrichment; SVD raw + gsr, ~1 min each, local). Page v21
  Figs 10.13-10.16. Next analysis unchanged: is the shift timing / behavior (B24 NEXT).
- **11-08 ~15:00 (running list B24):** page v17 = basis switch for sections 3-7 and 10 + new section 10 "what the
  change is made of" (8-part split per animal / per session, raw + minus cortex-wide, part maps on cortex; Table 10)
  + two "Where things stand" rows. Page source: session scratchpad `state_space_page/index.html` (scripts
  `add_parts_section.py`, `add_basis_switch.py` there; v16 backup `index_v16_backup.html`). Shared shift = sharper
  lick-locked transient over posterior-medial cortex; minus cortex-wide a medial-then-lateral-orofacial sequence.
  All runs done 17:47; page v20 has every basis version of Figs 10.2 / 10.4. Nothing running. Next analysis: is the
  shift timing (B24 NEXT).
- **11-08 ~12:45 (in flight; running list B20-B23):** page v16 has the basis comparison (Figs 3.10 / 3.11) and the
  method check per basis (2.2 / 2.3). Sections 3-7 rendered in SVD + LocaNMF (basis switch built, not yet published).
  `scripts/state_space_change_parts.py` = exact 8-part split of the change (Allen raw done: on manifold; largest
  chronic part a static per-area level, largest acute part the time-varying shared shift; rotation small, above
  floor). Running: SVD / LocaNMF, raw + minus cortex-wide, animal + per session (scratchpad masked/p8_lane_*.log),
  then `scripts/state_space_part_maps.py` (what the parts look like on cortex; masked/partmaps_lane.log). When done:
  publish the page (basis switch + a "what the change is made of" section with per-session figures + part maps),
  update "Where things stand", commit.
- **Open next:** FINDINGS once A8 is decided (wording must cover B15); residual-map comparison; two-part maps
  (behavior-predicted vs residual, plan B14, inputs `frozen_encoder_gate-attempted_v7g3_t3d_spont_frA_basis`); A8
  (0c item 1 wording), A9 (mask bug).

## Where things stand (one line each; details in the plan)

1. Method check (planted changes): rotation detectable only >= ~20-30 deg in the main plane; per-area gains look
   "outside the pre space"; shared shift barely detectable.
2. Raw 62-area space: larger (amp +.30-.38 chronic), shared shift, small out-of-space part; NO rotation beyond gain on
   like-for-like trials. Not the changed movement (on-target / landing / contact-only all agree).
3. Alignment: cue window misses late acute responses; first-lick alignment is the fairer one (acute r -.08 vs -.20).
4. Dimensionality: one cortex-wide mode carries 84-96 % of the mean response and 77-83 % of single-trial
   fluctuations. Removing it, or keeping only the spout-specific part, shows a clearly re-patterned code (chronic r
   -.11 / -.20) with 5-10x more out-of-space activity (fraction of a smaller total variance: size not comparable
   across variants). The raw "gain + shift" picture is the cortex-wide mode. k = 8 checked (k = 4 / 12 match).
5. Path: acute = a sideways detour (~75-90 % of the length of acute - pre, ~55-85 % of its variance, off the pre -> chronic line, noise-corrected, every animal and basis; along-line part away from chronic in PS92 / PS94, toward in PS93 / PS95); subacute
   slides back toward and along the line. Slide vs search-then-select at the TRIAL level: NOT TESTABLE (11-08 power
   check, running list B3: single-trial scatter along the line ~1.4x the whole pre -> chronic distance; planted
   two-state sessions score like the data). The earlier "trial clouds move as a whole" is withdrawn.
6. Gradual, not a step: readouts and the behavioral far-contra aim are ~linear in log days; the acute-departure axis
   tracks far-contra aim within animal (r -.78 to -.93), also within epochs. All six spouts track it; beyond log(day),
   position on the line still tracks aim at every spout.
7. Hemispheric balance: at matched target + landing, subacute / chronic balance moves contralesional at contra-side
   targets; target-to-target spread compresses. (The 11-07 chat claim "follows the target" was RETRACTED.)
8. Noise: trial noise does not scale with the signal -> spread read RAW.
9. Contact vs incomplete (Priya: contact + incomplete primary, contact-only robustness, incomplete-only descriptive,
   within-session contrast): contact-only reproduces everything; incomplete licks smaller and lag on the recovery line.
10. Within session: every epoch drifts (later trials less chronic-like, pre included). PS94 subacute / chronic drift
    strongly; behavior over the same trials: contact flat, landings drift ipsilesionally, latency lengthens; neural
    slope tracks landing drift in PS94 / PS93 only. ALL animals (change vs own pre of the within-session slopes):
    latency lengthens more acutely (+180 ms first -> last trial, 4/4, boot * and LMM dagger; pre itself +81 to +193)
    and subacutely (+75, 4/4, n.s.), back by chronic; far-contra landings drift ipsilesionally within SUBACUTE
    sessions in 4/4 (-.24 mm vs own pre, boot *; acute strong in PS93 -.75 / PS95 -1.28; chronic only PS94 -.40).
    A within-session fatigue of aim that is larger after the stroke -- candidate follow-up with the behavior arm.
    OUTSOURCED 11-07 (Priya: "i wonder how this relates to the animals 'quitting' after stroke ... independent of our
    BCI analyses"): a separate agent in its own worktree / branch (`analysis/within-session-decline-quitting-20261107`
    or similar), doc `docs/WITHIN_SESSION_DECLINE_QUITTING.md`, outputs `Widefield/within_session_decline/`. Brief:
    reconcile with the 09-20 engagement findings (no within-session motor fatigue in ILI; quitting time-driven),
    prodrome of quitting, fatigue vs satiety vs time, far-contra specificity. Check its branch before redoing this.
11. Pre-stroke drift: part of chronic - pre lies along the pre drift in PS92-94; drift slows after the stroke in those
    three; PS95 barely drifted pre. Needs a no-stroke reference (cohort 2 pre, kept separate).
12. OPEN: close-spout PC2 transient post-stroke (posterior-medial vs contralesional orofacial); not latency, not reach.

## Decisions (Priya 11-07)

- Working trials = contact + incomplete (primary); contact-only robustness; incomplete-only descriptive.
- Trial spread read raw (noise checks).
- Stars: bootstrap ("*", boot_delta CI) PRIMARY; epoch_change_lmm only as a backup dagger. (Memory
  `stars-bootstrap-primary-lmm-backup`; the repo docstring of epoch_change_lmm says the reverse.)
- Units of each readout in panel titles + captions; cue before lick everywhere; American spelling.
- Lick-evoked cue-vs-lick traces already exist on the goal page (14.7-14.22): cite, do not rebuild.

## How to run (all light; read the per-alignment cache, not the share)

Cache of binned per-trial arrays: `state_space_recovery/cache/<cue|cue25|lick|splice>/<label>.npz`
(`state_space_sim.fill_cache`, one share read). Environment switches (outputs carry the suffix):
`WF_SS_ALIGN` (cue default / cue25 / lick / splice), `WF_SS_K` (default 8), `WF_SS_XFORM` (z / gsr / spout).
Use `C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe` from `scripts/`.

| script | what | outputs (under `Widefield/state_space_recovery/`) |
|---|---|---|
| state_space_sim.py | method check + loader / alignments / transforms | sim/, cache/ |
| state_space_recovery_trials.py [--condition target, on_target, landing, contact, incomplete] | core readouts | trials*/ |
| state_space_landing.py | 3-D first-lick join | landing/first_lick_landing.csv |
| state_space_balance_cells.py, state_space_balance_angle.py | balance by cell / matched landing | landing/ |
| state_space_path_tests.py [--trial-alpha] | drift direction, repertoire, trial alpha | path_tests*/ |
| state_space_trajectories.py | 2-D planes, loops (+ bootstrap bands), response size | trajectories*/ |
| state_space_noise.py | noise checks | noise*/ |
| state_space_contact.py | within-session contact contrast | contact*/ |
| state_space_pertrial.py | per-trial plane / order / size (+ recovery-colored order) | pertrial*/ |
| state_space_within_session_behavior.py | within-session behavior slopes | pertrial*/ |
| state_space_step_vs_gradual.py | time models + vs far-contra aim | step_vs_gradual*/ |
| state_space_per_position.py | per-spout x / y vs aim and own landing | per_position*/ |
| state_space_pc2.py | PC2 close-spout diagnostic | pc2_lick/ |
| state_space_page_figs.py [--only ...] | page figures (WF_SS_ALIGN=cue / lick) | page_figs/ |

LMM results for every run: `state_space_recovery/lmm_change_vs_pre.csv`.

## Open / next (ask Priya before the heavy ones)

1. Option 4 -- finer basis: per-trial, time-resolved first-lick extraction in the joint-LocaNMF basis (cue- and
   lick-aligned, spout labels). NOT available: the decoder cache (`_trial_features`, joint basis) is time-averaged;
   `iti_licks/basis_locanmf/sessions` (10-26, pre-rebuild) holds 90 components x 21 lags around ITI licks + a random
   1,500 post-cue licks, no first-lick / spout identity. 11-08: JOB WRITTEN, `scripts/o2_state_space_lnmf.py` (bundle
   `rebuild_20261103/o2/ss_lnmf_20261108`, COMMANDS.md inside). Reuses the rebuild's O2 encoding bundle inputs
   (`enc_rebuild_20261104b/inputs` on scratch: V / ft), rows = the Allen extract's rows timed by the run's
   `cued_first_lick_times.csv`; projection = `joint_locanmf.Basis.project_uv` (new; `project` calls it); U_atlas + pinned
   bases staged share -> scratch on the transfer node. PLUS June 6/6-6/8 (Priya 11-08 "add june where possible ...
   contact-only"): rows = first DAQ contact in cue .. stop of engaged trials, V / ft shipped in the bundle. Output
   `<bundle>/out/sessions/<label>.npz`. 11-08: LOADER SWITCH DONE (326d1930): `WF_SS_BASIS=lnmf` (own cache
   `cache_lnmf`, outputs `*_lnmf`, components `<area>#<k>_<side>`, cols per animal, June = epoch 'june' loaded only by
   `load_sessions(june=True)`; balance_cells / pc2 refuse). June script `scripts/state_space_june_drift.py` written
   (plan = running list C). Next, once the outputs land: re-run the variants; June = the no-stroke drift reference (upper bound: includes learning). (The local input cache does NOT hit
   from this worktree -- its keys include module source -- so desktop behaviour loads read the share, ~2.5 min each.)
2. Close-spout PC2 transient: still unexplained.
3. Matched-landing balance in FINDINGS 0b's first-lick window (0.1-0.5 s, with ALM / tjM1).
4. Power check for the repertoire test (plant a chronic-like subset).
5. No-stroke drift reference from cohort 2's pre period (separate cohort, never pooled).
6. FINDINGS: section 0c added on main 11-08 (Priya "yes add findings to main"; commit 4598ec00, NOT pushed: main also
   held another window's unpushed 4e6630be). Keep 0c in step with this branch.

## Code status: EXPLORATORY, clean up before anything is finalized (Priya 11-07)

Priya: "ensure the code you're writing follows the same principles as the rest of the repo, and isn't redundant; use
existing functions wherever possible. we'll have to clean up things when we finalize them."

Already conforming: shared statistics from `analysis_kit` (`boot_delta`, `boot_ci`, `epoch_change_lmm`), epochs from
`epochs`, colors / markers from `orofacial_cohort`, star placement via `sig_marks.star_top` + `reserve_headroom`, every
figure save through `figure_layout.save_png_svg` (PNG + svg twin; switched 11-07 night -- figures not re-rendered since
then have no svg yet), ruff-clean, outputs only under `MICROSCOPE/Priya/...`.

**STATUS 11-08 (each step verified by re-running every affected script, cue + lick, and diffing all 44 output CSVs
against the pre-refactor snapshot -- identical; tests `tests/test_state_space.py`, 6 tests):**
- DONE 1 Paths: every location in `state_space_sim` (PROFILE / REBUILD from the render profile, ENCODING, XTAG / WTAG
  pinned by a subprocess test against run_mode / orofacial_outputs routing, ITI, ITI_W, LICKS3D via pose3d_dir, ROOT
  via PathResolver, LANDING, CACHE). Not done: writeguard before writes; applying the profile in-process (deliberately
  not: module caches would keep it).
- DONE 2 (partly) One frame: `state_space_recovery_trials.frames` / `xy` used by trajectories, contact, pertrial.
  `path_tests.trial_alpha` (leave-self-out chronic template) and `R.cloud` / `repertoire` still project on their own.
- DONE 3 One loader: `state_space_sim.load_sessions(hook)` in all ten scripts.
- DONE (verified 11-08: 43 CSVs identical, only the known new s_y column differs; 326d1930) 4 One response measure: `state_space_sim.evoked` + `hemi`; definitions
  unchanged (it is NOT 0b's primary marker -- that is the angle-matched gain ratio of `lick_recovery_primary`; the
  angle-matched version here is `state_space_balance_angle`).
- DONE 6 Lag grids + spout order imported (`iti_licks.LAGS_C / LAGS_P`, `orofacial_cohort.POS_ORDER`).
- DONE 8 Tests (binning, transforms, frame axes, loader order, path tags, evoked / hemi, LocaNMF basis switch: 7).
- DONE 7 (51788a0a) The binned-cache fill (`state_space_sim.prefill`) fans over 3 workers via `analysis_kit.fan_sessions`;
  the worker gets cache dir + atlas names in its item (a spawned worker re-imports under the DEFAULT env -- module
  globals there are not the caller's; caught 11-08 before it wrote anywhere, now pinned by the test).
- OPEN 5 (landing join at extraction), 9 (throwaway diagnostics: Priya to decide), 10 (page builder).

Cleanup list (do these when an analysis is promoted from exploratory; each item says what to reuse):

1. **Paths.** `state_space_sim.ITI`, `state_space_step_vs_gradual.AIM`, `state_space_recovery_trials.LANDING`, the
   `ROOT` constants and `KIN` hard-code `M:/...` and the rebuild's `_daqlk`-tagged folder. Resolve through the repo:
   extracts = `iti_licks.XOUT / "sessions_roi8fr"` and tables = `iti_licks.OUT_W` under the right run mode
   (`run_mode.subdir`, `fr_musall.E_BASE`, rebuild routing), 3-D licks via `pose3d.pose3d_dir`, own outputs under a
   `paths.yaml` / `PathResolver` root instead of a literal, and `writeguard.assert_writable` before writes.
2. **One pre -> chronic frame.** Built four times: `state_space_recovery_trials.cloud`, `state_space_path_tests.
   trial_alpha` / `repertoire`, `state_space_trajectories.days_coords`, `state_space_contact.frames`. Keep ONE
   (`frames` returns P, mu and per-spout zp / u / v / L) and have the others call it.
3. **One session loader.** `load_all` exists in `state_space_path_tests` and `state_space_trajectories`, plus inline
   loops in trials / noise / contact / pertrial / balance scripts; move a single `load_sessions()` (sorted, with day)
   into `state_space_sim` next to `load`.
4. **One response-size / balance measure.** `state_space_recovery_trials.balance`, `state_space_balance_cells.sizes`,
   `state_space_balance_angle.collect`, `state_space_trajectories.resp_size`, `state_space_noise.cells`,
   `state_space_pertrial.collect` each compute "response window minus baseline window" norms or hemisphere sizes; one
   helper with the window from `state_space_sim.BAL_BINS`. Also check it against the repo's existing balance /
   response-size definitions (`scripts/lick_recovery_markers.py`, FINDINGS 0b) and reuse those where the question
   is the same.
5. **Landing join.** `state_space_landing.py` matches extract rows to 3-D first licks by spout + timing (99 % exact on
   PS94_0820) because `iti_licks._first_lick_times` reloads `_behaviour` (~65 s / session). For a final version, write
   the trial id into the per-trial extract at extraction time (iti_licks) instead of re-matching afterwards.
6. **Windows / bins.** `state_space_sim.SEGMENTS` / `BINS` / `LAGS_P` re-declare the extract's lag grids
   (`iti_licks.LAGS_C` / `LAGS_P`); import them.
7. **Parallel loops.** Per-session loops are serial (cheap now because they read the local binned cache); route the
   ones that read the share (`fill_cache`, any future extraction) through `analysis_kit.fan_sessions` (rule 6).
8. **Tests.** None yet. Before promotion: pin the default cue binning to the old binning (checked by hand 11-07),
   the transforms (`_xform`), and the frame / alpha definitions on a small synthetic set.
9. **Throwaway diagnostics** (`state_space_pc2.py`, `state_space_within_session_behavior.py`, the k / dimensionality
   checks run inline): fold into the main scripts or delete once their result is recorded; `pc2_trials.csv`'s
   `n_follow` column is a join bug (all 0).
10. **Page builder.** The page is edited by hand + small insert scripts in the scratchpad; `renumber.py` and
    `make_fig_ids.py` are in the repo. If the page becomes routine, add a builder like the imaging page's
    (`docs/results_pages/imaging/page_figs.py`).

Going forward in this arm: reuse `analysis_kit` / `epochs` / `sig_marks` / `figure_layout` / `orofacial_cohort`; no new
copy of a quantity that already exists (check `analysis_kit`, `iti_licks`, `lick_recovery_markers` first); new code
calls the existing state-space helpers (`state_space_sim.load`, `state_space_page_figs.epoch_panel` / `change` /
`lmm_p`) instead of re-implementing them.

## Pitfalls hit this session

- Bash heredocs / sed turned `\n` inside Python string literals into real newlines several times (broken titles).
  Write patch scripts with the Write tool (use `chr(92) + "n"` if a literal backslash-n must be inserted), then lint.
- Artifact publish: `edits` is refused while a file-edit hook is active -> Edit the file, then publish without edits.
  `files` must be a map of published name -> absolute path (the `root` + list form was refused).
- A republish does not upload regenerated PNGs unless they are in `files`.
- The per-trial extract's `first_cue` ends 2.5 s after the cue (`LAGS_C`); a longer cue window needs `iti_licks.py`.
- In the landing join, `side_ml` < 0 = ipsilesional (left).

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
5. Path: acute = a sideways detour (80-95 % of acute - pre orthogonal to pre -> chronic, every variant); subacute
   slides back toward and along the line; trial clouds move as a whole (no search-then-select).
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
   1,500 post-cue licks, no first-lick / spout identity. Needs SVD reads (~107 x 150 MB): Priya OK'd O2; job script
   not written yet (pattern: scripts/o2_encoding.py; reuse `_trial_features` with an injected joint-basis signal).
2. Close-spout PC2 transient: still unexplained.
3. Matched-landing balance in FINDINGS 0b's first-lick window (0.1-0.5 s, with ALM / tjM1).
4. Power check for the repertoire test (plant a chronic-like subset).
5. No-stroke drift reference from cohort 2's pre period (separate cohort, never pooled).
6. FINDINGS (main checkout) not updated with any of this yet -- exploratory; decide with Priya what goes in.

## Code status: EXPLORATORY, clean up before anything is finalized (Priya 11-07)

Priya: "ensure the code you're writing follows the same principles as the rest of the repo, and isn't redundant; use
existing functions wherever possible. we'll have to clean up things when we finalize them."

Already conforming: shared statistics from `analysis_kit` (`boot_delta`, `boot_ci`, `epoch_change_lmm`), epochs from
`epochs`, colors / markers from `orofacial_cohort`, star placement via `sig_marks.star_top` + `reserve_headroom`, every
figure save through `figure_layout.save_png_svg` (PNG + svg twin; switched 11-07 night -- figures not re-rendered since
then have no svg yet), ruff-clean, outputs only under `MICROSCOPE/Priya/...`.

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

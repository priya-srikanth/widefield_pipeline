# FaceRhythm handoff (2026-10-24; afternoon update at the top of section 2) -- START HERE

Supersedes `STATUS_2026-10-22_FACE_RHYTHM_HANDOFF.md` for current state (that file keeps the 10-22 / 10-23 history:
pilot, timing fix, master planning, rank-10 fit). Findings: `docs/FACE_RHYTHM_FINDINGS.md` (status table at the top).
Decisions: `DECISIONS.md` 2026-10-24 (rank 12; names + grouping master_v1; close / far parts). How it runs:
`docs/FACE_RHYTHM_PIPELINE.md`.

Rules still in force: don't run heavy things locally without checking the box (shared; d4's encoder runs); commit only
your own files, staged by name; never delete data; write only under MICROSCOPE/Priya. Document as you go.

## 1. State

**Master = `master_pre_v1`, RANK 12, frozen 10-24 05:49, spec `f4f84d67773df9bc`**
(`M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1/`). Fit on O2 (job 57395128_12):
4 animals x 120 trials (20 / position, even over pre days), pre-cue rows, run / rest between rows, true bin times;
EVR .803, stability .912, leave-one-animal-out factor similarity .82-.90, split half .911.

| on the share | state |
|---|---|
| `master.npz`, `rank12.*`, `labels.json` | done; names in FINDINGS 4a |
| `proj/` 108 | done (full frozen basis for every session; `basis` full_in_sample 32 / full 76) -> **ENCODER** |
| `loso/rank12_*` 31, `proj_loso/` 32 | done (pre sessions on their left-out-day basis; FR scoring only) |
| `splithalf/rank12.json` | done (.911; lick pair .96 / .99; f6 run_onset_jaw .36, f10 prep_jaw .76) |
| `own/` (3 starts) + `own8/` (8 starts, O2 rerun 10-24) | done, 108 each; readouts / matching read `own8/` automatically (`face_rhythm_readouts.own_sub`) |
| `report/` | rank-12 gallery (abs + share, with names), rank sweep, stability, loso tables |
| rank-10 outputs | record kept: `master_r10.npz`, `rank10.*`, `own_r10/`, `report_r10/`, validation `master_pre_v1_r10` |

Local clips (one pre session per animal, 12 factors) `face_rhythm/cam4/factor_clips/master_pre_v1/`; validation
`face_rhythm/cam4/validation/master_pre_v1/` (+ `lick_epochs/`).

**Encoder (d4)**: has proj/ + labels + grouping `fr_groups.master_v1` (lick 2, 4, 5, 7, 8, 11; left 5; right 4; close 7;
far 8; other 0, 1, 3, 6, 9, 10; modes A / B / S / D; side = S - B, distance = D - B). Bundling into
`enc_v36_full_20261024` (107 sessions; raw 12 factors, grouping applied at submit).

## 1b. Results and figure index (10-24 afternoon; FINDINGS 5b-master, 5d, 5e, 6-master, 6b-6d)

All under `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/` unless noted.

| question (Priya) | result | figure(s) / script |
|---|---|---|
| contralesional deficit per lick? | lick_right per right lick .57-.79 of pre chronically, 4 / 4 | `validation/master_pre_v1/lick_epochs/` (`face_rhythm_lick_epochs.py`) |
| peak factor amplitude over recovery (manuscript FR panel) | contra-spout lick_right chronic -.30 [-.40, -.21] 4/4; contra-spout lick_left +1.08 [.35, 1.87] 4/4 (ratio scale); per position: far_R loses most contralesional, close_R gains most ipsilesional | `peak/peak_by_day*.png`, `peak_cohort_by_{day,epoch}*.png`, `peak_side_headline.png`, `peak_position_headline.png` -- LOG2 scale since 10-24 pm (`face_rhythm_peak.py --split none/side/position`) |
| KEY: contralesional licks made with an ipsilesional face movement | factors (above) + raw point movement, lick-aligned, far_R asym pre +.22-.27 -> chronic below pre 4/4 | `asym_aligned/asym_aligned_traces.png` (cue / lick / bout-onset columns), `asym_aligned_by_animal.png` (`face_rhythm_asym_aligned.py`) |
| does the pre vocabulary fit post? (EVR, normalised) | frozen / own ratio vs own pre: acute -.035 [-.045, -.025] 4/4, chronic -.008 (3/4, borderline) | `evr_own_vs_frozen.png` (bottom row pre = 1, +-1 SEM), `evr_epochs.png` (`face_rhythm_evr_epochs.py`) |
| why does own EVR drop too? | less movement vs fixed tracking noise + less stereotyped movement (partial: r with between-trial speed .48-.66 in PS94 / PS95 only); the ratio divides it out | -- |
| factor angles / morphing | NOT consistent across animals (Priya: "no clear consistent changes") -> secondary; lead with amplitudes | `matching/match_by_day.png`, `match_example.png` (`face_rhythm_match.py`) |
| asymmetry two ways (points vs own-fit loadings) | agree acutely; loadings: lick_right moves off the contralesional face | `matching/asymmetry_two_ways.png` |
| per-position face speed maps | whole face moves less acutely at every position; contralesional half more, strongest far_R; chronic overshoot, ipsi half up on R spouts | `position_maps/position_maps_response_{mean,PS9x}.png` (`face_rhythm_position_maps.py`; cue 0..+4 s) |
| lick- / bout-aligned maps + patch statistics | DONE: whole face down per TRIAL acutely, not per LICK; per lick the contralesional half loses more at close_R / far_R, 4/4, every epoch (FINDINGS 5f) | `aligned_maps/aligned_maps_{cue,lick,bout}.png`, `halves_*.png` (`face_rhythm_aligned_maps.py`, needs the cache from `face_rhythm_asym_aligned.py`) |
| close_R asymmetry before bout onset? | carry-over from earlier licks + post-cue preparation + small sustained offset; mirror at close_L (target-directed), FINDINGS 5f | `asym_aligned/bout_split/` (`face_rhythm_bout_split.py`, `--plot-only`) |
| non-task drop = less active? | no: drops within rest (-.99 log2, 4/4) and "other" (-.37, 4/4); running time not consistently down (FINDINGS 6d) | `nontask_state/` (`face_rhythm_nontask_state.py`, `--plot-only`) |
| WHEN is each factor expressed vs the cue? | lick_right on contra-spout trials LATER subacute / chronic (+.40 / +.41 s, 4/4) while the first lick is EARLIER (-.06 s, 4/4): re-timed within the response, not reaction time (FINDINGS 5g) | `peak/timing_by_epoch_{com,t_half}[_side,_position].png`, data/timing_epoch_stats*.csv (`face_rhythm_peak.py`) |
| factor shape, cleaner test | SUBMITTED to O2 (Priya): `fitepoch`, one pooled own fit per animal x epoch + two pre-day halves (noise floor), 11,000 rows each | -> `own_epoch/` on the share; analysis script still to write (angle post vs preA / preB against preA vs preB) |
| pre + post master for pushes? | no: frozen error LOWER at pushes, 27 / 32 cells | `push_check/` (`face_rhythm_push_check.py`); DECISIONS 10-24 |
| task vs non-task, severity | non-task NOT preserved acutely; no ordering by lesion size (n = 4) | `task_vs_nontask.png`, `severity.png` |

## 2. Next, in order

0b. **10-25 afternoon -- UPDATE (newest; read first).**
   * Imaging (page owner 5b, v26+): map drift COMPLETE for contact / LP_licks / DAQ_ENL1s (FINDINGS 3.14); trend LMM
     re-run on d4's final-frame tables -> ceiling vs deficit is DEFINITION-DEPENDENT (contact z -3.9, ttc n.s.; FINDINGS
     3.13); `mixed_models.plot_fits` x-labels fixed. Pages: sidebar breakpoint 640 px, "Where things stand" cites figures,
     window order pre-cue > cue > lick, window > reference > gate nesting (docs/templates/README.md).
   * FR page (owner e3, v20): pooled within-animal twins of every Fig 7 panel (subagents wrote the pooled parts of
     `face_rhythm_match.py`, `face_rhythm_readouts.py --pooled-only`, `face_rhythm_push_check.py --pooled-only`); FR
     pre-cue position decode (`fr_enl_decode.py`, FINDINGS 5.13).
   * FR SHAPE PAIRING CHANGED (DECISIONS 10-25 afternoon, Priya): per-epoch own fits are paired with the master by BEST
     SPATIAL match (`face_rhythm_matching.columns(..., "best")`, `face_rhythm_epoch_shape.py` ASSIGN / MATCH_ON; MIN_COS
     .5 drops unrecovered fits), not Hungarian. Lateral lick factors less lateralised 3 / 4 every epoch; the acute
     central-lick (lick_ongoing_7Hz) shift is WITHDRAWN; lick_far_nose not recovered even pre. Tools: `--gallery`,
     `--candidates ANIMAL GROUP FACTOR`, `--factors ...` (+ `--out`).
   * RUNNING ON O2 (submitted by Priya 10-25): `fr_factors.py fitcohort` (`run_fitcohort.sbatch`, 30 tasks = cohort +
     4 leave-one-animal-out x {pre, acute, subacute, chronic, preA, preB}; 5,500 rows / animal, 8 inits, 16 c / 96 GB /
     12 h). When done (transfer node): `rsync -ah $B/factors/master_pre_v1/own_epoch_cohort/
     $S/factors/master_pre_v1/own_epoch_cohort/` (B / S as in item 1 below; expect 30 files). THEN (code READY 10-25
     ~15:50; share folder still empty then): `python scripts/face_rhythm_epoch_shape.py --own-dir own_epoch_cohort` ->
     `readouts/master_pre_v1/epoch_shape_cohort/` (each tag COHORT / LOAO-<animal> stands where an animal did; its pre
     spout templates = its member animals' pre sessions, `members()`). Read COHORT against its own preA / preB floor,
     LOAO tags as "does any one animal carry it"; the pooled ">= 3 / N" figures then mix COHORT with the LOAOs -- read
     them as LOAO agreement only. Redo the shape figures on the FR page.
   * 10-26 EVENING -- STATE NOW (newest). PAGE FORMAT (Priya: every page like the attempt-encoding / goal page;
     rule in docs/templates/README.md): N.k figure numbers, a title above each image, captions with the full method.
     (1) FR page v33 (https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1; source scratchpad `fr_page/`): `toc_patch.py`
     numbers figures N.k (ids fig<N>-<k>), adds `ftitles.py` FT titles + a Model / Input / Baseline line by figure
     family + the figure's saved METHODS (from the readout READMEs, `../fr_fmt/readme_methods.json`, regenerate with
     `../fr_fmt/readmes.py`) + script; in-text refs are `<a href="#src:fig/x.png">@fig/x.png@</a>` placeholders resolved
     at build; `rebuild.py` remaps status links. Four page figures were older renders than the share (same numbers) and
     were refreshed. Stale prose fixed (lick_left "recovers" -> beyond the null subacute + chronic). (2) KINEMATICS page
     v7 (https://claude.ai/artifact/DKcKRJKbhBoratxxRc6qZv) -- taken over from 5b / c67b with their OK: source now this
     window's scratchpad `kin_page/` (`src_v6.html` = live v6, `convert_fmt.py` -> `index.html`; edit index.html or
     src + rerun). Sections renumbered 1-10 (were 1, 1b, 2, 2b ...), 48 figures N.k, titles, Data / Window / Stats /
     Script per caption. (3) Imaging page v27 (https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV) done by 5b / c67b
     (old -> new map: c67b scratchpad `imaging_fig_map.csv`). Goal page: d4 already in this format. (4) PI DECK
     (M:/.../Widefield/presentations/PI_overview_kinematics_FR_encoder_2026-10-26.pptx): speaker-note figure citations
     moved to the new N.k numbers (47 edits; `_source_PI_overview_2026-10-26/edit9.py`, backup
     `backup_before_fignum_notes.pptx`). Encoding / imaging / kinematics resolved by image file against each page's
     pre-renumber source; FR by topic (the deck predates two FR reorders, so its letters match no saved FR version).
     Stale FR null notes (slides 39, 40, 59 "null running") replaced by the final 5.10 result. Slides unchanged.
   * 10-27. NUMBERING AUDIT (asked by 07): FR + kinematics pages checked by script (number label = anchor = caption
     header; no title / alt starting with a number; link text = target; refs exist). Kinematics clean. FR: "as Fig 8d"
     (pre-renumber; = Fig 8.1) in the session_shape README text -> now names the map file; FR page v34. FINDINGS FR
     block: 6 stale FR page refs -> N.k (8eaca89). GOAL PAGE REORDER (07, v129: Key -> Supporting -> Supplementary,
     numbers changed; old -> new `wf_attempt/_goal_page/fig_map_20261027_reorder.csv`): PI deck notes remapped, 28
     citations, each now "<number> [#f-<fid>]" (stable anchor, registry `_goal_page/fig_ids.csv`) -- `edit10.py`,
     backup `backup_before_goal_reorder_notes.pptx`; slide 52 now also cites the figure it shows (2.18). OPEN with
     Priya: slide 4 bullets say within-session refit but the figure is imaging Fig 1.1 FROZEN decoder; slides 4-5 notes
     say "pooled session means", page says trial-pooled accuracy; provenance (alt text + notes FIGURE SOURCE + csv;
     visible tag?) not yet applied; 07's decoder figures NOT final (will message).
     AFTER ANY GOAL-PAGE REORDER run `_source_PI_overview_2026-10-26/refresh_goal_numbers.py` (re-runnable; rewrites
     every "<n> [#f-<fid>]" from `_goal_page/fig_ids.csv`, `--dry-run` to preview). v130 (+5 section-3 figs): slide 54
     decoder_retention_pooled 3.19 -> 3.21, nothing else cited moved.
     PERMANENT FIGURE IDS (Priya 10-27, as the goal page): FR page v35 + kinematics page v9 give every figure
     id="f-<image name>" (old "#figN-K" anchors kept inside each figure), in-page links use the f- ids, registry
     fig_ids.csv (fid, number, title, src). Tool: `scripts/page_stable_ids.py <index.html>` (idempotent; run after any
     page build -- FR rebuild.py now calls it). PAGE SOURCES MOVED OFF SCRATCH to the share:
     `Widefield/fr_page_src/{fr_page,fr_fmt}` (build: cd fr_page; python rebuild.py) and `Widefield/kin_page_src/kin_page`.
     DECK PROVENANCE (Priya: source tag in the notes): `edit11.py` (re-runnable) adds a "FIGURE SOURCE:" line per
     picture at the end of the notes (page + current number + [#f-id] + link, share original, deck image, crop /
     composite), the same text as alt text, shape name fig:<id>, and `figure_provenance.csv`; 49 pictures / videos,
     46 slides; backup `backup_before_provenance.pptx`. Deck build inputs copied to `_source.../build_inputs/`
     (crops, enc, fr, img, kin, vid; 191 MB). Imaging numbers from 5b's registry `docs/results_pages/imaging/imaging_fig_ids.csv`
     (kept current by the imaging page chain). Not on any page: slides 8-10 (RESTW maps), 32-34 (FR set-up).
     SLIDES 4-6 (Priya "yes", 10-27; `edit12.py`, backup `backup_before_slide4_refit.pptx`): slide 4 = the REFIT
     within-session decoder (native per-animal table from imaging section 1; was the frozen Fig 1.1 beside refit
     bullets); slide 5 notes: bars = trial-pooled accuracy, frozen decoder trained on ALL pre (June + August), not
     August only; slide 6 adds imaging Fig 4.2 (refit - frozen gap minus pre gap, by spout; title block cropped) beside
     Fig 13.1 + a by-spout bullet. Checked by a PowerPoint render. The IMAGING PAGE section 1 has the same mismatch
     (refit "take" text beside the frozen Fig 1.1) -- flagged to 5b, fixed in imaging v33.
     IMAGING v33 (5b): section 1 = refit only (new 1.1 / 1.2 refit by position, slow band 1.3), frozen pre-cue -> 2.1,
     old 2.1-2.3 -> 2.2-2.4; slide 4's table is a SEPARATE refit (Allen-ROI, engaged, August-only pre), acute sessions
     5 / 4 / 6 / 1 (was 4 / 4 / 4 / 1). `edit13.py` fixed the slide 4 / 5 notes; `refresh_goal_numbers.py` now reads
     ALL four page registries (ambiguous fids resolved by the page named before the citation) -- run it, then
     edit11.py, after any page reorder.
   * 10-26 AFTERNOON. PERMUTATION NULL SCORED: all 600 / 600 files on the share (Priya copied the
     last 4 chronic halves late afternoon) = 100 permutations per epoch; rescored, unchanged except chronic lick_right
     142 map points (was 144 at n = 96). `python scripts/face_rhythm_cohort_null.py` (19 s). lick_right beyond the null at EVERY epoch on
     all four scalars (Holm); lick_left inside it acutely, beyond it subacute + chronic -> the lick_left cohort vs
     per-session disagreement is SETTLED (changed after acute, smaller than acute). Other ten: none Holm, a RESOLUTION
     limit (min p .0099 at n ~100, Holm over ten needs < .005) -> >= 500 permutations if they matter. Maps now coloured
     by t (change / null SD), not raw change: low-loading points that change reliably drew as white. FINDINGS 5.10
     "COHORT PERMUTATION NULL" rewritten; FR page v31: Figs 7m-7o (scalars / maps / spectral), status rows updated,
     pending = >= 500 permutations. Still open: FR spout-in-mesh control; 13b with d4.
   * 10-26 MORNING. (1) PI OVERVIEW DECK (Priya, for a 30 min PI meeting): `M:/MICROSCOPE/Priya/Widefield/
     presentations/PI_overview_kinematics_FR_encoder_2026-10-26.pptx`, 59 slides (4 hidden backups). Sections 1 cortical
     spout coding (decoders + RESTW position maps: ENL 1 s from grant_figures/DAQ_ENL1s, cue / lick LP-gated from
     grant_figures/LP_licks), 2 kinematics (pose methods, 3-D clips re-encoded H.264, softmax landing / trajectory
     confusion), 3 FaceRhythm, 4 kinematics + encoder; main point on each slide, methods + data in the speaker notes.
     PRIYA EDITS THE DECK HERSELF: build scripts + every pre-edit backup in `presentations/_source_PI_overview_2026-10-26/`
     (`build.py` first build, `edit_maps.py` ... `edit8.py` = passes on her edited copy; each pass snapshots her file and
     overwrites only if unchanged). Style rules in memory `slide-deck-style`. (2) FR page v30: Fig 7l = drift floor. (3)
     Permutation null still running on O2: 81 / 600 done at last check (80 on the share, acute only); the two CANCELLED
     entries are the original submissions (0 s), nothing to resubmit. When more lands: rsync own_epoch_cohort_null/
     (command in the entry below), rerun `python scripts/face_rhythm_cohort_null.py`, add cohort_null_*.png to the FR page
     next to 7l, update FINDINGS 5.10 "COHORT PERMUTATION NULL". (4) `kinematics_pooled_figs.py` gained fig_paths3d(shared=
     True) -> pose3d_paths_pooled_shared.png (one axis box per view across spouts; 1cde99a).
   * 10-25 PAST MIDNIGHT. Drift fits back (10 / 10) and scored: `scripts/face_rhythm_drift_floor.py`
     (drift 5-7 deg; lick_right 22-26 deg beyond it, 5 / 5 fits; FINDINGS 5.10 "DRIFT FLOOR"). Null scorer WRITTEN:
     `scripts/face_rhythm_cohort_null.py` (templates per permutation from pseudo-pre keys, max-|t| maps / spectra, Holm
     over the lick pair + the other ten; ~10 s, `--plot-only`). Only 50 / 600 null files were on the share (acute p000-
     p026, 25 pairs) -> copy back the rest (transfer node: `rsync -ah $B/factors/master_pre_v1/own_epoch_cohort_null/
     $S/factors/master_pre_v1/own_epoch_cohort_null/`), rerun the scorer, then put drift_floor.png + cohort_null_*.png on
     the FR page (section 7, after the cohort fits) and update FINDINGS 5.10 "COHORT PERMUTATION NULL" (now PARTIAL).
   * 10-25 LATE NIGHT. FR page v29 (owner: this window; source scratchpad `fr_page/`:
     edit `index_reordered.html` + `titles.py`, then `python rebuild.py` -- rebuilds via `toc_patch.py` and REMAPS the
     "Where things stand" links by image src, so figures can move freely; write a new link as `href="#src:fig/x.png"`).
     HEADLINE (Priya) = Fig 7: per-session per-point lick-factor loading change / each animal's own pre-session SD, FDR;
     Fig 7b its spectral twin (SEM + significance strips); 7c side-specificity; 7e / 7f the same tests for the other ten
     factors; 7g-7u cohort fits; section 8 per-animal fits, by-day matching now BEST SPATIAL (`face_rhythm_match.py --on
     spatial --assign best` -> matching_spatial_vector_best/). The whole page pairs factors by best spatial match. Page
     rules added (docs/templates/README.md + memory): pooled / cohort BEFORE per-animal in a section, headline first;
     change maps RdBu_r, one colour scale per figure, significant points large + outlined, rest faded.
     `face_rhythm_session_shape.py` now: all 12 factors (`ok` = spatial cos >= .5), loading + spectral, raw and own-SD
     ("z") scales, `--plot-only`. RUNNING ON O2 (Priya submitted 10-25 night): null `frf_null` 1-600 and drift
     `fitcohort` 31-40 -> copy back own_epoch_cohort/ and own_epoch_cohort_null/ (commands in item 0b below), then WRITE
     the null scorer: per permutation, rebuild spout templates from that permutation's pseudo-pre keys; max-statistic over
     mesh points for the maps; compare observed COHORT_<epoch> - COHORT_pre. Then the drift floor (preEarly vs preLate)
     against lick_right's 23-27 deg. Open: lick_left after acute disagrees between cohort (recovered) and per-session /
     per-animal fits (still changed) -- the null + drift decide; FR spout-in-mesh control; 13b with d4.
   * 10-25 NIGHT: shape STATISTICS. (1) Per-session version DONE (`scripts/face_rhythm_session_shape.py`, FINDINGS 5.10
     "PER-SESSION STATS", FR page v24 section 7): lick_left loses side-specificity 4 / 4 at every epoch (Holm .002-.016);
     lick_right 3 / 4, not Holm. (2) ON O2 (Priya to submit; code deployed to the share): permutation null
     `run_fitcohortnull.sbatch` (600 fits = 3 epochs x 100 perms x pseudo-pre / pseudo-post; test task 2 first, then
     `--array=1,3-600%60`) -> own_epoch_cohort_null/; DRIFT fits `run_fitcohort.sbatch --array=31-40` (COHORT / LOAO x
     preEarly / preLate) -> own_epoch_cohort/. SCORING THE NULL (to write): rebuild spout templates from each
     permutation's pseudo-pre keys (saved in the npz) -- the real pre fit shares sessions with the real templates, so a
     fixed template makes the null anti-conservative for Fig 7; max-statistic over mesh points for Fig 7b; the null is
     within-animal (fixed effects) and includes drift. Cohort Fig 7d / 7e: pooled line +- jackknife SE over animals.
   * DONE 10-25 evening: fitcohort back (30 / 30; tasks 4, 23, 28 finished after the first rsync) and run
     (`--own-dir own_epoch_cohort`, + `--gallery`) -> FINDINGS 5.10 "COHORT FITS": lick_right stays changed through
     chronic in the cohort and all 4 LOAO fits, lick_left recovers. FR page v21: new section 7 (cohort), 8 (per-animal
     shape), 9 (secondary); page source scratchpad `fr_page/index_reordered.html` + `toc_patch.py` (sections dict now
     shape / shapeanimal / secondary). NEXT for shape: the early-vs-late-pre DRIFT floor (cohort preA / preB interleave
     days, floor 3-8 deg, too lenient).
   * DONE 10-25 ~16:40: cortical ENL decode with FR factors regressed out (`scripts/enl_fr_removed_decode.py`, local,
     107 sessions, ~25 min at 6 jobs) -> 70-100 % of the cortical pre-cue code survives, residual tracks raw r .91-.97;
     NOT face posture as FR sees it (FINDINGS 5.14, FACE_RHYTHM_FINDINGS 7b). Not yet on either results page.
   * Still open: FR spout-in-mesh control; 13b (dimension comparison + composite deficit; agree the list with d4).

0a. **10-25 ~08:30 -- UPDATE (supersedes the OPEN list's item a and the cd_loo_pooled line below).**
   * cd_loo_pooled switched to DIFFERENCE from each animal's own held-out pre (the ratio blew up where pre ~ 0); on the
     imaging page as 9f-9k (5b). Map drift (`map_drift_prepre.py`) COMPLETE for contact / LP_licks / DAQ_ENL1s (output
     folders are the writeguard tag names, not the arm names); results FINDINGS 3.14, on the imaging page section 7
     (v20, nested window > reference > gate, every title "window . reference . gate"). Headline far-contra losses are
     beyond drift everywhere; cue-window far-middle acute is NOT separable from drift; chronic far-middle PRE-CUE loss
     IS beyond drift (new).
   * FR: `scripts/fr_enl_decode.py` -- face (FR factors) decodes spout position in the lick-free pre-cue window (2 s
     .43, .34 without the last 0.5 s; chance .17); acute far-contra recall to chance. FINDINGS 5.13, FACE_RHYTHM_FINDINGS
     7, FR page section 6. NEXT here: cortical ENL decode with FR factors regressed out (is the cortical pre-cue code
     face posture?), and a spout-in-mesh control for FR.
   * Pages: Priya's standing rules (docs/templates/README.md + memory): windows always PRE-CUE, CUE, LICK; like figures
     adjacent, nested window > reference > gate; collapsible sidebar per figure group. FR page (owner e3,
     X8NDdQNHi3zMP77PBrZAC1, v12) built from `scratchpad/fr_page/index_reordered.html` + `toc_patch.py` (titles in
     `titles.py`) -- that scratch dir is session-local; the published page is the source of truth (Artifact read).
   * Still open from below: b (13b dimensions + composite), c onward.

0. **10-25 LATE -- STATE NOW (before compaction): this window also took IMAGING-page work (Priya; page owner = 5b,
   https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV, now v5; 5b republishes from numbers we send).** Done + on main:
   * Items 1 / 3 / 4 of the page's pending list (DECISIONS 2026-10-25 "Imaging page pending items"): decoder trajectory
     re-rendered (October days 46 / 50 restored), chronic_stability + chronic_drift_vs_reorg re-run, session LMMs re-run
     on the rebuilt trial_attempts (ttc + contact), per-animal template / state tables (`scripts/imaging_per_animal_tables.py`;
     state drop = PS94 alone; template amplitude returns 2 / 4). 5b integrated all of it (page sections 3, 4, 6, 10, 13).
   * `scripts/cd_loo_pooled.py` (Priya: "within-animal normalized pooled version of 9c-e"): each animal / its held-out
     pre, pooled over animals, boot_delta per epoch -> labcams/grant_figures/epoch/cd_loo/cd_loo_pooled_{window,traj}_
     {precue,cue,lick}.png + cd_loo_pooled_stats.csv. far_R: pre-cue subacute -.39* (4 / 4 below), chronic -.25 (4 / 4,
     CI touches 0); cue subacute -.18* (4 / 4); lick no consistent change. NOT YET SENT to 5b for the page.
   OPEN, in this order:
   a. `scripts/map_drift_prepre.py` (Priya: a late-pre vs early-pre version of figure 15r to compare map drift):
      June pre vs August pre position maps, LATE - EARLY beside acute / chronic - late pre, same pooling + test as 15r.
      Written + lint-clean; a TEST run (contact arm, cue, rest -> scratchpad imaging/drift_test) was still collecting maps
      at compaction (cold cache?). When it works: run `--arm contact`, `--arm lp`, `--arm enl1s` (refs rest + precue,
      aligns cue / lick / precue) -> labcams/grant_figures/map_drift_prepre/<arm>/; send figures + drift_vs_stroke_panels
      .csv to 5b. Answered Priya: the existing 15r family already has pooled within-animal pre-normalised maps for REST /
      PRECUE refs x cue / contact-lick / ENL in grant_figures/epoch (contact), LP_licks (LP gate), DAQ_ENL1s (1 s ENL);
      the page shows only REST cue + LP lick -> ask 5b to add the PRECUE ref, contact lick and ENL panels.
   b. 13b "compare dimensions" + "composite deficit score" (Priya chose BOTH): replace the single ttc deficit with
      session-level behaviour dimensions -- attempt rate / completion (trial_attempts_sessions.csv), aim (cam4 ipsi_bias /
      angle_gain_far from cohort/tables/prepost/session_metrics.csv; 3-D far_R midline error, 5b's pose3d tables), jaw
      steering (FAR_DIR), FaceRhythm (lick_right per right lick: readouts/master_pre_v1 lick_epochs data; EVR ratio
      evr_epochs data; contra-ipsi asymmetry asym_aligned data) -- each per-animal-normalised vs own August pre. (1) one
      LMM per dimension on the SAME sessions + variance partitioning (which dimension explains the decoder / template
      deficit beyond the others); (2) composite = PC1 of the standardised dimensions, as a new `mixed_models.DEFICITS`
      entry. Agree the variable list with d4 first (goal-question overlap). Inputs: imaging_lmm decoder / epoch tables.
   c. 13c drift: Priya noted pre points cluster -- explained: family 11 pre = 8 August sessions (~10-day span), June not
      included, so it cannot show 2-month drift. Proposed: a template June -> August arm (template from June, scored on
      August), like 5b's june_august_drift. Not started; same blind spot in the CD loo family (August pre only).
   d. Ask 5b: add the refit-deficit definition to the Fig 13 caption (refit deficit = pre_refit - refit = F - G;
      explained to Priya in chat).
0. **10-25 -- clean-up + findings.** Current conclusions now in docs/FINDINGS.md (section 5 = FaceRhythm; all windows
   agreed one file). Results page v8: figures open in a zoomable viewer. DELETED (Priya-approved 10-25, derived only,
   writeguard): o2/fr_cam4_20261021/out_test/ the 10 unchosen tracking parameter sets (kept PS95_20260812_cam4_midlk12.npz,
   the chosen set; decision DECISIONS 10-21), face_rhythm/cam4/register_v1..v4_*_20261021 (superseded registrations;
   register/ is current), and a local pilot scratch (fr/pilot: a 20k-frame clip of PS94 0810 cam4 -- the original video is
   on MICROSCOPE -- plus its points / spec). 9.6 GB. KEPT: o2/fr_cam4_20261021/out/ (2.2 TB, the ONLY copy of the FR VQT
   outputs: trial windows 25 Hz, whole session 5 Hz, points 25 Hz; whole-session 25 Hz VQT and 250 fps tracks were never
   saved), the pilot factor set (encoder FR run A used it), param_test figures.
0. **10-24 NIGHT -- STATE NOW (before compaction).** Everything below committed; results page v7
   https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1 (source + figure copies in the session scratchpad `fr_page/`;
   from a new session: Artifact read the URL, then publish with `url`). Since the late-evening entry:
   * fitepoch DONE (24 fits) -> `face_rhythm_epoch_shape.py` (FINDINGS 6c-pooled): lick factors change shape beyond the
     pre-half floor 4 / 4; loading DIFFERENCE MAPS (loading leaves the contralesional face near the midline -> mouth /
     chin rim + ipsilesional cheek, PS92 / 94 / 95) and SPOUT TEMPLATES (post lick_right still R-like but less
     side-specific; not a re-aimed lick except PS92 subacute). Contra - ipsi index called too coarse by Priya.
   * Per-POINT face maps replace the square patches (Priya): `analysis_kit.boot_delta_columns` (boot_delta's draw on
     all points at once, pinned per column) + `bh` FDR; aligned_maps `--unit point` (default) and position (speed)
     maps both; agreement threshold >= 3 of 4 animals (Priya; DECISIONS 10-24 late); PS93 flagged as a possible outlier
     (pre-existing right orofacial deficit).
   * Display order of per-spout rows = close ipsi / centre / contra, far ipsi / centre / contra
     (`common.POSITIONS_DISPLAY`); "response speed" = movement speed of each face point in cue 0..+4 s (titles say so).
   * NEXT for the results page (Priya's conventions via the encoding window, 10-24): clickable hidable left-panel table
     of contents; every figure numbered (Fig 1, 2b, ...) in caption and contents; each section tagged neural vs
     behaviour (FR = behaviour); captions say how to read the figure + headline numbers; per-animal results first, then
     within-animal-normalised pooled results with stats. Template: docs/templates/ (copy its CSS, #tocbtn / #toc markup
     and bottom script); example https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d. Page is at v7.
   * Open: early-vs-late pre-day halves as a stricter shape floor (8 O2 tasks, not set up); per-bout timing of
     lick_right; per_lick_by_epoch.png keeps old line styles until a full `lick_epochs` run.
   **Code clean-up status:** steps 1-4 of 2c DONE and verified byte-identical against a baseline run of the old code
   (every readout table: peak x 3, asym, bout split, readouts, EVR, position maps, matching, push check, aligned maps).
   **Step 5 DONE 10-25** (all O2 jobs back): the factor stages are the package `wfield_local/face_rhythm/factors/`
   (config / files / cp / sample / fit / project / validate / o2 / report / cli; every definition moved verbatim by a
   script, imports generated, two circular imports resolved by placing `session_group` with the row sampling);
   `scripts/face_rhythm_factors.py` is a launcher that forwards the old attribute names (tests, `o2_face_rhythm`
   tracking QC, clips / validate scripts unchanged in behaviour). O2: `o2.deploy_code(sd)` (called by `plan`) writes
   `fr_factors.py` + `fr_factors_lib/` + sidecars into the set; the launcher prefers the copy beside it. Verified:
   every stage (build, fit + LOAO, splithalf, loso, select, project, project --loso-basis, fitsession, fitepoch, report)
   on the synthetic bundle, old script vs package: all 47 output files identical except the run-time fields
   (`minutes`, master `frozen` stamp), which differ between two runs of the old script too; 28 FR tests pass incl. a
   new deployed-set test. The share's master_pre_v1 still holds the old single-file `fr_factors.py` (still runs); the
   next code change there goes through `deploy_code`.
   Earlier late-evening entry:
0. **10-24 late evening -- STATE NOW.** (i) STATISTICS on every claim + `*` / `**` marks (Holm over the family each
   figure tests) and the orofacial epoch palette (pink / gold / teal) on all FR figures (DECISIONS 10-24 "FaceRhythm
   figures: epoch colours and significance marks"); FINDINGS 5b-master / 5e / 5f / 5g / survey carry the numbers;
   results page REPUBLISHED (version 2). Corrections: close_R pre-onset "sustained offset" n.s.; close_L mirror only a
   trend. (ii) CLEAN-UP steps 1-3 of 2c DONE and on main (0234cb6, 9faf322; verified byte-identical against a
   baseline run of the old code: peak x 3 splits, asym maps cache + epochs table, bout split windows); step 1b (the
   remaining scripts onto `face_rhythm.common`) verifying at handoff (`fr_verify2.sh` in the session scratchpad:
   readouts / evr / position maps / match / push / bout tables vs baseline). Still to do: step 4 (compute / plot split
   for readouts, aligned_maps, match), step 5 (split `face_rhythm_factors.py`, after fitepoch). (iii) fitepoch DONE (24 fits on the share, `own_epoch/`) and
   `scripts/face_rhythm_epoch_shape.py` run (FINDINGS 6c-pooled): lick factors change shape beyond the pre-half floor
   in 4 / 4 at every epoch; both drift toward a symmetric face in 3 / 4 (PS93 opposite). Results page v3. Optional next:
   early-vs-late pre-day halves as a stricter floor (8 O2 tasks; `EPOCH_GROUPS` in face_rhythm_factors.py).
   Older 10-24 evening notes follow.
   Story order DECIDED (Priya; FINDINGS top, DECISIONS 10-24 "FaceRhythm framing"): acute R > L
   lick-factor drop -> persistent lick_right deficit (several methods) -> contra / ipsi face asymmetry -> pre-factor EVR
   drop -> survey of all factors (peak + timing); angles secondary. 5f wording CORRECTED (not "fewer licks, not smaller":
   DLC shows slower / shorter acute licks; the mesh-speed window is insensitive to that). (a) bout split DONE, (c2)
   non-task by state DONE, cue timing DONE (FINDINGS 5g; next could be per-bout timing of lick_right). ON O2: `fitepoch`
   (24 tasks, `run_fitepoch.sbatch`; SUBMITTED by Priya 10-24 evening) -> rsync `own_epoch/` back, then write the angle comparison (post vs preA / preB
   against preA vs preB, per factor; `face_rhythm_matching.match`). FR RESULTS PAGE PUBLISHED 10-24: https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1 (private; to update from a new
   session, read it then publish with `url`; figure copies were in the session scratchpad -- re-copy from `readouts/`).
   Done: an FR RESULTS ARTIFACT PAGE (Priya via the encoding window: every result family as a published
   Artifact -- per-animal figures + maps, one-line takeaway per figure, sections keyed to FINDINGS; example
   https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d), then CODE + DOC CLEAN-UP (section 2c).
1. (done) **Rsync `own/`** once `frf_own` is done (transfer node; RESET the variables -- the encoding bundle reuses B / S):
   `B=/n/scratch/users/p/ps150/o2/fr_cam4_20261021; S=/n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/fr_cam4_20261021`
   `rsync -ah --exclude fit_tensor.npy --exclude 'proj_r10' --exclude 'proj_loso_r10' $B/factors/master_pre_v1/ $S/factors/master_pre_v1/`
   -> expect `own/` 108. (The excludes keep the rank-10 projections deleted from the share on 10-24 from coming back.)
2. Run on all four animals (fr env; `export PYTHONPATH=<repo>`; push check needs the locanmf env for behaviour):
   * `python scripts/face_rhythm_readouts.py --set master_pre_v1` -> movement maps, amplitudes, EVR gap, recovery by
     day, task vs non-task, severity (`face_rhythm/cam4/readouts/master_pre_v1/`)
   * `python scripts/face_rhythm_match.py --set master_pre_v1` -> factor angles vs master, pre floor, chance (`matching/`)
   * `python scripts/face_rhythm_push_check.py --set master_pre_v1 --jobs 4` -> recon error at treadmill pushes
     (`push_check/`); decides whether a pre + post fit is needed (see 3). Test on 2 PS95 sessions was still loading
     behaviour at handoff time -- check `scratchpad` output / run one session first.
3. Redo FINDINGS 5 / 6 on four animals; add the matching, push, task / non-task and severity results.
4. Other cameras (cam1 / cam2 / cam3): await Priya's landmark clicks -> register -> ROIs -> O2 -> multi-view set.

## 2b. Clean-up done 10-24 (Priya OK)

Deleted from the share (all DERIVED, rebuildable; `writeguard.assert_deletable(derived=True, approved=True)`), ~14.7 GB:
`master_pre_v1/fit_tensor.npy` (STALE: the earlier sweep sample's tensor, 27,392 rows, no longer matching fit_index's
27,448; the live tensor is on O2 scratch, `build` rebuilds it), `master_pre_v1/proj_r10/` + `proj_loso_r10/` (rank-10
projections; `master_r10.npz` kept), `face_rhythm/cam4/factor_clips/master_pre_v1_r10/` (rank-12 clips complete: 48 =
12 factors x 4 animals), `factors/preview_r10_PS92/` + `face_rhythm/cam4/preview_r10_PS92/` (single-session preview,
superseded). Kept: `register_v1..v4_20261021/` (1.7 GB, comparison sets) and `param_test/` (1.1 GB) pending Priya; O2
scratch left to purge itself. JupyterLab (ROI notebook) stopped -- restart for the cam1-3 landmark clicks. Other
windows asked to prune their stale git worktrees.

## 2c. TO DO: make the FaceRhythm code modular and easy to use (Priya 10-24)

"Clean up the code to make it modular and easy to use, as a Claude engineer would organise it for a systems
neuroscientist to use." Not mid-analysis (after the four-animal readouts), and refactors must carry the reasoning in
the comments forward (CLAUDE.md engineering note). Starting points:
* `scripts/face_rhythm_factors.py` (~1,480 lines) mixes the CP fit, planning / sampling, O2 sbatch generation,
  projection, LOSO / split half, and the report. Split into `wfield_local/face_rhythm/` modules (fit, sample, project,
  validate, report, o2) with a thin CLI; keep the O2 single-file copy working (`plan` copies the sidecars today).
* One entry point per question a user asks ("fit a factor set", "project sessions", "name / validate", "readouts",
  "match fits"), each with a short docstring stating inputs, outputs and where they land; one config block (set name,
  bundle, rank, windows) instead of constants spread over modules.
* A worked notebook / README walk-through: from FR outputs to a named factor set to the encoder regressors, with the
  pitfalls (in-sample pre, absolute vs share, factor numbers shift between fits).
* Tests per module (the existing ones in tests/test_face_rhythm_*.py are the base).
* SEPARATE COMPUTE FROM PLOT (Priya 10-24: "can figures re-render without redoing calculations?"): every heavy script
  saves what its figures need and takes ``--plot-only``. Done: `face_rhythm_push_check.py` (data/push_check_traces.npz).
  To do: `face_rhythm_readouts.py` (~30 min recompute), `face_rhythm_asym_aligned.py` (traces; maps already cached),
  `face_rhythm_aligned_maps.py` (patch / halves statistics + mean maps), `face_rhythm_match.py` (chance shuffles).

**Progress (10-24 late evening):** 1 DONE (`common`, `face`, `licks`, `stats`; epoch panel helper not needed yet --
`stats.tick_label` / `mark` carry the statistics; `face_rhythm_readouts._tasks / _label / _day` now delegate to
`common`), 2 DONE (`python -m wfield_local.face_rhythm list | show | run`, a test checks every entry), 3 DONE
(`docs/FACE_RHYTHM_GUIDE.md`, linked from PIPELINE). Compute / plot split DONE for peak (`--plot-only`), asym
(`--plot-only`), per-lick (`--stats-only`), bout split, non-task, push check. Verification record: scratchpad
`verify/baseline` (old code, 047c09d) vs `verify/after*`; `boot_delta(draws=True)` leaves every interval unchanged
(pinned tests).

**Plan adopted 10-24 evening (Priya: "once we submit these jobs, please work on cleaning up the facerhythm code and
documentation").** Measured: ~8,000 lines of FR code; `EPOCHS` defined in 8+ files, `POSITIONS` in two different orders
(peak: contra first; the rest: animal-left first), the point-speed reconstruction, rule-8 per-animal table + `boot_delta`
epoch statistics and the epoch panel re-written per script. On a branch (`fr-cleanup`, own worktree), each step
verified by re-running the affected scripts into a scratch ``--out`` and diffing the CSVs against the current outputs
(a refactor must not move a number):
1. `wfield_local/face_rhythm/` package for the ANALYSIS layer: `common` (epochs, positions in ONE canonical order + a
   contra-first display order, sides, frame rate, set / bundle defaults, palettes, session iteration from `proj/`),
   `face` (point-speed reconstruction from the frozen point PCs, event alignment, window weights), `licks` (lick table,
   bout classes, first-lick latency), `stats` (log2 vs pre, per-animal epoch table + `boot_delta`, printed rule-8
   first), `figures` (epoch panel). Existing flat modules that other code or O2 imports BY NAME stay where they are
   (`face_rhythm_outputs`, `fr_groups` -- encoder; `face_rhythm_timing`, `face_rhythm_multiview` -- O2 sidecars);
   moved modules leave a one-line re-export so nothing breaks.
2. One catalogue CLI: `python -m wfield_local.face_rhythm list` (each analysis: the question it answers, inputs,
   outputs, runtime) and `... run <analysis> [--plot-only]`.
3. A user guide `docs/FACE_RHYTHM_GUIDE.md`: concepts (frozen master, own fits, absolute vs share, log2 vs pre,
   image sides), run order from FR outputs to figures and encoder regressors, the analysis catalogue (question ->
   command -> figure -> FINDINGS section), pitfalls. PIPELINE (O2) / FINDINGS / handoffs keep their roles.
4. Compute / plot split for the remaining heavy scripts (list above).
5. DONE 10-25: `face_rhythm_factors.py` split into `wfield_local/face_rhythm/factors/` (see item 0, NIGHT entry,
   clean-up status); O2 copy via `o2.deploy_code`.

## 3. Open questions

* **Pre + post master?** (Priya 10-24, for the encoder's treadmill "pushes"). Not yet: the pre-only basis makes the EVR
  gap / reconstruction error the readout of new movement. Decide from `face_rhythm_push_check.py`: if the frozen basis's
  error rises at pushes chronically (not pre, not at random times), fit pre + post (or a post-only fit matched to the
  master with `face_rhythm_match.py`; unmatched factors = the new movements). If not, pushes stay a treadmill regressor.
* Rank-12 weak factors: 6 run_onset_jaw (split half .36) and 10 prep_jaw (.76) -- grouped in "other", never read singly.
* PS93: weakest side selectivity in both lick factors; lick sum not position-balanced (far_L .79, close ~1.15).

## 4. Code added 10-23 / 10-24 (all committed)

* 10-24 evening: `face_rhythm_factors.py fitepoch` (+ `epoch_fit_tasks`, `EPOCH_GROUPS`; share `run_fitepoch.sbatch`),
  `face_rhythm_bout_split.py`, `face_rhythm_nontask_state.py`, `face_rhythm_peak.py` timing (centre of mass, half-rise,
  first-lick latency; `timing()`), all with cached data + `--plot-only` where they recompute.
* 10-24 afternoon: `scripts/face_rhythm_peak.py` (peak amplitude, cohort views, boot_delta, log2, side / position
  splits, headline figures), `face_rhythm_evr_epochs.py` (normalised EVR ratio by epoch), `face_rhythm_position_maps.py`
  (per-position speed maps), `face_rhythm_asym_aligned.py` (cue / lick / bout-onset aligned contra vs ipsi speed + cached
  per-point maps), `face_rhythm_aligned_maps.py` (aligned maps + patch / halves statistics), readouts EVR ratio row,
  matching loading asymmetry, `fitsession --own-dir`.

* `wfield_local/face_rhythm_matching.py` + `scripts/face_rhythm_match.py` (+ tests): Hungarian matching of independent
  fits (on product / spatial / spectral / sum; spatial as vectors or magnitudes), angles in degrees, split / merge
  flags, chance (point / frequency permutation), pre-vs-pre floor.
* `scripts/face_rhythm_push_check.py`: frozen-basis error + face energy around treadmill push onsets vs random times.
* `face_rhythm_readouts`: `lick_split`, `task_nontask`, `severity_table` (histology burden) + two figures.
* `face_rhythm_factors.py`: `match` handles unequal ranks; gallery names on the timecourse panel.
* `fr_groups.master_v1` with close / far parts and modes S / D.

## 5. Pitfalls met this session

* The `fr` env lacks the repo install: `export PYTHONPATH=C:/Users/SabatiniLab/Github/widefield_pipeline`.
* `select` refuses an existing `master.npz`; `project` / `fitsession` skip existing files -> move old outputs aside
  before re-selecting a rank (done for rank 10 as `*_r10`). Plan `loso_jobs` carries the rank (switched to 12 on O2;
  backup `logs/plan_loso_rank10.json`).
* O2 shell variables B / S collide between the FR and encoding bundles.
* Gallery / validation number factors from 0; FINDINGS 4a uses the same numbering.
* Multi-line bash heredocs with quotes break in this shell: write patch files instead.

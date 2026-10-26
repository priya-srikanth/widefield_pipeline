# Encoding handoff — 2026-10-26 (updated ~13:50) — READ FIRST

Supersedes the RUNNING / TO-DO lists of `STATUS_2026-10-22_ENCODING_HANDOFF.md` §000z (its addenda record 10-25 night →
10-26 early morning; keep it for history and pitfalls). Scratchpad `S = C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad`.
Run all python from the worktree with `PYTHONPATH=<worktree>` (another `wfield_local` install shadows it).

## Page

https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d **v103 LIVE; v104 PREPARED LOCALLY, NOT PUBLISHED** (14:15: Artifact tool switched off in the session -- publish S/goal_page/index.html with files={"iti_area_compare.png", "iti_basis_decoder.png"} when available; adds orofacial-vs-control PETHs 3.24, basis decoder 3.14 + corrected whole-cortex claim; v103 12:00: the 9 images missing from v98-v102 uploaded; v102 11:30: target vs execution 1.34 / 1.35, pixel parts 1.36; v100 09:40: recovery marker 3.10, binned decoder 3.11 / 3.12, whole-cortex supplement 3.19-3.21; contact + paired + lick-count lines in 3.9), source `S/goal_page/index.html`; publish with the LONG path
`C:/Users/SabatiniLab/AppData/Local/Temp/...` (the 8.3 path is blocked for images).

- **Numbering = Fig SECTION.POSITION** (Priya 10-26: "why two letters"); ids `fig<sec>-<k>`. To add a figure: insert it
  with any temporary label (e.g. `Fig 93a`, id `fig93a`; links `href="#fig93a"`), then run `S/renumber_sections.py`
  (rewrites labels, ids, links, cross-references; writes `goal_page/fig_map.csv`). Keep `docs/status/GOAL_PAGE_FIG_MAP_2026-10-26.csv`
  = pre-10-26 numbers → current (compose with the new fig_map after every renumber; docs written before 10-26 cite OLD numbers).
- Epoch colours on every new figure = `orofacial_cohort.EPOCH_LINE` (pre black, acute pink, subacute goldenrod, chronic teal).
- Where things stand: rows added 10-26 — self-initiated licks (KEY), cortex–face-lick coupling; Pending list updated.

## RUNNING (do not restart; check logs)

### NOW (16:20) -- read this block first
- **DONE 15:46: TE partial refit** (exit 0; ENCODING_FINDINGS 8.20 top, FINDINGS 4d x; LOCAL page 1.37 as
  `te_partial_refit_lines.png` -- NOT `partial_refit_lines.png`, which is the OLD design's figure, now 1.38). Result:
  landing-labelled minus target-labelled lick kernels lower chronically in every area (0/4 up); acute target advantage
  falls. Same without PS95_0907. To do: retitle the line figure for the TE nulls + draw the landing - target contrast.
- **"go" SENT to sabatinilab-5b 15:50; waiting for them to report the table live.** Was: AFTER the refit exits: message sabatinilab-5b "go" for the kinematics_all swap (their regenerated table in
  `session_poses/pose3d/kinematics_all__tmp_20261026/`: only daq_peak_s differs, diff checked; Priya APPROVED the rename
  10-26 and runs the lines 5b gives her). Then re-run `python -m wfield_local.stray_licks` (patched 1d6e227; move the old
  `<E>/stray_licks/sessions` aside first -- it skips sessions already done; derived outputs, Priya's OK not needed to
  rename) and update ENCODING_FINDINGS 2.6.
- **DONE 14:35: frozen decoder in other bases** (`iti_basis_decoder.py`): result holds in joint LocaNMF and the pre-stroke
  SVD basis (ENCODING_FINDINGS 8.17 top, FINDINGS 4d vii); figure added to the LOCAL page after 3.13 (unpublished).
- **WHERE decoder** (Priya 10-26: angle / distance together, "or lateral location instead of angle"):
  `scripts/iti_where_decoder.py` DONE 15:00 (ENCODING_FINDINGS 8.17 top, FINDINGS 4d viii); on the LOCAL page as 3.15
  (unpublished) (it picks up the basis
  folders if present and raises if a basis file does not line up). Targets lateral (side_ml), outward (out_ap), 2-D
  vector r, angle / reach polar check; prints reach spread per epoch. -> page next to 3.9.
- **ITI lick RATE vs recovery** (Priya 10-26: "does inter-trial licking correlate with recovery (and exceed pre-stroke
  levels)"): `scripts/iti_lick_rate.py` DONE 15:40 (ENCODING_FINDINGS 8.17 top, FINDINGS 4d ix; LOCAL page 3.11; `--plot-only`
  redraws from the CSVs). Answer: dips with the deficit and returns, exceeds pre only in PS95. Rate = ITI licks / ITI minutes (next strobe - stop, clipped 0..60 s), no-contact rate, per trial, contact share;
  change vs pre + within-animal correlation with far-contra behaviour (raw, log-day detrended). Earlier count check:
  fewer ITI licks post-stroke in PS92-94 (so exceeding pre is unlikely in counts; rate is the fair test).
- **Page publish pending**: files = iti_area_compare.png, iti_basis_decoder.png, iti_where_decoder.png, iti_lick_rate.png, iti_direction_decoding.png (3.7 now with SEM bands),
  te_partial_refit_lines.png
  (all copied into S/goal_page; renumbered 16:00, map in GOAL_PAGE_FIG_MAP_2026-10-26.csv).
- Pushed to origin through e9656c8; LOCAL commits since (1d6e227 stray fix, b60d363, ff61daa, d47d32c, + this) need a
  cherry-pick push (memory note: never rebase this branch; cherry-pick onto origin/main in a temp worktree).


1. **DONE 11:05 (areas 08:49, basis 10:10, self-fit 11:05; first readouts ENCODING_FINDINGS 8.20, page 1.34 / 1.35,
   `scripts/te_readout.py`). TE partial refit now running (`S/partial_te.log`).** 1. **Encoder redesign, FROZEN_DESIGN=te** (`S/run_te.sh`, log `S/te_chain.log`; started 06:34):
   frozen AREA run `frozen_encoder_gate-attempted_v7g3_t3d_frA_te` (log `S/frozen_te.log`, `--no-drop-spout`, default
   variants) → PIXEL basis run `..._frA_te_basis` (FROZEN_PIX_MUSALL=1; `S/frozen_te_basis.log`) → self-fit
   (`session_vs_frozen.py --tag ..._frA_te --variants joint no_target no_execution no_contact no_task no_movement`, `S/svf_te.log`).
   Design (DECISIONS 2026-10-26; code `wfield_local/frozen_encoder.py` TE_BLOCK + `te_lick_parts`): TARGET = pos_enl,
   pos_cue + lick-locked kernels labelled by the trial's target spout (post-cue, pre-cue/ENL, cam1-only licks; never ITI);
   EXECUTION = unsplit 3-D tongue/jaw, exec angle, all FaceRhythm, lick-locked kernels by 3-D LANDING (+ unknown), ITI
   licks included; CONTACT = every DAQ-detected lick of the session (Priya: contacts just outside trials count); nuisance
   timing / other. Frames = whole engaged session incl. ITI (14 % of frames pre, 29 % acute). Smoke PS92_0806 / 0819 OK.
2. **DONE 07:55 -- ITI chain v4** (wide decoding lags -1..+1 s): `S/iti_full4.log` (re-extract + summary) → `S/iti_key4.log` (key
   figures) → `S/iti_ctrl4.log` (controls) → `S/iti_binned.log` (binned frozen-decoder sweep). Outputs `<E>/iti_licks/`
   (session npz backups of earlier versions: `sessions_v1..v3`, figures `figures_v3`; derived, deletable).
   At 06:59: TE frozen area run in its pre-statistics pass (17/31 pre sessions); ITI v4 extraction DONE (summary
   running, then key / controls / binned); whole-cortex maps (job 4) STARTED; old partial refit 64/101.
3. **DONE 08:31 (results ENCODING_FINDINGS 8.19 top; figures in its partial_refit/figures) -- Lick-kernel partial refit, OLD design** (`S/run_partial_ls.sh`, 7 workers, `S/partial_ls7.log`; 62/101 at 06:56).
   Superseded for the contest by the TE design, but let it finish: then `run_partial.sh`-style `--plot-only` with
   `--tag gate-attempted_v7g3_t3d_frA_ls`.
4. **DONE (re-run 08:30 on RAW dF/F AUC; per-area normalisation dropped) -- Whole-cortex supplement** (Priya 10-26: "non-orofacial per-area activity maps ... supplemental figure with VIS
   areas"): `scripts/iti_area_maps.py`, queued to start when the ITI re-extraction finishes (`S/run_area.sh`, log
   `S/iti_area.log`) -> `iti_area_maps.png` (per-area response / build-up AUC change from pre on the Allen atlas,
   significant areas outlined), `iti_area_peth.png` (non-orofacial PETHs: VISp, VISl, VISam, RSPd, SSp-bfd / ul / ll / tr,
   ACAd, PL), `iti_area_groups.png` (orofacial vs visual + RSP, session dots) -> page supplement (section 3).
5. **DONE (all 10-26 work through 13:00 on origin)** -- Push of the 10-26 commits: the first attempt hit a rebase CONFLICT in DECISIONS.md (another session appended a
   10-25 FaceRhythm entry at the same place); resolved by keeping both entries (theirs first), rebase completed, push
   retrying (task in this session). CONFIRM on origin: `git log origin/main --oneline | grep "Handoff 2026-10-26"`.

## TO DO (in order, with why)

A0. **DONE v98.** **Whole-cortex supplement → page** (running job 4): say in the caption whether the larger response / weaker
   build-up are orofacial-specific or cortex-wide (orofacial vs visual + RSP test).
A. **DONE v98 (results ENCODING_FINDINGS 8.17 top, FINDINGS 4d iii-iv).** **ITI v4 outputs → page**: frozen time courses now -1..+1 s (3.8), refreshed 3.9 / 3.12 / 3.13; binned decoder
   (`iti_binned_sweep.png` = configuration sweep, boxed = chosen on PRE only; `iti_binned_summary.png` = one number per
   session at the chosen configuration, raw + normalised, all areas / orofacial / visual+RSP). Priya asked for this
   ("binned time dynamics ... single-number decoder comparison across sessions").
B. **(i) + (ii) DONE 09:00** (ENCODING_FINDINGS 8.17 top, page 3.9 caption; `scripts/iti_contact_check.py`): no-contact
   in-trial licks recover too; paired ITI - in-trial gap widens chronically -0.062 [-0.110, -0.014]. (iii) DONE 09:30 (`iti_recovery_marker.py`,
   page 3.10): in-trial tracks the recovery time course (+0.62; spread-controlled +0.53), ITI does not;
   beyond the time trend +0.16 n.s.
   B. **3.9 decisive checks** (Priya: "3.9 looks really good — is it as important as I think?"): (i) in-trial licks
   WITHOUT spout contact through the frozen decoder (does in-trial recovery ride on contact?); (ii) session-paired
   nested test of the ITI − in-trial frozen difference (per animal chronic gap: PS92 -0.11, PS94 -0.16, PS95 -0.24,
   PS93 -0.01). (iii) **Recovery marker** (Priya: "perhaps a marker of recovery?"): within animal, does the in-trial
   frozen decoder track that day's behaviour (far-contra hit rate / pre, `recovery_timing.series`) across post sessions,
   and the ITI one not? Chronic behaviour has no between-animal spread, so it is a within-animal time-course test.
C. **Frozen unique + self-fit DONE (8.20).** Pixel CHANGE maps of the group parts DONE (page 1.36, `frozen_pixel_maps.py` now shows `execution`). Still open:
   PRE-level group-kernel maps (each group's evoked part per spout in pre; replace 1.20-1.21);
   pixel unique maps; TE partial refit readout when it lands (`partial_refit.py --plot-only --tag ..._frA_te`
   with FROZEN_DESIGN=te). C. **TE readouts** when run 1 lands: unique(target) vs unique(execution) (frozen, epoch − pre, nested; per area +
   pixel); self-fit remapping; GROUP-KERNEL maps = each group's summed contribution averaged around the event per spout
   (from the basis run's per-group parts, PART_VARIANTS) — replaces page 1.20–1.21 single-beta maps and the contests
   1.33–1.36. **TE partial refit: code DONE (61743c3) and QUEUED** (`S/run_partial_te.sh`, log `S/partial_te.log`; waits
   for `done` in te_chain.log and `exit` in partial_ls7.log, then 6 workers; ~11 design builds x ~2 min per session ->
   ~6 h local, no O2). Sets target / target_spout / target_lick / execution / exec_land / contact / target_land; nulls:
   ONE block permutation of trial spouts fed to raw_blocks AND te_lick_parts (`tpos_perm`), landing weights circularly
   shifted along the landed licks (`land_roll`, 10-90 %). Smoke PS92_0819: target null changes only pos_enl / pos_cue /
   lick_onset@tgt:*, landing null only lick_onset@land:*, defaults rebuild identically (label agreement 0.11).
   Decoder / encoder signals = per-session adopted SVD x U_atlas averaged per Allen area (NOT a shared LocaNMF basis;
   Priya asked 10-26) -- registration drift cannot explain ITI-down / in-trial-recovered (same sessions, same features).
D. **Pose window TOLD 11:35 (sabatinilab-5b): it regenerates the table into kinematics_all__tmp_20261026/, diffs vs live (only daq_peak_s may differ), sends me the diff; swap = rename on M: -> needs Priya, and only after the TE partial refit ends (~16:30). stray_licks PATCHED (1d6e227, lick_uid join); re-run after the refit (memory).** Tell the pose window: `pose3d/kinematics_all/licks.csv.gz` daq_peak_s was rounded to 4 significant digits (writer fixed
   c2aa922; regenerates on its next kinematics step); `stray_licks.py` (ENCODING_FINDINGS 2.6) must be re-run with exact
   times (lick_uid → licks_all.csv).
E0. **Frozen decoder in other bases RUNNING** (Priya 10-26: LocaNMF + the pre-stroke SVD basis; `scripts/iti_basis_decoder.py`,
   log `S/iti_basis.log`): joint LocaNMF bases are per animal, fit 08-15 on 11 PRE sessions (06-06..08-14), every session
   PROJECTED from current SVTcorr; SVD basis = te_basis fit/spatial_<animal>.pkl. Decoder trained on August pre only.
   Per-animal basis is fine (decoders are per animal; no cross-animal basis needed). -> page next to 3.9.
E. Open from before: epoch-kernel drift-RATE test (8.18; within-chronic r per day vs pre rate); the narrow every-lick
   window (Priya to confirm first-lick primary); June→August drift ceiling; lesion burden / leave-PS93-out; share
   deletions / nokin retirement (Priya decides).

## Results of 10-26 (all PROVISIONAL; details ENCODING_FINDINGS 8.17–8.19, FINDINGS 4d / 4e)

- **Self-initiated ITI licks (KEY, page 3.7–3.25):** within-session direction decoding intact; FROZEN pre-stroke decoder
  down every epoch for ITI licks (-0.135 r chronic, -22 %), largely recovers for cued in-trial licks (-20 % acute → -9 %
  chronic). Each lick evokes a larger response (~2× acutely, +0.6–0.8 chronic, also isolated licks; post-stroke bouts are
  SHORTER); slow build-up before the lick weaker chronically in tjM1 / SSp-m (3/4). Not lick kinematics (post-stroke ITI
  licks slightly slower; kinematics-adjusted / range-matched responses unchanged) and not face movement (face-adjusted
  build-up still weaker). "Relative" direction for ITI = relative to the PREVIOUS trial's spout (spout retracted in ITI).
- **Lick-count control (Priya: "is this a confound")**: ITI licks are fewer post-stroke (PS92-94), but the frozen
  decoder on test sets matched to 40 licks per session gives the same numbers (ITI -0.136 / -0.115 / -0.123,
  in-trial -0.144 / -0.080 / -0.064); more licks -> slightly lower r within epoch. `scripts/iti_nmatch_check.py`
  (uses the complete v3 extraction). Not on the page yet: add one line to the 3.9 caption.
- **FaceRhythm lick factor (1.16 / 1.17):** chronic trial-by-trial unique gain in bilateral orofacial motor cortex with
  LOWER trial-to-trial variance (-40 % / -44 %) → tighter coupling, not more variability.
- **Block-permutation nulls** reproduce the single-shift partial-refit contrasts (1.33 lines).
- **Landing vs target lick kernels, lagged FR, self-fit with lick kernels** (8.19; 1.35 / 1.36 / 9.21): landing kernels do
  not absorb the chronic target term; no shift toward cortex leading the face; remapped spout information is shared
  between target and lick-locked spout terms (old design — superseded by TE).
- **Epoch kernel fits** (8.18; 1.20 / 1.21): lick kernels diverge steadily from early pre but not yet beyond drift; the
  single-lag lick-kernel maps are leftover terms (collinear with templates / kinematics) → replaced by TE group kernels.

## PITFALLS (10-26)

- Heredocs / `python -` with `\n` inside string literals STILL break (hit twice more): write patch scripts with the Write
  tool; for small fixes use the Edit tool (Edit needs a prior Read of that file in the session).
- The whole-session 3-D table rounds daq_peak_s (above); `t_peak_3d` (<= 6.9 s) is fine.
- ITI lick `position` = the trial WINDOW's label (tail = trial just finished, lead = UPCOMING trial); use the previous
  trial's spout (`prev_pos`).
- `_pix_musall_variant` must accept `novar_*` (fixed); `frozen_pixel_musall.py --layout novar` uses the real variable bases.
- `summarise_musall` rewrites `frozen_musall_r2*.csv` with only the listed variants: re-score extra variants into a
  SEPARATE tag (copy fit/templates.pkl, pre_stats.pkl, spatial_*.pkl), as `..._frA_ls_basis_novar`.
- Editing `iti_licks.py` while its chain runs breaks later steps if array shapes change (LAGS_D): wait for the chain.
- TE runs need FROZEN_LICKSPOUT / LICKLAND / FR_SPLIT / FR_LAGS unset (TE raises otherwise); frozen_encoder backups:
  `S/frozen_encoder_backup3.py` (before TE).
- Push: pre-push tests ~8–14 min; races with other sessions retry automatically (temp worktree loop).
- PAGE IMAGES: publishing index.html alone does NOT upload new / changed PNGs (v98-v102 had 9 broken figures). Always
  pass every new or regenerated image in the Artifact `files` map (root = goal_page) on the same publish.

# Encoding handoff — 2026-10-26 (updated ~22:45) — READ FIRST

## 00. START HERE (22:45, second session of 10-26; supersedes section 0 for state / to-dos, section 0 kept as history)

Session = `f082ee0c` (new scratchpad `S2 = C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad`;
push scripts `S2/push<NN>.sh`, last `push39.sh`). Python / PYTHONPATH as below; `WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache`
for anything that loads imaging. E = `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding`.

### 00a. Page state
- LIVE: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d = **v118** (this session published v104-v118; the Artifact
  tool works here). Source `_goal_page/index.html` + PNGs (backup `M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/`,
  refreshed at every publish).
- Section 3 was renumbered TWICE tonight: new 3.21 (`iti_where_delta.png`) and 3.22 (`iti_where_per_spout.png`) -> old
  3.21-3.31 are now 3.23-3.32 (kinematics control = 3.27 now). Map: `_goal_page/fig_map.csv` (last renumber only) +
  `docs/status/GOAL_PAGE_FIG_MAP_2026-10-26.csv`. Insert new figures with a temp label like `Fig 93x` (renumber_sections.py
  cannot parse `3.20b`) and run `S/renumber_sections.py <index.html>`.
- Publishing: (1) `Artifact read` first in a new session (or a refusal hands you the live version); (2) replacing old
  PNGs needs `Artifact list scope=files` first; (3) the full PNG set (~65 MB) is over the 64 MB cap -> batches < 35 MB;
  (4) `edits` on publish is refused in this session (file-edit hook) -> Edit the file, then publish without edits.

### 00b. RUNNING at 22:45 (background in session f082ee0c; may die with it)
1. **Sagittal chain** `S2/run_sagittal.sh`, log `S2/sagittal.log` (started 22:43): iti_where_decoder (3 bases) ->
   iti_where_scatter -> iti_where_beyond_pull -> iti_zscore_check -> iti_where_per_spout (+ --z) -> iti_where_mae ->
   iti_decoder_contrib. Ends with `sagittal chain done`. If it died: rerun the missing steps (same order; each reads the
   previous step's CSV). Old lateral + OUTWARD outputs backed up in `<E>/iti_licks/outward_version_20261026/`.
2. **Push** `S2/push39.sh` (4972bdf sig bars + handoff). Then push **f253fa7** (bar style / y-labels) and this handoff
   commit by cherry-pick (template = push39.sh with the new SHA). origin/main had bf0b70e (FR window) at 22:44.

### 00c. AFTER the sagittal chain (in order)
1. Check every redrawn figure (3.15, 3.15 MAE, 3.16-3.18 bases / LocaNMF / SVD, 3.19-3.21 scatter + delta, 3.22 +
   z version, contrib). Compare the SAGITTAL numbers with the outward ones (KEY row says cued licks "read closer" on the
   OUTWARD axis: beyond-pull -0.30 / -0.30 / -0.31, z-scored -0.22 / -0.21 / -0.17). **Tell Priya before changing the KEY
   row / captions / findings if sagittal differs.**
2. Copy into `_goal_page`: iti_where_decoder.png, iti_where_decoder_bases.png, iti_where_decoder_locanmf.png,
   iti_where_decoder_svdbasis.png, iti_where_mae.png, iti_where_scatter_{trial,iti}.png, iti_where_delta.png,
   iti_where_per_spout{,_z}.png; update captions (axes are now lateral + SAGITTAL extension = sqrt(forward^2 + down^2);
   angle = atan2(lateral, sagittal) = the encoder's exec_angle; outward still scored in where_decoder.csv); publish with
   those files; back up; commit; push.
3. Then tell **sabatinilab-e3** (PI deck) the decoder figures are final: per figure the share PNG path, page Fig N.k,
   script + args, headline numbers, methods text (they asked 10-26 ~22:30; reply sent: not final yet).

### 00-COMPACT-9 (10-27 ~21:30) -- READ FIRST, then 8
- **Merged** the split-basis + like-for-like branch (147575c6; helper commits 4785cff3, 0918aab1). Switches:
  WFIELD_AREA_BASIS = split (default) | plain (iti_licks.AREA_BASIS; helpers il.area_features / area_sd / split_table /
  pred_file / split_only; loader iti_where_decoder.area_basis()); WFIELD_LFL_REF = gauss (default, Priya "ok gaussian")
  | loclin (likeforlike.offset). iti_lick_order default SOURCE = areas_split. CANONICAL UNTAGGED outputs = split basis
  (page stems unchanged); plain = "_plain"; per-lick caches carry the basis (*_splitr8; lick_order_preds_areas_splitr8,
  plain lick_order_preds_r8). Scripts without basis tags refuse a plain run (il.split_only). 'beyond' columns -> 'lfl'
  (lick_order, per_spout, first_lick, overcorrection, recovery_link, 3axis/check_figures, where_beyond_pull ->
  where_lfl_*, mixed-model off_*); iti_where_angle_reach lost its bey_* panels (-> lick_matched_angle_reach, now Gaussian
  primary + local-linear edge check beside it, 4cc5eb04).
- **v1 folder fix done (3ab92858):** `iti_licks.labels()` (current ROI folder; raises if empty or if any frozen-run
  session is missing) replaces 16 globs of the v1 `sessions/`; iti_nmatch_check retired to scripts/archive. Remaining
  SESS_NAME globs take an explicit sess_dir (June). v1 `sessions/` may now go on the delete list after the split run.
- **INCIDENT:** the helper's PS93-only tests overwrote 31 `iti_licks/figdata/*.pkl` on M: (pathlib writes slipped its
  redirect; list scratchpad split_test_figdata_overwritten.txt). Derived; all regenerated by the split run. Until
  V8SPLIT_DONE do not --plot-only those stems.
- **RUNNING:** scratchpad run_v8split.sh (started 21:29; logs rr8/ws_*.log; moves superseded files -> <E>/iti_licks/
  superseded_split_20261027/ (list rr8/ws_moved.txt); ends V8SPLIT_DONE + "ws tracebacks in:"). Est. 25-35 min.
- **NEXT:** check tracebacks; v2 vs v8 comparison (roi_v2_backup) incl. ramp_session_tests tjM1 build-up for Priya;
  page update (re-copy PNGs, captions 10-16, swaps 13.1/13.2 lfl, 11.9 -> lick_matched_angle_reach, 3.14 basis 4 cols,
  re-tier level maps Key, 15.2/15.4 captions, Fig 16.12 stale caption), publish, tell e3 / 5b. Then ALM/tjM1 rows in
  encoder / FR area figures (0.3, 0.4, 0.9-0.11, 0.13, 2.2, 2.6, 18.x, 21.4, 21.11, 22.3, 22.4).

### 00-COMPACT-8 (10-27 ~22:30) -- READ FIRST, then 7b / 7 / 6
- **Priya answered:** primary basis = Allen + ALM/tjM1 split ("I agree"; DECISIONS 10-27 late; robustness = plain Allen,
  masked LocaNMF, SVD). The "lost tjM1 chronic build-up significance" figure = `lick_iti_ramp_summary`
  (ramp_session_tests.csv). v1 vs v2 there: NOTHING lost -- raw build_auc chronic CI crosses 0 in both (right v1 -1.03
  [-2.27, 0.23], v2 -1.05 [-2.22, 0.08]); normalised right significant in both (v1 -0.19 [-0.36, -0.011], v2 -0.17 [-0.32,
  -0.019]), normalised left n.s. in both (upper CI +0.010 / +0.003). Check v8 and answer Priya with all three.
- **Code (0f56a173):** iti_licks default ROI_VERSION = "8"; plain-Allen per-lick preds CACHE tagged `_r8`
  (lick_order_preds_r8.csv.gz; outputs stay untagged = canonical); offset map TAGS + filter-compare BASIS -> areas_splitr8.
- **v2 tables backed up** (295 csv / csv.gz, 270 MB) -> <E>/iti_licks/roi_v2_backup_20261027/ (same relative paths) for
  the v2 vs v8 comparison (v1 = roi_v1_backup_20261027/). Add to SUPERSEDED list once v8 is on the page.
- **RUNNING:** scratchpad run_v8redo.sh (frozen env; logs + runtimes in scratchpad rr8/; ends with V8REDO_DONE and a
  "tracebacks in:" line): track A = iti_licks main/key/controls/binned, check_figures, area_maps, hemo (+summarise), f0;
  track B = split decoder (all + gap2 / far1 / gap2far1, gauss read-out), plain Allen (lick_order, gauss, decomposition,
  where_decoder); then offset maps (areas, split), mixed models, filter compare, lmm figures.
- **10-27 late decisions (DECISIONS.md):** split basis EVERYWHERE ("use split bases, for consistency"); 'beyond the
  pull' -> like-for-like offset everywhere ("yes replace beyond pull"); like-for-like reference = LOCAL-LINEAR ("ok local
  linear it is"; lateral + angle tie with Gaussian, Gaussian = supplement). A helper is coding split + beyond-pull in an
  isolated WORKTREE (no M: writes) and writes scratchpad run_v8split.sh -> merge after waves 2/3, then run it.
- **TODO after the split merge (Priya: "fix the code base so we don't need the v1 folder"):** ~18 scripts list labels
  from `il.OUT / "sessions"` (v1): add `iti_licks.labels()` (stems of OUT/SESS_NAME, fail loudly if empty / < 107) and
  replace every call site (iti_area_maps x2, iti_area_split_sd, iti_contact_check, iti_decoder_contrib, iti_hemo_check,
  iti_lick_order, iti_lick_rate, iti_lick_timing_audit, figdata.imaging_grid, iti_where_decoder.load, iti_zscore_check.sds,
  iti_basis_decoder, lick_session_level_maps, iti_pattern_area_stats, lick_roi_*, lick_offset_contrib_map -- re-grep);
  retire / convert iti_nmatch_check (hard-codes sessions_v3). Then v1 `sessions/` can join the delete list.
- **Duplicate run:** the stopped run_v8wave2.sh waiter survived TaskStop (orphan PID 7444) and started a DUPLICATE wave 2
  at 20:52 when V8REDO_DONE appeared; Priya asked to `taskkill /PID 7444 /T /F`. Harmless for results (split pass redoes
  all) but costs CPU. Pitfall: TaskStop does not kill a Git-Bash script's subshell tree -- stop waiters by PID tree.
- **PAGE-FIGURE AUDIT (helper, all 235 page images; map in scratchpad audit_map.txt):** pattern maps 15.x unaffected
  (SVD basis; lick times identical in v8 files); sections 0-9 / 17-22 unaffected (encoder stat mask). STALE + not in
  waves 1/2 -> WAVE 3 (scratchpad run_v8wave3.sh, logs rr8/w3_*, ends V8WAVE3_DONE): plain --ref place / spout / pooled
  (12.1, 12.2, 13.6, matched_sessions.csv) + gain_prediction; split decomposition + split --ref place + split loclin on
  gap2 / far1 / gap2far1; after V8WAVE2_DONE: filter_compare (now reads matched_loclin_*), mixed models, lmm ALL groups
  (iti_mixed_models.matched default ref place -> loclin). Wave 1's check_figures output is stale (use wave 2's rerun).
  Lateral loclin = gauss to 2 decimals on v8 split (share beyond pre range <= 8 %; ref-refit spread ~ +/-0.03 mm).
  OPEN for Priya (code needed): (i) split basis for the iti_licks frozen/key/binned + the z-scored where-decoder family
  (10.x, 11.x, 12.3-12.13, 13.3-13.9, 16.3-16.5) -- no split option there yet; (ii) replace 'beyond the pull' columns by
  loclin in lick_order / per_spout / first_lick / overcorrection / recovery_link / 3axis. New-rows-requested figures
  (ALM / tjM1 / rest rows): 0.3, 0.4, 0.9-0.11, 0.13, 2.2, 2.6, 18.5/6/9, 21.4, 21.11, 22.3, 22.4.
  Page swaps planned: 13.1 / 13.2 -> _areas_splitr8; 12.1 / 12.2 / 13.6 -> loclin; 11.9 -> lick_matched_angle_reach.
  Do NOT delete v1 `sessions/`: ~20 scripts list labels from it.
- **WAVE 2 STARTED EARLY (~23:15)** as scratchpad run_v8wave2b.sh (run_v8wave2.sh's waiter stopped before it ran anything):
  wave 1 was down to the hemo check (3 workers, CPU 20 %), and wave 2 needs only the wave-1 decoders. More workers
  (june_ref 6, zscore 8, rim_overlay 8) + added `lick_matched_angle_reach.py` (after angle_reach) and plain-Allen
  `lick_matched_readout --ref loclin`; its final check_figures / mixed models / lmm wait for the exact line V8REDO_DONE.
  Local-linear test on the split basis ran separately (rr8/loclin_split_all.log). CPU ~94 %, 36 GB free after start.
- **WAVE 2 queued (superseded by the line above):** scratchpad run_v8wave2.sh (waits for V8REDO_DONE; logs rr8/w2_*.log; ends V8WAVE2_DONE). Helper
  audit (10-27): iti_zscore_check was NOT in wave 1 -> every `_z` page figure was still on v1 numbers (where_decoder_z /
  where_preds_z dated 10-26); wave 2 runs zscore_check -> where_decoder --z -> scatter / per_spout / rmse / spread /
  overcorrection / recovery_measure / recovery_link (angle_z, lateral_z, hit), june_ref (june_ref/ held v1 only) -> drift
  --june, basis --summarise, decoder_contrib, farcontra, 3axis, drift, angle_reach (+extension_dist), first_lick (+misdirected,
  aim_deficit), contact_check (+recovery_marker), gain_prediction, ROI-mask figures; then check_figures / mixed models / lmm
  AGAIN (wave 1 ran them on stale inputs). Deleted 2 stale caches first (first_lick_preds, where_angle_reach_preds; copies in
  roi_v2_backup). NOT affected (helper): decoder / spout / session pattern maps + level maps (lick times / SVD / stat-mask
  only). Known: iti_nmatch_check hard-codes sessions_v3 (not on page; left); ~20 scripts list labels from v1 `sessions/`
  (same 107 files; harmless today).
- **Second wave (audit by a helper):** every other script whose outputs depend on the v8 session files / SDs / area
  features (Haufe pattern maps, spout patterns, session level maps, ...) -- run after the first wave, then compare v2 vs v8,
  re-tier + publish page, tell e3 / 5b.

### 00-COMPACT-7b (10-27 ~22:00): the env-correct v8 extraction FINISHED and PASSED -- 107 files in sessions_roi8fr/, FR env
applied (master_pre_v1 / master_v1:A / attempted), 0 sessions with all-NaN bout_fr, 0 failures. NEXT = 00-COMPACT-6 step 2
(default ROI_VERSION -> "8", full redo under the frozen env).

### 00-COMPACT-7 (10-27 ~21:15) -- READ FIRST, then 00-COMPACT-6 (still valid except as noted here)
- **v8 extraction RUNNING (env-correct):** S2/run_roi8fr.sh (sources S2/rr2/frozenenv.sh, WFIELD_ROI_VERSION=8) ->
  <E>/iti_licks/sessions_roi8fr/ (iti_licks._ROI["8"] now points there). Log S2/roi8fr.log ends with ROI8FR_DONE + a check
  line "107 files; sessions with all-NaN bout_fr: 0 []" -- if that count is not 0, the env did not apply: stop and fix.
  SD tables for v8 already exist (zscore_check/session_sd_roi8_rimlat_almline.csv, session_sd_split_roi8_rimlat_almline.csv;
  imaging-only, env-independent). The env-less first v8 run (sessions_roi8/) was deleted.
- **Then (00-COMPACT-6 step 2 onward):** default ROI_VERSION -> "8"; full redo UNDER THE FROZEN ENV (source rr2/frozenenv.sh
  before every iti_licks-based script); split decoder tag becomes areas_splitr8 (check lick_iti_filter_compare BASIS and
  lick_offset_contrib_map TAGS, currently areas_splitr2); plain Allen decoder chain too; compare v2 (sessions_roi2 +
  roi_v1_backup for v1) vs v8; then page re-tier + publish + tell e3 / 5b.
- **Split decoder on v2 ROIs finished** (areas_splitr2: all ITI + gap2 / far1 / gap2far1 read-outs, offset map, filter
  compare figure redrawn with BASIS areas_splitr2) -- numbers not yet reviewed; superseded by v8 anyway.
- **Deleted on M: with Priya's permission (10-27 ~21:00):** 5.7 GB + small files, listed (rows marked DELETED) in
  docs/status/SUPERSEDED_OUTPUTS_20261027.md. Remaining rows wait for their stated condition; Priya: "ask me if you are
  unsure" -- standing permission to delete superseded / incorrect derived outputs on M: (still never original data).
- **Open questions to Priya:** primary decoder basis (recommendation: Allen + ALM/tjM1 split; masked LocaNMF + SVD
  robustness); which figure showed "lost tjM1 chronic build-up significance" (v2 numbers kept it: -0.37 [-0.63, -0.09],
  17/20 bins); within-spout change test option (relax done; first-lick rows hidden).

### 00-COMPACT-6 (10-27 ~20:30) -- READ FIRST (supersedes 00-COMPACT-3..5 for state; they stay as history)
**Pages:** goal page LIVE v139 (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d; v136 = 23 named sections, v137 = Haufe pre
map PuOr_r vs change RdBu_r + collapsible *_levels supplements, v138 = sign key in 11 Haufe captions, v139 = Fig 12.14 ITI
lick-set comparison). `_goal_page/index.html` ALSO holds UNPUBLISHED v2-ROI caption edits (20, by the ROI-redo helper; list in
its report = scratchpad rr2/, pre-edit copy rr2/index_before_roi2.html) + 16 refreshed PNGs (lick_iti_ramp_*, area_compare /
groups, hemo variants / state_gap, check_f0*, kinematics / face control, binned_*, lmm_*_response) -- they will be redone
for v8, so do not publish them as-is unless v8 is abandoned. Review page (loose ends + ROI decisions):
https://claude.ai/artifact/P4Md3KfTSt3Nqq9hxU2knq (v11; source `_review_1027/`, sections A-G).
**Decisions today (DECISIONS.md):**
- ITI licks: ALL ITI licks primary; 2 s quiet / >= 1 s from window end and next spout arrival / both = SECONDARY
  (WFIELD_ITI_FILTER gap2 / far1 / gap2far1; offset holds in all; page 12.14). The ITI is only ~3.1 s; ~70 % of ITI licks
  sit within 1 s of a trial edge (iti_lick_timing_audit.py).
- ALM / tjM1 / mask FINAL = ROI v8 (`scripts/iti_licks.py` `_ROI`, WFIELD_ROI_VERSION): ALM 40 % of MOs, tjM1 50 % of MOp,
  16-px rim thinned to 4 px along the LATERAL edge (anterior of AP -0.5), ALM continued to one horizontal front line
  (8-px front rim), >= 0.19 mm from FRP, one connected region each. Literature: ALM Guo 2014 / Esmaeili 2021 (AP 2.5 ML 1.5),
  tjM1 Mayrhofer 2019 (AP 1.8 ML 2.6) inside; CFA / wM2 outside; Esmaeili's tjM1 targeting point (2.0, 2.0) falls in Allen MOs
  here. Per-pixel map statistics keep the 16-px stat mask. v3-v7 rejected (DECISIONS).
- Per-session Haufe LEVEL maps (amplitude kept; `scripts/lick_session_level_maps.py`) -> Key map figure; per-session
  amplitude-kept change maps Supporting; held-out-epoch figures (15.12-15.15) + epoch-decoder maps (15.6-15.8) ->
  Supplementary. Within-spout first-lick rows hidden (decoders fit ~nothing). APPROVED; re-tier together with the v8 redo.
- Map tests: NO session / animal gate (Priya never asked for >= 3 / >= 3): every animal with >= 1 session per group;
  per-animal effect size with a pooled within-animal session SD, summed / sqrt k (shared `iti_session_cluster_tests`
  change_z / level_z / test_one / test_level); ">= 3 animals agree" only as the dashed outline. Big moves: shape-only ITI
  lateral subacute change 83 -> 7.5 %; cued lateral level acute / subacute 1 / 7 -> 60 / 46 %; ITI far-contra acute level 91 %.
**Basis question (OPEN, Priya leaning "if LocaNMF is better use it"):** held-out pre lateral r ITI / cued: Allen split
0.625 / 0.768, Allen 0.622 / 0.767, LocaNMF masked (glue / MOB comps dropped) 0.622 / 0.761, LocaNMF 0.608 / 0.754, SVD 0.597 /
0.736. Recommendation given: Allen + ALM/tjM1 split primary, masked LocaNMF + SVD robustness. Offset map
(`lick_offset_contrib_map.py`): uniform cortex-wide change contributes ~0; SVD's smaller read-out = held-out pre-reference
term. The joint LocaNMF basis is PRE-STROKE only (corrected earlier claim) and ~1/3 of its comps sit on glue / MOB -> spout
classifiers on it need the same mask check.
**Findings today (PROVISIONAL):** ITI offset robust to stricter ITI sets; cued "layout drops then returns" holds (per-session
maps); ITI "coding increases and spreads" NOT supported as a sustained effect (subacute, PS94-driven, PS95 opposite; the
cortex-wide component is clearest in ITI first licks chronic); the strongest pre lick response sits at AP -0.1 ML 2.4-2.6 =
Allen SSp-ul (limb S1 or a registration shift of orofacial S1 -- unresolved; describe by position); ALM is not noisier than
MOs (SNR 1.35 vs 1.24); tjM1 40 -> 50 % changed tjM1 numbers by ~5-10 %, all directions unchanged (helper report).
**RUNNING / NEXT (in order):**
1. Priya asked to kill the v8 extraction (PID 18800; started WITHOUT the frozen env -> empty FaceRhythm fields). Relaunch v8
   under the frozen env: source scratchpad rr2/frozenenv.sh (COMBINED_ENCODING_FR=master_pre_v1, ..._FR_GROUPS=master_v1:A,
   OROFACIAL_GATE=attempted, COMBINED_ENCODING_LICKS=combined, ... see the file) then S2/roi8_extract.py with
   WFIELD_ROI_VERSION=8; writes sessions_roi8fr/ (sessions_roi8/ = complete env-less run, superseded) -- check 107 files and bout_fr not NaN (the launcher S2/run_roi8fr.sh does).
   The split-SD pass (session_sd_split_roi8_...) does not depend on that env. ALSO compute session_sd_roi8 (iti_zscore_check
   sds inside roi8_extract.py). EVERY extraction must use the frozen env (pitfall).
2. Switch iti_licks ROI_VERSION default to "8"; rerun under the frozen env: the ROI-redo set (helper's list: iti_licks
   --summarise / --binned / --key / --controls, iti_area_maps, iti_hemo_check (cache per ROI version: extend to v8),
   iti_f0_check, iti_check_figures, iti_mixed_models, iti_lmm_figures --groups response) + split decoder
   (WFIELD_LICK_DECODER=areas_split; tag areas_splitr8 -- check SRC_TAG rule 'r<ROI_VERSION>') + the 3 ITI filters +
   lick_offset_contrib_map + lick_iti_filter_compare (BASIS -> areas_splitr8) + the PLAIN Allen decoder chain (where_decoder,
   lick_order, lick_matched_readout gauss, decomposition, lmm figures): the plain areas features change with the lateral rim.
   Compare v2 vs v8.
3. Then: re-tier the page (level maps Key, etc.), fix Fig 16.12 caption (stale deficit numbers -- helper found it; new
   values in its report), update 15.2 / 15.4 captions for the new test, publish, tell e3 / 5b.
4. Encoder / FR area figures: per-pixel SSE / SST are saved as 2x2-block sums -> ALM / tjM1 / MOs-rest / MOp-rest R2 rows can
   be re-aggregated cheaply; trial-PETH-by-area figures may need re-extraction (cost it first). Priya: yes, after the ITI redo.
5. Basis decision with Priya; then the SVD-GAUSS queue items (SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md).
**Superseded outputs to delete later (Priya decides):** docs/status/SUPERSEDED_OUTPUTS_20261027.md.
**Pitfalls (today):** (a) any iti_licks extraction MUST run under the frozen env (rr2/frozenenv.sh) -- without it bout_fr /
first_fr are NaN and the first-lick set shrinks, silently (log=None hides it); (b) Windows spawn: driver scripts need
`if __name__ == "__main__":`; (c) never delete on M: -- use new file names (an empty session_sd_roi2.csv blocked a cache);
(d) `R.map` is the DataFrame method -- use R['map']; (e) lick_order_preds are cached by SRC_TAG -- new inputs need a new tag.

### 00-COMPACT-5 (10-27 ~19:00)
Page v138: Haufe figures pre map PuOr_r vs change RdBu_r, collapsible *_levels.png supplements (15.12-15.15), sign key in
11 Haufe captions. Priya: lick_decoder_pattern_stats_lat_levels "may be a key figure"; also candidates for Key: 15.5 within-
spout, 15.4 position, 15.6 decoder patterns, 15.8 cued spout patterns -> re-tier ONCE the per-session level maps land.
RUNNING: helper building scripts/lick_session_level_maps.py = per-session self-fit Haufe maps WITH amplitude (/ animal's pre
RMS), level TFCE (sign-flip) + change TFCE, all licks AND first licks, global-share per animal (tests Priya's reading
"ITI lateral coding increases and spreads; cued drops then returns" against a cortex-wide component).
ALM / tjM1 check (scripts/lick_roi_check.py, figures/lick_roi_check.png, roi_check.csv): within MOs the anterolateral 40 %
carries 1.4-2.5x the |pre map| of the rest (ALM placement supported); tjM1 mixed (0.7-2.7x; the cued SAGITTAL hot spot sits
posterior to tjM1, ratio < 1); the strongest lateral coding is LATERAL to both ROIs (lateral orofacial S1 edge).

### 00-COMPACT-4 (10-27 ~18:00) -- BASIS COMPARISON (primary basis = Priya's call, pending)
Held-out pre lateral r (ITI / cued): Allen split (ALM + MOs_rest, tjM1 + MOp_rest; source areas_split) 0.625 / 0.768;
Allen 0.622 / 0.767; LocaNMF masked (comps > 50 % outside glue / MOB mask dropped; tag _locanmfFM) 0.622 / 0.761; LocaNMF
0.608 / 0.754; SVD frame-scaled 0.597 / 0.736.
Same-place gauss, change vs pre (acute / subacute / chronic): ITI all licks -- split Allen +0.37 / +0.22 / +0.26 (all 4/4,
chronic CI [0.09, 0.46]); masked LocaNMF +0.36 / +0.25 / +0.23 (chronic CI incl. 0, 3/4); SVD +0.19 / +0.07 / +0.14.
Cued first lick -- split Allen +0.32 / +0.22 / +0.15 (4/4 each).
Offset map (scripts/lick_offset_contrib_map.py; offset_contrib/): uniform cortex-wide change contributes ~0 in Allen and
SVD (global part <= 0.02). With all-pre scoring the chronic ITI activity-change offset is similar in every basis (Allen
0.25, split 0.27, LocaNMF 0.29, masked 0.25, SVD 0.22); SVD's smaller READ-OUT (0.14) is the held-out pre-reference term
(-0.08 chronic, -0.15 acute; PS95 -0.18, PS94 -0.09): with 200 features, held-out pre sessions are read differently from
in-sample ones (least stable decoder; matches its lowest held-out accuracy). Unmasked LocaNMF decoder leans on one
posterior-right component + edge pixels (see maps) -> do not use unmasked LocaNMF; the joint-LocaNMF SPOUT classifiers
(sections with trial spout decoding) need the same mask check.

### 00-COMPACT-3 (10-27 ~17:00)
**CORRECTION:** the 'joint' LocaNMF basis is joint ACROSS PRE-STROKE SESSIONS ONLY (manifest labels: 11 sessions per animal,
June + August, last day -2/-3); NOT fit on post-stroke data. So all three bases are fixed before the stroke and the decoder
is frozen pre-stroke in all three. The SVD-vs-Allen/LocaNMF persistence gap is NOT 'pre-only vs joint basis'. Leading
difference: LocaNMF components are LOCALISED (area-constrained) like Allen areas; SVD components are global modes.
Proposed diagnostic: per-pixel offset-contribution map per basis (decoder pixel weights x post - pre activity at matched
landing) to show where in cortex each basis's chronic offset comes from. LocaNMF frame-scaled run (tag _locanmfF)
rerunning after a NaN-padding fix in iti_lick_order.py (component counts differ by animal).
**Haufe pre-map colormap + collapsible absolute maps:** queued (queue doc), after the figdata / restructure helpers.

### 00-COMPACT-2 (10-27 ~16:00) -- READ FIRST (adds to 00-COMPACT below)
**Priya's answers on the review page (P4Md3KfTSt3Nqq9hxU2knq):** (1) aim-deficit comparison figure -> SUPPLEMENTARY next to
the recovery measure; (2) TFCE contours replace the per-pixel contours on the per-session change-map figures (per-pixel kept
as supplementary; caption: TFCE = family-wise, session labels permuted within animal; a large significant area = broad
layout change of a shape-only map); (5) shape-distance dots figure (lick_session_shape_d.png) REPLACES the whole-map
heatmap (heatmap linked as table view); (6, 7) map similarity over days (cued, ITI) -> SUPPLEMENTARY; (8) drift sensitivity
-> repo only (no page).
**Page restructure (Priya):** split ballooning sections into named topical sections of ~6-15 figures, Key / Supporting /
Supplementary within each, every sidebar entry = number + short descriptive name, stable #f-<id> links. A helper agent is
doing the restructure + the figure changes above on `_goal_page/index.html` (NOT published; backup
encoding_goal_page_src/index_pre_sections_20261027.html; map docs/status/GOAL_PAGE_FIG_MAP_2026-10-27_SECTIONS.csv). When
it reports: check, copy the listed new images into the publish `files`, publish as v136, tell e3 + 5b (numbers changed;
stable ids unchanged).
**Open observation (not a result):** PS95 cued FAR-MID per-session map similarity to pre dips to 0.31-0.41 on days 25-39
(pre floor 0.45-0.87, mean ~0.66), recovers by days 46-50; close mid also dips (days 25-36); far contra unaffected (0.79-0.90);
behaviour unchanged (far-mid landing -0.20..-0.27 mm, aim improving, lick counts normal). Offered checks (Priya: not yet
answered): session state in 9/10-9/24 (running, engagement, lick-response size); imaging / acquisition notes around 9/10
(docs/EXPERIMENT_ERRORS.md, preprocessing QC).
**Priority notes:** compensation check (first-lick contact rate etc.) LOW, after the non-linear decoder.
**FRAME-SCALED SVD (svdF) RESULT (16:00):** same as lick-scaled and unscaled SVD -> scaling is NOT the reason SVD differs.
Same-place gauss, change vs pre: ITI all licks +0.19 (4/4, CI [+0.01,+0.40]) / +0.07 / +0.14 (n.s.); ITI bout starts +0.27 (4/4) /
+0.12 / +0.17 (4/4, CI excl 0); cued first lick +0.23 (4/4) / +0.10 / +0.06; cued far contra -0.32 chronic (4/4). PS93
negative in SVD in later epochs (ITI -0.14 / -0.09), PS95 ~0; PS92 / PS94 positive. So: the acute ITI and first-lick
contralesional read-out is basis-robust; its PERSISTENCE (subacute / chronic) is Allen / LocaNMF-dependent and in the SVD
basis rests on PS92 + PS94. Decision needed with Priya before switching the page to SVD (which basis is primary for the
persistence claim; report basis-dependence explicitly). Run LocaNMF frame-scaled for completeness.
**Push:** 15 commits since 78e6593d pushing via S2/push51.sh (log S2/push51.log); later commits need a new push script.

### 00-COMPACT (10-27 ~15:40) -- READ FIRST; then the queue doc docs/status/SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md
**Pages:** goal page LIVE v135 (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d; source `_goal_page/index.html`, untracked,
backup M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/). Review page for loose-end figures (not yet on the goal page):
https://claude.ai/artifact/P4Md3KfTSt3Nqq9hxU2knq (source `_review_1027/`, untracked) -- Priya has not answered its 5
decisions yet (aim figure on page? TFCE contours replace per-pixel? shape-d replaces heatmap? shape-over-days as
Supporting? drift table in 3.13 caption?). Figure links: stable `#f-<image stem>` (registry `_goal_page/fig_ids.csv`);
renumber only via `scripts/goal_page_reorder.py --version vNNN --date ...` (new figures need a UNIQUE placeholder number
+ data-tier first; see scratchpad page_1027f.py). Other windows: e3 (PI deck) waits for "decoder figures final"; 5b
(imaging page) uses stable ids too.

**Decisions today (DECISIONS.md / queue doc / FINDINGS 4d0-LATE..4d0j):**
- Offsets: like-for-like SAME-PLACE read-out (each lick vs the animal's pre licks that landed at the same lateral place),
  Gaussian kernel (`lick_matched_readout.py --ref gauss`, width by held-out pre-session error: 0.1 mm); 'beyond the pull'
  RETIRED (sign depended on the assumed pull slope). Secondary: per-lick error coding vs pre (same-spout, misdirected).
- PRIMARY decoder = frozen pre-stroke decoder on the animal's pre-stroke SVD basis (`WFIELD_LICK_DECODER=svd`); Allen +
  LocaNMF = basis-robustness checks; decoder r and per-session refit decoders to SVD too; section-2 / 3.19-3.22 trial
  spout classifiers (joint LocaNMF now) -> SVD (needs an SVD frozen spout decoder via frozen_models.load_or_fit).
- Mixed models: TIME model primary (shared wfield_local/mixed_models.py), epoch model as paired supplementary panel,
  behaviour-model deficit = FIRST-lick weighted signed aim (`lick_aim_deficit.py`, reliability 0.98); significant terms
  bold; joint 'any change' test + AIC shape comparison approved (not done); 'time_drift' variant = sensitivity only.
- Behaviour axis everywhere = first-lick weighted aim; far-contra first-lick aim beside far-contra-specific measures.
- Figure names lick_ / lick_iti_ / lick_cued_; every figure script to save figdata + --plot-only (helper agent).

**Findings (PROVISIONAL):**
- ITI licks read contralesional of where they land: robust ACUTELY in all bases (Allen +0.32, LocaNMF +0.46, SVD +0.20,
  4/4); later epochs basis-dependent (Allen +0.21 / +0.24 4/4; LocaNMF +0.29 / +0.25; SVD lick-scaled +0.07 / +0.14 n.s.;
  SVD UNSCALED +0.06 / +0.14 n.s. -> lick-window scaling is NOT the explanation; PS93 negative in SVD). Frame-scaled SVD run
  in progress. Not explained by the larger lick response (attribution has the opposite sign); not previous-target
  carry-over (strongest after far-IPSI trials).
- Cued: the acute contralesional read-out comes largely from MORE target-carrying misses (misdirected licks read toward
  their target, like the rare pre-stroke misses); it tracks aim (~0 / slightly ipsi at pre aim). Cortex changes
  substantially (lateral r loss, compression, far-contra map replaced acutely, bigger lick response), mostly recovering.
  First cued lick +0.22..+0.35 acute in all bases; later epochs basis-dependent.
- Decomposition: ITI = uniform contralesional shift + compression; cued = compression + small uniform ipsi shift; chronic
  far-contra ipsilesional read-out = compression at the edge.
- First-lick aim shows ~2x the chronic deficit of the all-lick aim (corrective licks hide it).
- Mixed models (first-lick aim deficit): at pre aim ITI read-out +0.15, ITI lateral r -0.21 (unrelated to aim); cued r loss
  fully explained by aim.

**RUNNING at handoff (background bash in session f082ee0c; logs in S2 = scratchpad):**
- frame-scaled SVD read-out: `WFIELD_LICK_DECODER=svd` iti_lick_order + `lick_matched_readout --ref gauss`, outputs tagged
  `_svdF` (S2/svdF.log, ends with DONE). Then run LocaNMF frame-scaled the same way (basis_locanmf/frame_sd.csv exists).
- helper agent: figdata + --plot-only for every figure script -> docs/status/FIGDATA_PLOT_ONLY_2026-10-27.md when done;
  it edits the map / check / scatter scripts -- do not touch those until it reports.

**TO DO (order; figure ids in the queue doc):** (1) read svdF -> decide SVD scaling; switch page figures to SVD + gauss
(supersede tables A / B); (2) local-linear vs Gaussian reference (held-out error); (3) four covariate references (landing /
+kin / +FR / +FR+kin; no June FR); (4) same-target same-place; (5) mixed-model pixel maps + x/y/z / angle / elevation /
reach Haufe maps (after the helper); (6) epoch panels for every LMM figure, joint test, AIC shapes, copy-link icon on the
page; (7) June SVD projection -> June-anchored drift supplement; (8) recovery link on the new measure; per-animal
change-vs-pre scatter + CI markers; first-lick row on 3.29; (9) spout classifiers -> SVD; (10) non-linear decoder;
behaviour-only (FR) decoder; then the compensation check (first-lick contact rate etc.); LOW: 2-D match, June whole-map
reference. Then FINDINGS + page synthesis with the revised cued wording; tell e3 when decoder figures are final.
Unpushed local commits since 78e6593d (pushed as 4c2da52f): push with S2/push<NN>.sh.

**PITFALLS (today):**
- Bash heredocs turn "\n" inside Python string literals into real newlines and some multi-heredoc commands fail to parse
  -> write Python with the Write / Edit tools; bit us ~15 times.
- Artifact publish: files must be under the workspace (repo); M: and some scratchpad paths are refused -> copy into
  `_goal_page/` or `_review_1027/` first.
- Push: cherry-pick named commits onto origin/main in a temp worktree; pre-push pytest ~8-11 min; the test
  `test_parallel_results_are_ordered` rejects a fan_out result used without `sorted(` within 14 lines.
- Never kill a running job (Priya runs kills); never rename on M: (write copies).
- `lick_order_preds{SRC_TAG}.csv.gz` is reused if present -> change SRC_TAG (e.g. the F suffix) when inputs change.
- `#figS-N` old-number anchors follow the LAST renumber; quote stable ids.

### 00-LATE (10-27 evening, Priya back) -- read FIRST (supersedes 00-NIGHT for state / to-dos)
**Page:** LIVE v130 (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d). v129 = REORDER: every section Key ->
Supporting -> Supplementary, STABLE ids `#f-<image stem>` (registry `_goal_page/fig_ids.csv`), old numbers kept as
`#figS-N` anchors; renumber with `python scripts/goal_page_reorder.py --version vNNN --date YYYY-MM-DD` (it rewrites all
cross-references; insert new figures with a unique placeholder number + data-tier first, see scratchpad page_1027f.py).
Maps: docs/status/GOAL_PAGE_FIG_MAP_2026-10-27_REORDER.csv, _goal_page/fig_map_20261027_reorder.csv. Other windows
(e3 deck, 5b imaging page) told and remapped; e3 waits for "decoder figures final".
**Decisions (DECISIONS 10-27 late):** z-scored where-decoder figures primary (unscaled supplementary); mixed models WITH
TIME primary (shared `wfield_local/mixed_models.py`; `scripts/iti_lmm_figures.py`, figs lmm_time_* / lmm_behaviour_*),
epoch figures supplementary; SAME-PLACE like-for-like read-out = primary offset measure (`scripts/lick_matched_readout.py
--ref place|spout|pooled`), per-lick error coding vs pre = secondary; per-session self-fit maps primary for maps; scatter
ellipses 1 SD + session-level CI markers (TO DO); rename all iti_/cued_ figures to lick_ / lick_iti_ / lick_cued_ (helper
agent running at handoff: copies on M:, scripts' savefig, page img src; fids unchanged; table
docs/status/FIGURE_RENAME_LICK_20261027.csv) + captions rewritten to same-place numbers (log
docs/status/CAPTION_REWRITE_20261027.txt).
**New scripts:** lick_matched_readout.py, lick_gain_prediction.py, iti_decoded_decomposition.py, cued_misdirected_licks.py,
iti_lick_order.py, iti_lmm_figures.py, iti_session_cluster_tests.py (TFCE, session-level), goal_page_reorder.py.
**Results:** FINDINGS 4d0-LATE (synthesis), 4d0e-4d0j; review list REVIEW_FOR_PRIYA_2026-10-27.md.
**TO DO (in order):** (1) after the rename agent: check its report, publish, push; (2) REDO the figures that still PLOT
'beyond the pull' with the same-place measure -- lick order first/later + slope (old 3.54 / 3.71 / 3.72), recovery link
(3.37 / 3.39 / 3.40), per-spout offsets (3.22, overcorrection 3.55), cued per-lick histogram pull triangles (3.43), LMM
read-out / lickclass / perspout panels -- old versions to Supplementary; (3) scatter ellipses: session-level CI markers;
(4) animal shapes (orofacial_cohort.ANIMAL_MARKER) on every pooled session graph (done for lick_matched_*); (5) 2-D
(lateral + extension) matched sensitivity if Priya OKs; (6) 3.70 follow-ups (per-animal self-fit redraw, cross-animal
null, global-activity confound); (7) first-lick map shape (re-extract all first licks); (8) tell e3 when decoder figures
are final. Push: `S2/push<NN>.sh` cherry-picks named commits onto origin/main in a temp worktree; pre-push pytest takes
~8-11 min (`test_parallel_results_are_ordered` guards fan_out ordering -- sort results).
**00-LATE addendum (10-27 ~15:20) -- READ WITH the queue doc docs/status/SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md:**
- Page LIVE v135. Decided: PRIMARY = SVD-basis decoder + Gaussian same-place reference (`--ref gauss`, width by held-out
  pre sessions: 0.1 mm Allen / SVD, 0.15 LocaNMF); Allen + LocaNMF = basis-robustness checks; mixed models (time primary,
  epoch supplementary; behaviour deficit = weighted signed aim) on the same measures; significant terms bold.
- BASIS CHECK (same-place gauss, change vs pre): all bases agree ACUTELY (ITI +0.20..+0.46, cued first +0.22..+0.35,
  4/4); later epochs basis-dependent -- ITI chronic Allen +0.24 (4/4), LocaNMF +0.25 (3/4), SVD +0.14 n.s.; cued first
  chronic Allen +0.15, LocaNMF 0.00, SVD +0.06. OPEN: SVD features are z-scored by their SD over the session's LICKS
  (Allen by the SD over all FRAMES) -- the larger post-stroke lick response inflates the lick SD and may divide part of the
  effect out. Running: `WFIELD_LICK_DECODER=svdraw` (unscaled SVD), log S2/svdraw.log. Next: frame-level SVD SD (needs
  the session's SVD time courses), then decide the SVD scaling before switching the page.
- FIRST-LICK weighted aim (recovery_measure_first_vs_all_sessions.csv): split-half reliability 0.98 (all-lick 1.00),
  r = 0.94 with the all-lick measure, chronic deficit about 2x larger (PS94 -12.1 vs -5.8 deg) -- awaiting Priya's OK to
  switch the behaviour axis + LMM deficit to it.
- Drift: `mixed_models` model "time_drift" (time model + linear day term): day-1 step z unchanged, recovery terms trade
  off with drift -> sensitivity only. Pre-only drift extrapolation: ITI read-out trend is the opposite sign; r trend same
  sign but wide CI. June-anchored drift supplement queued (June has 3-D landing; NO June FaceRhythm).
- Approved and queued: x/y/z + angle + elevation + reach Haufe maps (pooled + per spout, epoch and epoch - pre), PIXEL
  mixed models with TFCE, four covariate references (landing / +kin / +FR / +FR+kin), local-linear vs Gaussian
  reference, per-animal change-vs-pre scatter, first-lick row on 3.29 (3.60 done in the script).
- Helper agent running: figdata + --plot-only for every figure script (docs/status/FIGDATA_PLOT_ONLY_2026-10-27.md when
  done); do not edit the map / check / scatter scripts until it reports.
- Parallel session loading in `iti_where_decoder.load` (fan_sessions); basis frame ranges cached (basis_frame_range/).

**QUEUE after the SVD switch (Priya 10-27 late, in this order):** (a) SVD-basis decoder = primary (chain S2/run_svd.sh,
env WFIELD_LICK_DECODER=svd; compare with the Allen-area version, then switch page figures); (b) Gaussian-kernel
same-place reference (width chosen by leave-one-pre-session-out prediction error, floor = landing noise; sensitivity
0.1-0.4 mm); (c) landing + FaceRhythm reference (kernel on landing + linear FR terms; also + kinematics); (d) same-target
same-place comparison ('more misses' vs 'each lick read differently'; lower priority but BEFORE the non-linear decoder);
(e) redo recovery-link (3.4 / 3.41 / 3.42) on the new measure (the behaviour-model figures already are the read-out vs
aim regression); (f) angle-matched secondary + coordinate family (lateral / sagittal / angle / elevation / reach / xyz);
(g) non-linear decoder (low priority). Mixed models now: like-for-like read-out, behaviour deficit = angle aim
(`iti_lmm_figures.py --deficit aim`), epoch model drawn by `mixed_models.plot_epoch`, significant terms bold.
**Pitfall:** python heredocs in bash turn "
" in string literals into real newlines -- use Write/Edit for Python.

### 00-NIGHT (10-27) -- read FIRST
- Page v124 (figs through 3.70; source backed up). Running list of results / decisions / open questions for Priya:
  `docs/status/REVIEW_FOR_PRIYA_2026-10-27.md` section "AFTERNOON / EVENING RUNNING LIST" -- present it all when she is back.
- Decisions today: DECISIONS.md 2026-10-27 (decoder r + offsets LATERAL primary; behaviour ANGLE as change vs pre; no
  extension filter; max-stat + FDR contours). FINDINGS 4d0 (synthesis), 4d0b (far-contra read-out vs landing), 4d0c
  (confound ledger, checks done), 4d0d (first lick read toward the target).
- New scripts today: iti_session_patterns / iti_session_pattern_tests (per-session maps, whole-map permutation, drift
  floor), iti_first_lick, iti_overcorrection [--angle], iti_recovery_measure, iti_where_angle_reach (polar), iti_check_figures
  (coord_table: lateral / angle / x-y-z), iti_hemo_check (agent), iti_mixed_models, iti_spout_animal_panels,
  iti_extension_dist, iti_june_ref + iti_where_drift --june, iti_where_decoder / iti_where_scatter --z (agent).
- Pitfall: pandas `T.map` is the DataFrame METHOD -- use T["map"] (silently broke subtitles once).

### 00-AFTERNOON (10-27) -- read FIRST
- Page v122 (figs 3.39-3.51 appended after 3.38; 3.37 now draws pre sessions; "Where things stand" first row =
  Priya's synthesis, FINDINGS 4d0). Source backed up to M:/.../encoding_goal_page_src.
- New scripts: iti_check_figures.py (per-lick error histograms ITI + cued with landing row, pull-only control, F0,
  3-axis, per-session courses; cued figures named cued_check_*), iti_decoder_pattern_stats.py / iti_spout_pattern_stats.py
  [--iti] (held-out per-session Haufe maps, pixel BH, >= 3/4), iti_pattern_area_stats.py (Allen areas), iti_june_ref.py +
  iti_where_drift.py --june, iti_recovery_link.py --plot-only (pooled + session scatter).
- PENDING to page: 3.35 stats (iti_spout_pattern_stats.png) + ITI version of 3.35 (iti_spout_pattern_stats_iti.png) +
  area-stat figures (iti_pattern_area_stats_{position,spout,spout_iti}.png) -- were re-drawing with the 3/4 rule.
- Priya to decide: switch pixel stats to the decoder page's FWER max-stat for consistency? z-scored pattern-map check?
  behaviour-only decoder (approved, low priority, not started).
- PITFALL again: Python heredoc replacement strings with 
 become real newlines -> fix with Edit (hit 5x today).

### 00-OVERNIGHT (10-27 ~04:00, autonomous while Priya slept) -- read FIRST
- REVIEW LIST for Priya: `docs/status/REVIEW_FOR_PRIYA_2026-10-27.md` (key results, questions, issues). Page v121.
- DONE overnight (all in FINDINGS 4d, scripts committed, pushed through d6ebdb1 + this batch):
  spread-matched r (`iti_where_spread_matched.py`), recovery link on the z-scored decoder + per-animal figure
  (`iti_recovery_link.py --beh angle --z`, page 3.37), far-contra pattern test (`iti_farcontra_pattern.py`, 3.38),
  drift vs day gap (`iti_where_drift.py`, 3.36), Haufe position maps (`iti_decoder_patterns.py`, 3.33 / 3.34), per-spout
  one-vs-rest maps (`iti_spout_patterns.py`, 3.35), F0 check (`iti_f0_check.py`), three-axis decoder
  (`iti_where_3axis.py`), significance bars on 3.14 / 3.23 / 3.24 / 3.26 / 3.31. New "Where things stand" row on the
  lateral r loss. Page figures 3.33-3.38 APPENDED (no renumbering; fig maps updated).
- NOT DONE / blocked: jaw per far-contra lick (licks.csv.gz has no per-lick jaw; jaw_trials is per trial -> needs the pose
  window); 415 nm lick response (haemodynamic arm closed per CLAUDE.md -> asked Priya); RMSE bias/SD split, ALM/tjM1
  non-overlapping ROIs (optional); older 00f item 7 items untouched.
- iti_where_decoder.coords now returns 5 columns (+ down_dv at index 4); load() also stores `<set>_pos` per lick.
- Pitfall hit 4x tonight: python heredoc replacement strings containing "\n" write a REAL newline into the file ->
  use the Edit tool for any string with 
.

### 00-NOW (10-27 ~00:30) -- read before 00a-00g
- Sagittal chain DONE (22:56); page v119 PUBLISHED with the sagittal figures (3.15-3.22 redrawn + recaptioned; scatter
  y = 0 line removed; decoder suptitle gap fixed) and the NEW KEY row (Priya liked: cued licks read AHEAD of the failing
  movement acutely; "as aim recovers, the ITI contralesional offset fades"). Z-SCORED features = PRIMARY readout in the
  KEY row for BOTH axes (Priya 10-26: OK for sagittal; lateral agrees either way); figures 3.15-3.22 still show
  unscaled (3.22 links the z version). Page src backed up; push of b5de1e7..HEAD queued (push41).
- Recovery link: `scripts/iti_recovery_link.py --beh {hit,angle,lateral}` (outputs recovery_link_*_{tag}.csv); hit rate
  saturates -> angle is the recovery measure (chronic PS92 1.01 / PS93 0.77 / PS94 0.13 / PS95 0.89). Priya says PS95
  recovered best -- ask which measure ranks PS95 first.
- TO-DO (low priority, Priya 10-27): 3-AXIS version (lateral, outward = forward, down) in one run, z-scored; sagittal
  ~ |down| (r 0.96), outward independent (r -0.05). Then: spread-matched r for lateral AND sagittal, projection on the
  pre far-contra pattern axis, jaw per far-contra lick. sabatinilab-e3 not yet told the figures are final.

### 00d. Results this session (all PROVISIONAL; FINDINGS 4d (xiii), ENCODING_FINDINGS 8.17 KEY block)
- **KEY (page "Where things stand"): cortex read with the frozen pre-stroke decoder places self-initiated (ITI) licks
  further CONTRALESIONAL, and cued licks CLOSER, than where they land.** Measure = decoded minus where the animal's
  unchanged pre-stroke pull toward the middle would put the lick (`iti_where_beyond_pull.py`; pull slope lateral
  0.31-0.65, outward 0.13-0.36). ITI lateral +0.34 / +0.20 / +0.23 mm (acute 4/4); cued outward -0.30 every epoch
  (0/4 further). Confirmed on licks at their usual (pre) spot (no pull model) and with within-session z-scored features
  (ITI +0.33 / +0.22 / +0.25, 4/4 every epoch; cued -0.22 / -0.21 / -0.17).
- **Per spout (3.22; ITI by PREVIOUS trial's spout):** ITI contralesional after every spout acutely, chronically after
  far-ipsi / far-mid trials; cued closer at every spout; chronic cued FAR-CONTRA licks read IPSIlesional (-0.24, 0/4
  contra) -- not the pull: usual-spot -0.22 / -0.46 / -0.17 (PS92 / 93 / 95; PS94 none), no mirror shift at far ipsi.
  23:10 far-contra decomposition -- CORRECTED 23:40 (Priya). Kept: ACUTE cued far-contra decoded is contralesional of
  the (ipsi-shifted) landing BEYOND the pull in 3/4 (+0.46 / +0.21 / +0.68, PS95 -0.05); CHRONIC read ipsi of the
  landing (sagittal run: -0.24 [-0.35, -0.12], 0/4 contra); chronic lateral landings near pre (1.26 / 1.18 / 1.27 vs
  1.36 / 1.43 / 1.43; PS94 0.13 vs 1.02); within-spout lateral r ~0. WITHDRAWN: (a) "decoded on the spout's side" --
  the decoder never sees the spout, and pre landings cluster MEDIAL of the spout (side_ml is the tongue tip, not the
  spout); say "contralesional of the landing". (b) the "smaller decoded size" / "acute size jumps" -- size was
  measured from the MOUTH in lateral x forward (out_ap) only; forward is not protrusion (licks run mostly down): on
  SAGITTAL extension acute far-contra protrusion is LOWER (4.69 / 4.23 / 4.57 / 4.03 vs pre 4.86 / 4.32 / 5.50 / 5.01).
  (c) the pull shrinks toward the mean of ALL the animal's pre licks (one decoder, spout unknown; centre lateral -0.11
  / +0.38 / -0.30 / +0.07, sagittal 4.15 / 3.97 / 4.56 / 4.54 mm), not toward the mouth or each spout's mean -> only
  beyond-pull per axis is a valid offset. Interpretation open (downstream gain / changed strategy / degraded far-contra
  pattern); tests: projection on the pre far-contra pattern axis, jaw per lick on far-contra.
  RECOVERY: PS95 recovered best, PS94 least (Priya); far-contra hit rate saturates chronically (1.03 / 1.00 / 1.01 /
  1.01) -> use tongue angle / lateral protrusion as the recovery measure (`scripts/iti_recovery_link.py --beh`).
- **r is unaffected by offsets / pull** (correlation ignores shift + scale); the r loss is a separate finding. Part of
  the lateral r loss is a narrower spread of lick positions (fully: ITI PS92 / PS95, in-trial PS93; not ITI PS93 / PS94).
  Lateral falls more than (horizontal) angle chronically (paired, 4/4 areas). In-trial angle r above pre 4/4.
- **Not a scale artifact:** `iti_zscore_check.py` (each area / its frame-wise SD over ALL frames of the session; NOT over
  licks, NOT a pre-lick baseline). r losses unchanged or larger.
- **Per-area split** of the decoded shift (`iti_decoder_contrib.py`): many small mixed-sign contributions (ridge spreads
  credit), no single area; cued lick-window activity UP in PS92 / PS93, DOWN in PS94 / PS95 (so "more tjM1/ALM" was not
  the premise). ALM / tjM1 ROIs are excluded from decoder features because they are subsets of MOs / MOp (MOs / MOp are in).
- **MAE in mm** (3.15 panel 2; includes pull + offsets, no direction): ITI lateral +0.39 acute; outward error stays up
  chronically for both lick types (ITI +0.09, cued +0.18, 4/4).
- **Significance bars** on 3.7 / 3.8 (`iti_licks.sig_bars`; per bin paired nested bootstrap, BH-FDR q 0.05 across bins;
  lw 2.6 alpha 0.5 = the kinematics trajectory style). 3.8: ITI lateral below pre in every bin chronically; cued below
  pre only AFTER the lick chronically. 3.7: per-session ITI relative-to-previous-spout decoding ABOVE pre chronically.
- Partial-refit figures now name their model in the image title (`partial_refit.model_label`); 1.37 draws landing -
  target lick kernels (old to-do 2 done).
- **Retracted tonight:** "all-licks ITI offset = moving licks + pull" (used the overall ipsi shift; nearest-spout grouping
  makes the within-spout shift much smaller).

### 00e. Decisions (Priya, 10-26 evening)
- Unscaled features = PRIMARY; within-session z-scored = SUPPLEMENT (label z-scored figures in title, axes, caption).
- Never subtract a pre-lick baseline from decoder features (the ITI carries position information).
- Decoder axes: lateral + SAGITTAL extension (replaces outward); angle = the encoder's exec_angle definition.
- Per-spout ITI grouping = previous trial's spout (not nearest-to-landing). Session-dot layout (3.9 style) for per-spout.
- "Licks at their usual spot" (not "landing held fixed"); always compare the change vs pre (delta error).
- MAE added as a companion to r (says nothing about direction; RMSE^2 = bias^2 + SD^2 split offered, not done).
- The encoder uses CARTESIAN tongue traces (lateral, forward, 3-D extension) + a per-lick within-spout angle kernel;
  a polar encoder rerun was offered, not requested.

### 00f. TO DO (after 00c)
1. Significance bars on the remaining time courses: 3.14 basis decoder (`iti_basis_decoder.py --summarise`), ITI PETHs
   (`iti_licks.py --ramp` / `--key`), orofacial-vs-control PETHs (`iti_area_maps.py --compare`) -- reuse `sig_bars`.
2. F0 / dF check on the larger post-lick response (raw F0 per session vs days; response in dF without /F0); 415 nm
   lick-aligned response pre vs post.
3. Frozen-decoder drift vs day gap within pre (is the ITI r loss beyond elapsed-time drift?).
4. Spread-matched r for lateral and the encoder-style angle (licks restricted / reweighted to the pre spread).
5. Per-epoch decoder PATTERN maps (Haufe: cov x weights) in each animal's pre SVD basis, back-projected; split-half
   reliability; per-spout one-vs-rest patterns across epochs (Priya's "map the decoder weights per position").
6. Optional: RMSE = bias + SD split panel; non-overlapping ALM / MOs-rest, tjM1 / MOp-rest ROIs.
7. Older: jaw per ITI lick (3.27 row); stray_licks rerun after sabatinilab-5b's kinematics_all swap; PRE-level TE maps;
   epoch-kernel drift-rate test; fold scratchpad figure scripts (fig_goal_summary.py etc. in S) into scripts/.

### 00g. Pitfalls (new tonight)
- Bash heredocs / python -c: a `\n` inside a Python string literal written through the tool layer becomes a real newline
  and breaks the file (hit twice). Use the Write / Edit tools for any code containing `\n`.
- ruff blocks commits (e.g. B007 unused loop variables); fix and re-commit, then regenerate the push script (a push
  script built after a failed commit cherry-picks nothing / the wrong SHA).
- Shared-y axes: raising ylim per axis compounds; raise once (`_headroom([axs[0, 0]])`).
- Nearest-spout grouping changes group membership when licks move: never scale an overall shift into a per-spout
  prediction.
- Per-spout r is low and spread-sensitive (small within-spout range); read offsets first.
- Z-score over all frames, never over licks (that forces each session's mean lick feature to 0 and erases offsets).
- `iti_where_*` scripts now write lateral + SAGITTAL columns (act_sag / dec_sag); `_groups` still uses outward for the
  nearest-spout assignment; `iti_decoder_contrib.AXES = {"lat": 0, "sag": 3}` (coords column order: lat, out, reach, sag).

Supersedes the RUNNING / TO-DO lists of `STATUS_2026-10-22_ENCODING_HANDOFF.md` §000z. Window = session `sabatinilab-d4`
(pose window `sabatinilab-5b`, FaceRhythm `sabatinilab-e3`). Worktree `C:/Users/SabatiniLab/Github/wf_attempt` (branch
analysis/attempt-coding). Old scratchpad (this session only; a NEW session can read it by absolute path but cannot
publish from it): `S = C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad`.
Python: `C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe`, run from the worktree with
`PYTHONPATH=$(cygpath -w $PWD)` (another `wfield_local` install shadows it). ITI outputs: `<E>/iti_licks/` =
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/` (figures in `figures/`).

## 0. START HERE (19:35)

### 0a. PUBLISH THE PAGE — exact steps for a new session

**PUBLISHED 10-26 evening by a new session: live = v105** (local `_goal_page/index.html` + all 183 referenced PNGs). Two
publishes (v104 = index + 121 PNGs, v105 = the other 62): the referenced PNGs total ~65 MB, over the 64 MB per-publish
cap, so split `files` into batches under ~35 MB. Old images not read in a session block the publish ("published content
is not what you last saw"): `action: "list", scope: "files"` on the URL first counts as seeing them. Spot-checked 1.37,
3.7, 3.15-3.20, 3.25 (published byte sizes = local). The steps below are kept as the procedure for the next republish.

**10-26 late (new session):** partial-refit figures now name their model in the image title (`partial_refit.model_label`;
1.37 also draws landing - target lick kernels = to-do 2 done); 3.19 / 3.20 redrawn (legend clear, symbols spelled out);
NEW 3.21 = change in decoded - actual position vs pre (`iti_where_delta.png`) -> old 3.21-3.30 are now 3.22-3.31
(`_goal_page/fig_map.csv`; kinematics control = 3.26). Signed-error / outward results: ENCODING_FINDINGS 8.17.
Pending Priya: KEY row wording (ITI-only contralesional reading), spread-matched lateral / angle check (angle here =
horizontal atan2(lat, out), not the encoder's exec_angle), relabel ITI / in-trial rows in 3.15-3.18.

**10-26 later (live v112):** KEY row = beyond-pull offsets (ITI read contralesional, cued read closer), robust to
within-session z-scoring (`iti_zscore_check.py`); per-spout version = NEW 3.22 (`iti_where_per_spout.py`) -> old
3.22-3.31 are now 3.23-3.32 (`_goal_page/fig_map.csv`). Per-area split (`iti_decoder_contrib.py`): no single area.
DONE 10-26 late: per-bin significance bars on 3.7 / 3.8 (`iti_licks.sig_bars`, `--plot-direction`,
`--plot-frozen-tc`; tables sig_bins_3_7/3_8.csv); MAE panel in 3.15; z-scored figures labelled. STILL TO GET BARS:
3.14 (basis decoder), the ITI PETH figures, orofacial-vs-control PETHs. NEXT: lateral + SAGITTAL extension
(sqrt(forward^2 + down^2)) replacing outward in 3.15-3.22 (Priya 10-26), z-scored kept supplemental. Then: F0 / dF check on the larger response; frozen-decoder drift vs day gap within pre; spread-matched
r (encoder-style angle); per-epoch decoder PATTERN maps (Haufe) in the SVD basis; 3.7 / 3.8 significance bars.

The Artifact tool was switched off in this session from ~14:15, so everything since v103 was LOCAL ONLY until v105.
- Live: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d = **v103** (12:00). Everything after it exists only locally.
- **Local source (newest, complete): `C:/Users/SabatiniLab/Github/wf_attempt/_goal_page/index.html`** (+ its 190 PNGs;
  untracked, listed in `.git/info/exclude`; 63 MB). Backup copy: `M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/`.
  The old `S/goal_page/` was identical at 19:30.
- What the local page adds over v103:
  - a TITLE above every figure (`<div class="ftitle">`, also used by the sidebar; source `_goal_page/fig_titles.csv`);
  - section-3 captions open with "Decoder: frozen / per-session" and "Measured: r vs dF/F";
  - 3.9 caption carries the paired ITI vs in-trial test;
  - new figures: 1.37 TE partial refit, 3.11 lick rate, 3.14 basis decoder, 3.15 where decoder, 3.16 lateral x bases,
    3.28 orofacial vs control PETHs; supplements 3.17 / 3.18 = the where decoder (every target) in the joint LocaNMF
    and pre-stroke SVD bases (`iti_where_decoder_locanmf.png`, `iti_where_decoder_svdbasis.png`);
  - corrected whole-cortex claim; 3.25 kinematics control with a face-adjusted row;
  - all section-3 figures redrawn: frame / measure labels, stars above data, SEM bands on 3.7 / 3.21 / 3.24,
    3.15 split in two.
- **Step 1 — DONE 20:40 (chain finished 19:33, all images copied into `_goal_page` + backup). Only needed again if figures are redrawn later; check `S/relabel.log`
  ends with `relabel3 done`):
  ```bash
  F="M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/figures"
  G=/c/Users/SabatiniLab/Github/wf_attempt/_goal_page
  for n in iti_basis_decoder iti_area_maps iti_area_peth iti_area_groups iti_area_compare iti_where_decoder \
           iti_where_decoder_bases iti_where_decoder_locanmf iti_where_decoder_svdbasis iti_frozen_timecourse iti_frozen_summary iti_ramp_sessions iti_ramp_summary \
           iti_ramp_peth iti_ramp_peth_isolated iti_kinematics_control; do cp "$F/$n.png" $G/; done
  ```
- **Step 2 — read the live page first.** Call Artifact with `action: "read"` and the URL. A publish to an artifact the
  session has not read is refused. Confirm the live page is our v103: its 3.x sections end at the whole-cortex
  supplement, and it has no `ftitle` divs. The local file is a strict superset of v103: no one edited the live page
  after 12:00.
- **Step 3 — publish** with ALL changed images in `files` (publishing index.html alone does NOT upload PNGs; v98-v102
  showed 9 broken figures that way):
  - `action: "publish"`, `url`: the page URL above;
  - `file_path`: `C:/Users/SabatiniLab/Github/wf_attempt/_goal_page/index.html`;
  - `root`: `C:/Users/SabatiniLab/Github/wf_attempt/_goal_page`;
  - `files`: every PNG in `_goal_page` modified after 2026-10-26 12:00, each mapped to itself. Generate the map with:
    ```bash
    cd /c/Users/SabatiniLab/Github/wf_attempt/_goal_page && find . -maxdepth 1 -name "*.png" -newermt "2026-10-26 12:00" \
      | sed 's|^\./||' | awk 'BEGIN{printf "{"} {printf "%s\"%s\": \"%s\"", (NR>1?", ":""), $0, $0} END{print "}"}'
    ```
    About 25 files. Expect at least: te_partial_refit_lines, iti_area_compare, iti_basis_decoder, iti_where_decoder,
    iti_where_decoder_bases, iti_lick_rate, iti_direction_decoding, iti_frozen_timecourse, iti_frozen_summary,
    iti_ramp_sessions, iti_ramp_summary, iti_binned_sweep, iti_binned_summary, iti_recovery_marker,
    iti_kinematics_control (+ the step-1 files).
  - `label`: e.g. "Titles, ITI labels, where + refit".
  - If the refusal hands back the live version, check it is v103 and publish the local file again. It is the merge,
    since local = v103 + our edits.
  - Then open the published page and spot-check 3.7, 3.15, 3.16, 3.25 and 1.37 for images.
- Numbering: **Priya may quote LIVE (v103) numbers.** Local numbers moved after several inserts: her "3.17" on
  10-26 evening was the kinematics control, which is 3.25 locally. Map with `_goal_page/fig_map.csv` (last renumber)
  and `docs/status/GOAL_PAGE_FIG_MAP_2026-10-26.csv` (pre-10-26 → current).

### 0b. RUNNING (background in THIS session; may die with it)
- **Redraw chain** `S/run_relabel.sh`, log `S/relabel.log`, started 18:45. Steps:
  1. plot-direction / key / binned / recovery marker: DONE;
  2. basis decoder `--summarise` was running at 19:30;
  3. then `iti_area_maps.py`, `iti_area_maps.py --compare`, `iti_where_decoder.py` (full, ~25 min);
  4. then `iti_licks.py --key` again (3.8 / 3.9 / 3.22 with the star fix);
  5. then `iti_licks.py --ramp` (SEM bands on 3.21 / 3.24).

  If the log does not end with `relabel3 done`, rerun the missing steps from the worktree with the PYTHONPATH above.
  Each step redraws from saved session data; nothing is lost. Then do 0a step 1.

- **DONE 20:10: decoded vs actual landing** (`scripts/iti_where_scatter.py`, per-lick `where_preds.csv`; LOCAL page
  3.19 in-trial / 3.20 ITI). Rig names in `tr_pos` (close_L ...) are mapped to lesion-relative names (all lesions LEFT).
  Ipsilesional-bias test (Priya): `where_bias_sessions.csv`, ENCODING_FINDINGS 8.17 top. Executed licks shift ipsi
  4/4; the decoded ITI position does NOT follow (bias at fixed actual position +0.18 mm contra chronic, 0/4 ipsi;
  gain lower) -> not an ipsilesional decoder deviation.
- **DONE 20:10: full movement regression** for the post-lick response (kinematics control, LOCAL 3.25 row 5: 3-D tongue
  position, reach, speed, duration, retraction + FaceRhythm post): response survives (tjM1 contra chronic +0.86).
  **JAW still missing:** jaw kinematics exist per trial (`kinematics_all/jaw_trials.csv.gz`), not per ITI lick ->
  needs a per-lick jaw extraction from the whole-session 3-D jaw trace (ask the pose window / see pose handoff 10-21),
  then add to `_more_kin` in `iti_licks.py` and rerun `--controls`.
- **Helper agents launched ~19:45 (results may land in this session):**
  - (1) TITLE VERIFICATION: DONE 20:45 and APPLIED to `_goal_page/index.html` (details
    `_goal_page/title_verification.csv`; 46 figures, 45 high confidence).
    - Corrected: 1.25 / 1.26 / 1.27 = frozen encoder WITHOUT FaceRhythm (run `..._v7g3_t3d_basis`, md5-matched), not the
      lick-kernel run.
    - Clarified: 9.13 / 9.14 = PER-SESSION FaceRhythm encoder (`combined_encoding` fr_A_tr, 10-fold within session).
    - Confirmed: 3.1-3.6 frozen pre-stroke decoder (`attempt_coding.py`); 1.15 / 1.16 / 1.24 frozen lick-kernel run with
      FR; 1.18 frozen without FR; 1.21-1.22 per-epoch pooled refit (`epoch_kernels.py`); 1.28 self-fit minus frozen;
      1.44-1.47 per-session encoder; 7.1-7.6 all-session shared encoder (`allsession_encoder.py`; the page may not
      define "Layer 2"); spot checks in sections 5 / 6 / 8 correct.
    - Writer not found: 6.1 (`fig7_residual_far_contra_16px.png`; frozen frame supported only by ENCODING_FINDINGS
      8.4).
    - Many page PNGs were drawn by scratchpad scripts outside the repo (`fig_goal_summary.py`, `fig_pooled.py`,
      `fig_layer2.py`, ...; in S). Folding them into scripts/ is a to-do.
  - (2) PUBLISH by a subagent: FAILED 19:50 -- subagents in this session do not get the Artifact tool either. Publish
    from a NEW session via section 0a (the page is still v103).

### 0c. WAITING ON OTHERS
- **sabatinilab-5b:** "go" for the kinematics_all swap sent 15:50; Priya runs the rename lines 5b gives her. When the
  table is live, re-run `python -m wfield_local.stray_licks`. Move the old `<E>/stray_licks/sessions` aside first; it
  skips sessions already done. These are derived outputs, so no approval is needed for that rename. Then update
  ENCODING_FINDINGS 2.6.

### 0d. TO DO (in order)
1. **Per-time-bin significance bars on the time-course figures** (Priya 19:30: "for the post-lick response curves and
   decoder curves -- significance testing and mark with lines (colored per epoch) where in the time bins curves
   differ from pre"). Plan, agreed in chat:
   - one helper in `scripts/iti_licks.py`, e.g. `sig_bars(ax, X_by_animal_epoch, lags, colors)`;
   - for each post epoch, `ak.boot_delta_columns(post, pre, rng)` with `{animal: array (n_sessions, n_lags)}`. This is
     the same paired animals → sessions draw as every dot-plot star, vectorised over time bins;
   - then `ak.boot_p_columns(draws)` → `ak.bh(p, q=0.05)` across the bins of that curve;
   - draw contiguous significant runs as thick horizontal segments in the epoch colour (`EPOCH_LINE`) in a strip below
     the curves, one row per post epoch (x in data, y in axes: blended transform);
   - say "paired nested bootstrap per time bin, BH-FDR q = 0.05 across bins" in the suptitle and caption.

   Figures and their data:
   - 3.7 left / middle: per-session decoder per lag, `decode_sessions.csv` (label, animal, epoch, set, target, lag, r)
     → redraw via `--plot-direction`.
   - 3.8: frozen decoder per lag, `decode_frozen_lags_sessions.csv` → the `key_figures` block (`--key`).
   - 3.14: basis decoder time courses (`iti_basis_decoder.py --summarise`).
   - 3.21 / 3.24: `ramp_figure`, animal-mean PETH normalised by pre amplitude. Test on per-session PETHs (same
     normalisation) → `--ramp`.
   - 3.22: `iti_ramp_sessions.png`, raw + normalised rows, per-session PETHs → `--key`.
   - 3.28 orofacial vs control PETHs (`iti_area_maps.py --compare`); optionally 3.27.

   Then update captions and republish.
2. TE partial-refit line figure (page 1.37): the title still names only the target null. Add the landing − target
   lick-kernel contrast, which is significant but not drawn (`boot.csv`).
3. 3.9 caption: add the lick-count control line (matched 40 licks / session: same numbers; `iti_nmatch_check.py`).
4. stray_licks re-run (0c).
5. Older open items: PRE-level TE group-kernel maps (replace 1.20-1.21); epoch-kernel drift-RATE test (8.18);
   every-lick window (Priya to confirm first-lick primary); June→August drift ceiling; lesion burden /
   leave-PS93-out; share deletions / nokin retirement (Priya decides).
6. **Push**: origin has everything through 66a287e. Pushed through b644af5 (20:05). Later local commits need a cherry-pick push:
   ```bash
   git fetch origin; git worktree add --detach ../wf_push_tmpNN origin/main; cd ../wf_push_tmpNN
   git cherry-pick <commits>; git push origin HEAD:main   # retry: fetch + rebase origin/main
   ```
   The pre-push tests take ~7–14 min. Never rebase the local branch, never force-push. `S/push28.sh` is the template.

### 0e. Results since 13:50 (PROVISIONAL; ENCODING_FINDINGS 8.17 / 8.20, FINDINGS 4d vii-x)
- **ITI lick rate (BEHAVIOUR):**
  - PS92-94 drop acutely and stay 14-28 % below pre chronically.
  - Only PS95 ends above pre: about 4×, from near-zero pre ITI licking.
  - Rate tracks far-contra hit rate within animal (r +0.67; +0.51 with the log-day trend removed, PS94 ~0).
  - So rate is not the cause of the decoder's ITI loss.
- **Where decoder:**
  - The ITI loss is LATERAL, in every basis (−19 to −20 %, 4/4); outward position is weakly decoded and unchanged.
  - In-trial: outward drops acutely and recovers; angle ends above pre (+7 to +11 %).
- **ITI vs in-trial, paired test (3.9):** chronic −0.062 [−0.110, −0.014], 3/4 animals (PS93 0.00). Say "in three of
  four animals", not "confidently".
- **TE partial refit (1.37):**
  - Landing-labelled minus target-labelled lick kernels: lower chronically in every area (contra SSp-m −0.019, cortex
    −0.024; 0/4 up).
  - Acute target advantage falls (−0.126), back by chronic.
  - Same without PS95_0907.
- **Kinematics control + FACE (3.25 row 4):** the larger post-lick response survives regressing out reach + speed +
  post-onset FaceRhythm (tjM1 contra chronic +0.74 [+0.22, +1.26], SSp-m +0.63 [+0.14, +1.11]); unchanged from reach
  + speed alone.
- **Ipsilesional bias? (3.19 / 3.20):** no. Executed licks shift ipsi in every epoch (4/4); decoded ITI positions stay
  near pre, so the bias at fixed executed position is CONTRAlesional (+0.32 acute, +0.18 chronic mm) and the ITI gain
  is lower. In-trial: no consistent bias.
- **CORRECTION: in-trial r is lower too** (-0.146 acute -> -0.063 chronic, CI excludes 0), smaller than ITI; it follows
  lick accuracy (target-only reader -0.246 -> -0.036; session corr 0.82). Never say the change is ITI-only; say ITI
  loss = remapping, in-trial loss = mostly deviated licks read as on-target. Page captions 3.9 / 3.10 may still say
  'in-trial recovers' -> soften to 'smaller and shrinking' (to do).
- **Ipsi-deviated in-trial licks look like on-target licks** (more so after the stroke): decoder follows 26 % of a
  within-spout deviation pre, 8 % acute, 14 % chronic (4/4 lower); spout layout unchanged (ENCODING_FINDINGS 8.17 top).
  Candidate figure for the page (not drawn yet): decoded diff vs actual diff per epoch.
- **Full tongue + face regression (3.25 row 5):** the larger post-lick response survives (tjM1 contra chronic +0.86
  [+0.24, +1.55]). Jaw not yet included.
- **Decoder AUC (Priya):** answered "keep the time-binned decoder (3.12-3.13) as the one number per session"; the area under
  the per-lag r curve double-counts overlapping lags; ROC-AUC needs a binary target (an ipsi-vs-contra AUC was offered
  as an optional supplement).
- **Decoder r vs AUC:**
  - r = information about lick position (decoded vs actual lateral tongue-tip position).
  - AUC = amount of activity (dF/F integrated over a window); not ROC.
  - They can move independently: post-stroke AUC is up, frozen ITI r is down.

### 0f. NEW PITFALLS (evening)
- **The Artifact tool can be switched off mid-session.** Keep the page source somewhere a new session can publish from
  (now `_goal_page/` in the worktree). Pass every changed PNG in `files`.
- **Regex over the page:** `<figure id="...">.*?X` with re.S spans ACROSS figures. This hit Fig 3.1, whose title and
  caption were overwritten, then restored. Match inside one figure: `<figure id="(fig[^"]+)">(?:(?!</figure>).)*?X`.
- **Escapes in Bash:** a `\\n` inside a Bash heredoc reaches Python as a real newline (the tool layer unescapes it).
  Build backslashes with `chr(92) + "n"`, write the script with the Write tool, or use Edit.
- **`renumber_sections.py`** defaults to `goal_page/index.html` next to the script. In `_goal_page/`, pass the
  index.html path explicitly.
- **Background jobs:** a script launched with `&` inside a run_in_background shell gives no completion notice (wait on
  a log line instead). Define every variable (`S=...`) inside chain scripts; one chain ran 8 steps with `$S` empty.
- **Editing a script while a chain runs:** a step already started keeps the old code. 3.8 / 3.9 were redrawn before the
  star fix, hence the queued `--key` rerun.
- **Old-design vs new-design figure names:** `partial_refit_lines.png` = OLD design (1.38);
  `te_partial_refit_lines.png` = TE design (1.37).

# EARLIER 10-26 (history; superseded by section 0)

## RUNNING as of 16:20 (history)

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
- **Page publish pending**: files = iti_area_compare.png, iti_basis_decoder.png, iti_where_decoder.png, iti_lick_rate.png, iti_direction_decoding.png (3.7 now with SEM bands), iti_where_decoder_bases.png (new 3.16) + every section-3 PNG redrawn 10-26 evening (frame + measure labels, stars above data, SEM on 3.21 / 3.24); every figure now has a title (.ftitle, fig_titles.csv in the scratchpad),
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

- **Self-initiated ITI licks (KEY, page 3.7–3.30):** within-session direction decoding intact; FROZEN pre-stroke decoder
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

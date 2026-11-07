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

00. **11-05 (window e0, at a8's request "run now whatever can be run").**
   * DONE: FR SPOUT-IN-MESH CONTROL (`scripts/fr_spout_mesh_control.py`, ~3 h at 3 jobs; outputs readouts/master_pre_v1/
     spout_mesh_control/; reduced projections cached in C:/Users/SabatiniLab/wfield_cache/fr_spout_mesh_control/). The
     ENL face decode does NOT come from the spout (FACE_RHYTHM_FINDINGS 7, FINDINGS 13). Not yet on the FR page (one
     figure + one sentence in section 6; the page is published from the HMS account).
   * READY, PRIYA SUBMITS: >= 500 permutations -> `run_fitcohortnull500.sbatch` (share copy of the set dir; new conda
     module, --n-perm 500, array skips the 600 done). Matters for the specificity claim (several of the other ten sit at
     the p .0099 floor), not the lick pair. Then rsync own_epoch_cohort_null/ back + `face_rhythm_cohort_null.py`.
     11-07 (a8) SUBMITTED + RUNNING; INTERIM SCORE on the first copy-back (acute 482 pairs; subacute / chronic still the
     original 100): readouts/master_pre_v1/epoch_shape_cohort/interim_500perm_20261107/ (page outputs one level up,
     n = 100, untouched; scored by importing compute(), no code change). Lick pair unchanged (lick_right angle_spatial /
     angle_spectral / asym / side_spec p .002-.004, Holm < .01; lick_left n.s.). SPECIFICITY WEAKENS ACUTELY: off the
     floor, of the other ten, Holm < .05 for angle_spatial sniff_prelick, nose_prelick_slow, lick_ongoing_7Hz,
     lick_fast_close, whisk_run; angle_spectral lick_fast_close; asym lick_ongoing_7Hz, lick_fast_close, lick_slow_late.
     NOTE sniff_prelick == nose_prelick_slow and run_onset_jaw == prep_jaw give identical spatial readouts (two master
     factors matched to one fit column) -- also in the n = 100 run; check before quoting them as separate factors.
     CHECKED 11-07 (Priya asked): master spatial twins -- sniff_prelick / nose_prelick_slow and run_onset_jaw / prep_jaw
     spatial cos .99 (differ only in spectrum), and a jaw cluster (lick_ongoing_7Hz, run_onset_jaw, lick_fast_close,
     prep_jaw, lick_slow_late) at spatial cos .91-.99. Spatial-only best matching (MATCH_ON spatial, ASSIGN best) cannot
     separate them: sniff / nose share a column in 98 % of null fits, the jaw pair 47 %, lick_right / lick_left 0 %.
     Re-score with MATCH_ON product (interim folder, cohort_null_productmatch.csv; figure master_factor_twins.png):
     lick pair identical; sniff_prelick drops out (p .48); lick_ongoing_7Hz + lick_fast_close spatial angle hold;
     asym for lick_ongoing_7Hz / lick_slow_late and whisk_run spatial angle lose Holm. Method choice = Priya's
     (10-25 decision was spatial best-match); asked her 11-07.
     11-07 Priya: compare best vs Hungarian (spatial score) -> cohort_null_hungarian.csv: lick_right identical; Hungarian
     re-creates the 10-25 lick_left problem (acute lick_left .86 -> .79, results flip) and forces poor twin matches
     (nose_prelick_slow .51-.61) -> Hungarian rejected again. Priya then proposed: spatial score, frequency only to break
     near-ties. Tested (tie margin .02 / .05 / .10; .05 scored -> cohort_null_tiebreak05.csv): 10 matches change across
     the 4 cohort fits; figure match_changes_tiebreak05.png (maps + spectra per change). 8 / 10 clearly better (spectrum
     ~ master, spatial within .01-.05); acute lick_fast_close + acute / subacute prep_jaw: neither pick matches the
     master spectrum (factor likely not recovered there). lick_right unchanged. Awaiting Priya's read of the figure.
     RECOVERY (Priya 11-07: mark poor fits, the factors that DISAPPEAR are interesting; combine similar footprints?):
     match_quality_tiebreak05.csv. Null fits (should hold every factor): spatial 1st pct .84, freq 5th pct .46 overall,
     but per factor run_onset_jaw .26, prep_jaw .33, sniff_prelick .46 (unstable even pre; the real COHORT_pre already
     loses run_onset_jaw, freq .25). Rule tried: recovered = spatial >= .8 and freq >= .5. ACUTE loses lick_ongoing_7Hz
     (freq .50, borderline) and lick_fast_close (.46): lost in 0-1 % of null fits -> a real acute loss of the fast-lick
     jaw rhythm, back by subacute. prep_jaw subacute loss = in 21-34 % of null fits -> not a stroke effect. Proposal to
     Priya: merge only the unstable twins (sniff_prelick + nose_prelick_slow = prelick nose; run_onset_jaw + prep_jaw =
     slow jaw); keep the stable fast jaw factors separate (merging the whole jaw cluster would hide the acute loss).
     11-07 Priya "yes" -> CODE (ed350a43): `--match-rule v2` on face_rhythm_epoch_shape / drift_floor / cohort_null
     (v1 default, reproduces old numbers; v2 -> <folder>_v2/) + NEW scripts/face_rhythm_recovery.py (factor loss vs the
     permutation null). RUN on interim null (acute 497, sub / chronic 100): epoch_shape_v2/, epoch_shape_cohort_v2/.
     RESULT: acute LOSES lick_fast_close (p .002, Holm .004; 1/4 animals) and lick_ongoing_7Hz (p .014, Holm .014;
     2/4); back by subacute. v1 -> v2 diff (scratchpad v1v2_diff.txt, rerun scripts there): lick_right IDENTICAL in
     cohort / null / drift; per-animal acute lick pair 4/4 -> 3/3 (PS95 acute not recovered at the stricter threshold),
     so per-animal mean asym change acute lick_right -.125 -> -.053, lick_left +.144 -> +.302; cohort subacute lick_left
     loses one LOAO fit. Fast jaw factors acute move from "shape changed" to "lost". prelick_nose spatial acute p .004*
     (one unit now). chronic lick_far_nose: v1 had picked nose_steps' component (6 deg) -> v2 43 deg. Priya asked 11-07
     whether to switch page + deck to v2: answer = yes once the full null lands; page / deck text to her first.
     11-07 later: ACUTE NULL COMPLETE (500 pairs; subacute 111, chronic 100). v2 rescored (epoch_shape_cohort_v2/):
     acute Holm < .05 = lick_right (all 4 measures), prelick_nose + whisk_run + lick_slow_late spatial angle, whisk_run
     spectral; LOST lick_fast_close (p .002, Holm .004), lick_ongoing_7Hz (p .014, Holm .014). Waiting on sub / chronic.
     11-07 eve (Priya): v2 recovery spatial threshold .8 -> .7 (PS95 acute lick_right = same column as pre, spectral .94,
     spatial .78: changed, not lost; .5 / .7 identical on real fits; no null cell < .7). MAP STATS (Priya OK): cluster
     mass = the map test, max-statistic = point level, within-animal session-label shuffles (face_rhythm.stats
     map_permutation). Speed maps `face_rhythm_position_maps.py --perm 2000` (August-only pre, refreshed today): acute
     LESS movement cluster at every spout (far_R -.62, 926 pts), sub / chronic MORE at close + centre. Pooled-fit null
     maps with clusters: lick_right acute 2 clusters (604 pts; max-stat 2 pts), lick_left acute none. NEW per-session
     weighting test `face_rhythm_session_loading_perm.py` (own8 fits, v2) -> session_loading_v2/: lick_right acute loses
     upper / central weight (645 pts), gains ipsi lower cheek + jaw; lick_left roughly mirrored; max-stat 263 / 413 pts.
     CAVEAT: session shuffle = fixed effects + pre drift -> OPEN: early-vs-late pre drift control for the speed and
     session-loading tests; re-run aligned_maps (cue / lick / bout, still 10-24 baseline) with --perm. PS93: own acute fit
     runs OPPOSITE to PS92 / 94 / 95 (pre-existing right facial deficit, 10-24 flag; Priya 11-07 confirms) -> keep PS93
     in main analyses, show "without PS93" (= LOAO-PS93, retitle) as sensitivity; open: PS93 per-session check. Maps now
     drawn like the speed maps (all points coloured, red / blue rings). Rhythm-band movement maps per lick factor:
     proposed, not started. Lick-rate (ILI) figure: requested, not started.
     11-07 late (Priya "go ahead with all"): CORRECTION -- FR "pre" was August-only all along (aligned_maps cache = the
     same 31 pre sessions); position-map vs aligned-map differences = window (0..4 s RMS vs 0..2 s mean speed from point
     PCs), not baseline. DONE: shared `face_rhythm.stats.session_shuffle_test` / `drift_split` / `chain_adjacency`;
     `--perm` + `--drift-control` on position_maps, aligned_maps (cue / lick / bout -> aligned_maps_<al>[_driftpre]_perm.png)
     and session_loading_perm; spectral null clusters along frequency; "LOAO-x" shown as "without x"; consistency maps
     captioned descriptive. DRIFT: position maps none; aligned cue 3 UP clusters (far_center 374, far_R 315 + 280 pts),
     lick 1 UP (close_center 254) -> acute DOWN effects safe, chronic UP clusters need the drift caveat; session loading:
     lick_right none, lick_left 145 pts MORE ipsi-central (opposite sign / place to its acute change), also
     lick_ongoing_7Hz, nose_steps, lick_far_nose. SPECTRAL: clusters find fewer than max-stat (lick_right acute max-stat 5
     bins, clusters 0): spectral changes are narrow peak shifts -> keep max-stat for spectra (recommend). PS93 per
     session: acute 08-18 / 08-20 go WITH PS92 / 94 / 95 (r +.52 / +.34), 08-19 / 08-21 AGAINST (r -.43 / -.57) ->
     PS93 inconsistent acutely, not reversed (session_loading_v2/ps93_acute_sessions_lick_right.png).
     11-07 NIGHT -- STATE FOR THE NEXT WINDOW (decisions: DECISIONS 2026-11-07). Per-session tests = main shape tests
     (session_shape_v2/, session_loading_v2/; scripts face_rhythm_session_shape_tests.py / _session_loading_perm.py),
     all delta vs own pre, drift-controlled. RESULTS (acute unless noted): spatial angle vs own pre up for most units
     (lick_right +14.6 deg, lick_left +12.1, Holm < .05; pre drift ~0.6 deg, none Holm); asymmetry lick_right -.15*,
     lick_left +.23* (also sub / chronic); side specificity lick_right -.38*, lick_left -.31*; LOSS per session:
     lick_ongoing_7Hz -.44 (p .006, Holm .065), lick_fast_close -.21 (p .044), chronic lick_left -.28 (Holm .005, mostly
     PS93 2/12) -> per-session loss weaker than the pooled-fit loss; MAPS lick_right robust (838 / 880 scaled / 786
     unambiguous pts acute), lick_left acute untestable without ambiguous sessions (2 left); right-lick share does NOT
     explain the lick_right map change acutely (rho -.14, p .61); fragility: lick_left 7.2x, lick_far_nose 5.8x (pooled
     inconclusive); lick direction: acute right share -.16, left +.24 (p < .001). NEXT: (1) make v2 + per-session the
     default and re-render every FR figure in one pass (wait for Priya on the sub / chronic O2 null: keep or scancel);
     (2) draft FR page (fr_page_src on M:, published from Priya's HMS account) + PI deck FR slides + FINDINGS wording ->
     Priya approves before any text change; (3) rhythm-band movement maps per lick factor (proposed, not started);
     (4) lick-rate (ILI) figure for page + deck (requested, not started); (5) encoder FR regressors read proj/ (pre
     sessions slightly in-sample; proj_loso exists) -- flag only.
     11-07 LATE -- FINAL STATE. 10,000-shuffle per-session tests done, Holm within family ACROSS all epochs x
     measures (Priya); FINDINGS 5.10 (PER-SESSION paragraph), FINDINGS 0 "Where things stand, face movement",
     FACE_RHYTHM_FINDINGS 6e written. Figures: session_shape_v2/session_scalars.png (normalised, leave-one-out pre
     column) + session_scalars_raw.png (Priya: both, separate) + session_loss.png; headline map = scaled
     (session_maps_scaled_lick.png); `face_rhythm_session_shape_tests.py --plot-only` redraws. FR PAGE: a helper agent
     was updating it (new per-session section before section 7, speed-map permutation figures, "Where things stand");
     check the live page version before more edits. PI DECK NOT UPDATED: Overview_2026-11-06.pptx was open (lock) all
     evening -- when closed, update slides 83 (matching method -> v2), 85 (loading maps -> session_maps_scaled_lick),
     86 (spectral), 87 (hidden drift floor -> session_loading_driftpre), add per-session shape (session_scalars) + loss
     slides; notes Data: block per slide; Priya approves the slide text. Lick direction is LP kinematics -> not an FR
     figure. O2 sub / chronic null kept running (Priya) -> rescore pooled view when files land (descriptive only).
     11-07 EVENING: PAGE v40 + DECK (198 slides) UPDATED -- details and slide map in the a8 handoff top block
     (STATUS_2026-11-01_QUIET_EVENTS_HANDOFF). Page: cue- and lick-aligned perm maps are the main cue / lick speed maps
     (same speed measure), 0-4 s response maps = supplement (Priya). Open: rhythm-band movement
     maps per lick factor, re-render remaining FR
     figures (pooled-fit figures with session dots / pre column), rescore the pooled null when the O2 sub / chronic
     files land.
     11-07 NIGHT: v2 is now the DEFAULT `--match-rule` in epoch_shape / drift_floor / cohort_null (v1 still selectable;
     output folders keyed on the rule name, so v2 keeps writing to *_v2 and v1 outputs stay put). LICK-RATE FIGURE
     (Priya asked): `scripts/orofacial_lick_rate_figure.py` redraws the ILI + fast-lick-share rows of the v36 cohort
     engagement figure from its saved tables -> LP_analysis/.../cohort/figures/prepost/cohort_lick_rate_by_position.png.
     ILI +15-25 ms at EVERY spout acutely (*), back by subacute except close spouts; fast-lick share falls acutely only
     at far contra (-.44) / far mid (-.13) -- partly fewer licks (far contra 3.1 licks / trial vs 8.7 pre; an isolated
     lick has no neighbour), so ILI is the rhythm measure. Not yet on page / deck (caption to Priya first).
     11-07 NIGHT (2): ALL MAP TESTS NOW 10,000 SHUFFLES (Priya), like the scalars: position, aligned, session_loading
     (default now 10000) and the new band maps re-run with drift controls; 2000-shuffle cluster tables kept in the session
     scratchpad for the comparison. Changes vs 2000: position maps none (17 cells, pts +-6); aligned: lick close_L
     subacute up (258 pts) and close_center chronic down (131) NEW; CUE DRIFT CONTROL 3 -> 2 clusters (far_center 374 pts
     gone; both left at far_R, where the post-stroke change is a decrease) -> page 3.10a / 3.10b text + slide 82 bullet
     need rewording (Priya approves); session loading: lick pair unchanged; lick_slow_late acute more (172) gone, whisk_run
     acute more (112) new (exploratory). RHYTHM-BAND MAPS (`scripts/face_rhythm_band_maps.py`, readouts/<set>/band_maps/):
     acute decrease at every spout in every lick band, graded by rhythm -- face mean log2 7Hz -.30, fast_close -.20, lick
     rhythm -.16, far_nose -.12, slow_late -.09, slow_jaw / prelick_nose ~0 -- fits the ILI slowing; later epochs +.06 to
     +.13 at close spouts; drift control has 13 small clusters (mean drift +.03; far_L up, close_R down) so later-epoch
     band increases need that caveat. The band gradient is not yet a test (proposed: per-session 7Hz-minus-rhythm contrast,
     within-animal shuffles). Not on page / deck yet.

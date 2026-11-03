# Handoff 2026-11-01: quiet-period cortical events after the VLS stroke (cohort1, PS92-95) + literature context

> UPDATE 11-03 (late+): helper (B) DONE -- spout decoder biased (prod understates acute far-contra drop ~.10; fix = bal_em
> centring, new spec, AWAITING PRIYA); lick decoder clean except cued all-lick acute offset caveat (tell 34 after Priya).
> PS110: refined 891/4,276 is on the MOSAIC detection; per-tile has only strict 559/3,804; asked Priya. See running list.
> **PRE-COMPACTION SNAPSHOT 11-03 (late) -- START HERE.** Two helpers RUNNING (resume by SendMessage if a rate limit stops
> them; never start duplicates):
> (A) bal_em -> PRODUCTION state decoder (Priya approved) + MIN-MASS FALLBACK (Priya approved: if a class's EM share < 5 %
>     (also test 2 / 10 %) centre with the pre-stroke class mix; default only if it clearly cuts the .09 absent-state drop
>     and keeps PS94 running .99 / .97 / .96 / .92 within .01). Edits wfield_local/locomotor_decoder.py (UNCOMMITTED, theirs;
>     e3 promised not to touch it), tests, refit via load_or_fit (new spec id records min_mass), re-render state families
>     (old -> _superseded_session_z_20261103/), DECISIONS / FROZEN_MODELS / BEHAVIOURAL_STATE_CONTROL / FINDINGS; writes
>     build_inputs/state_balem_swaps.json (deck slides 13-14) + docs/status/STATE_BALEM_PAGE_EDITS_2026-11-03.md.
> (B) decoder mix checks (spout frozen decoder: pool_sessions z-score on engaged trials, gated subsets, coupling; lick-
>     direction decoder: like-for-like labels but is any centre / scale over lick samples? frame-SD gain) ->
>     labcams/decoder_mix_checks/{spout,lick}/ (figures already written, report pending) + DECISIONS. If biased: tell Priya
>     first, and sabatinilab-34 (lick decoders = its wf_attempt code).
> DONE late 11-03: mix-invariance stress test (1f33189e): no normalisation is both mix-invariant and right on PS94 (fixed /
> per-segment centring fail PS94's session offset; mixture model breaks when a state dominates) -> keep bal_em.
> WAITING: sabatinilab-e3 re-rendering trial classes (slides 27-32), orofacial_cohort k_metrics / Fig 1.3 (24-25),
> cam1_incomplete_licks Fig 7.6 (attempts slide) after A1 (3.5 s hit window) / A2 / A4; will send PNG paths + the A3
> (animal-weighted change-vs-pre) star-change list -> recrop those slides + behaviour hit-rate slide.
> PAGES BLOCKED by account (session now on priya.srikanth@hhmi.org / earlier psrikanth87@gmail.com; pages owned by the HMS
> account): Priya to /login HMS or share with edit. Then publish per docs/status/FIGURE_RULES_AUDIT_2026-11-03.md
> (+ RESTW-first imaging reorder, section 23, bal_em section-3 edits) and send 34 the encoding-page figure-rules list.
> OPEN: PS110 mCherry count on slide 115 (559 strict / 891 refined pre-dedupe / grant panel says 817, unverified).
> Machine: plenty of CPU free (Priya 11-03) -- helpers may use fan_sessions up to its 8-worker cap; watch RAM.
>
> **WINDOW STATE 11-03 (evening); partly superseded by the snapshot above.**
> DECK (138 slides, this window = only writer; apply swap specs with apply_figure_rules_1103.py <spec.json>, check by
> PowerPoint export): figure-rules swaps applied (slides 10, 19, 24, 33, 56, 57, 58, 100), camera-view swaps (25, 36),
> trial_class_main_1103.py (slide 27 'Far contra: contact .95 -> .05 acutely' = MAIN trial-class slide; old jaw-only /
> incomplete slide hidden at 29), Big-picture quiet bullet reworded (stroke-amplified; smaller version maybe at baseline).
> RUNNING (2 helpers): (1) bal_em -> PRODUCTION state decoder (Priya 'ok bal_em it is'): new scaling / spec id in
> locomotor_decoder.py, tests, refit via load_or_fit, re-render state families (old -> _superseded_session_z_20261103/),
> docs; writes build_inputs/state_balem_swaps.json (slides 13-14) + docs/status/STATE_BALEM_PAGE_EDITS_2026-11-03.md.
> (2) mix-invariance stress test (diagnostic only; forced class mixes in held-out pre sessions; arms prod / bal_em /
> nocentre / per-segment norm D1-D2 / full mixture model E) -> state_decoder_ps94_check/mix_invariance/ + DECISIONS.
> (3) decoder mix checks (Priya 11-03, diagnostic only) -> labcams/decoder_mix_checks/{spout,lick}/ + DECISIONS: is the
> SPOUT frozen decoder (pool_sessions z-score on engaged trials; outcome-gated subsets? coupling when far contra loses
> drive) or the LICK-DIRECTION frozen decoder (like-for-like landing labels, but is any centre / scale computed over lick
> samples, i.e. landing-mix-dependent? frame-SD gain from acute quiet events) biased by a post-stroke class-mix shift?
> Matters for the read-out-offset finding (licks read contralesional of landing). Helpers 1-2 were rate-limited at
> ~12:30 and RESUMED; if a helper stops on a rate limit again, resume it (SendMessage), never start a duplicate.
> WHEN THEY LAND: apply state_balem_swaps.json; relay results; switching to D or E = Priya's call; if a lick / spout
> decoder is biased, it touches published results -> tell Priya before changing anything, and tell sabatinilab-34 (lick
> decoders are its code, wf_attempt).
> PAGES BLOCKED: Claude Code is signed in as psrikanth87@gmail.com; the four results pages belong to the HMS account
> (not visible). Priya to /login as HMS or share the pages with edit access. Then publish from
> docs/status/FIGURE_RULES_AUDIT_2026-11-03.md: kinematics (_nolines set), FR (re-rendered in place), imaging (bar family
> + cd / rest / per-animal _nolines; RESTW-first reorder -- Priya 11-03; section 1 text belongs to sabatinilab-5b),
> encoding section 23 (23.4, 23.5, 23.13, 23.15, 23.22, 23.23 _nolines; 512-file limit -> replace in place / inline; tell
> sabatinilab-34), plus imaging section 3 bal_em edits.
> WAITING ON OTHER WINDOWS: sabatinilab-e3 consolidation (one trial_classes module, A4 28f2a632; Priya 11-03: close-mid
> 'unknown' -> NO RESPONSE with a jaw_judged flag) will re-render the trial-class PNGs -> recrop slides 27-32. e3 owns
> epoch_figures.py / epoch_behaviour.py now (boot_delta switch). 5b may change imaging section 1 -> deck slide 9.
> PRIYA DECIDED 11-03: RESTW leads maps; slide 31 main; oblique 3-D relabelled OK; path axis + = contra leftward OK;
> 11-02 mirrored path files deleted; bal_em adopted. OPEN: PS110 mCherry count on slide 115 (559 strict vs refined 891
> before dedupe; grant panel A says 817 -- likely refined after tile dedupe, unverified); encoding-page figure-rules list
> for sabatinilab-34 not yet sent (in the audit doc).
>
> **WINDOW STATE 11-03 (afternoon).** This window = PI deck + figure-rules pass (+ quiet events section 23).
> Deck M:/MICROSCOPE/Priya/Widefield/presentations/PI_overview_2026-11-02.pptx = 138 slides, SHORT titles (map in
> docs/status/DECK_TITLES_2026-11-03.csv; scripts find slides by TITLE or shape_name). This window is the ONLY deck
> writer; helpers write swap specs (build_inputs/*.json), this window applies them (apply_*_1103.py, backed up, PowerPoint
> export check). Today's deck steps: audit_1103 (+fix, fix2), short_titles, recrop_trial_classes_1103 (one all-engaged
> class table, 07385b37), apply_camera_view_1103. Audit record: docs/status/DECK_AUDIT_2026-11-03.md.
> RUNNING: figure-rules helper (rules: no lines joining spouts / epochs; faded open session dots, shape = animal,
> colour = epoch; epochs offset; stars never overlap, top band; grey epoch palettes -> EPOCH_LINE) with sub-helpers on
> imaging (epoch_figures bar dots, cd / rest / per-animal, acc-by-position star overlap, grey -> epoch colours),
> FaceRhythm, kinematics (cohort_metrics_minus_pre = fig_cohort_scalars; Fig 1.1 stars; slide-24 4-panel re-cut) and
> encoding section 23 (ours; publish section 23 ourselves, then tell sabatinilab-34). It publishes imaging + FR pages
> itself; kinematics _nolines files + deck figure_rules_swaps.json come back to this window. Encoding page outside
> section 23 belongs to sabatinilab-34: audit chunks (scratchpad audit_chunk*_encoding.md, ~120 violations, mostly
> wf_attempt scripts + frozen_fr_compare.epoch_lines / bars) -> one list for 34 in docs/status/FIGURE_RULES_AUDIT_2026-11-03.md.
> Shared helper wfield_local/cat_points.py (a89c1f2d); camera_view.py (9f68c492).
> PAGES: imaging v49 (+ helper edits; sabatinilab-5b owns section 1 table / Fig 1.2 caption / takeaway -- swap images only;
> deck slide 9 quotes section 1, 5b will send changes); kinematics v24 (camera-view paths, + = contra default); encoding v312.
> OPEN FOR PRIYA: (a) state decoder: adopt CLASS-BALANCED session centring as a new frozen spec? (running-definition test
> 783c04b2: kinematic 1.5 mm/s rule helps PS94 but hurts PS92; balanced centring fixes PS94 .97 / .96 / .91, no animal
> worse); (b) slide 31 (all-engaged trial classes) as THE main trial-class slide?; (c) RESTW vs REST lead on maps; (d)
> PS110 mCherry count 559 / 3,804 vs refined 891 / 4,276; (e) Big-picture quiet bullet qualifier; (f) oblique 3-D views
> relabelled not mirrored -- OK?; (g) path axis + = contra running right-to-left -- OK?; (h) delete 11-02 mirrored path files?
> Not ours: cohort-1 swapped-frame / hpfit pass (another window); "working" = jaw moved or camera lick (5b, section 1 only).
>
> **OVERNIGHT 11-02 -> 11-03: live log + decisions for Priya in [`RUNNING_LIST_2026-11-02_OVERNIGHT.md`](RUNNING_LIST_2026-11-02_OVERNIGHT.md).**
> State decoder z-score fix DONE (abda8c9c; specificity ratio now an acute claim); cohort2_qc fan_out -> fan_sessions (d74ab907); pushed.
> **11-03 ~02:30: all overnight jobs DONE, nothing running.** Deck now 134 slides (apply_overnight_specs.py: hidden PS94
> state slide 14, slide 13 bullets, six kinematics pictures + = contralesional on 33-36 / 123-124). Pages: imaging v49
> (PS94 loses RUNNING under the z-scored decoder, not licking; Fig 3.5), kinematics v18 (+ = contralesional), encoding
> v312 (section 23: T13 LP follow-up unchanged, 4-8 Hz laterality does not revive the withdrawal; Figs 23.23-23.24 inline,
> artifact at its 512-file limit). Open decisions 1-9 in the running list.
>
> **PI DECK STATE (11-02 ~23:45).** M:/MICROSCOPE/Priya/Widefield/presentations/PI_overview_2026-11-02.pptx, 125 slides
> (Priya also edits it: never rebuild; edit in place by slide TITLE; never re-run build_1102.py). Source folder
> _source_PI_overview_2026-11-02/: every edit is a script that backs the deck up first (backup_before_<step>.pptx) and
> refuses to run twice; figure_provenance.csv tracks every picture. Tonight, in order: restructure_stepwise.py,
> fix_stepwise_layout.py, add_maps_lickresp.py (+crop_pos_maps_rows.py, crop_lick_response.py, fix_maps_lickresp.py),
> add_fr_matching.py (+fix), replace_tables.py (per-animal tables -> session-dot figures: fig_decoding_sessions.py,
> fig_state_position_sessions.py, fig_pose_validation_sessions.py, fig_farcontra_attempts_sessions.py),
> add_pooled_gap.py (fig_frozen_refit_pooled.py), fig_behaviour_hits.py (sessions as dots), add_missing_key_figs.py +
> fix_missing_key_figs.py (12 missing key figures from the page audit; supporting = hidden slides), add_videos.py
> (make_video_clips.py: PS122 epileptiform movie, cohort-4 VFSS pre 10-30 + post 11-02 clips), add_twop.py
> (build_inputs/twop/twop_slides.json: 2-photon H2B vs NES, six-spout CDs, ROICaT cross-day). Priya's deck rules:
> session dots, mean +- SEM dots, NO lines across epochs; multi-panel figures re-tiled into rows; main point in the
> title; supporting figures hidden. Headline now first on Big picture: recovery = compensation, not restoration.
> Pages updated (NOT the encoding page): imaging v47 (Figs 0.2, 1.4, 3.4, 4.4; section-1 acute n 4/4/4/1; quiet ~7x),
> kinematics v15 (Figs 7.6, 10.1). Waiting on Priya: her grant copy (2-photon panel check), cohort-4 stroke date /
> meaning of 'post-stroke1'; 20261029/ fluoroscopy files are named 20261028_* (likely mis-dated).
> **OPEN (running): PS94 state decoder** collapses gradually (.98 pre -> .45 chronic normalised; others intact). Lead
> hypothesis: by_animal_day never z-scores per session although the frozen spec says zscore=True. Diagnostics in
> M:/MICROSCOPE/Priya/Widefield/labcams/state_decoder_ps94_check/ (collect.py per animal, PIDs 27956 28444 12284 33940;
> then analyse.py -> per_session.csv, animal_epoch_medians.csv, fig1_confusion_*.png, fig2_speed_hist.png). A fix
> re-freezes the state reference: needs Priya's OK.
>
> **RESUME HERE (updated 11-02 ~07:15, before compaction).** Live status, every test's result + figure path, the overnight log,
> page versions and "conclusions that changed": [`RUNNING_LIST_2026-11-01_QUIET_EVENTS.md`](RUNNING_LIST_2026-11-01_QUIET_EVENTS.md)
> (keep it updated and commit each time). Section 4 = all tests (T1-T22); section 7 = Priya's reduced-drive hypothesis;
> section 8 = overnight state.
>
> **WHERE THE RESULTS LIVE:** encoding (goal-question) page SECTION 23 "Quiet-period cortical state",
> https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d#quiet-state, live = v308 (22 figures 23.1-23.22). My working copy =
> scratchpad/encpage/index.html (live page + my section; figures q_*.png beside it) -- scratchpad may not survive; if missing,
> re-read the live page (Artifact read) and edit that. RULES with the encoding window (sabatinilab-34): only touch section 23
> + its status-table row; re-read the live page before every publish; send 34 the new version number after; never edit
> C:/Users/SabatiniLab/Github/wf_attempt/_goal_page/index.html (34 merges our section into it; fig_ids rows already added).
> Imaging page section 15 = MOVED note (v41), not updated.
>
> **LICK SOURCE (Priya 11-01 / 11-02):** LP video licks replace contact-only everywhere appropriate. Switch QS_LICK_SOURCE
> (quiet_spikes_v2.LICK_SOURCE): "daq" (default, outputs v2_quiet/) | "daq+lp" (outputs v2_quiet_lp/). Quiet frames = no DAQ
> contact OR LP lick (lp_licks.lick_times: cam4 licks_all + cam1-only, whole session) within 1.5 s; task-window masks
> (quiet_task_events) the same; lick-LOCKED analyses (quiet_lick_lfo.lick_events -> lick LFO / ITPC, state response) use LP
> onsets + DAQ contacts with no LP lick within 0.1 s. Hit classes and the engaged / quit gate stay contact-based. PS93_0813 has
> no video (DAQ only). Effort analysis is not lick-locked (unchanged).
>
> **STATE 11-02 08:05:** nothing running. LP re-run COMPLETE; encoding page v311 = every section-23 figure on LP licks
> (v309 23.5/6/11/12/15, v310 23.16, v311 composition + T21 + ~7x titles). PI deck PI_overview_2026-11-02.pptx built (agent; build
> scripts in presentations/_source_PI_overview_2026-11-02/; movie script scripts/cohort2_qc/quiet_event_movie.py).
>
> **NEXT:** (1) DONE 11-02 (v309-v311): swap the still-DAQ section-23 figures to the v2_quiet_lp versions: 23.5 (T13
> outcome, v2_quiet_lp/task_events/t13_precue_event_vs_outcome.png), 23.6 (examples, v2_quiet_lp/quiet_event_example_traces.png),
> 23.11 (415, v2_quiet_lp/hemo415/), 23.12 (breathing, v2_quiet_lp/breathing/), 23.15 (task rates), 23.16 (lick LFO,
> v2_quiet_lp/lick_lfo/), and add T21 / composition numbers on LP licks; update captions + retag '[licks: DAQ contact + LP
> video ...]'; compare each headline number DAQ vs LP and record any change in the running list's "conclusions that
> changed"; publish (re-read live first), message 34. (2) T13 follow-up was run on DAQ-defined events only (decoder windows
> already exclude LP licks) -- optional LP re-run. (3) Decide with Priya whether to do T20-type laterality with a band free of
> the 7-11 Hz artefact (4-8 Hz) as the primary synchronisation measure.
>
> **CURRENT READING (PROVISIONAL, n = 4; LP-based quiet frames):** acutely, a pre-existing, local, non-movement 2-3 Hz event type
> in anterolateral orofacial cortex -- rare in pre-stroke ENGAGED quiet (0.1-0.5 /min), 8-60x more frequent when the animal
> disengages, same map in both states -- becomes common while engaged: face-regressed orofacial events x6.7 acute, x3.1
> subacute, x1.5 chronic (4/4; DAQ-only frames x5.1), orofacial 1-4 Hz power up with NO lesion-side bias. Holds at matched face
> stillness (largest when still), also in pre-cue / ITI windows. 15.7-type map difference = mostly WHICH events occur.
> RULED OUT: face movement, contact-only masking (LP makes it larger), haemodynamics (415 nm, T10), interictal (T11, T19),
> breathing (T9; weak video proxy). WITHDRAWN: ipsilesional > contralesional synchronisation (T22: non-calcium 7-11 Hz peak);
> 'acute events decay faster' (not robust to movement correction); 'common driver' from onset timing (31 Hz cannot resolve).
> MEANING: a pre-cue event goes with a weaker frozen-decoder position code acute / subacute (4/4; NOT decoder contamination,
> T13 follow-up) but does not predict failed far-contra attempts beyond time-in-session; quiet measures follow the recovery
> time course only; cued-lick ITPC tracks recovery beyond it in 3/4 (survives composition control, DAQ licks); the larger acute
> lick response is mostly NOT accounted for by pre-lick state (T21; extrapolation-only upper bound 14-22 %). WHY: unexplained --
> anatomy suggests the VLS -> SNr -> VM/Pf -> tjM1 loop, but no lesion-side asymmetry anywhere; a LOCAL orofacial 'disengaged'
> state is the best description. Ideas: pupil / arousal, VM or brainstem recordings, cohort2 replication, faster imaging.
>
> **LITERATURE (FINDINGS 8):** framing = tongue reaching as a precision model of post-stroke motor control (Bollu 2021, ALM
> planning, Xu 2022) with cautions (rodent reach homology contested; descending anatomy: rodent cortex -> brainstem premotor
> nuclei, human direct corticobulbar; striatum never direct in any species); T17 corrections; Zeiger 2021 compatible
> ('re-weighted mapping'); Cullins & Connor 2024 only partly agrees.

**START HERE for the quiet-period events arm.** Resume from section 4 (open tests, in priority order). Results live on
the ENCODING page section 23 (see RESUME block; the imaging page section 15 is a frozen copy) and in
[`docs/COHORT2_QC_2026-10-29.md`](../COHORT2_QC_2026-10-29.md) (quiet-event bullets, 10-31 .. 11-01). Section 6 = how
the results sit against the stroke literature (Priya asked for it 11-01; also offered as a FINDINGS / goal-page section).

Cohort rules apply: cohort1 only (never pooled with cohort2), August-only pre, per-animal table before any test, image
left = ipsilesional on brain maps.

---

## 1. What is running (11-01 ~22:05)

See the running list's "Still running" table (single source). Scripts started by me 11-01 (all done except the re-render):
`quiet_eta_split.py` (T1-T4), `quiet_stillness.py` (T7), `quiet_hemi_amp_recovery.py` (T8, T11, T14, T15),
`quiet_steinmetz_check.py` (T19), `quiet_sync_index.py` (T20), `quiet_eta_fr.py` (movement-corrected 15.7 + decay),
`attempt_effort.py` (effort by trial outcome). Agents' scripts: `quiet_breathing.py` (T9), `quiet_hemo_check.py` (T10),
`quiet_task_events.py` (T12/T13), `quiet_lick_lfo.py` (T16). Nothing kills by name; exact PIDs only.

## 2. Definitions (one place; every quiet script uses these)

* **QUIET frames** (`quiet_spikes_v2.busy_frames`): no DAQ contact lick and no running bout within 1.5 s before or
  after; spout may be out / cue ignored; no swapped imaging pair within 0.5 s. **ENGAGED** = before the session's quit.
* **Detector** (`quiet_spikes_v2.detect`): peaks of the 2-s-running-median-detrended dF/F % trace; height above the
  median of the preceding 1.0-0.2 s; half-width 1.5 frames .. 0.5 s (at half height above that baseline); merged within
  0.3 s; NEGATIVE events = the same on the sign-flipped trace.
* **Threshold k = 4 x the animal's August-pre robust SD** (1.4826 x MAD), common to orofacial and limb (Priya 11-01).
  Chosen to separate pre from acute (ROC), so the ACUTE contrast is optimistic; subacute / chronic were not used.
* **Face regression** (`quiet_spikes_v2_fr.py`): out-of-fold ridge on engaged quiet frames (10 blocks, lags +/-0.3 s),
  regressors = own-session FaceRhythm factors + 20 point PCs + speeds, 2-s high-passed; residual uses its own SD.
* **Stats**: `mixed_models.lmm_table` (epoch + time models vs August pre, FDR `q`); maps: `beta_maps.significance_contour`
  (nested animals -> sessions bootstrap, max-statistic family-wise threshold on 8x8-px bins, eroded stat mask,
  rim-concentrated results suppressed); per-animal map rows = sessions-only bootstrap (`quiet_fluct.session_boot_sig`).

## 3. Results so far (all PROVISIONAL, n = 4)

1. **Orofacial quiet events rise ~5x acutely and are not face movement.** Face-regressed, both polarities: pre ~3.8-4.1
   /min -> acute 15.5 / 32.1 / 19.8 / 20.9 (PS92-95), 4 / 4; LMM x5.1 acute, x2.5 subacute, x1.4 chronic (q < 0.005).
   Raw x4.1. Up +0.71 log10, down +0.68. (page 15.1, 15.3)
2. **Movement link lost acutely**: pre 43-73 % of raw orofacial events coincide with face movement; acute at chance
   (~0.35) in 4 / 4; face-model R2 on quiet frames unchanged acutely (+0.14 chronic). The v1 "R2 drop" did not reproduce.
3. **Limb** also up after regression (+0.33 / +0.36 / +0.38 log10) but acute only clear in PS93 / PS95.
4. **Spectrum**: power grows most at ~2-3 Hz (page 15.5).
5. **SD maps (15.6, outlines added 11-01)**: quiet SD rises over most of dorsal cortex at every epoch (pooled 1,457 /
   1,733 / 1,737 of 2,022 bins significant acute / sub / chronic); acutely the lateral barrel / upper-limb hotspots
   (highest pre SD) are the exception. Per-animal rows = sessions-only test (reliability within animal, not across mice).
6. **Event-triggered maps (15.7, delta rows added 11-01)**: pre-stroke orofacial events spread over lateral SSp
   (barrel / limb) and persist there at +0.2 s; ACUTE orofacial events are confined to anterolateral orofacial cortex
   (ALM / tjM1 / SSp-m) bilaterally -- delta significantly negative over lateral SSp at 0 and +0.2 s; undershoot at +1 s
   shallower. Limb events acutely larger medially / posteriorly. **Priya 11-01: "this feels important".**
7. **Event shape (11-01, `quiet_event_width.py`)**: acute events ~10 % narrower (FWHM orofacial -0.037 s, p 1e-4;
   height-matched -0.045), all DECAY (-0.033 s, ~18 %), RISE unchanged; 3 / 4 animals (PS94 pre already narrowest).
   Threshold-free ACF half-time unchanged -> the trace is not faster overall; events end sooner, consistent with an
   oscillatory 2-3 Hz component rather than sharper bursts.
8. **Quit period vs engaged quiet (all sessions, `quiet_quit_state.py`)**: the pre-stroke QUIT period already has the
   acute signature in orofacial cortex -- orofacial 1-4 Hz power 3.4-5.4x engaged quiet (every epoch ~3x) while cortex-wide
   1-4 Hz power is LOWER (0.6x) and 0.5-1 Hz 0.5x, face stiller (pre face-moving 0.3-0.5x in PS93 / PS94; PS95 ~1). Not
   cortex-wide slow-wave sleep; an orofacial-local low-frequency state of disengagement. AND engaged-quiet face speed
   halves acutely in 4/4 (1242 -> 812, 825 -> 492, 1947 -> 965, 1192 -> 593). Reading: acute engaged quiet resembles the
   pre-stroke disengaged state. Sparse: pre sessions with >= 60 s quit quiet = 0 / 1 / 4 / 6 (PS92-95). Priya: animals
   do not look asleep by eye closure. Confounds: quit is late in session (time / satiety / bleaching).

**Live status + results + figure links: [`RUNNING_LIST_2026-11-01_QUIET_EVENTS.md`](RUNNING_LIST_2026-11-01_QUIET_EVENTS.md).**

## 4. Every test proposed 11-01 to distinguish the hypotheses (status-tagged; run in this order)

Hypotheses: **CIRCUIT** (lesion-driven change in the VLS -> SNr -> motor-thalamus -> orofacial-cortex loop; diaschisis /
local sleep-like dynamics), **STATE** (drowsiness / low arousal / sickness), **ARTEFACT** (breathing / brain motion,
haemodynamics, GCaMP epileptiform events), **SELECTION** (a change in which events cross threshold, not in the events).

| # | test | separates | prediction | status |
|---|---|---|---|---|
| T1 | 15.7 selection control: event maps split by face movement (with / without, +/-0.3 s) AND height-matched (4-6 / 6-9 / > 9 x pre SD), pre vs acute | SELECTION vs changed event | same maps -> stroke made non-movement events common; acute still more local -> event changed | DONE 11-01: mostly selection (running list) |
| T2 | ipsi vs contra orofacial amplitude and onset lag per event (31 Hz, ~32 ms) | CIRCUIT (ipsilesional thalamic projection: ipsi first / larger) vs STATE (simultaneous, symmetric) | CIRCUIT: ipsi leads / larger acutely | DONE: no amplitude asymmetry; onset within +/-5 ms but 31 Hz + GCaMP6s cannot resolve 5-10 ms callosal delays, so callosal spread from one side is NOT excluded |
| T3 | up vs down event maps | local oscillation vs discrete bursts | same local footprint -> excitability oscillation | DONE: same structure, weaker |
| T4 | pre-stroke QUIT-period event map vs acute ENGAGED map | STATE (acute = pre disengaged state) | same local footprint -> state-like | DONE: same map, more local |
| T5 | event width / rise / decay, ACF half-time | thalamic bursts (faster rise) vs oscillation (faster decay) vs drowsiness (slower) | -- | DONE: decay ~18 % faster, rise unchanged, ACF unchanged -> oscillation (3.7) |
| T6 | quit-period vs engaged quiet: cortex bands, face speed | STATE | -- | DONE: pre quit already has the orofacial 1-4 Hz signature, cortex-wide lower; engaged-quiet face speed halves acutely 4/4 (3.8) |
| T7 | within engaged quiet: event rate vs face stillness (per session, binned face speed) | STATE contribution | rate rises with stillness -> state adds | DONE: rise persists at matched stillness, largest when still |
| T8 | per-animal ipsi vs contra comparison of the rate / SD increase (direct, not by eye from 15.6) | CIRCUIT (ipsilesional) vs global | CIRCUIT: ipsi > contra in >= 3/4 | DONE: no ipsi bias (orofacial +0.02 n.s.; limb contra > ipsi acute) -- running list |
| T9 | breathing: coherence of quiet orofacial trace with FaceRhythm breathing / sniff factors, pre vs acute; respiration rate change | ARTEFACT (breathing) | coherence high / tracks respiration -> artefact | DONE: not supported (low, unchanged coherence; no tracking; regression leaves the rise) |
| T10 | haemodynamics: the same events / band power in the 415 nm channel and in uncorrected 470 | ARTEFACT (haemodynamic) | events absent in 415 -> neural | DONE: excluded (415 unchanged, no event-locked 415) |
| T11 | amplitude distribution; count of events > 10 % dF/F and cortex-wide synchronous events (Steinmetz 2017 interictal shape) | ARTEFACT / epileptiform | none -> not interictal | DONE: not interictal (big events rarer acutely, less limb co-occurrence) -- running list |
| T12 | events in pre-cue / ITI (task) windows, not only quiet | relevance | -- | DONE: rise in every task window 4/4 |
| T13 | pre-cue event on a trial -> failed far-contra attempt / weaker pre-cue position code on that trial | CIRCUIT-noise mechanism for the deficit | events predict failure | DONE: not failure (beyond time-in-session), but weaker position code acute/sub 4/4 |
| T14 | within animal, event rate vs far-contra recovery with log-day removed (as `iti_recovery_link.py`) | link to recovery | negative relation | DONE: r -0.75 raw, -0.23 detrended (p .048, 2/4) |
| T15 | rate tracks each animal's deficit (PS93 largest acute rate; PS94 least recovered); pre-register orderings for cohort2 | CIRCUIT vs STATE / ARTEFACT | -- | DONE: no ordering (rho -0.20) |
| T16 | Ramanathan tension: movement-locked low-frequency (1-4 Hz) power in lick windows vs the quiet increase, same sessions | reconcile with Ramanathan 2018 (task LFOs fall) | task LFO down while quiet up | DONE: task-locked ITPC unchanged acutely, evoked LFO up -- no dissociation |
| T17 | (DONE 11-01, see end of section 6) literature: targeted PubMed search (striatal / capsular stroke + widefield GCaMP / cortical recordings); verify the *(memory)* citations (Allegra Mascaro 2019, Levin 2009, Nudo 1996, Lee-Wang-Sabatini 2020) | context | -- | DONE 11-01 |
| T19 | Steinmetz criteria explicitly: waveform (rise / decay vs their ~150 ms interictal shape), refractoriness (inter-event interval distribution, 0.5-1 s gap), positive-only vs both polarities, bilateral symmetry, per animal incl. PS93 (cortical injury) | ARTEFACT / epileptiform (T17 caveat: location + rate overlap) | none of the criteria met -> not interictal | DONE: not interictal |
| T20 | SYNCHRONISATION index (low / high frequency power, e.g. 1-4 Hz / 8-15 Hz) in quiet and pre-cue frames, orofacial vs control areas, ipsi vs contra, by epoch | Priya's original hypothesis: REDUCED tonic thalamocortical drive -> synchronised cortex (Poulet 2012) | more synchronised acutely, strongest orofacial (and ipsilesional if via the loop) | TO DO (added 11-01) |
| T21 | within PRE sessions: does pre-lick slow power (1-4 Hz, -1..0 s) predict the lick-evoked response size? | is the acute ~2x lick response a consequence of a synchronised state (evoked responses are larger in quiet wakefulness, Crochet & Petersen 2006) rather than more drive | yes -> state explains part of the larger response | TO DO |
| T22 | per-area F0 (470 minus the 415-predicted haemodynamic part), ipsi vs contra orofacial vs control areas | tonic level (the only measure close to it; weak) | reduced drive -> relative ipsi orofacial F0 drop | TO DO |
| T18 | k-choice bias: report subacute / chronic (not used to pick k) as the unbiased contrasts; held-out-animal k already LOAO | statistics | -- | NOTE (reported in 15.1 caption) |

## 5. Files

Scripts (`scripts/cohort2_qc/`): `quiet_spikes_v2.py` (detector, thresholds, `busy_frames`; `--summarise --k 4`),
`quiet_spikes_v2_fr.py` (face regression), `quiet_spikes_v2_examples.py` (example traces with +/- thresholds),
`quiet_fluct.py` (spectrum, SD maps + outlines, event-triggered traces / maps + delta rows), `quiet_event_width.py`,
`quiet_quit_state.py`. Older v1 (rest windows, prominence detector; superseded): `quiet_area_spikes.py`,
`quiet_spikes_fr.py`, `quiet_spikes_examples.py`.

Outputs `M:\MICROSCOPE\Priya\Widefield\labcams\cohort1_quiet_spikes\v2_quiet\`: `fr_regressed_own/`
(quiet_events_face_regressed[_detail].png, lmm_v2_fr.csv), `quiet_event_example_traces.png`, `fluct/`
(quiet_spectrum.png, quiet_sd_maps.png + `_significance.csv`, quiet_eta_maps.png + `_significance.csv`), `width/`
(quiet_event_width.png, lmm_width.csv, sessions csv), `quit_state/`.

Page: imaging page section 15 (15.1-15.7), v38; page PNGs copied into `docs/results_pages/imaging/imaging_page/`
(gitignored), publish with `url` + `files` map. Commits: aba7ca1f, a4ca4b31, db525a57.

Pitfalls hit 11-01: `analysis_kit.boot_ci(by_animal_dict, rng)` (not a DataFrame); `sig_marks.star_top(ax, x, text)`
takes the star text, and `reserve_headroom(fig)` takes the FIGURE, called once before layout; full-res maps are
540 x 640 (`beta_maps.MAP_SHAPE`), so pass `mask = stat_mask() & on-pixels` to the map tests.

## 6. Context in the field: how our results sit against the stroke literature (11-01)

Citations checked by web search on 11-01 unless marked *(memory)*; abstracts only, not full texts. Our results =
FINDINGS.md sections 0-5 plus this arm.

| our result | prior literature | verdict |
|---|---|---|
| Behaviour recovers by COMPENSATION (success back; contralesional licks use a changed pattern) | Established in human kinematics and rodent reaching (Levin / Kleim / Wolf definitions; Whishaw rat reaching) *(memory)* | SUPPORTS; orofacial / licking version in mouse new |
| LARGER task responses (ITI lick ~2x acute; chronic residual beyond the pre mapping), widespread, largest day 1, shrinking with recovery, persisting in the least recovered (PS94) | Human fMRI: task overactivation decreasing with recovery (Ward et al. 2003 Brain); meta-analysis: more contralesional M1, bilateral PMv, SMA, decreasing with time (Rehme et al. 2012 NeuroImage). Rodent widefield after CORTICAL stroke: activation spreads and refocuses with rehab (Allegra Mascaro et al. 2019) *(memory)* | STRONGLY SUPPORTS the human picture incl. time course and deficit link; new as rodent calcium after a striatal lesion |
| (same) | Rat INTERNAL-CAPSULE stroke, multi-unit in M1 / RFA / S1, weeks 5-8: task firing slightly REDUCED, population pattern PRESERVED (Cettina, Nudo, Bundy et al. 2025 bioRxiv) | CONFLICTS (see below) |
| CONTRALESIONAL > ipsilesional residual; contralesional shift of decoded lick position | Contralesional recruitment = most consistent fMRI finding (Rehme 2012). Orofacial: swallowing recovery tracks pharyngeal-cortex expansion in the UNAFFECTED hemisphere (Hamdy et al. 1998 Gastroenterology) | SUPPORTS direction; decoded-position shift is a new measure |
| LARGER QUIET FLUCTUATIONS (2-3 Hz, both signs, ~5x acute, recovering; faster decay) | Human EEG: more delta after stroke, contralesional delta prognostic of poor recovery; sleep-like slow waves / OFF-periods in awake cortex after cortico-subcortical lesions (Sarasso et al. 2020 Brain) | CONSISTENT with human; new in rodents after a subcortical lesion |
| (same) | Rat: reach-locked low-frequency oscillations in perilesional M1 WEAKENED after stroke, their return tracks recovery (Ramanathan et al. 2018 Nat Med). Rodent widefield cortical stroke: connectivity lost then partly restored (eLife 2024, 90080) | TENSION (movement-locked vs quiet; test 4.7) |
| ACUTE ENCODING LOSS = the changed movement (frozen encoder predicts the far-contra drop from the lick) | No study found that asks whether the pre-stroke mapping already predicts the change from altered behaviour | NEW, conceptually important |
| CHRONIC REMAPPING (same movement, different response; self-fit better; target and executed coding remapped) | Reorganisation is classic after CORTICAL lesions (Winship & Murphy 2008 J Neurosci; Latifi et al. 2020 Cereb Cortex; Nudo 1996 *(memory)*); after CAPSULAR stroke: no reorganisation (Cettina 2025) | NEW for remote cortex after a subcortical lesion; tension with capsular result; drift caveat |
| Failed attempts (no-contact licks) lose side coding and trial-by-trial coupling; first cued lick read toward target; coupling returns on hits; code plateaus ~3/4; cortex does not lead recovery | No direct precedent found | NEW |
| Lesion size orders recovery speed | Expected | SUPPORTS |
| Acute orofacial quiet events LOCAL to anterolateral orofacial cortex (15.7) | Fits the VLS -> SNr -> VM/VAL -> ALM / tjM1 loop anatomy (Lee, Wang & Sabatini 2020) as a hypothesis; no stroke precedent found | NEW (pending test 4.1) |

**The conflict to take seriously (Cettina et al. 2025).** Differences that likely explain much of it: internal capsule
(corticospinal fibres) vs VLS (basal-ganglia loop node, whose loss should change thalamocortical drive); weeks 5-8
post-plateau vs our largest effects acute / subacute; ipsilesional only vs our bilateral / contralesional effects;
multi-unit spikes in 3 areas vs widefield calcium over dorsal cortex. Their "pattern preserved" agrees with parts of
ours (3 / 4 animals keep the lateral read-out map; PS94 remapped). The real tension is our chronic residual vs their
chronic ipsilesional spiking.

**Brand new, as far as found:** (1) longitudinal whole-dorsal-cortex calcium after a STRIATAL stroke in an orofacial
task; (2) separating movement change from mapping change with frozen pre-stroke models; (3) behaviour fully back while
the code on hits plateaus ~70-77 %; (4) failed attempts as directed movements without a matching cortical spout
signal; (5) self-initiated vs cued read-outs diverge; (6) quiet-period fluctuation increase as a rodent counterpart of
human post-stroke delta / local sleep after a subcortical lesion, with a local orofacial footprint.

**Reviewer caveats from this literature:** neurovascular coupling (human overactivation was questioned on this; our 415
nm / F0 checks address it, FINDINGS 4d0c); the lesion is not purely subcortical (PS93 anterior motor cortex injury;
DLS involvement in some); drift vs remapping without a no-stroke reference; n = 4.

Sources: Cettina et al. 2025 https://pmc.ncbi.nlm.nih.gov/articles/PMC12637610/ ; Ward et al. 2003
https://discovery.ucl.ac.uk/id/eprint/10102084/ ; Rehme et al. 2012 https://pmc.ncbi.nlm.nih.gov/articles/PMC3630767 ;
Hamdy et al. 1998 https://orca.cardiff.ac.uk/id/eprint/33926/ ; Sarasso et al. 2020 https://pubmed.ncbi.nlm.nih.gov/33188680/ ;
Ramanathan et al. 2018 https://neatlabs.ucsd.edu/pubs/2018/Ramanathan2018.pdf ; photothrombotic widefield / assemblies
https://elifesciences.org/articles/90080v1 ; Winship & Murphy 2008 https://www.jneurosci.org/content/28/26/6592 ; Latifi
et al. 2020 https://pmc.ncbi.nlm.nih.gov/articles/PMC8257065 ; EEG delta commentary
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5133428/ . Still to verify: Allegra Mascaro 2019, Levin 2009, Nudo 1996,
Lee, Wang & Sabatini 2020 details; a targeted PubMed search for striatal / capsular stroke + widefield GCaMP.

**LITERATURE UPDATE 11-01 late (T17 agent; abstracts unless noted) -- corrections and additions to the table above.**
CORRECTIONS: (1) eLife 90080 = Bandet & Winship 2024 (two-photon, forelimb-S1 photothrombosis; spiking / assemblies
disrupted but HIGHLY LOCAL -- cortex ~750 um from the border largely unchanged; https://pmc.ncbi.nlm.nih.gov/articles/PMC11060715);
"widefield interhemispheric connectivity lost then partly restored" = Cramer et al. 2019 NeuroImage 199:570
(https://epub.ub.uni-muenchen.de/79115/). (2) CIRCUIT: Lee, Wang & Sabatini 2020 (full text,
https://pmc.ncbi.nlm.nih.gov/articles/PMC7606600) = VLS -> lateral SNr -> VM (VM/VA) and Pf -> tjM1, PLUS SNr -> lateral SC,
PCRt, IRt (brainstem orofacial premotor) directly; ALM via VM/VAL is Guo et al. 2017 Nature 545:181
(https://pmc.ncbi.nlm.nih.gov/articles/6431254). So the lesion can change licking without going through cortex.
(3) Allegra Mascaro 2019 Cell Rep 28:3474: cortical photothrombosis, Thy1-GCaMP6f, forelimb retraction; motor map
re-segregation with ROBOT TRAINING + CONTRALESIONAL SILENCING (combined), not rehab alone; Cecchini 2021 found new, not
restored, propagation patterns (https://pmc.ncbi.nlm.nih.gov/articles/PMC8159272). Levin 2009, Nudo 1996, Whishaw 2000
(Neuropharmacology 39:788) verified as cited.
ADD -- precedents (soften "new"): Bauer et al. 2014 NeuroImage (mouse tMCAO, SUBCORTICAL-only infarcts reduce lateral-cortex
homotopic connectivity; resting, no task; https://pmc.ncbi.nlm.nih.gov/articles/PMC4332714); Cullins & Connor 2024 Front
Neurol (rat tMCAO, striatum-centred, often into cortex, ~1/3 subcortical; ICMS maps at 8 weeks under anaesthesia;
https://pmc.ncbi.nlm.nih.gov/articles/PMC10987714/) -- ONLY PARTLY AGREES (Priya 11-01, after reading it): ipsilesional
TONGUE map shrinks (aged rats only; young p .057; more with larger infarcts) whereas our ipsilesional orofacial cortex is
NOT reduced (chronic lick responses larger there too; lateral read-out map kept in 3/4); intact-hemisphere change is JAW
only (tongue unchanged) and function tracks the INFARCTED hemisphere, intact-hemisphere maps do not correlate with function,
whereas our contralesional changes are the larger ones (we do not separate tongue vs jaw cortex, nor show which hemisphere
tracks function). Different measure (ICMS output efficacy vs calcium activity in behaviour), lesions and ages. Shared only:
some intact-hemisphere change;
Ahmed et al. 2017 JCBFM (lick efficiency down after striatal-only MCAO to day 40, compensatory drinking clusters;
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5305038/); Fanciullacci 2017 (human subcortical stroke: more delta over the
affected hemisphere); Gloor 1977 (delta from white-matter / thalamic deafferentation); Carmichael & Chesselet 2002 (rat:
peri-lesion synchronous 0.1-2 Hz activity days 1-3, drives sprouting; cortical lesion); Kim et al. 2022 Cell Rep (local
delta waves in sleep, normalise with training). Bollu et al. 2021 (ALM needed for corrective submovements) = mechanistic
anchor for failed attempts lacking a spout code. No study found that records ALM / tjM1 after VLS lesion or silencing, or
that uses a frozen pre-stroke encoder: both still look new.
ADD -- counter-literature a reviewer would cite: Branscheidt et al. 2022 J Neurophysiol (human SUBCORTICAL stroke, 1 year:
no recovery-related cortical resting-connectivity change; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8896990/); Zeiger
et al. 2021 Nat Commun (potentiation of pre-existing circuits, not remapping; https://pmc.ncbi.nlm.nih.gov/articles/PMC8233353) --
COMPATIBLE, largely semantics (Priya 11-01): our chronic change is a RE-WEIGHTED mapping within the same areas (3/4 keep the
read-out map's pattern), not territorial remapping -> say 're-weighted mapping', not 'remapping';
Sarasso 2020 caveat: off-periods ABSENT with unilateral LACUNAR subcortical lesions (ours are larger, some with cortex / DLS);
Panzeri et al. 2025 bioRxiv (contralesional widefield activity reflects intact-paw compensatory use); Ramanathan 2018 /
Khanna 2021 Cell (task-locked < 4 Hz activity DOWN, our quiet 2-3 Hz UP: T16 tests the dissociation); Facchin 2020
(ipsilesional NREM slow waves reduced). Answers to place beside them: frozen-encoder movement control, drift floor, T16.
EPILEPTIFORM CAVEAT (R3): Steinmetz et al. 2017 interictal events are > 10 % dF/F, 0.1-0.5 Hz, largest ANTEROLATERAL and
bilaterally symmetric -- location and our acute rate (15-32 /min) overlap; but their Thy1-GCaMP6s GP4.3 mice (our line)
showed none, our events are 2-3 % with up and down rising equally, events > 10 % are RARER acutely (T11), and late
post-stroke seizure risk tracks cortical involvement (Galovic 2018; note PS93's cortical injury). Remaining check: waveform /
refractoriness vs the Steinmetz criteria. Post-stroke peri-lesional spikes with 1-4 Hz delta are reported in mice from day
15 (https://pubmed.ncbi.nlm.nih.gov/41192651/).
DESCENDING ANATOMY (Priya 11-01): rodent motor cortex reaches tongue / jaw / face motoneurons mainly via brainstem reticular
premotor neurons (PCRt / IRt), with sparse or no direct cortico-motoneuronal contacts (as for the rodent limb), whereas
human corticobulbar fibres also contact hypoglossal / facial / trigeminal motoneurons directly *(memory: Kuypers; Lemon --
verify)*. The striatum projects to the brainstem in no species (via SNr / GPi; SNr -> brainstem premotor / SC in rodents and
primates). Framing consequence: ground homology in control principles and cortical computation, not the final pathway
(reach models share the limitation); brainstem premotor networks are a candidate compensation site; the SNr -> brainstem
route makes the frozen-encoder movement-vs-mapping separation essential.

## 7. Priya's original hypothesis (11-01): VLS stroke -> less thalamocortical drive -> LOWER ipsilesional orofacial activity

Why it did not show as a decrease, and why it may still be right: (1) widefield dF/F is relative -- tonic level appears only
in F0, which rose ~11 % acutely (uniform, 415 similar; confounded by blood / light), so a tonic decrease is largely
unmeasurable here; (2) the acute effects we see -- larger 2-3 Hz quiet fluctuations, larger lick-evoked responses, engaged
quiet resembling pre-stroke disengagement -- are what a LESS thalamically driven (synchronised, quiet-wakefulness-like)
cortex shows (Poulet et al. 2012; Crochet & Petersen 2006 -- CHECKED 11-01 at abstract / review level, see below); (3) a whole-VLS lesion removes direct AND indirect pathways with opposing net
effects on SNr, and SPNs fire little at rest, so the clearest loss is the phasic, lick-locked SNr pause (timing / gain), not
mean drive. Against: the fluctuation increase is not ipsilesional (T8). Tests T20-T22; direct VM measurement in a future
cohort would settle it.

CITATION CHECK (Priya 11-01: "read those citations more carefully and see if it holds up"; abstracts + the Frontiers
review fnsys.2019.00040, full texts paywalled). Poulet, Fernandez, Crochet & Petersen 2012 Nat Neurosci 15:370: in BARREL
cortex, whisking-related desynchronisation comes with INCREASED VPM firing, is not peripheral (survives infraorbital nerve
cut), is halted by pharmacological thalamic inactivation and mimicked by optogenetic thalamic stimulation -> HOLDS for
"thalamic input desynchronises the active state"; does NOT show that REDUCED drive makes resting cortex MORE synchronised
(under inactivation, whisking Vm was hyperpolarised with REDUCED variance); thalamic inactivation literature is mixed and
state-dependent (anaesthesia: weaker slow oscillation, J Neurosci 2014 34:5689; an awake study cited as enhanced slow
oscillation, not verified; awake preprint: cortical baseline largely unchanged). Crochet & Petersen 2006 Nat Neurosci
9:608: quiet wakefulness = slow large Vm fluctuations, whisking = small fast; PASSIVE whisker responses robust in quiet,
suppressed during whisking -> HOLDS only for passive sensory responses in sensory cortex. OVERREACH: both are barrel
cortex / VPM, not orofacial MOTOR cortex / VM-VAL (layer-1 matrix input); in ALM, VM/VAL silencing COLLAPSES activity
(Guo et al. 2017) -- Priya's original prediction; lick responses are self-generated motor responses; arousal /
neuromodulators also set cortical state. Verdict: the reduced-drive -> synchronised-cortex reframing is a plausible
hypothesis, not supported directly by the literature.

T20 RESULT (11-01, `quiet_sync_index.py`, `V2/sync_index/`): synchronisation index log10(PSD 1-4 / 8-15 Hz), quiet engaged.
Acute: orofacial +0.59 (q < .001), whole cortex +0.39, limb +0.15 n.s.; orofacial - cortex +0.20 (q .001); orofacial IPSI -
CONTRA +0.13 (q 1e-4; 4/4 positive acute, pre ~0), subacute +0.09; 8-15 Hz power lower ipsilesionally in 4/4. Chronic:
whole-cortex synchronisation rises more than orofacial (orofacial - cortex -0.27). -> acutely orofacial cortex is more
synchronised and MORE SO IPSILESIONALLY: the laterality the reduced-drive hypothesis predicts, which the event rate (T8)
did not show. Caveat: 8-15 Hz with GCaMP6s is near the noise floor, so the index can rise from slow power alone and the
ipsi fast-power drop may partly reflect hemisphere brightness / noise differences (T22 checks F0).

T22 (11-01 23:10, commit 972954e3): no ipsilesional F0 drop; the T20 ipsi > contra synchronisation comes from a non-calcium 7-11 Hz peak (415 nm, lower ipsilesionally) -> WITHDRAWN. Section 7's hypothesis loses its laterality support.

## 8. Overnight 11-01 -> 11-02 (state at ~01:10)

* Section 23 lives on the ENCODING page (https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d#quiet-state), now v308. Source of my edits: scratchpad/encpage/index.html (= live page + my section); the encoding window (sabatinilab-34) merges section 23 into its local wf_attempt/_goal_page/index.html -- never edit that file; before each update re-read the live page and send 34 the new version.
* LP-based quiet frames are PRIMARY: QS_LICK_SOURCE=daq+lp -> v2_quiet_lp/ (driver scratchpad/run_lp_chain.sh; lp_licks.py loader, b51213d3). Not yet re-run with LP: examples (23.6), task windows / T13 (23.5, 23.15), lick LFO (23.16), breathing (23.12), 415 (23.11), effort (23.22), state response (T21) -- marked DAQ-only on the page.
* Done tonight: T9 breathing (not supported), T10, T13 follow-up (not contamination), T16 + composition, T21 (state does not account for most of the larger lick response; extrapolation-only upper bound), T22 (laterality WITHDRAWN), LP sensitivity + LP chain. Details: running list.
* Next: re-run the DAQ-only items with QS_LICK_SOURCE=daq+lp (their scripts take the switch via quiet_spikes_v2.busy_frames where they use it; task_events / lick_lfo / state_response need checking), then swap those figures.

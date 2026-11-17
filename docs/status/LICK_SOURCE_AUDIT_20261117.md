# Lick-source audit of every caption and slide note (2026-11-17)

Priya 11-17: "ensure all figure captions / slide notes are clear about what data they used -- I don't know if we specify
everywhere cam4 vs cam4 + cam1 LP licks". Branch `fix/lick-source-wording-1117` (worktree `wf_licksrc1117`, from
origin/main abb1623b). Decision record: `docs/DECISIONS.md` 2026-11-17. Wording module: `wfield_local/lick_provenance.py`.

**Standard wording** (use these and nothing else):

| train | caption phrase |
|---|---|
| DAQ sensor | `licks: DAQ spout contact` |
| LP train, post-cue (lick / no-lick, attempt classes, first lick, lick window) | `licks: LP video, cam4 + cam1 (cam1 droplets removed)` |
| LP train read before the cue (1 s ENL lick-free window, LP source) | `pre-cue lick-free window: LP video, cam4 (cam1 is scored only from the cue to the trial stop)` |
| cam4-only sets (ITI, kinematics / 3-D per-lick, FaceRhythm lick windows) | `licks: LP video, cam4 only` (+ `3-D position from cam1 + cam4 pose` where 3-D) |
| camera trial classes | `camera trial classes (DAQ spout contact + LP video, cam4 + cam1, cam1 droplets removed)` |
| RESTnl rest | `rest licks excluded: DAQ + cam4`, generated from `restnl_mask` (`lick_provenance.restnl_text`); reads `DAQ + cam4 + cam1` once the re-run (branch `analysis/restnl-cam1-1117`, Priya 11-17) is placed |
| restdock05 REST (time-local) | `rest licks excluded: DAQ spout contact` |

Sessions: PS93_0813 has no behavior video (no LP licks, no camera classes, no RESTnl). Every other curated cohort-1
session has cam4 AND cam1 lick events (checked 11-17: 120 curated sessions, 1 without `cam1_licks/*_events.csv`), so
"cam4, plus cam1 where it exists" is replaced by the plain phrase. (The `lp_licks` docstring's "CAM1 IS NOT FINISHED,
74 / 108 sessions" is the 10-21 state and is stale.)

## 1. Truth table (from the code, origin/main abb1623b)

| analysis / code path | lick train | evidence (file:line) |
|---|---|---|
| DAQ lick detector (every DAQ consumer) | DAQ spout contact, yaml profile 2.5 / 0.5 V, 40 ms floor, no refractory (`daqlk`) | `wfield_local/config.py:704` `lick_detector`; `plot_lick_aligned_averages._load_daq_events` (`allow_lp`, :49) |
| Imaging decode / encode / maps, default (DAQ) source, incl. the rebuild run `..._sel-camera-..._daqlk` | lick class, lick alignment and the 2 s ENL window: DAQ contact. Working / miss / stopped selection under the camera gate: camera classes (below). Untagged pre-11-03 renders: DAQ quit-tail working rule | `trial_classes.py:477` ("the lick class stays the DAQ contact set in both modes"); `locanmf_position_decoder.py:439` |
| `WIDEFIELD_LICK_SOURCE=lp` (LP_licks family: `_trial_features`, `nolick_decoder`, `attempt_coding`, `attempt_maps`, 15r LP arm, LP frozen decoders) | `lp_licks.lp_lick_times` = cam4 `licks_all.csv` (every context) + cam1-only events of `cam1_licks/<a>_<d>_events.csv` (not cam4-matched, `droplet` removed), edited so the first post-cue lick IS the attempt-class first lick | `lp_licks.py:457` (`_inputs`: the `_events.csv` file), `:495-506` (`_read_times`, filter at :502), `:509` (`edit_train`); the train is swapped in under LP by `_load_daq_events` (`plot_lick_aligned_averages.py:49`), `locanmf_position_decoder.py:439`, `attempt_coding.py:209` |
| cam1 `_events.csv` coverage | POST-CUE only: cue .. trial stop. `events_all.csv` (trial windows, strobe - 0.5 .. stop + 3 s) exists but the LP train does not read it | `cam1_licks.py:44` (`span == "post"`), `scripts/cam1_licks.py:124` (`_events.csv`) vs `:132` (`_events_all.csv`); checked on PS92_20260602: min on_ms = 0 |
| LP first lick / lick-aligned windows | first LP lick (trial_attempts `combined` = cam4 + cam1, else cam4); at a close spout the DAQ contact when LP is missing or > tol late | `lp_licks.py:157` (`first_lick_timing`), `:394`; encoder `combined_encoding.py:1313` (`trial_first_lick`) |
| LP pre-cue (ENL) lick-free window | 1.0 s under LP (`enl_post_s`), lick-free on the LP train, so in practice **cam4 only** (no cam1 event can precede the cue). No DAQ check under LP | `lp_licks.py:197` (`enl_post_s`), `:233` (`gate_licks_s`) |
| Camera trial classes (`trial_classes`, `scripts/trial_attempts.py`) | contact = DAQ; incomplete = cam4 LP lick or cam1-only lick (droplets removed) or cam1 tongue, no contact; jaw cam4 OR cam1; response window cue + 70 ms .. min(cue + 3.5 s, stop) | `trial_classes.py:1-60`, `:264`; `trial_attempts.py:8` |
| LP-inclusive engagement gate (`WIDEFIELD_GATE_SOURCE=lp`) | DAQ contact OR LP attempt (trial_attempts hit / incomplete) | `gate_source.py:17` |
| RESTnl lick-free rest (`restnl_mask`, `rest_frozen_decoder_nl`, `state_space_rest_baseline`) | restdock05 rest (itself minus DAQ contact [0.5, 1.0] s) minus every DAQ lick onset and every cam4 lick of `licks_all.csv`, each [0.5 s before, 2.0 s after]. **DAQ + cam4** (no cam1) for every current output | `restnl_mask.py:52` (BUF_S), `:59` (LICK_SOURCE_TEXT), `:118`; `rest_frozen_decoder_nl.py:107`; `state_space_rest_baseline.py:5` |
| restdock05 REST (time-local, "REST reference") | DAQ contact only, [0.5, 1.0] s | `rest_by_position._session_daq` (via `restnl_mask.py:11`) |
| Combined encoder (production `COMBINED_ENCODING_LICKS=combined`, `LICKS_ALL=split`) | post-cue lick kernel: cam4 + cam1-only (cam1 post-cue file); pre-cue / ITI kernels: cam4 `licks_all.csv` (+ cam1 by time, but the cam1 file is post-cue, so effectively cam4); contact kernel: DAQ; "enl" scoring window lick-free on the LP set, so cam4 before the cue | `combined_encoding.py:682-690`, `:862`, `:1186`; production env `docs/status/rebuild_20261103/ENCODER_RENDER_COMMANDS.md:22` |
| Frozen encoder (`FROZEN_LICKS_ALL=split`) | lick_onset = cam4 + cam1-only post-cue; licks_pre / licks_iti = cam4 (the docstring's "+ cam1-only pre-cue / ITI" finds none, same reason); contact = DAQ | `frozen_encoder.py:150-160`, `:424-470` |
| Encoder trial gate `OROFACIAL_GATE=attempted` | trial_attempts hit / incomplete (DAQ + cam4 + cam1) | `orofacial_cohort.py:1655` |
| ITI licks (`iti_licks`, `iti_dock_audit`, `iti_lick_rate`, `iti_first_lick`, `lick_iti_prelick`; key figure 2, sections 12 / 14) | cam4 only: `licks_all.csv` context iti joined to the 3-D whole-session table; bout gaps from cam4 licks; docked phase from DAQ / log events | `iti_licks.py:380-402`; `iti_lick_rate.py:3`; `iti_dock_audit.py` docstring |
| Trial first licks of the iti_licks extracts (state-space page, FR variance, cued ramp) | first cam4 lick onset (v36 `licks.csv`) on engaged trials (gate per OROFACIAL_GATE) | `iti_licks.py:406-414`, `:2302` |
| State-space ENL baseline | LP train (`lp_lick_times`), so cam4 before the cue | `state_space_enl_baseline.py:48` |
| Kinematics (cam4 2-D, 3-D) | cam4 LP licks (v36 `licks.csv` / `licks_all.csv`); 3-D = triangulated cam1 + cam4 pose ON cam4 lick events; contact per lick = DAQ | `pose3d_kinematics.py:6`, `:33` |
| FaceRhythm lick windows / asymmetry / lick rate | cam4 `licks_all.csv` | `face_rhythm_readouts.py:341`, `face_rhythm_asym_aligned.py:10` |
| FaceRhythm ENL decode | union of the LP train (cam4 before the cue) and DAQ contact | `fr_enl_decode.py:14`, `:83` (docstring says "cam4": right in effect) |
| Quiet events (`quiet_spikes_v2`, QS_LICK_SOURCE=daq+lp, the 11-01 production) | DAQ contact + cam4 `licks_all` + cam1-only `events_all` (droplets removed): both cameras over the trial windows | `quiet_spikes_v2.py:54`, `cohort2_qc/lp_licks.py:76` |
| Quiet task events, pre-cue window | `gate_licks_s` (source-dependent) | `quiet_task_events.py:123` |
| cam4 / cam1 whole-session coverage | both detected over the same trial windows (strobe - 0.5 s .. stop + 3 s, cut at the next trial), ~98-99 % of the video; ITIs ~1-1.5 s, so effectively the whole session | `orofacial_outputs.py:357` (`all_licks_session`), `scripts/cam1_licks.py:132` |

**Key correction.** The LP train's cam1 part comes from the POST-CUE cam1 file. "Within-trial lick-free windows use
cam4 + cam1" holds for the post-cue lick / no-lick call but NOT for the pre-cue (ENL) window, which is cam4 only in
every LP output. Captions that said the ENL window is free of "LP licks (cam4 + cam1)" are WRONG.

## 2. Counts

| source | copy audited (date) | WRONG | VAGUE | OK-reword | OK / no lick input |
|---|---|---|---|---|---|
| Image page | `pages/img/img_v58.html` (v57 + pending Fig 2.3 edit, 11-17 12:39) | 3 | 13 | 5 | 63 |
| Goal / encoding page | `pages/enc/page_new.html` (= tool-results artifact-77c5daa3-1794870505, 11-16 17:54; newest copy found, may be behind the live page) | 1 | 102 | 0 | 203 |
| State-space page | `pages/ss/ss_page_1117.html` (11-17 10:16, local; newer than the published 11-16 copy artifact-47cdaefb-1794871843) | 0 | 102 (every figure: no caption named a train) | 0 | 18 sub-panel captions (covered with their figure) |
| Kinematics page | `pages/kin1116/index.html` (= artifact-63c6d127-1794872517, 11-16 18:34) | 0 | 27 | 0 | 42 |
| FaceRhythm page | artifact-f3f9712b-1794866489 (11-16 17:05, "Updated 2026-11-16"; may be stale vs the FR window's v54) | 0 | 12 | 0 | 98 (mostly factor shape / timing, no lick events) |
| Analysis deck (built 12:43) | 511 slides with a "HOW IT WAS BUILT" line + key-figure / section K notes | 4 (key-figure captions: 2.5 and 1.4 ENL "cam4 + cam1"; 3-D tags "cam4 + cam1, 3-D" x2) | 427 window lines naming no train + 8 key-figure / K notes | 3 | 84 |
| PI deck `Overview_2026-11-06.pptx` (11-17 13:35) | 206 slides | 6 slides | 27 slides | 0 | rest |

`docs/results_pages/` in the repo holds older page snapshots (imaging, state_space): not audited, superseded by the
copies above.

## 3. Analysis deck (fixed in code, rebuilt)

- `window_provenance` (every figure slide's "HOW IT WAS BUILT"): always a `- licks:` line from
  `lick_provenance.licks_line`; source = the caller (LP section G), else the figure folder (`LP_licks` tag or
  `attempt_maps` = LP; video / encoder folders = "not resolved", none left in this build), else the build env. The
  engaged line names the first-lick train. DAQ slides of this tagged build add the camera-class clause only for
  figures of the run tag (untagged renders: "working trials: DAQ quit-tail rule").
- `deck_keyfigs.py`: Fig 2.3 ("camera (LP) licks"), 2.5 (lick-free window "cam4 + cam1" -> cam4), 1.4 (tag + "free of
  video licks (cam4 + cam1)" -> cam4), 1.4 aim and 7.5 tags ("cam4 + cam1, 3-D" -> cam4 only, 3-D from cam1 + cam4
  pose), 6.1 (tag + "camera (LP) licks"), 10.1 ("LP video licks" -> cam4 only), 7.18b RESTnl tag generated from the mask.
- `deck_registry.py`: `_K_LPGATE` and the two 15r LP notes: "cam4, plus cam1 where it exists" -> standard wording.
- Result (594 slides, 811 figures, the 2 known-missing 15f files): 511 / 511 window lines name the train: DAQ 424
  (2 of them untagged renders), LP 87 (40 pre-cue with the cam4 ENL clause, 47 cue / lick, incl. the 3 attempt-map slides).
  Previous deck kept as `spout_position_analysis_summary__20261117_1417.pptx` (the builder also kept `__1243`, `__1418`,
  `__1421`; `__1420` / `__1422` are my intermediate builds).
- Tests: `tests/test_window_provenance.py` (+9: LP / DAQ / caller / unresolved / every line names a train / RESTnl
  wording follows the mask / the LP cam1 input is the post-cue file / camera clause scoped to the run tag / attempt maps).

## 4. Pages: edit lists (apply to the LIVE page later)

Edit files (exact raw old -> new, applied in order, each must match exactly once; a preview of the result beside
each): `scratchpad/pages/{img,goal,ss,kin,fr}/lick_source_edits_1117.json` (+ `lick_source_preview_1117.html`);
generator `scratchpad/licksrc/make_edits.py` (wording from `lick_provenance`; run it on the live HTML if the live page
has moved: it stops at any string it cannot find). Totals: img 42, goal 110, ss 142, kin 27, fr 12 replacements.
After the RESTnl cam1 re-run is placed, re-run the generator (its `RESTNL` then reads "DAQ + cam4 + cam1"); the image
page's RESTnl captions ("minus every DAQ spout-contact lick and every cam4 LP lick", tag "RESTnl baseline DAQ contact
+ cam4 LP video") need the same update then.

Also noted (not lick source): a goal-page caption labelled "(was Fig 14.20)" breaks the replace-not-annotate rule; image
page Fig 15.x says quiet frames exclude DAQ contact only, while goal-page section 23 and the PI deck use DAQ + LP:
confirm which run the image-page figures come from.

## 4b. Page tables (every VAGUE / WRONG item, grouped by identical replacement)

### Image page (Cortical Spout Coding)

Figcaptions: 84. Figures flagged: WRONG 3, VAGUE 13, OK-reword 5; OK (or no lick input) 63.

| class | figures | stated | replacement | why |
|---|---|---|---|---|
| WRONG | Fig 1.4 | the pre-cue window is free of video licks (cam4 + cam1) | the pre-cue window is free of LP video licks (cam4; cam1 is scored only from the cue to the trial stop) | lp_licks reads the post-cue cam1 file (cue .. stop): no cam1 event can fall in a pre-cue window |
| WRONG | Fig 5.1 | LP video (cam4 + cam1) defines the attempt classes, the lick-free pre-cue window and the lick alignment | LP video (cam4 + cam1, cam1 droplets removed) defines the attempt classes and the lick alignment; the lick-free pre-cue window uses LP video licks (cam4; cam1 is scored only from the cue to the trial stop) | cam1 never enters a pre-cue window |
| WRONG | Fig 7.4b | LP video licks (cam4, plus cam1 where it exists) choose the trials and the 1 s lick-free window | LP video licks choose the trials (cam4 + cam1, cam1 droplets removed) and the 1 s lick-free window (cam4; cam1 is scored only from the cue to the trial stop) | cam1 never enters a pre-cue window |
| VAGUE | Fig 1.4 | [licks: camera trial classes (cam4 + cam1)] | [licks: camera trial classes (DAQ contact + LP video, cam4 + cam1); pre-cue lick-free: LP cam4] | classes also use DAQ contact; the pre-cue check of this LP_licks_ENL2s run is cam4 |
| VAGUE | Fig 2.5, Fig 2.6 | LP video licks choose the lick-free window and, with DAQ contact, the engaged / working trials | LP video licks choose the lick-free window (cam4; cam1 is scored only from the cue to the trial stop) and, with DAQ contact, the engaged / working trials (LP video, cam4 + cam1, cam1 droplets removed) |  |
| VAGUE | Fig 7.3, Fig 7.4 | LP video licks choose the trials and the 1 s lick-free window | LP video licks choose the trials (cam4 + cam1, cam1 droplets removed) and the 1 s lick-free window (cam4; cam1 is scored only from the cue to the trial stop) |  |
| VAGUE | Fig 7.3b | [licks: LP video (cam4 + cam1); camera trial classes; RESTnl rest mask: DAQ contact + cam4 LP] | [licks: camera trial classes (DAQ contact + LP video, cam4 + cam1); ENL lick-free: LP cam4; RESTnl rest licks excluded: DAQ + cam4] | pre-cue figure: the tag implied cam1 in the ENL check |
| VAGUE | Fig 7.3c | [licks: LP video (cam4 + cam1); camera trial classes; session mean] | [licks: camera trial classes (DAQ contact + LP video, cam4 + cam1); ENL lick-free: LP cam4; session mean] | pre-cue figure: the tag implied cam1 in the ENL check |
| VAGUE | Fig 7.10, Fig 7.11 | LP video licks choose the trials (working = attempted, by video) | LP video licks (cam4 + cam1, cam1 droplets removed) choose the trials (working = attempted, by video) |  |
| VAGUE | Fig 7.18, Fig 7.19, Fig 7.20 | LP video licks choose the trials and start the lick window; the REST | LP video licks (cam4 + cam1, cam1 droplets removed) choose the trials and start the lick window; the REST |  |
| VAGUE | Fig 7.3, Fig 7.4, Fig 7.10, Fig 7.11, Fig 7.18, Fig 7.19, Fig 7.20, Fig 8.1, Fig 8.2 | [licks: mixed - LP video gate; REST baseline DAQ contact] | [licks: mixed - LP video gate (cam4 + cam1); REST baseline DAQ contact] |  |
| OK-reword | Fig 2.4, Fig 7.11b, Fig 7.20b, Fig 8.1, Fig 8.1b, Fig 8.2b | (cam4, plus cam1 where it exists) | (cam4 + cam1, cam1 droplets removed) | accurate (cam1 exists for every curated session but PS93_0813); standard wording |

### Goal / encoding page (Attempt Encoding Figures)

Figcaptions: 306. Figures flagged: WRONG 1, VAGUE 102, OK-reword 0; OK (or no lick input) 203.

| class | figures | stated | replacement | why |
|---|---|---|---|---|
| WRONG | Fig 8.6 | pre-cue window = ENL, kept free of spout contact by the task's DAQ lick sensor | pre-cue window = ENL 1.0 s, kept free of LP video licks (cam4; cam1 is scored only from the cue to the trial stop) | attempt_coding runs under WIDEFIELD_LICK_SOURCE=lp: the ENL window is the LP 1.0 s window (lp_licks.enl_post_s), gated on the LP train, not on DAQ contact |
| VAGUE | Fig 0.1, Fig 0.2, Fig 0.7, Fig 0.8, Fig 0.11, Fig 1.2, Fig 1.3, Fig 1.4, Fig 1.5, Fig 1.6, Fig 1.11, [Image moved off the page, Fig 1.13, Fig 1.14 (+77 more) | trials gated on an attempt (LP lick or DAQ contact, trial_attempts.csv) | trials gated on an attempt (DAQ spout contact or LP video lick, cam4 + cam1, cam1 droplets removed; trial_attempts.csv) | frozen / combined encoder captions |
| VAGUE | Fig 8.1, Fig 8.2, Fig 8.3, Fig 8.4, Fig 8.5, Fig 8.6, Fig 9.1, Fig 9.2, Fig 9.3 | lick window aligned to the first LP video lick | lick window aligned to the first LP video lick (cam4 + cam1, cam1 droplets removed; close spouts: DAQ contact when the LP lick is missing or late) | attempt_coding lick window (trial_attempts first_lick_ms) |
| VAGUE | BASELINE UPDATED 10-29 (P | RESTnl rest excludes every LP cam4 lick (contact or not) | RESTnl rest licks excluded: DAQ + cam4 (every DAQ spout-contact lick and every cam4 LP lick, contact or not) | omitted the DAQ part of the mask |
| VAGUE | Fig 23.22 | (no lick-source sentence) | Licks: trial classes from trial_attempts.csv (hit = DAQ spout contact; lick-no-contact = LP video lick, cam4 + cam1, cam1 droplets removed). | no lick source stated |
| VAGUE | (was Fig 14.20) | (no lick-source sentence) | Licks: LP video, cam4 only (3-D position from cam1 + cam4 pose); no DAQ spout-contact input. | no lick source stated; NB its label '(was Fig 14.20)' breaks the replace-not-annotate rule |

### State-space page

Figcaptions: 120. Figures flagged: WRONG 0, VAGUE 102, OK-reword 0; OK (or no lick input) 18.

| class | figures | stated | replacement | why |
|---|---|---|---|---|
| VAGUE | Fig 2.1, Fig 2.2, Fig 2.3, Fig 3.1, Fig 3.2, Fig 3.3, Fig 3.4, Fig 3.5, Fig 3.6, Fig 3.7, Fig 3.8, Fig 3.10, Fig 3.11, Fig 4.2 (+43 more) | (no lick-source sentence) | Licks: trials = camera classes contact + incomplete (DAQ spout contact; LP video, cam4 + cam1, cam1 droplets removed); first lick = first cam4 LP lick onset (v36 licks.csv). | no lick-source sentence (state-space page) |
| VAGUE | Fig 3.9, Fig 4.1, Fig 4.3, Fig 5.1, Fig 5.2, Fig 5.3, Fig 5.4, Fig 5.5, Fig 5.6, Fig 5.7, Fig 5.8, Fig 6.3, Fig 6.4, Fig 6.11 (+1 more) | (no lick-source sentence) | Licks: trials = camera classes contact + incomplete (DAQ spout contact; LP video, cam4 + cam1, cam1 droplets removed); first lick = first cam4 LP lick onset (v36 licks.csv); 3-D landing from cam1 + cam4 pose on cam4 licks. | no lick-source sentence (state-space page) |
| VAGUE | Fig 9.1, Fig 10.17, Fig 10.19, Fig 10.26 | every DAQ and camera lick | every DAQ spout-contact lick and every cam4 LP lick | state-space first-lick extracts = first cam4 lick (iti_licks) |
| VAGUE | Fig 9.1 | first lick = camera (LP) lick onset | first lick = first cam4 LP lick onset | state-space first-lick extracts = first cam4 lick (iti_licks) |
| VAGUE | Fig 9.1, Fig 9.1b, Fig 9.1d, Fig 9.2, Fig 9.3b, Fig 9.3d, Fig 9.3, Fig 9.4, Fig 9.10, Fig 10.17, Fig 10.19 | (no lick-source sentence) | Licks: trials = camera classes contact + incomplete (DAQ spout contact; LP video, cam4 + cam1, cam1 droplets removed); first lick = first cam4 LP lick onset (v36 licks.csv); RESTnl rest licks excluded: DAQ + cam4. | no lick-source sentence (state-space page) |
| VAGUE | Fig 9.1c, Fig 9.1, Fig 9.2, Fig 9.3c, Fig 9.3, Fig 10.21, Fig 10.22, Fig 10.23, Fig 10.25 | (no lick-source sentence) | Licks: trials = camera classes contact + incomplete (DAQ spout contact; LP video, cam4 + cam1, cam1 droplets removed); first lick = first cam4 LP lick onset (v36 licks.csv); 3-D landing from cam1 + cam4 pose on cam4 licks; RESTnl rest licks excluded: DAQ + cam4. | no lick-source sentence (state-space page) |
| VAGUE | Fig 9.5, Fig 9.6, Fig 10.18, Fig 10.20 | (no lick-source sentence) | Licks: trials = camera classes contact + incomplete (DAQ spout contact; LP video, cam4 + cam1, cam1 droplets removed); first lick = first cam4 LP lick onset (v36 licks.csv); ENL lick-free window: LP video, cam4; cam1 is scored only from the cue to the trial stop. | no lick-source sentence (state-space page) |
| VAGUE | Fig 9.8, Fig 10.24, Fig 10.26 | (no lick-source sentence) | Licks: trials = camera classes contact + incomplete (DAQ spout contact; LP video, cam4 + cam1, cam1 droplets removed); first lick = first cam4 LP lick onset (v36 licks.csv); RESTnl rest licks excluded: DAQ + cam4; ENL lick-free window: LP video, cam4; cam1 is scored only from the cue to the trial stop. | no lick-source sentence (state-space page) |
| VAGUE | Fig 10.1, Fig 10.2, Fig 10.3, Fig 10.5 | first camera (LP) lick | first cam4 LP lick | state-space first-lick extracts = first cam4 lick (iti_licks) |
| VAGUE | Fig 10.17, Fig 10.18, Fig 10.19, Fig 10.20, Fig 10.21, Fig 10.22, Fig 10.24, Fig 10.25, Fig 10.26, Fig 11.2, Fig 11.4, Fig 11.6, Fig 11.8 | camera LP first lick | first cam4 LP lick | state-space first-lick extracts = first cam4 lick (iti_licks) |
| VAGUE | Fig 11.1, Fig 11.2, Fig 11.3, Fig 11.4, Fig 11.5, Fig 11.6, Fig 11.7, Fig 11.8 | (no lick-source sentence) | Licks: lick trials and lick window = LP video licks (cam4 + cam1, cam1 droplets removed; first LP lick, close spouts: DAQ contact when the LP lick is missing or late); gate = DAQ contact OR LP attempt. | no lick-source sentence (state-space page) |

### Kinematics page

Figcaptions: 69. Figures flagged: WRONG 0, VAGUE 27, OK-reword 0; OK (or no lick input) 42.

| class | figures | stated | replacement | why |
|---|---|---|---|---|
| VAGUE | Fig 6.1 | camera (LP) licks for the other classes | LP video licks (cam4 + cam1, cam1 droplets removed) for the other classes |  |
| VAGUE | Fig 1.4, Fig 1.5, Fig 1.6, Fig 3.2, Fig 3.3, Fig 3.4, Fig 3.5, Fig 3.7, Fig 4.7, Fig 4.8, Fig 4.9, Fig 4.10, Fig 4.11, Fig 6.7 (+12 more) | (no lick-source sentence) | Licks: LP video, cam4 only (3-D position from cam1 + cam4 pose). | 3-D lick table = cam4 lick events; licks not named |

### FaceRhythm page

Figcaptions: 110. Figures flagged: WRONG 0, VAGUE 12, OK-reword 0; OK (or no lick input) 98.

| class | figures | stated | replacement | why |
|---|---|---|---|---|
| VAGUE | Fig 3.1, Fig 3.2, Fig 3.3, Fig 3.10, Fig 3.10c, Fig 3.10d, Fig 3.11, Fig 3.12, Fig 3.13 | (no lick-source sentence) | Licks: LP video, cam4 only (licks_all.csv; lick and bout onsets); engaged trials = the orofacial engagement gate. | LICK / BOUT alignment; licks not named |
| VAGUE | Fig 5.3, Fig 5.4 | median first-lick latency of the same trials (LP licks) | median first-lick latency of the same trials (LP video licks, cam4 only) |  |
| VAGUE | Fig 6.1 | (no lick-source sentence) | Licks: ENL lick-free window = no DAQ spout contact and no LP video lick (cam4; cam1 is scored only from the cue to the trial stop). | fr_enl_decode: lp_lick_times + DAQ contacts |


## 5. PI deck fix list (owned by a8; NOT edited here)

`M:/MICROSCOPE/Priya/Widefield/presentations/Overview_2026-11-06.pptx` (11-17 13:35), notes read with python-pptx.

| slide(s) | class | stated | replacement |
|---|---|---|---|
| 26, 186, 206 (pre-cue maps) | WRONG | "the 1.0 s ending at the cue if it holds no LP lick (cam4 + cam1)" | "... if it holds no LP lick (cam4; cam1 is scored only from the cue to the trial stop)" |
| 7 | WRONG | "LP video licks (cam4 LP v36 + cam1, triangulated 3-D tongue)" | "LP video licks, cam4 only (3-D position from cam1 + cam4 pose)" |
| 117, 118 | WRONG | "LP cam4 + cam1 3-D licks + DAQ contact" | "LP video licks, cam4 only, 3-D landing from cam1 + cam4 pose; + DAQ contact" |
| 15, 16 | VAGUE | "LP video licks (WIDEFIELD_LICK_SOURCE=lp; lick-free window and lick column on the video train)" | "LP video licks (WIDEFIELD_LICK_SOURCE=lp): lick column = LP video, cam4 + cam1 (cam1 droplets removed); pre-cue lick-free window = LP video, cam4 (cam1 is scored only from the cue to the trial stop)" |
| 14 | VAGUE | "Licks: LP video licks (cam4 + cam1)" (s1 table, pre-cue column too) | add "; pre-cue lick-free window: LP video, cam4 (cam1 is scored only from the cue to the trial stop)" |
| 57 | VAGUE | "LP video licks (cam4 LP v36) + cam1 (3-D)" | "LP video licks, cam4 only (3-D position from cam1 + cam4 pose)" |
| 73 | VAGUE | "Licks: none (factors named afterwards against LP licks)" | "Licks: none in the fit (factors named afterwards against LP video licks, cam4 only)" |
| 74 | VAGUE | "Licks: none (names validated against LP licks / running)" | "Licks: none in the fit (names validated against LP video licks, cam4 only, and running)" |
| 77 | VAGUE | "Licks: none for the peak (per-lick text: LP licks)" | "Licks: none for the peak (per-lick text: LP video licks, cam4 only)" |
| 82, 83, 87 | VAGUE | "Licks: LP licks" | "Licks: LP video, cam4 only (licks_all.csv; lick and bout onsets)" |
| 84 | VAGUE | "Licks: LP (video) licks" | "Licks: LP video, cam4 only (licks_all.csv)" |
| 88, 90, 91, 92, 93, 95 | VAGUE | "Licks: LP (video) licks define the lick factors' context; no DAQ lick gate" | "Licks: LP video, cam4 only (licks_all.csv) define the lick factors' context; no DAQ lick gate" |
| 96 | VAGUE | "Licks: LP (video) licks (FR lick window); 3-D = camera licks" | "Licks: FR lick window = LP video, cam4 only (licks_all.csv); 3-D = cam4 LP licks (3-D position from cam1 + cam4 pose)" |
| 99, 102, 147, 148, 149, 196, 197 | VAGUE | "FROZEN_LICKS_ALL=split adds own pre-cue (ENL) + ITI LP lick kernels" | "... adds own pre-cue (ENL) + ITI LP lick kernels (cam4 only: the cam1 events are post-cue)" |
| 100 | VAGUE | "(own pre-cue + ITI lick kernels)" | "(own pre-cue + ITI lick kernels, cam4 only)" |
| 196 | VAGUE | "LP cam4 + cam1 video licks (3-D landing) + DAQ contact" | "LP cam4 + cam1 video licks (3-D landing on cam4 licks) + DAQ contact" |
| 153 | VAGUE | quiet example, no "Licks:" line | add "Licks: DAQ spout contact + LP video licks (cam4 + cam1, cam1 droplets removed) define non-quiet frames (QS_LICK_SOURCE=daq+lp)." |
| 24, 25, 26, 187, 188, 204-206 (RESTnl maps) | OK now | "minus every DAQ spout-contact lick and every cam4 LP lick" | keep; after the RESTnl cam1 re-run is placed: "... every DAQ spout-contact lick and every LP video lick (cam4 + cam1, cam1 droplets removed)" |

Everything else that names a train in the PI deck (the 11-16 provenance sweep's "Licks:" lines: DAQ daqlk, camera
classes cam4 + cam1 + DAQ, cam4 3-D, encoder combined, quiet daq+lp) matches the truth table.

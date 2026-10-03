# Handoff — 2026-10-02: session-level kinematics (PS93 pre / acute / chronic), lick phase, reach measures, O2

**START HERE for orofacial kinematics.** It continues `STATUS_2026-10-01_KINEMATICS_HANDOFF.md` (port, spout frame,
trial windows, first QC — still valid background). Tracking models and labelling: `STATUS_2026-09-30_LP_OCCLUSION_
ROUND4_HANDOFF.md`. Reasoning for everything below: `DECISIONS.md`, entries dated 2026-10-01 (late) and 2026-10-02.
Analysis desktop; **M: = MICROSCOPE, N: = standby — take paths from `PathResolver`, never type a letter.**

---

## 0. RESUME CHECKLIST

Nothing is running locally.

1. **O2 (Priya runs it):** the first bundle is ready — `M:\MICROSCOPE\Priya\DeepLabCut\Widefield\o2\r3_PS93_20261002\`
   (PS93 0814 / 0821 / 0908 whole videos, DLC round 3 + prior, user ps150, env `deeplabcut` = the existing env from
   the 2025-11 stroke_orofacial O2 runs). Follow its `COMMANDS.md`: step 0 checks the env versions, step 2 `--bench`
   proves the env runs our runner and picks the batch size. When the outputs are copied back:
   `python -m wfield_local.o2_inference collect --name r3_PS93_20261002` → `session_poses/PS93_<date>_full/`; then
   the session scripts with `PS93:<date>:full` (checks, figures, `session_lick_summary`). Compare to the 60-trial
   subset results (same sessions) as the end-to-end check.
2. **Decisions waiting on Priya:** none blocking. Open questions: does the post-stroke image-left shift of the
   successful-lick direction survive more sessions / animals and 3-D (head pose)? close_L disagreement between DLC
   and LP on that measure.
3. **Next analyses:** more sessions / animals (via O2); cam1 once labelled (horizontal-plane direction; the
   spout-referenced reach angle makes sense there); 3-D.
4. Still pending from before: labelling round 4 (131 cam4 frames, Priya) + cam1 (student) → retrain DLC + LP;
   `python -m scripts.chronic_stability` once 0928 stage 2 has landed.

## 0b. Added later on 10-02
* **More sessions without GPU:** `python -m scripts.session_poses from-scan` turned the round-4 scan cache into
  `session_poses/<a>_<d>_r4/` (PS93 0819 acute, 0826 subacute; PS92 0821, 0824; PS94 0921; PS95 0917 — 12 trials
  each, cue −1 → +3.5 s, DLC + LP). PS93 now has a five-session course; summaries per animal
  (`lick_summary_PS9x.*`, `lick_direction_shift_PS93.png`). PS92 / PS94 / PS95 lack pre-stroke predictions.
* **Centered lick phase, cleaned:** tongue-in frames count as at the lips (session lip level) for protrusion
  (`protrusion_filled_px`); direction values hidden where < 20 % of licks show the tongue (`phase.min_coverage`,
  `tongue_kinematics.phase_mean`). The per-lick-extent rows look smoother only by construction.
* **Next proposed:** pre-stroke 60-trial subsets for PS92 / PS94 / PS95 (and PS92 chronic) — locally overnight or on
  O2. **Design for using DLC / LP in the widefield GCaMP analyses: `docs/DLC_IN_WIDEFIELD_ANALYSES.md`** (5 open
  decisions in its §6, incl. the two DAQ lick detectors that already disagree between imaging and behaviour code).

* **O2 on hold** until the next DLC / LP iteration (Priya). **Movement-regressor encoding framework built**
  (`movement_encoding`, `movement_inputs`; synthetic-tested) — next: wire imaging frame times + the per-session
  orofacial event table, then one animal. DAQ lick detection for DLC analyses: keep `lick_detection` (Priya).

## 1. What exists now (code; tests in `tests/`, config in `configs/defaults.yaml`)

| piece | what |
|---|---|
| `tongue_kinematics` switches (all OFF in module DEFAULTS = parity; ON in defaults.yaml) | `fix_detect_offset`; `velocity_window: visible_rise` (+ retraction speed); `lick_geometry` (per-lick on/off, protrusion, ap/lr); `detect_on: protrusion` (detect on tongue–mouth distance); `angle.max_signed_within_lick` / `max_signed_min_frac_of_peak`; `contact` (DAQ contact per lick, `contacts_ms` input); `spout_ref` (per-trial spout tip, `spout_xy` input); `phase.mode: centered` |
| `TongueKinematics.lick_phase` | per lick × phase: ap, lr, protrusion, angle, tongue−spout angle; **centered on the peak, window = cycle_ms** (stroke_orofacial) or per-lick extent |
| `lick_reference` | deviation from the successful-lick (contact) direction, at the peak and over the phase; references `session` and pooled `pre` |
| `jaw_kinematics` `degenerate_as_unknown` | jaw invisible pre-cue → `jaw_pass_qc` NaN; mismatch unknown for quiet trials |
| `o2_inference` + `o2_pose_runner` | bundle on the share (model, prior, runner, job array, COMMANDS.md); `collect` → per-trial windows; runner == local predictor + prior (tested, 150 real frames) |
| scripts | `session_poses` (trial-subset clips + DLC), `kinematics_checks`, `tongue_angle_phase`, `tongue_nocontact_qc`, `session_lick_summary` (table + 3 cross-session figures), `pose_kinematics_demo.run` (driver; `daq_contacts_ms`) |

Data: `session_poses/PS93_{20260814,20260821,20260908}/` (60 trials each: clip, index, DLC + LP csv, checks,
figures); `session_poses/lick_summary_PS93.{csv,png}`, `lick_phase_epochs_PS93.png`, `lick_phase_modes_PS93.png`.

## 2. Decisions (2026-10-01 late → 10-02)

* Cleaning cutoff **0.4**; one-frame v7 offset **fixed**; lick-height floor stays **20 px** (no 5th-percentile gate
  exists; a pre-stroke-percentile floor would be circular and clip the small licks that are the readout).
* Velocity over the **whole visible rise** (the inherited window was peak ±16 ms for lmax licks → 2× model gap).
* Jaw: degenerate baseline = **unknown**, not fail (hope: fill from other cameras / 3-D).
* Lick detection on **protrusion distance** (matters for cam1; ~no change on cam4).
* **Tongue − spout-tip angle is NOT accuracy on cam4**: it is ±30-35° even on contact licks — the spout-tip label is
  the tube's top edge beside the mouth, the tongue tip is pressed past the spout (`tongue_spout_rays_PS93_20260908.png`).
  Replaced by **deviation from the successful-lick direction** (`lick_reference`); the dA columns are kept, labelled.
* Lick phase **centered on the peak, fixed window = the animal's pre-stroke median within-bout ILI** (stroke_orofacial's
  `synthetic_only` mode). The per-lick visible-extent stretch made shape differences partly artefactual
  (`lick_phase_modes_PS93.png`). Short / incomplete licks INCLUDED everywhere; only the lip zone (< 30 px) is masked.
* O2: commands generated, never executed from Python (DUO); bundle on the share; our runner, not `analyze_videos`.

## 3. Findings (one animal, 10 trials / position / session — hypotheses)

* QC: pre-clean removes almost nothing on our tracking, and what it removes is right (frames checked); gates reject
  only gate-M second peaks. No-contact licks are real tongue movements (60 of 64 kept by both models).
* Acute (0821): licking collapses at the far positions (far_L 6.4 → 2.2, far_R 7.3 → 1.6 licks / trial; far_R 0 / 14
  contacts); protrusion and rise / retraction speed drop 20-40 %. Chronic (0908): counts and speeds mostly recover;
  far_L stays impaired (contact 56 → 35 %, jaw movement 0.9 → 0.4).
* Direction: post-stroke CONTACT licks run image-left (= mouse RIGHT on cam4) of the pre-stroke successful path —
  close_center ~−20° (acute and chronic, both models), far_L −8° chronic; close_L −4/−5° (DLC) vs −17/−18° (LP).
  Chronic no-contact licks at far_L / far_center run far left (−35 / −18°). Carries head-pose / camera differences
  between sessions; confirm on more sessions and in 3-D.
  **The clearest view: `lick_direction_shift_PS93.png`** (all licks vs the pre-stroke successful path: mean paths in
  the spout frame, deviation over the lick, peak deviation with 95 % CIs) — chronic ~−20° at far_L and close_center.
* At far_L, full-length misses run 12-18° image-left of hits in every epoch, pre included — how far_L licks miss.
* Lick shape on a real-time scale: licks are ~60-80 ms tongue-out-to-in inside a 156 ms cycle (pre-stroke ILI).

## 4. Pitfalls (new; the 10-01 list still applies)

1. **Short licks read strongly image-left** from geometry alone (short vector, tongue slightly left of the mouth
   point) — compare like with like; the lip zone is masked.
2. **pandas reads an EMPTY first data row after a 3-row header as an index-name row** and drops it (a 1-row shift).
   `o2_inference.collect` reads the runner's .npz for that reason.
3. The figure-layout guard test flags `out / f"{stem}.csv"` as a figure sidecar — name non-figure outputs otherwise.
4. Never delete on MICROSCOPE — a bundle that needs fixing gets `bundle --refresh` (text only) or a new `--name`.
5. Shell heredocs with nested quotes / `\n` broke edits twice — write edit scripts to a file, or use the Edit tool.
6. DLC round 3 runs ~40 fps on this RTX 5060 whatever the batch (16 ms forward pass at 680×680) — whole sessions
   belong on O2.

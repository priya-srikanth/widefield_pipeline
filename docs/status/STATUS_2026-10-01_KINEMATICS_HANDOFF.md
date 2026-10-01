# Handoff — 2026-10-01: clean traces, spout reference frame, tongue/jaw kinematics port, first QC on our data

**START HERE for orofacial KINEMATICS (tongue / jaw traces, licks, angles, mismatch).** For the tracking models and
labelling (DLC round 3/4, Lightning Pose, cam1), the previous handoff
`STATUS_2026-09-30_LP_OCCLUSION_ROUND4_HANDOFF.md` still holds; §1 below lists what changed there today.
Reasoning for everything: `DECISIONS.md`, entries dated 2026-10-01 (several, with addenda). Analysis desktop;
**M: = MICROSCOPE, N: = standby since the 09-30 reboot — take paths from `PathResolver`, never type a letter.**

---

## 0. RESUME CHECKLIST

Nothing is running. The pipeline below exists end to end and has been exercised only on the **PS93 0908 cue clip**
(24 trials × cue −0.5 … +3.5 s, both DLC round 3 and LP occlusion predictions, at
`M:\MICROSCOPE\Priya\DeepLabCut\Widefield\lp_vs_dlc_cue_traces\PS93_20260908\`; LP CSV locally at
`%USERPROFILE%\lp_cue_tmp\cue_windows_LP.csv`).

1. **Open decisions for Priya** (do not pick for her):
   * **DLC likelihood cutoff 0.6 → 0.4?** Evidence: DAQ-contact recall 95.9 % → 97.7 %, +9 kept licks, the added
     licks' frames mostly show the tongue out. LP unaffected (near-binary confidence). Proposed, not applied.
   * **Fix the inherited one-frame v7 offset?** (DECISIONS "Tongue / jaw kinematics … PORTED"): peak indices are
     detect-window-relative but used as slice indices → x-at-peak, rise/fall, gate D/M windows, velocity windows read
     4 ms early; lick times correct. Kept for exact parity with stroke_orofacial. One-line fix.
2. **Next QC step (Priya agreed to QC every stage visually):** the ~60 kept licks per model with NO spout contact.
   The DAQ cannot see incomplete licks (no contact), so these are judged on video frames only: real incomplete
   licks vs false detections. Then retune px thresholds stage by stage (pre-clean → detector → gates) with
   `scripts/tongue_qc_steps.py`, in MOUTH-relative px.
3. Then: angles (`angle_max_signed_lick*` often has the opposite sign to the lick-1 angle — check the ±52 ms window),
   jaw (`jaw_pass_qc` fails all close_center trials for DLC — spout hides the chin?), velocities (2× between models).
4. Then full sessions: needs whole-session predictions (one batch inference per model; LP in WSL ONLY with
   16-frame DALI chunks + memory guard).
5. Still pending from before: labelling round 4 (Priya, 131 cam4 frames) + cam1 (student); retrain DLC + LP after;
   `python -m scripts.chronic_stability` once 0928 stage 2 has landed.

## 1. What changed today (10-01)

**Labelling / models**
* Round 4 add-on: one DLC-confident / LP-hidden frame per session (alternating tongue / jaw) → **131** cam4 frames
  to label (`dlc_hard_frames addon`; backup `round4_scan/round4_rows.before_addon.csv`). Worksheet rebuilt.
* LP-vs-DLC comparison figures (PS93 0908): cue-aligned trial overlays raw / median-5 / CLEANED, a zoomed 4-trial
  version with below-cutoff points coloured (DLC green, LP grey), and the "only one model confident" contact sheet.
  **LP's below-cutoff point is always the image centre (~338, 338 px)** — mask by likelihood, never use it.

**Code (all in `wfield_local/`, tests in `tests/`, config in `configs/defaults.yaml`)**
| module | what | provenance |
|---|---|---|
| `orofacial_clean.py` | x-range / lk / isolated drop / PCHIP gap fill / baseline; jaw v3.4 cleanup; snippets | stroke_orofacial v5p3 + jaw v3.4, transcribed |
| `spout_frame.py` | per-session frame: origin = where the close→far spout lines meet (the mouth), AP/LR axes, angles | ours |
| `trial_windows.py` | per-trial strobe / cue / **trial stop** (DAQ analog `trial_end`), `end_at_stop` | ours |
| `tongue_detect.py`, `tongue_kinematics.py` | v7 pre-clean, 3 detectors + merge, gates F/D/E/M, per-trial + per-lick features, angles in the spout frame | stroke_orofacial v7, **exact parity** (16 seeds) |
| `jaw_kinematics.py`, `tongue_jaw_mismatch.py` | jaw per-trial + "jaw moved"; mismatch = jaw moved & zero licks | stroke_orofacial, exact parity |
| `lp_labels.py` | audit / drop all-blank rows in an LP label file | ours |
| scripts | `pose_cue_traces.py` (plot/zoom/onlyone/--cleaned), `spout_reference_frame.py` (+examples), `pose_kinematics_demo.py`, `tongue_qc_steps.py` (--model, --lk), `tongue_contact_recall.py` | ours |

## 2. Decisions (with the evidence; full text in DECISIONS)
* **Spout reference frame is the reference** (Priya: nose/jaw can change post-stroke). The three spout lines meet to
  0.1–0.8 px in every session, on the mouth opening; stable within a session (< 4 px), shifts between sessions
  (PS93 20 px) are absorbed by rebuilding per session. Nose / resting jaw are MEASURED in the frame (a facial-weakness
  readout). Angle 0 = along the centre line, + = image-right = **mouse LEFT on cam4**. cam4 foreshortens AP; cam1 / 3-D
  for horizontal-plane angles. Far-L 3.5 vs 4 mm (PS93) moves along its line — no effect.
* **Tongue zero = the mouth** (spout-frame origin), not a percentile; jaw keeps the data-driven baseline. All tongue px
  thresholds are therefore mouth-relative (tongue y, PS93: rest ~40–50, median 103–111, lick peaks 45 / 148 / 207 at
  p5 / 50 / 95).
* **Windows end at each trial's own trial stop** (Priya: "interval between position strobe and trial stop, in case this
  changes"). Trial stop = DAQ analog `trial_end` (agrees with the log-mapped `trial_stop_ttl` to ≤ 4.3 ms); stop − cue
  median 3.75 s (3.5–6.0). Response windows end at stop; detection-slack windows at stop + 3 s (so a lick straddling
  the stop is measured whole, counted only if it peaks before the stop); starts (70 ms) and 1-s bins fixed. No stop →
  ported fixed windows (parity tests unchanged).
* **Gate F off** — it rejected 7/7 real licks on PS93 (tongue held on the spout; incomplete licks). Gate M kept
  (correctly rejects a second peak inside one lick).
* Min lick-peak height (< 5th percentile of tongue y) and the "real lick ~ median" levels are acceptable (Priya), but
  QC every stage visually before changing a number.
* Kinematics ported faithfully first (parity), adapted second: one spout at 6 positions instead of L/R; float cue
  frames; no writers / plots / cohort code.

## 3. Pitfalls
1. **Every px threshold in `orofacial_clean`, `orofacial_kinematics.*` is the OLD rig's** (Flea3 view) until retuned;
   ms values transfer (both 250 fps). Frame-count constants transfer only at 250 fps.
2. **A low-confidence LP point is the image centre** — any interpolation/averaging must mask first.
3. **The DAQ lick sensor sees contacts only** — it lower-bounds missed licks and says nothing about incomplete licks.
4. Trial stop is on the DAQ as an **analog** channel (`trial_end`), not digital — I first said there was none.
5. `clean_windows` on a cue-window clip: windows must be separated by more than the tongue slice (+stop+3 s) or
   trials overwrite each other in the session arrays (`pose_kinematics_demo` uses 2,200 NaN frames).
6. `ffmpeg` inside a piped WSL script needs `-nostdin`; WSL cannot mount SMB drives — stage through `/mnt/c`.
7. LP GPU prediction in WSL: 16-frame DALI chunks only (96 blue-screened the box twice).
8. Long jobs: detached + cached per session (a time-limited background task kills its children).
9. `test_no_hardcoded_machine_paths` blocks pushes on typed paths; `ruff --fix` can delete an import a later edit
   needs — re-run the file after fixing.
10. Other windows leave uncommitted edits in this repo — commit only your own files.

## 4. To do (not started)
* No-contact lick QC (frames) → px retune of pre-clean / detector / gates on mouth-relative px → then the
  orofacial_clean px thresholds (x-range, v3.4 ceilings).
* Decide DLC cutoff; decide the one-frame fix.
* Angle checks (signed-max window), jaw close_center QC, velocity comparison.
* Whole-session predictions (DLC + LP) and a session-level run; spout frame from full strobe → stop spans.
* Spout frame in cam1 once labelled; 3-D later.
* Unported from stroke_orofacial (deliberately): session-inclusion gate (old-cohort thresholds), plotting /
  visual bundles, cohort/day-binning, phase-aligned endpoints.

# Orofacial (cam4) findings: pre- vs post-stroke licking kinematics

Status: **preliminary**, written 2026-10-17. The numbers come from the cohort run on that date. Re-check every number
against `cohort/tables/prepost/stats/RESULTS.md` after each re-run; that file is re-written on every run, and this one
is not.

## What the data are

- **Pose model:** LP r4 with cleaning v35 (`LP_analysis/lp_r4_heatmap_20261013__clean_v35`).
- **Sessions:** 107 curated sessions (PS92 29, PS93 19, PS94 30, PS95 29). Lesions are all on the LEFT ventrolateral
  striatum, so "ipsi" = the mouse's left = image right.

| animal | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | 10 | 5 | 2 | 12 |
| PS93 | 6 | 4 | 3 | 6 |
| PS94 | 11 | 6 | 5 | 8 |
| PS95 | 10 | **1** | 7 | 11 |

- **Trials:** engaged trials only, using the behaviour pipeline's reference-position gate. That keeps 50,246 of 56,388
  trials. An ungated control is in `figures/prepost_alltrials`.
- **Licks:** LP-detected post-cue licks (cue + 70 ms to the DAQ trial stop). "Lick n" is the n-th detected lick,
  whether or not it touched the spout.
- **Statistics:** the change of each session from its own animal's pre-stroke mean.
  - Headline test: session Mann-Whitney with BH-FDR within each metric.
  - Robustness test: animal-weighted bootstrap.
  - Traces: within-animal cluster permutation.
  - Full methods: `cohort/tables/prepost/stats/STATS_METHODS.md`.
  - Below, "[a, b]" is the animal-weighted 95 % bootstrap CI.

## Findings

**1. An acute, position-graded motor deficit, worst at the far contralesional spout.**
All of the following are acute changes at far contra, significant by both tests, and present in all 4 animals:

| measure (far contra, acute) | change |
|---|---|
| fraction of licks touching the spout | −0.60 [−0.79, −0.42] (no-contact licks rise by the same amount) |
| first lick | +531 ms later [111, 856] |
| first spout contact | +953 ms later [246, 1379] |
| licks per responded trial | −5.4 [−6.5, −4.4] |
| measured peak protrusion | −31 px [−59, −22] |
| reach (protrusion / spout distance) | −0.15 |
| peak protrusion speed | −2,400 px/s |
| peak retraction speed | −1,560 px/s |
| lick duration | +20 ms |
| inter-lick interval | +25 ms [12, 42] |
| licks with a neighbour within 250 ms | −0.44 (isolated licks) |

- The same measures change at far mid and far ipsi, but less.
- Close positions barely change. Contact at close positions even rises slightly.

**2. Acute licks aim toward the ipsilesional side.**
- Executed peak angle shifts toward ipsi: far contra +29° [16, 39], far mid +11°, far ipsi +7°. Smaller shifts appear
  at all close positions.
- In the lick-aligned traces the lateral tip offset is present through the whole lick, both with and without
  in-mouth frames. At far contra it is about +70 px acutely.
- Subacute, the shift is about +8°; chronic, it is small but still significant at far mid (+3°).

**3. The jaw stops steering toward the target, in all 4 animals.**
- Jaw FAR_DIR, the sideways jaw displacement toward the target side, changes by −11.6 px acutely [−13.9, −10.1].
  It stays below baseline subacute (−5.0) and chronic (−4.4).
- This is not a shorter-reach effect: the trials where the tongue reached the spout show the same drop.
- Recovery timing differs by animal (DECISIONS 2026-10-16).

**4. The jaw moves without a lick on more acute trials.**
- These are trials where the jaw opens but no LP lick follows: +0.23 at far contra, +0.19 far mid, +0.13 far ipsi.
- This is the widefield analogue of stroke_orofacial's jaw-without-tongue dissociation.
- Caveat: LP cannot track a tongue within about 20 px of the mouth, so some of these trials may contain tiny licks.
  The planned example gallery (to-do 7 in the 10-16 handoff) is the check.

**5. Licking slows down and is more effortful.**
- Peak speeds fall at all far positions acutely.
- In the per-lick phase traces, speed magnitude |v| drops at the start of the lick and on the return.
- The phase portraits agree: slower into the turnaround at the same peak protrusion (far mid acute −0.5 to
  −0.8 px/ms around the peak, all 4 animals).

**6. Most of the acute drop in responding was quitting.**
- Responded fraction, all trials: 0.93 pre, 0.64 acute.
- Responded fraction, engaged trials: 0.97 pre, 0.82 acute.
- The engaged-trial acute deficit is still real and position-graded: −0.28 at far contra, −0.21 at far ipsi.
- Per-lick kinematics barely change with the engagement gate.

**7. Recovery.**
- Most measures return to baseline by subacute. Contact (−0.18), angle (+8°) and protrusion (−15 px) at far contra
  still differ subacute.
- Chronic, licks per responded trial are *above* baseline (+1.6 to +2.2 at far positions), and licks are shorter
  (−8 to −11 ms).

## Not yet established (do not quote)

- **Small chronic shifts** have no drift control yet: chronic protrusion −18 px at far ipsi and far contra, lick
  duration −8 to −11 ms, the far-mid angle +3°, and the chronic jaw FAR_DIR −4.4. Head position and camera framing
  can drift over months by this much. Wait for the pre-stroke early-vs-late drift control (to-do 8).
- **PS95 has one acute session**, so its acute values rest on a single day.
- **No within-trial homing:** the angle does not move toward the spout over licks 1 to 5. This was preliminary at the
  time and has not been re-checked on the full cohort.
- **The imaging-side readout of executed angle** is untested on whole sessions (to-do 3).

## Where the figures are

- **Folder:** `LP_analysis/lp_r4_heatmap_20261013__clean_v35/cohort/figures/prepost/`.
- **Deck:** `cohort/deck/orofacial_prepost_<date>.pptx`.
- **Key figures:**
  - `cohort_metrics_minus_pre.png`
  - `cohort_epochs_by_position_{accuracy,kinematics,engagement}.png`
  - `cohort_lick_{time,phase}_{first,all}_minus_pre_inmouth_{excl,incl}.png`
  - `cohort_trajectory_xy_within_*.png`
  - `cohort_phase_portrait_within_*.png`
  - `jaw_lateral/epochs.png`

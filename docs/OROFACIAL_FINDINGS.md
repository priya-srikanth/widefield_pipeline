# Orofacial findings: pre- vs post-stroke licking kinematics (cam4, plus 3-D and cam1)

Status: **preliminary, living document.** Last updated **2026-10-23** (pose / 3-D window). Rewritten 2026-10-18 for cam4;
the 10-23 pass adds the status table below, the 3-D / cam1 section (8) on all four animals, and corrections. Update the
section AND its status tag whenever a result changes, and add a line to the change log at the bottom.

Status tags: **STANDS** (both tests, August-only baseline, visible in >= 3 / 4 animals individually), **PROVISIONAL** (one
check pending or data incomplete), **PENDING** (not yet run), **CORRECTED** (an earlier statement here was wrong; the new
numbers follow), **NOT ESTABLISHED** (fewer than 3 animals, or one test only).

> **Where things stand (10-23), with what each rests on.** cam4 numbers = LP run v36, AUGUST-ONLY baseline (primary),
> `cohort/tables/prepost/stats/RESULTS.md` (10-20). 3-D / cam1 = `session_poses/pose3d/` and `session_poses/cam1_licks/`
> (rebuilt 10-23 on all four animals).
>
> | result | status | data it rests on / what is incomplete |
> |---|---|---|
> | 1. Acute global slowing (ILI +14 to +26 ms at every position) + far-contra-graded deficit (contact -0.67, first lick +488 ms, first contact +898 ms) | STANDS | cam4, 4 / 4 animals; PS95 acute = ONE session |
> | 1. Far-contra protrusion loss -26 px | PROVISIONAL (mostly a FAR-spout effect, see caveat) | far ipsi also drops in all four; 3-D reach drop is at far_R AND far_L in PS93 / PS94 (8.1) |
> | 2. Acute licks aim ipsi (+30 deg far contra, +7 to +13 deg elsewhere) | STANDS; confirmed in 3-D | cam4 4 / 4; 3-D 4 / 4 (far_R midline error +25 / +9.5 / +36 / +33 deg); PS95 acute = one session |
> | 3. Jaw stops steering toward the target acutely (FAR_DIR -11.4 px) | STANDS; confirmed in 3-D | cam4 4 / 4 (-7.8 to -13.9 px); 3-D 4 / 4 (-0.18 to -0.40 mm) |
> | 3. ... and does not fully recover (chronic -4.4 px) | **CORRECTED -> NOT ESTABLISHED per animal** | pooled cell is significant, but per animal only PS92 (-7.5 px) and PS93 (-5.5) keep a deficit; PS95 -2.9, PS94 -0.5; 3-D agrees (-0.16 / -0.14 / -0.01 / -0.02 mm) -> 2 / 4 animals |
> | 4. Jaw moves without a lick on more acute trials | STANDS; survives the cam1 correction | cam4 4 / 4; cam1-corrected (removes cam1-seen tongues) far_R pre -> acute 0.016 -> 0.28, 0.009 -> 0.28, 0.050 -> 0.39, 0.075 -> 0.58 (4 / 4). These are REACTIONS, not attempts (blind video review 0 / 24, 10-21) |
> | 5. Tongue cloud shift (W2), nose posture shift | PROVISIONAL | cam4 v36 W2 tables (10-18), not re-checked per animal in this pass |
> | 6. Most acute response loss was quitting | STANDS | cam4 |
> | 7. Chronic persistence (more licks / trial +1.5 to +4.2; shorter far licks; earlier first lick / contact) | STANDS for licks / trial; others PROVISIONAL | cam4 v36 August-only; per-animal check of each chronic cell not redone in this pass |
> | 8a. cam4 lateral / angle / jaw measures are faithful to 3-D | STANDS (4 / 4) | per-lick r 0.94-0.97 lateral, 0.99 'out' vs DOWN, 0.69-0.87 angle |
> | 8b. 3-D carries more target information than cam4 2-D | STANDS (4 / 4) | 6-way target decode from the lick endpoint 0.67-0.86 (3-D) vs 0.42-0.78 (cam4), August pre, LOSO |
> | 8c. Acute licks reach less far at far spouts, in 3-D | PROVISIONAL | far_R -0.06 / -0.63 / -0.44 / -1.50 mm, far_L +0.12 / -0.45 / -0.36 / -0.79: larger at far contra in PS95 / PS93, not in PS92 |
> | 8d. Incomplete (no-contact) licks rise acutely at far contra | STANDS (4 / 4); chronic PROVISIONAL | share of licks 0.17-0.38 pre -> 0.73-0.99 acute; chronic still raised in PS92 / PS94 only |
> | 8e. 3-D CLOSE_DIR (jaw away from a close side spout) | NOT ESTABLISHED | PS92 / PS93 only; PS94 / PS95 flat or opposite |
>
> **Coverage (10-23 evening): COMPLETE.** cam4: all 119 curated sessions. cam1 LP production: 127 / 127 curated videos
> (the last two, PS95_20260817 = PS95's only acute and PS95_20260831, landed 10-23 17:15). 3-D: all 119 curated sessions
> (PS92 30 / PS93 29 / PS94 30 / PS95 30). PS95 acute numbers (cam4, cam1, 3-D) rest on ONE session.

> **BASELINE RULE (Priya 2026-10-18): the AUGUST-ONLY pre baseline is PRIMARY.** Primary tables / figures:
> `cohort/{tables,figures}/{prepost,w2,heatmaps,jaw_lateral}/` of `lp_r4_heatmap_20261013__clean_v36`; the June-including
> versions (`*_allpre`, marked SUPERSEDED) may be quoted only for scale-free measures, as a sensitivity check. Run v35 is
> SUPERSEDED in full.

**What counts as a finding here -- CORRECTED 10-23.** This section used to require significance under BOTH baselines
(all-pre and August-only). Since the 10-18 baseline rule the criterion is: significant by **both** tests (session
Mann-Whitney with BH-FDR, and the animal-weighted bootstrap) under the **August-only** baseline (v36), AND visible in at
least three animals individually (CLAUDE.md rule 8). The all-pre (June-including) result is a sensitivity check only.
Numbers in findings 1-7 below were written as "all-pre / August-only" pairs on v35; where v36 August-only differs it is
given in the status table above, which is authoritative. Re-check against `cohort/tables/prepost/stats/` after each re-run;
this file is not regenerated.

## Data

- **Pose model:** LP r4 with cleaning v36 (v35 superseded: June camera rescale).
- **Sessions:** 119 curated sessions (PS92 30, PS93 29, PS94 30, PS95 30). The lesions are all LEFT VLS, so ipsi = the
  mouse's left = image right.

| pre sessions per animal | all curated pre | August only |
|---|---|---|
| PS92 / PS93 / PS94 / PS95 | 11 / 10 / 11 / 11 | 8 / 7 / 8 / 8 |

- **Post sessions per animal:**

| animal | acute | subacute | chronic |
|---|---|---|---|
| PS92 | 5 | 2 | 12 |
| PS93 | 4 | 3 | 12 |
| PS94 | 6 | 5 | 8 |
| PS95 | **1** | 7 | 11 |

- **Trials:** engaged trials only (the behaviour pipeline's reference-position gate): 56,629 of 62,771. The ungated
  control is `*_alltrials`.
- **Licks:** LP-detected post-cue licks (cue + 70 ms to the DAQ trial stop).

## Why two baselines: the pre period itself drifts

> **2026-10-18 evening: June camera scale.** June sessions are on a 12.9 % larger camera scale (spout ruler; DECISIONS).
> The pixel rows of the table below (protrusion, speed) are mostly that scale, not learning. Run v36 rescales June
> (x 0.8853); this file will be re-checked against v36. Until then, px results under the all-pre baseline are
> superseded; angles, timing, contact, counts and every August-only result stand.

Between June and August, before the stroke, all four animals changed a lot:

| June → August pre | change |
|---|---|
| close-spout contact | +0.40 to +0.47 |
| measured protrusion | −14 to −36 px at every position |
| peak protrusion speed | −550 to −1,140 px/s |
| lick duration | −2 to −15 ms |

This is learning and/or framing, not stroke. With the all-pre baseline it leaks into the "post-stroke" change. Several
chronic results existed only because of it: close-spout contact +0.13, chronic close-ipsi protrusion −11 px, and the
subacute speed losses. Those are dropped.

**Positional drift floor.** All positions here are in the mouth frame. Its control is the spout, a fixed apparatus
point that should not move relative to the mouth. Against the August reference, the spout's W2 is +1-3 px chronically
(bootstrap-only), so about **2-3 px** is the floor for chronic positional claims. Raw camera framing drifts 25-60 px over
months, which is why raw image positions are never used.

## Findings (v35-era text; status and v36 numbers in the table at the top)

**1. An acute, global slowing of licking, plus a position-graded deficit that is worst at the far contralesional spout.**

*Global (all or most positions, acute):*

| measure | change |
|---|---|
| inter-lick interval | +15 to +25 ms |
| peak protrusion speed | ~ −1,000 to −2,300 px/s |
| licks per responded trial | −1.5 to −5.5 |
| responded trials (engaged) | −0.08 at close spouts, −0.21 to −0.28 at far spouts |

*Far contra, acute:*

| measure | change |
|---|---|
| fraction of licks touching the spout | −0.61 / −0.67 |
| first lick | +521 / +488 ms |
| first spout contact | +940 / +898 ms |
| measured peak protrusion | −30 / −26 px (**caveat below**) |
| reach | −0.16 / −0.18 |
| lick duration | +20 ms |
| licks with a neighbour within 250 ms | −0.44 (isolated licks) |
| pre-cue licking | −0.38 Hz |

> **Protrusion caveat (2026-10-20, per-animal check, v36 August pre):** far-contra protrusion acute is −12 / −27 / −30 /
> −65 px (PS92 / 93 / 94 / 95), but **far ipsi also drops −9 to −19 px in all four**, so most of it is a far-spout effect.
> The far-contra-specific part (contra minus ipsi) is −4 / −8 / −16 / −46 px; PS95's −46 rests on its ONE acute session
> (96 far-contra licks), and PS94's first two acute days have 24 and 63. Contact loss (−0.37 to −0.83) holds in all four.
> 3-D (PS92, PS93) agrees per animal: per-lick reach far_R −3 % / −15 % (cam4 −7 % / −16 %). In PS92 the acute far-contra
> licks are full length but aimed ~3 mm toward ipsi; the trial-averaged "amplitude" loss is mostly fewer, later licks
> (one late full-size first lick, then no train). Details: `docs/status/STATUS_2026-10-20_POSE3D_TIMING_HANDOFF.md` §4a.

**2. Acute licks aim toward the ipsilesional side.**
- Executed peak angle shifts toward ipsi at every position. Far contra is the largest shift (about +29°); the close
  positions shift by about +9 to +13°.
- In the lick- and cue-aligned traces, the lateral tip offset toward ipsi lasts through the whole lick, both with and
  without in-mouth frames.
- Subacute, the shift persists at far contra (+8°) and far ipsi (+4°).
- Chronic, it is small but still present at close mid and far mid (+3 to +4°).

**3. The jaw stops steering toward the target acutely (all four). Chronic persistence: 2 / 4 animals only (CORRECTED 10-23).**
- Jaw FAR_DIR, the sideways jaw displacement toward the target side:

| epoch | change |
|---|---|
| acute | −11.5 / −11.4 px |
| subacute | −5.0 / −4.6 px |
| chronic | −4.7 / −4.5 px |

- The jaw opening angle shows the same pattern.
- FAR_DIR is a difference between far ipsi and far contra, so drift common to both cancels out. The chronic deficit,
  about 40 % of the acute loss, is therefore not explained by framing drift.
- **CORRECTED 10-23 (per-animal check):** the chronic deficit is carried by TWO animals. cam4 v36 FAR_DIR, August pre ->
  chronic (px): PS92 13.0 -> 5.5, PS93 8.2 -> 2.7, PS94 12.0 -> 11.5, PS95 15.3 -> 12.4 (`cohort/tables/jaw_lateral/
  epoch_by_animal.csv`); 3-D chronic change -0.16 / -0.14 / -0.02 / -0.01 mm. The pooled chronic cell stays
  significant, but "does not fully recover" is NOT ESTABLISHED under rule 8 (2 / 4 animals; PS95 partial). The ACUTE loss
  is in all four (cam4 -7.8 to -13.9 px; 3-D -0.18 to -0.40 mm in the three animals with acute 3-D).

**4. The jaw moves without a lick on more acute trials.**
- These are trials where the jaw opens but no LP lick follows: +0.23 at far contra, +0.19 far mid, +0.13 far ipsi.
- Still +0.03 to +0.04 at the far spouts subacute.
- Caveat: LP cannot track a tongue within about 20 px of the mouth, so some of these trials may contain tiny licks. The
  example gallery (to-do 7) is the check.
- **Checked 10-21 / 10-23.** (a) cam1 sees the tongue from below: removing trials where cam1 has a tongue event leaves the
  acute rise almost intact (cam1-corrected jaw-without-lick share, far_R: PS92 0.016 -> 0.277, PS93 0.009 -> 0.282, PS94
  0.050 -> 0.387; `cam1_licks/incomplete/`). (b) Blind video review of 24 jaw-only / just-below-floor trials (Priya,
  10-21): 0 / 24 were attempted licks; jaw responses without opening, jaw-up, walking-related movement. So these trials
  are REACTIONS, not attempts; the encoding window's attempted = hit + incomplete (jaw-only excluded). PS95 acute: cam1
  pending.

**5. The tongue's spatial distribution moves acutely and only partly returns (W2).**
- The tongue point cloud in the mouth frame moves away from the animal's own pre cloud at every position acutely: +14
  to +17 px at close spouts, +61 px at far contra.
- Subacute: +5 to +18 px.
- Chronic, under both baselines: +6 to +11 px at far ipsi and far contra, above the ~2-3 px spout drift floor.
- The jaw cloud moves acutely (up to +17 px at far contra). Chronically it stays +3.5 to +5 px, slightly above the
  floor; treat it cautiously.
- **The nose (snout / head posture) also moves:** +3-5 px acutely at every position and +3-4 px chronically, against
  the August baseline. The spout control does not move, so this is a candidate postural change after the stroke, not
  drift. It is small, and only clear against the August baseline.

**6. Most of the acute drop in responding was quitting.**
- Responded fraction, all trials: 0.94 pre, 0.64 acute.
- Responded fraction, engaged trials: 0.98 pre, 0.82 acute.
- The engaged-trial deficit is still position-graded.

**7. Recovery: most acute measures return by subacute. A few persist into the chronic period under both baselines:**
- Licks per responded trial are **higher** chronically, by +2.3 to +3.4.
- Licks are shorter at far ipsi and far mid (−7 to −10 ms).
- The first lick comes earlier at close mid (−78 / −103 ms).
- The first contact comes earlier at far mid (−90 / −171 ms).
- Protrusion is shorter at close contra and far ipsi (−9 to −16 px). That is near the positional drift floor AND
  PS93-sensitive (see below), so do not quote it.

**8. 3-D and cam1 (10-23, all four animals, complete).** 3-D = cam1 + cam4 triangulated per session (PnP on
the commanded spouts; `docs/POSE_MODELS.md` D10-D12). Per animal, post sessions minus the animal's August pre median
(`session_poses/pose3d/kinematics/sessions.csv`), order PS92 / PS93 / PS94 / PS95:

| measure (far_R = far contra) | acute | subacute | chronic |
|---|---|---|---|
| midline error toward ipsi, far_R (deg) | +25.0 / +9.5 / +36.1 / +32.7 | +5.4 / +5.7 / +14.8 / +5.2 | -0.8 / +3.1 / +8.5 / +1.6 |
| miss distance, far_R (mm) | +2.06 / +0.34 / +2.45 / +2.09 | +0.61 / +0.01 / +1.50 / +0.30 | +0.14 / +0.04 / +1.10 / +0.11 |
| reach far_R / far_L (mm) | -0.06 / -0.63 / -0.44 / -1.50 ; +0.12 / -0.45 / -0.36 / -0.79 | mixed | -0.24 / +0.31 / -0.85 / -0.30 ; small |
| jaw FAR_DIR (mm) | -0.26 / -0.18 / -0.40 / -0.31 | -0.23 / -0.17 / -0.15 / -0.04 | -0.16 / -0.14 / -0.02 / -0.01 |
| jaw CLOSE_DIR (mm) | -0.09 / -0.08 / +0.07 / +0.16 | -0.26 / -0.21 / +0.01 / +0.04 | -0.12 / -0.15 / +0.01 / +0.02 |

- **8a. cam4 is faithful to 3-D for what it measures** (`pose3d_crosscheck`, every cam4 v36 lick matched to its 3-D peak,
  95-97 % matched): lateral r 0.94-0.97 (34-36 px / mm), cam4 'out' = the DOWN axis (r 0.99 with -dv; -0.2..+0.1 with the
  true forward ap), protrusion vs 3-D reach r 0.97-0.99, peak angle r 0.69 (PS93) - 0.87. Jaw per trial r 0.66-0.83
  lateral, 0.57-0.83 opening. So findings 1-3 stand as measured; how far the tongue comes FORWARD is invisible to cam4.
- **8b. 3-D carries more target information** (`pose3d_vs_cam4`, target spout decoded from the lick endpoint, August pre,
  leave-one-session-out, 6 positions, chance 0.17): 3-D 0.81 / 0.67 / 0.86 / 0.74 vs cam4 0.53 / 0.42 / 0.78 / 0.59.
  Largest gain on centre close vs far (depth). CORRECTED from the 10-19 PS92-only line (0.80 vs 0.53), which holds.
- **8c. Reach** drops acutely at BOTH far spouts in PS93 / PS94 (not contra-specific), consistent with finding 1's
  protrusion caveat; PS92's far-contra licks are full length but aimed ipsi.
- **8d. Incomplete licks** (no contact; cam4 + cam1-only, `cam1_incomplete_licks`): share of far_R licks pre -> acute
  0.17 -> 0.86 (PS92), 0.38 -> 0.73 (PS93), 0.22 -> 0.98 (PS94), 0.25 -> 0.99 (PS95, one session); chronic 0.35 / 0.35 / 0.55 / 0.32 (pre 0.17 / 0.38 / 0.22
  / 0.25): persistent in PS92 and PS94 only. cam1-only events are 3-8 % of cam1 events in every animal and epoch.
- **8e. CLOSE_DIR** (jaw moves away from a close side spout, 10-20 in PS92 / PS93) does NOT generalise: PS94 / PS95 flat.
- Whole-session 3-D lick table (pre-cue + post-cue + ITI licks; ITI licks have no target) exists since 10-23:
  `session_poses/pose3d/kinematics_all/licks.csv.gz` (`context` column). Not yet analysed here.

## Robustness to PS93 (leave-PS93-out)

PS93 had a right orofacial deficit before the stroke: rightward tongue deviation and low far-ipsi contact. Pose
tracking on PS93 is not worse. All model validation (human lick review, contact recall 98.8 / 99.6 %) was done on PS93
clips.

`scripts/orofacial_robustness_table.py` re-runs every cohort cell under four variants: {all-pre, August-only} x
{with, without PS93}. It writes `cohort/tables/robustness/robustness.csv` and `robust_cells.md`, the full per-cell
list with a "without PS93" column.

**Of 85 cells significant by both tests under both baselines, 70 still hold without PS93.** The other 15 keep almost
the same effect size but drop to one test, which is expected with 3 animals.

- **Hold without PS93:**
  - every acute far-contra effect in findings 1-2 and 4;
  - the global acute slowing.
- **PS93-sensitive** (drop to a single test, or none, without PS93):
  - chronic protrusion at far ipsi (−16 px, "neither" under August-only without PS93) and acute / subacute protrusion
    at close contra;
  - chronic licks per responded trial at far ipsi / far contra;
  - chronic lick duration at far ipsi;
  - chronic first contact at far mid;
  - acute licks per trial and reach at far ipsi;
  - subacute far-contra jaw-without-lick.
- **Stronger without PS93:** chronic far-contra protrusion (−23 vs −14 px) and contact (−0.08 vs −0.04). PS93 dilutes
  these rather than drives them. Its acute far-contra changes run the same way, but smaller: contact −0.35 vs −0.67 to
  −0.82, angle +10 vs +27 to +42°. That fits an animal that already deviated at baseline.
- **Decision (Priya 10-18): keep PS93.** It is needed for n = 4 (CLAUDE.md rule 8: visible in at least 3 animals). It
  is the conservative member. Within-animal changes remove its baseline offset. Do not pool it for LEVELS, such as raw
  curves or absolute accuracy.
- Jaw FAR_DIR and W2 are not in the table: they come from their own modules. Their per-animal tables show PS93 in line
  with the others.

## Not established (do not quote)

- **Anything significant under only one baseline** (listed by the comparison in DECISIONS 2026-10-18):
  - chronic close-spout contact;
  - chronic close-ipsi protrusion;
  - subacute protrusion speeds;
  - the faster chronic retraction and earlier far-contra first lick seen only with the August baseline.
- **Chronic positional shifts under about 3-5 px** in the mouth frame (near the spout drift floor).
- **PS95's acute values**, which rest on a single session.
- **Within-trial homing.** The angle did not move toward the spout over licks 1-5 in the preliminary data; not
  re-checked.
- **Chronic jaw-steering deficit** (finding 3; 2 / 4 animals) and **3-D CLOSE_DIR** (8e; 2 / 4).
- **PS95 acute values** as levels: one session (cam4, cam1 and 3-D alike).
- **The imaging-side readout of executed angle** (to-do 3). Whether the imaging readouts also need an August-only
  sensitivity check is being scoped.

## Where

- **CORRECTED 10-23:** the paths below said v35. Primary is `lp_r4_heatmap_20261013__clean_v36`, where `prepost/` IS the
  August-only baseline and `prepost_allpre/` the June-including sensitivity version (stats: `cohort/tables/prepost/stats/
  RESULTS.md`; per-animal jaw: `cohort/tables/jaw_lateral/epoch_by_animal.csv`).
- **3-D / cam1 (DeepLabCut/Widefield/session_poses/):** `pose3d/kinematics/{licks.csv.gz, sessions.csv, by_animal.csv}`,
  `pose3d/kinematics_all/` (whole session), `pose3d/crosscheck/`, `pose3d/vs_cam4/`, `pose3d/spout_error/`,
  `pose3d/figures/summary_<animal>.png`, `pose3d/phase/`, `cam1_licks/` (+ `incomplete/`). Rebuilt by
  `python -m scripts.pose_chain` after every O2 copy-back.
- **Figures (older paths, v35 era):**
  - `LP_analysis/lp_r4_heatmap_20261013__clean_v35/cohort/figures/prepost/` (all-pre) and
    `prepost_latepre/` (August-only);
  - `figures/w2/` and `w2_latepre/`;
  - `jaw_lateral/`.
- **Decks:** `cohort/deck/orofacial_prepost_<date>.pptx` and `orofacial_prepost_latepre_<date>.pptx`.
- **Key figures:**
  - `cohort_metrics_minus_pre`;
  - `cohort_epochs_by_position_{accuracy,kinematics,engagement}`;
  - `cohort_lick_{time,phase}_{first,all}_{minus_pre,raw}_inmouth_{excl,incl}`;
  - `cohort_cue_{minus_pre,raw}_inmouth_*`;
  - `cohort_trajectory_xy_within_*`;
  - `w2/epochs_by_position`.

## Next

- 3-D jaw / nose onsets as appended `jaw_onsets.csv` columns, checked against Priya's 24 reviewed trials (10-23, in progress).
- Per-animal check of each chronic cell in finding 7 and of W2 (finding 5) on v36.
- Analyse the whole-session 3-D licks (`kinematics_all`, ITI / spontaneous licks; the encoding window's stray test uses them).

## Change log

- **2026-10-23** (pose / 3-D window): status table + coverage; criterion CORRECTED to August-only + rule 8; finding 3
  chronic jaw deficit CORRECTED to 2 / 4 animals (NOT ESTABLISHED); finding 4 cam1 check + attempt review added; new
  section 8 (3-D / cam1 on all four animals: cam4 faithful, 3-D > 2-D target decode in 4 / 4, acute ipsi aim and jaw
  steering loss confirmed in 3-D in 3 / 3, CLOSE_DIR not general, incomplete licks); paths CORRECTED to v36; PS95 acute
  3-D / cam1 pending (task 119).
- **2026-10-23 evening:** cam1 / 3-D COMPLETE (127 / 127; PS95 acute landed): rows 2, 3, 4, 8d now 4 / 4; 8c updated.
- 2026-10-20: protrusion caveat (far-spout effect; per-animal).
- 2026-10-18: rewritten after the full cohort review (v35; June scale issue noted).

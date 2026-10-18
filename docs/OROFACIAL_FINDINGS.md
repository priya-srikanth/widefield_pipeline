# Orofacial (cam4) findings: pre- vs post-stroke licking kinematics

Status: **preliminary**. Rewritten 2026-10-18 after a full review of the data with all 136 sessions processed.

**What counts as a finding here.** An effect is reported only if it is significant by **both** tests (session
Mann-Whitney with BH-FDR, and the animal-weighted bootstrap) under **both** pre-stroke baselines (Priya 10-18: "both, side
by side"):
- all curated pre sessions (June 0606-0608 + August);
- the August pre sessions only, immediately before the stroke (`--baseline late`).

The numbers below are "all-pre / August-only". Re-check against `cohort/tables/prepost{,_latepre}/stats/` after each
re-run; this file is not regenerated.

## Data

- **Pose model:** LP r4 with cleaning v35.
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

**Camera framing also drifts after the stroke.** Measured by W2 in raw image pixels against the tight August reference:
about 7-13 px pre, about 30-45 px chronic, in all four animals. In the mouth frame, the nose (a control) is displaced by
**+3-4 px** acutely and chronically. That ~3-4 px is the floor for any chronic positional change in the mouth frame.

## Findings that hold under both baselines

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
| measured peak protrusion | −30 / −26 px |
| reach | −0.16 / −0.18 |
| lick duration | +20 ms |
| licks with a neighbour within 250 ms | −0.44 (isolated licks) |
| pre-cue licking | −0.38 Hz |

**2. Acute licks aim toward the ipsilesional side.**
- Executed peak angle shifts toward ipsi at every position. Far contra is the largest shift (about +29°); the close
  positions shift by about +9 to +13°.
- In the lick- and cue-aligned traces, the lateral tip offset toward ipsi lasts through the whole lick, both with and
  without in-mouth frames.
- Subacute, the shift persists at far contra (+8°) and far ipsi (+4°).
- Chronic, it is small but still present at close mid and far mid (+3 to +4°).

**3. The jaw stops steering toward the target, and does not fully recover.**
- Jaw FAR_DIR, the sideways jaw displacement toward the target side:

| epoch | change |
|---|---|
| acute | −11.5 / −11.4 px |
| subacute | −5.0 / −4.6 px |
| chronic | −4.7 / −4.5 px |

- The jaw opening angle shows the same pattern.
- FAR_DIR is a difference between far ipsi and far contra, so drift common to both cancels out. The chronic deficit,
  about 40 % of the acute loss, is therefore not explained by framing drift.

**4. The jaw moves without a lick on more acute trials.**
- These are trials where the jaw opens but no LP lick follows: +0.23 at far contra, +0.19 far mid, +0.13 far ipsi.
- Still +0.03 to +0.04 at the far spouts subacute.
- Caveat: LP cannot track a tongue within about 20 px of the mouth, so some of these trials may contain tiny licks. The
  example gallery (to-do 7) is the check.

**5. The tongue's spatial distribution moves acutely and only partly returns (W2).**
- The tongue point cloud in the mouth frame moves away from the animal's own pre cloud at every position acutely: +14
  to +17 px at close spouts, +61 px at far contra.
- Subacute: +5 to +18 px.
- Chronic, under both baselines: +6 to +11 px at far ipsi and far contra. That is above the ~3-4 px mouth-frame drift
  floor.
- The jaw cloud moves acutely (up to +17 px at far contra). Its chronic shift of +3.5 to +5 px is at the drift floor, so
  it is not counted as a chronic effect.

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
- **Chronic jaw W2 and chronic positional shifts under about 5 px** in the mouth frame (the drift floor).
- **PS95's acute values**, which rest on a single session.
- **Within-trial homing.** The angle did not move toward the spout over licks 1-5 in the preliminary data; not
  re-checked.
- **The imaging-side readout of executed angle** (to-do 3). Whether the imaging readouts also need an August-only
  sensitivity check is being scoped.

## Where

- **Figures:**
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

# Findings: the current science conclusions (VLS stroke, PS92-PS95)

**The one place to read what we currently conclude.** Each claim is 2-4 lines: what, the key numbers, how many animals
agree, its status, and where the evidence is (figure / results page / detail doc). The per-arm FINDINGS documents stay
the detailed record (nothing is deleted from them; superseded claims are marked there in place). Started 2026-10-25
(Priya: consolidate the findings documents; all windows).

**Status tags.** **STANDS** (survived the listed checks on the data named) - **PROVISIONAL** (a control, a per-animal
check or the four-animal set still pending) - **CORRECTED** (an earlier statement was wrong; the corrected one follows) -
**WITHDRAWN** (no longer claimed) - **NOT ESTABLISHED** (tested, not supported; kept so it is not re-claimed).

**Cohort rules.** Four mice (PS92-PS95), all lesions LEFT ventrolateral striatum: RIGHT = contralesional for spouts,
licks and the body. Image sides: brain / encoding maps ipsilesional = image LEFT; cam1 + cam4 face video ipsilesional =
image RIGHT. Consistency threshold: >= 3 of 4 animals (Priya 10-24); every claim needs the per-animal table first
(CLAUDE.md rule 8). PS93 had a right orofacial deficit BEFORE the stroke and is named where it is the exception.
August-only pre-stroke baseline. Epochs: pre / acute / subacute / chronic (`wfield_local.epochs`). Changes vs pre:
`analysis_kit.boot_delta` (paired animals -> sessions, animal-weighted, 95 % interval); `*` = interval excludes 0,
`**` = also survives Holm over the figure's family.

**Owners (edit your own section; fetch + rebase before pushing).** 1-2: pose / 3-D window (5b). 3: imaging decode /
state control / anatomy (claims extracted from the docs by d4; owner to confirm). 4: encoding window (d4). 5: FaceRhythm
window (e3). 6: histology. 7: everyone.

**Results pages.** FaceRhythm https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1 - goal question / encoding
https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d - imaging https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV - licking kinematics (2-D + 3-D, sections 1-2) https://claude.ai/artifact/DKcKRJKbhBoratxxRc6qZv

## 1. Behaviour and orofacial kinematics (cam4)

*Owner 5b; detail [OROFACIAL_FINDINGS.md](OROFACIAL_FINDINGS.md) (its status table, v36, is authoritative). cam4 = LP
v36, August-only baseline, both tests (session Mann-Whitney + BH-FDR and the animal-weighted bootstrap); stats table
`LP_analysis/lp_r4_heatmap_20261013__clean_v36/cohort/tables/prepost/stats/RESULTS.md`. Checked by 5b 10-25.*

1. **Acute global slowing plus a far-contralesional-graded deficit.** Inter-lick interval +14 to +26 ms at every
   position; far contra: contact -0.67, first lick +488 ms, first contact +898 ms. **STANDS**, 4 / 4 (PS95 acute = 1
   session).
2. **Acute licks aim ipsilesionally**: +30 deg at far contra, +7 to +13 deg elsewhere. **STANDS** (cam4 4 / 4; 3-D 4 / 4,
   far_R midline error +25 / +9.5 / +36 / +33 deg).
3. **The jaw stops steering toward the target acutely**: FAR_DIR -11.4 px. **STANDS** (cam4 4 / 4; 3-D 4 / 4, -0.18 to
   -0.40 mm). "Does not fully recover" **CORRECTED -> NOT ESTABLISHED**: a chronic deficit only in PS92 (-7.5 px) and
   PS93 (-5.5); PS94 -0.5, PS95 -2.9; 3-D agrees.
4. **Jaw movement without a lick on more acute trials** (far_R +0.23). **STANDS**, survives the cam1 correction 4 / 4
   (far_R pre -> acute .02 -> .28, .01 -> .28, .05 -> .39, .08 -> .58). They are REACTIONS, not attempts (blind video
   review 0 / 24), so attempted = hit + incomplete. Source `session_poses/cam1_licks/incomplete/`; review page
   https://claude.ai/artifact/TbQ4pud711MBrT6ygAH5FX.
5. **Far-contralesional protrusion -26 px.** **PROVISIONAL**: mostly a far-spout effect (far ipsi drops too; 3-D reach
   drops at both far spouts in PS93 / PS94).
6. **Chronic: more licks per responded trial**, +1.5 to +4.2. **STANDS**. Other chronic cells not re-checked per
   animal: **PROVISIONAL**.
7. **Incomplete (no-contact) licks rise acutely at far contra**: share of far_R licks .17-.38 pre -> .73-.99 acute.
   **STANDS** 4 / 4. Chronically still raised only in PS92 / PS94 (**PROVISIONAL**).
8. **Licks and misses** (contact, DAQ + LP): far-contra trials are attempted acutely but almost never completed.
   Attempt rate, engaged trials, pooled (recount on the rebuilt trial_attempts.csv, 10-25): .79 / .79 / .58 / .34
   (PS92 / 93 / 94 / 95; n 336 / 292 / 450 / 120; all-trials denominator .62 / .61 / .445 / .342); completion .06 / .15 /
   .04 / .02. PS92-94 "try and fail" (incomplete >> no attempt), PS95 mostly "fail to try" ([ENCODING_FINDINGS 2.5]
   (ENCODING_FINDINGS.md)); completion back to pre chronically (.95-.99); per-lick contra misses mildly up chronically
   in 3 / 4. **STANDS** acutely / **PROVISIONAL** chronically ([FACE_RHYTHM_FINDINGS 5c](FACE_RHYTHM_FINDINGS.md)).

## 2. Pose, 3-D tongue and cam1

*Owner 5b; detail [POSE_MODELS.md](POSE_MODELS.md) Part D (status box) and OROFACIAL_FINDINGS section 8. 3-D = cam1 +
cam4 triangulated with cameras posed per session on the commanded spouts (PnP), mm in the stage frame, origin = the spout
mouth zero; complete (127 / 127 curated cam1 videos, all 119 curated sessions). Outputs `DeepLabCut/Widefield/
session_poses/pose3d/` (per-animal summaries `figures/summary_<animal>.png`); rebuilt by `python -m scripts.pose_chain`.
Checked by 5b 10-25.*

1. **cam4 measures are faithful to 3-D**: lateral r .94-.97 (34-36 px / mm); cam4 "out" is the DOWN axis (r .99 with
   -dv, about 0 with the forward axis), so forward reach is invisible to cam4; jaw per trial r .57-.83. **STANDS** 4 / 4
   (`pose3d/crosscheck/`).
2. **3-D carries more target information than 2-D**: target spout decoded from the lick endpoint (August pre, leave
   one session out, 6 positions, chance .17) 3-D .81 / .67 / .86 / .74 vs cam4 .53 / .42 / .78 / .59. **STANDS** 4 / 4
   (`pose3d/vs_cam4/`).
3. **Acute far-contra licks miss the spout and aim ipsilesionally in 3-D**: closest approach to the far_R spout +2.06 /
   +0.34 / +2.45 / +2.09 mm vs August pre (PS93 small), midline error +25 / +9.5 / +36 / +33 deg; partly back by
   subacute (+5 to +15 deg), near pre chronically except PS94 (+8.5 deg, +1.1 mm). **STANDS** acutely (aim 4 / 4, miss
   3 / 4); chronic **PROVISIONAL** (`pose3d/kinematics/sessions.csv`).
4. **Acute reach drops at both far spouts, not only contra** (far_R -0.06 / -0.63 / -0.44 / -1.50 mm, far_L +0.12 /
   -0.45 / -0.36 / -0.79). **PROVISIONAL**: a far-spout effect, matching 1.5.
5. **3-D CLOSE_DIR** (the jaw moves away from a close side spout). **NOT ESTABLISHED**: PS92 / PS93 only; PS94 / PS95
   flat or opposite.
6. **Whole-session 3-D licks exist** (pre-cue, post-cue and ITI; ITI licks have no target, so target errors are
   blank): `pose3d/kinematics_all/licks.csv.gz`, key `lick_uid`. A resource, not yet a finding; used by the encoding
   window's stray-lick test.

## 3. Imaging: position decoding, state control, where the code moves

*Extracted 10-25 by d4 from PRELIM_DATA_VLS_STROKE (09-26 render), BEHAVIOURAL_STATE_CONTROL, WHERE_THE_CODE_MOVES,
STATUS_2026-10-07, REST_ENGAGEMENT_AUDIT and DECISIONS, with 5b's 10-24 per-animal recheck (imaging page
https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV). **Reviewed 10-25 by 5b**, with e3's 10-25 per-animal / October /
mixed-model re-runs folded in. Imaging decode keeps its
default (June + August) baseline; an August-only rerun confirms the acute numbers (DECISIONS 10-18).*

**Caveat that qualifies several older claims: "no cortex is damaged" is CORRECTED.** BEHAVIOURAL_STATE_CONTROL, PRELIM,
VLS_STROKE_SYNTHESIS and CLAUDE.md argue specificity partly from "the lesion is striatal, no cortex is damaged";
HISTOLOGY.md records an ipsilesional ANTERIOR MOTOR-CORTEX injury in PS93 (and DLS involvement in some animals, from a
fibre-hit vessel). The position-vs-state ratio (claim 2) does not depend on it, but any PS93 frontal-map result should
be read with it, and the "rules out generic explanations" argument holds for 3 / 4 animals cleanly.

1. **A frozen pre-stroke decoder loses the far-contralesional spout acutely, graded by eccentricity.** Far contra
   .896 -> .321 acute (-.575, Bonferroni [-.738, -.394]); 5 / 6 positions significant acutely; far contra and near
   middle still below pre chronically (-.171, -.163). **STANDS** (PRELIM §2 / H1).
2. **Specificity: position loses far more than behavioural state.** Above-chance performance retained (current sidecar
   `epoch_13n_state_vs_position_cue.csv`): position .863 -> .428 / .696 / .761 (acute / subacute / chronic), state
   .976 -> .872 / .820 / .838, running alone ~.97. State DOES fall (acute -.069, Bonferroni interval excludes 0); the
   claim is the RATIO (~5x). **STANDS**; the state numbers in BEHAVIOURAL_STATE_CONTROL / VLS_STROKE_SYNTHESIS
   (.917 -> .812, .982 -> .918, .902) are **CORRECTED** to these. **Per animal (10-25, CORRECTED reading):** the cohort
   state drop is PS94 alone (above chance .978 -> .747 / .577 / .447); PS92 / PS93 / PS95 lose only .01-.09 (chronic
   .873 / .951 / .960), so state is nearly intact in 3 / 4 and the specificity claim is stronger there.
3. **The frozen decoder is NOT back at recovery, but that gap is within two months of no-stroke drift.** Deficit at
   behavioural recovery F(0) +.083, only PS92 recovered; refit-over-frozen G > 0 in 4 / 4 but a healthy mouse already
   shows G +.05-.10 (DECISIONS 09-12, 10-23). June -> August no-stroke drift group (5b, 10-24,
   `scripts/june_august_drift.py`): frozen-decoder deficit / refit gain June -> Aug vs Aug -> chronic .13 / .09 vs
   .03 / .11, .28 / .16 vs .09 / .21, .17 / .18 vs .18 / .20, .18 / .18 vs .11 / .11 -> not beyond drift (3 / 4).
   "Chronic reorganisation" (PRELIM §3, H5) is **CORRECTED -> NOT ESTABLISHED**; "decoder back" **CORRECTED**.
   October sessions now included (10-25 re-render; the 10-07 table had dropped 2 chronic sessions / animal): frozen
   chronic level / pre 1.04 / .85 / .83 / .90, flat 4 / 4; G still rising in 3 / 4 (mixed-model log-day slope +.030,
   z 4.5) but does not track behaviour (q .16) -- drift-sized per the June -> August group.
4. **Cortical position-coding gain follows behaviour, lagging ~1 week, not practice.** 3 / 4 (not PS94). **STANDS**
   (DECISIONS 10-19). Plateau by the end of imaging in every decode / encoder readout, 4 / 4 (STATUS 10-07, F18);
   the October decoder rerun after the `_day` fix (660bcbe) was DONE 10-25 (e3): verdicts unchanged.
5. **Acute map loss graded by eccentricity, far contra largest.** Rest-referenced, engagement-gated,
   cue window: far-contra acute amplitude / pre .50 / .45 / .53 / .40 (4 / 4); chronic .90 / .98 / .60 / 1.12 -- these
   per-animal numbers are the POSITION-WEIGHTED rest reference (15r RESTW; 5b found 10-25 that the page's "rest"
   figure and table were RESTW). Plain REST pooled gives the same picture: far contra .48, far middle .66, near ipsi
   1.43 acutely (RESTW .47 / .68 / 1.46). **STANDS**; the
   09-12 QUIET / MEAN amplitude tables in WHERE_THE_CODE_MOVES are **CORRECTED** (superseded by 15r).
6. **The acute far-contra map change has the same shape across animals** (between-animal r MEAN +.82, PRECUE +.86 vs a
   cross-position null ~.02-.08, p <= .001). **STANDS** acutely; chronic agreement is at its own null (DECISIONS
   09-17). Near-position "increases" under the MEAN reference are a coupling artefact (**STANDS**).
7. **Region level, acute.** Licks recruit more bilateral orofacial sensorimotor cortex (SSp-m / SSp-n lead, 4 / 4, all
   three references); at far contra the cue-evoked increment falls 1.7-3x more in the ipsilesional hemisphere (4 / 4).
   **STANDS** acutely; chronic region cells **PROVISIONAL** (the `15k` table predates the full chronic cohort).
8. **Representational geometry.** Crossnobis distance from each position's pre template: far contra +1.02 acutely;
   chronic +.30 to +.78 at all six positions. The still-RISING chronic distance is drift, not ongoing reorganisation
   (within-chronic rate not above each animal's pre drift floor, 4 / 4; F18). But chronic distance from the August
   template exceeds June -> August drift in 3 / 4 (Aug-vs-June / chronic-vs-Aug .63 / 1.01, .95 / 1.16, .43 / .43,
   .59 / .81; June -> Aug includes learning, so an upper bound). "Chronic diffuse = replacement" (PRELIM H4)
   **PROVISIONAL**.
9. **Rest carries spout position** (engagement-gated): observed / null 1.634, 43 / 44 sessions, 4 / 4; per-session
   decode 94 / 96; persistence, not anticipation (r_prev - r_next +.077, 4 / 4); a different code from the task
   (cosine .09 / .16). **STANDS** (REST_ENGAGEMENT_AUDIT §3; ungated predecessors 1.622 / 1.443 **CORRECTED**). "Rest shows
   the same acute dip" **WITHDRAWN** (DECISIONS 09-15).
10. **Pre-cue (ENL) position code: no consistent acute loss** (up PS92 / PS95, down PS93 / PS94; the slow band accounts
    for PS93's drop). Coding-direction leave-one-out: held-out far-contra pre-cue projection drops acutely 3 / 4 (not
    PS94), **PROVISIONAL**. Mixed model re-run 10-25 on the rebuilt attempts table: step at the stroke -.53 (z -3.0,
    q .019), but it NO LONGER tracks the time-to-contact deficit (q .25; was .023 on the old table; contact deficit
    q .050, borderline).
11. **Haemodynamics: CLOSED** (the data cannot answer it; no evidence of changed 415 / 470 coupling).
12. **The pre-stroke spatial template (simple position encoder, family 11): amplitude falls in 4 / 4 but returns in only
    2 / 4; its shape stays off in 4 / 4.** Per animal amplitude a pre -> chronic .96 -> 1.19, .96 -> .84, .93 -> .68,
    .91 -> .73; variance captured below pre chronically 4 / 4 (.45 / .38 / .47 / .24 vs .71 / .73 / .66 / .56). "Amplitude
    fully returns" (the pooled 09-10 three-animal number .749 -> .745) is **CORRECTED**; the shape claim **STANDS**.
13. **Session mixed models (re-run 10-25 on the rebuilt attempts table, `scripts/imaging_lmm.py`, both deficit
    definitions):** every decoder / template readout steps down at the stroke, recovers with log(days) and tracks the
    behavioural deficit session by session (frozen -.040 / unit, z -3.2; template amplitude z -8.7; best-match z -7.6).
    The decoder / template rows **STAND**. Position-coding TREND rows re-run 10-25 ~08:45 on d4's final-frame trend
    tables (107 August+ sessions): the ceiling (category - none) has NO acute step (-.014, z -1.0), rises with log-day
    (+.014, z 2.4, q .017) and is chronically above pre (+.038, z 5.0) in 3 / 4 (pre -> chronic .041 -> .108, .058 ->
    .108, .047 -> .076; PS94 flat .040 -> .037). Ceiling vs deficit is **DEFINITION-DEPENDENT**: significant against the
    contact deficit (-.050 / unit, z -3.9, q < .001) but NOT against time-to-contact (z -1.5, q .13) -- the 01:40
    "z -4.0, now significant" is **CORRECTED** (that run read stale trend tables). The per-spout SPREAD tracks the deficit
    under both definitions (z -4.5 ttc / -5.3 contact; pre-cue spread -4.2 / -4.6). Imaging page section 13.
14. **Map drift without a stroke (June pre -> August pre, 15r maps; `scripts/map_drift_prepre.py`, 10-25; imaging page
    §7, drift panels).** CUE window, REST reference, contact gate, 4 / 4 animals (12 June / 32 August sessions), ratio = amplitude vs the
    reference epoch, sig bins of 2022 (nested bootstrap, maxstat). Far contra: drift .95 (7 bins) vs acute .49 (1178
    bins) -> the acute far-contra loss (claim 5) is far beyond drift, **STANDS**. Far MIDDLE is the exception: drift .77
    (244 bins) is as large as its acute change .71 (151 bins), so the far-middle acute loss is **not separable from
    drift**, and the pooled CD leave-one-out far_center chronic -.51 (4 / 4; scored against August pre only) needs the
    same caveat. Far ipsi drift .76 (294 bins) vs acute 1.06 (6 bins). Near positions: drift .82-.94; stroke changes
    are increases (near ipsi acute 1.51, chronic 1.43), the opposite sign. June includes learning -> drift is an upper
    bound. LICK window, REST, LP-lick gate (the headline lick map, imaging page §7.3): far contra drift 1.04 (15 bins) vs
    acute .67 (500 bins), beyond drift; far middle drift .90 (12 bins, rim-suppressed) vs acute 1.07 -- no far-middle
    acute loss in the lick window, so the far-middle caveat is specific to the cue window; near positions drift
    .86-.98 vs acute increases 1.32-1.58. LICK, REST, contact gate: far contra drift 1.11, acute empty (contact gating
    leaves < 20 acute far-contra trials -- read the LP panel), far middle drift .93 vs acute 1.29, near drift .86-.98
    vs acute 1.32-1.58. LICK, PRECUE, contact: far contra drift 1.32 UP (the cue-evoked lick increment grew June ->
    August, consistent with learning; acute empty under contact), near / far-middle / far-ipsi drift .86-1.08 vs acute
    increases 1.34-1.92. CUE, PRECUE, contact: far contra drift 1.10 (33 bins) vs acute .18 (1,883), far middle .83
    (30) vs .55 (800) -- both beyond drift. PRE-CUE window (2 s), REST, contact: the far-contra pre-cue map FALLS with
    time alone (drift .67, 67 bins); acute goes UP 1.85 (339 bins), beyond drift; chronic .65 (0 bins) is drift-sized,
    so a chronic pre-cue decrease is not a stroke effect. Contact panels complete (CSV
    `map_drift_prepre/contact/drift_vs_stroke_panels.csv`).
    LP-lick gate (`map_drift_prepre/LP_licks/`): cue window as contact (far contra REST drift .94 / acute .49; PRECUE
    1.11 / .18). LICK, PRECUE, LP -- the only lick-window panel with an acute far-contra estimate under PRECUE: drift
    1.24 (72 bins) vs acute .56 (436), opposite signs, beyond drift. PRE-CUE 2 s, REST, LP: far contra drift .65 /
    acute 1.84 / chronic .81 (0 bins) -- as contact. ENL 1 s, REST, contact (`DAQ_ENL1s/`): far contra drift .63 / acute
    2.21 (186 bins) / chronic .58 (0) -- the same picture at 1 s. **New, consistent across gates and windows: the
    chronic FAR-MIDDLE pre-cue map loss is beyond drift** -- chronic .53 (572 bins, 2 s contact), .30 (1,051, 2 s LP),
    .35 (954, 1 s) against drift .86-.97. Near-ipsi pre-cue ratios (2-3x, 0 bins) are unstable (near-zero reference
    amplitude), not results. All three arms complete 10-25. **PROVISIONAL** (June includes learning; June ~3 sessions
    per animal) until the remaining references / windows / arms render.

## 4. Encoding model and the goal question

*Owner d4; detail [ENCODING_FINDINGS.md](ENCODING_FINDINGS.md) (section numbers below), results page
https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (figure numbers as of 10-25). Frozen pre-stroke encoder = final frame
`gate-attempted_v7g3_t3d` (3-D tongue, attempted-trial gate, combined cam1 + cam4 licks). Within-animal changes, nested
bootstrap unless stated; LICK DATA = behaviour only.*

1. **The acute far-contralesional drop in cortex is carried by the LICK, not the target.** Frozen pre-stroke encoder
   decomposition: the acute drop = the predicted lick part in all model orderings, target part ~0 (<= 1.7 x1e3).
   Pooled: measured -.53 / -.30 of the pre response (ipsi / contra SSp-m), predicted -.62 (4 / 4). **STANDS** (3.4;
   Fig 8, 8b).
   **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** unchanged: predicted -.58 / -.61 vs measured -.30 / -.53 (SSp-m contra /
   ipsi, lick), target part ~0; FaceRhythm takes part of the lick share (page Fig 9-9b, 1).
2. **Chronically cortex responds MORE than the pre-stroke mapping predicts for the same behaviour (the residual).**
   Residual +.32 / +.43 of the pre response (ipsi / contra SSp-m) with predicted change ~0, 4 / 4. Pixel maps,
   calibrated bins (random pre splits within the nominal 5 %): all six spouts pooled, 32 px, lick window 19 / 35 / 30
   of 128 bins (acute / subacute / chronic), every one an INCREASE (413 up, 2 down across all tests), bilateral
   lateral frontal (ALM / tongue-jaw motor) + contralesional SSp; within animal 4 / 4 contralesional. By area (pre-SD
   units, chronic): MOp +1.8* / +2.3*, SSp-m +2.1* / +2.5*, SSp-n +.9 / +2.5*, MOs +.7 / +1.4*, contralesional >
   ipsilesional in every area; a raised plateau, not continuing drift (3 / 4). **STANDS**; the earlier "pooled pixel
   test weak" (8 px) is **CORRECTED** (8.4; Fig 9-9g, 10, 11, 11b).
   **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** NOT face movement: SSp-m chronic residual +9.4 / +9.4
   -> +9.5 / +10.6 (x1e3 dF/F, cue; lick +12.4 / +14.0 -> +10.9 / +13.4), 4/4 (page Fig 1, 1b).
3. **The behaviour-to-cortex mapping is rewritten chronically (Layer 2).** One fit per animal, epoch kernel changes from
   early-August pre: chronic gain beats the late-pre null (cortex +54.8 [+22.0, +87.4], SSp-m +52.5 / +51.6) and every
   one of 20 random pre pseudo-epochs in 4 / 4 (cortex and both SSp-m); grows with time; spread across kernel groups
   (none alone beyond the null). **STANDS** within August; its nulls do not include June -> August drift (cf. 3.3).
   The original pooled-refit version was **WITHDRAWN** (10-24; 8.5; Fig 12-13b).
   **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** unchanged: chronic change beyond pre drift cortex
   +0.055 -> +0.054, SSp-m +0.052 -> +0.059 / +0.061 (4/4). The FR direction weights change no more than
   pre-stroke drift (page Fig 1 panel 4).
4. **Licks to far contra go to the wrong spout from launch (LICK DATA).** Landing soft bins: own bin -.69 / -.38 / -.19,
   far mid +.35 / +.36 / +.20 (4 / 4, significant); whole tracked 3-D path: own -.65 / -.36 / -.22, acute toward far
   ipsi +.42 (part general forward overshoot) then far mid; mis-landing licks follow the other spout's course from the
   first tracked point. **STANDS** (2.4b, 8.1; Fig 3-3e, 5, 5b).
5. **On average the frozen pre-stroke DECODER reads far-contra trials as the spout the tongue goes to.** Own -.66 /
   -.30 / -.22, far mid +.26 / +.14 / +.11, far ipsi +.19 acute (4 / 4; cue window the same). But acutely the decoder's
   diagonal falls for EVERY spout while those licks land on target (general weakening). **STANDS** (8.2; Fig 3f, 3g).
6. **Trial by trial, cortex follows execution before and after the stroke, but NOT acutely.** Within session, neural
   (far mid - far contra) tracks executed (far mid - far contra) pre-stroke 4 / 4 (r .12-.41), subacute and chronic
   4 / 4; acutely r ~ 0 (only PS92 significant). Executed-bin vs decoder matrix: acute diagonal -.41 (far mid), -.34
   (far ipsi), 4 / 4. **STANDS** (8.2; Fig 3h-3j).
7. **When a lick misses, cortex mostly still looks like the TARGET, except at the far spouts.** Off-target trials,
   decoder = target vs = executed: close spouts .65-.95 vs ~.05 at every epoch; far contra .66 / .21 pre -> .25 / .29
   acute (executed wins only here) -> .45 / .33 chronic; far ipsi / far mid split acutely. **STANDS** (8.2; Fig 3k, 3l).
8. **Encoder relabel test: no single answer for "which spout is cortex in".** Best-fitting label (target + spout-locked
   pattern swapped, executed licks kept): far contra loses its label toward far mid / close contra but other spouts
   lose theirs too (close contra -.58 acute, far mid -.44 chronic); per animal PS95 re-aimed at far mid (transient),
   PS93 stays far contra, PS92 / PS94 nonspecific. TRIAL LEVEL (10-25, Fig 3o-3q): far contra loses its own label in
   every epoch (-.14 / -.17 / -.14, 4/4) but far ipsi as much (-.23 / -.21 / -.17), and trials at EVERY spout drift to the
   close-mid label (close ipsi +.20 acute, 4/4): a label-wide bias, not far contra re-aimed; on off-target far-contra
   trials the encoder reads target ~.12-.15, executed ~.17-.23, another spout ~.65 (unlike the decoder, 8.2).
   **PROVISIONAL** (2.7, 8.3; Fig 3m-3q).
9. **Failed attempts lose the side but keep the distance.** Frozen decoder on video attempt classes: acute failed
   (incomplete) far-contra attempts are read as A far spout (distance kept, side lost; 4 / 4, lick window); no
   pre-cue plan signature; retained code ~.1 of pre-stroke hits at every epoch while hits recover to .71. **STANDS**
   (2.8; Fig 6-6c).
10. **Failed attempts look like weaker hits, not like no-movement.** Same bilateral pattern; lower amplitude than hits
    (3 / 4) but not than touch-matched pre-stroke misses (2 / 4); ipsilesional ALM / tongue-jaw motor down acutely,
    tongue-jaw motor / SSp up chronically. **PROVISIONAL** (2.9; Fig 7-7c).
11. **Target vs executed: chronically cortex carries the target beyond the executed movement.** Binned contest (unique
    R2 of target bins beyond executed bins): chronic change +9.0 [+3.6, +16.0] lick, +9.8 [+3.9, +16.2] cue, at every
    softmax width; acute lick +12.5 (n.s.; PS93 opposite). Pre-stroke "cortex tracks target, not executed" remains
    **SUSPENDED** (2.4). **STANDS** chronically (2.4b, 8.6; Fig 1-2b) -- in the PER-SESSION REFIT frame only.
    **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** **REVISED.** Frozen target-unique R2 (the spout beyond every executed
    regressor) FALLS acutely and stays DOWN chronically (cortex lick -0.016 [-0.028, -0.004], 0/4 up; with FR -0.008,
    0/4); the executed movement's spout-locked part RISES acutely (cue cortex +0.041, 4/4) and stays slightly up with FR.
    The refit chronic target gain is not reproduced when the weights are frozen (page Fig 1e, 1f). CAVEAT (10-25): the refit and frozen numbers are SCORED differently (refit: cue 0-1 s, lick +/-0.2 s around EVERY lick peak, prediction offset penalised; frozen: cue 0-1.5 s, lick -0.1..+1.0 s around the FIRST lick, data and prediction centred separately), so "not reproduced" mixes frame and scoring; the per-session self-fit in the frozen windows (`scripts/session_vs_frozen.py`, running) separates them.
12. **Face movement (FaceRhythm) explains about half of the spout-position code and most of its chronic rise.**
    **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** the post-stroke CHANGE claims below were per-session refit; in the
    frozen frame: (a) contralesional position | FR rise NOT reproduced (position unique flat; contra - ipsi -0.010
    without FR) -- the SPOUT package (target + executed spout part) shifts contralesionally instead, contra - ipsi
    +0.039 [0.011, 0.081] cue / +0.048 [0.015, 0.094] lick with FR, 4/4 (page Fig 1c, 1d); (same scoring caveat as 11); (b) the spout code is a PETH
    code and the acute loss is a PETH loss (STANDS), but it does NOT recover chronically (PETH target unique cortex lick
    -0.096 without / -0.054 with FR, 0/4; Fig 1g, 1h); (c) FR direction unique in SSp-m falls chronically, but the
    direction weights do not change beyond drift (Layer 2): less direction signal through the same mapping, MIXED
    behaviour / neural; (d) recovery course: frozen spout unique pre -> chronic up 2/4, cortex reaches half its change at
    or after behaviour (Fig 1i); (e) movement WITHOUT lick-position information (pose + FR) gains unique R2 in SSp-m
    chronically (ipsi +0.050 [0.023, 0.074] 4/4), position-carrying movement alone falls contralesionally (quick run,
    Fig 1j, 1k). PROVISIONAL. Refit statements kept below for the record:
    Per-session encoder with FR (master_pre_v1, rank 12): unique position R2 pre-stroke 52 -> 30 x1e3 (cortex, lick);
    chronic rise +22 [+7, +38] -> +4 (n.s.) with FR (cue +39 -> +15 [+3, +27]); the acute ipsilesional SSp-m position
    loss survives FR; FR's side / distance information adds little beyond its lick amount (modes A ~ B ~ S ~ D).
    Musall-style with FR (Fig 16d-16h): FaceRhythm alone ~310-380 x1e3 (cortex), ~120-155 unique; it overlaps mostly
    with UNINSTRUCTED movement (that group's unique R2 halves, 241 -> 109 pre, lick window); in the lick window FR's
    unique R2 rises acutely (135 -> 156) while every other group's falls. Pixel maps: the chronic rise in position
    unique is broad without FR, focal contralesional anterior-lateral with FR. TESTS (10-25, Fig 16i-16k, R2 as a
    fraction): pixel test (16 / 32 px, calibrated 5 % false-bin rate) finds NO chronic bins for position unique | FR (69
    without FR), 13 up for task unique | FR; a priori AREA test: chronic position unique | FR in CONTRALESIONAL SSp-m cue
    +0.037 [0.010, 0.067] 4/4 (p_bonf .002), lick +0.025 3/4 (n.s. after correction), ipsilesional ~0; contra - ipsi
    +0.025 / +0.032 (CIs exclude 0); task the same; uninstructed ALONE rises in SSp-m on both sides (4/4). Beyond pre drift
    in PS94 / PS95 only (PS92 marginal, PS93 not); not a denominator effect (total variance rose).
    TRIAL vs PETH (8.9): FR explains trial-to-trial variability, the spout code lives in the PETH; the ACUTE spout loss is a PETH loss.
    DIRECTION (8.9): FR's side / distance factors explain less SSp-m acutely and chronically (0/4 up); the contralesional rise
    is present with directionless FR (cue +0.028, 4/4). Total variance rises (global). Frame = per-session refit encoder.
    WHY (8.9, model-free, Fig 16p): acutely the face's spout signal weakens (side eta2 -0.10, tongue -0.12) and FR tracks the
    tongue less (behaviour + measurement); chronically those are back at pre, but within-spout side movement is -21 % (3/4)
    and ipsilesional SSp-m loses its coupling with the FR side index (partial r -0.10 -> ~-0.02, 4/4), contra - ipsi -0.07
    [-0.11, -0.03] 4/4 -> chronic decoupling is CORTICAL (ipsilesional), not a measurement failure. Small r (<= 0.2). PROVISIONAL.
    **PROVISIONAL** (FR factors owned by e3; position beyond FR is a lower bound; carried by PS94 / PS95) (8.7, 8.9; Fig 16-16q).
13. **Not established / withdrawn here:** pre-stroke target-over-executed lead (SUSPENDED, 2.4); pooled toward-midline
    within-spout effect (NOT ESTABLISHED, 10-21); misdirected licks carry a plan signature (NOT ESTABLISHED, 2.5);
    lick-level strays decide target vs execution (NOT DECISIVE: reference fails validation, 2.6); Layer 2 pooled-refit
    claim (WITHDRAWN 10-24).

## 5. FaceRhythm: face movement on one frozen pre-stroke factor set

*Owner e3; detail [FACE_RHYTHM_FINDINGS.md](FACE_RHYTHM_FINDINGS.md) (section numbers below), page
https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1. Frozen rank-12 master `master_pre_v1` fit on pre-stroke sessions of
all four animals; every session projected onto it; expression = ABSOLUTE (weights x FR normaliser), log2 or ratio vs
each animal's own pre. Factor names, not numbers (numbers shift between fits).*

1. **Acute: the contralesional lick factor falls more than the ipsilesional one** (confirms the known acute deficit).
   Peak lick_right, all trials, acute .30-.46 of pre (log2 -1.37**), 4 / 4; on contra-spout trials -.60 [-.67, -.52];
   ipsi-spout lick_left n.s. at every epoch. **STANDS** (5d).
2. **The contralesional deficit persists PER LICK while the lick count recovers.** lick_right per right lick, log2 vs
   pre: acute -1.06**, subacute -.44*, chronic -.57 [-.79, -.34]** (chronic ratio .57-.79, 4 / 4 below pre at every
   epoch); right licks per trial back to pre chronically (-.14, n.s.; PS94 .52 the exception). lick_left per left lick
   n.s. **STANDS** (5b-master); same by peak amplitude on contra-spout trials, chronic -.30 [-.40, -.21], 4 / 4 (5d).
3. **Substitution: contralesional spouts are reached with more of the ipsilesional lick pattern.** lick_left on
   contra-spout trials subacute +.94 [.34, 1.54], chronic +1.08 [.35, 1.87] (1.2-3.3x pre), 4 / 4; not on ipsi-spout
   trials. A re-mixing of the pre-stroke repertoire, not a new movement (5d, 6b, 6-master). **STANDS**.
4. **Contra / ipsi face asymmetry at the contralesional spouts, per lick.** Contra-minus-ipsi half of the face, change
   vs pre, lick-aligned: far_R acute -.24**, chronic -.14 [-.23, -.05]**; close_R chronic -.09 [-.14, -.04]**; every
   epoch and alignment, 4 / 4. Speed maps (cue 0..+4 s, per point): close_R -.27** / -.16** / -.19**, far_R -.47** /
   -.29** / -.22* (acute / subacute / chronic); at the R spouts each lick moves the contralesional lower face less and
   the ipsilesional face more (per-point maps, >= 3 / 4 + FDR). n.s. at the L and centre spouts. **STANDS** (5e, 5f).
5. **The pre-stroke vocabulary fits post-stroke movement worse acutely.** EVR ratio frozen / own, vs pre: acute -.035
   [-.045, -.025] 4 / 4; subacute -.008, 4 / 4; chronic -.008 [-.016, -.0003], 3 / 4 (PS92 / PS94; PS93 fully back).
   **STANDS** acutely, **PROVISIONAL** chronically (6-master).
6. **The contralesional lick movement is re-timed later in the response window, not delayed at its start.** lick_right
   centre of mass, contra-spout trials: subacute +.40 s, chronic +.41 s [.23, .59], 4 / 4 later; first lick on the same
   trials EARLIER chronically (-.06 s, 0 / 4 later). **STANDS**, caveat: a trial average mixes within-trial timing with
   which trials carry the movement (per-bout timing pending) (5g).
7. **Survey of every factor (peak amplitude).** Acute drops in the shared lick and pre-lick nose / sniff factors
   (lick_ongoing_7Hz -1.22**, lick_fast_close -1.01**, nose_prelick_slow -.78**, sniff_prelick -.68**); chronically
   several factors ABOVE pre (lick_fast_close +.72**, lick_ongoing_7Hz +.50**, lick_far_nose +.42**, whisk_run +.42**).
   **STANDS** (5d).
8. **Non-task face movement is reduced acutely, not preserved** (the 2pRAM cohort's relative preservation does not
   replicate): between-trial movement falls within behavioural state, rest -.99 log2 [-1.24, -.80] 4 / 4, other -.37
   4 / 4; back to pre chronically. **STANDS** (6d).
9. **Treadmill pushes are inside the pre-stroke face repertoire**: frozen-basis error at pushes is below random times in
   27 / 32 animal x epoch x context cells, although pushes are 2-4x more frequent chronically. No pre + post master
   needed. **STANDS** (6b).
10. **The lick factors change shape after the stroke, becoming less side-specific rather than re-aimed.** Pooled per
    animal x epoch fits: lick_right / lick_left / lick_ongoing beyond the pre-half floor in 4 / 4 at every epoch;
    lick_right's loading leaves the contralesional near-midline face for the mouth rim and ipsilesional cheek (PS92 /
    94 / 95; PS93 opposite); it still resembles the R spouts' pre pattern, less strongly. Per-session matching:
    heterogeneous across animals. **PROVISIONAL** (descriptive, one value per animal x epoch) (6c).
11. **"Acutely fewer licks, not smaller ones" -- WITHDRAWN (10-24, Priya).** The face mesh cannot see tongue extent or
    lick rate; the DLC / LP kinematics show acute licks are slower and shorter (section 1). The acute per-trial drop =
    fewer, later and slower / shorter licks (5f).
12. **Severity does not order by lesion burden** (largest acute task drop in the mildest lesion, PS95). Descriptive,
    n = 4 (6d; section 6).
13. **The face carries spout position BEFORE the cue, on lick-free trials** (Priya 10-25; `scripts/fr_enl_decode.py`,
    FACE_RHYTHM_FINDINGS 7). 12 frozen factors in 0.5 s bins, trials with no LP lick and no spout contact in the
    window, the imaging decoder (logistic C .5, position-block CV), chance .167, 107 sessions. Pre-stroke 2 s window
    .43 (per animal .39 / .47 / .49 / .38; 31 / 31 sessions p < .05); 1 s .39. FR smooths acausally (~0.3-1 s), so
    features stopping 0.5 s BEFORE the cue are the clean test: 2 s window .34 (31 / 31), 1 s window .25 (18 / 31) --
    the position signal is real but sits mostly in the second before the cue. Acutely lower (2 s, above-chance change
    -.10 [-.21, +.01], 3 / 4; end -0.5 s -.08 [-.17, +.02], 3 / 4), back chronically (.46). The acute drop is FAR
    CONTRA: far_R recall .42 -> .19 (chance) -> .41 chronic; other spouts change little. **PROVISIONAL**. Bearing on
    imaging 3.10: the cortical pre-cue position code (.33-.55) sits in the same range as the face's, so part of it may
    be face posture / movement; a movement-removed cortical ENL decode with FR as regressors is the test. Untested:
    whether the FR mesh sees the spout itself (the video motion-energy check masked the spout; FR has no such check).

## 6. Histology

*Detail [HISTOLOGY.md](HISTOLOGY.md).* Lesion burden = 100 x ipsilesional / contralesional caudoputamen area per
coronal section (lower = more striatum lost), mean over sections after exclusions (10-21): PS93 45.0 %, PS92 46.7 %
(most severe), PS94 67.7 %, PS95 81.9 % (mildest). **STANDS** (5-7 sections per animal). Stroke-extent schematic on
Paxinos plates: `wfield_local/histology_map/`.

## 7. Open questions and withdrawn claims (index)

* WITHDRAWN / CORRECTED / NOT ESTABLISHED claims are listed in their sections above; collect the cross-arm ones here.
* Stale elsewhere (flagged 10-25, not yet edited): CLAUDE.md's pointer to STATUS_2026-09-20 says "NO within-session motor
  fatigue"; DECISIONS 09-20 revised it the same day to "partly fatigue" (acute within-bout slope +10.1 ms; within-bout
  deceleration +15-25 ms at every epoch, 4 / 4).
* Open (section 3-4): October decoder rerun after the `_day` fix; chronic region-level (`15k`) cells on the full chronic
  cohort; FR trial-to-trial vs PETH split (O2 fr_A_tr, lined up 10-25); drift-group null for Layer 2 / residual (June sessions lack
  FaceRhythm and have off kinematics, so a June -> August encoder null is not available).
* Open: per-bout timing of lick_right (5.6); early-vs-late pre-day floor for factor shape (5.10); other cameras for
  FaceRhythm (cam1-3).

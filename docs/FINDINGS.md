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
8. **Licks and misses** (contact, DAQ + LP): far-contra trials are attempted acutely (64-80 %, PS95 32 %) but almost
   never completed (.03-.15); completion back to pre chronically (.95-.99); per-lick contra misses mildly up chronically
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

*Owner to confirm (d4 is extracting the claims from BEHAVIOURAL_STATE_CONTROL, WHERE_THE_CODE_MOVES,
VLS_STROKE_SYNTHESIS, PRELIM_DATA_VLS_STROKE and DECISIONS). Known corrections to carry: the state-decoder numbers in
BEHAVIOURAL_STATE_CONTROL are stale (current = the 10-15 sidecar); the WHERE_THE_CODE_MOVES QUIET / MEAN tables are
superseded by 15r REST / RESTW; "decoder back at recovery" CORRECTED. New 10-24 (5b): a June -> August no-stroke drift
group puts the chronic frozen-decoder deficit within drift (3 / 4) while chronic crossnobis distance exceeds it
(3 / 4).*

(pending)

## 4. Encoding model and the goal question

*Owner d4; detail [ENCODING_FINDINGS.md](ENCODING_FINDINGS.md), page https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d.*

(pending; includes FaceRhythm in the encoder: FR takes ~half of the unique spout-position R2 pre-stroke and most of
the chronic position-coding rise, lick window +22 -> +4 n.s. with FR; modes A ~ B ~ S ~ D -- d4 to state and tag)

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

## 6. Histology

*Detail [HISTOLOGY.md](HISTOLOGY.md).* Lesion burden = 100 x ipsilesional / contralesional caudoputamen area per
coronal section (lower = more striatum lost), mean over sections after exclusions (10-21): PS93 45.0 %, PS92 46.7 %
(most severe), PS94 67.7 %, PS95 81.9 % (mildest). **STANDS** (5-7 sections per animal). Stroke-extent schematic on
Paxinos plates: `wfield_local/histology_map/`.

## 7. Open questions and withdrawn claims (index)

* WITHDRAWN / CORRECTED / NOT ESTABLISHED claims are listed in their sections above; collect the cross-arm ones here.
* Open: per-bout timing of lick_right (5.6); early-vs-late pre-day floor for factor shape (5.10); other cameras for
  FaceRhythm (cam1-3).

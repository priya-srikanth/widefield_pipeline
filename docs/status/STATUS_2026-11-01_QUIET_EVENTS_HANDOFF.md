# Handoff 2026-11-01: quiet-period cortical events after the VLS stroke (cohort1, PS92-95) + literature context

**START HERE for the quiet-period events arm.** Resume from section 4 (open tests, in priority order). Results live on
the imaging page section 15 (https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV, v38) and in
[`docs/COHORT2_QC_2026-10-29.md`](../COHORT2_QC_2026-10-29.md) (quiet-event bullets, 10-31 .. 11-01). Section 6 = how
the results sit against the stroke literature (Priya asked for it 11-01; also offered as a FINDINGS / goal-page section).

Cohort rules apply: cohort1 only (never pooled with cohort2), August-only pre, per-animal table before any test, image
left = ipsilesional on brain maps.

---

## 1. What is running (11-01 ~20:45)

| job | command | output | state |
|---|---|---|---|
| (DONE 11-01) quit-period state (cortex + face, quit vs engaged quiet) | `python scripts/cohort2_qc/quiet_quit_state.py --jobs 6` (background, log `scratchpad/quit_state.log`) | `labcams/cohort1_quiet_spikes/v2_quiet/quit_state/` (`sessions/*.npz`, `quit_state_sessions.csv`) | finished 108 / 108; result in 3.8 |

Nothing is running. Kill only the exact PID if needed (never by name / parent).

Environment for every script here: `set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache`; python =
`C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe`; FaceRhythm basis = `QS_FR_BASIS=own` (the scripts set it as
default; it is inherited by spawn workers).

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

## 4. Open tests (priority order; discussed with Priya 11-01)

1. **Selection control for 15.7 (decisive, do first).** Pre events are mostly movement-locked (whisking activates
   barrel / limb cortex); acute ones are not, and are 5x more numerous (smaller events dominate the average). Split
   events by face movement (move flag already in the v2 FR session npz: `move_raw_pos/neg`) AND match height, then
   compare pre non-movement vs acute non-movement maps. Same maps -> the stroke made non-movement events common;
   still more local acutely -> the event itself changed.
2. **Ipsi vs contra onset / amplitude** of acute orofacial events (31 Hz = ~32 ms lead-lag resolution). Motivation:
   VLS -> lateral SNr -> VM / VAL thalamus -> ALM / tjM1 is the licking loop (Lee, Wang & Sabatini 2020); thalamic
   projection mostly ipsilateral -> ipsilesional-first / larger predicts a loop origin; simultaneous bilateral onset
   predicts a common (state) driver. 15.6 shows ipsi > contra SD in PS93 / PS94 only.
3. **Up vs down event maps**: same local footprint -> local excitability oscillation (fits 2-3 Hz + faster decay).
4. **Breathing check (open since 10-31)**: 2-3 Hz = mouse resting respiration. Coherence of quiet orofacial trace with
   the FaceRhythm breathing / sniff factors, pre vs acute.
5. **State**: (quit run DONE, result 3.8) compare the pre-stroke QUIT-period event map with the acute engaged map (same
   local footprint?); then within engaged quiet, does acute event rate rise with face
   stillness? A cortex-wide state change would not explain a map confined to orofacial cortex, but can add to it.
6. **Behaviour link**: (a) do the same events occur in pre-cue / ITI windows; (b) does a pre-cue event predict a failed
   far-contra attempt or a weaker pre-cue position code on that trial; (c) within animal, event rate vs far-contra
   recovery with the time trend removed (as `scripts/iti_recovery_link.py`); (d) pre-register PS93 (largest acute rate)
   / PS94 (least recovered) orderings for cohort2.
7. **Ramanathan tension**: their reach-locked low-frequency oscillations FALL after stroke and their return tracks
   recovery; ours (quiet, spontaneous) RISE. Test movement-locked low-frequency power in our lick windows vs the quiet
   increase in the same sessions.
8. Offered, not started: a "Context in the field" section for FINDINGS.md / goal-question page (section 6 is the draft).

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

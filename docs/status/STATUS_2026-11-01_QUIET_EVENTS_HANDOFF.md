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
| (DONE) T1-T4 event-map split | `quiet_eta_split.py` | `v2_quiet/eta_split/` | finished 107 / 107 |
| (DONE) quit-period state | `quiet_quit_state.py` | `v2_quiet/quit_state/` | finished 108 / 108; result 3.8 |

 Kill only the exact PID if needed (never by name / parent).

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
| T9 | breathing: coherence of quiet orofacial trace with FaceRhythm breathing / sniff factors, pre vs acute; respiration rate change | ARTEFACT (breathing) | coherence high / tracks respiration -> artefact | RUNNING 11-01 (agent; `quiet_breathing.py`, `V2/breathing/`) |
| T10 | haemodynamics: the same events / band power in the 415 nm channel and in uncorrected 470 | ARTEFACT (haemodynamic) | events absent in 415 -> neural | RUNNING 11-01 (agent; `quiet_hemo_check.py`, `V2/hemo415/`) |
| T11 | amplitude distribution; count of events > 10 % dF/F and cortex-wide synchronous events (Steinmetz 2017 interictal shape) | ARTEFACT / epileptiform | none -> not interictal | DONE: not interictal (big events rarer acutely, less limb co-occurrence) -- running list |
| T12 | events in pre-cue / ITI (task) windows, not only quiet | relevance | -- | RUNNING 11-01 (agent; `quiet_task_events.py`) |
| T13 | pre-cue event on a trial -> failed far-contra attempt / weaker pre-cue position code on that trial | CIRCUIT-noise mechanism for the deficit | events predict failure | RUNNING 11-01 (agent; same script) |
| T14 | within animal, event rate vs far-contra recovery with log-day removed (as `iti_recovery_link.py`) | link to recovery | negative relation | DONE: r -0.75 raw, -0.23 detrended (p .048, 2/4) |
| T15 | rate tracks each animal's deficit (PS93 largest acute rate; PS94 least recovered); pre-register orderings for cohort2 | CIRCUIT vs STATE / ARTEFACT | -- | DONE: no ordering (rho -0.20) |
| T16 | Ramanathan tension: movement-locked low-frequency (1-4 Hz) power in lick windows vs the quiet increase, same sessions | reconcile with Ramanathan 2018 (task LFOs fall) | task LFO down while quiet up | RUNNING 11-01 (agent; `quiet_lick_lfo.py`) |
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
cortex shows (Poulet et al. 2012 thalamic control of cortical state; Crochet & Petersen 2006 larger evoked responses in
quiet wakefulness; both from memory, verify); (3) a whole-VLS lesion removes direct AND indirect pathways with opposing net
effects on SNr, and SPNs fire little at rest, so the clearest loss is the phasic, lick-locked SNr pause (timing / gain), not
mean drive. Against: the fluctuation increase is not ipsilesional (T8). Tests T20-T22; direct VM measurement in a future
cohort would settle it.

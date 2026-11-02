# Running list 2026-11-01: quiet-period cortical events, hypothesis tests (cohort1 PS92-95)

Priya 11-01: "proceed with all tasks, parallelize where possible, and keep a running list of tests completed, results, and
links to the figures". Test numbers = handoff [`STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md`](STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md)
section 4 (what each test separates and predicts). Figures live under
`M:\MICROSCOPE\Priya\Widefield\labcams\cohort1_quiet_spikes\v2_quiet\` (abbreviated `V2\` below); page figures =
imaging page https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV section 15.

## Decisions needed from Priya

* **Where the arm lives**: recommended = a section on the encoding (imaging + kinematics) page, spout page section 15 ->
  pointer; alternative = its own page. Will move once T1-T4 are in unless told otherwise.

## Status board

| test | status | result (one line) | figure |
|---|---|---|---|
| base: event rate | DONE 11-01 | face-regressed orofacial events x5.1 acute, x2.5 sub, x1.4 chronic, 4/4 | `V2\fr_regressed_own\quiet_events_face_regressed.png` (page 15.1) |
| base: maps | DONE 11-01 | quiet SD up over most of cortex; acute orofacial events local to anterolateral orofacial cortex | `V2\fluct\quiet_sd_maps.png` (15.6), `V2\fluct\quiet_eta_maps.png` (15.7) |
| T1 selection control (movement x height) | DONE 11-01 | MOSTLY SELECTION: pre-stroke still (no face movement) events are ALREADY local to anterolateral orofacial cortex; the pre lateral spread came from movement events. Acute still events x3-12 (46->553, 151->1060, 125->1251, 274->382) vs movement events ~x2. Residual after movement + height matching: acute still events less lateral lead-in (-0.3 s) and less lateral recruitment at +0.2 s (significant); spread 0.41/0.35/0.58/0.40 -> 0.34/0.31/0.29/0.37 (4/4, small except PS94); chronic spread ABOVE pre (PS92 1.00, PS93 0.63) | `V2\eta_split\T1_up_still_pre_vs_acute.png`, `T1_up_move_pre_vs_acute.png` |
| T2 ipsi vs contra onset / amplitude per event | DONE 11-01 | onset lag ipsi - contra within +/-5 ms every animal x epoch, log2 amplitude ratio within +/-0.2, no consistent sign. CORRECTED 11-01: 31 Hz + GCaMP6s cannot resolve 5-10 ms callosal delays, so this does NOT show a common driver -- an ipsilesional origin spreading via callosum is not excluded; only the amplitude symmetry stands | `V2\eta_split\events_ipsi_contra.csv` |
| T3 up vs down maps | DONE 11-01 | down events: same still-vs-move split (still spread 0.62-0.72), acute slightly more local in 2/4 -- same structure, weaker | `V2\eta_split\T3_down_still_pre_vs_acute.png` |
| T4 pre quit-period vs pre engaged map | DONE 11-01 (revised after Priya's question) | pre QUIT and pre ENGAGED still events have the SAME map (delta: only a few bins; the lower spread index 0.22-0.25 vs 0.35-0.58 is not a map difference) -> ONE event type whose RATE depends on state: still-event rate pre engaged 0.13 / 0.43 / 0.50 /min (PS93-95) vs pre quit 8.4 / 10.5 / 3.9 (x8-60); acute engaged 2.0-5.7 (approaching pre quit), acute quit 14-24; movement events are NOT more frequent in quit. Map similarity is consistent with a state-like change but does not identify state (T7 tests that). Pre quit sparse (PS92 none) and late in session | `V2\eta_split\T4_up_still_pre_eng_vs_pre_quit.png` |
| 15.7 MOVEMENT-CORRECTED maps + decay (Priya 11-01: replace the page map; regress movement from all) | RUNNING (`quiet_eta_fr.py`, forward 5 + reverse 4 workers, ~3 min / session) | -- | `V2\eta_fr\quiet_eta_maps_fr.png`, `V2\eta_fr\quiet_eta_decay_fr.png` |
| T5 event shape | DONE 11-01 | acute events ~10 % narrower, all decay (-18 %), rise unchanged, ACF unchanged (3/4) -> oscillation, not sharper bursts | `V2\width\quiet_event_width.png` |
| T6 quit-period state | DONE 11-01 | pre-stroke QUIT already has orofacial 1-4 Hz x3.4-5.4 with cortex-wide 1-4 Hz LOWER, face stiller; engaged-quiet face speed halves acutely 4/4 | csv `V2\quit_state\quit_state_sessions.csv` (no figure yet) |
| T7 rate vs face stillness within engaged quiet | DONE 11-01 | acute engaged quiet is STILLER (time at < 0.5 x pre face speed 16 % -> 34 %) BUT the rise holds at MATCHED stillness in every bin, 4/4: epoch effect log10 rate +0.98 (stillest) / +0.95 / +0.60 / +0.48 / +0.32 (most active), all q < .001; subacute and chronic too. And the rise is LARGEST when still: acute rate in the stillest bin 20-33 /min vs 7-14 when most active, whereas pre rates are flat or lower when still (raw events include movement-linked ones). -> a stiller face explains only a small part; after the stroke, stillness itself comes with many events (as in pre-stroke disengagement) | `V2\stillness\quiet_stillness.png` |
| T8 ipsi vs contra rate change | DONE 11-01 | NO ipsilesional bias: orofacial ipsi - contra vs pre +0.02 acute (n.s.; 3/4 slightly ipsi, PS92 contra); limb CONTRA > ipsi acutely (-0.11, q .01; PS92 / PS94) | `V2\hemi_amp_recovery\quiet_hemi_amp_recovery.png` (top left, middle) |
| T9 breathing coherence | RUNNING (agent) -- KEY (Priya 11-01: respiration looks FASTER post-stroke): respiration rate by epoch, cortical peak frequency tracking respiration, coherence at the respiration frequency, and the CORRECTION (respiration regressed out, then 1-4 Hz power + event rate recomputed) | -- | `V2\breathing\` |
| T10 415 nm / uncorrected 470 | RUNNING (agent) | -- | `V2\hemo415\` |
| T11 amplitude / epileptiform | DONE 11-01 | NOT interictal-like: median heights 2.4-3.1 %; events > 10 % dF/F < 1 % of events and RARER acutely (0.05-0.14 % vs 0.5-0.9 % pre); orofacial events SHARE limb events less acutely (pre .14-.40 -> acute .07-.25) = more local, not cortex-wide. Exception: PS94 chronic has 11 events > 10 % (1.5 %) -- look at them | same figure (top right, bottom left) |
| T12 events in pre-cue / ITI windows | RUNNING (agent) | -- | `V2\task_events\` |
| T13 pre-cue event -> failed attempt / weaker code | RUNNING (agent) | -- | `V2\task_events\` |
| T14 rate vs far-contra recovery (within animal) | DONE 11-01 | rate falls as behaviour recovers: r -0.75 raw (4/4, p < .001); beyond the shared time course -0.23 (p .048; PS93 -0.58, PS95 -0.49, PS92 +0.17, PS94 -0.03) -> mostly the common time course | same figure (bottom middle) |
| T15 deficit ordering (n = 4) | DONE 11-01 | no ordering: acute rate vs half-recovery day / lesion % rho -0.20 each; PS93 (smallest lesion, fastest recovery) has the LARGEST acute rate (x7.8) | same figure (bottom right) |
| T16 lick-locked LFO vs quiet (Ramanathan) | RUNNING (agent) | -- | `V2\lick_lfo\` |
| T17 literature search + citation check | DONE 11-01 | 3 corrections (eLife 90080 = two-photon, local; circuit = VLS -> SNr -> VM/Pf -> tjM1 + SNr -> brainstem, ALM via Guo 2017; Allegra Mascaro = combined therapy); precedents to add (Bauer 2014 subcortical connectivity, Cullins & Connor 2024 orofacial maps, Ahmed 2017 licking); counter-literature (Branscheidt 2022, Zeiger 2021, Sarasso lacunar, Panzeri 2025); epileptiform caveat (Steinmetz location / rate overlap) | text: FINDINGS 8 'LITERATURE UPDATE 11-01 late' |
| T19 Steinmetz criteria (waveform, refractoriness, polarity, symmetry) | DONE 11-01 | NOT interictal: acute up/down ratio ~1 (0.7-1.0; interictal = positive only), no refractory gap (IEI 0.3-0.5 s at or above Poisson), heights 2.4-3.1 %; matches only rate (0.10-0.18 Hz) and bilateral symmetry, shared with pre-stroke events | `V2\steinmetz_check\quiet_steinmetz_check.png` |
| T20 synchronisation index (Priya's reduced-drive hypothesis) | TO DO | -- | -- |
| T21 pre-lick slow power -> lick response size (pre sessions) | TO DO | -- | -- |
| T22 per-area F0 minus 415 part, ipsi vs contra | TO DO | -- | -- |
| T18 k-choice bias | NOTE | acute contrast optimistic (k picked on pre vs acute); subacute / chronic unbiased | -- |

## Running interpretation (update as tests land)

* SELECTION (T1, T4): the acute 15.7 map is mostly WHICH events -- the stroke multiplies a pre-existing, local,
  non-movement event type, with only a small further narrowing of its lateral recruitment. Pre-stroke this one event type
  is rare in engaged quiet and x8-60 more frequent in the quit period (same map in both states); acutely it becomes
  common while engaged (rate approaching pre-quit levels). The "event changed" reading is weak; the "event became common" reading is strong.
* ARTEFACT: epileptiform ruled out (T11, T19). Breathing (T9) and haemodynamics (T10) pending.
* CIRCUIT: no ipsilesional bias in rate (T8) or amplitude (T2); onset timing cannot separate a common driver from
  callosal spread at 31 Hz. Priya's reduced-thalamic-drive hypothesis is NOT refuted: dF/F cannot see tonic level, and
  the acute effects (bigger slow fluctuations, bigger evoked responses) are what a less-driven, synchronised cortex shows
  (handoff section 7; tests T20-T22).
* STATE: T4 + T6 + T7 -- acute engaged quiet is stiller (more time at low face speed), but the rate rise persists at
  matched stillness (every bin, 4/4) and is largest when the face is still: after the stroke, stillness goes with many
  orofacial events, as it does pre-stroke only when disengaged. A shift in state composition is a minor part.
* RECOVERY: rate tracks recovery mostly via the shared time course (T14); no between-animal ordering (T15).
* LITERATURE (Priya 11-01): Zeiger 2021 (potentiation, not remapping) is compatible -- use "re-weighted mapping", not
  "remapping"; Cullins & Connor 2024 only partly agrees (see FINDINGS 8).

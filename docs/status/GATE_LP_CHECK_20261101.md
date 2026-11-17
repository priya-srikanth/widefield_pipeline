# GATE vs LP CHECK — does the DAQ-contact engagement gate / quit point mislabel real LP lick attempts? (2026-11-01)

Priya-approved check. PROVISIONAL, n = 4 (PS92-95; cohort 2 not touched). Script `scripts/gate_lp_check.py`
(`extract` per-session pickles in `WFIELD_INPUT_CACHE/gate_lp_check/`, `analyse` tables + figure). Outputs:
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/gate_lp_check/`
(`per_session.csv`, `per_animal_epoch_counts.csv`, `quit_shift.csv`, `quit_shift_per_animal.csv`, `pooled_boot.csv`,
`coverage.csv`, `figdata/gate_lp_check.pkl`, `figures/gate_lp_check.png`).

## Question and decision thresholds (fixed before looking)

Every engaged / working gate and the S28 quit point use DAQ spout CONTACT. Post-stroke, far-contra attempts often miss
the spout. Does the gate score trials that contain real LP lick attempts as disengaged / quit, more so acutely?

| analysis | materially affected if |
|---|---|
| imaging classes (stopped vs working) / maps / Haufe | > 5 % of gate-STOPPED trials hold an LP attempt, in >= 3 / 4 animals or pooled, and post > pre |
| S28 quit tests | per-animal mean quit shift > 5 min, or a detected quit disappears in >= 2 animals |
| pre-cue (ENL) decoders | > 10 % of DAQ-lick-free pre-cue windows hold an LP lick, and differing by epoch |

## Method

* Sessions: `analysis_kit.curated_labels(PS92-95)` with `WIDEFIELD_PRE_FROM=0801` (August-only pre), epochs from
  `epochs.epoch_of`: 108 sessions. PS93_0813 has no video (no LP) and is excluded from LP fractions; 69 PS92 chronic
  trials lack an LP verdict. Trial universe = every DAQ cue with a resolved position (`classify_cues_with_backup`), not
  the imaging-kept subset.
* **Gates (existing code, called, not copied).** `img` = `precue_engagement_states.engagement_gate` with responded =
  first DAQ lick within `decode.max_rt_s` = 3.5 s (the gate behind `position_coding_directions` lick / miss_working /
  stopped). `quit` = `quit_point.session_quit` on `engagement_decomposition.session_trials` (2.0 s scoring; the S28
  detector; reproduces S28 `session_table.csv`, e.g. PS92_0819 quit 101.47 min). STOPPED = gated out AND no DAQ contact
  within 3.5 s (gated-out trials with a contact are the `success` class, not stopped).
* **LP attempt** = the LP gate's own definition (`lp_licks.session_lp`, cue-time matched): trial_attempts class hit /
  incomplete with the first LP lick (cam1+cam4 combined, else cam4) inside the window. jaw_only / no_try / unknown = no
  attempt.
* **Q3 LP quit rule**: the same detector, responded = DAQ hit OR LP attempt within 2.0 s (trials without an LP verdict
  keep their DAQ score). The gate looks only at the reference spouts (close ipsi, close mid), so only attempts there can
  move it.
* **Q4**: the decoder's own rule (`precue_window_start`, ms units, strobe-bounded, sliding) on the DAQ train for 2.0 and
  1.0 s; then counts LP lick onsets (cam4 every-context `licks_all` + cam1-only events; cam1 is post-cue in practice) in
  the window used. "LP-only" = an LP lick > 100 ms from any DAQ contact.
* Pooled: `boot_ci` levels on session values, `boot_delta` (paired, animal-weighted) for epoch vs pre, seed 20261101.
  PS92 has no pre-stroke stopped trials and PS95 one acute session with none, so the stopped-class acute delta rests on
  2 animals.

## Results — per animal first

### Q1. Gate-STOPPED trials holding an LP attempt (img gate; quit gate nearly identical)

Fraction (n stopped with an attempt / n stopped), all spouts:

| animal | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | — (0 stopped) | 0.044 (26/585) | — | — |
| PS93 | 0.050 (2/40) | 0.025 (13/518) | 0.046 (10/218) | 0.103 (12/116) |
| PS94 | 0.013 (4/301) | 0.078 (69/881) | 0.097 (43/445) | 0.075 (21/280) |
| PS95 | 0.024 (11/453) | — (0 stopped) | 0.026 (31/1177) | 0.031 (21/677) |

Far contra alone: PS92 acute 22/216 (0.10), PS93 acute 10/184 (0.05), PS94 acute 8/306 (0.03), PS95 subacute 16/370
(0.04); the others are 0-0.06. **Far contra is not where the stopped class picks up attempts**; PS94's are mostly
at the other spouts.
Pooled level: pre 0.025, acute 0.049, subacute 0.071, chronic 0.070. boot_delta vs pre: acute +0.018 [-0.024, +0.082]
(2 animals), subacute +0.043 [-0.003, +0.100], chronic +0.050 [+0.007, +0.108] (3 animals).

Seen the other way, **the share of all LP-attempt trials that the gate puts outside working** is small. Pooled: pre 0.005,
acute 0.030, subacute 0.032, chronic 0.008. The largest per-animal cell is 0.044 (PS94 acute / PS95 subacute). Far
contra: 0.000-0.040 per animal.

### Q2. Working-period DAQ misses holding an LP attempt (context: the known deficit)

Far contra, n with an attempt / n misses: acute PS92 500/642 (0.78), PS93 388/514 (0.76), PS94 504/878 (0.57), PS95
80/238 (0.34). Pre (few misses): 14/44, 36/48, 34/74, 28/100. Pooled far contra: pre 0.66, acute 0.70, subacute 0.59,
chronic 0.79. No epoch delta excludes 0, because the fraction is already high pre-stroke on the few pre misses. PS93 far
ipsi / far mid are 0.9 pre (its orofacial deficit). In the DAQ arm `miss_working` is largely "attempted, no contact" in
every epoch. The acute change is in the NUMBER of such trials, not the fraction. This is the known deficit, and the
reason the LP_licks family exists.

### Q3. Quit point under the LP rule

12 of 108 sessions move at all. Every quit moves later (as it must: the rule only adds responses).
* Per-animal mean shift (min) for pre / acute / subacute / chronic: PS92 0 / 0.53 / 0 / 0. PS93 0 / 1.00 / 0.52 / 0.59.
  PS94 0.08 / 0.24 / 2.14 / 0.72. PS95 0.08 / 0 / 0 / 0.
* Largest single shifts: PS93_0925 +7.0 min (chronic), PS94_0825 +5.4 min (subacute), PS94_0921 +5.3 min (chronic),
  PS93_0820 +4.0 min (acute), PS92_0818 +2.6 min (acute).
* **One quit disappears**: PS94_0921 (chronic) becomes censored. No acute or pre quit is lost, and every acute quit
  survives.
* boot_delta vs pre: acute +0.40 min [-0.06, +1.15], subacute +0.62 [-0.06, +1.75], chronic +0.29 [-0.07, +0.91].

### Q4. DAQ-lick-free pre-cue windows that contain an LP lick

Session means for 2 s / 1 s windows, all spouts:

| animal | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | 0.25 / 0.17 | 0.18 / 0.10 | 0.23 / 0.15 | 0.35 / 0.27 |
| PS93 | 0.42 / 0.28 | 0.22 / 0.13 | 0.32 / 0.20 | 0.27 / 0.19 |
| PS94 | 0.14 / 0.09 | 0.13 / 0.08 | 0.19 / 0.11 | 0.20 / 0.11 |
| PS95 | 0.04 / 0.02 | 0.09 / 0.05 | 0.08 / 0.05 | 0.17 / 0.12 |

* Pooled 2 s: pre 0.20, acute 0.17, subacute 0.17, chronic 0.25. Pooled 1 s: 0.135, 0.096, 0.103, 0.177.
* LP-only (> 100 ms from a contact) is not much lower: 2 s 0.16 / 0.15 / 0.16 / 0.23; 1 s 0.09 / 0.08 / 0.09 / 0.15.
* Far contra 2 s: 0.28 / 0.34 / 0.31 / 0.38.
* No boot_delta vs pre excludes 0 (2 s acute -0.055 [-0.152, +0.031]; far contra subacute +0.106 [-0.015, +0.230]).
* The decoder's no-lick arm is the more contaminated one in PS92 / PS93: 2 s pre 0.45 / 0.67, chronic 0.79 / 0.53.
  The engaged arm is 0.04-0.39.

## Verdict

* **S28 quit tests: NOT materially affected.** Every per-animal mean shift is <= 2.1 min. The largest single shift is
  7 min (one chronic session). One quit is lost (PS94 chronic), and none are lost in pre or acute, where the primary
  contrasts (P1-P4, acute / early post minus pre) live. No re-run needed. A noted sensitivity: PS94 chronic loses 1 of 5
  quits.
* **Imaging maps / Haufe on working trials (lick + miss_working): NOT materially affected by the gate.** Only 0.5 % (pre)
  to 3 % (acute / subacute) of LP-attempt trials fall outside working, at most 4.4 % in any animal x epoch. The maps
  lose almost no real attempts to the stopped class.
* **Stopped-class analyses** (enl_states working vs stopped, stopped ENL decode, stopped-trial traces, miss_vs_stopped):
  **borderline.** Post-stroke stopped trials hold an LP attempt in 5-10 % of cases (pooled 0.049-0.071). That exceeds
  the 5 % line in PS94 (all post epochs) and PS93 chronic, and the chronic rise over pre has a CI excluding 0 (3
  animals). But it is not far-contra-driven, not acute-specific, and small in trial count (13-69 per animal x epoch).
  Recommended, cheap, not urgent: a sensitivity re-run of any stopped-class claim with LP-attempt trials dropped from
  `stopped`. No re-run of the main maps.
* **Pre-cue (ENL) decoders, DAQ arm: ABOVE threshold, but not epoch-specific.**
  * The 2 s DAQ-lick-free window holds an LP lick on 17-25 % of trials pooled (far contra 28-38 %, PS93 pre 42 %). It is
    as high or higher pre-stroke as post, and no epoch delta excludes 0.
  * So the DAQ arm's "lick-free" means contact-free, not tongue-free, in every epoch. That is a level confound of the
    headline pre-cue number, not a stroke-specific artifact.
  * The existing LP_licks family (ENL window lick-free by LP, 1.0 s; `lp_licks.LP_ENL_WINDOW_S`, chosen 10-21 from
    `enl_lp_census`) already addresses this. Any claim that the pre-cue code is "not lick-driven" should cite that arm,
    not the DAQ arm. No new LP gate is needed for the decoders, provided the LP_licks ENL arm is the one quoted.
* **Not an LP-gate re-run of everything.** The engagement gate looks only at the reference spouts, where contact and
  attempt nearly coincide, so the far-contra miss-without-contact problem does not leak into the gate. It lives in the
  miss_working class (Q2), which is already handled by the LP_licks family.

## Limits / not computed

* The trial universe is the full behavioral list, not the imaging-kept subset (coverage and lick-free drops). Class
  fractions in the imaging pipeline may differ slightly.
* LP false-positive rate is unknown: cam4 `licks_all` counts every context, and the pre-cue train is cam4 only. Part of
  Q4's pre-stroke level may be tongue / jaw events that are not licks. The LP-only column bounds the contact-adjacent
  part only.
* Stopped-class acute deltas rest on 2 animals (PS92 no pre stopped, PS95 no acute stopped). Interpret them per animal.
* Q4 is computed in ms on the DAQ clock, not imaging frames (the census convention); edge effects are < 1 frame.
* `boot_p` is crude at n = 4 (ranking only).

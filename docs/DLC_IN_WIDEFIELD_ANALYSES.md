# Using DLC / Lightning Pose orofacial data in the widefield GCaMP analyses — design (2026-10-02)

Priya, 2026-10-02: *"let's start thinking about how we will use the DLC/LP data in the widefield gcamp activity
analyses. e.g. analyzing licks based on dlc detection instead of spout detection, determining 'engaged' trials based
on lick or dlc lick / jaw movement without spout contact. we will want to think about analyzing activity based on not
only spout position but, separately, by executed lick angle (i.e. if post-stroke far R licks are deviated left, does
brain activity correspond to pre-stroke activity going to that same tongue trajectory?)"* + *"we can also analyze the
incomplete lick and/or tongue-jaw mismatch trials."*

Status: DESIGN — nothing below is built yet except where marked. Orofacial side: `STATUS_2026-10-02_SESSIONS_REACH_
O2_HANDOFF.md`. Imaging side: `ARCHITECTURE.md`, `BEHAVIOURAL_STATE_CONTROL.md`, `REST_ENGAGEMENT_AUDIT.md`.

---

## 1. What the imaging side uses today (audited 2026-10-02)

* **Licks = DAQ spout contacts only.** `daq_trials.decode` → `lick_s` (DAQ clock). Mapped to imaging frames by
  `framemap_event_maps._corrected_frame_samples` (regime B) or the nearest `pco_exposure` pulse
  (`plot_lick_aligned_averages`); events outside imaging coverage → −1.
* **⚠ Two lick detectors.** The imaging / LocaNMF path calls `plot_lick_aligned_averages._load_daq_events(h5,
  "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)` — lower threshold 1.0 V, refractory 0.10 s — hard-coded in ~20
  modules (canonical: `analysis_kit.lick_samples`), while `configs/defaults.yaml lick_detection` (behaviour side,
  `behavior_events`, `dlc_frames.lick_onsets`) is 0.5 V / min ILI 40 ms. So imaging hits / RTs can already differ from
  `*_trials.csv`. **Decide before anything is joined** (§6, Q1).
* **Engagement.** Per trial `locanmf_position_decoder.is_engaged`: 0 < first-lick RT ≤ `decode.max_rt_s` (3.5 s);
  `nolick_decoder.category_for_rt` (engaged / late_rewarded / undetected). Session quit gate
  `precue_engagement_states.engagement_gate` (reference positions close_L / close_center). Three-state
  success / working / stopped (`enl_states`, `miss_vs_stopped`, `position_coding_directions`). **All contact-based.**
* **Position.** Per cue from the DAQ strobe (`daq_trials.positions_for_cues`); decoders (`locanmf_position_decoder`,
  frozen pre-stroke `locanmf_frozen_decoder`), encoders, coding directions (`position_coding_directions`,
  `cd_trajectories`), maps, RSA. Stroke side `animals.yaml stroke_laterality: L` (all four) → L ipsi, R contra.
* **No movement regressors anywhere** (`grant_encoder`: "NO MOVEMENT REGRESSORS (no DLC yet)").
* **Join key:** animal + date + `trial_id` (= DAQ cue index + 1, `*_trials.csv`); imaging rebuilds trials from the h5
  by cue index and drops cues outside coverage.

## 2. The bridge: one per-session orofacial event table on the DAQ clock

Everything below needs per-trial / per-lick orofacial events that imaging code can join and map to imaging frames
the same way it maps DAQ licks. Proposed module `wfield_local/orofacial_events.py`, one output per session × model:

* **Time base.** Pose frames → DAQ seconds with the behaviour camera's alignment template (inverse of
  `dlc_frames.frame_of`; `camera_sync.cam_seconds_to_daq_seconds`, ~1.2 ms residual) → imaging frames with the
  existing `framemap` / `pco_exposure` mapping. Then a DLC lick onset is an event exactly like a DAQ lick.
* **Per-lick rows:** trial_id, position, model; DAQ times of tongue ONSET (first visible, `on_frame`), PEAK, OFFSET;
  contact (DAQ, ±60 ms) and contact latency; protrusion / reach; rise & retraction speed; peak angle; deviation vs the
  session's and vs pre-stroke successful licks (`lick_reference`); size class (tip-at-lips < 45 px / partial / full);
  quality flags (across-spout jump, model disagreement, inside imaging coverage).
* **Per-trial rows:** n licks (DLC) / n contacts (DAQ) / n incomplete; first tongue-onset latency vs first-contact
  latency; jaw moved (or unknown); mismatch (jaw moved, no lick); trial outcome class (§3B); median direction metrics.
* **Source data:** whole-session predictions — the O2 runner (`o2_inference`, first bundle ready) — not the 60-trial
  subsets. Both models; a consensus / primary-model rule is a decision (§6, Q2).
* **Validation gate before use:** per session, DLC lick ↔ DAQ contact recall (expect ~98 %, as PS93), the tongue-
  onset → contact lag distribution (tens of ms; it shifts every lick-aligned window), and the share of licks inside
  imaging coverage.

## 3. Analyses, in the order I would build them

**A. Lick events from video instead of the spout.**
* Lick-aligned maps / traces re-aligned to tongue ONSET (movement start) rather than contact; contact-aligned kept for
  comparison. Incomplete licks (no contact) become visible events for the first time.
* First-lick latency from tongue onset → an RT that exists on miss trials too. `rt_drift`, `would_be_lick_offsets`
  (no-lick alignment) and `is_engaged` can take it as an alternative input (switch, not a replacement, until validated).

**B. Engagement from movement, not contact.** Per trial, in the response window (cue → trial stop):

| class | tongue | contact | jaw |
|---|---|---|---|
| hit | out | yes | – |
| **incomplete / missed attempt** | out (≥ partial) | no | – |
| **jaw-only (tongue–jaw mismatch)** | no lick | no | moved |
| tip-at-lips only | < 45 px | no | – |
| no movement | – | no | still |
| unknown | jaw unknown, no lick | | |

* "Engaged" = hit + incomplete + (probably) jaw-only. This splits today's contact-based "miss" into *attempted and
  failed* vs *not attempting* — the distinction the post-stroke far-position deficit turns on (acute PS93 / PS92 far_R:
  attempts that all miss vs trials with no lick).
* Re-run the engagement-dependent results on the new definition: frozen position decoder on engaged trials, the
  quit gate (`engagement_gate` could count attempts, not contacts), rest-engagement gating, miss-vs-stopped.

**C. Activity by EXECUTED direction, not only target position** (Priya's main question).
*(First look 2026-10-05, DECISIONS: `lick_templates` template matching — acute PS93 far_R licks executed toward the
mouse's left resembled pre-stroke licks of the same executed angle with one baseline only -- NOT robust (DECISIONS
addendum); `movement_position_angle` —
residual position encoding follows the decoder's pre → acute → chronic course.)*
* *The test:* post-stroke far_R licks run left of the pre-stroke far_R path. Does cortex on those trials look like
  (i) pre-stroke far_R activity (target / intention), or (ii) pre-stroke activity for licks that EXECUTED that
  trajectory (movement)?
* *Why it is testable:* pre-stroke, executed direction varies within a position (trial-to-trial spread of the peak
  angle) and overlaps between neighbouring positions — so target and executed direction are not perfectly collinear
  pre-stroke, and a model can be fit with both.
* *Approach 1 — frozen pre-stroke models, two labels.* Train on pre-stroke trials (a) a target decoder (exists) and (b)
  an executed-direction decoder / encoder (continuous peak angle, or angle bins pooled across positions). Apply both
  frozen to post-stroke trials; ask which label the post-stroke activity tracks — the target it was cued to, or the
  angle it actually executed. Within-position residual analysis: does trial-wise post-stroke angle deviation predict
  the decoder's error?
* *Approach 2 — template matching / RSA.* Pre-stroke activity templates indexed by executed angle; correlate each
  post-stroke trial with the template at its target vs at its executed angle.
* *Approach 3 — encoding model with both regressors* (target position + executed angle + protrusion / speed),
  pre-stroke fit, post-stroke prediction and variance partitioning.
* *Timing matters:* align to tongue onset (movement) and to cue (plan); a planning signal should follow the target,
  an execution signal the movement.
* *Direction measure:* cam4 angles are compressed and carry head-pose / camera differences between sessions; use the
  within-session deviation where possible, and switch to cam1 / 3-D direction when available (labelling under way).

**D. Incomplete-lick and tongue–jaw-mismatch trials** (Priya).
* Activity on *attempted-but-failed* trials vs hits at the same position and epoch (same target, different outcome):
  does the target code survive on failures? Does a pre-stroke-frozen decoder read the intended position?
* *Jaw-only* trials: a jaw movement with no tongue — orofacial motor command without tongue execution. Compare to hit
  trials aligned to jaw-movement onset, and to no-movement trials.
* Their frequency changes with epoch (acute far-position attempts that miss), so they are also a behavioural readout
  and must be controlled for in any epoch contrast.

**E. Movement regressors in the encoding models.** *(Framework BUILT 2026-10-02: `wfield_local/movement_encoding.py` +
`movement_inputs.py`, config `movement_encoding`, synthetic tests; needs real inputs — see DECISIONS.)* Tongue protrusion / velocity, jaw position / velocity, lick
events (onset, contact) as regressors in `grant_encoder` / `locanmf_encoding_model`: separate movement-execution
variance from target coding, and test whether the post-stroke loss of position decoding survives controlling for
the changed movements (the behavioural-state control already shows running is unaffected; this extends it to the
orofacial movement itself).

## 4. Pitfalls to design around

1. The two lick detectors (§1) — every DAQ-vs-DLC comparison must use one definition.
2. Imaging drops cues outside coverage; a DLC table covers the whole video. Join on trial_id and carry a coverage
   flag; never assume the trial lists match.
3. Tongue onset precedes contact by tens of ms — re-aligning shifts every window; re-derive `lick_post_s`-type windows.
4. Quit-period / engagement gating (`REST_ENGAGEMENT_AUDIT.md`) — new engagement classes must be gated the same way.
5. cam4 direction is a projection; between-session angle shifts include head pose — prefer within-session contrasts
   and confirm on cam1 / 3-D.
6. Model choice: DLC and LP agree on licks / timing / velocity but disagree on direction at close_L — any direction
   analysis reports both or a pre-declared rule.
7. Frozen-model logic (pre-stroke fit, post-stroke apply) must be kept for C/D, or epoch differences in the fitted
   model confound the readout.
8. Short licks read image-left from geometry (short vectors) — size-match or restrict direction analyses.

## 5. Proposed build order

1. Whole-session predictions on O2 (bundle ready) for the imaging sessions, both models.
2. `orofacial_events`: per-lick / per-trial tables on the DAQ clock + imaging-frame mapping + validation report.
3. A: tongue-onset-aligned maps next to contact-aligned (one animal, then all) — the quickest visible result.
4. B: outcome classes → re-run the frozen position decoder / quit gate on the movement-based engagement.
5. D: incomplete and jaw-only trial activity.
6. C: target vs executed direction (frozen two-label decoding first), cam4 now, cam1 / 3-D later.
7. E: movement regressors in the encoder.

## 6. Decisions for Priya

1. **Lick detector:** DECIDED for the DLC / orofacial side (2026-10-02): keep `lick_detection` (2.5 / 0.5 V, lockout
   1-20 ms, min ILI 40 ms — stroke_orofacial's rule + our floor). OPEN for the imaging path (hard-coded 1.0 V offset,
   0.10 s refractory): move it to the config (keeping the 0.10 s collapse as a named imaging setting), or keep it and
   state the definition at every DAQ ↔ DLC join.
2. **Primary pose model** for imaging analyses: DLC, LP, or consensus (both must agree, else flag)?
3. **Lick onset event:** tongue first visible (`on_frame`), peak, or both? (Onset for movement alignment, peak for
   direction, contact for reward — my proposal: carry all three.)
4. **Engagement rule:** does jaw-only count as engaged? Does tip-at-lips?
5. **Direction now or later:** start C on cam4 within-session deviations, or wait for cam1 / 3-D?

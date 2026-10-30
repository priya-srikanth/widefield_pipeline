# Running list — 2026-10-29 night (encoding / balance arm)

Kept up to date as helpers report (Priya 10-29: "keep a running list of figures, findings, conclusions, and decisions
you need from me, as well as updating handoff intermittently"). Page: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d
Handoff: STATUS_2026-10-26_ENCODING_HANDOFF.md (00-COMPACT-17 + 10-29 additions). All PROVISIONAL, n = 4.

## 1. Decisions needed from Priya
- [ ] **Code-change wording with bounds**: cued 'unchanged within ~10 deg'; ITI 'not re-mapped beyond ~10-15 deg; small acute rotation-like change (~6-9 deg) beyond 2-month drift + lasting gain loss along the pre axis' -- OK? (Re-run on docked-spout ITI licks if docked becomes primary.)
- [x] (YES, Priya 10-29 late: replace main ITI figures with docked versions where they support the same conclusions; helper running) **ITI primary set = docked-spout licks?** Offsets hold docked; the ITI BALANCE + its recovery tracking are moving-spout results. Make docked the primary ITI set (offsets), and treat moving-spout licks as their own category (possibly spout-driven)? Re-word the ITI-balance marker claims?
- [x] **Representational-change wording**: YES -- offset + gain / precision loss, re-map OPEN (applied page v242, FINDINGS); noise-matched transfer + power check RUNNING.
- [x] **ENL licks next?** YES (Priya 10-29 evening): balance + decoder offset + target code on ENLv licks (timed from spout arrival, window sweep), cue-to-lick level; name them ENLv, not ITI.
- [ ] **Reference for pre-cue levels**: restw vs session mean give opposite signs chronically in orofacial areas (rest falls relative to the session mean). Report the chronic contra ALM / tjM1 pre-cue rise as "relative to rest" only?
- [ ] **Check (mine, low; in restw check)**: precue_restw cache engaged-trial counts differ slightly from the current engagement gate (381 vs 364, 473 vs 472 in 2 sessions) -> audit all sessions; the restw LEVEL itself is unaffected.
- [x] (YES, Priya 10-29) **ENLv reading**: "ENLv = cued-like target-coded balance + ITI-like persistent decoder offset (~half size)" OK as the working description? Bearing on decision ITI primary set: the offset is present on all uncued lick types (docked, moving, ENLv); the balance loss on docked licks may reflect no target rather than no motion.
- [x] (Priya 10-29 late: use the no-lick rest; level family being re-rendered as RESTnl + current engaged gate) **Baseline: keep restw (conservative lower bound) or switch level results to strict lick-free rest?** strict makes the rises 1-4x larger (tjM1 most) but keeps only 17-35% of pre rest in PS92-94; also re-extract pre-cue levels on the current engaged-trial gate (86/107 sessions differ)?
- [ ] **Hypothesis line (3)**: the contralesional-share rise is not orofacial-specific (VISpm / RSPd rise too) -> reword?
- [x] **Pre-cue softmax test**: APPROVED 10-29 -- generate the pre-cue per-trial probabilities (`attempt_coding --aligns
      precue`) if they do not exist; helper told to write only beside the existing lick / cue outputs, not overwrite them.
- [x] (YES 10-29 morning, done) **Wording "consequence of the errors"**: narrow it (FINDINGS / hypothesis) to "the cued imbalance comes after the lick
      and is redundant with the error; causal role for recovery untested"? (asked 10-29, no answer yet)
- [x] (YES, without minimising the ITI offset + contralesional-share rise; done, page v236) **Hypothesis update**: cued balance/offset = landing error seen through a target code; the persistent signals are
      the ITI post-onset contralesional shift and the slow rise of the contralesional share on cued licks. Update the
      WTS / Biology-in-brief hypothesis line accordingly?
- [x] (cued = state readout; ITI offset + ITI SSp-n difference + share rise = mechanistic) **Primary marker framing**: keep the cued balance as the headline recovery marker (valid as a state readout) or
      lead with the ITI balance / decoder offset as the mechanistic marker?
- [x] **Delete 5 superseded cache files** -- DONE 10-29 (Priya approved; derived, read by nothing) (revup/sessions/PS92_0806/0807/0809/0812/0821 .pkl, from an aborted first run,
      onset-only lick rule, read by nothing)? Waiting until the imaging helper (same folder) finishes.
- [x] (standard engagement gate = reference; helper re-running behaviour shares) **Late licks after the spout left: spontaneous or delayed responses?** Behaviour (all canonical trials): at the ITI
      chance rate. Neural (engagement-gated, 48 trials): cued-size, cued-shaped responses. The two used different
      engagement gates; most behaviour 'after-left' trials are in the sated tail. Which engagement definition should be
      the reference for this question? (I can re-run the behaviour shares on the engaged set.)
- [x] (DEFERRED) **Movement-null plan encoding model (Musall-style)**: worth building given pre-cue aim beyond history/face is ~0
      at area level? (nested: spout, history, non-directional movement, directional face, upcoming deviation)

## 2. Figures added tonight (page section, file)
| page | figure | script |
|---|---|---|
| top | Sign-conventions box | — |
| 12.19, 14.14, 14.17, 14.18 | key balance figures now in IMPLIED DIRECTION (display only; `WFIELD_BAL_DISPLAY=hemisphere` restores) | lick_recovery_primary.py etc. |
| 12.17 supp | ACAd midline / vessel / registration control | lick_acad_midline_control.py |
| 12.17 supp | ITI pre-lick time course | lick_iti_prelick.py |
| 14.18 supp | Target vs executed model | lick_balance_target_model.py |
| 14.18 supp | Same target, same landing place (matched error) | lick_balance_matched_error.py |
| 14.21 supp | ENL lick-angle decode | lick_enl_angle_decode.py |
| 14.21 supp | ENL follow-ups (history test, offset explore) | lick_enl_offset_explore.py |
| 14.21 area | Revving up: survival, delay decomposition, phase shares, FR vigour, recovery | revup_behaviour.py |
| 14.21 supp | Pre-cue target vs landing (frozen decoder softmax, class-matched, confusion, decodability, tracking) | lick_enl_target_test.py |
| 14.18 supp | C/I by spout x epoch, target vs landed frame | lick_ci_spout_epoch.py |
| 14.21 area | Revving up follow-up: jaw (no ramp), late licks vs corrected chance | revup_followup.py |
| 14.18 supp | Same code or re-mapped (transfer asymmetry, axes vs June drift, areas) | lick_code_subspace.py |
| 14.18 supp | Per-area tuning + matched pixel maps | lick_area_tuning.py, lick_matched_pixel_maps.py |
| 14.18 supp | Correction licks after a miss | lick_correction.py |
| 14.18 supp | Coordination between hemispheres (noise corr, lead, lick vs rest, recovery) | lick_interhemispheric.py |
| 14.18 supp | ITI results on docked-spout licks (vs moving, size-matched) | iti_dock_audit.py, lick_iti_dock_compare.py |
| 14.18 supp | Noise-matched null + rotation power for the code-change test | lick_code_noisematch.py |
| 14.21 area | ENL lick audit (counts, task logic, aim, contact) | enl_lick_audit.py |
| 14.21 area | Revving up NEURAL: quartile traces, slow-only contrasts, late vs normal vs spontaneous, pre-cue vs latency | revup_neural.py |

## 3. Findings (one line each; details in FINDINGS 0b)
- ACAd left/right link: undetermined, not ACAd-specific (lives on the sinus strip; shares variance with MOs / SSp-n).
- ENL predicts lick angle weakly, mostly via the spout; within-spout part not robust beyond trial history + face.
- Cued post-lick activity carries a large TARGET component pre-stroke (w 0.76-0.92); target-aware reference removes the
  acute cued balance deficit and the cued decoder offset; target-aware cued balance no longer predicts recovery.
- At matched landed angle the target matters LESS after the stroke (cortex tracks the actual lick more).
- Same target + same landing place, no extrapolation: cued acute balance +0.02, decoder -0.04 mm (= pre-stroke);
  ITI decoder +0.29 / +0.19 / +0.22 mm (4/4) -> genuine ITI state shift.
- Cued within-spout balance rises ABOVE pre over weeks (contralesional hemisphere gains share; not beyond log day).
- ITI contralesional reading / imbalance appears at lick onset (no ITI spout contact), not preparatory; GCaMP6s caveat.
- ENL acute +1.55 deg offset: mostly the aim deficit through a target-reading decoder, BUT decoder compression means
  'pre-cue target representation unchanged' is NOT established (far-contra moves ipsilesional in 3/4).

- REVVING UP (behaviour): acute far-contra delay = mostly a quiet WAIT then a normal lick (or none), not aborted
  attempts; NO jaw ramp (jaw quiet, then abrupt normal opening); late licks after the spout left ABOVE corrected chance
  (delayed / spout-triggered; earlier 'chance rate' claim retracted); latency tracks aim (r -0.72 to -0.92).

- REVVING UP (neural): slow first licks = delayed start, not a slower ramp (same rise, same level at the lick);
  acute far-contra slow trials overshoot (3/4); engaged late licks look like delayed cued licks.

- PRE-CUE TARGET TEST: before the cue the target code is DEGRADED but NOT re-aimed at the landing spout (deviated
  trials look like same-session hits pre-cue; landing signature only after the cue); acute row flattens, close ipsi as
  much as far contra.
  PRE-STROKE same pattern (34 deviated trials, thin; standard output, animal-weighted): pre-cue ns, lick target -0.28
  [-0.52, 0.00] / landing +0.16 [0.00, 0.32] (borderline) -> post-stroke looks like the normal error signature, more frequent.

- ENL LICKS: plentiful (no cell < 20 except PS95 acute 1 session); acute far-contra ENL aim error = cued (-25.9 vs -25.4 deg) -> far-contra aim deficit does not need the cue. Non-contact ENL licks can fall inside the pre-cue level window.

## 4. Conclusions (current reading)
- ITI: the lasting PATTERN offset holds on docked-spout licks; the ITI hemispheric BALANCE + recovery tracking come from moving-spout licks.
- Lick-direction code CHANGES: cued = transient contralesional offset; ITI = persistent offset + lower gain / precision beyond June drift, orofacial (ipsilesional chronically). Re-map: open (low power).
- Pre-cue: target representation weaker but unbiased; the error emerges after the cue (execution / post-cue).
- Cued licks: the balance shift and decoder offset largely re-describe the landing error (target code); the cued
  balance is a valid state readout but not an independent mechanism.
- Persistent, behaviour-independent signals: (a) ITI post-onset contralesional shift (tracks recovery); (b) slow rise
  of the contralesional hemisphere's share on cued licks (reorganisation?).
- Grant: chronic contralesional silencing tests whether the extra share supports recovered behaviour; SSp-m/n both
  hemispheres first; ACAd dropped.

## 5. Running (will be added above when they report)

- Updates sent: ENLv primary = onset >= 1 s after arrival (+ >= 0 s and first-ENLv-lick versions, arrival-corrected); chronic dip = NON-restw traces.

- Updates sent: ENLv arrival correction by REGRESSION (not lick-free subtraction); cue excluded from ENLv windows (contact vs non-contact; licks < 2 s before cue reported); stopped helper adds jaw-no-lick vs stopped-no-jaw.


- ITI pre-lick re-run with windows ending >= 0.1 s (and 0.15 s) before lick onset, + jaw-onset variant (Priya) -> iti_licks/ (old fig kept as lick_iti_prelick_v1.png)

## 7. ENLv + cue-to-lick + trace results (figure links added as they report)
- ENLv CURVES + GAIN (page v258, Fig 14.21 S10, lick_enlv_curves*.png): ENLv curves sit above pre like ITI; cued change mostly flatter read-out (gain -0.11), ITI gain + offset, ENLv mostly offset.
- FIGURE CLEAN-UP (page v258): legends / titles off data in ~60 figures (plotting only); ENL audit heatmaps removed; simplified: area tuning decomp, interhemi recovery, ENLv sweep; area_tuning_raw cut from page. Open: lick_matched_pixel_areas, lick_correction_prediction, lick_enlv_target panel 2, lick_enl_offset_explore panels 3-6, 8 (cut / simplify?).
- ENLv NEURAL (page v252, Fig 14.21 S10 #s14-21-10; M:/.../iti_licks/enlv_licks/figures/lick_enlv_compare.png, _target, _perspout, _sweep, _first_later, _arrival): offset persists at matched error (~half ITI, PS92 ~0); balance target-coded (cued pattern, chronic within-spout +0.14 4/4).
- ARRIVAL -> CUE (page v251, Fig 14.21 S13 #s14-21-13; M:/.../iti_licks/arrival_to_cue/figures/): far-contra ENLv contact drops acutely 4/4 (no timer restart); last-ENLv-lick response larger acutely (contra tjM1 / SSp-n 4/4); chronic contra tjM1 cue response larger 4/4. Possible follow-ups: anchor on last CONTACT lick; per-animal panels.
- CUE TO LICK (page v250, Fig 14.21 S12 #s14-21-12; M:/.../iti_licks/cue_to_lick/figures/): accumulate-to-bound rejected (fixed level / delayed start); restw rise = sustained pre-cue raise; chronic raw dip in orofacial = rest level drop vs session mean; non-orofacial dip real.
- STOPPED FOLLOW-UP (page v250): restw-unit rise mostly a restw fall; raw only PS93; mostly bilateral; same in hit / miss_working.
- STOPPED TRIALS (page v249, Fig 14.21 S11 #s14-21-11; figures M:/.../iti_licks/stopped_traces/figures/): no cue response, flat raised level; 3 animals per epoch; jaw-onset transient global; contra tjM1 / SSp-m pre-cue rise over recovery within animals (contra-specific? follow-up running).

## 6. Log
- 10-29 late: docked ITI swapped into 39 main figure images (page v270); DIFFERENT ones kept all-ITI with notes. RESTnl level family swapped (v269). Pre-lick re-run (v268). Simplified pixel / tuning / interhemi / correction (v267).
- 10-29 late: Priya: docked = primary ITI (helper: docked versions of main ITI figures + verdicts); no-lick rest (RESTnl) for level results (helper re-rendering 14.21 family); ACAd 12.17 S2 off page; pixel-map epochs / tuning dots / interhemi simple / correction relabel helper; combined like-for-like figure on page (v264). Code-change wording: discuss evidence.
- 10-29 late: restw check reported -> Fig 14.21 S14 (v262); Priya: too many figures -> 39 secondary + 3 crowded-bar figures cut from page (docs/CUT_FIGURES.md); crowded-bar scan of the whole page running.
- 10-29 evening: ENLv curves + gain, figure clean-up -> page v258; pre-lick window helper started.
- 10-29 evening: Priya approved the ENLv working description; page v255: click-to-zoom now also on supplement images (was figure images only); ENLv curves helper started.
- 10-29 evening: page v253: supplements to Figs 12.17 / 14.18 / 14.21 numbered (S1..) with anchors #s<fig>-<n>; stale 14.20 refs fixed.
- 10-29 evening: ENLv neural reported -> page v252, FINDINGS. Only the restw check running.
- 10-29 evening: Priya: run the restw check (rest excludes only DAQ contact licks, 1 s post buffer) -> helper started.
- 10-29 evening: arrival -> cue traces reported -> page v251, FINDINGS. Only ENLv still running.
- 10-29 evening: cue-to-lick + stopped follow-up reported -> page v250, FINDINGS.
- 10-29 evening: stopped-trial traces reported -> page v249, FINDINGS; follow-up (contra-ipsi) sent; arrival->cue helper started.
- 10-29 evening: Priya: run ENLv (naming) balance / offset / target code + cue-to-lick level, AUC threshold, pre-lick window; chronic dip = decreased chronic pre-cue dF/F in cue-aligned area traces. 2 helpers started (4 workers each).
- 10-29 evening: ENL lick audit reported -> page v248 (14.21), FINDINGS. Priya reviewing pre-cue maps / traces (14.20); items 1-5 + overnight controls not yet reviewed.
- 10-29: noise-matched null + power reported -> page v247, FINDINGS. All helpers done; nothing running.
- 10-29: docked-spout ITI rerun reported -> page v246 (+ hypothesis line qualified), FINDINGS. Helper added ITI_SUBSET_FILTERS hooks (iti_licks.py, iti_lick_order.py, lick_matched_readout.py, lick_balance_subsets.py); outputs in itifilter_*/.
- 10-29: item 3 (interhemispheric) reported -> page v245, FINDINGS (global coupling rise, no lead). Deleted the helper's superseded test outputs interhemi_rest/PS92_0806.npz (2-s attempt, 0 windows) + interhemi_sessions_test.csv.gz.
- 10-29: item 4 (correction licks) reported -> page v243, FINDINGS (no cortical error-feedback signal found).
- 10-29: Priya: representational-change wording softened (re-map open); noise-matched transfer + power check started.
- 10-29: items 1 + 5 reported (area tuning, matched pixel maps) -> page v241, FINDINGS. Priya flagged 'no re-map' as too strong -> wording pending discussion.
- 10-29: item 2 (code subspace) reported -> page v240, FINDINGS. New frozen kind lick_code in the LOCAL store (not published to the shared store).
- 10-29: 4 more helpers started for Priya's items 1-5 (max 4 workers each; shared box).
- 10-29: Priya asked that ITI offset results be based on docked-spout licks -> helper started.
- 10-29 morning: rev-up follow-up reported (jaw: no ramp; late licks above corrected chance; neural set = 'attempted' gate) -> page v239, FINDINGS. All helpers done.
- 10-29 morning: pre-stroke class-matched now standard (--classmatched-only; new figure; offtarget labels fixed) -> page v238; deleted my superseded scratch enl_target/classmatched_pre_sessions.csv.
- 10-29 morning: C/I by spout x epoch reported -> page v237 (14.18 supp), FINDINGS.
- 10-29 morning: Priya decided items 1-5 (above); hypothesis line + FINDINGS updated; 3 helpers started.
- 10-29 morning: deleted revup/sessions/PS92_0806/0807/0809/0812/0821.pkl (approved; superseded derived caches).
- 10-29 morning: class-matched pre-cue test re-run incl. PRE-STROKE (Priya) -> same pattern; FINDINGS, page v235.
- 10-29 ~00:45: rev-up behaviour reported -> page, FINDINGS, list.
- 10-29 ~01:45: handoff updated; nothing running. Side effect noted: attempt_coding_pp/run.json was overwritten by
  the precue run and rewritten by the helper to list lick + cue + precue (original backed up in the session scratchpad).
  Minor: lick_enl_target_offtarget.png panels 2-3 have overlapping x labels (cosmetic).
- 10-29 ~01:40: pre-cue target test reported (precue attempt_coding outputs generated, approved) -> page, FINDINGS, list.
- 10-29 ~01:15: rev-up neural reported -> page, FINDINGS, list.
- 10-29 ~01:00: page v232 published. The artifact hit its 512-file cap: removed 75 published images the page no longer
  references (old iti_* duplicates, superseded variants; local copies kept in _goal_page/), added the rev-up figures.
- 10-29 night: list created (commit after d13f51f5).

# Running list — 2026-10-29 night (encoding / balance arm)

Kept up to date as helpers report (Priya 10-29: "keep a running list of figures, findings, conclusions, and decisions
you need from me, as well as updating handoff intermittently"). Page: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d
Handoff: STATUS_2026-10-26_ENCODING_HANDOFF.md (00-COMPACT-17 + 10-29 additions). All PROVISIONAL, n = 4.

## 1. Decisions needed from Priya
- [x] **Pre-cue softmax test**: APPROVED 10-29 -- generate the pre-cue per-trial probabilities (`attempt_coding --aligns
      precue`) if they do not exist; helper told to write only beside the existing lick / cue outputs, not overwrite them.
- [ ] **Wording "consequence of the errors"**: narrow it (FINDINGS / hypothesis) to "the cued imbalance comes after the lick
      and is redundant with the error; causal role for recovery untested"? (asked 10-29, no answer yet)
- [ ] **Hypothesis update**: cued balance/offset = landing error seen through a target code; the persistent signals are
      the ITI post-onset contralesional shift and the slow rise of the contralesional share on cued licks. Update the
      WTS / Biology-in-brief hypothesis line accordingly?
- [ ] **Primary marker framing**: keep the cued balance as the headline recovery marker (valid as a state readout) or
      lead with the ITI balance / decoder offset as the mechanistic marker?
- [ ] **Delete 5 superseded cache files** (revup/sessions/PS92_0806/0807/0809/0812/0821 .pkl, from an aborted first run,
      onset-only lick rule, read by nothing)? Waiting until the imaging helper (same folder) finishes.
- [ ] **Late licks after the spout left: spontaneous or delayed responses?** Behaviour (all canonical trials): at the ITI
      chance rate. Neural (engagement-gated, 48 trials): cued-size, cued-shaped responses. The two used different
      engagement gates; most behaviour 'after-left' trials are in the sated tail. Which engagement definition should be
      the reference for this question? (I can re-run the behaviour shares on the engaged set.)
- [ ] **Movement-null plan encoding model (Musall-style)**: worth building given pre-cue aim beyond history/face is ~0
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
  attempts; late licks after the spout left are at the ITI chance rate; latency tracks aim (r -0.72 to -0.92).

- REVVING UP (neural): slow first licks = delayed start, not a slower ramp (same rise, same level at the lick);
  acute far-contra slow trials overshoot (3/4); engaged late licks look like delayed cued licks.

- PRE-CUE TARGET TEST: before the cue the target code is DEGRADED but NOT re-aimed at the landing spout (deviated
  trials look like same-session hits pre-cue; landing signature only after the cue); acute row flattens, close ipsi as
  much as far contra.

## 4. Conclusions (current reading)
- Pre-cue: target representation weaker but unbiased; the error emerges after the cue (execution / post-cue).
- Cued licks: the balance shift and decoder offset largely re-describe the landing error (target code); the cued
  balance is a valid state readout but not an independent mechanism.
- Persistent, behaviour-independent signals: (a) ITI post-onset contralesional shift (tracks recovery); (b) slow rise
  of the contralesional hemisphere's share on cued licks (reorganisation?).
- Grant: chronic contralesional silencing tests whether the extra share supports recovered behaviour; SSp-m/n both
  hemispheres first; ACAd dropped.

## 5. Running (will be added above when they report)
(nothing running)

## 6. Log
- 10-29 ~00:45: rev-up behaviour reported -> page, FINDINGS, list.
- 10-29 ~01:45: handoff updated; nothing running. Side effect noted: attempt_coding_pp/run.json was overwritten by
  the precue run and rewritten by the helper to list lick + cue + precue (original backed up in the session scratchpad).
  Minor: lick_enl_target_offtarget.png panels 2-3 have overlapping x labels (cosmetic).
- 10-29 ~01:40: pre-cue target test reported (precue attempt_coding outputs generated, approved) -> page, FINDINGS, list.
- 10-29 ~01:15: rev-up neural reported -> page, FINDINGS, list.
- 10-29 ~01:00: page v232 published. The artifact hit its 512-file cap: removed 75 published images the page no longer
  references (old iti_* duplicates, superseded variants; local copies kept in _goal_page/), added the rev-up figures.
- 10-29 night: list created (commit after d13f51f5).

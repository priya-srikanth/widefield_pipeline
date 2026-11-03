# Running list, 2026-11-03 (Priya away), PI-deck window (sabatinilab-a8)

Read top to bottom: questions for Priya first, then findings, then decisions taken, then what is running. The detailed
log for the earlier part of the day is `RUNNING_LIST_2026-11-02_OVERNIGHT.md`.

## Questions for Priya (newest first)

_(none open yet. Items appear here as the decoder helper reports.)_

## Findings (newest first)

- **Mixed model vs bootstrap stars.** The LMM rule is a separate between-animal spread per epoch, df = 3. Under it,
  about half of the bootstrap stars on the wf_attempt figures drop. The LMM adds a star in only 2 cells, both on
  hemispheric balance. Four main-slide claims are not significant under the LMM:
  - CSM share: p .13 / .13 / .17;
  - chronic normality: p .14;
  - acute cortex - behaviour gap: p .21;
  - acute orofacial residuals: p .065-.13.

  The main captions stay as they are; each hidden slide states its result on the slide.
- **Refit decoder (slide 10), chronic.** The bootstrap stars chronic post-cue (+.075 [.015, .150] **) and chronic
  pre-cue (+.039 *). The LMM stars only acute post-cue.
- **Spout-decoder bal_em (production, 99be6122).**
  - Acute far contra (cue): -.536 -> -.644.
  - Close spouts: +.05 to +.06 (close ipsi CI now includes 0).
  - Overall accuracy moves by <= .03 everywhere.
- **bal_em extension (054c78df, a1455e2b).** Three numbers move by more than .03:
  - coding-direction confusions (grant 5b): cue .270 -> .233, lick .253 -> .213;
  - June -> August drift, acute-from-August frozen: .666 -> .713;
  - pre-cue engaged no-lick: .261 -> .241.

## Decisions taken today (Priya)

- **Pre-stroke baseline.** Frozen decoders and the whole imaging render use AUGUST-ONLY pre sessions.
- **Working trials.** Working = trial_classes.in_working on both sides: licks + incomplete licks + jaw movement, cut at
  the last camera lick. At close mid it is licks only.
- **State discriminators.** Engaged vs disengaged goes in the spout bal_em frame. The lick / jaw-only / no-movement
  3-class version is built as a reported variant for Priya.
- **Axes / CDs / patterns.** Same per-session bal_em centring before projecting.
- **Re-render.** One combined pass once the checks pass; A3 no longer gates it.
- **Stars.** Main slides keep bootstrap stars. LMM versions go on hidden slides right after each main slide.
- **Curve figures.** LMM on pre-specified per-session summaries (DECISIONS 82665657).

## Running

- Decoder helper (a44ea79f): steps A-C (August-only + in_working refit, state frame, projections), checks, then the
  render. It reports twice: after the checks, and after the render.
- Deck (182 slides): every starred figure has its hidden LMM slide. Waiting on the re-render for slides 10-12 and 19 and
  their LMM versions (e3 redraws from the new session tables).
- Pages: imaging v51 and FaceRhythm v36 were published by e0. The SPOUT_BALEM page edits are held for the re-render.
  Publishing may stop working again (account switched back to hhmi).

## Log (newest last)

- 11-03 late: list started.
- 11-03 late: Priya (via 34) said the jaw-steering hidden LMM slide is too wide for one graph. I asked e3 to re-render
  k_jaw_epochs_lmm (6 x 4.5 in), decoding_refit_sessions_lmm (8 x 4.5) and frozen_refit_pooled_lmm (9 x 4.5). I will
  place each at its natural width, centred, and check the other one-panel hidden slides.

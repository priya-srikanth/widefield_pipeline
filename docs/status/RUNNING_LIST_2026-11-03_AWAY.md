# Running list, 2026-11-03 (Priya away), PI-deck window (sabatinilab-a8)

Read top to bottom: questions for Priya first, then findings, then decisions taken, then what is running. The detailed
log for the earlier part of the day is `RUNNING_LIST_2026-11-02_OVERNIGHT.md`.

## Questions for Priya (newest first)

The combined render is ON HOLD until Q1 and Q2 are answered. Q3 is a default you can overrule later. Details:
`M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/spout_balem_production/summarise_{A,B,C}_log.txt`.

1. **State discriminators in the spout frame (B): switch on?** On the pre checks the frame changes nothing:
   - held-out pre engaged reads engaged: cue .925 -> .919, precue .742 -> .730;
   - precue-states leave-one-session-out AUC: .584 -> .585.

   The AUC misses the .656 target in both frames. The drop comes from the August-only baseline, not the frame
   (PS93 is .326 in both). After the stroke, the frame moves the section G (G4) cue readout a lot:
   - acute no-lick engaged-like: .195 -> .502;
   - fits_engaged: .170 -> .498.

   Precue moves by .09 or less. Built, default OFF.
2. **Projections in the spout frame (C): switch on for CDs and patterns only, or keep everything off?** The CD and
   pattern readouts stay within noise on held-out pre. The precue engagement axis FAILS the check: PS92 shifts by
   1.9x its between-session SD and PS93 by 1.6x. With C on, the headline numbers would move:
   - far-contra CD acute: cue -.099 -> -.332; precue .218 -> -.035;
   - precue engagement axis acute: .755 -> 1.092.

   The helper centred the projections on the engaged rows of the lick-gate spout model, so that no session drops.
   That is not the working-row centre the decoders use. Built, default OFF.
3. **PS93_0813 (default taken; reversible).** It drops only where a camera is needed, so cue and precue training
   sets are 8/7/8/8. The lick-window models keep it at 8/8/8/8, because the lick gate needs no camera.

FYI, the 3-class camera variant (lick / jaw-only / none) is thin in PS92 and PS93. Pre jaw-only trials are 20 / 37 /
165 / 306 per animal. Held-out pre recall at cue: lick .67-.93; jaw-only .60-.63 (PS94 and PS95 only); none .20-.77.

## Findings (newest first)

- **in_working + August-only pre is now production (26fd5b44; 29 new frozen models; all older models kept).** These
  are definition effects, and some are large. Frozen accuracy at cue on working trials, animal means, old -> new:
  - overall: pre .882 -> .841, acute .539 -> .516, subacute .716 -> .667, chronic .786 -> .759;
  - **close ipsi acute: .789 -> .631.** PS95 goes .975 -> .642, because its jaw-only trials now count as working;
  - close mid acute: .635 -> .762 (working = licks only there);
  - far contra acute: .219 -> .191;
  - precue: far contra acute .216 -> .179, close mid acute .365 -> .465;
  - lick window: within .04.

  Working trials per session (cue; old -> new): pre 478 -> 512 (no-lick 0 -> 26), acute 460 -> 456 (166 -> 163),
  subacute 450 -> 476 (43 -> 69), chronic 473 -> 478 (8 -> 15). 5b's s1 refit trial sets are reproduced exactly
  (214 of 214 records). Not switched (they keep their own trial selections): map families 14/15, nolick_decoder,
  coding-direction classes, precue_engagement_states.

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

- Decoder helper (a44ea79f): checks done and the helper has exited. A is live; B and C are built but OFF. The combined
  render waits on Q1 and Q2. Once those are answered, the render covers SPOUT_BALEM_PAGE_EDITS sections 2, 5 and 7.
  After it, these slides change: 10, 11, 12, 13 (position side), 19 and 89, plus every caption that mentions working
  trials (build_inputs/spout_balem_swaps.json).
- Deck (183 slides): every starred figure has its hidden LMM slide. Waiting on the re-render for slides 10-12 and 19 and
  their LMM versions (e3 redraws from the new session tables).
- Pages: imaging v51 and FaceRhythm v36 were published by e0. The SPOUT_BALEM page edits are held for the re-render.
  Publishing may stop working again (account switched back to hhmi).

## Log (newest last)

- 11-03 late: list started.
- 11-03 late: Priya (via 34) said the jaw-steering hidden LMM slide is too wide for one graph. I asked e3 to re-render
  k_jaw_epochs_lmm (6 x 4.5 in), decoding_refit_sessions_lmm (8 x 4.5) and frozen_refit_pooled_lmm (9 x 4.5). I will
  place each at its natural width, centred, and check the other one-panel hidden slides.
- 11-03 evening: e3's three narrower figures are written. The deck save is BLOCKED: the deck is open in PowerPoint on
  this box (window "PI_overview_2026-11-02.pptx - PowerPoint", started 18:27; not opened by this window, so I am not
  touching it). place_natural_width_1103.py is ready and will run once the deck is closed. It places the figures at
  6.0 / 8.0 / 9.0 in, centred, with the caption under each.
- 11-03 evening: e3 re-rendered the 3-D phase main slide (fig:p_phase) with cluster-permutation bars (Priya via 34).
  swap_p_phase_perm_1103.py is queued behind the same deck lock. Caveat for Priya: the midline row is marked almost
  everywhere, including ~2 deg chronic shifts. That is a session-level test within these 4 animals; the hidden LMM
  slide is the across-animal test.
- 11-03 evening: 34 flipped the far-spout mean-path figure (fig:rec_wlc_paths) so contralesional is LEFT (Priya; data
  unchanged). swap_rec_wlc_paths_flip_1103.py is queued behind the same deck lock.
- 11-03 evening: Priya (via 34) asked for a hidden backup "Backup: lick order, ITI bouts" right after "Lick order:
  tongue corrects, read-out stays". It shows the ITI row of Fig 13.2 (landing, decoded, lfl by lick order; BH-FDR
  bars), cropped like the cued row. The main slide's bullet / notes are repointed from a backup title that never
  existed. backup_lick_order_iti_1103.py is queued behind the deck lock.
- 11-03 evening: Priya saved her hidden / unhidden edits (18:50) and closed the deck; backup at
  backup_priya_saved_1850_1103.pptx. I applied the four queued changes on HER saved file: natural-width slides 12 / 18
  / 43, p_phase stats on slide 52, contra-left paths on slide 59, and the ITI backup at slide 119. Hidden-state diff
  keyed by title + shape names: the only change is the new hidden slide 119 (77 -> 78 hidden; 183 slides). PNGs checked.
  Waiting: 34's LMM 0-1 s swap for slides 95-98.
- 11-03 evening: slides 95-98 (hidden LMM for the lick-ramp main slide 94) now test slide 94's own quantity: per
  session, spouts pooled, AUC 0-1.0 s / pre; PEAK is secondary (Priya). Finding:
  - cued AUC: 8 / 21 cells *, subacute + chronic only, no acute cell (p .08-.40);
  - cued peak: 11 / 21, no acute;
  - ITI AUC: 0 / 21 (smallest p .07);
  - ITI peak: 8 / 21 incl. acute in 4 ROIs.

  The acute rise is carried by PS94 and PS95 (PS95: one acute session); PS92 is ~0 acutely. Hidden states unchanged
  (78 / 183). swap_ramp_lmm_w01_peak_1103.py.
- 11-03 late: the decoder helper reported. A (in_working + August-only) passed and is on main as 26fd5b44 (731 tests
  passed). B misses the literal AUC target, and C's precue engagement axis fails its check. Per the rule, no render
  was started. Questions 1-3 are at the top of this list.
- 11-03 late: Priya asked for checks on whether jaw-only trials belong in "working". A helper is running three
  rescorings of existing bundles (no render, production unchanged), with outputs in
  `labcams/decoder_mix_checks/jaw_only_checks_1103/` and the write-up in `docs/status/JAW_ONLY_CHECKS_2026-11-03.md`:
  - **test 1:** a decoder trained on pre contact + incomplete trials, scored per class (contact / incomplete /
    jaw-only / none). Jaw-only vs none is the key comparison;
  - **test 2:** train with vs without jaw-only, scored on the same held-out lick trials. Controls: count-matched,
    per-spout matched, A's centre;
  - **split of the old -> new change:** training set, scored set and August-only, switched one at a time (e.g. PS95
    close ipsi acute .975 -> .642).

  Cue window primary, precue secondary.

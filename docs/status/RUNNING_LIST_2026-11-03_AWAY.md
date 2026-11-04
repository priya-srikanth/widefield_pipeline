# Running list, 2026-11-03 (Priya away) -> overnight into 11-04, PI-deck window (sabatinilab-a8)

**Overnight plan (Priya asleep from ~11-03 late; my decisions, reversible):**
1. Implement jaw-only out of working (helper running).
2. ~~Then run ONE combined render with B and C OFF~~ HELD (question 0): 34's rebuild will replace those families
   on new data within 1-2 days.
3. Deck swaps and the hidden LMM redraws (e3) wait for the same answer.
4. Page edits (e0) are held too.
5. Other windows (e3, 34, e0) were told that working now excludes jaw-only.

Read top to bottom: questions for Priya first, then findings, then decisions taken, then what is running. The detailed
log for the earlier part of the day is `RUNNING_LIST_2026-11-02_OVERNIGHT.md`.

## Questions for Priya (newest first)

0. **NEW (night): interim deck swap now, or wait for 34's cohort-1 rebuild?** I HELD tonight's main-branch combined
   render; that was my call, and it is reversible. 34's rebuild uses new data: the swap mask and LocaNMF
   projection, plus today's rules (jaw-only out, August-only, bal_em). It will replace the same epoch families,
   decoders and maps. Its render may start late tonight; it takes about a day, then verifiers. Deck swaps are 1-2
   days out and need you for any conclusion change.
   - A main render now would run on the OLD data. Slides 10-13 / 19 / 89 would then be swapped twice, and the render
     would compete with the rebuild for CPU.
   - The rule effects on main are already measured (jaw-only checks + the helper's verification, findings above).
   - If you want the deck to show today's rules now, I'll render on main (workers capped) and swap. Interim status
     would go in the notes, not on the slide.

5. **NEW (11-04): split-basis details** (34's inventory: wf_attempt docs/status/rebuild_20261103/
   SPLIT_BASIS_INVENTORY_20261104.md). My suggestions are in brackets.
   - a. Encoder rows: v8 masks (stat mask + rim, like the decoders and the s1 table) or stat-mask-only? [v8: one
     definition; SSp / cortex rows move slightly.]
   - b. Groups that used whole MOs / MOp: orofacial group in specificity_control and quiet events, hemispheric MO
     group. [ALM + tjM1 + MOs_rest + MOp_rest, pixel-weighted, so the group covers the same cortex as before.]
   - c. precue_attribution's MOs-vs-MOp contrast. [Keep it as MOs vs MOp via the union, and add ALM vs tjM1 beside
     it.]
   - d. residual_roi Bonferroni family grows with the split areas. [Accept the new family.]
   - e. Keep the "cortex" summary rows. [Yes.]
   - f. LocaNMF-component analyses: the labels come from Allen seeds and cannot simply switch. [Leave them: split
     applies to Allen-ROI analyses.]
   - ANSWERED with 34 (Priya, 11-04, both branches):
     - f: LEAVE the LocaNMF-label analyses as they are.
     - b / c: every orofacial group = the quiet-events definition: ALM, tjM1, SSp-m, SSp-n (limb = SSp-ul, SSp-ll).
       MOs_rest / MOp_rest are NOT in it. The MO-group / MOs-vs-MOp analyses default to ALM vs tjM1.
     - d: 34 proposes that pixel-based significance stays primary and the area table becomes descriptive, so there is
       no Bonferroni-family question.
     - a (rim extent) and e (which "cortex" rows): still being clarified with 34.

The combined render is ON HOLD until Q0-Q2 are answered. Q3 is a default you can overrule later. Details:
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
4. **NEW (11-04): should section D's and G's SCORED sets also become working trials?** On main, only their model and
   frame switched. The scored sets kept their designs:
   - section D (pooled_frozen_loso) scores all DAQ-engaged rows;
   - section G's pre panel / crossed_confusion uses engaged rows.

   Those rows include jaw-only and after-last-lick trials. 34's rebuild converts both to working rows on both sides,
   so main and rebuild differ here. I think converting is the consistent choice (your "working on both sides"). But
   section G was designed to keep misses for the failed-attempt readout, so it is your call. Grant 5b is fine on
   main: under the camera selection it recomputes from working trials rather than reading coding_direction.json.
   UPDATE: 34's rebuild now follows main, so the branches don't split. If you say "convert", both branches change
   together. The rebuild also copied main's 5b guard.
3. **PS93_0813 (default taken; reversible).** It drops only where a camera is needed, so cue and precue training
   sets are 8/7/8/8. The lick-window models keep it at 8/8/8/8, because the lick gate needs no camera.

FYI, the 3-class camera variant (lick / jaw-only / none) is thin in PS92 and PS93. Pre jaw-only trials are 20 / 37 /
165 / 306 per animal. Held-out pre recall at cue: lick .67-.93; jaw-only .60-.63 (PS94 and PS95 only); none .20-.77.

## Findings (newest first)

- **11-04: working excludes jaw-only, now on main (48f2cc6d).** The production path reproduces the jaw-only checks
  exactly in every session (largest difference 1e-16). 20 new frozen models are published and SHA-verified; every
  older model is kept. The lick window is unchanged. The jaw-inclusive (`in_working_jaw`) and quit-tail settings stay
  as robustness options and reproduce their earlier rows.
  - Frozen accuracy, cue, animal means: close ipsi pre .923 / acute **.727** / subacute .824 / chronic .924.
    Per animal acute .829 / .629 / .619 / .832; the afternoon rule gave .631 and the old one .789.
  - Far contra acute .187 (.160 / .121 / .296 / .171).
  - Close mid acute .738.
  - Overall acute .592, subacute .714, chronic .775.
  - Pre panel .856.
  - Precue (pre / acute / subacute / chronic): close ipsi .737 / .567 / .524 / .709; far contra .522 / .186 / .300 /
    .349.

  Acute sessions 5 / 4 / 6 / 1, so PS95 acute is ONE session. Nothing is rendered; the deck and pages are unchanged
  (question 0).

- **Section-1 within-session refit, with jaw-only taken out (e0; LP_licks_ENL2s/qc/s1_refit_*_1104.csv).** The new gate
  drops exactly the 2,908 in-working jaw-only trials (pre 404 / acute 1,277 / subacute 808 / chronic 419), and no lick
  trials.
  - Allen split: post-cue .47-.90 (was .43-.89), pre-cue .25-.55 (was .26-.56).
  - The acute post-cue drop is still seen in 4 of 4 animals.
  - Pre-cue acute is unchanged: up in PS95, down in PS93 / PS94, flat in PS92.
  - LocaNMF, without minus with jaw-only: post-cue acute +.03 / +.04 / +.10 / +.08 (PS92-95), PS95 subacute +.10,
    pre-cue within ±.02.

  Conclusions are unchanged; numbers rise slightly. The table is built but NOT applied to the page. Deck slide 9 is not
  updated either; its numbers are in docs/results_pages/imaging/s1_update_20261103_numbers.md. Both go with question 0.

- **Jaw-only trials look like no-response trials, not lick attempts** (`docs/status/JAW_ONLY_CHECKS_2026-11-03.md`).
  This holds in PS94 and PS95 (96 / 261 pre jaw-only trials), at cue and precue, on the joint and ROI bases. PS92 and
  PS93 (18 / 25) are too thin.
  - A decoder trained on pre licks reads jaw-only trials at .21 / .29 at cue (chance .167). That is the same as
    no-response trials (.20 / .25) and far below contact trials (.93).
  - Trained within the class, jaw-only reads about .20-.29 (chance .20), against .67-.79 for licks with the same
    trial count.
  - Jaw-only errors do not land on the neighbouring spout any more than no-response errors do. Contact errors do.
  - Adding jaw-only trials to TRAINING does not move the pre reference (B - A = -.002 [-.03, +.03]).
  - **Most of PS95's close ipsi acute drop (.975 -> .642) comes from CENTRING.** The label-free per-session centre is
    computed on rows that include the session's jaw-only trials. PS95 acute has 107 of them, all at far mid / far
    contra, and centring on them lowers close ipsi in all four animals.
  - Old .975 -> .832 in PS95 is the move from DAQ-lick rows to camera classes + August-only, independent of jaw-only.
  - Precue: every non-lick class reads .25-.39, against .50-.70 for contact trials. That is state before the cue.

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

- **(11-04) All Allen-ROI analyses use our ALM / tjM1 atlas** (the "split" basis of iti_licks, Priya 10-27: ALM and
  tjM1 = the anterolateral 40 % of MOs / MOp, plus disjoint MOs_rest / MOp_rest; whole MOs / MOp out; plain basis =
  robustness only). Until now this was the default only for the ITI decoders and the s1 table. Passed to 34 to apply
  across the rebuild with one shared definition. 34 will send the inventory of what changes, and anything that cannot
  switch.
- **(11-04) B, C, D, G (questions 1, 2, 4).**
  - **B:** state discriminators go in the spout bal_em frame, AFTER a re-check under the new working definition
    (centre on working rows).
  - **C:** Priya asked why not all three projections in the spout frame, centred on working rows like the decoders.
    Being re-checked; not decided.
  - **D:** section D's scored set -> working trials.
  - **G:** section G's groups -> camera classes (contact / incomplete / jaw-only / none, plus "after end"); pre panel
    = working.

  A helper is implementing these on main (no render; outputs in decoder_mix_checks/bcdg_1104/). 34 will copy the
  commits to the rebuild.
- **(11-04) Wait on the rebuild (question 0).** No main-branch render and no interim deck / page swaps. Slides 9-13,
  19, 89 and imaging page §1-4, 6, 7 change once, in 34's replacement pass. Questions 1, 2 and 4 are being clarified.

- **(late) Jaw-only trials are OUT of working** and are reported as their own class, like no-response. Working =
  contact + incomplete camera licks, cut at the session's last camera lick. This applies to the frozen training set,
  the per-session centre, the pre panel and the scored set. It replaces the afternoon rule (jaw or any lick). A helper
  is implementing it, retraining the cue / precue frozen models (new ids; old ones kept) and verifying against the
  checks' "jaw as its own class" row: acute close ipsi .727, pre panel .856.

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

- Nothing of mine is running (11-04). Jaw-only-out is DONE (48f2cc6d). e0's s1 reruns are done. 34's rebuild may be
  rendering on its own fenced dirs.
- Decoder helper (a44ea79f): checks done and the helper has exited. A is live; B and C are built but OFF. The combined
  render waits on Q1 and Q2 and on the jaw-only-out change. Once those are answered, the render covers SPOUT_BALEM_PAGE_EDITS sections 2, 5 and 7.
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

  Cue window primary, precue secondary. (Priya then asked for within-class arms and equal cue / precue.)
- 11-04: 34 acknowledged the split area basis: ONE shared wfield_local function, new spec ids. The inventory of ROI
  consumers is running. Order: the O2 encoder bundle goes first (Priya asked for it ASAP); if combined_encoding uses
  Allen areas, its path switches before the bundle is built; everything else switches after.
- 11-04: slide 94 (cued lick ramp) got the caveat line Priya OK'd via 34: "Acute increase rests on 2 animals (PS94,
  PS95; PS95 has 1 acute session); subacute / chronic contralesional tjM1 / SSp-m hold under the LMM (hidden
  slides)." It sits as a second line under the cued-row subtitle; PNG checked; hidden states unchanged (78 / 183).
  Script ramp_caveat_1104.py.
- 11-04: 34 will port the B / D / G commits from main (not implement them itself); C waits on the re-check. 34
  reports a Priya decision on the rebuild: jaw-only trials COUNT toward the >= 6 consecutive non-responses of a stop
  run. Main has no stop rule; section G's "after end" group = after the last camera lick. I passed this to the
  helper.
- 11-03 night: e0 added jaw-free arms to the s1 tables, pushed to main:
  - snapshot columns working_pl_*_licks;
  - LocaNMF refit arms camlickpl_*;
  - Allen-refit columns *_camlickpl*;
  - imaging §1 table primary = camlickpl_lick, with the jaw-inclusive arm kept as a robustness row.

  The reruns are running into LP_licks_ENL2s/qc/s1_refit_{basis,locanmf}_1104.csv; the 11-03 files are not
  overwritten. The table will be built but NOT applied to the page until you decide.
- 11-03 night: I HELD the main-branch render (question 0) after 34's timing answer. The rebuild has its own fenced
  output dir (rebuild_20261103), frozen store and spec ids, so there is no file collision, only CPU. The rebuild deck
  swaps are 1-2 days out. e0 is holding the imaging page edits: §1-4, 6 and 7 are working-gated, and the old wording
  appears 3 times. e3 kept trial_classes' "working" gate jaw-inclusive on purpose, as the robustness definition
  (cec38954); production = in_working & camera_lick. Still jaw-inclusive and passed to e0: the s1 tables
  (working_pl_*, camwork* refit arms) and the imaging §1 table.
- 11-03 night: 34 adopted jaw-only-out on its cohort-1 rebuild branch (rebuild/cohort1-20261103, not rendered yet). No
  current deck figure from 34 moves: those use the DAQ engagement or lick gate. The rebuild's render will REPLACE the
  camera-working-gated epoch families, decoders and maps. I asked 34 about timing and collisions before starting
  tonight's main-branch render.
- 11-03 late: the jaw-only checks reported (findings top, question 0). Endpoints were reproduced exactly first. Nothing
  in production was changed. The write-up is on main (its doc was swept into another window's commit b22686fc;
  attribution commit 96618fbd).

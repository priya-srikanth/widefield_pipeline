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

9. **NEW (11-04): mix-shift test for the SPOUT decoder and the C projections?** Their centre is computed from working
   rows, and the incomplete-lick share rises post-stroke. Frames to compare: bal_em on working rows / bal_em on
   contact rows only / z. Awaiting Priya.
8. **NEW (11-04): convert the coding-direction G9 classes (poststroke_lick / miss_working / stopped) to camera
   classes?** It touches grant_confusion, miss_vs_stopped, position_axes, the deck and best_match. [Yes, under "camera
   class everywhere"; 34 told to assume yes.]
7. **NEW (11-04): camera-class inventory, 12 questions** (34: wf_attempt
   docs/status/rebuild_20261103/CAMERA_CLASS_INVENTORY_20261104.md section 5, 913b9d1a). To switch:
   - G discriminators + precue_engagement_states;
   - frozen decoders' "lick" class / scored sets;
   - no-lick arm lists;
   - hit rate (Fig 0.2 / PI slide 4);
   - ITI first-lick rows / rt_drift;
   - the quiet-events engaged mask (ours).

   UPDATE (Priya): measures OF engagement (quit point, stop runs, quit prodrome, bout structure, disengagement
   tests) ALSO switch to camera licks / classes, with the session quit = last camera lick and the camera stop rule;
   DAQ stays as robustness and where there is no video. Sent to 34. The 09-20 engagement findings were on DAQ and may
   move. ANSWERED (Priya 11-04; sent to 34):
   1. after_end = quiet in quiet events, and a separate "quit" class in per-class tables.
   2. Hit rate = contact / all working-period trials, with contact-given-attempt alongside; jaw-only is never a
      response.
   4. Unjudged close spouts kept APART; asked 34 whether cam2/3 can judge the jaw there first, and when.
   5. No-video sessions dropped, with a DAQ-only row.
   6. DAQ-no-lick / camera-contact trials stay excluded; asked 34 for their frequency and ~10 example clips for
      Priya.
   7. early_run -> no_response.
   9. Lick class = contact + incomplete with LP timing.
   10, 11, 12: yes (attempted encoder gate; pinned epochs; jaw-only / no-response no-lick arms).

   Q4 side cameras (34): only the 10-27 single-frame pilot exists; there is no production side-cam pose yet. Path:
   O2 inference (1-2 days), a side-view jaw rule plus a blind review with Priya (~1 day), then a class rebuild and
   re-render, so several days to a week. It would change only the unjudged close-spout no_response trials (some to
   jaw_only, which could move stop runs). Recommendation (34 and me): don't block the rebuild; add side-cam judging
   later as a class revision with a diff table. A census of trials at stake and of Q6 (counts + ~10 example clips) is
   running: wf_attempt docs/status/rebuild_20261103/CONTACT_DAQ_MISMATCH_20261104.md.
   Q3 (Priya): YES, the camera working period, AS LONG AS final far-contra / far-mid failure blocks before the quit
   tail are not cut. Risk: the last-camera-lick end cut, and a >= 6-trial stop run equal to one far-spout failure
   block, would both remove genuine deficits. Proposed guard (sent to 34 with a census request): a tail / stop run
   counts as stopped only if it includes no response at a REFERENCE spout (close_L / close_center).
   Confirmed from trial_classes: a DAQ contact makes a trial "contact" whatever the cameras see; camera rules only
   classify no-contact trials.
   Q9 detail (Priya): contact trials with no LP lick time are aligned on the DAQ contact time and flagged
   (timing_source = daq); 34 is counting them.
   Cue + 70 ms floor (Priya asked if too long): a helper is measuring the first-contact latency from the raw DAQ
   channel (decoder_mix_checks/cue70_check_1104/, docs/status/CUE70_CHECK_2026-11-04.md). Priya's intent: if 70 ms
   stays, licks in 0-70 ms count as NEITHER post-cue NOR pre-cue (today they go into licks_precue). 34 is told to wait
   for the result.
   Census (34, CONTACT_DAQ_MISMATCH_20261104.md, 74dc76cd; 107 sessions, 57,048 trials):
   - Q6: 23 DAQ-no-lick trials the camera calls contact (0.23 %). All are contacts after the 3.5 s window, with no
     table mismatches; 19 are at far spouts. There are 11 example clips.
   - Q4: 1,121 unjudged close-mid no_response trials. 852 are in stop runs (unchanged by side cams) and 264 are
     isolated lapses (3.1 % of close-mid working; acute 7.4 %). That is the most side cams could move.
   - Q9: 0 contact trials lack an LP time. The real case is LP timing a LATER lick than the contacting one: 64 trials
     > 50 ms late, 10 > 100 ms, mostly close spouts. Proposed: use the DAQ time when LP is missing or late by more
     than a tolerance. Priya to choose 50 or 100 ms. [Suggest 50 ms: LP onset should PRECEDE contact by 20-40 ms, so a
     lag > 50 ms means LP missed the lick.]
   - ANSWERED (Priya to 34 directly):
     - Q6: a contact after max_rt does NOT count, because the spout is already moving. Camera classes use licks from
       cue + 70 ms (by peak) to min(cue + max_rt, stop). About 23 trials change class (13 -> incomplete). It is
       implemented on the rebuild with the 0-70 ms exclusion and a transition table; the O2 engaged set is re-derived
       from the table, with no input rebuild.
     - Q4: OK, no wait for the side cams.
     - Q9: tolerance ~100 ms: use the DAQ contact time (timing_source = daq) if the first LP lick is missing or
       > 100 ms after the first contact. 34 is checking why LP misses those licks.

   - Stop-rule guard census (34, STOP_RULE_GUARD_CENSUS_20261104.md):
     - The end cut barely matters: 43 trials.
     - Stop runs DO matter: 85 / 277 have no reference-spout trial (48 acute; PS95_0817 62 trials, all far
       contra / mid).
     - Trial-level guard: acute hit rate .691 -> .631 (far ipsi .739 -> .662, close contra .916 -> .845), and the
       acute deficit -.265 -> -.310. The working set / O2 / decoders are unchanged.
     - But pre also drops (.956 -> .941), so the guard restores some real disengagement: current rule = lower
       bound, guard = upper bound.

     Questions:
     - a. Test 34's refinement (keep a no-reference run only if there are reference contacts on BOTH sides)
       before choosing? [Yes]
     - b. A stop starts at the first reference failure. [Yes]
     - c. Unjudged close-mid no_response counts as stop evidence. [Yes: no lick at an easy spout]
     - d. The 13 sessions with no reference trial after the last lick get no end cut, flagged.
     - e. Drop the block-level guard.
     - ANSWERED (Priya 11-04): a. test it. b. Also test longer runs: the stop starts at the FIRST trial of a run of
       > 10 non-responses (plus > 8 / > 12), so one far-spout failure block can't make a stop. c. OK. d. Trials
       after the last lick in those 13 sessions are STOPPED. e. Dropped. An "end cut only" arm was added. Sent to 34,
       with pre hit rate as the specificity check.
     - ROUND 2 (34, STOP_RULE_GUARD_CENSUS_R2_20261104.md; after the last camera lick = stopped in every arm).
       Pre hit rate Δ / acute pooled deficit:

       | arm | pre Δ | acute deficit |
       |---|---|---|
       | A0 current | 0 | -.265 |
       | A2 both sides | .000 (0 pre trials restored) | -.289 |
       | A3 > 10 | -.002 | -.299 |
       | A1 trimmed | -.011 | -.309 |
       | A4 end cut only | -.023 | -.383 |

       A4 is ruled out: it scores disengagement as failure (acute close ipsi .782). Above > 10, A3 starts restoring
       close-spout disengagement. Working set / rest are unchanged in every arm. [34 + me: A2 primary; A0 and A3 > 10
       as robustness rows.] DECIDED (Priya 11-04): A2 primary, A0 + A3 > 10 robustness; sent to 34 (last merge before
       the render).
   - 34 merge queue before the render: encoder-classes, cue70 (+ max_rt end), enl-window (ENL follows the source,
     branch rebuild/enl-window), then the stop-rule guard once chosen.

   STILL OPEN: 3 guard numbers; and 8 (discriminator training classes; helper testing).
   Stays DAQ: pinned epoch boundaries, per-lick outcome, June / PS93_0813 /
   cohort 2. My suggested answers are in the chat reply of 11-04 and copied into the log below once Priya answers.
6. **NEW (11-04): B and C re-check results (interim; the D / G runs are still going).**
   - **B (state discriminators in the spout frame), under the new working definition: one marginal miss.** PS93's
     cue held-out pre engaged reference is .869 -> .850 (-.019) against a between-session SD of .016. No animal is
     worse by more than .03, every other pre reference is within noise, and the precue-state AUC is .584 -> .587.
     The centre already uses the working rows. Post-stroke, the G4 movement is about half the 11-03 estimate: cue
     acute engaged-like .195 -> .323 (was .502), fits_engaged .170 -> .322. Kept OFF by the strict rule. [I'd adopt:
     -.019 is 1.2 SD in one cell and inside the .03 tolerance.]
   - **C (projections in the spout frame, working-row centre).** Coding directions and pattern correlation stay within
     noise under both centres. The ENGAGEMENT AXIS fails under both (1.6-9.7x the SD, cue and precue); the shift
     comes from each scored session's own centre, not from the axis definition. Headlines with z -> working centre:
     - far-contra CD cue acute -.051 -> -.130;
     - precue .333 -> .087;
     - precue engagement axis acute .906 -> 1.230.

     [I'd choose: spout frame for CDs + patterns; the engagement axis stays in the z frame.]

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

- **11-04: D / G on main (4a37cbc8, bcdg_1104). Section D scored on WORKING rows moves the acute far spouts a lot.**
  Animal means:

  | readout | old (engaged rows) | new (working rows) |
  |---|---|---|
  | cue far contra acute | .478 | .184 |
  | cue far mid acute | .518 | .416 |
  | cue overall acute | .663 | .570 |
  | precue far contra acute | .271 | .135 |

  - The close spouts move by <= .012. The new rows are mostly far-spout incomplete licks.
  - The old option reproduces the pre-change code exactly (8 / 8).
  - Section G frozen accuracy by camera class, cue (pre / acute):

    | class | pre | acute |
    |---|---|---|
    | contact | .856 | .662 |
    | incomplete | .322 | .205 |
    | jaw_only | .356 | .157 |
    | none | .250 | .151 |
    | after_end | .164 | .225 |

    Chance is .167. Pre cells are thin.
  - C engagement-axis failure explained: in z each session is centred on its engaged (lick) mean, so its lick trials
    sit at the lick pole by construction. The readout's noise is tiny, and any other centre moves it.
  - Ported to 34: the list was sent. Gaps not yet on main:
    - (a) a camera-class state discriminator for G4 (check-script only);
    - (b) coding-direction G9 classes still DAQ (question 8 below);
    - (c) decode_matched's lick arm on engaged rows;
    - (d) attempt_coding is not wired for C.

- **11-04: G4 camera-class split + mix-shift test (docs/status/G4_CAMCLASS_MIXSHIFT_2026-11-04.md, 5f01f2dc). B should
  NOT be adopted: the z frame is the right one.**
  - Mix-shift on held-out pre sessions: z is mix-stable in all 4 animals (|bias| <= .05). Every frame centred on
    WORKING rows (z-on-working, spout frame, 11-03 jaw-incl. centre) reads more engaged-like as the mix shifts toward
    post-stroke: pooled no-lick bias +.06 / +.07 / +.17, mostly PS94 / PS95. The fault is the centre rows, not bal_em.
  - That artefact explains about half of the .195 -> .323 -> .502 gap, and more with the real post-stroke mix.
  - The engaged-like share is NOT carried by incomplete licks: calls spread over the classes in proportion to their
    counts. Incomplete > none only in PS94.
  - Under the camera-class discriminator in z, acute no-lick trials are no more working-like than pre-stroke ones:
    .18 vs .20; post - pre -.03 [-.20, .12].
  - Caveats from the final report:
    - The camera-class discriminator's positive class is mostly contact trials, so it reads "contact-like vs none"
      (it scores pre incomplete trials .30).
    - The DAQ discriminator's small rise (.137 -> .197) sits in jaw_only / none trials in PS92 / PS93 only.
    - The mix-shift test cannot test absorption of a global post-lesion shift, but z's DAQ-lick control is
      unchanged post - pre (-.015 [-.04, .005]).
  - Production stays z; nothing changes. The B option stays OFF. The B / C / D / G code is on main (4a37cbc8).

- **11-04 interim: the cue + 70 ms contact floor is CLASS-NEUTRAL.** 54,808 curated trials, raw DAQ contacts
  (cue70_check_1104/trials.csv). Shortening the floor to 0 / 50 ms changes the class of 3 / 2 trials. So trial classes,
  the working set and the O2 engaged set don't depend on it, and the O2 submit need not wait.
  - 2,167 trials (4 %) have their first post-cue contact < 70 ms, mostly PS92 / PS93 (PS92 pre 401 / 2,314, chronic
    421 / 3,894; PS93 chronic 414 / 6,068; PS94 / PS95 0-2 %).
  - First-contact latency: median 155 ms, 5th percentile 73 ms.

  Priya's rule (0-70 ms licks excluded from the pre-cue AND post-cue tables) would move only the lick tables /
  lick-count readouts.
  - Helper interim (102 / 107 sessions): the 0-70 ms contacts are NOT cue responses. The first-contact histogram is
    flat (~280-340 per 10 ms bin) from 0 to 80 ms, rises from 85-90 ms and peaks at 130-140 ms.
  - They are not bout run-through: only 35 of 2,167 have a contact in the 250 ms before the cue, and the median last
    pre-cue contact is 2.2-2.7 s earlier. They are isolated anticipatory / spontaneous contacts that the cue happened
    to follow.
  - The ENL rule holds in the GUI's licks, but the DAQ sees contacts in the last 2 s that the GUI missed: PS92 pre
    11.8 % of trials, PS93 pre 7.9 %, PS94 1.7 %, PS95 0.2 %.
  - 80 ms is the most the data support; 100 ms would cut real responses.
  - Keep 70 ms.
  - FINAL (107 sessions; docs/status/CUE70_CHECK_2026-11-04.md, 9533758f): the conclusion holds. The rule sent to 34:
    keep 70 ms (DAQ contact time; camera lick PEAK); 0-70 ms licks are excluded from both tables with a flag. Camera
    licks are judged by PEAK: onsets at 60-70 ms are real responses, so excluding by onset would cut ~1000 licks.
    This removes 2,277 DAQ contacts and 608 camera licks from the pre-cue tables. orofacial_outputs.lick_context
    splits on onset while the table splits on peak, an existing inconsistency to fix.
  - ENL windows (Priya asked; I said 1.0 s for LP): the widefield LP ENL is 1.0 s (DECISIONS 10-21, Priya "1.0 s, plus
    DAQ 1.0 s"); the 1.5 s ENL was the 2-photon pipeline's (2pRAM handoff 09-26). The widefield task ENL is
    2000-3000 ms. Open: switch the widefield LP ENL to 1.5 s? Census (qc/enl_lp_census_summary.csv, LP licks, sliding
    window), trials kept per animal x spout x epoch:
    - at 1.5 s: close spouts 96-100 %, far spouts 78-99 %. Lowest: PS93 pre far_L 78 % / far_center 79 %, PS92
      subacute far_R 79 %, PS93 acute far_R 82 %;
    - at 1.0 s: >= 94 % everywhere.

    So 1.5 s drops trials by spout, mostly far and mostly PS92 / PS93. DECIDED (Priya 11-04): the LP ENL stays at
    1.0 s.
  - ENL audit of the rebuild code (Priya asked):
    - No widefield record that 1.5 s was found fine. The 09 window sweep (DECISIONS ~664) found 1.5 s no better than
      2.0 s once the filter artefact was fixed. "1.5 s settled" is the 2-photon pipeline (2pRAM handoff 09-26).
    - The decoders follow the source: DAQ 2.0 s (precue_post_s) or LP 1.0 s (lp_licks.apply_enl_window). This covers
      the position / nolick / frozen / attempt_coding decoders and map_collect.
    - BUT these read precue_post_s = 2.0 directly, even under LP: precue_engagement_states :204,
      enl_movement_removed_decode, enl_movement_multisession, enl_fr_removed_decode, enl_video_spout_check,
      enl_lick_rates, quiet_task_events (ENL_S = 2.0).
    - within_spout_deviation uses a fixed 1 s (-1..0).
    - The encoders use no lick-free window: they model pre-cue licks as kernels. combined_encoding's "enl" measure =
      the 2 s pre-cue; pos_enl = strobe + 0.3 s .. cue.
    - DECIDED (Priya 11-04): every ENL analysis follows the lick source (LP 1.0 s / DAQ 2.0 s) through the shared
      window function. The seven direct-2.0 s sites and within_spout_deviation were sent to 34 for the rebuild.

- **11-04: what the G4 / state discriminators compare.** They are trained on pre DAQ-lick vs DAQ-no-lick trials
  (position-balanced; poststroke_compare.looks_like_which), NOT on camera working. So jaw-only trials were always in the
  no-lick class, in both frames. The no-lick group also contains incomplete licks (genuine attempts) and no-response
  trials. The B check's PS93 -.019 and the .195 / .323 / .502 spread are therefore pure CENTRING effects. Running (Priya
  OK): G4 split by camera class (incomplete / jaw-only / none / after end) under the three centres, plus a mix-shift
  test on held-out pre sessions resampled to post-stroke class mixes (decoder_mix_checks/g4_camclass_mixshift_1104/).
  B waits on it.

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

- **(11-04) Split by CAMERA CLASS for essentially every analysis where it can and should apply** (contact /
  incomplete / jaw-only / none / after end), replacing DAQ lick / engaged / outcome groupings. First case: the G4 state
  discriminators. They currently train on DAQ lick vs no-lick, which is out of line with the camera-class rule, so the
  frame choice must not be judged on them. The helper is adding a camera-class discriminator (working vs none, with
  jaw-only and after end scored separately) and a 4th frame, z on working rows; the current z frame uses each session's
  DAQ-lick trials. 34 was asked for an inventory of every analysis grouped by DAQ lick / engagement / outcome.

- **(11-04) Small items.**
  - residual_roi: pixel-based significance stays primary. The area table ALSO gets stats, adjusted for the larger
    split-basis family (sent to 34).
  - PS110 09-25 mCherry = the refined per-tile call after dedupe, 969 / 3,752 (25.8 %). It was already on the deck since
    11-03 (slides 155 / 157 / 161 notes and the 161 outline); nothing to change.
  - The 4 interim unpublished state models were DELETED: unused and unreferenced (ids in FROZEN_MODELS.md).
  - Stale "carried mostly by one animal" removed from the epoch_13pos caption (epoch_state.py); it shows from the next
    render.

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
- 11-04: 34 started the port on rebuild/bcdg-port:
  - 4a37cbc8;
  - (a) the camera-class G4 discriminator as primary, z frame, new ids;
  - (b) G9 camera classes as a separate droppable commit, pending Priya;
  - (c) decode_matched's lick arm moved to working rows;
  - (d) attempt_coding working masks.

  g_group: early_run -> no_response, after_end = "quit", unjudged close-mid kept apart. A2 is coded on rebuild/cue70.
  Merge order before the render: encoder-classes -> cue70 (+ max_rt end + A2) -> enl-window -> bcdg-port.
- 11-04: 34 added the engagement switch to the camera-class inventory as row 3h (bd8019be):
  - quit point = last camera lick;
  - camera stop rule;
  - quit_prodrome / bout structure / disengagement tests / LP gate checks and analysis_kit.session_behavior (feeds
    the quiet-events mask) on camera;
  - DAQ = robustness or no-video only; epoch boundaries pinned; the 09-20 engagement findings flagged for re-test.

  Also: per-trial O2 encoder outputs (every class scored out of sample, joined by trial_id) are being added on a
  separate branch. The fit itself is unchanged.
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

# Running list -- 11-06 evening onward (cohort-1 `_daqlk` rebuild)

Priya 11-06 ~20:40: "parallelize as much as safe and proceed through all pending tasks; keep a running list of any decisions
or questions for me and any notable findings / changes." Newest entries at the END of each section. Mechanics / state live in
the encoding handoff (STATUS_2026-10-26_ENCODING_HANDOFF.md, top blocks); the A1-A31 Q&A log is OVERNIGHT_2026-11-05_FOR_PRIYA.md.

## A. Decisions / questions waiting for Priya

1. (~21:00) **Kinematics page section 6 (trial-class shares) after the working-period gate -- accept the changes?** (B7) Chronic
   below-pre stars at far contra disappear (cleaner "back to pre"); acute close-contra no-response +0.038** -> +0.020* (loses
   Bonferroni); acute far-contra no-response .12 -> .10; "chronic no response <= .008 everywhere" no longer true at close mid
   (.019, n.s. vs pre). Text would change only in the replacement pass.
2. (~21:00) **Close-spout INCOMPLETE counts rise 3-5x under the new gate (new small stars, acute / subacute).** Cause not yet
   known -- investigating before anything is read into it (possibly cam1-only / low tongue events now classed incomplete).
3. (~21:00) **Pipeline gap: kinematics_pooled_figs still reads the UNGATED 10-23 pose3d/kinematics/sessions.csv** (Fig 7.5,
   Fig 9.6, 3-D rows of the section 3 / 5 / 7 tables). Proposed fix: re-run pose3d_kinematics under the gate with a routed output
   (or a gated read in the pooled script), on a branch, then re-run k_pooled. Expected impact ~0 (all other 3-D tables moved <=
   0.06), but unverified. OK to do?

## B. Notable findings / result changes (nothing on pages or decks changed yet)

1. **A27 cam4 cohort on the camera working period: no headline change.** 294 cells, same sessions; 0 sign flips in a significant
   cell; 13 borderline cells change status (q ~ .05). Chronic far-ipsi response rate "above pre" disappears (was the quit tail).
   OROFACIAL_FINDINGS numbers to update in the replacement pass (APPROVED): responded-trials row -> "(camera working period)",
   far spouts -0.19 to -0.26; far-contra first contact +978 ms; far-contra pre-cue -0.41 Hz. (overnight page 18:15)
2. **A28 3-D exec-bin calibration on the working period = the 10-23 snapshot** (0.000 mm max diff; snapshot stays).
3. **Camera-gate REST bug (fixed, ba5b63bf).** 8 / 107 sessions (PS92_0813 pre; PS92_0822, PS93_0818, PS94_0817, PS94_0820,
   PS95_0818 acute; PS93_0822, PS93_0824 subacute) were DROPPED from every REST-referenced output (_RESTref_ map column,
   rest-variant attempt maps) and had empty R5 / R5far / TASK sets in slide-102 gain_bio. Cause: quit boundary taken at the first
   non-engaged cue, but the camera working period is a span (first .. last camera lick). restw / MEAN / PRECUE / RESTnl / CD
   trajectories / restnl masks NOT affected. Any REST-column number on a page / deck lacks 5 acute sessions -> may shift after
   the re-run (v9r / v9s). Impact table: rebuild_20261103/ENGAGED_MASK_IMPACT_20261106.md.
4. **15r crash (fixed, 4ecf72ed):** PS93_0813 (no behaviour video -> no camera working trials) produced an empty-session tuple that
   killed the 15r ENL + cue arms in g5_epoch (only lick/lick 15r was drawn). Re-running 15r for all arms (v9s).
5. **Kinematics cache / jaw cross-check routing (fixed, 8d6da32d / 0e2154f0):** stale unrouted caches made the 3-D steps fail
   under the rebuild fence; recomputed kinematics = the 10-23 snapshot path (frames unchanged since 10-23). All A20 / A27 / A30
   kinematics outputs now DONE (v9k / v9k2 / v9r). Comparison with the current kinematics page: pending (helper).
6. Earlier today (provisional, a8): the acute rise in unique R2 of "movement WITH lick-position info" is NOT more regressor
   variance (text calling it "more variable licks" would be wrong); frozen contact-only R2 decline has no epoch-LMM star (p .06-.09).

7. **(~21:00) 3-D kinematics on the working period vs the kinematics page: no 3-D lick result changes** (spout error 3 / 3,495
   rows move <= 0.03 mm; phase <= 0.06, none of 1,188 sig bars differ; cue traces <= 0.23 deg; FAR_DIR <= 0.003 mm; cam4 angle
   Fig 3.6 -29.1 -> -29.2). All conclusion / star changes are in section 6 (see A1, A2). Report:
   rebuild_20261103/KIN3D_WORKING_PERIOD_COMPARE_20261106.md.

## C. What is running / next (summary; detail in the handoff)

- Render: main heavy slot one at a time (+ slot 2 for medium L-family steps). Re-run queues: v9r (rest attempt maps, slide-102
  gain chain), v9s (15r all arms + restnl / lpfinal / map_drift). Extra lanes added 20:38-20:41 to unblock lane-order stalls:
  e_split_order_* (-> e_angle_reach -> lick-angle decoder chain), l_cued_ramp_extract (slot 2 -> l_enl_angle), the 7 light v9s
  map steps in their own lane.
- Final catch-up queue after the render: 10 HOLD steps (mixed models, goal summary), aim / decoder-retention / recovery-timing,
  A25 LP epoch arm, goal-summary fig 3. Then provenance merge + replacement pass (pages, analysis deck, PI deck).

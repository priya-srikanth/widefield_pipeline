# Running list -- 11-06 evening onward (cohort-1 `_daqlk` rebuild)

Priya 11-06 ~20:40: "parallelize as much as safe and proceed through all pending tasks; keep a running list of any decisions
or questions for me and any notable findings / changes." Newest entries at the END of each section. Mechanics / state live in
the encoding handoff (STATUS_2026-10-26_ENCODING_HANDOFF.md, top blocks); the A1-A31 Q&A log is OVERNIGHT_2026-11-05_FOR_PRIYA.md.

## A. Decisions / questions waiting for Priya

1. (~21:00) **Kinematics page section 6 (trial-class shares) after the working-period gate -- accept the changes?** (B7) Chronic
   below-pre stars at far contra disappear (cleaner "back to pre"); acute close-contra no-response +0.038** -> +0.020* (loses
   Bonferroni); acute far-contra no-response .12 -> .10; "chronic no response <= .008 everywhere" no longer true at close mid
   (.019, n.s. vs pre). Text would change only in the replacement pass.
2. (~21:00; ANSWERED by analysis ~21:40) **Close-spout INCOMPLETE rise 3-5x = the gate, not a bug.** The working period now
   includes the post-DAQ-quit tail where the animal keeps licking without contact: 141 / 142 added close-spout incomplete trials
   come after the session's last engaged trial; no trial changed class (class definition differs in 53 / 56,979 trials overall).
   Mostly acute / subacute, mostly PS94 (acute 18 -> 57). Decision for Priya: section 6 text should say these are late-session
   misses after the DAQ quit, NOT a close-spout aiming deficit; optional sensitivity row with WFIELD_KIN_GATE=legacy. OK?
   (rebuild_20261103/CLOSE_INCOMPLETE_CAUSE_20261106.md)
3. (~21:00) **Pipeline gap: kinematics_pooled_figs still reads the UNGATED 10-23 pose3d/kinematics/sessions.csv** (Fig 7.5,
   Fig 9.6, 3-D rows of the section 3 / 5 / 7 tables). Proposed fix: re-run pose3d_kinematics under the gate with a routed output
   (or a gated read in the pooled script), on a branch, then re-run k_pooled. Expected impact ~0 (all other 3-D tables moved <=
   0.06), but unverified. OK to do?

4. (~23:35) **e_drift_june (June-INCLUDED ITI where-decoder drift, a superseded sensitivity) fails**: every June session's
   area load raises "Object arrays cannot be loaded" (likely no camera classes in June -> empty / object per-session files).
   Proposal: leave it failed for now (August-only is primary), look at it after the render. OK?

5. (~23:40) **OK to push fix/lp-novideo-skip-20261106 (72729126)?** Under the LP lick source a session that was NEVER filmed
   (no camera classes AND no raw .avi/.mp4 -- both must agree; PS93_0813 only) contributes nothing, exactly as under the contact
   arm, instead of crashing the camera-gate map collector. Any other failure (incl. a filmed session with missing LP tables) stays
   fatal. Map cache keys unchanged (maps stay warm). 57 tests pass. Unblocks g5_restnl_maps_lp, g5_lpfinal_maps, g5_map_drift_lp,
   and the A25 LP epoch arm (15r + beta maps).

6. (~02:10) **scripts/stopped_trial_traces.py refuses the camera trial gate** ("not wired to the camera trial gate (it selects
   trials by the daq rule); run it with WIDEFIELD_TRIAL_GATE=daq or wire it to trial_classes"). Steps l_stopped_restw (mode STOP --
   would have halted v8 lane X-L2-L3) and l_stopped_restnl are parked on HOLD. Options: (a) wire it to trial_classes (camera
   working / stopped classes; code change, branch, then run in the catch-up); (b) run it once with WIDEFIELD_TRIAL_GATE=daq as a
   labelled DAQ-gate figure; (c) drop it from this rebuild. Recommendation: (a), since the stopped class is now defined by the
   camera rule (A2 + excise) and a DAQ-rule version would disagree with the rest of the page.

7. (~06:00) **Section G is the bottleneck: let one BIG step run beside it?** g5_section_g has held the MAIN heavy slot since
   20:49 (9 h) and is at session 29 / 88 of its main pass (+4 sessions in the last 90 min -> maybe 15-20 h more; it computes and
   caches per-session within-session accuracy). It uses only ~4 GB (39 GB free), but every BIG step waits behind it: the
   lick-angle decoder l_enl_angle, t_frame_te_basis (-> decoder-pattern stats + ROI steps), the 15r re-run, the slide-102 gain
   re-run, a_partial_ls, and the rest of the v8 L chain. Proposal: when the main slot is held by a low-RAM step (section G), let
   the SECOND slot also admit BIG steps if >= 30 GB is available. Needs a coordinated lane restart (HALT -> waiting lanes give
   their claims back -> relaunch from copies with that rule; running steps are not touched). Risk: two big steps at once (~20 GB
   peak each) -- 4 + 20 + light stays under the box; the 10 GB pause and the RAM log stay on. Or leave it as is.

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
- 21:00-22:00: rest attempt maps re-run DONE (8 sessions back in: 0 "no engaged rest" lines, was 24); e_split_order_* all DONE
  (v7b lane E moving toward e_angle_reach); l_cued_ramp_extract DONE (-> l_enl_angle, waits main slot); 21:58 slot-2 lane for the
  8 ready + unclaimed medium steps (b_wsd_lick -> within-spout mixed models, l_revup_restnl, l_restnl_rescue, l_arrival_restnl,
  l_stopped_restw, l_disengage, l_interhemi_rest, l_lc_extract).
- 22:15 e_where_z DONE (unblocks v8c-S lane, e_overcorr_angle, e_rlink_*, and the aim producer input). 22:33 e_rlink_angle_z
  FAILED: decoder_retention_sessions.csv has no producer in any render queue (undeclared dep; A21 / A29 retention step).
  e_rlink_hit parked (HOLD); e_rlink_lat_z was already taken by lane E (fails fast, harmless). 22:35 queue v9t on main 4ecf72ed:
  d_decoder_retention_g1106 -> d_recovery_timing_g1106 (--window lick, A21 denominator) + e_rlink_{angle_z,lat_z,hit}_r.
  The aim producer still waits on e_first_lick (not run yet) -> stays in the final catch-up.
- 22:56 queue v9t DONE, all 5 steps exit 0 (decoder retention, recovery_timing --window lick on the A21 Fig 0.2 hit-rate
  denominator, e_rlink_{angle_z,lat_z,hit}_r). Not yet compared with the page (recovery timing / goal Fig 0.x): first look --
  lick-window neural retention still below pre chronically (-0.29 [-0.43, -0.13], 4/4 animals, 43 sessions); behaviour-minus-
  neural "gap" acute +0.30 [+0.03, +0.56] (only 3 animals, 6 sessions), subacute ~0, chronic -0.29. Compare in the replacement pass.
- 23:05 e_angle_reach DONE (lick-angle chain unblocked; l_enl_angle waits for the main slot). 23:00 g5_restnl_maps_contact DONE
  (restnl_maps uses ~17 GB in one process -- treat as heavy in future queues).
- 23:28 g5_restnl_maps_lp FAILED: PS93_0813 (no video) has no LP lick tables -> LPUnavailable is fatal under the camera-gate
  map collector. Same path in lp_final_render maps, map_drift --arm lp and the A25 LP epoch arm. g5_lpfinal_maps parked (HOLD);
  g5_map_drift_lp skips on its dep. Helper writing a narrow fix on branch fix/lp-novideo-skip-20261106 (skip only sessions with
  no LP licks AND no camera working period; every other failure stays fatal). Push needs Priya's OK.
- 23:29 e_roi_check FAILED: its input (Fig 3.33 decoder-pattern stats figure data) comes from scripts/iti_decoder_pattern_stats.py,
  which was in NO queue (missing producer). Queue v9u on main 4ecf72ed: e_pat_stats_maps -> e_roi_check_r / foci_r / cand_r /
  rim_overlay_r (started 23:30).
- 23:31 e_pat_stats_maps (v9u) failed: needs encoder_run('_te_basis')/fit spatial_<animal>.pkl = t_frame_te_basis (still waiting
  for the main slot). v9u re-armed with that dep; the 4 ROI steps follow it.
- 23:38 queue v9v DONE (A29): a_aim_by_spout + a_recov_timing_aim, exit 0 (check vs recovery_timing_pooled_day max |diff| 4e-16).
  First look, NOT yet compared with the page (kinematics Figs 1.4 / 1.5; recovery timing): cued-lick AIM, weighted over spouts:
  post -19.7 deg (SE 3.6), recovering +4.0 deg per log-day (p < .001; 107 sessions); far_mid -18.0 then +3.5 / log-day.
  Recovery per animal (PS92 / 93 / 94 / 95), behaviour (Fig 0.2 hit rate, A21) back to pre by chronic in all 4 (1.02 / 1.02 /
  0.98 / 1.01); lick-window neural retention only partly (0.69 / 0.38 / -0.47 / PS95 acute missing) -> behaviour recovers
  before / beyond the neural code. Compare with the page's recovery-timing text in the replacement pass.
- 00:53 v9s light-map lane DONE: g5_restnl_maps_contact / _enl1s and g5_map_drift_contact / _enl1s exit 0 (8 rest-bug sessions
  now in their REST rows; PS93_0813 skipped under the contact arm as before). LP arm (g5_restnl_maps_lp failed, g5_lpfinal_maps
  HOLD, g5_map_drift_lp skipped) waits for the LP no-video fix (A5). g5_epoch_15r_r still waits for the main heavy slot.
- 02:06-02:09 my slot-2 "ready steps" lane ran l_restnl_rescue / l_arrival_restnl before their (undeclared) producers -> both
  failed fast on missing inputs, nothing written; markers moved *_early_* so their own lanes re-run them in file order. The lane
  then halted on l_stopped_restw (mode stop) -- a real gap, see A6. l_disengage / l_interhemi_rest / l_lc_extract left to their
  own lanes. b_wsd_lick DONE 01:40 (-> within-spout mixed models lane v8e-T8); l_revup_restnl DONE.

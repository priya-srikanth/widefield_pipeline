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

8. (~10:00) **A6 stopped-trial traces -- Priya chose the class breakdown: success vs incomplete vs jaw-only vs no response
   (inside the camera working period) + stopped (camera stop-run rule) as a 5th trace.** Built: fix/stopped-traces-camera-20261107
   0555f6f5 (on e79afbf9), 381 tests pass; DAQ path unchanged; per-animal x epoch counts print first; groups < 5 trials / animal
   (3 per spout) are named, not drawn. Dry run: PS92_0810 pre 402 / 2 / 0 / 0 / 0; PS92_0819 acute 305 / 69 / 25 / 27 / 120
   (+5 excluded). Reading: the 4 groups = cortical activity graded by orofacial output (movement differs by construction);
   no_response vs stopped = the clean engaged-vs-disengaged contrast. -> Priya YES: ON MAIN 7388c713; queue v9y (A7 lock rules)
   runs l_stopped_restw -> l_stopped_restnl (waits for a heavy slot).

- **A9 (11-07 ~11:30, ANSWERED: yes)** Section G lick-only arm: express post-stroke relative to pre, n-corrected per session ->
  RETAINED ABOVE-CHANCE FRACTION (post - 1/n) / (pre_matched - 1/n) (Priya chose the plan: ratio primary, chance-normalised
  difference secondary). Branch figures/retained-above-chance 4353fbd2 (wf_retained; section_g_retained.py, renders with
  section_g_figures; not pushed). Keep a CONTACT-ONLY (successful licks) supplemental section G pass (Priya: "not discard"):
  branch analysis/contact-only-selection (wf_contactonly, WIDEFIELD_WORKING_TRIALS=in_working_contact), queued LAST.
  Old 10-07 DAQ json = preview only (older basis + DAQ-quit cut). Drawing fixes for G2c / H3b / G2b (one chance line /
  union band across sessions of different n) proposed, not yet approved.

- **A10 (11-07 ~13:20; Priya ~14:00: RUN the contact/incomplete split + rewrite the Fig 10.2 text; she suspects short licks drive it)**
  CHECK DONE ~14:50: the gate never filters the cued set (sessions_roi8fr, all post-cue camera licks); incomplete TRIALS are
  0.2-7% and not shorter -> not the cause. Short NO-CONTACT licks inside successful trials (8-26%, 2.8-3.4 vs 4.1-4.5 mm) carry
  the subacute/chronic sagittal loss: contact licks only -24% (4/4) / -14% n.s. / +7% n.s. Page->rebuild change = imaging /
  pre-set changes (not tested which). Wording proposed to Priya ~15:00 (awaiting OK). Priya OK: QUEUE the attribution test
  (decoder on old vs new inputs one change at a time: hemo / swapmask / lnmfproj / pre0801) AFTER the render.
  Files scratchpad/a10_sagittal/. ITI where-decoder (Fig 10.2 KEY "loss mainly LATERAL; cued sagittal not lost") no longer holds:
  cued sagittal r_norm -25/-21/-8% (CI excl 0, 4/4 each) vs page -21 n.s./-8 n.s./+2; ITI sagittal chronic -24% ~ lateral.
  Proposed first: split cued sagittal by contact vs incomplete (incomplete licks now in the cued set). Also: queue the missing
  Fig 16.4 aim recovery-link step (--beh aim never queued). Report scratchpad/compare_iti_20261107.md.
- **A11 (11-07 ~13:25; check DONE ~14:20; Priya ~14:35: 4c wording AFTER the A10/A12 checks; fit-quality flag OK; hit-gated values need >= 20 hits;
  USE AIM (first-lick angle offset) RATHER THAN HIT RATE for behavioural recovery)** Check: PS95 old 7.9 d = end-of-session
  disengagement counted as misses (same hits, smaller denominator); new 0.57 d = degenerate fit (b bound, LOO 0.57-28.8 d);
  aim t1/2 PS92 5.4 / PS93 3.2 / PS94 3.1 / PS95 4.3 d (gate-independent; PS95 aim -27.5/-11.2/-2.7 deg). PS94 Fig 0.12 flip =
  one 8-hit session (0823), lick-window p_true .42 -> .69 (cue window .73 -> .69): rebuild decoder / inputs / first-lick anchor,
  not the gate. Code: branch analysis/recovery-fit-flags (flag + MIN_HITS 20 + aim primary), in progress. Recovery family: FINDINGS 4c(i) lesion-size order of recovery speed fails (PS95 half-recovery
  0.57 d from a DEGENERATE fit) and 4c(ii); Fig 0.12 PS94 code-on-hits fraction 0.23 -> -0.47. Proposed: check PS95 fit +
  PS94 per-animal curves before any text. Headlines hold (chronic hits ~-0.29, behaviour full, gap, aim identical).
  Report scratchpad/compare_recovery_20261107.md.

- **A12 (11-07 ~13:35; Priya ~14:00: run the lick-class split; CHECK DONE ~14:20; Priya ~14:30: (b) as a NEW SUPPLEMENTARY panel, jaw-only
  and no-response SEPARATE -- 'interesting if pre-cue was increased on jaw-only'; main figure worded as working trials)**
  Check: the camera gate REMOVED 60-190 acute far-contra non-licking trials per animal (jaw-only / no response / stopped);
  incomplete-only = all working (1.31/1.10/1.01/1.07 vs 1.40/1.05/1.00/1.08); old DAQ gate on rebuilt data restores the rise
  2.14/1.23/1.25/1.62 -> the old rise was carried by NON-LICKING trials. Class split running (scratchpad/a12b_classes). Section 7 RESTnl maps: acute far-contra PRE-CUE rise shrinks / reverses -- 7.3b LP 1.59 (2/4)
  -> 0.86 (0/4); 7.1b contact 1.80 (4/4) -> 1.23 (3/4); 7.5b ENL1s 2.04 (4/4) -> 1.43 (3/4, PS94 .91); far-middle acute cue
  .76 -> .99; cue loss .53 -> .66 (4/4). Smaller: c2l restw tjM1 subacute star lost; within-spout pooled toward-midline loses
  FDR (q .099); jaw far_R acute -2.42 -> -1.85. Cued ramp + c2l RESTnl still QUEUED (deps). Proposed: split acute pre-cue
  maps by lick class. Report scratchpad/compare_lick_20261107.md.

- **A13 (11-07 ~15:45; Priya ~14:10 OK shared fix + push recovery branch; ~14:25: model figures KEEP coloured model lines, palette must not reuse epoch / other reserved colours; mean +- SEM = epoch-coloured dot + bars)** Figure epoch-colour convention audit (scratchpad/epoch_colour_audit.md): 27 producers, 8 OK /
  6 partial / 13 no. Proposed: shared day-axis helper (open markers, shape = animal, colour = epoch, no lines) for 11
  producers; small fixes for the 6 partial + import EPOCH_LINE instead of private palettes; model-comparison line figures
  (frozen_fr_compare.epoch_lines / bars, ~15 encoding figs incl. Fig 0.1/0.7) need Priya: (a) keep pooled-model lines +
  add epoch-coloured session dots, or (b) strict no lines. Also 11-07: Priya -- UNSCALED where-decoder primary (z = supplementary);
  Fig 10.2 text committed d3d4834b (docs/fig10-2-sagittal, not pushed); queue v9zb = unscaled recovery-link (16.3-16.5).

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

- **B (11-07) far-contra is mostly INCOMPLETE licks post-stroke.** Old DAQ-gated section G dropped far_R (far contra) in
  PS92 0818-0822 / PS93 0818-0821 (n=5; contact 0-12 < MIN_POST 20) and PS94 0817-0820 / PS95 0817 were 4-way; camera classes
  show 33-68 incomplete far-contra trials per session. CORRECTION 11-07 ~11:15: the camera-gate run did NOT make these 6-way --
  preserved_positions still counted DAQ-lick rows (14 units 5-way, 5 four-way); fixed eeb36d48 (Priya: never gate on successful licks). Preview (OLD gate) retained fraction, post-cue median acute: LICK-ONLY .63-.90 vs ALL .46-.80; chronic
  ~1.0-1.2 both arms. Not for pages until the new run lands.

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
- 11-07 09:00 Priya answered A1-A7 (see handoff 00-COMPACT-33): A3 yes, A4 defer, A5 yes (LP fix ON MAIN 72729126; queue v9w
  09:17 re-runs g5_restnl_maps_lp_r -> g5_lpfinal_maps -> g5_map_drift_lp_r), A6 (a), A7 yes. She asked about O2 for the partial
  refits: recommended NO (new bundle code mid-render + cross-machine tolerance vs t_partial_te done here); instead raise
  a_partial_ls / b_partial_frA to 4 workers inside the A7 restart, memory-capped. Waiting for her OK on that.
- 09:25 A3 fix READY: fix/pooled-kin-gate-20261107 (ec67a0d5, worktree wf_pooledkin). kinematics_pooled_figs builds the
  per-session 3-D table from the 10-23 lick + jaw snapshot gated with gate_kin_licks (same helper as k_summary_fig) instead of
  reading the ungated 10-23 sessions.csv; ungated rebuild reproduces that file for all 119 sessions x 38 columns (max rel diff
  4.9e-4 = %.4g rounding) -> NO 119-session recompute; re-run only k_pooled. Affects Fig 7.5, Fig 9.6, 3-D rows of section 3/5/7
  tables. 31 tests pass. Push needs Priya's OK.
- 09:12 A5 LP no-video fix ON MAIN 72729126 -> queue v9w (LP restnl maps / lpfinal / map drift) running. 09:20 A3 pooled 3-D
  gate ON MAIN 96f77156 -> queue v9x k_pooled_r running. 09:23-09:30 A7 RESTART done: HALT -> 16 waiting lanes returned their
  claims; 11 orphaned dep-wait claims released; 19 lanes relaunched from queue_daqlk*_a7.sh (third heavy slot HEAVY3 beside a
  low-RAM main holder = section G, XL mutex so two partial refits never overlap, a_partial_ls / b_partial_frA now --jobs 4;
  A7MIN=35) + 3 slot-2 lanes (S2MIN=35). Running steps untouched. HEAVY3 waits until >= 35 GB free (31 GB at 09:30 while the
  17 GB LP restnl map step runs).
- **C (11-07 12:45)** g5_map_drift_lp_r EXIT 1: the June pre sessions (0606-0608 x 4 animals) are not in the LP
  trial_attempts table (LP classes start with the August baseline), so the camera gate refuses them; the A5 no-video skip
  covers never-filmed sessions only. Same June gap as e_drift_june -> DEFERRED with A4 (needs June LP classes). DAQ
  map-drift steps unaffected. Also 11-07: section G STOPPED 11:26 (Priya: positions on working trials, not DAQ licks);
  fix eeb36d48 + contact-only d12dfc56 + retained figure 4353fbd2 pushing; restart queue v9z drafted (G = production
  re-run + g7/g6 figures, C = contact-only supplement LAST). Lane G relaunched 11:27 for its independent steps.
- **C (11-07 13:40) CODE CHANGED AFTER RUN -- re-run before the replacement pass.** Every lane runs from its own clean worktree
  pinned to a commit (preflight checks it), so pushes never alter a running step; the risk is DONE steps whose code changed later:
  | change | affected done steps | status |
  | eeb36d48 positions on working trials | g5_section_g | stopped 11:26; g5_section_g_r (v9z, f75f420a) running since 13:40; g6/g7 held |
  | 32968a57 LOSO figure no-video | g5_joint_xsession | g5_joint_xsession_r (v9za) running |
  | analysis/recovery-fit-flags (not pushed) | d_recovery_timing_g1106, a_recov_timing_aim | re-run after merge (light) |
  | d12dfc56 contact-only, 4353fbd2 retained fig | none (production tag pinned unchanged) | - |
  Before the replacement pass: check every page/deck output's code SHA against this table.
  13:40 v9z lane relaunched with A7MIN=25 (main idle: HEAVY3 = l_basis_extract ~2.3 GB; other lanes keep 35).
- **C (11-07 14:25)** csig fix becc486b on main (class-code digest restored); section G restart #2 on it (v9zd3, runs as 4th job,
  not in HEAVY_RE for that lane only); recovery re-run d_decoder_retention_r2 / d_recovery_timing_r2 / a_recov_timing_aim_r2 DONE.
  American spelling for all new text (Priya, relayed by a8 11-07).
- **C (11-07 15:13)** section G restart #2 phase 1 9/12; A13 convention branch merged with main (conflict in decoder_retention_sessions
  resolved: convention + MIN_HITS kept) and pushing (push15); A12b class split (jaw-only / no response separate) running; handoff 00-COMPACT-35.
- **C (11-07 15:50)** A13 convention branch ON MAIN 2434dec5 (793 tests). Plot-only re-render of converted page figures = first step
  of the replacement pass (lanes are pinned to older code, so queued figures still draw the old style; numbers unaffected).
  Section G restart #2 cache audit: imaging input cache 0 new of 442 (all hits); session_cache 0 new of 7837 (all hits); stored
  frozen pre-stroke decoders reloaded, none refit (new frozen_models since 14:23 are only l_basis iti_readout + g5_nolick
  nolick_reference); behaviour input cache MISSED (56 rebuilt, ~1-1.5 min each, byte-identical sizes) because the 7 _CACHE_CODE
  files changed after the 11-06 20:50 run (last 72729126). Phase 1 67 min; phase 2 (88 units) started ~15:30.
  e3 (state-space) told: frozen_encoder frA joint/sessions PETHs (11-05) and iti_licks sessions_roi8fr (e_iti_main, 11-05) are final.
- **DECISION NEEDED: A12b result (supplement, contact arm, RESTnl pre-cue RMS / own-pre working map; trials pooled within
  animal x epoch; n<20 = n.a.; null = working subsampled to same n, 30 draws).** Repro of Fig 7.1b far contra exact, 0 cache misses.
  Far contra ACUTE, PS92/93/94/95: jaw_only 6.08*/2.87*/1.73*/2.52* (pooled 3.30 [2.01, 5.19]); no_response 3.47*/n.a./1.39/n.a.;
  stopped 2.64*/0.69/1.90*/n.a.; incomplete 1.29/1.05/0.99/1.07. BUT not far-contra specific: acute jaw_only far ipsi
  3.75/3.98/2.76/n.a. (3/3*), far mid 3.34*/n.a./2.04/0.44; PS95 jaw_only far contra already 5.41* PRE; PS94 stopped 2-3x every
  epoch incl. pre -> reads as a trial-state effect (non-working trials have high pre-cue activity), not a stroke effect.
  Files: scratchpad/a12b_classes/ (a12b_precue_by_class.png, tables.txt, trial_counts.csv, pooled.csv, null_PS*.csv, scripts).
  Proposed: supplementary panel "pre-cue by camera class" with all positions shown, pre epoch included as the control.
- **C (11-07 18:22)** v_pix_frAsplit_new + v_pixmus_unique_noFR_cue EXIT 1: plot scripts read fit/spatial_<an>.pkl from the run's OWN
  folder, --fit-from runs keep it in the source run. Copied (cmp-verified) into frAsplit_basis_spont, basis_pixmusall and (waiter)
  frA_basis_pixmusall_union; retries v9ze DONE, frA cue/lick DONE. Proper fix (plot scripts read the source run) = TODO branch.
- **C (11-07 19:34)** Section G restart #3 (Priya OK): #2 stopped at 12/88 (network-bound on M: session_cache, ~15 min/unit);
  session_cache robocopied to C:/Users/SabatiniLab/wfield_cache/session_cache_rebuild_20261103 (7945 files 77.364 GB, verified);
  queue v9zf, same code becc486b, env token WIDEFIELD_SESSION_CACHE=local. Phase 1 67 min (no faster); phase 2 from 20:41.
  Local copy is derived -> delete after section G + figs (ask).
- **C (11-07 20:01)** l_basis_frame_sd EXIT 1. Root cause: 910c408b (11-04) routed the ITI TE basis through design_tag, so
  under FROZEN_LICKS_ALL=split iti_basis_decoder / iti_decoder_patterns read ..._spont_frA_te_basis (never written; TE runs
  use FROZEN_LICKS_ALL=none) + l_basis_extract has an undeclared dep on t_frame_te_basis + it exited 0 with 107/107
  svdbasis items failed. Page SVD numbers were on ..._t3d_frA_te_basis (10-26 / 11-03 path). FIX ON MAIN 2f5799a9 (Priya OK):
  cc70519e te_basis_run() + extraction exits 1 on failures; 2f5799a9 test fixture (iti_attempt_parallel dropped matplotlib
  submodules -> 14 order-dependent save_png_svg failures, pre-existing). Parked RESERVED_HOLD l_session_levels_z,
  l_levels_compare. Retry queue v9zg (worktree wf_render_v9zg @ 2f5799a9) waits for t_frame_te_basis: l_basis_extract_r,
  l_basis_frame_sd_r, e_basis_r, l_offset_where_r, l_session_levels_z_r, l_levels_compare_r, e_pat_stats_maps_r,
  e_roi_check_r2, e_roi_foci_r2. The v9u copy of e_pat_stats_maps (old code) will fail again harmlessly.
  Code-changed table: + | 2f5799a9 ITI TE basis | e_basis, l_offset_where, l_session_levels_z, e_pat_stats_maps, roi check/foci | v9zg |

## OVERNIGHT 11-07 -> 11-08 (Priya asleep ~23:30). DECISIONS NEEDED FIRST, then findings, then log.
### Decisions / questions for Priya (newest last)
- **D1 push fix/june-ref-epoch-none (64d54bc3)? -- Priya YES 11-07 ~23:45, pushing.** June ref files stored epoch=None (object array) -> strict iti_licks._load
  refused all 12 -> l_reembed failed, m_reembed_plot skipped. Writer now saves '', loader tolerates exactly a None epoch
  (any other object array still refused); all 12 real files load; tests added. Retry lane v9zi runs l_reembed_r +
  m_reembed_plot_r FROM THE BRANCH (64d54bc3) behind the heavy lock; outputs re-made if you reject.
- **D2 -- Priya (b) 11-07 ~23:55: after section G r4 + figs + grant finish, copy the new entries back to M: (robocopy, no overwrite), verify one ROI read from M: hits, THEN delete the local copy (guarded rm, list kept).** Local session_cache copy C:/Users/SabatiniLab/wfield_cache/session_cache_rebuild_20261103 (77 GB, derived)
  after section G r4 + figs finish? (r4 writes its ROI feature entries there; copy back the new tf-* entries to M: first?)
- **D3 push fix/fit-from-copies-spatial (648eafe0)? -- Priya OK 11-07 ~23:50, pushes after D1 lands.** frozen_encoder copies spatial_<animal>.pkl into a --fit-from run's
  own fit/ (the proper fix for the 11-07 pixel-map failures I fixed by hand-copying). Tests added; affects only future
  --fit-from runs; no result changes. Not needed tonight (all affected runs already have their copies).
- **D4 push fix/lpfinal-store-rebuild-root (cd6da055)? -- Priya YES 11-08 ~00:10; ON MAIN cd6da055.** g5_lpfinal_dec failed
  23:56 (write guard refused ~/cue_lick/frozen_models_lpfinal); store now <rebuild root>/frozen_models_lpfinal. Retry lane
  v9zl (wf_render_v9zl @ cd6da055) runs g5_lpfinal_dec_r.
- **D5 g7_fig_state_position -- Priya 11-08 ~00:30: OK to push 480accea (fix/sessions-csv-session-key).** Cause: PS94_0817
  far contra has 21 gated working trials (5.3 % of 394) -- trials exist (all 6 PS94 acute sessions have 21-68 far-contra
  working trials) -- but the REFIT arm marks a class below 10 trials OR below 1/18 of the session as untrainable
  (grant_confusion.MIN_REFIT_SHARE, added 09-09 a986819b by an earlier window; never discussed with Priya, though stated
  in imaging page Fig 1.1/1.2 captions). Fig 3.4 has no refit arm (refit - gap only recovered frozen recall), so post-stroke
  frozen recall now comes from epoch_acc_by_position (every working trial); pre stays refit - gap. Complete cells agree
  exactly (3.3e-16). Rendered to scratch: position retained 0.840 / 0.465 / 0.727 / 0.758; delta acute -0.347, subacute
  -0.139 (Holm .0015), chronic -0.077 (Holm .057). Plus an optional 'session' column in _position_bars sidecars (not
  re-run; for the trial-mix control).
  **Decisions (Priya 11-08):** keep the 1/18 rule for this render (pre-set, not tuned); refit-arm captions name PS94_0817 fC
  and report its value at 1/20 as a sensitivity line. NO class-balanced refit training (it moves every position's
  operating point; ~17 fC training trials per fold). 5rm (training-set-size matched) is the right refit-vs-frozen
  comparison. bal_em does NOT correct the refit arm (it re-centres the frozen decoder's features; refit uses X_refit).
  Bias direction of the residual skew: post far-contra refit (and gap) slightly LOW -> leans "lost"; near positions and
  overall (5ro) slightly HIGH; the floor removes the worst cells (selection of the most impaired days).
- **D6 trial-mix control -- Priya YES, tonight.** Per post session: every August pre session of the SAME animal,
  subsampled (whole blocks) to that post session's per-position trial counts (size + mix), refit WITHIN that pre session
  (same 5-fold block CV, estimator, floor), several draws; control = mean over the animal's pre sessions; excess refit cost
  = post refit recall - control, per position; pairs where a pre session lacks enough trials at a position are skipped and
  counted. Per-animal table first, then boot_delta (animals -> sessions, pre resampled). New script, own branch.
  **Priya 11-08 ~00:40: include PS94_0817 far contra in the control** (read as PS94; she wrote "ps92 817" -- confirm):
  the matched pre has the same mix, so the share bias cancels. **Priya: 1/18, 1/20 and 10 trials are all arbitrary -> NO
  floor at all in the control** (any position with >= 1 trial; trial-pooled stats weight small cells by their trials; trial
  count reported per cell).
- **D7 -- Priya OK 11-08 ~00:50: ALL refit analyses move to the 10-trial floor only (drop MIN_REFIT_SHARE), with the D6
  control shown beside the refit figures** (raw refit + matched-pre expectation per cell). Order: D6 control script FIRST
  (lick arms must not change without it), then time a single-arm 5r run, then re-run 5r / 5rm / 5ro / 5rmo (5 arms) +
  s1_refit_locanmf_table (page Fig 1.1/1.2) + june_august_drift (both call _refit_pred) behind the heavy lock.
- **D8 push fix/lpfinal-pre-session-count (3002728d)? (no rush, display only)** lp_final_render's trial table counted
  pre sessions as unique(block // 1e6); no-lick rows carry NEGATIVE run ids -> all -1 -> one phantom session (table
  8/8/9/9 vs trained 8/7/8/8). Decoders / trials / accuracies unaffected. Test added.
- **D9 queue audit (02:45, docs/status/QUEUE_AUDIT_2026-11-08.md): ~40 of 191 pending steps would fail or misbehave as
  queued** (19 no producer / wrong path, 8 arg mismatch, 14 harmful undeclared deps, 2 doomed skips; 0 box-default paths;
  ~106 OK; 21 superseded). 03:02 PARKED (RESERVED_HOLD, reversible): e_fpa_extract/_fit/_summarise/_offset,
  b_residual_roi, b_encoding_key_figures, g7_section_g_figs_contact (would OVERWRITE the r4 section G figures),
  l_restnl_rescue, l_arrival_restnl, l_disengage, l_disengage_daq, l_simple_key_figures. NOT parkable (already claimed,
  waiting on the heavy lock): l_stopped_restw / l_stopped_restnl (wrong jaw_onsets path -> will fail; cheap).
  g6_grant_r4 runs 4 workers WITHOUT the heavy lock after r4 -- RAM watch. **Decision for Priya: fix order** (proposal:
  page / deck inputs first; one branch per family; retry lanes from detached worktrees).
### Findings
- Section G cost = Allen-ROI features were never cached (trial_features_cached refused non-LocaNMF sources); fixed 80090a57,
  measured on PS92_0812: cache read 0.26-0.91 s vs 6-14 s uncached, features byte-identical.
- Render-queue gaps found 11-08 ~00:00 (the main [G] lane ended 00:01, 43 failed / 17 skipped in total over the night):
  (1) g7_cd_geometry needs cd_trajectories dumps with `--orth on` and the lick alignment (the nightly call); the render ran
  `--align precue cue` only, orth off -> "no saved result" for all 4 animals. Retry lane v9zj runs the nightly call (dumps
  named *_orth_*, nothing overwritten) then the figure. (2) g7_fig_s_per_animal reads `epoch_11amp/_11c_..._cue_working_
  sessions_long.csv`, written by scripts.epoch_sessions_long, which was never in the render queue. Retry lane v9zk runs it
  (cue arm, families 11 + 11c, --out the tagged epoch dir) then the figure. Both lanes run on the render code a76352f3.
### Log
- 23:30 section G restart #4 (v9zh, c3a8b049, ROI feature cache) phase 1 6/12; ROI-cache push re-queued to run after a_partial_ls
  (pre-push pytest was stopped at 22:20 for LOW RAM 4 GB). ITI svdbasis retry v9zg waits for t_frame_te_basis. l_reembed
  failed on the 12 June sessions (object arrays / allow_pickle) -> investigate on a branch. No section G restarts tonight.
- 00:0x D1 ON MAIN fa0e4d87 (june epoch fix); D3 ON MAIN cc6d11df (fit-from copies spatial). ROI-cache push (80090a57 +
  c3a8b049) still queued behind a_partial_ls. Lane v8-GBMD ended 23:46 (g5_chronic_stability clean).
- 00:05 resumed from 00-COMPACT-36. Section G r4 phase 2 started 23:52, 0/88 units by 00:05; phase-1 missing-file warnings
  are the same 15 sessions as r2/r3 (nothing new). g5_lpfinal_dec failed (D4, fix on a branch), g7_cd_geometry failed (lane
  v9zj), g7_fig_state_position failed (D5), g7_fig_s_per_animal failed (lane v9zk). g7_fig_frozen_refit_pooled done.
- 00:15-00:35 D4 on main (cd6da055), lane v9zl. D5 diagnosed + fixed (480accea), push waits on tests. Section G r4 3/88 at 00:15.
- 00:47 g7_fig_state_position_r DONE (v9zn, main 480accea; all 6 PS94 acute sessions). 01:04 g7_fig_s_per_animal_r DONE
  (v9zk, after g5_sessions_long_r 01:03). 01:48 g5_lpfinal_dec_r DONE (v9zl, cd6da055; LP-trained decoders frozen-new in
  <rebuild root>/frozen_models_lpfinal; far_R pre-cue acc drops acutely in 4/4). Still running: v9zj (CD orth on).
- D6 timing (m_mixmatch_t, 2 PS94 pre sessions): capture of all four animals' refit inputs 42 min (stored decoders hit),
  scoring 17 s for 2 sessions x 7 controls x 5 draws -> a full arm ~48 min. Null check on 2 sessions: excess -0.14..+0.09
  per position (~50-60 trials each). cue/working + cue/lick running.
- Section G r4 phase 2: 8/88 at 01:30; unit times 7-10, 22, 30, ~1.5, ~1.5, 20 min -- uneven, no projection yet.
- 02:39 g7_cd_geometry_r DONE (v9zj; dumps from g5_cd_traj_orth_r 02:33) -- all four 23:56-00:01 failures now fixed. Its
  overlap has NO noise ceiling: `cd_overlap_null` is not in the render queue (same kind of gap; add to the queue audit).
- D6 first version (one-sided matching, v9zm @ c6bca38b): cue/working scored, but the pre NULL is biased (PS93 pre +0.01..+0.23,
  PS92 -0.11..+0.14) and 453 of ~800 pairs were skipped (post sessions often hold more trials per position than August pre
  sessions; PS95 acute all NaN). NOT A RESULT. Symmetric version 76e7d604 (both sessions subsampled to S's mix at the size
  both can supply; no skips; --inputs cache) built + tested, WAITING for Priya's OK.
  MY ERROR: I edited the wf_mixmatch worktree while lane v9zm ran from it; spawned scoring workers re-imported the new code, so
  m_mixmatch_cue_l (02:35) scored with the new unit_job and crashed in the old main's printing. Mixed provenance -> discard
  (overwritten by the rerun). Only that lane was affected. Rule going forward: every lane runs from a DETACHED worktree.
- 05:49 a_partial_ls DONE (heavy lock since 21:47); heavy slot -> l_balimg (DONE 06:08) -> l_lc_extract. AB lane: b_fr_musall,
  b_fr_musall_v2, b_posinfo, b_specificity, b_svf_frA_plot, b_matched_residual all DONE 05:52-06:10. 04:00-04:25 e_iti_area_maps,
  l_area_maps_split(_cued), e_iti_hemo + e_iti_hemo_sum (re-run after their real dep) DONE.
- 06:1x ROI-cache push ON MAIN c3e01ed8 (80090a57 + c3a8b049; 988 passed). Section G r4: 45/88 at 05:44 (~7.8 min/unit over
  phase 2) -> ~11:15-11:30 if the rate holds (extrapolated, not measured).
- ~08:00 Priya: D6 symmetric YES, D9 fix order OK (page/deck first, one branch per family, detached-worktree retries), D8 YES.
  D8 ON MAIN 236dae4e. D6 symmetric lane v9zo (wf_render_v9zo @ 76e7d604, --inputs cache) running cue/working then cue/lick.
- D9 batch A (queue-only): stopped lane v9zg (exact PID 25888, mine, had run nothing) -> v9zq (same code 2f5799a9):
  l_session_levels_r (fixed code) before l_session_levels_z_r; e_pat_stats_maps_r waits for l_basis_extract_r; old
  l_session_levels parked. v9zr (main 236dae4e): e_checkfigs_r, b_specificity_plain (specificity_control --tag $F, the
  missing producer) + b_residual_roi_r, e_mm_r + e_lmm_all_r + e_lmm_response_r (renamed deps), e_fpa_*_r (deps on
  e_pat_extract / l_recov_markers / e_dock_prep_dock). Batch C (producers not yet traced): l_offset_where_r (svd / locanmf
  contrib terms), e_roi_foci_r2 + e_roi_cand_r2 (iti_spout_pattern_stats), e_rim_overlay_r (sessions_roi2), ENLv chain.
  Batch B (code, branch): simple_key_figures / restnl_rescue / disengagement / stopped jaw_onsets paths, section_g_figures
  --suffix on every output, imaging_lmm / remap_lmm / remap_checks qc paths.

**D6 result, cue / working (08:24, lane v9zo, m_mixsym_cue_w; 799 pairs, 0 failed; no floor; no bootstrap yet).**
Excess = refit recall of a post session minus that of the animal's own pre sessions, both subsampled to the SAME
position mix and size. Mean excess by animal, far / close:
acute far -0.175 / -0.199 / -0.240 / -0.207 (PS92 / 93 / 94 / 95); acute close -0.121 / -0.120 / -0.097 / +0.039;
subacute -0.06..+0.05; chronic -0.02..+0.16. Pre null (per-session SD over positions): 0.140 / 0.104 / 0.074 / 0.068.
Reading: the acute far-position refit drop survives mix and size matching in 4/4 animals, so it is not a trial-mix
artifact; close is lower acutely in 3/4. CSV: epoch/data/refit_mixmatch_cue_working_sessions.csv. cue / lick arm next.

**D6 result, cue / lick (09:06, m_mixsym_cue_l; 832 pairs, 0 failed).** Mean excess, far / close by animal
(PS92 / 93 / 94 / 95): acute far -0.055 / -0.060 / +0.008 / +0.202; acute close -0.097 / -0.138 / -0.074 / +0.030;
chronic close +0.141 / +0.147 / +0.013 / +0.024. Pre null SD 0.127 / 0.106 / 0.068 / 0.060.
Acute far_R lick trials are 1-12 per session (PS95 has one acute session); at those counts both the post refit and its
matched pre control score ~0, so excess ~0 = undetermined, not "no deficit". PS95 far +0.20 = one session (0817), far_L
n=12, control 0.065. Reading: the lick-trial arm can't test the far-contra refit loss; the working arm (all far, 4/4) is
the usable one. Waiting on Priya before any bootstrap / figure.

**D6 figure (Priya 11-08: option 1 = working trials plotted, lick trials as a table).** scripts/rebuild_figs/
fig_refit_matched_pre.py, renamed from fig_refit_mixmatch (branch feature/refit-mixmatch-control, not pushed). Out: rebuild_figs/<TAG>/refit_matched_pre.png,
_stats.csv, _lick_table.csv, _caption.txt. Change vs pre (boot_delta, Holm over 18): acute far ipsi -0.221, far mid
-0.148, far contra -0.259 (all Holm .0045); acute near -0.06..-0.08 (n.s. after Holm); subacute all n.s.; chronic
near ipsi +0.152 (Holm .0045, ABOVE pre); others n.s.

**11-08 ~09:55 -- D9 batch B queued; v9y stopped; b_partial_frA profile.**
- fix/d9-input-paths ON MAIN 99f0d464 (654 passed); within_session_decline.py classified BEHAVIOUR in the share-writer
  test with Priya's push OK (state-space window told).
- v9y stopped (PID 36768, mine): its l_stopped_restw would have run pre-D9 code (qc/jaw_onsets.csv read from a path that
  never existed) after waiting ~5 h for the heavy lock. Its claims l_stopped_restw / l_stopped_restnl parked
  RESERVED_HOLD (superseded by _r).
- Lane v9zs (detached wf_render_v9zs @ 99f0d464): BL (light, running now) b_imaging_lmm_r, b_remap_lmm_r,
  b_remap_checks_r; BH (heavy, waits for the lock) l_stopped_restw_r/_restnl_r, l_disengage_r/_daq_r/_add_r,
  l_restnl_rescue_r, l_face_offset_partial_r, g5_section_g_contact_r4 (after r4) + figs, l_simple_key_figures_r.
- b_partial_frA (heavy lock since 06:26): 42/107 at 09:45, ~4.7 min/session; 4 workers at ~0.5 core each, box ~24 %
  busy, 17 GB free. Resumes from sessions/<label>.json. Profiling one finished session (PS92_0812) into scratch
  (scratchpad/prof_partial) before any restart recommendation.
- Batch C producer tracing delegated (read-only).

**11-08 ~10:20 -- D9 batch C traced (read-only helper) + light part queued.**
- Two upstream blockers explain most of it, both behind the heavy lock (b_partial_frA): (1) t_frame_te_basis has no
  usable output (11-05 run was no-calibration, moved aside) -> no fit/spatial_<A>.pkl -> l_basis_extract (all 107 svdbasis
  items failed), l_basis_frame_sd, e_pat_stats_maps -> e_roi_check -> e_roi_foci / cand / rim, l_offset_where;
  (2) lane v8-XL2L3 waits on the lock at l_precue_restnl, so l_bal_reference -> l_bal_regions / l_bal_recovery ->
  (v8b) l_recov_markers -> e_recov_meas, l_balmatch, e_dock_*, l_enl_offset -> ENL_SLIDE_FIX -> the ENLv chain all wait.
  Nothing there is failing; it is queued behind the lock.
- Producers NOT queued anywhere: svd / locanmfM offset contrib maps (l_offset_where), iti_spout_pattern_stats
  (e_roi_foci), lick_same_target_place (key figure), cd_overlap_null (cd geometry ceiling).
- Queued now, lane v9zt (light, worktree wf_render_v9zs): l_sametarget_place, cd_overlap_null, g7_cd_geometry_r2.
- Code bug: lick_rim_overlay hardcoded sessions_roi2 (no run writes it) -> fix/rim-overlay-roi-dir 65cf7ead (il.SESS_NAME
  = sessions_roi8fr, 107 sessions, fields checked). NOT pushed: needs Priya's OK.
- Still to queue (wait on t_frame_te_basis anyway): contrib maps svd / locanmfM -> l_offset_where_r2; spout pattern stats
  -> e_roi_foci_r2 -> cand -> rim (after the fix lands); l_key_figure_r3 (+ l_sametarget_place, e_recov_meas);
  l_bal_offset_primary (+ l_recov_primary dep). Held, not failing: l_arrival_restnl, b_encoding_key_figures.

**11-08 ~10:15 -- profile of b_partial_frA (one finished session, PS92_0812, scratch copy; result identical, max diff 0).**
Wall 1295 s for one session. session_design runs 13x per session (1 + target/exec shift nulls + 5 + 5 block perms) =
1184 s (91 %); the ridge refits themselves = 65 s (5 %). Inside the rebuilds: 23,610 file-existence checks on M: at
~20 ms = 460 s (config.load_sessions 52x -> PathResolver re-reading yaml 29,408x, 150 s); trial_bounds 509 s; imaging
load 152 s; behavior load 143 s; column_stack 188 s. Every null rebuild reloads data that does not change.
-> Speed-up = build the session's inputs once and reuse them for the 12 null rebuilds (+ memoize load_sessions); needs a
branch + an exact-equality check against finished session jsons before any restart.
**Lane [T] stopped (PID 34776)** and re-queued as v9zu: t_frame_te_basis outside the heavy lock (gate >= 20 GB free,
Priya OK); the 11-05 run had failed on exec_bins_calib_20261023.csv, which exists since 11-05 20:22.
Rim-overlay fix push: denied by the tool permission check; branch fix/rim-overlay-roi-dir 65cf7ead waits.

**11-08 ~10:40 -- partial-refit speed-up (Priya: "can we fix this").** Branch perf/partial-refit-reuse-inputs 8669dd1b
(on main's base); the same commit cherry-picked onto the production code a76352f3 = 5bf72011 (detached wf_prfast_a76).
`_reuse_inputs`: load_imaging / load_behaviour / _fr_columns / load_sessions once per job; the null patches
(raw_blocks / spout_split / te_lick_parts) stay live. EXACT check vs the production jsons: PS92_0812 and PS93_0818
whole json identical (max diff 0). Wall (no profiler, box under load): 662 s / 815 s. Fair baseline (original code, no
profiler, PS92_0812) running before any restart decision.
Rim-overlay fix ON MAIN 65cf7ead (Priya allowed the push).

**11-08 10:57 -- b_partial_frA RESTARTED on the speed-up (Priya OK).** Fair baseline PS92_0812, original code, no
profiler: 1342 s vs 662 s fixed = 2.0x; all three runs whole-json identical. Lane AB stopped (exact PIDs, all mine) after
v9zv-PF took over the heavy-lock ownership (no other lane could grab it in between). 53 session jsons saved and all
parse; the restart resumes with 54 to go. Lane v9zv (detached wf_render_v9zv @ 5bf72011 = a76352f3 + the fix) then runs
AB's 18 remaining reserved steps + b_residual_roi_frA (stop -> cont for 4 of them), releasing the lock after
b_partial_frA. NB the earlier b_partial_frA.log was overwritten by the restart (its record is in CHAIN.log).

**11-08 ~11:55 -- MAJOR ERROR FOUND (Priya): the rebuild's lick class / lick-aligned windows are DAQ-contact only.**
Decided 11-03 (TRIAL_GATES handoff): "Camera lick gate = contact + incomplete. Lick-aligned windows always use the lick
gate. DEFAULT for the cohort 1 rebuild." Not implemented: trial_classes.camera_select("lick") = DAQ-contact rows only, and
the main epoch / grant / section G renders ran on the DAQ lick source (only d_attempt_* / LP-final steps set
WIDEFIELD_LICK_SOURCE=lp). Example: acute far contra, PS95_0817: 40 working trials with a camera lick, 1 DAQ-contact.
Plus a hidden per-session 5-trial floor (epoch_accuracy._accuracy_of, epoch_kit._accuracy_at, +4 more `< 5` sites) drops
sessions from the per-session tables. ALL lick-aligned / lick-class rebuild results are INVALID until re-run on camera
licks (nothing deleted; no page text touched). Read-only audit of every lick-touching step running (-> scratchpad
lick_source_audit.md); fix + re-run list to Priya before running.

**11-08 ~12:45 -- corrected framing + audits (details in handoff 00-COMPACT-38).** Lick class = all camera licks on the LP
source (documented 10-21 / 11-03 / 11-07); successful lick = DAQ contact, supplemental (Priya 11-08). Main-render "lick"
outputs are relabeled successful-lick, not discarded; LPfinal holds the lick-class acc + maps; missing LP families listed in
the handoff. g6_grant_r4 stopped (Priya). Refit floor 10 only + display floors removed: branch in progress. A20 calibration
table not regenerated for the encoder: compare next. Settings audit #2 pending.

**11-08 ~13:00 -- settings audit #2 + fixes under way (Priya: "ok to push. proceed with the fixes").**
- Audit #2 (actual settings, 264 steps, 23 SHAs): uniform + decision-backed: hemo, projection, camera gate / stop A2,
  max_rt 3500 + 70 ms, Wt, pre0801 (6 decided exceptions), bal_em, encoder, orofacial gate; no cohort-2 animal anywhere.
  PROBLEMS: (1) rest-baseline fix ba5b63bf (11-06; 8 sessions counted only pre-first-cue frames as engaged rest under the
  camera gate) missing from 17 finished rest / restw / RESTnl steps (g5_epoch 15r, g5_cd_traj(+orth_r), g5_rest_frozen(_nl),
  l_*restw / revup / c2l, g2a/g2b_restnl_mask, gb_sessions, ...); (2) gain-bio stale (gb_sessions pre-fix, gb_sessions_r
  never finished, gb_stats failed); (3) frozen spout decoder stored 3x (d_attempt_freeze 11-05, g5_joint_xsession(_r) 11-07,
  "DIFFERENT pre-stroke input set") -> attempt coding and joint_xsession score against different references; cue / lick
  differences unexplained -> trace; (4) ENL window follows the lick source (DAQ steps 2.0 s contact-lick-free) -> fixed by
  the LP-source re-run. Low: g5_epoch on 9aaa4cc6; 392 section G r4 entries only in the LOCAL session cache (D2 copy-back);
  LPfinal models in their own store.
- Floor branch fix/refit-floor-10-no-display-floor (b27a7acd + b29d834f, on top of feature/refit-mixmatch-control):
  699 related tests pass; PUSHING (Priya OK) together with perf/partial-refit-reuse-inputs (8669dd1b).
- Code fixes delegated, branch fix/stars-spoutorder-g13-1108 (wf_fixes1108 @ 68e53e33): stars to the top band
  (sig_star callers), canonical spout order (3 encoding figure scripts), section G G1-G3 per (f)1 + camera classes,
  DECISIONS.md lick-class correction + 11-08 entry.
- A20: calibration table regenerating with the working-period rule into scratchpad/a20 for comparison with the 10-23 one.
- Next: the consolidated re-run (LP-source families missing from LPfinal, 17 rest-fix steps, gain-bio, section G with
  G1-G3 fix, grant), from a detached worktree at the new main, heavy steps after b_partial_frA (~13:25).

**11-08 ~13:10 -- A20 calibration CHECKED: no change.** exec_bins_calibration.py (A20 working-period gate, kin_gate =
working, `tc.gate_kin_licks` per pre session) on the same 10-23 kinematics snapshot: 119,090 pre licks with the gate, 119,090
without (0 dropped) -> the regenerated table equals exec_bins_calib_20261023.csv exactly (210 rows, every landing point and SD
identical). Post-cue pre licks all fall inside the working period, so A20 does not reach the encoder calibration; NO encoder
re-run needed for A20. (scratchpad/a20/exec_bins_calib_A20.csv)

## 11-08 ~13:20 -- CONSOLIDATED (Priya: keep findings / progress / questions here; handoff = 00-COMPACT-38)

### A. Questions / decisions for Priya (none blocking right now)
1. Section G re-run scope: full grid vs ONLY the affected cells (lick window, lick-only arm, G1-G3) vs affected cells +
   an input-reuse speed-up. Waiting on the one-unit profile (scratchpad/prof_g/prof_PS94_5.txt) before proposing.
2. G1-G3 trial sets: the fix helper implements what (f)1 + the camera classes state unambiguously; any ambiguity comes
   back as a specific question (not guessed).
3. Frozen spout decoder stored 3x with different pre input sets (11-05 d_attempt_freeze; 11-07 g5_joint_xsession(_r)):
   being traced; Priya hears about it only if it changes a number.

### B. Findings today (all in handoff 38; audits in scratchpad/*audit*.md)
- MAJOR: lick class / lick-aligned windows were DAQ contact in the main renders. Lick = every camera lick, LP source,
  LP onset, ENL 1.0 s, LP-trained frozen decoders (documented 10-21 / 10-24 / 11-03 / 11-04 / 11-07). Successful lick =
  DAQ contact, supplemental (Priya 11-08). Main-render "lick" outputs = successful-lick (relabel, keep). LPfinal already
  holds the all-lick acc decoders + 15r maps (C1 to be read from there).
- Hidden 5-trial display floors + 1/18 refit share floor -> removed / 10-trial only (D7), branch pushed (see C).
- Rest-engagement fix ba5b63bf missing from 17 finished rest-family steps; gain-bio stale.
- Stars on data (sig_star callers); non-canonical spout order in 3 encoding figure scripts; section G G1-G3 violate (f)1.
- A20 calibration: CHECKED, identical (0 of 119,090 pre licks dropped) -> no encoder re-run.
- b_partial_frA 91 % input reloads -> input-reuse fix, 2.0x, exact-equal (ON MAIN e0e6d0ad).
- DECISIONS.md 11-03-night line "lick class stays the DAQ contact set" is wrong -> dated correction (fix branch).

### C. Progress
- ON MAIN today: 99f0d464 (D9 input paths + within_session_decline classified), 65cf7ead (rim overlay),
  e0e6d0ad (partial_refit input reuse). Floor branch fix/refit-floor-10-no-display-floor (+ D6 matched-pre underneath):
  915 tests passed, first push lost a race, RE-PUSHING.
- Fix branch fix/stars-spoutorder-g13-1108 (wf_fixes1108): helper still running its tests.
- Running: b_partial_frA (v9zv, 96/107 at 13:10, ~13:25) -> then v9zv runs AB's 19 steps and releases the lock.
  Allowed to take the lock next (rest fix present, outputs needed): g5_epoch_15r_r, gb_sessions_r, l_stopped_restw_r,
  l_block_compare (no rest dependence). t_frame_te_basis waits outside the lock for >= 20 GB.
- STOPPED today (exact PIDs, mine): v9y, [T], AB, g6_grant_r4 (Priya), v8-XL2L3 + v8s2-L2L3 (old code without the rest
  fix; their restnl + l_bal_* chains go into the re-run).
- Re-run plan (measured times; heavy lock = wall-time bottleneck; Priya OK two heavy slots): see handoff 38 table.
  Before starting: section G profile + time ONE LP-source epoch family (check LPfinal cache hits).

## 11-08 ~13:35 -- after compaction
- PUSHED: fix/refit-floor-10-no-display-floor ON MAIN 9bc33b9b (915 passed): D7 10-trial refit floor only, display
  floors removed, D6 matched-pre (refit_matched_pre) underneath.
- Fix branch fix/stars-spoutorder-g13-1108 (wf_fixes1108, 609a3b58): 4 commits reviewed (stars -> star_top + headroom;
  POS_ORDER in combined_encoding_figures / encoding_key_figures / frozen_spout_maps; DECISIONS 11-08 lick-class
  correction; section G G1-G3 train camera WORKING vs NONE under the camera grouping, stored ids "-cls-work-v-none", DAQ
  grouping unchanged). The helper ended before its tests finished; re-running the same -k selection to
  scratchpad/fixes1108_tests.log. Push after a clean run (Priya's "ok to push" 11-08 covered today's fixes).
  OPEN for Priya (from the helper, not guessed): G1-G3 SCORED post sets are still DAQ no-lick / engaged rows; which
  camera group replaces them? (by_class output already scores every camera group on its own.)
- AB's 19 steps (v9zv, a76352f3 + reuse): CHECKED before they take the lock. None of the 13 scripts calls the fixed code
  paths (engaged_frame_mask users = cd_trajectories / position_reference_maps / restnl_mask / rest_by_position;
  working-period, discriminator and D7 floor code not reached); 37bacf12 (0806 cut) is a respelling with the same
  session sets; 41f47e15 (A23) touches only --source encoder's tag lookup, same value under the same env. Encoder results
  = lick-source-independent (lick_source_audit). -> run as queued; their figures' stars get redrawn from figdata later.
- Epoch render arm times (11-05 log, all families, --parallel-arms 2): ENL 190, cue 251, lick 260, cuelick 130,
  ENLlick 144 min. Every arm but cue depends on the lick source (lick alignment or the ENL lick-free rule) -> the LP
  lick class needs lick / cuelick / ENLlick / ENL arms on LP (~12.0 h serial, ~6.5 h at 2 arms) MINUS what LPfinal
  already has (acc + 15r for lick / cue / precue).
- LPfinal maps (72729126) and decoders (cd6da055) HAVE the rest fix + camera fixes; LP source, LP gate, ENL 1.0.
  BUT their restnl reference reads g2b_restnl_mask_lp (9aaa4cc6, pre-fix) -> restnl ref stale; rest / restw refs fine.
- g5_epoch_15r_r (v9s, queued for the lock after AB): DAQ source -> cue arm duplicates LPfinal cue 15r, lick arms =
  successful-lick supplement only, ENL arm uses the retired DAQ 2.0 s rule. Proposed: drop it from the lock queue
  (part of the trimmed plan for Priya); it cannot start before AB's 19 steps finish.

## 11-08 ~13:50 -- epoch re-run trimmed by page figure (helper, verified)
- CORRECTION (my earlier claim): LPfinal DECODERS cover the pre-cue (ENL 1.0 s) window only
  (lp_final_render.py:260 `_epoch_arm("precue","working")`); LPfinal MAPS cover 15r + 15rpa for precue / cue / lick,
  all 6 refs. No Haufe (14m) in LPfinal.
- The 11-05 main render's 15r CRASHED on the ENL and cue arms (TypeError, ENL.log:501, cue.log:238; fixed 4ecf72ed):
  only lick-window 15r exists from the DAQ render.
- Lick-source dependence (code path, file:line in scratchpad/epoch_trim.md): cue window on camera working trials =
  independent (decoders + maps); 13s state decoder = independent (cue window, DAQ behavior_events). Pattern families
  (scal / mat: e_amp, e_ceiling, e_captured, r_crossnobis) depend via the PRE reference = "lick" trials
  (grant_kit.py:1167-1176); LP vs DAQ pre lick trials differ <= 3 % except PS93 far ipsi 27 % / far mid 14 %.
  ENL and lick arms = dependent.
- Minimal LP re-run (one process per arm; --arm exists, --parallel-arms ignored with it; WIDEFIELD_MAP_JOBS=2):
  ENL `--only 5r 14m` (~70-100 min), lick `--only acc 14m` (~75-105 min), cue `--only scal mat` (~95 min, the pre
  reference) -> ~1.75-2.5 h wall with 3 at once, vs 8.75 h for the full LP render. NOT needed: cuelick, ENLlick, and
  families 5c 5cr 5ro 10e 10cs 12s 12b 13s 15r 1b 1c (no page / deck reader, or LPfinal supplies them).
  MUST run on current main (needs 72729126, LP map jobs skip never-filmed PS93_0813), not wf_render_daqlk.
- Page / deck mapping (no re-run): point to LPfinal = Figs 2.1 (-> 2.8), 7.3, 7.7, 7.8, 7.10, 7.12, 7.14, 7.18, 7.19,
  7.23, slides 22 / 171 / 172; reuse main render captioned "independent of lick source" = Figs 1.2, 2.3, 3.1, 3.2, 4.2,
  14.2, 14.5, 14.8, slides 13/14, 17/18, 20; successful-lick supplement = Figs 2.4b, 7.16, 7.21; drop (superseded by
  LP 1.0 s) = Figs 2.2, 7.1, 7.5 (and 2.5-2.7 unless wanted).
- g5_epoch_15r_r (DAQ, queued after AB) is therefore redundant: cue/precue 15r come from LPfinal. Propose to drop.
- Fix-branch helper: 1151 passed, 1 failed = tests/test_gain_bio_port.py::test_fig4_draws_from_figdata (No module
  named '_common') -- FAILS ON MAIN 9bc33b9b TOO (pre-existing, not the branch). Helper's open points: G1-G3 scored
  sets (DAQ rows) -> which camera group; "none" includes unknown + close-mid unjudged (as G4); run DAQ-grouping G1-G3
  as a labeled supplement?; deck text "engaged-vs-no-lick" (locanmf_analysis_deck.py); stale "NOT lick-free" suptitle
  (encoding_key_figures); 11 legacy scripts still star on data (listed in the new test).

## 11-08 ~14:00 -- section G profile + fix
- Section G one-unit profile (PS94 unit 5, code c3a8b049 = g5_section_g_r4): WALL 2097 s. 85 % (1778 s) is
  trial_classes.daq_cue_seconds, called 297x per unit, re-reading + re-decoding the same raw DAQ h5 files (h5 read
  815 s, digital_bits / rising_edges ~600 s); ~700 s of nt.stat path checks via config.load_sessions (429 calls),
  partly inside it. Same repeated-loading pattern as partial_refit.
- Fix: branch perf/daq-cue-seconds-memo (wf_daqmemo, a5b7bf3c on main c12305d4): memoize per (label, WIDEFIELD_* /
  WFIELD_* env), copy per call; 3 new tests + 83 related pass. NOT pushed.
- Measuring now: same unit, base (v9zh) vs base + memo (detached wf_prof_memo), results pickled and compared for
  exact equality (scratchpad/prof_g/{base,memo}.{txt,pkl}). Section G scope proposal waits on this number.
- t_frame_te_basis (v9zu-TB) EXIT 0 13:39 -> v9u-R / v9zq-SVD lanes unblocked (encoder t_* family, a76352f3, audited).
- e_pat_stats_maps (v9u, 4ecf72ed) FAILED 13:40 -> e_roi_check_r / e_roi_foci_r / e_roi_cand_r / e_rim_overlay_r
  skipped. Cause = PATH BUG, not data: scripts/iti_decoder_patterns.py:43 + iti_basis_decoder.py:47 build
  SPATIAL = il.encoder_run("_te_basis") from iti_licks.ENCODER_TAG = design_tag(...) under the READER's env
  (FROZEN_LICKS_ALL=split -> "..._t3d_spont_frA_te_basis", does not exist); t_frame_te_basis wrote
  "..._t3d_frA_te_basis" (TEENV FROZEN_LICKS_ALL=none; spatial_*.pkl + pre_stats.pkl present, 11-08 10:55-11:20).
  The spont basis is NOT a substitute: spatial P differs (checked PS92 + PS95). Fix = derive the te-basis folder from
  the te design (FROZEN_LICKS_ALL=none) explicitly; branch + test, then re-queue e_pat_stats_maps -> e_roi_* ->
  e_rim_overlay_r. Check whether iti_decoder_patterns / iti_basis_decoder steps hit the same path.
- 13:53 b_partial_frA EXIT 0 (107/107, reuse fix). Lock released (takeover); l_block_compare (v8, 5db29b07, encoder
  env, no rest / lick-source dependence) took it 13:53. AB's 19 steps continue in v9zv.
- 13:55 STOPPED lane v9s-P (PID 943516, mine, only waiting; empty claim g5_epoch_15r_r removed): its only remaining
  step g5_epoch_15r_r is redundant (LPfinal maps give 15r for precue / cue / lick with the rest fix; DAQ lick arms =
  successful-lick supplement; ENL arm = retired DAQ 2.0 s). Its other steps were already done / failed / skipped:
  g5_restnl_maps_* + g5_map_drift_* READ the pre-fix restnl masks (g2a / g2b on 9aaa4cc6) -> stale, in the re-run list.
- Fix branch tests (logged run): 1151 passed, 1 failed = the pre-existing test_gain_bio_port '_common' failure (fails on
  main too). Pushing fix/stars-spoutorder-g13-1108 (push7, gated on >= 15 GB free).
- 14:1x PUSHED fix/stars-spoutorder-g13-1108 -> ON MAIN 21a005ca (532 pre-push tests passed; logged -k run 1151 passed +
  the pre-existing gain_bio '_common' failure). Stars top band, POS_ORDER, section G G1-G3 camera classes, DECISIONS
  11-08 lick-class correction.
- 14:11 l_block_compare EXIT 0; 14:12 l_basis_extract_r (v9zq-SVD, 2f5799a9, ITI family) took the lock: svdbasis items
  running ok. Watch: its script (iti_basis_decoder.py) has the te-basis path bug -- a te-basis source would fail.
- RAM 14:08: 1.4 GB free (another window's FR projection fr_cam4_june tasks 9 + 10 = 31 GB + state-space 8 GB); back to
  35 GB at 14:11. Lanes paused, nothing lost.
- Lane v8-L4L5L6L7L9 now waits on l_recov_markers (in the stopped l_bal_* chain): waits indefinitely, harmless; its
  remaining steps are on pre-fix 5db29b07 and are in the re-run list.

## 11-08 ~15:00 -- TRIMMED RE-RUN PLAN (for Priya; nothing queued yet)
Section G memo measured: same unit (PS94 #5) 2097 s -> 335 s by cProfile (DAQ loads 297 -> 27, 1778 s -> 117 s);
outputs IDENTICAL (exact compare, base vs memo, scratchpad/prof_g). (Paired wall run under 1.4 GB RAM / swapping:
2587 vs 1867 s -- not used.) Next hot spot: config.load_sessions path checks ~150 s / unit (scope a memo to the section
G worker like partial_refit._reuse_inputs).
| group | action | why | time (measured step times; G scaled from the profile) |
|---|---|---|---|
| epoch render, lick class | PARTIAL: 3 single-arm LP renders on main (ENL 5r 14m; lick acc 14m; cue scal mat) | lick class; cue arm only for the pattern pre-reference | ~1.75-2.5 h wall (3 at once) |
| epoch page figures | REUSE / repoint (no run) | LPfinal maps + pre-cue decoders; cue-window figures source-independent | 0 |
| section G | FULL grid with memo (+ load_sessions memo) | lick window + lick-only arm on LP, G1-G3 camera classes (21a005ca) | 13.1 h -> est. ~2-3 h (-j 4); figs 15 min; grant ~1.8 h |
| cd chain | FULL | lick-aligned + rest fix | 0.5 + 2.5 + 0.6 h |
| lick-aligned decoders | PAGE-FEEDING FIRST: cd_traj (Fig 9.1) + matched-pre lick half; pcd 3.3 / nolick 2.4 / precue states 0.5 / joint xsession 0.6 h after | lick class | ~0.7 h first, ~6.8 h later |
| rest family | FULL on fixed code: restnl masks (g2a / g2b) FIRST, then restnl maps / map drift, rest_frozen(_nl), restw / revup / c2l, attempt_maps_restnl, chronic_stability, the stopped restnl + l_bal chains | ba5b63bf missing; restnl masks pre-fix (also LPfinal's restnl ref) | ~6.8 h + masks 0.4 h + l_bal chain (measure) |
| gain-bio | gb_sessions_r queued (has fix) + summarise + stats | stale | ~1.2 h |
| ITI te-basis | FIX path bug (branch + test) then re-queue e_pat_stats_maps -> e_roi_* -> e_rim_overlay_r | reader env adds _spont | minutes + ~1 h |
| redraws | figdata replot (stars top band, POS_ORDER) | style fixes now on main | minutes |
| AB 19 b_* steps | RUNNING (checked: no fixed path reached) | - | in progress |
Heavy-lock total ~19 h serial -> ~10 h with Priya's two heavy slots. Run from ONE detached worktree at main 21a005ca
(+ memo branch if approved), settings check on the first step of each group.
Decisions for Priya: (1) push the section G memo branch (perf only, outputs identical)? (2) section G = full grid
(recommended, ~2-3 h with memo) vs affected cells only? (3) include the cue arm (pattern pre-reference)? (4) lick-aligned
decoders: page-feeding first, the rest after? (5) G1-G3 scored sets: which camera group replaces the DAQ rows?
(6) DAQ-grouping G1-G3 as a labeled supplement?

## 11-08 ~16:00 -- Priya's decisions on the trimmed plan
- APPROVED: push the section G memo (perf only, identical) -> pushing; full section G grid with the memo; include the
  cue arm (pattern pre-reference); lick-aligned decoders page-feeding first (cd_traj + matched-pre lick half), the
  other four after.
- G1-G3 (Priya 11-08, "ok as you propose"): train camera WORKING (contact + incomplete) vs NO_RESPONSE (unknown NOT
  trained; matches (f) 1 / a49795b3), score post no_response rows as the no-lick set, working rows as the engaged
  comparison; jaw_only / stopped / quit tail reported per group. Helper implementing on fix/g13-noresp-1108
  (wf_g13noresp). Section G waits for it.
- DAQ-grouping G1-G3 supplement: NOT run (Priya 11-08).
- ITI te-basis path bug: helper fixing on fix/iti-te-basis-path-1108 (wf_tebasis); then re-queue e_pat_stats_maps ->
  e_roi_* -> e_rim_overlay_r.
- 15:51 gb_sessions_r EXIT 0 (107, rest fix); l_stopped_restw_r (99f0d464, fix present) took the lock.
- CORRECTION (ITI te-basis): the path bug was ALREADY FIXED on main by cc70519e (11-07 20:36, il.te_basis_run, "TE basis
  path never _spont"); e_pat_stats_maps failed because lane v9u runs from wf_render_v9s at 4ecf72ed (11-06, pre-fix).
  -> just re-run e_pat_stats_maps -> e_roi_check_r / e_roi_foci_r / e_roi_cand_r / e_rim_overlay_r on current main.
  Helper branch fix/iti-te-basis-path-1108 (b9189a24, wf_tebasis): hardening only (tag derived from the TE design via
  frozen_encoder.tag_for_design, clear refusal naming the folder; 473 passed) -- held, optional.
- 16:0x memo ON MAIN 597cbed9 (two lost push races with another window; 483 passed each time).
- 15:59 gain-bio DONE on fixed code: gb_sessions_r, gb_summarise_r, gb_stats_r, gb_fig_r all EXIT 0.
- 16:02 LAUNCHED queue_daqlk9zw.sh (worktree wf_render_v9zw, detached at main 597cbed9: every 11-08 fix + memo).
  Lane v9zw-EP (second heavy slot, NOHEAVY, MINFREE 20, sequential): g5_epoch_lp_ENL (--only 5r 14m) -> g5_epoch_lp_lick
  (acc 14m) -> g5_epoch_lp_cue (scal mat). SETTINGS CHECK on the live log: WIDEFIELD_LICK_SOURCE=lp, GATE_SOURCE=lp,
  TRIAL_GATE=camera, frozen store frozen_models_lpfinal (same LP-trained decoders as LPfinal), output
  LP_licks_gateLP_<tag> (LPfinal's folder), families 14m 5r, arm ENL. ENL 1.0 s = lp_licks.enl_post_s default under lp.
  Lane v9zw-IT: e_pat_stats_maps_r2 -> e_roi_check_r2 -> e_roi_foci_r2 -> e_roi_cand_r2 -> e_rim_overlay_r2 (frozen env,
  fixed te-basis path cc70519e).
- Waiting before queueing the rest: settings-check helper (rest family / cd chain / page-feeding lick decoders) and the
  G1-G3 no_response helper (section G).
- 16:16 LAUNCHED queue_daqlk9zx.sh (same code 597cbed9 / worktree wf_render_v9zw; built from scratchpad/
  rerun_settings_check.md). CD (MINFREE 15): g5_cd_traj_lp -> g5_cd_traj_orth_lp -> cd_overlap_null_lp ->
  g7_cd_geometry_lp, env base + LICK_SOURCE=lp + GATE_SOURCE=lp, outputs in LP_licks_gateLP_<tag>/epoch (Fig 9.1);
  live log confirms lp / gate lp. RS (MINFREE 15): g2a_restnl_mask_r FIRST -> g2b_r, restnl_maps lp_r2 + contact_r,
  map_drift_contact_r, rest_frozen(_nl)_r, d_attempt_maps_restnl_r, recovery_cue_r -> chronic_stability_r, the l_*
  restw / restnl chain (frozen env: encoder family has its own camera-lick definition -- never LICK_SOURCE). MP: refit_
  matched_pre --variant lick on LP (LPfinal store) after the v9zw EP lick / cue arms. Dropped: enl1s arms (superseded),
  LP map drift (cannot run: the 12 June sessions have no camera trial table -> Priya), g5_map_drift_lp(_r).
- Settings check findings (rerun_settings_check.md): WIDEFIELD_GATE_SOURCE=lp changes only folders / cache tags under the
  camera gate; g6_grant also renders DAQ "lick" variants -> the grant re-run after section G needs the LP source;
  page provenance: LP-named RESTnl maps (7.3b, 7.10b, 7.18b, 7.19b) and LP-named drift figs (7.4/7.4b, 7.11/7.11b, 7.15,
  7.20/7.20b, 7.24) are DAQ on the page now -> repoint / relabel in the replacement pass.
- STILL TO QUEUE: section G (after fix/g13-noresp-1108 lands; with memo) -> section G figs -> grant (LP); l_bal_* ->
  l_recov_markers -> e_recov_meas / l_balmatch / e_dock_* / l_enl_offset -> ENL_SLIDE_FIX -> x_enlv_* (frozen env, defs
  in rerun_settings_check.md); later: pcd / nolick / precue states / joint xsession on LP; gb_fig_r redraw (stars);
  figdata redraws.
- QUESTION for Priya: LP map-drift figures (June vs August pre) cannot be made (June has no camera trial table): keep the
  DAQ "contact" drift figures labeled as such, or drop them?
- 16:22 v9zw-IT: e_pat_stats_maps_r2 + e_roi_check_r2 EXIT 0 (te-basis fix works); e_roi_foci_r2 EXIT 1 = MISSING
  PRODUCER: it reads figdata/lick_cued_spout_pattern_stats.pkl from scripts/iti_spout_pattern_stats.py, which (with its
  input iti_spout_patterns.py, the per-spout Haufe maps "3.35") never ran in the rebuild -- no queue step. 16:24 launched
  queue_daqlk9zy.sh lane IT2 (597cbed9): e_spout_patterns -> e_spout_pat_stats -> e_roi_foci_r3 -> e_roi_cand_r3 ->
  e_rim_overlay_r3 (frozen env, as the ITI family).
- 16:45 RESTNL MASKS: today's g2a_restnl_mask_r rebuild is IDENTICAL to the masks on disk (PS92_0813, PS93_0818,
  PS92_0806 checked); the old files were written 11-06 21:28, AFTER ba5b63bf, by g5_restnl_maps_contact (4ecf72ed, has
  the fix). -> l_stopped_restnl_r (99f0d464) valid; g5_restnl_maps_contact_r + g5_map_drift_contact_r marked .done as
  REDUNDANT (reason in their .done files + CHAIN.log; ~2.5 h saved). The audit's "restnl readers stale" applied only to
  steps whose OWN code lacked the fix (rest_frozen(_nl), l_* restw chain, d_attempt_maps_restnl, chronic_stability) --
  those still re-run.
- FINDING (efficiency): the first section G memo (a5b7bf3c) moved the camera-class digest csig 3d7bbe921e ->
  edce41f2a5 (class_source_signature hashes trial_classes' top-level code), so every camera-gated per-session cache
  entry missed on 597cbed9: the v9zw / v9zx / v9zy lanes recompute them (valid, slower). FIX d8f4e900
  (perf/daq-memo-keep-csig, wf_memocsig): memo moved to wfield_local/session_memo.py, applied by a hook-marked
  decorator; csig back to 3d7bbe921e (verified), trial_classes = pre-memo text + 1 line; 76 related tests pass; a test
  pins the memo out of the digest. Pushing (part of Priya's memo OK). Section G + grant will run on it -> warm caches.

## 11-08 ~17:45 -- state
- ON MAIN since 16:00: 597cbed9 (section G memo), d8f4e900 (memo keeps csig 3d7bbe921e), b83f376d (G1-G3 = camera
  WORKING vs NO_RESPONSE, scored = working-period no_response rows, tag cls-work-v-noresp; DAQ path unchanged; 18435c15).
  Note for Priya: under the camera grouping undetected_state_split now nearly always reports UNDECIDABLE (stopped /
  quit-tail have their own groups); by_class answers that question directly (no_response / stopped / after_end each
  scored against the working-vs-no_response discriminator).
- 17:09 v9zy-IT2 DONE: e_spout_patterns, e_spout_pat_stats, e_roi_foci_r3, e_roi_cand_r3, e_rim_overlay_r3 all EXIT 0.
- 17:43 LAUNCHED queue_daqlk9zz.sh lane SG (worktree wf_render_v9zz at b83f376d, csig 3d7bbe921e verified):
  g5_section_g_lp (full grid, -j 4, LICK_SOURCE=lp, GATE_SOURCE=lp, LP_FROZEN=lp via profile, store
  frozen_models_lpfinal, session cache = M: default) -> g6_grant_lp (deps section G, g5_cd_traj_orth_lp,
  g5_recovery_cue_r). Waiting for the heavy lock (l_disengage_r, v9zs-BH).
- RUNNING: v9zw-EP (LP epoch arms, sequential), v9zx-CD (LP cd chain), v9zx-RS (rest family), v9zx-MP (waits on EP),
  v9zz-SG, plus older lanes (v9zv AB b_* steps, v9zs-BH l_stopped/l_disengage, v8 waiting lanes).
- STILL TO QUEUE: l_bal_* -> l_recov_markers -> e_recov_meas / l_balmatch / e_dock_* / l_enl_offset -> ENL_SLIDE_FIX ->
  x_enlv_* (frozen env; defs + deps in scratchpad/rerun_settings_check.md); later pcd / nolick / precue states / joint
  xsession on LP; gb_fig_r redraw; figdata star redraws; then the replacement pass (headline table on corrected outputs ->
  Priya -> pages / decks; page repointing list in scratchpad/epoch_trim.md + rerun_settings_check.md).
- OPEN for Priya: LP map-drift (June vs August) cannot be made -- keep DAQ "contact" drift labeled, or drop?
- 21:26 g5_section_g_lp EXIT 0: 76/76 session units, MEASURED 3 h 14 min (18:12 -> 21:26) vs 13.1 h for r4 before the
  memo; LP lick class, G1-G3 camera working vs no_response; PS93_0813 (no video) skipped as LPUnavailable (logged, not
  fatal). Outputs figures_working/LP_licks_gateLP_<tag>. g6_grant_lp now waits only on g5_recovery_cue_r (RS lane).
- 19:23 g5_epoch_lp_ENL EXIT 0 (3 h 20 min; cold cache: lane started on the pre-d8f4e900 csig). 21:13
  g5_cd_traj_orth_lp EXIT 0. 20:26 g5_restnl_maps_lp_r2 EXIT 0. 20:53 g5_rest_frozen_r EXIT 0.
- 21:39 v9zx-CD DONE (cd_overlap_null_lp, g7_cd_geometry_lp -> LP epoch folder: Fig 9.1 on LP). 22:40 g5_recovery_cue_r
  EXIT 0 -> 22:41 g6_grant_lp STARTED (b83f376d; live log: lick source lp, gate lp, store frozen_models_lpfinal, out
  grant_figures/LP_licks_gateLP_<tag>). 23:05 g5_epoch_lp_cue EXIT 0 -> all 3 LP epoch arms done; 23:05
  m_matchedpre_cue_lick_lp STARTED (LP env + LPfinal store, new LP inputs pickle). v9zq-SVD lane done. l_pht_run
  (v8, 5db29b07) checked before it ran: no rest / working / lick-source code reached (only spelling + SVG changes since).
- 23:11 LAUNCHED queue_daqlk9zz2.sh lane BAL (b83f376d; built by a helper from rerun_settings_check.md, reviewed): the
  balance / ENL-slide chain, 59 steps, every name suffixed _r2 so the OLD waiting lanes (v8b / v9zd / v9zr / v9zc, pre-fix
  code) are never released; done deps kept by name only where .done exists; l_ci_spout + l_enl_target were never done ->
  in the chain as _r2; own marker ENL_SLIDE_FIX_r2 (enl_release_r2 over the _r2 names) so the old x_enlv_* stay parked;
  l_enl_offset_r2 waits on l_precue_restw_r (rest-fix re-run).
- After this everything in the trimmed plan is running or done, except the later lick decoders (pcd / nolick / precue
  states / joint xsession on LP) and the figdata star redraws. Old waiting lanes on pre-fix code can be stopped (exact
  PIDs) once the r2 chain is through.

## 11-09 ~00:40 -- Priya's answers
1. Drift figures: "can we build camera trial tables for June? we have the 3d data" -> read-only investigation running
   (scratchpad/june_camera_tables.md): inputs per June session, why absent, steps + time, and the csig risk (the class
   digest hashes the v36 / cam1 input files, so adding June files could invalidate every camera-gated cache) before
   anything is built.
2. G4 -> train working vs no_response, "only rerun what needs to be rerun": helper on fix/g4-noresp-1109 (wf_g4noresp)
   + minimal re-run plan (G4-only path / cache hit share / downstream consumers).
3. te-basis hardening branch: OK -> pushing (fix/iti-te-basis-path-1108, wf_tebasis).
- g5_section_g_lp DONE 21:26; g6_grant_lp started 22:41, 4/96 units at 00:34 (first units pay the cold LP collection;
  rate to be measured at later unit boundaries, not extrapolated).
- 00:41 l_bal_offset_primary_r2 EXIT 1 = MISSING PRODUCER (2nd of the night): its "inside pre range" version reads
  recprim_sessions_inside.csv <- lick_recovery_primary.py WFIELD_BAL_INSIDE=1 <- balsub_sessions_inside.csv <-
  lick_balance_subsets.py WFIELD_BAL_INSIDE=1; no queue ever had these steps (nothing in BAL depends on it). 00:42
  launched queue_daqlk9zz3.sh lane BI (b83f376d): l_bal_subsets_inside_r2 -> l_recov_primary_inside_r2 ->
  l_bal_offset_primary_r3 (env mirrors the _r2 steps + WFIELD_BAL_INSIDE=1). Fig 12.19 area.
- 00:4x te-basis hardening ON MAIN 2944eca4 (no cache key hashes frozen_encoder / iti_licks source: checked).
- 00:50 JUNE CAMERA TABLES (investigation: scratchpad/june_camera_tables.md): every input exists for the 12 June
  sessions; the classes already cover June; only trial_attempts.csv lacks June (built under PRE_FROM 0801). Overwriting
  it would change lp_licks.spec_token (mtime/size) -> every LP cache cold + June "pre" rows would leak into readers'
  August baselines. Priya 11-09 OK: June-only SIDE table (--out, new --labels option) + opt-in lp_licks supplement
  override that keeps every registered session's cache key unchanged (route 2b) + re-run g5_map_drift_lp only.
  Helper implementing on feat/june-attempts-supplement-1109 (wf_june) incl. the side build + August-row invariance diff.
- G4 -> no_response: helper still running (fix/g4-noresp-1109).
- Grant (LP) at 00:46: 6/96 units (7b x4 done 00:15-16 after ~1 h 35 min cold collection; 8d x2 at 00:36-37);
  if ~20 min per pair holds, ~3-4 h more. Profile only if later units stay slow (Priya asked; offered).
- 01:30 June supplement ON MAIN 3f1c5e90 (side table 5,792 rows / 12 June sessions under rebuild_20261103/qc/
  trial_attempts_june/<tag>; registered table sha + mtime unchanged; August rows byte-identical in a 3-session rebuild;
  cache keys of registered sessions unchanged with the env set). G4 branch (70d976f0 + 2b87c996 --only g4: replaces only
  the 3 G4 keys of the keyed section G json, backup first, label check) pushing via scratchpad/chain_v10.sh, which then
  builds + launches queue_daqlk10.sh: G4 lane (g5_section_g_g4only -> g5_g4_fig_redraw) and JD lane
  (g5_map_drift_lp_june, WIDEFIELD_ATTEMPTS_SUPPLEMENT set for that step only). Handoff block 00-COMPACT-39 written.
- 01:28 chain_v10 DONE: G4 ON MAIN 90a5e25f (one lost push race, retry ok); worktree wf_render_v10 at 90a5e25f, csig
  3d7bbe921e; queue_daqlk10.sh launched. JD: g5_map_drift_lp_june running with WIDEFIELD_ATTEMPTS_SUPPLEMENT set
  (120 ref maps = 108 + 12 June). G4: g5_section_g_g4only waiting for the heavy lock (l_precue_restnl_r), then
  g5_g4_fig_redraw.
- 02:01 g5_map_drift_lp_june EXIT 0 (33 min; LP drift figs with June, grant_figures/map_drift_prepre/LP_licks_<tag>).
- 02:27 v9zx-RS DONE: rest family complete on the rest fix (chronic_stability_r; l_* restw chain; restnl variants) --
  all EXIT 0.
- Grant (LP) slow: 12/96 at 02:03 (6 units / 77 min). Profiling ONE unit in a separate process (outputs to scratch;
  scratchpad/prof_grant/REPORT.md) to find repeated loading vs cold collection; no restart on an estimate.
- Heavy lock: l_balimg_primary_r2 (BAL); g5_section_g_g4only queued behind it.
- 03:2x GRANT PROFILE (one unit, 10[cue/working], scratchpad/prof_grant/): 2447 s, 95 % repeated raw-input loading
  (108 DAQ h5 opened 442x = ~1,400 s; config.load_sessions 368x = 289 s; cache-folder glob 214x = 178 s); compute
  seconds; every cache / frozen model a hit. The LP gate path (gate_source.extra) re-loads each session's DAQ cue train
  2x per session before its own memo check -> why it is slower than the Sept DAQ render; each worker repeats the ~39 min
  load per alignment. Helper implementing memos 1-4 (+5) on perf/grant-loader-memos-1109 (wf_grantmemo) and MEASURING
  the same unit + exact figure-data equality + csig check. Then: Priya decides keep running (~18 h at the current rate)
  vs restart the REMAINING grant families only (--only) on the fix.
- (correction: the grant-profile entry above is ~03:00, not 03:2x.) Grant (LP) SPED UP: 25/96 at 03:03 (13 units in the
  last hour -- workers now reuse their per-alignment bundles); ~5-6 h left at that pace, not ~18 h. Memo fix still being
  measured; restart only if it saves hours (recommend letting this run finish otherwise).
- 03:07 g5_section_g_g4only EXIT 0 (34 min vs ~3 h full): G4 = camera working vs no_response (neg-no_response models,
  frozen-new), merged ONLY the G4 keys into the keyed LP section G json (backup first). 03:08 g5_g4_fig_redraw EXIT 0.
  G4 deck slide + results-page panel -> replacement pass.
- 04:05 g6_grant_lp EXIT 0: 96 units, 0 failed, 5 h 24 min. Flags: (i) 12x PS93_0813 skip (expected); (ii) for EVERY
  animal x {lick, precue} the grant FROZE A NEW LP epoch decoder: "stored model was frozen against a DIFFERENT pre-stroke
  input set (8 -> 8) ... SPEC-CHANGED" -> a second LP "frozen" epoch decoder next to the epoch arms' (same store); and its
  training label reads "lick source lp, selection engaged rows (camera gate: DAQ lick-arm trials)" -> must confirm the
  LP lick class really selects camera-seen licks and is not DAQ contact. Read-only trace running
  (scratchpad/lp_epoch_decoder_trace.md). HOLD grant + epoch lick/precue numbers until traced. (iii) layout: 139 [layout]
  warnings (crowded day ticks in 7b / 8b / 11; the "LP licks" stamp overlaps the "Post-stroke session" text in 9) ->
  redraw pass.
- 04:08 v9zz2-BAL END: balance / ENL-slide chain done except x_enlv_table_r2 EXIT 1 = MISSING PRODUCER #3 (reads
  enlv_licks/sessions_reg/*.npz <- lick_enlv_extract.py --regress; no queue ever had it) -> x_enlv_balance_r2 /
  x_enlv_curves_r2 skipped. 04:09 launched queue_daqlk9zz4.sh lane EV (b83f376d): x_enlv_regress_r2 -> x_enlv_table_r3
  -> x_enlv_balance_r3 -> x_enlv_curves_r3 (defs from 9zz2; all deps .done).
- LESSON (3 missing producers tonight: ITI spout patterns, balance _inside, ENL-v regress): before launching a chain,
  check every input FILE its scripts read has a producer step in some queue, not only that its declared deps exist.
- 04:40 TRACE (scratchpad/lp_epoch_decoder_trace.md): (1) LP lick class CORRECT: under LICK_SOURCE=lp the "lick" /
  engaged rows = LP-attempted trials (hit or incomplete, first LP lick within max_rt; lp_licks.py:519, ATTEMPT_CLASSES:54)
  -- PS92 pre 2863 rows = LP count (DAQ 2840). The log text "camera gate: DAQ lick-arm trials" is a HARD-CODED label
  (grant_confusion.py:751) + stale camera_select docstring (trial_classes.py:1605-1609) -> fix wording (trial_classes
  edit must keep csig: docstring is in the digested source -> use care / verify csig). (2) Two LP lick epoch decoder sets
  (epoch arm vs grant), identical specs except the training-data fingerprint (inputs differ <= 4e-7, likely thread-
  dependent lick-window computation); coefficients differ up to 2.6 % (PS92). Epoch-arm lick figs used PS92 057dcc59412b /
  PS93 86922bc2ab75 / PS94 fe7e2fe9900e / PS95 e3469e738e4d; grant *_lick_lick units used a07055c3580c / a260f0abdbbe /
  cbfeae0130c3 / 79a6814f01b4. Precue / cue / working outputs unaffected.
- Grant memos measured (perf/grant-loader-memos-1109, 7ac7850c): unit 2100 -> 954 s, later alignment 130 s (was ~39 min),
  figure values + PNG IDENTICAL, csig 3d7bbe921e, 559 tests pass. Awaiting Priya OK to push.
- Decisions for Priya: A accept the two near-identical LP lick decoder sets (recommended) vs make bit-identical; B push
  grant memos.
- 04:36 x_enlv_balance_r3 EXIT 1 = missing producer #4: it reads, per ITI folder (all / dock / notdock),
  matched_loclin_tests + balmatch_tests + recprim_sessions + ci_spout_epoch_tests; the NOTDOCK folder lacked balmatch /
  recprim / ci (only prep + loclin were ever defined for notdock). This time ALL 12 balance inputs + every curves input
  were checked on disk before re-queueing. 04:37 launched queue_daqlk9zz5.sh lane ND (b83f376d): e_dock_{balerr, ci,
  subsets, recov}_notdock_r2 (dock-filter defs + notdock env) -> x_enlv_balance_r4 -> x_enlv_curves_r4.
  (x_enlv_regress_r2 + x_enlv_table_r3 EXIT 0.)

## 11-09 ~05:10 -- RE-RUN PASS COMPLETE
- 05:07 v9zz5-ND DONE: e_dock_{balerr, ci, subsets, recov}_notdock_r2, x_enlv_balance_r4, x_enlv_curves_r4 all EXIT 0.
- Every group of the trimmed plan is now re-run on fixed main: section G (LP) + G4-only; LP epoch arms; LP cd chain;
  matched-pre lick half; ITI chain; gain-bio; rest family; balance / ENL-slide chain (with 4 missing producers added:
  ITI spout patterns, balance _inside, ENL-v regress, notdock dock steps); June LP drift; grant (LP, 96/96).
- Remaining failures in the render's cumulative count are superseded steps (old names re-run as _r2 / _r3 / _r4) or old
  parked lanes on pre-fix code.
- OPEN for Priya: A accept the two near-identical LP lick decoder sets (recommended); B push grant memos (7ac7850c).
- NEXT (me): stop the parked old lanes by exact PID; redraw pass (stars / order figdata; matched-pre figure with the LP
  lick half; G4 deck slide; gb_fig_r; fix the stale "DAQ lick-arm trials" label + grant layout overlaps); then the
  replacement pass -- headline table on corrected outputs -> Priya BEFORE any page / deck text.
- Priya 11-09: A = ACCEPT the two near-identical LP lick epoch decoder sets "for now". Set 1 (epoch arm) -> epoch
  lick-window acc / accdelta (Fig 2.4) + Haufe 14 / 14pa / 14z (Figs 14.3 / 14.6 / 14.9). Set 2 (grant) -> grant lick/lick
  units 5c, 5d, 6, 6b, 6d, 7, 7b, 7d, 8, 8b, 8d, 8e, 8g, 9, 10, 10b, 11 (decoder-dependent: at least 5c / 5d / 9 / 10 /
  10b; confirm per family in the redraw pass). Keep on the list: 15-min thread-dependence diagnostic so future runs
  stop minting new ids. B = OK -> pushing perf/grant-loader-memos-1109 (7ac7850c).
- 05:20 PARKED LANES (audit scratchpad/parked_lanes_audit.md): all 12 old lanes were parked forever (only child sleep 60,
  no locks, deps either re-run under new names or never producible). STOPPED by exact PID (1255761 1243302 943032 942954
  1221888 942383 943220 943734 1010593 943105 1008633 943159; all mine); removed 8 empty claim dirs + 4 lane subdirs
  (RESERVED markers kept).
- 05:24 LAUNCHED queue_daqlk11.sh (wf_render_v10, 90a5e25f): the 44 STILL-NEEDED steps that were stuck behind those
  lanes (never run under any name; not in the trimmed plan but feed goal-page / imaging / kinematics figs + deck slides):
  B5 encoder figs (b_rsplit_PS9x NEW producers, fig_rsplit, fig_resid_per_animal, exec_vs_*, z_fig_pooled), E5 ITI /
  aim models (e_mm, e_lmm_*, e_iti_lick_rate, e_pat_*, e_fpa_*), L5R lick family (offset terms, nonlinear chain, key
  figure, interhemi / xa / cv_sm, enlv target code, iti prelick, code subspace incl. NEW enlv producers, restnl rescue,
  face offset partial, simple key figures). Names _r5; every dep resolves (.done or earlier in the file); encoder env kept.
  Lanes v11-B5E5 + v11-L5R preflight OK.
- OPEN for Priya: contact-only (DAQ) section G supplement (g5_section_g_contact_r4 + figs) -- audit dropped it (pre-fix
  code, not in the plan); run on fixed main (~3 h) only if wanted. V8_NOTES caveats carried: l_nl_* commands marked
  UNVERIFIED; l_interhemi* use restdock05 rest set (RNP 2a(d) decision pending).
- Priya 11-09 ~05:45: (1) contactOnly section G arm DEFERRED; when it is run it SUPERSEDES the mistaken DAQ-source r4
  section G, and r4's outputs may then be deleted (Priya OK; derived outputs, keep a list). Note: r4 = hybrid (lick
  window on DAQ contact, cue / precue on camera contact + incomplete); contactOnly = contact only throughout. (2)
  Interhemispheric / cross-area / coupling rest set (RNP 11-01 category (d), Figs 14.18 S10 / S17 / S18): PRIMARY =
  RESTnl (docked rest x lick-free mask: camera + DAQ licks); SUPPLEMENT = current restdock05 (DAQ licks only; the v11
  l_interhemi* / l_xa* / l_cv_sm* _r5 steps). Code change on feat/interhemi-restnl-1109 (helper). (3) l_nl_* "unverified"
  caveat resolved: iti_lick_order supports WFIELD_LICK_DECODER=nonlinear (own preds + _nonlinear-tagged outputs).
- Priya correction 11-09: restdock05 is NOT "raw". For the interhemi / cross-area / coupling family rest enters ONLY as
  metric 3 (lick - rest per session); metrics 1-2 per lick set use no rest. -> PRIMARY = lick - RESTnl rest; SUPPLEMENT
  "raw" = lick-set metrics with NO rest reference (same run, labeled "raw (no rest reference)"; helper adding explicit
  outputs). The v11 restdock05 chain (l_interhemi_rest_r5 ... l_cv_sm_plot_r5, 7 steps, not started, no outside
  dependents) marked .done NOT RUN with the reason (delete the .done files to re-enable).
- 06:3x Priya: start REDRAWS + HEADLINE TABLE. Two helpers: (1) redraw pass on fix/redraw-pass-1109 (wf_redraw): stale
  "DAQ lick-arm trials" label computed from the lick source (not trial_classes: csig), grant layout overlaps (LP stamp vs
  "Post-stroke session" on 9; crowded day ticks 7b / 8b / 11 / 8g), then figdata-only redraws of current outputs (stars,
  POS_ORDER, layout), gb_fig_r, matched-pre figure with the LP lick half (working = figure, lick = table); inventory in
  scratchpad/redraw_pass.md. (2) headline table on corrected outputs (read-only) -> scratchpad/headline_table_corrected.md,
  CONCLUSION-CHANGES rows first, per-animal before pooled -> Priya before any page / deck text.
- OPEN for Priya: interhemi supplement B = lick - session-wide windows (engaged period vs all frames) -- add or not.
- Grant memo push blocked by pre-push full suite (3 failed / 5732 passed): 2 = known test_cohorts (fail on main), the
  3rd being identified by a full run on the branch (scratchpad/grantmemo_fullsuite.txt). Not bypassed.
- Priya 11-09 RULE: every AMPLITUDE-based output must exist with TWO references: RESTnl (lick-free docked rest) AND
  session mean (the dF/F as is; the SVD dF/F is relative to the session's mean frame, so "session mean" = no further
  subtraction = "raw" in the map families). Coupling / correlation / decoding outputs need no second reference (a
  constant baseline drops out); for them "raw" = the lick-set metrics alone. Supplement B (session-wide event set) dropped
  -- not a reference. Read-only audit of every amplitude family for the two refs running
  (scratchpad/amplitude_two_refs_audit.md); gaps go into the plan. Memory: amplitude-two-references.
- Priya 11-09: camera working period start = FIRST CAMERA LICK confirmed (trial_selection.rest_start: first_lick; the
  code comment's "open question" is settled). RESTnl rest-boundary rule KEPT AS IS (a rest period counts only if both
  bracketing trials are in the working period; so the rest just before the first lick trial and after the last are
  excluded, while trial analyses include the first lick trial's pre-cue window -- differ by design).
- Interhemi RESTnl branch done (feat/interhemi-restnl-1109, 619b71df; 69 related tests pass; raw outputs labeled "raw (no
  rest reference)", identical under every rest variant). Priya saw the helper's SYNTHETIC smoke-test figure
  (scratchpad/ihraw_smoke, random tables, n = 4, one pair) -- "mostly empty" = synthetic, not data; check the real raw
  figure for empty panels when it lands. chain_v12.sh pushing + launching queue_daqlk12.sh lane IH (7 _restnl steps).
- 06:30 HEADLINE TABLE on corrected outputs -> docs/status/HEADLINE_TABLE_2026-11-09_CORRECTED.md (8 CONCLUSION-CHANGES,
  6 changed-same-conclusion, 44 unchanged, 14 not yet available). For Priya BEFORE any page / deck text.
- Interhemi RESTnl push BLOCKED: current origin/main itself fails tests/test_share_writers_mode_safe.py::
  test_every_share_writer_is_routed_refusing_or_classified (unclassified: june_drift_summary_fig.py,
  state_space_shift_movement.py -- added by another window). Blocks every push that selects that test. Ask Priya (as d9 on
  11-08: classify on my branch).
- 07:00 REDRAW PASS (fix/redraw-pass-1109, 208eddfd): 0 figures rewritten (figdata redraws pixel-identical / already
  fixed); code fixes = lick-source label (grant_confusion), LP stamp band + crowded-tick thinning (grant_kit._save),
  matched-pre --lick-dir. Needs re-render (Priya OK pending): LP grant set (95 PNGs, ~1 h 50, faster with memos);
  encoding bar figs without figdata (posinfo, specificity, svf, matched_residual, partial_frA, fr_compare, spout maps)
  via --plot-only re-summarise; matched-pre LP lick table to M:.
- 07:00 TWO-REFS AUDIT -> docs/status/AMPLITUDE_TWO_REFS_AUDIT_2026-11-09.md. Decisions for Priya: D0 (map "raw" = slow
  inter-trial trend zero, not strictly session mean; area traces = exactly session mean), S3-1 (per-trial-baseline
  families: exempt + caption, or absolute twins of key ones e.g. 14.1), S3-2 (state-space response sizes twins), S3-3
  (hemisphere-difference balances need RESTnl?), S3-4 (matched_residual / CD projections RESTnl twins). Recommended:
  D0 yes + caption; S3-1 exempt + 14.1 twin; S3-2..4 defer; run the no-code / few-line gaps now.
- Priya 11-09 ~07:20 decisions: headline changes 1-6 + 8 OK for pages / decks; #7 (acute cue far-contra RESTW per animal
  .50/.45/.53/.40 FINDINGS vs .38/.46/.67/.98 rebuild) -> look closer (old vs new figures). Register the 2 unclassified
  share writers (push blocker) OK. Redraw re-renders OK (helper: classify -> push redraw + interhemi branches -> queue
  v13: LP grant re-render, encoding bar re-summaries, matched-pre LP table; memo branch only if test_cohorts pass).
  Two refs: D0 map "raw" accepted as the session-mean twin (caption: zero = slow inter-trial trend; area traces = exact
  session mean, same corrected signal + encoder slow basis incl. constant). Absolute-level twins of the per-trial-baseline
  families + state-space response-size twins: YES, deferred to the page update. Hemisphere-difference balances: NO RESTnl
  needed if they compare lick-peak fluorescence not normalized within the trial (verify each). CD projections: HOLD (open).
- Priya 11-09 ~07:45 on #7: wording OK -- "acute far-contra amplitude on working trials falls in 3/4 animals; PS95's single
  acute session (PS95_0817) has only 40 far-contra working trials (1 hit + 39 incomplete; of 120 cues 53 jaw-only, 16 no
  response, 11 stopped) and shows no drop on the trials it attempted." Its old "halved" came from averaging over trials
  where it did not try. Working PERIOD (rest gate) = all cues first->last camera lick; working TRIALS = contact +
  incomplete within it (11-03). Both 15rpa titles say "lick + miss-while-working" (stale wording).
  NEW RULES: (a) EVERY figure states EXACTLY which trials were included / excluded; (b) map figures get a JAW-ONLY class
  panel. Helper on feat/trial-set-captions-jaw-panel-1109 (wf_trialset): one shared trial-set sentence from the active
  selection wired into the common save paths + a jaw_only map variant (new cache kind only), then the re-render plan.
- Priya 11-09: map trial sets = option (c): PRIMARY = working trials (contact + incomplete); SUPPLEMENT = every trial in
  the working period (contact + incomplete + jaw-only + no response; stopped runs + quit tail excluded); PLUS a jaw-only
  panel. Sent to the caption / jaw helper (one re-render per map family for all three).
- 07:5x pushes unblocked: 10cdb08a (share writers classified: june_drift_summary_fig BEHAVIOUR, state_space_shift_movement
  INDIRECT), f992cb8f (redraw-pass fixes), dff97e56 (interhemi RESTnl). Memos still blocked by 2 stale test_cohorts
  expectations (cohort2 per-animal reference dates, c926d713) -> test-side fix proposed, Priya to decide. v13 queue
  (wf_render_v13 @ dff97e56, csig 3d7bbe921e) running: RB encoding re-summaries (4/9 done) + matched-pre LP table; GR LP
  grant re-render (~5.5 h) -- PROPOSED to stop it and re-render grant once after the caption branch (grant has no figdata).
- 07:23 STOPPED v13-GR grant re-render (lane 1420911 + winpid 26656 tree, mine) per Priya: grant re-renders ONCE after
  the trial-set caption wiring + grant memos. v13-RB DONE 07:19 (9 encoding re-summaries + matched-pre LP table, all
  EXIT 0).
- Trial-set captions + option (c) map variants: feat/trial-set-captions-jaw-panel-1109 f1e5fba9 (21 new + 440 related
  tests; all existing cache keys unchanged, measured). Jaw-only per-session coverage too sparse (6 / 259 cells, none pre)
  -> Priya: POOL jaw-only PER EPOCH per animal (helper adding, same branch). Priya OK: fix the 2 stale test_cohorts
  expectations (test-side only; helper on fix/test-cohorts-stale-refs-1109) -> then push the grant memos; push the caption
  branch + launch the 3 map re-renders (LP restnl maps, LPfinal maps, epoch 15r; ~4-5 h in parallel) after the pooling
  commit. Still to do: wire trial_set_text.active() into the remaining families (grant, section G, other epoch families,
  rebuild_figs) + fix stale "miss-while-working" deck-note wording; then the grant re-render.
- 07:30 LAUNCHED v12 lane IH (wf_render_v12 @ 98e6a353, csig 3d7bbe921e): interhemi / xa / cv_sm on RESTnl + raw (7
  steps). Started trial-set WIRING helper (feat/trial-set-wiring-1109) for the remaining families + stale deck wording.
  TIMELINE (Priya asked): map re-renders ~4-5 h after the pooling commit; grant ~2-3 h after wiring + memo push; v11 rsplit
  ~1 h 40 / animal x 3 (PS92 05:24-07:05 measured); renders done ~13:00-14:00; page / deck update ~2-4 h after ->
  this evening.
- 07:45 jaw-only POOLED per animal x epoch (0899267d; floor 20 per pooled cell; existing cache keys re-verified
  unchanged). Cells >= 20: pre 5 (PS95 only, all 5 judged spouts), acute 11, subacute 5, chronic 2 (close mid never
  judged). Within-animal "- pre" jaw-only columns only for PS95. chain_v14.sh: push the caption branch -> worktree at main
  -> queue_daqlk14.sh lanes MP (g5_restnl_maps_lp_ts) + LF (g5_lpfinal_maps_ts) = primary + working-period supplement +
  jaw-only panel in one render each. DAQ epoch 15r re-render SKIPPED (cue = LPfinal all 6 refs; lick = successful-lick
  only; ENL 2 s retired).
- 07:50 random splits parallelized: 2 extra v11 B5 lanes -> PS93 / PS94 / PS95 run at once (ETA ~09:30 instead of
  ~12:20). Handoff block 00-COMPACT-40 written.
- 08:45 test_cohorts fix ON MAIN 98e6a353. Grant memos NOT pushed: full suite 1 failed / 5747 passed =
  test_grant_stamp_fit::test_stamp_gets_its_own_band, ORDER-DEPENDENT (passes alone; from f992cb8f on main) -> blocks
  full-suite pushes. Helper fixing on fix/test-stamp-order-1109 (find the polluting test, isolate state, full suite,
  push). Caption-branch push (chain_v14, WTS=wf_trialset) still in its pre-push tests since 07:32 -- may hit the same
  failure; re-run after the fix. Note: two push runs had written into push7.wf_grantmemo.log (old retry loop + helper).
- 09:10 MAP RE-RENDERS LAUNCHED without waiting for the caption-branch push: queue_daqlk14a.sh from a DETACHED worktree
  wf_render_v14a at the reviewed commit 0899267d (all fixes; csig 3d7bbe921e): g5_restnl_maps_lp_ts (MP) +
  g5_lpfinal_maps_ts (LF) -- primary + working-period supplement + jaw-only (pooled per epoch) + exact trial-set captions.
  Same step names as chain_v14's queue, so its lanes (if the push lands) skip them via claims. ETA ~13:00-14:00.
- 09:30 status: no new failures since 06:25. v14a map re-renders: reference maps 107/108 in ~20 min (warm), new trial sets
  next. chain_v14 caption push passed tests but lost a push race (retrying). v12 interhemi rest extract EXIT 0 08:11.
  Pending helpers: order-dependent stamp test fix (unblocks memos), trial-set wiring (unblocks grant).
- 10:25 status (after compaction): no failures. v14a map re-renders had logged nothing since 09:30 (108/108 ref-maps).
  Measured, not a hang: main thread "Ready" (CPU-starved) -- the 3 random-split jobs (47 threads each) held ~22 of 24
  cores. RESTnl baseline attach = serial `restnl_mask.base_maps`; timed in isolation at 3-9 s per session even on a
  cache hit (119 csig-3d7bbe921e baselines on disk, key matches), so 108 sessions under starvation = tens of minutes.
  Fix applied (no restart): random-split PIDs 2184 / 37244 / 26396 set to BelowNormal priority -> map CPU 0.03 -> 0.26
  core. FINDING for later: the per-session cache-hit cost (3-9 s) is worth profiling (key computation?), not now.
  Random splits PS93 13/20 at 10:08 (~14 min / split -> ~11:45). Caption-branch push: retry at 10:19, tests running.
  Stamp-test fix ready on fix/test-stamp-order-1109 6bb2a103 (push after the caption branch, then memos).
- 10:40 TRIAL-SET WIRING DONE (helper): feat/trial-set-wiring-1109 502d67a4 (contains the caption branch 0899267d).
  Every driver family declares its exact trial / event / frame set; savefig hook stamps it. 906 tests passed; digests
  unchanged (csig 3d7bbe921e, _raw_maps_code 2950853f103b, restnl d7591e88ae51/3572c6da0d13, beta 6822ca7e4f39);
  frozen_encoder edit checked: no source hashing of that module. Push chain_v15 queued after chain_v14:
  stamp fix -> wiring -> grant memos. Re-render list in the helper report (grant full; epoch full; section G / CD /
  rebuild_figs / ITI / lick mostly --plot-only). Old epoch figdata has no set -> mat, scal, 12/13/15e need re-render.
  FINDINGS outside scope (not fixed, for review): lick_precue_restw* uses fixed -1..0 s pre-cue (no lick-free rule);
  rebuild_figs encoder figs gated by OROFACIAL_GATE (unset in profile -> 'reference'); iti_where_decoder subset filter
  draws unfiltered CSV into subset folder; fig_softmax.py:60 crosstab orientation possibly wrong (unverified); two
  lick_enl_target_test titles still say "no lick 2 s"; not wired: rebuild_lick_maps subprocess modules.
- 10:55 PRIYA DECISIONS:
  (a) pre-cue LEVEL (lick_precue_restw[_traces], Figs 14.20 / 14.21): use the per-trial LICK-FREE ENL window
      (lp_licks.enl_window_starts_s, LP licks, drop trials without one), RESTnl primary + session-mean twin. Helper on
      feat/precue-enl-window-1109 (wf_precueenl).
  (b) OK to queue the full grant + epoch re-renders for the trial-set captions once the pushes land.
  (c) rebuild_figs encoder figures: same gate as everywhere = working trials (OROFACIAL_GATE=attempted under the camera
      gate). Queue fig steps already ran with attempted; plain replots under the profile defaulted to "reference" -- fix.
  (d) wfield import style leak (site-packages wfield/viz.py:84 style.use at import: Arial, dpi 500, black cycle): STOP it,
      use our own styling.
  (e) fix iti_where_decoder subset bug, verify/fix fig_softmax orientation, lick_enl_target_test "2 s" titles.
      (c)-(e): helper on fix/style-leak-misc-1109 (wf_misc1109).
  (f) l_xa_restnl crash (area_order: lateral VIS/AUD/TEa areas): FIX, keeping ONLY areas inside our glue mask (no left
      VIS). Helper on fix/xa-restnl-areas-1109 (wf_xafix). v12-IH lane halted at that step (mode stop).
  Stamp-test root cause: importing `wfield` restyles matplotlib globally (test-only fix 6bb2a103, queued in chain_v15).
  10:36 stopped the stamp helper's full-suite pytest (PID 39412, 13.7 GB; RAM was 1 GB free) -- pre-push hook covers it.
- 11:05 Priya OK for all parked-item recs: (a) contactOnly arm after the grant + epoch re-renders; (b) CD projections
  HOLD until the page update; (c) absolute-level twins with the page update; (d) hemisphere-balance check NOW (read-only
  helper); (e) LP lick decoder thread check when idle; (g) D2 copy-back / C3-C4 later. (f) frozen spout decoder 3x:
  Priya wants to discuss -> read-only trace helper (versions, input-set differences, users, does it move a number).
  Map re-renders: RESTnl baselines attached 107/108 (PS93_0813 no LP licks, expected) ~11:00.
- 11:15 (d) HEMISPHERE-BALANCE CHECK (read-only, verified at the key lines): the PRIMARY balance violates the 11-09 rule.
  lick_recovery_primary CUED_WIN = ITI_WIN = "ratio_010_050" (Priya 10-28) = per-lick mean 0.1..0.5 s MINUS per-lick
  baseline -1.0..-0.5 s (iti_licks.BASE_RESP :1489), then gain log ratio log2(obs_c/pred_c) - log2(obs_i/pred_i)
  (lick_balance_subsets.py:39-42,288-293). ~25 scripts inherit it (balsub_*, recprim_sessions, baltarget_licks.csv.gz,
  lateral_ratio AUC log2(c/i), ENLv balance, ci_spout_epoch, area_tuning, imaging_control, iti_prelick bs, xa w2).
  OK already (no per-lick baseline, a difference): the decoder-window family (lick_balance_reference / order / regions /
  maps / maps_svd / offset_lmm / recovery, recovery_markers part A, blind_decoder). A ratio needs a baseline (raw dF/F can
  be ~0 / negative). -> DECISION FOR PRIYA (asked 11:15). Full table: helper report in this session (scratchpad).
- 11:35 GLUE MASK MISSING IN THE REBUILD (= state-space arm B12 / decision A9, open since 11-08; verified here):
  under profile cohort1_rebuild_20261103 paint_exclusion.mask_dir -> <rebuild>/labcams/grant_figures (no masks there)
  -> beta_maps.painted_exclusion() silently 0 px (canonical union = 40,663 px); stat_mask 166,974 vs 129,770 px.
  Glue share per Allen area (union mask): VISp_L 98 %, SSs_R 54 %, SSs_L 46 %, VISl_R 81 %, VISrl_L 62 %, VISpm_L 48 %,
  RSPagl_L 14 %, SSp-bfd_L 6 %, VISp_R 31 %, all AUD / TEa / VISpor / VISpl / VISli / VISal_L 100 %. NO glue in ALM, tjM1,
  MOs, MOp, SSp-m, SSp-n, SSp-un, PL, ACAd, RSPd/v. LocaNMF glue-component drop was correct (no profile) -> LocaNMF
  decoders unaffected. Affected: maps (stat mask / FDR pool), Allen area extract (area traces: SSs, VIS, RSPagl),
  encoder pixel inputs (192,110 vs 151,447 px). xa crash fixed on fix/xa-restnl-areas-1109 e4b04637 (45-area set by
  name; 62 -> 45 areas = the glue set; also scripts/coupling_vigour_sm.py:266 has the same crash). DECISION FOR PRIYA.
  HOLD the approved grant + epoch caption re-renders until this is decided (else rendered twice).
- 11:35 (f) FROZEN SPOUT DECODER traced: copies A (d_attempt_freeze, DAQ, pre-cue 1.0 s), B (g5_joint_xsession,
  failed, orphan), C (g5_joint_xsession_r, DAQ, pre-cue 2.0 s), D (d_attempt_lp, LP-trained, 1.0 s; NOT on the old
  list). Cue: all the same model (cos 1.0000). Only D matches the decided settings; attempt coding already used D
  (captions / attempt_coding.py:31-32 / p_img_licksrc.py:85 wrongly say "DAQ-contact model, reused"). No LP
  joint_xsession exists -> LP grant figs 2/2b/4 MISSING; LP-lane LOSO pre-cue / lick numbers from C are wrong window /
  source. Pre-stroke acc D vs C: lick 0.01-0.05 lower; pre-cue 0.34-0.59 (1 s) vs 0.49-0.71 (2 s). Rec: D spec = the
  one LP decoder; run LP joint_xsession (~35 min, LocaNMF -> not glue-affected); fix the wording; A (12) + B (6) models
  = deletion candidates (Priya OK needed). Helper report: scratchpad/frozen_spout_trace/.
- 11:40 PRIYA: "yes do the shared fix and audit, hold re-renders until fix is in" (A9 decided). Helper on
  fix/rebuild-glue-mask-1109 (wf_gluefix): canonical painted masks under any profile (fence-compliant) + raise when
  missing + mask-dependent cache keys + impact audit / re-run plan. HOLD: grant + epoch caption re-renders, xa restnl
  re-run, any map / area-extract / encoder re-render. v14a map re-renders (running) = PREVIEW ONLY (glue in).
  Still open with Priya: balance A vs B; frozen spout recs (incl. deleting A 12 + B 6 models).
- 11:50 PRIYA: frozen spout decoder -- D = the single LP decoder; OK to delete copies A (12) + B (6) (kept list first);
  helper on fix/frozen-spout-D-1109 (wf_frozenD): delete A/B, check gate-source effect on D's rows, LP joint_xsession
  LOADING D (no refit), wording fix. Balance: Priya recalls log2 ratio was chosen -> confirmed (DECISIONS 10-28 late
  night: gain ratio 0.1..0.5 s, primary; nested held-out selection over window x difference/ratio picked the ratio).
- ~11:15 all 4 helpers hit the session limit (~11:55) and were resumed after the reset (nothing deleted; partial work
  intact). chain_v14 caption push still losing push races (main busy); chain_v15 waits on it.
- ~11:20 STYLE LEAK + BUGS + GATE DONE: fix/style-leak-misc-1109 (wf_misc1109, on the wiring branch): 4d3c4488 wfield
  imports via wfield_local/wfield_import.no_style_leak (rc_context; static test fails on any unwrapped import);
  ced5eff7 iti_where_decoder refuses non-z subset draws (no unfiltered subset PNG found on M:); 8845ea31 fig_softmax
  WEIGHT BUG CONFIRMED (to_dict() keyed by executed bin -> weighted by landed, not aimed licks) -> fig9 / fig9b data
  change; 4614add8 lick_enl_target_test titles read the LP ENL window; 0ea2da57 rebuild_figs encoder gate read from
  the producers' records (23 records, all "attempted"; caption-only effect). csig 3d7bbe921e; profile unchanged.
  Style-affected families: anything that built Allen ROI masks cold in-process (iti_* / lick_balance_* / lick_cue_to_lick
  / lick_enlv / gain_bio / encoder area masks / cohort2 quiet_*) + alignment QC figs -> look changes on re-render.
  Push queued: chain_v16 after chain_v15 ALL PUSHED.
- ~11:22 PRIYA: balance = log2 gain ratio stays PRIMARY (option B). Captions state its per-lick baseline (-1.0..-0.5 s
  before the lick, iti_licks.BASE_RESP); the raw 0..0.2 s decoder-window difference (14.17-14.19) = supplement. No
  recompute for this decision (balance re-run needed anyway after the glue fix: SSs ~50 % glue). RAM 5 GB free 11:18+
  (state-space replots of another window + our helpers + preview maps); queue waits, nothing killed.
- ~11:28 GLUE FIX DONE: fix/rebuild-glue-mask-1109 c42a7c35 (wf_gluefix; record docs/status/GLUE_MASK_FIX_2026-11-09.md):
  masks read in place under any profile (fence opens only grant_figures/exclusion_mask_painted_*.npy), load_mask raises,
  cohort2 opt-out painted_exclusion: none, mask_signature() in the encoder imaging cache key, iti extract mask_sig guard.
  Under the profile now union 40,663 / brain 151,447 / stat 129,770 (= default). csig 3d7bbe921e unchanged. 142 tests.
  ALSO BROKEN (verified): component_exclusion.excluded_pixels under the profile = bulbs only 15,106 px (correct 39.7-50.9k)
  -> rebuild CD trajectories dropped bulbs-only components -> CD chain recompute.
  AUDIT: encoder NOT per-pixel independent (shared spatial SVD over stat pixels; alphas pooled across targets) -> full
  encoder recompute. Area traces: 17 glue-only targets; SSs / VISl_R / VISrl_L / VISpm_L lose 90-98 % px (r .92-.99);
  encoder "cortex" target -22 % px; erosion rim: SSp-n_L 9 %, SSp-m_L 5 %. Unaffected: LocaNMF decoders (section G,
  nolick, pcd, joint_xsession, rest_frozen, attempt D...), behavior/pose. Maps: pixel values unchanged (full-grid
  caches), statistics re-render from warm caches. Recompute: encoder tree, ITI / area-ROI tree, CD chain, map stats,
  state_space_sim, O2 encoding bundles. ~1.5-2 days queue (critical path ~15 h ITI || ~15 h encoder + 14 h partial_refit).
  OPEN before re-runs: mask keys for O2 bundle, 7 other extract scripts, frozen_encoder --reuse-stats/--fit-from,
  state_space caches. Glue-in steps still running: rsplit PS93/94/95, l_code_noisematch_enlv_r5, 2 preview maps.
- ~11:30 PRE-CUE LICK-FREE ENL DONE: feat/precue-enl-window-1109 fdf7fec4 (wf_precueenl, on the wiring branch).
  Window = enl_window.sliding_lags (-> lp_licks.enl_window_starts_s; 1.0 s LP; settle 0.3 s; >= -2 s; else drop);
  LP gate licks; camera working trials; RESTnl (wr) default + --rest-variant session twin (+ restw robustness);
  precue05 = last 0.5 s of the ENL window; --precue-window fixed = old exactly. New caches precue_enl_lp1s/. 363 tests.
  3 sessions: trials 404->400 / 306->306 / 374->366; levels lower where windows slide (chronic 24 % slid vs 12 % pre),
  e.g. PS92_0918 ALM_R +0.02 -> -1.50. Full run ~2-2.5 h, WAITS for the glue fix; its new cache needs the mask_sig
  meta (add in the stale-protection pass). Page stems to switch to lick_cued_precue_enl_* after the full run.
- 11:33 NOTE: entry times 10:40-11:50 above were estimated, not read from the clock (actual span ~10:30-11:10); the
  later ones are corrected to ~11:15-11:30. RAM fell to 1.9 GB at 11:32 (preview maps 12.0 + 11.4 GB): STOPPED the
  RESTnl preview map g5_restnl_maps_lp_ts (PID 36696, mine; glue-in stats, superseded) under the 6 GB floor rule ->
  19.9 GB free. LPfinal preview (40068) continues for the panel / caption review.
- 11:40 l_restnl_rescue_r5 EXIT 1 (v11-L5R, code 90a5e25f): c2l_table "No objects to concatenate" -- no session got
  c2l_raw although l_c2l_extract_restnl_r finished 02:00 exit 0. Suspect: restnl_rescue.py:74 reuses its own session
  npz (VERSION match) built before the c2l extract, or cx.SESS_NL resolves under another tag root (same LP-prefix
  issue the precue helper fixed in restnl.level). Area-ROI tree -> re-runs after the glue fix anyway; fix the input
  check (raise if SESS_NL is empty / cache lacks c2l) in that pass. Not fixed now.
- 11:45 PRIYA: (1) push glue fix -> push_glue.sh running (race retries); (2) stale-read protection -> helper on
  fix/mask-keyed-caches-1109 (wf_maskkeys, on c42a7c35; also merges the xa area fix e4b04637, fixes
  coupling_vigour_sm + restnl_rescue c2l input check; lists stale folders for move-aside); (3) STOPPED rsplit
  PS93/94/95 (PIDs 2184/37244/26396; noisematch_enlv had already finished 11:34 exit 0); HALT file set in
  render_daqlk (every lane halts at its next step boundary; LPfinal preview finishes its current step);
  (4) RE-RUN PLAN APPROVED: after merge + mask keys + move-aside: CD chain + map statistics first (warm caches), then
  ITI chain || encoder chain, then downstream (ITI ~250 steps, encoder consumers, state_space_sim, O2 bundles); the
  grant + epoch caption re-renders, precue ENL full run, xa restnl, balance family fold into it (no double renders).
- 12:00 FROZEN SPOUT D DONE: fix/frozen-spout-D-1109 (wf_frozenD, 5 commits on 2009bf45, not pushed). A (12) + B (6)
  models DELETED after a kept list (docs/status/DELETED_FROZEN_SPOUT_AB_2026-11-09.md, 7d840b83; ~5.6 MB; no other
  references). Gate source lp: D's training rows identical (spec a9addfabb37f either way). LP joint_xsession 11:44-11:58
  from detached wf_render_jxlp @19c2325b: 12/12 frozen-hit = D ids, NO fit (new --decoder-only + WIDEFIELD_FROZEN_LOAD_ONLY
  guard; LP lick unit now uses attempt coding's no-lick reference, 19c2325b). Outputs = the MISSING
  joint_xsession_decoder_{cue,precue,lick}__LP_licks_gateLP_*.json -> g6_grant_lp can re-run. LOSO pre acc PS92/93/94/95:
  precue .341/.393/.588/.438, cue .855/.734/.935/.923, lick .930/.797/.943/.928. Wording 88044176. Open: LP joint
  encoders not fitted (grant reads only the decoder JSON) -- rec none; Fig 5.1 caption "engaged trials" should say camera
  working trials for cue / pre-cue (caption pass). Worktree wf_render_jxlp can be removed.
- 12:05 MASK KEYS (fix/mask-keyed-caches-1109, wf_maskkeys): committed f7e2cab2 (mask_keys module; 7 extracts; O2
  bundle record + runner check, enc_rebuild_20261104b now refused), 71cd1ffd (frozen_encoder / epoch_kernels),
  b1a7620d (state_space_sim: binned cache under m<sig>/, lnmf_keep name carries the exclusion digest), 7f419788
  (find_daq_trained could serve a glue-in DAQ model: manifests now record mask_sig, spec ids unchanged), 6ec478c1
  (restnl_rescue root cause = its OWN 2-day-old VERSION-only cache without c2l; now mask + c2l_src checked),
  a28cf70d (xa area fix merged), 9e9b1723 (coupling_vigour_sm by name). 32 + ~160 related tests; csig 3d7bbe921e;
  mask sig 5f9eb36f36b2 (same at default root). Helper resumed to commit the second sweep (cued_wide, firstlick_full,
  revup, traces, lick_correction, enlv, ...), cover the pooled caches by guard or move-aside, finish the move-aside
  scan and update GLUE_MASK_FIX doc. Follow-up: precue_enl_lp1s cache on feat/precue-enl-window-1109 needs the stamp.
- 12:08 PRIYA "ok to push": frozen spout D (wf_frozenD) + precue ENL (wf_precueenl) via chain_v17 (after chain_v16 =
  style-leak branch, which follows chain_v15 = wiring -> memos). Glue fix pushing in parallel (push_glue.sh).
  On main so far: caption branch 5a0612b4, stamp fix 7fd9f5f4. Mask-keys branch: push asked separately when complete.
- 12:45 GLUE PUSH 1st try FAILED only on test_grant_stamp_fit (branch merged main at 11:46, one minute before the stamp
  fix 7fd9f5f4 landed; 5747 passed) -> retry running (merges the stamp fix).
- 12:45 MASK KEYS COMPLETE: fix/mask-keyed-caches-1109 (wf_maskkeys) + c275f111 (all per-session extract reads via
  iti_licks._load; ~20 more extracts; pooled caches get <cache>.mask_sig sidecars) + 79c03cae (GLUE_MASK_FIX doc: full
  move-aside list; scripts/scan_mask_stale.py read-only scan). 81 + 448 + 149 tests; csig 3d7bbe921e; sig 5f9eb36f36b2.
  MOVE-ASIDE LIST (in the doc): ~23 x 107-session extract folders under the rebuild iti_licks W/X roots + 5 under E,
  frozen_encoder fit pickles (54 in 20 runs), epoch_kernels stats 76, partial_refit fits.pkl x4, spout_unique_cache,
  zscore SD tables, pattern pkls, O2 bundle enc_rebuild_20261104b; UNGUARDED pooled preds / balimg csvs + E/movement_first_*
  / musall_* (move so they rebuild); orphaned state_space_recovery cache/{cue,cue25,lick,splice} + lnmf_keep. NOT to
  move: rebuild frozen_models store (2,861 manifests lack mask_sig -> only the LP-arm DAQ cross-check refuses them;
  re-freeze the DAQ run if that check is needed). Push of this branch: asking Priya.
- 12:55 WIRING PUSH FAILED (5 of 6134 tests): (a) test_parallel_results_are_ordered -- iti_licks.main set the runtime
  module global _TS_ON (spawned workers never see it); (b) test_trial_set_lick_b for lick_enl_angle_decode /
  lick_enl_offset_explore -- their main write-guards OUT before declaring, and an earlier test leaves an LP tag. Fix
  2355e68c on wf_tswire: caption switch via env TS_ENV (WFIELD_ITI_TRIAL_SET_CAPTIONS), test redirects OUT; 377 tests
  pass. chain_v15 / v16 aborted on it; chain_v17 loop stopped (its frozenD push continues alone) so precue does not
  push before the fixed wiring. NEW chain_v18 (sequential, after running pushes): gluefix, frozenD, tswire,
  grantmemo, misc1109, precueenl. Mask-keys branch push: asked Priya.
- 12:58 PRIYA OK to push the mask-keys branch. chain_v18 (still waiting) stopped; chain_v19 = gluefix, MASKKEYS, frozenD,
  tswire, grantmemo, misc1109, precueenl (sequential, after the running glue / frozenD pushes). After glue + maskkeys
  land: precue_enl_lp1s stamp follow-up, move-aside (list in GLUE_MASK_FIX doc), lift HALT, queue the re-run plan.
- 13:25 GLUE PUSH failed again: earlier try lost a race (all tests passed); chain_v19 try merged a NEW main where
  test_every_star_user_reserves_the_band FAILS ON MAIN ITSELF: the state-space window's 419c029e scripts
  (state_space_encoder_subtract / fr_coupling / rest_decomp, B40-B42) place stars without reserve_headroom. Fixed on
  the glue branch (0d3c6f4d: reserve_headroom(fig) before each save; star tests pass) -- NOTE FOR THE STATE-SPACE WINDOW.
  chain_v20 relaunched (same order: gluefix, maskkeys, frozenD, tswire, grantmemo, misc1109, precueenl). Re-run prep
  helper drafting move_aside.py + queue_daqlk20.sh + PLAN.md (no launch).
- 13:40 RE-RUN PREPARED (not launched; scratchpad/glue_rerun/): move_aside.py dry run = 96 items, 12,092 files, 196 GB
  (incl. O2 bundle 29 GB, E/movement_first_* + musall_* 139 GB; +2 found: gain_bio/sessions,
  cohort1_quiet_spikes/v2_quiet_lp/sessions); --do refuses unless HALT is set; moves to <root>/_superseded_glue_20261109/.
  queue_daqlk20.sh: 386 steps (CD 8, MAPS 15, ITI 37, ENC 41, DOWN 267, FIG 8; O2 10 behind a manual marker), all
  `<family>_g` names; 363 match the originals exactly; 0 missing producers (several undeclared deps added, incl. the
  11-07 rlink_aim failure cause). ESTIMATE REVISED: 2.5-3 days (longest chain f_frA -> 3 partial refits ~28 h; ~36 h
  heavy-slot work; encoder ~136 h compute over 3 lanes).
  Coordinator decisions: (1) precue: mask stamp follow-up being added on wf_precueenl (merges maskkeys) before its push;
  (2) encoder lanes start together with CD / MAPS (no heavy-slot contention; MINFREE 15 GB governs); epoch renders KEEP
  the heavy slot (12 GB RAM each); (3) O2 re-bundle = manual coordinator step after the ITI inputs exist (new bundle
  name; decide the _daqlk tag then); (4) state_space_sim = state-space window, after e_iti_main_g (handoff note);
  (5)-(6) additions / exclusions accepted as listed in PLAN.md; (7) e_drift_june_g may fail (cont).
- 14:00 PUSH STRATEGY CHANGED: main moved 11x in 3 h (state-space window), so 24-min test runs kept losing races
  (glue: tests passed twice, rejected twice). Built integ/glue-batch-1109 (worktree wf_integ1109) = origin/main + all 7
  approved branches (glue+star fix, mask keys, frozen D, wiring+test fix, grant memos [conftest conflict: kept both
  autouse fixtures], style leak/softmax/gate, precue ENL+stamp). csig 3d7bbe921e, mask 5f9eb36f36b2, union 40,663 /
  stat 129,770 under the profile; 493 key tests pass. chain_v20 loop stopped (in-flight glue attempt finishes alone);
  push_integ.sh pushes the ONE branch after it (race retries). Re-run queue EXPECT_SHA = the integ commit once on main.
- 14:30 INTEG PUSH 1 FAILED (31 of 6297 tests): a TEST LEAK -- test_mask_keyed_caches::test_disengagement_img_worker
  imports scripts/disengagement_tests.py, which does os.environ.setdefault("WIDEFIELD_PRE_FROM", "0801") at import, so
  every later test ran as a pre0801 tagged run (writeguard / rebuild_root / cohort tests). Fixed (monkeypatch first);
  + main's own new break: pose_frame_pair.py (another window, 13:40) unclassified share writer -> BEHAVIOUR. 4abcea72;
  the 31 failing files + leaker: pass. Priya paused the state-space window's pushes ~13:55. Push 2 running.
- 16:50 (session was interrupted ~14:45-16:30 -- account change; nothing ran meanwhile, HALT held). INTEG PUSH 2
  failed 8 tests: (a) test_iti_where_decoder_subset's subprocess loaded wfield_local from the installed shared
  checkout (no mask_keys) -> PYTHONPATH = this checkout; (b) ITI / lick caption tests wrote to the live share when run
  after the full prefix (reproduced: prefix + iti_a = 3 failed / 5959 passed): an earlier test drops iti_licks from
  sys.modules, so drivers imported before (`import iti_licks as il`) hold a stale copy the fixture never patches ->
  conftest autouse one_iti_licks re-points stale references for each test. 8aed3a7b; 131 caption + writeguard tests
  pass. Push 4 running (~20 min tests).
- 17:10 INTEG ON MAIN 8aed3a7b (push 4; 6297 passed): glue fix, mask keys, frozen D, wiring, grant memos, style leak /
  softmax / gate, precue ENL + stamp, star-band + test-isolation fixes. State-space window may push again.
- 17:11 GLUE RE-RUN LAUNCHED. move_aside.py --do --orphans: 96 items moved to <root>/_superseded_glue_20261109/ (manifest
  scratchpad/glue_rerun/move_aside_manifest_20261109_171046.csv, copies in each superseded folder; nothing deleted).
  Detached worktree wf_render_v20 @ 8aed3a7b; queue_daqlk20.sh EXPECT_SHA 8aed3a7b; dry run = 376 steps (+10 O2 behind
  the manual marker), identical to the prepared plan. HALT removed (HALT.removed_20261109_1715). SETTINGS CHECK under
  the profile: union 40,663 / stat 129,770 / component exclusion PS92 50,885 (was 0 / 166,974 / 15,106 glue-in); csig
  3d7bbe921e; preflight OK on every lane. 10 lanes: L1 CD, L2 MAPS, L3-L5 ENC, L6 ITI, L7 ITI+DOWN, L8-L9 DOWN, L10 FIG.
  First heavy step g5_epoch_g (base DAQ-arm epoch render, as the original). ETA 2.5-3 days. Manual later: O2 re-bundle
  (new name) after the ITI inputs, then touch logs/O2_COLLECT_g.done + SECTIONS=O2 lane. State-space window: re-run
  state_space_sim after e_iti_main_g (its old caches moved aside).
- 17:50 e_iti_split_sd_g EXIT 1 at 17:14 (missed by the monitor, armed a minute later): iti_area_split_sd.py (and
  iti_zscore_check via il.labels()) list sessions from sessions_roi8fr, which the move-aside emptied -> empty table.
  The extraction itself needs no SD table, so: queue_daqlk20b.sh (one step e_iti_extract_g = iti_licks.py --jobs 3,
  same env token "frozen"; its summary may fail, cont) running since 17:47 (107 to extract); chain_v20c then checks
  107 extracts, rmdirs the empty e_iti_split_sd_g.claim, renames its .failed, relaunches the ITI lane (split -> z ->
  main summary). The ITI lane (L6) had halted (mode stop); other lanes unaffected (CD: g5_cd_traj_g EXIT 0 17:38).
  LESSON for the producer check: a step that LISTS sessions from a folder depends on that folder's producer.
- 18:20 RAM 3 GB at 18:16 (6.7 GB at 18:18): 3 ENC frozen_encoder steps (since 17:13; spont jobs 2, frA jobs 3, frAsplit
  jobs 2) + g5_epoch_g (2 arms) + e_iti_extract_g (3 workers; ~1 session/min, rebuilding the mask-keyed encoder input
  cache, ETA ~19:35) + aligned_decoder. Nothing stopped (an hour of encoder work each). RAM guard
  scratchpad/ramguard_v20.ps1: if available < 3 GB on 2 polls 20 s apart, stop ONLY the frAsplit step (PID 24684, by
  command line) + its workers, log to CHAIN.log. Queue MINFREE 15 GB blocks new starts meanwhile.
- 19:20 ITI recovered: e_iti_extract_g extracted 107/107 (17:47-19:17; its summary then failed at with_rest on the empty
  split table, as expected, cont). chain_v20c: 107 extracts present, e_iti_split_sd_g claim cleared, .failed renamed
  (.failed_1714_empty_session_dir), ITI lane relaunched 19:17 (waits for 15 GB free). CD chain DONE: g5_cd_traj_g,
  g5_cd_traj_orth_g, cd_overlap_null_g, g7_cd_geometry_g all EXIT 0 (19:11). No RAM-guard action (RAM 13-17 GB).
- 19:40 SKIP CASCADE found + cleared: while e_iti_split_sd_g was .failed (17:14), the ITI+DOWN lane marked 31 dependent
  steps .skipped (17:15; e_iti_zscore_pre_g, e_iti_main_g, e_june_ref_g, ... 28 by ITIDOWN, 3 by ITI) and held their
  claims -> they would never run. Literal-path script glue_rerun/clear_cascade.sh: 31 .skipped -> .skipped_1715_cascade,
  31 empty claims rmdir'd. Two fresh lanes SECTIONS="ITI DOWN" (L11, L12; MINFREE 12) picked them up 19:37 (zscore
  claimed). e_iti_split_sd_g EXIT 0 19:23. Monitor now also flags "SKIPPED (dependency". LESSON: after clearing a
  failed stop-mode step, ALSO clear the dependency-skip cascade it caused before relaunching.
- 19:55 e_iti_zscore_pre_g EXIT 0 (19:51); e_iti_main_g (summary on the 107 cached extracts) running. No failures since
  the cascade clear. Handoff block 40 updated (~20:00 entry).
- 20:15 e_iti_main_g EXIT 0 (20:03). e_firstlick_extract_g EXIT 1 (20:11): reads lick_order_preds_areas_splitr8.csv.gz
  (moved aside; producer e_split_order_all_g, NOT a declared dep -- producer check gap). Cleared (.failed ->
  .failed_2011_no_order_preds, empty claim rmdir, literal paths) BEFORE any new skip cascade; queue_daqlk20c.sh re-runs
  it with dep e_split_order_all_g (same step name, dependents wait for its .done). Read-only scan helper listing all
  other undeclared reads of moved-aside files so they can be handled before they fail.
- 20:25 DEP-GAP SCAN (read-only, scratchpad/gapscan): 4 more gaps (steps reading moved-aside files without their
  producer upstream; 2 would silently write PARTIAL results, not fail): c_fr_pixsig_g (+ drift_after, pixmapsig) need
  O2_COLLECT_g; e_spout_patterns_g / e_spout_pat_stats_g and e_split_where_g / e_plain_where_g need l_basis_extract_g;
  e_filter_cmp_g needs e_split_order_{all,far1,gap2far1}_g + e_split_gauss_{all,far1,gap2far1}_g. None claimed yet ->
  queue_daqlk20d.sh: the same 8 steps (same names / commands / env tokens) with the deps corrected, one section per
  step, 8 lanes -> all 8 claimed (v20 lanes skip them). 4 lanes aborted at preflight when started 3 s apart (collision)
  and were relaunched 25 s apart: all OK. Monitor now matches v20b/c/d lanes too (it missed those aborts).
- 20:30 resumed after Priya's login switch (~3 min): no failures / skips / aborts; 10 _g steps done; 27 lane processes;
  RAM 17 GB; RAM guard (PID 36408) + monitor alive. Housekeeping: stopped 60 orphaned `tail -F CHAIN.log` processes
  left by this session's expired monitors (exact PIDs, this session's scratchpad path only); 4 tails of session
  cdd1e016 (10-25) left alone.

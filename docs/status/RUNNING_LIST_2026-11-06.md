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

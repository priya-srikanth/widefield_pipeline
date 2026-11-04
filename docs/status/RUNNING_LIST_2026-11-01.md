# Running list — 2026-11-01 night (lick source, RESTnl roll-out, engagement gate)

Priya 11-01 (going to sleep): "please proceed with any new analyses you deem necessary - keeping a running list for when I
return and what was added to the page / any conclusions that changed. keep an updated handoff throughout."
Pages: goal https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d ; imaging https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV
Handoff: STATUS_2026-10-26_ENCODING_HANDOFF.md, section 00-COMPACT-19. All PROVISIONAL, n = 4.

## 1. Decisions needed from Priya
- [ ] PRM map-cache digest (affects MAIN too): commit 8e77bd0a (10-29, f"2026{mmdd}" -> config.date8) moved the prm
  map-cache digest, so the ~33 GB warm map caches are unreachable from main; next 15r / drift render on main = full
  rebuild (~1.5 h, +33 GB). Opt-in WIDEFIELD_PRM_LEGACY_DATE8=1 (branch analysis/restnl-maps 00f1e2c2) hashes as before
  (bit-identical cache names; refuses cohort-2 dates). RECOMMEND: adopt on main for cohort 1. Or: accept one rebuild.
- [ ] Flat RESTnl for PS94 late epochs: all-session addendum (RESTNL_MAP_SHORTCUT_CHECK) finds PS94 subacute/chronic
  sessions drift upward within session; flat shifts the far-contra map by a median 20-40 % there (all positions
  together; between-position contrasts unaffected). Pre/acute fine (2.8 % / 5.2 %). Your condition was "flat as long as
  appropriate". RECOMMEND: keep flat primary (contrasts unaffected), caveat stated in the 15r caption, and add a
  time-local lick-free sensitivity before quoting any PS94 late AMPLITUDE. Or: switch maps to time-local lick-free.
- [ ] Imaging page: RESTnl maps ADDED next to the REST ones (nothing removed). Which should be the headline?
- [ ] CAPTION ERROR on imaging Fig 2.4 (and the LP 1.0 s ENL comparison): caption says "frozen decoder trained on DAQ-contact
  trials", but the epoch acc family refits its pre-stroke decoder IN MEMORY on the current arm's (LP-gated) pre trials
  (grant_confusion.py `_pipe().fit(XE[e_pre], YE[e_pre])`; WIDEFIELD_LP_FROZEN never reaches it). So those figures were
  effectively LP-trained and not stored-frozen (breaks CLAUDE.md rule 10). Numbers barely move: final DAQ- vs LP-trained
  acute cells differ <= .05. Plan: APPEND a correction note to those captions (no removal); please confirm wording.
  Figs 5.1/5.2 (attempt classes) are a different script -- not checked yet.
- [ ] Make the stored-frozen epoch decoders (opt-in WIDEFIELD_EPOCH_FROZEN=1, branch analysis/lp-final) the default?
  RECOMMEND yes (rule 10). Would change epoch acc figures slightly everywhere -> discuss before re-rendering page figures.
- [ ] Section 7 takeaway "-55 / -48 / -38 / +2 %" (ipsi-hemisphere change, LP lick map, from 10-21 numbers): the 10-24
  render behind Fig 7.19 gives -55 / -49 / -38 / -9 % (PS95 -9 %, not +2 %). Headline-adjacent -> NOT edited; update it?
- [ ] Rest decoder 12.1 lick-free: August-only pre RESTnl is TOO THIN for PS92 / PS93 (frozen training pool far contra
  PS93 9 periods, PS92 far ipsi 25) -> not fitted. Options (REST_DECODER_NL_20261102.md section 7, worktree wf_restdec):
  (1) June+Aug pre as sensitivity [RUNNING as a labelled sensitivity, cheap]; (2) Aug-only PS94/PS95 per animal, no pooled
  claim; (3) drop thin positions (loses PS93 far contra); (4) shorter cam4 post-lick buffer; (5) fit anyway (no).
- [ ] 12.1 caption: says 50 permutations, code uses 200; its pre-stroke models are refit in memory each render (rule 10).
  Correct the caption (append note)?
- [ ] Fig 2.4 under stored frozen DAQ decoder: re-render (acute far contra pooled .314 -> .260; PS92 .27 -> .15; still
  4/4 below pre) or keep current labelled "LP-trained in memory"? (Correction note on 2.4 still awaiting Priya.)
- [ ] Move the remaining in-memory "frozen" fits to the store too (5c frozen-vs-refit column, grant_figures 5b
  confusions)? In the LP arm these are LP-trained while acc is now DAQ-trained. RECOMMEND yes.
- [ ] frozen_models.siblings() logs false SPEC-CHANGED (ignores window length / lick source). Fix?
- [ ] LP ENL window: 1.5 s looks better than 1.0 s (enough trials everywhere; pooled pre accuracy .42 -> .48; same acute
  loss 4/4). 2.0 s keeps trials selectively where the deficit is -> sensitivity only. Which as primary for the LP arm?

## 1b. Decisions taken by "ok" on 11-01 (my recommendations in RESTNL_ROLLOUT_PLAN_20261101.md; Priya said ok)
- RESTnl primary; old baselines (RESTW on the goal page, time-local REST on the imaging page) kept as collapsed sensitivity.
- Maps: FLAT RESTnl (one baseline per session), same rule as the goal page (wr: >= 50 frames at >= 4 of 6 positions).
- Maps: August-only pre for the RESTnl version; PS93_0813 (no LP licks) left out of RESTnl.
- Rest frames used as DATA: rest decoder 12.1 re-run on lick-free rest first; quiet events 15.x and 14.18 S10/S17/S18 as
  sensitivities (event counts first); state decoder 3.x not now.

## 2. Done tonight (page changes, conclusions that changed)
- Before Priya slept (11-01 evening): lick-source tag on every figure title (imaging v39, goal v301); 14.21 S1/S2/S11-S13
  captions re-read on RESTnl (goal v302) -> CHANGED: acute far-contra pre-cue raise NOT orofacial-specific (reference
  areas rise as much); S11 contra-ipsi 3/3 -> 2/3; S12 ipsi tjM1 chronic rise now 4/4; S12 D non-orofacial dip no longer
  significant; S13 last 0.5 s before cue chronic > pre 4/4. S12 stars moved inside the panels (v303).

## 3. Running / queued
- Engagement-gate vs LP-attempt check (helper; docs/status/GATE_LP_CHECK_20261101.md when done).
- Queue (RESTNL_ROLLOUT_PLAN order): 14.21 caption stale line; 14.21 S7 rev-up traces RESTnl; read-only shortcut check on
  cached maps; renderer code on a branch off main (widefield_pipeline); map runs; rest decoder 12.1 lick-free.

## 4. Log
- 23:10 launched: (a) 14.21 main caption -> RESTnl numbers + S7 rev-up traces RESTnl variant (revup_neural.py
  --rest-variant restnl; RESTW kept collapsed); (b) read-only map shortcut check -> RESTNL_MAP_SHORTCUT_CHECK_20261101.md.
  Gate check still running.
- 23:20 Priya: "keep old baseline as supplemental. flat restnl, as long as that's appropriate (no need for further
  within-session drift or artifact correction). august only i suppose." -> added a drift check to the shortcut helper
  (spread of the time-local REST across bins vs map effect sizes; by epoch / session position); flat RESTnl for maps
  only if it passes, else time-local lick-free (decision would come back to Priya).
- 23:35 quiet-events window (sabatinilab-a8) will add imaging section 15 to the GOAL page as section 23 (id quiet-state,
  before 'Pending'), publishing from the live version. I hold publishes ~45 min; after their publish I must merge their
  section into _goal_page/index.html before my next publish (re-read live). Asked them to tag lick source + note RESTnl.
- 23:55 GATE CHECK DONE (GATE_LP_CHECK_20261101.md): no LP-gate re-run needed for maps / Haufe / S28 quits; stopped class
  borderline (optional sensitivity); DAQ-arm pre-cue windows 17-25 % contain an LP lick in every epoch -> level caveat,
  not a stroke effect; LP_licks ENL arm is the one to cite. CONCLUSION CHANGE: none; adds caveat to imaging 1.1 / 2.1 / 7.x
  DAQ pre-cue figures (caption note to add).
- 23:55 quiet window published section 23 on the goal page = v304 (built on my v303). MUST merge their section (source
  in their scratchpad .../782a4a7a-.../scratchpad/encpage/index.html; fig_ids rows fig_ids_rows.csv) into
  _goal_page/index.html after my 14.21/S7 helper finishes, before my next publish. They are running the quiet-frame LP
  sensitivity themselves (do not duplicate).
- 00:10 MAP SHORTCUT CHECK DONE (RESTNL_MAP_SHORTCUT_CHECK_20261101.md): RESTnl map = cached raw - U @ b_nl, exact, no
  re-extraction, one baseline map per session serves all 7 prm arms (verified 8 sessions to <= 1e-7). FLAT IS APPROPRIATE
  (Priya's condition): time-local vs flat changes the far_R map by 0.9-5.4 % (15 % only where the map is ~0, acute
  PS94); bin-to-bin spread is mostly noise, no epoch dependence (0.41 vs 0.36e-3), quit period already gated out; a
  time-local lick-free baseline would mostly add noise. Caveat: near-zero cells (acute far contra) carry ~+-0.5e-3
  baseline uncertainty. -> launching the RESTnl map implementation on a branch off widefield_pipeline main.
- 00:20 launched RESTnl map implementation + renders: new worktree C:/Users/SabatiniLab/Github/wf_restnl (branch
  analysis/restnl-maps off widefield_pipeline main 7272d0be); new module wfield_local/restnl_mask.py, derived 'restnl'
  reference outside the cache key, outputs in NEW folders (epoch_restnl etc.), August-only pre; ~3-4 h. Results doc
  RESTNL_MAPS_20261101.md. Imaging page update after (RESTnl primary, REST collapsed supplement).
- 00:40 Priya: "1 or 1.5 s no-LP-lick decoders?" + "the encoding page may already have a version" -> YES, exists:
  LP 1.0 s ENL arm (10-24, ENCODING_FINDINGS "matched 1.0 s"): DAQ 1 s vs LP 1 s decoding within .03 everywhere;
  acute far contra .21 [.12,.29] 4/4; ENL far-contra MAP increase 9.9 % -> 4.4 % of bins under LP (part was non-contact
  licking). Not on the imaging page yet. QUEUED after maps: LP 1.5 s (+2 s if trials allow) with LP gate + LP-trained
  frozen decoders (DAQ-trained alongside); add the LP 1.0 s decoders to imaging page next to 2.1 / 2.2.
- 00:40 Priya: "LP gate anyway for consistency?" -> recommended YES for the LP arm (DAQ arms unchanged). Launched: LP-
  inclusive gate as an OPTION (WIDEFIELD_GATE_SOURCE=lp; default byte-identical, pinned test; no renders) ->
  LP_GATE_OPTION_20261101.md.
## DECISION FOR PRIYA (morning): full "final LP arm" re-render (LP licks + LP gate + RESTnl + August pre, each family
   once in final form; several hours, cold caches)? Not started overnight.
- 01:00 14.21 MAIN caption + S7 on RESTnl DONE (commit 5a1b541a). CONCLUSIONS CHANGED (14.21, a SUPPORTING figure; title
  changed accordingly -- REVIEW): acute far-contra pre-cue raise is BROAD (34 / 45 areas, 30 of them 4/4; was 8);
  chronic far contra now a bilateral orofacial rise 4/4 (was nothing significant) and the 11 posterior falls are gone;
  all-spouts acute 12 areas RISE (was only 2 falls); chronic falls 20 -> 11 (VIS / RSP only); "4/4 in all key areas both
  hemispheres" no longer true (ipsi SSp-n 3/4 n.s.). Caveat: the lick-free rest level itself falls (contra tjM1 -8.0
  4/4) more than the raw pre-cue level, so "rise above rest" = pre-cue falls less than rest. S7: no conclusion changed
  (flat shift; only levels move). Old RESTW versions kept collapsed.
- 01:05 merged the quiet window's section 23 into the local source (22 figs, status row, fig_ids rows).
- 01:05 Priya: "ok for the full job - ensure we discuss before you take anything off the page or replace headline
  results" -> final LP arm re-render APPROVED; RULE for tonight: ADD new versions only; nothing removed from the page and
  no headline result replaced without discussion (new versions go next to / below, old kept visible).
- 01:20 goal page PUBLISHED v305 (14.21 main + S7 RESTnl; section 23 merged; nothing on the live page lost, verified block by block). Told the quiet window.
- 01:35 quiet window v306 (section 23: ipsi > contra synchronisation WITHDRAWN -- non-calcium 7-11 Hz peak; quiet frames being re-run with DAQ + LP licks overnight); merged locally (commit below). Live = v306.
- 02:00 LP GATE OPTION DONE + committed (dea1de5f; LP_GATE_OPTION_20261101.md): WIDEFIELD_GATE_SOURCE=lp (response =
  own response OR DAQ contact OR LP hit/incomplete; jaw_only / no_try do not count); default byte-identical (18 tests);
  map caches stay warm (hook lines excluded from the digest), frozen ids unchanged; outputs go to *_gateLP folders.
  FINDING: the LP arm's imaging gate was ALREADY LP-based (shared lick loader) -> 0 trials flip there; under the switch
  only the rest-baseline gates, S28 quit and no-LP-verdict cues change. Trials flipped pooled: pre 0.07 %, acute 0.5 %,
  subacute 0.9 %, chronic 0.5 %; 12 / 108 quits later, one lost (PS94_0921). Note: ruff per-file ignore added for
  precue_engagement_states F401 (removing the import would cold-start all map caches).
- NEXT (after the RESTnl map helper finishes; box capacity): FINAL LP ARM render in wf_attempt with
  WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_GATE_SOURCE=lp, August-only pre; ENL 1.0 / 1.5 (/2.0 if trials allow) s; LP-trained
  frozen decoders as sensitivity; RESTnl derived reference (port restnl_mask from wf_restnl). ADD to pages only.
- 02:55 RESTNL MAPS DONE (helper; RESTNL_MAPS_20261101.md; code commit 00f1e2c2 on analysis/restnl-maps, docs 2f3f6c1f):
  7 pooled 15r + 7 per-animal + 7 drift + 18 attempt maps in NEW folders (G/epoch_restnl, G/LP_licks/epoch_restnl,
  G/DAQ_ENL1s/epoch_restnl, G/map_drift_prepre_restnl/*, imaging_encoding/attempt_maps_restnl). Validation: frame counts
  = restw_check 107/107; shortcut sessions reproduce exactly. CONCLUSIONS (page REST all-pre -> RESTnl Aug-only, far
  contra acute/pre): cue maps still 4/4 (contact .48->.53, LP .48->.52); LP lick 3/4 unchanged; attempt maps reading
  unchanged. CHANGED: (1) pre-cue far-contra rise 3/4 -> 4/4 in contact (7.1, 1.44->1.80) and ENL1s (7.5, 1.67->2.04)
  windows -- PS94 flips .92/.82 -> 1.17/1.13 (RAW agrees with RESTnl); (2) PS94 chronic far contra ~.9 -> ~.71, most
  likely the flat-baseline shift (decision above), not cortex; (3) June->Aug pre drift significant in more bins
  (contact 216 vs 67) -- acute changes still exceed drift. Two new decisions above (cache digest; PS94 flat).
- 03:05 launched (helpers): (a) imaging page -- ADD RESTnl maps next to the REST versions (b-suffix figs, both visible,
  nothing removed; publish from the full live read); (b) FINAL LP ARM render in a new worktree/branch analysis/lp-final
  (gate option + RESTnl combined; LP licks + LP gate + Aug-only pre; ENL 1.0/1.5(/2.0) s with trial counts first;
  LP-trained frozen decoders alongside DAQ-trained; new *_LPfinal folders; plan in LP_FINAL_ARM_20261102.md; <=4 workers).
- 05:20 FINAL LP ARM DONE (helper; worktree wf_lpfinal, branch analysis/lp-final = restnl-maps + gate cherry-pick 3544e4b9
  + 5822cbad; doc LP_FINAL_ARM_20261102.md there). 88 figures in NEW folders G/LPfinal/... and imaging_encoding/
  attempt_maps_LPfinal_{restnl,rest}; tables G/LPfinal/compare/. 3 h 54 min, <= 4 workers. NO HEADLINE CHANGES: maps vs
  current LP versions within .01 (cue RESTnl acute .52 pooled, 4/4 below pre; lick 3/4; drift within .015; attempt maps
  same). ENL far-contra acute loss 4/4 at 1.0 / 1.5 / 2.0 s, DAQ- and LP-trained alike (diff <= .05 acute). August pre
  makes PS92/PS93 pre accuracy slightly higher -> per-animal drops slightly larger than the page. FOUND: page 2.4 / LP 1.0 s
  ENL decoders were refit in memory on LP pre trials, not the stored DAQ model the caption names (decision above).
  ENL keep rate far contra: 1.0 s >= .94, 1.5 s >= .79, 2.0 s down to .53 (PS92/93 acute/subacute).
  Not done: 14m beta maps, 5c/5cr, cue/lick-window decoders 2.3/2.4, ENL maps at 1.5/2.0 s, page-style open markers.
  NEXT: after the imaging-page RESTnl helper reports, ADD LP-final figures (+ caption correction note) to the imaging page.
- 05:35 quiet window published goal v307 (23.12 breathing caption) + v308 (15 section-23 figs on LP-based quiet frames, DAQ contact + LP video; rest marked DAQ-only; 23.5 T13 follow-up). Merged locally. Live goal = v308.
- 05:45 IMAGING PAGE v43 PUBLISHED (helper; v42 = first 10 images only, page text unchanged; 64 MB publish cap): RESTnl
  twins 7.1b-7.11b, 7.16b-7.20b, 8.1b, 8.2b directly after each REST figure; all 620 original lines present in order, 18
  got only an appended pointer "RESTnl version: Fig X.Yb, directly below"; new "Where things stand" row; section-7 summary
  block (REST -> RESTnl per-animal table); Pending item listing the open decisions. Source committed on analysis/restnl-maps.
  8.1b (RESTnl, failed far-contra attempts vs pre hits) lower in 3/4 (not PS95); vs pre misses lower 2/4 (PS93 4.79 from a
  near-zero pre-miss map -> read as "not lower").
  UNRESOLVED: PS94 REST lick-window far contra = -38 % (~.62) in the 7.19 caption vs .72 in RESTNL_MAPS table; flagged in
  the 7.19b caption + Pending, to reconcile.
- 05:50 launched helper: ADD compact "Final LP arm (11-02)" block to imaging page (ENL 1.0/1.5 s decoders DAQ- vs LP-trained, per-animal + trial-count tables, section-7 pointer that LP-final maps reproduce 7.xb within .01, Pending items) + reconcile PS94 7.19 REST number. Fig 2.4 correction NOT applied (waits for Priya).
- 06:20 IMAGING PAGE v44 (helper): "Final LP arm (11-02)" block at #lpfinal after Fig 2.4 -- Figs 2.5-2.8 (ENL 1.5 s /
  1.0 s x DAQ- / LP-trained, stored frozen; pooled far contra 1.5 s DAQ .48 -> .21 [.13,.29], all 4/4), per-animal table
  (all windows x trainings), trial counts; section-7 paragraph (LP-final maps = current within .01 acute cue/lick, .015
  pre-cue 1 s, .04 subacute/chronic; pre-cue RESTnl chronic below pre 1/4 -> 2/4, PS94 1.02 -> .98); Where-things-stand
  row; Pending line. Fig 2.4 caption untouched. RECONCILED PS94 lick: -38 % = ipsi-hemisphere mean change; .72 = whole-
  mask RMS ratio all pre; .67 = same, August pre (gate no effect). Only the two v43 "not reconciled" flags were replaced.
- 06:30 launched helper: rest decoder 12.1 on lick-free (RESTnl) rest, new worktree/branch analysis/rest-decoder-nl, frame counts first, default byte-identical, new *_restnl folders, doc REST_DECODER_NL_20261102.md.
- 07:20 REST DECODER 12.1 LICK-FREE (helper; worktree wf_restdec, branch analysis/rest-decoder-nl 7c53214a): option added
  (default byte-identical, SHA-checked on real sessions); counts done, NOT fitted under Aug-only pre (fairness gate >= 30
  periods/position fails PS92/PS93). FINDING from counts alone: pre-stroke, 97 % of PS93's far-contra "rest" periods lie
  within [-0.5, +2.0] s of a video or DAQ lick (14 / 542 survive); PS92 keeps 30 %, PS94 36 %, PS95 89 %. So 12.1's pre rest
  code for PS92/PS93 is built mostly next to licks the DAQ never registered (its lick-gap matching uses DAQ licks only).
  CONCLUSION AT RISK (not yet tested): 12.1's rest code. Launched June+Aug pre sensitivity (REST reproduction + RESTnl).
- 07:45 Priya: "add correction note" (12.1) -> helper appending it (permutations 200 not 50; models refit each render;
  PS92/PS93 lick-adjacent rest caveat). Priya: "we should cache frozen decoders as we always do" -> helper making the stored
  frozen path the DEFAULT for the epoch acc family and 12.1 (branch analysis/frozen-default, worktree wf_frozendef; verify
  identical numbers where the training set is unchanged). 12.1 REST reproduction with the new driver (June+Aug pre): chronic
  retained 0.54 = page. Priya asked why the RESTnl NORMALISATION works if PS92/PS93 fail the decoder: the baseline needs a
  per-session mean (rule wr: >= 50 FRAMES at >= 4 of 6 positions, equal-weight mean over the positions present); the decoder
  needs >= 30 independent labelled PERIODS per position for training. CAVEAT FOUND: PS93 far contra is below 50 lick-free
  frames in 5 of 7 August pre sessions (0807, 0809-0812; 16-36 frames) and in 0828 -> its baseline there is the mean of the
  other 4-5 positions; PS92 drops one position in 0811 / 0812 / 0826. Per-position RESTnl counts in the npz cache
  (C:/Users/SabatiniLab/.widefield_session_cache_restnl/, nfr_pos / positions_used).
- 08:00 IMAGING PAGE v45: 12.1 correction note appended (only line changed). Verified: models fit in memory per full run (--from-csv redraw reuses scores); published run used --perm 200 (min p = 1/201). Script line in caption still lacks --perm 200 (left).
- 08:20 12.1 LICK-FREE SENSITIVITY (June+Aug pre; driver wf_restdec rest_frozen_decoder_nl.py; stored frozen models
  kind rest_frozen; figs G/rest_frozen_decoder_restnl/{rest_allpre,restnl_allpre}/; --compare stage crashed only because
  the Aug-pre CSVs do not exist -- figures + per-animal tables written). REST reproduction = page (acute .28, chronic .54).
  RESTnl, pre acc (sessions p<.05): PS92 .220 vs null .168 (2/9) [REST .383, 11/11]; PS93 .294 (8/9) [.402]; PS94 .358
  (10/10) [.470]; PS95 .358 (11/11) [.355]. Retained acute / chronic: PS93 -.10 / .51, PS94 .14 / .18, PS95 .10 / .34
  (PS92 pre at chance -> retained meaningless). Pooled flat retained acute .18, subacute .46, chronic .40 (page .28/.36/.54);
  delta acute -.140 [-.211, -.043], chronic -.100 [-.176, -.021], all CIs exclude 0.
  CONCLUSION CHANGE (provisional, sensitivity only): the rest position code is PARTLY LICK-ADJACENT -- pre accuracy falls
  .403 -> .311 pooled once video licks are removed, and PS92 has essentially no lick-free rest code. In the 3 animals with a
  lick-free code it still falls acutely and only partly recovers; the qualitative 12.1 claim holds in 3/4, carried by
  PS93-95. Not on the page (June+Aug pre breaks the Aug-only rule) -> ask Priya whether to add as 12.1b.
- 08:40 quiet window published goal v309 (23.5, 23.6, 23.11, 23.12, 23.15 on LP-lick quiet frames; 23.16 later). Merged locally. Live goal = v309.
- quiet window published goal v310 (23.16 lick LFO on LP lick onsets). Merged locally. Live goal = v310.
- quiet window published goal v311 (section 23 ~5x -> ~7x in h2 / take / 23.1 / status row; 23.16 LP composition; LP-lick switch for section 23 complete). Merged locally. Live goal = v311.
- RESTNL POSITION COVERAGE CHECK DONE (RESTNL_POSITION_COVERAGE_20261102.md; outputs G/restnl_position_coverage_check/):
  NO HEADLINE CHANGES. Counts (pre-cue rise 4/4, cue loss 4/4, LP lick loss 3/4) hold under imputation, wr25, and fixed
  position sets; pooled far-contra acute values move <= .023. Only effect: PS93 pre-cue ratio understated under wr
  (7.1 1.17 -> 1.24; 7.5 1.24 -> ~1.33). PS93 subacute/chronic cells near 1 fragile (do not read direction). PS94 low
  chronic is NOT a coverage effect. RECOMMEND keep rule wr at 50 frames primary; imputation as the one sensitivity; caveat
  sentence on PS93 pre-cue amplitude (wording in doc). Script committed on analysis/restnl-maps.
- FROZEN DEFAULT DONE (worktree wf_frozendef, branch analysis/frozen-default = rest-decoder-nl + merge lp-final d966e80b +
  1d0fc040; FROZEN_DEFAULT_20261102.md there). Epoch acc family + 12.1 now load_or_fit by default (opt-outs
  WIDEFIELD_EPOCH_FROZEN=0 / WIDEFIELD_REST_FROZEN=0); 28 new epoch_decoder models in C:/Users/SabatiniLab/frozen_models.
  Stored == in-memory bit for bit where the training set is the same (DAQ 2 s, ENL1s, LP LP-trained, 12.1). Pre-existing
  drift vs current page renders (reproduced by the in-memory path, so not this change): DAQ 2 s bars <= .024 (subacute
  far mid ENL ** -> *), ENL1s <= .009, LP <= .007, 12.1 retained .28 / .54 unchanged. REAL CHANGE: LP arm now loads the
  DAQ-trained model -> Fig 2.4 lick-window acute far contra pooled .314 -> .260 (decision above). Renders in
  G/frozen_default_20261102/. Not on the page.

### 11-02 (late morning): Priya's decisions on the frozen-default items
- **"cache all frozen decoders": yes.** Helper converting the remaining in-memory frozen fits on analysis/frozen-default: the 5c refit / paired / paired_matched frozen column, the 5b confusions, and any other pre-stroke reference fit per render (inventory to come). Write-up: docs/status/FROZEN_ALL_20261102.md (wf_frozendef).
- **siblings() false SPEC-CHANGED: fix, yes.** The same helper adds post_s and lick_source to the sibling grouping (warning only; no spec_id change).
- **Both trainings in the LP arm: yes** ("a version of both daq contact and lp lick-based"). The pre-cue arm already has both (imaging Figs 2.5-2.8). Lick window: helper adds Fig 2.4b (stored DAQ-trained, acute far contra pooled .26) and a CORRECTION note on Fig 2.4 (in memory = LP-trained, reproduced exactly by the stored LP-trained model). Add-only; nothing removed.
- **Imaging page v46 (added, nothing removed):** Fig 2.4 CORRECTION note (LP-trained in memory; stored LP-trained reproduces it exactly) + new Fig 2.4b (stored DAQ-trained, lick window, LP gate): pooled far contra .897 -> .260 ** -> .662 ** -> .710 **; acute < pre 4/4. wf_restnl commit 1cb58c9c. The 2.4b "Script" line names epoch_accuracy.py by inference; not read from render metadata.
- **Frozen store, pass 1 (wf_frozendef 8debc35e):** 5r/5ro frozen column + 5b confusions now load the acc family's stored model (identical to in-memory on the DAQ arm and on LP-trained LP; LP default = DAQ-trained, lick/lick acute far-contra gap +.255 -> +.308). siblings() false SPEC-CHANGED fixed (warning-only). Store unchanged at 2067. NOT yet converted (listed in FROZEN_ALL_20261102.md §2): position_coding_directions, nolick_decoder, poststroke_compare, attempt_coding CD axes, cd_trajectories, june_august_drift, frozen_encoder, ~12 ITI/cued read-out scripts. Need a decision: precue_engagement_states (editing it colds ~33 GB of map caches) and 5rm paired_matched (a different random pre subset per session). Pass 2 started on the rest.
- **Queued (Priya "sure", 11-02 evening): cohort1 on the swap-masked + 415-lag-fixed variant** (`hemo_meegkit_hpfit_lag_refitT_swapmask`, scripts/cohort2_qc/cohort1_impact.py). 10-30 checks (COHORT2_QC §0i 8 sessions, §0l 24 sessions) showed headline decoding / area amplitudes do not move (max .02), maps r >= .97, amplitudes -2 to -14% (session-dependent). Not yet re-run: LocaNMF / frozen decoders (Figs 2.x), encoding headline, RESTnl maps (cross-epoch amplitude ratios are the real exposure), quiet events. Plan: build the variant for all cohort1 sessions, re-render those families into new folders, report only what moves (add, never replace). Starts once the frozen-store pass 2 finishes (shared box).

- 2026-11-03: goal/encoding page is now v312 (sabatinilab-a8 edited section 23 only: new Supp 23.23 / 23.24 as INLINE images because the artifact is at its 512-file limit; 23.x headers retagged LP). Before any publish from _goal_page: read v312 first, merge section 23, and if making 23.23 / 23.24 real files, free slots + add fig_ids rows q_t13_followup_lp / q_sync_4to8 (PNGs on M: v2_quiet_lp/task_events_followup, v2_quiet_lp/sync_4to8). Their commits 4452139a, e7cfa946.

### 11-03 morning: Priya's answers to the open list (B1-B15)
- B1 swap-mask: yes to adding it, but FIRST discuss what changed and whether to re-do everything on the swap-masked data.
- B2 PS92 subacute outliers: asked why (quiet frames?). Answer: NOT quiet frames (RESTnl frame counts identical, hemo-independent); close-ipsi jumps in every window (pre-cue .79->3.19, cue 1.34->2.14, lick 1.24->1.98) = a session-level change in 2 sessions; per-session check offered.
- B3 merge swap-mask overrides + add a guard against session-cache pruning: YES.
- B4 precue_engagement_states: asked why a rebuild is needed (it is a side effect: the whole module is hashed into the map-cache key). Proposal: fold into one planned rebuild.
- B5 5rm: store under one seed-keyed spec, as recommended.
- B6 headline RESTnl; note sessions without all 6 positions (109/119 all 6, 9 with 5, 1 with 4) and whether conclusions differ from REST (pre-cue acute far-contra rise 3/4 -> 4/4, PS94 flips; others same).
- B7 legacy date option, or one convention for cohort 1. Fact: dates are already one convention (config.date8 = "2026MMDD" for cohort 1); the cache key moved only because hashed source TEXT changed -> fold into the one rebuild, drop the shim.
- B8 yes (flat primary + caveat). NB a time-local lick-free version is NOT formable (shortcut check §4 item 3) -> the check is against `raw`. Asked for a plot of RESTnl over epochs: exists only for RESTW (epoch_15j); RESTnl version offered.
- B9 update section 7 takeaway (-9 % PS95). B10 add PS93 caveat. B11 ok (DAQ pre-cue caveat).
- B12 1.5 s primary; lick-free ENL decoder = imaging Figs 2.5 (DAQ-trained) / 2.6 (LP-trained), 1.0 s = 2.7 / 2.8.
- B13 add 12.1b. B14 rewrite caption (and in general: replace captions, no correction notes). B15 merge branches: yes.
- New question: LP-trained vs DAQ-trained decoder for the LP-gated arm (answered in chat; decision pending).

### 11-03 ~08:00: Priya on drift / quiet guard / approvals
- Drift correction in decoders: "OK ... make primary", BUT "we should NOT be removing block-level information" -> ON HOLD until the
  cutoff is agreed. Facts: meegkit_hpfit already removes drift in preprocessing (order-10 robust polynomial, trial windows masked,
  50% cutoff ~43 min, 1-2 min block band retained 0.98-1.00); position blocks 57-121 s (pre); encoder removes >= 600 s in-model;
  the 10-20 slow-band check removed >= 300 s (only ~2.5-5x a block) so its .03-.09 may include block-level signal. Proposed:
  measure block durations per epoch + whether the spout order repeats across sessions; block-protected removal (slow basis
  fitted jointly with block indicators) at >= 600 s; one definition in the signal product, not per decoder. Discuss first.
- Quiet-event swap guard (0.5 s): set on UNREPAIRED data, where a swap is a spike and v2 measures event height against the
  preceding 1 s. On swap-masked data the frames are interpolated -> guard can shrink to +/-1 pair. Quiet-events session owns it.
- Approved: page text edits in the rebuild page pass; merges before rebuild; remove files of cut figures (goal page 512 limit);
  run RESTnl-over-epochs plot + PS92 subacute per-session check (on the corrected data, in the rebuild).
- All helpers hit the usage limit at ~07:45 (resets 11:00): inventories A / B partial; grant/epoch family-key map DONE.
- 11-03 (later): Priya: drift correction goes in the SHARED corrected signal; "10 min+ ok?" -> set from measurement (PS92/PS93
  blocks can last several minutes). Check launched (DRIFT_CUTOFF_CHECK_20261103.md): block durations per animal x epoch, spout
  order across sessions, residual band power after hpfit, block-protected cosine removal at 10/15/20/30 min (block-band
  retention >= .98 in every animal x epoch, block means intact, decoding effect). On the swap-mask variant; nothing adopted yet.
- 11-03: Inventory B complete (548 rows: 335 REDO / 190 NO / 23 UNSURE; scratchpad rebuild_inventory_B.csv + rebuild_order_B.md;
  copied into the repo at reconciliation). New traps: skip-if-exists per-session writers, --plot-only replays stored bundles,
  the hemo switch must drive ADOPTED + env/defaults + load_imaging together with a fresh frozen store.
- 11-03: "working" trial definition (jaw moved OR any camera lick; camera-lick gate = contact | incomplete) is pending Priya.
  sabatinilab-5b applies it only to its imaging section 1 table and defers the default to the rebuild. If adopted, import
  the single definition from widefield_pipeline scripts/jaw_only_incomplete_figs.classify() + its gate helper (no new copy).
- 11-03 DECISION (Priya): new default "working" = camera-defined: jaw moved OR any camera-seen lick (contact | incomplete |
  jaw_only). The DAQ quit-tail "working" becomes a robustness row. Applies in the cohort 1 rebuild (one switch, with the swap-mask
  correction, August-only pre, LP-trained LP arm, new drift step); import the single definition from
  jaw_only_incomplete_figs.classify() + gate helper. Changes frozen-decoder training sets -> new spec ids.
- 11-03: shared camera gates landed on main aca69271 (sabatinilab-5b): scripts.jaw_only_incomplete_figs GATES / session_classes /
  cue_classes (matched by cue TIME within 20 ms) / in_gate. No DAQ fallback: PS93_0813 (no video) -> None, PS92_0908 17.5% of
  cues "unknown" (in neither gate); cam4-only handled (none among curated imaging sessions). Mean share working .97 pre / .81
  acute / .88 subacute / .97 chronic. Rebuild consequence: PS93_0813 drops out of the August pre set under the camera gate
  (PS93 August pre 8 -> 7, as in the LP arm). Handoff of that work: STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md.
- 11-03 DECISIONS (Priya "ok for all"): preprocessing decks LEFT (+ one-line note: predate the swap-mask correction); analysis
  decks REGENERATED after renders; imaging section 15 body -> pointer to goal section 23 (approved removal); closed 10-27
  review page untouched; June alignment decks left; imaging 5.1/5.2 + goal 6.x/8.x -> LP-trained; lost / share-only scripts
  (imaging 0.2, 1.4, 3.3, 3.4, 3.5, 4.4, 5.1, 5.2, 11.2; goal 8.1, 8.2) recreated in the repo, each first reproduced on the
  old data.
- 11-03 Priya on within-day LocaNMF: leaning to REFIT on corrected data "for correctness", with all pre-stroke frozen
  decoders redone (already planned: fresh store). Cost estimate + joint-basis question sent to Priya; to measure first.
- 11-03: e3 repo consolidation (REPO_CONSOLIDATION_PLAN_2026-11-03.md): A1 per-session response window, A2 jaw_pass_qc -> GATES,
  A3 animal-weighted change-vs-pre (boot_delta) in epoch_figures / epoch_behaviour, A5 one responded(). Proposed to e3: land
  A1-A5 on main BEFORE the rebuild renders (all change numbers in re-rendered families -> render once). Files reserved for the
  rebuild sent to e3 (hemo_variants, defaults.yaml, combined_encoding, joint_locanmf, session_cache, map_collect,
  precue_engagement_states, frozen_models, grant_kit._class_select, config pre_from / date8).
- 11-03 (via e3, Priya's decision): ONE trial classification for everything = 5b's classify (4 classes; jaw = cam4 OR cam1 where
  each can judge; window parameter, default cue -> stop, whole-trial option for ENL). trial_attempts.py (owned by this window's
  lineage, last 58f8c14f) to be rebuilt on it by e3/5b BEFORE the rebuild renders, keeping Priya's 10-21 meanings (attempted =
  hit + incomplete; jaw_only = reaction; cue-deflected jaw without lick = unknown unless re-onset). Its consumers (failed-attempt
  decoder / maps, mixed_models.DEFICITS, attempt-coding) re-render once in the rebuild.
- 11-03 DECISIONS (Priya, on RECONCILIATION.md): D2 preprocessing decks -> REGENERATE the map slides (maps rebuilt anyway for 15e;
  new folders, never in place). D8 camera "working" gate applied EVERYWHERE a trial selection is made (behaviour / kinematics
  too); not used as a gate where engagement itself is the measured quantity (quit, engagement decline, attempt rate).
  D9 once the up-front drift step is in the shared signal, the encoder / ITI / stray_licks own drift steps are removed (kept as
  a sensitivity row). D10 (Aug-only pre per decoder render block) and D11 (DAQ robustness row on headline working arms) mine.
  LocaNMF refit: explained; proposed a footprint-stability test before deciding.
- 11-03: Priya: keep PS93_0813 wherever possible for spout-contact-only analyses (DAQ contact lick class / lick window / DAQ
  robustness rows need no video); it drops only where the camera classes are required. LocaNMF refit TEST launched
  (LOCANMF_REFIT_TEST_20261103.md): ~8 sessions refit vs projection, same-machine stability, one joint basis timed.
- 11-03: FROZEN-STORE PASS 2 DONE (wf_frozendef analysis/frozen-default, up to 876c87b0; FROZEN_ALL_20261102.md P2.1-P2.12).
  nolick / attempt / drift / coding / CD / ITI stored by default, all IDENTICAL to in-memory; LP arm default LP-trained again
  (WIDEFIELD_LP_FROZEN=daq = cross-check). Section G converted but UNCOMMITTED (PS92 lick-only arm differs, max .031: PS92 pre
  features changed between 11-02 21:23 and later with an unchanged spec -> behaviour/trial input changed; other animals identical).
  KEY FINDING: make_spec hashes pre labels + U_atlas / SVTcorr size+mtime only, so trial-selection / behaviour / gate changes
  are invisible to 14 kinds -> recommend a training-data signature (+ selection_tag) in make_spec for the rebuild's fresh store.
  frozen_encoder not converted (plan P2.6). Pre-existing bug: nolick_decoder crashes after the first animal (KeyError, arms
  < 30 trials) -> fix in the rebuild.
- 11-03: e3 A4 on main 28f2a632: trial_attempts.py classes from wfield_local.trial_classes (cue_stop); 62,746 / 62,771 trials
  identical (25 cam1-seen tongues -> incomplete); trial_classes.selection_tag(source, window) for cache keys / specs; GATES gains
  "attempted". trial_attempts.csv + readers re-run in the rebuild.
- 11-03 DECISION (Priya): YES to the training-data fingerprint. Rebuild: make_spec gains a hash of the actual training X / y
  (plus trial_classes.selection_tag and the hemo variant name) for every frozen kind; fresh store, so every ID changes once.
- 11-03: e3 A1 on main ef70a2a8 (analysis_kit.response_window_s: 3.5 s in all 120 curated; hit rate +.016 mean; quit later in 7/120)
  and A4 amendment 23c8f775 (close_center no contact / lick / tongue -> no_response; jaw_judged). A3 (paired_animal_draws for all
  change-vs-pre) coded, landing on main after tests; boot_ci untouched. Render once after merge; star flips -> Priya first.
- 11-03 interim: LocaNMF refits NOT reproducible even seeded (PS93_0826 x4: footprints r .95-.97 refit vs refit, .94-.95 refit vs
  stored; projection decode inside the refit spread). Drift replacement arm: cosine >= cutoff misses edge bleaching ramps the
  polynomial fitted (PS92_0819; 3.5% of brain-mean) -> consider polynomial + block-protected cosine, or an explicit bleach term.
- 11-03 (Priya: "lots of CPU and some memory available"): rebuild CODE PREP launched in parallel with the two checks. New worktree
  wf_rebuild, branch rebuild/cohort1-20261103 off origin/main; merges the 5 branches (not pushed); decision-independent changes
  a-k (single hemo switch, basis pin + project_all, output tagging, cache prune guard, spec fingerprint, camera working default,
  Aug-only pre per block, cache-rebuild items, nolick bug, per-session projection path, tagged preprocessing maps) + lost
  scripts recreated and reproduced on old data. Log: wf_rebuild docs/status/REBUILD_CODE_20261103.md.
- 11-03 (a8, Priya OK'd): lick-landing frozen decoder class-mix check CLEAN (frame-SD scaling over ALL frames; subsampling to the
  acute mix moves the offset +.003 / -.001 mm vs observed +.36 / +.17). KEEP frame-SD scaling in the rebuild (lick-sample z-scoring
  would create a +.21-.25 mm artefact). Caveat to state beside the cued ALL-lick acute offset: frame SD rises 1.06-1.20x acute;
  removing it gives .166 -> .119 [-.049, .281] (PS95 -> -.13); ITI and first-cued-lick offsets still exclude 0. Priya: do NOT
  undo the SD rise in production. Files decoder_mix_checks/lick/; DECISIONS fa70b651.
- 11-03: LocaNMF test DONE (LOCANMF_REFIT_TEST_20261103.md): refits not reproducible even seeded (counts 92-100 from identical
  input; joint 87 vs 103); refit vs stored ~ refit vs refit; refit - projection decode mean +.003 / +.001; refit ~11 h GPU vs
  projection < 1 h. RECOMMEND projection (per-session + joint). Awaiting Priya.
- 11-03: DRIFT CHECK DONE (DRIFT_CUTOFF_CHECK_20261103.md): block-protected FAILS (slow power x5-17, decode -.2), position-protected
  LEAKS labels (permuted null above chance 4/4), replacement worse than stacking (edge bleaching). Position CYCLE (all 6) 5-22 min
  p95 (PS92 up to 40) -> 10 / 15 / 20 min cutoffs strip 34 / 13 / 4% of PS92 cycle power; 25 min safe everywhere. Spout order not
  fixed (Kendall W <= .06; time-only decoder at chance). Label-free cos25 stacked: +.015 precue / +.007 cue, removes ~3-5% of
  10-43 min power, fit on ~21% out-of-trial frames, often ADDS > 43 min power. RECOMMEND no new drift step (option A); keep the
  encoder / ITI own steps but check the encoder at a 25 min own cutoff. PS92 subacute 2 sessions drift-dominated (26-31% of
  variance > 20 min) -> explains the PS92 subacute outliers; caption flag. Awaiting Priya.
- 11-03: imaging section 1 (5b) run order added to the rebuild; analysis/attempt-coding must also be merged into the rebuild branch
  (goal-page scripts, v8 ROIs) -> code-prep helper doing a trial merge first.
- 11-03: REBUILD CODE PREP DONE (wf_rebuild, rebuild/cohort1-20261103 tip 85af6485, not pushed; log REBUILD_CODE_20261103.md):
  5 branches merged (+ section G); items a-k + fingerprint (FREEZE_VERSION 2, every ID new; fresh store required) committed, 4,159
  tests pass; 10 recreated figure scripts reproduce their figures pixel-for-pixel (scripts/rebuild_figs/). Exposed + fixed: nolick
  joint signal mixed stored / projected C. Defaults unchanged (gate=daq, adopted variant) until flipped at render time. Finding:
  imaging 3.5 is NOT bit-reproducible across BLAS thread counts (contradicts CLAUDE.md). attempt-coding trial merge = 60 conflicts
  -> new helper resolving + wiring the remaining camera gates (fail closed) + narrowing the map key to the yaml keys used.
  Independent code REVIEWER launched in parallel (REBUILD_CODE_REVIEW_20261103.md). For Priya: camera "stopped" = no_response?
- 11-03: INDEPENDENT REVIEW of rebuild tip 85af6485 (wf_rebuild REBUILD_CODE_REVIEW_20261103.md, 8c6edf55): 4 HIGH (attempt_axes
  DAQ cross-check constant fingerprint; tagged runs overwrite local deck PNGs; attempt_maps / attempt_coding unrouted + skip-if-
  exists; camera-class inputs not in cache keys), 10 MED, 6 LOW. Verified OK: defaults unchanged, hemo switch everywhere, no stored /
  projected mixing, fingerprint on all 15 kinds, prune guard, camera lick class, merges. Imaging 3.4 no longer matches its 11-02
  original (input rewritten 11-03 by main's bal_em decoder). All findings sent to the merge / gate helper to fix; second review
  pass after.
- 11-03 DECISIONS (Priya): DRIFT option A = NO new drift step in the shared signal; final cohort 1 variant =
  hemo_meegkit_hpfit_lag_refitT_swapmask (already built, 120 sessions). KEEP the encoder / ITI own drift steps (reverses D9). They
  are one constant: combined_encoding.DRIFT_MIN_PERIOD_S = 600 s, projected out of Y and every design column (FWL), reused by
  scripts/iti_licks.py:361. Check whether 10 min is too short (position cycle 5-22 min p95, PS92 up to 40): encoder + ITI
  headline at 600 vs longer cutoffs, per animal. LocaNMF: PROJECTION (per-session + joint), no refit.
- 11-03 DECISION (Priya): camera "stopped" = no_response trials in a SUSTAINED stop. Implemented as a run of >= behavior.tail_min_misses
  (6) consecutive camera no_response trials ("unknown" skipped, never stopped); isolated no_response in neither class; "any
  no_response" = sensitivity; rule name in selection_tag. Applies to section 11 / ENL states, epoch 12s / 12b, miss_vs_stopped,
  nolick categories.
- 11-03 ~12:50 INCIDENT (checked): the encoder drift-check helper was flagged for "interfering with workloads". Its log shows: at
  4-6 GB free RAM it tried Stop-Process on its own ITI run (36004) plus two workers chosen BY PARENT PID (rule breach); the
  classifier blocked that; it then stopped its own background task (TaskStop). Only its own processes ended; other sessions'
  jobs (freeze_models, pytest, encoder, jupyter) untouched. ITI run paused at 6-7 / 107 sessions; resumes at 1 worker. Helpers
  re-told: stop only via TaskStop / exact own PIDs.
- 11-03 interim: under the ENCODER's own procedure (engaged trial rows only, gapped), slow cosines are not orthogonal to faster
  bands: cycle-band kept .71 / .78 / .82 / .88 / .95 at 10 / 15 / 20 / 25 / 40 min. Full 107-session encoder runs + ITI going.
  Possible fix to raise with Priya: fit the drift on the continuous session, apply to the trial rows.
- 11-03 DECISION (Priya, via e3; main b506c1e3): primary camera "working" also ends at the session's LAST camera lick (contact |
  incomplete): trial_classes.in_working(..., end="lick"); robustness anchors end="contact" / "close" / "none"; drops 1.0% of working
  trials (1.8% acute; jaw-only trials after the last lick). Tag "sel-camera-cue_stop-endlick". Sent to the rebuild merge / gate helper.
- 11-03: rebuild code tip f7bb4302: attempt-coding merged (60 conflicts resolved, 39 byte-identical, 9 by hand), camera gate wired /
  fail-closed with a call-site register test, sustained stopped + end-at-last-lick in cue_classes_for (tag sel-camera-cue_stop-
  endlick-stopRun6), map key = only the yaml keys builders read, all review H/M/L1-L4 fixed; 12 attempt-coding make_spec calls fixed.
  Full suite @86280b8a: 4,402 passed, 11 pre-existing parallel-ordering failures in attempt-coding scripts. SECOND review pass sent.
  Questions for Priya: no-lick trials classed contact; rest-baseline working period; empty noattempt class under camera; 7 scripts
  break the ordering rule; 3 scripts not on stored read-outs.
- 11-03: SECOND REVIEW @f7bb4302 (630c277b): suite 4,402 passed / 11 known failures; merge sound; camera gate no fail-open path;
  13 / 20 first-pass findings fixed, 4 partial, M7 regressed. New HIGH: N1 attempt readers read the old folder, N2 lp_final DAQ
  cross-check cannot find its model, N3 writeguard refuses the lp_final store; 8 MED. Sent back for fixing; third review after.
  New questions for Priya: N6 exclude PS92_0908 (82.5% video) from camera-gated cells; N7 does a jaw-only trial after the last
  lick break a no-response run.
- 11-03 DECISIONS (Priya): (1) no-lick trials classed contact -> neither class; count them per animal x epoch to see what they are
  (undetected DAQ contacts vs grooming); (2) rest-baseline working period ends at the last camera lick; (3) camera noattempt =
  isolated no_response inside the working period (lapse vs sustained stop); (4) fix the 7 ordering-rule scripts; (5) convert the 3
  in-memory decoders to stored read-outs; (6) KEEP PS92_0908 in camera gating for its 82.5% covered trials (rules computed within the
  covered span; partial coverage flagged + captioned); (7) jaw-only after the last lick does not break a stop run. Sent to the fix helper.
- 11-03 DECISION (Priya): the no-lick analysis uses the RESPONSE WINDOW (per-session, 3.5 s) for its camera classes, not cue_stop.
  Requested window="response" in trial_classes from e3 (owner); rebuild no-lick categorisation calls it; tag carries the window.
  (Correction: the "no-lick trials classed contact" are post-cue no-DAQ-lick trials, not pre-cue windows; likely late licks /
  sensor misses, being counted.)
- 11-03: checked trial_classes: "contact" = the DAQ spout-contact sensor (from cue + 70 ms), so "no-lick trials classed contact" are
  LATE contacts (after the no-lick response window, before trial stop), not sensor misses (Priya: sensor misses unlikely). Under the
  response-window rule they should vanish; residuals = mismatch between detect_licks and the contact table, to be reported.
- 11-03 DECISION (Priya, supersedes the response-window rule): classifier AND no-lick analysis both count to the TRIAL STOP (cue_stop;
  contacts from cue + 70 ms). Trial stop (v36 response_end_ms) >= 3.5 s, usually 3.525 s, mean 3.7-4.0 s, max ~7 s (PS92 / PS93 / PS94
  sessions checked) -> helper reports trials > 3.6 s per animal x epoch (trial-length differences matter for licks-per-trial).
  e3's window="response" request cancelled.
- 11-03 (Priya via e3): LMM hidden-slide versions of 14 wf_attempt PI-deck figures (analysis_kit.epoch_change_lmm + lmm_stars, Holm
  across the figure; scripts/lmm_change_fig.py on main). Helper building them on CURRENT data with per-session extractors
  (scripts/lmm_hidden/, re-runnable on rebuild outputs) -> grant_figures/lmm_hidden_20261103/ + README (star differences vs
  bootstrap). Tell a8 when they land; tell e3 when the rebuild's *_sessions.csv exist (slides 10 / 11 / 12 / 19).
- 11-03: imaging page LIVE v50 (published by sabatinilab-e0 under Priya's HMS account): section 1 on camera-defined working up to the
  last camera lick; Fig 1.4 = a8's campllick re-plot. e0 can publish any window's pages if this window's Artifact tool is off.
  Read v50 live before any imaging-page edit in the rebuild page pass.
- 11-03: LMM HIDDEN-SLIDE FIGURES DONE (grant_figures/lmm_hidden_20261103/ + README; code wf_lmm_main branch lmm-hidden-20261103,
  053dfb76 / e677657b / 09d767e9, not pushed; extractors re-runnable with --root). All 14 extractors reproduce the published per-cell
  changes exactly. Under the LMM ~half of the bootstrap stars drop, incl. caption headlines: rec_wlc csm_share (all epochs n.s.),
  rec_kn chronic norm (p .14), spec_groups acute residuals (p .065-.16); kept: acute ITI / cued landing offsets, first-lick acute /
  subacute, landing slopes, decoder_retention failed-attempt cells (now **). Curve shapes: cued_ramp_summary_row2 + iti_ramp_row1
  made (pre-specified AUC); gauss_curves / lick_order curves not time courses -> not made. a8 + e3 told.
- 11-03: rebuild code tip 5c17d76a: second-review fixes + Priya decisions 1-7 (classify_session); FULL SUITE 4,441 passed, 0 failed
  (ordering failures fixed). Check on 3 sessions: undetected-but-contact disagreement 0 / 1 / 0; trial stop > 3.6 s on most trials
  (median stop 3.66-3.77 s). Remaining before render: ~44 older unrouted writers, ITI run labels, cache-key gaps, attempt readers,
  a single render profile (variant, LocaNMF projection, camera end=lick stopRun6, Aug-only pre, LP-trained, fresh store) ->
  helper; THIRD review pass launched in parallel.
- 11-03: LMM figures re-drawn through main's fixed lmm_change_fig (9a820fbb) + 9 slide-layout splits (_a/_b, _cued/_cue/_iti), stats
  byte-identical (sha256), each image checked against its _lmm.csv; Holm stays the whole figure's family. Branch lmm-hidden-20261103
  (09b8feae merge main, 07e2b8de, 73732319). File names sent to a8.
- 11-03: THIRD REVIEW @5c17d76a (62d1dbc4): suite 4,435 passed / 0 failed. HIGH T1 attempt writers refused under LP, T2 ITI refused
  in non-data modes, T3 decision 7 labelled jaw-only trials themselves "stopped" (stopped +62% cohort-wide; PS94_0825 run = 12 no-resp
  + 28 jaw-only) -> implementing as approved: jaw-only after the last lick BRIDGES but is not stopped and not counted; >= 6 no_response
  needed. MED T4-T10 (empty RESTnl maps, tagged nightly overwrites deck, reader/writer mismatches, coverage edge cases, s1 second
  classifier). Provisional: no-response runs before the first lick = warm-up, neither class (to confirm with Priya). Fourth review after.
- 11-03: THIRD-PASS FIXES DONE (rebuild tip 85786d3b; code 3ef70b5b): T1-T10 + LOW; DECISIONS.md "2026-11-03 (night) -- Camera trial gate"
  records all of Priya's camera rules; suite 4,476 passed / 0 failed. T3 effect (119 sessions): 513 trials stopped -> after_end (stopped
  2,339 -> 1,830); working / camera-lick sets unchanged. Warm-up (provisional): 4 trials in 4 sessions. Partial coverage: PS93_0813 none,
  PS92_0904 377/378, PS92_0908 321/389. Contact / no-lick disagreements all late licks -> 0 under the cue_stop rule. TRIAL LENGTH: acute
  only 53-68% of trials > 3.6 s vs 86-92% pre -> licks-per-trial readouts confounded by trial length (decision for Priya). Run tag:
  hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_sel-camera-cue_stop-endlick-stopRun6. Tables copied to rebuild_20261103/. Fourth review sent.
- 11-03: FOURTH REVIEW (a1b6064a): T1-T9 hold in real modes; profile tag verified; suite 4,470 / 0 failed (6 git-history tests skip in a
  clean export). NOT SAFE TO RENDER YET: R1 Fig 12.1b stale + default overwrite, R2 tagged deck reads untagged figures, R3 rebuild_figs
  read old inputs by default, R4 attempt_maps silently partial, R5 refusals ignore LP / ENL modes and the profile does not clear stray
  env vars; ~120 more unrouted writers in subfolders / wfield_local. Switched to a STRUCTURAL fix: one rebuild root for every derived
  output under the profile, writeguard refuses writes outside it, readers resolve inside it (raw inputs untouched). Fifth review after.
- 11-03: fourth-pass ADDENDUM (0fc49ee8): T3 rules verified correct (jaw-only bridges but not stopped / counted; >= 6 no_response; warm-up;
  MIN_UNCOVERED_RUN 5; rest ends at last camera lick); DECISIONS entry matches the code (3 behaviours to add). PS94_0825 stopped 40 -> 12,
  PS93_0925 86 -> 38 (only the bridging jaw-only trials move). Correction: the moved trials are mostly the jaw-only bridges themselves, not
  runs below 6. Remaining blockers R1 / R2 / R3 / R5 being fixed via the rebuild-root design.
- 11-03: ENCODER / ITI DRIFT CUTOFF CHECK DONE (ENCODER_DRIFT_CUTOFF_20261103.md; outputs grant_figures/encoder_drift_cutoff_20261103/).
  600 s reproduces the 11-02 swapmask encoder exactly. No cutoff reaches cycle retention >= .98 everywhere under the encoder procedure
  (gapped engaged rows leak); 600 s strips up to 82% of PS92 subacute cycle power. Headlines keep sign / counts / significance at every
  cutoff: target unique chronic -.013 -> -.012 (0/4), spout contra - ipsi +.018 -> +.021 (4/4) at 1500 s. Longer cutoffs bring back
  PS92_0824's drift failure (frozen R2 .44 -> .19 at 1500 s). ITI baseline-subtracted readouts unchanged (max .014); only the pre-cue
  level moves. RECOMMEND keep 600 s; add 1200 / 1500 s as a sensitivity row for spout contra - ipsi + ITI level readouts. Decision: Priya.
- 11-03 DECISIONS (Priya): encoder / ITI drift = keep 600 s + a 20-25 min sensitivity row (spout contra - ipsi, ITI level readouts).
  Licks per trial = TOTAL licks from cue to trial stop while the spout is present; lick RATE is a separate metric (reported alongside).
- 11-03 DECISIONS (Priya): task rule = after the first lick the animal has 3.5 s; no lick -> trial ends at cue + 3.5 s (so trial length
  = first-lick latency + 3.5 s). Licks per trial = total cue -> trial stop; lick RATE from inter-lick intervals, NOT count / time.
  "Stopped" runs must start >= 5 min after the first cue (replaces the warm-up rule); checking that no current stopped run moves.
  LMM captions: main slides keep their bootstrap captions; each LMM hidden slide carries the LMM results + its own LMM caption (a8 asked).
- 11-03: a8 DONE: the four LMM hidden slides carry an on-slide bold LMM caption (estimate [95% CI], p, df = 3; "main-slide claim =
  bootstrap"): rec_wlc csm_share -.13 / -.15 / -.14, p .13 / .13 / .17; rec_kn chronic norm -.10, p .14 (acute .052); recovery_timing
  acute gap +.17, p .21; spec_groups acute residuals p .065-.13 (on spec_groups_b). Main captions untouched. Script lmm_not_sig_captions_1103.py.

## 11-03 EVENING (Priya away): questions, findings, decisions, running -- newest block, read this first

### Questions for Priya
- REST PERIOD START: with the warm-up rule replaced by the 5-min stop rule, should the rest-baseline working period still start
  at the session's FIRST camera lick? RECOMMEND yes (pre-first-lick time is not working; keeps the rest baseline unchanged).
  Implemented as a named setting.
- Lick metrics NOT moved to cue -> trial stop (need your call): session licks-per-minute curves (a session-time rate, different
  quantity), enl_lick_rates (pre-cue ENL), FaceRhythm fixed-window licks. RECOMMEND leave all three (none is "licks per trial").
- Slide 94 text: acute cued rise carried by 2 animals (PS94, PS95); keep or add that caveat? (see 95-98 entry)
- PI deck slide 34 (far contra attempts) has NO tests: a8's fig_farcontra_attempts_sessions.py is descriptive only (made to replace
  the per-animal table). RECOMMEND: add paired animal-weighted change-vs-pre stars in the top band + an LMM hidden slide, drawn in
  the rebuild from the new camera-class trial_attempts_sessions.csv (the current inputs are the pre-rebuild v36 tables).
- Pre-cue maps on PI slide 24 (epoch_15r precue, RESTW ref) are the 1.0 s lick-free window; B12 made 1.5 s primary for the LP ENL
  arm, but NO 1.5 s map render exists, and the rebuild profile does not set WIDEFIELD_ENL_WINDOW_S=1.5 (LP_ENL_WINDOW_S = 1.0).
  RECOMMEND: the rebuild sets 1.5 s for the LP precue arm (decoders AND maps), 1.0 s as the labelled sensitivity. Confirm B12 covers maps.
- PI slide 35 "no response": denominator = DAQ-engaged trials (orofacial_cohort.engagement_keep = behaviour `engaged`: sated tail
  >= 6 trailing misses + rolling-rate collapse removed). So the quit tail is out by the DAQ rule; DAQ misses include incomplete
  (no-contact) licks, so acute far-contra runs of trying can be cut. In the rebuild the share figure's denominator must be every
  trial INSIDE the camera working period (to the last camera lick, stop runs out) -- NOT in_gate("working"), which excludes
  no_response by definition. (answered Priya 11-03 evening)
- PI slide 43 (one-panel LMM hidden) too wide: asked a8 to re-render with --figsize ~6x4.5, place ~6-7 in wide.
- PI slide 52 (3-D lick phase, kinematics_pooled_figs fig_phase): NO test drawn; no sharey (per-panel y blows up near-spout noise),
  phase 0-1 incl. noisy ends, all post-cue licks (no engagement gate; slide 46 is engaged), >= 3 animals per bin, no smoothing.
  Asked e3: add the 2-D phase cluster-permutation bars, sharey per row, 0.1-0.9, engaged licks, figdata / --plot-only.
  e3 DONE ceb8d2b4 (test bars, sharey, 0.1-0.9, smoothing, animal lines, figdata); a8 has the path. curves.csv.gz has no engaged
  lick set -> REBUILD LIST: an engaged / working-trial 3-D phase lick set. Kin page Fig 9.7 caption ("engaged") rewritten by e3.
- PI slide 59 (goal Fig 14.18 S21c paths): x axis flipped, contralesional on the LEFT (c217ae07, within_lick_correction.py,
  per-animal Fig 5 too); goal-page copy updated locally (page publish at the page pass); a8 asked to re-crop.
- PI slides 65 vs 66 (landing change matrices): both "epoch minus pre within animal, mean of 4", same licks / width 1, but cells
  differ by .01-.05 (largest close ipsi own -.19 vs -.22, close contra own -.19 vs -.24, cI->fI +.26 vs +.30; far contra own
  -.68 vs -.69). The per-animal means on 65 reproduce its "mean of 4" (fC own: mean(-.63,-.54,-.80,-.75) = -.68), so 65 = each
  animal's licks pooled over sessions; 66 is the bootstrap's point estimate, most likely session-weighted (mean of per-session
  matrices, as its trajectory twin lick_path_softmax.boot_change does, probably with a >= 10 licks per row floor). NOT confirmed:
  the landing script is not in the repo (not in rebuild_figs either; lost scratch). REBUILD LIST: recreate it; one estimate
  (session-weighted, = the test) on both figures.
- PI slides 95-98 (LMM hidden, cued / ITI lick-ramp AUC): MY ERROR. They test AUC 0-0.5 s PER SPOUT (126 cells), taken from the
  per-spout supplement's window via a DECISIONS 11-03 row ("the response AUC 0-0.5 s already used on lick_cued_ramp_perspout"),
  while main slide 94 is AUC 0-1.0 s / pre, spouts POOLED (Priya 10-28 evening: "0-1 s should be default now"; RESP_W, d57b6098).
  Helper redoing both on the main slide's quantity (new files, old kept as per-spout supplement); then a8 swaps; the DECISIONS row
  gets rewritten (e3).
- PEAK vs AUC (ramp_window_tests.csv, spouts pooled, change vs pre): CUED -- AUC 0-0.5, AUC 0-1 and peak give the same picture
  (the same 18 of 21 cells significant under all three; median CI width .35 / .32 / .30). ITI -- peak CIs are much tighter (median
  .71 vs 1.92 for AUC 0-1) because the 0-1 s AUC integrates the slow ITI tail; peak flags ALM acute / subacute where AUC 0-1
  does not. RECOMMEND keep AUC 0-1 primary (pre-decided; it captures DURATION -- the acute far-contra response reaches the pre
  peak but decays sooner, which a peak cannot see), peak as a labelled sensitivity. Caveat on peak: the max of a noisy PETH is
  biased up, more so with fewer trials (acute sessions have fewer licks), so it can inflate acute changes.
  DECISION (Priya 11-03 evening): regional response sizes = AUC 0-1 s PRIMARY, PEAK (max of the session PETH 0..1.5 s / pre)
  SECONDARY. Now: the slides 95-98 helper also makes _peak LMM companions. REBUILD LIST: every regional response-size figure
  (goal 14.10 / 14.11 response sizes, 14.23 row 2 cued, ITI response summary) gets a labelled peak row / supplement beside the
  0-1 s AUC; caption the peak's small-n upward bias. DECISIONS entry to add at the rebuild merge.
- PI slide 118 (lick order, cued row): Priya wants an ITI backup (landing offset + decoder offset by lick number). The panels exist
  (top row of goal Fig 13.2, docked ITI; whole figure already on slide 177 "Backup: lick-order curves"). Asked a8: hidden backup
  right after 118 with the ITI row only, and fix 118's bullet / notes, which name a non-existent backup title. Pending the deck
  being closed (a8: open in a PowerPoint window since 18:27, likely Priya's).
- PI DECK UPDATED 18:50 by a8 on Priya's saved file (her hide / unhide edits kept; backup backup_priya_saved_1850_1103.pptx;
  hidden-state diff before / after: only the new slide changed). Applied: slide 59 paths contra-left; slide 52 p_phase with
  permutation bars; natural-width one-panel hidden slides 12 / 18 / 43; NEW hidden slide 119 "Backup: lick order, ITI bouts"
  (slide 118 bullet repointed). Deck now 183 slides (slides after 118 shift by +1). Waiting: 95-98 LMM 0-1 s (+ peak) swap.
- Slides 95-98 REDONE (4cc7cd8d, lmm-hidden-20261103): slide 94's quantity (spouts pooled, AUC 0-1 s / pre), reproduced exactly
  (cued 749 rows, docked ITI 714 rows; ramp_window_tests 21/21). LMM (df 3, Holm over 21): cued AUC 8 / 21 (subacute + chronic
  contralesional tjM1 / SSp-m, NO acute cell, p .08-.40); cued peak 11 / 21; docked ITI AUC 0 / 21 (smallest p .07); ITI peak
  8 / 21 (acute ALM / tjM1 / SSp-m contra, cortex; chronic tjM1 / SSp-m). Bootstrap on the same numbers: cued 18 / 18, ITI AUC 9,
  peak 17. Per animal: the ACUTE cued rise is PS94 + PS95 (PS95 one session), PS92 ~0; ITI ratios unstable (PS94 small pre AUC ->
  +8 to +10). Sent to a8 (95 cued AUC, 96 cued peak, 97 ITI AUC, 98 ITI peak); main slide 94 unchanged.
  FOR PRIYA: slide 94's "larger after stroke" holds subacute / chronic under the LMM but the ACUTE part does not (2 animals);
  the ITI AUC does not survive the LMM at all, the ITI peak partly does. Main-slide text unchanged pending you.
  a8 SWAPPED 95-98 in the deck (backup backup_before_ramp_w01_peak_1103.pptx; hidden states unchanged, 78 / 183; slide 94 untouched).
- CODE HELPER round done (rebuild branch f58c5d86, suite 4,517 passed / 0 failed; log 8.10-8.12):
  - 5-MIN STOP RULE CHECK: 0 of 159 stopped runs (1,830 trials, 119 sessions) start < 5 min after the first cue (earliest 46 min,
    PS95_0820) -> Priya's condition met; rule being IMPLEMENTED (warm-up class removed; its 4 trials become isolated lapses).
  - Licks per trial = cue + 70 ms -> trial stop (DAQ trial_end TTL; fallback first contact + 3.5 s / cue + 3.5 s); rate = 1 / median
    ILI. Moved: spout_behavior, engagement_decomposition, quit_prodrome, lick_bout_structure (were cue -> next cue, incl. ITI licks).
  - Trial-stop check (6 sessions): hits stop - first contact median 3.528 s [3.39, 3.72]; misses stop - cue 3.525 s. 49 hits > 250 ms
    off (mostly stop EARLY: the task timer started on a lick before the first DAQ contact after cue + 70 ms).
  - Rebuild root + fence + R1-R5 + 1500 s drift arm done. Next: merge main (helper) -> fifth review -> rebuild_root init -> render.
- CODE HELPER: 5-MIN RULE + MAIN MERGE DONE (rebuild tip a7b95785; suite 4,550 passed / 178 skipped / 0 failed; log 8.13-8.15).
  - 5-min rule (c23a7aeb): stop = >= 6 no_response run STARTING >= 300 s after the first cue (trial_selection.stop_min_start_s);
    an earlier run = new class early_run (neither working nor stopped). Warm-up removed (its 4 trials = isolated lapses); no
    stopped run moved. Rest start = named setting trial_selection.rest_start: first_lick (question above still open).
  - Merge of main e8491dc8 (698c4e57 + fixes c6f9be2d): all 9 conflicts kept both sides. Every stored pre-stroke model is now
    fitted in the bal_em frame (FrozenFrame store= hook; spec carries scaling + train-data fingerprint); camera-gate masks kept
    everywhere. in_working on-hold item = option (b), nothing extra needed. A5 lands with this branch (cue_classes_for).
- QUESTION (method, decided by default -- tell me if you disagree): the bal_em EM centre was fitted on ALL DAQ-engaged rows of a
  session (main's design), so under the camera gate it included rows the camera calls after_end / unknown. Your 11-03 rule is
  "camera gate everywhere", so I asked the helper to fit the centre on the camera WORKING rows only (named setting; the
  DAQ-engaged centre kept as the alternative). Still label-free, still never no-lick rows.
- GAP being closed before the review: june_august_drift's stored month model was the session_z fit; under bal_em it fitted in
  memory (breaks "every pre-stroke decoder is frozen"). Helper storing it in the bal_em frame.
- BOTH DONE (rebuild tip 642d7b3b, log 8.16-8.18; run tag unchanged): bal_em.centre_rows auto (= working under camera gate,
  engaged under DAQ gate) | working | engaged; fails closed if a site has no camera classes; sessions with < 2 working rows
  (no-video, e.g. PS93_0813) fall back to engaged rows, logged; centre set in the stored-model spec id. june_august_drift month
  model now stored. Suite 4,556 passed / 2 source-text failures fixed in 642d7b3b (full re-run is part of the review).
- FIFTH REVIEW sent (reviewer aa0138ac1e9c354a7) on 642d7b3b. Next: review -> rebuild_root init -> render G0-G8.
- MERGE REVIEW (scope C, a38758e5) on 642d7b3b: 1 BLOCKER -- four stored pre-stroke kinds (nolick_reference, coding_decoder,
  engagement_states posdec-ff, g_lickonly -ff) key their leave-one-session-out models by pooled INTEGER id, not label; under the
  DAQ-trained LP cross-check the ids shift (LP pool lacks PS93_0813) so PS93 pre sessions get scored by a model trained on them
  (silent leak; proven by probe). Same hazard latent in the default arm for nolick (N1). Non-blockers: two except blocks swallow
  the camera fail-closed raise (N2), dead in-memory fits (N3), june_august_drift size-matched draws still in memory (N4), memo
  keys miss gate / centre (N8). Merge conflict resolutions otherwise all OK; A3 draws respect ordering rules. Sent to the helper.
- FIFTH REVIEW (partial): full suite at 642d7b3b 4,552 passed / 0 failed; profile --check and drift1500 arm tags correct and
  separate. Confirms the same B1 + N1-N8; N5 (A3 path-argument scripts outside the writer register, no run-tag check) added to
  the helper batch. STILL OWED: early_run consumers / 5-min rule (Q3) and centre_rows / stores / < 2-row fallback (Q1, Q4).
  Render waits for B1 fixed + those two checks.
- FIFTH REVIEW, scopes A + B DONE: early_run read as neither working nor stopped everywhere (Q3 OK); licks-per-trial window and
  5-min rule verified on real sessions (159 runs / 1,830 trials match). New non-blockers sent to helper: GUI-fallback sessions
  (08-05, 08-06, PS92 08-12) cut licks/trial at 2.0 s not the stop (behaviour writers refused under the profile, so not in the
  render, but the pre lpt__ baseline mixes windows); rest_start missing from cache keys; a snapshot script bypasses the 5-min rule;
  edge tests; "licks / trial" labels; one shared 70 ms constant. Still owed: Q1 / Q4 (centre rows, stores, fallback).
- FIFTH REVIEW, scope D + E DONE: 3 more BLOCKERS, sent to helper -- (1) section G silently refits a pre-stroke decoder in
  memory, session_z frame, untagged, on ANY store error (poststroke_compare frozen()); (2) the "fail-closed" centre errors are
  swallowed by two generic excepts; (3) the PS93_0813 centre fallback was log-only. My decision (method; tell me if you
  disagree): PS93_0813's centre = its DAQ contact (lick-class) rows, the rows the gate actually scores for it; recorded in the
  spec id, every result sidecar and the log; minimum raised to 20 working rows (all other sessions have >= 20). Store identity
  OK (no two settings share an entry unless training rows identical); june_august_drift month model stored OK.
- QUESTION (open): PRE_FROM 0801 is only passed to the two decoder blocks; asked helper to list other G0-G8 families that would
  use June + August pre, before deciding whether the profile needs more block_args.
- PRIYA ASKED (11-03 late): is PS93_0813 (no video) still used for DAQ-contact-only analyses, e.g. hits? Answer: lick-aligned
  imaging YES (lick class = DAQ contacts); hit rate / miss / no-response / cue-aligned working / rest NO (denominator = camera
  working trials, none; no DAQ fallback by design). Recommended keep; alternative = DAQ-gate denominator for that session's
  DAQ-only behaviour rates, labelled. Waiting on her choice.
- HELPER BATCH DONE (rebuild tip c86be4a0; suite 4,572 passed / 1 failed at a06f7fb7, fixed by a rename in c86be4a0, full re-run
  pending; log 8.19-8.22): all review blockers fixed (LOSO keyed by label; section G refuses instead of refitting; centre errors
  can't be swallowed; PS93_0813 centre = its DAQ contact rows, recorded in spec / sidecars / captions / log, min 20 rows) and
  the non-blockers (A3 scripts check run tag; GUI-fallback licks/trial uses the session window; rest_start in tags; edge tests;
  labels; one 70 ms constant; last trial included).
- DECIDED (mine, method): lick COUNTS use one detector (detect_licks, yaml params -- the trial table / camera contacts one);
  two count consumers used a second detector (2.5 / 1.0 V, 100 ms refractory): PS93_0908 differs on 101 / 511 trials by 1 lick.
  Imaging lick alignment keeps its deliberate 0.1 s refractory.
- QUESTION FOR PRIYA (blocks the render): August-only pre is passed only to the two decoder render blocks. A source scan finds
  33 other scripts (rest arm, transfer / rotation maps, locomotor decoder, grant encoder, CD trajectories, chronic stability,
  lick ramps ...) that would take June + August pre. Make Aug-only pre profile-wide (except june_august_drift and 12.1b, which
  pick their own)? RECOMMEND yes (rebuild plan: "Aug-only pre ... aligning with the imaging maps, encoding model and
  kinematics"). Helper checking each at run time first.
- LP blocks: the profile gets per-block WIDEFIELD_LICK_SOURCE=lp; three LP writers will refuse without it.
- HELPER ROUND DONE (rebuild tip 3a772a26; FULL suite 4,576 passed / 0 failed at 8b67097d; log 8.23-8.25): lick counts on the one
  detector (lick_count_samples; imaging alignment + engagement-gate response detection unchanged); profile block_env sets the LP
  lick source per block, LP_licks writers refuse without it. Reviewer re-verifying all fixes on 3a772a26.
- PRE-SET CHECK (run time, under the profile): pre = 44 sessions incl. 12 June; Aug-only = 32. 27 scripts would use June + Aug
  pre (12 non-rest incl. grant_encoder, locomotor_decoder, cd_trajectories, chronic_stability / drift_vs_reorg, lick_iti_prelick;
  15 rest_migration incl. rest_frozen_decoder, rest_coupling, transfer_arms, rotation_maps, quit_point, flatpool_vs_restw);
  4 inherit their upstream tables' labels (lick_cued_ramp, lick_code_noisematch, reference_family_figure, face_prelaunch_aim).
  Profile-wide = WIDEFIELD_PRE_FROM 0801 in the profile (adds pre0801 to every tag) + block_env able to unset it for
  june_august_drift and 12.1b. Epoch boundaries unaffected. STILL WAITING ON PRIYA.
- RE-VERIFY at 3a772a26: NO BLOCKERS; LOSO keying (0 leaks), section G refusal, centre refusal, 20-row minimum + PS93_0813
  fallback in spec id, cache keys -- all confirmed by probes. Non-blockers sent to helper: disengagement_tests had moved hit /
  latency to the count detector (would shift goal S28); sidecars list configured not used fallback; profile --check ignores
  --block and unknown blocks pass silently; one memo key. Plus a dry count of working engaged imaging rows per session (the
  20-row minimum was checked on cues, not rows) before the render.
- PRIYA (11-03 late): (1) Aug-only pre profile-wide -- "yes i guess so". Helper: PRE_FROM 0801 in the profile, unset for
  june_august_drift + 12.1b, tags updated, DECISIONS line. (2) PS93_0813 in hit rate: "why can't we use?" -- answered: it can,
  with a DAQ-defined working period (last DAQ lick, same stop rule on DAQ no-response) for the DAQ-only behaviour rates only;
  cost = one session on a different gate definition. Helper building it + an agreement check (camera vs DAQ fallback on every
  Aug pre session with video); wire it in only if they agree (~1-2 points) and Priya confirms.
  PRIYA: "yes if so let's use it" -> criterion set: <= 2 points (hit, no-response) and <= 20 ms (median latency) on every Aug
  pre session with video; pass = wired automatically, fail = back to Priya with the table.
- PRIYA ASLEEP (11-03 ~23:30): "progress without me ... running list of changed results, new findings, decisions, questions".
- DECISION RECEIVED via a8 (Priya 11-03 late, main 39bb3d5e; evidence widefield_pipeline JAW_ONLY_CHECKS_2026-11-03.md):
  JAW-ONLY OUT OF WORKING. Working = contact + incomplete, cut at the last camera lick; jaw_only its own class like
  no_response; applies to frozen training, bal_em centre, pre panel, scored set. Checks: jaw-only trials decode like
  no-response (PS94 / PS95 cue .21 / .29 vs none .20 / .25 vs contact .93 / .94); adding them to training does not move the pre
  reference (<= .01) but centring acute sessions on them lowered every spout (PS95 acute close ipsi .832 -> .642).
  Rebuild helper implementing it first (new selection tag; validation vs the checks' row: pre .856, acute cI .727).
- QUESTION FOR PRIYA (default chosen, reversible): stop runs with jaw-only in them. Default = jaw_only BRIDGES a run of
  no_response but does not count toward the 6 and is not labelled stopped (the old after-last-lick rule, now session-wide).
  Alternative: jaw_only counts as a non-response toward the 6.
- Told a8: no current PI-deck figure from this window uses in_working with jaw; the rebuild replacements will move (per-slide
  old vs new list at the replacement pass).
- a8: will NOT run a main-branch render tonight; holds PI slides 10-13 / 19 / 89 for the rebuild replacement pass (asks Priya in
  the morning whether she wants an interim swap; will tell us before any render). a8's jaw-only helper on main is light, capped.
  OWED to a8: per-slide old-vs-new list at the replacement pass.
- 11-04: jaw-only-out landed on main (48f2cc6d; WIDEFIELD_WORKING_TRIALS in_working / in_working_jaw / quit_tail; tag
  cls-contact-incomplete; 20 new frozen ids in main's store). Rebuild helper told to MATCH main's names / tag / stop-run
  semantics (and adopt main's stop-run rule if it differs from my default). Box free of a8's jobs.
- HELPER (rebuild tip a8791963; FULL suite 4,588 / 0 failed; log 8.27-8.32):
  - Aug-only pre PROFILE-WIDE done (june_august_drift + 12.1b unset it; ITI extracts keyed by pre set).
  - Jaw-only-out MATCHES main 48f2cc6d (working_trials.py byte-identical; tag ...-endlick-cls-contact-incomplete-stopRun6;
    in_working_jaw = robustness arm). Main has no stop rule, so my default stands (jaw_only bridges, not counted) -- question
    for Priya stays open.
  - Dry count: every session >= 174 working engaged rows (min PS94_0819 acute) except PS93_0813 (0, contact fallback).
  - RESULT FOR PRIYA: the PS93_0813 DAQ-fallback AGREEMENT CHECK FAILED -> NOT used. On 31 Aug pre sessions with video the
    DAQ-defined working period reads a HIGHER hit rate on 9 sessions (+2.2 to +15.9 points; PS95_0806 +15.9, PS95_0812 +8.2,
    PS94_0812 +4.4, PS93_0814 +3.9), latency identical. Why: camera incomplete / jaw-only trials are DAQ misses, so DAQ miss runs
    get longer and the DAQ stop rule cuts misses the camera keeps. PS93_0813 stays out of hit / miss / no-response rates.
  - BLOCKER FOUND (mine to fix, sent): the rebuild trained / centred / scored on DAQ-contact rows only, so camera-INCOMPLETE
    trials never entered; Priya's rule (and main) is contact + incomplete. Helper mirroring main's row construction, then a
    two-arm validation: rebuild code on the checks' old inputs must reproduce .856 / .727 (code check); profile minus that =
    the data effect (first try on rebuild data, contact-only rows: pre .876, acute close ipsi .839).
- BLOCKER FIXED (rebuild tip 6a85304d; FULL suite 4,593 / 0 failed; log 8.33-8.34): cue / precue frozen decoder trains,
  centres and scores the camera contact + incomplete rows (DAQ-engaged AND no-lick arms), as main. Acute adds 1,324 no-lick
  working rows cohort-wide. Dry count: min 224 rows (PS92_0812 pre); only PS93_0813 has 0 (neither trained nor scored in
  working selections; still in lick-aligned).
- VALIDATION (finding): rebuild code on the jaw checks' data reproduces their "jaw as its own class" row to .001 in every
  animal (pre .856, acute close ipsi .727) -> no code difference. DATA EFFECT of the rebuild inputs (swap-mask hemo + LocaNMF
  projection): pre panel +.004 (.856 -> .860); acute close ipsi +.007 mean (.727 -> .734) but mixed per animal (PS94 +.10,
  PS93 -.05, PS92 -.01, PS95 -.02). The earlier .839 was the contact-only rows + trial pooling, not data.
- Final pre-render review sent (reviewer aa0138ac1e9c354a7) on 6a85304d.
- REVIEW 7A (working rows) at 6a85304d: 2 BLOCKERS, sent to helper. (1) Fig 5b would still be drawn from coding_direction.json,
  whose decoder is trained / scored on DAQ-engaged rows (incl. jaw_only / after_end / unknown and PS93_0813) -> the working
  path never ran. MAIN HAS THE SAME SHORTCUT (main grant_confusion :332) -> tell a8 (main's 5b is not jaw-only-out either).
  (2) pooled_frozen_loso / joint_xsession and section G scored / pre-panel sets still engaged while the model is working-
  trained. Non-blockers: grant similarity / geometry / encoder / matching pre reference = DAQ lick rows; "working" captions
  stale; PS93_0813's exclusion silent. Spec ids verified distinct for in_working vs in_working_jaw; quit_tail refused under
  camera; jaw_only bridges stop runs as intended. Validation was cue-only, in-memory: helper extending to precue + stored path.
- REVIEW 7B (Aug-only pre / profile blocks): profile --check tags correct for primary, drift1500, LP and June blocks; typo
  blocks now raise; every decoder / encoder / CD / ITI family is Aug-only under the profile. 3 BLOCKERS sent to helper:
  june_august_drift (Fig 4.3) refuses to run under ANY profile tag (its June exemption was dead); map_drift_prepre (Figs 7.2 /
  7.4 / 7.6) has no June block; reembedding_test silently DROPS its June reference epoch under Aug-only. Also: the analysis
  deck reads the untagged epoch folder and skips missing files silently (section I would be empty) -> fixing before the pass.
- CORRECTION OF MY 7A INSTRUCTION (a8 + main DECISIONS 11-03): main deliberately keeps the section D / G SCORED sets as
  engaged / all trials (only the frozen model + frame are working) and the pattern families' PRE reference as lick trials.
  Main's 5b already has the camera guard (my "main has the same shortcut" was wrong). Rebuild now FOLLOWS MAIN on D / G /
  pattern families (captions to say so); 5b gets main's guard.
  QUESTION FOR PRIYA (a8 is asking it too): should the D / G scored sets and the pattern families' pre reference also be
  camera working (contact + incomplete), i.e. "working on both sides" everywhere? Both branches would change together.
- 7A + 7B FIXED (rebuild tip 11817e27, code cf94e7c3, log 8.35): 5b always on the working path (main's guard); D / G / pattern
  families documented as main's design (corner stamp + sidecar + captions); PS93_0813's exclusion logged + in sidecars;
  Fig 4.3 / 7.2-7.6 / RESTnl CLI run in June blocks; reembedding keeps June; deck reads the tagged folders and counts MISSING.
- VALIDATION (finding, clean export, stored path): rebuild code on the jaw checks' data reproduces cue AND precue to .001
  (cue pre .856 / acute cI .727; precue .558 / .568). DATA EFFECT of the rebuild inputs: cue pre +.004, acute cI +.007;
  precue pre +.006, acute cI -.017 (per animal PS92 -.14, PS93 .00, PS94 +.05, PS95 +.02) -> small on average, PS92 precue
  acute close ipsi is the one sizeable move (one animal).
- Focused re-verify + full suite sent to reviewer. If clean -> rebuild_root init -> render.
- REVIEW 8A (part of re-verify): 5b / captions / _collect_7 / PS93 log FIXED; 1 BLOCKER, a regression from the NB4 fix --
  section D (pooled_frozen_loso / joint_xsession) would crash for PS93 cue + precue because PS93_0813 (pre, no video) gets
  treated as a post session with 0 working rows. Sent to helper with small non-blockers (jaw-arm titles, design stamp, sidecar
  wording, env-sensitive test).
- REVIEW 8B: map_drift_prepre / restnl_mask blocks, reembedding June, 12.1b compare FIXED; --check tags all correct. BLOCKER:
  Fig 4.3 (june_august_drift) still can't render -- it used the input labcams root, and the recovery CSV it needs has no tagged
  writer. Deck: LP deck mislabels the DAQ-ENL1s section K figure; sections H / I still silent when a whole family is missing;
  section K legends carry pre-rebuild numbers (marked for the replacement pass). Sent to helper.
- 8A + 8B FIXED (rebuild tip 78e60f5a, log 8.36; suite 4,612 passed from root and from tests/, 1 scanner-list failure fixed
  in 2ddcf9fb): pre / post decided by session phase (PS93 section D crash gone, tested); Fig 4.3 + recovery CSV + chronic_stability
  on tagged paths; deck counts whole missing families; section K legends marked REBUILD-TODO(numbers).
- RENDER TRAP found: pytest / python run outside the worktree imports wfield_local from the editable install = the MAIN
  checkout. The render must import the wf_rebuild code; helper making render_profile --check refuse otherwise. Also routing
  chronic_drift_vs_reorg (Fig 10.2) + a scan for other untagged writers in REDO rows. Then render.
- Helper (tip 47a36184; suite 4,615 / 0 failed; log 8.37): Fig 10.2 routed; render_profile --check prints the imported
  wfield_local and refuses the main checkout. SCAN: 47 REDO scripts still refuse under the profile by design (encoder family,
  frozen_* / encoding_* / spout_* / within_spout_* / exec_*, fr_*, layer2_*, summary_figures_1021, within_lick_correction) --
  they fail loudly, never write wrong. DECISION (mine, layout): route each to <its default dir>/<run tag>/ like
  chronic_stability; helper doing it family by family. 22 possible untagged writers being triaged.
- RENDER STARTED (render runner, from a PINNED worktree wf_rebuild_render at 47a36184, PYTHONPATH set; log
  M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/RENDER_LOG.md): rebuild_root init -> RESTnl baselines -> frozen fits ->
  maps / epoch decoders / grant figures / LP arm / rest 12.1 + 12.1b / section G / nolick / cd / coding / attempt / recovery /
  chronic / June drift / drift maps / joint. Encoder / fr / ITI / quiet / O2 families join as they are routed.
- Render: rebuild_root init was blocked by its own read fence (fixed 41cfe67e; a `python -m` double-import bug remains, init
  run in-process with my OK). INIT DONE: 4 seeds copied + sha256-verified; camera_coverage.csv missing (its producer refused
  under the profile) -> figure families HELD until it is routed and run, so camera-gated stamps name PS92_0908's 82.5 %
  coverage. RESTnl baselines + frozen fits running.
- Coverage table routed (38f81c0c; a missing table now raises in a tagged run) + init CLI fixed (f31b72b1). ENCODER FAMILY
  routed (b97a39f9, 29 scripts -> imaging_encoding/<tag>/); render runner told to render it after the current families.
  DECISIONS (mine): trial_attempts.py becomes REDO (the mixed models and PI slides 34 / 35 need its camera-class table);
  compare_map_tests + imaging_baseline_compare -> NO (they compare against the pre-rebuild archive / June+Aug pre).
- trial_attempts routed (f49709ca) then ALL attempts readers switched to the new camera-class table, old seeded table dropped
  (c22507df): LP lick source, attempt_coding / maps, combined_encoding fingerprint, orofacial gates, 10 REDO scripts. In the
  new table stopped trials are "stopped" (not no_try); jaw_only = non-attempt; denominator = working period. Render runner:
  trial_attempts first, then any attempts-dependent family; anything already rendered on the old table marked SUPERSEDED and
  redone.
- Render progress: setup done; DAQ-gate RESTnl baselines 120 / 120 OK (PS93_0813 excluded, no video licks). LP-gate baselines
  read the old attempts table -> SUPERSEDED, rebuilt after trial_attempts. No figure drawn yet. fr_* routed (74986bd6),
  family 3 (layer2, summary_figures_1021, within_lick_correction + allsession_encoder) routed (f8e690b9). Helper: 10 misc
  refusers + 22-writer triage next.
- All 47 refusing scripts routed (934f9ccf). learned_compensation (needed by the prelaunch-aim goal-page rows) and
  misdirected_lick_counts made REDO + routed (09290e7f). FINDING: the misdirected-lick behaviour counts used by
  misdirected_lick_decode went through the old DAQ engagement gate -> the existing ones are stale; the rebuild recomputes them on
  the camera working period with attempts (Aug-only pre).
- INCIDENT (render runner, 11-04 05:46): camera_coverage_tables ran ONCE without the profile (a shell-script argument bug; the
  profile was never applied, so no fence) and wrote 4 derived tables into the ORIGINAL tree:
  M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/qc/{camera_coverage,
  camera_contact_nolick, camera_contact_nolick_trials, camera_stop_lengths}.csv. No older camera_* files there and the table
  was previously only in scratch -> very likely NEW files, not overwrites. Nothing deleted. Every other run had the profile (log
  headers checked). Runner's wrapper now fails closed and runs render_profile --check before every command.
  QUESTION FOR PRIYA: delete those 4 files (derived, default-mode = adopted hemo / DAQ gate), or keep? I kept them.
- 22-writer triage done (5cc29488): quiet-events family, cohort1_subset_headline, rest_baseline_epoch_drift fixed (would have
  been refused); 4 DeepLabCut-tree writers being mapped into the root (my decision); iti_nmatch_check retired (NO).
- CODE COMPLETE for the render (rebuild tip 4dd21a7c, code 621c81de; FULL suite 4,690 passed / 0 failed; log 8.38): every
  refusing REDO script routed; nothing known un-renderable. Render runner moving to it; required order recorded in log 8.38.
- ENCODER COMMANDS compiled (docs/status/rebuild_20261103/ENCODER_RENDER_COMMANDS.md) from handoffs + scratchpad launchers.
  FINDINGS: (1) no rebuild doc held the production command lines -- they lived only in two TEMP scratchpads; backed up to
  M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/scratchpad_scripts_backup_20261104/. (2) Several goal-page figure
  scripts exist ONLY in scratch (fig_goal_summary / fig_goal_epochs, Layer-2 fig_layer2 / fig_rsplit / fig_pooled, softmax /
  residual fig_*) -> rebuild gap; need bringing into the repo (RECONCILIATION section 7).
  QUESTION FOR PRIYA: the per-session combined_encoding runs (Figs 1.13-1.15 binned contest, 21.x FR, 21.12 variance maps,
  22.1) came from an O2 bundle of ADOPTED data; rebuilding them needs a new O2 bundle + ~14 job arrays. Deferred -- want me to
  build and submit it (needs your O2 login), or run a reduced set locally?
- Scratch-only page figure scripts brought into the repo + routed (rebuild 1806eaf6, suite 4,722 / 0 failed).
- RENDER 06:35-06:58: the epoch decoders failed on all 5 arms at the first write to the FROZEN MODEL STORE (the write
  fence's tag rule refused <root>/frozen_models; the store was never registered as a data store). Nothing written. Helper
  registering every profile store + an end-to-end store-write smoke test. trial_attempts running since 05:47 (checking it
  is not stuck); coverage table after it.
- Store fix 7b5d3334 (suite 4,723 / 0 failed). RENDER MILESTONE G2 DONE (07:30): coverage note reads "PS93_0813 0.0%,
  PS92_0908 82.5%"; RESTnl baselines DAQ gate + LP gate (new attempts table) 120 / 120 each, 0 errors (PS93_0813 excluded, no
  video licks). trial_attempts 63 min. Epoch decoders (all families but 15e) running since 07:30; then grant figures,
  section G, then the encoder frames.
- a8 (Priya): WAIT for the rebuild -- no main render, no interim swaps. The replacement pass's per-slide old-vs-new list
  must cover PI slides 9-13, 19, 89 and imaging page sections 1-4, 6, 7. Still open with Priya: B (state discriminators in
  spout frame), C (projections in spout frame), the D / G scored-set question; a8 forwards answers.
- PRIYA 11-04 ~08:00:
  - (2) YES: jaw_only COUNTS toward the 6 in a stop run. The render was stopped (it started at 07:30, epoch decoders, no
    outputs yet); the helper is implementing it with a new selection tag; then redo trial_attempts / coverage / RESTnl /
    epoch decoders.
  - (6) DELETED the 4 mistaken tables in LP_analysis/.../imaging_encoding/qc (camera_coverage, camera_contact_nolick,
    camera_contact_nolick_trials, camera_stop_lengths; all timestamped 05:46:27, the incident run). Kept list = this entry.
  - Asked to clarify: (1) the D / G scored-set question, (3) the rest-period start, (4) the three lick measures, (5) the
    slide 94 caveat, and (7) why the O2 encoder runs need rebuilding. Answered in chat.
  - The render runner's stop of its own epoch render was DENIED by the session permission classifier. Still running under
    the OLD tag (PIDs 19008 parent, 35904 / 24008 arms); outputs go to the old-tag folder (to be marked SUPERSEDED); the
    runner's queue is held by a STOP file. ASKED PRIYA: stop those 3 PIDs herself (recommended), or let them finish.
- PRIYA via a8 (11-04), for both branches:
  - (B) state discriminators in the spout bal_em frame (centre from working rows), after a re-check;
  - (C) projections: she asked why not all three (CDs, patterns, engagement axis) in the spout frame with a working-row
    centre; being re-checked;
  - (D) the section D scored set becomes working trials, pre and post;
  - (G) section G groups by camera classes (contact / incomplete / jaw_only / none within cue -> stop, plus "after end");
    pre panel = working; the old DAQ grouping kept as robustness.
  a8's helper implements these on main; the rebuild PORTS those commits (not reimplemented). "Rebuild follows main 11-03"
  is superseded for D / G. The pattern families' pre reference (lick vs working) is not addressed yet: still my question 1.
- PRIYA (11-04 ~08:30):
  - D / G confirmed (D scores working, engaged = robustness; G groups by camera class).
  - (3) asked: "I thought rest was spout docked" -- yes, see chat.
  - (4) asked what the FR fixed-window licks are -- see chat.
  - (5) slide 94 caveat YES -> sent to a8.
  - (7) REBUILD the O2 encoder runs, wants the O2 commands -> the helper is making o2_encoding rebuild-aware (bundle in the
    root, full profile env recorded and applied on O2, tag refusal, bundle-vs-local check), then building bundle
    enc_rebuild_20261104 + COMMANDS.md for Priya.
  - The old-tag epoch render (PIDs 19008 / 35904 / 24008) is still running.
- Jaw_only stop rule in (c3cb3c0c, tag ...stopRun6j; related tests 144 passed, full suite running). A run of >= 6 jaw_only
  trials with no no_response now counts as a stop (rule as written). The redo runs from a SECOND worktree
  (wf_rebuild_render2) so the code under the still-running old render doesn't change. Old-vs-new stop counts per animal x
  epoch coming.
- Old-tag epoch render stopped externally at 07:57 (Priya); it wrote 0 figures; the old-rule frozen models it stored
  (PS92-94) are listed SUPERSEDED in RENDER_LOG, not deleted. Redo under stopRun6j started from wf_rebuild_render at c3cb3c0c.
- PRIYA: (3) rest start stays at the first camera lick (rest = docked lick-free stretches inside it) -- settled.
  (4) asked whether the FR validation's fixed window (cue -2..+4 s) should run "until trial stop"; I recommended keeping it
  (it keeps pre-cue licks and a fixed length matched to the FR window); middle option -2 s -> trial stop offered. Waiting.
- PRIYA wants the O2 bundle ASAP: helper building it (o2_encoding rebuild-aware first, then the bundle). Waiting on the new
  attempts table (~09:05-09:10).
- PRIYA via a8 (11-04): ALL Allen-ROI analyses use the SPLIT area basis (iti_licks AREA BASIS, Priya 10-27: no cortex /
  whole MOs / MOp; ALM + tjM1; MOs_rest / MOp_rest per hemisphere; iti_licks.roi_masks); plain = robustness only. ONE shared
  function + new spec ids. Inventory of every ROI consumer running (for a8, to put "can't switch" cases to Priya). Checking
  first whether combined_encoding (the O2 bundle) uses Allen areas; if so, the encoder path switches before the bundle.
- O2 BUNDLE PLAN (helper, 08:10):
  - The attempts table is read at bundle time (engagement gate), but the heavy imaging (~28 GB) is pre-warmed into the input
    cache now.
  - combined_encoding uses Allen areas ONLY for its region tables. The split basis is computed on O2 from the bundled
    imaging, so no re-bundle is needed; it goes into the bundle's code as primary, with plain as robustness rows.
  - The 10-24 bundle took ~2 h. ETA to a checked bundle ~2.5-3 h from 08:10 (~10:40-11:10), mostly share reads.
- SPLIT BASIS, Priya's replies:
  - (5) leave the LocaNMF-label analyses.
  - (2) "should already have these": yes, quiet events define orofacial = ALM, tjM1, SSp-m, SSp-n (limb = SSp-ul, SSp-ll).
    Adopted for every orofacial group; MOs-vs-MOp contrasts become ALM vs tjM1.
  - (3) "why not px based": proposed pixel-based significance as primary, with the area table descriptive.
  - (1) asked what the rim is: v8 = the thin LATERAL rim anterior of AP -0.5 (not the full rim) + the ALM front line.
  - (4) asked where the cortex rows are.
  Replies in chat; a8 told.
  PRIYA CONFIRMED:
  - (1) v8 masks everywhere, encoder rows included. Cortex +3.2 % pixels, SSp-m +32 %, MOs / MOp +5-6 %.
  - (2) orofacial = ALM, tjM1, SSp-m, SSp-n; MOs-vs-MOp contrasts become ALM vs tjM1.
  - (3) residual_roi: pixel-based stays primary; the per-area stats adjust for the larger family of comparisons (split
    areas).
  - (4) keep the cortex rows, with v8 cortex.
  - (5) leave the LocaNMF-label analyses.
  - residual_roi detail (a8): the area table gets real tests, corrected over the split family (not descriptive).
- FINDING, jaw_only counting toward stop runs (120 curated sessions):
  - Stop runs 177 -> 283; stopped trials 3,267 -> 5,944 (2,353 of them jaw_only; +324 no_response newly stopped).
  - Working trials unchanged (54,676); early_run 0 under both rules.
  - Biggest moves:
    - PS94 acute: 543 -> 1,012 stopped;
    - PS95 subacute: 714 -> 1,173;
    - PS95 chronic: 246 -> 567;
    - PS95 pre: 214 -> 425.
  - Table: scratchpad r4/jawcmp2/jaw_count_compare.csv.
  - Effect: stopped-trial analyses (5c stopped, nolick stopped, miss_vs_stopped) get ~2x the trials; attempt-rate
    denominators lose the jaw trials inside stop runs.
- O2 bundle code committed (fb3b6304 cache key, a6a29dd7 split basis in the encoder, a7aec872 rebuild-aware o2_encoding; 14
  TAGs). Pre-warm done (107 / 107, 23 min). The bundle waits only on the stopRun6j attempts table.
- O2 BUNDLE enc_rebuild_20261104 building since 09:27 (--jobs 6, 1.8 sessions / min). ETA inputs ~10:25, checked ~10:45.
  Smoke on PS92_0806: v8 cortex R2 0.6152 vs old-mask cortex_stat 0.6160 (-0.001).
- RENDER RUNNER handed back while HELD (nothing running). Done under stopRun6j: init (6 seeds), trial_attempts
  (07:59-09:29), coverage tables (07:59-09:39; note = PS93_0813 0 %, PS92_0908 82.5 %).
  Queue ready: scratchpad/queue2.sh (RESTnl DAQ + LP -> epoch -> grant -> section G -> joint -> nolick / cd / coding /
  recovery / chronic / June drift / rest 12.1 + 12.1b -> attempt coding / maps -> restnl / LP final / drift maps ->
  misdirected). To launch: remove HOLD_j_g2a / HOLD_j_g2b, `nohup bash queue2.sh &`. Then the encoder family
  (ENCODER_RENDER_COMMANDS A, B, D-F), rebuild_figs, the drift1500 arm. The superseded old-rule caches / models are listed in
  RENDER_LOG; none deleted.
- Slide 94 caveat added by a8 (backup_before_ramp_caveat_1104.pptx; hidden states unchanged).
- The 05:46 incident coverage copies differ from the rebuild ones only by the 12 June sessions (Aug-only pre); the partial-
  coverage rows are identical.

### Findings
- main moved while the rebuild branch was being fixed (34 commits not in rebuild/cohort1-20261103). Two matter for the rebuild:
  - 054c78df: bal_em centring on every cross-session spout decoder (in-memory fits; store models unchanged). The merge brings it
    into the rebuild, so rebuild numbers are compared against the bal_em-era pages (imaging v51), not older pages.
  - 054c78df also put the epoch families' camera "working" gate ON HOLD on main: 13 of 44 pre sessions have no camera classes
    (all June 0606-0608 + PS93_0813), options (a) DAQ-engaged pre, (b) in_working pre, (c) mixed. The REBUILD already answers it:
    August-only pre removes June, and PS93_0813 is in neither working class but stays in the lick (contact) class (rebuild
    DECISIONS "Camera trial gate", "Sessions without video"). That is option (b), consistent with Priya's 11-03 decisions
    (camera gate everywhere; PS93_0813 for contact-only analyses). No new decision needed; the merge must keep the rebuild's
    wiring where the two touch poststroke_compare / precue_engagement_states.
  - e9a7dc31 (DECISIONS, Priya): firmware check -- the spout holds 3.5 s after the first lick, a miss docks at cue + 3.5 s,
    measured stop - first DAQ contact = 3.53 s; lick counts = cue + 70 ms -> trial stop. Matches the rebuild's task rule.

### Running
- Code helper (wf_rebuild): rebuild root DONE (c812a3c0, b422ae83: one derived root, fenced by an audit hook, R1-R5 closed);
  600 s + 1500 s drift arm DONE (bc57f786). Now: the 5-min stop-rule check, licks/trial cue -> stop + ILI rate, trial-stop
  verification. `rebuild_root init` (copies seed inputs into the root on M:) not yet run -- runs before the render.
- 11-03 ~19:45: helper hit the session limit mid-task (lick-count edits uncommitted in 5 files; stop-rule check had not
  produced a result); resumed with its context.
- SEQUENCING CHANGE (mine, reversible): merge main BEFORE the fifth review, so the review covers the merged decoder code
  (git merge-tree: 9 conflicts -- DECISIONS, poststroke_compare, precue_engagement_states, nolick_decoder, grant_confusion,
  locanmf_frozen_decoder, position_coding_directions, june_august_drift, quiet_task_events_followup).
- A3 is on main (1decbe53). A5 (one responded(source)) is still open on e3's side; told e3 the rebuild's trial_classes already
  implements that design (gate_mode / run_selection / selection_tag) and proposed A5 lands with the rebuild merge.
- Next: helper finishes -> merge main -> fifth review -> init root -> render G0-G8 -> two verifiers -> replacement pass.

- **11-04 ~10:05: DECISION (Priya via a8): split by camera class wherever trials are grouped or gated by outcome.**
  - This covers lick / miss / stopped / DAQ lick vs no-lick / engaged vs undetected.
  - Classes: contact, incomplete, jaw_only, none, after_end. Close mid (jaw not judged) counts as none, flagged with jaw_judged.
  - Report per class where it makes sense.
  - Example: the section G state discriminators train on DAQ lick vs no-lick. a8's helper is testing working vs none, with
    jaw_only / after_end scored separately; it will be sent for porting.
  - Inventory requested, now being built (read-only helper):
    docs/status/rebuild_20261103/CAMERA_CLASS_INVENTORY_20261104.md.
  - Bundle at 10:05: 85 / 107 inputs.
- **11-04 ~10:12: camera-class inventory written** (docs/status/rebuild_20261103/CAMERA_CLASS_INVENTORY_20261104.md).
  - 6 groups are already on camera classes, 7 kinds should switch, 8 cases stay on DAQ, and there are 12 questions for
    Priya. Sent to a8.
  - **The O2 bundle is NOT affected.** It runs with OROFACIAL_GATE=attempted. Its engaged set = attempted (contact +
    incomplete) in this run's camera-class trial_attempts.csv (stopRun6j), so stopped, after_end, jaw_only and none trials
    are all non-engaged.
  - Per-class readouts (e.g. jaw_only on its own) would be a reader-side or model-spec change made later, not a rebundle.
  - Inventory item 3e ("encoder default gate is reference") applies only to runs that don't set OROFACIAL_GATE. Every
    production encoder block sets it.
- **11-04 ~10:25: DECISION (Priya): every camera class must be reportable from the O2 encoder data.**
  - **Gap.** The encoder fits and scores only working trials (contact + incomplete) and saves per-position averages. It
    has no per-trial output and no trial ids, so it cannot split contact from incomplete and cannot score jaw_only / none /
    stopped / after_end / early_run.
  - **Fix in progress** on branch rebuild/encoder-classes (worktree wf_rebuild_cls, helper):
    - the fit is unchanged and the existing outputs stay identical;
    - every cue trial is scored out of sample (non-working trials with the full-data beta);
    - a new sessions/<label>_trials.npz holds per-trial cue- and lick-aligned area traces for data / prediction /
      components;
    - classes are joined locally by trial_id;
    - a per-class reader is added;
    - an o2_encoding `refresh-code` step updates the bundle's code snapshot without rebuilding the 40 GB inputs.
  - **Effect on O2.** Priya can rsync the inputs as soon as the bundle is done. Submitting waits for the code refresh and
    the check.
- **11-04 ~10:25: DECISION (Priya via a8): measures OF engagement move to camera.**
  - quit = last camera lick; the stop rule; quit_prodrome etc.
  - DAQ versions become robustness rows.
  - Recorded as inventory row 3h (bd8019be).
  - The 09-20 engagement findings need a re-test on camera.
- **11-04 ~10:30: ERROR CAUGHT. The first O2 bundle had the wrong encoder inputs and is being rebuilt.**
  - Cause: `o2_encoding --profile` -> render_profile.apply clears COMBINED_ENCODING_* / WFIELD_* before setting the
    profile env.
  - Effect: the exported production settings were wiped, so the 107 pickles carried the default cam4 licks / tongue / exec:
    - LICKS=combined, TONGUE=3d, EXEC=polar;
    - the EXEC3D / TARGET3D / BINS_CALIB tables.
  - The checks would have passed anyway, because they compare like with like.
  - Nothing was transferred or submitted.
  - Rebuild: the profile is applied first, then the vars; same commit; more workers. ETA ~11:00-11:10.
  - The code fix (preserve the caller's encoder vars across apply, and a test that env.json carries them) goes on
    rebuild/encoder-classes. The helper is also checking whether other --profile encoder scripts (frozen_encoder,
    allsession_encoder, the ENCODER_RENDER_COMMANDS frames) have the same wipe.
  - The render queue stays HELD until that is answered.
- **11-04 ~10:40: DECISIONS (Priya via a8): camera-class inventory section 5.**
  - Answered: 1, 2, 4, 5, 6, 7, 9-12. Still open: 3 (hit-rate period) and 8 (discriminator classes).
  - The answers are recorded in CAMERA_CLASS_INVENTORY_20261104.md section 5.
  - Running now: a census for Q6 (contact vs DAQ mismatch, with examples) and for Q4 (unjudged close spouts), both
    read-only.
- **11-04 ~10:50: DECISION (Priya via a8), Q3: hit rate uses the camera working period.**
  - Condition: the last far-spout failure block(s) before the quit tail must not be left out.
  - **Proposed guard:** a stop run / quit tail counts as stopped only if it includes non-response at a reference (close)
    spout. The end of session moves to the first reference-spout block with no camera lick after the last camera lick.
  - A census of the current rule vs the guarded rule is running: STOP_RULE_GUARD_CENSUS_20261104.md, per animal x epoch
    x spout, with hit rates.
  - **Expected effect on the O2 bundle: none.** Stop runs and the quit tail contain only no_response / jaw_only, so the
    guard only moves failure trials back into the working period. contact + incomplete (the encoder's engaged set and the
    decoders' working rows) cannot change. The census confirms this.
- **11-04 ~11:00: PENDING (Priya via a8): the cue + 70 ms contact floor (trial_classes.CONTACT_FROM_CUE_S).**
  - a8 is measuring first-contact latency. No change until a8 sends the rule.
  - If 70 ms stays, licks at 0-70 ms are excluded from both post-cue and pre-cue (today they count in licks_precue).
    That moves the pre-cue lick tables only; the classes stay the same.
  - If the floor is SHORTENED, trials with a contact at <floor..70 ms> can change class (e.g. no_response -> contact).
    That changes:
    - the working set;
    - trial_attempts (a new tag);
    - the O2 bundle's engaged set (an input rebuild);
    - every class-dependent render.
  - So decide this before the O2 jobs are submitted.
- **11-04 ~11:05: FINDING (a8, interim): the cue + 70 ms floor does not change classes.**
  - Counted over 54,808 cohort-1 trials. Only 3 trials change class at floor 0, and 2 at floor 50.
  - So trial_attempts, the working set and the O2 engaged set do not depend on the floor, and O2 does not wait for it.
  - 2,167 trials (4 %) have their first post-cue contact < 70 ms, mostly PS92 / PS93. Examples: PS92 pre 401 / 2,314,
    PS93 chronic 414 / 6,068.
  - Excluding 0-70 ms licks from licks_precue moves only the pre-cue lick tables and any cue-anchored latency readout.
  - The encoder's continuous lick kernels use every lick and are not cue-windowed, so they are unaffected.
  - The final rule is pending from a8.

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
- (none blocking yet; the stop-rule check result and the fifth review will add items here)

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
- Next: fifth review -> merge main -> init root -> render G0-G8 -> two verifiers -> replacement pass.

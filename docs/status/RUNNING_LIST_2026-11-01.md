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

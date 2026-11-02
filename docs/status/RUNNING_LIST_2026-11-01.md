# Running list — 2026-11-01 night (lick source, RESTnl roll-out, engagement gate)

Priya 11-01 (going to sleep): "please proceed with any new analyses you deem necessary - keeping a running list for when I
return and what was added to the page / any conclusions that changed. keep an updated handoff throughout."
Pages: goal https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d ; imaging https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV
Handoff: STATUS_2026-10-26_ENCODING_HANDOFF.md, section 00-COMPACT-19. All PROVISIONAL, n = 4.

## 1. Decisions needed from Priya
- [ ] (filled as they arise)

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

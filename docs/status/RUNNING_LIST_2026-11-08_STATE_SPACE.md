# Running list, 2026-11-08 overnight (state-space arm, unattended)

Priya 11-08 ~23:30: "i'm going to sleep - please continue to proceed through tasks, keep a running list of findings, and
any questions or decisions you need me for." Order agreed before: (4) cleanup -> (1) detour vs balance marker -> (2)
LocaNMF switch + June plan -> (3) neural vs behavioral within-session drift -> (5) repertoire power check -> (6) housekeeping.
Branch `analysis/state-space-recovery-20261107` (worktree wf_statespace); nothing pushed.

## A. Decisions / questions for Priya (newest first)

9. REBUILD MASK BUG (B12): under the rebuild profile the painted glue mask is silently empty, so every rebuild step that
   calls `beta_maps.brain_mask()` (encoder, area masks of the Allen extract, maps) includes the glue. OK to fix
   `paint_exclusion.mask_dir` in shared code (canonical masks regardless of profile + raise when missing)? The re-runs
   it implies belong to the rebuild / encoding arms -- tell them, or me?

8. FINDINGS 0c item 1, after the LocaNMF method check (B9): "In the Allen-area space the chronic code is the pre
   mapping, larger and shifted (r with pre -.04, amp +.30*). In the finer LocaNMF components (per animal) it is
   re-patterned: rotation beyond per-component gain +.050* and variance outside the pre space +.043* in 4/4 -- in the
   method check that is the size of a ~25 deg rotation of the main pre plane plus ~5 % new-dimension variance, and both
   readouts respond only to their own planted change; the pattern-correlation drop (-.21) is larger than any single
   planted change gives (<= -.07). The re-patterning is resolved only at the component level." OK to apply (FINDINGS +
   page)? Also: the cohort-SVD run (B8) is a third basis to check it in.

**ANSWERED 11-08 (Priya: "ok for noise-corrected wording. run the method check"):** A6 applied (FINDINGS 0c item 2,
page summary row + section 7 takeaway, handoff item 5, plan note; the along-line sign split named as Allen, LocaNMF
PS92 ~0 / others toward); A7 running (`sim{tag}` folders: lnmf lick, lnmf cue, allen lick; Allen cue = `sim/`).
Also asked: loops in the LocaNMF basis (B7).

7. Run the planted-change method check (`state_space_sim.py`) in the LocaNMF basis before rewording 0c item 1 (B6)? It
   is local and moderate (pre sessions only); without it the LocaNMF rotation / out-of-space sizes have no calibration.

6. FINDINGS 0c item 2 + page summary / section 7 ("80-95 % of acute - pre lies orthogonal ... in every variant") ->
   "Most of the acute change lies off the pre -> chronic line: ~75-90 % of its length (~55-85 % of its variance; noise-
   corrected, B5), every animal and basis. Its smaller along-line part points away from the chronic state in PS92 /
   PS94 and toward it in PS93 / PS95." OK to apply?

**ANSWERED 11-08 morning (Priya: "1. yes 2. yes if it doesn't conflict the other windows 3. yes 4. now - again as long
as no conflict 5. yes"):** A1 applied to FINDINGS 0c item 3 + rebuild re-render of the balance marker queued; A4a main
pushed (with the other window's 4e6630be, after merging origin/main: no conflicts); A4b decline branch merged, 0c item 6
shortened to a pointer; A4c S28 subsection published into the behavior page by this window; A4d state-space branch
merged to main (no conflicts); A5 applied to FINDINGS 0c item 4 and the page caption. Items below kept as the record.

5. FINDINGS 0c item 4 ("Trial clouds move as a whole ... Power check pending. DESCRIPTIVE") -> rewrite as "Whether trials
   slide or switch between pre-like and chronic-like states cannot be tested here: single-trial scatter along the
   pre -> chronic line is ~1.4x the whole pre -> chronic distance, and planted two-state sessions score the same as the
   data (power check B3)." Same rewrite for the page's slide-vs-search figure caption (Fig ss_repertoire / trial clouds)
   and the handoff "where things stand" item 5. OK to apply? (I have NOT changed FINDINGS / the page yet.)

4. Housekeeping (item 6), all waiting on you:
   a. PUSH main? It holds FINDINGS 0c (4598ec00 + merge 0f4b5879) behind another window's unpushed 4e6630be
      ("Handoff: kinematics v29 / behavior v3"). Per the shared-checkout rule I only push when every listed commit is
      mine -- either that window pushes first, or you OK pushing theirs with mine.
   b. MERGE the within-session-decline branch (`analysis/within-session-decline-quitting-20261107`, 4 commits incl. a
      FINDINGS line) into main? Its FINDINGS line and 0c item 6 overlap -- I would shorten 0c item 6 to a pointer.
   c. The S28 behavior-page subsection (`docs/results_pages/within_session_decline/section.html` on that branch):
      hand to the behavior-page window, or should I publish it into that page myself?
   d. The state-space branch itself (this one): merge to main now (exploratory, tests pass, outputs unchanged by the
      cleanup), or keep it separate until the LocaNMF results are in?

1. FINDINGS 0c item 3 -- replace "OPEN: whether this axis is more than the 0b balance marker" with: "The axis shares ~70 %
   of its within-animal variance with the 0b balance marker but tracks aim beyond it in 4/4 animals (partial r -.35 to
   -.88, first-lick aligned); the balance beyond the axis is weaker (3/4)." OK? (Balance from the pre-rebuild run; a
   rebuild re-render of `lick_recovery_primary` would remove that caveat -- worth queuing?)

## B. Findings (newest first; exploratory)

46. **Aligned frozen decoder: acute RE-EMBEDDED, chronic PARTIAL (both bases; `scripts/aligned_decoder.py`, plan in C,
   gate switched to alignment efficiency by Priya 11-09, DECISIONS).** Frozen-to-refit gap +.13 / +.10 / +.15 vs pre
   (4/4), refit unchanged. Per-spout Procrustes closes the extra gap acutely (index .79 LocaNMF / .73 SVD, dR n.s.) and
   ~45 % chronically (.43 / .49, dR *); label-free covariance alignment ~0; shuffled labels at chance. Cortex: mostly
   within-hemisphere rotation (cosine -.2 to -.3 acute, 4/4) + a small contralesional re-weighting (tjM1 in LocaNMF,
   SSp-m in SVD: area basis-dependent). FINDINGS 0 "restored / re-embedded / replaced".

45. **Why the raw rest level falls: everything else rises relative to rest; the time budget barely moves (handoff item d;
   `scripts/state_space_rest_decomp.py`, `rest_decomp_svd/{<label>.npz,sessions.csv,changes.csv}`, page_figs
   `ss_rest_decomp_lick_svd.png`).** The signal is relative to the session's slow mean, so rest = -sum_k f_k (m_k -
   m_rest) over frame classes (identity checked: sum f_k m_k = 0 to 1e-15; RESTnl level vs rest-frame mean cos .998).
   Classes on the drift-removed SVT: RESTnl frames (pre .03 of frames), in-trial (cue to +3.5 s, engaged; .26), other
   lick-buffered (+/- RESTnl's [0.5, 2.0] s lick buffer, outside trials; .52), quit (.02), pretask (~0), other (.17).
   Orofacial contra (B32 units), acute / subacute / chronic: RESTnl -.48 / -.53 / -.76 (rest-frame mean -.41 / -.45 /
   -.77); split -> via time budget (epoch fractions, pre gaps) -.06 / -.01 / -.03 (ns); via gaps (pre fractions,
   epoch gaps) -.37 / -.45 / -.73 (all CIs exclude 0). Gaps above rest: in-trial +.26 / +.68 / +1.21 (pre .96; chronic
   [+.87, +1.52], 4/4), other lick-buffered +.63 / +.39 / +.63 (pre .28). Fractions: rest +.02 acute only, in-trial
   +.04 acute only, else ~0. Ipsi and whole brain the same pattern. Reading: the raw rest level falls because the
   task and licking periods (~80 % of frames) sit further above rest, not because the session holds more of them --
   the same increase as the post-lick response (Figs 9.1, 10.25), seen through the session-mean reference, and it
   extends to licking outside cued trials. Whether rest itself moves in absolute terms cannot be measured (no absolute
   level across days, B32). Consequence: the tonic "pre-cue minus rest" elevation and the raw rest fall are two faces
   of one change in the task-vs-rest contrast. EXPLORATORY.

44. **It is the movement MODEL, not the window (Priya 11-09 "yes run"; `scripts/state_space_encoder_window_kbs.py`,
   `encoder_subtract/window_kbs.csv`).** On the SAME trials, window (-0.1..+1.0 s after the first lick) and targets (the
   frozen encoder's own per-trial Allen-area values, trials with the KBs covariates, joined on cue time), spout-specific
   far contra, acute: kept along the data change -- encoder prediction removed .44 (PS92-95 .04 / .58 / .64 / .49), KBs
   bout-summary regression removed .87 (.46 / .72 / 1.41 / .91). SSp-m ipsi -2.07 [-2.97, -1.19] (4/4) -> -0.64
   [-1.26, -.05] (3/4, PS95 +.22) encoder vs -1.66 [-2.77, -.99] (4/4) KBs; SSp-n ipsi -1.79 -> -.44 (CI incl. 0) vs
   -1.51. So the bout summaries under-model the movement that carries the acute far-contra difference; the lick-locked
   kernels (with contact) capture it. Implication for Figs 9.1-9.3 b-d: they bound movement from BELOW -- "survives KBs"
   is not evidence that a between-spout change is not movement. The activity result (9.1) is unaffected: it holds under
   every model, including the epoch-own coupling (item 43). EXPLORATORY.

43. **Letting the movement -> cortex coupling change with epoch removes most of the between-spout change; the activity
   increase stays (handoff item b, second half; `WF_SS_FR=KBw|KBe`, `chronic_maps_lick_svd_rest_fr{KBw,KBe}`, page_figs
   `ss_cmap_*_rest_fr{KBw,KBe}_lick_svd.png`).** Kept along the same-trials control's (K0) change, cohort mean of
   per-animal projections over stat-mask pixels (significant share of the stat mask, control -> corrected):
   KBs (frozen pre fit) / KBw (pre fit on within-session x spout variation only) / KBe (each epoch's OWN within-spout
   coupling, leave-session-out). Mean activity acute 1.03 / 1.01 / 1.09, chronic .98 / .98 / 1.02 (46 -> 68 / 67 / 80 %).
   Far contra spout-specific acute .76 / .79 / .42 (16 -> 9 / 8 / 0 %), chronic .49 / .45 / .03. Close ipsi chronic
   .71 / .63 / .46 (40 -> 7 / 5 / 0 %). KBs ~ KBw: the within-centering does not matter; the frozen-coupling ASSUMPTION
   does. With the post-stroke coupling (reach coupling grown, contact coupling lost, item 41), the acute far-contra
   spout-specific deficit falls to .42 with no pixel significant -- converging with the encoder's own prediction (item
   40: .35 kept). Caveat: KBe fits the coupling on the same epoch it corrects (leave-that-session-out, within-spout
   only, so no session-level change can be fitted directly), so it is the upper bound on what movement can explain.
   Reading across items 38-43: the ACTIVITY increase is not movement under any model; the BETWEEN-SPOUT change (acute
   far-contra deficit, chronic spout differences) is largely accounted for by movement once movement is modeled with
   lick-locked kernels (encoder) or with the post-stroke coupling (KBe); a residual acute far-contra deficit in
   ipsilesional SSp-m survives only the frozen models (3/4 under the encoder). EXPLORATORY.

42. **June no-stroke reference on the maps: the lick response SHRANK June -> August, and the post-stroke increase runs
   the OPPOSITE way (handoff 11-09 ~12:15 item c; Priya "go ahead with all"; `scripts/state_space_june_maps.py`,
   `june_maps_lick_svd/{maps.npz,animals.csv}`, page_figs `ss_june_maps_{full,nocw,spout}_lick_svd.png`).** Contact first
   licks everywhere (June rows = first DAQ contacts), lick-aligned 0.1-0.5 s minus RESTnl (June has RESTnl), cohort SVD,
   units = own August-pre response RMS, nested-bootstrap contours. June -> August pre (no stroke, ~64 d): a significant
   DECREASE over bilateral lateral orofacial / anterolateral cortex and posterior cortex, every spout. August pre ->
   acute / chronic: the increase over overlapping territory. Per animal over stat-mask pixels, mean over spouts: cos(drift,
   chronic) -.84 / -.71 / -.46 / -.63 (PS92-95), chronic along the drift direction -1.03 / -2.05 / -.51 / -.67, RMS
   chronic / drift 1.2 / 2.9 / 1.1 / 1.1. Minus cortex-wide: cos -.44 / -.57 / -.62 / +.20. Spout-specific far contra:
   mixed (cos +.38 / -.55 / -.46 / -.21; chronic 1.0 / 3.0 / 2.3 / .55 x drift). Reading: no-stroke drift cannot produce
   the post-stroke increase (opposite sign); the increase looks like a partial REVERSAL of a June -> August shrinkage
   (consistent with practice-related sparsening undone by the stroke) -- an interpretation, not a test. Caveats: June ->
   August includes learning + a room / rig move; the RESTnl reference itself may differ June vs August (item d); 3 June
   sessions per animal. EXPLORATORY.

41. **The movement -> cortex relation: face coupling holds, kinematics coupling shifts (handoff item b;
   `scripts/state_space_fr_coupling.py`, `fr_coupling_lick_svd/{sessions,changes}.csv`, maps.npz; page_figs
   `ss_fr_coupling_lick_svd.png`).** KBs covariates (8 kinematics + contact, 7 face), within-session x spout trial-to-trial
   variation only, covariates in pre-SD units; per-session coupling vs the animal's pre coupling (pre sessions vs the
   other pre sessions). Gain along the pre coupling (1 = unchanged; pre .99): change -.11 / -.20 / -.14 (acute / subacute
   / chronic, CIs incl. 0); kinematics block chronic -.34 [-.68, -.03] (4/4 lower); face block -.11 / -.20 / -.13 (ns).
   Off-axis coupling energy / pre energy (split-half unbiased; pre .14): +.09 / +.04 / +.13 [+.005, +.28] chronic (3/4).
   Within-spout variance explained by the PRE coupling (pre .185): -.04 / -.09 [-.14, -.04] (4/4) / -.06. Maps: the
   FaceRhythm lick-sum coupling (dominant, peak ~1.0 per SD) unchanged every epoch; contact-count coupling over bilateral
   sensorimotor cortex (peak .13 pre) largely gone after the stroke (.04-.08); reach coupling grows (.04 pre -> .17 / .23
   subacute / chronic) over the same territory. Caveat: reach and contact are correlated, so a joint fit can move weight
   between them (one relation re-split, not necessarily two changes). Figure scale: one per row over all epochs (Priya
   11-09: the first draw, scaled to pre, saturated acute n_licks / chronic reach). Whether this matters for Figs 9.1-9.3:
   `WF_SS_FR=KBw` (within-centered pre coupling) vs `KBe` (each epoch's own coupling) -- see the next item when done.

40. **Subtracting the ENCODER's own lick-locked prediction removes much more of the acute far-contra deficit than the
   bout-summary controls (handoff item a; `scripts/state_space_encoder_subtract.py`, `encoder_subtract/{kept,areas}.csv`,
   page_figs `ss_encoder_subtract.png`).** No frame-level predictions are saved, so the state-space readouts were computed
   on the frozen encoder's OWN per-trial values (run `frozen_encoder_gate-attempted_v7g3_t3d_spont_frA`, `trial_parts`,
   Allen areas, first-lick window -0.1..+1.0 s -- NOT 0.1-0.5 s; attempted gate = contact + incomplete; pre sessions with
   leave-session-out betas), before / after removing its predicted lick + exec_locked + exec_within + contact ("-lick"),
   fr + fr_dir ("-face"), or both. Spout-specific far contra, kept along the data change: acute .63 / .73 / .35 (-lick /
   -face / -both), chronic .54 / .57 / .11. SSp-m ipsi far contra acute -2.06 [-2.85, -1.28] (4/4) -> -1.12 (4/4) /
   -1.58 (4/4) / -.64 [-1.20, -.13] (3/4; PS95 +.10). Mean over spouts (reference-dependent here: the encoder's signal is
   drift-removed with no RESTnl): kept acute 1.07 / .95 / 1.01, subacute .70 / .58 / .28, chronic .77 / .78 / .56.
   Reading: with lick-locked kernels (onset, tongue / jaw magnitude, spout-locked templates, direction, contact) about two
   thirds of the acute far-contra spout-specific deficit is predicted movement; a third remains (SSp-m ipsi, 3/4). That
   reconciles the gap with the encoder (lick + contact groups carry the acute far-contra drop): the bout-summary controls
   (Ks / KBs: kept .91 / .73) under-model lick-locked movement. Caveat: the encoder window includes the late,
   contact-dependent part (B31) that the 0.1-0.5 s maps mostly exclude, and contact is in "-lick". EXPLORATORY.

39. **Why the 9.2 movement versions are identical (Priya 11-09 "why do the 9.2 versions all look identical?"):** what
   the movement covariates remove lies mostly along the cortex-wide mode that 9.2 projects out first -- share of the
   removed session-to-session variation along it: face (Bs) .88 (.66-.97 per animal), kinematics + contact .72, both .90;
   of the removed chronic change .91 / .68 / .71; vs .53 for the pre responses' own session variation. Same fact as the
   9.1 contours growing (the removed whole-cortex fluctuation was noise in the pre reference). v37 layout (Priya "1. ok;
   collapsible supplemental; 2. ok"): 9.1 / 9.3 main b = face, c = kinematics + contact, d = both; S1-S5 collapsed
   (`details.supp-levels`); 9.2 original + 8 collapsed versions. `scripts/page_file_names.py` now covers supplements.

38. **Tongue / jaw kinematics + contact control (Priya 11-09 "yes"): activity untouched; acute far-contra spout-specific
   loss mostly survives; chronic spout differences = kinematics + face together.** `WF_SS_FR=K0|Ks|KBs` (8 per-trial
   covariates: bout licks, contacts, latency, reach, speed + first-lick 3-D landing; spout-centered; KBs adds the Bs face
   covariates; K0 = same-trials control; 96 % of trials). Kept along the control's direction (Ks / KBs), sig % K0 -> Ks /
   KBs: 9.1 mean 0.96 / 1.05 acute, 1.01 / 1.01 chronic (areas identical: SSp-m contra +.69 -> +.67 / +.67); 9.2 mean
   0.96-0.98. 9.3 ACUTE far contra 0.91 / 0.73, contour 16 -> 21 / 9 %, ipsilesional orofacial loss still significant
   (SSp-m ipsi -.63 -> -.48 / -.49, SSp-n -.55 -> -.42 / -.41, tjM1 -.46 -> -.36 / -.37): kinematics + contact take ~25 %,
   PREDICTION (absorbs much) NOT MET. 9.3 CHRONIC far contra 0.76 / 0.24, contour 10 -> 1 / 0 % (SSp-tr contra -.33 -> -.27
   * / -.13; VISa -.23 -> -.18 * / -.07); close ipsi 1.01 / 0.62 (40 -> 32 / 7 %). Contralesional SSp-m spread +.07 4/4
   in every mode. Vs encoder (acute far-contra cortex evoked DROP carried by its LICK + contact groups): different readout
   (cortex-total evoked vs the spout-specific ipsilesional orofacial pattern) and our covariates are bout summaries, not
   the encoder's time-resolved lick-locked kernels -- a remaining gap, not a contradiction yet. Figures on the share
   (page_figs ss_cmap_{full,nocw,spout}_rest_fr{K0,Ks,KBs}_lick_svd.png, family scales); page v37 build pending.

37. **Spout-centered face covariates (no spout information): activity unchanged; acute far-contra loss untouched; chronic
   far-contra loss PARTLY face (Priya 11-09 "add the covariate relative to spout's pre-stroke mean, and relabel frB").**
   `WF_SS_FR=Bs|As` = B / A covariates minus the animal's pre mean at the trial's spout (training sessions only). WHY: raw
   B / A covariates decode the spout on pre trials (B .40-.60, side .71-.86, ring .64-.86; A .61-.87; lick sum alone
   .22-.27; chance .17 / .50 / .50) -- the non-lick factors follow the spout, so B was NOT target-free (B36's 9.3 reading
   was partly circular). Results (kept along the control's direction, Bs / As; sig % of stat mask): 9.1 mean 0.96 / 0.83
   acute, 0.99 / 0.90 chronic; 9.2 mean 0.95 / 0.98 (Bs acute / chronic); 9.3 acute far contra 1.08 (SSp-m ipsi -.64 ->
   -.64, SSp-n -.56 -> -.59, tjM1 -.47 -> -.48, all *); 9.3 chronic far contra 0.57 / 0.76, contour 10 -> 0 / 1 %, area
   level SSp-tr contra -.34 -> -.22 * and VISa contra -.23 -> -.14 * survive, SSp-ll contra -.31 -> -.19 n.s.; chronic
   close ipsi 0.60 (35 -> 10 %). SIGN vs encoder: across-spout spread (SD over 6 spouts) of the spout-specific response
   in contralesional SSp-m rises chronically 4/4 (+.07 control, +.06 Bs, +.07 As) = same direction as the encoder's
   contralesional SSp-m position rise (8.19 B, landing frame, FR-A); far contra's -.12 there is one spout leaving the
   others. Also: pre-stroke response row on EVERY cohort map (all versions / baselines; `--pre-map` merges per-version
   pre rows into pre_maps.npz via the response cache), shared scales per family (full +-0.9 pre 0-2.3; nocw +-0.6 pre
   -0.6-0.9; spout +-0.43 pre -0.4-0.45; WF_SS_PREMIN / PREMAX). Page v37 BUILT LOCALLY, NOT PUBLISHED (account switch:
   the page belongs to the other claude.ai account): scratchpad state_space_page/index.html, page script
   `rebuild_fr_section9.py` (9.1b-e, 9.2b-e, 9.3b-e = Bs / As / B relabeled / A).

36. **The CHRONIC spout-specific change is mostly face movement; the ACUTE far-contra loss is not (Priya 11-09 "yes 1
   and 2", "how do we reconcile the encoder map and state space map results"; `WF_SS_VERSIONS=nocw,spout WF_SS_FR=B|0|A`;
   page v36 Figs 9.3b-c, section 9 take + summary row rewritten).** Along the control's own direction (B / A): chronic far
   contra 0.46 / 0.52 kept, significant area 10 -> 1 / 4 % (only SSp-m contra -.12 significant, B); chronic close ipsi
   0.58 / 0.70 (35 -> 10 / 12 %); acute far contra 0.99 / 0.90 kept, ipsilesional orofacial loss still significant (SSp-m
   ipsi -.64 -> -.59 / -.43, SSp-n / SSp-ul / tjM1 ipsi too). 9.2 (nocw) like 9.1 (0.83-1.08). RECONCILES with the
   encoder: activity change (9.1) not face; chronic spout differences = face-linked (encoder chronic position R2 rise +22
   -> +4 with FR); acute ipsilesional SSp-m loss survives in both. BUGS FIXED 11-09 in `state_space_chronic_maps.py`: a
   WF_SS_VERSIONS subset run (1) overwrote maps.npz / areas.csv (now merged) and (2) took its unit from the first version
   run, not the FULL pre response (now always full; needs the full cache). The first 9.2/9.3 FR outputs were ~2.8x off in
   units (ratios / retention unaffected); rerun from the response cache, full maps rebuilt.

35. **Balance in the first-lick window (Priya 11-09 "yes 1 and 2"; `WF_SS_ALIGN=lick` balance scripts, tables
   `landing_lick/` (cue tables untouched in `landing/`); page v35 Figs 5.5-5.8).** 0.1-0.5 s after the first lick minus
   -1.0..-0.6 s. Same picture as the cue window. Orofacial matched: far contra +.30 / +.27 / +.29 (chronic 4/4), close
   contra chronic +.22, far mid chronic +.11 (4/4); acute ipsi-side reversal (far ipsi -.22 0/4, close ipsi -.16 1/4);
   spread -.28 / -.16 / -.13 (0/4 chronic). Motor matched: far contra +.29 / +.25 / +.25, far mid +.13 / +.10 / +.14, far
   ipsi sub / chr +.14 / +.16 (4/4); no acute reversal (far ipsi +.01, close ipsi +.03), no spread compression (-.13 /
   -.01 / +.01, CIs incl. 0). Also: every page caption now ends with its file name (`scripts/page_file_names.py`,
   docs/templates/README.md rule). ENCODER COMPARISON (grant window's note, checked): the encoder's chronic position-R2
   rise falls +22 -> +4 (n.s.) with FR in the model (ENCODING_FINDINGS), while Fig 9.1 activity keeps 94 % (FR-A) --
   different readouts and controls (FR in-model per session vs frozen pre-only subtraction); 9.3 (spout-specific) with FR
   is the like-for-like check (running 11-09).

34. **The post-lick change is NOT face movement (Priya 11-09 "combining state space fig 9.1 with FR ... to regress out
   FR-explained variance", "directionless first, then all"; `WF_SS_FR=B|0|A` in `state_space_chronic_maps.py`; page v32
   Figs 9.1b-c).** Per trial FaceRhythm `master_pre_v1` factors 0.0..0.5 s after the first lick (cue-time join; 0.5-1.9 %
   of trials without FR dropped); per animal a PRE-ONLY lstsq fit of the response (100 SVD comps) on the centered
   covariates (B: lick sum + 6 non-lick = 7; A: + left / right / close / far = 11), every trial minus fit x (cov - pre
   mean), pre sessions leave-session-out. FR explains 13-23 % (B) / 16-25 % (A) of held-out pre trial variance
   (`scratchpad fr_r2`), yet vs the same-trial control (0) the spout-mean change keeps 97 / 98 / 98 % (B) and 92 / 95 /
   94 % (A) of its size, pattern r >= .987; far contra 86 / 79 % acute, 115 / 108 % chronic -- RMS ratios; CORRECTED
   11-09 (Priya "the change is MORE?"): measured ALONG the control's change direction far contra keeps 86 / 76 % acute,
   110 / 103 % chronic; the RMS gain is the face map's off-axis pattern (0.3 of the change) and the per-animal face part
   along the change is mixed in sign (-.07 / +.03 / +.19 / -.12) -> not larger, same + noise. Spout mean along-axis
   97 / 98 / 98 (B), 92 / 94 / 94 (A). Figs 9.1 / 9.1b / 9.1c on one scale (WF_SS_VMAX=0.9, WF_SS_PREMAX=2.3; page v33).
   CONTOURS (Priya 11-09 "areas of significance seem larger with FR regressed out"; page v34): chronic, B vs control,
   significant stat-mask area far contra 28 -> 48 %, far mid 14 -> 25, far ipsi 19 -> 25, mean 46 -> 56 (pre session-to-
   session SD x0.82-0.88, change along its direction 0.98-1.10: less noise, not more change); close ipsi 73 -> 51, close
   mid 57 -> 41 (change kept 0.82 / 0.86, SD 1.03 / 0.98: ~15 % of the change goes with the face). Chronic areas (B): SSp-m
   contra +.67, tjM1 contra +.64, SSp-m ipsi +.60, 4/4 (Fig 9.1 +.70 / +.66 / +.62). Also 11-09: Fig 9.1 pre-response
   row (viridis); every map in the encoder design (white unanalyzed, full CCF outline incl. bulbs); quick-look-first +
   response cache (`chronic_maps_resp/`) so a stats run never reloads.

33. **Motor-area balance (ALM + tjM1) as well (Priya 11-09 "for the orofacial balance figures, can we have motor version
   as well (ALM, tjM1)"; `state_space_balance_cells.py --motor`, `state_space_balance_angle.py --collect --motor` then
   `--motor`, tables `landing/*_motor.csv`; page v29 Figs 5.3-5.4).** Cue 0-1 s, Allen areas, same selections / models
   as Figs 5.1-5.2. Matched target + landing, subacute / chronic: contralesional at contra-side and middle targets (far
   contra +.23 / +.25, far mid +.09 / +.13, close mid +.11 / +.07, 4/4; close contra +.15 / +.12), ipsi-side targets not
   different from pre (chronic far ipsi +.10, close ipsi -.03). Unlike orofacial: no acute ipsi-side reversal (far ipsi
   -.11, close ipsi -.03, 2/4; orofacial -.39 / -.24, 0/4) and no spread compression (chronic -.06, 2/4, CI incl. 0;
   orofacial -.27, 0/4). Pre motor spread smaller (.71-1.53 vs 1.24-2.46).

32. **The tonic elevation is mostly REST MOVING (Priya 11-09 "should we check with raw (instead of RESTnl)";
   `scripts/state_space_shift_rawref.py`; page v28 Fig 10.26; B29 / B30 statements rewritten on the page).** Raw = dF/F
   scale, zero = the session's mean frame (wfield `svd_blockwise` subtracts / divides by `frames_average`; drift removal
   incl. the constant). Orofacial contra raw RESTnl -.48 / -.53 / -.76 (4/4 chronic), raw pre-cue -.46 / -.35 / -.37 ->
   pre-cue - RESTnl +.02 / +.17 / +.39. Post sessions animal-demeaned: 78 % of the tonic level's variation = rest
   (r .81-.92 per animal), 22 % pre-cue (ipsi 77 / 23). Aim coupling (B29) is the rest level's: raw pre-lick vs aim slope
   -.0004 / +.0017 per deg (p .88 / .50 contra / ipsi); raw RESTnl -.013 / -.021 (p .001 / <.001). Phasic post-lick
   survives the session-mean reference: raw +.68 / +.72 / +.59 contra (ipsi +.42 / +.35 / +.30, CIs incl. 0).
   Priya asked about absolute corrected 470 (not dF/F): no absolute level exists across days -- LED power is titrated by
   hand each day (`crossday_intensity` / `hemispheric_intensity` docstrings), and 470/415 does not cancel two separately
   set LEDs; within a session any reference gives the same differences. Possible next: an anatomical reference (orofacial
   relative to other cortex at rest vs pre-cue).

31. **The phasic post-lick increase is shared across spouts; its late half needs spout contact (Priya 11-09 "sure go
   ahead"; `scripts/state_space_shift_spout_timing.py`; page v27 Fig 10.25).** Region means minus RESTnl per spout, change
   vs the animal's own pre trace for that spout. Every spout up early and late in every epoch (contra late chronic +.96 to
   +1.65). Split-half cross-validated energy 0..0.7 s (trial noise removed; pre sessions vs a leave-session-out pre
   trace): shared +1.59 / +1.75 / +1.97 contra (pre .18); spout-specific +.34 / +.16 / +.15 (pre .04; 4/4 up but ~10x
   smaller after acute); specific timing +.60 acute, then +.06 / +.11. Acute spout-specific part = far contra (late contra
   +.06, ipsi -1.50, 0/4) = CONTACT: ~90 % of acute far-contra first bouts miss; all spouts, contact trials carry the late
   increase (+1.67 / +1.44 / +1.43 contra), no-contact trials lose it (-.41 contra, -1.54 ipsi 4/4 acute); the early part
   (0..0.3 s) is there either way (+.98 vs +.89 acute contra). Note: a within-session spout-label shuffle was tried first
   and dropped (the pre spout structure stays in the reference, so it is not a null).

30. **The pre-lick rise IS the pre-cue level: two components (Priya 11-09 "great keep going";
   `scripts/state_space_shift_precue.py`; page v26 Fig 10.24).** Region means minus RESTnl in the trial's own pre-cue
   ENL window, -1.0..-0.1 s before the first lick, 0..+0.7 s after. Orofacial chronic change: pre-cue +.39 contra /
   +.31 ipsi, pre-lick +.37 / +.28 (4/4); pre-lick - pre-cue ~0 in every epoch (-.03..+.07); across post sessions
   pre-cue vs pre-lick within-animal slope +.92..+.98 (r .76-1.0 per animal). Post-lick orofacial +1.16 / +1.25 / +1.35
   contra, does NOT covary with the pre-cue level (slope +.21 p=.34 contra, +.09 p=.64 ipsi). Synthesis with B26-B29:
   (1) TONIC orofacial elevation before the cue (vs rest), grows with far-contra aim recovery; (2) PHASIC post-lick
   orofacial increase, ~3x larger, from acute on, not licking / jaw / face / running, not tracking aim. Caveat: the
   tonic part is vs RESTnl; vs the session mean it reads as a decrease (Fig 9.9) -- may be partly rest moving.

29. **Against behavioral recovery (Priya 11-09 "great keep going"; `scripts/state_space_shift_recovery.py`, on B27's
   sessions; page v25 Fig 10.23).** Per post-stroke session, change beyond licking (minus RESTnl) vs far-contra
   first-cued-lick aim (Fig 6.3 axis, aim_deficit_sessions farcontra_first). Orofacial BEFORE the lick rises with aim
   recovery in 4/4 (contra r .22-.74, ipsi .39-.82; within-animal slope +.023 / +.032 per deg, shuffle p < .001);
   within epochs ipsi p = .016, contra p = .46 (contra = the epoch step). Post-lick orofacial does NOT track aim
   consistently (late r PS92 +.64, PS94 -.64, PS95 -.47; within epochs worse aim -> slightly larger late increase,
   p = .001 contra / .026 ipsi). Posterior-medial transient: no coupling (shuffle p .11-.29). Reading: the lasting
   post-lick orofacial increase is present regardless of aim; the pre-lick rise grows with recovery (likely the same as
   Fig 9.8's pre-cue rise vs rest -- not yet tested directly).

28. **Nor face / jaw / body movement for the orofacial increase (Priya 11-09 "go ahead"; "don't use motion energy - we
   have FR and 3d keypoints"; `scripts/state_space_shift_movement.py`, on B27's trial table; page v24 Fig 10.22).**
   Nested pre-stroke models on the SAME trials (all covariates present; FaceRhythm sessions Aug-Oct, so pre = August):
   licking; + jaw (pose3d jaw_trials: open_max_mm, lr_wide_mm, n_wide); + FaceRhythm master_pre_v1 (12 factors x D
   over -1.0..+0.7 s from the first lick, log1p, + D) + running (fraction of the window in a behavior_events running
   bout). No motion energy (Priya). Pre leave-session-out R^2 beyond spout: orofacial .02-.14 -> .15-.26,
   posterior-medial .02-.05 -> .38-.45 (the model is strong). Orofacial contra late observed +1.22 / +1.31 / +1.39,
   residual (full model) +1.29 / +1.14 / +1.20 (4/4); early / ipsi the same; chronic before the lick +.39 -> +.32 contra,
   +.30 -> +.29 ipsi (4/4). Posterior-medial transient partly movement: early contra chronic +.85 -> +.56 (3/4), ipsi
   +.54 -> +.21 (n.s.), subacute +1.16 -> +.94 (4/4); late posterior-medial -> ~0. Reading: the sustained orofacial
   increase is not a by-product of licking / face / jaw / running as measured. Not measured: body posture / paws
   (side cams pilot only). Next candidates: same increase as the pre-cue rise vs rest (Fig 9.8) / Figs 9.1-9.2;
   relation to behavioral recovery within animal.

27. **The shared increase is NOT explained by licking (Priya 11-09 "go ahead" on the behavior test;
   `scripts/state_space_shift_behavior.py`, SVD, lick-aligned, minus RESTnl; page v23 Fig 10.21).** Per trial: region
   means (orofacial = tjM1 + SSp-m + SSp-n; posterior-medial = RSPd + SSp-tr + SSp-ll; contra / ipsi) in three windows
   (-1.0..-0.1, 0..+0.3, +0.3..+0.7 s); behavior from the 3-D lick table via the landing trial_id (licks and contacts
   among licks peaking within 700 ms of the first, latency, reach, speed). Per animal x readout, OLS on PRE trials
   (spout FE + behavior z-scored on pre), pre sessions scored leave-session-out; observed change vs the spout's pre
   mean, predicted change from licking, residual. Orofacial contra late: observed +1.23 / +1.33 / +1.41, predicted
   -.21 / +.05 / +.16, residual +1.43 / +1.29 / +1.25 (4/4); early, ipsi the same; chronic before the lick observed
   +.38 contra / +.29 ipsi, residual +.41 / +.32 (4/4). Model sensitive: pre-stroke late orofacial activity rises with
   bout licks in all 4 animals (+.23..+.83 per SD); but acutely bouts SHORTEN (licks -0.98 SD, contacts -1.1 SD, 0/4;
   chronic +.44 / +.42, 4/4) while activity grows. Licking explains only the acute posterior-medial pre-lick activity
   (predicted +.47 from the longer acute latency; residual n.s.). Posterior-medial post-lick transient = residual (early
   contra +.65 / +1.30 / +1.03). Caveats: pre coupling model, not a causal control; jaw-only / body movement not in it.
   Next: jaw / face (FaceRhythm, jaw tables) and body around the first lick; relation to the pre-cue rise vs rest
   (Fig 9.8).

26. **The shared change against a real reference (Priya 11-09 "couldn't we use the rest or pre-cue lick-free ENL as a
   per session normalization?"; `WF_SS_SHIFTBASE=rest|enl scripts/state_space_shift_carried.py`, SVD raw + gsr; page v22
   Figs 10.17-10.20).** Every trial minus the session's RESTnl level, or minus its own pre-cue ENL window, before the
   templates; readout = per-area offset + shift (spout-mean change beyond gain, level kept). RESULT: the dips before
   and after the lick (B22 / B25) were the zero-window-mean construction -- against either reference the shared change
   is an INCREASE with two components: (1) a post-lick transient (0..+0.3 s) over posterior-medial / trunk-limb cortex
   (acute RSPd strongest, both sides; chronic vs ENL largest SSp-tr / SSp-ll); (2) a SUSTAINED anterolateral orofacial
   increase (tjM1, SSp-m, SSp-n, ALM; contra first, both sides by chronic), present BEFORE the lick against rest
   (chronic tjM1 / SSp-m contra + ipsi family-wise from -1.0 s) and growing to ~+1.0 after it. Only RSPd keeps a small
   late decrease chronically. Minus cortex-wide vs rest: chronic orofacial areas take a larger share all window (to ~+2
   late), ipsi SSp-tr / SSp-ll / VISa less (= Fig 9.2's pattern). Page take + Fig 10.6 caption + summary row rewritten
   (not annotated). Caveats: RESTnl itself may move with epoch (Fig 9.4); the ENL window can reach past the cue on
   short-latency trials (13 % pre vs 26 % acute).

25. **Where and when the shared shift is carried (Priya 11-09 "did you do the 'where is the shift carried' analysis?"
   -> "spatial and temporal change"; `scripts/state_space_shift_carried.py`, cohort SVD, raw + gsr; page v21 Figs
   10.13-10.16).** Shift per session as the part maps (B22). WHEN: energy concentrated just after the first lick
   (subacute / chronic peak +0.15..+0.25 s; share +.08..+.11 above pre); acutely raised across the whole window.
   WHERE x WHEN (area x bin, max-stat family-wise over cells): post-lick transient (0..+0.3 s) largest over
   posterior-medial / trunk-limb cortex both sides (chronic SSp-ll contra +.55, SSp-tr +.35..+.39) with a late
   undershoot; MOST CONSISTENT = contralesional orofacial cortex (SSp-m / SSp-n / tjM1: 12-16 of 17 bins significant
   subacute + chronic; below its window mean before the lick, above from ~+0.1 s on; SSp-m ipsi later). Minus
   cortex-wide: the orofacial areas carry the late rise (to +1.0, every epoch), RSPd the early peak then late fall.
   WHERE by size (energy share / pixel share vs pre null): broad -- no area > ~.05 above (RSPd contra +.03..+.05 around
   the lick, contra SSp-m +.02..+.04 before it; SSp-ll ipsi below). CAVEAT now on the page: the shift has zero window
   mean per pixel by construction (static level -> offsets), so "below pre before the lick" partly mirrors "above
   after" -- read as a change of time course relative to the area's own window mean (replaces the B24 "earlier /
   briefer response" reading).

24. **The split in every basis + what the parts look like on cortex (11-08 ~15:00; page v17 section 10).** Shares,
   acute / chronic: SVD = Allen (shift .30 / .13, per-area offset .16 / .34, per-spout offset .22 / .18, global gain
   .03 / .15, rotation .05 / .03, off manifold .01); LocaNMF same order, more per-area gain (.16 / .20) and a larger
   chronic shift (.22). MINUS the cortex-wide mode (WF_SS_XFORM=gsr): global gain -> ~0 and most of the per-area level
   goes (Allen chronic .078 -> .003, .247 -> .070 in units of pre response energy) -- both were the cortex-wide mode;
   the shared shift leads (chronic share Allen .24, SVD .31) with per-spout levels (.25-.26). Per session (Allen raw
   + gsr, LocaNMF raw): pre leave-one-out dots ~0 for every part; shift above pre in every epoch 4/4, acute y > 0 in
   every acute session; rotation ~.02, 4/4. "Response energy" (Priya asked) = sum of squares of the animal's pre
   template (spouts x bins x features) after removing each feature's mean -- the task modulation, NOT the level.
   PART MAPS (cohort SVD, `state_space_part_maps.py`): the SHARED SHIFT is a sharper lick-locked transient -- below pre
   before the lick, above from -0.1 to +0.4 s (peak ~+0.15 s, +0.6-0.8 pre pixel RMS; acute rise from ~-0.25 s),
   below again +0.4..+0.7 s; largest RSPd / SSp-tr / SSp-ll / VISa / MOp, bilateral, smallest ALM / tjM1. Minus the
   cortex-wide mode: posterior-medial up at +0.1..+0.4 s (RSPd +1.2) with lateral limb / trunk SSp down (SSp-ll
   -0.8), then lateral orofacial SSp (SSp-m to +1.3, SSp-n +0.8) up at +0.4..+0.7 s while RSPd falls (-0.8); every
   epoch. Reading: down-up-down = an earlier / briefer lick response minus the pre one; spout-independent and
   bilateral -> may be how licking unfolds after the first lick (bout length, contact, body movement). Other parts:
   gain adds over bilateral lateral orofacial / SSp (subacute, chronic); per-area offset 0 significant pixels
   chronically (reference-sensitive, as warned); per-spout level up at close ipsi (every epoch), down at far mid
   (chronic); spout-specific rest mostly n.s. except far contra (central / posterior band chronically = Fig 9.3).
   NEXT: (i) fit the shift as the pre response shifted / compressed in time; (ii) compare with lick-bout length /
   contact / body movement; (iii) repeat the split against RESTnl (the per-area level). All per-session runs done
   17:47 (page v20: SVD / LocaNMF versions of Figs 10.2 / 10.4 via the basis switch). SVD minus cortex-wide, per
   session (median energy, pre response energy = 1; pre / acute / subacute / chronic): shared shift .025 / .163 /
   .237 / .200, per-spout offset .051 / .159 / .145 / .156, global gain ~.003 throughout -- the shift is the largest
   part from subacute on, as at the animal level.

23. **What the change is made of: an exact split into eight parts (Priya 11-08 "split into parts" -> "per area shift?"
   -> "is there a per spout shift piece"; `scripts/state_space_change_parts.py`).** Per animal and epoch, Delta = epoch
   template - pre template (spouts x bins x features, first-lick-aligned), projected on nested pattern sets from the
   pre template: global gain, per-area gain, per-area offset (static level, all spouts), per-spout offset (static
   level per spout x area), shared shift (spout-constant, TIME-VARYING), rotation (small-angle, within the top-8 pre
   space, beyond all of the above), other on manifold, off manifold. Orthogonal: energies sum to |Delta|^2. Noise
   cross-fitted (pre thirds / epoch halves, patterns from an independent pre third, 20 splits). Floor = odd vs even
   pre days; chance = an isotropic random change through the same split (rotation .005, off manifold .68).
   ALLEN, raw, energy in units of the animal's pre response energy (floor -> acute -> chronic): per-area offset .011 ->
   .089 -> .247; per-spout offset .004 -> .100 -> .077; shared time-varying shift .001 -> .111 -> .052; global gain
   .009 -> .025 -> .078; per-area gain .002 -> .038 -> .021; rotation .001 -> .023 -> .017; other on manifold .002 ->
   .033 -> .039; off manifold .000 -> .007 -> .007. Shares acute / chronic: offsets .16 + .20 / .34 + .18, shared
   shift .27 / .11, global gain .04 / .18, rotation .07 / .04 (14x / 8x chance), off manifold .02 / .02 (chance .68).
   Readings: (a) the change is overwhelmingly ON manifold; (b) the largest chronic part is a static per-area LEVEL
   change -- reference-sensitive (signal drift-removed incl. the constant: a level relative to the session mean, as the
   no-subtraction maps, B16); (c) the time-varying shared shift is the largest ACUTE part and carries most of the acute
   sideways (y) displacement; (d) rotation is small but well above floor and chance -- the held-out "rotation beyond
   gain" ~0 chronically then says it does not generalize across spouts (spout-specific). The first 6-part run (11-08
   ~12:15, shift incl. offsets ~.5 of the change) is superseded by the 8-part split. In flight 11-08 ~12:45: SVD /
   LocaNMF, raw and minus cortex-wide (WF_SS_XFORM=gsr: the split inside the space left after removing each animal's
   pre cortex-wide mode, Priya "what if we look at the minus cortex manifold"), animal-level and PER SESSION
   (`--sessions`: each session vs its animal's pre template, pre sessions leave-one-out = null dots, energies in units
   of the animal's pre response energy; Priya "per-session dots ... normalized within animal"). Outputs
   change_parts{tag}/{parts,plane,sessions}.csv; figures page_figs/ss_change_parts[_sessions]{tag}.png.

22. **What the parts LOOK like (Priya 11-08 "what IS the shared shift, on cortex, over time"; script ready):**
   `scripts/state_space_part_maps.py`, cohort SVD (raw + minus cortex-wide): per session, sequential fit in the split's
   order -> gain map (0.1..0.5 s), per-area offset map, six per-spout offset maps, the shared shift as a movie in six
   windows (-1.0..0.7 s) + time courses in split-basis areas (ALM / tjM1 / SSp-m / SSp-n / SSp-tr / SSp-ll / VISa /
   RSPd / MOp / MOs rest, both hemispheres), and the spout-specific rest per spout (0.1..0.5 s); units = the animal's
   pre-template pixel RMS; nested-bootstrap contours vs the pre (leave-one-out) session maps. Chained after the SVD
   split lane; figures page_figs/ss_part_maps_{shift,shift_tc,static,spout_offset,rest}_lick_svd[...].png.

21. **Sideways (orthogonal) vs along-line shifts by basis (Priya 11-08 "see if the svd pre-stroke vs chronic and
   orthogonal axis shifts are different"):** SVD = Allen, LocaNMF differs. Lick, vs own pre, units of the pre ->
   chronic length (along acute / subacute; orthogonal acute / subacute): Allen +.08 / +.69, +.82 / +.23; SVD +.09 /
   +.67, +.83 / +.23; LocaNMF +.19 / +.40, +.94 / +.19 (per animal SVD within ~.05 of Allen; cue the same).
   Noise-corrected off-line share of the acute change: SVD .77-.92 vs Allen .76-.92; LocaNMF .82-.98. Clarified for
   Priya: "orthogonal" = the acute displacement points a different way from the chronic one (both inside the pre
   space); "rotation beyond gain" = whether the code is TURNED within the pre space -- a sideways displacement can be
   built from gain / shift / pattern change without rotation (B23 measures what it is made of).

20. **Sections 3-7 in all three bases (Priya 11-08 "render and add the SVD basis version of section 3-7 figures" /
   "and LocaNMF"):** `state_space_page_figs.py` is basis-aware (inputs `<dir><align><basis>`, figures
   `<name><align><basis>`, basis label top left; `state_space_sim.btag()`), and `state_space_noise.py` /
   `state_space_contact.py` name figures with the basis (they would have overwritten the Allen page figures). New
   runs (both bases, both alignments): `--condition on_target | landing | incomplete`, `path_tests --trial-alpha`,
   noise, contact, xz. Not possible: balance (5.1 / 5.2; named orofacial areas per hemisphere: Allen-only,
   `allen_only` -- in SVD it could be enabled via the exact area readout, not done) and the splice alignment (7.9).
   Page: a basis switch for sections 3-7 (scratchpad `state_space_page/add_basis_switch.py`) -- publish pending with
   B22 / B23.

19. **RESULT of B17: the maps differ by WHAT they measure, not by denoising.** `scripts/state_space_why_cleaner.py`,
   107 sessions (PS93_0813 dropped: no video, no RESTnl baseline), lick-aligned, RESTnl, chronic vs own pre (mean over
   spouts / far contra): r with the state-space map .63 / .86 at S0 (15r as rendered) and unchanged through S1 glue-
   free mask (.63 / .86), S2 one unit per animal (.63 / .84), S3 cohort SVD (.63 / .84). The 15r maps are ALREADY
   smooth and reliable: 99.6 % of their energy lies inside the 100-component SVD space, split-half r .989 (chronic
   mean) / .976 (far contra); none of the three "cleaning" steps moves the map, the reliability or the contour (stat-
   mask fraction .16 / .12-.13 at every step). The whole difference is S3 -> S4 (contour .47 / .28): the WINDOW and
   trials. 15r lick arm = 0..+2 s after the first lick (8 bins averaged, SVTcorr, camera-gated working trials); state-
   space = 0.1..0.5 s (drift-removed signal, first-lick trials). Figure `page_figs/ss_why_cleaner_lick.png`: both agree
   on the chronic bilateral anterolateral increase (tjM1 / SSp-m, inside every contour); they DISAGREE over central
   cortex -- blue (decrease) in the 0..2 s 15r map, red (increase, inside the contour) in the 0.1..0.5 s map. Reading:
   the "cleaner" look is an earlier window that excludes the later lick-bout phase (where the central change is
   negative), not less noise. The S3 -> S4 step cannot be split further without new extractions (the state-space
   trials stop at +0.7 s; the 15r cache holds only the 0..2 s mean). Also visible: S0 colors the painted-glue area
   (posterior lateral) -- the A9 mask bug in a rendered 15r figure. Next, if wanted: a 15r-style map at 0.1..0.5 s
   (one cached feature build per session with post_s=0.5; heavy-ish, not started) would separate window from trials.

17. **PLAN: why are the state-space change maps cleaner than the 15r activity maps? Measured one step at a time
   (Priya 11-08).** Start from the 15r lick / lick arm as rendered (rebuild session cache `prm-lick-lick-*`, the
   working-trial camera gate, RESTnl reference = `raw - restnl_base`, per-POSITION pre RMS over `brain_mask()`, which
   under the rebuild profile still holds the glue, A9) and change ONE thing per step toward the state-space map:
   S0 15r as rendered -> S1 glue-free analysis mask (scale + display) -> S2 one unit per animal (pre response RMS over
   all spouts) instead of per position -> S3 through the cohort SVD (U_c U_c^T on the mask: the low-rank denoising)
   -> S4 the state-space map itself (RESTnl, drift-removed signal, first-lick 0.1..0.5 s window, its trials). Per
   step and per panel (chronic, mean over spouts + far contra): r with S4, split-half reliability (odd vs even sessions
   within animal, change map r), fraction of the stat mask inside the nested-bootstrap contour, and rim enrichment.
   Expected (not assumed): S3 carries most of the gain in reliability; S1 / S2 mostly change the scale of the far-
   field. The 15r cache is read as files (no recompute); if the code digest moved and the cache is cold, stop and say.

18. **Pre-cue map (the trial's lick-free >= 1 s ENL window itself; Priya 11-08 "make it the >= 1 s lick-free ENL"):
   the sign depends on the reference.** Chronic, full, mean over spouts: vs RESTnl the pre-cue state RISES over
   anterolateral cortex (tjM1 contra +.98, SSp-m contra +.79, SSp-m ipsi +.74, tjM1 ipsi +.74, all 4/4, boot *; minus
   cortex-wide: tjM1 / SSp-m / ALM contra +.63 to +1.06, 4/4); vs the session mean (no subtraction) the same areas
   read as DECREASES (SSp-m -.79 to -.85, 4/4). Second time the session-mean reference flips a sign (B16): the
   session mean itself moves with epoch, so "none" is not a neutral reference here. Spout-specific pre-cue (far
   contra): no area change survives (2-3/4, CIs include 0) -- the pre-cue position code does not measurably move at
   area level. Units: each animal's pre RMS of the pre-cue map itself. All 10 sweep variants done 11-08 ~11:10.

16. **PS94's opposite sign without subtraction is mostly the session-mean reference (maps, lick-aligned, 11-08).** With
   no baseline (session-mean reference) PS94's chronic full change is negative almost everywhere; against RESTnl
   (drift-removed signal) its anterior change turns positive like the other three animals (tjM1 contra +.62, SSp-m
   contra +.55, pre-RMS units), and only a lateral central patch (both hemispheres, far ipsi / far mid) stays negative.
   RESTnl, chronic, full, mean over spouts (4/4, boot *): SSp-m contra +.70, tjM1 contra +.66, SSp-m ipsi +.62, SSp-tr
   contra +.53, SSp-n contra +.50. Spout-specific far contra under RESTnl repeats the loss over SSp-tr / SSp-ll / VISa /
   RSPd (-.17 to -.33, 4/4): baseline-robust. Figures `ss_cmap_*_rest_{lick_,}svd.png`.

15. **One cohort SVD basis (glue-free, 100 components) reads like the Allen space, not like LocaNMF.** Lick, chronic vs
   own pre: r -.045* [-.058, -.031], amp +.30* [.14, .45] (4/4), rotation beyond gain +.002 [-.008, .009] (n.s.),
   outside the pre space .008 (out_resid .008). Method check in this basis (`sim_lick_svd`; 11-08 corrected -- the
   first entry read the wrong columns): rotation beyond gain detects a main-plane (PC1-2) rotation of 20 deg .68 / 30 deg
   .87 (.65 / .77 at 15 trials per spout), a random-plane 45 deg .90; 5 % new dimensions 1.00 (out_resid). BETTER than
   in LocaNMF (`sim_lick_lnmfbm`: main plane 20 / 30 deg .48 / .68; 5 % new dims .65), so the SVD null rotation is not a
   power problem and the LocaNMF rotation is not a more sensitive readout. Figures: page Figs 2.2 / 2.3 (method
   check per basis), 3.10 / 3.11 (`ss_basis[_lick].png`, the readouts in all three bases). Minus cortex-wide (xgsr): alpha .36 / .58 / .91, chronic_like .20 / .31 / .50 (4/4 each); spout-
   specific (xspout): alpha .25 / .52 / .84. Reading for A8: "larger, same pattern" holds in two smooth large-scale
   bases (Allen areas, cohort SVD); re-patterning (rotation, new dimensions) appears only in the per-animal LocaNMF
   components, so it lives at the fine, local scale those components resolve (or in how per-animal bases are
   aligned). The FINDINGS 0c item 1 wording should say so, not pick one basis.

6. **LocaNMF basis (107 sessions, ~90 components per animal; rows = the Allen rows, checked) changes finding 0c item 1:
   in the finer basis the RAW change already looks re-patterned, not "gain + shift".** First-lick aligned, change vs own
   pre, chronic (Allen -> LocaNMF): pattern r -.04* -> -.21* (0/4 back both); amplitude +.30* (4/4) -> +.17 (3/4, n.s.;
   PS94 -.25); rotation beyond gain .000 -> +.050* (4/4); outside the pre space beyond gains +.003* -> +.043* (4/4,
   ~14x); shared shift +29* -> +23* (4/4). Holds on contact-only first licks (r -.20*, rot +.057* 4/4, out +.037* 4/4).
   Per animal chronic - pre (r / rot / out): PS92 -.17 / .07 / .09, PS93 -.26 / .03 / .05, PS94 -.30 / .07 / .01, PS95 -.11
   / .03 / .03. The LocaNMF raw picture resembles the Allen spout-specific variant (r -.20, out +.016), and the LocaNMF
   spout-specific variant goes further (r -.28*, rot +.071* 4/4). Unchanged in LocaNMF: acute change mostly off the line
   (B5), position along pre -> chronic (alpha .28 / .51 / .94), y tracks far-contra aim beyond the balance marker (pooled
   partial -.72 vs Allen -.69), x / y track aim at every far spout, gradual not step (step best in 3 / 24 fits). CAVEATS:
   the planted-change method check was Allen only -- how rotation / out-of-space read in a ~90-dim component space with
   the same top-8 is not measured (a rerun of state_space_sim in lnmf would answer it); component units, no ALM / tjM1
   split; PS94_0917 basis fit 61 %. Reading: the "consistent with the pre mapping, larger" summary is a property of the
   coarse Allen space; at the component level the post-stroke code is re-patterned from acute on. Proposed FINDINGS
   wording after the method check (A7).

12. **REBUILD MASK BUG (affects other arms -- decision for Priya, A9).** Under the rebuild profile
   (`cohort1_rebuild_20261103`) `paint_exclusion.mask_dir` resolves to `<rebuild>/labcams/grant_figures`, where the
   hand-painted glue masks do not exist, so `beta_maps.painted_exclusion()` silently returns an EMPTY mask and
   `brain_mask()` = Allen minus bulbs only: 192,110 px instead of 151,447 (glue IN). Measured: the rebuild encoding
   bundle's inputs (`enc_rebuild_20261104b/inputs/*.pkl`, imaging[2..4]) carry 192,110 brain pixels -> the rebuild's
   encoder fits, its per-session area masks (the Allen-area extract this arm reads: VIS / RSP areas under the glue
   included) and anything else calling `brain_mask()` under the profile include the glue. Default-profile runs are
   fine. This arm: the first brain-mask SVD (ss_svdbm, 192,110 px) also had glue IN -> deleted (49 outputs: the svd
   state-space set, chronic maps, cache_svd, page_figs *_svd) and rebuilt with the mask resolved before the profile and
   a guard that refuses an empty painted mask (`o2_state_space_svd.analysis_mask`). The LocaNMF glue drop
   (`component_exclusion`, no profile) was correct (28 / 22 / 24 / 28 dropped, = documented counts).
   Proposed shared fix (NOT applied, other arms' code): `paint_exclusion.mask_dir` reads the canonical painted masks
   regardless of profile (they are hand-made inputs, not run outputs), + a raise in `painted_exclusion` when no mask is
   found; then the rebuild's map / encoder / area-extract steps need a re-run.

11. **Where the change sits on cortex: full / minus cortex-wide / spout-specific maps (Priya 11-08 "can we map the
   'chronic' axis back to cortical regions" -> "lets see the spout specific, minus cortex-wide, and full versions").**
   `state_space_chronic_maps.py`, brain-mask cohort SVD (`_svd`), first-lick aligned (cue to run), evoked response
   (0.1..0.5 s minus -1.0..-0.6 s), change vs own pre in units of the animal's pre response RMS, mean of 4 animals.
   Figures `page_figs/ss_cmap_{full,nocw,spout}_lick_svd.png` (+ `ss_cmap_animals_*`), table `chronic_maps_lick_svd/
   areas.csv` (64 Allen areas; boot_delta_columns CI = * primary; k/4). Pixel contours: within-animal session shuffles,
   family-wise per panel -- they pool sessions, so they are more liberal than the animal-level area CIs (close ipsi
   spout-specific: pixel contour, area CIs include 0 at 2-3/4).
   - FULL, chronic: larger everywhere the response lives, all spouts; largest in contralesional SSp-tr / SSp-ll / SSp-m /
     MOp / VISa (+.5 to +.7 pre-RMS, 4/4; PS94 smallest). Acute: bilateral anterolateral increases at close spouts;
     far contra acute: an ipsilesional anterior decrease. Subacute: posterior-medial (right) increases.
   - MINUS CORTEX-WIDE, chronic: a re-weighting -- contralesional SSp-m (+.33, 4/4), MOp (+.16, 4/4), VISa / RSPd up;
     ipsilesional MOs (-.12, 4/4) and SSp-n (-.20, 3/4) down. Acute: anterolateral up, a central blob per hemisphere
     down (map).
   - SPOUT-SPECIFIC, chronic: the far-contra-specific response is SMALLER over a central / posterior-medial band
     (SSp-tr / SSp-ll / RSPd / VISa, both hemispheres, -.23 to -.35, 4/4; ipsi MOp / MOs / SSp-m / SSp-n -.08 to -.17,
     4/4). Acute: far mid / far contra lose bilaterally (anterolateral), close spouts gain ipsilesional anterolateral.
   EXPLORATORY; area names from the table (CCF), not read off the maps.

14. **PLAN: two-part change maps, "explained by the changed behavior" vs "beyond it" (Priya 11-08 "yes plan the two-
   part change map").** Delta data = Delta (frozen pre-stroke encoder's prediction from the ACTUAL post-stroke behavior)
   + Delta (residual). Inputs exist (checked): `<rebuild>/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/
   hv-...daqlk/frozen_encoder_gate-attempted_v7g3_t3d_spont_basis/<variant>/sessions/<label>.npz`, `evoked_lick` /
   `evoked_cue` = (12 parts x 6 spouts x 201 spatial targets): data, full, the seven groups timing / target / exec_locked
   / exec_within / lick / other / contact (+ spout, lick_all super-groups, spont); pre sessions LOSO-frozen. Targets ->
   pixels through `fit/spatial_<animal>.pkl`. Encoder windows: lick 0..1 s minus -0.3..-0.1 s, cue 0.5..1.5 s minus
   -0.5..0 s; engaged (gate attempted) first-lick trials.
   Steps: (1) BRIDGE -- the encoder's Delta data vs this arm's maps (same epochs, cohort SVD): r per spout / epoch, so
   differences in trials / windows / baseline are measured, not assumed. (2) every part -> pixels -> analysis mask ->
   cohort SVD (same smoothing as the maps), one unit per animal (its pre Delta-data RMS) for all parts. (3) per animal
   and epoch: Delta data, Delta full, Delta residual (= data - full, exact), Delta per group. (4) figures: rows data /
   predicted / residual, columns mean + six spouts, full + spout-specific (+ nocw), nested-bootstrap contours, per-animal
   + k/4. (5) numbers: share of the Delta-data energy the prediction carries (1 - |Delta resid|^2 / |Delta data|^2) and
   cos(Delta data, Delta pred) per animal x spout; which groups carry Delta pred (descriptive: groups are correlated);
   area table for all three. Local, ~10 min (119 small npz + four 133 MB bases).
   Open: (a) baseline -- the encoder's evoked parts carry ITS window (-0.3..-0.1 s before the lick); first pass uses it
   (consistent within the decomposition) and the bridge shows whether the baseline matters; a rest / none version needs
   `_predict_spatial` re-run with other windows. (b) the encoder was fit under the rebuild profile -> glue pixels IN
   (B12 / A9): drawn on the analysis mask, but the fit itself saw the glue; a clean version waits on the A9 fix.
   (c) over-control: movement that the stroke changed THROUGH cortex is removed with it -- read Delta residual as a
   lower bound on change beyond behavior.

13. **Masked LocaNMF re-run (`_lnmfbm`, glue / bulb components dropped).** Lick, chronic vs own pre (unmasked ->
   masked): r -.21* -> -.18* (0/4 back both); amp +.17 n.s. -> +.26* (4/4); rot_gain +.050* (4/4) -> +.046* (3/4);
   out_resid +.043* -> +.048* (4/4); shift +23* -> +27* (4/4). Contact-only: r -.17*, amp +.24* 4/4, rot +.059* 3/4,
   out +.048* 4/4. Spout-specific: r -.17*, rot +.067* 3/4. -> the re-patterning holds without glue components;
   amplitude now significant as in Allen. (Fixed on the way: gsr / z pre statistics used all cached columns -> 95 vs 67
   mismatch; now the masked columns.) Masked LocaNMF loops / method check: in `trajectories_*_lnmfbm`, `sim_lick_lnmfbm`.

10. **Brain mask only (Priya 11-08 "we should be looking only at the brain mask (without glue areas)" / "do we need to
   redo the locanmf as well with mask").** Audit: the Allen extract (`area_basis`) already uses `beta_maps.brain_mask()`
   (Allen mask minus bulbs and the painted fibre glue); the joint-LocaNMF bases (raw Allen mask) and the first cohort SVD
   (every pixel of the frame) did NOT. Fixes: (a) LocaNMF: components > 50 % of their mass in that animal's glue / a bulb
   dropped at load (`component_exclusion.occluded`, the repo's route, no refit): PS92 95 -> 67, PS93 87 -> 65, PS94
   90 -> 66, PS95 95 -> 67; outputs `_lnmfbm` (the unmasked `_lnmf` runs kept for comparison; B6 / B7 / B9 numbers are
   UNMASKED and get re-measured). (b) cohort SVD rebuilt on `brain_mask()` (192,110 px), run LOCALLY (`run-local`;
   inputs off the share), bundle `ss_svdbm_20261108`; the all-pixel `ss_svd_20261108` is superseded.
   PS94_0917 (58 % captured by the all-pixel basis): 98.6 % of what the basis missed is OFF-brain at the frame edges, 76 %
   of it in the first fifth of the session (bursts in the first ~20 min); inside the brain 98.9 % captured (neighbor
   0914: 99.5 %); off-brain pixels hold 47 % of its total variance (0914: 5 %). Not a bad session for any brain-masked
   readout; cause of the edge bursts not established.
   SVD hemisphere / area readouts: an Allen area mean is an exact linear readout of the components (mean of U_cohort over
   the area's pixels); vs the Allen extract on the same trials r = .999 median (58 areas, 3 sessions, min .93).
   DELETED (own derived, superseded, Priya's rule): state_space_recovery/cache_svd, trials_lick_svd,
   trials_contact_lick_svd, path_tests_lick_svd, trajectories_lick_svd, pertrial_lick_svd, step_vs_gradual_lick_svd,
   june_drift_lick_svd, page_figs/ss_{resp_size,traj_days,traj_loops,traj_loops_animals,traj_loops_time}_lick_svd
   (.png + svg/) -- the partial all-pixel SVD batch, stopped.

9. **Method check in the LocaNMF basis (A7; Priya 11-08 "run the method check").** `state_space_sim.py`, lick-aligned,
   `sim_lick_lnmf/` (+ `sim_lnmf/` cue; Allen lick `sim_lick/` for the like-for-like comparison; Allen cue = `sim/`).
   `run()` fixed to pass each animal's own columns to the ipsi-gain plant. All trials, detection (share of planted
   sessions outside the null's 95 % range; false-positive .065):
   - rot_gain (rotation beyond per-component gain): rotation in the top pre plane 10 / 20 / 30 deg -> .39 / .68 / .90;
     random-plane 45 deg .68; NOT produced by gains (.10-.16), new dimensions (.03-.07) or spread (.16-.19). Planted
     change sizes: top-plane 20 deg +.036, 30 deg +.079; random 45 deg +.045.
   - out_resid (outside the pre space beyond gains): new dimension 5 % / 15 % variance -> .74 / 1.00 (+.033 / +.094);
     random gains .30 -> .36 (+.004), rotations <= .39.
   - r (pattern correlation): weak in LocaNMF (null pre-pre r .81 vs Allen .93; detection <= .29; largest planted drop
     -.07 at 30 deg top-plane rotation).
   - 15 trials per spout: rot_gain top-plane 30 deg .81, new dims .61 / .97; everything else near chance.
   Reading against the observed chronic LocaNMF changes (B6): rot_gain +.050 ~ a 20-30 deg rotation of the main pre
   plane; out_resid +.043 ~ 5-6 % added variance in new dimensions; both readouts specific; r -.21 is larger than any
   single planted change (a combination, or a change the plants do not cover). Allen for comparison (same plants):
   detection similar or higher, so the Allen chronic nulls (rot_gain .000, out_resid +.003) are real "absent at the
   area level", not blindness. Proposed FINDINGS 0c item 1 wording: A8.

8. **Cohort SVD basis bundle (Priya 11-08 "could we use one svd basis" -> "yes").** `scripts/o2_state_space_svd.py`,
   bundle `<rebuild>/o2/ss_svd_20261108` (COMMANDS.md): stage 1 one task builds U_cohort (top 100) from the 31 August
   pre sessions of all four animals (joint_basis trick across animals, drift-removed V, equal total weight per animal;
   component side from footprint energy, left > 2/3, right < 1/3, else both); stage 2 projects all 119 sessions
   (rows reused from the LocaNMF bundle, so row i = the same trial in every basis). Waiting for Priya to submit.
   Desktop side (`WF_SS_BASIS=svd` in state_space_sim) to add when the outputs land.

7. **Loops in the LocaNMF basis (Priya 11-08 "can we do loops in locaNMF basis any way?").** `state_space_trajectories`
   `align_maps`: each animal's own fixed pre top-8 space, then generalized orthogonal Procrustes of the four animals' PRE
   spout x time templates (rotations / reflections, no scaling; post epochs out of sample), then the same cohort top-2
   PCA + nested bootstrap as the Allen loops. Test: `test_align_maps_recovers_a_shared_code`. CHECK in Allen
   (`WF_SS_LOOPS=aligned`, figures `ss_traj_loops*_{lick,cue}_aligned.png`): aligned vs shared-space loops r >= .998 in
   every epoch after a pre-only 2-D rotation, loop-size ratios vs pre within .03 (both alignments) -> the alignment is
   faithful. Leave-one-animal-out r of the aligned pre template: Allen lick .92-.98 (cue .73-.83); LocaNMF lick .79-.86
   (cue .64-.76) -- the LocaNMF animals line up less well. LocaNMF loops (`ss_traj_loops{,_time,_animals}_{lick_,}lnmf.png`):
   wide bands; acute dips on PC2 at the lick at every spout; pre far-spout loops larger than post. NOT the Allen picture;
   given the weaker alignment, read descriptively. Alternative raised by Priya ("could we use one svd basis"): one
   COHORT SVD basis from all animals' pre sessions on the shared Allen pixel grid (`joint_basis` trick across animals) ->
   shared components without alignment; needs an O2 run (proposed, not started).

5. **"Orthogonal" re-checked (Priya 11-08 "how are we defining orthogonal" -> "yes run"; `state_space_orth_check.py`,
   `orth_check{_lick,}{,_lnmf}/`).** The reported share is a LENGTH ratio |a - (a.u)u| / |a| per spout (a = acute -
   pre template, u = unit pre -> chronic), averaged over spouts, with no noise correction and no null. Cross-fitted
   (independent trial splits for the two acute estimates and for u; 50 splits): noise is only 2-12 % of the acute
   change's squared length, so the share barely moves: Allen lick .88 / .76 / .92 / .87 (PS92-95; cue .69 / .93 / .83 /
   .84); LocaNMF lick .99 / .94 / .77 / .82. As VARIANCE: Allen lick .77 / .57 / .84 / .75 (cue .47-.87). The along-line
   part is real (|cos| .40-.73; a random direction would give ~.07) and its SIGN differs by animal: PS92 / PS94 acute
   moves AWAY from chronic (Allen lick cos -.48 / -.40, position -.17 / -.18 of the pre -> chronic length), PS93 / PS95
   toward it (+.65 / +.50; +.18 / +.48); LocaNMF: PS92 ~0, the others toward. Reading: "mostly off the line" holds
   (~75-90 % of its length, ~55-85 % of its variance), but "80-95 % orthogonal" overstates it and the along part is not
   consistently toward chronic. Proposed rewording of FINDINGS 0c item 2 + the page: in section A (A6).

4. **June drift reference (LocaNMF basis, contact first licks): the chronic change is about as large as the June ->
   August-pre change** (`state_space_june_drift.py`, `june_drift{_lick,}_lnmf/june_drift.csv`; distances noise-corrected,
   in units of each animal's pre-pre median, fixed August-pre top-8 space). chronic_to_pre / june_to_pre, lick: PS92 1.17,
   PS93 1.17, PS94 1.04, PS95 .96 (cue 1.11 / .94 / .94 / 1.05). Direction: partly along the June -> August direction in
   PS92 / PS94 (cos .43 / .45 lick, ~20 % of the squared length; cue .49 / .27), not in PS93 / PS95 (-.14 / -.30; cue ~0);
   chance scale .03. Gaps: June -> August pre ~64 days, August pre -> chronic ~37-43 days. Pre-registered reading:
   comparable size -> NOT separable from drift + learning by distance alone. Caveats: June -> August includes learning
   (an upper bound on drift); June rows are DAQ first contacts vs 3-D contact licks later; PS94's pre-pre unit is large
   (2.3 / 3.1 raw vs .25-.46 for the others). Also: trajectories in LocaNMF reproduce the detour (acute share orthogonal
   to pre -> chronic .99 / .94 / .78 / .84, Allen .80-.95); PS94_0917 basis captures only 61 % (next lowest 85 %).

3. **RETRACT "trials move as a whole (no search-then-select)": the repertoire test has no power** (handoff item 4;
   `scripts/state_space_repertoire_power.py`, `path_tests_lick/repertoire_power.csv`; first-lick aligned, 33 acute /
   subacute sessions, 20 draws each). Planted two-state sessions (each trial at the pre OR the chronic template, P(chronic)
   = the session's own position on the line, plus the trial's real deviation) give the SAME statistic as the data:
   excess +.003 (planted mean) vs +.001 (observed); planted > observed in >= 95 % of draws in only 8 / 33 sessions, and
   per animal x epoch the planted fraction above observed is .15-1.0 (mostly ~.5-.6). Why: single-trial scatter along
   the pre -> chronic line is ~1.4 x the WHOLE pre -> chronic distance (median within-spout SD of the per-trial position,
   acute / subacute), while a two-state mixture adds at most SD .5 (variance alpha(1 - alpha) <= .25 against ~2). Trial
   states cannot be resolved at this noise level in the Allen-area space; the slide-vs-search question is OPEN (maybe the
   LocaNMF basis is less noisy -- to check when the O2 output lands).

2. **Within a session, the neural drift goes with the landing drift -- after the stroke only** (session-by-session join of
   the state-space within-session slopes with the decline arm's behavior slopes; `state_space_within_session_behavior.py`
   join, `pertrial_lick/within_session_join.csv`; slopes per normalized session; first-lick aligned). Post-stroke (76
   sessions): sessions whose far landings drift more ipsilesionally (first -> last trial) also drift more back toward the
   pre-stroke state on the pre -> chronic line: s_x vs all-far landing drift r .51 within animal (PS92 .19, PS93 .72,
   PS94 .60, PS95 .32; 4/4 positive; within-animal shuffle p < .0005); vs far-contra landing .39 (4/4 positive, .18-.52,
   p .002); vs contact drift .27 (4/4 positive, p .009). Latency drift: no consistent link (PS94 +.74, the others
   -.28 to -.50). Pre-stroke (31 sessions): no landing link (signs split 2/2), so it is not a generic session-state
   coupling. The acute-departure axis (s_y) links weakly the other way (all-far -.27, p .03; 3/4). Reading: within a
   session the neural position tracks where the licks go, the same coupling as across sessions (x and y track far-contra
   aim) -- consistent with cortex following (or setting) the executed aim, not separable here. Extends handoff item 10
   ("tracks landing drift in PS94 / PS93 only") to 4/4 with the decline arm's all-far measure.

1. **The acute-departure axis is mostly, but not only, the 0b balance marker** (`scripts/state_space_detour_vs_balance.py`,
   `state_space_recovery/detour_vs_balance{,_lick}/`; post-stroke sessions, 19 per animal). Within animal, y and the
   primary balance marker (cued first lick, ALM + tjM1 + SSp-m + SSp-n + SSs gain ratio, 0.1..0.5 s) share ~70 % of
   their variance (pooled within-animal r -.85 lick / -.83 cue; per animal -.55 to -.96: PS93 / PS94 ~-.95, PS92 / PS95
   ~-.6). Beyond the balance, y still tracks far-contra aim in 4/4 (partial r, lick: PS92 -.88, PS93 -.58, PS94 -.35,
   PS95 -.67; pooled -.69, within-animal shuffle p < .0005; cue: -.86 / -.32 / -.39 / -.55, pooled -.59). The balance
   beyond y is weaker and not 4/4 (lick .51 / -.10 / .44 / .62). Position along pre -> chronic (x) also tracks aim beyond
   the balance 4/4 (lick .78 / .43 / .50 / .60). Reading: the population axis contains the balance marker and carries
   additional aim-related structure, most where the balance is a weaker proxy (PS92, PS95). FINDINGS 0c item 3's caveat
   can be replaced by this (proposed wording in section A). CAVEAT: the balance values are the PRE-REBUILD run's
   (recprim_sessions.csv was never re-rendered in the rebuild); same sessions, different hemo variant / gate.

## C. Plans written down before the data arrive (so the test is fixed in advance)

- **Aligned-decoder arm of frozen vs refit (Priya 11-09; `scripts/aligned_decoder.py`, written and launched 11-09 ~17:40
  before any post number).** Data = the 5r / 5rm lick arm's refit inputs (D6 capture hook; rebuild profile, block
  epoch_accuracy, LP licks + LP gate, August pre; joint-LocaNMF components x 8 lick bins, session z). Per animal k = 10
  PCs of single-trial (trial x bin) activity (also 5, 20). Per session, 2-fold over position blocks; calibration half
  fits everything session-specific: frozen (pre PCs + pre decoder trained on a SIZE-MATCHED pre subset = the
  calibration size), (a) label-free covariance alignment (own PCs whitened, re-coloured by the pre latent covariance;
  axis correspondence = orthogonal factor of W_own' W_pre through the shared components, because the session z-score
  removes the mean trajectory), (b) orthogonal Procrustes + one scale on per-spout condition-averaged latent
  trajectories, refit (own PCs, calibration labels). Controls: rotation-only (a) and (b); (b) with shuffled calibration
  labels; pre -> pre against the LOSO pre reference (ceiling); pairwise single-pre-session references (k = 10) for the
  ceiling at matched lag. Readouts: gap fraction (aligned - frozen) / (refit - frozen) per animal / epoch (nested
  draw); change vs pre of every arm, A, R, G; index dA / dG. Cortical decomposition: the frozen decoder's readout
  carried to pixels through the footprints, aligned vs reference: contra / ipsi loading share in tjM1 and SSp-m (ROI v8)
  and the within-hemisphere cosine. **Verdict rule (fixed now):** gate = pre -> pre gap fraction >= .7 in >= 3/4;
  RESTORED = frozen and G not changed; RE-EMBEDDED = G changed, index >= .67 in >= 3/4, R not changed; REPLACED = G
  changed, index <= .33 in >= 3/4, A not changed; else PARTIAL; gate fails -> UNINFORMATIVE.

- **LocaNMF basis (O2 option 4).** `WF_SS_BASIS=lnmf` switches `state_space_sim` to the O2 output
  (`rebuild_20261103/o2/ss_lnmf_20261108/out/sessions`; same rows as the Allen extract for the 107 rebuild sessions),
  its own binned cache and an `_lnmf` output tag; every script then runs unchanged. Re-run: trials (change vs pre: r,
  amp, out-of-space), path tests (orthogonal detour), trajectories (x / y), step vs gradual, per position, detour vs
  balance. Question: do the 0c claims (gain + shift on the shared mode, re-patterned spout-specific part, orthogonal
  acute detour, y tracks aim) hold in the finer basis? Report per animal first.
- **June drift reference** (June 6/6-6/8, contact-only first contacts; LocaNMF basis only -- the Allen extract has no
  June). In each animal's fixed pre space (August pre template), compare (a) the June -> August-pre distance with (b)
  the August-pre -> chronic distance, both with the trial-noise expectation removed (`state_space_recovery_trials.vdist`),
  and (c) the angle between them (does chronic - pre lie along the June -> August drift?). Like-for-like: the August
  and chronic sessions restricted to CONTACT first licks (landing table) so all three use contact trials. June -> August
  includes learning, so (a) is an upper bound on drift over ~2 months; August pre -> chronic spans ~1.5-2 months.
  Reading: chronic change >> June drift and not along it -> beyond drift; comparable / along it -> not separable.
- **Batch, once `ss_lnmf_20261108/out/sessions` holds 119 files** (all outputs tagged `_lnmf`; page figures untouched):
  `WF_SS_BASIS=lnmf`, for WF_SS_ALIGN in lick (primary), cue: `state_space_recovery_trials.py` (+ `--condition
  contact`), `path_tests`, `trajectories`, `pertrial`, `step_vs_gradual`, `per_position`, `detour_vs_balance`,
  `june_drift`; then lick trials with `WF_SS_XFORM=spout` and `gsr` (the 0c "shared mode vs re-patterned
  spout-specific part" claim). Order matters: step_vs_gradual / detour_vs_balance read trials + trajectories,
  per_position reads pertrial. First check: a LocaNMF file's rows = the Allen file's rows (first_pos, first_rt equal).

## D. Done / status log

- 11-08 ~11:45: sweep done (10 variants, B16 / B18); all figures re-rendered with wrapped titles; SVD state-space set
  done (B15); "why cleaner" measured (B19). Page v15 published (https://claude.ai/artifact/9sGELLFEwADEt9k64fXTxe):
  new section 9 "Where on cortex" (Figs 9.1-9.10, table 9: RESTnl primary, ENL / pre-lick / none / cue / pre-cue /
  why-cleaner), summary rows for maps, basis dependence and the 15r comparison; open questions updated (two-part maps,
  15r window vs trials, A9). FINDINGS not changed: waits on A8 (wording should now cover B15 -- re-patterning only in
  the per-animal LocaNMF components, not in the cohort SVD or Allen areas).
- 11-08 10:40: maps -- baselines rest (RESTnl on drift-removed signal, 119/119 usable), enl (lick-free >= 1 s, slid back
  to the strobe from the continuous signal; median 99 % of trials kept), none; pre-cue map (ENL window); split-basis
  area table + ALM / tjM1 outlines; nested-bootstrap contours + per-animal shuffle contours / k/4. Final 10-variant
  sweep running. SVD state-space set re-run on the glue-free basis (chain svd_chain.sh). No-subtraction maps (lick +
  cue): far-contra-specific loss robust to baseline; full-change size depends on it; PS94 opposite sign (level vs
  session mean) -- rest / enl maps will tell.

- 11-08: June drift script written (`scripts/state_space_june_drift.py`; the plan in C exactly; contact first licks
  for every session; per-animal table only). Smoke-tested on a synthetic animal (June offset orthogonal to the chronic
  change -> cos at the chance scale, distance ratio follows the planted sizes). Waits for the O2 outputs.
- 11-08: cleanup item 4 verified + committed WITH the LocaNMF switch (326d1930): `WF_SS_BASIS=lnmf` -> loaders read
  `ss_lnmf_20261108/out/sessions`, cache `state_space_recovery/cache_lnmf`, outputs tagged `_lnmf`; components named
  `<Allen area>#<k>_<side>` from each basis's `regions.npy` (orofacial balance = SSp-m / SSp-n / SSs components; the
  LocaNMF atlas has no ALM / tjM1 -- they sit inside MOs / MOp there); cols per animal; June sessions get epoch 'june'
  and load only with `load_sessions(june=True)` (many scripts read "not pre" as post-stroke); balance_cells and pc2
  (areas pooled across animals) refuse. Allen path unchanged (the last four verification scripts ran on the new code:
  identical). O2 outputs: still none on the share at ~01:30.
- 11-08 night: O2 bundle `rebuild_20261103/o2/ss_lnmf_20261108` staged on scratch by Priya (U_atlas 119/119; bases need
  the mkdir -p rerun -- see COMMANDS.md); jobs not yet confirmed submitted.
- 11-08: cleanup items 2, 3, 6, 8 committed (3fdd7838), outputs verified identical (44 CSVs, cue + lick).

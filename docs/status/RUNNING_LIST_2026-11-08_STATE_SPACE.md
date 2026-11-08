# Running list, 2026-11-08 overnight (state-space arm, unattended)

Priya 11-08 ~23:30: "i'm going to sleep - please continue to proceed through tasks, keep a running list of findings, and
any questions or decisions you need me for." Order agreed before: (4) cleanup -> (1) detour vs balance marker -> (2)
LocaNMF switch + June plan -> (3) neural vs behavioral within-session drift -> (5) repertoire power check -> (6) housekeeping.
Branch `analysis/state-space-recovery-20261107` (worktree wf_statespace); nothing pushed.

## A. Decisions / questions for Priya (newest first)

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

# Findings: the current science conclusions (VLS stroke, PS92-PS95)

**The one place to read what we currently conclude.** Each claim is 2-4 lines: what, the key numbers, how many animals
agree, its status, and where the evidence is (figure / results page / detail doc). The per-arm FINDINGS documents stay
the detailed record (nothing is deleted from them; superseded claims are marked there in place). Started 2026-10-25
(Priya: consolidate the findings documents; all windows).

**Status tags.** **STANDS** (survived the listed checks on the data named) - **PROVISIONAL** (a control, a per-animal
check or the four-animal set still pending) - **CORRECTED** (an earlier statement was wrong; the corrected one follows) -
**WITHDRAWN** (no longer claimed) - **NOT ESTABLISHED** (tested, not supported; kept so it is not re-claimed).

**Cohort rules.** Four mice (PS92-PS95), all lesions LEFT ventrolateral striatum: RIGHT = contralesional for spouts,
licks and the body. Image sides: brain / encoding maps ipsilesional = image LEFT; cam1 + cam4 face video ipsilesional =
image RIGHT. Consistency threshold: >= 3 of 4 animals (Priya 10-24); every claim needs the per-animal table first
(CLAUDE.md rule 8). PS93 had a right orofacial deficit BEFORE the stroke and is named where it is the exception.
August-only pre-stroke baseline. Epochs: pre / acute / subacute / chronic (`wfield_local.epochs`). Changes vs pre:
`analysis_kit.boot_delta` (paired animals -> sessions, animal-weighted, 95 % interval); `*` = interval excludes 0,
`**` = also survives Holm over the figure's family.

**Owners (edit your own section; fetch + rebase before pushing).** 1-2: pose / 3-D window (5b). 3: imaging decode /
state control / anatomy (claims extracted from the docs by d4; owner to confirm). 4: encoding window (d4). 5: FaceRhythm
window (e3). 6: histology. 7: everyone.

**Results pages.** FaceRhythm https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1 - goal question / encoding
https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d - imaging https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV - licking kinematics (2-D + 3-D, sections 1-2) https://claude.ai/artifact/DKcKRJKbhBoratxxRc6qZv

## 0. Synthesis: how cortical motor encoding changes after the stroke (10-25, d4's reading; all arms may edit)

*The goal question: how does cortical motor encoding change after a VLS stroke? This section reads across sections 1-5;
each statement points to the claims it rests on and inherits their status. Frame named where it matters (FROZEN pre-stroke
encoder = headline for every change claim since 10-25; per-session refit = supporting).*

**In one line (revised 10-25 afternoon after analyses 1-2):** acutely cortical activity stays mostly consistent with the worse movement under the
pre-stroke mapping (shared variance; direction -- cortex driving vs following the movement -- not determined), with a
transient orofacial-specific disruption; chronically the movement is largely back and most of the cortical excess has
shrunk with it -- what persists is a residual in BOTH windows (orofacial-specific in the cue window; in the lick
window larger in orofacial cortex but shared with non-orofacial cortex), a contralesional shift of the spout package and more position-free movement coding, with less spout-specific tuning; the large chronic mapping change is
cortex-wide and grows with time -- not concentrated in orofacial cortex; whether it is a stroke effect (diaschisis, state)
or slow drift cannot be decided without a time-matched no-stroke reference.

**Where things stand, face movement (FaceRhythm, updated 11-07; claim 5.10, DECISIONS 2026-11-07):** tested per
session against each animal's own pre (within-animal shuffles, Holm across every epoch and measure, drift-controlled),
both lateral lick factors CHANGE SHAPE after the stroke and lose side specificity at every epoch (lick_right spatial
+14.6 / +7.6 / +5.5 deg, side specificity -.38 / -.25 / -.13; lick_left +12.1 / +10.5 / +9.9 deg, -.31 / -.23 / -.10),
lick_left moves toward the contralesional face (+.23 / +.18 / +.10), and lick_right's weight leaves the central face
for the ipsilesional cheek and jaw (map, 4 / 4 animals' sessions); none of this is pre-stroke drift or the loss of
right licks. Acutely the lick rhythm slows (video licks: within-bout interval +15-25 ms at every spout) and the face
movement shifts to lower frequencies (spectrum centre 6.77 -> 6.56 Hz, 4 / 4 animals, Holm .0003; the upper lick
band falls more than the main lick rhythm), the pattern slower or less sharp licks produce; the 7 Hz factor is not
LOST more often per session after correction. Map tests use 10,000 shuffles (11-07 night). PS93, with a pre-existing facial deficit, is inconsistent.

**Where things stand, quiet-period state (updated 11-02, claim 3.Q; handoff STATUS_2026-11-01_QUIET_EVENTS_HANDOFF):** in quiet
periods, acutely orofacial 1-4 Hz power rises (no lesion-side bias; the earlier 'more so on the lesioned side' is
WITHDRAWN, T22) and a normally rare, local, non-movement 2-3 Hz event type (common pre-stroke only when the animal disengages)
becomes ~7x more frequent while the animal is engaged (face-regressed x6.7 acute / x3.1 subacute / x1.5 chronic, LP-lick
quiet frames, 4/4); not face movement, not interictal, not haemodynamic (415 nm), not breathing (weak video proxy). A
pre-cue event goes with a weaker position code and with NOT attempting far contra (+0.028, 4/4), not with failed attempts.
No laterality support for Priya's reduced-thalamocortical-drive hypothesis after T22 (PROVISIONAL; 3.Q).

1. **Behaviour changed, and recovery is compensation, not restoration.** Acute global slowing + far-contra-graded deficit,
   licks aimed ipsilesionally from launch, jaw stops steering, far-contra trials attempted but failed in 3 / 4 (1.1-1.8,
   2.3, 4.4). Chronic completion returns, but each contralesional lick moves the contralesional face less and contra
   spouts are reached with more of the ipsilesional lick pattern, re-timed later (5.2-5.4, 5.6). -> Cortical change must
   be judged against behaviour that differs even when it succeeds; hence the frozen frame.
2. **Acute: losses are mostly the movement, plus a transient decoupling.** Frozen decoder loses far contra (.90 -> .32;
   ~.11 more under the 11-03 bal_em spout decoder, near spouts ~.05 less, 3.1), position ~11x more than state acutely
   (3.1, 3.2; bal_em state decoder, 11-03; position side to re-render under bal_em). The frozen encoder predicts the acute far-contra drop from the changed lick;
   target part ~0, with FaceRhythm too (4.1). Frozen pixel maps: the acute fit loss is cortex-wide and movement-driven (page
   Fig 1l). On CONTACT TRIALS alone (11-05, `_daqlk`, wording Priya 11-05): the frozen pre-stroke encoder explains less
   trial-to-trial variance after the stroke in every epoch (acute -.09, subacute -.07, chronic -.17 R2), lower in all
   four animals acutely; the bootstrap interval excludes zero but the mixed model does not reach p < .05 (p .06-.09) --
   a consistent TREND, not a significant result. Not only the contact / incomplete mix, but PS92's decline starts
   before the stroke (docs/status/reanalysis_20261105/FROZEN_B12_CIS_2026-11-05.md, FROZEN_FOLLOWUP_2026-11-05.md). Beyond the movement: the decoder weakens for every spout, even on-target licks (4.5); acute far-contra trials
   are each decoded CONFIDENTLY as some other spout, which one varying trial to trial and by animal (4.6b, 10-25 night);
   trial-by-trial coupling of cortex to execution is ABSENT ON FAILED ATTEMPTS (4/4) -- the acute "r ~ 0" was mostly
   that 92 % of acute far-contra trials are failed attempts; the few acute hits look like pre-stroke hits (thin; 4.6).
3. **Chronic: same task, partly different cortical solution.** Residual: cortex responds MORE than the pre mapping
   predicts for the same behaviour, bilateral ALM / tongue-jaw motor + contralesional SSp, not face movement (4.2) -- but
   (analyses 1-2, 10-25) it is partly GLOBAL in the lick window, largest at day 1, shrinks with recovery and tracks the
   deficit (mostly PS94), with an orofacial-specific part persisting at recovery in the cue window; not an
   extrapolation artefact (unchanged on behaviour-matched trials). Layer 2's chronic mapping change is
   NOT concentrated in orofacial cortex: non-orofacial cortex changes as much, rising steadily with time -- cortex-wide
   (stroke-related or drift: undecided); only an acute orofacial excess is orofacial-specific (4.3). Frozen
   frame: target-unique coding stays down, the PETH spout code does not recover, the spout package shifts contralesionally,
   position-free movement gains unique share in SSp-m, ipsilesional SSp-m decouples from the face's side signal (4.11, 4.12);
   template shape stays off 4 / 4 (3.12).
4. **Target vs execution -- how the answer changed with the frame (10-25 night; the key reconciliation):**
   (a) The old "target > executed" result was CHRONIC and PER-SESSION REFIT only (binned contest: target bins beyond
   executed bins +9.0 lick / +9.8 cue chronic; acute +12.5 n.s.; the pre-stroke "target, not executed" was already
   SUSPENDED). It never conflicted with the ACUTE result (frozen decomposition: the acute far-contra drop is the
   predicted lick part, target ~0).
   (b) In the frozen pre-stroke frame the chronic target gain disappears (target-unique stays down, 0/4 up).
   (c) Self-fit (each session refit on itself, frozen windows) brings it back (SSp-m +0.017 / +0.019, 3-4/4): target
   information IS present chronically, but in a NEW pattern the pre-stroke mapping cannot read -> target coding REMAPPED.
   (d) Partial refit (only one group refit, against its shuffled-label null): chronically BOTH are specific in
   contralesional SSp-m -- target +0.034, executed movement +0.054 / +0.064 (4/4) -- so the chronic remapping is NOT
   target-only; executed movement is remapped at least as much. (Confirmed 10-26 against 5-draw block-permutation
   nulls: target +0.036 / +0.034, executed +0.055 / +0.064, 4/4.) Acutely both are non-specific except a target-specific gain in ipsilesional SSp-m.
   (e) Per-spout lick kernels in the frozen model (both terms spout-specific and lick-aligned): pre-stroke, target and
   spout-specific lick kernels carry similar unique variance (target ahead only in contralesional SSp-m, 0.015 vs 0.012
   cue; ipsilesional SSp-m / MOp favour the lick kernels), and the lick kernels absorb part of the old target term
   (contra SSp-m 0.020 -> 0.015) -- part of "target" was spout-specific lick-locked activity. Post-stroke (frozen) neither
   gains; only the executed spout-locked templates do (+0.023 acute, +0.008 chronic, 4/4) = the changed movement.
   -> READING: acutely cortical activity stays consistent with the changed movement under the pre-stroke mapping
   (direction not determined); chronically neither target nor executed spout coding recovers
   in the pre-stroke frame, while with session-specific weights both are readable in a remapped form. "Target >
   executed" is better read as "target remapped, executed remapped too". (f) SELF-FIT on the per-spout lick-kernel
   design (10-26, ENCODING_FINDINGS 8.19): with spout-specific lick kernels present, the target's chronic remapping
   roughly HALVES (contra SSp-m cue +0.014 -> +0.005 n.s., lick +0.018 -> +0.010) and the lick kernels' own remapping
   is ~0, while the whole spout group (target + executed + lick kernels) stays remapped (+0.027 / +0.039 cortex, 4/4):
   the remapped information is SHARED by the target and the spout-specific lick-locked terms, so unique variance
   cannot assign it to either. Landing-split lick kernels (by where the tongue landed) do NOT absorb the chronic target
   term in the frozen frame (contra SSp-m target unique +0.010 / +0.011, 4/4). The lick-kernel PARTIAL refit (with
   permutation nulls) is still running. Block-permutation nulls DONE 10-26 (same numbers as the single shift). Failed attempts keep distance, lose side (4.9).
4b. **Recovery = behavioural compensation with PARTIAL cortical recovery, not restoration (10-25 night; key point).**
   **NOTE (Priya 10-30, important; 4.22): the residual ipsilesional bias is the SAME in cued and spontaneous (ITI) licks**
   (ITI landing vs pre after the same previous spout -7.5 deg chronic vs cued first-lick landing -6.8 deg; gap -0.7 n.s.):
   no capacity-vs-performance gap, so the cortical ITI persistence does not reflect a deficit that cued licks overcome.
   Judge recovery PER SUCCESSFUL TRIAL, not per epoch: acutely ~92 % of far-contra trials are failed attempts, which carry
   no far-contra code (decoder retention ~0 in every window, 4.9 / page 7e) and no trial-by-trial coupling (4.6), so
   epoch-level "cortical recovery" is partly just hits becoming common again (composition). On hits:
   (a) BEHAVIOUR is compensated, not restored: success returns, but contralesional licks move the contralesional face
   less and contra spouts are reached with more of the ipsilesional lick pattern, re-timed later (point 1; 5.2-5.6).
   (b) The far-contra CODE recovers only partly: frozen-decoder retention on chronic hits = ~70-77 % of pre-stroke hits
   (lick -0.30 [-0.46, -0.13], cue -0.24 [-0.40, -0.11], pre-cue -0.27 [-0.50, -0.04]; session values, nested
   bootstrap; page 7d / 7e); the PETH spout code and target-unique R2 do not recover in the frozen frame (4.11, 4.12b).
   (c) TRIAL-BY-TRIAL COUPLING to the executed movement on hits IS back at the pre-stroke level (chronic -0.07 / -0.06
   n.s.; page 4i): within a session, cortex again follows where the tongue goes.
   (d) Cortex does it with a DIFFERENT mapping: on behaviour-matched trials (inside the pre-stroke movement envelope) the
   residual is unchanged (4.2) -- same movement, different cortical response; session-specific weights fit better
   chronically (self-fit gap +0.06-0.09, 4/4) and target / executed coding are readable only in a remapped form (point 4).
   (e) Timing: NO EVIDENCE THAT CORTEX LEADS behavioural recovery -- but no good evidence it lags either. Half-change
   days (spout unique vs far-contra contact, page 1i): cortex later in 3/4 (PS92 +6, PS93 +2, PS95 +1 days), earlier in
   PS94 (-2); differences within the early session spacing (1-4 days), no CIs, a cortical measure without a consistent
   chronic gain, all trials (composition). HITS-ONLY TEST (10-25 night, `scripts/recovery_timing.py`, page 1i2): far-contra hit rate vs the decoder's far-contra code on HITS per session. Half-recovery days (exponential fits, session resampling): cortex - behaviour PS92 -1.5 [-3.8, +0.7], PS93 +1.9 [-3.8, +135], PS94 -5.7 [-10.7, +6.4] (cortex series starts day 7: no earlier hits), PS95 -1.8 (CI unbounded) -> no consistent order, every CI includes 0; by eye PS92 / PS93 move together, PS95's cortex is near pre by day 5 while behaviour lags. Session-to-session: neither predicts the other beyond its own past (behaviour[t] ~ cortex[t-1] +0.03 [-0.30, +0.37]; cortex[t] ~ behaviour[t-1] +0.18 [-0.09, +0.40]; 60 pairs). Clearer: behaviour returns to ~100 % while the code on hits plateaus at ~50-75 % in PS92-94 (PS95 ~95 %). POOLED (page 1i3 / 1i4; per-animal normalised, nested bootstrap): behaviour chronic +0.01 [-0.03, +0.06] vs pre (fully back); cortex on hits chronic -0.30 [-0.46, -0.13]; per-session GAP cortex - behaviour subacute -0.03 (none) -> chronic -0.31 [-0.47, -0.13]: both climb together to ~0.8 by day ~9, then behaviour completes recovery and cortex plateaus -- the dissociation opens in the chronic phase.
   DIRECTION: encoding models measure shared variance; "cortex follows the movement" and "cortex drives the movement"
   are both consistent with them. Hints: failed attempts are directed movements without a matching cortical spout
   signal (generated without spout-specific cortex?); hits-only coupling hints touch feedback; the weaker pre-cue code on
   failing trials is suggestive only. Tests: within-trial lead / lag (pre-lick vs post-lick prediction of the lick's
   direction), lagged FaceRhythm kernels, pre-cue -> lick coupling.
   -> Behaviour comes back through a changed movement strategy; cortex regains its moment-to-moment covariation with the
   movement but with ~3/4 of the original spout code and a partly remapped, not restored, mapping. Caveats: n = 4; the
   cortex-wide part of the remapping cannot be separated from drift without a time-matched no-stroke reference; whether
   the remapping is compensatory (helps behaviour) is untested (correlations with recovery only, 4.2 / 4.3 mixed models).
4c. **Do early cortical features predict the chronic outcome per animal? EXPLORATORY, n = 4** (10-25 night,
   `scripts/early_predictors.py`, page 1i5). Exact Spearman over the 24 orderings (perfect = p .083 two-sided / .042
   one-sided; nothing else can be significant). Perfect orderings: (i) LESION SIZE orders behavioural recovery speed
   (half-recovery day PS93 3.9 < PS92 4.7 < PS94 7.0 < PS95 7.9; lesion 45.0 / 46.7 / 67.7 / 81.9 %) -- structural, not
   cortical, and the one prediction plausible a priori (one-sided .042). (ii) Acute far-contra retention (all trials)
   orders the chronic cortical code on hits INVERSELY (rho -1) and the chronic cue residual positively (+1): animals that
   kept more far-contra code acutely end with a LESS pre-like code and a larger residual chronically -- but acute
   retention is near 0 in 3/4 and PS94 alone sets the order (PS94 is extreme on early residual, acute retention, chronic
   code and chronic residual), so read (ii) as "PS94 differs". Day-1 / days 1-3 residual: chronic code rho -0.8, chronic
   residual +0.8 (persistence). Chronic behaviour has no between-animal variance (all ~1.0), so it cannot be predicted.
   PRE-REGISTER for the next cohort (direction stated): lesion % -> slower behavioural recovery (+); day 1-3 orofacial
   residual -> lower chronic cortical code on hits (-) and larger chronic residual (+); acute far-contra retention ->
   chronic code (sign open: - here); acute coupling on failed attempts -> recovery speed (open).
4d0-DAY. **10-28 daytime additions (page v147-v166; handoff 00-COMPACT-12). PROVISIONAL.** (1) LICK-LOCKED ACTIVITY
   (section 14): ITI bout-start response larger acutely across cortex (41 areas) and reverting by chronic (5 areas: tjM1
   both sides, contralesional MOp-rest, SSp-m); FIRST CUED lick response larger in every epoch and still in 35 areas
   chronically (tjM1 contra +5.0 / +5.9 / +5.4 x1e3 dF/F*s); cue response unchanged acutely, up subacute / chronic.
   Far-contra target trials: acute lick response NOT up, acute cue response LOWER (ALM contra -3.4, 27 areas), both up
   chronically. Window (0-0.3 .. 0-1.5 s, peak) does not change the cued effect; the chronic ITI effect grows with longer
   windows (0.53 -> 1.01 of pre). (2) OFFSET CONTRIBUTION MAPS: offset totals agree across bases, decoder weights share
   the broad lateral structure but agree only modestly in detail (ITI weight-map r 0.13-0.39; cued contributions ~0) --
   not an area-level attribution. (3) NON-LINEAR decoder: cued offset larger (+0.06 / +0.08 / +0.05 mm, 4/4; subacute
   +0.11 vs linear +0.03), partly kernel shrinkage on unfamiliar post-stroke patterns (extra offset ~2x larger where few
   pre licks landed); same-target and per-lick-order versions running. (4) Decoded vs actual position per spout (13.7):
   cued far-spout read-out follows the landing about half way; ITI read-out stays near pre while landings move
   ipsilesional.
4d0-NIGHT. **Overnight 10-28 controls (page v146; docs/status/OVERNIGHT_20261027.md). PROVISIONAL.** ITI offset
   ROBUST: same-target same-place +0.26 / +0.24 mm acute / chronic (4/4); covariates (kinematics + FaceRhythm) explain
   5-7 %; non-linear (RBF kernel ridge) decoder: same offset and same r loss; angle offset patterned (uniform part ~0),
   same cortex as lateral (maps r 0.90-0.95). Behaviour-only decoder (FaceRhythm + reach / speed): ITI r unchanged
   post-stroke vs cortex -21..-27 % -> the ITI change is cortical. CUED offset mostly TARGET COMPOSITION (same-target
   +0.01 n.s., 2/4): post-stroke misses read toward their target as pre misses were (10-28, Priya OK: re-worded everywhere --
   the cued offset is NORMAL TARGET CODING of more misses, not a changed read-out; first cued lick +0.31 -> +0.05 n.s.). Maps: ITI cortex-wide components = variance effect (z-scored check); FWER: ITI lateral pattern shape unchanged,
   cued lateral pattern changes acutely + chronically; pixel LMM: cued lateral map drops, partly returns, stays changed at
   full behavioural recovery; far-contra maps scale with first-lick aim deficit.
4d0-V8. **SYNTHESIS on ROI v8 + Allen areas WITH the ALM / tjM1 split (all decoders; DECISIONS 10-27 late), Gaussian
   same-place like-for-like reference; supersedes the numbers in 4d0-LATE / 4d0 / 4d0b. PROVISIONAL** (page v140; v2 -> v8
   in docs/status/V2_VS_V8_SPLIT_20261027.md). HOLDS: cortex reads licks CONTRALESIONAL of where they land -- ITI every
   epoch +0.37 / +0.23 / +0.28 mm (4/4 each); cued acute +0.16 (4/4), first cued lick +0.31 / +0.22 / +0.15; cued far
   contra acute +0.22 (4/4) [10-28: TARGET COMPOSITION, not "read ahead of the failing movement" -- same-target
   reference +0.01 n.s., 4d0-NIGHT]; cued far contra chronic -0.27 (4/4, ipsilesional). Lateral frozen r: ITI -26 / -22 / -24 % (4/4), cued -20 / -12
   (4/4) / -8 % (n.s., 3/4). CHANGED vs v139: (1) "read further OUT acutely" is GONE -- cued down +0.20 -> +0.03 n.s.;
   cued read CLOSER subacute (-0.25) and chronic (-0.13), 4/4; cued reach like-for-like acute n.s., -0.27 / -0.14 later
   (4/4); (2) SAGITTAL r is lost too (rebuild 11-07): cued -25 / -21 / -8 % (4/4; the later part = short licks missing the spout),
   ITI -13 n.s. / -23 / -24 % (4/4), as large as lateral from subacute on;
   (3) cued LATER licks also read contralesional acutely (+0.14, 4/4; was first lick only); (4) no response remains at
   pre aim in any area (16.12); (5) the pull-based details of 4d0 / 4d0b (60 % expected from the pull; read/landing ratios
   0.35-0.60) are obsolete with the measure. REFIT DECODERS (lick_refit_decoders.py, page 10.2b): position is still
   decodable post-stroke -- lateral ITI frozen -0.16 / -0.14 / -0.15 vs epoch refit (vs pre with matched training n)
   -0.07 [-0.10, -0.04] (3/3) / -0.04 n.s. / -0.02 n.s.; cued acute frozen -0.16 vs refit -0.08 (3/3); angle same. So
   after the acute epoch the code is intact but the frozen read-out no longer fits it; acutely a smaller real loss remains.
4d0-LATE. **SYNTHESIS, 10-27 late (supersedes the measure in 4d0; conclusions unchanged). PROVISIONAL.** Primary offset
   measure = SAME-PLACE like-for-like read-out (4d0j; DECISIONS 10-27 late): cortex reads licks CONTRALESIONAL of where
   they land -- ITI licks in every epoch (+0.31 / +0.21 / +0.24 mm, 4/4 each; mixed model with time: step at day 1, no
   recovery), cued licks acutely (+0.14, 4/4) and on the FIRST cued lick in every epoch (+0.27 / +0.20 / +0.15, 4/4);
   cued far contra reads IPSIlesional chronically (-0.28, 4/4). Structure (decomposition 4d0g): ITI = a uniform
   contralesional shift + compression; cued = compression (lateral-r loss) + a small uniform ipsilesional shift. Not
   explained by the larger pan-cortical lick response (its decoder prediction has the opposite sign, 4d0i). Misdirected
   cued licks are read toward their target (normal target coding; whether stronger than pre cannot be decided). Lateral r:
   ITI -0.21, no recovery; cued -0.21, recovers. 'Beyond the pull' numbers below are RETIRED as primary (sign depended on
   the assumed pull slope); the lick-order slope result (4d0e) was mechanical (decoded flat with lick order).
4d0. **SYNTHESIS (Priya 10-27): the cortical lick read-out is SHIFTED after the stroke -- acutely contralesional and
   further out; the distance shift resolves, the contralesional shift lessens but does not resolve, more for ITI than
   cued licks. PROVISIONAL** (page v122 "Where things stand" first row; figs 3.37, 3.39-3.43, 3.48). Frozen decoder,
   z-scored, offset beyond the pre pull vs pre: lateral ITI +0.33 / +0.22 / +0.25 (4/4 each; PS94 +0.55 chronic), cued
   +0.19 (4/4) / +0.05 / -0.03; distance (down ~ sagittal) +0.20 acute (ITI 3/4, cued 4/4), ITI back (-0.03), cued
   reverses to CLOSER chronically (-0.14, 3/4); cued forward -0.22 -> -0.17 every epoch (4/4). Per-lick error: shift,
   width unchanged (ITI +0.89 / +0.48 / +0.43, 4/4); ~60 % of the acute cued shift is expected from the pull alone
   (control +0.28 of +0.47). Behaviour on the same cued licks: landing vs own pre landing at the spout -0.53 mm acute
   (4/4; far contra -1 to -3 mm) [10-28: read like pre misses of the same target, 4d0-NIGHT]. Tracks recovery within animal (3.37). ITI lateral r
   loss beyond drift with the JUNE no-stroke reference: -0.08 [-0.15, -0.02] chronic, 4/4 (August-only fit -0.13; the 8-day
   curve under-predicted 64-day drift in PS92 / PS95). Landing measure note: the lick table's closest-approach `lat_err_mm`
   hides most of the deviation (frame chosen nearest the spout); use extension-peak landing vs own pre landing.
   Pattern-map stats (3.50-3.51): held-out per-session Haufe maps, per-pixel nested bootstrap + BH + >= 3/4 sign; change
   maps dominated by one bilateral component (likely the cortex-wide response increase); Allen-area version
   `scripts/iti_pattern_area_stats.py`. Our FDR differs from the decoder page's FWER max-stat (section 14).
4d0b. **Far-contra CUED licks: the cortical read-out follows the ipsilesional deviation but holds back toward the
   target acutely, and over-reads it chronically. PROVISIONAL** (10-27, from zscore_check/where_preds_z.csv; per animal,
   shifts vs the animal's pre far-contra mean; b = the animal's pre pull slope). ACUTE: landing -1.83 / -0.98 / -3.00 /
   -2.52 mm (PS92-95); decoded -0.97 / -0.47 / -1.06 / -1.50 (raw ratio 0.53 / 0.48 / 0.35 / 0.60); an unchanged decoder
   would read b x landing = -1.18 / -0.50 / -1.74 / -1.56, so beyond the pull +0.21 / +0.04 / +0.68 / +0.05 (all
   contralesional; PS94 largest). So the code DOES carry the ipsilesional deviation (decoded moves ipsi, and follows
   lick-to-lick landing within the spout, slope 0.14-0.50), but sits contralesional of where an unchanged mapping would put
   the executed lick: the aim read-out is contralesional of the landing. CHRONIC: landing back near pre (-0.10 / -0.26 /
   -0.89 / -0.16) but decoded -0.41 / -0.41 / -0.81 / -0.25 -> beyond the pull -0.35 / -0.28 / -0.30 / -0.15, ipsilesional
   in 4/4 (raw ratios 4.2 / 1.6 / 0.9 / 1.6 are unstable because the chronic landing shifts are small; use the
   beyond-pull values). Earlier "fraction of execution-predicted shift" = decoded / (b x landing), NOT decoded / landing.
   Interpretation OPEN: a downstream (non-cortical) contralesional correction would make a cortical pattern that reads
   ipsilesional produce an on-target lick; alternatives = the frozen frame misreads a remapped / weaker far-contra pattern
   (4d far-contra pattern analysis), or cortex no longer sets lateral aim. Against a single fixed downstream offset: ITI
   licks are read CONTRALESIONAL chronically (+0.25), so the sign depends on context.
4d0c. **CONFOUND LEDGER for the larger lick-aligned response (one place; details in 4d).** RULED OUT: (1) smaller
   baseline F0 -- F0 rose (+11 % acute), response larger in dF (+96 %) than dF/F (+76 %) (`iti_f0_check.py`, page 3.46);
   (2) bout length / lick overlap -- isolated single licks give the same increase; (3) lick kinematics -- kinematics-
   adjusted and range-matched responses stay ~2x acute (`iti_licks.py --controls`); (4) face movement -- FaceRhythm-
   adjusted; (5) lick count -- matched to 40 licks / session; (6) for the decoder results only: response scale (within-
   session z-scoring). NOT A CONCERN: the drift filter -- the ITI analyses load the ADOPTED non-smearing variant
   `hemo_meegkit_hpfit` (`combined_encoding.load_imaging`), not the zero-phase original. CHECKS (10-27 evening,
   `scripts/iti_hemo_check.py`, commit ca96d32) -- DONE 10-27, 107 sessions, change vs pre in units of the animal's
   pre corrected response (key ROIs mean, acute / subacute / chronic; adopted reproduces the published numbers exactly):
   corrected +0.76* / +0.39* / +0.36*; ZERO-PHASE original SVTcorr +0.73* / +0.39* / +0.40* (filter: no effect);
   UNCORRECTED 470 +0.98* / +0.37* / +0.39*; 415 nm +0.17* [0.01, 0.33] acute only (mostly PS95, PS94), -0.02 / +0.02
   later (32 / 0 / 0 areas up vs 41 / 45 / 6 corrected). EDGE rim grows more than interior acutely (edge / interior ratio
   +0.12*, subacute +0.07*, chronic n.s.), interior alone still +0.66* / +0.49* / +0.39 (n.s.); VESSEL pixels and
   high-415-variance pixels: no change in ratio. Per-area F0 470 +12 % acute, fairly uniform (dF per area 39 / 41 / 4
   areas up). Running around licks unchanged except PS94 (runs during ~50 % of subacute / chronic licks vs 9 % pre).
   PRE-LICK GAP matched: unchanged (+0.76* / +0.41* / +0.41*). VERDICT: the later (subacute / chronic) increase is not a
   415-visible haemodynamic, motion, filter, F0, vessel or baseline-window artefact; up to ~1/5 of the ACUTE increase
   may be haemodynamic / motion (415 +0.17, rim-weighted). Caveats: 415 carries a small negative GCaMP term; PS94's
   chronic running; PS95's response was already rising across its pre sessions.
4d0d. **The FIRST cued lick is read toward the target; later licks are not. PROVISIONAL** (10-27,
   `scripts/iti_first_lick.py`; cued, lateral, frozen KEY-row decoder z-scored; each lick class vs its own pre mean at the
   spout). Offset beyond the pull, all spouts: first +0.39 / +0.27 / +0.19 mm (4/4 each epoch), later +0.17 / +0.03 /
   -0.06; first - later +0.22 / +0.24 / +0.25 (4/4, CIs exclude 0). Far contra: first +0.63 acute (4/4) / +0.27 / +0.11
   n.s.; later +0.15 n.s. / -0.16 / -0.33 (4/4). The first lick lands further ipsilesional than later licks (-1.08 vs
   -0.71 acute). So the target-ward read-out is a property of the aimed launch, and the chronic ipsilesional read-out of
   far-contra licks (4d0b) is carried by the later, corrective licks.
4d0j. **SAME-PLACE like-for-like read-out (PRIMARY, DECISIONS 10-27 late; page v130 figs 3.15 / 3.16, stable ids
   #f-lick_matched_place_readout / _curves). PROVISIONAL.** Each lick vs the animal's pre licks (any spout) that landed
   at the same lateral place. ITI, all spouts +0.31 / +0.21 / +0.24 mm (4/4 each epoch); far ipsi +0.43 / +0.38 / +0.38.
   CUED, all spouts +0.14 acute (4/4) -> ~0 / -0.04; cued FIRST lick +0.27 / +0.20 / +0.15 (4/4 every epoch) vs later
   +0.12 / +0.02 / -0.06; cued far contra first lick +0.58 / +0.29 / +0.11; cued far contra all licks +0.20 acute ->
   -0.10 / -0.28 (4/4, ipsilesional chronically). So under the primary measure: the cortical read-out sits contralesional
   of where licks land -- ITI licks in every epoch, cued licks acutely and on the first lick throughout; cued far contra
   ipsilesional chronically (compression at the edge). The 'first cued lick read toward the target' (4d0d) and the acute
   far-contra contralesional read-out STAND under this measure; 4d0h / 4d0i's pooled-deviation and same-spout numbers
   are sensitivity analyses (same ITI answer; same-spout cannot score the acute large misses).
4d0i. **LIKE-FOR-LIKE read-out = the primary offset measure (Priya 10-27); response-size attribution. PROVISIONAL**
   (`scripts/lick_matched_readout.py`, `scripts/lick_gain_prediction.py`). Each lick vs the animal's pre licks with the
   same landing deviation (no pull slope). ITI, all spouts: +0.23 / +0.16 / +0.22 mm (4/4 each); bout starts +0.26 /
   +0.20 / +0.20 (4/4); per spout strongest after far-ipsi trials (+0.41 / +0.37 / +0.39, 4/4) and far-mid (+0.21 every
   epoch). CUED, all spouts: +0.03 / -0.06 / -0.08 (chronic 4/4 ipsi); first lick +0.09 acute (3/4) then ~0; far
   contra read IPSIlesional every epoch (-0.28 / -0.34 / -0.39, 4/4 each; later licks -0.32 / -0.39 / -0.45) -- the
   earlier acute 'far-contra read contralesional' (+0.25 to +0.38) was the across-spout pull slope (CORRECTED);
   close ipsi +0.18 / +0.07 / +0.11. ATTRIBUTION: the response-size change along each animal's pre lick-response
   pattern, passed through the frozen decoder, predicts -0.08 / -0.07 / -0.05 mm for ITI (PS94 -0.32) and +0.03 for cued
   chronically -- the OPPOSITE sign to the observed shifts in both sets, so the uniform shifts are not the pan-cortical
   response change seen through the decoder (which stays a separate, biological result).
4d0h. **LANDING-MATCHED read-out (no pull slope at all) -- the cleanest test. PROVISIONAL** (10-27 late; per lick,
   landing deviation and decoded deviation from the animal's pre mean at the spout; each post-stroke lick compared with
   the animal's PRE licks that deviated by the same amount (bins -3..-1.5..-1..-0.5..-0.25..0.25..0.5..1..3 mm)). ITI:
   read +0.22 / +0.16 / +0.22 mm more contralesional than equally-deviated pre-stroke licks (4/4 every epoch; CIs exclude
   0; PS94 / PS95 largest, PS93 ~0), the whole landing->read-out curve shifted up. CUED: the SAME curve as before the
   stroke (+0.02 / -0.06 n.s.; chronic -0.08, 4/4, slightly ipsi). So cued licks being read 'near the target' while they
   land ipsilesional is how pre-stroke cortex already read equally deviated licks (it follows ~20 % of a small
   within-spout deviation, ~35-45 % of a large one); the abnormal read-out is the ITI one. Supersedes the 'beyond the
   pull' framing of 3.54 / 3.71 for cued licks (all spouts). Far-contra cued first vs later (4d0e) still to be re-run
   matched.
4d0g. **Per-spout DECOMPOSITION of the decoded lateral shift (no pull slope assumed). PROVISIONAL** (10-27 late,
   `scripts/iti_decoded_decomposition.py`). Per session, over spouts: dDecoded_s = SHIFT + GAIN x (spout's pre place) +
   TRACK x dLanding_s. ITI: a UNIFORM contralesional SHIFT +0.27 / +0.17 / +0.14 mm (4/4, 3/4, 3/4; with TRACK fixed at
   the pre within-spout slope +0.23 / +0.17 / +0.18, 4/4 each) PLUS compression (GAIN -0.13 / -0.18 / -0.09). Cued: NO
   contralesional shift (+0.05 / -0.05 / -0.09) -- the cued effect is COMPRESSION (GAIN -0.13 / -0.16 / -0.17; 4/4 from
   subacute), i.e. the lateral-r loss seen per spout. TRACK does not change. Caveat: post-stroke the far spouts both sit
   furthest out and deviate most (corr(place, landing change) -0.5 to -0.7), so GAIN and TRACK trade off; SHIFT is the
   stable term. So the ITI contralesional read-out survives without any assumed pull, and the cued 'read toward the
   target' in the beyond-pull figures is mostly compression + the pull slope.
4d0f. **Mixed models WITH TIME for the ITI / lick headlines** (10-27 late, `scripts/iti_mixed_models.py`, shared
   `wfield_local.mixed_models`, random intercept per animal -- the slope / step random effects did not converge). Time
   model (step at day 1, change per e-fold of days): ITI lateral r -0.21 step, NO recovery (+0.004 n.s.); ITI lateral
   offset +0.28 step, NO recovery (-0.013 n.s.); cued lateral r -0.21 step, recovers (+0.038 per e-fold); cued offset
   +0.28, recovers (-0.089); far-contra cued offset +0.38, recovers past 0 (-0.185); ITI response +0.88, recovers
   (-0.157). Behaviour model (far-contra time to contact): change still present at full behavioural recovery -- ITI
   offset +0.23, ITI r -0.22, ITI response +0.36, cued r -0.08; cued offset none (-0.02 n.s., tied to the deficit).
   Offsets carry the pull-slope caveat of 4d0e. PROVISIONAL.
4d0e. **Lick ORDER, ITI and cued: the ITI contralesional read-out runs through the whole bout; the cued one is a
   first-lick property. PROVISIONAL** (10-27 late, `scripts/iti_lick_order.py`, page 3.71 / 3.72; lateral, frozen
   KEY-row decoder per lick set, z-scored; ITI order = index within the lick bout, gap >= 1 s over all the session's
   licks; cued order = index after the cue). Offset beyond the pull, change vs pre: ITI bout starts +0.35 / +0.25 / +0.24
   and later ITI licks +0.29 / +0.21 / +0.25 mm (first - later n.s.); cued first +0.38 / +0.27 / +0.19 vs later +0.15 /
   +0.03 / -0.05 (first - later +0.22 / +0.24 / +0.24, 4/4) -- reproduces 4d0d. Far contra: ITI bout start carries it
   (+0.47 / +0.18 / +0.21; later ~0; first - later +0.42 acute 4/4). Continuous order (slope per lick, spout means
   removed): cued landing moves contralesional (toward the target) with each later lick in every epoch (+0.049 / +0.035
   / +0.019 mm per lick, 4/4) while the beyond-pull offset falls (-0.030 acute 4/4; far contra -0.082 / -0.049 / -0.049,
   4/4 each) -- the read-out moves less than the pull predicts as the landing corrects. ITI slope steeper only acutely
   (-0.036, 4/4); the ITI read-out already falls with bout order pre-stroke (-0.017 mm per lick). Mixed-model check
   (wfield_local.mixed_models epoch model) agrees with the bootstrap on 124 / 144 cells.
   CAVEAT (10-27 late): 'beyond the pull' uses the across-spout pre slope (~0.6); with the within-spout slope (0.2-0.3)
   the cued all-spout first-lick offset is n.s. and the decoded position is flat with lick order -- see the running list.
4d. **Self-initiated (ITI) licks, no trial / cue / target: PROVISIONAL** (10-26, ENCODING_FINDINGS 8.17,
   `scripts/iti_licks.py`). (i) Executed lick DIRECTION is decoded from cortex as well as before the stroke within each
   session (r ~0.53, unchanged every epoch), but the FROZEN pre-stroke decoder loses ~0.13 r in every epoch (4/4): the
   target-free execution code is intact but remapped. For CUED in-trial licks the pre-stroke decoder largely recovers (-20 % acute ->
   -9 % chronic) while for self-initiated licks it does not (-22 % chronic) (session-pooled, page 3.8 / 3.9). (ii) Before self-initiated licks, the slow pre-lick build-up in
   tjM1 / SSp-m is unchanged acutely and WEAKER chronically (3/4 animals; PS95 not), and the post-onset cortical
   response is ~2x larger acutely and still elevated chronically (4/4). Not bout length: post-stroke bouts are shorter and isolated licks show the same
   larger response. An earlier half-amplitude-timing readout
   ("later ramp") was an amplitude artefact and is not a finding. Controls (10-26): not lick kinematics (post-stroke
   licks are slightly slower; kinematics-adjusted and range-matched responses stay ~2x acute, +0.6-0.8 chronic) and
   not face movement (face-adjusted build-up still weaker). Caveats: ITI vs in-trial gap 3/4 animals (PS93 none);
   no-contact in-trial test pending. Not lick count: test sets matched to 40 licks per session give the same
   frozen-decoder result.
   (v) Not spout contact: in-trial licks WITHOUT contact recover too (-0.21 acute -> -0.03 chronic); ITI licks without
   contact stay down (-0.15 chronic). Session-paired ITI - in-trial gap widens chronically (-0.062 [-0.110, -0.014];
   no-contact -0.125), 3/4 animals (PS93 none).
   (vii) Basis-independent (10-26): the same in joint LocaNMF and a pre-stroke SVD basis (ITI chronic -0.12 r in all
   three bases, 4/4 each; in-trial recovers in all three).
   (viii) Which dimension: the ITI loss is LATERAL (-19..-20 % in all bases, 4/4); outward position is weakly decoded and
   unchanged. In-trial: outward drops acutely and recovers fully; angle decoding ends ABOVE pre chronically (+7..+11 %).
   Lateral falls more than angle chronically (paired, ITI -0.10 [-0.14, -0.06] areas, 4/4). Caveats (10-26 late): part
   of the lateral r loss is a narrower spread of lick positions (fully accounts for ITI PS92 / PS95 and in-trial PS93,
   not ITI PS93 / PS94); the angle here is horizontal atan2(lateral, outward), not the encoder's exec_angle. A
   spread-matched check is proposed.
   (ix) BEHAVIOUR, ITI lick rate: drops acutely in PS92-94 and stays 14-28 % below pre chronically; only PS95 (near-zero
   ITI licking pre) ends above pre (~4x). Tracks far-contra hit rate within animal (r +0.67; +0.51 with log-day removed,
   PS94 ~0) -- the shared dip-and-return; not the cause of the decoder's ITI loss.
   (x) Encoder, TE partial refit (10-26): target-labelled lick kernels gain over landing-labelled ones chronically
   (landing - target -0.019 contra SSp-m, -0.024 cortex, 0/4 up); acute target advantage falls (-0.126), back by chronic.
   (xi) Decoded vs actual (10-26): executed licks drift ipsilesionally; the frozen decoder does not follow; for cued
   licks, ipsi-deviated licks decode MORE like on-target licks after the stroke (share of the deviation followed
   0.26 -> 0.14 chronic, 4/4) -> target over execution. Superseded as a summary by (xiii).
   (xiii) **KEY (10-26 late, page 3.19-3.21, ENCODING_FINDINGS 8.17): post-stroke, cortex read with the pre-stroke
   decoder places SELF-INITIATED licks further CONTRALESIONAL, and CUED licks CLOSER to the mouth, than where they
   land.** Per lick, decoded position minus where the animal's unchanged pre-stroke pull toward the middle would place
   it (pull = decoded on actual, fit per animal and axis on pre licks; slope lateral 0.31-0.65, outward 0.13-0.36),
   change vs pre (`scripts/iti_where_beyond_pull.py`). ITI lateral +0.34 [+0.26, +0.43] acute (4/4), +0.20 subacute
   (3/4), +0.23 [+0.04, +0.44] chronic (sign 4/4; PS94 +0.54, PS95 +0.28, PS92 / PS93 ~0.03); ITI outward n.s. Cued
   outward -0.30 [-0.39, -0.22] acute, -0.30 subacute, -0.31 [-0.43, -0.18] chronic (0/4 further in each); cued lateral
   n.s. (ipsi-strayed cued licks read at their target). Same answer on licks landing at their usual (pre-stroke) spot
   (ITI +0.34 acute / +0.16 chronic; cued outward -0.30 chronic). CORRECTION (same evening): an earlier line here said
   the all-licks ITI offset follows "moving licks + the pull"; that multiplied the OVERALL ipsi shift, but ITI licks are
   grouped by nearest spout so the within-spout shift is much smaller -- done per lick the pull explains only part of
   it. Mean offsets do not enter r (3.15-3.18). Shared variance only: whether cortex carries the aim or does not
   register the changed movement is not determined.
   SAGITTAL CHECK (10-26 23:40, decoder re-fit on lateral + SAGITTAL extension sqrt(out_ap^2 + down_dv^2); licks run
   mostly DOWN, so outward = forward only is not protrusion): ITI lateral unchanged (+0.34 [+0.26, +0.43] / +0.20 /
   +0.22 [+0.04, +0.44]; z-scored 4/4 every epoch) -> the ITI-contralesional half STANDS. The CUED-CLOSER half is
   AXIS-DEPENDENT -> REVISED to "closer on the FORWARD axis": cued sagittal beyond the pull +0.03 acute (2/4), -0.20
   [-0.38, -0.04] subacute (0/4 further), -0.12 [-0.25, +0.04] chronic (1/4); z-scored +0.16 acute (4/4 FURTHER),
   -0.15 subacute and chronic (0/4 further, CIs exclude 0). So on total extension cued licks read closer only from
   subacute on. Chronic cued far-contra read ipsilesional also on the sagittal run (-0.24 [-0.35, -0.12], 0/4).
   Note: the pull shrinks every prediction toward the mean of ALL the animal's pre licks (one decoder; it never sees
   the spout), not toward the mouth or each spout's mean; pre-stroke landings cluster MEDIAL of the spouts.
   Frozen r change vs pre, normalised to pre (REBUILD 11-07, camera working period / new imaging basis, areas; replaces the
   10-26 "the r loss is LATERAL" line, Priya 11-07): BOTH AXES are lost on cued licks acutely -- lateral -20 / -10 % (4/4),
   -8 % chronic (n.s., 3/4); sagittal -25 / -21 / -8 % (4/4 each). The subacute / chronic sagittal loss sits in short licks
   that MISS the spout within successful trials: on spout-contact licks only, sagittal -24 % acute (4/4), -14 % subacute
   (n.s., 3/4), +7 % chronic (n.s., 2/4) (scratchpad a10_sagittal, 11-07). ITI: lateral -23 / -22 / -24 % (4/4); sagittal
   -13 % acute (n.s., 3/4), then -23 / -24 % (4/4) -- as large as lateral from subacute on. Encoder angle atan2(lateral,
   sagittal): ITI -0.11 / -0.07 / -0.10 (chronic CI excludes 0), in-trial
   -0.09 / -0.05 / -0.04 (chronic n.s., 2/4). The earlier "in-trial angle r ABOVE pre 4/4" used the HORIZONTAL angle
   (lateral vs forward) -> superseded: on the encoder's angle in-trial r returns to ~pre, not above.
   SPREAD-MATCHED r (10-27 ~01:00, `scripts/iti_where_spread_matched.py [--z]`; each session's licks resampled so their
   actual-position distribution matches the animal's pre distribution, 0.2 mm bins, 30 draws of half the licks; partial
   match where a post session never reaches part of the pre range: chronic SD / pre 0.79-0.99 after matching). LATERAL
   r loss SURVIVES: ITI -0.12 / -0.12 / -0.11 (raw -0.13 / -0.12 / -0.13), cued -0.12 / -0.08 / -0.05, 4/4 in every
   cell; z-scored the same (ITI chronic -0.13, cued -0.05). Sagittal: ITI chronic -0.05 [-0.09, -0.01] matched (4/4,
   small), cued ~0 or slightly above pre (z chronic +0.04). -> the lateral loss is not a narrower lick spread.
   SAGITTAL Z-SCORED (Priya 10-26 23:55: "may make most sense" for sagittal; unscaled and z-scored disagree acutely):
   cued sagittal beyond pull acute +0.16 (4/4 read FURTHER: .09 / .08 / .16 / .32) while acute far-contra protrusion is
   actually LOWER; chronic -0.15 (4/4 closer: -.08 / -.05 / -.16 / -.30); per spout z acute further at close contra /
   far ipsi / far mid 4/4. Both decoders are contrasts across areas (net / total weight ~0 on both axes), so the
   scale-sensitivity is per-area, not an overall-amplitude readout.
   RECOVERY LINK (10-26 late, `scripts/iti_recovery_link.py`; within animal over post sessions, log(day) removed from
   both; permutation p). Recovery = far-contra cued lick ANGLE / pre (encoder angle; hit rate saturates chronically at
   ~1.0 in all four). Chronic angle recovery PS92 1.01, PS93 0.77, PS94 0.13, PS95 0.89. As aim recovers, the ITI
   contralesional offset fades (-0.53, 4/4 negative, p 2e-4) and ITI lateral error falls (-0.48, 4/4, p 2e-4) -- ITI
   licks are different licks from the far-contra cued licks that define recovery, so not circular; cued lateral offset
   -0.46 (3/4, PS95 +0.18; same licks, partly circular). Frozen r tracks the time course (cued +0.60 raw) but not
   beyond it (+0.16 n.s.). Lateral-protrusion recovery gives the same. Between animals (n = 4, descriptive): PS94 = least
   recovered AND the most persistent chronic ITI signal (r 0.64 of pre, offset +0.54 mm, lateral error 1.81 x pre);
   the other three are not cleanly ranked (PS93 recovers less than PS95 but has the smaller offset).
   Z-SCORED decoder (10-27 ~01:10, `iti_recovery_link.py --beh angle --z`, figure iti_recovery_link_angle_z.png): ITI
   lateral offset vs far-contra angle recovery, time trend removed -0.48 (p 2e-4) with ALL FOUR negative (PS92 -0.35,
   PS93 -0.44, PS94 -0.62, PS95 -0.51; PS95 was + on the hit-rate measure); ITI lateral error -0.45 (4/4); cued
   sagittal offset -0.57 (4/4: cued licks read CLOSER as aim recovers, i.e. the acute "read further" fades into the
   chronic "read closer"); cued lateral -0.42 (3/4, PS95 +0.22). Frozen r: not beyond the time trend.
   FAR-CONTRA PATTERN TEST (10-27 ~01:30, `scripts/iti_farcontra_pattern.py`, farcontra_pattern.csv; cued licks,
   z-scored Allen areas lags 0..0.2 s, spout patterns centred on the mean over the six spouts per epoch so epoch-wide
   changes drop out). Chronic far-contra vs its pre pattern, gain / cosine (pre half-vs-half floor ~0.9-1.0 / 0.90-0.99):
   PS92 0.84 / 0.97, PS93 0.55 / 0.56, PS94 0.84 / 0.97, PS95 0.57 / 0.85 -> in 3/4 the SAME pattern at reduced gain
   (a weaker copy); PS93 degraded with a rise in similarity to the close-contra pattern (cos 0.07 -> 0.58), i.e. partial
   drift toward a nearer spout. BUT the mirror control loses gain too: far-ipsi chronic 0.68 / 0.72, 0.25 / 0.28,
   0.93 / 0.94, 0.56 / 0.44 -> the spatial contrast across spouts shrinks broadly (PS93 / PS95 most), not only at far
   contra; yet only far contra is READ ipsilesional (far-ipsi offset ~0), so the weaker-copy account alone does not
   explain the offset sign. Usual-spot subset too sparse to test (pre halves missing). OPEN.
   NOT ELAPSED-TIME DRIFT (10-27 ~02:00, `scripts/iti_where_drift.py`, figure iti_where_drift.png; decoders trained on
   ONE pre session, z-scored Allen areas, tested on every other session; drift curve = log fit of r on day gap over
   pre -> pre pairs). Within pre the lateral decoder barely drifts (slope -0.008..-0.035 r per log day; PS95 +0.02).
   ITI LATERAL r below the drift curve: -0.12 [-0.19, -0.03] acute (3/4), -0.15 subacute, -0.13 [-0.19, -0.07] chronic
   (4/4; chronic mean gap 39 days); cued lateral -0.16 / -0.09 (4/4), chronic -0.04 n.s.; sagittal ~0 for both.
   Caveat: the August pre sessions span only 8 days, so chronic gaps are extrapolated (the fitted slope is near flat,
   so drift would have to accelerate sharply to explain the loss).
   WHERE POSITION IS CARRIED: HAUFE PATTERN MAPS (10-27 ~02:30, `scripts/iti_decoder_patterns.py`, decoder_patterns.csv,
   figures/iti_decoder_patterns_{lat,sag,animals_lat}.png; activation pattern Cov(activity, decoded) / Var of a ridge
   decoder refit per epoch in each animal's pre-stroke SVD basis (200 comps), mapped to pixels; similarity to pre =
   pixel correlation corrected by split-half reliability). PRE lateral pattern: licks further CONTRALESIONAL go with more
   activity in the LEFT (lesioned-side) lateral orofacial sensorimotor cortex and less in the right one -- a
   crossed, push-pull map (PS92 / PS94 / PS95 cued clearly; PS93 and PS95-ITI mostly one-signed). Chronic similarity
   to pre, ITI lateral: PS92 0.83, PS93 0.90, PS95 0.76, PS94 0.19 (reliability 0.99 -> a genuinely DIFFERENT map in
   PS94, -0.01 subacute); cued lateral 0.87 / 0.68 / 0.83 / 0.98 (PS93 / PS94 dip acutely to 0.49 / 0.44 and recover).
   Sagittal patterns stay similar (0.70-0.97). So the lateral read-out map is largely preserved in 3 animals while
   lick-to-lick r falls; PS94 -- least recovered, largest persistent ITI offset -- remapped its self-initiated lateral
   map. Unscaled SVD features; PS95 acute has one session (no pattern). PROVISIONAL; per-spout one-vs-rest maps next.
   SAGITTAL map = BILATERAL, symmetric orofacial S1/M1 (longer licks -> more activity in both hemispheres), preserved
   (0.70-0.97). PER-SPOUT ONE-VS-REST maps (`scripts/iti_spout_patterns.py`, spout_patterns.csv, figures/
   iti_spout_patterns.png; cued licks): each spout's map is crossed (ipsi spouts -> right lateral orofacial cortex,
   contra spouts -> left). FAR CONTRA is the one spout whose map is REPLACED acutely (similarity to pre 0.31 / 0.28 /
   0.42, PS95 n/a) and comes back (chronic 0.97 / 0.61 / 0.96 / 0.92; PS93 partial); the other spouts stay 0.65-0.95
   (exception PS93 close mid 0.13-0.26 throughout). Fits the acute far-contra failure and the decoder results (4.x).
   F0 CHECK on the larger ITI post-lick response (10-27 ~03:15, `scripts/iti_f0_check.py`, f0_check.csv; F0 = mean of the
   pipeline's per-pixel temporal mean image `frames_average.npy`, central 60 % of the frame, raw counts): 470 nm F0 / pre
   +11 % [+3, +20] acute, +14 % subacute, +4 % n.s. chronic (415 nm similar, no trend) -> F0 did NOT fall, so the larger
   dF/F response is not a smaller-baseline artefact; in dF-equivalent units (dF/F x F0 / pre F0) the increase is slightly
   LARGER (+96 / +59 / +44 %, all CIs exclude 0; dF/F +76 / +39 / +36 %). Within animal F0 and response are not
   consistently related (r -0.60..+0.54). Caveat: whole-frame F0 includes LED power / focus; per-area F0 on the atlas grid
   and the 415 nm lick-aligned response are not done.
   PER-BIN SIGNIFICANCE on the ITI-lick PETHs (10-27, page 3.23 / 3.24 / 3.26, sig_bins_*.csv; paired nested bootstrap per
   bin, BH-FDR): larger post-lick response significant in every ROI acutely / subacutely, chronically in tjM1 and SSp-m;
   weaker chronic build-up (~-1.2..-0.2 s) significant in tjM1 and SSp-m bilaterally. 3.14 (basis decoder): ITI lateral r
   below pre at every lag, every basis, every epoch.
   THREE-AXIS VERSION (10-27, `scripts/iti_where_3axis.py`, where_3axis_{sessions,tests}.csv; z-scored, one ridge per
   axis: lateral, forward = out_ap, down = down_dv). Same picture as the outward and sagittal runs: ITI lateral beyond
   pull +0.33 / +0.22 / +0.25 (4/4 each), r -25 / -22 / -23 % (0/4 up); cued FORWARD read less forward every epoch
   (-0.22 / -0.21 / -0.17, 0/4 further) -> the old "outward" result; cued DOWN read further down acutely (+0.20 [+0.09,
   +0.31], 4/4) and less chronically (-0.14 [-0.27, -0.01]) -> the sagittal "ahead acutely, closer chronically" result;
   cued lateral +0.19 acute (4/4). r on forward / down not lost except cued forward acutely (-23 %).
   RMSE = BIAS + SCATTER (10-27, `scripts/iti_where_rmse_split.py`, where_rmse_split.csv; z-scored decoder; error =
   decoded - actual per lick, RMSE^2 = mean^2 + SD^2 per session): the post-stroke error increase is almost all BIAS
   (a shifted prediction), not scatter: ITI lateral |bias| +0.81 / +0.41 / +0.36 mm (CIs exclude 0), error SD -0.01 /
   +0.04 / -0.01 (n.s.); cued lateral |bias| +0.40 / +0.15 / +0.06, SD n.s.; cued sagittal SD even falls chronically
   (-0.15). PS94 carries the chronic ITI bias (+0.90). Raw bias includes the pull acting on changed landings (the
   beyond-pull offsets of the KEY row remove that).
   ROBUST TO WITHIN-SESSION Z-SCORING (10-26 late, `scripts/iti_zscore_check.py`; each area / its frame-wise SD in the
   session, pre and post alike; the traces already have session mean and slow drift removed): ITI lateral beyond the
   pull +0.33 / +0.22 / +0.25 [+0.08, +0.45], 4/4 in EVERY epoch (chronic PS92 .12, PS93 .05, PS94 .55, PS95 .26); cued
   outward -0.22 / -0.21 / -0.17 [-0.25, -0.06], 0/4 further every epoch (~30-45 % smaller than unscaled; PS95 chronic
   ~0); cued lateral +0.19 acute 4/4. Frozen r losses unchanged or slightly larger (ITI lateral chronic -23 %). So
   neither the offsets nor the r loss are a response-scale (F0 / variance) artifact.
   PER SPOUT (10-26 late, page 3.22, `scripts/iti_where_per_spout.py`; ITI grouped by the PREVIOUS trial's spout, fixed
   before the lick): ITI read contralesional after every spout acutely (+0.24..+0.43, 4/4 at 5 of 6) and chronically
   after far-ipsi / far-mid trials (+0.33 [+0.10, +0.51] / +0.19 [+0.08, +0.36], 4/4 each); cued read closer at every
   spout and epoch (-0.16..-0.40, 0/4 further); chronic cued FAR-CONTRA licks read IPSIlesional (-0.24 [-0.35, -0.12],
   0/4 contra; +0.32 acutely); cued far-ipsi frozen r down every epoch (-0.32 / -0.16 / -0.20, 0/4 up). Same with
   z-scored features (chronic ITI far ipsi +0.39, far mid +0.23; cued far contra -0.27).
   Chronic cued far-contra ipsi reading is NOT the pull correction: far-contra licks at their usual spot (no pull model)
   -0.22 / -0.46 / -0.17 (PS92 / PS93 / PS95; PS94 has none there), and cued far-ipsi licks show no mirror shift (not a
   symmetric gain loss). DECODING ERROR in mm (MAE, page 3.15 second panel, `scripts/iti_where_mae.py`; includes the
   pull and offsets): ITI lateral +0.39 acute / +0.15 subacute (4/4), chronic +0.12 n.s.; outward error stays up
   chronically for both lick types (ITI +0.09, cued +0.18 mm, 4/4).
   (xii) CORRECTION (10-26): in-trial frozen r is also lower (-0.146 acute -> -0.063 chronic, CI excludes 0), smaller than
   ITI; its time course follows lick accuracy (a target-only reader loses -0.246 -> -0.036; session corr 0.82), so the
   in-trial loss is mostly 'decoder reads the target, licks deviate'; the ITI loss is the remapping.
   (vi) Recovery marker: within animal, the in-trial frozen decoder tracks far-contra behaviour across post sessions
   (r +0.62, 4/4; +0.53 controlling lick-direction spread) and the ITI one does not (+0.10); beyond the shared time
   trend the link is small (+0.16, 4/4 positive, n.s.) -> a marker of the recovery time course, not yet of day-to-day
   behaviour.
   (iii) Binned frozen decoder (10-26): same result (ITI -22 % chronic, in-trial -19 -> -10 %); with visual + RSP areas
   only, both lick types lose ~half and in-trial does not recover -> the in-trial recovery is orofacial-specific.
   (iv) Whole cortex: the acute / subacute response increase is global (41-45 Allen areas); chronically the larger
   response (MOp, SSp-m, tjM1) and the weaker build-up (SSp-n, SSp-un, ALM) reach significance only in orofacial cortex,
   but control areas move the same way (corrected 10-26 14:10): strongest in, not confined to, orofacial cortex.
4g. **Target vs execution kernels (new encoder design, 10-26, ENCODING_FINDINGS 8.20, page 1.34 / 1.35): PROVISIONAL.**
   In the frozen pre-stroke encoder neither the target nor the execution group's unique contribution changes much
   after the stroke; in SSp-m execution falls slightly relative to target (-0.03). A session-specific refit gains a
   lot (+0.06..+0.12 R2, 4/4), but almost none of it is unique to target or execution: the remapped information is
   shared between them. Partial refit with label-shuffled nulls pending.
4e. **Chronic cortex - face-lick coupling tightens in orofacial motor cortex** (10-26, ENCODING_FINDINGS 8.17b,
   page 1.16 / 1.17): the FaceRhythm lick factor's trial-by-trial unique contribution rises chronically while its
   trial-to-trial variance FALLS (-40 % / -44 %), so it is not more face variability. PROVISIONAL.
5. **Working interpretation (hypothesis, not a result):** losing striatal input removes spatially specific (directional)
   drive to orofacial cortex; cortex compensates with more gain in lateral frontal / orofacial motor areas, mostly in the
   contralesional hemisphere (intact-striatum side), and encodes movement more generically -- matching the behavioural
   substitution.

**Caveats that bound this:** n = 4; drift nulls for residual / Layer 2 only within August (June lacks FaceRhythm and good
kinematics); PS93 anterior motor-cortex injury (section 3 caveat); total variance rises GLOBALLY (VISp x1.83 > cortex x1.54),
and analysis 1 shows the chronic Layer 2 change is global and the residual partly global; refit vs frozen claims are scored in
different windows (4.11 caveat); lick-direction results carry the standing 3-D re-run rule.

**Analyses that would sharpen it (proposed 10-25; Priya: run 1, queue 2 and 4):**
1. *Specificity control* (DONE frozen + FR, `scripts/specificity_control.py`; no-FR repeat running): Layer 2 NOT
   orofacial-specific; residual partly global (lick), orofacial-weighted (cue). ENCODING_FINDINGS 8.11.
2. *Does remapping track recovery?* (DONE, `scripts/remap_lmm.py`): residual = disruption tracking the deficit + a
   persistent part; Layer 2 = steady cortex-wide rise with time, larger when behaviour is better (not compensation-specific).
3. *Target-free execution test* from inter-trial 3-D licks (no target): per-lick encoding pre vs post.
4. *Behaviour-matched residual* (DONE, `scripts/matched_residual.py`): residual unchanged on inside-envelope trials ->
   not extrapolation. Design-space distance diagnostic still queued with the self-fit second pass.: chronic trials inside the pre range of 3-D
   trajectory / latency / ILI; distance of post designs from the pre support.
5. *Preparatory timing:* pre-lick ramp in ALM / tongue-jaw motor relative to the first lick, per spout x epoch.
6. *June -> August drift ceiling* with a reduced model June supports (no FR, 2-D pose).
7. *Per-animal links:* contra vs ipsi change vs lesion burden; leave-PS93-out for every chronic map claim.
Also running / queued: per-session vs frozen self-fit in the same windows (`scripts/session_vs_frozen.py`; remapping +
the scoring caveat), then a second pass in the narrow every-lick window (+/-0.2 s, the refit window) as a secondary
per-lick measure (proposed 10-25: the first-lick window stays primary; Priya to confirm).

### 0b. Lick-evoked activity and the hemispheric balance (10-28; goal page 12.17-12.18, 14.7-14.22) -- PROVISIONAL

- **Lick-evoked activity is LARGER after the stroke, in both hemispheres.** Cued licks: larger at every target in every
  epoch (response AUC 0-1 s), EXCEPT far-contra targets acutely -- the response reaches about the pre peak but decays
  sooner, so over 0-1 s it is, if anything, lower (0.5-1.0 s below pre in 3-4/4 animals; survives lick-vigour and landing
  regression). ITI licks: larger acutely across most of cortex; chronically mainly orofacial motor cortex (contralesional
  tjM1). PROVISIONAL.
- **The hemispheric BALANCE tracks the decoder's read-out OFFSET -- not the tongue's ipsilesional deviation.**
  Behaviour: acutely the tongue lands ipsilesional of the target (aim deficit). Read-out offset: the frozen decoder reads
  each lick as more contralesional than where it landed. Balance: at a fixed lick angle, post-stroke licks carry
  relatively more ipsilesional-hemisphere activity; the pre-stroke mapping reads that as a more contralesional lick, and
  sessions with a bigger balance shift have a bigger offset (within-animal r -0.76 to -0.95 for cued licks, 4/4; 3/4 for
  ITI; partly mechanical since the decoder reads the same features). The behavioural link is separate and weaker: pre-stroke
  the balance tracks lick angle (each hemisphere larger for licks toward the opposite side), post-stroke ~30% flatter
  (PS94/PS95 every epoch, 3/4 chronically). By target, the contra/ipsi ratio rises mostly for FAR spouts (far ring 4/4,
  close ring ~0). OPEN (running 10-28): balance-blind decoder, per-hemisphere imaging control, non-orofacial balance.
- **IMPORTANT (Priya 10-28): cued-lick ORofacial balance estimates how far an UNSEEN animal has recovered, ~2x better
  than days since the stroke; only orofacial balance does this.** Held-out-animal cvR2 of the far-contra first-lick aim:
  orofacial balance (same angle & order) 0.50 vs days 0.23 vs non-orofacial balance -0.19; within-animal shuffle null
  <= -0.12 (p < 0.0005). Lower error than the calendar in 3/4 held-out animals (PS95 the exception). The calendar gets each
  trajectory's SHAPE; the balance adds HOW FAR the animal recovers (mostly flags PS94, least recovered). Survives
  nonlinear time models (per-animal splines +0.68 deg/unit, p 0.006); does NOT predict next-session improvement; ITI
  balance does not do it. Bounds: WITHIN an animal days alone already explains 0.70-0.81 and balance adds little (PS94
  0.70 -> 0.80, PS92 0.80 -> 0.83); leave-one-SESSION-out (76 sessions, pooled) balance 0.63 vs days 0.48 with the animal
  unknown, but days wins (0.83; days + balance 0.86) once the animal is known -> the value is placing an animal you have
  no history for. 4 animals -> a strong hint, not a demonstrated biomarker. PROVISIONAL. (goal page 14.16)
- Controls 10-28: balance-blind decoders leave the offset unchanged -- global balance removed, ORofacial pair
  differences zeroed (ITI +0.38/+0.22/+0.27, pre r 0.59), or non-orofacial zeroed (12.20): the offset is carried by the
  pattern across orofacial areas, not by a left/right difference, so the balance TRACKS the offset without being its
  sole carrier; orofacial-only decoder reproduces most of the ITI offset; VIS+RSP-only barely decodes (r 0.15); per-hemisphere imaging control: F0 / 415 nm do not explain the
  subacute / chronic balance shift (14.18; part of the acute shift in PS92 / PS93 could be F0); head-to-head: orofacial
  balance predicts the offset, non-orofacial adds nothing (12.19).
- **PRIMARY recovery marker (10-28 late, page 14.17 -- method box 14.16; DECISIONS 10-28 late night):** balance of ALM + tjM1 + SSp-m + SSp-n
  + SSs as an angle-matched GAIN RATIO 0.1-0.5 s after the FIRST lick (cued: first post-cue lick; ITI: first lick after
  >= 2 s quiet). Unseen-animal cvR2: cued 0.70 vs days 0.23 (r 0.71/0.82/0.94/0.75; within-animal p 0.001, splines p
  0.002); ITI vs ITI's own angle axis 0.42 vs 0.17 (r 0.58/0.80/0.91/0.10 -- PS95 no); window confirmed on held-out
  animals (0.69), pool from 63 subsets with selection cross-validated (SSs drives ITI). No next-session prediction; within
  a known animal days already 0.75 (balance + days 0.76). ROBUSTNESS: ~1/3 of acute first licks lie beyond the pre
  angle range (line extrapolated); with only in-range licks cued holds (0.53 vs days 0.23) but ITI cross-animal does not
  (0.07; within-animal link remains 3/4). Gaussian same-place reference worse (cued 0.37); lateral reference equivalent;
  lick count cancels (min-20 / weighted unchanged). Tracks the decoder offset (cued r -0.58..-0.92, 4/4). PROVISIONAL.
- **Controls 10-29:** kinematics-matched reference (angle + reach + speed + elevation): cued 0.63 survives, ITI
  cross-animal fails (-0.26; within-animal intact). MOs-rest / MOp-rest add nothing (0.70 -> 0.70). The acute excess of
  cued first licks was the 1500-lick random cap in iti_licks.extract (shorter acute bouts), NOT behaviour (engaged trials /
  tracked first licks flat or lower acutely; session_pace.csv) -- the earlier "first-lick count tracks recovery" is
  WITHDRAWN. Balance explains ~65% of the cued acute decoder offset but only 5-15% of the persistent ITI offset; for ITI
  the decoder offset (0.39) is the better recovery marker. PROVISIONAL.
- **Decoder-offset decomposition (10-29, page 12.17):** the linear decoder's like-for-like offset splits EXACTLY into
  per-area left/right DIFFERENCE and SHARED parts. ITI offset is ~70-75% hemispheric difference (acute +0.29 of +0.39,
  chronic +0.20 of +0.29), carried by orofacial + SSs and concentrated in the SSp-n difference (+0.34 / +0.20); other
  cortex nets ~0 (RSP/VIS shared terms cancel). The SSp-n difference term tracks recovery WITHIN every animal for ITI and
  cued licks (r -0.45..-0.94, 4/4, mixed p < 0.001), the shared parts do not; it does not transfer across animals
  (per-animal decoder scales; rescaling does not help) -- a pre-stroke-referenced, within-animal marker (Priya: fine
  for now). Cued offset ~0 chronically, so its SSp-n signal is the acute -> chronic course. Trial level: dominated by
  decoder shrinkage (present pre-stroke) -- uninformative about cause. PROVISIONAL.
- **Primary balance by lick order (page 14.14):** cued acute ~-0.35 at every order 0-4, ~0 chronic (the raw balance's
  chronic overshoot is gone); ITI only the acute bout start (-0.28). Uncapped first licks (~476 / session vs ~138):
  within-animal r 0.77/0.87/0.94/0.74, unseen animal 0.65. PROVISIONAL.
- **HYPOTHESIS (updated 10-29, with Priya):** after the ipsilesional VLS lesion, the orofacial cortical code for lick
  direction SHIFTS toward the contralesional reading (frozen-decoder offset), persistently for self-initiated licks. The
  recovery-relevant part of the shift is a HEMISPHERIC difference concentrated in SSp-n (nose / upper-lip S1) as the
  pre-stroke code weights it, plus a broader orofacial + SSs gain imbalance (ipsilesional cortex over-responds acutely
  relative to contralesional at the same lick direction) that places unseen animals for cued licks. Both are
  POST-MOVEMENT STATE measures (no pre-lick predictive signal, no next-session forecast; the calendar tracks a known
  animal). Recovery = re-balancing toward the pre-stroke hemispheric relation, complete for cued (goal-directed) licks,
  incomplete for spontaneous licks. Response SIZE follows a time course that differs between lesion groups (PS92/93 re-lesioned, larger striatal loss: grows; PS94/95 single lesion, smaller loss: largest acutely; PS95 least severe), not recovery. Open alternatives:
  the balance as altered SENSORY reafference (SSp-n / SSs are sensory; cued licks have spout contact, ITI none) vs a motor
  command; state / arousal (contralesional VIS tracks ITI recovery). Next tests: cued hit vs miss (contact) balance;
  SSp-n vs SSp-m sensory mapping; a perturbation.
- **Is SSp-n special? (10-29, 12.17 supplement)** (1) decoder refit WITHOUT SSp-n: offset still tracks recovery within
  animals but weaker (ITI mean r -0.48 -> -0.38; -> cued aim -0.28 -> -0.15; cued -0.72 -> -0.69) -- largest single
  contributor, not necessary; (2) WEIGHT-FREE same-place left/right difference per area: SSp-n #1-2 of 21 for every target
  (cued r 0.94/0.84/0.84/0.69) but SSp-ul, SSp-un, MO-rest, SSp-m, tjM1, ACAd close behind -> a BROAD sensorimotor
  left/right imbalance, strongest in SSp-n; (3) FaceRhythm face movement partialled: unchanged -> not facial reafference.
  Hypothesis wording updated: 'broad sensorimotor hemispheric difference, strongest in SSp-n'. PROVISIONAL.
- **ACAd midline control (10-29, 12.17 supplement):** ACAd's weight-free left/right link (ITI -> ITI angle r 0.59) is
  UNDETERMINED, probably not ACAd-specific cortex: it weakens away from the midline (>=0.1 mm 0.48, >=0.2 mm 0.38 ns)
  and the <0.1 mm strip over the sagittal sinus alone carries it (0.57); registration ruled out (sinus within 7-15 um
  across sessions, <=3% of d variance; offset-shifted masks unchanged); 415 nm d does not track recovery but F0
  asymmetry does (partialling either leaves ~0.47); shares most variance with MOs-rest / SSp-n (all confounds together
  0.18, p 0.065); pixel maps = broad medial-frontal pattern peaking on the sinus in PS92 / PS95. Not a grant candidate
  region on this evidence. scripts/lick_acad_midline_control.py. PROVISIONAL.
- **Raw vs angle-referenced balance reconciled (10-29, scratch raw_spout_balance.py, cued first licks, pool ALM/tjM1/
  SSp-m/SSp-n/SSs, log2 contra/ipsi of post responses):** PRE the raw balance is high for ipsilesional-spout licks (+0.4
  to +0.6) and low for contralesional ones (-0.5 to -0.8) -> each hemisphere responds more to licks toward the OPPOSITE
  side (same layout as the SSp-n decoder weights). ACUTE far-contra raw balance rises toward 0 (PS92 -0.04, PS93 -0.40,
  PS94 +0.13, PS95 +0.18; Priya's 'looks like far-mid') because those licks LAND ipsilesionally (-11 / 0 / -28 / -13 deg);
  relative to pre licks of the SAME landed angle it is LOWER (referenced -0.18 / -0.47 / -0.94 / -0.14) -> the
  activity looks like a lick further contralesional than the one made = the decoder's contralesional offset. Same
  imbalance, two references. Sign conventions differ across analyses (balance + = contralesional hemisphere larger;
  angle + = contralesional side; aim - = ipsilesional; offset + = read contralesional) -- check each before comparing.
- **TARGET vs EXECUTED (10-29, 14.18 supplement, IMPORTANT CAVEAT to the cued marker):** pre, the post-lick response
  (contra - ipsi, primary pool) is predicted better by the TARGET spout than by the landed angle (4/4; target weight w
  0.80/0.92/0.92/0.76; within-spout angle slope 12-35% of across-spout). A target-aware reference removes the cued
  acute balance deficit (-0.34 -> +0.01) and the decoder's cued offset (+0.30 -> -0.05 mm), and the target-aware cued
  balance no longer predicts recovery across animals (LOAO 0.70 -> -0.40) -> the CUED marker is largely the landing
  error seen through a target component. ITI: no consistent target component; balance deficit survives (-0.32 acute,
  4/4), decoder keeps a contralesional offset beyond landing error (+0.36/+0.24/+0.25 mm), still tracks recovery
  (LOAO 0.32-0.37) -> ITI = genuine state shift. Caveat: pre w likely inflated (angle noise), post w lower -> a small
  cued state shift may remain, not separable. scripts/lick_balance_target_model.py. PROVISIONAL.
  KEY POINT (Priya 10-29, record): at the SAME landed angle the target matters LESS after the stroke -- the target
  effect falls from 1.0-1.45x the angle effect pre to 0.25-1.06x acutely (baltarget_matched.csv), and acute far-contra
  licks look somewhat MORE like where they landed than the pre model predicts (mix-reference acute +0.41). If anything
  cortex tracks the ACTUAL lick more after the stroke, not less -- the opposite of a stronger target / plan signal.
  Framing (Priya): post-stroke cued off-target licks look like pre-stroke off-target licks to the same target landing
  at the same place; such errors are just far more frequent (and larger) after the stroke.
  TESTED (scripts/lick_balance_matched_error.py, no extrapolation: post licks inside the pre range for their spout vs
  pre licks to the same spout at the same angle): CUED acute within-spout balance +0.02 [-0.07, 0.09] and decoder
  offset -0.04 mm (ns) -> framing HOLDS acutely (target-blind reference on the same licks -0.15, 4/4); subacute /
  chronic within-spout balance +0.13 / +0.16 (4/4) with decoder ~0. ITI: within-spout balance acute -0.18 (4/4),
  decoder +0.29 / +0.19 / +0.22 mm (4/4, CIs exclude 0) -> genuine ITI state shift.
  CUED WITHIN-SPOUT BALANCE RISES ABOVE PRE (per animal pre / acute / subacute / chronic: PS92 0 / +0.06 / +0.17 /
  +0.06, PS93 0 / -0.10 / +0.07 / +0.18, PS94 0 / +0.08 / +0.20 / +0.27, PS95 -0.01 / +0.03 / +0.06 / +0.13): an
  overshoot, not a return; tracks cued aim within animals r -0.15 / 0.85 / 0.81 / 0.46 but not beyond log(day)
  (p 0.76). ITI within-spout balance tracks the ITI angle (r 0.53 / 0.74 / 0.61 / 0.23, p 0.005).
  NOTE (Priya 10-29, keep for the grant): the contralesional hemisphere gains relative weight over weeks, independent
  of where the lick lands -- fits gradual reorganisation, the intact hemisphere taking on a larger share, more than a
  direct correlate of aim. Silencing contralesional cortex in the CHRONIC phase would test whether that extra share
  supports the recovered behaviour.
  DECIDED (Priya 10-29 morning): (1) WORDING -- the cued imbalance is REDUNDANT with the landing error and appears
  after the lick; whether it drives the deficit or recovery is UNTESTED (not "a consequence of the errors"). (2)
  HYPOTHESIS updated (page Biology in brief) without minimising the persistent signals: the ITI contralesional offset
  and the slow rise of the contralesional share are co-headline. (3) MARKERS: cued balance = sensitive state readout
  (headline for recovery staging); ITI decoder offset + ITI SSp-n left/right difference + the contralesional-share rise =
  the mechanistic markers. (5) movement-null plan encoding model DEFERRED (built 10-30 as maps from existing outputs: section 4, 12c).
  C/I BY SPOUT AND EPOCH (10-29, Priya: "C/I increased after stroke then back to pre? but for the most impaired licks
  it increases over recovery?"; scripts/lick_ci_spout_epoch.py, pre LOSO, page 14.18 supplement). The answer
  depends on the reference, HEMISPHERE sign (+ = C/I up):
  RAW C/I vs pre at the SAME SPOUT (relative to the target): UP after the stroke, largest at far contra, partly falls
  but stays above pre -- cued far contra +0.58 / +0.34 / +0.35 (4/4 each epoch), far ipsi rises late +0.01 / +0.25 /
  +0.28; ITI after far-contra trials +0.42 / +0.28 / +0.27 (4/4, 3/4, 4/4).
  RELATIVE TO WHERE THE LICK LANDED (pre angle line): DOWN acutely at every spout and returns to pre -- cued far
  contra -0.27 [-0.61, -0.02] / -0.03 / +0.11, ITI after far contra -0.60 [-1.34, -0.12] / -0.13 / -0.05 (PS94 carries
  the size of the acute; PS95 one acute session); all spouts cued -0.36 (4/4) / -0.12 / -0.03; most negative and
  persistent at close / far ipsi (cued close ipsi chronic -0.40, 4/4; ITI far ipsi chronic -0.43, 4/4). Target frame,
  other five spouts also drift up later (cued +0.04 / +0.13 / +0.14).
  SAME TARGET AND SAME LANDING (matched error, above): ~0 acutely, then rises ABOVE pre (+0.13 / +0.16, 4/4).
  So "up, then back" is the landing-referenced view; in the target view C/I stays raised; at matched target + landing it
  climbs over recovery. ITI licks after far-contra trials ARE the most ipsilesionally deviated (acute -29.7 deg vs
  -8 to -17 deg after other spouts, 4/4; -13.4 / -9.8 later) and follow the same pattern as cued far-contra licks.
- **HOW DOES THE LICK-DIRECTION CODE CHANGE? (10-29, Priya item 2; scripts/lick_code_subspace.py; page 14.18 supp):**
  REPRESENTATIONAL CHANGE = contralesional offset + lower gain / precision; RE-MAP OPEN (Priya 10-29: 'no re-map' too strong).
  WORDING APPROVED (Priya 10-30): "No lick type shows a code rotation beyond ~10-17 deg (power-limited). Uncued (ITI,
  ENLv) licks show a lasting contralesional offset and a loss of gain; cued first licks show a small offset and no gain
  change." June-drift comparison NOT used (room move + other rig between June and August).
  Matched licks, frozen basis, new frozen kind lick_code; drift floor = June 6-8 no-stroke sessions (~60 d before Aug).
  Cued: pre->post intercept +0.16 [0.08, 0.23] mm acute 4/4, +0.08 sub 4/4, +0.04 chronic ns; r / slope unchanged;
  nothing beyond June. ITI (all and far1): r -0.20/-0.15/-0.16, slope -0.11/-0.12/-0.12, intercept +0.28/+0.18/+0.22 mm
  (4/4); beyond June intercept +0.20/+0.09/+0.14 and slope -0.09 (4/4). Post-trained decoders read pre licks as well as
  their own epoch (post->pre ~ post->post) -> argues against a wholly NEW code only: post licks are less decodable even
  within epoch, so a noisier shifted code (or a partial rotation) gives the same; the pre ITI axis drifts as much June ->
  Aug without a stroke, so axis overlap has little power.
  NOISE-MATCHED NULL + POWER (scripts/lick_code_noisematch.py): cued transfer sits on a noise-matched unrotated null;
  rotations >= ~10 deg pooled (15-25 per animal) excluded; nothing beyond June. ITI (all, far1): both transfer
  directions BELOW the null every post epoch, 4/4 (~17 deg acute, ~12 sub / chronic) but June also below (~8-11 deg);
  beyond June only acute post->pre (-0.054, 4/4; ~6-9 deg) and subspace overlap; frozen slope drops beyond June every
  epoch (gain / projection loss). Asymmetry = the null's. Axis cosine needs 20-40 deg (unusable for ITI). Proposed
  reading: cued unchanged within ~10 deg; ITI not re-mapped beyond ~10-15 deg, small acute rotation-like change beyond
  2-month drift + lasting gain loss. (Pending Priya's discussion.)
  ENLv + ALL-TYPES SUMMARY 10-29 (lick_code_rotation_summary.png, Fig 14.18 S7): vs the noise-matched null (NOT June:
  Priya: June->Aug not a clean no-stroke reference, room move + other rig): cued first at the null (0-11 deg); ITI all below
  4/4 (~12-17 deg); ENLv forward at the null, backward -0.05 (~10 deg, at the detection limit); ENLv axis cosine / subspace
  below the null 4/4 (implied ~30-37 deg, detectable only >17-43 deg). ENLv offset +0.50/+0.33/+0.23 mm (4/4), gain -0.09
  chronic. Below the null = more than added noise, not necessarily rotation.
  Where: ITI excess over ceiling in orofacial cortex bilateral acutely, IPSILESIONAL chronically (SSp-n, SSs, tjM1,
  SSp-m); non-orofacial change = June-level drift. Caveats: PS95 one acute session; PS94 cued acute 16 % retained; area
  terms point estimates. PROVISIONAL.
- **PER-AREA TUNING + MATCHED PIXEL MAPS (10-29, Priya items 1 + 5; lick_area_tuning.py, lick_matched_pixel_maps.py):**
  cued contralesional-share rise (pool +0.13 / +0.16, 4/4) = both hemispheres above pre with the ipsilesional excess
  fading faster, plus chronically FLATTER contralesional tuning (depth -0.56, 4/4) -> offset + broader tuning, NOT gain.
  Strongest SSp-n (+0.30 chronic) / SSs / ALM / tjM1, MOp ~0, BUT VISpm +0.37 and RSPd +0.20 rise too (4/4) -> broad
  hemispheric re-weighting, NOT orofacial-specific (qualifies the hypothesis line (3)). ITI share deficit acute only
  (-0.18, 4/4; SSs, SSp-n); lasting piece MOp outside tjM1 (-0.13 chronic; far1 -0.14, 4/4) -> the lasting ITI DECODER
  offset is not a hemispheric level difference (it is a pattern offset, cf. code subspace). Pixel maps localise the
  same (cued chronic: anterolateral frontal / SSp-n + posterior-medial; ITI acute: lateral S1 / SSs). PROVISIONAL.
- **CORRECTION LICKS (10-29, Priya item 4; scripts/lick_correction.py):** behaviour -- the miss-minus-hit correction at
  matched error (step_b, controlling the previous lick's error) is NEGATIVE pre (-4.1 deg: misses repeat) and rises
  post-stroke +4.9 / +2.3 / +3.1 deg (4/4), later licks; yet after a far-contra miss the next lick hits ~10 % acutely
  (51 % pre). Neural -- no evidence of an error signal that predicts the correction: activity adds 0.02-0.04 cvR2
  (control 0.01), already before lick k and after hits (next-command information); no post-stroke increase; the acute
  contra-ipsi miss-hit rise (+9.5, 4/4) is an ipsilesional DROP, decays with log day, not robust to deconvolution.
  GCaMP overlap of lick k+1's command limits all of this. PROVISIONAL.
- **INTERHEMISPHERIC COORDINATION (10-29, Priya item 3; scripts/lick_interhemispheric.py):** noise correlation between
  hemispheres rises post-stroke (ITI pool +0.17 sub / chronic, 4/4) but equally in VISpm / RSPd and at docked rest
  (+0.21 chronic) -> GLOBAL, not lick- or orofacial-specific (lick minus rest ~0 or negative). No measurable lead (peak
  lag ~0, changes < 1 frame); cross-correlation shape tilts contralesional-earlier (+0.03-0.04, 4/4; lick-specific only
  acutely); Granger direction moves contralesional but as much at rest (global; gain / SNR-sensitive). More tilt goes
  with WORSE aim (cued r -0.44, 4/4; also at rest) -- hint only (many tests). GCaMP6s: shape, not lead. PROVISIONAL.
  Suggested next: partial out the hemispheric balance before Granger / asymmetry; global-signal-removed noise corr.
  FOLLOW-UPS (10-30, scripts/interhemi_crossarea.py -> iti_licks/interhemi_followup/; raw variant reproduces
  lick_interhemispheric exactly): GLOBAL REMOVED (partial corr given the mean of all OTHER areas, leave-pair-out) --
  the orofacial interhemispheric rise SHRINKS to ~0 during licks (ITI all w2 sub / chronic +0.17 -> +0.07 / +0.05, CIs
  touch 0; cued -0.07 / -0.03) and halves at rest (chronic +0.21 -> +0.11, 4/4), while the CONTROL pairs keep rising
  (RSPd / VISpm at rest chronic +0.25 / +0.20, 4/4; cued w2 +0.16 / +0.17, 4/4) -> part of the rise is the global mode,
  and what survives is NOT orofacial-specific (controls >= orofacial). xcorr TILT survives partialling the per-event
  hemispheric balance AND the global signal on lick sets (cued w2 acute +0.039 -> +0.032 bal / +0.034 gsr / +0.027 both,
  all 4/4; ITI dock chronic +0.040 -> +0.039 / +0.051 / +0.047, 4/4); still absent at rest. Session-level adjustment for
  the log SD ratio contra/ipsi or the cued contralesional share (slope within animal x epoch) leaves it unchanged.
  GRANGER shift: balance partialling changes nothing; global removal REDUCES the rest shift (chronic +0.044 -> +0.011,
  CI incl. 0; acute +0.055 -> +0.030, 4/4) but INCREASES the acute ITI-docked shift (+0.021 -> +0.062, 4/4) -> after
  GSR the Granger change looks more lick-set than rest, but values are tiny and variant-sensitive. Tilt <-> worse aim
  survives both controls (cued w2 within-animal r -0.44 raw, -0.38 bal, -0.35 gsr, 4/4; p <= 0.002; 112 recovery tests run).
  GCaMP6s caveat unchanged (shape, not lead). PROVISIONAL.
- **CROSS-AREA CORRELATION STRUCTURE (10-30, Priya "synchronization / correlation across areas pre vs post?";
  scripts/interhemi_crossarea.py -> iti_licks/cross_area/):** 45 areas, noise correlation of the residual (spout +
  angle mean removed) 1-s window mean across events, plus within-window ('fast') correlation; raw and global-controlled
  (partial | mean of other 43 areas). REST: everything rises (all pairs z +0.16 / +0.12 / +0.19 acute / sub / chronic,
  4/4 each; ipsi = contra; PC1 variance fraction +0.07, 4/4) -> a GLOBAL synchrony increase at quiet rest from acute
  on; with the global mode controlled the orofacial network still gains relative to other areas (+0.14 / +0.10 / +0.12,
  4/4 every epoch) and homotopic pairs gain chronically (+0.11, 4/4). LICKS (ITI docked, cued): NO global rise (all
  pairs, PC1 ns) but the orofacial / frontal-lateral network covaries MORE (raw ITI-dock +0.25 every epoch 4/4; cued
  +0.18 / +0.09 / +0.19 4/4) while posterior sensory pairs fall -> lick-state change is network-specific, rest change is
  global. No ipsi- vs contralesional hemisphere difference anywhere. CONFOUND NOT YET TESTED: the residual model does
  not remove trial-to-trial lick vigour / kinematics, which drive exactly the orofacial areas -- a post-stroke rise in
  movement variability would produce the lick-state orofacial rise (add reach / speed / lick count to the design next).
  The global-controlled 'all pairs' mean is ~constant by construction (not a readout). Recovery: 15 / 120 within-animal
  tests p < 0.05 (~6 expected), 4 also 4/4; rest orofacial coupling vs far-contra aim r -0.49 4/4 (more coupling with
  worse aim) is the only rest hit with all animals agreeing -> hint only. PROVISIONAL (n = 4).
- **LICK VIGOUR CONTROL OF THE LICK-STATE COUPLING RISE (10-30; scripts/coupling_vigour_sm.py -> iti_licks/coupling_vigour/):**
  same events / code as cross_area, residual design + per-event movement (3-D reach, peak speed, lick duration, cam4
  protrusion, lick count +-0.5 / +-1 s, FaceRhythm 5 factors; cued also trial jaw |deflection| and 3-D landing error);
  'old' design reproduces xa_sessions exactly; control = same covariates row-permuted (same sessions; PS95 ITI-dock pre
  has only 2 sessions with enough events for the movement design). Coverage: every event matched to licks_all / 3-D
  reach; speed 90-99 %, FR ~100 %, jaw 91-100 % (cued); ITI licks have no jaw or landing error. Movement explains ~27 %
  (cued) / ~33 % (ITI dock) of the orofacial-pool window-mean variance pre (permuted ~3-4 %). RESULT (raw, slow, change vs
  pre, acute / sub / chronic): CUED orofacial network SURVIVES: +0.17 / +0.09 / +0.19 permuted -> +0.13 / +0.08 / +0.19
  movement (4/4, 3/4, 4/4); ITI DOCKED absolute rise mostly ABSORBED: +0.25 / +0.22 / +0.25 (4/4) -> +0.05 / +0.10 / +0.17
  (2/4, 4/4, 3/4; CIs include 0). Orofacial-minus-other-areas survives in both (ITI dock +0.30 / +0.20 / +0.29, cued
  +0.20 / +0.17 / +0.25, 3-4/4) because movement also lowers the other areas' covariance. Global-controlled: unchanged
  (nothing in ITI dock either way; cued acute oro +0.08 4/4 only with movement). MOVEMENT VARIABILITY does NOT rise:
  within-spout SDs of reach, duration, protrusion, lick count FALL post (cued 4/4 for duration all epochs), only jaw
  (cued acute +20 % of pre SD, 4/4) and peak speed (subacute +11 %, 4/4) rise. What rises in the ITI is the COUPLING of
  orofacial cortex to movement (R2 of movement on the oro pool +0.27 acute 4/4, +0.12 subacute 4/4, +0.06 chronic). =>
  The cued-lick orofacial coupling rise is not a vigour artefact; the ITI-docked one is largely shared movement-linked
  variance (movement-to-cortex coupling is stronger after the stroke, not movement more variable). Linear covariates
  only; FR in the ITI is 5 Hz; jaw is per trial. PROVISIONAL (n = 4).
- **SENSORY -> MOTOR COUPLING KINETICS (10-30, Priya "sensory-motor coupling kinetics, inter or intrahemispheric";
  scripts/coupling_vigour_sm.py -> iti_licks/sensorimotor_kinetics/):** S = SSp-n / SSp-m / SSs / SSp-ul, M = tjM1 / ALM /
  MOp_rest / MOs_rest; groups Sc->Mc, Si->Mi, Sc->Mi, Si->Mc (16 pairs each); residuals (lick sets movement-partialled),
  deconvolved tau 0.6, xcorr asymmetry (+ = S earlier) and Granger S->M minus M->S. PRE: Granger S->M > M->S in every
  group / state (+0.01-0.02 cued, +0.03-0.07 ITI dock, +0.04-0.08 rest); xcorr ~symmetric. POST: the S-earlier side
  WEAKENS in all four groups and all three states, from acute on: asymmetry -0.02 to -0.07 (mostly 4/4 every epoch;
  cued subacute / chronic Si->Mc -0.047 / -0.044 4/4), Granger direction -0.02 to -0.07 in ITI dock and rest (3-4/4;
  cued small, mostly ns) -- carried by LOWER S->M gain (rest -0.03 to -0.05, M->S unchanged). Intra = inter (difference
  ~0); hemisphere contrast: only Granger at rest / ITI-dock chronic, where ipsilesional Si->Mi loses MORE than Sc->Mc
  (+0.02, 4/4) -- hint. GLOBAL SIGNAL REMOVED: asymmetry change survives acute / subacute (4/4), fades chronic; the
  Granger loss VANISHES (rest ~0; cued even +0.01) -> the Granger change rides on the global mode that rises at rest.
  Same with the published (non-movement) design. READING: post-stroke a sensory-led component of S-M covariation is
  reduced everywhere, bilaterally and at rest -> state-wide, not lick- or lesion-side-specific; GCaMP6s and any change
  in relative S vs M signal / kinetics can shift these shape metrics, so this is SHAPE, not lag. PROVISIONAL (n = 4).
- **DOCKED-SPOUT ITI LICKS (10-29, Priya: ITI offset results should use docked-spout licks; iti_dock_audit.py,
  lick_iti_dock_compare.py, page 14.18 supp):** docked (dock -> next trial_start) = 35 % of ITI licks. HOLD: same-place
  offset +0.30 / +0.22 / +0.29 mm, matched-error offset +0.22 / +0.18 / +0.22, SSp-n L/R term +0.26 / +0.23 / +0.20
  (3-4/4, CIs exclude 0). LOST: primary ITI balance (acute -0.11 ns vs -0.34), within-spout balance, C/I landed frame,
  ITI-balance recovery tracking (LOAO -0.11 vs 0.42); SSp-n tracking weakened. Moving-spout licks subsampled to the
  docked count reproduce everything -> not power. => The lasting ITI PATTERN offset is a docked-spout result; the ITI
  hemispheric BALANCE and its recovery link come from licks while the spout moves (retraction / outbound) -- possibly
  spout-driven licks. Earlier ITI-balance claims (0b: ITI balance deficit 4/4, LOAO 0.32-0.37 tracking, matched-error
  ITI within-spout balance) are MOVING-SPOUT results. PROVISIONAL.
- **ITI pre-lick window (10-29, 12.17 supplement):** around spontaneous ITI licks (first after >= 2 s quiet), the
  frozen decoder's post-stroke contralesional offset is FLAT ~+0.1 mm before the lick (no ramp; mostly PS94; not a
  cohort effect) and steps up AT LICK ONSET to +0.25-0.40 mm (4/4, every epoch); SSp-n difference rises from ~-0.2 s
  acutely (tongue protrusion / interpolation) and only at the lick chronically; chronic balance imbalance only from
  +0.2 s; pre-lick windows do not track recovery, post window does (4/4, p < 0.001). Pre-stroke there is no direction
  information beyond the spout before the lick -> plan untestable at area level. Reading: the persistent ITI shift is
  post-ONSET (execution or self-generated tongue / mouth feedback; NO spout contact on ITI licks -- Priya), not
  preparatory. GCaMP6s CAVEAT: fluorescence lags spiking by ~0.1-0.2 s, so a step at 0-0.1 s after onset is compatible
  with a late motor command (~0.1-0.2 s before onset) as well as early feedback -- timing alone cannot separate them.
  scripts/lick_iti_prelick.py. PROVISIONAL.
- **'Revving up' to far-contra (10-29, behaviour; 14.21 area supplement; scripts/revup_behaviour.py):** acute far-contra
  median latency +556 ms (vs other spouts +470, 4/4), contact +766 ms, no-lick +0.27; near pre by subacute, chronic 43 ms
  faster than pre (4/4). PS92 / PS93 = whole distribution later (slowed initiation); PS94 / PS95 = bimodal (normal or
  failure to initiate; far-mid affected too). Delay = mostly cue -> first movement WAIT (+190 ms) > movement -> lick
  (+129 ms); aborted attempts before the first lick do not increase; no long vigour ramp; aborted (no-contact) licks rise
  AFTER initiation (+0.74 / trial). First licks after the spout left (5.7-10.7% acute): [CORRECTED below -- the
  'at/below ITI chance' comparison used a ~10x too-high chance rate]. Latency tracks aim within animals (r -0.72 to -0.92, p < 0.001). PROVISIONAL.
  NEURAL (scripts/revup_neural.py): slow first licks = DELAYED START, not slower ramp -- lick-aligned latency quartiles
  overlap (same rise, same level at the lick; deconvolution unchanged); acute far-contra slow trials OVERSHOOT
  (contralesional ALM/tjM1/SSp-m/SSp-n, 3/4); shape conclusions rest on PS92/PS93 (only ones with slow trials). Higher
  pre-cue level -> slower lick in PS92/PS93 in every epoch incl. pre (pre-existing). Engaged late first licks (48)
  look like DELAYED CUED licks (cued-size response, cued-like shape) -- conflicts with the behaviour chance-rate result
  (whose after-left trials are mostly disengaged sated-tail trials) -> reconcile engagement definitions.
  FOLLOW-UP (10-29, scripts/revup_followup.py; page panel after rev-up NEURAL): JAW -- no ramp: cam4 jaw motion energy
  and displacement flat until ~-0.2 s then an abrupt normal opening in every latency quartile / epoch incl. acute
  far-contra slow trials (wait late vs early -0.10 [-0.37, 0.17]; slope slowest - fastest vs pre -2.6 [-16, 11] px/s);
  slightly more background jitter during acute slow waits (+0.40 [0.11, 0.65], 3/4); slow trials open LESS at lick
  onset (-6.7 px, 4/4); FR prep-jaw factor LOWER in long waits (-1.03). -> rev-up = a WAIT, not a ramp. LATE LICKS --
  the behaviour chance rate (all-ITI bout starts) was ~10x too high; against the same no-lick trials' own rate > 1.5 s
  after the spout left, first licks within 1 s of the spout leaving are above chance in 4/4 acutely (excess +0.12
  [0.04, 0.22]; far contra +0.15 [0.03, 0.29]) but also pre (+0.25) -> delayed or spout-triggered responses, not chance
  licks. GATES: the reference (canonical, sated tail out) is what revup_behaviour used; revup_neural used the
  'attempted' gate (OROFACIAL_GATE=attempted from frozenenv; 200 no-lick trials) -> its 48 late licks are a selected
  subset, too thin to test (+0.11 [-0.02, 0.30]). PROVISIONAL.
- **ENL -> first-lick angle (10-29, 14.21 supplement):** pre-cue (-1..-0.1 s, no lick in 2 s) area features predict the
  upcoming first-lick angle weakly (held-out r 0.30/0.34/0.50/0.45 vs spout means alone 0.89/0.82/0.82/0.76); the part
  beyond the spout is significant in 4/4 but tiny (r 0.06-0.12) and partly trial history (collapses in PS95 / PS92 with
  previous-trial controls) -> not yet an aim / plan signal. Frozen all-pre decoder post-stroke: raw r -0.17 acute (4/4),
  recovered chronic; like-for-like offset +1.55 deg contralesional acute (4/4), gone by subacute. Pre decoders now via
  frozen_models.load_or_fit (rule 10; results identical). scripts/lick_enl_angle_decode.py. PROVISIONAL.
  FOLLOW-UPS (scripts/lick_enl_offset_explore.py): within-spout ENL signal over trial history 2/4, over history +
  pre-cue FaceRhythm 1/4 (PS94) -> no robust pre-cue aim signal. The acute +1.55 deg offset: at the same TARGET the
  decoded angle barely moves (-0.19, ns) while execution moves -12.9 deg (4/4) -> ENL reads the target; the offset is
  the aim deficit seen through a target-reading decoder (execution term +1.97 / +1.35 / +0.95, 4/4); not far-contra
  specific, not a session baseline shift; F0 part in PS92 only. Same lesson as the cued post-lick balance.
  CAVEAT (10-29): decoder compressed (decoded spread ~1/5-1/2 of actual); at FAR-CONTRA the pre-cue read-out moves
  ipsilesional acutely in 3/4 (PS92 -5.0 deg = ~full tracking given its gain; PS93 -1.8, PS94 -2.2; PS95 +0.2) ->
  'pre-cue target representation unchanged' NOT established; next: gain-normalised tracking + frozen spout classifier.
  RESOLVED (10-29, scripts/lick_enl_target_test.py; frozen position decoder pre-cue softmax, attempt_coding precue
  generated 10-29): before the cue the target code is DEGRADED but NOT re-aimed -- class-matched within session,
  deviated far-contra trials look like same-session hits pre-cue (target -0.07 [-0.19, 0.05], landing +0.04 ns) while
  cue (-0.21 / +0.13) and lick (-0.24 / +0.18) windows show the landing signature 4/4; acute accuracy -0.12 (4/4),
  far-contra AND close-ipsi diagonals drop alike, no off-diagonal cell grows (row flattens); ridge 'tracking' = PS92 +
  regression to the mean. Deviation becomes visible in cortex only after the cue. F12 applies (spout present).
  PRE-STROKE SAME PATTERN (10-29, Priya "what about pre stroke"; now a standard output of lick_enl_target_test.py,
  --classmatched-only, figure lick_enl_target_classmatched_epochs.png; animal-weighted boot_delta as the post rows):
  only 34 deviated pre far-contra trials (PS92 1 / PS93 17 / PS94 12 / PS95 4) -> thin. Primary rule (sessions with
  >= 2 deviated; PS92 drops out): pre-cue target +0.19 [-0.18, 0.75], landing +0.08 [-0.23, 0.36] (ns); cue ~0 (ns);
  LICK target -0.28 [-0.52, 0.00], landing +0.16 [0.00, 0.32] (borderline). >= 1 rule: lick -0.44 [-0.83, -0.10] /
  +0.09 [0.00, 0.19]. Pre-stroke errors also look like hits before the cue and show the landing signature at the lick
  -> the post-stroke pattern looks like the normal error signature, made more frequent (pre-stroke CIs are wide). Post
  rows unchanged by the change (checked to 4e-6). (An earlier scratch version pooled sessions flat: +0.06 / -0.38.)
- **Contact vs miss (10-29):** the primary balance is as large or larger on first licks that MISS the spout (acute PS94
  -1.08 vs -0.67) and tracks recovery as well (cross-animal 0.66 both) -> not mainly touch reafference; on STRICT misses
  (no contact within 0.5 s) the recovery link fails in PS92 / PS95 -> a partial touch / reward contribution to the link
  is not excluded. scripts/lick_balance_contact.py. PROVISIONAL.
- **Lesion size vs phenotype (10-29, n=4 descriptive):** tissue loss PS93 55 ~ PS92 53 > PS94 32 > PS95 18 %, but
  behavioural severity PS94 > PS95 > PS92 > PS93 (acute aim -39 / -31 / -27 / -21 deg; Priya: PS94 most severe, PS95
  least); more tissue loss -> smaller deficit (PS92/93 lesioned twice) -> lesion LOCATION likely matters. Acute bilateral
  response increase and acute balance follow behavioural severity (PS94 largest); chronic response growth follows tissue
  loss / re-lesioning (PS92/93). lesion_component.csv. PROVISIONAL.
- **Imaging control, primary balance (10-29):** F0 asymmetry drops acutely (PS92/93) -- wrong sign for an artefact
  (undoing F0 enlarges the shift); 415 nm balance ~0 and unshifted; F0 shares the time course (partialling it out:
  cued acute -0.23, recovery cvR2 0.42 cued / 0.27 ITI, both within-animal p < 0.02). Not an imaging artefact.
  scripts/lick_balance_imaging_control_primary.py. PROVISIONAL.
- **Trial by trial (10-28 late):** the balance AFTER the first post-cue lick follows that lick's direction within a
  session and target (r 0.3-0.5) -- equally BEFORE the stroke (normal hemispheric direction code); the balance BEFORE the
  lick (-0.5..0 s) predicts nothing (r ~ 0, all epochs). No evidence a pre-movement imbalance causes the error; the
  stroke shifts the post-lick code across sessions (a state). Pre-cue at the far-contra target: contralesional orofacial
  cortex SUSTAINED raised acutely (not a ramp; 14.20). The balance is only PART of the post-stroke direction change: ~35-50%
  of the ITI offset (12.17), balance-blind decoders keep the offset (12.20), and for ITI the decoder offset is the better
  recovery marker (0.39 vs balance 0.07 in-range) -> the change lives in the orofacial PATTERN; balance = its simplest summary. PROVISIONAL.
- **Recovery markers 10-28 (earlier definition; now a collapsed supplement of 14.17): BALANCE tracks recovery, response SIZE does not.** Cued orofacial balance (contra -
  ipsi): leave-one-animal-out cvR2 0.50 (z-scored per area 0.53; ALM 0.61, SSp-m 0.60, tjM1 0.52 alone), leave-one-EPOCH-
  out 0.41 vs days 0.23; same sign 4/4. Orofacial SIZE (mean of both sides), contra or ipsi alone, and the 0-1 s response /
  cue AUC all fail (LOAO -0.28 to -0.70; epoch-out to -1.23): their sign FLIPS between LESION GROUPS (corrected 10-29: PS94 / PS95 = single 3 mW lesion, SMALLER loss, PS95 least severe; PS92 / PS93 = re-lesioned, larger loss) (PS94 / PS95:
  largest responses acutely at the worst aim, then shrink; PS92 / PS93: responses grow over recovery). SSp-n balance
  tracks within animals (r ~0.9) but not across (0.04; z-scoring does not fix it). Cued FIRST lick only: weak (0.06, one
  lick per trial). ITI all licks: none; ITI first lick after >= 2 s quiet: balance 0.29 (shuffle p 0.006; SSp-n 0.43,
  z 0.52), 3/4 animals, no epoch-out gain. Pre-fit weighted balance does not transfer across animals (fair weights
  -0.08). ITI licks have their own recovery: mean ITI angle -12.6 / -6.0 / -3.8 deg vs pre (r with cued aim 0.77-0.95);
  only SSp-n first-after-quiet balance predicts it in an unseen animal (0.32 vs days 0.17). PROVISIONAL.
- Pre-cue LEVEL vs restw 10-28 (page Fig 14.21; restw approved by Priya): far-contra target acute, contralesional ALM / tjM1 /
  SSp-m / SSp-n RAISED before the cue (+4.5 to +5.5 x1e3 dF/F, 4/4, PS94 ~0); chronic all spouts ALM / tjM1 contra
  +1.5 / +1.7 (4/4) and a posterior VIS / RSP / SSp-tr / bfd decrease (partly a rest-level rise). PROVISIONAL.
  Caveat (10-29 ENL audit): non-contact ENL licks occur up to the cue on 10-37% of ENL trials, so the pre-cue window is not
  movement-free; the acute far-contra raise needs an ENL-lick / movement check before it is read as preparatory.
- ENL lick audit 10-29 (page Fig 14.21 S9, scripts/enl_lick_audit.py, enl_licks/): ENL licks (strobe <= onset < cue; spout
  stationary, no cue, no reward) are plentiful (107 sessions; no animal x epoch x spout cell < 20; thin only PS95 acute =
  1 session). Only CONTACT licks reset the pre-cue timer; none abort the trial. Excluded from the ITI and cued decoder sets
  (in the frozen encoder). Acute far-contra ENL aim error = cued (pooled -25.9 vs -25.4 deg; per animal -26.6 / -13.0 /
  -30.6 / -34.3 vs -22.2 / -15.5 / -32.5 / -31.3), contact 0.15 vs 0.13, recovers alongside cued -> the far-contra aim
  deficit does not need the cue. Caveats: selected set (trials with an ENL lick fall to 0.11-0.54 acutely); spout-arrival
  overlap (PS94 / PS95); chronic ENL vs cued differ in PS94 / PS95 (untested). PROVISIONAL.
- ENLv licks NEURAL 10-29 (page Fig 14.21 S10, scripts/lick_enlv_*.py, enlv_licks/): primary = onset >= 1 s after spout arrival,
  window ends before the cue (bout starts alongside; all-lick balance unstable); 21-variant sweep incl. arrival regression,
  contact-only, first lick: offsets robust. Decoder offset (frozen pre ITI reader, local store) same-place +0.20 / +0.16 /
  +0.12 mm (4/4, ~half ITI); matched target + landing (bout starts) +0.22 (3/4) / +0.13 / +0.16 -> does NOT vanish (cued
  does). PS92 ~0 on bout starts (pooled leans on PS93-95). Per spout: largest at far mid (4/4); far contra depends on the
  lick set (all licks +0.11 / -0.03 / -0.09; bout starts +0.34 / +0.15 / +0.19, PS94 / PS95; contact-only subacute / chronic
  -0.17 / -0.27, CI excludes 0 chronic: far-contra licks that reach the spout read slightly ipsilesional). SAME in cued
  licks (Priya 10-29): cued far-contra chronic -0.27 [-0.39, -0.14] 4/4; Fig 12.2 furthest contra landings below pre. With
  ipsilesional landings reading contralesional, this looks like a flatter read-out (lower gain), not decoder edge
  compression. GAIN/OFFSET (10-29, scripts/lick_enlv_curves.py, Fig 14.21 S10 curves): ITI slope -0.12 / -0.11 / -0.12 (4/4)
  + offset +0.35 / +0.22 / +0.27 mm; CUED slope -0.11 / -0.10 / -0.11 (4/4, 3/4, 3/4), offset only acute (+0.17) -> the cued
  change is mostly a FLATTER read-out; ENLv first lick >= 1 s: mostly offset (+0.33 / +0.21 / +0.22) + modest gain loss.
  Edge check: pre curves not compressed at the contra end (cued pre local slope 0.49-0.93 vs post 0.18-0.25) -> activity
  differs; at the ipsi end the pre curve is flat (decoder floor). Untested: noisier post landing measurement (dilution).
  ALL LICK TYPES (lick_like_for_like_all.png): ITI first-after-quiet and cued FIRST lick also read contralesional every
  epoch (offset +0.30/+0.19/+0.23; +0.16/+0.13/+0.08 mm, 4/4); cued first: no gain loss. Far-contra chronic ipsilesional
  reading is in LATER cued licks: 1st +0.11 n.s., 2nd-3rd -0.21 (4/4), 4th+ -0.38 (4/4), growing with time since stroke,
  with or without contact. Confound: lick order ~ time in trial / repeated contact. Balance (bout starts)
  acute -0.24 (4/4) -> -0.08 n.s. target-aware; chronic within-spout +0.14 (4/4) = the CUED pattern; pre ENLv post-lick
  activity carries a target component (w 0.59-0.90). Working description (Priya OK 10-29): ENLv = cued-like target-coded balance + ITI-like
  persistent decoder offset. First ENLv lick: largest offset but mostly < 0.5 s after arrival. PROVISIONAL.
- **DISENGAGEMENT tests 1-4 (10-31, pre-registered `docs/status/DISENGAGEMENT_PREREG_20261031.md`, `scripts/disengagement_tests.py`,
  page S28):** quitting sooner post-stroke is clear only in PS92 / PS93 (pre quits 0/8, 1/8; PS94 / PS95 4/8, 6/8). Holm over
  P1-P4, primary epoch early post. P1 SATIETY: collected water at quit / session end early post -290 ul [-428, -154], Holm
  .001, 4/4 (acute -327, subacute -210, chronic -53 n.s.), slower accrual, weight -5 % -> NOT satiety at the old level; the
  quit threshold moved (motivation / malaise vs lowered set-point undecided). P2 FRUSTRATION (far-contra block lapses)
  +0.037 n.s.; giving-up rate mixed. P3 FATIGUE n.s. (bouts shorten before quits, tongue does not slow). NB 11-01: P2 / P3 test only
  failure-SPECIFIC lapses and within-bout lick interval; broader frustration / fatigue is NOT ruled out. P4 STATE: global
  pre-cue level quit minus time-matched engaged +3.65 [+1.85, +5.30], Holm .001, 4/4 -- BUT running rises 4/4 and on
  still pre-cue windows the level is BELOW engaged, -7.2 [-10.3, -4.3], 4/4 (post hoc). Cue response falls at, not before,
  the quit; no prodrome in cortex / face / running. No pupil. Open: RESTW sensitivity, running-regressed level, exact
  80-min horizon re-run. **PROVISIONAL (n = 4).**
  ADDENDUM 10-31 (pre-registered A0-A7, page S28d2-g): P1's water shortfall is partly SESSION LENGTH (pre ran to ~150 min,
  post stopped at 100 trials / spout or 120 min): exact 80-min re-detection -105 ul [-225, +10], 3/4 -> satiety NOT rejected
  on equal-length data. Minutes to quit (capped 90) -5.6 [-14.3, +0.7] Holm .18 (80-min: -4.5, 4/4); trials capped 300 null;
  responded trials to quit-or-end -146, 4/4. P6 running-regressed quit-period level -7.8 [-11.3, -4.2], Holm .001, 4/4
  (RESTW -7.4) = CONFIRMS still disengaged cortex below engaged; mainly RSPd / VISpm (+ ALM), orofacial ~0. Test 4b
  (Priya: running = self-activation of orofacial cortex?) NOT supported: no re-engagement / response gain after running,
  orofacial < reference running effect (4/4), running lowers engaged responding (-0.11, 4/4) -> generic disengagement.
  Licks / responded trial vs own pre per spout: acute far contra ~1/3 (4/4); chronic far contra +0.23 log2 (Holm .018, 4/4,
  cam4) but not in DAQ contacts (-0.01). Session-end classes for Priya: disengagement/tables/a0_end_class.csv.
  UNCAPPED time / trials to quit (Priya 10-31, after the capped result; outside Holm): KM restricted mean to a per-animal
  common horizon, early post -15.1 min [-25.4, -5.6] 4/4, -77 trials [-144, -20] 4/4, responded -103 4/4; chronic ~0.
  Lower bound (censored ends = quits): -27.4 min, -69 trials (3/4), -146 responded. -> animals DO quit sooner early post
  (~15 min / ~75 trials), recovered chronically; the 90-min / 300-trial caps sat below most quits.
- CAVEAT 10-31 to the next line: the raised stopped-trial level is largely RUNNING (disengagement tests P4, post hoc).
- Stopped-trial traces 10-29 (page Fig 14.21 S11, scripts/stopped_trial_traces.py, stopped_traces/): stopped (quit-period, no
  lick) trials have NO cue-evoked response in any area / epoch, only a flat level above restw (~4-12 x1e3); miss_working a
  small cue bump. Stopped reaches only 3 animals per epoch (animal set changes; PS93 pre 40 trials). Engaged jaw-no-lick
  (mostly incomplete licks) small cue bump; jaw-onset transient ~0.3 s in every area incl. references (broad, movement).
  Priya's observation: stopped contra tjM1 / SSp-m / ALM pre-cue level rises over recovery within animals (tjM1 PS94 5.1 /
  8.8 / 11.6, PS93 6.0 / 6.4 / 7.8; chronic > pre in 3/3); ipsi ALM also rises in PS94 / PS95 -> contra specificity
  untested. FOLLOW-UP: the restw-unit rise (pre->chronic 3/3: ALM +3.9, tjM1 +6.5, SSp-m +7.4) is mostly a FALL of the
  restw reference; raw dF/F shows it only in PS93 (pre = 1 session); contra-ipsi < 1 unit except PS93 (mostly bilateral);
  the same chronic restw-unit rise is in hit + miss_working (4/4) -> general level change vs rest, not stopped-specific.
  No consistent chronic rise in quit-period movement. Late-session confound. DESCRIPTIVE.
- Cue to lick 10-29 (page Fig 14.21 S12, scripts/lick_cue_to_lick*.py, cue_to_lick/): (A) mean level cue->lick minus restw:
  contra ALM / tjM1 up every post epoch 4/4 (~+2-3), latency-matched too; above the trial's own pre-cue level acute far
  contra ~-2 n.s. -> the restw rise is the sustained pre-cue raise. (B) accumulate-to-bound REJECTED (level at lick flat /
  slightly rising with latency, slope ~ 1/latency, pre and every epoch); a DELAYED START (M3) fits all three (fixed level M2
  fails the AUC above own pre-cue, which stays small and flat; corrected 10-29); AUC rise
  with latency is mostly the cue-locked response; acute far-contra slow trials = flat wait then normal rise to a normal
  level (delayed start). Pre latencies only ~0.1-0.6 s. (C) fixed pre-lick windows after the cue keep few trials
  (effectively PS92 / PS93). (D) chronic raw pre-cue "dip" in orofacial areas = drop of the restw level relative to the
  session mean (4/4), removed by restw; F0 not the cause chronically; non-orofacial (bfd, VISp, MOs_rest ipsi) dip is a
  real pre-cue decrease. CAVEAT for Fig 14.21: the chronic contra ALM / tjM1 pre-cue rise is relative to rest, not absolute
  (dF/F is session-mean centred). PROVISIONAL.
- DOCKED-SPOUT ITI = PRIMARY ITI SET 10-29 (Priya): 39 main ITI figure images swapped to docked versions (itidock_main/,
  scripts/iti_dock_page_figures.py, iti_dock_where_preds.py; default-off WFIELD_ITI_FILTER hooks in 14 scripts). SAME on docked:
  like-for-like offset (+0.31/+0.21/+0.28), same-target offset, per-lick error shift, balance-blind decoders, lick order,
  lateral r loss (larger), offset-vs-aim detrended (-0.38, 4/4). WEAKER-SAME: curves, per-spout, decomposition, response
  increase (subacute lost), LMMs. NOT on docked (kept all-ITI with a caption note): ITI hemispheric balance + recovery
  tracking (14.18), balance-offset link (12.19), bout-start balance (14.14), SSp-n-term ITI tracking (12.17, 2/4),
  per-animal offset-vs-aim slopes (16.5), chronic build-up decrease (14.22). PROVISIONAL.
- RESTnl BASELINE ADOPTED 10-29 (Priya): level family (Fig 14.21, S1, S2, S11-S13) re-rendered on the lick-free rest + current
  engaged gate (scripts/restnl.py, --rest-variant restnl in 7 scripts; *_restnl outputs; restw kept as sensitivity). Chronic
  pre-cue rise larger and 4/4 in all key areas, both hemispheres (contra tjM1 +5.2, SSp-m +4.8, SSp-n +3.2, ALM +2.9; ipsi tjM1
  +4.1, SSp-m +4.5) -> bilateral; tjM1 leads every epoch; acute far-contra raise holds (tjM1 +7.4, 4/4); chronic non-orofacial
  falls mostly restw artefacts (significant falls 20 -> 4 areas; some posterior / RSP still fall) and the reference-area part
  of the dip decomposition is n.s. Cue-to-lick level
  larger (tjM1 chronic +6.3, 4/4); far-contra ALM acute loses significance (+2.7 n.s., 4/4). Stopped: rest-referenced rise is
  ~76-83% the reference falling. Threshold test unchanged (own pre-cue reference). Caveat: PS93 pre = 2 sessions. PROVISIONAL.
- RESTnl RULE -> RESCUE VARIANT s20_f50_wr 10-30 (scripts/restnl.py default; strict kept as `--restnl-rule strict` ->
  *_restnl_strict outputs; sweep scripts/restnl_rescue.py): same mask and 2.0 s post-lick buffer, >= 50 frames per
  position, but the per-position medians are averaged over the positions present (>= 4 of 6) instead of requiring all 6.
  107 / 107 sessions (was 97; PS93 pre 2 -> 7, PS92 pre 6 -> 8). All *_restnl level-family outputs re-rendered (same names;
  per-session caches in new *_wr dirs, strict caches untouched). Headlines unchanged in sign, significance and 4/4:
  H1 chronic contra pre-cue tjM1 5.24 -> 5.21, ALM 2.93 -> 3.11, SSp-m 4.78 -> 4.44, SSp-n 3.20 -> 2.61; H2 acute
  far-contra tjM1 7.38 -> 7.44, ALM 6.09 -> 6.40, SSp-m 7.26 -> 7.06, SSp-n 7.22 -> 6.50; H3 chronic cue-to-lick (raw) tjM1
  6.34 -> 6.39, ALM 3.93 -> 4.29, SSp-m 5.65 -> 5.42, SSp-n 2.88 -> 2.47 (still n.s., 3/4). Only PS92 / PS93 move (PS93's
  SSp-n / SSp-m shrink most with 5 more pre sessions). PROVISIONAL.
- RESTnl SUPPLEMENT CAPTIONS 11-01 (goal page Fig 14.21 S1/S2/S11-S13 numbers rewritten from the rescue-rule *_restnl
  outputs; baseline notes removed). Changed materially: S2 acute far-contra raise now NOT orofacial-specific (contra
  MOp_rest / MOs_rest / SSp-bfd +7.1 / +6.9 / +5.1, significant; ipsi tjM1 +6.1; PS94 +2-4, was ~0); S11 stopped
  chronic rise tjM1 +8.4 [+6.5], contra-ipsi only 2/3 positive [3/3], hit chronic rise tjM1 +5.3 [+1.8]; S12 A tjM1
  +4.3/+4.1/+6.4 [+2.4/+2.2/+2.9], ipsi tjM1 chronic +4.3 4/4 (was no change), B pre AUC residual slope +0.7..+1.1
  (was ~0; trial set, PS94/95, fragile), C more significant cells (PS95-driven), D RESTnl drop -8.0/-9.0 > raw dip
  (rel +5.2/+4.4) and the non-orofacial dip is n.s. vs RESTnl; S13 last-lick tjM1 pre values lower, chronic > pre
  4/4 in the last 0.5 s before the cue (tjM1, ALM). Unchanged: exponents / threshold verdict, own-pre-cue A (-2 n.s.),
  counts. PROVISIONAL.
- RESTnl MAIN CAPTION + S7 11-01 (goal Fig 14.21 caption numbers now rule-wr *_restnl; S7 revup_neural --rest-variant restnl,
  RESTW collapsed). Changed under RESTnl: all-spouts ACUTE = 12 areas RISE 4/4 (RSP / VIS largest, +2.6..+4.7; contra ALM +1.6)
  [RESTW: only ipsi SSp-ul / -un -4.4, now -2.5 n.s.]; far-contra acute raise broad, 34 / 45 areas (RSPd +8.2, tjM1 +7.4)
  [RESTW 8]; far-contra chronic bilateral tjM1 / SSp-m / SSp-n +3.4..+5.4, 4/4 [n.s.] and its 11 posterior falls gone;
  all-spouts chronic falls 20 -> 11 (VIS / RSP only; the earlier '-> 4' was the strict rule). Caveat made explicit: lick-free
  rest itself falls in contra tjM1 / SSp-m (-8.0 / -9.0 chronic, 4/4) more than the unreferenced pre-cue level (-2.7 / -4.6),
  so 'rise above rest' = falls less than rest. S7: every within-session revup result identical (flat shift); only trace
  levels and the session-level r move. PROVISIONAL.
- FIG 0.12 + FAR-CONTRA AIM 10-30 (Priya: "can 0.12 include the far contra lick angle index?"; scripts/recovery_timing.py
  --aim [--plot-only] -> attempt_coding/figures/recovery_timing_pooled_day_lick_aim.png, panel B = far-contra first-lick
  3-D aim, change vs own August pre, deg, + = contralesional; panel A reproduces 0.12 exactly). Aim drops to -32 deg
  (pooled, day 1; 4/4 ipsilesional) and recovers on the HIT-RATE time course (t1/2 aim 3.1-5.4 d vs hit rate 3.9-7.9 d;
  ~70% back by day 9-12 when hit rate is 0.8-0.95) but PLATEAUS SHORT like the code: chronic -5 to -6 deg pooled (95% CI
  below 0 in every day bin to day 51, upper ends -0.1 to -1.4 deg late); recovered fraction of the acute deficit at chronic: aim 0.84 / 0.76 / 0.55 / 0.90
  (PS92-95), hit rate 1.0 in 4/4, code on hits 0.67 / 0.50 / 0.23 / n.a. (PS95 no acute code). Residual chronic aim is
  mostly PS94 (-14 to -18 deg; also lowest code); PS92 / PS93 / PS95 within ~0-5 deg. PROVISIONAL, n = 4.
- ITI PRE-LICK RE-RUN 10-29 (Fig 12.17 S3; windows end >= 0.1 s before lick onset, + jaw-onset windows; jaw leads 52-120 ms):
  decoder offset flat pre-lick (+0.14/+0.12/+0.15 ns, mostly PS94), steps up post (4/4) -- unchanged; SSp-n L/R difference
  above 0 pre-lick acutely / subacutely (+0.21-0.24, 3-4/4; also before jaw opening), ~tonic, mostly PS94 -> NOT a movement
  artefact; acute baseline-subtracted balance build-up remains (4/4); recovery tracking none pre-lick; direction beyond spout
  ~0 pre-lick. Revised: offset + recovery link post-onset; SSp-n difference / acute balance have a tonic pre-lick part. PROVISIONAL.
- RESTW check 10-29 (Fig 14.21 S14, scripts/restw_strict_check.py, restw_check/): restw rest contains many NON-contact licks
  (13-57/min, most pre-stroke); a strict lick-free rest (all cam4 licks, [0.5, 2.0] s; strict_w50) shifts by epoch (contra
  tjM1 chronic -3.66, ALM -1.72, 4/4; bilateral, cortex-wide) -> restw is NOT epoch-neutral, but conservatively: chronic
  contra pre-cue rise ALM 1.53 -> 3.00, tjM1 1.73 -> 5.31; acute far-contra raise tjM1 5.50 -> 7.35; cue-to-lick tjM1
  chronic 2.89 -> 6.39; all keep sign + significance. Contra-ipsi contrasts little affected; ALM vs tjM1 ordering is
  baseline-dependent. Pre-cue cache engaged counts differ from the current gate in 86/107 sessions (chronic result
  unchanged, acute +0.4 n.s.). PROVISIONAL.
- Arrival -> cue traces 10-29 (page Fig 14.21 S13, scripts/arrival_to_cue_traces.py, arrival_to_cue/): three anchors (arrival |
  last ENLv lick | cue) + wait-sorted heatmaps. Far-contra ENLv contact drops acutely 4/4 (e.g. PS94 0.61 -> 0.00), so those
  licks do not restart the timer (short acute far-contra waits). Response to the last ENLv lick larger acutely than pre in
  contra tjM1 / SSp-n (4/4; lick count / bouts differ); contra tjM1 cue response larger chronically (4/4); last 0.5 s
  before the cue near rest. Long waits = PS92 / PS93 only. Open check: precue_restw cache engaged counts differ slightly
  from the current engagement gate (2/2 sessions checked: 381 vs 364, 473 vs 472). DESCRIPTIVE.
- Decoder area attribution 10-28 (11.13 alone / unique, 11.14 nested blocks + Shapley): the orofacial group (+SSs) carries
  ~40% of the code (Shapley pre r 0.27 of 0.62 ITI, 0.29 of 0.77 cued) and of the ITI offset (0.15 of 0.37 mm acute);
  the rest is spread and redundant (no block necessary). SSs alone decodes but adds nothing uniquely (a copy of the
  orofacial signal). PROVISIONAL.

## 1. Behaviour and orofacial kinematics (cam4)

*Owner 5b; detail [OROFACIAL_FINDINGS.md](OROFACIAL_FINDINGS.md) (its status table, v36, is authoritative). cam4 = LP
v36, August-only baseline, both tests (session Mann-Whitney + BH-FDR and the animal-weighted bootstrap); stats table
`LP_analysis/lp_r4_heatmap_20261013__clean_v36/cohort/tables/prepost/stats/RESULTS.md`. Checked by 5b 10-25.*

1. **Acute global slowing plus a far-contralesional-graded deficit.** Inter-lick interval +14 to +26 ms at every
   position; far contra: contact -0.67, first lick +488 ms, first contact +898 ms. **STANDS**, 4 / 4 (PS95 acute = 1
   session). TIME COURSE (11-05 re-test on camera classes, August-only pre; Priya 11-05): the lick-interval deficit
   **persists into subacute and recovers by chronic** (+18.8 ms acute, +8.0 ms subacute). The 09-20 "recovered by
   subacute" (+12.5 ms acute) used the June + August baseline (docs/status/reanalysis_20261105/
   ENGAGEMENT_CAMERA_2026-11-05.md).
2. **Acute licks aim ipsilesionally**: +30 deg at far contra, +7 to +13 deg elsewhere. **STANDS** (cam4 4 / 4; 3-D 4 / 4,
   far_R midline error +25 / +9.5 / +36 / +33 deg).
3. **The jaw stops steering toward the target acutely**: FAR_DIR -11.4 px. **STANDS** (cam4 4 / 4; 3-D 4 / 4, -0.18 to
   -0.40 mm). "Does not fully recover" **CORRECTED -> NOT ESTABLISHED**: a chronic deficit only in PS92 (-7.5 px) and
   PS93 (-5.5); PS94 -0.5, PS95 -2.9; 3-D agrees.
4. **Jaw movement without a lick on more acute trials** (far_R +0.23). **STANDS**, survives the cam1 correction 4 / 4
   (far_R pre -> acute .02 -> .28, .01 -> .28, .05 -> .39, .08 -> .58). They are REACTIONS, not attempts (blind video
   review 0 / 24), so attempted = hit + incomplete. Source `session_poses/cam1_licks/incomplete/`; review page
   https://claude.ai/artifact/TbQ4pud711MBrT6ygAH5FX.
5. **Far-contralesional protrusion -26 px.** **PROVISIONAL**: mostly a far-spout effect (far ipsi drops too; 3-D reach
   drops at both far spouts in PS93 / PS94).
6. **Chronic: more licks per responded trial**, +1.5 to +4.2. **STANDS**. Other chronic cells not re-checked per
   animal: **PROVISIONAL**.
7. **Incomplete (no-contact) licks rise acutely at far contra**: share of far_R licks .17-.38 pre -> .73-.99 acute.
   **STANDS** 4 / 4. Chronically still raised only in PS92 / PS94 (**PROVISIONAL**).
8. **Licks and misses** (contact, DAQ + LP): far-contra trials are attempted acutely but almost never completed.
   Attempt rate, engaged trials, pooled (recount on the rebuilt trial_attempts.csv, 10-25): .79 / .79 / .58 / .34
   (PS92 / 93 / 94 / 95; n 336 / 292 / 450 / 120; all-trials denominator .62 / .61 / .445 / .342); completion .06 / .15 /
   .04 / .02. PS92-94 "try and fail" (incomplete >> no attempt), PS95 mostly "fail to try" ([ENCODING_FINDINGS 2.5]
   (ENCODING_FINDINGS.md)); completion back to pre chronically (.95-.99); per-lick contra misses mildly up chronically
   in 3 / 4. **STANDS** acutely / **PROVISIONAL** chronically ([FACE_RHYTHM_FINDINGS 5c](FACE_RHYTHM_FINDINGS.md)).

## 2. Pose, 3-D tongue and cam1

*Owner 5b; detail [POSE_MODELS.md](POSE_MODELS.md) Part D (status box) and OROFACIAL_FINDINGS section 8. 3-D = cam1 +
cam4 triangulated with cameras posed per session on the commanded spouts (PnP), mm in the stage frame, origin = the spout
mouth zero; complete (127 / 127 curated cam1 videos, all 119 curated sessions). Outputs `DeepLabCut/Widefield/
session_poses/pose3d/` (per-animal summaries `figures/summary_<animal>.png`); rebuilt by `python -m scripts.pose_chain`.
Checked by 5b 10-25.*

1. **cam4 measures are faithful to 3-D**: lateral r .94-.97 (34-36 px / mm); cam4 "out" is the DOWN axis (r .99 with
   -dv, about 0 with the forward axis), so forward reach is invisible to cam4; jaw per trial r .57-.83. **STANDS** 4 / 4
   (`pose3d/crosscheck/`).
2. **3-D carries more target information than 2-D**: target spout decoded from the lick endpoint (August pre, leave
   one session out, 6 positions, chance .17) 3-D .81 / .67 / .86 / .74 vs cam4 .53 / .42 / .78 / .59. **STANDS** 4 / 4
   (`pose3d/vs_cam4/`).
3. **Acute far-contra licks miss the spout and aim ipsilesionally in 3-D**: closest approach to the far_R spout +2.06 /
   +0.34 / +2.45 / +2.09 mm vs August pre (PS93 small), midline error +25 / +9.5 / +36 / +33 deg; partly back by
   subacute (+5 to +15 deg), near pre chronically except PS94 (+8.5 deg, +1.1 mm). **STANDS** acutely (aim 4 / 4, miss
   3 / 4); chronic **PROVISIONAL** (`pose3d/kinematics/sessions.csv`).
4. **Acute reach drops at both far spouts, not only contra** (far_R -0.06 / -0.63 / -0.44 / -1.50 mm, far_L +0.12 /
   -0.45 / -0.36 / -0.79). **PROVISIONAL**: a far-spout effect, matching 1.5.
5. **3-D CLOSE_DIR** (the jaw moves away from a close side spout). **NOT ESTABLISHED**: PS92 / PS93 only; PS94 / PS95
   flat or opposite.
6. **Whole-session 3-D licks exist** (pre-cue, post-cue and ITI; ITI licks have no target, so target errors are
   blank): `pose3d/kinematics_all/licks.csv.gz`, key `lick_uid`. A resource, not yet a finding; used by the encoding
   window's stray-lick test.

## 3. Imaging: position decoding, state control, where the code moves

*Extracted 10-25 by d4 from PRELIM_DATA_VLS_STROKE (09-26 render), BEHAVIOURAL_STATE_CONTROL, WHERE_THE_CODE_MOVES,
STATUS_2026-10-07, REST_ENGAGEMENT_AUDIT and DECISIONS, with 5b's 10-24 per-animal recheck (imaging page
https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV). **Reviewed 10-25 by 5b**, with e3's 10-25 per-animal / October /
mixed-model re-runs folded in. Imaging decode keeps its
default (June + August) baseline; an August-only rerun confirms the acute numbers (DECISIONS 10-18).*

**Caveat that qualifies several older claims: "no cortex is damaged" is CORRECTED.** BEHAVIOURAL_STATE_CONTROL, PRELIM,
VLS_STROKE_SYNTHESIS and CLAUDE.md argue specificity partly from "the lesion is striatal, no cortex is damaged";
HISTOLOGY.md records an ipsilesional ANTERIOR MOTOR-CORTEX injury in PS93 (and DLS involvement in some animals, from a
fibre-hit vessel). The position-vs-state ratio (claim 2) does not depend on it, but any PS93 frontal-map result should
be read with it, and the "rules out generic explanations" argument holds for 3 / 4 animals cleanly.

1. **A frozen pre-stroke decoder loses the far-contralesional spout acutely, graded by eccentricity.** Far contra
   .896 -> .321 acute (-.575, Bonferroni [-.738, -.394]); 5 / 6 positions significant acutely; far contra and near
   middle still below pre chronically (-.171, -.163). **STANDS** (PRELIM §2 / H1).
   **CORRECTED 11-03 (bal_em spout decoder, production; Priya "I would adopt B everywhere"; DECISIONS 2026-11-03):** the
   old engaged-trial z-score under-stated the far-contra loss and over-stated the near ones. Cue/working change vs pre,
   acute (boot_delta): far contra -.536 -> **-.644** (4/4 larger), far mid -.385 -> -.484, near positions -.19 / -.25 /
   -.26 -> **-.13 / -.20 / -.20**, overall -.314 -> -.318; chronic far contra -.130 -> -.179. Precue: far contra -.254 ->
   -.271 (the EM corrects only ~37 % there); lick: far contra -.465 -> -.483 (3 animals). The graded, far-contra-first
   pattern **STANDS (stronger, more specific)**; the levels above are the pre-11-03 frame. ~30-40 % of the bias remains
   when far contra is nearly absent (conservative). Exact re-rendered levels: page edits
   docs/status/SPOUT_BALEM_PAGE_EDITS_2026-11-03.md.
2. **CORRECTED 11-03: bal_em is the production state decoder (Priya approved).** It is trained on class-balanced pre
   sessions and scored with the label-free EM equal-weight centre (no test labels; DECISIONS 2026-11-03 "bal_em adopted").
   Position loses 50 / 19 / 12 % of its above-chance performance (.863 -> .428 / .696 / .761; acute / subacute / chronic).
   State now loses 4.6 / 2.3 / 5.1 % (retained .972 -> .927 / .950 / .922; session_z was .967 -> .903 / .865 / .872, losing
   7 / 11 / 10 %). So position falls ~11x / 8x / 2.3x further, and the ratio claim now holds at every epoch, not only acutely.
   PS94 running recall is .99 -> .97 / .96 / .92 (session_z .98 -> .84 / .54 / .49).
   Per animal, chronic balanced accuracy is .93 / .98 / .90 / .98. The costs: PS92 acute .91 (quiet recall .80) and PS94
   chronic quiet .89. State still falls slightly (`boot_delta`, balanced accuracy -.022 [-.047, -.003] / -.018 [-.041, .001] /
   -.037 [-.073, -.009]); running-recall changes do not exclude 0. So it is not flat, but the loss is small in every animal,
   and PS94 is no longer the exception. **STANDS (stronger).** The min-mass fallback was tested and kept off (no gain at 0 %
   mixes). Imaging page section 3 edits: docs/status/STATE_BALEM_PAGE_EDITS_2026-11-03.md. The 11-02 / 10-25 text below
   is the record.
2. *(record)* **CORRECTED 11-02 (late): state decoder now z-scored per session** (it was specified, never applied; DECISIONS
   2026-11-02 late): state retained .967 -> .903 / .865 / .873 (was .976 -> .872 / .820 / .838), position unchanged; PS94
   session means .98 -> .87 / .72 / .67, a within-session refit stays .97-.99, so the definitions hold. **CORRECTED 11-03:**
   under this z-scored decoder PS94's loss is the RUNNING class (.98 -> .84 / .54 / .49; licking holds .90-1.0); "licking read
   as quiet" was the unscaled decoder. PS94 runs ~4x more post-stroke, which shifts the all-segment session mean; class-
   balanced centring restores running to .97 / .96 / .91 (diagnostic, decision pending). Running alone pooled .98 -> .92 /
   .83 / .89, no change interval excludes 0 (imaging page Fig 3.5; DECISIONS 2026-11-03). Change vs pre (boot_delta, Holm):
   position -.41 / -.19 / -.09, state -.04 / -.10 / -.11 -> the specificity RATIO is an ACUTE claim (~8x); by chronic the two
   losses are similar, the state part mostly PS94 (PS92 -.10). The text below is the pre-fix record.
2. **Specificity: position loses far more than behavioural state.** Above-chance performance retained (current sidecar
   `epoch_13n_state_vs_position_cue.csv`): position .863 -> .428 / .696 / .761 (acute / subacute / chronic), state
   .976 -> .872 / .820 / .838, running alone ~.97. State DOES fall (acute -.069, Bonferroni interval excludes 0); the
   claim is the RATIO (~5x). **STANDS**; the state numbers in BEHAVIOURAL_STATE_CONTROL / VLS_STROKE_SYNTHESIS
   (.917 -> .812, .982 -> .918, .902) are **CORRECTED** to these. **Per animal (10-25, CORRECTED reading):** the cohort
   state drop is PS94 alone (above chance .978 -> .747 / .577 / .447); PS92 / PS93 / PS95 lose only .01-.09 (chronic
   .873 / .951 / .960), so state is nearly intact in 3 / 4 and the specificity claim is stronger there.
3. **The frozen decoder is NOT back at recovery, but that gap is within two months of no-stroke drift.** Deficit at
   behavioural recovery F(0) +.083, only PS92 recovered; refit-over-frozen G > 0 in 4 / 4 but a healthy mouse already
   shows G +.05-.10 (DECISIONS 09-12, 10-23). June -> August no-stroke drift group (5b, 10-24,
   `scripts/june_august_drift.py`): frozen-decoder deficit / refit gain June -> Aug vs Aug -> chronic .13 / .09 vs
   .03 / .11, .28 / .16 vs .09 / .21, .17 / .18 vs .18 / .20, .18 / .18 vs .11 / .11 -> not beyond drift (3 / 4).
   "Chronic reorganisation" (PRELIM §3, H5) is **CORRECTED -> NOT ESTABLISHED**; "decoder back" **CORRECTED**.
   October sessions now included (10-25 re-render; the 10-07 table had dropped 2 chronic sessions / animal): frozen
   chronic level / pre 1.04 / .85 / .83 / .90, flat 4 / 4; G still rising in 3 / 4 (mixed-model log-day slope +.030,
   z 4.5) but does not track behaviour (q .16) -- drift-sized per the June -> August group.
4. **Cortical position-coding gain follows behaviour, lagging ~1 week, not practice.** 3 / 4 (not PS94). **STANDS**
   (DECISIONS 10-19). Plateau by the end of imaging in every decode / encoder readout, 4 / 4 (STATUS 10-07, F18);
   the October decoder rerun after the `_day` fix (660bcbe) was DONE 10-25 (e3): verdicts unchanged.
5. **Acute map loss graded by eccentricity, far contra largest.** Rest-referenced, engagement-gated,
   cue window: far-contra acute amplitude / pre .50 / .45 / .53 / .40 (4 / 4); chronic .90 / .98 / .60 / 1.12 -- these
   per-animal numbers are the POSITION-WEIGHTED rest reference (15r RESTW; 5b found 10-25 that the page's "rest"
   figure and table were RESTW). Plain REST pooled gives the same picture: far contra .48, far middle .66, near ipsi
   1.43 acutely (RESTW .47 / .68 / 1.46). **STANDS**; the
   09-12 QUIET / MEAN amplitude tables in WHERE_THE_CODE_MOVES are **CORRECTED** (superseded by 15r).
6. **The acute far-contra map change has the same shape across animals** (between-animal r MEAN +.82, PRECUE +.86 vs a
   cross-position null ~.02-.08, p <= .001). **STANDS** acutely; chronic agreement is at its own null (DECISIONS
   09-17). Near-position "increases" under the MEAN reference are a coupling artefact (**STANDS**).
7. **Region level, acute.** Licks recruit more bilateral orofacial sensorimotor cortex (SSp-m / SSp-n lead, 4 / 4, all
   three references); at far contra the cue-evoked increment falls 1.7-3x more in the ipsilesional hemisphere (4 / 4).
   **STANDS** acutely; chronic region cells **PROVISIONAL** (the `15k` table predates the full chronic cohort).
8. **Representational geometry.** Crossnobis distance from each position's pre template: far contra +1.02 acutely;
   chronic +.30 to +.78 at all six positions. The still-RISING chronic distance is drift, not ongoing reorganisation
   (within-chronic rate not above each animal's pre drift floor, 4 / 4; F18). But chronic distance from the August
   template exceeds June -> August drift in 3 / 4 (Aug-vs-June / chronic-vs-Aug .63 / 1.01, .95 / 1.16, .43 / .43,
   .59 / .81; June -> Aug includes learning, so an upper bound). "Chronic diffuse = replacement" (PRELIM H4)
   **PROVISIONAL**.
9. **Rest carries spout position** (engagement-gated): observed / null 1.634, 43 / 44 sessions, 4 / 4; per-session
   decode 94 / 96; persistence, not anticipation (r_prev - r_next +.077, 4 / 4); a different code from the task
   (cosine .09 / .16). **STANDS** (REST_ENGAGEMENT_AUDIT §3; ungated predecessors 1.622 / 1.443 **CORRECTED**). "Rest shows
   the same acute dip" **WITHDRAWN** (DECISIONS 09-15).
10. **Pre-cue (ENL) position code: no consistent acute loss** (up PS92 / PS95, down PS93 / PS94; the slow band accounts
    for PS93's drop). Coding-direction leave-one-out: held-out far-contra pre-cue projection drops acutely 3 / 4 (not
    PS94), **PROVISIONAL**. Mixed model re-run 10-25 on the rebuilt attempts table: step at the stroke -.53 (z -3.0,
    q .019), but it NO LONGER tracks the time-to-contact deficit (q .25; was .023 on the old table; contact deficit
    q .050, borderline).
11. **Haemodynamics: CLOSED** (the data cannot answer it; no evidence of changed 415 / 470 coupling).
12. **The pre-stroke spatial template (simple position encoder, family 11): amplitude falls in 4 / 4 but returns in only
    2 / 4; its shape stays off in 4 / 4.** Per animal amplitude a pre -> chronic .96 -> 1.19, .96 -> .84, .93 -> .68,
    .91 -> .73; variance captured below pre chronically 4 / 4 (.45 / .38 / .47 / .24 vs .71 / .73 / .66 / .56). "Amplitude
    fully returns" (the pooled 09-10 three-animal number .749 -> .745) is **CORRECTED**; the shape claim **STANDS**.
13. **Session mixed models (re-run 10-25 on the rebuilt attempts table, `scripts/imaging_lmm.py`, both deficit
    definitions):** every decoder / template readout steps down at the stroke, recovers with log(days) and tracks the
    behavioural deficit session by session (frozen -.040 / unit, z -3.2; template amplitude z -8.7; best-match z -7.6).
    The decoder / template rows **STAND**. Position-coding TREND rows re-run 10-25 ~08:45 on d4's final-frame trend
    tables (107 August+ sessions): the ceiling (category - none) has NO acute step (-.014, z -1.0), rises with log-day
    (+.014, z 2.4, q .017) and is chronically above pre (+.038, z 5.0) in 3 / 4 (pre -> chronic .041 -> .108, .058 ->
    .108, .047 -> .076; PS94 flat .040 -> .037). Ceiling vs deficit is **DEFINITION-DEPENDENT**: significant against the
    contact deficit (-.050 / unit, z -3.9, q < .001) but NOT against time-to-contact (z -1.5, q .13) -- the 01:40
    "z -4.0, now significant" is **CORRECTED** (that run read stale trend tables). The per-spout SPREAD tracks the deficit
    under both definitions (z -4.5 ttc / -5.3 contact; pre-cue spread -4.2 / -4.6). Imaging page section 13.
14. **Map drift without a stroke (June pre -> August pre, 15r maps; `scripts/map_drift_prepre.py`, 10-25; imaging page
    §7, drift panels).** CUE window, REST reference, contact gate, 4 / 4 animals (12 June / 32 August sessions), ratio = amplitude vs the
    reference epoch, sig bins of 2022 (nested bootstrap, maxstat). Far contra: drift .95 (7 bins) vs acute .49 (1178
    bins) -> the acute far-contra loss (claim 5) is far beyond drift, **STANDS**. Far MIDDLE is the exception: drift .77
    (244 bins) is as large as its acute change .71 (151 bins), so the far-middle acute loss is **not separable from
    drift**, and the pooled CD leave-one-out far_center chronic -.51 (4 / 4; scored against August pre only) needs the
    same caveat. Far ipsi drift .76 (294 bins) vs acute 1.06 (6 bins). Near positions: drift .82-.94; stroke changes
    are increases (near ipsi acute 1.51, chronic 1.43), the opposite sign. June includes learning -> drift is an upper
    bound. LICK window, REST, LP-lick gate (the headline lick map, imaging page §7.3): far contra drift 1.04 (15 bins) vs
    acute .67 (500 bins), beyond drift; far middle drift .90 (12 bins, rim-suppressed) vs acute 1.07 -- no far-middle
    acute loss in the lick window, so the far-middle caveat is specific to the cue window; near positions drift
    .86-.98 vs acute increases 1.32-1.58. LICK, REST, contact gate: far contra drift 1.11, acute empty (contact gating
    leaves < 20 acute far-contra trials -- read the LP panel), far middle drift .93 vs acute 1.29, near drift .86-.98
    vs acute 1.32-1.58. LICK, PRECUE, contact: far contra drift 1.32 UP (the cue-evoked lick increment grew June ->
    August, consistent with learning; acute empty under contact), near / far-middle / far-ipsi drift .86-1.08 vs acute
    increases 1.34-1.92. CUE, PRECUE, contact: far contra drift 1.10 (33 bins) vs acute .18 (1,883), far middle .83
    (30) vs .55 (800) -- both beyond drift. PRE-CUE window (2 s), REST, contact: the far-contra pre-cue map FALLS with
    time alone (drift .67, 67 bins); acute goes UP 1.85 (339 bins), beyond drift; chronic .65 (0 bins) is drift-sized,
    so a chronic pre-cue decrease is not a stroke effect. Contact panels complete (CSV
    `map_drift_prepre/contact/drift_vs_stroke_panels.csv`).
    LP-lick gate (`map_drift_prepre/LP_licks/`): cue window as contact (far contra REST drift .94 / acute .49; PRECUE
    1.11 / .18). LICK, PRECUE, LP -- the only lick-window panel with an acute far-contra estimate under PRECUE: drift
    1.24 (72 bins) vs acute .56 (436), opposite signs, beyond drift. PRE-CUE 1 s (the LP default, `lp_licks.LP_ENL_WINDOW_S`; corrected 10-26 from "2 s"), REST, LP: far contra drift .65 /
    acute 1.84 / chronic .81 (0 bins) -- as contact. ENL 1 s, REST, contact (`DAQ_ENL1s/`): far contra drift .63 / acute
    2.21 (186 bins) / chronic .58 (0) -- the same picture at 1 s. **New, consistent across gates and windows: the
    chronic FAR-MIDDLE pre-cue map loss is beyond drift** -- chronic .53 (572 bins, 2 s contact), .30 (1,051, 1 s LP),
    .35 (954, 1 s) against drift .86-.97. Near-ipsi pre-cue ratios (2-3x, 0 bins) are unstable (near-zero reference
    amplitude), not results. All three arms complete 10-25. **PROVISIONAL** (June includes learning; June ~3 sessions
    per animal) until the remaining references / windows / arms render.

**3.Q Quiet-period cortical events (11-01; handoff `docs/status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md`, imaging
page section 15). PROVISIONAL, n = 4.** Quiet = no lick / running within 1.5 s, engaged; events = height > 4 x the
animal's pre robust SD, both polarities (k chosen to separate pre vs acute: acute contrast optimistic). (a) Face-regressed
ORofacial events ~3.8-4.1 /min pre -> 15.5 / 32.1 / 19.8 / 20.9 acute (4/4), LMM x5.1 / x2.5 / x1.4 acute / sub /
chronic; raw x4.1. (b) Pre 43-73 % of events coincide with face movement, acute at chance (4/4). (c) Power rises most at
2-3 Hz; quiet SD up over most of cortex, acutely except the lateral barrel / limb hotspots. (d) Acute orofacial events are
LOCAL to anterolateral orofacial cortex (ALM / tjM1 / SSp-m) bilaterally -- but this is MOSTLY SELECTION (T1, 11-01):
pre-stroke events without face movement are already local; the pre lateral spread came from movement events; acutely the
still events multiply x3-12 vs movement events ~x2; after movement + height matching only a small further narrowing remains
(4/4). Pre-stroke the same event type (same map) is rare in engaged quiet (0.1-0.5 /min) and x8-60 more frequent in the quit period; acutely it reaches 2-6 /min while engaged (T4; the rise persists at matched face stillness in every bin, 4/4, and is largest when the face is still -- T7: a stiller face is a minor part); bilateral (no ipsilesional bias in rate or amplitude; onset lag within +/-5 ms, which at 31 Hz
cannot exclude callosal spread; T2, T8), and not interictal (T11, T19). Running list:
docs/status/RUNNING_LIST_2026-11-01_QUIET_EVENTS.md. (e) Acute events decay ~18 % faster,
rise unchanged; the trace's ACF is unchanged (oscillatory component, not sharper bursts). (f) STATE: engaged-quiet face
speed roughly halves acutely in 4/4 (pre -> acute 1242 -> 812, 825 -> 492, 1947 -> 965, 1192 -> 593); and the
pre-stroke QUIT period already shows the same orofacial signature -- orofacial 1-4 Hz power 3.4-5.4x engaged quiet
while cortex-wide 1-4 Hz power is LOWER (0.6x) and the face stiller (sparse: 11 pre sessions with >= 60 s of quit quiet,
none in PS92). So the acute engaged state resembles the pre-stroke disengaged state in orofacial cortex, and it is not
cortex-wide slow-wave sleep. Limb also up after regression but acute only in PS93 / PS95. (g) **CORRECTED 11-01 23:10 -- the ipsilesional synchronisation is WITHDRAWN.** T20 found a quiet synchronisation index log10(PSD 1-4 / 8-15 Hz) rise (orofacial +0.59 acute) with ipsi > contra (+0.13, 4/4), but T22 shows the 8-15 Hz denominator is dominated by a NON-CALCIUM 7-11 Hz peak (equally in 415 nm and raw 470; cardiac / brain motion) that is lower on the lesioned side; with a fast band avoiding it the laterality holds in only 2/4 and limb shifts the same way. What STANDS: orofacial 1-4 Hz power rises acutely (415 unchanged in that band, T10) with NO ipsilesional bias (T8), and ipsilesional orofacial F0 does not drop (T22). Priya's reduced-thalamic-drive hypothesis therefore has no laterality support from these data (handoff section 7). (h) TASK WINDOWS (T12/T13, 11-01): the events also rise acutely in pre-cue and ITI
windows (4/4; pre-cue LMM +0.45, ~0.7x the quiet rise). A pre-cue event does NOT predict a failed far-contra attempt once
time-in-session is controlled (only PS95), but DOES go with a WEAKER frozen pre-stroke decoder position code on that
trial, only acute / subacute (-0.013, 4/4; hits only -0.020, 4/4; pre and chronic null) -- a candidate link between the
orofacial state change and the acute coding loss; NOT feature contamination: the frozen decoder (1.0-s window) rebuilt with the event frames removed from event and matched trials keeps the deficit (-0.016, 4/4), as do frames before the event and events ending before the window. (j) LP-LICK QUIET FRAMES (11-02; Priya: replace contact-only): quiet frames that also exclude LP video (non-contact) licks give a LARGER acute rise (face-regressed orofacial x6.7 vs x5.1; LMM +0.82 vs +0.71; 4/4) -- non-contact licks did not inflate it; chronic 1-4 Hz power / synchronisation become n.s.; outputs v2_quiet_lp/, now primary on the encoding page section 23. (k) T21: the larger acute lick response is mostly NOT accounted for by the higher pre-lick slow power -- at matched pre-lick power it is still larger in 3/4; extrapolating the pre-stroke within-session slope predicts only 14-22 % (causal power; 32-46 % with the leak-inflated zero-phase estimate), and that slope falls acutely, so even this is an upper bound reflecting a shared time course, not a mechanism. (l) LP RE-RUN of the remaining tests (11-02): task windows, T13, breathing, 415, lick LFO, composition, T21 all hold on LP licks; new: a pre-cue event predicts far-contra NO-ATTEMPT trials (+0.028, p .008, 4/4; DAQ masks +0.018, 3/4) but not incomplete attempts; cued-ITPC recovery link +0.35 (p .002, 3/4; composition: hits +0.27, without far contra +0.43); cued evoked 1-4 Hz amplitude acute n.s. (+0.06); T21 causal 15-21 %. (i) CHECKS (11-01): haemodynamics excluded -- 415 nm unchanged in SD / 1-4 Hz / event rate, no event-locked 415 response (T10); lick-locked 1-4 Hz ITPC unchanged acutely and evoked low-frequency amplitude up (T16: no Ramanathan-type task LFO loss; beyond the shared time course, cued-lick ITPC tracks far-contra recovery positively in 3/4, r +0.28 -- survives the composition control (n-matched; hits only +0.27; without far-contra +0.37, p .001; PS95 opposite) -- while quiet power runs negatively); every quiet measure returns toward pre with behavioural recovery (4/4) but none tracks behaviour beyond the shared time course. Breathing does not explain it (T9: coherence with a face-video respiration proxy low and unchanged; the cortical peak moves to ~3 Hz and does not track the proxy; regressing the proxy out leaves the 1-4 Hz and event-rate rise; weak-proxy caveat).

## 4. Encoding model and the goal question

*Owner d4; detail [ENCODING_FINDINGS.md](ENCODING_FINDINGS.md) (section numbers below), results page
https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (figure numbers as of 10-25). Frozen pre-stroke encoder = final frame
`gate-attempted_v7g3_t3d` (3-D tongue, attempted-trial gate, combined cam1 + cam4 licks). Within-animal changes, nested
bootstrap unless stated; LICK DATA = behaviour only.*

1. **The acute far-contralesional drop in cortex is carried by the LICK, not the target.** Frozen pre-stroke encoder
   decomposition: the acute drop = the predicted lick part in all model orderings, target part ~0 (<= 1.7 x1e3).
   Pooled: measured -.53 / -.30 of the pre response (ipsi / contra SSp-m), predicted -.62 (4 / 4). **STANDS** (3.4;
   Fig 8, 8b).
   **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** unchanged: predicted -.58 / -.61 vs measured -.30 / -.53 (SSp-m contra /
   ipsi, lick), target part ~0; FaceRhythm takes part of the lick share (page Fig 9-9b, 1).
2. **Chronically cortex responds MORE than the pre-stroke mapping predicts for the same behaviour (the residual).**
   Residual +.32 / +.43 of the pre response (ipsi / contra SSp-m) with predicted change ~0, 4 / 4. Pixel maps,
   calibrated bins (random pre splits within the nominal 5 %): all six spouts pooled, 32 px, lick window 19 / 35 / 30
   of 128 bins (acute / subacute / chronic), every one an INCREASE (413 up, 2 down across all tests), bilateral
   lateral frontal (ALM / tongue-jaw motor) + contralesional SSp; within animal 4 / 4 contralesional. By area (pre-SD
   units, chronic): MOp +1.8* / +2.3*, SSp-m +2.1* / +2.5*, SSp-n +.9 / +2.5*, MOs +.7 / +1.4*, contralesional >
   ipsilesional in every area; a raised plateau, not continuing drift (3 / 4). **STANDS**; the earlier "pooled pixel
   test weak" (8 px) is **CORRECTED** (8.4; Fig 9-9g, 10, 11, 11b).
   **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** NOT face movement: SSp-m chronic residual +9.4 / +9.4
   -> +9.5 / +10.6 (x1e3 dF/F, cue; lick +12.4 / +14.0 -> +10.9 / +13.4), 4/4 (page Fig 1, 1b).
   **SPECIFICITY + TIME COURSE (10-25, ENCODING_FINDINGS 8.11): PARTLY GLOBAL.** Non-orofacial comparison areas (visual /
   retrosplenial / trunk SSp; NOT a null -- movement, arousal and stroke effects reach them too) also carry a chronic residual in the lick window (+1.46 pre-SD, 3/4; cue +0.25 n.s.); orofacial - control
   cue +0.88 (3/4, borderline), lick n.s. The residual is largest at day 1,
   declines with log day, scales with the behavioural deficit, and persists at full recovery (cue orofacial - control
   +0.73 pre-SD, z 3.2). Read 2 as: an orofacial-weighted (cue) / partly global (lick) excess, mostly a disruption that
   shrinks with recovery, plus a persistent orofacial part. The deficit slope is carried by few sessions (PS94; PS92
   opposite), so "tracks the deficit" is not 4/4; persistence at recovery is the robust part. NOT an extrapolation
   artefact: on trials inside each animal's pre-stroke behavioural envelope (3-D endpoint, reach, speed, latency, lick
   count; 93-96 % of chronic trials in PS92 / PS93, 72-78 % in PS94 / PS95) the residual is the same (chronic SSp-m lick
   +5.5 / +5.6 x1e3, 4/4). Same in the no-FR frame (cue orofacial - control chronic +1.33 [0.17, 3.09], 4/4; at
   recovery +1.09 pre-SD, z 4.0). PROVISIONAL (n = 4; deficit slope animal-dependent).
3. **The behaviour-to-cortex mapping is rewritten chronically (Layer 2).** One fit per animal, epoch kernel changes from
   early-August pre: chronic gain beats the late-pre null (cortex +54.8 [+22.0, +87.4], SSp-m +52.5 / +51.6) and every
   one of 20 random pre pseudo-epochs in 4 / 4 (cortex and both SSp-m); grows with time; spread across kernel groups
   (none alone beyond the null). **STANDS** within August; its nulls do not include June -> August drift (cf. 3.3).
   The original pooled-refit version was **WITHDRAWN** (10-24; 8.5; Fig 12-13b).
   **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** unchanged: chronic change beyond pre drift cortex
   +0.055 -> +0.054, SSp-m +0.052 -> +0.059 / +0.061 (4/4). The FR direction weights change no more than
   pre-stroke drift (page Fig 1 panel 4).
   **SPECIFICITY + TIME COURSE (10-25, ENCODING_FINDINGS 8.11): CORRECTED -> NOT CONCENTRATED IN OROFACIAL CORTEX.** Non-orofacial comparison areas
   change as much chronically (+0.052 vs +0.053; contrast +0.002 [-0.017, 0.018]); every area is beyond pre_late. In
   control areas there is no step at the stroke but a steady rise with log day (z 6.1), larger in later / better-behaving
   sessions: a cortex-wide change -- a real stroke effect (diaschisis, neuromodulation / state) or longer-time-scale drift;
   the 8-day August null cannot separate them. Orofacial-SPECIFIC part = an ACUTE orofacial excess (+0.025, z 2.6) that
   decays. "The orofacial motor mapping is rewritten chronically" is withdrawn as an orofacial-specific claim. Same in the no-FR frame (contrast -0.005
   [-0.036, 0.019]).
4. **Licks to far contra go to the wrong spout from launch (LICK DATA).** Landing soft bins: own bin -.69 / -.38 / -.19,
   far mid +.35 / +.36 / +.20 (4 / 4, significant); whole tracked 3-D path: own -.65 / -.36 / -.22, acute toward far
   ipsi +.42 (part general forward overshoot) then far mid; mis-landing licks follow the other spout's course from the
   first tracked point. **STANDS** (2.4b, 8.1; Fig 3-3e, 5, 5b).
5. **On average the frozen pre-stroke DECODER reads far-contra trials as the spout the tongue goes to.** Own -.66 /
   -.30 / -.22, far mid +.26 / +.14 / +.11, far ipsi +.19 acute (4 / 4; cue window the same). But acutely the decoder's
   diagonal falls for EVERY spout while those licks land on target (general weakening). **STANDS** (8.2; Fig 3f, 3g).
6. **Trial by trial, cortex covaries with execution on HITS; on FAILED ATTEMPTS it does not** (REVISED 10-25 night,
   `scripts/coupling_by_class.py`, page Figs 4i-4i4). Within session, neural (decoder P far mid - P far contra) tracks
   executed (lick weight far mid - far contra) on far-contra trials. Pre-stroke (hits only, August pre) r .14-.35 landing /
   .18-.41 trajectory, 4 / 4. Split by class, change vs pre-stroke hits (pooled r, nested bootstrap animals -> sessions):
   FAILED attempts acute -0.26 [-0.42, -0.15] landing / -0.34 trajectory, significant in each of the 4 animals; subacute
   -0.29 / -0.40; HITS acute -0.02 [-0.43, +0.44] (64 trials, PS92 +0.43, PS93 / PS94 negative, PS95 1 hit -- uncertain),
   subacute -0.10 / -0.12, chronic -0.07 / -0.06 (n.s.). The earlier "acutely r ~ 0" (all trials) was mostly a
   composition effect: 92 % of acute far-contra trials are failed attempts and the pre reference is hits only. Not a
   range restriction (executed spread is LARGER acutely). Statistic is the class-matched change across epochs; the
   per-session null is block permutation (runs of consecutive far-contra trials; trial shuffle gave the same calls).
   Candidate readings: contact feedback (no touch, no location signal) vs execution quality; pre-stroke failed attempts
   (not classified) are the missing reference. Executed-bin vs decoder matrix: acute diagonal -.41 (far mid), -.34
   (far ipsi), 4 / 4. **REVISED** (8.2; Fig 3h-3j; page 4i-4i4).
6b. **What acute far-contra trials look like to the frozen decoder** (10-25 night): not "nothing" -- each trial still
   gets a confident posterior (max P ~.78-.82, entropy .26-.31 of flat; pre .95 / .05-.10) -- but mostly NOT far contra,
   and the wrong spout differs by animal (lick window, mean P): PS92 spread (far mid .24, close ipsi / mid .20, far contra
   .14); PS93 far ipsi .28; PS94 far contra .30 / far mid .29; PS95 close contra .28. Hits lean far contra (PS93 .42, PS94
   .63); failed attempts drift to a neighbouring far spout (PS92 far mid .31, PS93 far ipsi .39, PS95 far mid .46).
   Caveat: a logistic decoder can be confident off-distribution; the template-similarity check (does an acute trial
   match any pre spout pattern better than pre trials match their own) is not run. **PROVISIONAL** (no page figure yet).
7. **When a lick misses, cortex mostly still looks like the TARGET, except at the far spouts.** Off-target trials,
   decoder = target vs = executed: close spouts .65-.95 vs ~.05 at every epoch; far contra .66 / .21 pre -> .25 / .29
   acute (executed wins only here) -> .45 / .33 chronic; far ipsi / far mid split acutely. **STANDS** (8.2; Fig 3k, 3l).
8. **Encoder relabel test: no single answer for "which spout is cortex in".** Best-fitting label (target + spout-locked
   pattern swapped, executed licks kept): far contra loses its label toward far mid / close contra but other spouts
   lose theirs too (close contra -.58 acute, far mid -.44 chronic); per animal PS95 re-aimed at far mid (transient),
   PS93 stays far contra, PS92 / PS94 nonspecific. TRIAL LEVEL (10-25, Fig 3o-3q): far contra loses its own label in
   every epoch (-.14 / -.17 / -.14, 4/4) but far ipsi as much (-.23 / -.21 / -.17), and trials at EVERY spout drift to the
   close-mid label (close ipsi +.20 acute, 4/4): a label-wide bias, not far contra re-aimed; on off-target far-contra
   trials the encoder reads target ~.12-.15, executed ~.17-.23, another spout ~.65 (unlike the decoder, 8.2).
   **PROVISIONAL** (2.7, 8.3; Fig 3m-3q).
9. **Failed attempts lose the side but keep the distance.** Frozen decoder on video attempt classes: acute failed
   (incomplete) far-contra attempts are read as A far spout (distance kept, side lost; 4 / 4, lick window); no
   pre-cue plan signature; retained code ~.1 of pre-stroke hits at every epoch while hits recover to .71. **STANDS**
   (2.8; Fig 6-6c). RECHECK (10-25 night): the pre-cue far-contra P(true) is lower on failed attempts than on hits in
   3 / 4 acutely (.14 vs .21, .17 vs .23, .30 vs .40, .20 vs .20) and 4 / 4 chronically -- "no pre-cue signature" needs a
   class-matched test (time-in-session and movement matched); the pre-cue code is weak per trial (pre hits .24-.51,
   chance .17), so only class averages are readable.
10. **Failed attempts look like weaker hits, not like no-movement.** Same bilateral pattern; lower amplitude than hits
    (3 / 4) but not than touch-matched pre-stroke misses (2 / 4); ipsilesional ALM / tongue-jaw motor down acutely,
    tongue-jaw motor / SSp up chronically. **PROVISIONAL** (2.9; Fig 7-7c).
11. **Target vs executed** (HISTORY + current reading in synthesis point 4; the original heading "chronically cortex carries
    the target beyond the executed movement" holds only in the per-session refit frame and is now read as remapping of
    BOTH target and executed coding). Binned contest (unique
    R2 of target bins beyond executed bins): chronic change +9.0 [+3.6, +16.0] lick, +9.8 [+3.9, +16.2] cue, at every
    softmax width; acute lick +12.5 (n.s.; PS93 opposite). Pre-stroke "cortex tracks target, not executed" remains
    **SUSPENDED** (2.4). **STANDS** chronically (2.4b, 8.6; Fig 1-2b) -- in the PER-SESSION REFIT frame only.
    **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** **REVISED.** Frozen target-unique R2 (the spout beyond every executed
    regressor) FALLS acutely and stays DOWN chronically (cortex lick -0.016 [-0.028, -0.004], 0/4 up; with FR -0.008,
    0/4); the executed movement's spout-locked part RISES acutely (cue cortex +0.041, 4/4) and stays slightly up with FR.
    The refit chronic target gain is not reproduced when the weights are frozen (page Fig 1e, 1f). CAVEAT (10-25): the refit and frozen numbers are SCORED differently (refit: cue 0-1 s, lick +/-0.2 s around EVERY lick peak, prediction offset penalised; frozen: cue 0-1.5 s, lick -0.1..+1.0 s around the FIRST lick, data and prediction centred separately), so "not reproduced" mixes frame and scoring; the per-session self-fit in the frozen windows (`scripts/session_vs_frozen.py`, running) separates them.
    **RESOLVED (10-25, self-fit, 107 sessions): a FRAME effect, not scoring.** Each session refit on itself in the frozen
    design + windows: chronic target-unique R2 RISES (SSp-m +0.017 / +0.019, cortex +0.010 / +0.013, 3-4/4) while the frozen
    model's falls (cortex -0.006 / -0.008, 0/4). -> Chronically cortex still carries the target beyond the executed
    movement, but in a NEW pattern that a session-specific mapping reads and the pre-stroke mapping does not (target
    coding remapped). Full-model self - frozen gap rises acutely (SSp-m +0.07-0.12) and chronically (+0.06-0.09), 4/4;
    self spout-unique contra SSp-m +0.054 / +0.059 chronic 4/4. PROVISIONAL (different-size training sets; change from
    pre only).
    **PER-SPOUT LICK KERNELS (10-25 night, frozen + FR + FROZEN_LICKSPOUT, run `..._frA_ls`, ENCODING_FINDINGS 8.16):**
    pre unique R2 contra SSp-m target .015 / .019 vs per-spout lick kernels .012 / .015 (cue / lick); ipsi SSp-m .006 / .008
    vs .013 / .014; lick kernels absorb part of target (contra SSp-m .020 -> .015 cue). Epoch - pre (frozen): target acute
    -.008 / -.014 (1/4 up), chronic +.007 / +.002 (2/4); lick kernels acute -.007, chronic -.007 / -.010 (0-1/4); executed
    spout-locked templates acute +.023 / +.024, chronic +.008 / +.007 (4/4). PROVISIONAL.
    **WHICH PART (10-25 night, partial refit, `scripts/partial_refit.py`, page Figs 1yd / 1ye; nulls = 5-draw BLOCK
    PERMUTATIONS of same-spout runs (10-26; spouts come in ~6-trial runs, so one half-session shift was a noisy null).
    The permutation nulls reproduce the single-shift numbers: chronic contra SSp-m target - null +0.036 cue / +0.034 lick,
    executed - null +0.055 / +0.064, acute ipsi SSp-m target - null +0.039 / +0.041, all 4/4; acute otherwise CIs include 0.
    Single-shift numbers below):** per session refit ONLY
    the target or ONLY the executed-position columns (rest frozen, 5-block CV), each read against its own null (the same
    columns rebuilt with spout labels shifted by half the session). The nulls ALSO gain post-stroke (any refit
    spout-structured regressor absorbs part of the larger residual), so raw refit gains overstate specificity. Acute:
    mostly NON-SPECIFIC (target - null, executed - null ~0 in contra SSp-m and cortex-wide; the acute "executed beats
    target" is matched by the executed null) except a target-specific gain in IPSILESIONAL SSp-m (+0.037, 4/4 cue, 3/4
    lick). Chronic: BOTH specific in contralesional SSp-m (target - null +0.034 cue / +0.032 lick, executed - null
    +0.054 / +0.064, 4/4) and executed - null cortex-wide (+0.038 / +0.045, 4/4; target - null +0.022, 3/4). The two
    were not tested against each other. PROVISIONAL.
12. **Face movement (FaceRhythm) explains about half of the spout-position code and most of its chronic rise.**
    **FROZEN + FaceRhythm (10-25, ENCODING_FINDINGS 8.10):** the post-stroke CHANGE claims below were per-session refit; in the
    frozen frame: (a) contralesional position | FR rise NOT reproduced (position unique flat; contra - ipsi -0.010
    without FR) -- the SPOUT package (target + executed spout part) shifts contralesionally instead, contra - ipsi
    +0.039 [0.011, 0.081] cue / +0.048 [0.015, 0.094] lick with FR, 4/4 (page Fig 1c, 1d); (same scoring caveat as 11); (b) the spout code is a PETH
    code and the acute loss is a PETH loss (STANDS), but it does NOT recover chronically (PETH target unique cortex lick
    -0.096 without / -0.054 with FR, 0/4; Fig 1g, 1h); (c) FR direction unique in SSp-m falls chronically, but the
    direction weights do not change beyond drift (Layer 2): less direction signal through the same mapping, MIXED
    behaviour / neural; (d) recovery course: frozen spout unique pre -> chronic up 2/4, cortex reaches half its change (weak: within session spacing) at
    or after behaviour (Fig 1i); (e) movement WITHOUT lick-position information (pose + FR) gains unique R2 in SSp-m
    chronically (ipsi +0.050 [0.023, 0.074] 4/4), position-carrying movement alone falls contralesionally (quick run,
    Fig 1j, 1k). PROVISIONAL. Refit statements kept below for the record:
    Per-session encoder with FR (master_pre_v1, rank 12): unique position R2 pre-stroke 52 -> 30 x1e3 (cortex, lick);
    chronic rise +22 [+7, +38] -> +4 (n.s.) with FR (cue +39 -> +15 [+3, +27]); the acute ipsilesional SSp-m position
    loss survives FR; FR's side / distance information adds little beyond its lick amount (modes A ~ B ~ S ~ D).
    Musall-style with FR (Fig 16d-16h): FaceRhythm alone ~310-380 x1e3 (cortex), ~120-155 unique; it overlaps mostly
    with UNINSTRUCTED movement (that group's unique R2 halves, 241 -> 109 pre, lick window); in the lick window FR's
    unique R2 rises acutely (135 -> 156) while every other group's falls. Pixel maps: the chronic rise in position
    unique is broad without FR, focal contralesional anterior-lateral with FR. TESTS (10-25, Fig 16i-16k, R2 as a
    fraction): pixel test (16 / 32 px, calibrated 5 % false-bin rate) finds NO chronic bins for position unique | FR (69
    without FR), 13 up for task unique | FR; a priori AREA test: chronic position unique | FR in CONTRALESIONAL SSp-m cue
    +0.037 [0.010, 0.067] 4/4 (p_bonf .002), lick +0.025 3/4 (n.s. after correction), ipsilesional ~0; contra - ipsi
    +0.025 / +0.032 (CIs exclude 0); task the same; uninstructed ALONE rises in SSp-m on both sides (4/4). Beyond pre drift
    in PS94 / PS95 only (PS92 marginal, PS93 not); not a denominator effect (total variance rose).
    TRIAL vs PETH (8.9): FR explains trial-to-trial variability, the spout code lives in the PETH; the ACUTE spout loss is a PETH loss.
    DIRECTION (8.9): FR's side / distance factors explain less SSp-m acutely and chronically (0/4 up); the contralesional rise
    is present with directionless FR (cue +0.028, 4/4). Total variance rises (global). Frame = per-session refit encoder.
    WHY (8.9, model-free, Fig 16p): acutely the face's spout signal weakens (side eta2 -0.10, tongue -0.12) and FR tracks the
    tongue less (behaviour + measurement); chronically those are back at pre, but within-spout side movement is -21 % (3/4)
    and ipsilesional SSp-m loses its coupling with the FR side index (partial r -0.10 -> ~-0.02, 4/4), contra - ipsi -0.07
    [-0.11, -0.03] 4/4 -> chronic decoupling is CORTICAL (ipsilesional), not a measurement failure. Small r (<= 0.2). PROVISIONAL.
    **PROVISIONAL** (FR factors owned by e3; position beyond FR is a lower bound; carried by PS94 / PS95) (8.7, 8.9; Fig 16-16q).
12b. **(10-25 evening, ENCODING_FINDINGS 8.13) The post-stroke change lives in the trial-averaged (PETH) response;
    trial-to-trial movement coupling is preserved** (frozen frame, Allen-area PETH vs trial-by-trial maps, unique and
    Musall-layout alone models; page Figs 1t-1uf). **STANDS** at area level (pixel version queued). **Caveat for every
    mean target map:** PS94's target-unique R2 is ~10x the other animals' (cue PETH contra SSp-m .37 vs .004-.029), so mean
    maps are PS94-weighted (tests count animals); cause under test (block-aligned state vs tracking vs tuning).
12c. **(10-30) MOVEMENT-NULL PLAN MAPS (FINDINGS 0b item (5), built; Musall unique, no refit).** The target (spout
    identity) entered LAST after every movement group of the frozen encoder (non-directional lick vigour / timing, nose,
    running, spout-locked lick template, within-spout tongue / jaw direction, the upcoming lick's angle via lick kernels
    -0.5..+2 s, and in FR-A all FaceRhythm factors). Pixel maps from the existing FROZEN_PIX_MUSALL basis twins (107
    sessions, frame R2, cue 0-1.5 s / lick -0.1-1.0 s). NO lick-history regressor exists in this model (not added).
    Pre level is SMALL (contra SSp-n 0.015 cue / 0.021 lick, ALM 0.010 / 0.012, tjM1 0.013 / 0.017, FR-A) and PS94-heavy
    (PS94 0.04-0.07 vs <= 0.015 in the others). Change: only DECREASES, no bin increases anywhere -- FR-A target cue
    63 / 11 / 89 of 514 16-px bins down (acute / subacute / chronic), lick 81 / 7 / 91; noFR 123 / 45 / 111 and 367 /
    44 / 140; spout unique (target + executed spout part) 0-21 bins down, none chronic. By area (FR-A, contra): ALM cue
    -0.013 acute / -0.009 chronic (0/4 up, CIs exclude 0), SSp-n lick -0.018 acute (0/4) / -0.011 chronic (1/4, ns),
    tjM1 lick -0.012 acute (0/4) / -0.007 chronic (ns); SSp-m chronic ~0 (+0.005 cue, 3/4). CAVEAT: the CONTROL RSPd
    also falls chronically (cue -0.006, lick -0.008, 0/4 up) and whole cortex too -> the drop is NOT orofacial-specific;
    read it as "the plan term beyond movement does not RE-EMERGE after the stroke", not as a localised loss. Subacute
    is closest to pre (dip-recover-dip pattern). Script `scripts/movement_null_plan_maps.py` (outputs
    `<E>/movement_null_plan/`). **PROVISIONAL** (n = 4; PS94-weighted mean maps).
    TASK-A COMPANION (10-30): SSp-n / ALM / tjM1 now in the frozen Musall table (`scripts/frozen_fr_musall_v2.py` ->
    musall_change_v2.csv; Figs 0.2 / 0.8 v2 = `frozen_musall_<w>_simple_v2.png`): chronic spout-unique contralesional
    shift (contra - ipsi, FR-A) holds in tjM1 (cue +0.012 [0.001, 0.023], lick +0.020 [0.004, 0.037], 4/4) as in SSp-m;
    not detectable in SSp-n (+0.008 / +0.014, 3/4, ns) or ALM (~0). **PROVISIONAL.**
13. **Not established / withdrawn here:** pre-stroke target-over-executed lead (SUSPENDED, 2.4); pooled toward-midline
    within-spout effect (NOT ESTABLISHED, 10-21); misdirected licks carry a plan signature (NOT ESTABLISHED, 2.5);
    lick-level strays decide target vs execution (NOT DECISIVE: reference fails validation, 2.6); Layer 2 pooled-refit
    claim (WITHDRAWN 10-24).
14. **CURBD pilot (10-30): region -> region currents fitted to encoding residuals are NOT STABLE, so no pre vs chronic
    comparison was made.** PS94 only (n = 1 animal), pre 0807 / 0811 / 0812; 18 regions (SSp-n, SSp-m, SSs, SSp-ul,
    tjM1, ALM, MOp_rest, MOs_rest, RSPd x 2 hemispheres; ROI v8 on the stat mask) x 4 spatial units; input = held-out
    residuals of the per-session encoder's full design refitted on the unit traces, 10.4 Hz, reset to data every 2 s.
    Fit: pVar 0.27 train / 0.21-0.25 held-out half, against 0.15 for a J = 0 decay model and < 0 for hold-at-reset.
    Stability of the off-diagonal mean |current| matrix (gate stated beforehand in
    `docs/status/CURBD_PILOT_PREREG_20261030.md`): seeds r 0.51 (needed 0.8), halves 0.38 (0.7), sessions 0.29 (0.7),
    pair-specific halves 0.06 (0.5); the one-unit-per-region variant fails the same way (0.51 / 0.22 / 0.40). A weaker
    initial J (g 0-1, exploratory) does not fix the halves (0.18-0.24). EXPLORATORY comparator: a heavily ridged VAR(1)
    on the same data gives region matrices with r 0.84-0.97 across halves and sessions, but the matrix depends on the
    ridge strength (weak ridge vs heavy: r 0.07-0.21) and explains ~6 % of the one-step change. CURBD's chronic session
    (0910) was prepped but NOT fitted. Code `scripts/curbd_pilot.py`, `wfield_local/curbd_core.py` (port of
    rajanlab/CURBD, **GPL-3.0**); outputs `<E>/curbd_pilot/` (figures 1-3, tables/). **NOT ESTABLISHED / PILOT.**
15. **Region-level ridge MVAR (10-30, follow-up of 14): cross-region terms predict held-out, but the region matrix is
    NOT stable enough in enough animals (gate 2/4, needed 3), so NO pre vs post comparison was made.** Same 18-region
    residual data as 14 (one unit per region), 31 August pre sessions, all four animals. Lag 1-3 bins and ridge chosen
    by nested contiguous-block CV (whole trials, no lag crosses a trial); readout W = lag-summed coefficients.
    Prediction: full MVAR beats each region's own past (AR-only, own CV) in 31/31 sessions and 5/5 held-out folds;
    gain .010-.025 R2 = 9-15 % of the one-step CHANGE variance (median per animal). Stability (gate written beforehand,
    `docs/status/MVAR_PREREG_20261030.md`): PS92 and PS94 pass every criterion; PS93 fails only on one session
    (0814 halves r .28; median .87, sessions .84); PS95 fails (halves median .64, sessions .39). Global-removed and
    deconvolved (tau 0.6) variants: 1/4 and 2/4. CV picks p = 3 (top of range) and near-OLS ridge everywhere; the
    ridge-robustness criterion therefore tested a flat region; a heavy ridge (as in 14's comparator) gives a different
    matrix (r .25-.37). The S18 Granger result (`coupling_vigour_sm.py`) is NOT tested or supported by this. Script
    `scripts/mvar_regions.py`; outputs `<E>/mvar/` (figures 1-2, tables/). **NOT ESTABLISHED (gate failed; PROVISIONAL
    n = 4).**
16. **Pooled sensory / motor MVAR (10-30, follow-up of 15): FAILED its gate on raw (0/4), so again NO pre vs post.**
    Six nodes (sensory, motor, RSPd x hemisphere), lags up to 8 bins (~0.8 s), same data and CV as 15, gate written
    beforehand (`docs/status/MVAR_POOLED_PREREG_20261030.md`). Predicts held-out in 31/31 sessions (5-11 % of change
    variance) but the 6 x 6 matrix is less reproducible than the 18-region one (halves median .57-.87, worst sessions
    .26-.32 in three animals). Global-removed variant passes 4/4 (sensitivity only, not gated). CV again chooses the
    longest lag allowed (p = 8) everywhere, and one session's S->M estimate has split-half noise larger than the spread
    between sessions. Directed-influence models of these area traces are not usable at the session level; the S18
    Granger / xcorr result stands alone. Script `scripts/mvar_pooled.py`; outputs `<E>/mvar_pooled/`. **NOT ESTABLISHED
    (gate failed; PROVISIONAL n = 4).**
17. **Sensory-motor coherence vs CCA on cued first licks (10-30; Guo 2021 / Heimbuch 2026 style; pre-registered,
    `docs/status/COHERENCE_CCA_PREREG_20261030.md`).** 8 sensory vs 8 motor areas (both hemispheres), -1..+1 s.
    **CCA fails the gate (1/4)**: held-out CC1 ~0.98 (CC2 ~0.96) in every session, the S and M projections are nearly the
    same trace (a cortex-wide shared component, not a specific S-M channel), and one session's CC1 is mostly estimation
    noise (split-half SD > between-session SD in 3/4). So no CCA / stereotypy comparison and **the DLAG criterion is not
    met**. **Coherence / phase locking pass (4/4, delta coherence 3/4).** Raw S-M coherence stays ~0.9 pre to post;
    coherence beyond the shuffled-event null falls in the 0.5-1.5 Hz band (subacute -0.074, chronic -0.087, 4/4) because
    the null rises, i.e. more of the coherence is the shared lick-locked response; lick-locked phase consistency (ITPC)
    rises from subacute in both groups (sensory 1.5-4 Hz +0.11 / +0.17, others +0.04-0.10, mostly 4/4); acute unchanged;
    ITI docked licks unchanged. Reading: post-stroke cued-lick responses are MORE phase-locked to the lick, without extra
    trial-specific S-M coherence -- not a Guo-style coherence loss. Recovery tracking 5/24 tests p < .05 (1.2 expected;
    e.g. sensory delta ITPC vs far-contra hit rate partial r +0.34, 4/4): hints. Script `scripts/sm_coherence_cca.py`;
    outputs `iti_licks/sm_coherence_cca/`. Controls (addendum): movement regression leaves the 1.5-4 Hz ITPC rise
    intact (remaining share ~1.1), behavioural variability does not explain it (and does not clearly fall); response SNR
    (near-restatement) absorbs ~half the sensory and most of the motor rise -> a larger lick response relative to trial noise,
    not more stereotyped licking. Page 14.18 S19 (demoted 10-30). **PROVISIONAL (n = 4).**
    **CAVEAT (Priya 10-30, GCaMP6s kinetics): treat the phase measures as an amplitude / consistency readout of the
    lick-evoked calcium transient, not as oscillatory phase-locking (no Ramanathan / Khanna LFO analogue).** With
    tau ~0.6 s the indicator passes ~17 % (1.5 Hz) to ~7 % (4 Hz) of the underlying amplitude; in a +-1 s Hann window the
    1.5-4 Hz content is mostly the transient's onset edge, so the ITPC rise mostly restates the larger cued lick
    response (section 0b) relative to non-calcium noise. The pre-registered deconvolved sensitivity (tau 0.6) reproduces
    every number (e.g. sensory 1.5-4 Hz chronic +.18 / +.22 / +.22 / +.04 raw vs +.17 / +.22 / +.23 / +.06), but that is
    NOT a test: an exponential deconvolution is a linear filter that multiplies every event's Fourier coefficient by the
    same factor, so ITPC and coherence are invariant to it by construction (up to edge effects). Not a coupling result.
18. **Sensory vs motor timing of lick-ANGLE information (10-30; Lemke 2024 style; pre-registered,
    `docs/status/DIRECTION_INFO_LAG_PREREG_20261030.md`).** Per 0.05-s bin, CV ridge decode of the executed first-lick angle
    from 8 sensory vs 8 motor areas; half-max lag dL50 = L50_M - L50_S (+ = sensory first) and an amplitude-invariant
    curve-shape lag. Gate 3/4 (PS92 noisy halves). **Pre-stroke, motor information leads sensory by ~23 ms (4/4, both
    readouts).** Post-stroke the lead shrinks (dL50 +29 / +11 / +14 ms acute / sub / chronic, 4/4 each), but the shape
    lag confirms it only chronically (+12 ms, 4/4) -> **chronic motor lead ~23 -> ~10 ms**, mainly because sensory
    information arrives earlier (L50_S -31 to -38 ms every epoch, 4/4). Peak information rises in both groups
    chronically (+.07). Within-spout angle, hemisphere pairings and ITI docked fail the gate (not read). Caveats:
    sub-bin effects; calcium / haemodynamic timing; total angle includes the target; jaw leads tongue by 52-120 ms
    (movement-locked activity may set the motor-first crossing); information timing, not influence. Page 14.18 S20.
    **PROVISIONAL (n = 4).**
19. **Restitution vs compensation: kinematic normality of successful licks (10-30; SRRR / Levin 2009; pre-registered,
    `docs/status/KINEMATIC_NORMALITY_PREREG_20261030.md`; `scripts/kinematic_normality.py` -> `kinematic_normality/`).**
    Per hit trial, first post-cue lick, Mahalanobis distance (Ledoit-Wolf) from the animal's August-pre licks at the same
    spout on 7-9 features (lick angle at the extension peak, sagittal angle, reach, peak speed, landing lateral error, closest approach, latency,
    jaw onset lead, jaw lateral steering at side spouts); normality = fraction of hits inside the pre 95 % contour (pre
    scored leave-one-session-out: .89-.92). **Chronically hit rate is back (+0.03, 4/4) but successful licks are still
    less normal: -0.10 [-0.22, -0.03], 4/4** (PS92 -.03, PS93 -.05, PS94 -.27, PS95 -.06; far contra -0.10, 4/4) ->
    recovery = fast but INCOMPLETE restitution + compensation, mostly PS94. Normality recovers about as fast as hit rate
    (half-times PS93 3.6 vs 4.7 d, PS94 2.7 vs 4.1 d; PS92 slower 8.8 vs 4.9) but levels off below pre (asymptote 0.99 /
    0.87 / 0.56 / unresolved) -- not the expected slow kinematic course. Cortical readouts: the frozen far-contra code on
    hits follows task SUCCESS session to session (partial r +0.46, p .001, 3/4) while plateauing at 0.3-0.7 of pre; cued
    balance leans toward normality (PS94 asymptote 0.75 vs normality 0.56; 2/4, undetermined); ITI offset and contra tjM1
    response undetermined. No readout passes as following normality. The pre-vs-post classifier separates every post
    session (incl. drift) -> not used. Caveats: no no-stroke drift reference; few acute far-contra hits; permutation p
    ignores autocorrelation. WHICH features (10-30, `scripts/kn_hits_deviation.py`, page 14.18 S22c; z vs own pre hits at
    the spout, lateral + = contralesional): chronic hits at all spouts still land IPSILESIONALLY (lick angle -0.89 SD,
    landing lateral error -1.00 SD, 4/4) and are faster / longer (latency -0.34, speed +0.15, reach +0.39 SD, 4/4); far
    contra: closest approach +0.70 SD, jaw lateral -1.78 SD acutely only -> residual aim deficit compensated by vigour
    and reach. **PROVISIONAL (n = 4).**
20. **Re-embedding test: is the chronic spout code the old code re-embedded, or a new code? (10-30; Gallego 2020 /
    Degenhart 2020; pre-registered, `docs/status/REEMBEDDING_PREREG_20261030.md`; `scripts/reembedding_test.py` ->
    `reembedding/`). UNINFORMATIVE: the alignment failed its gate.** Spout decoding (6 classes) from k = 10 PCs of 45
    areas, cued first licks matched to the pre kinematic range; (i) frozen pre decoder, (ii) frozen decoder after a
    label-free CCA alignment on spout-pooled time-locked trajectories, (iii) same-session refit. **Gate G1 failed**: on
    pre -> pre the alignment recovers only 0.29-0.45 of the frozen performance (needed >= 0.7), because spout-pooled
    trajectories pin the shared dimensions but not the spout dimensions. So the index cannot separate re-embedding from a
    new code. Descriptive only: the frozen-to-refit gap grows chronically (+0.18, 4/4; PS95 small) and the alignment
    closes none of it (index 0.12 [-0.33, 0.58]); the June no-stroke reference (~60 d before pre) also opens a gap
    (+0.076, 4/4), so roughly half of the chronic gap is drift-sized. Next (new prereg): an alignment with spout-resolving
    but target-free correspondence (kinematics-binned or ITI trajectories, or stable-area anchors) that passes G1 first.
    Frozen references in the local store only (kind `reembed`; publishing = Priya's call). **NOT ESTABLISHED.**
21. **Within-lick corrections do NOT carry recovery (10-30; Bollu 2021 style; pre-registered,
    `docs/status/WITHIN_LICK_CORRECTION_PREREG_20261030.md`; `scripts/within_lick_correction.py` ->
    `within_lick_correction/`).** 250 Hz 3-D tongue (outward phase ~7-8 frames): launch angle (first 12 ms) vs endpoint
    (extension peak), each vs the animal's pre hit pattern; corrective submovements (CSMs) = extra speed peaks above a
    noise-calibrated 50 mm/s threshold (false rate 5 % vs real 15.6 % pre). Pre-stroke the Bollu signatures hold (CSMs
    .28-.33 at far side spouts vs .04-.08 close; cued .16 vs ITI .10). Far contra, cued: LAUNCH error -25.7 deg acute,
    -7.0 subacute (4/4, Holm), chronic -3.2 n.s.; the ENDPOINT follows the launch (chronic -3.5, 4/4); within-lick
    correction unchanged in every epoch (chronic -0.4 [-3.1, +2.4]); CSM share FALLS (-0.13 / -0.15 / -0.14, 3-4/4; not
    noise, not only slower licks) and missed later licks lose their pre-stroke in-lick re-aim (+10.9 deg pre; -8.0 acute,
    -5.0 subacute, 4/4). -> **Recovery is restored LAUNCH aim, not added correction; licks become simpler, not more
    corrected.** Neural: the larger acute lick response is the same on clean licks (+21 vs +18 all licks; carried in 0 /
    168 cells); CSM licks evoke LESS than matched CSM-free licks post-stroke (-10.7 chronic, 4/4; not in deconvolved).
    Caveats: 250 Hz resolves only >= ~20 ms submovements; launch noise 5-8 deg per lick. **PROVISIONAL (n = 4).**
22. **"Learned, context-specific compensation" NOT supported by behaviour (10-30; pre-registered,
    `docs/status/LEARNED_COMPENSATION_PREREG_20261030.md`; `scripts/learned_compensation.py` -> `learned_compensation/`).**
    T1 (post days 1-14, far contra): no within-session improvement of launch error (W -1.6 [-5.1, +2.1] deg) and no
    overnight loss; gains come BETWEEN sessions (overnight +5.8 deg toward the target, mixed model; day step +3.3, 4/4)
    -> the spontaneous-recovery pattern (cannot be told from offline consolidation). T2: ITI licks after a far-contra
    trial do NOT carry a re-aimed launch (contrast -5.9 subacute, 4/4, wrong sign; they track the cued deficit acutely).
    T3: the ITI ipsilesional bias IS behavioural (docked ITI landing vs pre after the same previous spout: chronic -7.5
    [-14.1, -1.0], 4/4; PS94 -11.0, PS95 -5.8) but the same size as the cued first-lick landing error at the same spouts
    (GAP -0.7 n.s.; cued landing -6.8 chronic) -> no capacity-vs-performance gap; gain on the previous spout's angle
    falls (0.23 -> ~0.18). Reading: recovery looks like spontaneous between-day improvement shared by cued and ITI
    licks; the cortical ITI persistence (offset) sits on a real behavioural bias that cued licks share, not one cued
    licks overcome. Caveats: PS95 has 1-2 pre ITI sessions; the hit rate falls within post sessions despite the gate.
    **PROVISIONAL (n = 4).**
23. **Pre-launch hemispheric tuning to the launch angle: NOT MEASURABLE (10-30; pre-registered,
    `docs/status/PRELICK_HEMISPHERE_TUNING_PREREG_20261030.md`; `scripts/prelick_hemisphere_tuning.py`).** Cued first licks
    (no lick in the prior 1 s), per hemisphere orofacial-pool slope on the WITHIN-spout launch angle, pre-launch
    -0.30..-0.05 s and post-contact +0.05..+0.30 s. Gate failed: 0/4 animals pre-launch, 1/4 post-contact (|partial r|
    ~0.04; per-session noise 0.24-0.58 ~ between-session spread). The contraversive pattern exists only post-contact
    (D -0.55, 4/4). No post-stroke change in either window / hemisphere (all Holm p >= 0.59). Later-lick total-angle
    flattening (+0.35, 4/4) = the known post-lick contralesional-share rise, not launch tuning. -> widefield area signals
    cannot separate the pathway routes (contralesional takeover vs ipsilesional bypass); needs perturbation (cohort 2).
    **NOT ESTABLISHED.**
    PIXEL add-on (Priya 10-30; `docs/status/PRELICK_SVD_TUNING_PREREG_20261030.md`, `scripts/prelick_svd_tuning.py`; page
    S25b): ALM + tjM1 pixels per hemisphere (PCA K = 10), pooled August pre sessions (LOSO), frozen decoders. PRE gate
    nominal 3/4 (all IPSILESIONAL hemisphere, r ~0.05; 2/4 Bonferroni x2, 2/4 K = 5, 1/4 K = 20) but POST positive control
    FAILED (2/4) -> underpowered, no demonstrated pre-launch pattern. No post-stroke change (Holm p >= 0.78). Area pool
    through the same pooled pipeline: nothing PRE -> pixels recover a little more (esp. post-contact; averaging cancels
    some spatial pattern). Still NOT ESTABLISHED; perturbation needed. Open check: within-session CV r (`_cv_r` form,
    fold-wise intercepts) is negatively biased near zero signal -- also affects the area decode's r_ex levels.
24. **Pre-launch AIM TRACKING with a frozen across-spout decoder: no tracking; INCONCLUSIVE by the prereg rule (10-31;
    `docs/status/PRELAUNCH_AIM_TRACKING_PREREG_20261031.md`; `scripts/prelaunch_aim_tracking.py`; page S26).** Stroke supplies
    the variation (launch -12.5 deg acute at the same target). Allen 0.05-s bins -0.50..-0.05 s from tongue emergence,
    pooled August pre, frozen. Gate: cued 3/4 (r .04-.10), ITI FAILED 2/4; calibration tiny (0.03-0.14 deg/deg). Tracking
    index (0 = target only, 1 = follows aim): cued acute -0.64 [-1.84, 0.81] (raw decoded +0.4 deg vs -12.5 executed; full
    tracking rejected p .02), subacute 0.56 [-0.72, 2.22], chronic not estimable; Holm p = 1.0 all; per-animal signs mixed.
    Positive control post-contact: cued 0.28 / 0.48 (4/4), far contra 0.63 -> cortex follows the EXECUTED angle only after
    the tongue moves (per-bin from +0.125 s; nothing before jaw onset). No-baseline sensitivity: acute -0.21 / 0.02, upper
    bounds < 0.5 (target-only reading; sensitivity only). SVD / LocaNMF / -0.10 s end / per hemisphere: no tracking.
    FACE CONFOUND: FaceRhythm pre-launch decodes angle r 0.44-0.52 and TRACKS acutely 0.49 [0.24, 0.80] (4/4); neural
    features residualised on face calibrate in no animal. -> no cortical pre-launch aim signal visible in widefield; the
    face already moves in the deviated direction before launch. NOT ESTABLISHED.
    CAVEAT (10-31, found while pre-registering the face follow-up): FaceRhythm factors are VQT outputs with centred Hann
    filters of length Q(f)/f (Q 4 -> 20 over 1-60 Hz); at the lick frequencies (5.8-8.2 Hz) they reach +/-0.56-0.69 s
    (half-height +/-0.28-0.34 s), so EVERY pre-launch FR bin (and part of the baseline) contains movement from at / after
    tongue onset. The FR tracking (0.49) may be the launch smeared backwards in time -> NOT a pre-launch face result
    until the timing-valid measures (face-point PCs, cam4 jaw) confirm it (face follow-up prereg 10-31).
25. **Face pre-launch aim follow-up: the face tracks the deviated aim only ONCE THE MOVEMENT STARTS; the ITI decoder
    offset is only PARTLY face-carried (10-31; `docs/status/FACE_PRELAUNCH_AIM_PREREG_20261031.md`;
    `scripts/face_prelaunch_aim.py`, `scripts/face_offset_partial.py`; page S27).** Timing-valid face measures (point
    PCs 25 Hz, cam4 jaw 250 Hz): no tracking before launch (cued points 0.39 / 0.19, CIs incl. 0; ITI points acute 0.68
    [0.13, 1.39] Holm p .09, gone (0.18) without the last 0.05-s bin = jaw opening; nothing before jaw onset); points track
    from +0.025 s after tongue onset. FaceRhythm reproduces S26 (0.52, 4/4) but equally before jaw onset -> filter leak.
    Within-spout pre-launch angle not readable from face points (0/4). Q5 offset, FR (11 traces) regressed out of cortex
    on pre licks, frozen (before reproduces published to 0.001 mm; planted offset >= 98 % removed): docked ITI +0.30/+0.22/
    +0.29 -> +0.23/+0.17/+0.24 (retained 0.78-0.82, lower bounds < 0.5 -> PARTIAL); PS92 -> ~0 every epoch, PS94/PS95 kept,
    PS93 small throughout. Variants: 5 groups 0.87-0.97, per-session 0.51-0.59, decoder refit 25-47 % retained. Cued
    matched-error ~0 -> +0.18 acute after partialling (Holm .051, mixed signs; not interpreted). **PROVISIONAL (n = 4).**

## 5. FaceRhythm: face movement on one frozen pre-stroke factor set

*Owner e3; detail [FACE_RHYTHM_FINDINGS.md](FACE_RHYTHM_FINDINGS.md) (section numbers below), page
https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1. Frozen rank-12 master `master_pre_v1` fit on pre-stroke sessions of
all four animals; every session projected onto it; expression = ABSOLUTE (weights x FR normaliser), log2 or ratio vs
each animal's own pre. Factor names, not numbers (numbers shift between fits).*

1. **Acute: the contralesional lick factor falls more than the ipsilesional one** (confirms the known acute deficit).
   Peak lick_right, all trials, acute .30-.46 of pre (log2 -1.37**), 4 / 4; on contra-spout trials -.60 [-.67, -.52];
   ipsi-spout lick_left n.s. at every epoch. **STANDS** (5d).
2. **The contralesional deficit persists PER LICK while the lick count recovers.** lick_right per right lick, log2 vs
   pre: acute -1.06**, subacute -.44*, chronic -.57 [-.79, -.34]** (chronic ratio .57-.79, 4 / 4 below pre at every
   epoch); right licks per trial back to pre chronically (-.14, n.s.; PS94 .52 the exception). lick_left per left lick
   n.s. **STANDS** (5b-master); same by peak amplitude on contra-spout trials, chronic -.30 [-.40, -.21], 4 / 4 (5d).
3. **Substitution: contralesional spouts are reached with more of the ipsilesional lick pattern.** lick_left on
   contra-spout trials subacute +.94 [.34, 1.54], chronic +1.08 [.35, 1.87] (1.2-3.3x pre), 4 / 4; not on ipsi-spout
   trials. A re-mixing of the pre-stroke repertoire, not a new movement (5d, 6b, 6-master). **STANDS**.
4. **Contra / ipsi face asymmetry at the contralesional spouts, per lick.** Contra-minus-ipsi half of the face, change
   vs pre, lick-aligned: far_R acute -.24**, chronic -.14 [-.23, -.05]**; close_R chronic -.09 [-.14, -.04]**; every
   epoch and alignment, 4 / 4. Speed maps (cue 0..+4 s, per point): close_R -.27** / -.16** / -.19**, far_R -.47** /
   -.29** / -.22* (acute / subacute / chronic); at the R spouts each lick moves the contralesional lower face less and
   the ipsilesional face more (per-point maps, >= 3 / 4 + FDR). n.s. at the L and centre spouts. **STANDS** (5e, 5f).
5. **The pre-stroke vocabulary fits post-stroke movement worse acutely.** EVR ratio frozen / own, vs pre: acute -.035
   [-.045, -.025] 4 / 4; subacute -.008, 4 / 4; chronic -.008 [-.016, -.0003], 3 / 4 (PS92 / PS94; PS93 fully back).
   **STANDS** acutely, **PROVISIONAL** chronically (6-master).
6. **The contralesional lick movement is re-timed later in the response window, not delayed at its start.** lick_right
   centre of mass, contra-spout trials: subacute +.40 s, chronic +.41 s [.23, .59], 4 / 4 later; first lick on the same
   trials EARLIER chronically (-.06 s, 0 / 4 later). **STANDS**, caveat: a trial average mixes within-trial timing with
   which trials carry the movement (per-bout timing pending) (5g).
7. **Survey of every factor (peak amplitude).** Acute drops in the shared lick and pre-lick nose / sniff factors
   (lick_ongoing_7Hz -1.22**, lick_fast_close -1.01**, nose_prelick_slow -.78**, sniff_prelick -.68**); chronically
   several factors ABOVE pre (lick_fast_close +.72**, lick_ongoing_7Hz +.50**, lick_far_nose +.42**, whisk_run +.42**).
   **STANDS** (5d).
8. **Non-task face movement is reduced acutely, not preserved** (the 2pRAM cohort's relative preservation does not
   replicate): between-trial movement falls within behavioural state, rest -.99 log2 [-1.24, -.80] 4 / 4, other -.37
   4 / 4; back to pre chronically. **STANDS** (6d). Pooled task vs non-task (10-25, `face_rhythm_readouts.py
   --pooled-only`, log2 vs own pre, paired bootstrap): acute task -.47* / non-task -.62* (4 / 4 each), task - non-task
   n.s. at every epoch; chronic task +.22* above pre (3 / 4; PS92 at pre).
9. **Treadmill pushes are inside the pre-stroke face repertoire**: frozen-basis error at pushes is below random times in
   27 / 32 animal x epoch x context cells (28 / 32 in the current cache), although pushes are 2-4x more frequent
   chronically. No pre + post master needed. **STANDS** (6b). Pooled (10-25, `face_rhythm_push_check.py
   --pooled-only`; push - random in own pre-SD units, paired bootstrap): the advantage is already there pre-stroke
   (between trials -2.2*, 4 / 4) and chronically (-2.2*, 4 / 4); it shrinks only acutely between trials (change vs pre
   +1.76*, 3 / 4) while push movement energy rises (+1.82*, 4 / 4). Lower error may partly reflect larger movement.
10. **The lick factors change shape after the stroke, becoming less side-specific rather than re-aimed.** Pooled per
    animal x epoch fits: lick_right / lick_left / lick_ongoing beyond the pre-half floor in 4 / 4 at every epoch;
    lick_right's loading leaves the contralesional near-midline face for the mouth rim and ipsilesional cheek (PS92 /
    94 / 95; PS93 opposite); it still resembles the R spouts' pre pattern, less strongly. Per-session matching:
    heterogeneous across animals. **PROVISIONAL** (descriptive, one value per animal x epoch) (6c).
    POOLED within-animal versions (Priya 10-25; FR page Figs 8.1, 8.15, 8.17; `face_rhythm_epoch_shape.py`, `face_rhythm_match.py`).
    FACTOR PAIRING (decided 10-25 with Priya, DECISIONS): each per-epoch own fit's factor for a master factor = its BEST
    SPATIAL match (allowed to repeat), not the one-to-one Hungarian assignment -- post-stroke leftward licking is split over
    several own factors, and one-to-one pairing handed lick_left poor partners (PS95 acute a ~12 Hz factor; PS95 subacute
    cos .35); spatial, not spatial x spectral, because the readouts ask WHERE a factor loads (PS93 acute lick_right: own 4
    has the place, .92, own 11 the rhythm). All lateral-lick picks then have spatial cos >= .74, none shared; fits whose
    best is < .5 are dropped (`MIN_COS`). Results (loading change / own pre-fit noise, mean of animals, rings = >= 3
    animals beyond 2x own noise, chance ~0 of 1,282): lick_right loses loading on the contralesional central face and
    gains on the ipsilesional cheek / mouth rim (397 / 311 / 169 points acute / subacute / chronic), lick_left the reverse
    (211 / 69 / 163); asymmetry change lick_right -.13 / -.08 / -.07, lick_left +.14 / +.13 / +.18 (3 / 4 each, PS93 the
    exception); angle / own floor lick_right 5.1 / 3.4 / 3.0, lick_left 3.6 / 3.3 / 2.5, 4 / 4 every epoch. Per-session
    own fit vs master (Hungarian, z vs own pre, paired bootstrap, Holm over 12): lick_left changed at every epoch (spatial
    ** 4 / 4), lick_right * (3 / 4, PS93 opposite). Together: the two LATERAL lick factors become LESS LATERALISED, 3 / 4
    animals at every epoch (PS93 the exception); no session-level test for the per-epoch fits. CENTRAL lick factor:
    lick_ongoing_7Hz barely changes (5 / 1 / 0 points, asymmetry -.10 acute 4 / 4) -- the first-version acute 438-point
    ipsilesional shift (Hungarian pairing, which gave acute lick_left and lick_ongoing_7Hz one shared own factor in 4 / 4)
    is **WITHDRAWN**. lick_far_nose is not recovered even in the pre-half fits (cos .27-.39): its shape readouts are void.
    Match galleries + per-fit candidate panels on the FR page (`--gallery`, `--candidates`).
    COHORT FITS (10-25 evening; `fr_factors.py fitcohort` on O2, 30 fits = all four animals pooled + each animal left out,
    x pre / preA / preB / acute / subacute / chronic, 5,500 rows per animal; `face_rhythm_epoch_shape.py --own-dir
    own_epoch_cohort` -> `readouts/master_pre_v1/epoch_shape_cohort/`; FR page section 7). lick_right is recovered as the
    same factor in every fit (best spatial match own 4, cos .87-.97) and **stays changed through chronic**: r with the
    pre far_R spout pattern .71 / .72 (pre halves) -> .53 / .54 / .46 (acute / subacute / chronic), with close-centre .25
    -> .50; loading moves from the contralesional mid-face to the contralesional mouth rim and lower cheeks; spatial angle
    to pre 26 / 23 / 27 deg while most factors return to 7-14 deg chronically; asymmetry -.19 / -.09 / -.19 (jackknife SE over animals .10 / .10 / .15; angle SE 7 / 4 / 8 deg). Same in all
    four leave-one-out fits. **lick_left changes acutely and recovers**: far_L r .77 -> .56 -> .71 / .75, angle 30 ->
    15 / 14 deg, asymmetry +.33 (SE .15) acute then -.06 / -.12; the acute matched factor is broad / partly bilateral and differs
    from the Hungarian pick in 4 / 5 fits; subacute recovery leans on PS92 (without it far_L .45). This REVISES the
    per-animal reading for lick_left (chronic +.18 there, not confirmed). CAVEATS: the cohort pre halves interleave days,
    so their floor (3-8 deg) leaves out across-day drift and nearly every factor is "beyond" it -- read lick_right against
    the other factors (it alone does not come back), and the early-vs-late-pre drift floor is still owed; best-match
    pairing gives sniff_prelick / nose_prelick_slow and nose_steps / lick_far_nose one shared cohort factor each (those
    pairs not separately recovered). **PROVISIONAL**. SIGNIFICANCE (jackknife over animals from the LOAO fits, t with
    3 df -- the only cross-animal test without refitting): NOTHING SURVIVES CORRECTION. Spout-template r changes (FR page
    Fig 7.7): uncorrected p .02-.05 in 5 of 36 cells, all q >= .17 (BH over 6 spouts x 3 epochs per factor). Per-point loading
    (Figs 7.8, 7.9): lick_right 80 / 286 / 217 points p < .05 against 64 expected, 0 at q < .10 (BH over 1,282); lick_left acute
    292, q < .10 at 5 points. One number per factor (side-specificity = mean r with own-side spouts minus other side):
    lick_right -.38 / -.33 / -.47 (p .19 / .07 / .07), lick_left -.43 acute (p .04), then -.10 / -.07 (n.s.). LOAO fits
    share 3 / 4 of their data, so 4 / 4 same sign is NOT independent evidence. A powered test needs a session-level null:
    refits on session-shuffled epoch labels within animal (O2).
    PER-SESSION STATS (10-25 night; `scripts/face_rhythm_session_shape.py` -> readouts/master_pre_v1/session_shape/; FR
    page section 7 "Per-session fits"): each session's own rank-12 fit (`own/`), best spatial match per lick factor, one
    value per session, paired animals -> sessions bootstrap, Holm within factor (3 measures x 3 epochs); pre sessions
    scored against leave-this-session-out spout templates. lick_left side-specificity (own-side r - other-side r) -.29 /
    -.27 / -.17 (Holm p .002 / .002 / .016, 4 / 4); far_R r +.18 / +.18 / +.10 (4 / 4); asymmetry +.19 acute (Holm .002,
    4 / 4). lick_right side-specificity -.38 / -.27 / -.13, intervals exclude 0 every epoch, Holm .24-.34, 3 / 4 (PS93
    pre already .46 vs 1.2-1.6; pre-existing right orofacial deficit); far_R r -.19 / -.14 / -.09. Per-point loading,
    FDR .05 of 1,282: lick_right 441 / 644 / 45, lick_left 667 / 186 / 17. CAVEAT: per-session cosine to the master also
    falls (lick_left -.08 / -.07 / -.06, 4 / 4), so single-session splitting / mixing contributes. DISAGREES with the
    cohort fits on lick_left after acute (cohort: recovered by subacute; per session: still changed chronically); the
    cohort permutation null + early-vs-late pre drift fits (O2, submitted 10-25 night: `fitcohortnull` tasks 1-600,
    `fitcohort` tasks 31-40) decide. **lick_left STANDS per session; lick_right PROVISIONAL.**
    HEADLINE (Priya 10-25; FR page Fig 7.1): per-point loading change with each animal / its own pre-session SD (floored at
    1/4 of its median SD), FDR .05 of 1,282 points: lick_right 717 / 760 / 54, lick_left 660 / 208 / 33. SPECTRAL twin
    (Fig 7.2; same test over the 36 frequencies of the matched factor's frequency loading): lick_right loses weight at its
    6.5-7.3 Hz peak (-2 SD) and gains at 4.6-5.1 and 10.4-11.7 Hz (+2 SD); 5 / 6 / 2 frequencies FDR .05, the 10-12 Hz
    gain persisting chronically. lick_left: mean peak shifts lower acutely (~5 Hz) and flattens, 0 frequencies survive.
    OTHER TEN FACTORS (same per-session tests; FR page Figs 7.5 / 7.6): the lateral pair changes most and longest; next
    whisk_run acute 407 points, lick_fast_close 334 / 185 / 65, lick_slow_late subacute 322; run_onset_jaw / prep_jaw <= 21
    points and no frequencies (in-data negative control). Spectra: nose_steps acute 33 / 36 frequencies (small tilt toward
    low frequencies), lick_far_nose acute 20 / 36 (loses high-frequency weight). BY-DAY MATCHING redone with best spatial
    match (`face_rhythm_match.py --on spatial --assign best` -> readouts/master_pre_v1/matching_spatial_vector_best/; FR
    page section 8): lick_left further from the master at every epoch, spatial z +2.4 / +1.9 / +1.7 (Holm ** 4 / 4),
    spectral ** acute and chronic; lick_right spatial +6.2 / +3.3 / +3.5 (* 3 / 4, PS93 opposite); acutely ** also
    sniff_prelick, nose_prelick_slow, lick_ongoing_7Hz, lick_fast_close, whisk_run, lick_slow_late (the first two and
    whisk_run also chronically). The whole FR page now pairs factors by best spatial match.
    DRIFT FLOOR (10-25 late night; `scripts/face_rhythm_drift_floor.py` -> readouts/.../epoch_shape_cohort/drift_floor.png;
    O2 `fitcohort` 31-40 = COHORT / LOAO fits on the first vs second half of the pre days IN TIME). Pre-stroke drift is
    small: early-vs-late angle 5-7 deg spatial (lick_right 5.2, lick_left 5.2), barely above the interleaved-day noise
    (3-8 deg). Against the LATE pre half, lick_right is 25 / 22 / 26 deg (acute / subacute / chronic), beyond its own
    drift in 5 / 5 fits (COHORT + 4 LOAO) at every epoch, and at chronic the largest of all 12 factors; lick_left 29 / 15 /
    15 deg (still ~3x drift chronically, 5 / 5 fits -- "recovered" in the cohort means smaller than acute, not back to
    pre). CAVEAT: EVERY factor's cohort fit is 10-17 deg from late pre after the stroke (jaw factors included), so the
    angle beyond drift is not by itself lick-specific; specificity comes from the per-session maps (jaw factors ~0 points)
    and lick_right being largest chronically. Drift does NOT explain the lick_right change. **STANDS** (drift part).
    COHORT PERMUTATION NULL (`scripts/face_rhythm_cohort_null.py` -> readouts/.../epoch_shape_cohort/cohort_null_
    {scalars,maps,spectral}.png + data/cohort_null.csv; O2 `fitcohortnull`, within-animal label shuffle of pre vs epoch
    sessions, both fits redone; templates rebuilt per permutation from its pseudo-pre sessions; max-|t| over mesh points
    / frequencies; Holm over the lick pair and separately over the other ten). 10-26: all 600 files = 100 permutations per epoch
    (acute / subacute / chronic). LICK_RIGHT beyond every
    permutation at EVERY epoch on all four scalars (spatial angle 26 / 23 / 27 deg, spectral 21 / 22 / 14, side-
    specificity -.38 / -.33 / -.48; asymmetry -.19 acute and chronic, subacute p .03 not Holm), Holm p .010 = the floor
    1 / (n + 1). Maps family-wise .05: 0 / 21 / 142 points -- chronically a solid block LOSES loading on the image-right
    (ipsilesional) cheek; spectra 6 / 6 / 5 of 36 frequencies (loses ~7 Hz, gains ~4.5 and ~10-12 Hz at every epoch).
    LICK_LEFT: acute INSIDE the null (spatial 30 deg but the acute null spans 5-35; p .34, 0 points), BEYOND it subacute
    and chronic (spatial 16 / 15 deg, spectral 30 / 13, side-specificity -.10 / -.08, Holm p .010; 26 / 56 points, contra
    / image-left cheek and lip; 10 / 3 frequencies). This SETTLES the lick_left cohort-vs-per-session disagreement: the
    cohort fit's lick_left after acute is smaller than acute but beyond the null, as the per-session and per-animal
    fits said; the ACUTE cohort lick_left is the uncertain one (few acute sessions, lick_left swaps column in ~1 / 6
    shuffled fits). OTHER TEN: none Holm, but this is a RESOLUTION limit, not a negative -- with ~100 permutations the
    smallest p is ~.0099 and Holm over ten needs < .005; many sit at the floor (e.g. whisk_run, lick_fast_close,
    lick_ongoing_7Hz spatial acute). Calling any of them needs >= 500 permutations. The null includes drift (within-
    animal shuffle across all pre days). **STANDS** (lick pair, subacute + chronic; lick_right acute too).
    PER-SESSION SHAPE TESTS -- THE MAIN TESTS FROM 11-07 (Priya; DECISIONS 2026-11-07). Every session's own fit (own8)
    matched to the master with rule v2 (spatial score + spectral tie-break; recovered at spatial cos >= .7 and spectral
    cos >= .5; merged units prelick_nose, slow_jaw); every readout = the session minus its animal's OWN pre mean, mean
    over animals; within-animal session-label shuffles (10,000, no refit); Holm within a family ACROSS all epochs and
    measures (lick pair 24 tests, other units 72); pre-stroke drift control (early vs late pre) for every test
    (`scripts/face_rhythm_session_shape_tests.py`, `face_rhythm_session_loading_perm.py` -> readouts/master_pre_v1/
    session_shape_v2/, session_loading_v2/). LICK PAIR (acute / subacute / chronic): spatial angle to own pre mean
    lick_right +14.6** / +7.6** / +5.5** deg, lick_left +12.1** / +10.5** / +9.9**; side specificity lick_right -.38** /
    -.25** / -.13**, lick_left -.31** / -.23** / -.10; asymmetry lick_left +.23** / +.18** / +.10* (more contralesional),
    lick_right -.15 acute (n.s. after the family-wide Holm); spectral angle lick_left +22** / +21** / +14**, lick_right
    +11** acute. Pre drift (late vs early pre): no readout of any unit passes Holm; lick pair ~0.6 deg spatial. MAPS (cluster
    mass, mesh neighbours |t| > 2): lick_right acute 843 points (882 when each animal is scaled by its own pre session SD
    = the headline version; 783 with ambiguous sessions excluded); lick_right loses weight over the upper / central face
    and gains on the ipsilesional lower cheek and jaw; lick_left roughly the mirror; the drift control finds no lick_right
    cluster and one lick_left cluster (145 points, ipsilesional-central, opposite sign and place to its acute change).
    The lick_right map change does NOT track the drop in right licks (acute rho -.14, p .61). lick_left is AMBIGUOUS acutely
    (median margin to the runner-up component .02 vs .20 pre): only 2 acute sessions are unambiguous, so its acute map has no
    exclusion sensitivity. OTHER UNITS: most change spatially acutely (e.g. whisk_run +12.4**, lick_ongoing_7Hz +12.7*,
    lick_fast_close +12.7*), exploratory family. FACTOR LOSS per session (recovery rate vs own pre, Holm over 30): only
    chronic lick_left survives (-.27, Holm .012; mostly PS93, 2/12 sessions); acute lick_ongoing_7Hz -.44 (p .008) and
    lick_fast_close -.21 (p .04) do not survive -> the acute fast-lick loss is SUGGESTIVE (significant only in the pooled-fit
    null). POOLED-FIT RESULTS are now the descriptive second view: a fragility check (refit-null 95th percentile / pre-only
    fit noise) flags lick_left (7.2x) and lick_far_nose (5.8x) as pooled-test-inconclusive; a pooled fit is not the average
    of the animals (chronic lick_left pooled -.12 asymmetry vs +.27 in 3 / 4 animals' own fits). PS93 (pre-existing right
    facial deficit, pre-stroke right-lick bias 46 % vs 6 % left) stays in; its lick_right is split over two components in
    many sessions, so its per-session maps flip. **STANDS** (lick pair spatial angle, side specificity, lick_left asymmetry,
    lick_right map); **PROVISIONAL** (acute fast-lick factor LOSS per session; other units). Since 11-07 night every
    map test uses 10,000 shuffles (only change vs 2,000 that touched text: the cue-aligned speed-map drift control lost
    its far_center cluster, 2 clusters left at far_R).
    RHYTHM-BAND MOVEMENT + LICK RATE (11-07 night, Priya; FACE_RHYTHM_FINDINGS 6f). Video licks (LP v36): the within-bout
    inter-lick interval lengthens 15-25 ms at EVERY spout acutely (session MWU + bootstrap agree), back by subacute except
    close mid / contra (`scripts/orofacial_lick_rate_figure.py`). Face movement per mesh point in each lick unit's rhythm
    (stored per-point spectra weighted by the unit's master frequency profile, cue 0..+4 s, no refit;
    `scripts/face_rhythm_band_maps.py`) falls acutely at every spout in every lick band (face mean log2 -.09 to -.30, Holm
    .002) and the movement spectrum SHIFTS TO LOWER FREQUENCIES: its power-weighted centre (3-15 Hz, all face points, log
    frequency) drops 6.77 -> 6.56 Hz acutely (-.21 Hz, 4 / 4 animals, Holm p .0003; 4-12 Hz -.12 Hz, p .0003), back by
    subacute (chronic +.08 Hz in 3-15 Hz only); the licks themselves slow 6.45 -> 5.71 Hz (1000 / ILI), the face spectrum
    less because it also holds non-lick movement and harmonics (`face_rhythm_band_maps.py --centre`). Band by band, the
    upper lick band (lick_ongoing_7Hz profile, 6.5-15 Hz) falls MORE than the main lick rhythm (4-9 Hz; -.13 log2, Holm p
    .002; subacute -.04, Holm .03), the slower lick bands less (slow_late +.07, far_nose +.05 relative); fast_close not
    different (-.04, n.s.). That is what a downward shift of the licking peak (or less sharp licks) produces -- not
    evidence for the loss of a separate fast movement. Chronically every lick band is above pre (+.06 to +.13). Drift control passes no
    band or contrast (10,000 within-animal shuffles; Holm over 7 x 3 bands, 6 x 3 contrasts). Limits: bands separate units by
    rhythm only (lick_right / lick_left share theirs, read by spout); profiles are broad and overlap. **STANDS** (acute lick
    slowing and the downward shift of face movement frequency).
11. **"Acutely fewer licks, not smaller ones" -- WITHDRAWN (10-24, Priya).** The face mesh cannot see tongue extent or
    lick rate; the DLC / LP kinematics show acute licks are slower and shorter (section 1). The acute per-trial drop =
    fewer, later and slower / shorter licks (5f).
12. **Severity does not order by lesion burden** (largest acute task drop in the mildest lesion, PS95). Descriptive,
    n = 4 (6d; section 6).
13. **The face carries spout position BEFORE the cue, on lick-free trials** (Priya 10-25; `scripts/fr_enl_decode.py`,
    FACE_RHYTHM_FINDINGS 7). 12 frozen factors in 0.5 s bins, trials with no LP lick and no spout contact in the
    window, the imaging decoder (logistic C .5, position-block CV), chance .167, 107 sessions. Pre-stroke 2 s window
    .43 (per animal .39 / .47 / .49 / .38; 31 / 31 sessions p < .05); 1 s .39. FR smooths acausally (~0.3-1 s), so
    features stopping 0.5 s BEFORE the cue are the clean test: 2 s window .34 (31 / 31), 1 s window .25 (18 / 31) --
    the position signal is real but sits mostly in the second before the cue. Acutely lower (2 s, above-chance change
    -.10 [-.21, +.01], 3 / 4; end -0.5 s -.08 [-.17, +.02], 3 / 4), back chronically (.46). The acute drop is FAR
    CONTRA: far_R recall .42 -> .19 (chance) -> .41 chronic; other spouts change little. **PROVISIONAL**. Bearing on
    imaging 3.10: the cortical pre-cue position code (.33-.55) sits in the same range as the face's, so part of it may
    be face posture / movement; a movement-removed cortical ENL decode with FR as regressors is the test. Untested:
    whether the FR mesh sees the spout itself -- TESTED 11-05, it does not: `scripts/fr_spout_mesh_control.py` removes
    the mesh points within 30 / 60 px of every session's spout tips (15 / 63 of 1282; none within 56 px of a far tip)
    before projection; the decode is unchanged (2 s pre .431 -> .428 / .418; session changes median 0 / -.006, all
    within +-.04; acute far_R recall .19 at every radius; same epoch deltas).
14. **The cortical pre-cue position code is NOT the face movement FR captures** (10-25; `scripts/enl_fr_removed_decode.py`,
    FACE_RHYTHM_FINDINGS 7b; answers 13's test). The video analysis's trials, features and in-fold removal
    (`enl_movement_multisession`; raw arm identical on all 107 shared sessions), with the 12 FR factors + running
    (lags 0-0.2 s) as the movement model. Mean raw .424 -> minus FR + running .390 (minus video + running was .272);
    105 / 107 sessions still p < .05. Share of raw's above-null code kept, per animal x epoch: .70-1.00 (pre .81 / .91
    / .86 / .91); with features ending 0.5 s before the cue (no acausal lick leak) .76-1.01. The residual tracks raw
    session by session (r .95 / .91 / .91 / .97, PS92-95), so every epoch pattern of the cortical ENL code (imaging
    3.10: no consistent acute loss; subacute dip PS92 / PS93) survives FR removal unchanged. FR alone decodes position
    about as well as cortex (.37-.51) but only partly the same trials' structure (r(raw, FR alone) .47-.72): the face
    and cortex carry position largely SEPARATELY. **PROVISIONAL** (linear removal, lower bound; FR's mesh omits body /
    whisker-pad regions video sees -- a likely, untested, reason video removes more).

## 6. Histology

*Detail [HISTOLOGY.md](HISTOLOGY.md).* Lesion burden = 100 x ipsilesional / contralesional caudoputamen area per
coronal section (lower = more striatum lost), mean over sections after exclusions (10-21): PS93 45.0 %, PS92 46.7 %
(most severe), PS94 67.7 %, PS95 81.9 % (mildest). **STANDS** (5-7 sections per animal). Stroke-extent schematic on
Paxinos plates: `wfield_local/histology_map/`.

## 8. Context in the field: our results against the stroke literature (11-01; all arms may edit)

**FRAMING (Priya 11-01).** Directional TONGUE REACHING in head-fixed mice as a model for dissecting post-stroke motor
control with a precision rodent limb-reach models rarely reach: a single targeted lick has the hallmarks of primate online
reach control -- corrective submovements after misses that need ALM (Bollu et al. 2021 Nature,
https://pmc.ncbi.nlm.nih.gov/articles/PMC8299742/); its planning circuit parallels primate premotor delayed-reach
preparation (ALM directional preparatory activity, Guo et al. 2014 and Svoboda-lab follow-ups,
https://pmc.ncbi.nlm.nih.gov/articles/PMC6212318); ALM carries intended lick angle while tongue-jaw S1 / M1 carry
lick-level kinematics (Xu et al. 2022 Nature, O'Connor lab, moving-port task, https://pmc.ncbi.nlm.nih.gov/articles/PMC9109820);
and orofacial deficits are clinically common (post-stroke dysphagia pooled 46.6 %, https://pmc.ncbi.nlm.nih.gov/articles/PMC11288910/;
dysarthria 44 % in one acute cohort). What our setup adds over typical reach models: 6 targets crossing distance x side
(graded laterality); hundreds of cued trials + thousands of cued / self-initiated licks per session; 3-D tongue (4 cameras,
250 fps), jaw, whole-face FaceRhythm, contact sensing; first vs corrective licks and no-contact attempts (Bollu's
corrections, post-stroke); whole dorsal cortex imaged on every trial; daily sessions over ~50 days. That precision is what
makes the frozen-encoder separation of movement change vs mapping change, compensation vs restoration on hits, and failed
attempts vs hits possible (section 0).
CAUTIONS: do not claim rodent reaching is "less homologous" -- contested (Whishaw lab: hand-shaping homology rat / human /
primates, https://pubmed.ncbi.nlm.nih.gov/19520119/; Karl & Whishaw 2013, https://pmc.ncbi.nlm.nih.gov/articles/PMC3870330);
say "shares primate reach control principles with far greater kinematic resolution, trial counts and target control". The
lick rhythm is brainstem-generated (cortex = targeting, initiation, correction); tongue innervation is partly bilateral
(answer: graded far-contra deficit, contralesional shifts); the lesion is a basal-ganglia (striatal) stroke, not
corticospinal (frame as subcortical stroke; note Cettina 2025 capsular contrast; PS93 cortical injury); n = 4 (cohort2 =
replication).

*Priya 11-01: what supports, conflicts with, or is new relative to preclinical and clinical stroke work. Draft from one session's searches (abstracts); extend as papers are read. Same table as the quiet-events handoff section 6.*

Citations checked by web search on 11-01 unless marked *(memory)*; abstracts only, not full texts. Our results =
FINDINGS.md sections 0-5 plus this arm.

| our result | prior literature | verdict |
|---|---|---|
| Behaviour recovers by COMPENSATION (success back; contralesional licks use a changed pattern) | Established in human kinematics and rodent reaching (Levin / Kleim / Wolf definitions; Whishaw rat reaching) *(memory)* | SUPPORTS; orofacial / licking version in mouse new |
| LARGER task responses (ITI lick ~2x acute; chronic residual beyond the pre mapping), widespread, largest day 1, shrinking with recovery, persisting in the least recovered (PS94) | Human fMRI: task overactivation decreasing with recovery (Ward et al. 2003 Brain); meta-analysis: more contralesional M1, bilateral PMv, SMA, decreasing with time (Rehme et al. 2012 NeuroImage). Rodent widefield after CORTICAL stroke: activation spreads and refocuses with rehab (Allegra Mascaro et al. 2019) *(memory)* | STRONGLY SUPPORTS the human picture incl. time course and deficit link; new as rodent calcium after a striatal lesion |
| (same) | Rat INTERNAL-CAPSULE stroke, multi-unit in M1 / RFA / S1, weeks 5-8: task firing slightly REDUCED, population pattern PRESERVED (Cettina, Nudo, Bundy et al. 2025 bioRxiv) | CONFLICTS (see below) |
| CONTRALESIONAL > ipsilesional residual; contralesional shift of decoded lick position | Contralesional recruitment = most consistent fMRI finding (Rehme 2012). Orofacial: swallowing recovery tracks pharyngeal-cortex expansion in the UNAFFECTED hemisphere (Hamdy et al. 1998 Gastroenterology) | SUPPORTS direction; decoded-position shift is a new measure |
| LARGER QUIET FLUCTUATIONS (2-3 Hz, both signs, ~x6.7 acute with LP-lick quiet frames, recovering; the 'faster decay' did not survive movement correction) | Human EEG: more delta after stroke, contralesional delta prognostic of poor recovery; sleep-like slow waves / OFF-periods in awake cortex after cortico-subcortical lesions (Sarasso et al. 2020 Brain) | CONSISTENT with human; new in rodents after a subcortical lesion |
| (same) | Rat: reach-locked low-frequency oscillations in perilesional M1 WEAKENED after stroke, their return tracks recovery (Ramanathan et al. 2018 Nat Med). Rodent widefield cortical stroke: connectivity lost then partly restored (eLife 2024, 90080) | TENSION (movement-locked vs quiet; test 4.7) |
| ACUTE ENCODING LOSS = the changed movement (frozen encoder predicts the far-contra drop from the lick) | No study found that asks whether the pre-stroke mapping already predicts the change from altered behaviour | NEW, conceptually important |
| CHRONIC REMAPPING (same movement, different response; self-fit better; target and executed coding remapped) | Reorganisation is classic after CORTICAL lesions (Winship & Murphy 2008 J Neurosci; Latifi et al. 2020 Cereb Cortex; Nudo 1996 *(memory)*); after CAPSULAR stroke: no reorganisation (Cettina 2025) | NEW for remote cortex after a subcortical lesion; tension with capsular result; drift caveat |
| Failed attempts (no-contact licks) lose side coding and trial-by-trial coupling; first cued lick read toward target; coupling returns on hits; code plateaus ~3/4; cortex does not lead recovery | No direct precedent found | NEW |
| Lesion size orders recovery speed | Expected | SUPPORTS |
| Acute orofacial quiet events LOCAL to anterolateral orofacial cortex (15.7) | Fits the VLS -> SNr -> VM/VAL -> ALM / tjM1 loop anatomy (Lee, Wang & Sabatini 2020) as a hypothesis; no stroke precedent found | NEW (pending test 4.1) |

**The conflict to take seriously (Cettina et al. 2025).** Differences that likely explain much of it: internal capsule
(corticospinal fibres) vs VLS (basal-ganglia loop node, whose loss should change thalamocortical drive); weeks 5-8
post-plateau vs our largest effects acute / subacute; ipsilesional only vs our bilateral / contralesional effects;
multi-unit spikes in 3 areas vs widefield calcium over dorsal cortex. Their "pattern preserved" agrees with parts of
ours (3 / 4 animals keep the lateral read-out map; PS94 remapped). The real tension is our chronic residual vs their
chronic ipsilesional spiking.

**Brand new, as far as found:** (1) longitudinal whole-dorsal-cortex calcium after a STRIATAL stroke in an orofacial
task; (2) separating movement change from mapping change with frozen pre-stroke models; (3) behaviour fully back while
the code on hits plateaus ~70-77 %; (4) failed attempts as directed movements without a matching cortical spout
signal; (5) self-initiated vs cued read-outs diverge; (6) quiet-period fluctuation increase as a rodent counterpart of
human post-stroke delta / local sleep after a subcortical lesion, with a local orofacial footprint.

**Reviewer caveats from this literature:** neurovascular coupling (human overactivation was questioned on this; our 415
nm / F0 checks address it, FINDINGS 4d0c); the lesion is not purely subcortical (PS93 anterior motor cortex injury;
DLS involvement in some); drift vs remapping without a no-stroke reference; n = 4.

Sources: Cettina et al. 2025 https://pmc.ncbi.nlm.nih.gov/articles/PMC12637610/ ; Ward et al. 2003
https://discovery.ucl.ac.uk/id/eprint/10102084/ ; Rehme et al. 2012 https://pmc.ncbi.nlm.nih.gov/articles/PMC3630767 ;
Hamdy et al. 1998 https://orca.cardiff.ac.uk/id/eprint/33926/ ; Sarasso et al. 2020 https://pubmed.ncbi.nlm.nih.gov/33188680/ ;
Ramanathan et al. 2018 https://neatlabs.ucsd.edu/pubs/2018/Ramanathan2018.pdf ; photothrombotic widefield / assemblies
https://elifesciences.org/articles/90080v1 ; Winship & Murphy 2008 https://www.jneurosci.org/content/28/26/6592 ; Latifi
et al. 2020 https://pmc.ncbi.nlm.nih.gov/articles/PMC8257065 ; EEG delta commentary
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5133428/ . Still to verify: Allegra Mascaro 2019, Levin 2009, Nudo 1996,
Lee, Wang & Sabatini 2020 details; a targeted PubMed search for striatal / capsular stroke + widefield GCaMP.

**LITERATURE UPDATE 11-01 late (T17 agent; abstracts unless noted) -- corrections and additions to the table above.**
CORRECTIONS: (1) eLife 90080 = Bandet & Winship 2024 (two-photon, forelimb-S1 photothrombosis; spiking / assemblies
disrupted but HIGHLY LOCAL -- cortex ~750 um from the border largely unchanged; https://pmc.ncbi.nlm.nih.gov/articles/PMC11060715);
"widefield interhemispheric connectivity lost then partly restored" = Cramer et al. 2019 NeuroImage 199:570
(https://epub.ub.uni-muenchen.de/79115/). (2) CIRCUIT: Lee, Wang & Sabatini 2020 (full text,
https://pmc.ncbi.nlm.nih.gov/articles/PMC7606600) = VLS -> lateral SNr -> VM (VM/VA) and Pf -> tjM1, PLUS SNr -> lateral SC,
PCRt, IRt (brainstem orofacial premotor) directly; ALM via VM/VAL is Guo et al. 2017 Nature 545:181
(https://pmc.ncbi.nlm.nih.gov/articles/6431254). So the lesion can change licking without going through cortex.
(3) Allegra Mascaro 2019 Cell Rep 28:3474: cortical photothrombosis, Thy1-GCaMP6f, forelimb retraction; motor map
re-segregation with ROBOT TRAINING + CONTRALESIONAL SILENCING (combined), not rehab alone; Cecchini 2021 found new, not
restored, propagation patterns (https://pmc.ncbi.nlm.nih.gov/articles/PMC8159272). Levin 2009, Nudo 1996, Whishaw 2000
(Neuropharmacology 39:788) verified as cited.
ADD -- precedents (soften "new"): Bauer et al. 2014 NeuroImage (mouse tMCAO, SUBCORTICAL-only infarcts reduce lateral-cortex
homotopic connectivity; resting, no task; https://pmc.ncbi.nlm.nih.gov/articles/PMC4332714); Cullins & Connor 2024 Front
Neurol (rat tMCAO, striatum-centred, often into cortex, ~1/3 subcortical; ICMS maps at 8 weeks under anaesthesia;
https://pmc.ncbi.nlm.nih.gov/articles/PMC10987714/) -- ONLY PARTLY AGREES (Priya 11-01, after reading it): ipsilesional
TONGUE map shrinks (aged rats only; young p .057; more with larger infarcts) whereas our ipsilesional orofacial cortex is
NOT reduced (chronic lick responses larger there too; lateral read-out map kept in 3/4); intact-hemisphere change is JAW
only (tongue unchanged) and function tracks the INFARCTED hemisphere, intact-hemisphere maps do not correlate with function,
whereas our contralesional changes are the larger ones (we do not separate tongue vs jaw cortex, nor show which hemisphere
tracks function). Different measure (ICMS output efficacy vs calcium activity in behaviour), lesions and ages. Shared only:
some intact-hemisphere change;
Ahmed et al. 2017 JCBFM (lick efficiency down after striatal-only MCAO to day 40, compensatory drinking clusters;
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5305038/); Fanciullacci 2017 (human subcortical stroke: more delta over the
affected hemisphere); Gloor 1977 (delta from white-matter / thalamic deafferentation); Carmichael & Chesselet 2002 (rat:
peri-lesion synchronous 0.1-2 Hz activity days 1-3, drives sprouting; cortical lesion); Kim et al. 2022 Cell Rep (local
delta waves in sleep, normalise with training). Bollu et al. 2021 (ALM needed for corrective submovements) = mechanistic
anchor for failed attempts lacking a spout code. No study found that records ALM / tjM1 after VLS lesion or silencing, or
that uses a frozen pre-stroke encoder: both still look new.
ADD -- counter-literature a reviewer would cite: Branscheidt et al. 2022 J Neurophysiol (human SUBCORTICAL stroke, 1 year:
no recovery-related cortical resting-connectivity change; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8896990/); Zeiger
et al. 2021 Nat Commun (potentiation of pre-existing circuits, not remapping; https://pmc.ncbi.nlm.nih.gov/articles/PMC8233353) --
COMPATIBLE, largely semantics (Priya 11-01): our chronic change is a RE-WEIGHTED mapping within the same areas (3/4 keep the
read-out map's pattern), not territorial remapping -> say 're-weighted mapping', not 'remapping';
Sarasso 2020 caveat: off-periods ABSENT with unilateral LACUNAR subcortical lesions (ours are larger, some with cortex / DLS);
Panzeri et al. 2025 bioRxiv (contralesional widefield activity reflects intact-paw compensatory use); Ramanathan 2018 /
Khanna 2021 Cell (task-locked < 4 Hz activity DOWN, our quiet 2-3 Hz UP: T16 tests the dissociation); Facchin 2020
(ipsilesional NREM slow waves reduced). Answers to place beside them: frozen-encoder movement control, drift floor, T16.
EPILEPTIFORM CAVEAT (R3): Steinmetz et al. 2017 interictal events are > 10 % dF/F, 0.1-0.5 Hz, largest ANTEROLATERAL and
bilaterally symmetric -- location and our acute rate (15-32 /min) overlap; but their Thy1-GCaMP6s GP4.3 mice (our line)
showed none, our events are 2-3 % with up and down rising equally, events > 10 % are RARER acutely (T11), and late
post-stroke seizure risk tracks cortical involvement (Galovic 2018; note PS93's cortical injury). Remaining check: waveform /
refractoriness vs the Steinmetz criteria. Post-stroke peri-lesional spikes with 1-4 Hz delta are reported in mice from day
15 (https://pubmed.ncbi.nlm.nih.gov/41192651/).
DESCENDING ANATOMY (Priya 11-01): rodent motor cortex reaches tongue / jaw / face motoneurons mainly via brainstem reticular
premotor neurons (PCRt / IRt), with sparse or no direct cortico-motoneuronal contacts (as for the rodent limb), whereas
human corticobulbar fibres also contact hypoglossal / facial / trigeminal motoneurons directly *(memory: Kuypers; Lemon --
verify)*. The striatum projects to the brainstem in no species (via SNr / GPi; SNr -> brainstem premotor / SC in rodents and
primates). Framing consequence: ground homology in control principles and cortical computation, not the final pathway
(reach models share the limitation); brainstem premotor networks are a candidate compensation site; the SNr -> brainstem
route makes the frozen-encoder movement-vs-mapping separation essential.

## 7. Open questions and withdrawn claims (index)

* WITHDRAWN / CORRECTED / NOT ESTABLISHED claims are listed in their sections above; collect the cross-arm ones here.
* Stale elsewhere (flagged 10-25, not yet edited): CLAUDE.md's pointer to STATUS_2026-09-20 says "NO within-session motor
  fatigue"; DECISIONS 09-20 revised it the same day to "partly fatigue" (acute within-bout slope +10.1 ms; within-bout
  deceleration +15-25 ms at every epoch, 4 / 4).
* Open (section 3-4): October decoder rerun after the `_day` fix; chronic region-level (`15k`) cells on the full chronic
  cohort; FR trial-to-trial vs PETH split (O2 fr_A_tr, lined up 10-25); drift-group null for Layer 2 / residual (June sessions lack
  FaceRhythm and have off kinematics, so a June -> August encoder null is not available).
* Open: per-bout timing of lick_right (5.6); cohort permutation null for factor shape, full 100 x 3 epochs (5.10); other cameras for
  FaceRhythm (cam1-3).

- 11-01 CHECK: the 10-29 line "far-contra ALM acute loses significance (+2.7 n.s.)" does not match the current rescue-rule c2l_A_changes_restnl.csv (far-contra ALM_right acute +3.3, significant, 4/4); likely the strict variant or another panel. Unresolved. Acute far-contra pre-cue raise on RESTnl is NOT orofacial-specific (reference areas rise as much; page 14.21 / S2).
- GATE vs LP ATTEMPTS 11-01 (docs/status/GATE_LP_CHECK_20261101.md, scripts/gate_lp_check.py): the DAQ-contact engagement
  gate rarely gates out trials with an LP attempt (pre 0.5 %, acute / subacute ~3 %; not far-contra driven) -> working-trial
  maps / Haufe and the S28 quit tests are NOT affected (quit shift under an LP rule acute +0.40 min [-0.06, +1.15]; one
  chronic quit lost). Stopped class: 2.5 % pre -> 4.9 % acute -> 7.0 % chronic hold an LP attempt (borderline; sensitivity
  re-run optional). DAQ-arm pre-cue "lick-free" windows contain an LP lick in 17-25 % (2 s) / 10-18 % (1 s) of trials in
  EVERY epoch (no epoch change): DAQ-arm ENL = contact-free, not tongue-free -> cite the LP_licks ENL arm for any
  "not lick-driven" claim. PROVISIONAL.

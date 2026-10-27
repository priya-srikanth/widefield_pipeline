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

1. **Behaviour changed, and recovery is compensation, not restoration.** Acute global slowing + far-contra-graded deficit,
   licks aimed ipsilesionally from launch, jaw stops steering, far-contra trials attempted but failed in 3 / 4 (1.1-1.8,
   2.3, 4.4). Chronic completion returns, but each contralesional lick moves the contralesional face less and contra
   spouts are reached with more of the ipsilesional lick pattern, re-timed later (5.2-5.4, 5.6). -> Cortical change must
   be judged against behaviour that differs even when it succeeds; hence the frozen frame.
2. **Acute: losses are mostly the movement, plus a transient decoupling.** Frozen decoder loses far contra (.90 -> .32),
   position ~5x more than state (3.1, 3.2). The frozen encoder predicts the acute far-contra drop from the changed lick;
   target part ~0, with FaceRhythm too (4.1). Frozen pixel maps: the acute fit loss is cortex-wide and movement-driven (page
   Fig 1l). Beyond the movement: the decoder weakens for every spout, even on-target licks (4.5); acute far-contra trials
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

## 1. Behaviour and orofacial kinematics (cam4)

*Owner 5b; detail [OROFACIAL_FINDINGS.md](OROFACIAL_FINDINGS.md) (its status table, v36, is authoritative). cam4 = LP
v36, August-only baseline, both tests (session Mann-Whitney + BH-FDR and the animal-weighted bootstrap); stats table
`LP_analysis/lp_r4_heatmap_20261013__clean_v36/cohort/tables/prepost/stats/RESULTS.md`. Checked by 5b 10-25.*

1. **Acute global slowing plus a far-contralesional-graded deficit.** Inter-lick interval +14 to +26 ms at every
   position; far contra: contact -0.67, first lick +488 ms, first contact +898 ms. **STANDS**, 4 / 4 (PS95 acute = 1
   session).
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
13. **Not established / withdrawn here:** pre-stroke target-over-executed lead (SUSPENDED, 2.4); pooled toward-midline
    within-spout effect (NOT ESTABLISHED, 10-21); misdirected licks carry a plan signature (NOT ESTABLISHED, 2.5);
    lick-level strays decide target vs execution (NOT DECISIVE: reference fails validation, 2.6); Layer 2 pooled-refit
    claim (WITHDRAWN 10-24).

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
    POOLED within-animal versions (Priya 10-25; FR page Fig 7; `face_rhythm_epoch_shape.py`, `face_rhythm_match.py`).
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
    Fig 7): uncorrected p .02-.05 in 5 of 36 cells, all q >= .17 (BH over 6 spouts x 3 epochs per factor). Per-point loading
    (Fig 7b): lick_right 80 / 286 / 217 points p < .05 against 64 expected, 0 at q < .10 (BH over 1,282); lick_left acute
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
    HEADLINE (Priya 10-25; FR page Fig 7): per-point loading change with each animal / its own pre-session SD (floored at
    1/4 of its median SD), FDR .05 of 1,282 points: lick_right 717 / 760 / 54, lick_left 660 / 208 / 33. SPECTRAL twin
    (Fig 7b; same test over the 36 frequencies of the matched factor's frequency loading): lick_right loses weight at its
    6.5-7.3 Hz peak (-2 SD) and gains at 4.6-5.1 and 10.4-11.7 Hz (+2 SD); 5 / 6 / 2 frequencies FDR .05, the 10-12 Hz
    gain persisting chronically. lick_left: mean peak shifts lower acutely (~5 Hz) and flattens, 0 frequencies survive.
    OTHER TEN FACTORS (same per-session tests; FR page Fig 7e / 7f): the lateral pair changes most and longest; next
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
    whether the FR mesh sees the spout itself (the video motion-energy check masked the spout; FR has no such check).
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

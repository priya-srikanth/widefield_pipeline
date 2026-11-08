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

# Why is the ITI lick-response integral sometimes <= 0? (lateral-ratio diagnostic, 11-17)

Question (Priya, 11-17): `scripts/lick_lateral_ratio.py` computes log2(contra AUC / ipsi AUC) per session x spout. The
ratio is undefined (NaN) when either AUC is <= 0, and this happens mostly in the ITI set. Why is the integral ever <= 0?
Can we adjust for it? Is 0..0.5 s better? Is this only ITI?

**Decision taken while this was running (Priya, 11-17):** ITI AUC and balance results move to **0..0.5 s**; cued stays
at 0..1.0 s. Everything below supports that choice. Section 4 checks whether 0..0.5 s changes any ITI conclusion.

Cohort 1 (PS92-95), rebuild tag `..._daqlk`. Read-only: no production output was rewritten.
- Scripts: `neg_auc_diag.py` (extract), `neg_auc_analyze.py`, `neg_auc_characterize.py`, `neg_auc_events.py`,
  `neg_auc_undershoot.py`, `neg_auc_compare.py`, `neg_auc_windows.py`, `neg_auc_tjm1_traces.py`, all in this folder.
- Figures: `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/neg_auc_diag_20261117/`.
- **Reproduction check.** The rebuilt session x spout AUCs match production `lateral_ratio_sessions.csv` to 7e-15. The
  undefined flags agree on 100% of rows. The 0..1 s and 0..0.5 s change-vs-pre stars match production
  `lateral_ratio_tests.csv` exactly: 360 cells all-ITI, 119-120 docked.
- **Counting rule.** Counts here use per-spout + pooled rows only, without the side-group rows. That is why the shares
  are slightly higher than in the production log: iti 0..1 s is 0.035 here vs 0.028 in the log.

## Plain-language answer

1. **Only ITI, and not because of a bad baseline.** Cued first licks have no undefined rows at all (0 of 3745). Their
   session-mean curves stay well above baseline through 1.2 s (median tjM1 trough +22; 0 of 214 session curves dip).
   So the cued response is **not** biphasic. The cue-aligned set has 20 undefined rows. All of them are four acute
   far-contra sessions (PS92 0820/0821, PS93 0820/0821) where the cue response is absent in **both** hemispheres. That
   is a real acute deficit, not an artifact.
2. **ITI bout starts give a short transient with a dip below baseline after it.**
   - The transient peaks around 0.2 s at about half the cued size (median peak 20 vs 38).
   - 52% of ITI bout starts (gap >= 2 s) are single licks with no follow-up lick.
   - For those single licks, the mean tjM1 contra AUC in 0.5..1 s is about 0 (+0.7 all-ITI, +0.1 docked; SSp-n -0.3).
     With 3-5 follow-up licks it is +14.
   - So over 0..1 s the second half adds noise and a below-baseline lobe, not signal.
   - In sessions whose mean is a single-lick transient, the curve dips below baseline at 0.6-1.2 s, to about -8 to -14
     in the undefined rows. That pulls the 0..1 s integral to <= 0.
3. **Most undefined values are just zero within noise.** 92-96% of the non-positive AUCs are within 2 SEM of zero
   (per-lick SEM). Their median z is -0.5 to -0.6. They are small responses, not strongly negative ones. In 87-88% of
   them the 0..0.5 s part is positive and only the 0.5..1 s part is negative.
4. **The dip depends on epoch, so 0..1 s biases the pre reference.** Share of tjM1 sessions whose 0.5..1 s AUC is < 0:
   - all-ITI: pre 26%, acute 12%, subacute 0%, chronic 0%
   - docked ITI: pre 43%, acute 30%, subacute 18%, chronic 9%
   - The pre-session trough is negative in PS93 (-3.5 all-ITI / -4.8 docked) and PS94 (-0.6 / -2.1).

   The ITI transient is smallest in PS94 pre and PS93 pre: mean tjM1 contra 0..1 s AUC is 2.9 and 4.5 (all-ITI) vs
   10-33 in other cells. The 0..1 s integral is therefore pulled toward zero mostly in the **pre** sessions, which every
   change-vs-pre is measured against.
5. **The docked filter makes it worse because it keeps the smaller responses.**
   - Undefined share at 0..1 s is 0.035 all-ITI vs 0.076 docked.
   - Bout starts with the spout retracted ("not docked", removed by the filter) have larger 0..1 s AUCs: 15.3 vs about
     9.6 for docked ones.
   - Timing inside the docked set does not matter: 0..1 s AUC is 9.1-9.9 for every since-dock bin (0-0.5, 0.5-1,
     1-1.5, > 1.5 s) and every to-next-trial bin.
   - So neither the spout arrival nor the upcoming cue explains the undefined rows.
6. **A minimum lick count does not help.** Undefined rows have somewhat fewer licks (median 15 vs 17 docked, 26 vs 28.5
   all-ITI). But keeping only rows with >= 15 licks leaves the undefined share unchanged: 0.039 vs 0.042 all-ITI,
   0.095 vs 0.087 docked.

**Recommendation.** Use 0..0.5 s for the ITI AUC and ratio, as Priya decided.
- It removes 88% (all-ITI) and 86% (docked) of the undefined rows: iti 0.035 -> 0.004, docked 0.076 -> 0.011.
- The docked tjM1 spout x epoch model converges (p = 0.016) instead of failing.
- It measures the lick transient, which is the part of the ITI response that exists.

Where an index must be defined for every row, use the bounded index (c - i)/(|c| + |i|) over 0..0.5 s as a robustness
companion. Its stars and LR results match the 0..0.5 s log ratio almost exactly (section 3). Do not use a later
baseline (it doubles the undefined rows) or a lick-count floor (no effect).

## 1. Who is undefined (0..1 s unless stated)

| set | rows | undefined 0..1 s | undefined 0..0.5 s | side of the <= 0 AUC at 0..1 s (both / contra / ipsi) |
|---|---|---|---|---|
| cued | 3745 | 0 (0.000) | 0 | - |
| cue | 3745 | 20 (0.005) | 26 (0.007) | 20 / 0 / 0 (4 acute far-contra sessions, both sides ~0) |
| iti, all ITI | 3340 | 118 (0.035) | 14 (0.004) | 35 / 53 / 30 |
| iti, docked | 3015 | 229 (0.076) | 33 (0.011) | 81 / 66 / 82 |

The contra side is not singled out: 0..1 s undefined rows are about evenly contra-only, ipsi-only and both.

Undefined share by pair at 0..1 s:

| run | ALM | tjM1 | SSp-m | SSp-n | orofacial |
|---|---|---|---|---|---|
| all ITI | 0.001 | 0.042 | 0.039 | 0.073 | 0.021 |
| docked | 0.022 | 0.095 | 0.090 | 0.114 | 0.060 |

ALM barely ever fails because its ITI transient is broader.

Undefined share by spout (docked, 0..1 s): close spouts 0.11-0.18, far spouts 0.03-0.05, pooled 0.02. Close-spout cells
have fewer ITI licks per session. The all-ITI pattern is the same: close 0.06-0.08, far 0.01-0.03.

By animal x epoch, the problem is concentrated in PS93 and PS94. PS95 and PS92 rarely fail.

| docked ITI, all pairs, 0..1 s | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | 1/275 | 2/175 | 0/55 | 23/350 |
| PS93 | **42/240** | **21/140** | 3/105 | 5/325 |
| PS94 | **51/275** | 5/130 | **34/175** | **40/280** |
| PS95 | 2/95 | 0/20 | 0/60 | 0/315 |

Docked tjM1 (per-spout + pooled rows): 57 of 605 undefined at 0..1 s and 8 of 605 at 0..0.5 s.
- PS94 pre: 17 of 47 per-spout rows at 0..1 s, 6 of 47 at 0..0.5 s.
- With the side-group rows the counts become 74 of 875 overall and 30 of 79 for PS94 pre (the numbers in the question).

Lick counts of undefined vs defined rows (median): docked 15 vs 17; all-ITI 26 vs 28.5.

## 2. Cause

Each <= 0 AUC "side" was compared with the same side in defined rows (`neg_auc_characterize.py`; medians; dF/F x1e3
and dF/F x1e3 x s).

| docked ITI, 0..1 s | AUC | z (AUC / per-lick SEM) | peak 0..0.5 | AUC 0..0.5 | AUC 0.5..1 | slope -2..-0.5 s | raw baseline vs session mean |
|---|---|---|---|---|---|---|---|
| non-positive sides (310) | -2.3 | -0.63 | 7.9 | +1.7 | -4.6 | +2.6 /s | +1.1 |
| positive sides (5720) | +11.6 | 3.4 | 20.0 | +8.0 | +3.6 | -0.8 /s | -7.4 |

All-ITI is the same: AUC -1.5 vs +13.8, peak 8.6 vs 21.6, 0.5..1 s part -3.9 vs +4.9.

How much each candidate explains, as a share of the non-positive sides at 0..1 s (docked / all-ITI):

| candidate | evidence | share |
|---|---|---|
| (a) small response + noise | within 2 SEM of zero | **92% / 96%** |
| (c) dip after the transient in 0.5..1 s | AUC 0..0.5 > 0 and AUC 0.5..1 < 0 | **87% / 88%** |
| 0..0.5 s window instead | the AUC becomes positive | **87-88% of sides / 86-88% of rows** |
| (b) elevated or decaying baseline | an earlier baseline (-1.5..-1.0 s) makes the AUC positive | 48% / 49% |
| (b) elevated baseline, direct measure | raw baseline vs session mean: about +1 vs -7.4 in defined rows; signal rising into the baseline (+2.6 vs -0.8 /s) | contributes; not the main driver |
| (d) slow drift | removing the linear pre-trend (-2..-0.5 s) makes the AUC positive | 21% / 26% |
| (b/d) decay from earlier events into the baseline | signal higher at -2..-1.5 s and falling | 28% / 34%, **less** than in defined rows (55% / 62%), so not the cause |
| later baseline (-0.5..-0.25 s) | makes the AUC positive | 13% / 20%; it doubles the undefined rows overall |

**Reading.** The undefined rows are session x spout cells whose ITI response is a small single-lick transient. The 0..1 s
window adds the 0.5..1 s segment, and there those sessions dip below baseline. The dip and the small transient make the
AUC zero within noise, and the sign is then decided by noise.

The baseline in those rows also sits somewhat high, rising into -1..-0.5 s. That makes the dip look deeper and explains
part of it: an earlier baseline rescues about half. But the baseline is not the main driver: in event time there is no
dependence on spout-dock or next-trial timing.

What the dip is, is open. It could be post-excitation suppression or a hemodynamic-correction residual: both hemispheres
dip together. It is absent wherever licking continues, so in cued licks it is buried under the sustained bout response.

Event level (`neg_auc_events.py`, tjM1 contra, mean per bout start):

| following licks | share of bout starts | AUC 0..0.5 | AUC 0.5..1 | AUC 0..1 |
|---|---|---|---|---|
| 0 | 52% | 6.4 / 5.8 | **0.7 / 0.1** | 7.1 / 5.9 |
| 1-2 | 38-40% | 9.6 / 8.6 | 5.3 / 4.0 | 14.8 / 12.6 |
| 3-5 | 8-9% | 12.1 / 8.9 | 14.5 / 10.6 | 26.6 / 19.5 |

Values are all-ITI / docked. Within animal x epoch, corr(log #following licks, 0..1 s AUC) has a median of 0.39 (range
0.25-0.68; docked 0.33). So the 0..1 s ITI AUC mostly measures whether licking continues, not the size of the response
to the bout-start lick.

**Undershoot by set** (`neg_auc_undershoot.py`; one session = the spout-pooled row; contra + ipsi):

| set | tjM1 median peak 0..0.5 | tjM1 median trough 0.5..1.2 | share sessions trough < 0 | share sessions AUC 0.5..1 < 0 | share spout rows AUC 0.5..1 < 0 |
|---|---|---|---|---|---|
| cued (first cued lick) | 38.3 | +22.2 | 0.00 | 0.00 | 0.002 |
| cue-aligned | 33.9 | +26.0 | 0.00 | 0.00 | 0.004 |
| ITI, all | 20.8 | +6.1 | 0.18 | 0.09 | 0.16 |
| ITI, docked | 18.6 | +2.3 | 0.37 | 0.24 | 0.27 |

The pooled orofacial pair shows the same pattern: cued and cue 0 / 0; ITI 0.06 / 0.13; docked 0.14 / 0.21. Cued curves
dip slightly below baseline only at 2-3 s (PS95, PS94 pre), outside every window used.

Figures:
- `tjm1_iti_traces_allITI.png`, `tjm1_iti_traces_docked.png`: animals x epochs; thin = sessions, thick = mean; the
  baseline and both windows are shaded; the corner gives the undefined counts.
- `tjm1_iti_traces_docked_PS94_pre_perspout.png`: the problem case, per spout; undefined sessions are dashed.
- `tjm1_cued_traces.png`, `orofacial_cued_traces.png`, `tjm1_cue_traces.png`, `orofacial_cue_traces.png`.
- `neg_auc_curves_iti_{all,dock}.png`: mean curves of undefined vs defined rows, every pair.

## 3. Adjustments, evaluated without changing production

Each metric is recomputed on the same rows with the production test code:
- pre spout effect: mixed-model LR
- spout x epoch: LR
- epoch: LR
- change vs own pre: `analysis_kit.boot_delta`, per spout + spout-mean "all" + pooled, 3 epochs, 5 pairs

Full tables are in `neg_auc_compare.py` output.

Undefined share after each adjustment (per-spout rows, ITI, all-ITI / docked):

| metric | all-ITI | docked |
|---|---|---|
| 0..1 s production | 0.042 | 0.087 |
| 0..0.5 s | **0.005** | **0.013** |
| baseline -1.5..-1.0 s | 0.038 | 0.073 |
| baseline -0.5..-0.25 s | 0.084 | 0.132 |
| >= 15 licks (among kept rows) | 0.039 | 0.095 |
| bounded index / difference | 0 | 0 |

Change-vs-pre stars compared with production 0..1 s. Each set has 120 cells (3 epochs x 8 spout groups x 5 pairs).

| metric | cued stars (lost/gained) | cue | ITI all | ITI docked | starred cells with flipped sign |
|---|---|---|---|---|---|
| production log2 0..1 s | 82 | 74 | 40 | 42 | - |
| log2 0..0.5 s | 92 (6/16) | 70 (13/9) | **49 (2/11)** | **49 (3/10)** | 0 |
| bounded 0..0.5 s | 92 (6/16) | 73 (11/10) | 49 (2/11) | 49 (4/11) | 0 |
| bounded 0..1 s | 82 (0/0) | 73 (2/1) | 43 (1/4) | 46 (4/8) | 0 |
| c - i 0..1 s | 79 (17/14) | 71 (16/13) | 33 (14/7) | 42 (6/6) | 0 |
| baseline -1.5..-1.0 s | 75 | 68 | 21 (25/6) | 34 (12/4) | 0 |
| baseline -0.5..-0.25 s | 79 | 74 | 50 (3/13) | 45 (10/13) | 0 |
| >= 10 / >= 15 licks | 82 / 82 | 74 / 74 | 39 / 32 | 42 / 28 | 0 |

LR tests: how many of the 20 set x pair cells agree with production at p < 0.05.

| metric | pre spout | spout x epoch | epoch |
|---|---|---|---|
| log2 0..0.5 s | 17 | 16 | 19 |
| bounded 0..0.5 s | 17 | 15 | 19 |
| bounded 0..1 s | 17 | 16 | 19 |
| c - i | 17 | 12 | 18 |
| baseline -1.5..-1.0 s | 14 | 15 | 14 |

For cued and cue every metric agrees on the pre spout effect and the spout x epoch interaction (all p < 0.01). The
disagreements are all in ITI (section 4). (Cue spout x epoch is weakest at 0..0.5 s: ALM p = 2e-4, tjM1 p = 0.007; all other cued / cue cells p < 1e-6.)

Assessment:
- **The bounded index over 0..1 s** keeps the undefined rows (value near 0, or +-1 when one side is negative). It
  barely changes anything; it is defined everywhere but still carries the 0.5..1 s problem.
- **The difference c - i** is not scale-free. It weights high-amplitude sessions and animals, loses 14 ITI stars and
  agrees least on LR tests. Not recommended as primary.
- **An earlier baseline** loses the most stars (ITI 40 -> 21) and the epoch effect in SSp-m / SSp-n / orofacial: the
  -1.5 s window sits closer to the previous licks.
- **The lick floor** only removes rows, so ITI stars fall.

## 4. Does 0..0.5 s change an ITI conclusion compared with 0..1 s?

Direction: no reversal. No starred cell changes sign, under either run or any metric. The headline stands: the
contra/ipsi ratio **rises acutely** in every pair, both runs, at both windows. Spout-mean "all" and pooled, acute, 0..1 s
-> 0..0.5 s:

| pair | ITI all | ITI docked |
|---|---|---|
| tjM1 | +0.35* -> +0.23* | +0.72* -> +0.48* |
| orofacial | +0.38* -> +0.19* | +0.69* -> +0.41* |
| SSp-n | +0.31* -> +0.26* | +0.70* -> +0.45* |
| ALM | +0.28* -> +0.19* | +0.46* -> +0.32* |

Effect sizes are about 2/3 as large at 0..0.5 s. This is expected: the 0..1 s ratio in pre was compressed by the dip.

What changes:
1. **The pre-stroke spout effect on the ITI ratio becomes clear.** p values, 0..1 s -> 0..0.5 s:

   | pair | all-ITI | docked |
   |---|---|---|
   | ALM | 0.41 -> 1e-10 | 0.014 -> 4e-5 |
   | tjM1 | 0.006 -> 2e-10 (not converged; bounded index 1.5e-8) | **0.12 -> 2e-6** |
   | SSp-m | 6e-4 (not converged) -> 9e-7 | 0.09 -> 0.003 |
   | SSp-n | 2e-8 -> 3e-11 | 0.007 -> 0.001 |
   | orofacial | 4e-4 -> 4e-13 | 9e-4 -> 1e-3 |

   At 0..0.5 s the ITI ratio has a pre spout dependence like cued and cue. At 0..1 s it was partly washed out.
2. **The spout x epoch interaction (does the spout pattern change post-stroke) gets stronger for ITI.**
   - All-ITI, 0..1 s: significant only for orofacial (0.027) and SSp-m (0.031, not converged).
   - All-ITI, 0..0.5 s: SSp-m 0.001, SSp-n 0.001, orofacial 0.001, tjM1 0.014; ALM stays ns (0.20).
   - Docked, 0..1 s: ALM 0.021, orofacial 0.016, tjM1 **failed (LR not finite)**.
   - Docked, 0..0.5 s: orofacial 0.005, tjM1 **0.016 (converged)**; ALM 0.084, SSp-m 0.050, SSp-n 0.081.

   Net effect on ITI at p < 0.05: all-ITI gains three pairs (SSp-n, orofacial now firm, tjM1). Docked gains tjM1
   and loses ALM (0.021 -> 0.084).
3. **Per-spout stars, change vs pre.**
   - All-ITI: 40 -> 49. Lost: ALM far-mid chronic, tjM1 far-mid subacute. Gained 11, mostly subacute and close-contra
     cells.
   - Docked: 42 -> 49. Lost: ALM close-contra chronic, ALM far-contra subacute, tjM1 close-mid subacute. Gained 10,
     mostly acute SSp-n / tjM1 / SSp-m cells.
   - No cell changes sign.

Slide notes that need updating with the window switch:
- slides 135-138 (lick_lateral_ratio_auc, ITI rows / panels)
- slide 203 (docked ITI ratio)
- the ITI side of slides 105, 108 (ITI ramp, AUC 0..1 s) and 109 (ITI peak; unaffected by the window)
- per-spout ITI AUC on slides 113 / 114, if they show ITI

The p values and stars above are the numbers to expect.

## 5. Other 0..1 s results that use these traces (report only, nothing changed)

`neg_auc_windows.py` compares the existing production test tables at 0..1 s vs 0..0.5 s:
- `ramp_window_tests.csv`: slides 105-109, normalized response AUC, change vs pre.
- `perspout_auc_tests.csv`: per-spout AUC change vs pre, behind slides 113-114.

| table / set | cells | stars 0..1 s | stars 0..0.5 s | only 0..1 s | only 0..0.5 s | starred with opposite sign |
|---|---|---|---|---|---|---|
| ramp window, cued | 21 | 17 | 18 | 0 | 1 | 0 |
| ramp window, ITI (all) | 21 | 16 | 19 | 0 | 3 | 0 |
| ramp window, ITI (docked) | 21 | 9 | 14 | 0 | 5 | 0 |
| per-spout AUC, cued lick | 72 | 49 | 60 | 0 | 11 | 0 |
| per-spout AUC, cue | 72 | 54 | 55 | 4 | 5 | 0 |
| per-spout AUC, ITI (all) | 72 | 48 | 59 | 0 | 11 | 0 |
| per-spout AUC, ITI (docked) | 72 | 25 | 46 | 0 | 21 | 0 |

- **ITI.** The 0..1 s per-spout and window results are **normalized by the animal's pre response over the same window**.
  The pre 0..1 s ITI AUC is the quantity that sits near zero for PS93 / PS94, so these changes blow up with very wide
  CIs. Examples, docked tjM1 change vs pre: far-ipsi acute +11.2 [-0.04, 33.9]; close-ipsi chronic -5.8 [-23.7, 2.2].
  At 0..0.5 s the same cells are +1.0 [0.2, 2.3] and +0.8 [0.2, 1.7].
  0..0.5 s only adds stars; no conclusion reverses. This is the same mechanism as the ratio, and a second reason to
  move ITI to 0..0.5 s.
- **Cued and cue.** 0..1 s vs 0..0.5 s barely matters. No sign changes. The cued per-spout table gains 11 small
  stars at 0..0.5 s (ALM / cortex chronic, far-mid acute). Keeping cued at 0..1 s is safe.
- Not checked here: other 0..1 s families that do not read these traces (matched-gauss readouts, LMM time readouts).
  They use their own windows.

## Files

- Repo, this folder: `ITI_NEG_AUC_DIAGNOSTIC.md`, `neg_auc_diag.py`, `neg_auc_analyze.py`,
  `neg_auc_characterize.py`, `neg_auc_events.py`, `neg_auc_undershoot.py`, `neg_auc_compare.py`,
  `neg_auc_windows.py`, `neg_auc_tjm1_traces.py`.
- Intermediate tables (scratch, not in the repo): `features.pkl`, `alt_tests.csv`, `alt_models.csv`,
  `compare_boot.csv`, `char.txt`, `events.txt`, `undershoot.txt`, `windows.txt`.
- How to rerun: run `neg_auc_diag.py --extract`, then `--events` under the env of `l_lateral_ratio_g.log` (+
  `WFIELD_ITI_FILTER=dock` for docked), then `--analyze <all.pkl> <dock.pkl> <out> <figdir>`. About 2 min for the
  extracts and about 50 min for the alternative-metric tests (bootstrap x 9 metrics).

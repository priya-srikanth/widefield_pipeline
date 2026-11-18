# Quiet-period events: common k = 4 and the figure-convention restyle, star changes (a8, 2026-11-18)

**For Priya (relayed by 3c); nothing on the deck or the movement page changes until she has seen this.**
Requests (11-18, via 3c): "use k = 4 for all"; restyle the quiet figures (bootstrap stars primary in the top band, mixed
model as a dagger backup, no top spine, open session markers, change vs own pre); one combined star-change list.

## 1. k 3.75 -> k 4 (the base summary only)
- What moves: ONLY the base summary of the cam1-guard run (pooled-epoch rates, `lmm_v2.csv`, the threshold figure).
  Every follow-on script (face-regressed, 415 check, stillness, task windows, decay, maps, examples, movie) already used
  its own fixed k = 4 (checked in the code: `K = 4.0` / `K_EVT = 4.0`), so their numbers do not change.
- Run: `v2_quiet_lp_c1guard_k4_20261118/` = `quiet_spikes_v2.py --summarise --k 4` on the guard per-session data
  (sessions/ copied from `v2_quiet_lp_c1guard_20261117`), code = guard f3484347; complete (108 sessions; same file set
  as the k 3.75 summary; limb also at k 4). Runner `quiet_c1guard/run_quiet_v2_k4.sh`.
- Effect: every rate 10-20 % lower (pre included), changes vs own pre a little smaller, same directions. Quiet minutes
  identical. Full per-animal tables (orofacial / limb, positive and both polarities, change vs own pre) and every LMM term
  old -> new: `COMPARE_k375_vs_k4.md` (this folder).
  - Orofacial both polarities, per-animal acute change vs own pre (events / min): PS92 26.7 -> 24.9, PS93 37.5 -> 35.2,
    PS94 21.8 -> 19.9, PS95 21.5 -> 19.3.
  - LMM (log10 rate + 0.2), orofacial both: acute .663 -> .692, subacute .411 -> .424, chronic .201 -> .197 (q .011 -> .019).
- **Star changes from k alone (mixed model, FDR): one.** Limb both-polarity chronic q .063 -> .048 (gains a star).
  Under the bootstrap (section 2) that cell has no star either way ([-0.22, +17.2] events / min, 3/4).

## 2. Mixed-model stars -> bootstrap stars (all families, at k = 4)
- New star `*` = `analysis_kit.boot_delta` (nested animals -> sessions, animal-weighted) 95 % CI of the change vs each
  animal's own pre excludes 0. Dagger = the script's own mixed model q < .05 (the q the old figures starred). Rates:
  change in events / min (the mixed model used log10(rate + 0.2), so skewed chronic rises can disagree); 415 check SD /
  power / rate: change of log10 (fold vs own pre).
- T13 is a within-session contrast (event trials minus the same-session strata mean), so its star tests the pooled
  per-session excess against 0 in each epoch (`boot_ci`, nested); before = the trial-pooled within-stratum permutation p
  (uncorrected). Per-session excess = STRICT strata (spout x trial-order quartile x clean-time bin); by-spout panels =
  session x spout. Footer now says: PS93_0813 excluded (no video): 31 pre sessions.
- Stillness time share had no test before: its 6 "gained" cells are new tests, not reversals.

| family | cells | stars before | stars after | gained | lost |
|---|---|---|---|---|---|
| pooled | 24 | 14 | 12 | 0 | 2 |
| fr | 12 | 12 | 10 | 0 | 2 |
| fr_detail | 48 | 30 | 31 | 4 | 3 |
| hemo | 45 | 16 | 20 | 5 | 1 |
| stillness | 30 | 10 | 15 | 7 | 2 |
| t12 | 24 | 23 | 20 | 0 | 3 |
| decay | 12 | 4 | 4 | 0 | 0 |
| t13 | 64 | 37 | 17 | 2 | 22 |

| panel | epoch | before | q / p before | after | change [95% CI] | animals in direction |
|---|---|---|---|---|---|---|
| pooled EITHER polarity orofacial contralesional | chronic | * | 0.0453 | - | 3.87 [-1.03, 11.7] (difference) | 3/4 |
| pooled EITHER polarity limb contralesional | chronic | * | 0.0481 | - | 7 [-0.221, 17.2] (difference) | 3/4 |
| fr res_both orofacial contralesional | chronic | * | 0.0126 | - | 2.7 [-1.47, 9.29] (difference) | 3/4 |
| fr res_both limb ipsilesional | acute | * | 0.0181 | - | 10.3 [-0.00683, 22.7] (difference) | 3/4 |
| fr_detail raw_both orofacial contralesional | chronic | * | 0.037 | - | 3.65 [-1.43, 11.6] (difference) | 3/4 |
| fr_detail raw_both limb ipsilesional | subacute | - | 0.167 | * | 2.37 [0.044, 5.41] (difference) | 3/4 |
| fr_detail raw_both limb contralesional | subacute | - | 0.136 | * | 3.26 [0.12, 7.32] (difference) | 4/4 |
| fr_detail raw_both limb contralesional | chronic | * | 0.0426 | - | 7.11 [-0.163, 17.3] (difference) | 3/4 |
| fr_detail res_both orofacial contralesional | chronic | * | 0.0126 | - | 2.7 [-1.43, 9.17] (difference) | 3/4 |
| fr_detail move_frac orofacial ipsilesional | subacute | - | 0.0564 | * | -0.147 [-0.261, -0.0452] (difference) | 4/4 |
| fr_detail r2 orofacial ipsilesional | acute | - | 0.718 | * | -0.0626 [-0.112, -0.016] (difference) | 4/4 |
| hemo corr470 p0510 | subacute | - | 0.0574 | * | 0.224 [0.0612, 0.396] (log10 fold) | 4/4 |
| hemo corr470 rate | chronic | * | 0.0442 | - | 0.187 [-0.00834, 0.401] (log10 fold) | 3/4 |
| hemo raw470 rate | subacute | - | 0.0654 | * | 0.232 [0.0983, 0.383] (log10 fold) | 4/4 |
| hemo raw415 sd | subacute | - | 0.0909 | * | 0.0405 [0.00922, 0.0775] (log10 fold) | 4/4 |
| hemo raw415 p0510 | acute | - | 0.336 | * | -0.178 [-0.324, -0.0316] (log10 fold) | 4/4 |
| hemo raw415 eta | acute | - | 0.223 | * | 0.191 [0.0166, 0.465] (difference) | 4/4 |
| stillness rate bin 0.25-0.5 | chronic | ** | 0.00298 | - | 11.5 [-1.86, 34.6] (difference) | 4/4 |
| stillness rate bin 0.5-1 | chronic | * | 0.0379 | - | 3.16 [-1.9, 10.9] (difference) | 3/4 |
| stillness rate bin 2-inf | acute | - | 0.0658 | * | 4.34 [0.342, 10.4] (difference) | 3/4 |
| stillness share bin 0-0.25 | acute | - |  | * | 0.144 [0.0525, 0.232] (difference) | 4/4 |
| stillness share bin 0-0.25 | chronic | - |  | * | -0.0633 [-0.122, -0.0195] (difference) | 4/4 |
| stillness share bin 0.25-0.5 | acute | - |  | * | 0.116 [0.0638, 0.168] (difference) | 4/4 |
| stillness share bin 0.5-1 | subacute | - |  | * | 0.0366 [0.00735, 0.0661] (difference) | 4/4 |
| stillness share bin 1-2 | acute | - |  | * | -0.15 [-0.228, -0.0754] (difference) | 4/4 |
| stillness share bin 2-inf | acute | - |  | * | -0.104 [-0.151, -0.0618] (difference) | 4/4 |
| t12 precue orofacial contralesional | chronic | * | 0.0263 | - | 4.9 [-0.393, 14.4] (difference) | 3/4 |
| t12 iti orofacial contralesional | chronic | *** | 3.49e-05 | - | 4.54 [-0.0301, 12.2] (difference) | 3/4 |
| t12 quiet orofacial contralesional | chronic | * | 0.0322 | - | 3.87 [-1.03, 11.7] (difference) | 3/4 |
| t13 farcontra_fail | subacute | - | 0.0589 | * | 0.0402 [0.00842, 0.0687] (level (vs 0)) | 4/4 |
| t13 p_true_enl_event | subacute | ** | 0.002 | - | -0.0178 [-0.0373, 0.00235] (level (vs 0)) | 3/4 |
| t13 allspout_miss close_ipsi | pre | * | 0.026 | - | 0.0073 [-0.00682, 0.0296] (level (vs 0)) | 2/4 |
| t13 allspout_miss close_ipsi | subacute | * | 0.034 | - | 0.0129 [-0.00273, 0.0515] (level (vs 0)) | 3/4 |
| t13 allspout_miss close_mid | pre | * | 0.023 | - | 0.00719 [0, 0.0177] (level (vs 0)) | 3/4 |
| t13 allspout_miss close_mid | acute | ** | 0.002 | - | 0.0216 [-0.0067, 0.0547] (level (vs 0)) | 3/4 |
| t13 allspout_miss close_mid | chronic | *** | 0.000999 | - | 0.0149 [0, 0.0436] (level (vs 0)) | 3/4 |
| t13 allspout_miss close_contra | pre | * | 0.031 | - | 0.00736 [-0.00179, 0.0232] (level (vs 0)) | 2/4 |
| t13 allspout_miss close_contra | subacute | *** | 0.000999 | - | 0.0507 [-0.00487, 0.116] (level (vs 0)) | 2/4 |
| t13 allspout_miss close_contra | chronic | * | 0.05 | - | 0.00952 [-0.00121, 0.0356] (level (vs 0)) | 2/4 |
| t13 allspout_miss far_mid | pre | * | 0.022 | - | 0.0144 [-0.0138, 0.0521] (level (vs 0)) | 3/4 |
| t13 allspout_miss far_mid | subacute | *** | 0.000999 | - | 0.0673 [-0.0123, 0.131] (level (vs 0)) | 2/4 |
| t13 allspout_miss far_mid | chronic | ** | 0.00599 | - | 0.0188 [-0.011, 0.0666] (level (vs 0)) | 2/4 |
| t13 allspout_miss far_contra | subacute | ** | 0.002 | - | 0.0854 [-0.01, 0.176] (level (vs 0)) | 2/4 |
| t13 allspout_miss far_contra | chronic | ** | 0.003 | - | 0.0306 [-0.0017, 0.0951] (level (vs 0)) | 3/4 |
| t13 p_true_enl_event close_ipsi | pre | ** | 0.003 | - | -0.0356 [-0.0855, 0.0298] (level (vs 0)) | 3/4 |
| t13 p_true_enl_event close_ipsi | subacute | *** | 0.000999 | - | -0.1 [-0.155, 0.00136] (level (vs 0)) | 3/4 |
| t13 p_true_enl_event close_ipsi | chronic | ** | 0.002 | - | -0.0338 [-0.081, 0.00336] (level (vs 0)) | 4/4 |
| t13 p_true_enl_event close_mid | chronic | * | 0.021 | - | -0.0301 [-0.0847, 0.00797] (level (vs 0)) | 2/4 |
| t13 p_true_enl_event close_contra | subacute | * | 0.014 | - | -0.0494 [-0.11, 0.000605] (level (vs 0)) | 4/4 |
| t13 p_true_enl_event far_ipsi | pre | * | 0.016 | - | -0.0455 [-0.111, 0.0135] (level (vs 0)) | 3/4 |
| t13 p_true_enl_event far_ipsi | chronic | *** | 0.000999 | - | -0.0148 [-0.0522, 0.0394] (level (vs 0)) | 3/4 |
| t13 p_true_enl_event far_mid | subacute | - | 0.0899 | * | 0.0348 [0.000613, 0.065] (level (vs 0)) | 3/4 |
| t13 p_true_enl_event far_contra | pre | ** | 0.00699 | - | 0.03 [-0.022, 0.0657] (level (vs 0)) | 3/4 |

## 3. What the changes mean (reading, for Priya)
- **The acute rise stands:** every acute orofacial event-rate cell (pooled, face-regressed, raw, T12 windows) keeps its star, 4/4.
- **Chronic contralesional rises lose their star** under the bootstrap (pooled both polarities, face-regressed, T12
  pre-cue / ITI / quiet): carried by 3/4 animals with a skewed spread; the mixed model on log rate starred them.
- **Stillness:** the acute rise now holds in every bin including the most active one (was q .066); the chronic bin stars
  go. Time share: acutely more time in the two stillest bins and less in the two most active (4/4 each).
- **415 nm:** the response at corrected-470 events is above pre acutely by the bootstrap (+0.19 [0.02, 0.47] dF/F %, 4/4;
  mixed model q .22): up to a fifth of the acute amplitude could be hemodynamic or motion. 415 0.5-1 Hz power falls
  acutely (log10 -0.18 [-0.32, -0.03], 4/4; 1-4 Hz n.s.) and 415 SD rises slightly subacutely (+0.04 log10, 4/4); corrected
  470 0.5-1 Hz subacute now starred (was q .057).
- **T13:** 22 of the 37 old permutation stars do not survive the session-level test, mostly the by-spout panels
  (uncorrected pooled permutations). The decoder p_true excess on event trials stays starred pre / acute / chronic;
  subacute loses it. Far-contra failure excess subacute gains one (+0.040, 4/4).
- Face-movement share of raw events: subacute ipsilesional now starred (fewer events with movement, 4/4).

## 4. Where the base summary is quoted
- Movement (Execution) page (formerly goal page), section 23: only **Fig 23.18** (threshold choice), already the k 4
  render (`v2_quiet_lp_c1guard_k4_20261118/threshold_choice.png`). No other figure or sentence in section 23 draws on
  `quiet_events_sessions.csv` / `lmm_v2.csv`; the other section-23 figures come from the follow-on tables (k 4 already).
- PI deck: no slide or note quotes the base summary; slides 155-159 and 204 cite follow-on tables only.

## 5. Restyled figures (not yet on the page or deck)
`<rebuild>/labcams/<tag>/cohort1_quiet_spikes/v2_quiet_lp_c1guard_k4_20261118/restyled_20261118/`:
quiet_events_pooled_epochs_restyled, quiet_events_face_regressed_restyled (q_fr), quiet_events_face_regressed_detail_restyled
(q_fr_detail), quiet_hemo_check_restyled (q_hemo415), quiet_stillness_restyled (q_stillness),
t12_task_window_event_rates_restyled (q_task_rates), quiet_eta_decay_fr_restyled (q_decay_fr),
t13_precue_event_vs_outcome_restyled (q_task_outcome); `.png` + `svg/`; cell table `star_changes_all.csv` (every cell:
point, CI, per-animal changes, old q, old / new star); `t13_session_excess.csv`; `restyle.log`.
Script: `scripts/cohort2_qc/quiet_restyle_figs.py` (reads the saved tables only; 75 s).
Follow-up if adopted: the deck crops for slides 155-159 / 204 need re-cutting from these renders (different layout).

# Fig 12.1 (rest15f, RESTnlc1 v3) — scored-session floor 30 vs 40 (2026-11-18)

Priya 11-18 asked how the 40-period minimum was chosen. Answer: a default set when the frozen rest decoder was built
(06ca7732, 09-16); no measurement chose 40. Sensitivity run (Priya OK): post-stroke scored sessions need >= 30
lick-free rest periods instead of 40 (`WF_REST_SCORE_MIN_POST=30`, da81882e). Pre sessions, training pools, LOSO and the
stored frozen models are unchanged (all four animals `frozen-hit`). Script `rest15f_restnl_1117.py --lp c1all`.
Outputs: `<rebuild>/labcams/grant_figures/rest15f_floor30_sensitivity_20261118/{all,aug}pre/{all,aug}pre/`.

| June+Aug pre | floor 40 (page) | floor 30 |
|---|---|---|
| sessions acute / sub / chronic | 13 / 16 / 42 | 16 / 16 / 43 |
| acute change | -0.120 [-0.188, -0.037] * | -0.122 [-0.194, -0.040] * |
| subacute | -0.030 [-0.118, +0.092] | -0.030 [-0.118, +0.092] |
| chronic | -0.081 [-0.153, -0.005] * | -0.083 [-0.154, -0.007] * |

- Added sessions: PS93 acute 0819 (37 periods), 0820 (31), 0821 (33); one PS92 chronic session. PS93 acute retained
  -0.09 (1 session) -> -0.14 (4 sessions); per-session above-chance -0.013 / +0.042 / -0.009 / -0.105.
- PS92 subacute stays ONE session: 0826 has only 22 duration- and gap-matched periods in trained classes.
- August-only pre: no session added; identical to floor 40 (acute *, subacute n.s., chronic *).
- Conclusion: no call depends on the floor.

## 11-18: floor 30 is the PRIMARY Fig 12.1 (Priya)
- `rest15f_restnl_1117.py --score-min-post N` (default 30) sets the scored POST-stroke floor; it reaches the fit workers
  in the item and `_fit_job` sets `rest_frozen_decoder_nl.SCORE_MIN_POST` (that module's own default stays 40). A floor
  other than 40 goes into the folder (`<base>_floor30_20261118`), the file stem (`..._durgap_floor30`), the suptitle,
  the confusion title and figdata.json (`score_min_post`).
- Renders (all four animals `frozen-hit`), under `<rebuild>/labcams/grant_figures/<primary tag>/epoch/`:
  - `rest15f_restnlc1all_floor30_20261118/allpre/rest15f_restnlc1v3wk_allpre_durgap_floor30.png` (+ `_confusion.png`)
  - `rest15f_restnlc1all_floor30_20261118/augpre/rest15f_restnlc1v3wk_augpre_durgap_floor30.png`
  - `rest15f_restnl_cam4_floor30_20261118/allpre/rest15f_restnlwk_allpre_durgap_floor30.png` (sensitivity)
- Numbers match the sensitivity run above. Per animal retained (acute / subacute / chronic): PS92 .99 / 3.07 / 1.30,
  PS93 -.14 / .42 / .52, PS94 .41 / .43 / .14, PS95 .13 / .70 / .55. Sessions 34 / 16 / 16 / 43.
- cam4-only at floor 30: no session added (113 sessions, as at floor 40; PS92_0826 has 23 matched periods), numbers
  identical to 11-17. Its collection re-ran (cache key 3dcede1404f2 -> 394e74b10f95: `lp_licks.cache_tag` changed with
  the 11-17 cam1 keep_mask commit), 15 min.
- Cache note (c1all): the 11-17 run read cache key dd2b5743d7d5; today's runs read 3e0e547a2d7d (re-collected 11-18
  06:57 after the restnl_licks digest change). They differ in 5 sessions: working flags of PS94_0817 / 0823 / 0831 and
  PS95_0924, and PS94_0910 has 119 instead of 136 periods. A floor-40 run on today's cache reproduces every cohort
  number of the 11-17 floor-40 run; only PS95_0924 changes (n 181 -> 179, acc .3692 -> .3696). Why the working flags
  moved between 11-17 20:46 and 11-18 06:57 was not traced.

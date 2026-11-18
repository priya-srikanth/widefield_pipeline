# Fig 12.1 (rest15f, RESTnlc1 v3, floor 30) — central 90 % vs central 80 % pre matching (2026-11-18)

Priya 11-18: "why central 80 %? what if central 90 %?" The 10 % per tail is the default set 09-16 (06ca7732); no
measurement chose it. Sensitivity: `WF_REST_MATCH_PCT=5` (9e9d29ab) trims 5 % per tail of the pre duration and lick-gap
distributions, for training AND scoring; the percentile is in the frozen align, so NEW frozen models (`frozen-new`,
`dur5-gap5`; the dur10 models are untouched). Post-stroke floor 30 as the new primary. Outputs:
`<rebuild>/labcams/grant_figures/rest15f_floor30_pct5_sensitivity_20261118/{all,aug}pre/{all,aug}pre/`.

| June+Aug pre, change vs own pre | central 80 % (primary) | central 90 % |
|---|---|---|
| sessions pre / acute / sub / chronic | 34 / 16 / 16 / 43 | 35 / 16 / 16 / 43 |
| acute | -0.122 [-0.194, -0.040] * | -0.139 [-0.200, -0.070] * |
| subacute | -0.030 [-0.118, +0.092] n.s. | -0.069 [-0.128, -0.008] * |
| chronic | -0.083 [-0.154, -0.007] * | -0.098 [-0.170, -0.024] * |
| retained (animal-weighted) acute / sub / chr | 0.35 / 1.16 / 0.63 | 0.26 / 0.66 / 0.52 |

- Per animal retained, central 90 %: PS92 0.68 / 1.15 (1 session) / 1.02; PS93 -0.16 / 0.32 / 0.31; PS94 0.35 / 0.46 / 0.13;
  PS95 0.16 (1 session) / 0.72 / 0.61. The subacute change is mostly PS92 (3.07 -> 1.15 on its one session, 0824) and PS93.
- PS92 subacute 0826 still out (27 matched periods).
- August-only pre: PS92 now passes the fairness gate (3 animals instead of 2). Acute -0.091 *, subacute -0.033 n.s.,
  chronic -0.090 *.
- Reading: acute and chronic drops hold under both windows; the subacute call depends on the window (n.s. at 80 %,
  significant at 90 %).

## 11-18: floor 30 + central 90 % is the PRIMARY Fig 12.1 (Priya)
- `rest15f_restnl_1117.py --score-min-post 30 --match-pct 5` (both now the script defaults): the pair travels in the fit
  item and `_fit_job` / `main` set `nl.SCORE_MIN_POST` and `nl.PCT` (`_apply_cfg`); `rest_frozen_decoder_nl` keeps its
  own defaults (40, 10). Non-default settings are named in the folder, stem, suptitle, confusion title and figdata.json.
- Renders under `<rebuild>/labcams/grant_figures/<primary tag>/epoch/`:
  - `rest15f_restnlc1all_floor30_pct5_20261118/allpre/rest15f_restnlc1v3wk_allpre_durgap_floor30_pct5.png` (+ `_confusion`)
  - `rest15f_restnlc1all_floor30_pct5_20261118/augpre/rest15f_restnlc1v3wk_augpre_durgap_floor30_pct5.png`
  - `rest15f_restnl_cam4_floor30_pct5_20261118/allpre/rest15f_restnlwk_allpre_durgap_floor30_pct5.png` (sensitivity)
  c1all all-pre and aug-pre: every animal `frozen-hit` (the dur5 models of the sensitivity run). cam4: `frozen-new` for
  all four (first cam4 dur5-gap5 models, now stored).
- All-pre numbers equal the sensitivity table above. August-only (3 animals): acute -0.091 [-0.178, -0.006] *, subacute
  -0.033 [-0.101, +0.049], chronic -0.090 [-0.150, -0.017] *; sessions 19 / 12 / 13 / 31.
- **cam4-only at these settings does NOT give the same calls**: acute -0.135 [-0.202, -0.066] *, subacute -0.049
  [-0.111, +0.024] n.s., chronic -0.077 [-0.150, +0.003] n.s. (sessions 40 / 16 / 16 / 43; PS92_0826 28 matched periods).
- Floor 40 at central 90 % (scratch run, not published): acute -0.125 [-0.182, -0.063] *, subacute -0.069 [-0.127, -0.008]
  *, chronic -0.098 [-0.170, -0.024] * (14 acute sessions): no outcome changes.

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

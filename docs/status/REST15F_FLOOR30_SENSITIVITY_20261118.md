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

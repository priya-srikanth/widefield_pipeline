# State-space Figs 6.3 / 6.4, SVD + LocaNMF bases: glue-free re-run (11-17)

Priya OK'd 11-17. Outputs: `rebuild_20261103/state_space_recovery/svd_lnmf_gluefree_20261117/` (run log `logs/RUN_LOG.md`,
old-vs-new tables `compare/`, figures `page_figs/`, replay of the current code on the 11-08 tables `replay_1108_inputs/`).

**Result: nothing changes.** Re-running recovery_trials -> trajectories -> pertrial -> step_vs_gradual -> per_position
(lick, `WF_SS_BASIS=svd` and `lnmf`) reproduces the 11-08 input tables byte for byte, and the figures are byte-identical
to the page v40 versions (`page_figs_nullfix_20261116/`). The 11-08 tables were already glue-free: the cohort SVD basis
(`ss_svdbm_20261108`) was built on the current glue-free brain mask (`S._check_svd_mask` passes), and the LocaNMF
components dropped on 11-08 are exactly those the 11-16 mask-keyed keep files drop (all 4 animals). The 11-09 glue bug
(empty `painted_exclusion` under the rebuild profile) affected the Allen extract only; this arm resolved masks without
the profile. The "11-08 pre-glue" HOLD on SS SVD / LocaNMF slides (provenance_sweep_20261116/peer_3c_1116.md) was a
file-date flag; it can be lifted for Figs 6.3 / 6.4 if Priya agrees.

The per-spout partial r on log(day) quoted in the Fig 6.4 caption is not written by `state_space_per_position.py`;
`compare_fig63_64.py` recomputes it (residualize readout and aim on [1, ln day] within animal and spout, post-stroke) and
reproduces the Allen caption numbers exactly (x: PS92 .58-.74, others .01-.47).

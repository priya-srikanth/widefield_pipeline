# Quiet-period events: cam1 far-from-mouth guard, old vs new (a8, 11-17)

**Why:** Priya OK'd the re-run via 3c. Under the new rule, cam1-only events count as licks only if they are non-droplet
AND med_out_px >= -120. The audit is `CAM1_FARMOUTH_BLAST_RADIUS_20261117.md` on branch analysis/restnl-cam1-1117.

**Runs:**
- **Code:** guard worktree wf_c1guard1117, branch fix/cam1-farmouth-guard-1117. HEAD at run time was 0100aabf; its
  code is identical to 3c's 95ba249b (only one test file differs).
- **Env:** replayed from the production run (v2_quiet_lp_20261117/logs/quiet_v2.log). A diff confirmed that only the
  code path and the output directory differ.
- **Outputs**, under `rebuild_20261103/labcams/<tag>/cohort1_quiet_spikes/`:
  - `v2_quiet_lp_c1guard_20261117`: the full re-run (108/108 sessions, 14 min). Its leave-one-animal-out threshold
    choice moved orofacial k from 3.75 to 4.5.
  - `v2_quiet_lp_c1guard_k375_20261117`: the same per-session data, summarized at the PRODUCTION orofacial k (3.75).
    `--k` sets one common k, so its limb numbers are at 3.75 and are NOT comparable. Its BH q-values are also not
    comparable, because the limb terms in the same family changed; use p.
- **Scripts:** `run_quiet_v2.sh`, `run_quiet_v2_k375.sh`, `compare_old_new.py`. Tables are in `COMPARE_OLD_NEW.md`
  (re-chosen k) and `COMPARE_OLD_NEW_k375.md` (fixed k).

## 1. The guard itself (same k = 3.75): essentially no change
- **Quiet time:** +0.0 to +0.1 min per animal x epoch. By session: PS92_0929 +52 % (1.99 -> 3.03 min), PS93_0929
  +11 %, PS93_1002 +9 %, PS93_0812 (pre) +8 %, PS94_0903 +6 %, the rest <= 3 %.
- **Orofacial rate, change vs own pre, chronic, old -> new (events/min):** PS92 +8.50 -> +8.49, PS93 -0.21 -> -0.20,
  PS94 -1.87 -> -1.80, PS95 +1.12 -> +1.14. Acute and subacute move by < 0.3.
- **Orofacial LMM (log rate), old -> new:**
  - coefficients: acute .585 -> .586, subacute .359 -> .362, chronic .126 -> .130
  - p: chronic p .124 -> .111; contralesional (left) chronic p .018 -> .017; both polarities chronic p .004 -> .004
- **Limb (k 8.75 in both full runs):** per-animal means move by <= 0.15 events/min (PS92 chronic 6.87 -> 6.72);
  LMM chronic .608 -> .607.

## 2. The full re-run with the threshold re-chosen (k 4.5)
- **Why k moved:** the k choice is the median of the 4 leave-one-animal-out choices. Two folds moved from 3.5 to 4.5
  (held-out PS92, PS95), and the pooled AUC is flat across 3.5-4.5 (.977 / .979 / .979 -> .980). So the move comes
  from a flat AUC curve, not from the data changing.
- **Effect of k 4.5:** every orofacial rate is lower, pre included (e.g. PS92 pre 4.76 -> 2.51). Changes vs own pre
  shrink in absolute size. Log-rate LMM: acute .69, subacute .41, chronic .14 (p .22); contralesional chronic .19
  (p .023); both polarities chronic .21 (p .020). Same direction everywhere.
- **Quiet minutes LMM:** q acute .02 -> .34 with the same coefficients (5.7 -> 4.8, -7.2 -> -7.2). The random-effect
  structure switched from a random intercept (~1) to a per-animal slope (~post), which roughly doubles the SE.
  lmm_table picks the structure by fit, so a tiny data change flipped it. This is a model-selection artifact, not a
  change in quiet time.

## Conclusion (for Priya, before any page / deck change)
- The guard does not change the quiet-event result. At the production threshold, every per-animal number and model
  term moves by under 2 % of its size.
- The only material differences come from (a) the leave-one-animal-out threshold choice flipping 3.75 -> 4.5 on a
  flat AUC curve, and (b) the quiet-minutes LMM switching its random-effect structure.
- Options:
  - keep k 3.75 and update captions only (the lick rule and code), or
  - adopt the re-chosen k 4.5, which re-renders the figures with lower absolute rates and the same conclusions.
- Page section 23 and deck slides 157 / 158 are NOT changed. 3c also listed slide 112.

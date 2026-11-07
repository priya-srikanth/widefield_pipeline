# CURBD pilot: stability criterion, written BEFORE any real-data fit (2026-10-30)

Code: `scripts/curbd_pilot.py`, `wfield_local/curbd_core.py`. Outputs:
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/curbd_pilot/`.

Pilot: PS94 pre-stroke 0807, 0811, 0812 (448 / 399 / 455 engaged trials); chronic candidate 0910 (day 25).
Primary variant = "multi" (4 spatial units per region, 18 regions = 72 units; regions SSp-n, SSp-m, SSs, SSp-ul, tjM1, ALM, MOp_rest, MOs_rest, RSPd x 2 hemispheres -- RSPd is the posterior control because VISp_left has 0 px on the stat mask; SSs is small: 179 / 130 px). Matrix compared =
mean |current| (target <- source), OFF-DIAGONAL entries (the 306 between-region pairs), Pearson r.

The gate (all on the primary variant; the one-unit variant is reported, not gated):

1. **Fit quality.** Free-pass pVar on the training data must beat the hold-at-reset baseline (rate held at
   the value it was reset to) by >= 0.05, AND the held-out-half pVar (J trained on one half, free-run on the
   other) must beat the held-out hold baseline. Otherwise the currents describe a model that does not
   predict the data better than persistence, and nothing downstream is interpreted.
2. **Seeds.** Median pairwise r across 3 seeds >= 0.8 (per session; median over sessions).
3. **Halves.** r(first half, second half) >= 0.7 median over the 3 pre sessions, and no session < 0.5.
4. **Sessions.** Median pairwise r across the 3 pre sessions >= 0.7.
5. **Pair-specific structure.** The same half-split r after removing source and target main effects
   (double-centered off-diagonal matrix) >= 0.5 median. Needed because a matrix of mean |current| can
   correlate across fits simply because some regions are louder sources / targets; a sensory -> motor
   PAIR comparison needs the pair-specific part to be stable.

Only if 1-5 all pass: fit PS94_0910 (chronic) and report pre vs chronic sensory -> motor currents
(SSp-n / SSp-m / SSs -> tjM1 / ALM / MOp_rest, within and across hemispheres). If any fails: no
chronic interpretation; report the numbers and what to change.

Design choices fixed before looking at stability (may be tuned on FIT QUALITY only): reset interval
2 s, 20 training passes, tau 0.2 s, g 1.5, dt_factor 4, data 3-frame means (~10.4 Hz).

## Outcome (appended after the fits, 2026-10-30)

GATE FAILED (criteria 2-5; criterion 1 passed). Multi-unit, median: seeds r 0.51, halves 0.38 (min 0.15),
sessions 0.29, pair-specific halves 0.06. One-unit: 0.51 / 0.22 / 0.40 / 0.09. Fit quality: pVar 0.27 train,
0.21-0.25 held-out half; hold-at-reset < 0; J = 0 decay 0.15. Chronic PS94_0910 prepped, NOT fitted.
Exploratory, after the gate: g in {0, 0.5, 1.0} -> seeds 0.53-0.66, halves 0.18-0.24 (no fix); heavy-ridge VAR(1)
region matrices r 0.84-0.97 across halves / sessions, but ridge-dependent. Tables: `<out>/tables/`.

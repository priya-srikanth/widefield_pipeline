"""POWER of the repertoire (slide vs search-then-select) test (handoff open item 4; Priya 11-08 "proceed through the rest").

The repertoire test (`state_space_path_tests.repertoire`) asks whether more acute / subacute trials sit nearer the chronic
template than a cloud SLIDING to the session's own position would put there ("excess" > 0 = a mixture of pre-like and
chronic-like trials, i.e. search then select). Observed excess was ~0 or negative. Could the test see a mixture at all?

Planted alternative, per acute / subacute session and spout: every trial is placed at EITHER the pre template OR the
chronic template (area space), chosen at random with P(chronic) = the session's own position on the pre -> chronic
line at that spout (alpha, clipped to [0, 1]), plus that trial's OWN deviation from its session spout mean (real
trial-to-trial variability, real trial counts). The session mean therefore sits at about the same alpha, so the planted
session differs from a slid cloud only in being a two-state mixture. The planted session is scored by the unchanged
`repertoire` (same pre / chronic templates and space); N_DRAW draws per session.
Output: <state_space_recovery>/path_tests{tag}/repertoire_power.csv (per session: observed excess, planted mean excess,
fraction of planted draws above the observed). EXPLORATORY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_path_tests as PT  # noqa: E402
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

N_DRAW = 20


def planted(Y, pos, Tp, Tc, P, mu, rng):
    """Two-state version of one session (see docstring); spouts without both templates / enough trials kept as they are."""
    Z = Y.copy()
    zp, zc = (Tp - mu) @ P, (Tc - mu) @ P
    for si, p in enumerate(S.POS):
        m = np.flatnonzero(pos == p)
        if len(m) < R.MIN_TRIALS or not (np.isfinite(Tp[si]).all() and np.isfinite(Tc[si]).all()):
            continue
        dlt = (zc[si] - zp[si]).ravel()
        zbar = ((Y[m].mean(0) - mu) @ P).ravel()
        alpha = float(np.clip((zbar - zp[si].ravel()) @ dlt / (dlt @ dlt), 0, 1))
        dev = Y[m] - Y[m].mean(0)
        pick = rng.random(len(m)) < alpha
        Z[m] = np.where(pick[:, None, None], Tc[si], Tp[si]) + dev
    return Z


def main():
    rng = np.random.default_rng(0)
    data = S.load_sessions()[0]
    rows = []
    for an, sess in sorted(data.items()):
        pre = [s for s in sess if s[1] == "pre"]
        chr_ = [s for s in sess if s[1] == "chronic"]
        Tp, Tc = R.template(pre), R.template(chr_)
        P, mu = R.fixed_space(Tp)
        obs = {r["label"]: r for r in PT.repertoire(sess, P, mu, np.random.default_rng(0))}
        for lab, ep, day, Y, pos in sess:
            if ep not in ("acute", "subacute") or lab not in obs:
                continue
            ex = []
            for _ in range(N_DRAW):
                syn = [s for s in sess if s[1] in ("pre", "chronic")] + [(lab, ep, day, planted(Y, pos, Tp, Tc, P, mu, rng), pos)]
                r = [q for q in PT.repertoire(syn, P, mu, rng) if q["label"] == lab]
                if r:
                    ex.append(r[0]["excess"])
            rows.append({"animal": an, "label": lab, "epoch": ep, "day": day, "observed_excess": obs[lab]["excess"],
                         "planted_excess_mean": float(np.mean(ex)), "planted_excess_sd": float(np.std(ex)),
                         "frac_planted_above_observed": float(np.mean(np.asarray(ex) > obs[lab]["excess"]))})
    D = pd.DataFrame(rows)
    out = PT.OUT / "repertoire_power.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(D.groupby(["animal", "epoch"])[["observed_excess", "planted_excess_mean", "frac_planted_above_observed"]]
              .mean().reindex(["acute", "subacute"], level="epoch"))
        print("sessions where planted > observed in >= 95 % of draws:",
              int((D.frac_planted_above_observed >= 0.95).sum()), "/", len(D))


if __name__ == "__main__":
    main()

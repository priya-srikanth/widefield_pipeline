"""Hemispheric balance at matched executed landing, using EVERY first lick (Priya 11-07: "we should be able to use the
angle / lateral offset to use all spouts, even the off-target far contra licks").

Bins (state_space_balance_cells.py) lose acute far-contra entirely (0-3 % land on target). Here the landing is the
continuous 3-D lateral coordinate `side_ml` of the first lick (licks.csv.gz via state_space_landing.py; mm, + toward the
contralesional side as written there). Per animal and RING (close / far), on PRE trials only:

    size_h(trial) = a_h[target] + b_h * side_ml        h = contralesional / ipsilesional orofacial response size
                                                        (SSp-m, SSp-n, SSs; 0-1 s after the cue minus -0.5-0 s)

The slope is shared across the ring's three targets, so a far-contra-aimed lick that lands far-ipsi is predicted with
the far-contra intercept and a slope learned over the whole far-ipsi..far-contra range (interpolation, not extrapolation
past one spout's own range). Post trials: balance at matched target AND landing =
log2(mean observed contra / mean observed ipsi) - log2(mean predicted contra / mean predicted ipsi), per session x target.
Also refits the same model per post epoch to compare the landing SLOPE (does the balance still track the landing?).

Step 1 (`--collect`, ~9 min of share I/O): per-trial table -> landing/trial_sizes.csv. Step 2 (default): the analysis
-> landing/balance_angle_{sessions,coefs}.csv. Exploratory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

OUT = S.ROOT / "landing"
RINGS = {"close": ["close_ipsi", "close_mid", "close_contra"], "far": ["far_ipsi", "far_mid", "far_contra"]}


def collect():
    L = pd.read_csv(R.LANDING)
    L = {lab: g for lab, g in L[L.side_ml.notna()].groupby("label")}
    rows = []
    data, cols = S.load_sessions()
    for an, sess in sorted(data.items()):
        for lab, ep, _day, Y, pos in sess:
            g = L.get(lab)
            if g is None:
                continue
            hs = S.hemi(S.evoked(Y), S.cols_of(cols, an), R.OROFACIAL)                                # per trial
            for r, sm, land in zip(g.row.to_numpy(), g.side_ml.to_numpy(), g.land.to_numpy()):
                rows.append({"animal": an, "epoch": ep, "label": lab, "target": pos[r], "land": land, "side_ml": sm,
                             "contra": float(hs["contra"][r]), "ipsi": float(hs["ipsi"][r])})
    pd.DataFrame(rows).to_csv(OUT / "trial_sizes.csv", index=False)
    print("wrote trial_sizes.csv", len(rows))


def design(d, targets):
    X = np.column_stack([(d.target == t).to_numpy(float) for t in targets] + [d.side_ml.to_numpy()])
    return X


def fit(d, targets):
    X = design(d, targets)
    return {h: np.linalg.lstsq(X, d[h].to_numpy(), rcond=None)[0] for h in ("contra", "ipsi")}


def analyze():
    T = pd.read_csv(OUT / "trial_sizes.csv")
    srows, crows = [], []
    for (an, ring), g in T.assign(ring=T.target.str.split("_").str[0]).groupby(["animal", "ring"]):
        targets = RINGS[ring]
        pre = g[g.epoch == "pre"]
        B = fit(pre, targets)
        lo, hi = pre.side_ml.quantile([0.01, 0.99])
        for ep, e in g.groupby("epoch"):
            if len(e) > 30 and e.target.nunique() == 3:
                Be = fit(e, targets)
                crows.append({"animal": an, "ring": ring, "epoch": ep, "n": len(e),
                              **{f"slope_{h}": Be[h][-1] for h in Be},
                              **{f"a_{h}_{t}": Be[h][i] for h in Be for i, t in enumerate(targets)}})
        g = g.assign(pc=design(g, targets) @ B["contra"], pi=design(g, targets) @ B["ipsi"],
                     inrange=g.side_ml.between(lo, hi))
        for (lab, ep, t), s in g.groupby(["label", "epoch", "target"]):
            if len(s) < R.MIN_TRIALS:
                continue
            row = {"animal": an, "ring": ring, "label": lab, "epoch": ep, "target": t, "n": len(s),
                   "share_inrange": float(s.inrange.mean()), "side_ml": float(s.side_ml.mean())}
            for nm, ss in (("", s), ("_inrange", s[s.inrange])):
                if len(ss) >= R.MIN_TRIALS and ss.contra.mean() > 0 and ss.ipsi.mean() > 0 and ss.pc.mean() > 0 \
                        and ss.pi.mean() > 0:
                    obs = np.log2(ss.contra.mean() / ss.ipsi.mean())
                    row[f"bal_obs{nm}"] = obs
                    row[f"bal_matched{nm}"] = obs - np.log2(ss.pc.mean() / ss.pi.mean())
            srows.append(row)
    S_, C = pd.DataFrame(srows), pd.DataFrame(crows)
    S_.to_csv(OUT / "balance_angle_sessions.csv", index=False)
    C.to_csv(OUT / "balance_angle_coefs.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 2, "display.max_rows", 200):
        m = S_.groupby(["animal", "target", "epoch"]).bal_matched.mean().unstack()
        print("balance at matched target + landing, minus pre-fit prediction (per animal, session mean):")
        print(m.reindex(columns=R.EPOCHS).unstack(0).reindex(S.POS))
        print("\nlanding slopes (size per mm), refit per epoch:")
        print(C.pivot_table(index=["animal", "ring"], columns="epoch", values=["slope_contra", "slope_ipsi"])
              .reindex(columns=R.EPOCHS, level=1))
        print("\nshare of trials inside the pre side_ml range, far contra:")
        print(S_[S_.target == "far_contra"].groupby(["animal", "epoch"]).share_inrange.mean().unstack())


if __name__ == "__main__":
    collect() if "--collect" in sys.argv else analyze()

"""DIRECTION vs EXTENT: the frozen where-decoder on lick ANGLE and 3-D REACH (Priya 10-27: "redo our lateral vs
protrusion analysis with angle vs 3d protrusion"; angle = primary coordinate, with x / y / z alongside).

Same decoder as the KEY row (3.15 / 3.21): per animal x lick set, ridge on Allen areas (lags 0..0.2 s), features z-scored
within session (`zscore_check/session_sd.csv`), trained on August pre licks, pre sessions leave-one-session-out. Targets:
  angle   atan2(lateral, sagittal) in degrees (the encoder's exec_angle), decoded DIRECTLY (one ridge)
  reach   3-D distance of the tongue tip from the mouth at the extension peak (`reach_mm`, the lick table)
  elev    ELEVATION below the axial (horizontal) plane, atan2(down, sqrt(lateral^2 + forward^2)), degrees (Priya 10-27:
          "should we add angle from axial plane"): with angle and reach a full 3-D polar description of each lick
Per session: r (decoded vs actual) / the animal's pre mean; for angle also on the LONGER licks only (sagittal extension
above the animal's pre median -- angle is unstable on short licks, where a small lateral error swings atan2); offset
beyond the pre pull (decoded - (a + b x actual), a, b fit on the animal's pre licks) minus pre. Change vs pre: paired
nested bootstrap; per-animal tables first (rule 8).
Outputs <E>/iti_licks/where_angle_reach_{preds.csv.gz,sessions.csv,tests.csv}, figures/iti_where_angle_reach.png.
Usage: python scripts/iti_where_angle_reach.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402

from wfield_local import analysis_kit as ak  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
EPS = ["pre", *EP]


def predictions() -> pd.DataFrame:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from iti_decoder_contrib import feature_names
    SD = pd.read_csv(il.OUT / "zscore_check" / "session_sd.csv", index_col=0)
    names = feature_names()
    D = wd.load("areas")
    rows = []
    for st in ("iti", "trial"):
        for a in AN:
            S = {}
            for x, d in sorted(D.items()):
                if d["animal"] != a:
                    continue
                X, C = d[st]
                elev = np.degrees(np.arctan2(C[:, 4], np.hypot(C[:, 0], C[:, 1])))   # below the axial plane
                Y = np.column_stack([np.degrees(np.arctan2(C[:, 0], C[:, 3])), C[:, 2], elev, C[:, 3]])  # +sag
                ok = np.isfinite(Y).all(1)
                if ok.sum() >= 20:
                    S[x] = (d["epoch"], X[ok] / SD.loc[x, names].to_numpy(float)[None, :], Y[ok],
                            np.asarray(d[f"{st}_pos"])[ok])
            pre = [x for x, v in S.items() if v[0] == "pre"]
            if len(pre) < 3:
                continue
            fits = {}
            for held in [*pre, None]:
                trn = [q for q in pre if q != held]
                Xt, Yt = np.vstack([S[q][1] for q in trn]), np.vstack([S[q][2] for q in trn])
                sc = StandardScaler().fit(Xt)
                fits[held] = (sc, [RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(Xt), Yt[:, k])
                                   for k in range(3)])
            for x, (ep, X, Y, P) in S.items():
                sc, ms = fits[x if x in pre else None]
                Pr = np.column_stack([m.predict(sc.transform(X)) for m in ms])
                rows.append(pd.DataFrame({"set": st, "animal": a, "label": x, "epoch": ep, "pos": P,
                                          "act_ang": Y[:, 0], "act_reach": Y[:, 1], "act_elev": Y[:, 2],
                                          "act_sag": Y[:, 3], "dec_ang": Pr[:, 0], "dec_reach": Pr[:, 1],
                                          "dec_elev": Pr[:, 2]}))
    return pd.concat(rows, ignore_index=True)


def sessions(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (st, a), g in P.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"]
        thr = pre.act_sag.median()
        pull = {t: np.polyfit(pre[f"act_{t}"], pre[f"dec_{t}"], 1) for t in ("ang", "reach", "elev")}
        for (lab, ep), h in g.groupby(["label", "epoch"]):
            r = dict(set=st, animal=a, label=lab, epoch=ep, n=len(h))
            for t in ("ang", "reach", "elev"):
                b, c = pull[t]
                r[f"r_{t}"] = float(np.corrcoef(h[f"dec_{t}"], h[f"act_{t}"])[0, 1])
                r[f"bey_{t}"] = float(np.mean(h[f"dec_{t}"] - (c + b * h[f"act_{t}"])))
            lg = h[h.act_sag > thr]
            r["r_ang_long"] = float(np.corrcoef(lg.dec_ang, lg.act_ang)[0, 1]) if len(lg) >= 20 else np.nan
            rows.append(r)
    S = pd.DataFrame(rows)
    for c in ("r_ang", "r_reach", "r_elev", "r_ang_long", "bey_ang", "bey_reach", "bey_elev"):
        pre = S[S.epoch == "pre"].groupby(["set", "animal"])[c].mean()
        ref = np.array([pre.get((s, a), np.nan) for s, a in zip(S.set, S.animal)])
        S[c + "_d"] = S[c] / ref if c.startswith("r_") else S[c] - ref
    return S


def tests(S: pd.DataFrame) -> pd.DataFrame:
    out = []
    for st, g in S.groupby("set"):
        for c in ("r_ang_d", "r_ang_long_d", "r_reach_d", "r_elev_d", "bey_ang_d", "bey_reach_d", "bey_elev_d"):
            prd = {a: h[c].dropna().to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            for ep in EP:
                post = {a: h[c].dropna().to_numpy() for a, h in g[g.epoch == ep].groupby("animal") if a in prd}
                post = {a: v for a, v in post.items() if len(v)}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                am = (g[g.epoch == ep].groupby("animal")[c].mean() - g[g.epoch == "pre"].groupby("animal")[c].mean()).dropna()
                out.append(dict(set=st, measure=c, epoch=ep, point=ci.point, lo=ci.lo, hi=ci.hi,
                                n_dir=int((am * np.sign(ci.point) > 0).sum()), n=len(am)))
    return pd.DataFrame(out)


def plot(S: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iti_check_figures import epoch_panel
    panels = [("r_ang_long_d", "ANGLE r / pre (longer licks)", 1.0), ("r_ang_d", "ANGLE r / pre (all licks)", 1.0),
              ("r_elev_d", "ELEVATION r / pre", 1.0), ("r_reach_d", "3-D REACH r / pre", 1.0),
              ("bey_ang_d", "ANGLE offset beyond pre pull (deg)", 0.0),
              ("bey_elev_d", "ELEVATION offset beyond pull (deg)", 0.0),
              ("bey_reach_d", "REACH offset beyond pre pull (mm)", 0.0)]
    fig, axs = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 8.2), squeeze=False)
    for i, st in enumerate(("iti", "trial")):
        g = S[S.set == st]
        for j, (c, lab, ref) in enumerate(panels):
            epoch_panel(axs[i, j], g, c, ref=ref, title=f"{'ITI' if st == 'iti' else 'cued'}: {lab}",
                        ylab=lab if j in (0, 4) else "")
    axs[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("NEURAL: DIRECTION vs EXTENT -- the frozen where-decoder (z-scored Allen areas, lags 0..0.2 s, trained on "
                 "August pre licks) on the lick's 3-D POLAR coordinates: lateral ANGLE (atan2(lateral, sagittal)), ELEVATION below the axial plane (atan2(down, horizontal extent); + = steeper down) and 3-D REACH.\nr / pre: how well cortex "
                 "tracks each lick; offset beyond the pull: + = read more contralesional (angle) / further (reach) than "
                 "the lick. Angle r also on licks longer than the animal's pre median\n(angle is unstable on short licks). "
                 "Thin lines = animals, black = mean, grey dots = sessions, diamonds = change vs pre with 95% paired "
                 "nested bootstrap CI (* = excludes 0). PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_size_inches()[1]))
    fig.savefig(il.OUT / "figures" / "iti_where_angle_reach.png", dpi=100)
    plt.close(fig)


def main() -> int:
    f = il.OUT / "where_angle_reach_preds.csv.gz"
    P = pd.read_csv(f) if f.exists() else predictions()
    if not f.exists():
        P.to_csv(f, index=False)
    S = sessions(P)
    S.to_csv(il.OUT / "where_angle_reach_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    for c in ("r_ang_long_d", "r_ang_d", "r_elev_d", "r_reach_d", "bey_ang_d", "bey_elev_d", "bey_reach_d"):
        print(f"== per animal (rule 8): {c}")
        print(S.pivot_table(index=["set", "animal"], columns="epoch", values=c).reindex(columns=EPS).round(2).to_string())
    T = tests(S)
    T.to_csv(il.OUT / "where_angle_reach_tests.csv", index=False)
    print("\n== change vs pre (paired nested bootstrap)\n" + T.round(3).to_string(index=False))
    plot(S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

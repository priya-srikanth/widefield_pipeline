"""LIKE-FOR-LIKE read-out on ANGLE and 3-D REACH (and, in the table only, elevation and lateral / forward / down) with the
LOCAL-LINEAR reference (Priya 10-27 late: "for the angle decoder, should we use gaussian smoother like for like licks
also?" -> local-linear with the edge line carried out ("can we use the same linear strategy for the edges?", "why can't
we decode outside pre reach? isn't that what the local linear is for" -> "sounds good"); plot angle + reach, the rest in
the table ("sagittal, elevation, reach, xyz ... may be too much")). Replaces 'offset beyond the pre pull' (bey_*) of
iti_where_angle_reach.py, whose answer depended on the assumed pull slope.

Per lick: resid = decoded - reference(actual), reference = the animal's PRE licks of the same set (any spout) by a
Gaussian-weighted local line on the actual value (`likeforlike.py`; pre licks leave-one-pre-session-out; width by
held-out pre error). + = read more contralesional (angle) / longer (reach) than pre licks that went to the same place.
Session means of resid: ALL licks (primary) and licks INSIDE the pre range (sensitivity); change vs pre = paired nested
bootstrap (iti_check_figures.epoch_compute); per-animal values first (rule 8). Also: share of licks beyond the pre
range per epoch, and the epoch effect with the reference refit on resampled pre sessions.
Inputs <E>/iti_licks/where_angle_reach_preds.csv.gz (iti_where_angle_reach.py) and where_3axis_preds.csv.gz
(iti_where_3axis.py). Outputs <E>/iti_licks/matched_axes_{sigma_cv,sessions,tests,coverage,refboot}.csv,
figures/lick_matched_angle_reach.png; figdata/lick_matched_angle_reach.pkl (--plot-only redraws from it).
Usage: python scripts/lick_matched_angle_reach.py [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import likeforlike as lfl  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
SETS = {"iti": "ITI", "trial": "cued"}
STEM = "lick_matched_angle_reach"
MIN = 8
DEG = (2.0, 3.0, 5.0, 7.5, 10.0, 15.0)
MM = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
# axis: (preds file, actual, decoded, sigma grid, unit, plotted)
AXES = {"angle": ("where_angle_reach_preds.csv.gz", "act_ang", "dec_ang", DEG, "deg", True),
        "reach": ("where_angle_reach_preds.csv.gz", "act_reach", "dec_reach", MM, "mm", True),
        "elevation": ("where_angle_reach_preds.csv.gz", "act_elev", "dec_elev", DEG, "deg", False),
        "lateral_3axis": ("where_3axis_preds.csv.gz", "act_lat", "dec_lat", MM, "mm", False),
        "forward": ("where_3axis_preds.csv.gz", "act_fwd", "dec_fwd", MM, "mm", False),
        "down": ("where_3axis_preds.csv.gz", "act_down", "dec_down", MM, "mm", False)}
SCOPES = ("all", "inside pre range")


def compute() -> dict:
    from iti_check_figures import epoch_compute
    cache, cvs, sess, cov, rbs, tests, panels = {}, [], [], [], [], [], {}
    for ax, (fn, x, y, sigmas, unit, shown) in AXES.items():
        if fn not in cache:
            cache[fn] = pd.read_csv(il.pred_file(fn))           # basis-specific cache name (il.PRED_TAG)
        P = cache[fn].dropna(subset=[x, y]).reset_index(drop=True)
        sg, cv = lfl.choose(P, x, y, sigmas)
        cvs.append(cv.assign(axis=ax, chosen=sg))
        R = lfl.ref_table(P, x, y, sg)
        P = P.assign(resid=P[y] - R.ref, beyond=R.beyond)
        print(f"[{ax}] local-linear sigma = {sg} {unit}", flush=True)
        cov.append(lfl.coverage(P, P.beyond.to_numpy()).assign(axis=ax))
        rbs.append(lfl.boot_reference(P, x, y, sg).assign(axis=ax))
        for scope in SCOPES:
            Q = P if scope == "all" else P[(P.beyond == 0) | (P.epoch == "pre")]
            S = Q.groupby(["set", "animal", "label", "epoch"]).agg(resid=("resid", "mean"), n=("resid", "size"))
            S = S[S.n >= MIN].reset_index().assign(axis=ax, scope=scope, unit=unit)
            sess.append(S)
            for st in SETS:
                F = epoch_compute(S[S.set == st], "resid")
                if shown:
                    panels[(st, ax, scope)] = F
                am = S[S.set == st].groupby(["animal", "epoch"]).resid.mean().unstack("epoch")
                for (xx, pt, lo, hi, _txt) in F["post"]:
                    ep = EP[xx - 1]
                    d = (am[ep] - am["pre"]).dropna() if ep in am else pd.Series(dtype=float)
                    tests.append(dict(axis=ax, unit=unit, scope=scope, set=st, epoch=ep, point=pt, lo=lo, hi=hi,
                                      n_dir=int((d * np.sign(pt) > 0).sum()), n=len(d),
                                      **{f"animal_{a}": d.get(a, np.nan) for a in AN}))
    out = {"sigma_cv": pd.concat(cvs), "sessions": pd.concat(sess), "coverage": pd.concat(cov),
           "refboot": pd.concat(rbs), "tests": pd.DataFrame(tests)}
    for k, v in out.items():
        v.to_csv(il.OUT / f"matched_axes_{k}.csv", index=False)
    pd.set_option("display.width", 250)
    T = out["tests"]
    print("== per animal (rule 8) + nested bootstrap")
    print(T.round(2).to_string(index=False))
    print("== share beyond the pre range")
    print(out["coverage"].round(3).to_string(index=False))
    print("== epoch effect, reference refit on resampled pre sessions")
    print(out["refboot"].round(3).to_string(index=False))
    sig = {a: float(out["sigma_cv"][out["sigma_cv"].axis == a].chosen.iloc[0]) for a in AXES}
    C = out["coverage"]
    cov_txt = {(r.axis, r.set, r.epoch): r.share_beyond for r in C.itertuples()}
    return {"panels": panels, "sigma": sig, "coverage": cov_txt}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iti_check_figures import epoch_draw
    cols = [(ax, sc) for ax in ("angle", "reach") for sc in SCOPES]
    fig, axs = plt.subplots(2, len(cols), figsize=(3.6 * len(cols), 8.4), squeeze=False)
    for i, st in enumerate(SETS):
        for j, (ax_, sc) in enumerate(cols):
            P = F["panels"].get((st, ax_, sc))
            a = axs[i, j]
            if P is None:
                a.set_axis_off()
                continue
            unit = AXES[ax_][4]
            cv = ", ".join(f"{e[:3]} {F['coverage'].get((ax_, st, e), np.nan):.0%}" for e in EP)
            epoch_draw(a, P, ref=0.0, title=f"{SETS[st]}: {ax_.upper()}, {'ALL licks' if sc == 'all' else 'inside pre range'}"
                       + (f"\nbeyond pre range: {cv}" if sc == "all" else ""),
                       ylab=f"read-out vs pre licks at the same {ax_} ({unit})" if j % 2 == 0 else "")
    fig.suptitle("LIKE-FOR-LIKE read-out on ANGLE and 3-D REACH (local-linear reference; no pull slope): each lick's decoded "
                 "value minus what the animal's PRE licks that went to the same place were read as.\n+ = read more "
                 "contralesional (angle) / longer (reach). Beyond the pre range the edge line is carried out; right panel "
                 "of each pair = licks inside the pre range only.\nwidth: " +
                 ", ".join(f"{a} {F['sigma'][a]:g} {AXES[a][4]}" for a in ("angle", "reach")) +
                 " (held-out pre error). Dots = sessions, lines = animals, change vs pre = 95 % paired nested bootstrap, "
                 "k/n animals. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(il.OUT / "figures" / f"{STEM}.png", dpi=100)
    plt.close(fig)


def main() -> int:
    il.split_only("lick_matched_angle_reach")   # outputs not basis-tagged: split basis only (10-27 late)
    if plot_only():
        plot(load_fd(STEM))
        return 0
    F = compute()
    save_fd(STEM, F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

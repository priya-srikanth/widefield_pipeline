"""TRIAL-BY-TRIAL test of the balance (10-28, test 1 of the impairment hypothesis): within a session, do trials whose
first post-cue lick has a more negative balance aim worse? And does the balance BEFORE the lick (-0.5..0 s) predict the
upcoming lick's aim (cause rather than consequence)?

Per first post-cue lick (cued, order 0), pooled over the primary areas (ALM, tjM1, SSp-m, SSp-n, SSs), per hemisphere:
  post  mean dF/F 0.1..0.5 s after the lick minus -1.0..-0.5 s before it (the primary window)
  pre   mean dF/F -0.5..0 s before the lick minus -1.0..-0.5 s (the build-up into the lick)
each referenced to the animal's PRE straight line on lick angle (pre sessions leave-one-session-out) and divided by the
pre SD of that residual; trial balance = z(contra residual) - z(ipsi residual).
Trial aim = the lick's angle minus the animal's pre mean angle at that target spout, signed so - = ipsilesional (as the
far-contra aim of key figure 1A, here per trial and every spout).
Both are centered within session x spout (so neither the session's recovery state nor the target can carry the link);
per animal x epoch: r of the centered values; far-contra trials alone too. PRE sessions are the control: the balance is
angle-referenced, so a mechanical coupling to the lick's own angle would show there as well.
Outputs <OUT_W>/baltrial_licks.csv.gz, baltrial_r.csv; figure lick_balance_trial.png.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
import lick_recovery_primary as rp  # noqa: E402

WINS = {"post": (0.1, 0.5), "pre": (-0.5, 0.0)}
EPS = ["pre", "acute", "subacute", "chronic"]


def _wm(lo_, hi_):
    L = il.LAGS_D
    return (L >= lo_ - 1e-9) & (L <= hi_ + 1e-9) if hi_ <= 0 else (L >= lo_ - 1e-9) & (L < hi_ - 1e-9)


def _session(label: str) -> pd.DataFrame:
    acts = lbr.cv._acts(label)
    d = il._load(label)
    cols = list(d["cols"])
    act = np.asarray(acts.get("trial", []), float)
    if not len(act):
        return pd.DataFrame()
    m = il.subseq_match(d["tr_side"].astype(float), act)
    ok, mm = m >= 0, np.maximum(m, 0)
    X = d["tr_X"].astype(float)
    base_m = (il.LAGS_D >= il.BASE_RESP[0] - 1e-9) & (il.LAGS_D <= il.BASE_RESP[1] + 1e-9)
    R = {"set": "trial", "label": label, "k": np.arange(len(act))}
    for side, h in (("c", "right"), ("i", "left")):
        tr = np.sum([X[:, :, cols.index(f"{a}_{h}")] for a in rp.POOL], 0)
        base = tr[:, base_m].mean(1)
        for w, (a_, b_) in WINS.items():
            v = tr[:, _wm(a_, b_)].mean(1) - base
            R[f"{w}_{side}"] = np.where(ok, v[mm], np.nan)
    return pd.DataFrame(R)


def compute() -> dict:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path())
    P = P[P.set == "trial"].reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), _session)
    if fail:
        raise RuntimeError(f"failed: {fail[:5]}")
    P = P.merge(pd.concat([v for _i, v in res], ignore_index=True), on=["set", "label", "k"], how="left",
                validate="one_to_one")
    P = P[P.order == 0].copy()
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    sgn = float(np.sign(P[(P.epoch == "pre") & (P.pos == "far_contra")].angle.mean()))
    out = []
    for _an, g in P.groupby("animal"):
        g = g.dropna(subset=["angle"]).copy()
        pre = g[g.epoch == "pre"]
        g["aim"] = sgn * (g.angle - g.pos.map(pre.groupby("pos").angle.mean()))
        for w in WINS:
            for s in ("c", "i"):
                v = f"{w}_{s}"
                g[f"r_{v}"] = np.nan
                pr = pre.dropna(subset=[v])
                for lab in [None, *pr.label.unique()]:
                    tr = pr[pr.label != lab] if lab else pr
                    if len(tr) < 30:
                        continue
                    b1, b0 = np.polyfit(tr.angle, tr[v], 1)
                    sel = (g.label == lab) if lab else (g.epoch != "pre")
                    g.loc[sel, f"r_{v}"] = g.loc[sel, v] - (b0 + b1 * g.loc[sel, "angle"])
                sd = g.loc[g.epoch == "pre", f"r_{v}"].std()
                g[f"z_{v}"] = g[f"r_{v}"] / sd
            g[f"bal_{w}"] = g[f"z_{w}_c"] - g[f"z_{w}_i"]
        out.append(g)
    P = pd.concat(out, ignore_index=True)
    for c in ("aim", "bal_post", "bal_pre"):
        P[f"{c}_cen"] = P[c] - P.groupby(["label", "pos"])[c].transform("mean")
    keep = ["animal", "label", "epoch", "pos", "angle", "aim", "bal_post", "bal_pre", "aim_cen", "bal_post_cen", "bal_pre_cen"]
    P[keep].to_csv(il.OUT_W / "baltrial_licks.csv.gz", index=False)
    rows = []
    for scope, Q in (("all spouts", P), ("far contra", P[P.pos == "far_contra"])):
        for (an, ep), g in Q.groupby(["animal", "epoch"]):
            for w in WINS:
                h = g.dropna(subset=[f"bal_{w}_cen", "aim_cen"])
                if len(h) > 20:
                    rows.append({"scope": scope, "animal": an, "epoch": ep, "window": w, "n": len(h),
                                 "r": np.corrcoef(h[f"bal_{w}_cen"], h.aim_cen)[0, 1]})
    R = pd.DataFrame(rows)
    R.to_csv(il.OUT_W / "baltrial_r.csv", index=False)
    pd.set_option("display.width", 200)
    print("== within-session r(trial balance, trial aim), per animal (rule 8)\n" + R.pivot_table(
        index=["scope", "window", "epoch"], columns="animal", values="r").reindex(EPS, level=2).round(3).to_string())
    return {"R": R}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    R = F["R"]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for j, (scope, w) in enumerate([(s, w) for s in ("all spouts", "far contra") for w in WINS]):
        ax = axs[j]
        t = R[(R.scope == scope) & (R.window == w)]
        for xi, ep in enumerate(EPS):
            u = t[t.epoch == ep]
            ax.bar(xi, u.r.mean(), color="0.75" if ep == "pre" else "#C44E52", width=0.6)
            for _, r in u.iterrows():
                ax.plot(xi + (["PS92", "PS93", "PS94", "PS95"].index(r.animal) - 1.5) * 0.08, r.r,
                        ANIMAL_MARKER.get(r.animal, "o"), color="k", ms=5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(EPS)), EPS)
        ax.set_title(f"{scope}, balance {'AFTER the lick (0.1-0.5 s)' if w == 'post' else 'BEFORE the lick (-0.5..0 s)'}",
                     fontsize=9.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].set_ylabel("within-session r (trial balance, trial aim)\ncentred within session x spout", fontsize=9)
    fig.suptitle("TRIAL BY TRIAL: within a session, do first post-cue licks with a more negative balance (contralesional "
                 "relatively weaker) aim more ipsilesionally? Bars = mean of animals, symbols = animals; pre (gray) = the "
                 "control for any mechanical coupling to the lick's own angle. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_balance_trial.png", dpi=95)
    plt.close(fig)


def trial_set_sentence() -> str:
    """The lick set of lick_balance_trial (Priya 11-09): the FIRST post-cue lick of each trial in the KEY set."""
    import lick_balance_subsets as lbs
    return lbs.set_sentence(("trial_1",), full=False, inside=False, subset=False, extra=["lick angle finite"])


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    F = compute()
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

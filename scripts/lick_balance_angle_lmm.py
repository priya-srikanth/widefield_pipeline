"""Hemispheric BALANCE vs LICK ANGLE, pre vs post -- mixed models per session x spout (Priya 10-28: "can we do LMM on
per-session balance (including multiple regions) and landing (for multiple spouts)?", "or maybe just balance and lick
angle", "pre vs post").

Unit = one session x spout (the spout = the target for cued licks, the PREVIOUS trial's spout for ITI licks), so each
session contributes one point per spout and the lick angle varies across spouts and sessions.
  LICK ANGLE  lateral angle of the 3-D tongue vector at the lick peak from the midline (sagittal) plane, degrees,
              asin(side_ml / reach) from the pose3d table (12.16's per-lick covariates file); + = contralesional.
  BALANCE     per pair (ALM, tjM1, SSp-m, SSp-n), two measures:
              dec  = X_right - X_left in the decoder's own window (mean 0..0.2 s from the lick peak; the decoder's inputs,
                     x1e3 dF/F, no per-lick baseline) -- the measure of 12.17
              resp = log2(contra / ipsi response AUC 0..1.0 s, baseline -1.0..-0.5 s) -- the measure of 14.9
              (cued: first cued licks; ITI: bout starts)
MODELS (statsmodels MixedLM, random intercept per animal, ML; centred angle):
  y ~ angle * post          post = any post-stroke epoch: (1) slope of balance on angle pre-stroke; (2) angle x post =
                            does the slope change; (3) post = the shift at the SAME angle (the landing-matched balance
                            shift, the question 12.17 vs 14.9 left open)
  y ~ angle * C(epoch)      the same per epoch
Sessions x spouts are the observations (anti-conservative at 4 animals); the per-animal slopes and shifts are printed
first (rule 8; OLS per animal x period) and drawn.
Outputs <OUT_W>/balance_angle_points.csv, balance_angle_lmm.csv, balance_angle_animals.csv; figure
lick_balance_angle.png (figdata + --plot-only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

PAIRS = lbr.PAIRS
SETMAP = {"trial": "cued", "iti": "iti"}
MIN_LICKS = 5
EP = ["acute", "subacute", "chronic"]


def points() -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lo.preds_path()).reset_index(drop=True)
    P["k"] = P.groupby(["set", "label"]).cumcount()
    C = pd.read_csv(il.OUT_W / f"covref_covariates{lo.SRC_TAG}.csv.gz", usecols=["set", "label", "k", "reach_mm", "side_k"])
    P = P.merge(C, on=["set", "label", "k"], how="left", validate="one_to_one")
    res, fail = ak.fan_sessions(sorted(P.label.unique()), lbr._session_bal)
    if fail:
        raise RuntimeError(f"balance failed for {len(fail)} sessions: {fail[:5]}")
    B = pd.concat([v for _it, v in res], ignore_index=True)
    P = P.merge(B, on=["set", "label", "k"], how="left", validate="one_to_one")
    P["angle"] = np.degrees(np.arcsin(np.clip(P.side_k / P.reach_mm, -1, 1)))
    P = P[P.order == 0]                    # first cued lick / ITI bout start (the licks of 14.9)
    print("lick angle finite on", f"{P.angle.notna().mean():.3f}", "of first / bout-start licks")
    g = P.groupby(["set", "label", "animal", "epoch", "pos"])
    S = g[["angle", *lbr.BAL]].mean().join(g.size().rename("n")).reset_index()
    S = S[S.n >= MIN_LICKS]
    S["set"] = S.set.map(SETMAP)
    R = pd.read_csv(il.OUT_W / "lateral_ratio_sessions.csv")
    R = R[R.spout.isin(lo.POS)].pivot_table(index=["set", "label", "spout"], columns="pair", values="logr_0-1")
    R.columns = [f"resp_{c}" for c in R.columns]
    S = S.merge(R.reset_index().rename(columns={"spout": "pos"}), on=["set", "label", "pos"], how="left")
    S = S.rename(columns={f"bal_{a}": f"dec_{a}" for a in PAIRS})
    S["post"] = (S.epoch != "pre").astype(int)
    return S


def _fit(df: pd.DataFrame, formula: str):
    import statsmodels.formula.api as smf
    return smf.mixedlm(formula, df, groups=df["animal"]).fit(reml=False)


def compute() -> dict:
    S = points()
    S.to_csv(il.OUT_W / "balance_angle_points.csv", index=False)
    rows, an = [], []
    for st, G0 in S.groupby("set"):
        mu = G0[G0.epoch == "pre"].angle.mean()
        for meas in ("dec", "resp"):
            for a in PAIRS:
                y = f"{meas}_{a}"
                G = G0.dropna(subset=["angle", y]).rename(columns={y: "y"})
                G["ac"] = G.angle - mu
                for (animal, post), h in G.groupby(["animal", "post"]):     # rule 8: per animal x period OLS
                    if len(h) >= 6 and h.ac.std() > 0:
                        b = np.polyfit(h.ac, h.y, 1)
                        an.append({"set": st, "measure": meas, "pair": a, "animal": animal, "post": post,
                                   "slope": b[0], "at_mean_angle": b[1], "n": len(h)})
                try:
                    f = _fit(G, "y ~ ac * post")
                    fe, p_ = f.fe_params, f.pvalues
                    rows.append({"set": st, "measure": meas, "pair": a, "model": "post", "n": len(G),
                                 "slope_pre": fe["ac"], "p_slope_pre": p_["ac"],
                                 "slope_change_post": fe["ac:post"], "p_slope_change": p_["ac:post"],
                                 "shift_post_same_angle": fe["post"], "p_shift": p_["post"], "angle_centre": mu})
                    f = _fit(G, "y ~ ac * C(epoch, Treatment('pre'))")
                    fe, p_ = f.fe_params, f.pvalues
                    for ep in EP:
                        t = f"C(epoch, Treatment('pre'))[T.{ep}]"
                        if t in fe:
                            rows.append({"set": st, "measure": meas, "pair": a, "model": ep, "n": len(G),
                                         "slope_pre": fe["ac"], "p_slope_pre": p_["ac"],
                                         "slope_change_post": fe.get(f"ac:{t}", np.nan),
                                         "p_slope_change": p_.get(f"ac:{t}", np.nan),
                                         "shift_post_same_angle": fe[t], "p_shift": p_[t], "angle_centre": mu})
                except Exception as ex:                  # reported, not hidden
                    rows.append({"set": st, "measure": meas, "pair": a, "model": f"FAILED {ex}"})
    M, A = pd.DataFrame(rows), pd.DataFrame(an)
    M.to_csv(il.OUT_W / "balance_angle_lmm.csv", index=False)
    A.to_csv(il.OUT_W / "balance_angle_animals.csv", index=False)
    pd.set_option("display.width", 230)
    print("== per animal (rule 8): OLS slope (per deg) and value at the pre-mean angle, pre (0) vs post (1)")
    print(A.pivot_table(index=["set", "measure", "pair", "animal"], columns="post",
                        values=["slope", "at_mean_angle"]).round(4).to_string())
    print("== mixed models\n" + M.round(4).to_string(index=False))
    return {"S": S, "M": M, "A": A}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    S, M = F["S"], F["M"]
    for meas, lab in (("dec", "decoder-window balance X_right - X_left (x1e3 dF/F)"),
                      ("resp", "response ratio log2(contra / ipsi AUC 0..1.0 s)")):
        fig, axs = plt.subplots(2, len(PAIRS), figsize=(20, 9.5), squeeze=False)
        for i, st in enumerate(("cued", "iti")):
            G = S[S.set == st]
            mu = G[G.epoch == "pre"].angle.mean()
            for j, a in enumerate(PAIRS):
                ax = axs[i, j]
                y = f"{meas}_{a}"
                for ep in il.EP4:
                    h = G[G.epoch == ep]
                    ax.plot(h.angle, h[y], "o", color=col[ep], ms=2.5, alpha=0.45, mec="none")
                m = M[(M.set == st) & (M.measure == meas) & (M.pair == a) & (M.model == "post")]
                if len(m) and np.isfinite(m.slope_pre.iloc[0]):
                    r = m.iloc[0]
                    x = np.linspace(np.nanpercentile(G.angle, 2), np.nanpercentile(G.angle, 98), 50)
                    # intercepts from the fixed effects are not stored; draw the slopes through the group means
                    pre = G[G.post == 0]
                    b0 = np.nanmean(pre[y]) - r.slope_pre * (np.nanmean(pre.angle) - mu)
                    ax.plot(x, b0 + r.slope_pre * (x - mu), color="k", lw=2)
                    ax.plot(x, b0 + r.shift_post_same_angle + (r.slope_pre + r.slope_change_post) * (x - mu),
                            color="#d95f02", lw=2, ls="--")
                    ax.set_title(f"{a}, {st}: pre slope {r.slope_pre:+.3f}/deg (p {r.p_slope_pre:.1g})\n"
                                 f"post: slope change {r.slope_change_post:+.3f} (p {r.p_slope_change:.1g}), shift at "
                                 f"same angle {r.shift_post_same_angle:+.2f} (p {r.p_shift:.1g})", fontsize=8)
                ax.axhline(0, color="0.8", lw=0.6)
                ax.axvline(0, color="0.8", lw=0.6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if i == 1:
                    ax.set_xlabel("lick angle from midline (deg, + = contralesional)", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(f"{'CUED first lick' if st == 'cued' else 'ITI bout start'}\n{lab}", fontsize=8)
        fig.legend(handles=[Line2D([], [], color=col[e], marker="o", lw=0, label=e) for e in il.EP4]
                   + [Line2D([], [], color="k", lw=2, label="pre fit"),
                      Line2D([], [], color="#d95f02", lw=2, ls="--", label="post fit")],
                   loc="lower center", ncol=6, fontsize=9, frameon=False)
        fig.suptitle(f"Hemispheric balance vs lick angle, pre vs post ({meas}): one point per session x spout (cued: target; "
                     "ITI: previous trial's spout); mixed model y ~ angle * post + (1 | animal), sessions x spouts as "
                     "observations\n(anti-conservative at 4 animals; per-animal slopes in balance_angle_animals.csv). "
                     "'shift at same angle' = post minus pre at the pre-mean angle. PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.92))
        fig.savefig(il.OUT_W / "figures" / f"lick_balance_angle_{meas}.png", dpi=100)
        plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_balance_angle"))
        return 0
    F = compute()
    save_fd("lick_balance_angle", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

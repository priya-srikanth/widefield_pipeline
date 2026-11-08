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
MODELS (statsmodels MixedLM, random intercept per animal, ML; centered angle):
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
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
import lick_balance_reference as lbr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

PAIRS = [*lbr.PAIRS, "orofacial"]       # + the four pooled (Priya 10-28)
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
    print("lick angle finite on", f"{P.angle.notna().mean():.3f}", "of licks")
    # first cued lick / ITI bout start (the licks of 14.9) AND the LATER licks of the same trials / bouts (order >= 1;
    # Priya 10-28: "add a 'later licks' that excludes lick 1"; decoder-window balance only -- the response-ratio AUC of
    # a later lick would overlap the next lick)
    P["cls"] = np.where(P.order == 0, "", " later")
    g = P.groupby(["set", "cls", "label", "animal", "epoch", "pos"])
    S = g[["angle", *lbr.BAL]].mean().join(g.size().rename("n")).reset_index()
    S = S[S.n >= MIN_LICKS]
    S["set"] = S.set.map(SETMAP) + S.cls
    S = S.drop(columns="cls")
    R = pd.read_csv(il.OUT_W / "lateral_ratio_sessions.csv")
    R = R[R.spout.isin(lo.POS)].pivot_table(index=["set", "label", "spout"], columns="pair", values="logr_0-1")
    R.columns = [f"resp_{c}" for c in R.columns]
    S = S.merge(R.reset_index().rename(columns={"spout": "pos"}), on=["set", "label", "pos"], how="left")
    S = S.rename(columns={f"bal_{a}": f"dec_{a}" for a in lbr.PAIRS})
    S["dec_orofacial"] = S[[f"dec_{a}" for a in lbr.PAIRS]].mean(1)   # = the mean of the four pair balances
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
                if G.empty:                              # later licks have no response-ratio measure
                    continue
                G["ac"] = G.angle - mu
                for (animal, ep), h in G.groupby(["animal", "epoch"]):      # rule 8: per animal x EPOCH OLS
                    if len(h) >= 6 and h.ac.std() > 0:
                        b = np.polyfit(h.ac, h.y, 1)
                        an.append({"set": st, "measure": meas, "pair": a, "animal": animal, "epoch": ep,
                                   "post": int(ep != "pre"), "slope": b[0], "at_mean_angle": b[1], "n": len(h)})
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
    print("== per animal (rule 8): OLS slope (per deg) and value at the pre-mean angle, per epoch")
    print(A.pivot_table(index=["set", "measure", "pair", "animal"], columns="epoch",
                        values=["slope", "at_mean_angle"]).reindex(columns=il.EP4, level=1).round(4).to_string())
    print("== mixed models\n" + M.round(4).to_string(index=False))
    return {"S": S, "M": M, "A": A}


ROWLAB = {"cued": "CUED first lick", "cued later": "CUED LATER licks\n(order >= 1)", "iti": "ITI bout start",
          "iti later": "ITI LATER licks\n(order >= 1 in the bout)"}


def _b(v: float, p: float, fmt: str) -> str:
    """Value as text, BOLD (mathtext) when p < 0.05 (Priya 10-28: "bold significance in the titles")."""
    if not np.isfinite(v):
        return "n/a"
    s = format(v, fmt)
    return rf"$\mathbf{{{s}}}$" if np.isfinite(p) and p < 0.05 else s


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
        sets = ("cued", "cued later", "iti", "iti later") if meas == "dec" else ("cued", "iti")
        fig, axs = plt.subplots(len(sets), len(PAIRS), figsize=(24, 4.75 * len(sets)), squeeze=False)
        for i, st in enumerate(sets):
            G = S[S.set == st]
            mu = G[G.epoch == "pre"].angle.mean()
            for j, a in enumerate(PAIRS):
                ax = axs[i, j]
                y = f"{meas}_{a}"
                for ep in il.EP4:
                    h = G[G.epoch == ep]
                    ax.plot(h.angle, h[y], "o", color=col[ep], ms=2.5, alpha=0.18, mec="none")
                # one fit line per epoch (Priya 10-28: "a slope for each epoch"), from y ~ angle * C(epoch)
                me = M[(M.set == st) & (M.measure == meas) & (M.pair == a) & M.model.isin(EP)].set_index("model")
                if len(me) and np.isfinite(me.slope_pre.iloc[0]):
                    sp0 = me.slope_pre.iloc[0]
                    x = np.linspace(np.nanpercentile(G.angle, 2), np.nanpercentile(G.angle, 98), 50)
                    # intercepts are not stored; the pre line goes through the pre group mean, epochs = pre + shift
                    pre = G[G.post == 0]
                    b0 = np.nanmean(pre[y]) - sp0 * (np.nanmean(pre.angle) - mu)
                    ax.plot(x, b0 + sp0 * (x - mu), color="k", lw=1.7, zorder=4)
                    for ep in EP:
                        if ep in me.index:
                            r = me.loc[ep]
                            ax.plot(x, b0 + r.shift_post_same_angle + (sp0 + r.slope_change_post) * (x - mu),
                                    color=col[ep], lw=1.4, ls="--", zorder=4)
                    sc = " / ".join(_b(me.slope_change_post.get(e, np.nan), me.p_slope_change.get(e, np.nan), "+.3f")
                                    for e in EP)
                    sh = " / ".join(_b(me.shift_post_same_angle.get(e, np.nan), me.p_shift.get(e, np.nan), "+.2f")
                                    for e in EP)
                    ax.set_title(f"{a}, {st}: pre slope {_b(sp0, me.p_slope_pre.iloc[0], '+.3f')} per deg\n"
                                 f"slope change a/s/c {sc}\nshift at same angle a/s/c {sh}", fontsize=8)
                ax.axhline(0, color="0.8", lw=0.6)
                ax.axvline(0, color="0.8", lw=0.6)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if i == len(sets) - 1:
                    ax.set_xlabel("lick angle from midline (deg, + = contralesional)", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel(f"{ROWLAB[st]}\n{lab}", fontsize=8)
        fig.legend(handles=[Line2D([], [], color=col[e], marker="o", lw=0, label=e) for e in il.EP4]
                   + [Line2D([], [], color="k", lw=1.7, label="pre fit")]
                   + [Line2D([], [], color=col[e], lw=1.4, ls="--", label=f"{e} fit") for e in EP],
                   loc="lower center", ncol=8, fontsize=9, frameon=False)
        fig.suptitle(f"Hemispheric balance vs lick angle by epoch ({meas}): one point per session x spout (cued: target; "
                     "ITI: previous trial's spout); mixed model y ~ angle * epoch + (1 | animal), sessions x spouts as "
                     "observations\n(anti-conservative at 4 animals; per-animal x epoch slopes in balance_angle_animals.csv). "
                     "Titles: a/s/c = acute / subacute / chronic vs pre; BOLD = p < 0.05; 'shift at same angle' = epoch "
                     "minus pre at the pre-mean angle. PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.92))
        save_png_svg(fig, il.OUT_W / "figures" / f"lick_balance_angle_{meas}.png", dpi=100)
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

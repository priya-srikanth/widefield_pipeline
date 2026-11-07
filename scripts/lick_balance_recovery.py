"""Does the orofacial hemispheric BALANCE predict behavioral RECOVERY? (Priya 10-28: "LMM of orofacial balance and recovery
(contra lick angle) -- does it predict recovery").

Recovery axis = the far-contra FIRST-cued-lick aim offset vs the animal's pre landing (deg, - = ipsilesional; key figure
1A; `aim_deficit_sessions.csv` farcontra_first). Balance = the session mean of X_right - X_left (decoder's 0..0.2 s
window) at the same lick ANGLE and ORDER (`lick_balance_regions` balreg_sessions.csv; orofacial pool = ALM, tjM1, SSp-m,
SSp-n; non-orofacial pool = the control). Post-stroke sessions; statsmodels MixedLM, random intercept per animal, ML;
sessions as observations (anti-conservative at 4 animals; per-animal correlations printed first, rule 8).
  (1) CONCURRENT  aim ~ bal_within + bal_animal + log(day)
  (2) PREDICTIVE  aim(next session) - aim(this session) ~ bal_within + bal_animal + aim(this session) + log(day)
      (does this session's balance predict how much the aim improves by the next session, beyond regression to the mean
      and time?)
Outputs <OUT_W>/balrec_sessions.csv, balrec_models.csv; figure lick_balance_recovery.png (figdata + --plot-only).
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

GROUPS = ["orofacial", "non-orofacial"]


def compute() -> dict:
    import statsmodels.formula.api as smf
    B = pd.read_csv(il.OUT_W / "balreg_sessions.csv")
    A = pd.read_csv(il.OUT_W / "aim_deficit_sessions.csv", usecols=["label", "animal", "day", "farcontra_first"])
    rows, an, out = [], [], []
    for st, b in B.groupby("set"):
        S = b.drop(columns=["day"]).merge(A.drop(columns="animal"), on="label", how="inner")
        S = S[(S.epoch != "pre") & (S.day > 0)].dropna(subset=["farcontra_first"]).sort_values(["animal", "day"]).copy()
        S["logday"] = np.log(S.day)
        S["aim"] = S.farcontra_first
        S["aim_next"] = S.groupby("animal").aim.shift(-1)
        S["d_aim"] = S.aim_next - S.aim
        S["set"] = st
        out.append(S)
        for grp in GROUPS:
            v = f"r_{grp}"
            g = S.dropna(subset=[v]).copy()
            g["w"] = g[v] - g.groupby("animal")[v].transform("mean")
            g["b"] = g.groupby("animal")[v].transform("mean")
            for a_, h in g.groupby("animal"):
                an.append({"set": st, "group": grp, "animal": a_, "n": len(h),
                           "r_concurrent": np.corrcoef(h[v], h.aim)[0, 1] if len(h) > 3 else np.nan,
                           "r_predictive": (np.corrcoef(h.dropna(subset=["d_aim"])[v], h.dropna(subset=["d_aim"]).d_aim)[0, 1]
                                            if h.d_aim.notna().sum() > 3 else np.nan)})
            for name, f, data in (("concurrent", "aim ~ w + b + logday", g),
                                  ("predictive", "d_aim ~ w + b + aim + logday", g.dropna(subset=["d_aim"]))):
                try:
                    m = smf.mixedlm(f, data, groups=data["animal"]).fit(reml=False)
                    ci = m.conf_int()
                    rows.append({"set": st, "group": grp, "model": name, "n": len(data), "slope": m.fe_params["w"],
                                 "lo": ci.loc["w", 0], "hi": ci.loc["w", 1], "p": m.pvalues["w"],
                                 "logday": m.fe_params["logday"], "p_logday": m.pvalues["logday"]})
                except Exception as ex:
                    rows.append({"set": st, "group": grp, "model": f"{name} FAILED {ex}", "n": len(data)})
    S = pd.concat(out, ignore_index=True)
    S.to_csv(il.OUT_W / "balrec_sessions.csv", index=False)
    M, AN = pd.DataFrame(rows), pd.DataFrame(an)
    M.to_csv(il.OUT_W / "balrec_models.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): within-animal r, balance vs aim (concurrent) and vs next-session aim change (predictive)")
    print(AN.pivot_table(index=["set", "group"], columns="animal", values=["r_concurrent", "r_predictive"]).round(2)
          .to_string())
    print("== models (slope = deg of aim per balance unit, within animal)\n" + M.round(4).to_string(index=False))
    return {"S": S, "M": M, "AN": AN}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    S, M, AN = F["S"], F["M"], F["AN"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    for i, st in enumerate(("cued", "ITI")):
        for j, (model, yv, ylab) in enumerate((("concurrent", "aim", "far-contra first-lick aim (deg vs pre; - = ipsilesional)"),
                                               ("predictive", "d_aim", "change in aim to the NEXT session (deg)"))):
            ax = axs[i, j]
            g = S[S.set == st]
            pre = F.get("pre")
            if model == "concurrent" and pre is not None:      # pre sessions, left-out reference (not in any fit)
                p_ = pre[pre.set == st]
                ax.plot(p_.oro_bal, p_.aim, "o", color="0.82", mec="0.6", mew=0.4, ms=6, zorder=0)
            for (ep, a_), h in g.groupby(["epoch", "animal"]):
                ax.plot(h.r_orofacial, h[yv], ANIMAL_MARKER.get(a_, "o"), color=EPOCH_LINE[ep], ms=6, mec="k", mew=0.3)
            m = M[(M.set == st) & (M.group == "orofacial") & (M.model == model)]
            r = AN[(AN.set == st) & (AN.group == "orofacial")].set_index("animal")[f"r_{model}"]
            ttl = f"{st}, orofacial balance, {model}"
            if len(m) and "slope" in m and np.isfinite(m.slope.iloc[0]):
                s_ = f"within-animal slope {m.slope.iloc[0]:+.2f} (p {m.p.iloc[0]:.2g})"
                ttl += "\n" + (rf"$\mathbf{{{s_.replace(' ', '~')}}}$" if m.p.iloc[0] < 0.05 else s_)
            mc = M[(M.set == st) & (M.group == "non-orofacial") & (M.model == model)]
            if len(mc) and "p" in mc and np.isfinite(mc.p.iloc[0]):
                ttl += f"\n(non-orofacial: {mc.slope.iloc[0]:+.2f}, p {mc.p.iloc[0]:.2g})"
            ttl += "\nper-animal r: " + " / ".join(f"{r.get(x, np.nan):+.2f}" for x in ("PS92", "PS93", "PS94", "PS95"))
            ax.set_title(ttl, fontsize=8.5)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.axvline(0, color="0.7", lw=0.6)
            ax.set_xlabel("session orofacial balance at the same angle & order (vs pre)", fontsize=8.5)
            ax.set_ylabel(ylab, fontsize=8.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    h = [Line2D([], [], color=EPOCH_LINE[e], marker="o", lw=0, label=e) for e in ("acute", "subacute", "chronic")]
    h += [Line2D([], [], color="0.3", marker=ANIMAL_MARKER.get(x, "o"), lw=0, label=x) for x in ("PS92", "PS93", "PS94", "PS95")]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=9, frameon=False)
    fig.suptitle("Does the orofacial hemispheric balance predict RECOVERY (far-contra first-lick aim)? One point per "
                 "post-stroke session (gray = pre sessions, left-out reference, not fitted). Mixed models with log(day) and a random intercept per animal (sessions as "
                 "observations, anti-conservative):\nleft = concurrent (aim ~ balance), right = predictive (next-session "
                 "aim change ~ this session's balance + current aim). Within-animal slope bold when p < 0.05. PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_balance_recovery.png", dpi=100)
    plt.close(fig)


def _pre_cloud() -> pd.DataFrame | None:
    """Pre sessions' orofacial balance (scored leave-one-pre-session-out) and aim, from lick_recovery_markers (Priya 10-28:
    "add pre references as gray cloud"); None if that script has not run."""
    f = il.OUT_W / "recmark_sessions.csv"
    if not f.exists():
        return None
    R = pd.read_csv(f, usecols=["set", "label", "epoch", "oro_bal", "aim"])
    return R[R.epoch == "pre"].dropna()


def main() -> int:
    if plot_only():
        F = load_fd("lick_balance_recovery")
        F["pre"] = _pre_cloud()
        plot(F)
        return 0
    F = compute()
    save_fd("lick_balance_recovery", F)
    F["pre"] = _pre_cloud()
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

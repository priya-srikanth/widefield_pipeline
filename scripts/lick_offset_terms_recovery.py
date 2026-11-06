"""Which PART of the decoder's like-for-like offset tracks recovery? (Priya 10-29: "prioritize the ITI per-area
decomposition").

The frozen lateral decoder is linear, so each session's offset = sum over features of beta_k * dh_k (dh = the feature's
change vs pre licks that landed at the same place; lick_offset_contrib_map, terms_sessions_<tag>.csv; pre sessions are
referenced to the OTHER pre sessions). For every left / right area pair the two terms split EXACTLY into
  DIFFERENCE part  1/2 (beta_r - beta_l)(dh_r - dh_l)   -- the hemispheric (left / right) component
  SUM part         1/2 (beta_r + beta_l)(dh_r + dh_l)   -- the component shared by the two hemispheres
(areas without a partner: 'unpaired'). Components per session: total offset; difference / sum parts summed over all
pairs; SSp-n difference alone; orofacial + SSs difference without SSp-n; orofacial + SSs sum; all other areas.
Each component vs recovery (ITI -> ITI's own lick-angle axis and the cued aim; cued -> the cued aim) with
lick_recovery_markers.models (per-animal r, within-animal MixedLM with log(day), leave-one-animal-out with shuffle null,
within-animal blocked, LOSO). Per-animal tables first (rule 8).
Outputs <OUT_W>/offterm_sessions.csv, offterm_cv.csv, offterm_models.csv; figure lick_offset_terms_recovery.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_recovery_markers as lrm  # noqa: E402

TAG = "areas_splitr8"
ORO = {"ALM", "tjM1", "SSp-m", "SSp-n", "SSs"}
COMPS = [("total", "total offset"), ("diff_all", "left/right DIFFERENCE, all pairs"), ("sum_all", "SHARED (sum), all pairs"),
         ("diff_SSp-n", "SSp-n difference"), ("diff_oro_noSSpn", "oro+SSs difference w/o SSp-n"),
         ("sum_oro", "oro+SSs shared"), ("other", "all other areas"), ("unpaired", "unpaired areas")]
TARGETS = {"iti": [("iti_ang", "ITI lick-angle axis"), ("aim", "cued far-contra aim")], "trial": [("aim", "cued far-contra aim")]}


def components(T: pd.DataFrame) -> pd.DataFrame:
    T = T.copy()
    T["area"] = T.feature.str.rsplit("_", n=1).str[0]
    T["hemi"] = T.feature.str.rsplit("_", n=1).str[1]
    rows = []
    for (st, an, lab, ep), g in T.groupby(["set", "animal", "label", "epoch"]):
        rec = {"set": st, "animal": an, "label": lab, "epoch": ep, "total": g.term.sum(), "diff_all": 0.0, "sum_all": 0.0,
               "diff_SSp-n": 0.0, "diff_oro_noSSpn": 0.0, "sum_oro": 0.0, "other": 0.0, "unpaired": 0.0}
        for ar, h in g.groupby("area"):
            h = h.set_index("hemi")
            if {"right", "left"} <= set(h.index):
                br, bl, dr, dl = (h.loc["right", "beta"], h.loc["left", "beta"], h.loc["right", "dh"], h.loc["left", "dh"])
                d_, s_ = 0.5 * (br - bl) * (dr - dl), 0.5 * (br + bl) * (dr + dl)
                rec["diff_all"] += d_
                rec["sum_all"] += s_
                if ar == "SSp-n":
                    rec["diff_SSp-n"] += d_
                elif ar in ORO:
                    rec["diff_oro_noSSpn"] += d_
                if ar in ORO:
                    rec["sum_oro"] += s_
                else:
                    rec["other"] += d_ + s_
            else:
                rec["unpaired"] += h.term.sum()
        rows.append(rec)
    S = pd.DataFrame(rows)
    chk = (S.diff_all + S.sum_all + S.unpaired - S.total).abs().max()
    print(f"== exactness: max |diff + sum + unpaired - total| = {chk:.2e}", flush=True)
    return S


def main() -> int:
    T = pd.read_csv(il.OUT_W / "offset_contrib" / f"terms_sessions_{TAG}.csv")
    S = components(T)
    R = pd.read_csv(il.OUT_W / "recmark_sessions.csv", usecols=["label", "day", "aim", "iti_ang"]).drop_duplicates("label")
    S = S.merge(R, on="label", how="left")
    S.to_csv(il.OUT_W / "offterm_sessions.csv", index=False)
    pd.set_option("display.width", 230)
    print("== epoch means (mean of animals, change vs pre is the pre-referenced value itself)\n"
          + S.groupby(["set", "epoch"])[[c for c, _ in COMPS]].mean().round(3).to_string())
    lrm.N_SHUF = 1000
    out = {k: [] for k in ("M", "CV", "AN")}
    for st, ys in TARGETS.items():
        for yv, _yl in ys:
            G = S[S.set == st][["set", "label", "animal", "epoch", "day", *[c for c, _ in COMPS]]].assign(
                aim=S.loc[S.set == st, yv])
            F = lrm.models(G)
            for k in out:
                out[k].append(F[k].assign(y=yv))
    F = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    F["CV"].to_csv(il.OUT_W / "offterm_cv.csv", index=False)
    F["M"].to_csv(il.OUT_W / "offterm_models.csv", index=False)
    print("== per animal r (rule 8)\n" + F["AN"].pivot_table(index=["set", "y", "pred"], columns="animal", values="r")
          .round(2).to_string())
    X = F["CV"].merge(F["M"][["set", "y", "pred", "slope", "p"]], on=["set", "y", "pred"], how="left")
    print("== cross-validated + within-animal\n" + X[["set", "y", "pred", "loao", "p_shuffle", "loblk", "loblk_plus_days",
                                                      "loso_unknown", "p"]].round(3).to_string(index=False))
    plot(S, X, F["AN"])
    return 0


def plot(S, X, AN) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(st, yv, yl) for st, ys in TARGETS.items() for yv, yl in ys]
    fig, axs = plt.subplots(2, len(panels), figsize=(6.5 * len(panels), 10), squeeze=False)
    names = dict(COMPS)
    for j, (st, yv, yl) in enumerate(panels):
        ax = axs[0, j]
        e = S[S.set == st].groupby(["epoch", "animal"])[[c for c, _ in COMPS]].mean().groupby("epoch").mean()
        e = e.reindex(["pre", "acute", "subacute", "chronic"])
        x = np.arange(len(COMPS))
        for k, ep in enumerate(e.index):
            ax.bar(x + (k - 1.5) * 0.2, e.loc[ep].to_numpy(), width=0.19, label=ep,
                   color=["0.7", "#C2185B", "#B8860B", "#00695C"][k])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, [names[c] for c, _ in COMPS], rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("offset component (mm), mean of animals")
        ax.set_title(f"{'ITI' if st == 'iti' else 'cued'} licks: the decoder offset split into exact components", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax = axs[1, j]
        c = X[(X.set == st) & (X.y == yv)].set_index("pred")
        ks = ["days", *[k for k, _ in COMPS]]
        v = [c.loc[k, "loao"] if k in c.index else np.nan for k in ks]
        ax.bar(np.arange(len(ks)), v, color=["0.55", *["#C44E52" if "diff" in k else "#4C72B0" for k in ks[1:]]])
        for xi, k in enumerate(ks):
            if k in c.index and np.isfinite(c.loc[k, "p_shuffle"]):
                ax.text(xi, max(v[xi], 0) + 0.03, f"p {c.loc[k, 'p_shuffle']:.2g}", ha="center", fontsize=6.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(max(-1, np.nanmin(v) - 0.05), 1)
        ax.set_xticks(np.arange(len(ks)), ["days since stroke", *[names[k] for k in ks[1:]]], rotation=35, ha="right",
                      fontsize=7.5)
        ax.set_title(f"{'ITI' if st == 'iti' else 'cued'} components -> {yl}: unseen-animal cvR²", fontsize=9)
        for a_ in axs[:, j]:
            for s_ in ("top", "right"):
                a_.spines[s_].set_visible(False)
    fig.suptitle("Which part of the decoder's like-for-like offset tracks recovery? The linear decoder's offset = sum of "
                 "per-area terms; each left/right pair splits exactly into a DIFFERENCE (hemispheric) and a SHARED part.\n"
                 "Top: components by epoch; bottom: each component's leave-one-animal-out cvR² of the recovery axis "
                 "(red = hemispheric difference parts). PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(il.OUT_W / "figures" / "lick_offset_terms_recovery.png", dpi=95)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

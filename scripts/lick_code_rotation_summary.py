"""ONE COMPACT SUMMARY of the noise-matched rotation test for every lick type (Priya 10-29): cued first lick, ITI (all,
far1), ENLv (primary onset >= 1 s after arrival, and its bout starts) side by side.

Reads lick_code_noisematch figdata (noisematch.pkl = cued / ITI run, noisematch_enlv.pkl = LICK_CODE_FAMILY=enlv run);
computes nothing new. Per lick type (column) and post epoch:
  rows 1-2  observed transfer r (filled, epoch gray) vs the animal's NOISE-MATCHED unrotated null (open green), joined per
            animal (markers = animals); text = pooled real minus null, analysis_kit.boot_delta 95 % CI, n animals same sign
            (noisematch_pooled, null 'matched'). Row 1 pre->post (frozen pre decoder), row 2 post->pre (post-trained).
  row 3     the rotation the real deficit corresponds to (equiv_rotation, primary 'rigid' rotation, noise-matched curve;
            dots = animals, bar = pooled) for pre->post (blue) and post->pre (orange), and the smallest rotation the
            pooled test detects with 80 % power (theta80, noise-matched; red dash; '>90' at the top when never reached).
NO June panel: June -> August is NOT a clean no-stroke drift estimate (the imaging room was vacated and the animals were
intermittently trained on a different rig in between); the June numbers (cued / ITI only; no June ENLv extraction) are
given in the caption footnote only, as secondary context.
Output <OUT_W>/figures/lick_code_rotation_summary.png (+ lick_code_noisematch/rotation_summary_table.csv).
Usage: python scripts/lick_code_rotation_summary.py
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import pickle
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
NM = il.OUT_W / "lick_code_noisematch"
FIG = il.OUT_W / "figures"
TYPES = [("cued_first", "", "Cued: first post-cue lick"), ("iti_all", "", "ITI: all licks"),
         ("iti_far1", "", "ITI: far1 (spout docked)"), ("enlv_on1", "_enlv", "ENLv: onset >= 1 s after arrival"),
         ("enlv_bout", "_enlv", "ENLv: bout starts (>= 1 s quiet)")]
PRIMARY = "rigid"
MET = [("fwd_r", "pre -> post r (frozen pre decoder)", "#2c7fb8"), ("back_r", "post -> pre r (post-trained)", "#e08214")]


def W(s: str, n: int) -> str:
    return "\n".join(textwrap.fill(p, n) for p in s.split("\n"))


def _load(sfx: str) -> dict:
    with open(NM / "figdata" / f"noisematch{sfx}.pkl", "rb") as f:
        return pickle.load(f)


def table(F: dict) -> pd.DataFrame:
    rows = []
    for st, sfx, _lab in TYPES:
        D = F[sfx]
        for ep in EP:
            for met, _l, _c in MET:
                pp = D["PP"][(D["PP"].set == st) & (D["PP"].epoch == ep) & (D["PP"].measure == met)
                             & (D["PP"].null == "matched")]
                eq = D["EQ"][(D["EQ"].set == st) & (D["EQ"].epoch == ep) & (D["EQ"].measure == met)
                             & (D["EQ"].rot == PRIMARY)].set_index("animal").theta_eq
                t8 = D["T80"][(D["T80"].set == st) & (D["T80"].epoch == ep) & (D["T80"].measure == met)
                              & (D["T80"].rot == PRIMARY) & (D["T80"].noise == "matched")].set_index("animal").theta80
                if pp.empty:
                    continue
                p = pp.iloc[0]
                rows.append({"set": st, "epoch": ep, "measure": met, "real": p.real, "null_matched": p.null_level,
                             "delta": p.delta, "lo": p.lo, "hi": p.hi, "n_same_sign": p.n_same_sign, "n": p.n,
                             "theta_eq_pooled": eq.get("pooled", np.nan), "theta80_pooled": t8.get("pooled", np.nan),
                             **{f"theta_eq_{a}": eq.get(a, np.nan) for a in AN},
                             **{f"theta80_{a}": t8.get(a, np.nan) for a in AN},
                             **{f"d_{a}": p.get(f"d_{a}", np.nan) for a in AN}})
    return pd.DataFrame(rows)


def june_note(F: dict) -> str:
    """Secondary context only: pooled June-equivalent rotation and post deficit minus June deficit (cued / ITI)."""
    D = F[""]
    parts = []
    for st, sfx, _lab in TYPES:
        if sfx:
            continue
        eq = D["EQ"][(D["EQ"].set == st) & (D["EQ"].epoch == "june") & (D["EQ"].rot == PRIMARY)
                     & (D["EQ"].animal == "pooled")].set_index("measure").theta_eq
        vj = D["VJ"][(D["VJ"].set == st) & (D["VJ"].measure.isin(["fwd_r", "back_r"]))]
        s = (f"{st}: June-equivalent rotation pre->post {eq.get('fwd_r', np.nan):.0f} deg, post->pre "
             f"{eq.get('back_r', np.nan):.0f} deg")
        for met in ("fwd_r", "back_r"):
            v = vj[vj.measure == met].set_index("epoch")
            s += f"; {met} deficit minus June " + ", ".join(
                f"{ep[:3]} {v.loc[ep, 'delta_vs_june']:+.2f} [{v.loc[ep, 'lo']:+.2f}, {v.loc[ep, 'hi']:+.2f}]"
                for ep in EP if ep in v.index)
        parts.append(s)
    return "; ".join(parts)


def trial_set_sentence() -> str:
    """The lick sets of the one-page rotation summary (Priya 11-09): every lick type of TYPES, i.e. both lick_code runs
    (cued + ITI, and ENLv)."""
    import lick_code_subspace as lcs
    from wfield_local import trial_set_text as tst
    return tst.union(lcs.lick_sets_sentence(""), lcs.lick_sets_sentence("enlv"))


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.epoch_figures import EPOCH_GREY
    F = {"": _load(""), "_enlv": _load("_enlv")}
    T = table(F)
    T.to_csv(NM / "rotation_summary_table.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(T.round(3).to_string(index=False))
    x0 = {ep: i for i, ep in enumerate(EP)}
    fig, axs = plt.subplots(3, len(TYPES), figsize=(4.0 * len(TYPES), 11.5), squeeze=False, sharey="row")
    for j, (st, sfx, lab) in enumerate(TYPES):
        PA = F[sfx]["PA"]
        PA = PA[PA.set == st]
        for i, (met, mlab, _c) in enumerate(MET):
            ax = axs[i, j]
            for ep in EP:
                h = PA[PA.epoch == ep]
                if h.empty:
                    continue
                for _, r in h.iterrows():
                    xx = x0[ep] - 0.18 + 0.12 * AN.index(r.animal)
                    ax.plot([xx - 0.04, xx + 0.04], [r[f"{met}_nullm"], r[f"{met}_real"]], color="0.55", lw=0.8)
                    ax.plot(xx - 0.04, r[f"{met}_nullm"], MARK[r.animal], color="w", mec="#3a7d3a", ms=5, mew=1.2)
                    ax.plot(xx + 0.04, r[f"{met}_real"], MARK[r.animal], color=EPOCH_GREY[ep], mec="k", ms=6)
                q = T[(T.set == st) & (T.epoch == ep) & (T.measure == met)]
                if len(q):
                    q = q.iloc[0]
                    ax.text(x0[ep], 0.02, f"{q.delta:+.2f}\n[{q.lo:+.2f}, {q.hi:+.2f}]\n{q.n_same_sign}/{q.n}",
                            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="0.2")
            ax.set_xticks(range(len(EP)), EP, fontsize=8)
            ax.set_xlim(-0.5, len(EP) - 0.5)
            ax.set_ylim(-0.12, 0.85)
            ax.grid(axis="y", color="0.92", lw=0.6)
            if j == 0:
                ax.set_ylabel(W(mlab, 26), fontsize=8.5)
            if i == 0:
                ax.set_title(lab, fontsize=10)
        ax = axs[2, j]
        for k, (met, _mlab, col) in enumerate(MET):
            off = -0.17 if k == 0 else 0.17
            for ep in EP:
                q = T[(T.set == st) & (T.epoch == ep) & (T.measure == met)]
                if q.empty:
                    continue
                q = q.iloc[0]
                xb = x0[ep] + off
                ax.bar(xb, q.theta_eq_pooled, 0.3, color=col, alpha=0.35, edgecolor=col, lw=0.8)
                for a in AN:
                    v = q[f"theta_eq_{a}"]
                    if np.isfinite(v):
                        ax.plot(xb - 0.09 + 0.06 * AN.index(a), v, MARK[a], color="w", mec=col, ms=4.5, mew=1.1)
                t8 = q.theta80_pooled
                ax.plot([xb - 0.15, xb + 0.15], [t8 if np.isfinite(t8) else 92] * 2, color="#b0413e", lw=2.2,
                        ls="-" if np.isfinite(t8) else ":")
        ax.set_xticks(range(len(EP)), EP, fontsize=8)
        ax.set_xlim(-0.5, len(EP) - 0.5)
        ax.set_ylim(0, 95)
        ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
        ax.grid(axis="y", color="0.92", lw=0.6)
        if j == 0:
            ax.set_ylabel(W("rotation (deg): implied (bars, dots); detectable (red)", 26),
                          fontsize=8.5)
    hand = ([Line2D([], [], marker="o", ls="", color="w", mec="#3a7d3a", mew=1.2, label="noise-matched null (unrotated)"),
             Line2D([], [], marker="o", ls="", color="0.6", mec="k", label="observed post session mean")]
            + [Line2D([], [], marker=MARK[a], ls="", color="w", mec="k", label=a) for a in AN]
            + [Line2D([], [], color=c, lw=6, alpha=0.4, label=f"implied rotation, {m.split(' (')[0]}") for m, _l, c in
               [(x[1], x[0], x[2]) for x in MET]]
            + [Line2D([], [], color="#b0413e", lw=2.2, label="theta80 pooled: min. detectable rotation (10 = <= 10, the smallest grid angle; dotted at top = >90)")])
    fig.legend(handles=hand, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.06))
    fig.suptitle(W(
        "Lick-direction code after stroke, every lick type: observed decoder transfer vs a NOISE-MATCHED, UNROTATED "
        "pre-stroke null (pre sessions degraded with their own shuffled residual noise to the post epoch's within-session "
        "decodability, same lick count). Real below the null = more than a noisier code; text = pooled real minus null, "
        "boot_delta 95 % CI (animals > sessions), animals with that sign. Bottom: the injected rigid rotation of the pre code "
        "giving the same mean transfer, and the smallest rotation the pooled test detects (80 % power, one-sided 5 %). "
        "n = 4 mice, PROVISIONAL.", 210), fontsize=9.5)
    note = ("Footnote (secondary context, NOT a clean no-stroke drift reference -- between June and August the imaging room "
            "was vacated and the animals were intermittently trained on a different rig; no June ENLv extraction): "
            + june_note(F))
    fig.text(0.01, 0.003, W(note, 260), fontsize=6.6, color="0.3", va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    FIG.mkdir(parents=True, exist_ok=True)
    save_png_svg(fig, FIG / "lick_code_rotation_summary.png", dpi=130)
    plt.close(fig)
    print("written", FIG / "lick_code_rotation_summary.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

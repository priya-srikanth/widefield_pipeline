"""SAME TARGET, SAME LANDING PLACE, NO EXTRAPOLATION (Priya 10-29: "go ahead with the post-stroke vs pre-stroke at same
error"). Her framing: post-stroke off-target licks look like PRE-stroke off-target licks to the same target landing at the
same place; such errors are just far more frequent (and larger) after the stroke. Test it where it can be tested without
extrapolation: keep only post-stroke licks whose landed angle lies INSIDE the animal's pre-stroke range for that target
spout (2.5-97.5 percentile of its pre licks to that spout) and compare them with pre-stroke licks to the same spout
landing at the same angle.

Per lick (baltarget_licks.csv.gz from lick_balance_target_model.py; same licks / responses as the primary balance):
c, i = summed post-lick response (0.1-0.5 s minus -1..-0.5 s) of ALM, tjM1, SSp-m, SSp-n, SSs in the contralesional /
ipsilesional hemisphere; angle + = contralesional; pos = target spout (ITI: the spout in place during the ITI).
Reference = WITHIN-SPOUT pre line: per animal x spout x hemisphere, OLS of the response on angle over pre licks to that
spout (pre sessions scored leave-one-session-out). Session balance over its matched licks:
  bal_spout = log2(sum c / sum pred_c) - log2(sum i / sum pred_i)      (+ = contralesional hemisphere larger)
and, for comparison, bal_angle = the same with the primary-style ACROSS-spout angle line (target-blind) on the same licks.
Decoder: the frozen 12.17 decoder's per-lick like-for-like read-out (offset_contrib terms_licks `readout`, + = read
contralesional), referenced the same way (pre within-spout line on landing `act`, mm) -> off_spout.
If Priya's framing holds: bal_spout and off_spout ~ 0 after the stroke, while bal_angle stays shifted.
Per-animal table first (rule 8), then the paired nested bootstrap (analysis_kit.boot_delta) of change vs pre.
Outputs <OUT_W>/balmatch_sessions.csv, balmatch_tests.csv, balmatch_retained.csv; figure lick_balance_matched_error.png.
Usage: python scripts/lick_balance_matched_error.py [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT = il.OUT_W
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
Q = (2.5, 97.5)
MIN_LICKS = 10
MEAS = ["bal_spout", "bal_angle", "off_spout", "lerr"]


def _readouts(L: pd.DataFrame) -> pd.DataFrame:
    """attach the frozen decoder's per-lick read-out (same key as lick_balance_target_model.decoder)."""
    T = pd.read_csv(OUT / "offset_contrib" / "terms_licks_areas_splitr8.csv.gz",
                    usecols=["set", "label", "act", "order", "pos", "readout"])
    T["set"] = T.set.replace({"cued": "trial"})
    key = ["set", "label", "order", "pos", "act"]
    L = L.copy()
    for df in (T, L):
        df["act"] = df.act.round(3)
        df["dup"] = df.groupby(key).cumcount()
    return L.merge(T, on=[*key, "dup"], how="left")


def _line(tr: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    if len(tr) < 10 or tr[x].std() == 0:
        return 0.0, float(tr[y].mean()) if len(tr) else np.nan
    b1, b0 = np.polyfit(tr[x], tr[y], 1)
    return b1, b0


def _predict(g: pd.DataFrame, by_spout: bool) -> pd.DataFrame:
    """pred_c, pred_i (angle) and pred_r (readout on act) from pre licks; pre sessions leave-one-session-out."""
    g = g.copy()
    for col in ("pred_c", "pred_i", "pred_r"):
        g[col] = np.nan
    pre = g[g.epoch == "pre"]
    groups = [(p, g.pos == p, pre[pre.pos == p]) for p in g.pos.unique()] if by_spout else [(None, g.pos == g.pos, pre)]
    for _p, sel, pr in groups:
        for lab in [None, *pr.label.unique()]:
            tr = pr[pr.label != lab] if lab else pr
            rows = sel & ((g.label == lab) if lab else (g.epoch != "pre"))
            if not rows.any():
                continue
            for y, x, col in (("c", "angle", "pred_c"), ("i", "angle", "pred_i"), ("readout", "act", "pred_r")):
                t = tr.dropna(subset=[y, x])
                b1, b0 = _line(t, x, y)
                g.loc[rows, col] = b0 + b1 * g.loc[rows, x]
    return g


def _bal(h: pd.DataFrame) -> float:
    sc, sp, si, sq = h.c.sum(), h.pred_c.sum(), h.i.sum(), h.pred_i.sum()
    if min(sc, sp, si, sq) <= 0:
        return np.nan
    return float(np.log2(sc / sp) - np.log2(si / sq))


def compute() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from wfield_local import analysis_kit as ak
    L = _readouts(pd.read_csv(OUT / "baltarget_licks.csv.gz"))
    srows, kept = [], []
    for (st, an), g in L.groupby(["set", "animal"]):
        g = g.dropna(subset=["angle", "c", "i"]).copy()
        pre = g[g.epoch == "pre"]
        lo = pre.groupby("pos").angle.quantile(Q[0] / 100)
        hi = pre.groupby("pos").angle.quantile(Q[1] / 100)
        g["inside"] = (g.angle >= g.pos.map(lo)) & (g.angle <= g.pos.map(hi))
        g["lerr"] = g.angle - g.pos.map(pre.groupby("pos").angle.mean())
        for ep, h in g.groupby("epoch"):
            kept.append({"set": st, "animal": an, "epoch": ep, "n": len(h), "frac_inside": h.inside.mean()})
        gs = _predict(g, by_spout=True)
        ga = _predict(g, by_spout=False)
        gs["pred_c_a"], gs["pred_i_a"] = ga.pred_c, ga.pred_i
        m = gs[gs.inside]
        for (lab, ep), h in m.groupby(["label", "epoch"]):
            if len(h) < MIN_LICKS:
                continue
            ha = h.assign(pred_c=h.pred_c_a, pred_i=h.pred_i_a)
            r = h.dropna(subset=["readout", "pred_r"])
            srows.append({"set": st, "animal": an, "label": lab, "epoch": ep, "n": len(h),
                          "bal_spout": _bal(h), "bal_angle": _bal(ha), "lerr": h.lerr.mean(),
                          "off_spout": float((r.readout - r.pred_r).mean()) if len(r) >= MIN_LICKS else np.nan})
        print(f"[{st} {an}] matched sessions done", flush=True)
    S = pd.DataFrame(srows)
    K = pd.DataFrame(kept)
    rows = []
    for st, g in S.groupby("set"):
        for c in MEAS:
            pre = {a: h[c].dropna().to_numpy() for a, h in sorted(g[g.epoch == "pre"].groupby("animal"))}
            pre = {a: v for a, v in pre.items() if len(v)}
            for ep in EPS[1:]:
                post = {a: h[c].dropna().to_numpy() for a, h in sorted(g[g.epoch == ep].groupby("animal"))}
                post = {a: v for a, v in post.items() if len(v) and a in pre}
                if not post:
                    continue
                ci = ak.boot_delta(post, pre, np.random.default_rng(0))
                am = {a: post[a].mean() - pre[a].mean() for a in post}
                rows.append({"set": st, "measure": c, "epoch": ep, "point": ci.point, "lo": ci.lo, "hi": ci.hi,
                             "n_neg": sum(v < 0 for v in am.values()), "n_animals": len(am),
                             "n_sessions": sum(len(v) for v in post.values()), **{f"d_{a}": v for a, v in am.items()}})
    return S, pd.DataFrame(rows), K


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = pd.read_csv(OUT / "balmatch_tests.csv")
    K = pd.read_csv(OUT / "balmatch_retained.csv")
    titles = {"bal_angle": "balance, ACROSS-spout angle reference\n(target-blind, primary-like)",
              "bal_spout": "balance, WITHIN-spout reference\n(same target, same angle)",
              "off_spout": "frozen decoder offset (mm), within-spout\n(same target, same landing place)",
              "lerr": "landing error of the kept licks (deg)\n(angle - pre mean at the spout)"}
    sets = [s for s in ("trial", "iti") if s in T.set.unique()]
    fig, axs = plt.subplots(len(sets), 5, figsize=(24, 5.2 * len(sets)), squeeze=False)
    mk = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
    for i, st in enumerate(sets):
        for j, c in enumerate(["bal_angle", "bal_spout", "off_spout", "lerr"]):
            ax = axs[i, j]
            t = T[(T.set == st) & (T.measure == c)].set_index("epoch").reindex(EPS[1:])
            x = np.arange(len(t))
            ax.errorbar(x, t.point, yerr=[t.point - t.lo, t.hi - t.point], fmt="o", color="k", capsize=4)
            for a in AN:
                if f"d_{a}" in t:
                    ax.plot(x + 0.12, t[f"d_{a}"], mk[a], mfc="none", color="#555", ms=5, label=a if j == 0 else None)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(x, EPS[1:])
            ax.set_title(f"{'CUED first lick' if st == 'trial' else 'ITI first after quiet'}: {titles[c]}\n"
                         "change vs pre (bar = 95% paired nested bootstrap; open = animals)", fontsize=8.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        ax = axs[i, 4]
        k = K[K.set == st].pivot_table(index="epoch", columns="animal", values="frac_inside").reindex(EPS)
        for a in k.columns:
            ax.plot(np.arange(len(k)), k[a], marker=mk.get(a, "o"), label=a)
        ax.set_xticks(np.arange(len(k)), EPS)
        ax.set_ylim(0, 1.02)
        ax.set_title("fraction of licks KEPT\n(landed inside the pre range for their spout)", fontsize=8.5)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    fig.suptitle("Post-stroke licks vs PRE-stroke licks to the SAME target landing at the SAME place (only licks inside "
                 "the pre range for their spout: no extrapolation). Balance + = contralesional hemisphere larger "
                 "(hemispheric sign); offset + = read contralesional. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_png_svg(fig, OUT / "figures" / "lick_balance_matched_error.png", dpi=90)
    plt.close(fig)


def trial_set_sentence() -> str:
    """The lick set of lick_balance_matched_error (Priya 11-09): baltarget_licks (lick_balance_target_model), kept only
    inside the animal's pre Q percentile angle range at the same spout, sessions with >= MIN_LICKS kept licks."""
    import lick_balance_target_model as tm
    keep = [f"kept only inside the animal's pre {Q[0]:g}-{Q[1]:g} percentile lick-angle range at the same spout",
            f"sessions with >= {MIN_LICKS} kept licks"]
    return tm.lick_set_sentence(cued_extra=keep, iti_extra=keep)


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    if "--plot-only" not in sys.argv:
        S, T, K = compute()
        S.to_csv(OUT / "balmatch_sessions.csv", index=False)
        T.to_csv(OUT / "balmatch_tests.csv", index=False)
        K.to_csv(OUT / "balmatch_retained.csv", index=False)
        pd.set_option("display.width", 250)
        print("== fraction of licks kept (inside pre range for their spout)\n"
              + K.pivot_table(index=["set", "epoch"], columns="animal", values="frac_inside").round(2).to_string())
        print("== change vs pre, per animal then bootstrap\n" + T.round(3).to_string(index=False))
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

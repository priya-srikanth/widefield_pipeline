"""Hemispheric C/I balance of the post-lick primary-pool response PER SPOUT x EPOCH, in two reference frames (10-29).

QUESTION. After the stroke, is the contralesional/ipsilesional (C/I) balance of the lick response shifted at every
spout, or only at some (far_contra)? And does the shift survive referencing each lick to WHERE IT LANDED (executed
angle) rather than to the spout it was aimed at?

INPUT <OUT_W>/baltarget_licks.csv.gz (built by lick_balance_target_model.py; see its docstring for conventions): one row
per first lick, set = 'trial' (cued first post-cue lick) or 'iti' (ITI first lick after >= 2 s quiet); pos = the
trial's spout (cued) or the previous trial's spout (ITI); angle + = CONTRALESIONAL; c / i = post-lick primary-pool
response (ALM, tjM1, SSp-m, SSp-n, SSs) of the contralesional / ipsilesional hemisphere.

SIGN CONVENTION: HEMISPHERE sign throughout. + = the CONTRALESIONAL hemisphere is relatively larger (C/I up).

PER set x animal x epoch x session x spout (>= MIN_LICKS licks):
  raw_ci       log2(sum c / sum i)
  d_raw_ci     raw_ci - the animal's pre mean raw_ci at the SAME spout          ("relative to the target")
  bal_angle    log2(sum c / sum pred_c) - log2(sum i / sum pred_i); pred_c, pred_i from the animal's PRE linear fit of
               c and of i on angle; pre sessions are predicted leave-one-pre-session-out
  d_bal_angle  bal_angle - the animal's pre mean bal_angle at the same spout  ("relative to where the lick landed")
  lerr         mean(angle - the animal's pre mean angle at that spout) (deg); - = landed ipsilesional of pre licks
Pooled over spouts: per session, the mean over its spouts ('all'); 'other' = mean over the five spouts other than
far_contra.

CHANGE VS PRE per epoch (acute / subacute / chronic): analysis_kit.boot_delta, post sessions vs pre sessions of the same
animal (paired nested animals -> sessions bootstrap, rng seed 0 per test). Per-animal differences of means are printed
and saved FIRST (rule 8), with n animals sharing the pooled sign.

OUTPUTS (<OUT_W>): ci_spout_epoch_sessions.csv (session x spout values), ci_spout_epoch_animals.csv (per animal x epoch x
spout: level and change vs pre), ci_spout_epoch_tests.csv (pooled change + 95% CI + n same sign + per-animal columns);
figures/lick_ci_spout_epoch.png and figures/lick_ci_spout_epoch_farcontra.png. --plot-only redraws from the CSVs.
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT = Path(il.OUT_W)
AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POST = EPS[1:]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
GROUPS = POS + ["all", "other"]
SETS = {"trial": "cued", "iti": "ITI"}
VARS = ("d_raw_ci", "d_bal_angle", "lerr")
YLAB = {"d_raw_ci": "d raw C/I vs pre, same spout\n(log2; + = contralesional hemisphere\nrelatively larger, C/I up)",
        "d_bal_angle": "d angle-referenced balance vs pre\n(log2; + = contralesional hemisphere\nlarger than landed angle "
                       "predicts)",
        "lerr": "landing error vs pre target\n(deg; - = landed ipsilesional)"}
FRAME = {"d_raw_ci": "relative to the TARGET (spout)", "d_bal_angle": "relative to WHERE THE LICK LANDED",
         "lerr": "behaviour"}
COL = {"pre": "0.45", "acute": "#C44E52", "subacute": "#DD8452", "chronic": "#4C72B0"}
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
MIN_LICKS = 5


def sessions(L: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (st, a), g in L.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"]
        tgt = pre.groupby("pos").angle.mean()
        full = (np.polyfit(pre.angle, pre.c, 1), np.polyfit(pre.angle, pre.i, 1))
        loso = {}
        for lab in pre.label.unique():
            o = pre[pre.label != lab]
            loso[lab] = (np.polyfit(o.angle, o.c, 1), np.polyfit(o.angle, o.i, 1))
        for (ep, lab, pos), h in g.groupby(["epoch", "label", "pos"]):
            if len(h) < MIN_LICKS:
                continue
            bc, bi = loso[lab] if ep == "pre" else full
            pc, pi = np.polyval(bc, h.angle).sum(), np.polyval(bi, h.angle).sum()
            bal = (np.log2(h.c.sum() / pc) - np.log2(h.i.sum() / pi)) if pc > 0 and pi > 0 else np.nan
            rows.append({"set": st, "animal": a, "epoch": ep, "label": lab, "day": h.day.iloc[0], "pos": pos,
                         "n": len(h), "raw_ci": np.log2(h.c.sum() / h.i.sum()), "bal_angle": bal,
                         "lerr": float((h.angle - tgt[pos]).mean())})
    S = pd.DataFrame(rows)
    ref = S[S.epoch == "pre"].groupby(["set", "animal", "pos"])[["raw_ci", "bal_angle"]].mean()
    S = S.join(ref.add_prefix("pre_"), on=["set", "animal", "pos"])
    S["d_raw_ci"] = S.raw_ci - S.pre_raw_ci
    S["d_bal_angle"] = S.bal_angle - S.pre_bal_angle
    return S


def grouped(S: pd.DataFrame) -> pd.DataFrame:
    """Session values per group: each spout, 'all' (mean over the session's spouts), 'other' (all but far_contra)."""
    k = ["set", "animal", "epoch", "label"]
    parts = [S[k + ["pos"] + list(VARS)].rename(columns={"pos": "grp"})]
    parts.append(S.groupby(k)[list(VARS)].mean().reset_index().assign(grp="all"))
    parts.append(S[S.pos != "far_contra"].groupby(k)[list(VARS)].mean().reset_index().assign(grp="other"))
    return pd.concat(parts, ignore_index=True)


def tests(G: pd.DataFrame):
    from wfield_local import analysis_kit as ak
    arows, trows = [], []
    for (st, grp), g in G.groupby(["set", "grp"]):
        for v in VARS:
            by = {(a, ep): h[v].dropna().to_numpy() for (a, ep), h in g.groupby(["animal", "epoch"])}
            pre = {a: by[(a, "pre")] for a in AN if len(by.get((a, "pre"), [])) > 0}
            for ep in EPS:
                for a in AN:
                    x = by.get((a, ep), np.array([]))
                    if len(x) and a in pre:
                        arows.append({"set": st, "grp": grp, "var": v, "epoch": ep, "animal": a, "n_sess": len(x),
                                      "level": float(x.mean()), "delta": float(x.mean() - pre[a].mean())})
            for ep in POST:
                post = {a: by[(a, ep)] for a in AN if len(by.get((a, ep), [])) > 0}
                ci = ak.boot_delta(post, pre, np.random.default_rng(0))
                d = {a: float(post[a].mean() - pre[a].mean()) for a in AN if a in post and a in pre}
                r = {"set": st, "grp": grp, "var": v, "epoch": ep}
                if ci is None:
                    r.update(est=np.nan, lo=np.nan, hi=np.nan, n_animals=len(d), n_same_sign=np.nan, star="")
                else:
                    sg = np.sign(ci.point)
                    r.update(est=ci.point, lo=ci.lo, hi=ci.hi, n_animals=ci.n_animals,
                             n_same_sign=int(sum(np.sign(x) == sg for x in d.values())),
                             star="*" if ci.excludes_zero else "")
                r.update({a: d.get(a, np.nan) for a in AN})
                trows.append(r)
    A = pd.DataFrame(arows)
    T = pd.DataFrame(trows)
    order = {g: i for i, g in enumerate(GROUPS)}
    T = T.sort_values(["set", "var", "grp", "epoch"], key=lambda s: s.map(order) if s.name == "grp" else (
        s.map({e: i for i, e in enumerate(EPS)}) if s.name == "epoch" else s)).reset_index(drop=True)
    return A, T


def report(A: pd.DataFrame, T: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    for st in SETS:
        for v in VARS:
            print(f"\n===== {SETS[st]}  {v}  ({FRAME[v]}): PER-ANIMAL change vs pre (difference of session means)")
            h = A[(A.set == st) & (A["var"] == v) & (A.epoch != "pre")]
            t = h.pivot_table(index=["grp", "animal"], columns="epoch", values="delta").reindex(columns=POST)
            t = t.reindex([(g, a) for g in GROUPS for a in AN if (g, a) in t.index])
            print(t.round(2).to_string())
            print("----- pooled (boot_delta, mean of animals, 95% CI, [n same sign / n])")
            tt = T[(T.set == st) & (T["var"] == v)]
            for g in GROUPS:
                s = []
                for ep in POST:
                    r = tt[(tt.grp == g) & (tt.epoch == ep)]
                    if not len(r) or not np.isfinite(r.est.iloc[0]):
                        s.append(f"{ep[:3]}  n/a")
                        continue
                    r = r.iloc[0]
                    s.append(f"{ep[:3]} {r.est:+.2f} [{r.lo:+.2f},{r.hi:+.2f}]{r.star:1s} "
                             f"{int(r.n_same_sign)}/{int(r.n_animals)}")
                print(f"{g:13s} " + "   ".join(s))


def _style(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.axhline(0, color="0.7", lw=0.6, zorder=0)


def plot() -> None:
    import matplotlib.pyplot as plt
    A = pd.read_csv(OUT / "ci_spout_epoch_animals.csv")
    T = pd.read_csv(OUT / "ci_spout_epoch_tests.csv")
    (OUT / "figures").mkdir(exist_ok=True)
    x = np.arange(len(POS))
    off = {"acute": -0.22, "subacute": 0.0, "chronic": 0.22}
    fig, axs = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for j, st in enumerate(SETS):
        for i, v in enumerate(VARS):
            ax = axs[i, j]
            for ep in POST:
                t = T[(T.set == st) & (T["var"] == v) & (T.epoch == ep)].set_index("grp").reindex(POS)
                ax.errorbar(x + off[ep], t.est, yerr=[t.est - t.lo, t.hi - t.est], fmt="o-", color=COL[ep], ms=6,
                            lw=1.4, capsize=3, label=f"{ep}: pooled mean, 95% CI", zorder=3)
                h = A[(A.set == st) & (A["var"] == v) & (A.epoch == ep)]
                for a in AN:
                    ha = h[h.animal == a].set_index("grp").reindex(POS)
                    ax.plot(x + off[ep] + 0.07, ha.delta, MK[a], color=COL[ep], ms=3.5, alpha=0.6, mec="none",
                            zorder=2)
            _style(ax)
            ax.set_ylabel(YLAB[v], fontsize=8.5)
            ax.set_title(f"{SETS[st]} first lick: {v} ({FRAME[v]})", fontsize=9.5)
            ax.set_xticks(x, POS, rotation=25, fontsize=8.5)
    hs = [plt.Line2D([], [], color=COL[e], marker="o", label=f"{e}: pooled mean, 95% CI") for e in POST]
    hs += [plt.Line2D([], [], color="0.4", marker=MK[a], ls="none", ms=4, label=a) for a in AN]
    fig.legend(handles=hs, fontsize=8, ncol=7, loc="lower center", frameon=False)
    title = ("Post-lick primary-pool hemispheric balance (ALM, tjM1, SSp-m, SSp-n, SSs) per spout, change vs pre per "
             "epoch. HEMISPHERE sign: + = contralesional hemisphere relatively larger (C/I up). Top: raw C/I relative "
             "to the animal's pre value at the same spout (target frame). Middle: balance relative to the pre angle "
             "model of c and i (landed-angle frame; pre leave-one-session-out). Bottom: landing error. Pooled = paired "
             "nested bootstrap (animals -> sessions); small markers = animals; spouts with >= 5 licks per session. "
             "ITI: spout = previous trial's.")
    fig.suptitle(textwrap.fill(title, 150), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save_png_svg(fig, OUT / "figures" / "lick_ci_spout_epoch.png", dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(3, 2, figsize=(11, 10.5), sharex=True)
    gc = {"far_contra": "#C44E52", "other": "0.35"}
    xe = np.arange(len(EPS))
    for j, st in enumerate(SETS):
        for i, v in enumerate(VARS):
            ax = axs[i, j]
            for k, g in enumerate(("far_contra", "other")):
                dx = (k - 0.5) * 0.12
                h = A[(A.set == st) & (A["var"] == v) & (A.grp == g)]
                for a in AN:
                    ha = h[h.animal == a].set_index("epoch").reindex(EPS)
                    ax.plot(xe + dx, ha.delta, "-", marker=MK[a], color=gc[g], lw=0.7, ms=3.5, alpha=0.45)
                t = T[(T.set == st) & (T["var"] == v) & (T.grp == g)].set_index("epoch").reindex(POST)
                est = np.r_[0.0, t.est.to_numpy()]
                lo_ = np.r_[0.0, (t.est - t.lo).to_numpy()]
                hi_ = np.r_[0.0, (t.hi - t.est).to_numpy()]
                ax.errorbar(xe + dx, est, yerr=[lo_, hi_], fmt="o-", color=gc[g], lw=2.2, ms=6, capsize=3,
                            label=("far_contra" if g == "far_contra" else "other 5 spouts (mean)")
                            + ": pooled, 95% CI", zorder=3)
            _style(ax)
            ax.set_ylabel(YLAB[v], fontsize=8.5)
            ax.set_title(f"{SETS[st]} first lick: {v} ({FRAME[v]})", fontsize=9.5)
            ax.set_xticks(xe, EPS)
    hs = [plt.Line2D([], [], color=gc[g], marker="o", lw=2, label=lb) for g, lb in
          (("far_contra", "far_contra: pooled, 95% CI"), ("other", "other 5 spouts (mean): pooled, 95% CI"))]
    hs += [plt.Line2D([], [], color="0.4", marker=MK[a], lw=0.7, ms=4, label=f"{a} (thin)") for a in AN]
    fig.legend(handles=hs, fontsize=8, ncol=6, loc="lower center", frameon=False)
    title = ("far_contra vs the other spouts across epochs: change vs pre of the post-lick primary-pool balance. "
             "HEMISPHERE sign: + = contralesional hemisphere relatively larger (C/I up). Pre = 0 by construction "
             "(each animal's pre mean at the same spout). Thin lines = animals; thick = pooled paired nested bootstrap.")
    fig.suptitle(textwrap.fill(title, 120), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save_png_svg(fig, OUT / "figures" / "lick_ci_spout_epoch_farcontra.png", dpi=100)
    plt.close(fig)


def trial_set_sentence() -> str:
    """The lick set of lick_ci_spout_epoch (Priya 11-09): baltarget_licks (lick_balance_target_model), session x spout
    cells with >= MIN_LICKS licks."""
    import lick_balance_target_model as tm
    keep = [f"session x spout cells with >= {MIN_LICKS} licks"]
    return tm.lick_set_sentence(cued_extra=keep, iti_extra=keep)


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    if "--plot-only" not in sys.argv:
        L = pd.read_csv(OUT / "baltarget_licks.csv.gz")
        S = sessions(L)
        S["pos"] = pd.Categorical(S.pos, POS, ordered=True)
        S = S.sort_values(["set", "animal", "day", "label", "pos"]).reset_index(drop=True)
        S.to_csv(OUT / "ci_spout_epoch_sessions.csv", index=False)
        S["pos"] = S.pos.astype(str)
        A, T = tests(grouped(S))
        A.to_csv(OUT / "ci_spout_epoch_animals.csv", index=False)
        T.to_csv(OUT / "ci_spout_epoch_tests.csv", index=False)
        report(A, T)
    plot()
    return 0


if __name__ == "__main__":
    sys.exit(main())

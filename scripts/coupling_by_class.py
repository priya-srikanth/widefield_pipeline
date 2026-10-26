"""TRIAL-BY-TRIAL COUPLING SPLIT BY TRIAL CLASS: hits (lick reaches and touches the spout) vs failed attempts (incomplete
licks, no contact). Priya 10-25: the acute loss of coupling in `trial_neural_vs_exec.py` was mostly a COMPOSITION effect --
acutely 92 % of far-contra trials are failed attempts and they carry no coupling, while the few acute hits look like
pre-stroke hits; the pre-stroke reference is hits only.

Per far-contra target trial (lick window, August pre): neural = frozen decoder P(far mid) - P(far contra); executed = the
trial's lick weight far mid - far contra (landing: 3-D endpoints; trajectory: lick paths; `softmax_neural_vs_lick`). Both
centred within SESSION x CLASS, then r is pooled over the trials of an animal x epoch x class (or all animals).

STATISTIC = class-matched change across epochs. Acute hits are ~4 per session, so a per-session r does not exist; the test
resamples the pooled r: nested bootstrap (animals, then sessions within animal, 2000 draws), delta = r[epoch, class] -
r[pre, hits] with the SAME animal draw in both. Hits vs pre hits is class-matched; failed attempts have no pre-stroke
counterpart (pre trials are hits only), so their delta is against pre hits and labelled as such. Per animal: sessions
resampled within the animal. Session dots (figures 2-4) need >= MIN_SESS trials of that class in the session.

Figures (<E>/attempt_coding_pp/figures/): coupling_cls_animals.png (page 4i), coupling_cls_sessions.png (4i2),
coupling_cls_pooled.png (4i3), coupling_cls_pooled_norm.png (4i4).
Usage: python scripts/coupling_by_class.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from softmax_neural_vs_lick import lick_weights, path_weights  # noqa: E402

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
D = E / "attempt_coding_pp"
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}
CLS = [("hit", "hits (contact)", "#2c3e50"), ("failed", "failed attempts (no contact)", "#c0392b")]
MIN_SESS = 10
N_BOOT = 2000


def load() -> dict:
    T = pd.concat([pd.read_csv(D / f"trials_lick_{an}.csv.gz") for an in AN])
    T = T[(T.method == "dom") & (T.spout == "far_contra") & T.cls.isin(["pre_hit", "hit", "incomplete"])
          & (T.label.str[-4:] >= "0801")]
    T["neural_d"] = T.pp_far_mid - T.pp_far_contra
    T["cls2"] = np.where(T.cls == "incomplete", "failed", "hit")
    out = {}
    for kind, W in (("landing", lick_weights(1.0)), ("trajectory", path_weights(1.0))):
        pre = "lw_" if kind == "landing" else "tw_"
        J = T.merge(W, on=["label", "trial_id"], how="inner")
        J["exec_d"] = J[f"{pre}far_mid"] - J[f"{pre}far_contra"]
        J = J[np.isfinite(J.exec_d) & np.isfinite(J.neural_d)].copy()
        for c in ("exec_d", "neural_d"):
            J[c + "_c"] = J[c] - J.groupby(["label", "cls2"])[c].transform("mean")
        out[kind] = J.reset_index(drop=True)
    return out


def r_of(x, y) -> float:
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _index(J):
    """(animal, epoch, class) -> {session label: row indices}."""
    ix = {}
    for (an, ep, c, lab), g in J.groupby(["animal", "epoch", "cls2", "label"]):
        ix.setdefault((an, ep, c), {})[lab] = g.index.to_numpy()
    return ix


def boot_delta_r(J, ix, ep, cl, animals, rng, nested=True):
    """Pooled r[ep, cl] - r[pre, hit] (point) and 95 % interval, resampling animals then sessions (or sessions only when
    `animals` is one animal)."""
    x, y = J.exec_d_c.to_numpy(), J.neural_d_c.to_numpy()

    def gather(an_list, key_ep, key_cl, draw):
        rows = []
        for an in an_list:
            ss = ix.get((an, key_ep, key_cl))
            if not ss:
                continue
            labs = list(ss)
            pick = rng.choice(len(labs), len(labs)) if draw else range(len(labs))
            rows.extend(ss[labs[i]] for i in pick)
        return np.concatenate(rows) if rows else np.array([], int)

    com = [an for an in animals if ix.get((an, ep, cl)) and ix.get((an, "pre", "hit"))]
    if not com:
        return None
    a0, b0 = gather(com, ep, cl, False), gather(com, "pre", "hit", False)
    point = r_of(x[a0], y[a0]) - r_of(x[b0], y[b0])
    d = []
    for _ in range(N_BOOT):
        an_d = [com[i] for i in rng.choice(len(com), len(com))] if (nested and len(com) > 1) else com
        a, b = gather(an_d, ep, cl, True), gather(an_d, "pre", "hit", True)
        d.append(r_of(x[a], y[a]) - r_of(x[b], y[b]))
    d = np.array(d)
    d = d[np.isfinite(d)]
    if len(d) < 50 or not np.isfinite(point):           # too few trials for an r in most draws (e.g. PS95 acute: 1 hit)
        return None
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"point": point, "lo": lo, "hi": hi, "p": float(2 * min((d <= 0).mean(), (d >= 0).mean())), "animals": "/".join(com),
            "n": int(len(a0))}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import analysis_kit as ak
    from wfield_local.sig_marks import sig_star
    data = load()
    pd.set_option("display.width", 220)
    lev, tst, ses = [], [], []
    for kind, J in data.items():
        ix = _index(J)
        x, y = J.exec_d_c.to_numpy(), J.neural_d_c.to_numpy()
        for (an, ep, c), ss in ix.items():                                   # levels per animal
            rows = np.concatenate(list(ss.values()))
            lev.append({"exec": kind, "animal": an, "epoch": ep, "class": c, "r": r_of(x[rows], y[rows]), "n": len(rows),
                        "n_sessions": len(ss)})
        for ep in EP:                                                         # pooled level, all animals
            for c, _l, _col in CLS:
                rows = [v for an in AN for v in ix.get((an, ep, c), {}).values()]
                if rows:
                    rows = np.concatenate(rows)
                    lev.append({"exec": kind, "animal": "ALL", "epoch": ep, "class": c, "r": r_of(x[rows], y[rows]),
                                "n": len(rows), "n_sessions": sum(len(ix.get((an, ep, c), {})) for an in AN)})
        rng = np.random.default_rng(0)
        for ep in EP:
            for c, _l, _col in CLS:
                if ep == "pre" and c == "hit":
                    continue
                t = boot_delta_r(J, ix, ep, c, AN, rng)
                if t:
                    tst.append({"exec": kind, "animal": "ALL", "epoch": ep, "class": c, **t})
                for an in AN:
                    t = boot_delta_r(J, ix, ep, c, [an], rng, nested=False)
                    if t and t["n"] >= 10:
                        tst.append({"exec": kind, "animal": an, "epoch": ep, "class": c, **t})
        for (an, ep, c, lab), g in J.groupby(["animal", "epoch", "cls2", "label"]):   # per session, n >= MIN_SESS
            if len(g) >= MIN_SESS:
                ses.append({"exec": kind, "animal": an, "epoch": ep, "class": c, "label": lab, "day": int(g.day.iloc[0]),
                            "r": r_of(g.exec_d_c.to_numpy(), g.neural_d_c.to_numpy()), "n": len(g)})
    L, Tt, S = pd.DataFrame(lev), pd.DataFrame(tst), pd.DataFrame(ses)
    L.to_csv(D / "coupling_cls_levels.csv", index=False)
    Tt.to_csv(D / "coupling_cls_tests.csv", index=False)
    S.to_csv(D / "coupling_cls_sessions.csv", index=False)
    print("== per animal (rule 8): pooled within-session r, by class (n trials)")
    print(L.pivot_table(index=["exec", "animal"], columns=["epoch", "class"], values="r")
          .reindex(columns=EP, level=0).round(2).to_string())
    print(L.pivot_table(index=["exec", "animal"], columns=["epoch", "class"], values="n")
          .reindex(columns=EP, level=0).to_string())
    print("== change vs PRE HITS (nested bootstrap animals > sessions for ALL; sessions for one animal)")
    print(Tt[["exec", "animal", "epoch", "class", "point", "lo", "hi", "p", "n"]].round(3).to_string(index=False))

    out = D / "figures"
    xo = {"hit": -0.18, "failed": 0.18}

    def legend(fig, extra=()):
        h = [Line2D([], [], marker="o", ls="", mfc=col, mec=col, label=lab) for _c, lab, col in CLS]
        h += [Line2D([], [], marker=MK[a], ls="", mfc="w", mec="k", label=a) for a in AN] + list(extra)
        fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=8, frameon=False)

    def delta_text(ax, kind, ep, c, y, animal="ALL"):
        r = Tt[(Tt.exec == kind) & (Tt.animal == animal) & (Tt.epoch == ep) & (Tt["class"] == c)]
        if len(r):
            r = r.iloc[0]
            sig = r.lo > 0 or r.hi < 0
            ax.text(ep_x[ep] + xo[c], y, f"{r.point:+.2f}\n[{r.lo:+.2f},\n{r.hi:+.2f}]", ha="center", va="top", fontsize=6.5,
                    color=dict((k, col) for k, _l, col in CLS)[c], fontweight="bold" if sig else "normal")

    ep_x = {e: i for i, e in enumerate(EP)}

    # FIG A (4i): per animal x epoch x class; bar = pooled r of all animals; numbers = change vs pre hits
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True)
    for k, kind in enumerate(data):
        ax = axs[k]
        ax.axhline(0, color="k", lw=0.6)
        for ep in EP:
            for c, _lab, col in CLS:
                a = L[(L.exec == kind) & (L.epoch == ep) & (L["class"] == c)]
                if not len(a):
                    continue
                xx = ep_x[ep] + xo[c]
                al = a[a.animal == "ALL"]
                if len(al):
                    ax.bar(xx, al.r.iloc[0], width=0.32, color=col, alpha=0.25)
                for j, an in enumerate(AN):
                    v = a[a.animal == an]
                    if len(v) and v.n.iloc[0] >= 10:
                        ax.scatter(xx + (j - 1.5) * 0.05, v.r.iloc[0], marker=MK[an], s=30, facecolor=col if c == "hit" else "w",
                                   edgecolor=col, lw=1.2, zorder=5)
                delta_text(ax, kind, ep, c, 0.98)
        ax.set_xticks(range(4)); ax.set_xticklabels(EP); ax.set_ylim(-0.45, 1.0)
        ax.set_title(f"executed = {kind}", fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if k == 0:
            ax.set_ylabel("within-session r: neural (fM - fC)\nvs executed (fM - fC)")
    legend(fig, [plt.Rectangle((0, 0), 1, 1, color="0.6", alpha=0.3, label="all animals pooled")])
    fig.suptitle("Trial-by-trial coupling on far-contra trials BY TRIAL CLASS (dots = animals with >= 10 trials; bar = all animals "
                 "pooled). Numbers = change vs PRE-STROKE HITS,\n95% nested CI (animals > sessions; bold = excludes 0). Pre-stroke "
                 "trials are hits only: failed attempts have no class-matched pre reference", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(out / "coupling_cls_animals.png", dpi=125)
    plt.close(fig)

    # FIG B (4i2): per session vs day, by class
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for k, kind in enumerate(data):
        ax = axs[k]
        ax.axvspan(S.day.min() - 2, 0, color="0.93", lw=0, zorder=0)
        ax.axhline(0, color="k", lw=0.6)
        for c, _lab, col in CLS:
            for an in AN:
                v = S[(S.exec == kind) & (S["class"] == c) & (S.animal == an)].sort_values("day")
                if len(v):
                    ax.plot(v.day, v.r, color=col, lw=0.6, alpha=0.35)
                    ax.scatter(v.day, v.r, marker=MK[an], s=28, facecolor=col if c == "hit" else "w", edgecolor=col, lw=1.1,
                               zorder=5)
        ax.set_xlabel("days from stroke (grey = pre-stroke)")
        ax.set_title(f"executed = {kind}", fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if k == 0:
            ax.set_ylabel("within-session r (sessions with\n>= 10 trials of that class)")
    legend(fig)
    fig.suptitle(f"Trial-by-trial coupling PER SESSION, by trial class (sessions with >= {MIN_SESS} trials of the class; acute "
                 "sessions have ~4 hits each, so acute hits appear only pooled, Fig A / C)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(out / "coupling_cls_sessions.png", dpi=125)
    plt.close(fig)

    # FIG C (4i3) and D (4i4): pooled per epoch, one dot per session; D normalised to the animal's pre-hit r
    pre_hit = L[(L.epoch == "pre") & (L["class"] == "hit")].set_index(["exec", "animal"]).r
    S["rn"] = S.r / [pre_hit.get((e, a), np.nan) for e, a in zip(S.exec, S.animal)]
    for fname, vcol, ylab, ylim, ref, ttl in (
            ("coupling_cls_pooled.png", "r", "within-session r", (-0.6, 1.25), None, ""),
            ("coupling_cls_pooled_norm.png", "rn", "r / the animal's pre-stroke hit r\n(1 = pre-stroke coupling)", (-3.2, 5.0),
             1.0, ", NORMALISED within animal to its pre-stroke hits")):
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True)
        prow = []
        for k, kind in enumerate(data):
            ax = axs[k]
            ax.axhline(0, color="k", lw=0.6)
            if ref is not None:
                ax.axhline(ref, color="0.5", lw=0.8, ls="--")
            for ep in EP:
                for c, _lab, col in CLS:
                    v = S[(S.exec == kind) & (S.epoch == ep) & (S["class"] == c)]
                    xx = ep_x[ep] + xo[c]
                    if len(v):
                        for j, an in enumerate(AN):
                            w = v[v.animal == an]
                            if len(w):
                                xs = xx + (j - 1.5) * 0.04 + np.random.default_rng(j + 10 * ep_x[ep]).uniform(-0.015, 0.015, len(w))
                                ax.scatter(xs, w[vcol], marker=MK[an], s=20, facecolor=col if c == "hit" else "w", edgecolor=col,
                                           lw=1.0, zorder=4)
                        by = {an: v[v.animal == an][vcol].to_numpy() for an in AN if (v.animal == an).any()}
                        ax.plot([xx - 0.15, xx + 0.15], [np.mean([b.mean() for b in by.values()])] * 2, color=col, lw=2.2, zorder=5)
                        ref_by = {an: S[(S.exec == kind) & (S.epoch == "pre") & (S["class"] == "hit") & (S.animal == an)][vcol]
                                  .to_numpy() for an in by}
                        ref_by = {an: b for an, b in ref_by.items() if len(b)}
                        com = [an for an in by if an in ref_by]
                        if ep != "pre" and len(com) >= 2:
                            iv = ak.boot_delta({an: by[an] for an in com}, {an: ref_by[an] for an in com}, np.random.default_rng(0))
                            if iv is not None:
                                prow.append({"exec": kind, "epoch": ep, "class": c, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                                             "animals": "/".join(com)})
                                sig = iv.lo > 0 or iv.hi < 0
                                ax.text(xx, ylim[1] - 0.02 * (ylim[1] - ylim[0]), f"{iv.point:+.2f}\n[{iv.lo:+.2f},\n{iv.hi:+.2f}]",
                                        ha="center", va="top", fontsize=6.5, color=col, fontweight="bold" if sig else "normal")
            ax.set_xticks(range(4)); ax.set_xticklabels(EP); ax.set_ylim(*ylim)
            ax.set_title(f"executed = {kind}", fontsize=10)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if k == 0:
                ax.set_ylabel(ylab)
        Pd = pd.DataFrame(prow)
        Pd.to_csv(D / fname.replace(".png", ".csv"), index=False)
        print(f"== {fname}: session-level epoch - pre hits (boot_delta, animals > sessions)\n" + Pd.round(3).to_string(index=False))
        legend(fig, [Line2D([], [], color="0.3", lw=2.2, label="mean of animals")])
        fig.suptitle(f"Coupling pooled per epoch, one dot per SESSION (>= {MIN_SESS} trials of the class){ttl}. Numbers = change vs "
                     "pre-stroke hits,\nsession values, 95% nested CI (animals > sessions; bold = excludes 0). Acute hits: too few per "
                     "session to show (see Fig A for their pooled r)", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.07, 1, 0.9))
        fig.savefig(out / fname, dpi=125)
        plt.close(fig)
    _ = sig_star
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

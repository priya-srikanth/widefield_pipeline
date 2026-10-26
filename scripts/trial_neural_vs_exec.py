"""TRIAL-BY-TRIAL: do far-contra licks that look like far mid come with cortex that looks like far mid? (Priya 10-24:
"are the licks that look like far mid instead of far contra corresponding to cortical activity that looks more like far
mid than far contra?")

Per far-contra TARGET trial (lick window): NEURAL = the frozen pre-stroke spout decoder's probabilities (attempt_coding
pp_<spout>, LOSO for pre); EXECUTED = the trial's lick softmax weights (landing, and trajectory) over the six spouts,
width 1 (`softmax_neural_vs_lick.lick_weights / path_weights`).

  1. split by the trial's dominant executed bin (far contra / far mid / far ipsi): mean neural P(far contra),
     P(far mid), P(far ipsi) per animal x epoch (n >= 8 trials), pre included as the control (pre far-contra trials
     that happened to land like far mid).
  2. within-session coupling: neural (P far mid - P far contra) regressed on executed (w far mid - w far contra), both
     centred within session; slope and r per animal x epoch; null = executed values block-permuted within session (runs of consecutive far-contra trials, 1000x),
     p = share of shuffled |slopes| >= observed.

Usage: python scripts/trial_neural_vs_exec.py
Output: <E>/attempt_coding_pp/trial_neural_vs_exec_*.csv + figures/trial_neural_vs_exec*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from softmax_neural_vs_lick import SP, lick_weights, path_weights  # noqa: E402

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
FAR = ["far_contra", "far_mid", "far_ipsi"]
COL = {"far_contra": "#d94801", "far_mid": "#252525", "far_ipsi": "#08519c"}
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}


def coupling(g: pd.DataFrame, x: str, y: str, rng, n_perm=1000):
    xs = g[x] - g.groupby("label")[x].transform("mean")
    ys = g[y] - g.groupby("label")[y].transform("mean")
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys, labs = xs[ok].to_numpy(), ys[ok].to_numpy(), g.label[ok].to_numpy()
    tid = g.trial_id[ok].to_numpy()
    if len(xs) < 20 or xs.std() == 0:
        return None
    b = np.polyfit(xs, ys, 1)[0]
    r = np.corrcoef(xs, ys)[0, 1]
    # BLOCK-PERMUTATION null (Priya 10-25): far-contra trials come in runs of ~6 consecutive trials and both quantities
    # drift within a session, so a trial-level shuffle is too narrow. Per session, the executed values of whole RUNS
    # (consecutive far-contra trials; a trial_id jump > 2 starts a new run) are permuted in run order, each run's
    # internal order kept, and laid back onto the session's trial positions.
    sess = []
    for lab in np.unique(labs):
        ix = np.flatnonzero(labs == lab)
        ix = ix[np.argsort(tid[ix], kind="stable")]
        cut = np.flatnonzero(np.diff(tid[ix]) > 2) + 1
        sess.append((ix, np.split(ix, cut)))
    null = np.empty(n_perm)
    for k in range(n_perm):
        xp = xs.copy()
        for ix, runs in sess:
            xp[ix] = xs[np.concatenate([runs[j] for j in rng.permutation(len(runs))])]
        null[k] = np.polyfit(xp, ys, 1)[0]
    return b, r, float((np.abs(null) >= abs(b)).mean()), len(xs)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = E / "attempt_coding_pp"
    T = pd.concat([pd.read_csv(D / f"trials_lick_{an}.csv.gz") for an in AN])
    T = T[(T.method == "dom") & (T.spout == "far_contra") & T.cls.isin(["pre_hit", "hit", "incomplete"])]
    T = T[T.label.str[-4:] >= "0801"]          # AUGUST-ONLY pre baseline (primary, Priya 10-18); June was in landing only
    T["neural_d"] = T.pp_far_mid - T.pp_far_contra
    rows, coup, sess_rows = [], [], []
    rng = np.random.default_rng(0)
    rng_s = np.random.default_rng(1)          # per-session nulls draw from their own stream: per-animal draws unchanged
    for kind, W in (("landing", lick_weights(1.0)), ("trajectory", path_weights(1.0))):
        pre = "lw_" if kind == "landing" else "tw_"
        J = T.merge(W, on=["label", "trial_id"], how="inner")
        J["exec_d"] = J[f"{pre}far_mid"] - J[f"{pre}far_contra"]
        J["dom"] = np.array(SP)[J[[f"{pre}{p}" for p in SP]].to_numpy().argmax(1)]
        for (an, ep), g in J.groupby(["animal", "epoch"]):
            for dm in FAR:
                h = g[g.dom == dm]
                if len(h) >= 8:
                    rows.append({"exec": kind, "animal": an, "epoch": ep, "landed": dm, "n": len(h),
                                 **{f"neural_{p}": float(h[f"pp_{p}"].mean()) for p in FAR}})
            c = coupling(g, "exec_d", "neural_d", rng)
            if c:
                coup.append({"exec": kind, "animal": an, "epoch": ep, "slope": c[0], "r": c[1], "p_perm": c[2], "n": c[3]})
        for (an, ep, lab), g in J.groupby(["animal", "epoch", "label"]):      # PER SESSION (Priya 10-25)
            c = coupling(g, "exec_d", "neural_d", rng_s)
            if c:
                sess_rows.append({"exec": kind, "animal": an, "epoch": ep, "label": lab, "day": int(g.day.iloc[0]),
                                  "slope": c[0], "r": c[1], "p_perm": c[2], "n": c[3]})
    B = pd.DataFrame(rows); C = pd.DataFrame(coup)
    B.to_csv(D / "trial_neural_vs_exec_by_landing.csv", index=False)
    C.to_csv(D / "trial_neural_vs_exec_coupling.csv", index=False)
    Sx = pd.DataFrame(sess_rows).sort_values(["exec", "animal", "day"])
    Sx.to_csv(D / "trial_neural_vs_exec_coupling_sessions.csv", index=False)
    pd.set_option("display.width", 200)
    print(B[B.exec == "landing"].round(2).to_string(index=False))
    print(C.round(3).to_string(index=False))
    print("== per session (landing): r, p, n")
    print(Sx[Sx.exec == "landing"][["animal", "epoch", "label", "day", "r", "p_perm", "n"]].round(3).to_string(index=False))
    print("== sessions significant / tested, per animal x epoch (landing)")
    print(Sx[Sx.exec == "landing"].assign(sig=lambda d: d.p_perm < 0.05).groupby(["animal", "epoch"]).sig
          .agg(lambda x: f"{int(x.sum())}/{len(x)}").unstack("epoch").reindex(columns=EP).to_string())

    # figure 1: neural probabilities by where the licks landed (landing), animals x epochs
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6), sharey=True)
    for r, kind in enumerate(("landing", "trajectory")):
        for k, ep in enumerate(EP):
            ax = axs[r, k]
            b = B[(B.exec == kind) & (B.epoch == ep)]
            for i, dm in enumerate(FAR):
                for j, p in enumerate(FAR):
                    v = b[b.landed == dm][f"neural_{p}"]
                    if not len(v):
                        continue
                    x = i * 4 + j
                    ax.bar(x, v.mean(), color=COL[p], width=0.85, alpha=0.85,
                           label=f"decoded P({p})" if (i == 0 and k == 0 and r == 0) else None)
                    for an in b[b.landed == dm].animal:
                        ax.scatter(x, b[(b.landed == dm) & (b.animal == an)][f"neural_{p}"], marker=MK[an], s=16,
                                   facecolor="w", edgecolor="k", lw=0.7, zorder=5)
                nn = b[b.landed == dm].n.sum()
                ax.text(i * 4 + 1, 0.99, f"n={nn} ({len(b[b.landed == dm])} an.)", ha="center", va="top", fontsize=7)
            ax.set_xticks([1, 5, 9]); ax.set_xticklabels([f"landed\n{d.replace('_', ' ')}" for d in FAR], fontsize=8)
            ax.set_ylim(0, 1); ax.set_title(f"{ep} ({kind})", fontsize=10)
            if k == 0:
                ax.set_ylabel("frozen decoder probability\n(far-contra TARGET trials)")
    from matplotlib.lines import Line2D
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[p]) for p in FAR] + [Line2D([], [], marker=MK[a], ls="", mfc="w", mec="k") for a in AN]
    fig.legend(h, [f"decoded P({p.replace('_', ' ')})" for p in FAR] + AN, loc="lower center", ncol=7, fontsize=8.5, frameon=False)
    fig.suptitle("NEURAL vs EXECUTED, trial by trial: far-contra target trials split by where their licks went (top: landing bin; bottom: "
                 "trajectory bin),\nbars = mean over animals of the frozen pre-stroke decoder's probability for far contra / far mid / far ipsi "
                 "(groups with >= 8 trials); pre = pre-stroke control", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(D / "figures" / "trial_neural_vs_exec_by_landing.png", dpi=125)
    plt.close(fig)

    # figure 2: within-session coupling slope per animal x epoch
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for r, kind in enumerate(("landing", "trajectory")):
        ax = axs[r]
        c = C[C.exec == kind]
        for k, ep in enumerate(EP):
            cc = c[c.epoch == ep]
            for _, row in cc.iterrows():
                x = k + {"PS92": -0.15, "PS93": -0.05, "PS94": 0.05, "PS95": 0.15}[row.animal]
                ax.scatter(x, row.r, marker=MK[row.animal], s=40, facecolor="k" if row.p_perm < 0.05 else "w", edgecolor="k")
            if len(cc):
                ax.plot([k - 0.25, k + 0.25], [cc.r.mean()] * 2, color="k", lw=2)
        ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(EP)
        ax.set_title(f"executed = {kind}", fontsize=10)
        if r == 0:
            ax.set_ylabel("within-session r: neural (fM - fC)\nvs executed (fM - fC)")
    fig.legend([Line2D([], [], marker=MK[a], ls="", mfc="w", mec="k") for a in AN] +
               [Line2D([], [], marker="o", ls="", mfc="k", mec="k")], AN + ["filled = p < 0.05 (within-session block permutation)"],
               loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("Trial-by-trial coupling on far-contra target trials: the more a trial's licks look like far mid rather than far contra, "
                 "\nthe more its cortex is decoded as far mid rather than far contra? (both centred within session; line = mean of animals)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    fig.savefig(D / "figures" / "trial_neural_vs_exec_coupling.png", dpi=125)
    plt.close(fig)

    # figure 3: PER SESSION, r against days from stroke (Priya 10-25: "can we do this per session instead of per animal")
    ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for r, kind in enumerate(("landing", "trajectory")):
        ax = axs[r]
        ax.axvspan(Sx.day.min() - 2, 0, color="0.92", lw=0, zorder=0)
        ax.axhline(0, color="k", lw=0.6)
        for an in AN:
            c = Sx[(Sx.exec == kind) & (Sx.animal == an)].sort_values("day")
            if not len(c):
                continue
            ax.plot(c.day, c.r, color=ACOL[an], lw=0.8, alpha=0.5)
            sig = c.p_perm < 0.05
            ax.scatter(c.day[sig], c.r[sig], marker=MK[an], s=34, color=ACOL[an], edgecolor="k", lw=0.5, zorder=5)
            ax.scatter(c.day[~sig], c.r[~sig], marker=MK[an], s=30, facecolor="w", edgecolor=ACOL[an], lw=1.3, zorder=5)
        ax.set_xlabel("days from stroke (grey = pre-stroke)")
        ax.set_title(f"executed = {kind}", fontsize=10)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if r == 0:
            ax.set_ylabel("within-session r: neural (fM - fC)\nvs executed (fM - fC)")
    h = [Line2D([], [], marker=MK[a], color=ACOL[a], mfc=ACOL[a], mec="k", lw=0.8, label=a) for a in AN]
    h += [Line2D([], [], marker="o", ls="", mfc="0.4", mec="k", label="p < 0.05 (within-session block permutation)"),
          Line2D([], [], marker="o", ls="", mfc="w", mec="0.4", mew=1.3, label="n.s.")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8.5, frameon=False)
    fig.suptitle("Trial-by-trial coupling PER SESSION (far-contra target trials, sessions with >= 20 such trials): does cortex follow "
                 "where each lick went?\nr between the decoder's (P far mid - P far contra) and the lick's (far mid - far contra) "
                 "weight, centred within session", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    fig.savefig(D / "figures" / "trial_neural_vs_exec_coupling_sessions.png", dpi=125)
    plt.close(fig)

    # figure 4: POOLED PER EPOCH, one dot per session (Priya 10-25). Line = mean of animals' session means; epoch - pre
    # tested with the nested animals -> sessions bootstrap (analysis_kit.boot_delta, animal-weighted).
    from wfield_local import analysis_kit as ak
    # NORMALISED version (Priya 10-25): each session's r / its animal's mean pre-stroke r (per executed measure);
    # 1 = the animal's pre-stroke coupling, 0 = none
    pre_m = Sx[Sx.epoch == "pre"].groupby(["exec", "animal"]).r.mean()
    Sx["rn"] = Sx.r / [pre_m.get((e, an), np.nan) for e, an in zip(Sx.exec, Sx.animal)]
    for ver, vcol, ylab, ylim, ytxt, ref, ttl in (
            ("pooled", "r", "within-session r: neural (fM - fC)\nvs executed (fM - fC)", (-0.32, 0.98), 0.97, None, ""),
            ("pooled_norm", "rn", "r / the animal's mean pre-stroke r\n(1 = pre-stroke coupling)", (-2.3, 4.1), 4.05, 1.0,
             ",\nNORMALISED within animal to its pre-stroke level")):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        prow = []
        for r, kind in enumerate(("landing", "trajectory")):
            ax = axs[r]
            c = Sx[Sx.exec == kind]
            ax.axhline(0, color="k", lw=0.6)
            if ref is not None:
                ax.axhline(ref, color="0.5", lw=0.8, ls="--")
            by = {ep: {an: c[(c.animal == an) & (c.epoch == ep)][vcol].to_numpy() for an in AN
                       if ((c.animal == an) & (c.epoch == ep)).any()} for ep in EP}
            for k, ep in enumerate(EP):
                for j, an in enumerate(AN):
                    v = c[(c.animal == an) & (c.epoch == ep)]
                    if not len(v):
                        continue
                    xs = k + (j - 1.5) * 0.12 + np.random.default_rng(j * 10 + k).uniform(-0.04, 0.04, len(v))
                    sig = (v.p_perm < 0.05).to_numpy()
                    ax.scatter(xs[sig], v[vcol][sig], marker=MK[an], s=26, color=ACOL[an], edgecolor="k", lw=0.4, zorder=4)
                    ax.scatter(xs[~sig], v[vcol][~sig], marker=MK[an], s=22, facecolor="w", edgecolor=ACOL[an], lw=1.1, zorder=4)
                lev = np.mean([x.mean() for x in by[ep].values()]) if by[ep] else np.nan
                ax.plot([k - 0.3, k + 0.3], [lev, lev], color="k", lw=2.2, zorder=5)
                if ep != "pre" and by[ep]:
                    com = [an for an in by[ep] if an in by["pre"]]
                    iv = ak.boot_delta({an: by[ep][an] for an in com}, {an: by["pre"][an] for an in com},
                                       np.random.default_rng(0))
                    if iv is not None:
                        prow.append({"exec": kind, "epoch": ep, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                                     "animals": "/".join(com), "n_sessions": sum(len(by[ep][an]) for an in com)})
                        sig_ = iv.lo > 0 or iv.hi < 0
                        ax.text(k, ytxt, f"{iv.point:+.2f}\n[{iv.lo:+.2f}, {iv.hi:+.2f}]"
                                + (" *" if sig_ else ""), ha="center", va="top", fontsize=8,
                                fontweight="bold" if sig_ else "normal")
            ax.set_xticks(range(4)); ax.set_xticklabels(EP)
            ax.set_ylim(*ylim)
            ax.set_title(f"executed = {kind}", fontsize=10)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if r == 0:
                ax.set_ylabel(ylab)
        Pd = pd.DataFrame(prow)
        Pd.to_csv(D / f"trial_neural_vs_exec_coupling_{ver}.csv", index=False)
        print(f"== {ver} epoch - pre (sessions, nested bootstrap animals -> sessions)\n" + Pd.round(3).to_string(index=False))
        h = [Line2D([], [], marker=MK[a], ls="", mfc=ACOL[a], mec="k", label=a) for a in AN]
        h += [Line2D([], [], marker="o", ls="", mfc="0.4", mec="k", label="session p < 0.05 (block permutation)"),
              Line2D([], [], marker="o", ls="", mfc="w", mec="0.4", mew=1.1, label="session n.s."),
              Line2D([], [], color="k", lw=2.2, label="mean of animals")]
        fig.legend(handles=h, loc="lower center", ncol=7, fontsize=8.5, frameon=False)
        fig.suptitle(f"Trial-by-trial coupling pooled per epoch, one dot per session (far-contra target trials; August pre){ttl}. Numbers = "
                     "epoch minus pre,\nmean of animals with 95% nested CI (animals > sessions; * excludes 0)", fontsize=10)
        fig.tight_layout(rect=(0, 0.08, 1, 0.9))
        fig.savefig(D / "figures" / f"trial_neural_vs_exec_coupling_{ver}.png", dpi=125)
        plt.close(fig)
    print(f"-> {D / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

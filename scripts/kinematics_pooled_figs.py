"""POOLED, within-animal pre-normalised versions of the per-animal kinematics figures (Priya 10-25: "a pre-stroke-
normalized within-animal normalized cross-animal session pooled version of fig 2 (and anything else that needs a pooled
fig)"). Every value is a SESSION minus ITS ANIMAL's August pre mean; the bold estimate is the mean over animals of each
animal's mean change (animals weighted equally) and the interval is `analysis_kit.boot_delta` (paired animals ->
sessions bootstrap, 95 %). Dots = sessions, coloured by animal. A cell needs >= 3 animals to be drawn with an interval.

  angle_change_pooled.png     cam4 executed peak angle (+ = ipsi), change by position x epoch  (pooled twin of
                              `orofacial_cohort` angle_tuning, which is per animal)
  pose3d_change_pooled.png    3-D per-session medians (`pose3d_kinematics` sessions.csv): midline error toward ipsi,
                              miss distance, reach, peak speed, widest jaw opening, by position x epoch; jaw FAR_DIR /
                              CLOSE_DIR
  pose3d_cue_pooled.png       3-D cue-aligned traces (`pose3d_cue_traces`, engaged trials): midline angle and jaw
                              opening, each session's trace minus its animal's August pre trace; bold = mean over
                              animals, band = SEM over animals
Numbers behind each figure are written beside it (*_pooled.csv).

    python -m scripts.kinematics_pooled_figs [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EPOCHS = ("acute", "subacute", "chronic")
POS_CAM4 = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
POS_3D = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")    # L = ipsi, R = contra
POS_LABEL = dict(zip(POS_3D, POS_CAM4))
ACOL = {"PS92": "#1b9e77", "PS93": "#d95f02", "PS94": "#7570b3", "PS95": "#e7298a"}
ECOL = {"acute": "#c51b7d", "subacute": "#b8860b", "chronic": "#01665e"}
V36 = "lp_r4_heatmap_20261013__clean_v36"


def _is_aug_pre(df: pd.DataFrame) -> pd.Series:
    return (df.epoch == "pre") & (df.date.astype(str).str[-4:] >= "0801")


def pooled_change(df: pd.DataFrame, col: str, group: str, groups, rng) -> pd.DataFrame:
    """Rows (group value, epoch): per-animal mean change, the animal-weighted estimate and its boot_delta interval."""
    from wfield_local.analysis_kit import boot_delta
    out = []
    pre = df[_is_aug_pre(df)]
    for g in groups:
        pg = pre[pre[group] == g]
        base = {a: list(pg[pg.animal == a][col].dropna()) for a in ANIMALS}
        base = {a: v for a, v in base.items() if v}
        for e in EPOCHS:
            q = df[(df[group] == g) & (df.epoch == e)]
            post = {a: list(q[q.animal == a][col].dropna()) for a in ANIMALS}
            post = {a: v for a, v in post.items() if v}
            r = {"group": g, "epoch": e, "measure": col}
            for a in ANIMALS:
                r[a] = (np.mean(post[a]) - np.mean(base[a])) if a in post and a in base else np.nan
            iv = boot_delta(post, base, rng) if len(set(post) & set(base)) >= 3 else None
            r.update({"est": iv.point if iv else np.nan, "lo": iv.lo if iv else np.nan, "hi": iv.hi if iv else np.nan,
                      "n_animals": len(set(post) & set(base))})
            out.append(r)
    return pd.DataFrame(out)


def _sessions_minus_pre(df: pd.DataFrame, col: str, group: str) -> pd.DataFrame:
    pre = df[_is_aug_pre(df)].groupby(["animal", group])[col].mean().rename("base")
    d = df[df.epoch.isin(EPOCHS)].join(pre, on=["animal", group])
    d["chg"] = d[col] - d["base"]
    return d


def _panel(ax, d: pd.DataFrame, P: pd.DataFrame, g, ylabel: str | None, rng):
    for k, e in enumerate(EPOCHS):
        s = d[(d.epoch == e)]
        for a in ANIMALS:
            v = s[s.animal == a].chg.dropna()
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=ACOL[a], alpha=0.5, lw=0)
        r = P[(P.group == g) & (P.epoch == e)]
        if len(r) and np.isfinite(r.est.iloc[0]):
            est, lo, hi = r.est.iloc[0], r.lo.iloc[0], r.hi.iloc[0]
            ax.errorbar(k, est, yerr=[[est - lo], [hi - est]], fmt="o", color=ECOL[e], ms=6, capsize=3, lw=1.6, zorder=5)
            if lo > 0 or hi < 0:
                ax.text(k, hi, "*", ha="center", va="bottom", fontsize=11, color=ECOL[e])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(EPOCHS)), [e[:4] for e in EPOCHS], fontsize=7)
    ax.tick_params(labelsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5)


def _legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=ACOL[a], label=a) for a in ANIMALS]
    h += [Line2D([], [], marker="o", ls="-", color=ECOL[e], label=f"{e}: animal-weighted mean, 95 % CI") for e in EPOCHS]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=7, frameon=False)


def fig_angle(out: Path, rng) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from wfield_local.paths import PathResolver
    Pm = pd.read_csv(Path(PathResolver().root("orofacial")) / V36 / "cohort" / "tables" / "prepost" / "position_metrics.csv")
    P = pooled_change(Pm, "peak_angle_deg", "pos", POS_CAM4, rng)
    d = _sessions_minus_pre(Pm, "peak_angle_deg", "pos")
    fig, axs = plt.subplots(1, 6, figsize=(15, 3.6), sharey=True)
    for j, p in enumerate(POS_CAM4):
        _panel(axs[j], d[d.pos == p], P, p, "executed angle, change vs own pre\n(deg; + = ipsi)" if j == 0 else None, rng)
        axs[j].set_title(p.replace("_", " "), fontsize=9)
    fig.suptitle("cam4 executed peak lick angle, POOLED: each session minus its animal's August pre mean (all post-cue "
                 "licks, per-position median); * = 95 % CI excludes 0", fontsize=9)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out / "angle_change_pooled.png", dpi=130)
    plt.close(fig)
    return P


def fig_3d(out: Path, rng) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from wfield_local.paths import PathResolver
    S = pd.read_csv(Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics" / "sessions.csv")
    meas = [("midline_err_ipsi", "midline error\n(deg; + = ipsi)"), ("miss_mm", "closest approach\nto spout (mm)"),
            ("reach_mm", "reach from\nmouth zero (mm)"), ("speed_peak", "peak tongue\nspeed (mm/s)"),
            ("open_max_mm", "widest jaw\nopening (mm)")]
    long = []
    for m, _ in meas:
        for p in POS_3D:
            c = f"{m}|{p}"
            if c in S:
                long.append(S[["animal", "date", "epoch", c]].rename(columns={c: "v"}).assign(measure=m, pos=p))
    L = pd.concat(long, ignore_index=True)
    PP = []
    fig, axs = plt.subplots(len(meas) + 1, 6, figsize=(15, 2.5 * (len(meas) + 1)), sharex=True)
    for i, (m, lab) in enumerate(meas):
        Lm = L[L.measure == m]
        P = pooled_change(Lm, "v", "pos", POS_3D, rng).assign(measure=m)
        PP.append(P)
        d = _sessions_minus_pre(Lm, "v", "pos")
        for j, p in enumerate(POS_3D):
            _panel(axs[i, j], d[d.pos == p], P, p, f"{lab}, change" if j == 0 else None, rng)
            if i == 0:
                axs[i, j].set_title(POS_LABEL[p].replace("_", " "), fontsize=9)
    for j, (c, lab) in enumerate((("jaw_far_dir_mm", "jaw FAR_DIR (mm;\n+ = toward target)"),
                                  ("jaw_close_dir_mm", "jaw CLOSE_DIR (mm;\n+ = toward target)"))):
        J = S[["animal", "date", "epoch", c]].rename(columns={c: "v"}).assign(pos="all")
        P = pooled_change(J, "v", "pos", ["all"], rng).assign(measure=c)
        PP.append(P)
        _panel(axs[-1, j], _sessions_minus_pre(J, "v", "pos"), P, "all", f"{lab}, change", rng)
    for j in range(2, 6):
        axs[-1, j].axis("off")
    fig.suptitle("3-D tongue / jaw, POOLED: each session minus its animal's August pre mean (per-session medians of "
                 "trial licks); * = 95 % CI excludes 0", fontsize=10)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out / "pose3d_change_pooled.png", dpi=120)
    plt.close(fig)
    return pd.concat(PP, ignore_index=True)


def fig_cue(out: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from wfield_local.paths import PathResolver
    T = pd.read_csv(Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "cue_traces" / "traces.csv.gz")
    T = T[(T.set == "engaged") & T.measure.isin(["mid", "jaw_open"])]
    rows = []
    for (m, p), g in T.groupby(["measure", "position"]):
        aug = g[(g.epoch == "pre") & (g.date.astype(str).str[-4:] >= "0801")]
        base = aug.groupby(["animal", "t_ms"]).v.mean()
        for e in EPOCHS:
            q = g[g.epoch == e]
            if not len(q):
                continue
            q = q.join(base.rename("base"), on=["animal", "t_ms"])
            per = q.assign(d=q.v - q.base).groupby(["animal", "t_ms"]).d.mean().unstack(0)
            per = per[[a for a in ANIMALS if a in per]]
            rows.append(pd.DataFrame({"measure": m, "position": p, "epoch": e, "t_ms": per.index,
                                      "mean": per.mean(1).to_numpy(), "sem": (per.std(1) / np.sqrt(per.notna().sum(1))).to_numpy(),
                                      "n_animals": per.notna().sum(1).to_numpy()}))
    R = pd.concat(rows, ignore_index=True)
    fig, axs = plt.subplots(2, 6, figsize=(15, 6), sharex=True)
    for i, (m, lab) in enumerate((("mid", "midline angle, change\n(deg; + = ipsi)"), ("jaw_open", "jaw opening, change (mm)"))):
        for j, p in enumerate(POS_3D):
            ax = axs[i, j]
            for e in EPOCHS:
                r = R[(R.measure == m) & (R.position == p) & (R.epoch == e) & (R.n_animals >= 3)]
                ax.plot(r.t_ms, r["mean"], color=ECOL[e], lw=1.4, label=e)
                ax.fill_between(r.t_ms, r["mean"] - r["sem"], r["mean"] + r["sem"], color=ECOL[e], alpha=0.2, lw=0)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.axvline(0, color="0.8", lw=0.8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(POS_LABEL[p].replace("_", " "), fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7.5)
            if i == 1:
                ax.set_xlabel("ms from cue", fontsize=7.5)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("3-D cue-aligned traces, POOLED: each session's trace minus its animal's August pre trace, mean over "
                 "animals (band = SEM over animals; drawn where >= 3 animals); engaged trials", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "pose3d_cue_pooled.png", dpi=130)
    plt.close(fig)
    return R


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "figures" / "pooled"
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261025)
    import matplotlib
    matplotlib.use("Agg")
    A = fig_angle(out, rng)
    A.to_csv(out / "angle_change_pooled.csv", index=False)
    B = fig_3d(out, rng)
    B.to_csv(out / "pose3d_change_pooled.csv", index=False)
    C = fig_cue(out)
    C.to_csv(out / "pose3d_cue_pooled.csv", index=False)
    pd.set_option("display.width", 220)
    print(A[["group", "epoch", "PS92", "PS93", "PS94", "PS95", "est", "lo", "hi"]].round(2).to_string(index=False))
    print(B[B.group.isin(["far_R", "all"])][["measure", "group", "epoch", "PS92", "PS93", "PS94", "PS95", "est", "lo", "hi"]]
          .round(2).to_string(index=False))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

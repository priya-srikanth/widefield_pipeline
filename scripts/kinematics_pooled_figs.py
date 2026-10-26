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
  pose3d_paths_pooled.png     3-D tongue / jaw paths through the lick (front / side / top views), cohort pre path +
                              mean over animals of each session's change from its animal's pre path (twin of the
                              per-animal `pose3d_phase` trajectories3d)
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


def fig_paths3d(out: Path, shared: bool = False) -> pd.DataFrame:
    """POOLED 3-D tongue / jaw paths (twin of `pose3d_phase` trajectories3d, per animal): per (position, phase bin),
    each session's mean path minus its animal's August pre path, averaged within animal per epoch, then over animals
    (>= 3), added to the cohort pre path (mean over animals of each animal's August pre path) -- the convention of the
    cam4 `cohort_trajectory_xy_within` figures. Phase-normalised (`kind == "norm"`), all post-cue licks, bins with
    coverage >= 0.5 per session. ``shared`` (Priya 10-26, PI deck): every view row on ONE set of axes across the six
    spouts (same limits, equal aspect), so the mouth sits in the same place in every panel of a row ->
    pose3d_paths_pooled_shared.png."""
    import matplotlib.pyplot as plt
    from wfield_local.paths import PathResolver
    C = pd.read_csv(Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "phase" / "curves.csv.gz")
    C = C[(C.kind == "norm") & (C.lickset == "all") & C.measure.isin(["side", "out", "down", "j_side", "j_out", "j_down"])
          & (C["cov"] >= 0.5)]
    aug = C[(C.epoch == "pre") & (C.date.astype(str).str[-4:] >= "0801")]
    base = aug.groupby(["animal", "position", "measure", "x"]).v.mean()
    rows = []
    for (p, m), g in C.groupby(["position", "measure"]):
        b = base.xs((p, m), level=("position", "measure"))
        pre = b.groupby(level="x").mean().rename("pre")                      # cohort pre path (mean over animals)
        n_pre = b.groupby(level="x").size()
        rows.append(pd.DataFrame({"position": p, "measure": m, "epoch": "pre", "x": pre.index, "v": pre.to_numpy(),
                                  "n_animals": n_pre.reindex(pre.index).to_numpy()}))
        for e in EPOCHS:
            q = g[g.epoch == e].join(b.rename("base"), on=["animal", "x"])
            if not len(q):
                continue
            per = q.assign(d=q.v - q.base).groupby(["animal", "x"]).d.mean().unstack(0)
            d = per.mean(1)
            rows.append(pd.DataFrame({"position": p, "measure": m, "epoch": e, "x": d.index,
                                      "v": (pre.reindex(d.index) + d).to_numpy(), "n_animals": per.notna().sum(1).to_numpy()}))
    R = pd.concat(rows, ignore_index=True)
    R = R[R.n_animals >= 3]
    views = (("side", "down", "FRONT: side (+ = mouse left = ipsi) x down"), ("out", "down", "SIDE: out (forward) x down"),
             ("side", "out", "TOP: side (+ = ipsi) x out"))
    fig, axs = plt.subplots(3, 6, figsize=(11.5, 9.4) if shared else (16, 8.4), sharex="row" if shared else False,
                            sharey="row" if shared else False)
    col = {"pre": "#222222", **ECOL}
    for j, p in enumerate(POS_3D):
        for i, (a, b, lab) in enumerate(views):
            ax = axs[i, j]
            for part, ls in (("", "-"), ("j_", "--")):
                for e in ("pre",) + EPOCHS:
                    A = R[(R.position == p) & (R.measure == part + a) & (R.epoch == e)].set_index("x").v
                    B = R[(R.position == p) & (R.measure == part + b) & (R.epoch == e)].set_index("x").v
                    ix = A.index.intersection(B.index)
                    if not len(ix):
                        continue
                    yb = -B[ix] if b == "down" else B[ix]                     # draw "down" downward
                    ax.plot(A[ix], yb, ls, color=col[e], lw=1.5 if part == "" else 1.0,
                            label=(e if part == "" else None) if (i == 0 and j == 0) else None)
                    mid = ix[len(ix) // 2]
                    ax.plot(A[mid], (-B[mid] if b == "down" else B[mid]), "o", color=col[e], ms=3)
            ax.plot(0, 0, "+", color="k", ms=8)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(POS_LABEL[p].replace("_", " "), fontsize=9)
            if j == 0:
                ax.set_ylabel(lab + " (mm)", fontsize=7)
            ax.set_aspect("equal", adjustable="box" if shared else "datalim")
    if shared:                                                # one box per view row: data of all spouts + the mouth
        for i, (a, b, _lab) in enumerate(views):
            xs = R[R.measure.isin([a, "j_" + a])].v.to_numpy()
            ys = R[R.measure.isin([b, "j_" + b])].v.to_numpy() * (-1 if b == "down" else 1)
            (x0, x1), (y0, y1) = (min(np.nanmin(xs), 0), max(np.nanmax(xs), 0)), (min(np.nanmin(ys), 0), max(np.nanmax(ys), 0))
            mx, my = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
            axs[i, 0].set_xlim(x0 - mx, x1 + mx)
            axs[i, 0].set_ylim(y0 - my, y1 + my)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("3-D tongue (solid) and jaw (dashed) paths through the lick, POOLED within animal: cohort August-pre path\n"
                 "+ mean over animals of each session's change from its animal's own pre path (phase 0.1-0.9, all post-cue "
                 "licks, >= 3 animals); + = mouth zero; dot = mid-lick; mm"
                 + ("\naxes shared across spouts within each view" if shared else ""), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.2 if shared else None)
    fig.savefig(out / ("pose3d_paths_pooled_shared.png" if shared else "pose3d_paths_pooled.png"), dpi=130)
    plt.close(fig)
    return R


DAY_BINS = ((1, 3), (4, 7), (8, 14), (15, 21), (22, 35), (36, 70))


def _draw_days(ax, S, aug, col, key, rows):
    base = S[aug].groupby("animal")[col].mean()
    d = S[S.day.notna()].assign(chg=S[col] - S.animal.map(base))
    d = d[(d.day > 0) | aug]
    for a in ANIMALS:
        q = d[d.animal == a]
        ax.scatter(q.day, q.chg, s=8, color=ACOL[a], alpha=0.55, lw=0)
    xs, ms, ss = [], [], []
    for lo, hi in DAY_BINS:
        per = d[(d.day >= lo) & (d.day <= hi)].groupby("animal").chg.mean().dropna()
        ok = len(per) >= 3
        rows.append({"measure": key, "bin": f"{lo}-{hi}", "n_animals": len(per),
                     **{a: per.get(a, np.nan) for a in ANIMALS},
                     "mean": per.mean() if ok else np.nan, "sem": per.std() / np.sqrt(len(per)) if ok else np.nan})
        if ok:
            xs.append((lo + hi) / 2)
            ms.append(per.mean())
            ss.append(per.std() / np.sqrt(len(per)))
    xs, ms, ss = np.asarray(xs), np.asarray(ms), np.asarray(ss)
    ax.plot(xs, ms, color="k", lw=1.6)
    ax.fill_between(xs, ms - ss, ms + ss, color="k", alpha=0.15, lw=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.tick_params(labelsize=6.5)


def fig_days(out: Path) -> pd.DataFrame:
    """POOLED twin of the per-animal 3-D summaries (`pose3d_summary_fig`): every session's change from ITS animal's
    August pre mean against days from that animal's stroke, all animals on one axis (dots coloured by animal); line =
    mean over animals of each animal's mean change within day bins (>= 3 animals), band = SEM over animals."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from wfield_local.epochs import days_since_stroke
    from wfield_local.paths import PathResolver
    S = pd.read_csv(Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics" / "sessions.csv")
    S["day"] = [days_since_stroke(f"{a}_{str(d)[-4:]}") for a, d in zip(S.animal, S.date)]
    aug = _is_aug_pre(S)
    meas = [("midline_err_ipsi", "midline error (deg; + = ipsi)"), ("miss_mm", "closest approach (mm)"),
            ("lat_err_mm", "lateral error at closest (mm)"), ("reach_mm", "reach (mm)"),
            ("speed_peak", "peak speed (mm/s)"), ("open_max_mm", "widest jaw opening (mm)")]
    rows = []
    fig, axs = plt.subplots(len(meas) + 1, 6, figsize=(16, 2.3 * (len(meas) + 1)), sharex=True)
    for i, (m, lab) in enumerate(meas):
        for j, p in enumerate(POS_3D):
            c = f"{m}|{p}"
            if c not in S:
                axs[i, j].axis("off")
                continue
            _draw_days(axs[i, j], S, aug, c, c, rows)
            if i == 0:
                axs[i, j].set_title(POS_LABEL[p].replace("_", " "), fontsize=9)
            if j == 0:
                axs[i, j].set_ylabel(lab + ",\nchange", fontsize=7)
    for j, (c, lab) in enumerate((("jaw_far_dir_mm", "jaw FAR_DIR (mm), change"),
                                  ("jaw_close_dir_mm", "jaw CLOSE_DIR (mm), change"))):
        _draw_days(axs[-1, j], S, aug, c, c, rows)
        axs[-1, j].set_ylabel(lab, fontsize=7)
        axs[-1, j].set_xlabel("days from stroke (left of 0 = August pre)", fontsize=7)
    for j in range(2, 6):
        axs[-1, j].axis("off")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=ACOL[a], label=a) for a in ANIMALS]
               + [Line2D([], [], color="k", label="mean over animals of within-animal bin means (+- SEM)")],
               loc="lower right", ncol=5, fontsize=7, frameon=False)
    fig.suptitle("3-D measures over days, POOLED within animal: each session minus its animal's August pre mean; "
                 "day bins 1-3, 4-7, 8-14, 15-21, 22-35, 36-70", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out / "pose3d_days_pooled.png", dpi=120)
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_phase(out: Path) -> pd.DataFrame:
    """POOLED twin of the per-animal lick-phase figures (`pose3d_phase` phase_angle / phase_dist): phase-normalised
    midline angle and 3-D distance from the mouth zero, each session minus its animal's August pre curve, averaged
    within animal per epoch, then over animals (>= 3; band = SEM over animals)."""
    import matplotlib.pyplot as plt
    from wfield_local.paths import PathResolver
    C = pd.read_csv(Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "phase" / "curves.csv.gz")
    C = C[(C.kind == "norm") & (C.lickset == "all") & C.measure.isin(["mid", "dist"]) & (C["cov"] >= 0.5)]
    aug = C[(C.epoch == "pre") & (C.date.astype(str).str[-4:] >= "0801")]
    base = aug.groupby(["animal", "position", "measure", "x"]).v.mean()
    rows = []
    for (p, m), g in C.groupby(["position", "measure"]):
        b = base.xs((p, m), level=("position", "measure"))
        for e in EPOCHS:
            q = g[g.epoch == e].join(b.rename("base"), on=["animal", "x"])
            if not len(q):
                continue
            per = q.assign(d=q.v - q.base).groupby(["animal", "x"]).d.mean().unstack(0)
            rows.append(pd.DataFrame({"position": p, "measure": m, "epoch": e, "x": per.index,
                                      "mean": per.mean(1).to_numpy(),
                                      "sem": (per.std(1) / np.sqrt(per.notna().sum(1))).to_numpy(),
                                      "n_animals": per.notna().sum(1).to_numpy()}))
    R = pd.concat(rows, ignore_index=True)
    fig, axs = plt.subplots(2, 6, figsize=(15, 5.6), sharex=True)
    for i, (m, lab) in enumerate((("mid", "midline angle, change\n(deg; + = ipsi)"),
                                  ("dist", "3-D distance from mouth\nzero, change (mm)"))):
        for j, p in enumerate(POS_3D):
            ax = axs[i, j]
            for e in EPOCHS:
                r = R[(R.measure == m) & (R.position == p) & (R.epoch == e) & (R.n_animals >= 3)]
                ax.plot(r.x, r["mean"], color=ECOL[e], lw=1.4, label=e)
                ax.fill_between(r.x, r["mean"] - r["sem"], r["mean"] + r["sem"], color=ECOL[e], alpha=0.2, lw=0)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(POS_LABEL[p].replace("_", " "), fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7.5)
            if i == 1:
                ax.set_xlabel("lick phase (0 = out, 1 = back in)", fontsize=7)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("3-D lick phase, POOLED within animal: each session's phase curve minus its animal's August pre curve, "
                 "mean over animals (band = SEM over animals, >= 3 animals); all post-cue licks", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "pose3d_phase_pooled.png", dpi=130)
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
    D = fig_paths3d(out)
    D.to_csv(out / "pose3d_paths_pooled.csv", index=False)
    fig_paths3d(out, shared=True)
    fig_days(out).to_csv(out / "pose3d_days_pooled.csv", index=False)
    fig_phase(out).to_csv(out / "pose3d_phase_pooled.csv", index=False)
    pd.set_option("display.width", 220)
    print(A[["group", "epoch", "PS92", "PS93", "PS94", "PS95", "est", "lo", "hi"]].round(2).to_string(index=False))
    print(B[B.group.isin(["far_R", "all"])][["measure", "group", "epoch", "PS92", "PS93", "PS94", "PS95", "est", "lo", "hi"]]
          .round(2).to_string(index=False))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

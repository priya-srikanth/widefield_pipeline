"""POOLED, within-animal pre-normalised versions of the per-animal kinematics figures (Priya 10-25: "a pre-stroke-
normalized within-animal normalized cross-animal session pooled version of fig 2 (and anything else that needs a pooled
fig)"). Every value is a SESSION minus ITS ANIMAL's August pre mean; the bold estimate is the mean over animals of each
animal's mean change (animals weighted equally) and the interval is `analysis_kit.boot_delta` (paired animals ->
sessions bootstrap, 95 %). Dots = sessions: OPEN markers, shape = animal, colour = epoch (PI decision A13, Priya 11-07;
they were filled and coloured by animal). A cell needs >= 3 animals to be drawn with an interval.

  angle_change_pooled.png     cam4 executed peak angle (+ = ipsi), change by position x epoch  (pooled twin of
                              `orofacial_cohort` angle_tuning, which is per animal)
  angle_change_pooled_contra.png  the same signed + = CONTRALESIONAL (goal-page convention, Priya 10-30); re-seeded,
                              so every number is the + = ipsi one negated. `--angle-only` draws just these two.
  pose3d_change_pooled.png    3-D per-session medians (`sessions3d`): midline error toward ipsi,
                              miss distance, reach, peak speed, widest jaw opening, by position x epoch; jaw FAR_DIR /
                              CLOSE_DIR. Also pose3d_days_pooled (`fig_days`). LICK / TRIAL GATE (Priya 11-07, A20 gap):
                              the per-session medians are RE-MADE over camera-working-period licks / jaw trials
                              (`pose3d_summary_fig.gated_sessions` on the kinematics licks / jaw_trials snapshot ->
                              `trial_classes.gate_kin_licks`), written beside the figures as pose3d_sessions.csv; before
                              11-07 this read the UNGATED kinematics/sessions.csv (10-23), which is now read only
                              under ``WFIELD_KIN_GATE=legacy`` (outputs then go to the _legacygate folder)
  pose3d_paths_pooled.png     3-D tongue / jaw paths through the lick (front / side / top views), cohort pre path +
                              mean over animals of each session's change from its animal's pre path (twin of the
                              per-animal `pose3d_phase` trajectories3d)
  pose3d_cue_pooled.png       3-D cue-aligned traces (`pose3d_cue_traces`, camera-working-period trials since 11-06): midline angle and jaw
                              opening, each session's trace minus its animal's August pre trace; bold = mean over
                              animals, band = SEM over animals
Numbers behind each figure are written beside it (*_pooled.csv).

``--contra`` (Priya 10-30, the goal page's sign): only the 3-D figures again, every lesion-signed axis signed + =
CONTRALESIONAL (mouse right; all four lesions left), flipped at PLOT level -- the same numbers negated, the + = ipsi
figures and CSVs untouched -> ``*_contra.png`` (+ ``pose3d_change_pooled_contra.csv``, the midline rows negated). The
bootstrap of `pose3d_change_pooled` is replayed from the same seed and draw order, so every interval is the + = ipsi one.
Side axes of the 3-D paths are mirrored (+ = mouse right); for the CENTRE spouts the outward-signed lateral error is +
= mouse left, so it is negated there too.

    python -m scripts.kinematics_pooled_figs [--out DIR] [--contra]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local.sig_marks import reserve_headroom, star_top
from wfield_local.pose3d import pose3d_dir

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
EPOCHS = ("acute", "subacute", "chronic")
POS_CAM4 = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
POS_3D = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")    # L = ipsi, R = contra
POS_LABEL = dict(zip(POS_3D, POS_CAM4))
#: THE epoch palette (orofacial_cohort.EPOCH_LINE; A13, Priya 11-07: the private copy had drifted, acute #c51b7d).
#: The private animal colour dict is gone (config rule 3): animals are told apart by marker SHAPE.
from wfield_local.orofacial_cohort import EPOCH_LINE as _EPOCH_LINE  # noqa: E402

ECOL = {e: _EPOCH_LINE[e] for e in EPOCHS}
V36 = "lp_r4_heatmap_20261013__clean_v36"


#: lesion-signed 3-D measures (+ = ipsi = mouse LEFT for these left lesions): negated at plot level by ``contra``
SIGNED_3D = ("midline_err_ipsi", "mid", "side", "j_side")
#: outward-signed lateral errors: + = mouse LEFT at the CENTRE spouts only (outward elsewhere, no lesion sign)
CENTRE_SIGNED_3D = ("lat_err_mm",)


def _sg(contra: bool, m: str, pos: str | None = None) -> float:
    """Plot-level sign for measure ``m`` (at position ``pos``): -1 when ``contra`` and the measure is lesion-signed."""
    if not contra:
        return 1.0
    if m in SIGNED_3D:
        return -1.0
    return -1.0 if (m in CENTRE_SIGNED_3D and pos is not None and "center" in pos) else 1.0


def _is_aug_pre(df: pd.DataFrame) -> pd.Series:
    from wfield_local import config
    return (df.epoch == "pre") & (~df.date.astype(str).map(config.early_pre))


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
    from wfield_local.cat_points import animal_marker
    for k, e in enumerate(EPOCHS):
        s = d[(d.epoch == e)]
        for a in ANIMALS:
            v = s[s.animal == a].chg.dropna()
            # the jitter draw comes from the SHARED rng the bootstrap also uses: drawn exactly as before (same call,
            # same size, same order), or every later interval in the figure set would move
            xj = k + rng.uniform(-0.18, 0.18, len(v))
            ax.plot(xj, v, ls="none", marker=animal_marker(a), ms=3.2, mfc="none", mec=ECOL[e], mew=0.7, alpha=0.6)
        r = P[(P.group == g) & (P.epoch == e)]
        if len(r) and np.isfinite(r.est.iloc[0]):
            est, lo, hi = r.est.iloc[0], r.lo.iloc[0], r.hi.iloc[0]
            ax.errorbar(k, est, yerr=[[est - lo], [hi - est]], fmt="o", color=ECOL[e], ms=6, capsize=3, lw=1.6, zorder=5)
            if lo > 0 or hi < 0:
                star_top(ax, k, "*", ECOL[e])       # top band, clear of the data and the 0 line (Priya 10-30)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(EPOCHS)), [e[:4] for e in EPOCHS], fontsize=7)
    ax.tick_params(labelsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5)


def _legend(fig):
    from matplotlib.lines import Line2D

    from wfield_local import cat_points as cp
    h = cp.animal_handles(ANIMALS)
    h += [Line2D([], [], marker="o", ls="-", color=ECOL[e], label=f"{e}: animal-weighted mean, 95 % CI") for e in EPOCHS]
    fig.legend(handles=h, loc="lower center", ncol=7, fontsize=7, frameon=False)


def fig_angle(out: Path, rng, contra: bool = False, save: bool = True) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from wfield_local import config
    from wfield_local.paths import PathResolver
    from wfield_local import orofacial_cohort as oc        # the cohort's gate-named tables (A27; WFIELD_KIN_GATE)
    Pm = pd.read_csv(oc.cohort_dir("tables", run=V36, rv=PathResolver()) / "position_metrics.csv")
    side = "ipsi"
    if contra:      # the table is + = image right = ipsi for a LEFT lesion; flip per animal from its lesion side
        lat = {a: str((v or {}).get("stroke_laterality", "L")) for a, v in config.animals().items()}
        Pm = Pm.assign(peak_angle_deg=Pm.peak_angle_deg * Pm.animal.map(lambda a: -1.0 if lat.get(a, "L") == "L" else 1.0))
        side = "contra"
    P = pooled_change(Pm, "peak_angle_deg", "pos", POS_CAM4, rng)
    d = _sessions_minus_pre(Pm, "peak_angle_deg", "pos")
    fig, axs = plt.subplots(1, 6, figsize=(15, 3.6), sharey=True)
    for j, p in enumerate(POS_CAM4):
        _panel(axs[j], d[d.pos == p], P, p, f"executed angle, change vs own pre\n(deg; + = {side}lesional)" if j == 0
               else None, rng)
        axs[j].set_title(p.replace("_", " "), fontsize=9)
    fig.suptitle("cam4 executed peak lick angle, POOLED: each session minus its animal's August pre mean (all post-cue "
                 "licks, per-position median); * = 95 % CI excludes 0", fontsize=9)
    _legend(fig)
    reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    if save:                                      # save=False: replay the draws only (--contra)
        save_png_svg(fig, out / ("angle_change_pooled_contra.png" if contra else "angle_change_pooled.png"), dpi=130)
    plt.close(fig)
    return P


_S3D: dict = {}


def sessions3d() -> pd.DataFrame:
    """The 3-D per-session medians behind `fig_3d` / `fig_days` (one row per session; ``<measure>|<position>`` columns +
    jaw FAR_DIR / CLOSE_DIR), under THE kinematics gate (A20, Priya 11-07: the pooled-table gap). "working" (default):
    re-made from the kinematics licks / jaw trials snapshot through `trial_classes.gate_kin_licks` with the same
    per-session medians as `pose3d_kinematics._session_result` (`pose3d_summary_fig.gated_sessions`), so no session is
    recomputed; a session with no camera classes contributes nothing. ``WFIELD_KIN_GATE=legacy``: the ungated 10-23
    kinematics/sessions.csv, as before. Memoised per gate (the gate is read once per figure set)."""
    tc = _tc()
    g = tc.kin_gate()
    if g not in _S3D:
        d3 = pose3d_dir()
        if g == "working":
            from scripts.pose3d_summary_fig import gated_sessions
            _S3D[g] = gated_sessions(pd.read_csv(d3 / "kinematics" / "licks.csv.gz", dtype={"date": str}),
                                     pd.read_csv(d3 / "kinematics" / "jaw_trials.csv.gz", dtype={"date": str}))
        else:
            _S3D[g] = pd.read_csv(d3 / "kinematics" / "sessions.csv", dtype={"date": str})
    return _S3D[g].copy()


def fig_3d(out: Path, rng, contra: bool = False) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    S = sessions3d()
    meas = [("midline_err_ipsi", "midline error\n(deg; + = contra)" if contra else "midline error\n(deg; + = ipsi)"), ("miss_mm", "closest approach\nto spout (mm)"),
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
        Pd, dd = P, d
        if _sg(contra, m) < 0:                     # plot level: same numbers negated (CI bounds swap)
            Pd = P.assign(est=-P.est, lo=-P.hi, hi=-P.lo)
            dd = d.assign(chg=-d.chg)
        for j, p in enumerate(POS_3D):
            _panel(axs[i, j], dd[dd.pos == p], Pd, p, f"{lab}, change" if j == 0 else None, rng)
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
                 "trial licks); * = 95 % CI excludes 0"
                 + ("; midline error signed + = CONTRALESIONAL (mouse right)" if contra else ""), fontsize=10)
    _legend(fig)
    reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_png_svg(fig, out / ("pose3d_change_pooled_contra.png" if contra else "pose3d_change_pooled.png"), dpi=120)
    plt.close(fig)
    return pd.concat(PP, ignore_index=True)


def _tc():
    """`wfield_local.trial_classes` (the kinematics gate's folder helpers, A20 11-06)."""
    from wfield_local import trial_classes as tc
    return tc


def fig_cue(out: Path, contra: bool = False) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    from wfield_local import config
    from scripts.pose3d_cue_traces import out_dir as _cue_out     # routed + gate-named (A20, 11-06)
    T = pd.read_csv(_cue_out(pose3d_dir(), "cue") / "traces.csv.gz")
    sset = "working" if (T.set == "working").any() else "engaged"     # pose3d_cue_traces' all-trials set (11-06: working)
    T = T[(T.set == sset) & T.measure.isin(["mid", "jaw_open"])]
    rows = []
    for (m, p), g in T.groupby(["measure", "position"]):
        aug = g[(g.epoch == "pre") & (~g.date.astype(str).map(config.early_pre))]
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
    side = "contra" if contra else "ipsi"
    for i, (m, lab) in enumerate((("mid", f"midline angle, change\n(deg; + = {side})"), ("jaw_open", "jaw opening, change (mm)"))):
        for j, p in enumerate(POS_3D):
            ax = axs[i, j]
            for e in EPOCHS:
                r = R[(R.measure == m) & (R.position == p) & (R.epoch == e) & (R.n_animals >= 3)]
                mu = r["mean"] * _sg(contra, m)
                ax.plot(r.t_ms, mu, color=ECOL[e], lw=1.4, label=e)
                ax.fill_between(r.t_ms, mu - r["sem"], mu + r["sem"], color=ECOL[e], alpha=0.2, lw=0)
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
                 "animals (band = SEM over animals; drawn where >= 3 animals); "
                 + ("camera-working-period trials" if sset == "working" else "engaged trials (legacy gate)")
                 + ("; midline angle signed + = CONTRALESIONAL (mouse right)" if contra else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_svg(fig, out / ("pose3d_cue_pooled_contra.png" if contra else "pose3d_cue_pooled.png"), dpi=130)
    plt.close(fig)
    return R


def _paths3d_sig(C: pd.DataFrame, aug: pd.DataFrame) -> dict:
    """{(position, epoch, measure): bool Series over phase x} for the TONGUE coordinates side / out / down: the epoch's
    path differs from pre (Priya 11-03: stats on the 3-D paths, as the 2-D cam4 paths have). The cam4 test
    (`orofacial_cohort._path_sig`): per-session curve minus its animal's August-pre mean curve, `cluster_permutation`
    of the epoch's sessions vs the August-pre sessions, labels permuted within animal. Three coordinates per path, so
    each is tested at ALPHA / 3 (the 2-D figure, with two, uses ALPHA / 2). Sign-symmetric, so + = contra flips do not
    change it."""
    from wfield_local.orofacial_cohort import CLUSTER_PERM
    from wfield_local.orofacial_stats import ALPHA, cluster_permutation
    out = {}
    T = C[C.measure.isin(["side", "out", "down"])]
    T = T[(T.epoch != "pre") | T.session.isin(set(aug.session))]
    for (p, m), g in T.groupby(["position", "measure"]):
        pv = g.pivot_table(index=["animal", "session", "epoch"], columns="x", values="v")
        an = pv.index.get_level_values(0).to_numpy()
        ep = pv.index.get_level_values(2).to_numpy()
        base = pv[ep == "pre"].groupby(level=0).mean()
        ch = pv.to_numpy(float) - base.reindex(an).to_numpy(float)
        for e in EPOCHS:
            keep = (ep == "pre") | (ep == e)
            if (ep == e).sum() < 2:
                continue
            sub = ch[keep]
            ok = np.isfinite(sub).all(0)                       # bins every kept session covers
            mask = np.zeros(len(pv.columns), bool)
            if ok.sum() >= 3:
                mask[ok] = cluster_permutation(sub[:, ok], ep[keep] == e, an[keep], n_perm=CLUSTER_PERM,
                                               alpha=ALPHA / 3)
            out[(p, e, m)] = pd.Series(mask, index=pv.columns)
    return out


def fig_paths3d(out: Path, shared: bool = False, contra: bool = False) -> pd.DataFrame:
    """POOLED 3-D tongue / jaw paths (twin of `pose3d_phase` trajectories3d, per animal): per (position, phase bin),
    each session's mean path minus its animal's August pre path, averaged within animal per epoch, then over animals
    (>= 3), added to the cohort pre path (mean over animals of each animal's August pre path) -- the convention of the
    cam4 `cohort_trajectory_xy_within` figures. Phase-normalised (`kind == "norm"`), all post-cue licks, bins with
    coverage >= 0.5 per session. ``shared`` (Priya 10-26, PI deck): every view row on ONE set of axes across the six
    spouts (same limits, equal aspect), so the mouth sits in the same place in every panel of a row ->
    pose3d_paths_pooled_shared.png."""
    import matplotlib.pyplot as plt
    from wfield_local import config
    C = pd.read_csv(_tc().kin_out_dir(pose3d_dir(), "phase") / "curves.csv.gz")    # routed + gate-named (A20)
    C = C[(C.kind == "norm") & (C.lickset == "all") & C.measure.isin(["side", "out", "down", "j_side", "j_out", "j_down"])
          & (C["cov"] >= 0.5)]
    aug = C[(C.epoch == "pre") & (~C.date.astype(str).map(config.early_pre))]
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
    sig = _paths3d_sig(C, aug)
    if contra:                                    # plot level: side signed + = mouse RIGHT = contra
        R = R.assign(v=R.v * R.measure.map(lambda m: _sg(True, m)))
    # contra (Priya 11-03): the side axis of the FRONT / TOP views is drawn in CAMERA orientation -- contra on the image
    # LEFT as in the face videos, so the + = contra axis is inverted and its ends are labelled
    views = (("side", "down", "FRONT: side (+ = contra) x down" if contra else
              "FRONT: side (+ = mouse left = ipsi) x down"), ("out", "down", "SIDE: out (forward) x down"),
             ("side", "out", "TOP: side (+ = contra) x out" if contra else "TOP: side (+ = ipsi) x out"))
    fig, axs = plt.subplots(3, 6, figsize=((11.5, 10.2) if contra else (11.5, 9.4)) if shared else ((16, 9.2) if contra else (16, 8.4)), sharex="row" if shared else False,
                            sharey="row" if shared else False)
    col = {"pre": _EPOCH_LINE["pre"], **ECOL}
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
                    if part == "" and e != "pre":             # where this epoch's TONGUE path differs from pre
                        ma = sig.get((p, e, a), pd.Series(dtype=bool)).reindex(ix, fill_value=False).to_numpy()
                        mb = sig.get((p, e, b), pd.Series(dtype=bool)).reindex(ix, fill_value=False).to_numpy()
                        for sel, mk, fc in ((ma & ~mb, "o", "none"), (mb & ~ma, "^", "none"), (ma & mb, "o", col[e])):
                            if sel.any():
                                ax.scatter(A[ix][sel], yb[sel], marker=mk, s=11, facecolors=fc, edgecolors=col[e],
                                           linewidths=0.7, zorder=4)
            ax.plot(0, 0, "+", color="k", ms=8)
            ax.tick_params(labelsize=6.5)
            if i == 0:
                ax.set_title(POS_LABEL[p].replace("_", " "), fontsize=9)
            if j == 0:
                ax.set_ylabel(lab + " (mm)", fontsize=7)
            ax.set_aspect("equal", adjustable="box" if shared else "datalim")
            if contra and a == "side":                                # camera view: contra LEFT
                ax.set_xlabel("side (mm; + = contra)", fontsize=6.5)
    if shared:                                                # one box per view row: data of all spouts + the mouth
        for i, (a, b, _lab) in enumerate(views):
            xs = R[R.measure.isin([a, "j_" + a])].v.to_numpy()
            ys = R[R.measure.isin([b, "j_" + b])].v.to_numpy() * (-1 if b == "down" else 1)
            (x0, x1), (y0, y1) = (min(np.nanmin(xs), 0), max(np.nanmax(xs), 0)), (min(np.nanmin(ys), 0), max(np.nanmax(ys), 0))
            mx, my = 0.06 * (x1 - x0), 0.06 * (y1 - y0)
            axs[i, 0].set_xlim(x0 - mx, x1 + mx)
            axs[i, 0].set_ylim(y0 - my, y1 + my)
    if contra:
        from wfield_local.camera_view import camera_x
        for i, (a, _b, _lab) in enumerate(views):
            if a == "side":
                for ax in axs[i]:
                    camera_x(ax)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("3-D tongue (solid) and jaw (dashed) paths through the lick, POOLED within animal: cohort August-pre path\n"
                 "+ mean over animals of each session's change from its animal's own pre path (phase 0.1-0.9, all post-cue "
                 "licks, >= 3 animals); + = mouth zero; dot = mid-lick; mm"
                 + ("\naxes shared across spouts within each view" if shared else "")
                 + "\nmarkers on the tongue paths = where that epoch differs from pre (cluster permutation per coordinate on "
                   "session changes from the animal's own Aug pre,\nlabels permuted within animal, p < .05/3 each): "
                   "open circle = horizontal axis only, open triangle = vertical only, filled = both"
                 + ("\nside signed + = CONTRALESIONAL (mouse right); FRONT / TOP in CAMERA VIEW: contra on the image "
                    "LEFT as in the face videos (side axis inverted, ends labelled)" if contra else ""),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.2 if shared else None, h_pad=2.2 if contra else None)
    save_png_svg(fig, out / (("pose3d_paths_pooled_shared" if shared else "pose3d_paths_pooled")
                       + ("_contra_cam" if contra else "") + ".png"), dpi=130)   # 11-03: *_contra (mirrored) superseded
    plt.close(fig)
    return R


DAY_BINS = ((1, 3), (4, 7), (8, 14), (15, 21), (22, 35), (36, 70))


def _draw_days(ax, S, aug, col, key, rows, sg: float = 1.0):
    """Sessions over days (A13, Priya 11-07): open markers, shape = animal, colour = epoch, no line; filled = epoch mean
    +- SEM over animals. The day-bin table (rows -> pose3d_days_pooled.csv) is computed exactly as before; it is no
    longer drawn as a black line across the bins."""
    from wfield_local import cat_points as cp
    base = S[aug].groupby("animal")[col].mean()
    d = S[S.day.notna()].assign(chg=(S[col] - S.animal.map(base)) * sg)          # sg: plot-level sign (--contra)
    d = d[(d.day > 0) | aug]
    cp.day_dots(ax, d.day, d.chg, d.animal, d.epoch, ms=2.8, summary_ms=4.5)
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
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.tick_params(labelsize=6.5)


def fig_days(out: Path, contra: bool = False) -> pd.DataFrame:
    """POOLED twin of the per-animal 3-D summaries (`pose3d_summary_fig`): every session's change from ITS animal's
    August pre mean against days from that animal's stroke, all animals on one axis (open markers, shape = animal,
    colour = epoch); filled = epoch mean +- SEM over animals. The day-bin means (>= 3 animals, SEM over animals) are
    still computed and written (pose3d_days_pooled.csv), not drawn."""
    import matplotlib.pyplot as plt
    from wfield_local.epochs import days_since_stroke
    S = sessions3d()
    S["day"] = [days_since_stroke(f"{a}_{str(d)[-4:]}") for a, d in zip(S.animal, S.date)]
    aug = _is_aug_pre(S)
    meas = [("midline_err_ipsi", f"midline error (deg; + = {'contra' if contra else 'ipsi'})"),
            ("miss_mm", "closest approach (mm)"),
            ("lat_err_mm", "lateral error at closest (mm; + outward, centre + = contra)" if contra else
             "lateral error at closest (mm)"), ("reach_mm", "reach (mm)"),
            ("speed_peak", "peak speed (mm/s)"), ("open_max_mm", "widest jaw opening (mm)")]
    rows = []
    fig, axs = plt.subplots(len(meas) + 1, 6, figsize=(16, 2.3 * (len(meas) + 1)), sharex=True)
    for i, (m, lab) in enumerate(meas):
        for j, p in enumerate(POS_3D):
            c = f"{m}|{p}"
            if c not in S:
                axs[i, j].axis("off")
                continue
            _draw_days(axs[i, j], S, aug, c, c, rows, _sg(contra, m, p))
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
    from wfield_local import cat_points as cp
    fig.legend(handles=cp.legend_handles(ANIMALS, ("pre",) + EPOCHS, sessions_label="{a}",
                                         epoch_label="{e}: mean +- SEM over animals"),
               loc="lower right", bbox_to_anchor=(0.98, 0.02), ncol=2, fontsize=8, frameon=False)
    fig.suptitle("3-D measures over days, POOLED within animal: each session minus its animal's August pre mean; "
                 "open = sessions (shape = animal, colour = epoch), filled = epoch mean +- SEM over animals; day-bin means (1-3 .. 36-70) in pose3d_days_pooled.csv"
                 + ("; midline error (and the centre spouts' lateral error) signed + = CONTRALESIONAL" if contra else ""),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save_png_svg(fig, out / ("pose3d_days_pooled_contra.png" if contra else "pose3d_days_pooled.png"), dpi=120)
    plt.close(fig)
    return pd.DataFrame(rows)


PHASE_WIN = (0.1, 0.9)           # shown AND tested: the noisy lick ends are left out, as in the cam4 phase figures


def _smooth(y: np.ndarray, sigma_bins: float = 1.0) -> np.ndarray:
    """NaN-aware Gaussian smoothing over phase bins for DISPLAY (1 bin, as the cam4 phase figures); not circular,
    because the curve is cut to PHASE_WIN. The test runs on the unsmoothed curves."""
    y = np.asarray(y, float)
    k = np.exp(-0.5 * (np.arange(-3, 4) / sigma_bins) ** 2)
    ok = np.isfinite(y)
    num = np.convolve(np.where(ok, y, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    return np.where(ok, num / np.where(den > 0, den, np.nan), np.nan)


def phase_data(contra: bool = False) -> pd.DataFrame:
    """Figure data for `fig_phase`, long: ``row`` = "mean" (mean over animals of each animal's mean change, SEM over
    animals, n_animals), "animal" (each animal's mean change) or "sig" (cluster-permutation mask). Each session's
    phase curve minus its animal's August-pre curve, phase PHASE_WIN, coverage >= 0.5 per session; the midline angle
    signed + = contra when ``contra``. Test per (spout, measure, epoch) = the cam4 phase figures' test
    (`orofacial_cohort.fig_cohort_delta`): `cluster_permutation` of the epoch's session changes against the August-pre
    sessions' changes, labels permuted within animal, p < .05; on the bins every tested session covers."""
    from wfield_local import config
    from wfield_local.orofacial_cohort import CLUSTER_PERM
    from wfield_local.orofacial_stats import cluster_permutation
    C = pd.read_csv(_tc().kin_out_dir(pose3d_dir(), "phase") / "curves.csv.gz")    # routed + gate-named (A20)
    C = C[(C.kind == "norm") & (C.lickset == "all") & C.measure.isin(["mid", "dist"]) & (C["cov"] >= 0.5)
          & C.x.between(*PHASE_WIN)]
    C = C[(C.epoch != "pre") | (~C.date.astype(str).map(config.early_pre))]          # August pre only
    rows = []
    for (p, m), g in C.groupby(["position", "measure"]):
        pv = g.pivot_table(index=["animal", "session", "epoch"], columns="x", values="v") * _sg(contra, m)
        an = pv.index.get_level_values(0).to_numpy()
        ep = pv.index.get_level_values(2).to_numpy()
        base = pv[ep == "pre"].groupby(level=0).mean()
        ch = pd.DataFrame(pv.to_numpy(float) - base.reindex(an).to_numpy(float), index=pv.index, columns=pv.columns)
        xs = pv.columns.to_numpy(float)
        for e in ("pre",) + EPOCHS:
            sel = ep == e
            if not sel.any():
                continue
            per = ch[sel].groupby(level=0).mean()                                  # animals x bins
            n = per.notna().sum(0).to_numpy()
            mu, sd = per.mean(0).to_numpy(), per.std(0).to_numpy()
            rows.append(pd.DataFrame({"row": "mean", "position": p, "measure": m, "epoch": e, "x": xs, "v": mu,
                                      "sem": sd / np.sqrt(np.maximum(n, 1)), "n_animals": n}))
            for a, r in per.iterrows():
                rows.append(pd.DataFrame({"row": "animal", "position": p, "measure": m, "epoch": e, "animal": a,
                                          "x": xs, "v": r.to_numpy(float)}))
            if e == "pre" or sel.sum() < 2:
                continue
            keep = (ep == "pre") | sel
            sub = ch.to_numpy(float)[keep]
            ok = np.isfinite(sub).all(0)
            mask = np.zeros(len(xs), bool)
            if ok.sum() >= 3:
                mask[ok] = cluster_permutation(sub[:, ok], sel[keep], an[keep], n_perm=CLUSTER_PERM)
            rows.append(pd.DataFrame({"row": "sig", "position": p, "measure": m, "epoch": e, "x": xs, "v": mask}))
    return pd.concat(rows, ignore_index=True)


def fig_phase(out: Path, contra: bool = False, plot_only: bool = False) -> pd.DataFrame:
    """POOLED twin of the per-animal lick-phase figures (`pose3d_phase` phase_angle / phase_dist): phase-normalised
    midline angle and 3-D distance from the mouth zero, change from each animal's August pre curve; bold = mean over
    animals (>= 3; band = SEM over animals), faint = each animal; bars at the top = where the epoch differs from pre
    (`phase_data`). Rows share y. Priya 11-03 (deck slide 52): stats, phase 0.1-0.9, light smoothing, shared y per row,
    faint animals -- as the cam4 phase figure. Figure data -> <png stem>_figdata.csv; ``plot_only`` redraws from it."""
    import matplotlib.pyplot as plt
    from wfield_local.sig_marks import reserve_headroom
    stem = "pose3d_phase_pooled_contra" if contra else "pose3d_phase_pooled"
    from wfield_local import figure_layout as FL
    fd = FL.sidecar_for(out, stem, "_figdata.csv", mkdir=not plot_only)   # data/ sidecar; the reader is this function
    R = pd.read_csv(fd) if plot_only else phase_data(contra)
    if not plot_only:
        R.to_csv(fd, index=False)
    M = R[R.row == "mean"]
    fig, axs = plt.subplots(2, 6, figsize=(15, 6.0), sharex=True, sharey="row")
    for i, (m, lab) in enumerate((("mid", f"midline angle, change\n(deg; + = {'contra' if contra else 'ipsi'})"),
                                  ("dist", "3-D distance from mouth\nzero, change (mm)"))):
        for j, p in enumerate(POS_3D):
            ax = axs[i, j]
            ax.axhline(0, color="0.6", lw=0.8, zorder=0)
            ax.axvline(0.5, color="0.85", ls="--", lw=0.8, zorder=0)
            for k, e in enumerate(EPOCHS):
                A = R[(R.row == "animal") & (R.measure == m) & (R.position == p) & (R.epoch == e)]
                for _a, g in A.groupby("animal"):
                    ax.plot(g.x, _smooth(g.v.to_numpy(float)), color=ECOL[e], lw=0.6, alpha=0.3, zorder=1)
                r = M[(M.measure == m) & (M.position == p) & (M.epoch == e)]
                mu = np.where(r.n_animals >= 3, r.v, np.nan)
                mu_s, se_s = _smooth(mu), _smooth(np.where(r.n_animals >= 3, r["sem"], np.nan))
                ax.plot(r.x, mu_s, color=ECOL[e], lw=1.6, zorder=3, label=e)
                ax.fill_between(r.x, mu_s - se_s, mu_s + se_s, color=ECOL[e], alpha=0.2, lw=0, zorder=2)
                S = R[(R.row == "sig") & (R.measure == m) & (R.position == p) & (R.epoch == e)]
                mask = S.v.astype(str).isin(["True", "1", "1.0"]).to_numpy()
                if mask.any():                                   # one bar per significant run of bins, row k
                    xs = S.x.to_numpy(float)
                    dx = np.median(np.diff(xs)) if len(xs) > 1 else 0.025
                    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
                    for s0, s1 in zip(edges[::2], edges[1::2]):
                        ax.plot([xs[s0] - dx / 2, xs[s1 - 1] + dx / 2], [0.975 - 0.045 * k] * 2, color=ECOL[e],
                                lw=2.6, alpha=0.6, transform=ax.get_xaxis_transform(), solid_capstyle="butt", zorder=4)
            ax._wf_mark_rows, ax._wf_mark_step = 3, 0.045        # reserve_headroom: the bars sit above the data
            ax.set_xlim(*PHASE_WIN)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(POS_LABEL[p].replace("_", " "), fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=7.5)
            if i == 1:
                ax.set_xlabel("lick phase (0 = out, 1 = back in)", fontsize=7)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    reserve_headroom(fig)
    fig.suptitle("3-D lick phase, change from each animal's August pre: bold = mean over animals (>= 3, band = SEM), "
                 "faint = each animal; all post-cue licks"
                 + ("; midline angle + = contralesional" if contra else "") +
                 "\nbars = cluster permutation p < .05 vs August-pre sessions (labels permuted within animal; rows: "
                 "acute / subacute / chronic); phase 0.1-0.9, 1-bin display smoothing", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_svg(fig, out / f"{stem}.png", dpi=130)
    plt.close(fig)
    return M


def contra_figs(out: Path) -> int:
    """The 3-D figures signed + = CONTRALESIONAL (Priya 10-30 / 11-03), the page / deck versions: `pose3d_change_pooled`
    with its bootstrap REPLAYED from the same seed and draw order (the angle figure's draws come first), so every
    interval is the + = ipsi one negated (checked against the stored CSV); cue, days, phase; 3-D paths in camera view
    (``*_contra_cam.png``)."""
    rng = np.random.default_rng(20261025)
    fig_angle(out, rng, save=False)                       # replay the draws, so fig_3d's intervals are the same
    B = fig_3d(out, rng, contra=True)
    sg = np.where(B.measure.isin(SIGNED_3D), -1.0, 1.0)
    Bc = B.copy()
    for c in ("PS92", "PS93", "PS94", "PS95", "est"):
        Bc[c] = B[c] * sg
    Bc["lo"], Bc["hi"] = np.where(sg < 0, -B.hi, B.lo), np.where(sg < 0, -B.lo, B.hi)
    Bc.to_csv(out / "pose3d_change_pooled_contra.csv", index=False)
    old = out / "pose3d_change_pooled.csv"
    if old.exists():                                      # the replay must reproduce the stored + = ipsi numbers
        O = pd.read_csv(old)
        dev = np.nanmax(np.abs(O[["est", "lo", "hi"]].to_numpy(float) - B[["est", "lo", "hi"]].to_numpy(float)))
        print(f"replay vs stored pose3d_change_pooled.csv: max |diff| = {dev:.3g}")
    fig_cue(out, contra=True)
    fig_paths3d(out, contra=True)
    fig_paths3d(out, shared=True, contra=True)
    fig_days(out, contra=True)
    fig_phase(out, contra=True)
    print(Bc[Bc.measure.isin(SIGNED_3D)][["measure", "group", "epoch", "PS92", "PS93", "PS94", "PS95", "est", "lo",
                                          "hi"]].round(2).to_string(index=False))
    print("->", out)
    return 0


def main(argv=None) -> int:
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--angle-only", action="store_true", help="only the two cam4 angle figures (+ = ipsi and + = contra)")
    ap.add_argument("--contra", action="store_true",
                    help="ONLY the 3-D figures signed + = CONTRALESIONAL (*_contra.png; 3-D paths *_contra_cam.png, camera "
                         "view); the + = ipsi outputs untouched. A full run writes them too (the default since 11-03)")
    ap.add_argument("--no-contra", action="store_true", help="full run without the + = contra figures")
    ap.add_argument("--phase-only", action="store_true", help="only the 3-D lick-phase figures (+ = ipsi and + = contra)")
    ap.add_argument("--plot-only", action="store_true", help="with --phase-only: redraw from the *_figdata.csv")
    a = ap.parse_args(argv)
    out = a.out or _tc().kin_out_dir(pose3d_dir(), "figures/pooled", "kinematics_pooled_figs")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261025)
    import matplotlib
    matplotlib.use("Agg")
    if a.phase_only:
        for c in (False, True):
            fig_phase(out, contra=c, plot_only=a.plot_only)
        print("->", out)
        return 0
    if a.contra:
        return contra_figs(out)
    A = fig_angle(out, rng)
    A.to_csv(out / "angle_change_pooled.csv", index=False)
    Ac = fig_angle(out, np.random.default_rng(20261025), contra=True)      # re-seeded: same draws, numbers negated
    Ac.to_csv(out / "angle_change_pooled_contra.csv", index=False)
    if a.angle_only:
        print(Ac[["group", "epoch", "PS92", "PS93", "PS94", "PS95", "est", "lo", "hi"]].round(2).to_string(index=False))
        return 0
    B = fig_3d(out, rng)
    B.to_csv(out / "pose3d_change_pooled.csv", index=False)
    sessions3d().to_csv(out / "pose3d_sessions.csv", index=False)     # the gated per-session table read
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
    if not a.no_contra:                                   # Priya 11-03: + = contralesional is the DEFAULT output
        contra_figs(out)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

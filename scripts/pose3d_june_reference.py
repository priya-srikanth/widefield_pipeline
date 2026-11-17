"""June pre-stroke sessions as a REFERENCE for the 3-D kinematics (Priya 11-07: "can we re-evaluate why we can't use the
june sessions' kinematics"; docs/POSE_MODELS.md D13).

    python -m scripts.pose3d_june_reference [--perm 10000] [--replot]

Why a reference and not part of the baseline: the per-session PnP poses every camera in stage mm, so June's 3-D geometry
holds (D13: tongue reprojection 3.8-4.2 px vs August 3.3-4.3; the June nose sits where August's does), but June is an
earlier training stage and its licks land differently (per-animal common shift 0.1-0.8 mm, spout-specific spread 0.3-0.5
mm vs August 0.07 / 0.1-0.17). The AUGUST-ONLY pre baseline stays the baseline (10-18); June is read against it, next to
the post-stroke epochs, so a stroke change can be compared with the June -> August (learning) change of the same animal.

Rows: the same per-session x position medians as the 3-D summary figures (`pose3d_summary_fig`): the camera-working-period
gate (A20; June has camera classes, computed on demand by `trial_classes`), spout-error rows from `pose3d_spout_error`'s
per_session.csv (subset "all"), kinematics rows from `pose3d_summary_fig.gated_sessions`.
Per animal x measure x position: delta of each June / post session from that animal's OWN August pre mean (rule: delta vs
own pre). TEST June vs August: within-animal session-label shuffles (each animal's June and August pre sessions
exchanged, ``--perm`` draws), statistic = mean over animals of (June mean - August mean); two-sided; Holm within each
measure over the six positions. Post epochs are drawn for comparison only (their tests live in the main figures).
``kind``: "frame" = depends on where the head / set point sits in the stage frame (reach, endpoint components, angles
from the mouth zero); "spout" = measured at the spout (miss, lateral error at closest approach); "shape" = frame-free
(speed, jaw opening, peak timing).
Output `session_poses/pose3d/june_reference/`: june_vs_august.csv, sessions_delta.csv, june_reference.png.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

#: measures kept in the tables but not drawn: dt_peak_ms moves in whole 4 ms frames (almost every session median is 0)
FIG_SKIP = ("dt_peak_ms",)
POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
#: (source, column, label, kind)
MEASURES = (("se", "miss_med", "closest approach to spout (mm)", "spout"),
            ("se", "lat_med", "lateral error at closest (mm; + outward)", "spout"),
            ("k", "reach_mm", "reach from mouth zero (mm)", "frame"),
            ("k", "side_ml", "endpoint side (mm)", "frame"),
            ("k", "out_ap", "endpoint out (mm)", "frame"),
            ("k", "down_dv", "endpoint down (mm)", "frame"),
            ("k", "ang3d_err", "3-D angle error (deg)", "frame"),
            ("k", "speed_peak", "peak tongue speed (mm/s)", "shape"),
            ("k", "open_max_mm", "widest jaw opening (mm)", "shape"),
            ("k", "dt_peak_ms", "3-D peak - cam4 peak (ms)", "shape"))
JUNE_CUT = "20260801"
EPOCHS = ("june", "acute", "subacute", "chronic")


def long_table(SE: pd.DataFrame, K: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, measure, position): animal, group (june / aug_pre / acute / subacute / chronic), value."""
    rows = []
    se = SE[SE.subset == "all"]
    for src, col, _lab, _kind in MEASURES:
        if src == "se":
            d = se[["session", "animal", "date", "epoch", "position", col]].rename(columns={col: "v"})
        else:
            cs = [c for c in K.columns if c.startswith(col + "|")]
            d = K.melt(id_vars=["session", "animal", "date", "epoch"], value_vars=cs, var_name="position", value_name="v")
            d["position"] = d.position.str.split("|").str[1]
        d = d.assign(measure=col).dropna(subset=["v"])
        rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    L["date"] = L.date.astype(str)
    L["group"] = np.where(L.epoch != "pre", L.epoch, np.where(L.date < JUNE_CUT, "june", "aug_pre"))
    return L[L.position.isin(POS)]


def june_test(L: pd.DataFrame, n_perm: int, seed: int = 0) -> pd.DataFrame:
    """June vs August pre per measure x position: within-animal session-label shuffles."""
    rng = np.random.default_rng(seed)
    out = []
    for (m, p), g in L[L.group.isin(["june", "aug_pre"])].groupby(["measure", "position"]):
        per = []
        for an, ga in g.groupby("animal"):
            v, j = ga.v.to_numpy(float), (ga.group == "june").to_numpy()
            if j.sum() and (~j).sum() >= 2:
                per.append((an, v, j))
        if len(per) < 3:
            continue
        obs = np.mean([v[j].mean() - v[~j].mean() for _a, v, j in per])
        null = np.zeros(n_perm)
        for _an, v, j in per:
            idx = np.argsort(rng.random((n_perm, len(v))), axis=1)
            jj = np.zeros((n_perm, len(v)), bool)
            np.put_along_axis(jj, idx[:, :j.sum()], True, axis=1)
            vv = np.broadcast_to(v, jj.shape)
            null += (np.where(jj, vv, 0).sum(1) / j.sum()) - (np.where(~jj, vv, 0).sum(1) / (~j).sum())
        null /= len(per)
        pval = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (n_perm + 1)
        aug_sd = np.mean([v[~j].std(ddof=1) for _a, v, j in per])
        out.append({"measure": m, "position": p, "n_animals": len(per), "june_minus_aug": obs,
                    "aug_day_sd": aug_sd, "z_vs_aug_days": obs / aug_sd if aug_sd > 0 else np.nan, "p": pval,
                    "animals_same_sign": int(sum(np.sign(v[j].mean() - v[~j].mean()) == np.sign(obs) for _a, v, j in per))})
    T = pd.DataFrame(out)
    T["p_holm"] = np.nan
    for _m, g in T.groupby("measure"):                    # Holm over the six positions of one measure
        order = g.p.sort_values().index
        k = len(order)
        adj = np.maximum.accumulate([min(1.0, (k - i) * T.at[ix, "p"]) for i, ix in enumerate(order)])
        T.loc[order, "p_holm"] = adj
    kind = {c: kd for _s, c, _l, kd in MEASURES}
    T["kind"] = T.measure.map(kind)
    return T


def deltas(L: pd.DataFrame) -> pd.DataFrame:
    """Each June / post session minus its animal's OWN August pre mean, per measure x position."""
    base = L[L.group == "aug_pre"].groupby(["animal", "measure", "position"]).v.mean().rename("aug")
    D = L[L.group != "aug_pre"].join(base, on=["animal", "measure", "position"])
    return D.assign(delta=D.v - D.aug).dropna(subset=["delta"])


def figure(D: pd.DataFrame, T: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import cat_points as cp
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import EPOCH_LINE
    from wfield_local.figure_layout import save_png_svg
    col = {"june": "0.55", **{e: EPOCH_LINE.get(e, "0.4") for e in EPOCHS[1:]}}
    off = cp.epoch_offsets(len(EPOCHS))
    rows = [mm for mm in MEASURES if mm[1] not in FIG_SKIP]
    fig, axs = plt.subplots(len(rows), 1, figsize=(10, 2.3 * len(rows)), squeeze=False)
    for r, (_s, m, lab, kind) in enumerate(rows):
        ax = axs[r, 0]
        for i, p in enumerate(POS):
            for k, e in enumerate(EPOCHS):
                g = D[(D.measure == m) & (D.position == p) & (D.group == e)]
                if not len(g):
                    continue
                cp.session_dots(ax, i + off[k], g.delta, g.animal, col[e], seed=10 * i + k)
                am = g.groupby("animal").delta.mean()
                cp.summary_point(ax, i + off[k], am.mean(), am.std(ddof=1) / np.sqrt(len(am)) if len(am) > 1 else None,
                                 color=col[e])
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xticks(range(len(POS)))
        ax.set_xticklabels(POS if r == len(rows) - 1 else [])
        ax.set_ylabel(f"{lab}\nvs own Aug pre", fontsize=7)
        ax.set_title(f"[{kind}]", fontsize=7, loc="left")
        ax.spines["top"].set_visible(False)
        t = T[T.measure == m].set_index("position")
        for i, p in enumerate(POS):
            if p in t.index and t.at[p, "p_holm"] < .05:
                sig_marks.star_top(ax, i + off[0], "*", color=col["june"])
    handles = [plt.Line2D([], [], ls="none", marker="o", color=col[e], label=("June pre (reference)" if e == "june" else e))
               for e in EPOCHS]
    fig.suptitle("3-D kinematics: June pre (gray) and post-stroke epochs, each session minus the animal's own August pre "
                 "mean", fontsize=9, y=0.997)
    fig.legend(handles=handles + cp.animal_handles(sorted(D.animal.unique())), fontsize=7, ncol=8, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.988))
    fig.text(0.5, 0.968, "camera-working-period licks / jaw trials (A20); filled point = mean of animal means +- SE "
             "(descriptive); * = June vs August, within-animal session shuffles, Holm over positions", ha="center",
             fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    sig_marks.reserve_headroom(fig)
    save_png_svg(fig, path, dpi=130)
    plt.close(fig)


def _wp_ids(label: str) -> set:
    """Worker: one session's camera-working-period trial ids (`trial_classes.working_period_ids`)."""
    from wfield_local import trial_classes as tc
    return tc.working_period_ids(label)


def _precompute_working_period(sessions, jobs: int = 2) -> None:
    """Compute each session's working-period ids ONCE, 2 workers (this box is RAM-bound), and serve
    `trial_classes.working_period_ids` from them for the rest of the run. `gated_sessions` gates the licks and the jaw
    trials separately, so unmemoised every session's classes were built twice (~12-16 s cold, 3-7 s warm per session:
    ~35 min serial for 119 sessions, 11-08). Same ids, same rule; only the evaluation is shared."""
    from wfield_local import analysis_kit, trial_classes as tc
    res, fail = analysis_kit.fan_sessions(sorted(sessions), _wp_ids, jobs=jobs)
    if fail:
        raise RuntimeError(f"working-period ids failed for {len(fail)} sessions: {fail[:3]}")
    ids = dict(res)
    tc.working_period_ids = ids.__getitem__


def main(argv=None) -> int:
    from scripts.pose3d_summary_fig import gated_sessions
    from wfield_local import trial_classes as tc
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    from wfield_local.pose3d import pose3d_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--replot", action="store_true", help="redraw the figure from the saved CSVs only")
    a = ap.parse_args(argv)
    d3 = pose3d_dir(PathResolver())
    out = tc.kin_out_dir(d3, "june_reference", "pose3d_june_reference")
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    if a.replot:
        figure(pd.read_csv(out / "sessions_delta.csv"), pd.read_csv(out / "june_vs_august.csv"), out / "june_reference.png")
        return 0
    SE = pd.read_csv(tc.kin_out_dir(d3, "spout_error") / "per_session.csv", dtype={"date": str})
    KL = pd.read_csv(d3 / "kinematics" / "licks.csv.gz", dtype={"date": str})
    KJ = pd.read_csv(d3 / "kinematics" / "jaw_trials.csv.gz", dtype={"date": str})
    _precompute_working_period(set(KL.session.astype(str)) | set(KJ.session.astype(str)))
    K = gated_sessions(KL, KJ)
    L = long_table(SE, K)
    n = L.drop_duplicates("session").groupby(["animal", "group"]).size().unstack(fill_value=0)
    print("sessions per animal x group:\n" + n.to_string())
    T = june_test(L, a.perm)
    D = deltas(L)
    T.to_csv(out / "june_vs_august.csv", index=False, float_format="%.4g")
    D.to_csv(out / "sessions_delta.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(T.assign(cell=[f"{d:+.2f} z{z:+.1f} p{p:.3f}{'*' if h < .05 else ''}" for d, z, p, h in
                         zip(T.june_minus_aug, T.z_vs_aug_days, T.p, T.p_holm)])
          .pivot(index="measure", columns="position", values="cell")[list(POS)].to_string())
    figure(D, T, out / "june_reference.png")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

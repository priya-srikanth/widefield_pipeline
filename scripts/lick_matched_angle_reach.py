"""LIKE-FOR-LIKE read-out on ANGLE and 3-D REACH (and, in the table only, elevation and lateral / forward / down).
REFERENCE: GAUSSIAN weighted mean = PRIMARY (Priya 10-27 late "ok gaussian", confirmed); LOCAL-LINEAR with the edge line
carried out = the edge check shown beside it (scope "loclin"). History: first written local-linear only (Priya 10-27 late: "for the angle decoder, should we use gaussian smoother like for like licks
also?" -> local-linear with the edge line carried out ("can we use the same linear strategy for the edges?", "why can't
we decode outside pre reach? isn't that what the local linear is for" -> "sounds good"); plot angle + reach, the rest in
the table ("sagittal, elevation, reach, xyz ... may be too much")). Replaces 'offset beyond the pre pull' (bey_*) of
iti_where_angle_reach.py, whose answer depended on the assumed pull slope.

Per lick: resid = decoded - reference(actual), reference = the animal's PRE licks of the same set (any spout) by a
Gaussian-weighted local line on the actual value (`likeforlike.py`; pre licks leave-one-pre-session-out; width by
held-out pre error). + = read more contralesional (angle) / longer (reach) than pre licks that went to the same place.
Session means of resid per reference (gauss: licks beyond the pre range / weight < 3 get none; loclin: every lick); change vs pre = paired nested
bootstrap (iti_check_figures.epoch_compute); per-animal values first (rule 8). Also: share of licks beyond the pre
range per epoch, and the epoch effect with the reference refit on resampled pre sessions.
Inputs <E>/iti_licks/where_angle_reach_preds.csv.gz (iti_where_angle_reach.py) and where_3axis_preds.csv.gz
(iti_where_3axis.py). Outputs <E>/iti_licks/matched_axes_{sigma_cv,sessions,tests,coverage,refboot}.csv,
figures/lick_matched_angle_reach.png; figdata/lick_matched_angle_reach.pkl (--plot-only redraws from it).
Writes go to il.OUT_W (= OUT unless WFIELD_ITI_OUT_W redirects them, or OUT/itifilter_<f> under a SUBSET ITI filter;
11-17: the CSVs and figure used to go to OUT even then). Under WFIELD_ITI_FILTER=dock / gap2dock / ... the filtered where
caches of iti_dock_where_preds.py are read (il.pred_file); an axis without a filtered cache (3-axis) is skipped.
--curves (11-17): also figures/lick_matched_angle_reach_curves.png, decoded vs executed ANGLE per epoch (all / first licks).
Usage: python scripts/lick_matched_angle_reach.py [--plot-only] [--curves]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import likeforlike as lfl  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

from wfield_local import trial_set_text as tst  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
SETS = {"iti": "ITI", "trial": "cued"}
STEM = "lick_matched_angle_reach"
MIN = 8
DEG = (2.0, 3.0, 5.0, 7.5, 10.0, 15.0)
MM = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
# axis: (preds file, actual, decoded, sigma grid, unit, plotted)
AXES = {"angle": ("where_angle_reach_preds.csv.gz", "act_ang", "dec_ang", DEG, "deg", True),
        "reach": ("where_angle_reach_preds.csv.gz", "act_reach", "dec_reach", MM, "mm", True),
        "elevation": ("where_angle_reach_preds.csv.gz", "act_elev", "dec_elev", DEG, "deg", False),
        "lateral_3axis": ("where_3axis_preds.csv.gz", "act_lat", "dec_lat", MM, "mm", False),
        "forward": ("where_3axis_preds.csv.gz", "act_fwd", "dec_fwd", MM, "mm", False),
        "down": ("where_3axis_preds.csv.gz", "act_down", "dec_down", MM, "mm", False)}
SCOPES = ("gauss", "loclin")      # reference: Gaussian weighted mean (PRIMARY) | local-linear edge check


def _subset_ok(path) -> bool:
    """Under a SUBSET ITI filter (il.ITI_SUBSET) only the filtered where-cache copies under il.OUT_W
    (iti_dock_where_preds.py) may be read; an axis whose cache has no filtered copy (the 3-axis table) is skipped rather
    than scored on every ITI lick. Always True without a subset filter."""
    import iti_where_decoder as wd
    return not il.ITI_SUBSET or wd.subset_in(path) == il.ITI_SUBSET


def compute() -> dict:
    from iti_check_figures import epoch_compute
    cache, cvs, sess, cov, rbs, tests, panels = {}, [], [], [], [], [], {}
    for ax, (fn, x, y, sigmas, unit, shown) in AXES.items():
        if not _subset_ok(il.pred_file(fn)):
            if shown:
                raise SystemExit(f"ITI subset {il.ITI_SUBSET}: no filtered copy of {fn} under {il.OUT_W} "
                                 "(run iti_dock_where_preds.py with the same WFIELD_ITI_FILTER / WFIELD_ITI_OUT_W)")
            print(f"[{ax}] skipped: ITI subset {il.ITI_SUBSET} and no filtered copy of {fn}", flush=True)
            continue
        if fn not in cache:
            cache[fn] = pd.read_csv(il.pred_file(fn))           # basis-specific cache name (il.PRED_TAG)
        P = cache[fn].dropna(subset=[x, y]).reset_index(drop=True)
        sgs = {}
        for scope in SCOPES:
            sgs[scope], cv = lfl.choose(P, x, y, sigmas, {"gauss": "nw", "loclin": "loclin"}[scope])
        cvs.append(cv.assign(axis=ax, chosen_gauss=sgs["gauss"], chosen_loclin=sgs["loclin"]))
        Rg = lfl.ref_table(P, x, y, sgs["gauss"], "nw")
        Rl = lfl.ref_table(P, x, y, sgs["loclin"], "loclin")
        P = P.assign(resid_gauss=P[y] - Rg.ref, resid_loclin=P[y] - Rl.ref, beyond=Rl.beyond)
        print(f"[{ax}] sigma gauss {sgs['gauss']}, loclin {sgs['loclin']} {unit}", flush=True)
        cov.append(lfl.coverage(P, P.beyond.to_numpy()).assign(axis=ax))
        rbs.append(lfl.boot_reference(P, x, y, sgs["loclin"]).assign(axis=ax))
        for scope in SCOPES:
            Q = P.assign(resid=P[f"resid_{scope}"])
            S = Q.groupby(["set", "animal", "label", "epoch"]).agg(resid=("resid", "mean"), n=("resid", "size"))
            S = S[S.n >= MIN].reset_index().assign(axis=ax, scope=scope, unit=unit)
            sess.append(S)
            for st in SETS:
                F = epoch_compute(S[S.set == st], "resid")
                if shown:
                    panels[(st, ax, scope)] = F
                am = S[S.set == st].groupby(["animal", "epoch"]).resid.mean().unstack("epoch")
                for (xx, pt, lo, hi, _txt) in F["post"]:
                    ep = EP[xx - 1]
                    d = (am[ep] - am["pre"]).dropna() if ep in am else pd.Series(dtype=float)
                    tests.append(dict(axis=ax, unit=unit, scope=scope, set=st, epoch=ep, point=pt, lo=lo, hi=hi,
                                      n_dir=int((d * np.sign(pt) > 0).sum()), n=len(d),
                                      **{f"animal_{a}": d.get(a, np.nan) for a in AN}))
    out = {"sigma_cv": pd.concat(cvs), "sessions": pd.concat(sess), "coverage": pd.concat(cov),
           "refboot": pd.concat(rbs), "tests": pd.DataFrame(tests)}
    for k, v in out.items():
        v.to_csv(il.OUT_W / f"matched_axes_{k}.csv", index=False)     # = OUT unless WFIELD_ITI_OUT_W / a subset filter
    pd.set_option("display.width", 250)
    T = out["tests"]
    print("== per animal (rule 8) + nested bootstrap")
    print(T.round(2).to_string(index=False))
    print("== share beyond the pre range")
    print(out["coverage"].round(3).to_string(index=False))
    print("== epoch effect, reference refit on resampled pre sessions")
    print(out["refboot"].round(3).to_string(index=False))
    sig = {a: float(out["sigma_cv"][out["sigma_cv"].axis == a].chosen_gauss.iloc[0]) for a in AXES
           if (out["sigma_cv"].axis == a).any()}           # skipped axes (subset run without a filtered cache) absent
    C = out["coverage"]
    cov_txt = {(r.axis, r.set, r.epoch): r.share_beyond for r in C.itertuples()}
    return {"panels": panels, "sigma": sig, "coverage": cov_txt}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iti_check_figures import epoch_draw
    cols = [(ax, sc) for ax in ("angle", "reach") for sc in SCOPES]
    fig, axs = plt.subplots(2, len(cols), figsize=(3.6 * len(cols), 8.4), squeeze=False)
    for i, st in enumerate(SETS):
        for j, (ax_, sc) in enumerate(cols):
            P = F["panels"].get((st, ax_, sc))
            a = axs[i, j]
            if P is None:
                a.set_axis_off()
                continue
            unit = AXES[ax_][4]
            cv = ", ".join(f"{e[:3]} {F['coverage'].get((ax_, st, e), np.nan):.0%}" for e in EP)
            epoch_draw(a, P, ref=0.0, title=f"{SETS[st]}: {ax_.upper()}, {'GAUSSIAN reference' if sc == 'gauss' else 'local-linear (edge check)'}"
                       + (f"\nbeyond pre range (no Gaussian ref.): {cv}" if sc == "gauss" else ""),
                       ylab=f"read-out vs pre licks at the same {ax_} ({unit})" if j % 2 == 0 else "")
    fig.suptitle("LIKE-FOR-LIKE read-out on ANGLE and 3-D REACH (Gaussian same-place reference; no pull slope): each lick's decoded "
                 "value minus what the animal's PRE licks that went to the same place were read as.\n+ = read more "
                 "contralesional (angle) / longer (reach). Right panel of each pair = LOCAL-LINEAR reference (edge line "
                 "carried beyond the pre range, every lick scored) = edge check.\nGaussian width: " +
                 ", ".join(f"{a} {F['sigma'][a]:g} {AXES[a][4]}" for a in ("angle", "reach")) +
                 " (held-out pre error). Dots = sessions, lines = animals, change vs pre = 95 % paired nested bootstrap, "
                 "k/n animals. PROVISIONAL", fontsize=9)
    fig.tight_layout(rect=(0, tst.footer_bottom(fig), 1, 0.88))     # bottom: the trial-set footer (Priya 11-17)
    save_png_svg(fig, il.OUT_W / "figures" / f"{STEM}.png", dpi=100)
    plt.close(fig)


# --curves (11-17, Priya "re-render docked" of the decoded-vs-executed figure): the ANGLE twin of lick_matched_gauss_curves
# -- decoded vs executed angle, one line per epoch (Gaussian-smoothed with the angle's own Gaussian-reference sigma, mean
# +/- SEM of the animals, solid where all 4 animals contribute, dashed where 3), ITI and cued columns, rows ALL licks and
# FIRST licks (ITI order 0 = bout start, cued order 0 = first post-cue lick; the order of each where-cache row comes from
# the lick_order cache by sequence alignment on the lateral landing, as iti_dock_where_preds.flags). Off by default.
CURVES = "--curves" in sys.argv
ANG_GRID = np.arange(-45.0, 45.01, 1.0)
CURVE_MIN_W = 10.0


def _orders(W: pd.DataFrame) -> np.ndarray:
    """Lick order per row of a where cache (set, label, act_lat; cache order) from the frozen lick_order cache (same
    licks, same lateral landing): per (set, session) the two lateral sequences are aligned (difflib, exact to 1e-4 mm);
    rows without a match get NaN."""
    import difflib

    import iti_lick_order as lo
    L = pd.read_csv(lo.preds_path(), usecols=["set", "label", "act", "order"])
    out = np.full(len(W), np.nan)
    Lg = {k: g for k, g in L.groupby(["set", "label"], sort=False)}
    for key, w in W.groupby(["set", "label"], sort=False):
        if key not in Lg:
            continue
        l_ = Lg[key]
        a = list(np.round(w.act_lat.to_numpy(float), 4))
        b = list(np.round(l_.act.to_numpy(float), 4))
        o = l_.order.to_numpy(float)
        idx = w.index.to_numpy()
        for bl in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
            out[idx[bl.a:bl.a + bl.size]] = o[bl.b:bl.b + bl.size]
    return out


def curves_compute() -> dict:
    fz, fa = il.pred_file("zscore_check/where_preds_z.csv"), il.pred_file(AXES["angle"][0])
    if not (_subset_ok(fz) and _subset_ok(fa)):
        raise SystemExit(f"ITI subset {il.ITI_SUBSET}: filtered where caches missing under {il.OUT_W}")
    Wz = pd.read_csv(fz, usecols=["set", "label", "act_lat", "act_sag"])
    A = pd.read_csv(fa)
    if len(Wz) != len(A) or not np.allclose(Wz.act_sag.to_numpy(float), A.act_sag.to_numpy(float), equal_nan=True):
        raise SystemExit(f"{fa.name} not row-aligned with {fz.name}")
    A["order"] = _orders(Wz)
    print("share of where rows with a lick order, by set:", A.groupby("set").order.apply(lambda s: s.notna().mean())
          .round(4).to_dict(), flush=True)
    A = A.dropna(subset=["act_ang", "dec_ang"]).reset_index(drop=True)
    sg = float(lfl.choose(A, "act_ang", "dec_ang", DEG, "nw")[0])
    D = pd.concat([lfl.smooth_curves(Q, "act_ang", "dec_ang", sg, ANG_GRID, CURVE_MIN_W).assign(cls=cls)
                   for cls, Q in (("all", A), ("first", A[A.order == 0]))], ignore_index=True)
    n = (A.assign(first=A.order == 0).groupby(["set", "epoch"])
         .agg(n_all=("act_ang", "size"), n_first=("first", "sum")).reset_index())
    # licks per (set, cls, epoch, animal) for the dashed-line notes (lfl.dash_notes)
    nl = pd.concat([Q.groupby(["set", "epoch", "animal"]).agg(n_s=("label", "nunique"), n_l=("act_ang", "size"))
                    .reset_index().assign(cls=cls) for cls, Q in (("all", A), ("first", A[A.order == 0]))])
    return {"curves": D, "sigma": sg, "n": n, "nlick": nl}


def curves_plot(C: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    eps = ["pre", *EP]
    D, sg = C["curves"], C["sigma"]
    rowlab = {"all": {"iti": "ITI licks", "trial": "cued licks"},
              "first": {"iti": "ITI FIRST lick (bout start)", "trial": "cued FIRST lick"}}
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
    for i, cls in enumerate(("all", "first")):
        for j, st in enumerate(SETS):
            ax = axs[i, j]
            for ep in eps:
                q = D[(D.cls == cls) & (D.set == st) & (D.epoch == ep)].groupby("mid").y
                mu, se, nn = q.mean(), q.sem(), q.count()
                if mu.empty:
                    continue
                m4 = mu.where(nn >= 4)
                xx = mu.index.to_numpy(float)
                ax.plot(xx, mu.where(nn >= 3), color=col[ep], lw=0.9, ls="--", zorder=3)
                ax.plot(xx, m4, color=col[ep], lw=2.0, zorder=4)
                ax.fill_between(xx, m4 - se, m4 + se, color=col[ep], alpha=0.08, lw=0, zorder=2)
            ax.axhline(0, color="0.7", lw=0.6, ls=":")
            ax.axvline(0, color="0.7", lw=0.6, ls=":")
            nt = C["n"][C["n"].set == st].set_index("epoch")
            ntxt = ", ".join(f"{e[:3]} {int(nt.loc[e, 'n_all' if cls == 'all' else 'n_first'])}" for e in eps
                             if e in nt.index)
            ax.set_title(f"{rowlab[cls][st]}: decoded vs executed ANGLE\n(licks: {ntxt})", fontsize=9)
            ax.set_xlabel("executed angle atan2(lateral, sagittal) (deg, + = contralesional)", fontsize=8)
            ax.set_ylabel("decoded angle (deg, + = contralesional)", fontsize=8)
            ax.legend(handles=[Line2D([], [], color=col[e], lw=2, label=e) for e in eps], fontsize=8, frameon=False,
                      loc="lower right")
            Dc = D[D.cls == cls]
            print(f"== angle curve coverage ({cls}, {st}; deg, weight >= {CURVE_MIN_W:g})\n"
                  + lfl.curve_coverage(Dc[Dc.set == st]).round(1).to_string(index=False), flush=True)
            nlk = C.get("nlick")
            nl = ({(r.epoch, r.animal): (r.n_s, r.n_l) for r in nlk[(nlk.cls == cls) & (nlk.set == st)].itertuples()}
                  if nlk is not None else None)
            notes = lfl.dash_notes(Dc, st, eps, CURVE_MIN_W, unit="deg", nlick=nl, digits=0)
            if notes:          # which animal is missing where a line is mostly dashed (upper left = empty quadrant)
                print("   " + "\n   ".join(notes), flush=True)
                ax.text(0.01, 0.97, "\n".join(notes), transform=ax.transAxes, ha="left", va="top", fontsize=7,
                        color="0.25")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("ANGLE like-for-like curves: decoded lick angle for a given executed angle, before and after the stroke\n"
                 "(frozen where-decoder, angle ridge, z-scored split Allen areas). A curve shifted UP = read more "
                 "contralesional than pre licks with the same executed angle.\n"
                 f"Gaussian-smoothed (sigma {sg:g} deg = the angle's Gaussian-reference width); mean +/- SEM of 4 animals, "
                 "dashed = only 3 animals there. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, tst.footer_bottom(fig), 1, 0.93))     # bottom: the trial-set footer (Priya 11-17)
    save_png_svg(fig, il.OUT_W / "figures" / f"{STEM}_curves.png", dpi=100)
    plt.close(fig)


def trial_set_sentence() -> str:
    """THE lick set of the figure (Priya 11-09): the polar where-decoder licks of iti_where_angle_reach.predictions
    (where-decoder loader, every polar coordinate finite, >= 20 per session, animals with >= 3 pre sessions); the
    GAUSSIAN panels score only licks with a same-place reference."""
    import iti_where_decoder as wd
    import lick_refit_decoders as rd
    return rd.where_sets(20, extra=["finite 3-D angle, reach, elevation and sagittal extension",
                                    "animals with >= 3 pre sessions",
                                    "GAUSSIAN-reference panels: only licks inside the animal's pre range with kernel "
                                    "weight >= 3 (local-linear panels: every lick)"],
                         filt=wd.subset_in(il.pred_file(AXES["angle"][0])))


def main() -> int:
    il.split_only("lick_matched_angle_reach")   # outputs not basis-tagged: split basis only (10-27 late)
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    if plot_only():
        plot(load_fd(STEM))
        if CURVES:
            curves_plot(load_fd(f"{STEM}_curves"))
        return 0
    (il.OUT_W / "figures").mkdir(parents=True, exist_ok=True)
    F = compute()
    save_fd(STEM, F)
    plot(F)
    if CURVES:
        C = curves_compute()
        save_fd(f"{STEM}_curves", C)
        curves_plot(C)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

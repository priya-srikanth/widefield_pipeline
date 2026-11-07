"""Do the frozen where-decoder's post-stroke changes (r loss, like-for-like offsets, MAE) track behavioural recovery?
(Priya 10-26: "does PS94, who had the least complete recovery, show the most persistence of the decreased r and
increased error signal?")

Behaviour (--beh) per session: "hit" = far-contra hit rate / the animal's pre mean (`recovery_timing.series`;
saturates chronically, all animals ~1.0); "angle" = mean tongue angle atan2(lateral, sagittal) (= encoder exec_angle)
of cued far-contra licks and "lateral" = their mean lateral protrusion (side_ml), both as CHANGE vs the pre mean;
"aim" = THE behaviour axis (DECISIONS 2026-10-27 late): the session's weighted signed aim (far contra 3 : far mid 2 :
others 1; `iti_recovery_measure.py`), FIRST post-cue lick by default, as CHANGE vs the animal's pre mean (deg); with env
WFIELD_AIM=all the all-lick sensitivity, outputs tagged "aimall" instead of "aim". The kinematic measures use the same
licks the decoder scores (not independent of r / offsets; read with that in mind).
Decoder measures per session and lick set (ITI / in-trial), from the where-decoder outputs:
  r_lat      decoded vs actual lateral r / the animal's pre mean (`where_decoder.csv`, Allen areas)
  lfl_lat    LIKE-FOR-LIKE lateral offset (replaces bey_lat 'beyond the pre pull', Priya 10-27 late "yes replace beyond
             pull"): session mean of decoded - the animal's pre licks of the set that landed at the same place
             (`likeforlike.offset`; Gaussian reference by default, WFIELD_LFL_REF=loclin), minus the pre mean
  lfl_<ax2>  same on the second axis (sagittal, or outward in the 10-26 outward backup)
  mae_lat, mae_2d   MAE / pre mean (`where_mae_sessions.csv`; includes pull + offsets, no direction)
Two levels:
  BETWEEN animals (n = 4, descriptive only): chronic means per animal, ranked against chronic behaviour.
  WITHIN animal across post sessions: Pearson r of measure vs behaviour, raw and with log(day) regressed out of both
  (the shared recovery trend); mean of animals, within-animal permutation of behaviour (N_PERM).
Figure data (figdata.py convention): <DIR>/figdata/lick_recovery_link{,_pooled,_sessions}_<tag>.pkl (tag = beh + "_z"
if --z) hold everything each figure draws (session table, within-animal r table, the pooled nested-bootstrap CIs, the
per-animal fit lines); `--plot-only` redraws the three figures from those files alone (no permutation, no bootstrap).
Usage: [WFIELD_AIM=all] python scripts/iti_recovery_link.py [--src DIR] [--beh hit|angle|lateral|aim] [--z] [--plot-only]
       (DIR default <E>/iti_licks; output written there)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import textwrap
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_recovery_measure as rm  # noqa: E402
import likeforlike as lfl  # noqa: E402
from figdata import load_fd, save_fd  # noqa: E402
from recovery_timing import series  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["acute", "subacute", "chronic"]
N_PERM = 5000


def _files(src: Path, z: bool) -> tuple[Path, Path]:
    """(per-lick predictions, per-session decoder r) for unscaled or within-session z-scored features."""
    # predictions: basis-specific cache names (il.PRED_TAG); decoder r tables: basis output tag (split = untagged)
    if z:
        return (src / "zscore_check" / il.tagged("where_preds_z.csv", il.PRED_TAG),
                src / "zscore_check" / il.tagged("where_decoder_z.csv"))
    return src / il.tagged("where_preds.csv", il.PRED_TAG), src / il.tagged("where_decoder.csv")


def session_measures(src: Path, z: bool = False) -> tuple[pd.DataFrame, str]:
    fp, fd = _files(src, z)
    p = pd.read_csv(fp)
    ax2 = "sag" if "act_sag" in p.columns else "out"
    p = p.dropna(subset=["act_lat", f"act_{ax2}", "dec_lat", f"dec_{ax2}"]).reset_index(drop=True)
    for ax_ in ("lat", ax2):                      # like-for-like offset per lick (same-place pre reference)
        p[f"lfl_{ax_}"] = lfl.offset(p, f"act_{ax_}", f"dec_{ax_}", lfl.SIGMA_MM, f"{ax_} (mm)")[0]
    parts = []
    for (_st, _a), g in p.groupby(["set", "animal"]):
        g = g.copy()
        g["mae_lat"] = (g.dec_lat - g.act_lat).abs()
        g["mae_2d"] = np.hypot(g.dec_lat - g.act_lat, g[f"dec_{ax2}"] - g[f"act_{ax2}"])
        parts.append(g)
    S = (pd.concat(parts).groupby(["set", "animal", "epoch", "label"])[["lfl_lat", f"lfl_{ax2}", "mae_lat", "mae_2d"]]
         .mean().reset_index())
    D = pd.read_csv(fd)
    D = D[(D.source == "areas") & (D.target == "lateral")][["set", "label", "r"]].rename(columns={"r": "r_lat"})
    S = S.merge(D, on=["set", "label"], how="left")
    meas = ["r_lat", "lfl_lat", f"lfl_{ax2}", "mae_lat", "mae_2d"]
    pre = S[S.epoch == "pre"].groupby(["set", "animal"])[meas].mean()
    for m in meas:
        ref = np.array([pre.loc[(s_, a_), m] if (s_, a_) in pre.index else np.nan for s_, a_ in zip(S.set, S.animal)])
        S[m] = S[m] / ref if m.startswith(("r_", "mae")) else S[m] - ref
    return S, ax2


def _resid(y, d):
    X = np.column_stack([np.ones(len(y)), d])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def within(G: pd.DataFrame, meas: list[str], beh: dict) -> dict:
    out = {}
    for st in ("trial", "iti"):
        for m in meas:
            for kind in ("raw", "detrended"):
                rs = []
                for a in AN:
                    g = G[(G.animal == a) & (G.set == st)].dropna(subset=[m])
                    if len(g) < 5:
                        rs.append(np.nan)
                        continue
                    b, y = beh[a][g.idx.to_numpy()], g[m].to_numpy(float)
                    if kind == "detrended":
                        d = np.log(g.day.to_numpy(float))
                        b, y = _resid(b, d), _resid(y, d)
                    rs.append(float(np.corrcoef(b, y)[0, 1]))
                out[(st, m, kind)] = rs
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(il.OUT))
    ap.add_argument("--beh", choices=("hit", "angle", "lateral", "aim"), default="hit")
    ap.add_argument("--z", action="store_true", help="decoder on within-session Z-SCORED features")
    ap.add_argument("--plot-only", action="store_true", help="redraw the three figures from the figure data alone")
    args = ap.parse_args(argv)
    il.split_only("iti_recovery_link")
    beh_tag = ("aim" if rm.AIM == "first" else "aimall") if args.beh == "aim" else args.beh
    src, tag = Path(args.src), beh_tag + ("_z" if args.z else "")
    fdroot = src / "figdata"
    if args.plot_only:                                    # no permutation, no bootstrap: figdata only
        plot_course(load_fd(f"lick_recovery_link_{tag}", fdroot))
        plot_pooled(load_fd(f"lick_recovery_link_pooled_{tag}", fdroot))
        plot_session_scatter(load_fd(f"lick_recovery_link_sessions_{tag}", fdroot))
        return 0
    S, ax2 = session_measures(src, args.z)
    meas = ["r_lat", "lfl_lat", f"lfl_{ax2}", "mae_lat", "mae_2d"]
    B = series("lick")[["label", "day", "beh"]]
    if args.beh == "aim":                                  # THE behaviour axis (iti_recovery_measure; rule 9)
        k = pd.read_csv(rm.sessions_csv())[["animal", "epoch", "label", "weighted_signed"]].rename(
            columns={"weighted_signed": "k"})
        k["beh"] = k.k - k.animal.map(k[k.epoch == "pre"].groupby("animal").k.mean())
        B = B[["label", "day"]].merge(k[["label", "beh"]], on="label")
    elif args.beh != "hit":
        p = pd.read_csv(_files(src, args.z)[0])
        p = p[(p.set == "trial") & (p.pos == "far_contra")].dropna(subset=["act_lat", "act_sag"])
        p["k"] = np.degrees(np.arctan2(p.act_lat, p.act_sag)) if args.beh == "angle" else p.act_lat
        k = p.groupby(["animal", "epoch", "label"]).k.mean().reset_index()
        # CHANGE vs pre (deg / mm), not a ratio: pre angles differ ~2x between animals (10-22 deg) and acute angles
        # cross 0, so a ratio inflates small-pre animals (Priya 10-27). Within-animal r is unchanged by this.
        k["beh"] = k.k - k.animal.map(k[k.epoch == "pre"].groupby("animal").k.mean())
        B = B[["label", "day"]].merge(k[["label", "beh"]], on="label")
    GA = S.merge(B, on="label", how="inner").dropna(subset=["beh", "day"])   # pre sessions kept for the figures
    G = GA[(GA.epoch != "pre") & (GA.day > 0)].copy()
    pd.set_option("display.width", 220)
    # BETWEEN animals (descriptive)
    bb = G.drop_duplicates("label").groupby(["animal", "epoch"]).beh.mean().unstack()[EP]
    print(f"== behaviour ({_blab(tag)}), per animal\n" + bb.round(2).to_string())
    for st in ("trial", "iti"):
        T = G[G.set == st].groupby(["animal", "epoch"])[meas].mean().unstack("epoch")
        T = T.reindex(columns=EP, level=1)
        print(f"\n== {st}: per animal (r, MAE = / pre; lfl = like-for-like offset, change vs pre, mm)\n" + T.round(2).to_string())
    # WITHIN animal
    sess = {a: sorted(G[G.animal == a].label.unique()) for a in AN}
    beh = {a: G.drop_duplicates("label").set_index("label").beh.reindex(sess[a]).to_numpy(float) for a in AN}
    G["idx"] = [sess[a].index(x) for a, x in zip(G.animal, G.label)]
    obs = within(G, meas, beh)
    rng = np.random.default_rng(0)
    null = {k: [] for k in obs}
    for _ in range(N_PERM):
        s_ = within(G, meas, {a: rng.permutation(v) for a, v in beh.items()})
        for k, v in s_.items():
            null[k].append(np.nanmean(v))
    rows = []
    for (st, m, kind), rs in obs.items():
        mu = np.nanmean(rs)
        p = (np.sum(np.abs(null[(st, m, kind)]) >= abs(mu)) + 1) / (N_PERM + 1)
        rows.append({"set": st, "measure": m, "kind": kind, **dict(zip(AN, np.round(rs, 2))), "mean_r": round(mu, 3),
                     "p_perm": round(p, 4)})
    R = pd.DataFrame(rows)
    print("\n== within-animal r(measure, behaviour / pre) over post sessions\n" + R.to_string(index=False))
    R.to_csv(src / f"recovery_link_within_{tag}.csv", index=False)
    G.to_csv(src / f"recovery_link_sessions_{tag}.csv", index=False)
    GA.to_csv(src / f"recovery_link_sessions_withpre_{tag}.csv", index=False)
    figs = {f"lick_recovery_link_{tag}": (plot_course, dict(G=GA, R=R, tag=tag, z=args.z, src=src)),
            f"lick_recovery_link_pooled_{tag}": (plot_pooled, pooled_data(GA, tag, args.z, src)),
            f"lick_recovery_link_sessions_{tag}": (plot_session_scatter, scatter_data(GA, tag, args.z, src))}
    for stem, (fn, D) in figs.items():
        save_fd(stem, D, fdroot)
        fn(D)
    return 0


def _bname(tag: str) -> str:
    """Short behaviour name for the figure titles."""
    return {"aim": "first-lick weighted aim", "aimall": "all-lick weighted aim"}.get(tag.replace("_z", ""),
                                                                                    "far-contra aim")


def _blab(tag: str) -> str:
    return {"hit": "far-contra hit rate / pre", "angle": "far-contra cued lick angle, change vs pre (deg)",
            "aim": "weighted aim, FIRST cued lick, change vs pre (deg)",
            "aimall": "weighted aim, all cued licks, change vs pre (deg)",
            "lateral": "far-contra cued lateral protrusion, change vs pre (mm)"}[tag.replace("_z", "")]


def _bref(tag: str) -> float:
    """Behaviour's pre reference: 1 for the hit-rate ratio, 0 for the angle / lateral changes."""
    return 1.0 if tag.startswith("hit") else 0.0


def scatter_data(GA: pd.DataFrame, tag: str, z: bool, src: Path) -> dict:
    """Figure data for the session scatter: the session dots (set, animal, epoch, beh, lfl_lat), per (set, animal) the
    least-squares line over post sessions (x, y), per set the pooled post r."""
    pts = GA[["set", "animal", "epoch", "beh", "lfl_lat"]].copy()
    lines, r_all = {}, {}
    for st in ("iti", "trial"):
        g = GA[GA.set == st]
        for a in AN:
            post = g[(g.animal == a) & (g.epoch != "pre")].dropna(subset=["beh", "lfl_lat"])
            if len(post) >= 5:
                b, c = np.polyfit(post.beh, post.lfl_lat, 1)
                xx = np.linspace(post.beh.min(), post.beh.max(), 20)
                lines[(st, a)] = (xx, c + b * xx)
        post = g[g.epoch != "pre"].dropna(subset=["beh", "lfl_lat"])
        r_all[st] = float(np.corrcoef(post.beh, post.lfl_lat)[0, 1])
    return dict(pts=pts, lines=lines, r_all=r_all, tag=tag, z=z, src=src)


def plot_session_scatter(D: dict) -> None:
    """POOLED over animals, one dot per SESSION (pre included): decoder lateral offset vs far-contra aim. EPOCH-COLOUR
    CONVENTION (A13, Priya 11-07; was colour = animal, marker = epoch): OPEN marker, shape = animal, colour = epoch; a
    thin grey line per animal = its least-squares fit over post sessions (a fit, not a path through sessions)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import cat_points as cp
    GA, tag, z, src = D["pts"], D["tag"], D["z"], D["src"]
    blab = _blab(tag)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), squeeze=False)
    for i, st in enumerate(("iti", "trial")):
        ax = axs[0, i]
        g = GA[GA.set == st]
        for a in AN:
            h = g[g.animal == a]
            for ep in ["pre", *EP]:
                q = h[h.epoch == ep]
                ax.plot(q.beh, q.lfl_lat, ls="none", marker=cp.animal_marker(a), ms=5, mfc="none",
                        mec=cp.epoch_color(ep), mew=1.0, alpha=0.85)
            if (st, a) in D["lines"]:
                xx, yy = D["lines"][(st, a)]
                ax.plot(xx, yy, color="0.55", lw=1.0, ls=(0, (4, 2)))
        r_all = D["r_all"][st]
        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        ax.axvline(_bref(tag), color="0.7", lw=0.6, ls=":")
        ax.legend(handles=cp.legend_handles(AN, ["pre", *EP], sessions_label="{a}", epoch_label="{e}"), fontsize=7,
                  frameon=False, ncol=2)
        ax.set_xlabel(blab, fontsize=9)
        ax.set_ylabel("lateral like-for-like offset (same place), mm (change vs pre;\n+ = read contralesional)", fontsize=9)
        ax.set_title(f"{'ITI' if st == 'iti' else 'cued'} licks: all post sessions pooled r {r_all:+.2f} "
                     "(between + within animal)", fontsize=9.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    # three short lines: the first line overflowed the figure once the behaviour name got longer (first-lick aim, 10-28)
    fig.suptitle(("Z-SCORED FEATURES. " if z else "") + "POOLED, one dot per SESSION: FROZEN decoder's lateral "
                 f"like-for-like (same-place) offset\nvs {_bname(tag)}. Open markers: shape = animal, colour = epoch; dashed grey "
                 "lines = per-animal fit over post sessions.\nThe within-animal test with the shared time trend "
                 "removed is in the per-animal recovery-link figure (this pooled r mixes animals and includes the time "
                 "trend). PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_png_svg(fig, src / "figures" / f"lick_recovery_link_sessions_{tag}.png", dpi=100)
    plt.close(fig)


def pooled_data(GA: pd.DataFrame, tag: str, z: bool, src: Path) -> dict:
    """Figure data for the pooled figure: per set the animal x epoch means (am), per (set, measure) the mean of animals
    per epoch and the nested-bootstrap change vs pre per post epoch. Per-animal table printed first (rule 8); writes
    recovery_link_pooled_<tag>.csv."""
    from wfield_local import analysis_kit as ak
    eps = ["pre", *EP]
    rows, sets = [], {}
    for st in ("iti", "trial"):
        g = GA[GA.set == st]
        am = g.groupby(["animal", "epoch"])[["beh", "lfl_lat"]].mean()
        print(f"\n== POOLED {st}: per animal x epoch (beh / pre; lfl_lat change vs pre, mm)\n"
              + am.unstack("epoch").reindex(columns=eps, level=1).round(3).to_string())
        mus, cis = {}, {}
        for val in ("beh", "lfl_lat"):
            prd = {a: h[val].to_numpy() for a, h in g[g.epoch == "pre"].groupby("animal")}
            mus[val] = [am[val].xs(e, level="epoch").mean() for e in eps]
            for e in EP:
                post = {a: h[val].to_numpy() for a, h in g[g.epoch == e].groupby("animal")}
                ci = ak.boot_delta(post, prd, np.random.default_rng(0))
                cis[(val, e)] = (ci.point, ci.lo, ci.hi)
                rows.append(dict(set=st, measure=val, epoch=e, point=ci.point, lo=ci.lo, hi=ci.hi,
                                 n_dir=int(((am[val].xs(e, level="epoch") - am[val].xs("pre", level="epoch")) *
                                            np.sign(ci.point) > 0).sum())))
        sets[st] = dict(am=am, mu=mus, ci=cis)
    pd.DataFrame(rows).to_csv(src / f"recovery_link_pooled_{tag}.csv", index=False)
    return dict(sets=sets, tag=tag, z=z, src=src)


def plot_pooled(D: dict) -> None:
    """Per epoch (pre included): behaviour and lateral offset, animal means + nested-bootstrap CI of the change vs pre,
    and the per-animal epoch trajectory in (behaviour, offset) space."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag, z, src = D["tag"], D["z"], D["src"]
    from wfield_local import cat_points as cp
    from wfield_local.sig_marks import reserve_headroom, star_top
    eps = ["pre", *EP]
    blab = _blab(tag)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), squeeze=False)
    for i, st in enumerate(("iti", "trial")):
        am = D["sets"][st]["am"]
        for j, (val, lab) in enumerate((("beh", blab), ("lfl_lat", "lateral like-for-like offset (same place), mm\n"
                                                                   "(+ = read contralesional; change vs pre)"))):
            ax = axs[i, j]
            # A13 (Priya 11-07): animals = OPEN markers (shape = animal, colour = epoch), no line across epochs;
            # filled dot = mean +- SEM over animals; open diamond = pre mean + the nested-bootstrap change (95% CI)
            for k, a in enumerate(AN):
                for x, e in enumerate(eps):
                    v = am[val].get((a, e), np.nan)
                    if np.isfinite(v):
                        ax.plot([x - 0.18 + 0.05 * k], [v], ls="none", marker=cp.animal_marker(a), ms=5, mfc="none",
                                mec=cp.epoch_color(e), mew=1.0, zorder=3)
            mu = D["sets"][st]["mu"][val]
            for x, e in enumerate(eps):
                cp.epoch_mean_sem(ax, x + 0.05, [am[val].get((a, e), np.nan) for a in AN], e, ms=6)
            for x, e in enumerate(EP, start=1):
                point, lo, hi = D["sets"][st]["ci"][(val, e)]
                ref = mu[0]
                ax.errorbar(x + 0.22, ref + point, yerr=[[point - lo], [hi - point]], fmt="_", mew=2.2,
                            color=cp.epoch_color(e), ms=11, capsize=3, zorder=6)
                sig = lo > 0 or hi < 0
                ax.text(x + 0.22, ref + hi, f"{point:+.2f}", ha="center", va="bottom", fontsize=8,
                        color=cp.epoch_color(e))
                if sig:
                    star_top(ax, x + 0.22, "*", cp.epoch_color(e))
            ax.axhline(_bref(tag) if val == "beh" else 0, color="0.7", lw=0.7, ls=":")
            ax.set_xticks(range(4), eps)
            ax.set_ylabel(lab, fontsize=8.5)
            ax.set_title(f"{'ITI' if st == 'iti' else 'cued'} licks: {'behaviour' if val == 'beh' else 'decoder'}",
                         fontsize=9.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        ax = axs[i, 2]
        for a in AN:                       # per-animal epoch means: open, shape = animal, colour = epoch, no path
            for e in eps:
                xx, yy = am["beh"].get((a, e), np.nan), am["lfl_lat"].get((a, e), np.nan)
                ax.plot([xx], [yy], ls="none", marker=cp.animal_marker(a), ms=6.5, mfc="none",
                        mec=cp.epoch_color(e), mew=1.1)
        for e in eps:                      # epoch mean of animals +- SEM on both axes
            xs_ = np.array([am["beh"].get((a, e), np.nan) for a in AN], float)
            ys_ = np.array([am["lfl_lat"].get((a, e), np.nan) for a in AN], float)
            ok = np.isfinite(xs_) & np.isfinite(ys_)
            if ok.sum():
                sx = xs_[ok].std(ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else 0.0
                cp.epoch_mean_sem(ax, xs_[ok].mean(), ys_[ok], e, xerr=sx)
        ax.axhline(0, color="0.7", lw=0.7, ls=":")
        ax.axvline(_bref(tag), color="0.7", lw=0.7, ls=":")
        ax.set_xlabel(blab, fontsize=8.5)
        ax.set_ylabel("lateral offset (mm, change vs pre)", fontsize=8.5)
        ax.set_title("per animal, epoch means (open) and their mean +- SEM (filled):\naim vs decoder offset",
                     fontsize=9.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.suptitle(("Z-SCORED FEATURES. " if z else "") + f"POOLED per epoch (pre included): {_bname(tag)} and the "
                 "FROZEN pre-stroke decoder's lateral like-for-like offset (same place).\nOpen markers = animals (session "
                 "means per epoch; shape = animal, colour = epoch); filled = mean +- SEM over animals; bars with a dash = "
                 "mean of animals' pre + change vs pre with 95% nested bootstrap CI\n(animals > sessions; * top = CI "
                 "excludes 0). TOP = ITI licks, BOTTOM = "
                 "cued licks. Behaviour is the same in both rows ("
                 + ("cued licks" if tag.startswith("aim") else "cued far-contra licks") + ").", fontsize=9.5)
    reserve_headroom(fig)
    fig.legend(handles=cp.legend_handles(AN, eps, sessions_label="{a}", epoch_label="{e}: mean +- SEM over animals"),
               loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save_png_svg(fig, src / "figures" / f"lick_recovery_link_pooled_{tag}.png", dpi=100)
    plt.close(fig)


def plot_course(D: dict) -> None:
    """Behaviour (/ pre) and the decoder's lateral like-for-like offset (same place) over days, pre-stroke sessions
    included (left of the stroke line; they scatter around 1 / 0 by construction, the reference). D = figure data: G
    (session table incl. pre), R (within-animal r table), tag, z, src.

    EPOCH-COLOUR CONVENTION (PI decision A13, Priya 11-07; before: one panel per animal, black behaviour and red decoder
    lines across days on twin axes): rows = lick set (ITI, cued), columns = behaviour / decoder offset, all animals
    pooled; one OPEN point per session (shape = animal, colour = epoch), no line; filled = epoch mean +- SEM over
    animals. The within-animal r per animal stays in the panel text (rule 8)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import cat_points as cp
    G, R, tag, z, src = D["G"], D["R"], D["tag"], D["z"], D["src"]
    blab = _blab(tag)
    zl = " [decoder on Z-SCORED features]" if z else ""
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.6), sharex=True, squeeze=False)
    for i, st in enumerate(("iti", "trial")):
        g = G[G.set == st]
        r = R[(R.set == st) & (R.measure == "lfl_lat")].set_index("kind")
        per = "  ".join(f"{a} {r.loc['raw', a]:+.2f} ({r.loc['detrended', a]:+.2f})" for a in AN if a in r)
        for j, (col, lab, ref) in enumerate((("beh", blab, _bref(tag)),
                                             ("lfl_lat", "decoded lateral offset, mm\n(+ = contralesional; change vs pre)",
                                              0.0))):
            ax = axs[i, j]
            ax.axhline(ref, color="0.75", lw=0.7, ls=":")
            ax.axvline(0, color="k", lw=0.8, ls="--")
            cp.day_dots(ax, g.day, g[col], g.animal, g.epoch)
            ax.set_ylabel(lab if "\n" in lab else "\n".join(textwrap.wrap(lab, 34)), fontsize=8.5)
            ax.set_title(f"{'ITI' if st == 'iti' else 'cued'} licks: {'behaviour' if col == 'beh' else 'decoder offset'}",
                         fontsize=9.5)
            if col == "lfl_lat":                      # per animal first (rule 8)
                ax.text(0.01, 0.99, f"within-animal r (time trend removed):\n{per}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=7, color="0.3")
            if i == 1:
                ax.set_xlabel("day after stroke", fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    axs[0, 1].legend(handles=cp.legend_handles(AN, ["pre", *EP], sessions_label="{a}", epoch_label="{e}: mean +- SEM"),
                     fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    rr = R[(R.measure == "lfl_lat")].set_index(["set", "kind"])
    fig.suptitle(("Z-SCORED FEATURES. " if z else "") + f"NEURAL vs BEHAVIOUR: as {_bname(tag)} "
                 f"recovers, does the decoder's contralesional reading fade?\nLeft = {blab}; right = FROZEN pre-stroke "
                 "decoder's lateral like-for-like offset (same place)" + zl + "\n(" + il.basis_label() + ", lags 0..0.2 s; "
                 "+ = cortex reads the lick more contralesional than it landed).\nOpen = one point per session (shape = animal, "
                 "colour = epoch; pre = the reference), filled = epoch mean +- SEM over animals. TOP = ITI licks, BOTTOM = "
                 "cued licks.\nMean within-animal r: ITI "
                 f"{rr.loc[('iti', 'raw'), 'mean_r']:+.2f} (time trend removed {rr.loc[('iti', 'detrended'), 'mean_r']:+.2f},"
                 f" p {rr.loc[('iti', 'detrended'), 'p_perm']:.4f}); cued {rr.loc[('trial', 'raw'), 'mean_r']:+.2f} "
                 f"({rr.loc[('trial', 'detrended'), 'mean_r']:+.2f}, p {rr.loc[('trial', 'detrended'), 'p_perm']:.4f}); "
                 "p = within-animal permutation of behaviour.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.88, 0.84))
    save_png_svg(fig, src / "figures" / f"lick_recovery_link_{tag}.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

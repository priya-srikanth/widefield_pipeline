"""Trial-aligned temporal expression of the per-session own-fit columns matched to lick_left / lick_right, and of their
runner-ups, by spout position (Priya 11-16: "look at the expression of the identified lick left and lick right factors
during each spout position trial group ... and the same for runner ups ... absolute not normalized ... are the
identified vs runner-up lick-left factors present on lick-left trials").

The own8 fits store only spatial (A) and spectral (B) loadings, so the time course is recomputed: each session's trial
spectrograms (``out/<key>_trials.npz``, cue -2..+4 s, 25 Hz, engaged trials) are projected onto ALL 12 own columns with
A and B fixed (`cp.project_rows`, exact per-row NNLS, rows / the animal's RMS scale as in the fit), A and B columns
scaled to unit norm first so a column's weight is in data units and columns / sessions are comparable. ABSOLUTE
expression = weight x FR normalizer D (as `face_rhythm_peak.py`). Up to ``--per-pos`` trials per spout (evenly spaced
through the session; lossy bins dropped). Per-session trial means per position for all 12 columns are cached locally
(``<cache>/fr_own_temporal/<key>.npz``), so the pick / runner choice can change without re-reading the share.

Pick / runner-up = `face_rhythm_lick_left_match_check.col_info` (rule v2 pick; runner-up = best-spatial other column).

    python scripts/face_rhythm_own_temporal.py [--per-pos 12] [--jobs 6]
-> ``<readouts>/master_pre_v1/lick_left_match_check/``: own_temporal_lick.png, own_temporal_lick_<animal>.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = Path(os.environ.get("FR_OWN_TEMPORAL_CACHE", Path.home() / "wfield_cache" / "fr_own_temporal"))
POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")     # canonical order (ipsi, mid, contra)
POS_LAB = {"close_L": "close ipsi", "close_center": "close mid", "close_R": "close contra", "far_L": "far ipsi",
           "far_center": "far mid", "far_R": "far contra"}


def pos_style(p: str) -> dict:
    """The cohort palette (`spout_behavior.position_style`: hue = side -- L / ipsilesional blue, center purple, R /
    contralesional crimson; close deeper, far paler; linestyle by side)."""
    from wfield_local.spout_behavior import position_style
    c, _m, ls = position_style(p)
    return dict(color=c, ls=ls)
EPOCHS = ("pre", "acute", "subacute", "chronic")
UNITS = ("lick_left", "lick_right")


def _unit(M: np.ndarray) -> np.ndarray:
    return M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-12)


def _tag(per_pos: int) -> str:
    return "nall" if per_pos <= 0 else f"n{per_pos}"


def session_curves(key: str, per_pos: int = 12) -> str:
    """Worker: per-position trial-mean ABSOLUTE expression (151, 12) of every own8 column -> cache npz."""
    from wfield_local.face_rhythm import common as frc
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.face_rhythm.factors.cp import normaliser, project_rows, _scaled
    from wfield_local.face_rhythm.factors.files import load_master, load_plan, npz_memmap, npz_small, out_dir
    dst = CACHE / f"{key}_{_tag(per_pos)}.npz"
    if dst.exists():
        return str(dst)
    sd = frc.default_set_dir(set_name=fro.SET)
    _A, _B, scales, M = load_master(sd)
    with np.load(sd / "own8" / f"{key}.npz") as z:
        A, B, animal = _unit(z["A_own"].astype(np.float64)), _unit(z["B_own"].astype(np.float64)), str(z["animal"])
    s = float(scales[animal])
    pr = load_plan(sd)["params"]
    freqs = M["freqs"].astype(np.float64)
    dfn = lambda x: normaliser(x * s, freqs, float(pr.get("vqt_norm", 0.95)), float(pr.get("one_over_f", 0.5)))  # noqa: E731
    f = out_dir(sd) / f"{key}_trials.npz"
    S = npz_memmap(f, "spec")
    meta = npz_small(f, ("t_ms", "position"))
    with np.load(sd / "proj" / f"{key}.npz") as z:
        lossy = z["trials_lossy"].astype(bool) if "trials_lossy" in z.files else np.zeros(S.shape[:2], bool)
    pos = meta["position"].astype(str)
    out = {"t_ms": meta["t_ms"]}
    for p in POS:
        idx = np.nonzero(pos == p)[0]
        if 0 < per_pos < len(idx):
            idx = idx[np.round(np.linspace(0, len(idx) - 1, per_pos)).astype(int)]
        cur = []
        for ti in idx:
            if lossy[ti].all():
                continue
            W, _E, _N2, D = project_rows(_scaled(np.asarray(S[ti:ti + 1][0]), s), A, B, row_fn=dfn)
            X = W.astype(np.float64) * D[:, None]
            X[lossy[ti]] = np.nan
            cur.append(X)
        out[f"{p}_mean"] = np.nanmean(cur, 0) if cur else np.full((S.shape[1], A.shape[1]), np.nan)
        out[f"{p}_n"] = np.array(len(cur))
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.npz")
    np.savez(tmp, **out)
    os.replace(tmp, dst)
    return str(dst)


def _session_cols(fdir, keys):
    """(key -> {unit: (pick col, runner col, rec, flagged)}) under rule v2."""
    import face_rhythm_epoch_shape as es
    import face_rhythm_lick_left_match_check as mc
    from wfield_local import face_rhythm_readouts as frr
    M, A_ref, B_ref, names = es.reference(fdir)
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    out = {}
    for key in keys:
        with np.load(fdir / "own8" / f"{key}.npz") as z:
            A, B = z["A_own"].astype(float), z["B_own"].astype(float)
        out[key] = {u: (lambda d: (d["pick"]["col"], d["runner"]["col"], d["rec"], mc.is_bad(d)))(
            mc.col_info(A_ref, B_ref, A, B, names.index(u), home, mid)) for u in UNITS}
    return out




ROWS = (("lick_left", "pick"), ("lick_left", "runner"), ("lick_right", "pick"), ("lick_right", "runner"))
ROW_LAB = {"pick": "matched", "runner": "runner-up"}
RESP = (0.0, 2.0)          # readout window, s from cue
BAND_ALPHA = 0.15


def animal_curves(T, curves, unit, which, epoch, p):
    """(animals, time) array: per animal, the mean over its sessions (unit recovered) of column ``which``'s trial-mean
    expression on spout ``p`` trials."""
    sel = T[(T.epoch == epoch) & T[f"{unit}_rec"]]
    out = []
    for _an, g in sel.groupby("animal"):
        cs = [curves[k][f"{p}_mean"][:, col] for k, col in zip(g.key, g[f"{unit}_{which}"])
              if int(curves[k][f"{p}_n"]) > 0]
        if cs:
            out.append(np.nanmean(cs, 0))
    return np.array(out)


def _band(ax, t, Y, color, ls="-", lw=1.1, label=None):
    """Mean over animals +- SEM across animals (band). Returns the top of the drawn band."""
    if not len(Y):
        return 0.0
    y = np.nanmean(Y, 0)
    ax.plot(t, y, color=color, ls=ls, lw=lw, label=label)
    if len(Y) > 1:
        se = np.nanstd(Y, 0, ddof=1) / np.sqrt(len(Y))
        ax.fill_between(t, y - se, y + se, color=color, alpha=BAND_ALPHA, lw=0)
        return float(np.nanmax(y + se))
    return float(np.nanmax(y))


def _tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axvline(0, color="0.6", lw=0.5)


def fig_by_epoch(T, curves, t, title, out_png, methods):
    """Rows: matched / runner-up lick_left, lick_right; columns: epochs; lines: spouts (cohort palette)."""
    from wfield_local import face_rhythm_figures as frf
    with frf.style():
        fig, axs = frf.subplots(len(ROWS), len(EPOCHS), figsize=(2.7 * len(EPOCHS), 1.95 * len(ROWS)), squeeze=False,
                                sharex=True)
        for r_i, (u, w) in enumerate(ROWS):
            top = 0.0
            for c_i, ep in enumerate(EPOCHS):
                ax = axs[r_i, c_i]
                n_s = int((T[(T.epoch == ep) & T[f"{u}_rec"]]).shape[0])
                for p in POS:
                    st = pos_style(p)
                    top = max(top, _band(ax, t, animal_curves(T, curves, u, w, ep, p), st["color"], st["ls"],
                                         label=POS_LAB[p]))
                _tidy(ax)
                if r_i == 0:
                    ax.set_title(ep, fontsize=9)
                ax.text(0.98, 0.97, f"{n_s} sess", transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
                        color="0.4")
                if c_i == 0:
                    ax.set_ylabel(f"{u}\n{ROW_LAB[w]}\n(absolute)", fontsize=7.5)
                if r_i == len(ROWS) - 1:
                    ax.set_xlabel("time from cue (s)")
            for ax in axs[r_i]:
                ax.set_ylim(0, top * 1.05 if top > 0 else 1)
        axs[0, -1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(title, fontsize=10)
        frf.save(fig, out_png, methods)


def fig_by_position(T, curves, t, title, out_png, methods):
    """Rows as `fig_by_epoch`; columns: spouts (canonical order); lines: epochs."""
    from wfield_local import face_rhythm_figures as frf
    from wfield_local.face_rhythm import common as frc
    ecol = frc.epoch_colors()
    with frf.style():
        fig, axs = frf.subplots(len(ROWS), len(POS), figsize=(2.3 * len(POS), 1.95 * len(ROWS)), squeeze=False,
                                sharex=True)
        for r_i, (u, w) in enumerate(ROWS):
            top = 0.0
            for c_i, p in enumerate(POS):
                ax = axs[r_i, c_i]
                for ep in EPOCHS:
                    top = max(top, _band(ax, t, animal_curves(T, curves, u, w, ep, p), ecol[ep], label=ep))
                _tidy(ax)
                if r_i == 0:
                    ax.set_title(POS_LAB[p], fontsize=9, color=pos_style(p)["color"])
                if c_i == 0:
                    ax.set_ylabel(f"{u}\n{ROW_LAB[w]}\n(absolute)", fontsize=7.5)
                if r_i == len(ROWS) - 1:
                    ax.set_xlabel("time from cue (s)")
            for ax in axs[r_i]:
                ax.set_ylim(0, top * 1.05 if top > 0 else 1)
        axs[0, -1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(title, fontsize=10)
        frf.save(fig, out_png, methods)


def animal_curves_norm(T, curves, unit, which, epoch, p, floor=1.0, sessions=False):
    """(animals, time): per animal, the mean over its sessions of log2(session trace) minus the mean log2 trace of the
    animal's PRE sessions, per time bin. PRE sessions are each scored against the mean of the animal's OTHER pre sessions
    (leave-one-out), which averages to exactly 0 per animal and keeps the real session-to-session spread (animals with
    one pre session give no pre curve)."""
    out = []
    rec = T[T[f"{unit}_rec"]]
    for _an, g in rec.groupby("animal"):
        def trace(r):
            c = curves[r.key]
            return c[f"{p}_mean"][:, getattr(r, f"{unit}_{which}")] if int(c[f"{p}_n"]) > 0 else None
        pre = [(r.key, trace(r)) for r in g[g.epoch == "pre"].itertuples()]
        pre = [(k, x) for k, x in pre if x is not None]
        if not pre:
            continue
        lg = lambda a: np.log2(np.maximum(a, floor))  # noqa: E731
        if epoch == "pre":
            if len(pre) < 2:
                continue
            cs = [lg(x) - np.nanmean([lg(y) for k2, y in pre if k2 != k], 0) for k, x in pre]
        else:
            m = np.nanmean([lg(x) for _k, x in pre], 0)
            cs = [lg(x) - m for x in (trace(r) for r in g[g.epoch == epoch].itertuples()) if x is not None]
        if cs:
            out.extend(cs) if sessions else out.append(np.nanmean(cs, 0))
    return np.array(out)


def fig_by_position_norm(T, curves, t, title, out_png, methods):
    """As `fig_by_position`, each animal normalized to its own pre trace (log2), pre = leave-one-out."""
    from wfield_local import face_rhythm_figures as frf
    from wfield_local.face_rhythm import common as frc
    ecol = frc.epoch_colors()
    with frf.style():
        fig, axs = frf.subplots(len(ROWS), len(POS), figsize=(2.3 * len(POS), 1.95 * len(ROWS)), squeeze=False,
                                sharex=True, sharey="row")
        for r_i, (u, w) in enumerate(ROWS):
            for c_i, p in enumerate(POS):
                ax = axs[r_i, c_i]
                ax.axhline(0, color="0.5", lw=0.5)
                for ep in EPOCHS:
                    Y = animal_curves_norm(T, curves, u, w, ep, p, sessions=(ep == "pre"))
                    if len(Y):
                        # pre: the animal-weighted mean of leave-one-out values is 0 by construction; band from sessions
                        y = np.zeros(len(t)) if ep == "pre" else np.nanmean(Y, 0)
                        ax.plot(t, y, color=ecol[ep], lw=1.1, label=ep)
                        if len(Y) > 1:
                            se = np.nanstd(Y, 0, ddof=1) / np.sqrt(len(Y))
                            ax.fill_between(t, y - se, y + se, color=ecol[ep], alpha=BAND_ALPHA, lw=0)
                _tidy(ax)
                if r_i == 0:
                    ax.set_title(POS_LAB[p], fontsize=9, color=pos_style(p)["color"])
                if c_i == 0:
                    ax.set_ylabel(f"{u}\n{ROW_LAB[w]}\nlog2(vs own pre)", fontsize=7.5)
                if r_i == len(ROWS) - 1:
                    ax.set_xlabel("time from cue (s)")
        axs[0, -1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.suptitle(title, fontsize=10)
        frf.save(fig, out_png, methods)


def readouts(T, curves, t):
    """One row per session (split videos merged by label) x unit x column x spout: response-window mean (cue 0..+2 s)
    of the absolute expression, and log2 vs the same animal's pre sessions."""
    import pandas as pd

    from wfield_local.face_rhythm import stats as frs
    w = (t >= RESP[0]) & (t <= RESP[1])
    rows = []
    for r in T.itertuples():
        c = curves[r.key]
        for u, wh in ROWS:
            if not getattr(r, f"{u}_rec"):
                continue
            col = getattr(r, f"{u}_{wh}")
            for p in POS:
                if int(c[f"{p}_n"]) > 0:
                    rows.append(dict(label=r.label, key=r.key, animal=r.animal, epoch=r.epoch, unit=u, which=wh, pos=p,
                                     flag=bool(getattr(r, f"{u}_flag")), n_trials=int(c[f"{p}_n"]),
                                     resp=float(np.nanmean(c[f"{p}_mean"][w, col])),
                                     base=float(np.nanmean(c[f"{p}_mean"][t < 0, col]))))
    R = pd.DataFrame(rows)
    R = (R.groupby(["label", "animal", "epoch", "unit", "which", "pos"], as_index=False)
         .agg(resp=("resp", "mean"), base=("base", "mean"), n_trials=("n_trials", "sum"), flag=("flag", "max")))
    R["log2_vs_pre"] = frs.log2_vs_pre(R, "resp", ["animal", "unit", "which", "pos"])
    return R


def stats_table(R):
    import pandas as pd

    from wfield_local.face_rhythm import stats as frs
    rng = np.random.default_rng(0)
    rows = []
    for u, wh in ROWS:                       # fixed order: the seeded draws reproduce
        for p in POS:
            d = R[(R.unit == u) & (R.which == wh) & (R.pos == p)]
            rows += [dict(unit=u, which=wh, pos=p, **r) for r in frs.epoch_change(d, "log2_vs_pre", rng)]
    return frs.add_holm(pd.DataFrame(rows), ["unit", "which"])


def fig_stats(R, ST, title, out_png, methods):
    """Rows as `fig_by_epoch`; columns: spouts; epochs on x: session dots, mean +- SEM over animals, stars on top."""
    import face_rhythm_peak as frp

    from wfield_local import cat_points as cp
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import sig_marks
    from wfield_local.face_rhythm import stats as frs
    with frf.style():
        fig, axs = frf.subplots(len(ROWS), len(POS), figsize=(2.35 * len(POS), 2.7 * len(ROWS)), squeeze=False,
                                sharey="row")
        for r_i, (u, wh) in enumerate(ROWS):
            for c_i, p in enumerate(POS):
                ax = axs[r_i, c_i]
                d = R[(R.unit == u) & (R.which == wh) & (R.pos == p)]
                frp.epoch_points(ax, d, "log2_vs_pre", seed=100 * r_i + 10 * c_i)
                ch = {r["epoch"]: r for r in ST[(ST.unit == u) & (ST.which == wh) & (ST.pos == p)].to_dict("records")}
                for e_i, ep in enumerate(EPOCHS[1:], start=1):
                    if ep in ch and frs.mark(ch[ep]):
                        sig_marks.star_top(ax, e_i, frs.mark(ch[ep]))
                ax.axhline(0, color="k", lw=0.5)
                ax.set_xticks(range(len(EPOCHS)), ["pre", *(frp.change_tick(ep, ch.get(ep), interval=False) for ep in EPOCHS[1:])],
                              fontsize=6.5)
                if r_i == 0:
                    ax.set_title(POS_LAB[p], fontsize=9, color=pos_style(p)["color"])
                if c_i == 0:
                    ax.set_ylabel(f"{u} {ROW_LAB[wh]}\nlog2(resp / own pre)", fontsize=7.5)
        cp.add_animal_entries(axs[0, -1], sorted(R.animal.unique()))
        axs[0, -1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        sig_marks.reserve_headroom(fig)
        fig.suptitle(title, fontsize=10)
        frf.save(fig, out_png, methods)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    import face_rhythm_epoch_shape as es
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    from wfield_local.face_rhythm import common as frc
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-pos", type=int, default=0, help="trials per spout per session; 0 = all trials")
    ap.add_argument("--cache-only", action="store_true", help="fill the per-session cache, no figures")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    es.RULE = "v2"
    fdir = frc.default_set_dir(set_name=fro.SET)
    out = frc.readouts_dir(fro.SET) / "lick_left_match_check"
    writeguard.assert_writable(out)
    S = frc.sessions(fdir, "own8")
    keys = list(S.key)
    from functools import partial
    _res, fails = ak.fan_sessions(keys, partial(session_curves, per_pos=a.per_pos), jobs=a.jobs)
    if fails:
        print("FAILED:", fails)
        return 1
    if a.cache_only:
        return 0
    curves = {k: dict(np.load(CACHE / f"{k}_{_tag(a.per_pos)}.npz")) for k in keys}
    t = curves[keys[0]]["t_ms"] / 1000.0
    cols = _session_cols(fdir, keys)
    T = S[["key", "label", "animal", "date", "epoch"]].copy()
    for u in UNITS:
        for i, f in enumerate(("pick", "runner", "rec", "flag")):
            T[f"{u}_{f}"] = [cols[k][u][i] for k in keys]
    ntr = "all trials" if a.per_pos <= 0 else f"up to {a.per_pos} trials per spout (evenly spaced)"
    base = f"""
ABSOLUTE expression (temporal weight x FR normalizer D, not normalized) of the per-session own-fit (own8) column that
rule v2 matches to the master lick_left / lick_right ("matched") and of its runner-up (the other column with the
highest spatial cosine to the master). Each session's trial spectrograms (engaged trials, cue -2..+4 s, 25 Hz) are
projected onto all 12 own columns with spatial and spectral loadings fixed and scaled to unit norm, so amplitudes are
comparable across columns and sessions; {ntr} per session; lossy bins dropped. Only sessions where that unit is
recovered (spatial cos >= .7, spectral >= .5). Spout colors = the cohort palette (blue = left = ipsilesional, purple =
middle, crimson = right = contralesional; close deeper, far paler). Script: scripts/face_rhythm_own_temporal.py."""
    traces = base + """ Lines: mean over animals of each animal's mean over its sessions; band = +- SEM across
animals (none where one animal). y axis shared within a row. Descriptive."""
    sfx = "" if a.per_pos <= 0 else f"_n{a.per_pos}"
    fig_by_epoch(T, curves, t, "Per-session fits: matched vs runner-up lick columns by spout (all animals)",
                 out / f"own_temporal_lick{sfx}.png", traces + " Columns = epochs, lines = spouts.")
    fig_by_position(T, curves, t, "Per-session fits: matched vs runner-up lick columns by epoch, per spout",
                    out / f"own_temporal_lick_by_position{sfx}.png", traces + " Columns = spouts, lines = epochs.")
    fig_by_position_norm(T, curves, t, "Per-session fits: lick columns by epoch, per spout, vs each animal's own pre",
                         out / f"own_temporal_lick_by_position_norm{sfx}.png", base + """ Each session's log2 trace minus the
mean log2 trace of the same animal's pre sessions, per time bin (values floored at 1 before the log); PRE sessions each
against the mean of the animal's OTHER pre sessions (leave-one-out), so pre averages exactly 0 per animal and the band
shows its real session-to-session spread. Post epochs: per animal the mean over its sessions; line = mean over animals, band = +- SEM across animals.
PRE: line = 0 by construction (mean over animals of each animal's mean leave-one-out value); band = +- SEM across all
pre sessions' leave-one-out traces (pre-session variability; across animals it would be 0). Columns =
spouts, lines = epochs; y shared within a row. Descriptive (tests: own_temporal_lick_stats).""")
    for an in sorted(T.animal.unique()):
        fig_by_epoch(T[T.animal == an], curves, t, f"{an}: matched vs runner-up lick columns by spout",
                     out / f"own_temporal_lick_{an}{sfx}.png",
                     traces.replace("mean over animals of each animal's mean over its sessions; band = +- SEM across\n"
                                    "animals (none where one animal)", "mean over this animal's sessions (no band)"))
    R = readouts(T, curves, t)
    ST = stats_table(R)
    R.to_csv(figure_layout.sidecar_for(out, f"own_temporal_lick{sfx}", ".csv"), index=False, float_format="%.5g")
    ST.to_csv(figure_layout.sidecar_for(out, f"own_temporal_lick_stats{sfx}", ".csv"), index=False,
              float_format="%.5g")
    fig_stats(R, ST, "Per-session fits: lick-column response by spout and epoch, change vs own pre",
              out / f"own_temporal_lick_stats{sfx}.png", base + f"""
Readout: per session, the mean absolute expression in the response window (cue {RESP[0]:g} to +{RESP[1]:g} s) on that
spout's trials, as log2 minus the mean log2 of the same animal's pre sessions (pre = 0, -1 = halved). Faded open
markers = sessions (shape = animal, color = epoch); filled = mean +- SEM across animals of each animal's mean. Under
each epoch: change vs pre, `analysis_kit.boot_delta` (paired animals -> sessions bootstrap, animal-weighted); 95 %
intervals in data/own_temporal_lick_stats.csv. Stars (top band): * = interval excludes 0; ** = also Holm over the 18 tests of a row (6 spouts x 3
epochs).""")
    pd.set_option("display.width", 220)
    print(ST[["unit", "which", "pos", "epoch", "delta", "ci_lo", "ci_hi", "p_holm", "mark"]].round(3)
          .to_string(index=False))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

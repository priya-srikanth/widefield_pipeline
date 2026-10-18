"""Trial-aligned spatial occupancy heatmaps of the cam4 keypoints, pre vs post stroke (Priya 2026-10-18: "as in
stroke-orofacial", raw and delta).

    python -m wfield_local.orofacial_heatmaps [--run lp_r4_heatmap_20261013__clean_v36] [--jobs 6] [--baseline late]

Port of stroke_orofacial's tone-aligned proportion / proportion-diff heatmaps (`dlc_spatial.render_on_demand`,
`stats.spatial_proportion_diff`, Q-lock 12), on our rig:

  frames   every frame of every ENGAGED trial (the cohort's reference-position gate) from the cue to the DAQ trial stop
           (Priya 10-18; theirs was tone -500 .. +5000 ms), cut at spout-retraction onset (`retraction_cut`; only bites
           on PS92 / PS93 0606, where the spout starts retracting up to 300 ms before the DAQ stop).
  frame    the MOUTH / SPOUT frame: offset from the session's mouth (`tongue_origin`), rotated onto the session's spout
           axis (`tongue_kinematics.spout_coords`): x = lateral, + = IPSI (all lesions are LEFT = image right);
           y = forward, out of the mouth toward the spouts. The rotation also removes camera-angle drift. tongue = CLEANED
           tip, OUT frames only (>= 20 px; in-mouth frames are counted in the denominator, not placed at the origin);
           jaw = CLEANED jaw (rest-relative, rotated only); nose, spout = pose, likelihood >= 0.6. The SPOUT is the
           drift CONTROL (fixed apparatus); the NOSE is behaviour (head / snout posture).
  P        per session x position x keypoint: counts per bin / ALL frames in the window (their Q17 denominator), so a
           keypoint that is out less often (fewer licks) loses mass rather than being renormalised away.
  raw      within animal first (mean over the animal's sessions in the epoch), then the mean over animals; log colour.
  delta    per session P - the animal's PRE pool at that position (mean of its pre sessions; a pre session against the
           OTHER pre sessions = the noise floor, as in W2), averaged within animal, then over animals. Blue = less
           occupancy than pre, red = more (their cmap). White = no data in either.
  stats    none per pixel; the distributional test is W2 (`orofacial_w2`), which uses the same frames and gate.

Figures -> <LP_analysis>/<run>/cohort/figures/heatmaps{,_latepre}/ (+ svg/); histograms -> cohort/tables/heatmaps*/.
  nose_by_epoch.png                         nose, all positions pooled: row 1 raw, row 2 delta; columns = epochs
  <kp>_by_position_{raw,delta}.png          kp in tongue / jaw / nose / spout; rows = positions, columns = epochs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYPOINTS = ("tongue", "jaw", "nose", "spout")
LIK_MIN = 0.6
TONGUE_OUT_PX = 20.0
WIN_MS = (0.0, 3500.0)             # cue .. DAQ trial stop (Priya 10-18: "just use cue through trial stop")
NBINS = 80
#: (x = lateral, y = forward) extents in px, fixed across sessions / epochs / positions (from the pilot, 10-18)
EXTENT = {"tongue": ((-175.0, 175.0), (-25.0, 320.0)),
          "jaw": ((-70.0, 70.0), (-25.0, 140.0)),
          "nose": ((-65.0, 50.0), (-135.0, -30.0)),
          "spout": ((-125.0, 125.0), (20.0, 140.0))}
#: RETRACTION CUT (10-18): a trial's window ends at the first frame where the spout sits > RETRACT_PX from its own median
#: over cue .. stop for >= RETRACT_RUN consecutive frames AND stays there (median of the rest > RETRACT_PX; brief label
#: jumps come back). Only on PS92 / PS93 0606 (first ~15 / last ~30 trials) does the
#: spout start retracting before the DAQ stop (up to 300 ms); elsewhere it moves only after the stop. Lick pushes (16-25
#: px), single-frame label jumps and water on an untouched spout are NOT cut (an earlier 15 px whole-trial gate removed
#: those and was retired: it preferentially dropped licking trials). Reference = the trial's own window, no ENL ruler.
RETRACT_PX = 30.0
RETRACT_RUN = 3
LOG_VMIN = 1e-5
SMOOTH_BINS = 1.0                  # Gaussian sigma (bins) for DELTA maps and the agreement mask only


def _edges(kp: str):
    (x0, x1), (y0, y1) = EXTENT[kp]
    return np.linspace(x0, x1, NBINS + 1), np.linspace(y0, y1, NBINS + 1)


def frame_coords(sd: Path, animal: str, date: str, run: str) -> dict:
    """{kp: (x_lat_ipsi, y_fwd, ok)} per frame, plus "t", "stop", "pos", "tid" -- one session folder."""
    from wfield_local import orofacial_outputs as oo
    from wfield_local import tongue_kinematics as tk
    from wfield_local.paths import PathResolver
    S = oo.load_session(PathResolver(), run, animal, date)
    ix, pose, cl = S["index"], S["pose"], S["clean"]
    c = np.load(sd / "clean.npz")
    o = c["tongue_origin"].astype(float)
    fr = tk.SpoutFrame(tuple(o), tuple(c["spout_ap_axis"]))
    sgn = 1.0                                      # spout_coords lr is + = image right = the mouse's LEFT = ipsi (left lesions)
    out = {"t": ix.t_ms.to_numpy(float), "stop": ix.stop_ms.to_numpy(float), "pos": ix.position.to_numpy(str),
           "tid": ix.trial_id.to_numpy(int),
           "spout_raw": (pose.spout_x.to_numpy(float), pose.spout_y.to_numpy(float)),
           "spout_lk": pose.spout_likelihood.to_numpy(float)}
    tx, ty = cl.tongue_x.to_numpy(float), cl.tongue_y.to_numpy(float)
    src = {"tongue": (tx + o[0], ty + o[1], np.hypot(tx, ty) >= TONGUE_OUT_PX),
           "jaw": (cl.jaw_x.to_numpy(float) + o[0], cl.jaw_y.to_numpy(float) + o[1], np.ones(len(tx), bool))}
    for kp in ("nose", "spout"):
        src[kp] = (pose[f"{kp}_x"].to_numpy(float), pose[f"{kp}_y"].to_numpy(float),
                   pose[f"{kp}_likelihood"].to_numpy(float) >= LIK_MIN)
    for kp, (x, y, ok) in src.items():
        ap, lr = tk.spout_coords(x, y, fr)
        out[kp] = (sgn * lr, ap, ok & np.isfinite(ap) & np.isfinite(lr))
    return out


def retraction_cut(win: np.ndarray, tid: np.ndarray, spout_xy: tuple, lk: np.ndarray) -> np.ndarray:
    """`win` with each trial's frames from spout-retraction onset on removed (see RETRACT_*). Frames are in time
    order within a trial (windows.npz rows)."""
    x, y = spout_xy
    out = win.copy()
    ok = win & (lk >= LIK_MIN) & np.isfinite(x) & np.isfinite(y)
    for k in np.unique(tid[win]):
        idx = np.flatnonzero(win & (tid == k))
        good = idx[ok[idx]]
        if len(good) < RETRACT_RUN:
            continue
        mx, my = np.median(x[good]), np.median(y[good])
        far = np.zeros(len(idx), bool)
        far[ok[idx]] = np.hypot(x[good] - mx, y[good] - my) > RETRACT_PX
        dev = np.full(len(idx), np.nan)
        dev[ok[idx]] = np.hypot(x[good] - mx, y[good] - my)
        run = np.convolve(far.astype(int), np.ones(RETRACT_RUN, int), "valid") == RETRACT_RUN
        for j in np.flatnonzero(run):          # a retraction does not come back: the rest of the window stays far
            if np.nanmedian(dev[j:]) > RETRACT_PX:
                out[idx[j:]] = False
                break
    return out


def session_hist(sd: Path, animal: str, date: str, run: str, keep: set | None) -> dict:
    """{(kp, position): (counts (NBINS, NBINS) [x, y], n_window_frames)}; position "all" pools the six."""
    F = frame_coords(sd, animal, date, run)
    t, stop = F["t"], F["stop"]
    win = (t >= WIN_MS[0]) & (t <= np.where(np.isfinite(stop), stop, WIN_MS[1]))
    if keep is not None:
        win &= np.isin(F["tid"], np.fromiter(keep, int))
    n0 = int(win.sum())
    win = retraction_cut(win, F["tid"], F["spout_raw"], F["spout_lk"])
    pos = F["pos"]
    out = {("_gate", "moved"): (np.zeros((1, 1), np.float32), n0 - int(win.sum())),
           ("_gate", "trials"): (np.zeros((1, 1), np.float32), n0)}
    for kp in KEYPOINTS:
        x, y, ok = F[kp]
        ex, ey = _edges(kp)
        for p in [*np.unique(pos[win]), "all"]:
            m = win if p == "all" else win & (pos == p)
            if not m.any():
                continue
            mm = m & ok
            H, _, _ = np.histogram2d(x[mm], y[mm], bins=(ex, ey))
            out[(kp, p)] = (H.astype(np.float32), int(m.sum()))
            if p == "all":
                out[(kp, "outside")] = (np.zeros((1, 1), np.float32), int(mm.sum() - H.sum()))
                out[(kp, "valid")] = (np.zeros((1, 1), np.float32), int(mm.sum()))
    return out


def _session_worker(job):
    """Module-level worker: one session's histograms -> a local npz; returns (meta, file)."""
    from wfield_local.orofacial_cohort import _repo_on_path, engagement_keep
    _repo_on_path()
    sd, animal, date, epoch, run, cache = job
    sd = Path(sd)
    keep, gate = engagement_keep(animal, date, sd)
    H = session_hist(Path(sd), animal, date, run, keep)
    f = Path(cache) / f"{animal}_{date}.npz"
    np.savez_compressed(f, **{f"{kp}|{p}|H": h for (kp, p), (h, _n) in H.items()},
                        **{f"{kp}|{p}|n": np.int64(n) for (kp, p), (_h, n) in H.items()})
    return {"animal": animal, "date": date, "epoch": epoch, "gate": gate}, str(f)


def collect(rv, run: str, jobs: int | None, baseline: str, cache: Path) -> list:
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    items = []
    for s_ in ak.curated_sessions(order="sorted"):
        a = config.animal_of(s_["label"])
        date = "2026" + s_["label"].split("_")[-1]
        ep = epochs.epoch_of(s_["label"])
        if baseline == "late" and ep == "pre" and date[4:8] < oc.LATE_PRE_FROM:
            continue
        sd = oo.session_dir(rv, run, a, date)
        if (sd / "manifest.json").exists():
            items.append((str(sd), a, date, ep, run, str(cache)))
    res, fail = ak.fan_sessions(items, _session_worker, jobs=jobs, key=lambda it: (it[1], it[2]), label="heatmaps")
    if fail:
        raise RuntimeError(f"{len(fail)} session(s) failed: {fail}")
    return [v for _, v in res]


def proportions(sess: list, kp: str, p: str) -> pd.DataFrame:
    """One row per session with P (counts / window frames) for (kp, position)."""
    rows = []
    for meta, f in sess:
        with np.load(f) as z:
            if f"{kp}|{p}|H" in z.files:
                n = int(z[f"{kp}|{p}|n"])
                if n > 0:
                    rows.append({**meta, "P": z[f"{kp}|{p}|H"].astype(float) / n})
    return pd.DataFrame(rows)


def cohort_maps(D: pd.DataFrame) -> dict:
    """{epoch: {"raw", "delta", "agree", "n_sess", "n_anim"}}: within animal first, then the mean over animals.

    delta (post epochs) = each session minus its animal's pre pool, averaged within animal, then over animals.
    delta (PRE column)  = the NOISE / DRIFT FLOOR: per animal, the mean of its LATER half of pre sessions minus the
                          EARLIER half (chronological), then over animals. (A leave-one-out pre delta averages to exactly
                          zero by construction, so it cannot be the floor.) Under the all-pre baseline this half-split
                          contains the June -> August drift.
    agree = bins where EVERY animal's (smoothed) delta has the cohort's sign (rule 8: visible in each animal)."""
    from scipy.ndimage import gaussian_filter

    from wfield_local.orofacial_cohort import EPOCH_ORDER
    out = {}
    if not len(D):
        return out
    sm = lambda A: gaussian_filter(A, SMOOTH_BINS) if SMOOTH_BINS else A      # noqa: E731
    pre = {a: g.sort_values("date") for a, g in D[D.epoch == "pre"].groupby("animal")}
    pool = {a: np.mean(list(g.P), axis=0) for a, g in pre.items()}
    for e in EPOCH_ORDER:
        g = D[D.epoch == e]
        if not len(g):
            continue
        raw = np.mean([np.mean(list(gg.P), axis=0) for _, gg in g.groupby("animal")], axis=0)
        per_animal = []
        for a, gg in g.groupby("animal"):
            if e == "pre":
                if len(gg) >= 2:
                    h = len(gg) // 2
                    gg = gg.sort_values("date")
                    per_animal.append(np.mean(list(gg.P.iloc[len(gg) - h:]), axis=0) - np.mean(list(gg.P.iloc[:h]), axis=0))
            elif a in pool:
                per_animal.append(np.mean(list(gg.P), axis=0) - pool[a])
        delta = agree = None
        if per_animal:
            S = np.array([sm(x) for x in per_animal])
            delta = S.mean(0)
            big = np.abs(delta) >= 0.1 * np.nanpercentile(np.abs(delta), 99)    # ignore near-zero bins
            agree = (np.sign(S) == np.sign(delta)).all(0) & big if len(S) >= 3 else None
        out[e] = {"raw": raw, "delta": delta, "agree": agree, "n_sess": len(g), "n_anim": g.animal.nunique(),
                  "n_delta_anim": len(per_animal)}
    if "pre" in out:                                   # no-data mask: nothing in either the epoch or the pre map
        empty = sm((out["pre"]["raw"] > 0).astype(float)) == 0
        for v in out.values():
            if v["delta"] is not None:
                v["delta"] = np.where(empty & (sm((v["raw"] > 0).astype(float)) == 0), np.nan, v["delta"])
    return out


def _diff_cmap():
    import matplotlib.colors as mcolors
    cm = mcolors.LinearSegmentedColormap.from_list("rgb_diverging", ["blue", "#d0d0d0", "red"])   # stroke_orofacial
    cm.set_bad("white")
    return cm


def _panel(ax, M, kp, kind, vmax, ref=None, agree=None):
    import matplotlib.colors as mcolors
    (x0, x1), (y0, y1) = EXTENT[kp]
    if M is None:
        ax.set_axis_off()
        return None
    if kind == "raw":
        import matplotlib.pyplot as plt
        cm = plt.get_cmap("magma").copy()
        cm.set_bad("white")
        A = np.where(M > 0, M, np.nan)
        im = ax.imshow(A.T, origin="lower", extent=(x0, x1, y0, y1), cmap=cm, aspect="equal",
                       norm=mcolors.LogNorm(vmin=LOG_VMIN, vmax=vmax), interpolation="nearest")
    else:
        im = ax.imshow(M.T, origin="lower", extent=(x0, x1, y0, y1), cmap=_diff_cmap(), aspect="equal",
                       norm=mcolors.Normalize(-vmax, vmax), interpolation="nearest")
    if kind == "delta" and agree is not None and agree.any():
        xs = np.linspace(x0, x1, NBINS + 1)
        ys = np.linspace(y0, y1, NBINS + 1)
        ax.contour((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2, agree.T.astype(float), levels=[0.5],
                   colors="k", linewidths=0.7)
    if x0 <= 0 <= x1 and y0 <= 0 <= y1:
        ax.plot(0, 0, "+", color="c", ms=7, mew=1.4)               # the mouth (only when it is inside the extent)
    if ref is not None:
        ax.plot(*ref, "x", color="c", ms=6, mew=1.4)                # mean spout tip at this position
    ax.set_facecolor("white")
    ax.tick_params(labelsize=7)
    return im


def _vmax(maps, kind):
    vals = [np.abs(m[kind])[np.isfinite(m[kind])] for m in maps if m.get(kind) is not None]
    vals = [v for v in vals if v.size]
    if not vals:
        return 1.0
    v = np.concatenate(vals)
    return float(np.nanpercentile(v[v > 0], 99.5 if kind == "raw" else 99.0)) if (v > 0).any() else 1.0


KP_DESC = {"tongue": "tongue tip (cleaned, OUT frames >= 20 px)", "jaw": "jaw (cleaned, rest-relative)",
           "nose": "nose (pose; head / snout posture = behaviour)",
           "spout": "spout tip (pose; fixed apparatus = the drift CONTROL)"}


def _note(kp: str, title: str, kind: str) -> str:
    import textwrap
    what = (f"{KP_DESC[kp]}; x = lateral (+ = IPSI), y = forward (out of the mouth toward the spouts), plotted DOWN as the camera sees it, px, in the mouth / "
            f"spout frame{'' if kp != 'jaw' else ' (jaw: offset from its rest position)'}; frames: ENGAGED trials, cue .. "
            f"DAQ trial stop, cut at spout-retraction onset; P = frames in bin / ALL window frames (stroke_orofacial "
            f"Q17). ")
    what += ("RAW: within animal (mean over sessions), then mean over animals; log colour. " if kind in ("raw", "both")
             else "")
    what += (f"DELTA: each animal's epoch mean minus ITS OWN pre pool at the position, then the mean over animals; PRE "
             f"column = noise / drift floor = each animal's later half of pre sessions minus its earlier half; smoothed "
             f"(Gaussian sigma {SMOOTH_BINS:g} bin); red = more occupancy than pre, blue = less; white = no data; BLACK "
             f"CONTOUR = every animal's delta has the cohort's sign, |delta| >= 10 % of the map's 99th pct (needs >= 3 animals). Distributional test: W2 "
             f"(w2/ figures). " if kind in ("delta", "both") else "")
    what += "+ = mouth, x = mean spout tip."
    return "\n".join(textwrap.wrap(title, 150)), "\n".join(textwrap.wrap(what, 150))


def fig_by_position(maps: dict, kp: str, kind: str, path: Path, title: str, spout_ref: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import orofacial_cohort as oc
    eps = [e for e in oc.EPOCH_ORDER if any(e in maps.get(p, {}) for p in oc.POS_ORDER)]
    (x0, x1), (y0, y1) = EXTENT[kp]
    w = 2.6 * (x1 - x0) / (y1 - y0)
    fig, axs = plt.subplots(len(oc.POS_ORDER), len(eps), figsize=(max(2.2, w) * len(eps) + 1.2,
                                                                     2.6 * len(oc.POS_ORDER) + 1.6),
                            squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(top=0.93, bottom=0.09, right=0.86)
    vmax = _vmax([m for p in oc.POS_ORDER for m in maps.get(p, {}).values()], kind)
    im = None
    for i, p in enumerate(oc.POS_ORDER):
        for j, e in enumerate(eps):
            ax = axs[i][j]
            m = maps.get(p, {}).get(e)
            r = _panel(ax, None if m is None else m[kind], kp, kind, vmax,
                       spout_ref.get(p) if kp in ("tongue", "spout") else None,
                       None if m is None else m.get("agree"))
            im = r or im
            if i == 0:
                ax.set_title(e, fontsize=11, color=oc.EPOCH_LINE[e], fontweight="bold")
            if m is not None:
                ax.text(0.02, 0.98, f"n={m['n_sess']} s / {m['n_anim']} a", transform=ax.transAxes, fontsize=7,
                        va="top", color="0.25")
                if kind == "delta" and e == "pre":
                    ax.text(0.02, 0.02, "FLOOR: later - earlier\nhalf of pre sessions", transform=ax.transAxes,
                            fontsize=7, va="bottom", color="0.1")
            if j == 0:
                ax.set_ylabel(f"{oc.pos_disp(p)}\nforward (px; down)", fontsize=9, color=oc._pos_color(p, "L"))
            if i == len(oc.POS_ORDER) - 1:
                ax.set_xlabel("lateral (px; + = ipsi)", fontsize=8)
    axs[0][0].invert_yaxis()                     # CAMERA orientation (Priya 10-18): the tongue comes DOWN (shared y)
    if im is not None:
        cb = fig.colorbar(im, ax=axs, shrink=0.5, pad=0.02)
        cb.set_label("P (frames in bin / window frames)" if kind == "raw" else "P - own pre P", fontsize=9)
    _t, _n = _note(kp, title + f" -- {kp} by position, {kind.upper()}", kind)
    fig.suptitle(_t, fontsize=10, y=0.995)
    fig.text(0.5, 0.005, _n, ha="center", va="bottom", fontsize=8)
    oc.save_png_svg(fig, path, dpi=130)
    plt.close(fig)


def fig_nose_by_epoch(maps: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import orofacial_cohort as oc
    eps = [e for e in oc.EPOCH_ORDER if e in maps]
    fig, axs = plt.subplots(2, len(eps), figsize=(3.6 * len(eps) + 1.6, 9.0), squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(top=0.86, bottom=0.17, right=0.88, hspace=0.25)
    for i, kind in enumerate(("raw", "delta")):
        vmax = _vmax(list(maps.values()), kind)
        im = None
        for j, e in enumerate(eps):
            ax = axs[i][j]
            m = maps[e]
            im = _panel(ax, m[kind], "nose", kind, vmax, agree=m.get("agree")) or im
            if i == 0:
                ax.set_title(f"{e}\nn={m['n_sess']} sessions / {m['n_anim']} animals", fontsize=10,
                             color=oc.EPOCH_LINE[e])
            if i == 1 and e == "pre":
                ax.set_title("noise / drift FLOOR:\nlater - earlier half of pre sessions", fontsize=9)
            if j == 0:
                ax.set_ylabel(("RAW" if kind == "raw" else "DELTA vs own pre") + "\nforward (px; down)", fontsize=9)
            if i == 1:
                ax.set_xlabel("lateral (px; + = ipsi)", fontsize=8)
        if im is not None:
            cb = fig.colorbar(im, ax=list(axs[i]), shrink=0.85, pad=0.02)
            cb.set_label("P" if kind == "raw" else "P - own pre P", fontsize=9)
    axs[0][0].invert_yaxis()                     # CAMERA orientation (shared y)
    _t, _n = _note("nose", title + " -- nose by epoch, all positions pooled", "both")
    fig.suptitle(_t, fontsize=10, y=0.995)
    fig.text(0.5, 0.005, _n, ha="center", va="bottom", fontsize=8)
    oc.save_png_svg(fig, path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    import tempfile

    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    oc._repo_on_path()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--baseline", choices=("all", "late"), default="all",
                    help="pre pool: all curated pre sessions, or only those >= late_pre_from ('late' -> *_latepre)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    base = oo.root(rv) / a.run / "cohort"
    sfx = "_latepre" if a.baseline == "late" else ""
    figs = base / "figures" / f"heatmaps{sfx}"
    writeguard.assert_writable(base)
    figs.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "orofacial_heatmaps" / a.run
    cache.mkdir(parents=True, exist_ok=True)
    sess = collect(rv, a.run, a.jobs, a.baseline, cache)
    # mean spout tip per position (mouth frame), from the spout histograms' centre of mass over pre sessions
    spout_ref = {}
    ex, ey = _edges("spout")
    cx, cy = (ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2
    for p in {k.split("|")[1] for _m, f in sess for k in np.load(f).files if k.startswith("spout|")}:
        D = proportions(sess, "spout", p)
        if len(D) and p not in ("all", "outside", "valid"):
            H = np.mean(list(D[D.epoch == "pre"].P if (D.epoch == "pre").any() else D.P), axis=0)
            if H.sum() > 0:
                spout_ref[oc.pos_key(p, "L")] = (float((H.sum(1) * cx).sum() / H.sum()),
                                                 float((H.sum(0) * cy).sum() / H.sum()))
    t = (f"Orofacial heatmaps, pre vs post stroke -- {a.run} -- ENGAGED trials -- baseline: "
         f"{'LATE (August) pre only' if a.baseline == 'late' else 'all curated pre'} -- sessions: "
         f"{pd.Series([m['animal'] for m, _ in sess]).value_counts().sort_index().to_dict()}")
    raw_pos = sorted({k.split("|")[1] for _m, f in sess for k in np.load(f).files}
                     - {"all", "outside", "valid", "moved", "trials"})
    tot = {}
    for _m, f in sess:
        with np.load(f) as z:
            for k in z.files:
                if k.endswith("|n") and k.split("|")[1] in ("outside", "valid", "moved", "trials"):
                    tot[k[:-2]] = tot.get(k[:-2], 0) + int(z[k])
    print(f"retraction cut: {tot.get('_gate|moved', 0)} of {tot.get('_gate|trials', 0)} window frames "
          f"({tot.get('_gate|moved', 0) / max(tot.get('_gate|trials', 1), 1):.4%})")
    for kp in KEYPOINTS:
        print(f"  {kp}: {tot.get(kp + '|outside', 0) / max(tot.get(kp + '|valid', 1), 1):.4%} of valid frames outside the axes")
    summary = []
    for kp in KEYPOINTS:
        maps = {}
        for p in raw_pos:
            maps[oc.pos_key(p, "L")] = cohort_maps(proportions(sess, kp, p))
        for kind in ("raw", "delta"):
            fig_by_position(maps, kp, kind, figs / f"{kp}_by_position_{kind}.png", t, spout_ref)
        for p, mp in maps.items():
            for e, v in mp.items():
                summary.append({"keypoint": kp, "pos": p, "epoch": e, "n_sess": v["n_sess"], "n_anim": v["n_anim"],
                                "mass": float(v["raw"].sum()),
                                "abs_delta_sum": float(np.nansum(np.abs(v["delta"]))) if v["delta"] is not None
                                else np.nan})
    fig_nose_by_epoch(cohort_maps(proportions(sess, "nose", "all")), figs / "nose_by_epoch.png", t)
    S = pd.DataFrame(summary)
    S.to_csv(figs / "summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print("in-window occupancy mass and sum|delta| (total variation x2) by keypoint x epoch (mean over positions):")
    print(S.groupby(["keypoint", "epoch"])[["mass", "abs_delta_sum"]].mean().unstack("epoch").round(3).to_string())
    print(f"figures -> {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

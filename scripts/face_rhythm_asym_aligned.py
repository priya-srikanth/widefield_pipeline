"""Cue- and lick-aligned FACE ASYMMETRY per spout position, across epochs (Priya 10-24).

Time-resolved contra- vs ipsilesional face movement: per 25 Hz frame, the mean SPEED of the tracked mesh points on each
half of the face (split at the master's LP-nose midline; cam4 ipsilesional = IMAGE RIGHT, all lesions left),
reconstructed from each session's frozen point PCs (`proj/` ``points_pc``, 20 PCs of the centred (x, y) displacement,
`master.npz` ``points_components``): velocity = d(PCs)/dt @ components -> per-point |v|. The 20 PCs carry most but not
all point variance (fraction printed), so small movements are under-represented equally on both sides.

  CUE-aligned   engaged trials (`proj/` cues), cue -2 .. +4 s, per spout position
  LICK-aligned  LP licks (v36 ``licks_all``) inside engaged trials (cue -2 .. +4 s), lick -0.5 .. +0.5 s, per the
                trial's spout position -- PER LICK, so independent of how many licks there are
  BOUT-aligned  the FIRST lick of each bout (v36 bout_idx; n_in_bout == 1), -0.5 .. +2 s (Priya 10-24): the time course
                of a bout from its initiation, independent of reaction time after the cue
Per session: mean contra and ipsi speed traces per position; per animal x epoch: mean over sessions; ASYMMETRY =
(contra - ipsi) / (contra + ipsi) of those means (> 0 = more contralesional movement). Window summaries: cue 0 .. +2 s,
lick -0.1 .. +0.2 s, bout 0 .. +0.5 s. The same windows give PER-POINT mean speed maps per session x alignment x
position, cached in data/aligned_point_maps.npz for `face_rhythm_aligned_maps.py` (maps + patch statistics).

    python scripts/face_rhythm_asym_aligned.py [--set master_pre_v1] [--jobs 3] [--plot-only]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/asym_aligned/``: asym_aligned_epochs.csv (per animal x epoch x position x
alignment: window asymmetry, contra, ipsi), asym_aligned_traces.png (animal-mean traces, rows = positions, epochs as
colours; each panel lists the change vs pre with its mark), asym_aligned_by_animal.png (window asymmetry, per animal:
positions x epochs), asym_aligned_change.png + data/asym_aligned_stats.csv (Priya 10-24 statistics: per session the
window asymmetry minus the animal's pre mean, `boot_delta` per alignment x position x epoch; * = 95 % interval
excludes 0, ** = also Holm over the 6 positions of an alignment x epoch). data/asym_traces.pkl = the per-session
traces (``--plot-only`` redraws everything from it; the per-point maps cache is written only by a full run).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.face_rhythm import face as frface  # noqa: E402
from wfield_local.face_rhythm import stats as frs  # noqa: E402

FS = 25.0
CUE_LAGS = np.arange(-2.0, 4.0001, 1 / FS)
LICK_LAGS = np.arange(-0.5, 0.5001, 1 / FS)
BOUT_LAGS = np.arange(-0.5, 2.0001, 1 / FS)
CUE_WIN, LICK_WIN, BOUT_WIN = (0.0, 2.0), (-0.1, 0.2), (0.0, 0.5)
ALIGN = {"cue": (CUE_LAGS, CUE_WIN, "s from cue"), "lick": (LICK_LAGS, LICK_WIN, "s from lick onset"),
         "bout": (BOUT_LAGS, BOUT_WIN, "s from bout onset (first lick)")}
POSITIONS = frc.POSITIONS                      # animal's L (ipsilesional) -> R (contralesional)
EPOCHS = frc.EPOCHS


# speed reconstruction, event weights and event alignment: `wfield_local.face_rhythm.face` (shared with the
# aligned-map and bout-split readouts; kept as names here so `face_rhythm_asym_aligned.side_speed` still resolves)
side_speed = frface.side_speed
window_weights = frface.window_weights


def _aligned(t: np.ndarray, y: np.ndarray, ev: np.ndarray, lags: np.ndarray) -> np.ndarray:
    return frface.aligned_mean(t, y, ev, lags, FS)


def job(item: tuple) -> dict:
    fdir, key = item
    from wfield_local.face_rhythm import licks
    from wfield_local.paths import PathResolver
    fdir = Path(fdir)
    comps, contra = frface.master_points(fdir)                           # image LEFT = contralesional (cam4)
    with np.load(fdir / "proj" / f"{key}.npz", allow_pickle=True) as z:
        animal, date, epoch = str(z["animal"]), str(z["date"]), str(z["epoch"])
        pcs, t = z["points_pc"].astype(np.float32), z["points_t_daq_s"].astype(float)
        cues, pos = z["cue_daq_s"].astype(float), z["position"].astype(str)
    L = licks.lick_table(PathResolver(), animal, date, cues)
    L = L[np.isfinite(L.t_from_cue)]
    events = {}
    for p in POSITIONS:
        lp = L[L.position.astype(str) == p]
        events[("cue", p)] = cues[pos == p]
        events[("lick", p)] = lp.daq_on_s.to_numpy(float)
        events[("bout", p)] = lp[lp.n_in_bout == 1].daq_on_s.to_numpy(float)
    okf = np.r_[False, np.isfinite(pcs[1:]).all(1) & np.isfinite(pcs[:-1]).all(1)]
    cols = list(events)
    W = np.column_stack([window_weights(t, events[k], ALIGN[k[0]][1], okf) for k in cols])
    c, i, Ssum = side_speed(pcs, t, comps, contra, W)
    wsum = W.sum(0)
    out = {"key": key, "animal": animal, "date": date, "epoch": epoch, "label": frc.label(animal, date)}
    for j, (al, p) in enumerate(cols):
        ev, lags = events[(al, p)], ALIGN[al][0]
        out[(al, p)] = (_aligned(t, c, ev, lags), _aligned(t, i, ev, lags), len(ev))
        out[("map", al, p)] = Ssum[:, j] / wsum[j] if wsum[j] > 0 else np.full(Ssum.shape[0], np.nan)
    return out


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--min-events", type=int, default=5, help="per session x position, else that cell is skipped")
    ap.add_argument("--plot-only", action="store_true", help="redraw from data/asym_traces.pkl, no recompute")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "asym_aligned"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    M = np.load(fdir / "master.npz")
    print(f"point PCs carry {100 * float(np.sum(M['points_explained'])):.1f} % of point displacement variance")
    tcache = figure_layout.sidecar_for(out, "asym_traces", ".pkl")
    fail = []
    if a.plot_only:
        R = pd.read_pickle(tcache)
    else:
        keys = frc.session_keys(fdir)
        got, fail = ak.fan_sessions([(str(fdir), k) for k in keys], job, jobs=a.jobs, key=lambda it: it[1],
                                    label="asym")
        R = [r for _, r in got]
        # per-point window maps for `face_rhythm_aligned_maps.py`: (sessions, alignments, positions, P) + event counts
        aligns = tuple(ALIGN)
        maps = np.stack([[[r[("map", al, p)] for p in POSITIONS] for al in aligns] for r in R]).astype(np.float32)
        counts = np.array([[[r[(al, p)][2] for p in POSITIONS] for al in aligns] for r in R])
        np.savez(figure_layout.sidecar_for(out, "aligned_point_maps", ".npz"), maps=maps, counts=counts,
                 key=np.array([r["key"] for r in R]), label=np.array([r["label"] for r in R]),
                 animal=np.array([r["animal"] for r in R]), epoch=np.array([r["epoch"] for r in R]),
                 aligns=np.array(aligns), positions=np.array(POSITIONS),
                 windows=np.array([ALIGN[al][1] for al in aligns]), home=M["home"])
        R = [{k: v for k, v in r.items() if not (isinstance(k, tuple) and k[0] == "map")} for r in R]
        pd.to_pickle(R, tcache)

    # per animal x epoch x alignment x position: mean of SESSION traces (a split video, PS92_0812 v1 / v2, is one
    # session: its two halves averaged first)
    acc = {}
    for r in R:
        for al in ALIGN:
            for p in POSITIONS:
                cc, ii, n = r[(al, p)]
                if n < a.min_events:
                    continue
                acc.setdefault((r["animal"], r["epoch"], al, p, r["label"]), []).append((cc, ii))
    sess, srows = {}, []
    for (an, ep, al, p, lab), v in acc.items():
        cc, ii = np.nanmean([x[0] for x in v], 0), np.nanmean([x[1] for x in v], 0)
        sess.setdefault((an, ep, al, p), []).append((cc, ii))
        lags, win = ALIGN[al][0], ALIGN[al][1]
        w = (lags >= win[0]) & (lags <= win[1])
        cw, iw = float(np.nanmean(cc[w])), float(np.nanmean(ii[w]))
        srows.append({"animal": an, "epoch": ep, "label": lab, "align": al, "position": p, "asym": (cw - iw) / (cw + iw)})
    # STATISTICS (Priya 10-24): per SESSION the window asymmetry minus the mean of the animal's pre sessions (same
    # alignment x position), change vs pre by `boot_delta`; Holm over the 6 positions of each alignment x epoch
    Ss = pd.DataFrame(srows).sort_values(["align", "position", "animal", "label"], kind="stable")
    Ss["rel"] = Ss.asym - Ss.groupby(["animal", "align", "position"]).asym.transform(
        lambda x: x[Ss.loc[x.index, "epoch"] == "pre"].mean())
    rng = np.random.default_rng(0)
    st = []
    for al in ALIGN:
        for p in POSITIONS:
            d = Ss[(Ss["align"] == al) & (Ss.position == p)]
            st += [{"align": al, "position": p, **r} for r in frs.epoch_change(d, "rel", rng)]
    ST = frs.add_holm(pd.DataFrame(st), ["align", "epoch"])
    ST.to_csv(figure_layout.sidecar_for(out, "asym_aligned_stats", ".csv"), index=False, float_format="%.4g")
    rows, traces = [], {}
    for (an, ep, al, p), v in sess.items():
        cc, ii = np.nanmean([x[0] for x in v], 0), np.nanmean([x[1] for x in v], 0)
        lags, win = ALIGN[al][0], ALIGN[al][1]
        w = (lags >= win[0]) & (lags <= win[1])
        cw, iw = float(np.nanmean(cc[w])), float(np.nanmean(ii[w]))
        traces[(an, ep, al, p)] = (cc - ii) / (cc + ii)
        rows.append({"animal": an, "epoch": ep, "align": al, "position": p, "n_sessions": len(v),
                     "contra": cw, "ipsi": iw, "asym": (cw - iw) / (cw + iw)})
    E = pd.DataFrame(rows)
    E["epoch"] = pd.Categorical(E.epoch, EPOCHS, ordered=True)
    E["position"] = pd.Categorical(E.position, POSITIONS, ordered=True)
    E = E.sort_values(["align", "animal", "position", "epoch"])
    E.to_csv(figure_layout.sidecar_for(out, "asym_aligned_epochs", ".csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    for al in ALIGN:
        print(f"\n{al.upper()}-aligned asymmetry ({ALIGN[al][1][0]:+g}..{ALIGN[al][1][1]:+g} s), per animal "
              "(rows = position, cols = epoch; > 0 = more contralesional)")
        print(E[E["align"] == al].pivot_table(index=["animal", "position"], columns="epoch", values="asym",
                                              observed=True).round(3).to_string())
    if fail:
        print("failed:", [f[0] for f in fail])
    print("\nCHANGE vs pre of the window asymmetry (per animal first, rule 8; then boot_delta, Holm over positions)")
    print(ST[["align", "position", "epoch", *sorted(Ss.animal.unique()), "delta", "ci_lo", "ci_hi", "p_holm",
              "mark"]].round(3).to_string(index=False))

    animals = sorted(E.animal.unique())
    ecol = frc.epoch_colors()        # FR epoch palette (orofacial / 3-D kinematics); animals own `animal_colors`
    with frf.style():
        fig, axs = frf.subplots(len(POSITIONS), 3, figsize=(16, 2.4 * len(POSITIONS)), squeeze=False, sharey=True)
        for r_i, p in enumerate(frc.POSITIONS_DISPLAY):
            for c_i, (al, (lags, _, _)) in enumerate(ALIGN.items()):
                ax = axs[r_i, c_i]
                for ep in EPOCHS:
                    tr = [traces[(an, ep, al, p)] for an in animals if (an, ep, al, p) in traces]
                    if tr:
                        ax.plot(lags, np.nanmean(tr, 0), color=ecol[ep], lw=1.3, label=f"{ep} ({len(tr)} animals)")
                ax.axhline(0, color="k", lw=0.5)
                ax.axvline(0, color="k", lw=0.5)
                ax.axvspan(*ALIGN[al][1], color="0.92", zorder=0, lw=0)
                ax.set_title(f"{p}: {al}-aligned", fontsize=9)
                tt = ST[(ST["align"] == al) & (ST.position == p)].set_index("epoch")
                ax.text(0.99, 0.03, "change vs pre: " + "  ".join(
                    f"{e[:3]} {tt.loc[e, 'delta']:+.2f}{tt.loc[e, 'mark']}" for e in EPOCHS[1:] if e in tt.index),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
                if c_i == 0:
                    ax.set_ylabel("(contra - ipsi) / sum")
        for c_i, (_, _, xl) in enumerate(ALIGN.values()):
            axs[-1, c_i].set_xlabel(xl)
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: face asymmetry, cue-, lick- and bout-onset-aligned, per spout position (mean of animals)")
        frf.save(fig, out / "asym_aligned_traces.png", f"""
Per 25 Hz frame, mean speed of the tracked mesh points on the CONTRALESIONAL (image left on cam4; all lesions left)
and IPSILESIONAL halves of the face, reconstructed from the frozen point PCs; per session the mean contra and ipsi
traces around engaged-trial CUES (left), LP LICK onsets within engaged trials (middle, per lick) and BOUT onsets (right,
the first lick of each bout, -0.5..+2 s), per spout position
(rows; the animal's L = ipsilesional, R = contralesional spouts); per animal x epoch the mean over sessions; drawn: the
mean over animals of (contra - ipsi) / (contra + ipsi). > 0 = more contralesional movement. Sessions with < {a.min_events}
events at a position skipped there. Grey band: the summary window. Bottom right of each panel: change of the window
asymmetry vs pre (per session minus the animal's pre mean; `analysis_kit.boot_delta`, paired animals -> sessions);
* = 95 % interval excludes 0, ** = also Holm over the 6 positions of that alignment x epoch. Per-animal values:
asym_aligned_by_animal.png / data/asym_aligned_epochs.csv; statistics data/asym_aligned_stats.csv.""")

        fig, axs = frf.subplots(len(ALIGN), len(animals), figsize=(3.6 * len(animals) + 1, 3.8 * len(ALIGN)), squeeze=False)
        vmax = float(np.nanpercentile(np.abs(E.asym), 98)) or 0.1
        for r_i, al in enumerate(ALIGN):
            for j, an in enumerate(animals):
                ax = axs[r_i, j]
                T = E[(E["align"] == al) & (E.animal == an)].pivot_table(index="position", columns="epoch",
                                                                           values="asym", observed=False)
                T = T.reindex(index=list(frc.POSITIONS_DISPLAY), columns=list(EPOCHS))
                im = ax.imshow(T.to_numpy(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                ax.set_xticks(range(len(EPOCHS)), [e[:5] for e in EPOCHS])
                ax.set_yticks(range(len(POSITIONS)), frc.POSITIONS_DISPLAY if j == 0 else [])
                ax.set_title(f"{an}: {al}-aligned", fontsize=9)
                for (yy, xx), v in np.ndenumerate(T.to_numpy(float)):
                    if np.isfinite(v):
                        ax.text(xx, yy, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axs, shrink=0.5, label="(contra - ipsi) / sum")
        fig.suptitle(f"{fdir.name}: face asymmetry per animal (cue 0..+2 s; lick -0.1..+0.2 s; bout onset 0..+0.5 s)")
        frf.save(fig, out / "asym_aligned_by_animal.png", """
Window asymmetry (contra - ipsi) / (contra + ipsi) of the mean point speed on each half of the face, per animal (columns)
x spout position (rows) x epoch, CUE-aligned (cue 0..+2 s, engaged trials), LICK-aligned (lick onset -0.1..+0.2 s, LP
licks inside engaged trials, per lick) and BOUT-onset-aligned (first lick of a bout, 0..+0.5 s). Red = more
contralesional movement. Rule 8: read per animal.""")
        # the change, per position: per-animal epoch means (dots) + boot_delta point and interval, marks
        fig, axs = frf.subplots(1, len(ALIGN), figsize=(5.2 * len(ALIGN), 4.2), squeeze=False, sharey=True)
        acol = frc.animal_colors()
        for c_i, al in enumerate(ALIGN):
            ax = axs[0, c_i]
            for j, p in enumerate(frc.POSITIONS_DISPLAY):
                for e_i, ep in enumerate(EPOCHS[1:]):
                    x0 = j + (e_i - 1) * 0.25
                    r = ST[(ST["align"] == al) & (ST.position == p) & (ST.epoch == ep)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    for an in animals:
                        if an in r and np.isfinite(r[an]):
                            ax.plot(x0, r[an], ".", color=acol.get(an, "0.5"), ms=4, alpha=0.8)
                    if np.isfinite(r.delta):
                        ax.errorbar(x0, r.delta, yerr=[[r.delta - r.ci_lo], [r.ci_hi - r.delta]], fmt="s",
                                    color=ecol[ep], mec="k", ms=5, capsize=2, label=ep if j == 0 else None)
                        ax.text(x0, r.ci_hi, r.mark, ha="center", va="bottom", fontsize=9)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(len(POSITIONS)), frc.POSITIONS_DISPLAY, rotation=30, fontsize=8)
            ax.set_title(f"{al}-aligned ({ALIGN[al][1][0]:+g}..{ALIGN[al][1][1]:+g} s)", fontsize=10)
        axs[0, 0].set_ylabel("change of (contra - ipsi) / sum vs pre")
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: change of face asymmetry vs pre, per spout position (< 0 = less contralesional)")
        frf.save(fig, out / "asym_aligned_change.png", """
Per session, the window asymmetry (contra - ipsi) / (contra + ipsi) of the mean point speed (cue 0..+2 s; each lick
-0.1..+0.2 s; bout onset 0..+0.5 s), minus the mean of the animal's pre-stroke sessions. Dots: per-animal epoch means
(repo animal colours); squares: `analysis_kit.boot_delta` change vs pre and its 95 % interval (paired animals ->
sessions, animal-weighted; FR epoch colours acute / subacute / chronic left to right). * = interval excludes 0; ** =
also significant after Holm over the 6 positions of that alignment x epoch. Negative = the face moves relatively less on
the contralesional side than before the stroke. The animal's R spouts are contralesional (all lesions left).""")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cue- and lick-aligned FACE ASYMMETRY per spout position, across epochs (Priya 10-24).

Time-resolved contra- vs ipsilesional face movement: per 25 Hz frame, the mean SPEED of the tracked mesh points on each
half of the face (split at the master's LP-nose midline; cam4 ipsilesional = IMAGE RIGHT, all lesions left),
reconstructed from each session's frozen point PCs (`proj/` ``points_pc``, 20 PCs of the centred (x, y) displacement,
`master.npz` ``points_components``): velocity = d(PCs)/dt @ components -> per-point |v|. The 20 PCs carry most but not
all point variance (fraction printed), so small movements are under-represented equally on both sides.

  CUE-aligned   engaged trials (`proj/` cues), cue -2 .. +4 s, per spout position
  LICK-aligned  LP licks (v36 ``licks_all``) inside engaged trials (cue -2 .. +4 s), lick -0.5 .. +0.5 s, per the
                trial's spout position -- PER LICK, so independent of how many licks there are
Per session: mean contra and ipsi speed traces per position; per animal x epoch: mean over sessions; ASYMMETRY =
(contra - ipsi) / (contra + ipsi) of those means (> 0 = more contralesional movement). Window summaries: cue 0 .. +2 s,
lick -0.1 .. +0.2 s.

    python scripts/face_rhythm_asym_aligned.py [--set master_pre_v1] [--jobs 3]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/asym_aligned/``: asym_aligned_epochs.csv (per animal x epoch x position x
alignment: window asymmetry, contra, ipsi), asym_aligned_traces.png (animal-mean traces, rows = positions, epochs as
colours), asym_aligned_by_animal.png (window asymmetry, per animal: positions x epochs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FS = 25.0
CUE_LAGS = np.arange(-2.0, 4.0001, 1 / FS)
LICK_LAGS = np.arange(-0.5, 0.5001, 1 / FS)
CUE_WIN, LICK_WIN = (0.0, 2.0), (-0.1, 0.2)
POSITIONS = ("far_L", "close_L", "close_center", "far_center", "close_R", "far_R")   # animal's L = ipsilesional
EPOCHS = ("pre", "acute", "subacute", "chronic")
CHUNK = 20000


def side_speed(pcs: np.ndarray, t: np.ndarray, comps: np.ndarray, contra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(contra, ipsi) mean point speed per frame (px / s) from the frozen point PCs. ``comps`` (k, 2P) on the INTERLEAVED
    (x0, y0, x1, y1, ...) displacement layout of `points_stream`; ``contra`` (P,) bool."""
    dt = np.diff(t)
    c = np.full(len(t), np.nan, np.float32)
    i = np.full(len(t), np.nan, np.float32)
    P = comps.shape[1] // 2
    for c0 in range(1, len(t), CHUNK):
        c1 = min(c0 + CHUNK, len(t))
        dv = (pcs[c0:c1] - pcs[c0 - 1:c1 - 1]) / dt[c0 - 1:c1 - 1, None]          # PC velocity
        v = (dv @ comps).reshape(-1, P, 2)
        sp = np.sqrt((v ** 2).sum(-1))                                           # (frames, P)
        c[c0:c1] = sp[:, contra].mean(1)
        i[c0:c1] = sp[:, ~contra].mean(1)
    return c, i


def _aligned(t: np.ndarray, y: np.ndarray, ev: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Mean over events of ``y`` at the nearest frame to event + lag (NaN off the recording)."""
    if not len(ev):
        return np.full(len(lags), np.nan)
    q = ev[:, None] + lags[None]
    k = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
    k = np.where(np.abs(t[k - 1] - q) < np.abs(t[k] - q), k - 1, k)
    out = y[k].astype(float)
    out[np.abs(t[k] - q) > 1.5 / FS] = np.nan
    return np.nanmean(out, 0)


def job(item: tuple) -> dict:
    fdir, key = item
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import face_rhythm_validation as frv
    from wfield_local.paths import PathResolver
    fdir = Path(fdir)
    M = np.load(fdir / "master.npz")
    comps = M["points_components"].astype(np.float32)
    contra = M["home"][:, 0] <= float(frr.MASTER_MIDLINE_X)                     # image LEFT = contralesional (cam4)
    with np.load(fdir / "proj" / f"{key}.npz", allow_pickle=True) as z:
        animal, date, epoch = str(z["animal"]), str(z["date"]), str(z["epoch"])
        pcs, t = z["points_pc"].astype(np.float32), z["points_t_daq_s"].astype(float)
        cues, pos = z["cue_daq_s"].astype(float), z["position"].astype(str)
    c, i = side_speed(pcs, t, comps, contra)
    L = frv.lick_table(PathResolver(), animal, date, cues)
    L = L[np.isfinite(L.t_from_cue)]
    out = {"key": key, "animal": animal, "date": date, "epoch": epoch, "label": frr._label(animal, date)}
    for p in POSITIONS:
        ev = cues[pos == p]
        out[("cue", p)] = (_aligned(t, c, ev, CUE_LAGS), _aligned(t, i, ev, CUE_LAGS), len(ev))
        lk = L[L.position.astype(str) == p].daq_on_s.to_numpy(float)
        out[("lick", p)] = (_aligned(t, c, lk, LICK_LAGS), _aligned(t, i, lk, LICK_LAGS), len(lk))
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
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "asym_aligned"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    M = np.load(fdir / "master.npz")
    print(f"point PCs carry {100 * float(np.sum(M['points_explained'])):.1f} % of point displacement variance")
    keys = sorted(f.stem for f in (fdir / "proj").glob("*.npz") if not f.name.endswith(".part.npz"))
    got, fail = ak.fan_sessions([(str(fdir), k) for k in keys], job, jobs=a.jobs, key=lambda it: it[1], label="asym")
    R = [r for _, r in got]

    # per animal x epoch x alignment x position: mean of SESSION traces (a split video, PS92_0812 v1 / v2, is one
    # session: its two halves averaged first)
    acc = {}
    for r in R:
        for al in ("cue", "lick"):
            for p in POSITIONS:
                cc, ii, n = r[(al, p)]
                if n < a.min_events:
                    continue
                acc.setdefault((r["animal"], r["epoch"], al, p, r["label"]), []).append((cc, ii))
    sess = {}
    for (an, ep, al, p, _lab), v in acc.items():
        sess.setdefault((an, ep, al, p), []).append((np.nanmean([x[0] for x in v], 0), np.nanmean([x[1] for x in v], 0)))
    rows, traces = [], {}
    for (an, ep, al, p), v in sess.items():
        cc, ii = np.nanmean([x[0] for x in v], 0), np.nanmean([x[1] for x in v], 0)
        lags, win = (CUE_LAGS, CUE_WIN) if al == "cue" else (LICK_LAGS, LICK_WIN)
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
    for al in ("cue", "lick"):
        print(f"\n{al.upper()}-aligned asymmetry ({'cue 0..+2 s' if al == 'cue' else 'lick -0.1..+0.2 s'}), per animal "
              "(rows = position, cols = epoch; > 0 = more contralesional)")
        print(E[E["align"] == al].pivot_table(index=["animal", "position"], columns="epoch", values="asym",
                                              observed=True).round(3).to_string())
    if fail:
        print("failed:", [f[0] for f in fail])

    animals = sorted(E.animal.unique())
    ecol = dict(zip(EPOCHS, ("0.3", "tab:red", "tab:orange", "tab:blue")))
    with frf.style():
        fig, axs = frf.subplots(len(POSITIONS), 2, figsize=(11, 2.4 * len(POSITIONS)), squeeze=False, sharey=True)
        for r_i, p in enumerate(POSITIONS):
            for c_i, (al, lags) in enumerate((("cue", CUE_LAGS), ("lick", LICK_LAGS))):
                ax = axs[r_i, c_i]
                for ep in EPOCHS:
                    tr = [traces[(an, ep, al, p)] for an in animals if (an, ep, al, p) in traces]
                    if tr:
                        ax.plot(lags, np.nanmean(tr, 0), color=ecol[ep], lw=1.3, label=f"{ep} ({len(tr)} animals)")
                ax.axhline(0, color="k", lw=0.5)
                ax.axvline(0, color="k", lw=0.5)
                ax.set_title(f"{p}: {al}-aligned", fontsize=9)
                if c_i == 0:
                    ax.set_ylabel("(contra - ipsi) / sum")
        axs[-1, 0].set_xlabel("s from cue")
        axs[-1, 1].set_xlabel("s from lick onset")
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: face asymmetry, cue- and lick-aligned, per spout position (mean of animals)")
        frf.save(fig, out / "asym_aligned_traces.png", f"""
Per 25 Hz frame, mean speed of the tracked mesh points on the CONTRALESIONAL (image left on cam4; all lesions left)
and IPSILESIONAL halves of the face, reconstructed from the frozen point PCs; per session the mean contra and ipsi
traces around engaged-trial CUES (left) and LP LICK onsets within engaged trials (right, per lick), per spout position
(rows; the animal's L = ipsilesional, R = contralesional spouts); per animal x epoch the mean over sessions; drawn: the
mean over animals of (contra - ipsi) / (contra + ipsi). > 0 = more contralesional movement. Sessions with < {a.min_events}
events at a position skipped there. Per-animal values: asym_aligned_by_animal.png / data/asym_aligned_epochs.csv.""")

        fig, axs = frf.subplots(2, len(animals), figsize=(3.6 * len(animals) + 1, 7.5), squeeze=False)
        vmax = float(np.nanpercentile(np.abs(E.asym), 98)) or 0.1
        for r_i, al in enumerate(("cue", "lick")):
            for j, an in enumerate(animals):
                ax = axs[r_i, j]
                T = E[(E["align"] == al) & (E.animal == an)].pivot_table(index="position", columns="epoch",
                                                                           values="asym", observed=False)
                T = T.reindex(index=list(POSITIONS), columns=list(EPOCHS))
                im = ax.imshow(T.to_numpy(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                ax.set_xticks(range(len(EPOCHS)), [e[:5] for e in EPOCHS])
                ax.set_yticks(range(len(POSITIONS)), POSITIONS if j == 0 else [])
                ax.set_title(f"{an}: {al}-aligned", fontsize=9)
                for (yy, xx), v in np.ndenumerate(T.to_numpy(float)):
                    if np.isfinite(v):
                        ax.text(xx, yy, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axs, shrink=0.5, label="(contra - ipsi) / sum")
        fig.suptitle(f"{fdir.name}: face asymmetry per animal (cue 0..+2 s; lick -0.1..+0.2 s)")
        frf.save(fig, out / "asym_aligned_by_animal.png", """
Window asymmetry (contra - ipsi) / (contra + ipsi) of the mean point speed on each half of the face, per animal (columns)
x spout position (rows) x epoch, CUE-aligned (top; cue 0..+2 s, engaged trials) and LICK-aligned (bottom; lick onset
-0.1..+0.2 s, LP licks inside engaged trials, per lick). Red = more contralesional movement. Rule 8: read per animal.""")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

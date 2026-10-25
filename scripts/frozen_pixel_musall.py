"""PIXEL Musall layouts of the FROZEN encoder (Priya 10-25; handoff 000y task C): the single-variable / single-group
ALONE cvR2 maps (Musall et al. 2019 top row) and UNIQUE dR2 maps (bottom row), for the PETH (trial-averaged) and the
trial-by-trial parts of the signal, at pixel level.

Needs a spatial-target run saved with FROZEN_PIX_MUSALL=1 (keys mus_<window>_<avg|trial>_<E|S> in each session npz:
target-space second moments of the residual (E) and of the data (S) inside the MUSALL window; `avg` = per-spout
trial-averaged traces centred on their mean over spouts x time, `trial` = single trials minus their spout's mean;
see `frozen_encoder.musall_moments`). --measure all uses the all-frames moments (resid_mom / data_mom), which every
variant has (incl. the novar_* per-variable drops -> Musall bottom row per VARIABLE, task B).

Per session and pixel: SSE = diag(W E W'), SST = diag(W S W') (W = the animal's basis, `frozen_r2_maps.pixel_sse`),
  alone   R2 of <g>_only            = 1 - SSE[<g>_only] / SST
  unique  dR2 of g                  = (SSE[no_g] - SSE[joint]) / SST
(pre held out leave-one-session-out, post with the frozen all-pre betas). Display: per animal x epoch SSE and SST SUMMED
over sessions before the ratio, then the mean over animals (rule 8). Test: per-session maps, epoch minus pre, nested
bootstrap animals -> sessions (`fr_pixel_significance.test`, 16 px bins, Bonferroni). R2 is a fraction per pixel, but
PS94's target terms are ~10x the others' (ENCODING_FINDINGS 8.13): the per-animal table is printed first.

Rows: pre level per measure, then epoch minus pre per measure. Image left = ipsilesional.

Usage: python scripts/frozen_pixel_musall.py --tag gate-attempted_v7g3_t3d_frA_ls_basis --layout alone --window cue
       python scripts/frozen_pixel_musall.py --tag ... --layout novar --measures all
Output: <E>/frozen_encoder_<tag>/figures/pixel_musall_<layout>_<window>.png + pixel_musall_<layout>_<window>{,_sig}.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402
from fr_pixel_significance import FACTORS, block_mask, test  # noqa: E402
from frozen_r2_maps import pixel_sse  # noqa: E402

#: (label, variant); None = the joint model itself (full R2)
QS = {"alone": [("full", None), ("task", "task_only"), ("instructed (lick)", "lick_only"),
                ("uninstructed (other)", "other_only"), ("spout", "spout_only"), ("movement", "movement_only"),
                ("FaceRhythm", "fr_only")],
      "unique": [("full", None), ("task", "no_task"), ("target", "no_target"), ("spout", "no_spout"),
                 ("lick spout", "no_lickspout"), ("movement", "no_movement"), ("FaceRhythm", "no_fr")],
      # Musall bottom row per VARIABLE (Priya 10-25); order: task, licking, pose, running, FaceRhythm
      "novar": [("full", None)] + [(v, f"novar_{v}") for v in (
          "cue", "arrival", "target", "lick_onset", "contact", "exec_angle", "tongue", "jaw", "nose", "running",
          "fr_lick", "fr_right", "fr_left", "fr_close", "fr_far", "fr_other")]}
MLAB = {"avg": "PETH", "trial": "trial-by-trial", "all": "all frames"}
EPS = ("acute", "subacute", "chronic")


def keys(measure: str, window: str) -> tuple[str, str]:
    if measure == "all":
        return "resid_mom", "data_mom"
    return f"mus_{window}_{measure}_E", f"mus_{window}_{measure}_S"


def collect(run: Path, qs, measures, window):
    """per session: SSE / SST pixel vectors per (q, measure); also animal x epoch sums."""
    from wfield_local import combined_encoding as ce
    sess = sorted(js for js in (run / "joint" / "sessions").glob("*.json") if js.stem[-4:] >= "0801")
    _V, _ft, _Ub, flat, _pa, _nm, shape, pstat = ce.load_imaging(sess[0].stem)
    spatial, meta, sse, sst = {}, [], {}, {}
    for js in sess:
        m = json.loads(js.read_text(encoding="utf-8"))
        an = m["animal"]
        if an not in spatial:
            spatial[an] = pickle.loads((run / "fit" / f"spatial_{an}.pkl").read_bytes())
        keep = np.array([t != "cortex" for t in m["targets"]])
        z0 = np.load(js.with_suffix(".npz"))
        for me in measures:
            ke, ks = keys(me, window)
            if ks not in z0:
                continue
            sst[(js.stem, me)] = pixel_sse(spatial[an], z0[ks], keep)
            for q, v in qs:
                f = js.with_suffix(".npz") if v is None else run / v / "sessions" / f"{js.stem}.npz"
                if f.exists():
                    zz = np.load(f)
                    if ke in zz:
                        sse[(js.stem, q, me)] = pixel_sse(spatial[an], zz[ke], keep)
        meta.append({"label": js.stem, "animal": an, "epoch": m["epoch"]})
    return pd.DataFrame(meta), sse, sst, (int(shape[0]), int(shape[1])), flat[pstat]


def value(layout, q, v, e_q, e_joint, s):
    if v is None or layout == "alone":
        return 1.0 - e_q / s
    return (e_q - e_joint) / s


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from combined_encoding_figures import _outline
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layout", choices=tuple(QS), default="alone")
    ap.add_argument("--window", choices=("cue", "lick"), default="cue")
    ap.add_argument("--measures", nargs="+", default=["avg", "trial"], choices=tuple(MLAB))
    ap.add_argument("--epochs", nargs="+", default=["acute", "chronic"], choices=EPS)
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    qs = [(q, v) for q, v in QS[a.layout] if v is None or (run / v / "sessions").is_dir()]
    missing = [v for _q, v in QS[a.layout] if v is not None and not (run / v / "sessions").is_dir()]
    if missing:
        print("variants absent from the run (skipped):", missing, flush=True)
    meta, sse, sst, (H, W), pix = collect(run, qs, a.measures, a.window)
    meas = [me for me in a.measures if any(k[1] == me for k in sst)]
    if not meas:
        raise SystemExit(f"no {a.measures} moments in {run} (FROZEN_PIX_MUSALL=1 run needed for avg / trial)")
    qs = [(q, v) for q, v in qs if any((lab, q, me) in sse for lab in meta.label for me in meas)]
    mask = block_mask()

    def paint(vec):
        g = np.full(H * W, np.nan)
        g[pix] = vec
        return np.where(mask, g.reshape(H, W)[::2, ::2], np.nan)

    def sess_map(lab, q, v, me):
        k = (lab, q, me)
        if k not in sse or (lab, me) not in sst or (v is not None and a.layout != "alone"
                                                     and (lab, "full", me) not in sse):
            return None
        s = np.where(sst[(lab, me)] > 0, sst[(lab, me)], np.nan)
        return paint(value(a.layout, q, v, sse[k], sse.get((lab, "full", me)), s))

    # per animal x epoch pooled (sums before the ratio) -> table + display maps
    rows, pooled = [], {}
    for (an, ep), g in meta.groupby(["animal", "epoch"]):
        for me in meas:
            labs = [x for x in g.label if (x, me) in sst]
            if not labs:
                continue
            S = sum(sst[(x, me)] for x in labs)
            S = np.where(S > 0, S, np.nan)
            for q, v in qs:
                ok = [x for x in labs if (x, q, me) in sse and (x, "full", me) in sse]
                if len(ok) != len(labs):
                    continue
                Eq = sum(sse[(x, q, me)] for x in labs)
                Ej = sum(sse[(x, "full", me)] for x in labs)
                val = value(a.layout, q, v, Eq, Ej, S)
                pooled[(an, ep, q, me)] = paint(val)
                rows.append({"animal": an, "epoch": ep, "measure": me, "q": q, "n_sessions": len(labs),
                             "median_px": float(np.nanmedian(val)), "p95_px": float(np.nanpercentile(val, 95))})
    T = pd.DataFrame(rows)
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    win = "allframes" if meas == ["all"] else a.window                   # the window applies to PETH / trial only
    stem = f"pixel_musall_{a.layout}_{win}"
    T.to_csv(run / f"{stem}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(T.pivot_table(index=["measure", "q"], columns=["animal", "epoch"], values="p95_px").round(3)
              .reindex(columns=["pre", *a.epochs], level=1), flush=True)        # rule 8: per animal first

    def by(ep, q, v, me):
        d = {}
        for _, r in meta[meta.epoch == ep].iterrows():
            mp = sess_map(r.label, q, v, me)
            if mp is not None:
                d.setdefault(r.animal, []).append(mp)
        return d

    animals = sorted(meta.animal.unique())
    pics, srows = {}, []
    for q, v in qs:
        for me in meas:
            lv = [pooled[(an, "pre", q, me)] for an in animals if (an, "pre", q, me) in pooled]
            pics[(q, me, "pre")] = (np.nanmean(lv, 0) if lv else None, None)
            pre = by("pre", q, v, me)
            for ep in a.epochs:
                com = [an for an in animals if (an, "pre", q, me) in pooled and (an, ep, q, me) in pooled]
                if not com:
                    pics[(q, me, ep)] = (None, None)
                    continue
                mm = np.nanmean([pooled[(an, ep, q, me)] - pooled[(an, "pre", q, me)] for an in com], 0)
                post = by(ep, q, v, me)
                pb, qb = {x: pre[x] for x in com if x in pre}, {x: post[x] for x in com if x in post}
                sig, n = test(pb, qb, FACTORS[16], mask)
                f = FACTORS[16]
                up = dn = 0
                if sig is not None and sig.any():
                    hh, ww = sig.shape
                    sm = np.nanmean(mm[:hh * f, :ww * f].reshape(hh, f, ww, f), (1, 3))
                    up, dn = int((sig & (sm > 0)).sum()), int((sig & (sm < 0)).sum())
                srows.append({"q": q, "measure": me, "epoch": ep, "up": up, "down": dn, "n_bins": n,
                              "animals": "/".join(com)})
                pics[(q, me, ep)] = (mm, sig)
        print(q, [(r["measure"], r["epoch"], r["up"], r["down"]) for r in srows if r["q"] == q], flush=True)
    Sg = pd.DataFrame(srows)
    Sg.to_csv(run / f"{stem}_sig.csv", index=False)

    ol = _outline(meta[["label"]])
    rws = [(me, "pre") for me in meas] + [(me, ep) for ep in a.epochs for me in meas]
    fig, axs = plt.subplots(len(rws), len(qs), figsize=(2.2 * len(qs) + 0.6, 2.25 * len(rws) + 1.3), squeeze=False)
    for j, (q, _v) in enumerate(qs):
        lims = {}
        for me in meas:
            ds = [pics[(q, me, ep)][0] for ep in a.epochs if pics[(q, me, ep)][0] is not None]
            lims[me] = max([np.nanpercentile(np.abs(d), 99) for d in ds] + [1e-6])
        for r, (me, ep) in enumerate(rws):
            ax = axs[r, j]
            M, sig = pics[(q, me, ep)]
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{MLAB[me]}\n{'pre (level)' if ep == 'pre' else ep + ' - pre'}", fontsize=8)
            if r == 0:
                ax.set_title(textwrap.fill(q, 16), fontsize=8.5)
            if M is None:
                continue
            if ep == "pre":
                im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(np.nanpercentile(M, 99), 1e-6),
                               interpolation="nearest")
            else:
                im = ax.imshow(M, cmap="RdBu_r", vmin=-lims[me], vmax=lims[me], interpolation="nearest")
                if sig is not None and sig.any():
                    f = FACTORS[16]
                    big = np.zeros(M.shape)
                    hh, ww = min(sig.shape[0] * f, M.shape[0]), min(sig.shape[1] * f, M.shape[1])
                    big[:hh, :ww] = np.kron(sig, np.ones((f, f)))[:hh, :ww]
                    ax.contour(big, levels=[0.5], colors="w", linewidths=2.2)       # white halo
                    ax.contour(big, levels=[0.5], colors="k", linewidths=0.9)
                rr = Sg[(Sg.q == q) & (Sg.measure == me) & (Sg.epoch == ep)]
                if len(rr):
                    ax.text(0.02, 0.02, f"{rr.iloc[0].up}+/{rr.iloc[0].down}-", transform=ax.transAxes, fontsize=6.5,
                            bbox={"facecolor": "white", "alpha": 0.7, "pad": 1, "lw": 0})
            if ol is not None and ol.shape == M.shape:
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            cax = ax.inset_axes([0.62, 0.9, 0.36, 0.05])                      # inset colour bar, off the brain
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
            cb.locator = matplotlib.ticker.MaxNLocator(2)
            cb.update_ticks()
    what = {"alone": "single-group ALONE cvR2 (Musall top row)", "unique": "single-group UNIQUE dR2",
            "novar": "single-VARIABLE UNIQUE dR2 (Musall bottom row)"}[a.layout]
    ttl = (f"NEURAL: frozen-frame PIXEL maps, {what}, {win if win == 'allframes' else win + ' window'} ({', '.join(MLAB[m] for m in meas)}), run {a.tag}. "
           f"Pre level = held-out; epoch - pre = frozen all-pre betas; mean of animals (SSE, SST summed over each animal's "
           f"sessions). Outlines: significant 16-px bins (nested bootstrap animals > sessions, Bonferroni); +/- = bins up / "
           f"down. Grey = Allen areas. Image left = ipsilesional.")
    fig.suptitle("\n".join(textwrap.wrap(ttl, int(24 * len(qs)))), fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.995, top=1 - 1.0 / (2.25 * len(rws) + 1.3), bottom=0.01,
                        wspace=0.05, hspace=0.08)
    p = out / f"{stem}.png"
    fig.savefig(p, dpi=105)
    plt.close(fig)
    print("->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

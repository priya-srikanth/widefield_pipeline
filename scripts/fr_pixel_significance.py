"""SIGNIFICANCE of the Musall-style pixel maps with FaceRhythm (Priya 10-25: "can we test significance on these maps?").

Per-session maps = held-out R2 (fraction, 0-1) per 2 x 2-px block (`fr_musall.session_pixels`: the per-session encoder's stored
block SSE maps, main no-FR run + FR run). Test = the repo's map test for epoch changes
(`beta_maps.hierarchical_bootstrap_significance`: animals -> sessions nested bootstrap of epoch - pre, bootstrap-SE
interval, Bonferroni over bins), on the RAW R2 scale (R2 is already scale-free, so no per-animal pre normalisation).
Bins: 16 and 32 FULL-resolution px (= 8 and 16 block px on this half-resolution grid), stat mask = the eroded brain
mask (`beta_maps.stat_mask`) block-downsampled. Calibration as for the residual maps: random half-splits of each
animal's pre sessions (pseudo pre vs pseudo post) -> share of null splits with >= 1 significant bin.

Usage: python scripts/fr_pixel_significance.py [--mode A] [--windows lick cue] [--n-null 50]
Output: <E>/fr_encoding_summary/pixel_significance/ (sig_bins.csv, null.csv, sig_<q>_<w>_<F>px.npz, figures/)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E, session_pixels, table  # noqa: E402

QS = ["full_fr", "unique_fr", "alone_uninstructed", "uniquefr_uninstructed", "uniquefr_task", "uniquefr_position",
      "unique_position"]
QLAB = {"full_fr": "full R2 (with FR)", "unique_fr": "FaceRhythm unique", "alone_uninstructed": "uninstructed alone",
        "uniquefr_uninstructed": "uninstructed unique | FR", "uniquefr_task": "task unique | FR",
        "uniquefr_position": "position unique | FR", "unique_position": "position unique, no FR"}
EPS = ["acute", "subacute", "chronic"]
FACTORS = {16: 8, 32: 16}            # full-resolution px -> block px on the 270 x 320 grid


def _load(args):
    lab, mode, windows = args
    return lab, session_pixels(lab, mode, windows)


def block_mask():
    from wfield_local import beta_maps as bm
    m = bm.stat_mask().astype(float)
    return m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).mean((1, 3)) >= 0.5


def test(pre_by, post_by, f, mask, n_boot=2000):
    from wfield_local import beta_maps as bm
    got = bm.hierarchical_bootstrap_significance(pre_by, post_by, factor=f, n_boot=n_boot, mask=mask)
    if got is None:
        return None, 0
    return got[0], int(got[4])


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="A")
    ap.add_argument("--windows", nargs="+", default=["lick", "cue"])
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--zscore", action="store_true",
                    help="PER-ANIMAL z: every map of an animal divided by its pre-session SD map (floored at its median "
                         "pixel SD) before the test (Priya 10-25); output pixel_significance_z/")
    a = ap.parse_args(argv)
    out = E / "fr_encoding_summary" / ("pixel_significance_z" if a.zscore else "pixel_significance")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    meta = table(a.mode, a.windows)[["label", "animal", "epoch"]].drop_duplicates().sort_values("label")
    got = Parallel(n_jobs=8)(delayed(_load)((lab, a.mode, a.windows)) for lab in meta.label)
    maps = dict(got)
    mask = block_mask()
    print(f"{len(maps)} sessions loaded; stat mask {mask.sum()} block px", flush=True)

    def by(ep, q, w, z=None):
        d = {}
        for _, m in meta[meta.epoch == ep].iterrows():
            d.setdefault(m.animal, []).append(np.where(mask, maps[m.label][(q, w)], np.nan))
        if z is not None:                                  # per-animal z units (same SD for pre and post)
            d = {an: [x / z[an] for x in v] for an, v in d.items() if an in z}
        return d

    def zsd(q, w):
        """{animal: pre-session SD map, floored at its median} (None unless --zscore)."""
        if not a.zscore:
            return None
        out_ = {}
        for an, v in by("pre", q, w).items():
            sd = np.nanstd(np.stack(v), 0, ddof=1)
            out_[an] = np.maximum(sd, np.nanmedian(sd))
        return out_

    def job(q, w):
        res, pics = [], {}
        z = zsd(q, w)
        pre = by("pre", q, w, z)
        for ep in EPS:
            post = by(ep, q, w, z)
            com = sorted(set(pre) & set(post))
            pb, qb = {x: pre[x] for x in com}, {x: post[x] for x in com}
            mm = np.nanmean([np.nanmean(qb[x], 0) - np.nanmean(pb[x], 0) for x in com], 0)
            for F, f in FACTORS.items():
                sig, n = test(pb, qb, f, mask)
                k = 0 if sig is None else int(sig.sum())
                up = dn = 0
                if k:
                    hh, ww = sig.shape
                    sm = np.nanmean(mm[:hh * f, :ww * f].reshape(hh, f, ww, f), (1, 3))
                    up, dn = int((sig & (sm > 0)).sum()), int((sig & (sm < 0)).sum())
                res.append(dict(q=q, window=w, epoch=ep, px=F, sig=k, up=up, down=dn, n_bins=n, n_animals=len(com),
                                sessions="/".join(str(len(qb[x])) for x in com)))
                pics[(ep, F)] = (mm, sig)
                np.savez_compressed(out / f"sig_{q}_{w}_{ep}_{F}px.npz", mean=mm.astype(np.float32),
                                    sig=np.zeros((1, 1), bool) if sig is None else sig)
        print(q, w, [(r["epoch"], r["px"], r["sig"], r["up"], r["down"]) for r in res], flush=True)
        return res, pics

    R = [job(q, w) for q in QS for w in a.windows]
    rows = [r for x, _ in R for r in x]
    pd.DataFrame(rows).to_csv(out / "sig_bins.csv", index=False)
    pics = {(q, w): p for (q, w), (_, p) in zip([(q, w) for q in QS for w in a.windows], R)}

    # null: random half-splits of pre
    def null(q, w, i):
        rng = np.random.default_rng(1000 + i)
        pre = by("pre", q, w, zsd(q, w))
        pb, qb = {}, {}
        for an, p in pre.items():
            idx = rng.permutation(len(p))
            h = len(p) // 2
            pb[an] = [p[j] for j in idx[:h]]
            qb[an] = [p[j] for j in idx[h:]]
        rows_ = []
        for F, f in FACTORS.items():
            sig = test(pb, qb, f, mask)[0]
            rows_.append((q, w, i, F, 0 if sig is None else int(sig.sum())))
        return rows_
    N = [r for q in QS for w in a.windows for i in range(a.n_null) for r in null(q, w, i)]
    N = pd.DataFrame(N, columns=["q", "window", "split", "px", "false_bins"])
    N.to_csv(out / "null.csv", index=False)
    print(N.groupby(["px", "window"]).false_bins.agg(mean="mean", any_rate=lambda x: (x > 0).mean()).round(3), flush=True)

    # figure per window: rows = epochs, columns = quantities; 16 px contours black, 32 px dashed purple
    from combined_encoding_figures import _outline
    ol = _outline(meta)
    S = pd.DataFrame(rows)
    for w in a.windows:
        fig, axs = plt.subplots(3, len(QS), figsize=(2.35 * len(QS), 8.4), squeeze=False)
        for j, q in enumerate(QS):
            lim = max(np.nanpercentile(np.abs(pics[(q, w)][(ep, 16)][0]), 99) for ep in EPS)
            for i, ep in enumerate(EPS):
                ax = axs[i, j]
                mm = pics[(q, w)][(ep, 16)][0]
                im = ax.imshow(mm, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                if ol is not None and ol.shape == mm.shape:
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                txt = []
                for F, col, ls in ((16, "k", "solid"), (32, "#7a0177", "dashed")):
                    sig = pics[(q, w)][(ep, F)][1]
                    f = FACTORS[F]
                    if sig is not None and sig.any():
                        big = np.zeros(mm.shape)
                        big[:sig.shape[0] * f, :sig.shape[1] * f] = np.kron(sig, np.ones((f, f)))
                        ax.contour(big, levels=[0.5], colors=col, linewidths=1.0, linestyles=ls)
                    r = S[(S.q == q) & (S.window == w) & (S.epoch == ep) & (S.px == F)].iloc[0]
                    txt.append(f"{F}px {r.up}+/{r.down}- of {r.n_bins}")
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                ax.set_title(("" if i else QLAB[q] + "\n") + "\n".join(txt), fontsize=7)
                if j == 0:
                    ax.set_ylabel(f"{ep} minus pre", fontsize=9)
                if i == 2:
                    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
                    cb.ax.tick_params(labelsize=6)
        nr = N[N.window == w].groupby("px").false_bins.apply(lambda x: (x > 0).mean())
        fig.suptitle(f"NEURAL: significance of the FaceRhythm-encoder pixel maps, {w} window: epoch minus pre, " + ("PER-ANIMAL z (each animal / its pre-session SD map)" if a.zscore else "held-out R2 (fraction, 0-1)") + ", "
                     "mean of animals. Contours: black = significant 16 px bins, purple dashed = 32 px bins\n(nested bootstrap "
                     f"animals -> sessions, Bonferroni over bins). Null (random pre half-splits): >= 1 false bin in "
                     f"{nr.get(16, np.nan):.0%} (16 px) / {nr.get(32, np.nan):.0%} (32 px) of splits. Image left = ipsilesional.",
                     fontsize=8.5)
        fig.subplots_adjust(left=0.03, right=0.995, top=0.86, bottom=0.04, wspace=0.06, hspace=0.3)
        fig.savefig(out / "figures" / f"fr_pixel_significance_{w}_mode{a.mode}.png", dpi=115)
        plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

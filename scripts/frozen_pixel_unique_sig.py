"""FROZEN-frame unique R2 PIXEL maps WITH statistics (Priya 10-25: "make the frozen model the headline ... and make the pixel
maps ... with stats"; per-animal z-scored version too).

From a spatial-target frozen run (`frozen_encoder --target basis`; residual / data second moments per session saved in the
basis): per SESSION and pixel, R2 = 1 - SSE[joint] / SST and unique R2 of g = (SSE[no_g] - SSE[joint]) / SST (all frames;
pre held out leave-one-session-out, post with the frozen all-pre betas), painted on the 2 x 2-px block grid. Epoch - pre:
mean of animals' (epoch mean - pre mean); test = `fr_pixel_significance.test` (nested bootstrap animals -> sessions,
Bonferroni over 16 / 32 px bins) with a null of random pre half-splits. --zscore: every map of an animal divided by its
pre-session SD map (floored at its median pixel SD) before the test (per-animal z). Image left = ipsilesional.

Usage: python scripts/frozen_pixel_unique_sig.py --tag gate-attempted_v7g3_t3d_frA_basis [--zscore] [--n-null 20]
Output: <E>/frozen_encoder_<tag>/figures/unique_sig{_z}.png + unique_sig{_z}.csv
"""
from __future__ import annotations

import argparse
import json
import textwrap
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402
from fr_pixel_significance import FACTORS, block_mask, test  # noqa: E402
from frozen_r2_maps import pixel_sse  # noqa: E402

QS = [("full", None, "full R2"), ("target", "no_target", "position unique (target)"),
      ("spout", "no_spout", "spout unique (target + executed spout part)"), ("task", "no_task", "task unique"),
      ("movement", "no_movement", "movement unique (all)"), ("fr", "no_fr_all", "FaceRhythm unique"),
      ("fr_dir", "no_fr_dir", "FaceRhythm direction unique"),
      ("posinfo", "no_posinfo", "movement WITH lick-position info (pose + FR)"),
      ("noposinfo", "no_noposinfo", "movement WITHOUT lick-position info")]
EPS = ["acute", "subacute", "chronic"]


def per_animal_counts(pre_by: dict, post_by: dict, f: int, mask: np.ndarray, alpha: float = 0.05):
    """SECONDARY view (Priya 10-25: nested bootstrap stays primary): per animal, its own sessions as the unit (the repo's
    "musall" logic: Welch t per bin, Bonferroni over that animal's bins); returns (n_up, n_down) maps on the bin grid =
    how many animals change significantly in each direction, and the animals tested."""
    from scipy import stats

    from wfield_local import beta_maps as bm
    up = dn = None
    tested = []
    for an in sorted(set(pre_by) & set(post_by)):
        if len(pre_by[an]) < 2 or len(post_by[an]) < 2:
            continue
        P = np.stack([bm.downsample(m, f, mask)[0] for m in pre_by[an]])
        Q = np.stack([bm.downsample(m, f, mask)[0] for m in post_by[an]])
        ok = np.isfinite(P).all(0) & np.isfinite(Q).all(0)
        t, pv = stats.ttest_ind(Q, P, axis=0, equal_var=False)
        sig = ok & (pv < alpha / max(int(ok.sum()), 1))
        d = np.nanmean(Q, 0) - np.nanmean(P, 0)
        u, w = (sig & (d > 0)).astype(int), (sig & (d < 0)).astype(int)
        up = u if up is None else up + u
        dn = w if dn is None else dn + w
        tested.append(an)
    return up, dn, tested


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from combined_encoding_figures import _outline
    from wfield_local import combined_encoding as ce
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--n-null", type=int, default=20)
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    qs = [q for q in QS if q[1] is None or (run / q[1] / "sessions").is_dir()]
    sess = sorted(js for js in (run / "joint" / "sessions").glob("*.json") if js.stem[-4:] >= "0801")
    _V, _ft, _Ub, flat, _pa, _nm, shape, pstat = ce.load_imaging(sess[0].stem)
    H, W = (int(x) for x in shape)
    pix = flat[pstat]
    spatial, meta, maps = {}, [], {}
    for js in sess:
        m = json.loads(js.read_text(encoding="utf-8"))
        an = m["animal"]
        if an not in spatial:
            spatial[an] = pickle.loads((run / "fit" / f"spatial_{an}.pkl").read_bytes())
        keep = np.array([t != "cortex" for t in m["targets"]])
        z0 = np.load(js.with_suffix(".npz"))
        sst = pixel_sse(spatial[an], z0["data_mom"], keep)
        sst = np.where(sst > 0, sst, np.nan)
        e0 = pixel_sse(spatial[an], z0["resid_mom"], keep)
        for q, v, _l in qs:
            if v is None:
                val = 1.0 - e0 / sst
            else:
                ev = pixel_sse(spatial[an], np.load(run / v / "sessions" / f"{js.stem}.npz")["resid_mom"], keep)
                val = (ev - e0) / sst
            g = np.full(H * W, np.nan)
            g[pix] = val
            maps[(js.stem, q)] = g.reshape(H, W)[::2, ::2]
        meta.append({"label": js.stem, "animal": an, "epoch": m["epoch"]})
    meta = pd.DataFrame(meta)
    mask = block_mask()
    print(f"{len(meta)} sessions, {len(qs)} quantities; stat mask {mask.sum()} block px", flush=True)

    def by(ep, q, z=None):
        d = {}
        for _, r in meta[meta.epoch == ep].iterrows():
            d.setdefault(r.animal, []).append(np.where(mask, maps[(r.label, q)], np.nan))
        if z is not None:
            d = {an: [x / z[an] for x in v] for an, v in d.items() if an in z}
        return d

    def zsd(q):
        if not a.zscore:
            return None
        out = {}
        for an, v in by("pre", q).items():
            sd = np.nanstd(np.stack(v), 0, ddof=1)
            out[an] = np.maximum(sd, np.nanmedian(sd))
        return out

    rows, pics, pa_counts = [], {}, {}
    for q, _v, _l in qs:
        z = zsd(q)
        pre = by("pre", q, z)
        pics[(q, "pre")] = np.nanmean([np.nanmean(by("pre", q)[x], 0) for x in by("pre", q)], 0)
        for ep in EPS:
            post = by(ep, q, z)
            com = sorted(set(pre) & set(post))
            pb, qb = {x: pre[x] for x in com}, {x: post[x] for x in com}
            raw_pre, raw_post = by("pre", q), by(ep, q)
            mm = np.nanmean([np.nanmean(raw_post[x], 0) - np.nanmean(raw_pre[x], 0) for x in com], 0)
            sigs = {}
            for F, f in FACTORS.items():
                sig, n = test(pb, qb, f, mask)
                sigs[F] = sig
                k = 0 if sig is None else int(sig.sum())
                up = dn = 0
                if k:
                    hh, ww = sig.shape
                    sm = np.nanmean(mm[:hh * f, :ww * f].reshape(hh, f, ww, f), (1, 3))
                    up, dn = int((sig & (sm > 0)).sum()), int((sig & (sm < 0)).sum())
                rows.append({"q": q, "epoch": ep, "px": F, "sig": k, "up": up, "down": dn, "n_bins": n})
            pics[(q, ep)] = (mm, sigs)
            pa_counts[(q, ep)] = per_animal_counts(raw_pre, raw_post, FACTORS[16], mask)
        print(q, [(r["epoch"], r["px"], r["up"], r["down"]) for r in rows if r["q"] == q], flush=True)
    S = pd.DataFrame(rows)
    sfx = "_z" if a.zscore else ""
    S.to_csv(run / f"unique_sig{sfx}.csv", index=False)
    # null: random pre half-splits
    nul = []
    for i in range(a.n_null):
        rng = np.random.default_rng(1000 + i)
        for q, _v, _l in qs:
            pre = by("pre", q, zsd(q))
            pb, qb = {}, {}
            for an, p in pre.items():
                idx = rng.permutation(len(p))
                pb[an], qb[an] = [p[j] for j in idx[:len(p) // 2]], [p[j] for j in idx[len(p) // 2:]]
            for F, f in FACTORS.items():
                sig = test(pb, qb, f, mask)[0]
                nul.append({"split": i, "q": q, "px": F, "false_bins": 0 if sig is None else int(sig.sum())})
    N = pd.DataFrame(nul)
    N.to_csv(run / f"unique_sig_null{sfx}.csv", index=False)
    nr = N.groupby("px").false_bins.apply(lambda x: (x > 0).mean())
    print("null: share of splits with >= 1 false bin", nr.round(3).to_dict(), flush=True)

    ol = _outline(meta[["label"]])
    rows_ = [("pre (level)", "pre")] + [(f"{ep} minus pre", ep) for ep in EPS]
    fig, axs = plt.subplots(len(rows_), len(qs), figsize=(2.35 * len(qs), 2.5 * len(rows_) + 1.0), squeeze=False)
    for j, (q, _v, lab) in enumerate(qs):
        lim = max(np.nanpercentile(np.abs(pics[(q, ep)][0]), 99) for ep in EPS)
        for r, (title, ep) in enumerate(rows_):
            ax = axs[r, j]
            if ep == "pre":
                M = pics[(q, "pre")]
                im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(np.nanpercentile(M, 99), 1e-6), interpolation="nearest")
            else:
                M, sigs = pics[(q, ep)]
                im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                txt = []
                for F, col, ls in ((16, "k", "solid"), (32, "#7a0177", "dashed")):
                    sig, f = sigs[F], FACTORS[F]
                    if sig is not None and sig.any():
                        big = np.zeros(M.shape)
                        hh, ww = min(sig.shape[0] * f, M.shape[0]), min(sig.shape[1] * f, M.shape[1])
                        big[:hh, :ww] = np.kron(sig, np.ones((f, f)))[:hh, :ww]
                        ax.contour(big, levels=[0.5], colors=col, linewidths=0.9, linestyles=ls)
                    rr = S[(S.q == q) & (S.epoch == ep) & (S.px == F)].iloc[0]
                    txt.append(f"{F}px {rr.up}+/{rr.down}-")
                ax.text(0.02, 0.02, "  ".join(txt), transform=ax.transAxes, fontsize=6.5, va="bottom",
                        bbox={"facecolor": "white", "alpha": 0.7, "pad": 1, "lw": 0})
            if ol is not None and ol.shape == M.shape:
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if r == 0:
                ax.set_title(textwrap.fill(lab, 24), fontsize=8)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
            if r in (0, len(rows_) - 1):
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, orientation="horizontal")
                cb.ax.tick_params(labelsize=6)
    unit = "PER-ANIMAL z test (each animal / its pre-session SD map); maps shown in R2" if a.zscore else "raw R2 test"
    ttl = (f"NEURAL: FROZEN-frame pixel maps WITH significance, run {a.tag} (all frames; held-out R2, fraction). Row 1 pre "
           f"level; rows 2-4 epoch minus pre (one scale per column). Contours: significant bins, black 16 px, purple dashed 32 px "
           f"(nested bootstrap animals > sessions, Bonferroni; {unit}); null: >= 1 false bin in {nr.get(16, np.nan):.0%} / "
           f"{nr.get(32, np.nan):.0%} of random pre half-splits. Image left = ipsilesional.")
    fig.suptitle("\n".join(textwrap.wrap(ttl, int(21 * len(qs)))), fontsize=9)       # wrapped to the figure width
    fig.subplots_adjust(left=0.03, right=0.995, top=0.86, bottom=0.03, wspace=0.08, hspace=0.14)
    p = run / "figures" / f"unique_sig{sfx}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=105)
    plt.close(fig)
    print("->", p)
    # secondary: animals per bin (16 px), up - down, per quantity x epoch
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cm = ListedColormap(["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"])
    nb = BoundaryNorm(np.arange(-3.5, 4.5, 1), cm.N)
    fig, axs = plt.subplots(len(EPS), len(qs), figsize=(2.7 * len(qs) + 1.0, 2.6 * len(EPS) + 1.6), squeeze=False)
    crow = []
    for j, (q, _v, lab) in enumerate(qs):
        for r, ep in enumerate(EPS):
            ax = axs[r, j]
            up, dn, tested = pa_counts[(q, ep)]
            if up is None:
                ax.axis("off")
                continue
            net = np.where(np.isfinite(pics[(q, ep)][0][:up.shape[0] * 8:8, :up.shape[1] * 8:8]), up - dn, np.nan)
            im = ax.imshow(np.clip(net, -3, 3), cmap=cm, norm=nb, interpolation="nearest")
            n3u, n3d = int((up >= 3).sum()), int((dn >= 3).sum())
            crow.append({"q": q, "epoch": ep, "bins_3plus_up": n3u, "bins_3plus_down": n3d, "animals": "/".join(tested)})
            ax.text(0.02, 0.02, f">=3 animals: {n3u} up / {n3d} down", transform=ax.transAxes, fontsize=6.5,
                    bbox={"facecolor": "white", "alpha": 0.7, "pad": 1, "lw": 0})
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(textwrap.fill(lab, 24), fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{ep} minus pre", fontsize=9)
    fig.subplots_adjust(left=0.04, right=0.9, top=0.8, bottom=0.03, wspace=0.06, hspace=0.3)
    cb = fig.colorbar(im, cax=fig.add_axes([0.92, 0.2, 0.012, 0.55]), ticks=range(-3, 4))
    cb.set_label("animals significantly UP minus DOWN\n(per animal: sessions as unit, Welch t, Bonferroni)", fontsize=8)
    fig.suptitle(textwrap.fill(f"NEURAL: FROZEN-frame pixel maps, PER-ANIMAL view (secondary; the nested bootstrap is primary), "
                               f"run {a.tag}, 16 px bins: how many of the 4 animals change significantly in each direction, "
                               "each tested on its own sessions. Image left = ipsilesional.", int(21 * len(qs))), fontsize=9)
    fig.savefig(run / "figures" / "unique_sig_per_animal.png", dpi=105)
    plt.close(fig)
    pd.DataFrame(crow).to_csv(run / "unique_sig_per_animal.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

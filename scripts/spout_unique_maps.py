"""PER-SPOUT UNIQUE-CONTRIBUTION pixel maps, TARGET vs EXECUTION (Priya 10-25: "we want the unique contribution spout
target and execution maps"). From the frozen pixel (shared-SVD basis) run's drop-one-spout fits (`dropspout/sessions/
*_pix.npz`, saved per session on spout q's OWN trial frames; base = joint):
  target unique of q    (SSE[base|drop:q] - SSE[base]) / SST_q        q's target response folded into the others' mean
  executed unique of q  (SSE[base|drop:q whole] - SSE[base|drop:q]) / SST_q   the further loss when q's spout-locked
                        movement templates are ALSO swapped for the other spouts' (the executed movement's spout part)
Rows: target pre level, acute / chronic minus pre; executed the same. Columns: spouts in the canonical order. Change maps:
within animal, mean of 4; contours = significant 16 px bins (nested bootstrap animals -> sessions, Bonferroni; the same
test as `frozen_pixel_unique_sig.py`), white halo under black. Target and the executed spout-locked part follow the spout
together, so the split is credited by the joint ridge (the target-first / executed-first orderings bracket it).
Image left = ipsilesional; half resolution.

Usage: python scripts/spout_unique_maps.py [--tag gate-attempted_v7g3_t3d_basis]
Output: <E>/frozen_encoder_<tag>/figures/spout_unique_maps.png, spout_unique_sig.csv
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

POS_ORDER = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]   # canonical
EPS = ["acute", "chronic"]
F = 16


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")

    from wfield_local import combined_encoding as ce
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_basis")
    ap.add_argument("--base", default="joint")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    dd = run / "dropspout" / "sessions"
    pj = sorted(p for p in dd.glob("*_pix.json") if p.stem.split("_pix")[0][-4:] >= "0801")
    lab0 = pj[0].stem.split("_pix")[0]
    _V, _ft, _Ub, flat, _pa, _nm, shape, pstat = ce.load_imaging(lab0)
    H, W = (int(x) for x in shape)
    pix = flat[pstat]
    cache = run / f"spout_unique_cache_{a.base}.npz"
    spatial, meta, maps = {}, [], {}
    if cache.exists():
        zc = np.load(cache, allow_pickle=True)
        meta = pd.DataFrame(list(zc["meta"]))
        maps = {tuple(str(k).split("|")): m for k, m in zip(zc["keys"], zc["maps"])}
        pj = []
    for p in pj:
        m = json.loads(p.read_text(encoding="utf-8"))
        an, lab = m["animal"], p.stem.split("_pix")[0]
        sp = spatial.setdefault(an, pickle.loads((run / "fit" / f"spatial_{an}.pkl").read_bytes()))
        keep = np.array([t != "cortex" for t in m["targets"]])
        z = np.load(p.with_suffix(".npz"))
        for q in POS_ORDER:
            kb, kd, kw = f"{a.base}|{q}|{q}|base", f"{a.base}|{q}|{q}|drop", f"{a.base}|{q}|{q}|whole"
            if not {kb, kd, kw, f"data|{q}"} <= set(z.files):
                continue
            sst = pixel_sse(sp, z[f"data|{q}"], keep)
            sst = np.where(sst > 0, sst, np.nan)
            eb, ed, ew = (pixel_sse(sp, z[k], keep) for k in (kb, kd, kw))
            for name, val in (("target", (ed - eb) / sst), ("executed", (ew - ed) / sst)):
                g = np.full(H * W, np.nan)
                g[pix] = val
                maps[(lab, name, q)] = g.reshape(H, W)[::2, ::2]
        meta.append({"label": lab, "animal": an, "epoch": m["epoch"]})
    if pj:
        np.savez_compressed(cache, keys=np.array(["|".join(k) for k in maps]), maps=np.stack(list(maps.values())).astype(np.float32),
                            meta=np.array(meta, dtype=object))
    meta = pd.DataFrame(meta)
    figures(run, a, meta, maps, block_mask())
    return 0


EP4 = ["pre", "acute", "subacute", "chronic"]
PARTS = ("target", "executed")


def figures(run, a, meta, maps, mask) -> None:
    """(1) LEVELS per epoch, (2) CHANGES vs pre with significance, (3) PER ANIMAL with each animal's own colour bar.
    Means across animals use ANIMAL-NORMALISED maps (each animal / its own pre-stroke 99th percentile over spouts of that
    part), so one animal cannot set the picture (PS94's target unique is ~10x the others'); tests use the raw maps."""
    import matplotlib.pyplot as plt

    from combined_encoding_figures import _outline
    f = FACTORS[F]
    ol = _outline(meta[["label"]])

    def by(ep, name, q):
        d = {}
        for _, r in meta[meta.epoch == ep].iterrows():
            if (r.label, name, q) in maps:
                d.setdefault(r.animal, []).append(np.where(mask, maps[(r.label, name, q)], np.nan))
        return d

    def animal_sig(pre_list, post_list, alpha=0.05):
        """Within ONE animal, its sessions as the unit (as `frozen_pixel_unique_sig.per_animal_counts`): Welch t per 16 px
        bin, Bonferroni over the animal's bins -> (sig, post - pre) on the bin grid, or (None, None)."""
        from scipy import stats

        from wfield_local import beta_maps as bm
        if len(pre_list) < 2 or len(post_list) < 2:
            return None, None
        Pm = np.stack([bm.downsample(m, f, mask)[0] for m in pre_list])
        Qm = np.stack([bm.downsample(m, f, mask)[0] for m in post_list])
        ok = np.isfinite(Pm).all(0) & np.isfinite(Qm).all(0)
        _t, pv = stats.ttest_ind(Qm, Pm, axis=0, equal_var=False)
        return ok & (pv < alpha / max(int(ok.sum()), 1)), np.nanmean(Qm, 0) - np.nanmean(Pm, 0)

    # per-animal epoch means and normalising scale
    lev = {(name, ep, q): {an: np.nanmean(v, 0) for an, v in by(ep, name, q).items()} for name in PARTS for ep in EP4
           for q in POS_ORDER}
    scale = {}
    for name in PARTS:
        for an in sorted(meta.animal.unique()):
            st = [lev[(name, "pre", q)][an] for q in POS_ORDER if an in lev[(name, "pre", q)]]
            scale[(name, an)] = np.nanpercentile(np.stack(st), 99) if st else np.nan
    print("normalising scales (99th pct of pre level):", {k: f"{v:.4f}" for k, v in scale.items()}, flush=True)

    def nmean(name, ep, q, minus_pre=False):
        out = []
        for an, M in lev[(name, ep, q)].items():
            if minus_pre and an not in lev[(name, "pre", q)]:
                continue
            x = M - lev[(name, "pre", q)][an] if minus_pre else M
            out.append(x / scale[(name, an)])
        return np.nanmean(out, 0) if out else np.full(next(iter(maps.values())).shape, np.nan)

    def deco(ax, M):
        if ol is not None and ol.shape == M.shape:
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)

    def title(fig, txt, width=150):
        fig.suptitle("\n".join(textwrap.wrap(txt, width)), fontsize=9)

    nq = len(POS_ORDER)
    # (1) LEVELS per epoch
    fig, axs = plt.subplots(2 * len(EP4), nq, figsize=(2.4 * nq + 0.8, 2.3 * 2 * len(EP4) + 1.0), squeeze=False)
    for pi, name in enumerate(PARTS):
        vmax = np.nanpercentile(np.stack([nmean(name, ep, q) for ep in EP4 for q in POS_ORDER]), 99) or 1.0
        for ei, ep in enumerate(EP4):
            for j, q in enumerate(POS_ORDER):
                ax = axs[pi * len(EP4) + ei, j]
                M = nmean(name, ep, q)
                im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
                deco(ax, M)
                if pi == 0 and ei == 0:
                    ax.set_title(q.replace("_", " "), fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{name}\n{ep}", fontsize=8.5)
                if j == nq - 1:
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cb.ax.tick_params(labelsize=6)
                    cb.set_label("x animal's pre scale", fontsize=6)
    title(fig, f"NEURAL: PER-SPOUT UNIQUE R2 LEVELS by epoch, TARGET (top 4 rows) vs EXECUTED (bottom 4), frozen pixel run {a.tag} "
               f"(base {a.base}; each spout's own trials). Mean of 4 animals after normalising each animal by its own pre-stroke "
               "99th percentile (so one animal cannot dominate); one scale per part, so epochs compare directly. Canonical spout "
               "order. Image left = ipsilesional.")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.02, wspace=0.06, hspace=0.1)
    fig.savefig(run / "figures" / "spout_unique_levels.png", dpi=100)
    plt.close(fig)

    # (2) CHANGES with significance (raw-map nested bootstrap)
    rows = []
    fig, axs = plt.subplots(4, nq, figsize=(2.4 * nq + 0.8, 2.4 * 4 + 1.0), squeeze=False)
    for pi, name in enumerate(PARTS):
        Ms = {(ep, q): nmean(name, ep, q, minus_pre=True) for ep in ("acute", "chronic") for q in POS_ORDER}
        lim = np.nanpercentile(np.abs(np.stack(list(Ms.values()))), 99) or 1.0
        for ei, ep in enumerate(("acute", "chronic")):
            for j, q in enumerate(POS_ORDER):
                ax = axs[pi * 2 + ei, j]
                pre, post = by("pre", name, q), by(ep, name, q)
                com = sorted(set(pre) & set(post))
                sig, n = test({x: pre[x] for x in com}, {x: post[x] for x in com}, f, mask)
                M = Ms[(ep, q)]
                im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                up = dn = 0
                if sig is not None and sig.any():
                    big = np.zeros(M.shape)
                    hh, ww = min(sig.shape[0] * f, M.shape[0]), min(sig.shape[1] * f, M.shape[1])
                    big[:hh, :ww] = np.kron(sig, np.ones((f, f)))[:hh, :ww]
                    ax.contour(big, levels=[0.5], colors="w", linewidths=3.0)
                    ax.contour(big, levels=[0.5], colors="k", linewidths=1.4)
                    hh2, ww2 = sig.shape
                    sm = np.nanmean(M[:hh2 * f, :ww2 * f].reshape(hh2, f, ww2, f), (1, 3))
                    up, dn = int((sig & (sm > 0)).sum()), int((sig & (sm < 0)).sum())
                rows.append({"part": name, "spout": q, "epoch": ep, "up": up, "down": dn, "n_bins": n, "n_animals": len(com)})
                ax.text(0.02, 0.02, f"{up}+/{dn}-", transform=ax.transAxes, fontsize=6.5,
                        bbox={"facecolor": "white", "alpha": 0.75, "pad": 1, "lw": 0})
                deco(ax, M)
                if pi == 0 and ei == 0:
                    ax.set_title(q.replace("_", " "), fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{name}\n{ep} - pre", fontsize=8.5)
                if j == nq - 1:
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cb.ax.tick_params(labelsize=6)
                    cb.set_label("x animal's pre scale", fontsize=6)
    S = pd.DataFrame(rows)
    S.to_csv(run / "spout_unique_sig.csv", index=False)
    print(S.to_string(index=False), flush=True)
    title(fig, f"NEURAL: PER-SPOUT UNIQUE R2 CHANGES, TARGET vs EXECUTED, frozen pixel run {a.tag}: epoch - pre within animal, "
               "animal-normalised mean (x each animal's pre scale). Contours = significant 16 px bins on the RAW maps (nested "
               "bootstrap animals > sessions, Bonferroni); corner = bins up / down. Canonical spout order. Image left = ipsilesional.")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.02, wspace=0.06, hspace=0.12)
    fig.savefig(run / "figures" / "spout_unique_maps.png", dpi=105)
    plt.close(fig)

    # (3) PER ANIMAL: pre and chronic levels, one colour scale (and bar) per animal x part
    ans = sorted(meta.animal.unique())
    for name in PARTS:
        fig, axs = plt.subplots(2 * len(ans), nq, figsize=(2.4 * nq + 0.9, 2.3 * 2 * len(ans) + 1.0), squeeze=False)
        for ai, an in enumerate(ans):
            vals = [lev[(name, ep, q)].get(an) for ep in ("pre", "chronic") for q in POS_ORDER]
            vals = [v for v in vals if v is not None]
            vmax = np.nanpercentile(np.stack(vals), 99) if vals else 1.0
            for ei, ep in enumerate(("pre", "chronic")):
                for j, q in enumerate(POS_ORDER):
                    ax = axs[ai * 2 + ei, j]
                    M = lev[(name, ep, q)].get(an)
                    if M is None:
                        ax.axis("off")
                        continue
                    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
                    if ep == "chronic":                     # within-animal test, chronic vs pre (sessions as unit)
                        sig, d = animal_sig(by("pre", name, q).get(an, []), by("chronic", name, q).get(an, []))
                        if sig is not None and sig.any():
                            for sel, col in ((sig & (d > 0), "#e41a1c"), (sig & (d < 0), "#377eb8")):
                                if sel.any():
                                    big = np.zeros(M.shape)
                                    hh, ww = min(sel.shape[0] * f, M.shape[0]), min(sel.shape[1] * f, M.shape[1])
                                    big[:hh, :ww] = np.kron(sel, np.ones((f, f)))[:hh, :ww]
                                    ax.contour(big, levels=[0.5], colors="w", linewidths=3.0)
                                    ax.contour(big, levels=[0.5], colors=col, linewidths=1.4)
                        n_u = 0 if sig is None else int((sig & (d > 0)).sum())
                        n_d = 0 if sig is None else int((sig & (d < 0)).sum())
                        ax.text(0.02, 0.02, f"vs pre: {n_u} up / {n_d} down", transform=ax.transAxes, fontsize=6,
                                bbox={"facecolor": "white", "alpha": 0.75, "pad": 1, "lw": 0})
                    deco(ax, M)
                    if ai == 0 and ei == 0:
                        ax.set_title(q.replace("_", " "), fontsize=9)
                    if j == 0:
                        ax.set_ylabel(f"{an}\n{ep}", fontsize=8.5)
                    if j == nq - 1:
                        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                        cb.ax.tick_params(labelsize=6)
                        cb.set_label("unique R2", fontsize=6)
        title(fig, f"NEURAL: PER-SPOUT {name.upper()} UNIQUE R2 per ANIMAL, pre and chronic levels (frozen pixel run {a.tag}, base "
                   f"{a.base}). One colour scale per animal (pre and chronic share it; colour bar at right of each row) -- read "
                   "magnitudes from the bars: animals differ ~10x. Chronic panels: outlines = 16 px bins where THIS animal's chronic sessions differ from its pre sessions (Welch t, Bonferroni; red up, blue down; secondary view). Canonical spout order. Image left = ipsilesional.")
        fig.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.02, wspace=0.06, hspace=0.12)
        fig.savefig(run / "figures" / f"spout_unique_per_animal_{name}.png", dpi=95)
        plt.close(fig)
    print("->", run / "figures", "(spout_unique_levels / _maps / _per_animal_*)")


if __name__ == "__main__":
    raise SystemExit(main())

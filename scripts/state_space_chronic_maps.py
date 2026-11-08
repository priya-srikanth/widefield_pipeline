"""Where on cortex is the post-stroke change? The pre -> epoch change mapped back to pixels and Allen areas (Priya 11-08:
"how is post-stroke neural motor representation changing? can we map the 'chronic' axis back to cortical regions /
activity patterns" -> "lets see the spout specific, minus cortex-wide, and full versions").

Basis: the cohort SVD on the brain mask (`WF_SS_BASIS=svd`, bundle ss_svdbm_20261108: Allen brain minus bulbs and the
painted glue; one basis for all four animals, so a component vector IS a pixel map, identical across animals, and an
Allen area mean is an exact linear readout, `state_space_sim._svd_area_weights`). Trials: first-lick trials (camera /
3-D first lick; contact + incomplete), alignment from WF_SS_ALIGN. Response per session x spout = `state_space_sim.evoked`
of the spout mean (response window minus baseline window, `BAL_BINS`; lick: 0.1..0.5 s minus -1.0..-0.6 s), spouts with
>= R.MIN_TRIALS trials.

Change (per animal, Priya's rule: every readout vs the animal's OWN pre): mean over the epoch's sessions minus mean over
its pre sessions, divided by the RMS over brain pixels of the animal's pre response (full version, all spouts) -- maps in
units of each animal's own pre response, so the four animals average. Versions (the loader's WF_SS_XFORM definitions):
  full    the whole change
  nocw    minus the cortex-wide mode: the animal's pre PC1 (single-trial x bin vectors) projected out of every trial
          (`gsr`; it also removes any real cortex-wide change)
  spout   each spout minus the session's mean over spouts: the spout-specific part (its spout mean is 0 by construction)
Columns: the mean over spouts (full, nocw) and each spout in the canonical order.

Statistics. Pixels: THE repo's map test, `beta_maps.significance_contour` (as the 15r activity maps and the frozen-encoder
residual maps): nested bootstrap animals -> sessions (N_BOOT draws) on 8-px blocks inside `stat_mask` (brain mask eroded
16 px), max-statistic family-wise per panel, rim-concentrated results suppressed; run on the session maps already in
the drawn units (normalise=False). (The first render, 11-08, used within-animal session-label shuffles: sessions as the
unit, more liberal -- replaced as the cohort test.) PER ANIMAL ("visible in each animal", Priya 11-08 "both are
helpful"): each animal's chronic map gets its OWN contour from shuffling that animal's session labels (pre vs chronic,
counts kept, N_PERM), max |change| over `stat_mask` pixels per panel (family-wise); the bottom row counts, per pixel, the
animals significant in the direction of the cohort mean (k/4). Masks resolve WITHOUT the rebuild profile (glue
removed; see o2_state_space_svd).
Areas (64 Allen areas, both hemispheres): `analysis_kit.boot_delta_columns` (animals -> sessions, paired) on the session
area values; * = 95 % CI excludes 0 (primary); k/4 = animals with the change in the pooled direction; per-animal values
in the table (rule 8). Image: left hemisphere = image LEFT = ipsilesional (all lesions left).
Outputs: <state_space_recovery>/chronic_maps{tag}/areas.csv, maps.npz; page_figs/ss_cmap_<version>_<align>.png and
ss_cmap_animals_<version>_<align>.png (chronic, per animal). EXPLORATORY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local.atlas_overlay import region_edges  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402

VERSIONS = {"full": "", "nocw": "gsr", "spout": "spout"}
LABEL = {"full": "full change", "nocw": "minus the cortex-wide mode", "spout": "spout-specific part"}
EPS = ["acute", "subacute", "chronic"]
N_BOOT = 2000
N_PERM = 1000
OUT = S.ROOT / f"chronic_maps{S.tag()}"
FIGS = S.ROOT / "page_figs"
SUF = S.tag().lstrip("_")
BASIS_DIR = S.SVD.parent / "basis"


def session_responses(data):
    """{animal: {epoch: (n_sessions, 6, K) evoked spout means}} (NaN where a spout has < MIN_TRIALS trials)."""
    out = {}
    for an, sess in sorted(data.items()):
        d = {}
        for ep in ["pre", *EPS]:
            rows = []
            for _lab, e, _day, Y, pos in sess:
                if e != ep:
                    continue
                M = np.stack([S.evoked(Y[pos == p].mean(0)) if (pos == p).sum() >= R.MIN_TRIALS
                              else np.full(Y.shape[-1], np.nan) for p in S.POS])
                rows.append(M)
            if rows:
                d[ep] = np.stack(rows)
        out[an] = d
    return out


def with_mean(X):
    """(..., 6, K) -> (..., 7, K): the mean over spouts first, then the six spouts."""
    return np.concatenate([np.nanmean(X, axis=-2, keepdims=True), X], axis=-2)


def change(resp, unit, ep, rng=None):
    """Per animal: (epoch mean - pre mean) / unit, (7, K); with ``rng``: the same after shuffling session labels."""
    out = {}
    for an, d in resp.items():
        if ep not in d or "pre" not in d:
            continue
        A, B = d[ep], d["pre"]
        if rng is not None:
            allx = np.concatenate([A, B])
            idx = rng.permutation(len(allx))
            A, B = allx[idx[: len(A)]], allx[idx[len(A):]]
        out[an] = with_mean(np.nanmean(A, 0) - np.nanmean(B, 0)) / unit[an]
    return out


def pixel_maps(resp, unit, Uc, mask):
    """{epoch: (cohort mean (7, npix), contours [7 x full-frame bool or None], per-animal (n_an, 7, npix), animals,
    contour labels)}. Contours: `beta_maps.significance_contour` on every session's change map (pre sessions vs the
    epoch's sessions, per animal, in the drawn units)."""
    from wfield_local import beta_maps
    res = {}

    def frame(v):
        img = np.full(mask.shape, np.nan, np.float32)
        img[mask] = v
        return img
    for ep in EPS:
        ch = change(resp, unit, ep)
        ans = sorted(ch)
        V = np.stack([ch[a] for a in ans])                                        # n_an x 7 x K
        sess = {a: {e: (with_mean(resp[a][e]) / unit[a]) @ Uc.T for e in ("pre", ep)} for a in ans}   # n x 7 x npix
        cont, labs = [], []
        for ci in range(7):
            pre_by = {a: [frame(m) for m in sess[a]["pre"][:, ci] if np.isfinite(m).all()] for a in ans}
            post_by = {a: [frame(m) for m in sess[a][ep][:, ci] if np.isfinite(m).all()] for a in ans}
            cm, lab = beta_maps.significance_contour(pre_by, post_by, n_boot=N_BOOT, normalise=False)
            cont.append(cm)
            labs.append(lab)
        res[ep] = (np.nan_to_num(np.nanmean(V, 0)) @ Uc.T, cont, np.nan_to_num(V) @ Uc.T, ans, labs)
    return res


def animal_contours(resp, unit, Uc, statpx, rng, ep="chronic"):
    """{animal: (7, npix) bool}: pixels where that animal's change exceeds its own session-label shuffle null -- per
    panel the 95th percentile of the max |change| over `stat_mask` pixels (family-wise within the panel)."""
    out = {}
    for an in sorted(resp):
        d = resp[an]
        if ep not in d or "pre" not in d:
            continue
        obs = np.nan_to_num(change({an: d}, unit, ep)[an]) @ Uc.T                     # 7 x npix
        Us = Uc[statpx].T
        mx = np.stack([np.abs(np.nan_to_num(change({an: d}, unit, ep, rng)[an]) @ Us).max(1) for _ in range(N_PERM)])
        out[an] = (np.abs(obs) > np.percentile(mx, 95, axis=0)[:, None]) & statpx[None, :]
    return out


def area_table(resp, unit, W, names, version, rng):
    rows = []
    for ep in EPS:
        post, pre = {}, {}
        for an, d in resp.items():
            if ep in d and "pre" in d:
                post[an] = (with_mean(d[ep]) / unit[an]) @ W                        # sessions x 7 x areas
                pre[an] = (with_mean(d["pre"]) / unit[an]) @ W
        for ci, col in enumerate(["mean", *S.POS]):
            if version == "spout" and col == "mean":
                continue
            fin = lambda v: v[np.isfinite(v).all(1)]  # noqa: E731   sessions missing this spout are left out
            pp = {a: fin(v[:, ci]) for a, v in post.items() if len(fin(v[:, ci]))}
            qq = {a: fin(v[:, ci]) for a, v in pre.items() if len(fin(v[:, ci]))}
            b = kit.boot_delta_columns(pp, qq, rng)
            if b is None:
                continue
            per = {a: np.nanmean(post[a][:, ci], 0) - np.nanmean(pre[a][:, ci], 0) for a in sorted(post)}
            for j, n in enumerate(names):
                area, side = n.rsplit("_", 1)
                vals = np.array([per[a][j] for a in sorted(per)])
                rows.append({"version": version, "epoch": ep, "spout": col, "area": area,
                             "hemi": "ipsi" if side == "left" else "contra", "change": b[0][j], "lo": b[1][j],
                             "hi": b[2][j], "star": bool(b[1][j] > 0 or b[2][j] < 0),
                             "k_same": int((np.sign(vals) == np.sign(b[0][j])).sum()), "n_animals": len(vals),
                             **{f"d_{a}": v for a, v in zip(sorted(per), vals)}})
    return rows


def draw(ax, v, mask, edges, vmax, sig=None, brain=None):
    """One map. Analyzed pixels (``mask``) in color with area outlines inside them; brain pixels NOT analyzed (``brain``
    minus ``mask``: the painted glue and the bulbs, Priya 11-08 "make clear our mask") in flat grey; the analyzed
    field of view outlined."""
    img = np.full(mask.shape, np.nan)
    img[mask] = v
    full = mask if brain is None else (brain | mask)
    r, c = np.where(full)
    sl = (slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1))
    if brain is not None:
        ax.imshow(np.where((brain & ~mask)[sl], 1.0, np.nan), cmap="Greys", vmin=0, vmax=2.5, interpolation="nearest")
    ax.imshow(img[sl], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.imshow(np.where((edges & mask)[sl], 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1.5, alpha=0.5,
              interpolation="nearest")
    ax.contour(mask[sl].astype(float), [0.5], colors="0.25", linewidths=0.7)
    if sig is not None and np.any(sig):              # the repo's significance outline: black on a white halo
        sig = np.asarray(sig, float)                 # (epoch_figures, the 15r maps)
        ax.contour(sig[sl], [0.5], colors="white", linewidths=2.8, alpha=0.85)
        ax.contour(sig[sl], [0.5], colors="black", linewidths=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_cohort(res, mask, edges, version, brain=None):
    cols = list(range(7)) if version != "spout" else list(range(1, 7))
    vmax = np.nanpercentile(np.abs(res["chronic"][0][cols]), 99)
    fig, axs = plt.subplots(len(EPS), len(cols), figsize=(2.3 * len(cols), 2.5 * len(EPS) + 0.6), squeeze=False)
    for r, ep in enumerate(EPS):
        M, cont = res[ep][0], res[ep][1]
        for c, ci in enumerate(cols):
            ax = axs[r, c]
            draw(ax, M[ci], mask, edges, vmax, cont[ci], brain)
            if r == 0:
                ax.set_title("mean over spouts" if ci == 0 else S.POS[ci - 1].replace("_", " "))
            if c == 0:
                ax.set_ylabel(ep, fontsize=11)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=axs, fraction=0.015, pad=0.01, label="change vs own pre (units of the animal's pre response RMS)")
    fig.suptitle(f"Where the post-stroke response changes: {LABEL[version]}, mean of 4 animals ({S.ALIGN_LABEL[S.ALIGN]}; "
                 f"first-lick trials, camera first lick; response window minus baseline, change vs own pre). Contour: nested "
                 f"bootstrap animals -> sessions ({N_BOOT}), 8-px blocks, max-stat family-wise per panel "
                 f"(beta_maps.significance_contour). Image left = ipsilesional. Grey = brain not analyzed (painted fibre "
                 f"glue, olfactory bulbs); outline = analyzed field of view.",
                 y=1.02, fontsize=9.5)
    save_png_svg(fig, FIGS / f"ss_cmap_{version}_{SUF}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_animals(res, mask, edges, version, brain=None, asig=None):
    """Chronic, per animal, each with its own shuffle contour; bottom row: k/4 animals significant in the direction of
    the cohort mean."""
    cols = list(range(7)) if version != "spout" else list(range(1, 7))
    M, V, ans = res["chronic"][0], res["chronic"][2], res["chronic"][3]
    vmax = np.nanpercentile(np.abs(V[:, cols]), 99)
    nrow = len(ans) + (asig is not None)
    fig, axs = plt.subplots(nrow, len(cols), figsize=(2.3 * len(cols), 2.5 * nrow), squeeze=False)

    def frame(b):
        f = np.zeros(mask.shape, bool)
        f[mask] = b
        return f
    for r, an in enumerate(ans):
        for c, ci in enumerate(cols):
            sig = frame(asig[an][ci]) if asig is not None and an in asig else None
            draw(axs[r, c], V[r, ci], mask, edges, vmax, sig, brain=brain)
            if r == 0:
                axs[r, c].set_title("mean over spouts" if ci == 0 else S.POS[ci - 1].replace("_", " "))
            if c == 0:
                axs[r, c].set_ylabel(an, fontsize=11)
    if asig is not None:
        for c, ci in enumerate(cols):
            k = sum((asig[a][ci] & (np.sign(V[r, ci]) == np.sign(M[ci]))) for r, a in enumerate(ans) if a in asig)
            ax = axs[-1, c]
            img = np.full(mask.shape, np.nan)
            img[mask] = k
            full = mask if brain is None else (brain | mask)
            rr, cc = np.where(full)
            sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
            if brain is not None:
                ax.imshow(np.where((brain & ~mask)[sl], 1.0, np.nan), cmap="Greys", vmin=0, vmax=2.5,
                          interpolation="nearest")
            im = ax.imshow(img[sl], cmap=plt.get_cmap("magma_r", 5), vmin=-0.5, vmax=4.5, interpolation="nearest")
            ax.imshow(np.where((edges & mask)[sl], 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1.5, alpha=0.4,
                      interpolation="nearest")
            ax.contour(mask[sl].astype(float), [0.5], colors="0.25", linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel("k / 4 animals", fontsize=11)
        fig.colorbar(im, ax=axs[-1, :], fraction=0.02, pad=0.01, ticks=range(5), label="animals significant, same sign")
    fig.suptitle(f"Chronic change vs own pre, per animal: {LABEL[version]} ({S.ALIGN_LABEL[S.ALIGN]}). Contour: that "
                 f"animal's own session-label shuffle ({N_PERM}), max-stat per panel over stat_mask. Bottom: animals "
                 f"significant in the direction of the cohort mean. Grey = brain not analyzed (glue, bulbs)", y=1.01,
                 fontsize=9.5)
    save_png_svg(fig, FIGS / f"ss_cmap_animals_{version}_{SUF}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    if S.BASIS != "svd":
        sys.exit("state_space_chronic_maps: pixel maps need the cohort SVD basis -- run with WF_SS_BASIS=svd")
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    Uc = np.load(BASIS_DIR / "U_cohort.npy").astype(np.float64)                 # brain-mask px x K, orthonormal
    mask = np.load(BASIS_DIR / "mask.npy").reshape(np.load(S.SVD.parents[1] / "atlas.npy").shape)
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    edges = region_edges(np.where(mask, atlas, 0))
    from wfield_local import beta_maps
    brain = beta_maps.allen_mask()                                             # the whole Allen brain, nothing removed
    statpx = beta_maps.stat_mask()[mask]                                       # eroded brain mask, on the basis pixels
    if "--replot" in sys.argv:                                                 # figures only, from the saved maps
        z = np.load(OUT / "maps.npz")
        ans = sorted({c.split("_", 1)[1] for c in pd.read_csv(OUT / "areas.csv").columns if c.startswith("d_")})
        for version in VERSIONS:
            res = {ep: (z[f"{version}__{ep}__mean"], [c if c.any() else None for c in z[f"{version}__{ep}__sig"]],
                        z[f"{version}__{ep}__animals"], ans) for ep in EPS}
            asig = dict(zip(ans, z[f"{version}__chronic__animal_sig"]))
            fig_cohort(res, mask, edges, version, brain)
            fig_animals(res, mask, edges, version, brain, asig)
        print(f"replotted {OUT}")
        return
    W, names = S._svd_area_weights()
    unit, rows, store = None, [], {}
    for version, xf in VERSIONS.items():
        S.XFORM = xf                                                            # the loader reads it per call
        data, _ = S.load_sessions()
        resp = session_responses(data)
        if unit is None:                                                        # the FULL pre response sets the unit
            unit = {an: np.sqrt(np.nanmean(np.sum(np.nanmean(d["pre"], 0) ** 2, -1)) / Uc.shape[0])
                    for an, d in resp.items()}
        res = pixel_maps(resp, unit, Uc, mask)
        asig = animal_contours(resp, unit, Uc, statpx, np.random.default_rng(0))
        fig_cohort(res, mask, edges, version, brain)
        fig_animals(res, mask, edges, version, brain, asig)
        rows += area_table(resp, unit, W, names, version, np.random.default_rng(1))
        for ep, (M, cont, V, _ans, labs) in res.items():
            store[f"{version}__{ep}__mean"] = M.astype(np.float32)
            store[f"{version}__{ep}__sig"] = np.stack([np.zeros(mask.shape, bool) if c is None else c for c in cont])
            store[f"{version}__{ep}__animals"] = V.astype(np.float32)
            store[f"{version}__{ep}__labels"] = np.array(labs)
            print(f"  {version} {ep}: " + " | ".join(f"{c}: {lab or 'none'}" for c, lab in zip(["mean", *S.POS], labs)),
                  flush=True)
        store[f"{version}__chronic__animal_sig"] = np.stack([asig[a] for a in res["chronic"][3]])
        print(f"{version}: done", flush=True)
    S.XFORM = ""
    np.savez_compressed(OUT / "maps.npz", mask=mask, cols=np.array(["mean", *S.POS]), **store)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "areas.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.precision", 2):
        for version in VERSIONS:
            t = T[(T.version == version) & (T.epoch == "chronic") & (T.spout == ("far_contra" if version == "spout"
                                                                                  else "mean"))]
            t = t.reindex(t.change.abs().sort_values(ascending=False).index).head(12)
            print(f"\n=== chronic, {LABEL[version]}, {'far contra' if version == 'spout' else 'mean over spouts'}: "
                  f"largest area changes (units of own pre RMS response) ===")
            print(t[["area", "hemi", "change", "lo", "hi", "star", "k_same", *[c for c in t if c.startswith("d_")]]]
                  .to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

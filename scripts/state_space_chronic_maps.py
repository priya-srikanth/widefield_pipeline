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

Statistics. Pixels: within-animal session-label shuffles (pre vs the epoch's sessions, counts kept; N_PERM), cohort map =
mean over animals; family-wise threshold per panel = 95th percentile of the max |pixel| over shuffles -> contour.
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


def pixel_maps(resp, unit, Uc, rng):
    """{epoch: (cohort mean (7, npix), FWER threshold (7,), per-animal (n_an, 7, npix))}."""
    res = {}
    for ep in EPS:
        ch = change(resp, unit, ep)
        ans = sorted(ch)
        V = np.stack([ch[a] for a in ans])                                        # n_an x 7 x K
        mean_v = np.nanmean(V, 0)
        mx = np.zeros((N_PERM, 7))
        for k in range(N_PERM):
            pv = np.nanmean(np.stack([v for v in change(resp, unit, ep, rng).values()]), 0)    # 7 x K
            mx[k] = np.nanmax(np.abs(np.nan_to_num(pv) @ Uc.T), axis=1)
        res[ep] = (np.nan_to_num(mean_v) @ Uc.T, np.percentile(mx, 95, axis=0), np.nan_to_num(V) @ Uc.T, ans)
    return res


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


def draw(ax, v, mask, edges, vmax, thr=None):
    img = np.full(mask.shape, np.nan)
    img[mask] = v
    r, c = np.where(mask)
    sl = (slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1))
    ax.imshow(img[sl], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.imshow(np.where(edges[sl], 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1.5, alpha=0.5, interpolation="nearest")
    if thr is not None:
        sig = np.zeros(mask.shape)
        sig[mask] = np.abs(v) > thr
        if sig.any():
            ax.contour(sig[sl], [0.5], colors="k", linewidths=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_cohort(res, mask, edges, version):
    cols = list(range(7)) if version != "spout" else list(range(1, 7))
    vmax = np.nanpercentile(np.abs(res["chronic"][0][cols]), 99)
    fig, axs = plt.subplots(len(EPS), len(cols), figsize=(2.3 * len(cols), 2.5 * len(EPS) + 0.6), squeeze=False)
    for r, ep in enumerate(EPS):
        M, thr = res[ep][0], res[ep][1]
        for c, ci in enumerate(cols):
            ax = axs[r, c]
            draw(ax, M[ci], mask, edges, vmax, thr[ci])
            if r == 0:
                ax.set_title("mean over spouts" if ci == 0 else S.POS[ci - 1].replace("_", " "))
            if c == 0:
                ax.set_ylabel(ep, fontsize=11)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=axs, fraction=0.015, pad=0.01, label="change vs own pre (units of the animal's pre response RMS)")
    fig.suptitle(f"Where the post-stroke response changes: {LABEL[version]}, mean of 4 animals ({S.ALIGN_LABEL[S.ALIGN]}; "
                 f"first-lick trials, camera first lick; response window minus baseline, change vs own pre). Contour: "
                 f"family-wise p < .05 per panel, within-animal session shuffles ({N_PERM}). Image left = ipsilesional.",
                 y=1.02, fontsize=9.5)
    save_png_svg(fig, FIGS / f"ss_cmap_{version}_{SUF}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_animals(res, mask, edges, version):
    cols = list(range(7)) if version != "spout" else list(range(1, 7))
    V, ans = res["chronic"][2], res["chronic"][3]
    vmax = np.nanpercentile(np.abs(V[:, cols]), 99)
    fig, axs = plt.subplots(len(ans), len(cols), figsize=(2.3 * len(cols), 2.5 * len(ans)), squeeze=False)
    for r, an in enumerate(ans):
        for c, ci in enumerate(cols):
            draw(axs[r, c], V[r, ci], mask, edges, vmax)
            if r == 0:
                axs[r, c].set_title("mean over spouts" if ci == 0 else S.POS[ci - 1].replace("_", " "))
            if c == 0:
                axs[r, c].set_ylabel(an, fontsize=11)
    fig.suptitle(f"Chronic change vs own pre, per animal: {LABEL[version]} ({S.ALIGN_LABEL[S.ALIGN]})", y=1.01)
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
    W, names = S._svd_area_weights()
    unit, rows, store = None, [], {}
    for version, xf in VERSIONS.items():
        S.XFORM = xf                                                            # the loader reads it per call
        data, _ = S.load_sessions()
        resp = session_responses(data)
        if unit is None:                                                        # the FULL pre response sets the unit
            unit = {an: np.sqrt(np.nanmean(np.sum(np.nanmean(d["pre"], 0) ** 2, -1)) / Uc.shape[0])
                    for an, d in resp.items()}
        rng = np.random.default_rng(0)
        res = pixel_maps(resp, unit, Uc, rng)
        fig_cohort(res, mask, edges, version)
        fig_animals(res, mask, edges, version)
        rows += area_table(resp, unit, W, names, version, np.random.default_rng(1))
        for ep, (M, thr, V, _ans) in res.items():
            store[f"{version}__{ep}__mean"] = M.astype(np.float32)
            store[f"{version}__{ep}__thr"] = thr
            store[f"{version}__{ep}__animals"] = V.astype(np.float32)
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

"""SESSION-LEVEL PERMUTATION + TFCE change maps for the per-spout Haufe maps (Priya 10-27: "3.52 / 3.53 -- can we do per
session versions with better powered stats? do we have the shape data plotted or just as text?").

Input: the fully per-session self-fit maps of `iti_session_patterns.py` (each session's own decoder, unit-RMS = SHAPE
only, as `iti_session_pattern_tests.load`). Per map (cued / ITI x spout one-vs-rest; also lateral / sagittal) x epoch:
  statistic  `change_z` (shared with lick_session_level_maps.py): per pixel, per animal (mean of its epoch sessions -
             mean of its pre sessions) / s_pooled, s_pooled = the within-animal session SD pooled over ALL animals and
             both groups (df-weighted; floored at its 10th percentile over pixels), cohort z = sum over animals / sqrt(k)
             (each animal weighted equally, its own pre as control). EVERY animal with >= 1 pre and >= 1 epoch session
             contributes (a one-session group adds no df to s_pooled, its difference still counts); a cell is tested
             when any animal has data.
  null       session labels permuted between pre and that epoch WITHIN each animal (N_PERM), the same permutation for
             every pixel, so the spatial correlation of the maps is kept in the null; s_pooled is recomputed per draw.
  TFCE       threshold-free cluster enhancement (E 0.5, H 2) on the cortex image, both signs; FWER p per pixel = share of
             permutations whose max |TFCE| reaches the pixel's. Uses the fact that real effects are spatially extended
             (smooth maps from 200 SVD components), which the per-pixel max-statistic of Figs 3.50-3.53 / 3.67 ignores
             -> more power at the same family-wise error.
  per animal the share of animals whose own mean change has the pixel's sign is kept (drawn as a dashed >= 3/4 line).
SHARED MACHINERY (rule 9; DECISIONS 2026-10-27 late "pixel map stats = FWER max-statistic"): `change_z` / `level_z`,
`test_one` / `test_level` (+ `test_cell` for one fan-out over both) and `max_fwer` are imported by
iti_decoder_pattern_stats / iti_spout_pattern_stats (pixels) and iti_pattern_area_stats (areas) with stat "max" =
the plain max-|z| FWER over the tested elements (no TFCE); `drawn_index` = the drawn-pixel family used by all of them.
Shape read-outs DRAWN, not only printed: (1) whole-map distance d per animal + its early-vs-late pre drift floor, per
spout x epoch (from session_pattern_maplevel.csv); (2) per-session r to the animal's pre map over days, every spout.
Outputs <E>/iti_licks/session_cluster/{tests.csv, maps.pkl}, figures/lick_session_cluster_{position,within_spout}.png,
lick_{cued,iti}_session_cluster_spout.png,
figures/lick_session_shape_d.png, figures/lick_{cued,iti}_session_shape_days.png.
Figure data: figdata/<figure stem>.pkl per figure (the TFCE results + outline + n_perm for the map figures; the whole-map
table / similarity table for the shape figures). --plot-only redraws from those files alone (no permutation, no imaging
load); --reuse-maps rebuilds the figure data from session_cluster/maps.pkl without re-running the permutations.
Usage: python scripts/iti_session_cluster_tests.py [--plot-only | --reuse-maps] [--perm N]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import CHANGE_CMAP, CMAP_NOTE, PRE_CMAP, grid_fns, imaging_grid, load_fd, plot_only, save_fd  # noqa: E402
from iti_session_pattern_tests import EP, FAMILIES, load  # noqa: E402
from iti_session_patterns import AN, POS  # noqa: E402

N_PERM = 1000
DH = 0.2            # TFCE step (z units)
OUTD = il.OUT / "session_cluster"
FAMS = {"spout_cued": FAMILIES["spout_cued"], "spout_iti": FAMILIES["spout_iti"], "position": FAMILIES["position"],
        "within_spout": FAMILIES["within_spout"]}
FIG_NAME = {"spout_cued": "lick_cued_session_cluster_spout.png", "spout_iti": "lick_iti_session_cluster_spout.png",
            "position": "lick_session_cluster_position.png",
            "within_spout": "lick_session_cluster_within_spout.png"}       # lick_ names since 10-27 (ITI + cued = no set tag)


def _tfce(img: np.ndarray, struct=None) -> np.ndarray:
    """Signed TFCE of a 2-D image (NaN = off cortex), E 0.5, H 2, step DH."""
    from scipy import ndimage
    out = np.zeros_like(img)
    x0 = np.nan_to_num(img)
    for sgn in (1, -1):
        x = np.maximum(sgn * x0, 0)
        top = x.max()
        h = DH
        while h <= top:
            m = x >= h
            lab, nl = ndimage.label(m, structure=struct)
            if nl:
                ext = np.bincount(lab.ravel())[lab]
                out += sgn * np.where(m, np.sqrt(ext) * h ** 2 * DH, 0.0)
            h += DH
    return np.where(np.isfinite(img), out, np.nan)


def pooled_sd(groups: list, floor: float | None = None) -> np.ndarray:
    """Per-pixel within-group session SD pooled over ALL groups, df-weighted: sqrt(sum_g sum_i (x_gi - mean_g)^2 /
    sum_g (n_g - 1)). A group with ONE session adds no df but still counts in the statistic (its difference / mean). If
    no group has two sessions, the spread of all sessions around their grand mean is used instead. floor = minimum SD."""
    df = sum(len(g) - 1 for g in groups)
    if df > 0:
        v = sum(((g - g.mean(0)) ** 2).sum(0) for g in groups) / df
    else:
        Z = np.vstack(groups)
        v = Z.var(0, ddof=1) if len(Z) > 1 else np.ones(Z.shape[1])
    sd = np.sqrt(v)
    return sd if floor is None else np.maximum(sd, floor)


def change_z(data: dict, perms: dict | None = None, floor: float | None = None) -> np.ndarray:
    """THE change statistic (one definition for every per-session TFCE change map, CLAUDE.md rule 9): per animal
    (mean of its epoch sessions - mean of its pre sessions) / s_pooled, s_pooled = `pooled_sd` over every animal x group
    (pre and epoch), summed over animals / sqrt(k). data = {animal: (pre (n0, pix), epoch (n1, pix))}, n0, n1 >= 1;
    perms = {animal: permutation of its n0 + n1 sessions} (labels re-dealt within animal)."""
    an = sorted(data)
    G = {}
    for a in an:
        pre, post = data[a]
        if perms is not None:
            Z = np.vstack([pre, post])
            o = perms[a]
            pre, post = Z[o[: len(pre)]], Z[o[len(pre):]]
        G[a] = (pre, post)
    s = pooled_sd([g for a in an for g in G[a]], floor)
    return np.sum([(G[a][1].mean(0) - G[a][0].mean(0)) / s for a in an], 0) / np.sqrt(len(an))


def level_z(data: dict, signs: dict | None = None, floor: float | None = None) -> np.ndarray:
    """THE level statistic: per animal mean of its session maps / s_pooled (`pooled_sd` over every animal's sessions of
    the epoch, each around its own animal mean), summed over animals / sqrt(k). data = {animal: (n >= 1, pix)};
    signs = {animal: (n,) +-1} flips whole session maps."""
    an = sorted(data)
    M = {a: data[a] if signs is None else data[a] * signs[a][:, None] for a in an}
    s = pooled_sd([M[a] for a in an], floor)
    return np.sum([M[a].mean(0) / s for a in an], 0) / np.sqrt(len(an))


def _img_fn(idx_img: np.ndarray):
    ok = idx_img >= 0

    def img(v):
        g = np.full(idx_img.shape, np.nan)
        g[ok] = v[idx_img[ok]]
        return g
    return img


def max_fwer(z: np.ndarray, null_z, n_perm: int) -> np.ndarray:
    """THE max-statistic FWER p (DECISIONS 2026-10-27 late: every per-pixel / per-area map statistic): per element
    (pixel or area) the share of n_perm null draws whose max |z| over ALL elements reaches the element's |z|,
    (count + 1) / (n_perm + 1). null_z() returns one null draw of z (same shape)."""
    mx = np.sort([np.nanmax(np.abs(null_z())) for _ in range(n_perm)])
    a = np.abs(np.asarray(z, float))
    cnt = n_perm - np.searchsorted(mx, np.nan_to_num(a, nan=np.inf), side="left")
    return np.where(np.isfinite(a), (cnt + 1) / (n_perm + 1), np.nan)


def _fwer(z: np.ndarray, null_z, img, n_perm: int, stat: str = "tfce"):
    """TFCE of z (on the image) and the max-|TFCE| familywise p per pixel against n_perm null draws of z.
    stat="max": no TFCE -- (z, max-|z| FWER p) on the elements as given (`max_fwer`; img only reshapes)."""
    if stat == "max":
        return img(z), img(max_fwer(z, null_z, n_perm))
    T = _tfce(img(z))
    mx = np.empty(n_perm)
    for i in range(n_perm):
        mx[i] = np.nanmax(np.abs(_tfce(img(null_z()))))
    p = (np.sum(mx[None, None, :] >= np.abs(T)[..., None], -1) + 1) / (n_perm + 1)
    return T, np.where(np.isfinite(T), p, np.nan)


def test_one(item):
    """CHANGE test: (key, epoch, {animal: (pre (n0, pix), epoch (n1, pix))}, image index, n_perm[, stat]) -> result dict.
    stat "tfce" (default) = max-|TFCE| FWER on the image; "max" = max-|z| FWER over the elements (pixels / areas; pass a
    1-D index np.arange(n) to get vectors back).
    Statistic `change_z`; null = session labels permuted within each animal (same permutation for every pixel; s_pooled
    recomputed per draw); max-|TFCE| FWER. Every animal with >= 1 pre and >= 1 epoch session contributes. The SD floor
    (10th percentile over pixels of the observed s_pooled) is fixed across draws. Module-level (spawn)."""
    key, ep, data, idx_img, n_perm, *opt = item
    stat = opt[0] if opt else "tfce"
    import zlib
    rng = np.random.default_rng(zlib.crc32(f"{key}|{ep}".encode()))   # stable across processes (not hash())
    an = sorted(data)
    img = _img_fn(idx_img)
    floor = float(np.quantile(pooled_sd([g for a in an for g in data[a]]), 0.1))
    z = change_z(data, floor=floor)
    T, p = _fwer(z, lambda: change_z(data, {a: rng.permutation(len(data[a][0]) + len(data[a][1])) for a in an}, floor),
                 img, n_perm, stat)
    chg = np.vstack([data[a][1].mean(0) - data[a][0].mean(0) for a in an])
    agree = (np.sign(chg) == np.sign(chg.mean(0))[None, :]).sum(0)
    return dict(key=key, epoch=ep, kind="change", stat=stat, n=len(an), point=img(chg.mean(0)), z=img(z), tfce=T, p=p,
                agree=img(agree.astype(float)), n_sess=sum(len(d[0]) + len(d[1]) for d in data.values()),
                n_sess_animal={a: (len(data[a][0]), len(data[a][1])) for a in an})


def test_level(item):
    """LEVEL test (is the epoch map != 0): (key, epoch, {animal: (n >= 1, pix)}, image index, n_perm[, stat]) -> result
    dict (stat as `test_one`).
    Statistic `level_z`; null = sign-flip of each whole SESSION map, independently within each animal (valid with one
    session: a self-fit Haufe map flips sign with its target, so a no-information session is symmetric about 0;
    s_pooled recomputed per draw); max-|TFCE| FWER. Module-level (spawn)."""
    key, ep, data, idx_img, n_perm, *opt = item
    stat = opt[0] if opt else "tfce"
    import zlib
    rng = np.random.default_rng(zlib.crc32(f"{key}|{ep}|level".encode()))
    an = sorted(data)
    img = _img_fn(idx_img)
    floor = float(np.quantile(pooled_sd([data[a] for a in an]), 0.1))
    z = level_z(data, floor=floor)
    T, p = _fwer(z, lambda: level_z(data, {a: rng.choice([-1.0, 1.0], len(data[a])) for a in an}, floor), img, n_perm,
                 stat)
    am = np.vstack([data[a].mean(0) for a in an])
    agree = (np.sign(am) == np.sign(am.mean(0))[None, :]).sum(0)
    return dict(key=key, epoch=ep, kind="level", stat=stat, n=len(an), point=img(am.mean(0)), z=img(z), tfce=T, p=p,
                agree=img(agree.astype(float)), n_sess=sum(len(v) for v in data.values()),
                n_sess_animal={a: len(data[a]) for a in an})


def test_cell(item):
    """Dispatcher for one fan_out over mixed cells: (kind, item) -> `test_level` (kind "level") or `test_one` (else).
    Module-level (spawn)."""
    kind, it = item
    return test_level(it) if kind == "level" else test_one(it)


def drawn_index(to_img, npix: int) -> np.ndarray:
    """Indices (into the npix stat-mask pixels) of the pixels DRAWN on the half-resolution image (every 2nd row /
    column): the family over which the map FWER is taken, in every map family (rule 9)."""
    g = to_img(np.arange(npix, dtype=float))
    return g[np.isfinite(g)].astype(int)


def print_counts(R: dict, title: str) -> None:
    """Rule 8: sessions per animal in every tested cell (change cells: pre/epoch)."""
    rows = []
    for kk, r in sorted(R.items(), key=lambda t: str(t[0])):
        if "n_sess_animal" not in r:
            continue
        row = {"cell": " | ".join(str(x) for x in kk)}
        for a in AN:
            v = r["n_sess_animal"].get(a)
            row[a] = "-" if v is None else (f"{v[0]}/{v[1]}" if isinstance(v, tuple) else str(v))
        rows.append(row)
    print(f"\n== {title}: sessions per animal per cell (change = pre/epoch; '-' = no data)")
    print(pd.DataFrame(rows).to_string(index=False))


def run(n_perm: int) -> dict:
    from wfield_local import parallel
    L = load()
    to_img, _ol = grid_fns(imaging_grid())
    npix = next(iter(next(iter(L.values())).values()))["M"].shape[1]
    g = to_img(np.arange(npix, dtype=float))
    ok = np.isfinite(g)
    sel = drawn_index(to_img, npix)                       # only the pixels drawn on the (every 2nd row / column) image
    idx_img = np.full(g.shape, -1)
    idx_img[ok] = np.arange(ok.sum())
    items = []
    for keys in FAMS.values():
        for k in keys:
            for ep in EP[1:]:
                data = {}
                for a in AN:
                    v = L.get(a, {}).get(k)
                    if v is None:
                        continue
                    pre, post = v["M"][v["epoch"] == "pre"], v["M"][v["epoch"] == ep]
                    if len(pre) and len(post):                  # every animal with >= 1 session in each group
                        data[a] = (pre[:, sel], post[:, sel])
                if data:
                    items.append((k, ep, data, idx_img, n_perm))
    res, fails = parallel.fan_out(items, test_one, jobs=parallel.default_jobs(6), label="map x epoch",
                                  log=lambda m: print(m[:110], flush=True))
    res = sorted(res, key=lambda t: (t[1]["key"], t[1]["epoch"]))     # completion order -> fixed order (CLAUDE.md 6)
    if fails:
        print("FAILED:", [f[0][:2] for f in fails])
    out = {(r["key"], r["epoch"]): r for _it, r in res}
    # pre LEVEL maps (mean of animals' pre means) for column 1
    for keys in FAMS.values():
        for k in keys:
            pm = [L[a][k]["M"][L[a][k]["epoch"] == "pre"] for a in AN if a in L and k in L[a]]
            pm = [m for m in pm if len(m)]
            if pm:
                out[(k, "pre")] = dict(point=to_img(np.mean([m.mean(0) for m in pm], 0)), n=len(pm),
                                       n_sess=sum(len(m) for m in pm))
    return out


def tests_table(R: dict) -> pd.DataFrame:
    rows = []
    for (k, ep), r in R.items():
        if ep == "pre":
            continue
        cx = np.isfinite(r["p"])
        sig = (r["p"] < 0.05) & cx
        rows.append(dict(map=k, epoch=ep, n_animals=r["n"], n_sessions=r["n_sess"], frac_sig=float(sig.sum() / cx.sum()),
                         frac_sig_pos=float((sig & (r["point"] > 0)).sum() / cx.sum()),
                         frac_sig_neg=float((sig & (r["point"] < 0)).sum() / cx.sum()),
                         frac_sig_agree3=float((sig & (r["agree"] >= 3)).sum() / cx.sum()), min_p=float(np.nanmin(r["p"]))))
    return pd.DataFrame(rows).sort_values(["map", "epoch"])


def _lab(k: str) -> str:
    return k.replace("trial/", "cued ").replace("iti/", "ITI ").replace("sp_", "").replace("_", " ")


def plot_maps(D: dict) -> None:
    """D = figure data: R (TFCE results of this family's maps + pre levels), fam, keys, ol (outline), n_perm."""
    R, fam, keys, ol, n_perm = D["R"], D["fam"], D["keys"], D["ol"], D["n_perm"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in keys if (k, "pre") in R]
    fig, axs = plt.subplots(len(keys), 4, figsize=(12.8, 2.6 * len(keys) + 1.4), squeeze=False, layout="constrained")
    for i, k in enumerate(keys):
        im_l = im_c = None
        chg = [R[(k, e)]["point"] for e in EP[1:] if (k, e) in R]
        lim_c = np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in chg])), 99) if chg else 1
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            r = R.get((k, ep))
            if r is None:                           # blank only when NO animal has data
                ax.set_title(f"{ep} - pre: no data", fontsize=7)
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                continue
            if ep == "pre":                         # pre LEVEL: purple-orange, own scale (Priya 10-27)
                lim = np.nanpercentile(np.abs(r["point"]), 99)
                im_l = ax.imshow(r["point"], cmap=PRE_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
            else:
                im_c = ax.imshow(r["point"], cmap=CHANGE_CMAP, vmin=-lim_c, vmax=lim_c, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            if ep == "pre":
                ax.set_title(f"pre (level): {r.get('n_sess', '?')} sess / {r['n']} an", fontsize=7.5)
            else:
                sig = np.nan_to_num(r["p"], nan=1) < 0.05
                ax.contour(sig.astype(float), levels=[0.5], colors="k", linewidths=1.3)
                ax.contour((sig & (np.nan_to_num(r["agree"]) >= 3)).astype(float), levels=[0.5], colors="k",
                           linewidths=0.7, linestyles="--")
                cx = np.isfinite(r["p"])
                ax.set_title(f"{ep} - pre: {r['n_sess']} sess (pre + {ep}) / {r['n']} an\nTFCE sig "
                             f"{100 * sig.sum() / cx.sum():.0f}% (+{100 * (sig & (r['point'] > 0)).sum() / cx.sum():.0f} / "
                             f"-{100 * (sig & (r['point'] < 0)).sum() / cx.sum():.0f}), min p {np.nanmin(r['p']):.3f}",
                             fontsize=7)
            if j == 0:
                ax.set_ylabel(_lab(k), fontsize=8.5)
        if im_l is not None:
            cb = fig.colorbar(im_l, ax=axs[i, 0], fraction=0.06, pad=0.02)
            cb.set_label("pre LEVEL (unit RMS)", fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
        if im_c is not None:
            cb = fig.colorbar(im_c, ax=axs[i, 1:].tolist(), fraction=0.02, pad=0.01)
            cb.set_label("CHANGE epoch - pre", fontsize=7)
            cb.ax.tick_params(labelsize=6.5)
    import textwrap
    txt = "\n".join(textwrap.fill(q, 125) for q in [
        f"NEURAL: per-SESSION self-fit Haufe maps ({fam}), unit RMS (SHAPE only). Column 1 = pre map (mean of animals); "
        f"columns 2-4 = epoch - pre, mean of animals. {CMAP_NOTE[0].upper() + CMAP_NOTE[1:]}; one scale per row for each.",
        f"SOLID = TFCE family-wise p < 0.05: per animal (mean epoch - mean pre sessions) / within-animal session SD pooled "
        f"over all animals and both groups, summed over animals / sqrt(k); every animal with >= 1 session per group "
        f"contributes; null = session labels permuted within animal ({n_perm} x), max |TFCE| over cortex. DASHED = also >= 3 "
        "animals share the sign. ipsi (lesion) = image LEFT. PROVISIONAL"])
    fig.suptitle(txt, fontsize=8.5)
    save_png_svg(fig, il.OUT / "figures" / FIG_NAME[fam], dpi=95)
    plt.close(fig)


def plot_shape_d(D: dict) -> None:
    """Whole-map shape distance per animal (dots) vs its early-vs-late pre drift floor, per spout x epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import config
    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = config.animal_color()
    T = D["T"]
    keys = {"cued": [f"trial/sp_{p}" for p in POS], "ITI": [f"iti/sp_{p}" for p in POS]}
    fig, axs = plt.subplots(2, 6, figsize=(17, 6.8), sharey=True, squeeze=False)
    for i, (st, ks) in enumerate(keys.items()):
        for j, k in enumerate(ks):
            ax = axs[i, j]
            t = T[T["map"] == k].set_index("epoch").reindex(EP[1:])
            for x, (ep, r) in enumerate(t.iterrows(), start=1):
                if not np.isfinite(r.d_mean):
                    continue
                for q, a in enumerate(AN):
                    if f"d_{a}" in r and np.isfinite(r[f"d_{a}"]):
                        ax.plot(x + (q - 1.5) * 0.09, r[f"d_{a}"], "o", ms=4.5, color=col.get(a, "0.5"),
                                mec="k" if r[f"p_{a}"] < 0.05 else "none", mew=0.8)
                ax.plot([x - 0.25, x + 0.25], [r.d_mean] * 2, color=EPOCH_LINE[ep], lw=2.5)
                ax.text(x, r.d_mean, ("*" if r.sig_bh else "") + f"\n{int(r.n_sig_animals)}/{int(r.n_animals)}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold" if r.sig_bh else "normal")
            fl = t.d_pre_early_vs_late.dropna()
            if len(fl):
                ax.axhline(fl.mean(), color="0.4", ls="--", lw=0.9)
            ax.set_xticks([1, 2, 3], EP[1:], fontsize=7.5)
            ax.set_title(f"{st} {k.split('sp_')[1].replace('_', ' ')}", fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[i, 0].set_ylabel("shape distance d = 1 - r(epoch map, pre map)", fontsize=8)
    h = [Line2D([], [], color=col.get(a, "0.5"), marker="o", lw=0, label=a) for a in AN] + \
        [Line2D([], [], color="0.4", ls="--", label="drift floor (early vs late pre)"),
         Line2D([], [], color="k", marker="o", mfc="w", lw=0, label="black edge = animal's own p < 0.05")]
    axs[0, 0].legend(handles=h, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("NEURAL: whole-map SHAPE change per spout (per-session self-fit maps, unit RMS): dots = animals, bar = "
                 "cohort mean (epoch color), dashed = drift floor;\n* = cohort session-label permutation p, BH q 0.05 "
                 "over all maps x epochs (bold); k/n = animals with their own p < 0.05. Canonical spout order. PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, il.OUT / "figures" / "lick_session_shape_d.png", dpi=100)
    plt.close(fig)


def plot_shape_days(D: dict) -> None:
    """Per-session r to the animal's pre map over days, every spout (pre sessions vs the mean of the OTHER pre)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import config
    col = config.animal_color()
    S = D["S"]
    for st, pre_ in [(D["st"], "trial" if D["st"] == "cued" else "iti")]:
        fig, axs = plt.subplots(2, 3, figsize=(14, 7.2), sharex=True, sharey=True, squeeze=False)
        for j, p in enumerate(POS):
            ax = axs[j // 3, j % 3]
            g = S[S["map"] == f"{pre_}/sp_{p}"]
            for a, h in g.groupby("animal"):
                h = h.sort_values("day")
                ax.plot(h.day, h.r_to_pre, "o-", ms=2.8, lw=0.9, color=col.get(a, "0.5"))
                fl = h[h.epoch == "pre"].r_to_pre.mean()
                ax.axhline(fl, color=col.get(a, "0.5"), lw=0.6, ls=":")
            ax.axvline(0, color="k", lw=0.7, ls="--")
            ax.axhline(0, color="0.8", lw=0.5)
            ax.set_title(p.replace("_", " "), fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        for ax in axs[1]:
            ax.set_xlabel("day after stroke", fontsize=8)
        for ax in axs[:, 0]:
            ax.set_ylabel("r(session map, animal's pre map)", fontsize=8)
        axs[0, 0].legend(handles=[Line2D([], [], color=col.get(a, "0.5"), label=a) for a in AN], fontsize=7.5,
                         frameon=False, loc="lower left")
        fig.suptitle(f"NEURAL: per-session map SHAPE vs the animal's pre map over days, {'CUED' if st == 'cued' else 'ITI'} "
                     "per-spout one-vs-rest maps (canonical order);\npre sessions = vs the mean of the OTHER pre sessions; "
                     "dotted = the animal's pre mean (its noise floor). PROVISIONAL", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        save_png_svg(fig, il.OUT / "figures" / f"lick_{st}_session_shape_days.png", dpi=100)
        plt.close(fig)


def trial_set_sentence(fig: str | None = None) -> str:
    """The licks of one figure (Priya 11-09): a map family, "shape_d" (the per-spout maps of both sets), "cued" / "iti"
    (the shape-over-days figure of that set), or None = every family."""
    from iti_session_patterns import family_sentence
    from wfield_local import trial_set_text as tst
    if fig == "shape_d":
        return tst.union(family_sentence("spout_cued"), family_sentence("spout_iti"))
    return family_sentence({"cued": "spout_cued", "iti": "spout_iti", None: "all"}.get(fig, fig))


def _draw(fn, D: dict, fig: str) -> None:
    from wfield_local import trial_set_text as tst
    with tst.active(None, text=trial_set_sentence(fig)):
        fn(D)


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    if plot_only():                                       # figure data only: no permutation, no imaging load
        for fam in FAMS:
            _draw(plot_maps, load_fd(FIG_NAME[fam]), fam)
        _draw(plot_shape_d, load_fd("lick_session_shape_d"), "shape_d")
        for st in ("cued", "iti"):
            _draw(plot_shape_days, load_fd(f"lick_{st}_session_shape_days"), st)
        return 0
    n_perm = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else N_PERM
    OUTD.mkdir(exist_ok=True)
    f = OUTD / "maps.pkl"
    if "--reuse-maps" in sys.argv and f.exists():
        R = pickle.loads(f.read_bytes())
    else:
        R = run(n_perm)
        f.write_bytes(pickle.dumps(R))
    print_counts(R, "per-session TFCE change maps (shape only)")
    T = tests_table(R)
    T.to_csv(OUTD / "tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    _to_img, ol = grid_fns(imaging_grid())
    for fam, keys in FAMS.items():
        D = dict(R={k: v for k, v in R.items() if k[0] in keys}, fam=fam, keys=list(keys), ol=ol, n_perm=n_perm)
        save_fd(FIG_NAME[fam], D)
        _draw(plot_maps, D, fam)
    D = dict(T=pd.read_csv(il.OUT / "session_pattern_maplevel.csv"))      # from iti_session_pattern_tests.py
    save_fd("lick_session_shape_d", D)
    _draw(plot_shape_d, D, "shape_d")
    S = pd.read_csv(il.OUT / "session_pattern_similarity.csv")
    for st in ("cued", "iti"):
        D = dict(S=S, st=st)
        save_fd(f"lick_{st}_session_shape_days", D)
        _draw(plot_shape_days, D, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

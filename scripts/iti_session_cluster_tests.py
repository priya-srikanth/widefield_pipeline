"""SESSION-LEVEL PERMUTATION + TFCE change maps for the per-spout Haufe maps (Priya 10-27: "3.52 / 3.53 -- can we do per
session versions with better powered stats? do we have the shape data plotted or just as text?").

Input: the fully per-session self-fit maps of `iti_session_patterns.py` (each session's own decoder, unit-RMS = SHAPE
only, as `iti_session_pattern_tests.load`). Per map (cued / ITI x spout one-vs-rest; also lateral / sagittal) x epoch:
  statistic  per pixel, per animal Welch t of (epoch sessions - pre sessions); cohort z = sum of the animals' t / sqrt(k)
             (each animal weighted equally, its own pre as control).
  null       session labels permuted between pre and that epoch WITHIN each animal (N_PERM), the same permutation for
             every pixel, so the spatial correlation of the maps is kept in the null.
  TFCE       threshold-free cluster enhancement (E 0.5, H 2) on the cortex image, both signs; FWER p per pixel = share of
             permutations whose max |TFCE| reaches the pixel's. Uses the fact that real effects are spatially extended
             (smooth maps from 200 SVD components), which the per-pixel max-statistic of Figs 3.50-3.53 / 3.67 ignores
             -> more power at the same family-wise error.
  per animal the share of animals whose own mean change has the pixel's sign is kept (drawn as a dashed >= 3/4 line).
Shape read-outs DRAWN, not only printed: (1) whole-map distance d per animal + its early-vs-late pre drift floor, per
spout x epoch (from session_pattern_maplevel.csv); (2) per-session r to the animal's pre map over days, every spout.
Outputs <E>/iti_licks/session_cluster/{tests.csv, maps.pkl}, figures/lick_session_cluster_position.png, lick_{cued,iti}_session_cluster_spout.png,
figures/lick_session_shape_d.png, figures/lick_{cued,iti}_session_shape_days.png.
Usage: python scripts/iti_session_cluster_tests.py [--plot-only] [--perm N]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from iti_session_pattern_tests import EP, FAMILIES, _imaging_grid, load  # noqa: E402
from iti_session_patterns import AN, POS  # noqa: E402

N_PERM = 1000
DH = 0.2            # TFCE step (z units)
OUTD = il.OUT / "session_cluster"
FAMS = {"spout_cued": FAMILIES["spout_cued"], "spout_iti": FAMILIES["spout_iti"], "position": FAMILIES["position"]}
FIG_NAME = {"spout_cued": "lick_cued_session_cluster_spout.png", "spout_iti": "lick_iti_session_cluster_spout.png",
            "position": "lick_session_cluster_position.png"}       # lick_ names since 10-27 (ITI + cued = no set tag)


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


def _welch(pre: np.ndarray, post: np.ndarray, floor: float) -> np.ndarray:
    v = pre.var(0, ddof=1) / len(pre) + post.var(0, ddof=1) / len(post)
    return (post.mean(0) - pre.mean(0)) / np.sqrt(np.maximum(v, floor))


def test_one(item):
    """(key, epoch, {animal: (pre (n0, pix), post (n1, pix))}, image index, n_perm) -> result dict. Module-level (spawn)."""
    key, ep, data, idx_img, n_perm = item
    import zlib
    rng = np.random.default_rng(zlib.crc32(f"{key}|{ep}".encode()))   # stable across processes (not hash())
    an = sorted(data)
    floors = {a: np.quantile(np.r_[data[a][0].var(0, ddof=1) / len(data[a][0]),
                                   data[a][1].var(0, ddof=1) / len(data[a][1])], 0.1) for a in an}

    def img(v):
        g = np.full(idx_img.shape, np.nan)
        ok = idx_img >= 0
        g[ok] = v[idx_img[ok]]
        return g

    def cohort(perms=None):
        zs = []
        for a in an:
            pre, post = data[a]
            if perms is None:
                zs.append(_welch(pre, post, floors[a]))
            else:
                Z = np.vstack([pre, post])
                o = perms[a]
                zs.append(_welch(Z[o[: len(pre)]], Z[o[len(pre):]], floors[a]))
        return np.sum(zs, 0) / np.sqrt(len(an))
    z = cohort()
    T = _tfce(img(z))
    mx = np.empty(n_perm)
    for i in range(n_perm):
        perms = {a: rng.permutation(len(data[a][0]) + len(data[a][1])) for a in an}
        mx[i] = np.nanmax(np.abs(_tfce(img(cohort(perms)))))
    p = (np.sum(mx[None, None, :] >= np.abs(T)[..., None], -1) + 1) / (n_perm + 1)
    p = np.where(np.isfinite(T), p, np.nan)
    chg = np.vstack([data[a][1].mean(0) - data[a][0].mean(0) for a in an])
    agree = (np.sign(chg) == np.sign(chg.mean(0))[None, :]).sum(0)
    return dict(key=key, epoch=ep, n=len(an), point=img(chg.mean(0)), z=img(z), tfce=T, p=p, agree=img(agree.astype(float)),
                n_sess=sum(len(d[0]) + len(d[1]) for d in data.values()))


def run(n_perm: int) -> dict:
    from wfield_local import parallel
    L = load()
    to_img, _ol = _imaging_grid()
    npix = next(iter(next(iter(L.values())).values()))["M"].shape[1]
    g = to_img(np.arange(npix, dtype=float))
    ok = np.isfinite(g)
    sel = g[ok].astype(int)                               # only the pixels drawn on the (every 2nd row / column) image
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
                    if len(pre) >= 3 and len(post) >= 3:
                        data[a] = (pre[:, sel], post[:, sel])
                if len(data) >= 3:
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
            pm = [L[a][k]["M"][L[a][k]["epoch"] == "pre"].mean(0) for a in AN if a in L and k in L[a]]
            if pm:
                out[(k, "pre")] = dict(point=to_img(np.mean(pm, 0)), n=len(pm))
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


def plot_maps(R: dict, fam: str, keys: list, ol, n_perm: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in keys if (k, "pre") in R]
    fig, axs = plt.subplots(len(keys), 4, figsize=(11.5, 2.55 * len(keys) + 1.4), squeeze=False)
    for i, k in enumerate(keys):
        chg = [R[(k, e)]["point"] for e in EP[1:] if (k, e) in R]
        lim_c = np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in chg])), 99) if chg else 1
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            r = R.get((k, ep))
            if r is None:
                ax.axis("off")
                continue
            lim = np.nanpercentile(np.abs(r["point"]), 99) if ep == "pre" else lim_c
            ax.imshow(r["point"], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            if ep == "pre":
                ax.set_title(f"pre (level), n={r['n']} animals", fontsize=7.5)
            else:
                sig = np.nan_to_num(r["p"], nan=1) < 0.05
                ax.contour(sig.astype(float), levels=[0.5], colors="k", linewidths=1.3)
                ax.contour((sig & (np.nan_to_num(r["agree"]) >= 3)).astype(float), levels=[0.5], colors="k",
                           linewidths=0.7, linestyles="--")
                cx = np.isfinite(r["p"])
                ax.set_title(f"{ep} - pre: TFCE sig {100 * sig.sum() / cx.sum():.0f}% (+{100 * (sig & (r['point'] > 0)).sum() / cx.sum():.0f} / "
                             f"-{100 * (sig & (r['point'] < 0)).sum() / cx.sum():.0f}), min p {np.nanmin(r['p']):.3f}",
                             fontsize=7)
            if j == 0:
                ax.set_ylabel(_lab(k), fontsize=8.5)
    import textwrap
    txt = "\n".join(textwrap.fill(q, 125) for q in [
        f"NEURAL: per-SESSION self-fit Haufe maps ({fam}), unit RMS (SHAPE only). Column 1 = pre map (mean of animals); "
        "columns 2-4 = epoch - pre, mean of animals.",
        f"SOLID = TFCE family-wise p < 0.05: per-pixel Welch t per animal (epoch vs pre sessions), summed over animals; null = "
        f"session labels permuted within animal ({n_perm} x), max |TFCE| over cortex. DASHED = also >= 3/4 animals share the "
        "sign. ipsi (lesion) = image LEFT. PROVISIONAL"])
    fig.suptitle(txt, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.17 * (txt.count(chr(10)) + 1) + 0.25) / fig.get_size_inches()[1]))
    fig.savefig(il.OUT / "figures" / FIG_NAME[fam], dpi=95)
    plt.close(fig)


def plot_shape_d() -> None:
    """Whole-map shape distance per animal (dots) vs its early-vs-late pre drift floor, per spout x epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import config
    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = config.animal_color()
    T = pd.read_csv(il.OUT / "session_pattern_maplevel.csv")
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
                 "cohort mean (epoch colour), dashed = drift floor;\n* = cohort session-label permutation p, BH q 0.05 "
                 "over all maps x epochs (bold); k/n = animals with their own p < 0.05. Canonical spout order. PROVISIONAL",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(il.OUT / "figures" / "lick_session_shape_d.png", dpi=100)
    plt.close(fig)


def plot_shape_days() -> None:
    """Per-session r to the animal's pre map over days, every spout (pre sessions vs the mean of the OTHER pre)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local import config
    col = config.animal_color()
    S = pd.read_csv(il.OUT / "session_pattern_similarity.csv")
    for st, pre_ in (("cued", "trial"), ("iti", "iti")):
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
        fig.savefig(il.OUT / "figures" / f"lick_{st}_session_shape_days.png", dpi=100)
        plt.close(fig)


def main() -> int:
    n_perm = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else N_PERM
    OUTD.mkdir(exist_ok=True)
    f = OUTD / "maps.pkl"
    if "--plot-only" in sys.argv and f.exists():
        R = pickle.loads(f.read_bytes())
    else:
        R = run(n_perm)
        f.write_bytes(pickle.dumps(R))
    T = tests_table(R)
    T.to_csv(OUTD / "tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    _to_img, ol = _imaging_grid()
    for fam, keys in FAMS.items():
        plot_maps(R, fam, keys, ol, n_perm)
    plot_shape_d()
    plot_shape_days()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

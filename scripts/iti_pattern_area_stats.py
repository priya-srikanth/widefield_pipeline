"""ALLEN-AREA statistics for the Haufe pattern maps (3.33 / 3.35 / ITI 3.35; Priya 10-27: "is per-pixel nested
bootstrap too stringent? pixels correlate").

Complement to the per-pixel tests in `iti_decoder_pattern_stats.py` / `iti_spout_pattern_stats.py`: the same held-out
per-SESSION patterns (their caches), averaged within each Allen area (stat mask, per hemisphere; areas < 50 pixels
dropped), so ~40 tests instead of thousands of correlated pixels. Per area: level (epoch) and change (epoch - pre,
paired by animal), nested bootstrap animals > sessions (`_boot`, 2000 draws), two-sided p, BH q = 0.05 over areas x
(epoch, kind) within a map family; n_dir = animals whose own mean has the pooled sign. Rule 8: per-animal area means
are in the CSV. A sign-flip permutation across 4 animals (16 arrangements, min p 0.0625) cannot reach significance, so
the bootstrap is kept.
Outputs <E>/iti_licks/pattern_area_stats_{position,spout,spout_iti}.csv and figures/lick_pattern_area_stats_position.png (ITI + cued), lick_cued_pattern_area_stats_spout.png,
lick_iti_pattern_area_stats_spout.png.
Figure data: figdata/<figure stem>.pkl = dict(T=the area table, label); --plot-only redraws from it (no bootstrap).
Usage: python scripts/iti_pattern_area_stats.py [--plot-only]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import CHANGE_CMAP, CMAP_NOTE, PRE_CMAP, fd_path, levels_name, load_fd, plot_only, save_fd, sym_lim  # noqa: E402
from iti_decoder_pattern_stats import _bh, _boot  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
MIN_PIX = 50
FIG_NAME = {"position": "lick_pattern_area_stats_position.png",       # ITI + cued (lick_ names since 10-27)
            "spout": "lick_cued_pattern_area_stats_spout.png", "spout_iti": "lick_iti_pattern_area_stats_spout.png"}


def area_index():
    """(area name per stat pixel, names kept) on the shared Allen grid, in the order of the pattern maps' pixels."""
    from wfield_local import combined_encoding as ce
    lab0 = il.labels()[0]
    _V, _ft, _Ub, _flat, parea, names, _shape, pstat = ce.load_imaging(lab0)
    a = np.array([names.get(int(x), "") for x in parea[pstat]])
    keep = [n for n in sorted(set(a)) if n and (a == n).sum() >= MIN_PIX]
    return a, keep


def family(SM: dict, groups: list, label: str, A: np.ndarray, keep: list) -> pd.DataFrame:
    """SM: {(animal, group): {epoch: (n_sessions, pixels)}} -> one row per group x area x epoch x kind."""
    idx = {n: A == n for n in keep}
    rows = []
    rng = np.random.default_rng(0)
    for grp in groups:
        have = {}
        for a in AN:
            v = SM.get((a, *grp) if isinstance(grp, tuple) else (a, grp))
            if v and "pre" in v:
                have[a] = {ep: np.column_stack([m[:, idx[n]].mean(1) for n in keep]) for ep, m in v.items()}
        if not have:
            continue
        pre = {a: v["pre"] for a, v in have.items()}
        for ep in EP:
            lvl = {a: v[ep] for a, v in have.items() if ep in v}
            for kind in ("level", "change"):
                if kind == "change" and ep == "pre":
                    continue
                ref = pre if kind == "change" else None
                am = np.vstack([lvl[a].mean(0) - (ref[a].mean(0) if ref else 0) for a in sorted(lvl)])
                B = _boot(lvl, rng, ref)
                point = am.mean(0)
                p = np.minimum(1.0, 2 * np.minimum((B <= 0).mean(0), (B >= 0).mean(0)) + 1 / len(B))
                lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
                n_dir = (np.sign(am) == np.sign(point)[None, :]).sum(0)
                for k, n in enumerate(keep):
                    rows.append(dict(family=label, group=grp if isinstance(grp, str) else "/".join(grp), area=n,
                                     epoch=ep, kind=kind, point=point[k], lo=lo[k], hi=hi[k], p=p[k],
                                     n_dir=int(n_dir[k]), n=len(am),
                                     **{f"animal_{a}": float(am[i, k]) for i, a in enumerate(sorted(lvl))}))
    T = pd.DataFrame(rows)
    T["sig"] = False
    for _g, h in T.groupby(["group"]):
        T.loc[h.index, "sig"] = _bh(h.p.to_numpy())
    T["sig_3of4"] = T.sig & (T.n_dir >= np.minimum(3, T.n))
    return T


def plot(D: dict, levels: bool = False) -> None:
    """D = figure data: T (the family's area table), label. Main figure: pre LEVEL column (PRE_CMAP, own scale) + the
    three epoch - pre CHANGE columns (CHANGE_CMAP). levels=True: the supplementary <stem>_levels.png with the four epoch
    LEVEL columns on one shared scale (PRE_CMAP)."""
    T, label = D["T"], D["label"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = list(dict.fromkeys(T.group))
    allc = [("level", e) for e in EP] + [("change", e) for e in EP[1:]]
    cols = [("level", e) for e in EP] if levels else [("level", "pre")] + [("change", e) for e in EP[1:]]
    areas = sorted(set(T.area), key=lambda n: (n.rsplit("_", 1)[0], n.endswith("_right")))
    fig, axs = plt.subplots(1, len(groups), figsize=(2.3 * len(groups) + 2.6, 0.19 * len(areas) + 2.6), squeeze=False,
                            layout="constrained")
    n_ = len(areas)
    im_l = im_c = None
    for j, g in enumerate(groups):
        ax = axs[0, j]
        h = T[T.group == g]
        M = np.full((n_, len(allc)), np.nan)
        S = np.zeros_like(M, bool)
        for c, (kind, ep) in enumerate(allc):
            q = h[(h.kind == kind) & (h.epoch == ep)].set_index("area")
            M[:, c] = q.point.reindex(areas).to_numpy()
            S[:, c] = q.sig_3of4.reindex(areas).fillna(False).to_numpy(bool)
        lim = np.nanpercentile(np.abs(M), 98)                      # the scale as before (all seven columns)
        idx = [allc.index(c) for c in cols]
        M, S = M[:, idx], S[:, idx]
        if levels:
            lv = sym_lim([M], 98)
            im_l = ax.imshow(M, cmap=PRE_CMAP, vmin=-lv, vmax=lv, aspect="auto", interpolation="nearest",
                             extent=(-0.5, len(cols) - 0.5, n_ - 0.5, -0.5))
        else:
            lv = sym_lim([M[:, 0]], 98)
            im_l = ax.imshow(M[:, :1], cmap=PRE_CMAP, vmin=-lv, vmax=lv, aspect="auto", interpolation="nearest",
                             extent=(-0.5, 0.5, n_ - 0.5, -0.5))
            im_c = ax.imshow(M[:, 1:], cmap=CHANGE_CMAP, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest",
                             extent=(0.5, len(cols) - 0.5, n_ - 0.5, -0.5))
            ax.axvline(0.5, color="k", lw=0.8)
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_ylim(n_ - 0.5, -0.5)
        for r, c in zip(*np.nonzero(S)):
            ax.text(c, r, "*", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(cols)), [("" if k == "level" else "d ") + e[:3] for k, e in cols], fontsize=7,
                     rotation=90)
        ax.set_yticks(range(n_), areas if j == 0 else [], fontsize=6.5)
        ax.set_title(g, fontsize=8.5)
    if levels:
        cb = fig.colorbar(im_l, ax=axs[0, -1], fraction=0.12, pad=0.04, shrink=0.5)
        cb.set_label("epoch LEVEL (last panel's scale)", fontsize=7)
    else:
        cb = fig.colorbar(im_l, ax=axs[0, -1], fraction=0.12, pad=0.04, shrink=0.4)
        cb.set_label("pre LEVEL (last panel's scale)", fontsize=7)
        cb = fig.colorbar(im_c, ax=axs[0, -1], fraction=0.12, pad=0.04, shrink=0.4)
        cb.set_label("CHANGE d (last panel's scale)", fontsize=7)
    import textwrap
    if levels:
        head = (f"SUPPLEMENTARY to {FIG_NAME[label]}: ALLEN-AREA epoch LEVELS (not changes) of the Haufe patterns "
                f"({label}): the four epochs share one scale within each panel (purple-orange).")
    else:
        head = (f"ALLEN-AREA test of the Haufe patterns ({label}) -- {CMAP_NOTE}. Left column = pre level, d columns = "
                "change vs pre; the epoch levels are in the _levels supplementary.")
    paras = [head + " Held-out per-session patterns averaged per area; each panel has its own scale.",
             "* = nested bootstrap (animals > sessions) 95% CI excludes 0 after BH q 0.05 over areas x the 7 level + change "
             "columns AND >= 3 animals share the sign. _left = lesioned hemisphere. PROVISIONAL"]
    fig.suptitle("\n".join(textwrap.fill(q, int(12.5 * fig.get_size_inches()[0])) for q in paras), fontsize=9)
    out = FIG_NAME[label]
    fig.savefig(il.OUT / "figures" / (levels_name(out) if levels else out), dpi=100)
    plt.close(fig)


def plot_all(D: dict) -> None:
    """Main figure + the supplementary all-levels figure."""
    plot(D)
    plot(D, levels=True)


def main() -> int:
    if plot_only():
        for label, fn in FIG_NAME.items():
            if fd_path(fn).exists():
                plot_all(load_fd(fn))
            else:
                print(f"  (skip {label}: no figure data)")
        return 0
    A, keep = area_index()
    out = []
    srcs = [("position", "decoder_pattern_sessions.pkl", [(s, ax) for s in ("iti", "trial") for ax in ("lat", "sag")]),
            ("spout", "spout_pattern_sessions.pkl", None), ("spout_iti", "spout_pattern_sessions_iti.pkl", None)]
    from iti_spout_patterns import POS
    for label, fn, groups in srcs:
        f = il.OUT / fn
        if not f.exists():
            print(f"  (skip {label}: {fn} not built yet)")
            continue
        SM = pickle.loads(f.read_bytes())
        T = family(SM, groups or POS, label, A, keep)
        T.to_csv(il.OUT / f"pattern_area_stats_{label}.csv", index=False)
        D = dict(T=T, label=label)
        save_fd(FIG_NAME[label], D)
        plot_all(D)
        s = T[T.kind == "change"].groupby(["group", "epoch"]).sig_3of4.sum().unstack().reindex(columns=EP[1:])
        print(f"== {label}: number of areas (of {len(keep)}) whose change vs pre is significant (BH, >= 3/4)\n"
              + s.to_string())
        out.append(T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

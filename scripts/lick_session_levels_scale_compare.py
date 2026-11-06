"""UNSCALED vs Z-SCORED per-session Haufe LEVEL maps, side by side (Priya 10-27 late "yes do both", OVERNIGHT 5c-ter): are
the large cortex-wide components of the ITI level maps real, or an effect of the unscaled SVD features (the first
components carry most of the frame variance, so they dominate an unscaled pattern)?

Reads the two runs of `lick_session_level_maps.py` -- unscaled (session_levels/, figure data lick_session_levels_*.pkl)
and WFIELD_MAP_SCALE=frame (session_levels_z/, *_z.pkl: every SVD component / its session frame-wise SD, same licks,
same decoders, same test) -- nothing is refitted. Maps: LATERAL level maps (ITI / cued x all / first licks) and the ITI
and cued per-spout one-vs-rest level maps, pre and each epoch; per map the pooled map (mean of animals, as published)
with its TFCE FWER contour.
Number per map: GLOBAL SHARE = (mean over the stat mask)^2 / mean square of the map = the share of the map's power in
its uniform cortex-wide component (lick_session_level_maps.gshare; 1 = uniform). Per animal (rule 8) from each run's
per_animal.csv (the animal's epoch-mean map, `global_share`), and for the pooled map.
LOADING of the cortex-wide component (the map-level number cannot separate "real" from "scale" on its own: z-scoring
up-weights the many low-variance, high-spatial-frequency components whose own cortex mean is ~0, so a z-scored map has a
small global share almost by construction). Per session pattern A (feature space, before pixels): the SVD component with
the largest cortex mean (c0; component 0 in every animal, its own map ~80 % global) -- its |loading| / RMS loading over
the 200 components, unscaled (A) and z-scored (A / frame SD), and the percentile of its z loading. A z loading ~1 (a
typical percentile) = the cortex-wide component covaries with the lick target no more than any component does in its
own units, i.e. its dominance of the unscaled map is a variance (scale) effect; >> 1 = a real cortex-wide coupling.
Median over sessions per animal x map x epoch (scale_compare_c0.csv; panel 5 of the global figure).
Outputs <E>/iti_licks/session_levels_z/scale_compare.csv (animal x map x epoch: global share + cortex mean, unscaled vs
scaled) and scale_compare_pooled.csv; figures/lick_session_levels_scale_compare_{lat,spout_iti,spout_cued,global}.png;
figure data figdata/lick_session_levels_scale_compare.pkl (tables; the maps come from the two runs' figure data).
--plot-only redraws from figure data alone.
Usage: python scripts/lick_session_levels_scale_compare.py [--plot-only]   (after both lick_session_level_maps runs)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_session_level_maps as lv  # noqa: E402
from figdata import PRE_CMAP, load_fd, plot_only, save_fd, sym_lim  # noqa: E402

STEM = "lick_session_levels_scale_compare"
OUTZ = il.OUT / "session_levels_z"
FIG = il.OUT / "figures"
EP = lv.EP
AN = ["PS92", "PS93", "PS94", "PS95"]
VAR = (("", "unscaled"), ("_z", "z-scored"))
FAMS = {"lat": "lat", "spout_iti": "sp_iti", "spout_cued": "sp_trial"}


def _runs() -> dict:
    """{(fam, zs): figure data of the level figure} for both variants."""
    return {(fam, zs): load_fd(lv.ztag(lv.LEVEL_FIG0[fam], zs)) for fam in FAMS.values() for zs, _l in VAR}


def compute() -> dict:
    A0 = pd.read_csv(lv.BASE_OUTD / "per_animal.csv")
    A1 = pd.read_csv(OUTZ / "per_animal.csv")
    keys = ["animal", "key", "set", "sub", "target", "epoch"]
    cols = ["global_share", "global_share_debiased", "cortex_mean", "rms_mean", "rms_nonglobal", "r_pre"]
    C = A0[keys + cols].merge(A1[keys + cols], on=keys, suffixes=("_unscaled", "_z"), how="outer")
    C = C[(C.target == "lat") | C.target.str.startswith("sp_")].reset_index(drop=True)
    R = _runs()
    pooled = []
    for (fam, zs), D in R.items():
        for (k, ep, kind), r in D["R"].items():
            if kind != "level":
                continue
            pt = r["point"][np.isfinite(r["point"])]
            cx = np.isfinite(r["p"])
            pooled.append(dict(fam=fam, variant=zs or "unscaled", key=k, epoch=ep, n_animals=r["n"],
                               global_share=lv.gshare(pt), cortex_mean=float(pt.mean()),
                               frac_sig=float(((np.nan_to_num(r["p"], nan=1) < 0.05) & cx).sum() / cx.sum())))
    Pq = pd.DataFrame(pooled).pivot_table(index=["fam", "key", "epoch"], columns="variant",
                                          values=["global_share", "frac_sig", "n_animals"]).reset_index()
    Pq.columns = ["_".join(c for c in t if c) if isinstance(t, tuple) else t for t in Pq.columns]
    OUTZ.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUTZ / "scale_compare.csv", index=False)
    Pq.to_csv(OUTZ / "scale_compare_pooled.csv", index=False)
    pd.set_option("display.width", 250)
    print("== GLOBAL SHARE per animal (rule 8): unscaled -> z-scored, LATERAL level maps")
    L = C[C.target == "lat"].pivot_table(index=["set", "sub", "epoch"], columns="animal",
                                          values=["global_share_unscaled", "global_share_z"])
    print(L.reindex(EP, level="epoch").round(2).to_string())
    print("\n== GLOBAL SHARE per animal, per-spout maps (median over the six spouts)")
    S = C[C.target.str.startswith("sp_")].groupby(["set", "epoch", "animal"])[["global_share_unscaled", "global_share_z"]]
    print(S.median().unstack("animal").round(2).to_string())
    print("\n== pooled maps\n" + Pq.round(3).to_string(index=False))
    L0 = c0_loads()
    L0.to_csv(OUTZ / "scale_compare_c0.csv", index=False)
    print("\n== cortex-wide component c0: |loading| / RMS loading, median over sessions (u = unscaled, z = z-scored)")
    print(L0[L0.target == "lat"].pivot_table(index=["key", "animal"], columns="epoch", values=["u_load", "z_load"])
          .reindex(columns=EP, level=1).round(2).to_string())
    return {"per_animal": C, "pooled": Pq, "c0": L0}


def c0_loads() -> pd.DataFrame:
    """Per animal x map key x epoch: median over sessions of the cortex-mean component's |loading| / RMS loading,
    unscaled and z-scored, and of its z-loading percentile (module docstring)."""
    import pickle

    from iti_decoder_patterns import SPATIAL
    from figdata import grid_fns, imaging_grid
    P = pickle.loads((lv.BASE_OUTD / "patterns_gap2_enc.pkl").read_bytes())
    sd = pd.read_csv(il.OUT / "basis_svdbasis" / "frame_sd.csv", index_col=0)      # as lv.frame_scale
    inv = {x: 1.0 / sd.loc[x].to_numpy(float) for x in P}
    grid = imaging_grid()
    to_img, _ol = grid_fns(grid)
    g = to_img(np.arange(len(grid["pix"]), dtype=float))
    sel = g[np.isfinite(g)].astype(int)
    rows = []
    for a in AN:
        labs = sorted(x for x in P if P[x]["animal"] == a)
        if not labs:
            continue
        Wm = np.asarray(pickle.loads((SPATIAL / f"spatial_{a}.pkl").read_bytes())["P"], float)[sel]
        c0 = int(np.argmax(np.abs(Wm.mean(0))))
        own = float(Wm[:, c0].mean() ** 2 / (Wm[:, c0] ** 2).mean())
        for x in labs:
            for k, A in P[x]["A"].items():
                tg = k.split("/")[2]
                if not (tg == "lat" or tg.startswith("sp_")) or P[x]["fit"][k] < lv.MIN_FIT:
                    continue
                Az = A * inv[x]
                rows.append(dict(animal=a, label=x, epoch=P[x]["epoch"], key=k, target="lat" if tg == "lat" else "sp",
                                 c0=c0, c0_own_global=own, u_load=float(abs(A[c0]) / np.sqrt((A ** 2).mean())),
                                 z_load=float(abs(Az[c0]) / np.sqrt((Az ** 2).mean())),
                                 z_pct=float((np.abs(Az) < abs(Az[c0])).mean())))
    R = pd.DataFrame(rows)
    return (R.groupby(["animal", "key", "target", "epoch", "c0", "c0_own_global"])[["u_load", "z_load", "z_pct"]]
            .median().reset_index())


def _title(r, gs_an) -> str:
    fs = 100 * ((np.nan_to_num(r["p"], nan=1) < 0.05) & np.isfinite(r["p"])).sum() / np.isfinite(r["p"]).sum()
    pt = r["point"][np.isfinite(r["point"])]
    an = " ".join("-" if not np.isfinite(v) else f"{v:.2f}" for v in gs_an)
    return f"global {lv.gshare(pt):.2f}, sig {fs:.0f}%\nper animal {an}"


def plot_maps(F: dict, R: dict, name: str, fam: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = F["per_animal"]
    D0 = R[(fam, "")]
    keys = [k for k in D0["keys"] if any((k, e, "level") in D0["R"] for e in EP)]
    rows = [(k, zs, lab) for k in keys for zs, lab in VAR]
    fig, axs = plt.subplots(len(rows), 4, figsize=(12.5, 2.55 * len(rows) + 1.6), squeeze=False, layout="constrained")
    for i, (k, zs, lab) in enumerate(rows):
        D = R[(fam, zs)]
        lim = sym_lim([D["R"][(k, e, "level")]["point"] for e in EP if (k, e, "level") in D["R"]])
        im = None
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            r = D["R"].get((k, ep, "level"))
            ax.imshow(D["ol"], cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            if r is None:
                ax.set_title(f"{ep}: no data", fontsize=7)
                continue
            im = ax.imshow(r["point"], cmap=PRE_CMAP, vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(D["ol"], cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            lv._sig_contours(ax, r)
            c = C[(C.key == k) & (C.epoch == ep)].set_index("animal")
            col = "global_share_unscaled" if not zs else "global_share_z"
            ax.set_title(f"{ep}: " + _title(r, [c[col].get(a, np.nan) for a in AN]), fontsize=6.5)
            if j == 0:
                ax.set_ylabel(f"{lv._rowlab(k)}\n{lab.upper()}", fontsize=8)
        if im is not None:
            cb = fig.colorbar(im, ax=axs[i, :].tolist(), fraction=0.02, pad=0.01)
            cb.set_label("LEVEL / pre RMS", fontsize=6.5)
            cb.ax.tick_params(labelsize=6)
    what = ("LATERAL position level maps" if fam == "lat" else
            f"{lv.SETLAB[fam[3:]]} per-spout one-vs-rest level maps (all licks)")
    head = [f"NEURAL: UNSCALED vs Z-SCORED SVD features, per-session self-fit Haufe LEVEL maps -- {what}. Each pair of rows: "
            "unscaled (as published), then every SVD component divided by its session frame-wise SD (same licks, "
            "decoders, test; lick_session_level_maps.py WFIELD_MAP_SCALE=frame). Mean of animals, one scale per row.",
            "Title: GLOBAL SHARE = (cortex mean)^2 / mean square of the map (1 = uniform cortex-wide), pooled map and per "
            f"animal ({' '.join(AN)}); sig = % cortex TFCE FWER < 0.05 (solid contour; dashed = also >= 3 animals share "
            "the sign).", f"SIGN: {lv.SIGN['lat' if fam == 'lat' else 'sp']}. ipsi (lesion) = image LEFT. SUPPLEMENTARY, "
            "PROVISIONAL"]
    fig.suptitle("\n".join(textwrap.fill(q, 150) for q in head), fontsize=8)
    save_png_svg(fig, FIG / f"{STEM}_{name}.png", dpi=90)
    plt.close(fig)


def plot_global(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = F["per_animal"]
    cols = {"pre": "k", "acute": "#d6604d", "subacute": "#b8860b", "chronic": "#1b7837"}
    marks = dict(zip(AN, "osD^"))
    panels = [("iti", "lat", "ITI, lateral"), ("trial", "lat", "cued, lateral"), ("iti", "sp", "ITI, per spout"),
              ("trial", "sp", "cued, per spout")]
    fig, axs = plt.subplots(1, 5, figsize=(21, 4.6))
    for ax, (st, tg, lab) in zip(axs, panels):
        c = C[(C.set == st) & ((C.target == "lat") if tg == "lat" else C.target.str.startswith("sp_"))]
        c = (c[c["sub"] == "all"] if tg == "lat" else c).dropna(subset=["global_share_unscaled", "global_share_z"])
        for q in c.itertuples():
            ax.scatter(q.global_share_unscaled, q.global_share_z, color=cols[q.epoch], marker=marks[q.animal], s=22,
                       alpha=0.8, lw=0)
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("global share, UNSCALED", fontsize=8)
        ax.set_ylabel("global share, Z-SCORED", fontsize=8)
        k = (c.global_share_z < c.global_share_unscaled).sum()
        ax.set_title(f"{lab}{' (all licks)' if tg == 'lat' else ' (6 spouts)'}\nlower when z-scored: {k}/{len(c)} maps; "
                     f"median {c.global_share_unscaled.median():.2f} -> {c.global_share_z.median():.2f}", fontsize=8)
    for ep, col in cols.items():
        axs[0].scatter([], [], color=col, label=ep)
    for a, mk in marks.items():
        axs[0].scatter([], [], color="0.4", marker=mk, label=a)
    axs[0].legend(fontsize=7, frameon=False)
    ax = axs[4]
    L0 = F.get("c0")
    if L0 is not None and len(L0):
        q = L0[L0.key.isin(["iti/all/lat", "trial/all/lat"])]
        for i, k in enumerate(("iti/all/lat", "trial/all/lat")):
            for j, ep in enumerate(EP):
                for a in AN:
                    v = q[(q.key == k) & (q.epoch == ep) & (q.animal == a)]
                    if len(v):
                        x = i * 5 + j + 0.15 * (AN.index(a) - 1.5)
                        ax.scatter(x, v.u_load.iloc[0], color=cols[ep], marker=marks[a], s=22, alpha=0.35, lw=0)
                        ax.scatter(x, v.z_load.iloc[0], color=cols[ep], marker=marks[a], s=26, lw=0)
        ax.set_yscale("log")
        ax.axhline(1, color="0.5", ls="--", lw=0.8)
        ax.set_xticks([1.5, 6.5], ["ITI lateral", "cued lateral"], fontsize=8)
        ax.set_ylabel("|loading of cortex-wide comp.| / RMS loading", fontsize=8)
        ax.set_title("cortex-wide SVD component (comp 0) in the session pattern:\nfaint = unscaled, solid = z-scored "
                     "(1 = a typical component)", fontsize=8)
    fig.suptitle("NEURAL: share of each per-animal epoch-mean LEVEL map in its uniform cortex-wide component, unscaled vs "
                 "z-scored SVD features (each component / its session frame-wise SD). One dot = animal x map x epoch; "
                 "below the diagonal = the cortex-wide part shrinks when z-scored. SUPPLEMENTARY, PROVISIONAL", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_png_svg(fig, FIG / f"{STEM}_global.png", dpi=95)
    plt.close(fig)


def main() -> int:
    if plot_only():
        F = load_fd(STEM)
    else:
        F = compute()
        save_fd(STEM, F)
    R = _runs()
    for name, fam in FAMS.items():
        plot_maps(F, R, name, fam)
    plot_global(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Across-animal FACE MOVEMENT MAPS per spout position -- cue-, lick- and bout-onset-aligned -- with statistics on face
PATCHES (Priya 10-24: "the whole face moves less acutely across spout positions, with some L/R asymmetry"; make it
lick-aligned too, and test it).

Input: `face_rhythm_asym_aligned.py`'s cache (`asym_aligned/data/aligned_point_maps.npz`): per session x alignment x
spout position, each tracked mesh point's mean speed in the window (cue 0..+2 s; lick -0.1..+0.2 s, per lick; bout
onset 0..+0.5 s, the first lick of each bout), from the frozen point PCs.

Per animal: log2(session / mean of the animal's pre-stroke sessions) per point, averaged within the epoch. Statistics
(n = 4 animals; CLAUDE.md rule 8):
  * POINTS (default since 10-24 late; Priya: "I don't like the huge square bins -- smaller bins or per-point"): every
    mesh point tested on its own -- `analysis_kit.boot_delta_columns` (the SAME paired animals -> sessions draw as
    `boot_delta`, applied to all 1,282 points at once) of the log2 change vs pre; a point is MARKED when its 95 % interval
    excludes 0, >= 3 of 4 animals change the same way (Priya 10-24: "3/4 animals can be our threshold, not 4/4";
    ``--min-agree``), AND it survives Benjamini-Hochberg FDR < 0.05 over the face's
    points (bootstrap p from the same draws). Marked points are ringed in black.
  * PATCHES (``--unit patch``, the earlier version): mesh points binned into a grid of face patches (columns mirrored about the midline, so contra and ipsi
    patches pair up); per patch the session values' patch mean, then `analysis_kit.boot_delta` (paired animals ->
    sessions bootstrap, animal-weighted, 95 % interval) of the log2 change vs pre; a patch is MARKED when the interval
    excludes 0 AND >= ``--min-agree`` animals change the same way. Multiple patches are tested: read the marks as a map of where
    the change is consistent, not as independent discoveries.
  * HALVES: contra-half minus ipsi-half log2 change per session (the asymmetry of the change), same bootstrap -- the
    direct test of "the contralesional half drops more".

    python scripts/face_rhythm_aligned_maps.py [--set master_pre_v1] [--unit point|patch] [--grid 4 7] [--plot-only]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/aligned_maps/``: aligned_maps_<align>.png (rows = positions, cols = acute /
subacute / chronic; animal-mean log2 change per point, consistent patches outlined), halves_<align>.png (contra - ipsi
change per position x epoch, animals + mean +- SEM + interval), data/point_stats.csv (per alignment x position x epoch: marked points, contra / ipsi), data/patch_stats.csv (patch
unit), data/halves_stats.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402
from wfield_local.face_rhythm import stats as frs  # noqa: E402

EPOCHS = frc.EPOCHS
MIN_EVENTS = 5
UNIT_POINT = ("BLACK RINGS: mesh POINTS whose change vs pre has a paired animals -> sessions bootstrap 95 % interval "
              "(one draw for all points: `analysis_kit.boot_delta_columns`) excluding 0, the same sign as the change in "
              ">= {agree} of the animals, AND Benjamini-Hochberg FDR < {fdr} over the face (bootstrap p).")


def patches(home: np.ndarray, mid: float, ncol_half: int, nrow: int) -> tuple[np.ndarray, dict]:
    """Patch id per point: columns of equal width mirrored about the midline (column k on the contra side pairs with
    column k on the ipsi side), rows of equal height. -> (ids, grid geometry for `outline`)."""
    x, y = home[:, 0], home[:, 1]
    half = max(mid - x.min(), x.max() - mid)
    cw = half / ncol_half
    rh = (y.max() - y.min()) / nrow
    col = np.minimum((np.abs(x - mid) // cw).astype(int), ncol_half - 1)
    side = np.where(x <= mid, 0, 1)                                     # 0 = contra (image left on cam4)
    row = np.minimum(((y - y.min()) / rh).astype(int), nrow - 1)
    return (side * ncol_half + col) * nrow + row, {"mid": mid, "cw": cw, "rh": rh, "y0": float(y.min()),
                                                    "ncol": ncol_half, "nrow": nrow}


def outline(ax, sig_ids: list[int], g: dict, **kw) -> None:
    """Draw the BORDER of the union of significant patches (edges between a significant and a non-significant cell)."""
    nc, nr = g["ncol"], g["nrow"]
    G = np.zeros((2 * nc, nr), bool)                                    # grid x index: contra cols mirrored to the left
    for q in sig_ids:
        side, rem = divmod(q, nc * nr)
        c, r = divmod(rem, nr)
        G[nc - 1 - c if side == 0 else nc + c, r] = True
    x0 = g["mid"] - nc * g["cw"]
    X = lambda i: x0 + i * g["cw"]                                      # noqa: E731
    Y = lambda j: g["y0"] + j * g["rh"]                                 # noqa: E731
    for i in range(2 * nc):
        for j in range(nr):
            if not G[i, j]:
                continue
            if i == 0 or not G[i - 1, j]:
                ax.plot([X(i), X(i)], [Y(j), Y(j + 1)], **kw)
            if i == 2 * nc - 1 or not G[i + 1, j]:
                ax.plot([X(i + 1), X(i + 1)], [Y(j), Y(j + 1)], **kw)
            if j == 0 or not G[i, j - 1]:
                ax.plot([X(i), X(i + 1)], [Y(j), Y(j)], **kw)
            if j == nr - 1 or not G[i, j + 1]:
                ax.plot([X(i), X(i + 1)], [Y(j + 1), Y(j + 1)], **kw)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import analysis_kit as ak
    from wfield_local import config
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--grid", type=int, nargs=2, default=(4, 7), help="patch columns per face half, rows")
    ap.add_argument("--min-points", type=int, default=10, help="a patch needs this many mesh points")
    ap.add_argument("--unit", default="point", choices=("point", "patch"), help="map statistics per mesh point or patch")
    ap.add_argument("--patch-boot", type=int, default=2000, help="bootstrap draws per patch / point test (halves: 4000)")
    ap.add_argument("--fdr", type=float, default=0.05, help="point unit: Benjamini-Hochberg q over the face's points")
    ap.add_argument("--min-agree", type=int, default=3,
                    help="animals whose own change has the sign of the pooled change (Priya 10-24: 3 of 4, not 4 of 4)")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw from data/aligned_maps_cache.pkl (statistics + maps), no bootstrap")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    root = Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set
    out = a.out or root / "aligned_maps"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    Z = np.load(figure_layout.sidecar_for(root / "asym_aligned", "aligned_point_maps", ".npz", mkdir=False))
    maps, counts, home = Z["maps"].astype(float), Z["counts"], Z["home"]
    animal, epoch, label = Z["animal"].astype(str), Z["epoch"].astype(str), Z["label"].astype(str)
    aligns, positions, windows = list(Z["aligns"].astype(str)), list(Z["positions"].astype(str)), Z["windows"]
    mid = float(frr.MASTER_MIDLINE_X)
    contra = home[:, 0] <= mid
    pid, grid = patches(home, mid, *a.grid)
    keep = [q for q in np.unique(pid) if (pid == q).sum() >= a.min_points]
    cache = figure_layout.sidecar_for(out, f"aligned_maps_cache_{a.unit}", ".pkl")   # statistics + maps: --plot-only
    if a.plot_only:
        P, H, mean_maps, marks = pd.read_pickle(cache)
    else:
        rng = np.random.default_rng(0)            # map tests (patches / points)
        rng_half = np.random.default_rng(1)       # halves: own stream, so they do not depend on the map unit

        # one value per SESSION (a split video, PS92_0812 v1 / v2, averaged first), log2 vs the animal's pre mean
        lab_u = sorted(set(label))
        rows_patch, rows_half, mean_maps, marks = [], [], {}, {}
        for ai, al in enumerate(aligns):
            for pi, p in enumerate(positions):
                V, an_s, ep_s = [], [], []
                for lb in lab_u:
                    ii = np.flatnonzero((label == lb) & (counts[:, ai, pi] >= MIN_EVENTS))
                    if len(ii):
                        V.append(np.nanmean(maps[ii, ai, pi], 0))
                        an_s.append(animal[ii[0]])
                        ep_s.append(epoch[ii[0]])
                V, an_s, ep_s = np.array(V), np.array(an_s), np.array(ep_s)
                lg = np.log2(np.maximum(V, 1e-9))
                for an in np.unique(an_s):
                    m = an_s == an
                    pre = m & (ep_s == "pre")
                    if pre.any():
                        lg[m] -= lg[pre].mean(0)
                    else:
                        lg[m] = np.nan
                for ep in EPOCHS[1:]:
                    per = [np.nanmean(lg[(an_s == an) & (ep_s == ep)], 0) for an in np.unique(an_s)
                           if ((an_s == an) & (ep_s == ep)).any()]
                    if per:
                        mean_maps[(al, p, ep)] = np.nanmean(per, 0)
                # points: every mesh point at once, the same draw for all of them
                if a.unit == "point":
                    an_u = np.unique(an_s)
                    pre_d = {an: lg[(an_s == an) & (ep_s == "pre")] for an in an_u
                             if ((an_s == an) & (ep_s == "pre")).any()}
                    for ep in EPOCHS[1:]:
                        post_d = {an: lg[(an_s == an) & (ep_s == ep)] for an in an_u
                                  if ((an_s == an) & (ep_s == ep)).any()}
                        res = ak.boot_delta_columns(post_d, pre_d, rng, a.patch_boot)
                        if res is None:
                            continue
                        obs, lo, hi, o, _n = res
                        per = np.array([post_d[an].mean(0) for an in sorted(post_d)])
                        agree = np.where(obs < 0, (per < 0).sum(0), (per > 0).sum(0))
                        same = agree >= a.min_agree
                        excl = (lo > 0) | (hi < 0)
                        sig = excl & same & ak.bh(ak.boot_p_columns(o), a.fdr)
                        marks[(al, p, ep)] = sig
                        rows_patch.append({"align": al, "position": p, "epoch": ep,
                                           "n_points": int(np.isfinite(obs).sum()), "marked": int(sig.sum()),
                                           "marked_contra": int((sig & contra).sum()),
                                           "marked_ipsi": int((sig & ~contra).sum()),
                                           "marked_down": int((sig & (obs < 0)).sum()),
                                           "marked_up": int((sig & (obs > 0)).sum()),
                                           "excl0_same_sign_uncorrected": int((excl & same).sum())})
                # patches
                for q in (keep if a.unit == "patch" else []):
                    vq = np.nanmean(lg[:, pid == q], 1)
                    pre = {an: list(vq[(an_s == an) & (ep_s == "pre")]) for an in np.unique(an_s)}
                    for ep in EPOCHS[1:]:
                        post = {an: list(vq[(an_s == an) & (ep_s == ep)]) for an in np.unique(an_s)
                                if ((an_s == an) & (ep_s == ep)).any()}
                        iv = ak.boot_delta(post, pre, rng, n_boot=a.patch_boot)
                        per = [float(np.mean(v)) for v in post.values()]
                        same = bool(iv) and sum((x < 0) if iv.point < 0 else (x > 0) for x in per) >= a.min_agree
                        sig = bool(iv and (iv.lo > 0 or iv.hi < 0) and same)
                        rows_patch.append({"align": al, "position": p, "epoch": ep, "patch": int(q),
                                           "side": "contra" if q < a.grid[0] * a.grid[1] else "ipsi",
                                           "delta_log2": iv.point if iv else np.nan, "ci_lo": iv.lo if iv else np.nan,
                                           "ci_hi": iv.hi if iv else np.nan, "n_animals": len(per),
                                           "all_same_sign": same, "marked": sig})
                        if sig:
                            marks.setdefault((al, p, ep), []).append(int(q))
                # halves: contra - ipsi change per session
                d = np.nanmean(lg[:, contra], 1) - np.nanmean(lg[:, ~contra], 1)
                pre = {an: list(d[(an_s == an) & (ep_s == "pre")]) for an in np.unique(an_s)}
                for ep in EPOCHS[1:]:
                    post = {an: list(d[(an_s == an) & (ep_s == ep)]) for an in np.unique(an_s)
                            if ((an_s == an) & (ep_s == ep)).any()}
                    iv, draws = ak.boot_delta(post, pre, rng_half, draws=True)  # same draws as without `draws`
                    per = {an: float(np.mean(v)) for an, v in sorted(post.items())}
                    whole = {an: float(np.nanmean(lg[(an_s == an) & (ep_s == ep)])) for an in per}
                    rows_half.append({"align": al, "position": p, "epoch": ep, **{f"{an}": v for an, v in per.items()},
                                      "contra_minus_ipsi": iv.point if iv else np.nan, "ci_lo": iv.lo if iv else np.nan,
                                      "ci_hi": iv.hi if iv else np.nan,
                                      "excludes_0": bool(iv and (iv.lo > 0 or iv.hi < 0)),
                                      "p_boot": ak.boot_p(draws) if iv else np.nan,
                                      "n_contra_lower": sum(v < 0 for v in per.values()),
                                      "whole_face_log2": float(np.mean(list(whole.values()))),
                                      "n_whole_lower": sum(v < 0 for v in whole.values())})
        P = pd.DataFrame(rows_patch)
        # Priya 10-24 marks: * = interval excludes 0; ** = also Holm over the 6 positions of an alignment x epoch
        H = frs.add_holm(pd.DataFrame(rows_half), ["align", "epoch"])
        P.to_csv(figure_layout.sidecar_for(out, f"{a.unit}_stats", ".csv"), index=False, float_format="%.4g")
        H.to_csv(figure_layout.sidecar_for(out, "halves_stats", ".csv"), index=False, float_format="%.4g")
        pd.to_pickle((P, H, mean_maps, marks), cache)
    pd.set_option("display.width", 220)
    an_cols = [c for c in H.columns if c.startswith("PS")]
    print("HALVES: contra - ipsi log2 change per animal (rule 8 first), then the paired bootstrap; whole-face change")
    print(H[["align", "position", "epoch", *an_cols, "contra_minus_ipsi", "ci_lo", "ci_hi", "excludes_0",
             "n_contra_lower", "whole_face_log2", "n_whole_lower"]].round(3).to_string(index=False))
    print(f"\nMARKED {a.unit}s per alignment x epoch (interval excludes 0, >= {a.min_agree} animals agree"
          + (f", BH FDR < {a.fdr})" if a.unit == "point" else ")"), "summed over 6 positions:")
    print(P.groupby(["align", "epoch"]).marked.sum().unstack().reindex(columns=list(EPOCHS[1:])).to_string())

    with frf.style():
        for ai, al in enumerate(aligns):
            win = windows[ai]
            fig, axs = frf.subplots(len(positions), 3, figsize=(10.5, 3.1 * len(positions)), squeeze=False)
            sc = None
            for r_i, p in enumerate(positions):
                for c_i, ep in enumerate(EPOCHS[1:]):
                    ax = axs[r_i, c_i]
                    ax.axis("off")
                    m = mean_maps.get((al, p, ep))
                    if m is None:
                        continue
                    sc = ax.scatter(home[:, 0], home[:, 1], c=m, s=4, cmap="RdBu_r", vmin=-1, vmax=1)
                    if a.unit == "point":
                        sg = marks.get((al, p, ep), np.zeros(len(home), bool))
                        ax.scatter(home[sg, 0], home[sg, 1], s=7, facecolors="none", edgecolors="k", linewidths=0.35)
                    else:
                        outline(ax, marks.get((al, p, ep), []), grid, color="k", lw=1.0)
                    ax.axvline(mid, color="k", lw=0.4)
                    ax.invert_yaxis()
                    ax.set_aspect("equal")
                    h = H[(H["align"] == al) & (H.position == p) & (H.epoch == ep)].iloc[0]
                    ax.set_title(f"{p} -- {ep}\nwhole {h.whole_face_log2:+.2f} ({int(h.n_whole_lower)}/4 down); "
                                 f"C-I {h.contra_minus_ipsi:+.2f}{h['mark']}", fontsize=8)
            if sc is not None:
                fig.colorbar(sc, ax=axs, shrink=0.3, label="log2(epoch / pre), mean of animals")
            fig.suptitle(f"{a.set}: face movement per spout position, {al}-aligned ({win[0]:+g}..{win[1]:+g} s) vs pre")
            marks_text = (UNIT_POINT.format(fdr=a.fdr, agree=a.min_agree) if a.unit == "point" else
                          f"Black OUTLINES: the border of the face PATCHES ({a.grid[0]} columns per half mirrored about the "
                          f"midline x {a.grid[1]} rows, >= {a.min_points} points each) whose change vs pre has a paired "
                          f"animals -> sessions bootstrap 95 % interval excluding 0 AND >= {a.min_agree} animals agreeing "
                          "(many patches tested: a map of consistency).")
            frf.save(fig, out / f"aligned_maps_{al}.png", f"""
Per mesh point (image LEFT = CONTRALESIONAL on cam4, lesions left; vertical line = face midline): mean speed in the
{al.upper()}-aligned window ({win[0]:+g}..{win[1]:+g} s; cue = engaged-trial cue; lick = every LP lick in engaged trials,
per lick; bout = the first lick of each bout), per spout position (rows; R = contralesional spouts), as log2(session /
the animal's pre-stroke mean), averaged within the epoch per animal, then over the four animals. {marks_text}
The colour gives the sign. Panel titles: whole-face mean log2 change (animals lower than pre) and the
CONTRA - IPSI half difference of the change with its bootstrap (* = interval excludes 0; ** = also Holm over the 6
positions). Data: data/{a.unit}_stats.csv, data/halves_stats.csv.""")

            fig, axs = frf.subplots(1, len(positions), figsize=(3.0 * len(positions), 3.8), squeeze=False, sharey=True)
            x = np.arange(3)
            for c_i, p in enumerate(positions):
                ax = axs[0, c_i]
                hh = H[(H["align"] == al) & (H.position == p)].set_index("epoch").reindex(list(EPOCHS[1:]))
                for an in an_cols:
                    ax.plot(x, hh[an].to_numpy(), "o-", color=config.animal_color().get(an, "0.6"), ms=3, lw=0.9,
                            alpha=0.8, label=an)
                vals = hh[an_cols].to_numpy(float)
                ax.errorbar(x, np.nanmean(vals, 1), yerr=np.nanstd(vals, 1, ddof=1) / np.sqrt(np.isfinite(vals).sum(1)),
                            fmt="o", color="k", capsize=3, ms=5)
                ax.axhline(0, color="k", lw=0.5)
                ax.set_xticks(x, [f"{e[:5]}\n{r.contra_minus_ipsi:+.2f}{r['mark']}\n"
                                  f"{int(r.n_contra_lower)}/4 C<I" for e, r in hh.iterrows()], fontsize=7)
                ax.set_title(p, fontsize=9)
            axs[0, 0].set_ylabel("contra - ipsi log2 change")
            fig.suptitle(f"{a.set}: is the contralesional half's change larger? ({al}-aligned)")
            frf.save(fig, out / f"halves_{al}.png", f"""
Per session: mean log2 change vs the animal's pre of the CONTRALESIONAL half of the face minus that of the IPSILESIONAL
half ({al}-aligned window); per animal epoch means (repo animal colours), mean +- SEM across animals (black); under each epoch the
paired animals -> sessions bootstrap of the change vs pre (* = 95 % interval excludes 0; ** = also significant after
Holm over the 6 positions of this alignment x epoch, bootstrap p from the same draws) and how many animals have the
contralesional half changing more negatively (C<I). Negative = the contralesional half lost more movement.""")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

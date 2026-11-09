"""Whole-cortex view of the self-initiated-lick response (Priya 10-26: "should we also do some non-orofacial per-area
activity maps to look at the whole cortex question? (supplemental figure with VIS areas)").

From the saved ITI sessions (`scripts/iti_licks.py`, every Allen area per bout-start ITI lick, >= 2 s without licking):
per session x area, response AUC (0..0.5 s after lick onset = first frame the tongue is visible) and build-up AUC
(-0.4..-0.05 s), baseline -1.5..-1.0 s.
  lick_iti_area_maps.png   change from pre per area in RAW dF/F AUC (acute / subacute / chronic) on the Allen atlas, one color
                      scale per row; areas whose nested-bootstrap CI (animals > sessions) excludes 0 are outlined (black).
                      (10-26: per-area normalization by the area's own pre response was dropped -- small caudal areas
                      with near-zero pre responses gave huge ratios of either sign and dominated the maps.)
  lick_iti_area_peth.png   per-session mean PETH +- SEM (all animals pooled) for non-orofacial areas, both hemispheres.
  lick_iti_area_groups.png orofacial (ALM, tjM1, MOs, MOp, SSp-m, SSp-n) vs visual + retrosplenial areas: one dot per
                      session; group-mean AUCs divided by the animal's pre GROUP-mean response AUC.
  lick_iti_area_compare.png orofacial ROIs next to control areas (`compare`).
Image left = ipsilesional (_left); _right = contralesional.
FIGURE DATA (Priya 10-27): each figure's drawn numbers are saved to <OUT>/figdata/<figure stem>.pkl (the area image +
names + outline for the maps); `--plot-only` redraws from those files alone (no session load, no bootstrap).
Usage: python scripts/iti_area_maps.py [--compare] [--plot-only]
       python scripts/iti_area_maps.py --split [--cued] [--spout far_contra] [--plot-only]   (ALM / tjM1 separate; one previous spout)
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402
from fr_musall import AN  # noqa: E402

PETH_AREAS = ["VISp", "VISl", "VISam", "RSPd", "SSp-bfd", "SSp-ul", "SSp-ll", "SSp-tr", "ACAd", "PL"]
ORO = ("ALM", "tjM1", "MOs", "MOp", "SSp-m", "SSp-n")
VISR = ("VIS", "RSP")
GROUP_ROWS = (("resp_n", "response AUC / pre"), ("build_n", "build-up AUC / pre response"))
GROUP_COLS = ("orofacial", "visual + RSP", "orofacial - visual + RSP")


WOUT = il.OUT_W if il.ITI_SUBSET else il.OUT     # subset runs (dock, ...) write to their own folder (10-29)
RESP_W = (0.0, 1.0)      # response AUC window (0..1.0 s default since 10-28 evening; 0..0.5 s before)


def _wide(x: str) -> dict:
    """The long -2..+3 s traces of `lick_cued_ramp` (cued_wide/<label>.npz; same area columns as the session file,
    validated there): the stored PETHs end at +0.75 s, too short for the 0..1.0 s response window."""
    import lick_cued_ramp as cr
    z = np.load(cr.RIN / cr.WIDE / f"{x}.npz")
    return {k: z[k] for k in ("lick", "cue", "pos", "iti", "cols")}


def _resp_wide(X: np.ndarray, base: tuple) -> np.ndarray:
    """(licks, LAGS_W, areas) -> session-mean response AUC over RESP_W per area, after the per-lick baseline `base`."""
    import lick_cued_ramp as cr
    X = X.astype(float)
    X = X - np.nanmean(X[:, (cr.LAGS_W >= base[0]) & (cr.LAGS_W <= base[1]), :], 1, keepdims=True)
    mu = np.nanmean(X, 0)
    w = (cr.LAGS_W >= RESP_W[0]) & (cr.LAGS_W <= RESP_W[1])
    return np.trapezoid(mu[w], cr.LAGS_W[w], axis=0)


def session_table(D: dict) -> tuple[pd.DataFrame, dict]:
    rows, peths = [], {}
    for x, d in D.items():
        m = d["bout_gap"] >= il.GAP_MAIN
        if m.sum() < 10:
            continue
        P = d["bout_peth"][m].astype(float)
        P = P - P[:, (il.LAGS_P >= il.BASE[0]) & (il.LAGS_P <= il.BASE[1]), :].mean(1, keepdims=True)
        mu = P.mean(0)                                    # (lags, areas)
        rw = _resp_wide(_wide(x)["iti"][m], il.BASE_RESP)   # response 0..1.0 s, baseline -1.0..-0.5 s
        for j, c in enumerate(d["cols"]):
            if c == "cortex":
                continue
            rows.append({"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "area": str(c),
                         "resp": rw[j], "resp_05": il._auc(mu[:, j], 0.0, 0.5),
                         "build": il._auc(mu[:, j], -0.4, -0.05)})
            peths[(x, str(c))] = mu[:, j]
    T = pd.DataFrame(rows)
    den = T[T.epoch == "pre"].groupby(["area", "animal"]).resp.mean()
    d_ = np.array([den.get((a_, n_), np.nan) for a_, n_ in zip(T.area, T.animal)])
    d_ = np.where(np.abs(d_) > 0, d_, np.nan)
    T["resp_n"], T["build_n"] = T.resp / d_, T.build / d_
    return T, peths


def split_image(label: str) -> tuple[np.ndarray, list]:
    """Allen area image with ALM / tjM1 painted inside MOs / MOp (the v8 ROIs on the stat mask, `iti_licks.roi_masks`)
    and the remaining parent pixels renamed MOs_rest / MOp_rest (Priya 10-28: "ALM / tjM1 separate here")."""
    from frozen_area_maps import area_image

    from wfield_local import combined_encoding as ce
    img, nm = area_image(label)
    _V, _ft, _Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    nm = list(nm)
    flat_img = img.ravel().copy()
    for src, dst in il.SPLIT_PAIRS:
        for h in il.HEMIS:
            par = nm.index(f"{src}_{h}")
            nm[par] = f"{src}_rest_{h}"
            nm.append(f"{dst}_{h}")
            sub = np.zeros(flat_img.shape, bool)
            sub[flat[M[f"{dst}_{h}"]]] = True
            flat_img[sub & (flat_img == par)] = len(nm) - 1
    return trim_rest(flat_img.reshape(img.shape), nm), nm


def trim_rest(img: np.ndarray, nm: list) -> np.ndarray:
    """The map paints the full ATLAS area, but the ROI v8 masks exclude an edge rim from every area mean; lateral to
    ALM / tjM1 that rim showed up as a jagged MOs / MOp-rest sliver (checked 10-28: 134-159 atlas pixels per parent
    lateral to the sub-ROI, NONE used in any value). Blank them in the DISPLAY (morphological opening of each rest
    area, main body kept; Priya 10-28: "take out the jagged slivers at the edges").
    Idempotent, so draw time applies it to images saved before 10-28 evening."""
    from scipy import ndimage
    out = img.copy()
    for k, n in enumerate(nm):
        if "_rest_" in n:
            m = out == k
            keep = ndimage.binary_opening(m, iterations=4)
            lab, nlab = ndimage.label(keep)
            if nlab > 1:
                sizes = ndimage.sum(keep, lab, range(1, nlab + 1))
                keep = lab == (1 + int(np.argmax(sizes)))
            out[m & ~keep] = -1
    return out


def session_table_split(D: dict, spout: str | None = None, min_n: int = 10, cued: bool = False) -> pd.DataFrame:
    """As session_table, with MOs / MOp replaced by ALM + MOs_rest and tjM1 + MOp_rest (`iti_licks.with_rest`, exact).
    ITI: bout starts (gap >= GAP_MAIN); spout = only those whose PREVIOUS trial was at that spout
    (`lick_cued_ramp.bout_prev_pos`). cued (Priya 10-28 "cued-lick version of 14.14"): the FIRST post-cue lick of each
    engaged trial (`first_peth`, onset-aligned; same baseline) + the cue response (`first_cue`, 0..0.5 s after the cue,
    baseline -1.0..-0.5 s); spout = target spout (`first_pos`)."""
    from lick_cued_ramp import BASE_C, bout_prev_pos
    rows = []
    for x, d in D.items():
        if cued:
            m = np.ones(len(d["first_rt"]), bool)
            if spout is not None:
                m &= np.asarray(d["first_pos"], str) == spout
        else:
            m = d["bout_gap"] >= il.GAP_MAIN
            if spout is not None:
                m &= bout_prev_pos(x, np.asarray(d["bout_t"], float)) == spout
        if m.sum() < min_n:
            continue
        P = d["first_peth" if cued else "bout_peth"][m].astype(float)
        P = P - P[:, (il.LAGS_P >= il.BASE[0]) & (il.LAGS_P <= il.BASE[1]), :].mean(1, keepdims=True)
        mu, cols = il.with_rest(P.mean(0), d["cols"], x, basis="split")      # build-up / pre-lick window (unchanged)
        # response (and cue) AUC over RESP_W = 0..1.0 s from the long traces, response baseline -1.0..-0.5 s (10-28)
        Wd = _wide(x)
        if cued:
            mw = np.ones(len(Wd["pos"]), bool) if spout is None else np.asarray(Wd["pos"], str) == spout
            if mw.sum() < min_n:
                continue
            rw = _resp_wide(Wd["lick"][mw], il.BASE_RESP)
            cw = _resp_wide(Wd["cue"][mw], BASE_C)
            rw, _ = il.with_rest(rw[None, :], d["cols"], x, basis="split")
            cw, _ = il.with_rest(cw[None, :], d["cols"], x, basis="split")
            rw, cw = rw[0], cw[0]
        else:
            rw, _ = il.with_rest(_resp_wide(Wd["iti"][m], il.BASE_RESP)[None, :], d["cols"], x, basis="split")
            rw = rw[0]
        for j, c in enumerate(cols):
            if c == "cortex" or c.rsplit("_", 1)[0] in ("MOs", "MOp"):
                continue
            r = {"label": x, "animal": str(d["animal"]), "epoch": str(d["epoch"]), "area": str(c),
                 "resp": rw[j], "resp_05": il._auc(mu[:, j], 0.0, 0.5), "build": il._auc(mu[:, j], -0.4, -0.05),
                 "n": int(m.sum())}
            if cued:
                r["cue"] = cw[j]
            rows.append(r)
    return pd.DataFrame(rows)


def split_maps(spout: str | None = None, cued: bool = False) -> None:
    """lick_{iti,cued}_area_maps_split[_<spout>].png: the 14.14 map with ALM / tjM1 / MOs_rest / MOp_rest areas; spout =
    ITI bout starts after a trial at that spout / cued first licks to that target (>= 5 per session)."""
    from frozen_area_maps import outline
    labels = il.labels()
    D = {x: il.subset_bouts(il._load(x), x) for x in labels}
    T = session_table_split(D, spout, min_n=10 if spout is None else 5, cued=cued)
    kind = "cued" if cued else "iti"
    tag = "" if spout is None else "_" + spout.replace("_", "")
    stem = f"lick_{kind}_area_maps_split{tag}"
    ctag = "_cued" if cued else ""
    T.to_csv(WOUT / f"area_auc_sessions_split{ctag}{tag}.csv", index=False)
    vals = [("resp", "lick response AUC 0..1.0 s" if cued else "response AUC 0..1.0 s"),
            ("build", "pre-lick AUC -0.4..-0.05 s (overlaps the CUE)" if cued else "build-up AUC -0.4..-0.05 s")]
    if cued:
        vals.append(("cue", "cue response AUC 0..1.0 s after the cue"))
    C = pd.concat([changes(T, v) for v, _t in vals])
    C.to_csv(WOUT / f"area_auc_changes_split{ctag}{tag}.csv", index=False)
    print("== per animal (rule 8), response AUC, ALM / tjM1 / rest\n" + T[T.area.str.contains("ALM|tjM1|rest")]
          .pivot_table(index=["area", "animal"], columns="epoch", values="resp").reindex(columns=il.EP4).round(1)
          .to_string())
    print(C[C.sig].sort_values(["value", "epoch", "delta"]).round(3).to_string(index=False))
    img, nm = split_image(labels[0])
    rows = []
    for val, ttl in vals:
        c = C[C.value == val]
        lim = np.nanpercentile(np.abs(c.delta), 98) or 1.0
        pans = []
        for ep in il.EP4[1:]:
            ce_ = c[c.epoch == ep].set_index("area")
            pans.append((ep, ce_.delta.to_dict(), list(ce_.index[ce_.sig]), int(ce_.sig.sum())))
        rows.append((ttl, lim, pans))
    sp = "" if spout is None else spout.replace("_", "-")
    if cued:
        what = "CUED licks: first post-cue lick of each engaged trial" + (f", {sp} target only" if sp else "")
    else:
        what = "self-initiated (ITI) licks, " + (f"bout starts after a {sp} trial (previous spout)" if sp
                                                 else "all bout starts")
    save_fd(stem, {"img": img, "nm": nm, "ol": outline(img), "rows": rows, "what": what, "levels": _levels(T, vals)})
    draw_maps(stem)
    draw_levels(stem)


def draw_maps(stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import paint
    F = load_fd(stem)
    from frozen_area_maps import outline
    nm = F["nm"]
    img = trim_rest(F["img"], nm)
    ol = outline(img)
    nr = len(F["rows"])
    fig, axs = plt.subplots(nr, 3, figsize=(14, 4.5 * nr), squeeze=False)
    for i, (ttl, lim, pans) in enumerate(F["rows"]):
        for j, (ep, delta, sig_areas, n_sig) in enumerate(pans):
            ax = axs[i, j]
            im = ax.imshow(paint(img, nm, delta), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            _sig_halo(ax, img, nm, sig_areas, list(delta))
            ax.set_title(f"{ttl}\n{ep} - pre ({n_sig} areas significant)", fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01).set_label("change in dF/F AUC (x1e3 s)")
    what = F.get("what", "self-initiated (ITI) licks, all bout starts")
    fig.suptitle(f"NEURAL, NO DECODER: {what} -- change from pre per area in raw dF/F AUC (x1e3 s), with ALM / tjM1 separate "
                 "from the rest of MOs / MOp (ROI v8); black outline on a white halo = nested-bootstrap CI excludes 0. Image left = "
                 "ipsilesional.", fontsize=10, wrap=True)
    save_png_svg(fig, WOUT / "figures" / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def _levels(T: pd.DataFrame, vals) -> list:
    """Per-epoch LEVEL maps (Priya 10-28: "collapsible supplementals of the per-epoch responses, not the deltas"): per
    value, per epoch (pre, acute, subacute, chronic), area -> mean over animals of the animal's session mean; one color
    limit per value (98th percentile of |level| over all epochs)."""
    out = []
    for val, ttl in vals:
        L = T.groupby(["area", "epoch", "animal"])[val].mean().groupby(["area", "epoch"]).mean()
        lim = np.nanpercentile(np.abs(L.to_numpy()), 98) or 1.0
        out.append((ttl, lim, [(ep, L.xs(ep, level="epoch").to_dict() if ep in L.index.get_level_values("epoch")
                                else {}) for ep in il.EP4]))
    return out


def draw_levels(stem: str) -> None:
    """<stem>_levels.png from figdata <stem> (key "levels"): rows = values, columns = pre / acute / subacute / chronic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import paint
    F = load_fd(stem)
    if "levels" not in F:
        return
    from frozen_area_maps import outline
    nm = F["nm"]
    img = trim_rest(F["img"], nm)
    ol = outline(img)
    nr = len(F["levels"])
    fig, axs = plt.subplots(nr, 4, figsize=(18, 4.5 * nr), squeeze=False)
    for i, (ttl, lim, pans) in enumerate(F["levels"]):
        for j, (ep, lev) in enumerate(pans):
            ax = axs[i, j]
            im = ax.imshow(paint(img, nm, lev), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ax.set_title(f"{ttl}\n{ep}", fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axs[i, :], fraction=0.015, pad=0.01).set_label("dF/F AUC (x1e3 s)")
    what = F.get("what", "self-initiated (ITI) licks, all bout starts")
    fig.suptitle(f"NEURAL, NO DECODER: {what} -- per-epoch LEVEL per area (mean over animals of each animal's session mean; "
                 "not the change from pre), raw dF/F AUC (x1e3 s); one color scale per row. Image left = ipsilesional. "
                 "Descriptive.", fontsize=10, wrap=True)
    save_png_svg(fig, WOUT / "figures" / f"{stem}_levels.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _sig_halo(ax, img: np.ndarray, nm: list, sig_areas, shown_areas) -> None:
    """Significant areas: EACH outlined on its own (a black line on a wider white stroke, so neighboring significant
    areas no longer merge into one outline); non-significant areas FADED (white veil), so in vs out is unambiguous
    (Priya 10-28: "outline with halo black line significant areas"; "hard to tell what's in vs outside")."""
    from matplotlib import patheffects
    sig = set(sig_areas)
    veil = np.zeros(img.shape, bool)
    for i, n in enumerate(nm):
        if n in shown_areas and n not in sig:
            veil |= img == i
    if veil.any():
        ax.imshow(np.where(veil, 1.0, np.nan), cmap="Greys_r", vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
    for i, n in enumerate(nm):
        if n in sig and (img == i).any():
            ax.contour((img == i).astype(float), levels=[0.5], colors="k", linewidths=1.3,
                       path_effects=[patheffects.withStroke(linewidth=3.6, foreground="white")])


def changes(T: pd.DataFrame, val: str) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    out = []
    for area, t in T.groupby("area"):
        ref = {a: t[(t.animal == a) & (t.epoch == "pre")][val].dropna().to_numpy() for a in AN}
        for ep in il.EP4[1:]:
            post = {a: t[(t.animal == a) & (t.epoch == ep)][val].dropna().to_numpy() for a in AN}
            com = [a for a in AN if len(ref[a]) and len(post[a])]
            if len(com) < 3:
                continue
            iv = ak.boot_delta({a: post[a] for a in com}, {a: ref[a] for a in com}, np.random.default_rng(0))
            out.append({"area": area, "value": val, "epoch": ep, "delta": iv.point, "lo": iv.lo, "hi": iv.hi,
                        "sig": bool(iv.lo > 0 or iv.hi < 0), "n_animals": len(com)})
    return pd.DataFrame(out)


def main() -> int:
    """Compute every statistic, save the four figdata files, then draw (`draw_all` = the --plot-only path)."""
    from frozen_area_maps import area_image, outline
    labels = il.labels()
    D = {x: il.subset_bouts(il._load(x), x) for x in labels}
    T, peths = session_table(D)
    T.to_csv(WOUT / "area_auc_sessions.csv", index=False)
    C = pd.concat([changes(T, "resp"), changes(T, "build")])
    C.to_csv(WOUT / "area_auc_changes.csv", index=False)
    print(C[C.sig].sort_values(["value", "epoch", "delta"]).round(3).to_string(index=False))
    img, nm = area_image(labels[0])
    # ---- maps: per row the colour limit, per panel the area -> change dict + the significant areas
    rows = []
    vals = (("resp", "response AUC 0..1.0 s"), ("build", "build-up AUC -0.4..-0.05 s"))
    for val, ttl in vals:
        c = C[C.value == val]
        lim = np.nanpercentile(np.abs(c.delta), 98) or 1.0
        pans = []
        for ep in il.EP4[1:]:
            ce_ = c[c.epoch == ep].set_index("area")
            pans.append((ep, ce_.delta.to_dict(), list(ce_.index[ce_.sig]), int(ce_.sig.sum())))
        rows.append((ttl, lim, pans))
    save_fd("lick_iti_area_maps", {"img": img, "nm": nm, "ol": outline(img), "rows": rows, "levels": _levels(T, vals)})
    # ---- PETHs, non-orofacial areas
    pp = {}
    for j, base in enumerate(PETH_AREAS):
        for i, h in enumerate(("right", "left")):
            area = f"{base}_{h}"
            cur = []
            for ep in il.EP4:
                labs = [x for x in T[(T.area == area) & (T.epoch == ep)].label.unique() if (x, area) in peths]
                if not labs:
                    continue
                M = np.stack([peths[(x, area)] for x in labs])
                cur.append((ep, M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))))
            pp[(i, j)] = cur
    save_fd("lick_iti_area_peth", {"lags": il.LAGS_P, "panels": pp, "areas": list(PETH_AREAS)})
    # ---- group summary: orofacial vs visual + RSP
    T["group"] = np.where(T.area.str.rsplit("_", n=1).str[0].isin(ORO), "orofacial",
                          np.where(T.area.str.startswith(VISR), "visual + RSP", "other"))
    G = T[T.group != "other"].groupby(["label", "animal", "epoch", "group"])[["resp", "build"]].mean().reset_index()
    den = G[G.epoch == "pre"].groupby(["group", "animal"]).resp.mean()
    dg = np.array([den.get((g_, a_), np.nan) for g_, a_ in zip(G.group, G.animal)])
    G["resp_n"], G["build_n"] = G.resp / dg, G.build / dg
    rng = np.random.default_rng(4)
    tests, gp = [], {}
    for i, (val, _ttl) in enumerate(GROUP_ROWS):
        for j, g in enumerate(GROUP_COLS):
            if j < 2:
                sub = G[G.group == g]
            else:                                         # is the change orofacial-specific? session-paired difference
                Wd = G.pivot_table(index=["label", "animal", "epoch"], columns="group", values=val).dropna()
                sub = (Wd["orofacial"] - Wd["visual + RSP"]).rename(val).reset_index()
            S = il.dots_compute(sub, val, rng)
            gp[(i, j)] = S
            for r_ in S["tests"]:
                tests.append({"group": g, **r_})
    pd.DataFrame(tests).to_csv(WOUT / "area_group_tests.csv", index=False)
    print(pd.DataFrame(tests).round(3).to_string(index=False))
    save_fd("lick_iti_area_groups", {"panels": gp})
    compare(T, peths, draw=False)
    draw_all()
    return 0


def draw_all(only: tuple = ("maps", "peth", "groups", "compare")) -> None:
    """Draw the figures from figdata/<stem>.pkl alone (no session load, no bootstrap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import paint

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    if "maps" in only:
        draw_levels("lick_iti_area_maps")
        F = load_fd("lick_iti_area_maps")
        img, nm, ol = F["img"], F["nm"], F["ol"]
        fig, axs = plt.subplots(2, 3, figsize=(14, 9))
        for i, (ttl, lim, pans) in enumerate(F["rows"]):
            for j, (ep, delta, sig_areas, n_sig) in enumerate(pans):
                ax = axs[i, j]
                M = paint(img, nm, delta)
                im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                _sig_halo(ax, img, nm, sig_areas, list(delta))
                ax.set_title(f"{ttl}\n{ep} - pre ({n_sig} areas significant)", fontsize=9)
                ax.axis("off")
            fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01).set_label("change in dF/F AUC (x1e3 s)")
        fig.suptitle("NEURAL, NO DECODER: self-initiated (ITI) licks across the whole cortex -- change from pre per Allen area in raw dF/F "
                     "AUC (x1e3 s); black outline on a white halo = nested-bootstrap CI excludes 0. Image left = "
                     "ipsilesional.", fontsize=10, wrap=True)
        save_png_svg(fig, WOUT / "figures" / "lick_iti_area_maps.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    if "peth" in only:
        F = load_fd("lick_iti_area_peth")
        lags, areas = F["lags"], F["areas"]
        fig, axs = plt.subplots(2, len(areas), figsize=(2.3 * len(areas), 5.5), sharex=True)
        for j, base in enumerate(areas):
            for i, h in enumerate(("right", "left")):
                ax = axs[i, j]
                for ep, mu, se in F["panels"][(i, j)]:
                    ax.plot(lags, mu, color=colors[ep], lw=1.6, label=ep)
                    ax.fill_between(lags, mu - se, mu + se, color=colors[ep], alpha=0.2, lw=0)
                ax.axvline(0, color="0.6", lw=0.6)
                ax.axhline(0, color="0.8", lw=0.6)
                ax.set_xlim(-1.5, 0.75)
                ax.set_title(f"{base} {'contra' if h == 'right' else 'ipsi'}", fontsize=8)
                ax.tick_params(labelsize=7)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        axs[0, 0].legend(fontsize=6, frameon=False)
        axs[0, 0].set_ylabel("dF/F x1e3", fontsize=8)
        axs[1, 0].set_ylabel("dF/F x1e3", fontsize=8)
        fig.suptitle("NEURAL, NO DECODER: dF/F around self-initiated licks in NON-orofacial areas (per-session mean PETH, all animals "
                     "pooled; band = SEM over sessions; lick onset at 0).", fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save_png_svg(fig, WOUT / "figures" / "lick_iti_area_peth.png", dpi=110)
        plt.close(fig)
    if "groups" in only:
        F = load_fd("lick_iti_area_groups")
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        for i, (_val, ttl) in enumerate(GROUP_ROWS):
            for j, g in enumerate(GROUP_COLS):
                ax = axs[i, j]
                il.dots_draw(ax, F["panels"][(i, j)], colors)
                ax.set_title(f"{g}: {ttl}", fontsize=9)
        fig.suptitle("NEURAL, NO DECODER: self-initiated licks (dF/F), orofacial vs visual + retrosplenial areas (area means per session, divided by "
                     "the animal's pre group-mean response AUC; one dot per session; * = nested-bootstrap change from pre; right = "
                     "orofacial minus visual + RSP in the same session).", fontsize=9.5, wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save_png_svg(fig, WOUT / "figures" / "lick_iti_area_groups.png", dpi=110)
        plt.close(fig)
    if "compare" in only:
        compare_draw(load_fd("lick_iti_area_compare"))


#: orofacial ROIs next to control areas (Priya 10-26: "control area figures for activity traces, e.g. VIS to compare to
#: ALM"); contralesional (_right) hemisphere
COMPARE = ["ALM_right", "tjM1_right", "SSp-m_right", "VISp_right", "VISam_right", "RSPd_right", "SSp-bfd_right", "ACAd_right"]


def compare(T: pd.DataFrame, peths: dict, draw: bool = True) -> None:
    """Orofacial vs control areas side by side: per-session mean PETH (all animals pooled, band = SEM over sessions);
    top raw with ONE shared y-axis, bottom each session divided by its animal's pre-stroke peak (0..0.5 s) of that area.
    Saves figdata/lick_iti_area_compare.pkl (curves + per-bin tests), then draws it unless draw=False."""
    post = (il.LAGS_P >= 0.0) & (il.LAGS_P <= 0.5)
    pk = {}
    for (x, ar), v in peths.items():
        a_ = T[T.label == x].animal.iloc[0]
        if T[T.label == x].epoch.iloc[0] == "pre":
            pk.setdefault((a_, ar), []).append(v)
    pk = {k: float(np.mean(v, 0)[post].max()) for k, v in pk.items()}
    shown = (il.LAGS_P >= -1.5) & (il.LAGS_P <= 0.75)
    sig_recs, panels = [], {}
    for j, ar in enumerate(COMPARE):
        long = {False: [], True: []}
        curves = {0: [], 1: []}
        for ep in il.EP4:
            labs = sorted(x for x in T[(T.area == ar) & (T.epoch == ep)].label.unique() if (x, ar) in peths)
            if not labs:
                continue
            for i, norm in enumerate((False, True)):
                M = np.stack([peths[(x, ar)] / (pk.get((T[T.label == x].animal.iloc[0], ar), np.nan) if norm else 1.0)
                              for x in labs])
                mu, se = np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt(len(M))
                curves[i].append((ep, mu, se, len(M)))
                long[norm] += [pd.DataFrame({"label": x, "animal": T[T.label == x].animal.iloc[0], "epoch": ep,
                                             "lag": il.LAGS_P[shown], "v": m_[shown]}) for x, m_ in zip(labs, M)]
        for i, norm in enumerate((False, True)):
            S = None
            if long[norm]:
                S = il.sig_compute(pd.concat(long[norm], ignore_index=True), value="v", tag={"area": ar, "normalised": norm})
                sig_recs += S["recs"]
            panels[(i, j)] = {"curves": curves[i], "sig": S}
    pd.DataFrame(sig_recs).to_csv(WOUT / "sig_bins_3_31.csv", index=False)
    F = {"lags": il.LAGS_P, "areas": list(COMPARE), "panels": panels}
    save_fd("lick_iti_area_compare", F)
    if draw:
        compare_draw(F)


def compare_draw(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    lags, areas = F["lags"], F["areas"]
    fig, axs = plt.subplots(2, len(areas), figsize=(2.3 * len(areas), 6.2), sharex=True, sharey="row")
    for j, ar in enumerate(areas):
        n_ep = max(len(F["panels"][(0, j)]["curves"]), len(F["panels"][(1, j)]["curves"]))
        for k in range(n_ep):                             # same drawing order as before: per epoch, raw then normalised
            for i in (0, 1):
                cur = F["panels"][(i, j)]["curves"]
                if k < len(cur):
                    ep, mu, se, n = cur[k]
                    axs[i, j].plot(lags, mu, color=colors[ep], lw=1.6, label=f"{ep} ({n})")
                    axs[i, j].fill_between(lags, mu - se, mu + se, color=colors[ep], alpha=0.2, lw=0)
        for i in (0, 1):
            if F["panels"][(i, j)]["sig"] is not None:
                il.sig_draw(axs[i, j], F["panels"][(i, j)]["sig"], colors)
        for i in range(2):
            ax = axs[i, j]
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.8", lw=0.6)
            if i == 1:
                ax.axhline(1, color="0.8", lw=0.6, ls="--")
                ax.set_xlabel("s from lick onset", fontsize=8)
            ax.set_xlim(-1.5, 0.75)
            ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        grp = "OROFACIAL" if j < 3 else "control"
        axs[0, j].set_title(f"{ar.replace('_right', '')} contra\n({grp})", fontsize=9, fontweight="bold" if j < 3 else None)
    axs[0, 0].set_ylabel("dF/F x1e3 (shared axis)", fontsize=8)
    axs[1, 0].set_ylabel("/ animal's pre peak of that area", fontsize=8)
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle("NEURAL, NO DECODER: self-initiated (ITI) licks (dF/F), orofacial ROIs next to control areas (contralesional; first ITI lick "
                 "after >= 2 s without licking; per-session mean PETH, all animals pooled, band = SEM over sessions; "
                 "bottom = each session / its animal's pre-stroke peak of that area).\n" + il.SIG_NOTE, fontsize=9.5,
                 wrap=True)
    il._headroom([axs[0, 0], axs[1, 0]])                 # y shared per row: raise each row once
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    save_png_svg(fig, WOUT / "figures" / "lick_iti_area_compare.png", dpi=110)
    plt.close(fig)


def trial_set_sentence(split: bool = False, cued: bool = False, spout: str | None = None) -> str:
    """The lick set of this run's figures (Priya 11-09). ITI: bout starts (gap >= GAP_MAIN) with the ITI SUBSET filter
    (`il.subset_bouts`), >= 10 per session (>= 5 for one previous spout, `split_maps`); cued (--split --cued): the trial
    first licks of the session files (engaged trials), one target spout when --spout is given."""
    min_n = 10 if spout is None else 5
    if split and cued:
        return il.trial_set_first_licks(min_n=min_n, extra=[f"target spout {spout} only"] if spout else [])
    return il.trial_set_bouts(il.GAP_MAIN, min_n=min_n, filt=il.ITI_SUBSET,
                              extra=[f"previous trial at {spout} only"] if split and spout else [])


def cli() -> int:
    """The command line (sys.argv): every figure of the run states its lick set (`trial_set_sentence`, declared before
    any draw, so the --plot-only replot carries it too)."""
    from wfield_local import trial_set_text as _tst
    if "--split" in sys.argv:                            # ALM / tjM1 split maps; --spout <pos> for one previous spout
        sp_ = sys.argv[sys.argv.index("--spout") + 1] if "--spout" in sys.argv else None
        cu_ = "--cued" in sys.argv                       # first cued lick instead of ITI bout starts
        _tst.declare(text=trial_set_sentence(True, cu_, sp_))
        if plot_only():
            draw_maps(f"lick_{'cued' if cu_ else 'iti'}_area_maps_split" + ("" if sp_ is None else "_" + sp_.replace("_", "")))
            draw_levels(f"lick_{'cued' if cu_ else 'iti'}_area_maps_split" + ("" if sp_ is None else "_" + sp_.replace("_", "")))
        else:
            split_maps(sp_, cued=cu_)
        return 0
    _tst.declare(text=trial_set_sentence())              # the four ITI figures (and --compare / --plot-only)
    if plot_only():                                      # redraw from figdata only
        draw_all(("compare",) if "--compare" in sys.argv else ("maps", "peth", "groups", "compare"))
        return 0
    if "--compare" in sys.argv:                          # only the comparison figure
        labels_ = il.labels()
        compare(*session_table({x: il.subset_bouts(il._load(x), x) for x in labels_}))
        return 0
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())

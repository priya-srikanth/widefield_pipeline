"""SPECIFICITY CONTROL for the chronic residual and Layer 2 (Priya 10-25, analysis 1 of the synthesis: "is the chronic
increase specific or global?"). Total variance rose cortex-wide after the stroke (VISp x1.83, cortex x1.54; variance_maps),
so a cortex-wide change in signal scale (expression, haemodynamics, arousal) could produce a positive residual and a
changed mapping everywhere. Three tests, all within animal, nested bootstrap animals -> sessions (per-animal table first):

  1. RESIDUAL BY AREA (data - frozen prediction, evoked: lick 0..1 s vs -0.3..-0.1 s around the first lick; cue 0.5..1.5 s
     vs -0.5..0 s; from the frozen run's saved PETHs, mean over the six spouts), in each animal's PRE-SD units per area
     (z = (residual - pre mean) / pre SD). OROFACIAL areas vs CONTROL areas (visual, retrosplenial, trunk / lower-limb
     somatosensory) and their contrast.
  2. LAYER 2 BY AREA (`allsession_encoder` combined fit, change_ref, epoch minus pre_late = beyond pre-stroke drift),
     orofacial vs control and the contrast.

Usage: python scripts/specificity_control.py [--tag gate-attempted_v7g3_t3d_frA] [--layer2 allsession_encoder_v7g3_t3d_frA]
Output: <E>/specificity_control/<tag>/{sessions.csv, boot.csv, layer2_boot.csv, figures/*.png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

WINDOWS = {"cue": ((0.5, 1.5), (-0.5, 0.0)), "lick": ((0.0, 1.0), (-0.3, -0.1))}     # cue before lick (Priya 10-25)
SIDES = ("_left", "_right")
OROFACIAL = [a + s for a in ("SSp-m", "SSp-n", "MOp", "MOs") for s in SIDES]
CONTROL = [a + s for a in ("VISp", "VISam", "VISa", "VISrl", "VISal", "VISl", "VISpm", "RSPagl", "RSPd", "RSPv",
                           "SSp-ll", "SSp-tr") for s in SIDES]
GROUPS = {"orofacial": OROFACIAL, "orofacial contra": [a for a in OROFACIAL if a.endswith("_right")],
          "orofacial ipsi": [a for a in OROFACIAL if a.endswith("_left")], "control": CONTROL}
EP4 = ["pre", "acute", "subacute", "chronic"]


def session_rows(run: Path, variant: str = "joint") -> pd.DataFrame:
    rows = []
    for js in sorted((run / variant / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if m["label"][-4:] < "0801":                     # August-only pre (June excluded from encoder analyses)
            continue
        z = np.load(js.with_suffix(".npz"))
        off, parts = z["peth_off_s"], m["peth_parts"]
        for w, ((r0, r1), (b0, b1)) in WINDOWS.items():
            P = z["peth_lick" if w == "lick" else "peth"]
            rs, bs = (off >= r0) & (off <= r1), (off >= b0) & (off <= b1)
            ev = np.nanmean(np.nanmean(P[..., rs], -1) - np.nanmean(P[..., bs], -1), 1)   # (parts, areas), mean of spouts
            for ai, ar in enumerate(m["areas"]):
                d, f = ev[parts.index("data"), ai] * 1e3, ev[parts.index("full"), ai] * 1e3
                rows.append({"animal": m["animal"], "label": m["label"], "epoch": m["epoch"], "window": w, "area": ar,
                             "data": d, "full": f, "residual": d - f})
    return pd.DataFrame(rows)


def add_measures(T: pd.DataFrame) -> pd.DataFrame:
    """Residual in pre-SD z units per animal x window x area (a global-gain correction was tried and dropped, Priya
    10-25: it assumes one multiplicative scale shared by all areas and was unstable; the orofacial - control contrast is
    the specificity test)."""
    T = T.copy()
    for col in ("residual",):
        pre = T[T.epoch == "pre"].groupby(["animal", "window", "area"])[col].agg(["mean", "std"])
        j = T.join(pre, on=["animal", "window", "area"])
        T[f"{col}_z"] = (j[col] - j["mean"]) / j["std"]
    return T


def group_means(T: pd.DataFrame, cols) -> pd.DataFrame:
    """Per session x window: mean over each group's areas present in that animal's every session (fixed area set)."""
    rows = []
    for (an, w), g in T.groupby(["animal", "window"]):
        nlab = g.label.nunique()
        have = set(g.groupby("area").label.nunique().loc[lambda s, n=nlab: s == n].index)
        for gname, areas in GROUPS.items():
            use = [a for a in areas if a in have]
            if not use:
                continue
            s = g[g.area.isin(use)].groupby(["label", "epoch"])[list(cols)].mean().reset_index()
            rows.append(s.assign(animal=an, window=w, area=gname, n_areas=len(use)))
    G = pd.concat(rows, ignore_index=True)
    o = G[G.area == "orofacial"].set_index(["label", "window"])
    c = G[G.area == "control"].set_index(["label", "window"])
    con = (o[list(cols)] - c[list(cols)]).reset_index().merge(o[["epoch", "animal"]].reset_index(), on=["label", "window"])
    return pd.concat([G, con.assign(area="orofacial - control")], ignore_index=True)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    from frozen_area_maps import area_image, outline, paint
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA")
    ap.add_argument("--layer2", default="allsession_encoder_v7g3_t3d_frA")
    ap.add_argument("--reuse", action="store_true", help="reuse sessions.csv (figures / tables only)")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    out = E / "specificity_control" / a.tag
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if a.reuse and (out / "sessions.csv").exists():
        T = pd.read_csv(out / "sessions.csv")
    else:
        T = add_measures(session_rows(run))
        T.to_csv(out / "sessions.csv", index=False)
    ffc.ALAB.update({g: g for g in (*GROUPS, "orofacial - control")})
    cols = ("residual_z", "residual")
    G = group_means(T, cols)
    pd.set_option("display.width", 230)
    per = G[G.area.isin(["orofacial", "control", "orofacial - control"])] \
        .groupby(["window", "area", "animal", "epoch"])[["residual_z"]].mean().unstack("epoch")
    print("== per animal (rule 8): residual in pre-SD units, group means\n"
          + per.reindex(columns=EP4, level=1).round(2).to_string())
    B = pd.concat([ffc.boot(G, c).assign(q=c) for c in ("residual_z",)], ignore_index=True)
    B.to_csv(out / "boot.csv", index=False)
    show = B[B.area.isin(["orofacial contra", "orofacial ipsi", "control", "orofacial - control"]) & (B.epoch != "pre_split")]
    print("\n== group changes vs pre (nested bootstrap)\n"
          + show[["q", "window", "area", "epoch", "point", "lo", "hi", "n_up", "n"]].round(3).to_string(index=False))
    # per-area chronic change (for the maps)
    A = pd.concat([ffc.boot(T, c, epochs=["acute", "chronic"]).assign(q=c) for c in ("residual_z",)],
                  ignore_index=True)
    A.to_csv(out / "area_boot.csv", index=False)

    # LAYER 2 by area
    L = pd.read_csv(E / a.layer2 / "allsession_changes_by_session.csv")
    L = L[L.fit == "combined"].copy()
    L["change_ref"] = L["change_ref"] / 1e3                 # stored x1e3 -> R2 fraction
    LG = []
    for (an,), g in L.groupby(["animal"]):
        nlab = g.label.nunique()
        have = set(g.groupby("area").label.nunique().loc[lambda s, n=nlab: s == n].index)
        for gname, areas in GROUPS.items():
            use = [x for x in areas if x in have]
            if use:
                LG.append(g[g.area.isin(use)].groupby(["label", "epoch"]).change_ref.mean().reset_index()
                          .assign(animal=an, area=gname))
    LG = pd.concat(LG, ignore_index=True)
    o = LG[LG.area == "orofacial"].set_index("label")
    c = LG[LG.area == "control"].set_index("label")
    LG = pd.concat([LG, (o[["change_ref"]] - c[["change_ref"]]).join(o[["epoch", "animal"]]).reset_index()
                    .assign(area="orofacial - control")], ignore_index=True)
    LB = ffc.boot(LG, "change_ref", ref_epoch="pre_late", keys=("area",))
    LB.to_csv(out / "layer2_boot.csv", index=False)
    perL = LG[LG.area.isin(["orofacial", "control", "orofacial - control"])].groupby(["area", "animal", "epoch"]) \
        .change_ref.mean().unstack("epoch")
    print("\n== per animal: Layer 2 change_ref (R2), group means\n"
          + perL.reindex(columns=["pre_late", "acute", "subacute", "chronic"]).round(4).to_string())
    print("\n== Layer 2 epoch - pre_late (nested bootstrap)\n"
          + LB[LB.epoch != "pre_late_split"][["area", "epoch", "point", "lo", "hi", "n_up", "n"]].round(4).to_string(index=False))
    LA = ffc.boot(L, "change_ref", ref_epoch="pre_late", epochs=["chronic"], keys=("area",))

    # FIGURE 1: group bars
    gl = ["orofacial contra", "orofacial ipsi", "control", "orofacial - control"]
    ffc.COL.update({"residual_z": "#d95f02"})
    fig, axs = plt.subplots(3, 1, figsize=(13, 11))
    for ax, w in zip(axs[:2], WINDOWS):
        Bw = B[(B.window == w) & B.q.isin(["residual_z"])].rename(columns={"q": "frame"})
        ffc.bars(ax, Bw, ["residual_z"], gl, "change from pre (pre-SD units)",
                 f"RESIDUAL (data - frozen prediction), {w} window, in each animal's pre-SD units")
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    LBp = LB.copy()
    LBp["frame"] = "layer2"
    LBp["epoch"] = LBp.epoch.replace({"pre_late_split": "pre_split"})
    ffc.COL["layer2"] = "#1b9e77"
    ffc.bars(axs[2], LBp, ["layer2"], gl, "change_ref minus pre_late (R2)",
             "LAYER 2: mapping change beyond pre-stroke drift, by area group")
    fig.suptitle(f"NEURAL: is the chronic change SPECIFIC to orofacial cortex? Frozen frame ({a.tag}); orofacial = SSp-m, "
                 "SSp-n, MOp, MOs; control = visual, retrosplenial, SSp trunk / lower limb.\nBars = mean of animals; "
                 "whisker 95 % nested CI; * excludes 0; dots = animals; grey band = pre split-half range. contra = right.",
                 fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.91, bottom=0.05, hspace=0.45)
    fig.savefig(out / "figures" / "specificity_groups.png", dpi=105)
    plt.close(fig)

    # FIGURE 2: area maps (chronic - pre)
    img, nm = area_image(T.label.iloc[0])
    ol = outline(img)
    panels = []
    for w in WINDOWS:
        for q, lab in (("residual_z", "residual"),):
            s = A[(A.q == q) & (A.window == w) & (A.epoch == "chronic")].set_index("area")
            panels.append((f"{lab}, {w}", s.point.to_dict(), (s.lo > 0) | (s.hi < 0), "pre-SD units"))
    s = LA[LA.epoch == "chronic"].set_index("area")
    panels.append(("Layer 2 change beyond drift", s.point.to_dict(), (s.lo > 0) | (s.hi < 0), "R2"))
    fig, axs = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 4.2))
    for ax, (title, vals, sig, unit) in zip(axs, panels):
        M = paint(img, nm, vals)
        lim = np.nanpercentile(np.abs(M), 98) or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6, interpolation="nearest")
        sg = paint(img, nm, {k: 1.0 for k, v in sig.items() if v})
        if np.isfinite(sg).any():
            ax.contour(np.nan_to_num(sg), levels=[0.5], colors="k", linewidths=1.2)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, orientation="horizontal")
        cb.set_label(unit, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle("NEURAL: CHRONIC minus pre by Allen area (frozen frame). Black outline = area whose 95 % nested CI "
                 "excludes 0 (uncorrected, 41 areas). Image left = ipsilesional.", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.08, wspace=0.05)
    fig.savefig(out / "figures" / "specificity_area_maps.png", dpi=110)
    plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

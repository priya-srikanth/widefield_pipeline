"""ALLEN-ROI version of the pooled residual pixel test (page Fig 10b; Priya 10-25: "should we do an allen ROI version of
10b"). Same quantity and frame as 10b -- frozen pre-stroke encoder residual (data - prediction, evoked), all six spouts
pooled, epoch minus the leave-one-session-out pre sessions, within animal -- but per ALLEN AREA (the area-target frozen run)
instead of per 32 px bin. Per area x window x epoch: nested bootstrap animals -> sessions (`analysis_kit.boot_delta` via
`frozen_fr_compare.boot`), Bonferroni over the areas tested (as 10b corrects over bins), and the number of animals whose
own change has the same sign. Two units: raw x1e3 dF/F and each animal's pre-SD per area (so areas with different signal
size compare).

Reads the per-session table written by `scripts/specificity_control.py` (`specificity_control/<tag>/sessions.csv`, residual
evoked: lick 0..1 s vs -0.3..-0.1 s around the first lick; cue 0.5..1.5 s vs -0.5..0 s), so the two analyses share one
definition.

Usage: python scripts/residual_roi.py [--tag gate-attempted_v7g3_t3d]   (10b's frame; also run --tag ..._frA)
Output: <E>/specificity_control/<tag>/residual_roi.csv, figures/residual_roi_{raw,z}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

EPS = ["acute", "subacute", "chronic"]
WINDOWS = ["cue", "lick"]
SPOUTS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
RESP = {"cue": ((0.5, 1.5), (-0.5, 0.0)), "lick": ((0.0, 1.0), (-0.3, -0.1))}


def spout_table(run: Path) -> pd.DataFrame:
    """Per session x window x spout x area residual (x1e3) + pre-SD z per animal x window x spout x area (Fig 10 / 10c-g
    counterparts). Same evoked windows as specificity_control.session_rows, spouts kept separate."""
    import json
    rows = []
    for js in sorted((run / "joint" / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if m["label"][-4:] < "0801":
            continue
        z = np.load(js.with_suffix(".npz"))
        off, parts = z["peth_off_s"], m["peth_parts"]
        for w, ((r0, r1), (b0, b1)) in RESP.items():
            Pk = z["peth_lick" if w == "lick" else "peth"]
            rs, bs = (off >= r0) & (off <= r1), (off >= b0) & (off <= b1)
            ev = np.nanmean(Pk[..., rs], -1) - np.nanmean(Pk[..., bs], -1)          # (parts, spouts, areas)
            res = (ev[parts.index("data")] - ev[parts.index("full")]) * 1e3
            for si, sp in enumerate(m["pos_order"]):
                for ai, ar in enumerate(m["areas"]):
                    if ar != "cortex":
                        rows.append((m["label"], m["animal"], m["epoch"], w, sp, ar, float(res[si, ai])))
    T = pd.DataFrame(rows, columns=["label", "animal", "epoch", "window", "spout", "area", "residual"])
    pre = T[T.epoch == "pre"].groupby(["animal", "window", "spout", "area"]).residual.agg(["mean", "std"])
    j = T.join(pre, on=["animal", "window", "spout", "area"])
    T["residual_z"] = (j.residual - j["mean"]) / j["std"]
    return T


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    from frozen_area_maps import area_image, outline, paint
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d")
    ap.add_argument("--by-spout", action="store_true", help="per-spout version (Fig 10, 10c-10g counterparts)")
    a = ap.parse_args(argv)
    if a.by_spout:
        return by_spout(a.tag)
    out = E / "specificity_control" / a.tag
    T = pd.read_csv(out / "sessions.csv")
    T = T[T.area != "cortex"]
    B = pd.concat([ffc.boot(T, c, epochs=EPS).assign(q=c) for c in ("residual", "residual_z")], ignore_index=True)
    B = B[B.epoch.isin(EPS)].copy()
    n_areas = B.groupby(["q", "window", "epoch"]).area.transform("nunique")
    B["p_bonf"] = np.minimum(B.p * n_areas, 1.0)
    an_cols = [c for c in ("PS92", "PS93", "PS94", "PS95") if c in B]
    B["n_same"] = [int(np.nansum(np.sign(r[an_cols].to_numpy(float)) == np.sign(r.point))) for _, r in B.iterrows()]
    B.to_csv(out / "residual_roi.csv", index=False)
    pd.set_option("display.width", 230)
    sig = B[(B.q == "residual") & (B.p_bonf < 0.05)]
    print("== areas with a Bonferroni-significant residual change (raw x1e3), per window x epoch")
    for (w, ep), g in sig.groupby(["window", "epoch"]):
        print(f"  {w:4s} {ep:8s} {len(g):2d} / {int(B[(B.q == 'residual') & (B.window == w) & (B.epoch == ep)].area.nunique())}: "
              + ", ".join(f"{r.area} {r.point:+.1f} ({r.n_same}/{r.n})" for _, r in g.sort_values("point", ascending=False).iterrows()))
    img, nm = area_image(T.label.iloc[0])
    ol = outline(img)
    for q, unit in (("residual", "x1e3 dF/F"), ("residual_z", "pre-SD units")):
        fig, axs = plt.subplots(len(WINDOWS), len(EPS), figsize=(3.6 * len(EPS), 3.5 * len(WINDOWS) + 0.8), squeeze=False)
        for i, w in enumerate(WINDOWS):
            Bw = B[(B.q == q) & (B.window == w)]
            lim = np.nanpercentile(np.abs(Bw.point), 98) or 1.0
            for j, ep in enumerate(EPS):
                ax = axs[i, j]
                s = Bw[Bw.epoch == ep].set_index("area")
                im = ax.imshow(paint(img, nm, s.point.to_dict()), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6, interpolation="nearest")
                sg = paint(img, nm, {k: 1.0 for k, v in (s.p_bonf < 0.05).items() if v})
                if np.isfinite(sg).any():
                    ax.contour(np.nan_to_num(sg), levels=[0.5], colors="k", linewidths=1.4)
                up = int(((s.p_bonf < 0.05) & (s.point > 0)).sum())
                dn = int(((s.p_bonf < 0.05) & (s.point < 0)).sum())
                ax.set_title(f"{w} window, {ep} minus pre\n{up + dn} / {len(s)} areas ({up} up, {dn} down)", fontsize=8.5)
                ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=list(axs[i]), fraction=0.025, pad=0.02)
            cb.set_label(f"residual change ({unit})", fontsize=7)
        fig.suptitle(f"NEURAL: frozen-encoder RESIDUAL by ALLEN AREA, all 6 spouts pooled, epoch minus pre, mean of 4 animals "
                     f"({a.tag}).\nBlack outline = area significant after Bonferroni over areas (nested bootstrap animals > "
                     "sessions). ROI counterpart of Fig 10b. Image left = ipsilesional.", fontsize=9)
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.03, top=0.85, wspace=0.05, hspace=0.3)
        fig.savefig(out / "figures" / f"residual_roi_{'raw' if q == 'residual' else 'z'}.png", dpi=115)
        plt.close(fig)
    print("->", out)
    return 0


def by_spout(tag: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    from frozen_area_maps import area_image, outline, paint
    out = E / "specificity_control" / tag
    cache = out / "sessions_by_spout.csv"
    T = pd.read_csv(cache) if cache.exists() else spout_table(E / f"frozen_encoder_{tag}")
    T.to_csv(cache, index=False)
    B = pd.concat([ffc.boot(T, c, epochs=EPS, keys=("window", "spout", "area")).assign(q=c)
                   for c in ("residual", "residual_z")], ignore_index=True)
    B = B[B.epoch.isin(EPS)].copy()
    B["p_bonf"] = np.minimum(B.p * B.groupby(["q", "window", "spout", "epoch"]).area.transform("nunique"), 1.0)
    an_cols = [c for c in ("PS92", "PS93", "PS94", "PS95") if c in B]
    B["n_same"] = [int(np.nansum(np.sign(r[an_cols].to_numpy(float)) == np.sign(r.point))) for _, r in B.iterrows()]
    B.to_csv(out / "residual_roi_by_spout.csv", index=False)
    R = B[B.q == "residual"]
    cnt = R.assign(up=(R.p_bonf < .05) & (R.point > 0), down=(R.p_bonf < .05) & (R.point < 0)) \
        .groupby(["window", "spout", "epoch"])[["up", "down"]].sum().unstack("epoch")
    pd.set_option("display.width", 230)
    print("== Bonferroni-significant areas per spout (raw x1e3; up / down)\n" + cnt.to_string())
    fc = B[(B.q == "residual") & (B.spout == "far_contra") & (B.p_bonf < .05)]
    print("== far contra (Fig 10 counterpart)\n" + fc[["window", "epoch", "area", "point", "n_same", "n"]]
          .sort_values(["window", "epoch", "point"]).round(1).to_string(index=False))
    img, nm = area_image(T.label.iloc[0])
    ol = outline(img)
    for w in WINDOWS:
        for q, unit in (("residual", "x1e3 dF/F"), ("residual_z", "pre-SD units")):
            Bw = B[(B.q == q) & (B.window == w)]
            lim = np.nanpercentile(np.abs(Bw.point), 98) or 1.0
            fig, axs = plt.subplots(len(EPS), len(SPOUTS), figsize=(2.6 * len(SPOUTS), 2.6 * len(EPS) + 0.9), squeeze=False)
            for i, ep in enumerate(EPS):
                for j, sp in enumerate(SPOUTS):
                    ax = axs[i, j]
                    s_ = Bw[(Bw.epoch == ep) & (Bw.spout == sp)].set_index("area")
                    im = ax.imshow(paint(img, nm, s_.point.to_dict()), cmap="RdBu_r", vmin=-lim, vmax=lim,
                                   interpolation="nearest")
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6, interpolation="nearest")
                    sg = paint(img, nm, {k: 1.0 for k, v in (s_.p_bonf < 0.05).items() if v})
                    if np.isfinite(sg).any():
                        ax.contour(np.nan_to_num(sg), levels=[0.5], colors="k", linewidths=1.2)
                    up, dn = int(((s_.p_bonf < .05) & (s_.point > 0)).sum()), int(((s_.p_bonf < .05) & (s_.point < 0)).sum())
                    ax.set_title(f"{sp.replace('_', ' ')}, {ep}\n{up} up / {dn} down", fontsize=7.5)
                    ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01)
            cb.set_label(f"residual change ({unit})", fontsize=7)
            fig.suptitle(f"NEURAL: frozen-encoder RESIDUAL by ALLEN AREA, PER SPOUT, {w} window, epoch minus pre, mean of 4 "
                         f"animals ({tag}).\nBlack = area significant after Bonferroni over areas (nested bootstrap animals > "
                         "sessions); one colour scale. ROI counterpart of Figs 10 / 10c-10g. Image left = ipsilesional.",
                         fontsize=8.5)
            fig.subplots_adjust(left=0.01, right=0.9, bottom=0.02, top=0.86, wspace=0.04, hspace=0.3)
            fig.savefig(out / "figures" / f"residual_roi_by_spout_{w}_{'raw' if q == 'residual' else 'z'}.png", dpi=110)
            plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

ALM / tjM1 / MOs_rest / MOp_rest ROWS (Priya 10-27): the split ROIs (ROI v8 on the stat mask, from the frozen run's basis
twin, `encoder_split_rows`) are tested beside the Allen areas -- same bootstrap, and the SAME Bonferroni count (the Allen
areas only: the split ROIs re-partition MOs / MOp, they are not extra areas, so no existing p_bonf moves). Pooled figure:
one extra map row per window with MOs / MOp drawn as their split ROIs; per spout: companion `*_split.png` figures.

Usage: python scripts/residual_roi.py [--tag gate-attempted_v7g3_t3d]   (10b's frame; also run --tag ..._frA)
Output: <E>/specificity_control/<tag>/residual_roi.csv, figures/residual_roi_{raw,z}.png
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
from wfield_local.frozen_encoder import design_tag as _design_tag  # frame tag of the FROZEN_LICKS_ALL design (8.51)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E_BASE  # noqa: E402

EPS = ["acute", "subacute", "chronic"]
WINDOWS = ["cue", "lick"]
SPOUTS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
RESP = {"cue": ((0.5, 1.5), (-0.5, 0.0)), "lick": ((0.0, 1.0), (-0.3, -0.1))}


def spout_table(run: Path) -> pd.DataFrame:
    """Per session x window x spout x area residual (x1e3) + pre-SD z per animal x window x spout x area (Fig 10 / 10c-g
    counterparts). Same evoked windows as specificity_control.session_rows, spouts kept separate."""
    from wfield_local import config
    import json
    rows = []
    for js in sorted((run / "joint" / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if config.early_pre(m["label"]):
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
    return add_split_z(T, run)


def add_split_z(T: pd.DataFrame, run: Path) -> pd.DataFrame:
    """Append the ALM / tjM1 / rest rows per spout (basis twin of ``run``) when missing, then the pre-SD z of every row
    (per animal x window x spout x area, as before)."""
    import encoder_split_rows as esr
    T = T.drop(columns=["residual_z"], errors="ignore")
    if Path(f"{run}_basis").is_dir() and not T.area.isin(esr.SPLIT_AREAS).any():
        T = pd.concat([T, esr.residual_rows(run, list(T.label.unique()), by_spout=True)], ignore_index=True)
    pre = T[T.epoch == "pre"].groupby(["animal", "window", "spout", "area"]).residual.agg(["mean", "std"])
    j = T.join(pre, on=["animal", "window", "spout", "area"])
    T["residual_z"] = (j.residual - j["mean"]) / j["std"]
    return T


def _only_tested(img: np.ndarray, nm: list, tested) -> np.ndarray:
    """Blank (display only) atlas areas with no value in this analysis -- frontal pole, AUDd / AUDpo right, VISp left (no
    ROI) -- whose outlines otherwise showed as dotted 'edge fragments' along the frontal and lateral edges (checked 10-28:
    the only disconnected pieces of TESTED areas are 10 single pixels)."""
    drop = [k for k, n in enumerate(nm) if n not in set(tested)]
    return np.where(np.isin(img, drop), -1, img)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    from frozen_area_maps import area_image, outline, paint
    from iti_area_maps import _sig_halo
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=_design_tag("gate-attempted_v7g3_t3d"))
    ap.add_argument("--by-spout", action="store_true", help="per-spout version (Fig 10, 10c-10g counterparts)")
    a = ap.parse_args(argv)
    if a.by_spout:
        return by_spout(a.tag)
    from wfield_local import run_mode
    out = run_mode.subdir(E_BASE / "specificity_control") / a.tag      # specificity_control's mode-routed folder (N10)
    T = pd.read_csv(out / "sessions.csv")
    T = T[T.area != "cortex"]
    B = pd.concat([ffc.boot(T, c, epochs=EPS).assign(q=c) for c in ("residual", "residual_z")], ignore_index=True)
    B = B[B.epoch.isin(EPS)].copy()
    import encoder_split_rows as esr
    allen = ~B.area.isin(esr.SPLIT_AREAS)                 # split ROIs re-partition MOs / MOp: not counted (docstring)
    n_areas = B[allen].groupby(["q", "window", "epoch"]).area.nunique()
    B["p_bonf"] = np.minimum(B.p * B.set_index(["q", "window", "epoch"]).index.map(n_areas).to_numpy(), 1.0)
    an_cols = [c for c in ("PS92", "PS93", "PS94", "PS95") if c in B]
    B["n_same"] = [int(np.nansum(np.sign(r[an_cols].to_numpy(float)) == np.sign(r.point))) for _, r in B.iterrows()]
    B.to_csv(out / "residual_roi.csv", index=False)
    pd.set_option("display.width", 230)
    sig = B[(B.q == "residual") & (B.p_bonf < 0.05) & allen]
    print("== areas with a Bonferroni-significant residual change (raw x1e3), per window x epoch")
    for (w, ep), g in sig.groupby(["window", "epoch"]):
        print(f"  {w:4s} {ep:8s} {len(g):2d} / {int(B[(B.q == 'residual') & (B.window == w) & (B.epoch == ep)].area.nunique())}: "
              + ", ".join(f"{r.area} {r.point:+.1f} ({r.n_same}/{r.n})" for _, r in g.sort_values("point", ascending=False).iterrows()))
    sp_ = B[(B.q == "residual") & B.area.isin(["MOs_right", "MOs_left", "MOp_right", "MOp_left", *esr.SPLIT_AREAS])]
    if sp_.area.isin(esr.SPLIT_AREAS).any():
        print("== MOs / MOp and their split ROIs (raw x1e3; point [CI], p_bonf, animals with the same sign)")
        print(sp_[["window", "epoch", "area", "point", "lo", "hi", "p_bonf", "n_same", "n"]].round(3).to_string(index=False))
    img, nm = area_image(T.label.iloc[0])
    split = bool(B.area.isin(esr.SPLIT_AREAS).any())
    img_s, nm_s = esr.split_image(img, nm) if split else (img, nm)
    img, img_s = _only_tested(img, nm, B.area), _only_tested(img_s, nm_s, B.area)
    ol, ol_s = outline(img), outline(img_s)
    # rows: per window the Allen areas, then (split) the same test with MOs / MOp drawn as ALM + MOs_rest / tjM1 +
    # MOp_rest (Priya 10-27), on the same colour scale
    rows_ = [(w, kind) for w in WINDOWS for kind in (("allen", "split") if split else ("allen",))]
    for q, unit in (("residual", "x1e3 dF/F"), ("residual_z", "pre-SD units")):
        fig, axs = plt.subplots(len(rows_), len(EPS), figsize=(3.6 * len(EPS), 3.5 * len(rows_) + 0.8), squeeze=False)
        for i, (w, kind) in enumerate(rows_):
            Bw = B[(B.q == q) & (B.window == w)]
            lim = np.nanpercentile(np.abs(Bw[~Bw.area.isin(esr.SPLIT_AREAS)].point), 98) or 1.0
            im_, nm_, ol_ = (img, nm, ol) if kind == "allen" else (img_s, nm_s, ol_s)
            for j, ep in enumerate(EPS):
                ax = axs[i, j]
                s = Bw[Bw.epoch == ep].set_index("area")
                im = ax.imshow(paint(im_, nm_, s.point.to_dict()), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol_, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6, interpolation="nearest")
                _sig_halo(ax, im_, nm_, s.index[s.p_bonf < 0.05], s.index)   # each sig area outlined, rest faded
                if kind == "allen":
                    s = s[~s.index.isin(esr.SPLIT_AREAS)]
                    up = int(((s.p_bonf < 0.05) & (s.point > 0)).sum())
                    dn = int(((s.p_bonf < 0.05) & (s.point < 0)).sum())
                    ax.set_title(f"{w} window, {ep} minus pre\n{up + dn} / {len(s)} areas ({up} up, {dn} down)", fontsize=8.5)
                else:
                    s = s[s.index.isin(esr.SPLIT_AREAS)]
                    ax.set_title(f"{w}, {ep}: ALM / tjM1 split\n{int((s.p_bonf < 0.05).sum())} / {len(s)} split ROIs "
                                 "significant", fontsize=8.5)
                ax.axis("off")
            cb = fig.colorbar(im, ax=list(axs[i]), fraction=0.025, pad=0.02)
            cb.set_label(f"residual change ({unit})", fontsize=7)
        fig.suptitle(f"NEURAL: frozen-encoder RESIDUAL by ALLEN AREA, all 6 spouts pooled, epoch minus pre, mean of 4 animals "
                     f"({a.tag}).\nBlack outline (white halo) = each area significant after Bonferroni over areas, others faded (nested bootstrap animals > "
                     "sessions). ROI counterpart of Fig 10b. Image left = ipsilesional."
                     + ("\n'split' rows: MOs / MOp drawn as ALM + MOs_rest / tjM1 + MOp_rest (ROI v8), Bonferroni over the "
                        "Allen areas." if split else ""), fontsize=9)
        fig.subplots_adjust(left=0.02, right=0.88, bottom=0.03, top=1 - 1.3 / (3.5 * len(rows_) + 0.8), wspace=0.05,
                            hspace=0.3)
        save_png_svg(fig, out / "figures" / f"residual_roi_{'raw' if q == 'residual' else 'z'}.png", dpi=115)
        plt.close(fig)
    print("->", out)
    return 0


def by_spout(tag: str) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    from frozen_area_maps import area_image, outline, paint
    from iti_area_maps import _sig_halo
    from wfield_local import frozen_encoder, run_mode
    from wfield_local.writeguard import assert_writable
    out = run_mode.subdir(E_BASE / "specificity_control", make=True) / tag       # review N10 (11-03)
    assert_writable(out / "_")
    cache = out / "sessions_by_spout.csv"
    run = frozen_encoder.out_root() / f"frozen_encoder_{tag}"
    T = add_split_z(pd.read_csv(cache), run) if cache.exists() else spout_table(run)     # split rows added if missing
    T.to_csv(cache, index=False)
    B = pd.concat([ffc.boot(T, c, epochs=EPS, keys=("window", "spout", "area")).assign(q=c)
                   for c in ("residual", "residual_z")], ignore_index=True)
    B = B[B.epoch.isin(EPS)].copy()
    import encoder_split_rows as esr
    allen = ~B.area.isin(esr.SPLIT_AREAS)                 # Bonferroni over the Allen areas only (module docstring)
    n_areas = B[allen].groupby(["q", "window", "spout", "epoch"]).area.nunique()
    B["p_bonf"] = np.minimum(B.p * B.set_index(["q", "window", "spout", "epoch"]).index.map(n_areas).to_numpy(), 1.0)
    an_cols = [c for c in ("PS92", "PS93", "PS94", "PS95") if c in B]
    B["n_same"] = [int(np.nansum(np.sign(r[an_cols].to_numpy(float)) == np.sign(r.point))) for _, r in B.iterrows()]
    B.to_csv(out / "residual_roi_by_spout.csv", index=False)
    R = B[(B.q == "residual") & allen]
    cnt = R.assign(up=(R.p_bonf < .05) & (R.point > 0), down=(R.p_bonf < .05) & (R.point < 0)) \
        .groupby(["window", "spout", "epoch"])[["up", "down"]].sum().unstack("epoch")
    pd.set_option("display.width", 230)
    print("== Bonferroni-significant areas per spout (raw x1e3; up / down)\n" + cnt.to_string())
    fc = B[(B.q == "residual") & (B.spout == "far_contra") & (B.p_bonf < .05) & allen]
    print("== far contra (Fig 10 counterpart)\n" + fc[["window", "epoch", "area", "point", "n_same", "n"]]
          .sort_values(["window", "epoch", "point"]).round(1).to_string(index=False))
    img, nm = area_image(T.label.iloc[0])
    img_full = img
    img = _only_tested(img_full, nm, B.area)
    ol = outline(img)
    # COMPANION '_split' figures (Priya 10-27): the same test with MOs / MOp drawn as ALM + MOs_rest / tjM1 + MOp_rest, on
    # the Allen figure's colour scale; the Allen figures (scale, counts) are computed on the Allen rows only, as before
    kinds = [("allen", img, nm, ol, "")]
    if (~allen).any():
        img_s, nm_s = esr.split_image(img_full, nm)
        img_s = _only_tested(img_s, nm_s, B.area)
        kinds.append(("split", img_s, nm_s, outline(img_s), "_split"))
    for w in WINDOWS:
        for q, unit in (("residual", "x1e3 dF/F"), ("residual_z", "pre-SD units")):
            Bw = B[(B.q == q) & (B.window == w)]
            lim = np.nanpercentile(np.abs(Bw[~Bw.area.isin(esr.SPLIT_AREAS)].point), 98) or 1.0
            for kind, im_, nm_, ol_, sfx in kinds:
                fig, axs = plt.subplots(len(EPS), len(SPOUTS), figsize=(2.6 * len(SPOUTS), 2.6 * len(EPS) + 0.9),
                                        squeeze=False)
                for i, ep in enumerate(EPS):
                    for j, sp in enumerate(SPOUTS):
                        ax = axs[i, j]
                        s_ = Bw[(Bw.epoch == ep) & (Bw.spout == sp)].set_index("area")
                        im = ax.imshow(paint(im_, nm_, s_.point.to_dict()), cmap="RdBu_r", vmin=-lim, vmax=lim,
                                       interpolation="nearest")
                        ax.imshow(ol_, cmap="Greys", vmin=0, vmax=1.2, alpha=0.6, interpolation="nearest")
                        _sig_halo(ax, im_, nm_, s_.index[s_.p_bonf < 0.05], s_.index)
                        s_ = s_[s_.index.isin(esr.SPLIT_AREAS) == (kind == "split")]
                        up, dn = int(((s_.p_bonf < .05) & (s_.point > 0)).sum()), int(((s_.p_bonf < .05) & (s_.point < 0)).sum())
                        ax.set_title(f"{sp.replace('_', ' ')}, {ep}\n{up} up / {dn} down"
                                     + (" (split ROIs)" if kind == "split" else ""), fontsize=7.5)
                        ax.axis("off")
                cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.25, 0.012, 0.5]))   # own axes: never over a panel
                cb.set_label(f"residual change ({unit})", fontsize=7)
                fig.suptitle(f"NEURAL: frozen-encoder RESIDUAL by ALLEN AREA, PER SPOUT, {w} window, epoch minus pre, mean of 4 "
                             f"animals ({tag}).\nBlack outline = each area significant after Bonferroni over areas, others faded (nested bootstrap animals > "
                             "sessions); one colour scale. ROI counterpart of Figs 10 / 10c-10g. Image left = ipsilesional."
                             + ("\nMOs / MOp drawn as ALM + MOs_rest / tjM1 + MOp_rest (ROI v8); Bonferroni over the Allen "
                                "areas; scale of the Allen figure." if kind == "split" else ""), fontsize=8.5)
                fig.subplots_adjust(left=0.01, right=0.9, bottom=0.02, top=0.86 if kind == "allen" else 0.84, wspace=0.04,
                                    hspace=0.3)  # bar at 0.915
                save_png_svg(fig, out / "figures" / f"residual_roi_by_spout_{w}_{'raw' if q == 'residual' else 'z'}{sfx}.png",
                            dpi=110)
                plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

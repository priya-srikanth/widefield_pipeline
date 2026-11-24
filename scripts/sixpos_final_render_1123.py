"""FINAL candidate panels for Fig 4B, Fig 4C, S11A and S11B (CC-B, 2026-11-23), drawn with the LIVE renderer
(manuscript_code/figures c2_scalar_compact_v26.py -> c2_scalar_aligned_render / c2_matched.endpoint, epoch-staggered
stars in the top band) from the FINAL tables of `scripts/sixpos_final_1123.py`:

  Fig 4B  C2_speed_all_FINAL_ccB1123    all post-cue licks, peak 3-D speed (onset -> peak), per-trial median, mean
  Fig 4C  C2_reach_all_FINAL_ccB1123    all post-cue licks, peak 3-D reach, per-trial median, session mean
  S11A    C2_reach_first_FINAL_ccB1123  first post-cue lick, peak 3-D reach, session median over trials
  S11B    C2_speed_first_FINAL_ccB1123  first post-cue lick, peak 3-D speed, session median over trials

Stars = the FINAL 180-cell family. The renderer is used unchanged; only the input, y label and output stem are set
here. It also writes a contact sheet (live panels left, FINAL right). Existing files are never overwritten.

    python scripts/sixpos_final_render_1123.py --out <FINAL_sixpos dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper")
MS_FIG = PAPER / "manuscript_code/figures"
LIVE_FIG = PAPER / "manuscript_live/figures"
DATA = PAPER / "data_4cam_20261122"
WIDE = DATA / "kinematics/c2_lick3d_session_summary_FINAL_ccB1123.csv"
FAMILY = DATA / "scalar_family/c2_scalar_family_FINAL_ccB1123_boot4000_cam3on.csv"
SUFFIX = "_FINAL_ccB1123"
#: metric -> (stem, n column, y label, live file shown beside it on the contact sheet, panel)
PANELS = {
    "speed_all": ("C2_speed_all", "n_speed_all", "peak speed,\nall licks (Δ mm/s)", "C2_speed_all_final", "Fig 4B"),
    "reach_all": ("C2_reach_all", "n_reach_all", "3-D reach,\nall licks (Δ mm)", "C2_reach_first_final", "Fig 4C"),
    "reach_first": ("C2_reach_first", "n_reach_first", "first-lick 3-D\nreach (Δ mm)", "C2_reach_all_final", "S11A"),
    "speed_first": ("C2_speed_first", "n_speed_first", "first-lick\npeak speed (Δ mm/s)", None, "S11B"),
}


def render(out: Path):
    sys.path.insert(0, str(MS_FIG))
    import c2_scalar_compact_v26 as V      # patches vls_style.save / star (v26 top-band stars); noqa
    import c2_matched as M
    import c2_scalar_aligned_render as R
    import vls_style as S
    S.apply() if hasattr(S, "apply") else None
    W = pd.read_csv(WIDE)
    F = pd.read_csv(FAMILY)
    V.SUFFIX = SUFFIX
    for metric, (stem, ncol, ylab, _live, _p) in PANELS.items():
        if any((out / f"{stem}{SUFFIX}.{x}").exists() for x in ("pdf", "svg", "png")):
            print("exists, skipped (never overwritten):", out / f"{stem}{SUFFIX}")
            continue
        sx = W.copy()
        sx["pos"] = sx.position.map(M.POS3)
        fam = F[F.metric == metric].copy()
        dd = R.lick3d_changes(sx, metric, ncol)
        R.check_n(dd, fam, stem)
        t = fam.assign(agree=np.where(fam.star_aligned == 1, "both", ""))
        M.endpoint(dd, t, ylab, out / stem, legend=False, level=2)
        print(stem, "stars:", int(fam.star_aligned.sum()))


def _mediabox(pdf: Path) -> str:
    import re
    m = re.search(rb"MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", pdf.read_bytes())
    return f"{float(m.group(3)):.2f} x {float(m.group(4)):.2f} pt" if m else "?"


def page_sizes(out: Path):
    for (stem, _n, _y, live, _p) in PANELS.values():
        s = f"{stem}{SUFFIX}: {_mediabox(out / f'{stem}{SUFFIX}.pdf')}"
        if live and (LIVE_FIG / f"{live}.pdf").exists():
            s += f"; live {live}: {_mediabox(LIVE_FIG / f'{live}.pdf')}"
        print(s)


def contact_sheet(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    png = out / f"contact_sheet{SUFFIX}.png"
    if png.exists():
        print("exists, skipped:", png)
        return
    F = pd.read_csv(FAMILY)

    def raster(pdf: Path):
        return plt.imread(str(pdf.with_suffix(".png")))      # the renderer's PNG twin of each PDF

    fig, axes = plt.subplots(len(PANELS), 2, figsize=(11, 2.6 * len(PANELS)))
    for i, (metric, (stem, _n, _y, live, panel)) in enumerate(PANELS.items()):
        for j in (0, 1):
            axes[i, j].axis("off")
        if live and (LIVE_FIG / f"{live}.png").exists():
            axes[i, 0].imshow(raster(LIVE_FIG / f"{live}.pdf"))
            axes[i, 0].set_title(f"LIVE in this slot: {live}.pdf", fontsize=8)
        else:
            axes[i, 0].text(0.5, 0.5, "no live panel (new S11B)", ha="center", va="center", fontsize=9, color="0.4")
        n = int(F[F.metric == metric].star_aligned.sum())
        axes[i, 1].imshow(raster(out / f"{stem}{SUFFIX}.pdf"))
        axes[i, 1].set_title(f"FINAL {panel}: {stem}{SUFFIX}.pdf ({n} stars)", fontsize=8)
    fig.suptitle("Six-position FINAL candidates (CC-B 11-23): all licks = per-trial median then session mean; "
                 "first lick = session median; stars = FINAL 180-cell family", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png, dpi=200)
    fig.savefig(png.with_suffix(".pdf"))
    print("wrote", png)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    render(a.out)
    page_sizes(a.out)
    contact_sheet(a.out)


if __name__ == "__main__":
    main()

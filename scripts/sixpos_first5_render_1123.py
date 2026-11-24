"""CANDIDATE Fig 4B / 4C panels for "post-cue licks 1-5" (Priya 11-23), in the live manuscript panel style: the SAME
renderer as the live `C2_speed_all_final.pdf` / `C2_reach_first_final.pdf` (manuscript_code/figures
c2_scalar_compact_v26.py -> c2_scalar_aligned_render / c2_matched.endpoint, epoch-staggered stars in the top band),
fed with the licks 1-5 session means of `scripts/sixpos_first5_1123.py` and its candidate family stars.

  4B  C2_speed_licks15_onsetpeak_candidate_ccB1123  speed: mean of speed_peak (peak 3-D tip speed, onset .. extension
      peak) over licks lick_idx < 5 per session x spout (ungated post-cue set, >= 20 licks); stars = family
      `c2_scalar_family_mean_first5_boot4000_cam3on.csv` metric speed_all
  4C  C2_reach_licks15_candidate_ccB1123  reach: mean of reach_mm over licks 1-5 (working-period set, >= 20 licks);
      stars = the same family, metric reach_all (the live 4C is FIRST-lick reach; this candidate replaces it)

The live renderer is used unchanged (imported from manuscript_code/figures); only the y label and the output stem are
set here. Before drawing the candidates it re-renders the live C2_speed_all from the production inputs into a scratch
folder (--check-dir) so the style can be compared with the live PDF.

    python scripts/sixpos_first5_render_1123.py --dir <sixpos_first5 out dir> --check-dir <scratch>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MS_FIG = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/manuscript_code/figures")
SUFFIX = "_candidate_ccB1123"
PROD_FAMILY = Path("M:/MICROSCOPE/Priya/Widefield/manuscript_stats/c2_4cam_sync60_20261122/"
                   "scalar_family_final_boot4000_cam3on/c2_scalar_family_aligned.csv")
PROD_LICK3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60/"
                   "kinematics_fused_postcue/c2_lick3d_session_summary_ungated_cam3on.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--check-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(MS_FIG))
    import c2_scalar_compact_v26 as V      # patches vls_style.save / star (v26 top-band stars); noqa
    import c2_scalar_aligned_render as R
    import c2_matched as M
    import vls_style as S
    S.apply() if hasattr(S, "apply") else None

    def draw(sx, fam_rows, col, ncol, ylab, stem, suffix, out):
        for x in ("pdf", "svg", "png"):
            if (out / f"{stem}{suffix}.{x}").exists():
                sys.exit(f"refusing to overwrite {out / f'{stem}{suffix}.{x}'}")
        V.SUFFIX = suffix
        dd = R.lick3d_changes(sx, col, ncol)
        R.check_n(dd, fam_rows, stem)
        t = fam_rows.assign(agree=np.where(fam_rows.star_aligned == 1, "both", ""))
        M.endpoint(dd, t, ylab, out / stem, legend=False, level=2)
        print(stem, "stars:", int(fam_rows.star_aligned.sum()))

    # style check: the live 4B from production inputs
    a.check_dir.mkdir(parents=True, exist_ok=True)
    F0 = pd.read_csv(PROD_FAMILY)
    sx0 = pd.read_csv(PROD_LICK3D)
    sx0["pos"] = sx0.position.map(M.POS3)
    draw(sx0, F0[F0.metric == "speed_all"].copy(), "speed_all", "n_all", "peak speed,\nall licks (Δ mm/s)",
         "C2_speed_all", "_stylecheck", a.check_dir)

    S_ = pd.read_csv(a.dir / "sixpos_first5_session_summary.csv")
    F = pd.read_csv(a.dir / "c2_scalar_family_mean_first5_boot4000_cam3on.csv")
    for metric, stem, ylab in (("speed_all", "C2_speed_licks15_onsetpeak", "peak speed,\nlicks 1-5 (Δ mm/s)"),
                               ("reach_all", "C2_reach_licks15", "3D reach,\nlicks 1-5 (Δ mm)")):
        sx = S_[S_.endpoint == metric].copy()
        sx["pos"] = sx.position.map(M.POS3)
        draw(sx, F[F.metric == metric].copy(), "mean_first5", "n_first5", ylab, stem, SUFFIX, a.dir)


if __name__ == "__main__":
    main()

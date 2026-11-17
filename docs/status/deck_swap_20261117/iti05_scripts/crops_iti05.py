"""Deck images for the ITI 0..0.5 s swap (11-17), cut like the current deck images; new names only (never overwrite)."""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SW = Path("M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs")
TAG = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
L = Path(f"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/{TAG}/lmm_hidden_20261117/iti05_20261117")
D = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
         f"iti_licks/lateral_ratio_dock_iti05_20261117/{TAG}/figures")


def put(src: Path, name: str) -> None:
    dst = SW / name
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    shutil.copyfile(src, dst)
    print(dst, Image.open(dst).size)


def depurple(a, y0=0):
    """White-out the purple provenance footer (blueviolet text + antialiasing), rows >= y0 (as make_crops.py)."""
    s = a[y0:].astype(int)
    r, g, b = s[..., 0], s[..., 1], s[..., 2]
    m = ((b - g > 40) & (r - g > 40) & (b > r + 15)) | ((b > 200) & (b - g > 12) & (b > r + 8) & (r > g + 4))
    s[m] = 255
    a[y0:] = s.astype(np.uint8)
    return a


put(L / "iti_ramp_row1_lmm_w0-0.5_iti05_notitle.png", "s108_iti_ramp_row1_lmm_w0-0.5_iti05.png")
put(L / "lick_lmm_sessz_stack_iti05.png", "s111_lick_lmm_sessz_stack_iti05.png")
for s, k in ((136, "cued"), (137, "cue"), (138, "iti")):
    put(L / f"lick_lateral_ratio_auc_docked_iti05_{k}_notitle.png", f"s{s:03d}_lick_lateral_ratio_auc_docked_iti05_{k}_lmm.png")
a = np.asarray(Image.open(D / "lick_lateral_ratio_auc.png").convert("RGB")).copy()
a = depurple(a, a.shape[0] - 90)
dst = SW / "s203_lick_lateral_ratio_auc_docked_iti05.png"
if dst.exists():
    raise SystemExit(f"refusing to overwrite {dst}")
Image.fromarray(a).save(dst)
print(dst, a.shape[1], "x", a.shape[0])

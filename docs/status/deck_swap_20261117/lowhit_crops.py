"""Low-hit deck crops (Priya 11-17). Each function cuts a render the way the current deck image was cut.
python lh_compose.py check   -> rebuild the CURRENT deck images from the OLD renders and report the pixel difference
python lh_compose.py make NEWDIR OUTDIR  -> cut the new renders in NEWDIR into OUTDIR (refuses to overwrite)"""
import os
import sys
import numpy as np
from PIL import Image

S = os.path.dirname(os.path.abspath(__file__))
OLD = ("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
       "attempt_coding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-"
       "incomplete-stopA2jexrt3500c70_daqlk/figures")


def crop_rows(src, y0, h):
    im = Image.open(src).convert("RGB")
    return im.crop((0, y0, im.width, y0 + h))


def _legend_box(im, y0=1199, y1=1232, pad=8):
    a = np.asarray(im.convert("L").crop((0, y0, im.width, y1)))
    cols = np.flatnonzero((a < 250).any(0))
    return (int(cols[0]) - pad, y0, int(cols[-1]) + pad + 1, y1)


def compose_143(src, legend_box=None):
    """2 x 2 by animal: each quadrant = hit-rate panel over cortex panel (render rows 2 + 3), legend below."""
    im = Image.open(src).convert("RGB")
    legend_box = legend_box or _legend_box(im)
    TY0, TH = 520, 652
    W = 523 + 520
    lg = im.crop(legend_box)
    lgs = [lg]
    if lg.width > W:                       # wider legend (extra entry): split at the widest gap near the middle
        a = np.asarray(lg.convert("L"))
        blank = (a >= 250).all(0)
        mid = lg.width // 2
        cand = [x for x in range(lg.width) if blank[x] and blank[max(0, x - 12):x + 12].all()]
        cut = min(cand, key=lambda x: abs(x - mid))
        lgs = [lg.crop((0, 0, cut, lg.height)), lg.crop((cut, 0, lg.width, lg.height))]
    H = 2 * TH + 1 + sum(t.height for t in lgs)
    c = Image.new("RGB", (W, H), "white")
    L = 576                                                                  # seam (measured on the old deck image)
    c.paste(im.crop((0, TY0, L, TY0 + TH)), (0, 0))                         # PS92
    c.paste(im.crop((L + 15, TY0, W + 15, TY0 + TH)), (L, 0))               # PS93
    c.paste(im.crop((0, TY0, 135, TY0 + TH)), (0, TH))                      # PS94: y label + ticks of column 1 ...
    c.paste(im.crop((966 + 135, TY0, 966 + L, TY0 + TH)), (135, TH))        # ... + the PS94 panels
    c.paste(im.crop((L + 981, TY0, W + 981, TY0 + TH)), (L, TH))            # PS95
    y = 2 * TH + 1
    for t in lgs:
        c.paste(t, ((W - t.width) // 2, y))
        y += t.height
    return c


JOBS = {  # slide: (render name, cutter, current deck image, new swap name)
    141: ("decoder_retention_pooled.png", lambda p: crop_rows(p, 128, 472), "cur_s141_fig_decoder_retention_pooled.png",
          "s141_decoder_retention_pooled_lowhit.png"),
    143: ("recovery_timing_lick.png", compose_143, "cur_s143_fig_recovery_timing_lick.png", "s143_recovery_timing_2x2_lowhit.png"),
    144: ("recovery_timing_pooled_epoch_lick.png", lambda p: crop_rows(p, 140, 412),
          "cur_s144_fig_recovery_timing_pooled_epoch.png", "s144_recovery_timing_epoch_body_lowhit.png"),
    146: ("recovery_timing_pooled_day_lick_aim.png", lambda p: crop_rows(p, 161, 943),
          "cur_s146_fig_recovery_timing_pooled_day_aim.png", "s146_recovery_timing_day_aim_body_lowhit.png"),
}

if __name__ == "__main__":
    if sys.argv[1] == "check":
        for k, (rn, f, cur, _n) in JOBS.items():
            a = np.asarray(f(os.path.join(OLD, rn)), float)
            b = np.asarray(Image.open(os.path.join(S, cur)).convert("RGB"), float)
            same = a.shape == b.shape
            print(k, a.shape, b.shape, "mad", float(np.abs(a - b).mean()) if same else "shape differs")
    else:
        new, out = sys.argv[2], sys.argv[3]
        for k, (rn, f, _cur, name) in JOBS.items():
            dst = os.path.join(out, name)
            if os.path.exists(dst):
                raise SystemExit(f"exists, not overwriting: {dst}")
            im = f(os.path.join(new, rn))
            im.save(dst)
            print(k, im.size, dst)

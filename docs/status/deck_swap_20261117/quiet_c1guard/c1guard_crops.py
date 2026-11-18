"""Deck images for the quiet-event slides from the cam1-guard production run (Priya 11-18 OK). Each cutter is the one
that made the image now on the slide (11-17 scratch crops.py for 155/156/158/203; build 11-02 prep_figs.hemo and
figure_rules_crops.q_stillness_rate for 157). Renders are the same size as before, so the pixel geometry carries over.
python c1guard_crops.py check   -> re-cut the OLD renders, compare with the images on the slides now (pixel diff)
python c1guard_crops.py make    -> cut the NEW renders into swap_inputs/ (refuses to overwrite)"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

B = Path("M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02")
sys.path.insert(0, str(B))
import figprep as FP  # noqa: E402

T = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
Q = Path(f"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/{T}/cohort1_quiet_spikes")
NEW = Q / "v2_quiet_lp_c1guard_k375_20261117"
OLD_1117 = Q / "v2_quiet_lp_20261117"
OLD_PRE = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort1_quiet_spikes/v2_quiet_lp")   # pre-rebuild (slide 157)
SWAP = Path("M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs")
CROPS = B / "build_inputs" / "crops"


def trim(im, pad=8, thr=250):
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a < thr)
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(im.width, xs.max() + pad + 1),
                    min(im.height, ys.max() + pad + 1)))


def examples(p):
    im = Image.open(p)
    return trim(im.crop((0, 190, im.width, 795)))


def fr_2x2(p):
    im = Image.open(p).convert("RGB")
    y0, y1, ax0 = 118, im.height, 85
    p1, p2 = im.crop((0, y0, 440, y1)), im.crop((440, y0, 805, y1))
    p3, p4 = im.crop((805, y0, 1170, y1)), im.crop((1170, y0, im.width, y1))
    p3l = Image.new("RGB", (ax0 + p3.width, p3.height), "white")
    p3l.paste(im.crop((0, y0, ax0, y1)), (0, 0))
    p3l.paste(p3, (ax0, 0))
    W, H = max(p1.width + p2.width, p3l.width + p4.width), p1.height
    c = Image.new("RGB", (W, 2 * H + 10), "white")
    c.paste(p1, (0, 0)); c.paste(p2, (p1.width, 0)); c.paste(p3l, (0, H + 10)); c.paste(p4, (p3l.width, H + 10))  # noqa: E702
    return trim(c)


def eta(p):
    im = Image.open(p)
    f = im.width / 1745.0
    return trim(im.crop((int(285 * f), int(85 * f), int(1550 * f), int(768 * f))))


def t13(p):
    im = Image.open(p)
    f = im.width / 1755.0
    return trim(im.crop((0, int(725 * f), int(592 * f), int(1195 * f))))


def stillness(p):          # figure_rules_crops.q_stillness_rate
    a = np.asarray(Image.open(p).convert("RGB"))
    m = FP.nonwhite(a).any(1)
    ink = FP._runs(m)
    body = a[max(int(ink[1][0]) - 6, int(ink[0][1])):]
    c = FP.cuts(FP.nonwhite(body).any(0), 2)
    return Image.fromarray(FP.trim(body[:, :c[0][1]]))


def hemo(p):               # prep_figs.hemo
    a = FP.load(p)
    w = int(0.73 * a.shape[1])
    return Image.fromarray(FP.trim(FP.vstack([a[150:475, :w], a[785:1088, :w]])))


# slide: (render relative path, cutter, old render root, current deck image, new image name)
JOBS = {
    155: ("quiet_event_example_traces.png", examples, OLD_1117, SWAP / "s154_quiet_examples_PS92.png",
          "s155_quiet_examples_PS92_c1guard.png"),
    156: ("fr_regressed_own/quiet_events_face_regressed.png", fr_2x2, OLD_1117, SWAP / "s155_quiet_fr_2x2.png",
          "s156_quiet_fr_2x2_c1guard.png"),
    158: ("task_events/t13_precue_event_vs_outcome.png", t13, OLD_1117, SWAP / "s157_t13_decoder.png",
          "s158_t13_decoder_c1guard.png"),
    203: ("eta_fr/quiet_eta_maps_fr.png", eta, OLD_1117, SWAP / "s202_quiet_eta_maps_orofacial.png",
          "s203_quiet_eta_maps_orofacial_c1guard.png"),
    "157s": ("stillness/quiet_stillness_nolines.png", stillness, OLD_PRE, CROPS / "q_stillness_rate_nolines.png",
             "s157_q_stillness_rate_nolines_c1guard.png"),
    "157h": ("hemo415/quiet_hemo_check.png", hemo, OLD_PRE, CROPS / "q_hemo415_rows.png",
             "s157_q_hemo415_rows_c1guard.png"),
}


def main(mode):
    for k, (rel, cut, old_root, cur, name) in JOBS.items():
        if mode == "check":
            a = np.asarray(cut(old_root / rel).convert("RGB")).astype(int)
            b = np.asarray(Image.open(cur).convert("RGB")).astype(int)
            same = a.shape == b.shape
            d = np.abs(a - b).max() if same else None
            print(k, "shape", a.shape, b.shape, "max |diff|", d)
        else:
            out = SWAP / name
            if out.exists():
                raise SystemExit(f"exists, not overwriting: {out}")
            cut(NEW / rel).convert("RGB").save(out)
            print(k, "->", out, Image.open(out).size)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "check")

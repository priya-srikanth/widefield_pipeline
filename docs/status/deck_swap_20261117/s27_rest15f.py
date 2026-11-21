"""Add slide 27 (shown): the frozen pre-stroke decoder on lick-free rest (imaging page Fig 12.1, rest15f, floor 30 +
central-90 % matching), right after the position-map slides 24-26 (Priya 11-18 via 3c: slide 27 OK, floor 30 + central
90 % = primary). Duplicates slide 26's shapes (title, rule, bullets, picture, caption), swaps in the cropped figure,
rewrites the texts and notes.
usage: python s27_rest15f.py <deck_in> <deck_out> [crop_png]   (crop_png is made if missing; never overwritten)"""
import copy
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation

sys.path.insert(0, str(Path(__file__).parent))
from apply_specs import set_para_text, swap_picture, write_notes  # noqa: E402

G = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/hv-meegkit_hpfit_lag_refitT_swapmask_"
         "lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/epoch")
SRC = G / "rest15f_restnlc1all_floor30_pct5_20261118/allpre/rest15f_restnlc1v3wk_allpre_durgap_floor30_pct5.png"
CROP = Path("M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/"
            "s27_rest15f_floor30_pct5_crop.png")


def make_crop(src, out):
    """Drop the suptitle (the methods go to the notes): cut at the first all-white band below the title text, trim."""
    if out.exists():
        return out
    im = Image.open(src).convert("RGB")
    a = np.asarray(im.convert("L")) < 250
    rows = a.any(1)
    y, run = int(np.argmax(rows)), 0             # from the first title row, find the first white band >= 40 px
    while run < 40:                              # (the gaps between title lines are a few px)
        run = run + 1 if not rows[y] else 0
        y += 1
    while not rows[y]:
        y += 1
    body = a[y:]
    ys, xs = np.where(body)
    pad = 10
    im.crop((max(0, xs.min() - pad), max(0, y + ys.min() - pad), min(im.width, xs.max() + pad + 1),
             min(im.height, y + ys.max() + pad + 1))).save(out)
    return out


def main(src, dst, img=None):
    img = Path(img) if img else make_crop(SRC, CROP)
    Image.open(img).verify()
    P = Presentation(src)
    base = P.slides[25]
    assert base.shapes[0].text_frame.text.startswith("Pre-cue maps"), base.shapes[0].text_frame.text
    nxt = P.slides[26].shapes[0].text_frame.text
    assert nxt.startswith("2 · Licking kinematics"), nxt
    new = P.slides.add_slide(base.slide_layout)
    for sh in list(new.shapes):
        sh._element.getparent().remove(sh._element)
    for sh in base.shapes:
        new.shapes._spTree.append(copy.deepcopy(sh._element))
    pic = [sh for sh in new.shapes if sh.shape_type == 13][0]
    pic.name = "fig:rest15f_floor30_pct5"
    swap_picture(new, pic, str(img), box=[365762, 2150000, 11460171, 4200000])   # below the 3 bullets
    T = {sh.name: sh for sh in new.shapes if sh.has_text_frame}
    set_para_text(T["TextBox 1"].text_frame.paragraphs[0],
                  "Lick-free rest decoder: drops acutely, stays below pre")
    b = T["TextBox 3"].text_frame.paragraphs
    set_para_text(b[0], "•  Decoder fit on pre-stroke rest (spout docked, every DAQ / video lick removed), scored "
                        "unchanged after the stroke")
    set_para_text(b[1], "•  Change vs own pre: acute −0.14, subacute −0.07, chronic −0.10 (all CIs exclude 0); acute "
                        "below pre in 4 / 4 animals")
    set_para_text(b[2], "•  Provisional: the subacute call depends on the matching window and on cam1 lick removal")
    for p in b[3:]:
        p._p.getparent().remove(p._p)
    set_para_text(T["TextBox 5"].text_frame.paragraphs[0],
                  "A: accuracy per session (chance 1/6); B: retained per animal; C: change in above-chance fraction "
                  "vs own pre, animal-weighted, 95% CI (* excludes 0)")
    block = {
        "Window": "lick-free rest periods between trials, 4 bins per period (no cue / lick alignment)",
        "Licks": "rest minus every DAQ lick, every cam4 LP lick and every guarded cam1-only LP event (non-droplet, "
                 "med_out_px >= -120 px), each widened 0.5 s before / 2.0 s after",
        "Trial period": "rest periods bracketed by two camera working trials (contact or incomplete) at the same spout, "
                        "inside the camera working period (first to last camera lick)",
        "Trial types": "rest (restdock05: spout docked, not running); periods matched in duration and gap to the "
                       "nearest lick to the central 90 % of each animal's pre periods",
        "Gate": "camera trial classes for the brackets; a post-stroke session is scored with >= 30 matched periods "
                "(pre sessions, training pools and models >= 40)",
        "Baseline": "each animal's own pre, June + August (35 pre sessions)",
        "Correction": "none across epochs; analysis_kit.boot_delta nested animals -> sessions, 4,000 draws; * = 95 % CI "
                      "excludes 0",
    }
    body = [
        "Imaging page Fig 12.1 [#f-d_rest15f] (v62). Decoder: frozen pre-stroke 6-way logistic regression (standardized, "
        "C = 0.1) on joint-LocaNMF (projected) rest-period features; one model per animal fit on its pre-stroke periods, "
        "stored and reloaded unchanged (align rest-restnlc1v3wk-b4-dur5-gap5); each pre session scored by its own "
        "leave-one-session-out model. Measured: balanced accuracy (chance 1/6); above-chance fraction = (accuracy - 1/6) "
        "/ (5/6); retained = that fraction relative to the animal's own pre. Null = 200 block-label permutations per "
        "session (gray dashes).",
        "Sessions: pre 35 (PS92 9 / PS93 6 / PS94 9 / PS95 11), acute 16 (5 / 4 / 6 / 1), subacute 16 (1 / 3 / 5 / 7), "
        "chronic 43 (12 / 12 / 8 / 11); 4 animals. PS93 acute = 4 sessions. PS92 subacute rests on one session that "
        "meets the minimum (PS92_0824; PS92_0826 has 27 matched periods). PS95's acute epoch is one recorded session "
        "(PS95_0817).",
        "Change in above-chance fraction vs own pre: acute -0.139 [-0.200, -0.070]*, subacute -0.069 [-0.128, -0.008]*, "
        "chronic -0.098 [-0.170, -0.024]*. Retained per animal (acute / subacute / chronic): PS92 .68 / 1.15 / 1.02; "
        "PS93 -.16 / .32 / .31; PS94 .35 / .46 / .13; PS95 .16 / .72 / .61. Status: provisional.",
        "Robustness: with central-80 % matching the subacute drop does not exclude zero (-0.030 [-0.118, +0.092]); acute "
        "and chronic hold. Requiring 40 matched periods per post-stroke session instead of 30 changes no outcome. "
        "cam4-only lick removal (no cam1): the acute drop holds (-0.135*), subacute (-0.049) and chronic (-0.077 "
        "[-0.150, +0.003]) do not exclude zero. August-only pre (3 animals: PS92, PS94, PS95): acute -0.091*, subacute "
        "-0.033 n.s., chronic -0.090*.",
        "Caveat: PS92's lick-free pre-stroke rest accuracy is the lowest of the four (.264 vs null .169), so its retained "
        "values (including subacute 1.15, from one session) are unstable; PS92 stays in the statistics.",
        "Licks: rest lick-free = DAQ + cam4 LP + guarded cam1 LP licks removed; brackets = camera trial classes.",
        "FIGURE SOURCE: Imaging page Fig 12.1 (https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV, v62) · original: "
        f"{SRC.as_posix()} (tables data/rest15f_restnlc1v3wk_allpre_durgap_floor30_pct5_{{per_animal,cohort,sessions}}"
        ".csv) · supplements: .../rest15f_restnlc1all_floor30_pct5_20261118/augpre/ and "
        ".../rest15f_restnl_cam4_floor30_pct5_20261118/allpre/ · script: scripts/rest_migration/rest15f_restnl_1117.py "
        "--lp c1all --pre all --score-min-post 30 --match-pct 5 (main 1a4f73f1) · deck image: suptitle removed by "
        "docs/status/deck_swap_20261117/s27_rest15f.py -> " + img.as_posix(),
    ]
    if new.notes_slide.notes_text_frame is None:
        new.notes_slide.shapes._spTree.append(copy.deepcopy(base.notes_slide.notes_placeholder._element))
    write_notes(new, block, body)
    lst = P.slides._sldIdLst
    el = lst[-1]
    lst.remove(el)
    lst.insert(26, el)                            # after slide 26 -> becomes slide 27 (shown)
    P.save(dst)
    print("saved", dst, "slides", len(P.slides))


if __name__ == "__main__":
    main(*sys.argv[1:4])

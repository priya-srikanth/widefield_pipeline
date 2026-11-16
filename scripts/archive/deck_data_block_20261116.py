"""Deck notes "Data:" pass (Priya 11-01 request, go-ahead 11-16): a fixed 7-line block at the TOP of a slide's notes,
labels exactly as the 11-07 FR slides: Window / Licks / Trial period / Trial types / Gate / Baseline / Correction.
Sources: (a) CONFIRMED rows of FIGURE_PROVENANCE_20261106.csv (deck rows matched by the slide's picture name, since
the table's slide numbers are from the 11-02 deck); (b) FR slides (a8's arm), from the FR page captions / slide notes.
INFERRED / UNKNOWN rows are NOT written. Slides whose notes already start with the block are skipped."""
import copy
import csv
import os
import shutil
import sys

import pandas as pd
from pptx import Presentation

F = "M:/MICROSCOPE/Priya/Widefield/presentations/Overview_2026-11-06.pptx"
PROV = "C:/Users/SabatiniLab/Github/wf_cleanup2/docs/status/rebuild_20261103/FIGURE_PROVENANCE_20261106.csv"
LOG = sys.argv[1]
LABELS = ("Window", "Licks", "Trial period", "Trial types", "Gate", "Baseline", "Correction")

ENG = "engagement gate"
AUG = "August pre"
FR = {  # slide title prefix -> 7 values (FR arm; from page captions / existing notes)
    "FaceRhythm master face, registered": (
        "whole session (mesh tracking; no trial window)", "none", "whole session", "n/a (registration QC)", "none",
        "n/a (master = medoid of the August pre templates)", "n/a (QC, no test)"),
    "FaceRhythm rank 12": (
        "trial windows cue -2 to +4 s (25 Hz) + pre-cue and between-trial rows (5 Hz)",
        "none (factors named afterwards against LP licks)", "engaged trials + between-trial rows",
        "20 engaged trials per spout per animal, all outcomes", ENG, "August pre sessions only (the fit set)",
        "none (model choice: EVR, stability, split-half)"),
    "The 12 frozen FaceRhythm factors": (
        "master fit as rank-12 slide; expression panels cue-aligned -2 to +4 s",
        "none (names validated against LP licks / running)", "engaged trials", "all engaged trials, by spout", ENG,
        "August pre (pre-stroke expression only)", "none (descriptive)"),
    "Frozen factors still fit post-stroke": (
        "whole session: trial windows cue -2 to +4 s + between-trial rows", "none",
        "engaged trials + between-trial rows", "all engaged trials", ENG,
        "August pre; pre sessions scored on the leave-their-day-out fit",
        "none; paired animals -> sessions bootstrap 95 % CI (boot_delta)"),
    "Lick right drops; lick left substitutes": (
        "cue-aligned 0 to +4 s (peak of the trial average)", "none for the peak (per-lick text: LP licks)",
        "engaged trials", "all engaged trials, contra (close_R, far_R) vs ipsi (close_L, far_L)", ENG,
        "August pre; log2 vs each animal's own pre",
        "none (planned contrast); paired animals -> sessions bootstrap 95 % CI, * excludes 0"),
    "Mixed-model stars: FaceRhythm lick-factor peak by side": (
        "cue-aligned 0 to +4 s (peak of the trial average)", "none", "engaged trials",
        "all engaged trials, contra vs ipsi spouts", ENG, "August pre; log2 vs each animal's own pre",
        "mixed model (epoch_change_lmm), * p < .05, ** Holm across the 12 cells"),
    "Per contra lick: face moves less": (
        "lick-aligned: expression at each LP lick (0 to +0.2 s)", "LP cam4 video licks, split by direction",
        "engaged trials", "all engaged trials; licks by direction", ENG, "August pre; log2 vs each animal's own pre",
        "** Holm over the 12 factors; paired animals -> sessions bootstrap 95 % CI"),
    "Mixed-model stars: FaceRhythm per-lick factors": (
        "lick-aligned: expression at each LP lick (0 to +0.2 s)", "LP cam4 video licks, split by direction",
        "engaged trials", "all engaged trials; licks by direction", ENG, "August pre; log2 vs each animal's own pre",
        "mixed model (epoch_change_lmm), * p < .05, ** Holm within panel (36 per-lick cells; licks per trial 6)"),
    "Per trial: contra lick movements": (
        "cue 0 to +4 s, single trials", "LP cam4 video licks (v36), direction by |angle| > 12 deg", "engaged trials",
        "all trial types (hit and miss), per spout; amplitude on trials with / without a lick of the factor's direction",
        ENG, "August pre; log2 (amplitude) or fraction (prevalence) vs each animal's own pre",
        "** Holm over the 6 spouts; boot_delta 95 % CI; tested only with a pre baseline in >= 3 animals"),
    "Face asymmetry falls at contra spouts": (
        "cue 0 to +2 s; lick -0.1 to +0.2 s around each LP lick; bout onset 0 to +0.5 s", "LP licks",
        "engaged trials", "all engaged trials, per spout", ENG, "August pre; minus each animal's pre mean",
        "** Holm over the 6 positions per alignment x epoch; paired animals -> sessions bootstrap 95 % CI"),
    "Mixed-model stars: FaceRhythm aligned asymmetry": (
        "cue 0 to +2 s; lick -0.1 to +0.2 s around each LP lick; bout onset 0 to +0.5 s", "LP licks",
        "engaged trials", "all engaged trials, per spout", ENG, "August pre; minus each animal's pre mean",
        "mixed model (epoch_change_lmm), * p < .05, ** Holm within panel (18)"),
    "PS94: lick factors match the master": (
        "per-session own fits: trial windows cue -2 to +4 s + between-trial rows", "none", "engaged trials",
        "60 engaged trials per session (10 per spout) + between-trial rows", ENG,
        "the frozen master factor (August pre) as reference", "none (descriptive match quality)"),
    "lick_right loses its 7 Hz peak": (
        "per-session own fits: trial windows cue -2 to +4 s + between-trial rows", "none", "engaged trials",
        "60 engaged trials per session (10 per spout) + between-trial rows", ENG,
        "August pre; change in units of each animal's own pre-session SD",
        "FDR .05 over 36 frequencies (paired bootstrap)"),
}


def has_block(tf):
    head = [p.text for p in tf.paragraphs[:8]]
    return sum(any(h.startswith(l + ":") for h in head) for l in LABELS) == 7


def insert_block(tf, vals):
    first = tf.paragraphs[0]._p
    for lab, v in zip(LABELS, vals):
        p = copy.deepcopy(first)
        for r in list(p):                                   # keep pPr only; drop runs / breaks / fields
            if r.tag.split("}")[1] in ("r", "br", "fld"):
                p.remove(r)
        first.addprevious(p)
        from pptx.text.text import _Paragraph
        _Paragraph(p, tf).text = f"{lab}: {v}"


B = F.replace(".pptx", ".bak_pre_a8_datablock_20261116.pptx")
if not os.path.exists(B):
    shutil.copy2(F, B)
d = pd.read_csv(PROV)
conf = d[(d.output == "PI deck 11-02") & (d.status == "CONFIRMED")]
P = Presentation(F)
log = []
for i, s in enumerate(P.slides, 1):
    if not s.has_notes_slide or s.notes_slide.notes_text_frame is None:
        continue
    tf = s.notes_slide.notes_text_frame
    title = next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame), "")
    if has_block(tf):
        continue
    pics = [sh.name for sh in s.shapes if sh.shape_type == 13]
    vals, src = None, None
    m = conf[conf.fig_id.isin(pics)]
    if len(m):
        r = m.iloc[0]
        vals = tuple(str(r[k]) for k in ("align_window", "lick_source", "trial_period", "trial_types",
                                          "engagement_gate", "pre_baseline", "statistic"))
        src = f"FIGURE_PROVENANCE row {r['row']} (CONFIRMED)"
    else:
        for k, v in FR.items():
            if title.startswith(k):
                vals, src = v, "FR arm (a8), FR page captions / slide notes"
                break
    if vals is None:
        continue
    insert_block(tf, vals)
    log.append({"slide": i, "title": title[:80], "source": src})
# slide 77 wording fix (peak of the trial average, not per trial)
for s in P.slides:
    t = next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame), "")
    if t.startswith("Lick right drops; lick left substitutes"):
        for p in s.notes_slide.notes_text_frame.paragraphs:
            for r in p.runs:
                if "peak expression per trial" in r.text:
                    r.text = r.text.replace("peak expression per trial", "peak of the trial-averaged expression")
                    log.append({"slide": "77", "title": t[:80], "source": "wording: 'peak expression per trial' -> "
                                "'peak of the trial-averaged expression'"})
P.save(F)
with open(LOG, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["slide", "title", "source"])
    w.writeheader()
    w.writerows(log)
print(len(log), "changes")
for r in log:
    print(r)

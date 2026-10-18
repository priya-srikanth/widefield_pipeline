"""PowerPoint deck of the orofacial pre- vs post-stroke figures (Priya 2026-10-17: "build a PowerPoint deck of the figures,
as we did in stroke_orofacial" -- after `stroke_orofacial_pipeline/scripts/chore_dlc_pooled_phenotype/
build_stats_figure_deck.py`: title, sections, one figure per slide with title + caption, significance grids).

    python -m wfield_local.orofacial_deck [--run lp_r4_heatmap_20261013__clean_v36]      (needs python-pptx: locanmf env)

Reads the figures `orofacial_cohort` and `orofacial_jaw_lateral` wrote and the stats table
(`cohort/tables/prepost/stats/change_vs_pre_stats.csv`); writes `<LP_analysis>/<run>/cohort/deck/
orofacial_prepost_<YYYYMMDD>.pptx`. Run it AFTER the two producers (the deck shows whatever is on disk). Missing
figures are skipped and listed at the end.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

RUN = "lp_r4_heatmap_20261013__clean_v36"
W_IN, H_IN = 13.333, 7.5

#: (section, [(file relative to cohort/figures, slide title, caption)]); the caption says what is plotted and from which
#: licks / frames, the speaker notes carry the method text of the family.
SECTIONS = [
    ("Cohort summaries -- change vs pre-stroke (session-weighted, within-animal)", [
        ("prepost/cohort_metrics_minus_pre.png", "Per-position lick metrics, change vs pre",
         "All post-cue LP licks; each session minus its animal's own pre mean. Marks: * MWU+FDR and bootstrap agree, "
         "(*) MWU only, b bootstrap only. Last panel: jaw steering."),
        ("prepost/cohort_epochs_by_position_accuracy.png", "Cohort accuracy / mismatch by position and epoch, change vs pre",
         "Bold = mean +- SEM over sessions of the within-animal change; faint = animals; marks = cohort tests."),
        ("prepost/cohort_timecourse_by_position_accuracy.png", "Cohort accuracy / mismatch by position, day bins, change vs pre", ""),
        ("prepost/cohort_epochs_by_position_kinematics.png", "Cohort kinematics by position and epoch, change vs pre",
         "Bold = mean +- SEM over sessions of the within-animal change; faint = animals; marks = cohort tests."),
        ("prepost/cohort_timecourse_by_position_kinematics.png", "Cohort kinematics by position, day bins, change vs pre", ""),
        ("prepost/cohort_epochs_by_position_engagement.png", "Cohort engagement by position and epoch, change vs pre",
         "Bold = mean +- SEM over sessions of the within-animal change; faint = animals; marks = cohort tests."),
        ("prepost/cohort_timecourse_by_position_engagement.png", "Cohort engagement by position, day bins, change vs pre", ""),
        ("prepost/cohort_trajectory_xy_within_inmouth_incl.png", "Cohort tongue-tip paths, pre vs post (within-animal)",
         "Each epoch = cohort pre path + mean within-animal change. Camera view; band = SEM."),
        ("prepost/cohort_phase_portrait_within_inmouth_incl.png", "Cohort lick phase portraits, pre vs post (within-animal)",
         "Protrusion speed vs protrusion; within-animal changes on the cohort pre portrait."),
        ("prepost/cohort_trajectory_xy_within_inmouth_excl.png",
         "Cohort tongue-tip paths, tongue OUT only (within-animal)",
         "Same as above, frames with the tongue in the mouth (< 20 px) left out."),
        ("prepost/cohort_phase_portrait_within_inmouth_excl.png",
         "Cohort phase portraits, tongue OUT only (within-animal)", ""),
        ("prepost/cohort_lick_time_first_minus_pre_inmouth_excl.png", "FIRST lick, lick-peak-aligned time -- tongue OUT only",
         "Rows: protrusion, lateral, angle (change vs pre), speed |v| raw and change."),
        ("prepost/cohort_lick_time_first_raw_inmouth_excl.png", "FIRST lick, lick-peak-aligned time -- tongue OUT only, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_lick_time_first_minus_pre_inmouth_incl.png", "FIRST lick, lick-peak-aligned time -- in-mouth INCLUDED",
         "Rows: protrusion, lateral, angle (change vs pre), speed |v| raw and change."),
        ("prepost/cohort_lick_time_first_raw_inmouth_incl.png", "FIRST lick, lick-peak-aligned time -- in-mouth INCLUDED, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_lick_time_all_minus_pre_inmouth_excl.png", "ALL post-cue licks, lick-peak-aligned time -- tongue OUT only",
         "Rows: protrusion, lateral, angle (change vs pre), speed |v| raw and change."),
        ("prepost/cohort_lick_time_all_raw_inmouth_excl.png", "ALL post-cue licks, lick-peak-aligned time -- tongue OUT only, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_lick_time_all_minus_pre_inmouth_incl.png", "ALL post-cue licks, lick-peak-aligned time -- in-mouth INCLUDED",
         "Rows: protrusion, lateral, angle (change vs pre), speed |v| raw and change."),
        ("prepost/cohort_lick_time_all_raw_inmouth_incl.png", "ALL post-cue licks, lick-peak-aligned time -- in-mouth INCLUDED, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_lick_phase_first_minus_pre_inmouth_excl.png", "FIRST lick, per-lick phase (onset to offset) -- tongue OUT only",
         "0 = tongue leaves the mouth, 1 = back in."),
        ("prepost/cohort_lick_phase_first_raw_inmouth_excl.png", "FIRST lick, per-lick phase (onset to offset) -- tongue OUT only, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_lick_phase_first_minus_pre_inmouth_incl.png", "FIRST lick, per-lick phase (onset to offset) -- in-mouth INCLUDED",
         "0 = tongue leaves the mouth, 1 = back in."),
        ("prepost/cohort_lick_phase_first_raw_inmouth_incl.png", "FIRST lick, per-lick phase (onset to offset) -- in-mouth INCLUDED, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_lick_phase_all_minus_pre_inmouth_excl.png", "ALL post-cue licks, per-lick phase (onset to offset) -- tongue OUT only",
         "0 = tongue leaves the mouth, 1 = back in."),
        ("prepost/cohort_lick_phase_all_raw_inmouth_excl.png", "ALL post-cue licks, per-lick phase (onset to offset) -- tongue OUT only, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_lick_phase_all_minus_pre_inmouth_incl.png", "ALL post-cue licks, per-lick phase (onset to offset) -- in-mouth INCLUDED",
         "0 = tongue leaves the mouth, 1 = back in."),
        ("prepost/cohort_lick_phase_all_raw_inmouth_incl.png", "ALL post-cue licks, per-lick phase (onset to offset) -- in-mouth INCLUDED, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_cue_minus_pre_inmouth_excl.png", "Cue-aligned (Family 8.2) -- tongue OUT only",
         "Every frame of every trial; bars: cluster permutation p < 0.05 vs pre."),
        ("prepost/cohort_cue_raw_inmouth_excl.png", "Cue-aligned (Family 8.2) -- tongue OUT only, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_cue_minus_pre_inmouth_incl.png", "Cue-aligned (Family 8.2) -- in-mouth INCLUDED", ""),
        ("prepost/cohort_cue_raw_inmouth_incl.png", "Cue-aligned (Family 8.2) -- in-mouth INCLUDED, RAW",
         "Raw levels per epoch, pre as its own curve; dashed = spout angle."),
        ("prepost/cohort_trajectory_xy_inmouth_incl.png", "Cohort tongue-tip paths, raw (pooled over animals)",
         "Not within-animal: PS93's pre-existing deficit enters; see the within-animal version."),
        ("prepost/cohort_phase_portrait_inmouth_incl.png", "Cohort lick phase portraits, raw (pooled over animals)", ""),
    ]),
    ("Spatial distributions -- Wasserstein-2 vs own pre-stroke cloud", [
        ("w2/epochs_by_position.png", "W2 distance from the animal's own pre-stroke keypoint cloud, by epoch",
         "All keypoints in the mouth frame. Spout = the drift control (fixed apparatus; licking pushes it 2-5 px); nose = "
         "head / snout posture (behaviour). Pre = leave-one-out noise floor. Spout-moved trials excluded."),
        ("w2/timecourse_by_position.png", "W2 distance from own pre, day bins", ""),
    ]),
    ("Trial-aligned spatial heatmaps (stroke_orofacial proportion / proportion-diff)", [
        ("heatmaps/nose_by_epoch.png", "Nose position by epoch (all positions pooled): raw and change vs own pre",
         "Cue - 500 ms .. trial stop, engaged trials, mouth / spout frame, camera orientation. Delta pre column = later "
         "minus earlier half of pre sessions (floor). Contour = every animal agrees in sign."),
        ("heatmaps/tongue_by_position_raw.png", "Tongue tip occupancy by position, raw", "Cleaned tip, OUT frames."),
        ("heatmaps/tongue_by_position_delta.png", "Tongue tip occupancy by position, change vs own pre", ""),
        ("heatmaps/jaw_by_position_raw.png", "Jaw occupancy by position, raw", "Cleaned jaw, rest-relative."),
        ("heatmaps/jaw_by_position_delta.png", "Jaw occupancy by position, change vs own pre", ""),
        ("heatmaps/nose_by_position_raw.png", "Nose occupancy by position, raw", ""),
        ("heatmaps/nose_by_position_delta.png", "Nose occupancy by position, change vs own pre", ""),
        ("heatmaps/spout_by_position_raw.png", "Spout tip by position, raw (control)", ""),
        ("heatmaps/spout_by_position_delta.png", "Spout tip by position, change vs own pre (control)", ""),
    ]),
    ("Jaw lateral readout (whole sessions)", [
        ("jaw_lateral/timecourse.png", "Jaw steering toward the target, day by day",
         "FAR_DIR = (far ipsi - far contra) / 2 of the jaw's sideways displacement at wide opening; every far trial."),
        ("jaw_lateral/epochs.png", "Jaw steering by epoch, with controls",
         "Incl. trials where the tongue reached the spout, opening angle, widest opening, reach."),
        ("jaw_lateral/by_position.png", "Jaw sideways displacement by position",
         "Close / mid spouts hide part of the chin: read the far positions."),
    ]),
    ("Per-animal figures", [
        ("prepost/epochs.png", "Session metrics by epoch, per animal", "All post-cue licks unless pre-cue / ITI."),
        ("prepost/timecourse.png", "Session metrics day by day, per animal", ""),
        ("prepost/epochs_by_position_engagement.png", "Engagement metrics by position and epoch",
         "* = that animal's epoch sessions vs its own pre sessions (MWU, FDR)."),
        ("prepost/epochs_by_position_kinematics.png", "Kinematics by position and epoch", "Measured peaks."),
        ("prepost/epochs_by_position_accuracy.png", "Accuracy / mismatch by position and epoch", ""),
        ("prepost/by_position.png", "Contact, reach, protrusion, latency, speed by position", ""),
        ("prepost/mismatch.png", "Tongue-jaw mismatch", "No-contact licks; jaw movement without a lick."),
        ("prepost/angle_tuning.png", "Executed vs target angle", "Per-position median peak angle vs spout angle."),
        ("prepost/lick1_vs_lick2.png", "Lick 1 vs lick 2", "Timing, peak protrusion, peak angle."),
        ("prepost/lick1_lick2_minus_pre.png", "Lick 1 / lick 2, change vs pre (Family 8.1)", ""),
        ("prepost/lick_walk_paired.png", "Within-trial walk: lick 1 vs lick 3 (same trial)", ""),
        ("prepost/lick_walk_angle.png", "Angle at peak protrusion by lick number", "Dashed = spout angle."),
        ("prepost/lick_walk_max_angle.png", "Max angle during the lick by lick number",
         "Tongue >= 50 % of its peak distance only."),
        ("prepost/lick_phase.png", "Lick-peak-aligned protrusion", ""),
        ("prepost/lick_phase_angle.png", "Lick-peak-aligned tongue angle", "Dashed = spout angle."),
        ("prepost/lick_phase_protrusion_px_minus_pre.png", "Lick-peak-aligned protrusion, change vs pre", ""),
        ("prepost/lick_phase_angle_deg_minus_pre.png", "Lick-peak-aligned angle, change vs pre", ""),
        ("prepost/phase_norm_protrusion_px_first_multi.png", "Protrusion vs normalised lick phase", ""),
        ("prepost/phase_norm_angle_deg_first_multi.png", "Angle vs normalised lick phase (tongue out)", ""),
        ("prepost/phase_norm_angle_mouth0_deg_first_multi.png", "Angle vs normalised lick phase (in-mouth = 0)",
         "stroke_orofacial's convention."),
        ("prepost/phase_portrait.png", "Lick phase portraits", ""),
        ("prepost/lick_xy_trajectory_first.png", "Tongue-tip paths: first post-cue lick", ""),
        ("prepost/lick_xy_trajectory_second.png", "Tongue-tip paths: second post-cue lick", ""),
        ("prepost/lick_xy_trajectory_postcue.png", "Tongue-tip paths: all post-cue licks", ""),
        ("prepost/cue_lat_px_minus_pre.png", "Cue-aligned lateral position, change vs pre (Family 8.2)", ""),
        ("prepost/cue_angle_deg_minus_pre.png", "Cue-aligned angle (tongue out), change vs pre", ""),
        ("prepost/cue_angle_mouth0_deg_minus_pre.png", "Cue-aligned angle (in-mouth = 0), change vs pre", ""),
        ("prepost/by_position_licks_first.png", "Per-lick metrics: first post-cue lick", ""),
        ("prepost/by_position_licks_postcue.png", "Per-lick metrics: all post-cue licks", ""),
        ("prepost/by_position_licks_all.png", "Per-lick metrics: all licks (pre-cue + post-cue + ITI)", ""),
    ]),
    ("Appendix -- animal-weighted cohort versions and all-bout phase plots", [
        ("prepost/cohort_metrics_minus_pre_animalw.png", "Per-position metrics, change vs pre (animal-weighted)", ""),
        ("prepost/cohort_cue_minus_pre_inmouth_excl_animalw.png", "Cue-aligned, tongue out (animal-weighted)", ""),
        ("prepost/cohort_lick_time_all_minus_pre_inmouth_excl_animalw.png",
         "All post-cue licks, lick-peak-aligned, tongue out (animal-weighted)", ""),
        ("prepost/cohort_lick_phase_all_minus_pre_inmouth_excl_animalw.png",
         "All post-cue licks, per-lick phase, tongue out (animal-weighted)", ""),
        ("prepost/phase_norm_protrusion_px_all_multi.png", "Protrusion vs normalised phase, every multi-lick bout", ""),
        ("prepost/phase_norm_angle_deg_all_multi.png", "Angle vs normalised phase, every multi-lick bout", ""),
    ]),
]


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _text(slide, text, left, top, width, height, size, bold=False, color=(0, 0, 0)):
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.text = line
        para.font.size = Pt(size)
        para.font.bold = bold
        para.font.color.rgb = RGBColor(*color)
    return tb


def _figure_slide(prs, png: Path, title: str, caption: str, notes: str) -> None:
    from PIL import Image
    from pptx.util import Inches
    s = _blank(prs)
    _text(s, title, 0.3, 0.1, W_IN - 0.6, 0.6, 20, bold=True)
    box_w, box_h, top = W_IN - 0.5, 6.0, 0.75
    with Image.open(png) as im:
        w, h = im.size
    sc = min(box_w / w, box_h / h)
    pw, ph = w * sc, h * sc
    s.shapes.add_picture(str(png), Inches((W_IN - pw) / 2), Inches(top), width=Inches(pw), height=Inches(ph))
    if caption:
        _text(s, caption, 0.3, 6.85, W_IN - 0.6, 0.6, 11, color=(70, 70, 70))
    s.notes_slide.notes_text_frame.text = notes


def _table_slide(prs, title: str, df: pd.DataFrame, note: str = "", font: int = 9) -> None:
    from pptx.util import Inches, Pt
    s = _blank(prs)
    _text(s, title, 0.3, 0.1, W_IN - 0.6, 0.6, 20, bold=True)
    nr, nc = len(df) + 1, len(df.columns)
    tbl = s.shapes.add_table(nr, nc, Inches(0.3), Inches(0.8), Inches(W_IN - 0.6),
                             Inches(min(6.0, 0.28 * nr))).table
    for j, c in enumerate(df.columns):
        tbl.cell(0, j).text = str(c)
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        for j, c in enumerate(df.columns):
            tbl.cell(i, j).text = str(r[c])
    for i in range(nr):
        for j in range(nc):
            for para in tbl.cell(i, j).text_frame.paragraphs:
                para.font.size = Pt(font)
    if note:
        _text(s, note, 0.3, 6.9, W_IN - 0.6, 0.5, 10, color=(70, 70, 70))


def main(argv=None) -> int:
    from pptx import Presentation
    from pptx.util import Inches

    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.orofacial_cohort import LABEL, POS_ORDER, STATS_METHODS, pos_disp
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--baseline", choices=("all", "late"), default="all",
                    help="'late' builds the deck from the *_latepre folders (August-only pre baseline)")
    a = ap.parse_args(argv)
    sfx = "_latepre" if a.baseline == "late" else ""
    rv = PathResolver()
    base = oo.root(rv) / a.run / "cohort"
    figs, tables = base / "figures", base / "tables"
    out = base / "deck"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(tables / f"prepost{sfx}" / "session_metrics.csv")
    ST = pd.read_csv(tables / f"prepost{sfx}" / "stats" / "change_vs_pre_stats.csv")

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W_IN), Inches(H_IN)
    # title
    s = _blank(prs)
    counts = S.groupby(["animal", "epoch"]).size().unstack(fill_value=0)
    cnt = "; ".join(f"{an}: " + ", ".join(f"{e} {int(counts.loc[an].get(e, 0))}" for e in
                                          ("pre", "acute", "subacute", "chronic")) for an in counts.index)
    _text(s, "Orofacial kinematics, pre vs post VLS stroke -- widefield cohort"
          + (" (baseline: LATE pre only)" if sfx else ""), 0.6, 2.2, 12, 1.0, 32, bold=True)
    _text(s, f"Pose model LP r4 (cleaning v35) -- run {a.run}\nCurated sessions: {len(S)} ({cnt})\n"
             f"Built {time.strftime('%Y-%m-%d %H:%M')} by wfield_local.orofacial_deck -- PRELIMINARY until every animal's "
             "sessions are in", 0.6, 3.4, 12, 2.0, 16, color=(60, 60, 60))
    # methods
    s = _blank(prs)
    _text(s, "Data and statistics", 0.3, 0.1, 12.5, 0.6, 22, bold=True)
    methods = ("* Every curated session (early June 6/1-6/5 and 8/5 excluded); ENGAGED trials only (the behaviour "
               "pipeline's reference-position gate; ungated control in figures/prepost_alltrials); LP r4 pose, cleaning v35; "
               "LP-detected licks ('lick n' = the n-th detected post-cue lick, contact or not).\n"
               "* Post-cue licks = cue + 70 ms .. DAQ trial stop; pre-cue = position strobe (spout arrived) .. cue + 70 ms; "
               "ITI = the rest of each trial window. Kinematics use measured (not interpolated / imputed) peaks.\n"
               "* Positions by the lesion: close / far x ipsi / mid / contra (all lesions left: L = ipsi). Angles + = image "
               "right = the L positions (or + = ipsi where marked).\n"
               "* In-mouth frames: EXCLUDED = given the tongue is out, is its trajectory different; INCLUDED (0 px / 0 deg) = "
               "also how much of the time it is out.\n"
               "* Statistics (session = unit; each session minus its animal's own pre mean): headline = Mann-Whitney U, post vs "
               "pre sessions, BH-FDR within each metric; robustness = animal-weighted bootstrap (* both, (*) MWU only, "
               "b bootstrap only); per-animal marks = that animal's sessions vs its own pre; trajectories = cluster "
               "permutation, labels permuted within animal. Full text: tables/prepost/stats/STATS_METHODS.md.")
    _text(s, methods, 0.4, 0.8, 12.5, 6.4, 14)
    s.notes_slide.notes_text_frame.text = STATS_METHODS
    # key results
    k = ST[ST.agree == "both"].copy()
    if len(k):
        k = k.reindex(k.mean_change.abs().sort_values(ascending=False).index)
        k = k.sort_values("q_fdr").head(16)
        tab = pd.DataFrame({"metric": [LABEL.get(m, m) for m in k.metric],
                            "position": [str(p).replace("_", " ") for p in k.pos], "epoch": k.epoch,
                            "change": [f"{v:+.3g}" for v in k.mean_change],
                            "sessions (animals)": [f"{n} ({na})" for n, na in zip(k.n_sessions, k.n_animals)],
                            "q (FDR)": [f"{q:.2g}" for q in k.q_fdr],
                            "bootstrap 95% CI": [f"[{lo:+.3g}, {hi:+.3g}]" for lo, hi in zip(k.boot_lo, k.boot_hi)]})
        _table_slide(prs, "Key results: strongest changes significant by both tests", tab,
                     "All cells: tables/prepost/stats/RESULTS.md and change_vs_pre_stats.csv. PRELIMINARY (PS93 thin).")
    # significance grids per epoch (stroke_orofacial's on-slide grids)
    mk = {"both": "*", "MWU only": "(*)", "bootstrap only": "b", "neither": ""}
    for e in ("acute", "subacute", "chronic"):
        g = ST[(ST.epoch == e) & ST.pos.isin(POS_ORDER)]
        if not len(g):
            continue
        grid = g.assign(mark=g.agree.map(mk)).pivot_table(index="metric", columns="pos", values="mark",
                                                          aggfunc="first").reindex(columns=list(POS_ORDER))
        grid = grid.fillna("")
        grid.insert(0, "metric", [LABEL.get(m, m) for m in grid.index])
        grid.columns = ["metric"] + [pos_disp(p) for p in POS_ORDER]
        _table_slide(prs, f"Significance grid -- {e} vs pre", grid.reset_index(drop=True),
                     "* MWU (BH-FDR q < 0.05) and animal bootstrap agree; (*) MWU only; b bootstrap only.", font=10)
    missing = []
    for sec, items in SECTIONS:
        s = _blank(prs)
        _text(s, sec, 0.6, 3.0, 12, 1.2, 28, bold=True)
        for rel, title, cap in items:
            png = figs / (rel.replace("prepost/", f"prepost{sfx}/").replace("w2/", f"w2{sfx}/").replace("heatmaps/", f"heatmaps{sfx}/"))
            if not png.exists():
                missing.append(rel)
                continue
            _figure_slide(prs, png, title, cap, f"{title}\n\n{cap}\n\nFigure file: {png}\n\n{STATS_METHODS}")
    path = out / f"orofacial_prepost{sfx}_{time.strftime('%Y%m%d')}.pptx"
    prs.save(path)
    print(f"{len(prs.slides)} slides -> {path}")
    if missing:
        print("missing figures (skipped):", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

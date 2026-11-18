"""PI deck: quiet-event slides onto the cam1-guard production run (Priya 11-18 "ok": swap 154-158 + 203, drop the
unverified "< 2 %" note from 112). Run folder v2_quiet_lp_c1guard_k375_20261117 (follow-on scripts at their fixed k = 4,
guard code wf_c1guard1117 f3484347; runner run_downstream_k375.sh). Images cut by c1guard_crops.py; old vs new numbers in
QUIET_C1GUARD_OLD_NEW_20261117.md. Every edit is an exact-match replace on one paragraph (fails loudly otherwise);
the movie swap is the 11-02 edit_movie_intask.py recipe. usage: python deck_swap_c1guard.py <deck_in> <deck_out>"""
import sys
from pathlib import Path

from pptx import Presentation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apply_specs import find_shape, set_para_text, swap_picture  # noqa: E402

T = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
QR = f"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/{T}/cohort1_quiet_spikes/"
OLD = QR + "v2_quiet_lp_20261117/"
NEW = QR + "v2_quiet_lp_c1guard_k375_20261117/"
PRE = "M:/MICROSCOPE/Priya/Widefield/labcams/cohort1_quiet_spikes/v2_quiet_lp/"
SWAP = "M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/"
CODE = "code wf_c1guard1117 f3484347 (cam1 guard)"
RUN = (f"QS_LICK_SOURCE=daq+lp, QS_V2_DIRNAME=v2_quiet_lp_c1guard_k375_20261117, env of the 11-17 deck pass, {CODE}; "
       "runner docs/status/deck_swap_20261117/quiet_c1guard/run_downstream_k375.sh")
CROPS = "cut by docs/status/deck_swap_20261117/quiet_c1guard/c1guard_crops.py, same geometry as the previous image"
NOTE = ("; cam1-only far-from-mouth labels (med_out < -120 px) were still counted as licks in this render -- removing them "
        "(rule of 11-17) changes every per-animal quiet-event rate and model term by < 2 % (re-run, same k 3.75)")
GUARD = ("; cam1-only labels far from the mouth (med_out < -120 px) are not counted as licks (cam1 guard, 11-17)")
MOVIE = NEW + "movie/PS92_0809_vs_PS92_0820_quiet_intask.mp4"
POSTER = NEW + "movie/PS92_0809_vs_PS92_0820_quiet_intask_poster.png"
WIN = NEW + "movie/PS92_0809_vs_PS92_0820_quiet_intask_windows.csv"

TITLES = {112: "Backup: post-stroke gain", 154: "Quiet example", 155: "Quiet traces", 156: "Quiet orofacial events",
          157: "Holds at matched stillness", 158: "Pre-cue event", 203: "Backup: quiet event maps"}

# slide: [(old substring, new substring)] applied to the ONE notes paragraph containing the old substring
NOTES = {
    112: [(NOTE, "")],
    154: [
        ("pre clip DAQ t 8505 s (98 % into the session, before the last cue), 13 orofacial events; acute clip DAQ t 3180 s, "
         "30 events (thresholds 4 x PS92's August-pre robust SD, 1.86 / 1.84 % dF/F)",
         "pre clip DAQ t 8503 s (98 % into the session, before the last cue), 13 orofacial events; acute clip DAQ t 3180 s, "
         "40 events (thresholds 4 x PS92's August-pre robust SD, 1.26 / 1.22 % dF/F)"),
        ("08-11's best in-task 15 s was only 49 % quiet. Pre-stroke engaged animals",
         "08-11's best in-task 15 s was only 49 % quiet (window search of 11-02, pre-rebuild run). Pre-stroke engaged animals"),
        ("(a disengagement-like state; pre quit-period rate is 8-60x engaged quiet, Fig 23.8 [#f-q_quit_map]), so if anything "
         "it OVERSTATES pre events; the pre engaged-quiet rate is 3.7 /min for PS92 vs 22.3 acute (Fig 23.1 [#f-q_fr]).",
         "(a disengagement-like state), so if anything it OVERSTATES pre events; PS92's engaged-quiet rate is 6.6 /min pre vs "
         "32.0 acute (face regressed, slide 156)."),
        (f"Windows: {PRE}movie/PS92_0809_vs_PS92_0820_quiet_intask_windows.csv. Script widefield_pipeline "
         "scripts/cohort2_qc/quiet_event_movie.py --pre PS92_0809 --post PS92_0820 --window intask.",
         f"Windows: {WIN}. Script widefield_pipeline scripts/cohort2_qc/quiet_event_movie.py --pre PS92_0809 --post "
         f"PS92_0820 --window intask ({RUN})."),
        ("FIGURE SOURCE: video PS92_0809_vs_PS92_0820_quiet_intask.mp4 · original: "
         f"{PRE}movie/PS92_0809_vs_PS92_0820_quiet_intask.mp4 · poster: {PRE}movie/PS92_0809_vs_PS92_0820_quiet_intask_poster.png "
         "(quiet_event_movie.py --window intask)",
         f"FIGURE SOURCE: video PS92_0809_vs_PS92_0820_quiet_intask.mp4 · original: {MOVIE} · poster: {POSTER} "
         "(quiet_event_movie.py --window intask, log logs/quiet_event_movie.log)"),
        ("(cam4 + cam1, cam1 droplets removed) define non-quiet frames (QS_LICK_SOURCE=daq+lp)" + NOTE,
         "(cam4 + cam1, cam1 droplets removed) define non-quiet frames (QS_LICK_SOURCE=daq+lp)" + GUARD)],
    155: [
        (NOTE, GUARD),
        ("In this window: pre 1 orofacial event (ipsi, up) and none contra; acute 46 (ipsi) / 56 (contra), up and down.",
         "In this window: pre 1 orofacial event (ipsi, up) and none contra; acute 45 (ipsi) / 56 (contra), up and down."),
        (f"FIGURE SOURCE: {OLD}quiet_event_example_traces.png (scripts/cohort2_qc/quiet_spikes_v2_examples.py, "
         "QS_LICK_SOURCE=daq+lp, QS_V2_DIRNAME=v2_quiet_lp_20261117, env of gb_quiet_lp_g; wf_cleanup2 3354c551 (main))",
         f"FIGURE SOURCE: {NEW}quiet_event_example_traces.png (scripts/cohort2_qc/quiet_spikes_v2_examples.py, {RUN}; "
         f"deck image {SWAP}s155_quiet_examples_PS92_c1guard.png, {CROPS})")],
    156: [
        (NOTE, GUARD),
        ("PS92 6.6 -> 32.0, PS93 10.3 -> 49.6, PS94 4.9 -> 30.9, PS95 3.8 -> 22.9 (4/4). Mixed model on log10(rate + 0.2) vs "
         "August pre: x6.6 acute, x3.3 subacute, x1.7 chronic (q <= .002)",
         "PS92 6.6 -> 32.0, PS93 10.3 -> 49.7, PS94 4.9 -> 31.0, PS95 3.8 -> 23.1 (4/4). Mixed model on log10(rate + 0.2) vs "
         "August pre: x6.6 acute, x3.3 subacute, x1.7 chronic (q <= .003)"),
        ("Both hemispheres rise alike (ipsi x6.5, contra x6.0 acute).",
         "Both hemispheres rise alike (ipsi x6.4, contra x6.1 acute)."),
        (f"FIGURE SOURCE: {OLD}fr_regressed_own/quiet_events_face_regressed.png (scripts/cohort2_qc/quiet_spikes_v2_fr.py, "
         "QS_LICK_SOURCE=daq+lp, QS_V2_DIRNAME=v2_quiet_lp_20261117, env of gb_quiet_lp_g; wf_cleanup2 3354c551 (main))",
         f"FIGURE SOURCE: {NEW}fr_regressed_own/quiet_events_face_regressed.png (scripts/cohort2_qc/quiet_spikes_v2_fr.py "
         f"--jobs 3, {RUN}; deck image {SWAP}s156_quiet_fr_2x2_c1guard.png, {CROPS})")],
    157: [
        ("(QS_LICK_SOURCE=daq+lp, v2_quiet_lp); face video = speed regressor only" + NOTE,
         "(QS_LICK_SOURCE=daq+lp, v2_quiet_lp_c1guard_k375_20261117); face video = speed regressor only" + GUARD),
        ("[traced: code + output file, no run log]", "[traced: render logs quiet_stillness.log, quiet_hemo_check.log]"),
        ("(log10 +0.93 stillest ... +0.36 most active, all q < .01) and is largest when still",
         "(log10 +0.74 stillest, +0.67, +0.52, +0.41, +0.25 most active; q <= .001 in the four stiller bins, q .066 in the "
         "most active) and is largest when still"),
        ("415 nm shows no post-stroke change in quiet SD (acute +0.01, q .77), 1-4 Hz power (-0.23, q .33), 0.5-1 Hz power "
         "(-0.15, q .36) or own-threshold event rate (-0.09, q .67) (orofacial LMM, log10), while corrected 470 rises (SD "
         "+0.21, 1-4 Hz +0.60, 0.5-1 Hz +0.38, rate +0.72, all q < 1e-3; 0.5-1 Hz subacute +0.21 q .02, chronic +0.10 n.s.; "
         "lmm_hemo415.csv); at 470 events the 415 response is ~1-2 % of the 470 amplitude (amplitude column and "
         "uncorrected-470 row cropped off; on the page).",
         "415 nm shows no post-stroke change in quiet SD (acute +0.01, q .75), 1-4 Hz power (-0.23, q .34), 0.5-1 Hz power "
         "(-0.15, q .34) or own-threshold event rate (-0.09, q .67) (orofacial LMM, log10), while corrected 470 rises (SD "
         "+0.28, 1-4 Hz +0.59, 0.5-1 Hz +0.35, rate +0.69, all q < 1e-3; 0.5-1 Hz subacute +0.20 q .057, chronic +0.08 "
         "n.s.; lmm_hemo415.csv); at 470 events the 415 response is small (session median 415 / 470 ratio -0.04) and does "
         "not change acutely (LMM +0.20, q .22) (amplitude column and uncorrected-470 row cropped off)."),
        ("OTHER CONTROLS: not interictal", "OTHER CONTROLS (pre-rebuild run, not re-run): not interictal"),
        (f"original: {PRE}stillness/quiet_stillness.png · deck image: presentations/_source_PI_overview_2026-11-02/build_inputs/"
         "crops/q_stillness_rate.png (crop: rate-by-stillness panel)",
         f"original: {NEW}stillness/quiet_stillness_nolines.png (scripts/cohort2_qc/quiet_stillness.py --jobs 3, then "
         f"--plot-only; {RUN}) · deck image: {SWAP}s157_q_stillness_rate_nolines_c1guard.png (rate-by-stillness panel, "
         f"{CROPS})"),
        (f"original: {PRE}hemo415/quiet_hemo_check.png · deck image: presentations/_source_PI_overview_2026-11-02/build_inputs/"
         "crops/q_hemo415_rows.png (crop: corrected-470 and 415 rows, first 4 columns)",
         f"original: {NEW}hemo415/quiet_hemo_check.png (scripts/cohort2_qc/quiet_hemo_check.py --jobs 3; {RUN}) · deck "
         f"image: {SWAP}s157_q_hemo415_rows_c1guard.png (corrected-470 and 415 rows, first 4 columns, {CROPS})"),
        ("Picture: M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/build_inputs/crops/"
         f"q_stillness_rate_nolines.png from {PRE}stillness/quiet_stillness_nolines.png (encoding page Fig 23.4; "
         "scripts/cohort2_qc/quiet_stillness.py --plot-only, 458f128b)",
         f"Picture: {SWAP}s157_q_stillness_rate_nolines_c1guard.png from {NEW}stillness/quiet_stillness_nolines.png "
         "(scripts/cohort2_qc/quiet_stillness.py --plot-only)")],
    158: [
        (NOTE, GUARD),
        ("-0.020 (p .001) / -0.017 (.001) / -0.020 (.001) / -0.010 (.002); per animal pre PS92 -.018, PS93 -.035, PS94 -.001, "
         "PS95 -.026;",
         "-0.019 (p .001) / -0.017 (.001) / -0.020 (.002) / -0.010 (.003); per animal pre PS92 -.018, PS93 -.032, PS94 .000, "
         "PS95 -.026;"),
        ("p_true 0.22-0.32 with an event vs 0.31-0.38 without", "p_true 0.22-0.32 with an event vs 0.30-0.38 without"),
        (f"FIGURE SOURCE: {OLD}task_events/t13_precue_event_vs_outcome.png (scripts/cohort2_qc/quiet_task_events.py --jobs 3, "
         "QS_LICK_SOURCE=daq+lp, WIDEFIELD_LICK_SOURCE=daq (the deck figure's DAQ 2-s window; the script default since 11-04 "
         "is the LP 1-s lick-free window), QS_V2_DIRNAME=v2_quiet_lp_20261117, env of gb_quiet_lp_g; wf_cleanup2 3354c551 "
         "(main)); tables t13_effects.csv, t13_by_spout.csv; p_true from the rebuild attempt_coding trials_precue_<animal> "
         "tables.",
         f"FIGURE SOURCE: {NEW}task_events/t13_precue_event_vs_outcome.png (scripts/cohort2_qc/quiet_task_events.py --jobs 3, "
         "WIDEFIELD_LICK_SOURCE=daq (the deck figure's DAQ 2-s window; the script default since 11-04 is the LP 1-s "
         f"lick-free window), {RUN}); tables t13_effects.csv, t13_by_spout.csv; deck image "
         f"{SWAP}s158_t13_decoder_c1guard.png ({CROPS}). p_true from the rebuild attempt_coding trials_precue_<animal> "
         "tables of the production camera-class key (csig 3d7bbe921e): the guard re-keys the csig and attempt_coding has "
         "no guard-keyed tables; p_true is the frozen decoder on the DAQ 2-s window, which no cam1 label enters "
         "(summary step via quiet_c1guard/task_events_summarise_prodptrue.py); event flags from the guard run.")],
    203: [
        (NOTE, GUARD),
        ("(outline 542 of 2,022 bins); at the event peak (0 s) a small ipsilesional lateral patch is larger acutely (55 bins); "
         "at +1.0 s small lateral patches are higher (70 bins); -0.3 s: 58 bins, edge-weighted (1.9x rim enrichment). The "
         "event-triggered decay does not change (half-decay time acute -0.0005 s vs pre, p .96).",
         "(outline 542 of 2,022 bins); at the event peak (0 s) a small ipsilesional lateral patch is larger acutely (56 bins); "
         "at +1.0 s small lateral patches are higher (85 bins); -0.3 s: 60 bins, edge-weighted (1.8x rim enrichment). The "
         "event-triggered decay does not change (half-decay time acute -0.0001 s vs pre, p .99)."),
        (f"FIGURE SOURCE: {OLD}eta_fr/quiet_eta_maps_fr.png (scripts/cohort2_qc/quiet_eta_fr.py --jobs 4, QS_LICK_SOURCE=daq+lp, "
         "QS_V2_DIRNAME=v2_quiet_lp_20261117, env of gb_quiet_lp_g; wf_cleanup2 3354c551 (main))",
         f"FIGURE SOURCE: {NEW}eta_fr/quiet_eta_maps_fr.png (scripts/cohort2_qc/quiet_eta_fr.py --jobs 3, {RUN}; deck image "
         f"{SWAP}s203_quiet_eta_maps_orofacial_c1guard.png, {CROPS})")],
}
SLIDE_TEXT = {158: [("strict strata: pre −.020, acute", "strict strata: pre −.019, acute")]}
PICS = {155: [("fig:q_examples", "s155_quiet_examples_PS92_c1guard.png")],
        156: [("fig:q_fr", "s156_quiet_fr_2x2_c1guard.png")],
        157: [("fig:q_stillness", "s157_q_stillness_rate_nolines_c1guard.png"),
              ("fig:q_hemo415", "s157_q_hemo415_rows_c1guard.png")],
        158: [("fig:q_task_outcome", "s158_t13_decoder_c1guard.png")],
        203: [("fig:q_eta_maps_fr", "s203_quiet_eta_maps_orofacial_c1guard.png")]}


def one_para(paras, old, where):
    m = [p for p in paras if old in p.text]
    if len(m) != 1:
        raise SystemExit(f"{where}: {len(m)} paragraphs contain {old[:70]!r}")
    return m[0]


def replace_movie(slide):
    old = find_shape(slide, "movie:quiet_pre_vs_acute")
    x, y, w, h = old.left, old.top, old.width, old.height
    rids = {str(v) for v in old._element.xpath(".//@*[namespace-uri()='http://schemas.openxmlformats.org/"
                                              "officeDocument/2006/relationships']")}
    old_id = old.shape_id
    old._element.getparent().remove(old._element)
    for tgt in slide._element.xpath(f".//p:timing//p:spTgt[@spid='{old_id}']"):
        node = tgt
        while node.tag.split("}")[1] != "video":
            node = node.getparent()
        node.getparent().remove(node)
    for rid in [r for r in rids if r and r in slide.part.rels]:
        slide.part.drop_rel(rid)
    mv = slide.shapes.add_movie(MOVIE, x, y, w, h, poster_frame_image=POSTER, mime_type="video/mp4")
    mv.name = "movie:quiet_pre_vs_acute"


def main(deck_in, deck_out):
    P = Presentation(deck_in)
    for n, title in TITLES.items():
        s = P.slides[n - 1]
        t = next(x.text_frame.text for x in s.shapes if x.has_text_frame and x.text_frame.text.strip())
        assert t.startswith(title), (n, t)
        notes = s.notes_slide.notes_text_frame.paragraphs
        plan = [(one_para(notes, o, f"{n} notes"), o, nw) for o, nw in NOTES.get(n, [])]
        tparas = [p for sh in s.shapes if sh.has_text_frame for p in sh.text_frame.paragraphs]
        plan += [(one_para(tparas, o, f"{n} slide"), o, nw) for o, nw in SLIDE_TEXT.get(n, [])]
        for p, o, nw in plan:
            set_para_text(p, p.text.replace(o, nw))
        for shape, img in PICS.get(n, []):
            swap_picture(s, find_shape(s, shape), SWAP + img)
        if n == 154:
            replace_movie(s)
        left = [p.text[:60] for p in s.notes_slide.notes_text_frame.paragraphs if "v2_quiet_lp_20261117" in p.text
                or "(re-run, same k 3.75)" in p.text or ("cohort1_quiet_spikes/v2_quiet_lp/" in p.text and n != 157)]
        print(n, f"{len(plan)} text edit(s), {len(PICS.get(n, []))} picture(s)", "| still old:" if left else "", left)
    P.save(deck_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

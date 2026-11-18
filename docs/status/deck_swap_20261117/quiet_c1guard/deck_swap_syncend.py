"""PI deck: quiet-event slides 155-159 + 204 onto the sync-end run (Priya 11-18: "yes you can swap the figures and
numbers"). Run v2_quiet_lp_c1guard_syncend_20261118 (QS_ENGAGED_END=lastsync: the analyzed time ends at the last sync
pulse; common k 4; code 871be8f9; runner run_chain_syncend.sh); slides 157-159 show the restyled figures (bootstrap *
primary, mixed-model dagger, change vs own pre, session markers; accepted 11-18). Images cut by syncend_crops.py; old vs
new numbers in docs/status/quiet_restyle_20261118/QUIET_SYNCEND_OLD_NEW_20261118.md.
Edits: GLOBAL exact substring swaps over these slides' notes (run tag, folder, cutter, image names; counts printed),
then per-slide paragraph rewrites selected by a unique prefix (fails loudly if not exactly one match). The movie swap
reuses deck_swap_c1guard.replace_movie.  usage: python deck_swap_syncend.py <deck_in> <deck_out>"""
import sys
from pathlib import Path

from pptx import Presentation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck_swap_c1guard as C1  # noqa: E402
from apply_specs import find_shape, set_para_text, swap_picture  # noqa: E402

NEW = C1.QR + "v2_quiet_lp_c1guard_syncend_20261118/"
RS = NEW + "restyled_20261118/"
SWAP = C1.SWAP
RUN = ("QS_LICK_SOURCE=daq+lp, QS_V2_DIRNAME=v2_quiet_lp_c1guard_syncend_20261118, QS_ENGAGED_END=lastsync (analyzed "
       "time ends at the last sync pulse), common k 4, code 871be8f9 (cam1 guard + task end); runner "
       "docs/status/deck_swap_20261117/quiet_c1guard/run_chain_syncend.sh")
OLD_RUN = ("QS_LICK_SOURCE=daq+lp, QS_V2_DIRNAME=v2_quiet_lp_c1guard_k375_20261117, env of the 11-17 deck pass, code "
           "wf_c1guard1117 f3484347 (cam1 guard); runner docs/status/deck_swap_20261117/quiet_c1guard/run_downstream_k375.sh")
CROPS = "cut by docs/status/deck_swap_20261117/quiet_c1guard/syncend_crops.py"
RESTYLE = "redrawn from the run's saved tables by scripts/cohort2_qc/quiet_restyle_figs.py (QS_RESTYLE_BASE = the run)"
STARS = ("* = nested animals -> sessions bootstrap 95 % CI of the change vs own pre excludes 0 (primary); † = mixed model "
         "q < .05 (backup)")
PAGE = "(https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d)"

GLOBAL = [
    (OLD_RUN, RUN),
    ("cut by docs/status/deck_swap_20261117/quiet_c1guard/c1guard_crops.py, same geometry as the previous image", CROPS),
    ("s155_quiet_examples_PS92_c1guard.png", "s156_quiet_examples_PS92_syncend.png"),
    ("s203_quiet_eta_maps_orofacial_c1guard.png", "s204_quiet_eta_maps_orofacial_syncend.png"),
    ("v2_quiet_lp_c1guard_k375_20261117", "v2_quiet_lp_c1guard_syncend_20261118"),
    ("engaged part of the session (before the quit)",
     "engaged part of the session (before the quit, or the last sync pulse if earlier)"),
    ("Engaged part of the session (before the quit)",
     "Engaged part of the session (before the quit, or the last sync pulse if earlier)"),
    ("Gate: engagement (DAQ quit);", "Gate: engagement (DAQ quit; analyzed time ends at the last sync pulse);"),
    ("Engagement = before the session's quit (quiet_spikes_v2 rule)",
     "Engagement = before the session's quit or the last sync pulse, whichever is first (quiet_spikes_v2 rule + "
     "QS_ENGAGED_END=lastsync)"),
]

TITLES = {155: ("Quiet example", None), 156: ("Quiet traces", None), 157: ("Quiet orofacial events", None),
          158: ("Holds at matched stillness; absent at 415 nm", "Holds at matched stillness; small at 415 nm"),
          159: ("Pre-cue event: weaker code at every epoch, pre-stroke included",
                "Pre-cue event: weaker code pre, acute and chronic"),
          204: ("Backup: quiet event maps", None)}

# slide: [(old substring, new substring)] inside ONE paragraph (notes or slide text)
SUBS = {
    155: [("40 events (thresholds 4 x PS92's August-pre robust SD, 1.26 / 1.22 % dF/F)",
           "43 events (thresholds 4 x PS92's August-pre robust SD, 1.20 / 1.19 % dF/F)"),
          ("PS92's engaged-quiet rate is 6.6 /min pre vs 32.0 acute", "PS92's engaged-quiet rate is 7.1 /min pre vs 36.2 acute")],
    156: [("(orofacial +/-1.26 % ipsi, +/-1.22 % contra)", "(orofacial +/-1.20 % ipsi, +/-1.19 % contra)"),
          ("acute 45 (ipsi) / 56 (contra)", "acute 52 (ipsi) / 57 (contra)")],
    157: [("×6.6 acute, ×3.3 subacute, ×1.7 chronic, 4 / 4 animals",
           "×7.3 acute, ×3.7 subacute, ×1.9 chronic, 4 / 4 animals acutely"),
          ("Events / min after regressing out face movement; open markers = sessions (shape = animal), filled = pooled "
           "mean, 95 % CI; stars = mixed model vs pre",
           "Events / min after regressing out face movement, change vs own pre; open markers = sessions (shape = animal), "
           "filled = animal-weighted change, 95 % CI; * = bootstrap, † = mixed model")],
    158: [("Left: orofacial event rate by face speed (left = stillest). Right: corrected 470 nm (top) vs 415 nm "
           "(bottom): quiet SD, 1–4 Hz and 0.5–1 Hz power, event rate, × each animal's pre",
           "Left: orofacial event rate by face speed (left = stillest). Right: corrected 470 nm (top) vs 415 nm "
           "(bottom): quiet SD, 1–4 Hz and 0.5–1 Hz power, event rate (× own pre), amplitude at 470 events; all as "
           "change vs own pre; * = bootstrap, † = mixed model")],
    159: [("•  A pre-cue orofacial event goes with a weaker position code before and after the stroke (strict strata: "
           "pre −.019, acute −.017, subacute −.020, chronic −.010)",
           "•  A pre-cue orofacial event goes with a weaker position code before and after the stroke (per session: "
           "pre −.019, acute −.020, chronic −.014; subacute −.018, interval touches 0)"),
          ("•  After the stroke it also goes with not attempting far contra (+.045, not in strict strata), not with an "
           "incomplete attempt",
           "•  Subacutely it also goes with a failed far-contra attempt (+.038, 4 / 4 animals), not with an incomplete "
           "attempt")],
}

OTHER = "OTHER CONTROLS (pre-rebuild run, not re-run):"
# slide: [(unique paragraph prefix, full new text)]; a "{KEEP_FROM:x}" tail keeps the old paragraph from x onward
REWRITE = {
    157: [
        ("Correction:", "Correction: " + STARS.replace("mixed model", "mixed model (animal random intercept, "
                                                        "log10(rate + 0.2), epoch vs August pre, BH-FDR)")
         + " [traced: render log quiet_v2_fr.log; restyle quiet_restyle_figs.py]"),
        ("Orofacial events / min (both hemispheres), pre -> acute:",
         "Orofacial events / min (both hemispheres), pre -> acute: PS92 7.1 -> 36.2, PS93 6.5 -> 50.9, PS94 4.9 -> "
         "31.0, PS95 3.8 -> 23.1 (4/4). Change vs own pre (bootstrap, events / min): ipsilesional acute +31.1 [22.2, "
         "42.7], subacute +11.5 [6.1, 16.7], chronic +4.1 [0.6, 9.1] (4/4 each); contralesional +26.4 [16.3, 36.9], "
         "+13.3 [6.9, 21.8], +4.0 [0.25, 10.0] (chronic 3/4). Mixed model on log10(rate + 0.2) vs August pre: x7.3 "
         "acute, x3.7 subacute, x1.9 chronic (q <= .001); before face regression x5.5 / x3.0 / x1.8. Limb also up "
         "(x2.4 acute, largest in PS93 / PS95). Both hemispheres rise alike (ipsi x7.5, contra x6.7 acute). Caveat: "
         "k = 4 was chosen to separate pre from acute, so the acute contrast is optimistic. The analyzed time ends at "
         "the last sync pulse (the task controller stops 0-25 s after the last cue), so imaging after the task no "
         "longer counts as engaged quiet; this mainly lowers PS93's pre (PS93_0806 lost a 12.6-min post-task tail). "
         "PS92_0828 (chronic, day 11), the one high session, keeps only its 2.3 task-period quiet minutes and is still "
         "high there (orofacial 68 / 82, limb 75 / 84 per min); it stays in the statistics."),
        ("FIGURE SOURCE:",
         f"FIGURE SOURCE: {RS}quiet_events_face_regressed_2x2_restyled.png ({RESTYLE}: "
         f"{NEW}fr_regressed_own/quiet_events_fr_sessions.csv + lmm_v2_fr.csv, from scripts/cohort2_qc/quiet_spikes_v2_fr.py "
         f"--jobs 3, {RUN}); deck image {SWAP}s157_quiet_fr_2x2_syncend.png (legend + four panels, {CROPS}); every "
         f"cell's point, CI and stars in {RS}star_changes_combined_vs_deck.csv. 107 of 108 sessions (PS93_0813 has no "
         "FaceRhythm task)."),
    ],
    158: [
        ("Baseline:", "Baseline: August pre (event threshold = 4 x animal's August-pre robust SD); every value shown as "
                      "the change vs the animal's own pre mean (log10 fold for SD / power / rate, difference for the "
                      "event rate by stillness and the amplitude)"),
        ("Correction:", "Correction: " + STARS.replace("mixed model", "LMM epoch vs pre, BH-FDR across stillness bins / "
                                                        "within measure,") + " [traced: render logs quiet_stillness.log, "
                                                        "quiet_hemo_check.log; restyle quiet_restyle_figs.py]"),
        ("PROVISIONAL (n = 4). LEFT",
         "PROVISIONAL (n = 4). LEFT (Fig 23.4 [#f-q_stillness], T7): engaged quiet frames binned by face speed "
         "(FaceRhythm point-PC speed, 2-s smoothed) relative to the animal's pre median. Acutely the face IS stiller "
         "(share of time in the stillest bin +0.14 [0.05, 0.23], 4/4), but the rate rise holds in every stillness bin "
         "(acute change, events / min: +39.2 stillest, +32.5, +19.3, +11.0, +5.9 most active; bootstrap * in all five, "
         "4/4 animals except 3/4 in the most active) and is largest when still: a stiller face is a minor part. "
         "Subacute is starred in every bin, chronic in the two middle bins (0.25-0.5 x pre speed +10.6 [3.7, 18.7], "
         "4/4; 0.5-1 +6.5 [0.2, 15.2], 3/4). RIGHT (Fig 23.11 [#f-q_hemo415], T10): same quiet frames and ROIs. "
         "Corrected 470 rises acutely in every measure (log10 fold: SD +0.30, 1-4 Hz +0.71, 0.5-1 Hz +0.48, rate +0.76, "
         "all 4/4) and stays up subacutely. 415 nm shows no acute change in quiet SD (-0.00 [-0.03, 0.03]), 1-4 Hz "
         "power (-0.17 [-0.52, 0.18]), 0.5-1 Hz power (-0.12 [-0.30, 0.10]) or own-threshold rate (-0.16 [-0.34, "
         "0.02]); its quiet SD is slightly up subacutely (+0.04 [0.01, 0.08], 4/4). The 415 response at corrected-470 "
         "events is above pre acutely (+0.22 % dF/F [0.03, 0.48], 4/4, largest in PS94 +0.59; corrected 470 +0.31) "
         "and subacutely (+0.17 [0.0001, 0.43]): a hemodynamic share of the event response is not excluded, mostly "
         "in one animal. Caveat: per animal the 415 1-4 Hz power varies widely chronically. {KEEP_FROM:" + OTHER + "}"),
        ("FIGURE SOURCE: Movement (Execution) page Fig 23.4",
         f"FIGURE SOURCE: Movement (Execution) page Fig 23.4 [#f-q_stillness] {PAGE} · original: "
         f"{RS}quiet_stillness_restyled.png ({RESTYLE}: {NEW}stillness/ tables, from scripts/cohort2_qc/quiet_stillness.py "
         f"--jobs 3; {RUN}) · deck image: {SWAP}s158_q_stillness_rate_syncend.png (legend + rate-by-stillness panel, "
         f"{CROPS})"),
        ("FIGURE SOURCE: Movement (Execution) page Fig 23.11",
         f"FIGURE SOURCE: Movement (Execution) page Fig 23.11 [#f-q_hemo415] {PAGE} · original: "
         f"{RS}quiet_hemo_check_restyled.png ({RESTYLE}: {NEW}hemo415/ tables, from scripts/cohort2_qc/quiet_hemo_check.py "
         f"--jobs 3; {RUN}) · deck image: {SWAP}s158_q_hemo415_rows_syncend.png (legend, corrected-470 and 415 rows, all "
         f"five columns, {CROPS}; the uncorrected-470 row is on the page)"),
        ("FIGURE RULES 11-03:",
         "FIGURE RULES (11-07 rules, restyle accepted by Priya 11-18): sessions as open markers minus the animal's own "
         "pre mean (shape = animal, color = epoch), filled = animal-weighted change with 95 % CI, epochs side by side "
         "within each stillness bin, no lines across epochs; stars in the top band, no top spine."),
    ],
    159: [
        ("Correction:",
         "Correction: * = nested animals -> sessions bootstrap 95 % CI of the per-session excess excludes 0 (primary; a "
         "within-session contrast, so tested against 0 in each epoch); † = the old trial-pooled within-stratum "
         "permutation p < .05 (backup) [traced: render logs quiet_task_events_daq2s.log, "
         "quiet_task_events_summ_prodptrue.log; restyle quiet_restyle_figs.py]"),
        ("PROVISIONAL (n = 4).",
         "PROVISIONAL (n = 4). Movement (Execution) page Fig 23.5 lower-left panel (T13): frozen pre-stroke decoder "
         "p(true spout) on the 2-s pre-cue window. Per session: the mean, over trials with an orofacial event (> 4 x pre "
         "SD, either polarity) in that window, of p_true minus its STRICT stratum mean (spout x trial-order quartile x "
         "clean-time bin); open markers = sessions (shape = animal), filled = pooled mean with 95 % CI (nested animals "
         "-> sessions); sessions with < 3 event trials left out; PS93_0813 excluded (no video)."),
        ("Excess p_true,",
         "Excess p_true per session, pre / acute / subacute / chronic: -0.019 [-0.031, -0.006] / -0.020 [-0.038, "
         "-0.006] / -0.018 [-0.038, +0.001] / -0.014 [-0.032, -0.001], 4/4 animals in the direction in every epoch; "
         "per animal pre PS92 -.025, PS93 -.022, PS94 -.002, PS95 -.028; acute PS92 -.006, PS93 -.010, PS94 -.032, "
         "PS95 -.058. Trial-pooled STRICT strata (backup): -0.018 (p .001) / -0.017 (.001) / -0.019 (.001) / -0.008 "
         "(.007); hits only, acute + subacute -0.018 (p .001). p_true 0.22-0.32 with an event vs 0.31-0.38 without "
         "(chance 0.167). Far-contra failed attempt, subacute +0.038 [0.007, 0.067] (4/4; other epochs n.s.); "
         "incomplete vs hit n.s. in every epoch; all-spout miss +0.008 [0.001, 0.020] already pre (3/4), +0.012 "
         "subacute (3/4). Most by-spout permutation stars of the old figure do not survive the session-level test."),
        ("FIGURE SOURCE:",
         f"FIGURE SOURCE: {RS}t13_precue_event_vs_outcome_restyled.png ({RESTYLE}: {NEW}task_events/t13_trials.csv.gz, "
         "per-session excess in restyled_20261118/t13_session_excess.csv; trial tables from "
         "scripts/cohort2_qc/quiet_task_events.py --jobs 3, WIDEFIELD_LICK_SOURCE=daq (the deck figure's DAQ 2-s window; "
         f"the script default since 11-04 is the LP 1-s lick-free window), {RUN}); deck image "
         f"{SWAP}s159_t13_decoder_syncend.png (lower-left panel, {CROPS}). p_true from the rebuild attempt_coding "
         "trials_precue_<animal> tables of the production camera-class key (csig 3d7bbe921e): the guard re-keys the "
         "csig and attempt_coding has no guard-keyed tables; p_true is the frozen decoder on the DAQ 2-s window, which "
         "no cam1 label enters (summary step via quiet_c1guard/task_events_summarise_prodptrue.py); event flags from "
         "the sync-end run."),
    ],
    204: [
        ("At +0.2 s acute events",
         "At +0.2 s acute events are followed by less activity over frontal cortex (outline 610 of 2,022 bins); at the "
         "event peak (0 s) an ipsilesional lateral patch is larger acutely (74 bins); at +1.0 s a patch over "
         "ipsilesional frontal to medial cortex is higher (331 bins); -0.3 s: 60 bins, edge-weighted (1.6x rim "
         "enrichment). The event-triggered decay does not change (half-decay time acute -0.0027 s vs pre, p .79). "
         "Image left = ipsilesional."),
    ],
}
PICS = {156: [("fig:q_examples", "s156_quiet_examples_PS92_syncend.png", None)],
        157: [("fig:q_fr", "s157_quiet_fr_2x2_syncend.png", None)],
        158: [("fig:q_stillness", "s158_q_stillness_rate_syncend.png", (274320, 1650000, 4050000, 4700000)),
              ("fig:q_hemo415", "s158_q_hemo415_rows_syncend.png", (4420000, 1650000, 7500000, 4700000))],
        159: [("fig:q_task_outcome", "s159_t13_decoder_syncend.png", None)],
        204: [("fig:q_eta_maps_fr", "s204_quiet_eta_maps_orofacial_syncend.png", None)]}


def one(paras, test, where):
    m = [p for p in paras if test(p.text)]
    if len(m) != 1:
        raise SystemExit(f"{where}: {len(m)} paragraphs match")
    return m[0]


def main(deck_in, deck_out):
    P = Presentation(deck_in)
    gcount = {o: 0 for o, _ in GLOBAL}
    for n, (title, new_title) in TITLES.items():
        s = P.slides[n - 1]
        tpara = next(p for x in s.shapes if x.has_text_frame for p in x.text_frame.paragraphs if p.text.strip())
        assert tpara.text.startswith(title), (n, tpara.text)
        notes = list(s.notes_slide.notes_text_frame.paragraphs)
        sparas = [p for sh in s.shapes if sh.has_text_frame for p in sh.text_frame.paragraphs]
        plan = []
        for o, nw in SUBS.get(n, []):
            p = one(notes + sparas, lambda t, o=o: o in t, f"{n} sub {o[:50]!r}")
            plan.append((p, p.text.replace(o, nw)))
        for pre, nw in REWRITE.get(n, []):
            p = one(notes, lambda t, pre=pre: t.startswith(pre), f"{n} rewrite {pre!r}")
            if "{KEEP_FROM:" in nw:
                head, key = nw.split("{KEEP_FROM:")
                key = key.rstrip("}")
                assert p.text.count(key) == 1, (n, key)
                nw = head + p.text[p.text.index(key):]
            plan.append((p, nw))
        if new_title:
            plan.append((tpara, tpara.text.replace(title, new_title)))
        for p, nw in plan:
            set_para_text(p, nw)
        for p in s.notes_slide.notes_text_frame.paragraphs:
            t = p.text
            for o, nw in GLOBAL:
                if o in t:
                    gcount[o] += t.count(o)
                    t = t.replace(o, nw)
            if t != p.text:
                set_para_text(p, t)
        for shape, img, box in PICS.get(n, []):
            swap_picture(s, find_shape(s, shape), SWAP + img, box=box)
        if n == 155:
            C1.MOVIE, C1.POSTER = (NEW + "movie/PS92_0809_vs_PS92_0820_quiet_intask.mp4",
                                   NEW + "movie/PS92_0809_vs_PS92_0820_quiet_intask_poster.png")
            C1.replace_movie(s)
        left = [p.text[:60] for p in s.notes_slide.notes_text_frame.paragraphs
                if "k375" in p.text or "c1guard_crops" in p.text or "_c1guard.png" in p.text]
        print(n, f"{len(plan)} edit(s), {len(PICS.get(n, []))} picture(s)", "| still old:" if left else "", left)
    for o, k in gcount.items():
        print(f"global x{k}: {o[:70]}")
    P.save(deck_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

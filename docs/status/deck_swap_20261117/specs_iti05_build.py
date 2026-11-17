"""Build specs/iti05.json: the ITI 0..0.5 s window swap (Priya 11-17, DECISIONS 2026-11-17). Every "old" string is read
from a read-only scratch copy of the deck (deck_all.json, dumped from Overview_2026-11-06.pptx at 12:17 on 11-17) and
asserted to exist; notes bodies are the old bodies with the ITI numbers / window phrases replaced (asserted too).
Numbers: render logs of the iti05 re-render (see ITI_WINDOW_AUDIT_20261117.md section 2)."""
import json
import sys
from pathlib import Path

DUMP = Path(sys.argv[1])          # deck_all.json (scratch dump of the current deck)
D = json.loads(DUMP.read_text(encoding="utf-8"))
HERE = Path(__file__).parent
SW = "M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-17/swap_inputs/"
TAG = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
IL = "M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/"
LMM = f"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/{TAG}/lmm_hidden_20261117/iti05_20261117/"
LABELS = ("Window", "Licks", "Trial period", "Trial types", "Gate", "Baseline", "Correction")


def title(n):
    return next(p for sh in D[str(n)]["shapes"] for p in sh.get("paras", []) if p.strip())


def para(n, shape, start):
    for sh in D[str(n)]["shapes"]:
        if sh["name"] == shape:
            for p in sh.get("paras", []):
                if p.startswith(start):
                    return p
    raise SystemExit(f"slide {n}: no paragraph in {shape} starting {start!r}")


def data_and_body(n):
    notes = [p for p in D[str(n)]["notes"] if p.strip()]
    data = {}
    for p in notes[:7]:
        k, v = p.split(": ", 1)
        assert k in LABELS, (n, k)
        data[k] = v
    return data, notes[7:]


def sub(text, reps, n):
    for a, b in reps:
        assert a in text, (n, a[:80])
        text = text.replace(a, b)
    return text


def body_sub(n, reps_by_index):
    _d, body = data_and_body(n)
    out = list(body)
    for i, reps in reps_by_index.items():
        out[i] = sub(out[i], reps, n)
    return out


def txt(n, shape, start, new):
    return {"shape_name": shape, "old": para(n, shape, start), "new": new}


specs = []

# ---------------------------------------------------------------- 105: notes only (row 1 curves and cued row unchanged)
d, _ = data_and_body(105)
d["Window"] = ("ITI: bout-start lick -2..+3 s, response AUC 0..0.5 s (Priya 11-17; ITI window), baseline -1.5..-1.0 s; "
               "cued: first cued lick, AUC 0..1.0 s, baseline -1.0..-0.5 s")
d["Correction"] = ("AUCs: paired nested bootstrap 95 % CI (boot_delta, animals -> sessions), uncorrected; curve bars: per-bin "
                   "bootstrap, BH-FDR q .05 [traced: render log itidock_iti05_20261117/logs/cued_ramp_dock.log (ITI) + "
                   "l_cued_ramp_g.log (cued)]")
old8 = ("Docked ITI response AUC 0-1.0 s / the animal's pre, change vs pre (acute / subacute / chronic): contralesional tjM1 "
        "+2.44 / +2.29 / +2.76 (subacute CI includes 0), SSp-m contra +1.67 / +1.14 / +1.58 (subacute n.s.), SSp-m ipsi "
        "+0.59 / +0.47 (n.s.) / +0.89, tjM1 ipsi +0.70 (n.s.) / +0.75 (n.s.) / +1.48; ALM contra +0.38 acute only; cortex "
        "+0.96 acute only; ALM ipsi n.s.")
new8 = ("Docked ITI response AUC 0-0.5 s / the animal's pre, change vs pre (acute / subacute / chronic): contralesional tjM1 "
        "+1.25 / +0.85 / +1.14, SSp-m contra +1.23 / +0.78 / +1.00, SSp-m ipsi +0.61 / +0.44 / +0.70 (all CIs exclude 0); "
        "tjM1 ipsi +0.60 / +0.45 (n.s.) / +0.84; ALM contra +0.46 acute only; cortex +0.67 / +0.53 / +0.52 (chronic n.s.); "
        "ALM ipsi n.s.")
body = body_sub(105, {1: [(old8, new8)],
                      3: [("itidock_main_20261117/" + TAG + "/figures/lick_iti_ramp_wide.png (row 1, raw dF/F, docked ITI; "
                           "run cued_ramp_dock 11-17)",
                           "itidock_main_20261117/" + TAG + "/figures/lick_iti_ramp_wide.png (row 1, raw dF/F, docked ITI; "
                           "run cued_ramp_dock 11-17; the curves do not depend on the AUC window, the AUC numbers above are "
                           "from the ITI 0..0.5 s re-render itidock_iti05_20261117)")]})
specs.append({"slide": 105, "title_now": title(105), "status": "ready", "skip_reason": "", "pictures": [],
              "slide_text": [], "notes_body": body, "data_block": d,
              "changes": [{"what": "docked ITI response AUC / pre, contralesional tjM1, acute / subacute / chronic",
                           "old": "+2.44* / +2.29 (n.s.) / +2.76* (0..1.0 s)", "new": "+1.25* / +0.85* / +1.14* (0..0.5 s)",
                           "source": "itidock_iti05_20261117/logs/cued_ramp_dock.log, iti_ramp_session_tests_wide.csv"},
                          {"what": "docked ITI response AUC / pre, SSp-m contra", "old": "+1.67* / +1.14 / +1.58*",
                           "new": "+1.23* / +0.78* / +1.00*", "source": "same"},
                          {"what": "docked ITI, cortex", "old": "+0.96* acute only", "new": "+0.67* / +0.53* / +0.52 (n.s.)",
                           "source": "same"}],
              "conclusion_changed": False,
              "conclusion_note": "Title is about cued licks (unchanged). ITI increases shrink by about a third to a half at "
                                 "0..0.5 s but every contralesional tjM1 / SSp-m cell now has a CI excluding 0 (subacute was "
                                 "n.s. at 0..1.0 s, where the pre integral was pulled down by the post-transient dip)."})

# ---------------------------------------------------------------- 108: LMM ITI AUC companion -> 0..0.5 s
d, _ = data_and_body(108)
d["Window"] = ("lick-aligned, per-trial baseline -1.5..-1.0 s, response baseline -1.0..-0.5 s; response AUC 0-0.5 s "
               "(ITI window, Priya 11-17) / the animal's pre, spouts pooled")
d["Correction"] = d["Correction"].replace("lmm_hidden_20261117/logs/iti_ramp_pooled_docked.log",
                                          "lmm_hidden_20261117/iti05_20261117/logs/iti_ramp_pooled_docked.log")
body = body_sub(108, {
    0: [("Docked-spout ITI bout starts, response AUC 0-1.0 s / the animal's pre",
         "Docked-spout ITI bout starts, response AUC 0-0.5 s (ITI window since Priya 11-17) / the animal's pre")],
    2: [("Starred cells: none.", "Starred cells: acute SSp-m contra *, acute SSp-m ipsi *, chronic SSp-m contra *, chronic "
                                 "SSp-m ipsi *, chronic tjM1 ipsi *.")]})
src = (f"FIGURE SOURCE: {SW}s108_iti_ramp_row1_lmm_w0-0.5_iti05.png (title cut of {LMM}iti_ramp_row1_lmm_w0-0.5_iti05.png); "
       f"statistics {LMM}iti_ramp_row1_lmm_w0-0.5_iti05_lmm.csv; source table {IL}{TAG}/cued_wide (key iti) + "
       f"iti_dock_audit.csv; reproduces {IL}itidock_iti05_20261117/{TAG}/iti_ramp_auc_sessions_wide.csv row for row (714) "
       "and its ramp_window_tests.csv auc_0_0.5_n 21/21 cells (checked in the same run, no ALLOW_NO_PUB). Code: wf_lmm_main "
       f"branch lmm-hidden-20261103 scripts/lmm_hidden/ (render log {LMM}logs/iti_ramp_pooled_docked.log carries the code sha "
       "and env).")
body = [p if not p.startswith("FIGURE SOURCE") else src for p in body]
specs.append({"slide": 108, "title_now": title(108), "status": "ready", "skip_reason": "",
              "pictures": [{"shape_name": "fig:iti_ramp_row1_lmm_w0-1", "new_image": SW + "s108_iti_ramp_row1_lmm_w0-0.5_iti05.png",
                            "source_render": LMM + "iti_ramp_row1_lmm_w0-0.5_iti05.png",
                            "crop_done": "title cut (crop.drop_title), as the old notitle image; same pixel size 2174 x 661"}],
              "slide_text": [txt(108, "TextBox 1", "Mixed-model stars: ITI-lick response AUC",
                                 "Mixed-model stars: ITI-lick response AUC 0-0.5 s (docked spout)"),
                             txt(108, "TextBox 3", "•  0 of 21",
                                 "•  5 of 21 tested cells starred (0 **): acute SSp-m contra *, acute SSp-m ipsi *, chronic "
                                 "SSp-m contra *, chronic SSp-m ipsi *, chronic tjM1 ipsi *")],
              "notes_body": body, "data_block": d,
              "changes": [{"what": "LMM starred cells (docked ITI AUC / pre, 21 tests)", "old": "0 of 21 (smallest p .06), 0..1.0 s",
                           "new": "5 of 21 (0 **), 0..0.5 s; contralesional tjM1 acute p .055, chronic .067",
                           "source": LMM + "logs/iti_ramp_pooled_docked.log"},
                          {"what": "bootstrap stars, same cells", "old": "9 of 21", "new": "14 of 21",
                           "source": "itidock_iti05_20261117/ramp_window_tests.csv"}],
              "conclusion_changed": True,
              "conclusion_note": "Hidden backup: the mixed model now supports the ITI increase in SSp-m (both sides, acute + "
                                 "chronic) and tjM1 ipsi chronic; at 0..1.0 s it supported nothing. Contralesional tjM1 stays "
                                 "just above p .05."})

# ---------------------------------------------------------------- 110: notes only (image: ITI row = curves, cued row unchanged)
d, _ = data_and_body(110)
d["Window"] = ("ITI: docked-spout bout-start lick -2..+3 s, baseline -1.5..-1.0 s (AUC numbers in the notes: 0..0.5 s, ITI "
               "window, Priya 11-17); cued: first cued lick, AUC 0..1.0 s, baseline -1.0..-0.5 s; both / session SD of the "
               "same lick windows")
d["Correction"] = d["Correction"].replace("[traced: render log slide95_sessz.log]",
                                          "[traced: render log slide95_sessz_iti05_20261117/logs/slide95_sessz.log]")
body = body_sub(110, {
    1: [("ITI (docked spout) AUC / own pre: contra tjM1 raw +2.44* / +2.29 / +2.76*, z +1.58* / +1.56 / +1.96*; contra SSp-m "
         "raw +1.67* / +1.14 / +1.58*, z +1.02* / +.72 / +1.03*.",
         "ITI (docked spout) AUC 0..0.5 s / own pre: contra tjM1 raw +1.25* / +.85* / +1.14*, z +.65* / +.52* / +.68*; "
         "contra SSp-m raw +1.23* / +.78* / +1.00*, z +.68* / +.46* / +.59*.")],
    2: [("docked ITI vs the docked arm's iti_ramp_auc_sessions_wide.csv (iti_licks/itidock_main_20261117, max |diff| 0).",
         "docked ITI (0..0.5 s) vs the docked arm's iti_ramp_auc_sessions_wide.csv (iti_licks/itidock_iti05_20261117, "
         "714 rows, max |diff| 0).")]})
body = [p.replace("iti_licks/slide95_sessz_20261117/", "iti_licks/slide95_sessz_iti05_20261117/") if p.startswith("FIGURE SOURCE")
        else p for p in body]
specs.append({"slide": 110, "title_now": title(110), "status": "ready", "skip_reason": "", "pictures": [],
              "slide_text": [txt(110, "TextBox 3", "•  ITI (docked)",
                                 "•  ITI (docked): contralesional tjM1 / SSp-m increase keeps ~1/2, every epoch")],
              "notes_body": body, "data_block": d,
              "changes": [{"what": "docked ITI contra tjM1 raw / session-z", "old": "+2.44* / +2.29 / +2.76*; z +1.58* / +1.56 / +1.96*",
                           "new": "+1.25* / +.85* / +1.14*; z +.65* / +.52* / +.68*",
                           "source": "slide95_sessz_iti05_20261117/logs/slide95_sessz.log"},
                          {"what": "share kept after session-z (tjM1 / SSp-m)", "old": "~2/3", "new": "~1/2 (.52-.61), CI excludes 0 in every epoch",
                           "source": "same"}],
              "conclusion_changed": True,
              "conclusion_note": "The ITI share that survives session-z drops from ~2/3 to ~1/2, but is now significant in "
                                 "all three epochs (subacute was n.s.). The image is unchanged (ITI row = curves)."})

# ---------------------------------------------------------------- 111: LMM session-z stack
d, _ = data_and_body(111)
d["Window"] = ("lick-aligned, per-trial baseline -1.5..-1.0 s, response baseline -1.0..-0.5 s; response AUC cued 0-1.0 s, "
               "ITI 0-0.5 s (Priya 11-17) / the session's SD over the same lick windows / the animal's pre, spouts pooled")
d["Correction"] = d["Correction"].replace(
    "lmm_hidden_20261117/logs/sessz_cued.log]",
    "lmm_hidden_20261117/logs/sessz_cued.log (cued) + lmm_hidden_20261117/iti05_20261117/logs/sessz_iti.log (ITI)]")
body = body_sub(111, {
    0: [("spouts pooled, cued (top) and docked ITI (bottom).",
         "spouts pooled, cued (top, AUC 0-1.0 s) and docked ITI (bottom, AUC 0-0.5 s).")],
    3: [("Starred cells: none.", "Starred cells: ITI acute cortex *, ITI chronic SSp-m ipsi *; cued none.")]})
src = (f"FIGURE SOURCE: {SW}s111_lick_lmm_sessz_stack_iti05.png (stack of the cued title cut "
       f"{LMM.replace('iti05_20261117/', '')}cued_ramp_summary_row2_lmm_w0-1_sessz_notitle.png (unchanged) and the ITI title "
       f"cut {LMM}iti_ramp_row1_lmm_w0-0.5_sessz_iti05_notitle.png); statistics <each>_lmm.csv; source tables "
       f"{IL}slide95_sessz_20261117/{TAG}/lmm_cued_sessz_sessions.csv (cued) + {IL}slide95_sessz_iti05_20261117/{TAG}/"
       "lmm_iti_dock_sessz_sessions.csv (ITI, 0..0.5 s). Code: wf_lmm_main branch lmm-hidden-20261103 scripts/lmm_hidden/ "
       f"(render log {LMM}logs/sessz_iti.log carries the code sha and env).")
body = [p if not p.startswith("FIGURE SOURCE") else src for p in body]
specs.append({"slide": 111, "title_now": title(111), "status": "ready", "skip_reason": "",
              "pictures": [{"shape_name": "fig:lick_lmm_sessz", "new_image": SW + "s111_lick_lmm_sessz_stack_iti05.png",
                            "source_render": LMM + "lick_lmm_sessz_stack_iti05.png",
                            "crop_done": "title cut of each panel (crop.drop_title), cued top / ITI bottom stacked (crop.stack), "
                                         "as the old stack; 2174 x 1315 (old 2174 x 1290: the ITI title cut is 25 px taller; "
                                         "keep the shape's box, aspect differs by 2 %)"}],
              "slide_text": [txt(111, "TextBox 3", "•  No cell starred",
                                 "•  Cued: no cell starred after session z-scoring (raw: 7 cells, all subacute / chronic); "
                                 "ITI: 2 cells (raw: 5)"),
                             txt(111, "TextBox 3", "•  Smallest p",
                                 "•  Starred ITI cells: cortex acute (p .034), SSp-m ipsi chronic (p .016); smallest cued p: "
                                 "SSp-m ipsi chronic .060, ALM contra subacute .063, tjM1 contra subacute .064"),
                             txt(111, "TextBox 5", "Stars = mixed-model backup",
                                 para(111, "TextBox 5", "Stars = mixed-model backup").replace(
                                     "response AUC 0-1 s / session SD / the animal's pre.",
                                     "response AUC (cued 0-1 s, ITI 0-0.5 s) / session SD / the animal's pre."))],
              "notes_body": body, "data_block": d,
              "changes": [{"what": "ITI session-z LMM stars", "old": "0 of 21 (smallest p cortex acute .070)",
                           "new": "2 of 21: cortex acute p .034, SSp-m ipsi chronic p .016",
                           "source": LMM + "logs/sessz_iti.log"},
                          {"what": "ITI raw LMM stars (slide 108 twin)", "old": "none", "new": "5", "source": LMM + "logs/iti_ramp_pooled_docked.log"}],
              "conclusion_changed": True,
              "conclusion_note": "Hidden backup: 'no cell starred' no longer holds for ITI (2 of 21 after session-z); cued "
                                 "row unchanged."})

# ---------------------------------------------------------------- 135: notes only (cued row image unchanged; ITI caveat)
d, _ = data_and_body(135)
d["Window"] = ("first cued lick-aligned (base -1..-0.5 s), cue-aligned (base -1..-0.5 s before the cue): AUC 0..1 s; ITI "
               "bout starts: AUC 0..0.5 s (Priya 11-17); the slide shows the cued row")
old9 = ("For ITI bout starts, the response balance at matched landing angle (Fig 14.13, 11-11 rebuild, shared-y redraw 11-16) "
        "does not shift significantly in ALM (+0.07, p .14) or tjM1 (+0.10, p .08) after the stroke; the tjM1 chronic shift "
        "is 3 / 4 animals (PS92 opposite); the orofacial pool shifts +0.15 (p .010).")
new9 = ("For ITI bout starts (response balance over 0..0.5 s), the balance at matched landing angle (Fig 14.13, ITI 0..0.5 s "
        "re-render 11-17) shifts toward the contralesional hemisphere in ALM (+0.06, p .005; subacute +0.09, chronic +0.06, "
        "p .02 each) but not in tjM1 (+0.02, p .58; chronic 3 / 4 animals, PS92 opposite) or the orofacial pool (+0.04, "
        "p .16); SSp-n chronic +0.11 (p .04).")
body = body_sub(135, {2: [(old9, new9)]})
specs.append({"slide": 135, "title_now": title(135), "status": "needs_priya", "skip_reason": "", "pictures": [],
              "slide_text": [], "notes_body": body, "data_block": d,
              "changes": [{"what": "Fig 14.13 ITI balance shift at matched angle (post, LMM)",
                           "old": "ALM +0.07 p .14, tjM1 +0.10 p .08, orofacial +0.15 p .010 (0..1.0 s)",
                           "new": "ALM +0.06 p .005, tjM1 +0.02 p .58, orofacial +0.04 p .16, SSp-n chronic +0.11 p .04 (0..0.5 s)",
                           "source": IL + "lateral_ratio_iti05_20261117/logs/balance_angle.log, balance_angle_lmm.csv"}],
              "conclusion_changed": True,
              "conclusion_note": "Slide title (cued) unchanged, but the ITI caveat flips: at 0..0.5 s the matched-angle ITI "
                                 "balance shift is significant in ALM and no longer in the orofacial pool or tjM1. Priya to "
                                 "decide how much of the caveat to keep (sessions x spouts as observations, anti-conservative)."})

# ---------------------------------------------------------------- 136 / 137 / 138: LMM balance companions
for n, k, _row in ((136, "cued", "first cued lick (row 1 of 3)"), (137, "cue", "cue-aligned (row 2 of 3)"),
                   (138, "iti", "docked ITI bout starts (row 3 of 3)")):
    d, body = data_and_body(n)
    d["Window"] = ("lick- or cue-aligned response, per-trial baseline -1.0..-0.5 s; log2 contra / ipsi AUC 0..1 s (cued / "
                   "cue rows), 0..0.5 s (ITI row, Priya 11-17)")
    d["Correction"] = (d["Correction"].replace("313 tested cells", "312 tested cells")
                       .replace("lmm_hidden_20261117/logs/lateral_ratio_docked.log",
                                "lmm_hidden_20261117/iti05_20261117/logs/lateral_ratio_docked.log"))
    nb = []
    for p in body:
        p = p.replace("313 tested cells", "312 tested cells").replace("313 tests", "312 tests")
        p = p.replace("log2 AUC 0-1 s,", "log2 AUC (cued / cue 0-1 s, ITI 0-0.5 s),")
        if p.startswith("FIGURE SOURCE"):
            p = (f"FIGURE SOURCE: {SW}s{n:03d}_lick_lateral_ratio_auc_docked_iti05_{k}_lmm.png (title cut of "
                 f"{LMM}lick_lateral_ratio_auc_docked_iti05_{k}.png); statistics {LMM}lick_lateral_ratio_auc_docked_iti05_lmm.csv; "
                 f"source table {IL}lateral_ratio_dock_iti05_20261117/{TAG}/lateral_ratio_{{sessions,tests}}.csv (= the "
                 "lateral_ratio_dock_20261117 tables, both windows; 315/315 cells reproduce the bootstrap points). Code: "
                 f"wf_lmm_main branch lmm-hidden-20261103 scripts/lmm_hidden/ (render log {LMM}logs/lateral_ratio_docked.log "
                 "carries the code sha and env).")
        nb.append(p)
    st = []
    b1 = para(n, "TextBox 3", "•  ")
    if n in (136, 137):
        st.append({"shape_name": "TextBox 3", "old": b1, "new": b1.replace("71 of 313", "72 of 312")})
    else:
        st.append({"shape_name": "TextBox 3", "old": b1,
                   "new": "•  14 cells starred in this row (0 **); 72 of 312 across the whole figure"})
        b2 = [p for sh in D["138"]["shapes"] if sh["name"] == "TextBox 3" for p in sh["paras"] if p.strip()][1]
        st.append({"shape_name": "TextBox 3", "old": b2,
                   "new": "•  acute ITI ALM all *, ALM far ipsi / mid / contra *, tjM1 all *, tjM1 far mid / contra *, SSp-m "
                          "far contra *, SSp-n all *, SSp-n far contra *, orofacial all *, orofacial close ipsi / far contra *; "
                          "subacute tjM1 far contra *"})
        nb = [p if not p.startswith("Starred cells:") else
              ("Starred cells: acute iti ALM far ipsi *, acute iti ALM far mid *, acute iti ALM far contra *, acute iti ALM "
               "all *, acute iti tjM1 far mid *, acute iti tjM1 far contra *, acute iti tjM1 all *, acute iti SSp-m far "
               "contra *, acute iti SSp-n far contra *, acute iti SSp-n all *, acute iti orofacial close ipsi *, acute iti "
               "orofacial far contra *, acute iti orofacial all *, subacute iti tjM1 far contra *.") for p in nb]
    t5 = para(n, "TextBox 5", "Stars = mixed-model backup")
    st.append({"shape_name": "TextBox 5", "old": t5, "new": t5.replace("313 tested cells", "312 tested cells")})
    specs.append({"slide": n, "title_now": title(n), "status": "ready", "skip_reason": "",
                  "pictures": [{"shape_name": [sh["name"] for sh in D[str(n)]["shapes"] if sh["name"].startswith("fig:")][0],
                                "new_image": SW + f"s{n:03d}_lick_lateral_ratio_auc_docked_iti05_{k}_lmm.png",
                                "source_render": LMM + f"lick_lateral_ratio_auc_docked_iti05_{k}.png",
                                "crop_done": "slide split (slides.SPLITS, render.redraw, no re-fit) + title cut, as the old "
                                             "image; same pixel size 1950 x 776"}],
                  "slide_text": st, "notes_body": nb, "data_block": d,
                  "changes": [{"what": "whole-figure LMM stars / tests", "old": "71 of 313 (0 **)", "new": "72 of 312 (0 **)",
                               "source": LMM + "logs/lateral_ratio_docked.log"}]
                  + ([{"what": "ITI row LMM stars", "old": "13 (0..1.0 s; incl. subacute / chronic ALM close contra, chronic "
                                                           "ALM far contra)",
                       "new": "14 (0..0.5 s; acute 13 + subacute tjM1 far contra)", "source": LMM + "logs/lateral_ratio_docked.log"}]
                     if n == 138 else
                     [{"what": f"{k} row stars", "old": "unchanged", "new": "unchanged (same cells; only the Holm family "
                       "size changed, 0 ** either way)", "source": LMM + "logs/lateral_ratio_docked.log"}]),
                  "conclusion_changed": n == 138,
                  "conclusion_note": ("ITI row: the post-acute LMM stars move (chronic ALM close / far contra drop, subacute "
                                      "tjM1 far contra appears); acute pattern holds." if n == 138 else "")})

# ---------------------------------------------------------------- 203: full Fig 14.9 docked, ITI row on 0..0.5 s
d, _ = data_and_body(203)
d["Window"] = ("first cued lick-aligned (base -1..-0.5 s), cue-aligned (base -1..-0.5 s before cue): AUC 0..1 s; ITI bout "
               "starts: AUC 0..0.5 s (Priya 11-17; 0..1 s twin on the page)")
d["Correction"] = ("MixedLM LR tests (ML); boot_delta per spout x epoch, no multiple-comparison correction [traced: render log "
                   "lateral_ratio_dock_iti05_20261117/logs/lateral_ratio_dock_plot.log (plot-only from the 11-17 docked "
                   "figure data, both windows) + lateral_ratio_dock.log]")
old7 = ("log2 contra / ipsi AUC 0..1 s per session; open = sessions, filled = mean ± SEM over animals; * (top band) = change "
        "vs pre at that spout, 95 % paired nested bootstrap CI excludes 0. The ITI row is noisier (fewer licks per spout) and "
        "is up acutely in every area: pooled over spouts acute +0.50 ALM, +0.88 tjM1, +0.88 SSp-m, +0.65 SSp-n, +0.67 "
        "orofacial (CIs exclude 0); chronic it stays up only in ALM (+0.18). Spout x epoch LR p .021 ALM, .016 orofacial, "
        ".098 SSp-m, .24 SSp-n (tjM1 fit degenerate). (0..0.5 s window: tjM1 +0.50 / +0.32 / +0.20, SSp-m +0.44 / +0.27 / "
        "+0.22, chronic n.s.)")
new7 = ("log2 contra / ipsi AUC per session, 0..1 s for the cued and cue rows and 0..0.5 s for the ITI row; open = sessions, "
        "filled = mean ± SEM over animals; * (top band) = change vs pre at that spout, 95 % paired nested bootstrap CI excludes "
        "0. The ITI row is noisier (fewer licks per spout) and is up acutely in every area: pooled over spouts acute +0.33 "
        "ALM, +0.50 tjM1, +0.44 SSp-m, +0.56 SSp-n, +0.43 orofacial (CIs exclude 0); subacute still up (+0.20 / +0.32 / "
        "+0.27 / +0.36 / +0.26, CIs exclude 0); chronic it stays up only in ALM (+0.13). Spout x epoch LR p .084 ALM, .016 "
        "tjM1, .050 SSp-m, .081 SSp-n, .005 orofacial. (0..1 s window: acute +0.50 / +0.88 / +0.88 / +0.65 / +0.67, chronic "
        "ALM +0.18; spout x epoch p .021 ALM, .016 orofacial, .098 SSp-m, .24 SSp-n, tjM1 fit degenerate.)")
body = body_sub(203, {0: [(old7, new7)]})
src = (f"FIGURE SOURCE: Encoding (goal) page Fig 14.9 [#f-lick_lateral_ratio_spout] · original: {IL}lateral_ratio_dock_iti05_"
       f"20261117/{TAG}/figures/lick_lateral_ratio_auc.png (ITI 0..0.5 s re-render 11-17: cued / cue rows 0..1 s, ITI row "
       "0..0.5 s; one-window twins lick_lateral_ratio_auc_0-1.png / _0-0.5.png in the same folder; log "
       f"{IL}lateral_ratio_dock_iti05_20261117/logs/lateral_ratio_dock_plot.log) · deck image: presentations/_source_PI_"
       "overview_2026-11-17/swap_inputs/s203_lick_lateral_ratio_auc_docked_iti05.png (full figure)")
body = [p if not p.startswith("FIGURE SOURCE") else src for p in body]
specs.append({"slide": 203, "title_now": title(203), "status": "ready", "skip_reason": "",
              "pictures": [{"shape_name": "fig:lick_lateral_ratio_full", "new_image": SW + "s203_lick_lateral_ratio_auc_docked_iti05.png",
                            "source_render": IL + f"lateral_ratio_dock_iti05_20261117/{TAG}/figures/lick_lateral_ratio_auc.png",
                            "crop_done": "full figure (as the old image); purple provenance footer whited out (bottom 90 px "
                                         "only); 2600 x 1250"}],
              "slide_text": [], "notes_body": body, "data_block": d,
              "changes": [{"what": "docked ITI pooled acute ALM / tjM1 / SSp-m / SSp-n / orofacial",
                           "old": "+0.50 / +0.88 / +0.88 / +0.65 / +0.67 (0..1 s)", "new": "+0.33 / +0.50 / +0.44 / +0.56 / +0.43 (0..0.5 s)",
                           "source": "lateral_ratio_dock_20261117 lateral_ratio_tests.csv (window logr_0-0.5)"},
                          {"what": "chronic pooled", "old": "ALM +0.18 only", "new": "ALM +0.13 only", "source": "same"},
                          {"what": "spout x epoch LR", "old": ".021 ALM, .016 orofacial, .098 SSp-m, .24 SSp-n, tjM1 degenerate",
                           "new": ".084 ALM, .016 tjM1, .050 SSp-m, .081 SSp-n, .005 orofacial",
                           "source": "lateral_ratio_dock_20261117 lateral_ratio_models.csv"}],
              "conclusion_changed": False,
              "conclusion_note": "Backup slide, no title claim; ITI balance still up acutely in every area and chronically "
                                 "only in ALM; magnitudes ~1/3-1/2 smaller and the tjM1 spout x epoch model now converges."})

out = HERE / "specs" / "iti05.json"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
out.write_text(json.dumps(specs, indent=1, ensure_ascii=False), encoding="utf-8")
print(out, len(specs), "slides")

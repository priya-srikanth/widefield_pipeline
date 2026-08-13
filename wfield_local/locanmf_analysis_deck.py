"""Refined ANALYSIS deck — spout-position decode/encode/RSA, grouped ANIMAL -> analysis-type -> date.

A focused successor to the date-first summary deck and the 6/5-8 xsession deck: it includes ONLY the
curated (filtered) sessions (``configs/animals.yaml date_policy.cross_session``) and only the analyses
Priya wants, and it is written to the ``labcams`` TOP LEVEL (not two dirs deep) as
``spout_position_analysis_summary.pptx``.

Sections (per animal, then type, then date):
  A  Per-animal decoding across sessions — post-cue 2 s (no-lick generalization) + pre-cue 2 s
     confusion + recall per date, then the rolling decoder (pre-cue ENL -> post-cue) across sessions.
  B  Per-animal encoder — expected SSp/MO activity by position, encoder predicted maps, explained
     variance per position (raw + ceiling-relative) across sessions, and r2 per Allen region.
  C  Cross-session summary — decoder recall + encoder accuracy across sessions; within-animal consistency.
  D  RSA — within- vs across-animal geometry, per-animal RDM, crossnobis RDM.

Dropped vs the old decks: laterality decoder, top-10 component maps, hemisphere-resolved RDMs, and the
per-session (rather than per-animal) encoder variance panels. Missing PNGs are skipped, so the deck builds
from whatever figures are present.

    python -m wfield_local.locanmf_analysis_deck                 # src = figures_working, out = labcams
    python -m wfield_local.locanmf_analysis_deck --src <dir> --out <path.pptx>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from wfield_local import config
from wfield_local.paths import PathResolver

NAVY = RGBColor(0x1F, 0x33, 0x55)
GREY = RGBColor(0x55, 0x55, 0x55)


# ---- methodology blurbs for the speaker NOTES (how each figure is made) ----
M_COMMON = ("Features = individual LocaNMF component activities (atlas-anchored NMF, r2=0.95, "
            "loc_thresh=80, maxrank=20). Spout position per trial from the DAQ spout-strobe bits; when the "
            "DAQ is short a bit (Aug-2026 dead bit1) it is repaired from the behavior-log pos_idx via "
            "classify_cues_with_backup (only when it validates >=0.9 on the DAQ's good positions). Engaged = "
            "movement/lick trials; the DAQ cue stream is the rewarded subset, which also trims the "
            "disengaged session tail. Curated pre-stroke sessions only (6/6-6/8 + 8/6 onward).")
M_DECODE = ("Decoder: multinomial logistic regression (L2, C=0.5) on standardized component activities, 6 "
            "positions, chance=0.167. Activity = mean over the aligned window, NO per-trial baseline. "
            "Cross-validation is BLOCK-AWARE (GroupKFold, groups = ~6-trial position blocks) so block drift "
            "cannot leak train->test. Post-cue align = window after cue onset (predicts held-out no-lick "
            "trials too = 'no lick generalization'); pre-cue align = the 2 s window ENDING at the cue (the "
            "motor-independent maintained code). Rolling = sliding 0.5 s window across pre-cue ENL -> "
            "post-cue. Per-position recall = diagonal of the row-normalized confusion matrix. " + M_COMMON)
M_FROZEN = ("FROZEN cross-day decoder (wfield_local.locanmf_frozen_decoder --loso). Same multinomial "
            "logistic regression (L2, C=0.5, standardized, chance=0.167), but NO trial from the plotted day "
            "was used to fit it: the model is trained on that animal's OTHER curated days and applied to "
            "this one (leave-one-SESSION-out). This is the pre-stroke dress rehearsal for the post-stroke "
            "confirmatory arm (train pre-stroke, apply post-stroke). "
            "FEATURES ARE ALLEN-ROI, NOT LocaNMF components: LocaNMF components are session-specific in "
            "both count and identity, so they cannot be pooled across days; Allen-ROI features are "
            "atlas-anchored, so column j is the same cortical area every day. Per session the features are "
            "z-scored using that session's own engaged trials, so session-level F0/SNR offsets cannot drive "
            "the result; CV groups are SESSIONS, so each held-out fold is an entire unseen day. The "
            "same-day ceiling quoted on each panel is that session's own within-day block-CV accuracy, so "
            "held-out-day minus ceiling is the true cost of freezing. Measured 2026-08-11: the cost is "
            "POSITIVE for every animal (PS92 +0.102, PS93 +0.012, PS94 +0.044, PS95 +0.047) - the frozen "
            "model beats the same-day model, because it trains on ~3000 trials instead of ~500 and ROI "
            "features are stable across days. Caveat for interpretation: a softmax decoder never abstains, "
            "so confidence alone is NOT evidence of preserved coding - see the OOD control (shuffled-label "
            "entropy floor + no-lick trials, which decode at chance yet stay confident). " + M_COMMON)
M_FROZEN_ENC = ("FROZEN cross-day ENCODER (wfield_local.locanmf_frozen_decoder --loso). Ridge (alpha=1) "
                "from a one-hot position design to Allen-ROI activity, fit on that animal's OTHER curated "
                "days and evaluated on the held-out day (leave-one-SESSION-out) -- the forward-model half "
                "of the post-stroke confirmatory arm, since a frozen encoder's RESIDUAL on post-stroke "
                "trials is the representational-change readout. Same pooling as the frozen decoder: "
                "Allen-ROI features (atlas-anchored, so column j is the same area every day; LocaNMF "
                "components cannot be pooled), z-scored per session using that session's own engaged "
                "trials, CV grouped by SESSION. Each day is shown against its OWN noise ceiling "
                "(between-position SS / total SS) because that is the most any position-only model could "
                "achieve there -- a low EV on a low-ceiling day is a property of the day, not a failure; "
                "FEVE = EV / ceiling is the comparable number. NB the frozen encoder's transfer cost is "
                "NEGATIVE where the frozen DECODER's is positive: the decision boundary transfers across "
                "days, but the exact activity magnitudes do not. Interpret post-stroke encoder residuals "
                "against this pre-stroke cross-day cost, not against zero. " + M_COMMON)
M_LICKFREE = (
    "MOTOR CONTROL for the pre-cue readout (wfield_local.precue_lickfree). Licking is an orofacial "
    "movement that may itself be spout-directed, so pre-cue 'position information' could be ongoing "
    "motor activity rather than a held intention. Trials are split by whether ANY lick falls in the 2 s "
    "window ending at the cue; decode = the pipeline's own block-CV multinomial logistic regression, "
    "encode = per-region cross-validated position EV with a Spearman-Brown-corrected split-half "
    "ceiling. "
    "WHAT THE TASK ALREADY DOES: the strobe->cue interval is an ENFORCED NO-LICK period that licking "
    "RESTARTS, which is why the lead is a median 3.0 s but reaches a p90 of 18.1 s (PS92). The final "
    "2 s is therefore quiet BY CONSTRUCTION, and 90.8-99.5% of windows contain no licks. "
    "RESULT (9 curated sessions/animal, per-session then averaged, chance 0.167): lick-free vs "
    "all-trials accuracy is PS92 0.451/0.465, PS93 0.438/0.438, PS94 0.427/0.433, PS95 0.472/0.462 -- "
    "a maximum difference of 0.014. Dropping EVERY licking trial changes nothing, including in PS93's "
    "8/6-8/9 sessions where exposure falls to 76-87% lick-free. The pre-cue code is NOT lick-driven. "
    "HOW TO READ IT: the lick-free arm is the evidence. The with-licks arm is contrast only and proves "
    "nothing either way -- it is a small self-selected subset (n=49 for PS94, n=0 for PS95), so a low "
    "value there reflects sample size, not the absence of a code. "
    "WHAT THIS DOES NOT ADDRESS: the pre-cue window sits ~3 s AFTER the spout reaches its position, so "
    "it cannot separate a maintained plan from somatosensory contact with the already-positioned "
    "spout. Vision was tested and rejected (removing every visual ROI costs nothing); somatosensation "
    "remains open, and SSp is where the signal concentrates. See DECISIONS.md '\"Pre-cue\" is AFTER the "
    "spout arrives'.")

M_ENCODE = ("Encoder (reverse model): cross-validated ridge regression (alpha=1) from a one-hot position"
            "design to each LocaNMF component's activity, GroupKFold by position block. Per-position "
            "explained variance = held-out R^2 on that position's trials (whole-cortex, summed over "
            "components). Noise ceiling = between-position SS / total single-trial SS (explainable var); "
            "FEVE = captured/ceiling (1 = all explainable captured; center positions often have ~0 ceiling "
            "so low raw EV there is no signal, not failure). Predicted maps = footprint-reconstructed "
            "expected activity per intended position. r2-per-region restricts to each Allen area's "
            "components. Per-animal EV: one graph/animal, sessions distinguished by colour. " + M_COMMON)
M_RSA = ("Per session build a 6x6 representational matrix from the 6 position mean-activity patterns. RDM = "
         "1 - Pearson correlation (diag 0). Second-order RSA = Spearman correlation between two sessions' "
         "RDMs (15 unique off-diagonal entries), basis-free and valid across sessions/animals; within-animal "
         "> across-animal = stable individual geometry; % = within / split-half noise ceiling. Crossnobis = "
         "noise-unbiased (cross-validated Mahalanobis) RDM, removing the positive noise bias. Sessions are "
         "animal-blocked then date-ordered. " + M_COMMON)


def _mmdd_label(mmdd: str) -> str:
    return f"{int(mmdd[:2])}/{int(mmdd[2:])}"


def build_analysis_deck(src: Path, out_path: Path, dates=None, animals=None, tag=None) -> dict:
    """Build the refined analysis deck at ``out_path`` from figures in ``src``. Returns a summary dict."""
    src = Path(src)
    dates = dates or config.cross_session_dates()
    animals = animals or [a for a in config.animals()]
    tag = tag or f"{dates[0]}-{dates[-1]}"
    date_labels = [(d, _mmdd_label(d)) for d in dates]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    BLANK = prs.slide_layouts[6]
    SW, SH = prs.slide_width, prs.slide_height
    placed = {"present": 0, "missing": 0}

    def slide():
        return prs.slides.add_slide(BLANK)

    def title(s, text, sub=None):
        tf = s.shapes.add_textbox(Inches(0.4), Inches(0.16), Inches(12.6), Inches(0.95)).text_frame
        tf.word_wrap = True
        r = tf.paragraphs[0].add_run()
        r.text = text
        r.font.size = Pt(24)
        r.font.bold = True
        r.font.color.rgb = NAVY
        if sub:
            r2 = tf.add_paragraph().add_run()
            r2.text = sub
            r2.font.size = Pt(12.5)
            r2.font.color.rgb = GREY

    def note(s, text):
        """Write the figure's methodology into the slide's speaker notes."""
        s.notes_slide.notes_text_frame.text = text

    def _exists(p):
        ok = Path(p).exists()
        placed["present" if ok else "missing"] += 1
        return ok

    def big(s, p, top=1.4, width=12.7):
        if _exists(p):
            w = Inches(width)
            s.shapes.add_picture(str(p), (SW - w) / 2, Inches(top), width=w)

    def grid(s, paths, cols=2, top=1.25, side=0.25, gap=0.18, bottom=0.25):
        paths = [Path(p) for p in paths]
        present = [p for p in paths if _exists(p)]
        if not present:
            return
        rows = (len(present) + cols - 1) // cols
        cell_w = (SW - Inches(side) * 2 - Inches(gap) * (cols - 1)) / cols
        cell_h = (SH - Inches(top) - Inches(bottom) - Inches(gap) * (rows - 1)) / rows
        for i, p in enumerate(present):
            r, c = divmod(i, cols)
            iw, ih = Image.open(str(p)).size
            scale = min(cell_w / iw, cell_h / ih)
            w, h = int(iw * scale), int(ih * scale)
            left = Inches(side) + c * (cell_w + Inches(gap)) + (cell_w - w) / 2
            t = Inches(top) + r * (cell_h + Inches(gap)) + (cell_h - h) / 2
            s.shapes.add_picture(str(p), left, t, width=w, height=h)

    def divider(text, sub=None):
        s = slide()
        tf = s.shapes.add_textbox(Inches(0.8), Inches(2.9), Inches(11.7), Inches(1.9)).text_frame
        tf.word_wrap = True
        r = tf.paragraphs[0].add_run()
        r.text = text
        r.font.size = Pt(34)
        r.font.bold = True
        r.font.color.rgb = NAVY
        if sub:
            r2 = tf.add_paragraph().add_run()
            r2.text = sub
            r2.font.size = Pt(15)
            r2.font.color.rgb = GREY

    def sess(label, align):
        return src / f"locanmf_position_session_{label}_locanmf_{align}_base-none_cv-block.png"

    # ---------------- title ----------------
    s = slide()
    tf = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(3.0)).text_frame
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = "Spout-position decoding & encoding from cortex"
    r.font.size = Pt(38)
    r.font.bold = True
    r.font.color.rgb = NAVY
    for t in [f"Curated pre-stroke sessions ({', '.join(_mmdd_label(d) for d in dates)}) — "
              f"{', '.join(animals)} (PS93 = right orofacial deficit)",
              "Individual LocaNMF components, block-aware CV, no per-trial baseline, chance = 0.17.",
              "Grouped by animal, then analysis type, then date."]:
        rr = tf.add_paragraph().add_run()
        rr.text = t
        rr.font.size = Pt(15)
        rr.font.color.rgb = GREY
    note(s, "Refined analysis deck: spout-position decode/encode/RSA, grouped animal -> analysis type -> "
            "date, curated pre-stroke sessions only. Each slide's speaker notes give how that figure is "
            "made. " + M_COMMON)

    # ---------------- A. per-animal decoding ----------------
    divider("A. Per-animal decoding across sessions",
            "Post-cue 2 s (predicts no-lick trials too = no lick generalization) and pre-cue 2 s "
            "(maintained position code) confusion + recall; then the rolling decoder across sessions.")
    for a in animals:
        s = slide()
        title(s, f"{a} — post-cue 2 s decoder (engaged, no-lick generalization)",
              "Per session: confusion matrix + per-position recall (engaged vs held-out no-lick trials).")
        note(s, M_DECODE)
        grid(s, [sess(f"{a}_{d}", "cue") for d, _ in date_labels], cols=3)
        s = slide()
        title(s, f"{a} — pre-cue 2 s decoder (maintained position code)",
              "Position decodable in the pre-cue ENL window, before movement — a motor-independent code.")
        note(s, M_DECODE)
        grid(s, [sess(f"{a}_{d}", "precue") for d, _ in date_labels], cols=3)
        s = slide()
        title(s, f"{a} — rolling decoder across sessions (pre-cue ENL → post-cue)",
              "Sliding 0.5 s window, block-CV, one line per session. Above-chance in the ENL = maintained code. "
              "(Per-animal accuracy across sessions is in the cross-mouse summary, Section C.)")
        note(s, M_DECODE)
        big(s, src / f"locanmf_decoder_rolling_by_animal_{a}.png", top=1.5, width=11.2)
        # Frozen cross-day decoder: same per-date confusion layout as the within-day slides above, but
        # each day is predicted by a model trained ONLY on this animal's other days (ROI features).
        #
        # PAGINATED 4-per-slide (2x2). Putting all 8 curated dates on one slide at cols=2 gives 4 rows,
        # so each panel gets ~1/4 of the slide height and the 6x6 confusion cells become unreadable.
        # 2x2 doubles the height per panel; the cost is one extra slide per animal.
        pages = [date_labels[i:i + 4] for i in range(0, len(date_labels), 4)]
        # BOTH alignments: post-cue (readout during/after the movement) and PRE-CUE (the maintained,
        # motor-independent code). The pre-cue one is the readout the stroke arm leans on, so whether
        # IT survives freezing across days is the more consequential question.
        for al, al_name, al_desc in (
                ("cue", "post-cue 2 s", "the readout during/after the movement"),
                ("precue", "PRE-CUE 2 s", "the maintained, motor-independent code — the window "
                                          "ENDING at the cue, before any movement")):
            for page in pages:
                s = slide()
                span = f"{page[0][1]}–{page[-1][1]}" if len(page) > 1 else page[0][1]
                suffix = f"  ({span})" if len(pages) > 1 else ""
                title(s, f"{a} — FROZEN cross-day decoder, {al_name}, held-out day (Allen-ROI){suffix}",
                      f"Per date: confusion + per-position recall from a decoder trained on this "
                      f"animal's OTHER days only (leave-one-session-out). {al_desc}. ROI features, not "
                      f"LocaNMF components — components are session-specific and cannot be pooled.")
                note(s, M_FROZEN)
                grid(s, [src / f"locanmf_frozen_session_{a}_{d}_roi_{al}.png" for d, _ in page],
                     cols=2, top=1.35)
            s = slide()
            title(s, f"{a} — FROZEN decoder ({al_name}): transfer cost & out-of-distribution control",
                  "Held-out day vs same-day ceiling per session; the cost of freezing across days; and "
                  "the OOD control — a softmax decoder never abstains, so confidence alone is not "
                  "evidence.")
            note(s, M_FROZEN)
            big(s, src / f"locanmf_frozen_decoder_loso_roi_{al}.png", top=1.9, width=12.7)
            s = slide()
            title(s, f"{a} — FROZEN cross-day ENCODER ({al_name}, Allen-ROI): position → activity",
                  "Held-out-day EV against that day's own noise ceiling, and the ceiling-normalised "
                  "FEVE. The forward model for post-stroke residuals — note its transfer cost is "
                  "NEGATIVE where the decoder's is positive.")
            note(s, M_FROZEN_ENC)
            big(s, src / f"locanmf_frozen_encoder_loso_roi_{al}.png", top=1.9, width=12.7)

    # ---------------- B. per-animal encoder ----------------
    divider("B. Per-animal encoder — expected activity, predicted maps & explained variance",
            "Position → expected cortical activity (SSp / MO), footprint-reconstructed predicted maps, and "
            "encoding explained variance per position (raw + relative to the noise ceiling) across sessions.")
    for a in animals:
        s = slide()
        title(s, f"{a} — expected SSp / MO activity by position (encoder, across sessions)",
              "Predicted per-position time-course of pooled SSp and MO activity. One panel per session.")
        note(s, M_ENCODE)
        grid(s, [src / f"locanmf_encoder_temporal_{a}_{d}.png" for d, _ in date_labels], cols=3)
        s = slide()
        title(s, f"{a} — encoder predicted maps by intended position (across sessions)",
              "Footprint-reconstructed expected cortical map per intended spout position. One panel per session.")
        note(s, M_ENCODE)
        grid(s, [src / f"locanmf_encoder_predicted_maps_{a}_{d}.png" for d, _ in date_labels], cols=3)
        s = slide()
        title(s, f"{a} — encoder explained variance per position across sessions (raw & vs ceiling)",
              "One graph per animal; sessions distinguished by colour/marker. Left: raw held-out R²; "
              "right: relative to the per-position noise ceiling.")
        note(s, M_ENCODE)
        grid(s, [src / f"locanmf_encoder_ev_by_position_animal_{a}.png",
                 src / f"locanmf_encoder_ev_ceiling_by_position_animal_{a}.png"], cols=2, top=1.5)
        s = slide()
        title(s, f"{a} — encoder r² per Allen region across sessions (MOp, MOs, SS areas, …)",
              "Explained variance per region; one panel per session. Absolute (explainable vs captured) + FEVE.")
        note(s, M_ENCODE)
        grid(s, [src / f"locanmf_encoder_r2_by_region_{a}_{d}.png" for d, _ in date_labels], cols=3)
    s = slide()
    title(s, "Encoder — explained-variance fraction (FEVE) by region, pooled per animal",
          "Fraction of EXPLAINABLE variance captured per Allen region, pooled over each animal's curated sessions.")
    note(s, M_ENCODE)
    big(s, src / "locanmf_encoder_feve_by_region_pooled.png", top=1.5, width=12.9)
    s = slide()
    title(s, "Encoder — FEVE by region, individual sessions per animal",
          "Same metric, one row per session → session-to-session stability.")
    note(s, M_ENCODE)
    big(s, src / "locanmf_encoder_feve_by_region_sessions.png", top=1.5, width=12.9)

    # ---------------- B2. pre-cue without licking ----------------
    divider("B2. Pre-cue code without licking — the motor-confound control",
            "Decoding AND encoding restricted to trials with no licks in the 2 s pre-cue window.")
    for src_name in ("roi", "locanmf"):
        for a in animals:
            p = src / f"precue_lickfree_{a}_{src_name}.png"
            if not p.exists():
                continue
            s = slide()
            title(s, f"{a} — pre-cue position code with NO licking in the window ({src_name})",
                  "Exposure, decode (lick-free vs all vs with-licks), lick-free confusion matrix, and "
                  "per-region encoding EV.")
            note(s, M_LICKFREE)
            big(s, p, top=1.5, width=12.9)

    # ---------------- C. cross-session summary ----------------
    divider("C. Cross-session summary — decoder recall & encoder accuracy across sessions")
    s = slide()
    title(s, f"Cross-mouse decoding & encoding across sessions ({_mmdd_label(dates[0])}–{_mmdd_label(dates[-1])})",
          "Per-mouse overall + per-position decoding and encoding EV, mean ± SEM across that animal's sessions "
          "(points = sessions).")
    note(s, M_DECODE + " " + M_ENCODE)
    big(s, src / f"locanmf_cross_mouse_comparison_{tag}.png", top=1.5, width=12.7)
    s = slide()
    title(s, "Within-animal consistency of per-position decode / encode",
          "Per-position profile per session + mean ± SD (the session-to-session noise floor).")
    note(s, M_DECODE + " " + M_ENCODE)
    big(s, src / f"locanmf_within_animal_consistency_{tag}.png", top=1.5, width=12.9)

    # ---------------- D. RSA ----------------
    divider("D. RSA — representational geometry of spout position",
            "Within- vs across-animal second-order RSA, per-animal RDM, and the noise-unbiased crossnobis RDM.")
    s = slide()
    title(s, "RSA — within- vs across-animal representational geometry",
          "6×6 position RDM per session; 2nd-order RSA (basis-free). Within-animal > across = stable individual geometry.")
    note(s, M_RSA)
    big(s, src / f"locanmf_rsa_sessions_{tag}.png", top=1.6, width=13.0)
    s = slide()
    title(s, "RSA — mean representational dissimilarity matrix per animal",
          "How the 6 positions relate (dark = similar patterns, bright = distinct).")
    note(s, M_RSA)
    big(s, src / f"locanmf_rsa_rdms_{tag}.png", top=1.9, width=12.7)
    s = slide()
    title(s, "RSA — crossnobis (noise-unbiased) RDM",
          "Crossnobis removes the positive noise bias → the honest cross-day / pre-post geometry metric.")
    note(s, M_RSA)
    big(s, src / f"locanmf_rsa_crossnobis_{tag}.png", top=1.65, width=13.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return {"out": str(out_path), "slides": len(prs.slides),
            "figures_present": placed["present"], "figures_missing": placed["missing"], "tag": tag}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None, help="figure dir (default: figures_working root)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output pptx (default: <labcams>/spout_position_analysis_summary.pptx)")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    rv = PathResolver(machine=args.machine)
    src = args.src or Path(rv.root("figures_working"))
    out = args.out or (Path(rv.root("labcams")) / "spout_position_analysis_summary.pptx")
    summary = build_analysis_deck(src, out)
    print(f"[analysis_deck] wrote {summary['out']}  ({summary['slides']} slides, "
          f"{summary['figures_present']} figures placed, {summary['figures_missing']} missing)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

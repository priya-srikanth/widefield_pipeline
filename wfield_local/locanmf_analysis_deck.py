"""Refined ANALYSIS deck — spout-position decode/encode/RSA, grouped ANIMAL -> analysis-type -> date.

A focused successor to the date-first summary deck and the 6/5-8 xsession deck: it includes ONLY the
curated (filtered) sessions (``configs/animals.yaml date_policy.cross_session``) and only the analyses
Priya wants, and it is written to the ``labcams`` TOP LEVEL (not two dirs deep) as
``spout_position_analysis_summary.pptx``.

Sections (per animal, then type, then date):
  A  WITHIN-DAY decoding — post-cue 2 s (no-lick generalization) + pre-cue 2 s confusion + recall per
     date, then the rolling decoder (pre-cue ENL -> post-cue) across sessions.
  B  WITHIN-DAY encoder — expected SSp/MO activity by position, encoder predicted maps, explained
     variance per position (raw + ceiling-relative) across sessions, and r2 per Allen region.
  C  Pre-cue code without licking — the motor-confound control on the study's key readout.
  D  CROSS-SESSION (frozen) decoders and encoders, in TWO INDEPENDENT BASES — Allen-ROI and the shared
     joint-LocaNMF basis. Each day is predicted by a model trained only on that animal's OTHER days.
     Two bases because a cross-day claim that holds in only one parcellation is a claim about the
     parcellation; one that holds in both is a claim about the cortex.
  E  Cross-session summary — decoder recall + encoder accuracy across sessions; within-animal consistency.
  F  RSA — within- vs across-animal geometry, per-animal RDM, crossnobis RDM.

RESTRUCTURED 2026-08-13: the frozen cross-day slides used to be buried inside each animal's Section A,
which mixed two different questions (does the code exist today? does it transfer across days?) under one
heading and put ~6 cross-day slides between an animal's within-day decode and its encoder. They are now
Section D, in both bases side by side.

Dropped vs the old decks: laterality decoder, top-10 component maps, hemisphere-resolved RDMs, and the
per-session (rather than per-animal) encoder variance panels. Also NOT here: the frozen fixed-A /
refit-C basis, REJECTED because its score depends on which session is nominated as the reference and no
reference wins for every animal (within-animal swing up to 0.36) — the joint basis in Section D is the
reference-free version of that same idea. See DECISIONS.md. Missing PNGs are skipped, so the deck builds
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
            "cue trials with a lick inside the response window; unengaged (no-lick) trials are kept "
            "separately as the generalization / OOD arm rather than discarded. NB the DAQ cue stream is "
            "NOT a rewarded subset -- an earlier note here said it was, corrected 2026-08-09: DAQ cue "
            "count equals the behavior log's scored-trial count exactly in every session and includes "
            "unrewarded trials, so unrewarded trials remain available for the post-stroke failed-attempt "
            "analysis. Curated pre-stroke sessions only (6/6-6/8 + 8/6 onward). HEMODYNAMIC/DRIFT REMOVAL (adopted 2026-08-14, docs/PREPROCESSING_DECISION.md): every panel in this deck -- decoders, encoders, RSA and the activity MAPS -- is built on the meegkit_hpfit SVTcorr, not the pipeline default. The default removes drift with a ZERO-PHASE 0.1 Hz filter, which is acausal: it smears each post-cue response BACKWARDS and inflated pre-cue decoding by ~0.21 across 36 sessions. meegkit_hpfit keeps that high-pass for the hemodynamic COEFFICIENT fit (which is what it is for) and replaces it for the OUTPUT with de Cheveigne robust polynomial detrending (order 10, 600 s) on a mask excluding whole trials. Post-cue decoding IMPROVED (0.684 -> 0.759) and the shadow signature vanished (negative pre/post correlation in 30/36 sessions -> 2/36).")
M_DECODE = ("Decoder: multinomial logistic regression (L2, C=0.5) on standardized component activities, 6 "
            "positions, chance=0.167. Activity = a SUB-BINNED TIME COURSE over the aligned window (adopted 2026-08-14), NO per-trial baseline: the window is split into equal bins and their means concatenated, so the decoder sees the window's temporal profile rather than one number. Pre-cue and post-cue use 4 x 0.5 s, post-lick 8 x 0.25 s (configs/defaults.yaml decode.bins). Measured on corrected data, 16 sessions: +0.032 pre-cue, +0.020 post-cue, +0.023 post-lick over the plain 2 s mean. Bin WIDTH matters only post-event -- 0.25 s wins post-lick but OVER-slices pre-cue, where 0.5 s is better. "
            "Cross-validation is BLOCK-AWARE (GroupKFold, groups = ~6-trial position blocks) so block drift "
            "cannot leak train->test. Post-cue align = window after cue onset (predicts held-out no-lick "
            "trials too = 'no lick generalization'); pre-cue align = the 2 s window ENDING at the cue (the "
            "pre-cue POSITION INFORMATION, not lick-driven). Rolling = sliding 0.5 s window across ENL -> "
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
    "motor activity rather than a held intention. "
    "THE WINDOW IS SEARCHED, NOT FIXED. Each trial's window is 2 CONSECUTIVE SECONDS lying between the "
    "spout-position strobe and the cue and containing NO licks: the window slides back through the "
    "strobe->cue interval until it finds such a stretch, taking the LATEST one that fits (closest to "
    "the cue = most informative about the upcoming action). Trials whose 2 s ending exactly at the cue "
    "is already clean keep it, so the common case stays cue-aligned. This is what recovers a trial that "
    "has one lick 200 ms before the cue but 2 s of clean data just earlier -- a fixed window would "
    "discard the whole trial. Features are the MEAN over that 2 s (the window integrates the whole 2 s; "
    "it is not an instantaneous sample). Bounded at the strobe on purpose: before the spout arrives "
    "this trial's position does not exist yet, and because the task avoids recent repeats, prior-trial "
    "activity predicts the upcoming position (last-5-distinct -> next is the missing one 45-53% vs ~17% "
    "uniform), so a window straying earlier would manufacture a pre-cue code. window_offset_s reports "
    "how far recovered windows sit from the cue. Decode = the pipeline's own block-CV multinomial "
    "logistic regression; encode = per-region cross-validated position EV with a Spearman-Brown "
    "split-half ceiling. "
    "WHAT THE TASK ALREADY DOES: the strobe->cue interval is an ENFORCED NO-LICK period that licking "
    "RESTARTS, which is why the lead is a median 3.0 s but reaches a p90 of 18.1 s (PS92). The final "
    "2 s is therefore quiet BY CONSTRUCTION, and 90.8-99.5% of fixed windows already contain no licks; "
    "the search recovers most of the rest. "
    "HOW TO READ IT: the lick-free arm is the evidence -- information present with NO licking in the "
    "window cannot be lick-driven. The with-licks arm is contrast only and proves nothing either way: "
    "it is now only those trials where no clean 2 s exists ANYWHERE in the interval, a small "
    "self-selected subset, so a low value there reflects sample size, not the absence of a code. "
    "WHAT THIS DOES NOT ADDRESS: the pre-cue window sits ~3 s AFTER the spout reaches its position, so "
    "it cannot separate a held intention from somatosensory contact with the already-positioned "
    "spout. THAT IS ACCEPTED, NOT A DEFECT (Priya, 2026-08-13): the readout is pre-cue POSITION "
    "INFORMATION, which is informative if it changes post-stroke whichever of the two it is. Vision "
    "was tested and rejected (removing every visual ROI costs nothing); SSp is where the signal "
    "concentrates, which is consistent with a substantial somatosensory contribution. See DECISIONS.md.")

M_NOLICK = (
    "NO-DETECTED-LICK REFERENCE (wfield_local.nolick_decoder / nolick_analysis). Trained on ENGAGED "
    "trials (first detected lick within 2 s of the cue) and applied, frozen, to trials without one. "
    "Purpose: post-stroke a failed trial can mean the plan was never formed OR that it was formed "
    "and the movement failed, and the behaviour log cannot tell those apart. They make opposite "
    "predictions here -- plan-intact keeps the PRE-cue code while the POST-cue code collapses, "
    "because post-cue decoding is largely driven by the lick itself. "
    "\n\nTHREE ARMS, NOT TWO. 'late' = a detected lick 2-5 s after the cue; 'undetected' = none "
    "within 5 s. The pipeline's older arm pooled them, which is misleading: on PS93 8/12 the pre-cue "
    "survival is carried entirely by LATE trials (balanced 0.532, p=0.003) while undetected trials "
    "show nothing (0.153, p=0.76). "
    "\n\nTHE NULL IS NOT 1/6. These trials are heavily skewed across positions (PS93: 49% far_center) "
    "and the decoder's predictions on them are skewed too, so an information-free decoder scores "
    "above 1/6 -- 0.211 for PS93. Headline is BALANCED accuracy (macro-recall), whose null "
    "expectation is exactly 1/6 however skewed either side is; raw accuracy is judged against a "
    "permutation null computed on these trials with predictions held fixed; a position-matched "
    "subsample is stored as an independent check. An earlier 'above chance' flag here compared "
    "against uniform 1/6 and was meaningless; it is retired. "
    "\n\n'NO DETECTED LICK' IS NOT 'NO ATTEMPT'. The sensor needs contact, so an executed but short "
    "lick registers as nothing. PS93 has a pre-existing rightward tongue bias and reaches far_L "
    "poorly (Priya, 2026-08-17), so its far-position undetected trials are substantially "
    "attempted-and-short -- which makes PS93 far_L a PRE-stroke, within-subject instance of the very "
    "phenotype this analysis looks for post-stroke. DLC/facial tracking is needed to split attempted "
    "from unattempted; until then read the per-position breakdown, not the pooled number.")

M_PRECUE_CAVEAT = (
    "\n\nPRE-CUE NUMBERS ON THIS SLIDE ARE CORRECTED (as of 2026-08-14). They are built on the "
    "meegkit_hpfit SVTcorr, not the pipeline default. Read this before comparing them to anything "
    "produced before 14 Aug, which was inflated. "
    "THE ARTIFACT: wfield.hemodynamic_correction high-passes both channels at 0.1 Hz with scipy "
    "filtfilt -- zero-phase, therefore ACAUSAL -- and the high-passed 470 channel becomes SVTcorr. A "
    "zero-phase filter's impulse response is symmetric in time (measured on this filter: -0.496 before "
    "an impulse, -0.496 after), so each position-specific POST-cue response cast a scaled, SIGN-FLIPPED "
    "shadow BACKWARDS over the pre-cue window, and a linear decoder does not care about sign. "
    "MEASURED OVER ALL 36 CURATED SESSIONS: pre-cue 0.486 -> 0.352, while POST-CUE IMPROVED 0.684 -> "
    "0.759 -- the fix helps the readout we trust while shrinking the one we suspected, which is the "
    "strongest form the comparison could take. The mechanism check agrees: the pre-cue pattern was "
    "ANTI-correlated with the post-cue pattern (negative in 30 of 36 sessions); after correction that "
    "signature is gone (2 of 36). "
    "WHAT SURVIVES: pre-cue position information is REAL and significant in 35/36 sessions, at ~72% of "
    "the previously reported size -- PS92 0.225, PS93 0.349, PS94 0.500, PS95 0.334 (chance 0.167, "
    "empirical null 0.137-0.147 by block-label permutation). PS94 was essentially untouched; PS92 was "
    "the one substantially inflated and is now well above chance rather than at it. The cohort is NOT "
    "uniform, which is a result in its own right. "
    "This was the UPSTREAM method, not a local bug: churchlandlab/WidefieldImager SvdHemoCorrect.m does "
    "the same in-place filtfilt and Musall et al. 2019 state it in their methods. The artifact class is "
    "published -- van Driel, Olivers & Fahrenfort 2021, J Neurosci Methods -- including the negative "
    "sign, with trial-masked robust detrending as the recommended fix, which is what was adopted. "
    "TERMINOLOGY: this is called PRE-CUE POSITION INFORMATION, not a maintained motor plan. The spout "
    "arrives ~3 s before the cue, so a sustained sensory response and a held intention are temporally "
    "coextensive and this design cannot separate them. It does not need to: a pre-cue position signal "
    "that changes post-stroke is the readout either way. "
    "See docs/PREPROCESSING_DECISION.md, DECISIONS.md and wfield_local/filter_acausality_test.py.")

M_JOINT = (
    "CROSS-SESSION decoder/encoder in the SHARED JOINT-LocaNMF basis (wfield_local.joint_xsession). "
    "Same leave-one-SESSION-out design as the frozen ROI slides -- no trial from the plotted day was "
    "used to fit -- but the features are ~95-137 functionally-defined LocaNMF components instead of 66 "
    "anatomical Allen ROIs. "
    "WHY THIS IS POSSIBLE AT ALL: a session's OWN LocaNMF components are session-specific in count AND "
    "identity, so they cannot be pooled across days, which is why the frozen work started with ROI "
    "features. The joint basis (wfield_local.joint_locanmf) fits the footprints A ONCE over the "
    "animal's curated sessions and then holds them FIXED; a day not in that fit is PROJECTED onto the "
    "same footprints (C = pinv(A) U, contracted on the small Gram matrix), never refitted. Component j "
    "is therefore the same footprint on every day. The basis is SEEDED and PERSISTED with an id hashing "
    "its session set, inputs, rank and params, so a refit lands in a new directory and results can "
    "never silently mix two bases -- necessary because LocaNMF is stochastic (repeat runs differed by "
    "up to 5 components and moved RSA by 0.054). "
    "WHY TWO BASES: ROI and joint are not distinguishable on the RSA criterion (+0.817 vs +0.806) but "
    "they are different parcellations, so a cross-day effect that appears in only one is a fact about "
    "the parcellation. LocaNMF decodes better WITHIN a session in 4/4 animals (0.824 vs 0.763), so it "
    "is the more sensitive of the two, and the ROI version is the more conservative. "
    "NOT the rejected frozen fixed-A path: that nominated ONE session as the reference and the choice "
    "mattered (no reference won for every animal; within-animal swing up to 0.36). The joint basis is "
    "reference-free. "
    "⚠ variance_captured IS NOT A SUFFICIENT HEALTH CHECK -- corrected 2026-08-13, having claimed "
    "otherwise earlier the same day. 8/12 is the only PROJECTED day per animal, and ROI (which has no "
    "in-fit/projected distinction) adjudicates whether its joint-basis drop is the projection or the "
    "day. It is the PROJECTION: in ROI, 8/12 is an ordinary session and BETTER than its siblings in 3 "
    "of 4 animals (delta +0.059/+0.051/-0.038/+0.005, mean +0.019), while in the joint basis the same "
    "day costs a mean -0.079 (PS92 -0.172, PS95 -0.141) -- and variance_captured reads 98.9-99.4% "
    "throughout. It shows green while PS92 loses 0.172. It measures whether a session's total ENERGY "
    "lies in the frozen subspace, NOT whether the position-DISCRIMINATIVE directions survive; the "
    "discriminative signal is a tiny fraction of the variance, so it can be mangled while 99% of the "
    "energy is reproduced. Read the basis-health slide as necessary-but-not-sufficient. "
    "CONSEQUENCE: every POST-STROKE session will be a projected day, so a projection cost of ~0.08 (up "
    "to 0.17) is present BEFORE any lesion effect and this diagnostic will not reveal it. Either "
    "calibrate it on held-out pre-stroke days (refit the basis without day k, project day k, record the "
    "drop) or keep Allen-ROI -- which involves no projection -- as the primary readout. See DECISIONS.md. "
    "Note also that the basis was fitted using the in-fit days' data (unsupervised -- no labels), so "
    "their LOSO scores carry a mild transductive advantage that projected days do not; compare like "
    "with like. " + M_COMMON)

M_ENCODE = ("Encoder (forward model): cross-validated ridge regression (alpha=1) from a one-hot position "
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
    # curated_dates() is DERIVED (registered minus excluded), so a hand-run deck covers the same
    # dates as the nightly. The static policy list this used to read stopped at 8/7.
    dates = dates or config.curated_dates()
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
              "A–C within-day, grouped animal → analysis type → date.  D cross-session (frozen), "
              "grouped basis → alignment → animal.  E–F cohort summaries."]:
        rr = tf.add_paragraph().add_run()
        rr.text = t
        rr.font.size = Pt(15)
        rr.font.color.rgb = GREY
    note(s, "Refined analysis deck: spout-position decode/encode/RSA, grouped animal -> analysis type -> "
            "date, curated pre-stroke sessions only. Each slide's speaker notes give how that figure is "
            "made. " + M_COMMON)

    # ---------------- METHODOLOGY: right after the title, before any result ----------------
    # This slide used to warn that the pre-cue numbers were inflated ~2x. They are not any more: the
    # deck is built on the corrected variant. A stale warning is worse than none -- it tells a reader
    # to discount numbers that are now right.
    s = slide()
    title(s, "Drift removal — the pre-cue numbers in this deck are CORRECTED",
          "Built on meegkit_hpfit, not the pipeline default. Read this before comparing with anything "
          "produced before 14 Aug 2026.")
    tf = s.shapes.add_textbox(Inches(0.6), Inches(1.7), Inches(12.1), Inches(5.3)).text_frame
    tf.word_wrap = True
    for i, line in enumerate([
        "WHAT WAS WRONG: wfield.hemodynamic_correction high-passes both channels at 0.1 Hz with scipy "
        "filtfilt — zero-phase, therefore ACAUSAL — and that high-passed 470 channel becomes SVTcorr. "
        "Its impulse response is symmetric in time (−0.496 before an impulse, −0.496 after), so a "
        "position-specific POST-cue response cast a sign-flipped shadow BACKWARDS into the pre-cue "
        "window. A linear decoder does not care about sign, so the shadow read as pre-cue information.",
        "THE FIX (adopted 2026-08-14): keep the 0.1 Hz high-pass for the hemodynamic COEFFICIENT fit — "
        "that is what it is for — and replace it for the OUTPUT with de Cheveigné robust polynomial "
        "detrending (order 10, 600 s) on a mask that excludes whole trials.",
        "MEASURED over ALL 36 CURATED SESSIONS:",
        "                        pre-cue        post-cue (control)",
        "        zerophase (old)       0.486          0.684",
        "        meegkit_hpfit (now)   0.352          0.759      post-cue IMPROVED",
        "The variant that most IMPROVES the readout we trust also most REDUCES the one we suspected — "
        "the strongest form this comparison could take.",
        "WHAT SURVIVES: pre-cue position information is REAL and significant in 35/36 sessions, at "
        "~72% of the previously reported size. PS92 0.225, PS93 0.349, PS94 0.500, PS95 0.334 "
        "(chance 0.167; empirical null 0.137–0.147 by block-label permutation). PS94 was essentially "
        "untouched; PS92 was the one substantially inflated and is now well above chance, not at it.",
        "SIGN TEST: the pre-cue pattern used to be ANTI-correlated with the post-cue pattern (negative "
        "in 30 of 36 sessions; on the worst days the pre-cue MAP was literally the negative of the "
        "post-cue map, r = −0.93). After correction that signature is gone — negative in 2 of 36.",
        "NOT A LOCAL BUG: churchlandlab/WidefieldImager SvdHemoCorrect.m does the same in-place "
        "filtfilt; Musall et al. 2019 state it in their methods. The artifact class is published — "
        "van Driel, Olivers & Fahrenfort 2021, J Neurosci Methods — including the negative sign, with "
        "trial-masked robust detrending as the recommended fix, which is what was adopted.",
        "REPRODUCE: python -m wfield_local.filter_acausality_test <LABEL,...>   •   see "
        "docs/PREPROCESSING_DECISION.md and DECISIONS.md",
    ]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line
        r.font.size = Pt(12.5)
        r.font.color.rgb = NAVY if line.startswith(("WHAT", "THE FIX", "MEASURED", "SIGN", "NOT")) else GREY
    note(s, M_PRECUE_CAVEAT)

    # ---------------- A. per-animal WITHIN-DAY decoding ----------------
    divider("A. Per-animal WITHIN-DAY decoding across sessions",
            "Post-cue 2 s (predicts no-lick trials too = no lick generalization) and pre-cue 2 s "
            "(pre-cue position information) confusion + recall; then the rolling decoder across sessions. "
            "Cross-day (frozen) decoding is Section D.")
    for a in animals:
        s = slide()
        title(s, f"{a} — post-cue 2 s decoder (engaged, no-lick generalization)",
              "Per session: confusion matrix + per-position recall (engaged vs held-out no-lick trials).")
        note(s, M_DECODE)
        grid(s, [sess(f"{a}_{d}", "cue") for d, _ in date_labels], cols=3)
        s = slide()
        title(s, f"{a} — pre-cue 2 s decoder (pre-cue position information)",
              "Position decodable in the pre-cue ENL window, before movement. NB the accuracies shown "
              "are corrected (meegkit_hpfit); see slide 2 for the drift-removal decision.")
        note(s, M_DECODE + M_PRECUE_CAVEAT)
        grid(s, [sess(f"{a}_{d}", "precue") for d, _ in date_labels], cols=3)
        s = slide()
        title(s, f"{a} — rolling decoder across sessions (pre-cue ENL → post-cue)",
              "Sliding 0.5 s window, block-CV, one line per session. Above-chance in the ENL = position information present before the cue. "
              "(Per-animal accuracy across sessions is in the cross-mouse summary, Section C.)")
        note(s, M_DECODE)
        big(s, src / f"locanmf_decoder_rolling_by_animal_{a}.png", top=1.5, width=11.2)

    # ---------------- B. per-animal encoder ----------------
    divider("B. Per-animal WITHIN-DAY encoder — expected activity, predicted maps & explained variance",
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

    # ---------------- C. pre-cue without licking ----------------
    divider("C. Pre-cue code without licking — the motor-confound control",
            "Decode AND encode on a SEARCHED 2 s window: 2 consecutive lick-free seconds between the "
            "spout-position strobe and the cue, taken as late as possible.")
    for src_name in ("roi", "locanmf"):
        for a in animals:
            p = src / f"precue_lickfree_{a}_{src_name}.png"
            if not p.exists():
                continue
            s = slide()
            title(s, f"{a} — pre-cue position code with NO licking in the window ({src_name})"
                     "",
                  "Exposure, decode (lick-free vs all vs with-licks), lick-free confusion matrix, and "
                  "per-region encoding EV. The lick control itself is VALID — it just sits on top of "
                  "corrected pre-cue values (meegkit_hpfit); see slide 2.")
            note(s, M_LICKFREE + M_PRECUE_CAVEAT)
            big(s, p, top=1.5, width=12.9)

    # ---------------- D. cross-session (frozen) decoders & encoders, BOTH bases ----------------
    # Was interleaved into each animal's Section A, which mixed "does the code exist today" with "does
    # it transfer across days". Now its own section, and now in TWO bases: Allen-ROI (conservative,
    # atlas-anchored) and the shared joint-LocaNMF basis (finer, more sensitive). A cross-day claim
    # that survives both is about the cortex; one that appears in only one is about the parcellation.
    divider("D. CROSS-SESSION (frozen) decoders & encoders — two independent bases",
            "Every day predicted by a model trained ONLY on that animal's other days "
            "(leave-one-session-out). Allen-ROI (66 anatomical areas) and the shared joint-LocaNMF "
            "basis (~95–137 functional components, footprints frozen and shared across days).")
    # PAGINATED 4-per-slide (2x2). All curated dates on one slide at cols=2 gives 4+ rows, so each
    # panel gets ~1/4 of the slide height and the 6x6 confusion cells become unreadable. 2x2 doubles
    # the height per panel; the cost is one extra slide per animal.
    pages = [date_labels[i:i + 4] for i in range(0, len(date_labels), 4)]
    # BOTH alignments: post-cue (readout during/after the movement) and PRE-CUE (the maintained,
    # motor-independent code). The pre-cue one is the readout the stroke arm leans on, so whether IT
    # survives freezing across days is the more consequential question.
    ALIGNS = (("cue", "post-cue 2 s", "the readout during/after the movement"),
              ("precue", "PRE-CUE 2 s", "pre-cue position information — the window ENDING "
                                        "at the cue, before any movement"))
    BASES = (("roi", "Allen-ROI", M_FROZEN, M_FROZEN_ENC,
              "66 atlas-anchored anatomical areas — column j is the same cortical region every day"),
             ("joint", "joint-LocaNMF", M_JOINT, M_JOINT,
              "shared joint-basis components — footprints fitted once and FROZEN, new days projected "
              "onto them rather than refitted"))
    for bkey, bname, m_dec, m_enc, bdesc in BASES:
        if not any((src / f"locanmf_frozen_decoder_loso_{bkey}_{al}.png").exists()
                   for al, _, _ in ALIGNS):
            continue                      # basis not computed (e.g. no joint basis built yet)
        divider(f"D — {bname} basis", bdesc)
        if bkey == "joint":
            s = slide()
            title(s, "Joint-basis health — how much of each session the frozen footprints span",
                  "Sessions IN the fit are 1.0 by construction (hollow); a PROJECTED day (filled) is "
                  "not. Read a projected day's decode accuracy against its bar: low-and-low means the "
                  "basis under-describes that day, not that its representation changed.")
            note(s, M_JOINT)
            big(s, src / "joint_basis_health_precue.png", top=1.7, width=12.2)
        for al, al_name, al_desc in ALIGNS:
            # the pre-cue arm inherits the zero-phase-filter inflation; the post-cue arm does not
            cav = M_PRECUE_CAVEAT if al == "precue" else ""
            warn = ""
            for a in animals:
                for page in pages:
                    span = f"{page[0][1]}–{page[-1][1]}" if len(page) > 1 else page[0][1]
                    suffix = f"  ({span})" if len(pages) > 1 else ""
                    s = slide()
                    title(s, f"{a} — FROZEN cross-day decoder, {al_name}, held-out day "
                             f"({bname}){suffix}{warn}",
                          f"Per date: confusion + per-position recall from a decoder trained on this "
                          f"animal's OTHER days only. {al_desc}.")
                    note(s, m_dec + cav)
                    grid(s, [src / f"locanmf_frozen_session_{a}_{d}_{bkey}_{al}.png" for d, _ in page],
                         cols=2, top=1.35)
            s = slide()
            title(s, f"FROZEN decoder ({al_name}, {bname}): transfer cost & OOD control — all "
                     f"animals{warn}",
                  "Held-out day vs same-day ceiling per session; the cost of freezing across days; and "
                  "the OOD control — a softmax decoder never abstains, so confidence alone is not "
                  "evidence.")
            note(s, m_dec + cav)
            big(s, src / f"locanmf_frozen_decoder_loso_{bkey}_{al}.png", top=1.9, width=12.7)
            s = slide()
            title(s, f"FROZEN cross-day ENCODER ({al_name}, {bname}): position → activity — all "
                     f"animals{warn}",
                  "Held-out-day EV against that day's own noise ceiling, and the ceiling-normalised "
                  "FEVE. The forward model for post-stroke residuals — note its transfer cost is "
                  "NEGATIVE where the decoder's is positive.")
            note(s, m_enc + cav)
            big(s, src / f"locanmf_frozen_encoder_loso_{bkey}_{al}.png", top=1.9, width=12.7)

    # ---------------- D2. no-detected-lick reference ----------------
    # The pre-stroke reference for reading POST-stroke failed trials. Placed immediately after the
    # frozen decoder because it uses the same frozen model and answers the question that motivates
    # freezing one at all.
    if (src / "nolick_reference_locanmf.png").exists():
        divider("D2 — Trials with NO DETECTED LICK",
                "The pre-stroke reference for post-stroke failures. A failed trial can mean the plan "
                "was never formed or that it was formed and the movement failed; those are different "
                "injuries and identical in the behaviour log.")
        s = slide()
        title(s, "No-detected-lick: does the position code survive without a movement?",
              "Balanced accuracy (macro-recall) per arm, pre-cue beside post-cue. The BLACK RULE on "
              "each bar is that arm's OWN permutation null, not a shared 1/6 — the nulls differ per "
              "arm and a single chance line would misrepresent every bar but the engaged one.")
        note(s, M_NOLICK)
        big(s, src / "nolick_reference_locanmf.png", top=1.9, width=12.7)
        s = slide()
        title(s, "No-detected-lick: PRE-cue surviving while POST-cue collapses = plan formed, "
                 "movement failed",
              "The discriminating quantity. Post-cue decoding is largely driven by the lick itself, "
              "so it should collapse without one; pre-cue reflects a maintained code that need not.")
        note(s, M_NOLICK)
        big(s, src / "nolick_survival_locanmf.png", top=1.9, width=10.5)

    # ---------------- E. cross-session summary ----------------
    divider("E. Cross-session summary — decoder recall & encoder accuracy across sessions")
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

    # ---------------- F. RSA ----------------
    divider("F. RSA — representational geometry of spout position",
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

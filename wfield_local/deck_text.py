"""THE DECK'S PROSE — trial-population lines, per-slide reading notes, methodology blurbs.

Split out of `locanmf_analysis_deck` on 2026-09-21. This module is DATA: every name here is a
string that ends up on a slide or in its speaker notes, and nothing in it computes anything. It
was 1,155 of the deck module's 5,484 lines, sitting between the imports and the machinery, and
reading the builder meant scrolling past all of it.

WHAT EACH FAMILY IS, and the distinction is load-bearing:

  ``TRIALS_*``  the TRIAL POPULATION line under a slide title. "Engaged" means two different
                things in this deck -- a detected lick within `decode.max_rt_s` for the
                decode/encode/frozen/RSA families, the ENGAGEMENT GATE for sections G/H/I -- so
                every slide states its own. A reader comparing a frozen-decoder curve against a
                section-G contrast is comparing two trial sets, not two results.
  ``S_*``       the per-slide READING NOTE: what to look for, the trap specific to THAT panel, and
                where a contrary result would leave the claim. Drawn from DECISIONS.md rather than
                invented.
  ``M_*``       the METHODOLOGY blurb: how the figure was MADE. Deduped at write time by hashing
                the whole text, so two blurbs that differ only in their resolved numbers stay
                distinct -- see `_write_note`.

THE ``+=`` REBINDINGS AT THE FOOT OF THIS FILE ARE NOT TIDY-UP-ABLE. `M_GATE` carries the
engaged-cut warning and is appended to an EXPLICIT list of blurbs, not to all of them, because
M_HEMI / M_VESSEL / M_HEMIDYN / M_FIXEDSCALE read RAW fluorescence and never split on a lick --
warning there would name a dependency they do not have. Likewise `_M_LICK_UNIT` is prepended to
some and not others. Folding either into the literals would lose the record of which blurbs the
warning applies to and why.

`scripts/deck_claim_audit.py` reads THIS FILE as well as the builder; it is where most of the
deck's several hundred measured numbers live.
"""

# ---- per-slide READING NOTES: what to look for, and what would falsify it ----
# The title says what the figure IS and the M_* block says how it was MADE. Neither says what to do
# with it. These are the third thing, and they are drawn from DECISIONS.md rather than invented --
# each states the reading, the trap specific to THIS panel, and where a contrary result would leave
# the claim. Written 2026-08-23 after Priya pointed out that the repo already documents all of it.
# ---------------------------------------------------------------- TRIAL POPULATIONS
#
# "ENGAGED" MEANS TWO DIFFERENT THINGS IN THIS DECK and the slides never said which. The
# decode/encode/frozen/RSA families split trials by whether a LICK WAS DETECTED within the response
# window (`decode.max_rt_s`, 3.5 s) and treat the no-lick trials as one undifferentiated
# generalization arm. Sections G/H/I instead split the no-lick trials into miss-while-working and
# stopped using the ENGAGEMENT GATE. Those are different populations, and a reader comparing a
# frozen-decoder curve against a section-G contrast is comparing two trial sets, not two results.
#
# So each slide now states its own. Priya, 2026-09-07.
TRIALS_LICK = (
    "TRIALS: engaged = a lick detected within the response window (max_rt 3.5 s); no-lick trials "
    "are held out as the generalization arm and are NOT subdivided. The engagement gate is not "
    "used here.")
TRIALS_NOLICK = (
    "TRIALS: no-lick only -- trials with NO detected lick in the response window. Absence of a "
    "detection, not a judgement about engagement.")
TRIALS_WORKING = (
    "TRIALS: lick + miss-while-working. The no-lick trials are split by the ENGAGEMENT GATE "
    "(judged at the reference positions close_L/close_center, backdated to the first miss of the "
    "run that trips it); 'stopped' trials are analysed as their own class, not discarded.")
TRIALS_BEHAVIOUR = (
    "TRIALS: all scored trials, with disengaged trials REMOVED from the denominator by the "
    "engagement gate (reference positions only, backdated to the first miss of the run).")


S_G1B = (
    "READ THE DENOMINATOR FIRST. A position with zero engaged trials has no lick-only decoding "
    "number at all -- not a low one. PS94 8/20 has ZERO engaged far_R and 17 far_center against ~70 "
    "elsewhere; PS93 is at 8-10 far_R; PS92 reached 4 by 8/21. FALSIFIER: if the no-lick bars were "
    "also near zero, the position stopped being PRESENTED and this is a task artefact. They are not "
    "-- the spout still moves there every trial -- so this is the animal declining, which is the "
    "phenotype the rest of section G is trying to explain.")

S_G2 = (
    "The BAND is the animal's own pre-stroke leave-one-session-out range, so a point inside it means "
    "'no worse than this animal's ordinary day-to-day variation', not 'good'. ALL-trials arm: chance "
    "is 1/6 on every panel and the panels ARE comparable. Lick-only arm: chance is 1/n for that "
    "session's preserved positions, so the panels are NOT comparable with each other -- a higher "
    "number on a four-position session can be worse performance than a lower one on six.")

S_G2C = (
    "THE CENTRAL CLAIM OF THIS SECTION. Pre-cue survives, post-cue collapses without a lick. If it "
    "reversed -- post-cue intact and pre-cue gone -- the readout would be a sensory-response deficit "
    "rather than an execution failure, and the frozen-decoder framing would not hold.")

S_G4 = (
    "This asks whether a no-lick trial carries the pattern of the position that was PRESENTED. "
    "CAUTION documented at DECISIONS 2026-08-17: 'no detected lick' is not 'no attempt' -- the "
    "sensor needs contact, so a short or weak lick registers as nothing. PS93 has a pre-existing "
    "rightward tongue bias and reaches far_L poorly PRE-stroke, which makes PS93 far_L a "
    "within-subject instance of the phenotype owing nothing to the lesion.")

S_G5 = (
    "Same code at lower gain, or a different code? A uniformly shrunken confusion matrix means "
    "gain; off-diagonal mass moving to a NEW position means remapping. These have different "
    "implications for recovery and the figure is the only thing here that separates them.")

S_G7 = (
    "PS92/PS93 8/17 follow the 8/16 laser that did NOT take, so they belong to neither phase and are "
    "excluded from every other comparison. They are here because they are the within-animal control: "
    "the same two animals, the same rig, a lesion attempt with no overt deficit. If the post-stroke "
    "effects appeared here too, they would be attributable to the procedure rather than the lesion.")

S_G8E = (
    "Raw fluorescence, no lick split -- so this is independent of every behavioural gate elsewhere in "
    "the section. That independence is the point: if the decoding results were an artefact of which "
    "trials survive the engaged cut, this panel would not show anything.")

S_G9 = (
    "Coding DIRECTIONS, not accuracies: the question is whether the axis separating two positions "
    "still points the same way, regardless of how well it decodes. PAIRWISE is the sharper "
    "instrument -- each contrast is A vs B alone. Within-ring comparisons are safe; cross-ring ones "
    "confound distance with side (DECISIONS 2026-08-21).")


S_DRIFT = (
    "THE ONE THING TO CARRY OUT OF THIS SLIDE: every pre-cue number in this deck is ~72% of what a "
    "pre-14-Aug-2026 figure would have shown, and that is the CORRECTED value. Do not reconcile "
    "against older slides -- they are the artefact. FALSIFIER APPLIED: the acausal filter's shadow "
    "predicts pre-cue ANTI-correlated with post-cue; that was true in 30/36 sessions before the fix "
    "and 2/36 after. What survives is real -- significant in 35/36 sessions, PS92 0.225 to PS94 "
    "0.500, against an empirical null of 0.137-0.147 by block-label permutation, NOT 1/6.")

S_DEC_CUE = (
    "WITHIN-DAY, so nothing here can be compared across days -- Section D is the cross-day arm. "
    "The no-lick column is a GENERALIZATION test, not a second dataset: the decoder is fit on "
    "engaged trials only and the no-lick trials are held out entirely. Post-cue is the window that "
    "is largely movement-driven, so a no-lick number well below the engaged one is expected here, "
    "and it is the contrast that makes the PRE-cue slide interesting (DECISIONS 2026-08-17).")

S_DEC_PRECUE = (
    "'PRE-cue' means BEFORE THE CUE, not before the spout: the spout arrives ~3 s earlier, so a "
    "sustained sensory response and a held plan are temporally coextensive here and this design "
    "cannot separate them (DECISIONS 2026-08-13). Read it as pre-cue position INFORMATION. The "
    "accuracies are drift-corrected, so compare only with other post-14-Aug figures. Pre-cue "
    "survives without a lick far better than post-cue does -- that dissociation is the point.")

S_DEC_LICK = (
    "ALIGNED TO THE MOVEMENT, not the cue -- the arm that isolates EXECUTION from the plan. Trials "
    "are registered on the first lick inside the response window, so RT jitter no longer smears a "
    "movement-locked signal the way it does on the cue-aligned slide. THE UNIT IS ONE REFERENCE PER "
    "TRIAL, so an animal that licks many times contributes once. WHAT IT CANNOT TELL YOU: a trial "
    "with no detected lick has no reference at all and is simply absent here -- which post-stroke is "
    "exactly the population of interest, so read this beside G1b's denominators rather than alone. "
    "Shown from 2026-08-26; the figures were being generated nightly before that and displayed "
    "nowhere.")

S_DEC_ROLL = (
    "READ THE SHAPE, NOT THE PEAK. The question is whether the trace is already above chance BEFORE "
    "the cue and where it rises, not how high it gets -- height is set by trial count and window "
    "width. Each line is one session, block-CV'd within that session, so the lines are comparable "
    "in shape while the absolute level stays a within-day quantity. A rise that begins only AT the "
    "cue would mean no pre-cue information; an already-elevated ENL level is the claim.")

S_ENC_POS = (
    "ENCODER, so the direction is position -> activity: this asks how much of each feature's "
    "variance the six position means explain, not whether a boundary can be drawn. RAW EV alone is "
    "uninterpretable across animals -- PS93's ceiling is the lowest of the four (frozen FEVE 0.09 "
    "against 0.51-0.70), so ALWAYS read its EV beside its ceiling (DECISIONS 2026-08-11/12). The "
    "vs-ceiling panel is the one that compares animals; the raw panel compares sessions within an "
    "animal.")

#: UNUSED SINCE 2026-09-13 and kept deliberately: the pooled FEVE-by-region slide it annotated was
#: cut, and this is its method note. Deleting it would mean rewriting the method from scratch to put
#: the slide back, which is the one thing a cut should never cost. Kept, not orphaned by accident.
S_ENC_FEVE = (
    "POOLED per animal, so this is the anatomy question: WHERE position is encoded, not how much. "
    "Regions with a near-zero ceiling can show wild FEVE for free, so rank the regions and ignore "
    "the tail. A negative value is not a paradox -- ridge on held-out data can do worse than the "
    "mean, and several of PS93's frozen EVs are genuinely negative, which is its low ceiling "
    "showing through rather than a model failure.")

S_ENC_MATRIX = (
    "ONE COMMON SCALE across all animals and sessions -- that is the whole reason this slide exists, "
    "and it is what the per-animal EV slides cannot give you. So compare CELLS here, not panels. "
    "A blank cell is a position with no trials, which is NOT a zero: it has no estimate at all (see "
    "G1b for the denominators). Empty positions were silently contributing nan to the pooled "
    "statistic until 2026-08-22; those cells now declare themselves instead.")

S_LICKFREE = (
    "The cleanest form of the pre-cue claim: no lick was detected anywhere in the window, so the "
    "code cannot be a movement echo. CAVEAT THAT LIMITS IT: 'no detected lick' is not 'no attempt' "
    "-- the sensor needs contact, so a short or misaimed lick registers as nothing, and PS93's "
    "pre-existing rightward tongue bias makes its far_L a within-subject instance of exactly that "
    "(Priya, 2026-08-17). Resolving it needs DLC / facial tracking, which is not in this deck.")

S_JOINT = (
    "DIAGNOSTIC, NOT A RESULT -- but a load-bearing one: every cross-day LocaNMF number downstream "
    "assumes the frozen footprints still span the session they are applied to. In-fit sessions run "
    "99.4-99.7%, so that is the reference. WHAT IT CANNOT CATCH: a PS92 basis spans a PS95 session "
    "at 97%, because both are cortex on the same Allen grid -- only the label catches the wrong "
    "animal, and project() refuses one outright (DECISIONS 2026-08-12).")

S_FROZEN_SESS = (
    "TRAINED ON PRE-STROKE SESSIONS ONLY. A held-out PRE-stroke day is scored leave-one-out among "
    "pre-stroke days; a POST-stroke day is scored by one model fitted on ALL pre-stroke days and "
    "applied unchanged. Until 2026-08-26 training used every pooled session, so post-stroke nights "
    "were ~30% of the training data behind a number whose whole job is to be a lesion-free baseline "
    "-- every frozen figure before that date is superseded. Pooling for FEATURE ALIGNMENT still uses "
    "all sessions, which is what makes the post-stroke rows comparable at all. The reference holds: "
    "pre-stroke transfer cost is POSITIVE in every animal, so the frozen model BEATS the same-day "
    "model by pooling ~3000 trials against ~500, and a post-stroke drop reads as lesion rather than "
    "day gap. Correcting the contamination SHRANK the cost by 11-33% without changing that sign. "
    "ROI features, because LocaNMF component identity is not stable across days.")

S_FROZEN_ALL = (
    "TWO REFERENCES, BOTH MANDATORY. A softmax decoder never abstains: on quiet / running windows "
    "where no position is even defined it emits normalized entropy 0.24-0.54 and max-probability up "
    "to 0.997 -- MORE confident than the shuffled-label floor -- and collapses onto a single "
    "attractor position. So post-stroke confidence is not evidence of preserved coding. Read every "
    "value against the shuffled-label floor AND the no-lick trials, never on its own.")

S_FROZEN_ENC = (
    "THE ENCODER DOES NOT TRANSFER LIKE THE DECODER, and the disagreement in SIGN is the finding: "
    "decision boundaries transfer across days (cost positive in all four animals) while activity "
    "MAGNITUDES do not (PS93 and PS95 NEGATIVE; -0.063 and -0.032 on the clean 8/11 pool, -0.026 and "
    "-0.021 after the 2026-08-26 training fix -- the sign, which is the finding, survives). Like the "
    "decoder, this is now trained on PRE-STROKE sessions only; before 2026-08-26 it was fitted "
    "partly to post-stroke trials, which is worse here than for the decoder because the encoder's "
    "RESIDUAL on those trials IS the representational-change readout. The encoder estimates only 6 "
    "position means per "
    "feature, so it gains little from extra trials and is actively hurt by day-to-day differences "
    "in the mapping that per-session z-scoring does not remove. CONSEQUENCE: judge post-stroke "
    "encoder residuals against this NON-ZERO pre-stroke cross-day cost, not against zero. Read the "
    "intention with the DECODER (DECISIONS 2026-08-11/12).")

S_NOLICK_A = (
    "HEADLINE IS BALANCED ACCURACY, whose null expectation is exactly 1/6 however skewed either side "
    "is. Raw accuracy is shown against a permutation null computed on THESE trials with predictions "
    "held fixed. Both are needed because these trials are heavily skewed by construction -- PS93's "
    "are 49% far_center -- and a constant 'always guess far_center' scores 0.490, beating the real "
    "decoder's 0.293 outright. Reading any of this against a uniform 1/6 manufactures a result, and "
    "it did so in BOTH directions before this was fixed (DECISIONS 2026-08-17).")

S_NOLICK_B = (
    "THE DISSOCIATION THIS SECTION RESTS ON, and it is not 'engagement gates the code': the "
    "POST-cue code is largely movement-driven and collapses without a lick, while the PRE-cue code "
    "substantially survives. PS93 pooled over 11 sessions, ROI and joint agreeing: post-cue "
    "survival ratio 0.357 / 0.422, with pre-cue far higher. An animal can know where the spout is "
    "and still not lick (Priya, 2026-08-17). The ENCODER half is separate and stands: EV on these "
    "trials is ~0 once the baseline offset is removed.")

S_NOLICK_C = (
    "AGREEMENT BETWEEN BASES IS THE POINT, not which basis wins. ROI is atlas-anchored and poolable "
    "across days; joint LocaNMF retains ~98% of the pixel map against ROI's 64.5%. They answer the "
    "same question through different features, so a result present in ONE of them is a basis "
    "artefact and must not be reported. Both bases agreeing is what licensed the no-lick "
    "conclusion.")

S_XMOUSE = (
    "PRE-STROKE SESSIONS ONLY (since 2026-08-26) -- this is a BASELINE question, do these mice differ "
    "from each other, so it is answered from baseline data. It previously pooled every session, 39% "
    "of them post-stroke, and three of its six metrics are LATERALISATION measures while PS94's "
    "lateralisation is exactly what collapses after its lesion (DECISIONS 2026-08-19): a "
    "'between-mouse difference' could have been a between-lesion-severity one. ANIMAL IS THE UNIT, "
    "so n=4 and no p-value here carries weight. HYPOTHESIS-GENERATING, around one prediction: PS93's "
    "right orofacial deficit is represented contralaterally, predicting altered LEFT hemisphere "
    "representation and/or worse RIGHT-spout decoding. Read the L-vs-R spout and SSp-left-vs-right "
    "panels against that; the rest is description. The post-stroke version of these asymmetry "
    "questions belongs in section G, which has the position-matching this slide does not.")

S_XCONSIST = (
    "CONSISTENCY, NOT ACCURACY: whether an animal's per-position profile keeps the same SHAPE across "
    "its sessions. A low-accuracy animal can be highly consistent and a high-accuracy one erratic, "
    "and it is consistency -- not accuracy -- that licenses pooling sessions within an animal. "
    "PRE-STROKE ONLY (since 2026-08-26), and here that is load-bearing rather than tidiness: this "
    "slide IS the noise floor a post-stroke change has to exceed, and a floor built partly from "
    "post-stroke sessions is inflated by the very change it exists to be exceeded by. That error "
    "runs in the conservative direction -- it makes a real effect harder to clear -- but the "
    "reference was still measuring the wrong thing. Read this before trusting any pooled "
    "per-position number elsewhere in the deck.")

S_RSA_A = (
    "RELIABILITY IS NOT INFORMATION. Mean sibling RSA measures how reproducible an RDM is, and ROI "
    "wins it in 4/4 animals -- yet LocaNMF DECODES better in 4/4 (+0.061 mean). The dissociation "
    "tracks exactly one thing: whether the quantity requires estimating a covariance in feature "
    "space. Crossnobis does, and ROI's 66 well-conditioned features beat LocaNMF's 151 "
    "rank-deficient ones. That is an estimability cost, not an information deficit (DECISIONS "
    "2026-08-12).")

S_RSA_B = (
    "The 6x6 geometry itself. PRE-CUE AND POST-CUE GEOMETRY LARGELY AGREE: crossnobis +0.827 "
    "(LocaNMF) and +0.843 (ROI) pre-cue against post-cue, so the positional geometry is largely "
    "established BEFORE movement. Use the PRE-CUE pairs -- cue-to-lick (+0.93) is inflated by window "
    "overlap, since the lick usually falls inside the 2 s after the cue. Ignore any '% of "
    "reliability ceiling' column: it is invalid as computed (split-half on half the data, no "
    "Spearman-Brown correction, impossible >100% values).")

S_RSA_C = (
    "CROSSNOBIS IS THE NOISE-UNBIASED ONE -- cross-validated, so its expected value is 0 for two "
    "identical conditions, which 1-Pearson cannot promise. Prefer it wherever the covariance is "
    "estimable. SCOPE: these RDMs are built on Allen ROIs or LocaNMF components, both averages over "
    "anatomically defined sets, so the non-orthonormal-basis problem (DECISIONS 2026-08-19) does not "
    "touch them. That one applies to SVD COEFFICIENT-space RDMs, which is why none appear here.")

S_G0 = (
    "READ THIS BEFORE ANY OTHER G SLIDE. Two arms, two different chance levels: ALL-TRIALS is 1/6 on "
    "every panel and the panels ARE comparable; LICK-ONLY is 1/n for that session's attempted "
    "positions and the panels are NOT comparable with each other. The band on later slides is the "
    "animal's OWN pre-stroke leave-one-session-out range, so inside the band means 'within this "
    "animal's ordinary day-to-day variation', not 'normal'. And PS92/PS93 are the SMALL-LESION arm "
    "of a severity contrast -- NOT a negative control, because they were lesioned too, just mildly "
    "enough to leave no overt deficit (Priya, 2026-08-18).")

S_G1 = (
    "BEHAVIOUR FIRST, because most of Section G is only interpretable against it. The lesion marker "
    "is the axis every later comparison is anchored to, and the response-rate collapse is "
    "POSITION-SPECIFIC in the animals with overt deficits -- that specificity is what separates a "
    "lesion effect from a bad night. LASER POWER DOES NOT PREDICT MAGNITUDE: PS94 at the lowest dose "
    "(3 mW) has the largest post-cue deficit, PS93 at the highest (5.5 mW) one of the smallest. "
    "Behavioural severity tracks the neural effect; dose does not (DECISIONS 2026-08-19).")

S_G2B = (
    "PER-POSITION RECALL NEEDS ITS COLUMN BASELINE, which is printed under each column. The frozen "
    "decoder predicts far_R on ~35% of ALL PS94 post-stroke trials, so far_R 'recall' is inflated by "
    "prediction bias before any position information is involved. Under a label permutation the "
    "expected recall for a position is exactly its prediction rate -- that is the number a diagonal "
    "must clear. Read against 1/6 instead and this figure manufactures a result (Priya, 2026-08-18).")

S_G3 = (
    "The POST arm includes NO-LICK trials by design (Priya, 2026-08-18): PS94 has ZERO engaged "
    "trials at far_center and far_R, the two positions worth reading, so an engaged-only matrix "
    "leaves both rows blank. The PRE arm stays engaged-only because it is the reference for what the "
    "code looks like when the movement succeeds. THE QUESTION: does far_center get read as far_R -- "
    "the animal aiming right and undershooting -- or does the row simply disperse? Off-diagonal "
    "STRUCTURE is the claim; off-diagonal magnitude means nothing without the pre-stroke matrix "
    "beside it.")

S_G4B = (
    "DISTRIBUTION FIT, not accuracy -- a different question from G4 and a stricter one. G4 asks "
    "whether no-lick trials DECODE like pre-stroke engaged trials; this asks whether they lie in the "
    "same REGION of feature space at all. A trial can be classified correctly and still sit far "
    "outside the training distribution, which is exactly the failure mode the OOD control exists "
    "for: a softmax decoder is at its most confident where it has no business being.")

S_G6 = (
    "THE SPLIT IS PER SESSION, because 'still attempts this position' is a behavioural state that "
    "changes overnight -- PS95 far_R is 1/120 on 8/17 and 88/112 on 8/18. A trial is therefore "
    "labelled by what the animal was doing THAT night, not by an animal-level verdict. READ THE "
    "PRESERVED ARM AS THE CONTROL: if the decoder cannot read a position from a no-lick trial even "
    "where the animal is working that position, then 'no lick -> no code' is a property of no-lick "
    "trials in general and says nothing about the lesion. That arm is thin per session -- PS95 8/17 "
    "is n=38, p=0.11, which is UNDERPOWERED, not negative. G6b is the sharper form of this contrast.")

S_G7B = (
    "THE PREMISE OF THE WHOLE SMALL-LESION ARM: all six positions still attempted. If that fails, "
    "every G7 comparison collapses into the same denominator problem as G1b and the neural "
    "comparison stops being like-for-like -- so check this slide before reading G7c or G7d. These "
    "animals control for the DAY and the PROCEDURE (same rig, anaesthesia, handling, preprocessing, "
    "frozen decoder), and an 8/17 artefact would have hit all four and did not. They CANNOT show "
    "that a lesion is necessary for an effect.")

S_G7C = (
    "The like-for-like reading of G3. These animals attempt every position, so their post arm is not "
    "carried by no-lick trials the way PS94's is -- which makes this the cleanest available test of "
    "whether the confusion STRUCTURE in G3 is a lesion effect or a property of the frozen decoder on "
    "any post-stroke day. The same column-baseline rule as G2b applies to every diagonal here.")

S_G7D = (
    "IT CONTROLS FOR THE NO-LICK TRIAL CLASS, NOT FOR LESION SEVERITY. G4b's misfit has an "
    "alternative reading that owes nothing to the stroke: no-lick trials might fail that test ALWAYS "
    "-- different arousal, different movement, no first-lick reference. This is the identical test "
    "on the only sessions where an animal was recorded in the post-stroke era with NO EFFECTIVE "
    "LESION: PS92/PS93 on 8/17, after the 8/16 laser that did not take and before the effective 8/17 "
    "stroke. Same rig, same week, same pipeline, same frozen decoder, same trial class, minus the "
    "lesion. NOT the small-lesion arm -- that is these animals from 8/18 ONWARD. TWO LIMITS: it is "
    "PS93 alone in practice, since PS92 responded on essentially every trial and has too few no-lick "
    "trials to test; and it is one session, so read it as an existence check on the alternative "
    "explanation, not as an estimate.")

S_G9B = (
    "DIAGNOSTIC, NOT A RESULT -- it exists to make the G9 panels readable and carries no claim of "
    "its own. If a G9 panel looks surprising, come here first: most surprises turn out to be the "
    "linear projection being read as a probability, or a cross-ring contrast being read as a "
    "distance effect when it confounds distance with side.")

S_G9C = (
    "TWO CONTROLS, AND THEY SPLIT BY RING -- the actionable half is the last paragraph of the "
    "subtitle. The would-be-lick offset uses ONE session-median RT for every no-lick trial, so an "
    "animal that SLOWED through the session has its late trials placed progressively too early, "
    "which is the exact shape of a within-session decline. CLOSE positions are flat in every animal "
    "(drift <=0.03 s), so the control holds where it was used. IT DOES NOT HOLD for PS93 far_L: a "
    "session median near 0.6 s against a last-quartile 1.03 s misplaces those windows by ~0.4 s, a "
    "fifth of the window. Read PS93's far no-lick cells with that discount.")

S_G9E = (
    "EARLY + LATE IS THE LICK CLASS, CELL FOR CELL. The two post panels are a PARTITION of the "
    "post-stroke lick trials at 2.0 s, scored by the SAME frozen pre-stroke decoder, stored as raw "
    "counts -- so adding them back reproduces the poststroke_lick matrix every other section G "
    "slide shows, and any difference between the panels is a regrouping of the same trials rather "
    "than two differently-selected populations. Pinned by `tests/test_class_confusions.py`.\n\n"
    "WHY THE SPLIT EXISTS. decode.max_rt_s moved to 3.5 s on 2026-08-21 -- the task's real response "
    "window -- because the no-lick arm was holding rewarded hits (39.3% of it for PS92, 33.9% for "
    "PS93). That was the right fix, but it put a 0.2 s lick and a 3.0 s lick in one ENGAGED class, "
    "so the class every decode number is computed on could change composition after the stroke with "
    "no figure able to show it.\n\n"
    "MEASURED 2026-08-28, and it is NOT what this split was built expecting. The late arm is 1029 of 30645 post-stroke rewarded trials -- 3.4% -- and it is not evenly spread: PS92 5.6%, PS93 7.5%, PS94 1.0% (26 trials), PS95 0.8% (27 trials). The prediction written into this figure's own design was that post-stroke the mass would move late. It does not, at a 2.0 s cut.\n\nTHAT IS A RESULT, not a failed figure, and it points the same way as the rest of section G: the two most impaired animals do not lick SLOWLY, they lick fast or not at all. Slowed execution would have filled the late arm; the missing licks being the phenotype predicts exactly this. Read it beside the no-detected-lick arm, which is where those trials went.\n\nSO PS94 AND PS95's LATE PANELS CANNOT BE READ, and they say so in red on the figure itself: 26 trials over six positions is ~4 per row, and a per-position recall computed on 3 trials is 0.0 or 0.33 by arithmetic, not by measurement. Points below n=10 are hollow and carry their n; below n=3 no point is drawn at all. PS92 and PS93 carry enough to look at.\n\n"
    
    "WHAT THE PANELS DISCRIMINATE. Position coding PRESERVED on late trials = the plan is intact "
    "and execution is slow. DEGRADED on late trials = a different injury. Those are the two "
    "readings the study has to separate, and averaging them is what the merged class does.\n\n"
    "THE BOUNDARY IS FIXED AT 2.0 s, not the session's median RT. A session-relative cut would make "
    "'late' mean a different thing on every day, so it could be compared neither across sessions "
    "nor against the late_rewarded category the no-detected-lick reference already defines. 2.0 s "
    "is what nolick_decoder uses and what decode.max_rt_s was before 8/21.\n\n"
    "THE PRE PANEL IS LEAVE-ONE-SESSION-OUT and the post panels are not, which is not an "
    "inconsistency: post-stroke trials are held out by construction, pre-stroke trials are the "
    "training set. Scoring pre in-sample gives 0.89-0.99 against 0.45-0.66 held out, so an "
    "in-sample pre panel would read as a post-stroke collapse that is mostly overfitting.\n\n"
    "N IS PRINTED ON EVERY PANEL. The late arm is the smaller one by construction and in some "
    "animal-windows it is empty -- that panel then says 'no trials' rather than drawing a matrix of "
    "zeros, which would read as a decoder failure instead of an absence of data.")

S_G8 = (
    "THE 470 AND 415 QUESTIONS ARE ONE MEASUREMENT and cannot be asked separately -- 415 is the "
    "isosbestic channel and therefore the control for 470, so only the ratio of ratios is "
    "GCaMP-specific. WHAT WAS RETRACTED (2026-08-18, same day it was written): the argument that a "
    "415 change would have meant hypoperfusion. The raw violet trace RISES with activation in all "
    "four animals (+0.54% to +1.96%), which a simple absorption account does not predict. The null "
    "below stands; what is withdrawn is the interpretation. Perfusion DIRECTION is UNRESOLVED and "
    "needs an independent measure -- G8c is the one panel in this line that does not need the sign.")

S_G8B = (
    "DYNAMICS, so this is immune to the static baseline problem that limits G8: a slow L/R "
    "difference in raw counts cannot produce a task-locked difference in coupling. Read "
    "cross-hemisphere coupling as the summary quantity -- a lesion that disconnects should reduce "
    "it -- and treat per-hemisphere amplitude with the same caution as G8e, where a 'rise' turned "
    "out to be spatial BROADENING rather than a stronger response.")

S_G8C = (
    "THE ONE PERFUSION-DIRECT MEASURE IN THIS DECK. Vessels appear dark because haemoglobin absorbs, "
    "so their contrast against the surrounding cortex reads out how much blood sits in the light "
    "path: fainter vessels mean less absorption, i.e. less blood. Crucially it does NOT depend on "
    "knowing the sign of the evoked 415 response, which is what forced the G8 retraction. "
    "GAIN-INVARIANT by construction -- contrast is normalised by the hemisphere's own median -- so "
    "an exposure or LED change cannot manufacture an effect. Priya's observation, 2026-08-18: the "
    "vessels look fainter post-stroke.")

S_G8D = (
    "COMMON COLOUR SCALE, and that is the entire point: the per-session maps elsewhere set their "
    "limits from each session's OWN percentile. PS94 is +/-0.02425 on 8/14 and +/-0.08854 on 8/17, "
    "a factor of 3.65 -- on 8/17's range the whole of 8/14's negative range renders near white, so "
    "'the post-stroke map lost its blue' is the SCALE, not the biology (DECISIONS 2026-08-22). The "
    "amplitude rise is real at 2 s (peaks 0.019-0.052 pre against 0.059-0.083 post) and much smaller "
    "at 150 ms. A panel marked 'not attempted' has under 8 engaged trials; that is an absence of "
    "data, not a low value.")

S_G8F = (
    "MIDLINE TRANSFER IS A CLEAN NULL -- no transfer at any position, in any animal, at either "
    "alignment, on either arm, and it replicates in the joint basis with transfer at exactly ONE "
    "cell (PS94 8/18 pre-cue far_center). The 'left map moved right' reading is not supported. What "
    "is there instead is PATTERN LOSS, and cue-aligned it concentrates at exactly the positions the "
    "animals stop attempting -- so on that arm it is confounded with the missing movement and must "
    "not be read as a lesioned sensory map. CROSS-DAY PRE-CUE PATTERN COMPARISONS ARE WEAK BY "
    "NATURE: pre-stroke day-to-day reproducibility is 0.67-0.84 mean and as low as r=0.07 at worst "
    "(PS95), against 0.88-0.96 post-cue.")

S_GEXCL = (
    "THE HONEST LEDGER, and it earns its place. PS92/PS93 8/17 were nearly dropped as failed-lesion "
    "sessions, and keeping them analysable is the ONLY reason the within-animal before/after control "
    "exists -- nothing outside the band on 8/17, the dissociation present one day later, same animal "
    "and same rig. EXCLUDED IS NOT DELETED: every session listed here is on disk and re-analysable, "
    "and the reason is recorded per session. Check this slide before concluding a session is missing "
    "from a grid.")

# ---- methodology blurbs for the speaker NOTES (how each figure is made) ----
#: THE UNIT, stated on every note whose figure is lick-aligned.
#:
#: Every ANALYSIS figure aligned to the lick takes one reference per TRIAL -- the first lick inside
#: the response window, on engaged trials. Every PREPROCESSING deck post-lick map takes one per
#: LICK. Both are right for what they are, and neither said so, which made them look like they
#: contradicted each other: PS94 8/17 far_R reads n=83 on the preprocessing map and "not attempted"
#: on fixed_scale_maps, because those 83 licks belonged to fewer than 8 trials (Priya, 2026-08-23:
#: "but there ARE still first-lick-aligned maps, right?").
_M_LICK_UNIT = (
    "\n\nUNIT -- THE FIRST LICK OF EACH TRIAL. Everything lick-aligned on the ANALYSIS side takes "
    "ONE reference per TRIAL: that trial's FIRST lick, on ENGAGED trials (engaged = a lick within "
    "decode.max_rt_s of the cue -- NOT the task's timing.response_window, which is a different "
    "setting that happens to hold the same value). An n "
    "here therefore counts TRIALS. The PREPROCESSING deck's post-lick maps use the other "
    "convention, one reference per LICK (every lick inside a trial), so an n there counts LICKS and "
    "is several times larger. Both are right for what they are; the two must NOT be compared by n. "
    "PS94 8/17 far_R is n=83 on the preprocessing map and 'not attempted' on fixed_scale_maps, "
    "because those 83 licks belonged to fewer than 8 trials.")


_NL2 = chr(10) * 2

M_MISS_STOPPED = (
    "MISS-WHILE-WORKING vs STOPPED, per position, per session (wfield_local.miss_vs_stopped). "
    "Nothing is recomputed: coding_direction.json already holds every value per position, per "
    "session, per class; this draws the one contrast the other figures do not put side by side."
    + _NL2 +
    "WHY THE SPLIT IS THE ANALYSIS. The two post-stroke failure modes are different phenomena. "
    "MISS WHILE WORKING = still working the task, fails to lick at THIS position; position-specific, "
    "34-44% of these trials are far_R. STOPPED = quit for the day, licks nowhere; verified "
    "position-GENERAL (response ~0 at every position, close included). They differ in position "
    "composition by a total variation of 0.31-0.65, and ENL activity CARRIES position -- so a "
    "no-lick analysis that pools them compares the spout, not the state. That is what produced a "
    "spurious PS95 effect on the first pass."
    + _NL2 +
    "HOW TO READ IT. 1.0 (green dotted) = that position's own PRE-STROKE pole; 0 = no position "
    "code. Miss clearly above zero with stopped AT zero, in the same animal and position, is the "
    "plan-intact / execution-failed signature -- the code is there when the animal is trying and "
    "gone when it has quit. Hollow unjoined points are n<20: a working animal rarely misses at a "
    "position it can still reach, so close-position cells are structurally thin and must not be "
    "read (PS95 close_center reaches +7.16 on n=6)."
    + _NL2 +
    "WHAT IT SHOWS (2026-08-23). The pattern holds at the impaired-but-attempted positions in three "
    "animals and is ABSENT at far_L, which is what makes it evidence rather than a trend: PS92 "
    "far_center miss +1.0 to +2.0 across all five sessions against stopped near +0.8; PS93 "
    "far_center miss consistently positive against stopped flat at zero; PS94 far_R miss ~+0.6 on "
    "four of five sessions at >=2 SEM, against stopped +0.15/+0.07/-0.02. PS92 shows NOTHING at "
    "far_R (~0) and its strongest effect at far_center, which fits the documented severity ordering "
    "-- far_R is far enough gone that there is no code left to find."
    + _NL2 +
    "PS95 CANNOT ANSWER THIS, structurally: it recovered, and a working animal generates few misses "
    "(n 119 -> 24 -> 20 -> 4). Its STOPPED values also EXCEED its miss values, inverting the "
    "pattern. See DECISIONS 2026-08-23 for the open question that follows -- whether stopped-trial "
    "coding distinguishes a motivational quit from a representational collapse, and whether that "
    "predicts recovery. PS95 vs PS94 fits; PS93 is a counterexample; the causal direction is "
    "unresolved." + _M_LICK_UNIT)

M_COMMON = ("Features = individual LocaNMF component activities (atlas-anchored NMF, r2=0.95, "
            "loc_thresh=80, maxrank=20). Spout position per trial from the DAQ spout-strobe bits; when the "
            "DAQ is short a bit (Aug-2026 dead bit1) it is repaired from the behavior-log pos_idx via "
            "classify_cues_with_backup (only when it validates >=0.9 on the DAQ's good positions). "
            "ENGAGED = cue trials whose FIRST DETECTED LICK falls within decode.max_rt_s = 3.5 s of the "
            "cue. That is the DECODER's cut and is NOT the task's response window, which has run 3500 ms "
            "throughout (read per session from that session's gui_config.json): a lick at 2.5 s is a "
            "REWARDED HIT that this cut calls a no-lick trial. Across the curated set that is 2.15% of "
            "all hits -- PS93 5.1%, worst session 10.1% -- and section D2 reports BOTH cuts. Trials with "
            "no detected lick are kept separately as the generalization / OOD arm rather than discarded. NB the DAQ cue stream is "
            "NOT a rewarded subset -- an earlier note here said it was, corrected 2026-08-09: DAQ cue "
            "count equals the behavior log's scored-trial count exactly in every session and includes "
            "unrewarded trials, so unrewarded trials remain available for the post-stroke failed-attempt "
            "analysis. Curated pre-stroke sessions only (6/6-6/8 + 8/6 onward). HEMODYNAMIC/DRIFT REMOVAL (adopted 2026-08-14, docs/PREPROCESSING_DECISION.md): every panel in this deck -- decoders, encoders, RSA and the activity MAPS -- is built on the meegkit_hpfit SVTcorr, not the pipeline default. The default removes drift with a ZERO-PHASE 0.1 Hz filter, which is acausal: it smears each post-cue response BACKWARDS and inflated pre-cue decoding by ~0.21 across 36 sessions. meegkit_hpfit keeps that high-pass for the hemodynamic COEFFICIENT fit (which is what it is for) and replaces it for the OUTPUT with de Cheveigne robust polynomial detrending (order 10, 600 s) on a mask excluding whole trials. Post-cue decoding IMPROVED (0.684 -> 0.759) and the shadow signature vanished (negative pre/post correlation in 30/36 sessions -> 2/36)."
            "\n\nEVERY NUMBER QUOTED IN THESE NOTES WAS COMPUTED WITH THE ENGAGED CUT AT 2.0 s. On 2026-08-21 decode.max_rt_s moved to 3.5 s -- the task's real response window -- because the no-lick arm was holding rewarded hits (39.3% of it for PS92, 33.9% for PS93). Trials licking between 2.0 and 3.5 s move from the no-lick arm into the engaged one, so every decode/encode number shifts on the next rebuild. The FIGURES are current; the numbers written into this prose are pre-change until re-measured. CACHE_VERSION was bumped so nothing silently reuses the old features.")

M_COMMON = M_COMMON + _M_LICK_UNIT

M_DECODE = ("Decoder: multinomial logistic regression (L2, C=0.5) on standardized component activities, 6 "
            "positions, chance=0.167. Activity = a SUB-BINNED TIME COURSE over the aligned window (adopted 2026-08-14), NO per-trial baseline: the window is split into equal bins and their means concatenated, so the decoder sees the window's temporal profile rather than one number. Pre-cue and post-cue use 4 x 0.5 s, post-lick 8 x 0.25 s (configs/defaults.yaml decode.bins). PRE-CUE SUB-BINNING IS UNESTABLISHED (re-measured on all 44 curated sessions, 2026-08-17): +0.009 over the plain 2 s mean, better in 23/44 -- a coin flip. The +0.032 previously quoted here came from a 16-session pilot and did not replicate. roll2x1.0 is nominally best (+0.016, 28/44) but is the max of six arms scored on the same sessions. precue=4 is retained because changing it would move every pre-cue number again for no demonstrated gain, not because it is better. Post-cue (+0.020) and post-lick (+0.023) remain 16-session pilot values and have NOT been re-run. Bin WIDTH matters only post-event -- 0.25 s wins post-lick but OVER-slices pre-cue. "
            "Cross-validation is BLOCK-AWARE (GroupKFold, groups = position blocks) so block drift "
            "cannot leak train->test, and the StandardScaler sits INSIDE the pipeline so it is refit on "
            "each training split -- no held-out trial enters it. A block runs until the position changes "
            "OR until it reaches the scheduler block_size_max from that session's gui_config (8 so far). "
            "That max-length rule was added 2026-08-18: without it, two blocks scheduled back-to-back at "
            "the SAME position merged into one CV group -- 118 of 4216 blocks (2.8%), measured against "
            "the firmware's own block_number. Merging made groups LARGER and the CV therefore more "
            "CONSERVATIVE, so it understated rather than inflated accuracy (mean +0.011 on the "
            "worst-affected sessions). Residual limit: 4+4 merges land at exactly block_size_max and "
            "cannot be separated by length. Post-cue align = window after cue onset (predicts held-out no-lick "
            "trials too = 'no lick generalization'); pre-cue align = a 2 s LICK-FREE window ending at the "
            "cue (adopted 2026-08-17, configs/defaults.yaml decode.precue_lickfree). If a lick falls in "
            "the fixed window it slides earlier to the latest lick-free gap, BOUNDED AT THE SPOUT STROBE "
            "so it cannot reach back to before this trial's position existed; a trial with no clean "
            "window anywhere is DROPPED. Cost: 0% of trials in most sessions, 9.7% in PS93 8/9. Until "
            "2026-08-17 the headline used ALL engaged trials and relied on the task's enforced no-lick "
            "period to keep the window quiet -- which it does for 90.8-99.5% of windows, but only 76% in "
            "PS93 8/9, the animal whose licking is already atypical. Rolling = sliding 0.5 s window across ENL -> "
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
            "the result; CV groups are SESSIONS, so each held-out fold is an entire unseen day. "
            "WHAT 'NO TRIAL FROM THE PLOTTED DAY' MEANS EXACTLY: no trial from that day contributes to "
            "the CLASSIFIER fit. The per-session z-scoring necessarily uses that day's own trials for "
            "its mean and SD -- label-free, and what makes cross-session pooling possible at all, but "
            "not nothing, and the stronger reading of that phrase would be wrong. Blocks play no part "
            "in the cross-day arm: it groups on SESSION, so the block rule affects only the same-day "
            "ceiling quoted beside it. The "
            "same-day ceiling quoted on each panel is that session's own within-day block-CV accuracy, so "
            "held-out-day minus ceiling is the true cost of freezing. Measured 2026-08-11: the cost is "
            "POSITIVE for every animal and BOTH alignments -- post-cue PS92 +0.140, PS93 +0.068, PS94 +0.071, PS95 +0.072; pre-cue +0.159, +0.078, +0.117, +0.124 (recomputed 2026-08-17 after bug 17, which had every ROI frozen number running on four copies of bin 0 since 8/14; the conclusion held, the magnitudes did not) - the frozen "
            "model beats the same-day model, because it trains on ~3000 trials instead of ~500 and ROI "
            "features are stable across days. Caveat for interpretation: a softmax decoder never abstains, "
            "so confidence alone is NOT evidence of preserved coding - see the OOD control (shuffled-label "
            "entropy floor + trials with no detected lick, which stay confident regardless). "
            "CORRECTED 2026-08-17: this note previously said those trials 'decode at chance'. They do not, "
            "and the flag that said they were ABOVE chance was equally wrong -- both compared against a "
            "uniform 1/6, which is not this arm's null, because the trials are skewed across positions "
            "(PS93: 49% far_center) and the decoder's predictions on them are skewed too. Against a "
            "permutation null computed on these trials the PRE-cue code does survive while the POST-cue "
            "code collapses, which is the intended readout, not a defect. See section D2. " + M_COMMON)
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
    "\n\nTHREE ARMS, NOT TWO, AND ALL THREE BOUNDED BY THE RESPONSE WINDOW. 'engaged' = a lick "
    "within decode.max_rt_s (2.0 s); 'late_rewarded' = a first lick between 2.0 s and that session's RESPONSE WINDOW (3.5 s, read from gui_config.json timing.response_window) -- slow, but a HIT by the task's own scoring; 'undetected' = no detected lick within that window. There is deliberately NO arm past the response window: a lick at 4 s arrives while the spout is already moving and belongs to no trial cleanly. An earlier version of this note said '2-5 s' and 'none within 5 s' -- 5 s was never in the code (nolick_decoder.categorize); corrected 2026-08-19. The pipeline's older arm pooled them, which is misleading: on PS93 8/12 the pre-cue "
    "survival is carried entirely by LATE trials (balanced 0.532, p=0.003) while undetected trials "
    "show nothing (0.153, p=0.76). "
    "\n\nTHE NULL IS NOT 1/6. These trials are heavily skewed across positions (PS93: 49% far_center) "
    "and the decoder's predictions on them are skewed too, so an information-free decoder scores "
    "above 1/6 -- 0.211 for PS93. Headline is BALANCED accuracy (macro-recall), whose null "
    "expectation is exactly 1/6 however skewed either side is; raw accuracy is judged against a "
    "permutation null computed on these trials with predictions held fixed; a position-matched "
    "subsample is stored as an independent check. An earlier 'above chance' flag here compared "
    "against uniform 1/6 and was meaningless; it is retired. "
    "\n\nQUOTE THE DIRECTION, NOT THE LABEL. The per-animal verdict binarizes a continuous ratio at "
    "1.5x, so an animal near the cut flips between bases and the consensus reads 'DISAGREEMENT' for "
    "what is 1.4x versus 1.6x. The threshold-free summary is `direction_consistency` in the "
    "reference JSON: pre-stroke, pre-cue survival exceeds post-cue in 8/8 animal x basis "
    "comparisons, ratios 1.2-3.3x (PS95 strongest at 3.1-3.3x, PS93 1.7-1.8x in both bases; PS92 "
    "and PS94 straddle the threshold and are the two whose labels differ by basis). The ROBUST "
    "claim is the direction; the per-animal label is a convenience on top of it. "
    "\n\n'NO DETECTED LICK' IS NOT 'NO ATTEMPT'. The sensor needs contact, so an executed but short "
    "lick registers as nothing. PS93 has a pre-existing rightward tongue bias and reaches far_L "
    "poorly (Priya, 2026-08-17), so its far-position undetected trials are substantially "
    "attempted-and-short -- which makes PS93 far_L a PRE-stroke, within-subject instance of the very "
    "phenotype this analysis looks for post-stroke. DLC/facial tracking is needed to split attempted "
    "from unattempted; until then read the per-position breakdown, not the pooled number.")

M_HEMI = (
    "HEMISPHERIC RAW FLUORESCENCE (wfield_local.hemispheric_intensity). Per session, the median raw "
    "count in the LEFT and RIGHT Allen masks on the atlas grid, per channel, from "
    "frames_average_atlas.npy. Hemisphere comes from the signed Allen area code, checked against the "
    "_left/_right name suffix, not from an image midline -- which would be wrong under any headplate "
    "rotation."
    "\n\nWHY RATIOS. LED power is titrated by hand day to day (crossday_intensity warns about this "
    "on its own figure), so an absolute cross-day trend can be the LED setting. That confound is "
    "common to both hemispheres WITHIN a session and cancels in an L/R ratio, as do exposure, gain "
    "and bleaching. What does NOT cancel is anything spatially asymmetric -- window clarity, focus "
    "tilt, uneven illumination, headplate shift -- so the question is never 'is L/R != 1' (it never "
    "is) but 'did L/R MOVE from this animal's own pre-stroke range'."
    "\n\nTHE 470 AND 415 QUESTIONS ARE THE SAME MEASUREMENT. 415 nm is the isosbestic "
    "channel, meant to carry the optical component without calcium, so only the ratio of "
    "ratios (470 L/R)/(415 L/R) is GCaMP-specific."
    "\n\nTHE PERFUSION DIRECTION IS UNRESOLVED, and an earlier version of this note "
    "asserted it wrongly (Priya challenged it, 2026-08-18). It argued that haemoglobin "
    "absorbs, so more blood means less light, so a 415 RISE meant hypoperfusion. Measured "
    "here that is backwards: cue-triggered averages of the RAW violet trace RISE in all four "
    "animals (+0.54 to +1.96%), tracking the blue positive control (+2.31 to +3.69%) at about "
    "a third of its amplitude. Nor can the sign simply be flipped -- that test is the DYNAMIC "
    "task-locked regime while the L/R ratio is a STATIC months-long baseline, and they need "
    "not agree. Read a 415 change as an optical asymmetry of UNKNOWN perfusion sign until an "
    "independent measure (laser speckle, or a manipulation of known direction) settles it."
    "\n\nRESULT ON 8/17: NO detected change. Every measure sits inside the animal's own pre-stroke "
    "range in both region groups -- PS94 whole-hemisphere 415 z=+0.7, 470 z=+0.3, GCaMP-specific "
    "z=-0.9; PS95 z=+0.3, -0.3, -0.9; SSp similar (|z| <= 0.8). No evidence here for a left-sided "
    "470 increase, and no detectable change in the optical asymmetry, one day post-lesion."
    "\n\nBUT THE NULL DOES NOT HOLD ONCE 8/18-8/19 ARE IN (updated 2026-08-21; the sentence above "
    "was written when 8/17 was the only post-stroke session and describes only that day). PS93 goes "
    "OUTSIDE its own pre-stroke range on 8/19 in the whole-hemisphere group, on BOTH raw channels: "
    "ratio_415 z=+2.50 and ratio_470 z=+2.75. Because both channels move together and by a similar "
    "amount, the GCaMP-specific ratio does NOT clear the band -- which is exactly the signature of an "
    "OPTICAL change (window, focus, blood volume) rather than a change in the calcium signal, and is "
    "why the 415 channel is reported beside the 470 one rather than divided out silently. Read it as "
    "a preparation change in PS93, not as evidence about its cortex. No other animal clears the band "
    "on any session."
    "\n\nREAD THAT NULL WITH ITS POWER. The 415 L/R ratio DRIFTS monotonically across the "
    "pre-stroke period in all four animals (PS94 SSp 0.66 -> 0.82), so the min-max pre-stroke band "
    "spans a trend rather than noise, and a step change would have to be large to escape it. A "
    "sharper test compares against the EXTRAPOLATED trend, or against the last few sessions only. "
    "That drift is present in PS92/PS93 as well, who had no effective lesion until after 8/17, so it "
    "is not lesion-related -- it is a property of the preparation or the rig and is unexplained."
    "\n\nAND IT MEASURES THE WRONG THING FOR 'ACTIVITY'. This is the session MEAN image: static "
    "baseline fluorescence. An impression of more activity is about DYNAMICS, which the mean image "
    "cannot show. The matching test is a per-hemisphere temporal SD or task-evoked amplitude; it is "
    "not run here. n=1 post-stroke session, one day post-lesion, and perfusion changes evolve.")

M_HEMIDYN = (
    "PER-HEMISPHERE DYNAMICS AND CROSS-HEMISPHERE COUPLING (wfield_local.hemispheric_dynamics). Two "
    "measures a MEAN image cannot give, which is what the hemispheric-intensity slides are limited to: "
    "TEMPORAL SD (how much each hemisphere's signal actually moves over the session, as an L/R ratio) "
    "and HOMOTOPIC CONCORDANCE (the correlation between the same Allen area in the two hemispheres). "
    "Features are the same ones every decoder here uses, on the adopted hemodynamic variant."
    "\n\nBELIEVE THE CORRELATION OVER THE AMPLITUDE. Temporal SD inherits every asymmetric optical "
    "confound the intensity ratio has -- window clarity, focus tilt, uneven illumination -- because it "
    "IS an amplitude and those are what move amplitudes. A homotopic correlation is invariant to "
    "per-hemisphere gain: dim one hemisphere by any factor and its correlation with the other is "
    "unchanged. Where the two disagree, the correlation is the trustworthy one."
    "\n\nTHE SPECIFICITY CONTROL IS THE THIRD ROW. If EVERYTHING decorrelates post-stroke -- more "
    "movement, different arousal, a noisier recording -- then a homotopic drop says nothing about "
    "interhemispheric coupling. The interpretable result is homotopic falling while WITHIN-hemisphere "
    "coupling holds, which is what homotopic-minus-within reports."
    "\n\nGREY POINTS ARE PS92/PS93 8/17 — SMALL strokes with no overt deficit. They are "
    "NOT a no-lesion control: the 8/16 laser did lesion them (Priya, 2026-08-18, correcting an "
    "earlier claim of mine). What they DO control for is the recording DAY — same rig, "
    "anaesthesia, handling, preprocessing and frozen decoder — so an artefact of 8/17 would "
    "have hit all four animals. They also give a LESION-SEVERITY contrast. What they cannot show "
    "is that a lesion is NECESSARY for an effect: a null in a small-stroke animal is equally "
    "consistent with small stroke, small effect."
    "\n\nAllen-ROI basis by default, because homotopic pairing is then exact (SSp_left <-> "
    "SSp_right). The joint LocaNMF basis pairs through each component's dominant area and is "
    "approximate; where they disagree the ROI answer is the conservative one.")

M_VESSEL = (
    "SURFACE VESSEL CONTRAST (wfield_local.vessel_contrast). Vessels image DARK because haemoglobin "
    "absorbs, so their contrast against surrounding cortex is an optical readout of blood in the light "
    "path: fainter vessels mean less blood. Measured on the session mean image as the depth of dark "
    "structure against a blurred background (90th percentile, plus the mean as a total-energy "
    "companion), divided by the median so it is invariant to LED power, exposure and gain. A Frangi "
    "vesselness filter runs alongside as an independent estimator."
    "\n\nWHY THE L/R RATIO IS THE HEADLINE. Focus drift, a clouding window and a changed working "
    "distance all reduce apparent vessel contrast BILATERALLY, so the two-hemisphere mean cannot "
    "separate optics from biology and is shown only to reveal such a global change. A ratio is "
    "untouched by a symmetric optical change."
    "\n\nRESULT, ALL POST-STROKE SESSIONS 8/17-8/19: NO detected change (scope widened 2026-08-21; the "
    "numbers below are 8/17, and NO session of any animal has since cleared its own pre-stroke range on "
    "any of the three estimators). PS94 415 depth L/R z=-1.0, energy z=-0.7, Frangi z=-0.4; "
    "PS95 z=+1.0, +0.1, -0.1 -- opposite directions, both inside the animal's own pre-stroke range. "
    "PS94 does trend toward fainter LEFT vessels, the direction the observation predicted, but it stays "
    "within a pre-stroke range that spans 0.911-1.204, so the measure cannot resolve it."
    "\n\nTHE LIMIT THAT MATTERS MOST HERE IS ANATOMICAL, NOT STATISTICAL. These are PIAL vessels over "
    "DORSAL cortex; the lesion is VENTROLATERAL STRIATUM -- deep, lateral, and largely outside the "
    "imaged field. A null in this measure is weak evidence about perfusion at the lesion, and it would "
    "be wrong to read it as showing that striatal perfusion is intact."
    "\n\nSANITY CHECK, and it fails in 11 of 65 sessions. 415 nm should show MORE vessel contrast "
    "than 470 because haemoglobin absorbs far more strongly there. The failures cluster in PS92 (6) and "
    "PS93 (4) rather than scattering, which points at something systematic in those animals' windows. "
    "Every PS94 and PS95 session passes, including both 8/17 sessions, so the numbers quoted above are "
    "not affected -- but PS93_0817 fails, so its small-lesion value should not be read."
    "\n\nThis is the session MEAN image, so an acute or spatially focal vessel change would not "
    "appear. It also does NOT settle the perfusion direction left unresolved in M_HEMI: that "
    "retraction stands.")

M_FIXEDSCALE = (
    _M_LICK_UNIT +
    "PRE- vs POST-STROKE ACTIVITY MAPS ON ONE COMMON COLOUR SCALE (wfield_local.fixed_scale_maps). "
    "Built to answer Priya directly: the preprocessing decks show much larger amplitude bars post-stroke, and the question was whether any existing figure bears that out. None does, and one "
    "actively hides it."
    "\n\nWHY THE STANDARD MAPS CANNOT SHOW IT. plot_spout_trial_averages sets its colour limit from a percentile of THAT SESSION's own maps, so every session is renormalised to fill the same "
    "colour range. A session whose responses are three times larger looks identical -- only the "
    "number on the colourbar changes. That is the right default for reading one session's spatial "
    "pattern and exactly wrong for comparing amplitude across sessions, which is why the "
    "observation had to be made by reading colourbar numbers. Here every panel in a figure shares "
    "one symmetric vmin/vmax computed across ALL panels, so a 2-3x amplitude difference appears as "
    "a 2-3x difference in colour saturation."
    "\n\nMETHOD. Maps are reconstructed on the ATLAS grid (U_atlas @ window-mean SVT), so pixels are comparable across sessions and animals, and the window is the same one the decoders use. "
    "Minimum 8 trials per position. NO z-scoring anywhere in this figure -- z-scoring is what would "
    "destroy the amplitude comparison it exists to make. Post-stroke sessions use ALL trials."
    "\n\nTHE dF/F DENOMINATOR WAS CHECKED FIRST, because a figure that dramatises an artefact is worse than no figure. Baseline F is unchanged post-stroke (PS94 pre/post ratio 1.01, PS95 1.02, "
    "PS92 0.99), so the rise is in the numerator, not the denominator."
    "\n\nWHAT THE MAPS ACTUALLY SHOW, AND IT IS NOT WHAT THE SUMMED MEASURE SAID. PS94 PEAK amplitude rises only at close_L (0.039 -> 0.073) and close_center (0.029 -> 0.055), is flat at "
    "close_R, and FALLS at the far positions (far_center 0.018 -> 0.008) -- the positions the animal "
    "stopped reaching. Meanwhile evoked_amplitude's SUMMED |response| rose at far_L (0.278 -> "
    "1.070) while its peak stayed flat (0.019 -> 0.017). Those reconcile only one way: the response "
    "became spatially BROADER, not stronger. Read this figure as the evidence for SPREAD; read G8e "
    "for the decomposition."
    "\n\nCAVEAT. LED power is set by hand daily, so absolute pixel values carry a session confound that this figure deliberately does NOT normalise away. It is here to make the raw "
    "comparison visible and honest, not to settle it; the scale-free measures in G8e do that.")

M_EVOKED = (
    "PER-AREA EVOKED AMPLITUDE (wfield_local.evoked_amplitude). Built to test Priya's reading of the "
    "preprocessing decks: more R sensorimotor activity post-stroke, especially at the far positions, "
    "and much larger amplitude bars overall. The four hemispheric nulls cannot speak to that -- "
    "intensity, dynamics, concordance and vessels all collapse across space, so a focal change "
    "averages away in every one of them."
    "\n\nTHREE QUANTITIES, because 'bigger' is ambiguous. ABSOLUTE = mean windowed response per area "
    "x position, the quantity the map colourbars show, SUMMED over 66 areas. SHARE = each area's "
    "|response| as a fraction of the session total. R-L INDEX = (right - left)/(|right| + |left|) per "
    "homotopic pair. Only ABSOLUTE carries the baseline confound: the signal is a deviation from the "
    "session's own mean and LED power is set by hand daily, so a rise can be a larger response OR a "
    "smaller baseline. SHARE and R-L are scale-free. The R-L index validates on pre-stroke data, "
    "giving +0.44 at close_L and -0.81 at close_R in PS94 -- clean contralateral organisation. Post "
    "arm = ALL trials; reference = the CURATED pre-stroke set only."
    "\n\nTHE HEADLINE THIS SLIDE USED TO CARRY WAS FALSE, and the way it was false matters. It read "
    "AMPLITUDE RISES IN ALL FOUR ANIMALS, GRADED BY SEVERITY. PS92 and PS93 had contributed NO "
    "post-stroke rows to it: the session filter was hardcoded to `curated_dates() | {'0817'}`, and "
    "8/17 is in those two animals' exclude list because their effective lesion is 8/18 -- so the "
    "claim covered four animals on data from two. With all four measured, PS92 shows NO rise "
    "anywhere (close_R z=-1.6, far_center -1.4, far_R -2.4; everything else flat). The correction is "
    "recorded in DECISIONS.md (2026-08-19)."
    "\n\nWHAT THE SUMMED AMPLITUDE ACTUALLY DOES. Close positions rise or hold; FAR positions FALL, "
    "and far_R falls in all four animals on day 1 (z = -2.4 PS92, -2.1 PS93, -1.3 PS94, -4.7 PS95). "
    "The far positions are the ones the animals stop reaching, so part of this is trial composition "
    "on the all-trials arm rather than a change in the response to a given movement -- PS95's far_R "
    "goes 0.086 with one lick trial on 8/17 to 0.643 with 84 on 8/18, tracking the behaviour exactly. "
    "Read it as amplitude COVARYING WITH ATTEMPT, not as a lesion effect on the sensory response."
    "\n\nAND THE SUMMED MEASURE CONFLATES AMPLITUDE WITH SPATIAL EXTENT. G8d's common-scale maps "
    "separate them: PS94 PEAK amplitude rises only at close_L (0.039 -> 0.073) and close_center "
    "(0.029 -> 0.055), is flat at close_R and FALLS at the far positions (far_center 0.018 -> 0.008), "
    "while the SUMMED measure rose at far_L (0.278 -> 1.070) with its peak flat (0.019 -> 0.017). "
    "Those reconcile one way: the response became spatially BROADER, not stronger."
    "\n\nFINDING 2 SURVIVES AND IS NOW SHARPER: LATERALISATION COLLAPSES, AND ONLY IN PS94. "
    "Restricting to positions that were actually lateralised before the lesion (|pre-stroke R-L| > "
    "0.15 -- a position with no lateralisation to begin with cannot lose any), PS94 has all SIX "
    "lateralised and FIVE change: close_L, close_center, close_R and far_R all move TOWARD ZERO "
    "(close_R -0.81 -> -0.26 -> -0.20), and far_center REVERSES SIGN (-0.30 -> +0.31). Identical "
    "position list on 8/17 and 8/18 and at both alignments -- four readings, no disagreement."
    "\n\nTHE OTHER THREE DO NOT DO THIS, and the DIRECTION is what separates them -- not whether a "
    "value is outside the band. PS92: nothing outside at any position, either alignment. PS93: its "
    "changes go the OTHER WAY, far_R lick-aligned -0.19 -> -0.47, MORE lateralised, not less. PS95: "
    "far_R also moves AWAY on 8/17 (-0.33 -> -0.64), and by 8/18 two positions have moved toward "
    "zero. Counting 'positions outside the pre-stroke range' would score PS93 4/6 and PS94 5/6 and "
    "make them look alike; they are moving in opposite directions."
    "\n\nREAD THE COLLAPSE AS LOSS OF LATERALISATION, NOT AS 'MORE RIGHT'. The index moves toward "
    "zero from BOTH signs -- at right-spout positions that reads as relatively more right activity, "
    "which is what the maps show; at left-spout positions it reads as less right. A uniform rightward "
    "shift would move every position the same way, and it does not. The R-L index cannot distinguish "
    "a bilateral convergence from a rightward relocation at all, which is why the midline test in "
    "G8f exists."
    "\n\nIT ALSO GAIN-CHANGES AND REDISTRIBUTES, not one or the other. PS94 per-area SHARE z "
    "exceeds |2| in 8-9 of 66 areas at close_R and close_L against ~3 expected by chance, led by "
    "SSp-bfd, SSp-n, SSs and VISC -- somatosensory and visceral, consistent with the sensorimotor "
    "reading. close_center and far_L are at chance. Reported per post-stroke DAY; an earlier version "
    "read `post[0]` and so described day 1 only."
    "\n\nMECHANISTICALLY THIS FITS THE DECODING. If contralateral lateralisation collapses, the "
    "position-specific spatial patterns become less separable -- which is what G3b shows as PS94's "
    "far_R over-prediction (35% of all trials) and its precision collapse from 0.92 to 0.28."
    "\n\nREFERENCE IS THE CURATED SET ONLY. An earlier version built the pre-stroke band from every "
    "date resolving to phase=='pre', including the noisy early-June sessions curated_dates() exists "
    "to exclude -- PS95_0605 has a mean |amplitude| of 16.3 against ~0.53 elsewhere, and put PS95's "
    "band at [0.15, 18.09], inside which no post-stroke value could ever fall. Curation now applies "
    "to the PRE side only (config.analysis_sessions), which is what it was always for.")

M_SPATIAL = (
    "DID THE POSITION CODE CONVERGE, AND DID IT CROSS THE MIDLINE? "
    "(wfield_local.spatial_reorganisation). Two tests on the spatial maps, both following from the "
    "same prediction: if PS94's contralateral lateralisation collapses (G8e) and its position "
    "information survives but is unreadable by the pre-stroke decoder (G2c), those should be one "
    "fact seen from two sides."
    "\n\nTEST 1, CONVERGENCE. Losing lateralisation should make the six position patterns LESS "
    "distinguishable from one another. Measured as the mean CROSSNOBIS distance over the position "
    "pairs -- cross-validated Mahalanobis, computed across disjoint block folds so the estimate is "
    "unbiased by trial noise. That is not optional here: a plain correlation RDM is inflated by "
    "noise, and post-stroke sessions differ in trial count AND in spatial extent, so a raw metric "
    "would move even if the geometry were identical."
    "\n\nTHE PRE-STROKE BAND IS REBUILT ON EACH SESSION'S OWN POSITIONS. Mean distance averages "
    "over PAIRS -- 15 for six positions, 6 for four. On the lick-only arm a session keeps only the "
    "positions it still licks at (PS94 has four), so scoring it against a band computed over six "
    "would compare a mean over one pair set with a mean over another. That is a different quantity, "
    "not a smaller one, and it is the same error class as the decoding arms' chance level moving "
    "with behaviour. Corrected 2026-08-19; the pre-session matrices are stored, so the matched band "
    "follows from them exactly."
    "\n\nRESULT 1: POST-CUE GEOMETRY CONVERGES IN THREE OF FOUR. All-trials arm, day 1 after an "
    "effective lesion: PS92 z=-3.3, PS93 -2.3, PS94 -3.0, all below their own pre-stroke minimum; "
    "PS95 -1.2, inside. On the LICK-ONLY arm those weaken to -1.3, -1.4 and -1.4, so a substantial "
    "part of the post-cue convergence is carried by the no-lick trials -- which is what one expects "
    "if the missing movement is the thing that changed, and is a reason to read the two arms "
    "together rather than picking one."
    "\n\nRESULT 2, AND IT QUALIFIES THE HEADLINE: PS94's PRE-CUE GEOMETRY IS ALSO DEGRADED. "
    "PS94 pre-cue z=-3.7 on 8/17 and -5.5 on 8/18, and this SURVIVES the lick-only arm almost "
    "unchanged (-3.2, -5.4) -- so it is not an artefact of folding heterogeneous no-lick trials into "
    "a within-position covariance estimate, which was the obvious explanation and was tested for "
    "exactly this reason. The other three animals' pre-cue geometry stays inside the band (PS92 "
    "-0.1, PS93 -1.2, PS95 +0.2/+0.5)."
    "\n\nHOW THAT RECONCILES WITH G2c, WHERE PS94's PRE-CUE DECODING IS INSIDE THE BAND (z=-0.2). "
    "Crossnobis measures how far apart the patterns are in units of noise; decoding measures whether "
    "a boundary can still be drawn between them. PS94's PRE-STROKE pre-cue crossnobis is unusually "
    "large -- 5.71, against 1.30 (PS92), 1.98 (PS93) and 1.65 (PS95) -- so falling to 2.45 leaves it "
    "at roughly the level the other three animals run at normally, which is comfortably decodable. "
    "Both statements are true: the distances shrank a great deal from an unusually high baseline, "
    "and the code remained readable. For PS94 the dissociation is therefore a matter of DEGREE -- "
    "both windows lose separability and only the post-cue loss crosses the threshold where six-way "
    "decoding fails -- rather than 'pre-cue untouched'. Stated that way in DECISIONS.md."
    "\n\nTEST 2, MIDLINE TRANSFER. 'More right activity' has two very different readings: the "
    "right hemisphere doing more of its own thing, or the LEFT hemisphere's pattern having RELOCATED "
    "to the right. Each post-stroke pattern is correlated against its own pre-stroke pattern AND "
    "against the HEMISPHERE-SWAPPED version (each Allen area's _left value exchanged with its "
    "_right -- a mirror at region resolution, robust to the pixel-level registration error a literal "
    "image flip would inherit). The R-L index of G8e cannot answer this: it is symmetric, so a "
    "rightward shift and a bilateral convergence both move it toward zero."
    "\n\nRESULT: THE MIDLINE TEST IS A CLEAN NULL. No transfer at any position, in any animal, at "
    "either alignment, on either arm. The 'left map moved right' reading of the map observation is "
    "not supported."
    "\n\nTHREE VERDICTS, BECAUSE TWO WERE NOT ENOUGH. TRANSFER requires the mirrored pattern to "
    "actually be matched (mirror_r >= 0.20), to exceed the normal correlation, AND to beat the "
    "pre-stroke baseline difference by 0.15 -- a symmetric brain already has substantial mirror "
    "correlation, so the raw ordering carries almost no information. REDUCED ASYMMETRY is the weaker "
    "claim and keeps its own flag. PATTERN LOST is the third: when the post-stroke pattern resembles "
    "NEITHER its own pre-stroke pattern nor the mirrored one, 'which hemisphere does it look like' "
    "has no answer. Two earlier versions of this rule reported transfer where none existed -- one "
    "flagged a 0.005 correlation difference, the other flagged PS94 far_center where normal_r was "
    "-0.632 and mirror_r -0.480, i.e. where the representation had disappeared. Both are recorded in "
    "DECISIONS.md."
    "\n\nWHAT PATTERN LOSS FINDS. Cue-aligned, all trials, day 1: far_R has lost its pattern in "
    "ALL FOUR animals, and PS94 and PS95 lose far_center as well. Those are the positions the "
    "animals stop attempting -- so on this arm the finding is confounded with the absence of the "
    "movement itself and must not be read as a lesioned sensory representation. The PRE-CUE arm, "
    "which precedes the movement, does NOT show the same far-position concentration: its losses are "
    "scattered and fall mostly on PS95's close positions. That asymmetry is the caveat, not a "
    "footnote to it."
    "\n\nFIGURES. spatial_reorganisation_{cue,precue}.png is the ALL-trials arm; the "
    "_lickonly suffix is the engaged-only arm. Per-position bars are the two correlations (own "
    "pre-stroke pattern in blue, hemisphere-swapped in orange); orange above blue would be transfer, "
    "and it never happens.")

M_RECODING = (
    "PLAN vs EXECUTION (poststroke_compare.recoding_test; figure poststroke_grid.png). Within-session "
    "decoding -- a decoder trained on the post-stroke session ITSELF -- against that animal's own "
    "pre-stroke range for the same measure. ALL-TRIALS arm: all six positions, chance 1/6, which is "
    "the only arm comparable across sessions and animals."
    "\n\nTHE RESULT, IN ALL FOUR ANIMALS. On the first session after an EFFECTIVE lesion the PRE-CUE "
    "window sits inside the pre-stroke band while the POST-CUE window sits outside it: PS94 8/17 "
    "pre-cue z=-0.2 vs post-cue z=-7.1; PS95 8/17 +1.6 vs -3.4; PS92 8/18 +0.2 vs -2.4; PS93 8/18 "
    "-0.5 vs -3.6. Four animals, two lesion days, three laser powers, no exception."
    "\n\nWHY IT RESISTS THE CONFOUNDS THAT SANK EVERYTHING ELSE. Pre-cue and post-cue are two "
    "windows on the SAME TRIALS, so LED power, baseline F, evoked amplitude, arousal, engagement and "
    "trial count act on both equally and cannot produce a difference between them. It is a "
    "within-trial contrast, which is what makes it survive when session-level comparisons do not."
    "\n\nPS92 AND PS93 SUPPLY A WITHIN-ANIMAL BEFORE/AFTER CONTROL. Their 8/17 sessions follow the "
    "8/16 laser that did NOT take, and show NOTHING outside the band at any alignment (PS92 -0.1, "
    "+0.2, -0.3; PS93 -1.4, -0.0, -0.5). One day later, after the effective 8/17 lesion, the "
    "dissociation is present. Same animal, same rig, one day apart -- far stronger than a "
    "between-animal comparison, and it exists only because the excluded sessions were kept analysable "
    "instead of discarded."
    "\n\nDAY 2 SEPARATES THE ANIMALS. PS94 loses the plan as well (pre-cue z=-3.4, post-cue z=-12.1) "
    "while PS95 returns fully inside the band (+2.1, +1.3) alongside a behavioural recovery of the far "
    "positions (far_center 10 -> 99 trials, far_R 1 -> 84). Deterioration versus recovery, tracking "
    "the behaviour in both cases."
    "\n\nLASER POWER DOES NOT PREDICT MAGNITUDE. PS94 at the LOWEST dose (3 mW) has the largest "
    "post-cue deficit (z=-7.1) and PS93 at the highest (5.5 mW) has -3.6. Behavioural severity tracks "
    "the effect; dose does not. PS93 was flagged in advance as the dose test and came out negative."
    "\n\nTHE ALL-TRIALS ARM IS THE ONE TO READ ACROSS SESSIONS. Its position set is fixed at six, so "
    "chance is fixed at 1/6. The LICK-ONLY arm uses each session's preserved positions, which change "
    "with behaviour -- PS95 has 4 on 8/17 and 6 on 8/18 -- so its numbers carry different chance "
    "levels and CANNOT be laid side by side. Both arms are reported because their DIFFERENCE separates "
    "a degraded code from a code that is fine whenever the animal manages to lick."
    "\n\nEARLIER VERSIONS OF THIS NOTE WERE WRONG and the corrections are recorded in DECISIONS.md "
    "(2026-08-19). A claim that PS94's information was INTACT and only the code had changed came from "
    "filtering to engaged trials; a claim that PS95 was impaired on day 1 and recovered came from a "
    "pooled (union) position basis that scored its 8/17 session over a position with ONE engaged "
    "trial. Both are withdrawn."
    "\n\nCAVEATS. One session per animal per day. And PRE-CUE means pre-cue position information, "
    "not a demonstrated motor intention: the spout arrives ~3 s before the cue, so a sustained sensory "
    "response and a held plan are temporally coextensive and this design cannot separate them (see "
    "DECISIONS.md). The dissociation is between two WINDOWS, which is solid; naming the earlier one a "
    "plan is an interpretation.")

M_CODING_DIR = (
    "PER-POSITION CODING DIRECTIONS. The feature space has one axis per (LocaNMF component, time "
    "sub-bin) -- 348-380 dimensions for ENL/cue, ~700 for lick -- so a direction is a weight per "
    "component PER MOMENT in the window. For each spout position P it is fitted on PRE-STROKE "
    "trials WITH A SUCCESSFUL LICK, P against the other five, in the SHARED joint-LocaNMF basis "
    "(fixed footprints; post-stroke sessions are projected, never refitted). It is a CONTRAST: "
    "without the comparison there is no axis.\n\nREPORTED AS A LINEAR PROJECTION, pole-normalised "
    "so 0 = pre-stroke not-this-position and 1 = pre-stroke lick here. NOT a probability: a sigmoid "
    "saturates, these directions are strong (AUC up to 0.98), so degradation measured from a "
    "saturated reference is understated and unevenly so between positions of different "
    "separability -- which would corrupt exactly the orderings this is for.\n\nENGAGEMENT IS "
    "PROJECTED OUT for ENL and cue. The plain difference-of-means direction carries a large "
    "lick/no-lick component -- cos(w, engagement) of 0.82 / 0.91 / 0.71 / 0.52 in PS92/93/94/95, on "
    "a different position in each animal -- so a no-lick class could score low because it was a "
    "no-lick trial rather than because its position code changed. After removal, pre-stroke no-lick "
    "sits at one consistent value on every axis (0.16-0.17 for PS94) instead of scattering -2.03 to "
    "+1.38, and the pre-stroke lick diagonal improves. Logistic-regression directions were already "
    "clean (|cos| <= 0.07) because they account for covariance; they are on disk as the independent "
    "check.\n\nWHAT THE WINDOWS CAN SAY. ENL is the clean one: nothing has happened yet and the "
    "window is lick-free by construction, so 'with lick following' means a lick came AFTER the cue, "
    "not during the window. The cue and lick windows CONTAIN the movement (median first-lick "
    "latency 0.137-0.255 s pre-stroke, minimum 0.109 s -- there is no movement-free cue window), so "
    "a no-lick class sitting low there says nothing about whether a plan formed. The pre-stroke-lick "
    "vs post-stroke-lick contrast IS like-for-like, since both contain a lick.\n\nCAVEAT ON "
    "ONE-VS-REST. 'Not P' mixes the five other positions, and for MIDDLE positions that mixture is "
    "majority-far, so the axis becomes largely close-vs-far and the position it is named for need "
    "not be the extreme on it. Prefer the WITHIN-RING pairwise panels (close-vs-close, far-vs-far) "
    "for remapping questions -- measured 2026-08-22, they carry half the close-vs-far loading of a "
    "one-vs-rest axis (|cos| 0.33 vs 0.70) and show NO coherent within-session drift even in the two "
    "animals that disengage (6/12 positive, mean -0.01). CROSS-RING pairwise axes are NOT safe: in "
    "those same two animals all 18 drift the same way (mean +0.19), the far position becoming more "
    "far-like as the session runs.\n\nTHE LICK "
    "WINDOW'S NO-LICK CLASSES SIT AT AN INFERRED TIME (2026-08-21). Their window starts at the cue "
    "plus that session's own median RT at that position -- cue-referencing would offset the arms by "
    "the whole reaction time, a median of 2.439 s at post-stroke far_R against a 2 s window. A "
    "position with NO engaged trial that session is DROPPED: the fallback to the session median "
    "fired precisely where the animal had stopped licking while that median was set by the close "
    "positions that still worked, putting PS94's far_R windows at 0.17-0.23 s when its own "
    "successful licks there took 1.80-2.25 s. Where the offset rests on 1-4 trials the log says so. "
    "Read those classes as inference, most cautiously at the far positions.\n\nAMPLITUDE WAS TESTED "
    "AND IS NOT DRIVING THIS. The projection is unbounded, so a trial sitting further from its "
    "session's engaged centroid scores higher whether or not its ANGLE to the direction changed -- "
    "and the two are not independent, since moving further out ALONG the direction raises both. "
    "Re-projecting UNIT-NORMALISED trials (cos(x,w), blind to magnitude, sensitive only to "
    "direction) leaves the picture intact: per-cell shifts are at most 0.15 in PS93/PS94/PS95, and "
    "the cell-to-cell pattern is preserved at r=+0.86 to +1.00 in every animal. PS92's far_center "
    "outlier moves 2.12 -> 1.75, so magnitude contributes about a third of its excess over 1.0 and "
    "direction carries the rest; every other PS92 cell moves by <=0.07. Its earlier r=+0.97 with "
    "the norm ratio reflects far_center being both the most distinctive and the highest-norm cell, "
    "which is one phenomenon measured twice, not a confound. Post-stroke values ABOVE 1.0 are real.")


M_POSTSTROKE = (
    _M_LICK_UNIT +
    "POST-STROKE COMPARISON (wfield_local.poststroke_compare / plot_poststroke). THE COHORT HAS TWO "
    "LESION DATES (configs/animals.yaml stroke_date). PS94/PS95: 2026-08-16 at 3 mW, deficit -> "
    "stroke_date 20260816, and 8/17 is their first POST-stroke session. PS92/PS93: the 8/16 attempt "
    "did NOT take, redone 2026-08-17 AFTER that session at 3.75 and 5.5 mW -> stroke_date 20260817, "
    "with 8/17 belonging to NEITHER phase (lesion followed the recording, but the animal had already "
    "been lasered once). The higher powers are why the second attempt took. stroke_cutoff() is the "
    "EARLIEST date across the cohort (0816), so a pooled pre-stroke reference stays safe for every "
    "animal regardless of which was lesioned when. The pre-stroke reference is FROZEN: 11 curated "
    "dates ending 8/14, 44 sessions, all resolving to phase=='pre'. "
    "\n\nBEHAVIOUR IS SLIDE ONE, AND THAT IS NOT PRESENTATIONAL. On 8/17 both animals stopped "
    "attempting the far positions -- PS94 has ZERO engaged trials at far_center and far_R, PS95 has "
    "10 and 1. A 6-way accuracy computed across that is mostly a statement about which trials exist. "
    "The first version of this analysis reported a PS94 'neural deficit' whose larger part was trial "
    "composition; every decoding slide here is therefore position-MATCHED to what the animal still "
    "attempts, and matched numbers are 4-way (chance 0.25) and NOT comparable to the 6-way numbers "
    "in sections A\u2013F. "
    "\n\nAND READ THE MATCHED DECODING AS A STATEMENT ABOUT THE MODEL, NOT THE CORTEX. G2c trains a decoder on the post-stroke session itself and recovers normal accuracy in both animals, so the frozen decoder's failure is a CHANGED CODE rather than lost information. Wherever these slides say PS94 is impaired, the supported claim is that the FROZEN DECODER is impaired on PS94. "
    "\n\nPRE-ENGAGED vs POST-ALL IS DELIBERATE. Post-stroke trials are NOT filtered to those with "
    "a detected lick, because the missing licks ARE the phenotype -- filtering them out would remove "
    "the effect being measured. Pre-stroke keeps the engaged cut (decode.max_rt_s). The mismatch is "
    "declared in nolick_analysis.SANCTIONED_MISMATCHES rather than left implicit, so "
    "assert_comparable passes it by NAME while any other mismatched pair still raises. "
    "\n\nTHERE IS NO POST-STROKE 'DISENGAGED' LABEL (Priya, 2026-08-18). Engagement filtering "
    "post-stroke is RETIRED (poststroke_compare.POSTSTROKE_ENGAGEMENT_FILTERING = False). The "
    "pre-stroke gate read motor failure as lost motivation -- it called 59% of PS94 8/17 disengaged "
    "against 6.8% for a spared-position gate. The spared-position gate that replaced it was better "
    "but also unvalidated: its 29 'disengaged' PS94 trials were no-lick trials in a local dip of the "
    "response rate at close_L/close_center, and a short run of MOTOR failures produces that dip just "
    "as readily as a motivational lapse. Nor has it a general form -- in a severe stroke every "
    "position may be impaired, leaving no spared reference to anchor engagement on. CONSEQUENCE: the "
    "earlier working-vs-disengaged result (PS94 -0.060) is UNINTERPRETABLE, not negative, and is not "
    "shown anywhere in this deck. The response rate at spared positions is still reported as a "
    "DESCRIPTIVE statistic (PS94 0.89, PS95 0.97); it is never used to split trials. "
    "\n\nWHAT REPLACES IT: the same question asked WITHIN the post-stroke session, splitting "
    "no-lick trials on the TRUE spout position -- impaired versus preserved -- which is measured "
    "rather than inferred and needs no engagement label. Above-null decoding at IMPAIRED positions "
    "means the position was represented and the movement did not happen. "
    "\n\nPS92 AND PS93 ARE EXCLUDED FROM EVERY POOLED SLIDE HERE. Their 8/16 lesion produced no "
    "deficit and was redone after the 8/17 session, so 8/17 belongs to neither phase "
    "(config.session_phase -> 'excluded'). They may be projected onto the joint bases and shown "
    "per-session. Pools are built ONLY from config.phase_labels('post'); selecting by DATE would "
    "sweep them in silently, which is why the guard is a test and not a convention. "
    "\n\n'NO LICK DETECTED' IS NOT 'NO TONGUE PROTRUSION'. The spout needs contact, so a short or "
    "misdirected lick registers as nothing -- PS93's pre-existing rightward bias already produces "
    "exactly this pre-stroke at far_L. Every no-lick conclusion here is provisional on DLC/facial "
    "tracking, which replaces the inference with a measurement. In a severe stroke, spout contact may "
    "not be a usable behavioural readout at all. "
    "\n\nTHE NO-LICK ARM USED TO CONTAIN REWARDED HITS; FIXED 2026-08-21. _trial_features split on decode.max_rt_s = 2.0 s while the task's response window is 3.5 s, so a lick at 2.5 s was a HIT by the task's own scoring that every imaging analysis filed under 'no lick'. That contamination was 9.7% of PS94's no-lick arm and 4.7% of PS95's, but 39.3% for PS92 and 33.9% for PS93. This note previously warned that the slides must be rebuilt when PS92/PS93 re-entered as post-stroke animals or they would report a late-lick effect as a no-lick effect; they re-entered on 8/18, and the fix taken was to move the cut itself -- decode.max_rt_s is now 3.5 s, so 'engaged' means the same thing here as in the behaviour pipeline's hit/miss. EVERY NUMBER COMPUTED BEFORE THAT CHANGE USED THE 2.0 s CUT. nolick_decoder keeps its own 2.0 s boundary so the three-arm split stays available: the late-vs-undetected distinction is a real result, and on PS93 8/12 the entire pre-cue survival sat in the LATE arm (balanced 0.532, p=0.003) while undetected showed nothing (0.153, p=0.76). "
    "\n\nONE SESSION. n=1 post-stroke night per animal: PS94 and PS95 differ (PS94 below every "
    "pre-stroke session in all three readouts, PS95 inside the band), and with one session each that "
    "is a description of two animals, not an established dissociation. Joint-basis replication and "
    "the decoder-similarity analysis are deferred to the second post-stroke session.")

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

# THE ENGAGED-CUT WARNING HAS TO REACH EVERY NOTE THAT SPLITS TRIALS, not just the six that embed
# M_COMMON. decode.max_rt_s moved 2.0 -> 3.5 s on 2026-08-21, so any number resting on the
# engaged/no-lick boundary was measured under the old cut and shifts on the next rebuild. Appended
# rather than pasted seven times, and applied by an explicit list rather than to every note, because
# M_HEMI / M_VESSEL / M_HEMIDYN / M_FIXEDSCALE read RAW fluorescence and never split on a lick --
# adding it there would warn about a dependency they do not have.
M_GATE = (
    _M_LICK_UNIT +
    "\n\nENGAGED CUT: numbers here predate 2026-08-21. decode.max_rt_s was 2.0 s while the task's "
    "response window is 3.5 s, so trials licking between the two were scored as NO-LICK -- 39.3% of "
    "PS92's no-lick arm and 33.9% of PS93's. The cut is now 3.5 s. Anything on this slide that rests "
    "on the engaged/no-lick split moves on the next rebuild; the FIGURES are current, this prose is "
    "not until re-measured.")

M_EVOKED += M_GATE
M_SPATIAL += M_GATE
M_RECODING += M_GATE
M_NOLICK += M_GATE
M_PRECUE_CAVEAT += M_GATE
M_LICKFREE += M_GATE
M_CODING_DIR += M_GATE

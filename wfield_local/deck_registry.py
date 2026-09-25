"""THE DECK'S FIGURE REGISTRIES — what the analysis deck places, as data rather than control flow.

Split out of `locanmf_analysis_deck` on 2026-09-21, completing the lift that `EPOCH_FIGURES`
started. Every name here is a table of (filename pattern, title, legend) or of arm descriptors;
none of it computes anything and none of it reads the enclosing run. The builder is the thing that
resolves paths and decides what exists -- this file only says what the deck KNOWS ABOUT.

WHY THIS IS A MODULE AND NOT FUNCTION LOCALS. `EPOCH_FIGURES` lived inside `build_analysis_deck`
until 2026-09-21, and the cost was concrete: nothing could import it, so
`scripts/deck_figure_coverage.py` had to REGEX-SCRAPE the source to find out what the deck places.
A scrape cannot see a pattern built by f-string, so the coverage check reported live figures as
orphans and stayed silent about figures the deck had never been told about -- the scheme before it
left 33 of 75 figures unreferenced. `tests/test_epoch_registry.py` can only assert that every
`name=` the renderer emits has an entry BECAUSE the registry is importable.

  ``EPOCH_FIGURES``   109 entries. Section I, the pooled pre/acute/subacute/chronic set from
                      `wfield_local.epoch_grant_figures`. NUMBERED BY POSITION -- the keys used to
                      be written into the data, so inserting a family in narrative order meant
                      renumbering every entry after it, and the cost of not doing that was the 33
                      unreferenced figures above.
  ``GRANT_FIGURES``   Section H, the summary set from `wfield_local.grant_figures`. Placed in
                      LABEL order, not written order -- see `_HPLACE` in the builder for the one
                      family that is deliberately out of it, and why.
  ``ALIGNS``          the three alignment arms (cue / precue / lick) sections D and E iterate.
  ``BASES``           the two independent parcellations section D shows side by side.
  ``NOLICK_BASES``    section D2's basis list, deliberately one entry (see the comment on it).
  ``_*_LEGEND``       shared legend text, formatted per arm. ONE COPY: four hand-copied legends is
                      how two of them end up describing different methods.

SAFE TO MOVE BECAUSE NOTHING HERE READS A LOCAL, checked by AST before the move rather than
asserted after it: every `Name` loaded by each assignment's value resolves to a module-level name
or a builtin. The runtime locals these sat beside (`_grant`, `_epoch`, `src`) stayed in the
function, because they are paths resolved per run.

`scripts/deck_figure_coverage.py` reads this file; keep it in that script's `DECKS` list.
"""
from wfield_local.deck_text import M_FROZEN, M_FROZEN_ENC, M_JOINT

# ---------------------------------------------------------------------------------------------
# THE EPOCH FIGURE REGISTRY, AT MODULE SCOPE SO IT CAN BE IMPORTED
#
# It lived inside `build_analysis_deck` until 2026-09-21, which meant nothing could import it and
# `scripts/deck_figure_coverage.py` had to REGEX-SCRAPE this file to find out what the deck places.
# A scrape cannot see a pattern built by f-string, so the coverage check reported live figures as
# orphans and stayed silent about ones the deck was never told about -- the previous scheme left
# 33 of 75 figures unreferenced, and 25 are unregistered as of today.
#
# IT NEEDS NOTHING FROM THE ENCLOSING SCOPE, which is why the lift is safe: checked by AST before
# moving it, and each assignment was unparsed before and after and compared. The runtime locals it
# sits next to (`_grant`, `_epoch`) stay in the function, because they are paths resolved per run.
# ---------------------------------------------------------------------------------------------

#: Shared legends for the rotation arm. ONE COPY, formatted per arm -- the four `15h` slides
#: and the three `15k` slides differ by a sentence each, and four hand-copied legends is how
#: two of them end up describing different methods.
_ROT_LEGEND = (
    "Haufe patterns for the {arm} window: `A = Cov(X) . beta`, the activity pattern the"
    " decoder`s weights actually read, which is interpretable as anatomy where the raw weights"
    " are not. Rows are spout position, columns the epoch contrast, and each cell is scored"
    " against that component`s OWN split-half null."
    "\n\nEXCESS-OVER-NOISE, NOT RAW CHANGE. The raw |change| map correlates with its own"
    " pre-stroke null at r = +0.83, so it localises the BASIS rather than the lesion."
    "\n\nPATTERNS ARE UNIT-NORMALISED PER POSITION before comparison, so these maps are SHAPE"
    " ONLY -- gain is divided out by construction and lives in the gain-vs-rotation slide."
    "\n\nTHE TEST FAMILY IS MASKED. A component enters only if >= 0.85 of its footprint mass"
    " lies inside `brain_mask` (olfactory bulbs and the painted fibre-glue already removed) AND"
    " >= 0.50 inside the eroded `stat_mask`. Two criteria because there are two failure modes:"
    " the first asks whether it is cortex, the second whether enough of it sits away from the"
    " window rim to be testable. Gating on the eroded mask ALONE -- as every artefact before"
    " 2026-09-17 evening did -- penalises a region for being LATERAL and cost ipsilesional"
    " mouth cortex entirely."
    "\n\nPER POSITION, NEVER AVERAGED OVER THEM. {note}"
    "\n\nTWO LEVELS OF STATISTIC, AND THE SLIDE SHOWS THE FIRST. The per-cell values drawn"
    " here -- cosine, its p against estimation noise, the residual-gain p, the interval --"
    " are all WITHIN ANIMAL, and the map outlines are family-wise corrected across components"
    " within a position. The COHORT statistic is a separate table"
    " (`epoch_15h_rotation_cohort.csv`): a nested animals-to-sessions bootstrap, reported as"
    " an interval and NEVER as a p, because with four animals an animal-level sign-flip null"
    " has 2^4 = 16 assignments and 0.0625 is its floor -- no cell could reach 0.05 at any"
    " effect size. Read a per-animal panel as replication and the cohort table as the test."
    "\n\nACROSS CELLS THE CORRECTION IS A PERMUTATION MAX-STATISTIC. It takes each draw`s largest"
    " cohort z across the family and thresholds at the 95th percentile of those maxima, so it"
    " assumes nothing about dependence between cells -- it measures it. The family is WINDOW x"
    " CONTRAST x POSITION = 72; animals are replicates of one hypothesis and are deliberately NOT"
    " in it. Calibration checked: 5.0% of null draw-maxima sit above the threshold it returns,"
    " which is what exactness means here."
    "\n\nTHE REASON FIRST GIVEN FOR PREFERRING IT DID NOT SURVIVE MEASUREMENT. The argument was that"
    " the six positions are strongly coupled -- multinomial coefficients are identified only up to"
    " a constant shift across classes -- so Bonferroni would be badly conservative. Measured over"
    " the 72 cells, the mean pairwise correlation between their per-draw statistics is +0.077: the"
    " cells are very nearly INDEPENDENT. The max-statistic is still the right choice, because it"
    " needs no independence assumption and is exact by construction -- but it is a safeguard here"
    " rather than the power rescue it was advertised as."
    "\n\nAND DO NOT READ THE z MAGNITUDES AS EFFECT SIZES. The null is the cosine between two"
    " PRE-STROKE HALVES, which is high and tightly clustered, so any real cross-epoch change lands"
    " many null-SDs away and the observed cohort z runs to +47. That says the readout direction"
    " moves by more than split-half estimation noise -- which the noise-ceiling bars already show."
    " The discriminating content is the cosine MAGNITUDE and its ordering, never the p."
    "\n\nAN EARLIER VERSION OF THIS ARM REPORTED THAT NOTHING SURVIVED CORRECTION. That was"
    " the DRAW COUNT, not the effect sizes -- an empirical p cannot go below 1/(draws+1), so"
    " at 200 draws no cell could clear a Bonferroni alpha whatever the data said. A"
    " max-statistic has no such floor; it needs only enough draws to place one percentile."
    "\n\nPer-cell cosines, gains, intervals and noise ceilings are in"
    " `epoch_15h_rotation_regions.csv`, the cohort intervals and family-wise calls in"
    " `epoch_15h_rotation_cohort.csv`; the figures redraw from a component-space cache in"
    " seconds via `--replot`."
)
_REF_LEGEND = (
    "Rows are epoch, columns are the reference. Each cell is the COHORT delta (epoch minus"
    " pre-stroke) for one Allen region x spout position, from the nested animals -> sessions"
    " bootstrap. DOT = that interval excludes zero, per cell and UNCORRECTED across regions."
    " RING = every reference agrees. Last column is that agreement SIGNED: blue = decrease,"
    " red = increase, darker = more references; an x marks the rare cell whose significant"
    " references disagree in DIRECTION, which is never averaged into a direction it lacks."
    "\n\nTHE ARGUMENT IS THE DISAGREEMENT. Each reference is wrong in its own way and the"
    " ways do not overlap. `raw` has no subtrahend, so nothing in it can drift between epochs"
    " -- it carries cross-day multiplicative scaling instead, which no subtraction removes"
    " either. `precue` subtracts a PER-TRIAL baseline so it cannot drift, but it violates F12:"
    " the pre-cue window carries real anticipatory position signal, so it measures the"
    " cue-evoked INCREMENT rather than the position map. `restw` keeps positions independent"
    " but its baseline MOVES across epochs, by the amount `epoch_15j` measures. An effect in"
    " ALL of them is not a property of any one subtrahend; an effect in ONE names the"
    " subtrahend to suspect."
    "\n\nCOLOUR SCALES DIFFER BETWEEN COLUMNS ON PURPOSE, because the three are not in the"
    " same units. Epochs WITHIN a column are comparable; amplitudes ACROSS columns are not."
    "\n\nTHE UNIT IS A LocaNMF COMPONENT GROUPED BY ITS ALLEN LABEL, never a pixel mean over"
    " the area -- MOs is 15,613 px and a focal change in part of it is swamped by the rest."
    " Components do not correspond across animals (95/87/90/95) but their Allen labels do, and"
    " the vocabulary is the INTERSECTION, so no region`s cohort mean rests on a different"
    " subset of animals than its neighbour`s."
    "\n\nTHE CLAIM IS AN INTERVAL, NEVER A p. With four animals an animal-level permutation"
    " has 2^4 = 16 assignments and a floor of 0.0625, so no cell could reach 0.05 at any"
    " effect size. The guard is agreement across references, not a threshold."
    "\n\nThis arm is {arm}."
    "\n\nCROSS-CHECKED AGAINST `15r`, WHICH SHARES NOTHING WITH THIS FAMILY BUT THE TRIALS AND THE"
    " REST BASELINE. 15r tests ~2,022 PIXEL bins with a bootstrap max-statistic over bins; this"
    " family tests 33 ALLEN REGIONS built from footprint-weighted LocaNMF components, with a nested"
    " animals-to-sessions CI gated on three references. Different unit, different estimator,"
    " different multiplicity treatment -- so agreement is evidence rather than bookkeeping."
    "\n\nTHEY AGREE ON THE PANEL AND ON THE PLACE. Per-panel counts correlate at Spearman +0.810"
    " (lick) and +0.916 (cue). Spatially, a region flagged by ALL THREE references has a median"
    " ~52% of its Allen footprint inside 15r`s significant bins against 3-5% for a region not"
    " flagged -- a within-panel difference of +0.564 (lick) and +0.506 (cue), p = 0.0001 at 10,000"
    " draws. The null shuffles the agreed label WITHIN panel, holding blob size and the number of"
    " agreed regions fixed, so location is the only thing left to explain it."
    "\n\nREAD THE LICK ARM AS THE LOAD-BEARING ONE. The cue acute panel is ~75% GLOBAL, and a"
    " whole-cortex shift agrees with any anatomical claim; the lick arm is 11-31% global and gives"
    " the LARGER effect. Numbers in `epoch_15kr_cross_check.csv`."
)
_CCF_LEGEND = (
    "The same result as the preceding matrix, painted on the brain: rows are epoch, columns"
    " are spout position, and a region is coloured only where the cohort interval excludes"
    " zero in EVERY reference. Colour is the `restw` value -- one reference supplies the"
    " number because the references are not in the same units and averaging them would invent"
    " a quantity none of them measures."
    "\n\nTHE FLAT COLOUR INSIDE A REGION IS THE RESULT, NOT A RENDERING SHORTCUT. The unit"
    " is a LocaNMF component mean spread over its Allen footprint, so the sharp edges are the"
    " parcellation`s, not the data`s. Read it beside `epoch_15r`, whose blobs are pixel-level"
    " and whose edges ARE data -- agreement between two different test units and two different"
    " correction families is much stronger than either alone."
    "\n\nDashed line is the analysed mask. Pale grey means the references agreed on nothing"
    " there; a column labelled NO TRIALS IN ANY ANIMAL is a different fact entirely and is"
    " labelled rather than left as anonymous grey."
    "\n\n`_left` IS THE ANIMAL`S LEFT, verified rather than assumed: pre-stroke, a"
    " right-side spout drives every `_left` area harder and every `_right` area less, across"
    " six independent area pairs (11 of 11). Every lesion in this cohort is LEFT-sided, so"
    " LEFT is the IPSILESIONAL hemisphere AND the one representing the impaired right side."
    "\n\nThis arm is {arm}-aligned."
)
#: Section J -- the CD TRAJECTORIES, per animal (`wfield_local.cd_trajectories`).
#:
#: ONE ENTRY PER (layout, alignment), with `*` standing for the animal, so a new animal needs no
#: registry edit. The builder pulls the animal out of the filename for the slide title.
#:
#: THE NOTES ARE WRITTEN AS STANDALONE FIGURE LEGENDS, the rule section I already follows: a reader
#: should be able to lift one into a proposal and have it stand alone. For this family that matters
#: more than usual, because **the naive reading of a one-vs-rest CD figure is wrong** -- the six
#: directions nearly cancel, so a signal common to every trial is FORCED to split sign across
#: positions, and the dramatic position differences in an unorthogonalised panel are that geometry
#: rather than the code. A caveat that lives only in the deck does not travel with the figure.
_CD_METHOD = (
    "METHOD. For each spout position P a coding direction is fitted on PRE-STROKE successful-lick "
    "trials as w_P = mean(P) - mean(not-P) in the animal's frozen joint-LocaNMF component space, "
    "using the WINDOW MEAN over a 2 s feature window (measured optimum: a 1 s window costs 28% of "
    "d' in PS92, and 3 s keeps only 458 of 3911 trials because the lick-free gate has no slack). "
    "The direction is therefore TIME-INVARIANT, which is what lets the signal be projected onto it "
    "frame by frame to give a trajectory; `position_coding_directions` fits in (component x "
    "sub-bin) space and its vector cannot be applied to a single frame. Components are Z-SCORED in "
    "a frozen pre-stroke frame -- per-component sd spans 107x and the top five hold 82% of the "
    "variance, so an unstandardised difference of means is set by about five components on "
    "amplitude alone. Trials are averaged by MEAN and the band is the 95% CI OF THE MEAN, not the "
    "trial spread; the per-trial projection is strongly skewed, so a median reads below zero at "
    "three of six positions for pre-stroke success alone. Display smoothing is a 0.2 s CENTRED "
    "boxcar (costs 4% of peak; 0.4 s costs 8%, and the cost falls on the fast far-position "
    "transients, not the slow close ones). Axis: 0 = pre-stroke not-P, 1 = pre-stroke lick at P. "
    "HAEMODYNAMICS ARE SLOW -- read amplitude and gross time course, never onset or ordering.")

_CD_ORTH = (
    "WHY THE CONDITION-INDEPENDENT MODE IS PROJECTED OUT. The directions are one-vs-rest, so across "
    "six positions the unit vectors SUM TO A VECTOR OF LENGTH 0.289 where six aligned ones would "
    "give 6.0. A signal common to every trial therefore cannot load positively on all six: the "
    "geometry FORCES it positive on some and negative on others. The lick response is such a signal "
    "and it is large -- decomposed on PS95 pre-cue, close_center reads +14.08 of which +14.32 is "
    "the shared term and -0.24 is position-specific, and the position-specific part is ~1-2 at "
    "EVERY position. So the striking position differences in an unorthogonalised panel are an "
    "artefact of the cancellation geometry. The top-K components of the pre-stroke grand-mean "
    "trajectory are projected out (K by variance explained, 2-3 here); K=1 is not enough because a "
    "SUBSPACE rotates where a single vector appears not to (K=1 cosines 0.88/0.95/0.88 against K=2 "
    "subspace overlaps 0.68/0.61/0.66). The mode is fitted on PRE-STROKE sessions and applied to "
    "every epoch -- a per-epoch mode would subtract away the change being measured. "
    "TRACES ARE SCALED BY THE UNROTATED GAP, so a direction that barely survives the projection "
    "draws SMALL rather than being renormalised back to 1; each panel prints its surviving "
    "fraction, RED below 70%, and a panel below that should not be read.")

_CD_MASK = (
    "COMPONENT SELECTION. Components whose spatial mass is more than half inside that animal's "
    "hand-painted fibre-glue mask or an olfactory bulb are DROPPED -- about a quarter of every "
    "basis (19-24 of 87-95). The map analyses already exclude that territory and LocaNMF does not. "
    "Measured before adopting it: the direction loads on those components at their share BY COUNT, "
    "which is the null expectation under equal-variance weighting, and the real fit comes in at or "
    "below it; dropping them costs 9-14% of the pole gap, which is what dropping any random quarter "
    "costs. The reason to drop is not contamination but that KEEPING them leaves the post-stroke "
    "panels sensitive to the choice -- PS95 acute close_center reverses sign. `--occluded keep` "
    "reproduces the earlier arm.")

CD_FIGURES = (
    ("cd_epochs_*_precue_dom_contrast_lick_orth_cortexonly.png",
     "Pre-cue coding direction, four epochs overlaid",
     "Projection of pre-cue population activity onto each spout position's own coding direction, "
     "one panel per position, with the four recovery epochs OVERLAID so 'did this position's code "
     "change after the lesion' is one comparison rather than four. Trials are successful licks; "
     "x = 0 is the CUE and the direction is fitted on the 2 s BEFORE it, i.e. the enforced-no-lick "
     "period, on trials with no lick anywhere in that window. A dashed trace is a cell with fewer "
     "than 10 trials -- drawn rather than omitted, because 'could not test' and 'tested and found "
     "nothing' are different facts. Y-AXIS SHARED ACROSS PANELS: the pole normalisation makes the "
     "six commensurable, so a per-panel scale would make positions of different amplitude look "
     "alike. " + _CD_ORTH + " " + _CD_METHOD + " " + _CD_MASK),
    ("cd_epochs_*_cue_dom_contrast_lick_orth_cortexonly.png",
     "Cue-aligned coding direction, four epochs overlaid",
     "As the pre-cue panel, with the direction fitted on the 2 s AFTER the cue instead of before "
     "it. NOTE that both are drawn against the cue -- the alignment token selects which window the "
     "DIRECTION is fitted on, not where the trace is centred -- so this is not an independent "
     "replication of the pre-cue figure, and the two agreeing is expected. " + _CD_ORTH + " "
     + _CD_METHOD + " " + _CD_MASK),
    ("cd_epochs_*_lick_dom_contrast_lick_orth_cortexonly.png",
     "Lick-aligned coding direction, four epochs overlaid",
     "As above, centred on the FIRST LICK and fitted on the 2 s after it. This is the independent "
     "alignment: no-lick trials have no lick to align to, so the gate is forced back to successful "
     "trials whatever was asked for -- a trajectory through an inferred lick time would have an "
     "x-axis that stretches with the latency, and post-stroke the latency is long and variable. "
     + _CD_ORTH + " " + _CD_METHOD + " " + _CD_MASK),
    ("cd_cross_*_precue_dom_contrast_lick_orth_cortexonly.png",
     "Pre-cue CROSS-projection: every position's trials on every position's direction",
     "Rows are recovery epochs, columns are the coding direction being projected onto, and each "
     "panel overlays all six spout positions' trials. THE HEAVY TRACE IS THE PANEL'S OWN POSITION; "
     "the other five are the comparison, so a selective direction shows one trace rising and five "
     "flat. That is the claim 'this is a position code' made visible rather than inferred from a "
     "decoder score. Y is shared WITHIN a row, since the question here is within-panel "
     "selectivity. It costs nothing extra to compute: the courses already exist for all six "
     "directions over the whole session, so slicing any position's trials out of any course is "
     "free. " + _CD_ORTH + " " + _CD_METHOD + " " + _CD_MASK),
    ("cd_cross_*_cue_dom_contrast_lick_orth_cortexonly.png",
     "Cue-aligned CROSS-projection", "As the pre-cue cross-projection, direction fitted after the "
     "cue. " + _CD_ORTH + " " + _CD_METHOD + " " + _CD_MASK),
    ("cd_cross_*_lick_dom_contrast_lick_orth_cortexonly.png",
     "Lick-aligned CROSS-projection", "As above, centred on the first lick. " + _CD_ORTH + " "
     + _CD_METHOD + " " + _CD_MASK),
    ("cd_cim_geometry_*.png",
     "The condition-independent mode after stroke: orientation, magnitude, and what orthogonalising leaves behind",
     "Three rows, and the third is the only one that licenses a claim about a post-stroke panel. "
     "(A) SUBSPACE OVERLAP of each epoch's condition-independent subspace with the pre-stroke one, "
     "drawn against its CHANCE LEVEL of K/n -- about 0.02 on a 95-component basis, shaded. An "
     "overlap of 0.6 is therefore ~30x chance and the mode is strongly CONSERVED; reading it "
     "against 1.0 instead, as 'a third rotated away', is wrong. (B) MAGNITUDE of the same mode as a "
     "ratio to pre-stroke, because overlap is SCALE-INVARIANT: a response that keeps its "
     "orientation exactly and halves in size scores 1.00, which for a lesion study is a blind spot "
     "on the most likely effect. MEASURED, the shared response is LARGER after the lesion in 12 of "
     "12 (animal, epoch) cells, mean 1.41, and it replicates on the lick alignment. (C) the "
     "UNREMOVED shared amplitude, sqrt(1 - overlap) x scale, in units of the pre-stroke shared "
     "response; the dashed line is the same quantity with the scale assumed to be 1.0, i.e. what "
     "row B corrects. Read A and B TOGETHER or neither: same subspace with a smaller scale is "
     "weaker drive, a rotated subspace at the same scale is REORGANISATION, and a decoder score "
     "conflates them because both lower accuracy. PS94 chronic is the only cell with overlap DOWN "
     "and scale UP in all three alignments, and its row-C value exceeds 1.0 -- more unremoved "
     "shared signal than the whole pre-stroke shared response, against a position-specific signal "
     "of ~1-2, so that panel is not readable. THE OVERLAP HAS NO MATCHED NULL YET: two subspaces "
     "estimated from different sessions do not fully overlap even when nothing changed, and "
     "`wfield_local.cd_overlap_null` builds the trial-matched pre-to-pre comparison that gives "
     "these numbers a scale. Until it has run, read A against K/n only."),
)


EPOCH_FIGURES = (
    ("epoch_1c_behaviour_timecourse.png",
     "Behavioural deficit and recovery, and where the epochs come from",
     "Licking accuracy at each of six spout positions across days from lesion, pooled"
     "across four mice (N = 4, n = 74 sessions). Each point is one session of one animal;"
     "colour identifies the animal. The pre-stroke baseline is that animal's entire pre-"
     "lesion record collapsed to a single point (hits and trials summed), which is the"
     "value the epoch rule is measured against. Days are counted from each animal's OWN"
     "lesion, since PS94/PS95 and PS92/PS93 were lesioned on different dates. Shaded span,"
     "the ACUTE epoch, defined as days on which far-contralateral accuracy is below 25% of"
     "that animal's pre-stroke baseline; dotted line, each animal's first SUBACUTE day."
     "Accuracy falls at the far positions, most severely at far contralateral, and recovers"
     "at animal-specific rates -- which is why the analyses that follow are stratified by"
     "epoch rather than by day."),
    ("epoch_1b_behaviour_by_position.png",
     "Behaviour by position and epoch",
     "Licking accuracy per spout position, pooled across four mice and stratified by"
     "recovery epoch. Bars, session-weighted means; dots, individual sessions coloured by"
     "animal, so the imbalance between epochs is visible rather than asserted (the acute"
     "epoch is six PS94 sessions against one PS95 session). Error bars, 95% intervals from"
     "a hierarchical bootstrap resampling animals, then sessions within animal, then the"
     "scheduler's position blocks within session. *, the epoch-minus-pre interval excludes"
     "zero; **, it still excludes zero after Bonferroni correction across all twelve"
     "comparisons. Trials are gated for engagement at the SPARED positions only, so a run"
     "of misses at an impaired position counts as the deficit it is rather than being"
     "discarded as disengagement."),
    ("epoch_1bdelta_behaviour_by_position.png",
     "Behaviour, change from pre-stroke",
     "The contrasts of the preceding figure plotted as differences from the pre-stroke"
     "baseline. Point, the difference in the data; thick bar, 95% interval; thin bar, the"
     "Bonferroni-corrected interval -- both from one set of bootstrap draws, so the"
     "corrected interval necessarily contains the uncorrected one. Zero is drawn: the"
     "comparison that matters is each epoch against no change, not the epochs against each"
     "other."),
    ("epoch_1f_engagement_by_epoch.png",
     "Was the animal still working? Engagement by epoch",
     "The DENOMINATOR behind every behavioural number in this section, plotted in its own"
     " right. WORKING is the behaviour gate`s `engaged` flag, judged ONLY at the reference"
     " positions the lesion spares (near ipsi, near middle), so a run of misses at an impaired"
     " position can never lower it. That asymmetry is deliberate: reward is auto-held after a"
     " miss run, so a sated animal`s late misses are disengagement, while an impaired animal`s"
     " far-position misses are the effect being measured."
     "\n\nTHERE IS NO POSITION AXIS, AND ITS ABSENCE IS THE POINT. Because the gate reads only the"
     " spared positions, engagement is a SESSION-level property applied to every trial in that"
     " session; positions run in ~6-trial blocks, so a terminal stopped tail truncates all six"
     " about equally. The per-position panel this replaces was flat BY CONSTRUCTION -- measured,"
     " an acute spread of 0.031 across the six against an epoch effect of 0.18 -- and six bars"
     " invite a spatial reading the design cannot produce."
     "\n\nWHAT IT SHOWS. Stopped fraction 0.041 pre, 0.218 acute, 0.186 subacute, 0.069 chronic:"
     " post-stroke animals quit far earlier, about 5x pre acutely. Corroborated a completely"
     " different way -- the quit period is 3.1% of REST FRAMES pre-stroke and 18.7% acutely"
     " (docs/REST_ENGAGEMENT_AUDIT.md) -- and the worst sessions are severe (PS94 day 3, 62%)."
     "\n\nTHE DISSOCIATION WORTH NOTICING. Subacute recovers on ACCURACY (near-position response"
     " rates 0.96-0.97, near baseline) while engagement is still close to its acute value. Since"
     " accuracy is engagement-GATED, both describe the same trials: subacute animals are accurate"
     " WHEN WORKING and still stop working early. Carry the standing subacute caveat -- that bin"
     " is a RESIDUAL rather than a fixed interval, and it is PS94-weighted, 7 sessions to PS92`s 2."
     "\n\nSource is the behaviour pipeline`s own `cohort_session_metrics.csv` (`n_engaged` /"
     " `n_disengaged`), the same gate the epoch rule uses, so this cannot disagree with the other"
     " behaviour figures about what working means. Bars, session means; dots, individual sessions"
     " coloured by animal; intervals from the nested animals->sessions bootstrap."),
    ("epoch_1g_engagement_timecourse.png",
     "Engagement over days from lesion, one panel per animal",
     "The sessions behind the pooled bars, unpooled, against days since that animal`s OWN lesion."
     " Square, the pre-stroke mean with its range; line, the post-stroke sessions."
     "\n\nWHY UNPOOLED. The epoch BINS are animal-specific -- acute is a fraction of each animal`s"
     " post-stroke days and chronic starts when THAT animal`s hit rate flattens -- so PS94`s"
     " subacute runs to day 29 where PS92`s is days 7-9. A pooled engagement number therefore"
     " averages bins that do not mean the same thing across animals, and this panel shows what was"
     " averaged."
     "\n\nWHAT IT SHOWS THAT THE BARS CANNOT. Recovery is not uniform. PS92 and PS93 dip hard early"
     " (0.59 and 0.53 around day 3) and are back at ceiling by day 7-9. PS94 never settles --"
     " it oscillates between 0.37 and 1.0 through day 25, which is the same non-plateau that leaves"
     " it with NO chronic epoch. PS95 recovers and then drops again at day 25. An animal-level"
     " reading of the subacute bin should start here rather than at the bar."
     "\n\nBuilt by `scripts/engagement_by_epoch.py` from the behaviour pipeline`s own"
     " `cohort_session_metrics.csv`, the same gate the epoch rule uses."),
    ("epoch_acc_by_position_*_*.png",
     "Position decoding by epoch",
     "Accuracy of a decoder trained ONLY on pre-stroke trials, applied to held-out trials"
     "in each epoch, per spout position and pooled across animals; chance 1/6. Bars pool"
     "over trials -- raw confusion counts sum, so every trial counts once -- while dots are"
     "individual sessions. Intervals and marks as above. One figure per alignment window"
     "and trial class. FIVE ARMS as of 2026-09-12: the ENL (pre-cue) and post-cue windows each"
     "render twice, once on lick plus miss-while-working and once on LICK TRIALS ONLY, and the"
     "lick-aligned window admits lick trials only because a trial with no detected lick has no"
     "lick to align to. The lick-only pre-cue and post-cue arms are the SELECTION CONTROL for"
     "the post-lick result: they hold the trial set fixed and move only the window, so a"
     "finding that survives them is not an artefact of which trials the animal completed."
     "A sixth and seventh arm (the terminal quit period, pre-cue and post-cue) were RETIRED on"
     "2026-09-12 -- their pre-stroke reference was lick trials, so they compared post-stroke"
     "quitting against pre-stroke engaged cortex; the state-matched question is asked properly"
     "by the 12/12b family."),
    ("epoch_accdelta_by_position_*_*.png",
     "Decoding accuracy, change from pre-stroke",
     "Epoch minus pre-stroke at each position, with 95% and corrected intervals as above."
     "Note that the ENL (pre-cue) deficit is BROAD -- every position falls, not only the"
     "impaired ones -- while the behavioural deficit is spatially specific. The pre-cue"
     "position code degrades globally after the lesion even where the animal can still"
     "reach the spout."),
    ("epoch_5r_refit_by_position_*_*.png",
     "Refit within session: is the information still there?",
     "Per-position accuracy of a decoder REFITTED inside each session -- same trials, same"
     "estimator, same block grouping as the frozen decoder two figures up, differing only in"
     "what it was trained on. Five-fold block cross-validation within the session, so every"
     "prediction is out of sample. Bars, session-weighted means with 95% hierarchical"
     "bootstrap intervals; dots, individual sessions coloured by animal; chance 1/6. Read"
     "against the frozen panel: a position the refit decoder recovers is a position whose"
     "information survived and whose pre-stroke readout no longer points at it."),
    ("epoch_5rdelta_refit_by_position_*_*.png",
     "Refit decoding, change from pre-stroke",
     "The refit arm's epoch-minus-pre contrasts. Point, the difference in the data; thick"
     "bar, 95% interval; thin bar, the Bonferroni-corrected interval across all twelve"
     "comparisons, from one set of draws. A refit decoder that ALSO falls is measuring loss"
     "of information rather than loss of readout."),
    ("epoch_5rgap_frozen_vs_refit_*_*.png",
     "Recoverable information: refit minus frozen",
     "The paired difference between the two preceding arms, per position and epoch. Both"
     "arms score the SAME trials -- the two predictions are two columns of one record -- so"
     "the difference is paired at the trial level and stays paired through every level of"
     "the bootstrap. A positive gap means the position is decodable within the session but"
     "not by the pre-stroke model: information PRESENT and DISPLACED. A gap near zero with"
     "both arms low means no model recovers the position: information DEGRADED. THE PRE BAR"
     "IS NOT ZERO BY CONSTRUCTION and is not an effect: the frozen arm trains on ten"
     "pre-stroke sessions and the refit arm on one, so a gap exists at baseline from"
     "training-set size alone, with no lesion involved. Solid line, zero. The whole-session,"
     "per-animal version of this comparison is H3b; this is its per-position,"
     "epoch-stratified form, and the only one carrying an interval on the DIFFERENCE."),
    ("epoch_5rgapdelta_frozen_vs_refit_*_*.png",
     "Recoverable information, change from the pre-stroke gap",
     "The gap at each epoch minus the gap at pre -- the figure that carries the claim, because"
     "it is the only one from which the training-set-size handicap has been subtracted. An"
     "interval excluding zero means the lesion made MORE information recoverable by refitting"
     "than the design's own handicap accounts for, which is the signature of a displaced code."
     "Marks and correction as in the other contrast panels."),
    ("epoch_5rmgap_frozen_vs_refit_*_*.png",
     "Recoverable information, training-set MATCHED",
     "THE BASELINE DOES NOT GO TO ZERO WHEN THE TRAINING SETS ARE MATCHED -- IT GOES THE OTHER"
     "WAY. Unmatched, the frozen arm has ten sessions against the refit arm's four fifths of"
     "one, and refitting COSTS 0.073 pre-stroke (post-cue). Matched, refitting is +0.090"
     "BETTER pre-stroke, because a model trained inside a session shares that session's own"
     "nuisance structure -- alignment, haemodynamics, arousal, the LocaNMF projection -- while"
     "the matched frozen model has to generalise across days. So the no-lesion baseline is"
     "BRACKETED, -0.073 to +0.090, and neither bound is 'the' answer: one measures training-set"
     "size, the other cross-session generalisation. Both are read as epoch-minus-pre, and the"
     "headline survives either way -- far-contralateral carries the largest lesion-attributable"
     "recoverable component, +0.196 unmatched and +0.158 matched."),
    ("epoch_5rmgapdelta_frozen_vs_refit_*_*.png",
     "Recoverable information (matched), change from the pre-stroke gap",
     "The matched companion to the unmatched contrast panel, and the more conservative of the"
     "two: matching costs the frozen arm more at pre-stroke (where extra training data helps)"
     "than acutely (where it fails regardless), so the acute contrast shrinks from +0.100 to"
     "+0.039 pooled. Per position the ordering is unchanged and far-contralateral is still"
     "the largest, +0.158."),
    ("recovery_trajectory_*_*.png",
     "THE ROUTE, not the endpoint: readout deficit against reorganisation",
     "THE ONE FIGURE IN THIS SECTION THAT IS NOT A BAR CHART, and it asks a different question."
     "Every other panel asks WHERE a quantity ended up in each epoch; this asks what PATH it"
     "took between them. Each point is one post-stroke session, on two axes built from the same"
     "paired frozen/refit record the 5r family uses: F (x) = pre_frozen - frozen, how much the"
     "PRE-STROKE READOUT lost; G (y) = (refit - frozen) - (pre_refit - pre_frozen), how much of"
     "that a within-session refit recovers over and above the gap that already existed before"
     "the lesion. Both are signed so the lesion moves them POSITIVE, which is why they differ"
     "from the 5ro contrast panels -- those are epoch-minus-pre like every other contrast here,"
     "so their frozen arm goes negative. Same quantities, opposite convention, said plainly"
     "because the two figures sit near each other."
     "\n\nREAD THE DIAGONAL. F - G is algebraically pre_refit - refit, so the vertical distance"
     "from a point DOWN to the unity line is the REFIT arm's own deficit -- the information no"
     "decoder recovers. ON the line, the whole frozen deficit is readout mismatch and the code"
     "is displaced but intact; BELOW it, some of the code is genuinely gone; ABOVE it, the"
     "session decodes better than it did before the lesion. That line is the displaced/lost"
     "distinction this family exists to draw, and it is why the figure is a scatter rather than"
     "two time courses."
     "\n\nTHE ORIGIN IS A POINT ESTIMATE, NOT A FIXED MARK. Pre-stroke sessions are drawn"
     "faintly with their mean and +/-1 SEM on both axes, because without that scatter 'has it"
     "returned?' has no scale: a chronic point 0.05 from the origin is a full return or a"
     "residual deficit depending entirely on how far pre-stroke sessions sit from their own"
     "mean."
     "\n\nTWO FAMILIES, matched and unmatched, for the reason the 5r/5rm pair exists: the"
     "unmatched frozen arm trains on ten pre-stroke sessions against the refit arm's fraction"
     "of one, so G carries a training-set handicap at baseline and the matched family removes"
     "it. Read them together; the route should not depend on which bound is used."
     "\n\nNO INTERVALS ON THIS FIGURE. A session is a point, not a distribution, and the"
     "bootstrap every bar panel carries has no meaning on a trajectory. Use 5ro/5rmo for the"
     "tested statement and this for the shape of the path."),
    ("recovery_trajectory_byanimal_*_*.png",
     "The same route, one panel per animal",
     "THE POOLED SCATTER HIDES WHICH ANIMAL DREW THE PATH, and with four animals that matters:"
     "83% of the squared standard error at chronic is BETWEEN animals, so a pooled trajectory"
     "can be a shape no individual animal traced. Each panel carries that animal's own"
     "pre-stroke cloud, its own unity line and its own day-ordered sequence, so a reader can see"
     "whether the cohort route is four similar paths or an average of dissimilar ones."
     "\n\nThe square is the FIRST post-stroke session and the star the last. RETRACE versus"
     "MIGRATE is read off the shape: a path that returns along its outbound direction"
     "re-establishes the pre-stroke arrangement, one that returns by a different route arrives"
     "at similar accuracy through a different code."),
    ("epoch_5ro_frozen_refit_overall_*_*.png",
     "Frozen and refit accuracy POOLED OVER POSITIONS, by epoch",
     "THE PANEL THE POWER ANALYSIS SAYS TO LEAD WITH, and it exists because the per-position"
     "family spends five sixths of its trials on a resolution this cohort cannot support at"
     "chronic. Pooling over positions is what makes the interval narrower than the effect."
     "Three bars per epoch -- the FROZEN pre-stroke decoder, the WITHIN-SESSION refit, and"
     "their paired difference -- with one dot per session coloured by animal, and the"
     "epoch-minus-pre companion beneath. Both arms score the SAME trials, so the gap is paired"
     "at the trial level and stays paired through every level of the bootstrap."
     "\n\nSIGN CONVENTION, AND IT DIFFERS FROM THE RECOVERY-TRAJECTORY FIGURES ON PURPOSE. The"
     "delta panel is epoch MINUS pre, as every other contrast panel in this deck is, so the"
     "frozen arm goes NEGATIVE when the readout is worse. `recovery_trajectory`'s F is the same"
     "quantity with the sign flipped so that both of its axes move positive with the lesion."
     "Its G is this figure's gap arm, unflipped. Two conventions is one more than ideal; the"
     "alternative was a contrast panel whose bars point the opposite way from every other one."),
    ("epoch_5rodelta_frozen_refit_overall_*_*.png",
     "Pooled frozen/refit accuracy, change from pre-stroke",
     "The same three arms as epoch minus pre. READ THE FROZEN ARM FIRST: it crossing back"
     "toward zero at chronic is the recovery claim, and it is the one statement in this family"
     "that survives every version of the training-set argument. The gap arm here still carries"
     "the unmatched handicap -- the frozen model trains on ten pre-stroke sessions and the"
     "refit on four fifths of one -- so read it against the MATCHED companion two slides on,"
     "not against zero."),
    ("epoch_5rmo_frozen_refit_overall_*_*.png",
     "Pooled frozen and refit accuracy, training-set MATCHED",
     "THE SAME POOLED CONTRAST WITH BOTH ARMS GIVEN THE SAME AMOUNT OF TRAINING DATA, and the"
     "cleanest single statement of the headline result. The frozen arm is now handicapped the"
     "way the refit arm always was, so the two are comparable and the pre bar means something."
     "\n\nTHE NO-LESION BASELINE IS BRACKETED, NOT KNOWN, and this family is one of its two"
     "bounds. Unmatched, refitting COSTS 0.073 pre-stroke because the frozen model has ten"
     "times the data; matched, refitting is +0.090 BETTER pre-stroke, because a model trained"
     "inside a session shares that session's own nuisance structure -- alignment,"
     "haemodynamics, arousal, the LocaNMF projection -- while the matched frozen model must"
     "generalise across days. Both bounds are real effects and neither is 'the' answer, which"
     "is why both families are drawn and both are read as epoch-minus-pre."),
    ("epoch_5rmodelta_frozen_refit_overall_*_*.png",
     "Pooled frozen/refit accuracy (matched), change from pre-stroke",
     "THE CONTRAST PANEL TO CITE. It is the pooled, training-set-matched, epoch-minus-pre form"
     "of the whole frozen-versus-refit argument: the handicap is removed by matching, the"
     "position axis is removed by pooling, and what remains is the lesion. The frozen arm"
     "recovering toward zero by chronic is the claim; the gap arm is the recoverable component"
     "that recovery did not account for. A mark here means the interval excludes zero"
     "uncorrected, two marks that it survives Bonferroni, exactly as elsewhere."
     # QUOTED FROM THIS SLIDE'S OWN SIDECAR, not typed in. `SELF` is the figure placed here, so
     # each trial-class arm prints its own numbers from one caption -- and a re-render moves
     # these with the figure instead of leaving the prose behind. See `deck_values`.
     "\n\nON THIS ARM: the frozen deficit is"
     " {{SELF: epoch=acute, position=frozen -> point:+.3f}}"
     " [{{SELF: epoch=acute, position=frozen -> lo95:+.3f}},"
     " {{SELF: epoch=acute, position=frozen -> hi95:+.3f}}] acutely and"
     " {{SELF: epoch=chronic, position=frozen -> point:+.3f}}"
     " [{{SELF: epoch=chronic, position=frozen -> lo95:+.3f}},"
     " {{SELF: epoch=chronic, position=frozen -> hi95:+.3f}}] by chronic -- the recovery."
     " The gap arm is {{SELF: epoch=acute, position=gap -> point:+.3f}} acutely and"
     " {{SELF: epoch=chronic, position=gap -> point:+.3f}} at chronic."),
    ("epoch_5cr_refit_confusion_*_*.png",
     "WITHIN-SESSION REFIT decoder, confusion by epoch",
     "THE SAME PANEL FOR THE REFIT DECODER, and it is read against the frozen one immediately"
     "before it. The 5r family reduces the refit decoder to a per-position ACCURACY -- the"
     "diagonal -- and the whole reason the frozen family draws a confusion is that the"
     "diagonal is not the interesting part: WHERE the errors go is. Three readings. If 5c goes"
     "off-diagonal and this one restores the diagonal, the code is INTACT and the pre-stroke"
     "readout is pointing at the wrong place -- displacement. If both go off-diagonal in the"
     "SAME cells, that position is genuinely confusable with that neighbour and no readout"
     "recovers it -- degradation. If this one goes off-diagonal in DIFFERENT cells, the"
     "within-session structure has reorganised rather than weakened. THE TWO FAMILIES ARE NOT"
     "ON THE SAME FOOTING: the frozen arm trains on ten pre-stroke sessions and this one on"
     "four fifths of ONE, so its pre panel is already worse with no lesion involved. Read each"
     "family against ITS OWN pre column -- which is what the delta row beneath does -- and"
     "never a cell here against a cell there. Sessions the refit could not be fitted on are"
     "ABSENT rather than zero, so an epoch here can rest on fewer sessions than the same epoch"
     "in 5c; the per-animal counts are in the panel titles."),
    ("epoch_5c_frozen_confusion_*_*.png",
     "Frozen pre-stroke decoder, confusion by epoch",
     "Confusion matrices of the frozen pre-stroke decoder pooled across animals within each"
     "epoch (top row), and each epoch's change from pre-stroke (bottom row, placed beneath"
     "the epoch it describes). Rows, true position; columns, decoded position. Panels are"
     "row-normalised for display while the stored matrices remain raw counts, which is what"
     "makes pooling a sum. BOTH differences use the pre-stroke baseline and share one"
     "colour scale, so they are directly comparable -- the question a recovery figure is"
     "asked is whether the code returned to baseline, not whether it improved on its worst"
     "point. Panel titles carry n and the per-animal session counts."),
    ("epoch_6_matrices_pattern_*_*.png",
     "Mean-pattern similarity by epoch",
     "Correlation between each session's mean activity pattern at one position and the pre-"
     "stroke pattern at each position, averaged over sessions within an epoch, with the"
     "change from pre-stroke beneath. Pooling here is a MEAN OVER SESSIONS rather than a"
     "sum over trials: these matrices are already reduced per session, so the session is"
     "the unit and cannot be re-weighted by its trial count. This measure is sensitive to"
     "global gain and the coding-direction measures are not, so agreement between them is"
     "the claim worth making."),
    ("epoch_6diag_matrices_pattern_*_*.png",
     "Mean-pattern similarity, own position",
     "The DIAGONAL of the preceding matrix drawn as bars: each position's similarity to its"
     "own pre-stroke pattern. The matrix carries this on its diagonal but a reader cannot"
     "compare six diagonal cells across three panels by eye, which is the comparison the"
     "recovery question actually asks. Bars, mean over sessions; dots, individual sessions"
     "coloured by animal; error bars, 95% intervals from resampling animals then sessions."
     "*, the epoch-minus-pre interval excludes zero; **, it survives Bonferroni across the"
     "figure."),
    ("epoch_6diagdelta_matrices_pattern_*_*.png",
     "Mean-pattern similarity, own position, change from pre-stroke",
     None),
    ("epoch_7_matrices_splithalf_*_*.png",
     "Within-session split-half reliability by epoch",
     "The same matrix computed within each session by splitting its trials in half, which"
     "sets the ceiling any across-session similarity can reach. A drop in the preceding"
     "figure accompanied by a drop here is a code measured less repeatably; a drop without"
     "one is a code that MOVED."),
    ("epoch_7diag_matrices_splithalf_*_*.png",
     "Split-half reliability, own position",
     "The DIAGONAL of the preceding matrix drawn as bars: each position's similarity to its"
     "own pre-stroke pattern. The matrix carries this on its diagonal but a reader cannot"
     "compare six diagonal cells across three panels by eye, which is the comparison the"
     "recovery question actually asks. Bars, mean over sessions; dots, individual sessions"
     "coloured by animal; error bars, 95% intervals from resampling animals then sessions."
     "*, the epoch-minus-pre interval excludes zero; **, it survives Bonferroni across the"
     "figure. Read against the preceding figure this is the CEILING: a drop there with a"
     "drop here is a code measured less repeatably, a drop there without one is a code that"
     "moved."),
    ("epoch_7diagdelta_matrices_splithalf_*_*.png",
     "Split-half reliability, own position, change from pre-stroke",
     None),
    ("epoch_8_matrices_crossnobis_*_*.png",
     "Crossnobis geometry by epoch",
     "Cross-validated Mahalanobis distances between position patterns, in pre-stroke units."
     " Crossnobis is noise-unbiased, which is what makes it the arbiter when the correlation"
     " measures and the coding directions disagree. It is NOT, however, immune to a change in"
     " response magnitude: d(post P, pre Q) carries a |mu_postP|^2 term depending only on P, so"
     " a gain change shifts P's distance to every reference equally. Measured post-cue"
     " 2026-09-09, chronic elevation here is largest at SPARED positions (near-ipsi +0.862,"
     " far-middle +0.845) where the row-centred value is ~0 -- amplitude, not geometry. Use the"
     " row-centred family below for anything about WHERE a pattern went. The colour scale is"
     " taken from the data rather than fixed, because a distance has no natural range."),
    ("epoch_8diag_matrices_crossnobis_*_*.png",
     "Crossnobis distance, own position",
     "The DIAGONAL of the preceding matrix drawn as bars: each position's similarity to its"
     "own pre-stroke pattern. The matrix carries this on its diagonal but a reader cannot"
     "compare six diagonal cells across three panels by eye, which is the comparison the"
     "recovery question actually asks. Bars, mean over sessions; dots, individual sessions"
     "coloured by animal; error bars, 95% intervals from resampling animals then sessions."
     "*, the epoch-minus-pre interval excludes zero; **, it survives Bonferroni across the"
     "figure. Crossnobis is noise-unbiased and immune to uniform amplitude change, so this"
     "is the arbiter when the correlation measures and the coding directions disagree."),
    ("epoch_8diagdelta_matrices_crossnobis_*_*.png",
     "Crossnobis distance, own position, change from pre-stroke",
     None),
    ("epoch_8rc_matrices_crossnobis_rowcentred_*_*.png",
     "Crossnobis geometry, row-centred: which position did it move TOWARD",
     "The same matrices with each ROW centred on its own mean. The raw distance"
     " d(post P, pre Q) = |mu_postP|^2 - 2 mu_postP . mu_preQ + |mu_preQ|^2 carries a first"
     " term depending only on P, so a change in the overall MAGNITUDE of position P's"
     " post-stroke response shifts its distance to every pre-stroke position equally and"
     " paints a uniform row -- which reads as 'moved toward all six' and means nothing of the"
     " kind. Centring removes that term and leaves the contrast WITHIN the row, which is where"
     " a substitution lives: a far-contra row going negative under an ipsilateral column is"
     " that position's pattern having moved toward it. Needed because the decoder confusions"
     " and best-match fraction are LABEL-level -- they say which position the readout assigns,"
     " not which position the pattern moved toward (Priya, 2026-09-09)."),
    ("epoch_8rcdiag_matrices_crossnobis_rowcentred_*_*.png",
     "Crossnobis distance row-centred, own position",
     "The diagonal of the row-centred matrix: how far each position sits from its own"
     " pre-stroke pattern RELATIVE to its distance from the other five. Read against the raw"
     " diagonal two slides earlier -- the pair separates a pattern that moved from a response"
     " that merely changed size, and the two disagree chronically."),
    ("epoch_8rcdiagdelta_matrices_crossnobis_rowcentred_*_*.png",
     "Crossnobis distance row-centred, own position, change from pre-stroke",
     "THE ARBITER FOR 'DID THE CODE COME BACK'. Measured post-cue 2026-09-09: acutely every"
     " position is displaced and far-contra most (+0.665 vs +0.188 to +0.463 elsewhere);"
     " chronically the displacement resolves at every position except far-contra (+0.177) and"
     " near-middle (+0.107). The RAW diagonal disagrees, showing chronic elevation largest at"
     " SPARED positions (near-ipsi +0.862, far-middle +0.845) where the row-centred value is"
     " ~0 -- that is a global gain change, not reorganisation, and it is the reason this panel"
     " exists."),
    ("epoch_8rz_matrices_crossnobis_rownorm_*_*.png",
     "Crossnobis geometry, row-NORMALISED (scale-free)",
     "The row-centred matrix with every row also divided by its own SD across columns, so each"
     "row is a z-profile and a cell reads 'how many row-SDs from this row's mean'. Row-centring"
     "alone is NOT scale-free: writing the row out, the position-dependent term scales linearly"
     "with that position's response amplitude, so row-centred values are comparable across"
     "epochs in SIGN and RANK but not in MAGNITUDE -- and amplitude moved hard (fitted"
     "amplitude factor a: 0.94 pre-stroke to 0.36 acute, post-cue). This figure is magnitude-"
     "comparable and deliberately throws away HOW FAR a position moved, which the two preceding"
     "families still carry. Read direction here, distance there."),
    ("epoch_8rzdiag_matrices_crossnobis_rownorm_*_*.png",
     "Row-normalised crossnobis, own position",
     "The diagonal: how many row-SDs the own-position cell sits from that row's mean. More"
     "negative = the position still resembles its own pre-stroke pattern more than it resembles"
     "the others, which is the pre-stroke state."),
    ("epoch_8rzdiagdelta_matrices_crossnobis_rownorm_*_*.png",
     "Row-normalised crossnobis, own position, change from pre-stroke",
     "Scale-free, so these deltas are comparable between epochs even where amplitude is not."),
    ("epoch_10c_matrices_best_match_destination_*_*.png",
     "Where each position's best pre-stroke match went",
     "For every session, each post-stroke position's pattern is matched to whichever"
     "PRE-stroke position it correlates with best, and the winner is scored one-hot; the"
     "matrix is the average over sessions in that epoch. Rows are the position the animal was"
     "cued to, columns the pre-stroke position its activity most resembled, cells the fraction"
     "of sessions. The DIAGONAL is the best-match fraction plotted elsewhere; the OFF-DIAGONAL"
     "is what that number cannot show -- whether a position's code moved to ONE other position"
     "(substitution) or scattered evenly (collapse). Acutely far-contralateral keeps only 0.06"
     "of its own sessions and sends 0.50 to far-MIDDLE and 0.25 to far-ipsilateral: a"
     "concentrated substitution onto specific neighbours, not a dissolution. THE PRE COLUMN IS"
     "ONE-HOTTED PER SESSION AND THEN AVERAGED, matching how every post-stroke epoch is built;"
     "one-hotting the AVERAGED leave-one-session-out matrix instead makes the baseline a"
     "perfect identity and charges the difference to the lesion."),
    ("epoch_10cdiag_matrices_best_match_destination_*_*.png",
     "Best match is still the correct position, per position",
     "The diagonal of the preceding matrix as bars. Same quantity as the best-match-by-position"
     "figure later in this section, but with the pre baseline built the same way as the"
     "post-stroke epochs -- so it is the one to quote."),
    ("epoch_10cdiagdelta_matrices_best_match_destination_*_*.png",
     "Best match is still the correct position, change from pre-stroke",
     "Far-contralateral falls 0.98 -> 0.06 acutely, a drop of 0.92."),
    ("epoch_10cs_best_match_destination_marked_*_*.png",
     "Where the best match went, with a test on every cell",
     "The destination matrix again, with each cell carrying a nested animals->sessions"
     " bootstrap interval from the same draws the bar families use. THIS IS THE ONE THAT TESTS"
     " THE SUBSTITUTION CLAIM. 10c draws the matrix with no uncertainty at all, and 10cdiag"
     " tests only the DIAGONAL -- so `far-contra now best-matches far-middle` was, until this"
     " figure, a colour rather than a result. The off-diagonal is where that claim lives and"
     " this is where it is measured."
     "\n\nTHE REFERENCE IS CHANCE, NOT ZERO, on the absolute row. A cell is the fraction of"
     " sessions whose best match landed there, so the question a mark answers is whether it"
     " beats the 1/6 a coin would give; the delta row is against zero in the ordinary way."
     " Both are stated on the figure by `matrix_bootstrap.mark_note`, so the panel does not"
     " depend on this legend to be read correctly."),
    ("epoch_10e_best_match_grid_*_*.png",
     "Best match, PER ANIMAL and epoch",
     "The preceding three figures pool four animals; this one does not. Rows are animals,"
     "columns are epochs, and each cell is the fraction of THAT animal's sessions in THAT"
     "epoch whose best pre-stroke match was that column. It exists because the pooled panels"
     "average epochs resting on very unequal session counts -- PS95 contributes ONE acute"
     "session and PS94 six -- so a pooled off-diagonal cell can be one animal's whole story."
     "Here it cannot hide: every panel carries its own n, and a panel resting on one session"
     "looks like one session. FRACTIONS EVERYWHERE INCLUDING PRE, unlike the grant-section"
     "version which prints counts and has to warn that its two panels' totals differ. The pre"
     "column is a mean of PER-SESSION one-hots, leave-one-session-out, and is 92-100% rather"
     "than 100% -- read each post-stroke panel against that animal's own pre panel, never"
     "against a perfect diagonal. PS94 has no chronic sessions and its panel says so rather"
     "than being dropped, which would shift the columns and imply it does."),
    ("epoch_10edelta_best_match_grid_*_*.png",
     "Where each position's best match MOVED, per animal",
     "The same grid as a change from that animal's OWN pre-stroke panel, which is the form the"
     "substitution is legible in. BLUE on the diagonal = the position stopped matching itself;"
     "RED off the diagonal IN THE SAME ROW names where it went instead; a row that goes blue"
     "without any red cell scattered rather than substituted. Acutely, post-cue, three of the"
     "four animals move far-contralateral's best match onto another FAR position -- PS92 and"
     "PS93 completely (-1.00 on the diagonal, +1.00 onto far-middle and far-ipsilateral"
     "respectively) and PS94 partially (-0.74, +0.41 onto far-middle). PS95's acute panel"
     "rests on a single session and should not be read as a fourth replication."),
    ("epoch_8g_geometry_by_position_*_*.png",
     "Geometry preserved, per position",
     "For each position, the correlation between that position's row of the"
     "representational dissimilarity matrix and the same row pre-stroke -- whether this"
     "position still sits in the same relationship to the other five. One value per"
     "session, averaged within epoch; dots are sessions. The bootstrap resamples animals"
     "then sessions and has NO block level, because these values are already one number per"
     "session: the trial-level reduction happened inside the measure."),
    ("epoch_8gdelta_geometry_by_position_*_*.png",
     "Geometry preserved, change from pre-stroke",
     None),
    ("epoch_9_delta_trajectory_*_*.png",
     "Delta trajectory across the recovery epochs",
     "Change from the pre-stroke reference at each epoch. THIS FAMILY CARRIES ITS OWN"
     "BOOTSTRAP: unlike every other panel here its values are ALREADY day-minus-pre"
     "differences, so there is no pre-stroke bar and each mark tests that epoch's value"
     "against ZERO rather than against a baseline panel. Pre minus pre is zero by"
     "construction, and drawing it would invite comparing two real panels against a column"
     "of exact zeros. Bars, mean over sessions; dots, sessions; 95% intervals from"
     "resampling animals then sessions; *, excludes zero; **, survives Bonferroni across"
     "the figure."),
    ("epoch_9ci_delta_trajectory_*_*.png",
     "Delta trajectory, intervals",
     None),
    ("epoch_10_best_match_acc_*_*.png",
     "Is the best-matching pre-stroke pattern the correct one?",
     "Fraction of positions whose closest pre-stroke pattern is that same position, per"
     "session and averaged within epoch; chance 1/6. This uses the WHOLE similarity row"
     "rather than its diagonal, which is what separates 'the code is gone' from 'the code"
     "moved to a specific other position' -- 0.2 against everything, and 0.2 against itself"
     "with 0.7 against far ipsilateral, are the same diagonal and different results."),
    ("epoch_10delta_best_match_acc_*_*.png",
     "Best-match accuracy, change from pre-stroke",
     None),
    ("epoch_10_best_match_rank_*_*.png",
     "Best match, scored by rank",
     "The same question scored by RANK rather than by hit: where the correct pre-stroke"
     "pattern falls in the ordered list of six, so 1 is a hit and 6 is maximally wrong."
     "Rank degrades gracefully where the accuracy version is all-or-nothing -- a"
     "representation that has drifted but not moved to another position shows here and is"
     "invisible there."),
    ("epoch_10delta_best_match_rank_*_*.png",
     "Best match by rank, change from pre-stroke",
     None),
    ("epoch_10b_best_match_by_position_*_*.png",
     "Best match, per position",
     "The same question resolved per position: the fraction of sessions at which that"
     "position's closest pre-stroke pattern is itself. A position whose row is entirely"
     "ungated in a session scores nothing rather than zero -- counting missing data as a"
     "wrong match would turn absence into evidence of reorganisation."),
    ("epoch_10bdelta_best_match_by_position_*_*.png",
     "Best match per position, change from pre-stroke",
     None),
    ("epoch_11_encoder_gain_shape_*_*.png",
     "Encoder explained variance, before and after one best rescale",
     "Explained variance of the position encoder and its fitted gain term, per session and"
     "averaged within epoch. Separating gain from shape asks whether the post-stroke"
     "representation is the same pattern scaled down or a different pattern altogether."),
    ("epoch_11delta_encoder_gain_shape_*_*.png",
     "Encoder explained variance, change from pre-stroke",
     None),
    ("epoch_11amp_encoder_amplitude_*_*.png",
     "Fitted AMPLITUDE factor a (1.0 = unchanged, below 1 = SMALLER)",
     "H11's MIDDLE PANEL, by epoch. `_enc_terms` returns (raw, a, gain, per-position)"
     " and the epoch figures had been reading only `raw` and `gain` -- transfer before"
     " rescaling and after -- so `a`, the amplitude term the decomposition exists to"
     " isolate, was computed every night and never shown against epoch."
     "\n\nREAD IT FIRST, then the r² pair. a = 1 is no amplitude change,"
     " a < 1 a weaker code, a > 1 a stronger one. A high post-rescaling r² with `a`"
     " far from 1 is a code that is intact and SMALLER; a low one means the tuning"
     " changed whatever `a` says. The trap the per-day version already warns about"
     " applies here too: a code that is simply GONE also recovers a lot under"
     " rescaling, because the best gain collapses toward zero and predicting nothing"
     " beats predicting an unrelated pattern."
     "\n\nITS OWN AXIS, not a third bar beside raw and gain: those are"
     " variance-explained on [0, 1] and this is a ratio around 1.0, unbounded above."),
    ("epoch_11ampdelta_encoder_amplitude_*_*.png",
     "Fitted amplitude factor a, change from pre-stroke",
     None),
    ("epoch_11pos_encoder_shape_*_*.png",
     "Encoder shape r² after the gain, per position",
     "THE ENCODER'S PER-POSITION FIGURE, the counterpart of the decoding recall panels."
     " `_enc_terms` fits ONE gain for the whole session on purpose -- a per-position gain would"
     " absorb the position-specific amplitude loss that IS the deficit -- so the per-position"
     " quantity is what survives that division: with the session's amplitude change already"
     " removed, is THIS position still predicted by its pre-stroke pattern?"
     "\n\nREAD IT AGAINST THE DECODING RECALL, not instead of it. Recall asks whether a"
     " position stays DISCRIMINABLE from the other five; this asks whether its pattern is still"
     " the pre-stroke one. A position can remain decodable on a changed pattern, so the two"
     " coming apart is the measurement behind 'decodable but re-geometried' -- and the two"
     " agreeing rules that reading out."
     "\n\nr² IS BOUNDED ABOVE BY 1 AND NOT BELOW: a pattern unrelated to its reference"
     " goes sharply negative, so the axis is autoscaled. A position sitting well below zero has"
     " not merely weakened, it is being predicted worse than by predicting nothing."),
    ("epoch_11posdelta_encoder_shape_*_*.png",
     "Encoder shape r² per position, change from pre-stroke",
     None),
    ("epoch_11c_encoder_ceiling_*_*.png",
     "Encoder SHAPE ceiling vs the frozen template",
     "The frozen encoder's explained variance had nothing to be read against: acutely it is"
     "-0.388 post-cue, which says 'worse than predicting the mean' and nothing about what was"
     "ACHIEVABLE in that session. These three bars fix that, all scoring the SAME half-session"
     "means. CEILING: scored against the other half of the same session -- how much position"
     "structure the session has at all. FROZEN (MATCHED): scored against an equally sized draw"
     "from the pre-stroke pool, so the only difference from the ceiling is WHICH SESSIONS the"
     "template came from. FROZEN (ALL PRE): the whole pool, i.e. the encoder as it is actually"
     "used elsewhere in this deck, and the only one of the three not size-matched. Read"
     "ceiling minus MATCHED, and read it against its own pre-stroke value -- that gap is 0.276"
     "at baseline with no lesion involved, the cost of a template coming from other sessions."
     "ALL THREE ARE SCORED AFTER RESCALE, i.e. SHAPE only: a raw score at these amplitudes is"
     "dominated by the encoder not being allowed to rescale (with perfect shape and only a"
     "scale mismatch R2 = 1 - (1-a)^2/a^2, which is 0.000 at a = 0.5 and -2.13 at the a = 0.361"
     "observed acutely), and the ceiling's halves have matched amplitude by construction while"
     "the frozen arm's reference does not."),
    ("epoch_11cdelta_encoder_ceiling_*_*.png",
     "Encoder ceiling and frozen template, change from pre-stroke",
     "AMPLITUDE RECOVERS AND SHAPE DOES NOT, which is the headline and is invisible in raw"
     "scores. The fitted amplitude factor goes 0.286 acute, 0.527 subacute, 0.745 chronic"
     "against 0.749 pre-stroke -- back to baseline. The shape mismatch goes +0.192, +0.128,"
     "+0.186 -- flat, and no better at chronic than acute. An earlier reading of this family"
     "reported the mismatch as recovering monotonically; those were raw scores and what was"
     "recovering in them was the amplitude. None of these differences carries an interval."),
    ("epoch_11cpos_encoder_ceiling_by_position_*_*.png",
     "Encoder SHAPE ceiling, per position",
     "The pooled ceiling averages six positions that did very different things, and this is"
     "where that matters. FAR-CONTRALATERAL LOSES NO STRUCTURE AT ALL: its ceiling is 0.626"
     "acutely against 0.628 pre-stroke, a change of -0.002, so its own trials predict each"
     "other as well as before the lesion. The positions that lose ceiling are the FLANKING"
     "ones, far-ipsilateral (-0.323) and far-middle (-0.270). Read with the next figure: far-contra"
     "keeps its structure and loses its template; its neighbours lose the structure itself."),
    ("epoch_11cposdelta_encoder_ceiling_by_position_*_*.png",
     "Encoder shape ceiling per position, change from pre-stroke",
     "Zero at far-contralateral is the result (-0.002), not a missing bar."),
    ("epoch_11cfrac_encoder_captured_by_position_*_*.png",
     "Of the shape a position can predict, how much does the PRE-STROKE template capture?",
     "matched / ceiling, per position. A FRACTION rather than a difference because the"
     "positions do not share a ceiling: a gap of 0.2 means something different at a position"
     "whose ceiling is 0.3 than at one whose ceiling is 0.8. Computed PER SESSION and then"
     "pooled, so the bars carry the same animals-then-sessions bootstrap, dots and marks as"
     "every other bar family here -- an earlier version divided one pooled number by another,"
     "which has no distribution behind it and so could not be argued with. A session whose"
     "ceiling at that position falls below 0.10 is dropped rather than contributing a ratio to"
     "a near-zero denominator. Far-contra falls from 0.57 pre-stroke to 0.01 acutely and"
     "recovers only to 0.27-0.31."),
    ("epoch_11cfracdelta_encoder_captured_by_position_*_*.png",
     "Captured fraction per position, change from pre-stroke",
     "The contrast that carries the claim. Far-contralateral's template match is essentially"
     "abolished acutely and does not return to baseline, on a position whose own structure"
     "never degraded -- which is what DISPLACED rather than LOST means, stated in the"
     "encoder's units. Converges with the frozen-vs-refit decoder arm, which recovers 34% of"
     "far-contra's acute deficit by refitting and only 9% and 11% at far-ipsi and far-middle."),
    ("epoch_13_state_decoder_*.png",
     "DOES EVERYTHING DEGRADE? Frozen BEHAVIOURAL-STATE decoder",
     "PROVISIONAL NUMBERS, 2026-09-12: every statistic on this slide that involves the LICKING "
     "class was computed with the superseded lick-BOUT anchor and has not yet been "
     "recomputed against the post-cue anchor now in use. That is the class balance, "
     "the within-session refit ceiling and the session-time control -- not the quiet "
     "window length, the feature width or the PS92 8/12 exclusion, which do not "
     "depend on it. The FIGURE is current; these prose numbers are hand-copied and "
     "lag it. Read them off the value sidecar, not off this caption. "
     "THE SPECIFICITY CONTROL FOR THE WHOLE DECK. The lesion is ventrolateral STRIATAL, so no"
     "cortex is damaged anywhere in the field of view: same window, same LocaNMF basis, same"
     "estimator, same frozen-model discipline, and only the LABEL changes -- spout target, or"
     "behavioural state. If the target readout collapses and the state readout does not, the"
     "deficit is SPECIFIC, and every generic explanation a reader reaches for first fails at"
     "once -- window clouding, haemodynamic drift, arousal, basis drift, 'a lesion was made and"
     "everything got worse' -- because each of them would degrade this readout too."
     "\n\nMETHOD. The unit is a ONE-SECOND window, not a trial: trial-level labelling gives"
     "~17 running trials per session, which decodes nothing, while tiling the bouts gives"
     "33,060 running and 41,549 quiet one-second segments across the cohort. The window LENGTH"
     "is set by quiet and not chosen -- quiet periods have a median of 1.10 s, so the 2 s window"
     "every other family here uses fits 17% of them and 1 s fits 58%. Features are four 0.25 s"
     "sub-bins x 95 components = 380 columns, the SAME width as the trial arms and on the SAME"
     "joint basis, with no per-segment baseline (a segment inside a running bout has no"
     "'before' that is not also running). Classes are quiet / running / licking, MUTUALLY"
     "EXCLUSIVE: running only if not also licking, licking only if not also running, quiet only"
     "when the window is CONTAINED in a behavior_events quiet period -- overlaps are dropped and"
     "counted, never assigned. LICKING IS ANCHORED AT THE TRIAL'S FIRST LICK AFTER THE CUE"
     "(Priya, 2026-09-12) -- the SAME anchor every position decoder in this deck uses, so the"
     "licking class and the position trials observe the same event rather than two events"
     "sharing a word. It replaces an anchor at free-running lick-bout onset, which asked a"
     "different question ('is the animal licking' over whatever the ILI rule grouped) and"
     "answered it from bouts with a median of 0.37 s. Cue and reward are simultaneous here, so"
     "the first lick after the cue is also the first lick after reward. THE WINDOW IS 1 s, NOT"
     "THE 2 s THE POSITION DECODER USES, and that is deliberate: a 2 s licking window beside 1 s"
     "running and quiet windows would make window DURATION a cue the decoder could separate the"
     "classes on -- a longer window is a smoother binned feature whatever the behaviour -- so"
     "all three are duration-matched and only the anchor is shared with the position arm. A"
     "trial whose lick never arrives within the 3.5 s response window contributes nothing,"
     "because a miss has no lick to anchor on. Running and quiet are tiled at"
     "most 8 segments per period so no long period dominates. The model is multinomial logistic"
     "frozen on ALL pre-stroke segments, with the pre column leave-one-session-out. The score is"
     "BALANCED accuracy against a chance of 1/3, never raw, because the class balance moves with"
     "epoch (quiet is 3.4% of a pre session, 15.1% acute, 0.7% chronic). PS92 8/12 is excluded:"
     "its longest 'running bout' is 2,441 s -- 41 minutes, 29% of the session, against a cohort"
     "maximum of 54 s -- which is the crash+concat discontinuity read as locomotion."
     "\n\nWHY THE FROZEN ARM AND NOT A WITHIN-SESSION REFIT: refit inside a session this"
     "decodes at macro-AUROC 0.98-1.00, and a CEILING CANNOT DEMONSTRATE PRESERVATION. The"
     "position claim rests on a frozen pre-stroke model failing, so the control has to be the"
     "same object carrying the same cross-session burden. Two confounds were checked first:"
     "session TIME alone separates the classes at AUROC 0.165-0.752, near chance, and"
     "time-matching leaves the cortical score unchanged; and the animals RUN MORE acutely (5.1%"
     "of session against 3.1% pre-stroke), so this arm is not rescued by having more data at"
     "baseline than afterwards."),
    ("epoch_13delta_state_decoder_*.png",
     "Frozen state decoder, change from pre-stroke",
     "PROVISIONAL NUMBERS, 2026-09-12: every statistic on this slide that involves the LICKING "
     "class was computed with the superseded lick-BOUT anchor and has not yet been "
     "recomputed against the post-cue anchor now in use. That is the class balance, "
     "the within-session refit ceiling and the session-time control -- not the quiet "
     "window length, the feature width or the PS92 8/12 exclusion, which do not "
     "depend on it. The FIGURE is current; these prose numbers are hand-copied and "
     "lag it. Read them off the value sidecar, not off this caption. "
     "The same quantity as epoch-minus-pre. Read it beside the POSITION decoder's contrast"
     "panel earlier in this section, and read the normalised comparison on the next slide,"
     "which is the one that puts the two on a single axis."),
    ("epoch_13n_state_vs_position_*.png",
     "Of what each readout had ABOVE CHANCE, how much survived?",
     "PROVISIONAL NUMBERS, 2026-09-12: every statistic on this slide that involves the LICKING "
     "class was computed with the superseded lick-BOUT anchor and has not yet been "
     "recomputed against the post-cue anchor now in use. That is the class balance, "
     "the within-session refit ceiling and the session-time control -- not the quiet "
     "window length, the feature width or the PS92 8/12 exclusion, which do not "
     "depend on it. The FIGURE is current; these prose numbers are hand-copied and "
     "lag it. Read them off the value sidecar, not off this caption. "
     "THE COMPARISON ITSELF. (accuracy - chance) / (1 - chance), because chance is 1/6 for the"
     "six-way position decoder and 1/3 for the three-way state decoder, so their raw accuracies"
     "are not comparable and their raw DROPS are not either -- the same absolute fall means"
     "something different when the floor is 0.167 than when it is 0.333. POSITION falls 0.86 to"
     "0.43 acutely, losing 50% of what it had. STATE falls 0.92 to 0.81, losing 11%. RUNNING"
     "alone falls 0.98 to 0.95, losing 3%."
     "\n\nBOTH BARS ARE BALANCED ACCURACY -- the mean of the six row recalls for position, and"
     "sklearn's balanced_accuracy_score for state. An earlier version scored position"
     "trial-weighted and disclosed the mismatch as a caveat; scoring it the same way as the"
     "state arm is the fix, and it matters because the post-stroke position sets are skewed by"
     "construction (PS93's are 49% far_center). Neither bar carries an interval: each is one"
     "pooled number, so there is nothing to resample."),
    ("epoch_13pos_state_decoder_by_class_*.png",
     "State decoder recall PER CLASS -- which behavioural state changed?",
     "PROVISIONAL NUMBERS, 2026-09-12: every statistic on this slide that involves the LICKING "
     "class was computed with the superseded lick-BOUT anchor and has not yet been "
     "recomputed against the post-cue anchor now in use. That is the class balance, "
     "the within-session refit ceiling and the session-time control -- not the quiet "
     "window length, the feature width or the PS92 8/12 exclusion, which do not "
     "depend on it. The FIGURE is current; these prose numbers are hand-copied and "
     "lag it. Read them off the value sidecar, not off this caption. "
     # QUOTED FROM THIS FIGURE'S OWN SIDECAR (Priya, 2026-09-13: "the notes should now
     # reference the sidecars"). These four recalls per class were hand-copied, and by the
     # 2026-09-12 render they no longer matched the figure above them -- the caption said
     # licking ran 0.99/0.97/0.90/0.96 while the sidecar held 0.98/0.96/0.88/0.98. With
     # `SELF` each arm prints its own numbers and a re-render moves caption and figure together.
     "RUNNING IS THE CLEAN EXAMPLE and the one Priya asked for: "
     "{{SELF: epoch=pre, position=running -> value:.2f}} / "
     "{{SELF: epoch=acute, position=running -> value:.2f}} / "
     "{{SELF: epoch=subacute, position=running -> value:.2f}} / "
     "{{SELF: epoch=chronic, position=running -> value:.2f}}, flat at every epoch. LICKING is "
     "{{SELF: epoch=pre, position=licking -> value:.2f}} / "
     "{{SELF: epoch=acute, position=licking -> value:.2f}} / "
     "{{SELF: epoch=subacute, position=licking -> value:.2f}} / "
     "{{SELF: epoch=chronic, position=licking -> value:.2f}}. REST IS THE ONE THAT MOVES, "
     "{{SELF: epoch=pre, position=quiet -> value:.2f}} pre to "
     "{{SELF: epoch=acute, position=quiet -> value:.2f}} acutely, and that is probably real "
     "rather than noise: the REST class is a much larger share of a post-stroke session than a "
     "pre-stroke one, so a post-stroke animal sitting still may be in a genuinely different "
     "state from a pre-stroke one sitting still. That is a finding about IMMOBILITY, not a "
     "failure of the control, and it is why this panel exists rather than only the pooled bar "
     "-- a pooled score averages exactly that away. (The class-balance PERCENTAGES this "
     "sentence used to quote -- 3.4% pre, 15.1% acute -- were measured on the RETIRED "
     "reward-anchored quiet definition and are withdrawn pending re-measurement on REST; the "
     "direction is unchanged and is visible in the counts.)"
     "\n\nTWO LIMITS, both easy to miss. FIRST, licking windows are locked to a behavioural "
     "TRANSITION (the first post-cue lick) while running and rest are sampled from inside "
     "sustained STATES, so a decoder could separate them partly on transient-versus-sustained "
     "rather than on which behaviour it is. SECOND, THE CLASSES ARE NOT SPREAD ALIKE OVER "
     "SESSION TIME and the imbalance is steeper post-stroke than pre -- licking falls ~60% "
     "from the first fifth of an acute session to the last while rest and running rise. "
     "THAT SECOND ONE WAS TESTED AND IS INERT: scoring the frozen decoder separately within "
     "each fifth of the session gives a spread of 0.008 pre-stroke against a pre-to-subacute "
     "drop of 0.096, and the spread does NOT track the composition shift -- acute has the "
     "steepest drift and the second-smallest spread. The decoder is reading behaviour, not "
     "clock time, so the classes are deliberately NOT balanced across time bins: doing so "
     "would cost 65-75% of the segments and empty a class-bin in 30 of 91 sessions to remove "
     "something measured not to matter (DECISIONS.md 2026-09-13). This family answers 'does "
     "cortex still distinguish "
     "behavioural state at all', which is what the control needs; it is not a clean three-way "
     "contrast of matched epochs."),
    ("epoch_13posdelta_state_decoder_by_class_*.png",
     "State decoder recall per class, change from pre-stroke",
     "PROVISIONAL NUMBERS, 2026-09-12: every statistic on this slide that involves the LICKING "
     "class was computed with the superseded lick-BOUT anchor and has not yet been "
     "recomputed against the post-cue anchor now in use. That is the class balance, "
     "the within-session refit ceiling and the session-time control -- not the quiet "
     "window length, the feature width or the PS92 8/12 exclusion, which do not "
     "depend on it. The FIGURE is current; these prose numbers are hand-copied and "
     "lag it. Read them off the value sidecar, not off this caption. "
     "Quiet is the only class with a visible fall. See the preceding slide for why that is a"
     "result about immobility rather than a weakness of the control."),
    ("epoch_13c_state_confusion_*.png",
     "State decoder confusion by epoch -- where do the errors go?",
     "PROVISIONAL NUMBERS, 2026-09-12: every statistic on this slide that involves the LICKING "
     "class was computed with the superseded lick-BOUT anchor and has not yet been "
     "recomputed against the post-cue anchor now in use. That is the class balance, "
     "the within-session refit ceiling and the session-time control -- not the quiet "
     "window length, the feature width or the PS92 8/12 exclusion, which do not "
     "depend on it. The FIGURE is current; these prose numbers are hand-copied and "
     "lag it. Read them off the value sidecar, not off this caption. "
     "THE SAME ARGUMENT 5c MAKES FOR THE POSITION DECODER, applied to behavioural state: the"
     "per-class panel gives the DIAGONAL -- how often each class is recalled -- and says nothing"
     "about where the errors land, which is the part that names what changed. A quiet segment"
     "misread as LICKING and one misread as RUNNING are different failures: the first says the"
     "post-stroke immobile animal looks TASK-ENGAGED to the readout, the second that it looks"
     "like it is MOVING. Recall cannot tell those apart."
     "\n\nTop row is the epoch, bottom row is that epoch's change from pre-stroke, placed"
     "beneath the panel it describes. Panels are row-normalised for display while the stored"
     "matrices stay raw counts -- which is what makes pooling an epoch a SUM rather than a mean"
     "of rates. Chance is 1/3. Read each panel against the PRE panel of this figure, never"
     "against the position decoder's confusion: the two have different class counts, different"
     "chance levels and different units of observation (1 s segments here, trials there)."),
    ("epoch_14_beta_maps_MEANref_*_*.png",
     "Decoder BETA maps, MEAN-referenced -- the six positions are COUPLED",
     "WHAT THE PIXELS ARE. A multinomial logistic decoder (L2, C=0.5, 5-fold grouped by the"
     " scheduler's position blocks) is fitted PER SESSION on the rank-100 temporal component"
     " matrix SVT -- not on LocaNMF -- so U @ coefficients is a genuine 540 x 640 cortical map"
     " with no component-space intermediary. Coefficients are averaged over folds, the"
     " StandardScaler is undone so they live in feature space, and the sub-bins are averaged"
     " into one map. Following Musall et al. (Nat Neurosci 2022) with three departures, each"
     " MEASURED: SVT rather than LocaNMF (SVT beats it by 0.06-0.09 balanced accuracy); L2"
     " rather than their L1 (split-half of the pre-stroke mean map: L1 0.688, L2 0.715, L2 +"
     " Haufe 0.960); and the HAUFE TRANSFORM, which the paper does not use and which is the"
     " single biggest factor -- a decoder weight is a FILTER whose job includes cancelling"
     " correlated noise, so a channel with NO signal can carry a large weight as a suppressor."
     " A = Cov(X) x beta converts it to a PATTERN, cov(channel, decoder output), which is the"
     " anatomical question. Pattern and filter correlate at only r = 0.245."
     "\n\nREAD THIS FIRST -- THE REFERENCE MAKES THE SIX ROWS NON-INDEPENDENT. The Haufe"
     " pattern is a covariance against the mean over ALL SIX positions, so a position that"
     " loses drive lowers the reference and hands every other position an increase it did not"
     " earn. THIS FAMILY CANNOT BE USED AS THE DEFICIT MEASURE. Its amplitudes also INVERTED"
     " once the `working` class was corrected to include non-responded trials: far-contra went"
     " from BELOW pre-stroke to above it. Acute far-contra amplitude on THIS arm, as a"
     " multiple of pre-stroke:"
     # QUOTED FROM THIS ARM'S OWN STATS SIDECAR. The number was hard-coded at 2.08 until
     # 2026-09-12 and was wrong twice over: it had drifted to 2.18 post-cue, and one constant
     # cannot serve a glob-placed note whose PRE-CUE arm reads 4.78. See `deck_values`.
     #
     # THE `||` FALLBACK IS LOAD-BEARING ON THE LICK ARM, added 2026-09-17, and the sentence
     # was restructured around it. This note is glob-placed across cue/precue/lick; the first
     # two resolve, and the LICK arm has NO `Far Contra / acute - pre` row at all because the
     # 20-trial floor refuses that cell -- the animal does not lick far-contralateral acutely,
     # which IS the deficit and is stated two paragraphs below. Until the fallback existed
     # this printed a `[[? ...]]` marker on the published slide and reported the same
     # unresolved sidecar on every build. The token now ENDS its clause: the old wording put
     # "x of pre-stroke" after it, which reads correctly after a number and is garbled after
     # any fallback text.
     " {{SELF_stats: row=Far Contra, col=acute - pre -> amplitude_vs_pre:.2f"
     " || REFUSED on this arm -- under the 20-trial floor}}."
     " It inverts because acute far-contra is then dominated by trials"
     " the animal did not attempt and the pattern then says cortex is globally quieter than on"
     " the average trial -- a large map with no position content. Use this family to see WHAT"
     " DISTINGUISHES positions; use the REST-referenced family for how much each position is"
     " driven."
     "\n\nBALANCING on the post-lick arm only. `working` is uniform over positions by"
     " construction (16.1-17.2% each) so pre-cue and post-cue need none; `lick` runs"
     " far-contralateral at 9.2-14.2% against ~19% near, and that skew IS the deficit. A cell"
     " refused for fewer than 20 trials is DRAWN with its reason -- on the lick arm"
     " far-contralateral acute is refused because the animal does not lick there, which is the"
     " deficit rather than a rendering fault."
     "\n\nHOW TO READ IT. The colour scale is shared across ALL SIX POSITIONS, because they"
     " are the same quantity in the same units from the same sessions and far-falls-near-does"
     " -not is a BETWEEN-position claim. Difference columns keep their own diverging map and"
     " EXPAND their limit rather than clipping. r in each panel is the split-half reliability"
     " of that epoch's mean map -- the ceiling any difference involving it can reach, and where"
     " a map is near-absent a low r IS the result rather than a reason to doubt it."
     "\n\nREAD IT AGAINST `epoch_15r_position_MEANref_`, which is the SAME trials, the"
     " same window and the SAME reference with the decoder, the folds and the Haufe"
     " transform removed. The only difference between the two families is the ESTIMATOR,"
     " so a pattern present in both is in the evoked activity rather than manufactured by"
     " the fit. THEY SHARE THE COUPLING THOUGH -- both are referenced to the mean over all"
     " trials -- so neither can tell an earned increase from one handed to it by another"
     " position's collapse. That needs the REST or PRECUE family."
     "\n\nWORKED EXAMPLE OF THE ESTIMATOR TERM: this figure's acute far-contralateral"
     " amplitude INVERTED, 0.47 to 2.08, when the engaged-only trial-selection bug was"
     " fixed. The drive had not moved; the estimator had. AND OF THE REFERENCE TERM:"
     " between-animal agreement on the acute change, against a null that re-pairs animals"
     " across POSITIONS, puts near-middle at r = +0.655, p = 0.003 under this reference and"
     " at p = 0.24 / 0.19 under the two uncoupled ones. Far-contralateral survives"
     " everywhere (+0.819 / +0.769 / +0.856, p <= 0.001, nulls near zero)."
     "\n\nSTATISTICS, shared by every map family in this section. THE UNIT IS THE ANIMAL"
     " and the estimator is the NESTED ANIMALS-to-SESSIONS BOOTSTRAP -- the same object every"
     " bar family in this deck uses (`epoch_figures.contrast_draws`, minus its block level,"
     " which a session-mean map does not have). Animals are resampled with replacement, then"
     " that animal's sessions within it, 2,000 draws. Sessions sharpen each animal's estimate"
     " without being counted as independent animals, which is what a flat session-level test"
     " would do."
     "\n\nTHE FAMILY-WISE THRESHOLD COMES FROM THE BOOTSTRAP'S OWN MAX-STATISTIC, not from"
     " Bonferroni. Each draw is centred on the observed and studentised by the bootstrap SE;"
     " the maximum |z| across bins is taken per draw; the 95th percentile of those maxima is"
     " the threshold. Correlated bins produce a smaller maximum than independent ones, so the"
     " data's own covariance performs the correction -- no independence assumption and no"
     " smoothness estimate (Westfall-Young / Nichols-Holmes). THIS MATTERS: Bonferroni over the"
     " ~2,000 bins assumes that many independent tests, but at FWHM ~78 px these maps carry"
     " only about 21, a 95x over-correction that put the threshold at z = 4.22 where ~3.0 is"
     " warranted. Musall et al. used plain Bonferroni over pixels and had the same issue; their"
     " SESSION-level unit gave them the power to absorb it and four animals does not."
    "\n\nMULTIPLICITY IS HANDLED WELL HERE; THAT IS NOT THE SAME AS n = 4 BEING ENOUGH, and the"
    " sentence above is about the first only. The max-statistic fixes how many tests are being"
    " counted. It does nothing about the outer resample drawing FOUR animals with replacement --"
    " 1.6% of draws are one animal repeated four times, and the bootstrap SE is itself estimated"
    " from four units. Read a surviving bin as the best correction this design admits, not as a"
    " correction that makes four animals sufficient. The per-animal panels (15rpa) are where"
    " replication is actually visible, and at n = 4 replication carries more than an interval."
     "\n\nTESTED ON AN ERODED MASK (16 px) while DISPLAY keeps the full one. The rim is where U"
     " is smallest and the Allen warp least constrained, so it carries partial-volume mixing"
     " and alignment jitter -- and those artefacts are SYSTEMATIC, hence consistent across"
     " animals, which is exactly what a between-animal denominator rewards. The max-statistic"
     " fixes multiplicity, NOT this: measured, the near positions are 1.35-1.54x rim-enriched"
     " without erosion. Any result more than 2x concentrated in the rim is SUPPRESSED rather"
     " than annotated, because a drawn contour is read as a result and a subtitle caveat is"
     " not."
     "\n\nFIELD OF VIEW. Both olfactory bulbs are excluded by name, and the fibre-glue"
     " occlusion is excluded from a HAND-PAINTED per-animal mask -- pooled figures use the"
     " UNION, since a pixel occluded in ANY animal cannot contribute to a cross-animal average."
     " 207,213 -> 151,447 px. The glue could not be found automatically (brightness here is"
     " dominated by geometry, not occlusion), and the cut costs 19.6% of the mask but only 3-4%"
     " of each position's acute effect energy, because the position code lives in SSp/MO."
     "\n\nPOOLING IS WITHIN ANIMAL. A difference uses only animals present in BOTH epochs, and"
     " each animal is divided by its OWN pre-stroke map amplitude before pooling, per position."
     " That is a SCALE NORMALISATION and NOT a z-score: one constant per (animal, position),"
     " computed only from pre-stroke data, so it cancels exactly in every within-animal epoch"
     " ratio and cannot leak epoch information. It was needed because pre-stroke amplitude"
     " varies 2.44x across animals, so an unweighted pooled map was largely the brightest"
     " animal's."),
    ("epoch_14pa_beta_maps_MEANref_by_animal_*_*.png",
     "Decoder beta maps PER ANIMAL -- does the pattern replicate?",
     "THE POOLED FIGURE AVERAGES FOUR ANIMALS AND EVERY TEST ESTIMATES ITS SPREAD FROM FOUR, so"
     " neither can show whether a pattern REPLICATES -- which at n=4 is the stronger evidence."
     " One position per figure (far-contralateral, where the deficit is); its neighbours are on"
     " the pooled panel."
     "\n\nMETHOD is identical to the pooled figure. THE COLOUR SCALE IS PER ANIMAL here and"
     " NOT shared, because a row is a different mouse whose dF/F scale moves with expression"
     " and window clarity -- one scale would make a dim animal look like a weak effect. Rows"
     " are comparable across their own epochs, not to each other. NO significance is drawn: a"
     " per-animal panel has no between-animal spread to test, and a within-animal test would"
     " answer a different question from the pooled figures."
     "\n\nWHAT IT SHOWS, and it is NOT the spatial-code claim: all four animals replicate,"
     " but what replicates is a near-uniformly NEGATIVE acute map -- the engagement collapse"
     " described on the pooled slide. PS94 and PS95 took 3 mW and show overt deficits; PS92 and"
     " PS93 were milder, so a severity-graded difference between rows is a finding and a random"
     " one is a warning."),
    ("epoch_14z_beta_vs_ZERO_*_*.png",
     "WHERE the position code IS, epoch by epoch (Musall et al. 2023, fig. S6)",
     "A DIFFERENT QUESTION FROM EVERY OTHER MAP FAMILY HERE: not where the code CHANGED, but"
     " where it IS in each epoch. Each panel is tested on its own against ZERO, so it is immune"
     " to the two confounds the difference figures carry -- elapsed weeks (pre-vs-post is"
     " perfectly confounded with time, and no relabelling test can separate them) and a quiet"
     " baseline that drifts by chronic."
     "\n\nDO NOT READ TWO PANELS AS A DIFFERENCE. Significant in pre and not in acute is NOT"
     " evidence of a change: a map that just clears the threshold in one epoch and just misses"
     " it in the next may not differ at all, and nothing on this figure puts an error bar on"
     " that comparison. The difference question has its own figures and its own test."
     "\n\nMETHOD, following the paper: maps downsampled 8x to a 67x80 grid (about 2,000 bins"
     " inside the eroded Allen mask, against their 3,364), tested per bin against zero. THE"
     " UNIT IS THE ANIMAL, through this deck's nested bootstrap. Musall pooled SESSIONS, which"
     " is far more powerful and treats two sessions from one animal as independent -- that"
     " number is computed and printed in the render log beside ours for comparison, and is NOT"
     " what the contour draws. Since the underlying maps are one-vs-rest, a significant bin"
     " means this position's trials differ from the AVERAGE TRIAL there; read it with the"
     " coupling caveat that applies to every mean-referenced figure."
     "\n\nSTATISTICS, shared by every map family in this section. THE UNIT IS THE ANIMAL"
     " and the estimator is the NESTED ANIMALS-to-SESSIONS BOOTSTRAP -- the same object every"
     " bar family in this deck uses (`epoch_figures.contrast_draws`, minus its block level,"
     " which a session-mean map does not have). Animals are resampled with replacement, then"
     " that animal's sessions within it, 2,000 draws. Sessions sharpen each animal's estimate"
     " without being counted as independent animals, which is what a flat session-level test"
     " would do."
     "\n\nTHE FAMILY-WISE THRESHOLD COMES FROM THE BOOTSTRAP'S OWN MAX-STATISTIC, not from"
     " Bonferroni. Each draw is centred on the observed and studentised by the bootstrap SE;"
     " the maximum |z| across bins is taken per draw; the 95th percentile of those maxima is"
     " the threshold. Correlated bins produce a smaller maximum than independent ones, so the"
     " data's own covariance performs the correction -- no independence assumption and no"
     " smoothness estimate (Westfall-Young / Nichols-Holmes). THIS MATTERS: Bonferroni over the"
     " ~2,000 bins assumes that many independent tests, but at FWHM ~78 px these maps carry"
     " only about 21, a 95x over-correction that put the threshold at z = 4.22 where ~3.0 is"
     " warranted. Musall et al. used plain Bonferroni over pixels and had the same issue; their"
     " SESSION-level unit gave them the power to absorb it and four animals does not."
     "\n\nTESTED ON AN ERODED MASK (16 px) while DISPLAY keeps the full one. The rim is where U"
     " is smallest and the Allen warp least constrained, so it carries partial-volume mixing"
     " and alignment jitter -- and those artefacts are SYSTEMATIC, hence consistent across"
     " animals, which is exactly what a between-animal denominator rewards. The max-statistic"
     " fixes multiplicity, NOT this: measured, the near positions are 1.35-1.54x rim-enriched"
     " without erosion. Any result more than 2x concentrated in the rim is SUPPRESSED rather"
     " than annotated, because a drawn contour is read as a result and a subtitle caveat is"
     " not."
     "\n\nFIELD OF VIEW. Both olfactory bulbs are excluded by name, and the fibre-glue"
     " occlusion is excluded from a HAND-PAINTED per-animal mask -- pooled figures use the"
     " UNION, since a pixel occluded in ANY animal cannot contribute to a cross-animal average."
     " 207,213 -> 151,447 px. The glue could not be found automatically (brightness here is"
     " dominated by geometry, not occlusion), and the cut costs 19.6% of the mask but only 3-4%"
     " of each position's acute effect energy, because the position code lives in SSp/MO."
     "\n\nPOOLING IS WITHIN ANIMAL. A difference uses only animals present in BOTH epochs, and"
     " each animal is divided by its OWN pre-stroke map amplitude before pooling, per position."
     " That is a SCALE NORMALISATION and NOT a z-score: one constant per (animal, position),"
     " computed only from pre-stroke data, so it cancels exactly in every within-animal epoch"
     " ratio and cannot leak epoch information. It was needed because pre-stroke amplitude"
     " varies 2.44x across animals, so an unweighted pooled map was largely the brightest"
     " animal's."),
    ("epoch_15_evoked_CUEINCREMENT_PRECUEref_cue.png",
     "The CUE-EVOKED INCREMENT -- post-cue minus PRE-CUE, positions independent",
     "EACH MAP IS THAT POSITION'S OWN post-cue mean minus its own pre-cue mean, so the six"
     " positions are INDEPENDENT -- far-contralateral collapsing cannot leak into"
     " near-ipsilateral's map, because near-ipsilateral's map never looks at far-contra's"
     " trials. Maps come from `framemap_event_maps`; nothing is recomputed."
     "\n\nWHAT IT MEASURES, AND THE CAVEAT THAT GOES WITH IT. Subtracting the pre-cue window"
     " CONFLICTS WITH DECISION F12, which is marked load-bearing: a per-trial pre-cue baseline"
     " OVER-SUBTRACTS real anticipatory signal, because the pre-cue window decodes position"
     " above chance even under block-CV (LOSO 0.510 against post-cue 0.873 -- the spout arrives"
     " BEFORE the cue, so position is knowable). This family therefore measures the CUE-EVOKED"
     " INCREMENT, not the position representation, and an amplitude of 0.11 is a ratio of"
     " increments rather than a map that fell to 11%. The two windows also have DIFFERENT"
     " deficits, so their difference mixes two unequal effects. The conflict was INHERITED --"
     " the npz carries a post-minus-pre field and this family aggregates it -- not chosen. F13"
     " cuts the same way: the pre-cue window is the deck's MOTOR-INDEPENDENT readout, the only"
     " one decoding above chance on NO-LICK trials, i.e. exactly the post-stroke failed"
     " attempt."
     "\n\nUSE THE REST-REFERENCED FAMILY AS THE PRIMARY MAP. This one stays because the"
     " increment is a real quantity worth seeing, and because it is the most SENSITIVE of the"
     " three references -- it removes the most common signal, so the position-specific part is"
     " least diluted."
     "\n\nSTATISTICS, shared by every map family in this section. THE UNIT IS THE ANIMAL"
     " and the estimator is the NESTED ANIMALS-to-SESSIONS BOOTSTRAP -- the same object every"
     " bar family in this deck uses (`epoch_figures.contrast_draws`, minus its block level,"
     " which a session-mean map does not have). Animals are resampled with replacement, then"
     " that animal's sessions within it, 2,000 draws. Sessions sharpen each animal's estimate"
     " without being counted as independent animals, which is what a flat session-level test"
     " would do."
     "\n\nTHE FAMILY-WISE THRESHOLD COMES FROM THE BOOTSTRAP'S OWN MAX-STATISTIC, not from"
     " Bonferroni. Each draw is centred on the observed and studentised by the bootstrap SE;"
     " the maximum |z| across bins is taken per draw; the 95th percentile of those maxima is"
     " the threshold. Correlated bins produce a smaller maximum than independent ones, so the"
     " data's own covariance performs the correction -- no independence assumption and no"
     " smoothness estimate (Westfall-Young / Nichols-Holmes). THIS MATTERS: Bonferroni over the"
     " ~2,000 bins assumes that many independent tests, but at FWHM ~78 px these maps carry"
     " only about 21, a 95x over-correction that put the threshold at z = 4.22 where ~3.0 is"
     " warranted. Musall et al. used plain Bonferroni over pixels and had the same issue; their"
     " SESSION-level unit gave them the power to absorb it and four animals does not."
     "\n\nTESTED ON AN ERODED MASK (16 px) while DISPLAY keeps the full one. The rim is where U"
     " is smallest and the Allen warp least constrained, so it carries partial-volume mixing"
     " and alignment jitter -- and those artefacts are SYSTEMATIC, hence consistent across"
     " animals, which is exactly what a between-animal denominator rewards. The max-statistic"
     " fixes multiplicity, NOT this: measured, the near positions are 1.35-1.54x rim-enriched"
     " without erosion. Any result more than 2x concentrated in the rim is SUPPRESSED rather"
     " than annotated, because a drawn contour is read as a result and a subtitle caveat is"
     " not."
     "\n\nFIELD OF VIEW. Both olfactory bulbs are excluded by name, and the fibre-glue"
     " occlusion is excluded from a HAND-PAINTED per-animal mask -- pooled figures use the"
     " UNION, since a pixel occluded in ANY animal cannot contribute to a cross-animal average."
     " 207,213 -> 151,447 px. The glue could not be found automatically (brightness here is"
     " dominated by geometry, not occlusion), and the cut costs 19.6% of the mask but only 3-4%"
     " of each position's acute effect energy, because the position code lives in SSp/MO."
     "\n\nPOOLING IS WITHIN ANIMAL. A difference uses only animals present in BOTH epochs, and"
     " each animal is divided by its OWN pre-stroke map amplitude before pooling, per position."
     " That is a SCALE NORMALISATION and NOT a z-score: one constant per (animal, position),"
     " computed only from pre-stroke data, so it cancels exactly in every within-animal epoch"
     " ratio and cannot leak epoch information. It was needed because pre-stroke amplitude"
     " varies 2.44x across animals, so an unweighted pooled map was largely the brightest"
     " animal's."),
    ("epoch_15pa_evoked_CUEINCREMENT_PRECUEref_by_animal_cue.png",
     "Cue-evoked increment PER ANIMAL -- does the independent measure replicate?",
     "THE PER-ANIMAL VIEW of the increment, far-contralateral. Four animals is the ceiling on"
     " every test here and the pooled figure averages these rows, so neither can show"
     " REPLICATION -- which at n=4 is the stronger evidence."
     "\n\nCOLOUR SCALE IS PER ANIMAL and not shared: a row is a different mouse whose dF/F"
     " scale moves with expression and window clarity. NO significance is drawn -- a per-animal"
     " panel has no between-animal spread to test. Carries the same F12 caveat as the pooled"
     " slide: this is the cue-evoked INCREMENT, not the position map."),
    # MATCHES BOTH BASELINES, because during the migration two of them exist on the share and
    # the filename is what distinguishes them (`position_reference_maps.reference_tag`).
    # `RE*ref` catches RESTref and REWARD8ref while excluding MEANref and PRECUEref.
    ("epoch_15r_position_RE*ref_*_*.png",
     "Position maps vs the REST BASELINE -- the PRIMARY map family",
     "LEAD WITH THIS ONE. Each map is that position's window mean minus the session's REST"
     " baseline -- the mean over frames that are slow-treadmill, non-licking and buffered away"
     " from reward, as `behavior_events` defines them. The subtrahend is ONE MAP PER SESSION,"
     " identical for all six positions, so subtracting it CANNOT couple them; and it carries no"
     " position information, so nothing real is removed."
     "\n\nWHY THIS IS THE RIGHT REFERENCE. Decision F12 permits a SESSION-CONSTANT baseline"
     " (it is invisible to a standardised decoder) while rejecting a per-trial pre-cue one."
     " Maps are not standardised, so for maps that constant is a real choice -- and the rest"
     " baseline is the map-space analogue of what F12 endorses. It answers whether this"
     " position's cortex is driven AT ALL, where the mean-referenced family answers whether it"
     " is driven more than the others."
     "\n\nCHECK THE FILENAME BEFORE READING THE CHRONIC COLUMN. `_REWARD8ref_` is the RETIRED"
     " baseline: it excluded 8 s after every reward, a buffer carried over from a task whose"
     " post-tone window was 8 s where ours has a 3.5 s response window. Under it the baseline"
     " is estimated from 4.4% of frames pre-stroke, 17.1% acutely and a MEDIAN 0.7%"
     " chronically, and its chronic between-animal agreement is position-INDEPENDENT shared"
     " offset -- observed r = +0.494 against a null of +0.497, p = 0.21 to 0.97. That column"
     " carries no information. `_RESTref_` is the replacement: between trials, not running,"
     " not licking. See docs/REST_BASELINE_MIGRATION.md."
     "\n\nITS OWN CAVEAT, stated rather than buried: the rest baseline DRIFTS across epochs,"
     " so ACROSS-EPOCH amplitude ratios carry that confound while BETWEEN-POSITION contrasts"
     " do not -- a shift common to all six positions cancels in a contrast between them."
     " MEASURED DIRECTLY, and read the size from the drift figure later in this section rather"
     " than from this caption: 6 of 11 animal-epoch cells shift detectably, by 7-43% of the"
     " evoked signal, and that is an UPPER BOUND -- it is attained only if the shift aligns"
     " with the evoked pattern, where an orthogonal one costs roughly r-squared/2. Acute is the"
     " LEAST affected epoch despite carrying the largest deficit, so the drift does not simply"
     " track the lesion."
     "\n\nAN EARLIER VERSION OF THIS CAPTION PUT THE DRIFT AT ABOUT HALF THE SIGNAL. That"
     " number described the RETIRED `quiet` baseline, which was anchored on the animal's"
     " PERFORMANCE and so tracked the deficit by construction (4.4% of frames pre-stroke, 17.1%"
     " acutely, 0.7% chronically). The current baseline is anchored on the TRIAL, which happens"
     " whether or not the animal succeeds. Do not re-derive an argument from the old figure."
     "\n\nTHE TRIAL SELECTION IS THE BETA MAPS' -- same window, same class definitions"
     " including the no-lick arm, same 20-trial floor -- so a difference between this family"
     " and the beta maps is the REFERENCE and nothing else."
     "\n\nSTATISTICS, shared by every map family in this section. THE UNIT IS THE ANIMAL"
     " and the estimator is the NESTED ANIMALS-to-SESSIONS BOOTSTRAP -- the same object every"
     " bar family in this deck uses (`epoch_figures.contrast_draws`, minus its block level,"
     " which a session-mean map does not have). Animals are resampled with replacement, then"
     " that animal's sessions within it, 2,000 draws. Sessions sharpen each animal's estimate"
     " without being counted as independent animals, which is what a flat session-level test"
     " would do."
     "\n\nTHE FAMILY-WISE THRESHOLD COMES FROM THE BOOTSTRAP'S OWN MAX-STATISTIC, not from"
     " Bonferroni. Each draw is centred on the observed and studentised by the bootstrap SE;"
     " the maximum |z| across bins is taken per draw; the 95th percentile of those maxima is"
     " the threshold. Correlated bins produce a smaller maximum than independent ones, so the"
     " data's own covariance performs the correction -- no independence assumption and no"
     " smoothness estimate (Westfall-Young / Nichols-Holmes). THIS MATTERS: Bonferroni over the"
     " ~2,000 bins assumes that many independent tests, but at FWHM ~78 px these maps carry"
     " only about 21, a 95x over-correction that put the threshold at z = 4.22 where ~3.0 is"
     " warranted. Musall et al. used plain Bonferroni over pixels and had the same issue; their"
     " SESSION-level unit gave them the power to absorb it and four animals does not."
     "\n\nTESTED ON AN ERODED MASK (16 px) while DISPLAY keeps the full one. The rim is where U"
     " is smallest and the Allen warp least constrained, so it carries partial-volume mixing"
     " and alignment jitter -- and those artefacts are SYSTEMATIC, hence consistent across"
     " animals, which is exactly what a between-animal denominator rewards. The max-statistic"
     " fixes multiplicity, NOT this: measured, the near positions are 1.35-1.54x rim-enriched"
     " without erosion. Any result more than 2x concentrated in the rim is SUPPRESSED rather"
     " than annotated, because a drawn contour is read as a result and a subtitle caveat is"
     " not."
     "\n\nFIELD OF VIEW. Both olfactory bulbs are excluded by name, and the fibre-glue"
     " occlusion is excluded from a HAND-PAINTED per-animal mask -- pooled figures use the"
     " UNION, since a pixel occluded in ANY animal cannot contribute to a cross-animal average."
     " 207,213 -> 151,447 px. The glue could not be found automatically (brightness here is"
     " dominated by geometry, not occlusion), and the cut costs 19.6% of the mask but only 3-4%"
     " of each position's acute effect energy, because the position code lives in SSp/MO."
     "\n\nPOOLING IS WITHIN ANIMAL. A difference uses only animals present in BOTH epochs, and"
     " each animal is divided by its OWN pre-stroke map amplitude before pooling, per position."
     " That is a SCALE NORMALISATION and NOT a z-score: one constant per (animal, position),"
     " computed only from pre-stroke data, so it cancels exactly in every within-animal epoch"
     " ratio and cannot leak epoch information. It was needed because pre-stroke amplitude"
     " varies 2.44x across animals, so an unweighted pooled map was largely the brightest"
     " animal's."),
    ("epoch_15r_position_MEANref_*_*.png",
     "Position maps vs the MEAN OVER ALL TRIALS -- the coupled reference, for comparison",
     "THE SAME TRIALS AND WINDOW AS ITS REST-REFERENCED PARTNER, with only the subtrahend"
     " changed: here it is the mean over ALL trials in the window, which is what the beta-map"
     " decoder centres on. IT EXISTS TO BE COMPARED AGAINST, not to be read alone."
     "\n\nTHE SIX ROWS ARE NOT INDEPENDENT. A position that loses drive lowers the shared"
     " reference and hands every other position an increase it did not earn. This family shows"
     " that coupling as a PLAIN TRIAL AVERAGE -- no decoder, no folds, no Haufe transform -- so"
     " the artefact can be seen without three other things in the way. A rise that appears here"
     " and NOT under the rest reference is the artefact."
     "\n\nTHE THREE FAMILIES DECOMPOSE, which is what makes this a controlled set rather"
     " than three opinions. Figure 14 is data + mean reference + decoder/Haufe estimator; this"
     " family is data + mean reference; the REST and PRECUE families are data. So FIG 14 vs"
     " THIS ONE isolates the ESTIMATOR (the fit, the folds, the Haufe transform), and THIS ONE"
     " vs REST/PRECUE isolates the REFERENCE. Note what the first comparison canNOT do: figure"
     " 14 and this family SHARE the reference, so their agreement rules out the decoder and"
     " says nothing about coupling. A row that rises in both still needs an uncoupled"
     " reference to decide whether the rise was earned."
     "\n\nMEASURED, 2026-09-12. Between-animal agreement on the acute change, against a null"
     " that re-pairs animals ACROSS POSITIONS (preserving anatomy, warp, rim and glue, and"
     " destroying only position-specific agreement): NEAR MIDDLE reaches r = +0.655, p = 0.003"
     " under THIS reference and vanishes under both uncoupled ones (p = 0.24, p = 0.19). That"
     " is the coupling artefact with a number on it. FAR CONTRALATERAL survives everywhere"
     " (+0.819 / +0.769 / +0.856, all p <= 0.001, nulls near zero). Under the mean reference a"
     " high between-animal r is what coupling PREDICTS -- it is not evidence."
     "\n\nIT IS ALSO THE NOISIER REFERENCE, by construction: subtracting the mean removes the"
     " large common task response and leaves a small position-specific residual, so split-half"
     " reliability falls -- as low as r = -0.02 on the pre-cue arm, against 0.93-0.98 for the"
     " rest reference. Dramatic-looking difference panels on this family are frequently that"
     " residual noise rendered on a shared colour scale."
     "\n\nSTATISTICS, shared by every map family in this section. THE UNIT IS THE ANIMAL"
     " and the estimator is the NESTED ANIMALS-to-SESSIONS BOOTSTRAP -- the same object every"
     " bar family in this deck uses (`epoch_figures.contrast_draws`, minus its block level,"
     " which a session-mean map does not have). Animals are resampled with replacement, then"
     " that animal's sessions within it, 2,000 draws. Sessions sharpen each animal's estimate"
     " without being counted as independent animals, which is what a flat session-level test"
     " would do."
     "\n\nTHE FAMILY-WISE THRESHOLD COMES FROM THE BOOTSTRAP'S OWN MAX-STATISTIC, not from"
     " Bonferroni. Each draw is centred on the observed and studentised by the bootstrap SE;"
     " the maximum |z| across bins is taken per draw; the 95th percentile of those maxima is"
     " the threshold. Correlated bins produce a smaller maximum than independent ones, so the"
     " data's own covariance performs the correction -- no independence assumption and no"
     " smoothness estimate (Westfall-Young / Nichols-Holmes). THIS MATTERS: Bonferroni over the"
     " ~2,000 bins assumes that many independent tests, but at FWHM ~78 px these maps carry"
     " only about 21, a 95x over-correction that put the threshold at z = 4.22 where ~3.0 is"
     " warranted. Musall et al. used plain Bonferroni over pixels and had the same issue; their"
     " SESSION-level unit gave them the power to absorb it and four animals does not."
     "\n\nTESTED ON AN ERODED MASK (16 px) while DISPLAY keeps the full one. The rim is where U"
     " is smallest and the Allen warp least constrained, so it carries partial-volume mixing"
     " and alignment jitter -- and those artefacts are SYSTEMATIC, hence consistent across"
     " animals, which is exactly what a between-animal denominator rewards. The max-statistic"
     " fixes multiplicity, NOT this: measured, the near positions are 1.35-1.54x rim-enriched"
     " without erosion. Any result more than 2x concentrated in the rim is SUPPRESSED rather"
     " than annotated, because a drawn contour is read as a result and a subtitle caveat is"
     " not."
     "\n\nFIELD OF VIEW. Both olfactory bulbs are excluded by name, and the fibre-glue"
     " occlusion is excluded from a HAND-PAINTED per-animal mask -- pooled figures use the"
     " UNION, since a pixel occluded in ANY animal cannot contribute to a cross-animal average."
     " 207,213 -> 151,447 px. The glue could not be found automatically (brightness here is"
     " dominated by geometry, not occlusion), and the cut costs 19.6% of the mask but only 3-4%"
     " of each position's acute effect energy, because the position code lives in SSp/MO."
     "\n\nPOOLING IS WITHIN ANIMAL. A difference uses only animals present in BOTH epochs, and"
     " each animal is divided by its OWN pre-stroke map amplitude before pooling, per position."
     " That is a SCALE NORMALISATION and NOT a z-score: one constant per (animal, position),"
     " computed only from pre-stroke data, so it cancels exactly in every within-animal epoch"
     " ratio and cannot leak epoch information. It was needed because pre-stroke amplitude"
     " varies 2.44x across animals, so an unweighted pooled map was largely the brightest"
     " animal's."),
    ("epoch_15r_position_PRECUEref_*_*.png",
     "Position maps vs each trial's OWN PRE-CUE window -- the third reference",
     "BESIDE THE REST FAMILY, NOT INSTEAD OF IT. Same trials, same window, same estimator;"
     " the subtrahend is each trial's own pre-cue second, so what is drawn is the CUE-EVOKED"
     " INCREMENT rather than the position map."
     "\n\nITS OBJECTION, STATED FIRST. This violates F12. The pre-cue window carries genuine"
     " anticipatory position information -- LOSO 0.510 against post-cue 0.873, chance 0.167 --"
     " so subtracting it removes real position code, and an increment is not an amplitude."
     " That is why this family is not the primary one and why figure 15's `delta` was"
     " RELABELLED to name its reference rather than left to look like the position map."
     "\n\nWHY IT IS ON THE DECK ANYWAY, AND IT IS NOT A CONSOLATION. Its subtrahend is"
     " PER-TRIAL, so it cannot drift across epochs -- and across-epoch drift is precisely the"
     " weakness of the rest reference, measured at 7-43% of the evoked signal by the rest"
     " baseline drift figure later in this section. The two references fail in DIFFERENT"
     " directions, which is the whole logic of carrying more than one: an effect present under"
     " both is not a property of either subtrahend, and an effect in only one names the"
     " subtrahend to suspect. Read the three references as a set, and read a disagreement as"
     " information rather than as noise."
     "\n\nWHAT IT SHARES WITH THE REST FAMILY: the positions stay UNCOUPLED. No position's"
     " loss can raise another position's reference, which is the specific failure of the"
     " mean-referenced family above."),
    ("epoch_15rpa_position_*ref_by_animal_*_*.png",
     "Reference maps PER ANIMAL -- both references, at the animal level",
     "COMPLETES THE PER-ANIMAL SET (14pa for the decoder maps, 15pa for the increment, this for"
     " the two reference maps). Each pooled figure averages four rows and every test estimates"
     " its spread from four animals, so none of them can show REPLICATION."
     "\n\nBOTH REFERENCES, ONE PER FIGURE, so the pair reads side by side at the animal level"
     " the way the pooled pair reads at the group level. Same trials, window, class definition"
     " and floor; only the subtrahend differs, so a row that differs between the two differs"
     " BECAUSE of the reference. Under REST every animal should be broadly positive -- task"
     " activity above rest; under MEAN the common component is gone and what remains is"
     " position-specific and much noisier, so a row dramatic there and flat under rest is the"
     " one-vs-rest coupling showing itself in a single animal."
     "\n\nCOLOUR SCALE IS PER ANIMAL and NO significance is drawn, for the reasons given on"
     " the other per-animal slides."),
    ("epoch_12b_stopped_pooled_similarity_*.png",
     "STOPPED trials POOLED over positions -- the powered version",
     "THE SAME QUESTION AS THE PRECEDING SLIDE WITH SIX TIMES THE DATA PER MEASUREMENT. The"
     "quit period is short by definition and the per-position arm divides it six ways: pooled"
     "across animals the post-cue stopped sets hold 867 trials pre-stroke, 1,984 acute, 1,935"
     "subacute and 359 chronic, but split per position the chronic cell falls to 36-75 trials,"
     "which is not a mean pattern. Here ALL of a session's stopped trials become ONE mean"
     "pattern -- no position split -- and the question being asked, does the cortical pattern"
     "during the quit period still resemble the pre-stroke one, never needed the position axis"
     "(Priya, 2026-09-11: 'I more was thinking about mean pattern similarity for ALL stopped"
     "trials rather than per position')."
     "\n\nWHAT IS CORRELATED WITH WHAT: each session's pooled stopped pattern against THAT"
     "ANIMAL's pre-stroke ENGAGED pooled mean. Both stopped columns use the same reference, so"
     "the PRE bar is the no-lesion control -- how far QUITTING ALONE moves the pattern, with no"
     "lesion involved -- and only the difference between it and a post-stroke bar is"
     "attributable to the lesion. The floor is 20 stopped trials rather than the per-position"
     "arm's 5: a session contributes ONE number here, so there is no reason to accept a cell"
     "built from five."
     "\n\nTHE PRE BAR STILL RESTS ON TWO ANIMALS AND THAT IS NOT FIXABLE BY POOLING. PS92 has"
     "6 pre-stroke stopped trials and PS93 has 40 -- a well-trained pre-stroke animal barely"
     "quits, which is the same fact that makes the engagement gate worth having. The subtitle"
     "names whoever is excluded and their counts."
     "\n\nWHY THIS IS NOT AN ENCODER FIGURE. 'Explained variance' in the encoder families is"
     "variance ACROSS POSITIONS: `_enc_terms` centres a 6 x 380 matrix down the position axis"
     "and the denominator is the between-position sum of squares. Pool the positions away and"
     "that denominator is zero by construction, so an encoder EV would have nothing left to"
     "explain. Correlation against a reference pattern is the measure that survives pooling."),
    ("epoch_12bdelta_stopped_pooled_similarity_*.png",
     "Pooled stopped-trial similarity, change from pre-stroke",
     "The same quantity as epoch-minus-pre. Read against the PRE bar of the preceding slide,"
     "which is the quitting-alone control, and not against zero."),
    ("epoch_12_stopped_pattern_*.png",
     "STOPPED trials: does the position pattern survive the animal quitting?",
     "THE NULL THE WHOLE STOPPED ARM RESTS ON, and the figure to read before any of the"
     "other stopped-class panels in this section. Trials inside the terminal quit period ONLY"
     "-- the set every other figure here removes -- correlated against that animal's pre-stroke"
     "ENGAGED mean pattern. The pre column is pre-STROKE stopped trials against that same"
     "template, i.e. how far quitting alone moves the pattern with no lesion involved."
     "THE ANSWER IS THAT THERE IS ALMOST NOTHING THERE TO MOVE. The pre diagonal is -0.01 and"
     "every post-stroke diagonal is within 0.02 of zero, so stopped trials carry essentially no"
     "position pattern before the lesion either; the change row beneath is a difference between"
     "two near-zero numbers and its colours should not be read as structure. This agrees with"
     "the stopped DECODER arm, whose pre-stroke accuracy is 0.18-0.31 against a chance of"
     "0.167. The control rests on PS94 and PS95: an animal needs four of six positions and 100"
     "pre-stroke stopped trials to define it, and PS92 has 6 and PS93 has 40 -- one session"
     "each. A well-trained pre-stroke animal barely quits, which is the same fact that makes"
     "the engagement gate worth having."),
    # ---------------------------------------------------------------- THE REST ARM (15x/f/s/d)
    #
    # ADDED 2026-09-17. These four were on the share and three were regenerated nightly, and
    # NO SLIDE REFERENCED ANY OF THEM -- the inverse of the stale-figure failure this section's
    # header warns about. There a slide read a figure no nightly step regenerated; here the
    # nightly regenerated figures no slide read, so the work was invisible to a deck reader.
    # The completeness guard cannot catch this: it reports figures it EXPECTS and is silent
    # about ones it was never told about, which is the same blind spot that left 33 of 75
    # figures unreferenced before the numbering was moved out of the data.
    #
    # EVERY NUMBER IN THESE LEGENDS IS POST-AUDIT (docs/REST_ENGAGEMENT_AUDIT.md, 2026-09-16):
    # the engagement gate is applied and the repaired classifier is used. Pre-audit values
    # differ and must not be quoted.
    ("epoch_15x_REST_by_position_by_animal.png",
     "REST carries position information -- per animal, per position",
     "Each cell is one animal's mean REST map for one spout position, minus that animal's own"
     " across-position mean. Rest is the inter-trial interval on the `restdock05` definition:"
     " the spout DOCKED and out of reach, no target present anywhere, the animal not running"
     " and away from licking, with the terminal quit period excluded."
     "\n\nWHAT IT SHOWS. Rest is not position-neutral. A circular-shift permutation that keeps"
     " the block-time structure INSIDE the null -- shifting the position labels over"
     " time-ordered rest periods, so `drift aliased onto blocks` is fully present in the null --"
     " gives observed/null 1.634 over 44 pre-stroke sessions, above null in 43/44 and in 4/4"
     " animals (PS92 1.467, PS93 1.999, PS94 1.724, PS95 1.458). At the map level the"
     " between-position RMS is 1.97x the WITHIN-position split-half RMS, with up to 1,468 of"
     " 2,022 in-mask bins significant."
     "\n\nWHY IT MATTERS FOR EVERY OTHER MAP IN THIS DECK. The rest baseline is subtracted from"
     " the position maps. A subtrahend that itself carries position leaves that difference IN"
     " the maps, looking exactly like position coding -- which is why the baseline is"
     " POSITION-WEIGHTED (`restw`, six per-position medians averaged equally) rather than"
     " pooled over rest frames."),
    ("epoch_15d_delta_rest_flatpool_minus_restw.png",
     "Why the rest baseline is position-weighted -- and the bias GROWS after the lesion",
     "Frame-weighted rest baseline MINUS position-weighted rest baseline. Both arms are FLAT,"
     " so the only thing differing is COMPOSITION: `flatpool` weights every rest frame equally"
     " and therefore each position by however many rest frames it happened to supply, while"
     " `restw` weights the six positions equally. Rows are animals over a cohort row; columns"
     " are epochs."
     "\n\nONE MAP PER EPOCH, NOT SIX, and the algebra is why: flatpool_q - restw_q ="
     " (raw_q - flatpool) - (raw_q - restw) = restw - flatpool. The position's own data"
     " CANCELS, so the delta does not depend on position. It is the same cancellation that"
     " makes `reference each position to its own rest, then compare the maps` reduce to the"
     " 15x figure above."
     "\n\nTHIS IS NOT production-`rest` minus `restw`. That difference moves on the TEMPORAL"
     " axis as well (production rest is time-local, restw is flat), and a null across two"
     " changed variables says only that two different subtrahends agree."
     "\n\nWHAT IT SHOWS. The bias is 2.56x larger chronically than pre-stroke (cohort 0.00056"
     " -> 0.00145), larger in 3 of 3 animals with chronic data. Post-stroke the animal supplies"
     " fewer rest frames at the far positions, so a frame-weighted baseline drifts toward the"
     " near positions` rest -- the baseline changes WITH the deficit it is subtracted from."
     " Position-weighting is therefore load-bearing, not overhead: had the delta been flat, the"
     " honest verdict would have been correct-but-inert."),
    ("epoch_15f_rest_frozen_restdock05_durmatched_gapmatched.png",
     "Does the PRE-STROKE rest code survive, or is it REPLACED?",
     "A decoder is trained on PRE-STROKE rest periods, FROZEN, and applied across epochs"
     " (black); a second decoder is refit WITHIN each session on the SAME rest periods, with"
     " block-CV (grey, right panel). Both read the shared joint LocaNMF basis, so component i"
     " is the same cortical patch on every day -- a model frozen in one session`s own SVD basis"
     " and applied in another is not a degraded decoder but a meaningless one. Scored as"
     " BALANCED accuracy (mean of the six row recalls) and reported as the FRACTION OF"
     " ABOVE-CHANCE PERFORMANCE RETAINED, the normalisation that lets a 6-way problem be"
     " compared with the 3-way behavioural-state control."
     "\n\nWHAT THE PAIR SEPARATES. Frozen low with refit high means the information is PRESENT"
     " and the pre-stroke readout no longer points at it -- REPLACEMENT. Both low means the"
     " code is genuinely degraded. A per-session decoder alone cannot tell these apart, because"
     " recovery and replacement give it the same number."
     "\n\nWHAT IT SHOWS. Chronically the within-session refit reaches 1.181 of its own"
     " pre-stroke level while the frozen model reaches 0.611: position is in chronic rest, read"
     " by something other than the pre-stroke code. The task-side encoder says the same thing"
     " -- a refit ceiling of 0.783 against a pre-stroke 0.706, with the frozen model at 0.410."
     "\n\nCONTROLS, all four applied here: duration-matched (rest periods are not fixed"
     " length), lick-gap-matched, training-set-matched, and a BLOCK-PERMUTATION null rather"
     " than the analytic 1/6 -- positions come in ~6-trial blocks, so periods are not"
     " independent and 1/6 is the wrong reference."
     "\n\nDO NOT QUOTE THE ACUTE CELL. It is 16 sessions with PS95 contributing ONE, and"
     " removing 4% of periods moves it per animal by up to 0.21 (0.609 / 0.277 / 0.086 /"
     " -0.093). It is not a 4/4 replication. The CHRONIC result is the one that is stable"
     " across every control."),
    ("epoch_15f_rest_frozen_restdock05_durmatched_gapmatched_confusion.png",
     "Where the frozen rest decoder`s errors GO",
     "Confusion of the frozen pre-stroke rest decoder, pooled across animals, one panel per"
     " epoch with that epoch`s change from pre-stroke beneath it. Rows are the true position,"
     " panels are row-normalised for display while the stored matrices stay RAW COUNTS -- which"
     " is what makes pooling an epoch a sum rather than a mean of rates."
     "\n\nWHY IT EXISTS. A retained fraction is a scalar: it says how much position"
     " information survives and cannot say WHERE it goes. Errors scattering uniformly means the"
     " code DEGRADES; errors collapsing onto particular positions means it SHIFTS. The scalar"
     " is identical either way."
     "\n\nWHAT IT SHOWS. Acute errors COLLAPSE rather than scatter -- predictions pile onto"
     " near-contralateral in every row, and errors from the far rows move toward the near"
     " columns (0.56 of far-row error mass pre-stroke, 0.67 acutely)."
     "\n\nAND REST DISSOCIATES FROM TASK AT THE LESIONED POSITION. Far-contralateral is the"
     " BEST-decoded position in rest pre-stroke (0.50 diagonal) and it HOLDS UP (0.38 acute,"
     " 0.39 chronic) -- the opposite of far-contralateral in the task, which collapses from"
     " 1.46 to 0.47 of its pre-stroke amplitude. Rest and task carry position in largely"
     " different codes."),
    ("epoch_15s_shared_position_restdock05.png",
     "How much of the TASK map is already present in that position`s own REST?",
     "A regression coefficient per position, window and session: <rest_p - restw, trial_p -"
     " restw> / ||trial_p - restw||^2. One = the task map adds nothing its own rest did not"
     " already have; zero = they are orthogonal. Top row is the pre-cue window, bottom the"
     " post-cue. Shaded band is the circular-shift null (5th-95th percentile); dots are"
     " per-animal means."
     "\n\nTHE CIRCULARITY GUARD IS THE WHOLE DESIGN. `rest_p` and `trial_p` are disjoint"
     " frames but share the session`s slow drift, and shared drift alone produces a positive"
     " projection. So rest is taken from ODD position-blocks and trials from EVEN ones, drawn"
     " from interleaved and non-adjacent stretches of the session. The trial map is referenced"
     " to `restw`, never to the position`s own rest, which would make the projection circular"
     " by construction."
     "\n\nWHAT IT SHOWS, AND HOW SMALL IT IS. Post-cue is FLAT against its null at every"
     " epoch. Pre-cue is consistently positive -- +0.060 pre-stroke and +0.188 acute,"
     " null-corrected, 4/4 animals both times. But the effect is small: the COSINE between a"
     " position`s task map and its own rest map is 0.09 post-cue and 0.16 pre-cue, close to"
     " unrelated, and that is not a magnitude artefact (the amplitude ratio is 0.605 pre-cue,"
     " so a strong shape match would have shown). The useful reading is that rest carries"
     " position in a largely DIFFERENT spatial code from the task."
     "\n\nWHICH WAY THE SHARED COMPONENT POINTS -- this figure cannot say, and the"
     " block-boundary analysis can. Within a block the position just licked and the one coming"
     " next are the same; only at a boundary do they separate, and there rest resembles the"
     " position JUST LICKED AT (r_prev - r_next = +0.0772, 4/4 animals, 2,983 boundary"
     " periods). So the pre-cue window`s shared component is a trace of the LAST target, not"
     " preparation for the next -- a constraint on reading the pre-cue signal as anticipatory."
     "\n\nEXCLUDE SUBACUTE: it is PS92 alone (-0.639 against +0.212, +0.044, +0.027), and"
     " PS92 is the cohort`s SNR floor -- the six lowest rest fractions all belong to it."),
    # ---------------------------------------------------------------- THE ROTATION ARM
    # PLACED 2026-09-17. These twelve figures had been rendered for a week and referenced
    # NOWHERE -- the same failure as `locanmf_rsa`'s hemisphere panels, which ran every night
    # and were read by nobody until they were placed. The deck's completeness check cannot
    # catch it: it reports figures it EXPECTS and is silent about ones it was never told about.
    #
    # EXPLICIT FILENAMES, NOT A GLOB, and that is deliberate. `epoch_15h_rotation_maps_*.png`
    # would also match the `_mf075` and `_erodedgate` variants kept for comparison, and the
    # deck would show three versions of every arm with nothing on the slide saying which gate
    # produced which.
    ("epoch_15g_transfer_by_window.png",
     "Does the pre-stroke code still READ the post-stroke brain?",
     "A decoder is trained on PRE-STROKE sessions, FROZEN, and applied to each epoch, in all"
     " four windows: ENL (pre-cue), cue, lick, and rest. Accuracy minus the block-permutation"
     " null, so 0 is chance for that window`s own trial structure rather than an analytic 1/6"
     " -- positions run in ~6-trial blocks and are not independent."
     "\n\nWHAT THE FOUR WINDOWS BUY. A single window cannot separate `the code changed` from"
     " `the animal stopped doing the task`. ENL carries no movement, lick is movement-locked,"
     " and rest has no task at all; a change common to all four is not about the reach."
     "\n\nWHAT IT SHOWS. The fall is a PARTIAL ROTATION, not an erasure -- and every window"
     " ends CHRONICALLY ABOVE its own pre-stroke level. ADDITION was tested and rejected:"
     " re-fitting on post-stroke data does not simply add a new axis to the old one."
     "\n\nFRAMING (2), AND THAT IS A LIMIT. Every cell scores a position RELATIVE to the"
     " other five, so a change that hits all six equally is INVISIBLE here by construction."
     " `epoch_15k` is the framing-(1) companion that can see it. Per-cell numbers are in"
     " `epoch_15g_transfer_<window>_matrix.csv`."),
    ("epoch_15h_rotation_maps_ENL.png",
     "WHERE the ENL (pre-cue) position code turns",
     _ROT_LEGEND.format(arm="ENL / pre-cue", note=(
         "The pre-cue window carries real anticipatory position signal (LOSO 0.510), so this"
         " is a position map in its own right and not a baseline.")) ),
    ("epoch_15h_rotation_maps_cue.png",
     "WHERE the cue-evoked position code turns",
     _ROT_LEGEND.format(arm="cue", note=(
         "READ THE FAR-CONTRALATERAL ROW SEPARATELY. Averaging over the six positions is what"
         " hid this arm`s far-contra cosine of -0.042 among five values near +0.6, and"
         " far-contra is the lesion-relevant position.")) ),
    ("epoch_15h_rotation_maps_lick.png",
     "WHERE the lick-aligned position code turns",
     _ROT_LEGEND.format(arm="lick", note=(
         "CONDITIONED ON TRIALS THE ANIMAL LICKED, so each cell is the map of the attempts"
         " that HAPPENED -- a selection that changes across epochs. This arm is the most"
         " preserved of the four, which is consistent with it being the arm whose trials are"
         " selected for success.")) ),
    ("epoch_15h_rotation_maps_rest.png",
     "WHERE the RESTING position code turns",
     _ROT_LEGEND.format(arm="rest", note=(
         "The LEAST preserved of the four arms, and against the lowest noise ceiling -- rest"
         " maps are the noisiest, so a low cosine here is partly a measurement limit. Read it"
         " against its ceiling line, never against 1.0.")) ),
    ("epoch_15h_gain_vs_rotation.png",
     "ROTATION vs GAIN -- the two halves of `the code changed`",
     "One point per animal x spout position. X = the epoch`s Haufe-pattern amplitude relative"
     " to pre-stroke (1.0 = unchanged). Y = cosine between the epoch and pre-stroke patterns"
     " (1.0 = no rotation). Fill = p < 0.05 against the split-half null."
     "\n\nTHE FILL IS PER CELL AND UNCORRECTED, and on a plane of this many points some of it"
     " is chance. It marks which points to look at, not which to believe. The family-wise"
     " calls -- max-statistic across window x contrast x position -- and the cohort intervals"
     " are in `epoch_15h_rotation_cohort.csv`, and that table is what a claim should cite."
     "\n\nTHE TWO AXES ARE NEAR-ORTHOGONAL and all four quadrants are populated. If rotation"
     " and weakening were one phenomenon the diagonal quadrants would be empty; they are not."
     " Gain is >= 1 almost everywhere and RISES INTO CHRONIC, so the answer to `did the lesion"
     " degrade the code or move it` is MOVED, AND AMPLIFIED -- most in rest, least in lick."
     "\n\nGAIN IS NOT AN EVOKED RESPONSE AND IS NOT THE DEFICIT MEASURE. It is the DECODER"
     " pattern`s norm: second-moment, one-vs-rest, carrying the epoch-wide Cov(X) scale, and"
     " correlated with evoked amplitude at only r = +0.039. Roughly HALF of any gain number is"
     " how loud the epoch was. A position can become more SEPARABLE while its response"
     " collapses -- acute far-contra is dominated by unattempted trials. For the deficit read"
     " the rest-referenced maps and the encoder."
     "\n\nA CELL AT OR ABOVE ITS NOISE CEILING CARRIES NO ROTATION CLAIM IN EITHER"
     " DIRECTION. A cosine above the ceiling means the epoch pattern matches pre BETTER than"
     " two halves of pre match each other, which is only possible by chance -- such cells are"
     " UNMEASURABLE, not preserved. About a quarter of cells sit there, evenly spread across"
     " arms and epochs, which marks the resolution limit rather than a mis-estimated ceiling."
     " The result rests on the cells well below it, which is why `cos_attenuation_corrected`"
     " and `cos_p_vs_noise` are in `epoch_15h_rotation_regions.csv` and the raw cosine alone"
     " is not the statistic."),
    ("epoch_15j_rest_baseline_epoch_drift.png",
     "Does the REST BASELINE move across epochs, and does it matter?",
     "The subtrahend under every `_RESTWref_` map, measured as a quantity in its own right."
     " Each session`s `restw` baseline is divided by its OWN evoked norm before averaging,"
     " which is what cancels the cross-day MULTIPLICATIVE scaling no subtraction touches;"
     " without it this would largely report how bright the window was."
     "\n\nREAD AGAINST THE NULL, NEVER AGAINST ZERO. Baselines differ session to session for"
     " reasons unrelated to any lesion, so the shift is nonzero for ANY two groups of"
     " sessions. The null splits the PRE sessions into a group of the epoch`s size and the"
     " remainder and computes the identical statistic -- so the bar is read against that"
     " animal`s ordinary session-to-session spread. The null is conservative by construction"
     " (its second group is smaller, so its differences run slightly large): it can hide a"
     " real shift, it cannot manufacture one."
     "\n\nTOP ROW -- DOES IT MOVE? YES, in 6 of 11 animal-epoch cells. But it is a whole-map"
     " vector norm and therefore an UPPER BOUND on the amplitude bias -- attained only if the"
     " shift ALIGNS with the evoked pattern, while an orthogonal shift costs ~r^2/2. So a"
     " shift of 0.15 means somewhere between ~1% and ~15%. ACUTE is the LEAST affected epoch"
     " despite carrying the largest deficit, so this does not simply track the lesion."
     "\n\nBOTTOM ROW -- DOES IT MATTER? This is the arm that decides WHICH END of that range"
     " you are at, and it reached a slide only on 2026-09-19; before that the p existed in the"
     " CSV alone. 1 of 11 cells clears p < 0.05 against ~0.6 expected by chance: NO EVIDENCE"
     " that the drift is preferentially aligned with the evoked pattern, so the realistic bias"
     " sits toward the LOW end. THE TWO ROWS ARE DIFFERENT QUESTIONS AND THE SECOND IS NOT A"
     " RETRACTION OF THE FIRST -- the baseline does move; the movement is not pointed at the"
     " signal."
     "\n\nTHE COSINE GOT A p ON 2026-09-18 AND ONLY BECAUSE `e_ref` IS HELD OUT -- built from"
     " the other three animals` pre sessions, which is possible because these are atlas PIXELS"
     " on a shared grid. Self-referenced, the observed shared `-mean_pre(b)` with `e_ref` in"
     " FULL while a within-pre null shared it only partly, so the p ran anti-conservative:"
     " false-positive rate 0.160 against a nominal 0.05, versus 0.060 held out. The cost is"
     " comparability -- an animal is scored against the COHORT-TYPICAL evoked pattern rather"
     " than its own, attenuating the cosine by 6-14% (mean pairwise r = +0.868, worst PS93)."
     " That can HIDE a weak alignment; it cannot manufacture an absent one, so the null result"
     " stands as absence of evidence on a slightly blunted test."
     "\n\nSTILL NEVER READ THE COSINE AGAINST ZERO. Its own null median is ~0.21, not 0,"
     " because pre baselines are not isotropic -- they occupy a low-dimensional, cortically"
     " structured subspace that overlaps the evoked pattern, and that geometry survives any"
     " choice of reference. A bar of +0.4 is BELOW chance."
     "\n\nTHE SIGN IS NOT DECORATION. `map = post - (b_pre + d)`, so cos > 0 SHRINKS the"
     " measured amplitude and cos < 0 INFLATES it. The one cell clearing p < 0.05 is PS94"
     " subacute at cos = -0.913, which is also the largest shift in the table (0.432) -- so if"
     " it is real rather than the one false positive 11 uncorrected tests predict, PS94`s"
     " subacute amplitude is OVERSTATED by ~39%, not understated."
     "\n\nBETWEEN-POSITION contrasts are unaffected either way -- the same subtrahend is"
     " removed from all six. It is ACROSS-EPOCH AMPLITUDE comparisons that carry this, which"
     " is precisely why `epoch_15k` runs `restw` beside two references that cannot drift."),
    ("epoch_15k_reference_families_cue.png",
     "WHERE the CUE position map changes -- three references, three failure modes",
     _REF_LEGEND.format(arm="cue-aligned, so every trial contributes whether or not the animal"
                            " responded")),
    ("epoch_15k_agreed_ccf_cue.png",
     "The CUE regions ALL THREE references agree on, on the Allen CCF",
     _CCF_LEGEND.format(arm="cue")),
    ("epoch_15k_reference_families_lick.png",
     "WHERE the LICK position map changes -- three references, three failure modes",
     _REF_LEGEND.format(arm="conditioned on trials the animal LICKED, so a cell is the map of"
                            " the attempts that HAPPENED -- acute far-contra has no cell at"
                            " all because there were none")),
    ("epoch_15k_agreed_ccf_lick.png",
     "The LICK regions ALL THREE references agree on, on the Allen CCF",
     _CCF_LEGEND.format(arm="lick")),
    ("epoch_15k_reference_families_precue.png",
     "WHERE the PRE-CUE position map changes -- TWO references, not three",
     _REF_LEGEND.format(arm="the ENL window [cue - 2 s, cue]")
     + "\n\nTWO FAMILIES HERE, AND THE MISSING ONE IS THE POINT. Under the PRE-CUE alignment"
       " the `precue` REFERENCE would subtract the feature window from its own final second --"
       " self-referential -- so it is dropped at source. The two that remain, `raw` and"
       " `restw`, are the pair correlated at r = +0.95, so AGREEMENT HERE IS CLOSE TO ONE"
       " REFERENCE SAYING SO. Read these panels as one reference with a consistency check,"
       " not as two independent ones."),
    ("epoch_15k_agreed_ccf_precue.png",
     "The PRE-CUE regions BOTH references agree on, on the Allen CCF",
     _CCF_LEGEND.format(arm="pre-cue")
     + "\n\nTWO references here, not three -- see the preceding slide for why, and for why"
       " that makes this arm`s agreement gate much weaker than the cue and lick arms`."),

    # ---- THE 415 nm AUDIT (2026-09-19) ----------------------------------------------------
    ("epoch_20_channel_evoked_sign.png",
     "WHAT THE 415 nm CHANNEL ACTUALLY CARRIES -- cue-aligned, by animal",
     "415 IS BIPHASIC. A large FAST component peaking at 0.26-0.29 s, BEFORE the 470 calcium"
     " peak at 0.38 s and far too early for any vascular response, then a slower component"
     " from ~1 s."
     "\n\nTHE LATE DEFLECTION DISAGREES ACROSS ANIMALS, which is why no single sign can be"
     " quoted: PS93 goes clearly negative (-0.57% while 470 is still +0.79%), PS95 barely"
     " dips, PS92 reaches only -0.20%, and PS94 NEVER GOES NEGATIVE, holding +0.9% out to 3 s."
     "\n\nCues recur every few seconds, so a pre-cue baseline carries the previous trial`s"
     " tail and PS94`s 470 does not return to baseline within 3 s either -- part of the late"
     " plateau may be trial structure rather than physiology."),
    ("epoch_21_channel_vessel_sign.png",
     "IS THE 415 RESPONSE VESSEL-SHAPED OR PARENCHYMA-SHAPED?",
     "THE EARLY COMPONENT IS CALCIUM BLEED-THROUGH, and the cleanest evidence is a matched"
     " RATIO: at 0.4 s the vessel/parenchyma dilution is 0.41 in the 470 map against 0.43 in"
     " the 415 map (PS93; 0.40 against 0.43 for PS92). A vascular signal has no reason to"
     " dilute by exactly the factor the calcium signal does. A surface vein carries no GCaMP,"
     " so what the camera collects there is scattered light from the cortex beside and beneath"
     " it."
     "\n\nLATE, THE PARENCHYMA GOES NEGATIVE WHILE THE VESSELS GO FLAT -- PS93 at 1.2 s is"
     " -0.84% in parenchyma against 470`s +0.67%. So the VEINS ARE NOT INCREASING, they are"
     " NOT CHANGING while everything around them decreases, which on a diverging colormap"
     " renders as a bright streak. The observation is real; the reading is relative."
     "\n\nTHE VESSEL MASK IS LOCAL-CONTRAST, not an intensity threshold. A darkest-decile"
     " mask selected the dim anterior edge (row 183 +/- 198, mean intensity 2046 against the"
     " brain`s 13048) and only 49% of it overlapped real vasculature."),

    ("epoch_16_nvc_evoked.png",
     "CUE-EVOKED RAW 415 AND 470 at frame resolution -- is 415 isosbestic HERE?",
     "TIMING SETTLES WHAT AMPLITUDE CANNOT. Calcium is fast (hundreds of ms), haemodynamics"
     " slow (peaks 1-2 s). If 415 sits BELOW GCaMP`s neutral/anionic crossing there is an"
     " EARLY NEGATIVE deflection before the slow positive one; if 415 were truly isosbestic"
     " there is no early dip at all. The +0.54% to +1.96% cue-evoked 415 rises quoted"
     " elsewhere are WINDOW AVERAGES over the whole post-cue period, which average an early"
     " dip away completely -- which is exactly why they looked like a clean positive"
     " haemodynamic response."
     "\n\nTHE LITERATURE DISAGREES ON WHERE THE CROSSING SITS, and two of three estimates"
     " put our 415 BELOW it: conventional photometry practice 405-415; Simpson et al. 2024"
     " (Neuron primer) 420-430 for GCaMP6; Barnett/Drobizhev 2017 440-450 for GCaMP6m with NO"
     " true isosbestic point. Simpson also names this observable directly, from a 405 nm"
     " control against GCaMP6f: significant negative bleed-through `evident as a NEGATIVE PEAK"
     " IN THE EVENT-ALIGNED AVERAGE`."
     "\n\nREAD WITH `epoch_20` AND `epoch_21`, which reach the same conclusion from the"
     " time course and from the vessel/parenchyma dilution ratio."
     "\n\nNOT NORMALISED, DELIBERATELY. `U @ SVT` reconstructs the DEVIATION from each"
     " channel`s mean, so the reconstructed means are ZERO and dividing by them is division by"
     " ~0 -- a trap already paid here once (+692% and -2881% evoked responses). It is ALREADY"
     " fractional; the fix is not a different denominator, it is NO denominator."),
    # THE POOLED HALF OF `channel_position_maps`, REGISTERED 2026-09-21. These two have existed
    # since 2026-09-19 and had never been on a slide, because the module wrote them to
    # `labcams/channel_comparison` -- the directory its 2026-07-08 ancestor used -- and no deck
    # reads it. They now go to `grant_figures/epoch` like everything else in this family. The
    # SIXTEEN per-session maps stay in `channel_comparison`: they are channel-identity
    # diagnostics, not deck material, and moving them here would hand the coverage report sixteen
    # unregistered files in exchange for the one figure that was actually missing.
    ("channel_position_epoch_cue.png",
     "415 vs 470 BY SPOUT POSITION AND EPOCH -- cue-aligned",
     "ROW 3 IS THE COUPLING MEASURE AND ROWS 1-2 ARE WHY IT CANNOT BE READ ALONE."
     " ||415||/||470raw|| is haemodynamic response per unit neural response -- but A RATIO"
     " RISES WHEN ITS DENOMINATOR FALLS, and a stroke is expected to lower the 470 response."
     " Row 3 up while row 1 is down and row 2 holds is a shrinking denominator, not better"
     " coupling. Rows 1 and 2 carry the cross-day scaling confound (expression, bleaching,"
     " window clarity) that the ratio exists to cancel -- `crossday_intensity` owns that --"
     " so read them as diagnostics for row 3, never as amplitudes in their own right."
     "\n\nROW 5 IS THE CONTROL FOR ROW 4. r(415, 470) rising acutely looks like stronger"
     " coupling, but acute maps are also more GLOBAL (`epoch_15k` puts the cue acute panel at"
     " ~75% global) and two broad blobs correlate for reasons unrelated to coupling. Row 5"
     " subtracts r(415 here, 470 at the OTHER positions): if the rise survives it is"
     " position-specific; if row 5 is flat at zero the rise was globalness."
     "\n\nTHE DOTS ARE THE POINT, NOT DECORATION -- one per session, coloured by animal, so a"
     " cell carried by one animal with six sessions cannot read as four animals agreeing."
     " Acute far-contra is thin in event count for every animal and absent for some."
     "\n\nTHE SEM IS ACROSS SESSIONS AND IS THE WRONG ERROR BAR FOR A COHORT CLAIM: sessions"
     " within an animal are not independent, so it runs narrower than the animals->sessions"
     " bootstrap CI that `epoch_summary` prints. CITE THE BOOTSTRAP; read the dots for who"
     " carries the cell. Both are on the page so neither can be mistaken for the other."
     "\n\nWHAT THIS FIGURE CANNOT DO, and the module says so before it says anything else: it"
     " does NOT adjudicate whether 415 carries calcium. A 415 map resembling the 470 map is the"
     " EXPECTED result under a correction that is working perfectly -- `a spatial pattern of"
     " correlation is what we would expect with neurovascular coupling` (Priya, 2026-09-19)."
     " Both accounts, scaled calcium and NVC that scales with calcium, predict a positive"
     " spatial correlation AND an amplitude-invariant ratio. ONLY LATENCY SEPARATES THEM --"
     " see `epoch_16`. Source: scripts/rest_migration/channel_position_maps.py --epochs."),
    ("channel_position_epoch_lick.png",
     "415 vs 470 BY SPOUT POSITION AND EPOCH -- lick-aligned",
     "THE LICK-ALIGNED ARM of the panel above; read its legend for what each row is and for"
     " what this measurement cannot settle. The two are registered together because a coupling"
     " change that appears in ONE alignment and not the other is about the movement rather than"
     " the vasculature -- the comparison IS the result."
     " Source: scripts/rest_migration/channel_position_maps.py --epochs."),

    # ---- ENGAGEMENT: the quit, and what declines within a session (2026-09-20) ------------
    ("epoch_23_quit_prodrome_gated_h90.png",
     "IS THE QUIT A STEP OR AN ACCUMULATION? Lick rate across the session",
     "A RAMP AND THEN A STEP. Against TIME-MATCHED controls from the SAME ANIMAL -- sessions"
     " still engaged at that same absolute session time -- lick rate in the ten minutes before"
     " a quit is -18.7 licks/min [-22.9, -13.7] (45 quitters, 705 pairings). The decline"
     " starts 20-30 min out and collapses at the quit itself (24 -> 6 licks/min)."
     "\n\nTHE CONTROL IS THE WHOLE POINT. Lick rate declines in EVERY session, so a"
     " quit-aligned average on its own always shows a ramp and proves nothing."
     "\n\nDO NOT READ THE ACUTE `sessions with NO quit` PANEL. Censored means no DETECTED"
     " quit, and `engagement_gate` needs a sustained non-recovering run, so an animal stopping"
     " near the end leaves too little tail. Measured as each censored session`s final 10 min"
     " over its own mid-session rate: pre 0.77 (29 sessions), chronic 0.86 (18), and ACUTE"
     " 0.48 from FOUR sessions THREE of which fall below half."),
    ("epoch_24_session_quintiles_gated_h90.png",
     "LICKS PER TRIAL and INTER-LICK INTERVAL by session quintile",
     "THE SAME BINS AS THE HIT-RATE QUINTILES in `engagement_decomposition`, which is the"
     " point: across the quintiles where acute near-spout HIT RATE falls 0.972 / 0.937 / 0.792"
     " / 0.457, the INTER-LICK INTERVAL moves 180.7 / 180.1 / 179.3 / 178.3. A threefold"
     " collapse in performance against a 1.3% change in motor speed."
     "\n\nTOP licks per trial = engagement AND motor. BOTTOM inter-lick interval = the"
     " rhythm`s PERIOD only."
     "\n\nQ5 ILI IS SURVIVORSHIP-CONTAMINATED: ILI needs >= 4 licks/trial and acute Q5"
     " averages 7.9, so the survivors are the high-lick trials and the exclusion rate is"
     " epoch-dependent. Q1-Q4 is the trustworthy span. FAR-spout ILI is separately unreliable"
     " post-stroke, because contact-based detection scores a MISSED lick as a long interval."),
    ("epoch_25_quintiles_delta_from_pre_gated_h90.png",
     "WITHIN-ANIMAL DELTA FROM PRE, by quintile -- offset versus divergence",
     "THE LEVELS ARE NOT COMPARABLE ACROSS ANIMALS AND THE DIFFERENCES ARE. Pre-stroke"
     " near-spout ILI runs 151 ms (PS95) to 173 ms (PS93), so a cohort mean of levels is"
     " mostly determined by which animals land in each cell."
     "\n\nIT SEPARATES AN OFFSET FROM A DIVERGENCE. Acute FAR is a flat -6 licks/trial at"
     " every quintile -- a constant deficit. Acute NEAR starts at its own pre level and"
     " crosses below by Q3 -- a within-session divergence. Acute ILI is a flat +12-13 ms"
     " offset, significant in 4 of 5 quintiles and ALREADY PRESENT IN Q1."
     "\n\nIT ALSO CORRECTS A READING: in raw levels subacute near ran 20.9 -> 11.5, the"
     " steepest-looking decline of any epoch. Against each animal`s own baseline it is ABOVE"
     " pre at every quintile. The apparent collapse was baseline spread."
     "\n\nLAST COLUMN: change in the Q5-Q1 gap. The session steepens significantly at"
     " SUBACUTE (-6.8 near, -5.8 far), not acute."),

    # ---- IS THE DECLINE FATIGUE? The bout decomposition (2026-09-20) ----------------------
    ("epoch_26_within_bout_deceleration.png",
     "WITHIN-BOUT DECELERATION -- the motor deficit the median ILI hid",
     "READ THE BOTTOM (DELTA) ROW. The last third of a bout`s intervals minus its first third,"
     " as a WITHIN-ANIMAL delta from pre: +15 to +25 ms at EVERY EPOCH INCLUDING CHRONIC, at"
     " every quintile, in all four animals -- and it does NOT recover."
     "\n\nTHE DISSOCIATION IS THE FINDING. Median ILI is elevated acutely (+12.5 ms) and"
     " RECOVERS by subacute (+1.5 [-4.9, +7.7]); this does not. The average cycle period comes"
     " back; the ability to sustain a bout without slowing does not. The per-trial MEDIAN is"
     " exactly the statistic that averages end-of-bout slowing away, which is why this went"
     " unseen (Priya: the ILI is gated by a CPG, so effort may shorten bouts rather than slow"
     " the rhythm)."
     "\n\nACUTE ADDITIONALLY STEEPENS ACROSS THE SESSION, +10.1 [+0.9, +19.1] -- the fatigue"
     " signature proper."
     "\n\nTHE MISS ARTEFACT DOES NOT EXPLAIN IT. Contact-based detection turns a missed lick"
     " into a doubled interval, but CHRONIC HIT RATE HAS RECOVERED while chronic slope is"
     " still +15.8 to +20.8 and significant in all five quintiles. That is an argument, not a"
     " control -- DLC tongue tracking would make it one."),
    ("epoch_26_bouts_per_trial.png",
     "BOUTS PER TRIAL -- how many bouts are INITIATED",
     "READ THE BOTTOM (DELTA) ROW; the raw levels are cohort means over DIFFERENT ANIMAL SETS,"
     " and chronic near sits at 3.36 bouts against far`s 8.41 where no other epoch splits that"
     " way -- composition, not biology."
     "\n\nTHE Q5-Q1 CHANGE CLEARS ZERO NOWHERE (acute near -1.42 [-3.62, +0.22]). Read"
     " without intervals this looked like `the acute decline is entirely fewer bouts`, and it"
     " is not. At acute, bout LENGTH is a well-constrained null (-0.10 [-0.94, +0.66]) while"
     " bout COUNT is unconstrained, so initiation remains the better guess -- a guess."),
    ("epoch_26_licks_per_bout.png",
     "LICKS PER BOUT -- is a bout CUT SHORT as the session wears on?",
     "THE FATIGUE-COMPATIBLE DIRECTION, and the only place in this decomposition that clears"
     " zero: Q5-Q1 change from pre is -1.31 [-2.80, -0.04] at subacute near and"
     " -1.36 [-2.53, -0.20] at chronic near. Acute and all FAR cells are null."
     "\n\nREAD WITH `epoch_26_within_bout_deceleration`, not alone: bouts shortening AND"
     " decelerating is a fatigue account; either on its own is weaker."),
)
#: Legend for the interval companions, which share one form and should not repeat it.
_CI_LEGEND = ("Epoch minus pre-stroke for each quantity in the preceding figure. Point, the "
              "difference in the data; thick bar, 95% interval; thin bar, the "
              "Bonferroni-corrected interval across all comparisons on the figure -- both "
              "summarised from one set of bootstrap draws, so the corrected interval "
              "necessarily contains the uncorrected one. Zero is drawn.")


# ---------------------------------------------------------------------------------------------
# SECTION H — THE GRANT SUMMARY SET
# ---------------------------------------------------------------------------------------------
# THEY ARE DELIBERATELY CAVEAT-LIGHT, which is the opposite of the rest of this deck. Each makes
# ONE point for a reader who has not been in the weeds; the caveats live in the module docstring
# and in DECISIONS.md. The speaker notes below carry the ones that would change a reading.
GRANT_FIGURES = (
    ("grant_1b_behaviour_pre_collapsed.png",
     "H1. Licking accuracy per spout position, pre vs post",
     ("The whole pre-stroke baseline as ONE point per position (mean +/- SEM across sessions), "
     "then each day after the lesion. Engaged trials only, so the terminal quit period is "
     "excluded. This is the deficit every later panel is trying to explain.")),
    ("grant_1_behaviour_by_position.png",
     "H1b. The same, every pre-stroke session shown",
     ("Use this to check the baseline is flat before trusting H1's collapsed point. June is "
     "collapsed to one marker because the true axis would spend 85% of its width on empty "
     "space. Pre and post are drawn as SEPARATE segments: nothing was recorded between the "
     "last baseline session and the first post-stroke one, and joining them would draw a "
     "decline that was never measured.")),
    ("grant_2b_prestroke_crossday_cohort.png",
     "H2. Position decodes across sessions (pre-stroke, cohort)",
     ("Leave-one-session-out in the shared joint-LocaNMF basis: every trial scored by a decoder "
     "that never saw its session. THE ANIMAL IS THE UNIT -- bar = mean of the four per-animal "
     "accuracies, error bar = SEM across animals. Pooling all ~44 held-out sessions would give "
     "a far tighter interval describing how much a SESSION varies, not an ANIMAL.")),
    ("grant_2_prestroke_crossday_decoding.png",
     "H2b. The same, per animal, with every held-out session",
     ("Each dot is one held-out session. Read it beside H2 to see whether an animal's mean "
     "rests on a tight cluster or a spread.")),
    ("grant_3a_coding_retained.png",
     "H3. How much of each position's pre-stroke code survives, over days",
     ("Cosine between the post-stroke pairwise position axis and its pre-stroke reference, "
     "DISATTENUATED by each side's own split-half reliability -- a raw cosine confounds a lost "
     "code with a noisy estimate of a preserved one, and the post-stroke arms are small. The "
     "matched null (pooled vs held-out PRE-stroke sessions: PS92 0.79, PS93 0.93, PS94 0.89, "
     "PS95 0.84) is the line to beat, NOT 1.0 -- two pre-stroke sessions do not reproduce each "
     "other perfectly either. SOURCE IS coding_direction.json, which the footer reports; it can "
     "lag the config.")),
    ("grant_3b_frozen_vs_within.png",
     "H3b. Frozen decoder vs a decoder retrained within each session",
     ("The complement to H3: if a within-session decoder recovers accuracy the frozen one lost, "
     "the information is still present and only the READOUT has moved; if both fall, the "
     "information itself is degraded. Read the gap, not either line alone. Built from "
     "section_g.json.")),
    ("grant_4_confusion_prestroke_*.png",
     "H4. Pre-stroke cross-session confusion, per window",
     ("The strongest and least contestable result in the set: no lesion, no trial-class "
     "definitions, no engagement gate, no alignment inference. Counts summed over held-out "
     "sessions then row-normalised, so a 500-trial session is not weighted like a 200-trial "
     "one. Rows = TRUE position, columns = PREDICTED.")),
    # ------------------------------------------------------------------------------------
    # CUT FROM THE DECK 2026-09-13 (Priya): H5, H5b, H6, H8d, H8e, H10, H10b.
    #
    # THE FIGURES ARE STILL RENDERED. `grant_figures` writes every one of them each night and
    # nothing here changes that -- these entries only decide what the deck PLACES. Restoring a
    # family is un-commenting its entry, and the figure it points at is already on disk.
    #
    # WHY EACH ONE GOES (all supersession, none withdrawn):
    #   H5/H5b  the pooled pre-vs-post confusion -- section I's epoch-stratified I4/I5 carry
    #           the same comparison with bootstrap intervals and a per-epoch break that H5's
    #           single pre/post split cannot express.
    #   H6      the mean-pattern similarity matrix. H6b (session by session) is already marked
    #           PRIMARY OVER H6 in its own blurb, so H6 was the superseded member of its pair.
    #   H8d     H8's change-from-pre-stroke. H8 stays; the delta is recoverable from it.
    #   H8e     the asymmetry test, five slides answering a question nothing downstream reads.
    #   H10/H10b the best-match family -- WHICH pre-stroke position each post-stroke position
    #           resembles. Superseded by the map-level analyses in `WHERE_THE_CODE_MOVES.md`,
    #           which answer "where does the displaced code go" in pixels rather than in a
    #           6x6 index, and with the nested bootstrap these panels never had.
    # ------------------------------------------------------------------------------------
    # CUT: ("grant_5_confusion_pre_post_*.png", "H5. The frozen decoder before and after the
    #       lesion", ...)   -- superseded by I4/I5
    # CUT: ("grant_5b_confusion_working_*.png", "H5b. The same with the terminal quit period
    #       removed", ...)  -- superseded by I4/I5
    # `*_*` REQUIRES THE ARM TOKEN. These figures gained a `_lick`/`_working` suffix on
    # 2026-08-28, and the superseded `..._<align>.png` files from 8/26 are still on MICROSCOPE
    # (which is never deleted from). A single `*` globs both, so the deck placed four slides of
    # 8/26 data among 93 current ones -- undetectable by eye and invisible to the "0 missing"
    # count, because a stale figure is present. This is the 2026-08-24 `coding_engagement_*`
    # bug inverted: that pattern was too tight and dropped twelve slides silently, this one was
    # too loose and added four.
    ("grant_5c_confusion_per_session_*_*.png",
     "H5c. The frozen decoder before and after the lesion, session by session",
     ("ONE PANEL PER SESSION of the frozen pre-stroke decoder applied after the lesion. A pooled pre-versus-post panel would average a moving target: PS94 runs 0.39 to 0.76 across six days. "
     "Columns are DAYS FROM LESION so a column means the same thing in every row even though "
     "the animals were lesioned on different dates.")),
    ("grant_5d_confusion_delta_*_*.png",
     "H5d. The same, as CHANGE from pre-stroke",
     ("H5c minus its own first column, cell by cell. The pre-stroke confusion is far from "
     "uniform -- close positions are confusable with each other and far ones are not -- so an "
     "absolute post-stroke cell of 0.3 means different things in different places. Subtracting "
     "removes the baseline texture and leaves only what the lesion did: a negative DIAGONAL "
     "cell is recall lost at that position, and the positive cell in the SAME ROW says where "
     "those trials went instead. Two colour bars: column 1 is in probability, the rest in "
     "change of probability.")),
    # CUT 2026-09-13 (Priya) -- see the cut block above. H6:
    # ("grant_6_pattern_*.png",
    # "H6. Mean-pattern similarity, within and across positions",
    # ("The model-free counterpart to the coding directions, and it fails differently: a coding "
    # "axis needs a contrast and so breaks at exactly the impaired positions, while a mean "
    # "pattern for far_R is well defined from miss trials with no partner. Conversely THIS is "
    # "sensitive to global gain and the coding directions are not. Agreement between them is "
    # "the claim worth making. ROWS = the post-stroke pattern, COLUMNS = the pre-stroke "
    # "reference -- the opposite convention to the confusion matrices above. Green ring = beats "
    # "a position-label permutation null. Third panel = post minus baseline, differenced draw by "
    # "draw. Bootstrap resamples TRIALS WITHIN SESSIONS and does NOT resample sessions, because "
    # "days are not exchangeable when the animal is recovering. THE BASELINE PANEL SPLITS "
    # "PRE-STROKE SESSIONS, not pre-stroke trials (corrected 2026-08-25): a random trial split "
    # "puts both halves on the SAME DAYS, so it carries no day-to-day drift and is a ceiling no "
    # "across-day comparison can reach.")),
    ("grant_6b_pattern_per_session_*.png",
     "H6b. Mean-pattern similarity, within and across positions, session by session",
     ("THE ANCHOR OF THIS FAMILY, and the figure H7, H7b, H8 and H11 are read against. Resolving it per session rather than pooling is the point: when sessions move, the trajectory IS the result. Single-session mean "
     "patterns are noisier, so cells under 10 trials are blank rather than drawn. FIRST COLUMN "
     "IS LEAVE-ONE-SESSION-OUT (corrected 2026-08-25) -- each pre-stroke session against the "
     "pool of the others, averaged, which is one session against other days exactly like every "
     "post column. It is the CEILING and it is NOT 1.0: two pre-stroke days differ by ordinary "
     "drift, so a post column must be read against it and never against unity.")),
    ("grant_6d_pattern_delta_*.png",
     "H6d. The same, as CHANGE from pre-stroke",
     ("H6b minus its own leave-one-session-out first column. ZERO means this day looks exactly "
     "as much like the pre-stroke reference as one pre-stroke day looks like the others -- the "
     "honest null, which is NOT a correlation of 1. READ THE OFF-DIAGONAL: a negative diagonal "
     "cell says the position lost its own code, while a positive cell at (far_R, far_L) says "
     "far_R trials came to look more like pre-stroke far_L than they used to. That "
     "substitution is legible at a glance here and only from memory in the absolute panel. "
     "r is differenced, NOT r-squared -- squaring would erase the sign the substitution lives "
     "in.")),
    ("grant_7_splithalf_*.png",
     "H7. WITHIN-session split-half similarity — the ceiling H6b is measured against",
     ("Both halves come from the SAME session, so no lesion comparison, no pre-stroke "
     "reference and no alignment inference enters this. The diagonal is that session's own "
     "reliability, which is the CEILING any correlation involving its mean pattern can reach. "
     "H6b cannot distinguish a code that MOVED from one that merely became NOISIER, and a "
     "graded drop at every position is exactly what a global change in repeatability looks "
     "like.")),
    ("grant_7d_splithalf_delta_*.png",
     "H7d. The same, as CHANGE from pre-stroke — read this against H6d",
     ("THE CONTROL IN ITS MOST DIRECT FORM, and the pair to put side by side. If H6d's fall "
     "were really a reliability story, the diagonal HERE would fall by a comparable amount at "
     "the same positions on the same days, because both halves come from the same session and "
     "nothing about the lesion enters a single panel. Where H6d falls and this does not, the "
     "code MOVED. Where both fall together, the code is noisier and H6b cannot tell the "
     "difference on its own.")),
    ("grant_7b_reliability_*.png",
     "H7b. Moved code or noisier code? — H6b's diagonal, disattenuated",
     ("THE VERDICT PANEL for H6b. Right = middle divided by sqrt(rel_post x rel_pre): what the "
     "correlation would be if both means were noise-free. A drop that SURVIVES it is a code "
     "that moved; a drop that DISAPPEARS was a code measured less repeatably. Same correction "
     "the coding directions have used since 2026-08-20, which the pattern measure never had. "
     "A grey dot marks reliability below 0.5 on one side, where the ratio is not stable enough "
     "to print -- and that is worst exactly at the impaired positions, where the question is "
     "sharpest.")),
    ("grant_8_crossnobis_*.png",
     "H8. The mean-pattern similarity matrix rebuilt on cross-validated (crossnobis) distances",
     ("Same layout as H6b -- rows = post-stroke position, columns = pre-stroke reference -- but "
     "as NOISE-UNBIASED distance rather than correlation, so a noisier session does not read "
     "as a bigger change. LOW on the diagonal = the pattern did not move. Units are the mean "
     "pre-stroke between-position distance for that animal, because raw crossnobis units "
     "depend on the whitener and the dimensionality and are not comparable across animals. "
     "Still NOT gain-invariant: it is a distance between two patterns. H8b is that "
     "companion.")),
    # CUT 2026-09-13 (Priya) -- see the cut block above. H8d:
    # ("grant_8d_crossnobis_delta_*.png",
    # "H8d. The same, as CHANGE from pre-stroke",
    # ("H8 minus its own first column. POSITIVE = further from the pre-stroke pattern than a "
    # "held-out pre-stroke session is; NEGATIVE = closer. The reference distances are not "
    # "uniform -- close positions sit nearer each other than far ones -- so an absolute cell of "
    # "1.2 means different things in different places. A NEGATIVE OFF-DIAGONAL cell is a "
    # "substitution: that row's trials moved TOWARD the column's pre-stroke pattern.")),
    # CUT 2026-09-13 (Priya) -- see the cut block above. H8e:
    # ("grant_8e_asymmetry_*.png",
    # "H8e. Is the distance matrix ASYMMETRIC, and where?",
    # ("A[P,Q] = d(post at P, pre at Q) minus d(post at Q, pre at P). Rows and columns index "
    # "genuinely different sets, so symmetry is NOT expected and the gap between the two "
    # "orderings is the substitution: which way a position moved, not merely that it moved. "
    # "Green ring = the 95% block-bootstrap interval excludes zero; the count above each panel "
    # "is rung pairs out of 15. READ THE PRE COLUMN FIRST -- both sides there estimate the same "
    # "patterns, so its asymmetry has expectation zero and its rings are this construction's own "
    # "false-positive rate. Unlike H7 this must NOT be symmetrised: there the two cells "
    # "estimated one quantity and averaging them was strictly better; here they estimate "
    # "different quantities and the difference IS the result.")),
    ("grant_9_delta_trajectory_*.png",
     "H9. The bootstrap results, plotted — change from pre-stroke over days",
     ("THE SUMMARY SLIDE OF THE DELTA SET. The intervals in H6d and H7d sit as text above 6x6 "
     "matrices, which is the wrong shape for what they answer: whether a position is "
     "recovering, holding or worsening is a TRAJECTORY. Left = mean own-position change per "
     "day with its 95% block-bootstrap interval; right = the same split BY POSITION, which is "
     "where the deficit lives. ZERO IS A REAL NULL: it means this day differs from the "
     "pre-stroke reference no more than one pre-stroke day differs from the others, because "
     "the baseline is leave-one-session-out and not a correlation of 1. Blocks are resampled "
     "within session and SESSIONS ARE NOT RESAMPLED, so an interval is conditional on these "
     "days -- the trajectory is what speaks to days not recorded.")),
    ("grant_8b_crossnobis_geometry_*.png",
     "H8b. Second-order RSA — the gain-invariant test",
     ("Each session's OWN 6x6 crossnobis RDM correlated against the pre-stroke RDM. This is "
     "RSA proper, and scaling every distance leaves it unchanged, so a uniform post-stroke "
     "amplitude change CANNOT move it. That matters because H6b's headline (every position "
     "drops, far_R most) is precisely the signature a global change would leave. Per-position "
     "information survives in a weaker form: each position's ROW is its five distances to the "
     "others, so 'is far_R still arranged the way it was' is answerable and 'did far_R's "
     "pattern move' is not -- that question belongs to H8.")),
    ("grant_8g_geometry_by_position_*.png",
     "H8g. H8b split BY POSITION — one panel per spout, animals as lines",
     ("H8b's left panel is already per position, but as an animal-major heatmap: comparing one "
     "position ACROSS animals means reading four separate blocks. Here each position is a "
     "panel and each animal a line, which is the comparison the deficit is about. DASHED = "
     "that animal's own leave-one-session-out pre-stroke ceiling for that position; read a "
     "trace against its own dashed line, never against 1. SHADED = 95% block bootstrap. "
     "A GAP IS NOT A ZERO: a row correlation needs 4 of its 5 partner positions, so a session "
     "missing two positions has EVERY row uncomputable, including the positions the animal "
     "licked normally. That is a property of the estimator, not of the animal, and it is why "
     "whole days vanish for PS94 in the lick class; the `working` class fills most of them.")),
    # CUT 2026-09-13 (Priya) -- see the cut block above. H10:
    # ("grant_10_best_match_*.png",
    # "H10. Which pre-stroke position does each post-stroke position match BEST?",
    # ("Every panel above reduces a row of the similarity matrix to its DIAGONAL. That cannot "
    # "separate 'the code is gone' from 'the code moved to far_L': 0.2 against everything and "
    # "0.2 against itself with 0.7 against far_L are the same diagonal and different results. "
    # "This reduces the row to its ARGMAX instead, which uses all six entries and answers "
    # "'moved WHERE'. LEFT is the ceiling, leave-one-session-out -- it is NOT 6/6, and reading "
    # "the middle panel against 100% would overstate everything. RIGHT adds the mean RANK of "
    # "the true position, which degrades gracefully where the fraction is all-or-nothing: "
    # "slipping from first to second is not slipping to sixth. IMMUNE TO THE AMPLITUDE TERM -- "
    # "argmax and rank cannot move under a monotone change across a row, and the uniform row "
    # "shifts that dominate H8/H8d are exactly that. Ties go to the diagonal, so a flat row "
    # "never manufactures a substitution.")),
    # CUT 2026-09-13 (Priya) -- see the cut block above. H10b:
    # ("grant_10b_best_match_by_session_*.png",
    # "H10b. The true position's RANK, session by session",
    # ("The same rank statistic as the right panel of H10, resolved to one cell per session "
    # "rather than pooled -- so a fraction driven by two bad sessions is visible as two bad "
    # "sessions. Rank 1 means the position's post-stroke pattern still matches its own "
    # "pre-stroke pattern better than any other; rank 6 means five other positions match it "
    # "better. THIS IS THE RANK-BASED, THRESHOLD-FREE READOUT, and it is why an AUROC family "
    # "would add less here than it first appears: rank over the six class prototypes already "
    # "removes any monotone transform of the similarities. What it does NOT measure is "
    # "TRIAL-level discriminability -- it ranks classes for a session's mean pattern, not "
    # "trials within a class -- which is the gap the epoch 5r refit arm fills instead. "
    # "Rendered since 2026-08-27 and unplaced until 2026-09-09; earlier decks show H10's "
    # "pooled version only.")),
    ("grant_11_encoder_gain_shape_*.png",
     "H11. FROZEN ENCODER — did the position code MOVE, or just get SMALLER?",
     ("THE FIRST ENCODER FIGURE IN THE SET, and the only one that answers that question with "
     "two separately estimated numbers rather than two readings of one: correlations (H6b, "
     "H8b) are blind to amplitude by construction, distances (H8) are dominated by it. "
     "A one-hot position encoder trained on PRE-STROKE SESSIONS ONLY predicts each position's "
     "pre-stroke mean pattern; one gain fitted per session splits its failure. LEFT: transfer "
     "without rescaling and with, the shaded gap being what rescaling recovers. MIDDLE: the "
     "fitted gain, 1.0 = no amplitude change. RIGHT: what is left per position after the "
     "session gain, i.e. a genuine tuning change, localised. READ THE GAIN FIRST -- a wide gap "
     "alone is NOT an amplitude result, because an unrelated code also recovers a lot by "
     "collapsing the gain towards zero. Observed: PS92 shows no detectable change on any day, "
     "while PS93/94/95 keep a LOW after-gain score, so rescaling does not recover their loss. "
     "CAVEAT, and it is the big one: NO MOVEMENT REGRESSORS (no DLC yet), so a post-stroke "
     "change in how the animal moves would appear here as a change in tuning. A difference "
     "that holds across the lick, working and pre-cue variants is not a movement artefact; one "
     "that appears only in the lick window probably is.")),
)


# ---------------------------------------------------------------------------------------------
# SECTION D / D2 — THE ARMS AND BASES THE CROSS-SESSION SECTIONS ITERATE
# ---------------------------------------------------------------------------------------------
# BOTH alignments: post-cue (readout during/after the movement) and PRE-CUE (the maintained,
# motor-independent code). The pre-cue one is the readout the stroke arm leans on, so whether IT
# survives freezing across days is the more consequential question.
# LICK ADDED 2026-08-26 (Priya). This was cue+precue because the plan/execution dissociation is a
# cue-vs-precue contrast, and lick was never added -- while every OTHER consumer adopted it:
# grant_figures' WINDOWS is ENL/cue/lick, position_coding_directions emits lick, Section G has
# lick confusions, and the nightly has been running --align lick per day all along. The result was
# an arm computed nightly and never shown, and a frozen lick arm not computed at all.
ALIGNS = (("cue", "post-cue 2 s", "the readout during/after the movement"),
          ("precue", "PRE-CUE 2 s", ("pre-cue position information — the window ENDING "
                                    "at the cue, before any movement")),
          ("lick", "post-LICK 2 s", ("aligned to the first lick rather than the cue, so trials "
                                     "are registered on the movement itself — the arm that "
                                     "isolates execution from the plan")))
BASES = (("roi", "Allen-ROI", M_FROZEN, M_FROZEN_ENC,
          "66 atlas-anchored anatomical areas — column j is the same cortical region every day"),
         ("joint", "joint-LocaNMF", M_JOINT, M_JOINT,
          ("shared joint-basis components — footprints fitted once and FROZEN, new days projected "
          "onto them rather than refitted")))

# ONE BASIS, NOT TWO. Priya, 2026-09-13: drop the Allen-ROI arm of D2. The two bases were shown
# side by side to demonstrate the result is not an artefact of either one, and they agree -- so
# the ROI trio was three slides making a point the joint-LocaNMF trio already makes. The ROI
# figures are still written every night and `nolick_basis_agreement.png` (below) still carries
# the roi-vs-joint comparison, so the evidence for the agreement stays in the deck; what is gone
# is showing the same conclusion twice.
NOLICK_BASES = (("joint", "joint-LocaNMF, 2.0 s cut"),)

"""POOLED EPOCH FIGURES — MAPS

Beta maps and evoked maps, pooled and per animal, against each reference family.

Split out of `epoch_grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS
FROM THE CALL GRAPH: every function here is reached from this family's entry points and from no
other. Anything shared with a sibling is in `epoch_kit`, so this imports from there and never
from `epoch_grant_figures` -- that direction would be a cycle.

Entry points, dispatched by `epoch_grant_figures.main`:
  - `_fig_14_beta_maps`
  - `_fig_14pa_beta_maps_by_animal`
  - `_fig_14z_beta_vs_zero`
  - `_fig_15_evoked_maps`
  - `_fig_15pa_evoked_maps_by_animal`
  - `_fig_15r_reference_maps`
  - `_fig_15rpa_reference_by_animal`
"""
from __future__ import annotations

import numpy as np

from wfield_local import epoch_figures as ef
from wfield_local.epoch_kit import (  # noqa: F401
    _MEAN_NOTE,
    N_BOOT,
    _accuracy_at,
    _groups,
    _long_labels,
    _minor,
    _pre_counts,
    _scalar_figure,
    _seed_for,
    _session_counts,
    _short_labels,
    _totals,
)
from wfield_local.position_reference_maps import REST_WEIGHTED


def _long_of(q):
    """Position name -> the anatomical label the figures use ("far contra", not "far_R")."""
    from wfield_local.grant_figures import CONF_LABELS
    return dict(zip(CONF_LABELS, _long_labels()))[q]
def _fig_14pa_beta_maps_by_animal(out_dir, align, variant, wname):
    """14pa: the same maps PER ANIMAL, for the one position the deficit lives at.

    THE POOLED FIGURE AVERAGES FOUR ANIMALS AND THE PERMUTATION TEST RESTS ON FOUR, so neither can
    show whether a pattern REPLICATES -- and with n=4 replication across panels is stronger evidence
    than a p-value from a between-animal SE estimated on four numbers. Priya asked for these
    alongside the pooled view for exactly that reason.

    ONE POSITION PER FIGURE, far-contralateral by default: six positions x four animals x five
    epochs is 120 panels, which is a contact sheet rather than a figure. Far-contra is where the
    deficit is, and its neighbours are on the pooled panel.

    A ROW THAT LOOKS UNLIKE THE OTHER THREE IS THE POINT, not noise to be averaged away: PS94 and
    PS95 took 3 mW and show overt deficits while PS92 and PS93 were milder, so a severity-graded
    difference between rows is a finding and a random one is a warning.
    """
    if not ((variant == "working" and align in ("precue", "cue"))
            or (variant == "lick" and align == "lick")):
        return None
    from wfield_local import beta_maps as bm

    store, rel, ntr = bm.maps_by_epoch(align, variant)
    if not store:
        return None
    Q = "far_R"
    EPO = list(ef.PANELS)
    # ALL THREE POST-STROKE DELTAS, not just acute (Priya, 2026-09-12: "the pa map should have the
    # across-epoch deltas"). Acute-minus-pre alone shows the hit and hides the recovery, and
    # recovery is half of what the per-animal view is for -- a row that recovers and a row that
    # does not is exactly the between-animal difference this figure exists to expose.
    POST = ("acute", "subacute", "chronic")
    DCOLS = [f"{e} - pre" for e in POST]
    cells, titles = {}, {}
    for an, by_e in sorted(store.items()):
        per = {}
        for e in EPO:
            got = (by_e.get(e) or {}).get(Q)
            if not got:
                continue
            per[e] = np.mean(list(got.values()), axis=0)
            cells[(an, e)] = per[e]
            n_tr = sum((((ntr.get(an) or {}).get(e) or {}).get(Q) or {}).values())
            r = ((rel.get(an) or {}).get(e) or {}).get(Q)
            titles[(an, e)] = (f"{e}\n{len(got)} sess, n={n_tr}"
                               + (f"\nr={r:.2f}" if r is not None and np.isfinite(r) else ""))
        for e in POST:
            if "pre" in per and e in per:
                cells[(an, f"{e} - pre")] = per[e] - per["pre"]
                titles[(an, f"{e} - pre")] = f"{e.upper()} - PRE"
    if not cells:
        return None
    rows = [a for a in sorted(store) if any((a, e) in cells for e in EPO)]
    return ef.map_grid(
        cells, out_dir, name=f"epoch_14pa_beta_maps_MEANref_by_animal_{align}_{variant}",
        title=(f"Far-CONTRALATERAL decoder beta map, MEAN-referenced (positions COUPLED), "
               f"PER ANIMAL -- does the pattern replicate? {wname}"),
        row_labels=rows, col_labels=EPO + DCOLS, panel_titles=titles, delta_cols=tuple(DCOLS),
        edges=bm.atlas_edges(),
        blank=bm.excluded_mask(),
        cbar_label=("cov(pixel, decoder output)\nred = MORE active on this position's\n"
                    "trials than on the average trial"),
        delta_label="change vs pre-stroke\n(orange-blue; expands if larger)",
        subtitle=("The pooled figure averages these four rows and the permutation test rests on "
                  "them, so neither can show REPLICATION -- which at n=4 is the stronger evidence. "
                  "Method identical to the pooled figure. Colour scale is per ANIMAL, so each row "
                  "is comparable across its own epochs and rows are not comparable to each other. "
                  "PS94 and PS95 took 3 mW and show overt deficits; PS92 and PS93 were milder, so a "
                  "severity-graded difference between rows is a finding and a random one is a "
                  "warning."))
#: How each reference is described on its own figure -- title, what a red pixel means, and the
#: sentence that says whether the six rows are independent. Kept as data rather than as branches
#: inside the renderer because the INDEPENDENCE claim is the one a reader must not have to infer.
_REF_TEXT = {
    "mean": dict(
        short="one vs REST",
        title="Position maps referenced to the MEAN OVER ALL TRIALS -- figure 14's reference, "
              "with no decoder in the path",
        cbar="activity minus the mean\nover ALL trials in the window",
        note=("THE SIX ROWS ARE NOT INDEPENDENT, and that is what this figure is for. The "
              "subtrahend is the mean over all trials, so a position that loses drive lowers the "
              "reference and hands every other position an increase it did not earn -- Priya, "
              "2026-09-12: \"in acute there may be less ss-ul/ll activity in far-center trials, "
              "which makes the near ipsi acute trial map look as though there is a relative "
              "*increase* in ss-ul/ll activity compared to pre-stroke.\" This is figure 14's "
              "reference rendered as a plain trial average, so the coupling can be seen without "
              "the decoder, the folds and the Haufe transform in the way. READ IT AGAINST THE "
              "QUIET-REFERENCED FIGURE: a rise that appears here and NOT there is the artefact."),
    ),
    # ADDED 2026-09-13. `precue` was put into `position_reference_maps.REFERENCES` when the pre-cue
    # reference moved onto the deck's own window, and this table was not extended with it -- so
    # `_fig_15r_reference_maps` raised `KeyError 'precue'` on every arm, AFTER writing the mean- and
    # rest-referenced figures. The failure was per-arm and non-fatal, so the render reported exit 0
    # and simply produced two references where three were asked for. Found by counting the files
    # against the retired set (21 where 39 were expected), not by the exit status.
    # ADDED 2026-09-17, together with the `reference_maps` branch and the REFERENCES entry -- the
    # three that must land in one commit, per the note on that tuple.
    "raw": dict(
        short="NO subtrahend",
        title="Position maps with NO REFERENCE SUBTRACTED -- the drift-removed signal as it stands",
        cbar="post-cue activity\n(drift-removed, no subtrahend)",
        note=("THE FAMILY WITH NO SUBTRAHEND THAT CAN DRIFT, which is the whole reason to read it "
              "beside the other two. Every other reference here subtracts something that moves "
              "between epochs: `mean` couples the six positions (a position losing drive hands the "
              "other five an unearned rise), and the rest baseline MOVES across epochs -- measured "
              "at 7-43% of the evoked signal, significant in 6 of 11 animal-epoch cells "
              "(`rest_baseline_epoch_drift`). An across-epoch amplitude read off those "
              "inherits the movement -- though that is an UPPER bound, and the alignment arm added "
              "2026-09-18 finds no evidence it is attained (1 of 11 cells at p < 0.05 against ~0.6 "
              "expected), so the realistic cost sits at the low end. This one cannot, because there is nothing to move.\n"
              "WHAT IT TRADES: cross-day MULTIPLICATIVE scaling -- expression, bleaching, window "
              "clarity -- which no subtraction removes either, so it is the confound left standing "
              "rather than one this family adds. `crossday_intensity` owns it.\n"
              "AND IT IS NOT 'UNREFERENCED': the signal is high-passed, so it is implicitly "
              "referenced to its own temporal surround. Under `meegkit_hpfit` the drift fit "
              "EXCLUDES strobe-0.25s -> cue+4s, so the fit never sees the measured window -- which "
              "is why this is meaningful now and was not under the retired `zerophase` product.\n"
              "READ THE THREE TOGETHER. Their failure modes are DIFFERENT, so a regional effect "
              "present in all three is not a property of any one subtrahend; one that appears in "
              "only one IS."),
    ),
    "precue": dict(
        short="one vs its own PRE-CUE window",
        title="Position maps referenced to each trial's OWN PRE-CUE window -- the CUE-EVOKED "
              "INCREMENT, not the position map",
        cbar="post-cue minus that trial's\nown pre-cue mean",
        note=("THE SIX ROWS ARE INDEPENDENT -- the subtrahend is per trial, so one position cannot "
              "leak into another. COMPUTED ON THE DECK'S OWN WINDOW (cue-aligned 0 to +2.0 s, "
              "figure 14's trial selection, `trial_features` at baseline=\"precue\"), NOT from the "
              "preprocessing npz's 1 s-pre/1 s-post `delta` field, which figure 15 still uses. That "
              "distinction is measurable: the old window flips the sign of the near positions' "
              "between-animal agreement at acute (near-ipsi +0.194 vs -0.086) while leaving "
              "far-contralateral unchanged to three decimals."
              "\n\nIT VIOLATES F12 AND THAT IS THE POINT OF HAVING IT. The pre-cue window carries "
              "genuine anticipatory position signal (LOSO 0.510), so subtracting it removes real "
              "code and measures the cue-evoked INCREMENT. Read it as a third reference whose "
              "disagreements with REST localise what the pre-cue window contains, not as the "
              "position map."),
    ),
    "rest": dict(
        short="one vs REST",
        title="Position maps referenced to the TIME-LOCAL REST BASELINE -- position-independent, "
              "so the six positions are INDEPENDENT",
        cbar="activity minus the REST baseline\nat that trial's own time",
        note=("THE SIX ROWS ARE INDEPENDENT -- the subtrahend is IDENTICAL for all six, so "
              "subtracting it cannot couple them, which is what the mean reference does by "
              "construction. That is the property this reference is chosen for, and it does NOT "
              "require the subtrahend to be position-free: see the correction below, it is not. "
              "This answers \"is this position's cortex driven at all\", "
              "where the mean-referenced figure answers \"is it driven more than the others\". It "
              "is also the reference that does NOT sit inside the trial: figure 15's pre-cue "
              "baseline controls drift tightest but is blind to a sustained shift already present "
              "before the cue, and this one is not."
              "\n\nTIME-LOCAL SINCE 2026-09-13, AND THE PREVIOUS FORM WAS WRONG IN A WAY WORTH "
              "STATING. This used to subtract ONE SESSION MEAN, which is flat. Positions run in "
              "~6-TRIAL BLOCKS, so each one's trials cluster at particular times and a flat "
              "subtrahend leaves every position carrying its blocks' "
              "share of the session's drift. The baseline is now binned over the session, taken "
              "as the median of rest frames per bin and interpolated to every frame, and it is "
              "subtracted from the SVT BEFORE the trial features are built -- so each trial is "
              "referenced to the baseline AT ITS OWN MOMENT. `locanmf_position_encoder` has "
              "always done it this way; the map reference and the encoder now agree."
              "\n\nCORRECTION, SAME DAY: REST CARRIES POSITION INFORMATION. A claim that stood "
              "here -- that rest's between-position differences were \"drift aliased onto the "
              "block structure, not position coding\", from a ratio of 1.02 -- is WITHDRAWN. That "
              "inference does not hold: its two contrasts were not matched on time separation, "
              "and a ratio of magnitudes is not a test. A circular-shift permutation, which keeps "
              "the block-time structure INSIDE the null, gives observed/null **1.634** over 44 "
              "pre-stroke sessions, above null in **43/44**, and 4/4 animals (PS92 1.467, PS93 "
              "1.996, PS94 1.724, PS95 1.458; mean over animals 1.662) -- on the `restdock05` "
              "definition, which is docked BY CONSTRUCTION, so there is no longer a loose/strict "
              "distinction to mislabel. Re-measured 2026-09-15; the earlier 1.443 (41/44) and "
              "1.449 (39/42) were taken on superseded rest definitions, and two captions "
              "disagreed about which of them came from the docked window. The effect is STRONGER "
              "on the current definition, not weaker, and no session drops out. "
              "The block-boundary test says the signal is PERSISTENCE: rest resembles "
              "the position just licked at more than the one coming next (+0.0754, 4/4 animals)."
              "\n\nWHAT THAT MEANS FOR THIS FIGURE. The subtrahend is not position-NEUTRAL, so it "
              "leaves a contaminant in these maps -- it still cannot COUPLE the positions, since "
              "it is one map subtracted from all six, but it is not the blank reference it was "
              "described as. `_RESTWref_` is the fix: the same construction with each POSITION "
              "weighted equally instead of each rest FRAME, so the baseline's composition stops "
              "tracking which positions the animal still works. Read the two together."
              "\n\nWHICH BASELINE THIS IS, read it off the filename. `_REWARD8ref_` is the "
              "RETIRED definition -- slow treadmill, no licking, and 8 s excluded after every "
              "reward, a buffer carried over from a task whose post-tone window was 8 s while "
              "ours has a 3.5 s response window. Its quiet fraction tracks the deficit (4.4% of "
              "frames pre-stroke, 17.1% acutely, a median 0.7% chronically), and at chronic its "
              "between-animal agreement is position-INDEPENDENT shared offset: observed r = +0.494 "
              "against a null of +0.497. DO NOT READ THE CHRONIC COLUMN OF A `_REWARD8ref_` "
              "FIGURE. `_RESTref_` is the replacement: between trials, not running, not licking. "
              "See docs/REST_BASELINE_MIGRATION.md."),
    ),
    # ADDED 2026-09-13, IN THE SAME COMMIT AS THE BUILDER. `restw` was named in
    # `position_reference_maps` weeks before it existed, and putting it in `REFERENCES` ahead of
    # its builder took down EVERY 15r render for twenty minutes -- `maps_by_epoch` iterates
    # REFERENCES and `reference_maps` raises on an unknown name. This table is the second half of
    # that same trap: `_fig_15r_reference_maps` does `_REF_TEXT[reference]`, so a reference
    # registered without an entry here raises per-arm, AFTER writing the earlier references, and
    # the render still exits 0 having produced fewer figures than it was asked for. That is
    # precisely how `precue` was lost. Registry, builder and caption text go in together.
    REST_WEIGHTED: dict(
        short="one vs POSITION-WEIGHTED rest",
        title="Position maps referenced to the POSITION-WEIGHTED rest baseline -- each position "
              "weighted equally, not each rest frame",
        cbar="activity minus the position-weighted\nFLAT rest baseline",
        note=("THE SIX ROWS ARE INDEPENDENT, as with `_RESTref_`: one subtrahend for all six, so "
              "it cannot couple them. WHAT DIFFERS IS THE BASELINE'S COMPOSITION, and that is the "
              "whole figure. `_RESTref_` averages over rest FRAMES, so a position contributing "
              "more rest frames pulls the baseline toward its own resting state. This one builds "
              "each position's own FLAT rest level -- the median over that position's rest frames, "
              "one level, NOT binned over session time -- and averages the six EQUALLY."
              "\n\nFLAT, NOT TIME-LOCAL, AND THIS CAPTION SAID THE OPPOSITE UNTIL 2026-09-15. "
              "`restw` was SPECIFIED as per-position time-local and IMPLEMENTED that way; the flat "
              "form replaced it on 2026-09-14 after `flat_vs_timelocal` measured the two and found "
              "the time-local version moved no conclusion (same acute/pre ordering, same monotone "
              "near->far gradient, between-animal agreement at far-contra 0.812 vs 0.838 against a "
              "null of 0.07) while costing the defect that a thin per-position time bin is a noisy "
              "level and equal weighting amplifies exactly the thinnest estimates. The estimator "
              "moved; these two words did not, so the COLOURBAR of every `_RESTWref_` figure "
              "rendered between those dates names a construction the code had already stopped "
              "using. Read `restw_from_frames`, which is the one implementation, if in doubt."
              "\n\nWHY IT MATTERS, AND WHY IT MATTERS MORE POST-STROKE. Rest is not "
              "position-neutral -- observed/null **1.634** over 44 pre-stroke sessions, **43/44**, "
              "4/4 animals, on `restdock05` (re-measured 2026-09-15) -- so a frame-weighted rest "
              "average CARRIES POSITION. "
              "Pre-stroke the six positions contribute roughly equally and this hardly matters. "
              "After the lesion the animal stops attempting the far positions, their blocks "
              "shorten or vanish, and the frame-weighted mean drifts toward the NEAR positions' "
              "rest: the baseline then changes WITH the deficit. That is the same failure that "
              "retired the 8 s-post-reward definition, arriving by a different route."
              "\n\nWHAT THIS IS NOT: a per-position baseline. Referencing each position to its "
              "OWN rest would subtract the between-trial position signal itself -- the "
              "persistence trace measured at +0.0754 across 4/4 animals -- and report a null. "
              "This keeps it. The difference between the two is the measurement of it."
              "\n\nREAD IT AGAINST `_RESTref_`, BUT THEY DIFFER ON TWO AXES, NOT ONE. This caption "
              "said \"the two differ ONLY in the weighting\" until 2026-09-15, which was true when "
              "written and stopped being true on 2026-09-14 when `restw` went flat. The axes are "
              "COMPOSITION (`_RESTref_` weights rest FRAMES, this weights POSITIONS equally) and "
              "TEMPORAL (`_RESTref_` is time-local -- 12 bins, median per bin, interpolated, "
              "subtracted before the trial features; this is FLAT). So a difference between them "
              "is not purely the composition effect. What licenses still reading it that way is "
              "that the temporal axis was MEASURED and moves nothing: `flat_vs_timelocal` gives "
              "the same acute/pre ordering, the same monotone near->far gradient, and "
              "between-animal agreement at far-contra of 0.812 vs 0.838 against a null of 0.07. "
              "The composition axis is the one carrying the difference -- but say which claim is "
              "measured rather than asserting the arms are matched when they are not."
              "\n\nA cell that moves post-stroke but not pre-stroke is the signature this "
              "reference exists to remove. A session where fewer than SIX positions have a usable "
              "rest baseline (>= 200 rest frames each) has NO `_RESTWref_` column at all, rather "
              "than a differently-defined baseline under the same name. The gate was FOUR until "
              "2026-09-14 (Priya: \"the rest should include all positions\"); on the `restdock05` "
              "definition every one of the 94 curated sessions clears six, so no session currently "
              "loses the column."),
    ),
}
def _fig_15rpa_reference_by_animal(out_dir, align, variant, wname):
    """15rpa: the quiet- and mean-referenced position maps PER ANIMAL, at far-contralateral.

    THE THIRD PER-ANIMAL PANEL, and it completes a set that now covers every map family: 14pa for
    the decoder maps, 15pa for the within-trial evoked maps, and this for the two reference maps.
    Each pooled figure averages four rows and each significance test estimates its spread from
    four animals, so none of them can show whether a pattern REPLICATES -- which at n=4 is the
    stronger evidence.

    BOTH REFERENCES, ONE PER FIGURE, so the pair can be read side by side at the animal level the
    way the pooled pair is read at the group level. That comparison is the whole point of the 15r
    family: same trials, same window, same floor, only the subtrahend differs, so a row that looks
    different between the two figures differs BECAUSE of the reference.

    WHAT TO LOOK FOR. Under the QUIET reference every animal's map should be broadly positive --
    task activity above rest -- and the epoch differences should be modest and consistent. Under
    the MEAN reference the common component is subtracted away, so what remains is
    position-specific and much noisier; a row that is dramatic there and flat under quiet is the
    one-vs-rest coupling showing itself in a single animal.
    """
    if not ((variant == "working" and align in ("precue", "cue"))
            or (variant == "lick" and align == "lick")):
        return None
    from wfield_local import beta_maps as bm
    from wfield_local import position_reference_maps as prm

    store, rel, ntr = prm.maps_by_epoch(align, variant)
    if not store:
        return None
    Q, EPO = "far_R", list(ef.PANELS)
    POST = ("acute", "subacute", "chronic")
    DCOLS = [f"{e} - pre" for e in POST]
    edges, out = bm.atlas_edges(), []

    for reference in prm.REFERENCES:
        # THE PRE-CUE REFERENCE IS DEGENERATE ON THE PRE-CUE ARM, and it is not merely redundant:
        # this arm's FEATURE window is [cue - 2 s, cue] while the pre-cue baseline is
        # [cue - 1 s, cue] -- the baseline is the SECOND HALF OF THE WINDOW ITSELF. Subtracting it
        # drives half the bins to ~0 by construction and leaves an early-versus-late contrast
        # inside one window, which is not a position map and must not be rendered as one. The other
        # two arms are genuine: `cue` gives [cue, cue+2] minus [cue-1, cue] (the cue-evoked
        # increment), `lick` gives a post-lick window minus a true pre-cue baseline.
        # Priya, 2026-09-13: "we're normalizing precue to precue??"
        if reference == "precue" and align == "precue":
            print("  .. 15r: skipping PRECUEref on the precue arm (baseline is inside the window)",
                  flush=True)
            continue
        txt = _REF_TEXT[reference]
        cells, titles = {}, {}
        for an, by_e in sorted(store.items()):
            per = {}
            for e in EPO:
                got = {k: d[reference] for k, d in ((by_e.get(e) or {}).get(Q) or {}).items()
                       if reference in d}
                if not got:
                    continue
                per[e] = np.mean(list(got.values()), axis=0)
                cells[(an, e)] = per[e]
                n_tr = sum((((ntr.get(an) or {}).get(e) or {}).get(Q) or {}).values())
                r = ((rel.get(an) or {}).get(e) or {}).get(Q, {}).get(reference)
                titles[(an, e)] = (f"{e}\n{len(got)} sess, n={n_tr}"
                                   + (f"\nr={r:.2f}" if r is not None and np.isfinite(r) else ""))
            for e in POST:
                if "pre" in per and e in per:
                    cells[(an, f"{e} - pre")] = per[e] - per["pre"]
                    titles[(an, f"{e} - pre")] = f"{e.upper()} - PRE"
        if not cells:
            continue
        rows = [a for a in sorted(store) if any((a, e) in cells for e in EPO)]
        out.append(ef.map_grid(
            cells, out_dir,
            name=f"epoch_15rpa_position_{prm.reference_tag(reference)}ref_by_animal_{align}_{variant}",
            title=(f"Far-CONTRALATERAL, {txt['short']}, PER ANIMAL -- does it replicate? {wname}"),
            row_labels=rows, col_labels=EPO + DCOLS, panel_titles=titles,
            delta_cols=tuple(DCOLS), edges=edges, blank=bm.excluded_mask(),
            cbar_label=txt["cbar"], delta_label="change vs pre-stroke\n(same scale unless larger)",
            subtitle=(
                f"ONE OF A PAIR ({txt['short']}), at the animal level. The pooled 15r figures "
                f"average these rows and every significance test here estimates its spread from "
                f"four animals, so neither can show REPLICATION -- which at n=4 is the stronger "
                f"evidence. Same trials, window, class definition and 20-trial floor as figure 14; "
                f"only the subtrahend differs between this figure and its partner, so a row that "
                f"differs between them differs BECAUSE of the reference. NO significance is drawn "
                f"here: a per-animal panel has no between-animal spread to test, and a "
                f"within-animal test would be answering a different question from the pooled "
                f"figures. Colour scale is per ANIMAL, so each row is comparable across its own "
                f"epochs and rows are not comparable to each other. PS94 and PS95 took 3 mW and "
                f"show overt deficits; PS92 and PS93 were milder, so a severity-graded difference "
                f"between rows is a finding and a random one is a warning. {txt['note']}")))
    return [p for p in out if p]
#: How each correction reads in a figure's methods line. Keyed by `beta_maps.CORRECTION`.
_CORRECTION_TEXT = {
    "maxstat": ("family-wise corrected by the BOOTSTRAP MAX-STATISTIC over {n} in-mask bins of an "
                "8x downsampled grid -- each draw is centred on the observed and studentised by "
                "the bootstrap SE, the maximum |z| across bins is taken, and the threshold is the "
                "95th percentile of those maxima, so THE DATA'S OWN SPATIAL COVARIANCE PERFORMS "
                "THE CORRECTION and no independence or smoothness is assumed"),
    "resel": ("Bonferroni over the {n} RESOLUTION ELEMENTS the map actually carries (in-mask area "
              "/ FWHM^2), not over its bins"),
    "bonferroni": ("Bonferroni over {n} in-mask bins of an 8x downsampled grid -- CONSERVATIVE, "
                   "because neighbouring bins are not independent at this smoothness"),
    "none": "UNCORRECTED across bins",
}
def _n_stat_bins():
    """In-mask bins of the 8x downsampled statistics grid, computed rather than remembered.

    The count moved when the olfactory bulbs and the painted glue left the mask, and three
    subtitles went on quoting the old one. Cheap -- it is a mask reduction, not an analysis.
    """
    from wfield_local import beta_maps as bm

    m = bm.stat_mask()
    if m is None:
        return 0
    # `downsample` returns (small, small_mask) and the SECOND is the bin count the tests use --
    # `musall_significance` reads exactly this as `n_tested`. Counting finite values of the tuple
    # instead gave 10,720, precisely twice the 67x80 grid, because numpy stacked both arrays.
    return int(bm.downsample(m.astype(float), mask=m)[1].sum())
def _stats_sentence(stat_rows):
    """The significance methods sentence, derived from what the panels REPORTED.

    WRITTEN RATHER THAN TYPED, because the typed version went stale and nothing caught it. Three
    figure families carried the literal string "Bonferroni over 3,237 in-mask bins" in their
    subtitle for hours after the correction became the bootstrap max-statistic -- the figures were
    computing one thing and announcing another, and a reader had no way to tell. A redraw from a
    saved bundle reproduced the wrong sentence just as faithfully.

    Reads `correction` and `n_bins` off the rows `significance_contour` stashed, so the sentence
    cannot disagree with the test that produced the contours. Falls back to the module constant
    when a family drew no contour at all.
    """
    from wfield_local import beta_maps as bm

    rows = [r for r in (stat_rows or []) if r.get("correction")]
    corr = rows[0]["correction"] if rows else bm.CORRECTION
    n = max((int(r.get("n_bins") or 0) for r in rows), default=0)
    # THE BIN COUNT IS THE MAXIMUM OVER PANELS, not a constant: it is the in-mask bin count of the
    # eroded statistics mask, and a panel whose animals intersect differently can test slightly
    # fewer. Quoting the largest is the honest summary of "up to this many tests".
    txt = _CORRECTION_TEXT.get(corr, f"corrected by {corr}")
    txt = txt.format(n=f"{n:,}") if "{n}" in txt else txt
    tail = ""
    if any(int(r.get("suppressed") or 0) for r in rows):
        tail = (" PANELS WHOSE FLAGGED PIXELS WERE MORE THAN "
                f"{bm.EDGE_ENRICHMENT_MAX:.0f}x CONCENTRATED IN THE MASK RIM ARE SUPPRESSED and "
                "draw no contour -- that pattern is an imaging-window artefact, and because it is "
                "the same artefact in every animal a between-animal test cannot reject it.")
    return ("BLACK contours are bins significant under the NESTED animals->sessions bootstrap "
            f"(2,000 draws, resampled with replacement at both levels), {txt}. The test runs on "
            f"the brain mask ERODED {bm.STAT_ERODE_PX} px, with the olfactory bulbs and the "
            f"hand-painted fibre-glue occlusion removed.{tail}")
def _fig_15r_reference_maps(out_dir, align, variant, wname):
    """15r: the SAME position maps under two different references -- the reference IS the claim.

    Priya, 2026-09-12: "does that average across all trials make sense? should we compare it to
    'quiet' instead?", then "maybe let's trial comparison the one vs rest vs one vs quiet".

    WHAT IS HELD FIXED AND WHAT IS VARIED. Trials, window, class definition and the 20-trial floor
    are figure 14's, exactly; the ONLY thing that changes between the two figures this returns is
    what gets subtracted. So a difference between them is the reference and cannot be anything
    else, which is the only way the comparison answers the question.

    NO DECODER ANYWHERE. These are trial averages. Figure 14 needs a fit, folds and class weights
    because it asks what DISTINGUISHES the positions; a reference question is about what happens on
    a position's trials, and a decoder in the path only adds a second thing that could explain a
    difference.

    THE THREE REFERENCES AND WHERE EACH LIVES:

        mean   here     minus the mean over ALL trials -- figure 14's, positions COUPLED
        quiet  here     minus the session's quiet baseline -- positions INDEPENDENT
        self   fig 15   minus that position's own pre-cue window -- positions INDEPENDENT

    Two independent references that disagree would localise the problem to what sits between them,
    which is the pre-cue window's own content: anticipation and locomotor state.
    """
    # SAME THREE ARMS AS FIGURE 14, which is what "the only difference is the reference" requires.
    # `stopped` is excluded for figure 14's reason: too few trials per position to fill six rows.
    if not ((variant == "working" and align in ("precue", "cue"))
            or (variant == "lick" and align == "lick")):
        return None
    from wfield_local import beta_maps as bm
    from wfield_local import position_reference_maps as prm
    from wfield_local.grant_figures import CONF_LABELS

    store, rel, ntr = prm.maps_by_epoch(align, variant)
    if not store:
        return None
    EPO = list(ef.PANELS)
    POST = ("acute", "subacute", "chronic")
    DCOLS = [f"{e} - pre" for e in POST]
    edges, out = bm.atlas_edges(), []

    for reference in prm.REFERENCES:
        # THE PRE-CUE REFERENCE IS DEGENERATE ON THE PRE-CUE ARM, and it is not merely redundant:
        # this arm's FEATURE window is [cue - 2 s, cue] while the pre-cue baseline is
        # [cue - 1 s, cue] -- the baseline is the SECOND HALF OF THE WINDOW ITSELF. Subtracting it
        # drives half the bins to ~0 by construction and leaves an early-versus-late contrast
        # inside one window, which is not a position map and must not be rendered as one. The other
        # two arms are genuine: `cue` gives [cue, cue+2] minus [cue-1, cue] (the cue-evoked
        # increment), `lick` gives a post-lick window minus a true pre-cue baseline.
        # Priya, 2026-09-13: "we're normalizing precue to precue??"
        if reference == "precue" and align == "precue":
            print("  .. 15r: skipping PRECUEref on the precue arm (baseline is inside the window)",
                  flush=True)
            continue
        txt = _REF_TEXT[reference]
        cells, titles, amp, contours, rows = {}, {}, {}, {}, []
        _stats = []
        for q in CONF_LABELS:
            row = _long_of(q)
            pre_by = prm.by_animal(store, reference, q, "pre")
            if not pre_by:
                continue
            rows.append(row)
            # WITHIN ANIMAL, INTERSECTION ONLY, EACH ANIMAL ON ITS OWN PRE-STROKE SCALE.
            pre_p, _a, _b, _c, pre_ans = bm.within_animal_pooled(pre_by)
            cells[(row, "pre")] = pre_p
            titles[(row, "pre")] = (f"pre\n{len(pre_ans)} an, "
                                    f"{sum(len(v) for v in pre_by.values())} sess")
            amp[q] = {"pre": 1.0}
            for e in POST:
                post_by = prm.by_animal(store, reference, q, e)
                if not post_by:
                    continue
                _p, post_p, delta, ratio, ans = bm.within_animal_pooled(pre_by, post_by)
                if post_p is None:
                    continue
                n_s = sum(len(v) for a, v in post_by.items() if a in ans)
                n_tr = sum(sum((((ntr.get(an) or {}).get(e) or {}).get(q) or {}).values())
                           for an in ans)
                rs = [((rel.get(an) or {}).get(e) or {}).get(q, {}).get(reference) for an in ans]
                rs = [r for r in rs if r is not None and np.isfinite(r)]
                cells[(row, e)] = post_p
                titles[(row, e)] = (f"{e}\n{len(ans)} an, {n_s} sess, n={n_tr}"
                                    + (f"\nr={np.median(rs):.2f}" if rs else ""))
                cells[(row, f"{e} - pre")] = delta
                titles[(row, f"{e} - pre")] = f"{e.upper()} - PRE\n{len(ans)} an, within-animal"
                amp[q][e] = ratio
                if not (pre_by and post_by):
                    continue
                try:
                    cm, lab = bm.significance_contour(pre_by, post_by)
                    print(f"  .. 15r {reference} {q} {e}: {lab}", flush=True)
                    _stats.append(dict(row=row, col=f"{e} - pre",
                                       n_animals=len(set(pre_by) & set(post_by)),
                                       animals=";".join(sorted(set(pre_by) & set(post_by))),
                                       amplitude_vs_pre=round(amp.get(q, {}).get(e,
                                                                                float("nan")), 4),
                                       **getattr(bm.significance_contour, "last", {})))
                    if cm is not None and np.any(cm):
                        contours[(row, f"{e} - pre")] = cm
                except Exception as ex:                                # noqa: BLE001
                    print(f"  !! 15r sig {reference} {q} {e}: "
                          f"{type(ex).__name__} {str(ex)[:70]}", flush=True)
        if not cells:
            continue
        a_txt = ", ".join(f"{_long_of(q)} {amp[q].get('acute', float('nan')):.2f}"
                          for q in CONF_LABELS if q in amp)
        out.append(ef.map_grid(
            cells, out_dir,
            name=f"epoch_15r_position_{prm.reference_tag(reference)}ref_{align}_{variant}",
            title=f"{txt['title']} -- {wname}",
            row_labels=rows, col_labels=EPO + DCOLS, panel_titles=titles,
            delta_cols=tuple(DCOLS), edges=edges, contours=contours,
            blank=bm.excluded_mask(),
            cbar_label=txt["cbar"], delta_label="change vs pre-stroke\n(SAME scale as the maps)",
            row_scaled=False,
        stat_rows=_stats,
        subtitle=(
                f"ONE OF A PAIR ({txt['short']}). Both figures use the SAME trials, the same "
                f"window, the same class definition and the same 20-trial floor as figure 14; the "
                f"only thing that differs is what is subtracted, so a difference between them is "
                f"the reference and nothing else. These are TRIAL AVERAGES -- no decoder, no "
                f"folds, no Haufe transform. {txt['note']} "
                "Colour scale is PER ROW, so a position is comparable across its own epochs and "
                "rows are not comparable to each other; the difference columns share their row's "
                # NO COLOUR WORD HERE -- `_stats_sentence` supplies it. This line used to add
                # its own, so the caption read "GREEN GREEN contours" on every panel of this
                # family, and naming the colour in two places is how they come to disagree.
                "scale. THE THIN DARK OUTLINES ARE ALLEN CCF BOUNDARIES, not statistics. "
                f"{_stats_sentence(_stats)} This is the same statistical object every bar "
                "family in this deck uses. "
                "r = split-half reliability of that epoch's mean map. "
                f"Acute amplitude relative to each position's own pre-stroke value: {a_txt}.")))
    return [p for p in out if p]
def _fig_15pa_evoked_maps_by_animal(out_dir, align, variant, wname):
    """15pa: figure 15's per-position evoked maps, PER ANIMAL, at the position that carries it.

    WHY THIS AND NOT ONLY 14pa. Figure 14's per-animal panels replicate beautifully and they
    replicate the wrong thing: once the `working` class was fixed, its acute far-contralateral map
    is near-uniformly negative in all four animals, which is the ENGAGEMENT collapse rather than a
    spatial code (see DECISIONS, 2026-09-12 late). The measure that stands is figure 15's --
    per-position `post-cue minus pre-cue`, a within-trial reference, positions INDEPENDENT -- so
    that is the one whose replication matters.

    FOUR ANIMALS IS THE CEILING ON THE PERMUTATION TEST, which is the other reason this exists. The
    pooled figure averages these rows and the test estimates its between-animal SE from four
    numbers; neither can show whether a pattern REPLICATES, and at n=4 replication across panels is
    the stronger evidence. A row that looks unlike the other three is a finding, not noise to be
    averaged away: PS94 and PS95 took 3 mW and show overt deficits while PS92 and PS93 were milder.

    ONE POSITION PER FIGURE. Six positions x four animals x five epochs is a contact sheet; this
    draws far-contralateral, where the deficit lives, and the pooled figure carries its neighbours.
    """
    if align != "cue" or variant != "working":
        return None
    from wfield_local import beta_maps as bm
    from wfield_local import position_evoked_maps as pem

    store, _counts = pem.maps_by_epoch()
    if not store:
        return None
    Q, EPO = "far_R", list(ef.PANELS)
    POST = ("acute", "subacute", "chronic")
    DCOLS = [f"{e} - pre" for e in POST]
    cells, titles = {}, {}
    for an, by_e in sorted(store.items()):
        per = {}
        for e in EPO:
            got = (by_e.get(e) or {}).get(Q)
            if not got:
                continue
            per[e] = np.mean(list(got.values()), axis=0)
            cells[(an, e)] = per[e]
            titles[(an, e)] = f"{e}\n{len(got)} sess"
        for e in POST:
            if "pre" in per and e in per:
                cells[(an, f"{e} - pre")] = per[e] - per["pre"]
                titles[(an, f"{e} - pre")] = f"{e.upper()} - PRE"
    if not cells:
        return None
    rows = [a for a in sorted(store) if any((a, e) in cells for e in EPO)]
    return ef.map_grid(
        cells, out_dir, name="epoch_15pa_evoked_CUEINCREMENT_PRECUEref_by_animal_cue",
        title="Far-CONTRALATERAL CUE-EVOKED INCREMENT (post-cue minus PRE-CUE) PER ANIMAL -- "
              "does it replicate? Subtracts the anticipatory code; see F12",
        row_labels=rows, col_labels=EPO + DCOLS, panel_titles=titles, delta_cols=tuple(DCOLS),
        edges=bm.atlas_edges(),
        blank=bm.excluded_mask(),
        cbar_label="post-cue minus pre-cue\n(that position's OWN trials)",
        delta_label="change vs pre-stroke\n(SAME scale as the maps)",
        subtitle=(
            "THE PER-ANIMAL VIEW OF THE MEASURE THAT STANDS. Figure 14pa replicates too, but once "
            "the `working` class was fixed its acute far-contra map is near-uniformly negative in "
            "all four animals -- the ENGAGEMENT collapse, not a spatial code. Here each map is "
            "that position's OWN post-cue minus pre-cue, so the six positions are independent and "
            "a change cannot be inherited from another position's loss. Four animals is the "
            "ceiling on the permutation test and the pooled figure averages these rows, so "
            "neither can show REPLICATION -- which at n=4 is the stronger evidence. Colour scale "
            "is per ANIMAL: each row is comparable across its own epochs, rows are not comparable "
            "to each other. PS94 and PS95 took 3 mW and show overt deficits; PS92 and PS93 were "
            "milder, so a severity-graded difference between rows is a finding and a random one "
            "is a warning. Maps from `framemap_event_maps`; nothing recomputed."))
def _fig_15_evoked_maps(out_dir, align, variant, wname):
    """15: per-position EVOKED cortical maps -- the position-INDEPENDENT answer to "where".

    THE CONTROL THAT GATES FIGURE 14, not a complement to it. Priya, 2026-09-12: "in acute there may
    be less ss-ul/ll activity in far-center trials, which makes the near ipsi acute trial map look as
    though there is a relative *increase* in ss-ul/ll activity compared to pre-stroke." Figure 14's
    Haufe pattern is a covariance against the mean over ALL SIX positions, so one position losing
    drive lowers the reference and hands every other position an increase it did not earn. Four of
    its six rows cannot be read as written.

    HERE THE REFERENCE IS WITHIN TRIAL AND PER POSITION: each map is that position's own
    `post-cue mean - pre-cue mean`. Far-contralateral collapsing cannot leak into near-ipsilateral's
    map, because near-ipsilateral's map never looks at far-contralateral's trials.

    NOTHING IS RECOMPUTED. `framemap_event_maps` already writes these per session -- 123
    `*_spout_positions_1s_pre_post_delta_maps.npz` on the share, six positions x {pre, post, delta}
    as 540 x 640 Allen-aligned maps. This aggregates them by epoch.

    WHAT IT COSTS, so the two figures are not confused. The decoder pattern isolates what
    DISTINGUISHES positions but couples them; this keeps them independent but shows the WHOLE
    task-evoked response at that position -- cue, licking, movement, arousal -- not only the part
    carrying target identity. Disagreement is informative: a change here and not in figure 14 is a
    change in DRIVE that carries no position information; the reverse is a change in TUNING with no
    change in drive.

    ONE ALIGNMENT. The source maps are cue-referenced by construction (post-cue minus pre-cue), so
    there is nothing for a pre-cue or post-lick arm to be.
    """
    if align != "cue" or variant != "working":
        return None
    from wfield_local import beta_maps as bm
    from wfield_local import position_evoked_maps as pem
    from wfield_local.grant_figures import CONF_LABELS

    store, counts = pem.maps_by_epoch()
    if not store:
        return None
    EPO = list(ef.PANELS)
    cells, titles, amp, contours = {}, {}, {}, {}
    _stats = []

    def _arm(q, e):
        """``{animal: [session maps]}`` for one (position, epoch)."""
        d = {an: list(((by.get(e) or {}).get(q) or {}).values()) for an, by in store.items()}
        return {a: v for a, v in d.items() if v}

    for q in CONF_LABELS:
        row = _long_of(q)
        pre_arm = _arm(q, "pre")
        if not pre_arm:
            continue
        # POOLED WITHIN ANIMAL AND SCALE-NORMALISED -- see `beta_maps.within_animal_pooled`. Two
        # things this changed (Priya, 2026-09-12: "our post-stroke minus pre-stroke comparisons are
        # all within-animal only right?" and "how should we address the between-animal scale
        # variability"):
        #   * only animals present in BOTH epochs enter a difference. PS94 has no chronic, so
        #     `chronic - pre` had been 3-animal chronic minus 4-animal pre -- and PS94 is the
        #     DIMMEST animal, so leaving it on the pre side alone depressed the baseline and
        #     inflated every chronic ratio (near-ipsi 1.57 -> 1.27, far-middle 1.70 -> 1.45).
        #   * each animal is divided by its OWN pre-stroke amplitude, so the pooled map is an
        #     average of animals rather than of brightness (a 2.44x spread across animals).
        pre_p, _a, _b, _c, pre_ans = bm.within_animal_pooled(pre_arm)
        cells[(row, "pre")] = pre_p
        titles[(row, "pre")] = (f"pre\n{len(pre_ans)} an, "
                                f"{sum(len(v) for v in pre_arm.values())} sess")
        amp[q] = {"pre": 1.0}
        for e in ("acute", "subacute", "chronic"):
            arm = _arm(q, e)
            if not arm:
                continue
            _p, post_p, delta, ratio, ans = bm.within_animal_pooled(pre_arm, arm)
            if post_p is None:
                continue
            n_s = sum(len(v) for a, v in arm.items() if a in ans)
            cells[(row, e)] = post_p
            titles[(row, e)] = f"{e}\n{len(ans)} an, {n_s} sess"
            cells[(row, f"{e} - pre")] = delta
            titles[(row, f"{e} - pre")] = f"{e.upper()} - PRE\n{len(ans)} an, within-animal"
            amp[q][e] = ratio
        if True:
            # THE PERMUTATION TEST IS BETTER POSED HERE THAN ON FIGURE 14, because these six maps
            # are INDEPENDENT: the null "this position's epoch label carries no information" is a
            # real null, where on a one-vs-rest map relabelling one position perturbs the reference
            # of the other five. Labels shuffled WITHIN animal; black contour = cluster mass above
            # the 95th percentile of the null.
            pre_by = pre_arm
            for e in ("acute", "subacute", "chronic"):
                post_by = _arm(q, e)
                if not (pre_by and post_by):
                    continue
                try:
                    cm, lab = bm.significance_contour(pre_by, post_by)
                    print(f"  .. 15e {q} {e}: {lab}", flush=True)
                    _stats.append(dict(row=_long_of(q), col=f"{e} - pre",
                                       n_animals=len(set(pre_by) & set(post_by)),
                                       animals=";".join(sorted(set(pre_by) & set(post_by))),
                                       amplitude_vs_pre=round(amp.get(q, {}).get(e, float("nan")), 4),
                                       **getattr(bm.significance_contour, "last", {})))
                    if cm is not None and np.any(cm):
                        contours[(_long_of(q), f"{e} - pre")] = cm
                except Exception as ex:                                # noqa: BLE001
                    print(f"  !! 15e sig {q} {e}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
    if not cells:
        return None
    rows = [_long_of(q) for q in CONF_LABELS if any((_long_of(q), e) in cells for e in EPO)]
    a_txt = ", ".join(f"{_long_of(q)} {amp[q].get('acute', float('nan')):.2f}"
                      for q in CONF_LABELS if q in amp)
    return ef.map_grid(
        cells, out_dir, name="epoch_15_evoked_CUEINCREMENT_PRECUEref_cue",
        title=("CUE-EVOKED INCREMENT -- post-cue MINUS PRE-CUE. Positions are independent, but "
               "this SUBTRACTS the anticipatory code (see F12): not the position map"),
        row_labels=rows,
        col_labels=EPO + [f"{e} - pre" for e in ("acute", "subacute", "chronic")],
        panel_titles=titles,
        delta_cols=tuple(f"{e} - pre" for e in ("acute", "subacute", "chronic")),
        edges=bm.atlas_edges(), contours=contours,
        blank=bm.excluded_mask(),
        cbar_label="post-cue minus pre-cue\n(that position's OWN trials)",
        delta_label="change vs pre-stroke\n(SAME scale as the maps)",
        row_scaled=False,
        stat_rows=_stats,
        subtitle=(
            "READ THIS BEFORE FIGURE 14. Figure 14's decoder maps are one-vs-rest, centred on the "
            "mean over all six positions, so a position that loses drive lowers the reference and "
            "hands every other position an increase it did not earn -- four of its six rows cannot "
            "be read as written. Here each map is that position's OWN post-cue minus pre-cue, a "
            "WITHIN-TRIAL reference, so the positions are independent. "
            "THE COST: this shows the whole task-evoked response -- cue, licking, movement, "
            "arousal -- not only the part that carries target identity, which is what figure 14 "
            "isolates. A change visible here and absent there is a change in DRIVE without "
            "position information; the reverse is a change in TUNING without a change in drive. "
            "Maps from `framemap_event_maps`; nothing recomputed. Colour scale per ROW; the "
            "difference columns share their row's scale. THE THIN DARK OUTLINES ARE ALLEN CCF "
            "BOUNDARIES, not statistics. BLACK contours are bins where the change differs from "
            f"zero. {_stats_sentence(_stats)} "
            "It STEPS rather than curving because it is drawn on those "
            "bins: the maps carry no spatial detail finer than FWHM ~81 px, so a smooth "
            "full-resolution contour would claim a precision the data does not have. "
            f"Acute amplitude relative to each position's own pre-stroke value: {a_txt}."))
def _fig_14z_beta_vs_zero(out_dir, align, variant, wname):
    """14z: WHERE the position code is, epoch by epoch -- Musall et al. 2023 fig. S6's question.

    Priya, 2026-09-12: "we still haven't done the Musall-style analysis of comparing each pre/post
    epoch to zero and then comparing which locations are significantly encoding direction, right?"
    Correct -- `musall_significance` existed but had only ever been pointed at DIFFERENCES. This is
    their actual figure: per epoch, per position, which downsampled pixels carry a decoder weight
    that differs from zero.

    A DIFFERENT QUESTION FROM EVERY OTHER MAP FIGURE HERE, and the reason it earns its place is
    that it is immune to the two confounds the difference tests carry. Between-epoch comparisons are
    confounded with elapsed weeks (window clearing, expression, photobleaching), and the
    quiet-referenced ones additionally with a baseline that drifts +0.0026 by chronic. A vs-zero
    test lives inside ONE epoch and asks only whether the map is there.

    READ IT AS FOUR SEPARATE STATEMENTS, NEVER AS A DIFFERENCE. "Significant in pre, not in acute"
    is not evidence of a change: a map that just clears the threshold in one epoch and just misses
    it in the next may not differ at all, and nothing here puts an error bar on that difference.
    The difference question has its own figure and its own test.

    THE UNIT IS THE ANIMAL, via the nested animals-to-sessions bootstrap, as everywhere else in this
    deck. Musall pooled SESSIONS, which is far more powerful and treats two sessions from one animal
    as independent; that number is printed in the render log for comparison and is not what the
    contour draws.
    """
    if not ((variant == "working" and align in ("precue", "cue"))
            or (variant == "lick" and align == "lick")):
        return None
    from wfield_local import beta_maps as bm
    from wfield_local.grant_figures import CONF_LABELS

    store, _rel, ntr = bm.maps_by_epoch(align, variant)
    if not store:
        return None
    EPO = list(ef.PANELS)
    cells, titles, contours, rows = {}, {}, {}, []
    _stats = []
    for q in CONF_LABELS:
        row = _long_of(q)
        got_any = False
        for e in EPO:
            by_an = {an: list(((by.get(e) or {}).get(q) or {}).values())
                     for an, by in store.items()}
            by_an = {a: v for a, v in by_an.items() if v}
            if len(by_an) < 2:
                continue
            got_any = True
            cells[(row, e)] = np.mean([np.mean(v, 0) for v in by_an.values()], axis=0)
            n_s = sum(len(v) for v in by_an.values())
            n_tr = sum(sum((((ntr.get(an) or {}).get(e) or {}).get(q) or {}).values())
                       for an in by_an)
            try:
                cm, lab = bm.vs_zero_contour(by_an)
                print(f"  .. 14z {q} {e}: {lab}", flush=True)
                _stats.append(dict(row=row, col=e, n_animals=len(by_an),
                                   animals=";".join(sorted(by_an)), n_sessions=n_s, n_trials=n_tr,
                                   label=lab))
                # Musall's own unit, logged beside ours so the difference is visible rather than
                # asserted -- sessions give far more significance and are pseudo-replicated.
                _cm2, lab2 = bm.vs_zero_contour(by_an, method="musall")
                print(f"  .. 14z {q} {e}: [reference] {lab2}", flush=True)
                if cm is not None and np.any(cm):
                    contours[(row, e)] = cm
                n_sig = lab.split(":")[-1].strip().split(" of ")[0]
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 14z {q} {e}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
                n_sig = "?"
            titles[(row, e)] = f"{e}\n{len(by_an)} an, {n_s} sess, n={n_tr}\nsig {n_sig} bins"
        if got_any:
            rows.append(row)
    if not cells:
        return None
    return ef.map_grid(
        cells, out_dir, name=f"epoch_14z_beta_vs_ZERO_{align}_{variant}",
        title=("WHERE the position code IS, epoch by epoch -- decoder weights tested against ZERO "
               f"(Musall et al. 2023 fig. S6). {wname}"),
        row_labels=rows, col_labels=EPO, panel_titles=titles, edges=bm.atlas_edges(),
        blank=bm.excluded_mask(),
        contours=contours,
        cbar_label=("cov(pixel, decoder output)\nred = MORE active on this position's\n"
                    "trials than on the average trial"),
        row_scaled=False,
        stat_rows=_stats,
        subtitle=(
            "A DIFFERENT QUESTION FROM EVERY OTHER MAP FIGURE HERE: not where the code CHANGED, but "
            "where it IS in each epoch. Each panel is tested on its own against zero, so it is "
            "immune to the two confounds the difference figures carry -- elapsed weeks, and a quiet "
            "baseline that drifts by chronic. "
            "DO NOT READ TWO PANELS AS A DIFFERENCE. \"Significant in pre and not in acute\" is "
            "NOT evidence of a change: a map that just clears the threshold in one epoch and just "
            "misses it in the next may not differ at all, and nothing on this figure puts an error "
            "bar on that comparison. The difference question has its own figure and its own test. "
            "METHOD, following Musall et al. 2023 fig S6: maps downsampled 8x to a 67x80 grid "
            f"({_n_stat_bins():,} bins inside the eroded statistics mask, against their 3,364), "
            "tested per bin against zero, "
            "Bonferroni-corrected over those bins. THE UNIT IS THE ANIMAL, through the "
            "animals-to-sessions bootstrap this deck uses everywhere; Musall pooled SESSIONS, which "
            "is far more powerful and treats two sessions from one animal as independent -- that "
            "number is printed in the render log beside ours. Contours are drawn on an ERODED mask "
            "and any result concentrated more than 2x in the rim is suppressed, because the edge of "
            "the imaging window produces artefacts that are consistent ACROSS animals and so cannot "
            "be rejected by a between-animal test. Since the maps are one-vs-rest, a significant "
            "bin means this position's trials differ from the average trial THERE -- read it with "
            "the coupling caveat that applies to every mean-referenced figure."))
def _fig_14_beta_maps(out_dir, align, variant, wname):
    """14: WHERE the position code lives in cortex, and where it goes -- pooled decoder maps.

    Priya, 2026-09-12: "I want to start trying to answer *where* the displaced spout position codes
    move post-stroke", then "Could we do a beta weights mapping, as was done in this paper?"
    (Musall et al., Nat Neurosci 2022).

    EVERY OTHER FAMILY IN THIS SECTION ANSWERS "MOVED TOWARD WHAT", NOT "MOVED WHERE". Best-match
    destination collapses 380 features into one correlation per position pair and takes an argmax --
    a representational destination, not a location. Per component the joint basis gives 2-4
    components per Allen area, too thin to localise. This is the anatomical arm.

    METHOD, and the three departures from the paper are each measured -- see `wfield_local/beta_maps.py`:

      FEATURES   the temporal component matrix SVT, rank 100, NOT LocaNMF. `U @ A` is then a true
                 540 x 640 pixel map with no component-space intermediary, and it costs nothing:
                 SVT beats LocaNMF features by 0.06-0.09 balanced accuracy.
      PENALTY    L2, not the paper's L1. We are making a map, not selecting features, and L1's
                 choice among correlated predictors is free to change between days. Split-half of
                 the pre-stroke mean map: L1 0.688, L2 0.715, L2+Haufe 0.960.
      TRANSFORM  HAUFE, which the paper does not do and which is the single biggest factor above.
                 A decoder weight is a FILTER whose job includes cancelling correlated noise, so a
                 channel with NO signal can carry a large weight as a suppressor. `A = Cov(X) @ b`
                 makes it a PATTERN -- cov(channel, decoder output) -- which is the anatomical
                 question. Pattern and filter correlate at only r = 0.245 here.
      BALANCE    the lick arm only. `working` is uniform over positions by construction
                 (16.1-17.2% each) so pre-cue and post-cue need none; `lick` runs far-contra at
                 9.2-14.2% against ~19% near, and that skew IS the deficit.

    THE COLOUR SCALE IS PER ROW so each position is comparable across ITS OWN epochs, which is the
    comparison being made. Rows are not comparable to each other.

    `r` IN EACH PANEL IS THE SPLIT-HALF RELIABILITY of that epoch's mean map -- the ceiling any
    difference involving it can reach. IT IS NOT A CAVEAT TO DISCOUNT THE RESULT WITH: a split-half
    correlation of a near-absent signal is low BECAUSE the signal is near-absent. Far-contra acute
    is r = 0.53 AND 0.47 of its pre-stroke amplitude; those are one observation, not two. Far-middle
    falls to 0.48 amplitude while KEEPING r = 0.86, which is what shows the two are separable.

    THE RESULT, and it converges with the encoder from a completely different direction. Map
    amplitude relative to each position's own pre-stroke value:

        near ipsi 0.67   near middle 1.53   near contra 1.04
        far ipsi  1.09   far middle  0.48   far CONTRA  0.47      (acute)

    The two positions that lose more than half their map amplitude acutely are far-middle and
    far-CONTRA -- position-specific, in exactly the pair every other analysis implicates, and both
    recover by subacute (0.83 / 0.91). The encoder's fitted amplitude factor tells the same story in
    components rather than pixels: 0.286 acute against 0.749 pre-stroke, recovering to 0.745.
    """
    # THE THREE REAL ARMS: pre-cue and post-cue on `working`, and post-lick on `lick`. The two
    # `stopped` arms are excluded because these maps are fitted on the position label and the quit
    # period has too few trials per position to fit six classes -- that question is 12b's, pooled.
    if not ((variant == "working" and align in ("precue", "cue"))
            or (variant == "lick" and align == "lick")):
        return None
    from wfield_local import beta_maps as bm
    from wfield_local.grant_figures import CONF_LABELS

    store, rel, ntr = bm.maps_by_epoch(align, variant)
    if not store:
        return None

    EPO = [e for e in ef.PANELS]
    DELTA = "acute - pre"
    cells, titles, amp, contours = {}, {}, {}, {}
    _stats = []
    def _arm(q, e):
        """``{animal: [session maps]}`` for one (position, epoch)."""
        d = {an: list(((by.get(e) or {}).get(q) or {}).values()) for an, by in store.items()}
        return {a: v for a, v in d.items() if v}

    def _meta(q, e, ans):
        """``(n_sessions, n_trials, median reliability)`` over the animals actually used."""
        n_s = n_tr = 0
        rs = []
        for an in ans:
            got = (store.get(an, {}).get(e) or {}).get(q) or {}
            n_s += len(got)
            n_tr += sum((((ntr.get(an) or {}).get(e) or {}).get(q) or {}).values())
            r = ((rel.get(an) or {}).get(e) or {}).get(q)
            if r is not None and np.isfinite(r):
                rs.append(r)
        return n_s, n_tr, (float(np.median(rs)) if rs else float("nan"))

    for q in CONF_LABELS:
        row = _long_of(q)
        # A CELL REFUSED FOR TOO FEW TRIALS MUST SAY SO. Priya, 2026-09-12, of the lick-aligned
        # arm: "why is there no subacute or chronic far contra delta data?" -- because on that arm
        # far-contralateral needs actual LICKS at that position, and `MIN_TRIALS_PER_CLASS` refuses
        # the cell. That refusal IS the deficit, and a silently absent panel reads as a rendering
        # bug rather than as the finding it is.
        _refused = [e for e in EPO if not _arm(q, e)]
        for e in _refused:
            titles[(row, e)] = (f"{e}\nREFUSED\nno session reached\n"
                                f"{bm.MIN_TRIALS_PER_CLASS} trials")
        if _refused:
            print(f"  .. 14m {q}: refused {', '.join(_refused)} "
                  f"(<{bm.MIN_TRIALS_PER_CLASS} trials at this position)", flush=True)
        pre_arm = _arm(q, "pre")
        if not pre_arm:
            continue
        # POOLED WITHIN ANIMAL AND SCALE-NORMALISED -- `beta_maps.within_animal_pooled`. Only
        # animals present in BOTH epochs enter a difference (PS94 has no chronic, and it is also the
        # dimmest animal, so leaving it on the pre side alone inflated every chronic ratio), and
        # each animal is divided by its own pre-stroke amplitude so the pooled map is an average of
        # animals rather than of brightness.
        pre_p, _a, _b, _c, pre_ans = bm.within_animal_pooled(pre_arm)
        n_s, n_tr, rr = _meta(q, "pre", pre_ans)
        cells[(row, "pre")] = pre_p
        # TRIALS, not just sessions. A cell built from 30 trials cannot be allowed to look like one
        # built from 521 (Priya: "can you include the n? the far R n may be low").
        titles[(row, "pre")] = (f"pre\n{len(pre_ans)} an, {n_s} sess, n={n_tr}"
                                + (f"\nr={rr:.2f}" if np.isfinite(rr) else ""))
        amp[q] = {"pre": 1.0}
        per_epoch = {"pre": pre_p}
        for e in ("acute", "subacute", "chronic"):
            arm = _arm(q, e)
            if not arm:
                continue
            _p, post_p, delta, ratio, ans = bm.within_animal_pooled(pre_arm, arm)
            if post_p is None:
                continue
            n_s, n_tr, rr = _meta(q, e, ans)
            per_epoch[e] = post_p
            cells[(row, e)] = post_p
            titles[(row, e)] = (f"{e}\n{len(ans)} an, {n_s} sess, n={n_tr}"
                                + (f"\nr={rr:.2f}" if np.isfinite(rr) else ""))
            # ONE DELTA COLUMN PER POST-STROKE EPOCH (Priya asked for all three): acute-minus-pre
            # alone shows the hit and not the recovery, and recovery is half this deck's claim.
            col = DELTA if e == "acute" else f"{e} - pre"
            cells[(row, col)] = delta
            titles[(row, col)] = f"{e.upper()} - PRE\n{len(ans)} an, within-animal"
            amp[q][e] = ratio
        if "acute" in per_epoch:
            # THE TEST, not the eye. 345,600 pixels makes an uncorrected threshold meaningless;
            # this shuffles epoch labels WITHIN animal and keeps clusters larger than 95% of those
            # obtainable by relabelling. Same statistic the panel draws.
            pre_by, post_by = pre_arm, _arm(q, "acute")
            try:
                cm, lab = bm.significance_contour(pre_by, post_by)
                print(f"  .. 14m {q} acute: {lab}", flush=True)
                _stats.append(dict(row=row, col=DELTA,
                                   n_animals=len(set(pre_by) & set(post_by)),
                                   animals=";".join(sorted(set(pre_by) & set(post_by))),
                                   amplitude_vs_pre=round(amp.get(q, {}).get("acute",
                                                                            float("nan")), 4),
                                   **getattr(bm.significance_contour, "last", {})))
                if cm is not None and np.any(cm):
                    contours[(row, DELTA)] = cm
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 14m sig {q}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
    if not cells:
        return None

    rows = [_long_of(q) for q in CONF_LABELS if any((_long_of(q), e) in cells for e in EPO)]
    a_txt = ", ".join(f"{_long_of(q)} {amp[q].get('acute', float('nan')):.2f}"
                      for q in CONF_LABELS if q in amp)
    return ef.map_grid(
        cells, out_dir, name=f"epoch_14_beta_maps_MEANref_{align}_{variant}",
        title=(f"DECODER BETA MAPS -- referenced to the MEAN OVER ALL TRIALS, so the six "
               f"positions are COUPLED. {wname}"),
        row_labels=rows,
        col_labels=EPO + [DELTA, "subacute - pre", "chronic - pre"], panel_titles=titles,
        delta_cols=(DELTA, "subacute - pre", "chronic - pre"),
        edges=bm.atlas_edges(), contours=contours,
        blank=bm.excluded_mask(),
        cbar_label=("cov(pixel, decoder output)\nred = MORE active on this position's\n"
                    "trials than on the average trial"),
        delta_label="change vs pre-stroke\n(SAME scale as the maps)",
        row_scaled=False,
        stat_rows=_stats,
        subtitle=(
            "L2 logistic on the rank-100 SVT, beta Haufe-transformed to a PATTERN "
            "(A = Cov(X) beta) and rendered as U @ A -- full-resolution pixels, not components. "
            "Read the pattern for WHERE THE SIGNAL IS; the filter, which answers what the decoder "
            "USES, is a different map (they correlate at r = 0.245). "
            "Allen CCF boundaries overlaid. "
            f"{_stats_sentence(_stats)} "
            "Colour scale is PER ROW, so a position is comparable across its own epochs and rows "
            "are not comparable to each other. "
            "r = split-half reliability of that epoch's mean map, the ceiling a difference can "
            "reach -- and where a map is near-absent, low r IS the result rather than a reason to "
            "doubt it. "
            f"Acute map amplitude relative to each position's own pre-stroke value: {a_txt}."))

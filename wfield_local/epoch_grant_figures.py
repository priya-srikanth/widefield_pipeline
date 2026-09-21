"""Render the pooled cross-animal EPOCH figures.

Priya, 2026-08-28: pooled versions of the grant figures with three panels -- pre-stroke, acute and
subacute -- instead of a linear time axis, sized to be read at a quarter page or smaller. And:
*"I do NOT want this to have to re-create the wheel."*

SO THIS MODULE COMPUTES NOTHING. It calls `grant_figures`' existing collectors, groups their
per-day records by `epochs.epoch_of`, and hands the groups to `epoch_figures`' renderers. Every
population here is the SAME OBJECT the per-animal figures are drawn from, which is the only way two
figures captioned with the same trials can be relied on to contain them -- `tests/
test_epoch_figures.py` forbids this module from fitting a model or pooling sessions itself.

WHAT POOLING MEANS DIFFERS BY MEASURE, and the difference is not cosmetic:

  * `_collect_5c` returns TRIAL-LEVEL records `(y_true, y_pred, blocks)` scored by each animal's own
    frozen pre-stroke decoder. Pooling those is a CONCATENATION and the confusion matrix a SUM,
    because raw counts add. Every trial counts once, so a session with more trials counts for more.
  * `_matrices_*` return matrices that have already been reduced. Pooling those is a MEAN OVER
    SESSIONS -- which is exactly the weighting Priya asked for (*"weighted mean (ie just use each
    session's value)"*), and it is why the session dots matter: the acute panel is six PS94
    sessions against one PS95 session.

Run: ``python -m wfield_local.epoch_grant_figures [--only 1b 5c ...] [--output DIR]``
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wfield_local import epoch_figures as ef
from wfield_local.epoch_accuracy import (  # noqa: F401
    CHANCE,
    OVERALL_ARMS,
    _accuracy_of,
    _code_of,
    _confusion_rows,
    _epoch_arm,
    _fig_8g,
    _fig_9,
    _frozen_vs_refit,
    _frozen_vs_refit_matched,
    _frozen_vs_refit_overall,
    _frozen_vs_refit_overall_matched,
    _gap_at,
    _gap_of,
    _matrix_family,
    _named,
    _overall_at,
    _overall_of,
    _per_position_accuracy,
    _position_bars,
    _refit_at,
    _refit_confusion_rows,
    _refit_of,
    _report,
)
from wfield_local.epoch_behaviour import (  # noqa: F401
    _behaviour_contrast,
    _behaviour_records,
    _licks_per_trial_by_day,
    _position_codes,
    fig_behaviour,
    fig_behaviour_timecourse,
)

# THE FIGURE FAMILIES, one module each (2026-09-21). What is left here is the DRIVER:
# the --only/--arm dispatch, the family registry, and the CLI. Everything is re-exported,
# because `epoch_figures`, the deck's coverage tooling and a dozen tests address these
# names through this module.
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
from wfield_local.epoch_maps import (  # noqa: F401
    _CORRECTION_TEXT,
    _REF_TEXT,
    _fig_14_beta_maps,
    _fig_14pa_beta_maps_by_animal,
    _fig_14z_beta_vs_zero,
    _fig_15_evoked_maps,
    _fig_15pa_evoked_maps_by_animal,
    _fig_15r_reference_maps,
    _fig_15rpa_reference_by_animal,
    _long_of,
    _n_stat_bins,
    _stats_sentence,
)
from wfield_local.epoch_matching import (  # noqa: F401
    _fig_10,
    _fig_10b,
    _fig_10cs,
    _fig_10e_best_match_grid,
    _fig_11,
    _fig_11c,
    _fig_11cpos,
)
from wfield_local.epoch_state import (  # noqa: F401
    _fig_12_stopped,
    _fig_12b_stopped_pooled,
    _fig_13_state,
    _position_accuracy_by_epoch,
    _retained,
    _state_balance_line,
    _state_class_share,
    _state_epoch_values,
)
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

#: (display name, alignment, trial class). The lick window admits only lick trials -- a trial with
#: no detected lick has no lick to align to -- which is `grant_figures._variants`' rule, restated
#: here as data because these figures name the class in their titles.
#: (display key, alignment, trial class, caption). THE STOPPED ARMS ARE THE COMPLEMENT of the
#: `working` ones, not a subset: `flag_engagement`'s terminal quit period alone, which every other
#: arm removes (Priya, 2026-09-11: "the stopped class should basically be another frozen decoder /
#: encoder analysis set"). No lick-aligned stopped arm exists and none can -- a trial inside the
#: quit period is a non-response by construction, so there is no lick to align to.
ARMS = (("ENL", "precue", "working", "ENL (pre-cue), lick + miss-while-working"),
        ("cue", "cue", "working", "post-cue, lick + miss-while-working"),
        ("lick", "lick", "lick", "post-lick, lick trials only"),
        # `ENLstop` and `cuestop` REMOVED 2026-09-12 (Priya: "I think we can get rid of the
        # post-stroke stopped vs pre-stroke lick analyses and figures"). They ran every generic
        # family on the quit period, but `_collect_7` hardcodes the pre-stroke side to `lick`, so
        # they compared post-stroke stopped trials against pre-stroke ENGAGED cortex -- two
        # different behavioural states, where a drop is the null expectation rather than a result.
        # Families 12 and 12b ask it properly (12b state-matched against pre-stroke STOPPED) and
        # render on the `working` pass, so they are untouched. See `grant_figures._variants`.
        # THE SELECTION CONTROL FOR THE LICK-ALIGNED RESULT (Priya, 2026-09-12). The post-lick arm
        # is the ONLY place the refit-minus-frozen gap survives training-set matching -- +0.179
        # acute and +0.112 subacute, both Bonferroni-corrected. But that arm CONDITIONS ON A
        # DETECTED LICK, so post-stroke its trial population is itself a product of the deficit:
        # the animal licks less and licks different spouts, and acute far-contralateral is gated
        # out entirely below MIN_REFIT_SHARE.
        #
        # These two hold the TRIAL SET fixed at lick trials and move only the WINDOW. If the effect
        # is about position coding it should appear post-cue on the same trials; if it is about
        # executing a completed movement it should not. Neither exists in any other family, which is
        # why the lick-aligned result could not be interpreted before.
        ("cuelick", "cue", "lick", "post-cue, LICK TRIALS ONLY"),
        ("ENLlick", "precue", "lick", "ENL (pre-cue), LICK TRIALS ONLY"))















# --------------------------------------------------------------------------------- behaviour











# ------------------------------------------------------------- decoding, from `_collect_5c`



























































































# ------------------------------------------------------- the already-reduced matrix families

#: (key, collector name, colour bar unit, colormap, fixed scale or None, title stem).
#: The scale is fixed only where the quantity has a natural range: a correlation does, a
#: crossnobis distance does not, and forcing one on it would compress every panel into a corner.
MATRIX_FAMILIES = (
    ("6", "_matrices_pattern", "pattern correlation", "viridis", (-1.0, 1.0),
     "Mean-pattern correlation against the pre-stroke reference"),
    ("7", "_matrices_splithalf", "split-half correlation", "viridis", (-1.0, 1.0),
     "Within-session split-half pattern similarity"),
    ("8", "_matrices_crossnobis", "crossnobis distance", "magma", None,
     "Crossnobis geometry, in pre-stroke units"),
    # WHICH POSITION DID IT MOVE TOWARD (Priya, 2026-09-09). Family 8 answers "did this position
    # move"; its rows are confounded by amplitude, because a pure gain change in P shifts P's
    # distance to every pre-stroke position equally and paints a uniform row. Row-centring removes
    # that term, so the off-diagonal contrast is substitution rather than gain. Needed because the
    # substitution claim otherwise rests on decoder confusions and best-match fraction, both of
    # which are LABEL-level -- they say which position the readout assigns, not which position the
    # pattern moved toward. Diverging: the scale is centred on zero, so RdBu_r not magma.
    ("8rc", "_matrices_crossnobis_rowcentred", "crossnobis distance, row-centred", "RdBu_r", None,
     "Crossnobis geometry, row-centred -- which position did it move TOWARD"),
    # WHERE THE BEST MATCH WENT (Priya, 2026-09-10): "could we add a version that plots, for each
    # spout position across epochs, where the best matches were -- so we could see if there is a
    # shift to one other position or if it's evenly distributed". Family 10b already reduces this
    # argmax to its diagonal; the whole matrix is the part that distinguishes SUBSTITUTION (mass on
    # one off-diagonal cell) from COLLAPSE (mass spread evenly). Fraction of sessions, so the scale
    # is a real 0-1 and fixing it is correct here where it is not for a distance.
    ("10c", "_matrices_best_match_destination", "fraction of sessions", "viridis", (0.0, 1.0),
     "Where each position's best pre-stroke match went"),
    # SCALE-FREE COMPANION TO 8rc (Priya, 2026-09-10). Row-centring removes amplitude from the row's
    # offset but leaves it multiplying the row's shape, so 8rc is comparable in DIRECTION across
    # epochs and not in MAGNITUDE. Dividing each row by its own SD makes it a z-profile and fixes
    # that, at the cost of discarding how far the position moved -- which 8 and 8rc still carry.
    ("8rz", "_matrices_crossnobis_rownorm", "row z-score", "RdBu_r", (-2.0, 2.0),
     "Crossnobis geometry, row-NORMALISED -- direction only, scale-free"),
)






# --------------------------------------------------------- the per-day scalar families





















SCALAR_FAMILIES = (("8g", _fig_8g), ("9", _fig_9), ("10", _fig_10),
                   ("10b", _fig_10b), ("11", _fig_11), ("11c", _fig_11c),
                   ("11cpos", _fig_11cpos))


# ------------------------------------------------------------------------------ the driver





def main(argv=None) -> int:
    from wfield_local.console import use_utf8_stdout
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    # `extend`, NOT the default `store`. With plain nargs="+" a REPEATED flag REPLACES the previous
    # value: `--only acc --only 13s` silently resolves to ["13s"] alone. That cost a 40-minute
    # render on 2026-09-11 -- five families were asked for, one ran, and the only symptom was a
    # figure still showing a number the code no longer produced. `extend` appends, which is what
    # repeating a flag reads as.
    #: PARALLELISM FOR THIS RENDERER IS PER ARM, and that is a measured choice rather than a
    #: limitation. The families within one arm SHARE their expensive collectors -- `_epoch_arm` is
    #: built once and `beta_maps.maps_by_epoch` / `position_reference_maps.maps_by_epoch` are
    #: lru_cached -- so every family after the first is nearly free. Splitting by FAMILY, the way
    #: `grant_figures` splits by unit, would rebuild those caches inside each worker and can run
    #: SLOWER than serial. The three arms genuinely share nothing: different alignment, different
    #: trial class, different collectors, and figure names carry `_{align}_{variant}` so no two
    #: arms can write the same file.
    #: KEYED ON THE DISPLAY NAME, NOT THE ALIGNMENT. `align` is NOT unique -- `cue` names both
    #: `cue/working` and `cuelick/cue/lick`, and `precue` names both `ENL` and `ENLlick` -- so an
    #: align-keyed flag silently renders two arms where one was asked for.
    ap.add_argument("--arm", nargs="+", default=None, action="extend",
                    choices=tuple(a[0] for a in ARMS),
                    help="render only these alignment arms (default: all). Use to run the arms as "
                         "separate concurrent processes; families within an arm share caches and "
                         "must stay in one process.")
    ap.add_argument("--only", nargs="+", default=None, action="extend",
                    choices=("1b", "1c", "acc", "5c", "5cr", "5r", "5rm", "5ro", "5rmo", "10e", "10cs", "12s", "12b",
                             "13s", "14m", "15e", "15r", "mat", "scal"))
    args = ap.parse_args(argv)
    out = args.output or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ("1b", "1c", "acc", "5c", "5cr", "5r", "5rm", "5ro", "5rmo", "10e", "10cs", "12s", "12b", "13s", "14m", "15e",
                                 "15r", "mat", "scal"))
    # PRINTED, so "I asked for five families and one ran" is visible in the log rather than in a
    # stale figure three hours later.
    print(f"[epoch] rendering families: {' '.join(sorted(want))}", flush=True)

    if "1b" in want:
        try:
            _report("1b", fig_behaviour(out))
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 1b: {type(ex).__name__} {str(ex)[:160]}", flush=True)
    if "1c" in want:
        try:
            _report("1c", fig_behaviour_timecourse(out))
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! 1c: {type(ex).__name__} {str(ex)[:160]}", flush=True)

    #: The per-arm keys. DERIVED, not written out again: the guard below skips the whole arm loop
    #: when none of them is wanted, and listing them twice meant `--only scal` and `--only mat`
    #: broke out of the loop immediately and produced NOTHING, with no error and no report --
    #: an empty output directory and exit 0.
    ARM_KEYS = {"acc", "5c", "5cr", "5r", "5rm", "5ro", "5rmo", "10e", "10cs", "12s", "12b", "13s", "14m", "15e",
                "15r", "mat", "scal"}
    arms = [a for a in ARMS if not args.arm or a[0] in set(args.arm)]
    print(f"[epoch] arms: {' '.join(a[0] for a in arms)}", flush=True)
    for disp, align, variant, wname in arms:
        if not (want & ARM_KEYS):
            break
        try:
            per_animal = _epoch_arm(align, variant)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! collect {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                  flush=True)
            continue
        cov = ef.epoch_coverage(per_animal)
        # UNASSIGNED DAYS ARE REPORTED, never folded into a neighbouring panel. A day between the
        # acute range and the first subacute day belongs to neither, and silently rounding it to
        # one would move a boundary Priya set.
        if cov["unassigned"]:
            print(f"  .. {align}/{variant} days in no epoch: {cov['unassigned']}", flush=True)
        print(f"  .. {align}/{variant} sessions per epoch: {cov['n']} {cov['per_epoch']}",
              flush=True)
        if "acc" in want:
            try:
                _report(f"acc {align}/{variant}",
                        _per_position_accuracy(per_animal, out, disp, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! acc {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "5c" in want:
            try:
                for p in _confusion_rows(per_animal, out, disp, align, variant, wname):
                    _report(f"5c {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 5c {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "5cr" in want:
            try:
                for p in _refit_confusion_rows(out, align, variant, wname):
                    _report(f"5cr {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 5cr {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        # 12s IS DISPATCHED ON THE `working` ARMS ONLY, and the guard belongs here rather than
        # inside the figure. It draws the stopped trials of every arm against the ENGAGED template,
        # so it is one figure per ALIGNMENT, not one per (alignment, class) -- and returning None
        # from the other three arms made the renderer print "NO FIGURE" three times a render for a
        # case that is correct by construction. A warning that always fires is a warning nobody
        # reads.
        if "15r" in want:
            try:
                for p in (_fig_15r_reference_maps(out, align, variant, wname) or []):
                    _report(f"15r {align}/{variant}", p)
                for p in (_fig_15rpa_reference_by_animal(out, align, variant, wname) or []):
                    _report(f"15rpa {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 15r {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "15e" in want:
            try:
                _report(f"15e {align}/{variant}",
                        _fig_15_evoked_maps(out, align, variant, wname))
                _report(f"15pa {align}/{variant}",
                        _fig_15pa_evoked_maps_by_animal(out, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 15e {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "14m" in want:
            try:
                _report(f"14m {align}/{variant}",
                        _fig_14_beta_maps(out, align, variant, wname))
                _report(f"14pa {align}/{variant}",
                        _fig_14pa_beta_maps_by_animal(out, align, variant, wname))
                _report(f"14z {align}/{variant}",
                        _fig_14z_beta_vs_zero(out, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 14m {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "13s" in want:
            try:
                for p in (_fig_13_state(out, align, variant, wname) or []):
                    _report(f"13s {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 13s {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        # ONE FIGURE PER ALIGNMENT. 12b pools a session's stopped trials whatever class the arm
        # names, so running it on the stopped arms too printed "NO FIGURE" twice a render for a
        # case that is correct by construction -- the same guard 12s needed.
        if "10cs" in want:
            try:
                _report(f"10cs {align}/{variant}", _fig_10cs(out, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 10cs {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "12b" in want and variant == "working":
            try:
                _report(f"12b {align}/{variant}",
                        _fig_12b_stopped_pooled(out, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 12b {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "12s" in want and variant == "working":
            try:
                _report(f"12s {align}/{variant}",
                        _fig_12_stopped(out, align, variant, wname))
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 12s {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "10e" in want:
            try:
                for p in (_fig_10e_best_match_grid(out, align, variant, wname) or []):
                    _report(f"10e {align}/{variant}", p)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! 10e {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        for _k, _fn in (("5r", _frozen_vs_refit), ("5rm", _frozen_vs_refit_matched),
                        ("5ro", _frozen_vs_refit_overall),
                        ("5rmo", _frozen_vs_refit_overall_matched)):
            if _k not in want:
                continue
            try:
                for q in (_fn(out, align, variant, wname) or []):
                    _report(f"{_k} {align}/{variant}", q)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! {_k} {align}/{variant}: {type(ex).__name__} {str(ex)[:160]}",
                      flush=True)
        if "scal" in want:
            for key, fn in SCALAR_FAMILIES:
                try:
                    _report(f"{key} {align}/{variant}", fn(out, align, variant, wname))
                except Exception as ex:                                # noqa: BLE001
                    print(f"  !! {key} {align}/{variant}: {type(ex).__name__} "
                          f"{str(ex)[:160]}", flush=True)
        if "mat" in want:
            for key, collector, unit, cmap, scale, stem in MATRIX_FAMILIES:
                try:
                    _report(f"{key} {align}/{variant}",
                            _matrix_family(key, collector, unit, cmap, scale, stem,
                                           out, align, variant, wname))
                except Exception as ex:                                # noqa: BLE001
                    print(f"  !! {key} {align}/{variant}: {type(ex).__name__} "
                          f"{str(ex)[:160]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

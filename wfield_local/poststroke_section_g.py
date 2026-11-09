"""Compute EVERY Section G quantity, for every post-stroke and excluded session, on both arms.

ONE runner, ONE file per run. Section G used to be produced by four scratchpad scripts writing eight
JSONs with names like `poststroke_matched_0817.json`; twelve of the deck's figures were still being
drawn from files those scripts wrote on 2026-08-18, on a basis corrected the next morning, and nobody
could tell by looking. Everything Section G shows is now computed here and rendered by
`section_g_figures`, so "is this figure current?" is answered by one file's timestamp.

WHY PER SESSION. Earlier runs pooled an animal's post-stroke sessions into one "post" column. That
was invisible with one post-stroke night; with two it averages a day-1 and a day-2 brain, and for
PS94 those differ more from each other than pre differs from post (pre-cue z=-0.2 on 8/17 against
-3.4 on 8/18). Records are keyed by SESSION LABEL and the pooled form is not offered.

WHY BOTH ARMS.
  ALL       every cued trial, scored over all six positions. Chance is 1/6 for every session and
            animal, so this is the ONLY arm comparable across sessions -- and the missing licks are
            the phenotype, so it is also the arm that can see the abandoned positions at all.
  LICK-ONLY that session's preserved positions, chance 1/n. Forced, not chosen: a position with no
            engaged trials cannot be decoded from engaged trials. Its chance level moves with the
            animal's behavior (PS95: 4 positions on 8/17, 6 on 8/18), so its accuracies must NEVER
            be laid side by side across sessions.
The DIFFERENCE between the arms is the point: it separates "the code degraded" from "the code is
fine whenever the animal manages to lick". Post-lick has one arm by construction -- it aligns to the
first lick, so a trial without one has no reference point.

WHY THE NO-LICK READOUTS SIT OUTSIDE THE ARMS. `looks_like_which`, `fits_engaged_distribution` and
`impaired_nolick_readout` read the NO-LICK arm on purpose; that is what they are for. They are
computed once per session rather than once per arm, which is also why they are exempt from the
all-trials rule in nolick_analysis.SANCTIONED_MISMATCHES.

EXCLUDED SESSIONS ARE INCLUDED, TAGGED. PS92/PS93 8/17 follow the 8/16 laser that did not take, so
they belong to neither phase -- and they are the SMALL-LESION comparison (G7) and, more importantly,
the within-animal before/after control for the same animals' 8/18 sessions (G2c). Selecting them by
date is what the retired scripts did; they come from `excluded_labels` here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local import poststroke_compare as pc
from wfield_local.paths import PathResolver
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES

#: (alignment key, the condition name the figures use)
CONDITIONS = (("cue", "post-cue"), ("lick", "post-lick"), ("precue", "pre-cue"))
ARMS = ("all", "lickonly")


def post_animals(animals=None):
    """Animals with at least one post-stroke session, from the phase resolver, never a date."""
    if animals:
        return sorted(animals)
    return sorted({config.animal_of(lab) for lab in config.phase_labels("post")})


def _counts(d, session):
    """Engaged / undetected trials per position: this SESSION against the pre-session mean.

    Behavior is reported before any decoding number because it bounds what one can mean: a position
    with zero engaged trials has no lick-only decoding number at all, and the first pass at this
    analysis reported a PS94 "neural deficit" whose larger part was that fact.
    """
    pos = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    out = {"pre": {"engaged": {}, "undetected": {}}, "post": {"engaged": {}, "undetected": {}}}
    npre = max(len(d["pre_i"]), 1)
    for arm, Y, G in (("engaged", d["YE"], d["GE"]), ("undetected", d["YU"], d["GU"])):
        for c, p in zip(DISPLAY_ORDER, pos):
            out["pre"][arm][p] = float((np.isin(G, list(d["pre_i"])) & (Y == c)).sum()) / npre
            out["post"][arm][p] = int((np.isin(G, [session]) & (Y == c)).sum())
    # THE CAMERA GROUPS (main 4a37cbc8, Priya 11-04, item G; `poststroke_compare._camera_groups`): per group and
    # position, the pre per-session mean over the pre sessions WITH camera classes, and this session's count; absent
    # under the DAQ grouping, and an empty post block for a session without camera classes
    from wfield_local import poststroke_compare as pc
    cg = pc._camera_groups(d)
    if cg is not None:
        from wfield_local import working_trials as wt
        pre_c = [i for i in d["pre_i"] if i in cg["have"]]
        bc = {"pre": {}, "post": {}, "post_has_camera_classes": session in cg["have"],
              "n_jaw_unjudged_post": int(np.sum(~cg["jj_e"][d["GE"] == session])
                                         + np.sum(~cg["jj_u"][d["GU"] == session]))}
        for g in wt.G_ALL:
            bc["pre"][g], bc["post"][g] = {}, {}
            for c, p in zip(DISPLAY_ORDER, pos):
                npre = (np.sum(np.isin(d["GE"], pre_c) & (d["YE"] == c) & (cg["grp_e"] == g))
                        + np.sum(np.isin(d["GU"], pre_c) & (d["YU"] == c) & (cg["grp_u"] == g)))
                bc["pre"][g][p] = float(npre) / max(len(pre_c), 1)
                bc["post"][g][p] = int(np.sum((d["GE"] == session) & (d["YE"] == c) & (cg["grp_e"] == g))
                                       + np.sum((d["GU"] == session) & (d["YU"] == c) & (cg["grp_u"] == g)))
        out["by_class"] = bc
    return out


def _one_arm(d_by_align, session, keep, arm_all, keep_lick=None):
    """The arm-dependent half: matched decoding, recoding, similarity, confusion. ``keep_lick`` = the lick-only arm's
    positions at the LICK alignment (`lick_keep`; default ``keep``)."""
    keep_lick = keep if keep_lick is None else keep_lick
    rec = {"arm": "ALL trials" if arm_all else "lick-only",
           "n_positions_scored": len(DISPLAY_ORDER) if arm_all else len(keep),
           "chance": (1.0 / len(DISPLAY_ORDER)) if arm_all else (1.0 / max(len(keep), 1))}
    if len(keep) < 2 and not arm_all:
        rec["note"] = f"only {len(keep)} preserved position(s): this arm does not exist here"
        return rec

    for align, cond in CONDITIONS:
        d = d_by_align.get(align)
        if d is None or (align == "lick" and arm_all):     # no lick, no alignment point
            continue
        dd = dict(d)
        dd["post_i"] = {session}
        kp = keep_lick if align == "lick" else keep
        if align == "lick" and len(kp) < 2:
            rec[f"{cond} note"] = (f"only {len(kp)} position(s) with >= {pc.MIN_POST} DAQ-lick trials: no lick-aligned "
                                   "lick-only arm")
            continue
        for fn, key in ((pc.decode_matched, cond), (pc.recoding_test, f"{cond} within-session")):
            try:
                r = fn(dd, kp, post_all_trials=arm_all)
            except Exception as ex:                                       # noqa: BLE001
                print(f"      {key}: {fn.__name__} failed ({str(ex)[:60]})", flush=True)
                continue
            if r and "note" not in r:
                rec[key] = r

    d0 = d_by_align.get("cue")
    if d0 is not None:
        dd = dict(d0)
        dd["post_i"] = {session}
        try:
            rec["postcue_pattern_similarity"] = pc.pattern_similarity(
                dd, keep, post_all_trials=arm_all)
        except Exception as ex:                                           # noqa: BLE001
            print(f"      pattern_similarity failed ({str(ex)[:60]})", flush=True)
    # G2b needs the LICK alignment as well as cue/precue. Post-lick has ONE arm by construction:
    # it aligns to an event that exists only when the animal licked, so the all-trials arm -- whose
    # post panel is engaged PLUS no-lick -- has no meaningful lick-aligned version, and the no-lick
    # panels are undefined at that alignment for the same reason. Both guards mirror the one the
    # CONDITIONS loop above has always had ("no lick, no alignment point"); without them the record
    # gained a post-lick entry whose rows were 63-99% no-lick trials (PS94_0819).
    for align, cond in (("cue", "post-cue"), ("precue", "pre-cue"), ("lick", "post-lick")):
        d = d_by_align.get(align)
        if d is None or (align == "lick" and arm_all):
            continue
        dd = dict(d)
        dd["post_i"] = {session}
        try:
            rec.setdefault("confusion", {})[cond] = pc.crossed_confusion(
                dd, post_all_trials=arm_all, include_nolick=(align != "lick"))
        except Exception as ex:                                           # noqa: BLE001
            print(f"      {cond} confusion failed ({str(ex)[:60]})", flush=True)
    return rec


def _nolick_readouts(d_by_align, session, keep):
    """The arm-independent half: the three readouts that examine the NO-LICK trials on purpose."""
    out = {}
    dp = d_by_align.get("precue")
    if dp is not None:
        dd = dict(dp)
        dd["post_i"] = {session}
        try:
            out["looks_like_which"] = pc.looks_like_which(dd, keep)
        except Exception as ex:                                           # noqa: BLE001
            print(f"      looks_like_which failed ({str(ex)[:60]})", flush=True)
    # G4b at BOTH alignments (Priya, 2026-08-20). Only the pre-cue record was stored, so the
    # cue-aligned slide had no producer and sat on a superseded scratchpad figure.
    for align, key in (("precue", "fits_engaged_precue"), ("cue", "fits_engaged_cue")):
        d = d_by_align.get(align)
        if d is None:
            continue
        dd = dict(d)
        dd["post_i"] = {session}
        try:
            out[key] = pc.fits_engaged_distribution(dd, keep)
        except Exception as ex:                                           # noqa: BLE001
            print(f"      {key} failed ({str(ex)[:60]})", flush=True)
    # G4 CAMERA-CLASS DISCRIMINATOR (Priya 11-04, follow-on a): the PRIMARY G4 readout under the camera grouping
    # (`poststroke_compare.g4_camclass`: working vs raw no_response (Priya 11-09, as G1-G3; unknown never trained),
    # position-balanced, jaw_only / quit / no_response / unknown scored separately, z frame
    # on contact rows by default); the DAQ readouts above stay as the labelled robustness arm (WIDEFIELD_G4_DISC=daq
    # makes them primary again)
    out["g4_primary"] = pc.g4_disc()
    for align in ("precue", "cue"):
        d = d_by_align.get(align)
        if d is None or pc.g4_disc() != "camera":
            continue
        dd = dict(d)
        dd["post_i"] = {session}
        try:
            got = pc.g4_camclass(dd, keep)
            if got is not None:
                out[f"g4_camclass_{align}"] = got
        except Exception as ex:                                           # noqa: BLE001
            from wfield_local import trial_classes as _tcx
            _tcx.reraise_under_camera(ex, f"g4_camclass[{align}]")
            print(f"      g4_camclass[{align}] failed ({str(ex)[:60]})", flush=True)
    for align in ("precue", "cue"):
        d = d_by_align.get(align)
        if d is None:
            continue
        dd = dict(d)
        dd["post_i"] = {session}
        try:
            out.setdefault("impaired_nolick", {})[align] = pc.impaired_nolick_readout(
                dd, keep, alignment=align)
        except Exception as ex:                                           # noqa: BLE001
            print(f"      impaired_nolick[{align}] failed ({str(ex)[:60]})", flush=True)
    return out


def lick_keep(d_by_align, session, pres):
    """The lick-only arm's positions at the LICK alignment. Post-lick aligns to a DAQ lick, so it has engaged rows
    only: when the camera-working rule chose ``pres["keep"]`` (Priya 11-07), a position also needs >= MIN_POST
    DAQ-engaged rows of this session in the lick pool (else the cue pool) -- never scored on positions without
    lick-aligned rows. Under the DAQ rule ``keep`` is already that count: unchanged."""
    keep = pres["keep"]
    if pres["rule"] != "camera_working":
        return list(keep)
    dl = d_by_align.get("lick") or d_by_align.get("cue")
    n = pc._engaged_counts(dl, [session])
    return [c for c in keep if n[c] >= pc.MIN_POST]


def run_session(d_by_align, animal, session, label, phase_tag):
    """Everything Section G shows for ONE session: both arms plus the arm-independent readouts."""
    d0 = d_by_align.get("cue")
    if d0 is None:
        return None
    pres = pc.preserved_detail(d0, session)
    keep = pres["keep"]
    keep_lick = lick_keep(d_by_align, session, pres)
    # Priya 11-07: which rule chose the positions and the counts it used (camera WORKING rows, not successful licks,
    # under the camera grouping); the post-lick arm's own (DAQ-lick) set beside it
    rec = {"animal": animal, "label": label, "phase_tag": phase_tag,
           "preserved_positions": [POSITION_NAMES[c] for c in keep],
           "preserved_rule": pres["rule"], "preserved_counts": pres["counts"], "preserved_min_n": pres["min_post"],
           "preserved_positions_lick": [POSITION_NAMES[c] for c in keep_lick],
           "counts": _counts(d0, session), "arms": {}}
    for arm in ARMS:
        print(f"    {label} [{arm}]", flush=True)
        rec["arms"][arm] = _one_arm(d_by_align, session, keep, arm_all=(arm == "all"), keep_lick=keep_lick)
    rec.update(_nolick_readouts(d_by_align, session, keep))
    return rec


def collect_animal(an, include_excluded=True):
    """One animal's sessions, serially. Kept as the `--jobs 1` path and as the unit test seam."""
    out = {}
    for tag, labels in (("post", None),
                        ("excluded", pc.excluded_labels(an) if include_excluded else None)):
        if tag == "excluded" and not labels:
            continue
        print(f"\n##### {an} [{tag}]", flush=True)
        d_by_align = _pool_all(an, labels)
        d0 = d_by_align.get("cue")
        if d0 is None:
            continue
        for i in sorted(d0["post_i"]):
            rec = run_session(d_by_align, an, i, d0["kept"][i], tag)
            if rec:
                out[rec["label"]] = rec
    return out


def _pool_all(an, labels):
    """The three pooled bundles one session needs. Disk-cached, so a worker re-reads rather than
    recomputes -- which is what makes the SESSION affordable as a unit."""
    d = {}
    for align, _ in CONDITIONS:
        try:
            d[align] = pc._pooled(an, align, post_labels=labels)
        except Exception as ex:                                   # noqa: BLE001
            print(f"  {an} {align}: pool failed ({str(ex)[:60]})", flush=True)
    return d


def _session_worker(spec):
    """One session. Rebuilds its animal's bundles from the disk cache rather than receiving them.

    THE BUNDLES ARE NOT PICKLED ACROSS THE PROCESS BOUNDARY. `XE` alone is thousands of trials by
    hundreds of features; shipping three of those to each of ~30 workers would cost more than the
    analysis. `pool_sessions` is `session_cache`-backed, so the worker pays a read instead.
    """
    an, tag, labels, i = spec
    d_by_align = _pool_all(an, labels)
    d0 = d_by_align.get("cue")
    if d0 is None:
        return None
    return run_session(d_by_align, an, i, d0["kept"][i], tag)


def _warm_one(spec):
    """Pool BOTH tags of one (animal, alignment) and force its frozen model to exist.

    THE UNIT IS (animal, alignment) AND BOTH TAGS LIVE INSIDE IT, which is what makes this phase
    parallelisable at all. `frozen_models.make_spec` keys on animal and alignment but NOT on the
    comparison arm, so `post` and `excluded` at one alignment are the SAME stored model: split them
    across workers and both miss, both fit, both write. Different (animal, alignment) pairs are
    different specs and cannot collide.

    `pc.frozen(d)` IS THE WARM, not `pc._pooled`. The model is loaded lazily into the bundle's
    `_frozen_cache` on first USE -- inside `run_session`, not at pool time -- so the first version
    of this pass pooled everything and warmed nothing, leaving every session worker still racing to
    fit the same model. Pooling is not a pre-warm; forcing the lookup is.

    Returns the cue-alignment `(kept, post_i)` so the caller can name its session units without
    pooling a second time.
    """
    an, align, include_excluded = spec
    out = {}
    for tag, labels in (("post", None),
                        ("excluded", pc.excluded_labels(an) if include_excluded else None)):
        if tag == "excluded" and not labels:
            continue
        try:
            d = pc._pooled(an, align, post_labels=labels)
        except Exception as ex:                                   # noqa: BLE001
            print(f"  {an} {align} [{tag}]: pool failed ({str(ex)[:60]})", flush=True)
            continue
        if d is None:
            continue
        pc.frozen(d)                     # force the fit/load HERE, one worker per spec
        if align == "cue":
            out[tag] = (list(d["kept"]), sorted(d["post_i"]), labels)
    return out


def collect(animals=None, include_excluded=True, jobs=None):
    """Every post-stroke session, plus the excluded ones tagged as such. Two parallel phases.

    PHASE 1 -- (animal, alignment), both tags inside each unit. Pools, and forces the frozen decoder
    to exist via `pc.frozen`, because the model loads LAZILY on first use and pooling alone warms
    nothing. Measured 2026-08-28: ~12.5 min for ONE animal, so run serially it dominated the stage
    at ~50 min against phase 2's ~15 -- the phase that was already parallel was not the expensive
    one.

    PHASE 2 -- the SESSION, ~30 units, every frozen lookup now a read rather than a fit.

    A failed unit RAISES rather than yielding a short file: a `section_g.json` missing a session is
    indistinguishable from a session that legitimately produced nothing.
    """
    from wfield_local import parallel

    warm_units = [(an, align, include_excluded)
                  for an in post_animals(animals) for align, _ in CONDITIONS]
    warmed, wfail = parallel.fan_out(warm_units, _warm_one, jobs=jobs, label="pool")
    warmed = sorted(warmed, key=lambda kv: repr(kv[0]))  # completion order -> unit order (T10)
    if wfail:
        raise RuntimeError(f"section G phase 1 failed for {[u for u, _ in wfail]}: {wfail[0][1]}")

    units = []
    for (an, align, _inc), res in warmed:
        if align != "cue":
            continue
        for tag, (_kept, post_i, labels) in sorted(res.items()):
            units += [(an, tag, labels, i) for i in post_i]
    units.sort()

    print(f"\n  phase 1: {len(warm_units)} (animal, alignment) pool(s) warmed; "
          f"phase 2: {len(units)} session unit(s)", flush=True)
    results, failures = parallel.fan_out(units, _session_worker, jobs=jobs, label="session")
    if failures:
        raise RuntimeError(f"section G failed for {[u for u, _ in failures]}: {failures[0][1]} "
                           f"-- refusing to write a partial section_g.json")
    # **SORTED, NOT COMPLETION ORDER** (CLAUDE.md ground rule 9). `section_g_figures` re-sorts on
    # read so no panel depends on this, but the dict is dumped straight to `section_g.json` and
    # completion order permuted that file's keys on every run -- a nightly artefact that diffs
    # against itself is a nightly artefact nobody can check.
    out = {}
    for _u, rec in sorted(results, key=lambda kv: kv[0]):
        if rec:
            out[rec["label"]] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--skip-excluded", action="store_true",
                    help="omit the excluded (small-lesion / before-after control) sessions")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--jobs", "-j", type=int, default=None, metavar="N",
                    help="animals to process in parallel (default: cores-2 capped at 8, and at "
                         "the cohort size; pass 1 for the serial path)")
    args = ap.parse_args(argv)
    from wfield_local import run_mode  # tagged runs never overwrite default figures (review H2)
    out = run_mode.routed_output(args.output or Path(PathResolver().root("figures_working")))
    rec = collect(args.animals, include_excluded=not args.skip_excluded, jobs=args.jobs)
    from wfield_local import run_mode   # mode-tagged name (rebuild 11-03, trap T21): read by grant 3b / 5
    p = Path(out) / run_mode.keyed_name("section_g.json")
    json.dump(rec, open(p, "w"), indent=1, default=float)
    # review 7A: the design sidecar (scored = engaged / all trials, model = working-trained under the camera gate)
    from wfield_local import trial_classes as _tcg
    if _tcg.camera_gate_on():
        from wfield_local import poststroke_compare as _pcg
        from wfield_local import spout_scaling as _ssg
        json.dump({"scored_rows": (f"camera WORKING rows ({_tcg.sel_class_text()}) for the pre panel / pre band and the "
                                   "engaged-only post arm, all trials for the all-trials arm; by_class panels per "
                                   "camera group (Priya 11-04, main 4a37cbc8)"
                                   if _pcg.section_g_groups() == "camera" else
                                   "engaged / all trials (DAQ grouping, WIDEFIELD_SECTION_G_GROUPS=daq)"),
                   "groups": _pcg.section_g_groups(), "scored_centre": _ssg.score_centre_tag(),
                   "state_disc_frame": _pcg._state_frame_tag(),
                   "model_trials": _tcg.run_selection(), "design": _tcg.design_note(),
                   "working_excluded_sessions": _tcg.working_excluded_sessions()},
                  open(Path(out) / run_mode.keyed_name("section_g_design.json"), "w"), indent=1)
    n_post = sum(1 for r in rec.values() if r["phase_tag"] == "post")
    print(f"\nwrote {p}  ({len(rec)} sessions: {n_post} post, {len(rec) - n_post} excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

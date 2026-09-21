"""Smoke-test `restw` on real sessions BEFORE any render depends on it.

WHY THIS EXISTS AS A SCRIPT AND NOT AS A GLANCE AT THE CODE. Every silent failure this repo hit on
2026-09-13 rendered cleanly: a missing `_REF_TEXT` key dropped a third of the 15r figures while the
render printed "0 failed"; a self-subtracting pre-cue reference produced a complete, plausible,
wrong figure. A new reference cannot be trusted because it imports -- it has to be shown to produce
a baseline, to produce maps, and to produce maps that DIFFER from the reference it is supposed to
improve on.

THE THREE THINGS CHECKED, in order of what would be worst to get wrong:

1. THE BASELINE EXISTS and how many positions contributed. Fewer than four and the session has no
   RESTW column at all -- deliberately, see `MIN_POSITIONS_FOR_WEIGHTED`.
2. RESTW DIFFERS FROM REST. If the two baselines were identical the reference would be a no-op
   wearing a new name, which is the failure mode a caption cannot catch. Reported as the RMS of
   the difference against the RMS of the rest baseline itself, so it is readable as a fraction.
3. THE MAPS COME OUT, per position, with trial counts matching the REST column's -- the two
   references must be built on the SAME trials, or a difference between them is the trial set.

RUN:  python -m scripts.rest_migration.archive.restw_smoke [--limit 3] [--align cue]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main() -> int:
    from wfield_local import config, joint_basis
    from wfield_local import position_reference_maps as prm
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.rest_by_position import rest_frames_by_position

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3)
    # THE POST-STROKE SESSIONS ARE THE TEST, not the pre-stroke ones. `restw` is DESIGNED to be a
    # near-no-op pre-stroke -- with all six positions contributing rest equally the weighting
    # changes little -- and to bite when the animal stops attempting the far positions. A smoke
    # test that only ever sees pre-stroke sessions would report "barely differs" and be read as
    # "does not work".
    ap.add_argument("--labels", nargs="*", default=None, help="session labels, e.g. PS93_0814")
    ap.add_argument("--align", default="cue")
    ap.add_argument("--post-s", type=float, default=2.0)
    a = ap.parse_args()

    print(f"REFERENCES = {prm.REFERENCES}")
    if prm.REST_WEIGHTED not in prm.REFERENCES:
        print("!! restw is NOT registered -- the builder and the registry must land together")
        return 1
    # THE CAPTION TABLE IS PART OF THE CONTRACT, checked here rather than discovered mid-render:
    # a missing key raises per-arm AFTER earlier references are written, and the render exits 0.
    from wfield_local.epoch_grant_figures import _REF_TEXT
    missing = [r for r in prm.REFERENCES if r not in _REF_TEXT]
    if missing:
        print(f"!! _REF_TEXT has no entry for {missing} -- this is exactly how `precue` was lost")
        return 1
    print(f"_REF_TEXT covers every reference; restw tag = {prm.reference_tag('restw')!r}, "
          f"rest tag = {prm.reference_tag('rest')!r}")

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    if a.labels:
        want = set(a.labels)
        todo = [x for x in SESSIONS if x["label"] in want and x.get("h5")]
        got = {x["label"] for x in todo}
        for miss in sorted(want - got):
            print(f"!! no session {miss} with an h5 -- asked for it, not tested")
    else:
        todo = [x for x in SESSIONS if x["label"] in want and x.get("h5")][: a.limit]
    t0, ok = time.time(), 0
    for s in todo:
        lab = s["label"]
        print(f"\n=== {lab} " + "=" * (60 - len(lab)))
        _u, v = joint_basis._load_session(s["mc"])
        V = np.asarray(v)

        per, info = rest_frames_by_position(s, V.shape[1])
        if info.get("error"):
            print(f"  rest-by-position FAILED: {info['error']}")
            continue
        print(f"  {info['n_periods']} labelled rest periods; frames per position: "
              + ", ".join(f"{c}:{len(f)}" for c, f in sorted(per.items())))

        bw, used = prm.session_restw_svt(s, V)
        if bw is None:
            print(f"  NO RESTW BASELINE (positions with a usable baseline: {used}) -- "
                  f"this session would lose its RESTW column only")
            continue
        br = prm.session_rest_svt_timelocal(s, V)
        print(f"  restw baseline OK, {len(used)} positions contributed: {used}")
        if br is not None:
            d = float(np.sqrt(np.mean((np.asarray(bw) - np.asarray(br)) ** 2)))
            r = float(np.sqrt(np.mean(np.asarray(br) ** 2)))
            print(f"  restw vs rest: RMS difference {d:.6f} against rest RMS {r:.6f} "
                  f"({100 * d / max(1e-12, r):.1f}%)")
            if d < 1e-12:
                print("  !! IDENTICAL to the rest baseline -- restw would be a no-op")

        parts = prm.session_raw_maps(s, a.align, post_s=a.post_s, variant="working")
        mr = prm.reference_maps(parts, "rest")
        mw = prm.reference_maps(parts, prm.REST_WEIGHTED)
        print(f"  maps: rest {len(mr)} positions, restw {len(mw)} positions")
        if set(mr) != set(mw):
            print(f"  !! DIFFERENT POSITION SETS -- rest {sorted(mr)} vs restw {sorted(mw)}; "
                  f"a difference between the references would be the trial set")
        if parts.get("used_rest") and parts.get("used_restw"):
            bad = {q for q in set(parts["used_rest"]) & set(parts["used_restw"])
                   if parts["used_rest"][q] != parts["used_restw"][q]}
            print(f"  trial counts agree" if not bad else
                  f"  !! trial counts DIFFER at {sorted(bad)} -- the references are not "
                  f"built on the same trials")
        for q in sorted(set(mr) & set(mw)):
            a_, b_ = np.asarray(mr[q]), np.asarray(mw[q])
            fin = np.isfinite(a_) & np.isfinite(b_)
            print(f"    {q:<12} rest RMS {np.sqrt(np.mean(a_[fin] ** 2)):.5f}   "
                  f"restw RMS {np.sqrt(np.mean(b_[fin] ** 2)):.5f}   "
                  f"corr {np.corrcoef(a_[fin], b_[fin])[0, 1]:.4f}")
        ok += 1

    print(f"\n{ok}/{len(todo)} session(s) produced a restw column  [{time.time() - t0:.0f}s]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

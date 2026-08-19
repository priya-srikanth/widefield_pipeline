"""Compute every Section G quantity from the CONFIG-DERIVED post-stroke pool, per session, both arms.

WHY THIS EXISTS AS A MODULE. Section G's figures were built by two scripts that lived only in a
scratchpad directory, and one of them opened with

    for an in ("PS94", "PS95"):
        ...
    json.dump(res, open("E:/cue_lick/poststroke_matched_0817.json", "w"))

That was correct on 2026-08-18, when PS94 and PS95 were the only lesioned animals and 8/17 was the
only post-stroke date. It was wrong the next morning, and nothing said so: PS92 and PS93 were simply
absent from every figure downstream, and every 8/18 session was invisible. The same shape had already
appeared in `evoked_amplitude` and `spatial_reorganisation` (see config.analysis_sessions). A deck
that is regenerated from unversioned scripts cannot be checked, so the scripts move in here.

WHAT IS COMPUTED, AND WHY PER SESSION. Earlier runs pooled every post-stroke session of an animal
into one "post" column. With one post-stroke night that was invisible; with two it silently averages
a day-1 and a day-2 brain, and for PS94 those differ more than pre differs from post (pre-cue z=-0.2
on 8/17 against -3.4 on 8/18). Everything here is therefore keyed by SESSION LABEL, and the pooled
form is not offered.

BOTH ARMS, ALWAYS.
  ALL       every cued trial, scored over all six positions. Chance is 1/6 for every session and
            animal, so this is the ONLY arm comparable across sessions -- and the missing licks are
            the phenotype, so it is also the arm that can see the abandoned positions at all.
  LICK-ONLY that session's preserved positions, chance 1/n. Forced, not chosen: a position with no
            engaged trials cannot be decoded from engaged trials. Its chance level moves with the
            animal's behaviour (PS95: 4 positions on 8/17, 6 on 8/18), so its accuracies must NEVER
            be laid side by side across sessions.
The DIFFERENCE between the arms is the point: it separates "the code degraded" from "the code is
fine whenever the animal manages to lick".

Post-lick has ONE arm by construction. It aligns to the first lick, so a trial without one has no
reference point; its post arm is lick-only because no other arm exists.
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

#: (alignment, the name the deck's figures use for it)
CONDITIONS = (("cue", "post-cue"), ("lick", "post-lick"), ("precue", "pre-cue"))


def post_animals(animals=None):
    """Animals with at least one post-stroke session, from the phase resolver rather than a literal."""
    if animals:
        return sorted(animals)
    return sorted({config.animal_of(lab) for lab in config.phase_labels("post")})


def _counts(d, session):
    """Engaged / undetected trials per position: the post SESSION against the pre-session mean.

    Behaviour is reported before any decoding number because it bounds what one can mean. A position
    with zero engaged trials has no lick-only decoding number at all, and the first version of this
    analysis reported a PS94 "neural deficit" whose larger part was that fact.
    """
    pos = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    out = {"pre": {"engaged": {}, "undetected": {}}, "post": {"engaged": {}, "undetected": {}}}
    npre = max(len(d["pre_i"]), 1)
    for arm, Y, G in (("engaged", d["YE"], d["GE"]), ("undetected", d["YU"], d["GU"])):
        for c, p in zip(DISPLAY_ORDER, pos):
            pre_m = np.isin(G, list(d["pre_i"])) & (Y == c)
            out["pre"][arm][p] = float(pre_m.sum()) / npre          # per-session mean
            out["post"][arm][p] = int((np.isin(G, [session]) & (Y == c)).sum())
    return out


def run_session(d_by_align, animal, session, label, arm_all=True):
    """Every Section G quantity for ONE post-stroke session on ONE arm."""
    d0 = d_by_align.get("cue")
    if d0 is None:
        return None
    keep = pc.preserved_positions(d0, session=session)
    rec = {"animal": animal, "label": label,
           "arm": "ALL trials" if arm_all else "lick-only",
           "preserved_positions": [POSITION_NAMES[c] for c in keep],
           "n_positions_scored": len(DISPLAY_ORDER) if arm_all else len(keep),
           "chance": (1.0 / len(DISPLAY_ORDER)) if arm_all else (1.0 / max(len(keep), 1)),
           "counts": _counts(d0, session)}
    if len(keep) < 2 and not arm_all:
        rec["note"] = (f"only {len(keep)} preserved position(s): the lick-only arm does not exist "
                       f"for this session")
        return rec

    for align, cond in CONDITIONS:
        d = d_by_align.get(align)
        if d is None:
            continue
        dd = dict(d)
        dd["post_i"] = {session}
        # post-lick has no all-trials arm -- no lick, no alignment point. Skipping it here rather
        # than letting it fall through keeps the arm label on the record honest.
        if align == "lick" and arm_all:
            continue
        try:
            r = pc.decode_matched(dd, keep, post_all_trials=arm_all)
        except Exception as ex:                                          # noqa: BLE001
            print(f"    {label} {cond}: decode_matched failed ({str(ex)[:60]})", flush=True)
            continue
        if r:
            rec[cond] = r
            b = r.get("pre_band") or {}
            print(f"    {label:11s} {cond:9s} acc={r.get('accuracy', float('nan')):.3f} "
                  f"bal={r.get('balanced_accuracy', float('nan')):.3f}  "
                  f"pre {b.get('mean', float('nan')):.3f}"
                  f"[{b.get('min', float('nan')):.3f},{b.get('max', float('nan')):.3f}]  "
                  f"n_post={r.get('n_post', 0)}", flush=True)

    # RSM / pattern similarity and the crossed confusion, both on the post-cue window
    try:
        rec["postcue_pattern_similarity"] = pc.pattern_similarity(d0, keep, post_all_trials=arm_all)
    except Exception as ex:                                              # noqa: BLE001
        print(f"    {label} pattern_similarity failed ({str(ex)[:60]})", flush=True)
    for align, cond in (("cue", "post-cue"), ("precue", "pre-cue")):
        d = d_by_align.get(align)
        if d is None:
            continue
        dd = dict(d)
        dd["post_i"] = {session}
        try:
            rec.setdefault("confusion", {})[cond] = pc.crossed_confusion(
                dd, post_all_trials=arm_all)
        except Exception as ex:                                          # noqa: BLE001
            print(f"    {label} {cond} confusion failed ({str(ex)[:60]})", flush=True)
    return rec


def collect(animals=None, arms=("all", "lickonly")):
    out = {}
    for an in post_animals(animals):
        print(f"\n##### {an}", flush=True)
        d_by_align = {}
        for align, _ in CONDITIONS:
            try:
                d_by_align[align] = pc._pooled(an, align)
            except Exception as ex:                                      # noqa: BLE001
                print(f"  {an} {align}: pool failed ({str(ex)[:60]})", flush=True)
        d0 = d_by_align.get("cue")
        if d0 is None:
            continue
        for i in sorted(d0["post_i"]):
            label = d0["kept"][i]
            for arm in arms:
                print(f"  -- {label}  [{arm}]", flush=True)
                rec = run_session(d_by_align, an, i, label, arm_all=(arm == "all"))
                if rec:
                    out.setdefault(arm, {})[label] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--arm", nargs="+", default=["all", "lickonly"], choices=["all", "lickonly"])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    res = collect(args.animals, arms=tuple(args.arm))
    for arm, rec in res.items():
        p = Path(out) / f"section_g_{arm}.json"
        json.dump(rec, open(p, "w"), indent=1, default=float)
        print(f"wrote {p}  ({len(rec)} sessions)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

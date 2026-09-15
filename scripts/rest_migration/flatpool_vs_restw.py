"""Does POSITION-WEIGHTING earn its place, with the TEMPORAL axis held fixed?

THE TEST THAT `rest_vs_restw` FAILED TO BE. That script compared `raw_rest` against `raw_restw` and
returned a clean null, and it was withdrawn on 2026-09-14 because the two arms differ on BOTH axes:
`raw_rest` comes from `session_rest_svt_timelocal`, which bins the session and interpolates -- it is
TIME-LOCAL pooled -- while `raw_restw` is FLAT position-weighted. A null across two changed variables
says only that two quite different subtrahends agree; it cannot isolate either one.

THIS HOLDS THE TEMPORAL AXIS FIXED. Both arms are FLAT:

    flatpool   raw - quiet_flat     one value per session, pooled over ALL rest frames
    restw      raw_restw            one value per session, six per-position medians averaged EQUALLY

so the ONLY thing that differs is COMPOSITION, which is the question.

WHY IT MATTERS (Priya, 2026-09-14): pooling weights each FRAME equally, therefore each POSITION by how
many rest frames it happens to have -- and per-position ITI survival spans 17.5%. Since rest CARRIES
position (observed/null 1.449, 4/4 animals), a pooled baseline is pulled toward the positions that
supplied the most rest, so those positions have more of their OWN rest subtracted from their own map.
That is a position-dependent distortion of exactly the contrast the maps measure.

AGAINST THAT: the imbalance is FIXED at 17.5% and stable across epochs -- it does NOT grow with the
deficit -- and equal weighting costs ~3% added variance plus the 200-frame floor, the 4-of-6 rule and
the column-drop apparatus. So it is a statable bias versus real machinery, and only a result can
decide.

RESIDUAL IMPERFECTION, stated rather than hidden: `quiet_flat` is a MEAN over rest frames while
`restw` averages MEDIANS, so a mean/median difference remains. It is third-order next to the temporal
confound this replaces -- rest frames are engagement-gated and exclude licking, running and the
session-onset transient, so heavy-tailed contamination is largely already removed -- but it is not
zero, and a fully clean version would pool a MEDIAN.

RUN:  python -m scripts.rest_migration.flatpool_vs_restw [--align cue] [--limit 0]
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time

import numpy as np


def main() -> int:
    from wfield_local import beta_maps as bm
    from wfield_local import config
    from wfield_local import epoch_figures as ef
    from wfield_local import position_reference_maps as prm
    from wfield_local.grant_figures import ANIMALS, CONF_LABELS, _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    ap = argparse.ArgumentParser()
    ap.add_argument("--align", default="cue")
    ap.add_argument("--post-s", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    mask = bm.stat_mask()
    t0 = time.time()
    # {kind: {animal: {epoch: {position: [map, ...]}}}}
    store = {"flatpool": {}, "restw": {}}
    n_ok = n_skip = 0
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [x for x in SESSIONS if x["label"] in want]
    if a.limit:
        todo = todo[: a.limit]

    for s in todo:
        lab = s["label"]
        an = lab.split("_")[0]
        if an not in ANIMALS:
            continue
        d = _day(an, lab.split("_")[-1])
        if d is None:
            continue
        e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
        if e is None:
            continue
        try:
            parts = prm.session_raw_maps(s, a.align, post_s=a.post_s, variant="working")
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            n_skip += 1
            continue
        raw, qflat, rrestw = parts.get("raw"), parts.get("quiet_flat"), parts.get("raw_restw")
        if not raw or qflat is None or not rrestw:
            # A session WITHOUT a restw column cannot appear in EITHER arm, or the two arms would be
            # built on different session sets and any difference between them would be the sessions.
            n_skip += 1
            continue
        for qname, m in raw.items():
            # FLAT POOLED: subtraction commutes with averaging, so raw - quiet_flat IS the
            # flat-pooled-referenced map, from the same call. No second feature build.
            store["flatpool"].setdefault(an, {}).setdefault(e, {}).setdefault(
                qname, []).append(np.asarray(m) - np.asarray(qflat))
        for qname, m in rrestw.items():
            store["restw"].setdefault(an, {}).setdefault(e, {}).setdefault(
                qname, []).append(np.asarray(m))
        n_ok += 1
        print(f"  .. {lab} [{e}]  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{n_ok} sessions, {n_skip} skipped")
    if n_ok < 8:
        print("TOO FEW -- a failed run, not a negative result.")
        return 1

    order = [e for e in ("pre", "acute", "subacute", "chronic")]

    def pooled(kind, e, qname):
        """Mean over ANIMALS of each animal's epoch mean -- the deck's pooling rule."""
        per = []
        for an in store[kind]:
            v = ((store[kind][an].get(e) or {}).get(qname) or [])
            if v:
                per.append(np.mean(v, axis=0))
        return np.mean(per, axis=0) if per else None

    print(f"\n{'=' * 78}\n1. PER-POSITION AMPLITUDE (RMS inside the stat mask), and acute/pre\n"
          f"{'=' * 78}")
    for kind in ("flatpool", "restw"):
        print(f"\n--- {kind} ---")
        print(f"{'position':<14}" + "".join(f"{e:>11}" for e in order) + f"{'acute/pre':>11}")
        for qname in CONF_LABELS:
            cells, amps = [], {}
            for e in order:
                m = pooled(kind, e, qname)
                if m is None:
                    cells.append(f"{'--':>11}")
                    continue
                v = np.asarray(m)[mask]
                amps[e] = float(np.sqrt(np.mean(v[np.isfinite(v)] ** 2)))
                cells.append(f"{amps[e]:>11.5f}")
            r = (amps.get("acute", np.nan) / amps["pre"]) if amps.get("pre") else np.nan
            print(f"{qname:<14}" + "".join(cells) + f"{r:>11.3f}")

    print(f"\n{'=' * 78}\n2. BETWEEN-ANIMAL AGREEMENT, (epoch - pre), far-contralateral\n"
          f"{'=' * 78}")
    fc = CONF_LABELS[-1]
    print(f"far-contra label = {fc!r}")
    # THE KEY IS `flatpool`, NOT `rest`. This script was adapted from `flat_vs_timelocal`, whose
    # store key was "rest"; the rename was missed here and only here, so section 1 printed fine and
    # section 2 -- the STRONGER claim -- died with a KeyError after 880 s of collection.
    print(f"{'epoch':<12}{'flatpool':>12}{'restw':>12}{'difference':>13}")
    for e in ("acute", "subacute", "chronic"):
        out = {}
        for kind in ("flatpool", "restw"):
            per = {}
            for an in store[kind]:
                pre = ((store[kind][an].get("pre") or {}).get(fc) or [])
                ep_ = ((store[kind][an].get(e) or {}).get(fc) or [])
                if pre and ep_:
                    per[an] = np.mean(ep_, axis=0) - np.mean(pre, axis=0)
            rs = []
            for x, y in itertools.combinations(sorted(per), 2):
                u, w = np.asarray(per[x])[mask], np.asarray(per[y])[mask]
                ok = np.isfinite(u) & np.isfinite(w)
                if ok.sum() > 50:
                    rs.append(float(np.corrcoef(u[ok], w[ok])[0, 1]))
            out[kind] = float(np.mean(rs)) if rs else np.nan
        print(f"{e:<12}{out['flatpool']:>12.3f}{out['restw']:>12.3f}"
              f"{out['restw'] - out['flatpool']:>13.3f}")

    # THIS LEGEND WAS INHERITED FROM `flat_vs_timelocal` AND SAID "time-local" -- it described the
    # comparison this script was ADAPTED FROM, not the one it runs, so the output mislabelled its own
    # conclusion. Caught by reading the printed result rather than the numbers above it.
    print("\nHOW TO READ IT:")
    print("  ORDER and the acute between-animal r agree  ->  POSITION-WEIGHTING is NOT earning its")
    print("      place. Drop `restw` for plain `rest`, and the 200-frame floor, the 4-position rule")
    print("      and the column-drop apparatus go with it.")
    print("  amplitude ORDER or the acute between-animal r MOVES  ->  position-weighting is")
    print("      load-bearing and the machinery built on it is warranted.")
    print("  NB a uniform shift in EVERY ratio is what removing a FIXED composition bias looks like")
    print("      (the imbalance is 17.5% and stable across epochs). It is not a changed conclusion;")
    print("      only a change in ORDER or in the between-animal agreement would be.")
    print(f"\n[done in {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Does the TIME-LOCAL rest baseline change any CONCLUSION, or only the third decimal?

Priya, 2026-09-13/14, after being told the polynomial detrend is enough for the decoder:
*"but shouldn't that be enough for the 'rest' detrend too then?"*

THE INFERENCE DOES NOT CARRY, BUT THE QUESTION BEHIND IT WAS NEVER ANSWERED. The decoder is
invariant to drift -- per-fold standardisation ignores a shared additive offset, and it reports a
discrimination, not an amplitude. A map reference reports an amplitude, and an error in the
subtrahend lands in it directly; worse, drift is NOT common-mode across positions, because each
position's trials cluster in its own blocks and so sample drift at its own times.

SO "harmless to the decoder" and "harmless to the baseline" genuinely come apart. What has NOT been
shown is that the residual CHANGES A CONCLUSION. `timelocal_needed` established that slow structure
EXISTS after the polynomial detrend (7.8x a size-preserving shuffle, 4.05x a position-composition
control). Existence is not consequence, and the whole sweep-binning / carry-forward / survival
apparatus is justified ONLY by time-local being necessary:

    position-weighting   justified INDEPENDENTLY (rest carries position; composition tracks the
                         deficit post-stroke). Applies to a flat baseline too. Survives either way.
    time-local           justified only if it moves a result. THIS SCRIPT.
    sweep bins, trailing-chunk carry-forward, reach-back, the survival analysis
                         contingent ENTIRELY on time-local. If time-local goes, they all go.

THE COMPARISON IS ALMOST FREE, which is why it should have been run before any of that machinery.
A FLAT baseline is one map subtracted from an averaged map, and subtraction commutes with
averaging -- so `raw[q] - quiet_flat` IS the flat-referenced map, available from the SAME
`session_raw_maps` call that returns the time-local `raw_rest[q]`. No second feature build.

WHAT IS COMPARED, chosen because these are the claims the REST reference actually carries:

1. PER-POSITION AMPLITUDE BY EPOCH, and the acute/pre ratio. This is the graded-deficit claim
   ("1.48/1.21/1.27 near -> 1.01/0.69/0.48 far"). It is an amplitude, so it is where a subtrahend
   error shows up most directly.
2. BETWEEN-ANIMAL AGREEMENT at far-contralateral, acute minus pre -- the observed side of the
   cross-position null, and the strongest anatomical claim in the deck.

READ IT AS: if both are unchanged beyond rounding, time-local is not earning its place, and a FLAT
position-weighted baseline is the better definition -- fewer moving parts, no bin-coverage problem,
no trailing-chunk rule, and the survival concern largely evaporates.

RUN:  python -m scripts.rest_migration.flat_vs_timelocal [--align cue] [--limit 0]
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
    store = {"flat": {}, "timelocal": {}}
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
        raw, qflat, rrest = parts.get("raw"), parts.get("quiet_flat"), parts.get("raw_rest")
        if not raw or qflat is None or not rrest:
            n_skip += 1
            continue
        for qname, m in raw.items():
            # FLAT: one map subtracted from the averaged map. This IS the retired construction.
            store["flat"].setdefault(an, {}).setdefault(e, {}).setdefault(qname, []).append(
                np.asarray(m) - np.asarray(qflat))
        for qname, m in rrest.items():
            store["timelocal"].setdefault(an, {}).setdefault(e, {}).setdefault(
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
    for kind in ("flat", "timelocal"):
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
    print(f"{'epoch':<12}{'flat':>12}{'timelocal':>12}{'difference':>13}")
    for e in ("acute", "subacute", "chronic"):
        out = {}
        for kind in ("flat", "timelocal"):
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
        print(f"{e:<12}{out['flat']:>12.3f}{out['timelocal']:>12.3f}"
              f"{out['timelocal'] - out['flat']:>13.3f}")

    print("\nHOW TO READ IT:")
    print("  both columns agree to ~the third decimal  ->  time-local is NOT earning its place.")
    print("      Drop it, drop sweep-binning, drop the trailing-chunk and reach-back rules, and")
    print("      keep a FLAT POSITION-WEIGHTED baseline -- whose justification is independent.")
    print("  amplitude ratios or between-animal r move  ->  time-local is load-bearing and the")
    print("      machinery built on it is warranted.")
    print(f"\n[done in {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

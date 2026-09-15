"""Does POSITION-WEIGHTING the rest baseline change any CONCLUSION, or only the third decimal?

Priya, 2026-09-14: *"Are we overcomplicating things with the restw?"* and *"I worry normalizing to
thin data will do more harm than good."*

THE SAME QUESTION THAT RETIRED TIME-LOCAL, ASKED OF THE OTHER HALF. `flat_vs_timelocal` established
that the TEMPORAL half of the old `restw` earned nothing and dropped it. The COMPOSITION half --
averaging six per-position medians EQUALLY instead of pooling all rest frames -- survived on the
argument that `rest` carries position (observed/null 1.449, 4/4 animals) so a frame-weighted average
is composed unevenly. That argument was never tested against a RESULT.

WHAT IS ALREADY KNOWN, and it sets the prior:

  * the composition imbalance is 17.5% and STABLE across epochs, so it is a FIXED bias, not one that
    tracks the deficit. The stronger half of the case for `restw` was withdrawn on 2026-09-14.
  * split-half reliability makes `restw` ~3% NOISIER than `rest` (relRMS 0.1577 vs 0.1551, worse in
    55/91 sessions), because equal weighting maximises the influence of the thinnest estimates.
  * the real limit is PER-ANIMAL, not per-estimator: PS92 median relRMS 0.228 against PS94's 0.113
    on comparable frame counts.

So `restw` is a 3% variance cost to remove a fixed 17% composition bias. Whether that trade is worth
the 200-frame floor, the 4-position rule and the column-drop apparatus depends ENTIRELY on whether it
moves a conclusion.

BOTH ARMS ARE BUILT ON THE SAME SESSIONS. A session lacking a `restw` column is skipped from BOTH,
or the comparison would be confounded by which sessions each arm contains.

READ IT AS: if both comparisons are unchanged beyond rounding, `restw` is not earning its place and
plain `rest` is the better definition -- fewer moving parts, marginally steadier at the thin end, and
a bias that is fixed and therefore statable.

RUN:  python -m scripts.rest_migration.rest_vs_restw [--align cue] [--limit 0]
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
    store = {"rest": {}, "restw": {}}
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
        rrest, rrestw = parts.get("raw_rest"), parts.get("raw_restw")
        if not rrest or not rrestw:
            # A session WITHOUT a restw column cannot appear in EITHER arm, or the two arms would be
            # built on different session sets and any difference between them would be the sessions.
            n_skip += 1
            continue
        for qname, m in rrest.items():
            store["rest"].setdefault(an, {}).setdefault(e, {}).setdefault(
                qname, []).append(np.asarray(m))
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
    for kind in ("rest", "restw"):
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
    print(f"{'epoch':<12}{'rest':>12}{'restw':>12}{'difference':>13}")
    for e in ("acute", "subacute", "chronic"):
        out = {}
        for kind in ("rest", "restw"):
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
        print(f"{e:<12}{out['rest']:>12.3f}{out['restw']:>12.3f}"
              f"{out['restw'] - out['rest']:>13.3f}")

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

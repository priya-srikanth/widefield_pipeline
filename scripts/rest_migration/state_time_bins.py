"""Can the state decoder be balanced across SESSION TIME? The feasibility count, before the fix.

THE CONFOUND. `locomotor_state` splits a session into three classes -- licking, running, rest -- and
those classes are NOT distributed alike across session time. Licking is cue-locked, so it lands
wherever trials are; rest fills the inter-trial intervals; running is whenever the animal ran, which
for a sated mouse drifts later. Meanwhile cortex itself DRIFTS: `rest_position_vs_drift` measured
the same position's rest early-vs-late at RMS 0.00282, with no behavioural difference at all. So a
three-way decoder can separate the classes partly on WHEN the window sat rather than on what the
animal was doing, and nothing in the current figures says so.

The time-local baseline that fixes this for the MAPS does not reach here: rest is a CLASS in this
analysis, not a subtrahend, so there is nothing to subtract it from.

WHAT BALANCING WOULD MEAN, and why this script comes first. Priya, 2026-09-13: "do the simple
version for now, I don't know that it's worth doing the time regression or balancing across time
bins, unless it's pretty simple to implement and doable (ie there is enough of each class across all
time bins)." Balancing = within each session, split session time into bins and subsample so every
class contributes the SAME number of segments in every bin; the decoder then cannot read time,
because time no longer predicts class. It is a few lines to implement AND IT IS ONLY POSSIBLE IF
EVERY CLASS APPEARS IN EVERY BIN. This counts that, and prices what it would cost.

WHAT IT COSTS is the number to weigh: balancing keeps `3 x min-over-classes` per bin, so a bin where
one class is thin throws away the other two's surplus. The script reports the retained fraction
per epoch, which is the whole decision.

COVERAGE IS APPROXIMATED, deliberately and in the conservative direction. The real decoder drops
segments outside imaging coverage (`locomotor_features.segment_features` via `coverage_mask`); here
a segment is kept if it lies inside the session's PCO exposure span. That is the same cut at the two
session EDGES -- which is where it matters, since the edges are bins 1 and 5 -- and misses only
mid-session frame-map gaps in regime B. Counting those would need the full frame map per session;
this is a feasibility count, not a result.

RUN:  python -m scripts.rest_migration.state_time_bins [--bins 5]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

#: Session-time bins. Five is a compromise: enough to expose a monotone drift in class composition,
#: few enough that a class with a few hundred segments still populates each one.
NBINS = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=NBINS)
    a = ap.parse_args()
    nb = int(a.bins)

    from wfield_local import behavior_events as be
    from wfield_local import config, epochs
    from wfield_local import locomotor_state as ls
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.paths import PathResolver
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events

    rv = PathResolver()
    t0 = time.time()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    # counts[epoch][class] -> (nb,) ; per_session[epoch] -> list of (3, nb) arrays
    counts, per_session, skipped = {}, {}, []

    for s in [x for x in SESSIONS if x["label"] in want and x.get("h5")]:
        lab = s["label"]
        if lab in ls.EXCLUDE_SESSIONS:
            skipped.append(f"{lab} (EXCLUDE_SESSIONS)")
            continue
        ep = epochs.epoch_of(lab)
        if ep is None:
            skipped.append(f"{lab} (no epoch)")
            continue
        an, mmdd = lab.split("_")[0], lab.split("_")[1]
        ev = be.get_or_compute(rv, an, "2026" + mmdd)
        if not ev:
            skipped.append(f"{lab} (no events)")
            continue
        try:
            cue = _load_cue_events(s["h5"])
            cues = np.asarray(cue["cue_samples"], np.int64)
            pco = np.asarray(cue["pco_samples"], np.int64)
            smp, label, _per, _ndrop = ls.three_way_segments(
                ev, lick_mode="postcue", cue_samples=cues, lick_window_s=ls.SEGMENT_S)
        except Exception as ex:                                        # noqa: BLE001
            skipped.append(f"{lab} ({type(ex).__name__} {str(ex)[:50]})")
            continue
        if not len(smp) or pco.size < 2:
            skipped.append(f"{lab} (no segments)")
            continue

        # IMAGING SPAN, not session span: the time axis the decoder actually sees.
        fs = float(ev.get("fs", 5000.0))
        n = round(ls.SEGMENT_S * fs)
        lo, hi = int(pco[0]), int(pco[-1])
        keep = (smp >= lo) & (smp + n <= hi)
        smp, label = smp[keep], label[keep]
        if not len(smp):
            skipped.append(f"{lab} (all segments outside imaging span)")
            continue

        frac = (smp - lo) / max(1, hi - lo)
        b = np.clip((frac * nb).astype(int), 0, nb - 1)
        tab = np.zeros((len(ls.THREE_WAY), nb), np.int64)
        for i, cls in enumerate(ls.THREE_WAY):
            m = label == cls
            if m.any():
                tab[i] = np.bincount(b[m], minlength=nb)
        per_session.setdefault(ep, []).append((lab, tab))
        c = counts.setdefault(ep, np.zeros((len(ls.THREE_WAY), nb), np.int64))
        c += tab
        print(f"  .. {lab} [{ep}] {tab.sum()} segments "
              f"({dict(zip(ls.THREE_WAY, tab.sum(1).tolist()))})  {time.time() - t0:.0f}s",
              flush=True)

    # ------------------------------------------------------------------ report
    order = [e for e in ("pre", "acute", "subacute", "chronic") if e in counts]
    print(f"\n{'=' * 78}\nSEGMENTS PER CLASS PER SESSION-TIME BIN  ({nb} bins, equal spans of the "
          f"imaging period)\n{'=' * 78}")
    if not order:
        print("\nNOTHING WAS COUNTED -- 0 sessions produced segments. This is NOT a negative "
              "result; it is a failed run.")
        for x in skipped:
            print("   skipped:", x)
        return 1

    for ep in order:
        tab = counts[ep]
        nsess = len(per_session[ep])
        print(f"\n--- {ep.upper()}  ({nsess} sessions) ---")
        print(f"{'class':<10}" + "".join(f"{f'bin {j + 1}':>10}" for j in range(nb)) + f"{'total':>10}")
        for i, cls in enumerate(ls.THREE_WAY):
            print(f"{cls:<10}" + "".join(f"{v:>10,}" for v in tab[i]) + f"{tab[i].sum():>10,}")
        # THE SHAPE OF THE CONFOUND: how differently are the classes spread over time?
        share = tab / np.maximum(1, tab.sum(0, keepdims=True))
        print(f"{'--share--':<10}" + "".join(f"{f'{share[:, j].max():.2f}':>10}" for j in range(nb))
              + "   <- largest class share in that bin (0.33 = balanced)")

        # POOLED balancing (one subsample over the epoch) and PER-SESSION balancing (what a decoder
        # fitting session by session could actually do). They differ, and the second is the real one.
        pooled = int(3 * tab.min(0).sum())
        ps = sum(int(3 * t.min(0).sum()) for _l, t in per_session[ep])
        tot = int(tab.sum())
        print(f"balanced yield   pooled {pooled:,} / {tot:,} = {pooled / max(1, tot):.2f}"
              f"   per-session {ps:,} / {tot:,} = {ps / max(1, tot):.2f}")
        empty = [(l_, int((t.min(0) == 0).sum())) for l_, t in per_session[ep] if (t.min(0) == 0).any()]
        print(f"sessions with at least one EMPTY class-bin: {len(empty)}/{nsess}"
              + (f"  e.g. {empty[:4]}" if empty else ""))

    print(f"\n{'=' * 78}\nVERDICT\n{'=' * 78}")
    # A verdict that names what it tested, so it cannot read as "passed" having tested nothing.
    print(f"tested {sum(len(v) for v in per_session.values())} sessions over {len(order)} epochs; "
          f"{len(skipped)} skipped")
    for x in skipped:
        print("   skipped:", x)
    worst = min(float(3 * t.min(0).sum()) / max(1, t.sum())
                for ep in order for _l, t in per_session[ep])
    print(f"\nworst per-session retained fraction under balancing: {worst:.2f}")
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

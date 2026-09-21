"""Is BLOCK ORDER systematic across sessions? The precondition for reading the amplitude gradient.

WHY THIS MATTERS. Positions are presented in ~6-trial BLOCKS, so position is confounded with
time-within-session -- that is what `rest_position_vs_drift` established (POSITION/DRIFT = 1.02).
Within one session that confound is unavoidable. ACROSS sessions it only survives if the block order
is SYSTEMATIC: if position 4 tends to run early and position 1 late in every session, then every
session's drift pushes the same positions the same way and a position-graded amplitude effect can be
manufactured from drift alone. If the order is shuffled per session, drift averages out across
sessions and a gradient that survives pooling is not drift.

THE AMPLITUDE GRADIENT (1.48 / 1.21 / 1.27 near -> 1.01 / 0.69 / 0.48 far, acute vs pre) IS NOT YET
ESTABLISHED, and DECISIONS.md records that leaning on it needs this check first. This is that check,
and it is deliberately cheap: DAQ position codes only, no imaging, no maps.

THE STATISTIC. For each session give every trial its NORMALISED INDEX in the session (0 = first,
1 = last) and take each position's mean. Under a shuffled order that mean is ~0.5 for every position,
and -- more to the point -- a position's deviation from 0.5 is UNCORRELATED across sessions.

    per-position mean normalised index, averaged over sessions:  is any position systematically
                                                                 early or late in the cohort?
    between-session consistency (mean pairwise Spearman of the   the direct test: +1 = identical
    six per-position means, session against session)             order every session, 0 = shuffled

A NEGATIVE RESULT HERE IS THE USEFUL ONE, and it must be distinguishable from a run that tested
nothing -- so the session count is printed before the verdict and "nothing was tested" is its own
outcome.

RUN:  python -m scripts.rest_migration.archive.block_order
"""
from __future__ import annotations

import sys
import time

import numpy as np


def main() -> int:
    import h5py
    from scipy.stats import spearmanr

    from wfield_local import config, daq_io
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, SESSIONS, _load_cue_events
    from wfield_local.plot_spout_trial_averages import _classify_cues

    code_of = {nm: int(c) for c, nm in POSITION_NAMES.items()}
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    t0 = time.time()
    per_session, skipped = {}, []

    for s in [x for x in SESSIONS if x["label"] in want and x.get("h5")]:
        lab = s["label"]
        try:
            cue = _load_cue_events(s["h5"])
            codes = np.asarray(_classify_cues(cue["cue_samples"], cue["strobe_samples"],
                                              cue["strobe_codes"]))
            with h5py.File(s["h5"], "r") as f:      # opened only to confirm the file is readable
                _ = f["digital/channel_names"][:]
            _ = daq_io                              # imported for the module-level side effects
        except Exception as ex:                                        # noqa: BLE001
            skipped.append(f"{lab}: {type(ex).__name__} {str(ex)[:50]}")
            continue
        n = len(codes)
        if n < 60:
            skipped.append(f"{lab}: only {n} trials")
            continue
        # NORMALISED TRIAL INDEX, not sample time: the confound is about ORDER in the block
        # sequence, and trials are what the blocks are made of.
        idx = np.arange(n) / max(1, n - 1)
        row = {}
        for q in CONF_LABELS:
            c = code_of.get(q)
            sel = codes == c
            if sel.sum() >= 10:
                row[q] = float(idx[sel].mean())
        if len(row) >= 4:
            per_session[lab] = row
        else:
            skipped.append(f"{lab}: only {len(row)} positions with >=10 trials")

    print(f"\n{len(per_session)} sessions measured, {len(skipped)} skipped "
          f"({time.time() - t0:.0f}s)")
    if len(per_session) < 3:
        print("\nNOTHING USABLE WAS TESTED -- this is a failed run, not a negative result.")
        for x in skipped[:10]:
            print("   skipped:", x)
        return 1

    print("\nMEAN NORMALISED TRIAL INDEX PER POSITION (0 = start of session, 1 = end)")
    print(f"{'position':<14}{'mean':>8}{'sd over sessions':>20}{'n sess':>8}")
    cols = {}
    for q in CONF_LABELS:
        v = [r[q] for r in per_session.values() if q in r]
        if v:
            cols[q] = v
            print(f"{q:<14}{np.mean(v):>8.3f}{np.std(v):>20.3f}{len(v):>8}")

    # THE DIRECT TEST: session against session, do the six positions rank the same way?
    labs = sorted(per_session)
    shared = [q for q in CONF_LABELS if all(q in per_session[a] for a in labs)]
    rhos = []
    if len(shared) >= 4:
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                a = [per_session[labs[i]][q] for q in shared]
                b = [per_session[labs[j]][q] for q in shared]
                r = spearmanr(a, b).statistic
                if np.isfinite(r):
                    rhos.append(float(r))

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    print(f"tested {len(per_session)} sessions on {len(shared)} positions common to all of them; "
          f"{len(rhos)} session pairs compared")
    if not rhos:
        print("NO PAIRS COMPARED -- nothing was tested. Not a negative result.")
        return 1
    m = float(np.mean(rhos))
    frac = float(np.mean([abs(r) > 0.6 for r in rhos]))
    print(f"\nmean pairwise Spearman of per-position mean index: {m:+.3f}")
    print(f"fraction of session pairs with |rho| > 0.6:         {frac:.2f}")
    print("\n  near 0  -> block order is SHUFFLED between sessions. Session drift pushes different")
    print("            positions in different sessions, so it averages out under pooling and a")
    print("            pooled amplitude gradient is NOT explained by drift-through-blocks.")
    print("  near +1 -> the SAME positions run early every session. Drift then pushes the same")
    print("            positions the same way in every session and the gradient is confounded.")
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

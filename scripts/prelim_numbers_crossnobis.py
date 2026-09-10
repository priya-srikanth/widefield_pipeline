"""GAIN vs MOVE: the exact numbers behind the crossnobis table in `docs/PRELIM_DATA_VLS_STROKE.md`.

    PYTHONPATH=$(pwd) python scripts/prelim_numbers_crossnobis.py [align] [variant]

WHY THIS IS A COMMITTED SCRIPT AND NOT A SCRATCHPAD ONE. The prelim-data document cites exact
per-position per-epoch values -- the ones the grant paragraph rests on -- and a document that quotes
numbers with no committed way to regenerate them is a document nobody can check. Every value in its
"GAIN vs MOVE" and "Direction of the acute move" tables comes from one run of this script
(2026-09-09, cue/working).

WHAT IT COMPUTES. Row-centring is an EXACT decomposition of the crossnobis own-position distance,
not an approximation. Because ``rowcentred[i,j] = raw[i,j] - mean_j raw[i,:]``:

    raw_diag[i]  =  rowmean[i]            +  rc_diag[i]
                    GAIN                     MOVE
                    position-NONspecific     position-SPECIFIC
                    (a change in P's response magnitude shifts P's distance to EVERY pre-stroke
                     position equally, so it paints a uniform row)

The identity is checked numerically on every epoch and the residual printed; if it stops being
~1e-16 then one of the two collectors has changed and the tables in the document are stale.

READ THE SIGN CONVENTION BEFORE QUOTING ANYTHING. `_matrices_crossnobis` returns RAW distances
normalised to pre-stroke units, so LARGER = further from baseline = more changed. `_mats_crossnobis`
(figure 8d) NEGATES them. The two are one letter apart in the name and carry opposite signs.
"""
import sys

import numpy as np

from wfield_local import epoch_figures as ef
from wfield_local import grant_figures as G
from wfield_local.grant_figures import CONF_LABELS

ALIGN = sys.argv[1] if len(sys.argv) > 1 else "cue"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "working"
SHORT = ["nI", "nM", "nC", "fI", "fM", "fC"]
ORDER = ("pre", "acute", "subacute", "chronic")


def main() -> int:
    raw_m, _ = G._matrices_crossnobis(ALIGN, VARIANT)
    rc_m, _ = G._matrices_crossnobis_rowcentred(ALIGN, VARIANT)
    if not raw_m or not rc_m:
        print("no matrices -- nothing to report")
        return 1
    raw_p, cov = ef.mean_matrix_by_epoch(raw_m)
    rc_p, _ = ef.mean_matrix_by_epoch(rc_m)
    order = [e for e in ORDER if e in raw_p]
    print(f"=== {ALIGN}/{VARIANT}   epochs: {order}")

    print("\n--- raw own-position distance = row mean (GAIN) + row-centred (MOVE)")
    print(f"{'pos':4s}" + "".join(f" | {e:^24s}" for e in order))
    print(f"{'':4s}" + "".join(f" | {'raw':>7s} {'gain':>7s} {'move':>7s}" for _ in order))
    for i, p in enumerate(SHORT):
        line = f"{p:4s}"
        for e in order:
            A, C = np.asarray(raw_p[e], float), np.asarray(rc_p[e], float)
            line += f" | {A[i, i]:7.3f} {np.nanmean(A[i, :]):7.3f} {C[i, i]:7.3f}"
        print(line)

    print("\n--- CHANGE FROM PRE (this is what the document quotes)")
    P, Pc = np.asarray(raw_p["pre"], float), np.asarray(rc_p["pre"], float)
    print(f"{'pos':4s}" + "".join(f" | {e:^24s}" for e in order[1:]))
    print(f"{'':4s}" + "".join(f" | {'draw':>7s} {'dGAIN':>7s} {'dMOVE':>7s}" for _ in order[1:]))
    for i, p in enumerate(SHORT):
        line = f"{p:4s}"
        for e in order[1:]:
            A, C = np.asarray(raw_p[e], float), np.asarray(rc_p[e], float)
            line += (f" | {A[i, i] - P[i, i]:7.3f} "
                     f"{np.nanmean(A[i, :]) - np.nanmean(P[i, :]):7.3f} "
                     f"{C[i, i] - Pc[i, i]:7.3f}")
        print(line)

    # WHICH COLUMN DID THE IMPAIRED POSITION MOVE TOWARD. Negative = moved TOWARD that pre-stroke
    # position. No per-cell interval exists for these; they are point estimates and the document
    # says so.
    i = SHORT.index("fC")
    print("\n--- far-contra ROW, row-centred, minus pre (negative = moved TOWARD)")
    print(f"{'epoch':9s} " + " ".join(f"{c:>7s}" for c in SHORT))
    for e in order[1:]:
        C = np.asarray(rc_p[e], float)
        print(f"{e:9s} " + " ".join(f"{C[i, j] - Pc[i, j]:7.3f}" for j in range(len(CONF_LABELS))))

    print("\n--- additivity check (raw_diag - rowmean - rc_diag; must stay ~1e-16)")
    for e in order:
        A, C = np.asarray(raw_p[e], float), np.asarray(rc_p[e], float)
        r = [A[i, i] - np.nanmean(A[i, :]) - C[i, i] for i in range(len(SHORT))]
        print(f"  {e:9s} max|resid| = {np.nanmax(np.abs(r)):.2e}")

    print(f"\ncoverage (sessions per epoch per animal): {cov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

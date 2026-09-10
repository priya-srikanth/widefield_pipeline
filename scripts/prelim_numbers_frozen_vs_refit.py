"""LOST or MISREAD: the exact numbers behind the 5r table in `docs/PRELIM_DATA_VLS_STROKE.md`.

    PYTHONPATH=$(pwd) python scripts/prelim_numbers_frozen_vs_refit.py

WHY THIS IS A COMMITTED SCRIPT. The prelim-data document's "fraction recovered by refitting" table
is the evidence for qualifying "relocated rather than lost" down to "partly relocated", which is the
most load-bearing edit that document has taken. Its numbers came from one run of this script
(2026-09-09) and nothing else regenerates them: the epoch figures draw the bars, but the per-position
"recovered / frozen deficit" ratio is computed here and nowhere in the package.

WHAT IT COMPUTES, per position and epoch:

  frozen      the pre-stroke decoder, frozen, on that epoch's trials
  refit       a decoder refitted WITHIN each session, 5-fold block CV, SAME trials
  gap         refit - frozen, PAIRED at the trial level (`_collect_5c(mode="paired")`)
  gap - pre   the gap minus the PRE gap, which is what may be quoted

THE PRE GAP IS NOT ZERO AND IS NOT AN EFFECT. The frozen arm trains on ten pre-stroke sessions and
the refit arm on one, so refitting costs accuracy at baseline with no lesion involved. Quoting a raw
post-stroke gap charges the lesion for a handicap the design imposes.

GATED CELLS PRINT AS "gated", NOT AS A NUMBER. A position the session could not train on -- below
`grant_figures.MIN_REFIT_SHARE` -- is dropped from both arms. In practice this is acute
far-contralateral in the lick-aligned arm only, where the animal barely licks that spout, and where
scoring it produced a -0.42 "refit is worse" that was the behaviour rather than the code.
"""
import numpy as np

from wfield_local import epoch_figures as ef
from wfield_local import epoch_grant_figures as eg
from wfield_local import grant_figures as G
from wfield_local.grant_figures import CONF_LABELS

SHORT = ["nI", "nM", "nC", "fI", "fM", "fC"]
ORDER = ("pre", "acute", "subacute", "chronic")


def main() -> int:
    for disp, align, variant, _wname in eg.ARMS:
        per_animal, _days = G._collect_5c(align, variant, "paired")
        if not per_animal:
            print(f"\n===== {disp} ({align}/{variant}): no records")
            continue
        pooled = {e: ef.pool_records(per_animal, e) for e in ORDER}
        pooled = {e: v for e, v in pooled.items() if v is not None}
        print(f"\n===== {disp} ({align}/{variant}) =====")
        print(f"{'pos':4s}" + "".join(f" | {e:^22s}" for e in pooled))
        print(f"{'':4s}" + "".join(f" | {'froz':>6s} {'refit':>6s} {'gap':>6s}" for _ in pooled))

        tab: dict[str, dict[str, tuple]] = {}
        for i, q in enumerate(CONF_LABELS):
            code = eg._code_of(q)
            line, tab[SHORT[i]] = f"{SHORT[i]:4s}", {}
            for e, (y, p, _b) in pooled.items():
                gap, refit = eg._gap_at(y, p, code), eg._refit_at(y, p, code)
                if gap is None or refit is None:
                    line += f" | {'gated':>22s}"
                    continue
                frozen = refit - gap
                tab[SHORT[i]][e] = (frozen, refit, gap)
                line += f" | {frozen:6.3f} {refit:6.3f} {gap:6.3f}"
            print(line)

        # THE HEADLINE RATIO. How much of each position's frozen deficit does refitting recover?
        # Denominator is the frozen deficit against PRE, numerator the gap minus the pre gap -- so
        # the training-set-size handicap is out of both. The deficit is NEGATIVE and the recovery
        # POSITIVE, so the ratio is negated to read as "34% of the drop is recoverable".
        #
        # A RATIO NEEDS A DEFICIT TO BE A RATIO OF. Where the position barely dropped, the
        # denominator goes through zero and the percentage explodes -- chronic near-middle fell
        # 0.021 and "recovered" 0.175 of it, which prints as 846% and means nothing. Below
        # MIN_DEFICIT the ratio is suppressed and only the two raw numbers are shown, because a
        # meaningless percentage in a table is read as a number by everyone who did not compute it.
        MIN_DEFICIT = 0.05
        print("\n  recovered by refitting, as a fraction of that position's own frozen deficit")
        print(f"  {'pos':4s} " + "  ".join(f"{e:>27s}" for e in list(pooled)[1:]))
        for k in SHORT:
            if "pre" not in tab[k]:
                continue
            f_pre, _r_pre, g_pre = tab[k]["pre"]
            row = f"  {k:4s} "
            for e in list(pooled)[1:]:
                if e not in tab[k]:
                    row += f"  {'gated':>27s}"
                    continue
                deficit = tab[k][e][0] - f_pre
                recovered = tab[k][e][2] - g_pre
                frac = (f"{-recovered / deficit * 100:4.0f}%" if deficit < -MIN_DEFICIT
                        else "   --")
                row += f"  d {deficit:+6.3f} rec {recovered:+6.3f} {frac}"
            print(row)

        print("  OVERALL  " + "   ".join(
            f"{e}: froz {np.mean(p[:, 0] == y):.3f} refit {np.mean(p[:, 1] == y):.3f}"
            for e, (y, p, _b) in pooled.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

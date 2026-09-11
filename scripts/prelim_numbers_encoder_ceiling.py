"""The exact numbers behind the encoder-ceiling tables in `docs/ENCODER_CEILING.md`.

    PYTHONPATH=$(pwd) python scripts/prelim_numbers_encoder_ceiling.py

WHY THIS IS A COMMITTED SCRIPT, and not a paste. Those tables carry the displaced-vs-degraded
dissociation, and they have already been WRONG ONCE in a way no figure showed: the first version
compared RAW EV between the ceiling and the matched arm, which is confounded by amplitude, and it
reported shape as monotonically recovering when what was recovering was the fitted scale. The tables
were also first pulled at `ENC_CEILING_REPEATS = 8` and re-pulled at 32; without a script, "which
setting produced the published number" is unanswerable.

WHAT IT PRINTS, per window and epoch:

  ceiling      this session's half predicting its other half -- what its trials can predict at all
  matched      an equally sized draw from the pre-stroke pool: same estimator, same reference size,
               differing only in WHICH SESSIONS the reference came from
  all pre      the whole pre-stroke pool -- the encoder as actually used, and NOT size-matched
  gap          ceiling - matched, and that gap's change from its own PRE value
  captures     matched / ceiling
  a(m)         the amplitude factor the MATCHED arm fitted -- the one to read beside `matched`
  a(all)       the amplitude factor the UNMATCHED (as-used) frozen arm fitted

TWO AMPLITUDE COLUMNS, BECAUSE THEY DIFFER AND HAVE BEEN CONFUSED. Post-cue acutely they are 0.286
and 0.361; pre-stroke, 0.749 and 0.943. `a(m)` belongs beside the matched arm, which is what the
ceiling comparison uses. `a(all)` is the one quoted when explaining why the RAW frozen EV is
confounded, because there the frozen arm being described is the unmatched, as-actually-used one.

EVERY SCORE IS AFTER RESCALE (index 2), never raw (index 0). Amplitude is out of all of them and is
reported separately as `a`. With PERFECT shape and only a scale mismatch, `R2 = 1 - (1-a)^2/a^2` --
0.000 at a = 0.5 and -2.13 at the a = 0.361 observed acutely -- so a raw gap reports amplitude and
calls it template mismatch.

THE PRE GAP IS NOT ZERO AND IS NOT AN EFFECT: it is the cost of a template coming from other
sessions at equal training-set size, with no lesion involved. Read every post-stroke gap against it.
"""
import numpy as np

from wfield_local import epoch_figures as ef
from wfield_local import epoch_grant_figures as eg
from wfield_local import grant_figures as G
from wfield_local.grant_figures import CONF_LABELS

SHORT = ["nI", "nM", "nC", "fI", "fM", "fC"]
ORDER = ("pre", "acute", "subacute", "chronic")
MIN_CEILING = getattr(eg, "MIN_CEILING", 0.10)


def _shape(payload, _key):
    """Index 2: EV AFTER RESCALE. Never index 0 -- see the module docstring."""
    try:
        v = payload[2]
    except Exception:                                                  # noqa: BLE001
        return None
    return None if v is None or not np.isfinite(v) else float(v)


def _amp(payload, _key):
    try:
        v = payload[1]
    except Exception:                                                  # noqa: BLE001
        return None
    return None if v is None or not np.isfinite(v) else float(v)


def _per(payload, key):
    v = (payload[3] or {}).get(key)
    return None if v is None or not np.isfinite(v) else float(v)


def _pool(tab, fn, keys=("value",)):
    vals, _pts = ef.scalar_by_epoch(tab, fn, keys=list(keys))
    return vals


def main() -> int:
    print(f"ENC_CEILING_REPEATS = {G.ENC_CEILING_REPEATS}\n")
    for disp, align, variant, wname in eg.ARMS:
        raw_tab, _d = G._enc_tables(align, variant)
        cei_tab, mat_tab = G._enc_ceiling_tables(align, variant)[0], \
            G._enc_matched_tables(align, variant)[0]
        if not cei_tab or not mat_tab:
            print(f"== {disp} ({wname}): no data\n")
            continue

        cei = _pool(cei_tab, _shape)
        mat = _pool(mat_tab, _shape)
        allp = _pool(raw_tab, _shape)
        amp_m = _pool(mat_tab, _amp)
        amp_a = _pool(raw_tab, _amp)

        print(f"== {disp} -- {wname}")
        print(f"{'epoch':<9} {'ceiling':>8} {'matched':>8} {'all pre':>8} "
              f"{'gap':>7} {'d vs pre':>9} {'captures':>9} {'a(m)':>6} {'a(all)':>7}")
        base = None
        for e in ORDER:
            c = (cei.get(e) or {}).get("value")
            m = (mat.get(e) or {}).get("value")
            p = (allp.get(e) or {}).get("value")
            am = (amp_m.get(e) or {}).get("value")
            aa_ = (amp_a.get(e) or {}).get("value")
            if c is None or m is None:
                print(f"{e:<9} {'--':>8}")
                continue
            gap = c - m
            if e == "pre":
                base = gap
            dv = "--" if base is None or e == "pre" else f"{gap - base:+.3f}"
            cap = f"{m / c:.0%}" if c > 0 else "--"
            ps = "--" if p is None else f"{p:.3f}"
            s_am = "--" if am is None else f"{am:.3f}"
            s_aa = "--" if aa_ is None else f"{aa_:.3f}"
            print(f"{e:<9} {c:>8.3f} {m:>8.3f} {ps:>8} {gap:>7.3f} {dv:>9} {cap:>9} "
                  f"{s_am:>6} {s_aa:>7}")

        # PER POSITION. The fraction is formed PER SESSION and then averaged -- not as a ratio of
        # two pooled numbers, which is what the first version did and which left it with no
        # distribution at all. Cells whose ceiling is below MIN_CEILING are dropped, since a ratio
        # to a near-zero denominator reads as a result.
        frac_tab = {}
        for an, by_key in cei_tab.items():
            rec = {}
            for key, cpay in by_key.items():
                mpay = (mat_tab.get(an) or {}).get(key)
                if mpay is None:
                    continue
                cper, mper = (cpay[3] or {}), (mpay[3] or {})
                row = {q: float(max(0.0, min(1.0, mper[q] / cper[q])))
                       for q in CONF_LABELS
                       if cper.get(q) is not None and mper.get(q) is not None
                       and np.isfinite(cper[q]) and np.isfinite(mper[q]) and cper[q] >= MIN_CEILING}
                if row:
                    rec[key] = (np.nan, np.nan, np.nan, row)
            if len(rec) > 1:
                frac_tab[an] = rec

        cpos = _pool(cei_tab, _per, keys=CONF_LABELS)
        mpos = _pool(mat_tab, _per, keys=CONF_LABELS)
        fpos = _pool(frac_tab, _per, keys=CONF_LABELS) if frac_tab else {}
        print("\n   per position, ceiling / captured fraction (clipped to [0,1], as drawn)")
        print(f"   {'epoch':<9} " + " ".join(f"{s:>13}" for s in SHORT))
        for e in ORDER:
            cells = []
            for q in CONF_LABELS:
                c = (cpos.get(e) or {}).get(q)
                f = (fpos.get(e) or {}).get(q)
                cells.append("--" if c is None else
                             f"{c:.3f}/{'--' if f is None else f'{f:.2f}'}")
            print(f"   {e:<9} " + " ".join(f"{c:>13}" for c in cells))
        # The UNCLIPPED matched EV, because the clip hides a sign. Far-contra post-cue acutely is
        # -0.060: the pre-stroke template is further from that position's measured pattern than
        # predicting zero would be. The per-position score uses the session-GLOBAL scale factor,
        # fitted to a session five other positions dominate, so at a position whose amplitude has
        # collapsed that factor is simply the wrong one -- which the pooled after-rescale score,
        # free to choose a = 0, cannot be. Report the fraction; do not report the clip as if the
        # underlying number were positive.
        print("\n   per position, matched EV UNCLIPPED (negative = worse than predicting zero)")
        print(f"   {'epoch':<9} " + " ".join(f"{s:>13}" for s in SHORT))
        for e in ORDER:
            got = [(mpos.get(e) or {}).get(q) for q in CONF_LABELS]
            cells = ["--" if v is None else f"{v:.3f}" for v in got]
            print(f"   {e:<9} " + " ".join(f"{c:>13}" for c in cells))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

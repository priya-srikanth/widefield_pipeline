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
    _delta_rest_figure(store, pooled, order, mask)
    print(f"\n[done in {time.time() - t0:.0f}s]")
    return 0


def _delta_rest_figure(store, pooled, order, mask):
    """`epoch_15d` — THE DELTA-REST FIGURE: `flatpool − restw`, per position and epoch.

    Priya asked for a "delta rest figure" on 2026-09-16. **It must be FLAT-POOLED minus RESTW, and
    NOT production-`rest` minus `restw`.** Production `rest` is TIME-LOCAL and `restw` is FLAT, so
    the naive difference moves on BOTH axes at once — composition AND temporal — which is exactly
    the confound that got `rest_vs_restw` withdrawn on 2026-09-14. Both arms here are flat, so the
    only thing that differs is COMPOSITION, which is what the figure is about.

    WHAT IT SHOWS. `flatpool` weights each rest FRAME equally, so each POSITION by however many rest
    frames it happened to supply; `restw` weights the six positions equally. Their difference is
    therefore the composition bias itself, drawn as a map.

    **THE PREDICTION UNDER TEST: the delta should GROW post-stroke.** Post-stroke the animal stops
    attempting the far positions, so their share of rest frames falls and a frame-weighted baseline
    drifts toward the NEAR positions' rest — the bias should track the deficit. If it does NOT grow,
    the correctness argument for `restw` is sound but INERT, and that is worth knowing explicitly
    rather than assuming the machinery earns its keep.
    """
    import numpy as _np

    from wfield_local import beta_maps as bm
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.paths import PathResolver

    # THE DELTA IS ONE MAP PER EPOCH, NOT SIX. The algebra decides this and a first version of this
    # function got it wrong, drawing a 6 x 4 grid whose rows were identical by construction:
    #
    #     flatpool_q - restw_q = (raw_q - quiet_flat) - (raw_q - restw) = restw - quiet_flat
    #
    # The position's own data cancels -- both arms subtract a SESSION-LEVEL baseline from the same
    # `raw_q`, so their difference does not involve `q` at all. It is the same cancellation that
    # makes "reference each position to its own rest, then compare maps" reduce to `15x`. The smoke
    # test showed all six positions at an identical 0.00071 and that is CORRECT, not a bug; the
    # per-position grid was the error.
    #
    # ROWS ARE ANIMALS, so the pooled mean cannot hide one animal carrying the effect -- the failure
    # that produced the withdrawn acute dip.
    per_animal_cells, amps = {}, {}
    animals = sorted(set(store["flatpool"]) | set(store["restw"]))

    def _one(kind, an, e):
        """That animal's epoch-mean baseline-referenced map, averaged over positions present."""
        ms = [_np.mean(v, axis=0) for q in CONF_LABELS
              if (v := ((store[kind].get(an, {}).get(e) or {}).get(q) or []))]
        return _np.mean(ms, axis=0) if ms else None

    for an in animals:
        for e in order:
            a, b = _one("flatpool", an, e), _one("restw", an, e)
            if a is None or b is None:
                continue
            per_animal_cells[(an, e)] = _np.asarray(a) - _np.asarray(b)

    cells, amps = {}, {}
    for e in order:
        ds = [per_animal_cells[(an, e)] for an in animals if (an, e) in per_animal_cells]
        if not ds:
            continue
        d = _np.mean(ds, axis=0)
        cells[("cohort", e)] = d
        v = d[mask]
        amps[e] = float(_np.sqrt(_np.mean(v[_np.isfinite(v)] ** 2)))
    for an in animals:
        for e in order:
            if (an, e) in per_animal_cells:
                cells[(an, e)] = per_animal_cells[(an, e)]

    if not cells:
        print("\n!! delta-rest figure: nothing pooled, no figure written")
        return None

    print(f"\n{'=' * 78}\n3. DELTA-REST (restw - flatpool baseline) -- does the composition bias "
          f"GROW post-stroke?\n{'=' * 78}")
    print("ONE map per epoch, not six: the position's own data CANCELS between the two arms")
    print("(flatpool_q - restw_q = restw - quiet_flat), so the delta does not depend on q.\n")
    print(f"{'':<12}" + "".join(f"{e:>12}" for e in order) + f"{'chronic/pre':>13}")
    print(f"{'cohort':<12}" + "".join(f"{amps.get(e, float('nan')):>12.5f}" for e in order)
          + f"{(amps.get('chronic', float('nan')) / amps['pre']) if amps.get('pre') else float('nan'):>13.3f}")
    grew = 0
    for an in animals:
        row = {}
        for e in order:
            if (an, e) not in per_animal_cells:
                continue
            v = per_animal_cells[(an, e)][mask]
            row[e] = float(_np.sqrt(_np.mean(v[_np.isfinite(v)] ** 2)))
        if not row:
            continue
        r = (row.get("chronic", float("nan")) / row["pre"]) if row.get("pre") else float("nan")
        grew += int(_np.isfinite(r) and r > 1.0)
        print(f"{an:<12}" + "".join(f"{row.get(e, float('nan')):>12.5f}" for e in order)
              + f"{r:>13.3f}")
    print(f"\n{grew}/{len(animals)} animals have a LARGER delta chronically than pre-stroke.")
    print("  PREDICTION was that it GROWS -- post-stroke the far positions supply fewer rest")
    print("  frames, so a frame-weighted baseline drifts toward the near positions' rest.")
    print("  If it does NOT grow, `restw` is correct-but-INERT and that is the honest verdict.")

    d = __import__("pathlib").Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"
    out = ef.map_grid(
        cells, d, name="epoch_15d_delta_rest_flatpool_minus_restw",
        title="DELTA-REST: frame-weighted (flatpool) MINUS position-weighted (restw) baseline",
        row_labels=["cohort"] + list(animals),
        col_labels=list(order), edges=bm.atlas_edges(), blank=bm.excluded_mask(),
        cbar_label="flatpool - restw",
        subtitle=(
            "BOTH ARMS ARE FLAT, so the only thing differing is COMPOSITION -- `flatpool` weights "
            "each rest FRAME equally (hence each position by how many frames it supplied), `restw` "
            "weights the six positions equally. This is NOT production-`rest` minus `restw`: that "
            "difference moves on the temporal axis too, which is what got `rest_vs_restw` "
            "withdrawn. PREDICTION: the bias should GROW post-stroke as the animal stops "
            "attempting the far positions."))
    print(f"\nfigure: {out}")
    return out


if __name__ == "__main__":
    sys.exit(main())

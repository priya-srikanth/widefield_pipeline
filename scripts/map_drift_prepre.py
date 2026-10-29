"""How far do the position maps DRIFT with no stroke? Early pre (June) vs late pre (August), beside the stroke changes.

Priya 10-25: "can we also add a version of this [figure 15r] that is late pre vs early pre to compare the map drift".
The decoder / crossnobis drift group (`scripts/june_august_drift.py`) showed that two months of pre-stroke time move a
frozen decoder about as far as the chronic stroke does; this asks the same of the MAPS. Same object as figure 15r
(`epoch_maps._fig_15r_reference_maps`): per-session trial-average position maps from
`position_reference_maps.maps_by_epoch` (cached per arm by `map_collect`), pooled WITHIN animal on each animal's own
reference scale (`beta_maps.within_animal_pooled`, intersection of animals only), difference panels tested with
`beta_maps.significance_contour` (nested animals -> sessions bootstrap, the deck's primary test). Nothing new is
fitted.

The pre sessions are split by date: EARLY = before 08-01 (June), LATE = 08-01 onward (August, the primary baseline).
Columns: early pre | late pre | LATE - EARLY (the drift; scaled by the EARLY maps) | acute - late pre | chronic - late
pre (the stroke changes against the August baseline, scaled by the LATE maps). Each difference is on its reference
epoch's own scale, so the drift and stroke columns are both "change relative to the reference amplitude". Caveats:
June -> August also contains learning (the animals were still improving in June), so the drift column is an upper
bound on drift alone; June has ~3 sessions per animal against ~8 in August; the gap is ~2 months vs ~1-7 weeks after
the stroke.

    python scripts/map_drift_prepre.py --arm contact|lp|enl1s [--refs rest precue] [--aligns cue lick precue]
Arms: contact = DAQ spout contact (grant_figures/epoch), lp = LP video lick gate (WIDEFIELD_LICK_SOURCE=lp,
grant_figures/LP_licks), enl1s = 1.0 s pre-cue window (WIDEFIELD_ENL_WINDOW_S=1.0, grant_figures/DAQ_ENL1s; precue only).
Output: <labcams>/grant_figures/map_drift_prepre/{contact,LP_licks,DAQ_ENL1s}/epoch_15rdrift_<REF>ref_<align>_<variant>.png + a CSV of the
amplitude ratios and test summaries per panel.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

LATE_FROM = "0801"
ARMS = {"contact": {}, "lp": {"WIDEFIELD_LICK_SOURCE": "lp"}, "enl1s": {"WIDEFIELD_ENL_WINDOW_S": "1.0"}}
ALIGN_VARIANT = {"cue": "working", "precue": "working", "lick": "lick"}


def split_pre(store, reference, q):
    """({animal: [early pre maps]}, {animal: [late pre maps]}) for one reference and position."""
    early, late = {}, {}
    for an, by_e in store.items():
        for label, d in ((by_e.get("pre") or {}).get(q) or {}).items():
            if reference not in d:
                continue
            (late if label.split("_")[-1] >= LATE_FROM else early).setdefault(an, []).append(d[reference])
    return early, late


def post_maps(store, reference, q, epoch):
    out = {}
    for an, by_e in store.items():
        got = [d[reference] for d in ((by_e.get(epoch) or {}).get(q) or {}).values() if reference in d]
        if got:
            out[an] = got
    return out


def drift_figure(store, align, variant, reference, out_dir, arm):
    import numpy as np

    from wfield_local import beta_maps as bm
    from wfield_local import epoch_figures as ef
    from wfield_local import position_reference_maps as prm
    from wfield_local.epoch_maps import _long_of
    from wfield_local.grant_figures import CONF_LABELS

    COLS = ["early pre (June)", "late pre (Aug)", "LATE - EARLY", "acute - late pre", "chronic - late pre"]
    DCOLS = tuple(COLS[2:])
    cells, titles, contours, rows, stats = {}, {}, {}, [], []
    for q in CONF_LABELS:
        early, late = split_pre(store, reference, q)
        if not (early and late):
            continue
        row = _long_of(q)
        rows.append(row)
        e_p, l_p, d_el, r_el, ans = bm.within_animal_pooled(early, late)
        if e_p is None:
            continue
        cells[(row, COLS[0])], cells[(row, COLS[1])], cells[(row, COLS[2])] = e_p, l_p, d_el
        n_e = sum(len(early[a]) for a in ans)
        n_l = sum(len(late[a]) for a in ans)
        titles[(row, COLS[0])] = f"June pre\n{len(ans)} an, {n_e} sess"
        titles[(row, COLS[1])] = f"Aug pre\n{len(ans)} an, {n_l} sess"
        titles[(row, COLS[2])] = f"AUG - JUNE (no stroke)\nratio {r_el:.2f}"
        comps = [(COLS[2], early, late, r_el, ans)]
        for e, col in (("acute", COLS[3]), ("chronic", COLS[4])):
            post = post_maps(store, reference, q, e)
            if not post:
                continue
            _p, _post_p, d, r, a2 = bm.within_animal_pooled(late, post)
            if d is None:
                continue
            cells[(row, col)] = d
            titles[(row, col)] = f"{e.upper()} - AUG PRE\n{len(a2)} an, ratio {r:.2f}"
            comps.append((col, late, post, r, a2))
        for col, ref_by, cmp_by, ratio, an_used in comps:
            try:
                cm, lab = bm.significance_contour(ref_by, cmp_by)
                print(f"  .. drift {arm} {reference} {align} {q} {col}: {lab}", flush=True)
                stats.append(dict(row=row, col=col, n_animals=len(an_used), animals=";".join(an_used),
                                  amplitude_vs_reference=round(float(ratio), 4),
                                  **getattr(bm.significance_contour, "last", {})))
                if cm is not None and np.any(cm):
                    contours[(row, col)] = cm
            except Exception as ex:                                   # noqa: BLE001
                print(f"  !! drift sig {reference} {q} {col}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
    if not cells:
        return None, stats
    tag = prm.reference_tag(reference)
    name = f"epoch_15rdrift_{tag}ref_{align}_{variant}"
    p = ef.map_grid(
        cells, out_dir, name=name,
        title=f"Map DRIFT without a stroke (June -> August pre) beside the stroke changes -- {tag} reference, "
              f"{align} / {variant}, arm {arm}",
        row_labels=rows, col_labels=COLS, panel_titles=titles, delta_cols=DCOLS, edges=bm.atlas_edges(),
        contours=contours, blank=bm.excluded_mask(), row_scaled=False, stat_rows=stats,
        cbar_label="position map, own-reference scale", delta_label="change vs reference\n(SAME scale as the maps)",
        subtitle=("Same per-session maps and test as figure 15r; pre split by date (June = early, August = late). "
                  "LATE - EARLY is scaled by each animal's June maps, the stroke columns by its August maps; ratio "
                  "= amplitude relative to the reference epoch. Pooled within animal (animals in both epochs only). "
                  "Contours = nested animals -> sessions bootstrap. June -> August includes learning, so the drift "
                  "column is an upper bound on drift alone; June ~3 sessions per animal vs August ~8."))
    return p, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=tuple(ARMS), default="contact")
    ap.add_argument("--refs", nargs="+", default=["rest", "precue"])
    ap.add_argument("--aligns", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if os.environ.get("WIDEFIELD_PRE_FROM"):
        raise SystemExit("unset WIDEFIELD_PRE_FROM: this figure needs the June pre sessions")
    for k in ("WIDEFIELD_LICK_SOURCE", "WIDEFIELD_ENL_WINDOW_S"):
        os.environ.pop(k, None)
    os.environ.update(ARMS[a.arm])

    import pandas as pd

    from wfield_local import position_reference_maps as prm
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    aligns = a.aligns or (["precue"] if a.arm == "enl1s" else ["precue", "cue", "lick"])
    from wfield_local import lp_licks
    # a non-default arm must sit under its tag directory (LP_licks, DAQ_ENL1s) or writeguard refuses it
    out_dir = a.out or Path(PathResolver().root("cohort_out")) / "grant_figures" / "map_drift_prepre"
    tag = lp_licks.output_tag()
    out_dir = out_dir / (tag or "contact") if tag is None or f"/{tag.lower()}/" not in out_dir.as_posix().lower() + "/" else out_dir
    assert_writable(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for align in aligns:
        variant = ALIGN_VARIANT[align]
        store, _rel, _ntr = prm.maps_by_epoch(align, variant)
        for ref in a.refs:
            if ref == "precue" and align == "precue":      # baseline inside the window (figure 15r's rule)
                continue
            p, stats = drift_figure(store, align, variant, ref, out_dir, a.arm)
            print(f"-> {p}", flush=True)
            rows += [dict(arm=a.arm, align=align, variant=variant, reference=ref, **s) for s in stats]
    pd.DataFrame(rows).to_csv(out_dir / "drift_vs_stroke_panels.csv", index=False)
    print(f"-> {out_dir / 'drift_vs_stroke_panels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

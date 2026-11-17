"""IS THE NEURAL MEASURE STILL MOVING AFTER THE BEHAVIOR HAS PLATEAUED?

THE QUESTION (Priya, 2026-09-18). The chronic epoch is defined by BEHAVIOR: it begins where
far-contra hit rate stops changing, by the `epochs.CHRONIC_RULE` flat+settled test. Every claim we
make about chronic neural change is therefore a claim about a window whose boundary a *different*
measure drew. That leaves a question nobody has asked of this dataset: inside that window, has the
NEURAL measure plateaued too, or is cortex still changing after the animal looks recovered?

IT IS NOT THE SAME QUESTION AS "does the chronic epoch differ from pre-stroke", which the epoch
figures already answer. A persistent offset and an ongoing drift are indistinguishable from a
single epoch-vs-pre contrast -- both show up as "chronic != pre". Separating them needs the trend
WITHIN the chronic sessions, which is what this module measures.

WHY IT MATTERS FOR WHAT WE SAY. "The cortical representation remains altered after behavior has
normalized" is supported by the epoch contrast. "The circuits remain PLASTIC after behavioral
recovery" is not -- that is a statement about continuing change, and it needs this test. The
distinction was caught before it reached a grant (Priya, 2026-09-18: "we haven't looked for changes
DURING the chronic period"), and this module exists so the stronger claim can be made or dropped on
evidence.

THE TEST IS DELIBERATELY THE BEHAVIORAL ONE. We reuse `epochs._is_flat` and the same tolerances
(`CHRONIC_K_DRIFT` x pre-stroke SD for total fitted drift, `CHRONIC_K_RES` x pre-stroke SD for
residual scatter) applied to the neural series, so the sentence the result licenses is exact: "by
the same criterion that declared the behavior plateaued, the neural measure has / has not." Any
other tolerance would invite the objection that the two were judged by different standards.

Everything is normalized to that animal's own pre-stroke mean by `epochs._normalised_series`, which
is what makes a pre-stroke SD the right yardstick, and which works on any `{label: {position:
value}}` table -- behavioral or neural. That shape is the module's only input contract.

WHAT THIS CANNOT DO, and say it before quoting a result:
  * The widefield cohort has TWO animals with a chronic epoch (PS92, PS93), each with a handful of
    sessions. A non-significant flat result here is "not detected at n = 2", not "flat".
  * `epochs._fit_line` fits against SESSION INDEX, not day (its own docstring says so). Chronic
    sessions are unevenly spaced, so a per-day rate is reported separately and should be preferred
    when quoting a rate.
  * A drift inside the chronic window can also be ordinary representational drift, which is what
    the sham arm is for. With no shams recorded yet, a non-zero slope here is suggestive of
    continuing post-stroke change but does not separate it from drift a healthy animal would show.

Run: ``python -m wfield_local.chronic_trend --help``
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random

from wfield_local import epochs

#: Bootstrap draws for the interval on total drift. Sessions are the resampling unit, matching the
#: session-level inference used everywhere else in this repo.
N_BOOT = 2000


def _fit_per_day(points):
    """``(slope_per_day, intercept)`` by least squares against DAY, not session index."""
    n = len(points)
    if n < 2:
        return 0.0, (points[0][1] if points else 0.0)
    mx = sum(d for d, _v in points) / n
    my = sum(v for _d, v in points) / n
    sxx = sum((d - mx) ** 2 for d, _v in points)
    if sxx == 0:
        return 0.0, my
    slope = sum((d - mx) * (v - my) for d, v in points) / sxx
    return slope, my - slope * mx


def chronic_trend(by_session, animal, position, *, chronic_from=None, seed=0):
    """Flat/settled test applied INSIDE one animal's chronic window, for one measure.

    ``by_session`` is ``{session_label: {position_name: value}}`` -- the same contract
    `epochs.derive_chronic_boundaries` takes, so a neural table and a behavioral one are scored
    identically. Returns None when the animal has no chronic epoch or too few chronic sessions.
    """
    spec = epochs.EPOCH_SPEC.get(animal) or {}
    start = chronic_from if chronic_from is not None else spec.get("chronic_from")
    if start is None:
        return {"animal": animal, "note": "no chronic epoch for this animal"}
    pre, post = epochs._normalised_series(by_session, animal, position)
    if not pre:
        return {"animal": animal, "note": "no pre-stroke baseline for this measure"}
    sd_pre = epochs._pstdev(pre)
    tail = [(d, v) for d, v in post if d >= start]
    if len(tail) < epochs.CHRONIC_MIN_TAIL:
        return {"animal": animal, "note": f"only {len(tail)} chronic sessions "
                                          f"(need {epochs.CHRONIC_MIN_TAIL})"}
    ys = [v for _d, v in tail]
    ok_flat, slope_idx, resid = epochs._is_flat(ys, sd_pre)
    drift = slope_idx * (len(ys) - 1)
    slope_day, _ = _fit_per_day(tail)
    rng = random.Random(seed)
    boots = []
    for _ in range(N_BOOT):
        draw = [tail[rng.randrange(len(tail))] for _ in range(len(tail))]
        if len({d for d, _v in draw}) < 2:
            continue
        s, _ = _fit_per_day(draw)
        boots.append(s * (max(d for d, _ in tail) - min(d for d, _ in tail)))
    boots.sort()
    lo = boots[int(0.025 * len(boots))] if boots else float("nan")
    hi = boots[int(0.975 * len(boots))] if boots else float("nan")
    return {
        "animal": animal, "position": position, "chronic_from": start,
        "n_chronic": len(tail), "days": [d for d, _v in tail],
        "sd_pre": sd_pre, "level": sum(ys) / len(ys),
        "drift_total": drift, "drift_tol": epochs.CHRONIC_K_DRIFT * sd_pre, "flat": bool(ok_flat),
        "resid": resid, "resid_tol": epochs.CHRONIC_K_RES * sd_pre,
        "settled": bool(resid <= epochs.CHRONIC_K_RES * sd_pre),
        "slope_per_day": slope_day,
        "drift_over_window_ci": [lo, hi],
        "ci_excludes_zero": bool(boots and (lo > 0 or hi < 0)),
        "plateaued": bool(ok_flat and resid <= epochs.CHRONIC_K_RES * sd_pre),
    }


def run(tables, position=None, *, chronic_from=None, out=None):
    """Score every (measure, animal) pair. ``tables`` is ``{measure_name: by_session}``."""
    position = position or epochs.RULE_POSITION
    rows = []
    for name, tbl in tables.items():
        for animal in sorted(epochs.EPOCH_SPEC):
            r = chronic_trend(tbl, animal, position, chronic_from=chronic_from)
            if r:
                rows.append({"measure": name, **r})
    if out:
        out = pathlib.Path(out)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            cols = ["measure", "animal", "position", "chronic_from", "n_chronic", "sd_pre",
                    "level", "drift_total", "drift_tol", "flat", "resid", "resid_tol", "settled",
                    "slope_per_day", "drift_over_window_ci", "ci_excludes_zero", "plateaued",
                    "note"]
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {out}")
    return rows


def report(rows):
    """Print one line per (measure, animal), plus the honest cohort summary."""
    for r in rows:
        if r.get("note"):
            print(f"  {r['measure']:>28s}  {r['animal']}  -- {r['note']}")
            continue
        ci = r["drift_over_window_ci"]
        verdict = "PLATEAUED" if r["plateaued"] else "STILL CHANGING"
        print(f"  {r['measure']:>28s}  {r['animal']}  n={r['n_chronic']}  "
              f"drift={r['drift_total']:+.3f} (tol {r['drift_tol']:.3f}, "
              f"{'flat' if r['flat'] else 'NOT flat'})  "
              f"resid={r['resid']:.3f} (tol {r['resid_tol']:.3f}, "
              f"{'settled' if r['settled'] else 'NOT settled'})  "
              f"per-day={r['slope_per_day']:+.4f}  "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}]{'*' if r['ci_excludes_zero'] else ''}  {verdict}")
    scored = [r for r in rows if not r.get("note")]
    n_animals = len({r["animal"] for r in scored})
    if n_animals < 3:
        print(f"\n  READ WITH CARE: {n_animals} animal(s) have a chronic epoch at all. A 'plateaued'"
              f"\n  verdict here means 'no change detected at this n', not 'no change'.")


# --------------------------------------------------------------------------- loaders

def from_labelled_csv(path, *, value_col, label_col="label", position_col=None,
                      position=None, filter_col=None, filter_value=None):
    """``{label: {position: value}}`` from any CSV that carries a SESSION LABEL.

    The epoch figures' ``*_sessions.csv`` sidecars are ``epoch,position,animal,value`` and carry NO
    session identifier, so they cannot be used here -- that is a real gap and the reason this
    loader takes an explicit label column rather than defaulting to those files.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if filter_col and str(row.get(filter_col)) != str(filter_value):
                continue
            lab = row.get(label_col)
            v = row.get(value_col)
            if not lab or v in (None, ""):
                continue
            pos = row.get(position_col) if position_col else (position or epochs.RULE_POSITION)
            try:
                out.setdefault(lab, {})[pos] = float(v)
            except ValueError:
                continue
    return out


def from_coding_direction(path, *, cls="poststroke_engaged", method="dom_orth"):
    """``{label: {position: mean}}`` from ``coding_direction.json``.

    Schema confirmed against `wfield_local.miss_vs_stopped.series`:
    ``rec["methods"][method]["positions"][position]["by_session"][label][cls]["mean"]``.
    """
    rec = json.loads(pathlib.Path(path).read_text())
    out = {}
    for pos, d in ((rec.get("methods", {}).get(method, {}) or {}).get("positions", {}) or {}).items():
        for lab, per_cls in (d.get("by_session") or {}).items():
            v = ((per_cls or {}).get(cls) or {}).get("mean")
            if v is not None:
                out.setdefault(lab, {})[pos] = float(v)
    return out


def behaviour_tables(position=None):
    """The behavioral hit-rate table the chronic boundary itself was drawn from, for comparison."""
    from wfield_local import epoch_audit
    hit, licks = epoch_audit.load_far_position_tables(position=position)
    tables = {}
    if hit:
        tables["behavior: hit rate"] = hit
    if licks:
        tables["behavior: licks/trial"] = licks
    return tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--position", default=None, help=f"default {epochs.RULE_POSITION}")
    ap.add_argument("--chronic-from", type=int, default=None,
                    help="override the chronic onset day for every animal (sensitivity check)")
    ap.add_argument("--csv", action="append", default=[],
                    metavar="NAME=PATH:VALUE_COL[:LABEL_COL[:POSITION_COL]]",
                    help="add a neural measure from a session-labeled CSV")
    ap.add_argument("--coding-direction", default=None,
                    help="path to coding_direction.json (adds the per-position coding measure)")
    ap.add_argument("--no-behaviour", action="store_true",
                    help="skip the behavioral comparison series")
    ap.add_argument("--out", default=None, help="write results CSV here")
    a = ap.parse_args(argv)

    tables = {} if a.no_behaviour else behaviour_tables(a.position)
    for spec in a.csv:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        path, value_col = parts[0], parts[1]
        label_col = parts[2] if len(parts) > 2 else "label"
        position_col = parts[3] if len(parts) > 3 else None
        tables[name] = from_labelled_csv(path, value_col=value_col, label_col=label_col,
                                         position_col=position_col, position=a.position)
    if a.coding_direction:
        tables["coding direction"] = from_coding_direction(a.coding_direction)
    if not tables:
        raise SystemExit("no measures given: pass --csv and/or --coding-direction")
    print(f"chronic-window trend test, position={a.position or epochs.RULE_POSITION}, "
          f"flat_mode={epochs.CHRONIC_FLAT_MODE}, k_drift={epochs.CHRONIC_K_DRIFT}, "
          f"k_res={epochs.CHRONIC_K_RES}")
    rows = run(tables, a.position, chronic_from=a.chronic_from, out=a.out)
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

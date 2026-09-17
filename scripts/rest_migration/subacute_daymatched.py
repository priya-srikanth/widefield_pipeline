"""Is the SUBACUTE between-animal non-replication biology, or is it the BIN?

THE STANDING PROBLEM. `rest_null` reports mean pairwise between-animal r for each epoch's
`epoch - pre` map. Acute replicates at 0.810; subacute comes in at 0.030 (flatpool) / 0.052
(restw). That gap has blocked every subacute claim in the arm, and it has been carried as "science
still open" without a diagnosis.

THE HYPOTHESIS THIS TESTS, and it is not about biology. `acute` is the first `acute_fraction` of
post-stroke days; `chronic` starts when THAT ANIMAL's hit rate flattens (`configs/defaults.yaml
epochs.chronic`, a per-animal behavioural criterion). **Subacute is neither -- it is the RESIDUAL.**
A residual bin has no fixed meaning across animals, and here the spread is extreme:

    animal   acute days        subacute days          median subacute day
    PS92     1,2,3,4,5         7, 9                            8
    PS93     1,2,3,4           5, 7, 9                         7
    PS94     1,2,3,4,5,7       9,11,15,18,22,25,29            18      <- no chronic at all
    PS95     1                 2,3,4,5,7,9                     4.5    <- one acute session

PS94's subacute IS the other animals' chronic day set. PS95's subacute starts at day 2, inside the
others' acute range. So the subacute column correlates PS95's near-acute maps against PS94's
chronic-range maps and asks why they disagree. **Acute replicates because it is aligned** -- every
animal starts at day 1 -- not because acute is biologically special.

WHAT THIS SCRIPT DOES. It re-bins the SAME maps by POST-STROKE DAY instead of by epoch label, and
recomputes the identical statistic and null. If day-matching lifts subacute toward acute, the
non-replication was an artefact of the binning. If it stays flat, subacute is genuinely
heterogeneous and the bin was not the problem.

PS94 MUST NOT BE FOLDED INTO A DAY-BINNED CHRONIC, and an earlier version of this docstring said
the opposite. It claimed a day window would "recover" PS94 for chronic, since PS94 has no chronic
bin and its days 11-29 sit in subacute. **That was wrong, and measured to be wrong** (Priya,
2026-09-17, asking whether PS94 was simply plateauing later). PS94 fails the plateau test on DRIFT,
not on residual: across every candidate tail the residual passes (0.074-0.084 against a 0.091
tolerance) while the fitted change does not -- the points sit tightly around a line and the line is
still RISING. PS94 is genuinely still recovering through day 29, while PS92, PS93 and PS95 all
plateaued at day 11.

So a day-binned chronic containing PS94 would put a still-recovering animal in the plateau bin --
the SAME error as subacute, moved one bin over. It is excluded here, which makes the chronic window
a POSITIVE CONTROL instead: for the three animals whose behavioural chronic already starts at day
11, binning by day must reproduce binning by epoch. If it does not, the re-binning machinery is
wrong and the subacute result cannot be trusted either.

FOR THE RECORD, since it decides when PS94 gains a chronic epoch: its last two sessions are the
first at baseline (day 25 at 1.06, day 29 at 1.02), but a tail starting at day 25 holds n = 2
against `min_tail = 3` and cannot qualify at any flatness. ONE more session at >= 0.996 of
pre-stroke baseline sets `chronic_from = 25` -- not the date of that session, because `flat_mode:
drift` tests total change across the window and is deliberately insensitive to spacing. Below 0.996
it stays None.

THIS IS A DIAGNOSTIC, NOT A PROPOSED REDEFINITION. The behavioural chronic criterion exists because
animals recover on their own schedules, and replacing it with calendar days would discard that --
the same reasoning that put it there. What follows from a positive result is narrower: **do not
compare the residual bin BETWEEN animals**, and report PS94 separately rather than as a subacute
member.

THE GATE AND THE CLASSIFIER are inherited, not rebuilt: every map here comes from
`position_reference_maps.maps_by_epoch`, which applies the engagement gate (2026-09-16) and reaches
the repaired classifier through `trial_features_cached`. That is why this module is NOT on
`tests/test_rest_engagement.py::EXPOSED` -- that list is for analyses that COLLECT rest frames
themselves and must therefore build the gate in their own source. `rest_null` sits outside it for
the same reason. `test_subacute_daymatched.py` asserts the read-through instead.

RUN AS:  python -m scripts.rest_migration.subacute_daymatched
"""
from __future__ import annotations

import csv
import itertools
import pathlib
import time

import numpy as np

ALIGN, VARIANT = "cue", "working"
N_NULL = 2000
SEED = 0

# (label, lo_day, hi_day, exclude) -- day bounds INCLUSIVE, `exclude` is animals this window must
# not contain. The two subacute windows are the ones under test; acute is the like-for-like
# reference (it already replicates at ~0.82, so re-binning must not break it); chronic is a
# POSITIVE CONTROL on three animals whose behavioural chronic is day 11 anyway.
WINDOWS = (
    ("acute d1-7", 1, 7, ()),
    ("subacute d5-9", 5, 9, ()),
    ("subacute d7-9", 7, 9, ()),
    # PS94 EXCLUDED, and this is the whole reason the chronic row is a control rather than a
    # result -- see the docstring. It is still recovering through day 29; including it would put a
    # rising animal in the plateau bin.
    ("chronic d11-29 [PS94 excl]", 11, 29, ("PS94",)),
)


def _z(v):
    """Mean-removed and unit-norm, so a dot product IS Pearson r. Same as `rest_null`."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return None
    v = np.where(ok, v, 0.0) - v[ok].mean()
    n = float(np.linalg.norm(v))
    return None if n == 0 else v / n


def _day_of(an, label):
    from wfield_local.grant_figures import _day

    try:
        d = _day(an, label.split("_")[-1])
    except Exception:                                                  # noqa: BLE001
        return None
    return None if d is None else int(d)


def _pre(store, ref, q):
    """Each animal's mean PRE map for one position -- the subtrahend, unchanged from `rest_null`."""
    out = {}
    for an, by_e in store.items():
        v = [m[ref] for m in ((by_e.get("pre") or {}).get(q) or {}).values() if ref in m]
        if v:
            out[an] = np.mean(v, axis=0)
    return out


def _window(store, ref, q, lo, hi, exclude=()):
    """``({animal: mean map}, {animal: n_sessions})`` over POST days in ``[lo, hi]``.

    Pooled across the epoch keys deliberately: the whole point is to ignore the residual bin and
    select on day, so a session counts if its day is in range regardless of what it was labelled.

    ``exclude`` drops whole animals, which is NOT a day filter and must not be mistaken for one:
    PS94 is excluded from the chronic window because it has not plateaued, not because its sessions
    fall outside the range. See the module docstring.
    """
    out, n = {}, {}
    for an, by_e in store.items():
        if an in exclude:
            continue
        maps = []
        for e, by_q in by_e.items():
            if e == "pre":
                continue
            for lab, m in (by_q.get(q) or {}).items():
                if ref not in m:
                    continue
                d = _day_of(an, lab)
                if d is not None and lo <= d <= hi:
                    maps.append(m[ref])
        if maps:
            out[an], n[an] = np.mean(maps, axis=0), len(maps)
    return out, n


def _agree(pre, post, mask, rng):
    """``(observed r, null mean, p, animals)`` for one (reference, window).

    The null keeps every animal's REAL delta map and lets each independently draw one of the six
    POSITIONS. Anatomy, warp, rim and shared optical artefacts survive untouched; only "the animals
    changed in the same POSITION-SPECIFIC way" is destroyed. Identical to `rest_null`'s null, so
    the two tables are directly comparable.
    """
    vecs, index = [], {}
    for q in post:
        for a in sorted(set(pre[q]) & set(post[q])):
            z = _z((post[q][a] - pre[q][a])[mask])
            if z is not None:
                index[(a, q)] = len(vecs)
                vecs.append(z)
    if not vecs:
        return None
    R = np.asarray(vecs) @ np.asarray(vecs).T
    animals = sorted({a for a, _q in index})
    rows = {}
    for q in post:
        ans = [a for a in animals if (a, q) in index]
        if len(ans) < 2:
            continue
        pairs = list(itertools.combinations(range(len(ans)), 2))
        obs = float(np.mean([R[index[(ans[i], q)], index[(ans[j], q)]] for i, j in pairs]))
        pool = {a: [p for p in post if (a, p) in index] for a in ans}
        nulls = []
        for _ in range(N_NULL):
            pick = [index[(a, pool[a][rng.integers(len(pool[a]))])] for a in ans]
            nulls.append(float(np.mean([R[pick[i], pick[j]] for i, j in pairs])))
        rows[q] = (obs, float(np.mean(nulls)), float(np.mean([x >= obs for x in nulls])), ans)
    return rows


def main() -> int:
    from wfield_local import beta_maps as bm
    from wfield_local import position_reference_maps as prm
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.paths import PathResolver

    t0 = time.time()
    mask = bm.stat_mask()
    rng = np.random.default_rng(SEED)
    print(f"[daymatch] stat mask {int(mask.sum())} px; loading maps ({ALIGN}/{VARIANT})", flush=True)
    store, _rel, _ntr = prm.maps_by_epoch(ALIGN, VARIANT)
    print(f"[daymatch]   .. {len(store)} animals, {time.time() - t0:.0f}s", flush=True)

    rows = []
    for ref in prm.REFERENCES:
        pre = {q: _pre(store, ref, q) for q in CONF_LABELS}
        for lab, lo, hi, excl in WINDOWS:
            post, ns = {}, {}
            for q in CONF_LABELS:
                post[q], ns[q] = _window(store, ref, q, lo, hi, excl)
            got = _agree(pre, post, mask, rng)
            if not got:
                continue
            for q, (obs, nm, p, ans) in got.items():
                rows.append({
                    "reference": ref, "window": lab, "lo_day": lo, "hi_day": hi,
                    "excluded": "|".join(excl), "position": q,
                    "n_animals": len(ans), "animals": "|".join(ans),
                    "n_sessions": "|".join(f"{a}:{ns[q].get(a, 0)}" for a in ans),
                    "mean_r": round(obs, 4), "null_mean": round(nm, 4), "null_p": round(p, 4),
                })
        print(f"  .. {ref} done ({time.time() - t0:.0f}s)", flush=True)

    out = pathlib.Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"
    p = out / "epoch_15x_subacute_daymatched.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[daymatch] wrote {p}", flush=True)

    for ref in prm.REFERENCES:
        print(f"\n== {ref} ==   mean pairwise between-animal r  (null mean, p)")
        print(f"{'window':<18}{'n_an':>5}  " + "".join(f"{q:>20}" for q in CONF_LABELS))
        for lab, _lo, _hi, _x in WINDOWS:
            sub = [r for r in rows if r["reference"] == ref and r["window"] == lab]
            if not sub:
                continue
            na = max(r["n_animals"] for r in sub)
            cells = []
            for q in CONF_LABELS:
                m = [r for r in sub if r["position"] == q]
                cells.append(f"{m[0]['mean_r']:>9.3f} ({m[0]['null_mean']:>5.2f})" if m
                             else f"{'--':>20}")
            print(f"{lab:<18}{na:>5}  " + "".join(cells))
        # THE HEADLINE IS THE MEAN OVER POSITIONS, because the claim under test is about the
        # WINDOW, not about any one position -- reading a single position here would repeat the
        # cherry-pick the per-position table exists to make visible.
        print(f"{'':<18}{'':>5}  mean over positions:")
        for lab, _lo, _hi, _x in WINDOWS:
            sub = [r for r in rows if r["reference"] == ref and r["window"] == lab]
            if sub:
                print(f"   {lab:<18} r={np.mean([r['mean_r'] for r in sub]):+.3f}  "
                      f"null={np.mean([r['null_mean'] for r in sub]):+.3f}  "
                      f"n_animals={max(r['n_animals'] for r in sub)}")
    print(f"\n[daymatch] {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

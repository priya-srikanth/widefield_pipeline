"""False-positive calibration of the map significance tests on PRE-vs-PRE comparisons (Priya 2026-10-21: "please
carefully review the stats code"; "should we also do this on all pre vs august pre?").

The map families (epoch_maps 14 / 14z / 15 / 15r, within-spout repo tests) draw `beta_maps.significance_contour`:
the nested animals -> sessions bootstrap with a max-statistic family-wise threshold (PRIMARY) and the within-animal
relabelling cluster permutation (second line). With n = 4 animals the bootstrap may be anti-conservative (a bootstrap
over 4 units underestimates the sampling spread; only 35 distinct animal multisets). This script measures it on
comparisons with NO stroke, on figure 14's beta maps (cue / working by default):

  N1  August pre split in two (alternating sessions by date)        pure null, short gap
  N2  ALL pre (June + August) split in two                          pure null, longer span
  N3  June pre vs August pre                                        DRIFT null: ~2 months apart, no stroke
  REAL  August pre vs acute                                          for scale

--family position (Priya 10-21: "i meant the imaging-only analyses based on spout position, of ENL vs cue vs
lick-aligned"): the same on family 15r's POSITION ACTIVITY maps (`position_reference_maps.maps_by_epoch`), one CSV per
reference (raw / rest / restw / precue / mean); arms precue/working (ENL), cue/working, lick/lick.

Each with the nested bootstrap and the cluster permutation, on RAW maps and on PRE-NORMALISED maps (each animal divided
by its own pre-group RMS, `beta_maps.NORMALISE_BY_PRE`). Reported: fraction of tested bins (nested) / in-mask pixels
(cluster) flagged. A valid FWER test flags a non-empty set in <= 5 % of null comparisons; read N1 / N2 that way. N3
measures how much "change" two months of drift alone produces.

Output: <labcams>/grant_figures/epoch/results/map_test_calibration_<align>_<variant>.csv; maps cached in the scratch
pickle given as --cache.
Usage: PYTHONPATH=. python scripts/map_test_calibration.py --cache <file.pkl> [--align cue --variant working]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _split(labels):
    s = sorted(labels)
    return s[0::2], s[1::2]


def _rms_scale(pre_by, mask):
    out = {}
    for a, v in pre_by.items():
        m = np.mean(v, 0)
        r = float(np.sqrt(np.nanmean(m[mask] ** 2)))
        out[a] = r if r > 0 else 1.0
    return out


def _one(job):
    from wfield_local import beta_maps as bm
    name, q, pre_by, post_by, norm, n_perm = job
    mask = bm.brain_mask()
    if norm:
        sc = _rms_scale(pre_by, mask)
        pre_by = {a: [m / sc[a] for m in v] for a, v in pre_by.items()}
        post_by = {a: [m / sc[a] for m in v] for a, v in post_by.items()}
    got = bm.hierarchical_bootstrap_significance(pre_by, post_by, n_boot=2000)
    nb = (int(got[0].sum()), int(got[4])) if got is not None else (0, 0)
    keep = bm.cluster_permutation(pre_by, post_by, n_perm=n_perm)
    sm = bm.stat_mask()
    nc = int((keep & sm).sum()) if keep is not None else 0
    return {"comparison": name, "position": q, "norm": "pre-normalised" if norm else "raw",
            "n_animals": len(set(pre_by) & set(post_by)),
            "n_sessions": sum(len(v) for v in pre_by.values()) + sum(len(v) for v in post_by.values()),
            "nested_bins": nb[0], "nested_tested": nb[1], "nested_frac": nb[0] / nb[1] if nb[1] else np.nan,
            "cluster_px": nc, "cluster_frac": nc / int(sm.sum())}


def main(argv=None) -> int:
    from wfield_local import beta_maps as bm
    from wfield_local import parallel
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", default="cue")
    ap.add_argument("--variant", default="working")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--family", choices=("beta", "position"), default="beta")
    ap.add_argument("--references", default="raw,rest,restw,precue,mean", help="position family only")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    if a.cache.exists():
        full = pickle.loads(a.cache.read_bytes())
    else:
        if a.family == "beta":
            full, _rel, _ntr = bm.maps_by_epoch(a.align, a.variant)
        else:
            from wfield_local import position_reference_maps as prm
            full, _rel, _ntr = prm.maps_by_epoch(a.align, a.variant)
        a.cache.write_bytes(pickle.dumps(full, protocol=pickle.HIGHEST_PROTOCOL))
    if a.family == "beta":
        stores = {"beta": full}
    else:                                 # {an: {e: {q: {label: {ref: map}}}}} -> one store per reference
        stores = {r: {an: {e: {q: {lab: d[r] for lab, d in labs.items() if r in d} for q, labs in bq.items()}
                           for e, bq in be.items()} for an, be in full.items()}
                  for r in a.references.split(",")}
    rc = 0
    for tag, store in stores.items():
        rc |= run(store, tag, a, CONF_LABELS, parallel, PathResolver, assert_writable)
    return rc


def run(store, tag, a, CONF_LABELS, parallel, PathResolver, assert_writable) -> int:
    print(f"\n#### {a.family} / {tag} / {a.align}_{a.variant}", flush=True)
    jobs = []
    for q in CONF_LABELS:
        pre = {an: (by.get("pre") or {}).get(q) or {} for an, by in store.items()}
        acute = {an: list(((by.get("acute") or {}).get(q) or {}).values()) for an, by in store.items()}
        aug = {an: {k: v for k, v in d.items() if k[-4:] >= "0801"} for an, d in pre.items()}
        jun = {an: {k: v for k, v in d.items() if k[-4:] < "0801"} for an, d in pre.items()}
        comps = {}
        n1 = {an: _split(d) for an, d in aug.items() if len(d) >= 4}
        comps["N1 aug split"] = ({an: [aug[an][k] for k in s[0]] for an, s in n1.items()},
                                 {an: [aug[an][k] for k in s[1]] for an, s in n1.items()})
        n2 = {an: _split(d) for an, d in pre.items() if len(d) >= 4}
        comps["N2 all-pre split"] = ({an: [pre[an][k] for k in s[0]] for an, s in n2.items()},
                                     {an: [pre[an][k] for k in s[1]] for an, s in n2.items()})
        both = [an for an in pre if jun.get(an) and aug.get(an)]
        comps["N3 june vs aug"] = ({an: list(jun[an].values()) for an in both},
                                   {an: list(aug[an].values()) for an in both})
        comps["REAL aug pre vs acute"] = ({an: list(aug[an].values()) for an in aug if aug[an] and acute.get(an)},
                                          {an: acute[an] for an in aug if aug[an] and acute.get(an)})
        for name, (p, o) in comps.items():
            if len(set(p) & set(o)) >= 2:
                f32 = lambda d: {an: [np.asarray(m, np.float32) for m in v] for an, v in d.items()}  # noqa: E731
                for norm in (False, True):
                    jobs.append((name, q, f32(p), f32(o), norm, a.n_perm))
    print(f"{len(jobs)} tests over {a.jobs} workers", flush=True)
    res, fail = parallel.fan_out(jobs, _one, jobs=a.jobs, label="test",
                                 log=lambda m: print(m[:90], flush=True))   # items hold maps: short log
    res = sorted(res, key=lambda kv: (kv[1]["comparison"], kv[1]["position"], kv[1]["norm"]))  # rule 9: defined order
    for it, err in fail:
        print("FAILED", it[0], it[1], it[4], err)
    D = pd.DataFrame([r for _, r in res]).sort_values(["comparison", "position", "norm"])
    out = Path(PathResolver().root("labcams")) / "grant_figures" / "epoch" / "results"
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / (f"map_test_calibration_{a.align}_{a.variant}.csv" if a.family == "beta"
               else f"map_test_calibration_position_{tag}_{a.align}_{a.variant}.csv")
    D.to_csv(f, index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(D.pivot_table(index=["comparison", "position"], columns="norm",
                        values=["nested_frac", "cluster_frac"]).round(3).to_string())
    print(f"\n-> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

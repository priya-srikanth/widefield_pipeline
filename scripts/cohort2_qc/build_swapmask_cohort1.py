"""Build `hemo_meegkit_hpfit_lag_refitT_swapmask` for EVERY curated cohort-1 session (2026-11-02 re-run,
docs/status/COHORT1_SWAPMASK_RERUN_20261102.md in wf_attempt). Same call as `cohort1_impact.py`:

    hv.write(s, "meegkit_hpfit_lag", refit_t=True, func=func, bad_pairs=bad, tag="_swapmask")

with ``func = hv.functional_channel(s)`` (PS92_0828 is 0) and ``bad`` from the 10-30 sweep's `swapped_pairs.npz`
(never re-detected or re-written here). A session whose variant already exists is REUSED only if its manifest
matches (variant, refit, lag shift, functional channel, interpolated-pair count); a mismatch is REPORTED, never
overwritten. Cohort 1 only (PS92-95): cohort 2 is refused.

New `hemo_*` directory only; `SVTcorr.npy` / `T.npy` / `rcoeffs.npy` / every other variant stay untouched
(`hv.write` writes a temp dir then renames; `writeguard.assert_writable` refuses anything outside the Priya subtree).

    python scripts/cohort2_qc/build_swapmask_cohort1.py [--jobs 4] [--only PS93_0826 ...] [--check-only]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

VARIANT = "meegkit_hpfit_lag"
TAG = "_swapmask"
NEW = f"hemo_{VARIANT}_refitT{TAG}"
COHORT1 = ("PS92", "PS93", "PS94", "PS95")


def _session(label):
    from wfield_local import analysis_kit as ak
    return {s["label"]: s for s in ak.curated_sessions()}[label]


def check(s):
    """(status, detail) for an existing build: 'ok' | 'missing' | 'mismatch'."""
    from wfield_local import hemo_variants as hv
    res = Path(s["mc"]) / "wfield_local_results"
    man = res / NEW / "manifest.json"
    if not man.is_file() or not (res / NEW / "SVTcorr.npy").is_file():
        return "missing", ""
    m = json.loads(man.read_text())
    bad = np.load(res / "swapped_pairs.npz")["bad"]
    n_svt = np.load(res / "SVT.npy", mmap_mode="r").shape[1] // 2
    want = dict(variant=VARIANT, refit_t=True, iso_shift_16ms=True, functional_channel=hv.functional_channel(s),
                bad_pairs_interpolated=int(np.asarray(bad, bool)[:n_svt].sum()), label=s["label"])
    diff = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    return ("ok", m.get("built_utc", "")) if not diff else ("mismatch", json.dumps(diff))


def worker(label):
    from wfield_local import hemo_variants as hv
    s = _session(label)
    st, det = check(s)
    if st != "missing":
        return dict(label=label, status="reused" if st == "ok" else "MISMATCH", detail=det)
    res = Path(s["mc"]) / "wfield_local_results"
    t0 = time.time()
    bad = np.load(res / "swapped_pairs.npz")["bad"]
    func = hv.functional_channel(s)
    hv.write(s, VARIANT, refit_t=True, func=func, bad_pairs=bad, tag=TAG, verbose=False)
    st, det = check(s)
    return dict(label=label, status="built" if st == "ok" else f"BUILT-BUT-{st}", detail=det,
                func=func, bad=int(bad.sum()), s=round(time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args()
    from wfield_local import analysis_kit as ak
    labels = [s["label"] for s in ak.curated_sessions()]
    bad = [lab for lab in labels if not lab.startswith(COHORT1)]
    if bad:
        raise SystemExit(f"refusing: non-cohort-1 sessions in the curated set {bad[:5]}")
    if a.only:
        labels = [lab for lab in labels if lab in a.only]
    if a.check_only:
        ss = {s["label"]: s for s in ak.curated_sessions()}
        for lab in labels:
            print(lab, *check(ss[lab]), flush=True)
        return
    rows, fails = ak.fan_sessions(labels, worker, jobs=a.jobs)
    out = [v for _k, v in rows] + [dict(label=k, status="FAILED", detail=str(e)[-400:]) for k, e in fails]
    for r in out:
        print(json.dumps(r), flush=True)
    from collections import Counter
    summ = f"SUMMARY {dict(Counter(r['status'] for r in out))} of {len(labels)}"
    print(summ, flush=True)
    if a.log:
        a.log.write_text("\n".join(json.dumps(r) for r in out) + "\n" + summ + "\n")


if __name__ == "__main__":
    main()

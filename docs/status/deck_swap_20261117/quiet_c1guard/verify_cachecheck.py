"""Quit-time cache check (Priya 11-18 "yes" to the caching refactor): every table of the cache-check chain must equal the
production sync-end run EXACTLY. Compares every .csv / .csv.gz / .npz under both run folders (logs, figures and the
restyled/ subfolder skipped -- figures carry render metadata; restyled/ is not produced by the chain). Byte-identical
files pass; otherwise the tables are read and compared cell by cell with NO tolerance (rule 9: a tolerance would pass
the bug this guards against). Prints one line per differing / missing file and a summary; exit 1 on any difference.
usage: python verify_cachecheck.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
Q = Path(f"M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/{T}/cohort1_quiet_spikes")
A = Q / "v2_quiet_lp_c1guard_syncend_20261118"
B = Q / "v2_quiet_lp_c1guard_syncend_cachecheck_20261118"
SKIP = {"logs", "restyled_20261118", "svg"}


def files(root):
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file() and not SKIP & set(p.relative_to(root).parts)
            and (p.suffix in (".csv", ".npz") or p.name.endswith(".csv.gz"))}


def same(a, b):
    if a.read_bytes() == b.read_bytes():
        return True, "bytes"
    if a.suffix == ".npz":
        x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
        if sorted(x.files) != sorted(y.files):
            return False, "keys differ"
        bad = [k for k in x.files if not np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f")]
        return not bad, f"arrays differ: {bad[:5]}" if bad else "arrays equal"
    x, y = pd.read_csv(a), pd.read_csv(b)
    try:
        pd.testing.assert_frame_equal(x, y, check_exact=True)
        return True, "cells equal"
    except AssertionError as e:
        return False, str(e).splitlines()[0][:160]


def main():
    fa, fb = files(A), files(B)
    bad = 0
    for p in sorted(fa ^ fb):
        print("MISSING in", "check" if p in fa else "production", p)
        bad += 1
    how = {}
    for p in sorted(fa & fb):
        ok, why = same(A / p, B / p)
        how[why if ok else "DIFF"] = how.get(why if ok else "DIFF", 0) + 1
        if not ok:
            print("DIFF", p, "|", why)
            bad += 1
    print(f"{len(fa & fb)} common files: {how}; {len(fa ^ fb)} missing; {bad} problems")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

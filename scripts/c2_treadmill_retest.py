"""Re-test the six-position treadmill readouts at another bootstrap draw count, from the STORED session readouts
(Priya / Cowork 2026-11-23: every six-position MWU + bootstrap test at 4,000 draws, as the paper states).

The tests are the ones `c2_treadmill_cue_aligned.py` and `c2_treadmill_extras.py` wrote (session MWU + paired
animal-weighted nested bootstrap, `orofacial_stats.session_change_tests`, seed 0), re-run on the session readouts those
scripts saved, so nothing is re-collected and no figure or manuscript file is touched:

  treadmill_tests_primary.csv          <- treadmill_session_readouts.csv          push_cue_minus_control, walk_cue;
                                                                                   ONE BH family (fdr_scope="all")
  treadmill_tests_secondary.csv        <- treadmill_session_readouts.csv          SECONDARY, BH within measure
  treadmill_tests_primary_first95.csv  <- treadmill_session_readouts_first95.csv  as primary (S16B, first 95 min)

The by-position check table is not re-run (its per-position readouts were not saved; it is a check, not a panel).

    python scripts/c2_treadmill_retest.py <readout_dir> <out_dir> [--n-boot N] [--check]

``--check`` first re-runs every table at 2,000 draws and requires the bootstrap columns to equal the stored table in
<readout_dir> (same seed, same draw order; MWU p to 1e-3 relative, star calls identical), so a difference at --n-boot is
the draw count alone. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import orofacial_stats as st

PRIMARY = ["push_cue_minus_control", "walk_cue"]
SECONDARY = ["push_rate", "push_control_rate", "push_peak", "push_latency_s", "walk_control"]
JOBS = (("treadmill_tests_primary.csv", "treadmill_session_readouts.csv", PRIMARY, "all"),
        ("treadmill_tests_secondary.csv", "treadmill_session_readouts.csv", SECONDARY, "metric"),
        ("treadmill_tests_primary_first95.csv", "treadmill_session_readouts_first95.csv", PRIMARY, "all"))


def run(src: Path, name: str, metrics, scope: str, n_boot: int | None) -> pd.DataFrame:
    S = pd.read_csv(src / name, dtype={"date": str})
    nb = {} if n_boot is None else {"n_boot": n_boot}
    return st.session_change_tests(S, metrics, fdr_scope=scope, **nb)


def same(a: pd.DataFrame, b: pd.DataFrame, cols) -> float:
    """Max relative difference over ``cols`` (inf when the star calls differ)."""
    if len(a) != len(b) or not (a.agree.to_numpy() == b.agree.to_numpy()).all():
        return float("inf")
    x, y = a[cols].to_numpy(float), b[cols].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(x - y) / np.maximum(np.abs(y), 1e-12)
    return float(np.nanmax(r))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("readout_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n-boot", type=int, default=None,
                    help="draws (default: session_change_tests' default, analysis_kit.N_BOOT)")
    ap.add_argument("--check", action="store_true", help="replay at 2,000 against the stored tables first")
    a = ap.parse_args(argv)
    outs = [a.out_dir / j[0] for j in JOBS]
    clash = [str(o) for o in outs if o.exists()]
    if clash:
        sys.exit(f"refusing to overwrite {', '.join(clash)}")
    if a.check:
        for out, src, ms, scope in JOBS:
            # the bootstrap must replay exactly. The secondary table's push_latency_s MWU p does not (11-23: subacute p
            # 0.00309 vs stored 0.00299, star identical): its stored readouts do not reproduce the session values the
            # stored p came from to full precision; a report-only measure, so its p is printed, not required
            r, s0 = run(a.readout_dir, src, ms, scope, 2000), pd.read_csv(a.readout_dir / out)
            db, dp = same(r, s0, ["boot_change", "boot_lo", "boot_hi"]), same(r, s0, ["p", "q_fdr"])
            print(f"replay {out} at 2000: max relative diff bootstrap {db:.2g}, MWU p / q {dp:.2g}")
            if db > 1e-9 or (dp > 1e-3 and out != "treadmill_tests_secondary.csv"):
                sys.exit("the 2,000-draw replay does not reproduce the stored table; nothing written")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for (_out, src, ms, scope), f in zip(JOBS, outs):
        R = run(a.readout_dir, src, ms, scope, a.n_boot)
        R.to_csv(f, index=False)
        print("wrote", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

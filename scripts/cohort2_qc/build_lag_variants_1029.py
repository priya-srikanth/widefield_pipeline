"""Build the lag-fix / fit-band hemo CANDIDATES for the 10-29 sessions, swap-masked (2026-10-30, §0g of the QC doc).
Writes hemo_<variant>_refitT_swapmask/ per session; the adopted product is untouched.

    conda activate locanmf
    python scripts/cohort2_qc/build_lag_variants_1029.py [--variants ...] [--only PS121 ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from swapmask_1029 import DAQ, ROOT, SESS  # noqa: E402

VARIANTS = ["meegkit_hpfit_lag", "meegkit_lag_fit1to8", "meegkit_lag_fitheart"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    from wfield_local import hemo_variants as hv
    for animal, (sdir, h5) in SESS.items():
        if a.only and animal not in a.only:
            continue
        mc = ROOT / sdir / "motion_corrected"
        bad = np.load(mc / "wfield_local_results" / "swapped_pairs.npz")["bad"]
        s = dict(label=f"{animal}_1029", mc=str(mc), h5=str(DAQ / h5), regime="B", fmdir=None)
        for v in a.variants:
            hv.write(s, v, refit_t=True, func=1, bad_pairs=bad, tag="_swapmask")


if __name__ == "__main__":
    main()

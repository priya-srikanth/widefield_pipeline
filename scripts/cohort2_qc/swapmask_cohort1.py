"""Cohort1 sweep: swapped-frame mask + swap-masked hemo variant for every registered cohort1 session (Priya
2026-10-30: "let's set up the cohort 1 sweep"). Same detectors and build as `swapmask_1029.py`; channel order per
session (`hemo_variants.functional_channel`, PS92_0828 is 0).

COMPARISON VARIANT ONLY: writes `swapped_pairs.npz` and `hemo_meegkit_hpfit_refitT_swapmask/` beside each session's
results. The adopted `hemo_meegkit_hpfit` is untouched, and nothing reads the new variant until a comparison says so.
Regime-A sessions (legacy frame map, no strobedetrend mask) get the mask only. Resumable: a session whose variant
manifest exists is skipped. One line per session goes to --log (CSV-ish), plus a SUMMARY.

    conda activate locanmf
    python scripts/cohort2_qc/swapmask_cohort1.py [--only PS93_0826 ...] [--mask-only]
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from swapmask_1029 import detect  # noqa: E402

LOG = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029/swapmask_cohort1_log.txt")
TAG = "_swapmask"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--mask-only", action="store_true")
    ap.add_argument("--log", type=Path, default=LOG)
    a = ap.parse_args()
    from wfield_local import config
    from wfield_local import hemo_variants as hv
    ss = sorted(config.load_sessions(machine="analysis_desktop"), key=lambda s: s["label"])
    if a.only:
        ss = [s for s in ss if s["label"] in a.only]
    done = skipped = failed = 0
    for s in ss:
        res = Path(s["mc"]) / "wfield_local_results"
        dest = hv.out_dir(s, hv.ADOPTED, True, TAG)
        if (dest / "manifest.json").is_file() and (res / "swapped_pairs.npz").is_file():
            skipped += 1
            continue
        t0 = time.time()
        try:
            func = hv.functional_channel(s)
            m = detect(res, func)
            np.savez(res / "swapped_pairs.npz", **m)
            built = False
            if not a.mask_only and s["regime"] == "B":
                hv.write(s, hv.ADOPTED, refit_t=True, func=func, bad_pairs=m["bad"], tag=TAG, verbose=False)
                built = True
            line = dict(label=s["label"], regime=s["regime"], func=func, n_pairs=int(m["bad"].size),
                        core=int(m["core"].sum()), bad=int(m["bad"].sum()), built=built, s=round(time.time() - t0))
            done += 1
        except Exception as e:                     # one bad session must not stop the night
            line = dict(label=s["label"], error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-400:])
            failed += 1
        print(json.dumps(line), flush=True)
        with open(a.log, "a") as fh:
            fh.write(json.dumps(line) + "\n")
    summ = f"SUMMARY done {done} skipped(already) {skipped} failed {failed} of {len(ss)}"
    print(summ, flush=True)
    with open(a.log, "a") as fh:
        fh.write(summ + "\n")


if __name__ == "__main__":
    main()

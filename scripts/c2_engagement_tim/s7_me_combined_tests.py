"""Step 7: S15B six-position motion energy (keypoint + FaceRhythm ME averaged) from an s4 ``sessions_motion.csv``, and
its tests at 4,000 draws. Same rule as the 11-22 export (s22_compare part 2, `movement_combined.py` six-position
branch without TIM): per session, first 95 min, state "all", me_all_<w> = mean(me_kp_<w>, me_fr_all_<w>) (skipna False)
for w = response, iti; tests = `orofacial_stats.session_change_tests(fdr_scope="all")` (session change from the animal's
own pre mean; session MWU + paired animal-weighted nested bootstrap, `analysis_kit.N_BOOT` = 4,000 draws, seed 0; one
BH family over the two metrics x three epochs); star = q < 0.05 AND the bootstrap interval excludes 0. Also the
per-modality metrics (me_kp_*, me_fr_all_*) in their own family, for the report only.

Usage: python s7_me_combined_tests.py <s4_out_dir> <tag>
  -> <s4_out_dir>/movement_combined_me_c2_sessions_<tag>.csv, movement_combined_me_c2_tests_<tag>.csv,
     me_components_tests_<tag>.csv (refuses to overwrite)
"""
import sys
from pathlib import Path

import pandas as pd

from wfield_local import analysis_kit as ak
from wfield_local import orofacial_stats as st

OUT, TAG = Path(sys.argv[1]), sys.argv[2]
KEEP = ["label", "animal", "epoch", "date", "pos", "me_kp_response", "me_kp_iti", "me_fr_all_response", "me_fr_all_iti",
        "me_all_response", "me_all_iti"]


def write(name, df):
    f = OUT / name
    if f.exists():
        sys.exit(f"refusing to overwrite {f}")
    df.to_csv(f, index=False)
    print("wrote", f, len(df))


def tests(M, metrics):
    r = st.session_change_tests(M, metrics, fdr_scope="all", n_boot=ak.N_BOOT)
    r["star"] = (r.q_fdr < 0.05) & ((r.boot_lo > 0) | (r.boot_hi < 0))
    r["n_boot"] = ak.N_BOOT
    return r


M = pd.read_csv(OUT / "sessions_motion.csv", dtype={"date": str})
M = M[(M["match"] == "first95") & (M.state == "all")].copy()
for per in ("response", "iti"):
    M[f"me_all_{per}"] = M[[f"me_kp_{per}", f"me_fr_all_{per}"]].mean(axis=1, skipna=False)
M["pos"] = "all"
write(f"movement_combined_me_c2_sessions_{TAG}.csv", M[KEEP])
write(f"movement_combined_me_c2_tests_{TAG}.csv", tests(M, ["me_all_response", "me_all_iti"]))
write(f"me_components_tests_{TAG}.csv",
      tests(M, ["me_kp_response", "me_kp_iti", "me_fr_all_response", "me_fr_all_iti"]))

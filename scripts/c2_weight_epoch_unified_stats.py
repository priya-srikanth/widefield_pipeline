"""Six-position cohort (PS92-95): body weight by recovery epoch with the behavior-panel statistics (manuscript Fig 3F).

VLS manuscript, 2026-11-23 (Priya: one statistics rule for weight and behavior). Same animal-days, % of pre-stroke and
epochs as `c2_outcome_weight_epoch_stats.weights` (the panel's data); the test is the rule of the six-position behavior
panels (Fig 4A-E), `orofacial_stats.session_change_tests`, with one animal-day as the session:
  value = each day's % of pre-stroke weight minus the animal's own pre-stroke mean (= pct - 100; `deltas_given`);
  session Mann-Whitney U, post-epoch days vs pre-stroke days; BH-FDR over the three epochs only (the weight panel's own
  family); paired, animal-weighted animals -> days bootstrap (`analysis_kit.boot_delta`) with N_BOOT = 4000 resamples
  passed explicitly (session_change_tests defaults to 2000); '**' = both pass (`agree == "both"`); the behavior
  panels draw no single '*', so bootstrap-only cells are flagged (`boot_only`) but not starred.
The mixed-model table of `c2_outcome_weight_epoch_stats.weight_lmm` is joined alongside (`lmm_*`) for comparison.

Writes (refuses to overwrite) <out_dir>/c2_weight_epoch_unified.csv.
Usage: python scripts/c2_weight_epoch_unified_stats.py --out-dir DIR [--weights CSV]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import c2_outcome_weight_epoch_stats as cow  # noqa: E402
from wfield_local import orofacial_stats as ofs  # noqa: E402

N_BOOT = 4000


def unified(w):
    P = w.assign(pos="all", value=w.pct - w.animal.map(w[w.epoch == "pre"].groupby("animal").pct.mean()))
    P["date"] = P["date"].astype(str)
    # CLAUDE.md rule 8: the per-animal table before the bootstrap
    pa = P.groupby(["animal", "epoch"]).value.agg(["mean", "count"]).unstack("epoch")
    print("per-animal mean change (% of pre) and days:\n" + pa.round(2).to_string())
    st = ofs.session_change_tests(P, ["value"], seed=0, n_boot=N_BOOT, deltas_given=True)
    st["n_boot"] = N_BOOT
    st["boot_only"] = st.agree.eq("bootstrap only")
    st["star"] = np.where(st.agree.eq("both"), "**", "")
    lm = cow.weight_lmm(w).rename(columns={"diff_pct": "lmm_diff_pct", "ci_lo": "lmm_ci_lo", "ci_hi": "lmm_ci_hi",
                                           "p": "lmm_p", "q": "lmm_q", "stars": "lmm_stars"}).drop(columns="group")
    return st.merge(lm, on="epoch", how="left")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=cow.WEIGHTS_CSV)
    a = ap.parse_args(argv)
    out = a.out_dir / "c2_weight_epoch_unified.csv"
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    st = unified(cow.weights(a.weights, cow._bounds()))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    st.to_csv(out, index=False)
    print(st[["epoch", "n_animals", "n_sessions", "n_pre_sessions", "boot_change", "boot_lo", "boot_hi", "p", "q_fdr",
              "agree", "star", "lmm_q", "lmm_stars"]].round(4).to_string(index=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

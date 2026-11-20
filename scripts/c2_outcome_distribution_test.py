"""Six-position cohort trial outcomes (manuscript Fig 3B, earlier 4B): ONE change-in-distribution test per stacked bar.

Replaces the per-class tests of `scripts/c2_outcome_stats_aligned.py` with one test per position x post epoch
(Priya 2026-11-20). Same trials, filters and session x position class fractions (`load_outcomes` + the aligned
script's `session_fractions`; classes complete / incomplete / jaw_only / no_resp). Test
(`wfield_local.composition_test`): per animal TVD between its epoch-mean and pre-mean composition; D = animal-weighted
mean TVD; within-animal permutation of the pre / epoch session labels (exact when <= 20,000 relabellings, else 10,000
random, seed 0); BH-FDR over the 18 position x epoch cells; star_dist = q < 0.05. Per-class changes (d_<class>) are
descriptive only.

Writes (refuses to overwrite) under --out-dir (default M:/MICROSCOPE/Priya/Widefield/manuscript_stats):
  c2_trial_outcome_distribution_test.csv             one row per position x epoch
  c2_trial_outcome_distribution_test_per_animal.csv  per-animal TVD (and pre / epoch mean compositions)
  c2_trial_outcome_distribution_test_vs_aligned.csv  each cell next to the per-class aligned stars (any / which)

Usage: python scripts/c2_outcome_distribution_test.py [--out-dir DIR] [--classes CSV.GZ]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from c2_outcome_stats_aligned import session_fractions  # noqa: E402
from c2_outcome_weight_epoch_stats import CLASSES_CSV, CLS, OUT_DIR, _bounds, load_outcomes  # noqa: E402

from wfield_local import composition_test as ct  # noqa: E402

POST = ["acute", "subacute", "chronic"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--classes", type=Path, default=CLASSES_CSV)
    a = ap.parse_args(argv)
    base = a.out_dir / "c2_trial_outcome_distribution_test"
    outs = [Path(f"{base}.csv"), Path(f"{base}_per_animal.csv"), Path(f"{base}_vs_aligned.csv")]
    clash = [str(p) for p in outs if p.exists()]
    if clash:
        sys.exit(f"refusing to overwrite {', '.join(clash)}")
    P = session_fractions(load_outcomes(a.classes, _bounds()))
    cells, per = ct.distribution_tests(P, CLS, ["pos"], POST)
    cells = cells.rename(columns={"pos": "position"})
    per = per.rename(columns={"pos": "position"})
    print(per[["position", "epoch", "animal", "tvd", "n_pre_sessions", "n_epoch_sessions"]].round(3).to_string(index=False))
    al_p = a.out_dir / "c2_trial_outcome_stats_aligned.csv"
    al = pd.read_csv(al_p)
    st = (al[al.star_aligned == 1].groupby(["position", "epoch"]).cls.apply(lambda s: ";".join(s)).rename("aligned_classes"))
    C = cells[["position", "epoch", "star_dist", "D", "p", "p_min", "q", "n_animals"]].merge(
        st.reset_index(), on=["position", "epoch"], how="left")
    C["aligned_classes"] = C.aligned_classes.fillna("")
    C["any_aligned_star"] = (C.aligned_classes != "").astype(int)
    C["agree"] = (C.any_aligned_star == C.star_dist).astype(int)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(outs[0], index=False)
    per.to_csv(outs[1], index=False)
    C.to_csv(outs[2], index=False)
    print(C.round(4).to_string(index=False))
    print("distribution stars:", int(C.star_dist.sum()), "| bars with >= 1 aligned star:", int(C.any_aligned_star.sum()),
          f"(aligned stars {int(al.star_aligned.sum())}) | disagreements:", int((C.agree == 0).sum()))
    for p in outs:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

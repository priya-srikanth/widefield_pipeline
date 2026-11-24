"""Six-position cohort trial outcomes (manuscript Fig 4B) under the cohort-statistics ALIGNED rule (Priya 2026-11-20:
"yes" to re-testing Fig 4B with the rule every other cohort-2 and cohort-1 panel uses).

Same trials and per-session class fractions as `scripts/c2_outcome_weight_epoch_stats.py` (`load_outcomes`: working
period, August pre from 0806, PS92/93 0817 excluded, epochs from config.epoch_spec). Each session x position fraction of
each class (complete / incomplete / jaw_only / no_resp) is a metric of `orofacial_stats.session_change_tests`:
session change from the animal's own pre mean; session Mann-Whitney U (post vs pre sessions); paired animal-weighted
nested bootstrap `analysis_kit.boot_delta` (seed 0; 2,000 draws until 2026-11-23, now the session_change_tests
default 4,000, `--n-boot` to set); BH-FDR as ONE family over the 4 classes x 6 positions
x 3 post epochs (fdr_scope="all"; the per-class q is kept as q_fdr_metric). Star (drawn '**', as everywhere else) =
family q < 0.05 AND the bootstrap 95% CI excludes 0 (agree == "both").

Writes (refuses to overwrite) under --out-dir (default M:/MICROSCOPE/Priya/Widefield/manuscript_stats):
  c2_trial_outcome_stats_aligned.csv          one row per class x position x epoch
  c2_trial_outcome_stats_aligned_vs_old.csv   the same rows next to the current Fig 4B star (c2_trial_outcome_boot_stats.csv)

Usage: python scripts/c2_outcome_stats_aligned.py [--out-dir DIR] [--classes CSV.GZ] [--n-boot N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from c2_outcome_weight_epoch_stats import CLASSES_CSV, CLS, OUT_DIR, _bounds, load_outcomes  # noqa: E402

from wfield_local import orofacial_stats as st  # noqa: E402


def session_fractions(c2: pd.DataFrame) -> pd.DataFrame:
    """One row per session x position: fraction of each class (columns CLS) + animal, date, epoch, pos."""
    f = (c2.groupby(["label", "animal", "mmdd", "epoch", "position"]).cls.value_counts(normalize=True)
         .unstack().reindex(columns=CLS).fillna(0).reset_index())
    return f.rename(columns={"mmdd": "date", "position": "pos"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--classes", type=Path, default=CLASSES_CSV)
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap draws (default: orofacial_stats.session_change_tests' default = analysis_kit.N_BOOT, 4,000 since 2026-11-23; 2000 reproduces tables written before then; same seed)")
    a = ap.parse_args(argv)
    out = a.out_dir / "c2_trial_outcome_stats_aligned.csv"
    cmp_ = a.out_dir / "c2_trial_outcome_stats_aligned_vs_old.csv"
    clash = [str(p) for p in (out, cmp_) if p.exists()]
    if clash:
        sys.exit(f"refusing to overwrite {', '.join(clash)}")
    P = session_fractions(load_outcomes(a.classes, _bounds()))
    nb = {} if a.n_boot is None else {"n_boot": a.n_boot}
    T = st.session_change_tests(P, CLS, fdr_scope="all", **nb)
    T["star_aligned"] = (T.agree == "both").astype(int)
    T = T.rename(columns={"metric": "cls", "pos": "position"})
    T.to_csv(out, index=False)
    old_p = a.out_dir / "c2_trial_outcome_boot_stats.csv"
    if old_p.exists():
        old = pd.read_csv(old_p, keep_default_na=False)[["position", "epoch", "cls", "delta", "lo", "hi", "star"]]
        old = old.rename(columns={"delta": "old_delta", "lo": "old_lo", "hi": "old_hi", "star": "old_star"})
        C = T.merge(old, on=["position", "epoch", "cls"], how="outer")
        C["change"] = [("same" if (o != "") == bool(n) else ("gained" if n else "lost")) for o, n in
                       zip(C.old_star.fillna(""), C.star_aligned.fillna(0))]
        C.to_csv(cmp_, index=False)
        print(C[(C.old_star.fillna("") != "") | (C.star_aligned == 1)][
            ["cls", "position", "epoch", "old_delta", "old_star", "boot_change", "boot_lo", "boot_hi", "p", "q_fdr",
             "agree", "change"]].round(4).to_string(index=False))
        print("old stars:", int((C.old_star.fillna("") != "").sum()), "(** :", int((C.old_star == "**").sum()),
              ") -> aligned:", int(C.star_aligned.sum()))
    print("wrote", out, cmp_)
    return 0


if __name__ == "__main__":
    sys.exit(main())

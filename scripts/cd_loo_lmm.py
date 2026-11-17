"""Section 13 CD leave-one-out mixed models on a given cd_loo long table (the imaging_lmm.epoch_tables transform,
source epoch:cd_loo only, FDR within source as imaging_lmm). argv: cd_loo csv, out csv, profile|none, deficits..."""
import sys
from pathlib import Path
import wfield_local
print("wfield_local:", wfield_local.__file__, flush=True)
src, out, prof, *defs = sys.argv[1:]
if prof != "none":
    from wfield_local import render_profile
    render_profile.apply(prof)
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(wfield_local.__file__).parents[1] / "scripts"))
from wfield_local import mixed_models as mm, orofacial_cohort as oc, run_mode as _rm
R = pd.read_csv(src, dtype={"family": str})
pos = R.position.fillna("").astype(str).map(lambda p: oc.pos_key(p, "L") if p.split("_")[-1] in ("L", "R", "center") else p)
R = R.assign(source="epoch:" + R.family, y=R.value,
             measure=R.measure + " | " + R.arm + " / " + R.variant.astype(str) + np.where(pos != "", " | " + pos, ""))
F = R.dropna(subset=["y"])[["label", "source", "measure", "y"]]
F = F[F.source == "epoch:cd_loo"]
outs = []
for d in defs:
    qc = _rm.deficit_qc_dir(d)
    print("deficit", d, "from", qc, flush=True)
    Le = mm.add_deficit(mm.add_time_covariates(F), mm.deficit(qc, d))
    if d == defs[0]:
        print(mm.animal_epoch_table(Le[Le.measure.str.contains("far_contra")], ["measure"]).round(3).to_string())
    T = mm.lmm_table(Le, ["source", "measure"], ("time", "behaviour", "epoch"), fdr_by=("source",), print_animals=False)
    outs.append(T.assign(deficit=d))
T = pd.concat(outs, ignore_index=True)
T.to_csv(out, index=False, float_format="%.5g")
fc = T[T.measure.str.contains("far_contra") & ~T.measure.str.contains(r"lick \[")]
pd.set_option("display.width", 250)
print(fc[["deficit", "model", "measure", "term", "coef", "z", "p", "q", "n"]].to_string())

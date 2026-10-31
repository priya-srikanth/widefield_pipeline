"""Calcium in the 8s 415? Split the 415 lick response into VESSEL and PARENCHYMA pixels (2026-10-31).

Priya 10-31: for jGCaMP8s the 415 signal also RISES with calcium (below its isosbestic point, from the spectra), so
the 415 regression would subtract real calcium. Sign cannot separate that from hemodynamics (both in phase,
iso_phase.py); space can. Calcium in 415 is parenchymal and follows the 470 calcium hotspots; the hemodynamic /
vascular part is strongest ON vessels. Per session, first-lick response (lick 0-0.5 s minus pre-cue 1 s), raw 470 and
raw 415 dF/F:
  vessel mask   = brain pixels >= 10% darker in the 470 mean image than its local mean (gaussian sigma 10 px)
  parenchyma    = brain pixels within 2% of / brighter than the local mean, not within 3 px of a vessel
  hotspot       = top 2% of the corrected lick map, parenchyma only
Reports 415/470 on vessels, in parenchyma and at the parenchymal hotspot. Calcium in 415 predicts a parenchymal /
hotspot 415/470 ratio in RiboL1 ABOVE Thy1-6s (415 just past 6s's isosbestic point, little calcium) and above the
vessel ratio; hemodynamics predicts vessel > parenchyma in every group.

    conda activate locanmf
    python scripts/cohort2_qc/iso_vessel_split.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from indicator_kinetics_spread import IND, OUT, VAR, events, sessions  # noqa: E402


def one(s):
    from wfield_local import hemo_variants as hv
    res = Path(s["mc"]) / "wfield_local_results"
    func = s["func"]
    fa = np.load(res / "frames_average.npy")
    br = brain_mask(fa, func, res=res)
    im = np.asarray(fa[func], np.float64)
    w = gaussian_filter(br.astype(float), 10)
    loc = gaussian_filter(np.where(br, im, 0), 10) / np.maximum(w, 1e-6)
    rel = im / np.maximum(loc, 1e-6)
    ves = br & (rel <= 0.90)
    par = br & (rel >= 0.98) & ~binary_dilation(ves, iterations=3)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Uf = U.reshape(-1, U.shape[-1]).astype(np.float64)
    del U
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    C = np.load(res / VAR / "SVTcorr.npy").astype(np.float64)
    n = C.shape[1]
    X = {"470": np.asarray(SVT[:, func::2][:, :n], np.float64), "415": np.asarray(SVT[:, 1 - func::2][:, :n], np.float64),
         "corr": C}
    if s["cohort"] == "cohort1" and (res / "swapped_pairs.npz").exists():
        bad = np.load(res / "swapped_pairs.npz")["bad"][:n]
        X = {k: hv._interp_pairs(v, bad) for k, v in X.items()}
    cf, fl, _ = events(s, n)
    resp = {k: (M[:, fl[:, None] + np.arange(0, 16)].mean(2) - M[:, cf[:, None] + np.arange(-31, 0)].mean(2)).mean(1)
            for k, M in X.items()}
    pv, vv = par.reshape(-1), ves.reshape(-1)
    mc = Uf[pv] @ resp["corr"]
    hot_idx = np.flatnonzero(pv)[mc >= np.percentile(mc, 98)]
    out = dict(label=s["label"], indicator=IND[s["animal"]], vessel_frac=float(ves.sum() / br.sum()),
               par_frac=float(par.sum() / br.sum()))
    for nm, sel in (("vessel", np.flatnonzero(vv)), ("parenchyma", np.flatnonzero(pv)), ("hot", hot_idx)):
        u = Uf[sel].mean(0)
        for k in X:
            out[f"{nm}_{k}"] = float(100 * u @ resp[k])
        out[f"{nm}_ratio"] = out[f"{nm}_415"] / out[f"{nm}_470"]
    out["vessel_over_par_415"] = out["vessel_415"] / out["parenchyma_415"] if out["parenchyma_415"] else float("nan")
    m415, m470 = Uf[pv] @ resp["415"], Uf[pv] @ resp["470"]
    out["r_par_415_470"] = float(np.corrcoef(m415, m470)[0, 1])
    return out


def main():
    from wfield_local.parallel import fan_out
    res, fail = fan_out(sessions(), one, jobs=4, label="session")
    if fail:
        print("FAILED", fail)
    R = sorted((r for _, r in res), key=lambda r: (r["indicator"], r["label"]))
    (OUT / "iso_vessel_split.json").write_text(json.dumps(R, indent=1))
    print(f"{'session':11s} ind  ves%  | vessel 470  415  ratio | parench 470  415  ratio | hot 470  415  ratio "
          f"| ves/par 415 | r(415,470) par")
    for r in R:
        print(f"{r['label']:11s} {r['indicator']:4s} {100 * r['vessel_frac']:4.1f} | "
              + " | ".join(f"{r[f'{k}_470']:6.2f} {r[f'{k}_415']:5.2f} {r[f'{k}_ratio']:5.2f}"
                           for k in ("vessel", "parenchyma", "hot"))
              + f" | {r['vessel_over_par_415']:6.2f} | {r['r_par_415_470']:.2f}")


if __name__ == "__main__":
    main()

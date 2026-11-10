"""Frozen vs refit vs gap, LocaNMF vs cohort SVD, ONE pipeline (Priya 11-09: "how does an svd frozen vs refit vs gap
scatter compare to our locanmf frozen vs refit vs gap scatter").

Same estimator and design as the page's size-matched family (5rm / 5rmo, page Fig 4.4 `frozen_refit_pooled`), applied
to two feature sets so only the BASIS differs in the comparison:
  frozen  `locanmf_frozen_decoder._pipe` (standardize + multinomial logistic, C = 0.5) trained on the animal's pre
          sessions (the scored pre session left out), on a SIZE-MATCHED random set of whole blocks = 4/5 of the scored
          session (what the refit trains on), as `grant_confusion._matched_frozen`; scores every trial of the session
  refit   the same estimator, within-session GroupKFold(5) over blocks (`grant_confusion._refit_pred`, no floor)
  gap     refit - frozen, paired on the session's trials
Inputs = the two arms of `aligned_decoder.py` (nothing re-extracted):
  LocaNMF the 5r / 5rm LICK arm's refit inputs (joint-LocaNMF components x 8 bins of 2 s after the first lick, LP licks +
          LP gate, August pre; `aligned_decoder` capture)
  SVD     the cohort SVD state-space extract (100 shared components x 8 bins, -0.1..+0.7 s after the first lick,
          first-lick trials; blocks = runs of one spout)
NOT the page figure's window: Fig 4.4 is CUE 0-2 s, WORKING trials (incl. misses); the SVD extract has no miss trials,
so both bases are shown in the lick window here. Stats / layout = `rebuild_figs/fig_frozen_refit_pooled` (boot_delta
animals -> sessions, * 95 % CI, ** Bonferroni x 9; sessions open markers, animal mean +- SEM).
    python scripts/frozen_refit_basis_compare.py [--jobs 4]
Out: <state_space_recovery>/frozen_refit_basis/{sessions.csv, stats.csv, frozen_refit_<basis>.png}. EXPLORATORY.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rebuild_figs"))

LNMF_INPUTS = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures/LP_licks_gateLP_hv-meegkit_hpfit_"
                   "lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt"
                   "3500c70_daqlk_inmem-epoch/epoch/data/aligned_decoder/inputs_lick_lick.pkl")


def job(unit) -> dict | None:
    from sklearn.model_selection import GroupKFold, cross_val_predict

    from wfield_local.locanmf_frozen_decoder import _pipe
    S, pres = unit
    y, blk = np.asarray(S["y"]).astype(int), np.asarray(S["blk"])
    ng = np.unique(blk).size
    if ng < 2 or len(np.unique(y)) < 2:
        return None
    k = min(5, ng)
    refit = cross_val_predict(_pipe(), S["X"], y, cv=GroupKFold(k), groups=blk)
    tr = [P for P in pres if P["label"] != S["label"]]
    Xp = np.concatenate([P["X"] for P in tr])
    yp = np.concatenate([P["y"] for P in tr]).astype(int)
    bp = np.concatenate([np.asarray(P["blk"]).astype(np.int64) + i * 10**12 for i, P in enumerate(tr)])
    n_target = int(round(len(y) * (k - 1) / k))
    rng = np.random.default_rng(zlib.crc32(S["label"].encode()))
    keep = np.zeros(len(yp), bool)
    for b in rng.permutation(np.unique(bp)):
        keep |= bp == b
        if keep.sum() >= n_target:
            break
    frozen = _pipe().fit(Xp[keep], yp[keep]).predict(S["X"])
    fa, ra = float(np.mean(frozen == y)), float(np.mean(refit == y))
    return dict(animal=S["animal"], label=S["label"], epoch=S["epoch"], n=len(y), n_train=int(keep.sum()),
                frozen=fa, refit=ra, gap=ra - fa)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    os.environ["WF_SS_BASIS"], os.environ["WF_SS_ALIGN"] = "svd", "lick"
    import aligned_decoder as ad
    import fig_frozen_refit_pooled as F
    import state_space_sim as S

    from wfield_local import analysis_kit as ak
    out = S.ROOT / "frozen_refit_basis"
    out.mkdir(parents=True, exist_ok=True)
    with open(LNMF_INPUTS, "rb") as fh:
        lnmf = pickle.load(fh)
    inputs = {"lnmf": [c for c in lnmf if c.get("epoch") in ad.EPOCHS], "svd": ad.svd_inputs()}
    rows = []
    for basis, cap in inputs.items():
        units = []
        for an in sorted({c["animal"] for c in cap}):
            mine = [c for c in cap if c["animal"] == an]
            units += [(c, [p for p in mine if p["pre"]]) for c in mine]
        got, fail = ak.fan_sessions(units, job, jobs=a.jobs, key=lambda u: u[0]["label"])
        for u, err in fail:
            print(f"  FAILED {basis} {u[0]['label']}: {err}", flush=True)
        rows += [dict(basis=basis, **r) for _u, r in got if r]
    D = pd.DataFrame(rows)
    D.to_csv(out / "sessions.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    allst = []
    for basis, g in D.groupby("basis"):
        print(f"\n== {basis}: per-animal session means (lick window, size-matched frozen)")
        print(g.groupby(["animal", "epoch"])[["frozen", "refit", "gap"]].mean()
              .reindex(F.EPOCHS, level=1).round(3).to_string())
        w = g.assign(session_idx=g.groupby(["epoch", "animal"]).cumcount())[
            ["epoch", "animal", "session_idx", "frozen", "refit", "gap"]].reset_index(drop=True)
        st, lv = F.stats(w), F.level_table(w)
        print(st[["panel", "epoch", "delta_vs_pre", "lo95", "hi95", "mark", "n_animals_up"]].round(3).to_string(index=False))
        allst.append(st.assign(basis=basis))
        F.plot(w, st, lv, out / f"frozen_refit_{basis}.png")
    pd.concat(allst).to_csv(out / "stats.csv", index=False, float_format="%.4g")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

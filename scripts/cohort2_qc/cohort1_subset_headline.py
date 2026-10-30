"""Cohort1 epoch-subset check: do epoch-level results move with the swap-masked + 415-lag-fixed correction?
(Priya 2026-10-30: "can we do this with a subset of sessions"). Tier 1 of the plan in QC doc §0i: no LocaNMF.

Subset (24): per animal 2 pre (August), 1 acute, 1 subacute, 2 chronic (`wfield_local.epochs`). For each session and
each correction -- ADOPTED `hemo_meegkit_hpfit` vs NEW `hemo_meegkit_hpfit_lag_refitT_swapmask` (cohort1_impact.py
builds it) -- the pipeline's own spout-position decoder (`locanmf_position_decoder._trial_features`, source=roi =
Allen-area means of SVTcorr, cue-aligned, defaults.yaml decode settings; same 5-fold CV and C=0.5 logistic) gives:
  bal_acc   balanced accuracy, all areas
  amp       mean |trial feature| per session (all areas), and for MO (MOp+MOs) and SSp
Then, per animal, epoch minus pre for each, ADOPTED vs NEW. The variant is selected with WIDEFIELD_HEMO_VARIANT, which
`config.svtcorr_path` reads at call time; roi features are never disk-cached (`trial_features_cached`), so no stale
features can leak between the two.

    conda activate locanmf
    python scripts/cohort2_qc/cohort1_subset_headline.py
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029")
SUBSET = {"PS92": {"pre": ["0811", "0813"], "acute": ["0820"], "subacute": ["0826"], "chronic": ["0911", "0925"]},
          "PS93": {"pre": ["0812", "0814"], "acute": ["0820"], "subacute": ["0826"], "chronic": ["0911", "0925"]},
          "PS94": {"pre": ["0811", "0813"], "acute": ["0820"], "subacute": ["0827"], "chronic": ["0910", "0924"]},
          "PS95": {"pre": ["0811", "0813"], "acute": ["0817"], "subacute": ["0823"], "chronic": ["0914", "0928"]}}
VARS = {"adopted": "meegkit_hpfit", "new": "meegkit_hpfit_lag_refitT_swapmask"}


def decoder_args():
    from wfield_local import config
    dp = config.defaults()["decode"]
    return argparse.Namespace(align="cue", baseline=dp["baseline"], cv=dp["cv"], fs=31.23, pre_s=1.0,
                              post_s=dp["cue_post_s"], max_rt=dp["max_rt_s"], bins=None, source="roi",
                              per_session=False, date=None, output=None)


def one(s, args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from wfield_local import locanmf_position_decoder as lpd
    from wfield_local import nolick_analysis as na
    X, y, g, _Xnl, _ynl, reg = lpd._trial_features(s, args)
    names = {int(k): v for k, v in json.load(open(glob.glob(
        f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    if args.cv == "block":
        pred = cross_val_predict(clf, X, y, cv=GroupKFold(min(5, int(np.unique(g).size))), groups=g)
    else:
        pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    out = dict(n=int(X.shape[0]), bal_acc=float(na.balanced_accuracy(y, pred, labels=lpd.DISPLAY_ORDER)),
               amp=float(np.mean(np.abs(X))))
    for grp, prefs in (("MO", ("MOp", "MOs")), ("SSp", ("SSp",))):
        cols = [i for i in range(X.shape[1]) if any(names.get(int(reg[i]), "").startswith(p) for p in prefs)]
        out[f"amp_{grp}"] = float(np.mean(np.abs(X[:, cols]))) if cols else float("nan")
    return out


def main():
    from wfield_local import config
    ss = {s["label"]: s for s in config.load_sessions(machine="analysis_desktop")}
    args = decoder_args()
    rows = []
    for animal, eps in SUBSET.items():
        for ep, dates in eps.items():
            for d in dates:
                s = ss[f"{animal}_{d}"]
                r = dict(label=s["label"], animal=animal, epoch=ep)
                for vn, v in VARS.items():
                    if not Path(config.svtcorr_path(s["mc"], v)).is_file():
                        r[vn] = None
                        continue
                    os.environ["WIDEFIELD_HEMO_VARIANT"] = v
                    r[vn] = one(s, args)
                rows.append(r)
                print(json.dumps(r), flush=True)
    os.environ.pop("WIDEFIELD_HEMO_VARIANT", None)
    (OUT / "cohort1_subset_headline.json").write_text(json.dumps(rows, indent=1))
    # epoch - pre, per animal, per metric, per variant
    print("\nper-animal epoch minus pre (adopted -> new):")
    for m in ("bal_acc", "amp", "amp_MO", "amp_SSp"):
        for animal in SUBSET:
            line = []
            for ep in ("acute", "subacute", "chronic"):
                vals = {}
                for vn in VARS:
                    pre = [r[vn][m] for r in rows if r["animal"] == animal and r["epoch"] == "pre" and r[vn]]
                    e = [r[vn][m] for r in rows if r["animal"] == animal and r["epoch"] == ep and r[vn]]
                    vals[vn] = np.mean(e) - np.mean(pre) if pre and e else np.nan
                rel = "" if m == "bal_acc" else "  (rel to pre)"
                line.append(f"{ep} {vals['adopted']:+.3f} -> {vals['new']:+.3f}{rel}")
            print(f"  {m:8s} {animal}: " + " | ".join(line))


if __name__ == "__main__":
    main()

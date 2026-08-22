"""Is the corrected pre-cue position decoding significantly above chance? (Priya, 2026-08-13)

After correcting the zero-phase-filter artifact the pre-cue effect is roughly HALF its published size
(0.486 -> 0.306 under `strobedetrend`, chance 0.167), so "above chance" stops being obvious and has to
be tested rather than eyeballed. Two complementary statements, deliberately not conflated:

  CLUSTER BOOTSTRAP (decode_ci.bootstrap_recall) -- "how precisely do we know THIS session's accuracy".
      Resamples BLOCKS with replacement, not trials: the task presents positions in runs of ~6, so
      trial-level resampling would treat ~6 correlated observations as independent and give intervals
      far too narrow. The model is held fixed; this is precision of the estimate, not a null test.

  BLOCK-LABEL PERMUTATION -- "could this accuracy arise with NO position information at all".
      Permutes position labels BETWEEN BLOCKS, keeping each block's trials together and refitting the
      whole cross-validated decoder each time. Permuting individual trials would destroy the block
      structure and give a null that is far too tight, making everything look significant. The unit of
      exchangeability is the block, because that is the unit the task randomises.

Both are run on the CORRECTED data (default `strobedetrend`) because the artifact-laden numbers are not
worth testing.

    python -m wfield_local.precue_significance --variant strobedetrend --n-perm 200 --output <json>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config, hemo_variants as hv
from wfield_local.decode_ci import bootstrap_recall
from wfield_local.filter_acausality_test import roi_signal
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER

CHANCE = 1.0 / 6.0
FS = 31.23


from wfield_local.locanmf_frozen_decoder import (  # ONE decoder spec
    _pipe)


def _cv_predict(X, y, g):
    ng = min(5, int(np.unique(g).size))
    if ng < 2:
        return None
    return cross_val_predict(_pipe(), X, y, cv=GroupKFold(ng), groups=g)


def permute_block_labels(y, g, rng):
    """Reassign each BLOCK's position to a shuffled block order, keeping blocks intact.

    Each block has one position by construction, so permuting the block->position map is exactly the
    null "the activity in a block is unrelated to which position that block was". Trial-level shuffling
    would additionally destroy the within-block correlation and understate the null.
    """
    blocks = np.unique(g)
    lab = np.array([y[g == b][0] for b in blocks])
    perm = rng.permutation(lab)
    out = np.empty_like(y)
    for b, p in zip(blocks, perm):
        out[g == b] = p
    return out


def analyse(lab, variant="strobedetrend", align="precue", n_perm=200, seed=0, verbose=True):
    s = next((x for x in SESSIONS if x["label"] == lab), None)
    if s is None:
        return None
    ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
    if not ad:
        return None
    svtc, _T, _rc, _meta = hv.compute(s, variant, refit_t=True, verbose=False)
    sig, regs = roi_signal(ad[0], svtc)
    args = SimpleNamespace(source="roi", align=align, baseline="none",
                           pre_s=1.0, post_s=2.0, fs=FS, max_rt=float(config.defaults()["decode"]["max_rt_s"]))
    X, y, g, _, _, _ = _trial_features(s, args, signal=sig, feat_region=regs)
    del sig, svtc
    if len(y) < 60 or len(np.unique(y)) < len(DISPLAY_ORDER):
        return None

    pred = _cv_predict(X, y, g)
    if pred is None:
        return None
    acc = float(accuracy_score(y, pred))
    ci = bootstrap_recall(y, pred, blocks=g)

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = permute_block_labels(y, g, rng)
        pp = _cv_predict(X, yp, g)
        if pp is not None:
            null.append(accuracy_score(yp, pp))
    null = np.asarray(null, float)
    # +1 correction: a permutation p can never be 0 with finite resamples
    p = float((1 + (null >= acc).sum()) / (1 + null.size)) if null.size else float("nan")
    out = {"label": lab, "variant": variant, "align": align, "n_trials": int(len(y)),
           "n_blocks": int(np.unique(g).size), "accuracy": acc,
           "ci_lo": float(ci["accuracy_ci"][0]), "ci_hi": float(ci["accuracy_ci"][1]),
           "null_mean": float(null.mean()) if null.size else None,
           "null_p95": float(np.percentile(null, 95)) if null.size else None,
           "p_perm": p, "n_perm": int(null.size)}
    if verbose:
        print(f"  {lab:12s} acc {acc:.3f}  CI[{out['ci_lo']:.3f},{out['ci_hi']:.3f}]  "
              f"null {out['null_mean']:.3f} (p95 {out['null_p95']:.3f})  p={p:.4f}  "
              f"n={out['n_trials']} in {out['n_blocks']} blocks", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="strobedetrend", choices=sorted(hv.VARIANTS))
    ap.add_argument("--align", default="precue", choices=("precue", "cue"))
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--from", dest="from_dates", default=None)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    dates = (set(config.expand_dates(a.from_dates, width=4)) if a.from_dates
             else set(config.curated_dates()))
    only = config.normalize_animals(a.animals) or sorted({x["label"][:4] for x in SESSIONS})
    labs = [x["label"] for x in SESSIONS if x["label"][:4] in set(only) and x["label"][-4:] in dates]
    print(f"[significance] {len(labs)} sessions, {a.n_perm} block-label permutations each, "
          f"variant={a.variant} align={a.align} chance={CHANCE:.3f}", flush=True)

    rows = []
    for lab in labs:
        try:
            r = analyse(lab, a.variant, a.align, a.n_perm)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:80]}", flush=True)
            continue
        if r:
            rows.append(r)
    if not rows:
        return 1

    print(f"\n=== {len(rows)} sessions, {a.variant} {a.align} (chance {CHANCE:.3f}) ===")
    for an in sorted({r["label"][:4] for r in rows}):
        rr = [r for r in rows if r["label"].startswith(an)]
        acc = np.array([r["accuracy"] for r in rr])
        lo = np.array([r["ci_lo"] for r in rr], float)
        sig_ci = int((lo > CHANCE).sum())
        sig_p = int(sum(r["p_perm"] < 0.05 for r in rr))
        print(f"  {an}  mean acc {acc.mean():.3f}   CI lower bound > chance in {sig_ci}/{len(rr)}   "
              f"permutation p<0.05 in {sig_p}/{len(rr)}   null mean "
              f"{np.mean([r['null_mean'] for r in rr]):.3f}")
    allp = [r["p_perm"] for r in rows]
    lo = np.array([r["ci_lo"] for r in rows], float)
    print(f"  ALL  CI lower bound > chance in {int((lo > CHANCE).sum())}/{len(rows)}   "
          f"p<0.05 in {int(sum(p < 0.05 for p in allp))}/{len(rows)}")
    if a.output:
        Path(a.output).write_text(json.dumps(rows, indent=2, default=float))
        print(f"\nwrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

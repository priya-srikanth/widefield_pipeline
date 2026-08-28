"""Does extending the POST-CUE decode window past 2 s help? (Priya, 2026-08-13)

Motivation is specific: PS93's far-L responses are often late, so a 2 s window may cut them off. That
predicts a PER-POSITION effect (far_L recall improves, others roughly flat), not a uniform gain — and
the two are worth distinguishing, because a uniform gain would just mean "more averaging is better" and
would apply to every animal rather than fixing the thing that motivated it.

So this reports per-position RECALL as well as overall accuracy, per animal, at several window lengths,
on the ADOPTED preprocessing variant. Everything else is the pipeline's own decoder: engaged trials,
Allen-ROI features, block-aware GroupKFold, no per-trial baseline.

CAUTION ON INTERPRETATION. A longer post-cue window is not free: it reaches further into licking and
consumption, so a gain could be movement rather than a better-captured position response. That does not
make it wrong — post-cue decoding is expected to be movement-rich — but "3 s decodes better" should not
be read as "the position code lasts 3 s". The lick-aligned analyses are the place to separate those.

    python -m wfield_local.postcue_window_test [--windows 2.0 2.5 3.0] [--from 0806-0812]
"""
from __future__ import annotations

import argparse
import glob
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from wfield_local import config, hemo_variants as hv
from wfield_local.filter_acausality_test import roi_signal
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES
from wfield_local.precue_significance import FS, _cv_predict

WINDOWS = [2.0, 2.5, 3.0, 3.5]



_ANNOUNCED = False


def _max_rt() -> float:
    """The engaged cut, from `decode.max_rt_s`, announced once so a result carries its own boundary.

    FOUND 2026-08-28 by `tests/test_engaged_cut_comes_from_config.py`, which walks the AST rather
    than grepping -- the by-hand survey that listed the other three sweeps missed this one entirely.
    Same defect: the cut was pinned at 2.0 s while `decode.max_rt_s` moved to 3.5 s on 2026-08-21,
    so a lick at 2.5 s was filed as "no lick" when the task scored it as a REWARDED HIT.

    Especially wrong HERE. This module sweeps the post-cue WINDOW over [2.0, 2.5, 3.0, 3.5] s, so at
    a 2.0 s engaged cut the longer windows were being scored on trials selected by the shortest one
    -- the window under test and the trials it was tested on disagreed by up to 1.5 s.

    Numbers recorded for this module in DECISIONS were measured at the 2.0 s cut.
    """
    global _ANNOUNCED
    v = float(config.defaults()["decode"]["max_rt_s"])
    if not _ANNOUNCED:
        print(f"[{__name__.rsplit('.', 1)[-1]}] engaged cut = {v:g}s (decode.max_rt_s). Numbers "
              f"recorded in DECISIONS for this module were measured at 2.0s.", flush=True)
        _ANNOUNCED = True
    return v


def one(s, sig, regs, post_s, align="cue"):
    args = SimpleNamespace(source="roi", align=align, baseline="none",
                           pre_s=1.0, post_s=post_s, fs=FS, max_rt=_max_rt())
    X, y, g, _, _, _ = _trial_features(s, args, signal=sig, feat_region=regs)
    if len(y) < 60 or len(np.unique(y)) < len(DISPLAY_ORDER):
        return None
    p = _cv_predict(X, y, g)
    if p is None:
        return None
    cm = confusion_matrix(y, p, labels=DISPLAY_ORDER).astype(float)
    rec = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(len(DISPLAY_ORDER)), where=cm.sum(1) > 0)
    return {"acc": float(accuracy_score(y, p)), "recall": rec, "n": int(len(y))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", nargs="+", type=float, default=WINDOWS)
    ap.add_argument("--variant", default="meegkit_hpfit")
    ap.add_argument("--align", default="cue", choices=("cue", "lick", "precue"),
                    help="LICK alignment follows the response wherever it lands, so it handles "
                         "variable latency by construction -- the right instrument if the concern is "
                         "late responses. A longer CUE-aligned window is the wrong one: it adds "
                         "mostly-noise to every trial to catch a few late ones.")
    ap.add_argument("--from", dest="from_dates", default=None)
    ap.add_argument("--animals", nargs="+", default=None)
    a = ap.parse_args(argv)

    dates = (set(config.expand_dates(a.from_dates, width=4)) if a.from_dates
             else set(config.curated_dates()))
    only = config.normalize_animals(a.animals) or sorted({x["label"][:4] for x in SESSIONS})
    labs = [x["label"] for x in SESSIONS if x["label"][:4] in set(only) and x["label"][-4:] in dates]
    names = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    print(f"[window-test] {len(labs)} sessions x {len(a.windows)} windows, align={a.align}, "
          f"variant={a.variant}", flush=True)

    rows = {}
    for lab in labs:
        s = next(x for x in SESSIONS if x["label"] == lab)
        if not glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1"):
            continue
        try:
            svtc, _T, _rc, _m = hv.compute(s, a.variant, refit_t=False, verbose=False)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            continue
        sig, regs = roi_signal(glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")[0],
                               svtc)
        del svtc
        got = {}
        for w in a.windows:
            r = one(s, sig, regs, w, a.align)
            if r:
                got[w] = r
        del sig
        if got:
            rows[lab] = got
            print("  {:12s} ".format(lab)
                  + "  ".join(f"{w:.1f}s={got[w]['acc']:.3f}" for w in a.windows if w in got),
                  flush=True)

    print(f"\n=== overall accuracy by animal (chance {1/6:.3f}) ===")
    print(f"{'animal':8s}" + "".join(f"{w:>9.1f}s" for w in a.windows))
    for an in sorted({l[:4] for l in rows}):
        rr = [v for l, v in rows.items() if l.startswith(an)]
        print(f"{an:8s}" + "".join(
            f"{np.nanmean([x[w]['acc'] for x in rr if w in x]):>10.3f}" for w in a.windows))

    print(f"\n=== PER-POSITION recall — does a longer window rescue the LATE positions? ===")
    for an in sorted({l[:4] for l in rows}):
        rr = [v for l, v in rows.items() if l.startswith(an)]
        print(f"  {an}")
        print(f"    {'position':12s}" + "".join(f"{w:>9.1f}s" for w in a.windows) + "     delta")
        for i, nm in enumerate(names):
            vals = [np.nanmean([x[w]["recall"][i] for x in rr if w in x]) for w in a.windows]
            print(f"    {nm:12s}" + "".join(f"{v:>10.3f}" for v in vals)
                  + f"   {vals[-1]-vals[0]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

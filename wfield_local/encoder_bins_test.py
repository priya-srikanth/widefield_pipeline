"""Does the ENCODER benefit from sub-binned targets, the way the decoder does?

The decoder adopted sub-binned features on 2026-08-14 (+0.032 pre-cue, +0.020 post-cue, +0.023
post-lick). The encoder runs the same feature builder, so it would have inherited them silently --
turning "predict each component's mean activity" into "predict its 8-bin time course" and changing
what R^2, EV and the ceiling mean. It was pinned to `bins=1` pending this test rather than left to
change by accident.

WHY THE COMPARISON NEEDS FEVE, NOT EV. More bins means more targets, each intrinsically noisier (a
0.25 s mean is noisier than a 2 s mean), so raw EV falls with bin count almost by construction and
would "prove" that sub-binning hurts. The noise CEILING -- between-position SS over total SS -- is the
most any position-only model could achieve for those targets, so FEVE = EV / ceiling asks the question
that actually matters: of the position-related structure PRESENT at this temporal resolution, how much
does the forward model capture? That is comparable across bin counts; EV is not.

ROI features deliberately: they read the adopted `meegkit_hpfit` SVTcorr already, so this needs no
LocaNMF refit and sits in the same basis the decoder sub-binning was measured in.

    python -m wfield_local.encoder_bins_test [--bins 1 4 8] [--align lick] [--from 0607,0806,0810,0811]
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from wfield_local.locanmf_frozen_decoder import _ceiling
from wfield_local.locanmf_position_decoder import _build_signal, _trial_features

ALPHA = 1.0


def _args(align, post_s, bins):
    from types import SimpleNamespace
    return SimpleNamespace(source="roi", align=align, baseline="none", pre_s=1.0,
                           post_s=post_s, fs=31.23, max_rt=2.0, bins=bins)


def encoder_scores(X, y, g, alpha=ALPHA):
    """(EV, ceiling, FEVE) for a one-hot position -> activity ridge under block CV.

    EV is pooled over features by SUM of squares rather than averaged per feature: a per-feature mean
    lets a handful of near-silent ROIs with tiny denominators dominate, which changes with bin count
    and would confound exactly the comparison being made.
    """
    pos = np.unique(y)
    P = np.stack([(y == p).astype(float) for p in pos], axis=1)
    ng = min(5, int(np.unique(g).size))
    if ng < 2:
        return np.nan, np.nan, np.nan
    pred = np.zeros_like(X, dtype=float)
    for tr, te in GroupKFold(ng).split(X, y, g):
        pred[te] = Ridge(alpha=alpha).fit(P[tr], X[tr]).predict(P[te])
    ss_res = float(((X - pred) ** 2).sum())
    ss_tot = float(((X - X.mean(0)) ** 2).sum())
    ev = 1.0 - ss_res / max(ss_tot, 1e-12)
    ceil, _, _ = _ceiling(X, y)
    return ev, ceil, (ev / ceil if ceil > 1e-6 else np.nan)


def run(labels, bins_list, align="lick", post_s=2.0, verbose=True):
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    rows = []
    for lab in labels:
        s = next((x for x in SESSIONS if x["label"] == lab), None)
        if s is None:
            print(f"  !! {lab}: not registered", flush=True)
            continue
        try:
            sig, reg = _build_signal(s, "roi")          # load ONCE, reuse for every bin count
        except Exception as ex:                          # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        rec = {"label": lab}
        for b in bins_list:
            try:
                X, y, g, _Xn, _yn, _r = _trial_features(s, _args(align, post_s, b),
                                                        signal=sig, feat_region=reg)
                ev, ceil, feve = encoder_scores(X, y, g)
                rec[f"ev{b}"], rec[f"ceil{b}"], rec[f"feve{b}"] = ev, ceil, feve
                rec[f"nfeat{b}"] = X.shape[1]
            except Exception as ex:                      # noqa: BLE001
                print(f"  !! {lab} bins={b}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
                rec[f"ev{b}"] = rec[f"ceil{b}"] = rec[f"feve{b}"] = np.nan
        rows.append(rec)
        if verbose:
            print("  " + f"{lab:12s} " + "  ".join(
                f"b{b}: EV={rec.get(f'ev{b}', np.nan):.3f} ceil={rec.get(f'ceil{b}', np.nan):.3f} "
                f"FEVE={rec.get(f'feve{b}', np.nan):.3f}" for b in bins_list), flush=True)
    return rows


def summarise(rows, bins_list):
    print(f"\n=== ENCODER sub-binning, {len(rows)} sessions (ROI features, meegkit_hpfit) ===")
    print(f"  {'arm':>6s} {'nfeat':>6s} {'EV':>8s} {'ceiling':>8s} {'FEVE':>8s}   vs bins=1")
    base = np.array([r.get("feve1", np.nan) for r in rows], float)
    for b in bins_list:
        ev = np.nanmean([r.get(f"ev{b}", np.nan) for r in rows])
        ce = np.nanmean([r.get(f"ceil{b}", np.nan) for r in rows])
        fv = np.array([r.get(f"feve{b}", np.nan) for r in rows], float)
        d = fv - base
        ok = np.isfinite(d)
        nf = int(np.nanmax([r.get(f"nfeat{b}", np.nan) for r in rows]))
        tag = ("" if b == 1 else
               f"  mean {np.nanmean(d):+.3f}  median {np.nanmedian(d):+.3f} "
               f"(better in {int((d[ok] > 0).sum())}/{int(ok.sum())})")
        print(f"  {b:>6d} {nf:>6d} {ev:>8.3f} {ce:>8.3f} {np.nanmean(fv):>8.3f}{tag}")

    # The MEAN alone is misleading, so it is never printed alone. Sessions where the encoder FAILS
    # have small ceilings and wildly negative FEVE (one session: -0.586, i.e. worse than predicting
    # the grand mean); sub-binning moves those from one failure to a slightly smaller failure and
    # drags the mean positive, while the sessions where the model actually works go the other way.
    works = base > 0.5
    print("\n  Split by whether the forward model works at all (FEVE at bins=1 > 0.5):")
    for b in bins_list:
        if b == 1:
            continue
        d = np.array([r.get(f"feve{b}", np.nan) for r in rows], float) - base
        w, f = d[works], d[~works]
        print(f"    bins={b}: WORKS (n={int(works.sum())}) mean {np.nanmean(w):+.3f} "
              f"better {int((w > 0).sum())}/{int(works.sum())}   |   "
              f"FAILS (n={int((~works).sum())}) mean {np.nanmean(f):+.3f}")
    print("\n  EV falls with bin count almost by construction (finer targets are noisier);")
    print("  FEVE is the comparable number because the ceiling absorbs exactly that.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--align", default="lick", choices=("lick", "cue", "precue"))
    ap.add_argument("--post-s", type=float, default=2.0)
    ap.add_argument("--from", dest="dates", default="0607,0806,0810,0811")
    a = ap.parse_args(argv)

    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    want = {d.strip() for d in a.dates.split(",")}
    labels = [s["label"] for s in SESSIONS if s["label"].split("_")[1] in want]
    print(f"encoder bins={a.bins} align={a.align} post_s={a.post_s}  {len(labels)} sessions",
          flush=True)
    rows = run(labels, a.bins, a.align, a.post_s)
    if rows:
        summarise(rows, a.bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

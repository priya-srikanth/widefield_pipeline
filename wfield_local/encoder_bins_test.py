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


def ceiling_eta2(X, y):
    """The pipeline's existing ceiling: between-position SS / total SS (eta-squared).

    BIASED UPWARD. With finite trials the position means differ somewhat by chance even with no real
    effect, so SS_between is inflated -- and the inflation is largest exactly where the true effect is
    smallest. FEVE built on it is therefore conservative, and unstable on weak sessions.
    """
    from wfield_local.locanmf_frozen_decoder import _ceiling
    return _ceiling(X, y)[0]


def ceiling_omega2(X, y):
    """Bias-corrected explainable fraction (omega-squared) -- the exact analytic correction here.

    omega^2 = (SS_between - (k-1) * MS_within) / (SS_total + MS_within)

    Subtracts the between-group variance a null effect would produce by chance. Can go <= 0, and that
    is INFORMATIVE rather than a failure: it means this session has no position signal distinguishable
    from noise, so a fraction-of-explainable-variance score is undefined and should not be computed.
    """
    import numpy as _np

    y = _np.asarray(y)
    groups = _np.unique(y)
    k, n = groups.size, X.shape[0]
    if n <= k:
        return _np.nan
    gm = X.mean(0)
    betw = _np.zeros(X.shape[1])
    wit = _np.zeros(X.shape[1])
    for g in groups:
        m = y == g
        mu = X[m].mean(0)
        betw += m.sum() * (mu - gm) ** 2
        wit += ((X[m] - mu) ** 2).sum(0)
    ss_b, ss_w = float(betw.sum()), float(wit.sum())
    ms_w = ss_w / (n - k)
    denom = ss_b + ss_w + ms_w
    return (ss_b - (k - 1) * ms_w) / denom if denom > 1e-12 else _np.nan


def ceiling_splithalf(X, y, n_splits=50, seed=0):
    """Repeat-based ceiling: how reproducible is the position-mean pattern across independent halves?

    Splits each position's trials in two, builds the 6-condition mean pattern from each half, and
    correlates them across all features. Spearman-Brown steps that half-data reliability up to the
    full dataset, and squaring gives the fraction of variance a perfect position-only model could
    explain. Averaged over random splits because any single split is noisy.

    More general than omega^2 (it needs no ANOVA assumptions) but Monte-Carlo noisy, so it is a
    cross-check here rather than the primary number.
    """
    import numpy as _np

    rng = _np.random.default_rng(seed)
    y = _np.asarray(y)
    groups = _np.unique(y)
    rs = []
    for _ in range(n_splits):
        a, b = [], []
        ok = True
        for g in groups:
            idx = _np.flatnonzero(y == g)
            if idx.size < 4:
                ok = False
                break
            rng.shuffle(idx)
            h = idx.size // 2
            a.append(X[idx[:h]].mean(0))
            b.append(X[idx[h:2 * h]].mean(0))
        if not ok:
            return _np.nan
        A, B = _np.concatenate(a), _np.concatenate(b)
        if A.std() < 1e-12 or B.std() < 1e-12:
            continue
        rs.append(float(_np.corrcoef(A, B)[0, 1]))
    if not rs:
        return _np.nan
    r = float(_np.mean(rs))
    sb = 2 * r / (1 + r) if (1 + r) > 1e-12 else _np.nan     # Spearman-Brown to full data
    return sb ** 2 if _np.isfinite(sb) and sb > 0 else _np.nan


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
    eta = ceiling_eta2(X, y)
    om = ceiling_omega2(X, y)
    sh = ceiling_splithalf(X, y)
    def _f(c):
        # A ceiling at or below zero means there is no position signal to explain, so the RATIO is
        # undefined -- reporting a number there is what produced FEVE = -0.586 on a session with no
        # measurable effect. NaN is the honest value.
        return ev / c if (c is not None and np.isfinite(c) and c > 0.01) else np.nan
    return dict(ev=ev, ceil_eta=eta, ceil_omega=om, ceil_split=sh,
                feve_eta=_f(eta), feve_omega=_f(om), feve_split=_f(sh))


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
                sc = encoder_scores(X, y, g)
                for k, v in sc.items():
                    rec[f"{k}{b}"] = v
                rec[f"ev{b}"], rec[f"ceil{b}"], rec[f"feve{b}"] = sc["ev"], sc["ceil_eta"], sc["feve_eta"]
                rec[f"nfeat{b}"] = X.shape[1]
            except Exception as ex:                      # noqa: BLE001
                print(f"  !! {lab} bins={b}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
                for k in ("ev", "ceil_eta", "ceil_omega", "ceil_split",
                          "feve_eta", "feve_omega", "feve_split", "ceil", "feve"):
                    rec[f"{k}{b}"] = np.nan
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

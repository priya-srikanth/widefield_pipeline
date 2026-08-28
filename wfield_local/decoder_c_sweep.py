"""Regularization sweep for the position decoder's ``LogisticRegression(C=...)``.

``C`` is the INVERSE regularization strength (objective ``C*sum(log-loss) + 0.5*||w||^2``), so
smaller C = stronger shrinkage. The pipeline hardcodes ``C=0.5`` (2x the sklearn default of 1.0),
which was reasonable for ~650 trials x ~150 collinear components but had never been measured. This
module measures it.

Two numbers, because they answer different questions:

* **CURVE** -- plain block-CV accuracy at each fixed C. Shows the SHAPE of the objective and how
  flat the optimum is. Taking the argmax of this curve is mildly optimistic (C was chosen on the
  same folds it is scored on), so it is a description, not an estimate.
* **NESTED** -- outer ``GroupKFold``; inside each outer TRAIN split an inner ``GroupKFold`` picks C;
  that C is applied to the untouched outer TEST split. This is the honest estimate of the whole
  procedure "tune C per session, then use it", and it is the number that must beat a fixed C to
  justify tuning at all.

    python -m wfield_local.decoder_c_sweep --from curated --align cue
    python -m wfield_local.decoder_c_sweep --labels PS95_0810 PS94_0810 --output <dir>

Result (2026-08-11, 10 sessions, cue-aligned 2 s) is recorded in ``DECISIONS.md`` Part III: C=0.5 is
the argmax, the optimum is flat over an order of magnitude, and nested tuning is worth +0.004 --
inside noise. Conclusion: keep the fixed default; do not tune per session.

THE ENGAGED CUT COMES FROM CONFIG (`decode.max_rt_s`), not from a literal here. It was hardcoded to
2.0 s while the config moved to 3.5 s on 2026-08-21 -- the task's REAL response window, read per
session from `gui_config.json` -- so this module was filing a lick at 2.5 s as "no lick" when it is a
REWARDED HIT the task scored, and "engaged" meant one thing here and another in the analysis this is a
diagnostic FOR. A sweep is internally consistent at either cut, which is exactly why it could drift
unnoticed; what it cannot survive is being quoted against a headline computed at the other one.

EVERY NUMBER RECORDED FOR THIS MODULE IN DECISIONS WAS MEASURED AT 2.0 s and is pre-change until
re-measured. The cut actually used is printed at run time, so a result can never be read without it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.locanmf_position_decoder import _trial_features

GRID = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
FS = 31.23



_ANNOUNCED = False


def _max_rt() -> float:
    """The engaged cut, from `decode.max_rt_s`, announced once so a result carries its own boundary.

    Announced rather than merely read: this module's recorded numbers were measured at 2.0 s, and a
    silent change of boundary is how a diagnostic and the headline it is a diagnostic FOR come to
    disagree without either looking wrong.
    """
    global _ANNOUNCED
    v = float(config.defaults()["decode"]["max_rt_s"])
    if not _ANNOUNCED:
        print(f"[{__name__.rsplit('.', 1)[-1]}] engaged cut = {v:g}s (decode.max_rt_s). Numbers "
              f"recorded in DECISIONS for this module were measured at 2.0s.", flush=True)
        _ANNOUNCED = True
    return v


def _pipe(C):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))


def _args(source, align, post_s):
    return SimpleNamespace(source=source, align=align, baseline="none", pre_s=1.0,
                           post_s=post_s, fs=FS, max_rt=_max_rt())


def cv_accuracy(X, y, g, C, nsplit=5):
    """Block-CV accuracy at a fixed C (GroupKFold by position block, as the decoder uses)."""
    ng = min(nsplit, int(np.unique(g).size))
    return accuracy_score(y, cross_val_predict(_pipe(C), X, y, cv=GroupKFold(ng), groups=g))


def nested_accuracy(X, y, g, grid=GRID, nsplit=5):
    """Honest 'tune C then apply' accuracy: C chosen inside each outer TRAIN split only.

    Returns (accuracy, [C picked per outer fold]).
    """
    ng = min(nsplit, int(np.unique(g).size))
    yhat = np.empty_like(y)
    picks = []
    for tr, te in GroupKFold(ng).split(X, y, g):
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]
        ngi = min(nsplit, int(np.unique(gtr).size))
        best, bestC = -1.0, grid[0]
        for C in grid:
            a = accuracy_score(ytr, cross_val_predict(_pipe(C), Xtr, ytr,
                                                      cv=GroupKFold(ngi), groups=gtr))
            if a > best:
                best, bestC = a, C
        picks.append(bestC)
        yhat[te] = _pipe(bestC).fit(Xtr, ytr).predict(X[te])
    return accuracy_score(y, yhat), picks


def sweep(labels, source="locanmf", align="cue", post_s=2.0, grid=GRID, current=0.5, verbose=True):
    """Run the curve + nested estimate for each label. Returns a summary dict."""
    rows = {}
    for lab in labels:
        s = next((x for x in SESSIONS if x["label"] == lab), None)
        if s is None:
            print(f"[c-sweep] unknown session {lab} -- skipped", flush=True)
            continue
        X, y, g, _, _, _ = _trial_features(s, _args(source, align, post_s))
        curve = [cv_accuracy(X, y, g, C) for C in grid]
        nacc, picks = nested_accuracy(X, y, g, grid)
        bi = int(np.argmax(curve))
        rows[lab] = {"n_trials": int(X.shape[0]), "n_features": int(X.shape[1]),
                     "curve": {str(C): float(a) for C, a in zip(grid, curve)},
                     "best_C": grid[bi], "best_acc": float(curve[bi]),
                     "acc_at_current": float(curve[grid.index(current)]),
                     "nested_acc": float(nacc), "nested_picks": list(picks)}
        if verbose:
            print(f"{lab}: n={X.shape[0]:4d} feat={X.shape[1]:3d} | "
                  + "  ".join(f"C={C:<5g}{a:.3f}" for C, a in zip(grid, curve)), flush=True)
    if not rows:
        return {}
    A = np.array([[rows[l]["curve"][str(C)] for C in grid] for l in rows])
    mean = A.mean(0)
    bi = int(np.argmax(mean))
    nested = np.array([rows[l]["nested_acc"] for l in rows])
    fixed = A[:, grid.index(current)]
    allpicks = [p for l in rows for p in rows[l]["nested_picks"]]
    summary = {"sessions": list(rows), "per_session": rows, "grid": list(grid), "align": align,
               "source": source,
               "mean_curve": {str(C): float(m) for C, m in zip(grid, mean)},
               "sem_curve": {str(C): float(s) for C, s in zip(grid, A.std(0) / np.sqrt(len(rows)))},
               "best_mean_C": grid[bi], "best_mean_acc": float(mean[bi]), "current_C": current,
               "current_mean_acc": float(fixed.mean()), "nested_mean_acc": float(nested.mean()),
               "nested_minus_fixed": float(nested.mean() - fixed.mean()),
               "pick_histogram": {str(c): allpicks.count(c) for c in sorted(set(allpicks))}}
    if verbose:
        print(f"\nMEAN over {len(rows)} sessions ({align}-aligned, source={source})")
        for C, m in zip(grid, mean):
            tag = "  <-- current" if C == current else ("  <-- best" if C == grid[bi] else "")
            print(f"  C={C:<6g} acc={m:.4f}{tag}")
        print(f"\n  best mean C = {grid[bi]} ({mean[bi]:.4f}); current C={current} ({fixed.mean():.4f})")
        print(f"  NESTED (tune-then-apply) = {nested.mean():.4f}   FIXED C={current} = {fixed.mean():.4f}")
        print(f"  nested - fixed = {summary['nested_minus_fixed']:+.4f}  "
              f"(<= 0, or within SEM, means per-session tuning buys nothing)")
        print(f"  inner-CV picked C: {summary['pick_histogram']}  "
              f"(scatter across the grid = flat objective, argmax is noise)")
    return summary


def figure(summary, out_path):
    grid = summary["grid"]
    mean = np.array([summary["mean_curve"][str(C)] for C in grid])
    sem = np.array([summary["sem_curve"][str(C)] for C in grid])
    rows = summary["per_session"]
    cur = summary["current_C"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for lab, r in rows.items():
        ax[0].plot(grid, [r["curve"][str(C)] for C in grid], "-o", ms=3, lw=1, alpha=0.5, label=lab)
    ax[0].errorbar(grid, mean, yerr=sem, color="k", lw=2.5, marker="s", ms=5, zorder=5,
                   label="mean +/- SEM")
    ax[0].axvline(cur, color="tab:red", ls="--", lw=1.2, label=f"pipeline default C={cur}")
    ax[0].axhline(1 / 6, color="grey", ls=":", lw=1, label="chance")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("C  (inverse regularization strength; smaller = stronger shrinkage)")
    ax[0].set_ylabel("block-CV accuracy")
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title(f"Decoder accuracy vs C ({summary['align']}-aligned, block CV)")
    labs = list(rows)
    x = np.arange(len(labs)); w = 0.38
    ax[1].bar(x - w / 2, [rows[l]["acc_at_current"] for l in labs], w,
              label=f"fixed C={cur}", color="tab:blue")
    ax[1].bar(x + w / 2, [rows[l]["nested_acc"] for l in labs], w,
              label="nested (C tuned per outer fold)", color="tab:orange")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
    ax[1].set_ylabel("accuracy"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"Honest comparison: fixed default vs per-session tuning\n"
                    f"nested - fixed = {summary['nested_minus_fixed']:+.4f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=None, help="explicit session labels (default: --from set)")
    ap.add_argument("--from", dest="from_", default="curated", help="date span for the session set")
    ap.add_argument("--source", default="locanmf", choices=("locanmf", "roi"))
    ap.add_argument("--align", default="cue", choices=("cue", "lick", "precue"))
    ap.add_argument("--post-s", type=float, default=2.0)
    ap.add_argument("--output", type=Path, default=None, help="figure/JSON dir (default: cue_analysis on MICROSCOPE)")
    args = ap.parse_args(argv)

    labels = args.labels
    if not labels:
        dates = config.curated_dates() if args.from_ == "curated" else config.expand_dates([args.from_], width=4)
        labels = [s["label"] for s in SESSIONS if s["label"][-4:] in set(dates)]
    out = args.output or Path(config.resolver().resolve("labcams", "locanmf_lick_pooled/cue_analysis"))
    out.mkdir(parents=True, exist_ok=True)

    summary = sweep(labels, source=args.source, align=args.align, post_s=args.post_s)
    if not summary:
        print("[c-sweep] no sessions resolved")
        return 1
    p = figure(summary, out / f"decoder_C_sweep_{args.source}_{args.align}.png")
    (out / f"decoder_C_sweep_{args.source}_{args.align}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

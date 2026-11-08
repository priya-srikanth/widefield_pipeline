"""Mix-matched pre-stroke control for the within-session REFIT decoder (D6, Priya 11-08).

    python scripts/refit_mix_matched_control.py --out <epoch data dir> [--align cue] [--variant working] [--draws 5]
                                                [--jobs 6] [--animals PS94] [--limit N]

THE QUESTION. A refit decoder trained on a session where a position is rare learns a prior against predicting it, so its
recall there is low from the trial MIX alone -- post-stroke that reads as "position code lost". The 5r figures guarded
against it with a share floor (grant_confusion.MIN_REFIT_SHARE); Priya: every floor is arbitrary. This control measures
the mix effect instead of hiding it.

THE CONTROL, per post-stroke session S and position q (SYMMETRIC since 11-08 02:15; the first version subsampled only P
to S's full counts and skipped any P short of them -- 453 of ~800 pairs, because post sessions often hold more trials per
position than any August pre session -- and its pre null check came out biased, up to +0.23):
  * for EVERY August pre-stroke session P of the SAME animal: the common counts c_q = floor(f * S_q), f = min(1,
    min_q P_q / S_q) -- S's position MIX at the largest size both sessions can supply. BOTH S and P are subsampled to c
    (whole blocks, the last one trimmed to the exact count) and refit WITHIN themselves with the same estimator and block
    CV as the 5r refit (`grant_confusion._refit_pred`); `--draws` draws per (S, P). Never pooled across sessions.
  * excess(S, q) = mean over P and draws of [refit(S_sub, q) - refit(P_sub, q)]: same size, same mix, same procedure;
    only the session differs. No pair is skipped. refit_full = S's own refit on all its trials, for reference.
  * NO FLOOR (Priya 11-08: 1/18, 1/20 and 10 trials are all arbitrary): every position with >= 1 trial is scored, in S and
    in its controls alike (`mark_thin=False`); the trial count of every cell is written so small cells are visible.
NULL: each PRE session is run as if it were S, against the animal's OTHER pre sessions. Symmetric by construction (a pair
appears both ways with the same c), so its MEAN is ~0 by design; its SPREAD is the noise floor post excess must beat.
CACHE: `--inputs` saves / reuses the captured arrays (the capture is ~35-40 min; scoring ~1-2 min).

INPUTS: exactly the arrays the 5r figure refits (`grant_confusion.REFIT_CAPTURE` during `_collect_5c(align, variant,
"frozen")` -- the frozen mode loads the stored decoders and refits nothing).
OUT: <out>/refit_mixmatch_<align>_<variant>_sessions.csv (one row per session x position) and the per-animal x epoch x
position table printed (session means; trial counts). No bootstrap here: per-animal tables first (rule), stats after.
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

EPOCHS = ("pre", "acute", "subacute", "chronic")


def capture(align: str, variant: str) -> list[dict]:
    """Every session's refit inputs (animal, label, pre, X, y, blk), as the 5r figure sees them."""
    from wfield_local import grant_confusion as GC
    GC.REFIT_CAPTURE = []
    try:
        GC._collect_5c(align, variant, "frozen")
        return GC.REFIT_CAPTURE
    finally:
        GC.REFIT_CAPTURE = None


def recall(y, pred) -> dict:
    """{position code: (n trials, recall)} over every position present."""
    y, pred = np.asarray(y), np.asarray(pred)
    return {int(q): (int((y == q).sum()), float(np.mean(pred[y == q] == q))) for q in np.unique(y)}


def refit_recall(X, y, blk) -> dict | None:
    from wfield_local import grant_confusion as GC
    from wfield_local.locanmf_frozen_decoder import _pipe
    p = GC._refit_pred(_pipe, X, y, blk, mark_thin=False)
    return None if p is None else recall(y, p)


def matched_rows(P: dict, counts: dict, rng) -> np.ndarray | None:
    """Row indices of pre session P matching ``counts`` {position: n} exactly; whole blocks, last one trimmed.
    None when P holds fewer trials than needed at any position."""
    y, b = P["y"], P["blk"]
    take = []
    for q, n in counts.items():
        rows_q = np.flatnonzero(y == q)
        if len(rows_q) < n:
            return None
        got = []
        for blk in rng.permutation(np.unique(b[rows_q])):
            r = rows_q[b[rows_q] == blk]
            need = n - len(got)
            if len(r) >= need:
                got.extend(rng.choice(r, need, replace=False) if len(r) > need else r)
                break
            got.extend(r)
        take.extend(got)
    return np.sort(np.asarray(take, int))


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def common_counts(S_counts: dict, P_counts: dict) -> dict:
    """S's position mix at the largest size P can also supply: c_q = floor(f * S_q), f = min(1, min_q P_q / S_q)."""
    f = min([1.0] + [P_counts.get(q, 0) / n for q, n in S_counts.items() if n > 0])
    return {q: int(np.floor(f * n)) for q, n in S_counts.items()}


def _counts(sess: dict) -> dict:
    q, n = np.unique(sess["y"], return_counts=True)
    return {int(a): int(b) for a, b in zip(q, n)}


def _mean(v) -> float:
    return float(np.mean(v)) if v else np.nan


def unit_job(unit) -> list[dict]:
    """One scored session S against its animal's pre sessions, both subsampled to common counts -> rows per position."""
    S, pres, draws = unit
    full = refit_recall(S["X"], S["y"], S["blk"])
    if full is None:
        return []
    sc = _counts(S)
    diff, s_acc, p_acc, n_used = ({q: [] for q in sc} for _ in range(4))
    pairs, failed = 0, 0
    for P in pres:
        if P["label"] == S["label"]:
            continue
        c = {q: n for q, n in common_counts(sc, _counts(P)).items() if n > 0}
        if len(c) < 2:
            failed += 1
            continue
        got = False
        for d in range(draws):
            rs = matched_rows(S, c, np.random.default_rng(_seed(S["label"], P["label"], d, "S")))
            rp = matched_rows(P, c, np.random.default_rng(_seed(S["label"], P["label"], d, "P")))
            a = refit_recall(S["X"][rs], S["y"][rs], S["blk"][rs])
            b = refit_recall(P["X"][rp], P["y"][rp], P["blk"][rp])
            if a is None or b is None:
                continue
            got = True
            for q in c:
                if q in a and q in b:
                    diff[q].append(a[q][1] - b[q][1])
                    s_acc[q].append(a[q][1])
                    p_acc[q].append(b[q][1])
                    n_used[q].append(c[q])
        pairs += got
        failed += not got
    tot = sum(sc.values())
    return [dict(animal=S["animal"], label=S["label"], position=q, n=n, share=n / tot, refit_full=r,
                 refit_sub=_mean(s_acc[q]), control=_mean(p_acc[q]), excess=_mean(diff[q]),
                 n_matched=_mean(n_used[q]), pairs=pairs, pairs_failed=failed)
            for q, (n, r) in full.items()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--align", default="cue")
    ap.add_argument("--variant", default="working")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N sessions per animal (timing)")
    ap.add_argument("--inputs", type=Path, default=None, help="pickle of the captured arrays: reused if present, else written")
    a = ap.parse_args(argv)
    import time

    from wfield_local import analysis_kit as ak
    from wfield_local import epochs, writeguard
    from wfield_local.grant_figures import CONF_LABELS

    writeguard.assert_writable(a.out)
    t0 = time.time()
    import pickle
    if a.inputs is not None and a.inputs.exists():
        with open(a.inputs, "rb") as fh:
            cap = pickle.load(fh)
        print(f"inputs reused: {a.inputs}", flush=True)
    else:
        cap = capture(a.align, a.variant)
        if a.inputs is not None:
            writeguard.assert_writable(a.inputs.parent)
            a.inputs.parent.mkdir(parents=True, exist_ok=True)
            with open(a.inputs, "wb") as fh:
                pickle.dump(cap, fh, protocol=pickle.HIGHEST_PROTOCOL)
    cap = [c for c in cap if a.animals is None or c["animal"] in a.animals]
    print(f"captured {len(cap)} sessions in {time.time() - t0:.0f} s", flush=True)
    units = []
    for an in sorted({c["animal"] for c in cap}):
        mine = [c for c in cap if c["animal"] == an]
        pres = [c for c in mine if c["pre"]]
        scored = mine[: a.limit] if a.limit else mine
        units += [(S, pres, a.draws) for S in scored]
    t1 = time.time()
    got, fail = ak.fan_sessions(units, unit_job, jobs=min(a.jobs, len(units)), key=lambda u: u[0]["label"])
    for u, err in fail:
        print(f"  FAILED {u}: {err}")
    D = pd.DataFrame([r for _u, rr in got for r in rr])
    print(f"scored {len(units)} sessions in {time.time() - t1:.0f} s ({a.jobs} jobs, {a.draws} draws)", flush=True)
    if D.empty:
        return 1
    D["epoch"] = D.label.map(epochs.epoch_of)
    names = dict(enumerate(CONF_LABELS))
    D["position"] = D.position.map(lambda q: names.get(int(q), q))
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"refit_mixmatch_{a.align}_{a.variant}_sessions.csv"
    D.to_csv(f, index=False, float_format="%.4g")
    pd.set_option("display.width", 240)
    for col in ("refit_full", "refit_sub", "control", "excess"):
        t = D.groupby(["animal", "position", "epoch"])[col].mean().unstack("epoch").reindex(columns=list(EPOCHS))
        print(f"\n== {col} (session means) by animal x position x epoch; {a.align} / {a.variant}; pre = null check")
        print(t.round(3).to_string())
    n = D.groupby(["animal", "position", "epoch"]).n.agg(["sum", "min"]).unstack("epoch")
    print("\n== trials (sum / min over sessions)")
    print(n.to_string())
    u = D.drop_duplicates("label")
    print(f"\npairs scored: {int(u.pairs.sum())}; pairs that could not be refit: {int(u.pairs_failed.sum())}")
    m = D.groupby(["animal", "position", "epoch"]).n_matched.mean().unstack("epoch").reindex(columns=list(EPOCHS))
    print("\n== matched trials per position (mean)")
    print(m.round(1).to_string())
    print("->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

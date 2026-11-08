"""Mix-matched pre-stroke control for the within-session REFIT decoder (D6, Priya 11-08).

    python scripts/refit_mix_matched_control.py --out <epoch data dir> [--align cue] [--variant working] [--draws 5]
                                                [--jobs 6] [--animals PS94] [--limit N]

THE QUESTION. A refit decoder trained on a session where a position is rare learns a prior against predicting it, so its
recall there is low from the trial MIX alone -- post-stroke that reads as "position code lost". The 5r figures guarded
against it with a share floor (grant_confusion.MIN_REFIT_SHARE); Priya: every floor is arbitrary. This control measures
the mix effect instead of hiding it.

THE CONTROL, per post-stroke session S and position q:
  * for EVERY August pre-stroke session P of the SAME animal: subsample P to S's trial count at EACH position (size and
    mix matched together), drawing whole blocks (the ~6-trial runs at one position) and trimming the last block to the
    exact count; refit WITHIN P with the same estimator and block CV as the 5r refit (`grant_confusion._refit_pred`);
    `--draws` random draws per (S, P), averaged. Never pooled across sessions (cross-day drift would lower the control).
  * a (S, P) pair where P holds fewer trials at some position than S needs is SKIPPED and counted, never resampled with
    replacement.
  * control(S, q) = mean over the animal's usable pre sessions; excess(S, q) = refit(S, q) - control(S, q).
  * NO FLOOR (Priya 11-08: 1/18, 1/20 and 10 trials are all arbitrary): every position with >= 1 trial is scored, in S and
    in its controls alike (`mark_thin=False`); the trial count of every cell is written so small cells are visible.
NULL CHECK: each PRE session is run as if it were S, against the animal's OTHER pre sessions; its excess should be ~0.

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


def unit_job(unit) -> list[dict]:
    """One scored session S against its animal's pre sessions -> one row per position."""
    S, pres, draws = unit
    own = refit_recall(S["X"], S["y"], S["blk"])
    if own is None:
        return []
    counts = {q: n for q, (n, _r) in own.items()}
    ctrl = {q: [] for q in counts}
    used, skipped, failed = 0, 0, 0
    for P in pres:
        if P["label"] == S["label"]:
            continue
        per_draw = {q: [] for q in counts}
        for d in range(draws):
            rows = matched_rows(P, counts, np.random.default_rng(_seed(S["label"], P["label"], d)))
            if rows is None:
                break
            r = refit_recall(P["X"][rows], P["y"][rows], P["blk"][rows])
            if r is None:
                continue
            for q in counts:
                if q in r:
                    per_draw[q].append(r[q][1])
        if rows is None:
            skipped += 1
            continue
        if not any(per_draw.values()):
            failed += 1
            continue
        used += 1
        for q in counts:
            if per_draw[q]:
                ctrl[q].append(float(np.mean(per_draw[q])))
    out = []
    for q, (n, r) in own.items():
        c = float(np.mean(ctrl[q])) if ctrl[q] else np.nan
        out.append(dict(animal=S["animal"], label=S["label"], position=q, n=n, share=n / sum(counts.values()),
                        refit=r, control=c, excess=r - c if ctrl[q] else np.nan, pre_used=used, pre_skipped=skipped,
                        pre_failed=failed))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--align", default="cue")
    ap.add_argument("--variant", default="working")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N sessions per animal (timing)")
    a = ap.parse_args(argv)
    import time

    from wfield_local import analysis_kit as ak
    from wfield_local import epochs, writeguard
    from wfield_local.grant_figures import CONF_LABELS

    writeguard.assert_writable(a.out)
    t0 = time.time()
    cap = [c for c in capture(a.align, a.variant) if a.animals is None or c["animal"] in a.animals]
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
    for col in ("refit", "control", "excess"):
        t = D.groupby(["animal", "position", "epoch"])[col].mean().unstack("epoch").reindex(columns=list(EPOCHS))
        print(f"\n== {col} (session means) by animal x position x epoch; {a.align} / {a.variant}; pre = null check")
        print(t.round(3).to_string())
    n = D.groupby(["animal", "position", "epoch"]).n.agg(["sum", "min"]).unstack("epoch")
    print("\n== trials (sum / min over sessions)")
    print(n.to_string())
    print(f"\npairs skipped (pre session short of a position count): {int(D.drop_duplicates('label').pre_skipped.sum())}; "
          f"refit failed: {int(D.drop_duplicates('label').pre_failed.sum())}")
    print("->", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

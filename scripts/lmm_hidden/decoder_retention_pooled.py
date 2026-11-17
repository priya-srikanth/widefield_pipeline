"""LMM hidden slide for goal page Fig 8.5 `decoder_retention_pooled` (scripts/decoder_retention_sessions.py, FIG 2).

Value: per SESSION and trial class with >= 5 far-contra trials, the mean over trials of the frozen pre-stroke spout
decoder's retention (P(true spout) - 1/6) / (the animal's pre-stroke hit mean - 1/6): 1 = as well coded as the animal's
pre-stroke hits, 0 = chance. August pre only. Panels = window (pre-cue ENL, cue, lick); cells = trial class (hits =
pre_hit + hit; failed attempts = incomplete licks, no contact).
Source: <root>/attempt_coding/decoder_retention_sessions.csv (columns window, animal, epoch, cls2, label, day, ret,
p_true, n; already filtered to n >= 5), read as is.

PRE for failed attempts: pre-stroke trials are hits only, so the published figure tests failed attempts against PRE
HITS. The failed-attempt cell is given the pre-hit session rows as its pre rows (post rows = failed-attempt sessions); no
rows are invented. The hollow diamonds of the published figure (one animal's all-trial retention where too few sessions
exist) were never part of its test and are not used here.
Published test: `analysis_kit.boot_delta` (nested bootstrap animals > sessions) on these session values, change vs the
animal's pre hits, only when >= 2 animals have sessions in both arms; bold number = 95 % CI excludes 0, no correction.

    python -m scripts.lmm_hidden.decoder_retention_pooled [--root <imaging_encoding>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.lmm_hidden import _paths, render

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
NAME = "decoder_retention_pooled"
WIN = (("precue", "pre-cue (ENL)"), ("cue", "cue window"), ("lick", "lick window"))
CLS = ("hit", "failed")
LABELS = {"hit": "hits", "failed": "failed attempts\n(vs pre hits)"}


SHOWN: list = []      # rows below MIN_HITS: drawn as "<20 hits", not tested (Priya 11-17)


def build(root: Path) -> pd.DataFrame:
    S = pd.read_csv(_paths.sub(root, "attempt_coding") / "decoder_retention_sessions.csv")
    if "n" in S:                          # decoder_retention_sessions._cell: hit cells need >= MIN_HITS (11-07)
        H, dropped, X = _paths.hit_gate(S[S.cls2 == "hit"], "ret", "n", by=("window", "animal", "epoch"),
                                        return_rows=True)
        SHOWN.append(pd.DataFrame({"panel": X.window.map(dict(WIN)), "position": X.window + ":hit", "epoch": X.epoch,
                                   "animal": X.animal, "label": X.label, "value": X.ret, "n_hits": X.n}))
        print(f"[hit] hit gate (>= MIN_HITS hits per window x animal x epoch) dropped: {', '.join(dropped) or 'none'}")
        S = pd.concat([H, S[S.cls2 != "hit"]], ignore_index=True)
    rows = []
    for w, wl in WIN:
        K = S[S.window == w]
        pre_hit = K[(K.epoch == "pre") & (K.cls2 == "hit")]
        for c in CLS:
            post = K[(K.epoch != "pre") & (K.cls2 == c)]
            for D in (pre_hit, post):
                rows.append(pd.DataFrame({"panel": wl, "position": f"{w}:{c}", "epoch": D.epoch, "animal": D.animal,
                                          "label": D.label, "value": D.ret}))
    return pd.concat(rows, ignore_index=True)


def verify(T: pd.DataFrame, root: Path) -> None:
    """mean over animals of (animal post mean - animal pre mean), >= 2 animals in both arms, vs the published
    decoder_retention_pooled.csv delta."""
    P = pd.read_csv(_paths.sub(root, "attempt_coding") / "decoder_retention_pooled.csv")
    for pos, D in T.groupby("position", sort=False):
        w, c = pos.split(":")
        pre = D[D.epoch == "pre"].groupby("animal").value.mean()
        for ep in ("acute", "subacute", "chronic"):
            po = D[D.epoch == ep].groupby("animal").value.mean()
            com = [a for a in po.index if a in pre.index]
            mine = (po[com] - pre[com]).mean() if len(com) >= 2 else float("nan")
            p = P[(P.window == w) & (P.epoch == ep) & (P["class"] == c)]
            pub = p.delta.iloc[0] if len(p) else float("nan")
            print(f"  {pos:16s} {ep:9s} n_an={len(com)} extracted {mine:+.4f}  published {pub:+.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(_paths.ie_root(ROOT)))
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    T = build(Path(a.root))
    print(T.groupby(["position", "epoch"]).agg(sessions=("value", "size"), animals=("animal", "nunique")).unstack())
    verify(T, Path(a.root))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = out / f"{NAME}_sessions.csv"
    T.to_csv(src, index=False)
    cells = [f"{w}:{c}" for w, _ in WIN for c in CLS]
    labels = {f"{w}:{c}": LABELS[c] for w, _ in WIN for c in CLS}
    R = render.render(src, NAME, out, title="Frozen pre-stroke decoder: far-contra code retained, per session, by trial "
                      "class (failed attempts vs pre hits)", ylabel="retention (1 = pre hits), change vs own pre hits",
                      cells=cells, labels=labels, panel_by="panel",
                      shown_only=pd.concat(SHOWN, ignore_index=True) if SHOWN else None)
    print(R.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

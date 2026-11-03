"""LMM hidden slide for goal page Fig 6.3 `fig9i3_cls_pooled` (= coupling_by_class.py `coupling_cls_pooled.png`, 4i3).

Value: per SESSION (>= 10 far-contra trials of the class), the within-session trial-by-trial r between the frozen
pre-stroke decoder's P(far mid) - P(far contra) and the executed lick weight far mid - far contra (both centred within
session x class), August pre only. Panels = executed definition (landing: 3-D endpoints; trajectory: lick paths); cells =
trial class (hits = contact; failed attempts = incomplete licks, no contact).
Source: <root>/attempt_coding_pp/coupling_cls_sessions.csv (written by scripts/coupling_by_class.py, columns exec,
animal, epoch, class, label, day, r, n), read as is.

PRE for failed attempts: pre-stroke trials are hits only, so the published figure tests failed attempts against PRE HITS
(class-matched for hits, "vs pre hits" for failed). The same contrast is built here by giving the failed-attempt cell the
pre-hit session rows as its pre rows (the post rows are the failed-attempt sessions) -- no rows are invented.
Published test: `analysis_kit.boot_delta` (nested bootstrap animals > sessions) on these session values, change vs the
animal's pre hits, only when >= 2 animals have sessions in both arms; bold number = 95 % CI excludes 0, no correction.

    python -m scripts.lmm_hidden.fig9i3_cls_pooled [--root <imaging_encoding>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.lmm_hidden import render

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
NAME = "fig9i3_cls_pooled"
KINDS = ("landing", "trajectory")
CLS = ("hit", "failed")
LABELS = {"hit": "hits\n(contact)", "failed": "failed attempts\n(vs pre hits)"}


def build(root: Path) -> pd.DataFrame:
    S = pd.read_csv(Path(root) / "attempt_coding_pp" / "coupling_cls_sessions.csv")
    rows = []
    for kind in KINDS:
        K = S[S.exec == kind]
        pre_hit = K[(K.epoch == "pre") & (K["class"] == "hit")]
        for c in CLS:
            post = K[(K.epoch != "pre") & (K["class"] == c)]
            for D in (pre_hit, post):
                rows.append(pd.DataFrame({"panel": f"executed = {kind}", "position": f"{kind}:{c}", "epoch": D.epoch,
                                          "animal": D.animal, "label": D.label, "value": D.r}))
    return pd.concat(rows, ignore_index=True)


def verify(T: pd.DataFrame, root: Path) -> None:
    """mean over animals of (animal post mean - animal pre mean), animals with both arms and >= 2 of them, vs the
    published coupling_cls_pooled.csv delta."""
    P = pd.read_csv(Path(root) / "attempt_coding_pp" / "coupling_cls_pooled.csv")
    for pos, D in T.groupby("position", sort=False):
        kind, c = pos.split(":")
        pre = D[D.epoch == "pre"].groupby("animal").value.mean()
        for ep in ("acute", "subacute", "chronic"):
            po = D[D.epoch == ep].groupby("animal").value.mean()
            com = [a for a in po.index if a in pre.index]
            mine = (po[com] - pre[com]).mean() if len(com) >= 2 else float("nan")
            p = P[(P.exec == kind) & (P.epoch == ep) & (P["class"] == c)]
            pub = p.delta.iloc[0] if len(p) else float("nan")
            print(f"  {pos:20s} {ep:9s} n_an={len(com)} extracted {mine:+.4f}  published {pub:+.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    T = build(Path(a.root))
    print(T.groupby(["position", "epoch"]).agg(sessions=("value", "size"), animals=("animal", "nunique")).unstack())
    verify(T, Path(a.root))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = out / f"{NAME}_sessions.csv"
    T.to_csv(src, index=False)
    cells = [f"{k}:{c}" for k in KINDS for c in CLS]
    labels = {f"{k}:{c}": LABELS[c] for k in KINDS for c in CLS}
    R = render.render(src, NAME, out, title="Frozen pre-stroke decoder - lick coupling (far-contra trials), per session, "
                      "by trial class (failed attempts vs pre hits)", ylabel="within-session r, change vs own pre hits",
                      cells=cells, labels=labels, panel_by="panel")
    print(R.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

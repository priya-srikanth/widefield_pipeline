"""LMM hidden-slide version of goal-page Fig 0.6 `recovery_timing_pooled_epoch_lick` (deck crop recovery_timing_epoch_body;
Priya 2026-11-03).

Published figure: <root>/attempt_coding/figures/recovery_timing_pooled_epoch_lick.png by `scripts/recovery_timing.py`
(`pooled_figures` part (b); wf_attempt branch analysis/attempt-coding). Three panels, one dot per session, normalised
within animal:
  beh  behaviour: far-contra hit rate / the animal's pre far-contra hit rate (qc/trial_attempts.csv, August sessions);
  neu  cortex: frozen-decoder far-contra retention on HITS (attempt_coding/decoder_retention_sessions.csv, window lick,
       1 = pre hits by construction; NaN for sessions with too few hits, e.g. most acute sessions);
  gap  neu - beh per session.
Source (read-only): <root>/attempt_coding/recovery_timing_series_lick.csv (= recovery_timing.series("lick"), written by
the same script that drew the figure; pre sessions included, so levels not deltas).
Published test: paired nested bootstrap (analysis_kit.boot_delta, animals -> sessions) per measure x post epoch, 95 % CI,
bold = CI excludes 0 (no stars, no multiplicity correction; recovery_timing_pooled_epoch_lick.csv). Here:
epoch_change_lmm, Holm across all 9 cells x epochs.

    python -m scripts.lmm_hidden.recovery_timing_pooled_epoch_lick [--root <imaging_encoding>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.lmm_hidden import _paths, render

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
NAME = "recovery_timing_pooled_epoch_lick"
EP = ["pre", "acute", "subacute", "chronic"]
MEAS = {"beh": "behavior: far-contra hit rate", "neu": "cortex: far-contra code on hits", "gap": "gap: cortex - behavior"}


SHOWN: list = []      # rows below MIN_HITS: drawn as "<20 hits", not tested (Priya 11-17)


def build(root: Path) -> pd.DataFrame:
    S = pd.read_csv(_paths.sub(root, "attempt_coding") / "recovery_timing_series_lick.csv")
    S["gap"] = S.neu - S.beh
    out = []
    for m, lab in MEAS.items():
        G = S
        if m in ("neu", "gap"):           # code on hits: the main figure's MIN_HITS gate per animal x epoch (11-07)
            G, dropped, X = _paths.hit_gate(S, m, "n_hits", return_rows=True)
            SHOWN.append(pd.DataFrame({"position": m, "epoch": X.epoch, "animal": X.animal, "label": X.label,
                                       "value": X[m], "n_hits": X.n_hits, "panel": lab}))
            print(f"[{m}] hit gate (>= MIN_HITS hits per animal x epoch) dropped: {', '.join(dropped) or 'none'}")
        out.append(pd.DataFrame({"position": m, "epoch": G.epoch, "animal": G.animal, "label": G.label, "day": G.day,
                                 "value": G[m], "panel": lab}))
    return pd.concat(out, ignore_index=True).dropna(subset=["value"]).reset_index(drop=True)


def verify(S: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in S.groupby("position", sort=False):
        pre = g[g.epoch == "pre"].groupby("animal").value.mean()
        for ep in EP[1:]:
            po = g[g.epoch == ep].groupby("animal").value.mean()
            com = po.index.intersection(pre.index)
            rows.append(dict(position=c, epoch=ep, delta=float((po[com] - pre[com]).mean()), n_animals=len(com),
                             n_sessions=int(g[(g.epoch == ep) & g.animal.isin(com)].shape[0])))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(_paths.ie_root(ROOT)))
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    S = build(Path(a.root))
    pd.set_option("display.width", 200)
    print("== n sessions / animals per cell x epoch")
    print(S.groupby(["position", "epoch"]).agg(sessions=("value", "size"), animals=("animal", "nunique")).unstack())
    print("== published quantity (change vs own pre, mean of animals)\n" + verify(S).round(3).to_string(index=False))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = out / f"{_paths.out_name(NAME)}_sessions.csv"
    S.to_csv(src, index=False)
    T = render.render(src, NAME, out, title="Fig 0.6: far-contra hit rate vs frozen code on hits by epoch (lick window)",
                      ylabel="change vs own pre (fraction of pre)", cells=list(MEAS),
                      labels={"beh": "hit rate", "neu": "code on hits", "gap": "cortex -\nbehavior"}, panel_by="panel",
                      shown_only=pd.concat(SHOWN, ignore_index=True) if SHOWN else None)
    print(T.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LMM hidden-slide version of goal-page KEY FIGURE 1, panels B and C (deck crop o_key_BC; Priya 2026-11-03).

Published figure: `_goal_page/key_figure.png` by `scripts/key_figure.py` (wf_attempt branch analysis/attempt-coding).
  B  CUED licks, fraction of the pre-stroke level:
       code    the FROZEN pre-stroke far-contra spout code on far-contra HITS (`recovery_timing.series("lick")` column
               `neu`, already 1 = the animal's pre level by construction; sessions with too few hits are NaN there);
       frozen  the frozen pre-stroke LATERAL lick decoder, held-out r (pre sessions leave-one-session-out);
       refit   the decoder REFIT per epoch (leave-one-session-out within the epoch), against "pre, matched n" = pre LOSO
               trained on the same number of licks as that epoch's refit -- so each post epoch has its OWN pre rows.
  C  ITI licks: frozen and epoch-refit lateral decoder, as B.
Sources (read-only):
  <root>/attempt_coding/recovery_timing_series_lick.csv   (written by recovery_timing.py from qc/trial_attempts.csv
                                                           + attempt_coding/decoder_retention_sessions.csv; = series())
  <root>/iti_licks/refit_decoders_sessions.csv            (lick_refit_decoders.py; per session r, kind, target, vs)
Value per session: code = `neu`; decoders = r / pre_r, with pre_r = mean over animals of the animal's mean pre r for that
cell (exactly how key_figure._frac scales: fraction = 1 + delta / pre_r), so the LMM change is in "fraction of pre".
Because the epoch-refit reference differs per epoch, the refit is THREE cells per panel (refit_acute / _subacute /
_chronic), each = that epoch's refit sessions + its matched-n pre sessions.
Published test: paired nested bootstrap (analysis_kit.boot_delta, animals -> sessions), 95 % CI; no stars drawn, no
multiplicity correction. Here: epoch_change_lmm, Holm across all 15 cells x epochs of B + C.

    python -m scripts.lmm_hidden.key_figure_BC [--root <imaging_encoding>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import render

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
NAME = "key_figure_BC"
EP = ["pre", "acute", "subacute", "chronic"]
PANEL = {"trial": "B  cued licks", "iti": "C  ITI licks"}


def _scaled(rows: pd.DataFrame, cell: str, panel: str) -> pd.DataFrame:
    pre_r = rows[rows.epoch == "pre"].groupby("animal").r.mean().mean()
    return pd.DataFrame({"position": cell, "epoch": rows.epoch, "animal": rows.animal, "label": rows.label,
                         "value": rows.r / pre_r, "panel": panel})


def build(root: Path) -> pd.DataFrame:
    out = []
    S = pd.read_csv(root / "attempt_coding" / "recovery_timing_series_lick.csv")
    S = S[np.isfinite(S.neu)]
    out.append(pd.DataFrame({"position": "trial:code", "epoch": S.epoch, "animal": S.animal, "label": S.label,
                             "value": S.neu, "panel": PANEL["trial"]}))
    R = pd.read_csv(root / "iti_licks" / "refit_decoders_sessions.csv")
    R = R[(R.target == "lateral") & np.isfinite(R.r)]
    for st in ("trial", "iti"):
        g = R[R.set == st]
        out.append(_scaled(g[g.kind == "frozen"], f"{st}:frozen", PANEL[st]))
        for ep in EP[1:]:
            post = g[(g.kind == "epoch refit") & (g.epoch == ep)]
            ref = g[(g.kind == "pre, matched n") & (g.vs == ep)]
            if len(post) and len(ref):
                out.append(_scaled(pd.concat([ref, post]), f"{st}:refit_{ep}", PANEL[st]))
    return pd.concat(out, ignore_index=True)


def verify(S: pd.DataFrame) -> pd.DataFrame:
    """Fraction of pre per cell x epoch as the published panels draw it: 1 + mean over animals of (animal post mean -
    animal pre mean), over animals with both."""
    rows = []
    for c, g in S.groupby("position", sort=False):
        pre = g[g.epoch == "pre"].groupby("animal").value.mean()
        for ep in EP[1:]:
            po = g[g.epoch == ep].groupby("animal").value.mean()
            com = po.index.intersection(pre.index)
            if len(com):
                rows.append(dict(position=c, epoch=ep, frac=1 + float((po[com] - pre[com]).mean()), n_animals=len(com),
                                 n_sessions=int((g.epoch == ep).sum())))
    return pd.DataFrame(rows)


LABELS = {"trial:code": "frozen\nfar-contra\ncode (hits)", "trial:frozen": "frozen\nlateral\ndecoder",
          "iti:frozen": "frozen\nlateral\ndecoder",
          **{f"{s}:refit_{e}": f"refit\n{e}\n(vs matched\nn pre)" for s in ("trial", "iti") for e in EP[1:]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    S = build(Path(a.root))
    pd.set_option("display.width", 200)
    print("== n sessions / animals per cell x epoch")
    print(S.groupby(["position", "epoch"]).agg(sessions=("value", "size"), animals=("animal", "nunique")).unstack())
    print("== published quantity (fraction of pre, mean of animals)\n" + verify(S).round(3).to_string(index=False))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = out / f"{NAME}_sessions.csv"
    S.to_csv(src, index=False)
    cells = list(dict.fromkeys(S.position))
    T = render.render(src, NAME, out, title="Key figure 1 B-C: frozen read-out vs epoch refit (lateral lick decoder; "
                      "far-contra code on hits)", ylabel="change in fraction of pre-stroke level", cells=cells,
                      labels=LABELS, panel_by="panel")
    print(T.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

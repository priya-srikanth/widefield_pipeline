"""LMM hidden-slide version of goal page Fig 14.18 S22a, `rec_kn_epoch.png` (deck rec_kn_epoch_2x2): KINEMATIC NORMALITY
by epoch (restitution vs compensation).

Source: scripts/kinematic_normality.py (`session_table`, `epoch_stats`, `draw` Fig 2 = kn_fig2_normality_by_epoch.png,
copied to the page as rec_kn_epoch.png; run of 10-30 16:08) -> <kinematic_normality>/sessions.csv, one row per session
(PS92-95, curated, AUGUST-only pre; pre sessions scored leave-one-session-out, so pre carries real levels).
Values (the four published panels): norm = share of HITS inside the animal's pre 95 % contour (all spouts), norm_fc =
the same at far contra, norm_att = share of ATTEMPTED trials inside, norm_clf = classifier P(pre) on hits (the page says
it separates every post session from pre and is not interpreted; kept because the figure / deck shows it).
Cells: 4 panels x 1 cell; Holm family = all 4 x 3 epochs = 12 (the published figure did no correction).
Published test: `analysis_kit.boot_delta` per epoch vs August pre; * = 95 % CI excludes 0 AND >= 3/4 animals same sign
(epoch_tests.csv `change`).

    python -m scripts.lmm_hidden.rec_kn_epoch [--root <kinematic_normality>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import _paths, render

NAME = "rec_kn_epoch"
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
            "kinematic_normality")
METS = {"norm": ("normality, HITS, all spouts", "all\nspouts"), "norm_fc": ("normality, HITS, far contra", "far\ncontra"),
        "norm_att": ("normality, ATTEMPTED trials", "all\nspouts"),
        "norm_clf": ("classifier P(pre), hits (not interp.)", "all\nspouts")}
EP = ("acute", "subacute", "chronic")


def table(root: Path) -> pd.DataFrame:
    S = pd.read_csv(root / "sessions.csv")
    rows = [pd.DataFrame({"position": f"{m}:x", "panel": lab, "epoch": S.epoch.to_numpy(), "animal": S.animal.to_numpy(),
                          "label": S.label.to_numpy(), "value": S[m].to_numpy(float)}) for m, (lab, _x) in METS.items()]
    D = pd.concat(rows, ignore_index=True)
    return D[np.isfinite(D.value)].reset_index(drop=True)


def animal_change(D: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c, g in D.groupby("position", sort=False):
        a = g.groupby(["epoch", "animal"]).value.mean().unstack("animal")
        for ep in EP:
            if ep in a.index and "pre" in a.index:
                d = (a.loc[ep] - a.loc["pre"]).dropna()
                out.append(dict(position=c, epoch=ep, change=d.mean(), n_animals=len(d)))
    return pd.DataFrame(out)


def published(root: Path) -> pd.DataFrame:
    T = pd.read_csv(root / "epoch_tests.csv")
    T = T[T.measure.isin(list(METS))].copy()
    T["position"] = T.measure + ":x"
    T["boot_star"] = np.where(T.change.astype(bool), "*", "")
    return T[["position", "epoch", "delta", "lo", "hi", "same_sign", "boot_star"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    root, outdir = Path(a.root), Path(a.outdir)
    D = table(root)
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"{_paths.out_name(NAME)}_sessions.csv"
    D.to_csv(f, index=False)
    pd.set_option("display.width", 220)
    print("n sessions / animals per epoch, per cell:")
    print(D.groupby(["position", "epoch"]).agg(sessions=("label", "size"), animals=("animal", "nunique"))
          .unstack("epoch").to_string())
    cells = [f"{m}:x" for m in METS]
    T = render.render(f, NAME, outdir, title="Fig 14.18 S22a: kinematic normality by epoch (August pre, pre scored "
                      "leave-one-session-out), change vs own pre (LMM)",
                      ylabel="change vs pre", cells=cells, labels={f"{m}:x": x for m, (_l, x) in METS.items()},
                      panel_by="panel", ncols=4)
    V = animal_change(D).merge(published(root), on=["position", "epoch"], how="outer")
    V = V.merge(T[["position", "epoch", "beta", "p", "star"]], on=["position", "epoch"], how="left")
    V["match"] = np.isclose(V.change, V.delta, atol=1e-6)
    print(V.round(4).to_string(index=False))
    print(f"verification: {int(V.match.sum())}/{len(V)} cells match the published boot_delta point")
    V.to_csv(outdir / f"{NAME}_verify.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

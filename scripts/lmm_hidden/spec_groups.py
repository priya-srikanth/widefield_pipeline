"""LMM hidden slide for goal page Fig 2.2 `spec_groups` (= scripts/specificity_control.py `specificity_groups.png`, run
--tag gate-attempted_v7g3_t3d_frA --layer2 allsession_encoder_v7g3_t3d_frA_roi, 10-28 08:11; the published PNG is
byte-identical to <root>/specificity_control/gate-attempted_v7g3_t3d_frA/figures/specificity_groups.png).

Five readouts (panels), one row per SESSION:
  * RESIDUAL, cue / lick window: data - frozen pre-stroke encoder prediction (evoked; mean over the six spouts), per area
    in each animal's pre-SD units (`residual_z`), averaged per session over a GROUP's areas present in every session of
    that animal (orofacial contra = SSp-m/SSp-n/MOp/MOs right; orofacial ipsi = the same left; control = visual,
    retrosplenial, SSp trunk / lower limb, both sides; orofacial - control = the all-orofacial mean minus control, per
    session) -- `specificity_control.group_means`, re-done here from the saved per-session x area table.
    Source: <root>/specificity_control/<tag>/sessions.csv.
  * LAYER 2 by group: `allsession_encoder` combined-fit `change_ref` (stored x1e3 -> R2) per session x area, group means
    as above. Source: <root>/<layer2>/allsession_changes_by_session.csv. Its reference epoch is `pre_late` (change beyond
    pre-stroke drift), so pre_late sessions are the `pre` rows here and the early-pre sessions are dropped, exactly as
    the published test (ref_epoch="pre_late").
  * RESIDUAL by ROI, cue and lick (two panels here; one interleaved panel in the published figure): residual_z per
    session of MOp / tjM1 / MOp_rest / MOs / ALM / MOs_rest, each side.
  * LAYER 2 by ROI: change_ref of the same 12 ROIs (ROI targets of the _roi all-session run).
Published test: `frozen_fr_compare.boot` = `analysis_kit.boot_delta` (nested bootstrap animals > sessions) of each epoch
vs the animal's pre (pre_late for Layer 2); * = 95 % CI excludes 0, uncorrected, one test per cell x epoch.
Holm family here: all 48 cells x 3 epochs of the figure (render default).

    python -m scripts.lmm_hidden.spec_groups [--root <imaging_encoding>] [--tag ...] [--layer2 ...] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import render

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
NAME = "spec_groups"
SIDES = ("_left", "_right")
OROFACIAL = [a + s for a in ("SSp-m", "SSp-n", "MOp", "MOs") for s in SIDES]
CONTROL = [a + s for a in ("VISp", "VISam", "VISa", "VISrl", "VISal", "VISl", "VISpm", "RSPagl", "RSPd", "RSPv",
                           "SSp-ll", "SSp-tr") for s in SIDES]
GROUPS = {"orofacial": OROFACIAL, "orofacial contra": [a for a in OROFACIAL if a.endswith("_right")],
          "orofacial ipsi": [a for a in OROFACIAL if a.endswith("_left")], "control": CONTROL}
GL = ["orofacial contra", "orofacial ipsi", "control", "orofacial - control"]
ROIS = ("MOp_right", "MOp_left", "tjM1_right", "tjM1_left", "MOp_rest_right", "MOp_rest_left", "MOs_right", "MOs_left",
        "ALM_right", "ALM_left", "MOs_rest_right", "MOs_rest_left")
PANELS = [("res_cue", "RESIDUAL, cue window (pre-SD units)"), ("res_lick", "RESIDUAL, lick window (pre-SD units)"),
          ("l2", "LAYER 2 (R2, vs pre_late)"), ("roi_cue", "RESIDUAL by ROI, cue window (pre-SD units)"),
          ("roi_lick", "RESIDUAL by ROI, lick window (pre-SD units)"), ("l2roi", "LAYER 2 by ROI (R2, vs pre_late)")]


def _groups(T: pd.DataFrame, val: str, by) -> pd.DataFrame:
    """specificity_control.group_means for one value column: per animal (x window), each group = mean over its areas
    present in EVERY session of that animal; plus orofacial - control per session."""
    rows = []
    for _k, g in T.groupby(list(by)):
        nlab = g.label.nunique()
        have = set(g.groupby("area").label.nunique().loc[lambda s, n=nlab: s == n].index)
        for gname, areas in GROUPS.items():
            use = [a for a in areas if a in have]
            if use:
                s = g[g.area.isin(use)].groupby(["label", "epoch", "animal"])[val].mean().reset_index()
                rows.append(s.assign(area=gname))
    G = pd.concat(rows, ignore_index=True)
    o = G[G.area == "orofacial"].set_index("label")
    c = G[G.area == "control"].set_index("label")
    con = (o[[val]] - c[[val]]).join(o[["epoch", "animal"]]).reset_index().assign(area="orofacial - control")
    return pd.concat([G, con], ignore_index=True)


def build(root: Path, tag: str, layer2: str) -> pd.DataFrame:
    root = Path(root)
    T = pd.read_csv(root / "specificity_control" / tag / "sessions.csv")
    L = pd.read_csv(root / layer2 / "allsession_changes_by_session.csv",
                    usecols=["fit", "animal", "label", "epoch", "area", "change_ref"])
    L = L[L.fit == "combined"].copy()
    L["change_ref"] = L["change_ref"] / 1e3
    out = []

    def add(D, key, val, areas, epoch_map=None):
        D = D[D.area.isin(areas)].copy()
        if epoch_map:
            D = D[D.epoch.isin(epoch_map)].assign(epoch=lambda d: d.epoch.map(epoch_map))
        lab = dict(PANELS)[key]
        out.append(pd.DataFrame({"panel": lab, "position": key + ":" + D.area, "epoch": D.epoch, "animal": D.animal,
                                 "label": D.label, "value": D[val].to_numpy()}))

    for w in ("cue", "lick"):
        G = _groups(T[T.window == w], "residual_z", ("animal",))
        add(G, f"res_{w}", "residual_z", GL)
    l2map = {"pre_late": "pre", "acute": "acute", "subacute": "subacute", "chronic": "chronic"}
    add(_groups(L, "change_ref", ("animal",)), "l2", "change_ref", GL, l2map)
    for w in ("cue", "lick"):
        add(T[T.window == w], f"roi_{w}", "residual_z", [r for r in ROIS if r in set(T.area)])
    add(L, "l2roi", "change_ref", [r for r in ROIS if r in set(L.area)], l2map)
    S = pd.concat(out, ignore_index=True)
    return S[np.isfinite(S.value)].reset_index(drop=True)


def _label(pos: str) -> str:
    a = pos.split(":", 1)[1]
    if a in GL:
        return {"orofacial contra": "orofacial\ncontra", "orofacial ipsi": "orofacial\nipsi", "control": "control",
                "orofacial - control": "orofacial\n- control"}[a]
    base, side = a.rsplit("_", 1)
    return f"{base}\n{'contra' if side == 'right' else 'ipsi'}"


def verify(S: pd.DataFrame, root: Path, tag: str) -> None:
    """mean over animals of (animal post mean - animal pre mean), animals with both arms, vs the published boot tables."""
    d = Path(root) / "specificity_control" / tag
    B = pd.read_csv(d / "boot.csv")
    LB = pd.read_csv(d / "layer2_boot.csv")
    BS = pd.read_csv(d / "split_boot.csv")
    LS = pd.read_csv(d / "layer2_split_boot.csv")
    worst = 0.0
    for pos, D in S.groupby("position", sort=False):
        key, area = pos.split(":", 1)
        pre = D[D.epoch == "pre"].groupby("animal").value.mean()
        for ep in ("acute", "subacute", "chronic"):
            po = D[D.epoch == ep].groupby("animal").value.mean()
            com = [a for a in po.index if a in pre.index]
            mine = (po[com] - pre[com]).mean() if len(com) >= 2 else np.nan
            if key.startswith("res_"):
                p = B[(B.window == key[4:]) & (B.area == area) & (B.epoch == ep)]
            elif key == "l2":
                p = LB[(LB.area == area) & (LB.epoch == ep)]
            elif key.startswith("roi_"):
                p = BS[(BS.window == key[4:]) & (BS.area == area) & (BS.epoch == ep)]
            else:
                p = LS[(LS.area == area) & (LS.epoch == ep)]
            pub = p.point.iloc[0] if len(p) else np.nan
            diff = abs(mine - pub) / max(abs(pub), 1e-9) if np.isfinite(pub) else np.nan
            worst = np.nanmax([worst, diff])
            flag = "" if (np.isfinite(diff) and diff < 1e-6) else "  <-- differs"
            print(f"  {pos:32s} {ep:9s} n_an={len(com)} extracted {mine:+.5f}  published {pub:+.5f}{flag}")
    print(f"  worst relative difference: {worst:.2e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA")
    ap.add_argument("--layer2", default="allsession_encoder_v7g3_t3d_frA_roi")
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    S = build(Path(a.root), a.tag, a.layer2)
    print(S.groupby(["panel", "epoch"]).agg(sessions=("label", "nunique"), animals=("animal", "nunique")))
    verify(S, Path(a.root), a.tag)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = out / f"{NAME}_sessions.csv"
    S.to_csv(src, index=False)
    have = set(S.position)
    cells = [f"{k}:{a}" for k, _ in PANELS for a in (GL if k in ("res_cue", "res_lick", "l2") else ROIS)
             if f"{k}:{a}" in have]
    R = render.render(src, NAME, out, title=f"Is the post-stroke change SPECIFIC to orofacial cortex? Frozen frame ({a.tag}); "
                      "contra = right", ylabel="change vs own pre", cells=cells, labels={c: _label(c) for c in cells},
                      panel_by="panel", ncols=1)
    print(R.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

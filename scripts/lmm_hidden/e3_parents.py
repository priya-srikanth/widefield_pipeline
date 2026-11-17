"""LMM hidden-slide companions of three PI-deck figures that sabatinilab-e3 rendered on 11-03 straight through
scripts/lmm_change_fig (no extractor, no _draw.json kept; REDRAW_INVENTORY_2026-11-06 rows "lmm_change_fig_e3"):

  behaviour_hits_by_spout_lmm   hit rate per spout            (deck: "Mixed-model stars: hit rate by spout")
  d_acc_cue_lmm                 frozen decoder by spout, cue  (deck: "Mixed-model stars: frozen decoder by spout, cue")
  frozen_refit_pooled_lmm       frozen / refit / gap pooled   (deck: "Mixed-model stars: frozen / refit / gap pooled")
  d_refitgap_lmm                refit - frozen gap by spout   (deck: "Mixed-model stars: refit - frozen gap by spout")

This module makes them reproducible (a8 11-17): each reads its PARENT figure's per-session table, writes the render
input (position, epoch, animal, value [, label]) and draws through render.render (= lmm_change_fig.statistics + draw).
The parent table is given explicitly (--src); nothing is defaulted to a tree.

  behaviour   --src <rebuild_figs/<tag>/data/behaviour_hits_by_spout_figdata.pkl>   (key "M": one row per session,
              hit__<DAQ position> = the parent's per-session hit rate, already curated / August-only pre / epoch-labelled
              by fig_behaviour_hits.compute; the parent's dots are these rows)
  d_acc       --src <grant_figures/<tag>/epoch/data/epoch_acc_by_position_cue_working_sessions.csv>  (epoch, position
              nI..fC, animal, value [, session]; pre = one cross-validated value per animal, as the parent)
  refitgap    --src <grant_figures/<tag>/epoch/data/epoch_5rmgap_frozen_vs_refit_cue_working_sessions.csv>  (the
              matched gap family's per-session refit - frozen accuracy WITH pre rows; the parent figure
              epoch_5rmgapdelta_* plots exactly these values minus the animal's own pre mean -- checked to 8e-17 on the
              rebuild, 11-17 -- but its own sidecar has no pre rows, which the model needs)
  frozen      --src <rebuild_figs/<tag>/data/frozen_refit_pooled_figdata.pkl>  (key "w": epoch, animal, session_idx,
              frozen, refit, gap; gap = refit - frozen on the same trials)

    python -m scripts.lmm_hidden.e3_parents {behaviour,d_acc,refitgap,frozen} --src PATH [--outdir DIR]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from scripts.lmm_hidden import render

DAQ_POS = {"close_ipsi": "close_L", "close_mid": "close_center", "close_contra": "close_R",
           "far_ipsi": "far_L", "far_mid": "far_center", "far_contra": "far_R"}   # fig_behaviour_hits.POS (lesion LEFT)
SPEC = {
    "behaviour": dict(name="behaviour_hits_by_spout_lmm", ylabel="hit rate - own pre",
                      title="Hit rate per spout, camera working period (contact / all trials in the period)"),
    "d_acc": dict(name="d_acc_cue_lmm", ylabel="decoder accuracy - own pre",
                  title="Frozen decoder accuracy by spout, cue window, working trials (imaging 2.3)"),
    "refitgap": dict(name="d_refitgap_lmm", ylabel="refit - frozen gap - own pre",
                     title="Refit - frozen gap by spout (training-set matched), cue window, working trials"),
    "frozen": dict(name="frozen_refit_pooled_lmm", ylabel="accuracy - own pre", figsize=(9.0, 4.5),
                   title="Frozen / refit / gap, pooled over spouts, cue, working"),
}


def build(kind: str, src: Path) -> pd.DataFrame:
    if kind == "behaviour":
        M = pickle.load(open(src, "rb"))["M"]
        rows = [pd.DataFrame({"position": p, "epoch": M.epoch, "animal": M.animal, "label": M.key,
                              "value": M[f"hit__{c}"].astype(float)}) for p, c in DAQ_POS.items()]
        S = pd.concat(rows, ignore_index=True)
    elif kind in ("d_acc", "refitgap"):
        S = pd.read_csv(src)
        S = S.rename(columns={"session": "label"}) if "session" in S else S
    else:
        w = pickle.load(open(src, "rb"))["w"]
        S = w.melt(id_vars=["epoch", "animal", "session_idx"], value_vars=["frozen", "refit", "gap"],
                   var_name="position", value_name="value").rename(columns={"session_idx": "label"})
    return S.dropna(subset=["value"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=sorted(SPEC))
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    sp = SPEC[a.kind]
    S = build(a.kind, a.src)
    print(f"source: {a.src}")
    print(S.groupby(["position", "epoch"]).agg(sessions=("value", "size"), animals=("animal", "nunique"))
          .unstack("epoch").to_string())
    print("PER ANIMAL (rule 8): animal epoch mean - own pre mean")
    m = S.groupby(["position", "animal", "epoch"]).value.mean().unstack("epoch")
    for ep in ("acute", "subacute", "chronic"):
        m[f"d_{ep}"] = m[ep] - m["pre"]
    print(m[[c for c in m.columns if str(c).startswith("d_")]].round(3).to_string())
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{sp['name']}_sessions.csv"
    S.to_csv(f, index=False)
    cells = ["frozen", "refit", "gap"] if a.kind == "frozen" else None
    T = render.render(f, sp["name"], out, title=sp["title"], ylabel=sp["ylabel"], cells=cells,
                      figsize=sp.get("figsize"))
    pd.set_option("display.width", 220)
    print(T.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Layer 2 RANDOM-SPLIT null (Priya 10-24: "what if we do a random re-sampling of early and late pre-stroke sessions" ->
"let's do random splits"). August pre sessions only (June excluded: kinematics off, no FaceRhythm).

Per animal, every way of choosing k August pre sessions as a PSEUDO-epoch (k = the date-ordered pre_late size; the rest
are the reference) is fitted in the "combined" layout of `allsession_encoder` (pseudo + acute + subacute + chronic
changes from that reference, one fit), and `change_ref` is read for every held-out session. Per split: the pseudo
epoch's mean change_ref (the null: what a set of k pre sessions gains from its own change, with no stroke) and each post
epoch's. Removes TIME on purpose (random, not date-ordered), so it controls session noise and session count, not slow
drift -- that is the trend model's job. kappa is fixed per animal at its combined-fit value (no per-split grid search).

Usage: python scripts/layer2_random_splits.py PS92 [--max-splits 70]
Output: <E>/allsession_encoder_v7g3_t3d_ref/random_splits_<animal>.csv (one row per split x epoch)
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import allsession_encoder as ae
from wfield_local import epochs
from wfield_local import frozen_encoder as fe

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
SRC = E / "frozen_encoder_gate-attempted_v7g3_t3d"
L2 = E / "allsession_encoder_v7g3_t3d_ref"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("animal")
    ap.add_argument("--max-splits", type=int, default=70)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    an = a.animal
    pre_stats = [s for s in pickle.loads((SRC / "fit" / "pre_stats.pkl").read_bytes()) if s["animal"] == an]
    post_stats = [s for s in pickle.loads((L2 / "fit" / "post_stats.pkl").read_bytes()) if s["animal"] == an]
    S = pre_stats + post_stats
    col_var = pre_stats[0]["col_var"]
    areas = [x for x in pre_stats[0]["area_names"] if all(x in s["area_names"] for s in S)]
    sd = np.sqrt(sum(s["sxx"] for s in pre_stats) / sum(s["n"] for s in pre_stats))
    ok = sd > 1e-9
    scale = np.where(ok, sd, np.inf)
    sess = []
    for s in S:
        xy, y2 = fe._subset(s, areas)
        sess.append({"label": s["label"], "epoch": s["epoch"], "G": s["XtX"] / np.outer(scale, scale),
                     "b": xy / scale[:, None], "yy": y2})
    js = sorted((SRC / "joint" / "sessions").glob(f"{an}_*.json"))[0]
    alphas = json.loads(js.read_text(encoding="utf-8"))["alphas"]
    groups_of = np.array([fe.FGROUP[v] for v in col_var])
    pen_of = np.array([fe.FPEN[v] for v in col_var])
    kappa = float(np.load(L2 / "fit" / f"{an}_combined.npz")["kappa"])
    pre = sorted([x["label"] for x in sess if x["epoch"] == "pre"], key=epochs.days_since_stroke)
    k = len(pre) - len(pre) // 2                         # = the date-ordered pre_late size
    splits = list(itertools.combinations(pre, k))
    rng = np.random.default_rng(a.seed)
    if len(splits) > a.max_splits:
        splits = [splits[i] for i in sorted(rng.choice(len(splits), a.max_splits, replace=False))]
    date_late = tuple(pre[len(pre) // 2:])
    ai = {ar: areas.index(ar) for ar in ("cortex", "SSp-m_left", "SSp-m_right") if ar in areas}
    rows = []
    for si, sp in enumerate(splits):
        pset = set(sp)
        ss = [{**x, "epoch": "pseudo"} if x["label"] in pset else x for x in sess]
        f = ae.fit_animal(ss, alphas, groups_of, pen_of, ok, ("pseudo", *ae.POST_EPOCHS), kappa=kappa)
        for x in ss:
            if x["epoch"] == "pre":
                continue
            for ar, i in ai.items():
                g = (f["sse"]["ref"][x["label"]][i] - f["sse"]["all"][x["label"]][i]) / max(x["yy"][i], 1e-12) * 1e3
                rows.append({"animal": an, "split": si, "date_ordered": sp == date_late, "epoch": x["epoch"],
                             "label": x["label"], "area": ar, "change_ref": g})
        print(f"[random splits] {an} {si + 1}/{len(splits)}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(L2 / f"random_splits_{an}.csv", index=False)
    M = T.groupby(["split", "area", "epoch"]).change_ref.mean().unstack("epoch")
    for ar in ai:
        m = M.xs(ar, level="area")
        msg = [f"{an} {ar}: pseudo null median {m['pseudo'].median():.1f} [5-95% {m['pseudo'].quantile(.05):.1f}, "
               f"{m['pseudo'].quantile(.95):.1f}]"]
        for ep in ae.POST_EPOCHS:
            if ep in m:
                p = float((m["pseudo"] >= m[ep]).mean())
                msg.append(f"{ep} median {m[ep].median():.1f} (P[pseudo >= {ep}] = {p:.2f})")
        print("; ".join(msg), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

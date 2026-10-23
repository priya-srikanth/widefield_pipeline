"""Compare frozen-encoder RUNNING models (Priya 10-23: pick v7 among signed linear / spline / forward-only thresholds,
with and without the backward "push" term).

Per tag (frozen_encoder_gate-attempted_<tag>), animals first (rule 8):
  1. frozen R2 (cortex; all / cue / lick frames; pre = leave-one-session-out) per animal x epoch, joint model
  2. unique dR2 of lick (joint - no_lick) and of other (joint - no_other), per animal x epoch
  3. push dR2 = joint(<tag>push) - joint(<tag>), matched sessions
  4. fitted running curve (cortex, summed over lags) at a few speeds, per animal
  5. acute far-contra lick-window decomposition (contralesional SSp-m_left + cortex if present), per animal

Usage: python scripts/frozen_running_compare.py [--tags v6 v6spl v6thr3 v6thr1 v7fwd15 v7fwd15push]
Writes <E>/qc/frozen_running_compare_<first>_to_<last>.csv (long table) and prints the tables.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

E = Path(r"M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v36\imaging_encoding")
EPOCHS = ["pre", "acute", "subacute", "chronic"]
#: running mode + first knot per tag (knot0 is applied at run time from FROZEN_RUN_KNOT0, not stored in templates)
RUN = {"v6": ("linear", None), "v6spl": ("spline", None), "v6thr3": ("spline_thr", 3.0), "v6thr1": ("spline_thr", 1.0),
       "v7fwd15": ("spline_thr", 1.5), "v7fwd15push": ("spline_thr", 1.5), "v7pushfree": ("spline_thr", 1.5)}
#: push dR2 = joint(with push) - joint(without), matched sessions
PUSH_PAIRS = {"v7fwd15push": "v7fwd15", "v7pushfree": "v7fwd15"}
SPEEDS = np.array([-3.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])


def folder(tag: str) -> Path:
    return E / f"frozen_encoder_gate-attempted_{tag}"


def session_r2(tag: str, variant: str) -> pd.DataFrame:
    rows = []
    for js in sorted((folder(tag) / variant / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        for w, x in (m.get("frozen_r2_cortex") or {}).items():
            rows.append({"tag": tag, "variant": variant, "label": m["label"], "animal": m["animal"],
                         "epoch": m["epoch"], "window": w, "r2": x})
    return pd.DataFrame(rows)


def by_animal(T: pd.DataFrame, val: str) -> pd.DataFrame:
    A = T.groupby(["tag", "animal", "window", "epoch"])[val].mean().unstack("epoch")
    return A.reindex(columns=[e for e in EPOCHS if e in A.columns])


def running_curves(tag: str) -> pd.DataFrame:
    import wfield_local.frozen_encoder as fe
    mode, k0 = RUN.get(tag, ("linear", None))
    tp = folder(tag) / "fit" / "templates.pkl"
    if not tp.exists():
        return pd.DataFrame()
    T = pickle.loads(tp.read_bytes())
    rows = []
    for an, t in T.items():
        f = folder(tag) / "fit" / f"{an}_joint.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        vcol, areas = z["vcol"], list(z["areas"])
        ci = areas.index("cortex")
        sel = vcol == "running"
        beta = (z["beta_all"][:, ci] / z["scale"])[sel]
        knots = list(t.get("running_knots") or [])
        old = os.environ.get("FROZEN_RUN_KNOT0")          # the run's first knot (template value = the 3 mm/s threshold)
        os.environ["FROZEN_RUN_KNOT0"] = str(k0 if k0 is not None else (knots[0] if knots else fe.RUN_KNOT0_DEFAULT))
        try:
            y = fe.running_curve(beta, fe._knots(knots) if knots else knots, SPEEDS, t.get("running_cap"), mode=mode)
        finally:
            if old is not None:
                os.environ["FROZEN_RUN_KNOT0"] = old
            else:
                os.environ.pop("FROZEN_RUN_KNOT0", None)
        rows += [{"tag": tag, "animal": an, "speed": s, "y_x1e3": v * 1e3} for s, v in zip(SPEEDS, y)]
    return pd.DataFrame(rows)


def decomposition(tag: str) -> pd.DataFrame:
    f = E / "qc" / f"lp_encoder_decompose_frozen_encoder_gate-attempted_{tag}_joint_lick_by_animal.csv"
    if not f.exists():
        return pd.DataFrame()
    D = pd.read_csv(f)
    D = D[(D.pos == "far_contra") & (D.epoch == "acute") & D.region.isin(["SSp-m_left", "cortex"])]
    cols = ["d_data", "d_full", "d_target", "d_exec_locked", "d_exec_within", "d_lick", "d_other", "d_contact", "d_residual"]
    D = D[["animal", "region"] + cols].copy()
    D[cols] = D[cols] * 1e3
    D.insert(0, "tag", tag)
    return D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=list(RUN))
    a = ap.parse_args()
    tags = [t for t in a.tags if folder(t).exists()]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)

    J = pd.concat([session_r2(t, "joint") for t in tags], ignore_index=True)
    print("\n=== 1. frozen R2, cortex, joint (pre = LOSO), per animal")
    for w in ("all", "cue", "lick"):
        print(f"\n--- {w} frames")
        print(by_animal(J[J.window == w], "r2").round(3).to_string())

    out = [J.assign(measure="r2")]
    for g in ("lick", "other"):
        N = pd.concat([session_r2(t, f"no_{g}") for t in tags], ignore_index=True)
        if N.empty:
            continue
        M = J.merge(N, on=["tag", "label", "animal", "epoch", "window"], suffixes=("", "_no"))
        M["d"] = (M.r2 - M.r2_no) * 1e3
        print(f"\n=== 2. unique dR2 x1e3 of {g} (joint - no_{g}), cortex, per animal")
        for w in ("all", "lick"):
            print(f"\n--- {w} frames")
            print(by_animal(M[M.window == w], "d").round(1).to_string())
        out.append(M[["tag", "label", "animal", "epoch", "window", "d"]].rename(columns={"d": "r2"}).assign(measure=f"uniq_{g}_x1e3"))

    for t, t0 in PUSH_PAIRS.items():
        if t in tags and t0 in tags:
            P = J[J.tag == t].merge(J[J.tag == t0], on=["label", "animal", "epoch", "window"], suffixes=("", "_base"))
            P["d"] = (P.r2 - P.r2_base) * 1e3
            P["tag"] = t
            print(f"\n=== 3. push dR2 x1e3 = joint({t}) - joint({t0}), cortex, per animal")
            for w in ("all", "cue", "lick"):
                print(f"\n--- {w} frames")
                print(by_animal(P[P.window == w], "d").round(1).to_string())
            out.append(P[["tag", "label", "animal", "epoch", "window", "d"]].rename(columns={"d": "r2"}).assign(measure="push_x1e3"))

    C = pd.concat([running_curves(t) for t in tags], ignore_index=True)
    if len(C):
        print("\n=== 4. fitted running response, cortex (x1e3, summed over lags) vs speed mm/s (negative = backward)")
        print(C.pivot_table(index=["tag", "animal"], columns="speed", values="y_x1e3").round(2).to_string())

    D = pd.concat([decomposition(t) for t in tags], ignore_index=True)
    if len(D):
        print("\n=== 5. acute far-contra, lick window, frozen post - pre (x1e3), per animal")
        print(D.sort_values(["region", "animal", "tag"]).round(2).to_string(index=False))

    L = pd.concat(out, ignore_index=True)
    dst = E / "qc" / f"frozen_running_compare_{tags[0]}_to_{tags[-1]}.csv"
    L.to_csv(dst, index=False, float_format="%.5g")
    if len(C):
        C.to_csv(dst.with_name(dst.stem + "_curves.csv"), index=False, float_format="%.5g")
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()

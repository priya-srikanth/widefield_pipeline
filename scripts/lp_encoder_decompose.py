"""Is the acute far-contra post-lick activity drop the MOVEMENT or the TARGET? (Priya 10-21 night, LP_licks item 1:
"yes do 1-3 as provisional"; ENCODING_FINDINGS 3.3.)

The movement-first encoder (`combined_encoding`) saves, per session, the full model's cue-aligned prediction split by
regressor group (`peth`, parts data / full / task / instructed / uninstructed / pos; per position x area x time). Here,
per session and position: each part's EVOKED value = mean over the post-cue lick window (0.5-1.5 s, the encoder's
`c0.5-1.5` response bin) minus the pre-cue baseline (-0.5-0 s). Then per animal x position: acute (and chronic) minus
pre, for the data and for each part, plus the residual (data - full). If the acute far-contra drop in the DATA is
carried by `instructed` (licks + exec angle + tongue + jaw), it is the executed movement; if by `task` / `pos`, the
target (`pos_part` = the position-specific part of task); if by the residual, neither model explains it.

`instructed` here is the movement the video SAW (smaller / incomplete attempts lower it), `task` = cue + arrival +
position blocks (`pos` = the position-specific part of task). Parts are additive: full = task + instructed +
uninstructed + intercept (+ drift, not in the PETH), so the evoked parts sum to the evoked full prediction.

Run folder (default): movement_first_subset-drop_licks-combined_gate-attempted (attempted-trial gate, cam4 + cam1 lick
timing; cam4 only where cam1 is missing -- each session json's `lick_set` says which).
Output: <E>/qc/lp_encoder_decompose{,_by_animal}.csv.   Usage: python scripts/lp_encoder_decompose.py [--folder ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WIN_S = (0.5, 1.5)      # post-cue lick window (the encoder's c0.5-1.5 response bin)
BASE_S = (-0.5, 0.0)    # pre-cue baseline
REGIONS = ("cortex", "SSp-m_left", "SSp-m_right", "MOs_left", "MOs_right", "MOp_left", "MOp_right")


def session_rows(js: Path) -> list[dict]:
    meta = json.loads(js.read_text())
    z = np.load(js.with_suffix(".npz"))
    peth, off = z["peth"], z["peth_off_s"]
    w = (off >= WIN_S[0]) & (off < WIN_S[1])
    b = (off >= BASE_S[0]) & (off < BASE_S[1])
    parts, areas = meta["peth_parts"], meta["areas"]
    rows = []
    for pi, pos in enumerate(meta["pos_order"]):
        for reg in REGIONS:
            if reg not in areas:
                continue
            ai = areas.index(reg)
            ev = {("pos_part" if p == "pos" else p): float(np.nanmean(peth[ci, pi, ai, w]) - np.nanmean(peth[ci, pi, ai, b])) for ci, p in enumerate(parts)}
            ev["residual"] = ev["data"] - ev["full"]
            rows.append({"label": meta["label"], "animal": meta["animal"], "epoch": meta["epoch"],
                         "lick_set": meta["lick_set"], "pos": pos, "region": reg, **ev})
    return rows


def main(argv=None) -> int:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="movement_first_subset-drop_licks-combined_gate-attempted")
    a = ap.parse_args(argv)
    E = oo.root(PathResolver()) / oc.RUN / "imaging_encoding"
    sess = sorted((E / a.folder / "sessions").glob("*.json"))
    T = pd.DataFrame([r for js in sess for r in session_rows(js)])
    T = T[T.label.str[-4:] >= "0801"]                                  # August-only pre baseline (primary)
    parts = ["data", "full", "task", "pos_part", "instructed", "uninstructed", "residual"]
    A = T.groupby(["animal", "pos", "region", "epoch"])[parts].mean()
    n = T.groupby(["animal", "pos", "region", "epoch"]).label.nunique().rename("n_sess")
    A = A.join(n).reset_index()
    out = []
    for (an, pos, reg), g in A.groupby(["animal", "pos", "region"]):
        g = g.set_index("epoch")
        if "pre" not in g.index:
            continue
        for ep in ("acute", "subacute", "chronic"):
            if ep in g.index:
                d = {f"d_{p}": g.loc[ep, p] - g.loc["pre", p] for p in parts}
                out.append({"animal": an, "pos": pos, "region": reg, "epoch": ep, "n_pre": int(g.loc["pre", "n_sess"]),
                            "n_post": int(g.loc[ep, "n_sess"]), "pre_data": g.loc["pre", "data"], **d})
    D = pd.DataFrame(out)
    qc = E / "qc"
    assert_writable(qc)
    T.to_csv(qc / "lp_encoder_decompose.csv", index=False, float_format="%.5g")
    D.to_csv(qc / "lp_encoder_decompose_by_animal.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    cols = ["n_post", "pre_data", "d_data", "d_full", "d_task", "d_pos_part", "d_instructed", "d_uninstructed", "d_residual"]
    print(f"{len(sess)} sessions; lick sets: {T.drop_duplicates('label').lick_set.value_counts().to_dict()}")
    for reg in ("cortex", "SSp-m_left", "SSp-m_right"):
        print(f"\n=== {reg}: evoked ({WIN_S} minus {BASE_S} s post-cue), epoch minus pre, per animal")
        S = D[(D.region == reg) & (D.epoch.isin(["acute", "chronic"]))].set_index(["pos", "epoch", "animal"])[cols]
        print((S.assign(**{c: S[c] * 1e3 for c in cols if c != "n_post"})).round(2).to_string())
    print(f"\n-> {qc / 'lp_encoder_decompose_by_animal.csv'}  (values x1e3 above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

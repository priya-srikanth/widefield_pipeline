"""CC-B overnight rebuild 11-23, step 2 check: trial classes on the adopted jaw rule (``WIDEFIELD_JAW_CALL=3d_mouth`` +
the cap jaw table) vs production, per class x spout x epoch, for the s1 class snapshot AND the camera-tagged
trial_attempts table (whose ``gate`` column must say camera).

    python scripts/caption_review_1123/overnight_classes_compare.py --old-s1 A.csv.gz --new-s1 B.csv.gz \
        --old-att A.csv --new-att B.csv --out DIR

Writes (refuses to overwrite) under --out:
  s1_class_change_pos_epoch.csv      old cls_detail -> new cls_detail counts per position x epoch (all trials)
  s1_class_counts_pos_epoch.csv      counts per class x position x epoch, old / new / diff (all trials, and working)
  s1_working_flag_change.csv         trials whose working / cls_final changed (should be few: jaw-only and no response
                                     are both "no lick" for the stop rule)
  attempts_class_change_pos_epoch.csv old (reference gate, production) cls vs new (camera) cls_raw / cls per pos x epoch
  attempts_gate_check.csv            gate column values and canonical counts, old vs new
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]
EPS = ["pre", "acute", "subacute", "chronic"]


def _epoch(labels):
    from wfield_local import epochs
    return {lab: epochs.epoch_of(lab) for lab in pd.unique(labels)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-s1", type=Path, required=True)
    ap.add_argument("--new-s1", type=Path, required=True)
    ap.add_argument("--old-att", type=Path, required=True)
    ap.add_argument("--new-att", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    names = ["s1_class_change_pos_epoch", "s1_class_counts_pos_epoch", "s1_working_flag_change",
             "attempts_class_change_pos_epoch", "attempts_gate_check"]
    outs = {n: a.out / f"{n}.csv" for n in names}
    clash = [str(p) for p in outs.values() if p.exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite {clash}")
    a.out.mkdir(parents=True, exist_ok=True)

    O, N = pd.read_csv(a.old_s1), pd.read_csv(a.new_s1)
    k = ["label", "cue_idx"]
    M = O.merge(N, on=k, suffixes=("_old", "_new"), how="outer", indicator=True)
    print("s1 rows: old", len(O), "new", len(N), "matched", int((M._merge == "both").sum()))
    M = M[M._merge == "both"].copy()
    ep = _epoch(M.label)
    M["epoch"] = M.label.map(ep)
    M["position"] = M.position_old
    ch = (M.groupby(["position", "epoch", "cls_detail_old", "cls_detail_new"]).size().rename("n").reset_index())
    ch.to_csv(outs["s1_class_change_pos_epoch"], index=False)
    rows = []
    for sub, m in (("all", M), ("working_new", M[M.working_new.astype(bool)])):
        for side in ("old", "new"):
            c = m.groupby(["position", "epoch", f"cls_detail_{side}"]).size().rename("n").reset_index()
            c = c.rename(columns={f"cls_detail_{side}": "cls"})
            c["side"], c["set"] = side, sub
            rows.append(c)
    C = pd.concat(rows).pivot_table(index=["set", "position", "epoch", "cls"], columns="side", values="n",
                                    fill_value=0).reset_index()
    C["diff"] = C.new - C.old
    C.to_csv(outs["s1_class_counts_pos_epoch"], index=False)
    wf = M[(M.working_old != M.working_new) | (M.cls_final_old != M.cls_final_new)]
    wf[k + ["position", "epoch", "cls_detail_old", "cls_detail_new", "cls_final_old", "cls_final_new", "working_old",
            "working_new"]].to_csv(outs["s1_working_flag_change"], index=False)
    print("\nall-trial class change (old cls_detail -> new):")
    print(pd.crosstab(M.cls_detail_old, M.cls_detail_new, margins=True).to_string())
    print(f"\nworking flag changed on {int((M.working_old != M.working_new).sum())} trials; cls_final changed on "
          f"{int((M.cls_final_old != M.cls_final_new).sum())}")
    print(pd.crosstab(M.cls_final_old, M.cls_final_new).to_string())
    jo = C[(C.cls == "jaw_only")].pivot_table(index=["set", "epoch"], columns="position", values=["old", "new"])
    print("\njaw_only (cls_detail) counts by set x epoch x position:\n" + jo.to_string())

    A0, A1 = pd.read_csv(a.old_att), pd.read_csv(a.new_att)
    kk = ["label", "trial_id"]
    MA = A0.merge(A1, on=kk, suffixes=("_old", "_new"))
    print("\nattempts rows: old", len(A0), "new", len(A1), "matched", len(MA))
    MA["position"], MA["epoch"] = MA.pos_old, MA.epoch_old
    raw = "cls_raw" if "cls_raw" in A1 else "cls"
    MA["cls_raw_new"] = MA[raw] if raw in MA else MA["cls_new"]
    ac = MA.groupby(["position", "epoch", "cls_old", "cls_raw_new", "cls_new", "canonical_old", "canonical_new"]
                    ).size().rename("n").reset_index()
    ac.to_csv(outs["attempts_class_change_pos_epoch"], index=False)
    g = pd.DataFrame({"table": ["old", "new"],
                      "gate_values": [";".join(sorted(map(str, A0.gate.unique()))), ";".join(sorted(map(str, A1.gate.unique())))],
                      "n_rows": [len(A0), len(A1)],
                      "canonical": [int(A0.canonical.sum()), int(A1.canonical.sum())]})
    g.to_csv(outs["attempts_gate_check"], index=False)
    print(g.to_string(index=False))
    print("\nattempts trial-level class, old (production) -> new cls_raw:")
    print(pd.crosstab(MA.cls_old, MA.cls_raw_new, margins=True).to_string())
    print("\nnew canonical (camera working period) vs old canonical (reference gate):")
    print(pd.crosstab(MA.canonical_old, MA.canonical_new, margins=True).to_string())
    for p in outs.values():
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

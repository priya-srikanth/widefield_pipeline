"""CC-B overnight rebuild 11-23, step 4 input: the 18 Fig 3B hit-rate cells of the FINAL six-position family on the
new jaw rule (3d_mouth). The rule moves a few no-lick trials to / from "unknown", which leaves the camera working
period, so 11 of 642 session x position hit rates change (max 0.013). This script

  1. computes per session x position hit rate (contact / working-period trials; working = cls_final in contact,
     incomplete, jaw_only, no_response) from the production and the new s1 class snapshots;
  2. maps each changed cell onto the family's own session table (`hit_rate/epoch_1b_behaviour_by_position_sessions.csv`,
     no date column) by its exact old value; where the 1b value is not an exact unique match (3 cells) the s1 delta is
     added to the closest row of that animal x epoch x position;
  3. re-tests with `c2_scalar_family_aligned._test` (4,000 draws, seed 0) after checking that the unmodified table
     reproduces the FINAL family's hit-rate cells exactly;
  4. writes the FINAL scalar table with the 18 hit-rate cells replaced (input of final_family_with_s12.py).

    python overnight_family_hitrate.py <old s1> <new s1> <1b sessions csv> <FINAL scalar csv> <out dir>
Needs `c2_scalar_family_aligned` on sys.path (worktree wf_ccb_final1123 scripts/). Refuses to overwrite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

W = ["contact", "incomplete", "jaw_only", "no_response"]
P = {"close_L": "nI", "close_center": "nM", "close_R": "nC", "far_L": "fI", "far_center": "fM", "far_R": "fC"}


def hits(f):
    from wfield_local import epochs
    c = pd.read_csv(f)
    w = c[c.cls_final.isin(W) & (c.position != "unknown")]
    h = w.groupby(["label", "position"]).cls.apply(lambda s: (s == "contact").mean()).rename("hit").reset_index()
    h["animal"], h["epoch"], h["pos"] = h.label.str[:4], h.label.map(epochs.epoch_of), h.position.map(P)
    return h.set_index(["label", "position"])


def run(df, F):
    base = df[df.epoch == "pre"].groupby(["animal", "position"]).value.mean()
    df = df.copy()
    df["chg"] = df.value - base.reindex(pd.MultiIndex.from_frame(df[["animal", "position"]])).to_numpy()
    return F._test(F._long(df, "chg", F.HIT_POS, pos_col="position"), "hit_rate", 4000)


def main(old, new, lb, final, out):
    import c2_scalar_family_aligned as F
    out = Path(out)
    outs = [out / "epoch_1b_sessions_jaw3dmouth_ccB1123.csv", out / "hit_rate_cells_jaw3dmouth_ccB1123.csv",
            out / "c2_scalar_family_FINAL_hitjaw3dmouth_ccB1123_boot4000_cam3on_INPUT.csv"]
    if any(p.exists() for p in outs):
        sys.exit(f"refusing to overwrite {outs}")
    o, n = hits(old), hits(new)
    d = n.hit - o.hit
    lv = pd.read_csv(lb)
    lv2 = lv.copy()
    for k in d[d.abs() > 0].index:
        r = o.loc[k]
        cand = lv[(lv.animal == r.animal) & (lv.position == r.pos) & (lv.epoch == r.epoch)]
        i = (cand.value - r.hit).abs().idxmin()
        lv2.loc[i, "value"] = lv.loc[i, "value"] + d.loc[k]
        print(k, r.epoch, f"old {r.hit:.4f} -> new {n.loc[k].hit:.4f}; 1b row {i} (|1b - s1 old| = "
              f"{abs(lv.loc[i, 'value'] - r.hit):.4f})")
    A, B = run(lv, F), run(lv2, F)
    fin = pd.read_csv(final)
    fh = fin[fin.metric == "hit_rate"].merge(A, on=["pos", "epoch"], suffixes=("_f", ""))
    assert (fh.p - fh.p_f).abs().max() < 1e-12 and (fh.boot_lo - fh.boot_lo_f).abs().max() < 1e-12, "repro failed"
    Bi = B.set_index(["pos", "epoch"])
    for i, r in fin[fin.metric == "hit_rate"].iterrows():
        for c in ["n_sessions", "n_pre_sessions", "mean_change", "p", "boot_change", "boot_lo", "boot_hi"]:
            if c in Bi:
                fin.loc[i, c] = Bi.loc[(r.pos, r.epoch), c]
    out.mkdir(parents=True, exist_ok=True)
    lv2.to_csv(outs[0], index=False)
    B.to_csv(outs[1], index=False)
    fin.to_csv(outs[2], index=False)
    for p in outs:
        print("wrote", p)


if __name__ == "__main__":
    main(*sys.argv[1:6])

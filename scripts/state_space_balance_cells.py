"""Hemispheric balance by TARGET x executed LANDING (Priya 11-07: "balance follows the target, not the landing" -- key?).

The like-for-like runs (`state_space_recovery_trials.py --condition on_target | landing`) average the balance over
whichever far spouts keep >= 5 trials, and acutely far-contra drops out of both (0-3 % of its first licks land on
target), so those columns do not compare the same spouts. This tests it directly, cell by cell:

  same LANDING, different TARGET: e.g. first licks that landed in the far-ipsi bin, aimed at far-contra vs at far-ipsi;
  same TARGET, different LANDING: far-contra-aimed licks that landed on target vs elsewhere.

Balance = log2 contralesional (_right) / ipsilesional (_left) response size of the orofacial areas (SSp-m, SSp-n, SSs),
0-1 s after the cue minus -0.5-0 s, on the cell's trial mean (FINDINGS 0b measure; same as `balance()` in the trials
script). Cells pool the trials of all sessions of an animal x epoch (n >= MIN_TRIALS) and are also kept per session.
Also per session x TARGET over all first-lick trials (landing ignored; the "target" grouping).
Output: Widefield/state_space_recovery/landing/balance_cells{,_sessions,_target}.csv. Exploratory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

OUT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery/landing")


def sizes(M, cols):
    resp = M[R.RESP_BINS].mean(0) - M[R.BASE_BINS].mean(0)
    out = {}
    for h, suf in (("contra", "_right"), ("ipsi", "_left")):
        idx = [i for i, c in enumerate(cols) if c.endswith(suf) and c[: -len(suf)] in R.OROFACIAL]
        out[h] = float(resp[idx].mean())
    out["bal"] = float(np.log2(out["contra"] / out["ipsi"])) if out["contra"] > 0 and out["ipsi"] > 0 else np.nan
    return out


def main():
    L = pd.read_csv(R.LANDING)
    L = {lab: g for lab, g in L[L.land.notna()].groupby("label")}
    trials = []                                                        # (animal, epoch, label, target, land, Y row)
    trows = []
    data, cols = S.load_sessions()
    for an, sess in sorted(data.items()):
        for lab, ep, _day, Y, pos in sess:
            for p in S.POS:
                if (pos == p).sum() >= R.MIN_TRIALS:
                    trows.append({"animal": an, "epoch": ep, "label": lab, "target": p, "n": int((pos == p).sum()),
                                  **sizes(Y[pos == p].mean(0), cols)})
            g = L.get(lab)
            if g is None:
                continue
            for r, land in zip(g.row.to_numpy(), g.land.to_numpy()):
                trials.append((an, ep, lab, pos[r], land, Y[r]))
    T = pd.DataFrame([t[:5] for t in trials], columns=["animal", "epoch", "label", "target", "land"])
    Yall = np.stack([t[5] for t in trials])
    rows = []
    for keys, g in T.groupby(["animal", "epoch", "target", "land"]):
        if len(g) >= R.MIN_TRIALS:
            rows.append(dict(zip(["animal", "epoch", "target", "land"], keys), n=len(g),
                             n_sessions=g.label.nunique(), **sizes(Yall[g.index].mean(0), cols)))
    cells = pd.DataFrame(rows)
    srows = []
    for keys, g in T.groupby(["animal", "epoch", "label", "target", "land"]):
        if len(g) >= R.MIN_TRIALS:
            srows.append(dict(zip(["animal", "epoch", "label", "target", "land"], keys), n=len(g),
                              **sizes(Yall[g.index].mean(0), cols)))
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT / "balance_cells.csv", index=False)
    pd.DataFrame(srows).to_csv(OUT / "balance_cells_sessions.csv", index=False)
    pd.DataFrame(trows).to_csv(OUT / "balance_cells_target.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 400, "display.precision", 2):
        far = cells[cells.target.str.startswith("far") & cells.land.str.startswith("far")]
        print(far.pivot_table(index=["animal", "target", "land"], columns="epoch", values="bal")
              .reindex(columns=R.EPOCHS))
        print(far.pivot_table(index=["animal", "target", "land"], columns="epoch", values="n")
              .reindex(columns=R.EPOCHS))


if __name__ == "__main__":
    main()

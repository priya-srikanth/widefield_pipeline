"""BEHAVIOURAL RECOVERY MEASURE for ranking animals (Priya 10-27: "angle offset from intended spout as measure of recovery -
weighted for far contra then far mid, then all others").

Per cued lick: angle atan2(lateral, sagittal) (degrees, 3-D, extension peak) minus the animal's PRE mean angle at the
same target spout (its own pre landing, not the geometric spout direction: pre landings sit medial of the spout, and the
geometric angle carries the mouth-origin error). Per session: mean |offset| per spout, combined over spouts with weights
far contra 3, far mid 2, every other spout 1 (renormalised over the spouts present). Also the SIGNED far-contra offset
(- = ipsilesional). Recovery = the session's weighted |offset|; per animal x epoch means, and each animal's chronic level
relative to its acute level (fraction of the acute deficit remaining). Pre sessions are scored the same way (their
|offset| is the pre noise floor).
Outputs <E>/iti_licks/recovery_measure_{sessions,animals}.csv, figures/lick_cued_recovery_measure.png.
Usage: python scripts/iti_recovery_measure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EPS = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
W = {"far_contra": 3.0, "far_mid": 2.0}
MIN = 10


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from iti_check_figures import _days

    p = pd.read_csv(il.OUT / "zscore_check" / "where_preds_z.csv")
    p = p[(p.set == "trial") & p.pos.isin(POS)].dropna(subset=["act_lat", "act_sag"])
    p["ang"] = np.degrees(np.arctan2(p.act_lat, p.act_sag))
    ref = p[p.epoch == "pre"].groupby(["animal", "pos"]).ang.mean()
    p["off"] = p.ang - [ref.get((a, q), np.nan) for a, q in zip(p.animal, p.pos)]
    rows = []
    for (a, lab, ep), g in p.groupby(["animal", "label", "epoch"]):
        per = g.groupby("pos").off.agg(lambda x: np.abs(x).mean() if len(x) >= MIN else np.nan).dropna()
        sgn = g.groupby("pos").off.agg(lambda x: x.mean() if len(x) >= MIN else np.nan).dropna()
        if per.empty:
            continue
        w = np.array([W.get(q, 1.0) for q in per.index])
        ws = np.array([W.get(q, 1.0) for q in sgn.index])
        fc = g[g.pos == "far_contra"].off
        rows.append(dict(animal=a, label=lab, epoch=ep, weighted_abs=float((per.to_numpy() * w).sum() / w.sum()),
                         weighted_signed=float((sgn.to_numpy() * ws).sum() / ws.sum()),
                         far_contra_signed=float(fc.mean()) if len(fc) >= MIN else np.nan,
                         far_contra_abs=float(per.get("far_contra", np.nan)), n_spouts=len(per)))
    S = pd.DataFrame(rows).merge(_days(), on="label", how="left")
    S.to_csv(il.OUT / "recovery_measure_sessions.csv", index=False)
    A = (S.groupby(["animal", "epoch"])[["weighted_signed", "weighted_abs", "far_contra_signed", "far_contra_abs"]].mean()
         .unstack("epoch"))
    A = A.reindex(columns=EPS, level=1)
    w = A["weighted_signed"]           # PRIMARY: signed (|.| adds each animal's pre noise floor; PS93's is ~2x the others)
    A[("chronic_minus_pre_deg", "chronic")] = w["chronic"] - w["pre"]
    A[("deficit_remaining", "chronic")] = (w["chronic"] - w["pre"]) / (w["acute"] - w["pre"])
    A.to_csv(il.OUT / "recovery_measure_animals.csv")
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): weighted |angle offset from own pre landing| (deg), far-contra signed offset, and the "
          "fraction of the acute deficit remaining chronically")
    print(A.round(2).to_string())
    rank = A[("chronic_minus_pre_deg", "chronic")].sort_values(ascending=False)
    print("\n== recovery ranking by chronic weighted SIGNED offset vs pre (deg; closest to 0 first):",
          ", ".join(f"{a} {v:+.1f}" for a, v in rank.items()))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for k, (val, lab) in enumerate((("weighted_signed", "weighted SIGNED angle offset from own pre (deg; - = ipsi)"),
                                    ("far_contra_signed", "far-contra signed angle offset (deg; - = ipsilesional)"))):
        ax = axs[k]
        for j, a in enumerate(AN):
            g = S[S.animal == a].sort_values("day")
            ax.plot(g.day, g[val], "o-", ms=3, lw=1, color=plt.get_cmap("tab10")(j), label=a)
        ax.axvline(0, color="k", lw=0.7, ls="--")
        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        ax.set_xlabel("day after stroke")
        ax.set_ylabel(lab, fontsize=9)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].legend(fontsize=8, frameon=False)

    fig.suptitle("BEHAVIOUR: recovery measure -- cued lick angle offset from the animal's own pre landing at the target "
                 "spout, weighted far contra 3 : far mid 2 : others 1 (left), and far contra signed (right).\nOne dot per "
                 "session; pre sessions (left of the dashed line) = the noise floor. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(il.OUT / "figures" / "lick_cued_recovery_measure.png", dpi=100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""BEHAVIOUR-MATCHED RESIDUAL (Priya 10-25, analysis 4 of the synthesis): is the chronic residual (cortex responding more
than the frozen pre-stroke mapping predicts) a frozen-model EXTRAPOLATION artefact -- post trials whose movements lie
outside anything the pre-stroke model saw -- or present for movements inside the pre-stroke range?

Per trial (the frozen run's saved per-trial parts, `trial_parts`: data and full prediction averaged over each trial's
Musall window, cue 0..1.5 s / lick -0.1..+1.0 s around the first lick; per Allen area) the residual = data - full.
Behaviour per trial: first lick's 3-D endpoint (side_ml, out_ap, down_dv mm), reach_mm, speed_peak, its 3-D peak time
(latency proxy), and the trial's lick count (trial_attempts.csv). ENVELOPE, per animal x spout: the pre-stroke trials'
features standardised by the pre mean / SD; a trial is INSIDE when its Mahalanobis distance to the pre trials (pooled pre
covariance, ridge 1e-3) is <= the 95th percentile of the pre trials' own leave-one-session-out distances. Residual
change vs pre (within animal, nested bootstrap) for ALL attempted trials vs INSIDE-envelope trials only; per-animal
table first. If the residual survives on inside-envelope trials, extrapolation does not explain it.
Trials without 3-D (PS94 / PS95 cam4-only sessions) are kept out of the matched set, never imputed.

Usage: python scripts/matched_residual.py [--tag gate-attempted_v7g3_t3d_frA]
Output: <E>/matched_residual/<tag>/{trials.csv, boot.csv, figures/matched_residual.png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402

KIN = Path("C:/Users/SabatiniLab/wfield_cache/pose3d_inputs/pose3d_kin_licks_snapshot_20261023.csv.gz")
FEATS = ["side_ml", "out_ap", "down_dv", "reach_mm", "speed_peak", "t_peak_3d", "n_licks"]
AREAS = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left", "MOs_right", "MOs_left"]
POSMAP = {"close_L": "close_ipsi", "close_R": "close_contra", "close_C": "close_mid", "far_L": "far_ipsi",
          "far_R": "far_contra", "far_C": "far_mid"}


def trial_residuals(run: Path) -> pd.DataFrame:
    rows = []
    for js in sorted((run / "joint" / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if m["label"][-4:] < "0801":
            continue
        z = np.load(js.with_suffix(".npz"))
        if "trial_parts" not in z.files:
            continue
        tp, parts, wins, areas = z["trial_parts"], list(z["parts"]), list(z["trial_windows"]), list(z["areas"])
        res = (tp[parts.index("data")] - tp[parts.index("full")]) * 1e3         # (trials, windows, areas)
        for ar in AREAS:
            if ar not in areas:
                continue
            ai = areas.index(ar)
            for wi, w in enumerate(wins):
                rows.append(pd.DataFrame({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"],
                                          "trial_id": z["trial_id"], "pos": z["trial_pos"], "window": w, "area": ar,
                                          "residual": res[:, wi, ai]}))
    return pd.concat(rows, ignore_index=True)


def trial_behaviour() -> pd.DataFrame:
    K = pd.read_csv(KIN)
    K = K[K.lick_idx == 0].copy()
    K["label"] = K.animal + "_" + K.date.astype(str).str[-4:]
    A = pd.read_csv(E / "qc" / "trial_attempts.csv", usecols=["label", "trial_id", "n_licks", "attempted"])
    return K[["label", "trial_id", *FEATS[:-1]]].merge(A, on=["label", "trial_id"], how="inner")


def envelope(B: pd.DataFrame) -> pd.DataFrame:
    """+ maha (distance to the animal x spout pre trials) and inside (<= pre 95th pct, pre scored leave-one-session-out)."""
    out = []
    for (_an, _pos), g in B.groupby(["animal", "pos"]):
        g = g.dropna(subset=FEATS).copy()
        pre = g[g.epoch == "pre"]
        if len(pre) < 30:
            continue
        mu, sd = pre[FEATS].mean(), pre[FEATS].std().replace(0, 1)

        def dist(X, ref, mu=mu, sd=sd):
            Z, R = (X - mu) / sd, (ref - mu) / sd
            C = np.cov(R.to_numpy().T) + 1e-3 * np.eye(len(FEATS))
            D = Z.to_numpy() - R.to_numpy().mean(0)
            return np.sqrt(np.einsum("ij,jk,ik->i", D, np.linalg.inv(C), D))
        g["maha"] = dist(g[FEATS], pre[FEATS])
        loso = np.concatenate([dist(pre[pre.label == lab][FEATS], pre[pre.label != lab][FEATS])
                               for lab in pre.label.unique()])
        thr = np.percentile(loso, 95)
        g["inside"] = g.maha <= thr
        g.loc[g.epoch == "pre", "inside"] = True             # the pre set is the reference as a whole
        out.append(g.assign(thr=thr))
    return pd.concat(out, ignore_index=True)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    out = E / "matched_residual" / a.tag
    (out / "figures").mkdir(parents=True, exist_ok=True)
    R = trial_residuals(run)
    B = trial_behaviour()
    R["pos"] = R.pos.astype(str)
    B2 = R[["label", "animal", "epoch", "trial_id", "pos"]].drop_duplicates().merge(B, on=["label", "trial_id"])
    Env = envelope(B2)
    Env.to_csv(out / "trials.csv", index=False)
    pd.set_option("display.width", 220)
    share = Env.groupby(["animal", "epoch"]).inside.mean().unstack("epoch").reindex(columns=ffc.EP4)
    print("== share of 3-D-tracked trials INSIDE the pre-stroke behavioural envelope (per animal)\n" + share.round(2).to_string())
    fc = Env[Env.pos.str.contains("far_contra|far_R")].groupby(["animal", "epoch"]).inside.mean().unstack("epoch")
    print("== far contra only\n" + fc.reindex(columns=ffc.EP4).round(2).to_string())
    M = R.merge(Env[["label", "trial_id", "inside", "maha"]], on=["label", "trial_id"], how="left")
    S = []
    for sel, mask in (("all trials", np.ones(len(M), bool)), ("3-D tracked", M.inside.notna().to_numpy()),
                      ("inside envelope", (M.inside == True).to_numpy())):          # noqa: E712
        S.append(M[mask].groupby(["label", "animal", "epoch", "window", "area"]).residual.mean().reset_index()
                 .assign(sel=sel))
    S = pd.concat(S, ignore_index=True)
    per = S[S.area.isin(["SSp-m_right", "SSp-m_left"])].groupby(["sel", "window", "area", "animal", "epoch"]).residual \
        .mean().unstack("epoch").reindex(columns=ffc.EP4)
    print("== per animal (rule 8): residual (x1e3 dF/F, window mean), by trial selection\n" + per.round(2).to_string())
    Bt = pd.concat([ffc.boot(S[S.sel == s], "residual").assign(frame=s) for s in S.sel.unique()], ignore_index=True)
    Bt.to_csv(out / "boot.csv", index=False)
    print(Bt[Bt.area.isin(["cortex", "SSp-m_right", "SSp-m_left"]) & (Bt.epoch != "pre_split")]
          [["frame", "window", "area", "epoch", "point", "lo", "hi", "n_up", "n"]].round(2).to_string(index=False))
    sels = ["all trials", "3-D tracked", "inside envelope"]
    ffc.COL.update(dict(zip(sels, ["#999999", "#66a61e", "#d95f02"])))
    fig, axs = plt.subplots(2, 1, figsize=(15, 8.5))
    for ax, w in zip(axs, ffc.WINDOWS):
        ffc.bars(ax, Bt[Bt.window == w], sels, AREAS, "residual change from pre (x1e3 dF/F)",
                 f"{w} window: residual on all attempted trials vs trials whose movement lies INSIDE the pre-stroke range")
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"NEURAL: is the chronic residual a frozen-model extrapolation? ({a.tag}). Envelope = per animal x spout, "
                 "Mahalanobis distance on first-lick 3-D endpoint, reach, peak speed, latency and lick count <= the\npre "
                 "trials' leave-one-session-out 95th percentile. Bars = mean of animals; whisker 95 % nested CI; * excludes "
                 "0; dots = animals; grey band = pre split-half range. contra = right.", fontsize=9)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.88, bottom=0.07, hspace=0.4)
    fig.savefig(out / "figures" / "matched_residual.png", dpi=105)
    plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

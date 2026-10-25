"""NEURAL vs LICK softmax confusion matrices (Priya 10-24: "are you making the softmax neural confusion matrix? maybe bins
0.5 and 1").

NEURAL softmax = the stored frozen pre-stroke spout decoder's probabilities over the six spouts per trial (`attempt_coding`
pp_<spout>, LOSO for pre sessions). LICK softmax = each lick's weights over the six executed bins (3-D landing, the
encoder's rule: w_p ~ exp(-0.5 d_p^2 / w^2), d in within-spout SD units to each spout's pre-stroke landing centre,
`exec_bins_calib_20261023.csv`), averaged over the trial's licks; widths 0.5 and 1. Trials with licks only: pre-stroke
hits, post-stroke hits + failed attempts (incomplete).

  1. matrices per animal x epoch: rows = target spout, columns = neural (decoded-as) / lick (landed-as) spout
  2. epoch minus pre within animal, mean of animals, nested bootstrap (animals -> sessions), Bonferroni over 36 cells
  3. trial level, far-contra trials post-stroke: within session, does a trial's neural far-mid / far-ipsi probability
     follow its lick weight on those bins? (per-animal slope; and neural probabilities split by the dominant landing bin)

Usage: python scripts/softmax_neural_vs_lick.py [--window lick]
Output: <E>/attempt_coding_pp/softmax_neural_vs_lick_*.csv, figures/softmax_neural_vs_lick_*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
I = Path("C:/Users/SabatiniLab/wfield_cache/pose3d_inputs")
SP = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
LAB = ["cI", "cM", "cC", "fI", "fM", "fC"]
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
MOUSE = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra",
         "far_L": "far_ipsi", "far_center": "far_mid", "far_R": "far_contra"}


def lick_weights(width: float) -> pd.DataFrame:
    """Per trial (label, trial_id): mean soft weight of its licks on each executed bin."""
    K = pd.read_csv(I / "pose3d_kin_licks_snapshot_20261023.csv.gz",
                    usecols=["animal", "session", "trial_id", "side_ml", "out_ap", "down_dv"])
    K = K.dropna(subset=["side_ml", "out_ap", "down_dv"])
    C = pd.read_csv(I / "exec_bins_calib_20261023.csv")
    C = C[C.calib_for == "default"]
    out = []
    for an, k in K.groupby("animal"):
        c = C[C.animal == an].set_index("position").reindex(SP)
        G = c[["ml", "ap", "down"]].to_numpy(float)
        sc = c[["sd_ml", "sd_ap", "sd_down"]].iloc[0].to_numpy(float)
        X = k[["side_ml", "out_ap", "down_dv"]].to_numpy(float)
        d = (((X / sc)[:, None, :] - (G / sc)[None]) ** 2).sum(2)
        e = np.exp(-0.5 * (d - d.min(1, keepdims=True)) / width ** 2)
        Wt = e / e.sum(1, keepdims=True)
        df = pd.DataFrame(Wt, columns=[f"lw_{p}" for p in SP])
        df["label"] = k.session.str[:4].to_numpy() + "_" + k.session.str[-4:].to_numpy()
        df["trial_id"] = k.trial_id.to_numpy()
        out.append(df)
    return pd.concat(out).groupby(["label", "trial_id"]).mean().reset_index()


def path_weights(width: float):
    """Per trial (label, trial_id): mean TRAJECTORY softmax weight of its licks (`scripts/lick_path_softmax.py`), or None."""
    from wfield_local.paths import PathResolver
    f = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "trajectories" / "path_softmax_licks.csv.gz"
    if not f.exists():
        return None
    P = pd.read_csv(f)
    P = P[P.width == width]
    P["label"] = P.session.str[:4] + "_" + P.session.str[-4:]
    P = P.rename(columns={q: f"tw_{q}" for q in SP})
    return P.groupby(["label", "trial_id"])[[f"tw_{q}" for q in SP]].mean().reset_index()


def session_matrices(T: pd.DataFrame, prefix: str) -> dict:
    """(animal, epoch) -> list of per-session 6 x 6 matrices (row = target spout, mean of the trials' <prefix> columns)."""
    out = {}
    cols = [f"{prefix}{p}" for p in SP]
    for (an, ep, _lab), g in T.groupby(["animal", "epoch", "label"]):
        M = np.full((6, 6), np.nan)
        for r, t in enumerate(SP):
            x = g[g.spout == t]
            if len(x) >= 3:
                M[r] = x[cols].mean().to_numpy(float)
        out.setdefault((an, ep), []).append(M)
    return out


def boot_change(S: dict, ep: str, nb=4000, seed=0):
    rng = np.random.default_rng(seed)
    ans = [a for a in AN if (a, ep) in S and (a, "pre") in S]
    obs = np.nanmean([np.nanmean(S[(a, ep)], 0) - np.nanmean(S[(a, "pre")], 0) for a in ans], 0)
    B = np.empty((nb, 6, 6))
    for b in range(nb):
        d = []
        for a in rng.choice(ans, len(ans), replace=True):
            e_, p_ = S[(a, ep)], S[(a, "pre")]
            d.append(np.nanmean([e_[i] for i in rng.integers(0, len(e_), len(e_))], 0)
                     - np.nanmean([p_[i] for i in rng.integers(0, len(p_), len(p_))], 0))
        B[b] = np.nanmean(d, 0)
    al = 0.05 / 36
    lo, hi = np.nanquantile(B, al / 2, 0), np.nanquantile(B, 1 - al / 2, 0)
    agree = np.sum([np.sign(np.nanmean(S[(a, ep)], 0) - np.nanmean(S[(a, "pre")], 0)) == np.sign(obs) for a in ans], 0)
    return obs, (lo > 0) | (hi < 0), agree, len(ans)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default="lick", choices=("lick", "cue"))
    a = ap.parse_args(argv)
    D = E / "attempt_coding_pp"
    fd = D / "figures"
    fd.mkdir(exist_ok=True)
    T = pd.concat([pd.read_csv(D / f"trials_{a.window}_{an}.csv.gz") for an in AN])
    T = T[(T.method == "dom") & T.cls.isin(["pre_hit", "hit", "incomplete"])]
    rows = []
    for w in (0.5, 1.0):
        L = lick_weights(w)
        J = T.merge(L, on=["label", "trial_id"], how="inner")
        PW = path_weights(w)
        if PW is not None:                                  # same trials in every row: those with a scored path too
            J = J.merge(PW, on=["label", "trial_id"], how="inner")
        print(f"width {w}: {len(J)} of {len(T)} trials have 3-D licks{' and paths' if PW is not None else ''}", flush=True)
        panels = [(session_matrices(J, "pp_"), "NEURAL: frozen decoder softmax (decoded as)", "neural"),
                  (session_matrices(J, "lw_"), f"LICK: landing softmax, width {w:g} (landed as)", "lick")]
        if PW is not None:
            panels.append((session_matrices(J, "tw_"), f"LICK: trajectory softmax, width {w:g} (path follows)", "path"))
        fig, axs = plt.subplots(len(panels), 3, figsize=(13, 4.3 * len(panels)))
        for k, ep in enumerate(("acute", "subacute", "chronic")):
            for r, (S, name, kind) in enumerate(panels):
                obs, sig, agree, n = boot_change(S, ep)
                ax = axs[r, k]
                ax.imshow(obs, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
                for i in range(6):
                    for j in range(6):
                        if sig[i, j]:
                            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=2))
                        if abs(obs[i, j]) >= 0.06 or sig[i, j]:
                            ax.text(j, i, f"{obs[i, j]:+.2f}\n{agree[i, j]}/{n}", ha="center", va="center",
                                    fontsize=6.3, color="w" if abs(obs[i, j]) > 0.35 else "k")
                        rows.append({"window": a.window, "width": w, "kind": kind,
                                     "epoch": ep, "target": SP[i], "as": SP[j], "change": obs[i, j],
                                     "significant": bool(sig[i, j]), "same_sign": int(agree[i, j])})
                ax.set_xticks(range(6)); ax.set_xticklabels(LAB); ax.set_yticks(range(6)); ax.set_yticklabels(LAB)
                ax.set_title(f"{ep} minus pre", fontsize=10)
                if k == 0:
                    ax.set_ylabel(f"{name}\ntarget spout", fontsize=8)
        fig.suptitle(f"Same trials, {a.window} window: NEURAL (top) vs LICK landing / trajectory softmax, epoch minus pre within animal, "
                     f"mean of 4 animals. Boxed = nested bootstrap excludes 0 (Bonferroni over 36); second line = animals "
                     f"with the same sign.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(fd / f"softmax_neural_vs_lick_{a.window}_w{w:g}.png", dpi=130)
        plt.close(fig)
        # trial level, far-contra trials post-stroke: within-session slope of neural p on lick weight, per animal
        F = J[(J.spout == "far_contra") & (J.epoch != "pre")].copy()
        for an, g in F.groupby("animal"):
            for q in ("far_mid", "far_ipsi", "far_contra"):
                x = g[f"lw_{q}"] - g.groupby("label")[f"lw_{q}"].transform("mean")
                y = g[f"pp_{q}"] - g.groupby("label")[f"pp_{q}"].transform("mean")
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 20 or x[ok].std() == 0:
                    continue
                b = np.polyfit(x[ok], y[ok], 1)[0]
                rr = np.corrcoef(x[ok], y[ok])[0, 1]
                rows.append({"window": a.window, "width": w, "kind": "trial_slope", "epoch": "post", "target": "far_contra",
                             "as": q, "animal": an, "slope": b, "r": rr, "n": int(ok.sum())})
            dom = g[[f"lw_{p}" for p in SP]].to_numpy().argmax(1)
            for di in np.unique(dom):
                h = g[dom == di]
                if len(h) >= 10:
                    rows.append({"window": a.window, "width": w, "kind": "by_landing", "epoch": "post",
                                 "target": "far_contra", "as": SP[di], "animal": an, "n": len(h),
                                 **{f"neural_{p}": float(h[f"pp_{p}"].mean()) for p in ("far_ipsi", "far_mid", "far_contra")}})
    R = pd.DataFrame(rows)
    R.to_csv(D / f"softmax_neural_vs_lick_{a.window}.csv", index=False)
    pd.set_option("display.width", 220)
    print(R[R.kind == "trial_slope"][["width", "animal", "as", "slope", "r", "n"]].round(3).to_string(index=False))
    print(R[R.kind == "by_landing"][["width", "animal", "as", "n", "neural_far_ipsi", "neural_far_mid",
                                     "neural_far_contra"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""BEHAVIOURAL AIM DEFICIT per session, FIRST cued lick vs all licks, weighted vs far contra (Priya 10-27: switch the
behaviour axis / mixed-model deficit to the first-lick weighted aim; "can I see a comparison between the weighted deficit
and the first-lick far-contra deficit vs time across animals").

Per cued lick: angle atan2(lateral, sagittal) (deg; 3-D landing at the extension peak) minus the animal's PRE mean angle
at the same target spout (as `iti_recovery_measure.py`). First lick = lick_idx 0 (first post-cue lick of the trial;
joined via first_lick_preds.csv.gz). Per session:
  weighted_first / weighted_all   signed offset per spout, combined with weights far contra 3, far mid 2, others 1
  farcontra_first / farcontra_all signed offset at far contra
(- = aim shifted ipsilesional). Split-half reliability (random half of the session's licks, Spearman-Brown).
The mixed models' behaviour deficit = pre - session of weighted_first (deg; `iti_lmm_figures.py --deficit aim`).
Outputs <E>/iti_licks/aim_deficit_sessions.csv, figures/lick_cued_aim_deficit_compare.png
(figure data figdata/lick_cued_aim_deficit_compare.pkl; --plot-only redraws from it).
Usage: python scripts/lick_aim_deficit.py [--plot-only]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
W = {"far_contra": 3.0, "far_mid": 2.0}
FD = il.OUT / "figdata" / "lick_cued_aim_deficit_compare.pkl"


def _weighted(g: pd.DataFrame, min_n: int) -> float:
    m = g.groupby("pos").off.agg(["mean", "size"])
    m = m[m["size"] >= min_n]
    if m.empty:
        return np.nan
    w = np.array([W.get(s, 1.0) for s in m.index])
    return float((m["mean"] * w).sum() / w.sum())


def _fc(g: pd.DataFrame, min_n: int) -> float:
    h = g[g.pos == "far_contra"].off
    return float(h.mean()) if len(h) >= min_n else np.nan


def compute() -> pd.DataFrame:
    from wfield_local import epochs
    p = pd.read_csv(il.OUT / "zscore_check" / "where_preds_z.csv")
    p = p[(p.set == "trial") & p.pos.isin(POS)].dropna(subset=["act_lat", "act_sag"]).copy()
    p["ang"] = np.degrees(np.arctan2(p.act_lat, p.act_sag))
    F = pd.read_csv(il.OUT / "first_lick_preds.csv.gz")[["label", "act", "lick_idx"]].drop_duplicates(["label", "act"])
    p["k"], F["k"] = p.act_lat.round(6), F.act.round(6)
    p = p.merge(F[["label", "k", "lick_idx"]], on=["label", "k"], how="left")
    p["animal"] = p.label.str.split("_").str[0]
    pre = p[p.epoch == "pre"].groupby(["animal", "pos"]).ang.mean()
    p["off"] = p.ang - pre.reindex(pd.MultiIndex.from_arrays([p.animal, p.pos])).to_numpy()
    rng = np.random.default_rng(0)
    rows = []
    for (lab, ep, a), g in p.groupby(["label", "epoch", "animal"]):
        f = g[g.lick_idx == 0]
        hf = rng.random(len(f)) < 0.5
        rows.append(dict(label=lab, epoch=ep, animal=a, day=float(epochs.days_since_stroke(lab)),
                         weighted_all=_weighted(g, 5), weighted_first=_weighted(f, 5),
                         farcontra_all=_fc(g, 5), farcontra_first=_fc(f, 5),
                         wf_h1=_weighted(f[hf], 3), wf_h2=_weighted(f[~hf], 3), fcf_h1=_fc(f[hf], 3), fcf_h2=_fc(f[~hf], 3),
                         n_first=len(f), n_all=len(g)))
    S = pd.DataFrame(rows).sort_values(["animal", "day"])
    S.to_csv(il.OUT / "aim_deficit_sessions.csv", index=False)
    return S


def reliability(S: pd.DataFrame) -> dict:
    out = {}
    for nm, (a, b) in {"weighted_first": ("wf_h1", "wf_h2"), "farcontra_first": ("fcf_h1", "fcf_h2")}.items():
        d = S[[a, b]].dropna()
        r = float(np.corrcoef(d[a], d[b])[0, 1])
        out[nm] = (r, 2 * r / (1 + r), len(d))
    return out


def plot(S: pd.DataFrame, rel: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER
    from wfield_local.mixed_models import ANIMAL_COL
    series = [("weighted_first", "weighted aim, FIRST lick", "-", 2.0), ("farcontra_first", "far-contra aim, FIRST lick",
              "--", 1.6), ("weighted_all", "weighted aim, all licks", ":", 1.2)]
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.6), squeeze=False, sharey=True)
    for j, a in enumerate(AN):
        ax = axs[0, j]
        g = S[S.animal == a]
        for col, _lab, ls, lw in series:
            ax.plot(g.day, g[col], ls, color=ANIMAL_COL[a], lw=lw, marker=ANIMAL_MARKER[a], ms=3)
        ax.set_title(a, fontsize=10)
    ax = axs[0, 4]
    for col, _lab, ls, lw in series:
        A = S.assign(dbin=np.where(S.day < 0, -5, np.round(S.day / 3) * 3)).groupby("dbin")[col]
        mu, se = A.mean(), A.sem()
        ax.errorbar(mu.index, mu, yerr=se, fmt="o" + ls, color="k", lw=lw, ms=3, capsize=2)
    ax.set_title("all animals (3-day bins; pre = left), mean +/- SEM", fontsize=9)
    for ax in axs[0]:
        ax.axhline(0, color="0.6", lw=0.7)
        ax.axvline(0, color="0.6", lw=0.7, ls=":")
        ax.set_xlabel("days since stroke", fontsize=8)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0, 0].set_ylabel("aim vs own pre landing (deg, - = ipsilesional)", fontsize=8)
    axs[0, 0].legend(handles=[Line2D([], [], color="k", ls=ls, lw=lw, label=lab) for _c, lab, ls, lw in series],
                     fontsize=7.5, frameon=False, loc="lower right")
    txt = "; ".join(f"{k.replace('_', ' ')}: split-half r {v[0]:.2f}, reliability {v[1]:.2f} ({v[2]} sessions)"
                    for k, v in rel.items())
    fig.suptitle("BEHAVIOUR: cued-lick aim deficit per session vs days since stroke -- weighted (far contra x3, far mid x2, "
                 "others x1) vs far contra only, FIRST post-cue lick vs all licks.\n" + txt + ". The mixed models' "
                 "behaviour deficit = pre - session of the FIRST-lick weighted aim. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(il.OUT / "figures" / "lick_cued_aim_deficit_compare.png", dpi=100)
    plt.close(fig)


def main() -> int:
    if "--plot-only" in sys.argv and FD.exists():
        S, rel = pickle.loads(FD.read_bytes())
    else:
        S = compute()
        rel = reliability(S)
        FD.parent.mkdir(exist_ok=True)
        FD.write_bytes(pickle.dumps((S, rel)))
    print(rel)
    pd.set_option("display.width", 200)
    print(S.groupby(["animal", "epoch"])[["weighted_first", "farcontra_first", "weighted_all", "farcontra_all"]].mean()
          .round(1).unstack("epoch").to_string())
    plot(S, rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

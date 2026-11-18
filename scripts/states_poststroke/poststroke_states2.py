"""Post-stroke AR-HMM readouts, round 2 (Priya OK 11-17 items 1, 2, 3, 6). Scratch; locanmf env.

Trial cycle = previous trial's stop .. this trial's stop (ITI + pre-cue + response). ENGAGED trials = GLM-HMM K5
(design v2) argmax in the three engaged states {0, 1, 2} (state 3 = partially engaged, 4 = disengaged).
 1. syllable-group share of frames on ENGAGED trial cycles, per session; delta vs own August pre (boot_delta).
 2. lick direction: ipsi / central / contra as a share of LICK frames (all frames and engaged cycles).
 3. cue-aligned group probability (-1 .. +3 s) on engaged trials, per epoch (animal-weighted) -> figure.
 6. transitions between groups (runs collapsed, engaged cycles): P(next | current) per session, delta vs own pre;
    plus transition entropy.
Per-animal tables printed before every bootstrap (CLAUDE.md rule 8). Holm within each readout family.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import analysis_kit as ak

S = Path(sys.argv[1])
OUT = S / "poststroke_states"
Zd = S / "arhmm_local_fit" / "states_pooled_all_s1"
FEAT = S / "arhmm_feat_v3" / "sessions"
EP = json.load(open("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/arhmm_20261117/inputs/epochs.json"))
POST = ["acute", "subacute", "chronic"]
EPOCHS = ["pre", *POST]
GROUPS = {"quiet": [5, 8, 19], "nose_slow": [1], "lick_ipsi": [0, 7, 9, 23], "lick_central": [10, 12, 18],
          "lick_contra": [2, 15, 20], "jaw_chew": [21, 22], "push": [11, 13], "walk_slow": [4], "walk": [6]}
GN = list(GROUPS)
GI = np.full(32, -1)
for i, ks in enumerate(GROUPS.values()):
    GI[ks] = i
LICK = [GN.index(g) for g in ("lick_ipsi", "lick_central", "lick_contra")]
ENGAGED = {0, 1, 2}
T0, T1, FS = -1.0, 3.0, 25.0
NB = int((T1 - T0) * FS)


def epoch(lab):
    e = EP.get(lab)
    return None if e is None or (e == "pre" and lab[-4:] < "0801") else e


def one(item):
    lab, eng_ids = item
    from wfield_local import trial_classes as tc
    T = tc.session_trials(lab).sort_values("trial_id").reset_index(drop=True)
    z = np.load(Zd / f"{lab}.npz")["z"].astype(int)
    t = np.load(FEAT / f"{lab}.npz", allow_pickle=True)["t_daq_s"]
    g = np.where(z >= 0, GI[np.maximum(z, 0)], -1)
    stop = T.cue_daq.to_numpy() + np.nan_to_num(T.stop_ms.to_numpy(float) / 1000, nan=3.5)
    eng_mask = np.zeros(len(t), bool)
    eth, n_eth = np.zeros((len(GN), NB)), 0
    for i, r in T.iterrows():
        if r.trial_id not in eng_ids:
            continue
        a = stop[i - 1] if i > 0 else r.cue_daq - 5.0
        eng_mask[np.searchsorted(t, a):np.searchsorted(t, stop[i])] = True
        j0 = np.searchsorted(t, r.cue_daq + T0)
        seg = g[j0:j0 + NB]
        if len(seg) == NB and (seg >= 0).mean() > 0.9:
            eth[seg[seg >= 0], np.flatnonzero(seg >= 0)] += 1
            n_eth += 1
    out = {"label": lab}
    for name, m in (("all", g >= 0), ("eng", eng_mask & (g >= 0))):
        gg = g[m]
        for k, gname in enumerate(GN):
            out[f"{name}:{gname}"] = float(np.mean(gg == k)) if len(gg) else np.nan
        lk = np.isin(gg, LICK)
        for k in LICK:
            out[f"{name}:dir_{GN[k]}"] = float(np.mean(gg[lk] == k)) if lk.sum() else np.nan
    # transitions on engaged cycles: collapse runs; a gap (unassigned or outside engaged) breaks the sequence
    C = np.zeros((len(GN), len(GN)))
    gm = np.where(eng_mask, g, -1)
    br = np.flatnonzero(np.diff(gm) != 0) + 1
    runs = np.split(gm, br)
    labs_ = [r_[0] for r_ in runs]
    for x, y in zip(labs_[:-1], labs_[1:]):
        if x >= 0 and y >= 0:
            C[x, y] += 1
    return out, eth, n_eth, C


def boot_family(T: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    T = T.copy()
    T["animal"], T["epoch"] = T.label.str[:4], T.label.map(epoch)
    rows = []
    for c in cols:
        m = T.groupby(["animal", "epoch"])[c].mean().unstack().reindex(columns=EPOCHS)
        d = m[POST].sub(m["pre"], axis=0)
        print(f"\n--- {name} | {c}: pre and delta vs own pre, per animal")
        print(pd.concat([m[["pre"]], d.add_prefix("d_")], axis=1).round(3).to_string())
        pre = {a: g[c].dropna().to_numpy() for a, g in T[T.epoch == "pre"].groupby("animal")}
        for e in POST:
            post = {a: g[c].dropna().to_numpy() for a, g in T[T.epoch == e].groupby("animal")}
            iv, dr = ak.boot_delta(post, pre, np.random.default_rng(0), draws=True)
            rows.append({"family": name, "quantity": c, "epoch": e, "pre_mean": m["pre"].mean(), "delta": iv.point,
                         "lo": iv.lo, "hi": iv.hi, "p_boot": ak.boot_p(dr),
                         "animals_same_sign": int((np.sign(d[e]) == np.sign(iv.point)).sum())})
    R = pd.DataFrame(rows)
    R["p_holm"] = ak.holm(R.p_boot.to_numpy())
    R["ci_excl_0"] = (R.lo > 0) | (R.hi < 0)
    return R


def figure(E: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE as col        # the repo's epoch palette
    tt = T0 + (np.arange(NB) + 0.5) / FS
    fig, axs = plt.subplots(3, 3, figsize=(11, 8), sharex=True)
    for k, ax in enumerate(axs.ravel()):
        for e in EPOCHS:
            if e in E:
                A = E[e][:, k]                                              # (animals, time)
                m = A.mean(0)
                se = A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.zeros_like(m)
                ax.fill_between(tt, m - se, m + se, color=col[e], alpha=0.08, lw=0, zorder=2)  # = lick_matched_readout smoothed curves
                ax.plot(tt, m, color=col[e], lw=1.6, label=f"{e} (n={len(A)})", zorder=4)
        ax.axvline(0, color="0.6", lw=0.8, ls="--")
        ax.set_title(GN[k].replace("_", " "), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        if k % 3 == 0:
            ax.set_ylabel("P(state group)")
        if k >= 6:
            ax.set_xlabel("time from cue (s)")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig. 1. AR-HMM syllable groups around the cue, engaged trials (GLM-HMM K5 states 0-2)" + chr(10) + "mean over "
                 "animals (each = mean of its sessions) +- SEM across animals; pooled_all seed 1", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    d = pd.read_csv(S / "glmhmm_v1" / "states_pooled_all_K5_v2.csv.gz")
    pc = [f"p_state{k}" for k in range(5)]
    d["arg"] = d[pc].to_numpy().argmax(1)
    eng = {lab: set(g.trial_id[g.arg.isin(ENGAGED)]) for lab, g in d.groupby("label")}
    labs = [Path(f).stem for f in sorted(glob.glob(str(Zd / "*.npz"))) if epoch(Path(f).stem) and Path(f).stem in eng]
    res, fail = ak.fan_sessions([(lab, eng[lab]) for lab in labs], one, key=lambda it: it[0])
    print("failed:", fail)
    res = [(it[0], r) for it, r in res if r is not None]
    U = pd.DataFrame([r[0] for _l, r in res])
    U.to_csv(OUT / "round2_usage_direction_per_session.csv", index=False)
    pd.set_option("display.width", 220)
    # 3. ethogram: per session mean, then animal mean, then epoch mean
    per = {lab: r[1] / r[2] for lab, r in res if r[2] >= 10}
    E = {}
    for e in EPOCHS:
        an = {}
        for lab, m in per.items():
            if epoch(lab) == e:
                an.setdefault(lab[:4], []).append(m)
        if an:
            E[e] = np.stack([np.mean(an[a], 0) for a in sorted(an)])      # (animals, groups, time)
    np.savez(OUT / "cue_ethogram_engaged.npz", groups=GN, t=T0 + (np.arange(NB) + 0.5) / FS,
             **{e: v for e, v in E.items()})
    figure(E, OUT / "fig1_cue_ethogram_engaged.png")
    if "--fig-only" in sys.argv:
        raise SystemExit(0)
    # 1 + 2
    R1 = boot_family(U, [f"eng:{g}" for g in GN], "1 usage, engaged cycles")
    R2 = boot_family(U, [f"{w}:dir_{GN[k]}" for w in ("eng", "all") for k in LICK], "2 lick direction share")
    # 6. transitions
    rows = []
    for lab, r in res:
        C = r[3]
        P = C / np.maximum(C.sum(1, keepdims=True), 1)
        row = {"label": lab}
        for i in range(len(GN)):
            for j in range(len(GN)):
                if i != j:
                    row[f"{GN[i]}>{GN[j]}"] = P[i, j] if C[i].sum() >= 20 else np.nan
        p = C.sum(1) / max(C.sum(), 1)
        Hr = np.array([-(q[q > 0] * np.log2(q[q > 0])).sum() for q in P])
        row["entropy_bits"] = float((p * Hr).sum())
        rows.append(row)
    Tr = pd.DataFrame(rows)
    Tr.to_csv(OUT / "round2_transitions_per_session.csv", index=False)
    keep = [c for c in Tr if c != "label" and Tr[c].mean() >= 0.05] + ["entropy_bits"]
    keep = list(dict.fromkeys(keep))
    R6 = boot_family(Tr, keep, "6 transitions (engaged)")
    R = pd.concat([R1, R2, R6])
    R.to_csv(OUT / "round2_delta_vs_own_pre.csv", index=False)
    print("\n######## SUMMARY")
    print(R.round(3).to_string(index=False))

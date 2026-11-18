"""Peri-cue PUSH from the treadmill alone (Priya 11-17: "frequency of peri-cue push events looking at backward treadmill
events alone"). Model-free: the DAQ wheel's backward speed (`run_back`, mm/s, 25 Hz, the AR-HMM feature file's copy of
the canonical DAQ wheel) per trial.
  push      = max backward speed in the CUE window [-0.25, +0.75) s > THR
  control   = the same test in [-2, -1) s (spout already in place; ~1 s before the cue) -> cue-triggered vs a general
              rise in backward wheel motion
  peak, latency (first bin above THR, from the cue) on push trials
All trials and ENGAGED trials (GLM-HMM K5 argmax in {0,1,2}); delta vs own August pre, boot_delta (paired animals ->
sessions), per-animal table first. Figure: cue-aligned mean backward speed, engaged trials, +-SEM across animals.
Scratch; locanmf env.
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
FEAT = S / "arhmm_feat_v3" / "sessions"
EP = json.load(open("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/arhmm_20261117/inputs/epochs.json"))
POST = ["acute", "subacute", "chronic"]
EPOCHS = ["pre", *POST]
THR = 0.5
WALK = 3.0                       # mm/s forward = running (the repo's running threshold, behavior_events)
PANELS = [("fwd", "forward wheel speed (mm/s)"), ("back", "backward velocity (mm/s)"),
          ("p_walk", f"P(walking: forward > {WALK:g} mm/s)"), ("p_push", f"P(push: backward > {THR:g} mm/s)")]
CUE_W, CTL_W = (-0.25, 0.75), (-2.0, -1.0)
T0, T1, FS = -2.0, 4.5, 25.0
NB = int((T1 - T0) * FS)


def epoch(lab):
    e = EP.get(lab)
    return None if e is None or (e == "pre" and lab[-4:] < "0801") else e


def one(item):
    lab, eng = item
    from wfield_local import trial_classes as tc
    T = tc.session_trials(lab)
    F = np.load(FEAT / f"{lab}.npz", allow_pickle=True)
    cols = [str(c) for c in F["features"]]
    t, back, fwd = F["t_daq_s"], F["X"][:, cols.index("run_back")], F["X"][:, cols.index("run_fwd")]
    # the earlier (non-GLM-HMM) gate: the camera WORKING PERIOD, first .. last camera lick of the session
    # (`trial_classes.camera_engaged_period`) -- a period, not a per-trial "did it lick", so the gate does not select
    # on the response the push may belong to (Priya 11-17)
    period = tc.camera_engaged_period(lab, T.cue_daq.to_numpy(float))
    rows = []
    KEYS = ("lick_in", "lick_out", "glm_in", "glm_out")
    trace, ntr = {k: np.zeros((4, NB)) for k in KEYS}, dict.fromkeys(KEYS, 0)
    for i, r in enumerate(T.itertuples()):
        c = r.cue_daq

        def win(w, x=back, c=c):
            return x[np.searchsorted(t, c + w[0]):np.searchsorted(t, c + w[1])]
        bc, bk = win(CUE_W), win(CTL_W)
        fc, fk = win(CUE_W, fwd), win(CTL_W, fwd)
        if len(bc) < 20 or len(bk) < 20 or not (np.isfinite(bc).all() and np.isfinite(bk).all()):
            continue
        above = np.flatnonzero(bc > THR)
        rows.append({"label": lab, "trial_id": r.trial_id, "cls": r.cls, "engaged": r.trial_id in eng,
                     "in_period": bool(period[i]),
                     "push": bool(len(above)), "push_ctl": bool((bk > THR).any()), "peak": float(bc.max()),
                     "latency_s": CUE_W[0] + above[0] / FS if len(above) else np.nan,
                     "walk_cue": float(np.mean(fc > WALK)), "walk_ctl": float(np.mean(fk > WALK))})
        j0 = np.searchsorted(t, c + T0)
        sb, sf = back[j0:j0 + NB], fwd[j0:j0 + NB]
        if len(sb) == NB and np.isfinite(sb).all() and np.isfinite(sf).all():
            x = np.stack([sf, sb, sf > WALK, sb > THR])
            for k in ("lick_in" if period[i] else "lick_out", "glm_in" if r.trial_id in eng else "glm_out"):
                trace[k] += x
                ntr[k] += 1
    return pd.DataFrame(rows), {k: (trace[k] / ntr[k] if ntr[k] >= 5 else None) for k in trace}


def boot_family(T, cols, name):
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
            post = {a: v for a, v in post.items() if len(v)}
            pre_ = {a: v for a, v in pre.items() if len(v)}
            o = ak.boot_delta(post, pre_, np.random.default_rng(0), draws=True) if post and pre_ else None
            if o is None or o[0] is None:
                rows.append({"family": name, "quantity": c, "epoch": e, "pre_mean": m["pre"].mean(), "delta": np.nan,
                             "lo": np.nan, "hi": np.nan, "p_boot": np.nan, "animals_same_sign": 0})
                continue
            iv, dr = o
            rows.append({"family": name, "quantity": c, "epoch": e, "pre_mean": m["pre"].mean(), "delta": iv.point,
                         "lo": iv.lo, "hi": iv.hi, "p_boot": ak.boot_p(dr),
                         "animals_same_sign": int((np.sign(d[e]) == np.sign(iv.point)).sum())})
    R = pd.DataFrame(rows)
    ok = R.p_boot.notna()
    R["p_holm"] = np.nan
    R.loc[ok, "p_holm"] = ak.holm(R.p_boot[ok].to_numpy())
    R["ci_excl_0"] = (R.lo > 0) | (R.hi < 0)
    return R


GATES = {"lick": ("ENGAGED: lick working period", "NOT ENGAGED: outside it",
                   "the camera LICK working period (first .. last camera lick of the session)", "2a"),
         "glm": ("ENGAGED: GLM-HMM states 0-2", "NOT ENGAGED: GLM-HMM states 3-4",
                 "the GLM-HMM gate (K5 design v2, argmax state: 0-2 engaged; 3 partial + 4 disengaged = not)", "2b")}


def figure(per: dict, path: Path, gate: str = "lick"):
    """Rows: engaged / not engaged under ``gate``; columns: the 4 PANELS. ``per`` = {label: {key: (4, NB) or None}}."""
    r_in, r_out, gdesc, fno = GATES[gate]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as col
    tt = T0 + (np.arange(NB) + 0.5) / FS
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True)
    for row, (key, rname) in enumerate(((f"{gate}_in", r_in), (f"{gate}_out", r_out))):
        for k, (_key, ylab) in enumerate(PANELS):
            ax = axs[row, k]
            for e in EPOCHS:
                an = {}
                for lab, d in per.items():
                    if epoch(lab) == e and d[key] is not None:
                        an.setdefault(lab[:4], []).append(d[key][k])
                if not an:
                    continue
                A = np.stack([np.mean(an[a], 0) for a in sorted(an)])
                if k == 1:
                    A = -A                                   # backward plotted as negative velocity (Priya 11-17)
                m = A.mean(0)
                se = A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.zeros_like(m)
                ax.fill_between(tt, m - se, m + se, color=col[e], alpha=0.08, lw=0, zorder=2)
                ax.plot(tt, m, color=col[e], lw=1.6, label=f"{e} (n={len(A)})", zorder=4)
            ax.axvline(0, color="0.6", lw=0.8, ls="--")
            for w, lab_ in ((CUE_W, "cue window"), (CTL_W, "control")):
                ax.axvspan(*w, color="0.85", alpha=0.35, lw=0, zorder=0)
                ax.text(np.mean(w), ax.get_ylim()[1] - 0.03 * np.ptp(ax.get_ylim()), lab_, ha="center", va="top",
                        fontsize=7, color="0.4")
            ax.set_ylabel(ylab, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 1:
                ax.set_xlabel("time from cue (s)")
            if k == 0:
                ax.set_title(rname, loc="left", fontsize=10, fontweight="bold")
            if k == 1:
                ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Fig. {fno}. Treadmill around the cue (DAQ wheel), engaged vs not under {gdesc}\n"
                 "mean over animals (each = mean of its sessions) +- SEM across animals; n = animals with >= 5 trials "
                 "in that set", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    d = pd.read_csv(S / "glmhmm_v1" / "states_pooled_all_K5_v2.csv.gz")
    pc = [f"p_state{k}" for k in range(5)]
    d["arg"] = d[pc].to_numpy().argmax(1)
    eng = {lab: set(g.trial_id[g.arg.isin([0, 1, 2])]) for lab, g in d.groupby("label")}
    labs = [Path(f).stem for f in sorted(glob.glob(str(FEAT / "*.npz"))) if epoch(Path(f).stem) and Path(f).stem in eng]
    res, fail = ak.fan_sessions([(lab, eng[lab]) for lab in labs], one, key=lambda it: it[0])
    print("failed:", fail)
    Tr = pd.concat([r[0] for _it, r in res if r is not None])
    Tr.to_csv(OUT / "cue_push_per_trial.csv", index=False)
    per = {it[0]: r[1] for it, r in res if r is not None}
    figure(per, OUT / "fig2a_cue_wheel_lickgate.png", "lick")
    figure(per, OUT / "fig2b_cue_wheel_glmgate.png", "glm")
    pd.set_option("display.width", 220)
    # gate agreement, trial by trial, per epoch (animal-weighted shares)
    Tr["animal"], Tr["epoch"] = Tr.label.str[:4], Tr.label.map(epoch)
    Tr["agree"] = np.select([Tr.in_period & Tr.engaged, Tr.in_period & ~Tr.engaged, ~Tr.in_period & Tr.engaged],
                            ["both engaged", "lick only", "GLM only"], "neither")
    A = (Tr.groupby(["epoch", "animal"]).agree.value_counts(normalize=True).unstack(fill_value=0)
         .groupby("epoch").mean().reindex(EPOCHS))
    print("\n######## GATE AGREEMENT (share of trials; animal-weighted)")
    print(A.round(3).to_string())
    A.to_csv(OUT / "gate_agreement_lick_vs_glm.csv")
    out = []
    for sub, m in (("all", Tr.engaged | ~Tr.engaged), ("lick gate engaged", Tr.in_period),
                   ("lick gate NOT engaged", ~Tr.in_period), ("GLM-HMM engaged", Tr.engaged),
                   ("GLM-HMM NOT engaged", ~Tr.engaged)):
        g = Tr[m].groupby("label")
        P = pd.DataFrame({"push_rate": g.push.mean(), "control_rate": g.push_ctl.mean(),
                          "cue_minus_control": g.push.mean() - g.push_ctl.mean(),
                          "peak_on_push": g.apply(lambda x: x.peak[x.push].mean()),
                          "latency_on_push": g.apply(lambda x: x.latency_s[x.push].median()),
                          "walk_frac_cue": g.walk_cue.mean(), "walk_frac_ctl": g.walk_ctl.mean()}).reset_index()
        out.append(boot_family(P, ["push_rate", "control_rate", "cue_minus_control", "peak_on_push",
                                   "latency_on_push", "walk_frac_cue", "walk_frac_ctl"],
                               f"push, {sub} trials"))
    R = pd.concat(out)
    R.to_csv(OUT / "cue_push_delta_vs_own_pre.csv", index=False)
    print("\n######## SUMMARY")
    print(R.round(3).to_string(index=False))

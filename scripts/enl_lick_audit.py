"""ENL-LICK AUDIT: how many licks fall in the enforced-no-lick (ENL) pre-cue window, per animal x epoch x spout, and is
there enough of them for (1) the ITI balance / decoder offset on stationary-spout, unrewarded, uncued licks, (2) a test of
whether the far-contra aim deficit is cue-triggered, (3) a target-code test without reward activity? (Priya 10-29: PS92 /
PS93 have many.) COUNTS ONLY -- no imaging is read.

DEFINITION (as in the code base, nothing new):
  ENL lick  = a cam4 lick of `licks_all.csv` with context ``pre_cue`` (`orofacial_outputs.lick_context`: position strobe
              <= onset < cue). The firmware pulses the strobe AFTER the spout move completes (daq_trials docstring), so
              the spout is in place and stationary; the cue has not sounded; reward_mode = auto_after_delay with delay 0
              (gui_config) = water comes AT the cue, so no reward precedes an ENL lick within the trial.
  TASK      gui_config `enforce` = true and timing.precue_min..max = 2000..3000 ms for every June+ session: each lick in
              the ENL fires `pre_cue_reset_by_lick` (events.csv) and RESTARTS the pre-cue timer -- the trial is not aborted,
              the cue is delayed until 2-3 s pass lick-free. Checked here: cue - last ENL lick onset, cue - strobe with /
              without ENL licks, GUI reset count vs cam4 ENL licks, hit rate with / without ENL licks.
  BOUT      ENL licks of one trial split at onset gaps > orofacial_cohort.ILI_MAX_MS (300 ms). A bout start is
              CONTINUED when any lick (any context) has its onset in the 300 ms before it (licking that began before the
              strobe), and CLEAN when it is >= 1 s after the strobe and no lick of any context lies in the preceding 1 s.
  ANGLE     3-D tongue angle = degrees(arcsin(side_ml / reach_mm)) (as lick_enl_angle_decode.prepare; + = contralesional,
              all lesions left) from pose3d/kinematics_all/licks.csv.gz joined on lick_uid (= row of licks_all.csv).
              Aim error = angle - the animal's PRE (August) mean angle of all post-cue 3-D licks at that spout; the same
              error is given for the cell's own post-cue licks (cued) for a like-for-like comparison.
  CONTACT   licks_all.csv `contact` (DAQ contact matched to the cam4 lick).
  IMAGING   session in iti_licks.run_labels() (the frozen-encoder run = the ITI / cued_wide extraction set, August+).
  EPOCHS    wfield_local.epochs.epoch_of; pre = August sessions only (day >= 0801); 0817 excluded by epoch_of.

Outputs <il.OUT_W>/enl_licks/: enl_licks.csv.gz (one row per ENL lick), enl_trials.csv.gz (one row per trial),
cued_cells.csv (post-cue reference per session x spout), enl_audit_animal.csv (animal x epoch x spout + "all" rows),
enl_audit_pooled.csv (epoch x spout pooled over animals), enl_task_check.csv (reset / outcome check per animal x epoch),
figures/enl_lick_audit.png (+ figdata/enl_lick_audit.pkl).
Usage: python scripts/enl_lick_audit.py [--jobs 4] [--plot-only]
"""
from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT = il.OUT_W / "enl_licks"
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
BOUT_GAP_S = 0.300          # orofacial_cohort.ILI_MAX_MS
ARRIVAL_S = 1.0             # GCaMP6s spout-arrival response overlap
CLEAN_PRE_S = 1.0
THIN = 20


def _sessions() -> list[str]:
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    base = oo.root(PathResolver()) / oc.RUN / "sessions"
    out = []
    for a in AN:
        for d in sorted(p.name for p in (base / a).iterdir() if p.is_dir()):
            lab = f"{a}_{d[4:]}"
            ep = epochs.epoch_of(lab)
            if ep in EP and not (ep == "pre" and d[4:6] < "08") and (base / a / d / "licks_all.csv").exists():
                out.append(lab)
    return out


def _gui_resets(an: str, date: str) -> dict:
    hits = sorted(glob.glob(f"M:/MICROSCOPE/Priya/Behavior_logs/Widefield/{an}_{date}_*/events.csv"))
    if not hits:
        return {}
    ev = pd.read_csv(hits[0], usecols=lambda c: c in ("event_name", "trial_id"))
    r = ev[ev.event_name == "pre_cue_reset_by_lick"]
    return r.groupby("trial_id").size().to_dict()


_KIN = {}


def _kin(session: str) -> pd.DataFrame:
    """3-D kinematics rows of one session; the whole table is read once per worker process."""
    if "all" not in _KIN:
        from wfield_local.paths import PathResolver
        p = Path(PathResolver().root("dlc")) / "session_poses" / "pose3d" / "kinematics_all" / "licks.csv.gz"
        K = pd.read_csv(p, usecols=["session", "lick_uid", "side_ml", "reach_mm"])
        _KIN["all"] = {s: g.drop(columns="session") for s, g in K.groupby("session")}
    return _KIN["all"].get(session, pd.DataFrame(columns=["lick_uid", "side_ml", "reach_mm"]))


def worker(label):
    """label -> dict of DataFrames. Module level (spawn)."""
    K = _kin(f"{label[:4]}_2026{label.split('_')[-1]}")
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    an, date = label[:4], "2026" + label.split("_")[-1]
    ep = epochs.epoch_of(label)
    T = il._behaviour(label).trials.copy()
    A = pd.read_csv(oo.session_dir(PathResolver(), oc.RUN, an, date) / "licks_all.csv",
                    usecols=["trial_id", "position", "context", "contact", "daq_on_s", "daq_peak_s"])
    A["lick_uid"] = np.arange(len(A))
    K = K.drop_duplicates("lick_uid").set_index("lick_uid")
    A["angle"] = np.degrees(np.arcsin(np.clip(K.side_ml / K.reach_mm, -1, 1))).reindex(A.lick_uid).to_numpy()
    A["pos"] = [oc.pos_key(p, "L") if isinstance(p, str) and "_" in p else "" for p in A.position]
    on_all = np.sort(A.daq_on_s.dropna().to_numpy(float))
    tr = T.set_index("trial_id")
    E = A[A.context == "pre_cue"].copy()
    E = E[E.trial_id.isin(tr.index)]
    E["strobe_s"] = E.trial_id.map(tr.strobe_s)
    E["cue_s"] = E.trial_id.map(tr.cue_s)
    E["pos"] = E.trial_id.map(tr.pos)                                    # the DAQ trial's spout
    E["t_from_strobe"] = E.daq_on_s - E.strobe_s
    E["t_to_cue"] = E.cue_s - E.daq_on_s
    E = E.sort_values("daq_on_s")
    bouts = []
    for _, g in E.groupby("trial_id", sort=False):
        b = np.concatenate([[0], np.cumsum(np.diff(g.daq_on_s.to_numpy()) > BOUT_GAP_S)])
        bouts.append(pd.Series(b, index=g.index))
    E["bout"] = pd.concat(bouts) if bouts else pd.Series(dtype=int)
    E["bout_start"] = ~E.duplicated(["trial_id", "bout"])
    on = E.daq_on_s.to_numpy(float)
    prev = np.array([on_all[np.searchsorted(on_all, t - 0.005) - 1] if np.searchsorted(on_all, t - 0.005) > 0 else -np.inf
                     for t in on])                                      # previous onset of ANY context (not itself)
    E["prev_gap"] = on - prev
    E["continued"] = E.bout_start & (E.prev_gap <= BOUT_GAP_S)
    E["clean_start"] = E.bout_start & (E.t_from_strobe >= ARRIVAL_S) & (E.prev_gap >= CLEAN_PRE_S)
    E["label"], E["animal"], E["epoch"] = label, an, ep
    # trials
    res = _gui_resets(an, date)
    post = A[A.context == "post_cue"]
    hit = post.groupby("trial_id").size()
    n_enl = E.groupby("trial_id").size()
    last = E.groupby("trial_id").daq_on_s.max()
    TT = T[["trial_id", "pos", "strobe_s", "cue_s", "stop_s", "engaged"]].copy()
    TT["n_enl"] = TT.trial_id.map(n_enl).fillna(0).astype(int)
    TT["n_enl_bouts"] = TT.trial_id.map(E[E.bout_start].groupby("trial_id").size()).fillna(0).astype(int)
    TT["n_gui_resets"] = TT.trial_id.map(res).fillna(0).astype(int) if res else np.nan
    TT["hit_lp"] = TT.trial_id.map(hit).fillna(0).to_numpy() > 0
    TT["enl_dur"] = TT.cue_s - TT.strobe_s
    TT["prev_reward_to_strobe"] = TT.strobe_s - TT.cue_s.shift(1)      # auto reward is delivered AT the cue
    TT["cue_minus_last_enl"] = TT.cue_s - TT.trial_id.map(last)
    lastc = E[E.contact.astype(bool)].groupby("trial_id").daq_on_s.max()
    TT["cue_minus_last_contact_enl"] = TT.cue_s - TT.trial_id.map(lastc)
    TT["n_enl_contact"] = TT.trial_id.map(E[E.contact.astype(bool)].groupby("trial_id").size()).fillna(0).astype(int)
    TT["label"], TT["animal"], TT["epoch"] = label, an, ep
    # cued reference cells (post-cue 3-D licks of this session, by the DAQ trial's spout)
    P = post[post.trial_id.isin(tr.index)].copy()
    P["pos"] = P.trial_id.map(tr.pos)
    C = P.groupby("pos").agg(n_cued=("angle", "size"), n_cued_3d=("angle", "count"), cued_ang_sum=("angle", "sum"),
                             cued_ang_sq=("angle", lambda v: float(np.nansum(np.square(v)))),
                             cued_contact=("contact", "sum")).reset_index()
    C["label"], C["animal"], C["epoch"] = label, an, ep
    keep = ["label", "animal", "epoch", "trial_id", "pos", "lick_uid", "daq_on_s", "t_from_strobe", "t_to_cue", "bout",
            "bout_start", "continued", "clean_start", "prev_gap", "contact", "angle"]
    return {"E": E[keep], "T": TT, "C": C}


def extract(jobs: int):
    from wfield_local import analysis_kit as ak
    labs = _sessions()
    res, fail = ak.fan_sessions(labs, worker, jobs=jobs, label="enl session")
    if fail:
        print("FAILED:", [str(f)[:160] for f in fail])
    E = pd.concat([r["E"] for _, r in res], ignore_index=True)
    T = pd.concat([r["T"] for _, r in res], ignore_index=True)
    C = pd.concat([r["C"] for _, r in res], ignore_index=True)
    return E, T, C, len(labs), fail


def summarise(E, T, C):
    img = set(il.run_labels())
    for D in (E, T, C):
        D["imaging"] = D.label.isin(img)
    # pre cued reference angle per animal x spout (all post-cue 3-D licks of the August pre sessions)
    Cp = C[C.epoch == "pre"].groupby(["animal", "pos"])[["cued_ang_sum", "n_cued_3d"]].sum()
    ref = (Cp.cued_ang_sum / Cp.n_cued_3d).rename("ref")
    E = E.join(ref, on=["animal", "pos"])
    E["aim_err"] = E.angle - E.ref
    C = C.join(ref, on=["animal", "pos"])

    def cell(e, t, c):
        n3 = int(e.angle.notna().sum())
        nc = c.n_cued_3d.sum()
        cued_mean = c.cued_ang_sum.sum() / nc if nc else np.nan
        ref_ = c.ref.iloc[0] if len(c) and c.ref.nunique() == 1 else np.nan
        return {"n_sessions": t.label.nunique(), "n_trials": len(t), "n_trials_enl": int((t.n_enl > 0).sum()),
                "frac_trials_enl": (t.n_enl > 0).mean() if len(t) else np.nan,
                "n_enl_bouts": int(e.bout_start.sum()), "n_enl_licks": len(e),
                "licks_per_bout": len(e) / max(int(e.bout_start.sum()), 1),
                "n_bouts_continued": int(e.continued.sum()), "n_bouts_clean": int(e.clean_start.sum()),
                "med_t_from_strobe_s": e.t_from_strobe.median(), "n_licks_ge1s": int((e.t_from_strobe >= ARRIVAL_S).sum()),
                "contact_enl": e.contact.mean() if len(e) else np.nan,
                "contact_cued": c.cued_contact.sum() / c.n_cued.sum() if c.n_cued.sum() else np.nan,
                "n_enl_3d": n3, "angle_enl": e.angle.mean(), "angle_cued": cued_mean,
                "aim_err_enl": e.aim_err.mean(), "aim_err_enl_sd": e.aim_err.std(),
                "aim_err_cued": cued_mean - ref_ if np.isfinite(ref_) else np.nan,
                "n_sessions_img": t[t.imaging].label.nunique(), "n_enl_licks_img": int(e.imaging.sum()),
                "n_enl_3d_img": int((e.imaging & e.angle.notna()).sum()),
                "n_bouts_clean_img": int((e.imaging & e.clean_start).sum())}

    rows = []
    for (a, ep), te in T.groupby(["animal", "epoch"]):
        ee, ce = E[(E.animal == a) & (E.epoch == ep)], C[(C.animal == a) & (C.epoch == ep)]
        for p in [*__import__("wfield_local.orofacial_cohort", fromlist=["x"]).POS_ORDER, "all"]:
            sel = (lambda d: d) if p == "all" else (lambda d, p=p: d[d.pos == p])
            r = cell(sel(ee), sel(te), sel(ce) if p != "all" else ce.assign(ref=np.nan))
            rows.append({"animal": a, "epoch": ep, "spout": p, **r})
    S = pd.DataFrame(rows)
    from wfield_local.orofacial_cohort import POS_ORDER
    order = {p: i for i, p in enumerate([*POS_ORDER, "all"])}
    S = S.sort_values(["animal", "epoch", "spout"], key=lambda s: s.map({**{e: i for i, e in enumerate(EP)}, **order})
                      if s.name != "animal" else s).reset_index(drop=True)
    S["thin"] = (S.n_enl_licks_img < THIN) | (S.n_enl_3d_img < THIN) | (S.n_sessions_img < 3)   # < 3 imaging sessions too
    # pooled over animals: sums; means weighted by licks; aim errors = mean of per-animal cells (animal-equal)
    agg = {k: "sum" for k in ["n_sessions", "n_trials", "n_trials_enl", "n_enl_bouts", "n_enl_licks", "n_bouts_continued",
                              "n_bouts_clean", "n_licks_ge1s", "n_enl_3d", "n_sessions_img", "n_enl_licks_img",
                              "n_enl_3d_img", "n_bouts_clean_img"]}
    agg.update({k: "mean" for k in ["contact_enl", "contact_cued", "aim_err_enl", "aim_err_cued", "med_t_from_strobe_s"]})
    Pp = S.groupby(["epoch", "spout"], sort=False).agg(agg).reset_index()
    Pp["n_animals_ge20"] = S.assign(ok=S.n_enl_3d_img >= THIN).groupby(["epoch", "spout"], sort=False).ok.sum().to_numpy()
    Pp["frac_trials_enl"] = Pp.n_trials_enl / Pp.n_trials
    # task check
    tk = []
    for (a, ep), t in T.groupby(["animal", "epoch"]):
        w, wo = t[t.n_enl > 0], t[t.n_enl == 0]
        g = t.dropna(subset=["n_gui_resets"])
        tk.append({"animal": a, "epoch": ep, "n_trials": len(t),
                   "enl_dur_med_with": w.enl_dur.median(), "enl_dur_med_without": wo.enl_dur.median(),
                   "cue_minus_last_enl_min": w.cue_minus_last_enl.min(), "cue_minus_last_enl_p01": w.cue_minus_last_enl.quantile(.01),
                   "cue_minus_last_contact_enl_p01": t.cue_minus_last_contact_enl.quantile(.01),
                   "cue_minus_last_contact_enl_med": t.cue_minus_last_contact_enl.median(),
                   "cue_minus_last_enl_med": w.cue_minus_last_enl.median(),
                   "frac_enl_trials_last_lick_lt1s": (w.cue_minus_last_enl < 1.0).mean(),
                   "r_resets_vs_cam4_contact": np.corrcoef(g.n_gui_resets, g.n_enl_contact)[0, 1] if len(g) > 2 and g.n_enl_contact.std() > 0 else np.nan,
                   "gui_resets_per_cam4_contact": g.n_gui_resets.sum() / max(g.n_enl_contact.sum(), 1),
                   "prev_reward_to_strobe_med": t.prev_reward_to_strobe.median(),
                   "hit_with": w.hit_lp.mean(), "hit_without": wo.hit_lp.mean(),
                   "gui_resets_total": g.n_gui_resets.sum(), "cam4_enl_total": g.n_enl.sum(),
                   "r_resets_vs_cam4": np.corrcoef(g.n_gui_resets, g.n_enl)[0, 1] if len(g) > 2 and g.n_enl.std() > 0 else np.nan,
                   "frac_trials_reset_but_no_cam4": ((g.n_gui_resets > 0) & (g.n_enl == 0)).mean(),
                   "frac_trials_cam4_but_no_reset": ((g.n_gui_resets == 0) & (g.n_enl > 0)).mean()})
    TK = pd.DataFrame(tk)
    TK["_e"] = TK.epoch.map({e: i for i, e in enumerate(EP)})
    TK = TK.sort_values(["animal", "_e"]).drop(columns="_e")
    Pp["_e"] = Pp.epoch.map({e: i for i, e in enumerate(EP)})
    Pp["_s"] = Pp.spout.map(order)
    Pp = Pp.sort_values(["_e", "_s"]).drop(columns=["_e", "_s"])
    return E, S, Pp, TK


def figure(S, Pp):
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE, POS_ORDER
    sp = list(POS_ORDER)
    # The two count / fraction-of-trials heatmaps were removed from the figure (Priya 10-29: "get rid of the ENL
    # heatmap"); the counts stay in enl_audit_animal.csv / enl_audit_pooled.csv.
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
    for k, (col, ref, ttl, yl) in enumerate([("aim_err_enl", "aim_err_cued",
                                              "aim error vs PRE cued mean angle at the spout", "deg (+ = contralesional)"),
                                             ("contact_enl", "contact_cued", "spout-contact rate", "fraction of licks")]):
        ax = axs[k]
        x = np.arange(len(sp))
        for i, e in enumerate(EP):
            Q = S[(S.epoch == e) & (S.spout != "all")]
            m_enl = Q.groupby("spout")[col].mean().reindex(sp)
            m_c = Q.groupby("spout")[ref].mean().reindex(sp)
            off = (i - 1.5) * 0.18
            ax.plot(x + off, m_enl, "o", color=EPOCH_LINE[e], ms=6, label=f"{e} ENL")
            ax.plot(x + off, m_c, "x", color=EPOCH_LINE[e], ms=6, label=f"{e} cued")
            for a in AN:
                q = Q[Q.animal == a].set_index("spout")[col].reindex(sp)
                ax.plot(x + off, q, ".", color=EPOCH_LINE[e], ms=2.5, alpha=0.6)
        if k == 0:
            ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks(x, [p.replace("_", "\n") for p in sp], fontsize=7)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(ttl + "\no = ENL (animal mean), x = cued same epoch, dots = animals", fontsize=8)
    fig.suptitle("ENL (pre-cue, spout stationary, no cue, no reward) lick audit -- PS92-95, pre = August; "
                 "ENL licks are EXCLUDED from the ITI and cued decoder lick sets", fontsize=9)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=len(l_), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    f = OUT / "figures" / "enl_lick_audit.png"
    f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=130)
    plt.close(fig)
    print("wrote", f)


def main() -> int:
    import time
    from wfield_local import writeguard
    t0 = time.time()
    writeguard.assert_writable(OUT / "enl_audit_animal.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figdata").mkdir(exist_ok=True)
    fd = OUT / "figdata" / "enl_lick_audit.pkl"
    if "--plot-only" in sys.argv:
        with open(fd, "rb") as fh:
            figure(*pickle.load(fh))
        return 0
    jobs = min(int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4, 4)
    E, T, C, nlab, fail = extract(jobs)
    E, S, Pp, TK = summarise(E, T, C)
    E.to_csv(OUT / "enl_licks.csv.gz", index=False)
    T.to_csv(OUT / "enl_trials.csv.gz", index=False)
    C.to_csv(OUT / "cued_cells.csv", index=False)
    S.to_csv(OUT / "enl_audit_animal.csv", index=False)
    Pp.to_csv(OUT / "enl_audit_pooled.csv", index=False)
    TK.to_csv(OUT / "enl_task_check.csv", index=False)
    with open(fd, "wb") as fh:
        pickle.dump((S, Pp), fh)
    figure(S, Pp)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"{nlab} sessions, {len(fail)} failed, {time.time() - t0:.0f} s")
    print(TK.round(3).to_string(index=False))
    cols = ["animal", "epoch", "spout", "n_sessions", "n_trials", "frac_trials_enl", "n_enl_bouts", "n_enl_licks",
            "n_bouts_clean", "med_t_from_strobe_s", "n_licks_ge1s", "contact_enl", "contact_cued", "n_enl_3d",
            "aim_err_enl", "aim_err_cued", "n_sessions_img", "n_enl_3d_img", "n_bouts_clean_img", "thin"]
    print(S[cols].round(2).to_string(index=False))
    print(Pp.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

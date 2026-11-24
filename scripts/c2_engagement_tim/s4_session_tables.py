"""Step 4: per-session tables (six-position cohort, PS92-PS95) for engagement, TIM, motion energy, stillness, running,
the composite and the engaged-only licking readouts. PROMPT_engagement_movement_both_cohorts.md items 4-7, 10, 11.

TIME MATCHING (item 7; Priya 11-20 00:25): "first95" = trials / frames up to 95 min after the session's first cue
(`behavior_states.COMMON_SESSION_MIN`; sessions shorter than 95 min enter whole: 9 of 107); "first74" = 74 min, the
shortest session (PS94_20260811, 74.0 min), so every session is cut to the same length and no animal / session is
dropped (strict reading of item 7); "all" = unmatched. Every table carries a ``match`` column.
Sessions = the analysis sessions (`behavior_states.session_epoch`: August pre, acute, subacute, chronic; 107).

ENGAGEMENT: production GLM-HMM K5 v2 (`behavior_states.glmhmm_table`, merged 0+1), per session mean posterior per role
(`occupancy_per_session`); trial role = argmax state's role (production gate); refits (s3) the same way.
TIM (pipeline default, check): `behavior_states.tim_trials('within')`, working trials, per-session mean (px).
TIM (new windows, s2): working trials (pre-cue: lick-free ENL only), per-session mean (px).
MOTION ENERGY (Yin et al. STAR Methods): per body part frame-to-frame position change, > 5 SD dropped (s1), z-scored
per part against the ANIMAL'S OWN pre-stroke frames (mean / SD over its August pre sessions' frames inside the same
time match; z is affine, so the per-session level is kept), averaged over parts (keypoint ME: tongue 3-D, jaw, nose)
-> per frame; per trial = mean over the window's frames; per session = mean over trials (every trial in the time match;
pre-cue lick-free trials only; ITI lick-buffered docked frames). FaceRhythm ME = whole-face mean per-point speed
(own tracking), z vs the animal's pre frames; also the contra- and ipsilesional halves.
STILLNESS: fraction of frames below the animal's pre-stroke 10th percentile of the same quantity in the same frame set
(ITI = docked lick-buffered frames; within trials = ENL + response frames; whole session = every frame first cue ..
+N min); treadmill = speed (forward + backward) < 1 mm/s.
RUNNING (treadmill forward speed, `arhmm_features` run_fwd = the states page's `load_session` signal): running = > 3
mm/s (segmentation.running.thresh_speed_mm_s); bouts = contiguous running bins, gaps < 0.2 s merged, bouts < 0.2 s
dropped; whole session: mean forward speed, mean speed inside bouts, bouts / min, fraction of time running, median
bout duration; between trials (docked ITI frames): mean forward speed, fraction running.
STATE SPLIT: the same per-session means over trials (frames) whose owner trial's GLM-HMM role is engaged / partial /
disengaged ('not_engaged' = partial + disengaged).
Usage: WF_REPO=... MC=... python s4_session_tables.py <cache_dir> <out_dir> [--no-tim]
"""
from wfield_local.session_licks import kin_name as _kin_name  # 11-22: fused post-cue unless WIDEFIELD_KIN_LICKS=cam4
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline")))
sys.path.insert(0, str(Path(os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline"))) / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from wfield_local import behavior_states as B  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trial_scaffold  # noqa: E402  (collector trials, or the 11-22 fallback for a run without them)

CACHE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
# --no-tim (11-22, manuscript v26: TIM removed from the paper): skip the TIM tables (s2 not run) and the treadmill
# control-window rates (read only by s5's composite, which needs TIM); every other table is unchanged
NO_TIM = "--no-tim" in sys.argv[3:]
MATCH = {"first95": 95.0, "first74": 74.0, "all": np.inf}
FR_KEYS = ("fr_all", "fr_contra", "fr_ipsi")
KP_PARTS = ("tongue", "jaw", "nose")
ROLE_SETS = {"engaged": ("engaged",), "partial": ("partial",), "disengaged": ("disengaged",),
             "not_engaged": ("partial", "disengaged"), "all": ("engaged", "partial", "disengaged", "none")}


def W(name, df):
    f = OUT / name
    if f.exists():
        sys.exit(f"refusing to overwrite {f}")
    df.to_csv(f, index=False)
    print("wrote", f.name, len(df), flush=True)


C = {"trials": trial_scaffold.trials(OUT)}
T = C["trials"]
T = T[T.label.map(B.session_epoch).notna()].copy()
T["epoch"] = T.label.map(B.session_epoch)
T["date"] = T.label.str.split("_").str[1]
G = B.glmhmm_table()
ROLES = B.glmhmm_roles(G)
pc = B._pcols(G)
G["role"] = [ROLES[int(pc[i][7:])] for i in G[pc].to_numpy().argmax(1)]
G = B.add_session_time(G, C["trials"])
role_of = {(l, int(t)): r for l, t, r in zip(G.label, G.trial_id, G.role)}
tmin_of = {(l, int(t)): m for l, t, m in zip(T.label, T.trial_id, T.t_min)}
labels = sorted(T.label.unique())
ep_of = {l: B.session_epoch(l) for l in labels}


def base(df):
    df["animal"] = df.label.str.split("_").str[0]
    df["epoch"] = df.label.map(ep_of)
    df["date"] = df.label.str.split("_").str[1]
    df["pos"] = df.get("pos", "all")
    return df


# ---------------------------------------------------------------- engagement (production + refits)
rows = []
for m, N in MATCH.items():
    S = B.occupancy_per_session(G[~(G.t_min > N)])
    S["match"], S["model"] = m, "production"
    rows.append(S)
for v in ("ipsimid", "init"):
    f = OUT / f"glmhmm_refit_{v}_K5.csv.gz"
    if not f.exists():
        continue
    R = pd.read_csv(f)
    R = R[R.label.map(B.session_epoch).notna()]
    R = B.add_session_time(R, C["trials"])
    for m, N in MATCH.items():
        Rm = R[~(R.t_min > N)]
        S = Rm.groupby("label").role.value_counts(normalize=True).unstack(fill_value=0).reset_index()
        # occupancy as mean posterior per role (as production)
        rmap = Rm.groupby("state").role.first().to_dict()
        P = Rm.groupby("label")[[c for c in Rm if c.startswith("p_state")]].mean()
        for r in B.ROLES:
            P[r] = P[[f"p_state{k}" for k, rr in rmap.items() if rr == r]].sum(1) if any(rr == r for rr in rmap.values()) else 0.0
        P = P.reset_index()[["label", *B.ROLES]]
        P["match"], P["model"] = m, f"refit_{v}"
        rows.append(P)
E = base(pd.concat(rows, ignore_index=True)[["label", "match", "model", *B.ROLES]])
W("sessions_engagement.csv", E)

# trial roles of the refits (for engaged-only readouts)
REFIT_ROLE = {}
for v in ("ipsimid", "init"):
    f = OUT / f"glmhmm_refit_{v}_K5.csv.gz"
    if f.exists():
        R = pd.read_csv(f, usecols=["label", "trial_id", "role"])
        REFIT_ROLE[v] = {(l, int(t)): r for l, t, r in zip(R.label, R.trial_id, R.role)}

# ---------------------------------------------------------------- TIM: pipeline default + new windows
def tim_tables():
    D = B.tim_trials("within")
    D = D[D.working.astype(bool)].copy()
    D["t_min"] = [tmin_of.get((l, int(t)), np.nan) for l, t in zip(D.label, D.trial_id)]
    rows = []
    for m, N in MATCH.items():
        Dm = D[~(D.t_min > N)]
        for rs, rr in (("all", None), ("engaged", ("engaged",)), ("not_engaged", ("partial", "disengaged"))):
            Dr = Dm if rr is None else Dm[Dm.role.isin(rr)]
            S = Dr.pivot_table(index="label", columns=["period", "part"], values="tim_px").reset_index()
            S.columns = ["label"] + [f"timpipe_{a}_{b}" for a, b in S.columns[1:]]
            S["match"], S["state"] = m, rs
            rows.append(S)
    TP = base(pd.concat(rows, ignore_index=True))
    W("sessions_tim_pipeline.csv", TP)

    N_ = pd.read_csv(OUT / "tim_new_trials.csv.gz")
    N_ = N_[N_.label.isin(labels)].copy()
    N_["t_min"] = [tmin_of.get((l, int(t)), np.nan) for l, t in zip(N_.label, N_.trial_id)]
    N_["role"] = [role_of.get((l, int(t)), "none") for l, t in zip(N_.label, N_.trial_id)]
    for v, d in REFIT_ROLE.items():
        N_[f"role_{v}"] = [d.get((l, int(t)), "none") for l, t in zip(N_.label, N_.trial_id)]
    N_.to_csv(OUT / "tim_new_trials_annotated.csv.gz", index=False, compression="gzip") if not (OUT / "tim_new_trials_annotated.csv.gz").exists() else None
    rows = []
    for m, N in MATCH.items():
        Dm = N_[~(N_.t_min > N)]
        for rs, rr in ROLE_SETS.items():
            for wk in ("working", "alltrials"):
                Dr = Dm[Dm.role.isin(rr)]
                if wk == "working":
                    Dr = Dr[Dr.working.astype(bool)]
                S = Dr.pivot_table(index="label", columns=["source", "period", "part"], values="tim").reset_index()
                S.columns = ["label"] + [f"tim{a}_{b}_{c}" for a, b, c in S.columns[1:]]
                S["match"], S["state"], S["trials"] = m, rs, wk
                rows.append(S)
    TN = base(pd.concat(rows, ignore_index=True))
    W("sessions_tim_new.csv", TN)


if not NO_TIM:
    tim_tables()

# ---------------------------------------------------------------- frames: ME, stillness, running
# 11-23 (four-camera rule, CC-B audit violation 6): a frames cache built by the 11-23 s1 holds the four-camera build
# (fused-lick exclusions, 3-D jaw / nose) plus the legacy pieces for comparison. S4_KP_JAWNOSE=2d reads the cam4 2-D
# jaw / nose (kp2d_d_*); S4_LICKS=cam4 reads the cam4-list exclusions (*_nearlick_cam4, win_lick_cam4). Unset = the
# cache's own build. Older caches (frames_v3) have neither and are read as before.
KP_JAWNOSE = os.environ.get("S4_KP_JAWNOSE", "")
LICKS = os.environ.get("S4_LICKS", "")
# S4_GATE=working (Priya 11-23: every six-position trial gate = the camera working period): frames whose OWNER trial is
# outside the camera working period (<out_dir>/camera_working.csv ``work``, s0) get period code 0, so the pre-cue,
# response and ITI means use working-period trials only (an ITI span belongs to the trial that just ended). Unset = no
# trial gate (every trial inside the time match), as the manuscript's S15.
GATE = os.environ.get("S4_GATE", "")
_WORK = None


def _working_ids(label):
    global _WORK
    if _WORK is None:
        cw = pd.read_csv(OUT / "camera_working.csv")
        _WORK = {lab: set(g.trial_id[g.work.astype(bool)].astype(int)) for lab, g in cw.groupby("label")}
    return _WORK.get(label, set())


def load(label):
    with np.load(CACHE / f"{label}.npz") as z:
        d = {k: z[k] for k in z.files if not (k.startswith("pc") or k.startswith("tp"))}
    if KP_JAWNOSE == "2d":
        d["kp_d_jaw"], d["kp_d_nose"] = d["kp2d_d_jaw"], d["kp2d_d_nose"]
    if LICKS == "cam4":
        d["win_lick"] = d["win_lick_cam4"]
        for st in ("kp", "fr"):
            if f"{st}_nearlick_cam4" in d:
                d[f"{st}_nearlick"] = d[f"{st}_nearlick_cam4"]
    if GATE == "working":
        ids = np.array(sorted(_working_ids(label)), dtype=np.int64)
        for st in ("kp", "fr"):
            if f"{st}_per" in d:
                per = d[f"{st}_per"].copy()
                per[~np.isin(d[f"{st}_own"], ids)] = 0
                d[f"{st}_per"] = per
    for st in ("kp", "fr"):
        if f"{st}_own" in d:
            rm_ = {int(t): r for (l, t), r in role_of.items() if l == label}
            d[f"{st}_role"] = pd.Series(d[f"{st}_own"]).map(rm_).fillna("none").to_numpy()
    return d


Z = {lab: load(lab) for lab in labels}
animals = sorted({l.split("_")[0] for l in labels})


def kp_series(z):
    return {p: z[f"kp_d_{p}"].astype(float) for p in KP_PARTS}


def bouts(fwd, dt=0.04, thr=3.0, gap=0.2, mind=0.2):
    run = np.nan_to_num(fwd) > thr
    if not run.any():
        return np.zeros(0), np.zeros(0)
    d = np.diff(np.r_[0, run.astype(int), 0])
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep_s, keep_e = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if (a - keep_e[-1]) * dt < gap:
            keep_e[-1] = b
        else:
            keep_s.append(a); keep_e.append(b)
    s, e = np.array(keep_s), np.array(keep_e)
    ok = (e - s) * dt >= mind
    return s[ok], e[ok]


rows, prerows = [], []
for m, N in MATCH.items():
    # animal pre references (frames inside the match)
    ref = {}
    for a in animals:
        pre = [l for l in labels if l.startswith(a) and ep_of[l] == "pre"]
        kp = {p: np.concatenate([Z[l][f"kp_d_{p}"][Z[l]["kp_tmin"] <= N] for l in pre]).astype(float) for p in KP_PARTS}
        mu = {p: (np.nanmean(v), np.nanstd(v)) for p, v in kp.items()}
        frs = {k: np.concatenate([Z[l][k][Z[l]["fr_tmin"] <= N] for l in pre if k in Z[l]]).astype(float) for k in FR_KEYS}
        mf = {k: (np.nanmean(v), np.nanstd(v)) for k, v in frs.items()}
        ref[a] = (mu, mf)

    def zkp(z, a):
        mu = ref[a][0]  # noqa: B023
        X = np.column_stack([(z[f"kp_d_{p}"] - mu[p][0]) / mu[p][1] for p in KP_PARTS])
        return np.nanmean(X, 1)

    def zfr(z, a, k):
        mu, sd = ref[a][1][k]  # noqa: B023
        return (z[k].astype(float) - mu) / sd

    # per-frame series per session
    SER = {}
    for l in labels:
        z, a = Z[l], l.split("_")[0]
        s = {"kp": (zkp(z, a), z["kp_per"], z["kp_own"], z["kp_tmin"], z["kp_nearlick"], z["kp_role"]),
             "tread": (z["run_fwd"].astype(float) + z["run_back"].astype(float), z["kp_per"], z["kp_own"], z["kp_tmin"],
                       z["kp_nearlick"], z["kp_role"])}
        if "fr_all" in z:
            for k in FR_KEYS:
                s[k] = (zfr(z, a, k), z["fr_per"], z["fr_own"], z["fr_tmin"], z["fr_nearlick"], z["fr_role"])
        lick_pre = {int(t) for c, t, lk in zip(z["win_code"], z["win_own"], z["win_lick"]) if c == 2 and lk}
        SER[l] = (s, lick_pre)
    # stillness thresholds: animal pre 10th percentile per (signal, frame set)
    def frame_sets(per, tmin, near, x):
        inN = tmin <= N  # noqa: B023
        return {"iti": inN & (per == 1) & ~near & np.isfinite(x), "trial": inN & ((per == 2) | (per == 3)) & np.isfinite(x),
                "session": inN & (tmin >= 0) & np.isfinite(x)}
    thr = {}
    for a in animals:
        pre = [l for l in labels if l.startswith(a) and ep_of[l] == "pre"]
        for sig in ("kp", "fr_all"):
            for fs_ in ("iti", "trial", "session"):
                v = np.concatenate([SER[l][0][sig][0][frame_sets(SER[l][0][sig][1], SER[l][0][sig][3], SER[l][0][sig][4], SER[l][0][sig][0])[fs_]]
                                    for l in pre if sig in SER[l][0]])
                thr[(a, sig, fs_)] = np.nanpercentile(v, 10)
    for l in labels:
        s, lick_pre = SER[l]
        a = l.split("_")[0]
        for rs, rr in ROLE_SETS.items():
            r = {"label": l, "match": m, "state": rs}
            for sig, (x, per, own, tmin, near, role) in s.items():
                inr = np.isin(role, rr) if rs != "all" else np.ones(len(x), bool)
                FS = frame_sets(per, tmin, near, x)
                # per-trial window means -> session mean
                for code, wn in ((2, "precue"), (3, "response"), (1, "iti")):
                    mask = (per == code) & (tmin <= N) & inr & np.isfinite(x)
                    if code == 1:
                        mask &= ~near
                    if code == 2:
                        mask &= ~np.isin(own, list(lick_pre))
                    if sig == "tread":
                        continue
                    if mask.any():
                        tm = pd.Series(x[mask]).groupby(own[mask]).mean()
                        r[f"me_{sig}_{wn}"] = tm.mean()
                if sig in ("kp", "fr_all"):
                    for fs_ in ("iti", "trial", "session"):
                        mk = FS[fs_] & (inr if fs_ != "session" else True)
                        if mk.any():
                            r[f"still_{sig}_{fs_}"] = float(np.mean(x[mk] < thr[(a, sig, fs_)]))
                if sig == "tread":
                    for fs_ in ("iti", "trial", "session"):
                        mk = FS[fs_] & (inr if fs_ != "session" else True)
                        if mk.any():
                            r[f"still_tread_{fs_}"] = float(np.mean(x[mk] < 1.0))
            # running
            z = Z[l]
            fwd = z["run_fwd"].astype(float)
            per, own, tmin, near = z["kp_per"], z["kp_own"], z["kp_tmin"], z["kp_nearlick"]
            if rs == "all":
                ses = (tmin >= 0) & (tmin <= N) & np.isfinite(fwd)
                f = fwd[ses]
                bs, be = bouts(f)
                minutes = ses.sum() * 0.04 / 60
                r.update(run_mean_speed=np.mean(f), run_bout_speed=np.mean([f[a_:b_].mean() for a_, b_ in zip(bs, be)]) if len(bs) else np.nan,
                         run_bouts_per_min=len(bs) / minutes, run_frac=np.mean(f > 3.0),
                         run_bout_dur=np.median((be - bs) * 0.04) if len(bs) else np.nan)
            inr = np.isin(z["kp_role"], rr) if rs != "all" else np.ones(len(fwd), bool)
            it = (per == 1) & (tmin <= N) & inr & np.isfinite(fwd)
            if it.any():
                r.update(run_iti_mean_speed=np.mean(fwd[it]), run_iti_frac=np.mean(fwd[it] > 3.0))
            rows.append(r)
    print("frames done", m, flush=True)
M = base(pd.DataFrame(rows))
W("sessions_motion.csv", M)

# ---------------------------------------------------------------- treadmill control-window rates (composite)
if not NO_TIM:
    rows = []
    for m, N in MATCH.items():
        S = B.per_session(T[~(T.t_min > N)], "lick_period")
        S["match"] = m
        rows.append(S)
    TRD = base(pd.concat(rows, ignore_index=True))
    W("sessions_treadmill_rates.csv", TRD)

# ---------------------------------------------------------------- hit rate and first-lick aim by position, engaged-only
POSN = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra", "far_L": "far_ipsi",
        "far_center": "far_mid", "far_R": "far_contra"}
# hit-rate trial set = the manuscript's (Fig 3C epoch_1b camera working period: `trial_classes.camera_period_trials`,
# first cue .. last camera lick minus the stopped runs; hit = camera class contact); per-trial flags precomputed in
# camera_working.csv (s0_camera_working.py)
CW = pd.read_csv(OUT / "camera_working.csv")
Tw = T.merge(CW, on=["label", "trial_id"], how="inner")
Tw = Tw[Tw.work].copy()
Tw["role"] = [role_of.get((l, int(t)), "none") for l, t in zip(Tw.label, Tw.trial_id)]
for v, d in REFIT_ROLE.items():
    Tw[f"role_{v}"] = [d.get((l, int(t)), "none") for l, t in zip(Tw.label, Tw.trial_id)]
Tw["hit"] = Tw.contact.astype(float)
Tw["pos"] = Tw.position.map(POSN)
rows = []
for m, N in MATCH.items():
    for tset, sel in (("all", None), ("engaged", ("role", "engaged")),
                      *[(f"engaged_{v}", (f"role_{v}", "engaged")) for v in REFIT_ROLE]):
        D_ = Tw[~(Tw.t_min > N)]
        if sel:
            D_ = D_[D_[sel[0]] == sel[1]]
        g = D_.groupby(["label", "pos"]).hit.agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= 5].rename(columns={"mean": "hit_rate", "size": "n"})
        g["match"], g["trials"] = m, tset
        rows.append(g)
H = base(pd.concat(rows, ignore_index=True))
W("sessions_hit_by_position.csv", H)

# first-lick aim by position needs the run's ITI-lick predictions (`iti_recovery_measure.aim_licks`); 11-22: a run
# without them (the adopted four-camera run before its iti_licks re-run) skips this table instead of failing
def aim_table():
    import iti_recovery_measure as rm  # noqa: E402
    p = rm.aim_licks()
    p = p[p["first"]].copy()
    # 11-22 (plan item 3): the selected 3-D source / run tag (WIDEFIELD_POSE3D_SOURCE / _TAG); was the literal two-camera folder
    from wfield_local.pose3d import pose3d_dir as _pose3d_dir  # noqa: E402
    L3 = pd.read_csv(_pose3d_dir() / _kin_name() / "licks.csv.gz",
                     usecols=["trial_id", "side_ml", "out_ap", "down_dv", "session"])
    L3["label"] = L3.session.str[:5] + L3.session.str[-4:]
    L3["k"], L3["s"] = L3.side_ml.round(3), np.hypot(L3.out_ap, L3.down_dv).round(3)
    p["k"], p["s"] = p.act_lat.round(3), p.act_sag.round(3)
    p = p.merge(L3.drop_duplicates(["label", "k", "s"])[["label", "k", "s", "trial_id", "session"]], on=["label", "k", "s"], how="left")
    print(f"aim first licks matched to trial ids: {p.trial_id.notna().sum()} of {len(p)}", flush=True)
    p = p.dropna(subset=["trial_id"])
    p["label8"] = p.session
    p["role"] = [role_of.get((l, int(t)), "none") for l, t in zip(p.label8, p.trial_id)]
    for v, d in REFIT_ROLE.items():
        p[f"role_{v}"] = [d.get((l, int(t)), "none") for l, t in zip(p.label8, p.trial_id)]
    p["t_min"] = [tmin_of.get((l, int(t)), np.nan) for l, t in zip(p.label8, p.trial_id)]
    rows = []
    for m, N in MATCH.items():
        for tset, sel in (("all", None), ("engaged", ("role", "engaged")),
                          *[(f"engaged_{v}", (f"role_{v}", "engaged")) for v in REFIT_ROLE]):
            D_ = p[~(p.t_min > N)]
            if sel:
                D_ = D_[D_[sel[0]] == sel[1]]
            g = D_.groupby(["label8", "pos"]).off.agg(["mean", "size"]).reset_index()
            g = g[g["size"] >= rm.MIN_LICKS["first"]].rename(columns={"label8": "label", "mean": "aim_off", "size": "n"})
            g["match"], g["trials"] = m, tset
            rows.append(g)
    A = base(pd.concat(rows, ignore_index=True))
    A = A[A.epoch.notna()]
    W("sessions_aim_by_position.csv", A)


try:
    aim_table()
except FileNotFoundError as ex:
    print(f"sessions_aim_by_position.csv SKIPPED (input missing: {ex.filename})", flush=True)

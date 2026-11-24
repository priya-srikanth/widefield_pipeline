"""Step 5: statistics, correlations and figures from the s4 session tables (six-position cohort, PS92-PS95).

Statistics (manuscript scalar rule, as c2_treadmill_cue_aligned.py): `orofacial_stats.session_change_tests` = per
session change from the animal's own August-pre mean; session Mann-Whitney U (post vs pre changes) + paired
animal-weighted nested bootstrap (`analysis_kit.boot_delta`, 2,000 draws, seed 0); ONE Benjamini-Hochberg family per
measure family (fdr_scope="all" within the call; families listed in tests_ALL.csv ``family``); star = q < 0.05 AND
95 % CI excludes 0 (``agree == "both"``). Primary time match = first95; first74 and unmatched reported alongside.
Usage: WF_REPO=... MC=... python s5_stats_figures.py <out_dir> [--no-tim]
--no-tim (11-22, manuscript v26: TIM removed from the paper; s2 not run, s4 run with --no-tim): every TIM family, the
composite general-movement index (needs TIM and the treadmill rates), the TIM state contrasts and the TIM LMM are
skipped; the main figure is drawn without its TIM rows. Everything else is unchanged.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline")))
sys.path.insert(0, str(Path(os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline"))) / "scripts"))
sys.path.insert(0, str(Path(os.environ["MC"]) / "figures"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from wfield_local import orofacial_stats as st  # noqa: E402
from wfield_local import behavior_states as B  # noqa: E402

OUT = Path(sys.argv[1])
NO_TIM = "--no-tim" in sys.argv[2:]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trial_scaffold  # noqa: E402  (collector trials, or the 11-22 fallback for a run without them)
EPS = ["pre", "acute", "subacute", "chronic"]
POST = EPS[1:]
POS6 = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
PRIMARY = "first95"
WRITTEN = []


def W(name, df, idx=False):
    f = OUT / name
    if f.exists():
        sys.exit(f"refusing to overwrite {f}")
    df.to_csv(f, index=idx)
    WRITTEN.append(name)


def rd(name):
    return pd.read_csv(OUT / name, dtype={"date": str})


ALL = []


def tests(S, metrics, family, match, extra=None):
    metrics = [m for m in metrics if m in S and S[m].notna().sum() > 3]
    if not metrics:
        return pd.DataFrame()
    r = st.session_change_tests(S, metrics, fdr_scope="all")
    r["family"], r["match"] = family, match
    for k, v in (extra or {}).items():
        r[k] = v
    r["star"] = (r.q_fdr < 0.05) & ((r.boot_lo > 0) | (r.boot_hi < 0))
    ALL.append(r)
    return r


E = rd("sessions_engagement.csv")
TP = None if NO_TIM else rd("sessions_tim_pipeline.csv")
TN = None if NO_TIM else rd("sessions_tim_new.csv")
M = rd("sessions_motion.csv")
TRD = None if NO_TIM else rd("sessions_treadmill_rates.csv")
H = rd("sessions_hit_by_position.csv")
A = rd("sessions_aim_by_position.csv") if (OUT / "sessions_aim_by_position.csv").exists() else None   # 11-22: s4 skips it without ITI-lick preds
for D_ in (E, TP, TN, M, TRD):
    if D_ is not None:
        D_["pos"] = "all"

ROLES = list(B.ROLES)
TIM_DLC = [f"timDLC_{p}_{q}" for p in ("precue", "response") for q in ("tongue", "jaw", "nose")]
TIM_DLC_ITI = [f"timDLC_iti_{q}" for q in ("tongue", "jaw", "nose")]
TIM_FR = [f"timFR_{p}_face" for p in ("precue", "response", "iti")]
TIM_PIPE = [f"timpipe_{p}_{q}" for p in ("precue", "response") for q in ("tongue", "jaw", "nose")]
ME = [f"me_kp_{w}" for w in ("precue", "response", "iti")] + [f"me_fr_all_{w}" for w in ("precue", "response", "iti")]
ME_HALF = [f"me_fr_{h}_{w}" for h in ("contra", "ipsi") for w in ("precue", "response", "iti")]
STILL = [f"still_{s}_{f}" for s in ("kp", "fr_all", "tread") for f in ("iti", "trial", "session")]
RUN = ["run_mean_speed", "run_bout_speed", "run_bouts_per_min", "run_frac", "run_bout_dur", "run_iti_mean_speed",
       "run_iti_frac"]

# ------------------------------------------------------------------ families
for m in ("first95", "first74", "all"):
    e = E[(E.match == m)]
    tests(e[e.model == "production"], ROLES, "F1 engagement occupancy (production K5)", m)
    for mod in ("refit_ipsimid", "refit_init"):
        if (e.model == mod).any():
            tests(e[e.model == mod], ROLES, f"F1r engagement occupancy ({mod})", m)
    if not NO_TIM:
        tn = TN[(TN.match == m) & (TN.state == "all") & (TN.trials == "working")]
        tests(tn, TIM_DLC, "F2 TIM DLC (new windows; pre-cue E-G, response H-J)", m)
        tests(tn, TIM_DLC_ITI, "F2b TIM DLC ITI (docked)", m)
        tests(tn, TIM_FR, "F3 TIM FaceRhythm (new windows)", m)
        tp = TP[(TP.match == m) & (TP.state == "all")]
        tests(tp, TIM_PIPE, "F2c TIM DLC pipeline-default periods (check)", m)
    mm = M[(M.match == m) & (M.state == "all")]
    tests(mm, ME, "F4 motion energy (keypoint + FaceRhythm whole face)", m)
    tests(mm, ME_HALF, "F4b FaceRhythm ME face halves", m)
    tests(mm, STILL, "F5 stillness", m)
    tests(mm, RUN, "F6 running", m)
# state splits (primary match only)
for rs in ("engaged", "not_engaged", "disengaged"):
    if not NO_TIM:
        tn = TN[(TN.match == PRIMARY) & (TN.state == rs) & (TN.trials == "alltrials")]
        tests(tn, TIM_DLC + TIM_FR, f"F7 TIM within {rs} trials (all trial classes)", PRIMARY, {"state": rs})
    mm = M[(M.match == PRIMARY) & (M.state == rs)]
    tests(mm, ME + [s for s in STILL if not s.endswith("session")] + ["run_iti_mean_speed", "run_iti_frac"],
          f"F8 motion within {rs} trials", PRIMARY, {"state": rs})
def tim_and_composite():
    global CZ, COMP
    tn = TN[(TN.match == PRIMARY) & (TN.state == "all") & (TN.trials == "alltrials")]
    tests(tn, TIM_DLC + TIM_FR, "F7 TIM all trial classes (reference for F7)", PRIMARY, {"state": "all"})
    tn = TN[(TN.match == PRIMARY) & (TN.state == "engaged") & (TN.trials == "working")]
    tests(tn, TIM_DLC + TIM_FR, "F7w TIM engaged working trials", PRIMARY, {"state": "engaged"})

    # ------------------------------------------------------------------ composite
    def zpre(S, cols):
        out = S[["label", "animal", "epoch", "date"]].copy()
        for c in cols:
            pre = S[S.epoch == "pre"].groupby("animal")[c].agg(["mean", "std"])
            out[c] = (S[c] - S.animal.map(pre["mean"])) / S.animal.map(pre["std"])
        return out


    COMP = {"DLC": ["timDLC_precue_tongue", "timDLC_precue_jaw", "timDLC_precue_nose"],
            "FaceRhythm": ["timFR_precue_face", "me_fr_all_iti"],
            "treadmill": ["walk_control", "push_control_rate"]}
    comp_rows = []
    for m in ("first95", "first74", "all"):
        tn = TN[(TN.match == m) & (TN.state == "all") & (TN.trials == "working")][["label", "animal", "epoch", "date"] + COMP["DLC"] + ["timFR_precue_face"]]
        mm = M[(M.match == m) & (M.state == "all")][["label", "me_fr_all_iti"]]
        tr = TRD[TRD.match == m][["label", "walk_control", "push_control_rate"]]
        S = tn.merge(mm, on="label", how="outer").merge(tr, on="label", how="outer")
        S["animal"] = S.label.str.split("_").str[0]; S["epoch"] = S.label.map(B.session_epoch); S["date"] = S.label.str.split("_").str[1]
        Z = zpre(S, sum(COMP.values(), []))
        for mod, cols in COMP.items():
            Z[f"mod_{mod}"] = Z[cols].mean(1)
        Z["composite"] = Z[[f"mod_{k}" for k in COMP]].mean(1)
        Z["match"] = m
        comp_rows.append(Z)
    CZ = pd.concat(comp_rows, ignore_index=True)
    CZ["pos"] = "all"
    W("sessions_composite_z.csv", CZ)
    for m in ("first95", "first74", "all"):
        z = CZ[CZ.match == m]
        tests(z, ["composite"], "F9 composite general-movement index", m)
        tests(z, [f"mod_{k}" for k in COMP] + sum(COMP.values(), []), "F9s composite modalities + components (supplement)", m)
    z = CZ[CZ.match == PRIMARY]
    cr = []
    for subset, zz in (("all sessions", z), ("post sessions", z[z.epoch != "pre"])):
        for a_, b_ in (("DLC", "FaceRhythm"), ("DLC", "treadmill"), ("FaceRhythm", "treadmill")):
            x, y = zz[f"mod_{a_}"], zz[f"mod_{b_}"]
            ok = x.notna() & y.notna()
            r, p = spearmanr(x[ok], y[ok])
            # within-animal (each animal's sessions centered on its own mean)
            xc = (x - x.groupby(zz.animal).transform("mean"))[ok]; yc = (y - y.groupby(zz.animal).transform("mean"))[ok]
            rw, pw = spearmanr(xc, yc)
            cr.append({"subset": subset, "pair": f"{a_} vs {b_}", "n_sessions": int(ok.sum()), "spearman_r": r, "p": p,
                       "within_animal_r": rw, "within_animal_p": pw})
    W("composite_modality_correlations.csv", pd.DataFrame(cr))


CZ = None
if not NO_TIM:
    tim_and_composite()

# ------------------------------------------------------------------ hit rate / aim by position: all vs engaged-only
cmp_rows = []
for meas, D_, col in (("hit_rate", H, "hit_rate"), *((("aim_off", A, "aim_off"),) if A is not None else ())):
    for m in ("first95", "all"):
        for tset in sorted(D_.trials.unique()):
            d = D_[(D_.match == m) & (D_.trials == tset)].copy()
            r = tests(d, [col], f"F10 {meas} by position ({tset} trials)", m, {"trials": tset})
            cmp_rows.append(r)
CMP = pd.concat(cmp_rows, ignore_index=True)
piv = CMP.pivot_table(index=["metric", "match", "pos", "epoch"], columns="trials", values="boot_change").reset_index()
for c in [c for c in piv.columns if str(c).startswith("engaged")]:
    piv[f"shrink_pct_{c}"] = 100 * (1 - piv[c] / piv["all"])
W("engaged_only_vs_all_by_position.csv", piv)
pat = []
for (meas, m, tset, e), g in CMP.groupby(["metric", "match", "trials", "epoch"]):
    g = g.set_index("pos").reindex(POS6)
    v = g.boot_change
    worst = v.idxmin() if meas == "hit_rate" else v.abs().idxmax()
    ipsi = v[["close_ipsi", "far_ipsi"]].abs().mean(); contra = v[["close_contra", "far_contra"]].abs().mean()
    pat.append({"metric": meas, "match": m, "trials": tset, "epoch": e, "most_affected": worst,
                "far_contra_change": v["far_contra"], "far_contra_star": bool(g.star["far_contra"]),
                "mean_abs_ipsi": ipsi, "mean_abs_contra": contra, "far_minus_close_contra_abs": abs(v["far_contra"]) - abs(v["close_contra"]),
                "n_star": int(g.star.sum()), "stars": ",".join(g.index[g.star.fillna(False).astype(bool)])})
W("engaged_only_pattern_by_epoch.csv", pd.DataFrame(pat))

# ------------------------------------------------------------------ per-animal time course: disengagement vs far-contra hit
# 11-22 (plan item 3): the adopted run's table (was a literal v36 path; only the session -> day map is read)
from wfield_local import orofacial_outputs as _oo  # noqa: E402
_daysf = _oo.encoding_root() / "iti_licks" / "aim_by_spout_sessions.csv"
if _daysf.exists():
    days = pd.read_csv(_daysf)[["label", "day"]].drop_duplicates("label")
    days["label8"] = days.label.str[:5] + "2026" + days.label.str[-4:]
else:                     # 11-22: run without iti_licks tables -> the same day count from `epochs.days_since_stroke`
    from wfield_local import epochs as _ep  # noqa: E402
    days = pd.DataFrame({"label8": sorted(E.label.unique())})
    days["day"] = [_ep.days_since_stroke(lab[:5] + lab[-4:]) for lab in days.label8]
e = E[(E.match == PRIMARY) & (E.model == "production")][["label", "animal", "epoch", "disengaged", "engaged"]]
h = H[(H.match == PRIMARY) & (H.trials == "all") & (H.pos == "far_contra")][["label", "hit_rate"]]
tc_ = e.merge(h, on="label", how="left").merge(days[["label8", "day"]], left_on="label", right_on="label8", how="left")
tcr, tcs = [], []
for a, g in tc_.groupby("animal"):
    pre = g[g.epoch == "pre"]
    g = g.assign(d_dis=g.disengaged - pre.disengaged.mean(), d_hit=g.hit_rate - pre.hit_rate.mean())
    tcs.append(g)
    post = g[g.epoch != "pre"].sort_values("day")
    r, p = spearmanr(post.d_dis, post.d_hit, nan_policy="omit")
    def back(col, sd):
        thr = 2 * sd
        bad = post[col].abs() > thr  # noqa: B023
        if not bad.any():
            return np.nan
        last_bad = post.day[bad].max()  # noqa: B023
        nxt = post.day[post.day > last_bad]  # noqa: B023
        return nxt.min() if len(nxt) else np.inf
    tcr.append({"animal": a, "n_post": len(post), "spearman_r_ddis_vs_dhit": r, "p": p,
                "disengaged_peak_day": post.day[post.d_dis.idxmax()] if len(post) else np.nan,
                "hit_trough_day": post.day[post.d_hit.idxmin()] if len(post) else np.nan,
                "disengaged_back_within_2sd_from_day": back("d_dis", pre.disengaged.std()),
                "farcontra_hit_back_within_2sd_from_day": back("d_hit", pre.hit_rate.std()),
                "acute_d_dis": post[post.epoch == "acute"].d_dis.mean(), "chronic_d_dis": post[post.epoch == "chronic"].d_dis.mean(),
                "acute_d_hit": post[post.epoch == "acute"].d_hit.mean(), "chronic_d_hit": post[post.epoch == "chronic"].d_hit.mean()})
W("per_animal_timecourse_disengagement_vs_farcontra_hit.csv", pd.DataFrame(tcr))
TCS = pd.concat(tcs, ignore_index=True)
W("per_animal_timecourse_sessions.csv", TCS)

# ------------------------------------------------------------------ 10a: disengaged trials by spout
G = B.glmhmm_table()
roles = B.glmhmm_roles(G)
pc = B._pcols(G)
G["role"] = [roles[int(pc[i][7:])] for i in G[pc].to_numpy().argmax(1)]
G = B.add_session_time(G, trial_scaffold.trials(OUT))
G = G[~(G.t_min > 95)]
POSN = {"close_L": "close_ipsi", "close_center": "close_mid", "close_R": "close_contra", "far_L": "far_ipsi",
        "far_center": "far_mid", "far_R": "far_contra"}
G["pos6"] = G.position.map(POSN)
dz = G.assign(dis=G.role.eq("disengaged").astype(float), noteng=G.role.ne("engaged").astype(float)) \
      .groupby(["label", "animal", "epoch", "pos6"])[["dis", "noteng"]].mean().reset_index().rename(columns={"pos6": "pos"})
dz["date"] = dz.label.str.split("_").str[1]
W("disengaged_by_spout_sessions.csv", dz)
tests(dz, ["dis", "noteng"], "F11 P(disengaged / not engaged | spout) by spout (10a)", PRIMARY)
w = dz.pivot_table(index=["label", "animal", "epoch", "date"], columns="pos", values=["dis", "noteng"]).reset_index()
w.columns = ["_".join([c for c in col if c]) for col in w.columns]
for v in ("dis", "noteng"):
    w[f"{v}_farcontra_minus_ipsi"] = w[f"{v}_far_contra"] - w[[f"{v}_close_ipsi", f"{v}_far_ipsi"]].mean(1)
    w[f"{v}_contra_minus_ipsi"] = w[[f"{v}_close_contra", f"{v}_far_contra"]].mean(1) - w[[f"{v}_close_ipsi", f"{v}_far_ipsi"]].mean(1)
w["pos"] = "all"
tests(w, ["dis_farcontra_minus_ipsi", "dis_contra_minus_ipsi", "noteng_farcontra_minus_ipsi", "noteng_contra_minus_ipsi"],
      "F11c disengagement contra-minus-ipsi contrast (10a)", PRIMARY)
# role composition of trials by spout and epoch (pooled counts)
comp = G.groupby(["epoch", "pos6"]).role.value_counts(normalize=True).unstack(fill_value=0).reset_index()
W("glmhmm_role_share_by_epoch_spout.csv", comp)
# emission profile per state per spout (production model)
P = B.glmhmm_params()
em = []
from wfield_local import glm_hmm as gh  # noqa: E402
for p in gh.POSITIONS:
    o = B.glmhmm_emission(P, p)["outcome"]
    for k in range(o.shape[0]):
        em.append({"state": k, "role_merged": roles.get(0 if k in (0, 1) else k), "spout": POSN[p],
                   "p_attempt": o[k, 0], "p_jaw_only": o[k, 1], "p_no_response": o[k, 2]})
W("glmhmm_emission_by_state_spout.csv", pd.DataFrame(em))
# agreement of refits with production (trial-level engaged call)
agr = []
for v in ("ipsimid", "init"):
    f = OUT / f"glmhmm_refit_{v}_K5.csv.gz"
    if f.exists():
        R = pd.read_csv(f, usecols=["label", "trial_id", "role"]).merge(G[["label", "trial_id", "role", "epoch", "pos6"]],
                                                                        on=["label", "trial_id"], suffixes=("_refit", "_prod"))
        for (ep, ps), g in R.groupby(["epoch", "pos6"]):
            agr.append({"variant": v, "epoch": ep, "pos": ps, "n": len(g),
                        "agree_engaged_call": np.mean((g.role_refit == "engaged") == (g.role_prod == "engaged")),
                        "engaged_prod": np.mean(g.role_prod == "engaged"), "engaged_refit": np.mean(g.role_refit == "engaged"),
                        "disengaged_prod": np.mean(g.role_prod == "disengaged"), "disengaged_refit": np.mean(g.role_refit == "disengaged")})
W("glmhmm_refit_agreement.csv", pd.DataFrame(agr))

# ------------------------------------------------------------------ 10b: TIM / ME by state before stroke; TIM ~ epoch + state
def nested_mean(by_animal, rng, nb=2000):
    a = list(by_animal)
    pt = np.mean([np.mean(by_animal[x]) for x in a])
    bs = []
    for _ in range(nb):
        pick = rng.choice(a, len(a))
        bs.append(np.mean([np.mean(rng.choice(by_animal[x], len(by_animal[x]))) for x in pick]))
    return pt, *np.percentile(bs, [2.5, 97.5])


rng = np.random.default_rng(0)
yr = []
for src, D_, cols in ((() if NO_TIM else (("TIM", TN[(TN.match == PRIMARY) & (TN.trials == "alltrials")],
                                             TIM_DLC + TIM_DLC_ITI + TIM_FR),))
                      + (("ME", M[(M.match == PRIMARY)], ME),)):
    for contrast in ("disengaged", "not_engaged"):
        a_ = D_[D_.state == contrast].set_index("label"); b_ = D_[D_.state == "engaged"].set_index("label")
        for c in cols:
            d = (a_[c] - b_[c]).dropna()
            for grp, labs in (("pre", [l for l in d.index if B.session_epoch(l) == "pre"]),
                              ("post", [l for l in d.index if B.session_epoch(l) != "pre"])):
                dd = d[labs]
                by = {an: dd[[l for l in dd.index if l.startswith(an)]].tolist() for an in sorted({l[:4] for l in dd.index})}
                by = {k: v for k, v in by.items() if v}
                if len(by) < 2:
                    continue
                pt, lo, hi = nested_mean(by, rng)
                yr.append({"measure": src, "metric": c, "contrast": f"{contrast} minus engaged", "sessions": grp,
                           "n_sessions": len(dd), "n_animals": len(by), "animal_weighted_mean": pt, "lo": lo, "hi": hi,
                           "ci_excludes_0": bool(lo > 0 or hi < 0), "frac_sessions_positive": float(np.mean(dd > 0)),
                           "engaged_level_mean": float(b_.loc[labs, c].mean())})
W("yin_state_contrasts.csv", pd.DataFrame(yr))
if not NO_TIM:
    # LMM: session x role cells, TIM ~ epoch (+ role), animal random intercept
    import statsmodels.formula.api as smf  # noqa: E402
    lm = []
    cells = TN[(TN.match == PRIMARY) & (TN.trials == "alltrials") & TN.state.isin(ROLES)].copy()
    for c in TIM_DLC + TIM_FR:
        d = cells[["animal", "epoch", "state", c]].dropna().rename(columns={c: "y"})
        for form in ("y ~ C(epoch, Treatment('pre'))", "y ~ C(epoch, Treatment('pre')) + C(state, Treatment('engaged'))"):
            try:
                f = smf.mixedlm(form, d, groups=d.animal).fit(method="lbfgs")
            except Exception as ex:  # noqa: BLE001
                lm.append({"metric": c, "model": form, "error": str(ex)}); continue
            for e in POST:
                t_ = f"C(epoch, Treatment('pre'))[T.{e}]"
                lm.append({"metric": c, "model": "epoch + state" if "state" in form else "epoch only", "epoch": e,
                           "coef": f.params.get(t_), "se": f.bse.get(t_), "p": f.pvalues.get(t_), "n_cells": len(d)})
    W("tim_lmm_epoch_with_state_covariate.csv", pd.DataFrame(lm))

# ME vs P(engaged)
mvp = M[(M.match == PRIMARY) & (M.state == "all")][["label", "me_kp_iti", "me_fr_all_iti", "me_kp_precue", "me_fr_all_precue"]] \
    .merge(E[(E.match == PRIMARY) & (E.model == "production")][["label", "animal", "epoch", "engaged"]], on="label")
mv = []
for c in ("me_kp_iti", "me_fr_all_iti", "me_kp_precue", "me_fr_all_precue"):
    for grp, d in (("pre", mvp[mvp.epoch == "pre"]), ("post", mvp[mvp.epoch != "pre"])):
        ok = d[c].notna()
        r, p = spearmanr(d.engaged[ok], d[c][ok])
        xc = (d.engaged - d.groupby("animal").engaged.transform("mean"))[ok]; yc = (d[c] - d.groupby("animal")[c].transform("mean"))[ok]
        rw, pw = spearmanr(xc, yc)
        mv.append({"metric": c, "sessions": grp, "n": int(ok.sum()), "spearman_r": r, "p": p, "within_animal_r": rw, "within_animal_p": pw})
W("me_vs_pengaged.csv", pd.DataFrame(mv))

TALL = pd.concat(ALL, ignore_index=True)
W("tests_ALL.csv", TALL)
prim = TALL[TALL.match == PRIMARY]
print(prim[["family", "metric", "pos", "epoch", "n_sessions", "boot_change", "boot_lo", "boot_hi", "q_fdr", "star"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3g}"))

# ------------------------------------------------------------------ figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import vls_style as S  # noqa: E402
import c2_panels as CP  # noqa: E402
S.apply()


def delta_panel(ax, Sx, metric, family, ylab, match=PRIMARY, extra=None):
    d = Sx.copy()
    base_ = d[d.epoch == "pre"].groupby("animal")[metric].mean()
    d["d"] = d[metric] - d.animal.map(base_)
    tt = TALL[(TALL.metric == metric) & (TALL.family == family) & (TALL.match == match)]
    for k, v in (extra or {}).items():
        tt = tt[tt[k] == v]

    def tg(p, e):
        r = tt[tt.epoch == e]
        if r.empty:
            return None
        r = r.iloc[0]
        return r.boot_change, r.boot_lo, r.boot_hi, (2 if r.star else 0)
    S.zero_line(ax)
    w = CP.grouped(ax, ["all"], lambda p, e: d[d.epoch == e].d.dropna().values, tg)
    ax.set_xticks([]); ax.set_xlim(-0.5, 0.5); ax.set_ylabel(ylab, fontsize=6.2); S.headroom(ax, 0.3)
    for j, e in enumerate(EPS):
        r = tg("all", e)
        if r and r[3]:
            CP.star(ax, (j - 1.5) * w, j, S.EP[e], level=r[3])


def letter(ax, s):
    ax.text(-0.42, 1.08, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


e95 = E[(E.match == PRIMARY) & (E.model == "production")]
m95 = M[(M.match == PRIMARY) & (M.state == "all")]
ME_ROW = (("me_kp_iti", "F4 motion energy (keypoint + FaceRhythm whole face)", "keypoint ME, ITI\n(Δ z)"),
          ("me_fr_all_iti", "F4 motion energy (keypoint + FaceRhythm whole face)", "FaceRhythm ME, ITI\n(Δ z)"),
          ("run_mean_speed", "F6 running", "forward speed, session\n(Δ mm/s)"),
          ("run_bouts_per_min", "F6 running", "running bouts / min\n(Δ)"),
          ("still_kp_iti", "F5 stillness", "stillness, ITI keypoint\n(Δ fraction)"))
if NO_TIM:
    # v26: engagement + motion energy / running / stillness only (no TIM rows, no composite)
    fig, axs = plt.subplots(2, 5, figsize=(7.0, 3.8))
    L = iter("ABCDEFGHIJ")
    for i, r in enumerate(ROLES):
        delta_panel(axs[0, i], e95, r, "F1 engagement occupancy (production K5)", f"{r} occupancy\n(Δ, first 95 min)"); letter(axs[0, i], next(L))
    ax = axs[0, 3]; letter(ax, next(L))
    for e in EPS:
        d = mvp[mvp.epoch == e]
        ax.scatter(d.engaged, d.me_kp_iti, s=5, color=S.EP[e], alpha=0.75, lw=0, label=e)
    ax.set_xlabel("P(engaged)", fontsize=6.2); ax.set_ylabel("keypoint ME, ITI\n(z vs own pre)", fontsize=6.2)
    CP.epoch_legend(axs[0, 3], loc="upper left", ncol=1)
    axs[0, 4].axis("off")
    for i, (mt, fam, yl) in enumerate(ME_ROW):
        delta_panel(axs[1, i], m95, mt, fam, yl); letter(axs[1, i], next(L))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.06, top=0.95, wspace=0.95, hspace=0.45)
    S.save(fig, str(OUT / "C2_engagement_movement"))
    WRITTEN.append("C2_engagement_movement.(pdf|png|svg)")
else:
    tn95 = TN[(TN.match == PRIMARY) & (TN.state == "all") & (TN.trials == "working")]
    m95 = M[(M.match == PRIMARY) & (M.state == "all")]
    cz95 = CZ[CZ.match == PRIMARY]
    fig, axs = plt.subplots(4, 5, figsize=(7.0, 7.4))
    L = iter("ABCDEFGHIJKLMNOPQRST")
    for i, r in enumerate(ROLES):
        delta_panel(axs[0, i], e95, r, "F1 engagement occupancy (production K5)", f"{r} occupancy\n(Δ, first 95 min)"); letter(axs[0, i], next(L))
    ax = axs[0, 3]; letter(ax, next(L))
    for e in EPS:
        d = mvp[mvp.epoch == e]
        ax.scatter(d.engaged, d.me_kp_iti, s=5, color=S.EP[e], alpha=0.75, lw=0, label=e)
    ax.set_xlabel("P(engaged)", fontsize=6.2); ax.set_ylabel("keypoint ME, ITI\n(z vs own pre)", fontsize=6.2)
    ax = axs[0, 4]; letter(ax, next(L)); delta_panel(ax, cz95, "composite", "F9 composite general-movement index",
                                                     "general-movement index\n(Δ z, 3 modalities)")
    CP.epoch_legend(axs[0, 3], loc="upper left", ncol=1)
    for j, (row, per) in enumerate(((1, "precue"), (2, "response"))):  # noqa: B007
        for i, q in enumerate(("tongue", "jaw", "nose")):
            delta_panel(axs[row, i], tn95, f"timDLC_{per}_{q}", "F2 TIM DLC (new windows; pre-cue E-G, response H-J)",
                        f"TIM {q}, {'pre-cue' if per == 'precue' else 'response'}\n(Δ px)")
            letter(axs[row, i], next(L))
        delta_panel(axs[row, 3], tn95, f"timFR_{per}_face", "F3 TIM FaceRhythm (new windows)",
                    f"FaceRhythm TIM, {'pre-cue' if per == 'precue' else 'response'}\n(Δ px, 20 point PCs)")
        letter(axs[row, 3], next(L))
    delta_panel(axs[1, 4], tn95, "timDLC_iti_jaw", "F2b TIM DLC ITI (docked)", "TIM jaw, ITI docked\n(Δ px)"); letter(axs[1, 4], next(L))
    delta_panel(axs[2, 4], tn95, "timFR_iti_face", "F3 TIM FaceRhythm (new windows)", "FaceRhythm TIM, ITI\n(Δ px)"); letter(axs[2, 4], next(L))
    for i, (mt, fam, yl) in enumerate((("me_kp_iti", "F4 motion energy (keypoint + FaceRhythm whole face)", "keypoint ME, ITI\n(Δ z)"),
                                       ("me_fr_all_iti", "F4 motion energy (keypoint + FaceRhythm whole face)", "FaceRhythm ME, ITI\n(Δ z)"),
                                       ("run_mean_speed", "F6 running", "forward speed, session\n(Δ mm/s)"),
                                       ("run_bouts_per_min", "F6 running", "running bouts / min\n(Δ)"),
                                       ("still_kp_iti", "F5 stillness", "stillness, ITI keypoint\n(Δ fraction)"))):
        delta_panel(axs[3, i], m95, mt, fam, yl); letter(axs[3, i], next(L))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.04, top=0.97, wspace=0.95, hspace=0.45)
    S.save(fig, str(OUT / "C2_engagement_tim_movement"))
    WRITTEN.append("C2_engagement_tim_movement.(pdf|png|svg)")

    # supplement: composite components
    cols = [f"mod_{k}" for k in COMP] + sum(COMP.values(), [])
    fig, axs = plt.subplots(2, 5, figsize=(7.0, 3.6))
    for ax, c in zip(axs.flat, cols):
        delta_panel(ax, cz95, c, "F9s composite modalities + components (supplement)", c.replace("_", " ") + "\n(Δ z)")
    for ax in axs.flat[len(cols):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.05, top=0.95, wspace=0.95, hspace=0.45)
    S.save(fig, str(OUT / "C2_composite_components"))
    WRITTEN.append("C2_composite_components.(pdf|png|svg)")

# supplement: engaged-only hit rate / aim by position
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.0))
for col, (meas, D_, yl) in enumerate((("hit_rate", H, "hit rate (Δ)"),
                                      *((("aim_off", A, "first-lick aim offset (Δ deg, + = contra)"),) if A is not None else ()))):
    for row, tset in enumerate(("all", "engaged")):
        ax = axs[row, col]
        d = D_[(D_.match == PRIMARY) & (D_.trials == tset)].copy()
        base_ = d[d.epoch == "pre"].groupby(["animal", "pos"])[meas].mean()
        d["d"] = d[meas].to_numpy() - base_.reindex(pd.MultiIndex.from_frame(d[["animal", "pos"]])).to_numpy()
        tt = TALL[(TALL.metric == meas) & (TALL.match == PRIMARY) & (TALL.trials == tset) & TALL.family.str.startswith("F10")]

        def tg(p, e, tt=tt):
            r = tt[(tt.pos == p) & (tt.epoch == e)]
            if r.empty:
                return None
            r = r.iloc[0]
            return r.boot_change, r.boot_lo, r.boot_hi, (2 if r.star else 0)
        S.zero_line(ax)
        w = CP.grouped(ax, POS6, lambda p, e, d=d: d[(d.pos == p) & (d.epoch == e)].d.dropna().values, tg)
        S.headroom(ax, 0.3)
        for i, p in enumerate(POS6):
            for j, e in enumerate(EPS):
                r = tg(p, e)
                if r and r[3]:
                    CP.star(ax, i + (j - 1.5) * w, j, S.EP[e], level=r[3])
        ax.set_xticks(range(6)); ax.set_xticklabels([p.replace("_", "\n") for p in POS6], fontsize=6)
        ax.set_ylabel(f"{yl}\n{'all working trials' if tset == 'all' else 'GLM-HMM engaged trials only'}", fontsize=6.2)
CP.epoch_legend(axs[0, 0], loc="lower left", ncol=4)
fig.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.97, wspace=0.3, hspace=0.45)
S.save(fig, str(OUT / "C2_engaged_only_hit_aim_by_position"))
WRITTEN.append("C2_engaged_only_hit_aim_by_position.(pdf|png|svg)")

# supplement: motion energy / stillness / running components
mets = [("me_kp_precue", "F4"), ("me_kp_response", "F4"), ("me_fr_all_precue", "F4"), ("me_fr_all_response", "F4"),
        ("me_fr_contra_iti", "F4b"), ("me_fr_ipsi_iti", "F4b"), ("still_kp_session", "F5"), ("still_fr_all_iti", "F5"),
        ("still_tread_iti", "F5"), ("still_tread_session", "F5"), ("run_bout_speed", "F6"), ("run_frac", "F6"),
        ("run_bout_dur", "F6"), ("run_iti_mean_speed", "F6"), ("run_iti_frac", "F6")]
fam = {f.split(" ")[0]: f for f in TALL.family.unique()}
fig, axs = plt.subplots(3, 5, figsize=(7.0, 5.4))
for ax, (c, fk) in zip(axs.flat, mets):
    delta_panel(ax, m95, c, fam[fk], c.replace("_", " ") + "\n(Δ)")
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.04, top=0.97, wspace=0.95, hspace=0.45)
S.save(fig, str(OUT / "C2_motion_components"))
WRITTEN.append("C2_motion_components.(pdf|png|svg)")
print("\nWRITTEN:", WRITTEN)

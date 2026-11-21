"""Cue-aligned treadmill before and after stroke, six-position cohort (#9-#12 = PS92-PS95). PROMPT_c2_treadmill_cue_aligned.md
(Priya 2026-11-20). Review output only; no manuscript file is touched.

Reuses widefield_pipeline `wfield_local.behavior_states` (the behavioral-states page, Fig 2.1 / 2.3) unchanged:
  - collected tables `behavior_states.load_collected()` (LP v36 `arhmm/v1/poststroke_pooled_all_s1`, collector 11-18);
  - epochs `behavior_states.session_epoch` (epoch_boundaries via `epochs.epoch_of`; June pre-stroke sessions dropped,
    August pre = reference);
  - per-session readouts `behavior_states.per_session(trials, 'lick_period')`: trials inside the lick working period
    (up to the last camera-detected lick); push = backward speed > 0.5 mm/s in the cue window (-0.25..0.75 s), control
    window -2..-1 s; walking = forward speed > 3 mm/s, 2..3.5 s after the cue (configs/defaults.yaml segmentation);
  - cue-aligned traces `traces['wheel|lick_in|<session>']` = per-session mean over working-period trials of
    (forward speed, backward speed, P(walk), P(push)), DAQ wheel, 25 Hz.
Statistics: `orofacial_stats.session_change_tests` (session MWU + paired animal-weighted nested bootstrap
`analysis_kit.boot_delta`, 2,000 draws), change from each animal's own August pre mean; ONE BH family = the two primary
measures (push cue minus control, walking 2-3.5 s) x 3 epochs, pooled over positions (fdr_scope="all"); star = q < 0.05
AND CI excludes 0. Secondary measures: same test, BH within measure (report only). By position: same two measures x six
positions x three epochs, one family (check only).

Usage: >= 49283ea2> MC=<manuscript_code> python scripts/c2_treadmill_cue_aligned.py <out_dir>

Repo copy (2026-11-20) of review/c2_treadmill_2026-11-20/c2_treadmill_cue_aligned.py; only the import path changed (+ lint-only fixes).
Outputs (all in <out_dir>, refuses to overwrite): treadmill_session_readouts.csv, treadmill_tests_primary.csv,
treadmill_tests_secondary.csv, treadmill_tests_by_position.csv, treadmill_traces_by_epoch.csv,
arhmm_run_occupancy_by_epoch.csv, C2_treadmill_cue.{pdf,png,svg}. The review run used
<out_dir> = vls_stroke_paper/review/c2_treadmill_2026-11-20/. Needs MC (for vls_style / c2_panels).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this repo (the review copy used $WF_REPO)
sys.path.insert(0, str(Path(os.environ["MC"]) / "figures"))
import numpy as np, pandas as pd  # noqa: E402
from wfield_local import behavior_states as B  # noqa: E402
from wfield_local import orofacial_stats as st  # noqa: E402
import vls_style as S  # noqa: E402
import c2_panels as CP  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(sys.argv[1])
EPS = ["pre", "acute", "subacute", "chronic"]
PRIMARY = {"push_cue_minus_control": "push rate, cue minus control\n(Δ fraction of trials)",
           "walk_cue": "walking 2-3.5 s after cue\n(Δ fraction of trials)"}
SECONDARY = ["push_rate", "push_control_rate", "push_peak", "push_latency_s", "walk_control"]
files = {k: OUT / k for k in ("treadmill_session_readouts.csv", "treadmill_tests_primary.csv", "treadmill_tests_secondary.csv",
                              "treadmill_tests_by_position.csv", "treadmill_traces_by_epoch.csv", "arhmm_run_occupancy_by_epoch.csv",
                              "C2_treadmill_cue.pdf")}
if any(p.exists() for p in files.values()):
    sys.exit("refusing to overwrite: " + ", ".join(str(p) for p in files.values() if p.exists()))

C = B.load_collected()
T = C["trials"]
T = T[T.label.map(B.session_epoch).notna()].copy()          # drop sessions outside the analysis (June pre)
T["epoch"] = T.label.map(B.session_epoch)


def sessions(trials, by_position=False):
    D = trials.assign(label=trials.label + "|" + trials.position) if by_position else trials
    s = B.per_session(D, "lick_period")
    s["pos"] = s.label.str.split("|").str[1] if by_position else "all"
    s["label"] = s.label.str.split("|").str[0]
    s["date"] = s.label.str.split("_").str[1]
    return s


Sall = sessions(T)
Sall.to_csv(files["treadmill_session_readouts.csv"], index=False)
prim = st.session_change_tests(Sall, list(PRIMARY), fdr_scope="all")
sec = st.session_change_tests(Sall, SECONDARY)
pos = st.session_change_tests(sessions(T, by_position=True), list(PRIMARY), fdr_scope="all")
for df, k in ((prim, "treadmill_tests_primary.csv"), (sec, "treadmill_tests_secondary.csv"), (pos, "treadmill_tests_by_position.csv")):
    df.to_csv(files[k], index=False)

# ---- traces by epoch: mean and SEM across sessions (n_sessions >= 3), -1..3 s
fs, (a0, _b) = C["meta"]["fs"], C["meta"]["cue_align_s"]
tr = {k.split("|")[2]: v for k, v in C["traces"].items() if k.startswith("wheel|lick_in|")}
t = a0 + (np.arange(next(iter(tr.values())).shape[1]) + 0.5) / fs
keep = (t >= -1.0) & (t <= 3.0)
rows = []
for e in EPS:
    labs = [lab for lab in tr if B.session_epoch(lab) == e]
    X = np.stack([tr[lab] for lab in labs])                 # sessions x 4 x time
    for ch, name, sign in ((0, "forward_speed", 1.0), (1, "backward_velocity", -1.0)):
        x = sign * X[:, ch, :][:, keep]
        n = np.isfinite(x).sum(0)
        rows.append(pd.DataFrame({"epoch": e, "channel": name, "t": t[keep], "mean": np.nanmean(x, 0),
                                  "sem": np.nanstd(x, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), "n_sessions": n,
                                  "n_animals": len({lab.split("_")[0] for lab in labs})}))
TR = pd.concat(rows, ignore_index=True)
TR.to_csv(files["treadmill_traces_by_epoch.csv"], index=False)

# ---- AR-HMM run-state occupancy (agreement check only; not a measure here)
U = C["usage"].copy()
U["epoch"] = U.label.map(B.session_epoch); U = U.dropna(subset=["epoch"]); U["animal"] = U.label.str.split("_").str[0]
occ = []
for col in ("all:walk", "all:walk_slow", "all:push"):
    base = U[U.epoch == "pre"].groupby("animal")[col].mean()
    d = U.assign(chg=U[col] - U.animal.map(base)).groupby(["epoch", "animal"]).chg.mean().groupby("epoch").mean()
    occ.append(d.rename(col))
pd.concat(occ, axis=1).reindex(EPS).to_csv(files["arhmm_run_occupancy_by_epoch.csv"])

# ---- figure (manuscript style)
S.apply()
fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9), gridspec_kw={"width_ratios": [1.25, 1.25, 1, 1]})
shade = {tuple(B.push_params()["control"]): "control", tuple(B.push_params()["cue"]): "cue", tuple(B.walk_window()): "walk"}
for i, (ch, ylab) in enumerate((("forward_speed", "forward speed (mm/s)"), ("backward_velocity", "backward velocity (mm/s)"))):
    a = ax[i]
    for (w0, w1), lab in shade.items():
        a.axvspan(max(w0, -1.0), min(w1, 3.0), color="0.92", lw=0, zorder=0)
        a.text((max(w0, -1.0) + min(w1, 3.0)) / 2, 1.0, lab, transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=6)
    a.axvline(0, color="0.6", lw=0.6, ls="--", zorder=0)
    for e in EPS:
        q = TR[(TR.channel == ch) & (TR.epoch == e)]
        ok = q.n_sessions.to_numpy() >= S.TRAJ_MIN_SESSIONS
        y, se = np.where(ok, q["mean"], np.nan), np.where(ok, q["sem"], np.nan)
        col = "#222222" if e == "pre" else S.EP[e]
        a.fill_between(q.t, y - se, y + se, color=col, alpha=S.TRAJ_BAND, lw=0)
        a.plot(q.t, y, color=col, lw=S.TRAJ_LW, label=e)
    a.set_xlim(-1, 3); a.set_xlabel("s from cue"); a.set_ylabel(ylab, fontsize=7)
ax[0].legend(loc="upper left", fontsize=5.8, handlelength=1.0)


def deltas(m):
    base = Sall[Sall.epoch == "pre"].groupby("animal")[m].mean()
    return Sall.assign(d=Sall[m] - Sall.animal.map(base))


for i, (m, ylab) in enumerate(PRIMARY.items()):
    a = ax[2 + i]; S.zero_line(a); D = deltas(m)
    def tg(p, e, m=m):
        r = prim[(prim.metric == m) & (prim.epoch == e)]
        if r.empty: return None
        r = r.iloc[0]
        sig = bool(r.q_fdr < 0.05 and (r.boot_lo > 0 or r.boot_hi < 0))   # MWU (family q) AND bootstrap CI excludes 0
        return r.boot_change, r.boot_lo, r.boot_hi, (2 if sig else 0)
    w = CP.grouped(a, ["all"], lambda p, e, D=D: D[D.epoch == e].d.dropna().values, tg)
    a.set_xticks([]); a.set_xlim(-0.5, 0.5); a.set_ylabel(ylab, fontsize=6.6); S.headroom(a, 0.3)
    for j, e in enumerate(EPS):
        r = tg("all", e)
        if r and r[3]: CP.star(a, (j - 1.5) * w, j, S.EP[e], level=r[3])
CP.epoch_legend(ax[3], loc="lower right", ncol=1)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.86, wspace=0.55)
S.save(fig, str(OUT / "C2_treadmill_cue"))
print(prim.to_string(index=False))

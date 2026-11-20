"""Cue-aligned treadmill before and after stroke, LENGTH-MATCHED (first 95 min), six-position cohort (#9-#12 = PS92-PS95).
The first-95 version of c2_treadmill_cue_aligned.py (Priya 2026-11-20: the treadmill analysis length-matched like
behavioral-states Fig 1.5). Review output only; no manuscript file and no existing output is touched.

Differences from c2_treadmill_cue_aligned.py:
  * trials = lick-working-period trials (`lick_period`, as before) with t_min <= 95 (minutes from the session's first
    cue, `behavior_states.COMMON_SESSION_MIN`; the 9 sessions shorter than 95 min enter whole);
  * traces RECOMPUTED (the collector's traces.npz are whole-session): treadmill forward speed and backward speed from
    `behavior_states.load_session` (the states page's loader: `arhmm_features` run_fwd / run_back, mm/s, 25 Hz, DAQ
    time), linearly interpolated at cue + t (t = -1..3 s, 40 ms grid), mean over the session's trials (sessions with
    >= `MIN_TRACE_TRIALS` trials), then mean +- SEM across sessions per epoch; backward shown as negative velocity;
    bins drawn only where >= TRAJ_MIN_SESSIONS sessions contribute; the same traces on ALL trials are written too, so
    the matched / unmatched comparison uses one loader;
  * panels C / D = treadmill_tests_primary_first95.csv (c2_treadmill_extras.py; same test: session_change_tests, BH over
    2 measures x 3 epochs, star = q < 0.05 AND bootstrap CI excludes 0), session dots from
    treadmill_session_readouts_first95.csv;
  * panel D's epoch legend placed outside the axes (no overlap with data).
Usage: WF_REPO=<widefield_pipeline> MC=<manuscript_code> python c2_treadmill_cue_aligned_first95.py <out_dir>
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline")))
sys.path.insert(0, str(Path(os.environ["MC"]) / "figures"))
import numpy as np, pandas as pd  # noqa: E402
from wfield_local import behavior_states as B  # noqa: E402
from wfield_local import trial_classes as tc  # noqa: E402
import vls_style as S  # noqa: E402
import c2_panels as CP  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(sys.argv[1])
EPS = ["pre", "acute", "subacute", "chronic"]
N = B.COMMON_SESSION_MIN
PRIMARY = {"push_cue_minus_control": "push rate, cue minus control\n(Δ fraction of trials)",
           "walk_cue": "walking 2-3.5 s after cue\n(Δ fraction of trials)"}
files = {k: OUT / k for k in ("treadmill_traces_by_epoch_first95.csv", "treadmill_traces_by_epoch_loadsession_alltime.csv",
                              "C2_treadmill_cue_first95.pdf", "C2_treadmill_cue_first95.png", "C2_treadmill_cue_first95.svg")}
if any(p.exists() for p in files.values()):
    sys.exit("refusing to overwrite: " + ", ".join(str(p) for p in files.values() if p.exists()))
Sx = pd.read_csv(OUT / "treadmill_session_readouts_first95.csv")
prim = pd.read_csv(OUT / "treadmill_tests_primary_first95.csv")

C = B.load_collected()
T = C["trials"]
T = T[T.label.map(B.session_epoch).notna() & T.lick_period].copy()
T["epoch"] = T.label.map(B.session_epoch)
tgrid = np.round(np.arange(-1.0, 3.0 + 1e-9, 0.04), 3)
per = {"first95": {}, "alltime": {}}
for lab, g in T.groupby("label"):
    TR = tc.session_trials(lab)
    cue = dict(zip(TR.trial_id.astype(int), TR.cue_daq.astype(float)))
    t, fwd, back, _z = B.load_session(lab, None)
    for key, gg in (("first95", g[~(g.t_min > N)]), ("alltime", g)):
        cs = np.array([cue.get(int(i), np.nan) for i in gg.trial_id])
        cs = cs[np.isfinite(cs)]
        if len(cs) < B.MIN_TRACE_TRIALS:
            continue
        F = np.stack([np.interp(c + tgrid, t, fwd, left=np.nan, right=np.nan) for c in cs])
        K = np.stack([np.interp(c + tgrid, t, back, left=np.nan, right=np.nan) for c in cs])
        per[key][lab] = (np.nanmean(F, 0), -np.nanmean(K, 0))
    print("done", lab, flush=True)


def by_epoch(d):
    rows = []
    for e in EPS:
        labs = [l for l in d if B.session_epoch(l) == e]
        for ch, name in ((0, "forward_speed"), (1, "backward_velocity")):
            X = np.stack([d[l][ch] for l in labs])
            n = np.isfinite(X).sum(0)
            rows.append(pd.DataFrame({"epoch": e, "channel": name, "t": tgrid, "mean": np.nanmean(X, 0),
                                      "sem": np.nanstd(X, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), "n_sessions": n,
                                      "n_animals": len({l.split("_")[0] for l in labs})}))
    return pd.concat(rows, ignore_index=True)


TR95, TRall = by_epoch(per["first95"]), by_epoch(per["alltime"])
TR95.to_csv(files["treadmill_traces_by_epoch_first95.csv"], index=False)
TRall.to_csv(files["treadmill_traces_by_epoch_loadsession_alltime.csv"], index=False)

# ---- figure (as c2_treadmill_cue_aligned.py)
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
        q = TR95[(TR95.channel == ch) & (TR95.epoch == e)]
        ok = q.n_sessions.to_numpy() >= S.TRAJ_MIN_SESSIONS
        y, se = np.where(ok, q["mean"], np.nan), np.where(ok, q["sem"], np.nan)
        col = "#222222" if e == "pre" else S.EP[e]
        a.fill_between(q.t, y - se, y + se, color=col, alpha=S.TRAJ_BAND, lw=0)
        a.plot(q.t, y, color=col, lw=S.TRAJ_LW, label=e)
    a.set_xlim(-1, 3); a.set_xlabel("s from cue"); a.set_ylabel(ylab, fontsize=7)
ax[0].legend(loc="upper left", fontsize=5.8, handlelength=1.0)
for i, (m, ylab) in enumerate(PRIMARY.items()):
    a = ax[2 + i]; S.zero_line(a)
    base = Sx[Sx.epoch == "pre"].groupby("animal")[m].mean()
    D = Sx.assign(d=Sx[m] - Sx.animal.map(base))

    def tg(p, e, m=m):
        r = prim[(prim.metric == m) & (prim.epoch == e)]
        if r.empty:
            return None
        r = r.iloc[0]
        sig = bool(r.q_fdr < 0.05 and (r.boot_lo > 0 or r.boot_hi < 0))
        return r.boot_change, r.boot_lo, r.boot_hi, (2 if sig else 0)
    w = CP.grouped(a, ["all"], lambda p, e: D[D.epoch == e].d.dropna().values, tg)  # noqa: B023
    a.set_xticks([]); a.set_xlim(-0.5, 0.5); a.set_ylabel(ylab, fontsize=6.6); S.headroom(a, 0.3)
    for j, e in enumerate(EPS):
        r = tg("all", e)
        if r and r[3]:
            CP.star(a, (j - 1.5) * w, j, S.EP[e], level=r[3])
CP.epoch_legend(ax[3], loc="upper left", ncol=1)
ax[3].get_legend().set_bbox_to_anchor((1.02, 1.0))                  # outside the axes, clear of the data
fig.subplots_adjust(left=0.07, right=0.89, bottom=0.22, top=0.86, wspace=0.55)
S.save(fig, str(OUT / "C2_treadmill_cue_first95"))

# ---- matched vs unmatched (same loader), cue-window means
for ch, (a0, a1) in (("forward_speed", (2.0, 3.0)), ("backward_velocity", (-0.25, 0.75))):
    for e in EPS:
        m95 = TR95[(TR95.channel == ch) & (TR95.epoch == e) & TR95.t.between(a0, a1)]["mean"].mean()
        mall = TRall[(TRall.channel == ch) & (TRall.epoch == e) & TRall.t.between(a0, a1)]["mean"].mean()
        print(f"{ch:18s} {e:9s} window {a0}..{a1}s: first95 {m95:+.3f}  all {mall:+.3f}")
print(prim[["metric", "epoch", "boot_change", "boot_lo", "boot_hi", "q_fdr", "agree"]].to_string(index=False))

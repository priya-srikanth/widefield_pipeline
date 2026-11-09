"""Does the shared increase track face / jaw / body movement? (Priya 11-09 "go ahead" on: "what drives the orofacial
increase, if not licking? jaw and face movement and body movement around the first lick").

Extends `state_space_shift_behavior.py` (licking only, B27) on ITS per-trial table (shift_behavior{tag}/trials.csv.gz:
region-mean activity minus RESTnl in three windows, licking covariates), adding per trial:
  jaw      pose3d kinematics jaw_trials.csv.gz: maximum opening (mm), lateral jaw position when wide, frames wide
  face     FaceRhythm master_pre_v1 (12 factors; `o2/fr_cam4_20261021/factors/master_pre_v1/proj`): each factor's
           absolute expression (trials x trials_D) averaged over -1.0..+0.7 s from the first lick (the neural window;
           first-lick latency from the landing table), log1p; plus the face-motion magnitude D itself
  running  fraction of the same window inside a treadmill running bout (behavior_events running_starts / stops,
           > 3 mm/s for >= 2 s; DAQ time = the FR file's cue_daq_s + latency)
Joined on (session label, trial_id) -- all tables share the cam4 window trial_id (as licks.csv.gz).

Models (per animal x readout, OLS on PRE trials, spout fixed effects + covariates z-scored on pre; pre sessions scored
leave-session-out), all fit on the SAME trials (complete for the largest model), so they differ only in covariates:
  lick            the B27 covariates (licks / contacts in the 700-ms bout, latency, reach, speed)
  lick + jaw
  lick + jaw + face + running
Per model: residual change vs own pre (observed change minus the change the pre-stroke fit predicts from the trial's
movement), session means, animal-weighted nested bootstrap (`boot_delta`, 2,000); * = 95 % CI excludes 0. Model power:
leave-session-out R^2 on pre trials (how much of the pre trial-to-trial variance beyond spout the covariates carry).
Coverage: FaceRhythm has no June sessions on this basis and only engaged trials; the running bouts are coarse.
Outputs: <state_space_recovery>/shift_movement{tag}/{changes,r2}.csv, page_figs/ss_shift_movement{tag}.png.
EXPLORATORY.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_chronic_maps as C  # noqa: E402
import state_space_shift_behavior as SB  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE  # noqa: E402

EPS = SB.EPS
OUT = S.ROOT / f"shift_movement{S.tag()}"
FIGS = S.ROOT / "page_figs"
PRIYA = Path("M:/MICROSCOPE/Priya")
JAW = PRIYA / "DeepLabCut/Widefield/session_poses/pose3d/kinematics/jaw_trials.csv.gz"
FR = PRIYA / "DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1/proj"
EVENTS = PRIYA / "Behavior_logs/Widefield/behavior_summary/events"
WIN = (-1.0, 0.7)
JAWC = ["open_max_mm", "lr_wide_mm", "n_wide"]
FRC = [f"fr{k}" for k in range(12)] + ["fr_D"]
MODELS = {"lick": SB.BEH, "lick + jaw": SB.BEH + JAWC, "lick + jaw + face + running": SB.BEH + JAWC + FRC + ["run"]}


def face_running(T):
    """Per (label, trial_id): FaceRhythm factor expression and running fraction in the first-lick window."""
    rt = T.dropna(subset=["trial_id", "rt_s"]).set_index(["label", "trial_id"]).rt_s
    rows = []
    for f in sorted(glob.glob(str(FR / "PS9*_cam4.npz"))):
        z = np.load(f)
        an, date = Path(f).stem.split("_")[:2]
        lab = f"{an}_{date[4:]}"
        ev = EVENTS / an / f"{date}.npz"
        bouts = None
        if ev.exists():
            e = np.load(ev)
            bouts = np.stack([e["running_starts"], e["running_stops"]], 1) / float(e["fs"])
        X = z["trials"].astype(float) * z["trials_D"][..., None]
        D = z["trials_D"].astype(float)
        tms = z["t_ms"]
        for i, tid in enumerate(z["trial_id"]):
            r = rt.get((lab, float(tid)))
            if r is None or not np.isfinite(r):
                continue
            ix = (tms >= (r + WIN[0]) * 1000) & (tms <= (r + WIN[1]) * 1000)
            if ix.sum() < 10:
                continue
            row = {"label": lab, "trial_id": float(tid), "fr_D": np.log1p(D[i, ix].mean())}
            row.update({f"fr{k}": np.log1p(max(X[i, ix, k].mean(), 0)) for k in range(12)})
            a, b = z["cue_daq_s"][i] + r + WIN[0], z["cue_daq_s"][i] + r + WIN[1]
            run = 0.0
            if bouts is not None and len(bouts):
                run = float(np.clip(np.minimum(bouts[:, 1], b) - np.maximum(bouts[:, 0], a), 0, None).sum() / (b - a))
            row["run"] = run if bouts is not None else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def fit(T, reads, cov, tag):
    """Adds obs / res columns for one covariate set (as SB.main); returns pre leave-session-out R^2 rows."""
    r2 = []
    for an, g in T.groupby("animal"):
        for col in reads:
            gg = g[g[col].notna()]
            pre = (gg.epoch == "pre").to_numpy()
            mu, sd = gg[pre][cov].mean(), gg[pre][cov].std().replace(0, 1)
            Xs = pd.get_dummies(gg.spout).astype(float)
            X = pd.concat([Xs, (gg[cov] - mu) / sd], axis=1).to_numpy()
            y = gg[col].to_numpy()
            labs = gg.label.to_numpy()
            pred, base = np.full(len(gg), np.nan), np.full(len(gg), np.nan)
            beta, *_ = np.linalg.lstsq(X[pre], y[pre], rcond=None)
            pred[~pre] = X[~pre] @ beta
            for lab in np.unique(labs[pre]):
                m = pre & (labs != lab)
                b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
                pred[labs == lab] = X[labs == lab] @ b
                bs, *_ = np.linalg.lstsq(Xs.to_numpy()[m], y[m], rcond=None)        # spout-only, for R^2
                base[labs == lab] = Xs.to_numpy()[labs == lab] @ bs
            ss0 = ((y[pre] - base[pre]) ** 2).sum()
            r2.append({"animal": an, "readout": col, "model": tag, "r2_beyond_spout": 1 - ((y[pre] - pred[pre]) ** 2).sum()
                       / ss0 if ss0 > 0 else np.nan})
            sp = gg.spout.to_numpy()
            mo = {s: y[pre & (sp == s)].mean() for s in np.unique(sp[pre])}
            mp = {s: pred[pre & (sp == s)].mean() for s in np.unique(sp[pre])}
            bo = np.array([mo.get(s, np.nan) for s in sp])
            bp = np.array([mp.get(s, np.nan) for s in sp])
            T.loc[gg.index, f"obs | {col}"] = y - bo
            T.loc[gg.index, f"res {tag} | {col}"] = (y - bo) - (pred - bp)
    return r2


def main():
    src = S.ROOT / f"shift_behavior{S.tag()}" / "trials.csv.gz"
    T = pd.read_csv(src)
    reads = [c for c in T.columns if " | " in c and not c.startswith(("obs", "pred", "res"))]
    T = T[list(dict.fromkeys(["animal", "label", "epoch", "day", "spout", "trial_id", "rt_s", *reads, *SB.BEH]))]
    T = T.merge(face_running(T), on=["label", "trial_id"], how="left")
    J = pd.read_csv(JAW, usecols=["session", "trial_id", *JAWC])
    J["label"] = J.session.str.replace("_2026", "_", regex=False)
    J["trial_id"] = J.trial_id.astype(float)
    T = T.merge(J.drop(columns="session"), on=["label", "trial_id"], how="left")
    full = MODELS["lick + jaw + face + running"]
    n0 = len(T)
    T = T[T[full].notna().all(1)].copy()
    cover = T.groupby(["animal", "epoch"]).label.nunique().unstack()
    print(f"trials {n0} -> {len(T)} with every covariate; sessions per animal x epoch:\n{cover}")
    r2 = []
    for tag, cov in MODELS.items():
        r2 += fit(T, reads, cov, tag)
    OUT.mkdir(parents=True, exist_ok=True)
    R2 = pd.DataFrame(r2)
    R2.to_csv(OUT / "r2.csv", index=False)
    print(R2.groupby(["readout", "model"]).r2_beyond_spout.mean().unstack().round(3).to_string())
    cols = [c for c in T.columns if c.startswith(("obs |", "res "))]
    Sx = T.groupby(["animal", "label", "epoch", "day"])[cols].mean().reset_index()
    rng = np.random.default_rng(0)

    def change(col, ep):
        pre = {a: g[g.epoch == "pre"][col].dropna().tolist() for a, g in Sx.groupby("animal")}
        post = {a: g[g.epoch == ep][col].dropna().tolist() for a, g in Sx.groupby("animal")}
        post = {a: v for a, v in post.items() if v and pre.get(a)}
        iv = kit.boot_delta(post, pre, rng) if post else None
        up = sum(np.mean(post[a]) > np.mean(pre[a]) for a in post)
        return (iv.point, iv.lo, iv.hi, up, len(post)) if iv is not None else (np.nan,) * 3 + (0, 0)

    kinds = [("obs", "observed change", "o", 1.0)] + [(f"res {m}", f"residual beyond {m}", "D", a)
                                                     for m, a in zip(MODELS, (0.35, 0.6, 1.0))]
    regs = sorted({c.split(" | ")[0] for c in reads}, key=lambda x: (not x.startswith("orofacial"), x))
    wins = list(SB.WINDOWS)
    fig, axs = plt.subplots(len(regs), len(wins), figsize=(14, 3.0 * len(regs)), squeeze=False)
    recs = []
    for i, reg in enumerate(regs):
        for j, wn in enumerate(wins):
            ax = axs[i, j]
            col = f"{reg} | {wn}"
            for k, ep in enumerate(EPS):
                for m, (kk, _lab, mk, al) in enumerate(kinds):
                    pt, lo, hi, up, n = change(f"{kk} | {col}", ep)
                    x = k + (m - 1.5) * 0.18
                    ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt=mk, color=EPOCH_LINE[ep], ms=4.5, lw=1.3,
                                capsize=0, alpha=al)
                    if lo > 0 or hi < 0:
                        sig_marks.star_top(ax, x, "*", color=EPOCH_LINE[ep], fontsize=8)
                    recs.append({"readout": col, "kind": kk, "epoch": ep, "change": pt, "lo": lo, "hi": hi, "k_up": up,
                                 "n": n})
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(range(3), EPS)
            r2m = R2[R2.readout == col].groupby("model").r2_beyond_spout.mean()
            ax.set_title(f"{reg}, {wn}\npre R$^2$ beyond spout: " + " / ".join(f"{r2m.get(m, np.nan):.2f}"
                                                                             for m in MODELS), fontsize=9)
            if j == 0:
                ax.set_ylabel("change vs own pre\n(pre pixel RMS)")
    hs = [plt.Line2D([], [], marker=mk, ls="", color="0.2", alpha=al, label=lab) for _k, lab, mk, al in kinds]
    axs[0, 0].legend(handles=hs, frameon=False, fontsize=7, loc="upper left")
    pd.DataFrame(recs).to_csv(OUT / "changes.csv", index=False)
    fig.suptitle(C._wrap(
        "Does the shared increase track face / jaw / body movement? Region-mean activity minus RESTnl (cohort SVD, "
        "first-lick-aligned; units of the animal's pre-template pixel RMS), change vs own pre: OBSERVED (circle) and "
        "RESIDUAL beyond what each animal's pre-stroke fit predicts from the trial's movement (diamonds, light -> dark): "
        "licking; + jaw; + FaceRhythm (12 factors + face-motion magnitude, -1.0..+0.7 s) + treadmill running. Same "
        "trials for every model (all covariates present; FaceRhythm sessions Aug-Oct). Pre sessions leave-session-out. "
        "Titles: pre-trial leave-session-out R^2 beyond spout per model. Nested bootstrap animals -> sessions (2,000); "
        "* = 95 % CI excludes 0."), y=1.03, fontsize=9.5)
    sig_marks.reserve_headroom(fig)
    fig.tight_layout()
    save_png_svg(fig, FIGS / f"ss_shift_movement{S.tag()}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    P = pd.DataFrame(recs)
    print(P[P.readout.str.startswith("orofacial")].round(2).to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

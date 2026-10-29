"""Is the CORTICAL pre-cue (ENL) position code face movement? Cortical ENL decoding after regressing out the
FaceRhythm factors.

    python -m scripts.enl_fr_removed_decode [--labels PS93_0814 ...] [--jobs 6]

Priya 10-25 (FR handoff 0b open item; FINDINGS 5.13): the face already carries spout position before the cue (FR ENL
decode .34-.43, chance .17), in the same range as cortex (.33-.55). Is the cortical code that face movement, read out
by cortex? `enl_movement_multisession` asked this for running + 4-camera video motion energy (residual keeps ~40-50 %
of the above-chance code, tracks raw session by session); this is the same test with the FROZEN FR factors
(`master_pre_v1`, 12 named face movements) as the movement model -- interpretable, and no video read, so it runs
locally on the cohort.

SAME TRIALS, FEATURES, DECODER AND IN-FOLD REMOVAL as `enl_movement_multisession` (its `enl_table`, `lagged`,
`features`): [cue - 2 s, cue) lick-free (DLC lick onsets), spout strobe >= 2 s before the cue, engaged (lick within
max_rt after the cue), all LocaNMF components, 4 x 0.5 s bin means, logistic C 0.5, position runs as GroupKFold
groups; every movement model (RidgeCV, movement -> dF/F, lags 0 / 0.1 / 0.2 s, movement leads) is fit inside each
fold on TRAINING trials and subtracted from every frame. Arms:
  raw              no removal (should equal the video analysis's raw on shared sessions -- the check)
  resid_run        minus running only (what the treadmill alone removes)
  resid_fr         minus the 12 FR factors
  resid_fr_run     minus FR + running (THE readout)
  fr_only          the FR factors themselves as the decoder input (cf. `fr_enl_decode`: here on the imaging frames
                   and the cortical trial set)
  *_end05          raw / resid_fr_run / fr_only with the features stopping 0.5 s before the cue: the FR spectrogram
                   is ACAUSAL (VQT sigma ~0.3-1 s), so FR in the last bins can carry post-cue licking; the end-0.5 s
                   arms are free of that (the movement model is still fit on all frames).
FR on frames: `face_rhythm_outputs.fr_signals` (absolute expression, the encoder's default), NaN (outside the FR grid
cue -2 .. +4 s, so the lag lead-in, or a gap) -> 0 = the mean after z-scoring. FR has no June (August-only pre baseline,
the FR rule), so sessions without a projection are skipped. Removal is LINEAR and over-removal is possible (FR also
captures cortex-driven movement): the residual is a lower bound, as for video.

Output: <labcams>/enl_fr_removed_decode/ (sessions/<label>.csv, enl_fr_removed_decode.csv, figure).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("raw", "resid_run", "resid_fr", "resid_fr_run", "fr_only", "raw_end05", "resid_fr_run_end05", "fr_only_end05")
END_S = 0.5


def run_session(label: str, n_perm: int = 1000) -> list[dict]:
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    import scripts.enl_movement_multisession as EM
    import scripts.movement_encoding_session as MS
    from wfield_local import config, epochs
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.behavior_events import session_speed
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    s = next(x for x in config.load_sessions() if x["label"] == label)
    dp = config.defaults()["decode"]
    win_s, lead_s = float(dp["precue_post_s"]), 0.6
    animal, date, tr = EM.enl_table(s, rv, win_s, float(dp["max_rt_s"]))
    Y, _reg, ft = MS.imaging(label)
    trial_frames = [np.flatnonzero((ft >= c - win_s) & (ft < c)) for c in tr.cue_s]
    cov = np.array([len(f) >= 0.9 * int(round(win_s * EM.FPS_IMG)) for f in trial_frames])
    tr = tr[cov].reset_index(drop=True)
    trial_frames = [f for f, k in zip(trial_frames, cov) if k]
    trial_frames_e = [f[ft[f] < c - END_S] for f, c in zip(trial_frames, tr.cue_s)]
    y, groups = tr.pos_name.to_numpy(), tr.block.to_numpy()
    enl = [(c - win_s - lead_s, c) for c in tr.cue_s]
    wmask = np.zeros(len(ft), bool)
    for t0, t1 in enl:
        wmask |= (ft >= t0) & (ft < t1)
    frame_idx = np.flatnonzero(wmask)
    fr = fro.fr_signals(animal, date, rv, ft[frame_idx], blank=False, absolute=True)
    F = np.column_stack([v for _t, v in fr.values()])
    ts, sp = session_speed(Path(s["h5"]))
    run = np.interp(ft[frame_idx], ts, sp)[:, None]

    def z(M):
        M = (M - np.nanmean(M, 0)) / np.where(np.nanstd(M, 0) > 0, np.nanstd(M, 0), 1.0)
        return np.nan_to_num(M, nan=0.0)

    F, run = z(F), z(run)
    lags = [int(round(L * EM.FPS_IMG)) for L in EM.LAGS_S]
    D = {"resid_run": EM.lagged(run, frame_idx, lags), "resid_fr": EM.lagged(F, frame_idx, lags),
         "resid_fr_run": EM.lagged(np.hstack([F, run]), frame_idx, lags)}
    Yw = Y[frame_idx]
    trial_of_frame = np.full(len(frame_idx), -1)
    for k, (t0, t1) in enumerate(enl):
        trial_of_frame[(ft[frame_idx] >= t0) & (ft[frame_idx] < t1)] = k
    nb = int(dp["bins"]["precue"])
    X = {"raw": EM.features(Yw, frame_idx, trial_frames, nb), "fr_only": EM.features(F, frame_idx, trial_frames, nb),
         "raw_end05": EM.features(Yw, frame_idx, trial_frames_e, nb),
         "fr_only_end05": EM.features(F, frame_idx, trial_frames_e, nb)}
    pred = {a: np.empty(len(y), dtype=object) for a in ARMS}

    def fitpred(Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        return LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))

    for trn, tst in GroupKFold(min(5, np.unique(groups).size)).split(X["raw"], y, groups):
        for a in ("raw", "fr_only", "raw_end05", "fr_only_end05"):
            pred[a][tst] = fitpred(X[a][trn], y[trn], X[a][tst])
        fr_tr = np.isin(trial_of_frame, trn)
        for a, Dm in D.items():
            R = Yw - RidgeCV(alphas=np.logspace(0, 6, 7)).fit(Dm[fr_tr], Yw[fr_tr]).predict(Dm)
            Xr = EM.features(R, frame_idx, trial_frames, nb)
            pred[a][tst] = fitpred(Xr[trn], y[trn], Xr[tst])
            if a == "resid_fr_run":
                Xr = EM.features(R, frame_idx, trial_frames_e, nb)
                pred["resid_fr_run_end05"][tst] = fitpred(Xr[trn], y[trn], Xr[tst])
    rng = np.random.default_rng(0)
    ep, day = epochs.epoch_of(label), epochs.days_since_stroke(label)
    rows = []
    for a in ARMS:
        acc = float((pred[a] == y).mean())
        null = np.array([(pred[a] == rng.permutation(y)).mean() for _ in range(n_perm)])
        rows.append({"label": label, "animal": animal, "epoch": ep, "day": day, "arm": a, "n_trials": len(y),
                     "n_pos": int(np.unique(y).size), "acc": acc, "null_mean": float(null.mean()),
                     "p": float((np.sum(null >= acc) + 1) / (n_perm + 1))})
    return rows


def _worker(label):
    """Module-level for spawn (CLAUDE.md rule 6); one session -> its rows, cached as sessions/<label>.csv."""
    import os
    out = Path(os.environ["ENL_FR_OUT"]) / "sessions" / f"{label}.csv"
    if out.exists():
        return pd.read_csv(out).to_dict("records")
    rows = run_session(label)
    pd.DataFrame(rows).to_csv(out, index=False)
    return rows


def summary(d: pd.DataFrame) -> pd.DataFrame:
    """Per animal x epoch: mean accuracy per arm, and the share of raw's above-null code each residual keeps."""
    m = d.pivot_table(index=["animal", "epoch", "label"], columns="arm", values="acc")
    nl = d.pivot_table(index=["animal", "epoch", "label"], columns="arm", values="null_mean")
    for a in ("resid_run", "resid_fr", "resid_fr_run"):
        m[f"kept_{a}"] = (m[a] - nl[a]) / (m["raw"] - nl["raw"])
    m["kept_resid_fr_run_end05"] = (m["resid_fr_run_end05"] - nl["resid_fr_run_end05"]) / (m["raw_end05"]
                                                                                         - nl["raw_end05"])
    return m.reset_index()


def figure(d: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    from wfield_local.face_rhythm import common as frc
    acol = frc.animal_colors()
    eps = [e for e in ("pre", *frc.POST_EPOCHS)]
    arms = [("raw", "raw cortex"), ("resid_fr_run", "minus FR + running"), ("resid_run", "minus running"),
            ("fr_only", "FR factors alone")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for c_i, end in enumerate(("", "_end05")):
        ax = axs[c_i]
        for j, (a, lab) in enumerate(arms):
            arm = a + end if a + end in set(d.arm) else None
            if arm is None:
                continue
            g = d[d.arm == arm].groupby(["animal", "epoch"]).acc.mean().unstack().reindex(columns=eps)
            for an, row in g.iterrows():
                ax.plot(np.arange(len(eps)) + 0.08 * j, row.to_numpy(), "o-", color=acol.get(an, "0.5"),
                        alpha=[1, 0.9, 0.45, 0.6][j], ls=["-", "--", ":", "-."][j], lw=1, ms=3)
            ax.plot([], [], color="k", ls=["-", "--", ":", "-."][j], label=lab)
        ax.axhline(1 / 6, color="0.5", ls=":", lw=0.8)
        ax.set_xticks(range(len(eps)), eps)
        ax.set_ylabel("ENL position decoding accuracy")
        ax.set_title("features [cue - 2 s, cue)" if not end else "features end 0.5 s before the cue", fontsize=9)
        ax.legend(fontsize=7)
    S = summary(d)
    ax = axs[2]
    for an, g in S.groupby("animal"):
        ax.scatter(g.raw, g.resid_fr_run, s=12, color=acol.get(an, "0.5"), label=an)
    lim = [0.1, max(0.8, float(S.raw.max()) + 0.05)]
    ax.plot(lim, lim, color="0.6", lw=0.7)
    ax.axhline(1 / 6, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("raw cortex (per session)")
    ax.set_ylabel("minus FR + running")
    ax.legend(fontsize=7)
    fig.suptitle("Cortical pre-cue (ENL, lick-free) position code after regressing out the FaceRhythm factors "
                 "(in-fold ridge, lags 0-0.2 s; per animal, mean over sessions)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    import os

    import matplotlib
    matplotlib.use("Agg")

    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    rv = PathResolver()
    out = Path(rv.root("cohort_out")) / "enl_fr_removed_decode"
    assert_writable(out)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    os.environ["ENL_FR_OUT"] = str(out)
    fdir = fro.factor_dir(rv)
    have = {"_".join(p.stem.split("_")[:2]) for p in (fdir / "proj").glob("*.npz")}
    sess = [s["label"] for s in config.load_sessions() if s["regime"] == "B" and epochs.epoch_of(s["label"]) is not None
            and f"{s['label'].split('_')[0]}_{Path(s['h5']).parent.name}" in have]
    if a.labels:
        sess = [x for x in sess if x in a.labels]
    if not a.plot_only:
        print(f"{len(sess)} sessions with an FR projection", flush=True)
        res, fail = ak.fan_sessions(sess, _worker, jobs=a.jobs)
        for item, err in fail:
            print(f"!! {item}: {err}", flush=True)
    d = pd.concat([pd.read_csv(f) for f in sorted((out / "sessions").glob("*.csv"))], ignore_index=True)
    d.to_csv(out / "enl_fr_removed_decode.csv", index=False, float_format="%.4g")
    S = summary(d)
    pd.set_option("display.width", 220)
    cols = ["raw", "resid_run", "resid_fr", "resid_fr_run", "fr_only", "kept_resid_fr", "kept_resid_fr_run",
            "raw_end05", "resid_fr_run_end05", "fr_only_end05", "kept_resid_fr_run_end05"]
    print("\nPER ANIMAL x EPOCH (rule 8), mean over sessions; kept = (resid - null) / (raw - null)")
    print(S.groupby(["animal", "epoch"])[cols].mean().round(3).to_string())
    print(S.groupby(["animal", "epoch"]).size().unstack().to_string())
    figure(d, out / "enl_fr_removed_decode.png")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

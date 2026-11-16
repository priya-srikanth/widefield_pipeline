"""TASK-INDEPENDENT MOVEMENT (TIM), after Yin, Melin, ... Musall, Churchland (Neuron 2025, "Spontaneous movements and
their relationship to neural activity fluctuate with latent engagement states"; Priya 11-16: "task independent
movement metric would be good to incorporate"; "include cued response period").

    python -m wfield_local.task_independent_movement PS92_20260811 [...]     # sessions -> per-trial table
    python -m wfield_local.task_independent_movement --all [--jobs 4]        # every curated session with LP windows

WHAT THEIR METRIC IS (Neuron Methods): every DeepLabCut body part's x / y at every frame is regressed on seven task
regressors (stimulus, choice, reward, choice x reward + the previous trial's stimulus, choice, reward) + intercept;
TIM(t) = Euclidean distance between predicted and actual position, z-scored per part, averaged over parts and over the
stimulus + delay frames -> one value per trial. It is a BEHAVIOUR-ONLY measure of how much movement the task does not
predict (they found it rises when mice disengage, while total motion energy does not).

OUR VERSION (differences, each deliberate):
- TRACES: tongue x / y and jaw x / y in the mouth frame (`clean.npz`, LP run v36; tongue in the mouth = 0) and nose x / y
  (`windows.npz` raw LP px, likelihood < NOSE_MIN_P = NaN). One "part" = an (x, y) pair -> three parts.
- FRAMES: the v36 windows tile the session (each trial's window runs from the previous trial's stop + 3 s to its own
  stop + 3 s, cue-relative ms), binned to BIN_MS. Three PERIODS, scored separately:
    precue    strobe .. cue        (spout present, licking forbidden: the ENL)
    response  cue .. stop          (Priya: included -- task-independent movement exists there too; but with only the
                                    spout + timing as regressors this period's TIM also carries AIM ERROR, so it is
                                    reported separately and never pooled with the others)
    iti       stop .. next strobe  (time since the stop; post-stop frames of trial k and pre-strobe frames of k + 1)
- REGRESSORS, one ordinary least-squares fit PER (period, time bin) -- the frame-wise regression of the paper:
    precue / response: intercept + current spout (6 one-hot, deviation-free: 5 dummies) + previous spout (5) +
                       previous trial contact (hit) [+ current hit in the RESPONSE period only: the outcome happens there;
                       in the pre-cue period it has not happened yet, so it is not a task regressor of that period]
    iti:               intercept + the spout of the trial that just ended (5) + its hit
  Time bins: cue-locked (precue back to -PRECUE_CAP_S, earlier frames pooled into the first bin; response to
  +RESP_CAP_S), ITI locked to the stop (to ITI_CAP_S, later frames pooled). The spout x time interaction is therefore free
  (a separate fit per bin), as in the paper.
- CROSS-VALIDATED (the paper does not say; ours must be): predictions for WORKING trials (the production working gate,
  `trial_classes.in_sel` & `before_session_end`) come from fits on the other folds, folds = GroupKFold over `block_ids` (spout BLOCKS, so a
  held-out trial's block-mates are never in its training set); every other trial (jaw-only / no response / stopped /
  quit tail) is predicted by the fit on ALL working trials -> their TIM is deviation from the working-trial movement.
- OUTPUT per trial x period x part: mean Euclidean residual (px), mean Euclidean deviation from the period-bin MEAN of the
  working trials (the same CV, intercept-only: "total" movement variability), frames used. TIM = residual; the task
  share = 1 - residual^2 / total^2 is reported alongside. Scaling to the animal's own pre (Priya 11-07: every readout as
  delta vs the animal's own pre) happens at the cohort step (`cohort_tables`), not here.

Kept apart from the encoders on purpose: nothing here reads imaging, so it can run on every session with LP windows.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 1
BIN_MS = 40.0                     # 10 frames at 250 fps
TBIN_MS = {"precue": 100.0, "response": 100.0, "iti": 250.0}
PRECUE_CAP_S = 10.0
RESP_CAP_S = 8.0
ITI_CAP_S = 30.0
NOSE_MIN_P = 0.5
N_FOLDS = 5
PARTS = ("tongue", "jaw", "nose")
PERIODS = ("precue", "response", "iti")
MIN_TRAIN = 12                    # working trials with data in a (period, bin) needed to fit it


def _run() -> str:
    from wfield_local import orofacial_cohort as oc
    return os.environ.get("TIM_RUN", oc.RUN)


def session_dir(label: str) -> Path:
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    animal, date = label.split("_")
    date = date if len(date) == 8 else f"2026{date}"
    return Path(oo.session_dir(PathResolver(), _run(), animal, date))


def out_dir() -> Path:
    env = os.environ.get("TIM_OUT")
    if env:
        return Path(env)
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    return Path(oo.root(PathResolver())) / _run() /"task_independent_movement" / f"v{VERSION}"


# ----------------------------------------------------------------------------------------------------------- frames ----
def frame_table(sd: Path) -> pd.DataFrame:
    """Per-BIN_MS bin of every trial window: trial_id, t_ms (bin centre, cue-relative), strobe_ms, stop_ms, the six
    traces (bin means; NaN where no valid frame)."""
    W = np.load(sd / "windows.npz", allow_pickle=True)
    C = np.load(sd / "clean.npz", allow_pickle=True)
    parts = list(W["parts"])
    nose = W["pose"][:, parts.index("nose"), :]
    nx = np.where(nose[:, 2] >= NOSE_MIN_P, nose[:, 0], np.nan)
    ny = np.where(nose[:, 2] >= NOSE_MIN_P, nose[:, 1], np.nan)
    t = W["ix_t_ms"].astype(float)
    F = pd.DataFrame({"trial_id": W["ix_trial_id"], "b": np.floor(t / BIN_MS).astype(np.int64),
                      "strobe_ms": W["ix_strobe_ms"], "stop_ms": W["ix_stop_ms"],
                      "tongue_x": C["tongue_x"], "tongue_y": C["tongue_y"], "jaw_x": C["jaw_x"], "jaw_y": C["jaw_y"],
                      "nose_x": nx, "nose_y": ny})
    G = F.groupby(["trial_id", "b"], sort=True).mean().reset_index()
    G["t_ms"] = (G.b + 0.5) * BIN_MS
    # window start per trial (= previous stop + 3 s): ITI time for pre-strobe bins
    G["t0_ms"] = G.groupby("trial_id").t_ms.transform("min") - BIN_MS / 2
    return G.drop(columns="b")


def label_periods(G: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """period, time-bin key and the OWNER trial (whose spout / outcome are the regressors) of every bin."""
    t, st, sp = G.t_ms.to_numpy(), G.strobe_ms.to_numpy(float), G.stop_ms.to_numpy(float)
    order = trials.trial_id.to_numpy()
    prev_of = dict(zip(order[1:], order[:-1]))
    resp = (t >= 0) & (t < sp)
    pre = (t < 0) & (t >= st)
    post = t >= sp                                       # after this trial's stop: ITI owned by THIS trial
    early = t < st                                       # before this trial's strobe: ITI owned by the PREVIOUS trial
    period = np.select([resp, pre, post | early], ["response", "precue", "iti"], "")
    tid = G.trial_id.to_numpy()
    owner = np.where(early, pd.Series(tid).map(prev_of).to_numpy(), tid)
    # ITI time since the owner's stop: post-stop bins = t - stop; pre-strobe bins = time from window start + 3 s
    t_iti = np.where(post, t - sp, t - G.t0_ms.to_numpy() + 3000.0)
    tb = np.select([resp, pre, post | early],
                   [np.minimum(t // TBIN_MS["response"], RESP_CAP_S * 1000 // TBIN_MS["response"]),
                    np.maximum(t // TBIN_MS["precue"], -PRECUE_CAP_S * 1000 // TBIN_MS["precue"]),
                    np.minimum(t_iti // TBIN_MS["iti"], ITI_CAP_S * 1000 // TBIN_MS["iti"])], np.nan)
    G = G.assign(period=period, owner=owner, tbin=tb)
    return G[(G.period != "") & np.isfinite(G.owner.astype(float))].copy()


# ---------------------------------------------------------------------------------------------------------- design ----
def trial_design(trials: pd.DataFrame, pos_order) -> dict:
    """trial_id -> regressor vector per period (see module docstring)."""
    P = {p: i for i, p in enumerate(pos_order)}
    k = len(pos_order) - 1

    def onehot(p):
        v = np.zeros(k)
        i = P.get(p, -1)
        if i > 0:
            v[i - 1] = 1.0
        return v
    pos = trials.position.to_numpy()
    hit = trials.hit.to_numpy(float)
    out = {}
    for j, tid in enumerate(trials.trial_id.to_numpy()):
        pp, ph = (pos[j - 1], hit[j - 1]) if j else (None, np.nan)
        prev = np.r_[onehot(pp), ph if np.isfinite(ph) else 0.0]
        cur = onehot(pos[j])
        out[tid] = {"precue": np.r_[1.0, cur, prev],
                    "response": np.r_[1.0, cur, prev, hit[j]],
                    "iti": np.r_[1.0, cur, hit[j]]}    # owner = the trial that just ended
    return out


# ------------------------------------------------------------------------------------------------------------- fit ----
TRACES = [f"{p}_{a}" for p in PARTS for a in "xy"]


#: light RIDGE on every regressor but the intercept (units: trials). Plain OLS failed on the first session (PS92_0811):
#: within spout BLOCKS the previous spout equals the current one on most trials, so the combination at a block
#: boundary is barely seen in training and OLS extrapolated it wildly on held-out trials (nose CV "task share" -3.8).
RIDGE = 2.0


def _fit_predict(Xtr, Ytr, Xte):
    """Ridge (RIDGE, intercept unpenalised) per trace with that trace's finite training rows; NaN where it cannot fit."""
    out = np.full((len(Xte), Ytr.shape[1]), np.nan)
    pen = np.full(Xtr.shape[1], RIDGE)
    pen[0] = 0.0
    for c in range(Ytr.shape[1]):
        ok = np.isfinite(Ytr[:, c])
        if ok.sum() < MIN_TRAIN:
            continue
        A = Xtr[ok]
        b = np.linalg.solve(A.T @ A + np.diag(pen), A.T @ Ytr[ok, c])
        out[:, c] = Xte @ b
    return out


def residuals(G: pd.DataFrame, design: dict, working: set, fold_of: dict) -> pd.DataFrame:
    """Adds pred_<trace> (task model) and mean_<trace> (intercept-only) to every bin, cross-validated over folds."""
    G = G.copy()
    for c in TRACES:
        G[f"pred_{c}"] = np.nan
        G[f"mean_{c}"] = np.nan
    for (per, _tb), H in G.groupby(["period", "tbin"], sort=False):
        X = np.vstack([design[o][per] for o in H.owner.to_numpy()])
        Y = H[TRACES].to_numpy(float)
        w = H.owner.map(lambda o: o in working).to_numpy(bool)
        fo = H.owner.map(lambda o: fold_of.get(o, -1)).to_numpy()
        pred = np.full_like(Y, np.nan)
        mean = np.full_like(Y, np.nan)
        if w.sum() < MIN_TRAIN:
            continue
        for f in np.unique(fo[w]):                       # held-out working trials
            te = w & (fo == f)
            tr = w & (fo != f)
            pred[te] = _fit_predict(X[tr], Y[tr], X[te])
            mean[te] = _fit_predict(X[tr][:, :1], Y[tr], X[te][:, :1])
        if (~w).any():                                   # non-working trials: fit on every working trial
            pred[~w] = _fit_predict(X[w], Y[w], X[~w])
            mean[~w] = _fit_predict(X[w][:, :1], Y[w], X[~w][:, :1])
        idx = H.index
        G.loc[idx, [f"pred_{c}" for c in TRACES]] = pred
        G.loc[idx, [f"mean_{c}" for c in TRACES]] = mean
    return G


def per_trial(G: pd.DataFrame) -> pd.DataFrame:
    """Per owner trial x period x part: mean Euclidean residual (TIM, px), mean deviation from the bin mean (total, px),
    bins used."""
    rows = []
    for p in PARTS:
        x, y = G[f"{p}_x"].to_numpy(float), G[f"{p}_y"].to_numpy(float)
        res = np.hypot(x - G[f"pred_{p}_x"].to_numpy(float), y - G[f"pred_{p}_y"].to_numpy(float))
        tot = np.hypot(x - G[f"mean_{p}_x"].to_numpy(float), y - G[f"mean_{p}_y"].to_numpy(float))
        D = pd.DataFrame({"trial_id": G.owner.to_numpy(), "period": G.period.to_numpy(), "part": p,
                          "res": res, "tot": tot, "res2": res ** 2, "tot2": tot ** 2})
        D = D[np.isfinite(D.res) & np.isfinite(D.tot)]
        rows.append(D.groupby(["trial_id", "period", "part"]).agg(tim_px=("res", "mean"), total_px=("tot", "mean"),
                                                                  res2=("res2", "mean"), tot2=("tot2", "mean"),
                                                                  n_bins=("res", "size")).reset_index())
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------------------------- session ----
def session_tim(label: str) -> pd.DataFrame | None:
    from wfield_local import block_ids as bi
    from wfield_local import trial_classes as tc
    sd = session_dir(label)
    if not (sd / "windows.npz").exists() or not (sd / "clean.npz").exists():
        return None
    animal, date = label.split("_")
    date = date if len(date) == 8 else f"2026{date}"
    S = tc.session_trials(f"{animal}_{date}")
    if S is None or not len(S):
        return None
    S = S.sort_values("trial_id").reset_index(drop=True)
    S["hit"] = (S.cls == "contact").astype(float)
    pos_order = sorted(S.position.dropna().unique())
    # PRODUCTION working gate (11-03 late): the run's camera classes (contact + incomplete) AND at or before the
    # session's last anchor trial (`before_session_end`: the quit tail is not working)
    work = tc.in_sel(S.cls.to_numpy()) & tc.before_session_end(S.cls.to_numpy(), S.trial_id.to_numpy(float),
                                                               position=S.position.to_numpy())
    codes = S.position.map({p: i for i, p in enumerate(pos_order)}).fillna(-1).astype(int).to_numpy()
    blocks = bi.block_ids(codes)
    from sklearn.model_selection import GroupKFold
    wi = np.flatnonzero(work & (blocks >= 0))
    fold_of = {}
    if len(np.unique(blocks[wi])) >= N_FOLDS:
        for f, (_tr, te) in enumerate(GroupKFold(N_FOLDS).split(wi, groups=blocks[wi])):
            for i in wi[te]:
                fold_of[S.trial_id.iat[i]] = f
    working = set(S.trial_id.to_numpy()[wi])
    G = label_periods(frame_table(sd), S[["trial_id"]])
    G = residuals(G, trial_design(S, pos_order), working, fold_of)
    R = per_trial(G)
    meta = S[["trial_id", "position", "cls"]].assign(working=work, block=blocks)
    R = R.merge(meta, on="trial_id", how="left")
    R.insert(0, "label", f"{animal}_{date}")
    R.insert(1, "animal", animal)
    return R


def _job(label: str):
    out = out_dir() / "sessions" / f"{label}.csv.gz"
    if out.exists() and not os.environ.get("TIM_FORCE"):
        return label, str(out)
    R = session_tim(label)
    if R is None:
        return label, None
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    R.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out)
    return label, str(out)


def all_labels() -> list[str]:
    from wfield_local import analysis_kit as ak
    return sorted(ak.curated_ids())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    labels = all_labels() if a.all else a.labels
    from wfield_local import analysis_kit as ak
    res, fails = ak.fan_sessions(labels, _job, jobs=a.jobs)
    print(f"done {len(res)}  failed {len(fails)}  -> {out_dir()}")
    for f in fails:
        print("  FAIL", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())

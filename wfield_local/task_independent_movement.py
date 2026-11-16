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

ENGAGEMENT STATES (planned; Priya 11-16): if states are defined by a GLM-HMM as in Yin et al., the HMM is fitted on ALL
trials of the session -- never on engaged / working trials, which would remove the disengaged stretches it exists to
find. Our observation per trial is the outcome class (contact / incomplete / jaw-only / no response), inputs = spout
position + previous outcome (+ latency); the camera classes and the quit tail are compared with the inferred states,
not used to define them. TIM / the encoders are then split by state (trial counts matched, as the paper does).

Kept apart from the encoders on purpose: nothing here reads imaging, so it can run on every session with LP windows.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 2
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
    return Path(oo.root(PathResolver())) / _run() / "task_independent_movement" / f"v{VERSION}"


# ----------------------------------------------------------------------------------------------------------- frames ----
#: the six spouts in ONE fixed order for every session (dummy coding must mean the same column in every session, or the
#: frozen pre-stroke model would read another spout's column)
POSITIONS = ("close_L", "close_R", "close_center", "far_L", "far_R", "far_center")
NX = 13                                                  # design width (response period; others zero-padded)


def frame_table(sd: Path) -> pd.DataFrame:
    """Per-BIN_MS bin of every trial window: trial_id, t_ms (bin centre, cue-relative), strobe_ms, stop_ms, the six
    traces (bin means; NaN where no valid frame). Tongue / jaw are in the mouth frame (`clean.npz`); the nose is made
    MOUTH-RELATIVE here (raw LP px minus the session's `tongue_origin`), so a different placement of the animal in the
    frame is not read as movement by the FROZEN model (within-session fits are unaffected: a constant)."""
    W = np.load(sd / "windows.npz", allow_pickle=True)
    C = np.load(sd / "clean.npz", allow_pickle=True)
    parts = list(W["parts"])
    nose = W["pose"][:, parts.index("nose"), :]
    mouth = np.asarray(C["tongue_origin"], float)
    nx = np.where(nose[:, 2] >= NOSE_MIN_P, nose[:, 0] - mouth[0], np.nan)
    ny = np.where(nose[:, 2] >= NOSE_MIN_P, nose[:, 1] - mouth[1], np.nan)
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
def trial_design(trials: pd.DataFrame) -> dict:
    """trial_id -> regressor vector per period (module docstring), zero-padded to NX; spouts dummy-coded in POSITIONS
    order (close_L = reference)."""
    P = {p: i for i, p in enumerate(POSITIONS)}
    k = len(POSITIONS) - 1

    def onehot(p):
        v = np.zeros(k)
        i = P.get(p, -1)
        if i > 0:
            v[i - 1] = 1.0
        return v

    def pad(v):
        return np.r_[v, np.zeros(NX - len(v))]
    pos = trials.position.to_numpy()
    hit = trials.hit.to_numpy(float)
    out = {}
    for j, tid in enumerate(trials.trial_id.to_numpy()):
        pp, ph = (pos[j - 1], hit[j - 1]) if j else (None, np.nan)
        prev = np.r_[onehot(pp), ph if np.isfinite(ph) else 0.0]
        cur = onehot(pos[j])
        out[tid] = {"precue": pad(np.r_[1.0, cur, prev]),
                    "response": pad(np.r_[1.0, cur, prev, hit[j]]),
                    "iti": pad(np.r_[1.0, cur, hit[j]])}    # owner = the trial that just ended
    return out


# ------------------------------------------------------------------------------------------------------------- fit ----
TRACES = [f"{p}_{a}" for p in PARTS for a in "xy"]
XCOLS = [f"x{i}" for i in range(NX)]

#: light RIDGE on every regressor but the intercept (units: trials). Plain OLS failed on the first session (PS92_0811):
#: within spout BLOCKS the previous spout equals the current one on most trials, so the combination at a block
#: boundary is barely seen in training and OLS extrapolated it wildly on held-out trials (nose CV "task share" -3.8).
#: The zero-padded columns are penalised too, so every system stays invertible.
RIDGE = 2.0


def _pen(n: int) -> np.ndarray:
    pen = np.full(n, RIDGE)
    pen[0] = 0.0
    return np.diag(pen)


def _stats(X, Y):
    """Sufficient statistics per trace: (A [C, n, n], b [C, n], m [C]) over each trace's finite rows."""
    C = Y.shape[1]
    A = np.zeros((C, X.shape[1], X.shape[1]))
    b = np.zeros((C, X.shape[1]))
    m = np.zeros(C)
    for c in range(C):
        ok = np.isfinite(Y[:, c])
        Xo = X[ok]
        A[c], b[c], m[c] = Xo.T @ Xo, Xo.T @ Y[ok, c], ok.sum()
    return A, b, m


def _predict(st, Xte, ncol=None):
    """Predictions [rows, C] from statistics (task model; ``ncol=1`` = intercept-only); NaN where too few rows."""
    A, b, m = st
    C = A.shape[0]
    k = A.shape[1] if ncol is None else ncol
    out = np.full((len(Xte), C), np.nan)
    for c in range(C):
        if m[c] < MIN_TRAIN:
            continue
        beta = np.linalg.solve(A[c][:k, :k] + _pen(k), b[c][:k])
        out[:, c] = Xte[:, :k] @ beta
    return out


# --------------------------------------------------------------------------------------------------------- session ----
def session_bins(label: str) -> pd.DataFrame | None:
    """Every bin of the session with period / tbin / owner, the six traces, the owner's design row (x0..), working flag,
    CV fold (working trials only; -1 otherwise) and the owner's position / class / block."""
    from sklearn.model_selection import GroupKFold

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
    # PRODUCTION working gate (11-03 late): the run's camera classes (contact + incomplete) AND at or before the
    # session's last anchor trial (`before_session_end`: the quit tail is not working)
    work = tc.in_sel(S.cls.to_numpy()) & tc.before_session_end(S.cls.to_numpy(), S.trial_id.to_numpy(float),
                                                               position=S.position.to_numpy())
    codes = S.position.map({p: i for i, p in enumerate(POSITIONS)}).fillna(-1).astype(int).to_numpy()
    blocks = bi.block_ids(codes)
    wi = np.flatnonzero(work & (blocks >= 0))
    fold = np.full(len(S), -1)
    if len(np.unique(blocks[wi])) >= N_FOLDS:
        for f, (_tr, te) in enumerate(GroupKFold(N_FOLDS).split(wi, groups=blocks[wi])):
            fold[wi[te]] = f
    G = label_periods(frame_table(sd), S[["trial_id"]])
    des = trial_design(S)
    X = np.vstack([des[o][p] for o, p in zip(G.owner.to_numpy(), G.period.to_numpy())])
    G[XCOLS] = X
    meta = pd.DataFrame({"owner": S.trial_id.to_numpy(), "working": work & (blocks >= 0), "fold": fold,
                         "position": S.position.to_numpy(), "cls": S.cls.to_numpy(), "block": blocks})
    G = G.merge(meta, on="owner", how="left")
    G["label"] = f"{animal}_{date}"
    keep = ["label", "owner", "period", "tbin", *TRACES, *XCOLS, "working", "fold", "position", "cls", "block"]
    G = G[keep].astype({c: np.float32 for c in TRACES})
    G["working"] = G.working.fillna(False).astype(bool)
    G["fold"] = G.fold.fillna(-1).astype(int)
    return G.reset_index(drop=True)


def cached_bins(label: str) -> pd.DataFrame | None:
    f = out_dir() / "bins" / f"{label}.pkl"
    if f.exists() and not os.environ.get("TIM_FORCE"):
        return pd.read_pickle(f)
    G = session_bins(label)
    if G is None:
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + f".tmp{os.getpid()}")
    G.to_pickle(tmp)
    os.replace(tmp, f)
    return G


def within_predictions(G: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """REFIT frame: (task, intercept-only) predictions, cross-validated over the session's block folds for working
    trials; other trials from the fit on every working trial."""
    pred = np.full((len(G), len(TRACES)), np.nan)
    mean = np.full_like(pred, np.nan)
    for (_per, _tb), H in G.groupby(["period", "tbin"], sort=False):
        ix = H.index.to_numpy()
        X, Y = H[XCOLS].to_numpy(float), H[TRACES].to_numpy(float)
        w, fo = H.working.to_numpy(bool), H.fold.to_numpy()
        if w.sum() < MIN_TRAIN:
            continue
        st_all = _stats(X[w], Y[w])
        for f in np.unique(fo[w & (fo >= 0)]):
            te = w & (fo == f)
            st = _stats(X[w & (fo != f)], Y[w & (fo != f)])
            pred[ix[te]], mean[ix[te]] = _predict(st, X[te]), _predict(st, X[te], 1)
        rest = ~(w & (fo >= 0))
        if rest.any():
            pred[ix[rest]], mean[ix[rest]] = _predict(st_all, X[rest]), _predict(st_all, X[rest], 1)
    return pred, mean


def session_stats(G: pd.DataFrame) -> dict:
    """(period, tbin) -> sufficient statistics of the session's WORKING bins (the frozen model's training unit)."""
    out = {}
    for key, H in G[G.working].groupby(["period", "tbin"], sort=False):
        out[key] = _stats(H[XCOLS].to_numpy(float), H[TRACES].to_numpy(float))
    return out


def frozen_predictions(G: pd.DataFrame, pooled: dict, own: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """FROZEN frame: predictions from the animal's August-pre statistics ``pooled`` minus this session's own ``own``
    (a pre session is scored leave-one-session-out; ``own`` None for a post session)."""
    pred = np.full((len(G), len(TRACES)), np.nan)
    mean = np.full_like(pred, np.nan)
    for key, H in G.groupby(["period", "tbin"], sort=False):
        if key not in pooled:
            continue
        A, b, m = pooled[key]
        if own is not None and key in own:
            A, b, m = A - own[key][0], b - own[key][1], m - own[key][2]
        ix = H.index.to_numpy()
        X = H[XCOLS].to_numpy(float)
        pred[ix], mean[ix] = _predict((A, b, m), X), _predict((A, b, m), X, 1)
    return pred, mean


def per_trial(G: pd.DataFrame, pred: np.ndarray, mean: np.ndarray, frame: str) -> pd.DataFrame:
    """Per owner trial x period x part: mean Euclidean residual (TIM, px), mean deviation from the bin mean (total, px),
    their squares (for the variance share), bins used, the owner's position / class / working flag."""
    rows = []
    for j, p in enumerate(PARTS):
        x, y = G[f"{p}_x"].to_numpy(float), G[f"{p}_y"].to_numpy(float)
        res = np.hypot(x - pred[:, 2 * j], y - pred[:, 2 * j + 1])
        tot = np.hypot(x - mean[:, 2 * j], y - mean[:, 2 * j + 1])
        D = pd.DataFrame({"trial_id": G.owner.to_numpy(), "period": G.period.to_numpy(), "part": p,
                          "res": res, "tot": tot, "res2": res ** 2, "tot2": tot ** 2})
        D = D[np.isfinite(D.res) & np.isfinite(D.tot)]
        rows.append(D.groupby(["trial_id", "period", "part"]).agg(tim_px=("res", "mean"), total_px=("tot", "mean"),
                                                                  res2=("res2", "mean"), tot2=("tot2", "mean"),
                                                                  n_bins=("res", "size")).reset_index())
    R = pd.concat(rows, ignore_index=True)
    meta = G.drop_duplicates("owner")[["owner", "position", "cls", "working", "block"]].rename(
        columns={"owner": "trial_id"})
    R = R.merge(meta, on="trial_id", how="left")
    R.insert(0, "frame", frame)
    R.insert(0, "label", G.label.iat[0])
    R.insert(1, "animal", G.label.iat[0].split("_")[0])
    return R


def session_tim(label: str) -> pd.DataFrame | None:
    """Within-session (refit) TIM of one session."""
    G = cached_bins(label)
    if G is None:
        return None
    return per_trial(G, *within_predictions(G), "within")


# ----------------------------------------------------------------------------------------------------------- driver ----
def is_august_pre(label: str) -> bool:
    from wfield_local import epochs, run_mode
    return epochs.epoch_of(label) == "pre" and label[-4:] >= run_mode.PRIMARY_PRE_FROM


def _write(R: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    R.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out)


def _within_job(label: str):
    out = out_dir() / "sessions" / f"{label}_within.csv.gz"
    if out.exists() and not os.environ.get("TIM_FORCE"):
        return str(out)
    R = session_tim(label)
    if R is None:
        return None
    _write(R, out)
    return str(out)


def _frozen_job(animal: str):
    """One animal: pooled August-pre statistics, then every session of the animal scored in that frame."""
    labels = [lab for lab in all_labels() if lab.startswith(animal + "_")]
    pre = [lab for lab in labels if is_august_pre(lab)]
    stats = {}
    for lab in pre:
        G = cached_bins(lab)
        if G is not None:
            stats[lab] = session_stats(G)
    if len(stats) < 2:
        return f"{animal}: only {len(stats)} August pre session(s)"
    pooled = {}
    for st in stats.values():
        for key, (A, b, m) in st.items():
            if key in pooled:
                pA, pb, pm = pooled[key]
                pooled[key] = (pA + A, pb + b, pm + m)
            else:
                pooled[key] = (A.copy(), b.copy(), m.copy())
    n = 0
    for lab in labels:
        out = out_dir() / "sessions" / f"{lab}_frozen.csv.gz"
        if out.exists() and not os.environ.get("TIM_FORCE"):
            n += 1
            continue
        G = cached_bins(lab)
        if G is None:
            continue
        R = per_trial(G, *frozen_predictions(G, pooled, stats.get(lab)), "frozen")
        R["frozen_pre"] = ";".join(sorted(stats))
        _write(R, out)
        n += 1
    return f"{animal}: {n} sessions ({len(stats)} pre)"


def all_labels() -> list[str]:
    from wfield_local import analysis_kit as ak
    return sorted(ak.curated_ids())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--frozen", action="store_true", help="also the frozen-pre frame (per animal; needs the pre bins)")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    labels = all_labels() if a.all else a.labels
    from wfield_local import analysis_kit as ak
    res, fails = ak.fan_sessions(labels, _within_job, jobs=a.jobs)
    print(f"within: {len(res)} sessions, {len(fails)} failed -> {out_dir()}")
    if a.frozen:
        animals = sorted({lab.split("_")[0] for lab in labels})
        fres, ffails = ak.fan_sessions(animals, _frozen_job, jobs=min(a.jobs, len(animals)))
        for _k, v in fres:
            print("frozen:", v)
        fails = [*fails, *ffails]
    for f in fails:
        print("  FAIL", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())

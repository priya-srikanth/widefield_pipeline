"""Movement-regressor ENCODING MODELS for widefield activity: a framework (design matrix, grouped ridge, trial-block
cross-validation, variance partitioning, frozen pre-stroke models, movement-removed residuals).

Priya, 2026-10-02: "start building the movement regression code as a framework" (the next DLC / LP iteration comes
before any O2 whole-session run, so this is built and tested on synthetic data first; real inputs plug in through
`MovementInputs`). Design: `docs/DLC_IN_WIDEFIELD_ANALYSES.md` §3E.

WHAT IT GENERALISES. `locanmf_encoding_model` regresses each LocaNMF component's continuous dF/F on time-lagged CUE and
DAQ-LICK indicators (ridge, lags -0.5 .. +1.5 s) -> a cue kernel and a lick kernel. Here any number of regressors, of
three kinds, each in a named GROUP:
  * EventRegressor       an event train (DAQ seconds) -> one kernel over its lags (an FIR: one column per lag).
                         Optional per-event `amplitude` -> a MODULATED kernel (e.g. tongue-onset kernel scaled by the
                         lick's executed angle), centred so it carries only the modulation.
  * ContinuousRegressor  a sampled signal (e.g. tongue protrusion at 250 fps, DAQ-second timestamps) -> averaged into
                         each imaging frame, z-scored with TRAINING statistics, plus copies at a few lags.
  * the intercept        always fitted, never penalised.
Typical groups: "task" (cue kernels per spout position, reward), "lick_events" (tongue onset, contact, jaw onset),
"tongue" / "jaw" (continuous), "state" (running).

FITTING. Ridge per output with a SEPARATE penalty per group, implemented by column scaling (X_g / sqrt(alpha_g) with a
unit penalty == penalty alpha_g on group g). Penalties chosen by cross-validation over a log grid, one group at a time
(coordinate search). Cross-validation folds are contiguous BLOCKS OF TRIALS (frames are assigned to the trial whose
start precedes them), so autocorrelated neighbouring frames never straddle train and test.

WHAT IT REPORTS.
  * cross-validated R^2 per output (component / area), from held-out predictions concatenated over folds;
  * unique variance per group = R^2(full) - R^2(full without the group), and R^2 of the group alone;
  * kernels per regressor (lags x outputs);
  * a frozen model (fit on pre-stroke sessions, `MovementModel`) applied to new sessions -> predictions, R^2, and the
    RESIDUAL after removing chosen groups' prediction (e.g. activity minus its movement-predicted part, for decoding
    the target from what movement cannot explain).
Pure numpy; no I/O. Time bases: everything in DAQ seconds; imaging frames are given by their DAQ-second times.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- regressor specs


@dataclass
class EventRegressor:
    name: str
    times_s: np.ndarray                      # event times, DAQ seconds
    lags_s: tuple[float, float]              # kernel support relative to the event, e.g. (-0.5, 1.5)
    group: str
    amplitude: np.ndarray | None = None      # per-event modulator (None = plain indicator kernel)


@dataclass
class ContinuousRegressor:
    name: str
    t_s: np.ndarray                          # sample times, DAQ seconds (e.g. camera frames)
    values: np.ndarray                       # same length; NaN = unknown (contributes 0 after z-scoring)
    group: str
    lags_s: tuple[float, ...] = (0.0,)       # copies of the signal shifted by these lags (s; + = signal leads)


@dataclass
class MovementInputs:
    """One session's regressors on the DAQ clock + its imaging frame times. ``trial_starts_s`` define CV blocks."""
    frame_times_s: np.ndarray
    regressors: list = field(default_factory=list)
    trial_starts_s: np.ndarray | None = None


@dataclass
class Design:
    X: np.ndarray                            # (T, P) raw (un-standardised) columns
    names: list[str]                         # column labels "regressor@lag"
    regressor: np.ndarray                    # (P,) regressor name per column
    group: np.ndarray                        # (P,) group per column
    lag_s: np.ndarray                        # (P,) lag per column
    kind: np.ndarray                         # (P,) "event" | "continuous"


@dataclass
class MovementModel:
    """A fitted model: everything needed to apply it to a NEW session's design (same regressor set)."""
    beta: np.ndarray                         # (P, n_out) in standardised-column units
    intercept: np.ndarray                    # (n_out,)
    mu: np.ndarray                           # (P,) column means used for standardisation (training)
    sd: np.ndarray                           # (P,) column sds (training)
    names: list[str]
    regressor: np.ndarray
    group: np.ndarray
    lag_s: np.ndarray
    alphas: dict


# --------------------------------------------------------------------------- design matrix


def frame_fs(frame_times_s) -> float:
    return float(1.0 / np.median(np.diff(np.asarray(frame_times_s, float))))


def nearest_frame(times_s, frame_times_s, tol_s: float | None = None) -> np.ndarray:
    """Nearest imaging frame per time; -1 outside the imaging coverage (± ``tol_s``, default one frame period) --
    the `framemap_event_maps.coverage_mask` rule."""
    ft = np.asarray(frame_times_s, float)
    t = np.asarray(times_s, float)
    if ft.size < 2 or t.size == 0:
        return np.full(t.shape, -1, np.int64)
    tol = float(np.median(np.diff(ft))) if tol_s is None else float(tol_s)
    i = np.clip(np.searchsorted(ft, t), 1, len(ft) - 1)
    i = np.where(np.abs(t - ft[i - 1]) <= np.abs(ft[i] - t), i - 1, i)
    ok = (t >= ft[0] - tol) & (t <= ft[-1] + tol) & np.isfinite(t)
    return np.where(ok, i, -1).astype(np.int64)


def bin_to_frames(t_s, values, frame_times_s) -> np.ndarray:
    """Mean of the samples falling in each imaging frame's bin (edges = midpoints between frame times; the outer
    bins extend half a period). NaN samples are ignored; a frame with no finite sample is NaN."""
    ft = np.asarray(frame_times_s, float)
    t = np.asarray(t_s, float)
    v = np.asarray(values, float)
    half = np.median(np.diff(ft)) / 2.0
    edges = np.concatenate([[ft[0] - half], (ft[1:] + ft[:-1]) / 2.0, [ft[-1] + half]])
    k = np.searchsorted(edges, t, side="right") - 1
    ok = (k >= 0) & (k < len(ft)) & np.isfinite(v)
    s = np.bincount(k[ok], weights=v[ok], minlength=len(ft))
    n = np.bincount(k[ok], minlength=len(ft)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """x shifted by k frames (k > 0: value at t comes from t - k, i.e. the signal LEADS the activity), NaN-padded."""
    out = np.full_like(x, np.nan, dtype=float)
    if k == 0:
        out[:] = x
    elif k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def build_design(inputs: MovementInputs) -> Design:
    ft = np.asarray(inputs.frame_times_s, float)
    T, fs = len(ft), frame_fs(ft)
    cols, names, regs, groups, lags, kinds = [], [], [], [], [], []
    for r in inputs.regressors:
        if isinstance(r, EventRegressor):
            f = nearest_frame(r.times_s, ft)
            amp = None if r.amplitude is None else np.asarray(r.amplitude, float)
            if amp is not None:                      # centre the modulator over the events that land on frames
                use = (f >= 0) & np.isfinite(amp)
                amp = amp - (np.mean(amp[use]) if use.any() else 0.0)
            lag_frames = np.arange(int(np.round(r.lags_s[0] * fs)), int(np.round(r.lags_s[1] * fs)) + 1)
            for L in lag_frames:
                col = np.zeros(T)
                ii = f + L
                ok = (f >= 0) & (ii >= 0) & (ii < T)
                if amp is not None:
                    ok &= np.isfinite(amp)
                    np.add.at(col, ii[ok], amp[ok])
                else:
                    np.add.at(col, ii[ok], 1.0)
                cols.append(col)
                names.append(f"{r.name}@{L / fs:+.3f}")
                regs.append(r.name)
                groups.append(r.group)
                lags.append(L / fs)
                kinds.append("event")
        elif isinstance(r, ContinuousRegressor):
            base = bin_to_frames(r.t_s, r.values, ft)
            for lag in r.lags_s:
                L = int(np.round(lag * fs))
                cols.append(_shift(base, L))
                names.append(f"{r.name}@{lag:+.3f}")
                regs.append(r.name)
                groups.append(r.group)
                lags.append(lag)
                kinds.append("continuous")
        else:
            raise TypeError(f"unknown regressor {type(r).__name__}")
    X = np.column_stack(cols) if cols else np.zeros((T, 0))
    return Design(X=X, names=names, regressor=np.array(regs), group=np.array(groups), lag_s=np.array(lags),
                  kind=np.array(kinds))


def standardise(X: np.ndarray, mu=None, sd=None):
    """Column z-scoring with given (training) statistics; NaN -> 0 AFTER centring (= 'unknown contributes nothing').
    Constant columns get sd 1."""
    if mu is None:
        mu = np.nanmean(X, axis=0) if len(X) else np.zeros(X.shape[1])
        sd = np.nanstd(X, axis=0) if len(X) else np.ones(X.shape[1])
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    Z = (X - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0), mu, sd


# --------------------------------------------------------------------------- ridge with per-group penalties


def _penalty_scale(group: np.ndarray, alphas: dict) -> np.ndarray:
    return np.array([1.0 / np.sqrt(float(alphas[g])) for g in group])


def ridge_fit(Z: np.ndarray, Y: np.ndarray, group: np.ndarray, alphas: dict):
    """(beta in Z units, intercept). Penalty alpha_g on group g via column scaling; intercept unpenalised."""
    if Z.shape[1] == 0:
        return np.zeros((0, Y.shape[1])), Y.mean(axis=0)
    s = _penalty_scale(group, alphas)
    zm, ym = Z.mean(axis=0), Y.mean(axis=0)
    Zs = (Z - zm) * s
    A = Zs.T @ Zs
    A[np.diag_indices_from(A)] += 1.0
    b_s = np.linalg.solve(A, Zs.T @ (Y - ym))
    beta = b_s * s[:, None]
    return beta, ym - zm @ beta


def r2_score(Y, P) -> np.ndarray:
    """1 - SS_res / SS_tot per output column (NaN-safe rows)."""
    ok = np.isfinite(Y).all(axis=1) & np.isfinite(P).all(axis=1)
    Yo, Po = Y[ok], P[ok]
    ss_res = ((Yo - Po) ** 2).sum(axis=0)
    ss_tot = ((Yo - Yo.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def trial_block_folds(frame_times_s, trial_starts_s, n_folds: int = 5) -> np.ndarray:
    """Fold label per imaging frame: trials split into ``n_folds`` CONTIGUOUS blocks; a frame belongs to the trial
    whose start precedes it (frames before the first trial -> the first block). Without trial starts: contiguous
    time blocks."""
    ft = np.asarray(frame_times_s, float)
    if trial_starts_s is None or len(trial_starts_s) < n_folds:
        return np.minimum((np.arange(len(ft)) * n_folds) // max(len(ft), 1), n_folds - 1)
    ts = np.sort(np.asarray(trial_starts_s, float))
    trial_of = np.clip(np.searchsorted(ts, ft, side="right") - 1, 0, len(ts) - 1)
    block_of_trial = np.minimum((np.arange(len(ts)) * n_folds) // len(ts), n_folds - 1)
    return block_of_trial[trial_of]


def cv_predict(Z, Y, group, alphas, folds) -> np.ndarray:
    """Held-out predictions, each fold fit on the others (standardisation is already global-safe: Z must come from
    `standardise` on the same session; for cross-SESSION use `fit` + `predict`)."""
    P = np.full(Y.shape, np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        b, c = ridge_fit(Z[tr], Y[tr], group, alphas)
        P[te] = Z[te] @ b + c
    return P


def choose_alphas(Z, Y, group, folds, grid=(0.1, 1.0, 10.0, 100.0, 1000.0, 1e4), n_sweeps: int = 2,
                  init: float = 10.0) -> dict:
    """Per-group penalties by coordinate search over ``grid``, maximising the mean CV R^2 over outputs."""
    groups = list(dict.fromkeys(group.tolist()))
    alphas = {g: float(init) for g in groups}

    def score(a):
        return float(np.nanmean(r2_score(Y, cv_predict(Z, Y, group, a, folds))))

    best = score(alphas)
    for _ in range(n_sweeps):
        for g in groups:
            for a in grid:
                trial = {**alphas, g: float(a)}
                sc = score(trial)
                if sc > best + 1e-9:
                    best, alphas = sc, trial
    return alphas


# --------------------------------------------------------------------------- top-level API


def fit(design: Design, Y: np.ndarray, alphas: dict | None = None, folds=None, grid=None) -> MovementModel:
    """Fit on one (or a stacked set of) session(s). ``alphas`` None -> chosen by CV (needs ``folds``)."""
    Z, mu, sd = standardise(design.X)
    if alphas is None:
        if folds is None:
            raise ValueError("choosing alphas needs folds (trial_block_folds)")
        alphas = choose_alphas(Z, Y, design.group, folds, **({"grid": grid} if grid else {}))
    beta, c = ridge_fit(Z, Y, design.group, alphas)
    return MovementModel(beta=beta, intercept=c, mu=mu, sd=sd, names=list(design.names), regressor=design.regressor,
                         group=design.group, lag_s=design.lag_s, alphas=dict(alphas))


def _check_same_columns(model: MovementModel, design: Design) -> None:
    if list(design.names) != list(model.names):
        raise ValueError("design columns differ from the model's (same regressors, lags and frame rate needed)")


def predict(model: MovementModel, design: Design, groups: list[str] | None = None, intercept: bool = True):
    """Prediction on a (possibly NEW) session with the model's TRAINING standardisation. ``groups`` restricts the
    prediction to those groups' columns (e.g. only movement)."""
    _check_same_columns(model, design)
    Z, _, _ = standardise(design.X, model.mu, model.sd)
    use = np.ones(len(model.names), bool) if groups is None else np.isin(model.group, groups)
    return Z[:, use] @ model.beta[use] + (model.intercept if intercept else 0.0)


def residual(model: MovementModel, design: Design, Y: np.ndarray, remove_groups: list[str]) -> np.ndarray:
    """Y minus the part the model attributes to ``remove_groups`` (no intercept removed) -- e.g. activity with its
    movement-predicted component taken out, for decoding the target from what movement cannot explain."""
    return Y - predict(model, design, groups=remove_groups, intercept=False)


def kernels(model: MovementModel) -> dict:
    """{regressor: (lags_s, weights (n_lags, n_out))} in standardised-column units."""
    out = {}
    for r in dict.fromkeys(model.regressor.tolist()):
        m = model.regressor == r
        order = np.argsort(model.lag_s[m])
        out[r] = (model.lag_s[m][order], model.beta[m][order])
    return out


def variance_partition(design: Design, Y: np.ndarray, folds, alphas: dict, out_names=None,
                       partitions: dict | None = None) -> pd.DataFrame:
    """Cross-validated R^2 per output: full model, each partition alone, the full model without it, and its UNIQUE
    variance (full - without). One row per (output, partition).

    ``partitions`` {label: [groups]} (default: every group on its own). Use COMBINED partitions for the main question
    (e.g. {"task": ["task"], "movement": ["lick_events", "tongue", "jaw"]}): event kernels and continuous signals of
    the SAME movement are largely redundant (a stereotyped lick makes the tongue-onset kernel and the protrusion trace
    carry the same information), so their individual unique variances are small even when movement explains a lot --
    found on the synthetic test, 2026-10-02."""
    Z, _, _ = standardise(design.X)
    groups = list(dict.fromkeys(design.group.tolist()))
    parts = partitions or {g: [g] for g in groups}
    full = r2_score(Y, cv_predict(Z, Y, design.group, alphas, folds))
    names = out_names if out_names is not None else [f"out{k}" for k in range(Y.shape[1])]
    rows = []
    for g, members in parts.items():
        m = np.isin(design.group, members)
        alone = r2_score(Y, cv_predict(Z[:, m], Y, design.group[m], alphas, folds))
        without = (r2_score(Y, cv_predict(Z[:, ~m], Y, design.group[~m], alphas, folds)) if (~m).any()
                   else np.zeros(Y.shape[1]))
        for k, nm in enumerate(names):
            rows.append({"output": nm, "group": g, "r2_full": full[k], "r2_alone": alone[k],
                         "r2_without": without[k], "unique": full[k] - without[k]})
    return pd.DataFrame(rows)

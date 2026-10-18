"""Combined task + movement ENCODING MODEL of pixel-level widefield activity, Musall et al. 2019 (Nat Neurosci) Fig. 4-5
style (Priya 2026-10-18: "better isolate spout position effects in ENL and cue / lick-aligned analyses with a combined
model ... the Musall fig 4 analysis of reduced vs single variable models ... task-aligned and unique contribution maps
and cvR2 / delta R2"; Fig. 5 PETH vs trial-by-trial; executed vs intended angle).

    python -m wfield_local.combined_encoding PS92_0811 [PS93_0821 ...] [--order movement_first|task_first] [--jobs 3]
    python -m wfield_local.combined_encoding --all [--jobs 3]          # every curated session with LP pose + imaging

DATA. Pixel-level, as Musall: the model is fitted to the session's hemodynamically corrected SVD temporal components
(the ADOPTED `hemo_meegkit_hpfit/SVTcorr`, 100 components, 31.25 Hz -- kept at the full rate: tongue movements are
fast) and every R^2 is evaluated PER PIXEL on the shared Allen grid through `U_atlas` (exactly, from the component-
space residual covariance: R^2_pix = 1 - diag(U C_err U') / diag(U C_tot U')), so maps average across animals. Region
values = the same sums restricted to an Allen area. Rows = imaging frames of ENGAGED trials (the cohort's reference-
position gate), position strobe - 0.5 s .. trial stop + 1 s. CV = 10 folds of contiguous trial blocks (Musall: 10-fold).

VARIABLES (group: variable = regressors; event kernels on a 0.1-s raised-cosine basis):
  task         pos_enl   per-position spout-ARRIVAL kernel (strobe, 0..2 s) + per-position SPOUT-PRESENT state
                         (strobe .. cue; the ENL period is 2.5-49 s long because licks reset it)
               pos_cue   per-position CUE kernel (0 .. 2 s)
               reward    reward kernel (-0.2 .. 1.5 s), shared
  instructed   licks     LP lick-onset + DAQ spout-contact kernels (-0.3 .. 0.8 s), shared across positions
               exec_angle lick-onset kernel MODULATED by the lick's executed peak angle minus its position's mean (the
                         within-position deviation; measured peaks, reach >= 60 px) -- "where the tongue went, beyond
                         where the spout was"
               tongue    tongue tip x / y (mouth frame), protrusion, |speed| (250 fps -> frame means; lags 0/0.1/0.2 s)
               jaw       jaw x / y, |speed|
  uninstructed nose      nose x / y relative to the mouth (head / posture)
               running   treadmill speed
Position specificity lives ONLY in the task group: lick kernels are shared, pose is continuous, so "position" =
position-specific activity beyond what the tongue / jaw / body did.

ORTHOGONALISATION (Musall: video vs movement, uninstructed vs instructed, by QR). ``movement_first`` (primary, Priya
10-18): uninstructed is residualised on instructed, task on all movement -> variance shared between movement and
position is credited to MOVEMENT; position's dR^2 is what movement cannot explain (conservative). ``task_first``:
movement residualised on task (sensitivity). The full model's predictions are the same either way.

MEASURES (per variable and per group; windows all / enl (pre-cue 2 s) / cue (0..1 s) / lick (+-0.2 s of a lick peak);
inside a window the held-out residual's window MEAN is removed -- each model gets a free window offset):
  alone      cvR^2 of a model with only that variable (UNORTHOGONALISED) -- upper bound of its linear information
  dR2        full cvR^2 minus cvR^2 with the variable SHUFFLED in the full (orthogonalised) model -- lower bound of
             its unique information. Shuffle = circular shift (>= 60 s) of its columns; for pos_enl / pos_cue the
             position LABELS are permuted across trials (only position SPECIFICITY is removed; the shared
             arrival / cue response stays) -- stated in every figure.
  Fig. 5     task-independent = cvR^2(task + variable) - cvR^2(task alone); task-aligned = alone - task-independent.
             PETH partition: the FULL model fitted once, its prediction split by group weights (no refit), cue-aligned
             per position, per Allen area. Time-resolved SSE (cue-aligned) per model for time-resolved dR^2.
Collinearity: Musall's cumulative subspace angles (|diag R| of the QR of the column-normalised design).

Outputs: <LP_analysis>/<run>/imaging_encoding/<order>/sessions/<label>.npz (+ .json summary). Figures:
`scripts/combined_encoding_figures.py`.
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RUN_ENV = "COMBINED_ENCODING_RUN"     # set by --run; an env var so spawned workers see it (tests/test_parallel_results...)


def _run() -> str:
    """The LP run whose pose / clean / licks feed the movement regressors (MIN_REACH_PX is in px, so the run matters)."""
    import os

    from wfield_local import orofacial_cohort as oc
    return os.environ.get(RUN_ENV) or oc.RUN


FOLDS = 10
BASIS_S = 0.1
LAGS = {"strobe": (0.0, 2.0), "cue": (0.0, 2.0), "reward": (-0.2, 1.5), "lick": (-0.3, 0.8)}
CLAGS = (0.0, 0.1, 0.2)
MIN_REACH_PX = 60.0
SHIFT_MIN_S = 60.0
ALPHA_GRID = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
WINDOWS = ("all", "enl", "cue", "lick")
PETH_S = (-3.0, 3.0)
GROUP_OF = {"pos_enl": "task", "pos_cue": "task", "reward": "task", "licks": "instructed", "exec_angle": "instructed",
            "tongue": "instructed", "jaw": "instructed", "nose": "uninstructed", "running": "uninstructed"}
GROUPS = ("task", "instructed", "uninstructed")
ORDERS = {"movement_first": ("instructed", "uninstructed", "task"),
          "task_first": ("task", "instructed", "uninstructed")}
POS_VARS = ("pos_enl", "pos_cue")


# =============================================================================== inputs


def load_imaging(label: str):
    """(V (T, K) float32 adopted hemo-corrected SVD temporal components, frame DAQ s, U_atlas (P_brain, K) float32,
    brain-mask flat indices, area labels per brain pixel, area names {code: name})."""
    from wfield_local import config
    from wfield_local.hemo_variants import ADOPTED
    from wfield_local.locanmf_lick_aligned import _corrected_frame_samples
    from wfield_local.plot_spout_trial_averages import _load_daq_events
    s = next(x for x in config.load_sessions() if x["label"] == label)
    if s.get("regime") != "B":
        raise ValueError(f"{label}: regime {s.get('regime')} (corrected frame map needed)")
    res = Path(s["mc"]) / "wfield_local_results"
    V = np.load(res / f"hemo_{ADOPTED}" / "SVTcorr.npy").T.astype(np.float32)
    at = res / "allen_aligned_affine8v1"
    U = np.load(at / "U_atlas.npy", mmap_mode="r")
    mask = np.load(at / "allen_brain_mask_native_grid.npy").astype(bool)
    area = np.load(at / "allen_area_atlas_native_grid.npy")
    names = {int(c): n for c, n in json.load(open(at / "allen_area_names.json"))}
    flat = np.flatnonzero(mask.ravel())
    Ub = np.asarray(U.reshape(-1, U.shape[-1])[flat], np.float32)
    cue = _load_daq_events(s["h5"])
    fmdir = s["fmdir"] or s["mc"]
    fm = glob.glob(f"{fmdir}/*cleanpairs_frame_map.npz")[0]
    off = json.loads(Path(glob.glob(f"{fmdir}/*cleanpairs_summary.json")[0]).read_text())["chosen_exposure_offset"]
    ft = np.asarray(_corrected_frame_samples(fm, cue["pco_samples"], int(off)), float) / float(cue["sample_rate_hz"])
    n = min(len(ft), len(V))
    return V[:n], ft[:n], Ub, flat, area.ravel()[flat], names, mask.shape


@dataclass
class Behaviour:
    trials: pd.DataFrame          # trial_id, pos, strobe_s, cue_s, stop_s, engaged
    lick_on_s: np.ndarray
    lick_peak_s: np.ndarray
    lick_pos: np.ndarray
    lick_angle_within: np.ndarray  # executed angle minus position mean; NaN = not usable
    contact_s: np.ndarray
    reward_s: np.ndarray
    analog: dict                   # name -> (t_s, values)
    gate: str


def load_behaviour(animal: str, date: str, rv) -> Behaviour:
    from wfield_local import daq_io, daq_trials
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import trial_windows as TW
    from wfield_local.behavior_events import session_speed
    oc._repo_on_path()
    import scripts.pose_kinematics_demo as D
    lat = "L"
    sd = oo.session_dir(rv, _run(), animal, date)
    keep, gate = oc.engagement_keep(animal, date, sd)
    b = TW.trial_bounds(animal, date, rv)
    cue_t = oc._cue_table(sd).set_index("trial_id")
    T = pd.DataFrame({"trial_id": b.trial_id.astype(int), "pos": [oc.pos_key(p, lat) for p in b.pos_name],
                      "strobe_s": b.strobe_s.to_numpy(float), "cue_s": b.cue_s.to_numpy(float),
                      "stop_s": b.stop_s.to_numpy(float)})
    T["engaged"] = True if keep is None else T.trial_id.isin(keep)
    T = T[T.trial_id.isin(cue_t.index)].reset_index(drop=True)
    L = pd.read_csv(sd / "licks.csv")
    L = L[L.trial_id.isin(cue_t.index)]
    c0 = L.trial_id.map(cue_t.cue_daq).to_numpy(float)
    on_s, pk_s = c0 + L.on_ms.to_numpy(float) / 1000.0, c0 + L.t_ms.to_numpy(float) / 1000.0
    pos = np.array([oc.pos_key(p, lat) for p in L.position])
    ang = L.peak_angle_deg.to_numpy(float).copy()
    ok = (~L.peak_interp.astype(bool).to_numpy()) & (L.protrusion_px.to_numpy(float) >= MIN_REACH_PX) & np.isfinite(ang)
    ang[~ok] = np.nan
    within = ang.copy()
    for p in np.unique(pos):                         # within-position deviation of the executed angle
        m = pos == p
        if np.isfinite(ang[m]).any():
            within[m] = ang[m] - np.nanmean(ang[m])
    contacts = D.daq_contacts_ms(animal, date, b, rv)
    cue_of = dict(zip(b.trial_id.astype(int), b.cue_s.to_numpy(float)))
    contact_s = np.concatenate([cue_of[t] + np.asarray(v, float) / 1000.0 for t, v in contacts.items()
                                if t in cue_of and len(v)]) if contacts else np.zeros(0)
    h5 = daq_io.session_h5(animal, date, rv)
    reward_s = np.asarray(daq_trials.decode(h5)["reward_s"], float)
    # analog pose on the pose rows (DAQ s): tongue / jaw cleaned (mouth- / rest-relative), nose relative to the mouth
    S = oo.load_session(rv, _run(), animal, date)
    t_pose = S["index"].daq_s.to_numpy(float)
    cl, pose = S["clean"], S["pose"]
    o = np.load(sd / "clean.npz")["tongue_origin"].astype(float)
    tx, ty = cl.tongue_x.to_numpy(float), cl.tongue_y.to_numpy(float)
    prot = np.hypot(tx, ty)
    jx, jy = cl.jaw_x.to_numpy(float), cl.jaw_y.to_numpy(float)
    nlik = pose.nose_likelihood.to_numpy(float) >= 0.6
    nx = np.where(nlik, pose.nose_x.to_numpy(float) - o[0], np.nan)
    ny = np.where(nlik, pose.nose_y.to_numpy(float) - o[1], np.nan)
    fps = 250.0
    sp = lambda v: np.abs(np.gradient(v)) * fps                                   # noqa: E731
    run_t, run_v = session_speed(h5)
    analog = {"tongue_x": (t_pose, tx), "tongue_y": (t_pose, ty), "tongue_prot": (t_pose, prot),
              "tongue_speed": (t_pose, sp(prot)), "jaw_x": (t_pose, jx), "jaw_y": (t_pose, jy),
              "jaw_speed": (t_pose, np.hypot(np.gradient(jx), np.gradient(jy)) * fps),
              "nose_x": (t_pose, nx), "nose_y": (t_pose, ny), "running": (np.asarray(run_t, float),
                                                                          np.asarray(run_v, float))}
    return Behaviour(T, on_s, pk_s, pos, within, np.asarray(contact_s, float), reward_s, analog, gate)


# =============================================================================== design


def _event_cols(times, ft, lags, amp=None) -> np.ndarray:
    from wfield_local import movement_encoding as ME
    r = ME.EventRegressor("e", np.asarray(times, float), lags, "g", amplitude=amp, basis_spacing_s=BASIS_S)
    return ME.build_design(ME.MovementInputs(ft, [r])).X.astype(np.float32)


def _state_cols(starts, stops, ft) -> np.ndarray:
    col = np.zeros(len(ft), np.float32)
    for a, b in zip(starts, stops):
        if np.isfinite(a) and np.isfinite(b) and b > a:
            col[(ft >= a) & (ft < b)] = 1.0
    return col[:, None]


def _analog_cols(t, v, ft) -> np.ndarray:
    from wfield_local import movement_encoding as ME
    base = ME.bin_to_frames(t, v, ft)
    fs = ME.frame_fs(ft)
    return np.column_stack([ME._shift(base, int(round(lag * fs))) for lag in CLAGS]).astype(np.float32)


def raw_blocks(B: Behaviour, ft, pos_perm: dict | None = None) -> dict:
    """{variable: (T, p) raw columns}. ``pos_perm`` {variable: permuted position per trial} for the label shuffle."""
    T = B.trials
    pos_order = sorted(T.pos.unique())
    out = {}
    for var in POS_VARS:
        lab = T.pos.to_numpy() if not pos_perm or var not in pos_perm else pos_perm[var]
        cols = []
        for p in pos_order:
            m = lab == p
            if var == "pos_enl":
                cols.append(_event_cols(T.strobe_s[m], ft, LAGS["strobe"]))
                cols.append(_state_cols(T.strobe_s[m], T.cue_s[m], ft))
            else:
                cols.append(_event_cols(T.cue_s[m], ft, LAGS["cue"]))
        out[var] = np.hstack(cols)
    out["reward"] = _event_cols(B.reward_s, ft, LAGS["reward"])
    out["licks"] = np.hstack([_event_cols(B.lick_on_s, ft, LAGS["lick"]), _event_cols(B.contact_s, ft, LAGS["lick"])])
    out["exec_angle"] = _event_cols(B.lick_on_s, ft, LAGS["lick"], amp=B.lick_angle_within)
    out["tongue"] = np.hstack([_analog_cols(*B.analog[k], ft) for k in ("tongue_x", "tongue_y", "tongue_prot",
                                                                         "tongue_speed")])
    out["jaw"] = np.hstack([_analog_cols(*B.analog[k], ft) for k in ("jaw_x", "jaw_y", "jaw_speed")])
    out["nose"] = np.hstack([_analog_cols(*B.analog[k], ft) for k in ("nose_x", "nose_y")])
    out["running"] = _analog_cols(*B.analog["running"], ft)
    return out


def _z(X, rows):
    """z-score with fit-row statistics; NaN -> 0 (unknown contributes nothing); constant columns -> 0."""
    Xr = X[rows]
    mu = np.nanmean(Xr, 0)
    sd = np.nanstd(Xr, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, np.inf)
    Z = (X - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)


def _residualise(X, E, rows):
    """X minus its least-squares projection on E (fit rows), re-z-scored (Musall's QR orthogonalisation)."""
    if E is None or E.shape[1] == 0:
        return X
    Er, Xr = E[rows].astype(np.float64), X[rows].astype(np.float64)
    G = Er.T @ Er + 1e-6 * np.eye(E.shape[1])
    beta = np.linalg.solve(G, Er.T @ Xr)
    return _z((X - E @ beta.astype(np.float32)), rows)


# =============================================================================== Gram engine


class Engine:
    """Per-fold / per-window sufficient statistics of a column universe A (rows = fit frames) and Y."""

    def __init__(self, A: np.ndarray, Y: np.ndarray, fold: np.ndarray, win: dict, chunk: int = 20000):
        self.P, self.K = A.shape[1] + 1, Y.shape[1]
        nf = int(fold.max()) + 1
        self.G = np.zeros((nf, self.P, self.P))
        self.b = np.zeros((nf, self.P, self.K))
        self.Gw = {w: np.zeros((nf, self.P, self.P)) for w in win}
        self.bw = {w: np.zeros((nf, self.P, self.K)) for w in win}
        self.YYw = {w: np.zeros((nf, self.K, self.K)) for w in win}
        self.sYw = {w: np.zeros((nf, self.K)) for w in win}
        self.sZw = {w: np.zeros((nf, self.P)) for w in win}
        self.nw = {w: np.zeros(nf) for w in win}
        for s in range(0, len(Y), chunk):
            a = np.hstack([np.ones((min(chunk, len(Y) - s), 1), np.float32), A[s:s + chunk]])
            y, f = Y[s:s + chunk], fold[s:s + chunk]
            for k in np.unique(f):
                m = f == k
                ak, yk = a[m], y[m]
                self.G[k] += (ak.T @ ak).astype(np.float64)
                self.b[k] += (ak.T @ yk).astype(np.float64)
                for w, wm in win.items():
                    mm = wm[s:s + chunk][m]
                    if mm.any():
                        aw, yw = ak[mm], yk[mm]
                        self.Gw[w][k] += (aw.T @ aw).astype(np.float64)
                        self.bw[w][k] += (aw.T @ yw).astype(np.float64)
                        self.YYw[w][k] += (yw.T @ yw).astype(np.float64)
                        self.sYw[w][k] += yw.sum(0)
                        self.sZw[w][k] += aw.sum(0)
                        self.nw[w][k] += mm.sum()
        self.Gt, self.bt = self.G.sum(0), self.b.sum(0)

    def betas(self, idx, pen):
        """Per-fold ridge solutions for universe columns ``idx`` (intercept added, unpenalised); ``pen`` per idx."""
        ii = np.r_[0, np.asarray(idx) + 1]
        lam = np.diag(np.r_[0.0, pen])
        out = []
        for k in range(self.G.shape[0]):
            Gtr = self.Gt[np.ix_(ii, ii)] - self.G[k][np.ix_(ii, ii)]
            btr = self.bt[ii] - self.b[k][ii]
            out.append(np.linalg.solve(Gtr + lam + 1e-8 * np.eye(len(ii)), btr))
        return ii, out

    def err_cov(self, idx, pen, w, fitted=None):
        """Summed held-out residual covariance (K, K) in window w, and the window's total covariance."""
        ii, B = fitted if fitted is not None else self.betas(idx, pen)
        E = np.zeros((self.K, self.K))
        se = np.zeros(self.K)
        for k, bk in enumerate(B):
            G, b = self.Gw[w][k][np.ix_(ii, ii)], self.bw[w][k][ii]
            E += self.YYw[w][k] - bk.T @ b - b.T @ bk + bk.T @ G @ bk
            se += self.sYw[w][k] - bk.T @ self.sZw[w][k][ii]
        n = self.nw[w].sum()
        mu = self.sYw[w].sum(0) / max(n, 1)
        Tc = self.YYw[w].sum(0) - n * np.outer(mu, mu)
        # WINDOW OFFSET FREE (2026-10-18): every model is fitted to whole trials but scored inside a window; without
        # this, a model that cannot track the window's MEAN level (running alone in the cue window) scored R^2 -0.6
        # and shuffled-model differences exceeded the full model's R^2. One common offset per component per window.
        E = E - np.outer(se, se) / max(n, 1)
        return E, Tc

    def full_beta(self, idx, pen):
        ii = np.r_[0, np.asarray(idx) + 1]
        lam = np.diag(np.r_[0.0, pen])
        return ii, np.linalg.solve(self.Gt[np.ix_(ii, ii)] + lam + 1e-8 * np.eye(len(ii)), self.bt[ii])


def _diag_quad(U, C, chunk=50000):
    """diag(U C U') for U (P, K)."""
    out = np.empty(len(U))
    for s in range(0, len(U), chunk):
        u = U[s:s + chunk].astype(np.float64)
        out[s:s + chunk] = np.einsum("ij,ij->i", u @ C, u)
    return out


# =============================================================================== one session


def run_session(label: str, order: str = "movement_first", seed: int = 0) -> dict:
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal = config.animal_of(label)
    date = "2026" + label.split("_")[-1]
    rng = np.random.default_rng(seed)
    V, ft, Ub, flat, parea, names, shape = load_imaging(label)
    B = load_behaviour(animal, date, rv)
    T = B.trials
    E = T[T.engaged & np.isfinite(T.cue_s)].reset_index(drop=True)
    st = np.where(np.isfinite(E.strobe_s), E.strobe_s, E.cue_s - 3.0)
    rows = np.zeros(len(ft), bool)
    tr_of = np.full(len(ft), -1)
    for k, (a, b_) in enumerate(zip(st - 0.5, E.stop_s.to_numpy(float) + 1.0)):
        m = (ft >= a) & (ft <= b_)
        rows |= m
        tr_of[m] = k
    ridx = np.flatnonzero(rows)
    # folds = contiguous trial blocks
    blk = np.array_split(np.arange(len(E)), FOLDS)
    f_of_trial = np.zeros(len(E), int)
    for k, bl in enumerate(blk):
        f_of_trial[bl] = k
    fold = f_of_trial[tr_of[ridx]]
    # windows
    ftr = ft[ridx]
    win = {"all": np.ones(len(ridx), bool), "enl": np.zeros(len(ridx), bool), "cue": np.zeros(len(ridx), bool),
           "lick": np.zeros(len(ridx), bool)}
    for s_, c in zip(st, E.cue_s.to_numpy(float)):
        win["enl"] |= (ftr >= max(s_, c - 2.0)) & (ftr < c)
        win["cue"] |= (ftr >= c) & (ftr < c + 1.0)
    pk = np.sort(B.lick_peak_s)
    j = np.searchsorted(pk, ftr)
    d = np.minimum(np.abs(ftr - pk[np.clip(j - 1, 0, len(pk) - 1)]), np.abs(ftr - pk[np.clip(j, 0, len(pk) - 1)]))
    win["lick"] = d <= 0.2
    # raw blocks -> z -> orthogonalised (order) -> universe
    raw = {v: _z(X, ridx)[ridx] for v, X in raw_blocks(B, ft).items()}
    perm = {v: rng.permutation(E.pos.to_numpy()) for v in POS_VARS}
    full_perm = T.pos.to_numpy().copy()
    rawp = {}
    for v in POS_VARS:                                      # label-permuted position blocks (engaged trials permuted)
        lab = full_perm.copy()
        eng_idx = np.flatnonzero(T.engaged.to_numpy() & np.isfinite(T.cue_s.to_numpy()))
        lab[eng_idx] = perm[v]
        rawp[v] = _z(raw_blocks(B, ft, {v: lab})[v], ridx)[ridx]
    seq = ORDERS[order]
    orth, earlier = {}, None
    for g in seq:
        vs = [v for v in GROUP_OF if GROUP_OF[v] == g]
        Xg = np.hstack([raw[v] for v in vs])
        Xo = _residualise(Xg, earlier, slice(None)) if earlier is not None else Xg
        c = 0
        for v in vs:
            orth[v] = Xo[:, c:c + raw[v].shape[1]]
            c += raw[v].shape[1]
        earlier = Xg if earlier is None else np.hstack([earlier, Xg])
    # shuffled orthogonalised blocks
    shift_min = int(SHIFT_MIN_S / np.median(np.diff(ft)))
    shuf = {}
    for v in GROUP_OF:
        if v in POS_VARS:
            gi = seq.index("task")
            prior = [w for w in GROUP_OF if seq.index(GROUP_OF[w]) < gi]
            Epr = np.hstack([raw[w] for w in prior]) if prior else None
            shuf[v] = _residualise(rawp[v], Epr, slice(None)) if Epr is not None else rawp[v]
        else:
            s_ = int(rng.integers(shift_min, max(shift_min + 1, len(ridx) - shift_min)))
            shuf[v] = np.roll(orth[v], s_, axis=0)
    # universe layout
    blocks, layout = [], {}

    def add(key, X):
        layout[key] = np.arange(sum(b.shape[1] for b in blocks), sum(b.shape[1] for b in blocks) + X.shape[1])
        blocks.append(X)
    for v in GROUP_OF:
        add(("orth", v), orth[v])
        add(("shuf", v), shuf[v])
        if GROUP_OF[v] != seq[0]:                       # orthogonalised groups also keep their raw columns
            add(("raw", v), raw[v])
        else:
            layout[("raw", v)] = layout[("orth", v)]
    A = np.hstack(blocks)
    Y = V[ridx]
    eng = Engine(A, Y, fold, win)
    grp_of_col = np.empty(A.shape[1], object)
    for (_kind, v), ix in layout.items():
        grp_of_col[ix] = GROUP_OF[v]

    def cols(spec):
        """spec: list of (kind, var) -> universe column indices."""
        return np.concatenate([layout[s] for s in spec])
    # penalties: per group, coordinate search on the full model's held-out R^2 (all window, summed components)
    full_spec = [("orth", v) for v in GROUP_OF]
    alphas = {g: 100.0 for g in GROUPS}

    def score(spec, al, w="all"):
        ix = cols(spec)
        pen = np.array([al[grp_of_col[i]] for i in ix])
        Ecov, Tc = eng.err_cov(ix, pen, w)
        return 1.0 - np.trace(Ecov) / np.trace(Tc)
    for _ in range(2):
        for g in GROUPS:
            best = max(ALPHA_GRID, key=lambda a: score(full_spec, {**alphas, g: a}))
            alphas[g] = best

    # 2x2-pixel blocks for the stored maps (sums of SSE / SST per block -> exact block R^2)
    rr, cc = np.unravel_index(flat, shape)
    blk_id_full = (rr // 2) * ((shape[1] + 1) // 2) + (cc // 2)
    ublk, blk = np.unique(blk_id_full, return_inverse=True)

    def bsum(x):
        return np.bincount(blk, weights=x, minlength=len(ublk))
    # models
    models = {"full": full_spec}
    for v in GROUP_OF:
        models[f"alone:{v}"] = [("raw", v)]
        models[f"shuf:{v}"] = [("shuf", v) if w == v else ("orth", w) for w in GROUP_OF]
    for g in GROUPS:
        vs = [v for v in GROUP_OF if GROUP_OF[v] == g]
        models[f"alone:{g}"] = [("raw", v) for v in vs]
        models[f"shuf:{g}"] = [("shuf", w) if GROUP_OF[w] == g else ("orth", w) for w in GROUP_OF]
    task_raw = [("raw", v) for v in GROUP_OF if GROUP_OF[v] == "task"]
    for v in GROUP_OF:
        if GROUP_OF[v] != "task":
            models[f"task+:{v}"] = task_raw + [("raw", v)]
    for g in ("instructed", "uninstructed"):
        models[f"task+:{g}"] = task_raw + [("raw", v) for v in GROUP_OF if GROUP_OF[v] == g]
    # evaluate: per model x window -> pixel SSE (as e), shared total t per window
    areas = sorted({int(a) for a in np.unique(parea) if a != 0})
    res = {"label": label, "animal": animal, "date": date, "epoch": epochs.epoch_of(label),
           "day": epochs.days_since_stroke(label), "order": order, "alphas": alphas, "gate": B.gate,
           "n_trials": int(len(E)), "n_frames": int(len(ridx)), "fs": float(1 / np.median(np.diff(ft)))}
    maps, region = {}, []
    tot_pix = {}
    amask = {a: parea == a for a in areas}
    for name, spec in models.items():
        ix = cols(spec)
        pen = np.array([alphas[grp_of_col[i]] for i in ix])
        fitted = eng.betas(ix, pen)
        for w in WINDOWS:
            Ecov, Tc = eng.err_cov(ix, pen, w, fitted)
            e = _diag_quad(Ub, Ecov)
            if ("TOTAL", w) not in maps:
                t = _diag_quad(Ub, Tc)
                maps[("TOTAL", w)] = bsum(t)
                tot_pix[w] = t
            t = tot_pix[w]
            maps[(name, w)] = bsum(e)
            region.append({"model": name, "window": w, "area": "cortex", "r2": 1 - e.sum() / t.sum()})
            for a, m in amask.items():
                region.append({"model": name, "window": w, "area": names.get(a, str(a)),
                               "r2": 1 - e[m].sum() / t[m].sum()})
    # collinearity: cumulative subspace angles of the full design (normalised columns)
    ixf = cols(full_spec) + 1
    Gn = eng.Gt[np.ix_(ixf, ixf)]
    dsc = np.sqrt(np.clip(np.diag(Gn), 1e-12, None))
    Gc = Gn / np.outer(dsc, dsc) + 1e-10 * np.eye(len(ixf))
    try:
        Lc = np.linalg.cholesky(Gc)
        csa = np.abs(np.diag(Lc))
    except np.linalg.LinAlgError:
        csa = np.full(len(ixf), np.nan)
    res["collinearity_min"] = {v: float(np.nanmin(csa[np.isin(ixf - 1, layout[("orth", v)])])) for v in GROUP_OF}
    # Fig. 5: full model fitted ONCE, prediction split by group; cue-aligned per position, per area (+ cortex)
    ix = cols(full_spec)
    pen = np.array([alphas[grp_of_col[i]] for i in ix])
    ii, beta = eng.full_beta(ix, pen)
    Afull = A[:, ii[1:] - 1]
    gcol = grp_of_col[ix]
    Ra = np.stack([Ub[parea == a].mean(0) for a in areas] + [Ub.mean(0)], 1)       # (K, n_areas + cortex)
    comp = {"data": Y.astype(np.float64), "full": Afull @ beta[1:] + beta[0]}
    for g in GROUPS:
        comp[g] = Afull[:, gcol == g] @ beta[1:][gcol == g]
    off = np.arange(int(PETH_S[0] * res["fs"]), int(PETH_S[1] * res["fs"]) + 1)
    cue_f = np.searchsorted(ftr, E.cue_s.to_numpy(float))
    pos_order = sorted(E.pos.unique())
    peth = np.full((len(comp), len(pos_order), Ra.shape[1], len(off)), np.nan, np.float32)
    for ci, Cm in enumerate(comp.values()):
        R = (Cm @ Ra).astype(np.float32)
        for pi, p in enumerate(pos_order):
            idx = cue_f[(E.pos == p).to_numpy()][:, None] + off[None, :]
            ok = (idx >= 0) & (idx < len(R))
            seg = np.where(ok[..., None], R[np.clip(idx, 0, len(R) - 1)], np.nan)
            peth[ci, pi] = np.nanmean(seg, 0).T
    # time-resolved held-out SSE (cue-aligned, all trials) per group model, cortex + areas
    tr_models = ["full", "alone:task", "alone:instructed", "alone:uninstructed",
                 "shuf:task", "shuf:instructed", "shuf:uninstructed", "shuf:pos_enl", "shuf:pos_cue",
                 "shuf:exec_angle"]
    Rd = Y.astype(np.float64) @ Ra
    sse = np.zeros((len(tr_models), Ra.shape[1], len(off)))
    sst = np.zeros((Ra.shape[1], len(off)))
    idx = cue_f[:, None] + off[None, :]
    okm = (idx >= 0) & (idx < len(Rd))
    idc = np.clip(idx, 0, len(Rd) - 1)
    mu_t = np.nanmean(np.where(okm[..., None], Rd[idc], np.nan), 0)              # (off, areas)
    sst = np.nansum(np.where(okm[..., None], (Rd[idc] - Rd.mean(0)) ** 2, np.nan), 0).T
    for mi, mname in enumerate(tr_models):
        spec = models[mname]
        ixm = cols(spec)
        penm = np.array([alphas[grp_of_col[i]] for i in ixm])
        iim, Bk = eng.betas(ixm, penm)
        pred = np.zeros_like(Rd)
        Am = A[:, iim[1:] - 1]
        for k, bk in enumerate(Bk):
            m = fold == k
            pred[m] = (Am[m] @ bk[1:] + bk[0]) @ Ra
        sse[mi] = np.nansum(np.where(okm[..., None], (Rd[idc] - pred[idc]) ** 2, np.nan), 0).T
    res.update({"tr_models": tr_models, "pos_order": pos_order, "peth_parts": list(comp),
                "areas": [names.get(a, str(a)) for a in areas] + ["cortex"]})
    del mu_t
    return {"res": res, "maps": maps, "flat": ublk, "shape": ((shape[0] + 1) // 2, (shape[1] + 1) // 2),
            "region": pd.DataFrame(region),
            "peth": peth, "peth_off_s": off / res["fs"], "sse": sse, "sst": sst}


def out_dir(rv, order: str) -> Path:
    from wfield_local import orofacial_outputs as oo
    return oo.root(rv) / _run() / "imaging_encoding" / order


def save(out: dict, d: Path) -> None:
    from wfield_local import writeguard
    (d / "sessions").mkdir(parents=True, exist_ok=True)
    writeguard.assert_writable(d)
    lab = out["res"]["label"]
    keys = sorted(out["maps"])
    np.savez_compressed(d / "sessions" / f"{lab}.npz",
                        map_keys=np.array([f"{a}|{b}" for a, b in keys]),
                        maps=np.stack([out["maps"][k] for k in keys]).astype(np.float32),
                        flat=out["flat"], shape=np.array(out["shape"]), peth=out["peth"],
                        peth_off_s=out["peth_off_s"], sse=out["sse"], sst=out["sst"])
    out["region"].assign(label=lab).to_csv(d / "sessions" / f"{lab}_regions.csv", index=False, float_format="%.5f")
    (d / "sessions" / f"{lab}.json").write_text(json.dumps(out["res"], indent=1, default=str), encoding="utf-8")


def _worker(job):
    from wfield_local.orofacial_cohort import _repo_on_path
    _repo_on_path()
    label, order = job
    from wfield_local.paths import PathResolver
    d = out_dir(PathResolver(), order)
    out = run_session(label, order)
    save(out, d)
    r = out["region"]
    full = r[(r.model == "full") & (r.window == "all") & (r.area == "cortex")].r2.iloc[0]
    return {"label": label, "full_cvR2": float(full), "n_frames": out["res"]["n_frames"]}


def sessions_available() -> list[str]:
    from wfield_local import analysis_kit as ak
    from wfield_local import config
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    out = []
    for s in ak.curated_sessions(order="sorted"):
        a = config.animal_of(s["label"])
        date = "2026" + s["label"].split("_")[-1]
        if (oo.session_dir(rv, _run(), a, date) / "manifest.json").exists() and s.get("regime") == "B":
            out.append(s["label"])
    return out


def main(argv=None) -> int:
    from wfield_local import parallel
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--order", choices=tuple(ORDERS), default="movement_first")
    ap.add_argument("--jobs", type=int, default=3, help="sessions in parallel (each ~3-4 GB)")
    ap.add_argument("--run", default=None, help="LP run (default: orofacial_cohort.RUN); outputs go under it")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        import os
        os.environ[RUN_ENV] = a.run
    from wfield_local.paths import PathResolver
    labels = sessions_available() if a.all else a.labels
    if a.skip_done:
        d = out_dir(PathResolver(), a.order) / "sessions"
        labels = [x for x in labels if not (d / f"{x}.json").exists()]
    res, fail = parallel.fan_out([(x, a.order) for x in labels], _worker, jobs=a.jobs, label="session")
    for _it, r in sorted(res, key=lambda r: r[0][0]):
        print(f"  {r['label']}: full cvR2 (cortex, all frames) = {r['full_cvR2']:.3f}  ({r['n_frames']} frames)")
    for it, err in fail:
        print(f"  FAILED {it[0]}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

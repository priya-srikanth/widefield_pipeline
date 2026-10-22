"""FROZEN pre-stroke encoder in Allen-area space (Priya 10-22: "yes - and do frozen cue-aligned and lick-aligned also").

`combined_encoding` fits every session on its own (kernels, penalties, z-scoring and the movement-first
orthogonalisation all per session), so its split of an acute change into task / movement parts cannot tell FEWER /
SMALLER movements from a CHANGED cortical response to the same movement. Here one model per animal is fitted on its
August pre-stroke sessions and applied, unchanged, to every session:

- regressors: the encoder's full-model variables (`combined_encoding.GROUP_OF`: cue, arrival, the six position
  deviation blocks, lick kernels, lick angle, tongue, jaw, nose, running) from `raw_blocks`, in RAW units, slow drift
  projected out per session exactly as the encoder does (`_slow_basis`, DRIFT_MIN_PERIOD_S; the constant is in the
  basis, so every column and target is centred per session);
- SCALING: each column divided by the animal's POOLED pre-stroke SD (not the session's own), so a post-stroke session
  with smaller / rarer licks keeps smaller regressors -- the point of freezing. Pose traces are in LP pixels on a fixed
  rig (v36 timing / June rescale), so their units are comparable across sessions;
- NO ORTHOGONALISATION: all groups are fitted jointly (one ridge, the encoder's penalty groups `PEN_GROUP_OF`), so no
  group is credited by ordering. The per-session encoder residualises task on movement (movement_first), a second
  reason it credits movement; this model does not;
- target: area activity Rd = Y @ Ra (the encoder's area averages on the eroded stat mask, + "cortex"), per area name;
  areas absent from any of the animal's sessions are dropped for that animal;
- penalties: per penalty group, coordinate search (2 passes over ALPHA_GRID, as the encoder) on LEAVE-ONE-PRE-SESSION-
  OUT R^2 summed over areas; pre sessions are then scored with the model fitted WITHOUT them, post sessions with the
  model fitted on all of the animal's August pre sessions.

Per session the output matches `combined_encoding`'s session files where the decomposition reads them: `peth` (cue-
aligned) and `peth_lick` (first post-cue lick), parts data / full / task / instructed / uninstructed / pos, x position
x area x offset (PETH_S), plus the frozen model's R^2 (all / cue / lick windows, cortex). `data - full` = activity the
FROZEN model does not predict: at acute far contra, negative = cortex responds LESS than the pre-stroke mapping
predicts from the movements actually made.

Usage (same env as the encoder run whose trials / licks it should use, e.g. OROFACIAL_GATE=attempted
COMBINED_ENCODING_LICKS=combined): python -m wfield_local.frozen_encoder --all --jobs 6 [--tag gate-attempted]
Output: <E>/frozen_encoder_<tag>/<variant>/sessions/<label>.{npz,json}, fit/<animal>_<variant>.npz,
fit/pre_stats.pkl.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

STATS_VERSION = 3   # 2: FGROUP split (contact / direction / movement); 3: protrusion -> direction


def _ce():
    from wfield_local import combined_encoding as ce
    return ce


#: FROZEN-MODEL GROUPS (Priya 10-22, "yes change as you suggest"): finer than the encoder's task / instructed /
#: uninstructed, because two things the encoder lumps into "instructed" are not movement magnitude:
#:   contact    the spout-CONTACT kernel (inside the encoder's `licks` block): a touch / outcome event that happens only
#:              when the lick reaches the target -- acute far contra has few, so its missing contact-evoked activity
#:              would otherwise be credited to "movement";
#:   direction  WHERE the tongue / jaw / nose go, on either axis of position: tongue x / y, tongue PROTRUSION (reach
#:              magnitude -- unsigned but it scales with spout DISTANCE, the other position axis; 10-22 audit, Priya
#:              "ensure our variable groups appropriately include or exclude ... directionless ... vs direction-
#:              specific"), jaw x, nose x, per-lick executed angle (within-spout deviation, zero mean per spout):
#:              target-correlated, since the tongue goes toward the spout (the 10-19 no-kinematics reasoning);
#:   movement   DIRECTIONLESS timing and vigour only: lick onsets (cam4 + cam1-only), tongue speed (unsigned), jaw
#:              opening (y) and speed (unsigned), nose y (up / down orienting), running. Residual target information
#:              in these (e.g. wider opening for far spouts) is MEASURED, not assumed: `scripts/frozen_group_leakage.py`;
#:   task       cue, arrival, the six position deviation blocks.
FGROUP = {"cue": "task", "arrival": "task", "pos_enl": "task", "pos_cue": "task",
          "lick_onset": "movement", "tongue_mag": "movement", "jaw_mag": "movement", "nose_mag": "movement",
          "running": "movement",
          "tongue_dir": "direction", "jaw_dir": "direction", "nose_dir": "direction", "exec_angle": "direction",
          "contact": "contact"}
FGROUPS = ("task", "movement", "direction", "contact")
#: penalty groups as the encoder's PEN_GROUP_OF (task / lick / orofacial / uninstructed)
FPEN = {"cue": "task", "arrival": "task", "pos_enl": "task", "pos_cue": "task",
        "lick_onset": "lick", "contact": "lick", "exec_angle": "lick",
        "tongue_mag": "orofacial", "tongue_dir": "orofacial", "jaw_mag": "orofacial", "jaw_dir": "orofacial",
        "nose_mag": "uninstructed", "nose_dir": "uninstructed", "running": "uninstructed"}


def split_blocks(raw: dict) -> dict:
    """The encoder's raw blocks split into the FGROUP variables. Column order inside each encoder block is fixed by
    `combined_encoding.raw_blocks`: licks = [lick-onset kernel, contact kernel] (equal widths, same lags); tongue =
    x, y, prot, speed; jaw = x, y, speed; nose = x, y; each analog variable len(CLAGS) columns."""
    ce = _ce()
    if os.environ.get("COMBINED_ENCODING_LICKS") == "lips" or os.environ.get("COMBINED_ENCODING_LICKS_ALL"):
        raise ValueError("frozen_encoder: lips / licks-all kernels change the licks block layout; not supported")
    nl = len(ce.CLAGS)
    lk = raw["licks"]
    if lk.shape[1] % 2:
        raise ValueError(f"licks block has {lk.shape[1]} columns; expected [onset, contact] of equal width")
    h = lk.shape[1] // 2
    out = {v: raw[v] for v in ("cue", "arrival", "pos_enl", "pos_cue", "running", "exec_angle")}
    out["lick_onset"], out["contact"] = lk[:, :h], lk[:, h:]
    out["tongue_dir"], out["tongue_mag"] = raw["tongue"][:, :3 * nl], raw["tongue"][:, 3 * nl:]   # x, y, prot | speed
    out["jaw_dir"], out["jaw_mag"] = raw["jaw"][:, :nl], raw["jaw"][:, nl:]
    out["nose_dir"], out["nose_mag"] = raw["nose"][:, :nl], raw["nose"][:, nl:]
    for k, w in (("tongue_mag", 1), ("jaw_mag", 2), ("nose_mag", 1), ("tongue_dir", 3), ("jaw_dir", 1),
                 ("nose_dir", 1)):
        if out[k].shape[1] != w * nl:
            raise ValueError(f"{k}: {out[k].shape[1]} columns, expected {w * nl}")
    return out


def session_design(label: str) -> dict:
    """Raw (drift-removed, session-centred) design X, area targets Rd, column variables, trial alignment frames."""
    ce = _ce()
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = config.animal_of(label), "2026" + label.split("_")[-1]
    V, ft, Ub, _flat, parea, names, _shape, pstat = ce.load_imaging(label)
    try:
        B = ce.load_behaviour(animal, date, rv)
    except FileNotFoundError as ex:
        # NO cam1 LICK EVENTS for this session (PS95 all, PS94_1005; cam1 unfinished): cam4 lick timing, as LP_licks
        # and the per-session encoder runs do; recorded as lick_set "cam4". Any other missing file is still an error.
        if "cam1_licks" not in str(ex) or not os.environ.get(ce.LICKS_ENV):
            raise
        saved = os.environ.pop(ce.LICKS_ENV)
        try:
            B = ce.load_behaviour(animal, date, rv)
        finally:
            os.environ[ce.LICKS_ENV] = saved
    T = B.trials
    E = T[T.engaged & np.isfinite(T.cue_s)].reset_index(drop=True)
    st = np.where(np.isfinite(E.strobe_s), E.strobe_s, E.cue_s - 3.0)
    rows = np.zeros(len(ft), bool)
    fpos = np.full(len(ft), "", dtype=object)                          # each frame's trial position (drop-spout)
    for a, b_, p_ in zip(st - 0.5, E.stop_s.to_numpy(float) + 1.0, E.pos.to_numpy()):   # as run_session
        m_ = (ft >= a) & (ft <= b_)
        rows |= m_
        fpos[m_] = p_
    ridx = np.flatnonzero(rows)
    ftr = ft[ridx]
    raw = split_blocks(ce.raw_blocks(B, ft))
    vars_ = list(FGROUP)
    X = np.hstack([raw[v][ridx] for v in vars_]).astype(np.float64)
    # unknown frames (NaN pose) contribute nothing: the session's column mean, which the centring below removes
    # (the encoder's `_z` maps NaN to 0 AFTER z-scoring = the same thing)
    mu = np.nanmean(np.where(np.isfinite(X), X, np.nan), 0)
    X = np.where(np.isfinite(X), X, np.where(np.isfinite(mu), mu, 0.0)[None, :])
    col_var = np.concatenate([[v] * raw[v].shape[1] for v in vars_])
    Q = np.linalg.qr(ce._slow_basis(ftr, ce.DRIFT_MIN_PERIOD_S))[0]
    X -= Q @ (Q.T @ X)
    areas = sorted({int(a) for a in np.unique(parea[pstat]) if a != 0})
    Ra = np.stack([Ub[(parea == a) & pstat].mean(0) for a in areas] + [Ub[pstat].mean(0)], 1)
    Y = V[ridx].astype(np.float64)
    Y -= Q @ (Q.T @ Y)
    Rd = Y @ Ra
    area_names = [names.get(a, str(a)) for a in areas] + ["cortex"]
    ex_ = getattr(B, "extra_lick_on_s", None)
    lk = np.sort(np.concatenate([np.asarray(B.lick_on_s, float), np.asarray([] if ex_ is None else ex_, float)]))
    cue_f = np.searchsorted(ftr, E.cue_s.to_numpy(float))
    first_f = np.full(len(E), -1)
    for i, (c, stp) in enumerate(zip(E.cue_s.to_numpy(float), E.stop_s.to_numpy(float))):
        stp = stp if np.isfinite(stp) else c + 3.5
        j = np.searchsorted(lk, c)
        if j < len(lk) and lk[j] < stp:
            first_f[i] = int(np.searchsorted(ftr, lk[j]))
    win = {"all": np.ones(len(ridx), bool), "cue": np.zeros(len(ridx), bool), "lick": np.zeros(len(ridx), bool)}
    for c in E.cue_s.to_numpy(float):
        win["cue"] |= (ftr >= c) & (ftr < c + 1.0)
    for f in first_f[first_f >= 0]:
        t0 = ftr[min(f, len(ftr) - 1)]
        win["lick"] |= (ftr >= t0 - 0.1) & (ftr < t0 + 1.0)
    return {"label": label, "animal": animal, "epoch": epochs.epoch_of(label), "day": epochs.days_since_stroke(label),
            "X": X, "Rd": Rd, "area_names": area_names, "col_var": col_var, "pos": E.pos.to_numpy(),
            "cue_f": cue_f, "first_f": first_f, "fs": float(1.0 / np.median(np.diff(ft))), "win": win,
            "lick_set": "combined" if ex_ is not None and len(ex_) else "cam4", "n_trials": len(E),
            "frame_pos": fpos[ridx], "pos_order": sorted(T.pos.unique()),
            "pos_counts": {str(p_): int((T.pos == p_).sum()) for p_ in sorted(T.pos.unique())}}


def stats_job(label: str):
    """Sufficient statistics of one session (column layout + per-area Gram terms), for the per-animal fit."""
    d = session_design(label)
    X, Rd = d["X"], d["Rd"]
    return {"label": label, "animal": d["animal"], "epoch": d["epoch"], "col_var": d["col_var"],
            "area_names": d["area_names"], "n": len(X), "XtX": X.T @ X, "XtY": X.T @ Rd, "YtY": (Rd ** 2).sum(0),
            "sxx": (X ** 2).sum(0), "pos_order": d["pos_order"], "pos_counts": d["pos_counts"]}


def _subset(s: dict, keep: list[str]) -> tuple:
    ix = [s["area_names"].index(a) for a in keep]
    return s["XtY"][:, ix], s["YtY"][ix]


#: FROZEN VARIANTS (Priya 10-22: "should we also compare encoders with/without movement information?", the ordered
#: bounds, and contact / direction split out). All from the same pre-stroke Gram terms; each is a column transform C
#: of the scaled design (Z' = Z @ C). Groups: FGROUP.
#:   joint              task + movement + direction + contact, jointly (the ridge splits shared variance) -- main
#:   movement_first     task residualised on movement + direction + contact (pre projection): shared -> non-task
#:   task_first         movement + direction + contact residualised on task: shared -> task
#:   no_movement        task only: what the pre-stroke TARGET responses alone predict
#:   task_movement      task + directionless movement (no direction, no contact)
#:   task_movement_dir  task + movement + direction (no contact)
#:   movement_only      directionless movement only
#:   no_task            movement + direction + contact (no task): Musall's unique task = joint - no_task; task-aligned
#:                      non-task variance = no_task - (joint - no_movement)
#: Readouts: residual (data - full) per variant; no_movement -> task_movement = what movement MAGNITUDE / timing adds;
#: -> task_movement_dir = what direction adds; -> joint = what contact adds. Ordered pair brackets the group split.
VARIANTS = ("joint", "movement_first", "task_first", "no_movement", "task_movement", "task_movement_dir",
            "movement_only", "no_task")
_KEEP = {"no_movement": ("task",), "task_movement": ("task", "movement"),
         "task_movement_dir": ("task", "movement", "direction"), "movement_only": ("movement",),
         "no_task": ("movement", "direction", "contact")}


#: FROZEN DROP-ONE-SPOUT (Priya 10-22: "should we do a leave-one-spout out but without removing movement first" -> yes):
#: `<base>|drop:<p>` = base variant (joint / task_first / movement_first) with spout p's position deviation constrained
#: to the trial-weighted mean of the others, exactly `combined_encoding.raw_blocks(drop_pos=p)`: with deviation blocks
#: D_q = X_q - m_q * shared, the drop-p columns are D_q + m_q / (1 - m_p) * D_p (q != p) and D_p is removed -- a column
#: recombination, so it is one more transform of the same Gram terms (m = the animal's POOLED pre trial shares, frozen
#: like everything else). Scored on every session as the loss of variance vs the base (window offset free, all frames
#: and spout p's own trials). task_first = position BEFORE movement (upper bound on spout-specific coding),
#: movement_first = the encoder's order (lower bound), joint = ridge split.
DROP_BASES = ("joint", "task_first", "movement_first")


def _drop_transform(col_var: np.ndarray, pos_order: list, frac: dict, drop: str) -> tuple[np.ndarray, np.ndarray]:
    ce = _ce()
    p = len(col_var)
    C = np.eye(p)
    gone = []
    k = pos_order.index(drop)
    mp = frac[drop]
    for var in ce.POS_VARS:
        idx = np.flatnonzero(col_var == var)
        if len(idx) % len(pos_order):
            raise ValueError(f"{var}: {len(idx)} columns not divisible by {len(pos_order)} positions")
        ch = idx.reshape(len(pos_order), -1)                       # raw_blocks order: one chunk per position
        for j, q in enumerate(pos_order):
            if j != k:
                C[ch[k], ch[j]] += frac[q] / max(1.0 - mp, 1e-9)
        gone.extend(ch[k].tolist())
    keep = np.setdiff1d(np.arange(p), gone)
    return C[:, keep], keep


def _variant_transform(variant: str, col_var: np.ndarray, Gt: np.ndarray, pos: dict | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """((p, p') column transform of the scaled design, (p',) source column of each variant column) for ``variant``
    (Gt = pooled scaled pre Gram, for the ordered projections; ``pos`` = {"order", "frac"} for drop variants)."""
    base, _, drop = variant.partition("|drop:")
    if drop:
        C1, src1 = _drop_transform(col_var, pos["order"], pos["frac"], drop)
        C2, src2 = _variant_transform(base, col_var[src1], C1.T @ Gt @ C1)
        return C1 @ C2, src1[src2]
    p = len(col_var)
    eye = np.eye(p)
    if variant == "joint":
        return eye, np.arange(p)
    grp = np.array([FGROUP[v] for v in col_var])
    if variant in _KEEP:
        keep = np.isin(grp, _KEEP[variant])
        return eye[:, keep], np.flatnonzero(keep)
    if variant in ("movement_first", "task_first"):
        task = grp == "task"
        a, b = (~task, task) if variant == "movement_first" else (task, ~task)    # residualise b on a
        Gaa = Gt[np.ix_(a, a)]
        P = np.linalg.solve(Gaa + 1e-6 * np.trace(Gaa) / a.sum() * np.eye(a.sum()), Gt[np.ix_(a, b)])
        C = eye.copy()
        C[np.ix_(a, b)] = -P
        return C, np.arange(p)
    raise ValueError(variant)


def fit_animal(stats: list[dict], alpha_grid=None, pen_of=None, variant: str = "joint") -> dict:
    """Ridge on the animal's pre sessions (scaled by pooled pre SD), penalties by leave-one-session-out R^2.
    Returns scale, C (variant column transform), col_var of the variant columns, areas, alphas, beta_all (p', R) and
    beta_loso {label: (p', R)}."""
    ce = _ce()
    alpha_grid = alpha_grid or ce.ALPHA_GRID
    pen_of = pen_of or FPEN
    col_var = stats[0]["col_var"]
    for s in stats:
        if len(s["col_var"]) != len(col_var) or (s["col_var"] != col_var).any():
            raise ValueError(f"{s['label']}: column layout differs from {stats[0]['label']}")
    areas = [a for a in stats[0]["area_names"] if all(a in s["area_names"] for s in stats)]
    n = sum(s["n"] for s in stats)
    sd = np.sqrt(sum(s["sxx"] for s in stats) / n)
    ok0 = sd > 1e-9
    scale = np.where(ok0, sd, np.inf)
    G0 = [s["XtX"] / np.outer(scale, scale) for s in stats]
    b0 = []
    yy = []
    for s in stats:
        xy, y2 = _subset(s, areas)
        b0.append(xy / scale[:, None])
        yy.append(y2)
    order = list(stats[0]["pos_order"])
    cnt = {q: sum(s["pos_counts"].get(q, 0) for s in stats) for q in order}
    frac = {q: cnt[q] / max(sum(cnt.values()), 1) for q in order}
    C, src = _variant_transform(variant, col_var, sum(G0), {"order": order, "frac": frac})
    vcol = col_var[src]
    ok = ok0[src]
    G = [C.T @ g @ C for g in G0]
    b = [C.T @ x for x in b0]
    Gt, bt = sum(G), sum(b)
    pen_col = np.array([pen_of[v] for v in vcol])
    groups = sorted(set(pen_col))

    def solve(Gm, bm, al):
        lam = np.array([al[g] for g in pen_col]) * ok + (~ok) * 1.0
        return np.linalg.solve(Gm + np.diag(lam) + 1e-8 * np.eye(len(lam)), bm)

    def loso_r2(al):
        err = tot = 0.0
        for k in range(len(stats)):
            beta = solve(Gt - G[k], bt - b[k], al)
            err += (yy[k] - 2 * (beta * b[k]).sum(0) + np.einsum("pr,pq,qr->r", beta, G[k], beta)).sum()
            tot += yy[k].sum()
        return 1.0 - err / tot
    alphas = {g: 100.0 for g in groups}
    if len(stats) >= 2:
        for _ in range(2):
            for g in groups:
                alphas[g] = max(alpha_grid, key=lambda a: loso_r2({**alphas, g: a}))
    beta_all = solve(Gt, bt, alphas)
    beta_loso = {s["label"]: solve(Gt - G[k], bt - b[k], alphas) for k, s in enumerate(stats)} if len(stats) >= 2 else {}
    return {"variant": variant, "scale": scale, "C": C, "vcol": vcol, "areas": areas, "alphas": alphas,
            "beta_all": beta_all, "beta_loso": beta_loso, "col_var": col_var, "src": src,
            "loso_r2": loso_r2(alphas) if len(stats) >= 2 else np.nan, "pre_labels": [s["label"] for s in stats],
            "pos_order": order, "pos_frac": frac}


def _peth(R: np.ndarray, frames: np.ndarray, off: np.ndarray) -> np.ndarray:
    idx = frames[:, None] + off[None, :]
    ok = (idx >= 0) & (idx < len(R))
    seg = np.where(ok[..., None], R[np.clip(idx, 0, len(R) - 1)], np.nan)
    return np.nanmean(seg, 0).T                                   # (areas, off)


#: MUSALL-STYLE READOUTS (Priya 10-22: "let's add trial to trial and task averaged r2 readouts a la musall"), per
#: window aligned on the cue (0 .. 1.5 s) or the first post-cue lick (-0.1 .. +1.0 s), per area, held out like every
#: frozen score: FRAME = every frame of every trial (condition-locked + single-trial variance; window offset free);
#: TRIAL = each spout's mean segment removed from data AND prediction (single-trial variance only, Musall's
#: "single-trial"); AVG = the per-spout mean segments only, offset free per area (Musall's "trial-averaged").
MUSALL_WINDOWS = {"cue": (0.0, 1.5), "lick": (-0.1, 1.0)}


def musall_r2(D: np.ndarray, P: np.ndarray, pos: np.ndarray) -> dict:
    """D, P: (trials, t, areas) data / prediction segments (NaN = outside the record); pos: (trials,) spout labels.
    -> {"frame", "trial", "avg": (areas,) R^2}."""
    ok = np.isfinite(D) & np.isfinite(P)
    D, P = np.where(ok, D, np.nan), np.where(ok, P, np.nan)

    def r2(y, yh):
        return 1.0 - np.nansum((y - yh) ** 2, axis=(0, 1)) / np.maximum(np.nansum(y ** 2, axis=(0, 1)), 1e-12)
    Dc, Pc = D - np.nanmean(D, axis=(0, 1)), P - np.nanmean(P, axis=(0, 1))
    out = {"frame": r2(Dc, Pc)}
    Dd, Pd = D.copy(), P.copy()
    Dm, Pm = [], []
    for q in np.unique(pos):
        m = pos == q
        dm, pm = np.nanmean(D[m], 0, keepdims=True), np.nanmean(P[m], 0, keepdims=True)
        Dd[m] -= dm
        Pd[m] -= pm
        Dm.append(dm[0])
        Pm.append(pm[0])
    out["trial"] = r2(Dd, Pd)
    Dm, Pm = np.stack(Dm), np.stack(Pm)                         # (spouts, t, areas)
    out["avg"] = r2(Dm - np.nanmean(Dm, axis=(0, 1)), Pm - np.nanmean(Pm, axis=(0, 1)))
    return out


def _segments(R: np.ndarray, frames: np.ndarray, off: np.ndarray) -> np.ndarray:
    idx = frames[:, None] + off[None, :]
    ok = (idx >= 0) & (idx < len(R))
    return np.where(ok[..., None], R[np.clip(idx, 0, len(R) - 1)], np.nan)


def predict_job(job):
    """Apply the animal's frozen models (one per variant) to one session; PETHs of data and each part, cue- and
    lick-aligned, written to <out>/<variant>/sessions/<label>.{npz,json}."""
    label, fits, out_dir = job
    ce = _ce()
    d = session_design(label)
    f0 = next(iter(fits.values()))
    if len(d["col_var"]) != len(f0["col_var"]) or (d["col_var"] != f0["col_var"]).any():
        raise ValueError(f"{label}: column layout differs from the animal's pre fit")
    ix = [d["area_names"].index(a) for a in f0["areas"]]
    Rd = d["Rd"][:, ix]
    Z = d["X"] / f0["scale"]
    off = np.arange(int(ce.PETH_S[0] * d["fs"]), int(ce.PETH_S[1] * d["fs"]) + 1)
    pos_order = sorted(set(d["pos"]))
    ci_ = f0["areas"].index("cortex")
    summary = {"label": label}
    # SSE subsets for the drop-spout scores: window x (all frames | each spout's own trials), window offset free
    subsets = {(w, q): wm & ((d["frame_pos"] == q) if q != "all" else True)
               for w, wm in d["win"].items() for q in ["all", *pos_order]}
    sst = {f"{w}|{q}": (((Rd[m] - Rd[m].mean(0)) ** 2).sum(0)).tolist() for (w, q), m in subsets.items() if m.any()}
    drop_sse = {}

    def _sse(pred):
        out = {}
        for (w, q), m in subsets.items():
            if m.any():
                r = Rd[m] - pred[m]
                out[f"{w}|{q}"] = (((r - r.mean(0)) ** 2).sum(0)).tolist()
        return out
    for variant, fit in fits.items():
        beta = fit["beta_loso"].get(label, fit["beta_all"])
        frozen_from = "loso" if label in fit["beta_loso"] else "all_pre"
        Ce = fit["C"]                                     # effective weights on the scaled design: Z @ (C @ beta)
        if "|drop:" in variant or variant in DROP_BASES:
            drop_sse[variant] = _sse(Z @ (Ce @ beta))
        if "|drop:" in variant:
            continue
        gcol = np.array([FGROUP[v] for v in fit["vcol"]])
        comp = {"data": Rd, "full": Z @ (Ce @ beta)}
        for g in FGROUPS:
            m = gcol == g
            comp[g] = Z @ (Ce[:, m] @ beta[m]) if m.any() else np.zeros_like(Rd)
        pm = np.isin(fit["vcol"], list(ce.POS_VARS))
        comp["pos"] = Z @ (Ce[:, pm] @ beta[pm]) if pm.any() else np.zeros_like(Rd)
        peth = np.full((len(comp), len(pos_order), len(ix), len(off)), np.nan, np.float32)
        peth_lick = np.full_like(peth, np.nan)
        n_lick = {}
        for ci, R in enumerate(comp.values()):
            for pi, p in enumerate(pos_order):
                m = d["pos"] == p
                peth[ci, pi] = _peth(R, d["cue_f"][m], off)
                fl = d["first_f"][m & (d["first_f"] >= 0)]
                n_lick[p] = int(len(fl))
                if len(fl):
                    peth_lick[ci, pi] = _peth(R, fl, off)
        musall = {}
        for al, (a0, a1) in MUSALL_WINDOWS.items():
            fr_ = d["cue_f"] if al == "cue" else d["first_f"]
            keep = fr_ >= 0
            if keep.sum() < 2:
                continue
            offm = np.arange(int(round(a0 * d["fs"])), int(round(a1 * d["fs"])))
            mr = musall_r2(_segments(Rd, fr_[keep], offm), _segments(comp["full"], fr_[keep], offm), d["pos"][keep])
            musall[al] = {k: [float(x) for x in v] for k, v in mr.items()}
        r2 = {}
        for w, wm in d["win"].items():
            y, yh = Rd[wm, ci_], comp["full"][wm, ci_]
            r2[w] = float(1.0 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)) if wm.any() else None
        sess = Path(out_dir) / variant / "sessions"
        sess.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sess / f"{label}.npz", peth=peth, peth_lick=peth_lick, peth_off_s=off / d["fs"])
        meta = {"label": label, "animal": d["animal"], "epoch": d["epoch"], "day": d["day"],
                "order": f"frozen_pre:{variant}", "variant": variant, "frozen_from": frozen_from,
                "pre_labels": fit["pre_labels"], "alphas": fit["alphas"], "loso_r2_pre": fit["loso_r2"],
                "gate": os.environ.get("OROFACIAL_GATE", "reference"), "lick_set": d["lick_set"],
                "n_trials": d["n_trials"], "n_lick_aligned_by_pos": n_lick, "pos_order": pos_order,
                "peth_parts": list(comp), "areas": fit["areas"], "frozen_r2_cortex": r2, "r2_musall": musall,
                "stats_version": STATS_VERSION}
        (sess / f"{label}.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        summary[variant] = r2
        summary["frozen_from"] = frozen_from
    if drop_sse:
        dd = Path(out_dir) / "dropspout" / "sessions"
        dd.mkdir(parents=True, exist_ok=True)
        (dd / f"{label}.json").write_text(json.dumps(
            {"label": label, "animal": d["animal"], "epoch": d["epoch"], "areas": f0["areas"], "pos_order": pos_order,
             "sst": sst, "sse": drop_sse, "lick_set": d["lick_set"], "stats_version": STATS_VERSION},
            default=str), encoding="utf-8")
    return summary


def summarise_drop(out: Path) -> None:
    """Frozen drop-one-spout table: per session, base, spout q, window, subset (all frames | q's own trials), area:
    dR2 = (SSE[base|drop:q] - SSE[base]) / SST; then animal x epoch means (rule 8: animals first).
    -> <out>/dropspout/frozen_dropspout{,_by_animal}.csv"""
    import pandas as pd
    rows = []
    for js in sorted((Path(out) / "dropspout" / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if m["label"][-4:] < "0801":
            continue
        for base in DROP_BASES:
            if base not in m["sse"]:
                continue
            for q in m["pos_order"]:
                v = f"{base}|drop:{q}"
                if v not in m["sse"]:
                    continue
                for key, sst in m["sst"].items():
                    w, sub = key.split("|")
                    if sub not in ("all", q):
                        continue
                    d = (np.asarray(m["sse"][v][key]) - np.asarray(m["sse"][base][key])) / np.maximum(sst, 1e-12)
                    for ai, area in enumerate(m["areas"]):
                        rows.append({"label": m["label"], "animal": m["animal"], "epoch": m["epoch"], "base": base,
                                     "spout": q, "window": w, "subset": "all" if sub == "all" else "own",
                                     "area": area, "dR2": float(d[ai])})
    if not rows:
        print("  (no drop-spout files)")
        return
    T = pd.DataFrame(rows)
    A = (T.groupby(["animal", "epoch", "base", "spout", "window", "subset", "area"])
         .agg(dR2=("dR2", "mean"), n_sess=("label", "nunique")).reset_index())
    dd = Path(out) / "dropspout"
    T.to_csv(dd / "frozen_dropspout.csv", index=False, float_format="%.5g")
    A.to_csv(dd / "frozen_dropspout_by_animal.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    order = ["pre", "acute", "subacute", "chronic"]
    for area in ("cortex", "SSp-m_left"):
        S = A[(A.spout == "far_contra") & (A.window == "lick") & (A.subset == "own") & (A.area == area)]
        if len(S):
            P = (S.pivot_table(index=["base", "animal"], columns="epoch", values="dR2") * 1e3)
            print(f"\n=== frozen drop far_contra, {area}, lick window, far-contra trials, dR2 x1e3")
            print(P.reindex(columns=[c for c in order if c in P.columns]).round(1).to_string())
    print(f"\n-> {dd / 'frozen_dropspout_by_animal.csv'}")


def summarise_musall(out: Path, variants) -> None:
    """Per variant x animal x epoch x align x measure (frame / trial / avg) x area, mean over sessions (rule 8: animals
    first) -> <out>/frozen_musall_r2{,_by_animal}.csv; prints cortex + contralesional SSp-m, both windows."""
    import pandas as pd
    rows = []
    for v in variants:
        for js in sorted((Path(out) / v / "sessions").glob("*.json")):
            m = json.loads(js.read_text(encoding="utf-8"))
            for al, d_ in (m.get("r2_musall") or {}).items():
                for meas, vals in d_.items():
                    for area, x in zip(m["areas"], vals):
                        rows.append({"variant": v, "label": m["label"], "animal": m["animal"], "epoch": m["epoch"],
                                     "align": al, "measure": meas, "area": area, "r2": x})
    if not rows:
        print("  (no Musall readouts)")
        return
    T = pd.DataFrame(rows)
    A = T.groupby(["variant", "animal", "epoch", "align", "measure", "area"]).r2.mean().reset_index()
    T.to_csv(Path(out) / "frozen_musall_r2.csv", index=False, float_format="%.4f")
    A.to_csv(Path(out) / "frozen_musall_r2_by_animal.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 240)
    order = [e for e in ("pre", "acute", "subacute", "chronic") if e in set(A.epoch)]
    for area in ("cortex", "SSp-m_left"):
        S = A[A.area == area].pivot_table(index=["align", "measure", "variant"], columns=["animal", "epoch"], values="r2")
        S = S.reindex(columns=pd.MultiIndex.from_product([sorted(set(A.animal)), order]))
        print(f"\n=== frozen Musall-style R2, {area} (held out; pre = leave-one-session-out)")
        print(S.round(3).to_string())
    print(f"\n-> {Path(out) / 'frozen_musall_r2_by_animal.csv'}")


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tag", default="gate-attempted")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--no-drop-spout", action="store_true", help="skip the frozen drop-one-spout variants")
    ap.add_argument("--reuse-stats", action="store_true", help="reuse fit/pre_stats.pkl (skip the pre statistics pass)")
    ap.add_argument("--summarise-drop", action="store_true", help="only rebuild the drop-spout table from saved files")
    a = ap.parse_args(argv)
    if a.summarise_drop:
        summarise_drop(oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}")
        return 0
    labels = sorted(a.labels or (_ce().sessions_available() if a.all else []))
    labels = [x for x in labels if x[-4:] >= "0801"]                       # August-only pre baseline (primary)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}"
    assert_writable(out)
    pre = [x for x in labels if epochs.epoch_of(x) == "pre"]
    print(f"[frozen] {len(labels)} sessions, {len(pre)} August pre -> {out}", flush=True)
    import pickle
    sp = out / "fit" / "pre_stats.pkl"
    if a.reuse_stats and sp.exists():                    # a re-scoring run: the pre statistics are already saved
        saved = pickle.loads(sp.read_bytes())
        got = [(s_["label"], s_) for s_ in saved if s_["label"] in pre and s_.get("pos_order")]
        print(f"  reusing {len(got)} pre-session statistics from {sp}", flush=True)
    else:
        got, fail = ak.fan_sessions(pre, stats_job, jobs=a.jobs)
        for it, err in fail:
            print(f"  !! stats {it}: {err}", flush=True)
        (out / "fit").mkdir(parents=True, exist_ok=True)
        # the pre sessions' sufficient statistics are KEPT: the ordered (movement- / task-first) bounds are linear
        # transforms of the same Gram terms, so they need no second pass over the data
        sp.write_bytes(pickle.dumps([s_ for _l, s_ in sorted(got, key=lambda kv: kv[0])],
                                    protocol=pickle.HIGHEST_PROTOCOL))
    by_an: dict = {}
    for _lab, s in got:
        by_an.setdefault(s["animal"], []).append(s["label"])
    # one fit job per ANIMAL (variants + drop-spout fits are independent across animals; serial in the main process
    # they took ~1.5 min / animal for the variants alone while the workers sat idle). Items stay small: the worker
    # reads the statistics from the pickle written above.
    fit_items = [(an, str(sp), tuple(sorted(by_an[an])), tuple(a.variants), not a.no_drop_spout) for an in sorted(by_an)]
    fres, ffail = ak.fan_sessions(fit_items, fit_animal_job, jobs=min(a.jobs, len(fit_items)), label="fit")
    if ffail:
        raise RuntimeError(f"frozen fits failed: {[(it[0], err) for it, err in ffail]}")
    fits: dict = {}
    for (an, _sp, labs, _v, _d), fa in fres:
        fits[an] = fa
        for v in a.variants:
            f = fa[v]
            np.savez_compressed(out / "fit" / f"{an}_{v}.npz", scale=f["scale"], C=f["C"], vcol=f["vcol"],
                                beta_all=f["beta_all"], col_var=f["col_var"], areas=np.array(f["areas"]))
            print(f"  {an} {v}: {len(labs)} pre sessions, LOSO R^2 (areas summed) {f['loso_r2']:.3f}, "
                  f"alphas {f['alphas']}", flush=True)
    jobs = [(x, fits[config.animal_of(x)], str(out)) for x in labels if config.animal_of(x) in fits]
    res, fail = _fan(jobs, a.jobs)
    for it, err in fail:
        print(f"  !! predict {it[0]}: {err}", flush=True)
    if not a.no_drop_spout:
        summarise_drop(out)
    summarise_musall(out, a.variants)
    for _it, r in sorted(res, key=lambda kv: kv[1]["label"]):
        print(f"  {r['label']} [{r['frozen_from']}] frozen R^2 cortex (all / cue / lick): "
              + " | ".join(f"{v} " + " ".join(f"{x:.3f}" if x is not None else "-" for x in r[v].values())
                           for v in a.variants), flush=True)
    return 0 if not fail else 1


def fit_animal_job(item):
    """All of one animal's frozen fits (each variant, then the drop-one-spout fits under DROP_BASES), from the saved
    pre statistics. Module-level for spawn."""
    import pickle
    an, sp, labs, variants, drop = item
    st = sorted((s for s in pickle.loads(Path(sp).read_bytes()) if s["label"] in set(labs)), key=lambda s: s["label"])
    fits = {v: fit_animal(st, variant=v) for v in variants}
    if drop:                                             # frozen drop-one-spout, each ordering (DROP_BASES)
        for base in DROP_BASES:
            if base in fits:
                for q in st[0]["pos_order"]:
                    fits[f"{base}|drop:{q}"] = fit_animal(st, variant=f"{base}|drop:{q}")
    return fits


def _fan(jobs, n):
    from wfield_local import parallel
    res, fail = parallel.fan_out(jobs, predict_job, jobs=n, label="frozen", log=lambda m: print(m[:120], flush=True))
    res = sorted(res, key=lambda kv: kv[1]["label"])            # rule 9: never completion order
    fail = sorted(fail, key=lambda kv: kv[0][0])
    return res, fail


if __name__ == "__main__":
    raise SystemExit(main())

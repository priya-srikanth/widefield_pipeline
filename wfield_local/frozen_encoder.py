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

GROUPS (STATS_VERSION 4, 10-22 afternoon; see FGROUP): timing / target / exec_locked / exec_within / lick / other /
contact, built by splitting every lick-related movement column into the animal's PRE-STROKE spout-locked template (s)
and the deviation from it (d) -- a behaviour-only first pass over the pre sessions (`template_job`, saved as
fit/templates.pkl).

Per session the output matches `combined_encoding`'s session files where the decomposition reads them: `peth` (cue-
aligned) and `peth_lick` (first post-cue lick), parts data / full / each group / spout / lick_all / pos (joint and the
ordered variants; data / full only for the reduced models), x position x area x offset (PETH_S), plus the frozen
model's R^2 (all / cue / lick windows, cortex). `data - full` = activity the
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

STATS_VERSION = 4   # 2: FGROUP split (contact / direction / movement); 3: protrusion -> direction; 4: lick / spout split


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
#: v3 layout above was REPLACED 10-22 afternoon (STATS_VERSION 4; Priya: "should we have a 'lick' group and a 'spout
#: position information' group?" -> "yes you can build the modified model"). Why: the leakage check
#: (`scripts/frozen_group_leakage.py`) found the "directionless" group decoding the spout at 0.60-0.69 (chance 0.17;
#: jaw opening 0.40-0.57, lick timing ~0.3), and licks / tongue / jaw are redundant readouts of one act, so their
#: separate drops split credit arbitrarily. The split is now BY CONSTRUCTION, not by variable name:
#:   every lick-related movement column x (lick onsets, tongue speed, jaw opening / speed, and the direction traces
#:   tongue x / y / protrusion, jaw x, nose x) = SPOUT-LOCKED part s + DEVIATION d, where s = the animal's PRE-STROKE
#:   mean trace for that spout at that time from the cue (`TEMPLATE_S`; pooled over its August pre sessions, each
#:   session centred) and d = x - s. s is "what the animal's licking looks like at this spout"; d is everything else,
#:   INCLUDING post-stroke changes in vigour / timing at a spout (s is frozen with the model, so a weaker acute lick
#:   at far contra lands in d = the LICK group, not in the spout group).
#: Groups:
#:   timing       cue, arrival (shared kernels; spout-independent)
#:   target       pos_enl, pos_cue (the six position blocks: the spout itself)
#:   exec_locked  spout-locked part s of every lick-related column (executed, but fully predictable from the spout)
#:   exec_within  deviation d of the DIRECTION traces + the per-lick executed angle (already within-spout): where the
#:                tongue / jaw / nose go relative to that spout's usual path
#:   lick         deviation d of lick onsets, tongue speed, jaw opening / speed: vigour and timing that is not
#:                about which spout (leakage-checked: should decode the spout ~ chance in pre sessions)
#:   other        nose y, running
#:   contact      the spout-contact kernel
#: Super-groups: task = timing + target; spout = target + exec_locked + exec_within; movement = lick + exec_locked +
#: exec_within + other. target vs exec_locked are perfectly confounded by design (both follow the spout); only
#: exec_within separates them, so read target / executed only as the ordered bracket (target_first / exec_first).
SPLIT_VARS = ("lick_onset", "tongue_mag", "jaw_mag", "tongue_dir", "jaw_dir", "nose_dir")
_LICK_BASE = ("lick_onset", "tongue_mag", "jaw_mag")
TEMPLATE_S = (-4.0, 8.0)                                 # cue-relative span of the spout-locked templates (s)
FGROUP = {"cue": "timing", "arrival": "timing", "pos_enl": "target", "pos_cue": "target",
          **{f"{v}:spl": "exec_locked" for v in SPLIT_VARS},
          **{f"{v}:dev": ("lick" if v in _LICK_BASE else "exec_within") for v in SPLIT_VARS},
          "exec_angle": "exec_within", "nose_mag": "other", "running": "other", "contact": "contact"}
FGROUPS = ("timing", "target", "exec_locked", "exec_within", "lick", "other", "contact")
SUPER = {"task": ("timing", "target"), "spout": ("target", "exec_locked", "exec_within"),
         "movement": ("lick", "exec_locked", "exec_within", "other"),
         "nontask": ("lick", "exec_locked", "exec_within", "other", "contact")}
#: penalty groups as the encoder's PEN_GROUP_OF (task / lick / orofacial / uninstructed); s and d of a variable share
#: its penalty group
_FPEN_BASE = {"cue": "task", "arrival": "task", "pos_enl": "task", "pos_cue": "task",
              "lick_onset": "lick", "contact": "lick", "exec_angle": "lick",
              "tongue_mag": "orofacial", "tongue_dir": "orofacial", "jaw_mag": "orofacial", "jaw_dir": "orofacial",
              "nose_mag": "uninstructed", "nose_dir": "uninstructed", "running": "uninstructed"}
FPEN = {v: _FPEN_BASE[v.split(":")[0]] for v in FGROUP}


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


#: spout-locked templates, in bins of TEMPLATE_BIN_S (one imaging frame at the cohort's 31.25 Hz). TWO-PART TIME AXIS
#: (10-22 afternoon): CUE-locked (TEMPLATE_S) before the trial's first post-cue lick (and on trials without one),
#: LICK-locked (LICK_TEMPLATE_S, from that first lick) after it. Cue-locked only, the first v4 leakage check left the
#: lick group decoding the spout at 0.38-0.42 (chance 0.17) and exec_within at ~0.53: first-lick latency differs by
#: spout, so a cue-locked mean smears each spout's lick-locked profile and the deviation keeps spout structure.
#: A bin needs >= TEMPLATE_MIN_N pre trials at that spout to get a template value (else s = 0 there: all of x is d).
TEMPLATE_BIN_S = 0.032
TEMPLATE_MIN_N = 20
LICK_TEMPLATE_S = 6.0


def _template_bins() -> tuple[int, int]:
    """(k0, K): absolute bin indices k0 .. k0 + K - 1; the first Kc are cue-locked, the rest lick-locked."""
    k0 = int(np.floor(TEMPLATE_S[0] / TEMPLATE_BIN_S))
    kc = int(np.ceil(TEMPLATE_S[1] / TEMPLATE_BIN_S)) - k0 + 1
    return k0, kc + int(np.ceil(LICK_TEMPLATE_S / TEMPLATE_BIN_S)) + 1


def template_index(t_cue: np.ndarray, t_lick: np.ndarray) -> np.ndarray:
    """Absolute template bin of each frame from its time since the trial's cue and since its first post-cue lick
    (NaN = none / not yet): lick-locked once the lick has happened, cue-locked before; k0 - 1 = outside the template."""
    k0, K = _template_bins()
    kc = int(np.ceil(TEMPLATE_S[1] / TEMPLATE_BIN_S)) - k0 + 1
    after = np.isfinite(t_lick) & (t_lick >= 0)
    kcue = np.round(np.nan_to_num(t_cue, nan=-1e9) / TEMPLATE_BIN_S).astype(np.int64)
    klk = k0 + kc + np.round(np.where(after, t_lick, 0.0) / TEMPLATE_BIN_S).astype(np.int64)
    k = np.where(after, klk, kcue)
    bad = np.where(after, klk >= k0 + K, (kcue < k0) | (kcue >= k0 + kc))
    return np.where(bad, k0 - 1, k)


def template_sums(Xc: np.ndarray, pos_of: np.ndarray, k_of: np.ndarray, pos_order: list) -> tuple:
    """Per spout x cue-relative bin: column sums and frame counts of session-centred raw columns Xc (n, c).
    pos_of: each frame's trial spout ("" = none); k_of: its cue-relative bin index (absolute, from _template_bins)."""
    k0, K = _template_bins()
    S = np.zeros((len(pos_order), K, Xc.shape[1]))
    N = np.zeros((len(pos_order), K))
    kk = k_of - k0
    for pi, p in enumerate(pos_order):
        m = (pos_of == p) & (kk >= 0) & (kk < K)
        np.add.at(S[pi], kk[m], Xc[m])
        np.add.at(N[pi], kk[m], 1.0)
    return S, N


def template_means(S: np.ndarray, N: np.ndarray) -> np.ndarray:
    ok = N >= TEMPLATE_MIN_N
    return np.where(ok[..., None], S / np.maximum(N, 1.0)[..., None], 0.0)


def spout_split(Xc: np.ndarray, pos_of: np.ndarray, k_of: np.ndarray, pos_order: list, M: np.ndarray) -> tuple:
    """(s, d): s = the spout's template at each frame's cue-relative bin (0 outside the template / for unknown spouts),
    d = Xc - s. M: (spouts, K, c) template means in pos_order."""
    k0, K = _template_bins()
    s = np.zeros_like(Xc)
    kk = k_of - k0
    for pi, p in enumerate(pos_order):
        m = (pos_of == p) & (kk >= 0) & (kk < K)
        s[m] = M[pi][kk[m]]
    return s, Xc - s


def _centred(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(np.where(np.isfinite(x), x, np.nan), 0)
    x = x - np.where(np.isfinite(mu), mu, 0.0)[None, :]
    return np.where(np.isfinite(x), x, 0.0)


def session_design(label: str, templates: dict | None = None, template_only: bool = False, spatial: dict | None = None,
                   basis_only: bool = False) -> dict:
    """Raw (drift-removed, session-centred) design X, area targets Rd, column variables, trial alignment frames.
    ``templates`` = the animal's spout-locked templates (`build_templates`), required for the STATS_VERSION 4 layout;
    ``template_only`` = return only this session's template sums (behaviour pass, no imaging targets);
    ``spatial`` = {"P": (stat pixels x R) projection, "names"} for basis / grid targets instead of Allen areas (Rd stays
    None above MAX_DENSE_TARGETS: Y and Ra are returned and consumers project on demand); ``basis_only`` = return this
    session's variance-weighted spatial components (the shared-basis pass)."""
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
    ftri = np.full(len(ft), -1)                                        # ... and its trial (later trial wins, as fpos)
    for i, (a, b_, p_) in enumerate(zip(st - 0.5, E.stop_s.to_numpy(float) + 1.0, E.pos.to_numpy())):  # as run_session
        m_ = (ft >= a) & (ft <= b_)
        rows |= m_
        fpos[m_] = p_
        ftri[m_] = i
    ridx = np.flatnonzero(rows)
    ftr = ft[ridx]
    pos_of = fpos[ridx]
    # template time axis: cue-locked until the trial's first post-cue lick (same rule as first_f below), then
    # lick-locked (TEMPLATE_BIN_S comment)
    ex0 = getattr(B, "extra_lick_on_s", None)
    lk0 = np.sort(np.concatenate([np.asarray(B.lick_on_s, float), np.asarray([] if ex0 is None else ex0, float)]))
    cue_t = E.cue_s.to_numpy(float)
    fl_t = np.full(len(E), np.nan)
    for i, (c, stp) in enumerate(zip(cue_t, E.stop_s.to_numpy(float))):
        stp = stp if np.isfinite(stp) else c + 3.5
        j = np.searchsorted(lk0, c)
        if j < len(lk0) and lk0[j] < stp:
            fl_t[i] = lk0[j]
    tri = ftri[ridx]
    k_of = template_index(ftr - cue_t[tri], ftr - fl_t[tri])
    pos_order = sorted(T.pos.unique())
    if basis_only:                                       # spatial-basis pass: this session's weighted spatial components
        Q = np.linalg.qr(ce._slow_basis(ftr, ce.DRIFT_MIN_PERIOD_S))[0]
        Y = V[ridx].astype(np.float64)
        Y -= Q @ (Q.T @ Y)
        w, Ue = np.linalg.eigh(Y.T @ Y)                  # Y'Y = Ue diag(w) Ue'  ->  pixel covariance = (Ub Ue sqrt w)(...)'
        return {"label": label, "animal": animal, "n": len(Y),
                "UbL": (Ub[pstat].astype(np.float64) @ (Ue * np.sqrt(np.clip(w, 0, None)))).astype(np.float32)}
    raw = split_blocks(ce.raw_blocks(B, ft))
    xc = {v: _centred(raw[v][ridx].astype(np.float64)) for v in SPLIT_VARS}
    if template_only:
        return {"label": label, "animal": animal, "pos_order": pos_order,
                "sums": {v: template_sums(xc[v], pos_of, k_of, pos_order) for v in SPLIT_VARS}}
    if templates is None:
        raise ValueError(f"{label}: the STATS_VERSION {STATS_VERSION} layout needs the animal's spout-locked templates")
    parts = {}
    for v in SPLIT_VARS:
        M = np.stack([templates["mean"][v][templates["pos_order"].index(p)] if p in templates["pos_order"]
                      else np.zeros(templates["mean"][v].shape[1:]) for p in pos_order])
        parts[f"{v}:spl"], parts[f"{v}:dev"] = spout_split(xc[v], pos_of, k_of, pos_order, M)
    vars_ = list(FGROUP)
    blocks = [parts[v] if v in parts else raw[v][ridx].astype(np.float64) for v in vars_]
    X = np.hstack(blocks)
    # unknown frames (NaN pose) contribute nothing: the session's column mean, which the centring below removes
    # (the encoder's `_z` maps NaN to 0 AFTER z-scoring = the same thing)
    mu = np.nanmean(np.where(np.isfinite(X), X, np.nan), 0)
    X = np.where(np.isfinite(X), X, np.where(np.isfinite(mu), mu, 0.0)[None, :])
    col_var = np.concatenate([[v] * b.shape[1] for v, b in zip(vars_, blocks)])
    Q = np.linalg.qr(ce._slow_basis(ftr, ce.DRIFT_MIN_PERIOD_S))[0]
    X -= Q @ (Q.T @ X)
    if spatial is None:                                  # Allen-area targets (the v1-v4 default)
        areas = sorted({int(a) for a in np.unique(parea[pstat]) if a != 0})
        Ra = np.stack([Ub[(parea == a) & pstat].mean(0) for a in areas] + [Ub[pstat].mean(0)], 1)
        area_names = [names.get(a, str(a)) for a in areas] + ["cortex"]
    else:                                                # SPATIAL targets: projection P (stat pixels x R) + cortex mean
        Ra = np.column_stack([np.asarray((spatial["P"].T @ Ub[pstat].astype(np.float64))).T, Ub[pstat].mean(0)])
        area_names = list(spatial["names"]) + ["cortex"]
    Y = V[ridx].astype(np.float64)
    Y -= Q @ (Q.T @ Y)
    Rd = Y @ Ra if Ra.shape[1] <= MAX_DENSE_TARGETS else None      # large target sets stay lazy (Y @ Ra on demand)
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
            "frame_pos": pos_of, "pos_order": pos_order,
            "pos_counts": {str(p_): int((T.pos == p_).sum()) for p_ in pos_order},
            "Y": Y if Rd is None else None, "Ra": Ra if Rd is None else None}


def template_job(label: str) -> dict:
    """One pre session's spout-locked template sums (behaviour only)."""
    return session_design(label, template_only=True)


def build_templates(parts: list[dict]) -> dict:
    """{animal: {"pos_order", "mean": {var: (spouts, K, c)}, "n_sessions", "labels"}} pooled over the animal's pre
    sessions (sums / counts added, so every pre trial weighs the same)."""
    by: dict = {}
    for t in sorted(parts, key=lambda t: t["label"]):
        by.setdefault(t["animal"], []).append(t)
    out = {}
    for an, ts in by.items():
        order = sorted({p for t in ts for p in t["pos_order"]})
        mean = {}
        for v in SPLIT_VARS:
            S = N = None
            for t in ts:
                s_, n_ = t["sums"][v]
                ix = [order.index(p) for p in t["pos_order"]]
                if S is None:
                    S = np.zeros((len(order),) + s_.shape[1:])
                    N = np.zeros((len(order),) + n_.shape[1:])
                S[ix] += s_
                N[ix] += n_
            mean[v] = template_means(S, N)
        out[an] = {"pos_order": order, "mean": mean, "n_sessions": len(ts), "labels": [t["label"] for t in ts],
                   "bin_s": TEMPLATE_BIN_S, "span_s": TEMPLATE_S, "min_n": TEMPLATE_MIN_N}
    return out


#: SPATIAL TARGETS (Priya 10-22: "is predicting allen areas really what we want? not a downsampled map?" -> the
#: literature route, Musall et al. 2019, fits in SVD space and reconstructs pixel maps; every session has its OWN SVD
#: basis, so a frozen model needs a basis common to the animal's sessions; Priya: "ok proceed"):
#:   basis  per animal, the top components of its pooled PRE-session data on the shared Allen-aligned stat mask (every
#:          session's U_atlas lives on the same 540 x 640 grid, 129,770 stat pixels): each pre session contributes
#:          Ub Ue sqrt(w) (its spatial components weighted by its trial-frame variance, drift removed), the stack's
#:          leading left singular vectors up to BASIS_VAR of the variance (<= BASIS_KMAX) are the targets; a session's
#:          targets are its data projected on them; maps = basis x (anything in target space)
#:   grid   GRID_BLOCK x GRID_BLOCK pixel block means on the stat mask (blocks with >= GRID_MIN_PIX pixels): the
#:          transparent check on the basis
#: Both add the cortex mean as a last target (scores / Musall readouts are reported on it).
BASIS_VAR = 0.95
BASIS_KMAX = 200
GRID_BLOCK = 8
GRID_MIN_PIX = 16


def basis_job(label: str) -> dict:
    return session_design(label, basis_only=True)


def build_basis(parts: list[dict]) -> dict:
    """{animal: spatial dict} from the pre sessions' weighted components (basis_job)."""
    by: dict = {}
    for t in sorted(parts, key=lambda t: t["label"]):
        by.setdefault(t["animal"], []).append(t)
    out = {}
    for an, ts in by.items():
        B = np.hstack([t["UbL"] for t in ts])                       # (stat pixels, 100 x sessions), float32
        G = (B.T.astype(np.float64) @ B.astype(np.float64))
        w, Qe = np.linalg.eigh(G)
        o = np.argsort(w)[::-1]
        w, Qe = np.clip(w[o], 0, None), Qe[:, o]
        cum = np.cumsum(w) / w.sum()
        K = int(min(BASIS_KMAX, np.searchsorted(cum, BASIS_VAR) + 1))
        W = ((B.astype(np.float64) @ Qe[:, :K]) / np.sqrt(w[:K])).astype(np.float32)   # orthonormal columns
        out[an] = {"kind": "basis", "P": W, "names": [f"b{j:03d}" for j in range(K)], "evr": cum[:K],
                   "labels": [t["label"] for t in ts]}
    return out


def grid_spatial(label: str) -> dict:
    """Block-mean projection on the stat mask (same for every session: the Allen-aligned grid is shared)."""
    from scipy import sparse
    ce = _ce()
    _V, _ft, _Ub, flat, _parea, _names, shape, pstat = ce.load_imaging(label)
    H, Wd = (int(x) for x in shape)
    pix = flat[pstat]
    r, c = pix // Wd, pix % Wd
    nbc = -(-Wd // GRID_BLOCK)
    blk = (r // GRID_BLOCK) * nbc + c // GRID_BLOCK
    ub, inv, cnt = np.unique(blk, return_inverse=True, return_counts=True)
    keep = cnt >= GRID_MIN_PIX
    col = np.cumsum(keep) - 1
    m = keep[inv]
    P = sparse.csr_matrix((1.0 / cnt[inv][m], (np.flatnonzero(m), col[inv][m])), shape=(len(pix), int(keep.sum())))
    return {"kind": "grid", "P": P, "names": [f"g{b}" for b in ub[keep]], "block": GRID_BLOCK, "blocks": ub[keep],
            "shape": (H, Wd)}


#: targets above this count are never materialised as (frames x targets) arrays (a 2,022-bin grid would be ~2.7 GB per
#: session in float64): statistics and PETHs are projected from Y (frames x SVD components) instead
MAX_DENSE_TARGETS = 300


def _spatial_of(spath: str | None, animal: str) -> dict | None:
    if not spath:
        return None
    import pickle
    return pickle.loads((Path(spath) / f"spatial_{animal}.pkl").read_bytes())


def _templates_of(tpath: str, animal: str) -> dict:
    import pickle
    return pickle.loads(Path(tpath).read_bytes())[animal]


def stats_job(item):
    """Sufficient statistics of one session (column layout + per-area Gram terms), for the per-animal fit.
    item = (label, path of fit/templates.pkl)."""
    from wfield_local import config
    label, tpath, spath = (*item, None)[:3]
    an = config.animal_of(label)
    d = session_design(label, _templates_of(tpath, an), spatial=_spatial_of(spath, an))
    X, Rd = d["X"], d["Rd"]
    if Rd is None:                                       # lazy targets: X'Y Ra and diag(Ra' Y'Y Ra)
        XtY = (X.T @ d["Y"]) @ d["Ra"]
        YtY = np.einsum("kr,kl,lr->r", d["Ra"], d["Y"].T @ d["Y"], d["Ra"])
    else:
        XtY, YtY = X.T @ Rd, (Rd ** 2).sum(0)
    return {"label": label, "animal": d["animal"], "epoch": d["epoch"], "col_var": d["col_var"],
            "area_names": d["area_names"], "n": len(X), "XtX": X.T @ X, "XtY": XtY, "YtY": YtY,
            "sxx": (X ** 2).sum(0), "pos_order": d["pos_order"], "pos_counts": d["pos_counts"]}


def _subset(s: dict, keep: list[str]) -> tuple:
    ix = [s["area_names"].index(a) for a in keep]
    return s["XtY"][:, ix], s["YtY"][ix]


#: FROZEN VARIANTS (STATS_VERSION 4 groups, see FGROUP). All from the same pre-stroke Gram terms; each is a column
#: transform C of the scaled design (Z' = Z @ C).
#:   joint            everything jointly (the ridge splits shared variance) -- main
#:   no_<g>           joint refit WITHOUT group / super-group g: unique(g) = joint - no_<g> (order-free; shared variance
#:                    credited to nobody). g in timing, target, exec_locked, exec_within, lick, other, contact, and the
#:                    super-groups spout (target + exec_locked + exec_within), task (timing + target), movement (lick +
#:                    exec_* + other: no_movement = the TASK-ONLY model)
#:   <g>_only         that group / super-group alone: task_only (timing + target), lick_only, spout_only, other_only,
#:                    movement_only; no_task = all non-task alone (no_movement keeps contact: not "task alone")
#:   target_first     exec_locked + exec_within residualised on target (pre projection): spout-shared -> target
#:   exec_first       target residualised on exec_locked + exec_within: spout-shared -> executed. The pair BRACKETS
#:                    target vs executed (confounded by design; only exec_within separates them)
#:   task_first       all non-task residualised on task (timing + target): shared -> task
#:   movement_first   task residualised on all non-task: shared -> non-task. The pair brackets Musall's task vs movement
#: Musall: alone = <g>_only / no_task (non-task); unique / task-independent = joint - no_<g> (non-task: joint - task_only);
#: task-aligned = alone - task-independent.
_GROUPSETS = {**{g: (g,) for g in FGROUPS}, **SUPER}
_DROP = {f"no_{g}": gs for g, gs in _GROUPSETS.items() if g != "nontask"}
_ONLY = {"task_only": SUPER["task"], "lick_only": ("lick",), "spout_only": SUPER["spout"], "other_only": ("other",),
         "movement_only": SUPER["movement"]}
_ORDERED = {"target_first": (("target",), ("exec_locked", "exec_within")),     # (A, B): residualise B on A
            "exec_first": (("exec_locked", "exec_within"), ("target",)),
            "task_first": (SUPER["task"], SUPER["nontask"]),
            "movement_first": (SUPER["nontask"], SUPER["task"])}
VARIANTS = ("joint", *_ORDERED, *_DROP, *_ONLY)
#: variants whose per-group PETH parts are saved (the others save data + full only: their parts are not a split)
PART_VARIANTS = ("joint", *_ORDERED)


#: FROZEN DROP-ONE-SPOUT (Priya 10-22: "should we do a leave-one-spout out but without removing movement first" -> yes):
#: `<base>|drop:<p>` = base variant with spout p's TARGET position deviation constrained to the trial-weighted mean of
#: the others, exactly `combined_encoding.raw_blocks(drop_pos=p)`: with deviation blocks D_q = X_q - m_q * shared, the
#: drop-p columns are D_q + m_q / (1 - m_p) * D_p (q != p) and D_p is removed -- a column recombination, so it is one
#: more transform of the same Gram terms (m = the animal's POOLED pre trial shares, frozen like everything else). Scored
#: on every session as the loss of variance vs the base (window offset free, all frames and spout p's own trials).
#: target_first = target BEFORE the executed spout-locked part (upper bound on spout-specific target coding),
#: exec_first = after it (lower bound), joint = ridge split. The executed spout-locked columns are NOT dropped (they
#: are the animal's behaviour at that spout, not its response to the spout).
DROP_BASES = ("joint", "target_first", "exec_first")


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


#: ordered projections: eigenvalues of the A-block Gram below this fraction of the largest are treated as zero (a
#: pseudo-inverse). The v3 code solved with a ~1e-6 ridge; the position deviation blocks are near-collinear by
#: construction (their trial-weighted sum is ~0), so the projection had HUGE weights along that null direction which
#: cancel in pre sessions but not when a post session's spout shares differ -> task_first scored -1.06 on PS95_0818
#: while joint scored 0.36 (10-22 afternoon). 1e-8 was still too permissive: PS95's target block keeps near-null
#: directions at ~1e-7 / 1e-5 of the largest eigenvalue (every real direction is > 1e-2), so target_first scored
#: -163 on PS95_0818 in the first v4 run; 1e-3 drops them (they carry no pre-stroke variance).
PROJ_RCOND = 1e-3


def _projection(Gaa: np.ndarray, Gab: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh(Gaa)
    keep = w > PROJ_RCOND * max(w.max(), 1e-300)
    return (U[:, keep] / w[keep]) @ (U[:, keep].T @ Gab)


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
    if variant in _DROP or variant in _ONLY:
        keep = ~np.isin(grp, _DROP[variant]) if variant in _DROP else np.isin(grp, _ONLY[variant])
        return eye[:, keep], np.flatnonzero(keep)
    if variant in _ORDERED:
        A, Bg = _ORDERED[variant]
        a, b = np.isin(grp, A), np.isin(grp, Bg)
        if not a.any() or not b.any():                   # nothing to residualise: = joint
            return eye, np.arange(p)
        C = eye.copy()
        C[np.ix_(a, b)] = -_projection(Gt[np.ix_(a, a)], Gt[np.ix_(a, b)])     # residualise b on a
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
    label, fits, out_dir, tpath, spath = (*job, None)[:5]
    ce = _ce()
    from wfield_local import config
    an = config.animal_of(label)
    d = session_design(label, _templates_of(tpath, an), spatial=_spatial_of(spath, an))
    f0 = next(iter(fits.values()))
    if spath:
        return _predict_spatial(label, fits, out_dir, d, f0)
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
        if variant in PART_VARIANTS:                      # a reduced model's parts are not a split of anything
            for g, gs in (*((g, (g,)) for g in FGROUPS), ("spout", SUPER["spout"]), ("lick_all", ("lick", "exec_locked"))):
                m = np.isin(gcol, gs)
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


#: evoked-response windows for the spatial outputs (as scripts/lp_encoder_decompose.py): (response, baseline), s
EVOKED_WINDOWS = {"lick": ((0.0, 1.0), (-0.3, -0.1)), "cue": ((0.5, 1.5), (-0.5, 0.0))}


def _predict_spatial(label: str, fits: dict, out_dir: str, d: dict, f0: dict) -> dict:
    """Spatial targets (basis / grid): per variant, the EVOKED response (response window minus baseline, cue- and
    lick-aligned) of the data and of each group's part, per spout x target, from PETHs computed in the small spaces --
    part PETH = PETH(Z) @ weights, data PETH = PETH(Y) @ Ra -- so no (frames x targets) array is built. R^2 and the
    Musall readouts on the cortex-mean target. No drop-spout (area runs have it)."""
    ce = _ce()
    ix = [d["area_names"].index(a) for a in f0["areas"]]
    lazy = d["Rd"] is None
    Ra = d["Ra"][:, ix] if lazy else None
    Rd = None if lazy else d["Rd"][:, ix]
    ci_ = f0["areas"].index("cortex")
    y_c = (d["Y"] @ Ra[:, ci_]) if lazy else Rd[:, ci_]
    Z = d["X"] / f0["scale"]
    fs = d["fs"]
    off = np.arange(int(ce.PETH_S[0] * fs), int(ce.PETH_S[1] * fs) + 1)
    offs = off / fs
    pos_order = sorted(set(d["pos"]))
    wins = {al: ((offs >= r0) & (offs < r1), (offs >= b0) & (offs <= b1))
            for al, ((r0, r1), (b0, b1)) in EVOKED_WINDOWS.items()}

    def evk(P):                                           # (..., off) -> (...,) response minus baseline
        return {al: np.nanmean(P[..., w], -1) - np.nanmean(P[..., b], -1) for al, (w, b) in wins.items()}
    # PETHs of the regressors and of the data, per spout and alignment, in the small spaces
    PZ, PD, n_al = {}, {}, {}
    for p in pos_order:
        m = d["pos"] == p
        for al in ("cue", "lick"):
            fr_ = d["cue_f"][m] if al == "cue" else d["first_f"][m & (d["first_f"] >= 0)]
            n_al[(p, al)] = int(len(fr_))
            if not len(fr_):
                continue
            PZ[(p, al)] = _peth(Z, fr_, off)                                  # (p, off)
            PD[(p, al)] = (Ra.T @ _peth(d["Y"], fr_, off)) if lazy else _peth(Rd, fr_, off)   # (R, off)
    summary = {"label": label}
    for variant, fit in fits.items():
        if "|drop:" in variant:
            continue
        beta = fit["beta_loso"].get(label, fit["beta_all"])
        frozen_from = "loso" if label in fit["beta_loso"] else "all_pre"
        Ce = fit["C"]
        gcol = np.array([FGROUP[v] for v in fit["vcol"]])
        parts = {"full": Ce @ beta}
        if variant in PART_VARIANTS:
            for g, gs in (*((g, (g,)) for g in FGROUPS), ("spout", SUPER["spout"]), ("lick_all", ("lick", "exec_locked"))):
                mm = np.isin(gcol, gs)
                parts[g] = Ce[:, mm] @ beta[mm] if mm.any() else np.zeros((Z.shape[1], len(ix)))
        names = ["data", *parts]
        ev = {al: np.full((len(names), len(pos_order), len(ix)), np.nan, np.float32) for al in ("cue", "lick")}
        for pi, p in enumerate(pos_order):
            for al in ("cue", "lick"):
                if (p, al) not in PZ:
                    continue
                e_d = evk(PD[(p, al)])[al]
                ev[al][0, pi] = e_d
                for k, Wm in enumerate(parts.values(), start=1):
                    ev[al][k, pi] = evk(Wm.T @ PZ[(p, al)])[al]
        pred_c = Z @ parts["full"][:, ci_]
        r2 = {}
        for w, wm in d["win"].items():
            yy, yh = y_c[wm], pred_c[wm]
            r2[w] = float(1.0 - ((yy - yh) ** 2).sum() / max(((yy - yy.mean()) ** 2).sum(), 1e-12)) if wm.any() else None
        musall = {}
        for al, (a0, a1) in MUSALL_WINDOWS.items():
            fr_ = d["cue_f"] if al == "cue" else d["first_f"]
            keep = fr_ >= 0
            if keep.sum() < 2:
                continue
            offm = np.arange(int(round(a0 * fs)), int(round(a1 * fs)))
            mr = musall_r2(_segments(y_c[:, None], fr_[keep], offm), _segments(pred_c[:, None], fr_[keep], offm),
                           d["pos"][keep])
            musall[al] = {k: [float(x) for x in v] for k, v in mr.items()}
        sess = Path(out_dir) / variant / "sessions"
        sess.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sess / f"{label}.npz", evoked_cue=ev["cue"], evoked_lick=ev["lick"])
        meta = {"label": label, "animal": d["animal"], "epoch": d["epoch"], "day": d["day"],
                "order": f"frozen_pre:{variant}", "variant": variant, "frozen_from": frozen_from,
                "pre_labels": fit["pre_labels"], "alphas": fit["alphas"], "loso_r2_pre": fit["loso_r2"],
                "gate": os.environ.get("OROFACIAL_GATE", "reference"), "lick_set": d["lick_set"],
                "n_trials": d["n_trials"], "n_aligned": {f"{p}|{al}": n for (p, al), n in n_al.items()},
                "pos_order": pos_order, "evoked_parts": names, "targets": f0["areas"], "evoked_windows": EVOKED_WINDOWS,
                "frozen_r2_cortex": r2, "r2_musall": musall, "areas": ["cortex"], "stats_version": STATS_VERSION}
        (sess / f"{label}.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        summary[variant] = r2
        summary["frozen_from"] = frozen_from
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
        S = A[(A.spout == "far_contra") & (A.window == "lick") & (A.subset == "own") & (A["area"] == area)]
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
        S = A[A["area"] == area].pivot_table(index=["align", "measure", "variant"], columns=["animal", "epoch"], values="r2")
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
    ap.add_argument("--target", default="areas", choices=("areas", "basis", "grid"),
                    help="Allen areas (default) | per-animal shared SVD basis | 8 x 8 pixel grid (SPATIAL TARGETS)")
    a = ap.parse_args(argv)
    if a.target != "areas":
        a.no_drop_spout = True                           # spatial runs: maps of the decomposition, no drop-spout
        if a.variants == list(VARIANTS):
            a.variants = list(PART_VARIANTS)             # the variants whose parts are a split
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
    tp = out / "fit" / "templates.pkl"
    (out / "fit").mkdir(parents=True, exist_ok=True)
    spath = str(out / "fit") if a.target != "areas" else None
    spat_ok = a.target == "areas" or all((out / "fit" / f"spatial_{config.animal_of(x)}.pkl").exists() for x in pre)
    saved = pickle.loads(sp.read_bytes()) if (a.reuse_stats and sp.exists() and tp.exists() and spat_ok) else None
    if saved is not None and any(s_.get("stats_version") != STATS_VERSION for s_ in saved):
        print(f"  saved statistics are not STATS_VERSION {STATS_VERSION}: recomputing", flush=True)
        saved = None
    if saved is not None:                                # a re-scoring run: the pre statistics are already saved
        got = [(s_["label"], s_) for s_ in saved if s_["label"] in pre and s_.get("pos_order")]
        print(f"  reusing {len(got)} pre-session statistics from {sp}", flush=True)
    else:
        # pass 1 (behaviour only): the animals' SPOUT-LOCKED templates from their pre sessions (FGROUP comment)
        tparts, tfail = ak.fan_sessions(pre, template_job, jobs=a.jobs, label="template")
        if tfail:
            raise RuntimeError(f"template pass failed: {tfail}")
        tp.write_bytes(pickle.dumps(build_templates([t for _l, t in tparts]), protocol=pickle.HIGHEST_PROTOCOL))
        print(f"  spout-locked templates -> {tp}", flush=True)
        if a.target == "basis":                          # pass 2: the per-animal shared spatial basis
            bparts, bfail = ak.fan_sessions(pre, basis_job, jobs=a.jobs, label="basis")
            if bfail:
                raise RuntimeError(f"basis pass failed: {bfail}")
            for an_, sd_ in build_basis([b for _l, b in bparts]).items():
                (out / "fit" / f"spatial_{an_}.pkl").write_bytes(pickle.dumps(sd_, protocol=pickle.HIGHEST_PROTOCOL))
                print(f"  {an_} shared basis: K = {len(sd_['names'])} components, {sd_['evr'][-1]:.3f} of pre variance",
                      flush=True)
        elif a.target == "grid":
            g_ = grid_spatial(pre[0])
            for an_ in sorted({config.animal_of(x) for x in labels}):
                (out / "fit" / f"spatial_{an_}.pkl").write_bytes(pickle.dumps(g_, protocol=pickle.HIGHEST_PROTOCOL))
            print(f"  grid: {len(g_['names'])} blocks of {GRID_BLOCK} x {GRID_BLOCK} px", flush=True)
        got, fail = ak.fan_sessions([(x, str(tp), spath) for x in pre], stats_job, jobs=a.jobs)
        got = [(it[0], s_) for it, s_ in got]
        for s_ in (s_ for _l, s_ in got):
            s_["stats_version"] = STATS_VERSION
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
    jobs = [(x, fits[config.animal_of(x)], str(out), str(tp), spath) for x in labels if config.animal_of(x) in fits]
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

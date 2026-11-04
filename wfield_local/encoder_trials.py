"""Reader for the combined encoder's PER-TRIAL outputs, `sessions/<label>_trials.npz` (Priya 11-04: "we must be able to
report every camera trial class").

The O2 run does NOT know the classes (they are this run's camera classes, fixed later than the bundle): it scores EVERY
cue trial -- fitted trials (E = engaged, the training rows) with their held-out fold prediction, every other trial with
the full-data model, never in training -- and saves per-trial outputs keyed by trial_id. Classes are joined HERE, locally,
by (label, trial_id) from the run's `trial_attempts.csv` (`lp_licks.attempts_path()`: cls, cls_raw, attempted,
canonical) and, optionally, `trial_classes.session_trials` (jaw_judged: the close-mid "jaw could not be judged" flag).

    z = load(path)                          # dict of arrays (+ "table": the per-trial DataFrame)
    z = join_classes(z, label)              # + cls / cls_raw / attempted / canonical [/ jaw_judged]
    class_peth(z, mask, "cue")              # (parts, positions, areas, off): as the session npz `peth`, over `mask`
    class_sse(z, mask, "cue")               # binned SSE / SST per time-resolved model (as `sse` / `sst`, binned)
    class_full_fine(z, mask, "cue")         # the full model, frame-resolved: sse, sst, sse_trial, sst_trial
    class_summary(z, label)                 # per class: n trials, cortex cvR2, trial-to-trial R2 (full model)

What is stored (`combined_encoding.TRIALS_ENV`): `t_*` the trial table; `sse_bins_<align>` (models, trials, bins, areas)
and `sst_bins_<align>` / `nfr_bins_<align>`; in mode full: `frames` (parts, rows, areas) = data, the full model's parts
and its held-out prediction "cv" for the fitted rows followed by the scored-only rows, with `t_cue_row` / `t_lick_row` /
`t_row_lo` / `t_row_hi` locating each trial (`combined_encoding.trial_rows`). The LICK alignment of this file is the first
LP lick, else the first DAQ contact in cue + 70 ms .. stop (`t_timing_source` lp / daq / none, Priya 11-04 Q9); the
session npz's peth_lick / sse_lick are LP-only (= timing_source lp).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: what each class is in the attempts table (`scripts/trial_attempts.py` names): hit = contact
CLASS_ORDER = ("hit", "incomplete", "jaw_only", "no_try", "unknown", "stopped", "after_end", "early_run")


def load(path) -> dict:
    with np.load(Path(path), allow_pickle=False) as f:
        z = {k: f[k] for k in f.files}
    z["table"] = pd.DataFrame({k[2:]: v for k, v in z.items() if k.startswith("t_")})
    return z


def join_classes(z: dict, label: str, attempts=None, jaw: bool = False) -> dict:
    """``z`` with cls / cls_raw / attempted / canonical joined onto the trial table by (label, trial_id) from
    ``attempts`` (a path or DataFrame; default this run's `lp_licks.attempts_path()`); a trial without a row gets cls
    "missing". ``jaw``: also jaw_judged from `trial_classes.session_trials(label)` (reads the share)."""
    if attempts is None:
        from wfield_local import lp_licks
        attempts = lp_licks.attempts_path()
    A = attempts if isinstance(attempts, pd.DataFrame) else pd.read_csv(attempts)
    A = A[A.label == label]
    cols = [c for c in ("cls", "cls_raw", "attempted", "canonical") if c in A]
    T = z["table"].drop(columns=[c for c in cols if c in z["table"]]).merge(
        A[["trial_id", *cols]].drop_duplicates("trial_id"), on="trial_id", how="left")
    T["cls"] = T["cls"].fillna("missing")
    if jaw:
        from wfield_local import trial_classes as tc
        J = tc.session_trials(label)[["trial_id", "jaw_judged"]]
        T = T.merge(J, on="trial_id", how="left")
    z = dict(z)
    z["table"] = T
    return z


def _gather(z: dict, align: str):
    from wfield_local.combined_encoding import trial_rows
    t = z["table"]
    return trial_rows(t[f"{align}_row"].to_numpy(), t.row_lo.to_numpy(), t.row_hi.to_numpy(), z["off"])


def class_peth(z: dict, mask, align: str = "cue", positions=None) -> tuple[np.ndarray, list, list]:
    """(peth (parts, positions, areas, off), parts, positions): trial-averaged traces of every stored part over the
    trials in ``mask``, per position -- the session npz's `peth` / `peth_lick` restricted to those trials (parts = the
    npz's `peth_parts` + "cv", the held-out full prediction). Needs mode full."""
    if "frames" not in z:
        raise ValueError("no frame traces in this file (mode sse): per-class PETHs need TRIALS mode full")
    t, F = z["table"], z["frames"]
    mask = np.asarray(mask, bool)
    pos = t.pos.to_numpy()
    positions = sorted(set(pos)) if positions is None else list(positions)
    idx, ok = _gather(z, align)
    out = np.full((len(F), len(positions), F.shape[2], len(z["off"])), np.nan, np.float32)
    for pi, p in enumerate(positions):
        m = mask & (pos == p)
        if not m.any():
            continue
        for ci in range(len(F)):
            seg = np.where(ok[m][..., None], F[ci][idx[m]].astype(np.float32), np.nan)
            with np.errstate(invalid="ignore"), _quiet():
                out[ci, pi] = np.nanmean(seg, 0).T
    return out, [str(x) for x in z["parts"]], positions


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        return self._w.__exit__(*a)


def class_sse(z: dict, mask, align: str = "cue") -> dict:
    """Summed per-trial binned SSE (models, bins, areas) and SST (bins, areas) over ``mask``, the models' names, and
    whether each model scores the scored-only trials (shuffle / relabel models do not: NaN there, skipped by the sum
    only if the mask has no such trials -- check ``scorable``)."""
    mask = np.asarray(mask, bool)
    sse = z[f"sse_bins_{align}"][:, mask].astype(np.float64)
    return {"sse": sse.sum(1), "sst": z[f"sst_bins_{align}"][mask].astype(np.float64).sum(0),
            "models": [str(x) for x in z["tr_models"]], "scorable": z["tr_scorable"].astype(bool),
            "n_frames": z[f"nfr_bins_{align}"][mask].astype(int).sum(0), "edges_s": z["bin_edges_s"]}


def class_full_fine(z: dict, mask, align: str = "cue") -> dict:
    """The FULL model frame-resolved over ``mask`` (mode full): sse / sst (areas, off) as the session npz's `sse[full]` /
    `sst` and the trial-to-trial sse_trial / sst_trial (each position's mean over the MASKED trials removed from data
    and prediction). Held-out on fitted trials, full-data on scored-only ones."""
    t, F = z["table"], z["frames"]
    parts = [str(x) for x in z["parts"]]
    D, P = F[parts.index("data")], F[parts.index("cv")]
    mask = np.asarray(mask, bool)
    idx, ok = _gather(z, align)
    idx, ok, pos = idx[mask], ok[mask], t.pos.to_numpy()[mask]
    Rd = np.where(ok[..., None], D[idx].astype(np.float64), np.nan)
    Rp = np.where(ok[..., None], P[idx].astype(np.float64), np.nan)

    def dev(X):
        out = np.empty_like(X)
        for p in np.unique(pos):
            m = pos == p
            with _quiet():
                out[m] = X[m] - np.nanmean(X[m], 0, keepdims=True)
        return out
    dd = dev(Rd)
    return {"sse": np.nansum((Rd - Rp) ** 2, 0).T, "sst": np.nansum((Rd - z["rd_mean"]) ** 2, 0).T,
            "sse_trial": np.nansum((dd - dev(Rp)) ** 2, 0).T, "sst_trial": np.nansum(dd ** 2, 0).T, "n": int(mask.sum())}


def class_summary(z: dict, label: str | None = None, by: str = "cls", area: int = -1, align: str = "cue",
                  window_s=(-3.0, 3.0)) -> pd.DataFrame:
    """Per class (column ``by`` of the joined table): trials, fitted / scored-only, lick timing sources, and the FULL
    model's cvR2 (binned SSE inside ``window_s``) and, in mode full, its trial-to-trial R2 over the same window, for
    area index ``area`` (-1 = cortex, -2 = cortex_stat)."""
    t = z["table"]
    if by not in t:
        if label is None:
            raise ValueError(f"no {by!r} column: join_classes(z, label) first")
        z = join_classes(z, label)
        t = z["table"]
    mi = [str(x) for x in z["tr_models"]].index("full")
    e = z["bin_edges_s"]
    bw = (e[:-1] >= window_s[0] - 1e-9) & (e[1:] <= window_s[1] + 1e-9)
    rows = []
    order = [c for c in CLASS_ORDER if c in set(t[by])] + sorted(set(t[by]) - set(CLASS_ORDER))
    for c in order:
        m = (t[by] == c).to_numpy()
        s = class_sse(z, m, align)
        r = {by: c, "n": int(m.sum()), "n_fitted": int((m & t.fitted.to_numpy()).sum()),
             "n_lick_lp": int((m & (t.timing_source == "lp").to_numpy()).sum()),
             "n_lick_daq": int((m & (t.timing_source == "daq").to_numpy()).sum()),
             "cvR2": float(1 - s["sse"][mi][bw, area].sum() / s["sst"][bw, area].sum())}
        if "frames" in z:
            f = class_full_fine(z, m, align)
            ow = np.isin(z["bin_of_off"], np.flatnonzero(bw))
            r["trial_R2"] = float(1 - f["sse_trial"][area, ow].sum() / f["sst_trial"][area, ow].sum())
        rows.append(r)
    return pd.DataFrame(rows)

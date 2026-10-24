"""Choosing the fit and session ROWS: engaged trials balanced over positions, between-trial 5 Hz bins, even draws over days / groups, and reading the chosen rows off the memory-mapped spectrograms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ARRIVAL_S, BUSY_S, EDGE_BINS, PRECUE_EXT_S, TRIAL_DS
from .files import _sidecar_module, load_plan, npz_memmap, npz_small, out_dir, plan_views, rows_of, view_out_dir


# ----------------------------------------------------------------------------------------------- rows (plan / O2)
def session_candidates(key: str, z, n_frames: int, timing: dict | None = None,
                       state: np.ndarray | None = None, strobe_daq_s: np.ndarray | None = None) -> tuple[list, list]:
    """A task's engaged trials [(key, trial, position)] and between-trial 5 Hz bins [(key, bin, state)] (its bundle
    trial grid ``z``: position, cue_daq_s, daq_fit frame -> DAQ s). ``timing`` (`face_rhythm_timing.load`, 10-23): the
    bins' TRUE DAQ times, and LOSSY / unaligned bins and trials (any lossy grid bin) are left out; without it the
    linear frame -> DAQ fit (wrong on heavy-drop videos). ``state`` per 5 Hz bin ("run" / "still" / "other",
    `between_states`) rides along for a balanced draw (`draw_rows` ``run_share``). ``strobe_daq_s`` (per trial, the
    position strobe = SPOUT ARRIVAL; NaN = unknown): each trial carries its 5 Hz PRE-CUE bins (Priya 10-23: spout
    arrival + "more ENL licks, peri-cue from 4 s before"): within ARRIVAL_S of its own strobe OR from cue + PRECUE_EXT_S,
    and before the 25 Hz trial grid (cue + t_ms[0]); non-lossy; a 4th element drawn with the trial. The between pool
    then also excludes all of that through the cue, whatever the trial's strobe -> cue lead."""
    frt = _sidecar_module("face_rhythm_timing")
    n5 = int(n_frames) // 50
    if timing is not None:
        t5 = timing["t5"][:n5]
        bad5 = frt.lossy(timing["drop5"][:n5])
        bad_tr = frt.lossy(timing["drop_trials"]).any(axis=1)
    else:
        t5 = z["daq_fit"][0] * (np.arange(n5) * 50 + 24.5) + z["daq_fit"][1]
        bad5, bad_tr = np.zeros(n5, bool), np.zeros(len(z["position"]), bool)
    busy = bad5.copy()
    busy[:EDGE_BINS] = busy[-EDGE_BINS:] = True
    for c in z["cue_daq_s"]:
        busy |= (t5 >= c + BUSY_S[0]) & (t5 <= c + BUSY_S[1])
    if strobe_daq_s is None:
        trials = [(key, i, str(p)) for i, p in enumerate(z["position"]) if not bad_tr[i]]
    else:
        trials = []
        for i, (p, sb, c) in enumerate(zip(z["position"], strobe_daq_s, z["cue_daq_s"])):
            if bad_tr[i]:
                continue
            win = (t5 >= c + PRECUE_EXT_S) & (t5 < c + float(z["t_ms"][0]) / 1000.0)
            busy |= (t5 >= c + PRECUE_EXT_S) & (t5 <= c)
            if np.isfinite(sb):
                win |= (t5 >= sb + ARRIVAL_S[0]) & (t5 <= sb + ARRIVAL_S[1]) & (t5 < c + float(z["t_ms"][0]) / 1000.0)
                busy |= (t5 >= sb + ARRIVAL_S[0]) & (t5 <= c)
            trials.append((key, i, str(p), tuple(int(b) for b in np.flatnonzero(win & ~bad5))))
    st = state if state is not None else np.full(n5, "other")
    return trials, [(key, int(b), str(st[b])) for b in np.nonzero(~busy)[0]]


def trial_strobes(rv, animal: str, date: str, trial_ids: np.ndarray, cues: np.ndarray) -> np.ndarray:
    """Per trial (bundle ``trial_id``, = v36 ``ix_trial_id``), the DAQ time of its position strobe (spout arrival):
    cue + v36 ``ix_strobe_ms`` (strobe relative to the cue); NaN when v36 has no strobe for the trial."""
    from wfield_local.orofacial_cohort import RUN
    w = np.load(Path(rv.root("orofacial")) / RUN / "sessions" / animal / date / "windows.npz")
    tid, sm = w["ix_trial_id"], w["ix_strobe_ms"].astype(float)
    first = {}
    for t, v in zip(tid[np.r_[0, np.flatnonzero(np.diff(tid)) + 1]], sm[np.r_[0, np.flatnonzero(np.diff(tid)) + 1]]):
        first.setdefault(int(t), float(v))
    rel = np.array([first.get(int(t), np.nan) for t in trial_ids], float) / 1000.0
    return np.asarray(cues, float) + rel


def between_states(rv, animal: str, date: str, t5: np.ndarray) -> np.ndarray:
    """Per 5 Hz bin (true DAQ s), from the CANONICAL per-session behaviour events on the server (`behavior_events`,
    ``behavior_summary/events/<animal>/<date>.npz``; no DAQ read): "run" inside a running bout (> 3 mm/s, gaps
    < 0.3 s merged, >= 2 s), "still" inside a REST period (the lab's quiet definition: between trials with the spout
    docked, not running, not near licks / rewards; `quiet_periods.rest_mask`), else "other" (Priya 10-23: the
    non-trial rows split evenly between running and still)."""
    from wfield_local import behavior_events as be
    out = np.full(len(t5), "other", dtype="<U5")
    ev = be.get_or_compute(rv, animal, date)
    if ev is None:
        return out
    fs = float(ev["fs"])
    for st, (a_, b_) in (("still", ("rest_starts", "rest_stops")), ("run", ("running_starts", "running_stops"))):
        for a, b in zip(np.asarray(ev[a_], float) / fs, np.asarray(ev[b_], float) / fs):
            out[(t5 >= a) & (t5 <= b)] = st
    return out


def even_draw(items: list, n: int, rng, group=None) -> list:
    """``min(n, len(items))`` of ``items`` spread as EVENLY as possible over their groups (default: the recording DAY of
    the item's key, `session_group`): quotas dealt one at a time round-robin over the days that still have items (a
    short day gives all it has, the rest go to the others; the dealing order is random, so a remainder does not always
    land on the earliest days), then a seeded random draw within each day."""
    group = group or (lambda it: session_group(it[0]))
    by = {}
    for it in items:
        by.setdefault(group(it), []).append(it)
    days = sorted(by)
    quota = dict.fromkeys(days, 0)
    left = min(n, len(items))
    deal = [days[i] for i in rng.permutation(len(days))]   # remainders to RANDOM days (not always the first ones)
    while left:
        for d in deal:
            if left and quota[d] < len(by[d]):
                quota[d] += 1
                left -= 1
    out = []
    for d in days:
        if quota[d]:
            out += [by[d][i] for i in sorted(rng.choice(len(by[d]), quota[d], replace=False))]
    return out


def draw_rows(trials: list, between: list, n_trials: int, between_frac: float, rng, ds: int = TRIAL_DS,
              run_share: float | None = None, even_days: bool = False) -> dict:
    """Positions balanced (n_trials // n_positions each, or all a position has) + between_frac x the trial rows of
    between-trial bins -> {"trials": {key: [trial, ...]}, "between": {key: [bin, ...]}, "ds": ds} (sorted). ``ds``:
    25 Hz trial bins averaged per fit row (2 = 12.5 Hz; 4 = 6.25 Hz -- twice the trials for the same rows: adjacent
    rows of one trial are near-duplicates, the spectrogram windows being far longer than 80 ms; Priya 10-23).
    ``run_share`` (between items carry a state, `session_candidates`): that share of the between rows from RUNNING
    bins, the rest from STILL ones (either short -> the other fills); None = uniform over all between bins.
    ``even_days`` (Priya 10-23): every draw (each position's trials, the run bins, the still bins) spread evenly over the
    recording DAYS (`even_draw`), so each pre day weighs the same within its animal; else uniform over the pooled items."""
    def take(items, k):
        if even_days:
            return even_draw(items, k, rng)
        return [items[i] for i in rng.choice(len(items), min(k, len(items)), replace=False)] if k else []
    pos = sorted({t[2] for t in trials})
    per = n_trials // max(len(pos), 1)
    pick = []
    for p in pos:
        pick += take([t for t in trials if t[2] == p], per)
    nb = min(int(round(between_frac * len(pick) * (151 // ds))), len(between))
    if run_share is None or not nb:
        bsel = take(between, nb)
    else:
        pools = {st: [b for b in between if len(b) > 2 and b[2] == st] for st in ("run", "still")}
        want = {"run": int(round(run_share * nb))}
        want["still"] = nb - want["run"]
        short = {st: max(0, want[st] - len(pools[st])) for st in want}
        want = {"run": want["run"] - short["run"] + short["still"], "still": want["still"] - short["still"] + short["run"]}
        bsel = []
        for st in ("run", "still"):
            bsel += take(pools[st], min(want[st], len(pools[st])))
    out = {"trials": {}, "between": {}}
    if any(len(t) > 3 for t in pick):
        out["arrival"] = {}
    for k, i, _, *arr in pick:
        out["trials"].setdefault(k, []).append(int(i))
        if arr and arr[0]:
            out["arrival"].setdefault(k, []).extend(arr[0])
    for k, b, *_ in bsel:
        out["between"].setdefault(k, []).append(int(b))
    n_state = {st: sum(1 for b in bsel if len(b) > 2 and b[2] == st) for st in ("run", "still", "other")}
    return {**{kind: {k: sorted(v) for k, v in sorted(d.items())} for kind, d in out.items()}, "ds": int(ds),
            "n_between": n_state}


def gather_rows(sd: Path, picks: dict, weighted: bool = True) -> tuple[np.ndarray, list]:
    """The rows a `draw_rows` pick names, from the outputs: trial windows at 12.5 Hz + session bins. -> (X (n, 2P, F)
    float32 UNSCALED, index [(key, kind, trial, bin)]). Rows with any NaN are dropped."""
    X, index = [], []
    ds = int(picks.get("ds", TRIAL_DS))                                  # plans before 10-23: TRIAL_DS
    for key, tr in picks["trials"].items():
        S = spec_of(sd, key, "trials", weighted)
        for ti in tr:
            x = rows_of(S[ti])                                           # (151, 2P, F)
            nb = (x.shape[0] // ds) * ds
            x = x[:nb].reshape(-1, ds, *x.shape[1:]).mean(1)
            ok = np.isfinite(x).all(axis=(1, 2))
            X.append(x[ok])
            index += [(key, "trial", int(ti), int(j)) for j in np.nonzero(ok)[0]]
    for kind in ("arrival", "between"):
        _session_rows(sd, picks.get(kind, {}), kind, weighted, X, index)
    return np.concatenate(X).astype(np.float32), index


def _session_rows(sd: Path, per_key: dict, kind: str, weighted: bool, X: list, index: list) -> None:
    for key, bins in per_key.items():
        S = spec_of(sd, key, "session", weighted)
        b = np.sort(np.asarray(bins, int))
        x = rows_of(S[b])
        ok = np.isfinite(x).all(axis=(1, 2))
        X.append(x[ok])
        index += [(key, kind, -1, int(v)) for v in b[ok]]


def spec_of(sd: Path, key: str, kind: str, weighted: bool = True):
    """Task ``key``'s ``kind`` ("trials" | "session") spectrogram, (lead..., 2, P, F): the reference run's memmap, or for
    a MULTI-VIEW set (`face_rhythm_multiview`) a lazy `ViewStack` of every view's memmap along the point axis, each
    view divided by its balance divisor (``view_scales.json``; ``weighted=False`` -> 1, as `build` needs before
    computing them). Trial rows match by index (cue times checked); session bins by nearest DAQ time."""
    ref = out_dir(sd) / f"{key}_{kind}.npz"
    S = npz_memmap(ref, "spec")
    views = plan_views(sd)
    if not views:
        return S
    fmv = _sidecar_module("face_rhythm_multiview")
    pl = load_plan(sd)
    rc = pl["views"]["ref"]
    arrays, maps = [S], [None]
    t_ref = npz_small(ref, ("cue_daq_s",) if kind == "trials" else ("t_daq_s",))
    frt = _sidecar_module("face_rhythm_timing")

    def true_t5(out: Path, k: str, lin: np.ndarray) -> np.ndarray:      # true session bin times when cached
        tm = frt.load(out.parent, k)
        return tm["t5"][:len(lin)] if tm is not None else lin
    if kind == "session":
        t_ref["t_daq_s"] = true_t5(out_dir(sd), key, t_ref["t_daq_s"])
    for v in views:
        f = view_out_dir(sd, v["cam"], v["bundle"]) / f"{fmv.view_key(key, rc, v['cam'])}_{kind}.npz"
        arrays.append(npz_memmap(f, "spec"))
        if kind == "trials":
            fmv.check_trials_match(t_ref["cue_daq_s"], npz_small(f, ("cue_daq_s",))["cue_daq_s"], f.name)
            maps.append(None)
        else:
            tv = true_t5(f.parent, fmv.view_key(key, rc, v["cam"]), npz_small(f, ("t_daq_s",))["t_daq_s"])
            maps.append(fmv.session_row_map(t_ref["t_daq_s"], tv))
    divs = None
    vs = sd / "view_scales.json"
    if weighted and vs.exists():
        divs = json.loads(vs.read_text())[key.split("_")[0]]
    return fmv.ViewStack(arrays, maps, divs)


def session_group(key: str) -> str:
    """A task's recording DAY (``PS92_20260812_cam4_v2`` -> ``PS92_20260812``): leave-one-session-out leaves out the
    whole day, both videos of a split session."""
    return key.split("_cam")[0]

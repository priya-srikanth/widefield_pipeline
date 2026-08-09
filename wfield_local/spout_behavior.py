"""Spout behavior-session figures (1 cue + 6 spout positions).

The behavior task-controller already emits a scored per-trial table (``trials.csv``:
``pos_idx``/``pos_name``, ``hit``, ``miss``, ``lick_in_response_window``, ``reward_type`` …),
so this module *reads* that table rather than re-detecting licks from the DAQ. It is the
widefield analogue of ``stroke_orofacial_pipeline``'s ``spout_behavior`` (that rig had 2 cues +
L/R; this rig has 1 cue and a 2x3 grid of spout positions: close/far x L/center/R).

Two things it adds on top of the raw scores:

* **Engagement gate.** Reward is auto-held after a run of misses (the task's
  ``auto_hold_after_miss_threshold``), so a sated animal's late misses are *disengagement*, not
  spatial *inaccuracy*. :func:`flag_engagement` separates a terminal sated tail (and any
  mid-session response collapse) from genuine misses; per-position accuracy is reported on the
  ENGAGED trials by default, with the raw all-trial rate shown alongside for transparency.
* **First-lick latency.** Derived from ``events.csv`` (first ``lick_on`` after each trial's
  ``cue``), if that file is present.

CLI::

    python -m wfield_local.spout_behavior 20260806                 # per-session figs, all animals
    python -m wfield_local.spout_behavior 20260806 --only PS92     # one animal
    python -m wfield_local.spout_behavior --cohort --from curated  # pooled cohort figures
    python -m wfield_local.spout_behavior 20260806 --dry-run       # discover only, write nothing
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.paths import PathResolver

ANIMAL_RE = re.compile(r"PS\d+")

# Canonical 6-position layout. row 0 = close (2 mm), row 1 = far (4 mm); col 0/1/2 = L/center/R.
# (idx matches the task's pos_idx / roi_activity.POSITION_NAMES.)
POSITIONS = [
    {"idx": 1, "name": "close_L",      "ring": "close", "side": "L", "row": 0, "col": 0},
    {"idx": 0, "name": "close_center", "ring": "close", "side": "C", "row": 0, "col": 1},
    {"idx": 2, "name": "close_R",      "ring": "close", "side": "R", "row": 0, "col": 2},
    {"idx": 4, "name": "far_L",        "ring": "far",   "side": "L", "row": 1, "col": 0},
    {"idx": 3, "name": "far_center",   "ring": "far",   "side": "C", "row": 1, "col": 1},
    {"idx": 5, "name": "far_R",        "ring": "far",   "side": "R", "row": 1, "col": 2},
]
IDX_ORDER = [p["idx"] for p in POSITIONS]           # bar/plot order: close L,C,R then far L,C,R
POS_BY_IDX = {p["idx"]: p for p in POSITIONS}
# color encodes ring (close/far); marker + linestyle encode side (L/center/R) so all 6 positions are
# distinguishable in line graphs (and in greyscale).
SIDE_MARKER = {"L": "o", "C": "D", "R": "^"}
SIDE_LS = {"L": "-", "C": "--", "R": ":"}


# --------------------------------------------------------------------------- loading

def load_trials(session_dir: Path) -> pd.DataFrame:
    """Read ``trials.csv`` and keep the real, scored trials (hit XOR miss).

    Drops the phantom setup row (trial_id 0, start==end, neither hit nor miss). Adds a boolean
    ``responded`` (licked in the response window) and ``is_free`` (free-reward) column.
    """
    df = pd.read_csv(session_dir / "trials.csv")
    for c in ("pos_idx", "hit", "miss", "lick_in_response_window", "free_reward_trial",
              "free_reward_delivered", "reward_delivered"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df = df[(df["hit"] == 1) | (df["miss"] == 1)].copy()      # scored trials only
    df = df.sort_values("trial_id").reset_index(drop=True)
    df["responded"] = df["lick_in_response_window"].astype(bool)
    # "not an accuracy test" == free water was actually DELIVERED regardless of the lick. A trial merely
    # *designated* free_reward_trial (the task's post-miss re-engagement offer) but with
    # free_reward_delivered==0 still required the animal to respond and is scored normally — keep it.
    df["is_free"] = df.get("free_reward_delivered", 0).astype(int) == 1
    df["free_designated"] = df.get("free_reward_trial", 0).astype(int) == 1
    return df


def load_gui_licks(session_dir: Path) -> dict | None:
    """Parse ``events.csv`` once into the GUI's own discrete lick events (device clock, seconds).

    Returns ``{"all_s", "cue_by_trial", "lick_by_trial", "precue_reset_by_trial"}`` or ``None`` if the
    file is missing/unreadable. ``all_s`` = every ``lick_on`` time sorted; ``cue_by_trial`` = per-trial
    cue time (Series); ``lick_by_trial`` = per-trial sorted ``lick_on`` times; ``precue_reset_by_trial`` =
    per-trial count of ``pre_cue_reset_by_lick`` events (anticipatory ENL-reset licks).
    """
    ev_path = session_dir / "events.csv"
    if not ev_path.exists():
        return None
    try:
        ev = pd.read_csv(ev_path, usecols=lambda c: c in ("device_t_ms", "event_name", "trial_id"))
    except Exception:
        return None
    ev["device_t_ms"] = pd.to_numeric(ev["device_t_ms"], errors="coerce")
    ev["trial_id"] = pd.to_numeric(ev["trial_id"], errors="coerce")
    licks = ev[ev["event_name"] == "lick_on"]
    resets = ev[ev["event_name"] == "pre_cue_reset_by_lick"]
    return {
        "all_s": np.sort(licks["device_t_ms"].to_numpy()) / 1000.0,
        "cue_by_trial": ev[ev["event_name"] == "cue"].groupby("trial_id")["device_t_ms"].min() / 1000.0,
        "lick_by_trial": {tid: np.sort(g["device_t_ms"].to_numpy()) / 1000.0
                          for tid, g in licks.groupby("trial_id")},
        "precue_reset_by_trial": resets.groupby("trial_id")["device_t_ms"].count(),
    }


def first_lick_latency_s(session_dir: Path, trials: pd.DataFrame, max_s: float,
                         gui: dict | None = None) -> pd.Series:
    """Per-trial first-lick latency (s): first ``lick_on`` at/after the trial's ``cue``. NaN where no
    events, no cue, or latency exceeds ``max_s``. Pass a pre-loaded ``gui`` dict to avoid re-parsing."""
    idx = pd.Series(np.nan, index=trials.index)
    gui = gui if gui is not None else load_gui_licks(session_dir)
    if gui is None:
        return idx
    cue_t, lick_by_trial = gui["cue_by_trial"], gui["lick_by_trial"]
    for i, row in trials.iterrows():
        tid = row["trial_id"]
        if tid not in cue_t.index:
            continue
        c = cue_t.loc[tid]
        arr = lick_by_trial.get(tid)
        if arr is None or not np.isfinite(c):
            continue
        after = arr[arr >= c]
        if after.size:
            lat = after[0] - c
            if 0 <= lat <= max_s:
                idx.loc[i] = lat
    return idx


# --------------------------------------------------------------------------- lick microstructure

def segment_bouts(onsets_s, max_ili_s: float, min_bout_licks: int) -> list[tuple[float, float, int]]:
    """Split a sorted lick-onset train (seconds) into bouts at ILI gaps > ``max_ili_s``.

    Returns ``[(start_s, end_s, n_licks), ...]`` for bouts with >= ``min_bout_licks`` licks
    (stroke_orofacial `find_lick_bouts`: strict-> split, equality stays in-bout)."""
    onsets_s = np.asarray(onsets_s, dtype=float)
    if onsets_s.size == 0:
        return []
    splits = np.flatnonzero(np.diff(onsets_s) > max_ili_s) + 1
    bouts = []
    for run in np.split(onsets_s, splits):
        if run.size >= min_bout_licks:
            bouts.append((float(run[0]), float(run[-1]), int(run.size)))
    return bouts


def lick_microstructure(session_dir: Path, trials: pd.DataFrame, params: dict,
                        gui: dict | None = None) -> dict | None:
    """Lick-rate / ILI / bout / peri-cue microstructure from the GUI lick events. ``None`` if no events.

    Per-position (engaged trials): mean post-cue licks/trial, mean anticipatory (pre-cue) licks/trial,
    median within-trial lick rate (Hz, from median post-cue ILI). Session-level: overall lick rate,
    ILI median, bout count/size/duration/within-bout-ILI. Plus peri-cue delays for the raster/PSTH.
    """
    gui = gui if gui is not None else load_gui_licks(session_dir)
    if gui is None or gui["all_s"].size == 0:
        return None
    lk = params["licking"]
    w0, w1 = lk["peri_cue_window_s"]
    resp = lk["response_window_s"]
    cue_t, lick_by_trial = gui["cue_by_trial"], gui["lick_by_trial"]

    reset_by_trial = gui["precue_reset_by_trial"]     # ENL-reset licks = impulsive/anticipatory pre-cue
    per_trial = []          # (pos_idx, n_post, n_pre, rate_hz)
    raster = []             # (delays_within_window, pos_idx, trial_order)
    for order, (i, row) in enumerate(trials.iterrows()):
        tid, pos = row["trial_id"], int(row["pos_idx"])
        if tid not in cue_t.index or not np.isfinite(cue_t.loc[tid]):
            continue
        c = cue_t.loc[tid]
        arr = lick_by_trial.get(tid, np.empty(0))
        d = arr - c
        win = d[(d >= w0) & (d <= w1)]
        raster.append((win, pos, order))
        post = d[(d >= 0) & (d <= resp)]
        # anticipatory = licks during the enforced no-lick period that reset the cue timer (pre_cue_reset)
        n_pre = int(reset_by_trial.get(tid, 0))
        rate = (1.0 / np.median(np.diff(post))) if post.size >= 2 else np.nan
        per_trial.append((pos, int(post.size), n_pre, rate))
    pt = pd.DataFrame(per_trial, columns=["pos_idx", "n_post", "n_pre", "rate_hz"])

    rows = []
    for idx in IDX_ORDER:
        sub = pt[pt["pos_idx"] == idx]
        rows.append({
            "pos_idx": idx, "pos_name": POS_BY_IDX[idx]["name"],
            "licks_per_trial": float(sub["n_post"].mean()) if len(sub) else np.nan,
            "anticipatory_licks": float(sub["n_pre"].mean()) if len(sub) else np.nan,
            "lick_rate_hz": float(np.nanmedian(sub["rate_hz"])) if len(sub) else np.nan,
        })
    per_pos = pd.DataFrame(rows)

    all_s = gui["all_s"]
    ili = np.diff(all_s)
    bouts = segment_bouts(all_s, lk["max_ili_ms"] / 1000.0, lk["min_bout_licks"])
    sizes = np.array([b[2] for b in bouts]) if bouts else np.array([])
    durs = np.array([b[1] - b[0] for b in bouts]) if bouts else np.array([])
    within = [np.median(np.diff(all_s[(all_s >= b[0]) & (all_s <= b[1])])) for b in bouts if b[2] >= 2]
    session = {
        "n_licks": int(all_s.size),
        "session_lick_rate_hz": float(all_s.size / (all_s[-1] - all_s[0])) if all_s.size > 1 else np.nan,
        "ili_median_s": float(np.median(ili)) if ili.size else np.nan,
        "n_bouts": len(bouts),
        "mean_bout_size": float(sizes.mean()) if sizes.size else np.nan,
        "mean_bout_dur_s": float(durs.mean()) if durs.size else np.nan,
        "within_bout_ili_s": float(np.nanmean(within)) if within else np.nan,
    }
    return {"per_position": per_pos, "session": session, "raster": raster, "ili": ili, "all_s": all_s}


def _daq_h5_for(rv, animal: str, date: str) -> Path | None:
    """Locate the session's DAQ ``.h5`` (``DAQ_recorder_output/<date>/<animal>_<date>_*.h5``)."""
    try:
        root = Path(rv.root("daq_recorder_output"))
    except Exception:
        return None
    for base in (root / date, root):
        if base.is_dir():
            hits = sorted(base.glob(f"{animal}_{date}_*.h5"))
            if hits:
                return hits[0]
    return None


def compare_daq_licks(h5_path: Path, params: dict) -> dict | None:
    """Run our DAQ ``lick_detection`` pipeline on ``lick_analog`` and return counts + ILIs for the
    GUI-vs-pipeline comparison. ``None`` if h5py/channel unavailable.

    A ``min_ili_ms`` refractory (behavior.licking) is applied so the pipeline rejects the sub-40 ms
    double-detections that the offset-locked lockout alone leaves behind — an ILI that short is not a
    real lick (mice lick ~7-9 Hz), and it matches the GUI's own ~40 ms debounce for a fair comparison."""
    try:
        import h5py

        from wfield_local.lick_detection import detect_licks
        from wfield_local.plot_lick_aligned_averages import _decode_analog_channel
    except Exception:
        return None
    ld = config.defaults()["lick_detection"]
    min_ili_s = ld.get("min_ili_ms", 40) / 1000.0
    try:
        with h5py.File(h5_path, "r") as f:
            fs = float(f.attrs["sample_rate_hz"])
            sig = _decode_analog_channel(f, "lick_analog")
    except Exception as e:
        print(f"[spout_behavior] DAQ compare skipped ({h5_path.name}): {e}", flush=True)
        return None
    det = detect_licks(sig, fs, thresh_upper=ld["thresh_upper"], thresh_lower=ld["thresh_lower"],
                       lockout_s=tuple(ld["lockout_falling_edge_s"]), min_ili_s=min_ili_s)
    clean = np.asarray(det["lick_onsets"], dtype=float) / fs      # lockout + >=min_ili floor
    raw = np.asarray(det["raw_onsets"], dtype=float) / fs
    return {"n_raw": raw.size, "n_clean": clean.size, "n_removed": raw.size - clean.size,
            "ili_clean": np.diff(clean), "ili_raw": np.diff(raw), "fs": fs, "min_ili_ms": min_ili_s * 1000}


# --------------------------------------------------------------------------- engagement

def flag_engagement(responded, *, window: int, min_rate: float, tail_min_misses: int):
    """Classify each trial engaged vs disengaged from the ``responded`` (lick-in-window) sequence.

    Two disengagement signals, unioned:

    * **terminal sated tail** — a trailing run of >= ``tail_min_misses`` consecutive
      non-responses (matches the task's auto reward-hold-after-misses), and
    * **response collapse** — a trailing rolling response rate (window ``window``) below
      ``min_rate`` (catches mid-session bouts where the animal stops working at every position).

    A rough patch of hard-position misses does NOT trip the collapse gate as long as the animal
    keeps hitting easy positions (rolling rate stays up). Returns ``(engaged_bool, info)``.
    """
    resp = np.asarray(responded, dtype=bool)
    n = resp.size
    if n == 0:
        return np.ones(0, bool), {"n": 0, "n_disengaged": 0, "tail_start": None}
    r = resp.astype(float)
    roll = np.array([r[max(0, i - window + 1):i + 1].mean() for i in range(n)])
    low = roll < min_rate

    tail = np.zeros(n, bool)
    tail_start = None
    if resp.any():
        last = int(np.max(np.flatnonzero(resp)))
        if (n - 1 - last) >= tail_min_misses:
            tail[last + 1:] = True
            tail_start = last + 1
    else:
        tail[:] = True
        tail_start = 0

    engaged = ~(low | tail)
    info = {"n": int(n), "n_disengaged": int((~engaged).sum()),
            "tail_start": tail_start, "n_tail": int(tail.sum())}
    return engaged, info


# --------------------------------------------------------------------------- metrics

def _wilson(hits: int, n: int, z: float = 1.96):
    """Wilson 95% CI half-widths (lo, hi) for a hit rate; (0,0) when n==0."""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def session_metrics(trials: pd.DataFrame, latency: pd.Series | None, params: dict) -> dict:
    """Per-position + session-level metrics, engaged-gated. ``params`` = defaults()['behavior']."""
    eng = params["engagement"]
    scored = trials[~trials["is_free"]] if params.get("free_reward_excluded", True) else trials
    engaged_bool, info = flag_engagement(
        scored["responded"].to_numpy(),
        window=eng["window_trials"], min_rate=eng["min_response_rate"],
        tail_min_misses=eng["tail_min_misses"])
    scored = scored.copy()
    scored["engaged"] = engaged_bool

    rows = []
    for idx in IDX_ORDER:
        p = POS_BY_IDX[idx]
        allp = scored[scored["pos_idx"] == idx]
        engp = allp[allp["engaged"]]
        h_all, n_all = int(allp["hit"].sum()), len(allp)
        h_eng, n_eng = int(engp["hit"].sum()), len(engp)
        lo, hi = _wilson(h_eng, n_eng)
        lat = np.nan
        if latency is not None and n_eng:
            lvals = latency.loc[engp.index].to_numpy(dtype=float)
            if np.isfinite(lvals).any():
                lat = float(np.nanmedian(lvals))
        rows.append({
            "pos_idx": idx, "pos_name": p["name"], "ring": p["ring"], "side": p["side"],
            "row": p["row"], "col": p["col"],
            "trials_all": n_all, "hits_all": h_all,
            "hit_rate_all": (h_all / n_all) if n_all else np.nan,
            "trials_engaged": n_eng, "hits_engaged": h_eng,
            "hit_rate": (h_eng / n_eng) if n_eng else np.nan,
            "ci_lo": lo, "ci_hi": hi, "median_latency_s": lat,
        })
    per_pos = pd.DataFrame(rows)
    n_eng_tot = int(scored["engaged"].sum())
    h_eng_tot = int(scored.loc[scored["engaged"], "hit"].sum())
    return {
        "per_position": per_pos,
        "engaged_mask": scored["engaged"].to_numpy(),
        "scored": scored,
        "n_scored": len(scored),
        "n_engaged": n_eng_tot,
        "n_disengaged": info["n_disengaged"],
        "tail_start": info["tail_start"],
        "hit_rate_engaged": (h_eng_tot / n_eng_tot) if n_eng_tot else np.nan,
        "hit_rate_all": (int(scored["hit"].sum()) / len(scored)) if len(scored) else np.nan,
    }


# --------------------------------------------------------------------------- per-session figure

def _grid_hit_rate(ax, per_pos: pd.DataFrame, min_eng: int):
    """2x3 spatial heatmap of engaged hit rate; rows close/far, cols L/center/R."""
    grid = np.full((2, 3), np.nan)
    col_of = {"L": 0, "C": 1, "R": 2}
    for _, r in per_pos.iterrows():
        if r["trials_engaged"] >= min_eng:
            grid[r["row"], col_of[r["side"]]] = r["hit_rate"]
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks([0, 1, 2], ["L", "center", "R"])
    ax.set_yticks([0, 1], ["close", "far"])
    for _, r in per_pos.iterrows():
        c = col_of[r["side"]]
        txt = ("—" if r["trials_engaged"] < min_eng
               else f"{r['hit_rate']:.2f}\n(n={r['trials_engaged']})")
        val = grid[r["row"], c]
        color = "w" if (np.isnan(val) or val < 0.6) else "k"
        ax.text(c, r["row"], txt, ha="center", va="center", fontsize=9, color=color)
    ax.set_title("hit rate (engaged)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _engagement_timeline(ax, scored: pd.DataFrame, eng: dict):
    """Trial-index raster (hit green / miss red) + trailing rolling response rate; disengaged shaded."""
    x = np.arange(len(scored))
    resp = scored["responded"].to_numpy().astype(float)
    w = eng["window_trials"]
    roll = np.array([resp[max(0, i - w + 1):i + 1].mean() for i in range(len(scored))])
    engaged = scored["engaged"].to_numpy()
    hit = scored["hit"].to_numpy().astype(bool)
    ax.scatter(x[hit], np.full(hit.sum(), 1.06), s=8, c="tab:green", marker="|", label="hit")
    ax.scatter(x[~hit], np.full((~hit).sum(), 1.02), s=8, c="tab:red", marker="|", label="miss")
    ax.plot(x, roll, color="k", lw=1.2, label=f"rolling resp. ({w})")
    ax.axhline(eng["min_response_rate"], color="grey", ls=":", lw=1)
    # shade contiguous disengaged spans
    dis = ~engaged
    i = 0
    lab = True
    while i < len(dis):
        if dis[i]:
            j = i
            while j < len(dis) and dis[j]:
                j += 1
            ax.axvspan(i - 0.5, j - 0.5, color="grey", alpha=0.18,
                       label="disengaged" if lab else None)
            lab = False
            i = j
        else:
            i += 1
    ax.set_ylim(-0.02, 1.12)
    ax.set_xlabel("trial index")
    ax.set_ylabel("response rate")
    ax.set_title("engagement over session")
    ax.legend(loc="lower left", fontsize=7, ncol=2)


def _hit_rate_bars(ax, per_pos: pd.DataFrame, min_eng: int):
    """Per-position engaged hit rate (bars + Wilson CI) with the raw all-trial rate overlaid."""
    x = np.arange(len(per_pos))
    hr = per_pos["hit_rate"].to_numpy()
    # Wilson CI is not centred on the point estimate (esp. at p=0/1), so clamp the
    # bar-relative error bars at 0 to avoid negative yerr.
    lo = np.clip(hr - per_pos["ci_lo"].to_numpy(), 0, None)
    hi = np.clip(per_pos["ci_hi"].to_numpy() - hr, 0, None)
    colors = ["tab:blue" if r == "close" else "tab:purple" for r in per_pos["ring"]]
    faded = per_pos["trials_engaged"].to_numpy() < min_eng
    bars = ax.bar(x, np.nan_to_num(hr), yerr=[np.nan_to_num(lo), np.nan_to_num(hi)],
                  color=colors, alpha=0.85, capsize=3)
    for b, f in zip(bars, faded):
        if f:
            b.set_alpha(0.25)
    ax.plot(x, per_pos["hit_rate_all"].to_numpy(), "kD", ms=5, label="raw (all trials)")
    ax.set_xticks(x, per_pos["pos_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hit rate")
    ax.set_title("per-position accuracy (blue=close, purple=far)")
    ax.legend(loc="lower left", fontsize=7)


def _latency_by_position(ax, scored: pd.DataFrame, latency: pd.Series):
    """First-lick latency distribution per position (engaged trials), close->far."""
    data, labels = [], []
    for idx in IDX_ORDER:
        m = (scored["pos_idx"] == idx) & scored["engaged"]
        vals = latency.loc[scored.index[m]].dropna().to_numpy()
        data.append(vals)
        labels.append(POS_BY_IDX[idx]["name"])
    if not any(len(d) for d in data):
        ax.text(0.5, 0.5, "no latency (events.csv absent)", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.boxplot([d if len(d) else [np.nan] for d in data], showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("first-lick latency (s)")
    ax.set_title("response latency (engaged)")


def _ring_color(pos_idx: int) -> str:
    return "tab:blue" if POS_BY_IDX[pos_idx]["ring"] == "close" else "tab:purple"


def _peri_cue_raster(ax, raster, resp_window_s, window_s):
    """One row per trial; a dot per lick at its time from cue. Cue at 0, response window shaded."""
    for y, (delays, pos, _order) in enumerate(raster):
        if delays.size:
            ax.plot(delays, np.full(delays.size, y), ".", ms=1.6, color=_ring_color(pos))
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(0, resp_window_s, color="gold", alpha=0.15)
    ax.set_xlim(*window_s)
    ax.set_ylim(-1, len(raster))
    ax.set_xlabel("time from cue (s)")
    ax.set_ylabel("trial")
    ax.set_title("peri-cue lick raster (blue=close, purple=far)")


def _peri_cue_psth(ax, raster, bin_ms, window_s):
    """Lick PSTH (licks/trial/s) aligned to cue, split close vs far."""
    edges = np.arange(window_s[0], window_s[1] + 1e-9, bin_ms / 1000.0)
    for ring, col in (("close", "tab:blue"), ("far", "tab:purple")):
        delays = [d for d, pos, _ in raster if POS_BY_IDX[pos]["ring"] == ring for d in d]
        n_tr = sum(1 for _, pos, _ in raster if POS_BY_IDX[pos]["ring"] == ring)
        if n_tr and delays:
            h, _ = np.histogram(delays, bins=edges)
            ax.step(edges[:-1], h / n_tr / (bin_ms / 1000.0), where="post", color=col, label=ring)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlim(*window_s)
    ax.set_xlabel("time from cue (s)")
    ax.set_ylabel("lick rate (Hz/trial)")
    ax.set_title("peri-cue lick PSTH")
    ax.legend(fontsize=8)


def _ili_hist(ax, ili, daq_cmp, max_ili_ms, min_ili_ms):
    """Log-x ILI histogram (GUI), with DAQ-pipeline overlay, bout-split + physiological-floor lines."""
    bins = np.logspace(np.log10(0.01), np.log10(10), 60)
    if ili.size:
        ax.hist(ili, bins=bins, density=True, color="tab:gray", alpha=0.6, label="GUI licks")
    if daq_cmp is not None and daq_cmp["ili_clean"].size:
        ax.hist(daq_cmp["ili_clean"], bins=bins, density=True, histtype="step",
                color="tab:red", lw=1.5, label="DAQ pipeline")
    ax.axvline(min_ili_ms / 1000.0, color="k", ls=":", lw=1, label=f"{min_ili_ms:.0f} ms floor")
    ax.axvline(max_ili_ms / 1000.0, color="green", ls="--", lw=1, label=f"{max_ili_ms:.0f} ms bout split")
    ax.set_xscale("log")
    ax.set_xlabel("inter-lick interval (s)")
    ax.set_ylabel("density")
    ax.set_title("ILI distribution")
    ax.legend(fontsize=7)


def _daq_compare_panel(ax, n_gui, daq_cmp):
    """Bar of lick counts: GUI vs DAQ raw crossings vs DAQ after lockout+floor."""
    if daq_cmp is None:
        ax.text(0.5, 0.5, "no DAQ .h5 for comparison", ha="center", va="center")
        ax.set_axis_off()
        return
    labels = ["GUI", "DAQ raw", f"DAQ clean\n(+{daq_cmp['min_ili_ms']:.0f}ms)"]
    vals = [n_gui, daq_cmp["n_raw"], daq_cmp["n_clean"]]
    ax.bar(labels, vals, color=["tab:gray", "salmon", "tab:red"])
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("lick count")
    ax.set_title(f"GUI vs DAQ pipeline ({daq_cmp['n_removed']} sub-floor removed)")


def _micro_pos_bars(ax, micro_pos, col, title, ylabel, color):
    x = np.arange(len(micro_pos))
    colors = [_ring_color(i) for i in micro_pos["pos_idx"]] if color == "ring" else color
    ax.bar(x, micro_pos[col].to_numpy(), color=colors)
    ax.set_xticks(x, micro_pos["pos_name"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_licking(session_dir: Path, sid: str, out_dir: Path, params: dict, trials: pd.DataFrame,
                 gui: dict, rv=None):
    """Write the lick-microstructure figure + lick-metrics CSV. Returns (png, csv) or (None, None)."""
    micro = lick_microstructure(session_dir, trials, params, gui=gui)
    if micro is None:
        return None, None
    lk = params["licking"]
    animal, date = _animal_date(sid)
    daq_cmp = None
    if lk.get("compare_daq", True) and rv is not None:
        h5 = _daq_h5_for(rv, animal, date)
        if h5 is not None:
            daq_cmp = compare_daq_licks(h5, params)

    out_sess = out_dir / "sessions" / animal / date
    png = out_sess / f"{sid}_licking.png"
    csv = out_sess / f"{sid}_lick_metrics.csv"
    out_sess.mkdir(parents=True, exist_ok=True)
    micro["per_position"].to_csv(csv, index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    _peri_cue_raster(axes[0, 0], micro["raster"], lk["response_window_s"], lk["peri_cue_window_s"])
    _peri_cue_psth(axes[0, 1], micro["raster"], lk["psth_bin_ms"], lk["peri_cue_window_s"])
    _ili_hist(axes[0, 2], micro["ili"], daq_cmp, lk["max_ili_ms"],
              config.defaults()["lick_detection"]["min_ili_ms"])
    _micro_pos_bars(axes[1, 0], micro["per_position"], "lick_rate_hz",
                    "within-trial lick rate", "Hz", "ring")
    _micro_pos_bars(axes[1, 1], micro["per_position"], "licks_per_trial",
                    "licks / trial (post-cue)", "licks", "ring")
    _daq_compare_panel(axes[1, 2], micro["session"]["n_licks"], daq_cmp)
    s = micro["session"]
    fig.suptitle(f"{sid} — licking microstructure   n_licks={s['n_licks']}  "
                 f"rate={s['session_lick_rate_hz']:.2f}Hz  ILI_med={s['ili_median_s']*1000:.0f}ms  "
                 f"bouts={s['n_bouts']} (mean {s['mean_bout_size']:.1f} licks/"
                 f"{s['mean_bout_dur_s']:.2f}s)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[spout_behavior] wrote {png.name}", flush=True)
    return png, csv


def plot_session(session_dir: Path, out_dir: Path, params: dict, dry: bool = False, rv=None):
    """Write the per-session behavior figure + per-position CSV, plus the lick-microstructure figure.

    Returns (behavior_png, position_csv). Near-empty (aborted) sessions are skipped."""
    trials = load_trials(session_dir)
    min_trials = params.get("min_session_trials", 20)
    if len(trials) < min_trials:
        print(f"[spout_behavior] skip {session_dir.name}: {len(trials)} scored trials "
              f"(< {min_trials}; aborted run?)", flush=True)
        return None, None
    gui = load_gui_licks(session_dir)
    latency = first_lick_latency_s(session_dir, trials, params.get("latency_max_s", 5.0), gui=gui)
    m = session_metrics(trials, latency, params)
    sid = session_dir.name
    animal, date = _animal_date(sid)
    sess_dir = out_dir / "sessions" / animal / date       # structured by animal/date
    png = sess_dir / f"{sid}_behavior.png"
    csv = sess_dir / f"{sid}_position_metrics.csv"
    if dry:
        print(f"[spout_behavior] {sid}: {m['n_scored']} scored, {m['n_engaged']} engaged, "
              f"hit_rate(engaged)={m['hit_rate_engaged']:.3f} -> {png.relative_to(out_dir)}", flush=True)
        return png, csv

    from wfield_local import writeguard
    writeguard.assert_writable(out_dir)
    sess_dir.mkdir(parents=True, exist_ok=True)
    m["per_position"].to_csv(csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _grid_hit_rate(axes[0, 0], m["per_position"], params["engagement"]["min_engaged_trials"])
    _engagement_timeline(axes[0, 1], m["scored"], params["engagement"])
    _hit_rate_bars(axes[1, 0], m["per_position"], params["engagement"]["min_engaged_trials"])
    _latency_by_position(axes[1, 1], m["scored"], latency)
    fig.suptitle(f"{sid}   scored={m['n_scored']}  engaged={m['n_engaged']}  "
                 f"disengaged-excluded={m['n_disengaged']}  "
                 f"hit-rate(engaged)={m['hit_rate_engaged']:.3f} (raw {m['hit_rate_all']:.3f})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[spout_behavior] wrote {png.name}  ({m['n_disengaged']} disengaged trials excluded)",
          flush=True)
    if gui is not None:
        plot_licking(session_dir, sid, out_dir, params, trials, gui, rv=rv)
    return png, csv


# --------------------------------------------------------------------------- discovery + cohort

def _animal_of(session_name: str) -> str | None:
    mo = ANIMAL_RE.match(session_name)
    return mo.group(0) if mo else None


def _animal_date(session_name: str) -> tuple[str, str]:
    """('PS92', '20260806') from 'PS92_20260806_124753'; falls back to ('unknown', 'unknown')."""
    mo = re.match(r"(PS\d+)_(\d{8})", session_name)
    return (mo.group(1), mo.group(2)) if mo else (_animal_of(session_name) or "unknown", "unknown")


def discover_sessions(rv: PathResolver, date: str | None, animals=None) -> list[Path]:
    """Behavior-log session dirs, optionally filtered by date (YYYYMMDD) and animal set."""
    root = Path(rv.root("behavior_logs"))
    if not root.is_dir():
        return []
    aset = set(animals) if animals else None
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not (p / "trials.csv").exists():
            continue
        if date and date not in p.name:
            continue
        if aset and _animal_of(p.name) not in aset:
            continue
        out.append(p)
    return out


# per-position metric families carried in each session record (for the per-animal figures).
# key prefix -> (source table, column, human label). hit rate is stored UNPREFIXED (by pos name) too,
# for backward compatibility with the cross-animal cohort figure.
POS_METRICS = [
    ("hit", "acc", "hit_rate", "hit rate (engaged)"),
    ("lat", "acc", "median_latency_s", "first-lick latency (s)"),
    ("lpt", "lick", "licks_per_trial", "licks / trial"),
    ("rate", "lick", "lick_rate_hz", "within-trial lick rate (Hz)"),
    ("ant", "lick", "anticipatory_licks", "anticipatory licks / trial"),
]


def session_row(session_dir: Path, params: dict) -> dict | None:
    """One pooled record per session (for cohort + per-animal figures): session-level scalars plus
    every per-position metric (hit rate, latency, licks/trial, lick rate, anticipatory), keyed
    ``<prefix>__<pos_name>``. Hit rate is ALSO stored unprefixed by pos name (cohort back-compat)."""
    try:
        trials = load_trials(session_dir)
    except Exception as e:
        print(f"[spout_behavior] skip {session_dir.name}: {e}", flush=True)
        return None
    if len(trials) < params.get("min_session_trials", 20):
        return None
    gui = load_gui_licks(session_dir)
    latency = first_lick_latency_s(session_dir, trials, params.get("latency_max_s", 5.0), gui=gui)
    m = session_metrics(trials, latency, params)
    micro = lick_microstructure(session_dir, trials, params, gui=gui) if gui is not None else None
    animal, date = _animal_date(session_dir.name)
    acc = m["per_position"].set_index("pos_idx")
    lick = micro["per_position"].set_index("pos_idx") if micro is not None else None
    src = {"acc": acc, "lick": lick}
    rec = {"session": session_dir.name, "animal": animal, "date": date,
           "n_engaged": m["n_engaged"], "n_disengaged": m["n_disengaged"],
           "hit_rate": m["hit_rate_engaged"]}
    if micro is not None:
        rec.update({"session_lick_rate_hz": micro["session"]["session_lick_rate_hz"],
                    "ili_median_s": micro["session"]["ili_median_s"],
                    "n_bouts": micro["session"]["n_bouts"]})
    for idx in IDX_ORDER:
        nm = POS_BY_IDX[idx]["name"]
        rec[nm] = acc.loc[idx, "hit_rate"] if idx in acc.index else np.nan   # back-compat (hit rate)
        for prefix, table, col, _label in POS_METRICS:
            t = src[table]
            rec[f"{prefix}__{nm}"] = (t.loc[idx, col] if (t is not None and idx in t.index) else np.nan)
    rec["close"] = np.nanmean([rec[p["name"]] for p in POSITIONS if p["ring"] == "close"])
    rec["far"] = np.nanmean([rec[p["name"]] for p in POSITIONS if p["ring"] == "far"])
    return rec


def cohort_summary(rv: PathResolver, dates, animals, out_dir: Path, dry: bool = False):
    """Pool sessions across ``dates`` into per-animal per-position accuracy + learning curves."""
    sessions = []
    for p in discover_sessions(rv, None, animals):
        mo = re.match(r"PS\d+_(\d{8})", p.name)
        mmdd = mo.group(1)[4:] if mo else None
        if dates is None or (mmdd in dates):
            sessions.append(p)
    params = config.defaults()["behavior"]
    rows = [r for r in (session_row(s, params) for s in sessions) if r]
    if not rows:
        print("[spout_behavior] cohort: no sessions matched", flush=True)
        return None
    df = pd.DataFrame(rows).sort_values(["animal", "date"]).reset_index(drop=True)
    csv = out_dir / "cohort" / "cohort_session_metrics.csv"
    png = out_dir / "cohort" / "cohort_behavior.png"
    if dry:
        print(f"[spout_behavior] cohort: {len(df)} sessions, animals={sorted(df['animal'].unique())} "
              f"-> {png.name}", flush=True)
        return df

    from wfield_local import writeguard
    writeguard.assert_writable(out_dir)
    (out_dir / "cohort").mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    colors = config.animal_color()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    # (A) per-animal per-position hit rate (pooled over sessions = mean of session rates)
    ax = axes[0]
    names = [POS_BY_IDX[i]["name"] for i in IDX_ORDER]
    anims = sorted(df["animal"].unique())
    x = np.arange(len(names))
    wd = 0.8 / max(1, len(anims))
    for k, a in enumerate(anims):
        sub = df[df["animal"] == a]
        means = [np.nanmean(sub[n]) for n in names]
        ax.bar(x + k * wd, means, wd, label=a, color=colors.get(a, None), alpha=0.85)
    ax.set_xticks(x + wd * (len(anims) - 1) / 2, names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hit rate (engaged, session-mean)")
    ax.set_title("per-position accuracy by animal")
    ax.legend(fontsize=8)
    # (B) learning curve: engaged hit rate vs date
    ax = axes[1]
    for a in anims:
        sub = df[df["animal"] == a].sort_values("date")
        ax.plot(sub["date"], sub["hit_rate"], "-o", label=a, color=colors.get(a, None))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hit rate (engaged)")
    ax.set_xlabel("date")
    ax.set_title("session hit rate over time")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=8)
    # (C) distance effect: close vs far per animal
    ax = axes[2]
    for a in anims:
        sub = df[df["animal"] == a]
        ax.plot([0, 1], [np.nanmean(sub["close"]), np.nanmean(sub["far"])],
                "-o", label=a, color=colors.get(a, None))
    ax.set_xticks([0, 1], ["close (2mm)", "far (4mm)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hit rate (engaged)")
    ax.set_title("distance effect")
    ax.legend(fontsize=8)
    fig.suptitle(f"spout behavior cohort — {len(df)} sessions, {len(anims)} animals", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[spout_behavior] wrote {png.name} ({len(df)} sessions)", flush=True)

    for a in anims:                                   # per-animal across-session summaries
        plot_animal_summary(a, df[df["animal"] == a], out_dir)
    return df


def plot_animal_summary(animal: str, adf: pd.DataFrame, out_dir: Path):
    """One across-session figure per animal: each per-position metric (hit rate, latency, licks/trial,
    lick rate, anticipatory) over that animal's sessions + a session-level panel. Same metrics as the
    per-session figures, tracked across days."""
    adf = adf.sort_values("date").reset_index(drop=True)
    if adf.empty:
        return None
    png = out_dir / "cohort" / "by_animal" / f"{animal}_across_sessions.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    dates = adf["date"].tolist()
    x = np.arange(len(dates))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    panels = [(p, lab) for p, _t, _c, lab in POS_METRICS]      # 5 per-position metric families
    for ax, (prefix, label) in zip(axes.flat[:5], panels):
        for idx in IDX_ORDER:                                 # color=ring, marker+linestyle=side
            p = POS_BY_IDX[idx]
            key = f"{prefix}__{p['name']}"
            if key in adf:
                ax.plot(x, adf[key].to_numpy(), color=_ring_color(idx), marker=SIDE_MARKER[p["side"]],
                        ls=SIDE_LS[p["side"]], ms=5, alpha=0.85, label=p["name"])
        ax.set_xticks(x, dates, rotation=45, ha="right", fontsize=7)
        ax.set_title(label)
        if prefix == "hit":
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, ncol=2)
    # session-level panel: hit rate, close/far, engagement fraction over sessions
    ax = axes.flat[5]
    ax.plot(x, adf["hit_rate"], "-o", color="k", label="hit rate (engaged)")
    ax.plot(x, adf["close"], "-o", color="tab:blue", alpha=0.7, label="close")
    ax.plot(x, adf["far"], "-o", color="tab:purple", alpha=0.7, label="far")
    if "n_disengaged" in adf and "n_engaged" in adf:
        frac = adf["n_engaged"] / (adf["n_engaged"] + adf["n_disengaged"]).replace(0, np.nan)
        ax.plot(x, frac, "--s", color="grey", alpha=0.7, label="engaged frac")
    ax.set_xticks(x, dates, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("session-level")
    ax.legend(fontsize=7)
    fig.suptitle(f"{animal} — behavior across {len(adf)} sessions "
                 f"(blue=close, purple=far positions)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[spout_behavior] wrote {png.relative_to(out_dir)}", flush=True)
    return png


# --------------------------------------------------------------------------- orchestration

def run(date, rv, animals=None, cohort=False, from_spec=None, dry=False) -> int:
    """Per-session figures for ``date`` (and/or a cohort summary). Returns 0."""
    params = config.defaults()["behavior"]
    out_dir = Path(rv.root("behavior_out"))
    if date:
        sessions = discover_sessions(rv, date, animals)
        print(f"[spout_behavior] {date}: {len(sessions)} session(s)", flush=True)
        for s in sessions:
            try:
                plot_session(s, out_dir, params, dry=dry, rv=rv)
            except Exception as e:
                print(f"[spout_behavior] FAILED {s.name}: {e}", flush=True)
    if cohort:
        dates = None
        if from_spec:
            dates = (config.cross_session_dates() if from_spec == "curated"
                     else config.expand_dates(from_spec))
        cohort_summary(rv, dates, animals, out_dir, dry=dry)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", nargs="?", metavar="YYYYMMDD", help="session date; omit with --cohort")
    ap.add_argument("--only", nargs="+", metavar="ANIMAL", help="restrict to these animals, or 'all'")
    ap.add_argument("--cohort", action="store_true", help="also build the pooled cohort figures")
    ap.add_argument("--from", dest="from_spec", metavar="SPEC",
                    help="cohort date set: 'curated' (default policy), a range/list, or 'all'")
    ap.add_argument("--dry-run", action="store_true", help="discover + report; write nothing")
    ap.add_argument("--machine", default=None, help="override machine (default: auto-detect)")
    args = ap.parse_args(argv)
    if not args.date and not args.cohort:
        ap.error("give a YYYYMMDD date, --cohort, or both")
    return run(args.date, PathResolver(machine=args.machine),
               animals=config.normalize_animals(args.only),
               cohort=args.cohort, from_spec=args.from_spec or ("curated" if args.cohort else None),
               dry=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

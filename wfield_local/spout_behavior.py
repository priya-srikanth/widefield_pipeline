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


def first_lick_latency_s(session_dir: Path, trials: pd.DataFrame, max_s: float) -> pd.Series:
    """Per-trial first-lick latency (s) from ``events.csv``: first ``lick_on`` at/after the trial's
    ``cue``. Returns NaN where no ``events.csv``, no cue, or latency exceeds ``max_s``."""
    idx = pd.Series(np.nan, index=trials.index)
    ev_path = session_dir / "events.csv"
    if not ev_path.exists():
        return idx
    try:
        ev = pd.read_csv(ev_path, usecols=lambda c: c in ("device_t_ms", "event_name", "trial_id"))
    except Exception:
        return idx
    ev["device_t_ms"] = pd.to_numeric(ev["device_t_ms"], errors="coerce")
    ev["trial_id"] = pd.to_numeric(ev["trial_id"], errors="coerce")
    cue_t = (ev[ev["event_name"] == "cue"].groupby("trial_id")["device_t_ms"].min())
    licks = ev[ev["event_name"] == "lick_on"]
    lick_by_trial = {tid: np.sort(g["device_t_ms"].to_numpy()) for tid, g in licks.groupby("trial_id")}
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
            lat = (after[0] - c) / 1000.0
            if 0 <= lat <= max_s:
                idx.loc[i] = lat
    return idx


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


def plot_session(session_dir: Path, out_dir: Path, params: dict, dry: bool = False):
    """Write one per-session behavior figure + a per-position metrics CSV. Returns (png, csv) paths."""
    trials = load_trials(session_dir)
    latency = first_lick_latency_s(session_dir, trials, params.get("latency_max_s", 5.0))
    m = session_metrics(trials, latency, params)
    sid = session_dir.name
    png = out_dir / "sessions" / f"{sid}_behavior.png"
    csv = out_dir / "sessions" / f"{sid}_position_metrics.csv"
    if dry:
        print(f"[spout_behavior] {sid}: {m['n_scored']} scored, {m['n_engaged']} engaged, "
              f"hit_rate(engaged)={m['hit_rate_engaged']:.3f} -> {png.name}", flush=True)
        return png, csv

    from wfield_local import writeguard
    writeguard.assert_writable(out_dir)
    (out_dir / "sessions").mkdir(parents=True, exist_ok=True)
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
    return png, csv


# --------------------------------------------------------------------------- discovery + cohort

def _animal_of(session_name: str) -> str | None:
    mo = ANIMAL_RE.match(session_name)
    return mo.group(0) if mo else None


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


def session_row(session_dir: Path, params: dict) -> dict | None:
    """One pooled record per session (for cohort figures): animal, date, engaged hit rates."""
    try:
        trials = load_trials(session_dir)
    except Exception as e:
        print(f"[spout_behavior] skip {session_dir.name}: {e}", flush=True)
        return None
    if trials.empty:
        return None
    m = session_metrics(trials, None, params)
    mo = re.match(r"(PS\d+)_(\d{8})", session_dir.name)
    per = m["per_position"].set_index("pos_idx")
    rec = {"session": session_dir.name,
           "animal": mo.group(1) if mo else _animal_of(session_dir.name),
           "date": mo.group(2) if mo else "",
           "n_engaged": m["n_engaged"], "hit_rate": m["hit_rate_engaged"]}
    for idx in IDX_ORDER:
        rec[POS_BY_IDX[idx]["name"]] = per.loc[idx, "hit_rate"] if idx in per.index else np.nan
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
    return df


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
                plot_session(s, out_dir, params, dry=dry)
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

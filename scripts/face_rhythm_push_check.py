"""Does the face do something at treadmill PUSHES that the pre-stroke FaceRhythm basis cannot describe? (Priya 10-24:
"do we need a pre + post master fit, eg to capture the pushes we found in the encoding window?" -- test first.)

Pushes (ENCODING_FINDINGS 4.4, `scripts/push_behaviour.py`): backward treadmill speed beyond PUSH_FLOOR (0.3 mm/s) for
>= 2 samples at 31.25 Hz -- a forepaw pushing / postural change; 2-5x more frequent after the stroke, above pre
chronically. A frozen PRE-stroke basis describes a post-stroke push only if the face movement at the push is already in
the pre repertoire. So, per session, around every push ONSET:
  err   the frozen basis's relative RECONSTRUCTION ERROR per 5 Hz session bin (`proj/` ``session_err``; PRE sessions on
        their held-out leave-one-day-out basis, ``proj_loso/``, so pre is not flattered by being in the fit)
  D     the face's movement energy per bin (``session_D``): is there face movement at the push at all?
against RANDOM times in the same session and the same context (in-trial: cue -1 .. +4 s; between trials: the rest),
>= 2 s from any push. Readout per session: (post 0 .. +1 s) - (pre -1.5 .. -0.5 s), push minus random.
  push error rises chronically but not pre  -> the face moves in a way the pre basis misses at pushes: a pre + post fit
                                               (or a post-only fit matched to the master, `face_rhythm_match.py`) is
                                               worth doing, its unmatched factors are the candidates;
  no rise (or the same pre and post)        -> the pre basis covers the face at pushes; pushes stay a TREADMILL regressor.

    python scripts/face_rhythm_push_check.py [--set master_pre_v1] [--jobs 4] [--plot-only]
COMPUTE once (~1 h: every session's treadmill speed is reloaded), then the per-session table WITH its traces is saved
(data/push_check_traces.npz); ``--plot-only`` redraws the figure and tables from it in seconds (Priya 10-24: re-render
figures without redoing calculations).
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/push_check/``: push_check_sessions.csv (per session x context), the per
animal x epoch table (printed; rule 8 first), push_check.png (traces by epoch, per animal).
"""
from __future__ import annotations

import argparse
import zlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FS = 31.25                 # = push_behaviour.FS
PUSH_FLOOR = 0.3           # = frozen_encoder.PUSH_FLOOR
LAGS = np.arange(-2.0, 3.0001, 0.2)
BASE, POST = (-1.5, -0.5), (0.0, 1.0)
TRIAL = (-1.0, 4.0)        # around the cue: "in-trial" pushes
EPOCHS = ("pre", "acute", "subacute", "chronic")


def push_onsets(t: np.ndarray, v: np.ndarray, floor: float = PUSH_FLOOR, min_len: int = 2) -> np.ndarray:
    """Onset times of runs of >= ``min_len`` samples with backward speed beyond ``floor`` (signed speed, negative =
    backward), on a 31.25 Hz grid over the recording."""
    g = np.arange(t[0], t[-1], 1 / FS)
    m = -np.interp(g, t, v) > floor
    d = np.diff(np.r_[0, m.astype(int), 0])
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return g[on[(off - on) >= min_len]]


def _traces(ts: np.ndarray, y: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """(n_events, lags) of ``y`` at the nearest session bin to event + lag (NaN outside / at lossy bins)."""
    if not len(ev):
        return np.empty((0, len(LAGS)))
    q = ev[:, None] + LAGS[None]
    i = np.clip(np.searchsorted(ts, q), 1, len(ts) - 1)
    i = np.where(np.abs(ts[i - 1] - q) < np.abs(ts[i] - q), i - 1, i)
    out = y[i].astype(float)
    out[np.abs(ts[i] - q) > 0.25] = np.nan
    return out


def _delta(tr: np.ndarray) -> float:
    if not len(tr):
        return np.nan
    b = np.nanmean(tr[:, (LAGS >= BASE[0]) & (LAGS <= BASE[1])], 1)
    p = np.nanmean(tr[:, (LAGS >= POST[0]) & (LAGS <= POST[1])], 1)
    return float(np.nanmean(p - b))


def job(item: tuple) -> list:
    """One proj task -> rows per context (in-trial / between) with push and random-time traces + deltas."""
    fdir, key = item
    from wfield_local import combined_encoding as ce
    from wfield_local.paths import PathResolver
    f_lo = Path(fdir) / "proj_loso" / f"{key}.npz"
    with np.load(Path(fdir) / "proj" / f"{key}.npz", allow_pickle=True) as z:
        animal, date, epoch = str(z["animal"]), str(z["date"]), str(z["epoch"])
        ts, D, cues = z["session_t_daq_s"], z["session_D"], z["cue_daq_s"]
        err = z["session_err"]
    basis = "full"
    if epoch == "pre" and f_lo.exists():
        with np.load(f_lo, allow_pickle=True) as z:
            err, basis = z["session_err"], "loso"
    B = ce.load_behaviour(animal, date, PathResolver())
    tr, vr = (np.asarray(x, float) for x in B.analog["running"])
    ok = np.isfinite(ts) & np.isfinite(err)
    lo, hi = ts[ok].min() + 3, ts[ok].max() - 3
    ev = push_onsets(tr, vr)
    ev = ev[(ev > lo) & (ev < hi)]
    cues = cues[np.isfinite(cues)]

    def in_trial(x):
        d = x[:, None] - cues[None]
        return ((d >= TRIAL[0]) & (d <= TRIAL[1])).any(1) if len(cues) else np.zeros(len(x), bool)
    rng = np.random.default_rng(zlib.crc32(key.encode()))      # not hash(): PYTHONHASHSEED (rule 9)
    cand = ts[ok & (ts > lo) & (ts < hi)]
    if len(ev):
        cand = cand[np.min(np.abs(cand[:, None] - ev[None]), 1) >= 2.0]
    rows = []
    for ctx, sel_e, sel_c in (("trial", in_trial(ev), in_trial(cand)), ("between", ~in_trial(ev), ~in_trial(cand))):
        e, c = ev[sel_e], cand[sel_c]
        c = rng.choice(c, min(len(c), max(3 * len(e), 50)), replace=False) if len(c) else c
        te, tc = _traces(ts, err, e), _traces(ts, err, c)
        de, dc = _traces(ts, D, e), _traces(ts, D, c)
        rows.append({"key": key, "animal": animal, "date": date, "epoch": epoch, "basis": basis, "context": ctx,
                     "n_push": len(e), "n_random": len(c),
                     "push_per_min": len(e) / max((hi - lo) / 60.0, 1e-9),
                     "d_err_push": _delta(te), "d_err_random": _delta(tc),
                     "d_D_push": _delta(de), "d_D_random": _delta(dc),
                     "err_level_push": float(np.nanmean(te[:, (LAGS >= POST[0]) & (LAGS <= POST[1])])) if len(e) else np.nan,
                     "err_level_random": float(np.nanmean(tc[:, (LAGS >= POST[0]) & (LAGS <= POST[1])])) if len(c) else np.nan,
                     "tr_err_push": np.nanmean(te, 0) if len(e) else np.full(len(LAGS), np.nan),
                     "tr_err_random": np.nanmean(tc, 0) if len(c) else np.full(len(LAGS), np.nan),
                     "tr_D_push": np.nanmean(de, 0) if len(e) else np.full(len(LAGS), np.nan)})
    return rows


def save_cache(S: pd.DataFrame, f: Path) -> None:
    """The per-session table with its traces: scalar columns as arrays, ``tr_*`` stacked (rows, lags)."""
    tr = [c for c in S.columns if c.startswith("tr_")]
    np.savez(f, lags=LAGS, columns=np.array([c for c in S.columns if c not in tr]),
             **{f"col_{c}": S[c].to_numpy() for c in S.columns if c not in tr},
             **{c: np.vstack(S[c].to_numpy()) for c in tr})


def load_cache(f: Path) -> pd.DataFrame:
    z = np.load(f, allow_pickle=True)
    S = pd.DataFrame({c: z[f"col_{c}"] for c in z["columns"].astype(str)})
    for c in (k for k in z.files if k.startswith("tr_")):
        S[c] = list(z[c])
    return S


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true", help="redraw from data/push_check_traces.npz, no recompute")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "push_check"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = figure_layout.sidecar_for(out, "push_check_traces", ".npz")
    fail = []
    if a.plot_only:
        S = load_cache(cache)
    else:
        keys = sorted(f.stem for f in (fdir / "proj").glob("*.npz") if not f.name.endswith(".part.npz"))
        got, fail = ak.fan_sessions([(str(fdir), k) for k in keys], job, jobs=a.jobs, key=lambda it: it[1],
                                    label="push")
        S = pd.DataFrame([r for _, rs in got for r in rs])
        save_cache(S, cache)
    tr_cols = [c for c in S.columns if c.startswith("tr_")]
    S.drop(columns=tr_cols).to_csv(figure_layout.sidecar_for(out, "push_check_sessions", ".csv"), index=False,
                                   float_format="%.5g")
    S["d_err_excess"] = S.d_err_push - S.d_err_random
    S["d_D_excess"] = S.d_D_push - S.d_D_random
    S["epoch"] = pd.Categorical(S.epoch, EPOCHS, ordered=True)
    cols = ["n_push", "push_per_min", "d_err_push", "d_err_random", "d_err_excess", "err_level_push",
            "err_level_random", "d_D_excess"]
    A = S[S.n_push >= 5].groupby(["context", "animal", "epoch"], observed=True)[cols].mean().reset_index()
    A["n_sessions"] = S[S.n_push >= 5].groupby(["context", "animal", "epoch"], observed=True).size().to_numpy()
    A.to_csv(figure_layout.sidecar_for(out, "push_check_epochs", ".csv"), index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    print("PER ANIMAL x epoch (sessions with >= 5 pushes; rule 8 first). d_err = error (0..1 s) - (-1.5..-0.5 s);\n"
          "excess = push - random times of the same context. Pre on its held-out basis.")
    print(A.round(4).to_string(index=False))
    if fail:
        print("failed:", [f[0] for f in fail])

    animals = sorted(S.animal.unique())
    from wfield_local.epoch_figures import EPOCH_GREY
    col = dict(EPOCH_GREY)           # the repo's epoch grey ramp (pre light -> chronic dark); animals own the colours
    with frf.style():
        fig, axs = frf.subplots(3, 2 * len(animals), figsize=(4 * 2 * len(animals), 9), squeeze=False, sharex=True)
        for j, an in enumerate(animals):
            for c_i, ctx in enumerate(("trial", "between")):
                jj = 2 * j + c_i
                for ep in EPOCHS:
                    g = S[(S.animal == an) & (S.context == ctx) & (S.epoch == ep) & (S.n_push >= 5)]
                    if g.empty:
                        continue
                    for r, (c, ttl) in enumerate((("tr_err_push", "recon error at push"),
                                                  ("tr_err_random", "recon error, random times"),
                                                  ("tr_D_push", "face movement energy D at push"))):
                        y = np.nanmean(np.vstack(g[c].to_numpy()), 0)
                        axs[r, jj].plot(LAGS, y, color=col[ep], label=f"{ep} ({len(g)})")
                        axs[r, jj].set_title(f"{an} {ctx}: {ttl}", fontsize=9)
                for r in range(3):
                    axs[r, jj].axvline(0, color="k", lw=0.5)
        for ax in axs[-1]:
            ax.set_xlabel("s from push onset")
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: frozen-basis reconstruction error and face movement around treadmill pushes")
        frf.save(fig, out / "push_check.png", f"""
Push = onset of >= 2 samples (31.25 Hz) of backward treadmill speed beyond {PUSH_FLOOR} mm/s (ENCODING_FINDINGS 4.4).
Per session, mean over pushes of the frozen master's relative reconstruction error per 5 Hz session bin (`proj/`
session_err; pre sessions on their held-out leave-one-day-out basis, `proj_loso/`) and of the face movement energy D,
-2 .. +3 s; then the mean over sessions per animal x epoch (sessions with >= 5 pushes; n in the legend). Columns: in-
trial pushes (cue -1 .. +4 s) and between-trial pushes. Row 2 = random times of the same context >= 2 s from any push
(the control: trial structure alone moves the error). A post-stroke error rise at pushes that random times and pre
sessions do not show = face movement the pre-stroke basis does not contain.""")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

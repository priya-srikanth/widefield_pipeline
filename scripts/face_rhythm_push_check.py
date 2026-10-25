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
animal x epoch table (printed; rule 8 first), push_check.png (traces by epoch, per animal), push_check_pooled.png
(cross-animal pooled, within-animal normalised push - random excess with tests; data/push_check_pooled*.csv).
``--pooled-only`` draws only the pooled figure from the cached session CSV (seconds).
"""
from __future__ import annotations

import argparse
import zlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402

FS = 31.25                 # = push_behaviour.FS
PUSH_FLOOR = 0.3           # = frozen_encoder.PUSH_FLOOR
LAGS = np.arange(-2.0, 3.0001, 0.2)
BASE, POST = (-1.5, -0.5), (0.0, 1.0)
TRIAL = (-1.0, 4.0)        # around the cue: "in-trial" pushes
EPOCHS = frc.EPOCHS


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
    ap.add_argument("--pooled-only", action="store_true",
                    help="only the pooled figure, from data/push_check_sessions.csv (no traces, no recompute)")
    ap.add_argument("--n-boot", type=int, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "push_check"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = figure_layout.sidecar_for(out, "push_check_traces", ".npz")
    fail = []
    if a.pooled_only:
        pooled_figure(pd.read_csv(figure_layout.sidecar_for(out, "push_check_sessions", ".csv")), out, fdir.name,
                      a.n_boot)
        return 0
    if a.plot_only:
        S = load_cache(cache)
    else:
        keys = frc.session_keys(fdir)
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
    col = frc.epoch_colors()         # FR epoch palette (orofacial / 3-D kinematics); animals own `animal_colors`
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
    pooled_figure(S.drop(columns=tr_cols), out, fdir.name, a.n_boot)
    print("->", out)
    return 0


MEASURES = (("d_err_excess", "reconstruction error"), ("d_D_excess", "face movement energy D"))
CONTEXTS = (("trial", "in-trial pushes"), ("between", "between-trial pushes"))


def pooled_table(S: pd.DataFrame) -> pd.DataFrame:
    """Per SESSION (animal x date; a split video's halves pooled, push-count weighted) x context: push - random
    excess of each measure, divided by that animal's SD of the same excess across its PRE sessions (same context).
    Sessions with >= 5 pushes, as the per-animal figure."""
    S = S[S.n_push >= 5].copy()
    S["epoch"] = S.epoch.astype(str)
    S["d_err_excess"] = S.d_err_push - S.d_err_random
    S["d_D_excess"] = S.d_D_push - S.d_D_random
    m = [c for c, _ in MEASURES]
    w = S.assign(**{c: S[c] * S.n_push for c in m}).groupby(["animal", "date", "epoch", "context"], as_index=False)
    P = w[m + ["n_push"]].sum()
    for c in m:
        P[c] = P[c] / P.n_push
        sd = P[P.epoch == "pre"].groupby(["animal", "context"])[c].std(ddof=1)
        P[f"{c}_preSD"] = [sd.get((an, cx), np.nan) for an, cx in zip(P.animal, P.context)]
        P[f"{c}_z"] = P[c] / P[f"{c}_preSD"]
    return P.sort_values(["animal", "context", "date"]).reset_index(drop=True)


def _k_same(post: pd.Series, pre: pd.Series, delta: float) -> int:
    """Animals whose own epoch - pre change has the sign of the pooled change."""
    dd = (post - pre).dropna()
    return int((np.sign(dd) == np.sign(delta)).sum())


def pooled_stats(P: pd.DataFrame, rng, n_boot: int | None = None) -> pd.DataFrame:
    """Per measure x context x epoch: per-animal means, mean +- SEM of animals, the level vs 0 and the change vs pre,
    both by `analysis_kit.boot_delta` (paired animals -> sessions). The level vs 0 is `boot_delta` against a constant
    0 baseline per animal = the animal-weighted nested bootstrap of the level (no second bootstrap). ``rng`` is
    advanced in measure, context, epoch order (sessions in date order, rule 9)."""
    from wfield_local import analysis_kit as ak
    from wfield_local.face_rhythm import stats as frs
    nb = n_boot or ak.N_BOOT
    rows = []
    for c, _ in MEASURES:
        for cx, _ in CONTEXTS:
            d = P[P.context == cx].rename(columns={f"{c}_z": "z"})
            zero = {an: [0.0] for an in sorted(d.animal.unique())}
            lvl = {ep: ak.boot_delta(frs.by_animal(d[d.epoch == ep], "z"), zero, rng, nb) for ep in EPOCHS}
            chg = {r["epoch"]: r for r in frs.epoch_change(d, "z", rng, nb, epochs=EPOCHS[1:])}
            pre_per = d[d.epoch == "pre"].groupby("animal").z.mean()
            for ep in EPOCHS:
                per = d[d.epoch == ep].groupby("animal").z.mean()
                mu = float(per.mean())
                iv, ch = lvl[ep], chg.get(ep)
                rows.append({"measure": c, "context": cx, "epoch": ep, **per.round(4).to_dict(),
                             "n_animals": len(per), "n_sessions": int((d.epoch == ep).sum()),
                             "mean": mu, "sem": float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else np.nan,
                             "k_same_sign": int((np.sign(per) == np.sign(mu)).sum()),
                             "lvl_lo": iv.lo if iv else np.nan, "lvl_hi": iv.hi if iv else np.nan,
                             "lvl_star": bool(iv and iv.excludes_zero),
                             "dpre": ch["delta"] if ch else np.nan, "dpre_lo": ch["ci_lo"] if ch else np.nan,
                             "dpre_hi": ch["ci_hi"] if ch else np.nan, "dpre_star": bool(ch and ch["excludes_0"]),
                             "k_dpre_same_sign": _k_same(per, pre_per, ch["delta"]) if ch else np.nan})
    return pd.DataFrame(rows)


def pooled_figure(S: pd.DataFrame, out: Path, set_name: str, n_boot: int | None = None) -> None:
    """push_check_pooled.png: the push - random excess per measure (rows) x context (columns), within-animal
    normalised, per epoch: animal dots, mean +- SEM of animals, level vs 0 and change vs pre. Per-animal table
    printed first (rule 8)."""
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import figure_layout
    P = pooled_table(S)
    P.to_csv(figure_layout.sidecar_for(out, "push_check_pooled_sessions", ".csv"), index=False, float_format="%.5g")
    pd.set_option("display.width", 220)
    print("\nPOOLED, PER ANIMAL x epoch (rule 8 first): push - random excess / the animal's pre-session SD of it "
          "(same context);\nerr_z < 0 = pushes better described than random times; D_z > 0 = more face movement at pushes.")
    A = P.groupby(["context", "animal", "epoch"]).agg(
        n=("date", "size"), err_z=("d_err_excess_z", "mean"), D_z=("d_D_excess_z", "mean"),
        err_preSD=("d_err_excess_preSD", "first"), D_preSD=("d_D_excess_preSD", "first")).reset_index()
    A["epoch"] = pd.Categorical(A.epoch, EPOCHS, ordered=True)
    print(A.sort_values(["context", "animal", "epoch"]).round(4).to_string(index=False))
    T = pooled_stats(P, np.random.default_rng(0), n_boot)
    T.to_csv(figure_layout.sidecar_for(out, "push_check_pooled", ".csv"), index=False, float_format="%.5g")
    print("\nPOOLED stats (mean of animals; lvl = vs 0, dpre = change vs pre; boot_delta 95 % CI)")
    print(T.drop(columns=[c for c in T.columns if c.startswith("PS")]).round(3).to_string(index=False))

    acol, ecol = frc.animal_colors(), frc.epoch_colors()
    animals = sorted(P.animal.unique())
    jit = dict(zip(animals, np.linspace(-0.18, 0.18, len(animals))))
    with frf.style():
        fig, axs = frf.subplots(2, 2, figsize=(11, 8.6), squeeze=False)
        for r, (c, ttl) in enumerate(MEASURES):
            for k, (cx, cttl) in enumerate(CONTEXTS):
                ax = axs[r, k]
                t = T[(T.measure == c) & (T.context == cx)].set_index("epoch")
                for e, ep in enumerate(EPOCHS):
                    if ep not in t.index:
                        continue
                    for an in animals:
                        v = t.loc[ep].get(an, np.nan)
                        if np.isfinite(v):
                            ax.plot(e + jit[an], v, "o", ms=5, color=acol[an], alpha=0.85,
                                    label=an if (e == 0 and r == 0 and k == len(CONTEXTS) - 1) else None)
                    ax.errorbar(e, t.loc[ep, "mean"], yerr=t.loc[ep, "sem"], fmt="s", ms=8, color=ecol[ep],
                                mec="k", ecolor="k", capsize=4, zorder=5)
                ax.axhline(0, color="k", lw=0.6)
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.25 * (hi - lo))
                labels = []
                for e, ep in enumerate(EPOCHS):
                    row = t.loc[ep] if ep in t.index else None
                    if row is None:
                        labels.append(ep)
                        continue
                    ax.text(e, 0.98, f"vs 0{'*' if row.lvl_star else ''}\n{row.k_same_sign}/{row.n_animals}",
                            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5)
                    if ep == "pre" or not np.isfinite(row.dpre):
                        labels.append(ep)
                    else:
                        labels.append(f"{ep}\nΔpre {row.dpre:+.2f}{'*' if row.dpre_star else ''}\n"
                                      f"{int(row.k_dpre_same_sign)}/{row.n_animals}")
                ax.set_xticks(range(len(EPOCHS)), labels, fontsize=8.5)
                ax.set_xlim(-0.5, len(EPOCHS) - 0.5)
                ax.set_title(f"{ttl}, {cttl}", fontsize=10.5)
                if k == 0:
                    ax.set_ylabel("push - random (pre-SD units)")
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{set_name}: pushes vs random times, pooled across animals")
        frf.save(fig, out / "push_check_pooled.png", f"""
Cross-animal pooled version of push_check.png. Per SESSION (sessions with >= 5 pushes; a split video's halves pooled,
push-weighted) and context (in-trial = cue -1 .. +4 s; between trials = the rest): (0 .. 1 s) - (-1.5 .. -0.5 s) of
the frozen basis's relative reconstruction error (row 1; pre sessions on their held-out leave-one-day-out basis) and of
the face movement energy D (row 2), push minus random times of the same context. NORMALISED WITHIN ANIMAL by DIVIDING
by that animal's SD of the same push - random excess across its PRE sessions (same context): the excess is a signed
difference, so a log ratio does not apply; the SD keeps 0 = "push same as random" and puts the animals on one noise
scale. Row 1 below 0 = the pre basis describes the face at pushes BETTER than at random times; row 2 above 0 = more
face movement at pushes. Dots = per-animal epoch means (animal colours), squares = mean of animals +- SEM (epoch
colours). Top of each epoch: "vs 0" = level against 0 by `analysis_kit.boot_delta` against a constant-0 baseline (the
paired animals -> sessions draw, animals weighted equally), * = 95 % interval excludes 0; k/n = animals on the side of
the mean. Post-epoch tick labels: change vs pre ("Dpre", `boot_delta`, paired animals -> sessions), * = 95 % interval
excludes 0, k/n = animals whose own change has that sign. {n_boot or 4000} draws, seed 0, uncorrected. PS95 acute = 1
session. Tables: data/push_check_pooled.csv (stats), data/push_check_pooled_sessions.csv (per session).""")


if __name__ == "__main__":
    raise SystemExit(main())

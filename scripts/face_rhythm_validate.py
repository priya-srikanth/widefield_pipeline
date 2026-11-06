"""FaceRhythm factor identity vs independent behaviour: LP licks (executed direction, time in trial, bout position, spout
position, lick frequency) and the treadmill (Priya 10-22; computation `wfield_local/face_rhythm_validation.py`).

    python scripts/face_rhythm_validate.py --set pilot_pre_PS92_PS93 [--epoch pre] [--jobs 3]

-> ``<dlc>/face_rhythm/cam4/validation/<set>/`` (methods for every figure in its README.md, `face_rhythm_figures`):
  factor_vs_licks.png        per factor, expression around each lick: (1) by executed direction x early / late in the
                             trial; (2) by the lick's number in its bout (1st / 2nd-3rd / 4th+); (3) by the trial's spout
                             position; (4) at the lick vs instantaneous lick frequency, by direction.
  factor_vs_running.png      per factor, expression around treadmill running-bout onsets (whole session).
  factor_correlations.png    factors x [lick rate L / C / R, lick frequency, time from cue, speed in trials, speed
                             outside licks]: Pearson r, session means.
  data/                      the tables (figure_layout.sidecar_for).
Session means first, then the mean over sessions (each session counts once). Absolute expression (weights x D).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DIR_COL = {"left": "tab:blue", "centre": "0.4", "right": "tab:red"}
PHASES = {"early": (0.0, 1.0), "late": (1.5, 3.0)}
NUM = {"1st lick": (1, 1), "2nd-3rd": (2, 3), "4th+": (4, 999)}
NUM_COL = {"1st lick": "tab:purple", "2nd-3rd": "tab:green", "4th+": "tab:olive"}
FREQ_BINS = [4, 5.5, 6.5, 7.5, 8.5, 11]
CORR_COLS = ["rate_left", "rate_centre", "rate_right", "lick_freq_hz", "t_from_cue", "speed_mm_s"]
MIN_LICKS = 5


def _class_curves(S: list, mask_fn) -> tuple[np.ndarray | None, int, int]:
    """Mean lick-triggered curve (lags, R) over sessions of the licks ``mask_fn(L)`` selects (a session needs
    MIN_LICKS), with the total lick count and the number of sessions."""
    xs, n = [], 0
    for s in S:
        m = mask_fn(s["licks"]).to_numpy()
        if m.sum() >= MIN_LICKS:
            xs.append(s["lta"][m].mean(0))
            n += int(m.sum())
    return (np.mean(xs, 0) if xs else None), n, len(xs)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")

    from wfield_local.face_rhythm.factors import position_style
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_validation as frv
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="pilot_pre_PS92_PS93", help="factor set (or <bundle>/<set>)")
    ap.add_argument("--epoch", default="pre", help="sessions of this epoch ('all' for every projected session)")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, set_name=a.set)
    sess = sorted({tuple(f.stem.split("_")[:2]) for f in (fdir / "proj").glob("*.npz")
                   if a.epoch == "all" or str(np.load(f)["epoch"]) == a.epoch})
    if not sess:
        sys.exit(f"no projected {a.epoch} sessions in {fdir}")
    res, fails = ak.fan_sessions([(str(fdir), an, d) for an, d in sess], frv.session, jobs=a.jobs,
                                 key=lambda j: (j[1], j[2]))
    for j, e in fails:
        print(f"  !! {j[1]}_{j[2]}: {e}")
    S = [v for _, v in res]
    R = S[0]["lta"].shape[-1]
    labels = fro.labels(fdir)
    names = [lb if not lb.startswith("fr_") or not lb[3:].isdigit() else "" for lb in labels]
    out = Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "validation" / fro.set_tag(a.set)
    writeguard.assert_writable(out.parent)
    out.mkdir(parents=True, exist_ok=True)
    labs = ", ".join(f"{s['animal']}_{s['date'][4:]}" for s in S)
    common = (f"Set `{a.set}`; {len(S)} {a.epoch} sessions ({labs}). Expression = projected temporal weights x the FR "
              f"normaliser D (ABSOLUTE; a.u. on the animal's scale), 25 Hz trial grid (engaged trials, cue -2..+4 s). "
              f"Licks: v36 LP `licks_all.csv`, every lick whose +-1 s window lies inside a projected trial window; "
              f"executed DIRECTION from the lick's own angle (`peak_angle_deg`, + = the animal's left): left > "
              f"+{frv.ANGLE_DEG:.0f} deg, right < -{frv.ANGLE_DEG:.0f} deg, centre between (not the spout). Curves: mean "
              f"over a session's licks of a class (>= {MIN_LICKS} licks), then the mean over sessions; legend n = licks. "
              f"Licks come in 6-8 Hz bouts, so a +-1 s window around one lick also holds its neighbours.")

    # ---------------------------------------------------------------- licks: 4 views per factor
    rows = {}
    for dname in DIR_COL:
        for ph, (lo, hi) in PHASES.items():
            rows[("dir", dname, ph)] = _class_curves(
                S, lambda L, d=dname, lo=lo, hi=hi: (L.direction == d) & (L.t_rel >= lo) & (L.t_rel < hi))
    for nm, (lo, hi) in NUM.items():
        rows[("num", nm)] = _class_curves(S, lambda L, lo=lo, hi=hi: (L.n_in_bout >= lo) & (L.n_in_bout <= hi))
    positions = sorted({p for s in S for p in s["licks"].position.dropna().unique()})
    for p in positions:
        rows[("pos", p)] = _class_curves(S, lambda L, p=p: L.position == p)
    w0 = (frv.LAGS >= 0) & (frv.LAGS <= 0.2)
    fq = {}
    for dname in DIR_COL:
        for lo, hi in zip(FREQ_BINS[:-1], FREQ_BINS[1:]):
            c, n, _ = _class_curves(S, lambda L, d=dname, lo=lo, hi=hi: (L.direction == d) & (L.freq_hz >= lo)
                                    & (L.freq_hz < hi))
            if c is not None:
                fq.setdefault(dname, []).append(((lo + hi) / 2, c[w0].mean(0), n))
    with frf.style():
        fig, axs = frf.subplots(R, 4, figsize=(24, 3.3 * R), squeeze=False, sharex="col")
        for k in range(R):
            a0, a1, a2, a3 = axs[k]
            nm = f" {names[k]}" if names[k] else ""
            for key, (c, n, _) in rows.items():
                if c is None:
                    continue
                if key[0] == "dir":
                    a0.plot(frv.LAGS, c[:, k], color=DIR_COL[key[1]], ls="-" if key[2] == "early" else "--", lw=1.3,
                            label=f"{key[1]} {key[2]} ({n})")
                elif key[0] == "num":
                    a1.plot(frv.LAGS, c[:, k], color=NUM_COL[key[1]], lw=1.4, label=f"{key[1]} ({n})")
                else:
                    a2.plot(frv.LAGS, c[:, k], lw=1.2, label=f"{key[1]} ({n})", **position_style(key[1]))
            for dname, pts in fq.items():
                a3.plot([p[0] for p in pts], [p[1][k] for p in pts], "o-", color=DIR_COL[dname], ms=4, lw=1.2,
                        label=dname)
            for ax, t in ((a0, "direction x time in trial"), (a1, "lick number in bout"), (a2, "spout position")):
                ax.axvline(0, color="k", lw=0.5)
                ax.set_title(f"factor {k}{nm}: {t}" if ax is a0 else t)
                ax.set_ylabel("a.u. (weight x D)" if ax is a0 else None)
            a3.set_title("at the lick vs lick frequency")
            if k == R - 1:
                for ax in (a0, a1, a2):
                    ax.set_xlabel("s from lick onset")
                a3.set_xlabel("lick frequency (Hz)")
            if k == 0:
                for ax in (a0, a1, a2, a3):
                    frf.legend_outside(ax, fontsize=8)
        fig.suptitle(f"{a.set}: factor expression around licks ({len(S)} {a.epoch} sessions)")
        frf.save(fig, out / "factor_vs_licks.png", f"""
{common}
* **direction x time in trial**: licks by executed direction (colour) and by when they fall after the trial's cue:
  EARLY = onset 0-1 s (solid), LATE = 1.5-3 s (dashed); licks at 1-1.5 s and before the cue are not in these curves.
* **lick number in bout**: the lick's position in its bout (v36 `bout_idx` within the trial, ordered by onset): the
  bout's FIRST lick, the 2nd-3rd, the 4th and later -- separates lick INITIATION from ongoing licking.
* **spout position**: the trial's spout position (all of its licks, any direction); left blue / right red / centre
  grey, close solid / far dashed.
* **at the lick vs lick frequency**: mean expression over 0..+0.2 s after onset, by the lick's instantaneous frequency
  (1 / interval to the next lick of the same trial, intervals < {frv.MAX_ILI_S} s; bins {FREQ_BINS} Hz), by direction.""")

    # ---------------------------------------------------------------- running onsets
    run = [s["run_ta"].mean(0) for s in S if len(s["run_ta"])]
    n_on = sum(s["run_onsets"] for s in S)
    ncol = 5
    nrow = int(np.ceil(R / ncol))
    with frf.style():
        fig, axs = frf.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False, sharex=True)
        for k in range(nrow * ncol):
            ax = axs.flat[k]
            if k >= R:
                ax.axis("off")
                continue
            if run:
                ax.plot(frv.RUN_LAGS, np.mean([r[:, k] for r in run], 0), color="tab:green", lw=1.4)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_title(f"factor {k}{' ' + names[k] if names[k] else ''}")
            if k % ncol == 0:
                ax.set_ylabel("a.u. (weight x D)")
            if k >= R - ncol:
                ax.set_xlabel("s from running onset")
        fig.suptitle(f"{a.set}: factor expression around running-bout onsets ({n_on} onsets, {len(run)} sessions)")
        frf.save(fig, out / "factor_vs_running.png", f"""
{common}
Running-bout onsets: the canonical DAQ treadmill bouts (`behavior_events.get_or_compute`: speed > 3 mm/s, gaps < 0.3 s
merged, >= 2 s long), every onset of the session whose -3..+5 s window lies inside the projected series. Expression:
the whole-session series (`face_rhythm_outputs.combined_series`: 25 Hz in engaged-trial windows, 5 Hz elsewhere),
interpolated every 0.2 s; mean over a session's onsets, then over sessions.""")

    # ---------------------------------------------------------------- correlation table
    C = []
    for s in S:
        B = s["bins"]
        row = {}
        for k in range(R):
            for c in CORR_COLS:
                m = B[[f"f{k}", c]].dropna()
                row[(k, c)] = np.corrcoef(m[f"f{k}"], m[c])[0, 1] if len(m) > 50 and m[c].std() > 0 else np.nan
            row[(k, "speed_lickfree_session")] = s["r_speed_lickfree"][k]
        C.append(row)
    cols = CORR_COLS + ["speed_lickfree_session"]
    M = np.array([[np.nanmean([c[(k, col)] for c in C]) for col in cols] for k in range(R)])
    T = pd.DataFrame(M, columns=cols, index=[f"f{k}{' ' + names[k] if names[k] else ''}" for k in range(R)])
    T.to_csv(figure_layout.sidecar_for(out, "factor_correlations", ".csv"), float_format="%.3f")
    pd.concat([s["licks"].assign(animal=s["animal"], date=s["date"]) for s in S]).to_csv(
        figure_layout.sidecar_for(out, "factor_vs_licks_licks", ".csv"), index=False, float_format="%.4f")
    (out / "data" / "sessions.json").write_text(json.dumps(labs.split(", ")))
    with frf.style():
        fig, ax = frf.subplots(figsize=(11, 0.55 * R + 2.6))
        lim = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(R):
            for j in range(len(cols)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(cols)), ["lick rate L", "lick rate C", "lick rate R", "lick freq", "time from cue",
                                         "speed (trials)", "speed, no licks"], rotation=30, ha="right")
        ax.set_yticks(range(R), T.index)
        fig.colorbar(im, ax=ax, label="Pearson r (mean over sessions)")
        fig.suptitle(f"{a.set}: factor vs licks and treadmill ({len(S)} {a.epoch} sessions)")
        frf.save(fig, out / "factor_correlations.png", f"""
{common}
Pearson r per session, then the mean over sessions. Columns 1-6 over the 40 ms bins of the projected trial windows:
lick RATES by executed direction (licks / s, Gaussian sigma {frv.SMOOTH_S} s), lick FREQUENCY (instantaneous, only
where a bout is ongoing), TIME from the cue, treadmill SPEED (`behavior_events.session_speed`, the canonical calibrated,
smoothed speed, cached per session at 25 Hz). Last column: the WHOLE session (25 / 5 Hz series), bins more than
{frv.LICK_FREE_S} s from any LP lick, r with speed. CAVEAT: absolute expression includes the frame's overall movement,
so every factor correlates positively with speed outside licks -- compare factors relatively. Table:
`data/factor_correlations.csv`; the per-lick table: `data/factor_vs_licks_licks.csv`.""")
    print(T.round(2).to_string())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

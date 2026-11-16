"""PER-TRIAL and PER-LICK readouts of the frozen master factors (Priya 11-16: "should other FR figures be done per
trial? eg temporal amplitude?" and "why doesn't far contra show increased lick_left when the animal makes leftward
licks after the stroke -- isolated licks not falling into these factors?").

The trial-averaged readouts (`face_rhythm_peak.py`, Figs 1.1 / 2.1 / 5.1-5.4) take the PEAK and CENTRE OF MASS of a
session's trial average. A trial average falls when the movement happens on fewer trials or at more scattered times,
even if each trial's movement is as large as before, so it mixes amplitude, prevalence and timing. Here every quantity
is computed per TRIAL (or per LICK) first and summarised per session; the statistics stay at the session level (the
paired animals -> sessions bootstrap, `face_rhythm.stats.epoch_change`), never treating trials as independent.

Inputs: `proj/` (frozen master `master_pre_v1`, engaged trials, cue -2 .. +4 s, 25 Hz; absolute = weight x D; lossy bins
NaN) and the LP v36 whole-session lick table (`face_rhythm.licks.lick_table`: direction by executed angle, |angle| > 12
deg; v36 bouts).

Per session x spout (sessions split over two videos are merged by label):
  licks       leftward / rightward licks per trial (cue 0 .. +4 s), fraction of trials with >= 1 leftward lick
  window      lick_left / lick_right mean expression, cue 0 .. +2 s vs 0 .. +4 s (trial mean)
  per_lick    lick_left at each LEFTWARD lick and lick_right at each RIGHTWARD lick (mean 0 .. +0.2 s after onset, as
              Fig 2.2), split ISOLATED (the lick's v36 bout has one lick, or no bout) vs IN-BOUT
  amplitude   peak of the trial average (as Fig 2.1, cue 0 .. +4 s) vs the MEDIAN of per-trial peaks
  timing      centre of mass of the trial average above its pre-cue level (as Fig 5.3) vs the median per-trial centre
              of mass and the median per-trial peak time
All amplitudes as log2 vs the animal's own pre sessions (pre = 0); times as seconds minus the animal's pre mean.

    python scripts/face_rhythm_per_trial.py [--jobs N] [--plot-only]
-> ``<readouts>/<set>/per_trial/``: data/per_trial_trials.pkl (one row per trial), data/per_trial_licks.pkl
   (one row per lick), data/per_trial_sessions.csv (session x spout x quantity), data/per_trial_stats.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402

UNITS = ("lick_left", "lick_right")
#: which lick direction each unit is read at (per-lick readout)
UNIT_DIR = {"lick_left": "left", "lick_right": "right"}
POS = frc.POSITIONS_DISPLAY
PER_LICK_S = 0.2
#: a cell is tested only with a pre baseline in at least this many animals
MIN_PRE_ANIMALS = 3


def _worker(task):
    """One proj file -> (trial rows, lick rows). Module level (spawn pickles by name)."""
    key, file, animal, date, epoch, label, names = task
    from wfield_local.face_rhythm import licks
    from wfield_local.paths import PathResolver
    with np.load(file, allow_pickle=True) as z:
        o = np.argsort(z["cue_daq_s"])
        X = (z["trials"].astype(float) * z["trials_D"][..., None])[o]               # (n_tr, 151, R)
        lossy = z["trials_lossy"][o].astype(bool)
        cues = z["cue_daq_s"][o].astype(float)
        pos = z["position"][o].astype(str)
        ts = z["t_ms"].astype(float) / 1000.0
    X[lossy] = np.nan
    pre_w, w2, w4 = ts < 0, (ts >= 0) & (ts <= 2), (ts >= 0) & (ts <= 4)
    t4 = ts[w4]
    trows = []
    for i in range(len(cues)):
        r = {"key": key, "label": label, "animal": animal, "epoch": epoch, "trial": i, "pos": pos[i]}
        for u in UNITS:
            x = X[i, :, names.index(u)]
            s4 = x[w4]
            r[f"{u}_mean02"] = np.nanmean(x[w2])
            r[f"{u}_mean04"] = np.nanmean(s4)
            if np.isfinite(s4).sum() >= 0.8 * len(s4):
                k = np.nanargmax(s4)
                r[f"{u}_peak04"], r[f"{u}_tpeak04"] = s4[k], t4[k]
                rise = np.clip(s4 - np.nanmean(x[pre_w]), 0, None)
                rise[~np.isfinite(rise)] = 0
                r[f"{u}_com04"] = (t4 * rise).sum() / rise.sum() if rise.sum() > 0 else np.nan
            else:
                r[f"{u}_peak04"] = r[f"{u}_tpeak04"] = r[f"{u}_com04"] = np.nan
        trows.append(r)
    T = pd.DataFrame(trows)
    try:
        L = licks.lick_table(PathResolver(), animal, date, cues)
    except FileNotFoundError:
        return T.assign(has_licks=False), pd.DataFrame()
    T["has_licks"] = True
    L = L[(L.cue_idx >= 0) & (L.t_from_cue >= 0) & (L.t_from_cue < 4)].copy()
    for d in ("left", "right", "centre"):
        c = L[L.direction == d].groupby("cue_idx").size()
        T[f"n_{d}04"] = T.trial.map(c).fillna(0).astype(int)
        c2 = L[(L.direction == d) & (L.t_from_cue < 2)].groupby("cue_idx").size()
        T[f"n_{d}02"] = T.trial.map(c2).fillna(0).astype(int)
    # isolation: bout size from the FULL session table (a bout can start before the window)
    Lall = licks.lick_table(PathResolver(), animal, date, cues)
    bsz = Lall.dropna(subset=["bout_idx"]).groupby(["trial_id", "bout_idx"]).size()
    L["bout_size"] = [bsz.get((t, b), 1) if np.isfinite(b) else 1 for t, b in zip(L.trial_id, L.bout_idx)]
    L["isolated"] = L.bout_size == 1
    lrows = []
    for rr in L[L.direction.isin(["left", "right"])].itertuples():
        i, t0 = int(rr.cue_idx), float(rr.t_from_cue)
        g = (ts >= t0) & (ts <= t0 + PER_LICK_S)
        if not g.any():
            g = np.abs(ts - t0) == np.abs(ts - t0).min()
        row = {"key": key, "label": label, "animal": animal, "epoch": epoch, "trial": i, "pos": pos[i],
               "direction": rr.direction, "t_from_cue": t0, "isolated": bool(rr.isolated),
               "bout_size": int(rr.bout_size)}
        for u in UNITS:
            row[u] = float(np.nanmean(X[i, g, names.index(u)]))
        lrows.append(row)
    return T, pd.DataFrame(lrows)


def session_table(T: pd.DataFrame, Lk: pd.DataFrame) -> pd.DataFrame:
    """Session x spout x quantity (long). Sessions split over two videos are merged by label (trials pooled)."""
    rows = []
    T = T.copy()
    T["pos_all"] = "all"
    # first lick of each direction per trial (s from cue, 0..+4 s); trials are indexed per proj KEY (split videos)
    for d in ("left", "right"):
        fl = Lk[Lk.direction == d].groupby(["key", "trial"]).t_from_cue.min().rename(f"first_{d}") if len(Lk) \
            else pd.Series(dtype=float, name=f"first_{d}")
        T = T.join(fl, on=["key", "trial"])
    for (lab, an, ep), g in T.groupby(["label", "animal", "epoch"], sort=True):
        for p in (*POS, "all"):
            m = g if p == "all" else g[g.pos == p]
            if len(m) < 3:
                continue
            base = {"label": lab, "animal": an, "epoch": ep, "pos": p, "n_trials": len(m)}
            if m.has_licks.all():
                for d in ("left", "right"):
                    rows.append({**base, "q": f"{d}_licks_per_trial", "v": m[f"n_{d}04"].mean()})
                    rows.append({**base, "q": f"frac_trials_{d}", "v": (m[f"n_{d}04"] > 0).mean()})
            for u in UNITS:
                for c, q in ((f"{u}_mean02", "mean02"), (f"{u}_mean04", "mean04")):
                    rows.append({**base, "q": f"{u}|{q}", "v": m[c].mean()})
                rows.append({**base, "q": f"{u}|median_trial_peak", "v": m[f"{u}_peak04"].median()})
                rows.append({**base, "q": f"{u}|median_trial_tpeak", "v": m[f"{u}_tpeak04"].median()})
                rows.append({**base, "q": f"{u}|median_trial_com", "v": m[f"{u}_com04"].median()})
                # CONDITIONED ON MOVEMENT (11-16): only trials with >= 1 lick in the unit's direction (cue 0..+4 s).
                # Removes the two per-trial artifacts of flat trials -- a peak that is just the largest noise value,
                # and a peak time / centre of mass that drifts to the window middle. Prevalence = frac_trials_<dir>.
                # Min 1 trial (pre far-contra leftward licks are 0-5 per session); n_trials records the count.
                if m.has_licks.all():
                    d = UNIT_DIR[u]
                    on = m[m[f"n_{d}04"] > 0]
                    if len(on):
                        b2 = {**base, "n_trials": len(on)}
                        rows.append({**b2, "q": f"{u}|lick_trial_peak", "v": on[f"{u}_peak04"].median()})
                        rows.append({**b2, "q": f"{u}|lick_trial_mean04", "v": on[f"{u}_mean04"].mean()})
                        rows.append({**b2, "q": f"{u}|lick_trial_tpeak", "v": on[f"{u}_tpeak04"].median()})
                        rows.append({**b2, "q": f"{u}|lick_trial_com", "v": on[f"{u}_com04"].median()})
                        rows.append({**b2, "q": f"{u}|lick_trial_first_lick", "v": on[f"first_{d}"].median()})
                        rows.append({**b2, "q": f"{u}|lick_trial_peak_minus_first_lick",
                                     "v": (on[f"{u}_tpeak04"] - on[f"first_{d}"]).median()})
                    off = m[m[f"n_{d}04"] == 0]
                    if len(off):
                        rows.append({**base, "n_trials": len(off), "q": f"{u}|nolick_trial_peak",
                                     "v": off[f"{u}_peak04"].median()})
    S = pd.DataFrame(rows)
    if len(Lk):
        for (lab, an, ep), g in Lk.groupby(["label", "animal", "epoch"], sort=True):
            for p in (*POS, "all"):
                m0 = g if p == "all" else g[g.pos == p]
                for u in UNITS:
                    m1 = m0[m0.direction == UNIT_DIR[u]]
                    for cls, m in (("all", m1), ("isolated", m1[m1.isolated]), ("inbout", m1[~m1.isolated])):
                        if len(m) >= 3:
                            rows.append({"label": lab, "animal": an, "epoch": ep, "pos": p, "n_trials": len(m),
                                         "q": f"{u}|per_{UNIT_DIR[u]}_lick_{cls}", "v": m[u].mean()})
        S = pd.DataFrame(rows)
    return S


def trial_average_readouts(fdir: Path, names: list[str]) -> pd.DataFrame:
    """Peak and centre of mass of the TRIAL AVERAGE per session x spout, exactly as `face_rhythm_peak.peaks`."""
    import face_rhythm_peak as frp
    rows = []
    for split in ("position", "none"):
        P = frp.peaks(fdir, split, (0.0, 4.0))
        for u in UNITS:
            d = P[P.factor == names.index(u)]
            for q, c in (("peak_of_average", "peak"), ("com_of_average", "com_s")):
                rows.append(pd.DataFrame({"label": d.label, "animal": d.animal, "epoch": d.epoch,
                                          "pos": d.group.replace({"all": "all"}), "n_trials": d.n_trials,
                                          "q": f"{u}|{q}", "v": d[c]}))
    return pd.concat(rows, ignore_index=True)


#: quantities on a log2-vs-pre scale; the rest (times, fractions) as difference from the animal's pre mean
TIME_Q = ("median_trial_tpeak", "median_trial_com", "com_of_average", "lick_trial_tpeak", "lick_trial_com",
          "lick_trial_first_lick", "lick_trial_peak_minus_first_lick")


def relative(S: pd.DataFrame) -> pd.DataFrame:
    S = S.copy()
    is_t = S.q.str.split("|").str[-1].isin(TIME_Q) | S.q.str.startswith("frac_")
    pre = S[S.epoch == "pre"]
    # log2 on the session value, minus the mean log2 of the animal's pre sessions (pre = 0). Floors: 1e-3 licks / trial;
    # factor expression floored at 1 (as Fig 7.14): a near-zero session median gave log2 ~ -20 and swamped a row
    floor = np.where(S.q.str.contains("|", regex=False), 1.0, 1e-3)
    S["lv"] = np.where(is_t, S.v, np.log2(np.maximum(S.v, floor)))
    ref = S[S.epoch == "pre"].groupby(["animal", "pos", "q"]).lv.mean().rename("ref")
    S = S.join(ref, on=["animal", "pos", "q"])
    S["rel"] = S.lv - S.ref
    S["scale"] = np.where(is_t, "diff", "log2")
    del pre
    return S


def stats(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local.face_rhythm import stats as frs
    rng = np.random.default_rng(0)
    rows = []
    for q in sorted(S.q.unique()):                       # fixed order: seeded draws reproduce
        for p in (*POS, "all"):
            d = S[(S.q == q) & (S.pos == p)]
            # no test without a pre baseline in >= 3 animals (e.g. leftward licks at close contra: absent pre-stroke)
            if d.epoch.nunique() < 2 or d[d.epoch == "pre"].animal.nunique() < MIN_PRE_ANIMALS:
                continue
            rows += [dict(q=q, pos=p, **r) for r in frs.epoch_change(d, "rel", rng)]
    ST = pd.DataFrame(rows)
    ST = frs.add_holm(ST[ST.pos != "all"], ["q"]).pipe(lambda a: pd.concat([a, ST[ST.pos == "all"]]))
    return ST


#: rows of the movement-conditioned figure: (quantity, y label, scale)
COND_ROWS = (("frac_trials_right", "lick_right:\ntrials with a right lick\n(fraction, vs own pre)", "diff"),
             ("lick_right|lick_trial_peak", "lick_right peak,\ntrials WITH a right lick\nlog2(vs own pre)", "log2"),
             ("frac_trials_left", "lick_left:\ntrials with a left lick\n(fraction, vs own pre)", "diff"),
             ("lick_left|lick_trial_peak", "lick_left peak,\ntrials WITH a left lick\nlog2(vs own pre)", "log2"),
             ("lick_left|nolick_trial_peak", "lick_left peak,\ntrials WITHOUT a left lick\nlog2(vs own pre)", "log2"))
POS_LAB = {"close_L": "close ipsi", "close_center": "close mid", "close_R": "close contra", "far_L": "far ipsi",
           "far_center": "far mid", "far_R": "far contra"}
EPOCHS = ("pre", "acute", "subacute", "chronic")


def fig_conditioned(S: pd.DataFrame, ST: pd.DataFrame, out_png: Path) -> None:
    """Prevalence vs amplitude-given-movement (11-16): rows `COND_ROWS`, columns = spouts (canonical order), epochs on
    x: session dots (shape = animal, colour = epoch), mean +- SEM across animals, stars in the top band."""
    import face_rhythm_peak as frp

    from wfield_local import cat_points as cp
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import sig_marks
    from wfield_local.face_rhythm import stats as frs
    from wfield_local.spout_behavior import position_style
    with frf.style():
        fig, axs = frf.subplots(len(COND_ROWS), len(POS), figsize=(2.35 * len(POS), 2.5 * len(COND_ROWS)),
                                squeeze=False, sharey="row")
        for r_i, (q, ylab, _sc) in enumerate(COND_ROWS):
            for c_i, p in enumerate(POS):
                ax = axs[r_i, c_i]
                d = S[(S.q == q) & (S.pos == p)]
                n_pre = d[d.epoch == "pre"].animal.nunique()
                if n_pre < MIN_PRE_ANIMALS:          # no baseline: show nothing, say why (no stars, not in Holm)
                    ax.text(0.5, 0.5, f"pre baseline in\n{n_pre} / 4 animals:\nnot tested", transform=ax.transAxes,
                            ha="center", va="center", fontsize=7.5, color="0.45")
                    ax.set_xticks(range(len(EPOCHS)), [e[:5] for e in EPOCHS], fontsize=6.5)
                    if r_i == 0:
                        ax.set_title(POS_LAB[p], fontsize=9, color=position_style(p)[0])
                    if c_i == 0:
                        ax.set_ylabel(ylab, fontsize=7.5)
                    continue
                frp.epoch_points(ax, d, "rel", seed=100 * r_i + 10 * c_i)
                ch = {r["epoch"]: r for r in ST[(ST.q == q) & (ST.pos == p)].to_dict("records")}
                for e_i, ep in enumerate(EPOCHS[1:], start=1):
                    if ep in ch and frs.mark(ch[ep]):
                        sig_marks.star_top(ax, e_i, frs.mark(ch[ep]))
                ax.axhline(0, color="k", lw=0.5)
                ax.set_xticks(range(len(EPOCHS)),
                              ["pre", *(frp.change_tick(ep, ch.get(ep), interval=False) for ep in EPOCHS[1:])],
                              fontsize=6.5)
                if r_i == 0:
                    ax.set_title(POS_LAB[p], fontsize=9, color=position_style(p)[0])
                if c_i == 0:
                    ax.set_ylabel(ylab, fontsize=7.5)
        cp.add_animal_entries(axs[0, -1], sorted(S.animal.unique()))
        axs[0, -1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        sig_marks.reserve_headroom(fig)
        fig.suptitle("Frozen master factors, per trial: how often the lick movement happens vs how large it is when "
                     "it does", fontsize=10)
        frf.save(fig, out_png, """
Per session and spout, from single trials (engaged trials, cue 0 .. +4 s; frozen master master_pre_v1, absolute =
weight x D). PREVALENCE: fraction of trials with >= 1 LP video lick of the factor's direction (|executed angle| > 12 deg;
lick_right -> rightward, lick_left -> leftward), minus the animal's pre mean. AMPLITUDE: median per-trial peak of the
factor over trials WITH such a lick (row 2, 4) or WITHOUT one (row 5), log2 minus the mean log2 of the animal's pre
sessions; sessions with >= 1 qualifying trial (pre leftward licks at contra spouts are rare: 0-5 trials per session).
Faded open markers = sessions (shape = animal, colour = epoch); filled = mean +- SEM across animals. Under each epoch:
change vs pre, `analysis_kit.boot_delta` (paired animals -> sessions bootstrap, animal-weighted); * = 95 % interval
excludes 0, ** = also Holm over the 6 spouts of that row x epoch. Script: scripts/face_rhythm_per_trial.py.""")


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--plot-only", action="store_true", help="reuse data/per_trial_*.pkl")
    ap.add_argument("--fig-only", action="store_true", help="redraw figures from data/per_trial_{sessions,stats}.csv")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, a.bundle, a.set)
    out = frc.readouts_dir(a.set, rv) / "per_trial"
    writeguard.assert_writable(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    names = frc.factor_names(fdir)
    if a.fig_only:
        fig_conditioned(pd.read_csv(out / "data" / "per_trial_sessions.csv"),
                        pd.read_csv(out / "data" / "per_trial_stats.csv"), out / "per_trial_conditioned.png")
        return 0
    ft, fl = out / "data" / "per_trial_trials.pkl", out / "data" / "per_trial_licks.pkl"
    if a.plot_only and ft.exists():
        T, Lk = pd.read_pickle(ft), pd.read_pickle(fl)
    else:
        tasks = [(r.key, str(r.file), r.animal, r.date, r.epoch, r.label, names)
                 for r in frc.sessions(fdir, "proj").itertuples()]
        res, fail = ak.fan_sessions(tasks, _worker, jobs=a.jobs, key=lambda t: t[0])
        if fail:
            print("FAILED:", [f[0][0] for f in fail])
        T = pd.concat([v[0] for _, v in res], ignore_index=True)
        Lk = pd.concat([v[1] for _, v in res if len(v[1])], ignore_index=True)
        T.to_pickle(ft)
        Lk.to_pickle(fl)
    S = pd.concat([session_table(T, Lk), trial_average_readouts(fdir, names)], ignore_index=True)
    S = relative(S)
    S.to_csv(out / "data" / "per_trial_sessions.csv", index=False, float_format="%.5g")
    ST = stats(S)
    ST.to_csv(out / "data" / "per_trial_stats.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 500)
    show = ["q", "pos", "epoch", "PS92", "PS93", "PS94", "PS95", "delta", "ci_lo", "ci_hi", "p_holm"]
    print(ST[[c for c in show if c in ST.columns]].round(3).to_string(index=False))
    fig_conditioned(S, ST, out / "per_trial_conditioned.png")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PEAK temporal-factor amplitude over recovery, relative to pre-stroke (Priya 10-24; the 2pRAM-cohort manuscript's FR
figure: pre-stroke components refit to post sessions, temporal amplitude over days -- licking components fall acutely
and recover, severity = depth + persistence).

Per session and frozen factor: the trial-averaged ABSOLUTE expression (weight x FR normaliser D; engaged trials, cue
-2 .. +4 s on the 25 Hz grid, `proj/`), its PEAK over the response window (cue 0 .. +4 s, default) and the time of that
peak; per animal divided by the mean peak of its pre-stroke sessions. ``--split side`` does the same per spout side
(contra = the animal's right spouts, all lesions left; centre; ipsi), so a contralesional lick factor is read on the
trials that drive it. Lick-count matters: a peak of a trial average falls when fewer trials have the movement (see
`face_rhythm_lick_epochs.py` for per-lick amplitude).

    python scripts/face_rhythm_peak.py [--set master_pre_v1] [--split none|side] [--window 0 4]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/peak/``: peak_sessions.csv, peak_epochs.csv (per animal x epoch, ratio to
pre), peak_by_day[_side].png (per factor: peak / pre over days since stroke, one line per animal, grey = the pre range);
  peak_cohort_by_day[_side].png  (Priya 10-24) per factor: mean +- SEM ACROSS ANIMALS of each animal's mean in day bins
                                 (values already / each animal's own pre mean);
  peak_cohort_by_epoch[_side].png + data/peak_epoch_stats[_side].csv  per factor: per-animal epoch means (dots), mean +-
                                 SEM across animals, change vs pre by `analysis_kit.boot_delta` (paired animals ->
                                 sessions bootstrap, animal-weighted; 95 % interval) + how many animals move the same way.
  timing_by_epoch_{com,t_half}[_split].png + data/timing_epoch_stats[_split].csv  (Priya 10-24: WHEN is each behaviour
                                 expressed relative to the cue, and does that change?) per factor: the trial average's
                                 CENTRE OF MASS above its pre-cue level over the window, and its HALF-RISE time (first
                                 bin at half the rise from pre-cue level to peak), s from cue; per-animal epoch means,
                                 change vs pre in seconds by `boot_delta`. Reference row: the median FIRST-LICK latency
                                 of the same trials (LP licks), so a lick factor's shift can be read against reaction time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EPOCHS = ("pre", "acute", "subacute", "chronic")
#: spout positions, contralesional first (lesions LEFT: the animal's R spouts are contralesional)
POSITIONS = ("far_R", "close_R", "far_center", "close_center", "close_L", "far_L")
GROUP_ORDER = ("contra", "centre", "ipsi", "all", *POSITIONS)
#: day bins for the cohort-by-day view (animals are imaged on different days): (label, first day, last day)
DAY_BINS = (("pre", -999, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8-14", 8, 14), ("15-21", 15, 21), ("22-35", 22, 35),
            ("36+", 36, 999))
SIDE = {"far_R": "contra", "close_R": "contra", "far_center": "centre", "close_center": "centre",
        "far_L": "ipsi", "close_L": "ipsi"}           # lesions LEFT: the animal's right spouts are contralesional


def _groups(pos: np.ndarray, split: str) -> dict[str, np.ndarray]:
    if split == "none":
        return {"all": np.ones(len(pos), bool)}
    if split == "side":
        return {s: np.array([SIDE.get(p) == s for p in pos]) for s in ("contra", "centre", "ipsi")}
    return {p: pos == p for p in POSITIONS}              # "position": each spout on its own (close vs far differ)


def first_lick(t, split: str) -> list[dict]:
    """Median first-lick latency (s from cue, LP licks, first lick at or after the cue) per trial group of a proj task."""
    from wfield_local import face_rhythm_validation as frv
    from wfield_local.paths import PathResolver
    with np.load(t.file, allow_pickle=True) as z:
        cues, pos, date = z["cue_daq_s"].astype(float), z["position"].astype(str), str(z["date"])
    o = np.argsort(cues)
    cues, pos = cues[o], pos[o]
    try:
        L = frv.lick_table(PathResolver(), t.animal, date, cues)
    except FileNotFoundError:
        return []
    L = L[(L.t_from_cue >= 0) & (L.cue_idx >= 0)]
    lat = np.full(len(cues), np.nan)
    f = L.groupby("cue_idx").t_from_cue.min()
    lat[f.index.to_numpy(int)] = f.to_numpy()
    return [{"label": t.label, "group": g, "first_lick_s": float(np.nanmedian(lat[m])),
             "n_lick_trials": int(np.isfinite(lat[m]).sum())}
            for g, m in _groups(pos, split).items() if np.isfinite(lat[m]).sum() >= 3]


def peaks(fdir: Path, split: str, window: tuple[float, float]) -> pd.DataFrame:
    from wfield_local import face_rhythm_readouts as frr
    rows = []
    for t in frr._tasks(fdir, "proj").itertuples():
        with np.load(t.file) as z:
            X = z["trials"].astype(float) * z["trials_D"][..., None]          # (trials, 151, R)
            tms, pos = z["t_ms"] / 1000.0, z["position"].astype(str)
        groups = _groups(pos, split)
        w = (tms >= window[0]) & (tms <= window[1])
        for g, m in groups.items():
            if m.sum() < 3:
                continue
            tr = np.nanmean(X[m], 0)                                          # (151, R) trial average
            sub = tr[w]
            k = np.nanargmax(sub, 0)
            # TIMING (Priya 10-24): above the pre-cue level (cue -2..0 s), the centre of mass over the window and the
            # half-rise time (first bin at half of peak - pre-cue); NaN when the factor does not rise above pre-cue
            base = np.nanmean(tr[tms < 0], 0)
            rise = np.clip(sub - base, 0, None)
            amp = sub[k, np.arange(sub.shape[1])] - base
            tw = tms[w]
            com = np.where(rise.sum(0) > 0, (tw[:, None] * rise).sum(0) / np.maximum(rise.sum(0), 1e-12), np.nan)
            half = np.array([tw[np.argmax(sub[:, j] >= base[j] + 0.5 * amp[j])] if amp[j] > 0 else np.nan
                             for j in range(sub.shape[1])])
            rows += [{"label": t.label, "key": t.key, "animal": t.animal, "epoch": t.epoch, "group": g,
                      "factor": j, "peak": float(sub[k[j], j]), "t_peak_s": float(tms[w][k[j]]),
                      "com_s": float(com[j]), "t_half_s": float(half[j]),
                      "n_trials": int(m.sum())} for j in range(X.shape[-1])]
    S = pd.DataFrame(rows)
    # one row per session (a split video, e.g. PS92_0812 v1 / v2, counts once: trial-weighted mean)
    S = (S.assign(wp=S.peak * S.n_trials).groupby(["label", "animal", "epoch", "group", "factor"], as_index=False)
         .agg(wp=("wp", "sum"), n_trials=("n_trials", "sum"), t_peak_s=("t_peak_s", "mean"), com_s=("com_s", "mean"),
              t_half_s=("t_half_s", "mean")))
    S["peak"] = S.wp / S.n_trials
    S = S.drop(columns="wp")
    S["day"] = [frr._day(x) for x in S.label]
    pre = S[S.epoch == "pre"].groupby(["animal", "group", "factor"]).peak
    S = S.join(pre.mean().rename("pre_mean"), on=["animal", "group", "factor"]) \
         .join(pre.min().rename("pre_min"), on=["animal", "group", "factor"]) \
         .join(pre.max().rename("pre_max"), on=["animal", "group", "factor"])
    for c in ("peak", "pre_min", "pre_max"):
        S[f"{c}_vs_pre"] = S[c] / S.pre_mean
    # LOG2 scale (Priya 10-24: halving and doubling equally large): log2(peak) minus the mean log2 peak of the animal's
    # pre-stroke sessions, so pre averages exactly 0; statistics on this scale are symmetric for losses and gains
    lg = np.log2(np.maximum(S.peak, 1e-12))
    S["log2_vs_pre"] = lg - lg.where(S.epoch == "pre").groupby([S.animal, S.group, S.factor]).transform("mean")
    return S


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--split", default="none", choices=("none", "side", "position"))
    ap.add_argument("--window", type=float, nargs=2, default=(0.0, 4.0), help="peak window, s from cue")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "peak"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n.removeprefix("fr_") for n in fro.labels(fdir)]
    sfx = "" if a.split == "none" else f"_{a.split}"
    S = peaks(fdir, a.split, tuple(a.window))
    S["name"] = [names[k] for k in S.factor]
    S.to_csv(figure_layout.sidecar_for(out, f"peak_sessions{sfx}", ".csv"), index=False, float_format="%.5g")
    E = S.groupby(["group", "factor", "name", "animal", "epoch"], observed=True)["peak_vs_pre"].mean().unstack("epoch") \
        .reindex(columns=list(EPOCHS))
    E.to_csv(figure_layout.sidecar_for(out, f"peak_epochs{sfx}", ".csv"), float_format="%.4g")
    pd.set_option("display.width", 220)
    print("PEAK amplitude / pre, per animal x epoch (rule 8: per animal)")
    print(E.round(2).to_string())

    groups = sorted(S.group.unique(), key=GROUP_ORDER.index)
    animals = sorted(S.animal.unique())
    from wfield_local import config
    col = config.animal_color()                          # the repo's one animal palette (configs/animals.yaml)
    with frf.style():
        fig, axs = frf.subplots(len(names), len(groups), figsize=(4.6 * len(groups) + 1.5, 2.3 * len(names)),
                                squeeze=False, sharex=True)
        for k, nm in enumerate(names):
            for j, g in enumerate(groups):
                ax = axs[k, j]
                for an in animals:
                    d = S[(S.factor == k) & (S.animal == an) & (S.group == g)].sort_values("day")
                    if d.empty:
                        continue
                    ax.axhspan(d.pre_min_vs_pre.iloc[0], d.pre_max_vs_pre.iloc[0], color=col[an], alpha=0.08, lw=0)
                    ax.plot(d.day, d.peak_vs_pre, "o-", ms=3, lw=1, color=col[an], label=an)
                ax.axhline(1, color="k", lw=0.5)
                ax.axvline(0.5, color="k", lw=0.5)
                ax.set_title(f"{k} {nm}" + ("" if g == "all" else f" -- {g} spouts"), fontsize=9)
                if j == 0:
                    ax.set_ylabel("peak / pre")
        for ax in axs[-1]:
            ax.set_xlabel("days since stroke")
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: peak temporal-factor amplitude vs pre-stroke"
                     + ("" if a.split == "none" else ", by spout side (contra = right spouts)"))
        frf.save(fig, out / f"peak_by_day{sfx}.png", f"""
Per session and frozen factor: the trial-averaged ABSOLUTE expression (temporal weight x FR normaliser D) on engaged
trials{' of that spout side (contra = the animal right spouts, lesions left; centre; ipsi)' if a.split == 'side' else ''},
cue-aligned on the 25 Hz grid; its PEAK over cue {a.window[0]:g} .. +{a.window[1]:g} s; divided by the mean peak of the
animal's pre-stroke sessions (1 = pre level). One line per animal over days since stroke (vertical line = stroke);
shaded band per animal = the range of its pre-stroke sessions. The peak of a trial average also falls when fewer
trials contain the movement (fewer licks): per-lick amplitude is `face_rhythm_lick_epochs.py` (FINDINGS 5b-master).
Frozen master `{fdir.name}` (pre and post on the same basis; `proj/`). The 2pRAM-cohort manuscript's FR panel.""")
    cohort(S, names, groups, out, sfx, fdir.name, tuple(a.window))
    from wfield_local import face_rhythm_readouts as frr
    FL = pd.DataFrame([r for t in frr._tasks(fdir, "proj").itertuples() for r in first_lick(t, a.split)])
    timing(S, FL, names, groups, out, sfx, fdir.name, tuple(a.window))
    if a.split == "side":
        headline(S, names, out, fdir.name, ("contra", "ipsi"), "peak_side_headline.png")
    elif a.split == "position":
        headline(S, names, out, fdir.name, POSITIONS, "peak_position_headline.png")
    print("->", out)
    return 0


def timing(S: pd.DataFrame, FL: pd.DataFrame, names: list[str], groups: list[str], out: Path, sfx: str, set_name: str,
           window: tuple[float, float]) -> None:
    """WHEN each factor is expressed relative to the cue, by epoch (Priya 10-24). Per animal x epoch the mean over
    sessions of the trial average's centre of mass / half-rise time (s from cue); change vs pre in SECONDS by
    `boot_delta`; reference row = median first-lick latency of the same trials."""
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import config, figure_layout
    acol = config.animal_color()
    rng = np.random.default_rng(0)
    meta = S[["label", "animal", "epoch"]].drop_duplicates("label")
    if len(FL):                               # split videos: trial-weighted mean latency per label
        FL = (FL.assign(w=FL.first_lick_s * FL.n_lick_trials).groupby(["label", "group"], as_index=False)
              .agg(w=("w", "sum"), n=("n_lick_trials", "sum")))
        FL["first_lick_s"] = FL.w / FL.n
        FL = FL.merge(meta, on="label")
    rows_q = [*enumerate(names), (-1, "FIRST LICK (median latency)")]
    stats = []
    for metric in ("com_s", "t_half_s"):
        lab = {"com_s": "centre of mass", "t_half_s": "half-rise time"}[metric]
        with frf.style():
            # NOT sharex: each panel carries its own change-vs-pre tick labels (a shared axis shows the last panel's)
            fig, axs = frf.subplots(len(rows_q), len(groups), figsize=(4.0 * len(groups) + 1.5, 2.0 * len(rows_q)),
                                    squeeze=False)
            for r_i, (k, nm) in enumerate(rows_q):
                for j, g in enumerate(groups):
                    ax = axs[r_i, j]
                    d = (FL[FL.group == g].rename(columns={"first_lick_s": "v"}) if k < 0 else
                         S[(S.factor == k) & (S.group == g)].rename(columns={metric: "v"}))
                    if d.empty:
                        continue
                    A = d.groupby(["animal", "epoch"]).v.mean().unstack().reindex(columns=list(EPOCHS))
                    x = np.arange(len(EPOCHS))
                    for an, row in A.iterrows():
                        ax.plot(x, row.to_numpy(), ".-", color=acol.get(an, "0.6"), lw=0.8, ms=4, alpha=0.8)
                    ax.errorbar(x, A.mean(0).to_numpy(), yerr=(A.std(0, ddof=1) / np.sqrt(A.notna().sum(0))).to_numpy(),
                                fmt="o", color="k", ms=5, capsize=3, zorder=5)
                    pre = {an: list(v.dropna()) for an, v in d[d.epoch == "pre"].groupby("animal").v}
                    pre = {an: v for an, v in pre.items() if v}
                    ticks = ["pre"]
                    for ep in EPOCHS[1:]:
                        post = {an: list(v.dropna()) for an, v in d[d.epoch == ep].groupby("animal").v}
                        iv = ak.boot_delta({an: v for an, v in post.items() if v}, pre, rng)
                        per = A[ep] - A["pre"]
                        sig = bool(iv and (iv.lo > 0 or iv.hi < 0))
                        ticks.append(f"{ep[:5]}\n{iv.point:+.2f}{'*' if sig else ''}" if iv else ep[:5])
                        if metric == "com_s":            # one stats row per factor (+ the first-lick reference)
                            stats.append({"metric": "first_lick_s" if k < 0 else metric, "group": g, "factor": k,
                                          "name": nm, "epoch": ep, **{an: per.get(an, np.nan) for an in sorted(A.index)},
                                          "delta_s": iv.point if iv else np.nan, "ci_lo": iv.lo if iv else np.nan,
                                          "ci_hi": iv.hi if iv else np.nan, "excludes_0": sig,
                                          "n_later": int((per > 0).sum()), "n_animals": int(per.notna().sum())})
                    ax.set_xticks(x, ticks, fontsize=6)
                    ax.set_title(nm + ("" if g == "all" else f" -- {g}"), fontsize=8)
                    if j == 0:
                        ax.set_ylabel("s from cue")
            fig.suptitle(f"{set_name}: WHEN each factor is expressed after the cue ({lab}), by epoch; under each "
                         "label the change vs pre in s (* = 95 % CI excludes 0)")
            how = ("the centre of mass of the expression ABOVE its pre-cue level (cue -2..0 s) over cue "
                   f"{window[0]:g}..+{window[1]:g} s" if metric == "com_s" else
                   "the first bin at which the expression reaches half of its rise from the pre-cue level (cue -2..0 s) "
                   "to its peak in the window")
            frf.save(fig, out / f"timing_by_epoch_{metric.removesuffix('_s')}{sfx}.png", f"""
Timing of each frozen factor's trial-averaged absolute expression (engaged trials{', per trial group' if sfx else ''}),
relative to the cue. {lab.upper()}: {how} (NaN when it does not rise above pre-cue). Coloured: per-animal epoch means
(repo animal colours); black: mean +- SEM across animals. Under each epoch label: change vs pre in SECONDS,
`analysis_kit.boot_delta` (paired animals -> sessions, animal-weighted). Last row: median first-lick latency of the same
trials (LP licks) -- a lick factor that shifts with it is tracking reaction time, not a re-timed movement. A trial
average also mixes fewer / later licks: low-amplitude sessions (acute contralesional licking) give noisy timing.
Frozen master {set_name}.""")
    T = pd.DataFrame(stats)
    T.to_csv(figure_layout.sidecar_for(out, f"timing_epoch_stats{sfx}", ".csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print("\nTIMING (centre of mass, s from cue) change vs pre: per animal FIRST (rule 8), then boot_delta")
    print(T.round(3).to_string(index=False))


def headline(S: pd.DataFrame, names: list[str], out: Path, set_name: str, rows: tuple, fn: str) -> None:
    """The lick-factor result in one figure (Priya 10-24): rows = trial groups (spout SIDE, or each spout POSITION --
    close and far contralesional spouts are different behaviours), columns = the contralesional (lick_right) and
    ipsilesional (lick_left) lick factors; per-animal epoch means (grey, joined), mean +- SEM across animals, the
    `boot_delta` change vs pre (95 % interval) under each epoch's tick label (kept off the data)."""
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_figures as frf
    want = [k for k in (names.index("lick_right") if "lick_right" in names else None,
                        names.index("lick_left") if "lick_left" in names else None) if k is not None]
    if len(want) < 2:
        return
    from wfield_local import config
    acol = config.animal_color()
    rng = np.random.default_rng(0)
    x = np.arange(len(EPOCHS))
    with frf.style():
        fig, axs = frf.subplots(len(rows), 2, figsize=(9.5, 3.9 * len(rows)), squeeze=False, sharey=True)
        for r_i, g in enumerate(rows):
            for c_i, k in enumerate(want):
                ax = axs[r_i, c_i]
                d = S[(S.factor == k) & (S.group == g)]
                A = d.groupby(["animal", "epoch"]).log2_vs_pre.mean().unstack().reindex(columns=list(EPOCHS))
                for an, row in A.iterrows():
                    ax.plot(x, row.to_numpy(), "o-", color=acol.get(an, "0.6"), ms=3, lw=0.9, alpha=0.8)
                    ax.text(x[-1] + 0.08, row.iloc[-1], an, fontsize=7, va="center", color=acol.get(an, "0.4"))
                m, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(A.notna().sum(0))
                ax.errorbar(x, m.to_numpy(), yerr=se.to_numpy(), fmt="o", color="k", ms=6, capsize=3, zorder=5)
                pre = {an: list(v.log2_vs_pre) for an, v in d[d.epoch == "pre"].groupby("animal")}
                ticks = ["pre"]
                for ep in EPOCHS[1:]:
                    post = {an: list(v.log2_vs_pre) for an, v in d[d.epoch == ep].groupby("animal")}
                    iv = ak.boot_delta(post, pre, rng)
                    ticks.append(f"{ep[:5]}\n{iv.point:+.2f}{'*' if (iv.lo > 0 or iv.hi < 0) else ''}\n"
                                 f"[{iv.lo:+.2f}, {iv.hi:+.2f}]" if iv else ep[:5])
                ax.axhline(0, color="k", lw=0.5)
                ax.set_xticks(x, ticks, fontsize=7)
                where = {"contra": "CONTRAlesional-spout trials", "ipsi": "IPSIlesional-spout trials"}.get(g, f"{g} trials")
                ax.set_title(f"{names[k]} -- on {where}", fontsize=10)
                if c_i == 0:
                    ax.set_ylabel("log2(peak / own pre)")
        fig.suptitle(f"{set_name}: contralesional (lick_right) and ipsilesional (lick_left) lick factors by "
                     + ("spout side" if rows[0] == "contra" else "spout position (R = contralesional)"))
        frf.save(fig, out / fn, f"""
Peak of the trial-averaged absolute expression (cue 0..+4 s) of the contralesional lick factor (lick_right, left
column) and the ipsilesional lick factor (lick_left, right column), on trials at each of {', '.join(rows)} (rows;
contra = the animal's right spouts far_R / close_R, ipsi = far_L / close_L; lesions left); each session / the mean of
that animal's pre-stroke sessions on the same trials, on a LOG2 scale (log2 peak minus the mean log2 peak of the
pre-stroke sessions: pre = 0, +1 = doubled, -1 = halved). Under each epoch label: the change vs pre in log2 units and
its interval. Grey: per-animal epoch means (joined); black: mean +- SEM
across the four animals. Numbers: change vs pre, `analysis_kit.boot_delta` (paired animals -> sessions bootstrap,
animal-weighted) with its 95 % interval; * = excludes 0. Frozen master {set_name}; FINDINGS 5d.""")


def cohort(S: pd.DataFrame, names: list[str], groups: list[str], out: Path, sfx: str, set_name: str,
           window: tuple[float, float]) -> None:
    """Cohort views of peak / pre (Priya 10-24): by day bin (mean +- SEM across animals) and by epoch with statistics."""
    from wfield_local import analysis_kit as ak
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import config, figure_layout
    acol = config.animal_color()
    rng = np.random.default_rng(0)
    S = S.copy()
    S["bin"] = None
    for lab, lo, hi in DAY_BINS:
        S.loc[(S.day >= lo) & (S.day <= hi) & ((S.epoch == "pre") == (lab == "pre")), "bin"] = lab
    bins = [b[0] for b in DAY_BINS]
    # per animal: mean of its sessions in each bin / epoch (one value per animal), then across animals
    A_bin = S.dropna(subset=["bin"]).groupby(["group", "factor", "animal", "bin"]).log2_vs_pre.mean().reset_index()
    A_ep = S.groupby(["group", "factor", "animal", "epoch"]).log2_vs_pre.mean().reset_index()
    rows = []
    for (g, k), d in S.groupby(["group", "factor"]):
        pre = {an: list(x.log2_vs_pre) for an, x in d[d.epoch == "pre"].groupby("animal")}
        for ep in EPOCHS[1:]:
            post = {an: list(x.log2_vs_pre) for an, x in d[d.epoch == ep].groupby("animal")}
            iv = ak.boot_delta(post, pre, rng)
            per = {an: float(np.mean(post[an])) for an in sorted(post)}
            n_dn = sum(v < 0 for v in per.values())
            rows.append({"group": g, "factor": k, "name": names[k], "epoch": ep,
                         "mean_vs_pre": float(np.mean(list(per.values()))) if per else np.nan,
                         "sem": float(np.std(list(per.values()), ddof=1) / np.sqrt(len(per))) if len(per) > 1 else np.nan,
                         "delta": iv.point if iv else np.nan, "ci_lo": iv.lo if iv else np.nan,
                         "ci_hi": iv.hi if iv else np.nan, "n_animals": iv.n_animals if iv else len(per),
                         "excludes_0": bool(iv and (iv.lo > 0 or iv.hi < 0)),
                         "n_below_pre": n_dn, "n_above_pre": len(per) - n_dn,
                         **{f"{an}": v for an, v in per.items()}})
    T = pd.DataFrame(rows)
    T.to_csv(figure_layout.sidecar_for(out, f"peak_epoch_stats{sfx}", ".csv"), index=False, float_format="%.4g")
    print("\nPER-ANIMAL epoch means of peak / pre FIRST (rule 8), then the paired bootstrap of the change vs pre")
    print(T[["group", "name", "epoch", *sorted(S.animal.unique()), "delta", "ci_lo", "ci_hi", "excludes_0",
             "n_below_pre"]].round(3).to_string(index=False))

    with frf.style():
        for view in ("day", "epoch"):
            fig, axs = frf.subplots(len(names), len(groups), figsize=(4.4 * len(groups) + 1.5, 2.3 * len(names)),
                                    squeeze=False, sharex=True)
            for kk, nm in enumerate(names):
                for j, g in enumerate(groups):
                    ax = axs[kk, j]
                    if view == "day":
                        d = A_bin[(A_bin.factor == kk) & (A_bin.group == g)]
                        x = np.arange(len(bins))
                        st = d.groupby("bin").log2_vs_pre.agg(["mean", "sem", "count"]).reindex(bins)
                        for _an, da in d.groupby("animal"):
                            ax.plot([bins.index(b) for b in da.bin], da.log2_vs_pre, ".", color=acol.get(_an, "0.6"), ms=4)
                        ax.errorbar(x, st["mean"], yerr=st["sem"], fmt="o-", color="k", ms=4, lw=1.2, capsize=2)
                        ax.set_xticks(x, bins, fontsize=7)
                    else:
                        d = A_ep[(A_ep.factor == kk) & (A_ep.group == g)]
                        x = np.arange(len(EPOCHS))
                        st = d.groupby("epoch").log2_vs_pre.agg(["mean", "sem"]).reindex(EPOCHS)
                        for _an, da in d.groupby("animal"):
                            da = da.set_index("epoch").log2_vs_pre.reindex(EPOCHS)
                            ax.plot(x, da.to_numpy(), ".-", color=acol.get(_an, "0.6"), lw=0.8, ms=4, alpha=0.8)
                        ax.errorbar(x, st["mean"], yerr=st["sem"], fmt="o", color="k", ms=5, capsize=3, zorder=5)
                        ax.set_xticks(x, [e[:5] for e in EPOCHS])
                        tt = T[(T.factor == kk) & (T.group == g)].set_index("epoch")
                        top = ax.get_ylim()[1]
                        for e_i, ep in enumerate(EPOCHS[1:], start=1):
                            if ep in tt.index:
                                r = tt.loc[ep]
                                ax.text(e_i, top, ("*" if r.excludes_0 else "") + f"{int(r.n_below_pre)}/{int(r.n_animals)}v",
                                        ha="center", va="bottom", fontsize=7)
                    ax.axhline(0, color="k", lw=0.5)
                    ax.set_title(f"{kk} {nm}" + ("" if g == "all" else f" -- {g} spouts"), fontsize=9)
                    if j == 0:
                        ax.set_ylabel("log2(peak / pre)")
            fig.suptitle(f"{set_name}: peak factor amplitude vs each animal's pre (log2), mean +- SEM across animals"
                         + (" (day bins)" if view == "day" else " (epochs; * = bootstrap 95 % CI of the change excludes 0; "
                            "k/n v = animals below pre)"))
            frf.save(fig, out / f"peak_cohort_by_{view}{sfx}.png", f"""
Peak of the trial-averaged absolute expression over cue {window[0]:g}..+{window[1]:g} s per session on a LOG2 scale:
log2 peak minus the mean log2 peak of that animal's pre-stroke sessions (each animal its own reference; pre = 0, +1 =
doubled, -1 = halved; statistics on this scale). {'Sessions binned by days since stroke (' + ', '.join(bins) + '); per animal the mean of its sessions in a bin (grey dots), then the mean +- SEM ACROSS ANIMALS (black; n = animals with a session in the bin, which differs between bins).' if view == 'day' else 'Per animal the mean of its sessions in each epoch (grey, lines join an animal), then the mean +- SEM ACROSS ANIMALS (black). Statistics: the change vs pre by `analysis_kit.boot_delta` -- paired animals -> sessions bootstrap, animals weighted equally, 95 % interval; * = interval excludes 0. k/n v = number of animals whose epoch mean is below their pre level (with n = 4 a result not seen in 3+ animals individually will not survive; CLAUDE.md rule 8). Per-animal values and intervals: data/peak_epoch_stats.csv.'}
Set {set_name}; frozen basis, absolute expression (weight x FR normaliser D).""")


if __name__ == "__main__":
    raise SystemExit(main())

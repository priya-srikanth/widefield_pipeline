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
pre), peak_by_day[_side].png (per factor: peak / pre over days since stroke, one line per animal, grey = the pre range).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EPOCHS = ("pre", "acute", "subacute", "chronic")
SIDE = {"far_R": "contra", "close_R": "contra", "far_center": "centre", "close_center": "centre",
        "far_L": "ipsi", "close_L": "ipsi"}           # lesions LEFT: the animal's right spouts are contralesional


def peaks(fdir: Path, split: str, window: tuple[float, float]) -> pd.DataFrame:
    from wfield_local import face_rhythm_readouts as frr
    rows = []
    for t in frr._tasks(fdir, "proj").itertuples():
        with np.load(t.file) as z:
            X = z["trials"].astype(float) * z["trials_D"][..., None]          # (trials, 151, R)
            tms, pos = z["t_ms"] / 1000.0, z["position"].astype(str)
        groups = {"all": np.ones(len(pos), bool)} if split == "none" else \
            {s: np.array([SIDE.get(p) == s for p in pos]) for s in ("contra", "centre", "ipsi")}
        w = (tms >= window[0]) & (tms <= window[1])
        for g, m in groups.items():
            if m.sum() < 3:
                continue
            tr = np.nanmean(X[m], 0)                                          # (151, R) trial average
            sub = tr[w]
            k = np.nanargmax(sub, 0)
            rows += [{"label": t.label, "key": t.key, "animal": t.animal, "epoch": t.epoch, "group": g,
                      "factor": j, "peak": float(sub[k[j], j]), "t_peak_s": float(tms[w][k[j]]),
                      "n_trials": int(m.sum())} for j in range(X.shape[-1])]
    S = pd.DataFrame(rows)
    # one row per session (a split video, e.g. PS92_0812 v1 / v2, counts once: trial-weighted mean)
    S = (S.assign(wp=S.peak * S.n_trials).groupby(["label", "animal", "epoch", "group", "factor"], as_index=False)
         .agg(wp=("wp", "sum"), n_trials=("n_trials", "sum"), t_peak_s=("t_peak_s", "mean")))
    S["peak"] = S.wp / S.n_trials
    S = S.drop(columns="wp")
    S["day"] = [frr._day(x) for x in S.label]
    pre = S[S.epoch == "pre"].groupby(["animal", "group", "factor"]).peak
    S = S.join(pre.mean().rename("pre_mean"), on=["animal", "group", "factor"]) \
         .join(pre.min().rename("pre_min"), on=["animal", "group", "factor"]) \
         .join(pre.max().rename("pre_max"), on=["animal", "group", "factor"])
    for c in ("peak", "pre_min", "pre_max"):
        S[f"{c}_vs_pre"] = S[c] / S.pre_mean
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
    ap.add_argument("--split", default="none", choices=("none", "side"))
    ap.add_argument("--window", type=float, nargs=2, default=(0.0, 4.0), help="peak window, s from cue")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "peak"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n.removeprefix("fr_") for n in fro.labels(fdir)]
    sfx = "" if a.split == "none" else "_side"
    S = peaks(fdir, a.split, tuple(a.window))
    S["name"] = [names[k] for k in S.factor]
    S.to_csv(figure_layout.sidecar_for(out, f"peak_sessions{sfx}", ".csv"), index=False, float_format="%.5g")
    E = S.groupby(["group", "factor", "name", "animal", "epoch"], observed=True)["peak_vs_pre"].mean().unstack("epoch") \
        .reindex(columns=list(EPOCHS))
    E.to_csv(figure_layout.sidecar_for(out, f"peak_epochs{sfx}", ".csv"), float_format="%.4g")
    pd.set_option("display.width", 220)
    print("PEAK amplitude / pre, per animal x epoch (rule 8: per animal)")
    print(E.round(2).to_string())

    groups = sorted(S.group.unique(), key=lambda g: ("contra", "centre", "ipsi", "all").index(g))
    animals = sorted(S.animal.unique())
    col = dict(zip(animals, ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")))
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
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

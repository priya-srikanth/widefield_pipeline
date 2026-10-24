"""Is the acute NON-TASK face-movement drop just "less active overall"? (Priya 10-24, FR handoff open item (c); FINDINGS
6d caveat.)

6d found between-trial (non-task) factor expression falls acutely as much as task expression. That could be (i) less
of each behaviour per unit time IN a given state, or (ii) the same behaviour per state but more time spent at rest /
less running. Here every between-trial session bin (> 3 s before / > 6 s after any engaged cue, the 6d definition, large
movements kept) is classed by the DAQ behaviour events (`behavior_events`): RUNNING (running bout), REST (the
`rest_*` mask: slow treadmill, away from licks / rewards), GROOMING, or OTHER. Per session x state x factor: mean
absolute expression (weight x FR normaliser D, `proj/`); per session: the fraction of between-trial time in each state.

Per animal x epoch (session means first): log2 change vs the animal's pre sessions (pre averages 0); `boot_delta`
(paired animals -> sessions) per epoch x state x factor; per-animal tables first (rule 8). If the drop holds INSIDE
rest and inside running, it is not explained by state occupancy.

    python scripts/face_rhythm_nontask_state.py [--set master_pre_v1] [--jobs 4] [--plot-only]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/nontask_state/``: nontask_state.png (per factor x state, epoch log2 change,
dots = animals), nontask_occupancy.png (time in each state by epoch), data/nontask_state_sessions.csv (cache;
``--plot-only`` redraws from it), nontask_state_stats.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EPOCHS = ("pre", "acute", "subacute", "chronic")
STATES = ("all", "rest", "running", "grooming", "other")
BUSY_S = (-3.0, 6.0)                      # = face_rhythm_readouts.BUSY_S (the 6d "between trials" definition)


def _in_bouts(t: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Boolean per time ``t`` (s): inside any [start, stop) bout (both s)."""
    if not len(starts):
        return np.zeros(len(t), bool)
    o = np.argsort(starts)
    s, e = np.asarray(starts, float)[o], np.asarray(stops, float)[o]
    k = np.searchsorted(s, t, side="right") - 1
    return (k >= 0) & (t < e[np.clip(k, 0, None)])


def job(item: tuple) -> list[dict]:
    fdir, key = item
    from wfield_local import behavior_events as be
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local.paths import PathResolver
    with np.load(Path(fdir) / "proj" / f"{key}.npz", allow_pickle=True) as z:
        animal, date, epoch = str(z["animal"]), str(z["date"]), str(z["epoch"])
        ts, W = z["session_t_daq_s"].astype(float), z["session"].astype(float) * z["session_D"][:, None]
        cues = np.sort(z["cue_daq_s"].astype(float))
    E = be.load_events(be.events_path(PathResolver(), animal, date))
    if E is None:
        return []
    fs = float(E["fs"])
    k = np.searchsorted(cues, ts)
    near = np.zeros(len(ts), bool)
    for kk in (k - 1, k):
        ok = (kk >= 0) & (kk < len(cues))
        d = ts - cues[np.clip(kk, 0, len(cues) - 1)]
        near |= ok & (d >= BUSY_S[0]) & (d <= BUSY_S[1])
    between = ~near & np.isfinite(W).all(1)
    run = _in_bouts(ts, E["running_starts"] / fs, E["running_stops"] / fs)
    rest = _in_bouts(ts, E["rest_starts"] / fs, E["rest_stops"] / fs) & ~run
    groom = _in_bouts(ts, E["grooming_starts"] / fs, E["grooming_stops"] / fs) & ~run & ~rest
    masks = {"all": between, "rest": between & rest, "running": between & run, "grooming": between & groom,
             "other": between & ~run & ~rest & ~groom}
    rows = []
    for st, m in masks.items():
        r = {"key": key, "label": frr._label(animal, date), "animal": animal, "epoch": epoch, "state": st,
             "n_bins": int(m.sum()), "frac": float(m.sum() / max(between.sum(), 1))}
        if m.sum():
            r.update({f"f{j}": float(v) for j, v in enumerate(W[m].mean(0))})
        rows.append(r)
    return rows


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import analysis_kit as ak
    from wfield_local import config
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import figure_layout, writeguard
    from wfield_local.face_rhythm import common as frc
    ecol = frc.epoch_colors()        # FR epoch palette (orofacial / 3-D kinematics)
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--min-bins", type=int, default=50, help="session x state bins, else that cell is skipped")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--plot-only", action="store_true", help="redraw from data/nontask_state_sessions.csv")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = a.fdir or fro.factor_dir(rv, a.bundle, a.set)
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "nontask_state"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = figure_layout.sidecar_for(out, "nontask_state_sessions", ".csv")
    if a.plot_only:
        S = pd.read_csv(cache)
    else:
        keys = sorted(f.stem for f in (fdir / "proj").glob("*.npz") if not f.name.endswith(".part.npz"))
        got, fail = ak.fan_sessions([(str(fdir), k) for k in keys], job, jobs=a.jobs, key=lambda it: it[1],
                                    label="nontask")
        if fail:
            print(f"FAILED {len(fail)}: {fail}")
        S = pd.DataFrame([r for _, rs in got for r in rs])
        S.to_csv(cache, index=False)
    names = [n.removeprefix("fr_") for n in fro.labels(fdir)]
    fcols = sorted((c for c in S.columns if c.startswith("f") and c[1:].isdigit()), key=lambda c: int(c[1:]))
    lick, other = frr.lick_split(names)
    S["lick"] = S[[fcols[names.index(n)] for n in lick]].sum(1, min_count=len(lick))
    S["nontask"] = S[[fcols[names.index(n)] for n in other]].sum(1, min_count=len(other))
    # a split video's two halves: bin-weighted mean per label
    val = fcols + ["lick", "nontask"]
    S = S[S.n_bins > 0]
    g = S.assign(**{c: S[c] * S.n_bins for c in val}).groupby(["label", "animal", "epoch", "state"], as_index=False)
    S = g[val + ["n_bins"]].sum()
    for c in val:
        S[c] = S[c] / S.n_bins
    tot = S[S.state == "all"].set_index("label").n_bins
    S["frac"] = S.n_bins / S.label.map(tot)

    # occupancy per animal x epoch (rule 8 first)
    occ = S[S.state != "all"].groupby(["animal", "epoch", "state"]).frac.mean().unstack("state")
    print("time in each state between trials (fraction; session means), per animal x epoch")
    print(occ.reindex(columns=[s for s in STATES if s != "all"]).round(3).to_string())

    V = S[S.n_bins >= a.min_bins].copy()
    L = []
    for c in val:
        lg = np.log2(np.maximum(V[c], 1e-12))
        pre = lg.where(V.epoch == "pre").groupby([V.animal, V.state]).transform("mean")
        L.append(V[["label", "animal", "epoch", "state"]].assign(quantity=c, log2_vs_pre=lg - pre))
    L = pd.concat(L, ignore_index=True)
    rng = np.random.default_rng(0)
    rows = []
    for (q, st), d in L.groupby(["quantity", "state"], sort=False):
        pre_d = d[d.epoch == "pre"]
        for ep in EPOCHS[1:]:
            post = d[d.epoch == ep]
            if post.animal.nunique() < 3:
                continue
            per = post.groupby("animal").log2_vs_pre.mean()
            rows.append({"quantity": q, "state": st, "epoch": ep, **{an: per.get(an, np.nan) for an in sorted(L.animal.unique())},
                         "n_down": int((per < 0).sum()), "n_animals": len(per),
                         **_boot(post, pre_d, rng, a.n_boot)})
    T = pd.DataFrame(rows)
    T.to_csv(out / "nontask_state_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nlog2 change vs pre (per animal, then boot_delta), lick vs non-task sums by state")
    print(T[T.quantity.isin(["lick", "nontask"])].round(3).to_string(index=False))

    # figure 1: per quantity (lick, nontask, each factor) x state: per-animal epoch dots + boot interval
    qs = ["lick", "nontask"] + fcols
    states = [s for s in STATES if s != "other"]
    fig, axs = frf.subplots(len(qs), len(states), figsize=(2.6 * len(states), 1.8 * len(qs)), sharey="row",
                            squeeze=False)
    for r_i, q in enumerate(qs):
        for c_i, st in enumerate(states):
            ax = axs[r_i, c_i]
            d = L[(L.quantity == q) & (L.state == st)]
            for e, ep in enumerate(EPOCHS):
                per = d[d.epoch == ep].groupby("animal").log2_vs_pre.mean()
                for an, v in per.items():
                    ax.plot(e + rng.uniform(-0.12, 0.12), v, "o", ms=3, color=config.animal_color()[an])
                t = T[(T.quantity == q) & (T.state == st) & (T.epoch == ep)]
                if len(t):
                    ax.errorbar(e, t.point.iloc[0], yerr=[[t.point.iloc[0] - t.lo.iloc[0]], [t.hi.iloc[0] - t.point.iloc[0]]],
                                fmt="s", color=ecol[ep], mec="k", ms=5, capsize=2)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(len(EPOCHS)), EPOCHS, fontsize=7)
            if r_i == 0:
                ax.set_title(st)
            if c_i == 0:
                ax.set_ylabel(q if q in ("lick", "nontask") else names[int(q[1:])], fontsize=7)
    fig.suptitle(f"{fdir.name}: between-trial factor expression by behavioural state, log2 vs pre")
    frf.save(fig, out / "nontask_state.png", f"""Between-trial (cue -3..+6 s excluded) absolute factor expression per
behavioural state (DAQ behaviour events: running bouts, rest mask, grooming), log2 vs the animal's pre sessions. Dots =
animals (repo colours), squares = `boot_delta` point estimate and 95 % interval (paired animals -> sessions; epoch
colours). Cells with < {a.min_bins} bins per session skipped. lick / nontask = summed lick / other factors (fr_groups
lick split, as 6d).""")

    fig, axs = frf.subplots(1, 4, figsize=(12, 3), sharey=True, squeeze=False)
    for c_i, st in enumerate(["rest", "running", "grooming", "other"]):
        ax = axs[0, c_i]
        d = S[S.state == st]
        for e, ep in enumerate(EPOCHS):
            per = d[d.epoch == ep].groupby("animal").frac.mean()
            for an, v in per.items():
                ax.plot(e + rng.uniform(-0.12, 0.12), v, "o", ms=4, color=config.animal_color()[an], label=an if e == 0 else None)
            if len(per):
                ax.bar(e, per.mean(), color=ecol[ep], alpha=0.6)
        ax.set_xticks(range(len(EPOCHS)), EPOCHS, fontsize=8)
        ax.set_title(st)
    axs[0, 0].set_ylabel("fraction of between-trial time")
    axs[0, -1].legend(fontsize=7, frameon=False)
    frf.save(fig, out / "nontask_occupancy.png", """Fraction of between-trial time in each behavioural state, per
animal (dots) and mean of animals (bars, FR epoch colours).""")
    print(f"-> {out}")
    return 0


def _boot(post: pd.DataFrame, pre: pd.DataFrame, rng, n_boot: int) -> dict:
    """`analysis_kit.boot_delta` on {animal: session values} (sorted, rule 9), log2 scale."""
    from wfield_local import analysis_kit as ak
    as_dict = lambda d: {an: g.sort_values("label").log2_vs_pre.to_numpy() for an, g in d.groupby("animal")}  # noqa: E731
    iv = ak.boot_delta(as_dict(post), as_dict(pre), rng, n_boot)
    return {"point": iv.point, "lo": iv.lo, "hi": iv.hi} if iv else {"point": np.nan, "lo": np.nan, "hi": np.nan}


if __name__ == "__main__":
    raise SystemExit(main())

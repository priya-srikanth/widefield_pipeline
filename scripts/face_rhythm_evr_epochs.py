"""Does the pre-stroke vocabulary fit post-stroke sessions worse? Epoch-pooled, each animal against its own pre-stroke
sessions, with statistics across animals (Priya 10-24).

Per session (`readouts/<set>/data/session_table.csv`, from `face_rhythm_readouts.py`): RATIO = EVR of the frozen
pre-stroke factors / EVR of the session's own rank-R fit (own fits from ``own8/``; pre sessions' frozen EVR on their
held-out leave-one-day-out basis). The ratio divides out what lowers BOTH fits (less movement against a fixed tracking
noise floor, less stereotyped movement). Normalised to each animal's mean pre-stroke ratio (1 = that animal's pre
level); per animal the mean of its sessions in each epoch; mean +- SEM across animals; the change vs pre by
`analysis_kit.boot_delta` (paired animals -> sessions bootstrap, animal-weighted, 95 % interval). Same for the EVR GAP
(own - frozen) and the frozen-basis reconstruction error, for comparison.

    python scripts/face_rhythm_evr_epochs.py [--set master_pre_v1] [--plot-only]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/evr_epochs.png`` + ``data/evr_epoch_stats.csv``. ``--plot-only`` redraws the
figure from the STORED data/evr_epoch_stats.csv (no bootstrap, no CSV written); the session markers are the per-session
normalised values from data/session_table.csv. Epochs are categories (Priya 11-03): no lines between them, summary =
mean +- SEM over animals, faded OPEN markers = sessions (shape = animal, colour = epoch), * in the top band.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402

EPOCHS = frc.EPOCHS
METRICS = {"ratio": ("EVR frozen / EVR own, / own pre", "ratio"),
           "evr_gap": ("EVR gap (own - frozen), - own pre", "diff"),
           "recon_err_session": ("reconstruction error (frozen), / own pre", "ratio")}


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
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true", help="redraw from the stored data/evr_epoch_stats.csv")
    a = ap.parse_args(argv)
    rv = PathResolver()
    out = a.out or Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set
    writeguard.assert_writable(out)
    S = pd.read_csv(figure_layout.sidecar_for(out, "session_table", ".csv", mkdir=False))
    S = S[S.epoch.isin(EPOCHS)].copy()
    S["ratio"] = S.evr_frozen / S.evr_own
    rng = np.random.default_rng(0)
    rows = []
    for m, (_, kind) in METRICS.items():
        pre_mean = S[S.epoch == "pre"].groupby("animal")[m].mean()
        S[f"{m}_n"] = S[m] / S.animal.map(pre_mean) if kind == "ratio" else S[m] - S.animal.map(pre_mean)
        base = 1.0 if kind == "ratio" else 0.0
        if a.plot_only:
            continue
        pre = {an: list(g[f"{m}_n"]) for an, g in S[S.epoch == "pre"].groupby("animal")}
        for ep in EPOCHS[1:]:
            post = {an: list(g[f"{m}_n"]) for an, g in S[S.epoch == ep].groupby("animal")}
            per = {an: float(np.mean(v)) for an, v in sorted(post.items())}
            iv = ak.boot_delta(post, pre, rng)
            rows.append({"metric": m, "epoch": ep, **per,
                         "mean": float(np.mean(list(per.values()))),
                         "sem": float(np.std(list(per.values()), ddof=1) / np.sqrt(len(per))),
                         "delta": iv.point if iv else np.nan, "ci_lo": iv.lo if iv else np.nan,
                         "ci_hi": iv.hi if iv else np.nan, "n_animals": iv.n_animals if iv else len(per),
                         "excludes_0": bool(iv and (iv.lo > 0 or iv.hi < 0)),
                         "n_worse": sum((v < base) if m == "ratio" else (v > base) for v in per.values())})
    stats_f = figure_layout.sidecar_for(out, "evr_epoch_stats", ".csv", mkdir=False)
    if a.plot_only:
        T = pd.read_csv(stats_f)
    else:
        T = pd.DataFrame(rows)
        T.to_csv(stats_f, index=False, float_format="%.4g")
    animals = sorted(S.animal.unique())
    pd.set_option("display.width", 200)
    print("PER-ANIMAL epoch means FIRST (rule 8), then the paired bootstrap of the change vs pre "
          "(n_worse = animals whose epoch mean is worse than their pre level)")
    print(T[["metric", "epoch", *animals, "delta", "ci_lo", "ci_hi", "excludes_0", "n_worse"]].round(4).to_string(index=False))

    from wfield_local import cat_points as cp
    from wfield_local import sig_marks
    ecol = frc.epoch_colors()
    x = np.arange(len(EPOCHS))
    with frf.style():
        fig, axs = frf.subplots(1, len(METRICS), figsize=(4.6 * len(METRICS) + 1.4, 4.6), squeeze=False)
        for ax, (m, (lab, kind)) in zip(axs[0], METRICS.items()):
            A = S.groupby(["animal", "epoch"])[f"{m}_n"].mean().unstack().reindex(columns=list(EPOCHS))
            c0 = 1.0 if kind == "ratio" else 0.0
            tt = T[T.metric == m].set_index("epoch")
            for e_i, ep in enumerate(EPOCHS):
                se_ = S[S.epoch == ep]
                cp.session_dots(ax, e_i, se_[f"{m}_n"].to_numpy(), se_.animal.to_numpy(), ecol[ep], jitter=0.12,
                                seed=e_i)
                if ep in tt.index:                       # stored statistics (per-animal means -> mean +- SEM)
                    mu, se = tt.loc[ep, "mean"], tt.loc[ep, "sem"]
                    if bool(tt.loc[ep, "excludes_0"]):
                        sig_marks.star_top(ax, e_i, "*", color=ecol[ep])
                else:                                    # pre: the animals' own pre level (c0 by construction)
                    v = A[ep].dropna()
                    mu, se = float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None)
                cp.summary_point(ax, e_i, mu, se, ecol[ep], ms=6.5)
            ax.axhline(c0, color="0.6", lw=0.6, zorder=0)
            pr = S[S.epoch == "pre"].groupby("animal")[f"{m}_n"]
            sem = float((pr.std(ddof=1) / np.sqrt(pr.count())).mean())
            ax.axhspan(c0 - sem, c0 + sem, color="0.9", lw=0, zorder=0)
            labs = ["pre"]
            for ep in EPOCHS[1:]:
                r = tt.loc[ep]
                labs.append(f"{ep[:5]}\n{r.delta:+.3f}\n[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]\n"
                            f"{int(r.n_worse)}/{int(r.n_animals)} worse")
            ax.set_xticks(x, labs, fontsize=7.5)
            ax.set_xlim(-0.5, len(EPOCHS) - 0.5)
            ax.set_title(lab, fontsize=10)
        lg = axs[0, -1]
        for ep in EPOCHS:
            lg.plot([], [], "o", color=ecol[ep], mec="w", ms=6.5, label=f"{ep}: mean +- SEM")
        cp.add_animal_entries(lg, S.animal.unique())
        frf.legend_outside(lg, fontsize=8)
        sig_marks.reserve_headroom(fig)
        fig.suptitle(f"{a.set}: fit of the frozen pre-stroke factors, each animal vs its own pre-stroke sessions")
        frf.save(fig, out / "evr_epochs.png", """
Per session (`data/session_table.csv`): LEFT the ratio EVR(frozen pre-stroke factors) / EVR(the session's own rank-R
fit), divided by the animal's mean pre-stroke ratio (1 = its pre level; the ratio divides out what lowers both fits);
MIDDLE the EVR gap (own - frozen) minus the animal's pre-stroke mean; RIGHT the frozen basis' median relative
reconstruction error / the animal's pre mean. Pre sessions are scored on their held-out (leave-one-day-out) basis.
Each epoch is its own category (Priya 11-03: no lines between epochs): filled marker = mean +- SEM across the four
animals' epoch means (epoch colours); faded OPEN markers = the individual SESSIONS (shape = animal, colour = epoch); grey band: +-1 SEM of the pre-stroke sessions around the pre level (per animal, then averaged over animals).
Numbers: change vs pre by `analysis_kit.boot_delta` (paired animals -> sessions bootstrap, animals weighted equally,
95 % interval; * in the top band = excludes 0) and how many animals' epoch means are worse than their pre level. n = 4
animals: read the per-animal values first (CLAUDE.md rule 8). Per-animal values: data/evr_epoch_stats.csv.""")
    print("->", out / "evr_epochs.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

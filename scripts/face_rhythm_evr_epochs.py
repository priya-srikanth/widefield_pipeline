"""Does the pre-stroke vocabulary fit post-stroke sessions worse? Epoch-pooled, each animal against its own pre-stroke
sessions, with statistics across animals (Priya 10-24).

Per session (`readouts/<set>/data/session_table.csv`, from `face_rhythm_readouts.py`): RATIO = EVR of the frozen
pre-stroke factors / EVR of the session's own rank-R fit (own fits from ``own8/``; pre sessions' frozen EVR on their
held-out leave-one-day-out basis). The ratio divides out what lowers BOTH fits (less movement against a fixed tracking
noise floor, less stereotyped movement). Normalised to each animal's mean pre-stroke ratio (1 = that animal's pre
level); per animal the mean of its sessions in each epoch; mean +- SEM across animals; the change vs pre by
`analysis_kit.boot_delta` (paired animals -> sessions bootstrap, animal-weighted, 95 % interval). Same for the EVR GAP
(own - frozen) and the frozen-basis reconstruction error, for comparison.

    python scripts/face_rhythm_evr_epochs.py [--set master_pre_v1]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/evr_epochs.png`` + ``data/evr_epoch_stats.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EPOCHS = ("pre", "acute", "subacute", "chronic")
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
    T = pd.DataFrame(rows)
    T.to_csv(figure_layout.sidecar_for(out, "evr_epoch_stats", ".csv"), index=False, float_format="%.4g")
    animals = sorted(S.animal.unique())
    pd.set_option("display.width", 200)
    print("PER-ANIMAL epoch means FIRST (rule 8), then the paired bootstrap of the change vs pre "
          "(n_worse = animals whose epoch mean is worse than their pre level)")
    print(T[["metric", "epoch", *animals, "delta", "ci_lo", "ci_hi", "excludes_0", "n_worse"]].round(4).to_string(index=False))

    from wfield_local import config
    acol = config.animal_color()                         # the repo's one animal palette (configs/animals.yaml)
    x = np.arange(len(EPOCHS))
    with frf.style():
        fig, axs = frf.subplots(1, len(METRICS), figsize=(4.6 * len(METRICS), 4.4), squeeze=False)
        for ax, (m, (lab, kind)) in zip(axs[0], METRICS.items()):
            A = S.groupby(["animal", "epoch"])[f"{m}_n"].mean().unstack().reindex(columns=list(EPOCHS))
            for an, r in A.iterrows():
                ax.plot(x, r.to_numpy(), "o-", color=acol.get(an, "0.6"), ms=3, lw=0.9, alpha=0.8)
                ax.text(x[-1] + 0.08, r.iloc[-1], an, fontsize=7, va="center", color=acol.get(an, "0.4"))
            ax.errorbar(x, A.mean(0).to_numpy(), yerr=(A.std(0, ddof=1) / np.sqrt(A.notna().sum(0))).to_numpy(),
                        fmt="o", color="k", ms=6, capsize=3, zorder=5)
            ax.axhline(1.0 if kind == "ratio" else 0.0, color="k", lw=0.5)
            pr = S[S.epoch == "pre"].groupby("animal")[f"{m}_n"]
            sem = float((pr.std(ddof=1) / np.sqrt(pr.count())).mean())
            c0 = 1.0 if kind == "ratio" else 0.0
            ax.axhspan(c0 - sem, c0 + sem, color="0.85", lw=0, zorder=0)
            tt = T[T.metric == m].set_index("epoch")
            for e_i, ep in enumerate(EPOCHS[1:], start=1):
                r = tt.loc[ep]
                ax.annotate(f"{r.delta:+.3f}{'*' if r.excludes_0 else ''}\n[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]\n"
                            f"{int(r.n_worse)}/{int(r.n_animals)} worse", (e_i, 0.02), xycoords=("data", "axes fraction"),
                            ha="center", fontsize=7)
            ax.set_xticks(x, [e[:5] for e in EPOCHS])
            ax.set_title(lab, fontsize=10)
        fig.suptitle(f"{a.set}: fit of the frozen pre-stroke factors, each animal vs its own pre-stroke sessions")
        frf.save(fig, out / "evr_epochs.png", """
Per session (`data/session_table.csv`): LEFT the ratio EVR(frozen pre-stroke factors) / EVR(the session's own rank-R
fit), divided by the animal's mean pre-stroke ratio (1 = its pre level; the ratio divides out what lowers both fits);
MIDDLE the EVR gap (own - frozen) minus the animal's pre-stroke mean; RIGHT the frozen basis' median relative
reconstruction error / the animal's pre mean. Pre sessions are scored on their held-out (leave-one-day-out) basis.
Coloured lines: per-animal epoch means (configs/animals.yaml colours) (sessions pooled within the epoch); black: mean +- SEM across the four animals;
grey band: +-1 SEM of the pre-stroke sessions around the pre level (per animal, then averaged over animals).
Numbers: change vs pre by `analysis_kit.boot_delta` (paired animals -> sessions bootstrap, animals weighted equally,
95 % interval; * = excludes 0) and how many animals' epoch means are worse than their pre level. n = 4 animals: read
the per-animal lines first (CLAUDE.md rule 8). Per-animal values: data/evr_epoch_stats.csv.""")
    print("->", out / "evr_epochs.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

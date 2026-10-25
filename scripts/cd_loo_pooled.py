"""POOLED, within-animal normalised versions of the CD leave-one-out figures (`scripts/cd_loo_figures.py`; imaging page
Figs 9c-e). Priya 10-25: "can we have a within-animal normalized pooled version of 9c-e".

Same inputs as `cd_loo_figures` (<labcams>/grant_figures/epoch/data/cd_window_projection_loo_sessions_long.csv and
cd_loo_traces.npz): each position's projection on its OWN coding direction, pole-normalised per animal (0 = pre-stroke
not-P, 1 = pre-stroke lick at P), August pre sessions scored leave-one-out. Here each animal is further divided by ITS
held-out pre mean for that position and alignment (window mean), so pre = 1 for every animal and the pooled figure is an
average of animals, not of sessions (the maps' convention, `beta_maps.within_animal_pooled`).

  cd_loo_pooled_window_<align>.png  per position: per-animal epoch means (animal colours, thin), mean of animals +- SEM
                                    (black); under each post epoch the change vs pre by `analysis_kit.boot_delta`
                                    (paired animals -> sessions, animal-weighted, 95 % interval; * = excludes 0) and
                                    k / n animals below pre
  cd_loo_pooled_traj_<align>.png    per position and epoch: each animal's session-mean trace / its pre window mean, then
                                    mean +- SEM over ANIMALS (n animals in the legend)
  cd_loo_pooled_stats.csv           every plotted change
Caveats: pre = August only (~10-day span; the June -> August drift of these projections is not measured here); PS95
acute = 1 session; haemodynamic caveat of `cd_trajectories` (read amplitude and gross time course, not onset timing).

    python -m scripts.cd_loo_pooled [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cd_loo_figures import ALIGN_LABEL, EP_COL, POS

EPOCHS = ("pre", "acute", "subacute", "chronic")


def normalised(C: pd.DataFrame) -> pd.DataFrame:
    """Add ``y`` = value / that animal's held-out pre mean (same align, position)."""
    pre = C[C.epoch == "pre"].groupby(["align", "animal", "position"]).value.mean().rename("pre_mean")
    D = C.join(pre, on=["align", "animal", "position"])
    return D.assign(y=D.value / D.pre_mean)


def changes(D: pd.DataFrame, align: str, rng) -> list[dict]:
    from wfield_local import analysis_kit as ak
    rows = []
    for p in POS:
        g = D[(D["align"] == align) & (D.position == p)]
        pre = {a: gg.y.to_numpy() for a, gg in g[g.epoch == "pre"].groupby("animal")}
        for e in EPOCHS[1:]:
            post = {a: gg.y.to_numpy() for a, gg in g[g.epoch == e].groupby("animal")}
            iv = ak.boot_delta(post, pre, rng)
            an = sorted(set(pre) & set(post))
            below = sum(np.mean(post[a]) < np.mean(pre[a]) for a in an)
            pt, lo, hi = ((iv.point, iv.lo, iv.hi) if iv is not None else (np.nan, np.nan, np.nan))
            rows.append(dict(align=align, position=p, epoch=e, change=pt, lo=lo, hi=hi,
                             mark="*" if iv is not None and (lo > 0 or hi < 0) else "", n_animals=len(an),
                             below_pre=below, per_animal=";".join(f"{a}:{np.mean(post[a]):.3f}" for a in an)))
    return rows


def fig_window(D: pd.DataFrame, stats: list[dict], align: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config
    col = config.animal_color()
    fig, axes = plt.subplots(1, len(POS), figsize=(2.9 * len(POS), 3.4), sharey=True)
    for ax, p in zip(axes, POS):
        g = D[(D["align"] == align) & (D.position == p)]
        M = g.groupby(["animal", "epoch"]).y.mean().unstack().reindex(columns=list(EPOCHS))
        for a, r in M.iterrows():
            ax.plot(range(4), r.values, "-o", color=col.get(a, "0.5"), lw=0.9, ms=3.5, alpha=0.85, label=a)
        mu, se = M.mean(), M.std(ddof=1) / np.sqrt(M.notna().sum().clip(lower=1))
        ax.errorbar(range(4), mu.values, yerr=se.values, color="k", lw=2.2, marker="s", ms=5, capsize=3,
                    label="mean of animals")
        ax.axhline(1, color="0.6", ls="--", lw=0.8)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(EPOCHS, fontsize=7)
        ax.set_title(p + (" (contra)" if p.endswith("_R") else ""), fontsize=9)
        ticks = []
        for e in EPOCHS[1:]:
            s = next(x for x in stats if x["position"] == p and x["epoch"] == e)
            ticks.append(f"{e[:3]} {s['change']:+.2f}{s['mark']} {s['below_pre']}/{s['n_animals']}")
        ax.text(0.02, 0.02, "\n".join(ticks), transform=ax.transAxes, fontsize=6.3, va="bottom", family="monospace")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("CD projection / own held-out pre\n(pre = 1)", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle(f"CD leave-one-out window means, {align}-aligned -- POOLED within animal (each animal / its own pre). "
                 "Text: change vs pre (paired animals -> sessions bootstrap; * = 95 % interval excludes 0), "
                 "animals below pre / animals", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_traj(Z, D: pd.DataFrame, align: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.epoch_sessions_long import CD_WINDOWS
    if f"trace_{align}" not in Z:
        return
    t, X = Z[f"t_{align}"], Z[f"trace_{align}"].astype(float)
    lab, ep, pos = Z[f"label_{align}"].astype(str), Z[f"epoch_{align}"].astype(str), Z[f"position_{align}"].astype(str)
    an = np.array([x[:4] for x in lab])
    pre_mean = D[(D["align"] == align) & (D.epoch == "pre")].groupby(["animal", "position"]).value.mean()
    _, lo, hi = CD_WINDOWS[align]
    fig, axes = plt.subplots(1, len(POS), figsize=(2.9 * len(POS), 3.2), sharey=True, sharex=True)
    for ax, p in zip(axes, POS):
        ax.axvspan(lo, hi, color="0.93", zorder=0)
        for e in EPOCHS:
            per = []
            for a in sorted(set(an)):
                m = (an == a) & (pos == p) & (ep == e)
                if m.sum() < 1 or (a, p) not in pre_mean.index:
                    continue
                with np.errstate(all="ignore"):
                    per.append(np.nanmean(X[m], 0) / pre_mean[(a, p)])
            if len(per) < 2:
                continue
            P = np.vstack(per)
            mu = np.nanmean(P, 0)
            se = np.nanstd(P, 0, ddof=1) / np.sqrt(len(per))
            ax.plot(t, mu, color=EP_COL[e], lw=1.6 if e == "pre" else 1.2, label=f"{e} ({len(per)} an)")
            ax.fill_between(t, mu - se, mu + se, color=EP_COL[e], alpha=0.18, lw=0)
        ax.axhline(1, color="0.6", ls="--", lw=0.7)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(p + (" (contra)" if p.endswith("_R") else ""), fontsize=9)
        ax.set_xlabel(ALIGN_LABEL.get(align, "s"), fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("CD projection / own held-out pre\n(window mean = 1)", fontsize=8)
    axes[-1].legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle(f"CD leave-one-out trajectories, {align}-aligned -- POOLED within animal: each animal's session-mean "
                 "trace / its pre window mean, then mean +- SEM over ANIMALS; shaded = window-mean window", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import cd_trajectories as cdt
    from wfield_local import figure_layout as FL
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    fig_dir = Path(cdt.default_out())
    C = pd.read_csv(FL.sidecar_for(fig_dir, "cd_window_projection_loo", "_sessions_long.csv", mkdir=False))
    Z = np.load(FL.sidecar_for(fig_dir, "cd_loo_traces", ".npz", mkdir=False))
    out = a.out or fig_dir / "cd_loo"
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    D = normalised(C)
    rng = np.random.default_rng(0)
    stats = []
    for align in ("precue", "cue", "lick"):
        s = changes(D, align, rng)
        stats += s
        fig_window(D, s, align, out / f"cd_loo_pooled_window_{align}.png")
        fig_traj(Z, D, align, out / f"cd_loo_pooled_traj_{align}.png")
        print("->", out / f"cd_loo_pooled_window_{align}.png", "+ traj")
    S = pd.DataFrame(stats)
    S.to_csv(out / "cd_loo_pooled_stats.csv", index=False, float_format="%.4g")
    print(S[S.position == "far_R"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PER-SPOUT activation maps across epochs (Priya 10-26: "map the decoder weights per position across epochs").

For each spout, a ONE-VS-REST ridge decoder (indicator of that spout from the activity of cued licks; features =
the animal's pre-stroke SVD basis, lags 0..0.2 s, as `iti_decoder_patterns.py`) is fit per epoch and turned into its
Haufe activation pattern (Cov(activity, decoded) / Var), mapped to pixels: where activity distinguishes that spout's
licks from the other five's. Per animal, each map / the RMS of that animal's pre map for the spout; mean of animals.
Similarity to pre per animal x spout x epoch = pixel correlation corrected by split-half reliability (sessions
alternating by date), as for the position maps.
Outputs <E>/iti_licks/spout_patterns.csv, figures/lick_cued_spout_patterns.png (rows = spouts in canonical order, columns =
epochs + chronic - pre).
Figure data (figdata.py convention): <E>/iti_licks/figdata/lick_cued_spout_patterns.pkl (mean-of-animals map, n and
similarity per spout x column + the imaging grid); `--plot-only` redraws from that file alone (no decoder fit).
Usage: python scripts/iti_spout_patterns.py [--plot-only]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_where_decoder as wd  # noqa: E402
from figdata import grid_fns, imaging_grid, load_fd, plot_only, save_fd  # noqa: E402
from iti_decoder_patterns import SPATIAL, pattern  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
COLS = [*EP, "chronic - pre"]
FIG = "lick_cued_spout_patterns"
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]


def compute():
    from wfield_local import orofacial_cohort as oc
    D = wd.load("svdbasis")
    rows, maps = [], {}
    for a in AN:
        W = np.asarray(pickle.loads((SPATIAL / f"spatial_{a}.pkl").read_bytes())["P"], np.float64)
        S = {}
        for x, d in sorted(D.items()):
            if d["animal"] != a:
                continue
            X, _C = d["trial"]
            P = np.array([p if p in POS or "_" not in p else oc.pos_key(p, "L") for p in d["trial_pos"]])
            ok = np.isin(P, POS)
            if ok.sum() >= 30:
                S[x] = (d["epoch"], X[ok], P[ok])
        by = {ep: sorted(x for x, v in S.items() if v[0] == ep) for ep in EP}
        if len(by["pre"]) < 3:
            continue

        def pat(labs, sp, S=S, W=W):
            X = np.vstack([S[x][1] for x in labs])
            y = np.concatenate([(S[x][2] == sp).astype(float) for x in labs])
            return W @ pattern(X, y) if 0 < y.sum() < len(y) else None
        for sp in POS:
            A = {ep: pat(by[ep], sp) for ep in EP if len(by[ep]) >= 2}
            A = {k: v for k, v in A.items() if v is not None}
            if "pre" not in A:
                continue
            rms = float(np.sqrt(np.mean(A["pre"] ** 2)))
            for ep, v in A.items():
                maps[(a, sp, ep)] = (v / rms).astype(np.float32)
            H = {ep: [pat(by[ep][0::2], sp), pat(by[ep][1::2], sp)] for ep in A}

            def c(u, v):
                return float(np.corrcoef(u, v)[0, 1]) if u is not None and v is not None else np.nan
            rel_pre = c(*H["pre"])
            for ep in A:
                rel = c(*H[ep])
                cross = np.nanmean([c(H[ep][i], H["pre"][j]) for i in (0, 1) for j in (0, 1)]) if ep != "pre" else rel_pre
                rows.append(dict(animal=a, spout=sp, epoch=ep, rel=rel, r_full=c(A[ep], A["pre"]),
                                 r_disatt=cross / np.sqrt(rel * rel_pre) if rel > 0 and rel_pre > 0 else np.nan,
                                 amp=float(np.sqrt(np.mean((A[ep] / rms) ** 2)))))
    return pd.DataFrame(rows), maps


def figure_data(T: pd.DataFrame, maps: dict) -> dict:
    """Everything the figure draws: per (spout, column) the mean-of-animals map, the number of animals and (post epochs)
    the mean noise-corrected similarity to pre; plus the imaging grid that places stat-mask pixels on the image."""
    cells = {}
    for sp in POS:
        for cname in COLS:
            if cname in EP:
                have = [maps[(a, sp, cname)] for a in AN if (a, sp, cname) in maps]
            else:
                have = [maps[(a, sp, "chronic")] - maps[(a, sp, "pre")] for a in AN
                        if (a, sp, "chronic") in maps and (a, sp, "pre") in maps]
            if not have:
                continue
            g = T[(T.spout == sp) & (T.epoch == cname)]
            cells[(sp, cname)] = dict(map=np.nanmean(have, 0), n=len(have),
                                      sim=g.r_disatt.mean() if cname in EP[1:] else None)
    return dict(cells=cells, grid=imaging_grid())


def plot(D: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    to_img, ol = grid_fns(D["grid"])
    cols = COLS
    fig, axs = plt.subplots(len(POS), len(cols), figsize=(2.6 * len(cols), 2.3 * len(POS)), squeeze=False)
    lim = 2.5
    for i, sp in enumerate(POS):
        for j, cname in enumerate(cols):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            c = D["cells"].get((sp, cname))
            if c is None:
                ax.axis("off")
                continue
            im = ax.imshow(to_img(c["map"]), cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
            ttl = f"{cname} (n={c['n']})" + (f"\nsim to pre {c['sim']:.2f}" if cname in EP[1:] else "")
            ax.set_title(ttl, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(sp.replace("_", " "), fontsize=9)
    fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01, label="pattern / animal's pre RMS for that spout")
    fig.suptitle("NEURAL: WHERE activity distinguishes each spout's cued licks from the other five's -- Haufe activation "
                 "pattern of a one-vs-rest ridge decoder, refit per epoch\n(each animal's pre-stroke SVD basis, lags "
                 "0..0.2 s from the lick peak), / the animal's pre RMS, mean of animals; sim = noise-corrected similarity to "
                 "pre (mean of animals). ipsi (lesion) = image LEFT. Unscaled features. PROVISIONAL", fontsize=9)
    fig.savefig(il.OUT / "figures" / f"{FIG}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if plot_only():                                       # redraw from the figure data alone
        plot(load_fd(FIG))
        return 0
    T, maps = compute()
    T.to_csv(il.OUT / "spout_patterns.csv", index=False)
    pd.set_option("display.width", 220)
    print("== per animal (rule 8): noise-corrected similarity to pre")
    print(T.pivot_table(index=["spout", "animal"], columns="epoch", values="r_disatt")[EP[1:]].reindex(POS, level=0)
          .round(2).to_string())
    print("\n== amplitude / pre")
    print(T.pivot_table(index=["spout", "animal"], columns="epoch", values="amp")[EP].reindex(POS, level=0).round(2)
          .to_string())
    D = figure_data(T, maps)
    save_fd(FIG, D)
    plot(D)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""FaceRhythm lick DIRECTION vs DIRECTIONLESS vs other (Priya 10-25: "for fr runs - what happened to our lick direction vs
lick directionless vs other analyses?"; "the FR direction vs directionless figures / maps?").

Grouping modes (`wfield_local/fr_groups.py`, master_v1): B = lick amount + other (DIRECTIONLESS FR); S = B + side
(left / right lick factors); D = B + distance (close / far factors); A = everything. Per session, per mode m (O2 runs
fr_A / fr_B / fr_S / fr_D, collected as `..._frg-<m>`):
  FR gain_m        fr:full_m - full                    (what FaceRhythm adds to the full behavioural model)
  position | FR_m  fr:full_m - fr:drop:position_m      (unique spout-position R2 with FaceRhythm in the model)
Direction through FR = S - B (side), D - B (distance), A - B (both); "other" is inside B with lick amount (no model drops
"other" alone, so it cannot be split off). R2 as a fraction.
Figures: (1) areas x modes, pre level + chronic - pre (nested bootstrap, animals -> sessions); (2) pixel maps (2 x 2-px
blocks), pre level and chronic - pre, of B and the S / D / A increments, for FR gain and position | FR. Image left =
ipsilesional.

Usage: python scripts/fr_direction.py [--windows lick cue]
Output: <E>/fr_encoding_summary/direction/ (areas.csv, bootstrap.csv, figures/)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402

MODES = "ABSD"
FRD = {m: E / f"movement_first_fr-master_pre_v1_subset-fr_exec-polar_licks-combined_frg-{m}" / "sessions" for m in MODES}
AREAS = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left", "MOs_right", "MOs_left", "SSp-n_right", "SSp-n_left"]
EPS = ["acute", "subacute", "chronic"]


def area_rows(windows) -> pd.DataFrame:
    from wfield_local import config, epochs
    rows = []
    for f in sorted(FRD["A"].glob("*_regions.csv")):
        lab = f.stem.replace("_regions", "")
        if lab[-4:] < "0801" or not all((FRD[m] / f.name).exists() for m in MODES):
            continue
        base = {"label": lab, "animal": config.animal_of(lab), "epoch": epochs.epoch_of(lab)}
        R = {m: pd.read_csv(FRD[m] / f.name) for m in MODES}
        R = {m: r[r.window.isin(windows) & r.area.isin(AREAS)].pivot_table(index=["area", "window"], columns="model",
                                                                            values="r2") for m, r in R.items()}
        for key in R["A"].index:
            row = dict(base, area=key[0], window=key[1])
            for m in MODES:
                x = R[m].loc[key]
                row[f"gain_{m}"] = x["fr:full"] - x["full"]
                row[f"pos_{m}"] = x["fr:full"] - x["fr:drop:position"]
            for q in ("gain", "pos"):
                row[f"{q}_side"] = row[f"{q}_S"] - row[f"{q}_B"]
                row[f"{q}_dist"] = row[f"{q}_D"] - row[f"{q}_B"]
                row[f"{q}_both"] = row[f"{q}_A"] - row[f"{q}_B"]
            rows.append(row)
    return pd.DataFrame(rows)


def session_maps(lab: str, windows) -> dict:
    out = {}
    z = {m: np.load(FRD[m] / f"{lab}.npz") for m in MODES}
    H, W = (int(x) for x in z["A"]["shape"])
    flat = z["A"]["flat"]
    for w in windows:
        r2 = {}
        for m in MODES:
            k = {str(x): i for i, x in enumerate(z[m]["map_keys"])}
            M = z[m]["maps"].astype(float)
            tot = M[k[f"TOTAL|{w}"]]
            ok = tot > 0
            den = np.where(ok, tot, 1.0)
            for mod in ("fr:full", "full", "fr:drop:position"):
                r2[(m, mod)] = np.where(ok, 1 - M[k[f"{mod}|{w}"]] / den, np.nan)
        q = {}
        for m in MODES:
            q[f"gain_{m}"] = r2[(m, "fr:full")] - r2[(m, "full")]
            q[f"pos_{m}"] = r2[(m, "fr:full")] - r2[(m, "fr:drop:position")]
        for qq in ("gain", "pos"):
            q[f"{qq}_side"] = q[f"{qq}_S"] - q[f"{qq}_B"]
            q[f"{qq}_dist"] = q[f"{qq}_D"] - q[f"{qq}_B"]
            q[f"{qq}_both"] = q[f"{qq}_A"] - q[f"{qq}_B"]
        for name, v in q.items():
            g = np.full(H * W, np.nan)
            g[flat] = v
            out[(name, w)] = g.reshape(H, W)
    return out


def main(argv=None) -> int:
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed

    from combined_encoding_figures import _outline
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", nargs="+", default=["lick", "cue"])
    a = ap.parse_args(argv)
    out = E / "fr_encoding_summary" / "direction"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    T = area_rows(a.windows)
    T.to_csv(out / "areas.csv", index=False)
    Q = ["gain_B", "gain_side", "gain_dist", "gain_both", "pos_B", "pos_side", "pos_dist", "pos_both"]
    A = T.groupby(["window", "area", "animal", "epoch"])[Q].mean().reset_index()
    pd.set_option("display.width", 220)
    print(f"{T.label.nunique()} sessions")
    print("== pre level, mean of animals (R2 fraction)")
    print(A[A.epoch == "pre"].groupby(["window", "area"])[Q].mean().round(4).to_string())
    rows = []
    for (w, ar), t in T.groupby(["window", "area"]):
        for q in Q:
            pre = {x: t[(t.epoch == "pre") & (t.animal == x)][q].dropna().to_numpy() for x in AN}
            for ep in EPS:
                post = {x: t[(t.epoch == ep) & (t.animal == x)][q].dropna().to_numpy() for x in AN}
                com = [x for x in AN if len(pre[x]) and len(post[x])]
                iv, d = ak.boot_delta({x: post[x] for x in com}, {x: pre[x] for x in com}, np.random.default_rng(0), draws=True)
                per = {x: float(post[x].mean() - pre[x].mean()) for x in com}
                rows.append({"window": w, "area": ar, "q": q, "epoch": ep, "point": iv.point, "lo": iv.lo, "hi": iv.hi,
                             "p": float(ak.boot_p(d)), "n_up": sum(v > 0 for v in per.values()), **per})
    B = pd.DataFrame(rows)
    B.to_csv(out / "bootstrap.csv", index=False)
    print("== chronic - pre (point [CI], p, animals up)")
    print(B[B.epoch == "chronic"][["window", "area", "q", "point", "lo", "hi", "p", "n_up"]].round(4).to_string(index=False))

    # figure 1: areas, pre level (B + increments) and chronic - pre
    lab = {"B": "directionless (B)", "side": "+ side (S-B)", "dist": "+ distance (D-B)", "both": "+ side & distance (A-B)"}
    col = {"B": "#7570b3", "side": "#d95f02", "dist": "#1b9e77", "both": "#e7298a"}
    for w in a.windows:
        fig, axs = plt.subplots(2, 2, figsize=(16, 8.5))
        for r, q in enumerate(("gain", "pos")):
            for c, kind in enumerate(("pre", "chronic")):
                ax = axs[r, c]
                for k, part in enumerate(("B", "side", "dist", "both")):
                    x = np.arange(len(AREAS)) + (k - 1.5) * 0.2
                    if kind == "pre":
                        t = A[(A.window == w) & (A.epoch == "pre")]
                        y = [t[t.area == ar][f"{q}_{part}"].mean() for ar in AREAS]
                        ax.bar(x, y, width=0.19, color=col[part], label=lab[part] if (r == 0 and c == 0) else None)
                        for i, ar in enumerate(AREAS):
                            v = t[t.area == ar][f"{q}_{part}"].to_numpy()
                            ax.scatter(np.full(len(v), x[i]), v, s=7, c="k", zorder=5)
                    else:
                        b = B[(B.window == w) & (B.epoch == "chronic") & (B.q == f"{q}_{part}")].set_index("area").reindex(AREAS)
                        ax.bar(x, b.point, width=0.19, color=col[part])
                        ax.errorbar(x, b.point, yerr=[b.point - b.lo, b.hi - b.point], fmt="none", ecolor="k", lw=0.8)
                ax.axhline(0, color="k", lw=0.6)
                ax.set_xticks(range(len(AREAS)))
                ax.set_xticklabels([x.replace("_right", "\ncontra").replace("_left", "\nipsi") for x in AREAS], fontsize=8)
                ttl = "what FaceRhythm adds (fr:full - full)" if q == "gain" else "unique spout-position R2 with FaceRhythm"
                ax.set_title(f"{ttl}: {'pre-stroke level (B; increments for S / D / A)' if kind == 'pre' else 'chronic - pre [95% nested CI]'}",
                             fontsize=9)
                if c == 0:
                    ax.set_ylabel("held-out R2 (fraction)")
        axs[0, 0].legend(fontsize=8, frameon=False, ncol=4, loc="upper right")
        fig.suptitle(f"NEURAL: FaceRhythm lick DIRECTION vs DIRECTIONLESS, per-session encoder, {w} window. B = lick amount + other "
                     "factors (directionless); S / D / A add side, distance, both.\nPre: B bars = level, others = what the direction "
                     "factors add on top of B (dots = animals). Bottom: negative increment = direction factors take that much more "
                     "of the position code.", fontsize=9)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.08, hspace=0.42, wspace=0.12)
        fig.savefig(out / "figures" / f"fr_direction_areas_{w}.png", dpi=115)
        plt.close(fig)

    # figure 2: pixel maps
    meta = T[["label", "animal", "epoch"]].drop_duplicates().sort_values("label")
    got = Parallel(n_jobs=6)(delayed(session_maps)(lab_, a.windows) for lab_ in meta.label)
    acc = {}
    for (_, m_), g in zip(meta.iterrows(), got):
        for key, v in g.items():
            acc.setdefault((m_.epoch, m_.animal, key), []).append(v)
    Mp = {k: np.nanmean(v, 0) for k, v in acc.items()}
    ol = _outline(meta)
    cols = [("gain_B", "FR adds: directionless (B)"), ("gain_side", "FR adds: + side (S-B)"), ("gain_dist", "+ distance (D-B)"),
            ("gain_both", "+ side & distance (A-B)"), ("pos_B", "position unique | FR-B"), ("pos_both", "position: A - B")]
    for w in a.windows:
        fig, axs = plt.subplots(2, len(cols), figsize=(2.5 * len(cols), 6), squeeze=False)
        for j, (q, t_) in enumerate(cols):
            for i, kind in enumerate(("pre", "chronic")):
                ans = [x for x in AN if ("pre", x, (q, w)) in Mp]
                if kind == "pre":
                    M = np.nanmean([Mp[("pre", x, (q, w))] for x in ans], 0)
                else:
                    M = np.nanmean([Mp[("chronic", x, (q, w))] - Mp[("pre", x, (q, w))] for x in ans if ("chronic", x, (q, w)) in Mp], 0)
                ax = axs[i, j]
                if kind == "pre" and q.endswith("_B"):
                    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=np.nanpercentile(M, 99), interpolation="nearest")
                else:
                    v = np.nanpercentile(np.abs(M), 99)
                    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                if ol is not None and ol.shape == M.shape:
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.45, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
                cb.locator = matplotlib.ticker.MaxNLocator(4)
                cb.update_ticks()
                cb.ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(t_, fontsize=8)
                if j == 0:
                    ax.set_ylabel("pre level" if kind == "pre" else "chronic - pre", fontsize=9)
        fig.suptitle(f"NEURAL: FaceRhythm direction vs directionless, pixel maps (2x2-px blocks), {w} window; held-out R2 (fraction), "
                     "within animal then mean of 4.\nViridis = level, RdBu = increment or change. Image left = ipsilesional.", fontsize=9)
        fig.subplots_adjust(left=0.03, right=0.995, top=0.86, bottom=0.03, wspace=0.08, hspace=0.18)
        fig.savefig(out / "figures" / f"fr_direction_maps_{w}.png", dpi=115)
        plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

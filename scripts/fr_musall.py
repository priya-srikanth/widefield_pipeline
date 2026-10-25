"""Musall-style variance figures and area maps for the encoder WITH FaceRhythm (Priya 10-25: "add the musall-style plots
and maps for the fr-including runs"). Per-session (not frozen) combined encoder, held-out R2, region tables.

Sources: main run `movement_first_exec-polar_licks-combined` (no FR: full, alone:<g>, drop:<g>) and the FR run
`movement_first_fr-master_pre_v1_subset-fr_exec-polar_licks-combined_frg-<mode>` (same base model + FR: fr:full,
fr:alone, fr:drop:<g>). Groups g: task, instructed, uninstructed, position (+ FR itself).
  ALONE        g alone (main run; FR is not in the alone models, so identical with / without FR); FR alone = fr:alone
  UNIQUE | FR  fr:full - fr:drop:g (with FR in the model); FR unique = fr:full - full
  UNIQUE       full - drop:g (without FR)
  TASK-ALIGNED = ALONE - UNIQUE | FR (Musall Fig. 5); position has no alone model (unique bars only)
Figures: (1) per window, rows = [stacked task-independent (unique | FR) + task-aligned, per group; unique without vs
with FR], columns = epochs, bars = mean of animals, dots = animals; (2) area maps (Allen areas painted with each area's
value): pre level and chronic minus pre of full R2 with FR, FR unique, position unique without / with FR, task and
uninstructed unique | FR; one colour scale and colorbar per column.

Usage: python scripts/fr_musall.py [--mode A] [--windows lick cue]
Output: <E>/fr_encoding_summary/figures/fr_musall_*.png, fr_area_maps_*.png, fr_musall_by_animal.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding")
MAIN = E / "movement_first_exec-polar_licks-combined" / "sessions"
AN = ["PS92", "PS93", "PS94", "PS95"]
EP = ["pre", "acute", "subacute", "chronic"]
GROUPS = ["fr", "task", "instructed", "uninstructed", "position"]
GCOL = {"fr": "#1b9e77", "task": "#7570b3", "instructed": "#d95f02", "uninstructed": "#666666", "position": "#e7298a"}
MK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}


def table(mode: str, windows) -> pd.DataFrame:
    from wfield_local import config, epochs
    frd = E / f"movement_first_fr-master_pre_v1_subset-fr_exec-polar_licks-combined_frg-{mode}" / "sessions"
    rows = []
    for f in sorted(MAIN.glob("*_regions.csv")):
        lab = f.stem.replace("_regions", "")
        g = frd / f.name
        if lab[-4:] < "0801" or not g.exists():
            continue
        m = pd.read_csv(f)
        r = pd.read_csv(g)
        m = m[m.window.isin(windows)].pivot_table(index=["area", "window"], columns="model", values="r2")
        r = r[r.window.isin(windows)].pivot_table(index=["area", "window"], columns="model", values="r2")
        for key in r.index.intersection(m.index):
            a, b = m.loc[key], r.loc[key]
            row = {"label": lab, "animal": config.animal_of(lab), "epoch": epochs.epoch_of(lab), "area": key[0],
                   "window": key[1], "full": a["full"], "full_fr": b["fr:full"],
                   "alone_fr": b["fr:alone"], "unique_fr": (b["fr:full"] - b["full"])}
            for gname in ("task", "instructed", "uninstructed", "position"):
                row[f"alone_{gname}"] = a.get(f"alone:{gname}", np.nan)
                row[f"unique_{gname}"] = (a["full"] - a[f"drop:{gname}"])
                row[f"uniquefr_{gname}"] = (b["fr:full"] - b[f"fr:drop:{gname}"])
            rows.append(row)
    return pd.DataFrame(rows)


PIX = [("full_fr", "full R2 (with FR)"), ("alone_fr", "FaceRhythm alone"), ("unique_fr", "FaceRhythm unique"),
       ("alone_uninstructed", "uninstructed alone"), ("uniquefr_uninstructed", "uninstructed unique | FR"),
       ("uniquefr_task", "task unique | FR"), ("unique_position", "position unique, no FR"),
       ("uniquefr_position", "position unique | FR")]


def session_pixels(lab: str, mode: str, windows) -> dict:
    """{(quantity, window): (H, W) block R2 (fraction, 0-1)} for one session from the stored 2 x 2-block SSE maps of the main
    (no-FR) and FR runs (same grid; R2 = 1 - SSE / SST per block, as combined_encoding_figures._grid_from)."""
    frd = E / f"movement_first_fr-master_pre_v1_subset-fr_exec-polar_licks-combined_frg-{mode}" / "sessions"
    zm, zf = np.load(MAIN / f"{lab}.npz"), np.load(frd / f"{lab}.npz")
    assert np.array_equal(zm["flat"], zf["flat"])
    H, W = (int(x) for x in zm["shape"])
    km = {str(k): i for i, k in enumerate(zm["map_keys"])}
    kf = {str(k): i for i, k in enumerate(zf["map_keys"])}
    Mm, Mf = zm["maps"].astype(float), zf["maps"].astype(float)
    out = {}
    for w in windows:
        tot = Mf[kf[f"TOTAL|{w}"]]
        ok = tot > 0
        den = np.where(ok, tot, 1.0)
        def r2(M, ix, k):
            return np.where(ok, 1 - M[ix[f"{k}|{w}"]] / den, np.nan)  # noqa: B023 (called within this iteration)
        full_fr = r2(Mf, kf, "fr:full")
        q = {"full_fr": full_fr, "alone_fr": r2(Mf, kf, "fr:alone"), "unique_fr": full_fr - r2(Mf, kf, "full"),
             "alone_uninstructed": r2(Mm, km, "alone:uninstructed"),
             "unique_position": r2(Mm, km, "full") - r2(Mm, km, "drop:position")}
        for g in ("task", "uninstructed", "position"):
            q[f"uniquefr_{g}"] = full_fr - r2(Mf, kf, f"fr:drop:{g}")
        for k, v in q.items():
            g_ = np.full(H * W, np.nan)
            g_[zm["flat"]] = v
            out[(k, w)] = g_.reshape(H, W)
    return out


def pixel_maps(mode: str, windows, T: pd.DataFrame, out: Path) -> None:
    """Musall-style PIXEL maps (Musall 2019 Fig. 4: full / alone / unique cvR2 per pixel): rows pre level, acute - pre,
    chronic - pre (within animal, then mean of animals); one colorbar per panel."""
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from combined_encoding_figures import _outline
    acc: dict = {}
    meta = T[["label", "animal", "epoch"]].drop_duplicates()
    for _, m in meta.iterrows():
        for k, g in session_pixels(m.label, mode, windows).items():
            acc.setdefault((m.epoch, m.animal, k), []).append(g)
    A = {key: np.nanmean(v, 0) for key, v in acc.items()}            # per animal x epoch
    ol = _outline(meta.rename(columns={}))
    np.savez_compressed(out / f"fr_musall_pixel_maps_mode{mode}.npz",
                        keys=np.array([f"{e}|{an}|{k}|{w}" for (e, an, (k, w)) in A]), maps=np.stack(list(A.values())))
    for w in windows:
        rows_ = [("pre (level)", None)] + [(f"{ep} minus pre", ep) for ep in ("acute", "chronic")]
        fig, axs = plt.subplots(3, len(PIX), figsize=(2.3 * len(PIX), 7.6), squeeze=False)
        for r, (title, ep) in enumerate(rows_):
            for j, (c, lab_) in enumerate(PIX):
                ans = sorted({an for (e, an, kk) in A if e == "pre" and kk == (c, w)})
                if ep is None:
                    M = np.nanmean([A[("pre", an, (c, w))] for an in ans], 0)
                else:
                    M = np.nanmean([A[(ep, an, (c, w))] - A[("pre", an, (c, w))] for an in ans if (ep, an, (c, w)) in A], 0)
                ax = axs[r, j]
                if ep is None:
                    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=np.nanpercentile(M, 99), interpolation="nearest")
                else:
                    v = np.nanpercentile(np.abs(M), 99)
                    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                if ol is not None and ol.shape == M.shape:
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, orientation="horizontal")
                cb.ax.tick_params(labelsize=6)
                if r == 0:
                    ax.set_title(lab_, fontsize=8)
                if j == 0:
                    ax.set_ylabel(title, fontsize=9)
        fig.suptitle(f"NEURAL: Musall-style PIXEL maps, per-session encoder with FaceRhythm (mode {mode}), {w} window; "
                     "held-out R2 (fraction, 0-1) per 2x2-px block (1 - SSE/SST), within animal then mean of 4 animals.\n"
                     "Row 1: pre-stroke level (viridis, 0 to p99); rows 2-3: epoch minus pre (RdBu, +-p99). unique = "
                     "full minus the model without the group; | FR = FaceRhythm in the model. Image left = ipsilesional.",
                     fontsize=9)
        fig.subplots_adjust(left=0.03, right=0.995, top=0.88, bottom=0.03, wspace=0.08, hspace=0.12)
        fig.savefig(out / "figures" / f"fr_pixel_maps_{w}_mode{mode}.png", dpi=110)
        plt.close(fig)
        print("->", out / "figures" / f"fr_pixel_maps_{w}_mode{mode}.png", flush=True)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="A")
    ap.add_argument("--windows", nargs="+", default=["lick", "cue"])
    ap.add_argument("--area", default="cortex")
    a = ap.parse_args(argv)
    out = E / "fr_encoding_summary"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    T = table(a.mode, a.windows)
    A = T.groupby(["area", "window", "animal", "epoch"]).mean(numeric_only=True).reset_index()
    A.to_csv(out / f"fr_musall_by_animal_mode{a.mode}.csv", index=False)
    print(f"{T.label.nunique()} sessions", flush=True)

    for w in a.windows:
        t = A[(A.area == a.area) & (A.window == w)]
        fig, axs = plt.subplots(2, 4, figsize=(15, 8), sharey="row")
        for k, ep in enumerate(EP):
            x_ = t[t.epoch == ep]
            ax = axs[0, k]
            for i, g in enumerate(GROUPS):
                uq = x_["unique_fr"] if g == "fr" else x_[f"uniquefr_{g}"]
                al = x_["alone_fr"] if g == "fr" else x_[f"alone_{g}"]
                u = uq.mean()
                ax.bar(i, u, color=GCOL[g], width=0.65, label="task-independent (unique, with FR)" if i == 0 and k == 0 else None)
                if al.notna().any():
                    ta = al.mean() - u
                    ax.bar(i, ta, bottom=max(u, 0), color="0.8", width=0.65, label="task-aligned (alone - unique)" if i == 0 and k == 0 else None)
                    for an, v in zip(x_.animal, al):
                        ax.scatter(i + 0.2, v, marker=MK[an], s=14, facecolor="w", edgecolor="k", lw=0.6, zorder=5)
                for an, v in zip(x_.animal, uq):
                    ax.scatter(i - 0.2, v, marker=MK[an], s=14, facecolor=GCOL[g], edgecolor="k", lw=0.6, zorder=5)
            ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(["FaceRhythm", "task", "instructed", "uninstr.", "position"], fontsize=8, rotation=20)
            ax.axhline(0, color="k", lw=0.6); ax.set_title(ep)
            ax = axs[1, k]
            for i, g in enumerate(GROUPS[1:]):
                for j, (col, lab_, cc) in enumerate(((f"unique_{g}", "unique, no FR", "0.82"), (f"uniquefr_{g}", "unique, with FR", GCOL[g]))):
                    v = x_[col]
                    ax.bar(i + (j - 0.5) * 0.36, v.mean(), width=0.34, color=cc, alpha=0.9,
                           label=lab_ if (i == 0 and k == 0) else None)
                    for an, y in zip(x_.animal, v):
                        ax.scatter(i + (j - 0.5) * 0.36, y, marker=MK[an], s=12, facecolor="w", edgecolor="k", lw=0.6, zorder=5)
            ax.set_xticks(range(4)); ax.set_xticklabels(["task", "instructed", "uninstr.", "position"], fontsize=8, rotation=20)
            ax.axhline(0, color="k", lw=0.6)
        axs[0, 0].set_ylabel(f"{a.area}, {w} window\nheld-out R2 (fraction, 0-1)"); axs[1, 0].set_ylabel("unique R2 (fraction, 0-1)\nwithout vs with FaceRhythm")
        axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left"); axs[1, 0].legend(fontsize=7, frameon=False, loc="upper left")
        fig.suptitle(f"NEURAL: Musall-style variance with FaceRhythm (master_pre_v1, mode {a.mode}) in the per-session encoder, {a.area}, {w} window.\n"
                     "Top: stacked task-independent (unique R2 with FR in the model; coloured) + task-aligned (alone - unique; grey); filled dots = unique, "
                     "open = alone, per animal (position has no alone model).\nBottom: each group's unique R2 without vs with FaceRhythm (what FR takes).",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(out / "figures" / f"fr_musall_{a.area}_{w}_mode{a.mode}.png", dpi=120)
        plt.close(fig)

    # area maps
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frozen_area_maps import area_image, outline
    img, names = area_image(sorted(T.label.unique())[0])
    ol = outline(img)
    cols = [("full_fr", "full R2 (with FR)"), ("unique_fr", "FaceRhythm unique"), ("unique_position", "position unique, no FR"),
            ("uniquefr_position", "position unique, with FR"), ("uniquefr_task", "task unique | FR"),
            ("uniquefr_uninstructed", "uninstructed unique | FR")]
    for w in a.windows:
        t = A[A.window == w]
        lev = t[t.epoch == "pre"].groupby("area")[[c for c, _ in cols]].mean()
        pre_an = t[t.epoch == "pre"].set_index(["area", "animal"])
        rows_ = [("pre (level)", lev, False)]
        for ep in ("acute", "chronic"):
            post = t[t.epoch == ep].set_index(["area", "animal"])
            d = (post[[c for c, _ in cols]] - pre_an[[c for c, _ in cols]]).dropna(how="all").groupby("area").mean()
            rows_.append((f"{ep} minus pre", d, True))
        fig, axs = plt.subplots(len(rows_), len(cols), figsize=(2.6 * len(cols), 2.5 * len(rows_) + 0.8), squeeze=False)
        for r, (title, M, diff) in enumerate(rows_):
            for j, (c, lab_) in enumerate(cols):
                ax = axs[r, j]
                pic = np.full(img.shape, np.nan)
                for i, nm in enumerate(names):
                    if nm in M.index and np.isfinite(M.loc[nm, c]):
                        pic[img == i] = M.loc[nm, c]
                v = np.nanmax(np.abs(pic)) if np.isfinite(pic).any() else 1
                im = ax.imshow(pic, cmap="RdBu_r" if diff else ("viridis" if j == 0 else "magma"), vmin=-v if diff else 0, vmax=v,
                               interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
                cb.ax.tick_params(labelsize=6)
                if r == 0:
                    ax.set_title(lab_, fontsize=8)
                if j == 0:
                    ax.set_ylabel(title, fontsize=9)
        fig.suptitle(f"NEURAL: area maps, per-session encoder with FaceRhythm (mode {a.mode}), {w} window; held-out R2 (fraction, 0-1), mean of animals "
                     "(each area painted with its value).\nRow 1: pre-stroke level; rows 2-3: epoch minus pre within animal, then mean. "
                     "Image left = ipsilesional.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / "figures" / f"fr_area_maps_{w}_mode{a.mode}.png", dpi=115)
        plt.close(fig)
    pixel_maps(a.mode, a.windows, T, out)
    show = A[(A.area == a.area)].groupby(["window", "epoch"])[["alone_fr", "unique_fr", "unique_position", "uniquefr_position",
                                                               "uniquefr_task", "uniquefr_uninstructed"]].mean().round(3)
    print(show.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

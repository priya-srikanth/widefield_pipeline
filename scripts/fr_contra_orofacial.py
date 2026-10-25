"""Is the chronic rise in task / position coding in contralesional orofacial sensorimotor cortex real once face movement
is in the model? (Priya 10-25, on Figs 16g-16h: "with regressing out uninstructed facial movements, we're left with
increased variance explained in contralesional orofacial sensorimotor cortex? The uninstructed alone, task unique, and
position unique look particularly important".)

Per-session encoder with FaceRhythm (mode A), region tables from `fr_musall.table`. Quantities (held-out R2 (fraction, 0-1)):
alone_uninstructed, uniquefr_uninstructed, uniquefr_task, uniquefr_position (| FR = FaceRhythm in the model) and the
no-FR uniques for comparison. A priori areas (as the residual test): SSp-m, MOp, MOs, SSp-n per hemisphere (+ SSp-ul,
SSp-bfd as controls), lesion LEFT -> *_right = contralesional.

  1. per-animal epoch - pre tables (rule 8) with the count of animals of each sign;
  2. nested bootstrap (analysis_kit.boot_delta, animals -> sessions), Bonferroni over the 8 a priori area x hemisphere
     cells per quantity x window;
  3. laterality: contra - ipsi per session, same bootstrap;
  4. drift null: August pre, LATE half - EARLY half (date order) per animal, and chronic late - early half;
  5. denominator check: the R2 is SSE / SST per area; if SST falls, the same explained variance gives a larger R2 ->
     per-area log SST and absolute explained variance (SST - SSE, from the stored block maps) epoch vs pre;
  6. per-animal pixel maps of chronic - pre for the four quantities (from fr_musall_pixel_maps_mode<M>.npz).

Usage: python scripts/fr_contra_orofacial.py [--mode A] [--windows lick cue]
Output: <E>/fr_encoding_summary/contra_orofacial/ (tables + figures)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E, table  # noqa: E402

Q = ["alone_uninstructed", "uniquefr_uninstructed", "uniquefr_task", "uniquefr_position", "unique_task",
     "unique_position", "unique_fr", "full_fr"]
QKEY = ["alone_uninstructed", "uniquefr_task", "uniquefr_position", "uniquefr_uninstructed"]
APRIORI = ["SSp-m", "MOp", "MOs", "SSp-n"]
CTRL = ["SSp-ul", "SSp-bfd"]
EPS = ["acute", "subacute", "chronic"]


def per_animal(T: pd.DataFrame, areas, windows) -> pd.DataFrame:
    rows = []
    for w in windows:
        for ar in areas:
            t = T[(T.window == w) & (T.area == ar)]
            for q in Q:
                pre = t[t.epoch == "pre"].groupby("animal")[q].mean()
                for ep in EPS:
                    post = t[t.epoch == ep].groupby("animal")[q].mean()
                    for an in AN:
                        if an in pre.index and an in post.index:
                            rows.append({"window": w, "area": ar, "q": q, "epoch": ep, "animal": an,
                                         "pre": pre[an], "post": post[an], "delta": post[an] - pre[an]})
    return pd.DataFrame(rows)


def bootstrap(T: pd.DataFrame, areas, windows, value=None) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rows = []
    for w in windows:
        for ar in areas:
            t = T[(T.window == w) & (T.area == ar)]
            for q in Q:
                col = t[q] if value is None else value(t, q)
                pre = {a: col[(t.epoch == "pre") & (t.animal == a)].dropna().to_numpy() for a in AN}
                for ep in EPS:
                    post = {a: col[(t.epoch == ep) & (t.animal == a)].dropna().to_numpy() for a in AN}
                    pre_ = {a: v for a, v in pre.items() if len(v) and len(post[a])}
                    post_ = {a: post[a] for a in pre_}
                    if not pre_:
                        continue
                    rng = np.random.default_rng(0)
                    iv, d = ak.boot_delta(post_, pre_, rng, draws=True)
                    p = float(ak.boot_p(d)) if len(d) else np.nan
                    rows.append({"window": w, "area": ar, "q": q, "epoch": ep, "point": iv.point, "lo": iv.lo,
                                 "hi": iv.hi, "p": p, "n_animals": iv.n_animals})
    return pd.DataFrame(rows)


def laterality(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for base in APRIORI + CTRL + ["cortex"]:
        if base == "cortex":
            continue
        L = T[T.area == f"{base}_left"].set_index(["label", "window"])
        R = T[T.area == f"{base}_right"].set_index(["label", "window"])
        ix = L.index.intersection(R.index)
        d = R.loc[ix, Q] - L.loc[ix, Q]
        d = d.join(R.loc[ix, ["animal", "epoch"]]).reset_index()
        d["area"] = f"{base}_contra-ipsi"
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def drift_null(T: pd.DataFrame, areas, windows) -> pd.DataFrame:
    rows = []
    for w in windows:
        for ar in areas:
            t = T[(T.window == w) & (T.area == ar)]
            for ep in ("pre", "chronic"):
                for an in AN:
                    x = t[(t.epoch == ep) & (t.animal == an)].sort_values("label")
                    if len(x) < 4:
                        continue
                    h = len(x) // 2
                    for q in Q:
                        rows.append({"window": w, "area": ar, "q": q, "epoch": ep, "animal": an, "n": len(x),
                                     "late_minus_early": x[q].iloc[-h:].mean() - x[q].iloc[:h].mean()})
    return pd.DataFrame(rows)


def sst_table(mode: str, labels, windows) -> pd.DataFrame:
    """Per session x area x window: SST and SSE of fr:full and of fr:drop:<g> summed over the area's blocks."""
    from frozen_area_maps import area_image
    img, names = area_image(labels[0])
    frd = E / f"movement_first_fr-master_pre_v1_subset-fr_exec-polar_licks-combined_frg-{mode}" / "sessions"
    rows = []
    for lab in labels:
        z = np.load(frd / f"{lab}.npz")
        k = {str(x): i for i, x in enumerate(z["map_keys"])}
        M = z["maps"].astype(float)
        H, W = (int(x) for x in z["shape"])
        im_ = img if img.shape == (H, W) else img[::2, ::2][:H, :W]    # block maps live on the half-resolution grid
        code = im_.ravel()[z["flat"]]
        for w in windows:
            tot = M[k[f"TOTAL|{w}"]]
            for i, nm in enumerate(names):
                m = code == i
                if not m.any():
                    continue
                r = {"label": lab, "window": w, "area": nm, "sst": tot[m].sum(), "npix": int(m.sum())}
                for mod in ("fr:full", "fr:drop:task", "fr:drop:position", "fr:drop:uninstructed"):
                    r[f"sse_{mod}"] = M[k[f"{mod}|{w}"]][m].sum()
                rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import config, epochs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="A")
    ap.add_argument("--windows", nargs="+", default=["lick", "cue"])
    a = ap.parse_args(argv)
    out = E / "fr_encoding_summary" / "contra_orofacial"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    T = table(a.mode, a.windows)
    areas = [f"{b}_{h}" for b in APRIORI + CTRL for h in ("left", "right")] + ["cortex"]
    pd.set_option("display.width", 250)

    # 1. per animal
    P = per_animal(T, areas, a.windows)
    P.to_csv(out / "per_animal_delta.csv", index=False)
    wide = P[P.q.isin(QKEY)].pivot_table(index=["window", "q", "epoch", "area"], columns="animal", values="delta").round(3)
    wide["n_up"] = (wide[AN] > 0).sum(1)
    print("== 1. per-animal epoch - pre (R2, fraction), a priori areas, key quantities")
    print(wide.loc[(slice(None), slice(None), ["acute", "chronic"]), :].to_string())

    # 2. bootstrap, Bonferroni over 8 a priori cells
    B = bootstrap(T, areas, a.windows)
    B["p_bonf8"] = np.minimum(B.p * 8, 1)
    B.to_csv(out / "bootstrap_delta.csv", index=False)
    print("\n== 2. nested bootstrap (point [95% CI], p, Bonferroni x8), chronic, key quantities")
    sh = B[(B.epoch == "chronic") & B.q.isin(QKEY)].round(3)
    print(sh.to_string(index=False))

    # 3. laterality
    Lt = laterality(T)
    Lb = bootstrap(Lt, sorted(Lt.area.unique()), a.windows)
    Lb.to_csv(out / "laterality_bootstrap.csv", index=False)
    print("\n== 3. contra - ipsi, epoch - pre, key quantities")
    print(Lb[Lb.q.isin(QKEY) & (Lb.epoch != "subacute")].round(3).to_string(index=False))
    lw = per_animal(Lt, sorted(Lt.area.unique()), a.windows)
    lw.to_csv(out / "laterality_per_animal.csv", index=False)

    # 4. drift null
    D = drift_null(T, areas, a.windows)
    D.to_csv(out / "drift_null.csv", index=False)
    cmp_ = (P[P.epoch == "chronic"].merge(D[D.epoch == "pre"].rename(columns={"late_minus_early": "pre_drift"})
                                          [["window", "area", "q", "animal", "pre_drift"]], on=["window", "area", "q", "animal"])
            .merge(D[D.epoch == "chronic"].rename(columns={"late_minus_early": "chronic_trend"})
                   [["window", "area", "q", "animal", "chronic_trend"]], on=["window", "area", "q", "animal"], how="left"))
    cmp_.to_csv(out / "chronic_vs_drift.csv", index=False)
    print("\n== 4. chronic - pre vs pre late - early (drift) and chronic late - early, contralesional SSp-m / MOp, key q")
    print(cmp_[cmp_.area.isin(["SSp-m_right", "MOp_right", "SSp-m_left", "MOp_left"]) & cmp_.q.isin(QKEY)]
          [["window", "area", "q", "animal", "delta", "pre_drift", "chronic_trend"]].round(3).to_string(index=False))

    # 5. denominator
    S = sst_table(a.mode, sorted(T.label.unique()), a.windows)
    S["animal"] = S.label.map(config.animal_of)
    S["epoch"] = S.label.map(epochs.epoch_of)
    S["log_sst"] = np.log(S.sst / S.npix)
    for g in ("task", "position", "uninstructed"):
        S[f"abs_unique_{g}"] = (S[f"sse_fr:drop:{g}"] - S["sse_fr:full"]) / S.npix
    S["abs_explained_full"] = (S.sst - S["sse_fr:full"]) / S.npix
    S.to_csv(out / "sst_by_area.csv", index=False)
    cols = ["log_sst", "abs_explained_full", "abs_unique_task", "abs_unique_position", "abs_unique_uninstructed"]
    Sa = S.groupby(["window", "area", "animal", "epoch"])[cols].mean().reset_index()
    pre = Sa[Sa.epoch == "pre"].set_index(["window", "area", "animal"])[cols]
    ch = Sa[Sa.epoch == "chronic"].set_index(["window", "area", "animal"])[cols]
    rat = ch.copy()
    rat["log_sst"] = ch.log_sst - pre.log_sst
    for c in cols[1:]:
        rat[c] = ch[c] / pre[c]
    rat = rat.dropna(how="all").reset_index()
    rat.to_csv(out / "sst_chronic_vs_pre.csv", index=False)
    print("\n== 5. chronic vs pre: dlog SST (per pixel) and chronic/pre ratio of ABSOLUTE explained variance")
    print(rat[rat.area.isin(["SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left", "SSp-ul_right"])].round(3).to_string(index=False))

    # 6. per-animal pixel maps
    from combined_encoding_figures import _outline
    z = np.load(E / "fr_encoding_summary" / f"fr_musall_pixel_maps_mode{a.mode}.npz")
    Mp = dict(zip([str(k) for k in z["keys"]], z["maps"]))
    ol = _outline(T[["label"]].drop_duplicates())
    lab_ = {"alone_uninstructed": "uninstructed alone", "uniquefr_task": "task unique | FR",
            "uniquefr_position": "position unique | FR", "uniquefr_uninstructed": "uninstructed unique | FR"}
    for w in a.windows:
        fig, axs = plt.subplots(len(QKEY), 5, figsize=(13, 2.9 * len(QKEY)), squeeze=False)
        for i, q in enumerate(QKEY):
            ds = {an: Mp[f"chronic|{an}|{q}|{w}"] - Mp[f"pre|{an}|{q}|{w}"] for an in AN
                  if f"chronic|{an}|{q}|{w}" in Mp and f"pre|{an}|{q}|{w}" in Mp}
            ds["mean"] = np.nanmean(list(ds.values()), 0)
            v = np.nanpercentile(np.abs(np.stack(list(ds.values()))), 98)
            for j, (an, M) in enumerate(ds.items()):
                ax = axs[i, j]
                im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                if ol is not None and ol.shape == M.shape:
                    ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                if i == 0:
                    ax.set_title(an, fontsize=10)
                if j == 0:
                    ax.set_ylabel(lab_[q], fontsize=9)
            cb = fig.colorbar(im, ax=axs[i, :].tolist(), fraction=0.02, pad=0.01)
            cb.set_label("chronic - pre, R2 (fraction, 0-1)", fontsize=8); cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"NEURAL: chronic minus pre PER ANIMAL, per-session encoder with FaceRhythm (mode {a.mode}), {w} window;\n"
                     "held-out R2 (fraction, 0-1) per 2x2-px block; one colour scale per row (+-p98). Image left = ipsilesional.",
                     fontsize=9.5)
        fig.subplots_adjust(left=0.04, right=0.9, top=0.93, bottom=0.02, wspace=0.04, hspace=0.08)
        fig.savefig(out / "figures" / f"chronic_per_animal_{w}_mode{a.mode}.png", dpi=110)
        plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

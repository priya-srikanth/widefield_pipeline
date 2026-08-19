"""Did the position code CONVERGE, and did it MOVE ACROSS THE MIDLINE? Two tests on the spatial maps.

Priya, 2026-08-19, asking what correlation analysis to run on the raw maps. These are the two that test
the mechanism rather than restate it, and both follow from the same prediction: `evoked_amplitude`
found PS94's contralateral lateralisation collapsing toward zero, and `recoding_test` found its
position information intact but unreadable by the pre-stroke decoder. If those are the same fact, then:

  CONVERGENCE (test 1). Losing lateralisation should make the six position patterns LESS distinguishable
  from one another -- a decoder failing and patterns converging are the same statement seen from two
  sides. Measured as the 6x6 between-position structure, pre vs post.

      NOISE-UNBIASED, and this is not optional here. A plain correlation RDM is biased by trial noise,
      and post-stroke amplitude rose 2-3x -- so a raw metric would move even if the geometry were
      identical. Crossnobis (cross-validated Mahalanobis) removes that bias, which is exactly why the
      project adopted it for cross-day RDMs (DECISIONS.md, CACHE_VERSION v3/v4).

  MIDLINE TRANSFER (test 2). "More right activity" has two very different readings: the right
  hemisphere does more of its own thing, or the LEFT hemisphere's pattern has relocated to the right.
  Correlating each post-stroke pattern against its own pre-stroke pattern AND against the
  HEMISPHERE-SWAPPED version separates them.

      mirror_r > normal_r  -> the pattern now sits where the opposite hemisphere's used to: transfer.
      normal_r > mirror_r  -> the pattern stayed put and changed in some other way.

      The swap is done on the Allen region vector (each area's _left value exchanged with its _right),
      which is a mirror at region resolution and is robust to the pixel-level registration error that
      a literal image flip would inherit.

WHY THE R-L INDEX CANNOT ANSWER EITHER. It is one number per position and it is symmetric: a rightward
shift and a bilateral convergence both move it toward zero. These two tests separate the cases.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from wfield_local import config                                      # noqa: E402
from wfield_local.paths import PathResolver                          # noqa: E402
from wfield_local.plot_lick_aligned_averages import (DISPLAY_ORDER,   # noqa: E402
                                                     POSITION_NAMES)

MIN_TRIALS_PER_POS = 8


def _names(s):
    f = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")
    return {int(c): str(n) for c, n in json.load(open(f[0]))} if f else None


def _area_matrix(s, align="cue", source="roi"):
    """(trials x areas) with bins collapsed, plus labels, block ids and the area-name list."""
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import _trial_features

    names = _names(s)
    if names is None:
        return None
    X, y, g, _Xn, _yn, reg = _trial_features(s, _args(source=source, align=align, post_s=2.0))
    if len(y) < 30:
        return None
    reg = np.asarray(reg)
    codes = sorted({int(c) for c in reg.tolist()})
    A = np.stack([X[:, reg == c].mean(axis=1) for c in codes], axis=1)
    return A, np.asarray(y), np.asarray(g), [names.get(c, "?") for c in codes]


def _crossnobis(A, y, g, labels):
    """Noise-unbiased 6x6 distance matrix; NaN where a position lacks trials.

    Cross-validated over blocks: patterns from disjoint folds are multiplied, so the estimate is
    unbiased by trial noise and can legitimately be compared across sessions of different noise level
    -- which is the whole point, since post-stroke amplitude is 2-3x pre-stroke.
    """
    ub = np.unique(g)
    if len(ub) < 4:
        return None
    fold = {b: i % 2 for i, b in enumerate(ub)}
    f = np.array([fold[b] for b in g])
    D = np.full((len(labels), len(labels)), np.nan)
    means = {}
    for half in (0, 1):
        for c in labels:
            m = (y == c) & (f == half)
            if m.sum() >= MIN_TRIALS_PER_POS // 2:
                means[(half, c)] = A[m].mean(axis=0)
    # residual covariance for whitening, from within-(fold, position) cells
    res = []
    for (half, c), mu in means.items():
        m = (y == c) & (f == half)
        res.append(A[m] - mu)
    if not res:
        return None
    R = np.vstack(res)
    try:
        from sklearn.covariance import LedoitWolf
        P = np.linalg.pinv(LedoitWolf().fit(R).covariance_)
    except Exception:                                                # noqa: BLE001
        P = np.diag(1.0 / np.maximum(R.var(axis=0), 1e-12))
    for i, ci in enumerate(labels):
        for j, cj in enumerate(labels):
            if i >= j:
                continue
            k = [(0, ci), (0, cj), (1, ci), (1, cj)]
            if any(x not in means for x in k):
                continue
            d0 = means[(0, ci)] - means[(0, cj)]
            d1 = means[(1, ci)] - means[(1, cj)]
            D[i, j] = D[j, i] = float(d0 @ P @ d1)          # cross-validated -> unbiased
    return D


def _swap_lr(vec, areas):
    """Hemisphere-swapped copy of a region vector: each area's _left value exchanged with its _right."""
    idx = {a: i for i, a in enumerate(areas)}
    out = np.array(vec, float).copy()
    for i, a in enumerate(areas):
        if a.endswith("_left"):
            j = idx.get(a[:-5] + "_right")
        elif a.endswith("_right"):
            j = idx.get(a[:-6] + "_left")
        else:
            j = None
        if j is not None:
            out[i] = vec[j]
    return out


def session_geometry(s, align="cue", source="roi"):
    got = _area_matrix(s, align, source)
    if got is None:
        return None
    A, y, g, areas = got
    labels = [c for c in DISPLAY_ORDER if (y == c).sum() >= MIN_TRIALS_PER_POS]
    out = {"label": s["label"], "animal": s["label"][:4], "date": s["label"].split("_")[-1],
           "align": align, "areas": areas,
           "positions": [POSITION_NAMES[c] for c in labels],
           "pattern": {POSITION_NAMES[c]: A[y == c].mean(axis=0).tolist() for c in labels}}
    D = _crossnobis(A, y, g, labels)
    if D is not None:
        out["crossnobis"] = D.tolist()
        iu = np.triu_indices(len(labels), k=1)
        v = D[iu]
        out["mean_distance"] = float(np.nanmean(v))       # LOWER = positions converged
        out["n_pairs"] = int(np.isfinite(v).sum())
    return out


def mirror_test(pre_rows, post_row):
    """For each position: correlation of the post pattern with the pre pattern, and with its MIRROR."""
    areas = post_row["areas"]
    out = {}
    for p, vec in post_row["pattern"].items():
        pre = [np.array(r["pattern"][p], float) for r in pre_rows if p in r["pattern"]
               and r["areas"] == areas]
        if len(pre) < 3:
            continue
        mu = np.mean(pre, axis=0)
        v = np.array(vec, float)
        normal = float(np.corrcoef(v, mu)[0, 1])
        mirror = float(np.corrcoef(v, _swap_lr(mu, areas))[0, 1])
        # the same two correlations computed WITHIN the pre-stroke sessions give the baseline: a
        # symmetric brain already has some mirror correlation, so mirror>normal is only meaningful
        # against how those two compare before the lesion
        bn, bm = [], []
        for i, r in enumerate(pre_rows):
            if p not in r["pattern"] or r["areas"] != areas:
                continue
            others = [np.array(q["pattern"][p], float) for j, q in enumerate(pre_rows)
                      if j != i and p in q["pattern"] and q["areas"] == areas]
            if len(others) < 2:
                continue
            m2 = np.mean(others, axis=0)
            x = np.array(r["pattern"][p], float)
            bn.append(float(np.corrcoef(x, m2)[0, 1]))
            bm.append(float(np.corrcoef(x, _swap_lr(m2, areas))[0, 1]))
        out[p] = {"normal_r": normal, "mirror_r": mirror, "mirror_minus_normal": mirror - normal,
                  "pre_normal_r": float(np.mean(bn)) if bn else float("nan"),
                  "pre_mirror_r": float(np.mean(bm)) if bm else float("nan"),
                  "pre_mirror_minus_normal": float(np.mean(np.array(bm) - np.array(bn)))
                                              if bn else float("nan")}
        out[p]["transfer"] = bool(out[p]["mirror_minus_normal"] >
                                  out[p]["pre_mirror_minus_normal"] + 0.15)
    return out


def collect(animals=None, align="cue", source="roi"):
    keep = set(config.curated_dates()) | {"0817", "0818"}
    rows = []
    for s in config.load_sessions():
        if s["label"].split("_")[-1] not in keep:
            continue
        if animals and s["label"][:4] not in set(animals):
            continue
        try:
            r = session_geometry(s, align, source)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  {s['label']}: skip ({str(ex)[:60]})", flush=True)
            continue
        if r is None:
            continue
        r["phase"] = config.session_phase(r["animal"], r["date"])
        rows.append(r)
        print(f"  {r['label']} {align:6s} mean crossnobis "
              f"{r.get('mean_distance', float('nan')):.4f}  ({len(r['positions'])} positions)",
              flush=True)
    return sorted(rows, key=lambda r: (r["animal"], r["date"]))


def summarise(rows):
    out = {}
    for a in sorted({r["animal"] for r in rows}):
        pre = [r for r in rows if r["animal"] == a and r["phase"] == "pre"]
        post = [r for r in rows if r["animal"] == a and r["phase"] == "post"]
        if len(pre) < 3 or not post:
            out[a] = {"note": f"{len(pre)} pre / {len(post)} post"}
            continue
        rec = {"n_pre": len(pre)}
        v = np.array([r.get("mean_distance", np.nan) for r in pre], float)
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            band = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
            rec["convergence"] = {"pre_band": band, "post": {}}
            for r in post:
                x = r.get("mean_distance", np.nan)
                z = (x - band["mean"]) / band["sd"] if band["sd"] else np.nan
                rec["convergence"]["post"][r["label"]] = {
                    "mean_distance": float(x), "z": float(z),
                    # LOWER distance = the six positions became less distinguishable
                    "converged": bool(x < band["min"])}
        rec["mirror"] = {r["label"]: mirror_test(pre, r) for r in post}
        out[a] = rec
    return out


def plot(rows, summ, out_dir, align="cue"):
    animals = [a for a in sorted(summ) if "note" not in summ[a]]
    if not animals:
        return None
    POS = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    fig, axes = plt.subplots(2, len(animals), figsize=(5.0 * len(animals), 8.6), squeeze=False)
    for k, a in enumerate(animals):
        rec = summ[a]
        ax = axes[0][k]
        c = rec.get("convergence")
        if c:
            b = c["pre_band"]
            ax.axhspan(b["min"], b["max"], color="tab:blue", alpha=0.18)
            ax.axhline(b["mean"], color="tab:blue", lw=1.8)
            for i, (lab, v) in enumerate(c["post"].items()):
                ax.plot(i, v["mean_distance"], "o", ms=11, color="tab:red", markeredgecolor="k")
                ax.text(i, v["mean_distance"], f"  z{v['z']:+.1f}", fontsize=8, va="center",
                        color=("firebrick" if v["converged"] else "dimgrey"))
            ax.set_xticks(range(len(c["post"])))
            ax.set_xticklabels(list(c["post"]), rotation=30, ha="right", fontsize=7)
            ax.set_xlim(-0.6, max(len(c["post"]) - 0.4, 0.6))
        ax.set_title(f"{a} - between-position CROSSNOBIS distance", fontsize=9)
        if k == 0:
            ax.set_ylabel("mean pairwise distance\n(LOWER = positions converged)", fontsize=8)
        ax = axes[1][k]
        mir = rec.get("mirror", {})
        labs = list(mir)
        if labs:
            m = mir[labs[0]]
            x = np.arange(len(POS))
            w = 0.38
            nm = [m.get(p, {}).get("normal_r", np.nan) for p in POS]
            mr = [m.get(p, {}).get("mirror_r", np.nan) for p in POS]
            bl = [m.get(p, {}).get("pre_mirror_minus_normal", np.nan) for p in POS]
            ax.bar(x - w / 2, nm, w, label="vs own pre-stroke pattern", color="tab:blue",
                   edgecolor="k", lw=0.4)
            ax.bar(x + w / 2, mr, w, label="vs MIRRORED pre-stroke pattern", color="tab:orange",
                   edgecolor="k", lw=0.4)
            for xi, dd in zip(x, bl):
                if dd == dd:
                    ax.plot([xi - w, xi + w], [dd, dd], color="grey", lw=1.4, ls="--")
            ax.axhline(0, color="k", lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels(POS, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(-1, 1.05)
            ax.set_title(f"{a} - {labs[0]}: mirror test", fontsize=9)
            if k == 0:
                ax.set_ylabel("correlation with pre-stroke pattern", fontsize=8)
                ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle(
        f"SPATIAL REORGANISATION, {align}-aligned. TOP: between-position CROSSNOBIS distance - "
        "noise-unbiased, which is required here because post-stroke amplitude is 2-3x pre-stroke and "
        "a plain correlation RDM would move on that alone. LOWER distance = the six positions became "
        "less distinguishable, which is the same statement as a decoder failing. BOTTOM: does each "
        "post-stroke pattern resemble its OWN pre-stroke pattern (blue) or the HEMISPHERE-SWAPPED one "
        "(orange)? Orange above blue means the pattern relocated across the midline; the dashed grey "
        "line is the pre-stroke mirror-minus-normal baseline, because a symmetric brain already has "
        "some mirror correlation and only an excess over that baseline means anything.",
        fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p = Path(out_dir) / f"spatial_reorganisation_{align}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--align", nargs="+", default=["cue", "precue"])
    ap.add_argument("--source", default="roi")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    for align in args.align:
        print(f"\n=== {align}-aligned ===", flush=True)
        rows = collect(args.animals, align=align, source=args.source)
        if not rows:
            continue
        summ = summarise(rows)
        for a, rec in summ.items():
            if "note" in rec:
                print(f"  {a}: {rec['note']}")
                continue
            c = rec.get("convergence")
            if c:
                b = c["pre_band"]
                for lab, v in c["post"].items():
                    print(f"  {a} CONVERGENCE {lab}: {v['mean_distance']:.4f} vs pre "
                          f"{b['mean']:.4f}[{b['min']:.4f},{b['max']:.4f}] z={v['z']:+.1f}"
                          f"{'  CONVERGED' if v['converged'] else ''}")
            for lab, m in rec.get("mirror", {}).items():
                tr = [p for p, v in m.items() if v.get("transfer")]
                print(f"  {a} MIRROR {lab}: transfer at {tr or 'no position'}")
                for p, v in m.items():
                    print(f"       {p:13s} normal {v['normal_r']:+.3f}  mirror {v['mirror_r']:+.3f}"
                          f"  diff {v['mirror_minus_normal']:+.3f}   (pre-stroke baseline diff "
                          f"{v['pre_mirror_minus_normal']:+.3f})")
        json.dump({"align": align, "summary": summ,
                   "sessions": [{k: v for k, v in r.items() if k != "pattern"} for r in rows]},
                  open(Path(out) / f"spatial_reorganisation_{align}.json", "w"), indent=1,
                  default=float)
        p = plot(rows, summ, out, align)
        if p:
            print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

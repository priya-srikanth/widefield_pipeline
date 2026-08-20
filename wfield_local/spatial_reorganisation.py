"""Did the position code CONVERGE, and did it MOVE ACROSS THE MIDLINE? Two tests on the spatial maps.

Priya, 2026-08-19, asking what correlation analysis to run on the raw maps. These are the two that test
the mechanism rather than restate it, and both follow from the same prediction: `evoked_amplitude`
found PS94's contralateral lateralisation collapsing toward zero, and `recoding_test` found its
position information intact but unreadable by the pre-stroke decoder. If those are the same fact, then:

  CONVERGENCE (test 1). Losing lateralisation should make the six position patterns LESS distinguishable
  from one another -- a decoder failing and patterns converging are the same statement seen from two
  sides. Measured as the 6x6 between-position structure, pre vs post.

      NOISE-UNBIASED, and this is not optional here. A plain correlation RDM is biased by trial noise,
      and the post-stroke response changes in both spatial EXTENT and trial count -- so a raw metric
      would move even if the geometry were identical. (An earlier note here said amplitude rose 2-3x;
      that summed measure conflated amplitude with spatial spread and is withdrawn -- peak amplitude
      rises only at close_L/close_center and FALLS at the far positions.) Crossnobis (cross-validated Mahalanobis) removes that bias, which is exactly why the
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


def _area_matrix(s, align="cue", source="roi", post_all_trials=True):
    """(trials x areas) with bins collapsed, plus labels, block ids and the area-name list.
    POST-STROKE SESSIONS USE ALL TRIALS (Priya, 2026-08-18). The missing licks ARE the phenotype, so
    discarding no-lick trials removes the effect being measured. This is not a small correction
    post-stroke: PS94 8/18 is only 40% engaged, so an engaged-only version reads a minority subset
    selected by the behaviour the lesion disrupted. Pre-stroke keeps the engaged cut (the mismatch is
    declared in nolick_analysis.SANCTIONED_MISMATCHES). An earlier version filtered BOTH sides to
    engaged trials, contradicting the decision it was written under.
   """
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import _trial_features

    names = _names(s)
    if names is None:
        return None
    X, y, g, Xn, yn, reg = _trial_features(s, _args(source=source, align=align, post_s=2.0))
    # post-stroke: fold the NO-LICK trials back in (see the note above)
    if (post_all_trials
            and config.session_phase(s["label"][:4], s["label"].split("_")[-1]) == "post"
            and len(yn)):
        X = np.vstack([X, Xn])
        y = np.concatenate([y, yn])
        g = np.concatenate([g, np.arange(g.max() + 1, g.max() + 1 + len(yn))])

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


#: How far the shift toward the mirrored pattern must exceed the pre-stroke baseline before it is
#: called anything. A symmetric brain already has substantial mirror correlation, so the raw
#: ordering carries almost no information on its own.
MIRROR_MARGIN = 0.15

#: A correlation below this means the post-stroke pattern resembles neither its own pre-stroke
#: pattern nor the mirrored one. "Which hemisphere does it look like" is then not a question with an
#: answer, and both flags are withheld in favour of `pattern_lost`.
MIN_RESEMBLANCE = 0.20


def _flag_mirror(rec, resemblance_floor=None):
    """Set `transfer` and `reduced_asymmetry` on one position's mirror record.

    ONE FUNCTION so the live run and any re-scoring of a saved JSON cannot drift apart.

    A MARGIN IS REQUIRED, not just an ordering. `mirror_r > normal_r` alone flagged TRANSFER for
    PS94 8/18 far_center on normal +0.677 against mirror +0.682 -- a 0.005 correlation difference,
    against a pre-stroke baseline difference of -0.015. Calling that a pattern relocating across the
    midline would be the same overclaiming this project has spent the week removing.

    TRANSFER means the pattern now resembles the OPPOSITE hemisphere's more than its own, so mirror
    must actually exceed normal AND the shift must beat the pre-stroke baseline by MIRROR_MARGIN. An
    even earlier version required only the shift, and so labelled PS94 close_R "transfer" when normal
    was +0.718 against mirror +0.323 -- the pattern had not moved anywhere, it had merely become less
    asymmetric. REDUCED ASYMMETRY is that weaker, different claim and keeps its own flag.
    """
    d = rec["mirror_minus_normal"] - rec.get("pre_mirror_minus_normal", float("nan"))
    # THE PATTERN HAS TO RESEMBLE SOMETHING before "where does it resemble" is a question at all.
    # PS94 far_center, cue-aligned: normal_r -0.632 and mirror_r -0.480 on 8/17, -0.100 and +0.043 on
    # 8/18. The post-stroke pattern is ANTI-correlated with its own pre-stroke pattern and barely
    # correlated with the mirrored one -- it resembles neither -- yet mirror beat normal by more than
    # the margin, so it was flagged TRANSFER on both days. The honest reading is that the
    # representation at that position is GONE, which is a different and stronger claim than
    # relocation, and the two must not be reported as one.
    # THE FLOOR IS PER FEATURE SPACE, and passing it in is not optional decoration.
    #
    # MIN_RESEMBLANCE = 0.20 was calibrated on ALLEN-ROI correlations, where a pattern is 66 numbers
    # each averaged over ~3000 pixels. That averaging smooths, and smoothing inflates correlation:
    # measured on the same sessions, the median post-vs-pre correlation is +0.830 in ROI space and
    # +0.219 in PIXEL space (PS95_0818 close_center: +0.968 vs +0.391). Applying the ROI floor to
    # pixel correlations therefore declares 46% of positions "pattern lost" against 25% in ROI --
    # a difference in the SMOOTHING, reported as a difference in the brain.
    #
    # Callers working in another space must pass a floor derived from that space, the natural one
    # being how well PRE-STROKE sessions resemble each other there.
    floor = MIN_RESEMBLANCE if resemblance_floor is None else float(resemblance_floor)
    rec["resemblance_floor"] = floor
    rec["pattern_lost"] = bool(max(rec["mirror_r"], rec["normal_r"]) < floor)
    rec["transfer"] = bool(rec["mirror_r"] >= floor
                           and rec["mirror_r"] > rec["normal_r"] and d > MIRROR_MARGIN)
    rec["reduced_asymmetry"] = bool(d > MIRROR_MARGIN and not rec["pattern_lost"])
    return rec


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


def session_geometry(s, align="cue", source="roi", post_all_trials=True):
    got = _area_matrix(s, align, source, post_all_trials)
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
        # TRANSFER means the pattern now resembles the OPPOSITE hemisphere's more than its own, i.e.
        # mirror_r must actually exceed normal_r. The first version flagged any shift of the
        # difference toward mirror by >0.15 and so labelled PS94 close_R "transfer" when normal was
        # +0.718 against mirror +0.323 -- the pattern had not moved anywhere, it had merely become
        # less asymmetric. Those are different claims and only the first deserves the word.
        _flag_mirror(out[p])
    return out


def collect(animals=None, align="cue", source="roi", post_all_trials=True):
    rows = []
    for s in config.analysis_sessions(animals=animals):
        try:
            r = session_geometry(s, align, source, post_all_trials)
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


def _mean_distance_over(row, positions):
    """Mean crossnobis distance over the PAIRS formed by `positions`, from a stored matrix.

    Recomputed rather than read off `mean_distance`, so a post session scored on four positions is
    compared with pre sessions scored on the SAME four. Returns NaN if the row lacks any of them.
    """
    have = list(row.get("positions") or [])
    M = np.asarray(row.get("crossnobis"), float) if row.get("crossnobis") is not None else None
    if M is None or not positions:
        return float("nan")
    try:
        idx = [have.index(p) for p in positions]
    except ValueError:
        return float("nan")
    sub = M[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), k=1)
    vals = sub[iu]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


def summarise(rows):
    out = {}
    for a in sorted({r["animal"] for r in rows}):
        pre = [r for r in rows if r["animal"] == a and r["phase"] == "pre"]
        post = [r for r in rows if r["animal"] == a and r["phase"] == "post"]
        if len(pre) < 3 or not post:
            out[a] = {"note": f"{len(pre)} pre / {len(post)} post"}
            continue
        rec = {"n_pre": len(pre)}
        rec["convergence"] = {"post": {}}
        for r in post:
            # THE PRE-STROKE BAND IS REBUILT ON THIS SESSION'S OWN POSITION SET.
            #
            # `mean_distance` is the mean over the position PAIRS a session has: 15 pairs for six
            # positions, 6 for four. On the LICK-ONLY arm the post session keeps only the positions
            # it still licks at -- PS94 has four, PS95 had five on 8/17 -- so scoring it against a
            # band computed over six positions compares a mean over one pair set with a mean over
            # another. That is not a smaller number because the code converged; it is a different
            # quantity. The same error class as the decoding arms' chance level moving with
            # behaviour, arriving here through the pair set instead.
            pos = list(r.get("positions") or [])
            v = np.array([_mean_distance_over(q, pos) for q in pre], float)
            v = v[np.isfinite(v)]
            if len(v) < 3:
                continue
            band = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max()), "n": int(len(v)),
                    "positions": pos}
            x = _mean_distance_over(r, pos)
            z = (x - band["mean"]) / band["sd"] if band["sd"] else np.nan
            rec["convergence"]["post"][r["label"]] = {
                "mean_distance": float(x), "z": float(z), "pre_band": band,
                "n_positions": len(pos),
                # LOWER distance = those positions became less distinguishable from one another
                "converged": bool(x < band["min"])}
        if rec["convergence"]["post"]:
            # kept for figures that want one band per animal; only meaningful when every post
            # session scored the same positions
            bands = [v["pre_band"] for v in rec["convergence"]["post"].values()]
            if all(b["positions"] == bands[0]["positions"] for b in bands):
                rec["convergence"]["pre_band"] = bands[0]
        # The mirror test needs the raw per-position PATTERNS, which the saved JSON drops (they are
        # large). Skipped rather than crashed when re-scoring a saved file, so the convergence half
        # can be recomputed from the stored matrices without a 40-minute re-run.
        if all("pattern" in r for r in post) and any("pattern" in r for r in pre):
            rec["mirror"] = {r["label"]: mirror_test(pre, r) for r in post}
        out[a] = rec
    return out


def plot(rows, summ, out_dir, align="cue"):
    """One column per POST-STROKE SESSION.

    Was one column per ANIMAL, which drew `mirror[labs[0]]` -- the first post-stroke session only.
    Invisible while each animal had one; with two it silently hid day 2. The convergence panel also
    read a single per-animal `pre_band`, which no longer exists once sessions score different
    position sets (the lick-only arm), so each session now carries and draws its OWN matched band.
    """
    cols = []
    for a in sorted(summ):
        rec = summ[a]
        if "note" in rec:
            continue
        for lab in sorted(rec.get("convergence", {}).get("post", {})):
            cols.append((a, lab, rec))
    if not cols:
        return None
    POS = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    fig, axes = plt.subplots(2, len(cols), figsize=(4.4 * len(cols), 8.6), squeeze=False)
    for k, (a, lab, rec) in enumerate(cols):
        v = rec["convergence"]["post"][lab]
        b = v["pre_band"]
        ax = axes[0][k]
        ax.axhspan(b["min"], b["max"], color="tab:blue", alpha=0.18)
        ax.axhline(b["mean"], color="tab:blue", lw=1.8)
        ax.plot(0, v["mean_distance"], "o", ms=12, color="tab:red", markeredgecolor="k")
        ax.text(0.06, v["mean_distance"], f"  z{v['z']:+.1f}", fontsize=9, va="center",
                color=("firebrick" if v["converged"] else "dimgrey"))
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.9)
        ax.set_title(f"{lab} - crossnobis distance ({v['n_positions']} positions, band matched)",
                     fontsize=8.5)
        if k == 0:
            ax.set_ylabel("mean pairwise distance (LOWER = converged)", fontsize=8)

        ax = axes[1][k]
        m = rec.get("mirror", {}).get(lab, {})
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
                ax.plot([xi - 0.45, xi + 0.45], [dd, dd], ls="--", color="grey", lw=1)
        # A pattern that resembles NEITHER is not evidence about which hemisphere it resembles.
        # Marked on the figure rather than left to the reader, because the flag rule got this
        # wrong twice before landing on a third verdict.
        for xi, pos_name in zip(x, POS):
            if m.get(pos_name, {}).get("pattern_lost"):
                ax.text(xi, -0.97, "pattern lost", ha="center", va="bottom", fontsize=5.5,
                        rotation=90, color="firebrick", fontweight="bold")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(POS, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(-1, 1.05)
        ax.set_title(f"{lab}: mirror test", fontsize=9)
        if k == 0:
            ax.set_ylabel("correlation with pre-stroke pattern", fontsize=8)
            ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle(
        f"SPATIAL REORGANISATION, {align}-aligned. TOP: between-position CROSSNOBIS distance -- "
        "noise-unbiased, required here because post-stroke sessions differ in trial count AND in "
        "response extent, so a plain correlation RDM would move on that alone. LOWER distance = the "
        "positions became less distinguishable, which is the same statement as a decoder failing. "
        "The band is that animal's pre-stroke range recomputed over THIS session's positions, "
        "because mean distance averages over PAIRS and a four-position session has six of them "
        "against fifteen. BOTTOM: does each post-stroke pattern resemble its OWN pre-stroke pattern "
        "(blue) or the HEMISPHERE-SWAPPED one (orange)? Orange above blue would mean the pattern "
        "relocated across the midline. Dashed grey = the pre-stroke mirror-minus-normal baseline, "
        "because a symmetric brain already has substantial mirror correlation and only an excess "
        "over it means anything. Where NEITHER bar is appreciably positive the pattern resembles "
        "nothing and is marked PATTERN LOST -- a different and stronger claim than relocation.",
        fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
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
    ap.add_argument("--lick-only", action="store_true",
                    help="score the post arm on ENGAGED trials only (default: ALL trials). Both arms "
                         "belong in the deck: crossnobis whitens by the WITHIN-position residual "
                         "covariance, so folding the no-lick trials in makes each position more "
                         "heterogeneous, inflates that covariance and shrinks every distance even "
                         "where the code is intact. The lick-only arm is the control for exactly that.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    arm = "lickonly" if args.lick_only else "all"
    for align in args.align:
        print(f"\n=== {align}-aligned "
              f"[{'LICK-ONLY' if args.lick_only else 'ALL-trials'} post arm] ===", flush=True)
        rows = collect(args.animals, align=align, source=args.source,
                       post_all_trials=not args.lick_only)
        if not rows:
            continue
        summ = summarise(rows)
        for a, rec in summ.items():
            if "note" in rec:
                print(f"  {a}: {rec['note']}")
                continue
            c = rec.get("convergence")
            if c:
                for lab, v in c["post"].items():
                    # each session's OWN matched band, not the per-animal one: that key is only set
                    # when every post session scored the SAME positions, which the lick-only arm
                    # breaks by construction. plot() was moved to the per-session band; this summary
                    # print was not, so the whole step died with KeyError('pre_band') on that arm --
                    # after the all-trials arm had already rewritten its half of the figures.
                    b = v["pre_band"]
                    print(f"  {a} CONVERGENCE {lab}: {v['mean_distance']:.4f} vs pre "
                          f"{b['mean']:.4f}[{b['min']:.4f},{b['max']:.4f}] z={v['z']:+.1f}"
                          f"{'  CONVERGED' if v['converged'] else ''}")
            for lab, m in rec.get("mirror", {}).items():
                tr = [p for p, v in m.items() if v.get("transfer")]
                ra = [p for p, v in m.items() if v.get("reduced_asymmetry")]
                print(f"  {a} MIRROR {lab}: TRANSFER (mirror>normal) at {tr or 'no position'}; "
                      f"reduced asymmetry at {ra or 'no position'}")
                for p, v in m.items():
                    print(f"       {p:13s} normal {v['normal_r']:+.3f}  mirror {v['mirror_r']:+.3f}"
                          f"  diff {v['mirror_minus_normal']:+.3f}   (pre-stroke baseline diff "
                          f"{v['pre_mirror_minus_normal']:+.3f})")
        json.dump({"align": align, "summary": summ,
                   "sessions": [{k: v for k, v in r.items() if k != "pattern"} for r in rows]},
                  open(Path(out) / f"spatial_reorganisation_{align}_{arm}.json", "w"), indent=1,
                  default=float)
        p = plot(rows, summ, out, align + ("_lickonly" if args.lick_only else ""))
        if p:
            print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

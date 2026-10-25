"""SIGNIFICANCE of the frozen-encoder RESIDUAL maps per epoch (Priya 10-24: "can we measure significance in the residual
maps in each epoch?").

The residual (data - full) of one session = the part of its evoked response the frozen pre-stroke model does not
predict from that session's behaviour. Pre-stroke sessions are scored leave-one-session-out, so THEY HAVE A RESIDUAL
TOO, and it drifts (FINDINGS 3.4: late pre - early pre +5..+12 x1e3 in SSp-m, 3/4 animals). So the test is NOT
"residual != 0" but "this epoch's residual maps differ from the pre-stroke sessions' residual maps":

    post_by_animal = that epoch's per-session residual maps;  pre_by_animal = the pre sessions' residual maps
    tests = the repo's map tests (`beta_maps.significance_contour`): "nested" (animals -> sessions bootstrap on the 8 x 8
            grid, Bonferroni over bins, pre-normalised -- PRIMARY) and "cluster" (cluster mass, within-animal
            relabelling, animal = unit)
    REFERENCE = ALL the pre sessions, each scored leave-one-session-out (Priya 10-24: "i don't want to use early vs late
            pre - just all loso pre"). The late-vs-early-pre calibration is optional (--drift), off by default.

Per-session maps are rebuilt exactly as `frozen_pixel_maps.py` (evoked parts saved by the basis run, reconstructed
through each animal's shared basis on the stat mask). Image left = ipsilesional.

PER ANIMAL (--per-animal, Priya 10-24 "are we testing residuals within animal?"): the same tests called with ONE animal,
so the nested bootstrap resamples only that animal's sessions (epoch sessions vs pre sessions, Bonferroni over bins) and
the cluster test relabels sessions within it. The nested bootstrap needs >= 2 animals (it returns nothing for one), so
per animal the bin test is the repo's "musall" method (sessions as the unit, Welch t per bin, corrected over bins).
Rule 8: per animal first.

Usage: python scripts/frozen_residual_significance.py --tag gate-attempted_v7g3_t3d_basis [--spout far_contra] [--per-animal]
Output: <E>/frozen_encoder_<tag>/figures/residual_significance_<variant>_<spout>_<align>.png + residual_significance.csv
"""
from __future__ import annotations

import argparse
import json
import re
import pickle
import sys
from pathlib import Path

import numpy as np

EPOCHS = ("acute", "subacute", "chronic")


def session_residual_maps(out: Path, variant: str, spout: str, align: str):
    """{animal: [(label, epoch, day, (H, W) residual map, NaN off the stat mask)]}, H, W."""
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    sess = sorted(js for js in (out / variant / "sessions").glob("*.json") if js.stem[-4:] >= "0801")
    _V, _ft, _Ub, flat, _parea, _names, shape, pstat = ce.load_imaging(sess[0].stem)
    H, W = (int(x) for x in shape)
    pix = flat[pstat]
    spatial: dict = {}
    by: dict = {}
    for js in sess:
        m = json.loads(js.read_text(encoding="utf-8"))
        if spout != "all" and spout not in m["pos_order"]:
            continue
        an = m["animal"]
        if an not in spatial:
            spatial[an] = pickle.loads((out / "fit" / f"spatial_{an}.pkl").read_bytes())
        z = np.load(js.with_suffix(".npz"))
        if spout == "all":                                 # mean over the session's spouts (Priya 10-24: "all 6 pooled")
            ev_all = z[f"evoked_{align}"][:, :, :-1].astype(np.float64)
            keep = [i for i in range(ev_all.shape[1]) if np.isfinite(ev_all[:, i]).all()]
            if len(keep) < 4:
                continue
            ev = ev_all[:, keep].mean(1)
        else:
            ev = z[f"evoked_{align}"][:, m["pos_order"].index(spout), :-1].astype(np.float64)
        if not np.isfinite(ev).all():
            continue
        names = m["evoked_parts"]
        v = ev[names.index("data")] - ev[names.index("full")]
        sp = spatial[an]
        if sp["kind"] != "basis":
            raise ValueError("basis targets only")
        g = np.full(H * W, np.nan)
        g[pix] = np.asarray(sp["P"], np.float64) @ v
        by.setdefault(an, []).append((m["label"], m["epoch"], epochs.days_since_stroke(m["label"]), g.reshape(H, W)))
    return by, H, W


def run_tests(pre_by, post_by, n_boot=2000, n_perm=500, methods=("nested", "cluster")):
    from wfield_local import beta_maps as bm
    out = {}
    for method in methods:
        keep, lab = bm.significance_contour(pre_by, post_by, method=method, n_boot=n_boot, n_perm=n_perm)
        m_ = re.search(r"(\d+) of (\d+) bins", lab or "")
        out[method] = {"mask": keep, "label": lab,
                       "sig_bins": int(m_.group(1)) if m_ else (0 if method != "cluster" else None),
                       "n_bins": int(m_.group(2)) if m_ else None,
                       "px": int(keep.sum()) if keep is not None else 0}
    return out


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from wfield_local import beta_maps as bm
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_basis")
    ap.add_argument("--variant", default="joint")
    ap.add_argument("--spout", default="far_contra", help="a spout, or 'all' = mean over the session's spouts")
    ap.add_argument("--aligns", nargs="+", default=["lick", "cue"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--per-animal", action="store_true", help="one test per animal (sessions as the resampled unit)")
    ap.add_argument("--drift", action="store_true", help="also test late vs early pre (off by default)")
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}"
    fd = out / "figures"
    assert_writable(fd)
    fd.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from frozen_area_maps import area_image, outline
    rows = []
    for al in a.aligns:
        by, H, W = session_residual_maps(out, a.variant, a.spout, al)
        img, _nm = area_image(next(iter(by.values()))[0][0])
        ol = outline(img)
        pre = {an: sorted([s for s in v if s[1] == "pre"], key=lambda s: s[2]) for an, v in by.items()}
        comps = {}
        h = {an: len(p) // 2 for an, p in pre.items()}
        if a.drift:
            comps["drift: late pre vs early pre"] = ({an: [s[3] for s in p[:h[an]]] for an, p in pre.items() if h[an] >= 2},
                                                     {an: [s[3] for s in p[h[an]:]] for an, p in pre.items() if h[an] >= 2})
        for ep in EPOCHS:
            post = {an: [s[3] for s in v if s[1] == ep] for an, v in by.items()}
            common = [an for an in by if post[an] and pre[an]]
            comps[f"{ep} vs pre"] = ({an: [s[3] for s in pre[an]] for an in common}, {an: post[an] for an in common})
        if a.per_animal:
            for name, (pb, qb) in comps.items():
                for an in sorted(qb):
                    if len(pb.get(an, [])) < 2 or len(qb[an]) < 2:
                        rows.append({"align": al, "comparison": name, "animal": an, "note": "< 2 sessions in a group"})
                        continue
                    r = run_tests({an: pb[an]}, {an: qb[an]}, a.n_boot, a.n_perm, methods=("musall", "cluster"))
                    rows.append({"align": al, "comparison": name, "animal": an, "n_pre": len(pb[an]),
                                 "n_post": len(qb[an]), "musall_sig_bins": r["musall"]["sig_bins"],
                                 "musall_n_bins": r["musall"]["n_bins"], "cluster_px": r["cluster"]["px"],
                                 "musall_label": r["musall"]["label"], "cluster_label": r["cluster"]["label"]})
                    print(f"[residual sig] {al} {name} {an}: sessions-as-unit {r['musall']['sig_bins']}/"
                          f"{r['musall']['n_bins']} bins, cluster {r['cluster']['px']} px", flush=True)
            continue
        res = {}
        for name, (pb, qb) in comps.items():
            print(f"[residual sig] {al} {name}: animals {sorted(qb)}, sessions {[len(qb[x]) for x in sorted(qb)]}",
                  flush=True)
            r = run_tests(pb, qb, a.n_boot, a.n_perm)
            res[name] = (pb, qb, r)
            rows.append({"align": al, "comparison": name, "n_animals": len(qb),
                         "nested_sig_bins": r["nested"]["sig_bins"], "nested_n_bins": r["nested"]["n_bins"],
                         "nested_label": r["nested"]["label"], "cluster_px": r["cluster"]["px"],
                         "cluster_label": r["cluster"]["label"]})
            print(f"   nested: {r['nested']['label']}\n   cluster: {r['cluster']['label']}", flush=True)
        # figure: mean (animals first, pre-normalised) of post - pre residual, with both contours
        fig, axs = plt.subplots(1, len(res), figsize=(3.6 * len(res), 3.7), squeeze=False)
        for j, (name, (pb, qb, r)) in enumerate(res.items()):
            sc = bm.pre_scale(pb)
            d = [np.nanmean(bm.scale_maps(qb, sc)[an], 0) - np.nanmean(bm.scale_maps(pb, sc)[an], 0) for an in qb]
            mm = np.nanmean(d, 0)
            lim = np.nanpercentile(np.abs(mm), 98) or 1.0
            ax = axs[0][j]
            ax.imshow(mm, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.4, interpolation="nearest")
            for meth, col in (("nested", "k"), ("cluster", "#7a0177")):
                mk = r[meth]["mask"]
                if mk is not None and mk.any():
                    ax.contour(mk.astype(float), levels=[0.5], colors=col, linewidths=1.0 if meth == "nested" else 0.8,
                               linestyles="solid" if meth == "nested" else "dashed")
            nb = r["nested"]
            ax.set_title(f"{name}\nnested {nb['sig_bins'] or 0}/{nb['n_bins'] or '-'} bins; cluster {r['cluster']['px']:,} px",
                         fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Frozen-encoder RESIDUAL (data - full), {a.spout}, {al} window: epoch residual minus pre-stroke "
                     f"residual,\npre-normalised, mean of animals. Contours: black = nested bootstrap (primary), purple "
                     f"dashed = cluster.\nReference = all pre sessions (LOSO). Image left = ipsilesional.", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(fd / f"residual_significance_{a.variant}_{a.spout}_{al}.png", dpi=130)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(out / f"residual_significance{'_per_animal' if a.per_animal else ''}.csv", index=False)
    print(f"-> {fd} (residual_significance_*.png), {out / 'residual_significance.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

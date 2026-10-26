"""Is the post-stroke factor shape change bigger than pre-stroke DRIFT? (FR handoff 10-25 night: the cohort preA / preB
floor interleaves days, so it measures fit noise only, 3-8 deg -- too lenient a floor for lick_right's 23-27 deg.)

Input: `fr_factors.py fitcohort` tasks 31-40 (DRIFT_GROUPS) -> ``<set>/own_epoch_cohort/<tag>_preEarly.npz`` /
``_preLate.npz``: COHORT and LOAO-<animal> fits on the FIRST / SECOND half of each animal's pre days (in time), the same
rows per animal as the epoch fits. Per tag and master factor k (best spatial match, as `face_rhythm_epoch_shape.py`):
  noise   angle(preA, preB)          two fits of interleaved pre days
  drift   angle(preEarly, preLate)   two fits of pre days ~1-2 weeks apart: noise + slow drift
  epoch   angle(epoch, preLate)      the post-stroke fit vs the LATEST pre half (closest in time; drift up to the stroke
                                     is then in the floor, not in the change)
plus loading asymmetry change (epoch - preLate) against |preEarly - preLate|. COHORT +- jackknife SE over the LOAO fits.

    python scripts/face_rhythm_drift_floor.py [--set master_pre_v1]
-> ``<readouts>/<set>/epoch_shape_cohort/``: drift_floor.png, data/drift_floor.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_rhythm_epoch_shape import ASSIGN, MATCH_ON, MIN_COS, deg, jackknife_se  # noqa: E402

from wfield_local.face_rhythm import common as frc  # noqa: E402

OWN = "own_epoch_cohort"
GROUPS = ("preA", "preB", "preEarly", "preLate", "acute", "subacute", "chronic")


def compute(fdir: Path) -> pd.DataFrame:
    from wfield_local import face_rhythm_matching as fm
    from wfield_local import face_rhythm_readouts as frr
    M = np.load(fdir / "master.npz")
    A_ref, B_ref = M["spatial"].reshape(-1, M["spatial"].shape[-1]), M["spectral"]
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    tags = sorted({f.stem.rsplit("_", 1)[0] for f in (fdir / OWN).glob("*_preEarly.npz")})
    rows = []
    for tag in tags:
        fits = {}
        for g in GROUPS:
            f = fdir / OWN / f"{tag}_{g}.npz"
            if f.exists():
                z = np.load(f)
                fits[g] = (z["A_own"].astype(float), z["B_own"].astype(float))
        P = fm.pairwise(fits, A_ref, B_ref, on=MATCH_ON, assign=ASSIGN)
        cos = {(r.key_a, r.key_b, r.factor): (r.cos_spatial, r.cos_spectral) for r in P.itertuples()}
        get = lambda a, b, k, cos=cos: cos.get((a, b, k), cos.get((b, a, k), (np.nan, np.nan)))  # noqa: E731
        asym, mcos = {}, {}
        for g, (A, B) in fits.items():
            mm = fm.match(A_ref, B_ref, A, B, on=MATCH_ON)
            order = fm.columns(mm, ASSIGN)
            mcos[g] = fm.cosines(A_ref, B_ref, A, B)[0][np.arange(len(order)), order]
            a = fm.loading_asymmetry(A, home, mid)
            asym[g] = np.array([a[o] if o >= 0 else np.nan for o in order])
        for k in range(A_ref.shape[1]):
            ok_pre = min(mcos[g][k] for g in ("preA", "preB", "preEarly", "preLate")) >= MIN_COS
            for ep in ("acute", "subacute", "chronic"):
                row = {"tag": tag, "factor": k, "epoch": ep,
                       "ok": bool(ok_pre and mcos[ep][k] >= MIN_COS)}
                for i, mode in enumerate(("spatial", "spectral")):
                    row.update({f"noise_{mode}": deg(get("preA", "preB", k)[i]),
                                f"drift_{mode}": deg(get("preEarly", "preLate", k)[i]),
                                f"epoch_{mode}": deg(get(ep, "preLate", k)[i]),
                                f"epoch_early_{mode}": deg(get(ep, "preEarly", k)[i])})
                row.update({"asym_change": asym[ep][k] - asym["preLate"][k],
                            "asym_drift": asym["preLate"][k] - asym["preEarly"][k]})
                rows.append(row)
    return pd.DataFrame(rows)


def figure(T: pd.DataFrame, names: list[str], out: Path, set_name: str, factors) -> None:
    from wfield_local import face_rhythm_figures as frf
    ecol = frc.epoch_colors()
    eps = list(frc.POST_EPOCHS)
    with frf.style():
        fig, axs = frf.subplots(len(factors), 3, figsize=(12.5, 2.6 * len(factors)), squeeze=False)
        for r_i, k in enumerate(factors):
            d = T[T.factor == k]
            for c_i, mode in enumerate(("spatial", "spectral", "asym")):
                ax = axs[r_i, c_i]
                if mode == "asym":
                    labs = ["drift\nlate - early", *[e[:5] for e in eps]]
                    curve = lambda t, d=d: np.array([d[d.tag == t].asym_drift.iloc[0],  # noqa: E731
                                                     *[d[(d.tag == t) & (d.epoch == e)].asym_change.iloc[0] for e in eps]])
                else:
                    labs = ["noise\nA vs B", "drift\nearly vs late", *[e[:5] for e in eps]]
                    curve = lambda t, d=d, mode=mode: np.array([d[d.tag == t][f"noise_{mode}"].iloc[0],  # noqa: E731
                                                                d[d.tag == t][f"drift_{mode}"].iloc[0],
                                                                *[d[(d.tag == t) & (d.epoch == e)][f"epoch_{mode}"].iloc[0]
                                                                  for e in eps]])
                curves = {t: curve(t) for t in sorted(d.tag.unique())}
                x = np.arange(len(labs))
                se = jackknife_se(curves)
                colors = ["0.6"] * (len(labs) - 3) + [ecol[e] for e in eps]
                ax.bar(x, curves["COHORT"], color=colors, width=0.65, yerr=se, capsize=3, error_kw={"lw": 1})
                for t, v in curves.items():
                    if t.startswith("LOAO-"):
                        ax.plot(x, v, "o", ms=2.5, color="k", alpha=0.45, zorder=3)
                if mode != "asym":
                    ax.axhline(curves["COHORT"][1], color="0.3", ls="--", lw=0.8)
                else:
                    ax.axhline(0, color="k", lw=0.5)
                    for s in (-1, 1):
                        ax.axhline(s * abs(curves["COHORT"][0]), color="0.3", ls="--", lw=0.8)
                ax.set_xticks(x, labs, fontsize=7)
                ok = d[d.tag == "COHORT"].ok.all()
                ax.set_title(f"{names[k]}: {mode if mode != 'asym' else 'loading asymmetry change'}"
                             + ("" if ok else "  (unrecovered fit, cos < .5)"), fontsize=8.5)
                ax.set_ylabel("angle (deg)" if mode != "asym" else "(contra - ipsi) / sum")
        fig.suptitle(f"{set_name}: post-stroke shape change vs pre-stroke DRIFT (all animals pooled; dots = each animal "
                     "left out)")
        frf.save(fig, out / "drift_floor.png", """
All-animals-pooled factor fits (`fitcohort`, equal rows per animal; bars +- jackknife SE over animals, dots = the four
leave-one-animal-out fits). Grey bars: NOISE = angle between fits of interleaved pre days (A vs B); DRIFT = angle
between fits of the first and second half of the pre days in time (noise + slow change). Coloured: each epoch's fit vs
the LATER pre half. Dashed = the drift bar: a change above it is more than the face drifts over ~two pre-stroke weeks.
Right: loading asymmetry (cam4 contralesional = image left), epoch - late pre; dashed +- the late - early pre change.""")


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--factors", nargs="+", default=None, help="factor names for the figure (default all)")
    a = ap.parse_args(argv)
    fdir = a.fdir or frc.default_set_dir(set_name=a.set)
    out = frc.readouts_dir(a.set) / "epoch_shape_cohort"
    writeguard.assert_writable(out)
    names = frc.factor_names(fdir)
    T = compute(fdir)
    T["name"] = [names[k] for k in T.factor]
    T.to_csv(figure_layout.sidecar_for(out, "drift_floor", ".csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    C = T[T.tag == "COHORT"]
    for mode in ("spatial", "spectral"):
        P = C.pivot_table(index="name", columns="epoch", values=f"epoch_{mode}").reindex(index=names, columns=list(frc.POST_EPOCHS))
        P.insert(0, "drift", C.groupby("name")[f"drift_{mode}"].first())
        P.insert(0, "noise", C.groupby("name")[f"noise_{mode}"].first())
        n_beyond = T.assign(b=T[f"epoch_{mode}"] > T[f"drift_{mode}"]).groupby(["name", "epoch"]).b.sum().unstack()
        print(f"\n{mode.upper()} angle (deg), COHORT; epoch vs late pre\n{P.round(1).to_string()}")
        print(f"fits (COHORT + 4 LOAO) with epoch angle > their own drift angle:\n"
              f"{n_beyond.reindex(index=names, columns=list(frc.POST_EPOCHS)).to_string()}")
    ks = [names.index(f) for f in a.factors] if a.factors else list(range(len(names)))
    figure(T, names, out, fdir.name, ks)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

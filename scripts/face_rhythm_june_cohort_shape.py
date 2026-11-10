"""June COHORT fit as a ~2-month pre-stroke drift floor for the factor SHAPE change (Priya 11-09; RUNNING_LIST
2026-11-08_a8_june item 1). Inputs: the June set's ``own_epoch_cohort/<tag>_june.npz`` (`face_rhythm_june_fit.py`, O2,
same procedure as the production cohort fits) and the production ``own_epoch_cohort/<tag>_<group>.npz``. Per tag (COHORT
+ 4 LOAO) and master factor k, matched exactly as `face_rhythm_drift_floor.py` (v2 rule, best spatial match):
  noise       angle(preA, preB)            interleaved pre days (fit noise)
  drift       angle(preEarly, preLate)     ~1-2 weeks of August pre (noise + slow drift)
  june_early  angle(june, preEarly)        June -> early August: ~2 months of pre-stroke drift (the new floor)
  june_pre    angle(june, pre)             June -> all August pre
  epoch       angle(epoch, preLate)        post-stroke change (as drift_floor.py)
  epoch_pre   angle(epoch, pre)
A post-stroke change "beyond June" = epoch > june_early (the epoch is 1-7 weeks from late pre, June ~8 weeks from early
pre, so this favours the floor). COHORT +- jackknife SE over the 4 LOAO fits; DESCRIPTIVE (pooled fits).

    python scripts/face_rhythm_june_cohort_shape.py
-> ``<readouts>/master_pre_v1/june_reference/cohort_shape_june/``: june_cohort_shape.png, data/june_cohort_shape.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import face_rhythm_epoch_shape as es  # noqa: E402
from face_rhythm_epoch_shape import MATCH_ON, deg, jackknife_se  # noqa: E402

from wfield_local.face_rhythm import common as frc  # noqa: E402

OWN = "own_epoch_cohort"
GROUPS = ("pre", "preA", "preB", "preEarly", "preLate", "acute", "subacute", "chronic")
TAGS = ("COHORT", "LOAO-PS92", "LOAO-PS93", "LOAO-PS94", "LOAO-PS95")
COLS = ("noise", "drift", "june_early", "june_pre")


def compute(fdir: Path, jdir: Path) -> pd.DataFrame:
    from wfield_local import face_rhythm_matching as fm
    M, A_ref, B_ref, _names = es.reference(fdir)
    rows = []
    for tag in TAGS:
        fits = {}
        for g, f in [(g, fdir / OWN / f"{tag}_{g}.npz") for g in GROUPS] + [("june", jdir / OWN / f"{tag}_june.npz")]:
            z = np.load(f)
            fits[g] = (z["A_own"].astype(float), z["B_own"].astype(float))
        P = fm.pairwise(fits, A_ref, B_ref, on=MATCH_ON, assign=es.rule()["assign"])
        cos = {(r.key_a, r.key_b, r.factor): (r.cos_spatial, r.cos_spectral) for r in P.itertuples()}
        get = lambda a, b, k, cos=cos: cos.get((a, b, k), cos.get((b, a, k), (np.nan, np.nan)))  # noqa: E731
        rec = {g: es.pick(A_ref, B_ref, A, B)[2] for g, (A, B) in fits.items()}
        for k in range(A_ref.shape[1]):
            ok_ref = all(rec[g][k] for g in ("pre", "preA", "preB", "preEarly", "preLate", "june"))
            for ep in frc.POST_EPOCHS:
                row = {"tag": tag, "factor": k, "epoch": ep, "ok": bool(ok_ref and rec[ep][k])}
                for i, mode in enumerate(("spatial", "spectral")):
                    row.update({f"noise_{mode}": deg(get("preA", "preB", k)[i]),
                                f"drift_{mode}": deg(get("preEarly", "preLate", k)[i]),
                                f"june_early_{mode}": deg(get("june", "preEarly", k)[i]),
                                f"june_pre_{mode}": deg(get("june", "pre", k)[i]),
                                f"epoch_{mode}": deg(get(ep, "preLate", k)[i]),
                                f"epoch_pre_{mode}": deg(get(ep, "pre", k)[i])})
                rows.append(row)
    return pd.DataFrame(rows)


def figure(T: pd.DataFrame, names: list[str], ks: list[int], out: Path) -> None:
    from wfield_local import cat_points as cp
    from wfield_local import face_rhythm_figures as frf
    ecol = frc.epoch_colors()
    eps = list(frc.POST_EPOCHS)
    labs = ["noise\nA vs B", "drift\nAug early\nvs late", "June vs\nAug early", *[e[:5] for e in eps]]
    colors = ["0.6", "0.6", "0.55", *[ecol[e] for e in eps]]
    with frf.style():
        fig, axs = frf.subplots(len(ks), 2, figsize=(9.5, 2.7 * len(ks)), squeeze=False)
        for r_i, k in enumerate(ks):
            d = T[T.factor == k]
            for c_i, mode in enumerate(("spatial", "spectral")):
                ax = axs[r_i, c_i]
                curves = {t: np.array([d[d.tag == t][f"{c}_{mode}"].iloc[0] for c in ("noise", "drift", "june_early")]
                                      + [d[(d.tag == t) & (d.epoch == e)][f"epoch_{mode}"].iloc[0] for e in eps])
                          for t in TAGS}
                se = jackknife_se(curves)
                loao = [t for t in TAGS if t.startswith("LOAO-")]
                for i in range(len(labs)):
                    cp.session_dots(ax, i, [curves[t][i] for t in loao], [t[5:] for t in loao], colors[i], seed=i)
                    cp.summary_point(ax, i, curves["COHORT"][i], se[i], color=colors[i])
                ax.axhline(curves["COHORT"][2], color="0.4", ls="--", lw=0.8)
                ax.set_xticks(range(len(labs)), labs, fontsize=7)
                ax.set_xlim(-0.6, len(labs) - 0.4)
                ok = d[d.tag == "COHORT"].ok.all()
                ax.set_title(f"{names[k]}: {mode}" + ("" if ok else "  (not recovered)"), fontsize=8.5)
                ax.set_ylabel("angle (deg)")
        fig.suptitle("Factor shape change vs a 2-month pre-stroke floor (June fit); all animals pooled, descriptive")
        frf.save(fig, out / "june_cohort_shape.png", """
All-animals-pooled factor fits (`fitcohort`; June = `face_rhythm_june_fit.py`, same procedure: 5500 rows per animal, 8
HALS starts at the master's rank, columns matched to the master). Filled point = the all-animals fit +- jackknife SE over
animals; open markers = the four leave-one-animal-out fits (shape = the animal left out). DESCRIPTIVE (pooled fits; the
per-session tests are the main tests). Gray: NOISE = interleaved August pre days; DRIFT = first vs second half of the
August pre days; JUNE (dark gray, dashed line) = the June (06/06-06/08) fit vs the first August pre half, ~2 months of
pre-stroke drift + training + camera set-up. Colored: each epoch's fit vs the later August pre half. Factors matched to
the master (rule v2, best spatial match). FaceRhythm cam4, master_pre_v1, engaged-trial and between-trial rows.""")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    es.RULE = "v2"
    fdir = frc.default_set_dir(set_name=fro.SET)
    jdir = oi.bundle_dir(PathResolver(), "fr_cam4_june_20261108") / "factors" / f"{fro.SET}_june"
    out = frc.readouts_dir(fro.SET) / "june_reference" / "cohort_shape_june"
    writeguard.assert_writable(out)
    names = es.reference(fdir)[3]
    T = compute(fdir, jdir)
    T["name"] = [names[k] for k in T.factor]
    T.to_csv(figure_layout.sidecar_for(out, "june_cohort_shape", ".csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 240)
    for mode in ("spatial", "spectral"):
        C = T[T.tag == "COHORT"]
        P = C.pivot_table(index="name", columns="epoch", values=f"epoch_{mode}").reindex(index=names,
                                                                                         columns=list(frc.POST_EPOCHS))
        for c in reversed(COLS):
            P.insert(0, c, C.groupby("name")[f"{c}_{mode}"].first())
        P["ok"] = C.groupby("name").ok.all()
        beyond = T.assign(b=T[f"epoch_{mode}"] > T[f"june_early_{mode}"]).groupby(["name", "epoch"]).b.sum().unstack()
        print(f"\n{mode.upper()} angle (deg), COHORT (epoch vs late Aug pre)\n{P.round(1).to_string()}")
        print(f"fits (of 5) with epoch angle > their own June-vs-early-Aug angle:\n"
              f"{beyond.reindex(index=names, columns=list(frc.POST_EPOCHS)).to_string()}")
    ks = [k for k, n in enumerate(names) if n.startswith("lick")]
    figure(T, names, ks, out)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

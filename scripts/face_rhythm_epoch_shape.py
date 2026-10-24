"""Do the factors change SHAPE after the stroke? The cleaner test on POOLED fits (Priya 10-24, FR handoff open item (c);
FINDINGS 6c: per-session own fits are too noisy to read -- pre own-vs-master angles 10-50 deg).

Input: `face_rhythm_factors.py fitepoch` (O2) -> ``<set>/own_epoch/<animal>_<group>.npz``: one rank-R own fit per animal
x {pre, acute, subacute, chronic} on that epoch's sessions pooled, plus two pre fits on DISJOINT alternating halves of
the pre days (preA, preB); every fit has the same number of rows (11,000, ~2 sessions; PS95 acute = its one session,
5,373 rows), so their noise is comparable.

Per animal and master factor k (every fit Hungarian-aligned to the master, `face_rhythm_matching.pairwise`, so factor k
is the same factor in all fits):
  floor   the angle between preA's and preB's factor k -- how different two fits of the SAME pre-stroke behaviour are
  change  the mean of the angles between the epoch's factor k and preA's / preB's
  beyond  the epoch's angle to BOTH halves exceeds the floor (a shape change beyond pre-stroke fit noise)
  loading asymmetry (contra - ipsi) / sum of |loading| (`face_rhythm_matching.loading_asymmetry`): the epoch's value
          minus the mean of preA / preB, against |preA - preB|
Spatial (where on the face) and spectral (how fast) angles separately. ONE fit per animal x epoch: n = 4 values, so the
read-out is per animal (how many of 4 beyond the floor), with no interval -- a bootstrap needs sessions to resample.

    python scripts/face_rhythm_epoch_shape.py [--set master_pre_v1] [--plot-only]
-> ``<dlc>/face_rhythm/cam4/readouts/<set>/epoch_shape/``: epoch_shape_angles.png (per factor: floor and each epoch's
angle, per animal), epoch_shape_asymmetry.png (loading asymmetry change of the lick factors), data/epoch_shape.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wfield_local.face_rhythm import common as frc  # noqa: E402

GROUPS = ("preA", "preB", "pre", "acute", "subacute", "chronic")


def deg(c):
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def compute(fdir: Path) -> pd.DataFrame:
    from wfield_local import face_rhythm_matching as fm
    from wfield_local import face_rhythm_readouts as frr
    M = np.load(fdir / "master.npz")
    A_ref, B_ref = M["spatial"].reshape(-1, M["spatial"].shape[-1]), M["spectral"]
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    rows = []
    animals = sorted({f.stem.split("_")[0] for f in (fdir / "own_epoch").glob("*.npz")})
    for an in animals:
        fits, nrow = {}, {}
        for g in GROUPS:
            f = fdir / "own_epoch" / f"{an}_{g}.npz"
            if f.exists():
                z = np.load(f)
                fits[g] = (z["A_own"].astype(float), z["B_own"].astype(float))
                nrow[g] = int(z["n_rows"])
        P = fm.pairwise(fits, A_ref, B_ref)                      # every pair, aligned to the master's factor order
        cos = {(r.key_a, r.key_b, r.factor): (r.cos_spatial, r.cos_spectral) for r in P.itertuples()}
        get = lambda a, b, k, cos=cos: cos.get((a, b, k), cos.get((b, a, k), (np.nan, np.nan)))  # noqa: E731
        asym = {}
        for g, (A, _B) in fits.items():                          # loading asymmetry on the master-aligned columns
            order = fm.match(A_ref, B_ref, *fits[g])["order"]
            a = fm.loading_asymmetry(A, home, mid)
            asym[g] = np.array([a[o] if o >= 0 else np.nan for o in order])
        R = A_ref.shape[1]
        for k in range(R):
            fl = get("preA", "preB", k)
            for ep in frc.POST_EPOCHS:
                if ep not in fits:
                    continue
                ca, cb = get(ep, "preA", k), get(ep, "preB", k)
                row = {"animal": an, "epoch": ep, "factor": k, "n_rows": nrow[ep]}
                for i, mode in enumerate(("spatial", "spectral")):
                    fa, ea, eb = deg(fl[i]), deg(ca[i]), deg(cb[i])
                    row.update({f"floor_{mode}": fa, f"angle_{mode}": (ea + eb) / 2,
                                f"excess_{mode}": (ea + eb) / 2 - fa, f"beyond_{mode}": bool(min(ea, eb) > fa)})
                pre_as = np.nanmean([asym["preA"][k], asym["preB"][k]])
                row.update({"asym_pre": pre_as, "asym_epoch": asym[ep][k], "asym_change": asym[ep][k] - pre_as,
                            "asym_floor": abs(asym["preA"][k] - asym["preB"][k])})
                rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true", help="redraw from data/epoch_shape.csv")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    fdir = a.fdir or frc.default_set_dir(set_name=a.set)
    out = a.out or frc.readouts_dir(a.set) / "epoch_shape"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    csv = figure_layout.sidecar_for(out, "epoch_shape", ".csv")
    T = pd.read_csv(csv) if a.plot_only else compute(fdir)
    if not a.plot_only:
        T.to_csv(csv, index=False, float_format="%.4g")
    names = frc.factor_names(fdir)
    T["name"] = [names[k] for k in T.factor]
    animals = sorted(T.animal.unique())
    pd.set_option("display.width", 220)
    for mode in ("spatial", "spectral"):
        print(f"\n{mode.upper()} angle, epoch vs the two pre halves minus the pre-half floor (deg); per animal "
              "(rule 8), * = beyond the floor (angle to BOTH halves > floor)")
        T["cell"] = [f"{e:+5.1f}{'*' if b else ' '}" for e, b in zip(T[f"excess_{mode}"], T[f"beyond_{mode}"])]
        print(T.pivot_table(index=["name", "epoch"], columns="animal", values="cell", aggfunc="first")
              .reindex(pd.MultiIndex.from_product([names, frc.POST_EPOCHS])).to_string())
        n = T.groupby(["name", "epoch"])[f"beyond_{mode}"].sum().unstack().reindex(index=names,
                                                                                    columns=list(frc.POST_EPOCHS))
        print(f"animals beyond the floor ({mode}):\n{n.to_string()}")
    T = T.drop(columns="cell")

    acol, ecol = frc.animal_colors(), frc.epoch_colors()
    x = np.arange(4)
    with frf.style():
        fig, axs = frf.subplots(len(names), 2, figsize=(9.5, 2.1 * len(names)), squeeze=False, sharey="col")
        for k, nm in enumerate(names):
            for c_i, mode in enumerate(("spatial", "spectral")):
                ax = axs[k, c_i]
                d = T[T.factor == k]
                for an in animals:
                    da = d[d.animal == an].set_index("epoch").reindex(list(frc.POST_EPOCHS))
                    if da[f"floor_{mode}"].notna().any():
                        y = [da[f"floor_{mode}"].dropna().iloc[0], *da[f"angle_{mode}"].to_numpy()]
                        ax.plot(x, y, ".-", color=acol.get(an, "0.5"), lw=0.9, ms=4, alpha=0.85)
                labels = ["pre A vs B"]
                for ep in frc.POST_EPOCHS:
                    nb = int(d[d.epoch == ep][f"beyond_{mode}"].sum())
                    labels.append(f"{ep[:5]}\n{nb}/{d[d.epoch == ep].animal.nunique()} >")
                ax.set_xticks(x, labels, fontsize=6)
                for e_i, ep in enumerate(frc.POST_EPOCHS, start=1):
                    ax.axvspan(e_i - 0.4, e_i + 0.4, color=ecol[ep], alpha=0.06, lw=0)
                ax.set_title(f"{k} {nm}: {mode}", fontsize=8)
                if c_i == 0:
                    ax.set_ylabel("angle (deg)")
        for an in animals:
            axs[0, -1].plot([], [], ".-", color=acol.get(an, "0.5"), label=an)
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: factor shape, pooled per-epoch fits vs the two pre-day halves (n = 4 animals)")
        frf.save(fig, out / "epoch_shape_angles.png", """
One own fit per animal x epoch on the epoch's sessions pooled (`face_rhythm_factors.py fitepoch`, equal rows per fit),
plus two pre fits on disjoint alternating halves of the pre days. Every fit Hungarian-aligned to the frozen master
(factor k = the master's factor k). First point: the angle between the two pre halves' factor k (the FLOOR: two fits of
the same pre-stroke behaviour); then each epoch's mean angle to the two halves. Lines = animals (repo colours). Under
each epoch: animals whose angle to BOTH pre halves exceeds their floor. Spatial = where on the face (x, y loadings),
spectral = which frequencies. One value per animal: descriptive, no interval (PS95 acute = one session, half the rows).""")

        lick = [k for k, nm in enumerate(names) if nm in ("lick_right", "lick_left")]
        fig, axs = frf.subplots(1, max(len(lick), 1), figsize=(5.0 * max(len(lick), 1), 3.8), squeeze=False,
                                sharey=True)
        for c_i, k in enumerate(lick):
            ax = axs[0, c_i]
            d = T[T.factor == k]
            for an in animals:
                da = d[d.animal == an].set_index("epoch").reindex(list(frc.POST_EPOCHS))
                ax.plot(x, [0.0, *da.asym_change.to_numpy()], ".-", color=acol.get(an, "0.5"), lw=0.9, ms=5)
                fl = da.asym_floor.dropna()
                if len(fl):
                    ax.errorbar(-0.15 + 0.1 * animals.index(an), 0, yerr=float(fl.iloc[0]), color=acol.get(an, "0.5"),
                                capsize=2, lw=0.8)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(x, ["pre", *[e[:5] for e in frc.POST_EPOCHS]])
            ax.set_title(f"{names[k]}: loading asymmetry change", fontsize=10)
        axs[0, 0].set_ylabel("(contra - ipsi) / sum, epoch - pre")
        for an in animals:
            axs[0, -1].plot([], [], ".-", color=acol.get(an, "0.5"), label=an)
        frf.legend_outside(axs[0, -1])
        fig.suptitle(f"{fdir.name}: do the lick factors move across the face? (pooled per-epoch fits)")
        frf.save(fig, out / "epoch_shape_asymmetry.png", """
Loading asymmetry (contra - ipsi) / (contra + ipsi) of |spatial loading| per factor (cam4: contralesional = image
left), each epoch's pooled fit minus the mean of the two pre-half fits; < 0 = the factor moved towards the
ipsilesional face. Bars at pre: +-|preA - preB| per animal, the pre-stroke noise of this number. Descriptive, n = 4.""")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

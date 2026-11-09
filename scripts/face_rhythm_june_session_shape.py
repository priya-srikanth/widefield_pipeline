"""June as a ~2-month pre-stroke drift floor for the PER-SESSION FaceRhythm shape tests (Priya 11-09: "yes" to the June
fits; the per-session test is the main shape test since 11-07, `face_rhythm_session_shape_tests.py`).

Input: the 12 June per-session own fits (`fr_factors.py fitsession --own-inits 8` on O2,
``o2/fr_cam4_june_20261108/factors/master_pre_v1_june/own8/``), matched to the frozen master with the same rule (v2),
scored exactly as the production sessions: loading asymmetry, spatial / spectral angle to the animal's AUGUST pre-mean
vector, side specificity (templates = the animal's own August pre spout maps), recovered (0 / 1).

Two tests, both animal-weighted with within-animal session-label shuffles (no refit), p = (1 + #null beyond) / (1 + n):
  JUNE vs PRE   June as one more "epoch" against the animal's own August pre (`scalars` / `loss` of the main script,
                angles recomputed against the pre mean under every shuffle). Holm within the lick pair (8 tests) and the
                other units.
  EPOCH vs JUNE per session the readout minus its animal's own August pre (the main script's `session_values`), then
                mean over animals of (epoch mean - June mean), June / epoch labels shuffled within animal. Angles and
                loss one-sided (further from pre / fewer recovered than June), asymmetry and side specificity two-sided.
                BEYOND FLOOR = p < .05 and the difference has the sign of the epoch's own change vs pre; Holm within a
                family (lick pair 24 tests, other units) across epochs x measures.
June -> August holds slow drift + the training stage + camera set-up, so it is a conservative floor.

    python scripts/face_rhythm_june_session_shape.py [--perm 10000]
-> ``<readouts>/master_pre_v1/june_reference/session_shape_june/``: june_session_scalars.png, data/june_vs_pre.csv,
data/epoch_vs_june.csv, data/june_session_values.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import face_rhythm_epoch_shape as es  # noqa: E402
import face_rhythm_session_shape_tests as ss  # noqa: E402

from wfield_local.face_rhythm import common as frc  # noqa: E402

JUNE_BUNDLE = "fr_cam4_june_20261108"
JUNE_COLOR = "0.55"
MEASURES = (*ss.SCALARS, "loss")
SIDE = {"angle_spatial": "greater", "angle_spectral": "greater", "loss": "less", "asym": "two", "side_spec": "two"}


def june_rows(fdir: Path, jdir: Path, names: list[str]) -> pd.DataFrame:
    """The main script's `load_sessions` rows for the June own8 fits (epoch "june"); side-specificity templates from
    each animal's own AUGUST pre sessions, as for the production sessions."""
    from wfield_local import face_rhythm_matching as fm
    from wfield_local import face_rhythm_readouts as frr
    import face_rhythm_cohort_null as N
    M, A_ref, B_ref, names_ref = es.reference(fdir)
    assert list(names_ref) == list(names)
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    S = frc.sessions(fdir, "proj")
    tmpl = {}
    for an, g in S[S.epoch == "pre"].groupby("animal"):
        acc = []
        for t in g.itertuples():
            with np.load(t.file, allow_pickle=True) as z:
                have = list(z["power_positions"].astype(str))
                sp = np.full((6, z["speed_response"].shape[1]), np.nan)
                for i, p in enumerate(frc.POSITIONS):
                    if p in have:
                        sp[i] = z["speed_response"][have.index(p)]
                acc.append(sp)
        m = np.nanmean(acc, 0)
        tmpl[an] = m - np.nanmean(m, 0, keepdims=True)
    rows = []
    for f in sorted((jdir / "own8").glob("*.npz")):
        key = f.stem
        an, date = key.split("_")[0], key.split("_")[1]
        with np.load(f, allow_pickle=True) as z:
            A, B = z["A_own"].astype(float), z["B_own"].astype(float)
        m, cols, ok = es.pick(A_ref, B_ref, A, B)
        vec, mag = fm.spatial_view(A, "vector"), fm.spatial_view(A, "magnitude")
        asym = fm.loading_asymmetry(A, home, mid)
        for k, nm in enumerate(names):
            c = int(cols[k])
            srt = np.sort(m["cs_all"][k])[::-1]
            ld = mag[:, c] / mag[:, c].sum()
            rows.append(dict(key=key, animal=an, date=date, epoch="june", unit=nm, recovered=bool(ok[k]),
                             margin=float(srt[0] - srt[1]), vec=ss.unit(vec[:, c]), load=ld,
                             spec=ss.unit(np.abs(B[:, c])), asym=float(asym[c]),
                             side_spec=N.side_spec(ld, tmpl[an], ss.OWN_SIDE[nm]) if nm in ss.OWN_SIDE else np.nan))
    return pd.DataFrame(rows)


def june_vs_pre(T, names, n_perm, rng):
    old = frc.POST_EPOCHS
    frc.POST_EPOCHS = ("june",)
    try:
        Rs = ss.scalars(T, names, n_perm, rng)
        Rl = ss.loss(T, names, n_perm, rng).rename(columns={"delta_recovery": "delta", "p_loss": "p"})
    finally:
        frc.POST_EPOCHS = old
    Rl["measure"] = "loss"
    R = pd.concat([Rs, Rl[["epoch", "measure", "unit", "family", "delta", "p", "n_animals"]]], ignore_index=True)
    return _holm(R)


def _holm(R, col="p", out="p_holm", keys=("family",)):
    from wfield_local import analysis_kit as ak
    R[out] = np.nan
    for _, idx in R.groupby(list(keys)).groups.items():
        ok = [i for i in idx if np.isfinite(R.at[i, col])]
        if ok:
            R.loc[ok, out] = ak.holm(R.loc[ok, col].to_numpy())
    return R


def epoch_vs_june(V, prod, names, n_perm, rng):
    rows = []
    for ep in frc.POST_EPOCHS:
        for m in MEASURES:
            for nm in names:
                d = V[(V.unit == nm) & (V.measure == m) & V.epoch.isin(["june", ep])]
                if d.empty:
                    continue
                groups = [(g.value.to_numpy(float), (g.epoch == "june").to_numpy()) for _, g in d.groupby("animal")]
                obs, p, n_an = ss.shuffle_test(groups, "value", n_perm, rng, SIDE[m])
                pr = prod[(prod.epoch == ep) & (prod.measure == m) & (prod.unit == nm)]
                jm = d[d.epoch == "june"].groupby("animal").value.mean().mean()
                em = d[d.epoch == ep].groupby("animal").value.mean().mean()
                rows.append(dict(epoch=ep, measure=m, unit=nm, family="lick pair" if nm in ss.LICK else "other",
                                 june_delta=jm, epoch_delta=em, epoch_minus_june=obs, p_floor=p, n_animals=n_an,
                                 epoch_delta_prod=float(pr.delta.iloc[0]) if len(pr) else np.nan,
                                 p_holm_prod=float(pr.p_holm.iloc[0]) if len(pr) else np.nan))
    R = _holm(pd.DataFrame(rows), "p_floor", "p_floor_holm")
    same = np.sign(R.epoch_minus_june) == np.sign(R.epoch_delta_prod)
    R["beyond_floor"] = same & (R.p_floor < .05)
    R["beyond_floor_holm"] = same & (R.p_floor_holm < .05)
    return R


YLAB = {**ss.YLAB}


def figure(V, Rj, Rf, prod, names, path):
    import matplotlib.pyplot as plt

    from wfield_local import cat_points as cp
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import sig_marks
    ecol = {**frc.epoch_colors(), "june": JUNE_COLOR}
    eps = ["pre", "june", *frc.POST_EPOCHS]
    off = cp.epoch_offsets(len(eps), step=0.17)
    units = [u for u in ss.ORDER if u in names] + [u for u in names if u not in ss.ORDER]
    with frf.style():
        fig = plt.figure(figsize=(14.5, 2.9 * len(MEASURES)))
        gs = fig.add_gridspec(len(MEASURES), len(units), hspace=0.8)
        for i, m in enumerate(MEASURES):
            us = [u for u in units if len(V[(V.measure == m) & (V.unit == u)])]
            ax = fig.add_subplot(gs[i, :len(us)])
            for u_i, u in enumerate(us):
                for k, ep in enumerate(eps):
                    v = V[(V.unit == u) & (V.epoch == ep) & (V.measure == m)]
                    x = u_i + off[k]
                    cp.session_dots(ax, x, v.value.to_numpy(float), v.animal, ecol[ep], jitter=0.06, seed=11 * u_i + k)
                    if len(v):
                        se = v.value.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else None
                        cp.summary_point(ax, x, v.value.mean(), se, ecol[ep], ms=5)
                    if ep == "pre":
                        continue
                    if ep == "june":
                        r = Rj[(Rj.unit == u) & (Rj.measure == m)]
                        ph = float(r.p_holm.iloc[0]) if len(r) else np.nan
                    else:
                        r = prod[(prod.unit == u) & (prod.measure == m) & (prod.epoch == ep)]
                        ph = float(r.p_holm.iloc[0]) if len(r) else np.nan
                        f = Rf[(Rf.unit == u) & (Rf.measure == m) & (Rf.epoch == ep)]
                        if len(f) and (f.beyond_floor_holm.iloc[0] or f.beyond_floor.iloc[0]):
                            sig_marks.star_top(ax, x, "##" if f.beyond_floor_holm.iloc[0] else "#", color=ecol[ep],
                                               row=1, fontsize=9)
                    if np.isfinite(ph) and ph < .05:
                        sig_marks.star_top(ax, x, "**" if ph < .01 else "*", color=ecol[ep])
            ax.axhline(0, color="0.6", lw=0.8, zorder=0)
            ax.set_xticks(range(len(us)), us, rotation=25, ha="right", fontsize=8)
            ax.set_xlim(-0.6, len(us) - 0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel(YLAB[m], fontsize=8)
            if i == 0:
                hs = [plt.Line2D([], [], color=ecol[e], marker="o", ls="none",
                                 label={"pre": "Aug pre (leave-one-out)", "june": "June (06/06-06/08)"}.get(e, e))
                      for e in eps]
                ax.legend(handles=hs + cp.animal_handles(sorted(V.animal.unique())), fontsize=7, frameon=False,
                          loc="upper left", bbox_to_anchor=(1.0, 1.0))
        sig_marks.reserve_headroom(fig)
        fig.suptitle("master_pre_v1: per-session shape readouts with June as a 2-month pre-stroke drift floor "
                     "(delta vs each animal's own August pre)", y=0.995)
        frf.save(fig, path, """
Per-session FaceRhythm fits (own8, 8 starts) matched to the frozen August-pre master (rule v2), recovered sessions only
(loss row: all sessions). Every value = the session's readout minus its animal's own AUGUST pre mean (angles: angle to
the animal's August pre-mean vector minus its mean leave-one-out pre angle). BLACK = each August pre session against its
animal's other pre sessions (leave-one-out); GRAY = June 06/06-06/08 (3 sessions per animal, fitted on O2 with the
production settings); colored = post epochs. Open markers = sessions (shape = animal), filled = mean +- SEM over
sessions. Top row of marks: * / ** = change vs own August pre, Holm < .05 / .01 (animal-weighted, within-animal
session-label shuffles, 10,000; post epochs = the published main test, June = the same test with June as the epoch,
Holm within the lick pair or the other units). Second row: # = the epoch differs from June (mean over animals of epoch
- June, June / epoch labels shuffled within animal; angles and loss one-sided, asymmetry and side specificity two-sided)
in the direction of its own change, p < .05; ## = also Holm within the family across epochs x measures. Whole-session
fits (all task + between-trial rows as production), no trial gate beyond the production row draw. Contra = right
(lesions left).""")


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, o2_inference as oi, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perm", type=int, default=10000)
    a = ap.parse_args(argv)
    es.RULE = "v2"
    rv = PathResolver()
    fdir = frc.default_set_dir(set_name=fro.SET)
    jdir = oi.bundle_dir(rv, JUNE_BUNDLE) / "factors" / f"{fro.SET}_june"
    out = frc.readouts_dir(fro.SET) / "june_reference" / "session_shape_june"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    side = lambda s: figure_layout.sidecar_for(out, s, ".csv")  # noqa: E731
    pd.set_option("display.width", 230)
    pd.set_option("display.max_rows", 300)
    rng = np.random.default_rng(0)
    T, names, _home = ss.load_sessions(fdir)
    J = june_rows(fdir, jdir, names)
    print(f"June: {J.key.nunique()} sessions; recovered per unit:\n"
          + J.groupby("unit").recovered.agg(["sum", "size"]).reindex(names).to_string())
    T = pd.concat([T, J], ignore_index=True)
    prod = pd.read_csv(frc.readouts_dir(fro.SET) / "session_shape_v2" / "data" / "session_scalars.csv")
    pl = pd.read_csv(frc.readouts_dir(fro.SET) / "session_shape_v2" / "data" / "session_loss.csv")
    pl = pl.rename(columns={"delta_recovery": "delta", "p_loss": "p"}).assign(measure="loss")
    prod = pd.concat([prod, pl[["epoch", "measure", "unit", "family", "delta", "p", "p_holm"]]], ignore_index=True)

    Rj = june_vs_pre(T, names, a.perm, rng)
    Rj.to_csv(side("june_vs_pre"), index=False, float_format="%.4g")
    Rj["cell"] = [f"{d:+.3g} p{p:.3f}{'*' if h < .05 else ''}" for d, p, h in zip(Rj.delta, Rj.p, Rj.p_holm)]
    print("\nJUNE vs own AUGUST PRE (* = Holm < .05):\n"
          + Rj.pivot(index="unit", columns="measure", values="cell").reindex(index=list(names)).to_string())
    V = ss.session_values(T, names)
    V.to_csv(side("june_session_values"), index=False, float_format="%.4g")
    Rf = epoch_vs_june(V, prod, names, a.perm, rng)
    Rf.to_csv(side("epoch_vs_june"), index=False, float_format="%.4g")
    Rf["cell"] = [f"{e:+.3g} (J{j:+.3g}) p{p:.3f}{'##' if h else '#' if b else ''}"
                  for e, j, p, b, h in zip(Rf.epoch_delta, Rf.june_delta, Rf.p_floor, Rf.beyond_floor,
                                           Rf.beyond_floor_holm)]
    for m in MEASURES:
        print(f"\nEPOCH vs JUNE -- {m}  (epoch delta (June delta) p_floor; # beyond floor, ## Holm)\n"
              + Rf[Rf.measure == m].pivot(index="unit", columns="epoch", values="cell")
              .reindex(index=[u for u in names if u in set(Rf[Rf.measure == m].unit)],
                       columns=list(frc.POST_EPOCHS)).to_string())
    figure(V, Rj, Rf, prod, names, out / "june_session_scalars.png")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

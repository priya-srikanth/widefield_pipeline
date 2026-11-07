"""Where does each lick factor's SPATIAL WEIGHTING move after the stroke? Per-SESSION fits as replicates (Priya 11-07:
"didn't we also do per-session FR fits -- could we use those to better compare the spatial loading?").

Input: the per-session own fits (`fr_factors.py fitsession`, 8 random starts) ``<set>/own8/<key>.npz``. Each fit is
matched to the master with the current rule (`face_rhythm_epoch_shape.MATCH_RULES`, default v2: spatial score with a
spectral tie-break, merged twin units, a unit recovered only at spatial cos >= .7 and spectral cos >= .5); a session
whose fit does not recover a unit is left out of that unit's map. Per session and unit: the matched column's loading
magnitude per mesh point as a FRACTION of its total (shape only, as `face_rhythm_epoch_shape.loadings_and_templates`).

Per unit x post epoch: per animal, the mean of its epoch sessions' maps minus the mean of its pre sessions' maps; the map
= the mean over animals with both. Null = the same with each animal's pre and epoch session labels shuffled WITHIN the
animal (counts kept), ``--perm`` times -- no refit, the session fits are fixed. Points are studentised by the null SD;
CLUSTER MASS (|t| > 2, mesh neighbours, sign-separate) is the family-wise map test, the MAX-STATISTIC the point-level
one (`face_rhythm.stats.map_permutation`). Single-session fits are noisy (pre own-vs-master angles 10-50 deg, FINDINGS
6c); the shuffle puts that noise into the null.

    python scripts/face_rhythm_session_loading_perm.py [--set master_pre_v1] [--match-rule v2] [--perm 2000]
-> ``<readouts>/<set>/session_loading_<rule>/``: session_loading_lick.png (lick_right, lick_left),
session_loading_other.png (the other lick / orofacial units), data/session_loading_clusters.csv,
data/session_loading_counts.csv (sessions recovered per animal x epoch x unit).
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

from wfield_local.face_rhythm import common as frc  # noqa: E402

LICK = ("lick_right", "lick_left")


def session_maps(fdir: Path) -> tuple[dict, pd.DataFrame, list[str], np.ndarray]:
    """-> ({(unit, animal, epoch): (n_sessions, P) loading fractions of recovered sessions}, counts, units, home)."""
    from wfield_local import face_rhythm_matching as fm
    M, A_ref, B_ref, names = es.reference(fdir)
    S = frc.sessions(fdir, "proj")
    maps, rows = {}, []
    for t in S.itertuples():
        f = fdir / "own8" / f"{t.key}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=True) as z:
            A, B = z["A_own"].astype(float), z["B_own"].astype(float)
        _m, cols, ok = es.pick(A_ref, B_ref, A, B)
        mag = fm.spatial_view(A, "magnitude")
        for k, nm in enumerate(names):
            rows.append({"unit": nm, "animal": t.animal, "epoch": t.epoch, "recovered": bool(ok[k])})
            if ok[k]:
                v = mag[:, cols[k]]
                maps.setdefault((nm, t.animal, t.epoch), []).append(v / v.sum())
    C = (pd.DataFrame(rows).groupby(["unit", "animal", "epoch"]).recovered.agg(["sum", "size"])
         .rename(columns={"sum": "recovered", "size": "sessions"}).reset_index())
    return {k: np.array(v) for k, v in maps.items()}, C, names, M["home"]


def test(maps: dict, units: list[str], home: np.ndarray, n_perm: int, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    from wfield_local.face_rhythm import stats as frs
    adj = frs.mesh_adjacency(home)
    rng = np.random.default_rng(seed)
    res, rows = {}, []
    animals = sorted({k[1] for k in maps})
    for nm in units:
        for ep in frc.POST_EPOCHS:
            per = [(np.vstack([maps[(nm, an, "pre")], maps[(nm, an, ep)]]), len(maps[(nm, an, "pre")]))
                   for an in animals if (nm, an, "pre") in maps and (nm, an, ep) in maps]
            if len(per) < 2:
                continue
            stat = lambda idx, per=per: np.mean([X[i[n:]].mean(0) - X[i[:n]].mean(0)  # noqa: E731
                                                 for (X, n), i in zip(per, idx)], 0)
            obs = stat([np.arange(len(X)) for X, _ in per])
            null = np.array([stat([rng.permutation(len(X)) for X, _ in per]) for _ in range(n_perm)])
            r = frs.map_permutation(obs, null, adj)
            r.update(mean=obs, n_animals=len(per), n_sessions=sum(len(X) - n for X, n in per),
                     n_pre=sum(n for _, n in per))
            res[(nm, ep)] = r
            for c, (m, pc) in enumerate(zip(r["mass"], r["p_cluster"])):
                sel = r["label"] == c
                rows.append({"unit": nm, "epoch": ep, "cluster": c, "n_points": int(sel.sum()),
                             "sign": "more" if obs[sel].mean() > 0 else "less", "mass": m, "p_cluster": pc,
                             "n_animals": len(per), "n_perm": n_perm})
    return res, pd.DataFrame(rows)


def figure(res: dict, units: list[str], home: np.ndarray, out: Path, fn: str, set_name: str) -> None:
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_readouts as frr
    units = [u for u in units if any((u, ep) in res for ep in frc.POST_EPOCHS)]
    if not units:
        return
    eps = list(frc.POST_EPOCHS)
    lim = float(np.nanpercentile(np.abs(np.concatenate([res[k]["mean"] for k in res if k[0] in units])), 98))
    mid = float(frr.MASTER_MIDLINE_X)
    with frf.style():
        fig, axs = frf.subplots(len(units), 3, figsize=(10.5, 3.1 * len(units)), squeeze=False)
        sc = None
        for r_i, nm in enumerate(units):
            for c_i, ep in enumerate(eps):
                ax = axs[r_i, c_i]
                ax.axis("off")
                r = res.get((nm, ep))
                if r is None:
                    continue
                sc = ax.scatter(home[:, 0], home[:, 1], c=r["mean"], s=4, cmap="RdBu_r", vmin=-lim, vmax=lim, lw=0)
                frf.sign_rings(ax, home, r["sig_cluster"], r["mean"])
                ax.axvline(mid, color="k", lw=0.4)
                ax.invert_yaxis()
                ax.set_aspect("equal")
                k = int((r["p_cluster"] < .05).sum())
                ax.set_title(f"{nm} -- {ep}\n{k} cluster{'s' if k != 1 else ''} p < .05 ({int(r['sig_cluster'].sum())} "
                             f"pts); max-stat {int(r['sig_point'].sum())}\n{r['n_animals']} animals, "
                             f"{r['n_sessions']} vs {r['n_pre']} pre sessions", fontsize=7)
        if sc is not None:
            fig.colorbar(sc, ax=axs, shrink=0.3, label="loading fraction, epoch - pre")
        fig.suptitle(f"{set_name}: spatial weighting per session fit, epoch vs pre (rings = permutation cluster p < .05)",
                     fontsize=11)
        frf.save(fig, out / fn, f"""
Per mesh point on the master face (image LEFT = CONTRALESIONAL on cam4; vertical line = face midline): the change in a
unit's spatial LOADING FRACTION (its loading magnitude at the point / its total), epoch minus pre, from the PER-SESSION
FaceRhythm fits (own8, 8 starts): per animal mean of epoch sessions minus mean of pre sessions, then the mean of
animals. Each session's fit is matched to the master with matching rule {es.RULE} (spatial score + spectral tie-break;
recovered at spatial cos >= .7 and spectral cos >= .5; sessions that do not recover the unit are left out). Red = more
weight at that point after the stroke, blue = less. RINGS = points in clusters with family-wise p < .05 against a
within-animal session-label permutation null (cluster mass over mesh neighbours with |t| > 2; red ring = more, blue =
less); title also gives the point-level max-statistic count and the sessions used.""")


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import figure_layout, writeguard
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--fdir", type=Path, default=None)
    ap.add_argument("--match-rule", choices=sorted(es.MATCH_RULES), default="v2")
    ap.add_argument("--perm", type=int, default=2000)
    a = ap.parse_args(argv)
    es.RULE = a.match_rule
    fdir = a.fdir or frc.default_set_dir(set_name=a.set)
    out = frc.readouts_dir(a.set) / f"session_loading_{a.match_rule}"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    maps, C, units, home = session_maps(fdir)
    C.to_csv(figure_layout.sidecar_for(out, "session_loading_counts", ".csv"), index=False)
    pd.set_option("display.width", 220)
    print("sessions recovered / total, per unit x epoch (summed over animals):")
    print(C.groupby(["unit", "epoch"])[["recovered", "sessions"]].sum()
          .apply(lambda r: f"{r.recovered}/{r.sessions}", axis=1).unstack().reindex(units).to_string())
    res, T = test(maps, units, home, a.perm)
    T.to_csv(figure_layout.sidecar_for(out, "session_loading_clusters", ".csv"), index=False, float_format="%.4g")
    print("\nclusters p < .05:")
    print(T[T.p_cluster < .05].to_string(index=False))
    figure(res, list(LICK), home, out, "session_loading_lick.png", fdir.name)
    figure(res, [u for u in units if u not in LICK], home, out, "session_loading_other.png", fdir.name)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

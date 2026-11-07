"""RHYTHM-BAND movement maps per lick unit and spout (Priya 11-07; proposed with the speed maps: "can we show the speed
maps for the lick left and lick right (and other licking) factors?").

`proj/` stores, per session and spout position, every mesh point's movement POWER per frequency in the response window
(``power_response``, (6 positions, x / y, P, 36 freqs), cue 0..+4 s of engaged trials, bin counts ``power_n_bins``).
Weighting that by a reference unit's frequency profile (master spectral loading, v2 units with the merged twins,
normalised to sum 1) gives each point's movement IN THAT UNIT'S RHYTHM -- no refit, no time courses needed. Mapped per
spout x epoch as each session's log2 band power minus the mean of its animal's own pre sessions, mean over animals.

LIMITS, stated on every figure: (1) a band map separates units by RHYTHM only -- lick_right and lick_left share theirs
(spectral cos .98), so they are ONE map ("lick rhythm"), read by spout (right-spout movement = lick_right's, left-spout =
lick_left's); where a footprint MOVED is the per-session loading map (`face_rhythm_session_loading_perm.py`). (2) The
profiles are broad and overlap (e.g. the 7 Hz and fast-close units both span ~6.5-15 Hz), so neighbouring bands are
correlated. (3) Window = cue 0..+4 s, the only window with stored spectra (the 0..2 s speed maps have no spectra).

STATISTICS = the speed maps' (`face_rhythm_position_maps.py --perm`): within-animal session-label shuffles (no refit),
points studentised by the null SD, CLUSTER MASS (|t| > 2, mesh neighbours, sign-separate) = the map test, max-statistic
= the point test (`face_rhythm.stats.session_shuffle_test`); ``--drift-control`` = the same on early vs late pre.

    python scripts/face_rhythm_band_maps.py [--set master_pre_v1] [--perm 2000] [--drift-control] [--plot-only]
-> ``readouts/<set>/band_maps/``: band_maps_<unit>.png (rows = spouts, cols = epochs), band_maps_driftpre.png,
data/band_maps_perm.csv (every cluster), data/band_maps_units.csv (band of each unit), data/band_maps_cache.pkl.
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

EPOCHS = frc.EPOCHS
# display unit -> reference units whose (normalised) profiles are averaged; the lateral pair shares one rhythm
UNITS = {"lick_rhythm": ("lick_right", "lick_left"), "lick_ongoing_7Hz": ("lick_ongoing_7Hz",),
         "lick_fast_close": ("lick_fast_close",), "lick_slow_late": ("lick_slow_late",),
         "lick_far_nose": ("lick_far_nose",), "slow_jaw": ("slow_jaw",), "prelick_nose": ("prelick_nose",)}


def unit_weights(fdir: Path) -> tuple[dict, pd.DataFrame]:
    """{unit: (F,) weights summing to 1} from the v2 reference spectra, and a table of each unit's band."""
    es.RULE = "v2"
    _, _, B, names = es.reference(fdir)
    fr = np.load(fdir / "master.npz")["freqs"]
    W, rows = {}, []
    for u, members in UNITS.items():
        w = np.mean([B[:, names.index(m)] / B[:, names.index(m)].sum() for m in members], 0)
        W[u] = w / w.sum()
        c = np.cumsum(W[u])
        rows.append({"unit": u, "members": "+".join(members), "peak_hz": fr[W[u].argmax()],
                     "q25_hz": fr[(c > .25).argmax()], "q75_hz": fr[(c > .75).argmax()],
                     "mean_hz": float((W[u] * fr).sum())})
    return W, pd.DataFrame(rows)


def session_band(fdir: Path, W: dict) -> dict:
    """{(unit, animal, label, epoch, position): (P,) band power} per SESSION (split halves pooled bin-weighted)."""
    acc = {}
    for t in frc.sessions(fdir, "proj").itertuples():
        with np.load(t.file) as z:
            if "power_response" not in z.files:
                continue
            pw = z["power_response"].sum(1)                     # (6, P, F): x + y
            nb = z["power_n_bins"][2::2]
            pos = [str(x) for x in z["power_positions"]]
        for k, p in enumerate(pos):
            if nb[k] <= 0 or not np.isfinite(pw[k]).all():
                continue
            for u, w in W.items():
                key = (u, t.animal, t.label, t.epoch, p)
                s0, n0 = acc.get(key, (0.0, 0))
                acc[key] = (s0 + (pw[k] @ w) * nb[k], n0 + nb[k])
    return {k: s / n for k, (s, n) in sorted(acc.items()) if n}


def perm_statistics(V: dict, home: np.ndarray, n_perm: int, drift: bool = False, seed: int = 0):
    from wfield_local.face_rhythm import stats as frs
    adj = frs.mesh_adjacency(home)
    rng = np.random.default_rng(seed)
    res, rows = {}, []
    for u in UNITS:
        for p in frc.POSITIONS:
            lg = {k: np.log2(np.maximum(v, 1e-12)) for k, v in V.items() if k[0] == u and k[4] == p}
            animals = sorted({k[1] for k in lg})
            sess = lambda an, ep, lg=lg: [lg[k] for k in sorted(lg) if k[1] == an and k[3] == ep]  # noqa: E731
            for ep in (["late_pre"] if drift else EPOCHS[1:]):
                if drift:
                    per = [frs.drift_split(np.vstack(sess(an, "pre"))) for an in animals if len(sess(an, "pre")) >= 2]
                else:
                    per = [(np.vstack(sess(an, "pre") + sess(an, ep)), len(sess(an, "pre"))) for an in animals
                           if sess(an, "pre") and sess(an, ep)]
                r = frs.session_shuffle_test(per, adj, n_perm, rng)
                if r is None:
                    continue
                res[(u, ep, p)] = r
                for c, (m, pc) in enumerate(zip(r["mass"], r["p_cluster"])):
                    sel = r["label"] == c
                    rows.append({"unit": u, "position": p, "epoch": ep, "cluster": c, "n_points": int(sel.sum()),
                                 "sign": "up" if r["mean"][sel].mean() > 0 else "down", "mass": m, "p_cluster": pc,
                                 "mean_log2": float(r["mean"][sel].mean()), "n_animals": r["n_animals"],
                                 "n_perm": n_perm})
    return res, pd.DataFrame(rows)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    from wfield_local import face_rhythm_figures as frf
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import face_rhythm_readouts as frr
    from wfield_local import figure_layout, writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=fro.SET)
    ap.add_argument("--bundle", default=fro.BUNDLE)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--drift-control", action="store_true", help="also the early vs late PRE test")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    rv = PathResolver()
    fdir = fro.factor_dir(rv, a.bundle, a.set)
    out = Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "readouts" / a.set / "band_maps"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    M = np.load(fdir / "master.npz")
    home, mid = M["home"], float(frr.MASTER_MIDLINE_X)
    cache = figure_layout.sidecar_for(out, "band_maps_cache", ".pkl")
    if a.plot_only:
        U, R, C, RD, CD = pd.read_pickle(cache)
    else:
        W, U = unit_weights(fdir)
        V = session_band(fdir, W)
        R, C = perm_statistics(V, home, a.perm)
        RD, CD = perm_statistics(V, home, a.perm, drift=True) if a.drift_control else ({}, pd.DataFrame())
        pd.to_pickle((U, R, C, RD, CD), cache)
    U.to_csv(figure_layout.sidecar_for(out, "band_maps_units", ".csv"), index=False, float_format="%.3g")
    C.to_csv(figure_layout.sidecar_for(out, "band_maps_perm", ".csv"), index=False, float_format="%.4g")
    if len(CD):
        CD.to_csv(figure_layout.sidecar_for(out, "band_maps_perm_driftpre", ".csv"), index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(U.round(2).to_string(index=False))
    sig = C[C.p_cluster < .05]
    print("\nclusters p < .05 (points; sign):")
    print(sig.assign(s=lambda d: d.n_points.astype(str) + d.sign.str[0]).groupby(["unit", "position", "epoch"])
          .s.agg(" ".join).unstack("epoch").reindex(columns=list(EPOCHS[1:])).fillna("").to_string())
    if len(CD):
        print("\nDRIFT CONTROL clusters p < .05:")
        print(CD[CD.p_cluster < .05][["unit", "position", "n_points", "sign", "p_cluster"]].to_string(index=False)
              if (CD.p_cluster < .05).any() else "none")

    limits = ("""
LIMITS: a band map separates units by RHYTHM only; lick_right and lick_left share theirs (spectral cos .98), so they are
one map ('lick_rhythm'), read by spout: right-spout rows = lick_right's movement, left-spout rows = lick_left's. Where a
footprint MOVED is the per-session loading map. Unit profiles are broad and overlap, so neighbouring bands are
correlated. Window cue 0..+4 s (the only window with stored spectra).""")
    for u in UNITS:
        b = U[U.unit == u].iloc[0]
        with frf.style():
            fig, axs = frf.subplots(len(frc.POSITIONS), 3, figsize=(10.5, 3.1 * len(frc.POSITIONS)), squeeze=False)
            sc = None
            for r_i, p in enumerate(frc.POSITIONS_DISPLAY):
                for c_i, ep in enumerate(EPOCHS[1:]):
                    ax = axs[r_i, c_i]
                    ax.axis("off")
                    r = R.get((u, ep, p))
                    if r is None:
                        continue
                    sc = ax.scatter(home[:, 0], home[:, 1], c=r["mean"], s=4, cmap="RdBu_r", vmin=-1, vmax=1)
                    frf.sign_rings(ax, home, r["sig_cluster"], r["mean"])
                    ax.axvline(mid, color="k", lw=0.4)
                    ax.invert_yaxis()
                    ax.set_aspect("equal")
                    k = int((r["p_cluster"] < .05).sum())
                    ax.set_title(f"{p} -- {ep}\n{k} cluster{'s' if k != 1 else ''} p < .05 "
                                 f"({int(r['sig_cluster'].sum())} pts); max-stat {int(r['sig_point'].sum())} pts",
                                 fontsize=8)
            if sc is not None:
                fig.colorbar(sc, ax=axs, shrink=0.3, label="log2(epoch / own pre)")
            fig.suptitle(f"{fdir.name}: movement in the {u} rhythm (peak {b.peak_hz:.1f} Hz, middle half "
                         f"{b.q25_hz:.1f}-{b.q75_hz:.1f} Hz),\nper spout vs own pre, mean of animals (rings = cluster "
                         f"p < .05, {a.perm} within-animal shuffles)", fontsize=10)
            frf.save(fig, out / f"band_maps_{u}.png", f"""
Per mesh point on the master face (image LEFT = CONTRALESIONAL on cam4; vertical line = face midline): movement power in
the {u} rhythm ({b.members}: master frequency profile, v2 units, as weights over the 36 stored frequencies) in the cue
0..+4 s window of engaged trials at ONE spout (rows; L = ipsilesional, R = contralesional). Each session's log2 band
power minus the mean of its animal's pre sessions, mean over the animal's epoch sessions, then over animals; clipped at
+-1 (2x). Rings: cluster-mass permutation p < .05 (|t| > 2, mesh neighbours, sign-separate; red = up, blue = down; null
= within-animal session-label shuffles). data/band_maps_perm.csv lists every cluster.{limits}""")
    if RD:
        with frf.style():
            fig, axs = frf.subplots(len(UNITS), 6, figsize=(17, 2.9 * len(UNITS)), squeeze=False)
            sc = None
            for r_i, u in enumerate(UNITS):
                for c_i, p in enumerate(frc.POSITIONS_DISPLAY):
                    ax = axs[r_i, c_i]
                    ax.axis("off")
                    r = RD.get((u, "late_pre", p))
                    if r is None:
                        continue
                    sc = ax.scatter(home[:, 0], home[:, 1], c=r["mean"], s=3, cmap="RdBu_r", vmin=-1, vmax=1)
                    frf.sign_rings(ax, home, r["sig_cluster"], r["mean"])
                    ax.axvline(mid, color="k", lw=0.4)
                    ax.invert_yaxis()
                    ax.set_aspect("equal")
                    k = int((r["p_cluster"] < .05).sum())
                    ax.set_title(f"{u}\n{p}: {k} cl. ({int(r['sig_cluster'].sum())} pts)", fontsize=7)
            if sc is not None:
                fig.colorbar(sc, ax=axs, shrink=0.25, label="log2(late pre / early pre)")
            fig.suptitle(f"{fdir.name}: DRIFT CONTROL, rhythm-band movement, late vs early pre ({a.perm} shuffles)")
            frf.save(fig, out / "band_maps_driftpre.png", """
DRIFT CONTROL for the band maps: each animal's PRE sessions split in date order (early half = reference, late half =
'epoch'), the same statistic and cluster-mass test. A post-stroke cluster of the same sign in the same place is not
evidence of a stroke effect.""")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

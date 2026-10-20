"""Within-spout executed-deviation check (Priya 10-20, agreed): with the spout FIXED, does the cortical response shift with
where the tongue actually went?

Pure TARGET coding predicts no within-spout slope; pure EXECUTED coding predicts a within-spout slope equal to the slope
across spouts (same executed angle -> same response, whichever spout it was aimed at). Model-light companion of the
encoding contest (`wfield_local/combined_encoding.py`, `nokin:*`):

  trial response   mean dF over cue + 0 .. 1.5 s minus the trial's own pre-cue 0.5 s (the trial-to-trial contest window;
                   trial level because GCaMP6s integrates over the ~7 Hz licks, so single-lick windows overlap)
  trial deviation  the executed LATERAL angle of the trial's usable licks in that window, minus the spout's mean
                   ('mean' = mean over those licks; 'first' = the first lick only), plus the same for reach
  within slope     pooled over spouts, everything demeaned within spout: response ~ lateral + reach + n_licks (licking
                   vigour and reach are the obvious confounds of a lateral deviation)
  between slope    across the six spouts: spout-mean response ~ 1 + spout-mean lateral + spout-mean reach (EXECUTED
                   means, so both slopes are per degree of executed angle)
  ratio            within / between lateral slope: ~0 target, ~1 executed. Measurement noise in the lick angle ATTENUATES
                   the within slope (regression dilution), so a ratio below 1 is not by itself evidence for target.

Executed coordinates: cam4 (peak angle deg, protrusion px) by default; with COMBINED_ENCODING_EXEC set (3-D modes, the
EXEC3D snapshot table) the 3-D lateral angle out of the sagittal plane and the 3-D distance (mm). Same usable licks as the
encoding model. The imaging is drift-filtered exactly as the encoding fit (< DRIFT_MIN_PERIOD_S removed).

Curated sessions only (ak.curated_labels(); from 08-06 by default = the August baseline + post-stroke).
Outputs: <run>/imaging_encoding/qc/within_spout_deviation[_<exec>]/ (per-session .npz maps, within_spout_deviation.csv,
terciles.csv, figures). Usage:
    PYTHONPATH=. python scripts/within_spout_deviation.py --jobs 2            # cohort (curated, from 08-06)
    PYTHONPATH=. python scripts/within_spout_deviation.py PS92_0811          # one session
    PYTHONPATH=. python scripts/within_spout_deviation.py --figures-only
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

WIN_S = (0.0, 1.5)
BASE_S = (-0.5, 0.0)
MIN_TRIALS = 9                      # per spout (3 per tercile)
SHOW = ("cortex", "MOs", "MOp", "SSp-m", "SSp-ul", "SSp-bfd")
EPOCHS = ("pre", "acute", "subacute", "chronic")


def trial_table(B, E: pd.DataFrame, mode: str) -> pd.DataFrame:
    """One row per trial: pos, lateral / reach of its usable licks in the window ('mean' and 'first'), n_licks."""
    if mode and B.exec_xyz is not None:
        ml, ap, dn = (np.asarray(x, float) for x in B.exec_xyz)
        lat, reach = np.degrees(np.arctan2(ml, np.hypot(ap, dn))), np.sqrt(ml ** 2 + ap ** 2 + dn ** 2)
    else:
        lat, reach = np.asarray(B.lick_angle, float), np.asarray(B.lick_reach, float)
    use = np.isfinite(lat) & np.isfinite(reach)
    on = np.asarray(B.lick_on_s, float)
    rows = []
    for k, r in enumerate(E.itertuples()):
        inw = (on >= r.cue_s + WIN_S[0]) & (on < r.cue_s + WIN_S[1])
        u = np.flatnonzero(inw & use)
        if not len(u):
            continue
        f = u[np.argmin(on[u])]
        rows.append({"k": k, "pos": r.pos, "lat_mean": lat[u].mean(), "reach_mean": reach[u].mean(),
                     "lat_first": lat[f], "reach_first": reach[f], "n_licks": int(inw.sum())})
    return pd.DataFrame(rows)


def slopes(R: np.ndarray, T: pd.DataFrame, var: str) -> dict:
    """R (trials, d) responses. Returns within / between coefficient arrays (d,) for lateral (+ within SE, terciles)."""
    lat, rch = T[f"lat_{var}"].to_numpy(float), T[f"reach_{var}"].to_numpy(float)
    nl, pos = T.n_licks.to_numpy(float), T.pos.to_numpy()
    keep = np.isin(pos, [p for p in np.unique(pos) if np.sum(pos == p) >= MIN_TRIALS])
    R, lat, rch, nl, pos = R[keep], lat[keep], rch[keep], nl[keep], pos[keep]
    P = np.unique(pos)
    if len(P) < 4:
        return {}
    Rd, X = R.copy(), np.column_stack([lat, rch, nl])
    Xd = X.copy()
    mR = np.stack([R[pos == p].mean(0) for p in P])
    mX = np.stack([X[pos == p].mean(0) for p in P])
    terc = np.zeros(len(R), int)
    for i, p in enumerate(P):
        m = pos == p
        Rd[m] -= mR[i]
        Xd[m] -= mX[i]
        terc[m] = np.searchsorted(np.quantile(Xd[m, 0], [1 / 3, 2 / 3]), Xd[m, 0], side="right")
    Xd = np.where(np.isfinite(Xd), Xd, 0.0)
    G = np.linalg.pinv(Xd)                                     # (3, n)
    bw = G @ Rd                                                # (3, d)
    res = Rd - Xd @ bw
    dof = max(len(R) - len(P) - 3, 1)
    se = np.sqrt(np.outer(np.diag(np.linalg.pinv(Xd.T @ Xd)), (res ** 2).sum(0) / dof))
    Xb = np.column_stack([np.ones(len(P)), mX[:, :2]])
    bb = np.linalg.pinv(Xb) @ mR                               # (3, d): intercept, lateral, reach
    return {"b_within": bw[0], "se_within": se[0], "b_between": bb[1], "n_trials": len(R), "n_spouts": len(P),
            "lat_sd_within": float(Xd[:, 0].std()), "lat_sd_between": float(mX[:, 0].std()),
            "terc_R": np.stack([Rd[terc == j].mean(0) for j in range(3)]),
            "terc_lat": np.array([Xd[terc == j, 0].mean() for j in range(3)])}


def session_check(label: str) -> dict:
    from wfield_local import combined_encoding as ce
    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver
    mode = os.environ.get(ce.EXEC_ENV, "")
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    B = ce.load_behaviour(config.animal_of(label), "2026" + label.split("_")[-1], PathResolver())
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0].astype(np.float32)
    V = V - Q @ (Q.T @ V)                                      # the encoding fit's drift filter
    E = B.trials[B.trials.engaged & np.isfinite(B.trials.cue_s)].reset_index(drop=True)
    T = trial_table(B, E, mode)
    cue = E.cue_s.to_numpy(float)[T.k.to_numpy()]

    def _mean(a, b):
        i0, i1 = np.searchsorted(ft, cue + a), np.searchsorted(ft, cue + b)
        return np.stack([V[x:y].mean(0) for x, y in zip(i0, i1)])
    Rk = _mean(*WIN_S) - _mean(*BASE_S)                        # (trials, K)
    areas = sorted({int(a) for a in np.unique(parea[pstat]) if a != 0})
    Ra = np.stack([Ub[(parea == a) & pstat].mean(0) for a in areas] + [Ub[pstat].mean(0)], 1)
    anames = [names.get(a, str(a)) for a in areas] + ["cortex"]
    rows, terc, maps = [], [], {}
    for var in ("mean", "first"):
        sK = slopes(Rk, T, var)
        if not sK:
            continue
        sA = slopes(Rk @ Ra, T, var)                            # area SEs need area-space residuals
        maps[f"within_{var}"] = (Ub @ sK["b_within"]).astype(np.float32)
        maps[f"between_{var}"] = (Ub @ sK["b_between"]).astype(np.float32)
        for i, an in enumerate(anames):
            rows.append({"label": label, "variant": var, "area": an, "b_within": sA["b_within"][i],
                         "se_within": sA["se_within"][i], "b_between": sA["b_between"][i],
                         "n_trials": sA["n_trials"], "n_spouts": sA["n_spouts"],
                         "lat_sd_within": sA["lat_sd_within"], "lat_sd_between": sA["lat_sd_between"]})
            for j in range(3):
                terc.append({"label": label, "variant": var, "area": an, "tercile": j,
                             "lat_dev": sA["terc_lat"][j], "resp_dev": sA["terc_R"][j, i]})
    return {"rows": rows, "terc": terc, "maps": maps, "flat": flat, "shape": shape,
            "epoch": epochs.epoch_of(label), "coords": "3d" if mode else "cam4"}


def _area_mean(D: pd.DataFrame, area: str) -> pd.DataFrame:
    """Both hemispheres of an area kept apart (name_L / name_R) plus the bare name."""
    return D[(D.area == area) | (D.area.str.rsplit("_", n=1).str[0] == area)]


def report(D: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    D = D.assign(b10_within=D.b_within * 10, b10_between=D.b_between * 10)
    for var in ("mean", "first"):
        G = D[D.variant == var]
        for ar in ("cortex", "MOs", "SSp-m"):
            A = _area_mean(G, ar)
            if A.empty:
                continue
            for c in ("b10_within", "b10_between"):
                t = A.groupby(["animal", "area", "epoch"])[c].mean().unstack("epoch")
                print(f"\n  [{var} deviation] {ar}: {c} (dF per 10 deg executed lateral; rule 8: animals first)\n"
                      + t.reindex(columns=list(EPOCHS)).round(5).to_string())
            # ratio of animal means (a ratio of noisy per-session values is unstable)
            m = A.groupby(["animal", "area", "epoch"])[["b_within", "b_between"]].mean()
            r = (m.b_within / m.b_between).unstack("epoch").reindex(columns=list(EPOCHS))
            print(f"  [{var}] {ar}: within / between (0 target, 1 executed; diluted by angle noise)\n"
                  + r.round(2).to_string())


def figures(out: Path, D: pd.DataFrame, Tt: pd.DataFrame, maps: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    # 1) tercile curves: within-spout response deviation vs lateral deviation, per epoch, cohort (animal means)
    areas = [a for a in ("cortex", "MOs_L", "MOs_R", "SSp-m_L", "SSp-m_R") if a in set(Tt.area)]
    fig, ax = plt.subplots(len(areas), 2, figsize=(8, 2.4 * len(areas)), squeeze=False, sharex="col")
    col = dict(zip(EPOCHS, ("0.2", "#d7301f", "#fc8d59", "#4575b4")))
    for i, ar in enumerate(areas):
        for j, var in enumerate(("mean", "first")):
            G = Tt[(Tt.area == ar) & (Tt.variant == var)]
            for ep in EPOCHS:
                g = G[G.epoch == ep].groupby(["animal", "tercile"])[["lat_dev", "resp_dev"]].mean()
                if g.empty:
                    continue
                m = g.groupby("tercile").mean()
                s = g.groupby("tercile").resp_dev.sem()
                ax[i, j].errorbar(m.lat_dev, m.resp_dev, yerr=s, color=col[ep], marker="o", ms=3, lw=1,
                                  label=f"{ep} (n={g.index.get_level_values(0).nunique()})")
            ax[i, j].axhline(0, color="0.7", lw=0.5)
            ax[i, j].set_title(f"{ar} -- {var} deviation", fontsize=8)
            ax[i, j].tick_params(labelsize=7)
    ax[-1, 0].set_xlabel("executed lateral deviation from spout mean (thirds)", fontsize=7)
    ax[0, 0].set_ylabel("response deviation (dF)", fontsize=7)
    ax[0, 0].legend(fontsize=6)
    fig.suptitle("Within-spout: does the response follow where the tongue went? (flat = target; slope = executed)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "within_spout_terciles.png", dpi=150)
    plt.close(fig)
    # 2) maps: within vs between lateral slope, per epoch (session mean of animal means), mean deviation variant
    if not maps:
        return
    sh = next(iter(maps.values()))["shape"]
    eps = [e for e in EPOCHS if any(v["epoch"] == e for v in maps.values())]
    fig, ax = plt.subplots(2, len(eps), figsize=(2.6 * len(eps), 5), squeeze=False)
    for j, ep in enumerate(eps):
        for i, kind in enumerate(("within_mean", "between_mean")):
            per_an = {}
            for lab, v in maps.items():
                if v["epoch"] == ep and kind in v["maps"]:
                    per_an.setdefault(lab[:4], []).append(v)
            if not per_an:
                continue
            imgs = []
            for vs in per_an.values():
                acc = np.full(sh[0] * sh[1], np.nan, np.float32)
                acc[vs[0]["flat"]] = np.mean([v["maps"][kind] for v in vs], 0)
                imgs.append(acc.reshape(sh))
            img = np.nanmean(imgs, 0) * 10
            lim = np.nanpercentile(np.abs(img), 98) or 1e-6
            ax[i, j].imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[i, j].set_title(f"{ep}: {kind.split('_')[0]} (n={len(per_an)})\n|lim| {lim:.2g} dF/10 deg", fontsize=7)
            ax[i, j].axis("off")
    fig.suptitle("Lateral slope maps: within spout (top) vs across spouts (bottom); ipsi = image left", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "within_spout_maps.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import epochs
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--from-date", default="0806", help="first MMDD (default: the August-only baseline)")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args(argv)
    mode = os.environ.get(ce.EXEC_ENV, "")
    out = oo.root(PathResolver()) / a.run / "imaging_encoding" / "qc" / ("within_spout_deviation"
                                                                          + (f"_{mode}" if mode else ""))
    writeguard.assert_writable(out.parent)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    cur = set(ak.curated_labels())               # CURATED only (Priya 10-20)
    labels = a.labels or [x for x in ce.sessions_available() if x in cur and x[-4:] >= a.from_date]
    bad = [x for x in labels if x not in cur]
    if bad:
        raise SystemExit(f"not curated: {bad}")
    if not a.figures_only:
        todo = [x for x in labels if not (out / "sessions" / f"{x}.npz").exists()]
        got, fail = ak.fan_sessions(todo, session_check, jobs=a.jobs)
        for it, err in fail:
            print(f"  FAILED {it}: {err}")
        for lab, r in got:
            pd.DataFrame(r["rows"]).to_csv(out / "sessions" / f"{lab}.csv", index=False, float_format="%.6g")
            pd.DataFrame(r["terc"]).to_csv(out / "sessions" / f"{lab}_terc.csv", index=False, float_format="%.6g")
            np.savez_compressed(out / "sessions" / f"{lab}.npz", flat=r["flat"], shape=np.array(r["shape"]),
                                keys=np.array(list(r["maps"])), maps=np.stack(list(r["maps"].values())),
                                coords=r["coords"])
    have = [x for x in labels if (out / "sessions" / f"{x}.csv").exists()]
    if not have:
        print("no sessions")
        return 1
    D = pd.concat([pd.read_csv(out / "sessions" / f"{x}.csv") for x in have], ignore_index=True)
    Tt = pd.concat([pd.read_csv(out / "sessions" / f"{x}_terc.csv") for x in have], ignore_index=True)
    for X in (D, Tt):
        X["animal"] = X.label.str[:4]
        X["epoch"] = X.label.map(epochs.epoch_of)
    D.to_csv(out / "within_spout_deviation.csv", index=False, float_format="%.6g")
    Tt.to_csv(out / "terciles.csv", index=False, float_format="%.6g")
    maps = {}
    for x in have:
        z = np.load(out / "sessions" / f"{x}.npz")
        maps[x] = {"epoch": epochs.epoch_of(x), "flat": z["flat"], "shape": tuple(z["shape"]),
                   "maps": dict(zip(z["keys"], z["maps"]))}
    report(D)
    figures(out, D, Tt, maps)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
#: LICK alignment (Priya 10-20: "there is a variable BIG delay in far contra licking after stroke"): a fixed cue window
#: then catches different licking phases -- or no licks -- on exactly the post-stroke far-contra trials. --align lick:
#: response = the 1 s after the trial's FIRST lick (any lick, cue .. stop), same pre-cue baseline; deviation from the
#: usable licks in that second. Latency (first lick - cue) is a covariate in both alignments.
ALIGN_ENV = "WITHIN_SPOUT_ALIGN"     # set by --align (the session workers read it)
LICK_WIN_S = (0.0, 1.0)
SIDES = ("contra", "mid", "ipsi")
WINDOW_ENV = "WITHIN_SPOUT_WINDOW"   # "enl" = pre-cue response window (set by --window)
ENL_S = (-1.0, 0.0)                  # relative to the cue
PRE_ARRIVAL_S = (-0.5, 0.0)          # relative to the spout arrival (strobe)
KIND_TITLE = {"within_mean": "WITHIN-spout lateral slope; red = more activity for licks toward the mouse's LEFT",
              "between_mean": "ACROSS-spout lateral slope; red = more activity for licks toward the mouse's LEFT",
              "dir_mean": "DIRECTION (within spout); red = more activity for licks toward the mouse's LEFT",
              "mid_mean": "TOWARD MIDLINE (within spout, lateral spouts); red = more activity for MEDIAL licks, "
                          "blue = for more LATERAL licks"}
MIN_TRIALS = 9                      # per spout (3 per tercile)
SHOW = ("cortex", "MOs", "MOp", "SSp-m", "SSp-ul", "SSp-bfd")
EPOCHS = ("pre", "acute", "subacute", "chronic")


def trial_table(B, E: pd.DataFrame, mode: str, align: str = "cue") -> pd.DataFrame:
    """One row per trial: pos, lateral / reach of its usable licks in the window ('mean' and 'first'), n_licks,
    alignment time t0 (cue or first lick) and latency (first lick - cue)."""
    if mode and B.exec_xyz is not None:
        ml, ap, dn = (np.asarray(x, float) for x in B.exec_xyz)
        # SIGN: pose3d side_ml is + toward the mouse's RIGHT (contra; calibration: close_contra +0.44 mm), the cam4 angle
        # + toward the LEFT (ipsi) -- flip so every lateral slope / direction / toward-midline sign means the same thing
        lat, reach = -np.degrees(np.arctan2(ml, np.hypot(ap, dn))), np.sqrt(ml ** 2 + ap ** 2 + dn ** 2)
    else:
        lat, reach = np.asarray(B.lick_angle, float), np.asarray(B.lick_reach, float)
    use = np.isfinite(lat) & np.isfinite(reach)
    on = np.asarray(B.lick_on_s, float)
    rows = []
    for k, r in enumerate(E.itertuples()):
        stop = r.stop_s if np.isfinite(r.stop_s) else r.cue_s + 7.0
        after = on[(on >= r.cue_s) & (on < stop)]
        if not len(after):
            continue
        l1 = after.min()
        t0, w = (r.cue_s, WIN_S) if align == "cue" else (l1, LICK_WIN_S)
        inw = (on >= t0 + w[0]) & (on < t0 + w[1])
        u = np.flatnonzero(inw & use)
        if not len(u):
            continue
        f = u[np.argmin(on[u])]
        us = u[np.argsort(on[u])]                              # split halves (odd / even licks) for reliability
        rows.append({"k": k, "pos": r.pos, "lat_mean": lat[u].mean(), "reach_mean": reach[u].mean(),
                     "lat_first": lat[f], "reach_first": reach[f], "n_licks": int(inw.sum()),
                     "t0": t0, "latency": l1 - r.cue_s,
                     "lat_odd": lat[us[0::2]].mean() if len(us) >= 2 else np.nan,
                     "lat_even": lat[us[1::2]].mean() if len(us) >= 2 else np.nan})
    return pd.DataFrame(rows)


def slopes(R: np.ndarray, T: pd.DataFrame, var: str) -> dict:
    """R (trials, d) responses. Returns within / between coefficient arrays (d,) for lateral (+ within SE, terciles)."""
    lat, rch = T[f"lat_{var}"].to_numpy(float), T[f"reach_{var}"].to_numpy(float)
    nl, pos = T.n_licks.to_numpy(float), T.pos.to_numpy()
    lt = T.latency.to_numpy(float) if "latency" in T else np.zeros(len(T))
    keep = np.isin(pos, [p for p in np.unique(pos) if np.sum(pos == p) >= MIN_TRIALS])
    R, lat, rch, nl, lt, pos = R[keep], lat[keep], rch[keep], nl[keep], lt[keep], pos[keep]
    P = np.unique(pos)
    if len(P) < 2:                                             # within needs >= 2 spouts, between >= 4 (below)
        return {}
    Rd, X = R.copy(), np.column_stack([lat, rch, nl, lt])
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
    G = np.linalg.pinv(Xd)                                     # (4, n)
    bw = G @ Rd                                                # (4, d): lateral, reach, n_licks, latency
    res = Rd - Xd @ bw
    dof = max(len(R) - len(P) - X.shape[1], 1)
    se = np.sqrt(np.outer(np.diag(np.linalg.pinv(Xd.T @ Xd)), (res ** 2).sum(0) / dof))
    if len(P) >= 4:
        Xb = np.column_stack([np.ones(len(P)), mX[:, :2]])
        bb = np.linalg.pinv(Xb) @ mR                           # (3, d): intercept, lateral, reach
    else:
        bb = np.full((3, R.shape[1]), np.nan)
    return {"b_within": bw[0], "se_within": se[0], "b_between": bb[1], "n_trials": len(R), "n_spouts": len(P),
            "lat_sd_within": float(Xd[:, 0].std()), "lat_sd_between": float(mX[:, 0].std()),
            "terc_R": np.stack([Rd[terc == j].mean(0) for j in range(3)]),
            "terc_lat": np.array([Xd[terc == j, 0].mean() for j in range(3)])}


def side_sign(pos) -> np.ndarray:
    """+1 contra spouts (mouse's right), -1 ipsi, 0 mid: deviation x side_sign = movement TOWARD THE MIDLINE (lateral
    angle + = toward the mouse's left)."""
    p = pd.Series(np.asarray(pos, str))
    return np.where(p.str.endswith("contra"), 1.0, np.where(p.str.endswith("ipsi"), -1.0, 0.0))


def dirmid(R: np.ndarray, T: pd.DataFrame, var: str = "mean") -> dict:
    """DIRECTION + TOWARD-MIDLINE model (Priya 10-20; the bilateral component had opposite signs at contra / ipsi spouts
    pre-stroke): within spout, response ~ dev + dev x side_sign + reach + n_licks + latency. b_dir = response per 10 deg
    toward the mouse's LEFT whatever the spout; b_mid = response per 10 deg toward the MIDLINE (lateral spouts only).
    Equivalent to the mean / half-difference of the contra- and ipsi-spout slopes, fitted jointly with proper SEs."""
    lat, rch = T[f"lat_{var}"].to_numpy(float), T[f"reach_{var}"].to_numpy(float)
    pos = T.pos.to_numpy()
    keep = np.isin(pos, [p for p in np.unique(pos) if np.sum(pos == p) >= MIN_TRIALS])
    if len(np.unique(pos[keep])) < 3:
        return {}
    X = np.column_stack([lat, lat * side_sign(pos), rch, T.n_licks.to_numpy(float), T.latency.to_numpy(float)])[keep]
    R, pos = R[keep].copy(), pos[keep]
    for p in np.unique(pos):
        m = pos == p
        R[m] -= R[m].mean(0)
        X[m] -= X[m].mean(0)
    X = np.where(np.isfinite(X), X, 0.0)
    b = np.linalg.pinv(X) @ R
    res = R - X @ b
    s2 = (res ** 2).sum(0) / max(len(R) - len(np.unique(pos)) - X.shape[1], 1)
    v = np.diag(np.linalg.pinv(X.T @ X))
    return {"b_dir": b[0], "se_dir": np.sqrt(v[0] * s2), "b_mid": b[1], "se_mid": np.sqrt(v[1] * s2)}


def split_half(T: pd.DataFrame) -> tuple[float, float, int]:
    """(r odd-even, Spearman-Brown reliability of the all-lick mean, n trials) of the WITHIN-SPOUT lateral deviation.
    Counts real lick-to-lick variation as noise, so slope / reliability is an UPPER bound on the disattenuated slope."""
    G = T.dropna(subset=["lat_odd", "lat_even"])
    if len(G) < 20:
        return np.nan, np.nan, len(G)
    o = G.lat_odd - G.groupby("pos").lat_odd.transform("mean")
    e = G.lat_even - G.groupby("pos").lat_even.transform("mean")
    r = float(np.corrcoef(o, e)[0, 1])
    return r, (2 * r / (1 + r) if r > -1 else np.nan), len(G)


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
    align = os.environ.get(ALIGN_ENV, "cue")
    T = trial_table(B, E, mode, align)
    cue = E.cue_s.to_numpy(float)[T.k.to_numpy()]
    t0 = T.t0.to_numpy(float)

    def _mean(ref, a, b):
        i0, i1 = np.searchsorted(ft, ref + a), np.searchsorted(ft, ref + b)
        return np.stack([V[x:y].mean(0) if y > x else np.full(V.shape[1], np.nan, V.dtype) for x, y in zip(i0, i1)])
    if os.environ.get(WINDOW_ENV) == "enl":
        # PRE-CUE (ENL) response (Priya 10-20, "queue the enl analysis"): the last second before the cue relative to the
        # 0.5 s before the spout arrived -- does the pre-cue state predict where (within the spout) the tongue will go?
        strobe = E.strobe_s.to_numpy(float)[T.k.to_numpy()]
        strobe = np.where(np.isfinite(strobe), strobe, cue - 3.0)
        Rk = _mean(cue, *ENL_S) - _mean(strobe, *PRE_ARRIVAL_S)
    else:
        Rk = _mean(t0, *(WIN_S if align == "cue" else LICK_WIN_S)) - _mean(cue, *BASE_S)   # (trials, K)
    ok = np.isfinite(Rk).all(1)                                # windows past the end of the recording
    T, Rk = T[ok].reset_index(drop=True), Rk[ok]
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
            rows.append({"label": label, "variant": var, "spouts": "all", "area": an, "b_within": sA["b_within"][i],
                         "se_within": sA["se_within"][i], "b_between": sA["b_between"][i],
                         "n_trials": sA["n_trials"], "n_spouts": sA["n_spouts"],
                         "lat_sd_within": sA["lat_sd_within"], "lat_sd_between": sA["lat_sd_between"],
                         "latency_med": float(T.latency.median())})
            for j in range(3):
                terc.append({"label": label, "variant": var, "area": an, "tercile": j,
                             "lat_dev": sA["terc_lat"][j], "resp_dev": sA["terc_R"][j, i]})
        # BY SPOUT SIDE (10-20: the bilateral component has opposite signs at contra / ipsi spouts pre-stroke, an
        # eccentricity pattern that cancels in the pooled slope) -- within slope only
        for side in SIDES:
            m = T.pos.str.endswith(side).to_numpy()
            sS = slopes((Rk @ Ra)[m], T[m], var) if m.any() else {}
            if not sS:
                continue
            for i, an in enumerate(anames):
                rows.append({"label": label, "variant": var, "spouts": side, "area": an,
                             "b_within": sS["b_within"][i], "se_within": sS["se_within"][i], "b_between": np.nan,
                             "n_trials": sS["n_trials"], "n_spouts": sS["n_spouts"],
                             "lat_sd_within": sS["lat_sd_within"], "lat_sd_between": np.nan,
                             "latency_med": float(T.latency[m].median())})
    dm = []
    dK, dA = dirmid(Rk, T), dirmid(Rk @ Ra, T)
    r_h, rel, n_h = split_half(T)
    if dK:
        maps["dir_mean"] = (Ub @ dK["b_dir"]).astype(np.float32)
        maps["mid_mean"] = (Ub @ dK["b_mid"]).astype(np.float32)
        for i, an in enumerate(anames):
            dm.append({"label": label, "area": an, "b_dir": dA["b_dir"][i], "se_dir": dA["se_dir"][i],
                       "b_mid": dA["b_mid"][i], "se_mid": dA["se_mid"][i]})
    for r_ in rows:
        r_.update({"r_half": r_h, "rel_half": rel, "n_half": n_h})
    return {"rows": rows, "terc": terc, "maps": maps, "flat": flat, "shape": shape, "dirmid": dm,
            "epoch": epochs.epoch_of(label), "coords": "3d" if mode else "cam4"}


def _area_mean(D: pd.DataFrame, area: str) -> pd.DataFrame:
    """Both hemispheres of an area kept apart (name_L / name_R) plus the bare name."""
    return D[(D.area == area) | (D.area.str.rsplit("_", n=1).str[0] == area)]


def report(D: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    D = D.assign(b10_within=D.b_within * 10, b10_between=D.b_between * 10,
                 spouts=D.spouts if "spouts" in D else "all")
    S, D = D[D.spouts != "all"], D[D.spouts == "all"]
    if "latency_med" in D:
        lt = D[(D.area == "cortex") & (D.variant == "mean")].groupby(["animal", "epoch"]).latency_med.mean()
        print("  median first-lick latency (s)\n" + lt.unstack().reindex(columns=list(EPOCHS)).round(2).to_string())
    for ar, kind in (("cortex", "bilateral"), ("SSp-m", "crossed")):   # by spout side, inverse-variance pooled z
        A = S[S.variant == "mean"]
        if ar == "cortex":
            A = A[A.area == "cortex"].assign(b=lambda x: x.b10_within, se=lambda x: x.se_within * 10)
        else:
            L, R_ = (A[A.area == f"{ar}_{h}"].set_index(["label", "spouts"]) for h in ("left", "right"))
            A = (L[["animal", "epoch"]].assign(b=L.b10_within - R_.b10_within,
                                               se=np.hypot(L.se_within, R_.se_within) * 10).reset_index())
        if A.empty:
            continue
        A = A.assign(w=1 / A.se ** 2)
        g = A.groupby(["animal", "spouts", "epoch"]).apply(
            lambda x: (x.b * x.w).sum() / np.sqrt(x.w.sum()), include_groups=False)
        print(f"\n  {ar} {kind} within slope by SPOUT SIDE, pooled z (mean deviation)\n"
              + g.unstack("epoch").reindex(columns=list(EPOCHS)).round(1).to_string())
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
            # CROSSED part (left - right hemisphere): the across-spout lateral tuning is lateralised, a global /
            # bilateral shift is not executed direction; this is the comparison that matters
            if {f"{ar}_left", f"{ar}_right"} <= set(A.area):
                W = A.pivot_table(index=["label", "animal", "epoch"], columns="area", values=["b10_within", "b10_between"])
                X = pd.DataFrame({k: W[k][f"{ar}_left"] - W[k][f"{ar}_right"] for k in ("b10_within", "b10_between")})
                g = X.groupby(["animal", "epoch"]).mean()
                print(f"  [{var}] {ar} LEFT - RIGHT (crossed): within, between, within / between\n"
                      + pd.concat({"within": g.b10_within.unstack(), "between": g.b10_between.unstack(),
                                   "ratio": (g.b10_within / g.b10_between).unstack()}, axis=1)
                      .reindex(columns=list(EPOCHS), level=1).round(5).to_string())
                pre = X.xs("pre", level="epoch").b10_within
                print(f"  pre sessions: crossed within mean {pre.mean():.5f}, t = "
                      f"{pre.mean() / (pre.std() / np.sqrt(len(pre))):.2f}, n = {len(pre)}")
            # ratio of animal means (a ratio of noisy per-session values is unstable)
            m = A.groupby(["animal", "area", "epoch"])[["b_within", "b_between"]].mean()
            r = (m.b_within / m.b_between).unstack("epoch").reindex(columns=list(EPOCHS))
            print(f"  [{var}] {ar}: within / between (0 target, 1 executed; diluted by angle noise)\n"
                  + r.round(2).to_string())


def figures(out: Path, D: pd.DataFrame, Tt: pd.DataFrame, maps: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out.parents[1] / "figures" / "within_spout" / out.name      # results, not qc (Priya 10-21)
    figs.mkdir(parents=True, exist_ok=True)
    # 1) tercile curves: within-spout response deviation vs lateral deviation, per epoch, cohort (animal means)
    areas = [a for a in ("cortex", "MOs_left", "MOs_right", "SSp-m_left", "SSp-m_right") if a in set(Tt.area)]
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


def report_dirmid(DM: pd.DataFrame, D: pd.DataFrame) -> None:
    """Direction / toward-midline coefficients (bilateral cortex, crossed SSp-m) and split-half reliability."""
    pd.set_option("display.width", 220)
    for meas, f in (("cortex bilateral", lambda X, c: X[X.area == "cortex"].set_index("label")[c]),
                    ("SSp-m crossed", lambda X, c: (X[X.area == "SSp-m_left"].set_index("label")[c]
                                                    - X[X.area == "SSp-m_right"].set_index("label")[c]))):
        for c in ("dir", "mid"):
            b = f(DM, f"b_{c}") * 10 * 1e3
            se = (np.hypot(*(DM[DM.area == f"SSp-m_{h}"].set_index("label")[f"se_{c}"] for h in ("left", "right")))
                  if meas.startswith("SSp") else DM[DM.area == "cortex"].set_index("label")[f"se_{c}"]) * 10 * 1e3
            X = pd.DataFrame({"b": b, "w": 1 / se ** 2})
            X["animal"], X["epoch"] = X.index.str[:4], X.index.map(DM.drop_duplicates("label").set_index("label").epoch)
            X = X[~X.index.duplicated()]
            g = X.groupby(["animal", "epoch"]).apply(lambda x: pd.Series(
                {"b": (x.b * x.w).sum() / x.w.sum(), "z": (x.b * x.w).sum() / np.sqrt(x.w.sum())}), include_groups=False)
            print(f"\n  {meas}: {'DIRECTION (+ = more for licks toward the mouse LEFT)' if c == 'dir' else 'TOWARD MIDLINE (+ = more for medial licks)'}"
                  f" x1e-3 dF / 10 deg, pooled b and z\n"
                  + g.unstack("epoch").reindex(columns=list(EPOCHS), level=1).round(2).to_string())
    if "rel_half" in D:
        R = D[(D.area == "cortex") & (D.variant == "mean") & (D.get("spouts", "all") == "all")]
        print("\n  split-half reliability of the within-spout lateral deviation (Spearman-Brown; median per animal x epoch)\n"
              + R.groupby(["animal", "epoch"]).rel_half.median().unstack().reindex(columns=list(EPOCHS)).round(2)
              .to_string())


def _bh(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg over the finite entries of p -> boolean mask."""
    out = np.zeros(p.shape, bool)
    f = np.flatnonzero(np.isfinite(p))
    if not len(f):
        return out
    o = f[np.argsort(p[f])]
    thr = q * np.arange(1, len(o) + 1) / len(o)
    k = np.flatnonzero(p[o] <= thr)
    if len(k):
        out[o[: k[-1] + 1]] = True
    return out


def epoch_tests(S: dict) -> dict:
    """S {epoch: (n, P) session maps} for ONE animal -> {epoch: (mean, p)}. Sessions are the unit. pre vs 0 (one-sample
    t); each post epoch vs that animal's pre sessions (Welch t; a single session -> prediction test against the pre
    distribution, t = (x - mean_pre) / (sd_pre * sqrt(1 + 1 / n_pre)), df n_pre - 1)."""
    from scipy import stats
    out = {}
    pre = S.get("pre")
    for ep, X in S.items():
        m = np.nanmean(X, 0)
        if ep == "pre":
            p = stats.ttest_1samp(X, 0.0, axis=0, nan_policy="omit").pvalue if len(X) >= 2 else np.full(m.shape, np.nan)
        elif pre is None or len(pre) < 2:
            p = np.full(m.shape, np.nan)
        elif len(X) >= 2:
            p = stats.ttest_ind(X, pre, axis=0, equal_var=False, nan_policy="omit").pvalue
        else:
            n0 = len(pre)
            t = (X[0] - np.nanmean(pre, 0)) / (np.nanstd(pre, 0, ddof=1) * np.sqrt(1 + 1 / n0))
            p = 2 * stats.t.sf(np.abs(t), n0 - 1)
        out[ep] = (m, np.asarray(p, float))
    return out


def figures_stats(out: Path, maps: dict, q: float = 0.05) -> None:
    """Per animal x epoch maps (within / between, mean deviation) with FDR-significant regions outlined: pre vs 0, each
    post epoch vs the SAME animal's pre sessions (Priya 10-20: "outline significance ... within-animal vs pre-stroke").
    Post panels show the epoch mean; the outline marks where it differs from pre (black) -- the pre panel's outline
    marks where pre differs from 0."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out.parents[1] / "figures" / "within_spout" / out.name      # results, not qc (Priya 10-21)
    figs.mkdir(parents=True, exist_ok=True)
    sh = next(iter(maps.values()))["shape"]
    animals = sorted({lab[:4] for lab in maps})
    rows = []
    for kind in ("within_mean", "between_mean", "dir_mean", "mid_mean"):
        fig, ax = plt.subplots(len(animals), len(EPOCHS), figsize=(2.6 * len(EPOCHS), 2.5 * len(animals)), squeeze=False)
        for i, an in enumerate(animals):
            S, flat = {}, None
            for lab, v in maps.items():
                if lab[:4] == an and kind in v["maps"]:
                    S.setdefault(v["epoch"], []).append(v["maps"][kind])
                    flat = v["flat"]
            S = {e: np.stack(x) * 10 for e, x in S.items()}        # dF per 10 deg
            R = epoch_tests(S)
            if not R:
                continue
            lim = np.nanpercentile(np.abs(np.concatenate([m for m, _ in R.values()])), 98) or 1e-6
            for j, ep in enumerate(EPOCHS):
                a_ = ax[i, j]
                a_.axis("off")
                if ep not in R:
                    continue
                m, p = R[ep]
                sig = _bh(p, q)
                img, msk = np.full(sh[0] * sh[1], np.nan), np.zeros(sh[0] * sh[1])
                img[flat], msk[flat] = m, sig
                a_.imshow(img.reshape(sh), cmap="RdBu_r", vmin=-lim, vmax=lim)
                if sig.any():
                    a_.contour(msk.reshape(sh), levels=[0.5], colors="k", linewidths=0.6)
                a_.set_title(f"{an} {ep} (n={len(S[ep])})\n{'vs 0' if ep == 'pre' else 'vs pre'}: "
                             f"{100 * sig.mean():.0f} % px q<{q}", fontsize=7)
                rows.append({"kind": kind, "animal": an, "epoch": ep, "n": len(S[ep]),
                             "frac_sig": float(sig.mean()), "lim": float(lim)})
        fig.suptitle(f"{kind.split('_')[0]}-spout lateral slope (dF / 10 deg), outlined: FDR q < {q} -- pre vs 0, "
                     "post epochs vs the same animal's pre sessions (sessions = units); ipsi = image left", fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / f"within_spout_maps_stats_{kind.split('_')[0]}.png", dpi=150)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "map_significance.csv", index=False, float_format="%.4g")


def figures_repo_sig(out: Path, maps: dict, method: str = "nested") -> None:
    """Cohort maps (mean over animals of animal epoch means) with the REPO's map tests (beta_maps; Priya 10-20: "for our
    maps we used cluster based permutation right?"): pre panel = `vs_zero_contour` (where the map is there at all),
    post panels = `significance_contour` vs pre (animal = unit; 'nested' animals -> sessions bootstrap on the 8x grid,
    Bonferroni over bins = the repo default; 'cluster' = within-animal relabelling cluster mass, the second line).
    Replaces the per-pixel FDR as the reading of these maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import beta_maps as bm
    figs = out.parents[1] / "figures" / "within_spout" / out.name      # results, not qc (Priya 10-21)
    figs.mkdir(parents=True, exist_ok=True)
    sh = bm.MAP_SHAPE
    rows = []
    for kind in ("within_mean", "between_mean", "dir_mean", "mid_mean"):
        by = {}                                                    # epoch -> animal -> [full-grid maps]
        for lab, v in maps.items():
            if kind in v["maps"] and tuple(v["shape"]) == sh:
                img = np.full(sh[0] * sh[1], np.nan, np.float32)
                img[v["flat"]] = v["maps"][kind] * 10 * 1e3          # x1e-3 dF per 10 deg
                by.setdefault(v["epoch"], {}).setdefault(lab[:4], []).append(np.nan_to_num(img.reshape(sh)))
        eps = [e for e in EPOCHS if e in by]
        if "pre" not in eps:
            continue
        fig, ax = plt.subplots(1, len(eps), figsize=(3.2 * len(eps), 3.4), squeeze=False)
        means = {e: np.mean([np.mean(x, 0) for x in by[e].values()], 0) for e in eps}
        lim = np.nanpercentile(np.abs(np.concatenate([m.ravel() for m in means.values()])), 98) or 1e-6
        mask = bm.brain_mask()
        for j, ep in enumerate(eps):
            try:
                if ep == "pre":
                    sig, lab_ = bm.vs_zero_contour(by["pre"], method=method)
                else:
                    common = sorted(set(by["pre"]) & set(by[ep]))
                    sig, lab_ = bm.significance_contour({a: by["pre"][a] for a in common},
                                                        {a: by[ep][a] for a in common}, method=method)
            except Exception as e:                                 # noqa: BLE001 -- report, draw no contour
                sig, lab_ = None, f"test failed: {e}"
            img = np.where(mask, means[ep], np.nan) if mask is not None else means[ep]
            ax[0, j].imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
            if sig is not None and np.any(sig):
                ax[0, j].contour(np.asarray(sig, float), levels=[0.5], colors="k", linewidths=0.7)
            frac = float(np.mean(np.asarray(sig, bool)[mask])) if sig is not None and mask is not None else 0.0
            ax[0, j].set_title(f"{ep} (animals {len(by[ep])})\n{'vs 0' if ep == 'pre' else 'vs pre'}: "
                               f"{100 * frac:.0f} % of brain", fontsize=7)
            ax[0, j].axis("off")
            rows.append({"kind": kind, "method": method, "epoch": ep, "frac_sig": frac, "label": lab_})
        fig.suptitle(f"{KIND_TITLE.get(kind, kind)} (x1e-3 dF / 10 deg; ipsi hemisphere = image left); |lim| {lim:.2g}; "
                     f"outline: beta_maps '{method}' (animal = unit)", fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / f"within_spout_maps_{method}_{kind.split('_')[0]}.png", dpi=150)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(out / f"map_significance_{method}.csv", index=False)


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
    ap.add_argument("--window", choices=("post", "enl"), default="post",
                    help="post: the cue / lick window (--align); enl: the last 1 s before the cue vs pre-arrival")
    ap.add_argument("--align", choices=("cue", "lick"), default="cue",
                    help="cue: cue + 0-1.5 s; lick: first lick + 0-1 s (see ALIGN_ENV)")
    ap.add_argument("--epochs", default="", help="comma list to restrict sessions, e.g. pre,acute")
    ap.add_argument("--redo", action="store_true", help="recompute sessions that already have outputs")
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args(argv)
    os.environ[ALIGN_ENV] = a.align                          # inherited by the session workers
    os.environ[WINDOW_ENV] = a.window
    mode = os.environ.get(ce.EXEC_ENV, "")
    out = oo.root(PathResolver()) / a.run / "imaging_encoding" / "qc" / ("within_spout_deviation"
                                                                          + (f"_{mode}" if mode else "")
                                                                          + ("_lick" if a.align == "lick" else "")
                                                                          + ("_enl" if a.window == "enl" else ""))
    writeguard.assert_writable(out.parent)
    (out / "sessions").mkdir(parents=True, exist_ok=True)
    cur = set(ak.curated_labels())               # CURATED only (Priya 10-20)
    labels = a.labels or [x for x in ce.sessions_available() if x in cur and x[-4:] >= a.from_date]
    if a.epochs:
        labels = [x for x in labels if epochs.epoch_of(x) in a.epochs.split(",")]
    bad = [x for x in labels if x not in cur]
    if bad:
        raise SystemExit(f"not curated: {bad}")
    if not a.figures_only:
        todo = [x for x in labels if a.redo or not (out / "sessions" / f"{x}.npz").exists()]
        got, fail = ak.fan_sessions(todo, session_check, jobs=a.jobs)
        for it, err in fail:
            print(f"  FAILED {it}: {err}")
        for lab, r in got:
            pd.DataFrame(r["rows"]).to_csv(out / "sessions" / f"{lab}.csv", index=False, float_format="%.6g")
            pd.DataFrame(r["terc"]).to_csv(out / "sessions" / f"{lab}_terc.csv", index=False, float_format="%.6g")
            pd.DataFrame(r.get("dirmid", [])).to_csv(out / "sessions" / f"{lab}_dirmid.csv", index=False,
                                                     float_format="%.6g")
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
    dmf = [out / "sessions" / f"{x}_dirmid.csv" for x in have]
    dmf = [f for f in dmf if f.exists() and f.stat().st_size > 5]
    if dmf:
        DM = pd.concat([pd.read_csv(f) for f in dmf], ignore_index=True)
        DM["animal"], DM["epoch"] = DM.label.str[:4], DM.label.map(epochs.epoch_of)
        DM.to_csv(out / "dirmid.csv", index=False, float_format="%.6g")
        report_dirmid(DM, D)
    figures(out, D[D.spouts == "all"] if "spouts" in D else D, Tt, maps)
    figures_stats(out, maps)                                   # per-animal, per-pixel FDR (supplementary)
    for method in ("nested", "cluster"):                       # the repo's map tests (primary reading)
        figures_repo_sig(out, maps, method)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

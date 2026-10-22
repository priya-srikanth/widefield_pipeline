"""FaceRhythm session readouts on the frozen master pre-stroke factor set (`scripts/face_rhythm_factors.py` outputs
``<set>/proj/<key>.npz`` and ``<set>/own/<key>.npz``). Computation only; figures + CLI: `scripts/face_rhythm_readouts.py`;
pipeline overview: docs/FACE_RHYTHM_PIPELINE.md.

The 2pRAM-cohort manuscript's FR analyses (FACE_RHYTHM_PLAN.md "What the 2pRAM-cohort manuscript does with FR"), here
ACROSS animals (one mesh, one frozen pre-stroke basis):
  amplitudes     pre-stroke components refit to every session: mean factor amplitude per session in the pre-cue
                 (cue -2..0 s) and response (0..+4 s) windows of engaged trials (25 Hz) and BETWEEN trials (5 Hz bins
                 outside cue -3..+6 s) -- "non-task movements preserved?" is between vs response; per position too.
  reconstruction relative reconstruction error of each session by the frozen basis (movement outside the pre-stroke
                 repertoire).
  evr_gap        each session's OWN rank-R fit vs the frozen basis on the same rows (EVR own - frozen), and how far its
                 own factors have moved from the frozen ones (matched spatial / spectral cosine: factor "morphing").
  movement       per point and window, two measures: SPEED (absolute movement, px / s RMS of the 25 Hz points) and
                 SHARE (normalised spectral power summed over xy and a band -- FR divides every frame by its mean
                 power over the whole face, so this is how movement is DISTRIBUTED, not how much). Per animal x epoch
                 the per-point log2 change from its pre-stroke mean (maps on the master face, averageable across
                 animals), and a CONTRA - IPSI asymmetry index per measure. Factor AMPLITUDES are on the normalised
                 spectrogram too: a change is a change in SHARE of the face's movement, not in its size. cam4: IPSILESIONAL = IMAGE RIGHT (Priya 10-21; all
                 lesions left), split at the face midline (MASTER_MIDLINE_X, the master's LP nose x, unless given).
One row per SESSION (a two-video session, PS92_0812, is joined). Epoch statistics: `orofacial_cohort.epoch_tables`
(per-animal epoch means + paired animal-weighted `boot_delta` vs pre) -- print the per-animal table first (rule 8).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = {"precue": (-2.0, 0.0), "response": (0.0, 4.0)}
BUSY_S = (-3.0, 6.0)
BANDS = {"all": (0.0, 1e9), "slow": (0.0, 3.0), "rhythm": (3.0, 12.0), "fast": (12.0, 1e9)}
#: face midline on the master (PS95_20260812) image: LP v36 nose x, median over pre-cue frames (likelihood > 0.9;
#: 10-21: nose 342.0, jaw 357.2, mesh median 350.0 -- the jaw sits off-centre, the nose tip is the midline marker).
MASTER_MIDLINE_X = 342.0


def _label(animal: str, date: str) -> str:
    return f"{animal}_{str(date)[4:]}"


def _tasks(fdir: Path, sub: str) -> pd.DataFrame:
    rows = []
    for f in sorted((fdir / sub).glob("*.npz")) if (fdir / sub).exists() else []:
        if f.name.endswith(".part.npz"):
            continue
        with np.load(f) as z:
            rows.append({"key": f.stem, "file": f, "animal": str(z["animal"]), "date": str(z["date"]),
                         "epoch": str(z["epoch"])})
    T = pd.DataFrame(rows, columns=["key", "file", "animal", "date", "epoch"])
    T["label"] = [_label(a, d) for a, d in zip(T.animal, T.date)]
    return T


def _day(label: str):
    from wfield_local import epochs
    return epochs.days_since_stroke(label)


def amplitudes(fdir: Path, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per session: amp_<comp>_<precue|response|between>, per session x position x comp: response amplitude)."""
    sums, pos_rows = {}, []
    for t in _tasks(fdir, "proj").itertuples():
        z = np.load(t.file)
        tr, tms = z["trials"].astype(float), z["t_ms"] / 1000.0
        acc = sums.setdefault(t.label, {})
        for w, (a, b) in WINDOWS.items():
            m = (tms >= a) & (tms < b)
            x = tr[:, m].reshape(-1, tr.shape[-1])
            ok = np.isfinite(x).all(1)
            s0, n0 = acc.get(w, (0.0, 0))
            acc[w] = (s0 + x[ok].sum(0), n0 + ok.sum())
            if w == "response":
                for p in np.unique(z["position"]):
                    xp = tr[z["position"] == p][:, m].reshape(-1, tr.shape[-1])
                    xp = xp[np.isfinite(xp).all(1)]
                    pos_rows += [{"label": t.label, "position": str(p), "comp": names[k], "amp": float(xp[:, k].mean()),
                                  "n_bins": len(xp)} for k in range(tr.shape[-1]) if len(xp)]
        ts, ws, cues = z["session_t_daq_s"], z["session"].astype(float), np.sort(z["cue_daq_s"])
        k = np.searchsorted(cues, ts)
        near = np.zeros(len(ts), bool)
        for kk in (k - 1, k):
            ok = (kk >= 0) & (kk < len(cues))
            d = ts - cues[np.clip(kk, 0, len(cues) - 1)]
            near |= ok & (d >= BUSY_S[0]) & (d <= BUSY_S[1])
        x = ws[~near]
        x = x[np.isfinite(x).all(1)]
        s0, n0 = acc.get("between", (0.0, 0))
        acc["between"] = (s0 + x.sum(0), n0 + len(x))
    rows = []
    for lab, acc in sums.items():
        r = {"label": lab}
        for w, (s, n) in acc.items():
            r.update({f"amp_{names[k]}_{w}": (s[k] / n if n else np.nan) for k in range(len(names))})
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(pos_rows)


def reconstruction(fdir: Path) -> pd.DataFrame:
    """Per session: median relative reconstruction error of trial-window and whole-session bins (frozen basis)."""
    acc = {}
    for t in _tasks(fdir, "proj").itertuples():
        z = np.load(t.file)
        a = acc.setdefault(t.label, ([], []))
        a[0].append(z["trials_err"].ravel())
        a[1].append(z["session_err"].ravel())
    return pd.DataFrame([{"label": k, "recon_err_trials": float(np.nanmedian(np.concatenate(v[0]))),
                          "recon_err_session": float(np.nanmedian(np.concatenate(v[1])))} for k, v in acc.items()])


def evr_gap(fdir: Path, names: list[str]) -> pd.DataFrame:
    """Per session (mean over its videos): EVR own / frozen / gap, and matched cosine of own to frozen factors
    (mean + per component)."""
    rows = []
    for t in _tasks(fdir, "own").itertuples():
        z = np.load(t.file)
        r = {"label": t.label, "evr_own": float(z["evr_own"]), "evr_frozen": float(z["evr_frozen"]),
             "cos_spatial_mean": float(np.mean(z["cos_spatial"])),
             "cos_spectral_mean": float(np.mean(z["cos_spectral"]))}
        r["evr_gap"] = r["evr_own"] - r["evr_frozen"]
        r.update({f"cos_spatial_{names[k]}": float(c) for k, c in enumerate(z["cos_spatial"])})
        rows.append(r)
    return pd.DataFrame(rows).groupby("label", as_index=False).mean(numeric_only=True) if rows else pd.DataFrame()


def point_measure(fdir: Path, freqs: np.ndarray, measure: str = "speed", band: str = "all") -> tuple[list[str], dict]:
    """Per session, per point: ``speed`` (RMS px / s) or ``share`` (normalised spectral power summed over xy and the
    band): {window: (n_sessions, P)} for between / precue / response (trial windows over positions, bin-weighted;
    speed averages speed^2 then roots)."""
    lo, hi = BANDS[band]
    fm = (freqs >= lo) & (freqs < hi)
    acc = {}
    for t in _tasks(fdir, "proj").itertuples():
        z = np.load(t.file)
        key = "speed" if measure == "speed" else "power"
        if f"{key}_n_bins" not in z.files:
            continue
        n = z[f"{key}_n_bins"]
        a = acc.setdefault(t.label, {})
        parts = {"between": (z[f"{key}_between"][None], n[:1]),
                 "precue": (z[f"{key}_precue"], n[1::2]), "response": (z[f"{key}_response"], n[2::2])}
        for w, (pw, nb) in parts.items():
            v = pw ** 2 if measure == "speed" else pw[..., fm].sum(-1).sum(-2)      # (k, P)
            ok = (nb > 0) & np.isfinite(v).all(1)
            s0, n0 = a.get(w, (0.0, 0))
            a[w] = (s0 + (v[ok] * nb[ok, None]).sum(0), n0 + nb[ok].sum())
    labels = sorted(acc)
    out = {w: np.array([acc[lab][w][0] / max(acc[lab][w][1], 1) for lab in labels]).reshape(len(labels), -1)
           for w in ("between", "precue", "response")}
    return labels, ({w: np.sqrt(v) for w, v in out.items()} if measure == "speed" else out)


def movement(fdir: Path, home: np.ndarray, freqs: np.ndarray, midline_x: float | None = None,
             band: str = "all", measure: str = "speed") -> tuple[pd.DataFrame, dict]:
    """Per session: contra / ipsi power and the asymmetry (contra - ipsi) / (contra + ipsi) per window; and per animal
    x epoch the per-point log2(epoch mean / pre mean) maps. cam4 IPSILESIONAL = IMAGE RIGHT (x > midline)."""
    labels, pw = point_measure(fdir, freqs, measure, band)
    mid = float(MASTER_MIDLINE_X if midline_x is None else midline_x)
    ipsi = home[:, 0] > mid
    rows = []
    for i, lab in enumerate(labels):
        r = {"label": lab}
        for w, M in pw.items():
            c, s = float(M[i, ~ipsi].mean()), float(M[i, ipsi].mean())
            r.update({f"{measure}_{w}_contra": c, f"{measure}_{w}_ipsi": s,
                      f"asym_{measure}_{w}": (c - s) / (c + s) if (c + s) > 0 else np.nan})
        rows.append(r)
    S = pd.DataFrame(rows)
    epoch_of = {lab: _epoch(lab) for lab in labels}
    maps = {"midline_x": mid, "ipsi_mask": ipsi, "band": band, "measure": measure}
    for a in sorted({lab[:4] for lab in labels}):
        idx = {ep: [i for i, lab in enumerate(labels) if lab[:4] == a and epoch_of[lab] == ep]
               for ep in ("pre", "acute", "subacute", "chronic")}
        if not idx["pre"]:
            continue
        for w, M in pw.items():
            pre = M[idx["pre"]].mean(0)
            for ep, ii in idx.items():
                if ii and ep != "pre":
                    maps[(a, ep, w)] = np.log2(np.maximum(M[ii].mean(0), 1e-12) / np.maximum(pre, 1e-12))
    return S, maps


def _epoch(label: str):
    from wfield_local import epochs
    return epochs.epoch_of(label)


def session_table(fdir: Path, names: list[str], home: np.ndarray, freqs: np.ndarray) -> pd.DataFrame:
    """Every readout, one row per session, with animal / epoch / day (curated epochs; June is not in the bundle)."""
    amp, _ = amplitudes(fdir, names)
    parts = [amp, reconstruction(fdir), evr_gap(fdir, names), movement(fdir, home, freqs, measure="speed")[0],
             movement(fdir, home, freqs, measure="share")[0]]
    S = parts[0]
    for p in parts[1:]:
        if len(p):
            S = S.merge(p, on="label", how="outer")
    S.insert(1, "animal", S.label.str[:4])
    S.insert(2, "epoch", [_epoch(x) for x in S.label])
    S.insert(3, "day", [_day(x) for x in S.label])
    return S.sort_values(["animal", "label"]).reset_index(drop=True)


def epoch_stats(S: pd.DataFrame, metrics: list[str], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-animal epoch means and paired change vs pre, through the cohort's one definition."""
    from wfield_local import orofacial_cohort as oc
    return oc.epoch_tables(S[S.epoch.isin(oc.EPOCH_ORDER)], seed=seed, metrics=metrics)


# ------------------------------------------------------------------------------------------- naming the components
#: face regions on the master image, relative to its LP nose (MASTER_MIDLINE_X, NOSE_Y) -- coarse, for naming only
NOSE_Y = 268.0
REGIONS = ("nose", "mouth_jaw", "whisker_pad", "upper_face", "other")


def region_of(home: np.ndarray) -> np.ndarray:
    """Coarse region per mesh point (master image coordinates)."""
    dx, dy = home[:, 0] - MASTER_MIDLINE_X, home[:, 1] - NOSE_Y
    r = np.full(len(home), "other", object)
    r[dy < -80] = "upper_face"
    r[(dy > -20) & (dy < 110) & (np.abs(dx) > 45) & (np.abs(dx) < 170)] = "whisker_pad"
    r[(dy > 60) & (np.abs(dx) < 70)] = "mouth_jaw"
    r[np.hypot(dx, dy) < 45] = "nose"
    return r


def _lick_rate(animal: str, date: str, t: np.ndarray) -> np.ndarray | None:
    """LP v36 licks (all tables) per session bin of ``t`` (DAQ s, regular), licks / s."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    f = oo.session_dir(PathResolver(), oc.RUN, animal, date) / "licks_all.csv"
    if not f.exists() or len(t) < 2:
        return None
    lk = pd.read_csv(f, usecols=["daq_peak_s"]).daq_peak_s.dropna().to_numpy()
    dt = float(np.median(np.diff(t)))
    edges = np.concatenate([t - dt / 2, [t[-1] + dt / 2]])
    return np.histogram(lk, edges)[0] / dt


def describe_components(fdir: Path, max_sessions: int = 8) -> pd.DataFrame:
    """Per frozen component: spectral peak, spatial weight share per region (+ contralesional share, rim share),
    correlation of its whole-session time course with LP lick rate and its response / between amplitude ratio over up
    to ``max_sessions`` PRE sessions, far-left vs far-right selectivity, and a SUGGESTED name (rules below; a
    suggestion for Priya, written to labels_suggested.json -- labels.json, which the loader uses, is hers)."""
    from wfield_local.orofacial_cohort import _repo_on_path
    _repo_on_path()
    from scripts.face_rhythm_mesh_qc import mesh_triangles, rim_triangles
    M = np.load(fdir / "master.npz")
    W, B, freqs, home = M["spatial"], M["spectral"], M["freqs"], M["home"]
    R = W.shape[-1]
    w2 = W[0] ** 2 + W[1] ** 2                                           # (P, R) spatial weight per point
    w2 = w2 / w2.sum(0, keepdims=True)
    reg = region_of(home)
    tri = mesh_triangles(home)
    rim_pts = np.zeros(len(home), bool)
    rim_pts[np.unique(tri[rim_triangles(tri)])] = True
    pre = _tasks(fdir, "proj")
    pre = pre[pre.epoch == "pre"].head(max_sessions)
    corr, ratio, farL, farR = [[] for _ in range(R)], [[] for _ in range(R)], np.zeros(R), np.zeros(R)
    nL = nR = 0
    for t in pre.itertuples():
        z = np.load(t.file)
        lr = _lick_rate(t.animal, t.date, z["session_t_daq_s"])
        S = z["session"].astype(float)
        ok = np.isfinite(S).all(1)
        tms = z["t_ms"] / 1000.0
        resp = np.nanmean(z["trials"][:, (tms >= 0) & (tms < 4)], (0, 1))
        for k in range(R):
            if lr is not None and ok.sum() > 10:
                corr[k].append(np.corrcoef(S[ok, k], lr[ok])[0, 1])
            ratio[k].append(resp[k] / max(np.nanmean(S[ok, k]), 1e-12))
        pos = z["position"]
        for p_, acc in (("far_L", "L"), ("far_R", "R")):
            m = pos == p_
            if m.any():
                v = np.nanmean(z["trials"][m][:, (tms >= 0) & (tms < 4)], (0, 1))
                if acc == "L":
                    farL += v
                    nL += 1
                else:
                    farR += v
                    nR += 1
    rows = []
    for k in range(R):
        share = {r: float(w2[reg == r, k].sum()) for r in REGIONS}
        pk = float(freqs[np.argmax(B[:, k])])
        lc = float(np.nanmedian(corr[k])) if corr[k] else np.nan
        rt = float(np.nanmedian(ratio[k])) if ratio[k] else np.nan
        sel = (farL[k] / max(nL, 1) - farR[k] / max(nR, 1)) / max(farL[k] / max(nL, 1) + farR[k] / max(nR, 1), 1e-12)
        rim = float(w2[rim_pts, k].sum())
        top = max(share, key=share.get)
        if rim > 0.5:
            name = "rim"
        elif np.isfinite(lc) and lc >= 0.3:
            name = "lick" + ("_farL" if sel > 0.3 else "_farR" if sel < -0.3 else "")
        elif pk < 2.0:
            name = "slow"
        elif 2.5 <= pk <= 5.5 and top == "nose":
            name = "sniff"
        elif 5.5 < pk <= 14 and top == "whisker_pad":
            name = "whisk"
        else:
            name = f"c{k}"
        rows.append({"comp": k, "suggested": name, "peak_hz": pk, "top_region": top, **{f"w_{r}": v for r, v in
                     share.items()}, "w_rim": rim, "w_contra": float(w2[home[:, 0] <= MASTER_MIDLINE_X, k].sum()),
                     "lick_rate_corr": lc, "response_over_session": rt, "farL_vs_farR": float(sel),
                     "n_pre_sessions": len(pre)})
    D = pd.DataFrame(rows)
    names, seen = [], {}
    for n in D.suggested:                                                # make names unique: whisk, whisk2, ...
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else f"{n}{seen[n]}")
    D["suggested"] = names
    return D

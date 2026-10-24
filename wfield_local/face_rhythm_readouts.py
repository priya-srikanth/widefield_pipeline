"""FaceRhythm session readouts on the frozen master pre-stroke factor set (`scripts/face_rhythm_factors.py` outputs
``<set>/proj/<key>.npz`` and ``<set>/own/<key>.npz``). Computation only; figures + CLI: `scripts/face_rhythm_readouts.py`;
pipeline overview: docs/FACE_RHYTHM_PIPELINE.md.

The 2pRAM-cohort manuscript's FR analyses (FACE_RHYTHM_PLAN.md "What the 2pRAM-cohort manuscript does with FR"), here
ACROSS animals (one mesh, one frozen pre-stroke basis):
  amplitudes     pre-stroke components refit to every session: mean factor amplitude per session -- as a SHARE of
                 the frame's movement (amp_) and as ABSOLUTE expression (abs_ = weights x FR normaliser D) -- in the pre-cue
                 (cue -2..0 s) and response (0..+4 s) windows of engaged trials (25 Hz), BETWEEN trials (5 Hz bins
                 outside cue -3..+6 s, LARGE MOVEMENT excluded; the MEDIAN, so rare large events cannot drive it) and in
                 LARGE MOVEMENT (`large_movement`, + large_frac) -- "non-task movements preserved?" is between vs
                 response; per position too.
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


def large_movement(ts: np.ndarray, D: np.ndarray, cues: np.ndarray, pct: float = 99.0, min_s: float = 0.4,
                   widen_s: float = 1.0) -> np.ndarray:
    """Boolean per session bin: LARGE non-task MOVEMENT -- whole-face movement (FR normaliser D) above the session's
    ``pct`` percentile of response-window bins (cue 0..+4 s: the licking range), outside response windows, in runs of
    >= ``min_s``, widened by ``widen_s``. NOT a grooming detector (10-21, Priya's calls on PS92_0806 clips): it catches
    the big grooming bouts but also vigorous spontaneous licking, and misses small grooming; neither LP licks (LP
    reports licks during grooming) nor regional speed (upper face / nose vs mouth) separated them. A grooming label
    needs the paws or hand-labelled bouts (factor clips --window large)."""
    resp = np.zeros(len(ts), bool)
    for c in cues:
        resp |= (ts >= c + WINDOWS["response"][0]) & (ts < c + WINDOWS["response"][1])
    if not resp.any() or not np.isfinite(D).any():
        return np.zeros(len(ts), bool)
    hi = (D > np.nanpercentile(D[resp], pct)) & ~resp
    dt = float(np.median(np.diff(ts)))
    out = np.zeros(len(ts), bool)
    i = 0
    while i < len(hi):
        if hi[i]:
            j = i
            while j < len(hi) and hi[j]:
                j += 1
            if (j - i) * dt >= min_s:
                w = int(round(widen_s / dt))
                out[max(i - w, 0):j + w] = True
            i = j
        else:
            i += 1
    return out


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


def amplitudes(fdir: Path, names: list[str], absolute: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per session: amp_<comp>_<precue|response|between>, per session x position x comp: response amplitude).
    ``absolute``: weights x each bin's FR normaliser D -> the component's absolute expression (columns ``abs_``);
    default = its SHARE of the frame's movement."""
    pre_ = "abs" if absolute else "amp"
    sums, pos_rows = {}, []
    for t in _tasks(fdir, "proj").itertuples():
        z = np.load(t.file)
        tr, tms = z["trials"].astype(float), z["t_ms"] / 1000.0
        if absolute:
            tr = tr * z["trials_D"][..., None]
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
                    pos_rows += [{"label": t.label, "position": str(p), "comp": names[k], pre_: float(xp[:, k].mean()),
                                  "n_bins": len(xp)} for k in range(tr.shape[-1]) if len(xp)]
        ts, ws, cues = z["session_t_daq_s"], z["session"].astype(float), np.sort(z["cue_daq_s"])
        if absolute:
            ws = ws * z["session_D"][:, None]
        k = np.searchsorted(cues, ts)
        near = np.zeros(len(ts), bool)
        for kk in (k - 1, k):
            ok = (kk >= 0) & (kk < len(cues))
            d = ts - cues[np.clip(kk, 0, len(cues) - 1)]
            near |= ok & (d >= BUSY_S[0]) & (d <= BUSY_S[1])
        gm = large_movement(ts, z["session_D"], cues) if "session_D" in z.files else np.zeros(len(ts), bool)
        for w_, sel in (("between", ~near & ~gm), ("large", gm)):         # MEDIANS: bins kept, summarised below
            x = ws[sel]
            acc.setdefault(f"_bins_{w_}", []).append(x[np.isfinite(x).all(1)])
        g0 = acc.get("_large_frac", (0.0, 0))
        acc["_large_frac"] = (g0[0] + gm.sum(), g0[1] + len(gm))
    rows = []
    for lab, acc in sums.items():
        r = {"label": lab}
        gf = acc.pop("_large_frac", (0.0, 0))
        r["large_frac"] = gf[0] / gf[1] if gf[1] else np.nan
        for w_ in ("between", "large"):
            xb = np.concatenate(acc.pop(f"_bins_{w_}", [np.empty((0, len(names)))]))
            r.update({f"{pre_}_{names[k]}_{w_}": (float(np.median(xb[:, k])) if len(xb) else np.nan)
                      for k in range(len(names))})
        for w, (s, n) in acc.items():
            r.update({f"{pre_}_{names[k]}_{w}": (s[k] / n if n else np.nan) for k in range(len(names))})
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(pos_rows)


def reconstruction(fdir: Path) -> pd.DataFrame:
    """Per session: median relative reconstruction error of trial-window and whole-session bins (frozen basis). A
    PRE session is scored on its HELD-OUT basis (``proj_loso/``, 10-23) when that exists -- error on the full basis is
    in-sample for pre; sets made before 10-23 already hold held-out pre projections in ``proj/``."""
    acc = {}
    for t in _tasks(fdir, "proj").itertuples():
        f_lo = fdir / "proj_loso" / Path(t.file).name
        z = np.load(f_lo if t.epoch == "pre" and f_lo.exists() else t.file)
        a = acc.setdefault(t.label, ([], []))
        a[0].append(z["trials_err"].ravel())
        a[1].append(z["session_err"].ravel())
    return pd.DataFrame([{"label": k, "recon_err_trials": float(np.nanmedian(np.concatenate(v[0]))),
                          "recon_err_session": float(np.nanmedian(np.concatenate(v[1])))} for k, v in acc.items()])


def own_sub(fdir: Path) -> str:
    """The per-session own-fit folder to read: ``own8`` (8 starts, 10-24 rerun) once it is COMPLETE (as many files as
    ``own``), else ``own`` (3 starts). More starts = the session's best solution more often found: a tighter pre range
    for the factor angles and an unbiased EVR gap (a missed optimum makes own EVR, hence the gap, too low)."""
    n = lambda d: len(list((fdir / d).glob("*.npz"))) if (fdir / d).exists() else 0  # noqa: E731
    return "own8" if n("own8") and n("own8") >= n("own") else "own"


def loso_floor(fdir: Path) -> pd.DataFrame:
    """Leave-one-pre-day-out EVR per pre day (``loso/rank<R>_<day>.json`` at the master's rank): held-out EVR = the
    pre-stroke level a session's frozen EVR is read against (FINDINGS 3)."""
    import json
    R = int(np.load(fdir / "master.npz")["rank"])
    rows = [json.loads(f.read_text()) for f in sorted((fdir / "loso").glob(f"rank{R:02d}_*.json"))]
    return pd.DataFrame(rows)[["animal", "day", "evr_held_out", "evr_in_fit"]] if rows else pd.DataFrame()


def evr_gap(fdir: Path, names: list[str]) -> pd.DataFrame:
    """Per session (mean over its videos): EVR own / frozen / gap, and matched cosine of own to frozen factors
    (mean + per component)."""
    rows = []
    for t in _tasks(fdir, own_sub(fdir)).itertuples():
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
    ab, _ = amplitudes(fdir, names, absolute=True)
    parts = [amp, ab, reconstruction(fdir), evr_gap(fdir, names), movement(fdir, home, freqs, measure="speed")[0],
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


def lick_split(names: list[str]) -> tuple[list[str], list[str]]:
    """(lick-locked components, the rest) by NAME: "lick" in the name, lick INITIATION ("prelick") excluded -- the
    master's labels.json (FINDINGS 4a); matches fr_groups.master_v1's lick part for master_pre_v1."""
    lick = [n for n in names if "lick" in n and "prelick" not in n]
    return lick, [n for n in names if n not in lick]


def task_nontask(S: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """The manuscript's "relative preservation of NON-TASK orofacial movements" (2pRAM cohort, Fig 4): per session,
    TASK = summed absolute expression of the lick components in the RESPONSE window; NON-TASK = summed absolute
    expression of the other components BETWEEN trials (median bins, large movement excluded). -> per animal x epoch,
    each as the ratio to the animal's pre-stroke mean (rule 8: per animal; one mean per session)."""
    lick, other = lick_split(names)
    T = S[["label", "animal", "epoch"]].copy()
    T["task"] = S[[f"abs_{n}_response" for n in lick]].sum(1, min_count=len(lick))
    T["nontask"] = S[[f"abs_{n}_between" for n in other]].sum(1, min_count=len(other))
    A = T.groupby(["animal", "epoch"])[["task", "nontask"]].mean().reset_index()
    pre = A[A.epoch == "pre"].set_index("animal")[["task", "nontask"]]
    for c in ("task", "nontask"):
        A[f"{c}_vs_pre"] = A[c] / A.animal.map(pre[c])
    return A


def severity_table(S: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """One row per animal: lesion burden (histology, caudoputamen ipsi / contra % area; LOWER = more tissue lost) and
    its FaceRhythm deficit -- lick (task) and non-task expression vs pre, acute and chronic; chronic minus pre
    reconstruction error, EVR gap and morphing where present. DESCRIPTIVE: n = 4 animals, no test (rule 8)."""
    from wfield_local import histology_area as ha
    H = ha.per_animal(ha.load())
    A = task_nontask(S, names).set_index(["animal", "epoch"])
    rows = []
    for an in sorted(S.animal.unique()):
        r = {"animal": an, "burden_ipsi_contra_pct": H["mean"].get(an, np.nan), "burden_sem": H["sem"].get(an, np.nan)}
        for ep in ("acute", "chronic"):
            for c in ("task", "nontask"):
                r[f"{c}_vs_pre_{ep}"] = A[f"{c}_vs_pre"].get((an, ep), np.nan)
        g = S[S.animal == an].groupby("epoch").mean(numeric_only=True)
        for c in ("recon_err_trials", "evr_gap", "cos_spatial_mean", "cos_spectral_mean"):
            if c in g and "pre" in g.index and "chronic" in g.index:
                r[f"{c}_chronic_minus_pre"] = g.loc["chronic", c] - g.loc["pre", c]
        rows.append(r)
    return pd.DataFrame(rows)


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

"""Is the ACAd left/right lick-response result REAL CORTEX, or the midline (vessel / registration / neighbors)?

QUESTION (10-29). check 2 of lick_sspn_checks.py: the weight-free same-place right - left difference d = dh_right -
dh_left of the ITI lick response in ACAd tracks recovery within animals about as strongly as SSp-n does (ITI -> iti_ang,
mean r 0.59, MixedLM p < 0.005). Left and right ACAd are ~0.33 mm strips either side of the Allen MIDLINE, over the
superior sagittal sinus, so d could come from (a) the VESSEL (hemodynamic / shadow artifact), (b) REGISTRATION (a
session-to-session left/right shift of the Allen midline moves sinus pixels from one ACAd into the other) or (c)
SPILL-OVER from MOs / PL. Priya chooses chemogenetic-grant regions from this.

MEASURE (identical to check 2 unless stated). Per session and mask: trace = V @ mean(U_atlas over the mask's pixels)
(V = adopted hemo-corrected SVT), slow drift (>= 600 s cosines) removed, x1e3 (iti_licks.extract); per lick the mean over
lags 0..0.2 s (iti_where_decoder W_d); divided by the trace's frame SD (iti_zscore_check.session_sd); per session the
like-for-like change dh = mean over licks of (x - Gaussian same-place mean of the animal's OTHER pre-session licks),
sigma 0.1 mm on the actual lateral landing -- lick_offset_contrib_map.session_dh, called on the SAME licks
iti_lick_order.predictions selects (its CONTRIB_HOOK; extra columns are appended to the area basis, so the lick set
and reference are exactly those of check 2). d = dh_right - dh_left (contra - ipsi; check 2's sign). Validation: the
full-mask ACAd rebuilt here reproduces check 2's ACAd d (terms_sessions_areas_splitr8.csv).
Recovery test: lick_sspn_checks._track (per-animal r on post sessions, then MixedLM yy ~ w + b + logday, groups =
animal); sets ITI -> iti_ang (headline), ITI -> aim, cued -> aim.

CHECKS
1 MIDLINE DISTANCE: ACAd masks restricted by column distance from the Allen midline (col 319.5 on the 540 x 640 atlas
  grid, 0.0194 mm/px): all; >= 0.1 mm; >= 0.2 mm (ACAd is only 0.33 mm wide after the stat mask, so 0.4 mm leaves nothing);
  the strip < 0.1 mm alone; 0.1-0.2 mm alone.
2 VESSEL: the same d on the RAW 415 nm half of SVT and the raw 470 (same U, drift removed, own frame SD); F0 asymmetry
  log2(F0_right / F0_left) of frames_average_atlas (470 and 415) per ACAd mask.
3 REGISTRATION: per session the actual midline in the session's F0 image (atlas grid): the sinus VALLEY (min of the
  row-averaged column profile, rows 120-300, linear trend removed, parabolic sub-pixel) and the MIRROR-SYMMETRY axis
  (high-passed image); offset = axis - 319.5 px. Offset by epoch, its own recovery link, d with the offset partialled
  within animal, and a MIDLINE-CORRECTED d (both ACAd masks moved by the session's rounded valley offset, -4..4 px).
4 NEIGHBORS: ACAd d residualised within animal (post sessions) on the MOs_rest and PL d of check 2.
5 PIXEL MAP: per pixel the same dh (pixel trace scaled by its own frame SD; pixels on a 2-px grid of the frontal midline
  crop), mirrored d(p) = dh(p) - dh(mirror p) at right-hemisphere positions, then per animal the within-animal r with
  iti_ang over post sessions; the mean-of-animals map and the per-animal maps, adopted and 415 nm.
EXTRAS (`extras`): REGISTRATION SIZE = per session the sensitivity of d to a 1-px mask shift ((d[+1] - d[-1]) / 2) times
the session's valley offset from the animal's mean -> the d change a registration error of that size can produce (SD vs
the SD of d, r with recovery); d with both masks moved by that RELATIVE offset (linear interpolation over SHIFTS, both
signs); every measure's within-animal partial r with iti_ang controlling log(day). PROFILE = the pixel r maps averaged
over ACAd's rows (118-302) per column distance from the midline.
Outputs (il.OUT_W): acad_control_sessions.csv (per session: every d, F0 asymmetry, midline), acad_control_track.csv
(every recovery test), acad_control_validation.csv, acad_control_confounds.csv, acad_control_registration.csv,
acad_control_profile.csv, acad_control_pixmaps.npz; figure figures/lick_acad_midline_control.png.
Usage: python scripts/lick_acad_midline_control.py [--jobs 6] [--cache DIR] | --plot-only
(--cache = a LOCAL folder for the per-session extraction pickle; never written on M:).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
H, W = 540, 640
MID = 319.5                         # Allen midline between columns 319 | 320 (ACAd_left ends at 319, ACAd_right starts at 320)
MM = 0.0194                         # mm per px (allen_register.RESOLUTION)
VARIANTS = {"all": (0.0, 99.0), "d>=0.1": (0.1, 99.0), "d>=0.2": (0.2, 99.0), "strip<0.1": (0.0, 0.1),
            "0.1-0.2": (0.1, 0.2)}   # mm from the midline (pixel centre)
SHIFTS = list(range(-4, 5))         # px, both ACAd masks moved together (registration sensitivity / correction)
CHANS = ("neu", "r470", "h415")
ROWS = np.arange(90, 312, 2)        # pixel-map crop (2-px grid)
RCOLS = np.arange(320, 380, 2)      # right-hemisphere columns; mirror = 639 - c
PROF_ROWS = (120, 300)              # midline estimate rows (ACAd extent 118-302)
OUT = il.OUT_W
SESS_CSV, TRACK_CSV, VAL_CSV, PIX_NPZ = (OUT / "acad_control_sessions.csv", OUT / "acad_control_track.csv",
                                         OUT / "acad_control_validation.csv", OUT / "acad_control_pixmaps.npz")
FIG = OUT / "figures" / "lick_acad_midline_control.png"


def _cols() -> list[str]:
    out = []
    for v in VARIANTS:
        out += [f"ACAd[{v}]_left", f"ACAd[{v}]_right"]
    for s in SHIFTS:
        out += [f"ACAd[shift{s:+d}]_left", f"ACAd[shift{s:+d}]_right"]
    return out


def _masks(M: dict, flat: np.ndarray) -> dict:
    """{column name: boolean over brain pixels} for every ACAd variant (_cols order)."""
    c = flat % W
    dist = (np.abs(c - MID)) * MM
    out = {}
    for v, (lo_, hi_) in VARIANTS.items():
        for h in ("left", "right"):
            out[f"ACAd[{v}]_{h}"] = M[f"ACAd_{h}"] & (dist >= lo_ - 1e-9) & (dist < hi_)
    inb = np.zeros(H * W, bool)
    inb[flat] = True
    for s in SHIFTS:
        for h in ("left", "right"):
            img = np.zeros(H * W, bool)
            img[flat[M[f"ACAd_{h}"]]] = True
            img = np.roll(img.reshape(H, W), s, axis=1).ravel()
            out[f"ACAd[shift{s:+d}]_{h}"] = (img & inb)[flat]
    return out


def _midline(img: np.ndarray) -> tuple[float, float, float]:
    """(valley column, mirror axis column, valley contrast) of one F0 image (H x W, atlas grid)."""
    from scipy import ndimage
    r0, r1 = PROF_ROWS
    c0, c1 = 290, 350
    prof = np.nanmean(img[r0:r1, c0:c1], 0)
    x = np.arange(c0, c1, dtype=float)
    p = prof - np.polyval(np.polyfit(x, prof, 1), x)
    k = int(np.argmin(p[5:-5])) + 5
    y0, y1, y2 = p[k - 1], p[k], p[k + 1]
    den = y0 - 2 * y1 + y2
    valley = x[k] + (0.5 * (y0 - y2) / den if den > 0 else 0.0)
    flank = np.r_[prof[max(k - 20, 0):max(k - 12, 0)], prof[k + 12:k + 20]]
    contrast = float((np.nanmean(flank) - prof[k]) / np.nanmean(flank)) if len(flank) else np.nan
    im = np.nan_to_num(img.astype(float))
    hp = im - ndimage.gaussian_filter(im, 15)
    band = hp[r0:r1]
    best, axis = -np.inf, np.nan
    for m in np.arange(310.0, 329.5, 0.5):
        cc = np.arange(int(np.ceil(m - 60)), int(np.floor(m)) + 1)
        mc = (2 * m - cc).astype(int)
        ok = (mc < W) & (cc >= 0) & (mc != cc)
        a, b = band[:, cc[ok]].ravel(), band[:, mc[ok]].ravel()
        r = np.corrcoef(a, b)[0, 1]
        if r > best:
            best, axis = r, m
    return float(valley), float(axis), contrast


def work(label: str) -> dict:
    """One session (module-level for spawn): per lick (ITI + cued, session-file rows) the 0..0.2 s mean of every ACAd
    variant trace for the adopted (neu), raw 470 and raw 415 signals + frame SDs; F0 per mask; midline estimates;
    per-lick SVD features and SD-scaled U over the pixel-map crop (adopted + 415)."""
    import json

    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local.hemo_variants import functional_channel
    d = il._load(label)
    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    if tuple(int(q) for q in shape) != (H, W):
        raise ValueError(f"{label}: grid {shape}")
    n = len(ft)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    MK = _masks(M, flat)
    cols = list(MK)
    fc = functional_channel(label)
    man = json.loads(Path(config.hemo_manifest_path(res)).read_text())   # the selected variant's (one hemo switch)
    if int(man.get("functional_channel", fc)) != fc:
        raise RuntimeError(f"{label}: functional slot {fc} vs manifest")
    Ra = np.stack([Ub[MK[k]].mean(0) if MK[k].any() else np.full(Ub.shape[1], np.nan) for k in cols], 1).astype(np.float64)
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    lw = il.LAGS_D[(il.LAGS_D >= 0.0) & (il.LAGS_D <= 0.2)]
    lt = {"iti": d["iti_t"].astype(float), "trial": d["tr_t"].astype(float)}
    svt = np.asarray(np.load(res / "SVT.npy", mmap_mode="r")[:, :2 * n], np.float32)
    out = {"label": label, "cols": cols, "npix": {k: int(v.sum()) for k, v in MK.items()}, "X": {}, "sd": {}}
    # pixel crop: right positions + their mirrors
    rr, cc = np.meshgrid(ROWS, RCOLS, indexing="ij")
    pr = (rr * W + cc).ravel()
    pl = (rr * W + (639 - cc)).ravel()
    pos = np.full(H * W, -1)
    pos[flat] = np.arange(len(flat))
    ir, il_ = pos[pr], pos[pl]
    okp = (ir >= 0) & (il_ >= 0)
    pix = {"okp": okp}
    for ch in CHANS:
        if ch == "neu":
            Vc = V[:n].astype(np.float64)
        else:
            slot = fc if ch == "r470" else (fc + 1) % 2
            Vc = svt[:, slot::2][:, :n].T.astype(np.float64)
        Vc -= Q @ (Q.T @ Vc)
        Vc *= 1e3
        R = Vc @ Ra
        out["sd"][ch] = R.std(0)
        for st, t in lt.items():
            out["X"][(ch, st)] = il._sample(R, ft, t, lw).astype(np.float64).mean(1) if len(t) else np.zeros((0, len(cols)))
        if ch == "neu":                                       # validation: reproduce the stored session-file ACAd columns
            sc = [str(q) for q in d["cols"]]
            wd_ = (il.LAGS_D >= 0.0) & (il.LAGS_D <= 0.2)
            ref = d["iti_X"].astype(float)[:, wd_].mean(1)[:, [sc.index("ACAd_left"), sc.index("ACAd_right")]]
            mine = out["X"][("neu", "iti")][:, [cols.index("ACAd[all]_left"), cols.index("ACAd[all]_right")]]
            out["val_maxrel"] = float(np.nanmax(np.abs(mine - ref)) / max(np.nanpercentile(np.abs(ref), 99), 1e-9))
        if ch in ("neu", "h415"):
            Vm = Vc - Vc.mean(0)
            C = (Vm.T @ Vm) / len(Vm)
            Ui = np.zeros((len(pr) * 2, Ub.shape[1]), np.float64)
            Ui[:len(pr)][okp] = Ub[ir[okp]]
            Ui[len(pr):][okp] = Ub[il_[okp]]
            sdp = np.sqrt(np.maximum(np.einsum("pk,kl,pl->p", Ui, C, Ui), 1e-12))
            pix[f"Un_{ch}"] = (Ui / sdp[:, None]).astype(np.float32)
            pix[f"v_{ch}"] = il._sample(Vc, ft, lt["iti"], lw).mean(1).astype(np.float32) if len(lt["iti"]) else None
        del Vc, R
    fa = np.load(res / "allen_aligned_affine8v1" / "frames_average_atlas.npy").astype(np.float64).reshape(2, H, W)
    f0 = {}
    for nm, slot in (("470", fc), ("415", (fc + 1) % 2)):
        im = fa[slot]
        for k in cols:
            f0[f"f0_{nm}|{k}"] = float(np.nanmean(im.ravel()[flat[MK[k]]])) if MK[k].any() else np.nan
        v_, a_, ct = _midline(im)
        f0[f"valley_{nm}"], f0[f"mirror_{nm}"], f0[f"contrast_{nm}"] = v_, a_, ct
    out["f0"] = f0
    pix["f0img"] = fa[fc][ROWS[0]:ROWS[-1] + 2, 260:380].astype(np.float32)
    pa = np.zeros(H * W, int)
    pa[flat] = parea
    pix["area_img"] = pa.reshape(H, W)[ROWS[0]:ROWS[-1] + 2, 260:380]
    pix["area_names"] = dict(names)
    out["pix"] = pix
    out["acad_cols"] = (int((flat % W)[M["ACAd_left"]].min()), int((flat % W)[M["ACAd_right"]].max()))
    return out


def extract(jobs: int, cache: Path | None) -> dict:
    import iti_where_decoder as wd

    from wfield_local import analysis_kit as ak
    if cache is not None and (cache / "acad_extract.pkl").exists():
        return pickle.loads((cache / "acad_extract.pkl").read_bytes())
    labs = wd.sess_labels()
    res, fails = ak.fan_sessions(labs, work, jobs=jobs, label="acad extract")
    if fails:
        raise RuntimeError(f"extract failed for {[f[0] for f in fails]}: {fails[0][1]}")
    E = {x: r for x, r in res}
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "acad_extract.pkl").write_bytes(pickle.dumps(E, protocol=5))
    return E


# ------------------------------------------------------------------------------------------------ same-place dh
HOOK: dict = {}


def _weights(S: dict, sigma: float, min_ref: int) -> dict:
    """Replica of lick_offset_contrib_map.session_dh's lateral rule returning, per session, the kept rows and the mean
    reference weight per pre lick: dh = X[ok].mean(0) - wbar @ Xp (validated against session_dh on the columns)."""
    pre = [x for x, v in S.items() if v[0] == "pre"]
    ap = np.concatenate([S[x][2] for x in pre])
    lab = np.concatenate([np.full(len(S[x][2]), x, dtype=object) for x in pre])
    out = {}
    for x, (_ep, _X, act, *_r) in S.items():
        ref = lab != x
        K = np.exp(-0.5 * ((act[:, None] - ap[None, ref]) / sigma) ** 2)
        ok = K.sum(1) > 1e-6
        if ok.sum() < min_ref:
            continue
        Kn = K[ok] / K[ok].sum(1, keepdims=True)
        wbar = np.zeros(len(ap))
        wbar[ref] = Kn.mean(0)
        out[x] = (ok, wbar, lab)
    return out


def _hook_factory(ncol: int):
    import lick_offset_contrib_map as lcm

    def hook(st, a, S, fits):
        def dec0(_x, X):
            return np.zeros(len(X))
        sess = lcm.session_dh(S, dec0, lcm.SIGMA)
        wts = _weights(S, lcm.SIGMA, lcm.MIN_REF)
        HOOK[(st, a)] = {x: (S[x][0], v[1][-ncol - 1:-1], S[x][1][:, -1].astype(int), *wts[x]) for x, v in sess.items()}
        HOOK[(st, a, "rows")] = {x: S[x][1][:, -1].astype(int) for x in S}
        HOOK[(st, a, "full")] = {x: v[1] for x, v in sess.items()}
        HOOK[(st, a, "check")] = {x: float(np.max(np.abs(v[1] - (S[x][1][wts[x][0]].mean(0) - wts[x][1] @ np.vstack(
            [S[q][1] for q in S if S[q][0] == "pre"]))))) for x, v in sess.items()}
    return hook


def compute(E: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Runs iti_lick_order.predictions with the ACAd variant columns (+ a row-index column) appended to the basis."""
    import iti_lick_order as lo
    import iti_where_decoder as wd
    cols = next(iter(E.values()))["cols"]
    extra = [f"{ch}|{k}" for ch in CHANS for k in cols]
    orig = wd.area_basis

    def patched(sess_dir=None, jobs=None, basis=None):
        D, names, SD = orig(sess_dir, jobs, basis)
        add = {}
        for x, rec in D.items():
            e = E[x]
            for st in ("iti", "trial"):
                X, C = rec[st]
                Xe = np.concatenate([e["X"][(ch, st)] for ch in CHANS], 1)
                rec[st] = (np.concatenate([X, Xe, np.arange(len(X))[:, None]], 1), C)
            add[x] = np.concatenate([e["sd"][ch] for ch in CHANS] + [np.ones(1)])
        SD2 = SD.join(pd.DataFrame(add, index=[*extra, "_row"]).T)
        return D, [*names, *extra, "_row"], SD2
    lo.wd.area_basis = patched
    lo.CONTRIB_HOOK = _hook_factory(len(extra))
    try:
        lo.predictions()
    finally:
        lo.wd.area_basis = orig
        lo.CONTRIB_HOOK = None
    names = wd.area_meta()[0]
    rows, val = [], []
    for key, sess in HOOK.items():
        if len(key) != 2:
            continue
        st, a = key
        for x, (ep, dh, _rows, _ok, _w, _lab) in sess.items():
            r = {"set": st, "animal": a, "label": x, "epoch": ep}
            dd = dict(zip(extra, dh))
            for ch in CHANS:
                for v in [*VARIANTS, *[f"shift{s:+d}" for s in SHIFTS]]:
                    r[f"d_{ch}[{v}]"] = dd[f"{ch}|ACAd[{v}]_right"] - dd[f"{ch}|ACAd[{v}]_left"]
            full = dict(zip(names, HOOK[(st, a, "full")][x][:len(names)]))
            for ar in ("ACAd", "MOs_rest", "PL", "SSp-n"):
                r[f"chk2_{ar}"] = full[f"{ar}_right"] - full[f"{ar}_left"]
            val.append({"set": st, "label": x, "weights_vs_session_dh_maxabs": HOOK[(st, a, "check")][x]})
            rows.append(r)
    S = pd.DataFrame(rows)
    # F0 / midline
    F = pd.DataFrame([{"label": x, **e["f0"]} for x, e in E.items()])
    for ch in ("470", "415"):
        for v in VARIANTS:
            F[f"f0asym_{ch}[{v}]"] = np.log2(F[f"f0_{ch}|ACAd[{v}]_right"] / F[f"f0_{ch}|ACAd[{v}]_left"])
        F[f"valley_off_{ch}"] = F[f"valley_{ch}"] - MID
        F[f"mirror_off_{ch}"] = F[f"mirror_{ch}"] - MID
    keep = [c for c in F.columns if c.startswith(("f0asym", "valley", "mirror", "contrast"))]
    S = S.merge(F[["label", *keep]], on="label", how="left")
    # midline-corrected d: masks moved by the session's rounded valley offset (470)
    sh = np.clip(np.round(S.valley_off_470).astype(int), SHIFTS[0], SHIFTS[-1])
    for ch in CHANS:
        S[f"d_{ch}[corrected]"] = [S.loc[i, f"d_{ch}[shift{s:+d}]"] for i, s in zip(S.index, sh)]
    T = pd.read_csv(il.OUT_W / "offset_contrib" / "terms_sessions_areas_splitr8.csv")
    T = T[T.feature.isin(["ACAd_right", "ACAd_left"])].pivot_table(index=["set", "label"], columns="feature", values="dh")
    ref = (T.ACAd_right - T.ACAd_left).rename("ref").reset_index()
    V = S[["set", "label", "d_neu[all]", "chk2_ACAd"]].merge(ref, on=["set", "label"], how="left")
    Vd = pd.DataFrame(val).merge(V, on=["set", "label"])
    Vd["trace_maxrel"] = Vd.label.map({x: e["val_maxrel"] for x, e in E.items()})
    pix = pixel_maps(E)
    return S, pix, Vd


def pixel_maps(E: dict) -> dict:
    """Per (set iti, animal): per session the per-pixel same-place dh (adopted + 415) mirrored d at right positions."""
    out = {}
    for key, sess in HOOK.items():
        if len(key) != 2 or key[0] != "iti":
            continue
        _st, a = key
        rows_all = HOOK[(key[0], a, "rows")]
        for ch in ("neu", "h415"):
            D = {}
            for x, (ep, _dh, _rws, ok, wbar, lab) in sess.items():
                e = E[x]["pix"]
                rx = rows_all[x][ok]
                own = e[f"Un_{ch}"] @ e[f"v_{ch}"][rx].astype(np.float64).mean(0)
                refc = np.zeros_like(own)
                for q in sorted(set(lab)):
                    wq = wbar[lab == q]
                    if wq.sum() == 0:
                        continue
                    vq = E[q]["pix"][f"v_{ch}"][rows_all[q]].astype(np.float64)
                    refc += E[q]["pix"][f"Un_{ch}"] @ (wq @ vq)
                dh = own - refc
                npx = len(dh) // 2
                dd = dh[:npx] - dh[npx:]
                dd[~e["okp"]] = np.nan
                D[x] = (ep, dd.astype(np.float32))
            out[(a, ch)] = D
    return out


# ------------------------------------------------------------------------------------------------ tests
def _resid(S: pd.DataFrame, y: str, xs: list[str]) -> pd.Series:
    """y residualised within (set, animal) on xs over the POST sessions (+ the animal's mean added back); pre = NaN."""
    out = pd.Series(np.nan, index=S.index)
    for _k, g in S[S.epoch != "pre"].groupby(["set", "animal"]):
        g = g.dropna(subset=[y, *xs])
        if len(g) < len(xs) + 3:
            continue
        X = np.column_stack([np.ones(len(g)), g[xs].to_numpy(float)])
        b = np.linalg.lstsq(X, g[y].to_numpy(float), rcond=None)[0]
        out.loc[g.index] = g[y].to_numpy(float) - X[:, 1:] @ b[1:]
    return out


def tests(S: pd.DataFrame) -> pd.DataFrame:
    import lick_sspn_checks as lsc
    S = S.copy()
    S["resid_offset"] = _resid(S, "d_neu[all]", ["valley_off_470"])
    S["resid_offset_mirror"] = _resid(S, "d_neu[all]", ["mirror_off_470"])
    S["resid_f0asym"] = _resid(S, "d_neu[all]", ["f0asym_470[all]"])
    S["resid_415"] = _resid(S, "d_neu[all]", ["d_h415[all]"])
    S["resid_MOs_rest"] = _resid(S, "d_neu[all]", ["chk2_MOs_rest"])
    S["resid_PL"] = _resid(S, "d_neu[all]", ["chk2_PL"])
    S["resid_MOs_rest+PL"] = _resid(S, "d_neu[all]", ["chk2_MOs_rest", "chk2_PL"])
    S["MOs_rest_resid_ACAd"] = _resid(S, "chk2_MOs_rest", ["d_neu[all]"])
    S["SSp-n_resid_ACAd"] = _resid(S, "chk2_SSp-n", ["d_neu[all]"])
    S["resid_all_confounds"] = _resid(S, "d_neu[all]", ["valley_off_470", "f0asym_470[all]", "d_h415[all]",
                                                         "chk2_MOs_rest", "chk2_PL"])
    groups = {
        "0 reference": ["chk2_ACAd", "d_neu[all]", "chk2_SSp-n", "chk2_MOs_rest", "chk2_PL"],
        "1 midline distance": [f"d_neu[{v}]" for v in VARIANTS],
        "2 vessel / hemodynamic": [f"d_h415[{v}]" for v in VARIANTS] + [f"d_r470[{v}]" for v in VARIANTS]
        + [f"f0asym_470[{v}]" for v in VARIANTS] + [f"f0asym_415[{v}]" for v in VARIANTS]
        + ["resid_415", "resid_f0asym"],
        "3 registration": ["valley_off_470", "valley_off_415", "mirror_off_470", "mirror_off_415", "contrast_470",
                           "resid_offset", "resid_offset_mirror", "d_neu[corrected]"]
        + [f"d_neu[shift{s:+d}]" for s in SHIFTS],
        "4 neighbors": ["resid_MOs_rest", "resid_PL", "resid_MOs_rest+PL", "MOs_rest_resid_ACAd", "SSp-n_resid_ACAd",
                         "resid_all_confounds"],
    }
    out = []
    for g, cols in groups.items():
        R = lsc._track(S, cols, g)
        out.append(R)
    return pd.concat(out, ignore_index=True), S


def confound_links(S: pd.DataFrame) -> pd.DataFrame:
    """Per animal (post sessions, within set): r of ACAd d with each confound, and confound means pre vs post."""
    rows = []
    for (st, a), g in S.groupby(["set", "animal"]):
        po, pr = g[g.epoch != "pre"], g[g.epoch == "pre"]
        for c in ["valley_off_470", "mirror_off_470", "f0asym_470[all]", "f0asym_415[all]", "d_h415[all]", "d_r470[all]",
                  "chk2_MOs_rest", "chk2_PL", "contrast_470"]:
            ok = po[[c, "d_neu[all]"]].dropna()
            rows.append({"set": st, "animal": a, "confound": c,
                         "r_with_ACAd_d_post": float(np.corrcoef(ok[c], ok["d_neu[all]"])[0, 1]) if len(ok) > 3 else np.nan,
                         "pre_mean": float(pr[c].mean()), "post_mean": float(po[c].mean()), "post_sd": float(po[c].std())})
    return pd.DataFrame(rows)


def pix_r(pix: dict) -> dict:
    """Per (animal, ch): r map of mirrored d with iti_ang over post sessions; plus the mean map and n."""
    R = pd.read_csv(il.OUT_W / "recmark_sessions.csv", usecols=["label", "day", "iti_ang"]).drop_duplicates("label")
    R = R.set_index("label")
    out = {}
    for (a, ch), D in pix.items():
        xs = [x for x, (ep, _d) in D.items() if ep != "pre" and x in R.index and R.loc[x, "day"] > 0
              and np.isfinite(R.loc[x, "iti_ang"])]
        Y = np.array([R.loc[x, "iti_ang"] for x in xs])
        X = np.stack([D[x][1] for x in xs]).astype(float)
        Xc, Yc = X - X.mean(0), Y - Y.mean()
        r = (Xc * Yc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (Yc ** 2).sum())
        out[f"r|{a}|{ch}"] = r.reshape(len(ROWS), len(RCOLS)).astype(np.float32)
        out[f"n|{a}|{ch}"] = np.array(len(xs))
        pre = [D[x][1] for x, (ep, _d) in D.items() if ep == "pre"]
        out[f"post_mean|{a}|{ch}"] = X.mean(0).reshape(len(ROWS), len(RCOLS)).astype(np.float32)
        if pre:
            out[f"pre_mean|{a}|{ch}"] = np.mean(pre, 0).reshape(len(ROWS), len(RCOLS)).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------------ main / plot
def trial_set_sentence() -> str:
    """The lick set of this driver's figures (Priya 11-09; `wfield_local.trial_set_text`): the licks `iti_lick_order.predictions` selects (CONTRIB_HOOK), = the KEY set."""
    import lick_covariate_reference as cv
    return cv.key_lick_set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())          # every figure states its lick set (Priya 11-09)
    pd.set_option("display.width", 250)
    if a.plot_only:
        plot()
        return 0
    E = extract(min(a.jobs, 8), a.cache)
    print(f"extracted {len(E)} sessions; ACAd col ranges {sorted(set(e['acad_cols'] for e in E.values()))}", flush=True)
    S, pix, Vd = compute(E)
    Vd.to_csv(VAL_CSV, index=False)
    print("== validation (rebuilt ACAd d vs check 2):",
          Vd.groupby("set").apply(lambda g: pd.Series({
              "r_vs_terms": np.corrcoef(g["d_neu[all]"], g.ref)[0, 1],
              "maxabs_vs_terms": np.nanmax(np.abs(g["d_neu[all]"] - g.ref)),
              "maxabs_vs_hookACAd": np.nanmax(np.abs(g["d_neu[all]"] - g.chk2_ACAd)),
              "weights_check": g.weights_vs_session_dh_maxabs.max(), "trace_maxrel": g.trace_maxrel.max()})).round(5)
          .to_string(), sep="\n")
    TR, S2 = tests(S)
    S2.to_csv(SESS_CSV, index=False)
    TR.to_csv(TRACK_CSV, index=False)
    CL = confound_links(S2)
    CL.to_csv(OUT / "acad_control_confounds.csv", index=False)
    PX = pix_r(pix)
    E0 = next(iter(E.values()))["pix"]
    f0m = {a: np.nanmean([E[x]["pix"]["f0img"] for x in E if x.startswith(a)], 0).astype(np.float32) for a in AN}
    np.savez_compressed(PIX_NPZ, rows=ROWS, rcols=RCOLS, area_img=E0["area_img"],
                        **PX, **{f"f0img|{a}": v for a, v in f0m.items()})
    extras(S2)
    _print(TR, CL)
    plot()
    return 0


def extras(S: pd.DataFrame) -> None:
    """Registration size, relative midline correction, log(day) partials (module docstring EXTRAS); profile."""
    import lick_sspn_checks as lsc
    sh = [f"d_neu[shift{s:+d}]" for s in SHIFTS]
    S = S.copy()
    mo = S[S.set == "iti"].groupby("animal").valley_off_470.mean()
    rel = (S.valley_off_470 - S.animal.map(mo)).clip(SHIFTS[0], SHIFTS[-1])
    D = S[sh].to_numpy(float)
    S["d_neu[relcorr+]"] = [np.interp(r_, SHIFTS, d_) for r_, d_ in zip(rel, D)]
    S["d_neu[relcorr-]"] = [np.interp(-r_, SHIFTS, d_) for r_, d_ in zip(rel, D)]
    S["reg_pred"] = (S["d_neu[shift+1]"] - S["d_neu[shift-1]"]) / 2 * rel
    T = lsc._track(S, ["d_neu[relcorr+]", "d_neu[relcorr-]", "reg_pred"], "3b registration size")
    R = lsc._recovery()
    g = S[S.epoch != "pre"].merge(R, on="label", how="left")
    g = g[g.day > 0].dropna(subset=["iti_ang"])
    rows = []
    for (st, a), h in g.groupby(["set", "animal"]):
        X = np.column_stack([np.ones(len(h)), np.log(h.day)])
        ry = h.iti_ang - X @ np.linalg.lstsq(X, h.iti_ang, rcond=None)[0]
        rec = {"set": st, "animal": a, "n": len(h), "sd_d": h["d_neu[all]"].std(), "sd_reg_pred": h.reg_pred.std(),
               "median_d_per_px": ((h["d_neu[shift+1]"] - h["d_neu[shift-1]"]) / 2).median(),
               "sd_offset_px": h.valley_off_470.std(), "r_offset_logday": np.corrcoef(h.valley_off_470, np.log(h.day))[0, 1]}
        for c in ["d_neu[all]", "chk2_SSp-n", "valley_off_470", "f0asym_470[all]", "d_h415[all]"]:
            rx = h[c] - X @ np.linalg.lstsq(X, h[c], rcond=None)[0]
            rec[f"partial_r_logday|{c}"] = np.corrcoef(rx, ry)[0, 1]
        rows.append(rec)
    G = pd.DataFrame(rows)
    T.to_csv(OUT / "acad_control_registration_tests.csv", index=False)
    G.to_csv(OUT / "acad_control_registration.csv", index=False)
    print("== registration size / log(day) partials (per animal, post sessions)")
    print(G.round(4).to_string(index=False))
    print(T.round(3).to_string(index=False))
    Z = np.load(PIX_NPZ)
    sel = (ROWS >= 118) & (ROWS <= 302)
    P = pd.DataFrame([{"ch": ch, "animal": a, "dist_mm": (c - MID) * MM, "r": float(v)} for ch in ("neu", "h415")
                      for a in AN for c, v in zip(RCOLS, np.nanmean(Z[f"r|{a}|{ch}"][sel], 0))])
    P.to_csv(OUT / "acad_control_profile.csv", index=False)


def _print(TR: pd.DataFrame, CL: pd.DataFrame) -> None:
    T = TR.copy()
    T["mean_r"] = T[[f"r_{a}" for a in AN]].mean(1)
    print("== recovery tests (per-animal r first; p = within-animal MixedLM)")
    print(T[["check", "set", "y", "measure", *[f"r_{a}" for a in AN], "mean_r", "p_within"]].round(3).to_string(index=False))
    print("== confounds per animal (ITI)")
    print(CL[CL.set == "iti"].round(3).to_string(index=False))


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    TR = pd.read_csv(TRACK_CSV)
    S = pd.read_csv(SESS_CSV)
    Z = np.load(PIX_NPZ, allow_pickle=True)
    TR["mean_r"] = TR[[f"r_{a}" for a in AN]].mean(1)
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 6, height_ratios=[1.05, 1.0, 1.1, 0.8], hspace=0.6, wspace=0.45,
                          top=0.95, bottom=0.07)
    t = TR[(TR.set == "iti") & (TR.y == "iti_ang")]
    pan = [("1 midline distance", [f"d_neu[{v}]" for v in VARIANTS], "1  ACAd d by distance from the Allen midline"),
           ("2 vessel / hemodynamic", ["d_h415[all]", "d_h415[d>=0.2]", "d_r470[all]", "f0asym_470[all]",
                                         "f0asym_415[all]", "resid_415", "resid_f0asym"], "2  vessel / hemodynamic"),
           ("3 registration", ["valley_off_470", "mirror_off_470", "contrast_470", "resid_offset", "resid_offset_mirror",
                               "d_neu[corrected]"], "3  registration (midline offset)"),
           ("4 neighbors", ["resid_MOs_rest", "resid_PL", "resid_MOs_rest+PL", "MOs_rest_resid_ACAd", "SSp-n_resid_ACAd",
                             "resid_all_confounds"], "4  neighbors partialled (within animal)")]
    ref = t[(t.check == "0 reference")]
    for i, (chk, ms, title) in enumerate(pan):
        ax = fig.add_subplot(gs[i // 2, (i % 2) * 3:(i % 2) * 3 + 3])
        tt = pd.concat([ref[ref.measure == "chk2_ACAd"], t[(t.check == chk) & t.measure.isin(ms)]])
        tt = tt.set_index("measure").loc[[m for m in ["chk2_ACAd", *ms] if m in set(tt.measure)]].reset_index()
        x = np.arange(len(tt))
        ax.bar(x, tt.mean_r, color=["#C44E52" if m == "chk2_ACAd" else "#4C72B0" for m in tt.measure])
        mk = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
        for xi, (_, r) in enumerate(tt.iterrows()):
            for a in AN:
                ax.plot(xi + (AN.index(a) - 1.5) * 0.08, r[f"r_{a}"], mk[a], color="k", ms=4, mfc="none")
            if np.isfinite(r.p_within):
                ax.text(xi, 1.02, f"p={r.p_within:.2g}", ha="center", fontsize=7, rotation=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-1, 1.12)
        ax.set_xticks(x, [m.replace("chk2_ACAd", "ACAd (check 2)") for m in tt.measure], rotation=35, ha="right",
                      fontsize=7.5)
        ax.set_ylabel("within-animal r with ITI lick angle\n(post sessions)", fontsize=8)
        ax.set_title(f"Fig {i + 1}. {title}\nbar = mean of 4 animals, markers = PS92 o PS93 s PS94 ^ PS95 D", fontsize=9)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    # pixel maps
    rows, rcols = Z["rows"], Z["rcols"]
    area = Z["area_img"]
    ext = (rcols[0] - 0.5 - MID, rcols[-1] + 1.5 - MID, rows[-1] + 1, rows[0] - 1)
    ext_mm = tuple(np.array(ext[:2]) * MM) + tuple((270 - np.array(ext[2:])) * MM)
    for j, ch in enumerate(("neu", "h415")):
        rm = np.nanmean([Z[f"r|{a}|{ch}"] for a in AN], 0)
        ax = fig.add_subplot(gs[2, j])
        im = ax.imshow(rm, cmap="RdBu_r", vmin=-0.7, vmax=0.7, extent=ext_mm, aspect="equal", interpolation="nearest")
        _bounds(ax, area, rows, rcols)
        ax.set_title(f"Fig 5{'ab'[j]}. mean-of-animals r map\n{'adopted (hemo-corrected)' if ch == 'neu' else 'raw 415 nm'}",
                     fontsize=9)
        ax.set_xlabel("mm from Allen midline (mirrored: right - left)", fontsize=7.5)
        ax.set_ylabel("mm anterior of bregma", fontsize=7.5)
        plt.colorbar(im, ax=ax, fraction=0.05)
    for k, a in enumerate(AN):
        ax = fig.add_subplot(gs[2, 2 + k])
        ax.imshow(Z[f"r|{a}|neu"], cmap="RdBu_r", vmin=-0.9, vmax=0.9, extent=ext_mm, aspect="equal",
                  interpolation="nearest")
        _bounds(ax, area, rows, rcols)
        ax.set_title(f"Fig 5c. {a} r map (adopted)\nn = {int(Z[f'n|{a}|neu'])} post sessions", fontsize=9)
        ax.tick_params(labelsize=7)
    P = pd.read_csv(OUT / "acad_control_profile.csv")
    G = pd.read_csv(OUT / "acad_control_registration.csv")
    for j, ch in enumerate(("neu", "h415")):
        ax = fig.add_subplot(gs[3, j * 3:j * 3 + 2])
        for a in AN:
            q = P[(P.ch == ch) & (P.animal == a)]
            ax.plot(q.dist_mm, q.r, "-", lw=1, label=a)
        m = P[P.ch == ch].groupby("dist_mm").r.mean()
        ax.plot(m.index, m.values, "k-", lw=2.5, label="mean")
        ax.axvspan(0, 0.33, color="0.9", zorder=0)
        ax.axvspan(0, 0.1, color="0.8", zorder=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("mm from Allen midline (gray = ACAd; dark = < 0.1 mm strip over the sinus)", fontsize=8)
        ax.set_ylabel("pixel r with ITI lick angle\n(mean over ACAd rows)", fontsize=8)
        ax.set_title(f"Fig 6{'ab'[j]}. medio-lateral profile of the pixel r, {'adopted' if ch == 'neu' else 'raw 415 nm'}",
                     fontsize=9)
        ax.legend(fontsize=7.5, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    gi = G[G.set == "iti"].set_index("animal")
    txt = "Fig 6c. registration size (ITI, post)\n" + "\n".join(
        f"{a}: SD d {r.sd_d:.3f}; d per px {r.median_d_per_px:+.4f};\n   SD offset {r.sd_offset_px:.2f} px -> "
        f"SD predicted {r.sd_reg_pred:.4f}\n   partial r|logday: ACAd {r['partial_r_logday|d_neu[all]']:+.2f}, "
        f"SSp-n {r['partial_r_logday|chk2_SSp-n']:+.2f}" for a, r in gi.iterrows())
    fig.text(0.80, 0.06, txt, fontsize=7.5, family="monospace", va="bottom")
    off = S.drop_duplicates("label").groupby(["animal", S.epoch.eq("pre").map({True: "pre", False: "post"})])[
        "valley_off_470"].agg(["mean", "std"]).round(2)
    fig.text(0.01, 0.0, "Sinus-valley offset from the Allen midline (px, 1 px = 0.0194 mm), mean / SD:  " + "; ".join(
        f"{a} {e} {r['mean']:+.2f}/{r['std']:.2f}" for (a, e), r in off.iterrows()), fontsize=8)
    fig.suptitle("Is the ACAd left/right lick-response link to recovery real cortex? Weight-free same-place d = dh_right - "
                 "dh_left (check 2), ITI licks -> ITI lick-angle axis. PROVISIONAL", fontsize=12)
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG}")


def _bounds(ax, area: np.ndarray, rows: np.ndarray, rcols: np.ndarray) -> None:
    """Allen area boundaries over the right-position grid (area_img crop starts at row rows[0], col 260)."""
    sub = area[::2, (rcols - 260)][: len(rows)]
    edge = np.zeros(sub.shape, bool)
    edge[:-1] |= sub[:-1] != sub[1:]
    edge[:, :-1] |= sub[:, :-1] != sub[:, 1:]
    yy, xx = np.nonzero(edge)
    ax.plot((rcols[xx] + 1 - MID) * MM, (270 - rows[yy] - 1) * MM, ",", color="k", alpha=0.6)


if __name__ == "__main__":
    raise SystemExit(main())

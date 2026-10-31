"""GCaMP6s vs jGCaMP8s with the SENSOR KINETICS equalised, and how far the responses SPREAD over cortex (2026-10-31).

Priya 10-31: "do the 6s 8s comparison including adjusting for sensor kinetics and looking at regional spread".
Supersedes the provisional §0j table (one session per mouse, no kinetics adjustment).

Sessions: cohort2 re-run 10-29 + 10-30 (pre-SVD swap repair + clip mask; hemo_meegkit_hpfit), PS118 = the only 6s mouse
with signal, PS121 / PS122 = RiboL1-8s, PS120 = GCaMP-negative-like NULL (blood flow / motion only); cohort1 pre-stroke
6s references on the same rig and binning (PS92_0811, PS93_0814, PS94_0811, PS95_0811; hemo_meegkit_hpfit, swapped
pairs interpolated). PS122 10-29 is cut at 110 min (epileptiform discharges, QC doc §0o). Native pixels, same 480-px
binning for every session, dark-relative brain mask minus clipped pixels.

KINETICS ADJUSTMENT. Fluorescence = activity * indicator kernel, kernel k(t) = exp(-t/tau_d) - exp(-t/tau_r), unit
area. An 8s trace is converted to what a 6s would have reported for the same activity by the exact discrete ratio
filter K6(z)/K8(z) (causal IIR, unity DC gain, stable both ways):
    H(z) = c (1 - pd8 z^-1)(1 - pr8 z^-1) / ((1 - pd6 z^-1)(1 - pr6 z^-1)),  p = exp(-dt / tau)
applied to SVTcorr (linear, so to the components). PRIMARY comparison = 8s -> 6s (smoothing; no noise amplification).
Check = 6s -> 8s (sharpening; amplifies noise). Kernel time constants are ASSUMPTIONS from in-vivo single-AP literature
(GCaMP6s Chen et al. 2013; jGCaMP8s Zhang et al. 2023), not fitted here: default 6s rise tau 0.10 s / decay t1/2
1.0 s, 8s rise tau 0.02 s / decay t1/2 0.30 s, and a 3 x 3 grid of decay half-times is re-scored (sensitivity).
Unit-area kernels keep the integral of the response, so a converted amplitude answers "same activity, 6s kinetics".

METRICS (per session; raw and kinetics-matched)
  waveform  first-lick-aligned cortex mean and hotspot (top 2% of the ODD-trial lick map); time to peak, decay to 50%
  d'        single-trial hotspot response / SD across trials; hotspot chosen on odd trials, scored on even
  maps      lick = [first lick, +0.5 s], cue = [cue, +1 s], both minus pre-cue [-1, 0] s; split-half r reported
  SPREAD    globalness  = brain-mean response / hotspot response (1 = uniform)
            halfmax     = fraction of brain >= half the hotspot response
            map_len     = distance (px) at which the spatial autocorrelation of the MEAN-REMOVED map falls to 0.5,
                          from odd x even half-maps (noise-unbiased)
            spout_share = reproducible spout-specific share of the lick maps, sum_q (o_q - o_mu)(e_q - e_mu) /
                          sum_q o_q e_q over the six spouts (noise-unbiased form of spout_maps' share)
            spont_len   = ongoing activity (0.3-5 Hz, first 30 min) pixel correlation vs distance, half-distance (px),
                          with the global (cortex-mean) signal kept and REGRESSED OUT
    conda activate locanmf
    python scripts/cohort2_qc/indicator_kinetics_spread.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, lfilter

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402

LABCAMS = Path("M:/MICROSCOPE/Priya/Widefield/labcams")
DAQ = Path("M:/MICROSCOPE/Priya/Widefield/DAQ_recorder_output")
OUT = LABCAMS / "cohort2" / "_tmp_qc_figures_20261031" / "indicator_kinetics"
FS = 31.23
DT = 1 / FS
VAR = "hemo_meegkit_hpfit"
C2_DATES = ("20261029", "20261030")
C1_REFS = ("PS92_0811", "PS93_0814", "PS94_0811", "PS95_0811")
IND = {"PS118": "6s", "PS120": "null", "PS121": "8s", "PS122": "8s", "PS92": "6s", "PS93": "6s", "PS94": "6s",
       "PS95": "6s"}
EXCLUDE_AFTER_MIN = {("PS122", "20261029"): 110.0}
ORDER = [1, 0, 2, 4, 3, 5]                     # close L / C / R, far L / C / R (strobe codes)
RESP_S = 3.5
K6 = dict(rise=0.10, half=1.0)
K8 = dict(rise=0.02, half=0.30)
GRID6 = (0.6, 1.0, 1.5)                        # 6s decay half-times (s) for the sensitivity grid
GRID8 = (0.2, 0.3, 0.45)
LN2 = np.log(2)


def ratio_filter(src, dst):
    """(b, a) for lfilter converting a trace with kernel `src` into one with kernel `dst` (unit area, causal)."""
    p = lambda k: (np.exp(-DT * LN2 / k["half"]), np.exp(-DT / k["rise"]))  # noqa: E731
    ds, rs = p(src)
    dd, rd = p(dst)
    c = (1 - dd) * (1 - rd) / ((1 - ds) * (1 - rs))
    return c * np.convolve([1, -ds], [1, -rs]), np.convolve([1, -dd], [1, -rd])


def sessions():
    out = []
    for d in C2_DATES:
        for sd in sorted((LABCAMS / d).glob("PS*")):
            animal = sd.name.split("_")[0]
            h5 = sorted((DAQ / d).glob(f"{animal}_{d}_*.h5"))
            if not h5:
                print("no DAQ .h5:", sd.name, flush=True)
                continue
            out.append(dict(label=f"{animal}_{d[4:]}", animal=animal, date=d, cohort="cohort2", func=1,
                            mc=str(sd / "motion_corrected"), h5=str(h5[0]), regime="B", fmdir=None))
    from wfield_local import config
    from wfield_local import hemo_variants as hv
    c1 = {s["label"]: s for s in config.load_sessions(machine="analysis_desktop")}
    for lab in C1_REFS:
        s = dict(c1[lab])
        s.update(animal=lab.split("_")[0], date="2026" + lab.split("_")[1], cohort="cohort1",
                 func=int(hv.functional_channel(c1[lab])))
        out.append(s)
    return out


def events(s, n):
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, lf, _ = _frames(s, cue, lk)
    cf = np.asarray(cf)
    lf = np.sort(np.asarray(lf)[np.asarray(lf) >= 0])
    st = np.asarray(cue["strobe_samples"])
    pos = np.asarray(cue["strobe_codes"])[np.clip(np.searchsorted(st, cue["cue_samples"]) - 1, 0, None)]
    k = np.searchsorted(lf, cf)
    fl = np.where(k < len(lf), lf[np.minimum(k, len(lf) - 1)], -1)
    fl = np.where((fl >= cf) & (fl - cf <= int(RESP_S * FS)), fl, -1)
    ok = (cf > 40) & (cf < n - 80) & (fl > 0) & (fl < n - 80)
    cut = EXCLUDE_AFTER_MIN.get((s["animal"], s["date"]))
    if cut is not None:
        ok &= cf < int(cut * 60 * FS)
    return cf[ok], fl[ok], pos[ok]


def halfdist(r, edges):
    """Distance (px, linear interpolation between bin centres) where the profile first falls below 0.5."""
    c = 0.5 * (edges[:-1] + edges[1:])
    r = np.asarray(r, float)
    k = np.flatnonzero(r < 0.5)
    if not len(k):
        return float("nan")
    k = k[0]
    if k == 0:
        return float(c[0])
    return float(c[k - 1] + (r[k - 1] - 0.5) / (r[k - 1] - r[k]) * (c[k] - c[k - 1]))


def binned(P, geom):
    """Mean of the pair matrix P in each distance bin (bin index precomputed in geom)."""
    _, edges, _, bi, cnt = geom
    return np.bincount(bi, weights=P.ravel(), minlength=len(edges) + 1)[1:len(edges)] / cnt


def pair_profile(a, b, geom):
    """Autocorrelation vs distance of a field from two independent halves a, b (mean-removed): noise-unbiased."""
    return binned(np.outer(a, b), geom) / np.mean(a * b)


def kinetics(w, t):
    pk = int(np.argmax(np.where((t >= -0.2) & (t < 1.5), w, -np.inf)))
    after = np.flatnonzero((np.arange(len(w)) > pk) & (w <= 0.5 * w[pk]))
    return dict(t_peak_s=float(t[pk]), decay50_s=float(t[after[0]] - t[pk]) if len(after) else float("nan"),
                peak_pct=float(100 * w[pk]))


def score(C, ev, Ub, um, geom, full=True):
    """Every event-locked metric for one version (raw / converted) of the corrected components C (K x n)."""
    cf, fl, pos = ev
    lags = np.arange(-31, 63)
    t = lags / FS
    base = C[:, cf[:, None] + np.arange(-31, 0)].mean(2)                         # K x trials, pre-cue
    E = C[:, fl[:, None] + lags] - base[:, :, None]                                # K x trials x lags
    lick = C[:, fl[:, None] + np.arange(0, 16)].mean(2) - base                     # [lick, +0.5 s]
    cue = C[:, cf[:, None] + np.arange(0, 31)].mean(2) - base                      # [cue, +1 s]
    odd, even = np.arange(len(fl)) % 2 == 1, np.arange(len(fl)) % 2 == 0
    mo = Ub @ lick[:, odd].mean(1)
    hot = mo >= np.percentile(mo, 98)
    uh = Ub[hot].mean(0)
    out = {}
    for nm, w in (("cortex", um), ("hotspot", uh)):
        out.update({f"{nm}_{k}": v for k, v in kinetics(w @ E.mean(1), t).items()})
    out["wave_cortex"] = (um @ E.mean(1)).tolist()
    out["wave_hotspot"] = (uh @ E.mean(1)).tolist()
    rt = uh @ lick[:, even]
    out["dprime"] = float(rt.mean() / rt.std())
    D, edges, idx = geom[:3]
    for kind, R in (("lick", lick), ("cue", cue)):
        m = Ub @ R.mean(1)
        a, b = Ub @ R[:, odd].mean(1), Ub @ R[:, even].mean(1)
        out[f"{kind}_splithalf_r"] = float(np.corrcoef(a, b)[0, 1])
        hv_ = m[hot].mean()
        out[f"{kind}_hot_pct"] = float(100 * hv_)
        out[f"{kind}_globalness"] = float(m.mean() / hv_)
        out[f"{kind}_halfmax"] = float(np.mean(m >= 0.5 * hv_))
        prof = pair_profile(a[idx] - a[idx].mean(), b[idx] - b[idx].mean(), geom)
        out[f"{kind}_map_len_px"] = halfdist(prof, edges)
        if full:
            out[f"{kind}_map_prof"] = prof.tolist()
        if full:
            out[f"{kind}_map"] = m
    num = den = 0.0
    for q in ORDER:
        so, se = odd & (pos == q), even & (pos == q)
        if so.sum() < 5 or se.sum() < 5:
            return out
    Mo = np.array([Ub @ lick[:, odd & (pos == q)].mean(1) for q in ORDER])
    Me = np.array([Ub @ lick[:, even & (pos == q)].mean(1) for q in ORDER])
    num = np.sum((Mo - Mo.mean(0)) * (Me - Me.mean(0)))
    den = np.sum(Mo * Me)
    out["spout_share"] = float(num / den)
    out["spout_specific_pct"] = float(100 * np.sqrt(max(num, 0) / Mo.size))      # reproducible RMS, dF/F %
    out["spout_specific_rel_hot"] = out["spout_specific_pct"] / out["lick_hot_pct"]
    return out


def spont(C, Ub, um, geom):
    D, edges, idx = geom[:3]
    b, a = butter(2, [0.3 / (FS / 2), 5 / (FS / 2)], btype="band")
    X = filtfilt(b, a, C[:, :min(C.shape[1], int(30 * 60 * FS))], axis=1)
    G = X @ X.T / X.shape[1]
    Us = Ub[idx]
    cov = Us @ G @ Us.T
    cg, vg = Us @ G @ um, um @ G @ um
    out = {}
    for nm, cv in (("kept", cov), ("removed", cov - np.outer(cg, cg) / vg)):
        sd = np.sqrt(np.clip(np.diag(cv), 1e-30, None))
        R = cv / np.outer(sd, sd)
        prof = binned(R, geom)
        out[f"spont_len_{nm}_px"] = halfdist(prof, edges)
        c = 0.5 * (edges[:-1] + edges[1:])
        for d in (25, 50, 100):
            out[f"spont_r{d}_{nm}"] = float(np.interp(d, c, prof))
        out[f"spont_prof_{nm}"] = prof.tolist()
    return out


def one(s):
    from wfield_local import hemo_variants as hv
    res = Path(s["mc"]) / "wfield_local_results"
    func = s["func"]
    fa = np.load(res / "frames_average.npy")
    br = brain_mask(fa, func, res=res)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)].astype(np.float64)
    del U
    um = Ub.mean(0)
    C = np.load(res / VAR / "SVTcorr.npy").astype(np.float64)
    n = C.shape[1]
    if s["cohort"] == "cohort1" and (res / "swapped_pairs.npz").exists():
        C = hv._interp_pairs(C, np.load(res / "swapped_pairs.npz")["bad"][:n])
    ev = events(s, n)
    yy, xx = np.nonzero(br)
    idx = np.random.default_rng(0).choice(len(Ub), min(2500, len(Ub)), replace=False)
    D = np.hypot(yy[idx][:, None] - yy[idx][None, :], xx[idx][:, None] - xx[idx][None, :])
    edges = np.arange(0, 301, 2)
    bi = np.digitize(D.ravel(), edges)                      # 1..len(edges)-1 inside, len(edges) beyond 300 px
    cnt = np.bincount(bi, minlength=len(edges) + 1)[1:len(edges)].astype(float)
    geom = (D, edges, idx, bi, cnt)
    ind = IND[s["animal"]]
    r = dict(label=s["label"], animal=s["animal"], date=s["date"], cohort=s["cohort"], indicator=ind,
             n_trials=int(len(ev[1])), br=br)
    r["raw"] = score(C, ev, Ub, um, geom)
    r["raw"].update(spont(C, Ub, um, geom))
    if ind in ("8s", "null"):                               # the null is put through the same conversion
        b, a = ratio_filter(K8, K6)
        Cm = lfilter(b, a, C, axis=1)
        r["to6s"] = score(Cm, ev, Ub, um, geom)
        r["to6s"].update(spont(Cm, Ub, um, geom))
        r["grid"] = {}
        for h6 in GRID6:
            for h8 in GRID8:
                b, a = ratio_filter(dict(K8, half=h8), dict(K6, half=h6))
                g = score(lfilter(b, a, C, axis=1), ev, Ub, um, geom, full=False)
                r["grid"][f"{h6}_{h8}"] = {k: v for k, v in g.items() if not k.startswith("wave")}
    if ind in ("6s", "null"):
        b, a = ratio_filter(K6, K8)
        Cm = lfilter(b, a, C, axis=1)
        r["to8s"] = score(Cm, ev, Ub, um, geom, full=False)
    return r


SCALARS = ["cortex_t_peak_s", "cortex_decay50_s", "hotspot_t_peak_s", "hotspot_decay50_s", "lick_hot_pct", "dprime",
           "lick_splithalf_r", "lick_globalness", "lick_halfmax", "lick_map_len_px", "cue_globalness", "cue_halfmax",
           "cue_map_len_px", "spout_share", "spout_specific_pct", "spout_specific_rel_hot", "spont_len_kept_px",
           "spont_len_removed_px", "spont_r25_removed", "spont_r50_removed", "spont_r100_removed", "spont_r100_kept"]


def main():
    from wfield_local.parallel import fan_out
    from wfield_local.writeguard import assert_writable
    S = sessions()
    print("sessions:", [s["label"] for s in S], flush=True)
    res, fail = fan_out(S, one, jobs=4, label="session")
    if fail:
        print("FAILED", fail)
    R = sorted((r for _, r in res), key=lambda r: (r["cohort"] != "cohort2", r["indicator"], r["label"]))
    assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    for r in R:
        for ver in ("raw", "to6s", "to8s"):
            if ver in r:
                table.append(dict(label=r["label"], indicator=r["indicator"], version=ver, n_trials=r["n_trials"],
                                  **{k: r[ver].get(k) for k in SCALARS}))
    grid = {r["label"]: r["grid"] for r in R if "grid" in r}
    (OUT / "indicator_kinetics_spread.json").write_text(json.dumps(dict(
        kernels=dict(K6=K6, K8=K8, grid6_half=GRID6, grid8_half=GRID8), table=table, grid=grid,
        waves={r["label"]: {v: dict(cortex=r[v]["wave_cortex"], hotspot=r[v]["wave_hotspot"])
                            for v in ("raw", "to6s") if v in r} for r in R},
        spont_prof={r["label"]: {v: {k: r[v][k] for k in ("spont_prof_kept", "spont_prof_removed")}
                                 for v in ("raw", "to6s") if v in r} for r in R},
        map_prof={r["label"]: {v: r[v]["lick_map_prof"] for v in ("raw", "to6s") if v in r} for r in R}),
        indent=1))
    for row in table:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    figure(R)


def figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"6s": "tab:green", "8s": "tab:orange", "null": "0.5"}
    t = np.arange(-31, 63) / FS
    fig = plt.figure(figsize=(22, 17))
    gs = fig.add_gridspec(4, 8, height_ratios=[1, 1, 1, 1.1])
    # row 1: waveforms, peak-normalised
    for j, nm in enumerate(("cortex", "hotspot")):
        ax = fig.add_subplot(gs[0, 4 * j:4 * j + 4])
        for r in R:
            ls = "-" if r["cohort"] == "cohort2" else ":"
            w = np.array(r["raw"][f"wave_{nm}"])
            if r["indicator"] == "null":
                ax.plot(t, w / max(np.abs(w).max(), 1e-9) * 0.3, color="0.6", lw=0.8, ls=ls)
                continue
            ax.plot(t, w / w.max(), color=col[r["indicator"]], lw=0.9 if r["indicator"] == "6s" else 0.7,
                    ls=ls if r["indicator"] == "6s" else "--")
            if "to6s" in r:
                w2 = np.array(r["to6s"][f"wave_{nm}"])
                ax.plot(t, w2 / w2.max(), color="tab:red", lw=1.2)
        ax.axvline(0, color="k", lw=0.4)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"first-lick {nm} waveform, peak-normalised: green = 6s (dotted cohort1), orange dashed = 8s raw, "
                     "red = 8s -> 6s kinetics, grey = PS120 null (x0.3 of its own max)", fontsize=8)
        ax.set_xlabel("s from first lick")
    # rows 2-3: lick maps normalised to hotspot, raw and to6s for 8s
    c2 = [r for r in R if r["cohort"] == "cohort2"] + [r for r in R if r["cohort"] == "cohort1"][:0]
    panels = []
    for r in R:
        panels.append((r, "raw"))
        if "to6s" in r and r["indicator"] == "8s":
            panels.append((r, "to6s"))
    panels = panels[:16]
    for i, (r, v) in enumerate(panels):
        ax = fig.add_subplot(gs[1 + i // 8, i % 8])
        H, W = r["br"].shape
        z = np.full(H * W, np.nan)
        z[r["br"].reshape(-1)] = r[v]["lick_map"] / r[v]["lick_hot_pct"] * 100
        ax.imshow(z.reshape(H, W), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{r['label']} {r['indicator']} {'' if v == 'raw' else '-> 6s'}\nlick map / hotspot "
                     f"({r[v]['lick_hot_pct']:.2f}%)", fontsize=7)
        ax.axis("off")
    del c2
    # row 4: spread metrics
    mets = [("lick_globalness", "lick: brain mean / hotspot"), ("lick_halfmax", "lick: frac brain > half hotspot"),
            ("lick_map_len_px", "lick map structure length (px)"), ("cue_globalness", "cue: brain mean / hotspot"),
            ("spout_specific_pct", "reproducible spout-specific RMS (%)"), ("dprime", "single-trial d' (hotspot)"),
            ("spont_len_kept_px", "ongoing corr half-dist, global kept"),
            ("spont_r50_removed", "ongoing corr at 50 px, global removed")]
    for j, (k, ttl) in enumerate(mets):
        ax = fig.add_subplot(gs[3, j])
        for x, ind in enumerate(("6s", "8s", "null")):
            for r in R:
                if r["indicator"] != ind:
                    continue
                mk = "o" if r["cohort"] == "cohort2" else "s"
                v = r["raw"].get(k)
                if v is not None:
                    ax.scatter(x - 0.12, v, color=col[ind], marker=mk, facecolors="none", s=28)
                if "to6s" in r and r["to6s"].get(k) is not None:
                    ax.scatter(x + 0.12, r["to6s"][k], color="tab:red", marker=mk, s=18)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["6s", "8s", "PS120"], fontsize=8)
        ax.set_title(ttl, fontsize=8)
    fig.suptitle("GCaMP6s vs jGCaMP8s, kinetics-adjusted (8s -> 6s kernel, red) and regional spread. open = raw; "
                 "circles = cohort2 10-29/10-30, squares = cohort1 pre-stroke 6s; PS120 = no-GCaMP null", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "indicator_kinetics_spread.png", dpi=75)
    print("wrote", OUT / "indicator_kinetics_spread.png")


if __name__ == "__main__":
    main()

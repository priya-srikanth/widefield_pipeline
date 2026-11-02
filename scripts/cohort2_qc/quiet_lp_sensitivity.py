"""Sensitivity check: do the quiet-period results survive excluding NON-CONTACT (video-only) licks? (cohort1, 2026-11-01)

WHY. Every quiet analysis (quiet_spikes_v2.busy_frames) defines QUIET from DAQ spout-CONTACT licks and running only: no
contact lick or running bout within 1.5 s. Acutely many licks never touch the spout (tongue out, no contact) and are
seen only in video, so they can sit inside 'quiet' frames and inflate the acute orofacial event rate (asked by the
encoding window and Priya's plan).

FRAMES (four variants, each a SUBSET of the previous one's frames, all engaged + the v2 rules):
  daq        the published v2 quiet frames (no DAQ contact lick / running bout within 1.5 s, no swapped pair within
             0.5 s); for the face-regressed count also a full face-regressor row (as quiet_spikes_v2_fr).
  lp         + no LP video lick within 1.5 s: cam4 LP licks of the v36 whole-session table (`licks_all.csv`, every
             context, `daq_on_s`) + cam1-only events (`cam1_licks/<s>_events_all.csv`, whole session, not cam4-matched,
             not water droplets, `daq_on_s`; the post-cue `<s>_events.csv` if events_all is absent) -- the
             `lp_licks` "combined" definition extended to the whole session. Both tables carry absolute DAQ seconds
             through the camera alignment templates (the same clock as `busy_frames`' DAQ lick samples / fs and the
             imaging frame times); the per-session median LP-onset-to-nearest-contact lag is stored as a check.
  lpjaw      + no jaw onset / re-onset within 1.5 s (`<v36>/imaging_encoding/qc/jaw_onsets.csv`: per trial, ms from
             the cue; cue on the DAQ clock from the session's windows.npz, `orofacial_cohort._cue_table`). Jaw onsets
             exist only for the cue response of each trial (not the ITI), so this variant adds little by construction.
  lpcov      lp, restricted to frames INSIDE the video trial windows (strobe - 0.5 s .. stop + 3 s, cut at the next
             trial): outside them the video was not tracked, so 'no LP lick' there is not evidence of no lick.
MEASURES per session (engaged quiet, August pre):
  (1) face-regressed orofacial event rate, k = 4: the quiet_spikes_v2_fr session events (`res_pos` / `res_neg`, frame
      indices) RECOUNTED inside the stricter frames; threshold = 4 x the animal's August-pre median of `sd_res` exactly
      as quiet_spikes_v2_fr.session_table (NOT re-estimated on the stricter frames); rate per stricter-quiet minute.
  (2) raw v2 event rate, k = 4 (quiet_spikes_v2 session `pos` / `neg`, threshold 4 x August-pre median `sd_q`).
  (3) quiet robust SD (1.4826 x MAD of the 2-s-median-detrended dF/F %) and log10 1-4 Hz power (Welch, 4-s segments,
      linear detrend, on quiet runs >= 4 s; quiet_fluct (1)) of the orofacial traces, on the stricter frames.
  (4) T20 synchronisation index log10(PSD 1-4 Hz / 8-15 Hz) (quiet_sync_index), orofacial and ipsi - contra.
  Recount vs re-detection: an event is kept when its PEAK frame is in the stricter quiet set; the detector's only use of
  the quiet mask is exactly that test, except that a removed larger peak no longer suppresses a smaller one within
  0.3 s (merge) -- negligible, and conservative (the stricter count can only be lower).
STATISTICS: per-animal animal x epoch table first; `mixed_models.lmm_table` epoch model vs August pre per measure x
variant (log10(rate + 0.2) for rates), FDR within measure-variant.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_lp_sensitivity.py [--jobs 4] [--summarise]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QS_FR_BASIS", "own")            # before the import: quiet_spikes_fr reads it at import (spawn too)
os.environ["QS_LICK_SOURCE"] = "daq"                   # the PUBLISHED DAQ-only frames are the baseline here (v2_quiet)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import DETREND_S, EP, EVENTS, columns, mad, session_traces  # noqa: E402
from quiet_spikes_fr import face_regressors, lagged  # noqa: E402
from quiet_spikes_v2 import BUF_S, V2, busy_frames, detrend  # noqa: E402
from quiet_spikes_v2_fr import FRV2  # noqa: E402

OUT = V2 / "lp_quiet"
K = 4.0
VARIANTS = ["daq", "lp", "lpjaw", "lpcov"]
VNAME = {"daq": "DAQ contact only (published)", "lp": "DAQ + LP video licks", "lpjaw": "DAQ + LP + jaw onsets",
         "lpcov": "DAQ + LP, video-covered frames only"}
PS_COLS = ["orofacial_left", "orofacial_right", "orofacial", "limb_left", "limb_right", "limb", "cortex"]
SEG_S, LO, HI = 4.0, (1.0, 4.0), (8.0, 15.0)


def _mark(T, ft, times, buf=BUF_S):
    m = np.zeros(T, bool)
    for x in np.asarray(times, float):
        if np.isfinite(x):
            m[np.searchsorted(ft, x - buf):np.searchsorted(ft, x + buf)] = True
    return m


def video_licks(label):
    """(LP onsets DAQ s, cam1 source tag, jaw onsets DAQ s or None, tracked windows (n x 2) DAQ s, coverage json,
    n cam4, n cam1) via lp_licks.py (the shared loader); None if the session has no video."""
    import json
    import lp_licks as LPQ
    lp, parts = LPQ.lick_times(label, return_parts=True)
    if not parts["cam4"]:
        return None
    cj = LPQ._paths(label)[0] / "licks_all_coverage.json"
    cov = json.loads(cj.read_text()) if cj.exists() else {}
    return lp, parts["cam1_src"], LPQ.jaw_onsets(label), LPQ.video_windows(label), cov, parts["cam4"], parts["cam1"]


def welch_runs(X, mask, fs):
    from scipy.signal import welch
    nseg = int(round(SEG_S * fs))
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    P, W, fr = None, 0, np.fft.rfftfreq(nseg, 1 / fs)
    for a, b in zip(e[::2], e[1::2]):
        if b - a < nseg:
            continue
        fr, p = welch(X[:, a:b], fs=fs, nperseg=nseg, noverlap=nseg // 2, detrend="linear", axis=-1)
        w = 1 + (b - a - nseg) // (nseg // 2)
        P = p * w if P is None else P + p * w
        W += w
    return fr, (P / W if W else np.full((X.shape[0], nseg // 2 + 1), np.nan)), W


def extract(label):
    f = OUT / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    vl = video_licks(label)
    if vl is None:
        raise FileNotFoundError(f"{label}: no LP lick tables")
    lp, src, jaw, win, cov, n4, n1 = vl
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    T = Xr.shape[1]
    busy, badw = busy_frames(label, ft, fs)
    base = ~busy & ~badw & np.isfinite(Xr).all(0) & engaged
    lpb = _mark(T, ft, lp)
    jawb = _mark(T, ft, jaw) if jaw is not None else np.zeros(T, bool)
    covered = np.zeros(T, bool)
    for t0, t1 in win:
        covered[np.searchsorted(ft, t0):np.searchsorted(ft, t1)] = True
    masks = {"daq": base, "lp": base & ~lpb, "lpjaw": base & ~lpb & ~jawb, "lpcov": base & ~lpb & covered}
    # face-regressor rows (the FR count's frames are v2 quiet & a full lagged regressor row)
    w = int(round(DETREND_S * fs)) | 1
    R, _ = face_regressors(label, ft, w)
    okX = np.isfinite(lagged(R, fs)).all(1)
    # DAQ <-> LP clock check: LP onset minus the nearest DAQ contact onset, for LP licks within 0.25 s of one
    z = np.load(EVENTS / label[:4] / f"2026{label[5:]}.npz", allow_pickle=True)
    dq = np.sort(z["lick_onsets"] / float(z["fs"]))
    lag = np.nan
    if len(dq) and len(lp):
        j = np.clip(np.searchsorted(dq, lp), 1, len(dq) - 1)
        nn = np.where(np.abs(dq[j] - lp) < np.abs(dq[j - 1] - lp), dq[j], dq[j - 1])
        dd = lp - nn
        lag = float(np.median(dd[np.abs(dd) < 0.25])) if (np.abs(dd) < 0.25).any() else np.nan
    cols = columns()
    pi = [cols.index(c) for c in PS_COLS]
    Xd = detrend(Xr, fs)
    out = dict(fs=fs, cols=np.array(cols), ps_cols=np.array(PS_COLS), lp_lag_s=lag, cam1_src=src, n_cam4=n4, n_cam1=n1,
               has_jaw=jaw is not None, covered_frac=float(cov.get("covered_frac", np.nan)),
               n_eng=int(engaged.sum()))
    for v, m in masks.items():
        out[f"mask_{v}"] = np.packbits(m)
        out[f"n_{v}"] = int(m.sum())
        out[f"nfr_{v}"] = int((m & okX).sum())
        out[f"sd_{v}"] = mad(Xd[:, m]) if m.sum() > 100 else np.full(len(cols), np.nan)
        fr, P, W = welch_runs(Xr[pi], m, fs)
        out[f"psd_{v}"], out[f"nseg_{v}"] = P, W
    out["freqs"] = fr
    out["okX"] = np.packbits(okX)
    out["T"] = T
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return label


# ------------------------------------------------------------------------------------------------------- summary
def session_table():
    import pandas as pd
    from wfield_local import epochs
    cols = columns()
    oi = {c: cols.index(c) for c in ("orofacial", "orofacial_left", "orofacial_right", "limb")}
    D = {}
    for f in sorted((OUT / "sessions").glob("*.npz")):
        ep = epochs.epoch_of(f.stem)
        if ep is None or (ep == "pre" and f.stem[-4:] < "0806"):
            continue
        D[f.stem] = (dict(np.load(f)), ep)
    v2 = {k: dict(np.load(V2 / "sessions" / f"{k}.npz")) for k in D}
    fr = {k: dict(np.load(FRV2 / "sessions" / f"{k}.npz")) for k in D}
    # thresholds exactly as the published tables: animal median over August-pre sessions (ALL pre sessions on file)
    pre_v2 = [f for f in (V2 / "sessions").glob("*.npz") if epochs.epoch_of(f.stem) == "pre" and f.stem[-4:] >= "0806"]
    pre_fr = [f for f in (FRV2 / "sessions").glob("*.npz") if epochs.epoch_of(f.stem) == "pre" and f.stem[-4:] >= "0806"]
    sd_v2 = {an: np.nanmedian([np.load(f)["sd_q"] for f in pre_v2 if f.stem.startswith(an)], 0) for an in
             sorted({k[:4] for k in D})}
    sd_fr = {an: np.nanmedian([np.load(f)["sd_res"] for f in pre_fr if f.stem.startswith(an)], 0) for an in
             sorted({k[:4] for k in D})}
    rows = []
    for k, (d, ep) in D.items():
        an, fs, T = k[:4], float(d["fs"]), int(d["T"])
        okX = np.unpackbits(d["okX"])[:T].astype(bool)
        fq = np.asarray(d["freqs"])
        lo, hi = (fq >= LO[0]) & (fq < LO[1]), (fq >= HI[0]) & (fq < HI[1])
        ps = list(d["ps_cols"])
        r = dict(label=k, animal=an, epoch=ep, day=epochs.days_since_stroke(k), lp_lag_ms=1000 * float(d["lp_lag_s"]),
                 cam1_src=str(d["cam1_src"]), has_jaw=bool(d["has_jaw"]), covered_frac=float(d["covered_frac"]),
                 n_cam4=int(d["n_cam4"]), n_cam1=int(d["n_cam1"]))
        for v in VARIANTS:
            m = np.unpackbits(d[f"mask_{v}"])[:T].astype(bool)
            mins, mins_fr = m.sum() / fs / 60, (m & okX).sum() / fs / 60
            r[f"quiet_min|{v}"], r[f"quiet_min_fr|{v}"] = mins, mins_fr
            r[f"kept|{v}"] = m.sum() / max(1, int(d["n_daq"]))
            for c in ("orofacial", "orofacial_left", "orofacial_right", "limb"):
                i = oi[c]
                n = 0
                for pol in ("res_pos", "res_neg"):
                    a = fr[k][pol]
                    p = a[:, 1].astype(int)
                    n += int(((a[:, 0] == i) & (a[:, 3] == 1) & (a[:, 2] > K * sd_fr[an][i]) & m[p] & okX[p]).sum())
                r[f"{c}|res_both|{v}"] = n / mins_fr if mins_fr >= 0.25 else np.nan
                n = 0
                for pol in ("pos", "neg"):
                    a = v2[k][pol]
                    p = a[:, 1].astype(int)
                    n += int(((a[:, 0] == i) & (a[:, 3] == 1) & (a[:, 2] > K * sd_v2[an][i]) & m[p]).sum())
                r[f"{c}|raw_both|{v}"] = n / mins if mins >= 0.25 else np.nan
                r[f"{c}|sd|{v}"] = float(d[f"sd_{v}"][i])
            P = d[f"psd_{v}"]
            for c in ("orofacial", "orofacial_left", "orofacial_right", "limb"):
                j = ps.index(c)
                r[f"{c}|p14|{v}"] = float(np.log10(np.nanmean(P[j, lo])))
                r[f"{c}|sync|{v}"] = float(np.log10(np.nanmean(P[j, lo]) / np.nanmean(P[j, hi])))
            r[f"oro_ipsi_minus_contra|sync|{v}"] = r[f"orofacial_left|sync|{v}"] - r[f"orofacial_right|sync|{v}"]
            r[f"nseg|{v}"] = int(d[f"nseg_{v}"])
        # published-table cross-check (DAQ variant must reproduce)
        r["pub_quiet_min_fr"] = int(fr[k]["n_quiet_eng"]) / fs / 60
        r["pub_quiet_min_v2"] = int(v2[k]["n_quiet_eng"]) / fs / 60
        r["pub_sd_oro"] = float(v2[k]["sd_q"][oi["orofacial"]])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


MEAS = [("orofacial", "res_both", "face-regressed events / min (k=4)"),
        ("orofacial", "raw_both", "raw v2 events / min (k=4)"),
        ("orofacial", "sd", "quiet robust SD (dF/F %)"),
        ("orofacial", "p14", "log10 1-4 Hz power"),
        ("orofacial", "sync", "sync index log10(1-4/8-15 Hz)"),
        ("oro_ipsi_minus_contra", "sync", "sync index, ipsi - contra")]


def stats(S):
    import pandas as pd
    from wfield_local import mixed_models as mm
    L = []
    for c, m, _ in MEAS + [("orofacial_left", "res_both", ""), ("orofacial_right", "res_both", ""),
                           ("limb", "res_both", "")]:
        for v in VARIANTS:
            y = S[f"{c}|{m}|{v}"]
            y = np.log10(y + 0.2) if m in ("res_both", "raw_both") else y
            L.append(S[["label"]].assign(column=c, measure=f"{m}|{v}", y=y.to_numpy()))
    L = mm.add_time_covariates(pd.concat(L), august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(L, ["measure", "column"], models=("epoch",), print_animals=False, fdr_by=("measure",))
    T.to_csv(OUT / "lmm_lp_quiet.csv", index=False)
    return T


def figure(S, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    show = ["daq", "lp", "lpjaw", "lpcov"]
    fig, ax = plt.subplots(len(MEAS), len(show), figsize=(3.3 * len(show) + 1.6, 2.35 * len(MEAS) + 1.2), sharey="row",
                           squeeze=False)
    animals = sorted(S.animal.unique())
    for i, (c, m, lab) in enumerate(MEAS):
        for j, v in enumerate(show):
            a, col = ax[i, j], f"{c}|{m}|{v}"
            for x, ep in enumerate(EP):
                B = S[S.epoch == ep]
                for an, A in B.groupby("animal"):
                    a.scatter(x + (animals.index(an) - 1.5) * 0.09 + np.zeros(len(A)), A[col], marker=ANIMAL_MARKER[an],
                              facecolors="none", edgecolors=EPOCH_LINE[ep], s=14, lw=0.8, alpha=0.8)
                am = B.groupby("animal")[col].mean()
                a.plot([x - 0.25, x + 0.25], [am.mean()] * 2, color=EPOCH_LINE[ep], lw=2.2)
                t = T[(T.measure == f"{m}|{v}") & (T.column == c) & (T.term == f"C(epoch)[T.{ep}]")]
                if x > 0 and len(t) and float(t.q.iloc[0]) < 0.05:
                    q = float(t.q.iloc[0])
                    sig_marks.star_top(a, x, "***" if q < 0.001 else "**" if q < 0.01 else "*", color=EPOCH_LINE[ep])
            if m == "sync" and c.startswith("oro_"):
                a.axhline(0, color="0.6", lw=0.7)
            a.set_xticks(range(len(EP)), EP, fontsize=7)
            if j == 0:
                a.set_ylabel(lab, fontsize=7.5)
            if i == 0:
                a.set_title(VNAME[v], fontsize=8.5)
    handles = [plt.Line2D([], [], marker=mk, ls="", mfc="none", mec="k", label=an) for an, mk in ANIMAL_MARKER.items()]
    handles += [plt.Line2D([], [], color=EPOCH_LINE[ep], lw=2.2, label=f"{ep} (mean of animals)") for ep in EP]
    sig_marks.reserve_headroom(fig)
    fig.legend(handles=handles, loc="center right", fontsize=7, frameon=False, bbox_to_anchor=(0.995, 0.5))
    fig.suptitle("Quiet-period orofacial measures with QUIET re-defined to also exclude video (LP) licks: columns = quiet "
                 "definition (each a subset of the previous; lpcov = video-tracked frames only).\nEngaged, August pre, "
                 "k = 4 x animal pre SD (published thresholds). Open markers = sessions (shape = animal, colour = epoch); "
                 "bar = mean of animals; stars = LMM epoch vs pre, FDR.", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.87, 0.95))
    fig.savefig(OUT / "quiet_lp_sensitivity.png", dpi=110)
    print("wrote", OUT / "quiet_lp_sensitivity.png")


def summarise():
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    S = session_table()
    S.to_csv(OUT / "lp_quiet_sessions.csv", index=False)
    print(f"{len(S)} sessions; cam1 source: {S.cam1_src.value_counts().to_dict()}; jaw table: {S.has_jaw.sum()}")
    print("\nCHECK DAQ variant vs published: max |quiet_min - published| v2 %.4f, FR %.4f; max |SD - published| %.2e"
          % ((S["quiet_min|daq"] - S.pub_quiet_min_v2).abs().max(), (S["quiet_min_fr|daq"] - S.pub_quiet_min_fr).abs().max(),
             (S["orofacial|sd|daq"] - S.pub_sd_oro).abs().max()))
    print("\nLP onset - nearest DAQ contact onset (ms, median per session), by animal x epoch:\n",
          S.pivot_table(index="animal", columns="epoch", values="lp_lag_ms", aggfunc="median").reindex(columns=EP).round(1).to_string())
    print("\nvideo coverage of the session (licks_all_coverage covered_frac), median:\n",
          S.pivot_table(index="animal", columns="epoch", values="covered_frac", aggfunc="median").reindex(columns=EP).round(3).to_string())
    for v in VARIANTS[1:]:
        print(f"\nshare of DAQ-quiet engaged time REMOVED by '{v}' (1 - kept), animal x epoch mean:\n",
              (1 - S.pivot_table(index="animal", columns="epoch", values=f"kept|{v}", aggfunc="mean").reindex(columns=EP)).round(3)
              .to_string())
        print("  pooled (sum of minutes) by epoch:",
              {ep: round(1 - S[S.epoch == ep][f"quiet_min|{v}"].sum() / S[S.epoch == ep]["quiet_min|daq"].sum(), 3)
               for ep in EP})
    print("\nquiet minutes per session (median), DAQ vs LP:\n",
          S.groupby(["animal", "epoch"])[["quiet_min|daq", "quiet_min|lp", "quiet_min|lpcov"]].median().round(1)
          .to_string())
    for c, m, lab in MEAS:
        tab = S.groupby(["animal", "epoch"])[[f"{c}|{m}|{v}" for v in VARIANTS]].mean()
        tab.columns = VARIANTS
        tab = tab.reindex(pd.MultiIndex.from_product([sorted(S.animal.unique()), EP]))
        print(f"\n=== {c} {lab}: animal x epoch mean, by quiet definition ===\n", tab.round(3).to_string())
        if m in ("res_both", "raw_both"):
            pre = tab.xs("pre", level=1)
            print("  acute / pre ratio per animal:\n",
                  (tab.xs("acute", level=1) / pre).round(2).to_string())
    T = stats(S)
    show = T[(T.model == "epoch") & T.term.str.startswith("C(epoch)")]
    print("\nLMM epoch model vs August pre (rates log10(rate + 0.2)):\n",
          show[["measure", "column", "term", "coef", "se", "p", "q", "n"]].round(4).to_string(index=False))
    figure(S, T)
    (OUT / "README.txt").write_text(__doc__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.parallel import fan_out
        labels = a.only or sorted(f.stem for f in (FRV2 / "sessions").glob("*.npz"))
        res, fail = fan_out(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
    summarise()


if __name__ == "__main__":
    main()

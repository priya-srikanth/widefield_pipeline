"""Example rest traces with the spike threshold drawn, pre vs acute per animal (cohort1, 2026-11-01).

Priya 11-01: "can you show example traces pre- and post-stroke with the thresholds you used? how did you decide spike
threshold". Same traces, rest frames and threshold as quiet_area_spikes.py / quiet_spikes_fr.py (frozen FR basis):
orofacial and limb (contralesional) dF/F % minus the 2-s running median; ENGAGED restdock05 rest shaded; licks / running
from the canonical behaviour events; threshold = 5 x the animal's AUGUST-pre median engaged-rest robust SD, applied to
PROMINENCE (height above the higher of the two flanking troughs), drawn as a bar from each detected peak's base; the
dashed line = rest median + threshold, a guide only. Grey = the face-regressed residual (out of fold) with its own
threshold. Window = the 60 s of the session's middle third with the most engaged rest.

    python scripts/cohort2_qc/quiet_spikes_examples.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import CAND_SD, DETREND_S, EVENTS, K_SD, MAX_HW_S, MIN_HW_FR, columns, session_traces  # noqa: E402
from quiet_spikes_fr import FR_OUT, N_FOLD, face_regressors, lagged, ridge_oof  # noqa: E402

PAIRS = {"PS92": ("0811", "0820", "0828"), "PS93": ("0811", "0820"), "PS94": ("0811", "0819"), "PS95": ("0811", "0818")}
SHOW = ["orofacial_right", "limb_right"]
WIN_S = 60.0
QUIET_MIN_S = 0.0      # Priya 11-01: no lick / running within 1.5 s before AND after (3 s total) is the whole rule; no extra minimum


def one(label):
    """-> dict with the window's traces, rest mask, lick / run times (worker; module-level for spawn)."""
    X, ft, fs, rest, engaged, _ = session_traces(label)
    w = int(round(DETREND_S * fs)) | 1
    R, _ = face_regressors(label, ft, w)
    XL = lagged(R, fs)
    okX = np.isfinite(XL).all(1)
    rest = rest & okX & np.isfinite(X).all(0)
    T = X.shape[1]
    pred, _ = ridge_oof(XL, X.T, rest, np.arange(T) * N_FOLD // T)
    Res = X - np.nan_to_num(pred.T)
    re = rest & engaged
    n = int(WIN_S * fs)
    cs = np.r_[0, np.cumsum(re)]
    starts = np.arange(T // 3, 2 * T // 3 - n, int(fs))
    a = int(starts[np.argmax(cs[starts + n] - cs[starts])])
    cols = columns()
    idx = [cols.index(c) for c in SHOW]
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    fsd = float(z["fs"])
    lk = [k for k in z.files if "lick" in k and z[k].ndim == 1 and z[k].dtype.kind in "if"]
    licks = z[lk[0]] / fsd if lk else np.array([])
    run = np.c_[z["running_starts"], z["running_stops"]] / fsd
    from wfield_local import config
    sess = next(x for x in config.load_sessions() if x["label"] == label)
    sw = Path(sess["mc"]) / "wfield_local_results" / "swapped_pairs.npz"
    bad = np.zeros(T, bool)
    if sw.exists():
        b = np.load(sw)["bad"][:T].astype(bool)
        bad[:len(b)] = b
    from wfield_local import trial_windows as TW
    tb = TW.trial_bounds(animal, f"2026{mmdd}")
    spout = np.c_[np.where(np.isfinite(tb.strobe_s), tb.strobe_s, tb.cue_s - 0.5), tb.stop_s].astype(float)
    cues = tb.cue_s.to_numpy(float)
    # QUIET (Priya 11-01: "rest for this particular analysis can include when the spout is present or moving, as long as
    # the animal isn't licking or running"): no contact lick or running within 1.5 s, stretches >= QUIET_MIN_S; spout / cue ignored
    busy = np.zeros(T, bool)
    for ev in (licks,):
        for x in ev[np.isfinite(ev)]:
            busy[np.searchsorted(ft, x - 1.5):np.searchsorted(ft, x + 1.5)] = True
    for r0, r1 in run:
        busy[np.searchsorted(ft, r0 - 1.5):np.searchsorted(ft, r1 + 1.5)] = True
    quiet = ~busy & ~bad
    e = np.flatnonzero(np.diff(np.r_[0, quiet.astype(int), 0]))
    q2 = np.zeros(T, bool)
    for q0, q1 in zip(e[::2], e[1::2]):
        if (q1 - q0) >= QUIET_MIN_S * fs:
            q2[q0:q1] = True
    return dict(label=label, quiet=q2[a:a + n], spout=spout, cues=cues, bad=bad[a:a + n], t=ft[a:a + n] - ft[a], t0=ft[a], X=X[idx, a:a + n], Res=Res[idx, a:a + n],
                rest=re[a:a + n], fs=fs, licks=licks, run=run, lick_key=lk[0] if lk else "")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import find_peaks
    from wfield_local.orofacial_cohort import EPOCH_LINE
    from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
    labels = [f"{an}_{d}" for an, ds in PAIRS.items() for d in ds]
    res, fail = fan_sessions(labels, one, jobs=4, label="session")
    if fail:
        raise SystemExit(f"failed: {fail}")
    D = {r["label"]: r for _, r in res} if isinstance(res[0], tuple) else {r["label"]: r for r in res}
    # thresholds = exactly what quiet_spikes_fr.session_table used
    S = {f.stem: dict(np.load(f)) for f in sorted((FR_OUT / "sessions").glob("*.npz"))}
    from wfield_local import epochs
    cols = columns()
    thr = {}
    for an in PAIRS:
        pre = [d for k, d in S.items() if k.startswith(an) and epochs.epoch_of(k) == "pre" and k[-4:] >= "0806"]
        for kind in ("raw", "res"):
            thr[an, kind] = K_SD * np.nanmedian([d[f"sd_{kind}"] for d in pre], 0)
    fig, ax = plt.subplots(len(labels), 1, figsize=(15, 2.3 * len(labels)), sharex=True)
    rows = []
    for i, lab in enumerate(labels):
        d, an, a = D[lab], lab[:4], ax[i]
        ep = epochs.epoch_of(lab)
        t, fs = d["t"], d["fs"]
        edges = np.flatnonzero(np.diff(np.r_[0, d["rest"].astype(int), 0]))
        for r0, r1 in zip(edges[::2], edges[1::2]):
            a.axvspan(t[r0], t[min(r1, len(t) - 1)], color="0.85", lw=0)
        qe = np.flatnonzero(np.diff(np.r_[0, d["quiet"].astype(int), 0]))
        for r0, r1 in zip(qe[::2], qe[1::2]):
            a.axvspan(t[r0], t[min(r1, len(t) - 1)], ymin=0.93, ymax=1.0, color="tab:purple", alpha=0.5, lw=0)
        for s0, s1 in d["spout"] - d["t0"]:
            if s1 > 0 and s0 < t[-1]:
                a.plot([max(s0, 0), min(s1, t[-1])], [-3.7] * 2, color="0.3", lw=2)
        cu = d["cues"] - d["t0"]
        cu = cu[(cu >= 0) & (cu <= t[-1])]
        a.plot(cu, np.full(len(cu), -2.5), "^", color="k", ms=4)
        off = 0.0
        for j, c in enumerate(SHOW):
            ci = cols.index(c)
            for kind, Y, colr, lw in (("res", d["Res"][j], "0.55", 0.7), ("raw", d["X"][j], EPOCH_LINE[ep] if j == 0
                                                                          else "tab:blue", 0.9)):
                th = float(thr[an, kind][ci])
                y = Y - np.median(Y[d["rest"]]) + off
                a.plot(t, y, color=colr, lw=lw)
                pk, pr = find_peaks(np.nan_to_num(Y), prominence=CAND_SD * th / K_SD, width=(MIN_HW_FR, MAX_HW_S * fs),
                                    rel_height=0.5)
                keep = (pr["prominences"] > th) & d["rest"][pk]
                for p, q in zip(pk[keep], pr["prominences"][keep]):
                    a.plot([t[p]] * 2, [y[p] - q, y[p]], color="k" if kind == "raw" else "0.4", lw=1.6)
                    a.plot(t[p], y[p] + 0.15, "v", color="k" if kind == "raw" else "0.4", ms=5)
                if kind == "raw":                        # v2 (11-01): HEIGHT above the preceding 1 s, no swapped frame
                    pk2, _ = find_peaks(np.nan_to_num(Y), width=(MIN_HW_FR, MAX_HW_S * fs), rel_height=0.5)
                    nb = int(fs)
                    badw = np.convolve(d["bad"].astype(float), np.ones(2 * int(0.5 * fs) + 1), "same") > 0
                    k2 = [p for p in pk2 if p >= nb and (d["rest"][p] or d["quiet"][p]) and not badw[p]
                          and Y[p] - np.median(Y[p - nb:p - int(0.2 * fs)]) > th]
                    for p in k2:
                        a.plot(t[p], y[p] + 0.6, "*", color="gold", mec="k", mew=0.5, ms=10, zorder=7)
                    rows.append(dict(label=lab, column=c, kind="raw_v2_height", threshold=th, n_in_window=len(k2)))
                if kind == "raw":
                    a.axhline(off + th, color=colr, ls="--", lw=0.8)
                    a.text(t[-1] + 0.5, off, f"{c.split('_')[0]}\n5x pre SD = {th:.2f}%", fontsize=7, va="center")
                rows.append(dict(label=lab, column=c, kind=kind, threshold=th, n_in_window=int(keep.sum())))
            off += 7.0
        lk = d["licks"] - d["t0"]
        lk = lk[(lk >= 0) & (lk <= t[-1])]
        a.plot(lk, np.full(len(lk), -3.0), "|", color="tab:red", ms=6)
        for r0, r1 in d["run"] - d["t0"]:
            if r1 > 0 and r0 < t[-1]:
                a.plot([max(r0, 0), min(r1, t[-1])], [-3.3] * 2, color="tab:green", lw=3)
        a.set_xlim(0, t[-1])
        a.set_ylabel(f"{lab}\n{ep}", fontsize=8)
        a.set_ylim(-4, off + 2)
    ax[-1].set_xlabel("s (60-s window, middle third of the session with the most engaged rest)")
    fig.suptitle("Rest traces (dF/F %, 2-s running median removed): orofacial (epoch colour) and limb (blue), "
                 "contralesional; grey = face-regressed residual. Shaded = engaged rest; black bar = a detected spike's "
                 "prominence (must exceed 5 x the animal's pre-stroke rest robust SD, labelled); dashed = rest median + "
                 "threshold (guide); gold star = v2 (height above the preceding 1 s > threshold, no swapped frame within 0.5 s, in rest OR quiet). Grey bands = engaged restdock05 rest; purple top strip = QUIET (no lick / running within 1.5 s before or after; spout may be out); black triangles = cues; dark bar = spout out (strobe -> trial end); red ticks = DAQ contact licks; green = running.", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0, 0.96, 0.95))
    out = FR_OUT / "rest_spike_example_traces.png"
    fig.savefig(out, dpi=110)
    pd.DataFrame(rows).to_csv(FR_OUT / "rest_spike_example_traces.csv", index=False)
    print("wrote", out, "| lick key", {k: D[k]["lick_key"] for k in D})


if __name__ == "__main__":
    main()

"""Example traces for the v2 quiet-event detector with the +/- thresholds drawn (cohort1, 2026-11-01).

Priya 11-01: "do you have an updated set of example traces with the +- thresholds". Same sessions as
quiet_spikes_examples.py (v1), drawn with exactly what quiet_spikes_v2.py counts at the common k = 4: the trace the
detector sees (dF/F % minus its 2-s running median), QUIET frames shaded (no lick -- DAQ contact, or LP video under QS_LICK_SOURCE=daq+lp -- / running within 1.5 s
before or after; spout may be out; no swapped pair within 0.5 s), and every event = a peak whose height above the median
of the preceding 1.0 .. 0.2 s exceeds 4 x the animal's August-pre robust SD, positive (up) or negative (down). Each event
is drawn as a bar from its baseline to its peak, so the bar length is the height the threshold was applied to. Window =
the 60 s of the session's middle third with the most engaged quiet.

    python scripts/cohort2_qc/quiet_spikes_v2_examples.py
"""
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import EVENTS, columns, session_traces  # noqa: E402
from quiet_spikes_v2 import BASE_S, LICK_SOURCE, V2, busy_frames, detect, detrend  # noqa: E402

PAIRS = {"PS92": ("0811", "0820", "0828"), "PS93": ("0811", "0820"), "PS94": ("0811", "0819"), "PS95": ("0811", "0818")}
SHOW = [("orofacial_left", "orofacial ipsi"), ("orofacial_right", "orofacial contra"), ("limb_left", "limb ipsi"),
        ("limb_right", "limb contra")]
K, WIN_S, OFF = 4.0, 60.0, 6.0
SEARCH = "middle"                                  # window search: "middle" third (Fig 23.6) | "intask" (movie)


def one(label):
    """Window traces, quiet mask, event (frame, signed height, baseline) per column, behaviour (spawn worker)."""
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    X = detrend(Xr, fs)
    busy, badw = busy_frames(label, ft, fs)
    quiet = ~busy & ~badw & np.isfinite(X).all(0)
    qe = quiet & engaged
    T, n = X.shape[1], int(WIN_S * fs)
    cs = np.r_[0, np.cumsum(qe)]
    starts = np.arange(T // 3, 2 * T // 3 - n, int(fs))
    if SEARCH == "intask":                         # movie (Priya 11-02: a FULLY quiet pre window): anywhere in the task,
        from wfield_local import trial_windows as TW   # the window ending before the session's last cue
        an, md = label.split("_")
        lastcue = np.nanmax(TW.trial_bounds(an, f"2026{md}").cue_s.to_numpy(float))
        starts = np.arange(0, T - n, max(int(fs) // 4, 1))
        starts = starts[ft[starts + n] < lastcue]
    a = int(starts[np.argmax(cs[starts + n] - cs[starts])])
    cols = columns()
    b0, b1 = int(round(BASE_S[0] * fs)), int(round(BASE_S[1] * fs))
    ev = {}
    for c, _ in SHOW:
        i, L = cols.index(c), []
        for sign in (1, -1):
            pk, h = detect(sign * X[i], quiet, fs)
            for p, v in zip(pk, h):
                if a <= p < a + n and engaged[p]:
                    L.append((p - a, sign * v, np.median(X[i, p - b0:p - b1])))
        ev[c] = np.array(L, float).reshape(-1, 3)
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    fsd = float(z["fs"])
    from wfield_local import trial_windows as TW
    tb = TW.trial_bounds(animal, f"2026{mmdd}")
    spout = np.c_[np.where(np.isfinite(tb.strobe_s), tb.strobe_s, tb.cue_s - 0.5), tb.stop_s].astype(float)
    lp = np.array([])
    if LICK_SOURCE == "daq+lp":                    # LP video licks (incl. non-contact) also define the quiet frames
        from lp_licks import lick_times
        lp = np.asarray(lick_times(label), float)
    return dict(label=label, t=ft[a:a + n] - ft[a], t0=ft[a], X=X[[cols.index(c) for c, _ in SHOW], a:a + n],
                quiet=qe[a:a + n], ev=ev, licks=z["lick_onsets"] / fsd, lp=lp,
                run=np.c_[z["running_starts"], z["running_stops"]] / fsd, spout=spout, cues=tb.cue_s.to_numpy(float))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from quiet_fluct import animal_sd
    from wfield_local import epochs
    from wfield_local.orofacial_cohort import EPOCH_LINE
    from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
    labels = [f"{an}_{d}" for an, ds in PAIRS.items() for d in ds]
    res, fail = fan_sessions(labels, one, jobs=5, label="session")
    if fail:
        raise SystemExit(f"failed: {fail}")
    D = {r["label"]: r for r in (x[1] if isinstance(x, tuple) else x for x in res)}
    sd = animal_sd()
    cols = columns()
    colr = ["#7b3294", "#c2a5cf", "#1b7837", "#a6dba0"]
    fig, ax = plt.subplots(len(labels), 1, figsize=(14, 3.0 * len(labels)), sharex=True, sharey=True)   # one y-scale for all
    rows = []
    for r_, lab in enumerate(labels):
        d, an, a = D[lab], lab[:4], ax[r_]
        ep = epochs.epoch_of(lab)
        t = d["t"]
        e = np.flatnonzero(np.diff(np.r_[0, d["quiet"].astype(int), 0]))
        for q0, q1 in zip(e[::2], e[1::2]):
            a.axvspan(t[q0], t[min(q1, len(t) - 1)], color="0.9", lw=0)
        for j, (c, name) in enumerate(SHOW):
            off = -j * OFF
            th = K * float(sd[an][cols.index(c)])
            a.plot(t, d["X"][j] + off, color=colr[j], lw=0.8)
            E = d["ev"][c]
            E = E[np.abs(E[:, 1]) > th]
            # Priya 11-02: short markers just OUTSIDE the event (above an up peak, below a down trough) so they never
            # cover the trace; the event height above its baseline is in the csv, not drawn.
            for p, h, base in E:
                p = int(p)
                a.plot(t[p], base + h + off + 0.45 * np.sign(h), "v" if h > 0 else "^", color="k", ms=3.5, mew=0)
            a.text(60.8, off, f"{name}\n+/-{th:.2f}%", fontsize=7, va="center", color=colr[j])
            rows.append(dict(label=lab, column=c, threshold_pct=th, n_pos=int((E[:, 1] > th).sum()) if len(E) else 0,
                             n_neg=int((E[:, 1] < -th).sum()) if len(E) else 0))
        ybot = -(len(SHOW) - 1) * OFF - 4.5
        cu = d["cues"] - d["t0"]
        cu = cu[(cu >= 0) & (cu <= t[-1])]
        a.plot(cu, np.full(len(cu), ybot + 1.2), "|", color="tab:blue", ms=7, mew=1.5)   # cues: blue ticks (not triangles)
        for s0, s1 in d["spout"] - d["t0"]:
            if s1 > 0 and s0 < t[-1]:
                a.plot([max(s0, 0), min(s1, t[-1])], [ybot + 0.3] * 2, color="0.35", lw=2)
        lk = d["licks"] - d["t0"]
        lk = lk[(lk >= 0) & (lk <= t[-1])]
        a.plot(lk, np.full(len(lk), ybot + 0.8), "|", color="tab:red", ms=6)
        lp = d["lp"] - d["t0"]
        lp = lp[(lp >= 0) & (lp <= t[-1])]
        a.plot(lp, np.full(len(lp), ybot + 1.6), "|", color="tab:orange", ms=6)
        for r0, r1 in d["run"] - d["t0"]:
            if r1 > 0 and r0 < t[-1]:
                a.plot([max(r0, 0), min(r1, t[-1])], [ybot + 0.55] * 2, color="tab:green", lw=3)
        a.set_xlim(0, 66)
        a.set_ylim(ybot, OFF * 0.8)
        a.set_yticks([])
        a.set_ylabel(f"{lab}\n{ep}", fontsize=9, color=EPOCH_LINE[ep])
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
    for a in ax:                                   # the same 2 % dF/F scale bar in every row (shared y-scale)
        a.plot([0.5, 0.5], [2, 4], color="k", lw=2)
        a.text(0.8, 3, "2% dF/F", fontsize=7, va="center")
    ax[-1].set_xlabel("s (60-s window in the middle third of the session with the most engaged quiet)")
    fig.suptitle("Quiet-period events, v2 detector at +/-4 x the animal's pre-stroke robust SD (value at right). Traces = "
                 "dF/F % minus a 2-s running median; same y-scale in every row. Grey = quiet (no lick / running within 1.5 s "
                 "before or after; spout may be out). Black triangles = events (height above the median of the preceding "
                 "1.0-0.2 s beyond the threshold), pointing down onto an UP peak from above / up onto a DOWN trough from "
                 "below. Bottom: blue ticks = cues, dark bar = spout "
                 "out, red = DAQ contact licks" + (", orange = LP video licks" if LICK_SOURCE == "daq+lp" else "") + ", green = running.", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = V2 / "quiet_event_example_traces.png"
    save_png_svg(fig, out, dpi=110)
    pd.DataFrame(rows).to_csv(V2 / "quiet_event_example_traces.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Are quiet-period fluctuations FASTER after the stroke? Event width + trace autocorrelation (cohort1, 2026-11-01).

Priya 11-01, of the quiet SD / event-triggered maps: "fluctuations also look faster acutely - is that true (maybe fwhm)".

FRAMES / EVENTS: exactly quiet_spikes_v2 -- quiet = no DAQ contact lick / running within 1.5 s before or after, no swapped
pair within 0.5 s; ENGAGED part of the session; August sessions. Events = the v2 events (sessions/*.npz) at k = 4 x the
animal's August-pre robust SD, positive and negative (negative measured on the sign-flipped trace).

MEASURES, on the trace the detector sees (dF/F % minus its 2-s running median):
  * FWHM: full width at half the event's height above its baseline (median of the preceding 1.0 .. 0.2 s), crossings
    linearly interpolated between frames. Also RISE (half-height crossing before the peak -> peak) and DECAY (peak ->
    half-height crossing after). The detector only keeps half-widths of 1.5 frames .. 0.5 s, so widths are truncated
    at both ends -- the same truncation pre and post.
  * HEIGHT-MATCHED FWHM: only events 4-8 x the pre SD, because acute events are larger and a width can depend on height.
  * ACF HALF-TIME: lag at which the autocorrelation of the quiet trace falls to 0.5 (quiet engaged runs >= 3 s, products
    pooled over runs). Uses no events and no threshold, so it cannot be moved by the event count.
STATISTICS: per-animal table first, mixed_models.lmm_table epoch + time models (August pre), FDR within measure.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_event_width.py [--jobs 6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import columns, session_traces  # noqa: E402
from quiet_spikes_v2 import BASE_S, MAX_HW_S, V2, busy_frames, detrend  # noqa: E402

K = 4.0
COLS = ["orofacial_left", "orofacial_right", "limb_left", "limb_right", "orofacial", "limb"]
OUTW = V2 / "width"
ACF_MAX_S, RUN_MIN_S, MATCH = 1.5, 3.0, (4.0, 8.0)


def half_crossings(y, p, base, h, lim):
    """(left, right) half-height crossing positions in fractional frames around peak p, or NaN."""
    half = base + h / 2
    out = []
    for step in (-1, 1):
        j = p
        while 0 <= j + step < len(y) and abs(j - p) < lim and y[j + step] > half:
            j += step
        k = j + step
        if not (0 <= k < len(y)) or abs(j - p) >= lim or y[k] > half:
            out.append(np.nan)
            continue
        out.append(j + step * (y[j] - half) / (y[j] - y[k]))
    return out


def acf_half(y, qe, fs):
    """Lag (s) where the quiet-run autocorrelation falls to 0.5."""
    nl = int(round(ACF_MAX_S * fs))
    e = np.flatnonzero(np.diff(np.r_[0, qe.astype(int), 0]))
    num, den, s2, n = np.zeros(nl + 1), np.zeros(nl + 1), 0.0, 0
    for a, b in zip(e[::2], e[1::2]):
        if b - a < RUN_MIN_S * fs:
            continue
        z = y[a:b] - y[a:b].mean()
        for lag in range(nl + 1):
            num[lag] += np.dot(z[:len(z) - lag], z[lag:])
            den[lag] += len(z) - lag
        s2 += np.dot(z, z)
        n += len(z)
    if not n:
        return np.nan
    r = (num / den) / (s2 / n)
    j = np.flatnonzero(r < 0.5)
    if not len(j) or j[0] == 0:
        return np.nan
    j = j[0]
    return float((j - 1 + (r[j - 1] - 0.5) / (r[j - 1] - r[j])) / fs)


def extract(label):
    f = OUTW / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    X = detrend(Xr, fs)
    busy, badw = busy_frames(label, ft, fs)
    qe = ~busy & ~badw & np.isfinite(X).all(0) & engaged
    cols = columns()
    v2 = np.load(V2 / "sessions" / f"{label}.npz")
    b0, b1 = int(round(BASE_S[0] * fs)), int(round(BASE_S[1] * fs))
    lim = int(MAX_HW_S * fs) + 2
    ev, acf = [], []
    for c in COLS:
        i = cols.index(c)
        acf.append(acf_half(X[i], qe, fs))
        for sign, kind in ((1, "pos"), (-1, "neg")):
            y = sign * np.nan_to_num(X[i])
            a = v2[kind]
            for p, h in a[(a[:, 0] == i) & (a[:, 3] == 1), 1:3]:
                p = int(p)
                base = np.median(y[p - b0:p - b1])
                lo, hi = half_crossings(y, p, base, h, lim)
                ev.append((i, sign, p, h, (hi - lo) / fs, (p - lo) / fs, (hi - p) / fs))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, ev=np.array(ev, float).reshape(-1, 7), acf_half=np.array(acf), cols=np.array(COLS), fs=fs)
    return label


def session_table():
    import pandas as pd
    from quiet_fluct import animal_sd
    from wfield_local import epochs
    sd = animal_sd()
    cols = columns()
    rows = []
    for f in sorted((OUTW / "sessions").glob("*.npz")):
        lab, ep = f.stem, epochs.epoch_of(f.stem)
        if ep is None or lab[-4:] < "0806" or lab[:4] not in sd:
            continue
        z = np.load(f)
        E = z["ev"]
        for j, c in enumerate(COLS):
            i = cols.index(c)
            s = float(sd[lab[:4]][i])
            m = (E[:, 0] == i) & (E[:, 3] > K * s) & np.isfinite(E[:, 4])
            mm = m & (E[:, 3] <= MATCH[1] * s)
            rows.append(dict(label=lab, animal=lab[:4], epoch=ep, column=c, n=int(m.sum()),
                             fwhm=np.median(E[m, 4]) if m.sum() >= 3 else np.nan,
                             fwhm_matched=np.median(E[mm, 4]) if mm.sum() >= 3 else np.nan,
                             rise=np.median(E[m, 5]) if m.sum() >= 3 else np.nan,
                             decay=np.median(E[m, 6]) if m.sum() >= 3 else np.nan,
                             height_sd=np.median(E[m, 3]) / s if m.sum() >= 3 else np.nan,
                             acf_half=float(z["acf_half"][j])))
    return pd.DataFrame(rows)


MEAS = {"fwhm": "event FWHM (s)", "fwhm_matched": "FWHM, events 4-8 x pre SD (s)", "rise": "rise, half -> peak (s)",
        "decay": "decay, peak -> half (s)", "acf_half": "quiet-trace ACF half-time (s)"}


def figure(S, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import sig_marks
    from wfield_local.analysis_kit import boot_ci
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    EP = ["pre", "acute", "subacute", "chronic"]
    show = ["orofacial", "limb"]
    fig, ax = plt.subplots(len(show), len(MEAS), figsize=(3.3 * len(MEAS), 3.4 * len(show)))
    for r, c in enumerate(show):
        for j, m in enumerate(MEAS):
            a = ax[r, j]
            G = S[S.column == c]
            for k, ep in enumerate(EP):
                g = G[G.epoch == ep].dropna(subset=[m])
                for an, gg in g.groupby("animal"):
                    x = k + (sorted(S.animal.unique()).index(an) - 1.5) * 0.08
                    a.plot(np.full(len(gg), x), gg[m], ANIMAL_MARKER[an], mfc="none", mec=EPOCH_LINE[ep], ms=4,
                           alpha=0.6)
                if len(g) and g.animal.nunique() >= 2:
                    ci = boot_ci({an: list(A[m]) for an, A in g.groupby("animal")}, np.random.default_rng(0))
                    a.errorbar(k + 0.3, ci.point, yerr=[[ci.point - ci.lo], [ci.hi - ci.point]], fmt="s",
                               color=EPOCH_LINE[ep], ms=6, capsize=3)
            for k, ep in enumerate(EP[1:], 1):
                t = T[(T.measure == m) & (T.column == c) & (T.model == "epoch") & T.term.str.contains(ep)]
                if len(t):
                    q = float(t.q.iloc[0])
                    if q < 0.05:
                        sig_marks.star_top(a, k, "**" if q < 0.01 else "*")
            a.set_xticks(range(4))
            a.set_xticklabels(EP, fontsize=8)
            a.set_title(f"{c}: {MEAS[m]}", fontsize=9)
    sig_marks.reserve_headroom(fig)
    fig.suptitle("Are quiet-period events faster after the stroke? v2 events > 4 x pre SD (up + down), engaged quiet; "
                 "open markers = sessions (shape = animal), square = bootstrap mean +/- 95% CI; stars = LMM epoch vs "
                 "August pre, FDR", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTW / "quiet_event_width.png", dpi=110)


def summarise():
    import pandas as pd
    from wfield_local import mixed_models as mm
    pd.set_option("display.width", 220)
    S = session_table()
    S.to_csv(OUTW / "quiet_event_width_sessions.csv", index=False)
    for m in MEAS:
        print(f"\n{m}: per animal x epoch median (s)\n",
              S[S.column.isin(["orofacial", "limb"])].pivot_table(index=["column", "animal"], columns="epoch",
                                                                   values=m, aggfunc="median")
              [["pre", "acute", "subacute", "chronic"]].round(3).to_string())
    print("\nheight (x pre SD), per animal x epoch:\n",
          S[S.column == "orofacial"].pivot_table(index="animal", columns="epoch", values="height_sd",
                                                 aggfunc="median")[["pre", "acute", "subacute", "chronic"]].round(2))
    L = pd.concat([S[["label", "column"]].assign(measure=m, y=S[m].to_numpy()) for m in MEAS])
    L = mm.add_time_covariates(L, august_pre=True).dropna(subset=["y"])
    T = mm.lmm_table(L, ["measure", "column"], models=("epoch", "time"), print_animals=False, fdr_by=("measure",))
    T.to_csv(OUTW / "lmm_width.csv", index=False)
    print(T[(T.model == "epoch") & T.column.isin(["orofacial", "limb"]) & (T.term != "Intercept")].round(4).to_string())
    figure(S, T)
    (OUTW / "README.txt").write_text(__doc__)
    print("wrote", OUTW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = sorted(f.stem for f in (V2 / "sessions").glob("*.npz") if f.stem[-4:] >= "0806")
        res, fail = fan_sessions(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
    summarise()


if __name__ == "__main__":
    main()

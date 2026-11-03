"""T7: within engaged quiet, does the orofacial event rate rise with face STILLNESS? (cohort1, 2026-11-01)

Handoff STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md section 4, T7 (STATE contribution). Engaged-quiet face speed halves
acutely (T6), and pre-stroke disengagement already carries an orofacial 1-4 Hz signature, so a stiller (drowsier) face
could carry part of the acute rate increase.

STATE per frame = cam4 FaceRhythm point-PC speed (own basis, summed over PCs; quiet_spikes_fr.face_regressors) smoothed
over 2 s (centred), in units of the animal's August-pre median engaged-quiet speed. Engaged quiet frames as quiet_spikes_v2.
Events = quiet_spikes_v2_fr raw orofacial (bilateral) events, up + down, > 4 x the animal's pre robust SD, engaged.
Rate per stillness bin (speed / pre median: < 0.25, 0.25-0.5, 0.5-1, 1-2, > 2) per session; bins with < 0.5 min skipped.
(a) Does rate rise as speed falls (within session, all epochs)? (b) At MATCHED stillness, is the acute rate still higher
than pre? If (b) holds, stillness does not explain the increase.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_stillness.py [--jobs 3]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QS_FR_BASIS", "own")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import DETREND_S, columns, session_traces  # noqa: E402
from quiet_spikes_v2 import V2, busy_frames  # noqa: E402

OUTT = V2 / "stillness"
FRV = V2 / "fr_regressed_own" / "sessions"
K = 4.0
BINS = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)
EP = ["pre", "acute", "subacute", "chronic"]


def extract(label):
    f = OUTT / "sessions" / f"{label}.npz"
    if f.exists() or not (FRV / f"{label}.npz").exists():
        return label
    from quiet_spikes_fr import face_regressors
    from scipy.ndimage import uniform_filter1d
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    busy, badw = busy_frames(label, ft, fs)
    qe = ~busy & ~badw & np.isfinite(Xr).all(0) & engaged
    _, sp = face_regressors(label, ft, int(round(DETREND_S * fs)) | 1)
    ok = np.isfinite(sp)
    s = np.where(ok, sp, 0.0)
    w = int(round(2.0 * fs))
    sm = uniform_filter1d(s, w, mode="nearest") / np.maximum(uniform_filter1d(ok.astype(float), w, mode="nearest"), 1e-9)
    sm[uniform_filter1d(ok.astype(float), w, mode="nearest") < 0.5] = np.nan
    z = np.load(FRV / f"{label}.npz")
    gi = columns().index("orofacial")
    ev = np.concatenate([z["raw_pos"], z["raw_neg"]])
    ev = ev[(ev[:, 0] == gi) & (ev[:, 3] == 1)]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, speed=sm[qe].astype(np.float32), ev_frames=ev[:, 1].astype(int), ev_h=ev[:, 2],
                        ev_speed=sm[ev[:, 1].astype(int)].astype(np.float32), fs=fs,
                        eng_speed_med=float(np.nanmedian(sm[qe])) if qe.any() else np.nan)
    return label


def summarise():
    import pandas as pd
    from quiet_fluct import animal_sd
    from wfield_local import epochs, mixed_models as mm
    pd.set_option("display.width", 220)
    sd = animal_sd()
    gi = columns().index("orofacial")
    D = {}
    for f in sorted((OUTT / "sessions").glob("*.npz")):
        ep = epochs.epoch_of(f.stem)
        if ep is None or f.stem[-4:] < "0806":
            continue
        D[f.stem] = dict(np.load(f), epoch=ep)
    pre_med = {}
    for k, d in D.items():
        if d["epoch"] == "pre":
            pre_med.setdefault(k[:4], []).append(float(d["eng_speed_med"]))
    pre_med = {a: np.nanmedian(v) for a, v in pre_med.items()}
    rows = []
    for k, d in D.items():
        an = k[:4]
        if an not in pre_med:
            continue
        fs = float(d["fs"])
        s = d["speed"] / pre_med[an]
        es = d["ev_speed"] / pre_med[an]
        big = d["ev_h"] > K * sd[an][gi]
        for b in range(len(BINS) - 1):
            mins = np.sum((s >= BINS[b]) & (s < BINS[b + 1])) / fs / 60
            n = int(np.sum(big & (es >= BINS[b]) & (es < BINS[b + 1])))
            rows.append(dict(label=k, animal=an, epoch=d["epoch"], bin=b, minutes=mins,
                             rate=n / mins if mins >= 0.5 else np.nan))
    S = pd.DataFrame(rows)
    S.to_csv(OUTT / "stillness_sessions.csv", index=False)
    lab = [f"{BINS[b]:g}-{BINS[b + 1]:g}" for b in range(len(BINS) - 1)]
    print("share of engaged-quiet time per stillness bin (speed / animal's pre median), epoch medians:")
    T = S.assign(share=S.minutes / S.groupby("label").minutes.transform("sum"))
    print(T.pivot_table(index="epoch", columns="bin", values="share", aggfunc="median").reindex(EP)
          .set_axis(lab, axis=1).round(3).to_string())
    print("\nevents / min by stillness bin, animal x epoch (session medians):")
    print(S.pivot_table(index=["animal", "epoch"], columns="bin", values="rate", aggfunc="median")
          .set_axis(lab, axis=1).round(1).to_string())
    L = S.dropna(subset=["rate"]).assign(y=lambda x: np.log10(x.rate + 0.2), measure="rate_by_bin",
                                         binl=lambda x: x.bin.map(dict(enumerate(lab))))
    L = mm.add_time_covariates(L[["label", "measure", "binl", "y"]], august_pre=True).dropna(subset=["y"])
    R = mm.lmm_table(L, ["measure", "binl"], models=("epoch",), print_animals=False, fdr_by=("measure",))
    R.to_csv(OUTT / "lmm_rate_by_bin.csv", index=False)
    print("\nepoch effect WITHIN each stillness bin (log10 rate + 0.2):")
    print(R[R.term != "Intercept"][["binl", "term", "coef", "se", "p", "q"]].round(4).to_string())
    figure(S, R, lab)
    (OUTT / "README.txt").write_text(__doc__)


def figure(S, R, lab, out=None):
    """Priya 11-03 figure rules: x = face-speed bins (ordered, so the epoch mean line stays); epochs at
    cat_points.epoch_offsets; every session's value as a faded open marker (shape = animal, colour = epoch); per-animal
    medians small filled. Returns the plotted epoch means / medians (for the --plot-only check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import cat_points as cp
    from wfield_local.orofacial_cohort import EPOCH_LINE
    out = OUTT / "quiet_stillness.png" if out is None else out
    off = cp.epoch_offsets(len(EP), step=0.2)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    plotted = {}
    for i, ep in enumerate(EP):
        G = S[S.epoch == ep]
        for b in range(len(lab)):
            g = G[G.bin == b]
            v = g.rate.where(g.rate > 0)                     # log axis: a zero-rate session cannot be drawn
            cp.session_dots(ax[0], b + off[i], v, g.animal, EPOCH_LINE[ep], jitter=0.03, seed=100 * i + b)
        A = G.groupby(["animal", "bin"]).rate.median().unstack()
        for an, r in A.iterrows():
            ax[0].plot(r.index + off[i], r.values, cp.animal_marker(an), ls="none", color=EPOCH_LINE[ep], mec="w",
                       mew=0.4, ms=4.5, zorder=3)
        m = A.mean(0)
        plotted[("rate", ep)] = m
        ax[0].plot(m.index + off[i], m.values, color=EPOCH_LINE[ep], lw=2, label=ep, zorder=4)
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(lab)))
    ax[0].set_xticklabels(lab)
    ax[0].set_xlabel("face speed / animal's pre median (left = stillest)")
    ax[0].set_ylabel("orofacial events / min (raw, up + down, > 4 x pre SD)")
    h, l_ = ax[0].get_legend_handles_labels()
    h += cp.animal_handles(sorted(S.animal.unique()))
    h += [plt.Line2D([], [], ls="none", marker="o", color="0.35", mec="w", ms=4.5, label="animal median")]
    ax[0].legend(handles=h, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax[0].set_title("T7 event rate by face stillness, engaged quiet\n(faded open = sessions, small filled = animal "
                    "medians, line = mean of animals)", fontsize=8)
    T = S.assign(share=S.minutes / S.groupby("label").minutes.transform("sum"))
    for i, ep in enumerate(EP):
        G = T[T.epoch == ep]
        for b in range(len(lab)):
            g = G[G.bin == b]
            cp.session_dots(ax[1], b + off[i], g.share, g.animal, EPOCH_LINE[ep], jitter=0.03, seed=100 * i + b)
        m = G.groupby("bin").share.median()
        plotted[("share", ep)] = m
        ax[1].plot(m.index + off[i], m.values, "-o", color=EPOCH_LINE[ep], ms=5, mec="w", mew=0.6, label=ep, zorder=4)
    ax[1].set_xticks(range(len(lab)))
    ax[1].set_xticklabels(lab)
    ax[1].set_xlabel("face speed / animal's pre median")
    ax[1].set_ylabel("share of engaged-quiet time")
    ax[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax[1].set_title("time spent at each stillness level\n(faded open = sessions, line = median of sessions)", fontsize=8)
    fig.suptitle("T7: is the acute rise in quiet orofacial events explained by a stiller face? (cohort1)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("wrote", out)
    return plotted


def plot_only():
    """Re-render from the stored stillness_sessions.csv only (no extraction, no LMM) -> quiet_stillness_nolines.png."""
    import pandas as pd
    S = pd.read_csv(OUTT / "stillness_sessions.csv")
    lab = [f"{BINS[b]:g}-{BINS[b + 1]:g}" for b in range(len(BINS) - 1)]
    P = figure(S, None, lab, out=OUTT / "quiet_stillness_nolines.png")
    # the plotted summaries must be the stored table's numbers (independent pivot of the same CSV)
    for ep in EP:
        G = S[S.epoch == ep]
        want = G.pivot_table(index="animal", columns="bin", values="rate", aggfunc="median").mean(0)
        assert np.allclose(P[("rate", ep)].to_numpy(), want.to_numpy(), equal_nan=True), ep
        sh = (G.minutes / S.groupby("label").minutes.transform("sum").loc[G.index]).groupby(G.bin).median()
        assert np.allclose(P[("share", ep)].to_numpy(), sh.to_numpy(), equal_nan=True), ep
    print("plotted summaries == stored stillness_sessions.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="extract in reverse order and stop (a second helper process)")
    ap.add_argument("--plot-only", action="store_true", help="re-render from stillness_sessions.csv -> *_nolines.png")
    a = ap.parse_args()
    if a.plot_only:
        plot_only()
        return
    if not a.summarise:
        from wfield_local.analysis_kit import fan_sessions  # fan_out + sort (rule 9)
        labels = sorted(f.stem for f in FRV.glob("*.npz") if f.stem[-4:] >= "0806")
        if a.reverse:
            fan_sessions(labels[::-1], extract, jobs=a.jobs, label="session")
            return
        res, fail = fan_sessions(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
    summarise()


if __name__ == "__main__":
    main()

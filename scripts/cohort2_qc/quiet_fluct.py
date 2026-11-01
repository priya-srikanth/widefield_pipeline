"""What changes in quiet-period cortical activity after the stroke: spectrum, where (pixel map), and event shape
(cohort1, 2026-11-01).

Priya 11-01, after quiet_spikes_v2.py showed the acute change is a larger quiet-period fluctuation in BOTH directions
(the quiet-trace robust SD separates pre vs acute as well as any event count): "I'd like all 3" -- (1) the power
spectrum of the quiet trace, (2) a pixel map of quiet-period fluctuation size, (3) event-triggered averages.

FRAMES: identical to quiet_spikes_v2 -- QUIET = no DAQ contact lick or running bout within 1.5 s before or after (spout /
cue ignored), no swapped pair within 0.5 s; ENGAGED part of the session (before the quit). August sessions (pre = August
pre).
(1) SPECTRUM: area traces (dF/F %, NOT detrended), Welch (4-s segments, linear detrend per segment, Hann) on every quiet
    run >= 4 s, averaged over segments. Per animal: each epoch's median PSD over sessions / the animal's pre median.
(2) MAP: per pixel SD of the quiet frames after a LINEAR 2-s moving-average high-pass (linear so it commutes with the
    SVD: pixel variance = diag(U C U'), C = covariance of the high-passed temporal components over quiet frames; the
    event analyses' 2-s running MEDIAN is not linear). Per animal: log2(epoch median SD map / pre median SD map); pooled =
    mean over animals. Shared Allen grid, eroded stat mask (the encoder's inputs).
(3) EVENT-TRIGGERED: v2 events at k = 4 (height above the preceding 1.0-0.2 s > 4 x the animal's August-pre robust SD),
    engaged quiet, in the bilateral ORofacial or LIMB group, POSITIVE and NEGATIVE separately; average of the 2-s-median-
    detrended area traces and of the pixel movie (high-passed as in (2)) from -1 to +2 s around the peak.

    set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_fluct.py [--jobs 6] [--summarise]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import EP, OUT, columns, session_traces  # noqa: E402
from quiet_spikes_v2 import V2, busy_frames, detrend  # noqa: E402

FL = OUT / "v2_quiet" / "fluct"
K_EVT = 4.0
SEG_S, LAG_S = 4.0, (-1.0, 2.0)
MAP_LAGS_S = (-0.3, 0.0, 0.2, 0.5, 1.0)
TRACE_COLS = ["orofacial_left", "orofacial_right", "limb_left", "limb_right", "orofacial", "limb", "cortex"]
GROUPS = ("orofacial", "limb")


def animal_sd():
    """{animal: robust SD per column} = median over the animal's AUGUST-pre v2 sessions (the v2 threshold base)."""
    from wfield_local import epochs
    out = {}
    for f in sorted((V2 / "sessions").glob("*.npz")):
        if epochs.epoch_of(f.stem) == "pre" and f.stem[-4:] >= "0806":
            out.setdefault(f.stem[:4], []).append(np.load(f)["sd_q"])
    return {a: np.nanmedian(v, 0) for a, v in out.items()}


def extract(label):
    f = FL / "sessions" / f"{label}.npz"
    if f.exists():
        return label
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import welch
    from wfield_local import combined_encoding as ce
    Xr, ft, fs, rest, engaged, _ = session_traces(label, detrend=False)
    V, ft2, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    T = Xr.shape[1]
    busy, badw = busy_frames(label, ft, fs)
    qe = ~busy & ~badw & np.isfinite(Xr).all(0) & engaged
    cols = columns()
    ci = [cols.index(c) for c in TRACE_COLS]
    # (1) spectrum on quiet runs >= SEG_S
    nseg = int(round(SEG_S * fs))
    e = np.flatnonzero(np.diff(np.r_[0, qe.astype(int), 0]))
    P, W = None, 0
    for a, b in zip(e[::2], e[1::2]):
        if b - a < nseg:
            continue
        fr, p = welch(Xr[ci, a:b], fs=fs, nperseg=nseg, noverlap=nseg // 2, detrend="linear", axis=-1)
        w = 1 + (b - a - nseg) // (nseg // 2)
        P = p * w if P is None else P + p * w
        W += w
    psd = P / W if W else np.full((len(ci), nseg // 2 + 1), np.nan)
    if not W:
        fr = np.fft.rfftfreq(nseg, 1 / fs)
    # (2) pixel SD map, linear 2-s high-pass
    V = np.asarray(V, np.float64)
    Vh = V - uniform_filter1d(V, int(round(2.0 * fs)), axis=0, mode="nearest")
    Q = Vh[qe]
    C = np.cov(Q, rowvar=False)
    sdpix = 100 * np.sqrt(np.maximum(np.einsum("pk,kl,pl->p", Ub, C, Ub), 0))
    # (3) event-triggered averages (v2 events at K_EVT x animal pre SD)
    sd = animal_sd()[label[:4]]
    v2 = np.load(V2 / "sessions" / f"{label}.npz")
    Xd = detrend(Xr, fs)
    lags = np.arange(int(round(LAG_S[0] * fs)), int(round(LAG_S[1] * fs)) + 1)
    mlag = [int(np.argmin(np.abs(lags / fs - s))) for s in MAP_LAGS_S]
    eta_tr, eta_map, n_ev = {}, {}, {}
    for g in GROUPS:
        gi = cols.index(g)
        for kind in ("pos", "neg"):
            a = v2[kind]
            fr_ev = a[(a[:, 0] == gi) & (a[:, 3] == 1) & (a[:, 2] > K_EVT * sd[gi]), 1].astype(int)
            fr_ev = fr_ev[(fr_ev + lags[0] >= 0) & (fr_ev + lags[-1] < T)]
            n_ev[f"{g}_{kind}"] = len(fr_ev)
            if len(fr_ev):
                idx = fr_ev[:, None] + lags[None, :]
                eta_tr[f"{g}_{kind}"] = Xd[ci][:, idx].mean(1)                       # cols x lags
                mv = Vh[idx].mean(0)                                                   # lags x k
                eta_map[f"{g}_{kind}"] = 100 * (Ub @ mv[mlag].T).T                     # nmaplag x pixels
            else:
                eta_tr[f"{g}_{kind}"] = np.full((len(ci), len(lags)), np.nan)
                eta_map[f"{g}_{kind}"] = np.full((len(mlag), len(flat)), np.nan)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, freqs=fr, psd=psd, sdpix=sdpix.astype(np.float32), flat=flat, shape=np.array(shape),
                        parea=parea, lags_s=lags / fs, map_lags_s=lags[mlag] / fs, quiet_min=qe.sum() / fs / 60,
                        **{f"etatr_{k}": v for k, v in eta_tr.items()},
                        **{f"etamap_{k}": v.astype(np.float32) for k, v in eta_map.items()},
                        **{f"n_{k}": v for k, v in n_ev.items()})
    return label


# ------------------------------------------------------------------------------------------------------- summary
def load():
    from wfield_local import epochs
    D = {}
    for f in sorted((FL / "sessions").glob("*.npz")):
        ep = epochs.epoch_of(f.stem)
        if ep is None or (ep == "pre" and f.stem[-4:] < "0806"):
            continue
        D[f.stem] = dict(np.load(f), epoch=ep)
    return D


def to_img(v, flat, shape):
    img = np.full(int(np.prod(shape)), np.nan)
    img[flat] = v
    return img.reshape(shape)


def area_edges(parea, flat, shape):
    lab = to_img(parea.astype(float), flat, shape)
    e = np.zeros(shape, bool)
    e[:-1] |= (lab[:-1] != lab[1:]) & np.isfinite(lab[:-1]) & np.isfinite(lab[1:])
    e[:, :-1] |= (lab[:, :-1] != lab[:, 1:]) & np.isfinite(lab[:, :-1]) & np.isfinite(lab[:, 1:])
    return e


def fig_spectrum(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    k0 = next(iter(D))
    fr = D[k0]["freqs"]
    show = [0, 1, 2, 3]
    fig, ax = plt.subplots(2, len(show), figsize=(4 * len(show), 7), sharex=True)
    animals = sorted({k[:4] for k in D})
    rows = []
    for j, ci in enumerate(show):
        for ep in EP:
            per_an = []
            for an in animals:
                P = [d["psd"][ci] for k, d in D.items() if k.startswith(an) and d["epoch"] == ep]
                pre = [d["psd"][ci] for k, d in D.items() if k.startswith(an) and d["epoch"] == "pre"]
                if not P or not pre:
                    continue
                r = np.nanmedian(P, 0) / np.nanmedian(pre, 0)
                per_an.append(r)
                if ep != "pre":
                    ax[1, j].plot(fr, r, color=EPOCH_LINE[ep], lw=0.6, alpha=0.6, marker=ANIMAL_MARKER[an],
                                  markevery=12, ms=3, mfc="none")
                for lo, hi in ((0.1, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 15.0)):
                    m = (fr >= lo) & (fr < hi)
                    rows.append(dict(column=TRACE_COLS[ci], animal=an, epoch=ep, band=f"{lo:g}-{hi:g} Hz",
                                     ratio=float(np.nanmean(r[m]))))
            allP = [d["psd"][ci] for d in D.values() if d["epoch"] == ep]
            ax[0, j].loglog(fr[1:], np.nanmedian(allP, 0)[1:], color=EPOCH_LINE[ep], lw=2, label=ep)
            if ep != "pre" and per_an:
                ax[1, j].plot(fr, np.exp(np.nanmean(np.log(per_an), 0)), color=EPOCH_LINE[ep], lw=2.5, label=ep)
        ax[1, j].axhline(1, color="0.5", lw=0.8)
        ax[1, j].set_xscale("log")
        ax[0, j].set_title(TRACE_COLS[ci].replace("_left", " ipsilesional").replace("_right", " contralesional"),
                           fontsize=9)
        ax[1, j].set_xlabel("Hz")
    ax[0, 0].set_ylabel("PSD, (dF/F %)^2 / Hz\n(median over sessions)")
    ax[1, 0].set_ylabel("power ratio vs the animal's pre\n(thin = animals, thick = geometric mean)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Quiet-period power spectrum (no lick / running within 1.5 s; engaged; Welch 4-s segments): "
                 "which frequencies grow after the stroke", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FL / "quiet_spectrum.png", dpi=110)
    import pandas as pd
    B = pd.DataFrame(rows)
    B.to_csv(FL / "quiet_spectrum_bands.csv", index=False)
    return B


def fig_maps(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k0 = next(iter(D))
    flat, shape, parea = D[k0]["flat"], tuple(D[k0]["shape"]), D[k0]["parea"]
    if any(len(d["flat"]) != len(flat) or not np.array_equal(d["flat"], flat) for d in D.values()):
        raise SystemExit("sessions are not on one pixel grid -- cannot pool maps")
    edges = area_edges(parea, flat, shape)
    animals = sorted({k[:4] for k in D})
    pre = {an: np.nanmedian([d["sdpix"] for k, d in D.items() if k.startswith(an) and d["epoch"] == "pre"], 0)
           for an in animals}
    fig, ax = plt.subplots(len(animals) + 1, 4, figsize=(13, 3.0 * (len(animals) + 1)))
    vmax = 1.0
    for i, an in enumerate(animals + ["pooled"]):
        for j, ep in enumerate(EP):
            a = ax[i, j]
            if ep == "pre":
                v = pre[an] if an != "pooled" else np.nanmean([pre[x] for x in animals], 0)
                im0 = a.imshow(to_img(v, flat, shape), cmap="viridis", vmin=0, vmax=np.nanpercentile(v, 99))
                fig.colorbar(im0, ax=a, fraction=0.04, label="SD, dF/F %")
            else:
                def ratio(x, ep=ep):
                    P = [d["sdpix"] for k, d in D.items() if k.startswith(x) and d["epoch"] == ep]
                    return np.log2(np.nanmedian(P, 0) / pre[x]) if P else np.full(len(flat), np.nan)
                v = ratio(an) if an != "pooled" else np.nanmean([ratio(x) for x in animals], 0)
                im = a.imshow(to_img(v, flat, shape), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            a.imshow(np.where(edges, 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1.5, alpha=0.5)
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title("pre: SD" if ep == "pre" else f"{ep}: log2(SD / pre)", fontsize=9)
            if j == 0:
                a.set_ylabel(an, fontsize=9)
    fig.colorbar(im, ax=ax[:, 1:].ravel().tolist(), fraction=0.02, label="log2(SD / animal's pre SD)")
    fig.suptitle("Where quiet-period fluctuations grow: per-pixel SD (linear 2-s high-pass) over quiet engaged frames; "
                 "per animal median over sessions; ipsilesional = image left; Allen area edges", fontsize=9)
    fig.savefig(FL / "quiet_sd_maps.png", dpi=110, bbox_inches="tight")


def fig_eta(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local.orofacial_cohort import EPOCH_LINE
    k0 = next(iter(D))
    lags, mlags = D[k0]["lags_s"], D[k0]["map_lags_s"]
    flat, shape, parea = D[k0]["flat"], tuple(D[k0]["shape"]), D[k0]["parea"]
    edges = area_edges(parea, flat, shape)
    animals = sorted({k[:4] for k in D})
    keys = [f"{g}_{p}" for g in GROUPS for p in ("pos", "neg")]
    # traces: rows = trigger, cols = orofacial / limb trace (contralesional + ipsilesional), pre vs acute (animal mean)
    fig, ax = plt.subplots(len(keys), 2, figsize=(10, 2.6 * len(keys)), sharex=True)
    for i, key in enumerate(keys):
        for j, (cname, idx) in enumerate((("orofacial", (0, 1)), ("limb", (2, 3)))):
            for ep in EP:
                per = []
                for an in animals:
                    Z = [np.nanmean(d[f"etatr_{key}"][list(idx)], 0) for k, d in D.items()
                         if k.startswith(an) and d["epoch"] == ep and d[f"n_{key}"] >= 3]
                    if Z:
                        per.append(np.nanmean(Z, 0))
                if per:
                    ax[i, j].plot(lags, np.mean(per, 0), color=EPOCH_LINE[ep], lw=2, label=f"{ep} ({len(per)} an.)")
            ax[i, j].axvline(0, color="0.6", lw=0.6)
            ax[i, j].axhline(0, color="0.6", lw=0.6)
            ax[i, j].set_title(f"trigger: {key.replace('_pos', ' UP').replace('_neg', ' DOWN')} event -> "
                               f"{cname} trace", fontsize=8)
        ax[i, 0].set_ylabel("dF/F % (2-s median removed)")
    ax[0, 0].legend(fontsize=6)
    ax[-1, 0].set_xlabel("s from event peak")
    fig.suptitle("Event-triggered average traces (events > 4 x pre robust SD, quiet engaged); mean of animal means "
                 "(sessions with >= 3 events)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FL / "quiet_eta_traces.png", dpi=110)
    # maps: orofacial UP and limb UP, pre vs acute, at MAP_LAGS_S
    fig, ax = plt.subplots(4, len(mlags), figsize=(2.6 * len(mlags), 10.5))
    r = 0
    for key in ("orofacial_pos", "limb_pos"):
        for ep in ("pre", "acute"):
            per = []
            for an in animals:
                Z = [d[f"etamap_{key}"] for k, d in D.items()
                     if k.startswith(an) and d["epoch"] == ep and d[f"n_{key}"] >= 3]
                if Z:
                    per.append(np.nanmean(Z, 0))
            M = np.mean(per, 0) if per else np.full((len(mlags), len(flat)), np.nan)
            v = np.nanpercentile(np.abs(M), 99)
            for j in range(len(mlags)):
                im = ax[r, j].imshow(to_img(M[j], flat, shape), cmap="RdBu_r", vmin=-v, vmax=v)
                ax[r, j].imshow(np.where(edges, 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1.5, alpha=0.5)
                ax[r, j].set_xticks([])
                ax[r, j].set_yticks([])
                if r == 0:
                    ax[r, j].set_title(f"{mlags[j]:+.1f} s", fontsize=9)
            ax[r, 0].set_ylabel(f"{key.replace('_pos', ' UP')}\n{ep} ({len(per)} an.)", fontsize=8)
            fig.colorbar(im, ax=ax[r, :].tolist(), fraction=0.02, label="dF/F %")
            r += 1
    fig.suptitle("Event-triggered average maps (high-passed movie), mean of animal means; ipsilesional = image left",
                 fontsize=9)
    fig.savefig(FL / "quiet_eta_maps.png", dpi=110, bbox_inches="tight")


def summarise():
    import pandas as pd
    pd.set_option("display.width", 220)
    D = load()
    print(len(D), "sessions")
    n = pd.DataFrame([{"label": k, "animal": k[:4], "epoch": d["epoch"], **{x: int(d[f"n_{x}"]) for x in
                      ("orofacial_pos", "orofacial_neg", "limb_pos", "limb_neg")}} for k, d in D.items()])
    print("events per session (median), animal x epoch:\n",
          n.groupby(["animal", "epoch"])[["orofacial_pos", "orofacial_neg", "limb_pos", "limb_neg"]].median().to_string())
    B = fig_spectrum(D)
    print("\npower ratio vs pre by band (per animal):\n",
          B[B.epoch != "pre"].pivot_table(index=["column", "band"], columns=["epoch", "animal"], values="ratio")
          .round(2).to_string())
    fig_maps(D)
    fig_eta(D)
    (FL / "README.txt").write_text(__doc__)
    print("wrote", FL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if not a.summarise:
        from wfield_local.analysis_kit import curated_labels
        from wfield_local.parallel import fan_out
        labels = a.only or [k for k in sorted(curated_labels()) if k[-4:] >= "0806"]
        res, fail = fan_out(labels, extract, jobs=a.jobs, label="session")
        print(f"{len(res)} done, {len(fail)} failed")
        for x in fail:
            print("FAILED", x)
        if len(fail) > 3:
            raise SystemExit("too many failures")
    summarise()


if __name__ == "__main__":
    main()

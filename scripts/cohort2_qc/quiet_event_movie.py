"""Side-by-side widefield dF/F movie of engaged QUIET periods, PRE vs ACUTE, same animal (cohort1, 2026-11-02).

For the PI deck (Priya 11-02: "a video clip of pre vs acute as an example" of the quiet-period fluctuations). Default pair
= PS92 08-11 (pre) vs PS92 08-20 (acute), the same sessions as the example traces
(quiet_spikes_v2_examples.py, encoding page Fig 23.6), windows chosen by the same rule (see --window): the CLIP_S of the
session's middle third with the most engaged quiet frames is shown.

What is drawn (both panels on ONE colour scale):
  * movie = the adopted hemo-corrected SVD (`combined_encoding.load_imaging`: SVTcorr of hemo_<ADOPTED>, U_atlas on the
    Allen-aligned 540 x 640 grid, imaging-map brain mask), reconstructed U . SVT for the window, dF/F %, minus a 2-s
    moving average (high-pass on the temporal components, linear so identical per pixel; the detector itself uses a
    2-s running MEDIAN on the area traces); light Gaussian smoothing (sigma 1.5 px) for display only;
  * Allen area outlines (grey), off-brain = white; ipsilesional = image LEFT (all four lesions are left);
  * under each map: orofacial ipsi / contra traces exactly as the detector sees them (quiet_spikes_v2_examples.one), grey =
    quiet frames (no DAQ contact or LP video lick, no running within 1.5 s), black triangles = events beyond +/-4 x the
    animal's August-pre robust SD, a moving cursor; red / orange ticks = DAQ / LP licks, blue = cues;
  * "EVENT" flag beside the map title for +/-0.15 s around each orofacial event.
Played at PLAY x real time (imaging 31.2 Hz). H.264 / yuv420p / faststart, so it plays in Keynote / PowerPoint for Mac.

    set QS_LICK_SOURCE=daq+lp & set WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
    python scripts/cohort2_qc/quiet_event_movie.py [--pre PS92_0811] [--post PS92_0820] [--clip 15] [--play 0.5]
--window example (default) = the CLIP_S sub-window with the most engaged quiet inside the Fig 23.6 60-s window;
--window session = the Fig 23.6 rule applied directly at CLIP_S. For PS92 08-11 / 08-20 both give the same clips (11-02,
within 0.2 s): pre 4118.2 s (49 % quiet -- an engaged pre-stroke animal licks most of the time, so even its quietest 15 s
of the middle third is half licking), acute 3180.2 s (100 % quiet, 30 orofacial events).
Outputs: cohort1_quiet_spikes/v2_quiet_lp/movie/ (<pre>_vs_<post>_quiet[_session].mp4, a poster PNG, a csv of the windows).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QS_LICK_SOURCE", "daq+lp")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quiet_area_spikes import DETREND_S  # noqa: E402
from quiet_spikes_v2 import V2  # noqa: E402
import quiet_spikes_v2_examples as QX  # noqa: E402
from quiet_spikes_v2_examples import K, one  # noqa: E402

OUTD = V2 / "movie"
OROF = ("orofacial_left", "orofacial_right")


def movavg_highpass(V, w):
    """V (T, K) minus its centred w-frame moving average (edges: shrinking window)."""
    c = np.cumsum(np.r_[np.zeros((1, V.shape[1])), V.astype(np.float64)], 0)
    T, h = len(V), w // 2
    lo, hi = np.clip(np.arange(T) - h, 0, T), np.clip(np.arange(T) + h + 1, 0, T)
    return V - ((c[hi] - c[lo]) / (hi - lo)[:, None]).astype(np.float32)


def clip_window(d, n):
    """Start (index into the 60-s example window) of the n-frame sub-window with the most engaged quiet frames."""
    cs = np.r_[0, np.cumsum(d["quiet"])]
    st = np.arange(0, len(d["quiet"]) - n + 1)
    return int(st[np.argmax(cs[st + n] - cs[st])])


def session(label, clip_s, window="example"):
    from scipy.ndimage import gaussian_filter
    from quiet_fluct import area_edges, animal_sd, to_img
    from quiet_area_spikes import columns
    from wfield_local import combined_encoding as ce
    # window="example": the Fig 23.6 60-s window (most engaged quiet in the middle third), then its CLIP_S sub-window
    # with the most quiet; window="session": the same rule applied directly at CLIP_S (the CLIP_S of the middle third
    # with the most engaged quiet) -- used when the 60-s example window is mostly licking (pre-stroke).
    if window == "session":
        QX.WIN_S = clip_s
    d = one(label)                                            # window traces, quiet mask, events, licks
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    fs = 1.0 / np.median(np.diff(ft))
    a60 = int(np.searchsorted(ft, d["t0"]))
    n = min(int(round(clip_s * fs)), len(d["quiet"]))
    s0 = clip_window(d, n)
    a = a60 + s0
    pad = int(DETREND_S * fs) * 2                              # high-pass a padded stretch, keep the clip
    lo, hi = max(a - pad, 0), min(a + n + pad, len(V))
    Vh = movavg_highpass(V[lo:hi], int(round(DETREND_S * fs)) | 1)[a - lo:a - lo + n]
    M = 100 * (Ub @ Vh.T.astype(np.float32))                    # P_brain x n, dF/F %
    frames = np.stack([gaussian_filter(np.nan_to_num(to_img(M[:, i], flat, shape)), 1.5) for i in range(n)])
    inb = np.isfinite(to_img(np.ones(len(flat)), flat, shape))
    frames[:, ~inb] = np.nan
    cols = columns()
    sd = animal_sd()[label[:4]]
    th = {c: K * float(sd[cols.index(c)]) for c in OROF}
    ev = []
    for c in OROF:
        E = d["ev"][c]
        E = E[np.abs(E[:, 1]) > th[c]] if len(E) else E
        ev += [(int(p) - s0, h, base, c) for p, h, base in E if s0 <= p < s0 + n]
    return dict(label=label, frames=frames, inb=inb, edges=area_edges(parea, flat, shape), fs=fs, n=n,
                t=d["t"][s0:s0 + n] - d["t"][s0], X=d["X"][:2, s0:s0 + n], quiet=d["quiet"][s0:s0 + n], ev=ev,
                th=th, t_clip0=d["t0"] + d["t"][s0], t_win0=d["t0"], s0_s=d["t"][s0],
                licks=d["licks"] - d["t0"] - d["t"][s0], lp=d["lp"] - d["t0"] - d["t"][s0],
                cues=d["cues"] - d["t0"] - d["t"][s0])


def render(P, Q, path, play, lim, poster):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import epochs
    n = min(P["n"], Q["n"])
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.04], height_ratios=[1, 0.24, 0.07], left=0.03, right=0.95,
                          top=0.90, bottom=0.07, wspace=0.08, hspace=0.12)
    ims, curs, flags = [], [], []
    tr_c = ["#7b3294", "#c2a5cf"]
    for j, D in enumerate((P, Q)):
        ep = epochs.epoch_of(D["label"])
        am = fig.add_subplot(gs[0, j])
        im = am.imshow(D["frames"][0], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="bilinear")
        am.imshow(np.where(D["edges"], 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1.6, alpha=0.55)
        am.set_xticks([]), am.set_yticks([])
        for s in am.spines.values():
            s.set_visible(False)
        an, mmdd = D["label"].split("_")
        am.set_title(f"{an} {mmdd[:2]}-{mmdd[2:]}  {ep.upper() if ep == 'pre' else ep}", fontsize=15,
                     color="black" if ep == "pre" else "#c51b7d", fontweight="bold")
        am.text(0.02, 0.02, "ipsilesional", transform=am.transAxes, fontsize=9, color="0.35")
        am.text(0.98, 0.02, "contralesional", transform=am.transAxes, fontsize=9, color="0.35", ha="right")
        fl = am.text(0.98, 0.97, "", transform=am.transAxes, fontsize=14, ha="right", va="top", color="black",
                     fontweight="bold")
        flags.append(fl)
        ims.append(im)
        at = fig.add_subplot(gs[1, j])
        t = D["t"][:n]
        e = np.flatnonzero(np.diff(np.r_[0, D["quiet"][:n].astype(int), 0]))
        for q0, q1 in zip(e[::2], e[1::2]):
            at.axvspan(t[q0], t[min(q1, n - 1)], color="0.88", lw=0)
        off = 5.0
        for k in range(2):
            at.plot(t, D["X"][k, :n] - k * off, color=tr_c[k], lw=0.9)
        for p, h, base, c in D["ev"]:
            if p < n:
                k = OROF.index(c)
                at.plot(t[p], base + h - k * off + 0.5 * np.sign(h), "v" if h > 0 else "^", color="k", ms=4, mew=0)
        yb = -off - 3.2
        for arr, col, y in ((D["licks"], "tab:red", yb + 0.4), (D["lp"], "tab:orange", yb + 1.1),
                            (D["cues"], "tab:blue", yb + 1.8)):
            x = arr[(arr >= 0) & (arr <= t[-1])]
            at.plot(x, np.full(len(x), y), "|", color=col, ms=6, mew=1.2)
        at.text(t[-1] * 1.005, 0, "orofacial\nipsi", fontsize=8, color=tr_c[0], va="center")
        at.text(t[-1] * 1.005, -off, "orofacial\ncontra", fontsize=8, color="#9970ab", va="center")
        at.set_xlim(0, t[-1] * 1.1)
        at.set_ylim(yb, 4.5)
        at.set_yticks([])
        for s in ("top", "right", "left"):
            at.spines[s].set_visible(False)
        at.tick_params(labelsize=8)
        at.set_xlabel("s", fontsize=8, labelpad=0)
        curs.append(at.axvline(0, color="k", lw=1.2))
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(ims[0], cax=cax)
    cb.set_label("dF/F %  (2-s high-pass)", fontsize=10)
    lg = fig.add_subplot(gs[2, :2])
    lg.axis("off")
    lg.text(0.0, 0.5, "grey = quiet (no lick / running within 1.5 s)   triangles = events > 4 x pre SD   "
            "red / orange = DAQ / video licks   blue = cue   same colour scale in both panels", fontsize=9,
            color="0.3", va="center")
    ttl = fig.suptitle("", fontsize=13, y=0.975)
    fps = P["fs"] * play
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1,
                            pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"]) as w:
        for i in range(n):
            for j, D in enumerate((P, Q)):
                ims[j].set_data(D["frames"][i])
                curs[j].set_xdata([D["t"][i]] * 2)
                hot = any(abs(i - p) <= int(0.15 * D["fs"]) for p, *_ in D["ev"])
                flags[j].set_text("EVENT" if hot else "")
            ttl.set_text(f"Engaged quiet period, widefield dF/F (hemo-corrected)   t = {P['t'][i]:4.1f} s   "
                         f"(played at {play:g}x)")
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            w.append_data(img)
            if i == poster:
                imageio.imwrite(path.with_name(path.stem + "_poster.png"), img)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="PS92_0811")
    ap.add_argument("--post", default="PS92_0820")
    ap.add_argument("--clip", type=float, default=15.0)
    ap.add_argument("--play", type=float, default=0.5)
    ap.add_argument("--window", choices=["example", "session"], default="example",
                    help="example = inside the Fig 23.6 60-s windows; session = most-quiet CLIP_S of the middle third")
    ap.add_argument("--lim", type=float, default=None, help="colour limit, dF/F %% (default: 99.5th pct of |both|)")
    a = ap.parse_args()
    import pandas as pd
    OUTD.mkdir(parents=True, exist_ok=True)
    P, Q = session(a.pre, a.clip, a.window), session(a.post, a.clip, a.window)
    lim = a.lim or float(np.nanpercentile(np.abs(np.r_[P["frames"][:, P["inb"]].ravel(),
                                                        Q["frames"][:, Q["inb"]].ravel()]), 99.5))
    lim = float(np.round(lim, 1))
    # poster = the acute clip's first UP orofacial event after 1 s (else mid-clip)
    poster = min([p for p, h, *_ in Q["ev"] if p > Q["fs"] and h > 0] or [Q["n"] // 2])
    path = OUTD / (f"{a.pre}_vs_{a.post}_quiet" + ("_session" if a.window == "session" else "") + ".mp4")
    render(P, Q, path, a.play, lim, poster)
    rows = [dict(label=D["label"], window=a.window, window_start_daq_s=D["t_win0"], clip_offset_in_window_s=D["s0_s"],
                 clip_start_daq_s=D["t_clip0"], clip_s=a.clip, frames=D["n"], quiet_frac=float(D["quiet"].mean()),
                 n_orofacial_events=len(D["ev"]), colour_lim_pct=lim, play_speed=a.play,
                 thr_ipsi_pct=D["th"][OROF[0]], thr_contra_pct=D["th"][OROF[1]]) for D in (P, Q)]
    pd.DataFrame(rows).to_csv(path.with_name(path.stem + "_windows.csv"), index=False)
    print(pd.DataFrame(rows).to_string())
    print("wrote", path)


if __name__ == "__main__":
    main()

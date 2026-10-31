"""Video + traces of the fast 470-only events in PS122 10-29 (possible epileptiform discharges; QC doc §0o).

Priya 10-31: "can you show me (compressed) video and traces of the possible seizure events".

Reconstructed from the SVD (U . SVT, raw per channel dF/F; no hemo correction, so 470 and 415 are shown exactly
as recorded), brain-masked, 2x2 binned, with DAQ licks / cues placed on the frame clock (the same `_frames` the
analyses use). Outputs in labcams/cohort2/_tmp_qc_figures_20261031/seizure/:

  PS122_1029_events_realtime.mp4   -- 113.2-116.3 min (7 events), real time: 470 | 415 movies + scrolling cortex trace
  PS122_1029_event_slowmo.mp4      -- one event, -1 s to +2 s, 5x slowed
  PS122_1029_event_traces.png      -- the last 20 min with events / licks / cues; a 30-s zoom; every event
                                      overlaid with its mean; a frame strip of one event (-64 to +320 ms)
  PS122_1030_event_traces.png      -- the single 10-30 event (121.5 min), same layout, for comparison

    conda activate locanmf
    python scripts/cohort2_qc/seizure_events_video.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402

LABCAMS = Path("M:/MICROSCOPE/Priya/Widefield/labcams")
QC = LABCAMS / "cohort2" / "_tmp_qc_figures_20261031"
OUT = QC / "seizure"
DAQ = Path("M:/MICROSCOPE/Priya/Widefield/DAQ_recorder_output")
FS = 31.23
SESS = {"1029": ("20261029/PS122_20261029_192217", "20261029/PS122_20261029_192406.h5"),
        "1030": ("20261030/PS122_20261030_163240", None)}


def load(day):
    sdir, h5 = SESS[day]
    mc = LABCAMS / sdir / "motion_corrected"
    res = mc / "wfield_local_results"
    fa = np.load(res / "frames_average_raw.npy")
    br = brain_mask(fa, 1, res=res)
    U = np.nan_to_num(np.load(res / "U.npy")).astype(np.float32)
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    um = U.reshape(-1, U.shape[-1])[br.reshape(-1)].mean(0)
    x470 = 100 * (um @ np.asarray(SVT[:, 1::2], np.float64))
    x415 = 100 * (um @ np.asarray(SVT[:, 0::2], np.float64))
    ev = next(r for r in json.loads((QC / "fast_events.json").read_text()) if r["session"] == Path(sdir).name)
    # fast_events.json stores times rounded to 0.01 min (+-0.3 s): re-find each PEAK to the pair
    evf = []
    for f in np.round(np.array(ev["t_min"]) * 60 * FS).astype(int):
        lo = max(1, f - 15)
        evf.append(lo + int(np.argmax(x470[lo:f + 16])))
    evf = np.array(evf, int)
    licks = cues = csmp = np.array([], int)
    if h5 and (DAQ / h5).exists():
        from wfield_local.framemap_event_maps import _corrected_frame_samples
        from wfield_local.locanmf_crossanimal_dff import _frames
        from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
        from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
        s = dict(label=f"PS122_{day}", mc=str(mc), h5=str(DAQ / h5), regime="B", fmdir=None)
        cue = _lc(s["h5"])
        lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
        cues, licks, _ = _frames(s, cue, lk)
        fm = next(mc.glob("*cleanpairs_frame_map.npz"))
        summ = next(mc.glob("*cleanpairs_summary.json"), None)
        off = int(json.loads(summ.read_text()).get("chosen_exposure_offset", 0)) if summ else 0
        csmp = _corrected_frame_samples(fm, np.asarray(cue["pco_samples"]), off)   # DAQ sample of every pair
    return dict(U=U, SVT=SVT, br=br, x470=x470, x415=x415, ev=evf, licks=np.asarray(licks), cues=np.asarray(cues),
                csmp=np.asarray(csmp), h5=(DAQ / h5) if h5 else None)


CAM4 = Path("M:/MICROSCOPE/Priya/Behavior_Cameras/Widefield/20261029/PS122/cam4_2026-10-29T19_27_54")


def render_cam(d, a, b, path, step=2, cam_px=300, lim=12.0, cam=CAM4, tpl=None, title="PS122 10-29"):
    """470 movie | cam4 face video, real time at FS/step fps (low res), with the scrolling cortex trace.

    Camera frames are placed by DAQ time: pair -> DAQ sample (frame map) and camera timestamp -> DAQ seconds (a
    camera_sync template fitted HERE, in memory: none exists for 20261029 on the share and this script writes
    none). The template's residual is printed; `quality_ok` must hold."""
    import cv2
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import camera_sync
    if tpl is None:                               # no saved template: fit one in memory (never written)
        tpl = camera_sync.build_template(str(d["h5"]), str(cam) + ".csv")
    print(f"cam4 template: resid {tpl['resid_ms_rms']:.2f} ms rms, quality_ok {tpl['quality_ok']}, "
          f"drops {tpl['n_frame_drops']}", flush=True)
    if not tpl["quality_ok"]:
        raise SystemExit("cam4 template failed quality_ok -- not rendering a misaligned video")
    ts = np.loadtxt(str(cam) + ".csv", delimiter=",", dtype=np.int64, usecols=1)
    fs = float(tpl["fs_daq"])
    pairs = np.arange(a, b, step)
    daq_s = d["csmp"][pairs] / fs
    cam_s = (daq_s - float(tpl["intercept_daqSec"])) / float(tpl["slope_daqSec_per_camSec"])
    rows = np.clip(np.searchsorted(ts, cam_s * 1e9), 0, len(ts) - 1)
    cap = cv2.VideoCapture(str(cam) + ".avi")                       # read-only
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(rows[0]))
    pos = int(rows[0])
    func = int(d.get("func", 1))
    b470 = np.nanmedian(movie(d, a, a + 15, func), 0)
    base = slice(a, a + 15)
    fig = plt.figure(figsize=(9.6, 5.6), dpi=70)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("k")
    h1 = a1.imshow(b470 * 0, cmap=cmap, vmin=-lim, vmax=lim)
    a1.set_title("470 dF/F (GCaMP), ±%g%%" % lim, fontsize=9)
    h2 = None
    a2.set_title("cam4 (face), DAQ-aligned", fontsize=9)
    for ax_ in (a1, a2):
        ax_.axis("off")
    t = np.arange(a, b) / FS
    a3.plot(t / 60, d["x470"][a:b] - np.median(d["x470"][base]), color="tab:green", lw=0.8, label="470")
    a3.plot(t / 60, d["x415"][a:b] - np.median(d["x415"][base]), color="tab:purple", lw=0.8, label="415")
    for f in d["licks"][(d["licks"] >= a) & (d["licks"] < b)]:
        a3.axvline(f / FS / 60, ymin=0, ymax=0.06, color="k", lw=0.5)
    for f in d["cues"][(d["cues"] >= a) & (d["cues"] < b)]:
        a3.axvline(f / FS / 60, color="tab:orange", lw=0.7, alpha=0.6)
    a3.set_xlabel("session time (min, pair clock)   ticks = licks, orange = cues")
    a3.legend(fontsize=7, loc="upper left")
    cur = a3.axvline(t[0] / 60, color="r", lw=1)
    sup = fig.suptitle("", fontsize=10)
    ev = d["ev"]
    CH = 300
    with imageio.get_writer(path, fps=FS / step, codec="libx264", quality=5, macro_block_size=1) as w:
        for i, p in enumerate(pairs):
            j = (p - a) % CH
            if j == 0:
                m470 = movie(d, p, min(b, p + CH), func) - b470
            while pos < rows[i]:                      # skip (grab is cheap; no decode-to-array)
                cap.grab()
                pos += 1
            ok, fr = cap.read()                       # reads row `pos`
            pos += 1
            if ok:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, (cam_px, int(cam_px * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA)
                if h2 is None:
                    h2 = a2.imshow(g, cmap="gray", vmin=0, vmax=255)
                else:
                    h2.set_data(g)
            h1.set_data(m470[j])
            cur.set_xdata([p / FS / 60] * 2)
            near = np.any((p - ev >= 0) & (p - ev < 8))
            sup.set_text(f"{title}   {p / FS / 60:.3f} min" + ("    ** EVENT **" if near else ""))
            sup.set_color("r" if near else "k")
            fig.canvas.draw()
            w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    cap.release()
    plt.close(fig)
    print("wrote", path, f"{Path(path).stat().st_size / 1e6:.1f} MB", flush=True)


def movie(d, a, b, slot):
    """(n, H/2, W/2) dF/F % for pairs a..b, NaN off-brain."""
    k = d["U"].shape[-1]
    M = np.einsum("hwk,kt->thw", d["U"], np.asarray(d["SVT"][:, 2 * a + slot:2 * b + slot:2], np.float32)) * 100
    M[:, ~d["br"]] = np.nan
    H, W = M.shape[1:]
    M = M[:, :H // 2 * 2, :W // 2 * 2].reshape(len(M), H // 2, 2, W // 2, 2)
    return np.nanmean(M, (2, 4)) if k else M


def render(d, a, b, path, slow=1.0, lim=12.0, title=""):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b470 = np.nanmedian(movie(d, a, a + 15, 1), 0)   # remove the slow offset at the clip start
    b415 = np.nanmedian(movie(d, a, a + 15, 0), 0)
    base = slice(a, a + 15)
    t = np.arange(a, b) / FS
    CH = 300                                         # frames reconstructed per chunk (memory)
    fig = plt.figure(figsize=(9.6, 6.4), dpi=80)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("k")
    h1 = a1.imshow(b470 * 0, cmap=cmap, vmin=-lim, vmax=lim)
    h2 = a2.imshow(b415 * 0, cmap=cmap, vmin=-lim, vmax=lim)
    for ax_, nm in ((a1, "470 (GCaMP)"), (a2, "415 (isosbestic)")):
        ax_.set_title(nm, fontsize=10)
        ax_.axis("off")
    fig.colorbar(h2, ax=[a1, a2], fraction=0.025, label="dF/F %")
    y470 = d["x470"][a:b] - np.median(d["x470"][base])
    y415 = d["x415"][a:b] - np.median(d["x415"][base])
    a3.plot(t / 60, y470, color="tab:green", lw=1, label="470 cortex mean")
    a3.plot(t / 60, y415, color="tab:purple", lw=1, label="415 cortex mean")
    for f in d["licks"][(d["licks"] >= a) & (d["licks"] < b)]:
        a3.axvline(f / FS / 60, ymin=0.0, ymax=0.06, color="k", lw=0.6)
    for f in d["cues"][(d["cues"] >= a) & (d["cues"] < b)]:
        a3.axvline(f / FS / 60, color="tab:orange", lw=0.8, alpha=0.7)
    a3.set_xlabel("session time (min)   ticks = licks, orange = cues")
    a3.set_ylabel("dF/F %")
    a3.legend(fontsize=7, loc="upper left")
    cur = a3.axvline(t[0] / 60, color="r", lw=1)
    sup = fig.suptitle("", fontsize=10)
    fps = FS / slow
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=6, macro_block_size=1,
                            ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
        for i in range(len(t)):
            if i % CH == 0:
                m470 = movie(d, a + i, min(b, a + i + CH), 1) - b470
                m415 = movie(d, a + i, min(b, a + i + CH), 0) - b415
            h1.set_data(m470[i % CH])
            h2.set_data(m415[i % CH])
            cur.set_xdata([t[i] / 60] * 2)
            sup.set_text(f"{title}   t = {t[i] / 60:.3f} min" + (f"   ({slow:g}x slowed)" if slow != 1 else ""))
            fig.canvas.draw()
            w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    print("wrote", path, f"{Path(path).stat().st_size / 1e6:.1f} MB", flush=True)


def traces(d, day, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev, x, y = d["ev"], d["x470"], d["x415"]
    n = len(x)
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.1])
    a0 = fig.add_subplot(gs[0, :])
    a, b = max(0, n - int(20 * 60 * FS)), n
    tt = np.arange(a, b) / FS / 60
    a0.plot(tt, x[a:b], color="tab:green", lw=0.5, label="470")
    a0.plot(tt, y[a:b] - 15, color="tab:purple", lw=0.5, label="415 (offset -15)")
    for f in ev:
        a0.axvline(f / FS / 60, color="r", lw=0.6, alpha=0.6)
    for f in d["cues"][d["cues"] >= a]:
        a0.axvline(f / FS / 60, ymin=0.95, ymax=1, color="tab:orange", lw=0.8)
    for f in d["licks"][d["licks"] >= a]:
        a0.axvline(f / FS / 60, ymin=0, ymax=0.04, color="k", lw=0.4)
    a0.set_title(f"PS122 {day[:2]}-{day[2:]}: last 20 min, cortex-mean raw dF/F (red = detected events, orange top ="
                 f" cues, black bottom = licks; {len(d['licks'][d['licks'] >= a])} licks, "
                 f"{len(d['cues'][d['cues'] >= a])} cues in this window)", fontsize=10)
    a0.set_xlabel("session time (min)")
    a0.legend(fontsize=8, loc="lower left")
    a1 = fig.add_subplot(gs[1, :4])
    c = ev[len(ev) // 3] if len(ev) else n - 500
    w0, w1 = max(0, c - int(5 * FS)), min(n, c + int(25 * FS))
    tz = (np.arange(w0, w1) - c) / FS
    a1.plot(tz, x[w0:w1], color="tab:green", lw=1, label="470")
    a1.plot(tz, y[w0:w1], color="tab:purple", lw=1, label="415")
    for f in ev[(ev >= w0) & (ev < w1)]:
        a1.axvline((f - c) / FS, color="r", lw=0.6, alpha=0.6)
    for f in d["licks"][(d["licks"] >= w0) & (d["licks"] < w1)]:
        a1.axvline((f - c) / FS, ymin=0, ymax=0.05, color="k", lw=0.6)
    a1.set_title(f"30-s zoom from {c / FS / 60:.2f} min", fontsize=10)
    a1.set_xlabel("s")
    a1.legend(fontsize=8)
    a2 = fig.add_subplot(gs[1, 4:])
    lag = np.arange(-31, 63)
    ok = ev[(ev > 31) & (ev < n - 63)]
    for f in ok:
        a2.plot(lag / FS * 1000, x[f + lag] - x[f - 3:f].mean(), color="tab:green", lw=0.4, alpha=0.4)
    if len(ok):
        a2.plot(lag / FS * 1000, np.mean([x[f + lag] - x[f - 3:f].mean() for f in ok], 0), color="k", lw=2,
                label="470 mean")
        a2.plot(lag / FS * 1000, np.mean([y[f + lag] - y[f - 3:f].mean() for f in ok], 0), color="tab:purple", lw=2,
                label="415 mean")
    a2.axvline(0, color="k", lw=0.5)
    a2.set_title(f"every event ({len(ok)}), aligned to peak", fontsize=10)
    a2.set_xlabel("ms")
    a2.legend(fontsize=8)
    if len(ok):
        f = ok[len(ok) // 2]
        M = movie(d, f - 2, f + 11, 1)
        M -= np.nanmean(movie(d, f - 6, f - 2, 1), 0)
        lim = np.nanpercentile(np.abs(M), 99.5)
        for j, k in enumerate((0, 1, 2, 3, 4, 5)):
            idx = [0, 2, 3, 4, 6, 12][k]
            ax_ = fig.add_subplot(gs[2, j])
            ax_.imshow(M[idx], cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax_.set_title(f"{(idx - 2) / FS * 1000:+.0f} ms (470, ±{lim:.0f}%)", fontsize=9)
            ax_.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)
    print("wrote", path, flush=True)


def main():
    from wfield_local.writeguard import assert_writable
    assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    d = load("1029")
    traces(d, "1029", OUT / "PS122_1029_event_traces.png")
    if "--cam-only" in sys.argv:
        render_cam(d, int(113.2 * 60 * FS), int(116.3 * 60 * FS), OUT / "PS122_1029_events_cam4.mp4")
        return
    a = int(113.2 * 60 * FS)
    render(d, a, int(116.3 * 60 * FS), OUT / "PS122_1029_events_realtime.mp4",
           title="PS122 10-29 (RiboL1-8s)")
    f = d["ev"][len(d["ev"]) // 2]
    render(d, f - int(FS), f + int(2 * FS), OUT / "PS122_1029_event_slowmo.mp4", slow=5,
           title="PS122 10-29 one event")
    d2 = load("1030")
    traces(d2, "1030", OUT / "PS122_1030_event_traces.png")


if __name__ == "__main__":
    main()

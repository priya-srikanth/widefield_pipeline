"""Pre-stroke PS92 / PS93 videos over QUIET late-session windows, to compare with the post-stroke events (2026-10-31).

Priya 10-31, watching the post-stroke event videos: PS93 08-20 / 08-21 look like "rapid mode switching of activation
of orofacial cortex and activation of limb (sensory) cortex, but while the animal is just sitting still and not
licking"; "can i see some pre-stroke 92 and 93 videos to see if the mode switching is there while quiet too?"

Window choice: 3 min, starting at >= 40 min, scored by the running fraction (canonical DAQ running bouts,
behavior_events npz) plus licks per minute / 60; the lowest score wins, ties to the later window. Same renderer and
color scale as the post-stroke videos (seizure_events_video.render_cam, raw 470, +/-12%), plus running bouts shaded
blue on the trace. Cohort1 was never swap-repaired before its SVD, so one-frame flashes are swapped frames.

    conda activate locanmf
    python scripts/cohort2_qc/prestroke_quiet_videos.py [PS92_0811 ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from seizure_events_video import FS, OUT, render_cam  # noqa: E402

CAMS = Path("M:/MICROSCOPE/Priya/Behavior_Cameras/Widefield")
EVENTS = Path("M:/MICROSCOPE/Priya/Behavior_logs/Widefield/behavior_summary/events")
DEFAULT = ["PS92_0811", "PS92_0814", "PS93_0811", "PS93_0814"]   # PS92 08-12 crash+concat, PS93 08-13 no video
WIN_MIN, START_MIN = 3.0, 40.0


def load(label):
    from wfield_local import config
    from wfield_local import hemo_variants as hv
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    s = next(s for s in config.load_sessions(machine="analysis_desktop") if s["label"] == label)
    res, func = Path(s["mc"]) / "wfield_local_results", int(hv.functional_channel(s))
    fa = np.load(res / "frames_average.npy")
    br = brain_mask(fa, func, res=res)
    U = np.nan_to_num(np.load(res / "U.npy")).astype(np.float32)
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    um = U.reshape(-1, U.shape[-1])[br.reshape(-1)].mean(0)
    x = 100 * (um @ np.asarray(SVT[:, func::2], np.float64))
    y = 100 * (um @ np.asarray(SVT[:, 1 - func::2], np.float64))
    n = min(len(x), len(y))
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog")
    cf, lf, csmp = _frames(s, cue, lk)
    csmp = np.asarray(csmp)
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    run = np.searchsorted(csmp, np.c_[z["running_starts"], z["running_stops"]])   # DAQ samples -> pairs
    return dict(U=U, SVT=SVT, br=br, x470=x[:n], x415=y[:n], func=func, ev=np.array([], int), n=n,
                licks=np.asarray(lf)[np.asarray(lf) >= 0], cues=np.asarray(cf)[np.asarray(cf) >= 0],
                csmp=csmp, h5=s["h5"], run=[tuple(r) for r in run if r[1] > r[0]])


def quietest(d):
    n, w = d["n"], int(WIN_MIN * 60 * FS)
    running = np.zeros(n, bool)
    for r0, r1 in d["run"]:
        running[r0:r1] = True
    lick = np.bincount(d["licks"][d["licks"] < n], minlength=n)
    best = None
    for a in range(int(START_MIN * 60 * FS), n - w - 70, int(15 * FS)):
        score = running[a:a + w].mean() + lick[a:a + w].sum() / WIN_MIN / 60
        if best is None or score <= best[0]:
            best = (score, a, running[a:a + w].mean(), lick[a:a + w].sum())
    return best


def main():
    for label in sys.argv[1:] or DEFAULT:
        animal, mmdd = label.split("_")
        date = f"2026{mmdd}"
        z = np.load(CAMS / "alignment_templates" / "cam4" / animal / f"{date}.npz", allow_pickle=True)
        tpl = {k: (z[k][()] if z[k].ndim == 0 else z[k]) for k in z.files}
        cams = sorted((CAMS / date / animal).glob("cam4_*.avi"))
        rec = str(tpl.get("recording", ""))
        cam = next((c for c in cams if rec and rec in c.name), cams[0])
        d = load(label)
        score, a, frun, nlick = quietest(d)
        b = a + int(WIN_MIN * 60 * FS)
        print(f"{label}: window {a / FS / 60:.1f}-{b / FS / 60:.1f} min of {d['n'] / FS / 60:.0f}, running "
              f"{100 * frun:.0f}%, {nlick} licks; cam {cam.name}", flush=True)
        render_cam(d, a, b, OUT / f"{label}_prestroke_quiet_cam4.mp4", cam=cam.with_suffix(""), tpl=tpl,
                   title=f"{animal} {mmdd[:2]}-{mmdd[2:]} PRE-stroke, quiet (Thy1-6s)")


if __name__ == "__main__":
    main()

"""Combined video (470 map | cam4 face | traces) for the epileptiform candidates beyond PS122 (2026-10-31).

Priya 10-31: "do any other recordings have this, and if so can you render them as well"; "real epileptiform events
should have the narrow spike like ps 122, though 6s could be slower". Candidates = epileptiform_scan.json events
with amp >= 5%, half-width 60-250 ms and 415 STRICTLY flat (|415| < 0.15 x 470 at the peak; the looser 0.35 let
through vessel-centred events where 415 rises with 470, epileptiform_gallery.png). One 3-min window per session,
over its densest cluster. Cohort1 cam4 templates are the saved ones (alignment_templates/cam4/<animal>/<date>.npz).
Same renderer as PS122 (seizure_events_video.render_cam).

    conda activate locanmf
    python scripts/cohort2_qc/epileptiform_videos.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from seizure_events_video import FS, OUT, render_cam  # noqa: E402

SCAN = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261031/epileptiform_scan")
CAMS = Path("M:/MICROSCOPE/Priya/Behavior_Cameras/Widefield")
WINDOWS = {"PS93_0820": (86.0, 89.0), "PS93_0821": (92.4, 95.4), "PS92_0828": (77.4, 80.4),
           "PS94_0823": (86.9, 89.9), "PS94_0825": (78.4, 81.4)}


def strict_flat(r):
    a, w, e = (np.array(r[k]) for k in ("amp", "halfwidth_ms", "amp415"))
    return np.array(r["pairs"], int)[(a >= 5) & (w >= 60) & (w <= 250) & (np.abs(e) < 0.15 * a)]


def load(label, r):
    from wfield_local import config
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    s = next(s for s in config.load_sessions(machine="analysis_desktop") if s["label"] == label)
    res, func = Path(r["res"]), int(r["func"])
    fa = np.load(res / "frames_average.npy")
    br = brain_mask(fa, func, res=res)
    U = np.nan_to_num(np.load(res / "U.npy")).astype(np.float32)
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    um = U.reshape(-1, U.shape[-1])[br.reshape(-1)].mean(0)
    x = 100 * (um @ np.asarray(SVT[:, func::2], np.float64))
    y = 100 * (um @ np.asarray(SVT[:, 1 - func::2], np.float64))
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, lf, csmp = _frames(s, cue, lk)
    return dict(U=U, SVT=SVT, br=br, x470=x, x415=y, func=func, ev=strict_flat(r), licks=np.asarray(lf)[np.asarray(lf) >= 0],
                cues=np.asarray(cf)[np.asarray(cf) >= 0], csmp=np.asarray(csmp), h5=s["h5"])


def main():
    R = {r["label"]: r for r in json.loads((SCAN / "epileptiform_scan.json").read_text())}
    labels = sys.argv[1:] or list(WINDOWS)
    for label in labels:
        animal, mmdd = label.split("_")
        date = f"2026{mmdd}"
        tpl_p = CAMS / "alignment_templates" / "cam4" / animal / f"{date}.npz"
        z = np.load(tpl_p, allow_pickle=True)
        tpl = {k: (z[k][()] if z[k].ndim == 0 else z[k]) for k in z.files}
        cams = sorted((CAMS / date / animal).glob("cam4_*.avi"))
        rec = str(tpl.get("recording", ""))
        cam = next((c for c in cams if rec and rec in c.name), cams[0])
        print(f"{label}: cam {cam.name}, template {tpl_p.name} (recording {rec})", flush=True)
        d = load(label, R[label])
        t0, t1 = WINDOWS[label]
        render_cam(d, int(t0 * 60 * FS), int(t1 * 60 * FS), OUT / f"{label}_events_cam4.mp4",
                   cam=cam.with_suffix(""), tpl=tpl, title=f"{animal} {mmdd[:2]}-{mmdd[2:]} (Thy1-6s)")


if __name__ == "__main__":
    main()

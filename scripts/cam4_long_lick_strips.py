"""Frame strips for `cam4_long_lick_qc.py` cases (11-22): per case, the 3-D tongue distance from the mouth zero over the
trial's licks, with the cam4 windows (bars), fused licks, 3-D lick events and DAQ contacts, and a row of cam4 frames
across the flagged window. Read-only; writes PNGs into the QC folder.

    python scripts/cam4_long_lick_strips.py SESSION:TRIAL:T_MS [...]      (T_MS = the cam4 lick peak, ms from cue)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cam4_long_lick_qc import OUT


def strip(session: str, trial: int, t_ms: float, note: str = "") -> Path:
    from wfield_local import kinematics_chain as KC
    from wfield_local import lp_licks
    from wfield_local import o2_inference as oi
    from wfield_local import orofacial_outputs as oo
    from wfield_local import session_licks as SL
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    from wfield_local.pose3d import pose3d_dir
    rv = PathResolver()
    an, date = session.split("_")
    F = pd.read_csv(pose3d_dir() / f"{an}_{date}_frames.csv.gz",
                    usecols=["trial_id", "t_ms", "src4", "tongue_ml", "tongue_ap", "tongue_dv"])
    F = F[F.trial_id == trial]
    C = pd.read_csv(oo.session_dir(rv, lp_licks.input_run(), an, date) / "licks.csv")
    C = C[C.trial_id == trial]
    E = pd.read_csv(pose3d_dir() / f"{an}_{date}_licks.csv")
    E = E[E.trial_id == trial]
    U = SL.session_licks(session, rv)
    U = U[U.trial_id == trial]
    cont = KC.daq_contacts_ms(an, date, TW.trial_bounds(an, date, rv), rv).get(trial, np.zeros(0))
    w = C.loc[(C.t_ms - t_ms).abs().idxmin()]
    t0, t1 = w.on_ms - 150, w.off_ms + 150
    d = np.linalg.norm(F[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float), axis=1)
    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_axes([0.06, 0.55, 0.9, 0.36])
    ax.plot(F.t_ms, d, "k-", lw=1)
    for r in C.itertuples():
        hl = abs(r.t_ms - w.t_ms) < 1
        ax.axvspan(r.on_ms, r.off_ms, ymin=0.92, ymax=0.99, color="tab:red" if hl else "tab:blue", alpha=0.6)
    for r in U.itertuples():
        ax.axvspan(r.on_ms, r.off_ms, ymin=0.83, ymax=0.9, color="tab:green", alpha=0.5)
    for r in E.itertuples():
        ax.plot(r.t_peak_ms, np.nanmax(d) * 1.02 if np.isfinite(d).any() else 1, "v", color="tab:orange")
    for c in cont:
        ax.axvline(c, color="tab:purple", lw=0.8, ls="--")
    ax.set_xlim(t0 - 200, t1 + 200)
    ax.set_xlabel("ms from cue")
    ax.set_ylabel("3-D tongue distance (mm)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{session} trial {trial} {w.position}: cam4 window {w.on_ms:.0f}-{w.off_ms:.0f} ms (red bar). "
                 "Bars: cam4 (blue/red), fused (green); orange = 3-D lick peaks; purple dashes = DAQ contacts"
                 + (f"\n{note}" if note else ""), fontsize=9)
    vid = str(oi.videos_for(rv, an, date, "cam4")[0])
    cap = cv2.VideoCapture(vid)
    tt = np.linspace(w.on_ms, w.off_ms, 8)
    for i, t in enumerate(tt):
        k = int(np.argmin(np.abs(F.t_ms.to_numpy(float) - t)))
        fr = F.src4.iloc[k]
        a2 = fig.add_axes([0.02 + i * 0.122, 0.04, 0.115, 0.4])
        a2.axis("off")
        if np.isfinite(fr):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
            ok, im = cap.read()
            if ok:
                a2.imshow(im[..., ::-1])
        a2.set_title(f"{F.t_ms.iloc[k]:.0f} ms", fontsize=8)
    cap.release()
    out = OUT / "strips"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{session}_t{trial}_{int(w.t_ms)}.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    return f


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        s, tr, t, *note = spec.split(":")
        print(strip(s, int(tr), float(t), ":".join(note)))

"""Visual QC of every tongue stage on our data, before retuning a threshold: raw -> clean -> v7 pre-clean ->
detectors -> gates -> kept licks, plus the video frame at every rejected peak.

    python -m scripts.tongue_qc_steps --model DLC      # or LP (with --lp <csv>)

Priya, 2026-10-01: "let's visually qc each step on our data". PS93 0908 cue clip (24 trials). Tongue y is px from
the MOUTH (spout-frame origin), y grows downward = tongue out. Per selected trial, three rows:
  1  raw tongue y (black = likelihood > 0.6, grey = below) and the v5p3 clean trace (blue)
  2  v7 pre-clean: cleaned trace; shaded = raw clusters the pre-clean labelled artifact (red) / keep_low (green)
  3  peaks: every merged peak, marker = detector (o lmax, s bounded, ^ legacy); green = kept, red X = rejected
     with its gate letter; grey band = the response window (cue + 70 ms .. trial stop); dashed lines = the current
     px thresholds (min peak 20, shape_low 80, high 130) in mouth-relative px, i.e. what a retune would move.
Trials shown: all with a rejected peak or an artifact cluster, then fill to `--n` with ordinary ones.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import orofacial_clean as oc
from wfield_local import spout_frame as SF
from wfield_local import trial_windows as TW

MARK = {"lmax": "o", "bounded": "s", "legacy": "^"}


def main(argv=None) -> int:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import scripts.pose_kinematics_demo as D
    from wfield_local import dlc_project
    from wfield_local.paths import PathResolver

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["DLC", "LP"], default="DLC")
    ap.add_argument("--lp", type=Path, default=Path.home() / "lp_cue_tmp" / "cue_windows_LP.csv")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args(argv)
    rv = PathResolver()
    d = dlc_project.project_dir(rv).parent / "lp_vs_dlc_cue_traces" / "PS93_20260908"
    idx = pd.read_csv(d / "cue_windows_index.csv")
    dlc = oc.read_pose(d / "cue_windows_DLC.csv")
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    frame = SF.from_medians(SF.position_medians(dlc[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    b = TW.trial_bounds("PS93", "20260908", rv)
    pose = d / "cue_windows_DLC.csv" if a.model == "DLC" else a.lp
    T, _J, _M, c = D.run(pose, idx, frame, dict(zip(b.trial_id, b.stop_s - b.cue_s)))
    tg, pos_map, _cues = c["tongue"]
    p = T.params
    clip_row = np.full(len(tg.y_final), -1)
    clip_row[pos_map] = np.arange(len(pos_map))                       # padded index -> clip video frame

    res = T.trial_results
    bad = [k for k, r in enumerate(res) if any(not x["keep"] for x in r.decisions)
           or any(cl.label.startswith("artifact") for cl in r.preclean_diag.clusters)]
    show = bad + [k for k in range(len(res)) if k not in bad][: max(0, a.n - len(bad))]
    fig, axs = plt.subplots(3, len(show), figsize=(3.4 * len(show), 9), sharex="col", squeeze=False)
    thumbs = []
    for col, k in enumerate(show):
        r, tr = res[k], T.per_trial.iloc[k]
        lo = r.session_frame_lo
        sl = slice(lo, r.session_frame_hi + 1)
        t = r.t_ms
        y_raw = tg.y_raw[sl] - tg.Y0
        ok = tg.lk[sl] > 0.6
        ax = axs[0, col]
        ax.plot(t[~ok], y_raw[~ok], ".", ms=1.5, color="0.75")
        ax.plot(t[ok], y_raw[ok], ".", ms=1.5, color="k")
        ax.plot(t, tg.y_masked[sl], "-", lw=0.8, color="tab:blue")
        ax.set_title(f"trial {tr.trial_id} | {tr.position}\nstop {tr.response_end_ms:.0f} ms", fontsize=8)
        ax = axs[1, col]
        ax.plot(t, np.where(r.is_baseline_fill_clean, np.nan, r.y_clean), "-", lw=0.8, color="tab:purple")
        for cl in r.preclean_diag.clusters:
            colr = "red" if cl.label.startswith("artifact") else ("green" if cl.label == "keep_low" else None)
            if colr:
                ax.axvspan(t[cl.lo], t[min(cl.hi, len(t) - 1)], color=colr, alpha=0.25 if colr == "red" else 0.08, lw=0)
        ax = axs[2, col]
        ax.plot(t, np.where(r.is_baseline_fill_clean, np.nan, r.y_clean), "-", lw=0.6, color="0.5")
        ax.axvspan(70, tr.response_end_ms, color="0.85", alpha=0.5, lw=0)
        for x in r.decisions:
            if x["keep"]:
                ax.plot(x["t_ms"], x["y"], MARK.get(x["src"], "o"), ms=4, mfc="none", mec="green")
            else:
                ax.plot(x["t_ms"], x["y"], "X", ms=8, color="red")
                ax.text(x["t_ms"], x["y"] + 8, x["gate"], color="red", fontsize=8, ha="center")
                f_clip = clip_row[lo + int(x["peak_frame_local"])]
                thumbs.append((f_clip, f"trial {tr.trial_id} {x['t_ms']:.0f} ms\nREJECTED {x['reason'][:28]}"))
        for yv, _lab in ((p["detector"]["min_peak_y_abs"], "min"), (p["preclean"]["shape_low_thr_px"], "low"),
                        (p["preclean"]["high_thr_px"], "high")):
            ax.axhline(yv, ls="--", lw=0.6, color="tab:orange")
        for row in range(3):
            axs[row, col].axvline(0, color="k", lw=0.6)
            axs[row, col].set_ylim(-20, 260)
        axs[2, col].set_xlabel("ms from cue")
    for row, lab in enumerate(("raw + v5p3 clean", "v7 pre-clean", "peaks + gates")):
        axs[row, 0].set_ylabel(f"{lab}\ntongue y from mouth (px)", fontsize=8)
    fig.suptitle(f"PS93 0908, {a.model}: tongue stages on our data (thresholds dashed: min 20 / shape_low 80 / high 130, "
                 f"old-rig values in mouth-relative px)", fontsize=10)
    fig.tight_layout()
    out = d / f"tongue_qc_steps_{a.model}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"-> {out}  ({len(bad)} trials with rejections/artifacts)")

    if thumbs:
        cap = cv2.VideoCapture(str(d / "cue_windows.mp4"))
        n = len(thumbs)
        fig2, ax2 = plt.subplots(1, n, figsize=(2.6 * n, 3.2), squeeze=False)
        for j, (f, lab) in enumerate(thumbs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            okf, im = cap.read()
            ax2[0, j].axis("off")
            if okf:
                ax2[0, j].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)[250:640, 170:510])
                ax2[0, j].plot(frame.origin[0] - 170, frame.origin[1] - 250, "*", ms=8, mfc="yellow", mec="k")
            ax2[0, j].set_title(lab, fontsize=6)
        cap.release()
        fig2.suptitle(f"{a.model}: video frame at every REJECTED peak (star = mouth)", fontsize=9)
        fig2.tight_layout()
        out2 = d / f"tongue_qc_rejected_frames_{a.model}.png"
        fig2.savefig(out2, dpi=140, bbox_inches="tight")
        print(f"-> {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

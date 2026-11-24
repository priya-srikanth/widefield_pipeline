"""3-D keypoint JUMP QC on the production four-camera run (Priya 11-22: "in the mouse 9 pre-stroke clip example, the
nose point jumps with '3d' interpolation from the cam1 view - do these nose jumps happen a lot?").

Read-only on `session_poses/pose3d_4cam_3net_20261121_sync60/<a>_<d>_frames.csv.gz` (+ `_licks.csv`). Per session and
part (nose, jaw, tongue), on consecutive frames of one trial (dt = 4 ms):
  step   = |X_k - X_{k-1}| (mm, 3-D)
  jump   = step > THR (0.5 and 1.0 mm)
  spike  = a jump at k that comes BACK: some m in 1..3 with |X_{k+m} - X_{k-1}| < THR / 2 (an excursion of <= 3 frames)
split by the number of views behind the point at k (`<part>_views`) and by whether the SET of views used (cameras whose
`<part>_rep_<cam>` is defined) changed between k-1 and k. Also: trials with >= 1 spike > 1 mm, and 3-D licks (from
`_licks.csv`, onset .. onset + 4 (n_frames - 1) ms) whose window holds a spike frame of that part.

    python scripts/pose3d_jump_qc.py --jobs 3        # -> <P3D>/qc_jumps_20261122/{steps_by_views.csv, sessions.csv}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60")
OUT = P3D / "qc_jumps_20261122"
PARTS = ("nose", "jaw", "tongue")
CAMS = ("cam1", "cam4", "cam2", "cam3")
THRS = (0.5, 1.0)


def session_qc(label: str) -> dict:
    """Module level (fan_sessions). label = ANIMAL_YYYYMMDD."""
    cols = ["trial_id", "t_ms"] + [f"{p}_{a}" for p in PARTS for a in ("ml", "ap", "dv", "views")] + \
        [f"{p}_rep_{c}" for p in PARTS for c in CAMS]
    F = pd.read_csv(P3D / f"{label}_frames.csv.gz", usecols=cols)
    L = pd.read_csv(P3D / f"{label}_licks.csv", usecols=["trial_id", "t_onset_ms", "n_frames"])
    tid, t = F.trial_id.to_numpy(), F.t_ms.to_numpy(float)
    cons = np.r_[False, (tid[1:] == tid[:-1]) & (np.abs(np.diff(t) - 4.0) < 0.5)]
    rows, sess = [], {"session": label}
    for p in PARTS:
        X = F[[f"{p}_ml", f"{p}_ap", f"{p}_dv"]].to_numpy(float)
        nv = F[f"{p}_views"].to_numpy()
        code = np.zeros(len(F), int)
        for b, c in enumerate(CAMS):
            code |= np.isfinite(F[f"{p}_rep_{c}"].to_numpy(float)).astype(int) << b
        step = np.full(len(F), np.nan)
        step[1:] = np.linalg.norm(X[1:] - X[:-1], axis=1)
        ok = cons & np.isfinite(step)
        changed = np.r_[False, code[1:] != code[:-1]]
        for thr in THRS:
            jump = ok & (step > thr)
            spike = np.zeros(len(F), bool)
            for k in np.flatnonzero(jump):
                for m in (1, 2, 3):
                    if k + m < len(F) and tid[k + m] == tid[k] and np.isfinite(X[k + m]).all() and \
                            np.linalg.norm(X[k + m] - X[k - 1]) < thr / 2:
                        spike[k] = True
                        break
            sess[f"{p}_steps"] = int(ok.sum())
            sess[f"{p}_jump{thr}"] = int(jump.sum())
            sess[f"{p}_spike{thr}"] = int(spike.sum())
            if thr == 1.0:
                tr = pd.Series(spike).groupby(tid).any()
                sess[f"{p}_trials"] = int(len(tr))
                sess[f"{p}_trials_spike1"] = int(tr.sum())
                hit = 0
                sp_t = pd.DataFrame({"tid": tid[spike], "t": t[spike]})
                for tr_, g in L.groupby("trial_id"):
                    s = sp_t[sp_t.tid == tr_].t.to_numpy()
                    if len(s):
                        a = g.t_onset_ms.to_numpy(float)
                        b = a + 4.0 * (g.n_frames.to_numpy() - 1)
                        hit += int(((s[None, :] >= a[:, None]) & (s[None, :] <= b[:, None])).any(1).sum())
                sess[f"{p}_licks"] = int(len(L))
                sess[f"{p}_licks_spike1"] = hit
            D = pd.DataFrame({"nviews": nv[ok], "changed": changed[ok], "jump": jump[ok], "spike": spike[ok]})
            g = D.groupby(["nviews", "changed"]).agg(steps=("jump", "size"), jumps=("jump", "sum"),
                                                     spikes=("spike", "sum")).reset_index()
            rows.append(g.assign(session=label, part=p, thr=thr))
    return {"sess": sess, "by": pd.concat(rows, ignore_index=True)}


def strips(label: str, part: str = "nose", n: int = 3, thr: float = 1.0) -> list:
    """Trace + frame strips of ``n`` spikes (seeded pick) of ``part`` in one session: 3-D coordinates +-60 ms with the
    number of views and the view set, and the raw cam1 / cam4 frames at k-1, k, k+1 (the side views are not drawn:
    their frame index is not in the frames table)."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d_side_video import Reader
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    cols = ["trial_id", "t_ms", "src1", "src4"] + [f"{part}_{a}" for a in ("ml", "ap", "dv", "views")] + \
        [f"{part}_rep_{c}" for c in CAMS]
    F = pd.read_csv(P3D / f"{label}_frames.csv.gz", usecols=lambda c: c in cols)
    X = F[[f"{part}_ml", f"{part}_ap", f"{part}_dv"]].to_numpy(float)
    tid, t = F.trial_id.to_numpy(), F.t_ms.to_numpy(float)
    st = np.r_[np.nan, np.linalg.norm(np.diff(X, axis=0), axis=1)]
    cons = np.r_[False, (tid[1:] == tid[:-1]) & (np.abs(np.diff(t) - 4.0) < 0.5)]
    ks = [k for k in np.flatnonzero(cons & (st > thr)) if any(
        k + m < len(F) and tid[k + m] == tid[k] and np.linalg.norm(X[k + m] - X[k - 1]) < thr / 2 for m in (1, 2, 3))]
    rng = np.random.default_rng(20261122)
    ks = [ks[i] for i in sorted(rng.choice(len(ks), min(n, len(ks)), replace=False))] if ks else []
    rv = PathResolver()
    a_, d_ = label.split("_")
    vid = {"cam1": oi.videos_for(rv, a_, d_, "cam1")[0], "cam4": oi.videos_for(rv, a_, d_, "cam4")[0]}
    out = []
    for k in ks:
        w = np.flatnonzero((tid == tid[k]) & (np.abs(t - t[k]) <= 60))
        fig = plt.figure(figsize=(11, 5.2))
        ax = fig.add_axes([0.06, 0.58, 0.55, 0.36])
        for i, (c, lab) in enumerate(zip(("C0", "C1", "C2"), ("ml", "ap", "dv"))):
            ax.plot(t[w] - t[k], X[w, i] - np.nanmedian(X[w, i]), ".-", color=c, label=lab, ms=3)
        ax.axvline(0, color="r", lw=0.6)
        ax.set_xlabel("ms from the jump frame")
        ax.set_ylabel(f"3-D {part} - median (mm)")
        ax.legend(fontsize=7, ncol=3)
        vs = ["+".join(c[-1] for c in CAMS if np.isfinite(F[f"{part}_rep_{c}"].iloc[j])) for j in w]
        ax.set_title(f"{label} trial {tid[k]} t = {t[k]:+.0f} ms: step {st[k]:.2f} mm; views (cam digits) per frame "
                     f"around the jump: {' '.join(vs[max(0, list(w).index(k) - 3):list(w).index(k) + 4])}", fontsize=7)
        for r, c in enumerate(("cam1", "cam4")):
            rd = Reader(vid[c])
            for q, kk in enumerate((k - 1, k, k + 1)):
                s = F[f"src{c[-1]}"].iloc[kk]
                im = rd.get(int(s)) if np.isfinite(s) else np.zeros((600, 600, 3), np.uint8)
                axi = fig.add_axes([0.63 + q * 0.12, 0.52 - r * 0.47, 0.115, 0.44])
                axi.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                axi.set_title(f"{c} {['k-1', 'k', 'k+1'][q]} (views {int(F[f'{part}_views'].iloc[kk])})", fontsize=6)
                axi.axis("off")
        ax2 = fig.add_axes([0.06, 0.08, 0.55, 0.36])
        ax2.plot(t[w] - t[k], F[f"{part}_views"].to_numpy()[w], ".-k", ms=3)
        ax2.set_ylabel("n views")
        ax2.set_xlabel("ms from the jump frame")
        f = OUT / f"strip_{part}_{label}_t{tid[k]}_{t[k]:+.0f}ms.png"
        fig.savefig(f, dpi=110)
        plt.close(fig)
        out.append(str(f))
    return out


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--sessions", nargs="*", default=None)
    a = ap.parse_args(argv)
    labs = a.sessions or sorted(f.name.replace("_frames.csv.gz", "") for f in P3D.glob("*_frames.csv.gz"))
    OUT.mkdir(exist_ok=True)
    res, fails = ak.fan_sessions(labs, session_qc, jobs=a.jobs)
    S = pd.DataFrame([v["sess"] for _, v in res])
    S["animal"] = S.session.str[:4]
    S["epoch"] = [epochs.epoch_of(f"{s[:4]}_{s[9:]}") or "none" for s in S.session]
    B = pd.concat([v["by"] for _, v in res], ignore_index=True)
    B = B.merge(S[["session", "animal", "epoch"]], on="session")
    S.to_csv(OUT / "sessions.csv", index=False)
    B.to_csv(OUT / "steps_by_views.csv", index=False)
    for it, err in fails:
        print("FAILED", it, err)
    print(len(S), "sessions ->", OUT)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

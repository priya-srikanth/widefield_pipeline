"""ILI FLOOR FOR FOUR-CAMERA 3-D LICKS (11-21, read-only diagnostic for Priya): is a 40-80 ms peak-to-peak pair of 3-D
licks ONE protrusion split in two, or TWO protrusions? Decides the `pose3d.lick_peaks` ``min_ili_ms`` floor (40 ms
applied, the pipeline-wide `lick_detection.min_ili_ms`; 60-80 ms proposed because cam4's own 2-D detector has no
interval below 60 ms).

Inputs (read only): the four-camera 3-D run ``session_poses/pose3d_4cam_3net_20261121`` (``<s>_frames.csv.gz``,
``<s>_licks.csv`` = `lick_peaks(S, rig, cuts=camera_licks.event_cuts(...), min_ili_ms=40)`; re-running that call on the
saved frames reproduces the saved licks: PS93 0814, 5,624 = 5,624 licks, same frame counts, peak times within the
3-decimal rounding of the saved table) and the fused camera-lick bundle
``session_poses/camera_licks/lp_side_r2b_prod_20261118_3net_20261121/<s>_events.csv`` (cam4 v37 lick windows, cam1
v37 events, cam2 / cam3 side events with the ``static`` / ``far`` guards).

Per consecutive 3-D lick pair of one trial (peak-to-peak ``dt`` < 400 ms):
  * reach r = |tip - mouth zero| (mm) at both extension peaks, and ``r_min`` = the smallest r over the 3-D frames
    STRICTLY between the peaks; ``retr_frac`` = (min(r1, r2) - r_min) / min(r1, r2) (0 = no return toward the mouth,
    1 = back to the mouth zero); NaN when no frame between the peaks has a 3-D tongue;
  * ``gap_ms`` = the longest run of frames without a 3-D tongue between the peaks (the tongue may be inside the mouth);
    ``split`` = "gap" (the two licks are separate runs of frames) or "cut" (one unbroken run split by `event_cuts`);
  * per camera (cam4, cam1, cam2, cam3; side events with a guard flag excluded): the event overlapping each lick's
    frame span most; ``<cam>_call`` = "two" (different events), "one" (the same event), "only1" (one lick has an
    event), "none".
  * ``max_step_mm`` = the largest |change of r| between consecutive tracked frames (4 ms) from peak to peak (a
    physically implausible jump = a tracking glitch); ``n_retr25`` = frames between the peaks >= 25 % of the smaller
    peak reach back toward the mouth; ``edge1`` / ``edge2`` = that lick's peak is on the first / last frame of its run
    (a monotonic piece: no extension peak inside it);
  * lick counts per session under floors 40 / 60 / 80 ms (`lick_peaks` re-run on the saved frames, same cuts).

    dlc env, PYTHONPATH=<worktree>, MKL_THREADING_LAYER=SEQUENTIAL:
    python scripts/pose3d_ili_floor_examples.py scan       # all sessions -> pairs.csv.gz, floors.csv
    python scripts/pose3d_ili_floor_examples.py examples   # August pre + post sample sheets + summary figure
    python scripts/pose3d_ili_floor_examples.py frames     # video strips (cam4 + cam1, LP points) for clear cases
-> session_poses/pose3d_4cam_3net_20261121/qc_phaseA/ili_floor_examples/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses")
RUN = ROOT / "pose3d_4cam_3net_20261121"
EVENTS = ROOT / "camera_licks" / "lp_side_r2b_prod_20261118_3net_20261121"
OUT = RUN / "qc_phaseA" / "ili_floor_examples"
COLS = ["trial_id", "position", "t_ms", "stop_ms", "src1", "src4", "tongue_ml", "tongue_ap", "tongue_dv",
        "tongue_views", "tongue_src", "tongue_rep_cam1", "tongue_rep_cam4", "tongue_rep_cam2", "tongue_rep_cam3",
        "tongue_rep1", "tongue_rep4"]
CAMS = ("cam4", "cam1", "cam2", "cam3")
BINS = {"40-60": (40, 60), "60-80": (60, 80), "120-200": (120, 200)}
FLOORS = (40, 60, 80)
FRAME_MS = 4.0


def sessions() -> list[str]:
    return sorted(p.name[:-10] for p in RUN.glob("*_licks.csv"))


def load_frames(sid: str) -> pd.DataFrame:
    return pd.read_csv(RUN / f"{sid}_frames.csv.gz", usecols=COLS, dtype={"position": str})


def load_events(sid: str) -> dict:
    E = pd.read_csv(EVENTS / f"{sid}_events.csv", low_memory=False)
    out = {}
    for c, g in E.groupby("cam"):
        g = g.reset_index(drop=True)
        for f in ("static", "far"):
            if f in g:
                g[f] = g[f].astype("boolean").fillna(False).astype(bool)
        out[c] = g
    return out


def epoch(sid: str) -> str:
    from wfield_local import epochs
    a, d = sid.split("_")
    e = epochs.epoch_of(f"{a}_{d[4:]}")
    return e or "none"


def _trial_index(S: pd.DataFrame) -> dict:
    tid = S.trial_id.to_numpy()
    b = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1], True])
    return {int(tid[x]): (x, y) for x, y in zip(b[:-1], b[1:])}


def _best_event(E: pd.DataFrame | None, s0: float, s1: float) -> int:
    """Index of the event of ``E`` (one trial) overlapping [s0, s1] most (overlap >= -1 frame), else -1."""
    if E is None or not len(E):
        return -1
    ov = np.minimum(E.off_ms.to_numpy(float), s1) - np.maximum(E.on_ms.to_numpy(float), s0)
    k = int(np.argmax(ov))
    return int(E.index[k]) if ov[k] >= -FRAME_MS else -1


def _call(b1: int, b2: int) -> str:
    if b1 >= 0 and b2 >= 0:
        return "two" if b1 != b2 else "one"
    return "only1" if (b1 >= 0) != (b2 >= 0) else "none"


def _usable(E: pd.DataFrame) -> pd.DataFrame:
    g = np.zeros(len(E), bool)
    for f in ("static", "far"):
        if f in E:
            g |= E[f].to_numpy(bool)
    return E[~g]


def pair_metrics(sid: str, S: pd.DataFrame, L: pd.DataFrame, ev: dict, cuts: dict, max_dt: float = 400.0):
    P = S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    r = np.linalg.norm(P, axis=1)                          # NaN where no 3-D tongue
    t = S.t_ms.to_numpy(float)
    ti = _trial_index(S)
    evt = {c: {int(k): g for k, g in (_usable(ev[c]) if c in ("cam2", "cam3") else ev[c]).groupby("trial_id")}
           for c in CAMS if c in ev}
    rows = []
    L = L.sort_values(["trial_id", "t_peak_ms"]).reset_index(drop=True)
    for tr, g in L.groupby("trial_id"):
        if len(g) < 2 or int(tr) not in ti:
            continue
        x, y = ti[int(tr)]
        tt, rr, ml = t[x:y], r[x:y], P[x:y, 0]
        on, pk, nf = g.t_onset_ms.to_numpy(float), g.t_peak_ms.to_numpy(float), g.n_frames.to_numpy(int)
        sv = g.share_single_view.to_numpy(float)
        end = on + (nf - 1) * FRAME_MS
        c_tr = np.asarray(cuts.get(int(tr), []), float)
        for i in range(len(g) - 1):
            dt = pk[i + 1] - pk[i]
            if not (0 < dt < max_dt):
                continue
            k1, k2 = np.searchsorted(tt, pk[i]), np.searchsorted(tt, pk[i + 1])
            if k2 >= len(tt) or k1 >= k2:
                continue
            r1, r2 = rr[k1], rr[k2]
            btw = rr[k1 + 1:k2]
            have = np.isfinite(btw)
            rmin = float(np.nanmin(btw)) if have.any() else np.nan
            kmin = (k1 + 1 + int(np.nanargmin(btw))) if have.any() else -1
            lo = min(r1, r2)
            miss = ~have
            gap = 0
            if miss.any():
                rl = np.diff(np.flatnonzero(np.diff(np.r_[0, miss.astype(int), 0])).reshape(-1, 2), axis=1)
                gap = int(rl.max()) * FRAME_MS
            seg_t, seg_r = tt[k1:k2 + 1], rr[k1:k2 + 1]
            stp = np.abs(np.diff(seg_r))[np.abs(np.diff(seg_t) - FRAME_MS) < 0.5]
            max_step = float(np.nanmax(stp)) if np.isfinite(stp).any() else np.nan
            n_retr = int(np.sum(btw[have] <= 0.75 * lo))         # frames >= 25 % of the smaller reach back
            cin = c_tr[(c_tr > pk[i]) & (c_tr < pk[i + 1])]
            row = {"session": sid, "trial_id": int(tr), "position": g.position.iloc[i], "t1": pk[i], "t2": pk[i + 1],
                   "dt": dt, "r1": r1, "r2": r2, "r_min": rmin, "t_min": tt[kmin] if kmin >= 0 else np.nan,
                   "retr_mm": lo - rmin, "retr_frac": (lo - rmin) / lo if lo > 0 else np.nan,
                   "n_between": len(btw), "n_tracked_between": int(have.sum()), "gap_ms": gap,
                   "split": "cut" if on[i + 1] - end[i] <= FRAME_MS + 0.5 else "gap",
                   "cut_ms": float(cin[0]) if len(cin) else np.nan,
                   "ml1": ml[k1], "ml2": ml[k2], "dml": ml[k2] - ml[k1],
                   "max_step_mm": max_step, "n_retr25": n_retr,
                   "n1": int(nf[i]), "n2": int(nf[i + 1]), "sv1": sv[i], "sv2": sv[i + 1],
                   # peak on the first / last frame of its run = a monotonic piece (no extension peak inside it)
                   "edge1": bool(pk[i] - on[i] < 0.5 or end[i] - pk[i] < 0.5),
                   "edge2": bool(pk[i + 1] - on[i + 1] < 0.5 or end[i + 1] - pk[i + 1] < 0.5)}
            for c in CAMS:
                E = evt.get(c, {}).get(int(tr))
                b1 = _best_event(E, on[i], end[i])
                b2 = _best_event(E, on[i + 1], end[i + 1])
                row[f"{c}_call"] = _call(b1, b2)
            rows.append(row)
    return pd.DataFrame(rows)


def cuts_for(S: pd.DataFrame, ev: dict) -> dict:
    from wfield_local import camera_licks as CL
    return CL.event_cuts({k: ev[k] for k in CL.ANCHORS if k in ev}, S=S)


def scan_worker(sid: str):
    from scripts.apparatus_mouth import _rig
    from wfield_local import pose3d as P3
    a, d = sid.split("_")
    S = load_frames(sid)
    L = pd.read_csv(RUN / f"{sid}_licks.csv")
    ev = load_events(sid)
    cuts = cuts_for(S, ev)
    pm = pair_metrics(sid, S, L, ev, cuts)
    fl = {"session": sid, "epoch": epoch(sid), "n_saved": len(L)}
    from wfield_local.paths import PathResolver
    rig = _rig(PathResolver(), a, d)
    for f in FLOORS:
        fl[f"n_floor{f}"] = len(P3.lick_peaks(S, rig, cuts=cuts, min_ili_ms=float(f)))
    fl["n_cam4"] = len(ev.get("cam4", []))
    return pm, fl


def cmd_scan(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    from wfield_local import writeguard
    writeguard.assert_writable(OUT / "_")
    OUT.mkdir(parents=True, exist_ok=True)
    res, fail = ak.fan_sessions(sessions(), scan_worker, jobs=jobs)
    if fail:
        print("FAILED:", fail)
    pm = pd.concat([v[0] for _, v in res], ignore_index=True)
    fl = pd.DataFrame([v[1] for _, v in res])
    pm["epoch"] = pm.session.map(dict(zip(fl.session, fl.epoch)))
    pm.to_csv(OUT / "pairs.csv.gz", index=False, float_format="%.4g")
    fl.to_csv(OUT / "floors.csv", index=False)
    print(f"{len(res)} sessions, {len(pm)} pairs; failures {len(fail)}")


# ------------------------------------------------------------------------------------------------ figures
def _tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


VIEW_COL = {"cam1": "#2a6fdb", "cam4": "#d1495b", "cam2": "#3c9d5d", "cam3": "#8e5bb5"}


def draw_pair(fig, spec, S: pd.DataFrame, ti: dict, ev: dict, row, w=(-150, 250)) -> None:
    """Three stacked axes for one pair: reach r(t), lateral ml(t), then per-camera tracking ticks + event rasters."""
    import matplotlib.gridspec as gs
    sub = gs.GridSpecFromSubplotSpec(3, 1, subplot_spec=spec, height_ratios=[2.2, 1.2, 2.2], hspace=0.08)
    x, y = ti[int(row.trial_id)]
    T = S.iloc[x:y]
    t = T.t_ms.to_numpy(float) - row.t1
    m = (t >= w[0]) & (t <= w[1])
    T, t = T[m], t[m]
    P = T[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    r = np.linalg.norm(P, axis=1)
    single = T.tongue_views.to_numpy() == -1
    a0 = fig.add_subplot(sub[0])
    a0.plot(t, r, "-", color="#333", lw=1.2)
    a0.plot(t[~single], r[~single], "o", ms=2.2, color="#333")
    a0.plot(t[single], r[single], "o", ms=2.6, mfc="white", mec="#333", mew=0.8)
    for tp in (0, row.dt):
        a0.axvline(tp, color="#d1495b", lw=1, ls="--")
    if np.isfinite(row.cut_ms):
        a0.axvline(row.cut_ms - row.t1, color="#2a6fdb", lw=1.2)
    if np.isfinite(row.t_min):
        a0.plot(row.t_min - row.t1, row.r_min, "v", color="#2a6fdb", ms=5)
    a0.set_xlim(*w)
    a0.set_ylim(0, max(4.0, np.nanmax(r) * 1.1) if np.isfinite(r).any() else 4)
    a0.set_ylabel("reach mm", fontsize=7)
    a0.set_title(f"{row.session} t{row.trial_id} {row.position} [{row.epoch}]\ndt {row.dt:.0f} ms, retr "
                 f"{row.retr_frac:.2f} ({row.retr_mm:.2f} mm), {row.split}; cam4 {row.cam4_call}, cam1 {row.cam1_call}",
                 fontsize=7)
    a0.tick_params(labelsize=6, labelbottom=False)
    _tidy(a0)
    a1 = fig.add_subplot(sub[1], sharex=a0)
    a1.plot(t, P[:, 0], ".-", color="#7a5c00", lw=0.8, ms=2)
    for tp in (0, row.dt):
        a1.axvline(tp, color="#d1495b", lw=1, ls="--")
    a1.set_ylabel("ml mm", fontsize=7)
    a1.tick_params(labelsize=6, labelbottom=False)
    _tidy(a1)
    a2 = fig.add_subplot(sub[2], sharex=a0)
    yl = []
    for j, c in enumerate(CAMS):                                   # tracked: this view's tongue entered the 3-D point
        y0 = 7 - j
        seen = np.isfinite(T[f"tongue_rep_{c}"].to_numpy(float))
        a2.plot(t[seen], np.full(seen.sum(), y0), "|", color=VIEW_COL[c], ms=5, mew=1)
        yl.append((y0, f"{c} 3-D"))
    for j, c in enumerate(CAMS):                                   # this camera's own lick events (span; x = peak)
        y0 = 3 - j
        E = ev.get(c)
        if E is not None:
            g = E[E.trial_id == row.trial_id]
            for e in g.itertuples():
                s0, s1 = e.on_ms - row.t1, e.off_ms - row.t1
                if s1 < w[0] or s0 > w[1]:
                    continue
                guard = bool(getattr(e, "static", False)) or bool(getattr(e, "far", False))
                a2.plot([s0, s1], [y0, y0], "-", lw=4, color="#bbb" if guard else VIEW_COL[c], solid_capstyle="butt")
                a2.plot(e.t_peak_ms - row.t1, y0, "k|", ms=7)
        yl.append((y0, f"{c} lick"))
    for tp in (0, row.dt):
        a2.axvline(tp, color="#d1495b", lw=1, ls="--")
    if np.isfinite(row.cut_ms):
        a2.axvline(row.cut_ms - row.t1, color="#2a6fdb", lw=1.2)
    a2.set_yticks([v for v, _ in yl], [s for _, s in yl], fontsize=5.5)
    a2.set_ylim(-0.7, 7.7)
    a2.set_xlabel("ms from first peak", fontsize=7)
    a2.tick_params(labelsize=6)
    _tidy(a2)


def sample_pairs(pm: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """August pre + post, a few animals; per bin half pre / half post, at most 2 pairs per session."""
    rng = np.random.default_rng(seed)
    A = pm[pm.session.str.contains(r"_202608") & pm.epoch.isin(["pre", "acute", "subacute", "chronic"])].copy()
    A["phase"] = np.where(A.epoch == "pre", "pre", "post")
    want = {"40-60": 12, "60-80": 12, "120-200": 8}
    out = []
    for b, (lo, hi) in BINS.items():
        B = A[(A.dt >= lo) & (A.dt < hi)]
        for ph in ("pre", "post"):
            C = B[B.phase == ph]
            C = C.iloc[rng.permutation(len(C))]
            C = C.groupby("session").head(2)
            # spread over animals: round-robin by animal
            C = C.assign(animal=C.session.str[:4], k=C.groupby(C.session.str[:4]).cumcount()).sort_values(["k", "animal"])
            out.append(C.head(want[b] // 2).assign(bin=b))
    return pd.concat(out, ignore_index=True)


def cmd_examples() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pm = pd.read_csv(OUT / "pairs.csv.gz")
    smp = sample_pairs(pm)
    smp.to_csv(OUT / "sample_pairs.csv", index=False, float_format="%.4g")
    cache = {}
    for b in BINS:
        R = smp[smp.bin == b].sort_values(["phase", "session"], ascending=[False, True])
        n = len(R)
        nc = 4
        nr = int(np.ceil(n / nc))
        fig = plt.figure(figsize=(4.2 * nc, 4.6 * nr))
        G = fig.add_gridspec(nr, nc, hspace=0.45, wspace=0.35)
        for k, row in enumerate(R.itertuples()):
            if row.session not in cache:
                S = load_frames(row.session)
                cache[row.session] = (S, _trial_index(S), load_events(row.session))
            S, ti, ev = cache[row.session]
            draw_pair(fig, G[k // nc, k % nc], S, ti, ev, row)
        fig.suptitle(f"3-D lick pairs, peak-to-peak {b} ms (August; pre first, then post). Red dashed = the two "
                     f"extension peaks; blue line = event_cuts split; blue triangle = closest return to the mouth "
                     f"between the peaks;\nreach = |tip - mouth zero|; open dots = one-view fill; ticks = views in the "
                     f"3-D point; bars = each camera's own lick events (grey = guarded side event), black tick = its "
                     f"peak", fontsize=9)
        fig.savefig(OUT / f"fig1_pairs_{b}ms.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("->", OUT / f"fig1_pairs_{b}ms.png")
    summary_fig(pm)


def summary_fig(pm: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    cols = {"40-60": "#d1495b", "60-80": "#e8a33d", "120-200": "#2a6fdb"}
    for b, (lo, hi) in BINS.items():
        B = pm[(pm.dt >= lo) & (pm.dt < hi)]
        v = B.retr_frac.dropna().clip(-0.2, 1.0)
        ax[0].hist(v, bins=np.linspace(-0.2, 1.0, 49), histtype="step", density=True, lw=1.6, color=cols[b],
                   label=f"{b} ms (n={len(B)}; no 3-D frame between: {B.retr_frac.isna().sum()})")
    ax[0].set_xlabel("retraction depth = (min peak reach - min reach between) / min peak reach\n(clipped at -0.2)")
    ax[0].set_ylabel("density")
    ax[0].legend(fontsize=7, frameon=False)
    edges = np.arange(40, 304, 8)
    mid = (edges[:-1] + edges[1:]) / 2
    q = [pm[(pm.dt >= a) & (pm.dt < b)].retr_frac.quantile([0.1, 0.25, 0.5, 0.75]).to_numpy()
         if ((pm.dt >= a) & (pm.dt < b)).sum() > 20 else np.full(4, np.nan) for a, b in zip(edges[:-1], edges[1:])]
    q = np.array(q)
    ax[1].fill_between(mid, q[:, 1], q[:, 3], color="#999", alpha=0.35, lw=0, label="IQR")
    ax[1].plot(mid, q[:, 2], "-", color="#333", lw=2, label="median")
    ax[1].plot(mid, q[:, 0], ":", color="#333", lw=1.2, label="10th pct")
    ax[1].set_xlabel("peak-to-peak interval (ms, 8 ms bins)")
    ax[1].set_ylabel("retraction depth")
    ax[1].legend(fontsize=7, frameon=False)
    for k, c in enumerate(("cam4", "cam1")):
        share = [(pm[(pm.dt >= lo) & (pm.dt < hi)][f"{c}_call"] == "two").mean() for lo, hi in BINS.values()]
        ax[2].bar(np.arange(3) + (k - 0.5) * 0.38, share, 0.36, color=["#d1495b", "#2a6fdb"][k], label=f"{c}: two events")
    ax[2].set_xticks(range(3), list(BINS))
    ax[2].set_xlabel("peak-to-peak bin (ms)")
    ax[2].set_ylabel("share of pairs where the camera calls TWO licks")
    ax[2].legend(fontsize=7, frameon=False)
    sh_edge = [((pm.dt >= a) & (pm.dt < b) & (pm.edge1 | pm.edge2)).sum() / max(((pm.dt >= a) & (pm.dt < b)).sum(), 1)
               for a, b in zip(edges[:-1], edges[1:])]
    sh_gap = [((pm.dt >= a) & (pm.dt < b) & (pm.gap_ms >= 12)).sum() / max(((pm.dt >= a) & (pm.dt < b)).sum(), 1)
              for a, b in zip(edges[:-1], edges[1:])]
    ax[3].plot(mid, sh_edge, "-", color="#d1495b", lw=2, label="a peak on the first / last frame of its lick (fragment)")
    ax[3].plot(mid, sh_gap, "-", color="#2a6fdb", lw=2, label="tongue untracked >= 12 ms between the peaks")
    ax[3].set_xlabel("peak-to-peak interval (ms, 8 ms bins)")
    ax[3].set_ylabel("share of pairs")
    ax[3].set_ylim(0, 1.02)
    ax[3].legend(fontsize=7, frameon=False)
    for a in ax:
        _tidy(a)
    fig.suptitle("All 126 sessions, consecutive 3-D lick pairs of one trial (four-camera 3net run, 40 ms floor applied)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_retraction_summary.png", dpi=140)
    plt.close(fig)
    print("->", OUT / "fig2_retraction_summary.png")


# ------------------------------------------------------------------------------------------------ video
def cmd_frames(n_each: int) -> None:
    import cv2

    from scripts.cam1_licks import _video
    from scripts.pose3d import PROD
    from scripts.pose3d_figures import _frame
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    pm = pd.read_csv(OUT / "pairs.csv.gz")
    A = pm[pm.session.str.contains(r"_202608") & (pm.epoch != "none")]
    S40 = A[(A.dt >= 40) & (A.dt < 80) & (A.n_tracked_between >= A.n_between - 2)]
    one = S40[S40.retr_frac < 0.08].sort_values("retr_frac")              # clearest "no return between peaks"
    two = S40[S40.retr_frac > 0.5].sort_values("retr_frac", ascending=False)
    ctl = A[(A.dt >= 120) & (A.dt < 200) & (A.n_tracked_between == A.n_between) & (A.retr_frac > 0.4)]
    pick = []
    for lab, D in (("one_protrusion", one), ("two_protrusions", two), ("control", ctl)):
        D = D.groupby(D.session.str[:4]).head(1)                           # one per animal first
        pick += [(lab, r) for r in D.head(n_each).itertuples()]
    wins = {}
    rows = []
    for lab, r in pick:
        a, d = r.session.split("_")
        if r.session not in wins:
            wins[r.session] = {c: oi.session_windows(rv, PROD[c], a, d, c, pre_s=1.0, post_s=5.0) for c in ("cam1", "cam4")}
            wins[r.session]["S"] = load_frames(r.session)
        W = wins[r.session]
        S = W["S"]
        T = S[(S.trial_id == r.trial_id) & (S.t_ms >= r.t1 - 40) & (S.t_ms <= r.t2 + 40)]
        T = T.iloc[::2]                                                    # every 8 ms
        tiles = {}
        for cam, scol in (("cam4", "src4"), ("cam1", "src1")):
            if W[cam] is None:
                continue
            ix, fl = W[cam]
            rix = pd.Series(np.arange(len(ix)), index=ix.src_frame.to_numpy())
            rix = rix[~rix.index.duplicated()]
            cap = cv2.VideoCapture(str(_video(rv, a, d, cam, ix, r.trial_id)))
            pts = [fl.iloc[int(rix[s])] if s in rix.index else None for s in T[scol].astype(int)]
            ctr = np.nanmedian([[q["jaw_x"], q["jaw_y"]] for q in pts if q is not None], axis=0)
            row_t = []
            for (_, f), q in zip(T.iterrows(), pts):
                im = _frame(cap, int(f[scol]))
                if q is not None:
                    for p, col in (("tongue", (60, 60, 230)), ("jaw", (230, 160, 30))):
                        if q[f"{p}_likelihood"] >= 0.4:
                            cv2.circle(im, (int(q[f"{p}_x"]), int(q[f"{p}_y"])), 5, col, 2, cv2.LINE_AA)
                h = 260
                x0 = int(np.clip(ctr[0] - h, 0, im.shape[1] - 2 * h))
                y0 = int(np.clip(ctr[1] - h, 0, im.shape[0] - 2 * h))
                tile = cv2.resize(im[y0:y0 + 2 * h, x0:x0 + 2 * h], (200, 200))
                rr = np.linalg.norm([f.tongue_ml, f.tongue_ap, f.tongue_dv])
                txt = f"{f.t_ms - r.t1:+.0f}ms r={rr:.1f}" if np.isfinite(rr) else f"{f.t_ms - r.t1:+.0f}ms no3D"
                mk = " P" if abs(f.t_ms - r.t1) < 1 or abs(f.t_ms - r.t2) < 1 else ""
                cv2.putText(tile, txt + mk, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                row_t.append(tile)
            cap.release()
            band = np.full((18, 200, 3), 30, np.uint8)
            cv2.putText(band, cam, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles[cam] = np.hstack(row_t)
        head = np.full((24, max(v.shape[1] for v in tiles.values()), 3), 20, np.uint8)
        cv2.putText(head, f"{lab}: {r.session} t{r.trial_id} {r.position} [{r.epoch}] dt {r.dt:.0f} ms, retr "
                    f"{r.retr_frac:.2f}; cam4 {r.cam4_call}, cam1 {r.cam1_call}; every 8 ms, P = 3-D peak; red = "
                    f"tongue, orange = jaw (LP 2-D, lik >= 0.4)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        img = np.vstack([head] + [tiles[c] for c in ("cam4", "cam1") if c in tiles])
        name = f"fig3_frames_{lab}_{r.session}_t{r.trial_id}_{int(r.t1)}.png"
        cv2.imwrite(str(OUT / name), img)
        rows.append({"file": name, "kind": lab, "session": r.session, "trial_id": r.trial_id, "t1": r.t1,
                     "dt": r.dt, "retr_frac": r.retr_frac, "cam4_call": r.cam4_call, "cam1_call": r.cam1_call})
        print("->", name)
    pd.DataFrame(rows).to_csv(OUT / "frames_index.csv", index=False, float_format="%.4g")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "examples", "frames"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--n-each", type=int, default=3)
    a = ap.parse_args(argv)
    if a.mode == "scan":
        cmd_scan(a.jobs)
    elif a.mode == "examples":
        cmd_examples()
    else:
        cmd_frames(a.n_each)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

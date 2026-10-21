"""Blind clip review of the ATTEMPTED gate (Priya 10-21: review the attempted / canonical_attempted gate before any
default change; "we can also look at xy jaw traces and nose traces as part of judging attempt/reaction qc").

    python -m scripts.attempt_review [--per 6] [--seed 1]

Four strata of ``--per`` trials each, from `trial_attempts.csv` (the encoding window's classes, rebuilt from
qc/jaw_onsets.csv) and curated post-stroke sessions:
  outside   attempted (jaw_only / incomplete -- not hits, which have a contact) but OUTSIDE the canonical (reference) gate:
            the trials the attempted gate ADDS
  inside    no_try INSIDE the canonical gate: the trials it REMOVES
  jaw_near  jaw_only whose largest jaw deflection is near the floor (cam1 15-20 px or cam4 4-6 px)
  try_near  no_try whose largest jaw deflection is just under it (cam1 10-15 px or cam4 3-4 px)
Each clip: -300 ms .. the trial's own STOP + 200 ms (Priya 10-21: reactions > 1.5 s after the cue; fallback 3500 ms),
every 8th frame (32 ms); cam1 (left, where it exists) and cam4 (right),
same DAQ times; plus a cue-aligned trace panel (jaw x / y both cameras, cam4 nose x / y, cam4 tongue likelihood, DAQ
cam4 WHISKER-PAD motion energy (both pads, `roi_motion` ROIs; Priya 10-21 "why no whisker ME?"), DAQ
TREADMILL speed with a 1 mm/s guide -- Priya 10-21: walking below the running threshold jiggles the jaw; raw,
baseline-subtracted, NO threshold drawn). Class, stratum and session are hidden (key.csv).
Output <dlc>/session_poses/attempt_review/{candidates.json, key.csv}.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import pose_windows

V36 = "lp_r4_heatmap_20261013__clean_v36"
KEYFRAME_EVERY = 12                                         # cam4 FMP4 encoder GOP (10-21)
CLIP_FROM, TRACE_FROM, AFTER_STOP = -300.0, -500.0, 200.0     # windows run to the trial's stop + AFTER_STOP
STEP = 8
TILE = 210
CROP4, CROP1 = (340, 300), (300, 260)
CUT = 0.6


def _cam1(rd: Path, s: str):
    d1 = rd / "session_poses" / f"{s}_cam1_full"
    if not (d1 / "windows_LP.csv").exists():
        return None
    ix = pose_windows.index(d1)
    if "t_daq_ms" not in ix:
        return None
    W = pose_windows.lp(d1)
    ok = W[("jaw", "likelihood")].to_numpy(float) >= CUT
    jaw = np.c_[W[("jaw", "x")].to_numpy(float), W[("jaw", "y")].to_numpy(float)]
    jaw[~ok] = np.nan
    return ix, jaw


def _cam4(ro: Path, s: str):
    a, d = s.split("_")
    w = np.load(ro / V36 / "sessions" / a / d / "windows.npz")
    ix = pd.DataFrame({k: w[f"ix_{k}"] for k in ("trial_id", "t_daq_ms", "src_frame", "stop_ms", "daq_s")})
    parts = [str(p) for p in w["parts"]]
    P = w["pose"].astype(float)
    out = {}
    for p in ("jaw", "nose", "tongue"):
        q = P[:, parts.index(p), :].copy()
        if p != "tongue":
            q[q[:, 2] < CUT, :2] = np.nan
        out[p] = q
    return ix, out


def _peak(t, y, stop):
    pre = (t >= -500) & (t < 0) & np.isfinite(y)
    win = (t >= 0) & (t <= stop) & np.isfinite(y)
    if pre.sum() < 2 or not win.any():
        return np.nan
    return float(np.nanmax(np.abs(y[win] - np.nanmedian(y[pre]))))


def _pools(rv, rng):
    from wfield_local import analysis_kit as ak
    ro = Path(rv.root("orofacial"))
    A = pd.read_csv(ro / V36 / "imaging_encoding" / "qc" / "trial_attempts.csv")
    A = A[A.label.isin(set(ak.curated_labels())) & (A.epoch != "pre")]
    A["attempted"] = A.attempted.astype(str).str.lower().isin(["true", "1", "1.0"])
    A["canonical"] = A.canonical.astype(str).str.lower().isin(["true", "1", "1.0"])
    A = A[A.pos != "close_mid"]                        # = close_center: jaw unknown there (both cameras masked)
    return A


def _with_peaks(rv, T: pd.DataFrame) -> pd.DataFrame:
    """cam4 / cam1 largest |jaw y - pre-cue median| in cue .. stop per trial (px)."""
    ro, rd = Path(rv.root("orofacial")), Path(rv.root("dlc"))
    rows = []
    for lab, g in T.groupby("label"):
        s = f"{lab[:4]}_2026{lab[5:]}"
        ix4, P4 = _cam4(ro, s)
        c1 = _cam1(rd, s)
        for tid in g.trial_id:
            h = ix4[ix4.trial_id == tid]
            stop = float(h.stop_ms.iloc[0]) if len(h) and np.isfinite(h.stop_ms.iloc[0]) else 3500.0
            p4 = _peak(h.t_daq_ms.to_numpy(float), P4["jaw"][h.index.to_numpy(), 1], stop) if len(h) else np.nan
            p1 = np.nan
            if c1 is not None:
                h1 = c1[0][c1[0].trial_id == tid]
                if len(h1):
                    p1 = _peak(h1.t_daq_ms.to_numpy(float), c1[1][h1.index.to_numpy(), 1], stop)
            rows.append({"label": lab, "trial_id": int(tid), "peak4_px": p4, "peak1_px": p1})
    return T.merge(pd.DataFrame(rows), on=["label", "trial_id"])


def pick(rv, per: int, rng) -> pd.DataFrame:
    A = _pools(rv, rng)
    seed = lambda: int(rng.integers(1 << 30))  # noqa: E731
    out = [A[A.attempted & ~A.canonical & A.cls.isin(["jaw_only", "incomplete"])].sample(per, random_state=seed())
           .assign(stratum="outside"),
           A[A.canonical & (A.cls == "no_try")].sample(per, random_state=seed()).assign(stratum="inside")]
    # near-threshold strata: peaks only for a random subset of trials (reading every session's tracks is slow)
    for cls, name, c1r, c4r in (("jaw_only", "jaw_near", (15, 20), (4, 6)), ("no_try", "try_near", (10, 15), (3, 4))):
        pool = A[A.cls == cls]
        pool = pool.sample(min(len(pool), 40 * per), random_state=seed())
        P = _with_peaks(rv, pool)
        near = P[P.peak1_px.between(*c1r) | P.peak4_px.between(*c4r)]
        print(f"{name}: {len(near)} of {len(P)} sampled {cls} trials near the floor")
        out.append(near.sample(min(per, len(near)), random_state=seed()).assign(stratum=name))
    return pd.concat(out, ignore_index=True)


def _trace_png(t4, P4, t1, j1, end: float, stop: float, tread=None, pad=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(7.6, 5.6), sharex=True)

    def rel(t, v):
        b = np.nanmedian(v[(t >= -500) & (t < 0)])
        return v - b

    ax = axes[0]
    for k, (lab, c) in enumerate((("x", "tab:orange"), ("y", "tab:blue"))):
        ax.plot(t4, rel(t4, P4["jaw"][:, k]), color=c, lw=0.9, label=f"cam4 jaw {lab}")
        if t1 is not None:
            ax.plot(t1, rel(t1, j1[:, k]), color=c, lw=0.9, ls="--", label=f"cam1 jaw {lab}")
    ax.legend(fontsize=6, ncol=4, loc="upper right", frameon=False)
    ax.set_ylabel("jaw px", fontsize=7)
    ax = axes[1]
    for k, (lab, c) in enumerate((("x", "tab:orange"), ("y", "tab:blue"))):
        ax.plot(t4, rel(t4, P4["nose"][:, k]), color=c, lw=0.9, label=f"cam4 nose {lab}")
    ax.legend(fontsize=6, ncol=2, loc="upper right", frameon=False)
    ax.set_ylabel("nose px", fontsize=7)
    ax = axes[2]
    ax.plot(t4, P4["tongue"][:, 2], color="tab:red", lw=0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("tongue\nlikelihood", fontsize=7)
    ax = axes[3]
    if pad is not None:
        for k, (lab, c) in enumerate((("image-left pad", "tab:purple"), ("image-right pad", "tab:brown"))):
            ax.plot(t4, rel(t4, pad[:, k]), color=c, lw=0.8, label=lab)
        ax.legend(fontsize=6, ncol=2, loc="upper right", frameon=False)
    ax.set_ylabel("whisker pad\nmotion", fontsize=7)
    ax = axes[4]
    if tread is not None:
        ax.plot(t4, tread, color="tab:green", lw=0.9)
    ax.axhline(1.0, color="0.6", lw=0.6, ls="--")
    ax.set_ylim(bottom=min(-0.5, float(np.nanmin(tread)) if tread is not None and np.isfinite(tread).any() else -0.5))
    ax.set_ylabel("treadmill\nmm/s", fontsize=7)
    ax.set_xlabel("ms from cue (DAQ)", fontsize=7)
    for ax in axes:
        ax.axvline(0, color="0.5", lw=0.6)
        ax.axvline(stop, color="0.5", lw=0.6, ls=":")
        ax.tick_params(labelsize=6)
        ax.set_xlim(TRACE_FROM, end)
    axes[0].text(stop, axes[0].get_ylim()[1], " stop", fontsize=6, va="top", color="0.4")
    fig.tight_layout(pad=0.4)
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode("ascii")


def build(rv, P: pd.DataFrame, rng):
    import cv2

    from scripts.cam1_licks import _video
    ro, rd = Path(rv.root("orofacial")), Path(rv.root("dlc"))
    cands, keys = [], []
    for lab, g in P.groupby("label"):
        s = f"{lab[:4]}_2026{lab[5:]}"
        a, d = s.split("_")
        ix4, P4 = _cam4(ro, s)
        c1 = _cam1(rd, s)
        cap4 = cv2.VideoCapture(str(_video(rv, a, d, "cam4", pd.DataFrame(), 0)))
        from wfield_local import roi_motion as RM
        nz = P4["nose"][:, :2]
        nz = nz[np.isfinite(nz).all(1)]
        pad_rois = RM.rois("cam4", tuple(np.median(nz, 0)) if len(nz) else (340.0, 270.0), (680, 680))
        try:                                             # DAQ treadmill speed (mm/s) for the walking row
            from wfield_local import behavior_events, daq_io
            ts, sp = behavior_events.session_speed(daq_io.session_h5(a, d, rv))
        except Exception as exc:
            print(f"{s}: no treadmill ({type(exc).__name__}: {exc})")
            ts = sp = None
        cap1 = cv2.VideoCapture(str(_video(rv, a, d, "cam1", pd.DataFrame(), 0))) if c1 is not None else None
        for r in g.itertuples():
            h4 = ix4[ix4.trial_id == r.trial_id]
            t4 = h4.t_daq_ms.to_numpy(float)
            k4 = h4.index.to_numpy()
            pre4 = (t4 >= -500) & (t4 < 0)
            jaw_c = np.nanmedian(P4["jaw"][k4[pre4], :2], 0)
            nose_c = np.nanmedian(P4["nose"][k4[pre4], :2], 0)
            c4 = (jaw_c + nose_c) / 2 if np.isfinite(nose_c).all() else jaw_c
            t1 = j1 = h1 = None
            cc1 = None
            if c1 is not None:
                h1 = c1[0][c1[0].trial_id == r.trial_id]
                if len(h1):
                    t1 = h1.t_daq_ms.to_numpy(float)
                    j1 = c1[1][h1.index.to_numpy()]
                    cc1 = np.nanmedian(j1[(t1 >= -500) & (t1 < 0)], 0)
            stop = float(h4.stop_ms.iloc[0]) if np.isfinite(h4.stop_ms.iloc[0]) else 3500.0
            end = stop + AFTER_STOP
            tr = (t4 >= TRACE_FROM) & (t4 <= end)
            sf = h4.src_frame.to_numpy(int)
            tile_k = np.flatnonzero((t4 >= CLIP_FROM) & (t4 <= end))[::STEP]
            want = {int(sf[k]) for k in tile_k}
            f0, f1 = int(sf[tr].min()) - 1, int(sf[tr].max())
            cap4.set(cv2.CAP_PROP_POS_FRAMES, max(f0, 0))
            pads, full4 = {}, {}
            for f in range(max(f0, 0), f1 + 1):
                ok, im = cap4.read()
                if not ok:
                    break
                g = im[..., 0]
                pads[f] = [g[y0:y1, x0:x1].copy() for (x0, y0, x1, y1) in pad_rois] + [g[0:60, 0:60].copy()]
                if f in want:
                    full4[f] = im
            me3 = np.full((int(tr.sum()), 3), np.nan)
            for j, f in enumerate(sf[tr]):
                if f in pads and f - 1 in pads:
                    me3[j] = [np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
                              for a, b in zip(pads[f], pads[f - 1])]
            del pads
            # ENCODER KEYFRAMES (10-21): the video's frame index % 12 == 0 carries ~+33 % motion energy (measured: phase
            # profile flat on the other 11; 20.8 Hz power 31-82x the median). Those exact frames are replaced by
            # interpolation from their neighbours: 20.8 Hz power down 4-8x (PS94 / PS93 0810 / 0922), 62.5 Hz removed.
            bad = (sf[tr] % KEYFRAME_EVERY) == 0
            pad_me = me3[:, :2].copy()
            idx = np.arange(len(pad_me))
            for k in range(2):
                ok = ~bad & np.isfinite(pad_me[:, k])
                if ok.sum() > 1:
                    pad_me[bad, k] = np.interp(idx[bad], idx[ok], pad_me[ok, k])
            png = _trace_png(t4[tr], {k: v[k4[tr]] for k, v in P4.items()},
                             None if t1 is None else t1[(t1 >= TRACE_FROM) & (t1 <= end)],
                             None if j1 is None else j1[(t1 >= TRACE_FROM) & (t1 <= end)], end, stop,
                             None if ts is None else np.interp(h4.daq_s.to_numpy(float)[tr], ts, sp), pad_me)
            frames = []
            for k in tile_k:
                tiles = []
                for cap, src, cc, crop in ((cap1, None if h1 is None or not len(h1) else
                                            int(h1.src_frame.iloc[int(np.argmin(np.abs(t1 - t4[k])))]), cc1, CROP1),
                                           (cap4, int(h4.src_frame.iloc[k]), c4, CROP4)):
                    th = int(TILE * crop[1] / crop[0])
                    if cap is None or src is None:
                        tiles.append(np.zeros((th, TILE, 3), np.uint8))
                        continue
                    if cap is cap4:
                        im = full4.get(src)
                        ok = im is not None
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, src)
                        ok, im = cap.read()
                    if not ok:
                        im = np.zeros((680, 680, 3), np.uint8)
                    if cc is None or not np.isfinite(cc).all():
                        cc = np.array([im.shape[1] / 2, im.shape[0] / 2])
                    x0 = int(np.clip(cc[0] - crop[0] / 2, 0, im.shape[1] - crop[0]))
                    y0 = int(np.clip(cc[1] - crop[1] / 2, 0, im.shape[0] - crop[1]))
                    tiles.append(cv2.resize(im[y0:y0 + crop[1], x0:x0 + crop[0]], (TILE, th)))
                hh = max(x.shape[0] for x in tiles)
                tiles = [np.pad(x, ((0, hh - x.shape[0]), (0, 0), (0, 0))) for x in tiles]
                fr = np.hstack(tiles)
                cv2.putText(fr, f"{t4[k]:+.0f} ms", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                            cv2.LINE_AA)
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), [cv2.IMWRITE_JPEG_QUALITY, 65])
                frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
            cid = f"a{len(cands):03d}"
            cands.append({"id": cid, "position": r.pos, "frames": frames, "trace": png, "has_cam1": cap1 is not None,
                          "stop_ms": stop})
            keys.append({"id": cid, "label": lab, "trial_id": int(r.trial_id), "position": r.pos, "epoch": r.epoch,
                         "stratum": r.stratum, "cls": r.cls, "attempted": bool(r.attempted),
                         "canonical": bool(r.canonical),
                         "peak4_px": getattr(r, "peak4_px", np.nan), "peak1_px": getattr(r, "peak1_px", np.nan)})
        cap4.release()
        if cap1 is not None:
            cap1.release()
    order = rng.permutation(len(cands))
    return [cands[i] for i in order], [keys[i] for i in order]


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    rv = PathResolver()
    rng = np.random.default_rng(a.seed)
    P = pick(rv, a.per, rng)
    cands, keys = build(rv, P, rng)
    out = Path(rv.root("dlc")) / "session_poses" / "attempt_review"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(cands))
    pd.DataFrame(keys).to_csv(out / "key.csv", index=False)
    print(pd.DataFrame(keys).groupby(["stratum", "cls"]).size().to_string())
    print(f"-> {out} ({len(cands)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

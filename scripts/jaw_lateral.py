"""Jaw x ceiling retune (handoff 10-14 to-do 9) and jaw lateral deviation on opening (to-do 10).

    PYTHONPATH=. python -m scripts.jaw_lateral ceiling   [--sessions PS93_20260814 ...] [--qc PS93_20260814]
    PYTHONPATH=. python -m scripts.jaw_lateral deviation [--sessions ...] [--rule v34|v35]

Reads `session_poses/<animal>_<date>[_r4]/windows_{index,DLC}.csv` (DLC round 3). Writes to
`session_poses/jaw_lateral/`.

ceiling: v3.4's `x_ceiling_px` (40, an OLD-RIG value) wipes every raw jaw frame further than 40 px sideways from the
session's resting jaw. On our rig that is 9 % of wide-open frames on PS93 0814 and 14 % on PS95 0917, and most of
those are reached smoothly (median entry step 4-6 px/frame), i.e. real. The candidate "v35" keeps a loose hard bound
(75 px = the first-stage `x_range_pm`) and adds a median-filter jump test on x (`frame_jump_thr_x_px`), which catches
the side-flips (40-80 px in one frame) without cutting smooth excursions. The QC sheet shows frames v3.4 wiped that
v35 keeps, and frames v35's jump test wipes, for Priya's visual check before the default changes.

deviation: per trial, the jaw's sideways displacement from rest while the mouth is open, in the SPOUT frame
(`spout_frame`: lr axis perpendicular to the apparatus centre line, so a tilted head / camera does not turn a
straight opening into an apparent sideways shift), reported two ways:
  * `lr_wide`  median lr over frames with opening > `--open-px` (50) -- px, + = image-right = the MOUSE'S LEFT on cam4
               (confirmed 2026-10-11 by the 3-D stage fit: *_L = mouse's left, and the *_L spouts sit image-right);
  * `angle`    the jaw opening angle, atan2(lr, opening) from a per-trial fit over open frames (> 15 px) -- degrees,
               independent of how wide each animal opens.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import orofacial_clean as oc

FPS = 250.0
SEP = 300
DEFAULT_SESSIONS = ("PS93_20260814", "PS93_20260819_r4", "PS93_20260821", "PS93_20260826_r4", "PS93_20260908",
                    "PS92_20260821_r4", "PS92_20260824_r4", "PS94_20260921_r4", "PS95_20260917_r4")
LICK_WIN_MS = (70.0, 3000.0)                     # = orofacial_kinematics lick12_detect_win_ms


def rules() -> dict:
    """v35 = the configured default (adopted 2026-10-14); v34 = the transcribed stroke_orofacial rule (40-px x ceiling,
    no x tests), kept for comparison."""
    v35 = dict(oc.params("jaw")["v34"])
    v34 = {**v35, "x_ceiling_px": 40.0, "frame_jump_thr_x_px": None, "x_flip_step_px": None}
    return {"v34": v34, "v35": v35}


def poses_root() -> Path:
    from scripts.session_poses import session_dir
    from wfield_local.paths import PathResolver
    return session_dir(PathResolver(), "PS93:20260814")[2].parent


def load(name: str):
    """(raw pose frame, index, padded big frame, row positions, cue frames, spout frame) for one session folder."""
    from wfield_local import spout_frame as SF
    d = poses_root() / name
    idx = pd.read_csv(d / "windows_index.csv")
    raw = oc.read_pose(d / "windows_DLC.csv").iloc[:len(idx)]
    idx = idx.iloc[:len(raw)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    fr = SF.from_medians(SF.position_medians(raw[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    parts, pos, cues, k = [], [], [], 0
    for _tk, g in idx.groupby("trial_k", sort=True):
        rows = g.index.to_numpy()
        parts.append(raw.iloc[rows].reset_index(drop=True))
        pos.append(np.arange(k, k + len(rows)))
        cues.append(k + int(np.argmin(np.abs(g.t_ms.to_numpy()))))
        k += len(rows) + SEP
        parts.append(pd.DataFrame(np.nan, index=range(SEP), columns=raw.columns))
    big = pd.concat(parts, ignore_index=True)
    return raw, idx, big, np.concatenate(pos), np.array(cues), fr


def clean_jaw(big, cues, q):
    """Pre-v3.4 jaw (v5p3) and the v3.4-style pass under rule `q`; both on the padded array."""
    pre = oc.clean_bodypart(big, "jaw", cues, FPS)
    return pre, oc.jaw_v34(pre, cues, q)


# --------------------------------------------------------------------------- ceiling

def cmd_ceiling(sessions, qc_sessions, out: Path) -> None:
    R = rules()
    rows = []
    qc_pool = {"rescued": [], "jump": [], "y_ceiling": []}
    for name in sessions:
        raw, idx, big, pos, cues, fr = load(name)
        res = {}
        for rname, q in R.items():
            pre, c = clean_jaw(big, cues, q)
            res[rname] = c
            by, bx = c.extra["v34_bmean"]
            raw_ok = (pre.fill_method == oc.FILL_NONE) & np.isfinite(pre.y_final)
            dy = pre.y_final - by
            wide = raw_ok & (dy > 50)
            kept = c.fill_method == oc.FILL_NONE
            dxk = np.where(kept, c.x_final - bx, np.nan)
            rows.append(dict(session=name, rule=rname, n_raw=int(raw_ok.sum()),
                             wiped_pct=100 * float(np.mean(~kept[raw_ok])),
                             wiped_x_ceiling_pct=100 * float(np.mean(c.extra["v34_wiped_x"][raw_ok])),
                             wiped_x_jump_pct=100 * float(np.mean(c.extra["v34_wiped_frame_outlier_x"][raw_ok])),
                             wide_n=int(wide.sum()), wide_wiped_pct=100 * float(np.mean(~kept[wide])),
                             wide_dx_p95=float(np.nanpercentile(np.abs(dxk[wide]), 95)),
                             wide_dx_p99=float(np.nanpercentile(np.abs(dxk[wide]), 99))))
        if name in qc_sessions:
            a, b = res["v34"], res["v35"]
            src = np.full(len(big), -1)
            src[pos] = idx.src_frame.to_numpy()
            bl = a.extra["v34_bmean"]
            for kind, m in (("rescued", a.extra["v34_wiped_x"] & (b.fill_method == oc.FILL_NONE)),
                            ("jump", b.extra["v34_wiped_frame_outlier_x"]),
                            ("y_ceiling", b.extra["v34_wiped_y"])):
                for i in np.flatnonzero(m & (src >= 0)):
                    qc_pool[kind].append((name, int(src[i]), float(pre.x_raw[i]), float(pre.y_raw[i]),
                                          bl[1] + pre.X0, bl[0] + pre.Y0, float(pre.x_raw[i] - pre.X0 - bl[1])))
        print(f"{name}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "jaw_ceiling_rules.csv", index=False, float_format="%.3f")
    print(df.round(2).to_string(index=False))
    if qc_sessions:
        qc_sheet(qc_pool, out / "jaw_ceiling_qc.png")


def qc_sheet(pool, path: Path, n: int = 12, crop=(260, 220), tw=200) -> None:
    """Two blocks of mouth crops: frames v3.4's x ceiling wiped that v35 keeps (should look RIGHT), and frames v35's
    x jump test wipes (should look WRONG). Cyan dot = DLC jaw, white cross = the session's resting jaw."""
    import cv2

    from wfield_local.paths import PathResolver
    rv = PathResolver()
    rng = np.random.default_rng(0)
    caps, blocks = {}, []
    for kind, title in (("rescued", "v3.4 x-ceiling WIPED, v35 KEEPS (expect: jaw point correct)"),
                        ("jump", "v35 x-jump test WIPES (expect: jaw point wrong / flipped)"),
                        ("y_ceiling", "y ceiling (125 px, old rig, unchanged) WIPES -- real wide openings?")):
        items = pool[kind]
        if not items:
            continue
        pick = [items[i] for i in sorted(rng.choice(len(items), min(n, len(items)), replace=False))]
        tiles = []
        for name, f, x, y, bx, by, dx in pick:
            animal, date = name.split("_")[:2]
            if name not in caps:
                vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam4_*.avi"))[0]
                caps[name] = cv2.VideoCapture(str(vid))                     # READ-ONLY
            cap = caps[name]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                continue
            cv2.drawMarker(im, (int(bx), int(by)), (255, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
            cv2.circle(im, (int(x), int(y)), 5, (255, 255, 0), 2, cv2.LINE_AA)
            cx, cy = int(bx), int(by) - 30
            x0, y0 = max(cx - crop[0] // 2, 0), max(cy - crop[1] // 2, 0)
            t = cv2.resize(im[y0:y0 + crop[1], x0:x0 + crop[0]], (tw, int(tw * crop[1] / crop[0])))
            cv2.putText(t, f"{animal} {date[4:]} f{f} dx{dx:+.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        head = np.zeros((26, grid.shape[1], 3), np.uint8)
        cv2.putText(head, f"{title}  ({len(items)} frames in pool)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        blocks.append(np.vstack([head, grid]))
    cv2.imwrite(str(path), np.vstack(blocks))
    print(f"QC sheet -> {path}")


# --------------------------------------------------------------------------- deviation

def session_deviation(name: str, q: dict, open_px: float) -> pd.DataFrame:
    raw, idx, big, pos, cues, fr = load(name)
    _, c = clean_jaw(big, cues, q)
    by, bx = c.extra["v34_bmean"]
    kept = c.fill_method == oc.FILL_NONE
    # displacement from the resting jaw rotated into the spout frame: `opening` = the component along the centre
    # line, `lr` = the component across it (+ = image-right)
    vx, vy = c.x_final - bx, c.y_final - by
    ap_ax, lr_ax = fr.ap_axis, fr.lr_axis
    opening = vx * ap_ax[0] + vy * ap_ax[1]
    lr = vx * lr_ax[0] + vy * lr_ax[1]
    if np.nanmedian(np.where(kept & (vy > 30), opening, np.nan)) < 0:   # sign so that + opening = jaw dropping
        opening = -opening
    animal, date = name.split("_")[:2]
    out = []
    t_ms = np.full(len(big), np.nan)
    t_ms[pos] = idx.t_ms.to_numpy()
    tk = np.full(len(big), -1)
    tk[pos] = idx.trial_k.to_numpy()
    meta = idx.groupby("trial_k").agg(trial_id=("trial_id", "first"), position=("position", "first"))
    win = kept & (t_ms >= LICK_WIN_MS[0]) & (t_ms <= LICK_WIN_MS[1])
    for k, g in meta.iterrows():
        m = win & (tk == k)
        o, lv = opening[m], lr[m]
        wide = o > open_px
        op = o > 15
        ang = np.nan
        if op.sum() >= 10:
            # least-squares direction of (opening, lr) through the rest point
            ang = float(np.degrees(np.arctan2(np.sum(lv[op] * o[op]), np.sum(o[op] ** 2))))
        out.append(dict(session=name, animal=animal, date=date, trial_k=k, trial_id=g.trial_id, position=g.position,
                        n_wide=int(wide.sum()), lr_wide=float(np.median(lv[wide])) if wide.sum() >= 3 else np.nan,
                        dx_img_wide=float(np.median(vx[m][wide])) if wide.sum() >= 3 else np.nan,
                        n_open=int(op.sum()), angle=ang, open_max=float(np.max(o)) if o.size else np.nan))
    return pd.DataFrame(out)


def cmd_deviation(sessions, rule: str, open_px: float, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config
    q = rules()[rule]
    T = pd.concat([session_deviation(s, q, open_px) for s in sessions], ignore_index=True)
    T["phase"] = [config.session_phase(a, d) for a, d in zip(T.animal, T.date)]
    T.to_csv(out / f"jaw_lateral_trials_{rule}.csv", index=False, float_format="%.3f")
    rng = np.random.default_rng(0)

    def ci(v):
        v = v[np.isfinite(v)]
        if v.size < 3:
            return np.nan, np.nan, np.nan
        bs = np.median(rng.choice(v, (2000, v.size)), axis=1)
        return float(np.median(v)), *np.percentile(bs, [2.5, 97.5]).tolist()

    S = []
    for (s, p), g in T.groupby(["session", "position"]):
        for col in ("lr_wide", "angle"):
            m, lo, hi = ci(g[col].to_numpy(float))
            S.append(dict(session=s, animal=g.animal.iloc[0], date=g.date.iloc[0], phase=g.phase.iloc[0],
                          position=p, measure=col, n=int(np.isfinite(g[col]).sum()), median=m, lo=lo, hi=hi))
    for s, g in T.groupby("session"):
        for col in ("lr_wide", "angle"):
            m, lo, hi = ci(g[col].to_numpy(float))
            S.append(dict(session=s, animal=g.animal.iloc[0], date=g.date.iloc[0], phase=g.phase.iloc[0],
                          position="ALL", measure=col, n=int(np.isfinite(g[col]).sum()), median=m, lo=lo, hi=hi))
    # FAR PAIR = the primary readout. Close / centre spouts cover part of the chin and the model then marks the jaw on
    # the visible edge (QC sheet 2026-10-14: close_R +45 / close_L +7 px on PS93 0814, i.e. AWAY from the spout);
    # far_L / far_R sit >= 85 px off the midline and leave the chin clear. Their MEAN cancels any spout-directed jaw
    # movement (side bias); half their DIFFERENCE is that spout-directed part. CI: trials resampled within position.
    for s, g in T.groupby("session"):
        for col in ("lr_wide", "angle"):
            a_ = g.loc[g.position == "far_L", col].dropna().to_numpy(float)
            b_ = g.loc[g.position == "far_R", col].dropna().to_numpy(float)
            if min(a_.size, b_.size) < 3:
                continue
            ma = np.median(rng.choice(a_, (2000, a_.size)), axis=1)
            mb = np.median(rng.choice(b_, (2000, b_.size)), axis=1)
            for lab, est, bs in (("FAR_BIAS", (np.median(a_) + np.median(b_)) / 2, (ma + mb) / 2),
                                 ("FAR_DIR", (np.median(a_) - np.median(b_)) / 2, (ma - mb) / 2)):
                S.append(dict(session=s, animal=g.animal.iloc[0], date=g.date.iloc[0], phase=g.phase.iloc[0],
                              position=lab, measure=col, n=int(min(a_.size, b_.size)), median=float(est),
                              lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
    S = pd.DataFrame(S)
    S.to_csv(out / f"jaw_lateral_summary_{rule}.csv", index=False, float_format="%.2f")
    print(S[S.position.isin(["ALL", "FAR_BIAS", "FAR_DIR"])].round(1).to_string(index=False))

    POS = ["far_L", "close_L", "far_center", "close_center", "close_R", "far_R"]
    sess = list(dict.fromkeys(sessions))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, col, lab in ((axes[0], "lr_wide", f"jaw sideways at opening > {open_px:.0f} px (px)"),
                         (axes[1], "angle", "jaw opening angle (deg)")):
        w = 0.8 / len(POS)
        for j, p in enumerate(POS + ["FAR_BIAS"]):
            sub = S[(S.measure == col) & (S.position == p)].set_index("session").reindex(sess)
            xs = np.arange(len(sess)) + (j - len(POS) / 2) * w * (len(POS) / (len(POS) + 1))
            col_ = "k" if p == "FAR_BIAS" else plt.cm.tab10(j)
            ax.errorbar(xs, sub["median"], yerr=[sub["median"] - sub["lo"], sub["hi"] - sub["median"]], fmt="o",
                        ms=4 if p != "FAR_BIAS" else 8, color=col_, label=p, capsize=2, lw=1)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_ylabel(lab + "\n+ = image-right = MOUSE'S LEFT")
    axes[0].legend(ncol=7, fontsize=8, loc="upper left")
    ph = T.groupby("session").phase.first().reindex(sess)
    axes[1].set_xticks(range(len(sess)))
    axes[1].set_xticklabels([f"{s.replace('_r4', '')}\n{ph[s]}" for s in sess], fontsize=8)
    fig.suptitle(f"Jaw lateral deviation on opening (DLC r3, cleaning {rule}, spout frame; median over trials, "
                 f"95% bootstrap CI)\nblack = FAR_BIAS (mean of far_L, far_R: the chin is unobstructed there) -- close / centre "
                 f"spouts cover the chin and bias the jaw point away from the spout", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"jaw_lateral_{rule}.png", dpi=130)
    print(f"-> {out / f'jaw_lateral_{rule}.png'}")


# --------------------------------------------------------------------------- flips (QC of the bracketed-step rule)

def cmd_flips(sessions, out: Path, n: int = 14, step: float = 30.0, max_ms: float = 200.0) -> None:
    """Side-flip events the proposed `x_flip_step_px` rule would wipe: runs of raw jaw frames entered and left by
    opposite x steps > `step` px within `max_ms`. One row per event: the frame BEFORE the run, the run's MIDDLE frame,
    the frame AFTER -- the rule wipes the middle one, so the sheet asks which side is right."""
    import cv2

    from wfield_local.paths import PathResolver
    rv = PathResolver()
    rng = np.random.default_rng(1)
    events = []
    for name in sessions:
        raw, idx, big, pos, cues, fr = load(name)
        pre, c = clean_jaw(big, cues, rules()["v35"])
        by, bx = c.extra["v34_bmean"]
        src = np.full(len(big), -1)
        src[pos] = idx.src_frame.to_numpy()
        ii = np.flatnonzero((c.fill_method == oc.FILL_NONE) & (src >= 0))
        xs = c.x_final[ii]
        d = np.where(np.diff(ii) <= 2, np.diff(xs), 0.0)
        bigs = np.flatnonzero(np.abs(d) > step)
        max_f = max(int(round(max_ms / 1000 * FPS)), 1)
        for a_, b_ in zip(bigs[:-1], bigs[1:]):
            if np.sign(d[a_]) != np.sign(d[b_]) and ii[b_] - ii[a_] <= max_f:
                k0, km, k1 = ii[a_], ii[(a_ + 1 + b_) // 2], ii[b_ + 1]
                pts = [(int(src[k]), float(pre.x_raw[k]), float(pre.y_raw[k]), float(c.x_final[k] - bx)) for k in
                       (k0, km, k1)]
                events.append((name, int(ii[b_] - ii[a_]), bx + pre.X0, by + pre.Y0, pts))
    print(f"{len(events)} flip events in {', '.join(sessions)}")
    pick = [events[i] for i in sorted(rng.choice(len(events), min(n, len(events)), replace=False))]
    caps, rows, crop, tw = {}, [], (300, 260), 210
    for name, nf, bxi, byi, pts in pick:
        animal, date = name.split("_")[:2]
        if name not in caps:
            vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam4_*.avi"))[0]
            caps[name] = cv2.VideoCapture(str(vid))                         # READ-ONLY
        tiles = []
        for lab, (f, x, y, dx) in zip(("before", f"FLIPPED ({nf} fr) - wiped", "after"), pts):
            caps[name].set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = caps[name].read()
            cv2.drawMarker(im, (int(bxi), int(byi)), (255, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
            cv2.circle(im, (int(x), int(y)), 5, (0, 0, 255) if "wiped" in lab else (255, 255, 0), 2, cv2.LINE_AA)
            x0, y0 = max(int(bxi) - crop[0] // 2, 0), max(int(byi) + 20 - crop[1] // 2, 0)
            t = cv2.resize(im[y0:y0 + crop[1], x0:x0 + crop[0]], (tw, int(tw * crop[1] / crop[0])))
            cv2.putText(t, f"{lab} f{f} dx{dx:+.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                        cv2.LINE_AA)
            tiles.append(t)
        row = np.hstack(tiles)
        head = np.zeros((20, row.shape[1], 3), np.uint8)
        cv2.putText(head, f"{animal} {date[4:]}", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        rows.append(np.vstack([head, row]))
    half = (len(rows) + 1) // 2
    left, right = np.vstack(rows[:half]), np.vstack(rows[half:] or [np.zeros_like(rows[0])])
    if right.shape[0] < left.shape[0]:
        right = np.vstack([right, np.zeros((left.shape[0] - right.shape[0], right.shape[1], 3), np.uint8)])
    sheet = np.hstack([left, np.full((left.shape[0], 8, 3), 60, np.uint8), right])
    top = np.zeros((28, sheet.shape[1], 3), np.uint8)
    cv2.putText(top, "Jaw side-flips: cyan = kept, RED = the run the proposed flip rule wipes. Which side is right?",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "jaw_flip_qc.png"), np.vstack([top, sheet]))
    print(f"-> {out / 'jaw_flip_qc.png'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("ceiling", "deviation", "flips"))
    ap.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    ap.add_argument("--qc", nargs="*", default=["PS93_20260814", "PS95_20260917_r4", "PS94_20260921_r4"])
    ap.add_argument("--rule", default="v35", choices=("v34", "v35"))
    ap.add_argument("--open-px", type=float, default=50.0)
    a = ap.parse_args(argv)
    out = poses_root() / "jaw_lateral"
    out.mkdir(exist_ok=True)
    if a.cmd == "ceiling":
        cmd_ceiling(a.sessions, set(a.qc), out)
    elif a.cmd == "flips":
        cmd_flips(a.sessions, out)
    else:
        cmd_deviation(a.sessions, a.rule, a.open_px, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

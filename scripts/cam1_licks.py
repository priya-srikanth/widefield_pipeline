"""cam1 2-D lick events on the production O2 output (`wfield_local.cam1_licks`; Priya 10-19) and a BLIND review sample.

    python -m scripts.cam1_licks run [ANIMAL:YYYYMMDD ...] [--jobs 3]     # every session with cam1 + cam4 O2 output
    python -m scripts.cam1_licks review [--per 12] [--seed 0]              # candidates.json + key.csv for human review

``run`` per session: cam1 tongue cleaned with the cam1 rules (`orofacial_clean_cam1`; origin = the cam1 spout-line mouth,
`spout_frame.from_medians_lr`), events = `cam1_licks.detect`; each event flagged ``cam4_matched`` (overlaps a v36 cam4 lick
window, same trial), ``in_3d`` (overlaps a combined-track 3-D event, `pose3d.lick_events` on the session's
``pose3d/<s>_frames.csv.gz``; NaN when that file is missing), ``contact`` (DAQ spout contact within +- 20 ms) and
``jaw_only_trial`` (the trial is a cam4 "jaw movement without a lick" trial, `tongue_jaw_mismatch.csv`, and the event lies
in its 70 ms .. quiet-window-end span). Sizes in cam1 px and mm (`cam1_licks.mm_per_px`, a size scale only).
Writes `session_poses/cam1_licks/<a>_<d>_events.csv`, `sessions.csv` (per session x position) and prints the
per-animal x epoch table.

``review``: blind sample over the events tables (cam1-only ``lips`` / cam1-only ``out`` x pre / acute, ``--per`` each),
plus cam4-matched ``lips`` events, contact-lick positive controls and NULL controls (random post-cue times with no cam1,
cam4 or 3-D event within 150 ms). Each candidate = two strips (cam1 above, cam4 below) of 7 mouth crops -24 .. +24 ms
around the event peak, NO markers. Writes `session_poses/cam1_licks/review/{candidates.json, key.csv}`; key.csv (the
stratum of each id) stays out of the review page.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROD = {"cam4": "lp_r4_prod_20261015", "cam1": "lp_cam1_ctx_prod_20261019"}
V36 = "lp_r4_heatmap_20261013__clean_v36"
OFFSETS_MS = (-24, -16, -8, 0, 8, 16, 24)
CROP, TILE_W = (300, 240), 150


def _out(rv) -> Path:
    p = Path(rv.root("dlc")) / "session_poses" / "cam1_licks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cam1_frame(flat, ix):
    from wfield_local import spout_frame as SF
    med = SF.ruler_medians(flat, ix)
    fr = SF.from_medians_lr(med)
    sgn = fr.mouse_left_sign or 1

    def to_frame(x, y):
        dx, dy = np.asarray(x, float) - fr.origin[0], np.asarray(y, float) - fr.origin[1]
        return -(dx * fr.ap_axis[0] + dy * fr.ap_axis[1]), sgn * (dx * fr.lr_axis[0] + dy * fr.lr_axis[1])
    return fr, med, to_frame


def run_session(s: str) -> dict:
    import scripts.pose_kinematics_demo as D
    from scripts.apparatus_mouth import _rig
    from wfield_local import cam1_licks as C1
    from wfield_local import epochs
    from wfield_local import o2_inference as oi
    from wfield_local import orofacial_clean as oc
    from wfield_local import pose3d as P3
    from wfield_local import spout_world as SW
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = s.split(":")
    row = {"session": f"{animal}_{date}"}
    c1 = oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    if c1 is None:
        return {**row, "error": "no cam1 output"}
    ix, flat = c1[0].reset_index(drop=True), c1[1].reset_index(drop=True)
    fr, med, to_frame = _cam1_frame(flat, ix)
    cols = [c for c in flat.columns if c.startswith(("tongue_", "jaw_"))]
    c, pos, _ = oc.clean_windows(flat[cols], ix, "tongue", x0y0=tuple(fr.origin), cam="cam1")
    x, y, fm = c.x_final[pos] + c.X0, c.y_final[pos] + c.Y0, c.fill_method[pos]
    out_, lat = to_frame(x, y)
    src = ix.src_frame.to_numpy() + (ix.video.astype("category").cat.codes.to_numpy().astype(np.int64) * 10**8 if "video" in ix else 0)
    E = C1.detect(out_, lat, fm != oc.FILL_BASELINE, ix.trial_id, ix.t_ms, src, ix.stop_ms, ix.position)
    geo = _rig(rv, animal, date)["geometry"]
    lat_pos = {p: float(to_frame(*med[p][:2])[1]) for p in med}
    ml_pos = {p: float(SW.commanded_world(geo, p)[0]) for p in med}
    k = C1.mm_per_px(lat_pos, ml_pos) if {"far_L", "far_R"} <= set(med) else np.nan
    for cname in ("peak_dist", "peak_out", "max_out"):
        E[f"{cname}_mm"] = E[f"{cname}_px"] * k
    sd = Path(rv.root("orofacial")) / V36 / "sessions" / animal / date
    if (sd / "licks.csv").exists():
        C4 = pd.read_csv(sd / "licks.csv")
        E["cam4_matched"] = C1.overlaps(E, C4)
    else:
        E["cam4_matched"] = np.nan
    f3 = Path(rv.root("dlc")) / "session_poses" / "pose3d" / f"{animal}_{date}_frames.csv.gz"
    if f3.exists():
        S = pd.read_csv(f3, usecols=["trial_id", "t_ms", "stop_ms", "position", "tongue_ml", "tongue_ap", "tongue_dv"])
        E["in_3d"] = C1.overlaps(E, P3.lick_events(S.drop_duplicates(["trial_id", "t_ms"])))
    else:
        E["in_3d"] = np.nan
    E["contact"] = C1.contact_flags(E, D.daq_contacts_ms(animal, date, TW.trial_bounds(animal, date, rv), rv))
    E["jaw_only_trial"] = False
    if (sd / "tongue_jaw_mismatch.csv").exists():
        M = pd.read_csv(sd / "tongue_jaw_mismatch.csv")
        M = M[M.candidate_no_lick_with_jaw_move.astype(bool)].set_index("trial_id").quiet_win_end_ms
        E["jaw_only_trial"] = [bool(r.trial_id in M.index and r.off_ms >= 70 and r.on_ms <= M[r.trial_id])
                               for r in E.itertuples()]
        row["jaw_only_trials"] = int(len(M))
        row["jaw_only_trials_with_cam1_tongue"] = int(E.loc[E.jaw_only_trial, "trial_id"].nunique())
    E["animal"], E["date"], E["session"] = animal, date, f"{animal}_{date}"
    E["epoch"] = epochs.epoch_of(f"{animal}_{date[4:]}")
    E["mm_per_px"] = k
    E.to_csv(_out(rv) / f"{animal}_{date}_events.csv", index=False, float_format="%.4g")
    row.update({"epoch": E.epoch.iloc[0] if len(E) else None, "n_trials": int(ix.trial_id.nunique()), "events": len(E),
                "mm_per_px": round(k, 4)})
    return row


def _worker(s):
    return run_session(s)


def summarise(d: Path) -> pd.DataFrame:
    E = pd.concat([pd.read_csv(f, dtype={"date": str}) for f in sorted(d.glob("PS*_events.csv"))], ignore_index=True)
    for c in ("cam4_matched", "in_3d", "contact", "jaw_only_trial"):
        E[c] = E[c].astype("boolean")
    E["cam1_only"] = ~E.cam4_matched.fillna(False) & ~E.in_3d.fillna(False)
    E["incomplete"] = ~E.contact
    g = E.groupby(["animal", "epoch", "session", "position"])
    T = pd.DataFrame({"events": g.size(), "lips": g.size_class.apply(lambda v: (v == "lips").sum()),
                      "cam1_only": g.cam1_only.sum(), "cam1_only_lips": g.apply(lambda x: (x.cam1_only & (x.size_class == "lips")).sum()),
                      "no_contact": g.incomplete.sum(), "cam4_matched": g.cam4_matched.sum(), "in_3d": g.in_3d.sum(),
                      "on_jaw_only_trials": g.jaw_only_trial.sum(),
                      "peak_dist_mm_cam1_only": g.apply(lambda x: x.loc[x.cam1_only, "peak_dist_mm"].median()),
                      "peak_dist_mm_matched": g.apply(lambda x: x.loc[x.cam4_matched.fillna(False), "peak_dist_mm"].median())})
    return E, T.reset_index()


def review(rv, per: int, seed: int) -> Path:
    import cv2

    from wfield_local import o2_inference as oi
    from wfield_local import spout_frame as SF
    d = _out(rv)
    E, _ = summarise(d)
    rng = np.random.default_rng(seed)
    E = E[E.epoch.isin(["pre", "acute"])]
    picks = []

    def take(m, n, stratum):
        g = E[m]
        if len(g):
            picks.append(g.sample(min(n, len(g)), random_state=int(rng.integers(1 << 30))).assign(stratum=stratum))
    for ep in ("pre", "acute"):
        e = E.epoch == ep
        take(e & E.cam1_only & (E.size_class == "lips"), per, f"cam1_only_lips_{ep}")
        take(e & E.cam1_only & (E.size_class == "out"), per, f"cam1_only_out_{ep}")
    take(E.cam4_matched.fillna(False) & (E.size_class == "lips"), max(per // 2, 4), "cam4_matched_lips")
    take(E.cam4_matched.fillna(False) & E.contact, max(per // 2, 4), "control_contact")
    P = pd.concat(picks, ignore_index=True)
    nulls = []                                       # random post-cue times with no event of any kind within 150 ms
    for _s, g in E.groupby("session"):
        nulls.append(g.sample(min(2, len(g)), random_state=int(rng.integers(1 << 30))).assign(stratum="control_null"))
    N = pd.concat(nulls, ignore_index=True).sample(max(per // 2, 4), random_state=seed)
    cands, keys = [], []
    for s, g in pd.concat([P, N], ignore_index=True).groupby("session"):
        animal, date = s.split("_")
        c1 = oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
        c4 = oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
        fr1, _, _ = _cam1_frame(c1[1].reset_index(drop=True), c1[0].reset_index(drop=True))
        o4 = SF.from_medians(SF.ruler_medians(c4[1], c4[0])).origin
        ev_all = E[E.session == s]
        for r in g.itertuples():
            t0 = r.t_peak_ms
            if r.stratum == "control_null":                       # move to a quiet time in the same trial
                busy = ev_all[ev_all.trial_id == r.trial_id][["on_ms", "off_ms"]].to_numpy(float)
                stop = float(c1[0][c1[0].trial_id == r.trial_id].stop_ms.iloc[0])
                for _ in range(200):
                    t0 = float(rng.uniform(100, max(stop - 50, 150)))
                    if not ((busy[:, 0] <= t0 + 150) & (busy[:, 1] >= t0 - 150)).any():
                        break
                else:
                    continue
            strips = []
            for cam, (ix, _), origin, shift in (("cam1", c1, fr1.origin, -40 * np.asarray(fr1.ap_axis)),
                                                ("cam4", c4, o4, np.array([0.0, 40.0]))):
                cap = cv2.VideoCapture(str(_video(rv, animal, date, cam, ix, r.trial_id)))
                h = ix[ix.trial_id == r.trial_id]
                tiles = []
                for off in OFFSETS_MS:
                    i = int(np.argmin(np.abs(h.t_ms.to_numpy(float) - (t0 + off))))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(h.src_frame.iloc[i]))
                    ok, im = cap.read()
                    if not ok:
                        im = np.zeros((1024, 1280, 3), np.uint8)
                    cx, cy = (np.asarray(origin, float) + shift).astype(int)
                    x0, y0 = max(cx - CROP[0] // 2, 0), max(cy - CROP[1] // 2, 0)
                    t = cv2.resize(im[y0:y0 + CROP[1], x0:x0 + CROP[0]], (TILE_W, int(TILE_W * CROP[1] / CROP[0])))
                    cv2.putText(t, f"{off:+d}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(t)
                cap.release()
                strips.append(np.hstack(tiles))
            ok, buf = cv2.imencode(".jpg", np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 82])
            cid = f"c{len(cands):03d}"
            cands.append({"id": cid, "position": r.position, "img": base64.b64encode(buf.tobytes()).decode("ascii")})
            keys.append({"id": cid, "stratum": r.stratum, "session": s, "trial_id": r.trial_id, "t_ms": round(t0, 1),
                         "size_class": r.size_class, "peak_dist_mm": r.peak_dist_mm, "contact": r.contact})
    order = rng.permutation(len(cands))
    rd = d / "review"
    rd.mkdir(exist_ok=True)
    (rd / "candidates.json").write_text(json.dumps([cands[i] for i in order]))
    pd.DataFrame([keys[i] for i in order]).to_csv(rd / "key.csv", index=False)
    return rd


def _video(rv, animal, date, cam, ix, tid):
    from wfield_local import o2_inference as oi
    vids = oi.videos_for(rv, animal, date, cam)
    if "video" in ix:
        name = str(ix[ix.trial_id == tid].video.iloc[0])
        hit = [v for v in vids if v.stem == name or v.name == name or name in v.stem]
        if hit:
            return hit[0]
    return vids[0]


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "review"])
    ap.add_argument("sessions", nargs="*", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rv = PathResolver()
    d = _out(rv)
    if a.mode == "review":
        print("->", review(rv, a.per, a.seed))
        return 0
    sess = a.sessions
    if not sess:
        have = [{p.name for p in (Path(rv.root("dlc")) / "o2" / b / "out").glob("PS*_2026*") if p.is_dir()}
                for b in PROD.values()]
        sess = [x.replace("_", ":", 1) for x in sorted(set.intersection(*have))]
    res, fails = ak.fan_sessions(sess, _worker, jobs=min(a.jobs, len(sess)))
    rows = [v for _, v in res] + [{"session": it.replace(":", "_"), "error": e} for it, e in fails]
    E, T = summarise(d)
    T.to_csv(d / "sessions.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    prev = pd.read_csv(d / "run_summary.csv") if (d / "run_summary.csv").exists() else pd.DataFrame()
    tr = pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates("session", keep="last")    # a subset run keeps the rest
    tr.to_csv(d / "run_summary.csv", index=False)
    tr = tr[tr.get("error", pd.Series(np.nan, index=tr.index)).isna()].set_index("session")
    g = E.groupby(["animal", "epoch"])
    A = pd.DataFrame({"sessions": g.session.nunique(), "events": g.size(),
                      "lips_frac": g.size_class.apply(lambda v: (v == "lips").mean()),
                      "cam4_matched": g.cam4_matched.mean(), "in_3d": g.in_3d.mean(), "cam1_only": g.cam1_only.mean(),
                      "cam1_only_lips": g.apply(lambda x: (x.cam1_only & (x.size_class == "lips")).mean()),
                      "no_contact": g.incomplete.mean(),
                      "cam1_only_no_contact": g.apply(lambda x: (x.cam1_only & x.incomplete).mean()),
                      "dist_mm_cam1_only": g.apply(lambda x: x.loc[x.cam1_only, "peak_dist_mm"].median()),
                      "dist_mm_cam4_matched": g.apply(lambda x: x.loc[x.cam4_matched.fillna(False), "peak_dist_mm"].median())})
    if "jaw_only_trials" in tr:
        j = tr[["jaw_only_trials", "jaw_only_trials_with_cam1_tongue"]].join(E.groupby("session")[["animal", "epoch"]].first()).groupby(["animal", "epoch"])
        A["jaw_only_trials"] = j.jaw_only_trials.sum()
        A["jaw_only_with_cam1_tongue"] = j.jaw_only_trials_with_cam1_tongue.sum()
    print(A.round(3).to_string())
    if fails:
        print("FAILED:", fails)
    print("->", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

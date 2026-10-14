"""Candidate licks for HUMAN review (Priya, 2026-10-14): blind sample of licks WITHOUT spout contact detected by any
cam4 model, plus a few contact licks as hidden positive controls, each as a strip of mouth crops around the lick's
peak. Output: <out>/candidates.json (id, session, position, strip as base64 JPEG) + <out>/key.csv (which models
detected each candidate -- kept OUT of the review page so the review is blind).

    PYTHONPATH=. python -m scripts.lick_review_candidates PS93:20260814 PS93:20260821 --out <dir> [--per-session 40]

Why: the models differ most on no-contact licks (incomplete licks matter most post-stroke), and no automatic measure
says whether a no-contact detection is a real incomplete lick or a false one. Candidates = the union over models
(detections within 40 ms in a trial are one candidate), sampled per session balanced over spout positions and over
how many models found them (1 / 2-3 / 4+), so disagreements are well represented.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

L = Path("C:/Users/SabatiniLab/lp_stage/r3_vs_r4_lp")
OFFSETS_MS = (-24, -16, -8, 0, 8, 16, 24)          # 7 frames, every 2nd frame at 250 fps, around the peak
CROP = (300, 260)                                  # w, h around the mouth (px, native)
TILE_W = 150


def models_for(d: Path, date: str) -> dict:
    return {"DLC r3": d / "windows_DLC.csv", "DLC r4": d / "r4_eval" / "windows_DLC_r4.csv",
            "LP r3": L / "r3" / f"PS93_{date}_windows.csv", "LP r4": L / "r4heatmap" / f"PS93_{date}_windows.csv",
            "LP r4 ctx": L / "r4mhcrnn_sup_win" / f"PS93_{date}_windows.csv",
            "EKS r4+ctx": L / "eks_r4_heat_ctx" / f"PS93_{date}_windows.csv"}


def cluster(per_model: dict, tol_ms: float = 40.0) -> pd.DataFrame:
    """One row per candidate: (trial, mean peak time, position, set of models), from per-model lick tables."""
    rows = []
    for m, pl in per_model.items():
        for r in pl.itertuples():
            rows.append((int(r.trial_id), float(r.t_ms), str(r.position), m, bool(r.contact) if r.contact == r.contact else False))
    df = pd.DataFrame(rows, columns=["trial_id", "t_ms", "position", "model", "contact"]).sort_values(["trial_id", "t_ms"])
    out = []
    for _tid, g in df.groupby("trial_id"):
        cur = []
        for r in g.itertuples():
            if cur and r.t_ms - cur[-1].t_ms > tol_ms:
                out.append(cur); cur = []
            cur.append(r)
        if cur:
            out.append(cur)
    return pd.DataFrame([{"trial_id": c[0].trial_id, "t_ms": float(np.median([r.t_ms for r in c])), "position": c[0].position,
                          "models": sorted({r.model for r in c}), "any_contact": any(r.contact for r in c)} for c in out])


def strip(cap, idx: pd.DataFrame, tid: int, t_ms: float, mouth) -> str:
    g = idx[idx.trial_id == tid]
    tiles = []
    for off in OFFSETS_MS:
        row = g.index[int(np.argmin(np.abs(g.t_ms.to_numpy() - (t_ms + off))))]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row))
        ok, im = cap.read()
        if not ok:
            im = np.zeros((680, 680, 3), np.uint8)
        cx, cy = int(mouth[0]), int(mouth[1]) + 35                  # mouth zero, shifted down to include the reach
        x0, y0 = max(cx - CROP[0] // 2, 0), max(cy - CROP[1] // 2, 0)
        t = im[y0:y0 + CROP[1], x0:x0 + CROP[0]]
        t = cv2.resize(t, (TILE_W, int(TILE_W * CROP[1] / CROP[0])))
        cv2.putText(t, f"{off:+d}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    ok, buf = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main(argv=None) -> int:
    import scripts.pose_kinematics_demo as D
    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local import trial_windows as TW
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-session", type=int, default=40)
    ap.add_argument("--controls", type=int, default=4)
    a = ap.parse_args(argv)
    rv = PathResolver()
    rng = np.random.default_rng(14)
    cands, keys = [], []
    for spec in a.sessions:
        animal, date, d = session_dir(rv, spec)
        idx = read_index(d)
        raw = oc.read_pose(d / "windows_DLC.csv").iloc[:len(idx)]
        spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
        fr = SF.from_medians(SF.position_medians(raw[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
        b = TW.trial_bounds(animal, date, rv)
        st, ct = dict(zip(b.trial_id, b.stop_s - b.cue_s)), D.daq_contacts_ms(animal, date, b, rv)
        per = {m: D.run(f, idx, fr, st, ct)[0].per_lick for m, f in models_for(d, date).items() if Path(f).exists()}
        C = cluster(per)
        C["n_models"] = C.models.map(len)
        nc = C[~C.any_contact].copy()
        nc["tier"] = pd.cut(nc.n_models, [0, 1, 3, 99], labels=["1", "2-3", "4+"])
        picks = []
        per_cell = max(1, a.per_session // (3 * 6))
        for (_t, _p), g in nc.groupby(["tier", "position"], observed=True):
            picks += list(g.sample(min(len(g), per_cell), random_state=int(rng.integers(1e9))).index)
        rest = nc.drop(picks)
        if len(picks) < a.per_session and len(rest):
            picks += list(rest.sample(min(len(rest), a.per_session - len(picks)), random_state=14).index)
        ctrl = C[C.any_contact & (C.n_models == len(per))]
        sel = pd.concat([nc.loc[picks].assign(kind="no_contact"),
                         ctrl.sample(min(len(ctrl), a.controls), random_state=14).assign(kind="control")])
        cap = cv2.VideoCapture(str(d / "windows.mp4"))
        for r in sel.itertuples():
            cid = f"{animal}_{date}_t{r.trial_id}_{int(round(r.t_ms))}"
            cands.append({"id": cid, "session": f"{animal} {date}", "position": r.position,
                          "strip": strip(cap, idx, r.trial_id, r.t_ms, fr.origin)})
            keys.append({"id": cid, "kind": r.kind, "trial_id": r.trial_id, "t_ms": round(r.t_ms, 1),
                         "position": r.position, "models": "|".join(r.models), "n_models": len(r.models)})
        cap.release()
        print(f"{animal} {date}: {len(nc)} no-contact candidates -> {len(picks)} sampled + {min(len(ctrl), a.controls)} controls")
    order = rng.permutation(len(cands))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "candidates.json").write_text(json.dumps([cands[i] for i in order]))
    pd.DataFrame([keys[i] for i in order]).to_csv(a.out / "key.csv", index=False)
    print(f"{len(cands)} candidates -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

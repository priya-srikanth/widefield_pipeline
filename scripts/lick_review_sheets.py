"""Image sheets from the human lick review (Priya, 2026-10-14): (1) LP r4 FALSE POSITIVES (judged "not a lick", LP r4
detected) and (2) EKS FALSE NEGATIVES (judged real, LP r4 detected, EKS r4+ctx did not). Each row: 5 mouth crops
(-16 .. +16 ms around the candidate's peak) with the LP r4 (orange circle) and EKS (green diamond) tongue points
drawn where likelihood >= 0.4 (the cleaning cut; a missing marker = that model says "not visible").

    PYTHONPATH=. python -m scripts.lick_review_sheets --review <lp_stage/lick_review>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

L = Path("C:/Users/SabatiniLab/lp_stage/r3_vs_r4_lp")
OFFS = (-16, -8, 0, 8, 16)
CROP, TW_ = (300, 260), 190
MODELS = {"LP r4": ("r4heatmap", (0, 165, 255), "o"), "EKS": ("eks_r4_heat_ctx", (60, 220, 60), "d")}


def load(path):
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = df.columns.droplevel(0)
    return df["tongue"][["x", "y", "likelihood"]].to_numpy(float)


def mark(img, x, y, colour, shape):
    if shape == "o":
        cv2.circle(img, (int(x), int(y)), 6, colour, 2, cv2.LINE_AA)
    else:
        pts = np.array([[x, y - 7], [x + 7, y], [x, y + 7], [x - 7, y]], np.int32)
        cv2.polylines(img, [pts], True, colour, 2, cv2.LINE_AA)


def row(cap, idx, P, tid, t_ms, mouth, label):
    g = idx[idx.trial_id == tid]
    tiles = []
    for off in OFFS:
        r = g.index[int(np.argmin(np.abs(g.t_ms.to_numpy() - (t_ms + off))))]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r))
        ok, im = cap.read()
        cx, cy = int(mouth[0]), int(mouth[1]) + 35
        x0, y0 = max(cx - CROP[0] // 2, 0), max(cy - CROP[1] // 2, 0)
        for name, (_, col, shp) in MODELS.items():
            x, y, p = P[name][int(r)]
            if p >= 0.4:
                mark(im, x, y, col, shp)
        t = cv2.resize(im[y0:y0 + CROP[1], x0:x0 + CROP[0]], (TW_, int(TW_ * CROP[1] / CROP[0])))
        txt = f"{off:+d} ms  LP{P['LP r4'][int(r)][2]:.2f} EKS{P['EKS'][int(r)][2]:.2f}"
        cv2.putText(t, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    strip = np.hstack(tiles)
    head = np.zeros((22, strip.shape[1], 3), np.uint8)
    cv2.putText(head, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, strip])


def main(argv=None) -> int:
    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", type=Path, required=True)
    a = ap.parse_args(argv)
    rv = PathResolver()
    d = pd.read_csv(a.review / "review_scored.csv")
    nc = d[d.kind == "no_contact"]
    has = lambda m: nc.models.str.split("|").apply(lambda s: m in s)        # noqa: E731
    sets = {"lp_r4_false_positives": nc[(nc.label == "not") & has("LP r4")],
            "eks_false_negatives": nc[(nc.label == "real") & has("LP r4") & ~has("EKS r4+ctx")]}
    ctx = {}
    for name, sub in sets.items():
        rows = []
        for r in sub.sort_values("id").itertuples():
            animal, date = r.id.split("_")[0], r.id.split("_")[1]
            if date not in ctx:
                _, _, sd = session_dir(rv, f"{animal}:{date}")
                idx = read_index(sd)
                raw = oc.read_pose(sd / "windows_DLC.csv").iloc[:len(idx)]
                spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
                fr = SF.from_medians(SF.position_medians(raw[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
                P = {m: load(L / sub_ / f"{animal}_{date}_windows.csv") for m, (sub_, _, _) in MODELS.items()}
                ctx[date] = (cv2.VideoCapture(str(sd / "windows.mp4")), idx, P, fr.origin)
            cap, idx, P, mouth = ctx[date]
            found = r.models.replace("|", ", ")
            rows.append(row(cap, idx, P, r.trial_id, r.t_ms, mouth,
                            f"{animal} {date[4:]} trial {r.trial_id} {r.position}  peak {r.t_ms:.0f} ms  |  found by: {found}"))
        if rows:
            out = a.review / f"{name}.png"
            cv2.imwrite(str(out), np.vstack(rows))
            print(f"{name}: {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Register every session's reference face onto ONE master face per camera (docs/FACE_RHYTHM_PLAN.md step 3; Priya 10-21:
"morph across animals to fit a single 'master' mouse face mesh for each camera"; master = "whichever you recommend").

    <fr env python> scripts/face_rhythm_register.py [--root <dlc>/face_rhythm/cam4] [--jobs 4]

Inputs: ``<root>/frames/<a>_<d>.npz`` from `scripts.face_rhythm_prep frames` (median rest face + LP landmarks).
1. ANIMAL TEMPLATE = the animal's MEDOID among its AUGUST PRE-STROKE sessions (``<root>/pre_august_sessions.json``,
   `wfield_local.epochs.epoch_of` == pre, dated >= 0801): highest mean correlation (CLAHE images, 4x downsampled) with
   the other candidates -- a real, sharp, pre-stroke face, not an average. (10-21, FaceRhythm window + Priya: the first
   run picked post-stroke templates, PS93's acute face as the master; a lesioned face may rest asymmetrically and every
   session's mesh would inherit it.)
2. MASTER = the medoid of the animal templates (``--master-exclude``: PS93, the animal with the right orofacial deficit),
   after a landmark similarity registration of each pair.
3. Fields: a COARSE similarity transform (rotation + scale + shift) from the LP landmarks the two images share (rest
   nose, rest jaw, the per-position spout tips -- fixed relative to the mouth), then face-rhythm's DeepFlow non-rigid
   (`rois.Image_Aligner.fit_nonrigid`) on top: session -> its animal template, animal template -> master. (10-21: ECC
   affine from a cold start did not converge across animals; the landmarks give the start it needs.) The coarse fit
   WEIGHTS the face (nose, jaw x FACE_WEIGHT) over the apparatus (spout tips). DeepFlow runs on the session already
   warped by the similarity and the two fields are CHAINED here: face-rhythm's ``remappingIdx_init`` applied the
   initial field twice (displacement 32 -> 58 px, landmark error 2 -> 28 px; checked 10-21). Measured on 6 sessions:
   similarity alone nose / jaw 2-22 px, + DeepFlow 1-6 px (spout tips 2 -> 7-13 px: apparatus, not anatomy).
   FALLBACK: when DeepFlow makes the face landmarks worse than the similarity (a June session diverged to ~60 px), the
   similarity field is kept (``method`` in qc_fields.csv). A field R (H, W, 2) maps OUTPUT (template) px to SOURCE px, so a master
   point p lands in a session at R_session(R_animal(p)) (`face_rhythm.helpers.remap_points` samples R at p).
QC (quantitative): the master session's LP landmarks (rest nose, rest jaw, per-position spout tips) pushed into every
session through its composed field vs that session's OWN LP landmarks (px), next to the error with no registration.
Outputs in ``<root>/register/``: ``fields/<s>.npy`` (composed master -> session field, float32 H x W x 2), ``master.npz``
(image, session, landmarks), ``templates.json``, ``qc_landmarks.csv``, ``qc_overlay_<animal>.png``.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(r"M:\MICROSCOPE\Priya\DeepLabCut\Widefield\face_rhythm\cam4")


def prep(img: np.ndarray) -> np.ndarray:
    import cv2
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img.astype(np.uint8))
    return cl.astype(np.float32) / 255.0


def _ncc(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def medoid(ims: dict) -> str:
    import cv2
    keys = sorted(ims)
    small = {k: cv2.resize(ims[k], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) for k in keys}
    score = {k: np.mean([_ncc(small[k], small[j]) for j in keys if j != k]) if len(keys) > 1 else 1.0 for k in keys}
    return max(score, key=score.get)


FACE = ("nose", "jaw", "eye_l", "eye_r")
#: eyes: Priya's clicks on the master + the animal templates (``register/eye_landmarks.json``, 10-21) -- every LP landmark
#: is in the lower face, so nothing else pins the eyes in the template -> master step (the eyes fell outside some ROIs)
EYE_R = 45.0
#: eyes are NOT kept raw (10-21: raw eye zones folded -> bunched points around the eyes, median 0 -> 24 per session);
#: they act through the coarse fit only. Set to ("eye_l", "eye_r") to restore raw eye zones.
RAW_EYES: tuple = ()
LOCAL_WIN, LOCAL_MARGIN = 41, 0.05
SMOOTH_PX, SHELL_PX, MOUTH_R, INNER_SMOOTH_PX = 25.0, 50.0, 80.0, 8.0
FOLD_DET, FOLD_GROW_PX, FOLD_SMOOTH_PX = 0.6, 25, 20.0
FACE_WEIGHT = 3


def _shared(lm_t: dict, lm_m: dict) -> list:
    return [k for k in lm_t if k in lm_m and np.isfinite(lm_t[k]).all() and np.isfinite(lm_m[k]).all()]


def similarity_field(lm_t: dict, lm_m: dict, shape) -> np.ndarray:
    """(H, W, 2) field template px -> moving px from the shared LP landmarks (similarity: rotation, scale, shift);
    face landmarks repeated FACE_WEIGHT times so the face, not the spout geometry, sets the fit."""
    import cv2
    keys = _shared(lm_t, lm_m)
    keys = keys + [k for k in keys if k in FACE] * (FACE_WEIGHT - 1)     # eyes weighted like nose / jaw (eyes-once was worse, 10-21)
    A, _ = cv2.estimateAffinePartial2D(np.array([lm_t[k] for k in keys], np.float32),
                                       np.array([lm_m[k] for k in keys], np.float32), method=cv2.LMEDS)
    H, W = shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return np.stack([A[0, 0] * gx + A[0, 1] * gy + A[0, 2], A[1, 0] * gx + A[1, 1] * gy + A[1, 2]], -1).astype(np.float32)


def face_err(F: np.ndarray, lm_t: dict, lm_m: dict) -> float:
    e = [np.hypot(*(sample(F, lm_t[k])[0] - lm_m[k])) for k in FACE if k in _shared(lm_t, lm_m)]
    return float(np.max(e)) if e else np.nan


def raw_zones(shape, lm: dict) -> np.ndarray:
    """(H, W) weight of the RAW field: the mouth (MOUTH_R around the rest jaw) and, where clicked, each eye (EYE_R)."""
    w = mouth_weight(shape, lm.get("jaw"))
    H, W = shape
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    for k in RAW_EYES:
        e = lm.get(k)
        if e is not None and np.isfinite(e).all():
            w = np.maximum(w, np.clip((EYE_R + SHELL_PX - np.hypot(gx - e[0], gy - e[1])) / SHELL_PX, 0, 1))
    return w.astype(np.float32)


def local_ncc(a: np.ndarray, b: np.ndarray, win: int = LOCAL_WIN) -> np.ndarray:
    """Windowed normalised cross-correlation map of two images."""
    import cv2
    k = (win, win)
    ma, mb = cv2.boxFilter(a, -1, k), cv2.boxFilter(b, -1, k)
    va = cv2.boxFilter(a * a, -1, k) - ma * ma
    vb = cv2.boxFilter(b * b, -1, k) - mb * mb
    cab = cv2.boxFilter(a * b, -1, k) - ma * mb
    return cab / np.sqrt(np.clip(va * vb, 1e-8, None))


def mouth_weight(shape, mouth) -> np.ndarray:
    """(H, W) 1 within MOUTH_R of the resting jaw, ramping to 0 over SHELL_PX: the RAW field is kept there."""
    H, W = shape
    if mouth is None or not np.isfinite(mouth).all():
        return np.zeros((H, W), np.float32)
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    return np.clip((MOUTH_R + SHELL_PX - np.hypot(gx - mouth[0], gy - mouth[1])) / SHELL_PX, 0, 1).astype(np.float32)


def shell_weight(poly, shape, mouth=None) -> np.ndarray:
    """(H, W) weight of the RAW non-rigid field: 1 inside the face polygon deeper than SHELL_PX, ramping to 0 at the
    contour, 0 outside (the OUTER SHELL gets the smoothed field); EXCEPT within MOUTH_R of the resting jaw (``mouth``),
    kept raw (Priya: keep the edge about the mouth, bottom centre)."""
    import cv2
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    w = np.clip(d / SHELL_PX, 0, 1)
    if mouth is not None and np.isfinite(mouth).all():
        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
        w = np.maximum(w, np.clip((MOUTH_R + SHELL_PX - np.hypot(gx - mouth[0], gy - mouth[1])) / SHELL_PX, 0, 1))
    return w.astype(np.float32)


def register(template: np.ndarray, moving: np.ndarray, lm_t: dict, lm_m: dict, poly_t=None) -> tuple[np.ndarray, str]:
    """((H, W, 2) field template px -> moving px, method): landmark similarity, then DeepFlow on the
    similarity-warped image, CHAINED here; the similarity alone when DeepFlow worsens the face landmarks."""
    import cv2

    import face_rhythm as fr
    S = similarity_field(lm_t, lm_m, template.shape)
    warped = cv2.remap(moving, S[..., 0], S[..., 1], cv2.INTER_LINEAR)
    al = fr.rois.Image_Aligner(verbose=False)
    D = np.asarray(al.fit_nonrigid(template=template, ims_moving=[warped], template_method="image",
                                   mode_transform="createOptFlow_DeepFlow"))[0].astype(np.float32)
    # OUTER-SHELL SMOOTHING (10-21, Priya: warped ROI contours wiggled, PS94_0823 bulged; "the inner alignment is good";
    # "an outer shell based choice"): DeepFlow is noisy where the face has little texture (outer fur). The raw field is
    # kept INSIDE the master face ROI (carried into this template), blended across a SHELL_PX band inside its contour to
    # a SMOOTH_PX Gaussian of it, which is used outside. (A uniform 10-20 px smoothing cost PS92 / PS93 face accuracy:
    # jaw 3-4 -> 6-7 px; a radius around the nose did not follow the face.) No ROI yet -> raw field everywhere.
    if poly_t is not None:
        H, W = template.shape
        grid = np.stack(np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)), -1)
        disp = D - grid
        # three zones (10-21): MOUTH raw; face INTERIOR lightly smoothed (INNER_SMOOTH_PX: DeepFlow fold seams squeezed
        # rows of points together, worst in PS93's template); OUTER SHELL / outside SMOOTH_PX
        wm = raw_zones((H, W), lm_t)[..., None]
        ws = shell_weight(poly_t, (H, W))[..., None]
        inner = cv2.GaussianBlur(disp, (0, 0), INNER_SMOOTH_PX)
        outer = cv2.GaussianBlur(disp, (0, 0), SMOOTH_PX)
        D = grid + wm * disp + (1 - wm) * (ws * inner + (1 - ws) * outer)
        # LOCAL FOLD repair (10-21, Priya: PS94_0823 left-cheek compression survived the 8 px interior): where the warp's
        # area change (Jacobian determinant) < FOLD_DET, grown by FOLD_GROW_PX and feathered, the displacement is
        # replaced by a FOLD_SMOOTH_PX smoothing of it; never in the mouth zone; nowhere else changes
        disp2 = D - grid
        gxx, gxy = np.gradient(D[..., 0], axis=1), np.gradient(D[..., 0], axis=0)
        gyx, gyy = np.gradient(D[..., 1], axis=1), np.gradient(D[..., 1], axis=0)
        fold = ((gxx * gyy - gxy * gyx) < FOLD_DET).astype(np.uint8)
        if fold.any():
            fold = cv2.dilate(fold, np.ones((2 * FOLD_GROW_PX + 1,) * 2, np.uint8)).astype(np.float32)
            fm = (cv2.GaussianBlur(fold, (0, 0), FOLD_GROW_PX / 2.0)[..., None]) * (1 - wm)
            D = grid + (1 - fm) * disp2 + fm * cv2.GaussianBlur(disp2, (0, 0), FOLD_SMOOTH_PX)
    T = cv2.remap(S, D[..., 0], D[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # LOCAL-QUALITY FALLBACK (10-21, Priya: PS94_0823's image-left jaw / cheek contour was pulled inward): where the
    # similarity-only warp matches the template better than the full warp (windowed NCC, LOCAL_WIN px, by more than
    # LOCAL_MARGIN), use the similarity field there (feathered); the raw zones (mouth, eyes) keep the full field
    q_t = local_ncc(template, cv2.remap(moving, T[..., 0], T[..., 1], cv2.INTER_LINEAR))
    q_s = local_ncc(template, cv2.remap(moving, S[..., 0], S[..., 1], cv2.INTER_LINEAR))
    use_s = cv2.GaussianBlur((q_s > q_t + LOCAL_MARGIN).astype(np.float32), (0, 0), 15.0)
    keep = raw_zones(template.shape, lm_t)
    ws_ = (use_s * (1 - keep))[..., None]
    T = (1 - ws_) * T + ws_ * S
    es, et = face_err(S, lm_t, lm_m), face_err(T, lm_t, lm_m)
    if np.isfinite(et) and np.isfinite(es) and et > max(es, 8.0):
        return S, f"similarity (deepflow face err {et:.1f} > {es:.1f})"
    return T, "similarity+deepflow"


def _reg_job(args):
    key, template, moving, lm_t, lm_m, poly_t = args
    f, how = register(template, moving, lm_t, lm_m, poly_t)
    return key, (f, how)


def sample(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    from face_rhythm import helpers
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    out = np.full_like(pts, np.nan)
    ok = np.isfinite(pts).all(1)
    if ok.any():
        out[ok] = helpers.remap_points(points=pts[ok], remappingIdx=field.astype(np.float64), interpolation="linear")
    return out


def landmarks(Z) -> dict:
    lm = {"nose": np.asarray(Z["nose"], float), "jaw": np.asarray(Z["jaw"], float)}
    for p, xy in zip(Z["tip_positions"], Z["tip_xy"]):
        lm[f"tip_{p}"] = np.asarray(xy, float)
    return lm


def main(argv=None) -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--master-exclude", nargs="*", default=["PS93"], help="animals not eligible as the master")
    a = ap.parse_args(argv)
    out = a.root / "register"
    (out / "fields").mkdir(parents=True, exist_ok=True)
    Z = {f.stem: np.load(f) for f in sorted((a.root / "frames").glob("PS*.npz"))}
    raw = {s: z["median"] for s, z in Z.items()}
    ims = {s: prep(im) for s, im in raw.items()}
    LM = {s: landmarks(z) for s, z in Z.items()}
    ef = a.root / "register_inputs" / "eye_landmarks.json"
    if ef.exists():
        for s_, e in json.loads(ef.read_text()).items():
            if s_ in LM:
                LM[s_]["eye_l"], LM[s_]["eye_r"] = np.asarray(e["left"], float), np.asarray(e["right"], float)
        print("eye landmarks:", sorted(json.loads(ef.read_text())), flush=True)
    animals = sorted({s[:4] for s in Z})
    pool_f = a.root / "pre_august_sessions.json"
    pool = set(json.loads(pool_f.read_text())) if pool_f.exists() else set(ims)
    tmpl = {an: medoid({s: ims[s] for s in ims if s.startswith(an) and s in pool}) for an in animals}
    print("animal templates:", tmpl, flush=True)

    import face_rhythm as fr
    # master: medoid of the animal templates AFTER a pairwise landmark similarity alignment
    T = {an: ims[tmpl[an]] for an in animals}
    sc = {}
    for i in animals:
        vals = []
        for j in animals:
            if i == j:
                continue
            f = similarity_field(LM[tmpl[i]], LM[tmpl[j]], T[i].shape)
            w = cv2.remap(T[j], f[..., 0], f[..., 1], cv2.INTER_LINEAR)
            vals.append(_ncc(T[i], w))
        sc[i] = float(np.mean(vals))
    master_an = max((k for k in sc if k not in a.master_exclude), key=sc.get)
    master_s = tmpl[master_an]
    M = ims[master_s]
    print("master:", master_an, master_s, {k: round(v, 3) for k, v in sc.items()}, flush=True)

    # fields: animal template -> master, session -> animal template (parallel)
    # the master FACE polygon (Priya's ROI), if drawn: decides the outer shell (`shell_weight`). Two passes: animal
    # templates -> master first, so the polygon can be carried into each template for its sessions' registrations.
    poly_m = None
    rf = a.root / "rois" / master_s / "ROIs.h5"
    if rf.exists():
        poly_m = np.asarray(fr.rois.ROIs(select_mode="file", path_file=str(rf), verbose=0).roi_points["face"], float)
    F, HOW = {}, {}
    jobs = [(("A", an), M, T[an], LM[master_s], LM[tmpl[an]], poly_m) for an in animals if an != master_an]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, (f, how) in ex.map(_reg_job, jobs):
            F[k], HOW[k] = f, how
    H0, W0 = M.shape
    ident0 = np.stack(np.meshgrid(np.arange(W0, dtype=np.float32), np.arange(H0, dtype=np.float32)), -1)
    poly_t = {an: (None if poly_m is None else sample(F.get(("A", an), ident0), poly_m)) for an in animals}
    jobs = [(("S", s), ims[tmpl[s[:4]]], ims[s], LM[tmpl[s[:4]]], LM[s], poly_t[s[:4]])
            for s in sorted(ims) if s != tmpl[s[:4]]]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for k, (f, how) in ex.map(_reg_job, jobs):
            F[k], HOW[k] = f, how
    H, W = M.shape
    ident = np.stack(np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)), -1)
    A = {an: F.get(("A", an), ident) for an in animals}
    rows = []
    mlm = LM[master_s]
    for s in sorted(ims):
        fa = A[s[:4]]
        fs = F.get(("S", s), ident)
        # composed master -> session field: R_s(R_a(p)) on the whole grid
        comp = cv2.remap(fs, fa[..., 0], fa[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        np.save(out / "fields" / f"{s}.npy", comp.astype(np.float32))
        for k, p in mlm.items():
            if k not in LM[s] or not np.isfinite(p).all() or not np.isfinite(LM[s][k]).all():
                continue
            q = sample(comp, p)[0]
            rows.append({"session": s, "animal": s[:4], "landmark": k,
                         "method": HOW.get(("S", s), "template") + " / " + HOW.get(("A", s[:4]), "master"),
                         "err_px": float(np.hypot(*(q - LM[s][k]))),
                         "err_unregistered_px": float(np.hypot(*(p - LM[s][k])))})
    import csv
    with open(out / "qc_landmarks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(out / "master.npz", image=raw[master_s], clahe=M, session=master_s,
                        **{f"lm_{k}": v for k, v in mlm.items()})
    (out / "templates.json").write_text(json.dumps({"animal_templates": tmpl, "master": master_s,
                                                    "master_scores": sc, "master_exclude": a.master_exclude,
                                                    "template_pool": sorted(pool)}, indent=1))
    # overlays: every session warped INTO master space (moving sampled at the composed field), per animal
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for an in animals:
        ss = [s for s in sorted(ims) if s.startswith(an)]
        c = 8
        fig, axes = plt.subplots(int(np.ceil(len(ss) / c)), c, figsize=(2 * c, 2.1 * np.ceil(len(ss) / c)),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, s in zip(axes.flat, ss):
            comp = np.load(out / "fields" / f"{s}.npy")
            wimg = cv2.remap(ims[s], comp[..., 0], comp[..., 1], cv2.INTER_LINEAR)
            rgb = np.dstack([M, wimg, M])                       # green/magenta: grey where they agree
            ax.imshow(np.clip(rgb, 0, 1))
            ax.set_title(s[5:], fontsize=7)
        fig.suptitle(f"{an} sessions warped onto the master ({master_s}); magenta = master, green = session",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"qc_overlay_{an}.png", dpi=80)
        plt.close(fig)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["animal"], r["landmark"].split("_")[0])].append((r["err_px"], r["err_unregistered_px"]))
    print("median landmark error (px): registered vs unregistered")
    for k in sorted(agg):
        v = np.array(agg[k])
        print(f"  {k[0]} {k[1]:5s} n={len(v):3d}  {np.median(v[:, 0]):6.1f}  vs {np.median(v[:, 1]):6.1f}")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

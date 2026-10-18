"""Wasserstein-2 spatial-distribution readout of the cam4 keypoints, pre vs post stroke (Priya 2026-10-18: "as in
stroke_orofacial").

    python -m wfield_local.orofacial_w2 [--run lp_r4_heatmap_20261013__clean_v35] [--jobs 6] [--n 800] [--draws 3]

stroke_orofacial compared each session's spatial distribution of a body part with the animal's pre-stroke distribution
by the Wasserstein-2 (earth mover's, squared-distance) metric, with the spout / eye as references that should not
change. Here, per session x spout position x keypoint:

  points   every tracked frame of every ENGAGED trial (the cohort's reference-position gate), cue + 70 ms .. DAQ trial
           stop. tongue = CLEANED tip in the mouth frame, tongue OUT only (>= 20 px; in-mouth frames would put a
           lick-rate-dependent mass at the origin); jaw = CLEANED jaw (rest-relative); nose, spout = pose (likelihood
           >= 0.6) in the SAME MOUTH FRAME (minus the session's `tongue_origin`) -- the CONTROLS: the spout is a fixed
           apparatus point per position and the nose sits on the head, so either moving in the mouth frame says the
           frame the tongue is measured in drifted, not the tongue. nose_img / spout_img = the same in raw IMAGE pixels:
           a FRAMING-drift readout only. Measured 10-18: in image pixels pre sessions differ from each other by 35-60 px
           (June vs August framing / head position), more than post sessions differ from pre -- raw image positions are
           unusable for this, which is why everything else is in the mouth frame.
  W2       exact W2 between equal-size subsamples (N points each, `linear_sum_assignment` on squared distances),
           averaged over `draws` seeded draws; the reference is the animal's pooled PRE-stroke points at the position
           (a pre session is compared with the OTHER pre sessions: leave-one-out = the noise floor, same N).
  readout  per-session W2 (px). Post-stroke W2 above the pre leave-one-out level = the distribution moved / changed
           shape. Stats: the cohort's session Mann-Whitney (+ BH-FDR) and animal-weighted bootstrap
           (`orofacial_stats.session_change_tests`) on W2, i.e. post sessions vs pre leave-one-out sessions.

Outputs: <LP_analysis>/<run>/cohort/tables/w2/ (sessions.csv, stats.csv) and figures/w2/ (epochs_by_position.png,
timecourse_by_position.png, + svg/).
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

KEYPOINTS = ("tongue", "jaw", "nose", "spout", "nose_img", "spout_img")
LIK_MIN = 0.6
TONGUE_OUT_PX = 20.0
MIN_POINTS = 100
WIN0_MS = 70.0


def _points(sd: Path, animal: str, date: str, keep: set | None) -> dict:
    """{(keypoint, position): (n, 2) array} for one session folder."""
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    S = oo.load_session(rv, sd.parent.parent.parent.name, animal, date)
    ix, pose, cl = S["index"], S["pose"], S["clean"]
    t = ix.t_ms.to_numpy(float)
    stop = ix.stop_ms.to_numpy(float)
    win = (t >= WIN0_MS) & (t <= np.where(np.isfinite(stop), stop, 3500.0))
    if keep is not None:
        win &= ix.trial_id.astype(int).isin(keep).to_numpy()
    pos = ix.position.to_numpy(str)
    tx, ty = cl.tongue_x.to_numpy(float), cl.tongue_y.to_numpy(float)
    o = np.load(sd / "clean.npz")["tongue_origin"].astype(float)          # the mouth, image pixels
    xy = {"tongue": (tx, ty, np.hypot(tx, ty) >= TONGUE_OUT_PX),
          "jaw": (cl.jaw_x.to_numpy(float), cl.jaw_y.to_numpy(float), np.ones(len(t), bool))}
    for kp in ("nose", "spout"):
        x, y = pose[f"{kp}_x"].to_numpy(float), pose[f"{kp}_y"].to_numpy(float)
        ok = pose[f"{kp}_likelihood"].to_numpy(float) >= LIK_MIN
        xy[kp] = (x - o[0], y - o[1], ok)                                   # mouth frame (the control)
        xy[f"{kp}_img"] = (x, y, ok)                                          # image pixels (framing drift)
    out = {}
    for kp, (x, y, ok) in xy.items():
        m = win & ok & np.isfinite(x) & np.isfinite(y)
        for p in np.unique(pos[m]):
            mm = m & (pos == p)
            if mm.sum() >= MIN_POINTS:
                out[(kp, p)] = np.c_[x[mm], y[mm]].astype(np.float32)
    return out


def w2(a: np.ndarray, b: np.ndarray, n: int, draws: int, seed: int) -> float:
    """Exact W2 (px) between equal-size random subsamples of a and b, mean over draws."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    k = min(n, len(a), len(b))
    vals = []
    for _ in range(draws):
        A = a[rng.choice(len(a), k, replace=False)]
        B = b[rng.choice(len(b), k, replace=False)]
        C = cdist(A, B, "sqeuclidean")
        r, c = linear_sum_assignment(C)
        vals.append(np.sqrt(C[r, c].mean()))
    return float(np.mean(vals))


def _cache_dir(run: str) -> Path:
    import tempfile
    d = Path(tempfile.gettempdir()) / "orofacial_w2_points" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_points(job):
    """Module-level worker: write one session's point clouds to the local cache; return (meta, cache file)."""
    from wfield_local.orofacial_cohort import _repo_on_path, engagement_keep
    _repo_on_path()
    sd, animal, date, epoch, day, cache = job
    keep, gate = engagement_keep(animal, date, sd)
    pts = _points(sd, animal, date, keep)
    f = Path(cache) / f"{animal}_{date}.npz"
    np.savez(f, **{f"{kp}|{p}": a for (kp, p), a in pts.items()})
    return {"animal": animal, "date": date, "epoch": epoch, "day": day, "gate": gate}, str(f)


def _w2_group(job):
    """Module-level worker: every session of one animal x keypoint x position against that animal's pre pool
    (a pre session against the OTHER pre sessions)."""
    metas, kp, p, n, draws = job
    key = f"{kp}|{p}"
    pts = {}
    for meta, f in metas:
        with np.load(f) as z:
            if key in z.files:
                pts[meta["date"]] = z[key]
    out = []
    for meta, _f in metas:
        if meta["date"] not in pts:
            continue
        ref = [a for d, a in pts.items() if d != meta["date"] and
               next(m for m, _ in metas if m["date"] == d)["epoch"] == "pre"]
        if not ref:
            continue
        # stable seed: Python's hash() of strings changes per run (PYTHONHASHSEED; CLAUDE.md rule 9)
        seed = zlib.crc32(f"{meta['animal']}|{meta['date']}|{kp}|{p}".encode())
        a = pts[meta["date"]]
        R = np.concatenate(ref)
        out.append({**meta, "keypoint": kp, "position": p, "n_points": len(a), "n_ref": len(R),
                    "w2_px": w2(a, R, n, draws, seed)})
    return out


def compute(rv, run: str, jobs: int | None, n: int, draws: int, baseline: str = "all") -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    from wfield_local import config, epochs, parallel
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    cache = _cache_dir(run)
    items = []
    for s_ in ak.curated_sessions(order="sorted"):
        a = config.animal_of(s_["label"])
        date = "2026" + s_["label"].split("_")[-1]
        sd = oo.session_dir(rv, run, a, date)
        ep = epochs.epoch_of(s_["label"])
        if baseline == "late" and ep == "pre" and date[4:8] < oc.LATE_PRE_FROM:
            continue                                   # late baseline: June pre sessions neither scored nor in the pool
        if (sd / "manifest.json").exists():
            items.append((sd, a, date, ep, epochs.days_since_stroke(s_["label"]), str(cache)))
    res, fail = ak.fan_sessions(items, _session_points, jobs=jobs, key=lambda it: (it[1], it[2]), label="points")
    if fail:
        raise RuntimeError(f"{len(fail)} session(s) failed: {fail}")
    sess = [v for _, v in res]
    groups = []
    for animal in sorted({m["animal"] for m, _ in sess}):
        metas = [(m, f) for m, f in sess if m["animal"] == animal]
        poss = sorted({k.split("|")[1] for _m, f in metas for k in np.load(f).files})
        for kp in KEYPOINTS:
            for p in poss:
                groups.append((metas, kp, p, n, draws))
    out, fail = parallel.fan_out(groups, _w2_group, jobs=jobs or parallel.default_jobs(), label="w2 group")
    if fail:
        raise RuntimeError(f"{len(fail)} W2 group(s) failed")
    W = pd.DataFrame([r for _, rs in out for r in rs])
    return W.sort_values(["animal", "date", "keypoint", "position"]).reset_index(drop=True)


def figures(W: pd.DataFrame, figs: Path, title: str) -> pd.DataFrame:
    """Cohort epochs-by-position and day-bin timecourse of W2 (rows = keypoints), with the cohort tests."""
    from wfield_local import orofacial_cohort as oc
    from wfield_local.orofacial_stats import session_change_tests
    lat = "L"
    P = W.copy()
    P["pos"] = [oc.pos_key(p, lat) for p in P.position]
    wide = P.pivot_table(index=["animal", "date", "epoch", "day", "pos"], columns="keypoint", values="w2_px").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={k: f"w2_{k}" for k in KEYPOINTS})
    ms = tuple(f"w2_{k}" for k in KEYPOINTS if f"w2_{k}" in wide)
    for k in KEYPOINTS:
        oc.LABEL[f"w2_{k}"] = (f"W2 {k.replace('_img', '')} (IMAGE px, framing) vs own pre" if k.endswith("_img")
                               else f"W2 {k} (mouth frame) vs own pre (px)")
    ST = session_change_tests(wide, ms)
    oc.fig_cohort_epochs_by_position(wide, ms, figs / "epochs_by_position.png",
                                     title + "\nWasserstein-2 distance of each session's keypoint cloud from the "
                                     "animal's pre-stroke cloud (pre = leave-one-out, the noise floor)", ST)
    oc.fig_cohort_timecourse_by_position(wide, ms, figs / "timecourse_by_position.png",
                                         title + "\nWasserstein-2 distance from the animal's own pre-stroke cloud")
    return ST


def main(argv=None) -> int:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    oc._repo_on_path()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--n", type=int, default=800, help="points per subsample")
    ap.add_argument("--draws", type=int, default=3, help="subsample draws averaged per W2")
    ap.add_argument("--baseline", choices=("all", "late"), default="all",
                    help="pre pool: all curated pre sessions, or only those >= late_pre_from ('late' -> *_latepre)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    base = oo.root(rv) / a.run / "cohort"
    sfx = "_latepre" if a.baseline == "late" else ""
    tables, figs = base / "tables" / f"w2{sfx}", base / "figures" / f"w2{sfx}"
    writeguard.assert_writable(base)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    W = compute(rv, a.run, a.jobs, a.n, a.draws, a.baseline)
    W.to_csv(tables / "sessions.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print("per-animal mean W2 (px) by epoch x keypoint (rule 8: look at animals before the pooled test):")
    print(W.groupby(["keypoint", "animal", "epoch"]).w2_px.mean().unstack("epoch")
          .reindex(columns=["pre", "acute", "subacute", "chronic"]).round(1).to_string())
    t = (f"Orofacial W2, pre vs post stroke -- {a.run} -- ENGAGED trials -- baseline: "
         f"{'LATE pre only' if a.baseline == 'late' else 'all curated pre'} -- N={a.n} x {a.draws} draws -- "
         f"sessions: {W.groupby('animal').date.nunique().to_dict()}")
    ST = figures(W, figs, t)
    ST.to_csv(tables / "stats.csv", index=False, float_format="%.4g")
    print(f"tables -> {tables}\nfigures -> {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""O2 stages `build` (the fit tensor + the frozen point-PCA basis), `fit` (rank sweep, random starts, leave-one-animal-out) and the desktop stage `select` (freeze one rank as master.npz)."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import POINTS_CHUNK
from .cp import match, ncp_hals, pooled_evr, project_rows, similarity
from .files import _sidecar_module, load_plan, npz_memmap, npz_small, out_dir, plan_views, view_out_dir
from .sample import gather_rows, spec_of


# ------------------------------------------------------------------------------------------- O2: build / fit
def cmd_build(sd: Path) -> None:
    pl = load_plan(sd)
    od = out_dir(sd)
    rows, index, scale, vscale = [], [], {}, {}
    for a, picks in pl["fit"].items():
        Xa, ix = gather_rows(sd, picks, weighted=False)
        if plan_views(sd):                                  # MULTI-VIEW: equal energy per view (face_rhythm_multiview)
            fmv = _sidecar_module("face_rhythm_multiview")
            S0 = spec_of(sd, next(iter(picks["trials"])), "trials", weighted=False)
            X4 = Xa.reshape(len(Xa), 2, -1, Xa.shape[-1])
            vscale[a] = fmv.balance_divisors(X4, S0.P)
            for sl, d in zip(fmv.view_slices(S0.P), vscale[a]):
                X4[:, :, sl] /= d
            print(f"  {a}: view divisors {dict(zip([pl['views']['ref']] + [v['cam'] for v in plan_views(sd)], [round(d, 4) for d in vscale[a]]))}")
        scale[a] = float(np.sqrt(np.mean(Xa.astype(np.float64) ** 2)))
        rows.append(Xa / scale[a])
        index += [(a, *r) for r in ix]
        print(f"  {a}: {len(Xa)} rows, scale {scale[a]:.4g}", flush=True)
    X = np.concatenate(rows).astype(np.float32)
    np.save(sd / "fit_tensor.npy", X)
    idx = np.array(index, dtype=object)
    np.savez(sd / "fit_index.npz", animal=idx[:, 0].astype(str), key=idx[:, 1].astype(str),
             kind=idx[:, 2].astype(str), trial=idx[:, 3].astype(int), bin=idx[:, 4].astype(int))
    (sd / "scales.json").write_text(json.dumps(scale, indent=1))
    if vscale:
        (sd / "view_scales.json").write_text(json.dumps(vscale, indent=1))
    print(f"fit tensor {X.shape} ({X.nbytes / 1e9:.1f} GB) -> {sd}")
    if pl.get("points_fit"):
        build_points_pca(sd, pl, od)


def _centred_disp(path: Path, bins=None) -> tuple[np.ndarray, np.ndarray]:
    """A task's 25 Hz displacements (bins, 2P) float32, point-major (point, xy), centred on the session mean (finite
    bins), and that mean. ``bins``: only these rows (the mean is still the whole session's)."""
    D = npz_memmap(path, "disp")
    n, P = D.shape[0], D.shape[1]
    tot, cnt = np.zeros(2 * P), 0
    for c0 in range(0, n, POINTS_CHUNK):
        x = np.asarray(D[c0:c0 + POINTS_CHUNK], np.float32).reshape(-1, 2 * P)
        ok = np.isfinite(x).all(1)
        tot += x[ok].sum(0)
        cnt += ok.sum()
    mu = (tot / max(cnt, 1)).astype(np.float32)
    rows = np.asarray(D[np.sort(np.asarray(bins, int))] if bins is not None else D, np.float32).reshape(-1, 2 * P)
    return rows - mu, mu


def build_points_pca(sd: Path, pl: dict, od: Path) -> None:
    """Frozen-to-be PCA of centred displacements on the plan's pre-stroke bins (equal per animal)."""
    rows = []
    for _a, per_key in pl["points_fit"].items():
        for key, bins in per_key.items():
            x, _ = _centred_disp(od / f"{key}_points.npz", bins)
            rows.append(x[np.isfinite(x).all(1)])
    X = np.concatenate(rows).astype(np.float64)
    K = int(pl["params"]["point_pcs"])
    _, sv, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    ev = sv ** 2 / np.sum(sv ** 2)
    np.savez(sd / "points_pca.npz", components=Vt[:K].astype(np.float32), explained=ev[:K], n_rows=len(X))
    print(f"points PCA: {len(X)} bins, top {K} PCs explain {ev[:K].sum():.1%} -> {sd / 'points_pca.npz'}")


def cmd_fit(sd: Path, rank: int, loao: bool = True) -> None:
    pl = load_plan(sd)
    X = np.ascontiguousarray(np.load(sd / "fit_tensor.npy", mmap_mode="r"))
    animals = np.load(sd / "fit_index.npz")["animal"]
    n_init, n_iter = pl["params"]["inits"], pl["params"]["n_iter"]
    t0 = time.time()
    fits = []
    for k in range(n_init):
        f = ncp_hals(X, rank, seed=1000 * rank + k, n_iter=n_iter)
        fits.append(f)
        print(f"rank {rank} init {k}: rel err {f['rel_err']:.4f} ({f['n_iter']} it, {(time.time() - t0) / 60:.0f} min)",
              flush=True)
    best = int(np.argmin([f["rel_err"] for f in fits]))
    b = fits[best]
    stab = [similarity([b["A"], b["B"]], [f["A"], f["B"]]) for i, f in enumerate(fits) if i != best]
    # PER-FACTOR stability (Priya 10-22: which factors move between starts, not one number per rank): for each of the
    # best start's components, its matched spatial x spectral cosine in every other start
    per = np.array([np.prod(match([b["A"], b["B"]], [f["A"], f["B"]])[1], axis=0)
                    for i, f in enumerate(fits) if i != best]).reshape(-1, rank)
    loao_res = {}
    for a in (np.unique(animals) if loao else []):          # --no-loao: a rank sweep ("how many is enough")
        m = animals == a
        f = ncp_hals(X[~m], rank, seed=7 + rank, n_iter=n_iter)
        _, e_out, n2 = project_rows(X[m], f["A"], f["B"])
        _, e_in, _ = project_rows(X[m], b["A"], b["B"])
        loao_res[str(a)] = {"held_out_err": float(np.sqrt(1 - pooled_evr(e_out, n2))),
                        "in_fit_err": float(np.sqrt(1 - pooled_evr(e_in, n2))),
                        "similarity_to_full": similarity([b["A"], b["B"]], [f["A"], f["B"]])}
        print(f"  leave out {a}: held-out err {loao_res[str(a)]['held_out_err']:.4f} vs in-fit "
              f"{loao_res[str(a)]['in_fit_err']:.4f}, factor similarity {loao_res[str(a)]['similarity_to_full']:.3f}",
              flush=True)
    np.savez(sd / f"rank{rank:02d}.npz", A=b["A"], B=b["B"], C=b["C"].astype(np.float32), best_init=best,
             rel_errs=np.array([f["rel_err"] for f in fits]), trace=b["trace"],
             A_starts=np.stack([f["A"] for f in fits]).astype(np.float32),        # every start (small: A, B only)
             B_starts=np.stack([f["B"] for f in fits]).astype(np.float32), stability_per_factor=per)
    summ = {"rank": rank, "rel_err": float(b["rel_err"]), "evr_uncentred": float(1 - b["rel_err"] ** 2),
            "rel_errs": [float(f["rel_err"]) for f in fits], "best_init": best,
            "stability_median": float(np.median(stab)) if stab else None, "stability": stab,
            "stability_per_factor_median": np.median(per, axis=0).round(4).tolist() if len(per) else None,
            "stability_per_factor_min": per.min(axis=0).round(4).tolist() if len(per) else None, "loao": loao_res,
            "minutes": (time.time() - t0) / 60}
    (sd / f"rank{rank:02d}.json").write_text(json.dumps(summ, indent=1))
    print(f"rank {rank}: err {b['rel_err']:.4f}, EVR {summ['evr_uncentred']:.3f}, stability "
          f"{summ['stability_median']}, {summ['minutes']:.0f} min")


def spec_hash(sd: Path, rank: int, best_init: int) -> str:
    pl = load_plan(sd)
    blob = json.dumps({"fit": pl["fit"], "points_fit": pl.get("points_fit"), "params": {k: v for k, v in pl["params"].items() if k != "session_trials"},
                       "rank": rank, "best_init": best_init,
                       "scales": json.loads((sd / "scales.json").read_text())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cmd_select(sd: Path, rank: int) -> None:
    R = np.load(sd / f"rank{rank:02d}.npz")
    h = spec_hash(sd, rank, int(R["best_init"]))
    m = sd / "master.npz"
    if m.exists():
        old = str(np.load(m)["spec_hash"])
        sys.exit(f"{m} already frozen (spec {old}); {'same spec, nothing to do' if old == h else 'REFUSED: different spec'}")
    pl = load_plan(sd)
    first = next(iter(pl["fit"].values()))["sessions"][0]
    meta = npz_small(out_dir(sd) / f"{first}_trials.npz", ("freqs", "home"))
    views = {}
    if plan_views(sd):                                  # multi-view: every view's home points, in stacking order
        fmv = _sidecar_module("face_rhythm_multiview")
        rc = pl["views"]["ref"]
        homes, Ps = [meta["home"]], [len(meta["home"])]
        for v in plan_views(sd):
            mv = npz_small(view_out_dir(sd, v["cam"], v["bundle"]) / f"{fmv.view_key(first, rc, v['cam'])}_trials.npz",
                           ("freqs", "home"))
            if not np.allclose(mv["freqs"], meta["freqs"]):
                sys.exit(f"{v['cam']}: frequencies differ from {rc} -- views must share the VQT params")
            homes.append(mv["home"])
            Ps.append(len(mv["home"]))
        meta["home"] = np.concatenate(homes)
        views = {"view_cams": np.array([rc] + [v["cam"] for v in plan_views(sd)]), "view_P": np.array(Ps),
                 "view_scales_json": (sd / "view_scales.json").read_text()}
    P = len(R["A"]) // 2
    np.savez(m, spatial=R["A"].reshape(2, P, rank).astype(np.float32), spectral=R["B"].astype(np.float32),
             freqs=meta["freqs"], home=meta["home"], scales_json=(sd / "scales.json").read_text(), rank=rank,
             spec_hash=h, set=pl["set"], frozen=time.strftime("%Y-%m-%d %H:%M"), role=pl.get("role", "master"),
             animals=np.array(pl.get("animals", sorted(pl["fit"]))), **views,
             **({f"points_{k}": v for k, v in np.load(sd / "points_pca.npz").items()}
                if (sd / "points_pca.npz").exists() else {}))
    print(f"FROZEN {pl['set']} [{pl.get('role', 'master')}: {', '.join(sorted(pl['fit']))}] rank {rank} spec {h} -> {m}")

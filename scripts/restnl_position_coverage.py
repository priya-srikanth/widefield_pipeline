"""RESTnl baseline position-coverage sensitivity check (read-only; 11-02). UNTRACKED helper, no page figures.

Question: rule `wr` averages the per-position rest medians over the positions with >= 50 lick-free frames (>= 4 of 6),
so the position SET varies by session (PS93 far contra missing in 5/7 August pre sessions + 0828; PS92 0811 / 0812 /
0826; PS93 0812 / 0820). Does that change any RESTnl map result?

Stages (run from the wf_restnl worktree with the usual env prefix):
    python scripts/restnl_position_coverage.py collect --arm contact|lp|enl1s   # raw per-position maps -> scratch
    python scripts/restnl_position_coverage.py posmaps                          # U_atlas @ posmed per session -> scratch
    python scripts/restnl_position_coverage.py analyse                          # tables -> OUT

Read-only on M: and on the RESTnl cache. Maps are read through `position_reference_maps.maps_by_epoch` (warm `prm`
caches, legacy digest pin as `restnl_maps.setup_env`). Writes: SCRATCH (intermediate maps, brain pixels only) and OUT
(CSV tables + JSON).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path(os.environ.get("RESTNL_COV_SCRATCH",
                              r"C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/"
                              r"f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/restnl_cov"))
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/restnl_position_coverage_check")
CACHE = Path("C:/Users/SabatiniLab/.widefield_session_cache_restnl")
POS = ["close_center", "close_L", "close_R", "far_center", "far_L", "far_R"]     # restnl_mask code order (0..5)
FAR_C = 5
ARM_ALIGNS = {"contact": [("precue", "working", "7.1"), ("cue", "working", "7.7")],
              "enl1s": [("precue", "working", "7.5")],
              "lp": [("cue", "working", "7.10"), ("lick", "lick", "7.18")]}
ANIMALS = ["PS92", "PS93", "PS94", "PS95"]


def brain():
    from wfield_local import beta_maps as bm
    return np.asarray(bm.brain_mask(), bool)


# ------------------------------------------------------------------------------------------------ collect
def collect(arm: str) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import restnl_maps as rm
    rm.setup_env(arm, refs="restnl")
    from wfield_local import parallel
    parallel.pin_blas()
    from wfield_local import position_reference_maps as prm
    m = brain()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for align, variant, fig in ARM_ALIGNS[arm]:
        store, _rel, ntr = prm.maps_by_epoch(align, variant)
        rows, raws, nls = [], [], []
        for an, by_e in store.items():
            for e, by_q in by_e.items():
                for q, by_s in by_q.items():
                    for lab, d in sorted(by_s.items()):
                        if "raw" not in d:
                            continue
                        rows.append((an, e, q, lab, int(((ntr.get(an) or {}).get(e) or {}).get(q, {}).get(lab, 0))))
                        raws.append(np.asarray(d["raw"])[m].astype(np.float32))
                        nls.append(np.asarray(d["restnl"])[m].astype(np.float32) if "restnl" in d
                                   else np.full(m.sum(), np.nan, np.float32))
        fp = SCRATCH / f"maps_{arm}_{align}_{variant}.npz"
        np.savez(fp, animal=np.array([r[0] for r in rows]), epoch=np.array([r[1] for r in rows]),
                 position=np.array([r[2] for r in rows]), label=np.array([r[3] for r in rows]),
                 ntrials=np.array([r[4] for r in rows]), raw=np.stack(raws), restnl=np.stack(nls), fig=fig)
        print(f"[cov] {arm} {align}/{variant} ({fig}): {len(rows)} session-position maps -> {fp}", flush=True)


# ------------------------------------------------------------------------------------------------ posmaps
def _posmap_job(label):
    from wfield_local import restnl_mask
    from wfield_local.position_reference_maps import MAP_SHAPE  # noqa: F401
    s = restnl_mask.session_of(label)
    fz = sorted(CACHE.glob(f"{label}__restnl__*.npz"))
    if len(fz) != 1:
        return {"label": label, "error": f"{len(fz)} cache files"}
    z = np.load(fz[0])
    ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
    U = np.load(f"{ad[0]}/U_atlas.npy", mmap_mode="r")
    U = np.nan_to_num(np.asarray(U).reshape(-1, U.shape[2])).astype(np.float32)
    m = brain().reshape(-1)
    Ub = U[m]
    pm = np.asarray(z["posmed"], float)
    P = np.full((6, Ub.shape[0]), np.nan, np.float32)
    for c in range(6):
        if np.isfinite(pm[c]).all():
            P[c] = Ub @ pm[c].astype(np.float32)
    base_chk = Ub @ np.asarray(z["b_nl"], np.float32)
    bm_cached = np.asarray(z["base_map"]).reshape(-1)[m]
    err = float(np.max(np.abs(base_chk - bm_cached)) / (np.sqrt(np.mean(bm_cached ** 2)) + 1e-12))
    return {"label": label, "P": P, "nfr": np.asarray(z["nfr_pos"], int), "used": np.asarray(z["positions_used"], int),
            "base_cached": bm_cached.astype(np.float32), "base_relerr": err}


def posmaps(jobs: int = 4) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from wfield_local import parallel
    parallel.pin_blas()
    labels = sorted({f.name.split("__")[0] for f in CACHE.glob("*__restnl__*.npz")})
    labels = [x for x in labels if x.split("_")[0] in ANIMALS and int(x.split("_")[1]) >= 801]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = {}
    with ProcessPoolExecutor(max_workers=min(jobs, 4)) as ex:
        print(f"[cov] posmaps: {len(labels)} sessions, workers {min(jobs, 4)}", flush=True)
        for r in ex.map(_posmap_job, labels):
            if "error" in r:
                print(f"  !! {r['label']}: {r['error']}", flush=True)
                continue
            out[r["label"]] = r
            print(f"  .. {r['label']} nfr {r['nfr'].tolist()} base relerr {r['base_relerr']:.2e}", flush=True)
    labs = sorted(out)
    np.savez(SCRATCH / "posmaps.npz", label=np.array(labs), P=np.stack([out[x]["P"] for x in labs]),
             nfr=np.stack([out[x]["nfr"] for x in labs]),
             base_cached=np.stack([out[x]["base_cached"] for x in labs]),
             base_relerr=np.array([out[x]["base_relerr"] for x in labs]))
    print(f"[cov] posmaps -> {SCRATCH / 'posmaps.npz'}", flush=True)


# ------------------------------------------------------------------------------------------------ baselines
def _mean_of(P, idx):
    return np.mean(P[list(idx)], axis=0)


def build_variants(lab, P, nfr):
    """{variant: (base map over brain pixels, positions used, note)} for each session (keyed by label)."""
    an = {x: x.split("_")[0] for x in lab}
    order = {a: sorted([x for x in lab if an[x] == a], key=lambda x: int(x.split("_")[1])) for a in ANIMALS}
    idx = {x: i for i, x in enumerate(lab)}

    def valid(x, thr):
        return [c for c in range(6) if nfr[idx[x]][c] >= thr and np.isfinite(P[idx[x]][c]).all()]

    fixed = {a: sorted(set.intersection(*[set(valid(x, 50)) for x in order[a]])) for a in ANIMALS}
    nofc_animals = {a for a in ANIMALS if any(FAR_C not in valid(x, 50) for x in order[a])}
    V = {}
    for x in lab:
        i, a = idx[x], an[x]
        v50, v25 = valid(x, 50), valid(x, 25)
        d = {"wr50": (v50, ""), "wr25": (v25, "")}
        d["fixA_intersection"] = (fixed[a], "")
        d["fixB_noFC_PS92_93"] = ([c for c in v50 if c != FAR_C] if a in nofc_animals else v50, "")
        d["fixC_noFC_all"] = ([c for c in v50 if c != FAR_C], "")
        out = {k: (_mean_of(P[i], u), u, n) for k, (u, n) in d.items()}
        # IMPUTE: each missing position c = this session's mean over its used set U + the neighbouring sessions'
        # (nearest earlier + nearest later session of the same animal with c and all of U at >= 50 frames) offset
        # of c from THEIR mean over the same U, in map space (shared Allen grid). Then the 6-position mean.
        miss = [c for c in range(6) if c not in v50]
        if not miss:
            out["impute"] = out["wr50"]
        else:
            Pm = P[i].copy()
            mU = _mean_of(P[i], v50)
            notes = []
            seq = order[a]
            k0 = seq.index(x)
            for c in miss:
                deltas, used_n = [], []
                for direction in (-1, 1):
                    k = k0 + direction
                    while 0 <= k < len(seq):
                        y = seq[k]
                        vy = valid(y, 50)
                        if c in vy and set(v50) <= set(vy):
                            deltas.append(P[idx[y]][c] - _mean_of(P[idx[y]], v50))
                            used_n.append(y.split("_")[1])
                            break
                        k += direction
                Pm[c] = mU + np.mean(deltas, axis=0)
                notes.append(f"{POS[c]}<-{'+'.join(used_n)}")
            out["impute"] = (np.mean(Pm, axis=0), list(range(6)), "; ".join(notes))
        V[x] = out
    return V, fixed, nofc_animals


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


# ------------------------------------------------------------------------------------------------ analyse
def analyse() -> None:
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(SCRATCH / "posmaps.npz")
    lab = [str(x) for x in z["label"]]
    P, nfr, base_c = z["P"], z["nfr"], z["base_cached"]
    print(f"[cov] base_map reproduction: max rel err {z['base_relerr'].max():.2e}", flush=True)
    V, fixed, nofc = build_variants(lab, P, nfr)
    i_of = {x: i for i, x in enumerate(lab)}
    chk = max(float(np.max(np.abs(V[x]["wr50"][0] - base_c[i_of[x]]))) / (rms(base_c[i_of[x]]) + 1e-12) for x in lab)
    print(f"[cov] wr50 rebuilt vs cached base_map: max rel err {chk:.2e}", flush=True)
    variants = list(next(iter(V.values())).keys())

    # coverage table
    cov = []
    for x in lab:
        i = i_of[x]
        cov.append(dict(label=x, animal=x.split("_")[0], nfr=" ".join(map(str, nfr[i])),
                        **{f"used_{k}": "".join(str(c) for c in V[x][k][1]) for k in variants},
                        impute_note=V[x]["impute"][2]))
    pd.DataFrame(cov).to_csv(OUT / "coverage_by_session.csv", index=False)

    from wfield_local import beta_maps as bm_mod  # noqa: F401  (import check only)

    # ---------------- Q1: map-space shift of dropping one position, all-6 sessions
    maps = {}
    for f in sorted(SCRATCH.glob("maps_*.npz")):
        with np.load(f) as mzf:
            maps[f.stem] = {k: mzf[k] for k in mzf.files}
    q1 = []
    for key in ("maps_contact_cue_working", "maps_contact_precue_working"):
        if key not in maps:
            continue
        mz = maps[key]
        an_e = {}
        for j in range(len(mz["label"])):
            if mz["position"][j] != "far_R":
                continue
            if str(mz["label"][j]) in V:
                an_e.setdefault((str(mz["animal"][j]), str(mz["epoch"][j])), []).append(j)
        for j in range(len(mz["label"])):
            x = str(mz["label"][j])
            if mz["position"][j] != "far_R" or x not in V or len(V[x]["wr50"][1]) < 6:
                continue
            a, e = str(mz["animal"][j]), str(mz["epoch"][j])
            i = i_of[x]
            full = V[x]["wr50"][0]
            nl = mz["raw"][j] - full
            # the animal's pre-mean far-contra RESTnl map (the ratio denominator)
            pre_js = an_e.get((a, "pre"), [])
            pre_mean = np.mean([mz["raw"][k] - V[str(mz["label"][k])]["wr50"][0] for k in pre_js], axis=0)
            for c in range(6):
                drop = np.mean(P[i][[k for k in range(6) if k != c]], axis=0)
                sh = rms(full - drop)
                q1.append(dict(window=key.split("_")[2], animal=a, epoch=e, label=x, dropped=POS[c],
                               shift_rms=sh * 1e3, session_fc_map_rms=rms(nl) * 1e3,
                               animal_pre_fc_map_rms=rms(pre_mean) * 1e3,
                               shift_over_session_map=sh / rms(nl), shift_over_animal_pre=sh / rms(pre_mean),
                               base_rms=rms(full) * 1e3))
    Q1 = pd.DataFrame(q1)
    Q1.to_csv(OUT / "q1_drop_one_position_shift_sessions.csv", index=False)

    # ---------------- Q2/Q3: per-animal far-contra acute/pre ratios under each variant
    rows = []
    for key, mz in maps.items():
        fig = str(mz["fig"])
        _, arm, align, variant = key.split("_")
        lab_m = [str(x) for x in mz["label"]]
        for q in POS:
            for vname in variants:
                by = {}
                for j, x in enumerate(lab_m):
                    if mz["position"][j] != q or x not in V:
                        continue
                    by.setdefault(str(mz["epoch"][j]), {}).setdefault(str(mz["animal"][j]), []).append(
                        mz["raw"][j] - V[x][vname][0])
                pre = by.get("pre", {})
                for e in ("acute", "subacute", "chronic"):
                    post = by.get(e, {})
                    for a in sorted(set(pre) & set(post)):
                        r = rms(np.mean(post[a], 0)) / rms(np.mean(pre[a], 0))
                        rows.append(dict(fig=fig, arm=arm, align=align, variant_win=variant, position=q,
                                         baseline=vname, epoch=e, animal=a, ratio=r,
                                         n_pre=len(pre[a]), n_post=len(post[a])))
                    ans = sorted(set(pre) & set(post))
                    if ans:
                        sc = {a: rms(np.mean(pre[a], 0)) for a in ans}
                        pre_p = np.mean([np.mean(pre[a], 0) / sc[a] for a in ans], 0)
                        post_p = np.mean([np.mean(post[a], 0) / sc[a] for a in ans], 0)
                        rows.append(dict(fig=fig, arm=arm, align=align, variant_win=variant, position=q,
                                         baseline=vname, epoch=e, animal="pooled", ratio=rms(post_p) / rms(pre_p),
                                         n_pre=sum(len(pre[a]) for a in ans), n_post=sum(len(post[a]) for a in ans)))
        # cross-check: the page's RESTnl column (cached restnl) vs wr50 rebuilt
        okn = np.isfinite(mz["restnl"]).all(1)
        if okn.any():
            d = max(float(np.max(np.abs(mz["restnl"][j] - (mz["raw"][j] - V[lab_m[j]]["wr50"][0]))))
                    for j in np.flatnonzero(okn) if lab_m[j] in V)
            print(f"[cov] {key}: page RESTnl vs raw - wr50 rebuilt: max abs diff {d:.2e}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "ratios_by_animal_all.csv", index=False)
    meta = {"fixed_sets": {a: [POS[c] for c in fixed[a]] for a in fixed}, "noFC_animals_fixB": sorted(nofc),
            "variants": variants, "scratch": str(SCRATCH)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("collect", "posmaps", "analyse"))
    ap.add_argument("--arm", choices=tuple(ARM_ALIGNS))
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    if a.stage == "collect":
        collect(a.arm)
    elif a.stage == "posmaps":
        posmaps(a.jobs)
    else:
        analyse()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

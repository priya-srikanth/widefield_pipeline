"""Why are the state-space change maps cleaner than the 15r activity maps? Measured one step at a time (Priya 11-08: "move
on to 'I can measure why these are cleaner by passing the older maps' inputs through the same basis and normalization,
one step at a time'"; plan: running list B17).

Start from the 15r lick / lick arm AS RENDERED and change ONE thing per step toward the state-space map:
  S0  15r map: the rebuild session cache (`position_reference_maps.maps_by_epoch("lick", "lick")`, camera working-trial
      gate), RESTnl reference (`raw - restnl_base`), each animal x position divided by its own pre RMS over
      `brain_mask()`, contours = `significance_contour` pre-normalized on the stat mask. Since the 11-09 glue fix (A9)
      the rebuild profile's brain mask IS the glue-free analysis mask (Allen brain - bulbs - painted glue, 151,447 px),
      so the former separate "+ glue-free mask" step gave identical values (re-run 11-16) and is merged into S0
      (Priya 11-16): S0 = 15r map, glue-free brain mask, per-position pre-RMS units
  S1  + ONE unit per animal (RMS over the analysis mask of its pre maps, all positions) instead of one per position
  S2  + every session map through the cohort SVD (U_c U_c^T on the analysis mask: the 100-component low-rank denoising)
  S3  the state-space map itself (`state_space_chronic_maps.py`, WF_SS_MAPBASE=rest: drift-removed signal, first-lick
      0.1..0.5 s window, its trials), read from chronic_maps_lick_svd_rest/maps.npz
Per step and panel (mean over spouts + six spouts, acute / subacute / chronic): r with S3 (analysis stat-mask pixels),
split-half reliability (sessions split odd / even within animal and epoch, r of the two half-cohort change maps),
energy outside the cohort SVD space (1 - |U_c^T m|^2 / |m|^2: noise and anything the 100 components cannot carry), and
the fraction of the stat mask inside the nested-bootstrap contour (chronic, mean + far contra).

The 15r cache is READ, never rebuilt: inside the child `session_cache.cached` and `restnl_mask.baseline_or_build` are
replaced by read-only lookups that raise on a miss, and the session loop runs in that process (WIDEFIELD_MAP_JOBS=1),
so a cold cache stops the run instead of starting a ~119-session recompute. (Not the builders: the key hashes their
source.) Runs under the rebuild profile in a child process (its write fence); the parent
writes <state_space_recovery>/why_cleaner_lick/ (steps.csv, maps.npz) and page_figs/ss_why_cleaner_lick.png.

    python scripts/state_space_why_cleaner.py [--out DIR]

`--out DIR` writes steps.csv, maps.npz and ss_why_cleaner_lick.png (+ .svg) all into DIR instead (a re-run beside the
published outputs, 11-16). `--plot-only SRC --out DIR` redraws the figure (and writes the merged steps.csv) from an
earlier run's SRC/steps.csv + SRC/maps.npz without touching the 15r cache; it reads both the merged format and the
pre-merge five-step format (S0..S4, S1 == S0), mapping old S0 / S2 / S3 / S4 to S0 / S1 / S2 / S3. The encoding bundle's settings are found by `o2_state_space_lnmf.enc_share` (follows a
move-aside manifest; raises if the bundle is nowhere).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_sim as S  # noqa: E402

BASIS_DIR = S.SVD_BASIS
OUT = S.ROOT / "why_cleaner_lick"
FIGS = S.ROOT / "page_figs"
SS_MAPS = S.ROOT / f"chronic_maps_lick_svd_rest{S.RSFX}" / "maps.npz"   # RSFX: the RESTnl lick set (11-25 fix)
EPS = S.POST
POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]     # = S.POS order (L = ipsilesional)
COLS = ["mean", *S.POS]
# Internal (child) step ids 0, 2, 3 = output S0, S1, S2; the state-space map = S3. Child step 1 (glue-free mask) is not
# run any more: since the glue fix it equals step 0 (Priya 11-16, merged).
CHILD_STEPS = (0, 2, 3)
STEPS = ["S0 15r map, glue-free mask, pre-RMS units", "S1 + one unit per animal", "S2 + cohort SVD",
         "S3 state-space map"]
OLD_STEP = {"S0 15r as rendered": 0, "S2 + one unit per animal": 1, "S3 + cohort SVD": 2, "S4 state-space map": 3}
N_BOOT = 2000


def _cold(*_a, **_k):
    raise RuntimeError("15r cache is COLD for the current code / run tags -- not recomputing (state_space_why_cleaner)")


def _child(mask1_path, stat1_path, uc_path):
    """Under the rebuild profile: child steps 0, 2, 3 (output S0-S2) from the cached 15r maps. Small arrays only."""
    import o2_state_space_lnmf as L
    L._env(L.CACHE)
    os.environ["WIDEFIELD_MAP_JOBS"] = "1"
    os.environ["WIDEFIELD_MAP_REFS"] = "restnl"
    import pickle

    from wfield_local import beta_maps as bm
    from wfield_local import position_reference_maps as prm
    from wfield_local import restnl_mask, session_cache

    # READ-ONLY at the cache layer. Never patch the builders themselves: the cache key hashes their SOURCE
    # (`_raw_maps_code`), so a replaced `session_raw_maps` moves every key and reads a warm cache as cold (11-08).
    def read_only(session, kind, compute, params=None, verbose=True):
        fp = (session_cache.CACHE_DIR / f"{session['label']}__{session_cache.effective_kind(kind, session['label'])}__"
              f"{session_cache.session_signature(session, params)}.pkl")
        if not fp.exists():
            _cold()
        with open(fp, "rb") as fh:
            return pickle.load(fh)

    def restnl_read_only(label):
        d = restnl_mask.baseline(label)
        if d is not None:
            return d
        try:
            fp = restnl_mask.cache_path(label)
        except StopIteration:                                   # excluded session: no baseline by definition
            return None
        if not fp.exists():                                     # never built: dropped from EVERY step, listed
            no_base.append(label)
            if len(no_base) > 2:
                _cold()
            return None
        return None                                             # rule wr failed: no baseline, as in the render
    no_base = []
    session_cache.cached = read_only
    restnl_mask.baseline_or_build = restnl_read_only
    mask0 = bm.brain_mask()                                     # the profile's: glue IN (A9) -- the 15r scale
    stat0 = bm.stat_mask()
    mask1 = np.load(mask1_path)
    stat1 = np.load(stat1_path)
    Uc = np.load(uc_path).astype(np.float64)
    # `maps_by_epoch("lick", "lick")`'s session plan and RESTnl column, one session at a time (its store holds every
    # reference as full frames for every session at once: ~10 GB on this shared box). Kept: RESTnl maps on the union
    # of the two masks only.
    from wfield_local import config
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import ANIMALS, _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    union = mask0 | mask1
    sess, miss = {}, []
    for an in ANIMALS:
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post") if x.startswith(an)}
        for s in [x for x in SESSIONS if x["label"] in want]:
            d = _day(an, s["label"].split("_")[-1])
            if d is None:
                continue
            ep = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
            if ep is None:
                continue
            parts = prm._as_parts(prm.session_raw_maps_cached(s, "lick", post_s=2.0, variant="lick"))
            # empty arm first (PS93_0813: no video -> no camera-gated trials, never given a RESTnl baseline)
            base = restnl_read_only(s["label"]) if parts["raw"] else None
            if base is None:
                miss.append(s["label"])
                continue
            base = np.asarray(base["base_map"], np.float32)
            assert base.shape == mask1.shape, (base.shape, mask1.shape)
            arr = np.full((6, int(union.sum())), np.nan, np.float32)
            for j, q in enumerate(POS):
                if q in parts["raw"]:
                    arr[j] = (np.asarray(parts["raw"][q], np.float32) - base)[union]
            sess.setdefault(an, {}).setdefault(ep, []).append(arr)

    def full(a):                                                # (n, 6, npix_union) -> (n, 6, H, W)
        X = np.full((*a.shape[:2], *mask1.shape), np.nan, np.float32)
        X[..., union] = a
        return X
    sess = {an: {ep: full(np.stack(v)) for ep, v in d.items()} for an, d in sess.items()}
    info = {"n_sessions": {an: {ep: len(v) for ep, v in d.items()} for an, d in sess.items()},
            "no_maps_or_restnl": miss, "restnl_never_built": no_base, "mask0_px": int(mask0.sum()), "mask1_px": int(mask1.sum())}
    print(info, flush=True)

    def with_mean(X):                                           # (..., 6, H, W) -> (..., 7, H, W)
        return np.concatenate([np.nanmean(X, axis=-3, keepdims=True), X], axis=-3)

    def project(X):                                             # each map through U_c U_c^T on mask1
        Y = np.full_like(X, np.nan)
        flat = X[..., mask1]                                    # (..., npix)
        ok = np.isfinite(flat).all(-1)
        P = np.nan_to_num(flat) @ Uc
        Y[..., mask1] = np.where(ok[..., None], P @ Uc.T, np.nan)
        return Y

    def units(step):
        """{animal: (7,) or scalar} per-step divisor."""
        out = {}
        for an, d in sess.items():
            if "pre" not in d:
                continue
            pre = np.nanmean(d["pre"], 0)                       # (6, H, W)
            if step == 0:
                u = [np.sqrt(np.nanmean(pre[j][mask0] ** 2)) for j in range(6)]
            elif step == 1:
                u = [np.sqrt(np.nanmean(pre[j][mask1] ** 2)) for j in range(6)]
            else:
                u = [np.sqrt(np.nanmean(pre[:, mask1] ** 2))] * 6
            u = np.asarray(u, float)
            out[an] = np.concatenate([[np.nanmean(u) if step < 2 else u[0]], u])  # mean column: its own below
        return out

    res = {"info": info, "maps": {}, "halves": {}, "sig": {}, "siglab": {}}
    for step in CHILD_STEPS:
        X = {an: {ep: (project(v) if step == 3 else v) for ep, v in d.items()} for an, d in sess.items()}
        U = units(step)
        disp = mask0 if step == 0 else mask1
        for ep in EPS:
            ans = sorted(a for a in X if ep in X[a] and "pre" in X[a] and a in U)
            per, h1, h2 = [], [], []
            for a in ans:
                pre, post = with_mean(X[a]["pre"]), with_mean(X[a][ep])
                if step < 2:                                    # 15r: the mean column = mean of position-scaled maps
                    pre = np.concatenate([np.nanmean(pre[:, 1:] / U[a][1:, None, None], 1, keepdims=True),
                                          pre[:, 1:] / U[a][1:, None, None]], 1)
                    post = np.concatenate([np.nanmean(post[:, 1:] / U[a][1:, None, None], 1, keepdims=True),
                                           post[:, 1:] / U[a][1:, None, None]], 1)
                else:
                    pre, post = pre / U[a][0], post / U[a][0]
                per.append(np.nanmean(post, 0) - np.nanmean(pre, 0))
                h1.append(np.nanmean(post[0::2], 0) - np.nanmean(pre[0::2], 0))
                h2.append(np.nanmean(post[1::2], 0) - np.nanmean(pre[1::2], 0))
            M = np.nanmean(per, 0)
            M[:, ~disp] = np.nan
            res["maps"][(step, ep)] = M.astype(np.float32)
            res["halves"][(step, ep)] = (np.nanmean(h1, 0)[:, stat1].astype(np.float32),
                                         np.nanmean(h2, 0)[:, stat1].astype(np.float32))
            if ep != "chronic":
                continue
            for ci in (0, 6):                                   # mean over spouts, far contra
                pre_by, post_by = {}, {}
                for a in ans:
                    pre = with_mean(X[a]["pre"])[:, ci]
                    post = with_mean(X[a][ep])[:, ci]
                    if step >= 2:
                        pre, post = pre / U[a][0], post / U[a][0]
                    pre_by[a] = [m for m in pre if np.isfinite(m[disp]).all()]
                    post_by[a] = [m for m in post if np.isfinite(m[disp]).all()]
                if step == 0:                                   # as rendered: pre-normalized, the profile's stat mask
                    keep, lab = bm.significance_contour(pre_by, post_by, n_boot=N_BOOT)
                    sm = stat0
                elif step == 1:                                 # pre-normalized on the glue-free mask
                    sc = bm.pre_scale(pre_by, mask1)
                    keep, lab = bm.significance_contour(bm.scale_maps(pre_by, sc), bm.scale_maps(post_by, sc),
                                                        n_boot=N_BOOT, normalise=False, mask=stat1)
                    sm = stat1
                else:
                    keep, lab = bm.significance_contour(pre_by, post_by, n_boot=N_BOOT, normalise=False, mask=stat1)
                    sm = stat1
                k = np.zeros(mask1.shape, bool) if keep is None else np.asarray(keep, bool)
                res["sig"][(step, ci)] = k
                res["siglab"][(step, ci)] = f"{lab or 'none'} | frac of stat mask {(k & sm).sum() / sm.sum():.3f}"
                print(f"  S{step} chronic {COLS[ci]}: {res['siglab'][(step, ci)]}", flush=True)
        print(f"S{step} done", flush=True)
    return res


def _r(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else np.nan


def main(out: Path = OUT, figs: Path = FIGS):
    from concurrent.futures import ProcessPoolExecutor

    from wfield_local import beta_maps as bm
    import o2_state_space_svd as V
    out.mkdir(parents=True, exist_ok=True)
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    mask1 = np.load(BASIS_DIR / "mask.npy").reshape(atlas.shape)
    assert int(mask1.sum()) == int(V.analysis_mask().sum()), "bundle mask != analysis mask"
    stat1 = bm.stat_mask() & mask1                              # no profile here: glue removed
    np.save(out / "_mask1.npy", mask1)
    np.save(out / "_stat1.npy", stat1)
    with ProcessPoolExecutor(1) as ex:                          # the profile lives only in the child
        res = ex.submit(_child, str(out / "_mask1.npy"), str(out / "_stat1.npy"),
                        str(BASIS_DIR / "U_cohort.npy")).result()
    Uc = np.load(BASIS_DIR / "U_cohort.npy").astype(np.float64)
    z = np.load(SS_MAPS)
    ss = {ep: np.full((7, *mask1.shape), np.nan, np.float32) for ep in EPS}
    for ep in EPS:
        ss[ep][:, mask1] = z[f"full__{ep}__mean"]
    rows = []
    for k, step in enumerate(CHILD_STEPS):
        for ep in EPS:
            M = res["maps"][(step, ep)]
            h1, h2 = res["halves"][(step, ep)]
            for ci, col in enumerate(COLS):
                m = M[ci][stat1]
                v = np.nan_to_num(M[ci][mask1])
                e_out = 1 - float(np.sum((v @ Uc) ** 2) / max(np.sum(v ** 2), 1e-30))
                rows.append({"step": STEPS[k], "epoch": ep, "spout": col, "r_with_S3": _r(m, ss[ep][ci][stat1]),
                             "split_half_r": _r(h1[ci], h2[ci]), "energy_outside_svd": e_out,
                             "sig_frac": (float(res["sig"][(step, ci)][stat1].mean())
                                          if ep == "chronic" and (step, ci) in res["sig"] else np.nan)})
    for ep in EPS:
        sig = z[f"full__{ep}__sig"]
        for ci, col in enumerate(COLS):
            rows.append({"step": STEPS[3], "epoch": ep, "spout": col, "r_with_S3": 1.0, "split_half_r": np.nan,
                         "energy_outside_svd": 0.0, "sig_frac": float(sig[ci][stat1].mean())})
    T = pd.DataFrame(rows)
    T.to_csv(out / "steps.csv", index=False)
    new = {st: k for k, st in enumerate(CHILD_STEPS)}
    np.savez_compressed(out / "maps.npz", **{f"S{new[s]}__{ep}": m for (s, ep), m in res["maps"].items()},
                        **{f"S{new[s]}__sig{ci}": k for (s, ci), k in res["sig"].items()})
    print(res["info"])
    with pd.option_context("display.width", 200, "display.precision", 3):
        for ep in EPS:
            t = T[(T.epoch == ep) & T.spout.isin(["mean", "far_contra"])]
            print(f"\n=== {ep} ===")
            print(t.pivot_table(index="step", columns="spout", values=["r_with_S3", "split_half_r",
                                                                         "energy_outside_svd", "sig_frac"],
                                sort=False).to_string())
    fig_steps({(new[s], ep): m for (s, ep), m in res["maps"].items()},
              {(new[s], ci): k for (s, ci), k in res["sig"].items()}, ss, mask1, atlas, z, figs)
    for p in ("_mask1.npy", "_stat1.npy"):
        (out / p).unlink()
    print(f"wrote {out}")


def plot_only(src: Path, out: Path):
    """Figure + merged steps.csv from an earlier run (merged or pre-merge five-step format)."""
    out.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(src / "steps.csv")
    zs = np.load(src / "maps.npz")
    if "S1 + glue-free mask" in set(T.step):                    # pre-merge: S1 must equal S0, then drop it
        a, b = T[T.step == "S0 15r as rendered"], T[T.step == "S1 + glue-free mask"]
        num = ["r_with_S4", "split_half_r", "energy_outside_svd", "sig_frac"]
        assert np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), equal_nan=True), "S1 != S0: do not merge"
        assert all(np.array_equal(zs[f"S0__{k}"], zs[f"S1__{k}"], equal_nan=True)
                   for k in [*EPS, "sig0", "sig6"]), "S1 != S0 maps: do not merge"
        T = T[T.step != "S1 + glue-free mask"].copy()
        T["step"] = T.step.map(lambda x: STEPS[OLD_STEP[x]])
        T = T.rename(columns={"r_with_S4": "r_with_S3"})
        old = {0: 0, 1: 2, 2: 3}                                # output step -> key in the old npz
    else:
        old = {0: 0, 1: 1, 2: 2}
    T.to_csv(out / "steps.csv", index=False)
    maps = {(s, ep): zs[f"S{o}__{ep}"] for s, o in old.items() for ep in EPS}
    sigs = {(s, ci): zs[f"S{o}__sig{ci}"] for s, o in old.items() for ci in (0, 6)}
    atlas = np.load(S.SVD.parents[1] / "atlas.npy")
    mask1 = np.load(BASIS_DIR / "mask.npy").reshape(atlas.shape)
    z = np.load(SS_MAPS)
    ss = {ep: np.full((7, *mask1.shape), np.nan, np.float32) for ep in EPS}
    for ep in EPS:
        ss[ep][:, mask1] = z[f"full__{ep}__mean"]
    fig_steps(maps, sigs, ss, mask1, atlas, z, out)
    print(f"wrote {out} (plot-only from {src})")


def fig_steps(maps, sigs, ss, mask1, atlas, z, figs: Path = FIGS):
    """Chronic, mean over spouts + far contra: one column per step (S0-S2 from `maps` / `sigs`, keyed (output step,
    epoch) / (output step, column); S3 = the state-space map), the step's own contour."""
    import state_space_chronic_maps as C
    from wfield_local import beta_maps as bm
    from wfield_local.atlas_overlay import region_edges
    from wfield_local.figure_layout import save_png_svg
    brain = bm.allen_mask()
    fig, axs = plt.subplots(2, 4, figsize=(13.8, 7.2))
    for c, (ci, title) in enumerate([(0, "mean over spouts"), (6, "far contra")]):
        for s in range(4):
            ax = axs[c, s]
            if s < 3:
                M = maps[(s, "chronic")][ci]
                sig = sigs[(s, ci)]
                disp = np.isfinite(M) & brain
            else:
                M, sig, disp = ss["chronic"][ci], z["full__chronic__sig"][ci], mask1
            vmax = np.nanpercentile(np.abs(M[disp]), 99)
            C.draw(ax, M[disp], disp, region_edges(np.where(disp, atlas, 0)), vmax, sig=sig if sig.any() else None,
                   brain=brain)
            if c == 0:
                ax.set_title(STEPS[s], fontsize=10)
            if s == 0:
                ax.set_ylabel(title, fontsize=11)
    fig.suptitle(C._wrap("Why the state-space change maps are cleaner: chronic change vs own pre, lick-aligned, RESTnl "
                         "reference, mean of 4 animals; one change per column (left to right). S0 = the 15r activity "
                         "map (rebuild cache, glue-free brain mask since the 11-09 glue fix, per-position pre-RMS "
                         "units); S1 one unit per animal; S2 every session map through the cohort SVD (100 components); "
                         "S3 the state-space map (drift-removed signal, first-lick 0.1..0.5 s). Contour: nested bootstrap "
                         "animals -> sessions (2,000), each column's own test. Color scale per panel (99th pct |value|)."),
                 y=1.03, fontsize=9.5)
    save_png_svg(fig, figs / "ss_why_cleaner_lick.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--out", type=Path, default=None, help="write every output (tables + figure) here")
    ap.add_argument("--plot-only", type=Path, default=None,
                    help="redraw from SRC/steps.csv + SRC/maps.npz (no 15r cache read); needs --out")
    a = ap.parse_args()
    if a.plot_only:
        if not a.out:
            sys.exit("--plot-only needs --out (never overwrite the source run)")
        plot_only(a.plot_only, a.out)
    else:
        main(a.out, a.out) if a.out else main()

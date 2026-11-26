"""JOINT 3-D like-for-like lick read-out (CC-B 11-26; Priya 11-26 "sure let's try the joint 3d match").

Question: the page's like-for-like offsets (Fig 12.1 lateral, Fig 11.1 forward / down) match each lick to the animal's
PRE licks on ONE axis only. Could the lateral offset partly reflect post-stroke licks landing at a different forward /
down place than their lateral-matched pre references? Here the reference weight is a 3-D Gaussian on the (lateral,
forward, down) landing (diagonal covariance, per-axis sigma in mm), and the lateral (primary), forward and down offsets
are recomputed with it, side by side with the per-axis (1-D) numbers.

Everything else is the page's recipe (scripts/lick_matched_readout.py --ref gauss):
  * reference = the animal's PRE licks of the same lick set, any spout; pre sessions scored leave-one-session-out;
  * Nadaraya-Watson mean of the stored per-lick decoded read-outs (no re-extraction, no decoder fit);
  * kernel width chosen by held-out PRE prediction (pooled over lick sets and animals), per run;
  * session means (>= 8 licks), change vs pre = paired nested bootstrap (analysis_kit.boot_delta, rng 0) through the
    SAME functions (`lick_matched_readout.sessions` / `.tests`), per-animal changes and k/n.
Joint-only additions: n_eff (Kish, (sum w)^2 / sum w^2) per lick; licks with n_eff < 5 (an empty 3-D neighborhood)
get no reference and are dropped (counted). A fixed sigma = 0.1 mm on all three axes is the sensitivity check.

Inputs (all READ-ONLY, the lane E reader inputs of run clean_v37_sync60; see INPUTS):
  lick_order_preds_areas_splitr8.csv.gz   frozen KEY-row lateral read-out per lick (Fig 12.1's `dec`, landing `act`)
  where_3axis_preds_splitr8.csv.gz        3-D landing (act_lat / act_fwd / act_down) + 3-axis read-outs (Fig 11.1)
  iti_dock_audit.csv                      ITI subset flag iti_after_quiet_1s (the page's docked ITI bout starts)
  pose3d_4cam_3net_20261121_sync60/kinematics_fused/licks.csv.gz   checked: every 3-D landing is a row of it
The two per-lick caches hold the same licks in the same order within (set, session) (the 3-axis one has 23 extra ITI
rows); rows are paired by landing lateral + read-out (`align`).

First step = the REPRODUCTION CHECK: the 1-D lateral reference through this code must reproduce the production
matched_gauss_tests.csv / matched_gauss_sigma_cv.csv of both reader folders; the run stops otherwise.

Writes ONLY to --record (a NEW folder; refuses an existing folder it did not create). The tables are then copied into
the rebuild root by a second call, `--copy-to <new folder inside the rebuild root>`, run with WIDEFIELD_REBUILD_ROOT set
(the rebuild-root fence allows writes there only in that mode, and that mode refuses the lane E input reads, so the two
steps cannot share one process).
Usage: python scripts/joint3d_1126/lick_matched_joint3d.py --record <new folder> [--jobs N]
       WIDEFIELD_REBUILD_ROOT=<root> python scripts/joint3d_1126/lick_matched_joint3d.py --record <same> --copy-to <new>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

ENC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding")
PATCH = ENC / "iti_licks_patched_fusedorder_laneE_20261124"
INPUTS = {
    "lick_order": PATCH / "lick_order_preds_areas_splitr8.csv.gz",
    "three_axis": PATCH / "where_3axis_preds_splitr8.csv.gz",
    "dock_audit": PATCH / "iti_dock_audit.csv",
    "kin_fused": Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60/"
                      "kinematics_fused/licks.csv.gz"),
}
PROD = {"all": ENC / "iti_licks_readers2_laneE_20261124",            # all ITI licks (Fig 12.1 "all ITI", 11.1, 11.10)
        "dock": ENC / "iti_licks_readers2_dock_laneE_20261124"}      # WFIELD_ITI_FILTER=iti_after_quiet_1s (Fig 12.1)
DOCK_FLAG = "iti_after_quiet_1s"
EP = ["acute", "subacute", "chronic"]
AN = ["PS92", "PS93", "PS94", "PS95"]
SIGMAS_1D = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)        # lick_matched_readout.SIGMAS (the 1-D grid)
GRID_LAT = (0.05, 0.1, 0.15, 0.2, 0.3)
GRID_FD = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)            # forward / down widths tried (pre SD ~0.5-1.0 / 1.0-1.8 mm)
NEFF_MIN = 5.0
CAND_COVER = 0.95      # a grid point competes only if it gives >= 95 % of pre licks a reference (n_eff >= NEFF_MIN)
FIXED = (0.1, 0.1, 0.1)
# the page's per-axis widths of Fig 11.1 (lick_matched_angle_reach.py on this lineage: forward 0.1, down 0.2 mm)
SIGMA_1D_PAGE = {"forward": 0.1, "down": 0.2}
TARGETS = {"lateral": ("act", "dec"), "forward": ("act_fwd", "dec_fwd"), "down": ("act_down", "dec_down")}
X3 = ["act", "act_fwd", "act_down"]
LICKSETS = [("dock", "iti", "all", "ITI docked bout starts"), ("all", "iti", "all", "ITI, all licks"),
            ("all", "trial", "all", "cued, all licks"), ("all", "trial", "first", "cued, FIRST lick")]


# ------------------------------------------------------------------------------------------------------- kernels
def kernel_ref(xr: np.ndarray, yr: np.ndarray, x: np.ndarray, sig, chunk: int = 4000):
    """N-D Nadaraya-Watson with a diagonal Gaussian: weights exp(-0.5 * sum_d ((x_d - xr_d) / sig_d)^2). Returns
    (mean of yr, sum of weights, Kish n_eff); mean NaN where sum w < 3 (the page rule). With ONE axis the arithmetic is
    the same sequence as lick_matched_readout._kernel_mean (bitwise; pinned by the reproduction check)."""
    xr, x = np.atleast_2d(xr.T).T, np.atleast_2d(x.T).T
    sig = np.atleast_1d(np.asarray(sig, float))
    n = len(x)
    out, sw_, ne = np.full(n, np.nan), np.zeros(n), np.zeros(n)
    for i in range(0, n, chunk):
        q = ((x[i:i + chunk, None, 0] - xr[None, :, 0]) / sig[0]) ** 2
        for d in range(1, len(sig)):
            q = q + ((x[i:i + chunk, None, d] - xr[None, :, d]) / sig[d]) ** 2
        w = np.exp(-0.5 * q)
        sw = w.sum(1)
        out[i:i + chunk] = np.where(sw >= 3, (w @ yr) / np.maximum(sw, 1e-12), np.nan)
        sw_[i:i + chunk] = sw
        ne[i:i + chunk] = sw ** 2 / np.maximum((w * w).sum(1), 1e-300)
    return out, sw_, ne


def ref_loso(P: pd.DataFrame, xcols, y: str, sig) -> tuple[np.ndarray, np.ndarray]:
    """Per lick: (reference, n_eff). Post licks vs all the animal's pre licks of the set; pre licks vs the OTHER pre
    sessions (leave-one-session-out). Same loop as lick_matched_readout.gauss_ref."""
    ref, ne = np.full(len(P), np.nan), np.full(len(P), np.nan)
    for (_st, _a), g in P.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"].dropna(subset=xcols)     # a reference lick needs every matched coordinate
        post = g[g.epoch != "pre"]
        i = P.index.get_indexer(post.index)
        r, _, e = kernel_ref(pre[xcols].to_numpy(), pre[y].to_numpy(), post[xcols].to_numpy(), sig)
        ref[i], ne[i] = r, e
        for lab, h in pre.groupby("label"):
            o = pre[pre.label != lab]
            i = P.index.get_indexer(h.index)
            r, _, e = kernel_ref(o[xcols].to_numpy(), o[y].to_numpy(), h[xcols].to_numpy(), sig)
            ref[i], ne[i] = r, e
    return ref, ne


def choose_1d(P: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """lick_matched_readout.choose_sigma through `kernel_ref` (1-D, lateral)."""
    Q = P[P.epoch == "pre"]
    rows = []
    for sg in SIGMAS_1D:
        r, _ = ref_loso(Q.copy(), ["act"], "dec", sg)
        e = (Q.dec.to_numpy() - r) ** 2
        rows.append(dict(sigma=sg, mse=float(np.nanmean(e)), coverage=float(np.isfinite(r).mean())))
    T = pd.DataFrame(rows)
    return float(T.loc[T.mse.idxmin(), "sigma"]), T


# --- grid search worker (module level: spawn pickles by name) ---------------------------------------------------
def _grid_unit(item):
    """One (set, animal): held-out PRE references of all three targets for every grid point (leave-one-session-out).
    Per-axis kernels are precomputed once per held-out block and multiplied, so a grid point costs no exp()."""
    key, X, Y, lab = item
    combos = [(a, b, c) for a in GRID_LAT for b in GRID_FD for c in GRID_FD]
    R = np.full((len(combos), len(X), Y.shape[1]), np.nan, np.float32)
    NE = np.zeros((len(combos), len(X)), np.float32)
    for lb in np.unique(lab):
        hi = np.flatnonzero(lab == lb)
        oi = np.flatnonzero(lab != lb)
        for c0 in range(0, len(hi), 400):
            h = hi[c0:c0 + 400]
            D = [(X[h, None, d] - X[None, oi, d]) ** 2 for d in range(3)]
            E = [{s: np.exp(-0.5 * D[d] / s ** 2) for s in (GRID_LAT if d == 0 else GRID_FD)} for d in range(3)]
            del D
            for k, (a, b, c) in enumerate(combos):
                w = E[0][a] * E[1][b] * E[2][c]
                sw = w.sum(1)
                R[k, h] = np.where((sw >= 3)[:, None], (w @ Y[oi]) / np.maximum(sw, 1e-12)[:, None], np.nan)
                NE[k, h] = sw ** 2 / np.maximum((w * w).sum(1), 1e-300)
    return key, combos, R, NE


def choose_joint(P: pd.DataFrame, jobs: int) -> tuple[dict, pd.DataFrame]:
    """Per target: the diagonal sigma (lat, fwd, down) with the smallest held-out PRE squared error, pooled over sets
    and animals, among grid points that give >= CAND_COVER of pre licks a reference with n_eff >= NEFF_MIN; the error
    is compared on the pre licks EVERY such candidate covers (so a narrow kernel cannot win by skipping hard licks)."""
    from wfield_local import parallel
    Q = P[P.epoch == "pre"].dropna(subset=X3)
    ycols = [TARGETS[t][1] for t in TARGETS]
    items = [((st, a), g[X3].to_numpy(float), g[ycols].to_numpy(float), g.label.to_numpy())
             for (st, a), g in Q.groupby(["set", "animal"])]
    with parallel.parent_blas_kept():          # the parent's arithmetic (reference sums) stays as without the fan-out
        res, fail = parallel.fan_out(items, _grid_unit, jobs=jobs, label="set x animal",
                                     log=lambda m: print(m[:110], flush=True))
    if fail:
        raise SystemExit(f"grid search failed: {fail}")
    res = sorted(res, key=lambda it_r: it_r[1][0])               # completion order -> (set, animal) order
    res = [r for _it, r in res]
    combos = res[0][1]
    R = np.concatenate([r[2] for r in res], axis=1)
    NE = np.concatenate([r[3] for r in res], axis=1)
    Yall = np.concatenate([it[2] for it in sorted(items, key=lambda t: t[0])], axis=0)
    ok = np.isfinite(R[..., 0]) & (NE >= NEFF_MIN)
    cover = ok.mean(1)
    cand = cover >= CAND_COVER
    com = ok[cand].all(0)
    rows, chosen = [], {}
    for j, t in enumerate(TARGETS):
        err = (R[..., j] - Yall[None, :, j]) ** 2
        for k, cb in enumerate(combos):
            rows.append(dict(target=t, s_lat=cb[0], s_fwd=cb[1], s_down=cb[2], coverage_neff5=float(cover[k]),
                             candidate=bool(cand[k]), mse_common=float(np.mean(err[k, com])),
                             mse_own=float(np.mean(err[k, ok[k]])), n_common=int(com.sum())))
    T = pd.DataFrame(rows)
    for t in TARGETS:
        L = T[(T.target == t) & T.candidate]
        b = L.loc[L.mse_common.idxmin()]
        chosen[t] = (float(b.s_lat), float(b.s_fwd), float(b.s_down))
    return chosen, T


# ------------------------------------------------------------------------------------------------------- inputs
def _stamp(p: Path) -> dict:
    st = p.stat()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return dict(path=str(p), size=st.st_size, mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                sha256=h.hexdigest())


def align(L: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """Row of X for every row of L: within (set, session) walk both caches in order, pairing a row when the landing
    lateral is equal and the lateral read-outs agree within 0.2 mm (the 3-axis lateral ridge = the KEY read-out to
    ~1e-3 on ITI, exactly on cued); X rows with no partner (23 extra ITI rows) are skipped. -1 = unpaired."""
    out = np.full(len(L), -1)
    Xg = X.groupby(["set", "label"]).indices
    a_all, d_all = L.act.to_numpy(), L.dec.to_numpy()
    b_all, e_all = X.act_lat.to_numpy(), X.dec_lat.to_numpy()
    for key, li in L.groupby(["set", "label"]).indices.items():
        xi = Xg.get(key, np.array([], int))
        j = 0
        for i in li:
            j0 = j
            while j < len(xi) and not (abs(b_all[xi[j]] - a_all[i]) < 1e-9 and abs(e_all[xi[j]] - d_all[i]) < 0.2):
                j += 1
            if j < len(xi):
                out[i] = xi[j]
                j += 1
            else:
                j = j0
    return out


def kin_check(X: pd.DataFrame) -> pd.DataFrame:
    """Share of 3-D landings (act_lat, act_fwd, act_down) found as (side_ml, out_ap, down_dv) rows of the fused
    four-camera kinematics table of the same session (3-decimal match)."""
    K = pd.read_csv(INPUTS["kin_fused"], usecols=["session", "side_ml", "out_ap", "down_dv"])
    K["label"] = K.session.str.split("_").str[0] + "_" + K.session.str.split("_").str[1].str[4:]
    key = lambda df, c: (df.label + "|" + df[c[0]].round(3).map("{:.3f}".format) + "|" + df[c[1]].round(3).map("{:.3f}".format)  # noqa: E731
                         + "|" + df[c[2]].round(3).map("{:.3f}".format))
    ks = set(key(K.dropna(subset=["side_ml", "out_ap", "down_dv"]), ["side_ml", "out_ap", "down_dv"]))
    hit = key(X, ["act_lat", "act_fwd", "act_down"]).isin(ks)
    return X.assign(hit=hit.to_numpy()).groupby("set").hit.agg(["mean", "size"]).reset_index()


def load() -> tuple[pd.DataFrame, dict]:
    L = pd.read_csv(INPUTS["lick_order"])
    X = pd.read_csv(INPUTS["three_axis"])
    L["k"] = L.groupby(["set", "label"]).cumcount()          # = iti_licks.iti_subset_keep's k (cache order)
    j = align(L, X)
    info = dict(n_lick_order=len(L), n_three_axis=len(X), unpaired=int((j < 0).sum()))
    m = j >= 0
    for c, s in (("act_fwd", "act_fwd"), ("act_down", "act_down"), ("dec_fwd", "dec_fwd"), ("dec_down", "dec_down"),
                 ("dec_lat3", "dec_lat")):
        L[c] = np.nan
        L.loc[m, c] = X[s].to_numpy()[j[m]]
    dd = np.abs(L.dec.to_numpy()[m] - X.dec_lat.to_numpy()[j[m]])
    info.update(dec_absdiff_median=float(np.median(dd)), dec_absdiff_max=float(dd.max()))
    info["kin_check"] = kin_check(X).to_dict("records")
    A = pd.read_csv(INPUTS["dock_audit"], usecols=["label", "k", DOCK_FLAG])
    L = L.merge(A.rename(columns={DOCK_FLAG: "dockflag"}), on=["label", "k"], how="left")
    L["dockflag"] = L.dockflag.astype("boolean").fillna(False).astype(bool)
    return L, info


def run_table(L: pd.DataFrame, run: str) -> pd.DataFrame:
    """The per-lick table of one reader run, as lick_matched_readout.per_lick builds it (subset filter, dLand / dDec,
    cls). run 'dock' keeps ITI licks flagged iti_after_quiet_1s (iti_licks.iti_subset_keep), cued licks all."""
    P = L if run == "all" else L[(L.set != "iti") | L.dockflag]
    P = P.reset_index(drop=True).copy()
    ref = P[P.epoch == "pre"].groupby(["set", "animal", "pos"])[["act", "dec"]].mean()
    P = P.join(ref, on=["set", "animal", "pos"], rsuffix="_ref")
    P["dLand"], P["dDec"] = P.act - P.act_ref, P.dec - P.dec_ref
    P["cls"] = np.where(P.order == 0, "first", "later")
    return P


# ------------------------------------------------------------------------------------------------------- stats
def stats(P: pd.DataFrame, resid: np.ndarray):
    import lick_matched_readout as lmr
    Q = P.assign(resid=resid)
    S = lmr.sessions(Q)
    return S, lmr.tests(S)


def _cmp(a: pd.DataFrame, b: pd.DataFrame, keys) -> dict:
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or not (a[keys].astype(str).to_numpy() == b[keys].astype(str).to_numpy()).all():
        return dict(ok=False, why=f"rows differ ({len(a)} vs {len(b)})")
    num = [c for c in b.columns if c not in keys and pd.api.types.is_numeric_dtype(b[c])]
    mx = max(float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) if len(a) else 0.0 for c in num)
    nan_ok = all((a[c].isna().to_numpy() == b[c].isna().to_numpy()).all() for c in num)
    return dict(ok=bool(nan_ok and mx < 1e-9), max_abs_diff=mx, n_rows=len(a), exact=bool(nan_ok and mx == 0.0))


# ------------------------------------------------------------------------------------------------------- figure
def figure(C: pd.DataFrame, SS: dict, path: Path, chosen: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import lick_matched_readout as lmr
    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    cols = [(t, v) for t in TARGETS for v in ("1d", "joint")]
    fig, axs = plt.subplots(len(LICKSETS), len(cols), figsize=(3.1 * len(cols), 3.5 * len(LICKSETS)), squeeze=False,
                            sharey="row")
    for i, (run, st, cls, lab) in enumerate(LICKSETS):
        for j, (t, v) in enumerate(cols):
            ax = axs[i, j]
            S = SS[(run, t, v)]
            g = S[(S.set == st) & (S.scope == "all spouts") & (S.cls == cls)]
            T = C[(C.run == run) & (C.set == st) & (C.cls == cls) & (C.target == t) & (C.kernel == v)].set_index("epoch")
            for k, ep in enumerate(["pre", *EP]):
                lmr._dots(ax, g[g.epoch == ep], k, col[ep])
                if ep in T.index:
                    lmr._stars(ax, k, T.loc[ep], col[ep])
            ax.axhline(0, color="0.6", lw=0.7, ls="--")
            ax.set_xticks(range(4), ["pre", *EP], fontsize=7.5)
            sg = ((f"1-D, sigma {T.sigma.iloc[0]}" if v == "1d" else
                   "3-D, sigma " + "/".join(f"{s:g}" for s in chosen[(run, t)]) + " mm") if len(T) else "")
            ax.set_title(f"{lab}\n{t}: {sg}", fontsize=8, pad=24)
            if j == 0:
                ax.set_ylabel("read-out vs pre licks\nat the same place (mm)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("LIKE-FOR-LIKE offset, per-axis (1-D) vs JOINT 3-D landing match (lateral = primary, + = contralesional;"
                 " forward + = further forward; down + = further down). Reference = the animal's PRE licks of the same "
                 "set (any spout),\nGaussian-weighted on landing (1-D: that axis; 3-D: lateral / forward / down, sigma "
                 "per axis chosen by held-out pre prediction); pre sessions leave-one-session-out; 3-D licks with n_eff < 5"
                 " dropped. Frozen pre-stroke lick readers, fused four-camera licks, 3-D landing\nfrom "
                 "pose3d_4cam_3net_20261121_sync60. Small filled = sessions, open = animal means (circle PS92, square "
                 "PS93, triangle PS94, diamond PS95), bar = mean of animals; text = change vs the animal's own pre, 95% "
                 "paired nested bootstrap, * = CI excludes 0, k/n animals. No multiple-comparison correction.",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    from wfield_local.figure_layout import save_png_svg
    save_png_svg(fig, path, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- main
def _new_dir(p: Path) -> Path:
    mark = p / ".lick_matched_joint3d_owner"
    if p.exists() and any(p.iterdir()) and not mark.exists():
        raise SystemExit(f"{p} exists and was not created by this script: give a NEW folder (rule 11)")
    p.mkdir(parents=True, exist_ok=True)
    mark.write_text("lick_matched_joint3d.py\n")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="NEW run-record folder (results note, tables, figure)")
    ap.add_argument("--copy-to", default="", help="copy the record's tables / meta into this NEW folder and exit")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    if a.copy_to:
        import shutil
        dst = _new_dir(Path(a.copy_to))
        for f in sorted(Path(a.record).iterdir()):
            if f.suffix in (".csv", ".json", ".md", ".png") and f.is_file():
                shutil.copy2(f, dst / f.name)
                print("copied", f.name, flush=True)
        return 0
    out = rec = _new_dir(Path(a.record))
    t0 = time.time()
    pd.set_option("display.width", 250)
    L, info = load()
    info["inputs"] = {k: _stamp(p) for k, p in INPUTS.items()}
    print(json.dumps({k: v for k, v in info.items() if k != "inputs"}, indent=1), flush=True)
    if info["unpaired"] > 0.001 * len(L):
        raise SystemExit(f"{info['unpaired']} lick_order rows have no 3-D partner")
    all_C, SS, chosen, repro, neff_rows, cv_all = [], {}, {}, {}, [], []
    for run in ("all", "dock"):
        P = run_table(L, run)
        print(f"== run {run}: {len(P)} licks ({(P.set == 'iti').sum()} ITI)", flush=True)
        # ---- 1. reproduction check (1-D lateral) ------------------------------------------------------------
        sg, cv = choose_1d(P)
        r1, _ = ref_loso(P, ["act"], "dec", sg)
        S1, T1 = stats(P, P.dec.to_numpy() - r1)
        prodT = pd.read_csv(PROD[run] / "matched_gauss_tests.csv")
        prodS = pd.read_csv(PROD[run] / "matched_gauss_sessions.csv")
        prodCV = pd.read_csv(PROD[run] / "matched_gauss_sigma_cv.csv")
        repro[run] = dict(sigma=sg, tests=_cmp(T1, prodT, ["scope", "cls", "set", "measure", "epoch"]),
                          sessions=_cmp(S1, prodS, ["scope", "cls", "set", "label"]),
                          sigma_cv=_cmp(cv, prodCV, ["sigma"]))
        print("reproduction:", json.dumps(repro[run]), flush=True)
        if not all(repro[run][k]["ok"] for k in ("tests", "sessions", "sigma_cv")):
            (rec / "REPRODUCTION_FAILED.json").write_text(json.dumps(repro, indent=1))
            raise SystemExit(f"REPRODUCTION FAILED for run {run}: stopping (see {rec / 'REPRODUCTION_FAILED.json'})")
        # ---- 2. joint sigma by held-out pre prediction ----------------------------------------------------------
        ch, cvj = choose_joint(P, a.jobs)
        cvj["run"] = run
        cv_all.append(cvj)
        print(f"joint sigma (lat, fwd, down) mm by target: {ch}", flush=True)
        # 1-D held-out error of the page's widths on the same common licks, for the CV table
        for t in TARGETS:
            chosen[(run, t)] = ch[t]
        # ---- 3. offsets: 1-D (page widths), joint (chosen), joint fixed 0.1 ---------------------------------------
        for t, (x, y) in TARGETS.items():
            s1 = sg if t == "lateral" else SIGMA_1D_PAGE[t]
            variants = {"1d": ([x], s1), "joint": (X3, ch[t]), "joint_fixed01": (X3, FIXED)}
            for v, (xc, sig) in variants.items():
                if v == "1d" and t == "lateral":
                    r, ne = r1, np.full(len(P), np.nan)
                else:
                    r, ne = ref_loso(P, xc, y, sig)
                drop = np.isfinite(r) & (ne < NEFF_MIN) if v != "1d" else np.zeros(len(P), bool)
                r = np.where(drop, np.nan, r)
                S, T = stats(P, P[y].to_numpy() - r)
                SS[(run, t, v)] = S
                S.assign(run=run, target=t, kernel=v).to_csv(out / f"sessions_{run}_{t}_{v}.csv", index=False)
                T = T[T.measure == "resid"].assign(run=run, target=t, kernel=v,
                                                   sigma=str(sig if np.ndim(sig) == 0 else "/".join(f"{s:g}" for s in sig)))
                all_C.append(T)
                if v != "1d":
                    D = P.assign(neff=ne, has=np.isfinite(r), drop=drop, noref_w3=~np.isfinite(r) & ~drop)
                    for cls_ in ("all", "first"):
                        Dc = D if cls_ == "all" else D[D.cls == "first"]
                        g = Dc.groupby(["set", "animal", "epoch"])
                        neff_rows.append(g.agg(n=("neff", "size"), neff_median=("neff", "median"),
                                               neff_p10=("neff", lambda s: s.quantile(0.1)),
                                               dropped_neff5=("drop", "sum"), noref_sumw3=("noref_w3", "sum"))
                                         .reset_index().assign(run=run, target=t, kernel=v, cls=cls_))
        print(f"run {run} done ({time.time() - t0:.0f} s)", flush=True)
    C = pd.concat(all_C, ignore_index=True)
    NE = pd.concat(neff_rows, ignore_index=True)
    CV = pd.concat(cv_all, ignore_index=True)
    for name, df in (("tests_all", C), ("neff", NE), ("joint_sigma_cv", CV)):
        df.to_csv(rec / f"{name}.csv", index=False)
    # side-by-side table of the four lick sets
    rows = []
    for run, st, cls, lab in LICKSETS:
        for t in TARGETS:
            for ep in EP:
                r = {"lick_set": lab, "target": t, "epoch": ep}
                for v in ("1d", "joint", "joint_fixed01"):
                    q = C[(C.run == run) & (C.set == st) & (C.cls == cls) & (C.scope == "all spouts") & (C.target == t)
                          & (C.kernel == v) & (C.epoch == ep)]
                    if len(q):
                        q = q.iloc[0]
                        r[f"{v}_point"], r[f"{v}_lo"], r[f"{v}_hi"] = q.point, q.lo, q.hi
                        r[f"{v}_sig"] = bool(q.lo > 0 or q.hi < 0)
                        r[f"{v}_k_n"] = f"{int(q.n_dir)}/{int(q.n)}"
                        for an in AN:
                            r[f"{v}_{an}"] = q.get(f"animal_{an}", np.nan)
                rows.append(r)
    SBS = pd.DataFrame(rows)
    SBS.to_csv(rec / "side_by_side.csv", index=False)
    figure(C[C.scope == "all spouts"], SS, rec / "joint3d_vs_1d_offset.png", chosen)
    meta = dict(info=info, reproduction=repro, chosen={f"{k[0]}|{k[1]}": v for k, v in chosen.items()},
                neff_min=NEFF_MIN, cand_cover=CAND_COVER, grid_lat=GRID_LAT, grid_fd=GRID_FD, fixed=FIXED,
                sigma_1d_page=SIGMA_1D_PAGE, seconds=round(time.time() - t0))
    (rec / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(SBS.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    raise SystemExit(main())

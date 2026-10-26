"""PER-EPOCH POOLED PIXEL KERNEL FITS (handoff 000y item E, approved 10-25; Priya 10-25 night: "what happened to our
cross-epoch event kernel maps" -> "just queue locally"). The frozen encoder keeps ONE set of pre-stroke kernels, so it
cannot show WHERE a kernel moves after the stroke; this refits the same design per animal x epoch group and compares the
kernels' pixel maps with the early-pre kernels.

FRAME = the pixel (shared-SVD basis) frozen run `frozen_encoder_<tag>` (default the per-spout lick-kernel frame
`gate-attempted_v7g3_t3d_frA_ls_basis`): its templates, its per-animal basis P (stat pixels x 200, orthonormal) and its
column layout. The environment must be that run's (FaceRhythm mode A + FROZEN_LICKSPOUT=1; checked against the saved
run tag). Pre-session statistics are reused from fit/pre_stats.pkl; post sessions get the same `stats_job` pass
(sufficient statistics X'X, X'Y per session; saved to <out>/stats/<label>.pkl, so a re-run resumes).

GROUPS per animal: early_pre / late_pre = the first / second half of its August pre sessions by date (late_pre = the
DRIFT reference: no stroke between the halves), acute, subacute, chronic (`epochs.epoch_of`). MATCHED: every group uses
n = the animal's smallest group size, sessions evenly spaced through the group, so no epoch is estimated from more data
than another (pooled fits sharpen with n); the unmatched fit (all of a group's sessions) is a
secondary readout in the table (mode = all). Each group is also split into two halves of its matched sessions
(alternating) for a within-epoch SPLIT-HALF reliability: the ceiling a between-epoch correlation could reach.

FIT = ridge with the FROZEN fit's scaling (pooled pre SD per column) and its penalties (alphas re-derived by refitting
the frozen joint model on the same pre statistics; the re-derived beta is checked against the saved fit), so kernels
are in the same units in every group and differences are not penalty choices. Joint variant only.

KERNELS: target (pos_cue: each spout's deviation from the shared cue kernel; 0-2 s every 0.1 s, then to 7 s every 0.4
s), per-spout lick onset (lick_onset@<spout>, -0.5 .. +2.0 s every 0.1 s), and the shared cue kernel. Pixel kernel =
P @ (beta[col, :K] / scale[col]) (dF/F per unit of the raw regressor, as `frozen_kernel_maps.py`).
READOUTS (per animal x group x kernel x spout): Pearson r over stat pixels with the early-pre kernel, per lag and for
the maps stacked over a window (target 0.2-1.5 s after the cue, lick -0.1..+0.5 s around onset); the kernel's norm
relative to early pre; split-half r. A stroke-specific change = r vs early pre below late_pre's (drift) and below the
group's own split-half reliability.
Figures: epoch_kernels_r.png (lines; main), epoch_kernel_maps_<kernel>.png (rows = groups, cols = spouts, canonical
order, one colour scale per figure, mean of animals; image left = ipsilesional).
Usage: python scripts/epoch_kernels.py [--tag ...] [--jobs 3] [--stats-only] [--plot-only] [--labels L ...]
Output: <E>/frozen_encoder_<tag>/epoch_kernels/
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import AN, E  # noqa: E402

POS = ["close_contra", "close_ipsi", "close_mid", "far_contra", "far_ipsi", "far_mid"]   # sorted block order (pos_cue)
POS_SHOW = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]   # canonical order
GROUPS = ["early_pre", "late_pre", "acute", "subacute", "chronic"]
NCUE, NLICK = 35, 26
WIN = {"target": (0.2, 1.5), "lick": (-0.1, 0.5), "cue": (0.2, 1.5)}
SHOW_LAG = {"target": 1.0, "lick": 0.1, "cue": 0.5}
OUT = "epoch_kernels"


def cue_lag(k: int) -> float:
    return 0.1 * k if k <= 20 else 2.0 + 0.4 * (k - 20)


def lick_lag(k: int) -> float:
    return -0.5 + 0.1 * k


def kernel_cols(col_var) -> dict:
    """{(kernel, spout): [(column index, lag s), ...]} in the variant's column order."""
    cv = np.asarray(col_var).astype(str)
    out = {}
    ic = np.flatnonzero(cv == "pos_cue")
    for j, q in enumerate(POS):
        out[("target", q)] = [(int(ic[j * NCUE + k]), cue_lag(k)) for k in range(NCUE)]
    for q in POS_SHOW:
        il = np.flatnonzero(cv == f"lick_onset@{q}")
        if len(il):
            out[("lick", q)] = [(int(c), lick_lag(k)) for k, c in enumerate(il)]
    out[("cue", "all")] = [(int(c), cue_lag(k)) for k, c in enumerate(np.flatnonzero(cv == "cue"))]
    return out


def stats_worker(item):
    """One session's sufficient statistics (frozen_encoder.stats_job), written to <out>/stats/<label>.pkl."""
    from wfield_local import frozen_encoder as fe
    label, tp, spath, sdir = item
    f = Path(sdir) / f"{label}.pkl"
    if f.exists():
        return label
    s = fe.stats_job((label, tp, spath))
    s["stats_version"], s["run_mode"] = fe.STATS_VERSION, fe._run_tag()
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_bytes(pickle.dumps(s, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(f)
    return label


def groups_of(labs: list[str]) -> dict:
    """{group: sorted labels} for one animal (pre split by date into halves)."""
    from wfield_local import epochs
    by = {g: [] for g in GROUPS}
    pre = sorted(x for x in labs if epochs.epoch_of(x) == "pre")
    h = len(pre) // 2
    by["early_pre"], by["late_pre"] = pre[:h], pre[h:]
    for x in sorted(labs):
        ep = epochs.epoch_of(x)
        if ep in by and ep != "pre":
            by[ep].append(x)
    return by


def matched(by: dict) -> tuple[dict, int]:
    n = min(len(v) for v in by.values() if len(v)) if any(by.values()) else 0
    return {g: [v[i] for i in np.round(np.linspace(0, len(v) - 1, n)).astype(int)] if len(v) >= n and n else []
            for g, v in by.items()}, n


def solve_group(ref: dict, stats: list[dict]) -> np.ndarray:
    """Ridge with the frozen fit's scale, column transform and penalties on the pooled statistics of ``stats``."""
    from wfield_local import frozen_encoder as fe
    scale, C, src = ref["scale"], ref["C"], ref["src"]
    ok = np.isfinite(scale)[src]
    G = sum(C.T @ (s["XtX"] / np.outer(scale, scale)) @ C for s in stats)
    b = sum(C.T @ (fe._subset(s, ref["areas"])[0] / scale[:, None]) for s in stats)
    pen = np.array([fe.FPEN[v] for v in ref["vcol"]])
    lam = np.array([ref["alphas"][g] for g in pen]) * ok + (~ok) * 1.0
    return np.linalg.solve(G + np.diag(lam) + 1e-8 * np.eye(len(lam)), b)


def pixel_kernels(beta: np.ndarray, ref: dict, P: np.ndarray, cols: dict) -> dict:
    """{(kernel, spout): (lags, (n_lags, stat pixels))} -- pixel kernels of one fit."""
    K = P.shape[1]
    scale = ref["scale"][ref["src"]]
    out = {}
    for key, cl in cols.items():
        idx = np.array([c for c, _t in cl])
        out[key] = (np.array([t for _c, t in cl]), (beta[idx, :K] / scale[idx, None]) @ P.T)
    return out


def _r(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def fit_all(run: Path, out: Path, labels_by_an: dict) -> pd.DataFrame:
    from wfield_local import frozen_encoder as fe
    pre_stats = pickle.loads((run / "fit" / "pre_stats.pkl").read_bytes())
    pre_by = {s["label"]: s for s in pre_stats}
    rows, kern = [], {}
    for an in AN:
        labs = labels_by_an.get(an, [])
        if not labs:
            continue
        st = {x: pre_by[x] if x in pre_by else pickle.loads((out / "stats" / f"{x}.pkl").read_bytes())
              for x in labs if x in pre_by or (out / "stats" / f"{x}.pkl").exists()}
        ref = fe.fit_animal(sorted((pre_by[x] for x in labs if x in pre_by), key=lambda s: s["label"]), variant="joint")
        z = np.load(run / "fit" / f"{an}_joint.npz", allow_pickle=True)
        dev = float(np.abs(ref["beta_all"] - z["beta_all"]).max() / (np.abs(z["beta_all"]).max() or 1))
        print(f"  {an}: frozen joint re-derived, max |beta - saved| / max |beta| = {dev:.2e}; alphas {ref['alphas']}",
              flush=True)
        if dev > 1e-3:
            raise RuntimeError(f"{an}: re-derived frozen fit differs from the saved one ({dev:.2e})")
        P = pickle.loads((run / "fit" / f"spatial_{an}.pkl").read_bytes())["P"].astype(np.float64)
        cols = kernel_cols(ref["vcol"])
        allg = groups_of(list(st))
        for mode, (by, _n) in (("matched", matched(allg)), ("all", (allg, None))):
            print(f"  {an} [{mode}]: " + "; ".join(f"{g} {len(by[g])}" for g in GROUPS), flush=True)
            K = {}
            for g in GROUPS:
                if len(by[g]) < 2:
                    continue
                S = [st[x] for x in by[g]]
                K[g] = pixel_kernels(solve_group(ref, S), ref, P, cols)
                K[g + "|A"] = pixel_kernels(solve_group(ref, S[0::2]), ref, P, cols)
                K[g + "|B"] = pixel_kernels(solve_group(ref, S[1::2]), ref, P, cols)
            if mode == "matched":
                kern[an] = {g: {k: v[1].astype(np.float32) for k, v in d.items()} for g, d in K.items() if "|" not in g}
            if "early_pre" in K:
                rows += _similarity(K, an, mode, by)
    (out / "kernels.pkl").write_bytes(pickle.dumps(kern, protocol=pickle.HIGHEST_PROTOCOL))
    R = pd.DataFrame(rows)
    R.to_csv(out / "kernel_similarity.csv", index=False, float_format="%.4f")
    return R


def _similarity(K: dict, an: str, mode: str, by: dict) -> list[dict]:
    """Rows of kernel similarity with early pre (window-stacked + per lag), norm ratio and split-half r."""
    rows = []
    for g in GROUPS:
        if g not in K:
            continue
        for key, (lags, M) in K[g].items():
            E0 = K["early_pre"][key][1]
            w = (lags >= WIN[key[0]][0]) & (lags <= WIN[key[0]][1])
            base = {"animal": an, "mode": mode, "group": g, "kernel": key[0], "spout": key[1], "n": len(by[g])}
            rows.append({**base, "lag": "window", "r_early": _r(M[w], E0[w]),
                         "norm_rel": float(np.linalg.norm(M[w]) / (np.linalg.norm(E0[w]) or np.nan)),
                         "split_half": _r(K[g + "|A"][key][1][w], K[g + "|B"][key][1][w])})
            for i, t in enumerate(lags):
                rows.append({**base, "lag": round(float(t), 2), "r_early": _r(M[i], E0[i]),
                             "norm_rel": float(np.linalg.norm(M[i]) / (np.linalg.norm(E0[i]) or np.nan)),
                             "split_half": _r(K[g + "|A"][key][1][i], K[g + "|B"][key][1][i])})
    return rows


def figures(run: Path, out: Path, R: pd.DataFrame, labels_by_an: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from frozen_area_maps import area_image, outline

    from wfield_local import combined_encoding as ce
    W = R[(R.lag == "window") & (R["mode"] == "matched")]
    print("\n== per animal (rule 8): r with the early-pre kernel, window-stacked\n"
          + W.pivot_table(index=["kernel", "spout", "animal"], columns="group", values="r_early")
          .reindex(columns=GROUPS).round(2).to_string())
    # (1) lines: r vs early pre by group, solid = mean of animals, dashed = split-half reliability
    fams = [("target", POS_SHOW), ("lick", POS_SHOW), ("cue", ["all"])]
    fig, axs = plt.subplots(len(fams), 6, figsize=(17, 9), sharey=True, squeeze=False)
    gx = GROUPS[1:]
    x = np.arange(len(gx))
    for i, (kname, sps) in enumerate(fams):
        for j in range(6):
            ax = axs[i, j]
            if j >= len(sps):
                ax.axis("off")
                continue
            T = W[(W.kernel == kname) & (W.spout == sps[j])]
            if T.empty:
                ax.axis("off")
                continue
            r = T.pivot_table(index="animal", columns="group", values="r_early").reindex(columns=gx)
            sh = T.pivot_table(index="animal", columns="group", values="split_half").reindex(columns=gx)
            for _an, v in r.iterrows():
                ax.plot(x, v.values, color="#2b8cbe", lw=0.6, alpha=0.35)
            ax.plot(x, r.mean().values, color="#2b8cbe", lw=2.2, marker="o", label="r with early-pre kernel")
            ax.plot(x, sh.mean().values, color="0.4", lw=1.4, ls="--", marker="s", ms=4,
                    label="within-group split-half r (ceiling)")
            ax.axhline(0, color="0.7", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(["late pre", "acute", "sub", "chronic"], fontsize=7.5)
            ax.set_title(f"{kname}: {sps[j].replace('_', ' ')}", fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{kname} kernel, r (pixels, {WIN[kname][0]:g} to {WIN[kname][1]:g} s)", fontsize=8)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("NEURAL: PER-EPOCH pooled pixel kernel fits (frozen frame, frozen penalties; matched session counts per "
                 "group): correlation of each group's kernel map with the EARLY-PRE kernel. late pre = drift reference "
                 "(no stroke); dashed = the group's own split-half reliability. Thick = mean of animals, faint = animals.",
                 fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "epoch_kernels_r.png", dpi=110)
    plt.close(fig)
    # (2) maps: rows = groups, cols = spouts, at SHOW_LAG; mean of animals
    kern = pickle.loads((out / "kernels.pkl").read_bytes())
    an0 = next(iter(kern))
    lab0 = sorted(labels_by_an[an0])[0]
    _V, _ft, _Ub, flat, _pa, _nm, shp, _ps = ce.load_imaging(lab0)
    H, Wd = (int(v) for v in shp)
    img, _names = area_image(lab0)
    ol = outline(img)
    for kname, sps in fams:
        lags = np.array([lick_lag(k) for k in range(NLICK)]) if kname == "lick" else np.array([cue_lag(k) for k in range(NCUE)])
        li = int(np.argmin(np.abs(lags - SHOW_LAG[kname])))
        grid = {}
        for g in GROUPS:
            for q in sps:
                ims = []
                for an, d in kern.items():
                    if g in d and (kname, q) in d[g]:
                        lab = sorted(labels_by_an[an])[0]
                        _V, _ft, _Ub, fl, _pa, _nm, _sh, pst = ce.load_imaging(lab)
                        m = np.full(H * Wd, np.nan)
                        m[fl[pst]] = d[g][(kname, q)][li]
                        ims.append(m.reshape(H, Wd))
                if ims:
                    grid[(g, q)] = np.nanmean(ims, 0) * 1e3
        if not grid:
            continue
        lim = np.nanpercentile(np.abs(np.stack(list(grid.values()))), 99) or 1.0
        fig, axs = plt.subplots(len(GROUPS), len(sps), figsize=(2.6 * len(sps) + 0.8, 2.6 * len(GROUPS)), squeeze=False)
        for i, g in enumerate(GROUPS):
            for j, q in enumerate(sps):
                ax = axs[i, j]
                ax.set_xticks([]); ax.set_yticks([])
                if (g, q) not in grid:
                    ax.axis("off")
                    continue
                im = ax.imshow(grid[(g, q)], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
                ax.imshow(ol, cmap="Greys", vmin=0, vmax=1.2, alpha=0.5, interpolation="nearest")
                if i == 0:
                    ax.set_title(q.replace("_", " "), fontsize=9)
                if j == 0:
                    ax.set_ylabel(g.replace("_", " "), fontsize=9)
        cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01)
        cb.set_label("beta x1e3 (dF/F per unit)", fontsize=8)
        fig.suptitle(f"NEURAL: {kname} kernel at {lags[li]:+.1f} s, per-epoch pooled fits (matched sessions), mean of "
                     "animals; one colour scale. Image left = ipsilesional.", fontsize=9.5)
        fig.savefig(out / f"epoch_kernel_maps_{kname}.png", dpi=110)
        plt.close(fig)
    print("->", out)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import config
    from wfield_local import frozen_encoder as fe
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA_ls_basis")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    out = run / OUT
    assert_writable(out)
    (out / "stats").mkdir(parents=True, exist_ok=True)
    pre_stats = pickle.loads((run / "fit" / "pre_stats.pkl").read_bytes())
    if pre_stats[0]["run_mode"] != fe._run_tag():
        raise RuntimeError(f"environment run tag {fe._run_tag()} != the frame's {pre_stats[0]['run_mode']}: set the "
                           "frame's environment (frA + FROZEN_LICKSPOUT=1)")
    pre = {s["label"] for s in pre_stats}
    labels = sorted(a.labels or (p.stem for p in (run / "joint" / "sessions").glob("*.json")))
    labels = [x for x in labels if x[-4:] >= "0801"]
    by_an: dict = {}
    for x in labels:
        by_an.setdefault(config.animal_of(x), []).append(x)
    if not a.plot_only:
        todo = [(x, str(run / "fit" / "templates.pkl"), str(run / "fit"), str(out / "stats")) for x in labels
                if x not in pre and not (out / "stats" / f"{x}.pkl").exists()]
        print(f"[epoch_kernels] {len(labels)} sessions ({len(pre & set(labels))} pre reused), {len(todo)} to compute "
              f"-> {out}", flush=True)
        if todo:
            _got, fail = ak.fan_sessions(todo, stats_worker, jobs=a.jobs, label="stats")
            for it, err in fail:
                print(f"  !! stats {it[0]}: {err}", flush=True)
        if a.stats_only:
            return 0
        R = fit_all(run, out, by_an)
    else:
        R = pd.read_csv(out / "kernel_similarity.csv")
        R["lag"] = R["lag"].astype(str)
    R["lag"] = R["lag"].astype(str)
    figures(run, out, R, by_an)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

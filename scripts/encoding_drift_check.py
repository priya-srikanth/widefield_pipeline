"""Slow-drift diagnostic for the continuous-time combined encoding model (Priya 2026-10-18: "measure how common it is
- and make some figures that show me the problem graphically").

    python scripts/encoding_drift_check.py [--jobs 4] [--run lp_r4_heatmap_20261013__clean_v36]

Why: the adopted preprocessing (`hemo_variants.ADOPTED` = meegkit_hpfit) detrends with an order-10 robust polynomial,
50 % cutoff ~43 min, chosen (08-13) to leave the 1-2 min position-block band untouched. Drift FASTER than that stays in
by design. Trial-contrast analyses (decoders, maps) are blind to it -- an offset shared by all positions cancels or is
label-independent noise. The encoding model is not: it explains every frame in continuous time and cross-validates on
CONTIGUOUS trial blocks (`combined_encoding.FOLDS`), so a held-out block whose mean sits SDs away from training gives a
NEGATIVE R2 even at maximal penalties (PS92_0824: full R2 -0.22).

Per session, on the cortex-mean signal (adopted SVTcorr x U, eroded stat mask) over the MODEL ROWS (engaged trials,
strobe - 0.5 s .. stop + 1 s, as `combined_encoding.run_session` selects them):
  block_range_sd  range of the FOLDS contiguous-block means / SD (what the CV sees)
  first_block_sd  the first block's mean / SD (an early-session transient)
  slow_frac       fraction of variance in periods >= SLOW_S (cosine basis over the session: what a nuisance drift basis
                  with that cutoff would remove; position blocks are 57-121 s, so >= 300 s cannot absorb them)
and, where the encoding session exists, its full-model cortex R2 (all frames).

Outputs -> <LP_analysis>/<run>/imaging_encoding/qc/: drift_sessions.csv, drift_traces.npz (10-s bins),
drift_examples.png, drift_cohort.png, drift_blocks_heatmap.png (+ svg/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SLOW_S = 300.0
BIN_S = 10.0


def _cos_basis(t: np.ndarray, min_period_s: float) -> np.ndarray:
    """DCT-II style cosine regressors over [t0, t1] with period >= min_period_s (incl. the constant)."""
    T = t[-1] - t[0]
    n = int(np.floor(2 * T / min_period_s))
    u = (t - t[0]) / T
    return np.column_stack([np.cos(np.pi * k * u) for k in range(n + 1)])


def session_drift(label: str) -> dict:
    """Module-level worker: drift metrics + the 10-s binned cortex-mean trace for one session."""
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import epochs
    from wfield_local.orofacial_cohort import _repo_on_path
    from wfield_local.paths import PathResolver
    _repo_on_path()
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    y_all = (V @ Ub[pstat].mean(0)).astype(np.float64)
    animal, date = config.animal_of(label), "2026" + label.split("_")[-1]
    B = ce.load_behaviour(animal, date, PathResolver())
    T = B.trials
    E = T[T.engaged & np.isfinite(T.cue_s)].reset_index(drop=True)
    st = np.where(np.isfinite(E.strobe_s), E.strobe_s, E.cue_s - 3.0)
    rows = np.zeros(len(ft), bool)
    tr_of = np.full(len(ft), -1)
    for k, (a, b_) in enumerate(zip(st - 0.5, E.stop_s.to_numpy(float) + 1.0)):
        m = (ft >= a) & (ft <= b_)
        rows |= m
        tr_of[m] = k
    ridx = np.flatnonzero(rows)
    y, t = y_all[ridx], ft[ridx]
    sd = y.std()
    fold_of_trial = np.zeros(len(E), int)
    for k, bl in enumerate(np.array_split(np.arange(len(E)), ce.FOLDS)):
        fold_of_trial[bl] = k
    fold = fold_of_trial[tr_of[ridx]]
    bm = np.array([y[fold == k].mean() for k in range(ce.FOLDS)])
    Xc = _cos_basis(t, SLOW_S)
    beta, *_ = np.linalg.lstsq(Xc, y - y.mean(), rcond=None)
    slow = Xc @ beta
    # 10-s binned traces over the WHOLE recording (for the figures), plus the slow fit on the model rows
    edges = np.arange(ft[0], ft[-1] + BIN_S, BIN_S)
    idx = np.clip(np.digitize(ft, edges) - 1, 0, len(edges) - 2)
    cnt = np.bincount(idx, minlength=len(edges) - 1)
    yb = np.bincount(idx, weights=y_all, minlength=len(edges) - 1) / np.maximum(cnt, 1)
    sb_idx = idx[ridx]
    sb = np.bincount(sb_idx, weights=slow + y.mean(), minlength=len(edges) - 1)
    sc = np.bincount(sb_idx, minlength=len(edges) - 1)
    return {"label": label, "animal": animal, "date": date, "epoch": epochs.epoch_of(label),
            "day": epochs.days_since_stroke(label), "n_trials": int(len(E)), "minutes": float((t[-1] - t[0]) / 60),
            "block_range_sd": float(np.ptp(bm) / sd), "first_block_sd": float((bm[0] - y.mean()) / sd),
            "slow_frac": float(slow.var() / y.var()), "block_means_sd": ((bm - y.mean()) / sd).tolist(),
            "trace_t_min": ((edges[:-1] + BIN_S / 2 - ft[0]) / 60).tolist(),
            "trace": np.where(cnt > 0, (yb - y.mean()) / sd, np.nan).tolist(),
            "slow_fit": np.where(sc > 0, (sb / np.maximum(sc, 1) - y.mean()) / sd, np.nan).tolist(),
            "block_t_min": [float((t[fold == k].mean() - ft[0]) / 60) for k in range(ce.FOLDS)]}


def _encoding_r2(sess_dir: Path, label: str) -> float:
    f, j = sess_dir / f"{label}_regions.csv", sess_dir / f"{label}.json"
    if not (f.exists() and j.exists()) or "video_k" not in json.loads(j.read_text(encoding="utf-8")):
        return np.nan                                            # missing, or an output of the old (invalid) design
    r = pd.read_csv(f)
    r = r[(r.area == "cortex") & (r.window == "all") & (r.model == "full")]
    return float(r.r2.iloc[0]) if len(r) else np.nan


def figures(S: pd.DataFrame, res: dict, out: Path) -> None:
    from wfield_local import orofacial_cohort as oc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11})
    acol = {a: c for a, c in zip(sorted(S.animal.unique()), ("#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"))}
    # 1. examples: the cleanest, the worst few, and PS92_0822 / 0824 / 0826 if present
    picks = [x for x in ("PS92_0822", "PS92_0824", "PS92_0826") if x in res]
    worst = [x for x in S.sort_values("block_range_sd", ascending=False).label if x not in picks][:3]
    best = [x for x in S.sort_values("block_range_sd").label if x not in picks][:1]
    ex = best + picks + worst
    fig, axs = plt.subplots(len(ex), 1, figsize=(13, 2.0 * len(ex) + 1.2), squeeze=False)
    for ax, lab in zip(axs[:, 0], ex):
        r = res[lab]
        tm, tr, sf = (np.asarray(r[k], float) for k in ("trace_t_min", "trace", "slow_fit"))
        ax.plot(tm, tr, color="0.55", lw=0.6, label="cortex mean (10-s bins, SD units)")
        ax.plot(tm, sf, color="#d01c8b", lw=2.0, label=f"component slower than {SLOW_S / 60:.0f} min")
        bt, bmv = np.asarray(r["block_t_min"]), np.asarray(r["block_means_sd"])
        ax.plot(bt, bmv, "s", color="k", ms=6, label="CV block means (10 contiguous blocks)")
        ax.axhline(0, color="0.8", lw=0.7)
        e2 = S.set_index("label").loc[lab]
        ax.set_title(f"{lab} ({e2.epoch}, day {e2.day}): block range {e2.block_range_sd:.2f} SD, slow variance "
                     f"{100 * e2.slow_frac:.0f} %, encoding full R$^2$ {e2.full_r2:.2f}", fontsize=10.5, loc="left")
        ax.set_ylabel("SD")
        ax.set_ylim(-3.5, 3.5)
    axs[-1, 0].set_xlabel("minutes from recording start")
    axs[0, 0].legend(loc="upper right", fontsize=9, ncol=3, frameon=False)
    fig.suptitle("Slow drift in the cortex-mean signal (adopted SVTcorr): what the encoding model's contiguous-block "
                 "cross-validation sees", fontsize=12)
    fig.tight_layout()
    oc.save_png_svg(fig, out / "drift_examples.png", dpi=130)
    plt.close(fig)
    # 2. cohort: metrics vs day per animal, and vs encoding R2
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, g in S.groupby("animal"):
        axs[0].plot(g.day, g.block_range_sd, "o", color=acol[a], ms=5, label=a)
        axs[1].plot(g.day, 100 * g.slow_frac, "o", color=acol[a], ms=5, label=a)
        axs[2].plot(100 * g.slow_frac, g.full_r2, "o", color=acol[a], ms=5, label=a)
    for ax in axs[:2]:
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("day from stroke")
    axs[0].set_ylabel("range of 10 CV-block means (SD)")
    axs[0].set_title("block-mean range (what the CV sees)")
    axs[1].set_ylabel(f"% variance slower than {SLOW_S / 60:.0f} min")
    axs[1].set_title("slow-drift share of the signal")
    axs[2].set_xlabel(f"% variance slower than {SLOW_S / 60:.0f} min")
    axs[2].set_ylabel("encoding full R$^2$ (cortex, all frames)")
    axs[2].axhline(0, color="0.7", lw=0.7)
    axs[2].set_title("drift vs encoding fit (sessions run so far)")
    axs[0].legend(frameon=False)
    fig.suptitle(f"Slow drift across all {len(S)} encoding sessions", fontsize=12)
    fig.tight_layout()
    oc.save_png_svg(fig, out / "drift_cohort.png", dpi=130)
    plt.close(fig)
    # 3. heatmap of the 10 block means, sessions sorted by animal / day
    S2 = S.sort_values(["animal", "day"])
    Bm = np.array([res[x]["block_means_sd"] for x in S2.label])
    fig, ax = plt.subplots(figsize=(7.5, 0.16 * len(S2) + 1.8))
    im = ax.imshow(Bm, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(S2)))
    ax.set_yticklabels([f"{x} ({e[:3]})" for x, e in zip(S2.label, S2.epoch)], fontsize=6.5)
    ax.set_xticks(range(Bm.shape[1]))
    ax.set_xticklabels([str(k + 1) for k in range(Bm.shape[1])])
    ax.set_xlabel("CV block (contiguous trial blocks, session order)")
    fig.colorbar(im, ax=ax, label="block mean (SD of the session)", shrink=0.5)
    ax.set_title("Block means of the cortex-mean signal: rows = sessions", fontsize=11)
    fig.tight_layout()
    oc.save_png_svg(fig, out / "drift_blocks_heatmap.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("labels", nargs="*")
    a = ap.parse_args(argv)
    import os
    os.environ[ce.RUN_ENV] = a.run
    base = oo.root(PathResolver()) / a.run / "imaging_encoding"
    out = base / "qc"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = a.labels or ce.sessions_available()
    got, fail = ak.fan_sessions(labels, session_drift, jobs=a.jobs)
    res = {lab: r for lab, r in got}
    for it, err in fail:
        print(f"  FAILED {it}: {err}")
    S = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in res.values()])
    S["full_r2"] = [_encoding_r2(base / "movement_first" / "sessions", x) for x in S.label]
    S = S.sort_values(["animal", "day"]).reset_index(drop=True)
    S.to_csv(out / "drift_sessions.csv", index=False, float_format="%.4g")
    np.savez_compressed(out / "drift_traces.npz", **{f"{k}|{x}": np.asarray(res[x][k], float) for x in res
                                                     for k in ("trace_t_min", "trace", "slow_fit")})
    print(S[["label", "epoch", "day", "minutes", "block_range_sd", "first_block_sd", "slow_frac", "full_r2"]]
          .round(3).to_string(index=False))
    for thr in (1.0, 1.5, 2.0):
        print(f"block range >= {thr} SD: {int((S.block_range_sd >= thr).sum())} of {len(S)} sessions")
    print("per animal x epoch, median block range (SD):")
    print(S.groupby(["animal", "epoch"]).block_range_sd.median().unstack("epoch").round(2).to_string())
    figures(S, res, out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

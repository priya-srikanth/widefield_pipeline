"""Does the state-space between-spout change survive subtracting the FROZEN ENCODER's lick-locked prediction? (handoff
11-09 ~12:15 open item a; Priya 11-09 "go ahead with all")

The movement controls of Figs 9.1-9.3 use bout-summary covariates (per-trial means); the encoder models movement with
lick-locked kernels (onset / tongue / jaw magnitude deviations, spout-locked templates, direction, contact), and found the
acute far-contra drop carried by its lick + contact groups. Here the state-space readouts are computed on the ENCODER'S
OWN saved per-trial values, before and after removing its predicted movement contributions -- like-for-like with the
encoder by construction.

Source: the frozen encoder run `frozen_encoder_gate-attempted_v7g3_t3d_spont_frA` (the run `state_space_recovery.py`
reads), `joint/sessions/<label>.npz` `trial_parts` (parts x trials x windows x 59 Allen-area targets), the LICK window =
mean over -0.1..+1.0 s after the first lick (`frozen_encoder.MUSALL_WINDOWS`; NOT the 0.1-0.5 s of Figs 9.x; the
predictions are not saved per frame, and rebuilding them reloads every session). Trials: the run's attempted gate
(contact + incomplete first licks), engaged; pre sessions predicted with leave-session-out betas (`frozen_from` loso),
post sessions with the all-pre fit. Conditions:
  data        the measured area responses
  -lick       minus lick + exec_locked + exec_within + contact (the encoder's lick-locked movement groups)
  -face       minus fr + fr_dir (FaceRhythm A inside the encoder)
  -both       minus all of the above
Readouts as Figs 9.1 / 9.3 on the 58-area vector (`cortex` dropped): per session x spout the trial mean (>= 5 trials);
full = mean over spouts, spout = each spout minus the session's mean over spouts (a per-session constant cancels, so the
encoder's signal reference does not matter here); change vs own pre in units of the animal's pre data response RMS over
areas; kept = along the data change's own direction ((v . v0) / (v0 . v0), per animal, cohort mean); key areas with
`boot_delta` (animals -> sessions, 2,000; per-animal values printed first). Image / hemisphere: left = ipsilesional.

    python scripts/state_space_encoder_subtract.py
Outputs: <state_space_recovery>/encoder_subtract/{kept,areas}.csv; page_figs/ss_encoder_subtract.png. EXPLORATORY.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402
from wfield_local import sig_marks  # noqa: E402
from wfield_local.figure_layout import save_png_svg  # noqa: E402
from wfield_local.orofacial_cohort import EPOCH_LINE  # noqa: E402

OUT = S.ROOT / "encoder_subtract"
EPS = ["acute", "subacute", "chronic"]
MIN_TR = 5
COND = {"data": [], "-lick": ["lick", "exec_locked", "exec_within", "contact"], "-face": ["fr", "fr_dir"],
        "-both": ["lick", "exec_locked", "exec_within", "contact", "fr", "fr_dir"]}
KEY = [("SSp-m", "left"), ("SSp-n", "left"), ("MOp", "left"), ("SSp-m", "right"), ("SSp-n", "right"), ("MOp", "right")]


def run_dir():
    hits = sorted(S.ENCODING.glob("*daqlk/frozen_encoder_gate-attempted_v7g3_t3d_spont_frA/joint/sessions"))
    if len(hits) != 1:
        sys.exit(f"state_space_encoder_subtract: expected one run folder, found {hits}")
    return hits[0]


def main():
    src = run_dir()
    rows, areas = {}, None
    for f in sorted(src.glob("PS9*.npz")):
        lab = f.stem
        if lab.endswith("_trials") or not (S.SVD / f"{lab}.npz").exists():
            continue
        ep = S._epoch(lab, str(np.load(S.SVD / f"{lab}.npz", allow_pickle=True)["epoch"]))   # the state-space epochs
        if ep not in ("pre", *EPS):
            continue
        z = np.load(f, allow_pickle=True)
        parts = [str(p) for p in z["parts"]]
        ar = [str(a) for a in z["areas"]]
        keep_a = [i for i, a in enumerate(ar) if a != "cortex"]
        areas = [ar[i] for i in keep_a]
        wl = [str(w) for w in z["trial_windows"]].index("lick")
        tp = z["trial_parts"][:, :, wl][:, :, keep_a].astype(float)                  # parts x trials x areas
        pos = np.asarray(z["trial_pos"]).astype(str)
        M = {}
        for c, sub in COND.items():
            v = tp[parts.index("data")] - sum((tp[parts.index(p)] for p in sub), np.zeros_like(tp[0]))
            sp = np.full((6, len(keep_a)), np.nan)
            for i, p in enumerate(S.POS):
                m = (pos == p) & np.isfinite(v).all(1)
                if m.sum() >= MIN_TR:
                    sp[i] = v[m].mean(0)
            M[c] = sp
        rows.setdefault(lab.split("_")[0], []).append((lab, ep, M))
    out_rows, kept = [], []
    rng = np.random.default_rng(0)
    res = rows
    print({an: {e: sum(1 for s in v if s[1] == e) for e in ["pre", *EPS]} for an, v in res.items()}, flush=True)

    def stack(an, ep, c, version):
        X = np.stack([M[c] for lab, e, M in res[an] if e == ep])                     # n x 6 x A
        if version == "spout":
            X = X - np.nanmean(X, axis=1, keepdims=True)
        return X
    unit = {an: float(np.sqrt(np.nanmean(np.nanmean(stack(an, "pre", "data", "full"), 0) ** 2))) for an in res}
    for version, col in [("full", "mean"), ("spout", "far_contra"), ("spout", "close_ipsi")]:
        for ep in EPS:
            V = {}
            for c in COND:
                V[c] = {}
                for an in res:
                    pre, post = stack(an, "pre", c, version), stack(an, ep, c, version)
                    d = (np.nanmean(post, 0) - np.nanmean(pre, 0)) / unit[an]
                    V[c][an] = np.nanmean(d, 0) if col == "mean" else d[S.POS.index(col)]
            for c in COND:
                if c == "data":
                    continue
                k = {an: float(V[c][an] @ V["data"][an] / (V["data"][an] @ V["data"][an])) for an in res}
                kept.append({"version": version, "column": col, "epoch": ep, "condition": c,
                             "kept_mean": float(np.mean(list(k.values()))), **{f"k_{a}": v for a, v in k.items()}})
            for c in COND:
                for area, hemi in KEY:
                    j = areas.index(f"{area}_{hemi}")
                    pre = {an: [((m[c] - (np.nanmean(m[c], 0) if version == "spout" else 0)) / unit[an])
                                [S.POS.index(col) if col != "mean" else slice(None), j].mean()
                                for lab, e, m in res[an] if e == "pre"] for an in res}
                    post = {an: [((m[c] - (np.nanmean(m[c], 0) if version == "spout" else 0)) / unit[an])
                                 [S.POS.index(col) if col != "mean" else slice(None), j].mean()
                                 for lab, e, m in res[an] if e == ep] for an in res}
                    pre = {a: [x for x in v if np.isfinite(x)] for a, v in pre.items()}
                    post = {a: [x for x in v if np.isfinite(x)] for a, v in post.items() if pre.get(a)}
                    post = {a: v for a, v in post.items() if v}
                    iv = kit.boot_delta(post, pre, rng)
                    per = {a: round(float(np.mean(post[a]) - np.mean(pre[a])), 3) for a in post}
                    out_rows.append({"version": version, "column": col, "epoch": ep, "condition": c, "area": area,
                                     "hemi": "ipsi" if hemi == "left" else "contra", "change": iv.point, "lo": iv.lo,
                                     "hi": iv.hi, "star": bool(iv.lo > 0 or iv.hi < 0),
                                     "k_same": sum(np.sign(v) == np.sign(iv.point) for v in per.values()),
                                     "per_animal": per})
    OUT.mkdir(parents=True, exist_ok=True)
    K = pd.DataFrame(kept)
    A = pd.DataFrame(out_rows)
    K.to_csv(OUT / "kept.csv", index=False)
    A.to_csv(OUT / "areas.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3, "display.max_rows", 400):
        print(K.to_string(index=False))
        print(A[A.area.isin(["SSp-m", "SSp-n"])].drop(columns="per_animal").to_string(index=False))

    # figure: key-area changes per condition (rows = readout, cols = epoch)
    reads = [("full", "mean"), ("spout", "far_contra"), ("spout", "close_ipsi")]
    fig, axs = plt.subplots(len(reads), 3, figsize=(15, 3.6 * len(reads)), squeeze=False, sharey="row")
    cc = {"data": "0.2", "-lick": "#d95f02", "-face": "#7570b3", "-both": "#1b9e77"}
    for i, (version, col) in enumerate(reads):
        for j, ep in enumerate(EPS):
            ax = axs[i, j]
            for k, c in enumerate(COND):
                g = A[(A.version == version) & (A.column == col) & (A.epoch == ep) & (A.condition == c)]
                x = np.arange(len(g)) + (k - 1.5) * 0.18
                ax.errorbar(x, g.change, yerr=[g.change - g.lo, g.hi - g.change], fmt="o", color=cc[c], ms=4,
                            lw=1.2, capsize=0, label=c if (i, j) == (0, 0) else None)
                for xx, st in zip(x, g.star):
                    if st:
                        sig_marks.star_top(ax, xx, "*", color=cc[c], fontsize=8)
                kk = K[(K.version == version) & (K.column == col) & (K.epoch == ep) & (K.condition == c)]
                if len(kk):
                    ax.text(0.99, 0.04 + 0.08 * k, f"{c} kept {kk.kept_mean.iloc[0]:.2f}", transform=ax.transAxes,
                            ha="right", fontsize=7.5, color=cc[c])
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(range(len(KEY)), [f"{a}\n{'ipsi' if h == 'left' else 'contra'}" for a, h in KEY], fontsize=8)
            ax.set_title(f"{ep}: {'mean over spouts' if col == 'mean' else col.replace('_', ' ') + ' (spout-specific)'}",
                         fontsize=9.5, color=EPOCH_LINE[ep])
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel("change vs own pre\n(pre data response RMS)")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(textwrap.fill("State-space readouts on the FROZEN ENCODER's own per-trial values, before / after removing its "
                 "predicted movement (lick-locked groups, FaceRhythm, both); first-lick window -0.1..+1.0 s, Allen "
                 "areas; kept = along the data change; * = boot_delta 95 % CI excludes 0", 170), fontsize=10)
    fig.tight_layout()
    fp = S.ROOT / "page_figs" / "ss_encoder_subtract.png"
    sig_marks.reserve_headroom(fig)                  # stars in the top band, no top spine (Priya 10-30 / 11-03)
    save_png_svg(fig, fp)
    print(f"wrote {fp}", flush=True)


if __name__ == "__main__":
    main()

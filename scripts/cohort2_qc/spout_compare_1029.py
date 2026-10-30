"""Per-spout coding, PS118 (Thy1-GCaMP6s) vs PS121 / PS122 (RiboL1-jGCaMP8s), with PS120 as the blood-flow-only floor
(Priya 2026-10-30: "what about per-spout 118 vs 121-122"). 10-29, lag + swap-masked correction, DAQ events, dark-relative
mask, native pixels.

Per session:
  * DECODING at matched trials: N_PER trials per spout (the minimum available across the four sessions), cue-window
    feature = mean over 0-1 s minus -1-0 s of the first K_COMP SVTcorr components (the same K for everyone; U is each
    session's own basis, so features are session-specific but equal in number). Standardised logistic regression
    (C 0.5), 5-fold stratified CV, balanced accuracy, mean +- SD over N_REP random trial subsamples. Chance 1/6.
  * SPECIFICITY of the per-spout cue maps (0-1 s): spout-specific share = sum ||m_p - mean_p m||^2 / sum ||m_p||^2
    (0 = all spouts give the same map); L-R contrast = ||m_L - m_R|| / ||(m_L + m_R)/2||, close and far.
    Noise floor: the same statistics with the spout labels shuffled within session (mean of 20).
  * A figure: the six cue maps per session (one scale per session), canonical order close L / C / R, far L / C / R.

    conda activate locanmf
    python scripts/cohort2_qc/spout_compare_1029.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from swapmask_1029 import DAQ, ROOT, SESS  # noqa: E402

FS = 31.23
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029")
VAR = "hemo_meegkit_hpfit_lag_refitT_swapmask"
CODE = {0: "close_center", 1: "close_L", 2: "close_R", 3: "far_center", 4: "far_L", 5: "far_R"}
ORDER = [1, 0, 2, 4, 3, 5]
K_COMP, N_REP = 50, 20
GENO = {"PS118": "Thy1-6s", "PS120": "no GCaMP", "PS121": "RiboL1-8s", "PS122": "RiboL1-8s"}


def load(animal):
    from wfield_local import hemo_variants as hv
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    sdir, h5 = SESS[animal]
    mc = ROOT / sdir / "motion_corrected"
    res = mc / "wfield_local_results"
    s = dict(label=f"{animal}_1029", mc=str(mc), h5=str(DAQ / h5), regime="B", fmdir=None)
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, _lf, _cs = _frames(s, cue, lk)
    st = np.asarray(cue["strobe_samples"])
    pos = np.asarray(cue["strobe_codes"])[np.clip(np.searchsorted(st, cue["cue_samples"]) - 1, 0, None)]
    C = np.load(res / VAR / "SVTcorr.npy").astype(np.float64)
    n = C.shape[1]
    bad = np.load(res / "swapped_pairs.npz")["bad"][:n]
    C = hv._interp_pairs(C, bad)
    ok = (cf > 40) & (cf < n - 40)
    ok &= np.array([not bad[f - 31:f + 32].any() if o else False for f, o in zip(cf, ok)])
    cf, pos = cf[ok], pos[ok]
    feat = C[:, cf[:, None] + np.arange(0, 31)].mean(2) - C[:, cf[:, None] + np.arange(-31, 0)].mean(2)   # K x trials
    fa = np.load(res / "frames_average.npy")
    br = brain_mask(fa, res=res)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)].astype(np.float64)
    return dict(feat=feat, pos=pos, Ub=Ub, br=br, shape=fa.shape[1:])


def specificity(Ub, feat, pos):
    maps = np.array([Ub @ feat[:, pos == p].mean(1) for p in ORDER])          # 6 x pix, in ORDER
    mu = maps.mean(0)
    share = float(((maps - mu) ** 2).sum() / (maps ** 2).sum())
    lr = {}
    for nm, (iL, iR) in (("close", (0, 2)), ("far", (3, 5))):
        lr[nm] = float(np.linalg.norm(maps[iL] - maps[iR]) / np.linalg.norm((maps[iL] + maps[iR]) / 2))
    return share, lr, maps


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    D = {a: load(a) for a in ("PS118", "PS120", "PS121", "PS122")}
    n_per = min(int((d["pos"] == p).sum()) for d in D.values() for p in ORDER)
    print(f"matched trials per spout: {n_per}", flush=True)
    rng = np.random.default_rng(0)
    rows = []
    fig, ax = plt.subplots(4, 6, figsize=(19, 13))
    for i, (a, d) in enumerate(D.items()):
        X = d["feat"][:K_COMP].T
        accs = []
        for _ in range(N_REP):
            idx = np.concatenate([rng.choice(np.flatnonzero(d["pos"] == p), n_per, replace=False) for p in ORDER])
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
            pred = cross_val_predict(clf, X[idx], d["pos"][idx], cv=StratifiedKFold(5, shuffle=True, random_state=0))
            accs.append(balanced_accuracy_score(d["pos"][idx], pred))
        share, lr, maps = specificity(d["Ub"], d["feat"], d["pos"])
        null = [specificity(d["Ub"], d["feat"], rng.permutation(d["pos"])) for _ in range(20)]
        r = dict(session=a, genotype=GENO[a], n_per_spout=n_per, decode_bal_acc=float(np.mean(accs)),
                 decode_sd=float(np.std(accs)), spout_specific_share=share,
                 share_null=float(np.mean([x[0] for x in null])), lr_close=lr["close"], lr_far=lr["far"],
                 lr_close_null=float(np.mean([x[1]["close"] for x in null])),
                 lr_far_null=float(np.mean([x[1]["far"] for x in null])))
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        lim = np.percentile(np.abs(maps), 99)
        H, W = d["shape"]
        for j, p in enumerate(ORDER):
            z = np.full(H * W, np.nan)
            z[d["br"].reshape(-1)] = 100 * maps[j]
            ax[i, j].imshow(z.reshape(H, W), cmap="RdBu_r", vmin=-100 * lim, vmax=100 * lim)
            ax[i, j].set_title(f"{a} ({GENO[a]}) {CODE[p]}" + (f"\n±{100 * lim:.2f}%  decode {r['decode_bal_acc']:.2f}"
                                                              f"  spout-share {share:.2f} (null {r['share_null']:.2f})"
                                                              if j == 0 else ""), fontsize=8)
            ax[i, j].axis("off")
    fig.suptitle("10-29 cue response (0-1 s) per spout, lag + swap-masked correction; one scale per session; "
                 f"decode = 6-way balanced accuracy at {n_per} trials/spout (chance .17)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "cohort2_spout_compare.png", dpi=95)
    (OUT / "cohort2_spout_compare.json").write_text(json.dumps(rows, indent=1))
    print("wrote", OUT / "cohort2_spout_compare.png")


if __name__ == "__main__":
    main()

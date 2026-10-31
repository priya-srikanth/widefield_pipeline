"""Is the RiboL1-8s signal small because the 415 correction removes it, or already in raw 470? (2026-10-31)

Priya 10-31: "is there a chance for 8s 415 is further off the isosbestic point? if we look just at 470 is there more
signal". If 415 carried calcium (past the isosbestic point Ca-bound is brighter, so 415 rises WITH calcium), the
regression would subtract real signal and the corrected response would be much smaller than raw 470 in 8s but not
in 6s. Per session (re-run 10-29 / 10-30; cohort1 pre-stroke 6s refs), first-lick response (lick 0-0.5 s minus
pre-cue 1 s) in raw 470 dF/F, raw 415 dF/F and corrected (hemo_meegkit_hpfit), at the cortex mean and the corrected
lick hotspot (top 2%), plus the 415/470 map correlation and the share of raw 470 the correction keeps.

    conda activate locanmf
    python scripts/cohort2_qc/raw470_vs_corrected.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from indicator_kinetics_spread import FS, IND, OUT, VAR, events, sessions  # noqa: E402


def one(s):
    from wfield_local import hemo_variants as hv
    res = Path(s["mc"]) / "wfield_local_results"
    func = s["func"]
    br = brain_mask(np.load(res / "frames_average.npy"), func, res=res)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)].astype(np.float64)
    del U
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    C = np.load(res / VAR / "SVTcorr.npy").astype(np.float64)
    n = C.shape[1]
    X = {"470": np.asarray(SVT[:, func::2][:, :n], np.float64), "415": np.asarray(SVT[:, 1 - func::2][:, :n], np.float64),
         "corr": C}
    if s["cohort"] == "cohort1" and (res / "swapped_pairs.npz").exists():
        bad = np.load(res / "swapped_pairs.npz")["bad"][:n]
        X = {k: hv._interp_pairs(v, bad) for k, v in X.items()}
    cf, fl, _ = events(s, n)
    lags = np.arange(-31, 63)
    maps, waves = {}, {}
    for k, M in X.items():
        base = M[:, cf[:, None] + np.arange(-31, 0)].mean(2)
        maps[k] = 100 * (Ub @ (M[:, fl[:, None] + np.arange(0, 16)].mean(2) - base).mean(1))
        waves[k] = (M[:, fl[:, None] + lags] - base[:, :, None]).mean(1)
    hot = maps["corr"] >= np.percentile(maps["corr"], 98)
    um, uh = Ub.mean(0), Ub[hot].mean(0)
    out = dict(label=s["label"], indicator=IND[s["animal"]], n=int(len(fl)))
    for k in X:
        out[f"cortex_{k}"] = float(maps[k].mean())
        out[f"hot_{k}"] = float(maps[k][hot].mean())
        out[f"wave_hot_{k}"] = (100 * uh @ waves[k]).tolist()
        out[f"wave_cortex_{k}"] = (100 * um @ waves[k]).tolist()
    out["r_map_415_470"] = float(np.corrcoef(maps["415"], maps["470"])[0, 1])
    out["kept_hot"] = out["hot_corr"] / out["hot_470"]
    out["kept_cortex"] = out["cortex_corr"] / out["cortex_470"]
    out["ratio415_hot"] = out["hot_415"] / out["hot_470"]
    return out


def main():
    from wfield_local.parallel import fan_out
    res, fail = fan_out(sessions(), one, jobs=4, label="session")
    if fail:
        print("FAILED", fail)
    R = sorted((r for _, r in res), key=lambda r: (r["indicator"], r["label"]))
    (OUT / "raw470_vs_corrected.json").write_text(json.dumps(R, indent=1))
    print(f"{'session':11s} ind   n   hot470 hot415 hotcorr kept  415/470 | ctx470 ctx415 ctxcorr kept | r(415,470)")
    for r in R:
        print(f"{r['label']:11s} {r['indicator']:4s} {r['n']:4d}  {r['hot_470']:6.2f} {r['hot_415']:6.2f} "
              f"{r['hot_corr']:6.2f}  {r['kept_hot']:5.2f} {r['ratio415_hot']:6.2f} | {r['cortex_470']:6.2f} "
              f"{r['cortex_415']:6.2f} {r['cortex_corr']:6.2f} {r['kept_cortex']:5.2f} | {r['r_map_415_470']:.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(-31, 63) / FS
    fig, ax = plt.subplots(2, len(R), figsize=(2.4 * len(R), 5.5), sharex=True)
    for j, r in enumerate(R):
        for i, w in enumerate(("hot", "cortex")):
            for k, c in (("470", "tab:green"), ("415", "tab:purple"), ("corr", "k")):
                ax[i, j].plot(t, r[f"wave_{w}_{k}"], color=c, lw=0.9, label=k)
            ax[i, j].axvline(0, color="0.6", lw=0.4)
            ax[i, j].axhline(0, color="0.6", lw=0.4)
            ax[i, j].set_title(f"{r['label']} {r['indicator']}\n{w}", fontsize=7)
    ax[0, 0].legend(fontsize=6)
    fig.suptitle("first-lick response, dF/F %: raw 470 (green), raw 415 (purple), corrected (black); own scale per panel",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "raw470_vs_corrected.png", dpi=80)
    print("wrote", OUT / "raw470_vs_corrected.png")


if __name__ == "__main__":
    main()

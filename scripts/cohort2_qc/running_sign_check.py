"""Running-onset response in raw 470, raw 415 and corrected, pre vs post stroke (2026-10-31).

Priya 10-31 on the PS94 08-25 event video: "when the animal is running sensory limb cortex is blue instead of the
bright red we normally see w activation during running". The videos show RAW 470 (no hemo correction), so a blue
limb cortex is either blood volume outweighing GCaMP there, or the 470 / 415 slots swapped over a stretch.
A swap predicts the "470" running map looks like the 415 one (and the 415 like a GCaMP map); a haemodynamic cause
predicts 415 more negative than 470 with the corrected map still positive.

Running onsets = canonical DAQ running bouts (behavior_events npz), bouts >= 2 s with >= 3 s of no running before.
Map = mean [onset, +2 s] minus [-2 s, onset]. Hotspot = top 2% of the corrected map. Prints per session:
n onsets, hotspot raw470 / raw415 / corrected (dF/F %), and r(raw470 map, corrected map), r(raw415, corrected).

    python scripts/cohort2_qc/running_sign_check.py PS94_0811 PS94_0814 PS94_0823 PS94_0825 ...
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402

FS = 31.23
EVENTS = Path("M:/MICROSCOPE/Priya/Behavior_logs/Widefield/behavior_summary/events")
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261031/seizure")
VAR = "hemo_meegkit_hpfit"


def one(label):
    from wfield_local import config
    from wfield_local import hemo_variants as hv
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    s = next(s for s in config.load_sessions(machine="analysis_desktop") if s["label"] == label)
    res, func = Path(s["mc"]) / "wfield_local_results", int(hv.functional_channel(s))
    _, _, csmp = _frames(s, _lc(s["h5"]), _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10))
    csmp = np.asarray(csmp)
    animal, mmdd = label.split("_")
    z = np.load(EVENTS / animal / f"2026{mmdd}.npz", allow_pickle=True)
    fs = float(z["fs"])
    st, sp = z["running_starts"], z["running_stops"]
    keep = ((sp - st) >= 2 * fs) & (np.r_[np.inf, st[1:] - sp[:-1]] >= 3 * fs)
    on = np.searchsorted(csmp, st[keep])
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    C = np.load(res / VAR / "SVTcorr.npy", mmap_mode="r")
    n = min(SVT.shape[1] // 2, C.shape[1])
    on = on[(on > 70) & (on < n - 70)]
    sw = res / "swapped_pairs.npz"
    if sw.exists():                                   # drop onsets whose window holds a swapped pair
        bad = np.load(sw)["bad"]
        on = on[[not bad[max(0, o - 63):o + 63].any() for o in on]]
    br = brain_mask(np.load(res / "frames_average.npy"), func, res=res)
    U = np.nan_to_num(np.load(res / "U.npy"))
    Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)].astype(np.float64)
    post, pre = np.arange(0, 63), np.arange(-63, 0)

    def resp(X):
        X = np.asarray(X, np.float64)
        return 100 * (Ub @ (X[:, on[:, None] + post].mean((1, 2)) - X[:, on[:, None] + pre].mean((1, 2))))
    r470 = resp(SVT[:, func::2][:, :n])
    r415 = resp(SVT[:, 1 - func::2][:, :n])
    rc = resp(C[:, :n])
    hot = rc >= np.percentile(rc, 98)
    out = dict(session=label, func=func, n_onsets=int(len(on)),
               hot_470=float(r470[hot].mean()), hot_415=float(r415[hot].mean()), hot_corr=float(rc[hot].mean()),
               r_470_corr=float(np.corrcoef(r470, rc)[0, 1]), r_415_corr=float(np.corrcoef(r415, rc)[0, 1]))
    return out, (br, r470, r415, rc)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = sys.argv[1:]
    fig, ax = plt.subplots(len(labels), 3, figsize=(7.5, 2.5 * len(labels)), squeeze=False)
    rows = []
    for i, lab in enumerate(labels):
        o, (br, *maps) = one(lab)
        rows.append(o)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
        lim = np.nanpercentile(np.abs(np.concatenate(maps)), 99)
        for j, (m, nm) in enumerate(zip(maps, ("raw 470", "raw 415", "corrected"))):
            z = np.full(br.size, np.nan)
            z[br.reshape(-1)] = m
            ax[i, j].imshow(z.reshape(br.shape), cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[i, j].set_title(f"{lab} {nm} (n={o['n_onsets']}) ±{lim:.1f}%", fontsize=7)
            ax[i, j].axis("off")
    fig.suptitle("running onset [0, 2 s] minus [-2, 0 s]", fontsize=9)
    fig.tight_layout()
    tag = "_".join(labels)
    fig.savefig(OUT / f"running_sign_{tag}.png", dpi=90)
    (OUT / f"running_sign_{tag}.json").write_text(json.dumps(rows, indent=1))
    print("wrote", OUT / f"running_sign_{tag}.png")


if __name__ == "__main__":
    main()

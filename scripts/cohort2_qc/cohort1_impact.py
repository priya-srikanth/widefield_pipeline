"""How much do the swapped frames + the 16 ms 415 lag change COHORT1 results? (Priya 2026-10-30: "how much is cohort 1
data affected"). For a spread of cohort1 sessions (each animal, early / late, the worst swap sessions):

  1. build hemo_meegkit_hpfit_lag_refitT_swapmask (swap frames interpolated, 415 advanced 16 ms, T refitted) --
     a COMPARISON variant; the adopted product is untouched;
  2. compare it with the adopted hemo_meegkit_hpfit on what the analyses use:
     * cue-locked map (0-1 s minus -1-0 s, all trials) and first-lick map (0-0.4 s): spatial r and amplitude slope
       (new on adopted, across brain pixels), and the cortex-mean time course r over -1..3 s;
     * heartbeat left (corrected / raw-470 power at the cardiac peak), adopted vs new;
     * rcoeff medians.
If the maps correlate > 0.98 with slope ~1, published-style trial-averaged results will not move.

    conda activate locanmf
    python scripts/cohort2_qc/cohort1_impact.py [--labels PS93_0826 ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402
from swapmask_1029 import detect  # noqa: E402

FS = 31.23
OUT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029")
DEFAULT = ["PS92_0811", "PS92_0911", "PS93_0814", "PS93_0826", "PS94_0820", "PS94_0910", "PS95_0831", "PS95_0914"]
NEW = "hemo_meegkit_hpfit_lag_refitT_swapmask"


def events(s):
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc
    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cf, _lf, csmp = _frames(s, cue, lk)
    ls = np.sort(lk["lick_samples"])
    j = np.searchsorted(ls, cue["cue_samples"])
    has = (j < len(ls)) & ((ls[np.minimum(j, len(ls) - 1)] - cue["cue_samples"]) / cue["sample_rate_hz"] < 1.5)
    first = np.searchsorted(csmp, ls[j[has]]) if csmp is not None else np.array([], int)
    return cf[cf >= 0], first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=DEFAULT)
    a = ap.parse_args()
    from wfield_local import config
    from wfield_local import hemo_variants as hv
    ss = {s["label"]: s for s in config.load_sessions(machine="analysis_desktop")}
    rows = []
    for lab in a.labels:
        s = ss[lab]
        res = Path(s["mc"]) / "wfield_local_results"
        func = hv.functional_channel(s)
        sp = res / "swapped_pairs.npz"
        bad = np.load(sp)["bad"] if sp.is_file() else detect(res, func)["bad"]   # never write: the sweep owns it
        if not (res / NEW / "SVTcorr.npy").is_file():
            hv.write(s, "meegkit_hpfit_lag", refit_t=True, func=func, bad_pairs=bad, tag="_swapmask", verbose=False)
        fa = np.load(res / "frames_average.npy")
        br = brain_mask(fa, func, res=res)
        U = np.nan_to_num(np.load(res / "U.npy"))
        Ub = U.reshape(-1, U.shape[-1])[br.reshape(-1)]
        del U
        um = Ub.mean(0)
        cue, lick = events(s)
        r = dict(label=lab, bad_pairs=int(bad.sum()))
        traces, maps = {}, {}
        svt = np.load(res / "SVT.npy", mmap_mode="r")
        raw470 = um @ np.asarray(svt[:, func::2], np.float64)
        for v, d in (("adopted", "hemo_meegkit_hpfit"), ("new", NEW)):
            C = np.load(res / d / "SVTcorr.npy", mmap_mode="r")
            n = C.shape[1]
            Cm = hv._interp_pairs(np.asarray(C, np.float64), bad[:n])       # compare like with like: spikes out
            c = um @ Cm
            r[f"{v}_rcoeff"] = float(np.median(np.load(res / d / "rcoeffs.npy").reshape(br.shape)[br]))
            f, Pc = welch(c, fs=FS, nperseg=2048)
            _, Pr = welch(raw470[:n] - raw470[:n].mean(), fs=FS, nperseg=2048)
            hb = (f > 8) & (f < 14)
            k = np.argmax(Pr[hb])
            r[f"{v}_heart_left"] = float(Pc[hb][k] / Pr[hb][k])
            for nm, ev, lo, hi in (("cue", cue, 0.0, 1.0), ("lick", lick, 0.0, 0.4)):
                ev = ev[(ev > 40) & (ev < n - 100)]
                if len(ev) < 20:
                    continue
                base = Cm[:, ev[:, None] + np.arange(-31, 0)].mean((1, 2))
                post = Cm[:, ev[:, None] + np.arange(int(lo * FS), int(hi * FS))].mean((1, 2))
                maps[(v, nm)] = Ub @ (post - base)
                tc = c[ev[:, None] + np.arange(-31, 94)]
                traces[(v, nm)] = (tc - tc[:, :31].mean(1, keepdims=True)).mean(0)
        for nm in ("cue", "lick"):
            if ("adopted", nm) not in maps:
                continue
            x, y = maps[("adopted", nm)], maps[("new", nm)]
            r[f"{nm}_map_r"] = float(np.corrcoef(x, y)[0, 1])
            r[f"{nm}_map_slope_new_on_adopted"] = float(x @ y / (x @ x))
            r[f"{nm}_tc_r"] = float(np.corrcoef(traces[("adopted", nm)], traces[("new", nm)])[0, 1])
            r[f"{nm}_peak_adopted_pct"] = float(100 * traces[("adopted", nm)][31:62].mean())
            r[f"{nm}_peak_new_pct"] = float(100 * traces[("new", nm)][31:62].mean())
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    (OUT / "cohort1_impact.json").write_text(json.dumps(rows, indent=1))
    print("wrote", OUT / "cohort1_impact.json")


if __name__ == "__main__":
    main()

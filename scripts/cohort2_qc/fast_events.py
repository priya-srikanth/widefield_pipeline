"""Large, fast, 470-only cortex-wide events in cohort2 (2026-10-31).

Found by session_qc.py's residual-spike count: PS122 10-29 has, in its last ~10 min only, cortex-mean 470 jumps of
6-9% dF/F within ONE pair (32 ms) that decay over ~150 ms, every 10-20 s, with 415 flat. A one-pair rise in 470 only
is calcium, not blood or light. Large synchronous recurring events of this shape are what epileptiform
discharges look like in widefield GCaMP data (Steinmetz et al. 2017), so they are surveyed here before any task
analysis pools over them. They are NOT repair artefacts: swap_repair left them in (a swap changes one frame, these
rise then decay over several).

Event = a pair where cortex-mean 470 rises > RISE % from the previous pair, 415 changes < ISO_MAX %, and the rise
is followed by a decay (the next 5 pairs average below the peak). Per session: count, rate per 10 min, peak
amplitude, and the event-triggered mean map (peak pair minus the 3 pairs before), plus the mean 470 / 415 waveform.

    conda activate locanmf
    python scripts/cohort2_qc/fast_events.py 20261029 20261030
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from brainmask import brain_mask  # noqa: E402

LABCAMS = Path("M:/MICROSCOPE/Priya/Widefield/labcams")
OUT = LABCAMS / "cohort2" / "_tmp_qc_figures_20261031"
FS = 31.23
RISE, ISO_MAX = 3.0, 1.0
LAGS = np.arange(-15, 31)


def one(sess: str) -> dict:
    res = Path(sess) / "motion_corrected" / "wfield_local_results"
    fa = np.load(res / "frames_average_raw.npy")
    br = brain_mask(fa, 1, res=res)
    U = np.load(res / "U.npy")
    Ub = np.nan_to_num(U.reshape(-1, U.shape[-1])[br.reshape(-1)]).astype(np.float64)
    del U
    um = Ub.mean(0)
    SVT = np.load(res / "SVT.npy", mmap_mode="r")
    S470 = np.asarray(SVT[:, 1::2], np.float64)
    x = 100 * (um @ S470)
    y = 100 * (um @ np.asarray(SVT[:, 0::2], np.float64))
    n = len(x)
    dx, dy = np.r_[0, np.diff(x)], np.r_[0, np.diff(y)]
    cand = np.flatnonzero((dx > RISE) & (np.abs(dy) < ISO_MAX))
    ev = []
    for t in cand:
        if t < 20 or t > n - 35 or (ev and t - ev[-1] < 10):
            continue
        pk = t + int(np.argmax(x[t:t + 3]))
        if x[pk + 1:pk + 6].mean() < x[pk]:
            ev.append(pk)
    ev = np.array(ev, int)
    out = dict(session=Path(sess).name, n_events=int(len(ev)), dur_min=n / FS / 60,
               per_10min=np.bincount((ev / FS / 600).astype(int), minlength=int(np.ceil(n / FS / 600))).tolist(),
               t_min=np.round(ev / FS / 60, 2).tolist())
    if len(ev):
        base = x[ev[:, None] + np.arange(-3, 0)].mean(1)
        out["amp_median_pct"] = float(np.median(x[ev] - base))
        out["wave470"] = (x[ev[:, None] + LAGS] - base[:, None]).mean(0).tolist()
        out["wave415"] = (y[ev[:, None] + LAGS] - y[ev[:, None] + np.arange(-3, 0)].mean(1)[:, None]).mean(0).tolist()
        m = Ub @ (S470[:, ev].mean(1) - S470[:, ev[:, None] + np.arange(-3, 0)].mean((1, 2)))
        img = np.full(br.size, np.nan)
        img[br.reshape(-1)] = 100 * m
        out["map"] = img.reshape(br.shape)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dates", nargs="+")
    a = ap.parse_args()
    from wfield_local.parallel import fan_out
    from wfield_local.writeguard import assert_writable
    sessions = sorted(str(p) for d in a.dates for p in (LABCAMS / d).glob("PS*")
                      if (p / "motion_corrected" / "wfield_local_results" / "SVT.npy").exists())
    res, fail = fan_out(sessions, one, jobs=4, label="session")
    if fail:
        raise SystemExit(f"failed: {fail}")
    R = sorted((r for _, r in res), key=lambda r: r["session"])
    assert_writable(OUT)
    for r in R:
        print(f"{r['session']:24s} events {r['n_events']:4d}  amp {r.get('amp_median_pct', float('nan')):5.1f}%  "
              f"per 10 min {r['per_10min']}", flush=True)
    (OUT / "fast_events.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "map"} for r in R],
                                                     indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R2 = [r for r in R if r["n_events"] >= 5]
    if not R2:
        return
    fig, ax = plt.subplots(2, len(R2), figsize=(4 * len(R2), 7.5), squeeze=False)
    for j, r in enumerate(R2):
        lim = np.nanpercentile(np.abs(r["map"]), 99)
        ax[0, j].imshow(r["map"], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[0, j].set_title(f"{r['session']}\n{r['n_events']} events, median +{r['amp_median_pct']:.1f}%\n"
                           f"peak map (470 dF/F %, ±{lim:.1f})", fontsize=8)
        ax[0, j].axis("off")
        t = LAGS / FS * 1000
        ax[1, j].plot(t, r["wave470"], color="tab:green", label="470")
        ax[1, j].plot(t, r["wave415"], color="tab:purple", label="415")
        ax[1, j].axvline(0, color="k", lw=0.5)
        ax[1, j].set_xlabel("ms from peak")
        ax[1, j].set_ylabel("cortex dF/F %")
        ax[1, j].legend(fontsize=7)
        ax2 = ax[1, j].inset_axes([0.55, 0.55, 0.42, 0.4])
        ax2.bar(np.arange(len(r["per_10min"])) * 10, r["per_10min"], width=8, color="0.4")
        ax2.set_title("events / 10 min", fontsize=6)
        ax2.tick_params(labelsize=5)
    fig.suptitle(f"fast 470-only events (rise > {RISE}% in one pair, 415 < {ISO_MAX}%): maps and mean waveform",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "cohort2_fast_events.png", dpi=90)
    print("wrote", OUT / "cohort2_fast_events.png")


if __name__ == "__main__":
    main()

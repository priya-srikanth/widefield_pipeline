"""Is the slow band that survives meegkit_hpfit behaviour (real slow signal) or drift? cohort2 10-29, 2026-10-30.

Replaces `slow_is_bleach.py`, whose raw-.dat bleach trajectories mixed 415 and 470 (see channel_drift.py).
Corrected cortex mean (brain-masked U mean . hemo_meegkit_hpfit SVTcorr) in 2 s bins vs behaviour from the
GUI log (`Behavior_logs/Widefield/<animal>_20261029_*/trials.csv`, `events.csv`):
  position = one-hot of the current trial's spout position (6), engagement = hit indicator + lick-on count.
Both sides projected on the same cosine band (period >= P), then OLS of the neural slow band on the behaviour
slow band. The null is the same fit with the behaviour CIRCULARLY SHIFTED (>= 10 min, 200 shifts): it keeps
the behaviour's own slow structure and the band's degrees of freedom, so chance R^2 is measured, not assumed.

Clock: imaging frame 0 ~ DAQ start (dat_head_dropped_before_daq = 0); behaviour times are GUI unix times, put
on the DAQ clock by the DAQ file's timestamp (its name). A few seconds of offset is irrelevant at >= 121 s.

    conda activate locanmf
    python scripts/cohort2_qc/slowband_behaviour.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

FS, BIN = 31.23, 2.0
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/labcams/20261029")
BEH = Path("M:/MICROSCOPE/Priya/Behavior_logs/Widefield")
SESS = [("PS118 Thy1 (bright)", "PS118_20261029_112630"), ("PS120 Thy1 (dim)", "PS120_20261029_133200"),
        ("PS121 RiboL1", "PS121_20261029_160848"), ("PS122 RiboL1", "PS122_20261029_192217")]
POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]


def cos_basis(n, dt, p):
    t = np.arange(n) * dt
    k = int(np.floor(2 * t[-1] / p))
    return np.linalg.qr(np.column_stack([np.cos(np.pi * j * t / t[-1]) for j in range(k + 1)]))[0]


def neural(s):
    res = ROOT / s / "motion_corrected" / "wfield_local_results"
    fa = np.load(res / "frames_average.npy")
    brain = binary_erosion(fa[1] > 0.45 * fa[1].max(), iterations=4)
    U = np.load(res / "U.npy")
    um = np.nan_to_num(U.reshape(-1, U.shape[-1])[brain.reshape(-1)]).mean(0)
    y = um @ np.load(res / "hemo_meegkit_hpfit" / "SVTcorr.npy").astype(np.float64)
    nb = int(BIN * FS)
    return y[:len(y) // nb * nb].reshape(-1, nb).mean(1)


def behaviour(s, nbins):
    animal = s.split("_")[0]
    lab = f"{animal}_1029"
    daq = Path(json.loads((ROOT / "photobleach" / f"photobleach_{lab}.json").read_text())["daq"]).stem
    t0 = datetime.strptime(daq.split("_", 1)[1], "%Y%m%d_%H%M%S").timestamp()
    d = next(BEH.glob(f"{animal}_20261029_*"))
    tr = pd.read_csv(d / "trials.csv")
    ev = pd.read_csv(d / "events.csv", usecols=["gui_timestamp_unix", "event_name"])
    ts = np.array([datetime.fromisoformat(x).timestamp() for x in tr.trial_start_gui_iso]) - t0
    te = np.array([datetime.fromisoformat(x).timestamp() for x in tr.trial_end_gui_iso]) - t0
    X = np.zeros((nbins, len(POS) + 2))
    edges = np.arange(nbins + 1) * BIN
    for a, b, p, h in zip(ts, te, tr.pos_name, tr.hit):
        i0, i1 = int(max(a, 0) // BIN), int(min(b, edges[-1]) // BIN) + 1
        if i1 <= 0 or i0 >= nbins:
            continue
        X[i0:i1, POS.index(p)] = 1
        X[i0:i1, len(POS)] = h
    lk = ev.gui_timestamp_unix[ev.event_name == "lick_on"].to_numpy() - t0
    X[:, len(POS) + 1] = np.histogram(lk, edges)[0]
    return X, dict(trials=len(tr), hit_rate=float(tr.hit.mean()), t_first=float(ts[0]), t_last=float(te[-1]))


def r2(y, X):
    A = np.column_stack([X, np.ones(len(y))])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    rng = np.random.default_rng(0)
    rows = []
    for name, s in SESS:
        y = neural(s)
        X, info = behaviour(s, len(y))
        y = y - y.mean()
        for P in (600, 300, 121):
            Q = cos_basis(len(y), BIN, P)
            ys = Q @ (Q.T @ y)
            out = dict(session=name, band=f">{P}s", band_share=float(ys.var() / y.var()), n_cos=Q.shape[1])
            for tag, cols in (("position", slice(0, 6)), ("engagement", slice(6, 8)), ("all", slice(0, 8))):
                Xs = Q @ (Q.T @ X[:, cols])
                obs = r2(ys, Xs)
                null = []
                for _ in range(200):
                    sh = rng.integers(int(600 / BIN), len(y) - int(600 / BIN))
                    Xn = np.roll(X[:, cols], sh, axis=0)
                    null.append(r2(ys, Q @ (Q.T @ Xn)))
                null = np.array(null)
                out[f"{tag}_R2"] = float(obs)
                out[f"{tag}_null95"] = float(np.percentile(null, 95))
                out[f"{tag}_p"] = float((1 + (null >= obs).sum()) / (1 + len(null)))
            rows.append(out)
        print(name, info, flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    o = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort2/_tmp_qc_figures_20261029/cohort2_slowband_behaviour.csv")
    df.to_csv(o, index=False)
    print("wrote", o)


if __name__ == "__main__":
    main()

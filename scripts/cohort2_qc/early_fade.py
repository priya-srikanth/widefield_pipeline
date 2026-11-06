"""Early-fade / slow-bleach residual on cohort2 adopted SVTcorr (hemo_meegkit_hpfit), native SVD space.

For each session with results: reconstruct the cortex-mean corrected trace (brain-masked U-mean . SVTcorr),
then (1) settling time of the early transient (2 s bins, 4 MAD over session, sustain 30 s) and
(2) slow-band share (>=300 s / >=600 s cosine projection) and the early-window excursion in SD.
Tells us whether the early fade is FAST (-> trim) or a SLOW ramp (-> maybe meegkit_2band / 600s FWL)."""
import numpy as np, json
from pathlib import Path
from scipy.ndimage import binary_erosion

FS = 31.23
SESS = [("PS118 Thy1",   "PS118_20261029_112630"),
        ("PS120 Thy1dim","PS120_20261029_133200"),
        ("PS121 RiboL1", "PS121_20261029_160848"),
        ("PS122 RiboL1", "PS122_20261029_192217")]
root = Path("E:/labcams_data/20261029")

def slow_frac(y, fs, min_period_s):
    t = np.arange(len(y)) / fs; T = max(t[-1], 1e-9)
    n = int(np.floor(2 * T / min_period_s))
    Q = np.linalg.qr(np.column_stack([np.cos(np.pi * k * t / T) for k in range(n + 1)]))[0]
    s = Q @ (Q.T @ y)
    return s, float(s.var() / y.var())

def settling_s(y, fs, bin_s=2.0, k=4.0, sustain_s=30.0):
    nb = int(bin_s * fs); nbin = len(y) // nb
    yb = y[:nbin * nb].reshape(nbin, nb).mean(1)
    med = np.median(yb); mad = np.median(np.abs(yb - med)) * 1.4826 + 1e-12
    dev = np.abs(yb - med) / mad
    over = dev > k
    # last early bin (within first 5 min) that is over threshold, requiring it then stays quiet sustain_s
    win = int(round(sustain_s / bin_s)); lim = int(round(300 / bin_s))
    last = 0
    for i in range(min(lim, nbin - win)):
        if over[i]:
            last = i + 1
    return last * bin_s, float(dev[:int(round(5/bin_s))].max())   # settle time, peak dev in first 5 s-bins

print(f"{'session':14s} {'min':>5s} {'first30_SD':>10s} {'first60_SD':>10s} {'settle_s':>8s} "
      f"{'peakMAD':>7s} {'slow300%':>8s} {'slow600%':>8s}")
out = {}
for name, s in SESS:
    res = root / s / "motion_corrected" / "wfield_local_results"
    f = res / "hemo_meegkit_hpfit" / "SVTcorr.npy"
    if not f.exists():
        print(f"{name:14s}  (not ready)"); continue
    C = np.load(f).astype(np.float64)                 # (K, T)
    U = np.load(res / "U.npy")                         # (H, W, K)
    fa = np.load(res / "frames_average.npy")
    favg = fa[0] if fa.ndim == 3 else fa
    brain = binary_erosion(favg > 0.45 * favg.max(), iterations=4)
    Uf = U.reshape(-1, U.shape[-1]); bm = brain.reshape(-1)
    umean = np.nan_to_num(Uf[bm]).mean(0)              # (K,)
    y = umean @ C                                      # cortex-mean corrected trace (T,)
    y = y - y.mean(); sd = y.std() + 1e-12
    n30, n60 = int(30 * FS), int(60 * FS)
    first30 = y[:n30].mean() / sd; first60 = y[:n60].mean() / sd
    _, sf300 = slow_frac(y, FS, 300.0); _, sf600 = slow_frac(y, FS, 600.0)
    st, peak = settling_s(y / sd, FS)
    out[s] = dict(minutes=len(y)/FS/60, first30_sd=first30, first60_sd=first60,
                  settle_s=st, peak_mad=peak, slow300=sf300, slow600=sf600)
    print(f"{name:14s} {len(y)/FS/60:5.1f} {first30:10.2f} {first60:10.2f} {st:8.0f} "
          f"{peak:7.1f} {100*sf300:8.1f} {100*sf600:8.1f}")
Path("E:/labcams_data/20261029/_earlyfade.json").write_text(json.dumps(out, indent=2))

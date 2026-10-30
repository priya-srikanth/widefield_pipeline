"""Where does the surviving slow variance live? Cumulative variance of the corrected cortex-mean trace
removed by a cosine high-pass at a range of cutoff PERIODS, cohort2 adopted SVTcorr.

Nested cosine projections (period>=p) are orthogonal increments, so band variance = var(slow at p_lo)
- var(slow at p_hi). Shows how fast a cutoff must be to catch the bleach drift, and how close that is
to the 57-121 s position-block band we must NOT smear."""
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_erosion

FS=31.23
SESS=[("PS118 Thy1(bright)","PS118_20261029_112630"),
      ("PS120 Thy1(dim)",   "PS120_20261029_133200"),
      ("PS121 RiboL1",      "PS121_20261029_160848"),
      ("PS122 RiboL1",      "PS122_20261029_192217")]
root=Path("E:/labcams_data/20261029")
CUTS=[1800,1200,900,600,450,300,180,121,90,60]   # 121 s = slowest position block

def slowvar(y,fs,p):
    t=np.arange(len(y))/fs; T=max(t[-1],1e-9); n=int(np.floor(2*T/p))
    if n<1: return 0.0
    Q=np.linalg.qr(np.column_stack([np.cos(np.pi*k*t/T) for k in range(n+1)]))[0]
    return float((Q@(Q.T@y)).var())

hdr="session         " + "  ".join(f">{c}s" for c in CUTS)
print(hdr); print("-"*len(hdr))
for name,s in SESS:
    f=root/s/"motion_corrected"/"wfield_local_results"/"hemo_meegkit_hpfit"/"SVTcorr.npy"
    if not f.exists(): print(f"{name:15s} (not ready)"); continue
    res=f.parent.parent
    C=np.load(f).astype(np.float64); U=np.load(res/"U.npy")
    fa=np.load(res/"frames_average.npy"); favg=fa[0] if fa.ndim==3 else fa
    g=binary_erosion(favg>0.45*favg.max(),iterations=4)
    um=np.nan_to_num(U.reshape(-1,U.shape[-1])[g.reshape(-1)]).mean(0)
    y=um@C; y=y-y.mean(); v=y.var()
    cum=[100*slowvar(y,FS,c)/v for c in CUTS]
    print(f"{name:15s} " + "  ".join(f"{c:4.0f}" for c in cum) + f"   (min {len(y)/FS/60:.0f})")
print("\n(values = % of corrected variance at periods LONGER than each cutoff; cumulative.")
print(" position blocks span 57-121 s, so a cutoff must stay >~121 s to avoid eating them.)")

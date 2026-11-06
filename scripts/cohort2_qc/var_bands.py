"""Absolute variance of the corrected cortex-mean trace by period band, cohort2 adopted SVTcorr.
Answers: where is each session's variance? Fractions hide whether a clean session has LESS variance
or the SAME fast variance without the slow drift. Report SD(total) and per-band variance in the SAME
(arbitrary SVD) units, comparable across sessions because U is orthonormal and SVT carries amplitude."""
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_erosion

FS=31.23
SESS=[("PS118 Thy1(bright)","PS118_20261029_112630"),
      ("PS120 Thy1(dim)",   "PS120_20261029_133200"),
      ("PS121 RiboL1",      "PS121_20261029_160848"),
      ("PS122 RiboL1",      "PS122_20261029_192217")]
root=Path("E:/labcams_data/20261029")
# band edges (s): >300 | 121-300 | 60-121 | 8-60 | <8 (fast: hemo residual + fast neural + noise)
EDGES=[300.0,121.0,60.0,8.0]

def slowvar(y,fs,p):
    t=np.arange(len(y))/fs; T=max(t[-1],1e-9); n=int(np.floor(2*T/p))
    if n<1: return 0.0
    Q=np.linalg.qr(np.column_stack([np.cos(np.pi*k*t/T) for k in range(n+1)]))[0]
    return float((Q@(Q.T@y)).var())

print(f"{'session':18s} {'SD_tot':>8s} | {'>300s':>8s} {'121-300':>8s} {'60-121':>8s} {'8-60':>8s} {'<8s':>8s}")
print("-"*74)
for name,s in SESS:
    f=root/s/"motion_corrected"/"wfield_local_results"/"hemo_meegkit_hpfit"/"SVTcorr.npy"
    if not f.exists(): print(f"{name:18s} (not ready)"); continue
    res=f.parent.parent
    C=np.load(f).astype(np.float64); U=np.load(res/"U.npy")
    fa=np.load(res/"frames_average.npy"); favg=fa[0] if fa.ndim==3 else fa
    g=binary_erosion(favg>0.45*favg.max(),iterations=4)
    um=np.nan_to_num(U.reshape(-1,U.shape[-1])[g.reshape(-1)]).mean(0)
    y=um@C; y=y-y.mean(); vtot=y.var()
    sv=[slowvar(y,FS,p) for p in EDGES]+[0.0]      # cumulative var above each edge, then 0
    bands=[sv[i]-sv[i+1] for i in range(len(EDGES))]+[vtot-sv[0]]  # >300,121-300,60-121,8-60 ; <8 = tot-(>8)
    # recompute <8 properly: var below 8s = vtot - var above 8s
    v8=slowvar(y,FS,8.0); bands=[sv[0], sv[1]-sv[0] if False else slowvar(y,FS,121.0)-sv[0]]
    # clearer: cumulative above edges
    c300=slowvar(y,FS,300.0); c121=slowvar(y,FS,121.0); c60=slowvar(y,FS,60.0); c8=slowvar(y,FS,8.0)
    b=[c300, c121-c300, c60-c121, c8-c60, vtot-c8]
    sc=1e3
    print(f"{name:18s} {np.sqrt(vtot):8.3f} | " + " ".join(f"{x*sc:8.1f}" for x in b) + f"   (x1e-3 var; min {len(y)/FS/60:.0f})")
print("\nSD_tot = total SD of the corrected cortex-mean (arbitrary SVD units, comparable across sessions).")
print("band columns = VARIANCE (x1e-3) in each period band; they sum to SD_tot^2.")

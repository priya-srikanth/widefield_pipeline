"""Per-pixel corrected-signal amplitude (fair across spatial scales) vs baseline F, cohort2.

Cortex-MEAN cancels fine GCaMP8s structure, so it lowballs PS121 and could mislead on PS120. Here:
per-pixel temporal SD of the corrected signal = diag(U @ cov(SVTcorr) @ U^T), via the (K,K) Gram so
the (npix,T) movie is never formed. Median over brain = real per-pixel dF/F amplitude. Compared to
F0 (frames_average) this separates 'genuinely poor dF/F' (low per-pixel SD) from 'averaging artifact'
(high per-pixel SD, low cortex-mean)."""
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_erosion

SESS=[("PS118 Thy1(bright)","PS118_20261029_112630"),
      ("PS120 Thy1(dim)",   "PS120_20261029_133200"),
      ("PS121 RiboL1",      "PS121_20261029_160848"),
      ("PS122 RiboL1",      "PS122_20261029_192217")]
root=Path("E:/labcams_data/20261029")

print(f"{'session':18s} {'F0_brain':>9s} {'perpix_SD':>10s} {'cortexmean_SD':>13s} {'ratio pp/cm':>11s} {'SD/F0(1e-3)':>11s}")
print("-"*80)
base=None
for name,s in SESS:
    f=root/s/"motion_corrected"/"wfield_local_results"/"hemo_meegkit_hpfit"/"SVTcorr.npy"
    if not f.exists(): print(f"{name:18s} (not ready)"); continue
    res=f.parent.parent
    C=np.load(f).astype(np.float64)                   # (K,T)
    U=np.load(res/"U.npy"); H,W,K=U.shape
    fa=np.load(res/"frames_average.npy"); favg=fa[0] if fa.ndim==3 else fa
    g=binary_erosion(favg>0.45*favg.max(),iterations=4)
    Uf=np.nan_to_num(U.reshape(-1,K)); bm=g.reshape(-1)
    Cov=(C@C.T)/C.shape[1]                             # (K,K) temporal covariance of components
    # per-pixel variance = sum_kl Uf[:,k] Cov[k,l] Uf[:,l] = ((Uf@Cov)*Uf).sum(1)
    pv=np.einsum('ik,ik->i',Uf@Cov,Uf)
    pv=pv[bm]; pv=pv[np.isfinite(pv)&(pv>0)]
    perpix_sd=float(np.sqrt(np.median(pv)))
    um=Uf[bm].mean(0); y=um@C; cm_sd=float(y.std())
    f0=float(favg[g].mean())
    r=perpix_sd/cm_sd
    print(f"{name:18s} {f0:9.1f} {perpix_sd:10.4f} {cm_sd:13.4f} {r:11.1f} {1e3*perpix_sd/f0 if f0>0 else 0:11.2f}")

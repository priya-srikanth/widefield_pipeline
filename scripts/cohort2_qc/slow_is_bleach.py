"""Is the slow band that survives meegkit_hpfit the raw differential bleach, or real slow signal?

Compare the SHAPE of the corrected cortex-mean slow(>=300s) trace to the raw 470 and 415 brain-mean
bleach trajectories (sampled from the .dat), as a function of session fraction. High |corr| with the
raw bleach (esp. the 470/415 ratio) => residual bleaching => removing it (meegkit_2band / FWL) is safe.
Low corr => it is slow neural/behavioural structure => removing it would cost signal (cohort1 lesson)."""
import numpy as np, re
from pathlib import Path
from scipy.ndimage import binary_erosion

FS=31.23
SESS=[("PS120 Thy1dim","PS120_20261029_133200"),("PS121 RiboL1","PS121_20261029_160848"),
      ("PS118 Thy1","PS118_20261029_112630")]
root=Path("E:/labcams_data/20261029")

def slow(y,fs,p):
    t=np.arange(len(y))/fs; T=max(t[-1],1e-9); n=int(np.floor(2*T/p))
    Q=np.linalg.qr(np.column_stack([np.cos(np.pi*k*t/T) for k in range(n+1)]))[0]
    return Q@(Q.T@y)

def rs(x,m=100):   # resample to m points over its length
    return np.interp(np.linspace(0,1,m),np.linspace(0,1,len(x)),x)

print(f"{'session':14s} {'corr(slow,470raw)':>17s} {'corr(slow,415raw)':>17s} {'corr(slow,470/415)':>18s}")
for name,s in SESS:
    res=root/s/"motion_corrected"/"wfield_local_results"
    C=np.load(res/"hemo_meegkit_hpfit"/"SVTcorr.npy").astype(np.float64)
    U=np.load(res/"U.npy"); fa=np.load(res/"frames_average.npy"); favg=fa[0] if fa.ndim==3 else fa
    brain=binary_erosion(favg>0.45*favg.max(),iterations=4)
    umean=np.nan_to_num(U.reshape(-1,U.shape[-1])[brain.reshape(-1)]).mean(0)
    y=umean@C; y=y-y.mean()
    sc=rs(slow(y,FS,300.0))
    # raw bleach trajectories from the .dat (brain-mean, func=brighter / iso=415)
    dat=next((root/s/"raw_widefield_data").glob("*_uint16.dat"))
    m=re.search(r"_(\d+)_(\d+)_(\d+)_uint16",dat.name); H,W=int(m[2]),int(m[3])
    nphys=dat.stat().st_size//(H*W*2); npair=nphys//2
    mm=np.memmap(dat,dtype=np.uint16,mode="r",shape=(nphys,H,W))
    a=np.asarray(mm[0:40:2],float).mean(); b=np.asarray(mm[1:40:2],float).mean(); fo,io=(1,0) if b>=a else (0,1)
    br=binary_erosion(favg>0.45*favg.max(),iterations=4)  # same mask size as camera frame
    # mask may differ in shape from U(SVD grid) if downsampled; rebuild from raw mean
    f0=np.mean([mm[2*k+fo] for k in np.linspace(0,npair-1,60).astype(int)],0)
    bmask=binary_erosion(f0>0.45*f0.max(),iterations=4)
    ts=np.linspace(0,npair-1,100).astype(int)
    r470=np.array([mm[2*k+fo][bmask].mean() for k in ts])
    r415=np.array([mm[2*k+io][bmask].mean() for k in ts])
    ratio=r470/r415
    def cc(u,v): u=u-u.mean(); v=v-v.mean(); return float(u@v/((np.linalg.norm(u)*np.linalg.norm(v))+1e-12))
    print(f"{name:14s} {cc(sc,r470):17.2f} {cc(sc,r415):17.2f} {cc(sc,ratio):18.2f}")

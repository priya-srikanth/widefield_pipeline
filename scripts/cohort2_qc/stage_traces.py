"""Per-session brain-mean traces at each processing stage, cohort2 2026-10-29.

Columns = sessions. Rows:
  (1) RAW brain-mean counts: 470 (func) and 415 (iso)  -> absolute levels + bleaching
  (2) RAW normalised to % of start: 470 and 415         -> DIFFERENTIAL bleaching between channels
  (3) CORRECTED cortex-mean (SVTcorr, hemo_meegkit_hpfit), SD units, 2 s bins,
      with the >=300 s slow component overlaid           -> the "surviving slow drift"
"""
import numpy as np, re
from pathlib import Path
from scipy.ndimage import binary_erosion
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FS=31.23
SESS=[("PS118 Thy1 (bright)","PS118_20261029_112630"),
      ("PS120 Thy1 (dim)",    "PS120_20261029_133200"),
      ("PS121 RiboL1",        "PS121_20261029_160848"),
      ("PS122 RiboL1",        "PS122_20261029_192217")]
root=Path("E:/labcams_data/20261029")

def slow(y,fs,p):
    t=np.arange(len(y))/fs; T=max(t[-1],1e-9); n=int(np.floor(2*T/p))
    Q=np.linalg.qr(np.column_stack([np.cos(np.pi*k*t/T) for k in range(n+1)]))[0]
    return Q@(Q.T@y)

ready=[(n,s) for n,s in SESS if (root/s/"motion_corrected"/"wfield_local_results"/"hemo_meegkit_hpfit"/"SVTcorr.npy").exists()]
ncol=len(ready)
fig,axs=plt.subplots(3,ncol,figsize=(4.4*ncol,9.2),squeeze=False)
for j,(name,s) in enumerate(ready):
    res=root/s/"motion_corrected"/"wfield_local_results"
    # ---- raw channels ----
    dat=next((root/s/"raw_widefield_data").glob("*_uint16.dat"))
    m=re.search(r"_(\d+)_(\d+)_(\d+)_uint16",dat.name); H,W=int(m[2]),int(m[3])
    nphys=dat.stat().st_size//(H*W*2); npair=nphys//2
    mm=np.memmap(dat,dtype=np.uint16,mode="r",shape=(nphys,H,W))
    a=np.asarray(mm[0:40:2],float).mean(); b=np.asarray(mm[1:40:2],float).mean(); fo,io=(1,0) if b>=a else (0,1)
    ks=np.linspace(0,npair-1,400).astype(int)
    f0=np.mean([mm[2*k+fo] for k in ks[::8]],0); bmask=binary_erosion(f0>0.45*f0.max(),iterations=4)
    r470=np.array([mm[2*k+fo][bmask].mean() for k in ks]); r415=np.array([mm[2*k+io][bmask].mean() for k in ks])
    tmin=ks/npair*(npair/FS/60)
    ax=axs[0,j]; ax.plot(tmin,r470,color="tab:green",lw=1.3,label="470 (func)"); ax.plot(tmin,r415,color="tab:purple",lw=1.3,label="415 (iso)")
    ax.set_title(name,fontsize=11); ax.set_ylabel("raw counts" if j==0 else "");
    if j==0: ax.legend(fontsize=8,loc="best")
    ax=axs[1,j]; ax.plot(tmin,r470/r470[0]*100,color="tab:green",lw=1.3); ax.plot(tmin,r415/r415[0]*100,color="tab:purple",lw=1.3)
    ax.axhline(100,color="0.7",ls=":",lw=.7); ax.set_ylabel("% of start" if j==0 else "")
    ax.set_title(f"470 {r470[-1]/r470[0]*100-100:+.0f}%   415 {r415[-1]/r415[0]*100-100:+.0f}%",fontsize=9)
    # ---- corrected ----
    C=np.load(res/"hemo_meegkit_hpfit"/"SVTcorr.npy").astype(np.float64)
    U=np.load(res/"U.npy"); fa=np.load(res/"frames_average.npy"); favg=fa[0] if fa.ndim==3 else fa
    gmask=binary_erosion(favg>0.45*favg.max(),iterations=4)
    umean=np.nan_to_num(U.reshape(-1,U.shape[-1])[gmask.reshape(-1)]).mean(0)
    y=umean@C; y=y-y.mean(); sd=y.std()+1e-12; y=y/sd
    sc=slow(y,FS,300.0); nb=int(2*FS); nbin=len(y)//nb
    yb=y[:nbin*nb].reshape(nbin,nb).mean(1); tb=(np.arange(nbin)*2+1)/60
    ax=axs[2,j]; ax.plot(tb,yb,color="0.55",lw=0.6,label="corrected cortex-mean (2 s bins)")
    ax.plot(np.arange(len(y))/FS/60,sc,color="#d01c8b",lw=2.0,label=">=300 s component")
    ax.axhline(0,color="0.8",lw=.7); ax.set_ylim(-4,4); ax.set_xlabel("minutes")
    ax.set_ylabel("SD units" if j==0 else ""); ax.set_title(f"slow(>=300s) = {100*sc.var()/y.var():.0f}% of variance",fontsize=9)
    if j==0: ax.legend(fontsize=8,loc="upper right")
fig.suptitle("Cohort2 2026-10-29  -  brain-mean traces by processing stage (415 / 470 raw -> hemo-corrected)",fontsize=12.5,y=0.997)
fig.tight_layout(rect=[0,0,1,0.985])
out=f"E:/labcams_data/20261029/cohort2_stage_traces.png"; fig.savefig(out,dpi=135); print("wrote",out,"(",ncol,"sessions )")

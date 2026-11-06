"""Processing-stage cascade of the cortex-mean signal, cohort2 (the real meegkit_hpfit pipeline).

Rows (top->bottom) follow hemo_variants.compute():
  A raw camera          brain-mean 470/415 counts, % of start      (bleaching)
  B after SVD + dF/F    umean . SVT, per channel (dF/F units)       (drift survives into dF/F)
  C 0.1 Hz high-pass    the FIT band -> rcoeffs fit on these        (hemodynamic band; drift gone)
  D meegkit o10 detrend the OUTPUT drift removal (~43 min cutoff)   (slow bleach removed, blocks kept)
  E hpfit corrected     umean . SVTcorr = D470 - T.D415, +>=300s    (final; surviving slow in pink)
Caption: a 14 Hz low-pass + zero-mean is also applied (in _finish); invisible at this timescale.
Production applies C/D in COMPONENT space with a trial mask; endpoint E is the exact saved product."""
import numpy as np, re
from pathlib import Path
from scipy.ndimage import binary_erosion
from scipy.signal import butter, filtfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FS,HP,LP=31.23,0.1,14.0
SESS=[("PS118 Thy1 (bright)","PS118_20261029_112630"),
      ("PS120 Thy1 (dim)",    "PS120_20261029_133200"),
      ("PS121 RiboL1",        "PS121_20261029_160848"),
      ("PS122 RiboL1",        "PS122_20261029_192217")]
root=Path("E:/labcams_data/20261029")
GRN,PUR,PINK="tab:green","tab:purple","#d01c8b"

def hp(x): b,a=butter(2,HP/(FS/2),"high"); return filtfilt(b,a,x,padlen=50)
def lp(x): b,a=butter(2,LP/(FS/2),"low");  return filtfilt(b,a,x,padlen=50)
def meeg_o10(x):
    try:
        from meegkit.detrend import detrend
        y,_,_=detrend(np.asarray(x,float)[:,None],order=10); return np.asarray(y)[:,0]
    except Exception:
        t=np.linspace(0,1,len(x)); p=np.polyfit(t,x,10); return x-np.polyval(p,t)
def slow(y,p):
    t=np.arange(len(y))/FS; T=max(t[-1],1e-9); n=int(np.floor(2*T/p))
    Q=np.linalg.qr(np.column_stack([np.cos(np.pi*k*t/T) for k in range(n+1)]))[0]
    return Q@(Q.T@y)

ready=[(n,s) for n,s in SESS if (root/s/"motion_corrected"/"wfield_local_results"/"hemo_meegkit_hpfit"/"SVTcorr.npy").exists()]
nc=len(ready); rows=["A  raw camera (counts, % start)","B  after SVD + dF/F","C  0.1 Hz high-pass (fit band)",
                     "D  meegkit o10 detrend (output)","E  hpfit corrected (+ >=300 s)"]
fig,axs=plt.subplots(5,nc,figsize=(4.5*nc,11.2),squeeze=False)
for j,(name,s) in enumerate(ready):
    res=root/s/"motion_corrected"/"wfield_local_results"
    # ---- A: raw camera ----
    dat=next((root/s/"raw_widefield_data").glob("*_uint16.dat"))
    m=re.search(r"_(\d+)_(\d+)_(\d+)_uint16",dat.name); H,W=int(m[2]),int(m[3])
    nphys=dat.stat().st_size//(H*W*2); npair=nphys//2
    mm=np.memmap(dat,dtype=np.uint16,mode="r",shape=(nphys,H,W))
    aa=np.asarray(mm[0:40:2],float).mean(); bb=np.asarray(mm[1:40:2],float).mean(); fo,io=(1,0) if bb>=aa else (0,1)
    ks=np.linspace(0,npair-1,350).astype(int)
    f0=np.mean([mm[2*k+fo] for k in ks[::7]],0); bmask=binary_erosion(f0>0.45*f0.max(),iterations=4)
    r470=np.array([mm[2*k+fo][bmask].mean() for k in ks]); r415=np.array([mm[2*k+io][bmask].mean() for k in ks])
    tA=ks/npair*(npair/FS/60)
    # ---- B-E: SVT space ----
    svt=np.load(res/"SVT.npy").astype(np.float64); a=svt[:,fo::2]; b=svt[:,(fo+1)%2::2]
    U=np.load(res/"U.npy"); fa=np.load(res/"frames_average.npy"); favg=fa[0] if fa.ndim==3 else fa
    g=binary_erosion(favg>0.45*favg.max(),iterations=4)
    um=np.nan_to_num(U.reshape(-1,U.shape[-1])[g.reshape(-1)]).mean(0)
    y470=um@a; y415=um@b                                   # cortex-mean per channel (dF/F units)
    tB=np.arange(len(y470))/FS/60
    C=np.load(res/"hemo_meegkit_hpfit"/"SVTcorr.npy").astype(np.float64); yc=um@C; yc=yc-yc.mean()
    sc=slow(yc,300.0)
    # downsample for plotting speed (2 s)
    def ds(y,n=int(2*FS)): nb=len(y)//n; return (np.arange(nb)*2+1)/60, y[:nb*n].reshape(nb,n).mean(1)
    stages=[("A",tA,r470/r470[0]*100,r415/r415[0]*100,"% of start"),
            ("B",*ds(y470),ds(y415)[1],"dF/F (a.u.)"),
            ("C",*ds(hp(y470)),ds(hp(y415))[1],"a.u."),
            ("D",*ds(meeg_o10(y470)),ds(meeg_o10(y415))[1],"a.u.")]
    for ri,(_tag,t,g470,g415,ylab) in enumerate(stages):
        ax=axs[ri,j]; ax.plot(t,g470,color=GRN,lw=1.1,label="470 (func)"); ax.plot(t,g415,color=PUR,lw=1.1,label="415 (iso)")
        if ri==0: ax.set_title(name,fontsize=11.5); ax.axhline(100,color="0.7",ls=":",lw=.7)
        else: ax.axhline(0,color="0.8",lw=.6)
        if j==0: ax.set_ylabel(rows[ri],fontsize=8.5)
        if ri==0 and j==0: ax.legend(fontsize=7.5,loc="lower left")
    # row E corrected
    tE,ycb=ds(yc); ax=axs[4,j]; ax.plot(tE,ycb,color="0.55",lw=0.6,label="corrected"); ax.plot(tB,sc,color=PINK,lw=1.8,label=">=300 s")
    ax.axhline(0,color="0.8",lw=.6); ax.set_xlabel("minutes"); ax.set_ylim(np.percentile(ycb,0.5)*1.3,np.percentile(ycb,99.5)*1.3)
    ax.set_title(f"slow>=300s = {100*sc.var()/yc.var():.0f}%",fontsize=9)
    if j==0: ax.set_ylabel(rows[4],fontsize=8.5); ax.legend(fontsize=7.5,loc="upper right")
fig.suptitle("Cohort2 2026-10-29  -  cortex-mean through the meegkit_hpfit pipeline stages (A raw -> E corrected)",fontsize=12.5,y=0.998)
fig.tight_layout(rect=[0,0,1,0.986])
out="E:/labcams_data/20261029/cohort2_processing_stages.png"; fig.savefig(out,dpi=135); print("wrote",out,nc,"sessions")

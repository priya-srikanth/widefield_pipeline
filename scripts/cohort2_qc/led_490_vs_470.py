"""490 nm vs 470 nm functional LED for the RiboL1 mice (PS121, PS122), cohort2.

On 2026-11-04 PS121/PS122 were imaged with a 490 nm functional LED (folder tag ``_490_``), driven ABOVE
its recommended current. Two questions:
  (1) SNR: is 490 better than 470?  -> per-pixel corrected dF/F amplitude vs the single-frame shot-noise
      floor (same metric as perpixel_snr.py), 490 session vs the animal's 470 sessions.
  (2) LED droop from overdrive: does the 490 LED get dimmer over the session?  -> within-session functional
      brain-mean trajectory (reconstructed from raw SVT on N:, so no raw .dat needed), fade % and SHAPE
      (fast drop then plateau = thermal/current droop; smooth monotonic = dye bleaching). The 415 LED
      (separate, not overdriven) is the reference.

All reads are from the MICROSCOPE push (SVT/SVTcorr/U/frames_average). Run in the `wfield` env.
  python scripts/cohort2_qc/led_490_vs_470.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FS=31.23; GAIN=0.46; NROOT=Path("N:/MICROSCOPE/Priya/Widefield/labcams")

def analyse(day, sess):
    res=NROOT/day/sess/"motion_corrected"/"wfield_local_results"
    svt=np.load(res/"SVT.npy").astype(float); C=np.load(res/"hemo_meegkit_hpfit"/"SVTcorr.npy").astype(float)
    U=np.load(res/"U.npy"); fa=np.load(res/"frames_average.npy"); favg=fa[0] if fa.ndim==3 else fa
    a=svt[:,1::2]; b=svt[:,0::2]; H,W,K=U.shape; Uf=np.nan_to_num(U.reshape(-1,K))
    dark=min(favg[:5,:5].mean(),favg[:5,-5:].mean(),favg[-5:,:5].mean(),favg[-5:,-5:].mean())
    brain=binary_erosion(favg>dark+0.3*(np.percentile(favg,99.5)-dark),iterations=3); bm=brain.reshape(-1)
    # functional baseline image (counts) from SVT mean
    f0img=(Uf@a.mean(1)); F0=float(f0img[bm].mean()); fdark=float(np.nanpercentile(f0img,1))
    shot=100.0/np.sqrt(max(F0-fdark,1)*GAIN)
    # per-pixel corrected dF/F SD (median over brain)
    Cov=(C@C.T)/C.shape[1]; pv=np.einsum('ik,ik->i',Uf[bm]@Cov,Uf[bm]); pv=pv[np.isfinite(pv)&(pv>0)]
    perpix=float(np.sqrt(np.median(pv)))*100  # %
    snr=perpix/shot
    # within-session functional + 415 brain-mean trajectory (raw SVT units)
    um=Uf[bm].mean(0); f=um@a; iso=um@b
    def fade(x): return (np.median(x[-len(x)//10:])/np.median(x[:len(x)//10])-1)*100
    led="490" if "_490_" in sess else "470"
    return dict(day=day,led=led,mins=C.shape[1]/FS/60,F0=F0,shot=shot,perpix=perpix,snr=snr,
                ffade=fade(f),isofade=fade(iso),ftrace=f/np.median(f[:len(f)//10]),t=np.arange(len(f))/FS/60)

def main():
    sess={}
    for day in sorted(p.name for p in NROOT.glob("2026*") if p.is_dir()):
        for sp in sorted((NROOT/day).glob("PS12[12]*")):
            if (sp/"motion_corrected"/"wfield_local_results"/"hemo_meegkit_hpfit"/"SVTcorr.npy").exists():
                sess.setdefault(sp.name.split("_")[0],[]).append((day,sp.name))
    print(f"{'animal':7s}{'date':6s}{'LED':>5s}{'min':>5s}{'F0':>7s}{'shot%':>7s}{'perpix%':>8s}{'SNR':>6s}{'func_fade%':>11s}{'415_fade%':>10s}")
    data={}
    for an in ("PS121","PS122"):
        data[an]=[]
        for day,s in sess.get(an,[]):
            try:
                r=analyse(day,s); data[an].append(r)
                print(f"{an:7s}{day[4:]:6s}{r['led']:>5s}{r['mins']:5.0f}{r['F0']:7.0f}{r['shot']:7.2f}{r['perpix']:8.2f}{r['snr']:6.1f}{r['ffade']:11.0f}{r['isofade']:10.0f}")
            except Exception as e: print(f"  !! {an} {day}/{s}: {type(e).__name__} {e}")
    # figure: per animal, functional trajectory 490 vs 470 (normalized), + SNR bar
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for i,an in enumerate(("PS121","PS122")):
        ax=axs[i,0]
        for r in data[an]:
            c="tab:red" if r['led']=="490" else "0.5"; lw=1.8 if r['led']=="490" else 0.9
            ax.plot(r['t'],r['ftrace'],color=c,lw=lw,label=f"{r['day'][4:]} {r['led']} (fade {r['ffade']:.0f}%)")
        ax.axhline(1,color="0.8",lw=.6); ax.set_title(f"{an} functional brightness over session (490=red)",fontsize=10)
        ax.set_xlabel("min"); ax.set_ylabel("rel. start"); ax.legend(fontsize=7)
        axb=axs[i,1]; labs=[f"{r['day'][4:]}\n{r['led']}" for r in data[an]]; snrs=[r['snr'] for r in data[an]]
        cols=["tab:red" if r['led']=="490" else "0.5" for r in data[an]]
        axb.bar(range(len(snrs)),snrs,color=cols); axb.set_xticks(range(len(snrs))); axb.set_xticklabels(labs,fontsize=7)
        axb.set_ylabel("SNR (perpix dF/F / shot floor)"); axb.set_title(f"{an} per-pixel SNR by session",fontsize=10)
    fig.suptitle("490 nm (overdriven) vs 470 nm functional LED — SNR and within-session droop (PS121/PS122)",fontsize=12)
    fig.savefig("E:/labcams_data/cohort2_490_vs_470.png",dpi=130,bbox_inches="tight")
    print("\nwrote E:/labcams_data/cohort2_490_vs_470.png")

if __name__=="__main__": main()

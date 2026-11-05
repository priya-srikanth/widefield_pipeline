"""Per-discharge trace+map panel for a RiboL1 session, matching the 10-31 seizure-folder format:
  top:        full-session corrected cortex-mean (robust-SD), events marked
  mid-left:   zoom on the largest event -- functional (490/470, green) + 415 (purple), high-passed
  mid-right:  event-triggered average (+-1 s, aligned to peak), functional vs 415
  bottom:     dF/F map montage across the largest event (propagation), -t..+t around peak
Functional = slot 1 (490 or 470); 415 = slot 0. Reads N: SVTcorr/SVT/U/frames_average.

  python scripts/cohort2_qc/build_event_traces.py <day> <session_folder> <out.png>
"""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FS=31.23; NROOT=Path("N:/MICROSCOPE/Priya/Widefield/labcams")
def rz(x): x=np.asarray(x,float); x=x-np.median(x); return x/(1.4826*np.median(np.abs(x))+1e-12)

def main():
    day,sess,out = sys.argv[1], sys.argv[2], sys.argv[3]
    res=NROOT/day/sess/"motion_corrected"/"wfield_local_results"
    C=np.load(res/"hemo_meegkit_hpfit"/"SVTcorr.npy").astype(float); svt=np.load(res/"SVT.npy").astype(float)
    U=np.load(res/"U.npy"); fa=np.load(res/"frames_average.npy"); favg=fa[1] if fa.ndim==3 else fa
    a=svt[:,1::2]; b=svt[:,0::2]; H,W,K=U.shape; Uf=np.nan_to_num(U.reshape(-1,K))
    dark=min(favg[:5,:5].mean(),favg[:5,-5:].mean(),favg[-5:,:5].mean(),favg[-5:,-5:].mean())
    brain=binary_erosion(favg>dark+0.3*(np.percentile(favg,99.5)-dark),iterations=3); bm=brain.reshape(-1)
    um=Uf[bm].mean(0); yc=rz(um@C)
    bh,ah=butter(2,0.1/(FS/2),"high"); d470=rz(filtfilt(bh,ah,um@a,padlen=50)); d415=rz(filtfilt(bh,ah,um@b,padlen=50))
    pk,_=find_peaks(yc,height=6.0,distance=int(0.5*FS)); big=pk[yc[pk]>=8]
    led="490" if "_490_" in sess else "470"
    fig=plt.figure(figsize=(13,9)); gs=fig.add_gridspec(3,7,height_ratios=[1,1.1,0.9],hspace=0.45,wspace=0.3)
    ax0=fig.add_subplot(gs[0,:]); t=np.arange(len(yc))/FS/60
    ax0.plot(t,yc,lw=0.3,color="0.5"); ax0.plot(pk/FS/60,yc[pk],"v",color="tab:orange",ms=3); ax0.plot(big/FS/60,yc[big],"v",color="red",ms=4)
    ax0.axhline(10,color="red",lw=.5,ls="--"); ax0.set_ylabel("corrected SD"); ax0.set_xlabel("session time (min)")
    ax0.set_title(f"{sess.split('_')[0]} {day[4:]} ({led} nm): {len(pk)} events>6SD, {len(big)}>8SD, max {yc.max():.0f}SD",fontsize=10)
    tp = big[np.argmax(yc[big])] if len(big) else (pk[np.argmax(yc[pk])] if len(pk) else len(yc)//2)
    # mid-left single event zoom
    W1=int(3*FS); s0,s1=max(0,tp-W1),min(len(yc),tp+W1); tt=(np.arange(s0,s1)-tp)/FS
    axL=fig.add_subplot(gs[1,:3]); axL.plot(tt,d470[s0:s1],color="tab:green",lw=1.2,label=f"{led} (func)"); axL.plot(tt,d415[s0:s1],color="tab:purple",lw=1.2,label="415 (iso)")
    axL.axvline(0,color="0.7",lw=.5); axL.set_xlabel("s from peak"); axL.set_ylabel("robust SD"); axL.legend(fontsize=8); axL.set_title(f"largest event @ {tp/FS/60:.1f} min",fontsize=9)
    # mid-right event-triggered average
    W2=int(1.0*FS); good=[p for p in big if p-W2>=0 and p+W2<len(yc)] or [p for p in pk if p-W2>=0 and p+W2<len(yc)]
    axR=fig.add_subplot(gs[1,3:]); tt2=np.arange(-W2,W2+1)/FS*1000
    if good:
        f_st=np.array([d470[p-W2:p+W2+1] for p in good]); i_st=np.array([d415[p-W2:p+W2+1] for p in good])
        axR.plot(tt2,f_st.mean(0),color="tab:green",lw=1.6,label=f"{led}"); axR.plot(tt2,i_st.mean(0),color="tab:purple",lw=1.6,label="415")
        for p in good[:30]: axR.plot(tt2,d470[p-W2:p+W2+1],color="tab:green",lw=0.3,alpha=0.3)
    axR.axvline(0,color="0.7",lw=.5); axR.set_xlabel("ms from peak"); axR.set_title(f"event-triggered avg (n={len(good)})",fontsize=9); axR.legend(fontsize=8)
    # bottom: map montage across the event
    offs=np.array([-0.3,-0.1,0,0.1,0.2,0.4,0.7]); base=C[:,max(0,tp-30):max(1,tp-12)].mean(1)
    vmax=None
    for j,off in enumerate(offs):
        k=int(tp+off*FS); k=min(max(k,0),C.shape[1]-1); mp=(Uf@(C[:,k]-base)).reshape(H,W)
        if vmax is None: vmax=np.nanpercentile(np.abs(mp[brain]),99.7) or 1
        axm=fig.add_subplot(gs[2,j]); axm.imshow(mp,cmap="magma",vmin=0,vmax=vmax); axm.contour(brain,levels=[.5],colors="c",linewidths=.3)
        axm.set_axis_off(); axm.set_title(f"{off*1000:+.0f}ms",fontsize=8)
    fig.suptitle(f"{sess.split('_')[0]} {day[4:]} epileptiform discharge — traces + dF/F map propagation ({led} nm functional)",fontsize=12)
    fig.savefig(out,dpi=135,bbox_inches="tight"); print("wrote",out,f"(largest {yc[tp]:.0f}SD @ {tp/FS/60:.1f}min, {len(big)} events>8SD)")

if __name__=="__main__": main()

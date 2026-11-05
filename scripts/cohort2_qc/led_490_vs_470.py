"""490 nm vs 470 nm functional LED for the RiboL1 mice (PS121, PS122), cohort2.

On 2026-11-04 PS121/PS122 used a 490 nm functional LED (folder tag ``_490_``) driven ABOVE its rated
current. Two questions:
  (1) SNR: is 490 better than 470?  -> per-pixel corrected dF/F amplitude (SVTcorr+U) vs the single-frame
      shot-noise floor from the functional baseline (frames_average functional slot, counts).
  (2) LED droop from overdrive: does 490 dim over the session?  -> the pipeline's photobleach per-channel
      drift% (photobleach_<label>.json), functional vs the 415 LED (separate, not overdriven) as reference,
      plus the normalized within-session trajectory shape (fast drop then plateau = thermal/current droop).

All reads from the MICROSCOPE push. Run in the `wfield` env.
  python scripts/cohort2_qc/led_490_vs_470.py
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FS=31.23; GAIN=0.46; NROOT=Path("N:/MICROSCOPE/Priya/Widefield/labcams")

def _pb(day, animal, mmdd):
    p = NROOT/day/"photobleach"/f"photobleach_{animal}_{mmdd}.json"
    if not p.exists(): return None
    return json.loads(p.read_text())

def _traj(norm, chan):
    """Return (t,y) normalized trajectory for channel `chan` from the photobleach _norm, or (None,None)."""
    try:
        v = norm[chan]
        if isinstance(v, dict) and "t" in v: return np.array(v["t"],float), np.array(v["y"],float)
        arr = np.array(v, float)
        if arr.ndim==2 and arr.shape[1]==2: return arr[:,0], arr[:,1]
    except Exception: pass
    return None, None

def analyse(day, sess, animal, mmdd):
    res=NROOT/day/sess/"motion_corrected"/"wfield_local_results"
    C=np.load(res/"hemo_meegkit_hpfit"/"SVTcorr.npy").astype(float)
    U=np.load(res/"U.npy"); fa=np.load(res/"frames_average.npy")   # (2,H,W): slot1=func(brighter), slot0=415
    func_slot = 1 if fa[1].mean()>=fa[0].mean() else 0
    ff, fi = fa[func_slot], fa[1-func_slot]
    dark=min(ff[:5,:5].mean(),ff[:5,-5:].mean(),ff[-5:,:5].mean(),ff[-5:,-5:].mean())
    brain=binary_erosion(ff>dark+0.3*(np.percentile(ff,99.5)-dark),iterations=3); bm=brain.reshape(-1)
    F0=float(ff[brain].mean()); shot=100.0/np.sqrt(max(F0-dark,1)*GAIN)
    Uf=np.nan_to_num(U.reshape(-1,U.shape[-1]))
    Cov=(C@C.T)/C.shape[1]; pv=np.einsum('ik,ik->i',Uf[bm]@Cov,Uf[bm]); pv=pv[np.isfinite(pv)&(pv>0)]
    perpix=float(np.sqrt(np.median(pv)))*100; snr=perpix/shot
    pb=_pb(day,animal,mmdd); fd=iso=np.nan; ft=fy=None
    if pb:
        chans=pb.get("channels",{}); fkey=next((k for k in chans if k!="415"),None)
        if fkey: fd=chans[fkey]["pct"]
        if "415" in chans: iso=chans["415"]["pct"]
        if fkey: ft,fy=_traj(pb.get("_norm",{}),fkey)
    led="490" if "_490_" in sess else "470"
    return dict(day=day,led=led,mins=C.shape[1]/FS/60,F0=F0,shot=shot,perpix=perpix,snr=snr,fdrift=fd,isodrift=iso,ft=ft,fy=fy)

def main():
    sess={}
    for day in sorted(p.name for p in NROOT.glob("2026*") if p.is_dir()):
        for sp in sorted((NROOT/day).glob("PS12[12]*")):
            if (sp/"motion_corrected"/"wfield_local_results"/"hemo_meegkit_hpfit"/"SVTcorr.npy").exists():
                sess.setdefault(sp.name.split("_")[0],[]).append((day,sp.name))
    print(f"{'animal':7s}{'date':6s}{'LED':>5s}{'min':>5s}{'F0':>7s}{'shot%':>7s}{'perpix%':>8s}{'SNR':>6s}{'func_drift%':>12s}{'415_drift%':>11s}")
    data={}
    for an in ("PS121","PS122"):
        data[an]=[]
        for day,s in sess.get(an,[]):
            try:
                r=analyse(day,s,an,day[4:]); data[an].append(r)
                print(f"{an:7s}{day[4:]:6s}{r['led']:>5s}{r['mins']:5.0f}{r['F0']:7.0f}{r['shot']:7.2f}{r['perpix']:8.2f}{r['snr']:6.1f}{r['fdrift']:12.1f}{r['isodrift']:11.1f}")
            except Exception as e: print(f"  !! {an} {day}/{s}: {type(e).__name__} {e}")
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for i,an in enumerate(("PS121","PS122")):
        ax=axs[i,0]
        for r in data[an]:
            if r['ft'] is not None:
                c="tab:red" if r['led']=="490" else "0.6"; lw=2.0 if r['led']=="490" else 1.0
                ax.plot(r['ft'],r['fy'],color=c,lw=lw,label=f"{r['day'][4:]} {r['led']} ({r['fdrift']:+.0f}%)")
        ax.axhline(1,color="0.85",lw=.6); ax.set_title(f"{an} functional LED trajectory (490=red)",fontsize=10)
        ax.set_xlabel("time"); ax.set_ylabel("norm (median/first)"); ax.legend(fontsize=7)
        axb=axs[i,1]; labs=[f"{r['day'][4:]}\n{r['led']}" for r in data[an]]; snrs=[r['snr'] for r in data[an]]
        cols=["tab:red" if r['led']=="490" else "0.6" for r in data[an]]
        axb.bar(range(len(snrs)),snrs,color=cols); axb.set_xticks(range(len(snrs))); axb.set_xticklabels(labs,fontsize=7)
        axb.set_ylabel("per-pixel SNR (dF/F / shot floor)"); axb.set_title(f"{an} SNR by session (490=red)",fontsize=10)
    fig.suptitle("490 nm (overdriven) vs 470 nm functional LED — SNR + within-session droop (PS121/PS122)",fontsize=12)
    fig.savefig("E:/labcams_data/cohort2_490_vs_470.png",dpi=130,bbox_inches="tight"); print("\nwrote E:/labcams_data/cohort2_490_vs_470.png")

if __name__=="__main__": main()

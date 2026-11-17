"""490 nm vs 470 nm functional LED, cohort2: PS118 (Thy1-GCaMP6s, first 490 on 2026-11-05) and
PS121/PS122 (RiboL1-jGCaMP8s, 490 from 2026-11-04). Both indicators peak near 488 nm.

The 490 LED is driven ABOVE its rated current. Three questions (per animal, 490 vs its own 470 sessions):
brightness (F0), SNR, and within-session LED fade/droop. Two core methods:
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
import os
import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FS=31.23; GAIN=0.46; NROOT=next((p for p in (Path(f"{d}:/MICROSCOPE/Priya/Widefield/labcams") for d in "NM") if p.exists()), Path("N:/MICROSCOPE/Priya/Widefield/labcams"))  # N: on the imaging box, M: on the analysis desktop

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
        # photobleach.analyze stores _norm[k] = (x.tolist(), y.tolist()) -> shape (2, N); only (N, 2) was read,
        # so the trajectory panels came out blank (deck slide 30, 2026-11-05)
        if arr.ndim==2 and arr.shape[0]==2 and arr.shape[1]!=2: return arr[0], arr[1]
        if arr.ndim==2 and arr.shape[1]==2: return arr[:,0], arr[:,1]
    except Exception: pass
    return None, None

def analyse(day, sess, animal, mmdd):
    res=NROOT/day/sess/"motion_corrected"/"wfield_local_results"
    C=np.load(res/"hemo_meegkit_hpfit"/"SVTcorr.npy").astype(float)
    U=np.load(res/"U.npy"); fa=np.load(res/"frames_average.npy")   # (2,H,W): slot1=func(brighter), slot0=415
    func_slot = 1 if fa[1].mean()>=fa[0].mean() else 0
    ff = fa[func_slot]
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

# Animals with a 490 nm session to compare against their own 470 sessions. PS118 = Thy1-GCaMP6s (first 490
# on 2026-11-05); PS121/PS122 = RiboL1-jGCaMP8s (490 from 2026-11-04). GCaMP6s and jGCaMP8s both peak near
# 488 nm, so 490 should raise brightness/dF/F for both.
ANIMALS=("PS118","PS121","PS122")
GENO={"PS118":"GCaMP6s","PS120":"GCaMP6s","PS121":"RiboL1-GCaMP8s","PS122":"RiboL1-GCaMP8s"}

def main():
    sess={}
    for day in sorted(p.name for p in NROOT.glob("2026*") if p.is_dir()):
        for sp in sorted((NROOT/day).glob("PS*")):
            an=sp.name.split("_")[0]
            if an in ANIMALS and (sp/"motion_corrected"/"wfield_local_results"/"hemo_meegkit_hpfit"/"SVTcorr.npy").exists():
                sess.setdefault(an,[]).append((day,sp.name))
    print(f"{'animal':7s}{'genotype':16s}{'date':6s}{'LED':>5s}{'min':>5s}{'F0':>7s}{'shot%':>7s}{'perpix%':>8s}{'SNR':>6s}{'func_drift%':>12s}{'415_drift%':>11s}")
    data={}
    for an in ANIMALS:
        data[an]=[]
        for day,s in sess.get(an,[]):
            try:
                r=analyse(day,s,an,day[4:]); data[an].append(r)
                print(f"{an:7s}{GENO.get(an,'?'):16s}{day[4:]:6s}{r['led']:>5s}{r['mins']:5.0f}{r['F0']:7.0f}{r['shot']:7.2f}{r['perpix']:8.2f}{r['snr']:6.1f}{r['fdrift']:12.1f}{r['isodrift']:11.1f}")
            except Exception as e: print(f"  !! {an} {day}/{s}: {type(e).__name__} {e}")
    na=len(ANIMALS); fig,axs=plt.subplots(na,3,figsize=(16,3.4*na),squeeze=False)
    for i,an in enumerate(ANIMALS):
        # col0: within-session functional LED trajectory (fade/droop); col1: SNR; col2: brightness F0
        ax=axs[i,0]
        for r in data[an]:
            if r['ft'] is not None:
                c="tab:red" if r['led']=="490" else "0.6"; lw=2.0 if r['led']=="490" else 1.0
                ax.plot(r['ft'],r['fy'],color=c,lw=lw,label=f"{r['day'][4:]} {r['led']} ({r['fdrift']:+.0f}%)")
        ax.axhline(1,color="0.85",lw=.6); ax.set_title(f"{an} {GENO.get(an,'')}: LED trajectory (490=red)",fontsize=9.5)
        ax.set_xlabel("time"); ax.set_ylabel("norm (median/first)"); ax.legend(fontsize=6.5)
        labs=[f"{r['day'][4:]}\n{r['led']}" for r in data[an]]; cols=["tab:red" if r['led']=="490" else "0.6" for r in data[an]]
        axb=axs[i,1]; axb.bar(range(len(data[an])),[r['snr'] for r in data[an]],color=cols)
        axb.set_xticks(range(len(data[an]))); axb.set_xticklabels(labs,fontsize=7); axb.set_ylabel("per-pixel SNR (dF/F / shot)"); axb.set_title(f"{an} SNR (490=red)",fontsize=9.5)
        axf=axs[i,2]; axf.bar(range(len(data[an])),[r['F0'] for r in data[an]],color=cols)
        axf.set_xticks(range(len(data[an]))); axf.set_xticklabels(labs,fontsize=7); axf.set_ylabel("brain F0 (counts)"); axf.set_title(f"{an} brightness (490=red)",fontsize=9.5)
    fig.suptitle("490 vs 470 functional LED — brightness (F0), SNR, within-session fade/droop  (PS118 GCaMP6s, PS121/PS122 RiboL1-GCaMP8s)",fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    out=os.path.join(os.environ.get("WORK","E:/labcams_data"),"cohort2_490_vs_470.png"); fig.savefig(out,dpi=130,bbox_inches="tight"); print("\nwrote",out)

if __name__=="__main__": main()

"""Epileptiform-discharge screen for the RiboL1 mice (PS121, PS122), all cohort2 sessions on MICROSCOPE.

Detects large, fast, hypersynchronous discharges on the adopted-corrected cortex-mean, then classifies
each big event by the 0.1 Hz high-passed raw 470 vs 415 at that time:
  470 >> 415  -> genuine CALCIUM (epileptiform concern)
  470 ~ 415   -> hemodynamic / vascular / motion (NOT a discharge)
Writes a per-session table (CSV) + a summary figure (one z-trace per session, events marked; event maps
for sessions with calcium discharges). Reads N: SVTcorr/SVT/U/frames_average. Run in the `wfield` env.

  python scripts/cohort2_qc/discharge_screen.py [--animals PS121 PS122] [--out <dir>]
"""
from __future__ import annotations
import argparse, csv, os
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FS = 31.23
NROOT = next((p for p in (Path(f"{d}:/MICROSCOPE/Priya/Widefield/labcams") for d in "NM") if p.exists()), Path("N:/MICROSCOPE/Priya/Widefield/labcams"))  # N: on the imaging box, M: on the analysis desktop

def rz(x):
    x = np.asarray(x, float); x = x - np.median(x)
    return x / (1.4826 * np.median(np.abs(x)) + 1e-12)

def screen_session(day, sess):
    res = NROOT/day/sess/"motion_corrected"/"wfield_local_results"
    C = np.load(res/"hemo_meegkit_hpfit"/"SVTcorr.npy").astype(float)
    svt = np.load(res/"SVT.npy").astype(float); U = np.load(res/"U.npy")
    fa = np.load(res/"frames_average.npy"); favg = fa[0] if fa.ndim==3 else fa
    a = svt[:, 1::2]; b = svt[:, 0::2]            # 470, 415 (functional slot = 1)
    dark = min(favg[:5,:5].mean(),favg[:5,-5:].mean(),favg[-5:,:5].mean(),favg[-5:,-5:].mean())
    brain = binary_erosion(favg > dark + 0.3*(np.percentile(favg,99.5)-dark), iterations=3)
    Uf = np.nan_to_num(U.reshape(-1, U.shape[-1])); um = Uf[brain.reshape(-1)].mean(0)
    bh, ah = butter(2, 0.1/(FS/2), "high")
    yc = rz(um@C); d470 = rz(filtfilt(bh,ah,um@a,padlen=50)); d415 = rz(filtfilt(bh,ah,um@b,padlen=50))
    pk,_ = find_peaks(yc, height=6.0, distance=int(0.5*FS)); big = pk[yc[pk]>=10]
    ratio = float(np.median(np.abs(d415[big])/np.maximum(np.abs(d470[big]),1e-9))) if len(big) else float('nan')
    m470 = float(np.median(d470[big])) if len(big) else float('nan')
    calcium = len(big)>0 and ratio < 0.3 and m470 > 6
    verdict = "CLEAN" if len(big)==0 else ("DISCHARGES (calcium)" if calcium else "vascular/motion (not neural)")
    return dict(day=day, sess=sess, mins=len(yc)/FS/60, n6=int(len(pk)), n10=int(len(big)),
                maxZ=float(yc.max()), m470=m470, ratio=ratio, verdict=verdict,
                yc=yc, pk=pk, big=big, C=C, Uf=Uf, brain=brain, favg=favg)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--animals", nargs="+", default=["PS121","PS122"])
    ap.add_argument("--out", default=os.environ.get("WORK", "E:/labcams_data"))
    a = ap.parse_args(argv)
    sessions = []
    for day in sorted(p.name for p in NROOT.glob("2026*") if p.is_dir()):
        for sp in sorted((NROOT/day).glob("PS*")):
            an = sp.name.split("_")[0]
            if an in a.animals and (sp/"motion_corrected"/"wfield_local_results"/"hemo_meegkit_hpfit"/"SVTcorr.npy").exists():
                sessions.append((day, sp.name, an))
    res = []
    for day, sess, an in sessions:
        try: res.append((an, screen_session(day, sess)))
        except Exception as e: print(f"  !! {day}/{sess}: {type(e).__name__} {e}")
    # table
    print(f"{'animal':7s}{'date':6s}{'min':>5s}{'ev>6SD':>8s}{'>10SD':>7s}{'maxZ':>6s}{'470@big':>9s}{'415/470':>9s}  verdict")
    rows=[]
    for an, r in res:
        print(f"{an:7s}{r['day'][4:]:6s}{r['mins']:5.0f}{r['n6']:8d}{r['n10']:7d}{r['maxZ']:6.1f}{r['m470']:9.1f}{r['ratio']:9.2f}  {r['verdict']}")
        rows.append({k:r[k] for k in ("day","sess","mins","n6","n10","maxZ","m470","ratio","verdict")})
    outdir=Path(a.out);
    with open(outdir/"cohort2_discharge_screen.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # figure: z-trace per session + event maps for calcium-discharge sessions
    n=len(res); fig=plt.figure(figsize=(16, 2.1*n+0.5)); gs=fig.add_gridspec(n,4,width_ratios=[2.4,1,1,1],hspace=0.55,wspace=0.3)
    for i,(an,r) in enumerate(res):
        ax=fig.add_subplot(gs[i,0]); t=np.arange(len(r['yc']))/FS/60
        col="tab:red" if "calcium" in r['verdict'] else ("tab:orange" if r['n10'] else "0.5")
        ax.plot(t,r['yc'],lw=0.3,color="0.55"); ax.plot(r['pk']/FS/60,r['yc'][r['pk']],"v",color=col,ms=3)
        ax.axhline(6,color="orange",lw=.5,ls="--"); ax.axhline(10,color="red",lw=.5,ls="--")
        ax.set_title(f"{an} {r['day'][4:]}: {r['n6']} ev>6SD, {r['n10']}>10SD, max {r['maxZ']:.0f}SD  [{r['verdict']}]",fontsize=8.5,loc="left")
        ax.set_ylabel("SD"); ax.set_ylim(-5,max(12,r['maxZ']+1));
        if i==n-1: ax.set_xlabel("minutes")
        order=r['big'][np.argsort(r['yc'][r['big']])[::-1]][:3] if r['n10'] else []
        H,Wd=r['favg'].shape
        for j in range(3):
            axm=fig.add_subplot(gs[i,1+j]); axm.set_axis_off()
            if j<len(order):
                tp=order[j]; base=r['C'][:,max(0,tp-30):max(1,tp-10)].mean(1); mp=(r['Uf']@(r['C'][:,tp]-base)).reshape(H,Wd)
                v=np.nanpercentile(np.abs(mp[r['brain']]),99.5) or 1
                axm.imshow(mp,cmap="magma",vmin=0,vmax=v); axm.contour(r['brain'],levels=[.5],colors="c",linewidths=.4)
                axm.set_title(f"{r['yc'][tp]:.0f}SD",fontsize=7.5)
    fig.suptitle("Cohort2 RiboL1 epileptiform-discharge screen (corrected cortex-mean; red = calcium discharges)",fontsize=12)
    fig.savefig(outdir/"cohort2_discharge_screen.png",dpi=130,bbox_inches="tight")
    print(f"\nwrote {outdir}/cohort2_discharge_screen.csv + .png")

if __name__=="__main__":
    main()

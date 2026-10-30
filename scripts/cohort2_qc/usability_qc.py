import numpy as np, re
from pathlib import Path
from scipy.ndimage import binary_erosion
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

S=[("PS118 Thy1","PS118_20261029_112630"),("PS120 Thy1 (dim?)","PS120_20261029_133200"),
   ("PS121 RiboL1","PS121_20261029_160848"),("PS122 RiboL1","PS122_20261029_192217")]
root=Path("E:/labcams_data/20261029")
D={}
for name,sess in S:
    dat=next((root/sess/"raw_widefield_data").glob("*_uint16.dat"))
    m=re.search(r"_(\d+)_(\d+)_(\d+)_uint16",dat.name); nch,H,W=int(m[1]),int(m[2]),int(m[3])
    nphys=dat.stat().st_size//(H*W*2); npair=nphys//2
    mm=np.memmap(dat,dtype=np.uint16,mode="r",shape=(nphys,H,W))
    a=np.asarray(mm[0:40:2],float).mean(); b=np.asarray(mm[1:40:2],float).mean(); fo,io=(1,0) if b>=a else (0,1)
    samp=np.linspace(0,npair-1,120).astype(int)
    F0=np.mean([mm[2*k+fo] for k in samp],0); I0=np.mean([mm[2*k+io] for k in samp],0)
    brain=binary_erosion(F0>0.45*F0.max(),iterations=4)
    # edge coverage: does brain reach within 10 px of the frame border?
    edge=np.zeros_like(brain); edge[:10]=edge[-10:]=edge[:,:10]=edge[:,-10:]=True
    brain_on_edge=int((brain&edge).sum())
    # vignetting: median brightness of a central disk vs the 4 corners (whole frame)
    yy,xx=np.indices((H,W)); cy,cx=H/2,W/2; cen=((yy-cy)**2+(xx-cx)**2)<(min(H,W)*0.18)**2
    corners=((xx<W*0.12)|(xx>W*0.88))&((yy<H*0.12)|(yy>H*0.88))
    vig=np.median(F0[corners])/np.median(F0[cen])
    sat=100*np.mean(F0>=64000)
    # within-session trajectory (brain-ROI mean)
    tsamp=np.linspace(0,npair-1,40).astype(int)
    ft=np.array([mm[2*k+fo][brain].mean() for k in tsamp]); it=np.array([mm[2*k+io][brain].mean() for k in tsamp])
    D[name]=dict(F0=F0,brain=brain,H=H,W=W,vig=vig,edge=brain_on_edge,brainpx=int(brain.sum()),sat=sat,
                 ft=ft/ft[0]*100,it=it/it[0]*100,func0=F0[brain].mean(),iso0=I0[brain].mean())
    print(f"{name:18s} {nch}_{H}_{W}  brain_px={int(brain.sum()):,}  brain_on_edge_px={brain_on_edge:,}  "
          f"corner/center F0={vig:.2f}  sat={sat:.2f}%  415_end/start={it[-1]/it[0]*100-100:+.0f}%  func_end/start={ft[-1]/ft[0]*100-100:+.0f}%")

fig=plt.figure(figsize=(16,7)); gs=fig.add_gridspec(2,4,height_ratios=[1.25,1],hspace=0.28,wspace=0.22)
for j,(name,_) in enumerate(S):
    d=D[name]; ax=fig.add_subplot(gs[0,j]); im=ax.imshow(d["F0"],cmap="gray")
    ax.contour(d["brain"],levels=[0.5],colors="lime",linewidths=1.0); ax.set_axis_off()
    ax.set_title(f"{name}\n{d['H']}x{d['W']}  brain={d['brainpx']//1000}k px\ncorner/center={d['vig']:.2f}  edge px={d['edge']}",fontsize=9)
    ax2=fig.add_subplot(gs[1,j]); t=np.linspace(0,100,len(d["ft"]))
    ax2.plot(t,d["ft"],color="tab:green",lw=1.6,label="func (470)"); ax2.plot(t,d["it"],color="tab:purple",lw=1.6,label="415")
    ax2.axhline(100,color="gray",ls=":",lw=.7); ax2.set_ylim(min(70,d["it"].min()-3),max(110,d["ft"].max()+3))
    ax2.set_xlabel("% session"); ax2.set_title(f"brain-ROI brightness (norm %)\n415 end {d['it'][-1]-100:+.0f}%",fontsize=8.5)
    if j==0: ax2.set_ylabel("% of start"); ax2.legend(fontsize=7)
fig.suptitle("Cohort2 2026-10-29 usability QC — functional mean image + brain contour (top); within-session brightness, 470 vs 415 (bottom)",fontsize=11,y=0.99)
out=r"E:/labcams_data/20261029/cohort2_usability_qc.png"; fig.savefig(out,dpi=135,bbox_inches="tight"); print("wrote",out)

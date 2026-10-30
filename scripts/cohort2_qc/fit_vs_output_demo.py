"""Why the pipeline FITS on the 0.1 Hz high-pass (C) but OUTPUTS through the meegkit detrend (D).

The difference is ACAUSALITY. Both remove slow drift. But the zero-phase high-pass (filtfilt) is acausal
-- it deposits a negative shadow BEFORE each event -- which is fine for estimating a single regression
coefficient (a slope over the whole session; phase doesn't bias it) but WRONG for the output timecourse
that analyses read (it would smear each cue response backward in time). The polynomial detrend has no
such kernel: it only subtracts a smooth trend, leaving event timing intact.

3 panels: (1) impulse response of the high-pass alone; (2) drift+transients through C vs D; (3) zoom."""
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FS,HP=31.23,0.1
def C_highpass(x): b,a=butter(2,HP/(FS/2),"high"); return filtfilt(b,a,x,padlen=50)
def D_detrend(x):  t=np.linspace(0,1,len(x)); return x-np.polyval(np.polyfit(t,x,10),t)  # meegkit o10 analog

# ---- panel 1: impulse response of the high-pass (the acausal kernel) ----
N=int(20*FS); imp=np.zeros(N); imp[N//2]=1.0
k=C_highpass(imp); tt=(np.arange(N)-N//2)/FS
pre=k[N//2-int(FS):N//2].sum()   # energy in the 1 s BEFORE the impulse

# ---- panels 2-3: realistic drift + GCaMP6s-like transients ----
T=120.0; t=np.arange(int(T*FS))/FS
drift=0.06*np.exp(-t/70)+0.01*(1-t/T)                      # bleach-like slow drift
def gcamp(t0,amp=0.05,rise=0.15,dec=1.6):
    x=np.zeros_like(t); m=t>=t0; dt=t[m]-t0
    x[m]=amp*(1-np.exp(-dt/rise))*np.exp(-dt/dec); return x
events=[18,42,66,93]; true=sum(gcamp(e) for e in events)
rng=np.cumsum(np.zeros_like(t))  # deterministic (no Date/random); use a fixed smooth wobble instead
noise=0.004*np.sin(2*np.pi*0.7*t)+0.003*np.sin(2*np.pi*1.9*t+1.0)
sig=drift+true+noise
cC=C_highpass(sig); cD=D_detrend(sig)

fig,ax=plt.subplots(3,1,figsize=(11,10))
# panel 1
ax[0].plot(tt,k,color="tab:blue",lw=1.6); ax[0].axvline(0,color="k",lw=0.8,ls="--")
ax[0].axvspan(-1,0,color="tab:red",alpha=0.12)
ax[0].set_xlim(-6,6); ax[0].set_xlabel("time relative to a unit impulse (s)"); ax[0].set_ylabel("filter output")
ax[0].set_title(f"(1) C = 0.1 Hz zero-phase high-pass: its IMPULSE RESPONSE is ACAUSAL -- there is output "
                f"BEFORE t=0\n(shaded: {pre:+.3f} deposited in the 1 s before the event -- a backward shadow). "
                f"Fine for a coefficient, not for the output.",fontsize=9.5)
# panel 2
ax[1].plot(t,sig,color="0.6",lw=0.9,label="signal = drift + transients (+wobble)")
ax[1].plot(t,drift,color="0.8",lw=1.2,ls=":",label="true slow drift")
ax[1].plot(t,cC,color="tab:blue",lw=1.0,label="C: 0.1 Hz high-pass (FIT band)")
ax[1].plot(t,cD,color="tab:orange",lw=1.0,label="D: meegkit o10 detrend (OUTPUT)")
for e in events: ax[1].axvline(e,color="k",lw=0.5,alpha=0.3)
ax[1].set_xlim(0,T); ax[1].set_xlabel("s"); ax[1].set_ylabel("a.u."); ax[1].legend(fontsize=8,ncol=2,loc="upper right")
ax[1].set_title("(2) Both remove the drift -- at full view C and D look the same.",fontsize=10)
# panel 3 zoom on event at 42 s
z0,z1=38,50; m=(t>=z0)&(t<=z1)
ax[2].plot(t[m],(true)[m],color="0.3",lw=2.0,ls=":",label="TRUE transient (drift-free)")
ax[2].plot(t[m],cC[m]-np.median(cC[m]),color="tab:blue",lw=1.6,label="C high-pass: dip BEFORE onset + undershoot")
ax[2].plot(t[m],cD[m]-np.median(cD[m]),color="tab:orange",lw=1.6,label="D detrend: shape + timing preserved")
ax[2].axvline(42,color="k",lw=0.8,ls="--"); ax[2].axhline(0,color="0.85",lw=0.7)
ax[2].axvspan(40.5,42,color="tab:red",alpha=0.12)
ax[2].set_xlim(z0,z1); ax[2].set_xlabel("s"); ax[2].set_ylabel("a.u."); ax[2].legend(fontsize=8.5,loc="upper right")
ax[2].set_title("(3) Zoom on the 42 s event: the acausal high-pass (C) pushes a negative shadow into the "
                "PRE-event window (red); the detrend (D) does not. That is why C is used only to FIT the "
                "415->470 coefficient and D is what becomes the OUTPUT.",fontsize=9.3)
fig.suptitle("Fit (C, high-pass) vs Output (D, detrend): same drift removal, but only C smears events backward in time",
             fontsize=12,y=0.997)
fig.tight_layout(rect=[0,0,1,0.985])
out="E:/labcams_data/20261029/fit_vs_output_demo.png"; fig.savefig(out,dpi=135); print("wrote",out,"pre-1s energy",pre)

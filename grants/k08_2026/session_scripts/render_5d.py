"""Fig 5D, cued: dF/F around the FIRST CUED LICK (lick-aligned, -2..+3 s), tjM1 and SSp-m, contra (right) and ipsi
(left), mean +- SEM over sessions by epoch, from iti_licks/figdata/lick_cued_ramp.pkl (sess['cued'], row 0; the same
sessions as 5E: Aug pre n=31). Units: stored dF/F x1e3 -> % (/10). Grey band: the 0..1 s AUC window of 5E/5F.
Bars on top: per-bin difference from pre (paired bootstrap animals > sessions, BH q < .05), as stored.
Drawn in the PAGE coordinates of Priya's Figure5 (pt), so it drops in at translate(0 Y0)."""
import pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'svg.fonttype':'none','font.family':['Arial','Liberation Sans'],'axes.linewidth':0.6,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.0,'ytick.major.size':2.0,
 'xtick.labelsize':6.6,'ytick.labelsize':6.6,'xtick.major.pad':1.5,'ytick.major.pad':1.5})
D='/mnt/user-data/uploads/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/figdata/'
c=pickle.load(open(D+'lick_cued_ramp.pkl','rb'))['sess']['cued']
LAGS=np.round(np.arange(-2.0,3.0001,0.05),3)
COL={'pre':'#111111','acute':'#e0218a','subacute':'#d4a017','chronic':'#1b7f79'}
W,Y0,Y1=400.0,110.0,208.0; H=Y1-Y0
fig=plt.figure(figsize=(W/72,H/72))
X0=[35.1,113.8,192.3,271.0]; AW=61.5; AT,AB=125.0,189.0
P=[(2,'tjM1 contra'),(3,'tjM1 ipsi'),(4,'SSp-m contra'),(5,'SSp-m ipsi')]
YL=(-0.4,7.4)
for i,(col,title) in enumerate(P):
    ax=fig.add_axes([X0[i]/W,(Y1-AB)/H,AW/W,(AB-AT)/H])
    v=c[(0,col)]
    ax.axvspan(0,1,color='0.9',lw=0,zorder=0)
    ax.axvline(0,color='0.6',lw=0.5,zorder=0.5)
    for ep,m,se,n in v['curves']:
        m=m/10; se=se/10
        ax.fill_between(LAGS,m-se,m+se,color=COL[ep],alpha=0.22,lw=0,zorder=1)
        ax.plot(LAGS,m,color=COL[ep],lw=1.0,zorder=2)
    for k,ep,segs in v['sig']['rows']:
        y=7.05-k*0.32
        for a,b in segs: ax.plot([max(a,-2),min(b,3)],[y,y],color=COL[ep],lw=1.4,solid_capstyle='butt',zorder=3)
    ax.set_xlim(-2,3); ax.set_ylim(*YL); ax.set_xticks([-2,-1,0,1,2,3]); ax.set_yticks([0,2,4,6])
    ax.set_xticklabels(['−2','−1','0','1','2','3'])
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.set_title(title,fontsize=8,pad=2.5)
    if i==0: ax.set_ylabel('ΔF/F (%)',fontsize=7,labelpad=2)
fig.text((X0[0]+X0[3]+AW)/2/W,(Y1-204.5)/H,'time from first cued lick (s)',ha='center',va='baseline',fontsize=7)
h=[plt.Line2D([],[],color=COL[e],lw=1.3,label=e) for e in COL]
h.append(matplotlib.patches.Patch(color='0.9',label='AUC window'))
fig.legend(handles=h,loc='upper left',bbox_to_anchor=(341/W,(Y1-123)/H),frameon=False,fontsize=7,handlelength=1.6,
           labelspacing=0.35,borderaxespad=0,borderpad=0,handletextpad=0.5)
fig.savefig('Fig5D_cued.svg'); fig.savefig('Fig5D_cued.pdf'); fig.savefig('Fig5D_cued.png',dpi=300)

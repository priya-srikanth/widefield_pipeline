import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from grant_panels import *
from grant_panels import star_y, fig_legend, set_scale

D='data/'

# ---------------------------------------------------------------------------
# All transcribed marks are GONE (9 Nov). Both per-spout behaviour panels now
# read stored statistics:
#   aim  -> iti_licks/aim_by_spout_tests.csv       stars = sig_bonf
#   hits -> prepost/stats/change_vs_pre_stats.csv  stars = q_fdr < 0.05
# Local copies of both live in aimstats/.

def offset_cued_first():
    """Cued, FIRST lick, like-for-like decoder offset."""
    d   = pd.read_csv(D+'lick_cued_first_lick_sessions.csv')
    lmm = pd.read_csv(D+'lick_cued_first_lick_lmm.csv')
    keep = [('all spouts|lfl:first','all spouts'), ('far contra|lfl:first','far contra')]
    set_scale('decoder_offset_cued_first_lick', 3.6)
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.75), sharey=True)
    for ax,(pos,title) in zip(axes, keep):
        epoch_strip(ax, d[d.position==pos])
        ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
        ax.set_title(title)
        headroom(ax, 0.20)
    axes[0].set_ylabel('like-for-like\noffset (mm)')
    lg = fig_legend(fig, ncol=4, y=-0.16)   # below the axes: cannot hit data or stars
    band = star_band(axes[0], lg)
    for ax,(pos,_) in zip(axes, keep):
        stars(ax, lmm, pos, band=band)
    fig.subplots_adjust(wspace=0.14)
    save(fig,'decoder_offset_cued_first_lick')

def hemispheric_balance_farcontra():
    """Cued lick hemispheric balance, far-contra spout only, tjM1 and SSp-m."""
    d   = pd.read_csv(D+'lick_lateral_ratio_auc_docked_sessions.csv')
    lmm = pd.read_csv(D+'lick_lateral_ratio_auc_docked_lmm.csv')
    d = d[d.panel.str.endswith(', cued') & d.position.str.endswith(':far_contra')].copy()
    d['roi'] = d.panel.str.split(',').str[0]
    rois = [r for r in ['tjM1','SSp-m'] if r in set(d.roi)]
    _W = 1.30*len(rois) + 0.55                 # skinnier sub-panels
    set_scale('hemispheric_balance_cued_farcontra', _W)
    fig, axes = plt.subplots(1, len(rois), figsize=(_W, 1.42), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, roi in zip(axes, rois):
        sub = d[d.roi==roi]
        epoch_strip(ax, sub)
        ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
        ax.set_title(roi)
        headroom(ax, 0.20)
    axes[0].set_ylabel('contra : ipsi balance\n(far contra spout)')
    lg = fig_legend(fig, ncol=4, y=-0.16)
    band = star_band(axes[0], lg)
    for ax, roi in zip(axes, rois):
        stars(ax, lmm, d[d.roi == roi].position.iloc[0], band=band)
    fig.subplots_adjust(wspace=0.18)
    save(fig,'hemispheric_balance_cued_farcontra')

def lick_auc_single():
    """Lick-response AUC 0-1.0 s as a fold of each animal's own pre, spouts
    POOLED, four ROI panels compressed to one axes.

    This is the quantity the deck's main slide plots (its `_w0-1` variant), and
    the marks are that figure's own published bootstrap stars (`boot_star`), not
    the LMM. Restricting to one spout, as an earlier version did, lost almost
    every mark because far contra is the weakest of the six cells."""
    d  = pd.read_csv(D+'cued_ramp_w01_sessions.csv')
    v  = pd.read_csv(D+'cued_ramp_w01_verify.csv')
    rois = ['tjM1_right','tjM1_left','SSp-m_right','SSp-m_left']
    d = d[d.position.isin(rois)]
    set_scale('lick_response_auc_compressed', 3.5)
    fig, ax = plt.subplots(figsize=(3.5, 1.38))
    w = 0.19
    rng = np.random.default_rng(0)
    for j, ep in enumerate(EPOCHS):
        xs, ys, es = [], [], []
        for i, roi in enumerate(rois):
            sub = d[(d.position==roi) & (d.epoch==ep)]
            if sub.empty: continue
            xc = i+(j-1.5)*w
            for an, g in sub.groupby('animal'):          # faded session dots
                ax.plot(xc + rng.uniform(-0.055, 0.055, len(g)), g.value,
                        MK.get(an,'o'), ms=2.4, mfc='none', mec=C[ep], mew=0.5,
                        alpha=0.45, zorder=1)
            xs.append(xc); ys.append(sub.value.mean())
            es.append(sub.value.std(ddof=1)/max(np.sqrt(sub.animal.nunique()),1))
        ax.errorbar(xs, ys, yerr=es, fmt='o', ms=4.0, color=C[ep], mfc=C[ep],
                    mec='k', mew=0.4, elinewidth=1.1, capsize=0, ls='none', zorder=3)
    ax.axhline(1.0, color='0.6', lw=0.6, ls='--', zorder=0)
    ax.set_xticks(range(len(rois)))
    ax.set_xticklabels([r.replace('_',' ') for r in rois])
    ax.set_ylabel('lick-response AUC\n(fold of own pre)')
    ax.set_xlim(-0.5, len(rois)-0.5)
    headroom(ax, 0.12)
    ax.set_ylim(top=min(ax.get_ylim()[1], 3.05))
    lg = fig_legend(fig, ncol=4, y=-0.02)   # below the axes: cannot hit data or stars
    band = star_band(ax, None)
    for i, roi in enumerate(rois):
        sub = v[v.position == roi]
        for j, ep in enumerate(EPOCHS):
            r = sub[sub.epoch==ep]
            if r.empty: continue
            m = str(r.iloc[0].get('boot_star','') or '').strip()
            if m and m.lower()!='nan':
                ax.text(i+(j-1.5)*w, star_y(ax, band, j, len(EPOCHS)), m,
                        ha='center', va='center', color=C[ep],
                        fontsize=10, fontweight='bold')
    save(fig,'lick_response_auc_compressed')

def frozen_refit_gap_pooled():
    """Pooled over spouts: frozen pre-stroke decoder, within-session refit, and the gap.
    Deck slide 'Pooled over spouts: the refit - frozen gap grows through recovery'."""
    d  = pd.read_csv(D+'epoch_5rmo_frozen_refit_overall_cue_working_sessions.csv')
    st = pd.read_csv(D+'epoch_5rmo_frozen_refit_overall_cue_working.csv')
    panels = [('frozen','frozen pre-stroke\ndecoder'),('refit','within-session\nrefit decoder'),
              ('gap','refit − frozen gap')]
    set_scale('frozen_refit_gap_pooled', 5.1)
    fig, axes = plt.subplots(1, 3, figsize=(5.1, 1.13))
    for ax,(pos,title) in zip(axes, panels):
        epoch_strip(ax, d[d.position==pos])
        ax.set_title(title, fontsize=7.6)
        headroom(ax, 0.10)
    axes[0].set_ylabel('decoding accuracy')
    axes[2].set_ylabel('accuracy difference')
    lg = fig_legend(fig, ncol=4, y=-0.16)   # below the axes: cannot hit data or stars
    band = star_band(axes[0], lg)
    for ax,(pos,_) in zip(axes, panels):
        sub = st[st.position==pos]
        for i, ep in enumerate(EPOCHS):
            rr = sub[sub.epoch==ep]
            if rr.empty: continue
            m = str(rr.iloc[0].get('mark','') or '').strip()
            if m and m.lower()!='nan':
                ax.text(i, star_y(ax, band, i, len(EPOCHS)), m, ha='center',
                        va='center', color=C[ep], fontsize=11, fontweight='bold')
    fig.subplots_adjust(wspace=0.42)
    save(fig,'frozen_refit_gap_pooled')

for fn in (offset_cued_first, hemispheric_balance_farcontra, lick_auc_single, frozen_refit_gap_pooled):
    print(fn.__name__); fn()

def first_lick_aim_single():
    """First-lick aim offset vs own pre, all six spouts in ONE panel.

    Statistics are the STORED paired nested bootstrap from
    `iti_licks/aim_by_spout_tests.csv` (animals -> sessions): `mean` with its
    95% CI, and stars from `sig_bonf` (Bonferroni over the 24 spout x epoch
    comparisons).  Session points are the per-session change vs that animal's
    own pre from `aim_by_spout_sessions.csv`, so the panel now follows the same
    faded-session-dots + mean/CI convention as every other per-epoch panel.
    """
    t = pd.read_csv('aimstats/aim_by_spout_tests.csv')
    s = pd.read_csv('aimstats/aim_by_spout_sessions.csv')
    order = ['close_ipsi','close_mid','close_contra','far_ipsi','far_mid','far_contra']
    eps = EPOCHS                                   # pre, acute, subacute, chronic
    set_scale('first_lick_aim_by_spout_single', 4.0)
    fig, ax = plt.subplots(figsize=(4.0, 1.52))
    w = 0.19
    ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
    rng = np.random.RandomState(7)
    for j, ep in enumerate(eps):
        for i, pos in enumerate(order):
            x0 = i + (j - (len(eps)-1)/2.0) * w
            g = s[(s.pos == pos) & (s.epoch == ep)]
            if len(g):
                jit = (rng.rand(len(g)) - 0.5) * w * 0.62
                ax.plot(x0 + jit, g.chg.values, 'o', ms=1.9, mfc=C[ep],
                        mec='none', alpha=0.30, zorder=1)
            r = t[(t.pos == pos) & (t.epoch == ep)]
            if r.empty: continue
            r = r.iloc[0]
            ax.errorbar([x0], [r['mean']],
                        yerr=[[r['mean']-r['lo']], [r['hi']-r['mean']]],
                        fmt='o', ms=4.2, color=C[ep], mfc=C[ep], mec='k',
                        mew=0.4, elinewidth=1.0, capsize=0, ls='none', zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([p.replace('close', 'near').replace('_', '\n') for p in order])
    ax.set_ylabel('first-lick aim offset\n(deg vs own pre)')
    ax.set_xlim(-0.5, len(order)-0.5)
    headroom(ax, 0.30)
    h = [plt.Line2D([],[],marker='o',ls='none',ms=4.0,color=C[e],mfc=C[e],mec='k',mew=0.4,label=e)
         for e in eps]
    lg = ax.legend(handles=h, loc='lower left', ncol=2, frameon=False,
                   handlelength=0.8, handletextpad=0.3, columnspacing=0.7,
                   labelspacing=0.25, borderaxespad=0.15)
    for tx in lg.get_texts(): tx.set_fontsize(6.6)
    band = 0.90            # legend is out of the way; stars ride the top band
    nsig = 0
    for i, pos in enumerate(order):
        for j, ep in enumerate(eps):
            r = t[(t.pos == pos) & (t.epoch == ep)]
            if r.empty: continue
            if not bool(r.iloc[0]['sig_bonf']): continue
            nsig += 1
            ax.text(i + (j - (len(eps)-1)/2.0) * w, star_y(ax, band, j, len(eps)),
                    '*', ha='center', va='center', color=C[ep],
                    fontsize=10, fontweight='bold')
    print('   aim: %d of %d cells sig_bonf' % (nsig, len(order)*len(eps)))
    save(fig,'first_lick_aim_by_spout_single')

print('first_lick_aim_single'); first_lick_aim_single()

def state_space_plane():
    """Sessions in the pre->chronic / orthogonal-departure plane, re-plotted from
    `per_position_lick_svd/per_position_sessions.csv` so the 1-SD trial-cloud
    ellipses can be dropped and the session markers faded. One point per session
    x spout; error bars are the 95% CI of the epoch mean over animals."""
    d = pd.read_csv(D+'ss_plane_sessions.csv')
    set_scale('state_space_plane_sessions', 2.6)
    fig, ax = plt.subplots(figsize=(2.6, 1.72))
    ax.axhline(0, color='0.85', lw=0.7, zorder=0)
    ax.axvline(0, color='0.85', lw=0.7, zorder=0)
    ax.axvline(1, color='0.85', lw=0.7, zorder=0)
    for ep in EPOCHS:
        sub = d[d.epoch == ep]
        if sub.empty: continue
        for an, g in sub.groupby('animal'):
            ax.plot(g.x, g.y, MK.get(an,'o'), ms=2.6, mfc='none', mec=C[ep],
                    mew=0.5, alpha=0.22, zorder=1)
        am = sub.groupby('animal')[['x','y']].mean()
        n = max(len(am), 1)
        mx, my = am.x.mean(), am.y.mean()
        ex = 1.96*am.x.std(ddof=1)/np.sqrt(n) if n > 1 else 0
        ey = 1.96*am.y.std(ddof=1)/np.sqrt(n) if n > 1 else 0
        ax.errorbar(mx, my, xerr=ex, yerr=ey, fmt=MK['PS92'], ms=6.0, color=C[ep],
                    mfc=C[ep], mec='k', mew=0.6, elinewidth=1.3, capsize=0, zorder=4)
        # displaced from pre on at least one axis (paired bootstrap, animals
        # then sessions): acute and subacute on the departure axis, subacute and
        # chronic along the recovery axis
        if ep in ('acute', 'subacute', 'chronic'):
            ax.text(mx + ex + 0.08, my + ey + 0.06, '*', ha='left', va='bottom',
                    color=C[ep], fontsize=10, fontweight='bold', zorder=5)
    ax.set_xlabel('along pre (0) → chronic (1)')
    ax.set_ylabel('acute departure, orthogonal\n(same units)')
    headroom(ax, 0.16)
    fig_legend(fig, ncol=4, y=-0.02)
    save(fig, 'state_space_plane_sessions')

print('state_space_plane'); state_space_plane()

def hit_rate_single():
    """Hit rate by spout in ONE panel, same shape and colours as the aim panel.

    SOURCE (9 Nov, canonical): the purpose-built grant figure
    `rebuild_20261103/labcams/grant_figures/<tag>/epoch/`

      point + CI   `data/epoch_1bdelta_behaviour_by_position.csv`
                   `point`, `lo95`, `hi95`, and the Bonferroni-corrected
                   `lo_corrected` / `hi_corrected` over 18 comparisons
      session dots `data/epoch_1bdelta_behaviour_by_position_sessions.csv`
      stars        the CORRECTED interval excludes zero, which is the same
                   convention as `sig_bonf` on the aim panel

    Bootstrap: animals -> sessions -> position blocks within session, 2000
    draws.  A hit is a spout contact; the working period is first cue to last
    camera lick with stopped runs removed.  N = 4 animals, n = 102 sessions
    (pre 26, acute 16, subacute 17, chronic 43).

    NOT `behaviour_hits_by_spout.png` from the deck / page: that one is
    per-session dots with mean +- SEM over animals and carries no bootstrap of
    its own.  NOT `contact_frac` from `cohort/tables/prepost`: different
    quantity, different gate.

    Pre is the reference the deltas are taken against, so it has no row here
    and no group is drawn for it; the dashed line at zero is the pre level.
    """
    t = pd.read_csv('hitstats/epoch_1bdelta_behaviour_by_position.csv')
    s = pd.read_csv('hitstats/epoch_1bdelta_behaviour_by_position_sessions.csv')
    # the delta table has no pre row, so build the pre spread from the levels
    # file: each pre session minus that animal's own pre mean at that position
    lv = pd.read_csv('hitstats/epoch_1b_behaviour_by_position_sessions.csv')
    pre = lv[lv.epoch == 'pre'].copy()
    pre['value'] = pre.value - pre.groupby(['animal', 'position']).value.transform('mean')
    s = pd.concat([s, pre[['epoch', 'position', 'animal', 'value']]], ignore_index=True)
    POS = [('nI','near\nipsi'), ('nM','near\nmid'), ('nC','near\ncontra'),
           ('fI','far\nipsi'),   ('fM','far\nmid'),   ('fC','far\ncontra')]
    eps = EPOCHS
    set_scale('hit_rate_by_spout_single', 4.0)
    fig, ax = plt.subplots(figsize=(4.0, 1.52))
    w = 0.19
    ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
    rng = np.random.RandomState(11)
    for j, ep in enumerate(eps):
        for i, (code, _) in enumerate(POS):
            x0 = i + (j - (len(eps)-1)/2.0) * w
            g = s[(s.position == code) & (s.epoch == ep)]
            if len(g):
                jit = (rng.rand(len(g)) - 0.5) * w * 0.62
                ax.plot(x0 + jit, g.value.values, 'o', ms=1.9, mfc=C[ep],
                        mec='none', alpha=0.30, zorder=1)
            r = t[(t.position == code) & (t.epoch == ep)]
            if r.empty:                       # pre: the reference, no test row
                if len(g):
                    ax.plot([x0], [g.value.mean()], 'o', ms=4.2, color=C[ep],
                            mfc=C[ep], mec='k', mew=0.4, zorder=3)
                continue
            r = r.iloc[0]
            ax.errorbar([x0], [r.point],
                        yerr=[[r.point-r.lo95], [r.hi95-r.point]],
                        fmt='o', ms=4.2, color=C[ep], mfc=C[ep], mec='k',
                        mew=0.4, elinewidth=1.0, capsize=0, ls='none', zorder=3)
    ax.set_xticks(range(len(POS)))
    ax.set_xticklabels([lab for _, lab in POS])
    ax.set_ylabel('hit rate\n(change vs own pre)')
    ax.set_xlim(-0.5, len(POS)-0.5)
    headroom(ax, 0.30)
    h = [plt.Line2D([],[],marker='o',ls='none',ms=4.0,color=C[e],mfc=C[e],mec='k',mew=0.4,label=e)
         for e in eps]
    lg = ax.legend(handles=h, loc='lower left', ncol=2, frameon=False,
                   handlelength=0.8, handletextpad=0.3, columnspacing=0.7,
                   labelspacing=0.25, borderaxespad=0.15)
    for tx in lg.get_texts(): tx.set_fontsize(6.6)
    nsig = 0
    for i, (code, _) in enumerate(POS):
        for j, ep in enumerate(eps):
            r = t[(t.position == code) & (t.epoch == ep)]
            if r.empty: continue
            r = r.iloc[0]
            if not (r.lo_corrected > 0 or r.hi_corrected < 0):   # corrected CI holds 0
                continue
            nsig += 1
            ax.text(i + (j - (len(eps)-1)/2.0) * w, star_y(ax, 0.92, j, len(eps)),
                    '*', ha='center', va='center', color=C[ep],
                    fontsize=10, fontweight='bold')
    print('   hits: %d of %d cells, Bonferroni-corrected CI excludes 0' % (nsig, len(POS)*len(eps)))
    save(fig,'hit_rate_by_spout_single')

print('hit_rate_single'); hit_rate_single()


def fr_side_specificity():
    """FaceRhythm lick-factor substitution, rebuilt from data in the house style.

    Replaces the deck raster `deck/f3_FR_lickright_lickleft.png` (Overview slide
    77).  Source:
    `DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/peak/data/`

      session dots `peak_sessions_side.csv`, column `log2_vs_pre`
      point + CI   `peak_epoch_stats_side.csv`, `delta`, `ci_lo`, `ci_hi`
                   (analysis_kit.boot_delta, paired animals -> sessions,
                   animal-weighted, 95 per cent interval)
      stars        `excludes_0` after Holm correction (`p_holm` < 0.05)

    Reading: on contralesional-spout trials the contralesional lick factor
    (lick_right) falls and stays down while the ipsilesional one (lick_left)
    rises; on ipsilesional-spout trials only the acute fall survives.  The
    ipsilesional factor substitutes for the contralesional one.
    """
    t = pd.read_csv('frstats/peak_epoch_stats_side.csv')
    s = pd.read_csv('frstats/peak_sessions_side.csv')
    CELLS = [('contra','lick_right'), ('contra','lick_left'),
             ('ipsi','lick_right'),   ('ipsi','lick_left')]
    eps = EPOCHS
    set_scale('fr_side_specificity', 3.5)
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    w = 0.19
    ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
    rng = np.random.RandomState(5)
    for j, ep in enumerate(eps):
        for i, (grp, fac) in enumerate(CELLS):
            x0 = i + (j - (len(eps)-1)/2.0) * w
            g = s[(s.group == grp) & (s.name == fac) & (s.epoch == ep)]
            if len(g):
                jit = (rng.rand(len(g)) - 0.5) * w * 0.62
                ax.plot(x0 + jit, g.log2_vs_pre.values, 'o', ms=1.9,
                        mfc=C[ep], mec='none', alpha=0.30, zorder=1)
            r = t[(t.group == grp) & (t.name == fac) & (t.epoch == ep)]
            if r.empty:                      # pre is the reference, no test row
                if len(g):
                    ax.plot([x0], [g.log2_vs_pre.mean()], 'o', ms=4.2,
                            color=C[ep], mfc=C[ep], mec='k', mew=0.4, zorder=3)
                continue
            r = r.iloc[0]
            ax.errorbar([x0], [r.delta],
                        yerr=[[r.delta-r.ci_lo], [r.ci_hi-r.delta]],
                        fmt='o', ms=4.2, color=C[ep], mfc=C[ep], mec='k',
                        mew=0.4, elinewidth=1.0, capsize=0, ls='none', zorder=3)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(['contra\nfactor','ipsi\nfactor']*2)
    ax.set_ylabel('lick factor amplitude\n(log$_2$ vs own pre)')
    ax.set_xlim(-0.5, len(CELLS)-0.5)
    headroom(ax, 0.58)
    y0, y1 = ax.get_ylim()
    for x, lab in [(0.5,'contralesional spout'), (2.5,'ipsilesional spout')]:
        ax.text(x, y1 - (y1-y0)*0.015, lab, ha='center', va='top', fontsize=6.8)
    ax.axvline(1.5, color='0.85', lw=0.7, zorder=0)
    h = [plt.Line2D([],[],marker='o',ls='none',ms=4.0,color=C[e],mfc=C[e],mec='k',mew=0.4,label=e)
         for e in eps]
    lg = fig_legend(fig, ncol=4, y=-0.11)   # below the axes: cannot hit data or stars
    nsig = 0
    for i, (grp, fac) in enumerate(CELLS):
        for j, ep in enumerate(eps):
            r = t[(t.group == grp) & (t.name == fac) & (t.epoch == ep)]
            if r.empty: continue
            if not bool(r.iloc[0]['excludes_0']): continue
            nsig += 1
            ax.text(i + (j - (len(eps)-1)/2.0) * w, star_y(ax, 0.78, j, len(eps)),
                    '*', ha='center', va='center', color=C[ep],
                    fontsize=10, fontweight='bold')
    print('   fr side: %d cells, Holm-corrected CI excludes 0' % nsig)
    save(fig,'fr_side_specificity')

print('fr_side_specificity'); fr_side_specificity()


def ss_path_over_days():
    """Position on the pre -> chronic axis per session against days from stroke.

    Replaces the page raster, whose axis labels and title were a few points once
    placed. Source: `state_space_recovery/trajectories_lick_svd/days_sessions.csv`
    (one row per session: day, x = position along the pre (0) -> chronic (1)
    line, y = the orthogonal acute departure).
    """
    d = pd.read_csv('ssdata/days_sessions.csv')
    set_scale('ss_path_over_days', 2.9)
    fig, ax = plt.subplots(figsize=(2.9, 1.76))
    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.axhline(1, color='0.85', lw=0.6, zorder=0)
    ax.axvline(0, color='0.85', lw=0.6, ls='--', zorder=0)
    for ep in EPOCHS:
        g = d[d.epoch == ep]
        for an, gg in g.groupby('animal'):
            ax.plot(gg.day, gg.x, MK.get(an, 'o'), ms=3.0, mfc='none',
                    mec=C[ep], mew=0.7, alpha=0.85, zorder=2)
    # paired bootstrap, animals then sessions, of each epoch minus pre
    x1 = ax.get_xlim()[1]
    for ep, sig in (('acute', False), ('subacute', True), ('chronic', True)):
        if not sig: continue
        m = d[d.epoch == ep].groupby('animal').x.mean().mean()
        ax.text(x1*1.01, m, '*', ha='left', va='center', color=C[ep],
                fontsize=10, fontweight='bold', clip_on=False)
    ax.set_xlabel('days from stroke')
    ax.set_ylabel('along pre (0)\n$\\rightarrow$ chronic (1)')
    fig_legend(fig, ncol=4, y=-0.18)
    save(fig, 'ss_path_over_days')

print('ss_path_over_days'); ss_path_over_days()

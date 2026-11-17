"""Figure 4 rows rendered as ONE canvas each, at final size (7.4 in wide).

Why: panels that sit side by side were separate files, each trimmed with
bbox_inches='tight', so their plot areas could never share a baseline. Here
every axes is placed in absolute inches on a shared canvas: the x-axis lines
of D and E (and of F, G, H) are the same line by construction, and the
compose step places each row at scale 1 so type is the true point size.
Axes positions are written to rows_fig4_layout.json for the panel letters.
"""
import sys, json; sys.path.insert(0, '.')
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from grant_panels import *
from grant_panels import star_y, _BASE

plt.rcParams.update(_BASE)
D = 'data/'
W = 7.4
LAYOUT = {}

def axin(fig, l, b, w, h, **kw):
    H = fig.get_figheight()
    return fig.add_axes([l / W, b / H, w / W, h / H], **kw)

def epoch_key(fig, y=0.03):
    h = [plt.Line2D([], [], marker='o', ls='none', ms=4.0, color=C[e], mfc=C[e],
                    mec='k', mew=0.4, label=e) for e in EPOCHS]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, y / fig.get_figheight()),
               ncol=4, frameon=False, handlelength=0.8, handletextpad=0.3,
               columnspacing=1.4, borderaxespad=0.0, fontsize=7.0)

def out(fig, name, axes):
    LAYOUT[name] = {'h_in': fig.get_figheight(),
                    'axes': {k: [a.get_position().x0 * W,
                                 (1 - a.get_position().y1) * fig.get_figheight()]
                             for k, a in axes.items()}}
    for ext in ('svg', 'png'):
        fig.savefig('%s/%s.%s' % (OUT, name, ext), transparent=False)
    plt.close(fig)
    print('  wrote', name)

# ------------------------------------------------------------------ D + E
def row_DE():
    H = 1.95; BOT, AH = 0.66, 1.06
    fig = plt.figure(figsize=(W, H))
    axD = axin(fig, 0.58, BOT, 3.15, AH)
    axE1 = axin(fig, 4.48, BOT, 1.22, AH)
    axE2 = axin(fig, 5.86, BOT, 1.22, AH, sharey=axE1)

    # D: lick-response AUC, fold of own pre, spouts pooled
    d = pd.read_csv(D + 'cued_ramp_w01_sessions.csv')
    v = pd.read_csv(D + 'cued_ramp_w01_verify.csv')
    rois = ['tjM1_right', 'tjM1_left', 'SSp-m_right', 'SSp-m_left']
    lab = {'tjM1_right': 'tjM1\ncontra', 'tjM1_left': 'tjM1\nipsi',
           'SSp-m_right': 'SSp-m\ncontra', 'SSp-m_left': 'SSp-m\nipsi'}
    d = d[d.position.isin(rois)]
    w = 0.19; rng = np.random.default_rng(0)
    for j, ep in enumerate(EPOCHS):
        xs, ys, es = [], [], []
        for i, roi in enumerate(rois):
            sub = d[(d.position == roi) & (d.epoch == ep)]
            if sub.empty: continue
            xc = i + (j - 1.5) * w
            for an, g in sub.groupby('animal'):
                axD.plot(xc + rng.uniform(-0.055, 0.055, len(g)), g.value, MK.get(an, 'o'),
                         ms=2.4, mfc='none', mec=C[ep], mew=0.5, alpha=0.45, zorder=1)
            xs.append(xc); ys.append(sub.value.mean())
            es.append(sub.value.std(ddof=1) / max(np.sqrt(sub.animal.nunique()), 1))
        axD.errorbar(xs, ys, yerr=es, fmt='o', ms=4.0, color=C[ep], mfc=C[ep], mec='k',
                     mew=0.4, elinewidth=1.1, capsize=0, ls='none', zorder=3)
    axD.axhline(1.0, color='0.6', lw=0.6, ls='--', zorder=0)
    axD.set_xticks(range(len(rois))); axD.set_xticklabels([lab[r] for r in rois])
    axD.set_xlim(-0.5, len(rois) - 0.5)
    axD.set_ylabel('lick-response AUC\n(fold of own pre)')
    axD.set_title('Lick-response magnitude', pad=3)
    lo, hi = axD.get_ylim(); hi = min(hi, 3.05)
    axD.set_ylim(lo, hi + (hi - lo) * 0.22)
    for i, roi in enumerate(rois):
        sub = v[v.position == roi]
        for j, ep in enumerate(EPOCHS):
            r = sub[sub.epoch == ep]
            if r.empty: continue
            m = str(r.iloc[0].get('boot_star', '') or '').strip()
            if m and m.lower() != 'nan':
                axD.text(i + (j - 1.5) * w, star_y(axD, 0.90, j, 4, step=0.03), m, ha='center',
                         va='center', color=C[ep], fontsize=10, fontweight='bold')

    # E: contra:ipsi balance, far contra spout, tjM1 and SSp-m
    e = pd.read_csv(D + 'lick_lateral_ratio_auc_docked_sessions.csv')
    lmm = pd.read_csv(D + 'lick_lateral_ratio_auc_docked_lmm.csv')
    e = e[e.panel.str.endswith(', cued') & e.position.str.endswith(':far_contra')].copy()
    e['roi'] = e.panel.str.split(',').str[0]
    np.random.seed(1)
    for ax, roi in ((axE1, 'tjM1'), (axE2, 'SSp-m')):
        sub = e[e.roi == roi]
        epoch_strip(ax, sub)
        ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
        ax.set_title(roi, pad=2)
    lo, hi = axE1.get_ylim(); axE1.set_ylim(lo, hi + (hi - lo) * 0.20)
    for ax, roi in ((axE1, 'tjM1'), (axE2, 'SSp-m')):
        pos = e[e.roi == roi].position.iloc[0]
        stars(ax, lmm, pos, band=0.89, step=0.03)
    plt.setp(axE2.get_yticklabels(), visible=False)
    axE1.set_ylabel('contra : ipsi balance\n(log$_2$, far contra spout)')
    epoch_key(fig, 0.02)
    out(fig, 'fig4_row_DE', {'d': axD, 'e': axE1})

# ------------------------------------------------------------------ F + G + H
def row_FGH():
    H = 2.02; BOT, AH = 0.62, 1.22
    fig = plt.figure(figsize=(W, H))
    axF = axin(fig, 0.60, BOT, 1.72, AH)
    axG = axin(fig, 3.02, BOT, 1.72, AH)
    axH = axin(fig, 5.40, BOT, 1.72, AH)

    # F: the 5E measure itself (log2 contra/ipsi lick-response AUC, first cued lick,
    # far contra spout, SSp-m) against far-contra first-lick aim, one point per session
    bal = pd.read_csv(D + 'lick_lateral_ratio_auc_docked_sessions.csv')
    bal = bal[bal.position == 'cued|SSp-m:far_contra'][['label', 'value']]
    t = pd.read_csv('recov/balance_aim_sessions.csv').merge(bal, on='label')
    axF.axhline(0, color='0.85', lw=0.6, zorder=0)
    for ep in EPOCHS:
        for an, g in t[t.epoch == ep].groupby('animal'):
            axF.plot(g.value, g.y, MK.get(an, 'o'), ms=3.0, mfc=C[ep], mec='none',
                     alpha=0.9 if ep != 'pre' else 0.75, zorder=2 if ep != 'pre' else 1)
    post = t[t.epoch != 'pre']
    for an, g in post.groupby('animal'):
        b1, b0 = np.polyfit(g.value, g.y, 1)
        xx = np.array([g.value.min(), g.value.max()])
        axF.plot(xx, b0 + b1 * xx, ':', color='0.5', lw=1.0, zorder=3)
    axF.set_xlabel('SSp-m contra : ipsi balance\n(log$_2$, far contra spout)')
    axF.set_ylabel('far contra aim\n(deg vs pre)')
    axF.set_title('Balance vs aim', pad=3)
    axF.text(0.03, 0.04, 'within-animal slope\np < 10$^{-9}$, adjusted for time', transform=axF.transAxes,
             ha='left', va='bottom', fontsize=6.8)

    # G: position on the pre -> chronic axis against days from stroke
    s = pd.read_csv('ssdata/days_sessions.csv')
    axG.axhline(0, color='0.85', lw=0.6, zorder=0); axG.axhline(1, color='0.85', lw=0.6, zorder=0)
    axG.axvline(0, color='0.85', lw=0.6, ls='--', zorder=0)
    for ep in EPOCHS:
        for an, gg in s[s.epoch == ep].groupby('animal'):
            axG.plot(gg.day, gg.x, MK.get(an, 'o'), ms=3.0, mfc='none', mec=C[ep], mew=0.7,
                     alpha=0.85, zorder=2)
    lo, hi = axG.get_ylim(); axG.set_ylim(lo, hi + (hi - lo) * 0.24)
    # paired bootstrap vs pre (animals, then sessions): subacute and chronic
    # displaced; acute n.s. One star per epoch, centred over its days, in a
    # band above the data.
    for ep in ('subacute', 'chronic'):
        dd = s[s.epoch == ep].day
        axG.text((dd.min() + dd.max()) / 2.0, star_y(axG, 0.93, 0, 1), '*', ha='center',
                 va='center', color=C[ep], fontsize=14, fontweight='bold')
    axG.set_xlabel('days from stroke')
    axG.set_title('Recovery axis over time', pad=3)
    axG.set_ylabel('along pre (0)\nto chronic (1)')

    # H: sessions in the recovery-axis / acute-departure plane
    p = pd.read_csv(D + 'ss_plane_sessions.csv')
    for v0 in (0,): axH.axhline(v0, color='0.85', lw=0.7, zorder=0)
    axH.axvline(0, color='0.85', lw=0.7, zorder=0); axH.axvline(1, color='0.85', lw=0.7, zorder=0)
    means = {}
    for ep in EPOCHS:
        sub = p[p.epoch == ep]
        for an, g in sub.groupby('animal'):
            axH.plot(g.x, g.y, MK.get(an, 'o'), ms=2.6, mfc='none', mec=C[ep], mew=0.5,
                     alpha=0.22, zorder=1)
        am = sub.groupby('animal')[['x', 'y']].mean(); n = max(len(am), 1)
        mx, my = am.x.mean(), am.y.mean()
        ex = 1.96 * am.x.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        ey = 1.96 * am.y.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        axH.errorbar(mx, my, xerr=ex, yerr=ey, fmt='o', ms=5.5, color=C[ep], mfc=C[ep], mec='k',
                     mew=0.6, elinewidth=1.3, capsize=0, zorder=4)
        means[ep] = mx
    lo, hi = axH.get_ylim(); axH.set_ylim(lo, hi + (hi - lo) * 0.22)
    # stars in a clear band above the cloud, each over its own epoch mean
    for j, ep in enumerate(('acute', 'subacute', 'chronic')):
        axH.text(means[ep], star_y(axH, 0.93, j, 3, step=0.0), '*', ha='center', va='center',
                 color=C[ep], fontsize=10, fontweight='bold')
    axH.set_xlabel('along pre (0) to chronic (1)')
    axH.set_title('Pre-to-chronic plane', pad=3)
    axH.set_ylabel('acute departure\n(orthogonal)')
    epoch_key(fig, 0.02)
    out(fig, 'fig4_row_FGH', {'f': axF, 'g': axG, 'h': axH})


# ------------------------------------------------------------------ A + B
def row_AB():
    H = 2.02; BOT, AH = 0.66, 1.0
    fig = plt.figure(figsize=(W, H))
    a1 = axin(fig, 0.58, BOT, 1.02, AH)
    a2 = axin(fig, 1.78, BOT, 1.02, AH, sharey=a1)
    a3 = axin(fig, 3.42, BOT, 1.02, AH)
    b1 = axin(fig, 5.10, BOT, 0.98, AH)
    b2 = axin(fig, 6.22, BOT, 0.98, AH, sharey=b1)
    # A: frozen / refit / gap on the LocaNMF basis (state_space_recovery/frozen_refit_basis,
    # 9-10 Nov; August-only pre). Marks: Bonferroni-corrected paired bootstrap vs pre.
    fs = pd.read_csv('frbasis/sessions.csv'); fs = fs[fs.basis == 'lnmf']
    d = fs.melt(id_vars=['animal', 'label', 'epoch'], value_vars=['frozen', 'refit', 'gap'],
                var_name='position', value_name='value')
    st = pd.read_csv('frbasis/stats.csv'); st = st[st.basis == 'lnmf'].rename(columns={'panel': 'position'})
    np.random.seed(2)
    for ax, (pos, title) in zip((a1, a2, a3), [('frozen', 'frozen pre-stroke\ndecoder'),
                                               ('refit', 'within-session\nrefit decoder'),
                                               ('gap', 'refit \u2212 frozen gap')]):
        epoch_strip(ax, d[d.position == pos]); ax.set_title(title, pad=2)
    for ax in (a1, a2):
        ax.axhline(1/6, color='0.6', lw=0.6, ls='--', zorder=0)
    a3.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
    a1.set_ylim(0.0, 1.12)
    lo, hi = a3.get_ylim(); a3.set_ylim(lo, hi + (hi - lo) * 0.26)
    plt.setp(a2.get_yticklabels(), visible=False)
    a1.set_ylabel('decoding accuracy'); a3.set_ylabel('accuracy difference')
    a1.text(3.45, 1/6 + 0.02, 'chance', ha='right', va='bottom', fontsize=6.4, color='0.45')
    for ax, pos in ((a1, 'frozen'), (a2, 'refit'), (a3, 'gap')):
        sub = st[st.position == pos]
        for i, ep in enumerate(EPOCHS):
            rr = sub[sub.epoch == ep]
            if rr.empty: continue
            m = str(rr.iloc[0].get('mark', '') or '').strip()
            if m and m.lower() != 'nan':
                ax.text(i, star_y(ax, 0.90, i, 4, step=0.03), m, ha='center', va='center', color=C[ep],
                        fontsize=11, fontweight='bold')
    o   = pd.read_csv(D + 'lick_cued_first_lick_sessions.csv')
    lmm = pd.read_csv(D + 'lick_cued_first_lick_lmm.csv')
    for ax, (pos, title) in zip((b1, b2), [('all spouts|lfl:first', 'all spouts'),
                                           ('far contra|lfl:first', 'far contra')]):
        epoch_strip(ax, o[o.position == pos]); ax.set_title(title, pad=2)
        ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
    lo, hi = b1.get_ylim(); b1.set_ylim(lo, hi + (hi - lo) * 0.22)
    plt.setp(b2.get_yticklabels(), visible=False)
    b1.set_ylabel('like-for-like\noffset (mm)')
    for ax, pos in ((b1, 'all spouts|lfl:first'), (b2, 'far contra|lfl:first')):
        stars(ax, lmm, pos, band=0.89, step=0.03)
    epoch_key(fig, 0.02)
    out(fig, 'fig4_row_AB', {'a': a1, 'b': b1})

row_AB(); row_DE(); row_FGH()
json.dump(LAYOUT, open('rows_fig4_layout.json', 'w'), indent=1)

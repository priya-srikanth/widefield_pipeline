"""Grant-sized panels re-plotted from the lmm_hidden session CSVs.

Why re-plot rather than crop: the deck figures carry supertitles, method blurbs,
full-width legends and every position/ROI. The grant needs a subset, bigger type
and a legend tucked into panel whitespace. Source data are the same CSVs the
deck figures were drawn from, so nothing is re-derived here.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

OUT = 'K08_2026_grant_panels'
os.makedirs(OUT, exist_ok=True)

EPOCHS = ['pre', 'acute', 'subacute', 'chronic']
C = {'pre': '#808080', 'acute': '#e0218a', 'subacute': '#d4a017', 'chronic': '#1b7f79'}
MK = {'PS92': 'o', 'PS93': 's', 'PS94': '^', 'PS95': 'D'}

plt.rcParams.update({
    'svg.fonttype': 'none',          # keeps text editable in Illustrator
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8, 'axes.titleweight': 'bold',
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 6.8, 'legend.handletextpad': 0.4,
    'legend.borderpad': 0.25, 'legend.labelspacing': 0.28,
    'legend.borderaxespad': 0.3, 'legend.columnspacing': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.0, 'figure.dpi': 200,
})


def epoch_strip(ax, df, value='value', jitter=0.11, ms=3.0):
    """Open session markers by animal + filled epoch mean with SEM."""
    for i, ep in enumerate(EPOCHS):
        sub = df[df.epoch == ep]
        if sub.empty:
            continue
        for an, g in sub.groupby('animal'):
            x = i + np.random.uniform(-jitter, jitter, len(g))
            ax.plot(x, g[value], MK.get(an, 'o'), ms=ms, mfc='none',
                    mec=C[ep], mew=0.6, alpha=0.75, zorder=1,
                    label='_' )
        m = sub[value].mean()
        se = sub[value].std(ddof=1) / max(np.sqrt(sub.animal.nunique()), 1)
        ax.errorbar(i, m, yerr=se, fmt=MK['PS92'], ms=5.2, color=C[ep],
                    mfc=C[ep], mec='k', mew=0.5, elinewidth=1.1, capsize=0, zorder=3)
    ax.set_xticks(range(len(EPOCHS)))
    # spelled out, slanted: four full epoch names do not fit horizontally in a
    # panel this narrow, and the three-letter forms were not worth the ambiguity
    ax.set_xticklabels(EPOCHS, rotation=32, ha='right', rotation_mode='anchor')
    ax.set_xlim(-0.5, len(EPOCHS) - 0.5)


def headroom(ax, frac=0.34):
    """Open a clean band at the top of the panel for stars and the key, so
    neither can ever sit on the data."""
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * frac)
    return y0, y1


def star_y(ax, band, j, n, step=0.055):
    """Y for one epoch's star, staggered slightly so neighbouring epochs do not
    collide when the groups sit close together on the x axis."""
    y0, y1 = ax.get_ylim()
    return y0 + (y1 - y0) * (band + (j - (n - 1) / 2.0) * step)


def stars(ax, lmm, position, band=None, size=11, step=0.055):
    """Significance marks above each epoch, read from the figure's own LMM table.
    `lmm` must have columns position, epoch, star."""
    sub = lmm[lmm.position == position]
    if sub.empty:
        return
    b = 0.80 if band is None else band
    for i, ep in enumerate(EPOCHS):
        row = sub[sub.epoch == ep]
        if row.empty:
            continue
        mark = str(row.iloc[0].get('star', '') or '').strip()
        if mark and mark.lower() != 'nan':
            ax.text(i, star_y(ax, b, i, len(EPOCHS), step), mark, ha='center',
                    va='center', color=C[ep], fontsize=size,
                    fontweight='bold', clip_on=False)


def _legend_fits(ax, lg):
    """True when the drawn legend sits inside the axes box horizontally."""
    fig = ax.figure
    fig.canvas.draw()
    lb = lg.get_window_extent()
    ab = ax.get_window_extent()
    return lb.x0 >= ab.x0 - 1 and lb.x1 <= ab.x1 + 1


def epoch_legend(ax, style='row', loc=None, size=6.6, shrink_to=5.2):
    """Compact frameless epoch key, auto-fitted so it never runs past the axes.

    `style` is a preference, not a promise: the key is drawn, measured, and if it
    overhangs the panel it falls back through narrower shapes and then smaller
    type until it fits. A four-across key suits a wide panel; a narrow one needs
    2 x 2 or a single column.
    """
    shape = {'row': (4, 'upper center'), 'pair': (2, 'upper right'),
             'block': (2, 'upper left'), 'stack': (1, 'upper left')}
    order = [style] + [k for k in ('row', 'pair', 'block', 'stack') if k != style]
    h = [plt.Line2D([], [], marker='o', ls='none', ms=4.0, color=C[e],
                    mfc=C[e], mec='k', mew=0.4, label=e) for e in EPOCHS]
    lg = None
    for sz in (size, size - 0.6, shrink_to):
        for st in order:
            ncol, default_loc = shape[st]
            if lg is not None:
                lg.remove()
            lg = ax.legend(handles=h, loc=loc or default_loc, ncol=ncol,
                           frameon=False, handlelength=0.8, handletextpad=0.3,
                           columnspacing=0.7, borderaxespad=0.2)
            for t in lg.get_texts():
                t.set_fontsize(sz)
            if _legend_fits(ax, lg):
                return lg
    return lg


def star_band(ax, lg=None, gap=0.07):
    """Y position (axes fraction) for the star row: just under the legend when
    there is one, otherwise near the top of the headroom band."""
    if lg is None:
        return 0.86
    ax.figure.canvas.draw()
    lb = lg.get_window_extent()
    inv = ax.transAxes.inverted()
    band = inv.transform((0, lb.y0))[1] - gap
    if band < 0.30:          # legend hung below the axes: it is not in the way
        return 0.86
    return band


def animal_legend(ax, animals, loc='upper left', ncol=2):
    h = [plt.Line2D([], [], marker=MK.get(a, 'o'), ls='none', ms=3.4,
                    mfc='none', mec='0.35', mew=0.6, label=a) for a in animals]
    ax.legend(handles=h, loc=loc, ncol=ncol, frameon=False, handlelength=0.8)


def save(fig, name):
    for ext in ('svg', 'png'):
        fig.savefig(os.path.join(OUT, name + '.' + ext), bbox_inches='tight',
                    pad_inches=0.02, transparent=False)
    plt.close(fig)
    print('  wrote', name)


def fig_legend(fig, eps=None, ncol=4, y=-0.015, size=6.8):
    """One legend for the whole panel, hung BELOW the figure so it can never
    collide with data, tick labels or the significance markers. `save()` uses
    bbox_inches='tight', which grows the canvas to include it."""
    import matplotlib.pyplot as _plt
    eps = eps or EPOCHS
    h = [_plt.Line2D([], [], marker='o', ls='none', ms=4.0, color=C[e], mfc=C[e],
                     mec='k', mew=0.4, label=e) for e in eps]
    lg = fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, y),
                    ncol=ncol, frameon=False, handlelength=0.8,
                    handletextpad=0.3, columnspacing=1.2, borderaxespad=0.0)
    for t in lg.get_texts(): t.set_fontsize(size)
    return lg


#: Final width IN THE FIGURE, in points, for every panel that gets placed by
#: make_fig1/2/3. A panel rendered wider than this is shrunk when it is placed,
#: and its 7.5 pt ticks then read as 7.5 x scale. `set_scale` compensates so the
#: text is the same true size in every panel of the grant.
PLACED_PT = {
    'hit_rate_by_spout_single':           0.62,
    'first_lick_aim_by_spout_single':     0.62,
    'ss_full_nocw_frKBs':                 0.62,
    'state_space_plane_sessions':         0.26,
    'frozen_refit_gap_pooled':            0.61,
    'decoder_offset_cued_first_lick':     0.36,
    'hemispheric_balance_cued_farcontra': 0.38,
    'lick_response_auc_compressed':       0.50,
    'fr_side_specificity':                0.41,
    'ss_path_over_days':                  0.30,
    'cam_frames':                         0.355,
}
FIG_W_PT = 7.4 * 72.0

_BASE = {'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 6.8}

def set_scale(name, fig_w_in):
    """Scale the type so that, once this panel is placed at its final width in
    the figure, the text reads at the sizes in `_BASE`. Call right before
    building a panel, with the figure width the panel is rendered at."""
    frac = PLACED_PT.get(name)
    if not frac:
        plt.rcParams.update(_BASE); return 1.0
    sc = (FIG_W_PT * frac) / (fig_w_in * 72.0)
    plt.rcParams.update({k: v / sc for k, v in _BASE.items()})
    return sc

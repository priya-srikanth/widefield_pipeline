"""Stand-alone colour bars for the FaceRhythm map panels.

The bars inside the source figures are a few points wide and carry 2 pt numbers;
cropping them out of the parent SVG and rescaling proved fragile, so they are
redrawn here at the figure's own type size with the same colormap and limits.
"""
import sys; sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from grant_panels import save

def bar(name, vmin, vmax, ticks, label, h=1.9):
    fig = plt.figure(figsize=(0.62, h))
    ax = fig.add_axes([0.04, 0.06, 0.26, 0.88])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap='RdBu_r'),
                      cax=ax, ticks=ticks)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2.0, width=0.5, labelsize=6.8, pad=1.6)
    cb.set_label(label, fontsize=6.8, labelpad=2.0)
    save(fig, name)

bar('cbar_loading', -6.2, 6.2, [-5, 0, 5], 'loading vs pre (SD)')
bar('cbar_speed', -1.0, 1.0, [-1, 0, 1], 'log$_2$ speed vs pre', h=1.4)

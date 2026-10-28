"""Figure-data cache for the ITI / cued figure scripts (Priya 10-27: "ensure all models / data analysis is being saved so
if we want to change plotting style we can re-make plots without re-doing analysis").

Convention: the analysis step of a figure script writes ONE pickle per figure (or per script) holding everything the
figure draws -- point estimates, CIs, per-session / per-animal values, curves, map arrays -- to
<il.OUT>/figdata/<figure_stem>.pkl. With `--plot-only` the script loads ONLY that file and redraws: no bootstrap, no fit,
so a restyle takes seconds and no number can shift. The drawing function takes the figure-data object as its input.

    from figdata import plot_only, save_fd, load_fd
    if plot_only():
        D = load_fd("lick_foo")
    else:
        D = compute()
        save_fd("lick_foo", D)
    draw(D)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

FIGDATA = il.OUT / "figdata"


def plot_only() -> bool:
    return "--plot-only" in sys.argv


def fd_path(stem: str, root: Path | None = None) -> Path:
    return (root or FIGDATA) / f"{Path(stem).stem}.pkl"


def save_fd(stem: str, obj, root: Path | None = None) -> Path:
    """Write the figure data atomically (tmp + replace: a crash mid-write cannot leave a truncated cache)."""
    p = fd_path(stem, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(p)
    return p


def load_fd(stem: str, root: Path | None = None):
    p = fd_path(stem, root)
    if not p.exists():
        raise SystemExit(f"--plot-only: no figure data at {p}; run the script once without --plot-only")
    return pickle.loads(p.read_bytes())


def imaging_grid() -> dict:
    """The arrays a cortex-map figure needs to place stat-mask pixels on the image (as `iti_decoder_pattern_stats.plot`
    built inline): pix (flat indices of the stat-mask pixels), shape (H, W), ol (Allen outline, every 2nd pixel).
    Picklable, so it is saved WITH the figure data and `--plot-only` needs no imaging load."""
    import numpy as np
    from frozen_area_maps import area_image, outline

    from wfield_local import combined_encoding as ce
    lab0 = il.labels()[0]
    _V, _ft, _Ub, flat, _pa, _nm, shape, pstat = ce.load_imaging(lab0)
    img, _ = area_image(lab0)
    return dict(pix=np.asarray(flat[pstat]), shape=tuple(int(x) for x in shape), ol=outline(img)[::2, ::2])


def grid_fns(grid: dict):
    """(to_img, ol) from `imaging_grid()`: to_img(v, fill=nan) puts a stat-mask vector on the half-resolution image."""
    import numpy as np
    Hh, Ww = grid["shape"]
    pix = grid["pix"]

    def to_img(v, fill=np.nan):
        g = np.full(Hh * Ww, fill, dtype=float)
        g[pix] = v
        return g.reshape(Hh, Ww)[::2, ::2]
    return to_img, grid["ol"]


# Map colour convention (Priya 10-27: "make the pre map a different colormap so it's clear they are different"). The pre
# (and any epoch LEVEL) Haufe map is a SIGNED push-pull map, so it stays diverging and centred on 0 -- but purple-orange,
# so it cannot be mistaken for an epoch - pre CHANGE map (red-blue). Level maps of one figure share one scale.
PRE_CMAP = "PuOr_r"
CHANGE_CMAP = "RdBu_r"
CMAP_NOTE = "pre map in purple-orange (level); change maps in red-blue"


def sym_lim(arrays, q: float = 99) -> float:
    """Symmetric colour limit: the q-th percentile of |values| over all arrays (NaN ignored); 1 if nothing finite."""
    import numpy as np
    v = np.abs(np.concatenate([np.ravel(a) for a in arrays])) if len(arrays) else np.array([])
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else 1.0


def levels_name(png: str) -> str:
    """'<stem>.png' -> '<stem>_levels.png' (the supplementary all-levels figure next to a pre + change figure)."""
    return f"{Path(png).stem}_levels.png"

"""Paint an exclusion mask by hand, on the real fluorescence, with the CCF outlines to steer by.

    python -m wfield_local.paint_exclusion

WHY A PAINTER AND NOT A THRESHOLD. The fibre glue is the thing this exists for, and it cannot be
found automatically: three attempts failed on 2026-09-12, each for a different reason. Thresholding
DARK pixels found the anterolateral somatosensory band, because the whole left hemisphere is dimmer
than the right (a tilt/illumination gradient, not glue). Thresholding the left/right MIRROR RATIO
found the same thing for the same reason. Thresholding locally BRIGHT pixels -- glue scatters light,
so bright is the right sign -- found MOs, RSPd and RSPagl, because the medial dorsal surface is
flatter to the camera and therefore brighter everywhere. Brightness in this preparation is dominated
by geometry, and no threshold on it isolates the glue.

The person who placed the fibre can see it in one glance. This turns that into a mask.

AND A REGION LIST IS NOT ENOUGH EITHER. Priya's description -- "L Vis and aud partial" -- is
PARTIAL, and Allen regions are all-or-nothing: excluding VISp_left to catch a corner of it throws
away 11,886 px, 6% of the whole mask. A painted mask can follow the actual footprint. The tool
still PRINTS the Allen breakdown on save, so a region list stays available if the painted shape
turns out to land cleanly on a few areas.

CONTROLS
    drag                paint            shift+drag / right-drag   erase
    [ and ]             brush smaller / larger        scroll   brush size
    1-4                 switch animal (mean motion-corrected frames for that animal)
    0                   the across-animal mean
    f                   toggle 470 (functional) / 415 (isosbestic)
    a                   toggle CCF outlines
    o                   toggle the already-excluded regions (MOB) in red
    u                   undo the last stroke            c   clear everything
    s                   SAVE and print the Allen breakdown
    q                   quit without saving

THE MASK IS ADDITIVE TO `beta_maps.EXCLUDE_REGIONS`, not a replacement: the bulbs stay excluded by
name, and this covers what a name cannot express.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

#: Where the painted mask lands. Under the labcams root so both machines see it, and guarded by
#: `writeguard.assert_writable` like every other write to the share.
MASK_NAME = "exclusion_mask_painted.npy"


#: Channel index -> what it is. `hemo_variants.FUNC` is the FUNCTIONAL channel's index in the
#: interleaved stream and it is 1, so channel 0 is the isosbestic 415. Named rather than assumed,
#: because getting it backwards would put the calcium-free image up as "the data" and nothing about
#: the picture would say so.
CHANNELS = {1: "470 (functional)", 0: "415 (isosbestic)"}


def _mean_fluorescence(animals=None, max_sessions=10):
    """``{(animal, channel): (540, 640)}`` -- motion-corrected mean frames, per animal per channel.

    BOTH CHANNELS, because they fail differently and the glue shows in both. 470 carries calcium
    and haemodynamics; 415 carries haemodynamics and whatever is optically in the way -- so a
    surface occlusion appears in the ISOSBESTIC image without any of the activity that complicates
    the functional one. Asked for by Priya, 2026-09-12, while identifying the fibre glue.

    NORMALISED PER SESSION by its own in-mask median before averaging, so one bright day does not
    set the picture; the absolute level is meaningless here and the shape is what is being looked
    at. `"mean"` is added as a pseudo-animal, averaged across animals within each channel.
    """
    from wfield_local import beta_maps as bm
    from wfield_local import config
    from wfield_local.grant_figures import ANIMALS
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    full = bm.brain_mask() | bm.excluded_mask()
    out = {}
    for an in (animals or ANIMALS):
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        got = {c: [] for c in CHANNELS}
        for s in [x for x in SESSIONS if x["label"] in want]:
            ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
            if not ad:
                continue
            try:
                fa = np.asarray(np.load(f"{ad[0]}/frames_average_atlas.npy"), float)
            except Exception:                                          # noqa: BLE001
                continue
            if fa.ndim == 2:
                fa = fa[None]
            for c in CHANNELS:
                if c >= fa.shape[0] or fa[c].shape != bm.MAP_SHAPE:
                    continue
                med = float(np.nanmedian(fa[c][full]))
                if med > 0:
                    got[c].append(fa[c] / med)
            if min(len(v) for v in got.values()) >= max_sessions:
                break
        for c, v in got.items():
            if v:
                out[(an, c)] = np.nanmean(v, axis=0)
    for c in CHANNELS:
        same = [v for (a_, c_), v in out.items() if c_ == c]
        if same:
            out[("mean", c)] = np.nanmean(same, axis=0)
    return out


def region_breakdown(mask):
    """``[(region, px_in_mask, px_in_region, fraction)]`` for what a painted mask covers.

    THE FRACTION IS THE POINT. A painted shape clipping 8% of VISp is a different fact from one
    covering 95% of it: the first says the region is mostly fine and a name-based exclusion would
    be far too blunt, the second says just exclude the region.
    """
    from wfield_local import beta_maps as bm

    atlas, names = bm._atlas_names()
    if atlas is None:
        return []
    full = bm.brain_mask() | bm.excluded_mask()
    mask = np.asarray(mask, bool) & full
    out = []
    for sid, nm in names.items():
        reg = (atlas == sid) & full
        n_reg = int(reg.sum())
        if n_reg < 50:
            continue
        n_hit = int((reg & mask).sum())
        if n_hit:
            out.append((nm, n_hit, n_reg, n_hit / n_reg))
    return sorted(out, key=lambda r: -r[1])


def save_mask(mask, path=None):
    """Write the mask and return its path. Guarded, like every write to the share."""
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable

    if path is None:
        path = Path(PathResolver().root("labcams")) / "grant_figures" / MASK_NAME
    path = Path(path)
    assert_writable(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(mask, bool))
    return path


def main(argv=None) -> int:
    from wfield_local.console import use_utf8_stdout
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="where to save (default: share)")
    ap.add_argument("--load", type=Path, default=None, help="start from an existing mask")
    ap.add_argument("--brush", type=int, default=14)
    a = ap.parse_args(argv)

    # EVERY wfield_local IMPORT FIRST, AND ONLY THEN THE BACKEND. `grant_figures`,
    # `locanmf_cue_lick_analysis` and several others call `matplotlib.use("Agg")` AT IMPORT TIME,
    # so selecting an interactive backend before importing them is silently undone -- the window
    # never appears and matplotlib says only "FigureCanvasAgg is non-interactive" at `show()`.
    # Import everything, load the data, and switch the backend last.
    from wfield_local import beta_maps as bm
    from wfield_local.atlas_overlay import region_edges

    import matplotlib

    atlas, _names = bm._atlas_names()
    full = bm.brain_mask() | bm.excluded_mask()
    already = bm.excluded_mask()
    edges = region_edges(atlas) if atlas is not None else None
    imgs = _mean_fluorescence()
    if not imgs:
        print("no fluorescence found")
        return 1
    animals_present = [a for a in dict.fromkeys(a for a, _c in imgs) if a != "mean"]
    order = ["mean"] + animals_present

    mask = np.zeros(bm.MAP_SHAPE, bool)
    if a.load and Path(a.load).exists():
        mask = np.load(a.load).astype(bool)
        print(f"loaded {int(mask.sum()):,} px from {a.load}")

    # NOW the backend, with every Agg-setting import already done.
    chosen = None
    for backend in ("TkAgg", "QtAgg", "Qt5Agg", "wxAgg"):
        try:
            matplotlib.use(backend, force=True)
            chosen = backend
            break
        except Exception:                                              # noqa: BLE001
            continue
    if chosen is None or chosen.lower().endswith("agg") and chosen == "Agg":
        print("no interactive matplotlib backend available -- cannot paint. "
              "Install tk (conda install tk) or PyQt.")
        return 2
    import matplotlib.pyplot as plt

    if not matplotlib.get_backend().lower().startswith(("tk", "qt", "wx")):
        print(f"backend is {matplotlib.get_backend()!r}, which is not interactive -- "
              "something re-set it after selection. Cannot paint.")
        return 2
    print(f"backend: {matplotlib.get_backend()}")

    # CHANNEL 1 (470) IS THE DEFAULT VIEW because it is what every other figure is built from;
    # `f` swaps to the isosbestic, where an optical obstruction shows without activity on top.
    state = {"key": "mean", "chan": 1 if ("mean", 1) in imgs else 0, "brush": int(a.brush),
             "painting": False, "erase": False, "outlines": True, "show_excluded": True,
             "undo": []}

    fig, ax = plt.subplots(figsize=(11, 10))
    fig.canvas.manager.set_window_title("paint the exclusion mask  --  s=save  u=undo  q=quit")

    yy, xx = np.mgrid[0:bm.MAP_SHAPE[0], 0:bm.MAP_SHAPE[1]]

    def draw():
        ax.clear()
        v = imgs.get((state["key"], state["chan"]))
        if v is None:
            v = imgs[(state["key"], 1 - state["chan"])]
        v = v.copy()
        v[~full] = np.nan
        ax.imshow(v, cmap="gray", vmin=np.nanpercentile(v, 2), vmax=np.nanpercentile(v, 99))
        if state["outlines"] and edges is not None:
            ax.contour(edges.astype(float), levels=[0.5], colors="#00d0ff", linewidths=0.7,
                       alpha=0.8)
        if state["show_excluded"] and already.any():
            ax.contourf(already.astype(float), levels=[0.5, 1.5], colors=["#d62728"], alpha=0.35)
        if mask.any():
            ax.contourf(mask.astype(float), levels=[0.5, 1.5], colors=["#ffcc00"], alpha=0.45)
            ax.contour(mask.astype(float), levels=[0.5], colors="#ff8800", linewidths=1.5)
        ax.set_axis_off()
        ax.set_title(f"{state['key']}   ch {state['chan']} = {CHANNELS[state['chan']]}   "
                     f"brush {state['brush']}px   painted {int(mask.sum()):,} px "
                     f"({100 * mask.sum() / max(full.sum(), 1):.1f}% of mask)\n"
                     f"drag=paint  shift/right-drag=erase  [ ]=brush  0-4=animal  "
                     f"f=415/470  a=outlines  u=undo  c=clear  s=SAVE  q=quit", fontsize=10)
        fig.canvas.draw_idle()

    def stamp(x, y, erase):
        r = state["brush"]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        sel = (d2 <= r * r) & full
        if erase:
            mask[sel] = False
        else:
            mask[sel] = True

    def on_press(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        state["undo"].append(mask.copy())
        del state["undo"][:-30]
        state["painting"] = True
        state["erase"] = bool(ev.button == 3) or (ev.key == "shift")
        stamp(ev.xdata, ev.ydata, state["erase"])
        draw()

    def on_move(ev):
        if not state["painting"] or ev.inaxes is not ax or ev.xdata is None:
            return
        stamp(ev.xdata, ev.ydata, state["erase"])
        draw()

    def on_release(_ev):
        state["painting"] = False

    def on_scroll(ev):
        state["brush"] = int(np.clip(state["brush"] + (2 if ev.step > 0 else -2), 2, 80))
        draw()

    def on_key(ev):
        k = (ev.key or "").lower()
        if k in ("[", "]"):
            state["brush"] = int(np.clip(state["brush"] + (-2 if k == "[" else 2), 2, 80))
        elif k in [str(i) for i in range(len(order))]:
            state["key"] = order[int(k)]
        elif k == "f":
            state["chan"] = 1 - state["chan"]
        elif k == "a":
            state["outlines"] = not state["outlines"]
        elif k == "o":
            state["show_excluded"] = not state["show_excluded"]
        elif k == "u" and state["undo"]:
            mask[:] = state["undo"].pop()
        elif k == "c":
            state["undo"].append(mask.copy())
            mask[:] = False
        elif k == "s":
            p = save_mask(mask, a.out)
            print(f"\nSAVED {int(mask.sum()):,} px -> {p}")
            print(f"{'region':18s}{'painted':>9s}{'of region':>11s}{'fraction':>10s}")
            for nm, hit, tot, frac in region_breakdown(mask):
                if frac < 0.02:
                    continue
                print(f"  {nm:16s}{hit:>9,d}{tot:>11,d}{frac:>9.0%}")
            print("\nRegions above ~90% are candidates for a NAME-based exclusion "
                  "(`beta_maps.EXCLUDE_REGIONS`); partial ones are why the painted mask exists.")
        elif k == "q":
            plt.close(fig)
            return
        draw()

    for name, fn in (("button_press_event", on_press), ("motion_notify_event", on_move),
                     ("button_release_event", on_release), ("scroll_event", on_scroll),
                     ("key_press_event", on_key)):
        fig.canvas.mpl_connect(name, fn)

    print(__doc__)
    print(f"animals: {', '.join(f'{i}={k}' for i, k in enumerate(order))}")
    print(f"channels (press f): {', '.join(f'{c}={n}' for c, n in CHANNELS.items())}")
    draw()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

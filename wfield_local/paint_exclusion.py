"""Paint an exclusion mask by hand, PER ANIMAL, on the real fluorescence with CCF outlines.

    python -m wfield_local.paint_exclusion

WHY A PAINTER AND NOT A THRESHOLD. The fibre glue is the thing this exists for, and it cannot be
found automatically: three attempts failed on 2026-09-12, each for a different reason. Thresholding
DARK pixels found the anterolateral somatosensory band, because the whole left hemisphere is dimmer
than the right (a tilt/illumination gradient, not glue). The left/right MIRROR RATIO found the same
thing for the same reason. Thresholding locally BRIGHT pixels -- glue scatters light, so bright is
the right sign -- found MOs, RSPd and RSPagl, because the medial dorsal surface is flatter to the
camera and brighter everywhere. Brightness in this preparation is dominated by geometry.

The person who placed the fibre can see it in one glance. This turns that into a mask.

ONE MASK PER ANIMAL, because the glue is per animal (Priya, 2026-09-12: "can i not paint a
different mask for each animal?"). A single shared mask would either miss one animal's occlusion or
throw away cortex that is perfectly good in the other three. The `all` mask is separate and applies
to EVERY animal, for anything genuinely common.

AND A REGION LIST IS NOT ENOUGH EITHER. "L Vis and aud partial" is partial, and Allen regions are
all-or-nothing: excluding VISp_left to catch a corner of it costs 11,886 px, 6% of the mask. Saving
prints the Allen breakdown WITH THE FRACTION of each region covered, so a name-based exclusion
stays available if a painted shape turns out to land cleanly on a few areas.

CONTROLS
    drag                    paint             shift+drag / right-drag    erase
    [ ]  or  scroll         brush smaller / larger
    0                       the `all` mask (applies to every animal)
    1-4                     that animal's own mask
    f                       toggle 470 (functional) / 415 (isosbestic)
    p                       show / hide what you painted
    n                       show the OTHER masks faintly, for consistency
    a                       toggle CCF outlines
    o                       toggle the already-excluded regions (MOB) in red
    u                       undo (per mask)                   c    clear THIS mask
    s                       SAVE every mask + the union, and print the Allen breakdown
    q                       quit without saving

THE MASKS ARE ADDITIVE TO `beta_maps.EXCLUDE_REGIONS`: the bulbs stay excluded by name, and these
cover what a name cannot express.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

#: Stem for the saved masks. One file per animal, plus `_all` and `_union`.
MASK_STEM = "exclusion_mask_painted"

#: Channel index -> what it is. `hemo_variants.FUNC` is the FUNCTIONAL channel's index in the
#: interleaved stream and it is 1, so channel 0 is the isosbestic 415. Named rather than assumed,
#: because getting it backwards would put the calcium-free image up as "the data" and nothing about
#: the picture would say so.
CHANNELS = {1: "470 (functional)", 0: "415 (isosbestic)"}


def _mean_fluorescence(animals=None, max_sessions=10):
    """``{(animal, channel): (540, 640)}`` -- motion-corrected mean frames, per animal per channel.

    BOTH CHANNELS, because they fail differently. 470 carries calcium and haemodynamics; 415 carries
    haemodynamics and whatever is optically in the way -- so a surface occlusion appears in the
    ISOSBESTIC image without any of the activity that complicates the functional one.

    NORMALISED PER SESSION by its own in-mask median before averaging, so one bright day does not
    set the picture; the absolute level is meaningless here and the shape is what is looked at.
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
            out[("all", c)] = np.nanmean(same, axis=0)
    return out


def region_breakdown(mask):
    """``[(region, px_in_mask, px_in_region, fraction)]`` for what a painted mask covers.

    THE FRACTION IS THE POINT. A shape clipping 8% of VISp is a different fact from one covering
    95% of it: the first says a name-based exclusion would be far too blunt, the second says just
    exclude the region.

    THE DENOMINATOR IS `allen_mask`, NOT `brain_mask`, and that distinction is load-bearing. This
    used to read ``brain_mask() | excluded_mask()``, which was right while only named regions were
    subtracted and broke silently once the painted mask was wired into `brain_mask`: the 40,663
    painted pixels were no longer IN the denominator, every region scored zero, and the function
    returned an empty list. It raises now rather than returning nothing, because "the glue covers
    nothing" and "the tool is broken" had looked identical.
    """
    from wfield_local import beta_maps as bm

    atlas, names = bm._atlas_names()
    if atlas is None:
        return []
    full = bm.allen_mask()
    if full is None:
        return []
    n_in = int((np.asarray(mask, bool) & full).sum())
    if int(np.asarray(mask, bool).sum()) and not n_in:
        raise ValueError("region_breakdown: none of the mask falls inside the Allen brain mask -- "
                         "the mask is on a different grid, or the denominator is wrong")
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


def mask_dir(out_dir=None):
    from wfield_local.paths import PathResolver
    return Path(out_dir) if out_dir else Path(PathResolver().root("labcams")) / "grant_figures"


def mask_path(key, out_dir=None):
    """Where one mask lands. `key` is an animal id, ``"all"`` or ``"union"``."""
    return mask_dir(out_dir) / f"{MASK_STEM}_{key}.npy"


def load_masks(out_dir=None):
    """``{key: mask}`` for whatever has been painted, or ``{}``. The union is not loaded back."""
    out = {}
    d = mask_dir(out_dir)
    if not d.exists():
        return out
    for p in sorted(d.glob(f"{MASK_STEM}_*.npy")):
        key = p.stem[len(MASK_STEM) + 1:]
        if key == "union":
            continue
        try:
            out[key] = np.load(p).astype(bool)
        except Exception:                                              # noqa: BLE001
            continue
    return out


def save_masks(masks, out_dir=None):
    """Write each non-empty mask plus the union. Returns the paths written.

    THE UNION IS WRITTEN TOO because the pooled figures need ONE mask: a pixel occluded in any
    animal cannot contribute to a cross-animal average. Per-animal files stay available for
    per-animal panels, where each row should lose only its own animal's occlusion.
    """
    from wfield_local.writeguard import assert_writable

    written = []
    d = mask_dir(out_dir)
    assert_writable(d)
    d.mkdir(parents=True, exist_ok=True)
    union = None
    for key, m in masks.items():
        m = np.asarray(m, bool)
        union = m.copy() if union is None else (union | m)
        if not m.any():
            continue
        p = mask_path(key, out_dir)
        np.save(p, m)
        written.append(p)
    if union is not None and union.any():
        p = mask_path("union", out_dir)
        np.save(p, union)
        written.append(p)
    return written


def main(argv=None) -> int:
    from wfield_local.console import use_utf8_stdout
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="directory to save into")
    ap.add_argument("--brush", type=int, default=14)
    a = ap.parse_args(argv)

    # EVERY wfield_local IMPORT FIRST, AND ONLY THEN THE BACKEND. `grant_figures`,
    # `locanmf_cue_lick_analysis` and ~10 others call `matplotlib.use("Agg")` AT IMPORT TIME, so
    # selecting an interactive backend before importing them is silently undone -- the window never
    # appears and matplotlib says only "FigureCanvasAgg is non-interactive" at `show()`.
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
    present = [x for x in dict.fromkeys(k for k, _c in imgs) if x != "all"]
    order = ["all"] + present

    masks = {k: np.zeros(bm.MAP_SHAPE, bool) for k in order}
    for k, m in load_masks(a.out).items():
        if k in masks and m.shape == bm.MAP_SHAPE:
            masks[k] = m
            print(f"loaded {int(m.sum()):,} px for {k}")

    st = {"key": order[0], "chan": 1 if ("all", 1) in imgs else 0, "brush": int(a.brush),
          "painting": False, "erase": False, "outlines": True, "show_excluded": True,
          "show_paint": True, "show_others": False, "undo": {k: [] for k in order}}

    for backend in ("TkAgg", "QtAgg", "Qt5Agg", "wxAgg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:                                              # noqa: BLE001
            continue
    if not matplotlib.get_backend().lower().startswith(("tk", "qt", "wx")):
        print(f"backend is {matplotlib.get_backend()!r}, not interactive -- cannot paint.")
        return 2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    print(f"backend: {matplotlib.get_backend()}")

    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    try:
        fig.canvas.manager.set_window_title("paint exclusion -- 0-4 mask  f channel  s save  q quit")
    except Exception:                                                  # noqa: BLE001
        pass
    ax.set_axis_off()

    # ARTISTS BUILT ONCE AND UPDATED IN PLACE. Rebuilding contours on every mouse-move made the
    # brush lag badly enough to be unusable; an RGBA overlay whose data is swapped is instant.
    base = ax.imshow(np.zeros(bm.MAP_SHAPE), cmap="gray")
    over = ax.imshow(np.zeros(bm.MAP_SHAPE + (4,)), interpolation="nearest")
    edge_art = None
    if edges is not None:
        e = np.zeros(bm.MAP_SHAPE + (4,))
        e[np.asarray(edges, bool)] = [0.0, 0.82, 1.0, 0.85]
        edge_art = ax.imshow(e, interpolation="nearest")
    cursor = Circle((0, 0), st["brush"], fill=False, ec="#00ff66", lw=1.6, zorder=9)
    ax.add_patch(cursor)
    yy, xx = np.mgrid[0:bm.MAP_SHAPE[0], 0:bm.MAP_SHAPE[1]]

    def overlay():
        rgba = np.zeros(bm.MAP_SHAPE + (4,))
        if st["show_others"]:
            oth = np.zeros(bm.MAP_SHAPE, bool)
            for k, m in masks.items():
                if k != st["key"]:
                    oth |= m
            rgba[oth & ~masks[st["key"]]] = [0.2, 0.6, 1.0, 0.25]
        if st["show_excluded"] and already.any():
            rgba[already] = [0.84, 0.15, 0.15, 0.45]
        if st["show_paint"]:
            rgba[masks[st["key"]]] = [1.0, 0.80, 0.0, 0.50]
        return rgba

    def redraw(full_refresh=True):
        if full_refresh:
            v = imgs.get((st["key"], st["chan"]), imgs.get((st["key"], 1 - st["chan"])))
            v = np.asarray(v, float).copy()
            v[~full] = np.nan
            base.set_data(v)
            base.set_clim(float(np.nanpercentile(v, 2)), float(np.nanpercentile(v, 99)))
            if edge_art is not None:
                edge_art.set_visible(st["outlines"])
        over.set_data(overlay())
        cursor.set_radius(st["brush"])
        n = int(masks[st["key"]].sum())
        tot = sum(int(m.sum()) for m in masks.values())
        ax.set_title(
            f"MASK: {st['key']}    ch{st['chan']} = {CHANNELS[st['chan']]}    brush {st['brush']}px"
            f"\n{n:,} px on this mask ({100 * n / max(full.sum(), 1):.1f}%)   |   "
            f"{tot:,} px painted across all masks"
            f"\ndrag paint · shift/right erase · [ ] brush · 0=all 1-4=animal · f channel · "
            f"p hide · n others · u undo · c clear · s SAVE · q quit", fontsize=9.5)
        fig.canvas.draw_idle()

    def stamp(x, y, erase):
        r = st["brush"]
        sel = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) & full
        masks[st["key"]][sel] = not erase

    def on_press(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        st["undo"][st["key"]].append(masks[st["key"]].copy())
        del st["undo"][st["key"]][:-40]
        st["painting"] = True
        st["erase"] = bool(ev.button == 3) or (ev.key == "shift")
        stamp(ev.xdata, ev.ydata, st["erase"])
        redraw(False)

    def on_move(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        cursor.set_center((ev.xdata, ev.ydata))
        if st["painting"]:
            stamp(ev.xdata, ev.ydata, st["erase"])
            over.set_data(overlay())
        fig.canvas.draw_idle()

    def on_release(_ev):
        st["painting"] = False
        redraw(False)

    def on_scroll(ev):
        step = 2 if getattr(ev, "step", 1) > 0 else -2
        st["brush"] = int(np.clip(st["brush"] + step, 1, 90))
        redraw(False)

    def on_key(ev):
        k = (ev.key or "").lower()
        full_refresh = False
        if k == "[":
            st["brush"] = max(1, st["brush"] - 2)
        elif k == "]":
            st["brush"] = min(90, st["brush"] + 2)
        elif k.isdigit() and int(k) < len(order):
            st["key"] = order[int(k)]
            full_refresh = True
        elif k == "f":
            st["chan"] = 1 - st["chan"]
            full_refresh = True
        elif k == "p":
            st["show_paint"] = not st["show_paint"]
        elif k == "n":
            st["show_others"] = not st["show_others"]
        elif k == "a":
            st["outlines"] = not st["outlines"]
            full_refresh = True
        elif k == "o":
            st["show_excluded"] = not st["show_excluded"]
        elif k == "u" and st["undo"][st["key"]]:
            masks[st["key"]] = st["undo"][st["key"]].pop()
        elif k == "c":
            st["undo"][st["key"]].append(masks[st["key"]].copy())
            masks[st["key"]][:] = False
        elif k == "s":
            paths = save_masks(masks, a.out)
            print(f"\nSAVED {len(paths)} file(s):")
            for p in paths:
                print(f"   {p}")
            for key in order:
                m = masks[key]
                if not m.any():
                    continue
                print(f"\n--- {key}: {int(m.sum()):,} px "
                      f"({100 * m.sum() / max(full.sum(), 1):.1f}% of mask) ---")
                print(f"  {'region':18s}{'painted':>9s}{'of region':>11s}{'fraction':>10s}")
                for nm, hit, tot_, frac in region_breakdown(m):
                    if frac < 0.02:
                        continue
                    print(f"  {nm:16s}{hit:>9,d}{tot_:>11,d}{frac:>9.0%}")
            print("\nRegions above ~90% are candidates for a NAME-based exclusion "
                  "(`beta_maps.EXCLUDE_REGIONS`); partial ones are why this painter exists.")
        elif k == "q":
            plt.close(fig)
            return
        redraw(full_refresh)

    for name, fn in (("button_press_event", on_press), ("motion_notify_event", on_move),
                     ("button_release_event", on_release), ("scroll_event", on_scroll),
                     ("key_press_event", on_key)):
        fig.canvas.mpl_connect(name, fn)

    print(__doc__)
    print(f"masks: {', '.join(f'{i}={k}' for i, k in enumerate(order))}")
    print(f"channels (f): {', '.join(f'{c}={n}' for c, n in CHANNELS.items())}")
    redraw(True)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

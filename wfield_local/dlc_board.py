"""Generate a printable ChArUco calibration board, at exact physical scale, with its own spec on it.

The 2026-08-05 board resolves to **2.1 px per code cell** in the side views against cam4's 5.1. A
DICT_4X4 marker is 6 cells across and stops decoding below ~3, so its squares are FOUND there and
then rejected -- 58 rejected candidates in a single cam2 frame. That is a board too small to read,
not a board that was never presented, and no detector setting fixes it (``dlc_calibration``). It
carries 43+ markers, which is why: a dense board on one sheet of paper means small markers.

This trades marker COUNT for marker SIZE. Fewer, larger squares is the right trade for a rig whose
widest view sees the board across a whole-animal field: `cam1`/`cam4` see it close and can spare the
corners, `cam2`/`cam3` cannot spare the pixels.

**THE SPEC IS PRINTED ON THE BOARD.** squaresX/Y, square mm, marker mm and the dictionary are not
recoverable from video and are exactly what metric 3D reconstruction needs; a board whose geometry
has been lost is a board that can only produce a scale-free reconstruction. Printing it in the
margin means it cannot be separated from the thing it describes.

**A 100 mm RULER IS PRINTED TOO.** "Fit to page" and "shrink oversized pages" are on by default in
most print dialogs and would silently rescale the board, which does not change the images at all --
it changes the millimetres they imply, and every downstream distance would be wrong by that factor
with nothing to reveal it. Measure the ruler before using the board.

CLI::

    python -m wfield_local.dlc_board                   # the configured board -> PDF + PNG
    python -m wfield_local.dlc_board --paper a4        # the smaller variant
    python -m wfield_local.dlc_board --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wfield_local import config
from wfield_local.writeguard import assert_writable

#: Print DPI for the rasterised board. 600 keeps the marker edges crisp at these sizes -- a soft
#: edge costs corner-localisation accuracy, which is the one thing a calibration board is for.
DPI = 600

#: Paper sizes in mm (portrait).
PAPER = {"a4": (210.0, 297.0), "a3": (297.0, 420.0), "letter": (215.9, 279.4),
         "tabloid": (279.4, 431.8)}

#: The card is taped to a 30 mm cage plate on a 1/2 inch post (Priya, 2026-09-08).
POST_MM = 12.7
CAGE_PLATE_MM = 38.1          # Thorlabs 30 mm cage plate, 1.5 in square

#: Pattern-free strip at the foot of the board for the tape/clamp, one cage plate deep. An occluded
#: marker is not a smaller board -- it is a missing ID, and the ChArUco corners that depended on it
#: go with it, so the mount needs somewhere to grip that is not pattern.
MOUNT_TAB_MM = CAGE_PLATE_MM


def board_spec(paper: str | None = None) -> dict:
    """The configured board, or the built-in default for ``paper``.

    SIZED FROM THE 08-05 RECORDING, not from paper. In cam2 the printed pattern spans ~145 px while
    the card it is taped to -- a 30 mm cage plate, 38.1 mm -- spans ~390 px, so the current markers
    are about **1.2 mm** and resolve to 2.1 px per code cell. An 8 mm marker is ~7x that, which puts
    cam2 near 14 px/cell at the same working distance: comfortable margin over the ~3 needed,
    without the board growing past the rig. A full-page board would be ~20x and cam4, whose field is
    the snout alone, would see two squares of it.
    """
    cfg = ((config.defaults().get("dlc") or {}).get("board") or {})
    if paper is None and cfg:
        return {"squares_x": int(cfg["squares_x"]), "squares_y": int(cfg["squares_y"]),
                "square_mm": float(cfg["square_mm"]), "marker_mm": float(cfg["marker_mm"]),
                "dictionary": str(cfg.get("dictionary", "DICT_4X4_50")),
                "paper": str(cfg.get("paper", "a3"))}
    p = (paper or "a3").lower()
    if p in ("a3", "tabloid"):
        # The bigger option, for working further from the rig: 154x112 mm, 44 markers.
        return {"squares_x": 11, "squares_y": 8, "square_mm": 14.0, "marker_mm": 10.5,
                "dictionary": "DICT_4X4_50", "paper": p}
    # DEFAULT: 94.5x73.5 mm -- about 2.5x the cage plate it mounts on, 31 markers, 8 mm markers.
    return {"squares_x": 9, "squares_y": 7, "square_mm": 10.5, "marker_mm": 8.0,
            "dictionary": "DICT_4X4_50", "paper": p}


def check_fits(spec: dict, margin_mm: float = 12.0) -> tuple[float, float]:
    """Board size in mm; raises if it will not fit the paper with ``margin_mm`` to spare.

    A board that overflows gets silently shrunk by the print dialog, which is the one failure this
    module exists to prevent -- so it is refused here rather than discovered on the ruler.
    """
    w = spec["squares_x"] * spec["square_mm"]
    h = spec["squares_y"] * spec["square_mm"]
    pw, ph = PAPER[spec["paper"]]
    # The mount tab and the ruler live on the page too, so the board cannot have the whole sheet.
    if w > pw - 2 * margin_mm or h > ph - 2 * margin_mm - MOUNT_TAB_MM - 26.0:
        raise ValueError(
            f"{spec['squares_x']}x{spec['squares_y']} at {spec['square_mm']} mm is {w:.0f}x{h:.0f} mm "
            f"and will not fit {spec['paper'].upper()} ({pw:.0f}x{ph:.0f} mm) with {margin_mm:.0f} mm "
            f"margins. Reduce the square count or the square size."
        )
    if spec["marker_mm"] >= spec["square_mm"]:
        raise ValueError("marker_mm must be smaller than square_mm -- the marker sits inside the "
                         "square with a white quiet zone around it.")
    return w, h


def n_markers(spec: dict) -> int:
    return (spec["squares_x"] * spec["squares_y"]) // 2


def px_per_bit_at(spec: dict, px_per_mm: float, bits_across: int = 6) -> float:
    """Code-cell size in pixels for a camera imaging at ``px_per_mm`` -- the decode budget.

    The number the whole redesign turns on. ``dlc_calibration`` measures it from a recording; this
    predicts it from a board, so a board can be sized before it is printed rather than after.
    """
    return round(spec["marker_mm"] * px_per_mm / bits_across, 2)


def render(spec: dict, out_dir, stem: str | None = None) -> list[Path]:
    """Write the board as PDF (exact scale) and PNG. Returns the paths written."""
    import cv2
    import matplotlib
    from cv2 import aruco
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w_mm, h_mm = check_fits(spec)
    d = aruco.getPredefinedDictionary(getattr(aruco, spec["dictionary"]))
    board = aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]),
                               spec["square_mm"], spec["marker_mm"], d)
    def px(mm: float) -> int:
        return round(float(mm) / 25.4 * DPI)

    img = board.generateImage((px(w_mm), px(h_mm)), marginSize=0, borderBits=1)

    pw, ph = PAPER[spec["paper"]]
    fig = plt.figure(figsize=(pw / 25.4, ph / 25.4), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, pw); ax.set_ylim(0, ph)
    ax.invert_yaxis(); ax.axis("off")
    x0, y0 = (pw - w_mm) / 2, 26.0
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(x0, x0 + w_mm, y0 + h_mm, y0),
              interpolation="none", zorder=2)

    # Mount tab: pattern-free, one cage plate deep, so tape or a clamp never covers a marker.
    ty = y0 + h_mm
    ax.add_patch(plt.Rectangle((x0, ty), w_mm, MOUNT_TAB_MM, facecolor="white",
                               edgecolor="0.75", lw=0.6, ls=(0, (4, 3)), zorder=3))
    ax.text(x0 + w_mm / 2, ty + MOUNT_TAB_MM / 2,
            f"MOUNT TAB - tape to the 30 mm cage plate here ({CAGE_PLATE_MM:.1f} mm)",
            ha="center", va="center", fontsize=6.5, color="0.45", family="monospace", zorder=4)

    caption = (f"ChArUco  {spec['squares_x']}x{spec['squares_y']} squares  |  "
               f"square {spec['square_mm']:.1f} mm  |  marker {spec['marker_mm']:.1f} mm  |  "
               f"{spec['dictionary']}  |  {n_markers(spec)} markers  |  board "
               f"{w_mm:.0f}x{h_mm:.0f} mm")
    ax.text(pw / 2, ty + MOUNT_TAB_MM + 8.0, caption, ha="center", va="center", fontsize=7, family="monospace",
            zorder=3)

    # 100 mm ruler: the only way to catch a print dialog that rescaled the page.
    ry = 12.0
    rx = (pw - 100.0) / 2
    ax.plot([rx, rx + 100], [ry, ry], color="black", lw=1.2, zorder=3)
    for i in range(11):
        t = 3.5 if i % 5 == 0 else 2.0
        ax.plot([rx + i * 10, rx + i * 10], [ry - t, ry + t], color="black", lw=1.0, zorder=3)
    ax.text(pw / 2, ry + 7.0, "MEASURE ME: this line is exactly 100 mm. If it is not, the print was "
                              "scaled and every distance from this board is wrong.",
            ha="center", va="center", fontsize=6.5, family="monospace", zorder=3)

    out_dir = Path(out_dir)
    assert_writable(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or (f"charuco_{spec['squares_x']}x{spec['squares_y']}_"
                    f"{spec['square_mm']:.0f}mm_{spec['paper']}")
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI)
        paths.append(p)
    plt.close(fig)
    cv2.imwrite(str(out_dir / f"{stem}_boardonly.png"), img)
    paths.append(out_dir / f"{stem}_boardonly.png")
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", default=None, choices=sorted(PAPER),
                    help="built-in variant for this paper size (default: the configured board)")
    ap.add_argument("--out", default=".", help="output directory")
    args = ap.parse_args(argv)
    spec = board_spec(args.paper)
    w, h = check_fits(spec)
    print(f"[dlc_board] {spec['squares_x']}x{spec['squares_y']} ChArUco, square "
          f"{spec['square_mm']:.0f} mm, marker {spec['marker_mm']:.0f} mm, {spec['dictionary']}, "
          f"{n_markers(spec)} markers -> {w:.0f}x{h:.0f} mm on {spec['paper'].upper()}", flush=True)
    for p in render(spec, args.out):
        print(f"[dlc_board] -> {p}", flush=True)
    print("[dlc_board] print at 100% / 'actual size', then MEASURE THE 100 mm RULER.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

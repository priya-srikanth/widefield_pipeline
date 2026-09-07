"""One PowerPoint per animal of the annotated example clips: six spout positions to a slide.

One slide per (session, trial class, example index); six cells, one per position; slides in
CHRONOLOGICAL order so pre-stroke leads.

THE DECK CURATES, THE LIBRARY DOES NOT. `behavior_clips` cuts five per cell to disk so any later
question can be answered without re-encoding video. This shows fewer, because a hit looks like a hit
and a stopped trial looks like a stopped trial -- the second example makes either point -- while a
`working` miss at the far contralateral or far middle spout IS the deficit and earns a run.

EVERY POSITION KEEPS ITS SLOT. The grid is indexed by position, not by what a slide happens to show,
so far middle and far contra sit bottom-centre and bottom-right on every slide. Packing the shown
clips from the top-left would move a position between slides and the reader would compare different
spouts in the same place without noticing. An unused slot says "-- none --" rather than going blank:
a position with no working misses is a position the animal was not failing at, which is a result.

CELLS ARE SQUARE BECAUSE THE CLIPS ARE. cam4 is 680x680; a 4.05 x 2.75in placeholder stretched every
frame by 1.47x horizontally, and on a mouse's face that reads as anatomy rather than as layout.

EMBEDDED AT 240 px. python-pptx embeds the bytes, so the deck carries every clip it shows: at the
library's 480 px a per-animal deck runs ~2.1 GB, at 240 px ~210 MB. Re-encoding for the deck is the
only reason to re-encode at all -- the on-disk clips stay full size.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from wfield_local import config
from wfield_local.behavior_clips import CAM, POST_S, PRE_S, out_root
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

DECK_PX = 240
POS_ORDER = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
NICE = {"close_L": "near ipsi", "close_center": "near middle", "close_R": "near contra",
        "far_L": "far ipsi", "far_center": "far middle", "far_R": "far contra"}
FOCUS_POSITIONS = ("far_center", "far_R")
DECK_CAP = {"success": 2, "stopped": 2}
WORKING_FOCUS, WORKING_OTHER = 5, 2
POSTER_FRAME = 145                    #: ~+0.16 s from cue -- spout up, response beginning


def deck_cap(position, category):
    """How many examples of one (position, category) the DECK shows."""
    if category == "working":
        return WORKING_FOCUS if position in FOCUS_POSITIONS else WORKING_OTHER
    return DECK_CAP.get(category, 2)


def parse(p):
    """``(position, category, take, available, trial_id)`` from a clip filename, or None."""
    m = re.match(r"(.+?)_(success|working|stopped)_(\d+)of(\d+)_trial(\d+)", Path(p).stem)
    if not m:
        return None
    return (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)))


def _shrink(src, dst):
    """Re-encode one clip to DECK_PX square and write a poster frame beside it."""
    import cv2

    cap = cv2.VideoCapture(str(src))
    vw, poster, last, i = None, None, None, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        f2 = cv2.resize(fr, (DECK_PX, DECK_PX), interpolation=cv2.INTER_AREA)
        if vw is None:
            vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                                 (DECK_PX, DECK_PX))
        vw.write(f2)
        last = f2
        if i == POSTER_FRAME:
            poster = dst.with_suffix(".png")
            cv2.imwrite(str(poster), f2)
        i += 1
    cap.release()
    if vw is not None:
        vw.release()
    if poster is None and last is not None:
        poster = dst.with_suffix(".png")
        cv2.imwrite(str(poster), last)
    return poster


def _sessions_for(animal_dir):
    """[(date, epoch, dir), ...] sorted by DATE, so pre-stroke leads by fact not by alphabet."""
    out = []
    for epoch_dir in animal_dir.iterdir():
        if not epoch_dir.is_dir() or epoch_dir.name.startswith("."):
            continue
        for date_dir in epoch_dir.iterdir():
            if date_dir.is_dir() and not date_dir.name.startswith("."):
                out.append((date_dir.name, epoch_dir.name, date_dir))
    return sorted(out, key=lambda t: t[0])


def build(animal, rv=None, dest=None, tmp=None, dates=None):
    """One deck for ``animal``. Returns its path, or None if there are no clips."""
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    rv = rv or PathResolver()
    root = out_root(rv) / animal
    if not root.is_dir():
        print("[clip_deck] %s: no clips" % animal, flush=True)
        return None
    tmp = Path(tmp) if tmp else Path(rv.root("behavior_out")) / "_clip_deck_tmp" / animal
    tmp.mkdir(parents=True, exist_ok=True)
    dest = Path(dest) if dest else out_root(rv) / ("%s_example_clips.pptx" % animal)

    navy, grey, faint = RGBColor(0x1F, 0x35, 0x64), RGBColor(0x66, 0x66, 0x66), \
        RGBColor(0xB0, 0xB0, 0xB0)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    cell = Inches(2.62)
    gap_x, gap_y, lab_h = Inches(0.16), Inches(0.30), Inches(0.28)
    top0 = Inches(1.22)
    left0 = (Inches(13.333) - (cell * 3 + gap_x * 2)) / 2

    made = 0
    sessions = [s for s in _sessions_for(root) if not dates or s[0] in set(dates)]
    for date, epoch, date_dir in sessions:
        clips = {}
        for f in date_dir.glob("*.avi"):
            got = parse(f)
            if got:
                clips.setdefault((got[1], got[0]), []).append((got[4], f))
        for cat in ("success", "working", "stopped"):
            depth = max([min(len(v), deck_cap(p, c)) for (c, p), v in clips.items() if c == cat]
                        or [0])
            for k in range(depth):
                shown = {p: sorted(clips[(cat, p)])[k] for p in POS_ORDER
                         if len(clips.get((cat, p), [])) > k and k < deck_cap(p, cat)}
                if not shown:
                    continue
                s = prs.slides.add_slide(blank)
                tf = s.shapes.add_textbox(Inches(0.4), Inches(0.13), Inches(12.5),
                                          Inches(0.72)).text_frame
                r = tf.paragraphs[0].add_run()
                r.text = "%s  %s  %s  |  %s  |  example %d of %d" % (
                    animal, date, epoch.upper(), cat.upper(), k + 1, depth)
                r.font.size, r.font.bold, r.font.color.rgb = Pt(21), True, navy
                r2 = tf.add_paragraph().add_run()
                r2.text = ("%.0f s pre-cue to %.1f s post-cue at 0.25x; ENL / Cue / Response "
                           "labelled per frame. 'N of M' = clips cut of trials available."
                           % (PRE_S, POST_S))
                r2.font.size, r2.font.color.rgb = Pt(10.5), grey
                for i, pos in enumerate(POS_ORDER):
                    col, row = i % 3, i // 3
                    left = left0 + col * (cell + gap_x)
                    top = top0 + row * (cell + lab_h + gap_y)
                    lab = s.shapes.add_textbox(left, top + cell, cell, lab_h).text_frame
                    lr = lab.paragraphs[0].add_run()
                    if pos not in shown:
                        lr.text = "%s (%s)  -- none --" % (NICE[pos], pos)
                        lr.font.size, lr.font.color.rgb = Pt(10), faint
                        continue
                    tid, f = shown[pos]
                    # UNIQUE per slide: a name reused across example index would leave every slide
                    # embedding whichever version was written last.
                    small = tmp / ("%s_%s_%s_%d_t%d.avi" % (date, cat, pos, k, tid))
                    poster = _shrink(f, small)
                    s.shapes.add_movie(str(small), left, top, cell, cell,
                                       poster_frame_image=str(poster) if poster else None,
                                       mime_type="video/x-msvideo")
                    n = parse(f)
                    lr.text = "%s (%s)  %d of %d" % (NICE[pos], pos, n[2], n[3])
                    lr.font.size, lr.font.color.rgb = Pt(10), grey
                    made += 1
    if not made:
        print("[clip_deck] %s: nothing to place" % animal, flush=True)
        return None
    assert_writable(dest.parent)
    prs.save(str(dest))
    mb = dest.stat().st_size / 1e6
    print("[clip_deck] %s: %d sessions, %d slides, %d clips, %.0f MB -> %s"
          % (animal, len(sessions), len(prs.slides._sldIdLst), made, mb, dest), flush=True)
    return dest


def run(rv=None, animals=None, dates=None):
    """A deck per animal. Returns the paths written."""
    rv = rv or PathResolver()
    want = config.normalize_animals(animals) or sorted(config.animals())
    return [p for p in (build(a, rv=rv, dates=dates) for a in want) if p]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="ANIMAL", help="restrict to these animals")
    ap.add_argument("--dates", nargs="+", metavar="YYYYMMDD", help="restrict to these dates")
    ap.add_argument("--machine", default=None)
    a = ap.parse_args(argv)
    run(PathResolver(machine=a.machine), animals=a.only, dates=a.dates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

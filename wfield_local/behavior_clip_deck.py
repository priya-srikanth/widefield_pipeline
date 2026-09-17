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

EMBEDDED AT 240 px, AS H.264/MP4. python-pptx embeds the bytes, so the deck carries every clip it
shows: at the library's 480 px a per-animal deck runs ~2.1 GB, at 240 px ~210 MB. Re-encoding for
the deck is the only reason to re-encode at all -- the on-disk clips stay full size.

THE CODEC IS NOT A DETAIL. Until 2026-09-17 this wrote MPEG-4 Part 2 into a `.avi` and declared it
`video/x-msvideo`; the bytes were embedded correctly and PowerPoint could not decode a single one of
them, so every deck ever built showed "cannot play media". H.264 + yuv420p + MP4 is the combination
PowerPoint plays on both Windows and Mac, and all three parts are load-bearing.
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



def _absence_note(category):
    """Why a class is empty, in the terms that make it a RESULT rather than a gap."""
    return {
        "success": ("No trials were hit in this session at any position. Every trial was either a "
                    "miss while still working or came after the animal stopped."),
        "working": ("The animal made NO engaged misses this session: every trial was either hit, "
                    "or came after it had stopped responding at the reference positions. On a "
                    "recovered animal this is the result -- PS92 09-04 hit 378 of 378."),
        "stopped": ("The animal never disengaged this session: the engagement gate marked no "
                    "trials, so every miss here was made while still working."),
    }.get(category, "No trials met the criteria for this class in this session.")


def parse(p):
    """``(position, category, take, available, trial_id)`` from a clip filename, or None."""
    m = re.match(r"(.+?)_(success|working|stopped)_(\d+)of(\d+)_trial(\d+)", Path(p).stem)
    if not m:
        return None
    return (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)))


def _ffmpeg() -> str | None:
    """Path to an ffmpeg binary, or None.

    Preferred over OpenCV's writer because OpenCV is built here WITHOUT an H.264 encoder -- `avc1`,
    `H264` and `X264` all fail to open, leaving `mp4v` (MPEG-4 Part 2) as the only option, and that
    is precisely the codec PowerPoint will not decode.
    """
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:                                   # ships with imageio-ffmpeg, which the dlc env has
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _shrink(src, dst):
    """Re-encode one clip to DECK_PX square and write a poster frame beside it.

    H.264 IN AN MP4, NOT MPEG-4 PART 2 IN AN AVI. The deck previously wrote `mp4v` into a `.avi` and
    declared it `video/x-msvideo`; PowerPoint ships no decoder for that pairing and every clip in
    every deck failed with "cannot play media" (Priya, 2026-09-17). The clips were EMBEDDED and
    intact the whole time -- python-pptx carries the bytes -- so this was never a broken link, only
    an unplayable stream.

    `-pix_fmt yuv420p` matters as much as the codec: H.264 in yuv444p is valid and PowerPoint still
    refuses it. `-movflags +faststart` puts the index first so playback can begin without reading
    the whole file.

    Falls back to OpenCV's `mp4v` when no ffmpeg is present, and SAYS SO, because a deck that
    silently reverts to the unplayable encoding is the failure this function exists to end.
    """
    import subprocess

    import cv2

    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(fr, (DECK_PX, DECK_PX), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        return None

    poster_frame = frames[POSTER_FRAME] if len(frames) > POSTER_FRAME else frames[-1]
    poster = dst.with_suffix(".png")
    cv2.imwrite(str(poster), poster_frame)

    exe = _ffmpeg()
    if exe is None:
        print("[clip_deck] NO FFMPEG -- falling back to mp4v, which PowerPoint cannot play",
              flush=True)
        vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (DECK_PX, DECK_PX))
        for fr in frames:
            vw.write(fr)
        vw.release()
        return poster

    proc = subprocess.run(
        [exe, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{DECK_PX}x{DECK_PX}", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
        input=b"".join(fr.tobytes() for fr in frames),
        capture_output=True,
    )
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed on {src}: {proc.stderr.decode()[:400]}")
    return poster


def _sessions_for(animal_dir):
    """[(date, epoch, dir), ...] sorted by DATE, so pre-stroke leads by fact not by alphabet."""
    from wfield_local import epochs

    out = []
    for epoch_dir in animal_dir.iterdir():
        if not epoch_dir.is_dir() or epoch_dir.name.startswith("."):
            continue
        for date_dir in epoch_dir.iterdir():
            if date_dir.is_dir() and not date_dir.name.startswith("."):
                # DERIVED, not read off the folder: `chronic_from` is recomputed from
                # behaviour each run, so a folder cut last week can carry an epoch the
                # animal has since left. `refile_stale_epochs` keeps storage in step;
                # this makes the LABEL right even if it has not run yet.
                lab = epochs.epoch_of("%s_%s" % (animal_dir.name, date_dir.name[4:]))
                out.append((date_dir.name, lab or epoch_dir.name, date_dir))
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

    made = made_empty = 0
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
            if depth == 0:
                # AN EMPTY CLASS GETS A SLIDE OF ITS OWN. Skipping it silently is what made PS92
                # 09-04 look like a rendering gap: the animal hit 378 of 378 trials, so there were
                # no working and no stopped trials to show, and "no engaged misses at all" is the
                # result rather than the absence of one. Priya, 2026-09-07.
                s = prs.slides.add_slide(blank)
                tf = s.shapes.add_textbox(Inches(0.4), Inches(0.13), Inches(12.5),
                                          Inches(0.9)).text_frame
                r = tf.paragraphs[0].add_run()
                r.text = "%s  %s  %s  |  %s  |  NO TRIALS IN THIS CLASS" % (
                    animal, date, epoch.upper(), cat.upper())
                r.font.size, r.font.bold, r.font.color.rgb = Pt(21), True, navy
                r2 = tf.add_paragraph().add_run()
                r2.text = _absence_note(cat)
                r2.font.size, r2.font.color.rgb = Pt(12), grey
                made_empty += 1
                continue
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
                    small = tmp / ("%s_%s_%s_%d_t%d.mp4" % (date, cat, pos, k, tid))
                    poster = _shrink(f, small)
                    s.shapes.add_movie(str(small), left, top, cell, cell,
                                       poster_frame_image=str(poster) if poster else None,
                                       mime_type="video/mp4")
                    n = parse(f)
                    lr.text = "%s (%s)  %d of %d" % (NICE[pos], pos, n[2], n[3])
                    lr.font.size, lr.font.color.rgb = Pt(10), grey
                    made += 1
    if not made and not made_empty:
        print("[clip_deck] %s: nothing to place" % animal, flush=True)
        return None
    assert_writable(dest.parent)
    prs.save(str(dest))
    mb = dest.stat().st_size / 1e6
    print("[clip_deck] %s: %d sessions, %d slides, %d clips, %d empty-class, %.0f MB -> %s"
          % (animal, len(sessions), len(prs.slides._sldIdLst), made, made_empty, mb,
             dest), flush=True)
    n_vid, problems = verify_playable(dest)
    if problems:
        print("[clip_deck] !! %d embedded clip(s) may NOT PLAY in PowerPoint:" % n_vid, flush=True)
        for p_ in problems:
            print("[clip_deck]    %s" % p_, flush=True)
    else:
        print("[clip_deck] verified %d embedded clip(s): H.264 / yuv420p" % n_vid, flush=True)
    return dest


def run(rv=None, animals=None, dates=None):
    """A deck per animal. Returns the paths written."""
    rv = rv or PathResolver()
    want = config.normalize_animals(animals) or sorted(config.animals())
    return [p for p in (build(a, rv=rv, dates=dates) for a in want) if p]


def verify_playable(deck: Path, sample: int = 3) -> tuple[int, list[str]]:
    """``(n_videos, [problems])`` -- check what a .pptx actually CONTAINS, not that it was written.

    THIS IS THE CHECK THAT WAS MISSING FOR THE LIFE OF THE MODULE. Every deck built before
    2026-09-17 embedded MPEG-4 Part 2 in an AVI, which PowerPoint cannot decode, and every existing
    check passed anyway: the slide count was right, the clip count was right, the poster frames
    showed the right mouse, the file size was plausible. What none of them asked was whether the
    bytes could be PLAYED, which is the only thing the deck is for.

    A .pptx is a zip; the embedded media sit under ``ppt/media/``. Probing a sample with ffmpeg is
    the cheapest question that would have caught it.
    """
    import subprocess
    import tempfile
    import zipfile

    problems: list[str] = []
    with zipfile.ZipFile(deck) as z:
        vids = [n for n in z.namelist()
                if n.startswith("ppt/media/") and Path(n).suffix.lower() in (".mp4", ".avi")]
        if not vids:
            return 0, ["no video embedded at all"]
        bad_ext = [n for n in vids if not n.lower().endswith(".mp4")]
        if bad_ext:
            problems.append(f"{len(bad_ext)} clip(s) are not .mp4, e.g. {Path(bad_ext[0]).name}")

        exe = _ffmpeg()
        if exe is None:
            problems.append("ffmpeg unavailable -- codec NOT verified")
            return len(vids), problems

        step = max(1, len(vids) // max(1, sample))
        for name in vids[::step][:sample]:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "probe.mp4"
                p.write_bytes(z.read(name))
                out = subprocess.run([exe, "-i", str(p)], capture_output=True).stderr.decode(
                    "utf-8", "replace")
            if "Video: h264" not in out:
                codec = next((ln.strip() for ln in out.splitlines() if "Video:" in ln), "?")
                problems.append(f"{Path(name).name} is not H.264 -- {codec}")
            elif "yuv420p" not in out:
                problems.append(f"{Path(name).name} is H.264 but not yuv420p -- PowerPoint "
                                f"refuses that")
    return len(vids), problems


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

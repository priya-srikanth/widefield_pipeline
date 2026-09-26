"""The labelling page for the BETWEEN-TRIAL spout frames — a short, self-contained job.

**WHY THESE FRAMES EXIST.** Every labelled frame in this project is trial-locked: `dlc.frames.phases`
samples ENL / Cue / early / late relative to a cue, and `lick_offsets_s` samples four points of one
protrusion. So the network has never been shown the spout **between** trials, which is where it moves
(Priya, 2026-09-26: *"the spout DURING TRIALS occupies 6 positions, but keep in mind that it moves
between trials"*). Measured consequence on a PS93 clip: the retracted spout is found at the right
place with likelihood **0.58**, just under the 0.6 cutoff, so it is silently dropped on 11% of
frames. The position is right and the confidence is not — the signature of a coverage gap rather
than an ambiguity, and the only fix is frames.

**THE FRAMES WERE CHOSEN BY THE NETWORK, NOT BY A TIME OFFSET.** Sampling the ITI at a fixed delay
mostly returns the spout sitting still somewhere the trial-locked frames already cover. Instead
every inter-trial gap that follows a POSITION CHANGE was scanned at ~21 Hz and two frames kept: the
largest jump in spout x (the move itself) and the lowest-confidence frame (where it cannot cope).
Stratified over all six positions so no position is missed.

**NO PREDICTIONS ARE DRAWN ON THESE PICTURES, DELIBERATELY.** `CORRECTION_GUIDE.html` shows the
network's guess because there the question is "who is right about a point you already placed". Here
the frames were selected precisely where the network is unreliable, so showing its guess would
anchor the labeller to the error being fixed. These are blank frames; they get labelled from
scratch.

**TWO FOLDERS, ON PURPOSE** (Priya, 2026-09-26: *"use only one or a few sessions if possible so she
doesn't have to open a ton of folders"*). One chronic, one subacute, different animals, all six
spout positions in each. The cost is that camera geometry varies a little session to session and
this samples two of fifteen; if the spout dropout persists after this round, widen to more sessions
rather than adding more frames to these two.

CLI::

    conda activate dlc
    python -m wfield_local.dlc_spout_guide            # writes SPOUT_FRAMES_GUIDE.html on the share
    python -m wfield_local.dlc_spout_guide --open
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import html
import io
from pathlib import Path

import pandas as pd

from wfield_local import dlc_project, dlc_train
from wfield_local.dlc_frames import staging_root
from wfield_local.dlc_review_guide import CSS, MAC_ROOT
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

GUIDE_NAME = "SPOUT_FRAMES_GUIDE.html"

#: The phases `find_iti_frames` writes into the manifest. Kept as a prefix match so a later round
#: can add `iti_<something>` without this page silently ignoring it.
ITI_PREFIX = "iti_"

WHY = {
    "iti_spout_move": ("the spout was MOVING", "#b45309"),
    "iti_low_conf": ("the network could not find it", "#7c2d12"),
}


def _thumb_b64(path: Path, width: int = 460) -> str | None:
    """Downsampled PNG as base64. Embedded rather than linked for the same reason the other guide
    embeds: the page has to work from her laptop with no share mounted."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        im = Image.open(path)
    except OSError:
        return None
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def iti_rows(rv=None) -> pd.DataFrame:
    """The between-trial frames, from `dlc_frames`' own manifest — not a hand-kept list."""
    man = staging_root(rv) / "frame_manifest.csv"
    if not man.is_file():
        raise SystemExit(f"no frame manifest at {man}")
    m = pd.read_csv(man, dtype=str)
    m = m[m["phase"].fillna("").str.startswith(ITI_PREFIX)].copy()
    if m.empty:
        raise SystemExit("no between-trial frames in the manifest — run the ITI selection first")
    m["frame"] = m["frame"].astype(int)
    return m.drop_duplicates(subset=["video_stem", "image"]).sort_values(["video_stem", "frame"])


def positions(live: Path, stem: str) -> dict[str, int]:
    """``{image: napari slider position}``, 0-based, in filename-sorted order.

    **This is the load-bearing part of the page.** These folders now hold ~24 already-labelled
    frames plus ~23 new ones, interleaved by filename, and napari opens a folder as one stack. Told
    only a filename she would have to scroll 47 frames and squint at each. `dlc_frames` zero-pads to
    seven digits so lexicographic order is numeric order, which makes the index stable — but it
    SHIFTS whenever frames are added, which is why this page is generated and not written once.
    """
    return {q.name: i for i, q in enumerate(sorted((live / "labeled-data" / stem).glob("img*.png")))}


def _card(r, live: Path, pos: dict[str, int]) -> str:
    img = _thumb_b64(live / "labeled-data" / r.video_stem / r.image)
    why, colour = WHY.get(r.phase, (r.phase, "#555"))
    slider = pos.get(r.image)
    where = (f'<b>slider {slider}</b> of {len(pos) - 1}' if slider is not None
             else '<b>not found in the folder</b>')
    pic = (f'<img alt="" src="data:image/png;base64,{img}">' if img
           else '<p class="legend">(thumbnail unavailable — open the frame in napari)</p>')
    return (f'<div class="card">'
            f'<h3 style="margin:.1em 0">{where} &nbsp;<code>{html.escape(r.image)}</code></h3>'
            f'<p class="legend" style="color:{colour};margin:.2em 0">'
            f'chosen because {html.escape(why)} &middot; spout position '
            f'<b>{html.escape(str(r.position))}</b></p>{pic}</div>')


def _session_block(stem: str, g: pd.DataFrame, live: Path) -> str:
    mac = f"{MAC_ROOT}/{live.name}/labeled-data/{stem}"
    pos = positions(live, stem)
    sliders = sorted(p for p in (pos.get(i) for i in g["image"]) if p is not None)
    return (f'<h2>{html.escape(stem)} &mdash; {len(g)} new frames</h2>'
            f'<p>In Finder press <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>G</kbd> and paste this, or use '
            f'napari&rsquo;s <i>File &rarr; Open Folder&hellip;</i>:</p>'
            f'<pre><code>{html.escape(mac)}</code></pre>'
            f'<p>The folder holds <b>{len(pos)}</b> frames. The ones below are new and blank; '
            f'everything else in there you have already done. '
            f'Slider positions to visit: <b>{", ".join(str(s) for s in sliders)}</b>.</p>'
            f'<div class="grid">{"".join(_card(r, live, pos) for r in g.itertuples())}</div>')


def build(rv=None, dest: Path | None = None) -> Path:
    rv = rv or PathResolver()
    live = dlc_project.project_dir(rv)
    rows = iti_rows(rv)
    today = _dt.date.today().isoformat()
    parts = dlc_train.parts()
    dest = dest or (live.parent / GUIDE_NAME)
    assert_writable(dest.parent)

    body = [f"""<div class="wrap">
<h1>New frames: the spout between trials</h1>
<p class="sub">cam4 &middot; {len(rows)} new frames in {rows.video_stem.nunique()} folders &middot;
generated {today}</p>

<p>These are <b>blank frames</b>, not corrections &mdash; nothing is labelled on them yet. They are
short: about {len(rows)} frames in two folders you have already worked in.</p>

<div class="card note">
<h3 style="margin-top:.2em">Why these frames</h3>
<p>Every frame you have labelled so far sits just before or just after a cue, or during a lick. That
means the network has never seen the spout <b>between</b> trials &mdash; which is exactly when it
moves. It currently finds the moved spout in the right place but is not confident enough to report
it, so those frames are thrown away. These frames are the fix.</p>
<p style="margin-bottom:.2em">They were picked by scanning the gaps between trials and keeping the
moment the spout <b>moved</b> and the moment the network was <b>least sure</b>. All six spout
positions are covered in each folder.</p>
</div>

<div class="card note">
<h3 style="margin-top:.2em">What to place</h3>
<p><b>The spout is the point of this round.</b> Place it on every frame where you can see it &mdash;
including when it is retracted, half out of frame, or blurred mid-move. A blurred spout still has a
tip; put the point where you would put it on a sharp frame.</p>
<p>Also place <b>nose</b> and <b>jaw</b> if you can see them, on the same landmarks you have been
using. <b>Tongue</b> will usually be in &mdash; leave it blank rather than guessing. A blank is not a
mistake: it tells the network nothing, which is the right answer when the part is not visible.</p>
<p style="margin-bottom:.2em">Project bodyparts are <code>{", ".join(parts)}</code>. There are no
whiskers and no eyes any more &mdash; if you see them listed, the project is stale, say so rather
than filling them in.</p>
</div>

<div class="card note">
<h3 style="margin-top:.2em">There are no predictions drawn on these pictures</h3>
<p style="margin-bottom:.2em">Unlike the correction page, these thumbnails are plain. That is
deliberate: these frames were chosen <i>because</i> the network is unreliable on them, so showing you
its guess would pull your label towards the very mistake this round is meant to fix. Label them the
way you would a fresh frame.</p>
</div>

<div class="card warn">
<h3 style="margin-top:.2em">Two things that go wrong at SAVE time</h3>
<p><b>Open the folder under <code>{html.escape(live.name)}</code>, never under
<code>_frame_staging</code> or <code>training</code>.</b> All three hold the same images under the
same names and look identical once open. Work in the wrong one and nothing complains until you
save &mdash; and in <code>training</code> the next retrain overwrites it anyway.</p>
<p style="margin-bottom:.2em"><b>Check <code>ls /Volumes</code> shows <code>Neurobio</code>, not
<code>Neurobio-1</code>.</b> A stale second mount is the usual reason a path on this page
&ldquo;cannot be found&rdquo;. Unmount both and remount once with Finder
<kbd>Cmd</kbd>+<kbd>K</kbd>, <code>smb://research.files.med.harvard.edu/Neurobio</code>.</p>
</div>

<p>Save with <i>File &rarr; Save Selected Layer(s)&hellip;</i> as you go. The
<code>CollectedData_Priya.h5</code> in each folder is the only record of the work.</p>
"""]
    for stem, g in rows.groupby("video_stem", sort=True):
        body.append(_session_block(str(stem), g, live))
    body.append("</div>")

    page = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Spout frames &mdash; cam4</title><style>{CSS}</style></head>"
            f"<body>{''.join(body)}</body></html>")
    dest.write_text(page, encoding="utf-8")
    print(f"[dlc_spout_guide] {len(rows)} frames across {rows.video_stem.nunique()} folders "
          f"-> {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    p = build(rv=PathResolver(machine=args.machine),
              dest=Path(args.dest) if args.dest else None)
    if args.open:
        import webbrowser
        webbrowser.open(p.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

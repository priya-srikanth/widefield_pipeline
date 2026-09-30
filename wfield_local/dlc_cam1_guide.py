"""The labelling page for camera 1 (the view from below) -- which folders, in which order, how to open them.

    conda activate dlc            # or locanmf; needs pandas + Pillow only
    python -m wfield_local.dlc_cam1_guide            # writes CAM1_GUIDE.html beside LABELLING_GUIDE.html
    python -m wfield_local.dlc_cam1_guide --open

WHY A SEPARATE PAGE. `LABELLING_GUIDE.html` is the permanent manual (install, napari, saving, what each
landmark means); this is the per-job worksheet for cam1, and it is GENERATED because the facts on it
change: which folders still have work in them, how many frames each holds, and which frames in the one
part-done folder are still blank. A hand-written copy of those would go stale the first time a folder is
finished.

WHAT IT KNOWS THAT THE MANUAL DID NOT (2026-09-29):
  * cam1 carries **nose, jaw, tongue, spout** (`dlc.cameras.cam1.bodyparts`). The nose was added on
    2026-09-12 and confirmed by Priya on 2026-09-29, with the caveat that the cam1 "nose" (ventral tip)
    and the cam4 "nose" (frontal tip) need not be the same 3-D point. The manual had said to leave it
    empty; it is corrected, and this page repeats the rule.
  * `cam1_2026-06-06T12_25_18` is not "mostly done": its 71 labelled frames carry jaw (71) and tongue
    (38) but **no nose and no spout on any frame**, and 15 of its 86 frames are blank. Its points file
    also still lists the retired twelve-part set (whiskers, eyes), so napari shows twelve names there.
  * The order is chosen so that a partial job is still balanced: each block of four folders covers all
    four animals and as many epochs as possible, so stopping halfway leaves no animal or epoch unlabelled.

TARGETS AND CONTEXT (2026-09-30). The eleven untouched folders now also hold CONTEXT frames
(`dlc_context_frames`): +-4 consecutive frames around spout-contact onset and contact end of each
lick target, there to be scrolled through when the tongue tip or jaw cannot be judged from one frame.
The manifest (`category == "context"`) is the only record of which is which, so this page reads it:
every count of "blank" / "to do" is over TARGETS only, and each folder lists its targets by slider
position. The rule the page teaches -- label a target completely; leave context blank or label it
completely, never partly -- is the one `dlc_train.drop_unlabelled` and the Lightning Pose export rely
on (DECISIONS.md, 2026-09-30).
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import html
import io
from pathlib import Path

import pandas as pd

from wfield_local import dlc_frames, dlc_project
from wfield_local.dlc_frames import staging_root
from wfield_local.dlc_review_guide import CSS, MAC_ROOT
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

GUIDE_NAME = "CAM1_GUIDE.html"
CAM = "cam1"
EPOCH_ORDER = ("pre", "acute", "subacute", "chronic")


def _thumb_b64(path: Path, width: int = 360) -> str | None:
    try:
        from PIL import Image
        im = Image.open(path)
    except (ImportError, OSError):
        return None
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def folder_status(folder: Path, parts: list[str], context: set[str] | None = None) -> dict:
    """Frames on disk, and per part how many frames carry it, from the folder's own CollectedData.

    ``context`` = image names the manifest marks as context. ``targets`` and ``blank`` are over the
    remaining images only; ``slider`` maps every image to its 0-based position in napari's slider
    (napari orders a folder by filename, which is what `sorted` gives for ``img%07d.png``).
    """
    context = context or set()
    imgs = sorted(p.name for p in folder.glob("img*.png"))
    targets = [i for i in imgs if i not in context]
    out = {"frames": len(imgs), "images": imgs, "targets": targets,
           "context": [i for i in imgs if i in context], "slider": {i: k for k, i in enumerate(imgs)},
           "has_file": False, "file_parts": [], "filled": {p: 0 for p in parts}, "blank": list(targets),
           "context_labelled": 0}
    csvs = sorted(folder.glob("CollectedData_*.csv"))
    if not csvs:
        return out
    d = pd.read_csv(csvs[0], header=[0, 1, 2], index_col=[0, 1, 2])
    x = d.xs("x", axis=1, level=2)
    x.columns = x.columns.get_level_values(-1)
    out["has_file"] = True
    out["file_parts"] = list(dict.fromkeys(x.columns))
    out["filled"] = {p: int(x[p].notna().sum()) if p in x.columns else 0 for p in parts}
    done = {ix[2] for ix, row in x.iterrows() if row.notna().any()}
    out["blank"] = [i for i in targets if i not in done]
    out["context_labelled"] = sum(i in done for i in out["context"])
    return out


def balanced_order(folders: pd.DataFrame) -> list[str]:
    """Greedy: next folder = the one whose animal, then epoch, is least covered so far.

    Empty folders first; a folder that already holds labels goes last, because finishing it is a
    different kind of job (adding two parts to frames that exist) and does not add a new session.
    """
    todo = folders.copy()
    order: list[str] = []
    a_n: dict[str, int] = {}
    e_n: dict[str, int] = {}
    for tier in (False, True):
        pool = todo[todo["has_file"] == tier]
        left = list(pool.itertuples())
        while left:
            left.sort(key=lambda r: (a_n.get(r.animal, 0), e_n.get(r.epoch, 0),
                                     EPOCH_ORDER.index(r.epoch) if r.epoch in EPOCH_ORDER else 9, r.stem))
            r = left.pop(0)
            order.append(r.stem)
            a_n[r.animal] = a_n.get(r.animal, 0) + 1
            e_n[r.epoch] = e_n.get(r.epoch, 0) + 1
    return order


def _folders(rv, live: Path, parts: list[str]) -> pd.DataFrame:
    man = pd.read_csv(staging_root(rv) / "frame_manifest.csv", dtype=str)
    man = man[man.cam == CAM]
    meta = man[man.category != "context"].drop_duplicates("video_stem").set_index("video_stem")
    ctx = man[man.category == "context"].groupby("video_stem")["image"].apply(set).to_dict()
    sug = man[man.category == "context_label"].groupby("video_stem")["image"].apply(set).to_dict()
    rows = []
    for folder in sorted((live / "labeled-data").glob(f"{CAM}_*")):
        st = folder_status(folder, parts, ctx.get(folder.name))
        st["suggested"] = [i for i in st["targets"] if i in sug.get(folder.name, set())]
        m = meta.loc[folder.name] if folder.name in meta.index else None
        rows.append({"stem": folder.name, "animal": (m["animal"] if m is not None else "?"),
                     "date": (m["date"] if m is not None else "?"),
                     "epoch": (m["epoch"] if m is not None else "?"), **st})
    return pd.DataFrame(rows)


def _block(n: int, r, live: Path, parts: list[str]) -> str:
    mac = f"{MAC_ROOT}/{live.name}/labeled-data/{r.stem}"
    img = _thumb_b64(live / "labeled-data" / r.stem / r.images[0]) if r.images else None
    pic = f'<img alt="" src="data:image/png;base64,{img}" style="max-width:360px">' if img else ""
    def pos(names):
        return ", ".join(str(r.slider[i]) for i in names)

    ctx_note = ""
    if r.context or r.suggested:
        fixed = [i for i in r.targets if i not in r.suggested]
        ctx_note = f"<br><b>Label these {len(fixed)} target frames</b> (slider positions): <b>{pos(fixed)}</b>."
        if r.suggested:
            ctx_note += (f"<br><b>Plus {len(r.suggested)} suggested lick frames</b>: <b>{pos(r.suggested)}</b> "
                         f"&mdash; about every 3rd frame through the start and end of each lick. Flexible: "
                         f"if a neighbour is harder to judge, label that one instead (or as well).")
        ctx_note += (f"<br>The other {len(r.context)} are <i>context</i> &mdash; scroll through them to "
                     f"judge the lick; leave them blank unless you choose to label one completely.")
        if r.context_labelled:
            ctx_note += (f" {r.context_labelled} context frame(s) already carry points &mdash; "
                         f"check each is labelled completely.")
    if not r.has_file:
        what = (f"<b>Empty &mdash; place every point.</b> {len(r.targets)} target frames. Load "
                f"<code>config.yaml</code> first, then this folder (step 4 above).{ctx_note}")
    else:
        missing = [p for p in parts if r.filled[p] == 0]
        partial = {p: r.filled[p] for p in parts if 0 < r.filled[p] < r.frames}
        blanks = [r.slider[b] for b in r.blank]
        what = (f"<b>Part-done &mdash; open the folder only, do NOT load config.yaml.</b> {r.frames} frames. "
                + (f"<b>{', '.join(missing)}</b> {'is' if len(missing) == 1 else 'are'} missing on every frame "
                   f"&mdash; add {'it' if len(missing) == 1 else 'them'} throughout. " if missing else "")
                + ("Already there: " + ", ".join(f"{p} on {k}" for p, k in partial.items()) + ". " if partial else "")
                + (f"Completely blank frames, by slider position: <b>{', '.join(map(str, blanks))}</b>. "
                   if blanks else "") + ctx_note)
        extra = [p for p in r.file_parts if p not in parts]
        if extra:
            what += (f"<br><span style='color:#7c2d12'>This file still lists the old parts "
                     f"({', '.join(extra)}), so napari shows {len(r.file_parts)} names here. Leave those "
                     f"empty &mdash; they are ignored in training.</span>")
    return (f'<div class="card"><h3 style="margin:.1em 0">{n}. {html.escape(r.stem)} &nbsp;'
            f'<span class="legend">{html.escape(r.animal)} &middot; {html.escape(r.epoch)} &middot; '
            f'{html.escape(r.date)}</span></h3>'
            f'<p style="margin:.3em 0">{what}</p>'
            f'<pre><code>{html.escape(mac)}</code></pre>{pic}</div>')


#: What each round-4 pick is, in the labeller's terms (`dlc_hard_frames` rule names).
ROUND4_WHY = {
    "incomplete_tongue": "the mouth opens a little and the network saw no tongue. Look for the tongue tip just "
                         "between the lips &mdash; if any tongue shows, place it; if the mouth is open with no "
                         "tongue, leave tongue blank.",
    "erratic_tongue": "the network was sure of a tongue that jumps or sits where the mouth is closed. Is there "
                      "really a tongue? Place it where it is, or leave it blank if there is none.",
    "erratic_jaw": "the network put the jaw somewhere odd (a jump, or off to one side &mdash; often on the "
                   "tongue's edge). Place the jaw where it truly is; blank if the tongue hides it.",
    "tricky_spout": "the network was unsure of the spout, or it is moving. Place the spout tip.",
    "disagree_tongue": "the two networks were both sure of the tongue but put it in different places (often a "
                       "sideways lick). Place the tip at the end of the tongue along the direction it points "
                       "&mdash; on a sideways lick, the end nearest the spout, not the lowest edge in the image.",
    "disagree_jaw": "the two networks were both sure of the jaw but disagree on where it is. Place it where it "
                    "truly is; blank if the tongue hides it.",
    "lp_erratic_tongue": "the second network (Lightning Pose) was sure of a tongue that jumps or sits where the "
                         "mouth is closed. Is there really a tongue here?",
    "lp_erratic_jaw": "the second network put the jaw somewhere odd (a jump, or on the tongue's edge). Place the "
                      "jaw where it truly is; blank if hidden.",
}


def _round4_folders(rv, live: Path) -> pd.DataFrame:
    """cam4 folders holding round-4 picks, with per-image category and phase from the manifest."""
    man = pd.read_csv(staging_root(rv) / "frame_manifest.csv", dtype=str)
    man = man[man.cam == "cam4"]
    stems = sorted(set(man.loc[man.category == "round4", "video_stem"]))
    parts = dlc_frames.bodyparts("cam4")
    rows = []
    for stem in stems:
        folder = live / "labeled-data" / stem
        if not folder.is_dir():
            continue
        m = man[man.video_stem == stem].set_index("image")
        st = folder_status(folder, parts, set(m.index[m.category == "context"]))
        rows.append({"stem": stem, "animal": m.animal.iloc[0], "date": m.date.iloc[0], "epoch": m.epoch.iloc[0],
                     "cat": m.category.to_dict(), "phase": m.phase.to_dict(), **st})
    return pd.DataFrame(rows)


def _round4_block(n: int, r, live: Path) -> str:
    mac = f"{MAC_ROOT}/{live.name}/labeled-data/{r.stem}"
    picks = [i for i in r.images if r.cat.get(i) == "round4"]
    items = []
    for i in picks:
        f0 = int(i[3:10])
        near = [j for j in r.images if r.cat.get(j) == "context_label" and abs(int(j[3:10]) - f0) <= 4]
        extra = f"; also label <b>{', '.join(str(r.slider[j]) for j in near)}</b>" if near else ""
        items.append(f"<li>slider <b>{r.slider[i]}</b>{extra} &mdash; {ROUND4_WHY.get(r.phase.get(i), '')}</li>")
    done = "" if not r.has_file else f" <i>({len(r.targets) - len(r.blank)} of {len(r.targets)} done)</i>"
    return (f'<div class="card"><h3 style="margin:.1em 0">{n}. {html.escape(r.stem)} &nbsp;'
            f'<span class="legend">{html.escape(r.animal)} &middot; {html.escape(r.epoch)} &middot; '
            f'{html.escape(r.date)}</span>{done}</h3><ul style="margin:.3em 0">{"".join(items)}</ul>'
            f'<p style="margin:.2em 0">The other {len(r.context)} frames are context: scroll through them, '
            f'leave them blank.</p><pre><code>{html.escape(mac)}</code></pre></div>')


def round4_section(rv, live: Path) -> tuple[str, int]:
    """(HTML, frames still blank) for the cam4 round-4 part of the page; empty when there are no picks yet."""
    df = _round4_folders(rv, live)
    if df.empty:
        return ("<h2 id='cam4'>Camera 4 &mdash; round 4</h2><p>No round-4 frames extracted yet.</p>", 0)
    todo = int(sum(len(b) for b in df.blank))
    head = f"""<h2 id="cam4">Camera 4 &mdash; round 4 (the frames the network gets wrong)</h2>
<p class="sub">{len(df)} folders &middot; {int(sum(len(t) for t in df.targets))} frames to label &middot; {todo}
still blank &middot; {int(sum(len(c) for c in df.context))} context frames</p>
<div class="card note">
<p>Same rules as cam1 above: label the listed frames <b>completely</b>, scroll the context, leave it blank.
Each pick says why it was chosen. <b>Most are incomplete licks</b> &mdash; the tongue tip only just between
the lips, never touching the spout. They are the whole point of this round, and they matter most after the
stroke. The &ldquo;also label&rdquo; frames are three frames either side of a pick; as on cam1, they are
a suggestion &mdash; if a neighbour is the harder frame, label that one instead or as well.</p>
<p style="margin-bottom:.2em">These are new folders: load <code>config.yaml</code> first, then the folder,
exactly as for an empty cam1 folder.</p>
</div>
"""
    blocks = [_round4_block(i + 1, r, live) for i, r in enumerate(df.itertuples())]
    return head + "".join(blocks), todo


def build(rv=None, dest: Path | None = None) -> Path:
    rv = rv or PathResolver()
    live = dlc_project.project_dir(rv)
    parts = dlc_frames.bodyparts(CAM)
    df = _folders(rv, live, parts)
    if df.empty:
        raise SystemExit(f"no {CAM}_* folders under {live / 'labeled-data'}")
    order = balanced_order(df)
    by = df.set_index("stem")
    todo_frames = int(sum(len(by.loc[s, "blank"]) for s in order))
    dest = dest or (live.parent / GUIDE_NAME)
    assert_writable(dest.parent)
    root = f"{MAC_ROOT}/{live.name}"
    today = _dt.date.today().isoformat()

    head = f"""<div class="wrap">
<h1>Labelling worksheet &mdash; camera 1 and camera 4 round 4</h1>
<p><a href="#cam1">Camera 1 (the view from below)</a> &middot; <a href="#cam4">Camera 4 round 4 (the frames the
network gets wrong)</a>. The rules and the how-to below apply to both.</p>
<h2 id="cam1" style="margin-top:.4em">Camera 1 &mdash; the view from below</h2>
<p class="sub">{len(df)} folders &middot; {int(sum(len(t) for t in df.targets))} target frames &middot;
{todo_frames} still blank &middot; {int(sum(len(c) for c in df.context))} context frames &middot; generated {today}</p>

<p>This page is the worksheet for cam1: which folders, in what order, and exactly how to open each one.
The manual &mdash; installing, how napari works, what each landmark means in general &mdash; is
<code>LABELLING_GUIDE.html</code> in the same folder. Where the two disagree, this page is newer.</p>

<div class="card note">
<h3 style="margin-top:.2em">Place these four &mdash; and nothing else</h3>
<ul>
<li><b>nose</b> &mdash; the tip of the nose as you see it from below. <i>New for cam1 today</i>: an older
version of the manual said to leave it empty; that was wrong. Caveat (Priya): the cam1 nose and the cam4
nose may not be exactly the same point in 3-D, because from below you see the underside of the tip. That
is accepted and does not change how you place it.</li>
<li><b>jaw</b> &mdash; the same landmark you use on cam4, found from below.</li>
<li><b>tongue</b> &mdash; the tip, <b>only when you can see the tip itself.</b> From below the spout often
sits between the camera and the tongue and hides it. If the tip is hidden, leave tongue blank; do not mark
the edge you can see instead &mdash; that edge is a different point from the tip cam4 sees.</li>
<li><b>spout</b> &mdash; the furthest point of the spout along its own length (the end of the tube in the
direction it points), the same rule as cam4. It enters from the top of the frame.</li>
</ul>
<p style="margin-bottom:.2em">Leave eyes and whiskers empty. <b>On a frame you label, a blank part means
&ldquo;I cannot see it here&rdquo;</b> &mdash; the network is taught that it is hidden. That is right when it is
hidden, and wrong if you simply skipped it. So on every frame you touch, place every part you can see.</p>
</div>

<div class="card note">
<h3 style="margin-top:.2em">Target frames and context frames &mdash; new 30 Sept</h3>
<p>Most folders now hold two kinds of frame. The entry for each folder below says which is which, by
<b>slider position</b> (the frame number shown on napari's slider, starting at 0).</p>
<ul>
<li><b>Target frames &mdash; label these, completely.</b> Every part you can see; a part you cannot see stays
blank.</li>
<li><b>Suggested lick frames &mdash; label these too.</b> Around each lick the folder holds a run of
consecutive frames covering the moment the tongue touches the spout and the moment it lets go &mdash; the
hardest moments to judge. About every third frame of that run is listed as suggested.
<b>The list is a starting point, not a rule:</b> the goal is to label the <i>toughest</i> frames. If the
frame beside a suggested one is harder &mdash; the tongue tip only just showing, the jaw half-hidden &mdash;
label that one instead, or as well.</li>
<li><b>Context frames &mdash; the rest of each run.</b> Scroll back and forth through them to see where the
tongue tip really is and what the jaw is doing, then label the target. Leave them blank unless you decide one
is worth labelling.</li>
<li>Any frame you label, label <b>completely</b>. <b>Never label only the tongue on a frame</b> &mdash; the
blanks beside it would be read as &ldquo;nose, jaw and spout hidden&rdquo;.</li>
</ul>
<p style="margin-bottom:.2em">Blank context frames cost nothing: frames with no points at all are removed
before training.</p>
</div>

<div class="card note">
<h3 style="margin-top:.2em">How to start, on your Mac (every time)</h3>
<ol>
<li><b>Check the server is mounted:</b> Finder sidebar shows <code>Neurobio</code> (Terminal:
<code>ls /Volumes</code> must show <code>Neurobio</code>, not <code>Neurobio-1</code>). If not: Finder
<kbd>Cmd</kbd>+<kbd>K</kbd> &rarr; <code>smb://research.files.med.harvard.edu/Neurobio</code>.</li>
<li>Open Terminal and type:<pre><code>conda activate label
napari</code></pre>Leave the Terminal window open behind napari.</li>
<li><b>Empty folder?</b> First <i>File &rarr; Open File(s)&hellip;</i>, <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>G</kbd>,
paste:<pre><code>{html.escape(root)}/config.yaml</code></pre>Nothing visible happens &mdash; that is correct.
<b>Part-done folder?</b> Skip this step.</li>
<li><i>File &rarr; Open Folder&hellip;</i>, <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>G</kbd>, paste the folder path
from the list below.</li>
<li><b>Count the layers</b> on the left: exactly one points layer, <code>CollectedData_Priya</code>. If there is a
second ending in <code>[1]</code>, close napari and start again &mdash; you loaded the config into a folder
that already had points.</li>
<li>Label, and save every ten minutes with <i>File &rarr; Save Selected Layer(s)&hellip;</i> with the points
layer selected (<kbd>Cmd</kbd>+<kbd>S</kbd> silently fails if the focus is in a side panel).</li>
</ol>
<p style="margin-bottom:.2em">At the rig computer instead: <code>conda activate dlc</code>, then
<code>python -m wfield_local.dlc_project --cam cam1 --label --folder &lt;folder name&gt;</code>.</p>
</div>

<div class="card warn">
<h3 style="margin-top:.2em">Three things that lose work</h3>
<p><b>The path must contain <code>{html.escape(live.name)}</code>.</b> <code>_frame_staging</code> and
<code>training</code> hold identical images under identical names. <code>_frame_staging</code> fails at save
time; <code>training</code> saves happily and is overwritten at the next retrain. Paste the paths below rather
than browsing.</p>
<p><b>One person per folder.</b> Saving rewrites the whole file; the second person to save erases the first.
Say which folder you are taking.</p>
<p style="margin-bottom:.2em"><b>A dropped server connection</b> makes a save fail or land nowhere. After a break,
check the sidebar before carrying on.</p>
</div>

<h2>The folders, in the order to do them</h2>
<p>Each block of four covers all four animals and a spread of recovery stages, so stopping partway leaves the
set balanced. The part-done June folder is last because it adds no new session.</p>
"""
    blocks = [_block(i + 1, by.loc[s].to_frame().T.assign(stem=s).iloc[0], live, parts)
              for i, s in enumerate(order)]
    cam4_html, cam4_todo = round4_section(rv, live)
    page = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Labelling worksheet</title><style>{CSS}</style></head>"
            f"<body>{head}{''.join(blocks)}{cam4_html}</div></body></html>")
    dest.write_text(page, encoding="utf-8")
    print(f"[dlc_cam1_guide] cam1: {len(df)} folders, {todo_frames} blank; cam4 round 4: {cam4_todo} blank "
          f"-> {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
    for i, s in enumerate(order):
        r = by.loc[s]
        print(f"   {i + 1:2d}. {s}  {r.animal} {r.epoch:9s} {len(r.targets):3d} targets "
              f"(+{len(r.context)} context), {len(r.blank):3d} blank, filled {r.filled}", flush=True)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--machine", default=None)
    a = ap.parse_args(argv)
    p = build(rv=PathResolver(machine=a.machine), dest=Path(a.dest) if a.dest else None)
    if a.open:
        import webbrowser
        webbrowser.open(p.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

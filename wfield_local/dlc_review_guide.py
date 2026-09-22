"""Generate the correction guide the LABELLER works from — one page, per session, with the pictures.

`dlc_train --review` produces a table and a folder of crops. That is the right output for whoever
runs the training; it is the wrong output for the undergraduate doing the labelling, who needs to
know which folder to open, which frames to look at inside it, and what to do to each one.

**GENERATED, NOT HAND-WRITTEN.** The frame list changes every refinement round — that is the whole
point of the loop — so a hand-maintained list would be wrong after the first correction pass and
wrong in the most dangerous way, by telling someone to move a point that has already been moved.
This reads `dlc_train.classify()` and rebuilds the page, so the guide and the network's actual
complaint cannot drift apart.

**WRITTEN FOR A MacBOOK, because that is where the labelling happens.** The labeller works from her
own laptop over SMB, not on the rig, and on a Mac she has `napari` + `napari-deeplabcut` and NOT
DeepLabCut and NOT this repo -- so `python -m wfield_local.dlc_project --label` is not a command she
can run. The Mac route is `conda activate label` -> `napari` -> *File -> Open Folder...*, which means
what she needs from this page is a PATH TO PASTE, not a command line. Each session block gives the
exact `/Volumes/Neurobio/...` folder for Cmd+Shift+G. The rig equivalent is kept as a secondary note
for whoever is sitting at the rig instead.

**The crops are EMBEDDED.** A guide that says "now go and find this folder on the server" loses
people exactly where the work is, and over SMB from a laptop that is a slow round trip. Base64 costs
a few MB and buys a page that works from anywhere, which is also why `LABELLING_GUIDE.html` beside it
embeds its own figures.

**It sends people to the LABELLING project, never the training copy.** `dlc_train.stage()` overwrites
the training copy's labels from the labelling project on every run, so a correction made there is
destroyed by the next retrain — silently, since the file is perfectly well-formed either way.

CLI::

    conda activate dlc
    python -m wfield_local.dlc_review_guide            # write CORRECTION_GUIDE.html
    python -m wfield_local.dlc_review_guide --open     # ... and open it
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import html
from pathlib import Path

import pandas as pd

from wfield_local import dlc_project, dlc_train
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

GUIDE_NAME = "CORRECTION_GUIDE.html"

#: Where the project is when the share is mounted at the Neurobio root, which is what
#: `LABELLING_GUIDE.html` tells people to do ("Use this one. Every path in this guide is written for
#: it"). Mounting further down gives /Volumes/Priya/... instead and every path below is wrong by a
#: middle segment, so the guide repeats the mount point rather than assuming it.
MAC_ROOT = "/Volumes/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield"

#: What the labeller actually does for each verdict, in the vocabulary `LABELLING_GUIDE.html`
#: already taught (the `+` tool, the select arrow, Delete) rather than in the classifier's.
ACTIONS = {
    "DELETE": {
        "title": "Delete the point",
        "how": "Switch to the select tool <b>↖</b>, click the dot, press <b>Delete</b>.",
        "why": "There is nothing there. The network is certain, and the picture agrees — "
               "usually fur, or the mouth is not in frame at all. <b>Not</b> for a point that is "
               "merely in the wrong place: that one gets moved.",
        "cls": "bad",
    },
    "REPLACE": {
        "title": "Move the point",
        "how": "Select tool <b>↖</b>, drag the dot onto the cyan circle — or close to it, once you "
               "have looked and agree.",
        "why": "The network learned this landmark from your other frames (how many: the table "
               "above) and is confident it sits elsewhere. That usually makes <i>this</i> frame the "
               "odd one out rather than the network.",
        "cls": "warn",
    },
    "DECIDE": {
        "title": "Look, then place it yourself",
        "how": "Zoom in and put the point where the agreed landmark says it goes, whatever the "
               "network did.",
        "why": "The network is unsure here, so it is <i>not evidence</i> — these are the ambiguous "
               "frames, and the agreed landmark plus your own eye is what settles them. Expect to "
               "move or delete more of these than anywhere else.",
        "cls": "ask",
    },
    "ADD": {
        "title": "Add a point",
        "how": "Pick the body part in the right-hand list, make sure the <b>+</b> tool is active, "
               "click the spot.",
        "why": "Nothing is placed here and the network thinks the part is visible. <b>That is its "
               "opinion, not a correction.</b> If you left it blank because the exact landmark was "
               "not visible — the tongue is out but the tip is behind the spout, say — "
               "then blank was the right answer and it stays blank.",
        "cls": "add",
    },
}

CSS = """
:root{--ink:#14181f;--ink-2:#414a57;--ink-3:#6b7686;--ground:#f3f4f6;--surface:#fff;--sunk:#e9ebef;
--rule:#d5dae1;--rule-soft:#e4e8ed;--accent:#1f5fd0;--accent-soft:#e8eefb;--warn:#a8460a;
--warn-soft:#fbeee4;--good:#1d6b3f;--good-soft:#e6f2ea;--bad:#a11b2e;--bad-soft:#fbe9ec;
--ask:#6b4ea8;--ask-soft:#efeafa;
--display:ui-sans-serif,"Segoe UI Variable Display","Segoe UI",system-ui,-apple-system,sans-serif;
--body:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
--mono:ui-monospace,"Cascadia Mono",Consolas,"SF Mono",Menlo,monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ink:#e7eaee;--ink-2:#aab3c0;
--ink-3:#7e8998;--ground:#12151a;--surface:#191d24;--sunk:#20252d;--rule:#2c333d;--rule-soft:#242a33;
--accent:#7aa5f0;--accent-soft:#1a2436;--warn:#e0954f;--warn-soft:#2e2318;--good:#6cc08b;
--good-soft:#16261c;--bad:#e98a9a;--bad-soft:#2e1a1e;--ask:#b6a0e8;--ask-soft:#221c30;}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--body);font-size:17px;
line-height:1.62;-webkit-text-size-adjust:100%}
.wrap{max-width:74ch;margin:0 auto;padding:40px 16px 96px}
h1,h2,h3,h4{font-family:var(--display);line-height:1.22;letter-spacing:-.011em}
h1{font-size:2.0rem;margin:.2em 0 .1em}
h2{font-size:1.36rem;margin:2.4em 0 .5em;padding-top:.7em;border-top:1px solid var(--rule)}
h3{font-size:1.09rem;margin:1.7em 0 .35em}
h4{font-size:.95rem;margin:1.2em 0 .3em;color:var(--ink-2)}
p{margin:.65em 0}
.sub{color:var(--ink-3);font-family:var(--display);font-size:.95rem;margin-top:.1em}
code,kbd{font-family:var(--mono);font-size:.87em;background:var(--sunk);padding:.1em .34em;
border-radius:4px}
pre{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--accent);
border-radius:8px;padding:14px 16px;overflow-x:auto;font-family:var(--mono);font-size:.84rem;
line-height:1.65}
pre code{background:none;padding:0}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:16px 18px;
margin:1.1em 0}
.note{border-left:3px solid var(--accent);background:var(--accent-soft)}
.stop{border-left:3px solid var(--warn);background:var(--warn-soft)}
.tag{display:inline-block;font-family:var(--display);font-size:.7rem;font-weight:650;
letter-spacing:.06em;text-transform:uppercase;padding:.2em .6em;border-radius:999px;
vertical-align:.14em}
.t-bad{background:var(--bad-soft);color:var(--bad)}
.t-warn{background:var(--warn-soft);color:var(--warn)}
.t-ask{background:var(--ask-soft);color:var(--ask)}
.t-add{background:var(--good-soft);color:var(--good)}
.frame{display:flex;gap:16px;align-items:flex-start;background:var(--surface);
border:1px solid var(--rule);border-radius:10px;padding:14px;margin:.85em 0}
.frame img{width:210px;height:210px;object-fit:contain;border-radius:6px;background:#000;flex:none}
.frame .meta{min-width:0}
.frame .fn{font-family:var(--mono);font-size:.82rem;color:var(--ink-2);word-break:break-all}
.frame .act{font-family:var(--display);font-weight:650;margin:.35em 0 .2em}
.frame .num{font-family:var(--mono);font-size:.78rem;color:var(--ink-3)}
.frame .pos{font-family:var(--display);font-size:.8rem;color:var(--accent);margin-top:.15em}
.lick{border:1px dashed var(--accent);border-radius:10px;padding:2px 14px 10px;margin:1.2em 0;
background:var(--accent-soft)}
.lick>h4{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-family:var(--display);font-size:.9rem;margin:.8em 0}
th,td{text-align:left;padding:.45em .6em;border-bottom:1px solid var(--rule-soft);vertical-align:top}
th{font-weight:650;color:var(--ink-2);border-bottom:1px solid var(--rule)}
ol,ul{padding-left:1.3em}
li{margin:.3em 0}
.legend{display:flex;gap:18px;align-items:center;flex-wrap:wrap;font-family:var(--display);
font-size:.9rem}
.dot{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:-1px;
margin-right:5px}
details{margin:.5em 0}
summary.rig{cursor:pointer;font-family:var(--display);font-size:.86rem;color:var(--ink-3)}
kbd{border:1px solid var(--rule);border-bottom-width:2px}
@media (max-width:620px){.frame{flex-direction:column}.frame img{width:100%;height:auto}}
"""


#: The labelling project's folder name, taken from the project itself rather than written down,
#: so the pasteable paths cannot drift from what is actually on the server.
PROJECT_NAME = ""


def _b64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def folder_positions(live: Path, stem: str) -> "dict[str, int]":
    """``{image: slider position}`` for one folder, 0-based, as napari will order it.

    A filename is how you CHECK you are on the right frame; it is not how you GET there. napari
    opens a folder as a stack and moves through it with the slider at the bottom and the D/A keys,
    so without a position the only way to reach `img0674392.png` is to scroll and squint.

    Position is the index in filename-sorted order. `dlc_frames` zero-pads the frame number to seven
    digits (`img0065207.png`), so lexicographic and numeric order are the same and this is stable --
    verified against the real folders, 2026-09-22. It shifts when frames are ADDED to a folder,
    which is another reason this page is regenerated rather than written once.
    """
    return {q.name: i for i, q in enumerate(sorted((live / "labeled-data" / stem).glob("img*.png")))}


def _crop_for(review_dir: Path, r) -> Path:
    return review_dir / f"{r.verdict}_{r.bodypart}_{r.session}_{Path(r.image).stem}.png"


def _frame_card(r, review_dir: Path, positions: "dict[str, int] | None" = None) -> str:
    a = ACTIONS[r.verdict]
    img = _b64(_crop_for(review_dir, r))
    pic = (f'<img alt="" src="data:image/png;base64,{img}">' if img else
           '<div style="width:210px;height:210px;background:var(--sunk);border-radius:6px"></div>')
    nums = (f"was {r.err_px:.0f} px away &middot; network {r.likelihood:.2f}"
            if pd.notna(r.err_px) else f"network {r.likelihood:.2f}")
    pos = (positions or {}).get(str(r.image))
    where = (f'<div class="pos">slider <b>{pos}</b></div>' if pos is not None else "")
    return f"""<div class="frame">{pic}<div class="meta">
  <span class="tag t-{a['cls']}">{r.verdict}</span>
  <span style="font-family:var(--display);font-weight:650"> {html.escape(str(r.bodypart))}</span>
  <div class="fn">{html.escape(str(r.image))}</div>{where}
  <div class="act">{a['title']}</div>
  <p style="margin:.2em 0">{a['how']}</p>
  <div class="num">{nums} · {html.escape(str(r.phase or ''))}</div>
</div></div>"""


def _session_block(stem: str, g: pd.DataFrame, review_dir: Path, live: Path) -> str:
    licks = [l for l, sub in g.groupby(g.lick) if len(sub) >= 2] if g.lick.notna().any() else []
    mac_path = f"{MAC_ROOT}/{PROJECT_NAME}/labeled-data/{stem}"
    positions = folder_positions(live, stem)
    parts = [f'<h2>{html.escape(stem)}</h2>',
             f'<p class="sub">{len(g)} point'
             f'{"s" if len(g) != 1 else ""} to change in this folder.</p>',
             '<p>In napari: <b>File \u2192 Open Folder\u2026</b>, then '
             '<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>G</kbd> and paste this:</p>',
             f'<pre><code>{html.escape(mac_path)}</code></pre>',
             '<details><summary class="rig">On the rig computer instead</summary>'
             f'<pre><code>python -m wfield_local.dlc_project --cam cam4 --label '
             f'--folder {html.escape(stem)}</code></pre></details>']

    for lick in licks:
        sub = g[g.lick == lick].sort_values("phase")
        bp = ", ".join(sorted(set(sub.bodypart)))
        parts.append(
            f'<div class="lick"><h4>One whole lick — {html.escape(bp)}</h4>'
            # NOT "16 ms apart" -- `dlc.frames.lick_offsets_s` is -16/0/+32/+64 ms, so consecutive
            # frames are 16, 32 and 32 ms apart and a group may hold 2, 3 or 4 of them. Naming one
            # spacing would be wrong on most pairs; what is true and what matters is that they are
            # one ~80 ms protrusion.
            f'<p style="margin:.3em 0">These {len(sub)} frames are the <b>same tongue flick</b> '
            f'&mdash; moments of one protrusion, about 80&nbsp;ms from start to finish. Do them '
            f'together and in order: the point has to travel smoothly across them, so fixing one '
            f'on its own just makes it disagree with its neighbours instead.</p>'
            + "".join(_frame_card(r, review_dir, positions) for _, r in sub.iterrows())
            + '</div>')

    rest = g[~g.lick.isin(licks)] if licks else g
    if len(rest):
        if licks:
            parts.append("<h4>Single frames in this folder</h4>")
        parts.append("".join(_frame_card(r, review_dir, positions)
                             for _, r in rest.sort_values("verdict").iterrows()))
    return "\n".join(parts)


def _training_counts(proj: Path, rv=None) -> "dict[str, tuple[int, int]]":
    """``{bodypart: (frames in the TRAINING split, frames labelled)}`` -- see `_counts_table`."""
    import numpy as np

    truth = dlc_train.labels(proj)
    _, test_idx = dlc_train.split(proj, rv=rv)
    is_test = np.zeros(len(truth), bool)
    is_test[test_idx] = True
    out = {}
    for bp in dict.fromkeys(truth.columns.get_level_values("bodyparts")):
        placed = np.isfinite(dlc_train._xy(truth, bp)).all(axis=1)
        out[bp] = (int((placed & ~is_test).sum()), int(placed.sum()))
    return out


def _counts_table(counts: "dict[str, tuple[int, int]]") -> str:
    """How many frames the network actually saw of each part -- i.e. how much its opinion is worth.

    Not a detail, and the first draft of this page got it wrong. It said the network "learned this
    landmark from your other ~190 frames" for EVERY part. True of `nose` (192) and `spout` (187);
    for `tongue` the real figure is 71, because the tongue is only out for a fraction of the time.
    That is a ~3x overstatement on the one part most of the flagged frames are about, and it errs
    toward telling the labeller to trust the network where she should be trusting her own eyes.
    """
    rows = []
    for bp, (n_train, n_all) in counts.items():
        weak = n_train < 100
        note = ("the network has seen FEW of these &mdash; trust your own eyes first" if weak
                else "a solid consensus")
        style = ' style="color:var(--warn)"' if weak else ""
        rows.append(f"<tr><td><code>{html.escape(bp)}</code></td><td>{n_train}</td>"
                    f"<td>{n_all}</td><td{style}>{note}</td></tr>")
    return ('<div class="card"><h3 style="margin-top:.2em">How much the network&rsquo;s opinion is '
            'worth, per part</h3>'
            '<p style="margin:.3em 0">It only knows a landmark from the frames it was trained on. '
            'The tongue is out for only a fraction of the time, so it has seen far fewer of those '
            '&mdash; which is also why most of this page is tongue frames.</p>'
            "<table><tr><th>Part</th><th>Trained on</th><th>You labelled</th><th></th></tr>"
            + "".join(rows) + "</table></div>")


def pending_frames(live: Path, which: str, bodyparts: list) -> "dict[str, list[str]]":
    """``{session: [image, ...]}`` for frames that are ON DISK but carry no label yet.

    Derived from the project rather than from a diff of two extraction plans, so it is right at
    every point in the cycle: empty before new frames are extracted, exactly the new ones after,
    and shrinking as they are labelled. A list computed once and pasted in would be wrong the
    moment she saved.

    "No label" means no row in ``CollectedData``, or a row whose four trained bodyparts are ALL
    blank. A row of NaNs is what DLC writes for a frame nobody has opened, and it is
    indistinguishable from a frame someone opened and correctly left empty -- so a frame where the
    tongue alone is blank is NOT pending, because the tongue is legitimately absent most of the
    time. Requiring all four to be blank is the conservative reading: it can leave a genuinely
    half-done frame off the list, which costs a revisit; the opposite error sends her to re-do work
    she has finished.
    """
    import numpy as np

    out: dict[str, list[str]] = {}
    for folder in sorted((live / "labeled-data").glob(f"{which}_*")):
        images = sorted(q.name for q in folder.glob("img*.png"))
        if not images:
            continue
        h5 = folder / f"CollectedData_{dlc_project.SCORER}.h5"
        labelled = set()
        if h5.is_file():
            df = pd.read_hdf(h5)
            have = set(df.columns.get_level_values("bodyparts"))
            cols = [b for b in bodyparts if b in have]
            for ix, row in df.iterrows():
                name = ix[2] if isinstance(ix, tuple) else str(ix)
                if cols and np.isfinite(
                        np.array([row[(c, b, co)] for c, b, co in df.columns
                                  if b in cols and co in ("x", "y")], float)).any():
                    labelled.add(name)
        todo = [im for im in images if im not in labelled]
        if todo:
            out[folder.name] = todo
    return out


#: A frame `dlc_train.seed_pending` filled in is NOT blank -- it carries the network's own guesses,
#: written only above `pcutoff`. Saying "place the four parts" would be wrong, and worse, it would
#: read as though the points already there were somebody's work.
SEEDED_INTRO = (
    "<p><b>They already have points on them, put there by the network — those points are NOT "
    "anybody&rsquo;s work.</b> It placed a point only where it was confident and left the rest "
    "blank, so your job here is the same as on the corrections above: look, move what is wrong, "
    "delete what is sitting on nothing, and add what is missing.</p>"
    "<p>Roughly what to expect, from how often the network was confident enough to place each part: "
    "<code>nose</code> on nearly every frame, <code>spout</code> almost as often, <code>jaw</code> "
    "on about three quarters, and <code>tongue</code> on about half. <b>A blank tongue does not mean "
    "the tongue is absent</b> — it means the network was unsure, which is exactly where your "
    "own eyes are worth most.</p>")

BLANK_INTRO = ("<p>They are <b>blank</b> — nothing has been placed on them yet, so this is "
               "ordinary labelling rather than correction.</p>")


def _pending_block(pending: "dict[str, list[str]]", bodyparts: list,
                   seeded: "dict[str, list[str]]" | None = None,
                   live: Path | None = None) -> str:
    """The NEW-FRAMES half of the page.

    Two different jobs wear the same clothes here. A frame `seed_pending` has filled in is
    NOT blank -- it carries the network's own guesses, above `pcutoff` only -- so telling
    the labeller to "place the four parts" would be wrong, and worse, it would read as if
    the points already there were somebody's work. `seeded` is passed separately because
    the files cannot tell the two apart.
    """
    seeded = seeded or {}
    if not pending and not seeded:
        return ""
    use = seeded or pending
    is_seeded = bool(seeded)
    total = sum(len(v) for v in use.values())
    parts = [f"""
<h2>New frames to label ({total})</h2>
<p>These frames were added to the folders you already know, to give the network more examples of the
parts it is weakest on \u2014 above all the tongue.</p>
{SEEDED_INTRO if is_seeded else BLANK_INTRO}

<div class="card note">
<h3 style="margin-top:.2em">Place only these four parts</h3>
<p style="margin:.3em 0"><b>{", ".join(f"<code>{html.escape(b)}</code>" for b in bodyparts)}</b>
\u2014 and nothing else. Leave the whiskers and the eyes <b>empty</b> on these frames.</p>
<p style="margin:.3em 0">The main labelling guide asks for ten parts on cam4. <b>That is out of
date as of 2026-09-22</b> — the whiskers are deferred, and the project now lists six names in
total (these four, plus one eye on each side camera). You should not see whisker rows in napari at
all. Four parts done well is what is wanted here.</p>
</div>

<div class="card">
<h3 style="margin-top:.2em">How to do one frame</h3>
<ol>
<li>Open the folder exactly as you do for the corrections \u2014 the paths are the same, and the new
frames are mixed in among the ones you have already done.</li>
<li>Press <kbd>Shift</kbd>+<kbd>&rarr;</kbd> to jump straight to the <b>first unlabelled frame</b>.
That is the fastest way to find them; the filenames below are there so you can check you are on the
right one.</li>
<li>Pick the body part in the right-hand list, make sure the <b>+</b> tool is active, and click.
<kbd>W</kbd> and <kbd>S</kbd> move up and down the part list without the mouse.</li>
<li><b>If a part is not visible, leave it blank.</b> Do not guess. The tongue is inside the mouth on
most frames and an invented point is worse than a missing one \u2014 blanks are simply skipped when
the network is trained, but a wrong point is taught as truth.</li>
<li><kbd>Shift</kbd>+<kbd>&rarr;</kbd> again for the next one.</li>
</ol>
</div>

<div class="card stop">
<p style="margin:.2em 0"><b>The tongue is the reason these frames exist.</b> The network has seen far
fewer tongues than anything else, so a tongue placed on a frame where it is genuinely out is the most
valuable click on this page. Use the tongue landmark you and Priya already agreed, and use it the
same way every time \u2014 consistency across frames is worth more than any single point being
perfect.</p>
</div>
"""]
    for stem, imgs in sorted(use.items(), key=lambda kv: -len(kv[1])):
        mac_path = f"{MAC_ROOT}/{PROJECT_NAME}/labeled-data/{stem}"
        parts.append(
            f'<h3>{html.escape(stem)} &mdash; {len(imgs)} new frame'
            f'{"s" if len(imgs) != 1 else ""}</h3>'
            f'<pre><code>{html.escape(mac_path)}</code></pre>'
            f'<p class="sub" style="margin-top:-.4em">'
            + ", ".join(
                f"<code>{html.escape(i)}</code>"
                + (f" <b>[{pos[i]}]</b>" if (pos := folder_positions(live, stem)) and i in pos
                   else "")
                for i in imgs)
            + " &mdash; the number in brackets is the slider position.</p>")
    return "\n".join(parts)


def build(proj: Path | None = None, rv=None, dest: Path | None = None) -> Path:
    """Write ``CORRECTION_GUIDE.html`` beside ``LABELLING_GUIDE.html``. Returns its path."""
    rv = rv or PathResolver()
    proj = proj or dlc_train.train_project(rv)
    live = dlc_project.project_dir(rv)
    global PROJECT_NAME
    PROJECT_NAME = live.name
    d = dlc_train.classify(proj, rv=rv)
    review_dir = proj / "review"
    today = _dt.date.today().isoformat()
    snap = dlc_train._newest_predictions(proj)
    snap_name = snap.stem if snap is not None else "unknown"

    COUNTS_TABLE = _counts_table(_training_counts(proj, rv))
    pending = pending_frames(live, dlc_train.cam(), dlc_train.parts())
    seeded = dlc_train.seeded_frames(proj, rv)
    n_pending = sum(len(v) for v in pending.values()) or sum(len(v) for v in seeded.values())
    counts = d.verdict.value_counts().to_dict() if len(d) else {}
    summary = " · ".join(f"{counts.get(v, 0)} {v.lower()}" for v in ACTIONS)
    n_sessions = d.session.nunique() if len(d) else 0

    body = [f"""<div class="wrap">
<h1>{"Two jobs: fixes, and new frames" if n_pending else "Fixing the flagged labels"}</h1>
<p class="sub">cam4 &middot; {len(d)} points to fix across {n_sessions} folders &middot; {summary}
{f"<br><b>plus {n_pending} new frames to check</b>" if n_pending else ""}<br>
generated {today} from <code>{html.escape(snap_name)}</code></p>

<p>You have already labelled these frames. The network has now been trained on your labels, and
these are the points where it disagrees with you strongly enough to be worth a second look. Most of
your labelling is <b>not</b> on this page, and that is the point — the median point is within 2&nbsp;px
of where the network expects it.</p>

<div class="card note">
<h3 style="margin-top:.2em">Read the pictures like this</h3>
<p class="legend">
<span><span class="dot" style="background:#e6194b"></span><b>Red cross</b> — your label</span>
<span><span class="dot" style="background:#00d2e6"></span><b>Cyan circle</b> — the network</span>
</p>
<p style="margin-bottom:.2em">The network is not automatically right. It learned the landmark from
your own other frames, so think of the cyan circle as <i>where you put it everywhere else</i>.
How many frames that is depends on the part, and it matters — see the table below.
Where it is confident and you disagree, usually this frame drifted. Where it is unsure, it is not
evidence at all.</p>
</div>

<div class="card stop">
<h3 style="margin-top:.2em">One rule before you start</h3>
<ol>
<li><b>Only ever open the paths on this page.</b> There is a second copy of the whole project on the
server, inside a folder called <code>training</code>, and a third folder of identical images called
<code>_frame_staging</code>. Both look completely normal when opened. The <code>training</code> copy
is overwritten every time the network is retrained, so anything fixed there is thrown away and
nothing warns you. Every path below is the real project — copy them rather than browsing.</li>
</ol>
</div>

<h2>Opening the labelling window</h2>
<p>Exactly the routine you already have. Connect to the server first — HMS network or VPN,
Finder <kbd>Cmd</kbd>+<kbd>K</kbd>, <code>smb://research.files.med.harvard.edu/Neurobio</code>
— then in <b>Terminal</b>:</p>
<pre><code>conda activate label
napari</code></pre>
<p>An empty grey window opens. Leave Terminal open behind it; closing it closes napari too. Then
<b>File → Open Folder…</b> (not drag-and-drop), press
<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>G</kbd>, and paste the path from whichever section below you are
working on. Each one is written out in full, ready to copy.</p>

<div class="card stop">
<p style="margin:.2em 0"><b>Check the path says
<code>widefield-Priya-2026-09-08</code>.</b> There is another folder on the server,
<code>_frame_staging</code>, holding copies of the very same images under the very same names.
Opening it looks completely normal and only goes wrong when you save — it asks you to
“associate with a DLC project” and then refuses. Every path on this page is already the
right one; the trap is only if you go browsing.</p>
<p style="margin:.2em 0"><b>And check <code>ls /Volumes</code> shows <code>Neurobio</code>, not
<code>Neurobio-1</code>.</b> A stale second mount is the usual reason a path here “cannot be
found”.</p>
</div>

{COUNTS_TABLE}

<h2>What each tag means</h2>
<table>
<tr><th>Tag</th><th>What to do</th><th>Why</th></tr>"""]

    for v, a in ACTIONS.items():
        body.append(f'<tr><td><span class="tag t-{a["cls"]}">{v}</span></td>'
                    f'<td><b>{a["title"]}.</b> {a["how"]}</td><td>{a["why"]}</td></tr>')
    body.append("</table>")

    body.append("""
<div class="card note">
<h3 style="margin-top:.2em">Getting to a frame without scrolling</h3>
<p style="margin:.3em 0">Every frame below carries a <b>slider number</b>. That is its position in
the folder, counting from <b>0</b>. At the bottom of the napari window there is a slider with the
current position shown next to it — <b>click that number, type the one from this page, and press
Enter</b>. You land on the frame directly.</p>
<p style="margin:.3em 0">The filename under each picture is how you CHECK you are in the right place,
not how you get there. <kbd>D</kbd> and <kbd>A</kbd> step forward and back one frame if you overshoot,
and <kbd>Shift</kbd>+<kbd>&rarr;</kbd> jumps to the first frame with nothing on it.</p>
<p style="margin:.3em 0"><b>The numbers move when frames are added to a folder.</b> They are right
for this version of the page — regenerate it rather than working from a printout.</p>
</div>

<div class="card note">
<h3 style="margin-top:.2em">Before you touch anything: be in the points layer</h3>
<p style="margin:.3em 0">On the right there are two layers stacked like transparent sheets — the
photograph underneath, the points on top. <b>Click the points layer so it is highlighted.</b> Nothing
works otherwise: clicks land nowhere, <kbd>Delete</kbd> does nothing, and the letter keys do nothing.
If your clicks seem to be ignored, you have selected the image layer by accident — click the
points layer once and carry on.</p>
<p style="margin:.3em 0">Then pick the tool. <b>+</b> places a new point; the arrow <b>↖</b>
selects and drags an existing one. And pick the body part in the list on the right before you place
anything — a point goes onto whichever part is currently selected, so placing a tongue while
<code>jaw</code> is highlighted puts a jaw where the tongue is, silently.</p>
<p style="margin:.3em 0">Hiding the points layer for a moment (the small eye icon) is the quickest
way to check whether a dot is covering the very thing you are trying to judge.</p>
</div>

<div class="card stop">
<h3 style="margin-top:.2em">When to delete a point, and when not to</h3>
<p style="margin:.3em 0"><b>Delete it</b> when the point is sitting on something that is not the
part: the network placed a tongue where there is no tongue, or the mouth is not even in frame. Several
of the seeded tongues will be like this — the tongue is only out for about 70&nbsp;ms per lick,
and the network guesses on the frames in between.</p>
<p style="margin:.3em 0"><b>Delete it</b> when the part is there but the <i>landmark</i> is not —
the tongue is out and the tip is hidden behind the spout, so there is nothing to put the point on.
An honest blank is better than a point placed where you think the tip probably is.</p>
<p style="margin:.3em 0"><b>Do NOT delete</b> a point that is merely in the wrong place. Drag it
instead. Deleting and re-placing gets the same result but loses you the chance to see how far it
moved, which is the thing worth noticing.</p>
<p style="margin:.3em 0"><b>Do NOT delete</b> a point just because you are unsure. If you can see the
landmark, place it; if you cannot, blank it. "Unsure" usually means zoom in further.</p>
<p style="margin:.3em 0">A blank costs nothing — the network simply skips that part on that
frame. <b>A wrong point is taught as the truth.</b> That asymmetry is the whole rule.</p>
</div>

<div class="card">
<h3 style="margin-top:.2em">Zoom in before you judge</h3>
<p style="margin-bottom:.2em">Scroll the wheel. At full-frame size a point that is 12&nbsp;px out
looks perfect. For scale: the frame is 680&nbsp;px across and the gap between the <code>nose</code>
and <code>jaw</code> labels runs about 174&nbsp;px, so 12&nbsp;px is roughly a fourteenth of the
distance from nose to chin. Every frame on this page is already cropped and doubled, so the picture
here is closer than what you will see in napari until you zoom.</p>
</div>""")

    if len(d):
        order = d.groupby("session").size().sort_values(ascending=False).index
        for stem in order:
            body.append(_session_block(stem, d[d.session == stem], review_dir, live))
    else:
        body.append('<h2>Nothing to fix</h2><p>The network agrees with every label within '
                    'tolerance. Nothing to do on this pass.</p>')

    body.append(f"""
{_pending_block(pending, dlc_train.parts(), seeded, live)}

<h2>Saving</h2>
<p>Exactly as in the main guide, and it is the part worth being fussy about:</p>
<ol>
<li>Click the <b>points layer</b> in the layer list on the right, so it is highlighted. <b>The save
only writes the SELECTED layer</b> — with the image layer selected you will save the photograph and
not your points, and napari will not complain.</li>
<li><b>File → Save Selected Layer(s)…</b></li>
<li>Accept the filename it offers. It should be <code>CollectedData_Priya.h5</code>, in the folder
you are working on. If it offers anything else, you have the wrong layer selected — go back to 1.</li>
</ol>
<p>That writes <code>CollectedData_Priya.h5</code> into the folder on the server. It is the only
record of your work — there is no autosave and no undo across sessions. <b>Save every ten minutes,
and always before switching folders or closing.</b></p>
<p><b>Deletions need saving too.</b> Removing a point changes the file exactly as placing one does,
so a folder where you only deleted things still has to be saved — it is the easiest save to forget,
because it does not feel like you added anything.</p>
<div class="card stop" style="margin-top:1em">
<p style="margin:.2em 0"><b>Do not trust <kbd>Cmd</kbd>+<kbd>S</kbd> here.</b> It only reaches
napari when the keyboard focus is on the image or the layer list, and it fails <i>silently</i> when
it is not — which looks exactly like a successful save. Use the File menu.</p>
<p style="margin:.2em 0">Saving writes across the network to the server, so on a laptop it can take a
moment. Let it finish before switching folders.</p>
</div>

<h2>When you have finished a folder</h2>
<p>Nothing else to run, and nothing to copy anywhere — your points save straight back to the
server. Tell Priya which folders you got through. The network gets retrained from your corrected
labels and a fresh version of this page is generated from the result, so the list above will be
different (and shorter) next time.</p>
<p class="sub" style="margin-top:2.5em">Generated by
<code>python -m wfield_local.dlc_review_guide</code> — do not edit this page by hand, it is
rebuilt from the network's own output every round. The permanent instructions live in
<code>LABELLING_GUIDE.html</code> beside it.<br>
Your labels live in <code>{html.escape(f"{MAC_ROOT}/{live.name}")}</code>.</p>
</div>""")

    dest = dest or (live.parent / GUIDE_NAME)
    assert_writable(dest.parent)
    page = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Fixing the flagged labels — widefield orofacial</title>"
            f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>")
    dest.write_text(page, encoding="utf-8")
    print(f"[dlc_review_guide] {len(d)} points across {n_sessions} folders -> {dest} "
          f"({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true", help="open the page when it is written")
    ap.add_argument("--dest", default=None, help="write somewhere else")
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

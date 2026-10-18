"""The labelling page for the SIDE cameras (cam2 = left side, cam3 = right side) -- SIDECAM_GUIDE.html.

    conda activate locanmf          # pandas + Pillow only
    python -m wfield_local.dlc_sidecam assign        # first: which camera labels which moment
    python -m wfield_local.dlc_sidecam_guide [--open]

Separate from CAM1_GUIDE.html (the student's cam4 / cam1 worksheet) on purpose (Priya, 2026-10-13: "we will want a
separate cam2/3 labelling guide"). Generated, like that page, because what it lists changes as folders are done.

WHAT IT ENCODES (`wfield_local/dlc_sidecam.py` for the reasons):
  * four parts only -- nose, jaw, tongue, spout -- on the raw image as recorded (cam3 is flipped later, for
    training, not for the labeller);
  * each listed frame is labelled in the camera it is listed under; a moment is listed under ONE side camera,
    except about one in five, listed under BOTH ("both" -- the overlap that lets cam2 and cam3 be checked
    against each other);
  * targets vs context exactly as on the cam1 page: label listed frames completely, leave context blank;
  * Part A (pilot) = 8 sessions chosen so every animal and epoch is covered; Part B = the cam4 round-4 moments;
    Part C = the remaining sessions.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
from pathlib import Path

import pandas as pd

from wfield_local import dlc_frames, dlc_project
from wfield_local.dlc_cam1_guide import EPOCH_ORDER, balanced_order, folder_status
from wfield_local.dlc_frames import manifest_path, out_root
from wfield_local.dlc_review_guide import CSS, MAC_ROOT
from wfield_local.dlc_sidecam import ASSIGN_NAME, ROUND, SIDE
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

GUIDE_NAME = "SIDECAM_GUIDE.html"
PILOT_SESSIONS = 8
CAM_NAME = {"cam2": "camera 2 (left side)", "cam3": "camera 3 (right side)"}
PAIR_WHY = {
    "incomplete_tongue": "an incomplete lick on camera 4", "erratic_tongue": "a doubtful tongue on camera 4",
    "lp_erratic_tongue": "a doubtful tongue on camera 4", "disagree_tongue": "a hard-to-place tongue on camera 4",
    "dlc_only_tongue": "a tongue near the spout on camera 4", "erratic_jaw": "a hard-to-place jaw on camera 4",
    "lp_erratic_jaw": "a hard-to-place jaw on camera 4", "disagree_jaw": "a hard-to-place jaw on camera 4",
    "dlc_only_jaw": "a jaw behind the spout on camera 4", "tricky_spout": "a hard-to-place spout on camera 4",
}


def _session_key(animal: str, date: str) -> str:
    """Session = animal + date. NOT the video timestamp: cam2 / cam3 files of one session can differ by a second
    (PS94 0820: cam2_..T09_44_54, cam3_..T09_44_55), which split one session into two."""
    return f"{animal}_{date}"


def sessions(rv, live: Path) -> pd.DataFrame:
    """One row per (session, side camera) folder: assigned targets, status, round-1 vs round-4 membership."""
    man = pd.read_csv(manifest_path(rv), dtype=str)
    a = pd.read_csv(out_root(rv) / ASSIGN_NAME, dtype=str)
    a["label"] = a.label.astype(str).str.lower() == "true"
    a["both"] = a.both.astype(str).str.lower() == "true"
    parts = dlc_frames.bodyparts("cam2")
    rows = []
    for stem in sorted(set(a.video_stem)):
        folder = live / "labeled-data" / stem
        if not folder.is_dir():
            continue
        m = man[man.video_stem == stem].drop_duplicates("image").set_index("image")
        mine = a[(a.video_stem == stem) & a.label]
        st = folder_status(folder, parts, set(m.index) - set(mine.image))      # only MY targets count as targets
        rows.append({"stem": stem, "session": _session_key(m.animal.iloc[0], m.date.iloc[0]), "cam": stem.split("_", 1)[0],
                     "animal": m.animal.iloc[0], "date": m.date.iloc[0], "epoch": m.epoch.iloc[0],
                     "mine": mine.set_index("image")[["category", "phase", "both", "position"]].to_dict("index"),
                     "round4": bool((mine.category == ROUND).any()), **st})
    return pd.DataFrame(rows)


def _block(label: str, r, live: Path) -> str:
    mac = f"{MAC_ROOT}/{live.name}/labeled-data/{r.stem}"
    items = []
    for img in sorted(r.mine, key=lambda i: r.slider.get(i, 0)):
        info = r.mine[img]
        ph = str(info["phase"])
        why = (f"same moment as {PAIR_WHY.get(ph[5:], 'a hard frame on camera 4')}" if ph.startswith("pair:")
               else f"{info['position']}, {ph}")
        both = " &nbsp;<span class='legend'>(also listed for the other side camera)</span>" if info["both"] else ""
        items.append(f"<li>slider <b>{r.slider.get(img, '?')}</b> &mdash; {html.escape(why)}{both}</li>")
    done = "" if not r.has_file else f" <i>({len(r.targets) - len(r.blank)} of {len(r.targets)} done)</i>"
    n_ctx = r.frames - len(r.targets)
    return (f'<div class="card"><h3 style="margin:.1em 0">{label} {html.escape(r.stem)} &nbsp;'
            f'<span class="legend">{CAM_NAME[r.cam]} &middot; {html.escape(r.animal)} &middot; '
            f'{html.escape(r.epoch)} &middot; {html.escape(r.date)}</span>{done}</h3>'
            f'<ul style="margin:.3em 0">{"".join(items)}</ul>'
            f'<p style="margin:.2em 0">The other {n_ctx} frames in this folder are <b>not for labelling</b>: context '
            f'to scroll through, or moments listed under the other side camera.</p>'
            f'<pre><code>{html.escape(mac)}</code></pre></div>')


def _part(title: str, anchor: str, intro: str, df: pd.DataFrame, order: list[str], prefix: str, live: Path) -> str:
    sub = df[df.session.isin(order)]
    n_t = int(sum(len(t) for t in sub.targets))
    n_b = int(sum(len(b) for b in sub.blank))
    out = [f'<h2 id="{anchor}">{title}</h2><p class="sub">{len(order)} sessions &middot; {len(sub)} folders '
           f'&middot; {n_t} frames to label &middot; {n_b} still blank</p>{intro}']
    k = 0
    for s in order:
        for cam in SIDE:
            g = sub[(sub.session == s) & (sub.cam == cam)]
            if g.empty or not len(g.iloc[0].mine):
                continue
            k += 1
            out.append(_block(f"{prefix}.{k}", g.iloc[0], live))
    return "".join(out)


def build(rv=None, dest: Path | None = None) -> Path:
    rv = rv or PathResolver()
    live = dlc_project.project_dir(rv)
    df = sessions(rv, live)
    if df.empty:
        raise SystemExit("no side-camera assignment yet: run `python -m wfield_local.dlc_sidecam assign`")
    # sessions: round-1 (the folders extracted with cam4 as anchor) vs those holding round-4 moments
    sess = df.groupby("session").agg(animal=("animal", "first"), epoch=("epoch", "first"),
                                     has_file=("has_file", "any"), round4=("round4", "any")).reset_index()
    r1 = sess[~sess.round4].rename(columns={"session": "stem"})
    r1_order = balanced_order(r1) if len(r1) else []
    pilot, rest = r1_order[:PILOT_SESSIONS], r1_order[PILOT_SESSIONS:]
    r4 = sess[sess.round4].sort_values(["epoch", "animal"], key=lambda c: c.map(
        {e: i for i, e in enumerate(EPOCH_ORDER)}) if c.name == "epoch" else c).session.tolist()
    dest = dest or (live.parent / GUIDE_NAME)
    assert_writable(dest.parent)
    root = f"{MAC_ROOT}/{live.name}"
    today = _dt.date.today().isoformat()
    head = f"""<div class="wrap">
<h1>Side-camera labelling worksheet &mdash; cameras 2 and 3</h1>
<p class="sub">generated {today}</p>
<p>Cameras 2 and 3 film the face from the left and right side. They show two things camera 4 cannot: the jaw
opening in profile, with no spout rod across the chin, and how far forward and down the tongue reaches. One network
will learn both side views together (camera 3 is mirrored for it automatically, later) &mdash; so <b>each moment
is labelled in only ONE of the two side cameras</b>, the one it is listed under below. About one moment in five is
listed under both; label it in both. Work in order: <a href="#A">Part A</a> (a pilot that covers every animal and
recovery stage), then <a href="#B">Part B</a>, then <a href="#C">Part C</a>. Installing and napari basics:
<code>LABELLING_GUIDE.html</code>; camera 4 / camera 1 work stays on <code>CAM1_GUIDE.html</code>.</p>

<div class="card note">
<h3 style="margin-top:.2em">Cameras 2 and 3: place these four &mdash; and nothing else</h3>
<ul>
<li><b>nose</b> &mdash; the tip of the snout, as it appears in profile.</li>
<li><b>jaw</b> &mdash; the same landmark you use on camera 4, seen from the side. From the side the spout rod
does not cover the chin, so the jaw is visible on most frames; if something does hide it (a paw, grooming),
leave it blank.</li>
<li><b>tongue</b> &mdash; the tip, by the same rule as on the other cameras, <b>only when you can see the tip
itself</b>; otherwise blank.</li>
<li><b>spout</b> &mdash; the furthest point of the spout along its own length, the same rule as cameras 4 and 1.</li>
</ul>
<p style="margin-bottom:.2em">Label the image as it is shown &mdash; camera 3 is <b>not</b> mirrored for you.
Leave eyes and whiskers empty. <b>On a frame you label, a blank part means &ldquo;I cannot see it
here&rdquo;</b>, so place every part you can see.</p>
</div>

<div class="card note">
<h3 style="margin-top:.2em">Frames to label, and everything else</h3>
<ul>
<li><b>Listed frames &mdash; label these, completely</b> (by slider position, starting at 0).</li>
<li><b>Everything else in the folder</b> is either context (the frames just before and after a lick, to scroll
through and see where the tongue tip really is) or a moment listed under the other side camera. <b>Leave them
blank</b> &mdash; that is correct and costs nothing.</li>
<li>Never label only one part on a frame &mdash; the blanks beside it would be read as hidden.</li>
</ul>
</div>

<div class="card note">
<h3 style="margin-top:.2em">How to start, on your Mac (every time)</h3>
<ol>
<li>Check the server is mounted: Finder sidebar shows <code>Neurobio</code> (Terminal <code>ls /Volumes</code>
shows <code>Neurobio</code>, not <code>Neurobio-1</code>); if not, Finder <kbd>Cmd</kbd>+<kbd>K</kbd> &rarr;
<code>smb://research.files.med.harvard.edu/Neurobio</code>.</li>
<li>Terminal: <pre><code>conda activate label
napari</code></pre></li>
<li>A folder with no points yet: first <i>File &rarr; Open File(s)&hellip;</i>, <kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>G</kbd>,
paste <pre><code>{html.escape(root)}/config.yaml</code></pre>(nothing visible happens; that is correct).</li>
<li><i>File &rarr; Open Folder&hellip;</i>, paste the folder path from the list below.</li>
<li>Exactly one points layer, <code>CollectedData_Priya</code>; a second ending in <code>[1]</code> means restart.</li>
<li>Save every ten minutes: <i>File &rarr; Save Selected Layer(s)&hellip;</i> with the points layer selected.</li>
</ol>
</div>

<div class="card warn">
<h3 style="margin-top:.2em">Three things that lose work</h3>
<p><b>The path must contain <code>{html.escape(live.name)}</code></b> (not <code>_frame_staging</code> or
<code>training</code>) &mdash; paste the paths below. <b>One person per folder.</b> <b>A dropped server
connection</b> makes a save fail: check the sidebar after a break.</p>
</div>
"""
    a_intro = ("<div class='card note'><p>The pilot: eight sessions chosen so that every animal and every recovery "
               "stage appears. A first side-camera network is trained on these before anything else is labelled.</p></div>")
    b_intro = ("<div class='card note'><p>Moments that were hard on camera 4 (mostly incomplete licks and a jaw "
               "behind the spout). They come with many context frames from camera 4's bursts; label only the listed "
               "frames.</p></div>")
    c_intro = "<div class='card note'><p>The remaining sessions, once the pilot has been checked.</p></div>"
    body = (_part("Part A &mdash; pilot", "A", a_intro, df, pilot, "A", live)
            + _part("Part B &mdash; camera 4 round-4 moments", "B", b_intro, df, r4, "B", live)
            + _part("Part C &mdash; the remaining sessions", "C", c_intro, df, rest, "C", live))
    page = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Side-camera labelling</title><style>{CSS}</style></head><body>{head}{body}</div></body></html>")
    dest.write_text(page, encoding="utf-8")
    for name, order in (("A pilot", pilot), ("B round-4", r4), ("C rest", rest)):
        sub = df[df.session.isin(order)]
        print(f"[dlc_sidecam_guide] {name}: {len(order)} sessions, {int(sum(len(t) for t in sub.targets))} frames "
              f"to label ({int(sum(len(b) for b in sub.blank))} blank)", flush=True)
    print(f"-> {dest}", flush=True)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--dest", default=None)
    a = ap.parse_args(argv)
    p = build(dest=Path(a.dest) if a.dest else None)
    if a.open:
        import webbrowser
        webbrowser.open(p.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

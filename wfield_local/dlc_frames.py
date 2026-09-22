"""Extract a stratified DeepLabCut LABELLING set from the widefield behavior cameras.

DLC's own ``extract_frames`` picks frames by k-means over appearance, which on a 1.5 M-frame
recording of a head-fixed mouse returns mostly the resting posture: the animal spends most of its
time not licking, so appearance clustering under-samples exactly the events the analysis is about.
This picks frames by EXPERIMENTAL DESIGN instead -- spout position x within-trial phase -- using the
cue times and the camera alignment template that ``behavior_clips`` already established.

**Coverage comes from spanning animals and EPOCHS, not from many frames of one session.** A network
labelled only on pre-stroke frames degrades precisely where the result lives; the 2pRAM cohort saw
post-stroke tongue tracking fall apart in its two severe animals, which is a tracking artefact
sitting on top of a real deficit and nearly impossible to separate afterwards. Hence a small
``per_session`` and a ``--cohort`` mode that walks one session per animal x epoch.

SOURCE VIDEOS ARE OPENED READ-ONLY AND NEVER WRITTEN, the same guarantee ``behavior_clips`` makes:
``cv2.VideoCapture`` is the only thing that touches them and ``assert_writable`` guards the output,
so a path bug lands as a refusal rather than as an edit to an irreplaceable recording.

FRAMES ARE WRITTEN AS PNG, never JPEG. Labels are placed on these images and inference then runs on
the raw video; a lossy intermediate would train the network on compression artefacts that the
inference frames do not have.

SELECTION IS DETERMINISTIC (``dlc.frames.seed``), keyed on animal+date+camera. Re-running extends
nothing and re-picks the same frames, so a second run cannot quietly grow a divergent labelling set
beside the one somebody has already spent hours annotating.

Layout, which is DLC's own so the project can adopt it without copying::

    <dlc>/labeled-data/<video-stem>/img<FRAME>.png
    <dlc>/labeled-data/frame_manifest.csv

The manifest carries the provenance of every frame -- animal, date, camera, epoch, trial, spout
position, trial category, phase, seconds from cue. A labelled frame whose provenance is unknown
cannot be audited later, and "which epochs is this network actually trained on?" is a question that
gets asked the first time tracking looks worse after the stroke.

CLI::

    python -m wfield_local.dlc_frames 20260907            # every session on one date
    python -m wfield_local.dlc_frames --cohort            # one session per animal x epoch
    python -m wfield_local.dlc_frames 20260907 --cam cam1 # a single camera
    python -m wfield_local.dlc_frames --cohort --dry-run  # what it would take, no decoding
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

MANIFEST_COLUMNS = ["animal", "date", "cam", "epoch", "video_stem", "frame", "trial_id",
                    "position", "category", "phase", "t_from_cue_s", "image"]


def _cfg() -> dict:
    return (config.defaults().get("dlc") or {})


def cameras() -> list[str]:
    """Camera names, in declaration order. Accepts the old flat list form too."""
    c = _cfg().get("cameras") or ["cam4"]
    return [str(k) for k in (c.keys() if isinstance(c, dict) else c)]


def role(cam: str) -> str:
    c = _cfg().get("cameras") or {}
    return str((c.get(cam) or {}).get("role", "")) if isinstance(c, dict) else ""


def bodyparts(cam: str | None = None) -> list[str]:
    """What ``cam`` can see, or the union across all cameras when ``cam`` is None.

    PER VIEW, not one shared list. cam1 looks up from below and has no nose or identifiable
    whiskers; a side view has only one side's. A label for a part a camera cannot see is not merely
    wasted -- it is invented, and a network trained on invented points learns to hallucinate.

    The union is what a DLC project's own `bodyparts:` must contain, since one project spans the
    views; the per-camera list is what may actually be placed in each. Names are shared where views
    overlap, which is what makes triangulation possible at all.
    """
    c = _cfg().get("cameras")
    if not isinstance(c, dict):                       # old flat form: one shared list
        return [str(b) for b in (_cfg().get("bodyparts") or [])]
    if cam is not None:
        return [str(b) for b in ((c.get(cam) or {}).get("bodyparts") or [])]
    out: list[str] = []
    for spec in c.values():
        for b in (spec or {}).get("bodyparts") or []:
            if str(b) not in out:
                out.append(str(b))
    return out


def shared_bodyparts(cams=None) -> list[str]:
    """Bodyparts visible in at least TWO of ``cams`` -- the only ones 3D can ever reconstruct."""
    cams = list(cams or cameras())
    counts: dict[str, int] = {}
    for cam in cams:
        for b in bodyparts(cam):
            counts[b] = counts.get(b, 0) + 1
    return [b for b in bodyparts() if counts.get(b, 0) >= 2]


def phases() -> dict[str, float]:
    """``{phase name: seconds from the cue}``, ordered as declared in the YAML."""
    return {str(k): float(v) for k, v in (_cfg().get("frames", {}).get("phases") or {}).items()}


def per_session() -> int:
    return int(_cfg().get("frames", {}).get("per_session", 24))


def lick_per_session() -> int:
    return int(_cfg().get("frames", {}).get("lick_per_session", 0))


def target_per_session() -> int:
    """Frames KEPT per session per camera after appearance pruning, or 0 to keep everything.

    The planner deliberately proposes more than this. Behaviour alignment decides which MOMENTS are
    eligible -- that is what guarantees a tongue-out frame exists at all, which appearance
    clustering on its own never does -- and the pruning below decides which of the eligible ones are
    worth a person's time. Neither half works alone.
    """
    return int(_cfg().get("frames", {}).get("target_per_session", 0))


def lick_fraction() -> float:
    """Share of ``target_per_session`` spent on lick-locked frames.

    Held ABOVE the tongue's share of wall-clock time on purpose: the tongue is the hardest part to
    track, the only one that is ever fully hidden, and the one the analysis actually turns on.
    """
    return float(_cfg().get("frames", {}).get("lick_fraction", 0.5))


def sessions_per_epoch() -> int:
    """Sessions kept per EPOCH, across animals, or 0 for one per animal x epoch.

    Priya, 2026-09-13: "we shouldnt need to label every session; 2 per epoch, across animals, should
    be ok." Correct on the frame count -- one session per animal x epoch is 15 sessions and several
    hundred frames per camera more than DLC needs.

    THE COST IS NOT UNIFORM, and it lands where this module's docstring already says the risk is.
    At 2 per epoch only 2 of 4 animals appear in each, so some animal is absent from some
    post-stroke epoch -- and "the 2pRAM cohort saw post-stroke tongue tracking fall apart in its two
    severe animals" is exactly that failure, a tracking artefact on top of a real deficit and nearly
    impossible to separate afterwards. So the sessions are chosen to ROTATE animals across epochs
    rather than taken in name order: every animal still appears, in as many distinct epochs as the
    quota allows. That mitigates the risk; it does not remove it. Cutting frames per session instead
    reaches the same total with every animal x epoch cell intact, and is the safer knob if the
    tracking later looks worse for one animal after its stroke.
    """
    return int(_cfg().get("frames", {}).get("sessions_per_epoch", 0))


def lick_offsets_s() -> list[float]:
    """Seconds from a lick onset to sample, spanning the tongue-out epoch.

    A LIST, not one offset. The tongue is out for ~70 ms per lick and looks very different across
    that -- emerging, at peak extension well clear of the spout, retracting -- so one offset per
    lick both wastes the event and teaches the network a single posture. Falls back to the older
    scalar ``lick_offset_s`` so a config that predates this still works.
    """
    f = _cfg().get("frames", {})
    if f.get("lick_offsets_s"):
        return [float(v) for v in f["lick_offsets_s"]]
    return [float(f.get("lick_offset_s", 0.008))]


#: Decoding one session's DAQ takes minutes and the cohort walk asks for the same session once per
#: camera. Keyed on (animal, date) because the licks are a property of the SESSION, not the view.
_LICKS: dict[tuple[str, str], np.ndarray] = {}


def _lick_cache_path(animal: str, date: str, rv=None) -> Path:
    return out_root(rv) / "lick_onsets" / f"{animal}_{date}.npz"


def _params_key(params: dict) -> str:
    """A stable digest of the lick-detection settings the cache was built under."""
    return ";".join(f"{k}={params[k]}" for k in sorted(params))


def _cached_licks(animal: str, date: str, params: dict, rv=None):
    """Previously decoded onsets, or None. Returns None on a params change rather than stale times.

    Re-decoding a session's DAQ is minutes of network I/O and the whole cohort is ~20 minutes, which
    is paid every time an offset is tuned -- but a cache of DERIVED event times is only safe while
    the settings that derived them hold, so the params digest is stored beside the array and a
    mismatch is a miss, not a warning nobody reads.
    """
    p = _lick_cache_path(animal, date, rv)
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=False)
        if str(z["params"]) != _params_key(params):
            print(f"[dlc_frames] {animal} {date}: lick cache built under different "
                  f"lick_detection params -> re-decoding", flush=True)
            return None
        return np.asarray(z["lick_s"], float)
    except (OSError, KeyError, ValueError):
        return None


def _store_licks(animal: str, date: str, licks: np.ndarray, params: dict, rv=None) -> None:
    p = _lick_cache_path(animal, date, rv)
    assert_writable(p.parent)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, lick_s=licks, params=np.array(_params_key(params)))


def lick_onsets(animal: str, date: str, sid: str, rv=None) -> np.ndarray:
    """DAQ lick-onset times (s) for one session, or an empty array if the recording is unavailable.

    Straight from ``daq_trials.decode`` -- the pipeline's single source for licks -- rather than any
    count column on the trials table, because what is needed here is the TIMES.
    """
    from wfield_local import daq_trials

    if (animal, date) in _LICKS:
        return _LICKS[(animal, date)]
    rv = rv or PathResolver()
    params = config.defaults()["lick_detection"]
    cached = _cached_licks(animal, date, params, rv)
    if cached is not None:
        _LICKS[(animal, date)] = cached
        return cached
    root = Path(rv.root("daq_recorder_output")) / date
    if not root.is_dir():
        return np.array([], float)
    # The DAQ filename's timestamp is a second or two off the session id's (separate clocks), so
    # match on the animal and date and take the one recording that exists for them.
    cands = sorted(root.glob(f"{animal}_{date}_*.h5"))
    if not cands:
        print(f"[dlc_frames] {animal} {date}: no DAQ .h5 -> no lick-locked frames", flush=True)
        return np.array([], float)
    try:
        dec = daq_trials.decode(cands[0], params)
    except Exception as exc:                                  # noqa: BLE001 - report, do not stop
        print(f"[dlc_frames] {animal} {date}: DAQ decode failed ({exc}) -> no lick-locked frames",
              flush=True)
        return np.array([], float)
    out = np.asarray(dec["lick_s"], float)
    print(f"[dlc_frames] {animal} {date}: {out.size} DAQ lick onsets (decoded)", flush=True)
    _store_licks(animal, date, out, params, rv)
    _LICKS[(animal, date)] = out
    return out


def select_licks(trials: pd.DataFrame, licks: np.ndarray, n_per_pos: int, window_s: float, rng):
    """``[(trial_row, lick_time), ...]`` -- licks inside the response window, spread over positions.

    Spread over spout POSITION for the same reason the cue-locked frames are: a tongue labelled only
    where the animal licks most would train a network that finds the tongue best exactly where the
    behaviour is already easiest.

    The window is ``[cue, min(cue + window_s, next_cue)]`` -- bounded by the NEXT CUE, which is
    `daq_trials`' own rule. Without that bound a lick belonging to the following trial can be
    attributed to this one, and since the following trial may be at a different spout position, the
    frame would be filed under a position the tongue was not reaching for.
    """
    if licks.size == 0 or trials.empty:
        return []
    cues = np.sort(trials["cue_s"].to_numpy(float))
    out = []
    for pos in sorted(trials["pos_name"].dropna().astype(str).unique()):
        g = trials[trials["pos_name"].astype(str) == pos]
        pool = []
        for _, row in g.iterrows():
            cue = float(row["cue_s"])
            nxt = cues[cues > cue]
            hi = min(cue + window_s, nxt[0]) if nxt.size else cue + window_s
            inside = licks[(licks >= cue) & (licks <= hi)]
            pool += [(row, float(t)) for t in inside]
        if not pool:
            continue
        take = min(n_per_pos, len(pool))
        for k in rng.choice(len(pool), size=take, replace=False):
            out.append(pool[int(k)])
    return out


def seed() -> int:
    return int(_cfg().get("frames", {}).get("seed", 92))


def out_root(rv=None) -> Path:
    """The DLC tree: ``Priya/DeepLabCut/Widefield`` (logical root ``dlc``).

    MOVED 2026-09-16 (Priya) out of ``<behavior_cameras>/dlc``. It belongs with the lab's other
    DeepLabCut work rather than inside the camera recordings, and the labelling set is not a
    recording -- it is derived from them and has a different lifetime.

    Falls back to the old location when the logical root is absent, so a checkout whose `paths.yaml`
    predates the move keeps resolving instead of raising somewhere unrelated. The fallback is a
    transition aid, not a supported second home: two trees is how the project tree and the extraction
    tree drifted apart on 2026-09-15 and left napari opening folders that were 5/6 stale frames.
    """
    rv = rv or PathResolver()
    try:
        return Path(rv.root("dlc"))
    except (KeyError, ValueError):
        return Path(rv.root("behavior_cameras")) / "dlc"


def anchor_cam() -> str | None:
    """The camera whose trial/lick choice every other camera copies, or None for independent picks.

    **WHY THIS EXISTS.** The selection RNG was seeded per CAMERA, so each view drew its own trials
    and its own licks. Measured on the 2026-09-12 manifest that left 875 events sampled by cam1
    alone, 866 by cam4 alone, and **22 by all four** -- out of ~900 each. Nothing was wrong with any
    individual view, but the four sets of labelled frames were of different MOMENTS.

    That does not hurt training, where each camera learns from its own frames, and it does not hurt
    3D at inference, where the network predicts every frame of synchronised video. What it costs is
    the ability to TRIANGULATE THE LABELS THEMSELVES -- label two views, reconstruct in 3D, and
    reproject to seed the remaining two. With 27 usable cam4+cam1 pairs that was not worth running.

    Seeding on the ANCHOR instead makes every camera choose the same trials and the same lick times.
    The frame NUMBER still differs per camera -- `frame_of` maps a DAQ time through that camera's own
    alignment template -- which is the point: same instant, each view's own frame.

    **THE ANCHOR MUST BE THE MOST-LABELLED CAMERA, and it is cam4 (7,642 points over 13 folders).**
    Seeding on cam4 reproduces cam4's existing selection EXACTLY, so its labels stay valid and only
    the unlabelled views move. Changing the anchor later would re-pick cam4's frames and orphan that
    work; treat it as fixed unless you are prepared to re-label.
    """
    v = _cfg().get("frames", {}).get("anchor_cam", None)
    return str(v) if v else None


def staging_root(rv=None) -> Path:
    """Where EXTRACTION writes, which is NOT where labelling happens.

    Named `_frame_staging` rather than `labeled-data` on purpose. The DLC project has its own
    `labeled-data/<video-stem>/` holding copies of the same images under the same names, and while
    both existed a person could open either. Opening the staging one WORKS -- images and seeds load
    -- and fails only at save, as a "associate with a DLC project?" prompt that reads like a plugin
    fault. That cost a labelling session on 2026-09-16.

    `napari_deeplabcut` treats any folder with a `labeled-data` component in its path as a labelling
    target, so the old name actively invited the mistake. This name is not a target, and a staging
    folder with no CollectedData in it is refused outright by the reader rather than opened and then
    found unsaveable.
    """
    return out_root(rv) / "_frame_staging"


def _rng(animal: str, date: str, cam: str) -> np.random.Generator:
    """A generator whose stream depends only on the session, so cameras and dates are independent.

    Seeding once per run and drawing in iteration order would make every session's choice depend on
    which other sessions happened to be processed first -- so adding a date would silently re-pick
    frames for dates already labelled.

    With an `anchor_cam` configured the camera part of the key is replaced by the anchor, so all
    views share one stream and therefore one set of moments. See `anchor_cam`.
    """
    a = anchor_cam()
    if a:
        # keep any suffix ("_lick") while swapping the camera, so the two streams stay distinct
        for c in cameras():
            if cam == c or cam.startswith(c + "_"):
                cam = a + cam[len(c):]
                break
    key = f"{animal}_{date}_{cam}_{seed()}".encode()
    return np.random.default_rng(np.frombuffer(key.ljust(32, b"\0")[:32], dtype=np.uint32))


def select_trials(trials: pd.DataFrame, n_cells: int, rng) -> pd.DataFrame:
    """One trial per spout position, rotating over trial CATEGORY so all three are represented.

    Positions are the design's own strata. Category (success / working / stopped, from
    ``behavior_clips.categorise``) is rotated rather than crossed: crossing would need 6 x 3 x
    n_phases frames per session, and the point of a small per-session budget is that breadth comes
    from epochs. Rotating still guarantees that failed and stopped trials -- where the mouth and
    tongue look least like the training distribution -- are in the labelled set.
    """
    from wfield_local.behavior_clips import CATEGORIES

    d = trials.copy()
    if "cat" not in d.columns:
        from wfield_local.behavior_clips import categorise
        d["cat"] = categorise(d)
    picks = []
    for i, pos in enumerate(sorted(d["pos_name"].dropna().astype(str).unique())):
        g = d[d["pos_name"].astype(str) == pos]
        if g.empty:
            continue
        # Rotate the PREFERRED category by position index; fall back through the rest so a session
        # missing a category still yields a frame for every position it ran.
        order = [CATEGORIES[(i + k) % len(CATEGORIES)] for k in range(len(CATEGORIES))]
        for cat in order:
            pool = g[g["cat"] == cat]
            if not pool.empty:
                picks.append(pool.iloc[int(rng.integers(len(pool)))])
                break
    out = pd.DataFrame(picks)
    return out.head(max(1, n_cells)) if not out.empty else out


def frame_of(tpl, t_daq_s: float, offset_s: float) -> int:
    """Camera frame index for ``offset_s`` after DAQ time ``t_daq_s``.

    Takes a DAQ TIME rather than specifically a cue, because the lick-locked frames anchor on a lick
    onset instead. Both are times on the same DAQ clock; the template does not care which event.

    The same affine ``behavior_clips`` cuts with: the alignment template maps DAQ samples to camera
    frames at ~1.2 ms residual, about a third of one frame at 250 fps.
    """
    fs, fps = float(tpl["fs_daq"]), float(tpl["fps_cam"])
    slope = float(tpl["slope_daqSample_per_camFrame"])
    icept = float(tpl["intercept_daqSample"])
    # Every term is coerced to a Python float first: `round` on a numpy float returns a numpy
    # float, which would flow into `cv2.CAP_PROP_POS_FRAMES` and the manifest as `2481.0`.
    return round((float(t_daq_s) * fs - icept) / slope + float(offset_s) * fps)


def plan_session(animal, date, sid, epoch, cam, rv=None) -> list[dict]:
    """The frames this session would contribute: one row per (trial, phase). No video is opened.

    Returns [] with a printed reason for any session that cannot be used, rather than raising --
    the cohort walk should report a missing template and carry on, not stop at the first gap.
    """
    rv = rv or PathResolver()
    tp = Path(rv.root("alignment_templates")) / cam / animal / (date + ".npz")
    if not tp.exists():
        print(f"[dlc_frames] {animal} {date} {cam}: no alignment template -> skip", flush=True)
        return []
    tpl = dict(np.load(tp, allow_pickle=True))
    if not bool(tpl["quality_ok"]):
        print(f"[dlc_frames] {animal} {date} {cam}: template quality_ok FALSE -> skip", flush=True)
        return []
    vids = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(cam + "_*.avi"))
    tpath = Path(rv.root("behavior_out")) / "sessions" / animal / date / (sid + "_trials.csv")
    if not vids or not tpath.exists():
        print(f"[dlc_frames] {animal} {date} {cam}: no video or trial table -> skip", flush=True)
        return []

    ph = phases()
    n_cells = max(1, per_session() // max(1, len(ph)))
    chosen = select_trials(pd.read_csv(tpath), n_cells, _rng(animal, date, cam))
    if chosen.empty:
        print(f"[dlc_frames] {animal} {date} {cam}: no usable trials -> skip", flush=True)
        return []

    stem, n_frames, rows = vids[0].stem, int(tpl["n_cam_frames"]), []
    for _, row in chosen.iterrows():
        for name, off in ph.items():
            f = frame_of(tpl, float(row["cue_s"]), off)
            if 0 <= f < n_frames:
                rows.append({"animal": animal, "date": date, "cam": cam, "epoch": epoch,
                             "video_stem": stem, "frame": f, "trial_id": int(row["trial_id"]),
                             "position": str(row["pos_name"]), "category": str(row["cat"]),
                             "phase": name, "t_from_cue_s": off,
                             # its own group: cue-locked frames are seconds apart and share no
                             # posture, so each stands or falls alone.
                             "_group": f"{int(row['trial_id'])}:{name}",
                             "image": f"img{f:07d}.png", "_video": str(vids[0])})

    n_lick = lick_per_session()
    if n_lick:
        from wfield_local.behavior_clips import POST_S, categorise
        trials = pd.read_csv(tpath)
        trials["cat"] = categorise(trials)
        # POST_S is the scored response window (3.5 s, from every session's gui_config.json) --
        # the same bound behavior_clips cuts to, so a lick-locked frame is inside a clip that exists.
        picks = select_licks(trials, lick_onsets(animal, date, sid, rv),
                             max(1, n_lick // 6), POST_S, _rng(animal, date, cam + "_lick"))
        for row, t_lick in picks:
            for off in lick_offsets_s():
                f = frame_of(tpl, t_lick, off)
                if 0 <= f < n_frames:
                    rows.append({"animal": animal, "date": date, "cam": cam, "epoch": epoch,
                                 "video_stem": stem, "frame": f, "trial_id": int(row["trial_id"]),
                                 "position": str(row["pos_name"]), "category": str(row["cat"]),
                                 # The phase names the offset, so a labelled frame can be traced to
                                 # a point in the lick cycle -- "lick" alone would lose that, and
                                 # the tongue's appearance is exactly what varies across it.
                                 "phase": f"lick{off * 1000:+.0f}",
                                 "t_from_cue_s": round(t_lick + off - float(row["cue_s"]), 4),
                                 # ALL OFFSETS OF ONE LICK SHARE A GROUP, and pruning keeps or
                                 # drops a group whole. Priya, 2026-09-13: "labeling a few
                                 # consecutive frames from one lick is probably helpful for the
                                 # human to ensure they're picking the same part of the tongue."
                                 # Frame-wise pruning would have kept one frame from each of many
                                 # licks -- best for the network, worst for the labeller, who then
                                 # never sees the tongue MOVE and has to guess at "the tip" on
                                 # isolated frames. The consistency of the labels is upstream of
                                 # everything the network can learn.
                                 "_group": f"{int(row['trial_id'])}:{t_lick:.4f}",
                                 "image": f"img{f:07d}.png", "_video": str(vids[0])})
    return rows


def _thumb(frame) -> np.ndarray:
    """A 64x64 z-scored greyscale thumbnail -- the space poses are compared in.

    Z-scoring per frame so a session that simply ran brighter does not read as a different posture.
    Coarse on purpose: the question is "is this a different pose", not "is this a different frame",
    and at 250 fps full resolution answers the second.
    """
    import cv2
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    v = cv2.resize(g, (64, 64)).astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def farthest_first(F: np.ndarray, k: int) -> list[int]:
    """Indices of ``k`` rows of ``F`` chosen to be as mutually UNLIKE as possible.

    Farthest-point sampling, not k-means. Three reasons, all of which matter here: it is
    deterministic without a seed, so a re-run re-picks the same frames the way the rest of this
    module promises; it has no convergence step to fail quietly on the small n a single stratum
    gives; and it optimises the thing actually wanted -- spread -- whereas k-means optimises
    within-cluster variance and will happily return two near-identical frames from a dense region.

    Starts at the MEDOID so the first pick is a typical frame rather than an outlier, which matters
    when k is small and the set contains a blurred or half-occluded frame.
    """
    n = len(F)
    if k >= n:
        return list(range(n))
    D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    picks = [int(np.argmin(D.sum(1)))]
    while len(picks) < k:
        picks.append(int(np.argmax(D[:, picks].min(1))))
    return picks


def _stratum(r: dict) -> tuple:
    """(pool, spout position) -- the cell a candidate is pruned WITHIN, never across.

    Pruning globally would let the resting frames, which are many and mutually similar, out-vote the
    lick-locked ones, which are few; the guarantee that tongue-out frames survive has to be
    structural rather than hoped for. Position is in the key for the same reason it is in the
    planner: a tongue labelled only where the animal licks best trains a network that finds it only
    there.
    """
    return ("lick" if str(r["phase"]).startswith("lick") else "phase", str(r["position"]))


def _decode_feats(grp: list[dict]):
    """``({id(row): thumbnail}, [rows that decoded])`` for one video's candidates.

    Split out as its own function so the pruning logic can be tested without a video file -- the
    decode is the only part of it that needs one, and the parts worth pinning are the parts that
    decide what a person ends up labelling.
    """
    import cv2

    feats, ok = {}, []
    cap = cv2.VideoCapture(str(grp[0]["_video"]))
    try:
        for r in sorted(grp, key=lambda x: x["frame"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            got, fr = cap.read()
            if got:
                feats[id(r)] = _thumb(fr)
                ok.append(r)
    finally:
        cap.release()
    return feats, ok


def prune_by_appearance(rows: list[dict], rv=None, target: int | None = None) -> list[dict]:
    """Keep the most visually DISTINCT ``target`` frames per session, within behaviour strata.

    Measured 2026-09-13, on the set this replaces: of the six offsets sampled around one lick onset,
    99-100% of the within-onset pairs are closer in appearance than the 5th percentile of pairs
    drawn from DIFFERENT onsets (median 16.4 against 46.5). Six offsets of one lick are six copies
    of one pose. Pose diversity lives BETWEEN licks, so the budget belongs there.

    PRUNING IS DECIDED ON THE ANCHOR CAMERA AND COPIED, when one is configured, for the same reason
    the trial choice is: the views have to stay simultaneous or nothing can be triangulated against
    a hand label. The cost is real and worth naming -- two poses distinct from the front can be
    identical from the side, so the side views inherit a selection that is not optimal for them.
    Matched frames are worth more than that margin.
    """
    target = target_per_session() if target is None else target
    if not target or not rows:
        return rows

    anchor = anchor_cam()
    lead = anchor if anchor and any(r["cam"] == anchor for r in rows) else None

    out, decided = [], {}
    for cam in dict.fromkeys(r["cam"] for r in rows):
        for stem in dict.fromkeys(r["video_stem"] for r in rows if r["cam"] == cam):
            grp = [r for r in rows if r["cam"] == cam and r["video_stem"] == stem]
            sess = (grp[0]["animal"], grp[0]["date"])
            if lead and cam != lead and sess in decided:
                out += [r for r in grp if (r["_group"], r["phase"]) in decided[sess]]
                continue
            feats, ok = _decode_feats(grp)
            if not ok:
                continue
            n_lick = int(round(target * lick_fraction()))
            quota = {"lick": n_lick, "phase": max(0, target - n_lick)}
            kept = []
            for pool in ("lick", "phase"):
                pr = [r for r in ok if _stratum(r)[0] == pool]
                if not pr:
                    continue
                groups: dict[str, list] = {}
                for r in pr:
                    groups.setdefault(r["_group"], []).append(r)
                size = max(1, round(len(pr) / len(groups)))       # frames per group
                # BOTH POOLS ARE STRATIFIED BY SPOUT POSITION. An earlier version left the lick pool
                # unstratified, arguing that at ~92 px between commanded positions against ~3 px
                # within a trial the spout IS an appearance difference, so farthest-first would
                # spread over positions for free. Measured, it does not: the kept licks came out
                # 8/8/16/24/24/40 across the six positions while the stratified phase pool sat at
                # 18-22. The comparison was the wrong one. What matters is the spout's signal
                # against the ANIMAL's, and at 64x64 a 92 px shift is ~8 px of a thumbnail dominated
                # by body posture -- so the spout never drove the choice at all.
                cells: dict[tuple, list] = {}
                for k in groups:
                    cells.setdefault(_stratum(groups[k][0]), []).append(k)
                n_cell = max(1, len(cells))
                base, extra = divmod(max(1, quota[pool] // size), n_cell)
                # Remainder ROTATED PER SESSION. Handing it to the first cells every time gave the
                # alphabetically-first positions an extra frame in all 15 sessions.
                spin = sum(ord(ch) for ch in stem) % n_cell
                for c0, (_cell, ks) in enumerate(sorted(cells.items())):
                    c = (c0 - spin) % n_cell
                    take = base + (1 if c < extra else 0)
                    if take <= 0:
                        continue
                    G = np.stack([np.mean([feats[id(r)] for r in groups[k]], 0) for k in ks])
                    for i in farthest_first(G, take):
                        kept += groups[ks[i]]
            out += kept
            if lead and cam == lead:
                # KEYED ON `_group`, NOT trial_id: two licks inside one trial share a trial_id AND
                # an offset name, so a (trial_id, phase) key silently matched both and the side
                # views came out LARGER than the anchor they were supposed to be copying. Caught by
                # the dry run's per-camera counts, 2026-09-13.
                decided[sess] = {(r["_group"], r["phase"]) for r in kept}
            print(f"[dlc_frames] {stem}: {len(grp)} candidates -> {len(kept)} kept", flush=True)
    return out


def extract(rows: list[dict], rv=None) -> int:
    """Decode and write the planned frames. Returns the number written.

    Seeks per frame rather than walking the file: a session contributes ~24 scattered frames out of
    1.5 M, so a sequential pass would decode 60000x more than it keeps. Rows are visited in
    ascending frame order so the seeks run forward through the stream.
    """
    import cv2

    written = 0
    for video, group in _by_video(rows):
        dest = staging_root(rv) / group[0]["video_stem"]
        assert_writable(dest)
        dest.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))          # READ-ONLY, always
        if not cap.isOpened():
            print(f"[dlc_frames] could not open {video} -> skip", flush=True)
            continue
        try:
            for r in sorted(group, key=lambda x: x["frame"]):
                out = dest / r["image"]
                if out.exists():
                    continue                        # already extracted; never re-write over labels
                cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
                ok, frame = cap.read()
                if not ok:
                    print(f"[dlc_frames] {r['video_stem']} frame {r['frame']}: read failed",
                          flush=True)
                    continue
                cv2.imwrite(str(out), frame)        # PNG: lossless, per the module docstring
                written += 1
        finally:
            cap.release()
    return written


def _by_video(rows):
    """[(video path, rows)] grouped so each recording is opened once."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["_video"], []).append(r)
    return sorted(groups.items())


def write_manifest(rows: list[dict], rv=None, dest=None) -> Path:
    """Append to the manifest, de-duplicated on (video_stem, frame).

    APPEND, not overwrite: the labelling set is built up over several runs -- a date at a time, and
    a camera at a time -- and a manifest rewritten from one run's rows would drop the provenance of
    every frame extracted before it while the images stayed on disk.
    """
    dest = Path(dest) if dest else staging_root(rv) / "frame_manifest.csv"
    assert_writable(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if dest.exists():
        with open(dest, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
    seen = {(r["video_stem"], str(r["frame"])) for r in existing}
    fresh = [r for r in rows if (r["video_stem"], str(r["frame"])) not in seen]
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for r in existing + fresh:
            w.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})
    return dest


def usable(animal: str, date: str, cams, rv=None) -> bool:
    """Does every requested camera have a passing alignment template for this session?

    ALL the requested cameras, not any: views that will be triangulated together have to be labelled
    on the same sessions, and a session with a template on one camera only cannot contribute a
    matched pair.
    """
    rv = rv or PathResolver()
    root = Path(rv.root("alignment_templates"))
    for cam in cams:
        tp = root / cam / animal / (date + ".npz")
        if not tp.exists():
            return False
        try:
            if not bool(dict(np.load(tp, allow_pickle=True))["quality_ok"]):
                return False
        except (OSError, KeyError, ValueError):
            return False
    return True


def _pick_middle(sessions, ok):
    """The most central session that ``ok`` accepts, searching outward from the middle.

    Plain "middle of the list" is what this replaces, and it was wrong in a way that hid itself: the
    camera alignment templates only start at 2026-06-06, while the `pre` epoch runs from 2026-05-27,
    so the median pre-stroke session for EVERY animal was an untemplated June date. The cohort walk
    duly skipped it and produced a labelling set with no pre-stroke frames at all -- the baseline the
    whole pre/post comparison rests on -- while reporting success. Falling back outward keeps the
    "representative of the epoch, not of its boundary" intent and still yields a session.
    """
    ordered = sorted(sessions)
    mid = len(ordered) // 2
    for i in sorted(range(len(ordered)), key=lambda j: (abs(j - mid), j)):
        if ok(ordered[i]):
            return ordered[i]
    return None


def cohort_sessions(rv=None, animals=None, cams=None) -> list[tuple[str, str, str, str]]:
    """One session per animal x epoch: ``[(animal, date, sid, epoch), ...]``.

    THE MIDDLE session of each epoch, not the first or last. A boundary session is the one most
    likely to be atypical of its epoch -- the first acute day is the animal at its worst, the last
    subacute day is nearly chronic -- and a labelling set should be representative of the epoch it
    is drawn from, not of the transition into it. See :func:`_pick_middle` for the usability
    fallback, which is not an optimisation.
    """
    from wfield_local import epochs

    rv = rv or PathResolver()
    cams = list(cams or cameras())
    base = Path(rv.root("behavior_out")) / "sessions"
    want = set(config.normalize_animals(animals) or [])
    by_epoch: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for an_dir in sorted(p for p in base.glob("PS*") if p.is_dir()):
        an = an_dir.name
        if want and an not in want:
            continue
        for sess in sorted(p for p in an_dir.iterdir() if p.is_dir()):
            date = sess.name
            ep = epochs.epoch_of(f"{an}_{date[4:]}")
            if ep is None:
                continue
            for t in sorted(sess.glob("*_trials.csv")):
                by_epoch.setdefault((an, ep), []).append((date, t.name[: -len("_trials.csv")]))
    out = []
    for (an, ep), sess in sorted(by_epoch.items()):
        pick = _pick_middle(sess, lambda ds, _an=an: usable(_an, ds[0], cams, rv))
        if pick is None:
            print(f"[dlc_frames] {an} {ep}: no session with templates on {'+'.join(cams)} -> "
                  f"THIS EPOCH WILL NOT BE LABELLED", flush=True)
            continue
        out.append((an, pick[0], pick[1], ep))
    return _thin_epochs(out, sessions_per_epoch())


def _thin_epochs(sessions, n_per_epoch: int):
    """Keep ``n_per_epoch`` sessions per epoch, spreading ANIMALS rather than taking them in order.

    Sorting by name and truncating would hand every epoch to PS92 and PS93 and leave PS95 out of the
    set entirely. Picking the least-used animal at each step instead keeps all four present and
    maximises the number of distinct epochs each one appears in -- which is the coverage that
    matters, since a network that never saw an animal after its stroke is the failure mode this
    module was written to avoid.
    """
    if not n_per_epoch:
        return sessions
    used: dict[str, int] = {}
    kept = []
    for ep in sorted({e for *_, e in sessions}):
        pool = sorted((a, d, s, e2) for a, d, s, e2 in sessions if e2 == ep)
        for _ in range(min(n_per_epoch, len(pool))):
            a, d, s, e2 = min(pool, key=lambda r: (used.get(r[0], 0), r[0]))
            pool.remove((a, d, s, e2))
            used[a] = used.get(a, 0) + 1
            kept.append((a, d, s, e2))
    dropped = len(sessions) - len(kept)
    print(f"[dlc_frames] {n_per_epoch} session(s) per epoch: kept {len(kept)}, dropped {dropped}"
          f"   animals {dict(sorted(used.items()))}", flush=True)
    return sorted(kept)


def date_sessions(date: str, rv=None, animals=None) -> list[tuple[str, str, str, str]]:
    """Every session on one date, as ``[(animal, date, sid, epoch), ...]``."""
    from wfield_local import epochs

    rv = rv or PathResolver()
    base = Path(rv.root("behavior_out")) / "sessions"
    want = set(config.normalize_animals(animals) or [])
    out = []
    for an_dir in sorted(p for p in base.glob("PS*") if p.is_dir()):
        an = an_dir.name
        if (want and an not in want) or not (an_dir / date).is_dir():
            continue
        ep = epochs.epoch_of(f"{an}_{date[4:]}")
        if ep is None:
            print(f"[dlc_frames] {an} {date}: no epoch -> skip", flush=True)
            continue
        for t in sorted((an_dir / date).glob("*_trials.csv")):
            out.append((an, date, t.name[: -len("_trials.csv")], ep))
    return out


def labelled_sessions(rv=None) -> set:
    """``{(animal, date)}`` already in the labelling set, from the manifest that recorded them.

    THE SESSION CHOICE DRIFTS, and that is the expensive kind of surprise. `cohort_sessions` takes
    the MIDDLE session of each animal x epoch, so it moves as sessions accumulate -- and the epoch
    boundaries move underneath it too (DECISIONS 2026-09-19 and 2026-09-21 both retuned
    `epochs.chronic`; the second shifted PS95's chronic start from day 15 to 11). Measured
    2026-09-22: re-running `--cohort` today plans 16 sessions of which only 11 are the ones already
    labelled, and would additionally extract five sessions nobody has opened.

    Nothing is LOST by that -- `extract` only writes new images and `write_manifest` appends,
    de-duplicated -- so the cost is scope, not data. But "add a few more frames to what she is
    already labelling" is a different request from "re-decide which sessions the cohort uses", and
    only this flag expresses the first. `dlc.frames.seed` keeps the choice WITHIN a session stable;
    this keeps the choice OF sessions stable, which is the part the seed cannot protect.
    """
    import csv

    man = staging_root(rv) / "frame_manifest.csv"
    if not man.is_file():
        raise SystemExit(f"No manifest at {man} -- nothing has been extracted yet, so there is no "
                         f"labelling set to grow. Run without --only-labelled.")
    with open(man, newline="", encoding="utf-8") as fh:
        return {(r["animal"], r["date"]) for r in csv.DictReader(fh)}


def run(date=None, cohort=False, cams=None, rv=None, animals=None, dry=False,
        only_labelled=False) -> list[dict]:
    rv = rv or PathResolver()
    cams = list(cams or cameras())
    sessions = (cohort_sessions(rv, animals, cams) if cohort
                else date_sessions(date, rv, animals))
    if only_labelled:
        # REPLACE the session list rather than intersecting with it. Intersecting sounds safer and
        # is worse: today's cohort pick and the labelled set overlap on only 11 of 15 sessions
        # (2026-09-22), so an intersection grows 11 folders and leaves 4 behind -- one per animal,
        # all late-epoch -- quietly skewing the additions away from chronic.
        keep = labelled_sessions(rv)
        by_pair = {(x[0], x[1]): x for x in sessions}
        planned = set(by_pair)
        for an, d in sorted(keep - planned):
            found = [x for x in date_sessions(d, rv, [an]) if x[0] == an]
            if len(found) == 1:
                by_pair[(an, d)] = found[0]
            else:
                print(f"    WARNING {an} {d} is in the labelling set but resolves to "
                      f"{len(found)} sessions -- skipped; add it by date instead", flush=True)
        dropped = sorted(planned - keep)
        sessions = [by_pair[k] for k in sorted(keep) if k in by_pair]
        print(f"[dlc_frames] --only-labelled: {len(sessions)} session(s) pinned to the existing "
              f"labelling set; {len(dropped)} newly-selected session(s) skipped", flush=True)
        for an, d in dropped:
            print(f"    skip (not in the labelling set): {an} {d}", flush=True)
    if not sessions:
        print("[dlc_frames] no sessions matched", flush=True)
        return []
    rows = []
    for cam in (cams or cameras()):
        for animal, d, sid, epoch in sessions:
            rows += plan_session(animal, d, sid, epoch, cam, rv)
    if not rows:
        return rows
    if target_per_session():
        before = len(rows)
        rows = prune_by_appearance(rows, rv)
        print(f"[dlc_frames] appearance pruning: {before} candidates -> {len(rows)} frames",
              flush=True)
    span = sorted({(r["animal"], r["epoch"]) for r in rows})
    print(f"[dlc_frames] {len(rows)} frames from {len(sessions)} session(s), "
          f"{len({r['cam'] for r in rows})} camera(s), {len(span)} animal-epoch cell(s)", flush=True)
    if dry:
        for r in rows[:10]:
            print(f"  [dry] {r['animal']} {r['date']} {r['cam']} {r['epoch']} "
                  f"{r['position']}/{r['category']}/{r['phase']} frame {r['frame']}", flush=True)
        if len(rows) > 10:
            print(f"  [dry] ... and {len(rows) - 10} more", flush=True)
        return rows
    written = extract(rows, rv)
    manifest = write_manifest(rows, rv)
    print(f"[dlc_frames] wrote {written} new image(s); manifest -> {manifest}", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", metavar="YYYYMMDD", nargs="?", default=None)
    ap.add_argument("--cohort", action="store_true",
                    help="one session per animal x epoch instead of one date")
    ap.add_argument("--cam", action="append", default=None,
                    help="camera to extract (repeatable; default: dlc.cameras from defaults.yaml)")
    ap.add_argument("--animals", default=None)
    ap.add_argument("--only-labelled", action="store_true",
                    help="restrict to sessions already in the manifest: GROW the existing "
                         "labelling set instead of re-deciding which sessions to use")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    if not args.cohort and not args.date:
        ap.error("give a date or --cohort")
    rv = PathResolver(machine=args.machine)
    run(args.date, args.cohort, args.cam, rv, args.animals, args.dry_run,
        args.only_labelled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

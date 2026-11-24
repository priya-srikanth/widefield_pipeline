"""THE lick-source wording for captions and slide notes (Priya 11-17: "ensure all figure captions / slide notes are
clear about what data they used -- I don't know if we specify everywhere cam4 vs cam4 + cam1 LP licks").

One phrase per lick train, so every caption names the same train the same way. What each train IS (from the code,
docs/status/LICK_SOURCE_AUDIT_20261117.md section 1):

  DAQ            the spout-contact sensor (`config.lick_detector`, run-tag part ``daqlk``).
  LP_POST        `lp_licks.lp_lick_times`: every cam4 LP lick of ``licks_all.csv`` + the cam1-only events of
                 ``cam1_licks/<a>_<d>_events.csv`` (not cam4-matched, water droplets removed). That cam1 file holds
                 POST-CUE events only (cue .. trial stop, `cam1_licks.detect(span="post")`), so cam1 adds licks to the
                 post-cue lick / no-lick call, the attempt classes and the first-lick time -- never to a pre-cue
                 window.
  LP_PRECUE      the same train read before the cue (the 1 s ENL lick-free window under WIDEFIELD_LICK_SOURCE=lp,
                 `lp_licks.gate_licks_s`): in practice cam4 only, for the reason above.
  CAM4           a cam4-only set: ITI licks (``licks_all.csv`` context iti), the per-lick kinematics and 3-D tables
                 (cam4 lick events; 3-D position from cam1 + cam4 pose), FaceRhythm lick windows.
  CAMERA_CLASSES `trial_classes`: contact = DAQ; incomplete = an LP lick (cam4, or cam1-only post-cue, droplets
                 removed) or a cam1 tongue without contact.
  RESTnl         `restnl_mask`: the sources its mask declares (`restnl_sources`), read from that module so the
                 wording follows the mask actually in the code.

Sessions without behaviour video (PS93_0813) have no LP licks and no camera classes; every curated cohort-1 session
other than that one has both cam4 and cam1 lick events (checked 11-17: 120 curated, 1 without).
"""
from __future__ import annotations

DAQ = "DAQ spout contact"
LP_POST = "LP video, cam4 + cam1 (cam1 droplets removed)"
LP_PRECUE = "LP video, cam4 (cam1 is scored only from the cue to the trial stop)"
CAM4 = "LP video, cam4 only"
CAMERA_CLASSES = "camera trial classes (DAQ spout contact + LP video, cam4 + cam1, cam1 droplets removed)"
LP_ALIGN = "first LP lick (close spouts: DAQ contact when the LP lick is missing or late)"
NO_VIDEO = "PS93_0813 has no video (no LP licks, no camera classes)"
#: FOUR-CAMERA wording (11-22 adoption, `session_licks.FUSED_DEFAULT`): the fused cam1 + cam2 + cam3 + cam4 train (60 ms
#: video-lick floor) covers the whole session, so the pre-cue window reads the same train; cam1-only events are not
#: licks (Priya 11-22); in the trial classes a trial with only cam1-only licks is jaw-only. Opt-in
#: (``licks_line(..., fourcam=True)``) for figures rendered on the four-camera inputs; the constants above stay the
#: wording of the cam4 + cam1 renders (v36) the deck still cites.
FOURCAM_POST = "four-camera video licks (fused cam1 + cam2 + cam3 + cam4 train; cam1-only events are not licks)"
FOURCAM_PRECUE = "four-camera video licks (the same fused train, whole session)"
FOURCAM_CLASSES = ("camera trial classes (DAQ spout contact + four-camera video licks; a trial whose only video licks "
                   "are cam1-only = tongue at the lips, grouped with jaw-only)")

_NAMES = {"daq": "DAQ", "cam4": "cam4", "cam1": "cam1"}


def restnl_sources() -> tuple[str, ...]:
    """The lick trains the RESTnl rest mask removes, as declared by `restnl_mask` (``LICK_SOURCES`` when the module
    defines it, else read from its ``LICK_SOURCE_TEXT``). Never hard-coded here: the mask moves to DAQ + cam4 + cam1
    after the 11-17 re-run (Priya 11-17), and the wording must follow the mask the outputs were made with."""
    from wfield_local import restnl_mask as rn
    src = getattr(rn, "LICK_SOURCES", None)
    if src:
        return tuple(str(s).lower() for s in src)
    text = str(rn.LICK_SOURCE_TEXT).lower()
    out = []
    if "daq" in text:
        out.append("daq")
    if "cam4" in text:
        out.append("cam4")
    if "cam1" in text:
        out.append("cam1")
    if not out:
        raise ValueError(f"restnl_mask.LICK_SOURCE_TEXT names no lick train: {rn.LICK_SOURCE_TEXT!r}")
    return tuple(out)


def restnl_text() -> str:
    """'rest licks excluded: DAQ + cam4' (or '+ cam1' once the mask includes it)."""
    return "rest licks excluded: " + " + ".join(_NAMES.get(s, s) for s in restnl_sources())


def licks_line(source: str, align: str | None = None, camera_gate: bool = False, fourcam: bool = False) -> str:
    """The 'licks:' clause for a figure drawn under ``source`` ("daq" | "lp") at ``align`` ("precue" | "cue" | "lick"
    | None). ``camera_gate`` = the run selects working / miss trials by `trial_classes` (WIDEFIELD_TRIAL_GATE=camera):
    those classes then use the video even in a DAQ-source run (the lick class stays DAQ contact there). ``fourcam`` =
    a figure rendered on the fused four-camera train (FOURCAM_* wording)."""
    src = str(source).strip().lower()
    post, precue, classes = (FOURCAM_POST, FOURCAM_PRECUE, FOURCAM_CLASSES) if fourcam else (LP_POST, LP_PRECUE,
                                                                                               CAMERA_CLASSES)
    if src == "lp":
        bits = [f"licks: {post}", f"lick alignment = {LP_ALIGN}" if align == "lick" else ""]
        if align == "precue":
            bits.append(f"pre-cue lick-free window: {precue}")
        bits.append("PS93_0813 (no video) is not in these outputs")
    elif src == "daq":
        bits = [f"licks: {DAQ}"]
        if align == "precue":
            bits.append(f"pre-cue lick-free window: {DAQ}")
        if camera_gate:
            bits.append(f"working / miss trials: {classes}; PS93_0813 (no video) has no camera classes")
    else:
        raise ValueError(f"lick source {source!r}; expected 'daq' or 'lp'")
    return "; ".join(b for b in bits if b)

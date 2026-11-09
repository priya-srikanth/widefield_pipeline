"""Shared plumbing for the recreated results-page figure scripts in `scripts/rebuild_figs/` (cohort-1 rebuild, 11-03).

Every script here follows one contract:
  --out DIR     where the PNG (+ sidecar CSVs) go; the plotting inputs are pickled to DIR/data/<name>_figdata.pkl.
                Default = $WF_REBUILD_FIGS_OUT, else <system temp>/wf_rebuild_figs -- a NEW folder, never a page,
                deck or share folder (the originals wrote into presentations/_source_PI_overview_2026-11-02/
                build_inputs/ or a session scratchpad; the rebuild publishes by an explicit copy step instead).
  --plot-only   redraw from DIR/data/<name>_figdata.pkl without loading any data (same draw function, so the PNG
                is pixel-identical to the full run's).
Run from the repo root with the repo on the path, e.g.
    PYTHONPATH=. python scripts/rebuild_figs/fig_behaviour_hits.py --out <dir>
"""
from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path

def default_out() -> Path:
    """$WF_REBUILD_FIGS_OUT, else <rebuild_root>/rebuild_figs/<run tag> under a rebuild root (coordinator 11-04: written
    under the run tag, so arms / blocks never share a folder), else <system temp>/wf_rebuild_figs."""
    if os.environ.get("WF_REBUILD_FIGS_OUT"):
        return Path(os.environ["WF_REBUILD_FIGS_OUT"])
    if os.environ.get("WIDEFIELD_REBUILD_ROOT", "").strip():
        from wfield_local import run_mode
        return run_mode.subdir(Path(os.environ["WIDEFIELD_REBUILD_ROOT"]) / "rebuild_figs")
    return Path(tempfile.gettempdir()) / "wf_rebuild_figs"


DEFAULT_OUT = default_out()


# R3 (fourth review, 11-03): DEFAULT INPUTS FOLLOW THE RUN. Inputs made by a routed writer default to that writer's
# folder for this run (inside the rebuild root under the profile); inputs with no routed writer (a dated backup, a
# one-off diagnostic cache) are refused in a non-default run unless given explicitly.
def nondefault_run() -> str | None:
    """The run's full output tag, or "rebuild-root" (None = a default run)."""
    from wfield_local import lp_licks, rebuild_root
    return lp_licks.output_tag() or ("rebuild-root" if rebuild_root.active() else None)


def epoch_data_dir() -> Path:
    """`grant_figures/epoch/data` of THIS run (`lp_licks.output_dir`: the tag folder; the rebuild root's under one)."""
    from wfield_local import config, lp_licks
    return lp_licks.output_dir(Path(config.resolver().root("cohort_out")) / "grant_figures" / "epoch") / "data"


def require_explicit(value, default, flag: str, *, ok_in_rebuild: bool = False) -> None:
    """SystemExit when ``value`` is the unrouted ``default`` in a non-default run (R3). ``ok_in_rebuild``: the default
    resolves inside the rebuild root (its writer writes there), so it is fine under one."""
    from wfield_local import rebuild_root
    mode = nondefault_run()
    if ok_in_rebuild and rebuild_root.active():
        return
    if mode and Path(value) == Path(default):
        raise SystemExit(f"{flag} defaults to a pre-rebuild input with no routed writer; in run {mode!r} pass "
                         f"{flag} explicitly (or run without the render profile)")


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output folder (default {DEFAULT_OUT}); figdata goes to <out>/data/")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw from <out>/data/<name>_figdata.pkl; no data load")
    return ap


def figdata_path(out: Path, name: str) -> Path:
    return Path(out) / "data" / f"{name}_figdata.pkl"


def save_figdata(out: Path, name: str, obj) -> Path:
    """Atomic write (tmp + replace) of everything the draw function needs."""
    p = figdata_path(out, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    tmp.replace(p)
    return p


def load_figdata(out: Path, name: str):
    p = figdata_path(out, name)
    if not p.exists():
        raise SystemExit(f"--plot-only: no figure data at {p}; run once without --plot-only")
    return pickle.loads(p.read_bytes())


# ----------------------------------------------------------------------------------------------- trial-set captions
# Priya 11-09: "EVERY figure must state EXACTLY which trials were included / excluded". Each driver here READS a table
# another family wrote, so its sentence is the PRODUCER's trial set, composed from the producer's constants and the
# run's switches (`wfield_local.trial_set_text`). A driver declares it in main() after argparse (the --plot-only replot
# is covered too); a driver whose PNGs draw different sets saves each through `save_with` / `tst.active`.
def declare(text: str) -> None:
    """The driver-level trial set: every figure this process saves carries ``text``."""
    from wfield_local import trial_set_text as tst
    tst.declare(text=text)


def save_with(text: str, fig, png, **kw):
    """`figure_layout.save_png_svg` with ``text`` as this one figure's trial set (wins over the driver's declaration)."""
    from wfield_local import trial_set_text as tst
    from wfield_local.figure_layout import save_png_svg
    with tst.active(None, text=text):
        return save_png_svg(fig, png, **kw)


def _split(sentence: str) -> tuple[str, str]:
    """``("<included>", "<excluded>")`` of a trial_set_text sentence (its "<unit>: " prefix dropped)."""
    body = sentence.split(": ", 1)[1] if ": " in sentence else sentence
    inc, _, exc = body.partition("; excluded: ")
    return inc, exc


def encoder_gate() -> str:
    """The encoders' trial gate (`orofacial_cohort.engagement_keep`, env OROFACIAL_GATE, default "reference"): NOT the
    widefield trial gate. Read from THIS process's env -- the producer's run / session json records the gate it used."""
    return os.environ.get("OROFACIAL_GATE", "reference")


def encoder_trials() -> str:
    """The trials an encoder (frozen / combined / allsession) is fitted on, in words, under `encoder_gate`."""
    from wfield_local import trial_classes as tc
    g = encoder_gate()
    if g == "reference":
        t = "behavior-pipeline engaged trials (reference gate: spout_behavior's per-trial engaged column)"
    else:
        if tc.camera_gate_on():
            from wfield_local import trial_set_text as tst
            inc, exc = _split(tst.sentence("working"))
            att = f"{inc}; not: {exc}, early-run" if exc else inc
        else:
            att = "trial_attempts.csv attempted = hit + incomplete"
        t = f"attempted trials ({att})" + (", inside the canonical period" if g == "canonical_attempted" else "")
    return f"{t} [encoder gate OROFACIAL_GATE={g}]"


#: the frames an encoder fit reads per trial (`frozen_encoder` / `combined_encoding`: strobe - 0.5 s .. stop + 1 s)
ENCODER_WINDOW = "imaging frames from spout strobe - 0.5 s to trial stop + 1 s"


def encoder_frames(extra=()) -> str:
    """Frame set of an all-frames encoder readout (Layer 2, random splits, the contest R2)."""
    from wfield_local import trial_set_text as tst
    return tst.frames(f"{ENCODER_WINDOW} of {encoder_trials()}", keep=extra,
                      excluded=["inter-trial frames outside those windows", "trials outside the encoder gate"])


#: the first post-cue lick the frozen encoder's lick window aligns on (`combined_encoding.trial_first_lick`, split)
FIRST_LICK = ("the trial's first post-cue lick (first cam4 LP or cam1-only onset from the cue to the trial stop; at a "
              "close spout the first DAQ contact when the LP onset is missing or > 50 ms later, at a far spout only "
              "when it is missing)")


def evoked_trials(spouts: str) -> str:
    """Trial set of the frozen-encoder EVOKED residual maps (`frozen_encoder.EVOKED_WINDOWS`) at ``spouts``."""
    from wfield_local import trial_set_text as tst
    from wfield_local.frozen_encoder import EVOKED_WINDOWS as W
    (cr0, cr1), (cb0, cb1) = W["cue"]
    (lr0, lr1), (lb0, lb1) = W["lick"]
    return tst.events(
        f"{encoder_trials()} at {spouts}", unit="trials",
        keep=[f"cue window: every such trial, cue-aligned, {cr0:g}..{cr1:g} s minus {cb0:g}..{cb1:g} s",
              f"lick window: the trials with {FIRST_LICK}, aligned on it, {lr0:g}..{lr1:g} s minus {lb0:g}..{lb1:g} s",
              "June pre sessions dropped"],
        excluded=["lick window: trials with no post-cue lick", "trials outside the encoder gate", "other spouts"])


def attempt_classes(*, epochs: str, windows: str, min_n: int | None = None) -> str:
    """Trial set of the attempt_coding far-contra class rows (`wfield_local.attempt_coding.classify`) under the run's
    switches: pre = pre-stroke hits; post = hit / incomplete / no attempt / reaction / stopped."""
    from wfield_local import trial_classes as tc
    from wfield_local import trial_set_text as tst
    try:
        rt = f"{tc.max_rt_ms() / 1000:g} s"
    except Exception:                                                   # noqa: BLE001 - a caption never kills a figure
        rt = "max_rt"
    lk = tst.lick_text()
    cls = ("pre = pre-stroke hits (camera contact; LOSO frozen decoder); "
           f"{epochs}: hit (camera contact, a lick within {rt}), incomplete (camera incomplete, a lick within {rt}), "
           f"no attempt (no lick, camera no response inside the working period), reaction (no lick, camera jaw-only "
           f"inside the working period), stopped (no lick, camera stopped: {tst.stop_text()} runs or the quit tail)")
    keep = [cls, windows] + ([f"a class is drawn only with >= {min_n} trials per animal" ] if min_n else [])
    return tst.events(f"far-contra trials ({lk}; camera classes of the run's trial_attempts table)", unit="trials",
                      keep=keep, excluded=["pre-stroke non-hits", "late (no-lick rows of camera contact / incomplete)",
                                           "stopped-licking", "early-run", "after-end", "unknown", "other spouts"])


#: the attempt_coding windows (`lp_licks.post_s_for`; cue and lick 2 s, the pre-cue ENL window lick-free)
ATTEMPT_WINDOWS = {"precue": "pre-cue window: the lick-free ENL window ending at the cue (a trial without one dropped)",
                   "cue": "cue window: cue-aligned",
                   "lick": "lick window: lick rows on their first lick, no-lick classes at the session's per-spout "
                           "median reaction time"}


def windows_text(windows) -> str:
    return "; ".join(ATTEMPT_WINDOWS[w] for w in windows)


def usable_licks() -> str:
    """Event set of the binned-contest executed bins (`combined_encoding.exec_bins`: every usable cam4 post-cue lick)."""
    from wfield_local import trial_set_text as tst
    return tst.events("every usable cam4 post-cue LP lick of licks.csv (cue + 70 ms .. trial stop) on every cue-table "
                      "trial: not peak-interpolated, protrusion >= MIN_REACH_PX, finite angle, a 3-D peak", unit="licks",
                      keep=["no engaged gate, no camera class or working-period cut"],
                      excluded=["cam1-only licks", "pre-cue / ITI licks", "licks within 70 ms of the cue"])


def contest_frames() -> str:
    """Frame set of the contest R2 (`combined_encoding` region measures): encoder fit frames, cue and lick windows."""
    return encoder_frames(extra=["cue window: cue .. cue + 1 s", "lick window: frames within 0.2 s of any cam4 lick peak",
                                 "the target / executed bin kernels use the usable cam4 licks only"])


def working_trials(extra=()) -> str:
    """The camera WORKING class rows of an epoch sidecar (`grant_kit._class_select` -> `trial_classes.camera_select`)."""
    from wfield_local import trial_set_text as tst
    inc, exc = _split(tst.sentence("working"))
    return tst.events(inc, unit="trials", keep=extra, excluded=exc or None)


def working_period_trials(extra=(), what: str | None = None) -> str:
    """Every trial of the camera working period (`trial_attempts` canonical rows)."""
    from wfield_local import trial_set_text as tst
    try:
        inc, exc = _split(tst.sentence("working_period"))
    except ValueError:                                                  # no camera trial gate in THIS process
        inc, exc = (f"the camera working period of the run's trial_attempts table (this process: {tst.gate_text()}; "
                    "the table is refused unless it was built under the camera trial gate)"), ""
    return tst.events(what or inc, unit="trials", keep=([inc] if what else []) + list(extra),
                      excluded=[exc, "early-run"] if exc else ["early-run"])


#: the behavioral-state decoder's unit (`locomotor_features` / `locomotor_state.three_way_segments`)
STATE_SEGMENTS = ("1 s segments of rest (behavior_events quiet periods: docked, between trials, no running or licking), "
                  "running (treadmill bouts) and licking (1 s from each cue's first DAQ lick within the engaged cut); at "
                  "most 8 segments per period; no engagement or camera trial gate")


def state_segments() -> str:
    from wfield_local import trial_set_text as tst
    return tst.frames(STATE_SEGMENTS, excluded=["running / licking overlaps", "segments outside imaging coverage"])

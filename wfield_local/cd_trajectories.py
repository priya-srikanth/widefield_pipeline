"""Per-position coding directions projected FRAME BY FRAME -- the CD trajectory.

Priya, 2026-09-24: *"In my 2pRAM_pipeline I've been doing a lot of CD analyses and plotting the CD
for different trial types/epochs... build a CD for each lick direction and plot all [trials] for each
spout position trial type aligned to lick? same for ENL etc"* -- and on how it is done there:
*"I think we're getting scalar CD but then plotting the projections onto the CD over time."*

Exactly that. `position_coding_directions` already fits per-position directions on the per-animal
frozen joint basis and reports a SCALAR per trial per window. This module reuses its `direction`,
`poles` and `project` unchanged and adds the only missing step: projecting the signal at EVERY FRAME
onto one fixed direction, so a trial becomes a time course instead of a number.

WHY A SEPARATE DIRECTION HAS TO BE FITTED, AND WHY THAT IS NOT A COMPROMISE.
`position_coding_directions` fits in (component x TIME SUB-BIN) space -- its own comment: "a
direction is a weight per component PER MOMENT in the window", 90 x 4 = 360 for an ENL window. Such
a vector CANNOT be applied to a single frame, and collapsing it would misrepresent what the
sub-binned classifier learnt. More importantly, a direction with time structure and a trajectory are
redundant: projecting frame t with weights fitted for a different moment is not a coherent readout.
**A trajectory requires a time-INVARIANT direction**, so this module fits on the window MEAN
(``bins=1``) and lets all the time structure live in the projection, which is the standard
construction and the one the 2p pipeline uses.

The cost is small and already measured. `configs/defaults.yaml` records pre-cue sub-binning as
UNESTABLISHED: "+0.009, better in 23/44 -- a coin flip, NOT the +0.032 the 16-session pilot
reported... `precue: 4` is kept only because changing it would move every pre-cue number again for
no demonstrated gain." So for the ENL window a mean-feature direction gives up essentially nothing.
Post-lick is where width matters (0.25 s wins, +0.009, 12/16), so read LICK-aligned trajectories
knowing their direction is blunter than the sub-binned one scored elsewhere.

NOTHING EXISTING IS REPLACED. `position_coding_directions` keeps its four sub-binned methods; this
is a fifth object beside them, in the spirit of that module's own note that "_orth variants are the
SAME construction with the engagement axis projected out. Kept as separate methods rather than
replacing the originals, so both are on disk and comparable."

THE DIRECTION IS DEFINED WHERE BEHAVIOUR IS INTACT: PRE-STROKE trials with a successful lick, P
against not-P, matching `position_coding_directions` exactly. Pole-normalised so 0 = pre-stroke
not-P and 1 = pre-stroke lick at P; every trajectory then reads as "fraction of the normal
position-P signature" and is comparable across positions, animals and epochs.

WHAT EACH ALIGNMENT CAN AND CANNOT SAY -- inherited from `position_coding_directions`, unchanged:

  precue (ENL)  the clean one. The window is lick-free by construction (`decode.precue_lickfree`).
  cue           a lick trial contains its lick from ~140 ms, so there is NO movement-free cue
                window to retreat to. Per-position construction is what keeps it interpretable:
                movement is common to every training class, so it cannot define the direction.
  lick          NO-LICK CLASSES HAVE NO LICK TO ALIGN TO and are therefore NOT PLOTTED here. The
                static module can place them at an inferred time (cue + that session's median RT at
                that position); a TRAJECTORY through an inferred time would draw a curve whose
                x-axis is a guess, one that grows with the latency, and post-stroke the latency is
                long and variable. Reporting a number with a caveat is defensible; drawing a
                time course through it is not.

AND THE ONE THING WIDEFIELD CANNOT DO THAT 2p CAN. The haemodynamic response is slow -- seconds
wide -- so these trajectories are that kernel convolved with whatever the underlying signal is.
Onset times, ramp slopes and the ORDER of two nearby events are NOT readable from them the way they
are from a 2p CD trajectory. Read amplitude and gross time course; do not read timing.

CLI::

    python -m wfield_local.cd_trajectories --animal PS95 --align precue
    python -m wfield_local.cd_trajectories --align precue cue --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import position_coding_directions as pcd
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER

#: Seconds before/after the alignment event to draw. The ENL window is 2 s and the spout arrives
#: ~3 s before the cue, so `precue` reaches back past spout arrival on purpose: the trajectory
#: should show the code appearing, not start after it has.
SPAN = {"precue": (-3.5, 1.0), "cue": (-3.0, 2.0), "lick": (-3.0, 2.0)}

#: Classes drawn per panel. `stopped` is included but is the scarce one everywhere
#: (pre-stroke 6/40/326/495), so `min_trials` marks rather than drops it.
CLASSES = ("success", "miss_working", "stopped")

#: A class with fewer than this many trials in a cell is drawn DASHED and labelled, never omitted --
#: "could not test" and "tested and found nothing" are different facts (the rule this whole ENL arm
#: follows).
MIN_TRIALS = 10

#: Lick-aligned no-lick classes are not drawn at all. See the module docstring: their alignment time
#: is an inference, and a trajectory through an inferred time has a guessed x-axis.
LICK_ALIGNED_CLASSES = ("success",)

#: Boxcar applied to the projected CD time course before slicing trials, in seconds.
#:
#: NOT cosmetic, and it costs no information that is there. The poles are calibrated on the WINDOW
#: MEAN -- ~62 frames for a 2 s window -- so a single-frame projection carries roughly sqrt(62) ~ 8x
#: that variance while being read on the same 0-1 scale. MEASURED on PS95's joint basis: per-frame
#: sd is 0.04-0.08 but the extremes reach +/-4 to +/-6, about 100 sd, so one artefact frame in one
#: trial moves a 50-trial mean by tens of normalised units. The first render ran to +/-150 on an
#: axis whose meaningful range is 0-1.
#:
#: 0.5 s is well inside the haemodynamic kernel this signal is already convolved with, which the
#: module docstring says outright is why onset and ordering must not be read from these curves. So
#: the smoothing removes variance the measurement never resolved.
SMOOTH_S = 0.5

#: Trials are aggregated by MEDIAN with a 25-75 band, not mean +/- SEM. Two reasons, both measured:
#: the rare large artefacts above, and a 2x spread in overall signal scale BETWEEN SESSIONS
#: (PS95_0606 sd 0.0756 vs PS95_0607 0.0377) while the poles are fitted on pooled sessions -- so a
#: high-scale session's trials sit systematically high and a mean tracks which sessions happened to
#: contribute. A median does not.
BAND = (25, 75)


def smooth(v, fs, seconds=SMOOTH_S):
    """Centred boxcar over the CD time course. Edges shrink the window rather than pad it."""
    n = max(1, int(round(seconds * fs)))
    if n < 2:
        return np.asarray(v, float)
    v = np.asarray(v, float)
    k = np.ones(n)
    num = np.convolve(v, k, mode="same")
    den = np.convolve(np.ones_like(v), k, mode="same")     # shrinks at the edges, no padding bias
    return num / den


def span_frames(align, fs):
    """``(pre_n, post_n)`` frame counts for the drawn span."""
    lo, hi = SPAN[align]
    return int(round(-lo * fs)), int(round(hi * fs))


def window_means(sig, ref0s, post_n):
    """``(n_trials, ncomp)`` -- the WINDOW MEAN per component, i.e. a ``bins=1`` feature.

    This is what makes the direction time-invariant and therefore projectable frame by frame. It is
    deliberately NOT `_window_feature(..., bins=4)`, which returns (component x sub-bin).
    """
    out = np.full((len(ref0s), sig.shape[0]), np.nan)
    for i, r in enumerate(ref0s):
        if r is None or r < 0 or r + post_n > sig.shape[1]:
            continue
        out[i] = sig[:, int(r):int(r) + post_n].mean(1)
    return out


def fit_directions(X, y, labels, method="dom"):
    """``{position: (w, p0, p1)}`` from PRE-STROKE successful-lick trials, P against not-P.

    `pcd.direction` and `pcd.poles` unchanged -- they are basis-agnostic, so handing them a
    ``bins=1`` feature gives a weight per COMPONENT rather than per (component, sub-bin), which is
    the whole point. Positions with too few trials on either side are omitted and reported.
    """
    out = {}
    X, y = np.asarray(X, float), np.asarray(y)
    ok = np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    for p in labels:
        m = y == p
        if m.sum() < MIN_TRIALS or (~m).sum() < MIN_TRIALS:
            continue
        w = pcd.direction(X[m], X[~m], method=method)
        p0, p1 = pcd.poles(X[m], X[~m], w)
        out[int(p)] = (w, p0, p1)
    return out


def trajectory(sig, align_f, w, p0, p1, pre_n, post_n, fs=None):
    """``(n_trials, pre_n + post_n)`` pole-normalised projections, one row per trial.

    The whole signal is projected ONCE (``w @ sig``, a single matvec over the session) and trials
    are then sliced out of it. Projecting per trial would repeat the same multiply for every
    overlapping window.
    """
    v = np.asarray(w) @ np.asarray(sig)                       # (T,) the session's own CD time course
    if fs:
        v = smooth(v, fs)                                     # before normalising; linear either way
    d = p1 - p0
    v = (v - p0) / d if abs(d) > 1e-12 else v - p0
    out = np.full((len(align_f), pre_n + post_n), np.nan)
    for i, f in enumerate(align_f):
        if f is None or not np.isfinite(f):
            continue
        a, b = int(f) - pre_n, int(f) + post_n
        if a < 0 or b > v.size:
            continue
        out[i] = v[a:b]
    return out


def session_arms(s, args, basis, align):
    """``(sig, {class: {"fit": ref0s, "at": align_frames, "y": codes}})`` for one session.

    THE TRIAL SET COMES FROM `nolick_decoder.categorize`, not from a classification written here.
    That function already defines `engaged` / `late_rewarded` / `undetected` and the per-session
    engagement gate, and `enl_decode` splits `undetected` by it into `miss_working` / `stopped`.
    Re-deriving any of that locally is how `enl_state_counts` ended up with a 2.0 s response window
    against the decoder's 3.5 s and reported 67 stopped trials where the decode saw 40 (rule 9).

    TWO FRAME SETS PER TRIAL, and they are different things:

      ``fit``  where the direction's feature window STARTS -- `precue_window_start` for the ENL
               (slid to a lick-free gap, or None meaning drop), the cue otherwise.
      ``at``   where the trajectory is CENTRED. Always a real observed event: the CUE for `precue`
               and `cue`, the FIRST LICK for `lick`. A slid feature window does not move it, so
               panels stay on a common x-axis.

    So the align token chooses WHICH WINDOW THE DIRECTION IS FITTED ON; `precue` and `cue` are both
    drawn against the cue and differ only in that.
    """
    from wfield_local.locanmf_position_decoder import precue_window_start
    from wfield_local.nolick_decoder import _joint_signal, categorize

    sig = _joint_signal(basis, s)[0] if basis is not None else None
    if sig is None:
        raise ValueError("cd_trajectories requires the per-animal frozen joint basis")
    codes, cat, _blk, _rt_s, cue_f, sess_eng, ls, strobe_f = categorize(s, args, with_licks=True)
    post_n = int(round(args.post_s * args.fs))
    lickfree = bool(args.align == "precue")

    j = np.searchsorted(ls, cue_f, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1)

    out = {c: {"fit": [], "at": [], "y": []} for c in CLASSES}
    for k in range(cue_f.size):
        if not cat[k] or codes[k] < 0 or int(cue_f[k]) < 0:
            continue
        if cat[k] == "engaged":
            cls = "success"
        elif cat[k] == "undetected":
            cls = "miss_working" if sess_eng[k] else "stopped"
        else:
            continue                                   # late_rewarded: a HIT, excluded as elsewhere
        c0 = int(cue_f[k])
        if align == "precue":
            ref0 = precue_window_start(c0, strobe_f[k], ls, post_n, lickfree=lickfree)
            if ref0 is None:
                continue                               # no lick-free window exists -> drop
        else:
            ref0 = c0
        at = c0 if align in ("precue", "cue") else (int(first[k]) if first[k] > 0 else None)
        if at is None:
            continue
        out[cls]["fit"].append(ref0)
        out[cls]["at"].append(at)
        out[cls]["y"].append(int(codes[k]))
    return sig, out


def analyse_animal(animal, align="precue", *, method="dom", post_s=2.0, verbose=True):
    """Per-position directions from PRE-STROKE success, and trajectories for every class x epoch."""
    from wfield_local import analysis_kit as ak
    from wfield_local import epochs, joint_locanmf
    from wfield_local.locanmf_frozen_decoder import _args

    basis = joint_locanmf.load(animal)
    args = _args(source="roi", align=align, post_s=post_s)
    pre_n, post_frames = span_frames(align, args.fs)
    # THE CURATED SET, in `load_sessions()` ORDER -- the same source `enl_decode.sessions_for` uses.
    # Iterating `config.load_sessions()` directly picks up sessions with no SVTcorr on disk, and
    # more importantly is a second definition of "which sessions this cohort is" (rule 9). Order is
    # preserved, not sorted: `curated_sessions` keeps it deliberately.
    sess = [s for s in ak.curated_sessions() if s["label"].startswith(animal)]

    per, errs = [], []
    for s in sess:
        lab = s["label"]
        try:
            sig, arms = session_arms(s, args, basis, align)
        except Exception as exc:
            errs.append(f"{lab}: {type(exc).__name__}: {exc}"[:140])
            continue
        per.append((lab, epochs.epoch_of(lab), sig, arms))
        if verbose:
            print(f"  {lab} [{epochs.epoch_of(lab)}] "
                  + " ".join(f"{c}={len(arms[c]['y'])}" for c in CLASSES), flush=True)

    # ---- the direction: PRE-STROKE SUCCESS only, window means, P vs not-P ----------------------
    Xf, yf = [], []
    for _lab, ep, sig, arms in per:
        if ep != "pre" or not arms["success"]["y"]:
            continue
        Xf.append(window_means(sig, arms["success"]["fit"], int(round(args.post_s * args.fs))))
        yf.append(np.asarray(arms["success"]["y"]))
    if not Xf:
        return {"animal": animal, "align": align, "skipped": "no pre-stroke success trials",
                "errors": errs}
    dirs = fit_directions(np.vstack(Xf), np.concatenate(yf), DISPLAY_ORDER, method=method)

    # ---- the trajectories -----------------------------------------------------------------------
    draw = LICK_ALIGNED_CLASSES if align == "lick" else CLASSES
    acc = {}
    for _lab, ep, sig, arms in per:
        for cls in draw:
            ys = np.asarray(arms[cls]["y"])
            if not ys.size:
                continue
            for p, (w, p0, p1) in dirs.items():
                m = ys == p
                if not m.any():
                    continue
                tr = trajectory(sig, np.asarray(arms[cls]["at"])[m], w, p0, p1,
                                pre_n, post_frames, fs=args.fs)
                acc.setdefault((ep, cls, p), []).append(tr)

    out = {"animal": animal, "align": align, "method": method,
           "basis_id": basis.basis_id, "ncomp": int(basis.ncomp),
           "fs": float(args.fs), "span": SPAN[align], "pre_n": pre_n, "post_n": post_frames,
           "positions": sorted(dirs), "n_sessions": len(per), "errors": errs,
           "drawn_classes": list(draw), "traces": {}}
    for key, chunks in acc.items():
        A = np.vstack(chunks)
        n = int(np.isfinite(A).any(1).sum())
        with np.errstate(invalid="ignore"):
            out["traces"][key] = {"mean": np.nanmedian(A, 0), "n": n,
                                  "lo": np.nanpercentile(A, BAND[0], axis=0),
                                  "hi": np.nanpercentile(A, BAND[1], axis=0)}
    return out


STYLE = {"success": ("tab:blue", "success (lick)"),
         "miss_working": ("tab:orange", "miss while working"),
         "stopped": ("tab:red", "stopped")}
EPOCHS = ("pre", "acute", "subacute", "chronic")


def figure(res, out):
    """Positions down, epochs across; one trace per class. x = 0 is the ALIGNMENT EVENT."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES

    pos = res["positions"]
    eps = [e for e in EPOCHS if any(k[0] == e for k in res["traces"])]
    if not pos or not eps:
        raise SystemExit("[cd_traj] nothing to draw")
    t = (np.arange(-res["pre_n"], res["post_n"]) / res["fs"])
    fig, axes = plt.subplots(len(pos), len(eps), figsize=(3.0 * len(eps), 1.9 * len(pos)),
                             squeeze=False, sharex=True, sharey=True, constrained_layout=True)
    for i, p in enumerate(pos):
        for j, ep in enumerate(eps):
            ax = axes[i][j]
            ax.axhline(0, color="0.8", lw=0.7)
            ax.axhline(1, color="0.8", lw=0.7, ls=":")
            ax.axvline(0, color="0.4", lw=0.8)
            for cls in res["drawn_classes"]:
                d = res["traces"].get((ep, cls, p))
                if d is None:
                    continue
                col, lab = STYLE[cls]
                # THIN CELLS ARE DASHED AND LABELLED, NEVER DROPPED
                thin = d["n"] < MIN_TRIALS
                ax.plot(t, d["mean"], color=col, lw=1.3, ls="--" if thin else "-",
                        label=f"{lab} (n={d['n']})" + (" THIN" if thin else ""))
                if not thin:
                    ax.fill_between(t, d["lo"], d["hi"], color=col, alpha=0.18, linewidth=0)
            if i == 0:
                ax.set_title(ep, fontsize=9)
            if j == 0:
                ax.set_ylabel(POSITION_NAMES.get(p, str(p)), fontsize=8)
            ax.legend(fontsize=5.5, frameon=False, loc="upper left")
            ax.tick_params(labelsize=7)
    zero = {"precue": "cue", "cue": "cue", "lick": "first lick"}[res["align"]]
    for ax in axes[-1]:
        ax.set_xlabel(f"s from {zero}", fontsize=8)
    fig.suptitle(
        f"{res['animal']} — projection onto the per-position {res['align'].upper()} coding "
        f"direction, frame by frame\n"
        f"direction fitted on PRE-STROKE SUCCESS (window mean, time-INVARIANT) · "
        f"0 = pre-stroke not-P, 1 = pre-stroke lick at P · basis {str(res['basis_id'])[:12]} "
        f"{res['ncomp']}c\n"
        f"MEDIAN across trials, band = {BAND[0]}-{BAND[1]}th pct · {SMOOTH_S:g}s boxcar · "
        f"HAEMODYNAMICS ARE SLOW — read amplitude and gross time course, NOT onset or ordering"
        + ("" if res["align"] != "lick" else
           " · no-lick classes omitted: their alignment time would be inferred"),
        fontsize=9)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animal", action="append", default=None)
    ap.add_argument("--align", nargs="+", default=["precue"],
                    choices=("precue", "cue", "lick"))
    ap.add_argument("--method", default="dom", choices=("dom", "lr"))
    ap.add_argument("--post-s", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.out:
        out = Path(args.out)
    else:
        from wfield_local.paths import PathResolver
        out = Path(PathResolver().root("figures_working"))
    for animal in (args.animal or ["PS92", "PS93", "PS94", "PS95"]):
        for align in args.align:
            try:
                res = analyse_animal(animal, align, method=args.method, post_s=args.post_s)
            except Exception as exc:
                print(f"[cd_traj] {animal} {align}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if res.get("skipped"):
                print(f"[cd_traj] {animal} {align}: SKIPPED {res['skipped']}", flush=True)
                continue
            p = figure(res, out / f"cd_traj_{animal}_{align}_{args.method}.png")
            print(f"[cd_traj] -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

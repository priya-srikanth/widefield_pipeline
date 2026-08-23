"""Decode intended spout position from cortical activity (LocaNMF components or SVD/Allen ROIs).

Baseline validation of the per-position model for the stroke study. Features = mean activity
in a window after the cue (default) or first lick, on ENGAGED trials (cue followed by a lick
within --max-rt; no-lick held aside). Multinomial logistic regression; reports accuracy vs
chance, confusion matrix, and per-area decoding (SSp / MO / all).

  --source locanmf : footprint-scaled dF/F per LocaNMF component (session-specific basis)
  --source roi     : SVD Allen-region ROI dF/F (mean_pixels(U_region) @ SVTcorr; atlas-anchored)
  --align cue|lick : feature window relative to cue (default) or first lick
  --baseline none|precue : per-trial pre-cue subtraction. DEFAULT none -- a per-trial pre-cue
      baseline over-subtracts genuine (anticipatory) position signal, and a *session-constant*
      baseline (e.g. quiet-period) is invisible to a standardized decoder anyway.
  --cv block|random : DEFAULT block. Spout positions are presented in BLOCKS (~6 trials), so
      random k-fold leaks each block's slow-drift fingerprint across train/test (trials in the
      same block are not independent) -- inflating accuracy, especially with no baseline.
      Block-aware CV (leave-whole-blocks-out) forces generalization to unseen blocks.

    python -m wfield_local.locanmf_position_decoder --date 0603 --source locanmf --output "<dir>"
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from wfield_local import config
from wfield_local import nolick_analysis as na
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.plot_lick_aligned_averages import _load_daq_events, POSITION_NAMES, DISPLAY_ORDER
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue_events, _classify_cues
from wfield_local.behavior_position import classify_cues_with_backup
from wfield_local.block_ids import block_ids, block_size_max_for
from wfield_local.locanmf_crossanimal_dff import _footprint_scale, _frames
from wfield_local.figgrid import blank_unused, grid_shape


def _build_signal(s, source):
    """Return (signal [nfeat,T], feat_region_labels [nfeat])."""
    mc = s["mc"]
    if source == "locanmf":
        C = np.load(f"{config.locanmf_dir(mc)}/{s['label']}_locanmf_C.npy").astype(np.float64)
        reg = np.load(f"{config.locanmf_dir(mc)}/{s['label']}_locanmf_regions.npy")
        A = np.load(f"{config.locanmf_dir(mc)}/{s['label']}_locanmf_A.npy", mmap_mode="r")
        return _footprint_scale(A, C.shape[0])[:, None] * C, reg
    ad = glob.glob(f"{mc}/wfield_local_results/allen_aligned_affine8v1")[0]
    U = np.load(f"{ad}/U_atlas.npy"); SVT = np.load(config.svtcorr_path(mc))
    if source == "svt":
        # RAW SVD COEFFICIENTS, no parcellation anywhere. For pixel-space RSA (wfield_local.pixel_rsa):
        # every other source averages pixels into regions or components first, so its geometry is a
        # property of that division of the cortex. Distances between these coefficients are NOT pixel
        # distances -- U is not orthonormal after Allen registration -- so a caller must apply the
        # Gram whitener from pixel_rsa before measuring anything.
        return SVT.astype(np.float64), np.arange(SVT.shape[0])
    atlas = np.load(f"{ad}/allen_area_atlas_native_grid.npy")
    mask = np.load(f"{ad}/allen_brain_mask_native_grid.npy").astype(bool)
    Uf = U.reshape(-1, U.shape[2]); at = atlas.reshape(-1); mk = mask.reshape(-1)
    rois, regs = [], []
    for l in np.unique(at):
        if l == 0:
            continue
        pix = (at == l) & mk
        if pix.sum() < 20:
            continue
        rois.append(np.nanmean(Uf[pix], 0) @ SVT); regs.append(int(l))
    return np.array(rois), np.array(regs)


def _bins_for(args) -> int:
    """Sub-bins for this alignment: explicit on ``args``, else ``defaults.yaml decode.bins``.

    Falls back through config rather than to a literal 1 so the joint-basis analyses -- which build
    their own args namespace -- get the same windowing as the per-session path. Two analyses
    disagreeing about the feature definition is precisely what makes cross-session numbers
    incomparable.
    """
    b = getattr(args, "bins", None)
    if b:
        return int(b)
    return int((config.defaults()["decode"].get("bins") or {}).get(getattr(args, "align", "cue"), 1))


def _window_feature(sig, w0, post_n, bins, base):
    """One trial's feature vector: the window MEAN, or a concatenated time course over sub-bins.

    ``bins <= 1`` reproduces the historical single mean. ``bins = n`` splits the window into n equal
    slices and concatenates their means, so the decoder sees the window's temporal PROFILE instead of
    one number -- worth +0.032 pre-cue, +0.020 post-cue and +0.023 post-lick on corrected data
    (DECISIONS.md). ``base`` is tiled to match, so a pre-cue baseline subtracts from every bin.

    BINS ARE CLAMPED TO THE NUMBER OF FRAMES. `decode.bins` is 8 for the lick alignment, chosen for
    a 2 s window (~62 frames). Asked for a SHORT window it silently produced empty slices -- 150 ms
    is 5 frames at 31.23 Hz, and linspace(0, 5, 9) repeats three edges, so three of the eight bins
    are `w[:, a:a]` and mean over an empty axis is NaN. Nothing downstream distinguishes that from
    a real value until the fit fails, and a caller exploring window length is exactly who hits it
    (Priya, 2026-08-22, asking for a 150 ms post-lick decoder).
    """
    w = sig[:, w0:w0 + post_n]
    bins = min(int(bins), int(post_n))
    if bins <= 1:
        return w.mean(1) - base
    edges = np.linspace(0, post_n, bins + 1).astype(int)
    f = np.concatenate([w[:, a:b].mean(1) for a, b in zip(edges[:-1], edges[1:])])
    return f - (np.tile(base, bins) if np.ndim(base) else base)


def is_engaged(first_lick, rt, max_rt):
    """THE canonical engaged rule: a detected lick after the cue, within ``max_rt``.

    Extracted from the loop below so that `nolick_decoder`, which subdivides the same trials, uses
    this predicate rather than a second copy of the condition. Two copies of a trial-selection rule
    is how an analysis and the deck it is compared against silently come to mean different things by
    "engaged". Units are whatever the caller uses consistently (the decoder passes frames).
    """
    return bool(first_lick > 0 and 0 < rt <= max_rt)


def lickfree_window(cue_f, strobe_f, licks, win_n):
    """Latest ``win_n``-frame window before the cue containing NO licks, or None.

    Moved here from `precue_lickfree` on 2026-08-17 when Priya decided the HEADLINE pre-cue number
    should be lick-free: the rule now governs the production decoder, so it belongs beside it rather
    than in the module that used to be its only consumer.

    A FIXED window ending at the cue throws a trial away for a single lick anywhere in it -- including
    one 200 ms before the cue, when 2 s of clean data sits just earlier in the same enforced no-lick
    period. Instead: take the lick-free GAPS in [strobe, cue] and use the last ``win_n`` frames of the
    LATEST gap long enough to hold the window. Closest to the cue is preferred, being most informative
    about the upcoming action.

    BOUNDED AT THE STROBE, deliberately. Before the spout arrives this trial's position does not exist,
    and because the task avoids recent repeats, prior-trial activity carries real information about the
    upcoming position -- a window straying earlier would smuggle that in and look like a pre-cue code.
    """
    lo = int(np.ceil(strobe_f)) if np.isfinite(strobe_f) else None
    hi = int(cue_f)
    if lo is None or hi - lo < win_n:
        return None
    inside = np.sort(licks[(licks >= lo) & (licks < hi)])
    edges = np.concatenate(([lo], inside, [hi]))
    for i in range(len(edges) - 2, -1, -1):
        gap_start = int(edges[i]) + (1 if i > 0 else 0)      # a lick occupies its own frame
        gap_end = int(edges[i + 1])
        if gap_end - gap_start >= win_n:
            return gap_end - win_n
    return None


def precue_window_start(c0, strobe_f, licks_sorted, win_n, lickfree=True):
    """Start frame of this trial's PRE-CUE window, or None meaning DROP the trial.

    THE HEADLINE PRE-CUE NUMBER IS LICK-FREE (Priya, 2026-08-17). Previously only the Section C
    control restricted; the headline used every engaged trial and relied on the ENL to keep the
    window quiet, which it does 90.8-99.5% of the time -- but "almost always" is per SESSION, and
    PS93 8/9 falls to 76%, in the animal whose licking is already atypical. A readout whose whole
    claim is motor-independence should not rest on a task contingency holding on average.

    Trials with a clean fixed window keep it, so the common case stays exactly cue-aligned. Trials
    with a lick in it slide to the latest clean gap. Trials with no clean window ANYWHERE are
    DROPPED rather than kept-and-flagged -- for a headline number, "mostly clean" is the thing being
    retired.
    """
    fixed = int(c0) - win_n
    if not lickfree:
        return fixed if fixed >= 0 else None
    if fixed < 0:
        return None
    if not np.any((licks_sorted >= fixed) & (licks_sorted < fixed + win_n)):
        return fixed                                          # already clean; keep it cue-aligned
    return lickfree_window(c0, strobe_f, licks_sorted, win_n)


def would_be_lick_offsets(codes, rt, engaged, min_trials=5):
    """Per-position median reaction time (frames) for placing a NO-LICK trial's lick-aligned window.

    Returns ``(per_position, overall, n_engaged)``. A no-lick trial has no lick to align to, so its
    window is placed where the lick WOULD have been: the cue plus this median. Taken from the
    session's OWN engaged trials, per position, because latency differs by animal, by position and
    tenfold between phases (0.137-0.255 s pre-stroke; a median of 2.439 s at post-stroke far_R) -- a
    cohort constant would be wrong on all three axes.

    A POSITION WITH NO ENGAGED TRIAL GETS NO OFFSET, and the caller DROPS those trials (Priya,
    2026-08-21). It used to fall back to the SESSION median, which was wrong in the worst possible
    way: the fallback fires precisely where the animal has stopped licking, and the session median
    is set by the CLOSE positions that still work. Measured on the post-stroke sessions --

        PS94 far_R      0 / 1 / 2 / 0 engaged across the four sessions -> fell back to 0.17-0.23 s,
                        while the two days with any lick at all give 1.80 s and 2.25 s
        PS94 far_center 0 engaged on 0817 -> 0.20 s, against 0.75 s the next day
        PS92 far_R      0 engaged on 0818 -> 0.23 s, against 2.20 s on 0819

    -- so those windows sat up to 2.1 s early inside a 2 s window, i.e. they did not overlap the
    inferred lick at all, and the error was largest at the most impaired positions. That correlates
    the artefact with severity, which is the very axis these figures report.

    ``min_trials`` no longer gates the offset, only the WEAK flag: below it the median rests on one
    or two trials and is order-of-magnitude evidence rather than a real median. It is still the
    right value to use -- being off by a factor of two beats being off by 2 s -- but a cell resting
    on it should say so, which is what ``n_engaged`` is returned for.

    IT IS AN INFERENCE. The time comes from other trials; this one has no lick, which is the point.
    """
    import numpy as _np

    codes = _np.asarray(codes)
    rt = _np.asarray(rt, float)
    engaged = _np.asarray(engaged, bool)
    if not engaged.any():
        return {}, None, {}
    overall = float(_np.median(rt[engaged]))
    per, n_eng = {}, {}
    for c in _np.unique(codes[codes >= 0]):
        m = engaged & (codes == c)
        n_eng[int(c)] = int(m.sum())
        if int(m.sum()) >= 1:
            per[int(c)] = float(_np.median(rt[m]))
    return per, overall, n_eng


def _trial_features(s, args, signal=None, feat_region=None, with_precue_licks=False,
                    with_indices=False, nolick_ref="cue"):
    """Trial-averaged features for one session.

    ``nolick_ref`` controls where a NO-LICK trial's window starts when ``args.align == "lick"``.

    * ``"cue"`` (default, historical): the cue. A lick trial's window starts at its FIRST LICK, so
      the two arms are then offset by the whole reaction time -- 0.137-0.255 s pre-stroke, but a
      median of 2.439 s at post-stroke far_R, where a 2 s window means the two do not overlap at
      all. Comparing them is not meaningful, which is why the no-lick arm is excluded from
      lick-aligned analyses rather than quietly used.
    * ``"would_be_lick"``: the cue plus this session's OWN median reaction time AT THAT POSITION,
      i.e. when the lick would have happened had it happened (Priya, 2026-08-21). Per session and
      per position rather than a cohort constant, because the latency differs by animal, by position
      and tenfold between phases -- taking it from the session's own engaged trials handles all
      three without a lookup table. Falls back to the session median where a position has too few
      engaged trials, and drops the trial if neither exists.

      IT IS AN INFERENCE, NOT A MEASUREMENT. The time is taken from OTHER trials; this trial has no
      lick, which is the whole point. It is shakiest exactly post-stroke, where latencies are long
      and variable.

    ``signal``/``feat_region`` let a caller INJECT an already-built (nfeat, T) signal instead of
    loading ``args.source`` from disk -- used by the joint-basis cross-session analyses, where the
    components come from one shared basis rather than that session's own LocaNMF fit. Everything
    downstream (engagement, position blocks, alignment, the no-lick arm) is then identical to the
    per-session path by construction, which is the point: the basis is the only thing that differs.
    """
    if signal is None:
        sig, feat_reg = _build_signal(s, args.source)
    else:
        sig = np.asarray(signal)
        feat_reg = np.arange(sig.shape[0]) if feat_region is None else np.asarray(feat_region)
    nfeat, T = sig.shape
    cue = _load_cue_events(s["h5"]); lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, csmp = _frames(s, cue, lk)
    codes = classify_cues_with_backup(s, cue)
    # BLOCK ID. A new block starts when the position changes OR when the run reaches this session's
    # scheduler block_size_max -- a longer run cannot be one block. The old rule keyed on position
    # change alone, so two adjacent blocks at the SAME position merged into one; audited against the
    # firmware's own block_number that is 118/4216 = 2.8% of blocks (Priya, 2026-08-18). See
    # wfield_local/block_ids.py for the audit, the residual limits, and why merging made the previous
    # CV conservative rather than inflated.
    blk_id = block_ids(np.asarray(codes), block_size_max_for(s))
    # CLAMPED HERE, not inside `_window_feature`, so the feature width and the component->region
    # labels tiled from it below cannot disagree. `decode.bins` is 8, sized for a 2 s window; a
    # 150 ms one is 5 frames and would otherwise give three empty (NaN) bins and a `feat_reg` eight
    # times too long for a five-block feature vector.
    bins = min(_bins_for(args), max(1, round(args.post_s * args.fs)))
    # PRE-CUE WINDOWS ARE LICK-FREE (Priya, 2026-08-17). Bounded at the spout strobe, so compute the
    # per-trial strobe frame; `precue_window_start` needs it and returns None for a trial with no
    # clean window anywhere, which is then dropped.
    lickfree = bool(config.defaults()["decode"].get("precue_lickfree", True)) and args.align == "precue"
    ls_sorted = np.sort(np.asarray(lick_f))
    if lickfree:
        cs = np.asarray(cue["cue_samples"]); ss = np.asarray(cue["strobe_samples"])
        sr = float(cue["sample_rate_hz"])
        jj = np.searchsorted(ss, cs, side="right") - 1
        lead_s = np.where(jj >= 0, (cs - ss[np.clip(jj, 0, len(ss) - 1)]) / sr, np.nan)
        strobe_f = cue_f - lead_s * args.fs
    else:
        strobe_f = np.full(cue_f.shape, np.nan)
    n_dropped_dirty = 0
    n_dropped_nolatency = 0     # no-lick trials at a position the animal never licked that session
    pre_n = int(round(args.pre_s * args.fs)); post_n = int(round(args.post_s * args.fs))
    maxrt_n = int(round(args.max_rt * args.fs))
    ls = np.sort(lick_f); j = np.searchsorted(ls, cue_f, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1); rt = first - cue_f
    subtract = args.baseline == "precue"
    X, y, g, Xn, yn = [], [], [], [], []
    # per-ENGAGED-trial flag: did the FIXED 2 s pre-cue window contain licks? Returned on request so
    # callers never have to replay this loop to find out -- reconstructing a trial filter elsewhere
    # is how bugs 15, 16 and 17 all happened, and a mask built outside here silently misaligned by 58
    # trials the first time it was tried.
    precue_lick = []
    # CUE INDICES for each arm, on request. Any per-trial attribute -- engagement state, lick
    # category, RT -- can then be aligned to the feature matrices by indexing, instead of replaying
    # this loop elsewhere and hoping the two agree. They did not: an externally rebuilt mask came
    # out 633 long against 575 kept trials, and bugs 15-17 were all this same shape.
    idx_eng, idx_nolick = [], []
    # WOULD-BE-LICK reference: this session's own median RT per position, in frames.
    med_rt, _med_rt_all, med_rt_n = {}, None, {}
    if args.align == "lick" and nolick_ref == "would_be_lick":
        _eng = np.array([bool(is_engaged(first[k], rt[k], maxrt_n)) for k in range(cue_f.size)])
        med_rt, _med_rt_all, med_rt_n = would_be_lick_offsets(codes, rt, _eng)
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        c0 = int(cue_f[k])
        # cue/precue-referenced window start (precue = the post_n window ENDING at the cue, slid
        # earlier if a lick falls in it; None -> no clean window exists, drop the trial)
        if args.align == "precue":
            ref0 = precue_window_start(c0, strobe_f[k], ls_sorted, post_n, lickfree=lickfree)
            if ref0 is None:
                n_dropped_dirty += 1
                continue
        else:
            ref0 = c0
        if ref0 < 0 or ref0 + post_n > T:
            continue
        if subtract:
            if c0 - pre_n < 0:
                continue
            base = sig[:, c0 - pre_n:c0].mean(1)
        else:
            base = 0.0
        if is_engaged(first[k], rt[k], maxrt_n):            # ENGAGED: cue + lick
            w0 = int(first[k]) if args.align == "lick" else ref0
            if w0 < 0 or w0 + post_n > T:
                continue
            X.append(_window_feature(sig, w0, post_n, bins, base))
            y.append(int(codes[k])); g.append(int(blk_id[k]))
            idx_eng.append(k)
            if with_precue_licks:
                fx = c0 - post_n
                precue_lick.append(bool(fx >= 0 and np.any((ls_sorted >= fx) & (ls_sorted < c0))))
        else:                                               # NO-LICK: no lick to align to
            w0n = ref0                                      # cue/precue-referenced by default
            if args.align == "lick" and nolick_ref == "would_be_lick":
                # NO session-median fallback: a position with no engaged trial has no evidence
                # about its own latency, and the session median is set by the positions that still
                # work. Drop instead -- see `would_be_lick_offsets` for the 2 s misplacement.
                _off = med_rt.get(int(codes[k]))
                if _off is None:
                    n_dropped_nolatency += 1
                    continue
                w0n = c0 + round(_off)      # round() on a float already yields int
            if w0n < 0 or w0n + post_n > T:
                continue
            Xn.append(_window_feature(sig, w0n, post_n, bins, base)); yn.append(int(codes[k]))
            idx_nolick.append(k)
    # component->region labels must be tiled with the features, or the encoder would group a
    # sub-binned feature vector by the wrong regions
    if bins > 1:
        feat_reg = np.tile(feat_reg, bins)
    if n_dropped_dirty:
        print(f"  [precue lick-free] {s['label']}: dropped {n_dropped_dirty} trial(s) with no "
              f"lick-free {args.post_s:g}s window between the spout strobe and the cue", flush=True)
    # Reported whenever EITHER condition holds. A session can have a thin position without an empty
    # one -- PS94_0819 rests far_R on two trials and drops nothing -- and gating the whole message on
    # the drop count would leave exactly those cells unflagged.
    _weak = sorted((POSITION_NAMES.get(c, str(c)), n) for c, n in med_rt_n.items() if 1 <= n < 5)
    if n_dropped_nolatency or _weak:
        bits = []
        if n_dropped_nolatency:
            bits.append(f"dropped {n_dropped_nolatency} no-lick trial(s) at position(s) with NO "
                        f"engaged trial -- no evidence about their latency")
        if _weak:
            bits.append("offset rests on <5 engaged trials at "
                        + ", ".join(f"{p} (n={n})" for p, n in _weak))
        print(f"  [would-be-lick] {s['label']}: " + "; ".join(bits), flush=True)
    base_out = (np.array(X), np.array(y), np.array(g), np.array(Xn), np.array(yn), feat_reg)
    extra = ()
    if with_precue_licks:
        extra += (np.array(precue_lick, bool),)
    if with_indices:
        extra += (np.array(idx_eng, int), np.array(idx_nolick, int))
    return base_out + extra if extra else base_out


def _save_session_fig(label, cmn, sm, labs, args, tag):
    """One compact (confusion | recall) figure for a single session, for the animal-first deck."""
    fig, (axc, axr) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    if cmn is not None:
        im = axc.imshow(cmn, vmin=0, vmax=1, cmap="magma")
        axc.set_xticks(range(6)); axc.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
        axc.set_yticks(range(6)); axc.set_yticklabels(labs, fontsize=7)
        axc.set_xlabel("predicted"); axc.set_ylabel("true")
        fig.colorbar(im, ax=axc, shrink=0.75)
    axc.set_title(f"{label}  {args.align} 0-{args.post_s:g}s\nacc={sm['acc']['all']:.2f} "
                  f"SSp={sm['acc']['SSp']:.2f} MO={sm['acc']['MO']:.2f} (chance .17)", fontsize=9)
    x = np.arange(6); nl_ok = args.align in ("cue", "precue") and not np.isnan(sm["acc_nolick"]["all"])
    w = 0.38 if nl_ok else 0.6
    axr.bar(x - (w / 2 if nl_ok else 0), sm["recall_by_position"].get("all", [np.nan] * 6), w,
            color="tab:blue", label=f"engaged ({args.cv}-CV)")
    if nl_ok:
        axr.bar(x + w / 2, sm["recall_nolick_by_position"].get("all", [np.nan] * 6), w,
                color="tab:red", label=f"no-lick (n={sm['n_nolick']})")
    axr.axhline(1 / 6, color="grey", ls="--", lw=0.8, label="uniform 1/6")
    _fl = sm.get("majority_class_floor")
    if _fl and _fl > 1 / 6 + 0.005:
        # Only drawn when it DIFFERS. Pre-stroke the two coincide (0.167 vs 0.178) and a second
        # line would be clutter; post-stroke they diverge by up to 65% and the uniform line alone
        # understates the no-information floor.
        axr.axhline(_fl, color="firebrick", ls=":", lw=1.2,
                    label=f"best constant guess ({_fl:.2f})")
    axr.set_xticks(x); axr.set_xticklabels(labs, rotation=45, ha="right", fontsize=7); axr.set_ylim(0, 1)
    nls = f"  no-lick={sm['acc_nolick']['all']:.2f}" if nl_ok else ""
    _bal = sm.get("balanced_accuracy")
    _p = sm.get("null_balanced_p")
    _extra = (f"  balanced={_bal:.2f}" if _bal is not None else "") +              (f" (perm p={_p:.3f})" if _p is not None else "")
    axr.set_title(f"per-position recall  eng={sm['acc']['all']:.2f}{nls}{_extra}", fontsize=9)
    axr.set_ylabel("recall"); axr.legend(fontsize=7)
    fig.tight_layout()
    p = args.output / f"locanmf_position_session_{label}_{tag}.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print("  wrote", p.name, flush=True)


def main() -> int:
    dp = config.defaults()["decode"]      # windows/CV/baseline/max_rt/chance (configs/defaults.yaml)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="0603")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--source", choices=("locanmf", "roi"), default="locanmf")
    ap.add_argument("--align", choices=("cue", "lick", "precue"), default="cue")
    ap.add_argument("--baseline", choices=("none", "precue"), default=dp["baseline"])
    ap.add_argument("--cv", choices=("block", "random"), default=dp["cv"])
    ap.add_argument("--fs", type=float, default=31.23)
    ap.add_argument("--pre-s", type=float, default=1.0)
    ap.add_argument("--post-s", type=float, default=dp["cue_post_s"], help="feature window after the alignment "
                    "event (2.0 = empirical optimum; spans the lick bout. >~2.5s dilutes the transient). Per-align "
                    "windows (lick/cue/precue) live in configs/defaults.yaml decode.*_post_s; nightly_figs passes them.")
    ap.add_argument("--max-rt", type=float, default=dp["max_rt_s"])
    ap.add_argument("--bins", type=int, default=None,
                    help="sub-bins across the feature window (time course instead of one mean). "
                         "Default per alignment from configs/defaults.yaml decode.bins; pass 1 for "
                         "the historical single-mean feature.")
    ap.add_argument("--per-session", action="store_true",
                    help="also write one compact confusion+recall figure per session "
                         "(locanmf_position_session_{label}_{tag}.png) for the animal-first deck")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sess = sorted([s for s in SESSIONS if s["label"].endswith(args.date)], key=lambda s: s["label"][:4])
    print(f"source={args.source} align={args.align} baseline={args.baseline} cv={args.cv}  "
          f"sessions={[s['label'] for s in sess]}  chance={dp['chance']}", flush=True)

    def _cv_predict(clf, Xc, yv, gv):
        if args.cv == "block":
            ng = min(5, int(np.unique(gv).size))
            return cross_val_predict(clf, Xc, yv, cv=GroupKFold(ng), groups=gv)
        return cross_val_predict(clf, Xc, yv, cv=StratifiedKFold(5, shuffle=True, random_state=0))

    groups = {"all": None, "SSp": ("SSp",), "MO": ("MOp", "MOs")}
    _rows, _cols = grid_shape(len(sess))
    fig, axes = plt.subplots(_rows, _cols, figsize=(5.0 * _cols, 4.5 * _rows), squeeze=False)
    summary = {}
    for si, s in enumerate(sess):
        X, y, gblk, Xnl, ynl, feat_reg = _trial_features(s, args)
        names = {int(k): v for k, v in json.load(
            open(glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]))}
        nl_ok = args.align in ("cue", "precue") and Xnl.shape[0] >= 6   # no-lick valid for cue/precue (no lick needed)
        accs = {}; recall = {}; acc_nl = {}; recall_nl = {}; cmn = None
        for g, prefs in groups.items():
            cols = (np.arange(X.shape[1]) if prefs is None else
                    np.array([i for i in range(X.shape[1]) if any(names.get(int(feat_reg[i]), "").startswith(p) for p in prefs)]))
            if cols.size == 0:
                accs[g] = recall[g] = float("nan"); acc_nl[g] = float("nan"); recall_nl[g] = [float("nan")] * 6; continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            pred = _cv_predict(clf, X[:, cols], y, gblk)
            accs[g] = accuracy_score(y, pred)
            cmg = confusion_matrix(y, pred, labels=DISPLAY_ORDER); cmg = cmg / np.maximum(cmg.sum(1, keepdims=True), 1)
            recall[g] = np.diag(cmg).tolist()      # engaged per-position recall (5-fold, out-of-sample)
            if g == "all":
                cmn = cmg
            if nl_ok:                              # train on engaged, apply to held-aside no-lick trials
                clf2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(X[:, cols], y)
                pnl = clf2.predict(Xnl[:, cols]); acc_nl[g] = accuracy_score(ynl, pnl)
                cmnl = confusion_matrix(ynl, pnl, labels=DISPLAY_ORDER)
                recall_nl[g] = (np.diag(cmnl) / np.maximum(cmnl.sum(1), 1)).tolist()
            else:
                acc_nl[g] = float("nan"); recall_nl[g] = [float("nan")] * 6
        npos = {POSITION_NAMES[c]: int((y == c).sum()) for c in DISPLAY_ORDER}
        # THE FLOOR IS NOT 1/6 ONCE THE ENGAGED TRIALS ARE SKEWED.
        # A uniform 1/6 assumes the animal attempted every position about equally. Post-stroke it
        # does not: PS94_0820 engaged is [71,72,68,63,17,0] and PS92_0822 is [58,65,81,44,41,5], so
        # a constant "always guess close_R" scores 0.247-0.276 with no information at all, against a
        # chance line drawn at 0.167. DECISIONS records exactly this error for the NO-LICK arm
        # (2026-08-17, "The null was wrong") and the corrected machinery went there; the same skew
        # has since arrived in the ENGAGED arm by a different route -- abandonment rather than
        # declining -- and the reference line did not follow it.
        #
        # Both references are cheap here because the model's predictions are held FIXED: the
        # permutation shuffles labels only, so nothing is refitted.
        floor = na.majority_class_floor(y, labels=DISPLAY_ORDER)
        nullv = na.permutation_null(y, pred, n_perm=1000, labels=DISPLAY_ORDER)
        balanced = na.balanced_accuracy(y, pred, labels=DISPLAY_ORDER)
        summary[s["label"]] = {"n_trials": int(X.shape[0]), "n_feat": int(X.shape[1]), "acc": accs,
                               "positions": [POSITION_NAMES[c] for c in DISPLAY_ORDER],
                               "recall_by_position": recall, "n_per_position": npos,
                               "n_nolick": int(Xnl.shape[0]), "acc_nolick": acc_nl,
                               "recall_nolick_by_position": recall_nl,
                               "majority_class_floor": float(floor),
                               "balanced_accuracy": float(balanced),
                               "null_raw_mean": nullv["raw_null_mean"],
                               "null_raw_ci": nullv["raw_null_ci"], "null_raw_p": nullv["raw_p"],
                               "null_balanced_mean": nullv["bal_null_mean"],
                               "null_balanced_p": nullv["bal_p"],
                               "confusion_all": cmn.tolist() if cmn is not None else None,
                               "source": args.source, "align": args.align}
        nlstr = (f" | NO-LICK(n={Xnl.shape[0]}) " + "  ".join(f"{g}={a:.2f}" for g, a in acc_nl.items())) if nl_ok \
            else f" | no-lick n={Xnl.shape[0]} (skipped: needs --align cue)"
        print(f"{s['label']}: engaged n={X.shape[0]} {args.source}feat={X.shape[1]} | "
              + "  ".join(f"{g}={a:.2f}" for g, a in accs.items()) + nlstr, flush=True)
        ax = axes[si // _cols][si % _cols]; im = ax.imshow(cmn, vmin=0, vmax=1, cmap="magma")
        labs = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
        ax.set_xticks(range(6)); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(6)); ax.set_yticklabels(labs, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{s['label']}  acc={accs['all']:.2f} (chance .17)\nSSp={accs['SSp']:.2f} MO={accs['MO']:.2f}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7)
        if args.per_session:
            _save_session_fig(s["label"], cmn, summary[s["label"]], labs, args, tag=f"{args.source}_{args.align}_base-{args.baseline}_cv-{args.cv}")
    blank_unused(axes, len(sess), _rows, _cols)
    fig.suptitle(f"Spout-position decoding [{args.source}, {args.align}-aligned 0-{args.post_s:g}s, "
                 f"baseline={args.baseline}, {args.cv}-CV, engaged], {args.date}", fontsize=12)
    fig.tight_layout()
    tag = f"{args.source}_{args.align}_base-{args.baseline}_cv-{args.cv}"
    fig.savefig(args.output / f"locanmf_position_decoder_{args.date}_{tag}.png", dpi=130); plt.close(fig)

    # ---- per-position recall: engaged (5-fold) vs no-lick (trained on engaged) ----
    posnames = [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    _rows2, _cols2 = grid_shape(len(sess))
    fig2, axes2 = plt.subplots(_rows2, _cols2, figsize=(5.0 * _cols2, 4.2 * _rows2),
                               squeeze=False, sharey=True)
    for si, s in enumerate(sess):
        ax = axes2[si // _cols2][si % _cols2]; sm = summary[s["label"]]; x = np.arange(6); w = 0.38
        ax.bar(x - w / 2, sm["recall_by_position"].get("all", [np.nan] * 6), w, color="tab:blue",
               label="engaged (5-fold)")
        ax.bar(x + w / 2, sm["recall_nolick_by_position"].get("all", [np.nan] * 6), w, color="tab:red",
               label=f"no-lick (n={sm['n_nolick']})")
        ax.axhline(1 / 6, color="grey", ls="--", lw=0.8, label="chance")
        ax.set_xticks(x); ax.set_xticklabels(posnames, rotation=45, ha="right", fontsize=7); ax.set_ylim(0, 1)
        nls = f"{sm['acc_nolick']['all']:.2f}" if not np.isnan(sm['acc_nolick']['all']) else "n/a"
        ax.set_title(f"{s['label']}  eng={sm['acc']['all']:.2f}  no-lick={nls}", fontsize=9)
        if si == 0:
            ax.set_ylabel("per-position recall"); ax.legend(fontsize=7)
    blank_unused(axes2, len(sess), _rows2, _cols2)
    fig2.suptitle(f"Per-position recall — engaged vs no-lick [{tag}], {args.date} "
                  f"(no-lick = baseline disengagement here; post-stroke = failed attempts)", fontsize=11)
    fig2.tight_layout()
    fig2.savefig(args.output / f"locanmf_position_recall_{args.date}_{tag}.png", dpi=130); plt.close(fig2)

    (args.output / f"locanmf_position_decoder_{args.date}_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", args.output / f"locanmf_position_decoder_{args.date}_{tag}.png",
          "+ recall fig + summary", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

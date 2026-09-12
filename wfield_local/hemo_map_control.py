"""Is the structure in the position maps CALCIUM, or is it leftover blood? Two independent checks.

WHY. Faint vessel-shaped structure is visible in the POST-stroke mean maps of figure 14, and a
stroke is exactly the manipulation that could make a haemodynamic artefact grow between epochs --
altered perfusion, altered vasomotion, a window that ages. If some of the post-stroke map is
residual blood volume rather than calcium, then "the far-contralateral map collapses" could in
principle be a statement about vessels. This module tries to falsify that.

THE DECISIVE CHECK IS THE ISOSBESTIC CHANNEL. At 415 nm GCaMP fluorescence does not depend on
calcium, so that channel carries haemodynamics and noise and nothing else. Build the SAME position
map from the 415 series -- same trials, same window, same basis, same code path, with only the
signal swapped -- and correlate it against the corrected map:

    r(SVTcorr map, 415 map) near 0     the structure is calcium; the correction did its job
    r high and RISING post-stroke      the maps are partly blood, and the epoch comparison is unsafe

This is stronger than the temporal residual check in `hemo_residual_check`, which asks whether the
global trace still correlates with 415. A map can be spatially contaminated while the global traces
look clean, because the correction is a single per-pixel regression and its residual has structure.

THE SECOND CHECK IS THE VESSEL TEMPLATE, and it is the weaker of the two. `frames_average_atlas`
is the session's mean fluorescence on the shared grid; vessels are the DARK linear structures in
it, so a spatial high-pass of the negated image is a crude vesselness map. Correlating |position
map| against it says whether map amplitude piles up on vessels. Crude because real cortical signal
is not uniform either and some areas are genuinely more vascular -- so a non-zero correlation is
not by itself evidence of artefact. WHAT MATTERS IS THE CHANGE BETWEEN EPOCHS: the pre-stroke value
is the control for how much vessel-alignment a clean map has in this preparation.

NEITHER CHECK CAN PROVE THE MAPS ARE CLEAN. They can only fail to show contamination, which is what
a control is for. Read them as "the artefact is not large enough to see by two methods", never as
"there is no artefact".
"""
from __future__ import annotations

import glob

import numpy as np

from wfield_local.beta_maps import MAP_SHAPE, MIN_TRIALS_PER_CLASS, _quit_mask, brain_mask, map_corr

#: Which half of the interleaved raw `SVT.npy` is the isosbestic (calcium-free) channel. `FUNC` is
#: the functional (470) index, so the other one is 415 -- taken from `hemo_variants` rather than
#: written down again, because getting it backwards would compare the map against itself and return
#: a reassuring r = 1.0 as evidence of cleanliness.
def _iso_slice():
    from wfield_local import hemo_variants as hv
    return slice((hv.FUNC + 1) % 2, None, 2)


def vessel_template(session, sigma=12.0):
    """A crude vesselness map on the shared grid, or None -- high-passed NEGATED mean fluorescence.

    Vessels are dark in the mean image, so negating makes them positive; the high-pass removes the
    smooth illumination and curvature gradients that would otherwise dominate any correlation.
    """
    from scipy import ndimage

    ad = glob.glob(f"{session['mc']}/wfield_local_results/allen_aligned_affine8v1")
    if not ad:
        return None
    try:
        fa = np.load(f"{ad[0]}/frames_average_atlas.npy")
    except Exception as ex:                                            # noqa: BLE001
        print(f"  !! vessel template {session['label']}: {type(ex).__name__} {str(ex)[:60]}",
              flush=True)
        return None
    img = np.nan_to_num(np.asarray(fa, float))
    if img.ndim == 3:
        img = img[0]                       # the functional channel; vessels are in both
    if img.shape != MAP_SHAPE:
        return None
    neg = -img
    return neg - ndimage.gaussian_filter(neg, sigma)


def session_control(session, align, *, post_s=2.0, variant="working"):
    """``{position: {"r_iso": float, "r_vessel": float, "n": int}}`` for one session.

    ONE LOAD, BOTH MAPS. The corrected and isosbestic maps are built from the same `U`, the same
    trial indices and the same window -- `trial_features_cached` is called twice with nothing
    different but the `signal` argument -- so any difference between the two maps is the channel
    and cannot be the trial selection.
    """
    from wfield_local import config, joint_basis
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached

    code_of = {nm: int(c) for c, nm in POSITION_NAMES.items()}
    u, v = joint_basis._load_session(session["mc"])
    raw = np.load(f"{session['mc']}/wfield_local_results/SVT.npy", mmap_mode="r")
    iso = np.asarray(raw[:, _iso_slice()], np.float32)
    # THE TWO SERIES MUST BE THE SAME LENGTH or the frame indices mean different times in each.
    # The raw file holds 2T interleaved samples against SVTcorr's T, but a repaired or truncated
    # session can be off by one pair, and silently trimming the WRONG one would shift every trial.
    n = min(iso.shape[1], np.asarray(v).shape[1])
    if abs(iso.shape[1] - np.asarray(v).shape[1]) > 2:
        print(f"  !! {session['label']}: isosbestic {iso.shape[1]} vs corrected "
              f"{np.asarray(v).shape[1]} frames -- skipping", flush=True)
        return {}
    iso, vv = iso[:, :n], np.asarray(v)[:, :n]

    args = _args("locanmf", align, post_s)
    K = u.shape[1]

    def _feats(sig, key):
        X, y, _g, Xn, yn, _reg, idx_e, idx_n = trial_features_cached(
            session, args, signal=np.asarray(sig), feat_region=np.arange(sig.shape[0]),
            signal_key=key, with_indices=True)
        X, y = np.asarray(X), np.asarray(y)
        if variant == "working" and len(yn):
            keep = ~_quit_mask(session, idx_e, idx_n, y, yn)
            if keep.any():
                X = np.vstack([X, np.asarray(Xn)[keep]])
                y = np.concatenate([y, np.asarray(yn)[keep]])
        return X, y

    Xc, yc = _feats(vv, f"svt:rank{K}")
    Xi, yi = _feats(iso, f"svt415:rank{K}")
    if not len(yc) or len(yc) != len(yi) or not np.array_equal(yc, yi):
        print(f"  !! {session['label']}: corrected and isosbestic trial sets differ "
              f"({len(yc)} vs {len(yi)}) -- skipping", flush=True)
        return {}
    n_bins = Xc.shape[1] // K
    mask, vess = brain_mask(), vessel_template(session)

    def _map(X, sel):
        return (u @ X[sel].mean(0).reshape(n_bins, K).mean(0)).reshape(MAP_SHAPE)

    out = {}
    for q in CONF_LABELS:
        c = code_of.get(q)
        if c is None:
            continue
        sel = yc == c
        if int(sel.sum()) < MIN_TRIALS_PER_CLASS:
            continue
        mc_, mi_ = _map(Xc, sel), _map(Xi, sel)
        # IN-MASK ONLY, for the reason the permutation test needed it: 40% of the frame is empty,
        # and two maps that are both ~0 over 138,387 pixels correlate at r ~ 1 on those pixels
        # alone. An off-brain correlation would read as contamination and be nothing at all.
        if mask is not None:
            a, b = mc_[mask], mi_[mask]
            w = None if vess is None else vess[mask]
        else:
            a, b, w = mc_.ravel(), mi_.ravel(), (None if vess is None else vess.ravel())
        out[q] = {"r_iso": map_corr(a, b),
                  "r_vessel": float("nan") if w is None else map_corr(np.abs(a), w),
                  "r_vessel_iso": float("nan") if w is None else map_corr(np.abs(b), w),
                  "n": int(sel.sum())}
    _ = config     # imported for the path convention the loaders share; kept explicit
    return out


def by_epoch(align="cue", variant="working", post_s=2.0, animals=None):
    """``{epoch: {"r_iso": [...], "r_vessel": [...]}}`` pooled over sessions and positions."""
    from wfield_local import config
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import ANIMALS, _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS

    out = {}
    for an in (animals or ANIMALS):
        want = {x for x in config.phase_labels("pre") + config.phase_labels("post")
                if x.startswith(an)}
        for s in [x for x in SESSIONS if x["label"] in want]:
            d = _day(an, s["label"].split("_")[-1])
            if d is None:
                continue
            e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
            if e is None:
                continue
            try:
                got = session_control(s, align, post_s=post_s, variant=variant)
            except Exception as ex:                                    # noqa: BLE001
                print(f"  !! hemo-control {s['label']}: {type(ex).__name__} {str(ex)[:70]}",
                      flush=True)
                continue
            for q, r in got.items():
                b = out.setdefault(e, {"r_iso": [], "r_vessel": [], "r_vessel_iso": [],
                                       "sessions": set()})
                b["r_iso"].append(r["r_iso"])
                b["r_vessel"].append(r["r_vessel"])
                b["r_vessel_iso"].append(r["r_vessel_iso"])
                b["sessions"].add(s["label"])
    return out


def main(argv=None) -> int:
    import argparse

    from wfield_local.console import use_utf8_stdout
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--align", default="cue", choices=("precue", "cue", "lick"))
    ap.add_argument("--variant", default="working", choices=("working", "lick"))
    ap.add_argument("--animals", nargs="+", default=None)
    a = ap.parse_args(argv)

    res = by_epoch(a.align, a.variant, animals=a.animals)
    if not res:
        print("no sessions produced a control map")
        return 1
    print(f"\n{'epoch':10s}{'sess':>6s}{'maps':>6s}"
          f"{'r(corrected, 415)':>20s}{'r(|corrected|, vessel)':>24s}{'r(|415|, vessel)':>18s}")
    for e in ("pre", "acute", "subacute", "chronic"):
        b = res.get(e)
        if not b:
            continue
        f = [np.nanmedian(b[k]) for k in ("r_iso", "r_vessel", "r_vessel_iso")]
        print(f"{e:10s}{len(b['sessions']):>6d}{len(b['r_iso']):>6d}"
              f"{f[0]:>20.3f}{f[1]:>24.3f}{f[2]:>18.3f}")
    print("\nREAD THE CHANGE, NOT THE LEVEL. A non-zero vessel correlation is expected -- cortex is "
          "not uniformly vascular. What would indicate an artefact is r(corrected, 415) RISING "
          "after the stroke, or the vessel correlation of the CORRECTED map converging on that of "
          "the 415 map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

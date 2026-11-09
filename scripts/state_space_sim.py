"""Can the state-space readouts tell gain, rotation, shift, new dimensions and trial spread apart? (Priya 11-07, plan
docs/STATE_SPACE_RECOVERY_PLAN.md, "method check before the render lands".)

Uses ONLY pre-stroke sessions. Plants a known change into one held-out pre session at a time, then scores it against
every other pre session of the same animal, single session vs single session at matched trial counts. The UNCHANGED
held-out sessions give the null. Real trial noise, real trial counts; the acute-sized subsample asks what survives when
far-contra trials are few.

Data: per-trial area extracts (`first_cue`: first-lick trials x 71 lags (-1..2.5 s) x 63 Allen areas incl. ALM / tjM1),
binned to 0.1 s over -0.5..2.0 s. Left hemisphere = ipsilesional (all four lesions left).

Planted changes (to every trial of the test session; mu = the reference grand mean, P = the reference top-k space):
  none                      the null
  amp a                     mu + a (Y - mu)
  gain_rand s               mu + g o (Y - mu), log g ~ N(0, s) per area
  gain_ipsi g               left-hemisphere areas x g
  rot th                    rotation by th degrees in a random plane inside P
  shift f                   + a condition-independent time course along a random direction in P (norm = f x the shared
                            trajectory's norm)
  newdim f                  + a spout-specific time course along a direction OUTSIDE P (f = added share of variance)
  spread f                  + a per-trial random offset inside P, f x the pre trial-to-trial SD (local search)

Readouts (session vs session, median over the comparison sessions):
  out_var   share of the test session's centered variance outside P
  r, amp    pattern correlation and amplitude ratio vs the comparison session
  rot_gain  held-out (leave one spout out) R2 of rotation-after-per-area-gain minus per-area-gain alone (ladder L4 - L3)
  gain_sd   SD of log per-area gains (gain fitted per area)
  shared    norm of the spout-averaged trajectory, ratio
  spread    trial-to-trial spread inside P around each spout mean, ratio
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

from wfield_local import orofacial_cohort as oc  # noqa: E402
from wfield_local import render_profile  # noqa: E402
from wfield_local.paths import PathResolver  # noqa: E402
from wfield_local.pose3d import pose3d_dir  # noqa: E402

# PATHS (handoff cleanup item 1, 11-08): EVERY location the state-space scripts read or write is defined here and nowhere
# else. Inputs are products of the cohort-1 REBUILD run (render profile PROFILE): its root from the profile, the encoder
# tree `<root>/LP_analysis/<orofacial_cohort.RUN>/imaging_encoding` (= orofacial_outputs.encoding_root under the
# profile) and the run tags run_mode gives under it. The tags stay literals so no profile is applied inside an analysis
# process (module caches would keep it); tests/test_state_space.py recomputes them in a subprocess and pins them.
PROFILE = "cohort1_rebuild_20261103"
REBUILD = Path(render_profile.load(PROFILE)["rebuild_root"])
ENCODING = REBUILD / "LP_analysis" / oc.RUN / "imaging_encoding"
XTAG = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801"           # run_mode.tag(include=XOUT parts): extracts
WTAG = XTAG + "_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"   # run_mode.tag(): tables
ITI = str(ENCODING / "iti_licks" / XTAG / il.SESS_NAME)                 # per-trial extracts (iti_licks.XOUT / SESS_NAME)
ITI_W = ENCODING / "iti_licks" / WTAG                                    # run tables (iti_licks.OUT_W): aim, first-lick times
LICKS3D = pose3d_dir() / "kinematics" / "licks.csv.gz"                   # 3-D first licks (landing join)
ROOT = Path(PathResolver().root("microscope")) / "Widefield" / "state_space_recovery"   # every state-space output
LANDING = ROOT / "landing" / "first_lick_landing.csv"
OUT = ROOT / "sim"                        # method-check tables; main() appends tag() (cue + Allen = sim/, as before)
POS = list(oc.POS_ORDER)                  # the canonical spout order (close ipsi .. far contra)
# top-k pre space. 8 = convention when written; checked 11-07 (Priya "how did you choose 8?"): top 8 capture 97-99 % of a
# HELD-OUT pre session's spout-specific variance in every animal / alignment (k=4: 89-97 %, >8 adds < 3 %).
# WF_SS_K overrides for the sensitivity runs (outputs then carry _k<K>).
K = int(os.environ.get("WF_SS_K", 8))
MET = ["out_var", "out_resid", "r", "amp", "rot_gain", "gain_sd", "shared", "shared_dist", "spread", "spread_norm"]
# the extract's lag grids (iti_licks.LAGS_C: first_cue, -1..2.5 s; LAGS_P: first_peth, -1.5..0.75 s), imported, not re-declared
LAGS, LAGS_P = il.LAGS_C, il.LAGS_P
BINS = [(a, a + 0.1) for a in np.round(np.arange(-0.5, 2.0, 0.1), 2)]
PLANTS = ([("none", 0)] + [("amp", a) for a in (1.25, 1.5)] + [("gain_rand", s) for s in (0.15, 0.3)]
          + [("gain_ipsi", g) for g in (0.8, 0.6)] + [("rot", t) for t in (10, 20, 30, 45)] + [("rot12", t) for t in (10, 20, 30)]
          + [("shift", f) for f in (0.25, 0.5)] + [("newdim", f) for f in (0.05, 0.15)]
          + [("spread", f) for f in (0.5, 1.0)])


# ALIGNMENT (Priya 11-07: "this should probably span the trial length; post-stroke some responses don't come for a few
# sec. is there a lick-aligned version?"). Acute far-contra first licks: median 0.78 s, 90th pct 2.4 s after the cue, so
# a fixed cue window misses many acute lick responses. Set WF_SS_ALIGN (default "cue" = everything published before):
#   cue     first_cue, -0.5..2.0 s from the cue (25 bins)
#   cue25   first_cue, -0.5..2.5 s (30 bins; the extract ends at 2.5 s)
#   lick    first_peth, -1.0..0.7 s from the first lick (17 bins; the extract spans -1.5..0.75 s)
#   splice  cue -0.5..0.5 s (10 bins) followed by lick -0.5..0.7 s (12 bins): both events at any latency. For first
#           licks within 1 s of the cue the two segments sample overlapping time (each segment is its own features).
ALIGN = os.environ.get("WF_SS_ALIGN", "cue")
_B = lambda a, b: [(t, t + 0.1) for t in np.round(np.arange(a, b - 1e-9, 0.1), 2)]  # noqa: E731
SEGMENTS = {"cue": [("first_cue", LAGS, _B(-0.5, 2.0))], "cue25": [("first_cue", LAGS, _B(-0.5, 2.5))],
            "lick": [("first_peth", LAGS_P, _B(-1.0, 0.7))],
            "splice": [("first_cue", LAGS, _B(-0.5, 0.5)), ("first_peth", LAGS_P, _B(-0.5, 0.7))]}
# balance windows (bin slices) per alignment: baseline / response. lick + splice: response 0.1..0.5 s after the lick
# (FINDINGS 0b's first-lick window); lick baseline -1.0..-0.6 s before the lick, splice baseline -0.5..0 s before the cue
BAL_BINS = {"cue": (slice(0, 5), slice(5, 15)), "cue25": (slice(0, 5), slice(5, 15)),
            "lick": (slice(0, 4), slice(11, 15)), "splice": (slice(0, 5), slice(16, 20))}
CACHE = ROOT / "cache"
# BASIS (O2 option 4, running list C). WF_SS_BASIS=lnmf reads the joint-LocaNMF per-trial files of the O2 job
# (`o2_state_space_lnmf.py`) instead of the Allen extract: the same rows for the rebuild sessions plus June contact-only
# sessions (epoch "june", never "pre"), its own binned cache, outputs tagged _lnmf. Columns are components named
# <Allen area>#<k>_<side> from the basis's region per component; the component set differs per animal, so
# `load_sessions` then returns {animal: cols} (`cols_of`) and scripts that stack areas across animals refuse
# (`allen_only`). WF_SS_BASIS=svd (Priya 11-08 "could we use one svd basis"): ONE cohort SVD basis (top 100 of the
# August pre sessions of all four animals on the shared Allen pixel grid, `o2_state_space_svd.py`), same rows; columns
# svd#<k>_<side> SHARED by all animals, so cross-animal stacking works. 97 / 100 components are bilateral, so `hemi`
# reads hemispheres / areas through the basis instead: an Allen area's mean is an exact linear readout of the
# components (the mean of U_cohort over the area's pixels, `_svd_area_weights`; Priya 11-08 "we can split right?").
BASIS = os.environ.get("WF_SS_BASIS", "allen")
LNMF = REBUILD / "o2" / "ss_lnmf_20261108" / "out" / "sessions"
SVD = REBUILD / "o2" / "ss_svdbm_20261108" / "out" / "sessions"     # brain mask, no glue / bulbs (ss_svd_: superseded)
if BASIS not in ("allen", "lnmf", "svd"):
    raise ValueError(f"WF_SS_BASIS={BASIS!r}")
SRC = {"lnmf": str(LNMF), "svd": str(SVD)}.get(BASIS, ITI)               # the per-trial files the loaders read
if BASIS != "allen":
    CACHE = ROOT / f"cache_{BASIS}"
# LocaNMF on the BRAIN MASK (Priya 11-08 "we should be looking only at the brain mask (without glue areas)" / "do we need
# to redo the locanmf as well with mask"): the joint-LocaNMF bases were fitted on the raw Allen mask (glue + bulbs in), so
# by default the components > 50 % of their mass in that animal's painted glue or an olfactory bulb are DROPPED
# (`component_exclusion.occluded`, the repo's route: no refit; ~1/4 of each basis). Outputs then carry _lnmfbm.
# WF_SS_LNMF_MASK=none keeps every component (the 11-08 morning runs, tagged _lnmf). The Allen extract (`area_basis`) and
# the cohort SVD (ss_svdbm) are on `beta_maps.brain_mask()` already.
LNMF_MASK = os.environ.get("WF_SS_LNMF_MASK", "brain")
if LNMF_MASK not in ("brain", "none"):
    raise ValueError(f"WF_SS_LNMF_MASK={LNMF_MASK!r}")
ALIGN_LABEL = {"cue": "cue-aligned -0.5..2.0 s", "cue25": "cue-aligned -0.5..2.5 s",
               "lick": "first-lick-aligned -1.0..0.7 s", "splice": "cue -0.5..0.5 s + first lick -0.5..0.7 s"}


def evoked(X):
    """THE response measure of every state-space size / balance readout: response-window mean minus baseline-window mean
    over the bin axis (-2) for the current alignment (`BAL_BINS`); X (..., bins, areas) -> (..., areas). Lick alignment:
    response 0.1..0.5 s after the first lick (FINDINGS 0b's first-lick window), baseline -1.0..-0.6 s. NOT 0b's primary
    marker itself (that is an angle-matched gain ratio vs the pre line, `lick_recovery_primary`; the angle-matched version
    here is `state_space_balance_angle`)."""
    base, resp = BAL_BINS[ALIGN]
    return X[..., resp, :].mean(-2) - X[..., base, :].mean(-2)


def hemi(v, cols, sel=None):
    """{"contra", "ipsi"}: mean of v (..., areas) over the contralesional (_right) / ipsilesional (_left) areas, restricted
    to the parent areas in ``sel`` if given (all four lesions are left: right = contralesional). NaN if none."""
    out = {}
    if BASIS == "svd":                                                  # components -> Allen area means -> sides
        W, names = _svd_area_weights()
        for h, suf in (("contra", "_right"), ("ipsi", "_left")):
            idx = [i for i, n in enumerate(names) if n.endswith(suf) and (sel is None or n[: -len(suf)] in sel)]
            out[h] = v @ W[:, idx].mean(-1) if idx else np.nan
        return out
    for h, suf in (("contra", "_right"), ("ipsi", "_left")):
        idx = [i for i, c in enumerate(cols) if c.endswith(suf) and (sel is None or c[: -len(suf)].split("#")[0] in sel)]
        out[h] = v[..., idx].mean(-1) if idx else np.nan
    return out


_SVDW: dict = {}


def _svd_area_weights():
    """(W (n_components, n_areas), area names '<area>_<side>') for the cohort SVD basis: W[:, a] = mean of U_cohort over
    the pixels of Allen area a (bundle atlas, same grid), so component scores @ W = the area means of the session's
    signal within the basis. Cached in CACHE."""
    if not _SVDW:
        f = CACHE / "svd_area_weights.npz"
        if not f.exists():
            Uc = np.load(SVD.parent / "basis" / "U_cohort.npy", mmap_mode="r")     # rows = brain-mask pixels
            at = np.load(SVD.parents[1] / "atlas.npy").ravel()[np.load(SVD.parent / "basis" / "mask.npy")]
            vals = sorted(int(x) for x in np.unique(at) if x != 0)
            W = np.stack([np.asarray(Uc[at == x], np.float64).mean(0) for x in vals], axis=1)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, W=W, names=np.array([_atlas_names()[x] for x in vals]))
        z = np.load(f)
        _SVDW.update(W=z["W"], names=[str(n) for n in z["names"]])
    return _SVDW["W"], _SVDW["names"]


def cols_of(cols, an):
    """The column names of animal ``an`` from `load_sessions`' cols (one list for the Allen basis, per animal for lnmf)."""
    return cols[an] if isinstance(cols, dict) else cols


def allen_only(what, shared_ok=False):
    """Refuse in the LocaNMF basis: ``what`` stacks or compares areas ACROSS animals (components differ per animal).
    ``shared_ok``: also fine in the cohort SVD basis (shared components; not for readouts that need named areas)."""
    if BASIS == "lnmf" or (BASIS == "svd" and not shared_ok):
        sys.exit(f"{what}: pools areas across animals -- Allen basis only (WF_SS_BASIS={BASIS})")


def btag():
    """The basis part of `tag` ('' for Allen, '_svd', '_lnmfbm', '_lnmf'): figure-name suffix of basis runs, so a basis
    run never overwrites the Allen page figure of the same name."""
    return ("" if BASIS == "allen" else f"_{BASIS}") + ("bm" if BASIS == "lnmf" and LNMF_MASK == "brain" else "")


BASIS_LABEL = {"allen": "", "svd": "cohort SVD basis (100 components, glue-free mask)",
               "lnmf": "LocaNMF components (per animal, glue / bulb components dropped)"}


def tag():
    """Output-folder suffix: '' for the original cue alignment, else '_<align>'."""
    return (("" if ALIGN == "cue" else f"_{ALIGN}") + ("" if K == 8 else f"_k{K}") + (f"_x{XFORM}" if XFORM else "")
            + ("" if BASIS == "allen" else f"_{BASIS}") + ("bm" if BASIS == "lnmf" and LNMF_MASK == "brain" else ""))


def _epoch(lab, ep):
    """June (month 06) sessions are the no-stroke drift reference, never 'pre' (epochs.epoch_of calls them pre)."""
    return "june" if lab.split("_")[1].startswith("06") else ep


_NAMES: dict = {}


def _atlas_names():
    """{atlas label -> '<area>_<side>'} (allen_area_names.json; the label convention is the same for every session)."""
    if not _NAMES:
        f = CACHE / "allen_area_names.json"
        if not f.exists():
            from wfield_local.locanmf_cue_lick_analysis import SESSIONS
            src = glob.glob(f"{SESSIONS[0]['mc']}/wfield_local_results/allen_aligned_affine8v1/allen_area_names.json")[0]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
        _NAMES.update({int(c): str(n) for c, n in json.loads(f.read_text(encoding="utf-8"))})
    return _NAMES


def _cols(z):
    """Column names of a per-trial file: the Allen extract's areas, or LocaNMF components <area>#<k>_<side>."""
    if "cols" in z.files:
        return [str(c) for c in z["cols"]]
    reg = np.asarray(z["regions"], int)
    if not len(reg):
        raise ValueError("LocaNMF file without regions (the basis's regions.npy was not staged)")
    seen, out = {}, []
    for r in reg:
        area, side = _atlas_names()[int(r)].rsplit("_", 1)
        seen[r] = seen.get(r, 0) + 1
        out.append(f"{area}#{seen[r]}_{side}")
    return out


def _bin(z, keep, align):
    parts = []
    for key, lags, bins in SEGMENTS[align]:
        x = z[key].astype(float)[:, :, keep]
        lag = lags[: x.shape[1]]
        parts.append(np.stack([np.nanmean(x[:, (lag >= a - 1e-9) & (lag < b - 1e-9)], axis=1) for a, b in bins], axis=1))
    return np.concatenate(parts, axis=1)


# MASK-KEYED BINNED CACHE (11-09, the rebuild glue-mask bug). The Allen-basis binned arrays are ROI means of the ITI
# extracts, so they depend on the analysis masks; the path used not to. They now live under
# <CACHE>/m<mask_keys.signature()>/<align>/ (the pre-fix <CACHE>/<align>/ folders are never read), and every extract read
# is checked against the current masks: these are REBUILD-ROOT extracts named explicitly (ITI), so one without mask_sig
# (built before the fix, glue IN) is refused whether or not the profile is active. The LocaNMF / SVD per-trial files
# are not on the analysis masks (LocaNMF: components dropped at load by `_lnmf_keep`; SVD: `_check_svd_mask`).
def _bcache():
    """The binned-cache root of this basis: mask-keyed for the Allen extract."""
    if BASIS != "allen":
        return CACHE
    from wfield_local import mask_keys
    return CACHE / f"m{mask_keys.signature()}"


def _check_extract(z, p):
    """An Allen ITI extract (it has `cols`) must be on the current analysis masks; component files pass."""
    if "cols" in z.files:
        from wfield_local import mask_keys
        mask_keys.check_obj(z, p, rebuild=True)


_SVD_OK: list = []


def _check_svd_mask():
    """The cohort SVD basis was built on `mask.npy` (shipped = `beta_maps.brain_mask()` at bundle time): refuse it when
    the current brain mask differs (a basis on other masks)."""
    if BASIS != "svd" or _SVD_OK:
        return
    from wfield_local import beta_maps
    f = SVD.parent / "basis" / "mask.npy"
    m = np.load(f).ravel().astype(bool)                               # saved ravelled (o2_state_space_svd MASK)
    bm = np.asarray(beta_maps.brain_mask(), bool).ravel()
    if m.size != bm.size or not np.array_equal(m, bm):
        raise RuntimeError(f"{f}: the SVD basis was built on another brain mask ({int(m.sum())} px) than the current "
                           f"one ({int(bm.sum())} px); rebuild the basis.")
    _SVD_OK.append(True)


def fill_cache(paths, cache=None):
    """One read of each extract -> binned arrays for EVERY alignment (<cache>/<align>/<label>.npz; default `_bcache()`)."""
    cache = Path(cache or _bcache())
    for p in paths:
        lab = Path(p).stem
        if all((cache / a / f"{lab}.npz").exists() for a in SEGMENTS):
            continue
        z = np.load(p, allow_pickle=True)
        _check_extract(z, p)
        cols = _cols(z)
        keep = [i for i, c in enumerate(cols) if c != "cortex"]
        for a in SEGMENTS:
            (cache / a).mkdir(parents=True, exist_ok=True)
            np.savez(cache / a / f"{lab}.npz", xb=_bin(z, keep, a), pos=np.array([str(q) for q in z["first_pos"]]),
                     cols=np.array([cols[i] for i in keep]), epoch=_epoch(lab, str(z["epoch"])), animal=str(z["animal"]),
                     rt=z["first_rt"])
        print("cached", lab, flush=True)


def _fill_one(item):
    """Worker: (path, cache dir, atlas names). Everything it needs travels in the item -- a spawned worker re-imports
    this module under the default env, so CACHE / BASIS / _NAMES there are NOT the caller's."""
    path, cache, names = item
    _NAMES.update(names)
    fill_cache([path], cache)
    return Path(path).stem


def prefill(paths, jobs=3):
    """Bin every uncached file once, fanned over a few workers (rule 6: these read the share; few, it is a shared box).
    The atlas names are resolved first, in this process, so workers never race to write them."""
    bc = _bcache()
    todo = [p for p in paths if not all((bc / a / f"{Path(p).stem}.npz").exists() for a in SEGMENTS)]
    if not todo:
        return
    names = dict(_atlas_names()) if BASIS == "lnmf" else {}
    from wfield_local import analysis_kit as kit
    _, fail = kit.fan_sessions([(p, str(bc), names) for p in todo], _fill_one, jobs=min(jobs, len(todo)), label="bin")
    if fail:
        raise RuntimeError(f"binning failed for {len(fail)} file(s): {fail}")


# DIMENSIONALITY VARIANTS (Priya 11-07: "this seems like an extraordinarily low dimensional state space" -> "can try
# options 1-4"). One cortex-wide mode carries 84-96 % of the mean response and 77-83 % of single-trial fluctuations.
# WF_SS_XFORM (outputs carry _x<name>), applied per trial at load, statistics from the animal's PRE sessions only:
#   z      each area divided by its pre SD (trial x bin values, per animal): every area weighs the same
#   gsr    project out the animal's pre cortex-wide mode (PC1 of pre single-trial x bin vectors over areas): like global
#          signal regression, with its caveats (it also removes any real cortex-wide change)
#   spout  subtract the session's cross-spout mean time course (mean of its spout means): the spout-specific part only;
#          removes the shared response AND any shared shift (shift_snr is then ~meaningless)
# Option 4 (a finer basis: LocaNMF components / pixels) needs a new per-trial extraction from SVT; not here.
XFORM = os.environ.get("WF_SS_XFORM", "")
_XSTATS: dict = {}


def _pre_stats(an):
    """Per-animal (area SD, cortex-wide unit vector) from the animal's pre sessions in the current alignment's cache."""
    if an not in _XSTATS:
        rows = []
        for f in sorted((_bcache() / ALIGN).glob(f"{an}_*.npz")):
            z = np.load(f)
            if str(z["epoch"]) == "pre":
                xb = _mask_cols(z["xb"], [str(x) for x in z["cols"]], an)[0]   # the same columns `load` returns
                rows.append(xb.reshape(-1, xb.shape[-1]))
        X = np.concatenate(rows)
        Xc = X - X.mean(0)
        w = np.linalg.svd(Xc[:: max(1, len(Xc) // 20000)], full_matrices=False)[2][0]
        _XSTATS[an] = (Xc.std(0), w / np.linalg.norm(w))
    return _XSTATS[an]


def _xform(xb, pos, an):
    if not XFORM:
        return xb
    if XFORM == "z":
        return xb / _pre_stats(an)[0]
    if XFORM == "gsr":
        w = _pre_stats(an)[1]
        return xb - (xb @ w)[..., None] * w
    if XFORM == "spout":
        means = [xb[pos == p].mean(0) for p in POS if (pos == p).any()]
        return xb - np.mean(means, axis=0)
    raise ValueError(f"WF_SS_XFORM={XFORM!r}")


def _lnmf_keep(animal):
    """bool (ncomp,): the pinned joint-LocaNMF components of ``animal`` NOT occluded by glue / bulb (cached in CACHE).

    The file name carries the digest of the animal's `component_exclusion.excluded_pixels` content (11-09): before the
    glue-mask fix that mask was bulbs-only under the rebuild profile, so the old ``lnmf_keep_<animal>_<bid>.npy`` files
    may keep glue components and are never read."""
    from wfield_local import joint_locanmf as jl
    from wfield_local import mask_keys
    bid = jl.pinned(animal)["id"]
    f = CACHE / f"lnmf_keep_{animal}_{bid}_x{mask_keys.exclusion_signature(animal)}.npy"
    if not f.exists():
        from wfield_local import component_exclusion as cx
        np.save(f, ~cx.occluded(jl.load(animal, bid=bid, warn_stale=False), animal))
    return np.load(f)


def _mask_cols(xb, cols, animal):
    """Drop the occluded LocaNMF components (`LNMF_MASK`); identity in every other basis."""
    if BASIS != "lnmf" or LNMF_MASK != "brain":
        return xb, cols
    keep = _lnmf_keep(animal)
    if len(keep) != len(cols):
        raise ValueError(f"{animal}: {len(cols)} columns vs {len(keep)} basis components")
    return xb[..., keep], [c for c, k in zip(cols, keep) if k]


def load(path):
    _check_svd_mask()
    c = _bcache() / ALIGN / f"{Path(path).stem}.npz"
    if BASIS != "allen" and not c.exists():
        fill_cache([path])                                             # compressed component files: bin once
    if c.exists():
        z = np.load(c)
        pos = z["pos"].astype(str)
        xb, cols = _mask_cols(z["xb"], [str(x) for x in z["cols"]], str(z["animal"]))
        return _xform(xb, pos, str(z["animal"])), pos, cols, str(z["epoch"]), str(z["animal"])
    if XFORM:
        raise FileNotFoundError(f"WF_SS_XFORM needs the binned cache: {c}")
    z = np.load(path, allow_pickle=True)
    _check_extract(z, path)
    cols = _cols(z)
    keep = [i for i, c in enumerate(cols) if c != "cortex"]
    xb = _bin(z, keep, ALIGN)                                          # trials x bins x areas
    pos = np.array([str(p) for p in z["first_pos"]])
    xb, cols = _mask_cols(xb, [cols[i] for i in keep], str(z["animal"]))
    return xb, pos, cols, _epoch(Path(path).stem, str(z["epoch"])), str(z["animal"])


def load_sessions(hook=None, june=False):
    """Every extract session -> ({animal: [(label, epoch, day, Y, pos, ...)] sorted by day}, area columns), the ONE
    session loader of the state-space scripts (`load` per file, days from `epochs.days_since_stroke`). ``hook(label, Y,
    pos)`` returns the tuple appended after (label, epoch, day) -- default (Y, pos); a relabel (trials) or an extra
    per-trial column (contact) goes there. Files in sorted path order, sessions sorted by day (stable): the order every
    caller used before (rule 9: pools need a defined order). LocaNMF basis: cols = {animal: cols} (`cols_of`). June
    sessions (epoch 'june', LocaNMF basis only) only with ``june=True``: many scripts read 'not pre' as post-stroke."""
    from wfield_local import epochs
    data, cols = {}, {}
    last = None
    files = [f for f in sorted(glob.glob(SRC + "/PS9*.npz")) if june or _epoch(Path(f).stem, "") != "june"]
    if BASIS != "allen":
        prefill(files)
    for f in files:
        Y, pos, last, ep, an = load(f)
        cols[an] = last
        lab = Path(f).stem
        data.setdefault(an, []).append((lab, ep, epochs.days_since_stroke(lab),
                                        *(hook(lab, Y, pos) if hook else (Y, pos))))
    return {a: sorted(v, key=lambda s: s[2]) for a, v in data.items()}, (cols if BASIS == "lnmf" else last)


def spout_means(Y, pos):
    return np.stack([Y[pos == p].mean(0) if (pos == p).any() else np.full(Y.shape[1:], np.nan) for p in POS])


def flat(M):
    return M.reshape(-1, M.shape[-1])


def topk(M, k=K):
    F = flat(M[np.isfinite(M).all(axis=(1, 2))])
    return np.linalg.svd(F - F.mean(0), full_matrices=False)[2][:k].T


def fit_gain(X, T):
    Tm, Xm = T.mean(0), X.mean(0)
    den = ((T - Tm) ** 2).sum(0)
    g = np.where(den > 0, ((T - Tm) * (X - Xm)).sum(0) / den, 1.0)
    return g, Xm - g * Tm


def rot_beyond_gain(X, T, P):
    """Held-out R2 (leave one spout out): L4 (per-area gain, then Procrustes in P) minus L3 (per-area gain)."""
    ok = [s for s in range(6) if np.isfinite(X[s]).all() and np.isfinite(T[s]).all()]
    sse3 = sse4 = sst = 0.0
    for h in ok:
        tr = [s for s in ok if s != h]
        Xtr, Ttr = flat(X[tr]), flat(T[tr])
        g, b = fit_gain(Xtr, Ttr)
        Ytr, Yte = Ttr * g + b, T[h] * g + b
        mu = Ytr.mean(0)
        u, _, vt = np.linalg.svd(((Ytr - mu) @ P).T @ ((Xtr - mu) @ P))
        Z = (Yte - mu) @ P
        pred4 = Yte + (Z @ (u @ vt) - Z) @ P.T
        sst += ((X[h] - Xtr.mean(0)) ** 2).sum()
        sse3 += ((X[h] - Yte) ** 2).sum()
        sse4 += ((X[h] - pred4) ** 2).sum()
    return (sse3 - sse4) / sst


def spread(Y, pos, P):
    d = []
    for p in POS:
        y = Y[pos == p]
        if len(y) > 1:
            z = flat(y - y.mean(0)) @ P
            d.append((z ** 2).sum() / len(y))
    return float(np.mean(d))


def readouts(Ya, pa, Yb, pb, P):
    A, B = spout_means(Ya, pa), spout_means(Yb, pb)
    ok = np.isfinite(A).all(axis=(1, 2)) & np.isfinite(B).all(axis=(1, 2))
    Af, Bf = flat(A[ok]), flat(B[ok])
    Ac, Bc = Af - Af.mean(0), Bf - Bf.mean(0)
    g, b0 = fit_gain(Af, Bf)
    As, Bs = A[ok].mean(0), B[ok].mean(0)
    # outside-P variance of what per-area gains CANNOT explain: separates new dimensions from region gains
    R = Af - (Bf * g + b0)
    amp = np.linalg.norm(Ac) / np.linalg.norm(Bc)
    return {"out_var": float(1 - ((Ac @ P) ** 2).sum() / (Ac ** 2).sum()),
            "r": float((Ac * Bc).sum() / np.linalg.norm(Ac) / np.linalg.norm(Bc)),
            "amp": float(amp),
            "out_resid": float((R - R @ P @ P.T).var() / Ac.var()),
            "rot_gain": float(rot_beyond_gain(A, B, P)),
            "gain_sd": float(np.std(np.log(np.clip(g, 1e-3, None)))),
            "shared": float(np.linalg.norm(As - As.mean(0)) / np.linalg.norm(Bs - Bs.mean(0))),
            "shared_dist": float(np.linalg.norm(As - Bs) / np.linalg.norm(Bs - Bs.mean(0))),
            "spread": spread(Ya, pa, P) / spread(Yb, pb, P),
            "spread_norm": spread(Ya, pa, P) / spread(Yb, pb, P) / amp ** 2}


def plant(Y, pos, kind, lvl, mu, P, ref_means, cols, rng):
    Yc = Y - mu
    if kind == "none":
        return Y
    if kind == "amp":
        return mu + lvl * Yc
    if kind == "gain_rand":
        return mu + Yc * np.exp(rng.normal(0, lvl, Y.shape[-1]))
    if kind == "gain_ipsi":
        g = np.array([lvl if c.endswith("_left") else 1.0 for c in cols])
        return mu + Yc * g
    if kind in ("rot", "rot12"):
        q = np.eye(P.shape[1])[:, :2] if kind == "rot12" else np.linalg.qr(rng.normal(size=(P.shape[1], 2)))[0]
        u, v = P @ q[:, 0], P @ q[:, 1]
        th = np.deg2rad(lvl)
        R = (np.eye(len(u)) + (np.cos(th) - 1) * (np.outer(u, u) + np.outer(v, v))
             + np.sin(th) * (np.outer(v, u) - np.outer(u, v)))
        return mu + Yc @ R.T
    shared = np.nanmean(ref_means, axis=0)                              # bins x areas
    if kind == "shift":
        w = P @ rng.normal(size=P.shape[1])
        w /= np.linalg.norm(w)
        prof = np.linalg.norm(shared - shared.mean(0), axis=1)
        add = np.outer(prof / np.linalg.norm(prof), w) * lvl * np.linalg.norm(shared - shared.mean(0))
        return Y + add
    if kind == "newdim":
        w = rng.normal(size=Y.shape[-1])
        w -= P @ (P.T @ w)
        w /= np.linalg.norm(w)
        code = ref_means - shared                                       # spout x bins x areas
        h = code @ P[:, 0]                                              # spout x bins: the main position code
        tot = (flat(np.nan_to_num(ref_means - flat(ref_means).mean(0))) ** 2).sum() / flat(ref_means).shape[0]
        h = h / np.sqrt(np.nanmean(h ** 2)) * np.sqrt(lvl * tot)
        add = np.stack([np.outer(np.nan_to_num(h[POS.index(p)]), w) for p in pos])
        return Y + add
    if kind == "spread":
        sd = np.sqrt(spread(Y, pos, P) / Y.shape[1] / P.shape[1])
        off = (rng.normal(0, lvl * sd, size=(len(Y), P.shape[1])) @ P.T)[:, None, :]
        return Y + off
    raise ValueError(kind)


def subsample(Y, pos, n, rng):
    idx = np.concatenate([rng.permutation(np.flatnonzero(pos == p))[:n] for p in POS])
    return Y[idx], pos[idx]


def run(n_per_spout, seed=0):
    rng = np.random.default_rng(seed)
    sess, cols_an = {}, {}
    for f in sorted(glob.glob(SRC + "/PS9*.npz")):
        Y, pos, cols, ep, an = load(f)
        if ep == "pre":
            sess.setdefault(an, []).append((Path(f).stem, Y, pos))
            cols_an[an] = cols                                          # per animal: LocaNMF components differ by animal
    rows = []
    for an, S in sorted(sess.items()):
        for i, (lab, Y, pos) in enumerate(S):
            others = [s for j, s in enumerate(S) if j != i]
            ref_means = np.nanmean(np.stack([spout_means(y, p) for _, y, p in others]), axis=0)
            P = topk(ref_means)
            mu = flat(ref_means[np.isfinite(ref_means).all(axis=(1, 2))]).mean(0)
            for kind, lvl in PLANTS:
                Yp = plant(Y, pos, kind, lvl, mu, P, ref_means, cols_an[an], rng)
                vals = []
                for _, Yo, po in others:
                    if n_per_spout:
                        n = min(n_per_spout, *[int((pos == p).sum()) for p in POS], *[int((po == p).sum()) for p in POS])
                        a, pa = subsample(Yp, pos, n, rng)
                        b, pb = subsample(Yo, po, n, rng)
                    else:
                        a, pa, b, pb = Yp, pos, Yo, po
                    vals.append(readouts(a, pa, b, pb, P))
                med = pd.DataFrame(vals).median().to_dict()
                rows.append({"animal": an, "label": lab, "plant": kind, "level": lvl, "n_per_spout": n_per_spout, **med})
    return pd.DataFrame(rows), cols_an


def detection(df):
    """Share of planted sessions beyond the pooled null's 2.5-97.5 % range, per readout (z within animal)."""
    mets = MET
    z = df.copy()
    for _an, g in df.groupby("animal"):
        nul = g[g.plant == "none"]
        for m in mets:
            z.loc[g.index, m] = (g[m] - nul[m].mean()) / nul[m].std(ddof=1)
    nul = z[z.plant == "none"]
    lo, hi = nul[mets].quantile(0.025), nul[mets].quantile(0.975)
    det = z.groupby(["plant", "level"])[mets].apply(lambda g: ((g < lo) | (g > hi)).mean())
    return det, z


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", nargs="+", type=int, default=[0, 15], help="trials per spout (0 = all, matched by session)")
    ap.add_argument("--out", default=str(OUT.with_name(OUT.name + tag())))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    order = list(dict.fromkeys(PLANTS))
    for n in a.n:
        df, _ = run(n)
        df.to_csv(out / f"sim_readouts_n{n}.csv", index=False)
        det, _ = detection(df)
        det = det.reindex(order)
        det.to_csv(out / f"sim_detection_n{n}.csv")
        with __import__("pandas").option_context("display.width", 200, "display.precision", 2):
            print(f"\n=== detection rate, n per spout = {n or 'all'} (null row = false-positive rate) ===")
            print(det)
            print("\nmedian readouts:")
            print(df.groupby(["plant", "level"])[MET]
                  .median().reindex(order))


if __name__ == "__main__":
    main()

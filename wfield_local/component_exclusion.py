"""WHICH JOINT-LocaNMF COMPONENTS SIT IN TERRITORY THE MAP ANALYSES REFUSE TO TEST.

Priya, 2026-09-24: *"for the map analyses we do in this repo, we masked the area covered by glue and
some rim regions. (1) should we apply that before doing CD on the locaNMF components? and bigger
blast radius (2) should we have done this for ALL the decoder analyses??"*

THE FACT THAT PROMPTED IT. `beta_maps.brain_mask()` subtracts the olfactory bulbs AND the
hand-painted fibre glue, and every MAP analysis goes through it. LocaNMF and every decoder load the
RAW `allen_brain_mask_native_grid.npy` instead, so the components were fitted over territory no map
will test. Measured here, per animal, at a 0.5 mass threshold:

    animal  ncomp  >50% under glue  >50% in a bulb   basis mass under glue
    PS92      95         24               4                 11.0%
    PS93      87         19               3                 10.7%
    PS94      90         20               4                 11.2%
    PS95      95         24               4                 12.6%

About a QUARTER of every animal's basis, and several components lie 100% inside the painted glue.

WHY THAT IS A SCOPE DIFFERENCE AND NOT AUTOMATICALLY A BUG. The exclusion is required wherever the
pixel's LOCATION is the claim: glue scatters light, so a map saying "this signal is in VISp" is false
where glue sits. A decoder or a coding direction claims that position is linearly readable from
cortical activity and asserts nothing about where, so a smeared-but-real signal costs spatial
precision, not validity. **Anywhere a component's weight is drawn on the brain or attributed to a
region, that reasoning stops applying** -- `locanmf_decoder_weights` loads the raw mask at both of its
map sites, and DECISIONS.md already records bulb and glue components reaching the max-statistic
family in `epoch_15h_rotation_regions`.

AND IT IS NOT CONTAMINATION, MEASURED. For the pre-cue coding direction the share of |w| on occluded
components (22.6-29.8%) matches their share BY COUNT (25.3-29.5%) in all four animals -- no
preferential loading. Dropping them costs 9-14% of the pole gap, which is what dropping a RANDOM
quarter costs (83.1-86.3% kept, against 85.8-91.0% for the occluded set: in PS92 the occluded quarter
was the least costly of 30 random draws). So this exists as an OPTION and a robustness arm, not as a
correction to results already published.

IT BECAME MORE PRESSING ON 2026-09-24, for an unrelated reason. `cd_trajectories` now Z-SCORES the
components before fitting (per-component sd spans 107x; the top five hold 82% of the variance). That
was right on its own terms, and it removed the amplitude protection that had been implicitly
down-weighting a dim, mostly-occluded component: after standardising, such a component is scaled to
unit variance and competes on equal terms.

WHAT THIS MODULE DOES NOT DO: refit LocaNMF on masked data. That would change `spec_id` for every
frozen model, so every post-stroke retained-fraction number would have to be refitted from a new
frozen pre-stroke model (rule 10, `docs/FROZEN_MODELS.md`). Dropping COLUMNS of an existing basis is
reversible and comparable; refitting is neither.
"""
from __future__ import annotations

import numpy as np

#: A component is "occluded" when this fraction of its spatial mass lies in excluded territory.
#:
#: 0.5 IS DELIBERATELY BLUNT and the distribution makes it safe: the counts at 0.10 / 0.25 / 0.50 are
#: 29/26/24 for PS92 and 30/28/24 for PS95, so almost every component that touches the glue at all is
#: MOSTLY in it. LocaNMF localises components to Allen regions, so a footprint tends to be inside the
#: occlusion or outside it rather than straddling the boundary, and the threshold sits in a flat part
#: of the curve rather than on a slope.
MASS_THRESHOLD = 0.5


def excluded_pixels(animal, include_bulbs=True):
    """``(H, W)`` bool -- that animal's painted glue, optionally with the olfactory bulbs.

    PER ANIMAL, NOT THE UNION. The glue is per animal, and charging PS92 for PS93's occlusion would
    throw away cortex that is perfectly good in three of the four -- the same reasoning
    `painted_exclusion` gives for the union being right for POOLED figures and wrong for per-animal
    ones.
    """
    from wfield_local import beta_maps as bm
    from wfield_local.paint_exclusion import mask_path

    bulbs = bm.excluded_mask()
    p = mask_path(animal)
    glue = np.load(p).astype(bool) if p.exists() else np.zeros(bulbs.shape, bool)
    return (glue | bulbs) if include_bulbs else glue


def mass_fraction(basis, animal, include_bulbs=True):
    """``(ncomp,)`` -- fraction of each component's |A| mass inside the excluded territory.

    NaN-SAFE, AND THAT IS NOT PEDANTRY. `Basis.A` carries NaN at off-brain pixels, so a reduction
    over it returns NaN; the first run of this measurement reported "0 components affected" for all
    four animals, because a count of NaNs and a count of zeros look identical in the output.
    """
    A = np.abs(np.nan_to_num(np.asarray(basis.A, dtype=np.float32)))
    ex = excluded_pixels(animal, include_bulbs)
    if ex.shape != A.shape[:2]:
        raise ValueError(f"mask {ex.shape} does not match basis footprints {A.shape[:2]}")
    flat = A.reshape(-1, A.shape[2])
    tot = flat.sum(0)
    out = np.full(A.shape[2], np.nan)
    ok = tot > 0
    out[ok] = (A * ex[:, :, None]).reshape(-1, A.shape[2]).sum(0)[ok] / tot[ok]
    return out


def occluded(basis, animal, thresh=MASS_THRESHOLD, include_bulbs=True):
    """``(ncomp,)`` bool -- the components to drop. A component with no mass at all counts as bad."""
    f = mass_fraction(basis, animal, include_bulbs)
    return ~np.isfinite(f) | (f > float(thresh))


def summarise(basis, animal, thresh=MASS_THRESHOLD):
    """One line of provenance, for a figure title or a log -- so a masked run says that it is one."""
    bad = occluded(basis, animal, thresh)
    return (f"{int(bad.sum())}/{len(bad)} components >{thresh:.0%} in glue/bulb "
            f"({bad.mean():.0%} of the basis) dropped")


__all__ = ["MASS_THRESHOLD", "excluded_pixels", "mass_fraction", "occluded", "summarise"]

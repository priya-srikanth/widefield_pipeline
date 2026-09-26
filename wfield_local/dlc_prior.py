"""Per-bodypart SPATIAL PRIORS for cam4 inference — mask the heatmap, then take the argmax.

**THE PROBLEM THIS SOLVES IS A CONFIDENT WRONG ANSWER, NOT A NOISY ONE.** Measured 2026-09-26 on a
20 s PS93 clip: nine frames put `spout` at x≈105 on bare fur, at likelihood **0.69–0.83**, while the
real spout sat at x≈370. A likelihood cutoff cannot catch that — 0.83 clears every threshold in this
project, `dlc.train.pcutoff` included — and it is exactly the kind of point that injects a spurious
270 px excursion into a kinematic measure that nobody would think to audit.

**MASKING BEATS REJECTING, AND THE DIFFERENCE IS THE WHOLE POINT.** The obvious fix is to drop
predictions outside a plausible box after the fact. That throws the frame away. Masking the heatmap
*before* the argmax instead suppresses the spurious peak and lets the SECOND peak win — which on all
nine frames is the real spout. Measured, same nine frames:

    no prior  ->  (103–109, 527)  p 0.69–0.83     on fur
    prior     ->  (345–374, 481–486)  p 0.26–0.81  on the spout

So the frames are recovered rather than lost. Post-hoc rejection would have scored nine dropouts.

**IT IS INERT ON GOOD FRAMES, AND THAT IS THE DESIGN.** Scored on the 96 held-out labelled frames,
the per-part error is byte-identical with and without the prior, at every padding tested down to
±10 px (nose 3.21, jaw 3.77, tongue 7.48, spout 1.85). Every ordinary prediction already lands inside
the box. This buys NOTHING on accuracy and is not an accuracy tool — it is insurance against the rare
gross failure. Do not expect it to move an RMSE, and do not read an unchanged RMSE as it not working.

**DERIVE THE BOXES FROM LABELS, NEVER FROM PREDICTIONS.** This is the trap that nearly landed on
2026-09-26: the first attempt at a crop width was taken from the extent of high-confidence
predictions over 10,000 frames, which put the spout's left edge at x=103 — i.e. it was set by the
very hallucination the prior exists to exclude, and would have widened the box to admit it. The
labelled points are human-verified; the predictions are the thing under test. `boxes()` reads
`CollectedData` only.

**THE SPOUT MOVES BETWEEN TRIALS, AND IT TURNS OUT NOT TO MATTER** (Priya, 2026-09-26: *"the spout
DURING TRIALS occupies 6 positions, but keep in mind that it moves between trials"*). The labelled
frames are trial-locked — `dlc_frames` samples cue/ENL/lick phases — so they see the spout AT its six
positions and never in transit, and a box drawn tight around them could in principle be too narrow in
exactly the interval no labelled frame covers. Measured over 400 s per animal (12,500 sampled frames
across PS93 and PS95, spanning many trials and ITIs): the spout's 0.5–99.5 percentile is x 279–416
against a labelled envelope of 215–440. **The transit never leaves the box.** The sampling gap is
real; the consequence feared from it is not.

**THE PART THAT ACTUALLY NEEDS THIS IS THE TONGUE.** The same measurement: 1.96% of CONFIDENT tongue
detections fall outside anatomical range, the worst at x=14 on a 680 px frame at p=0.66 — about 35
impossible points per 400 s of recording, on the part the whole study turns on. `spout` (0.10%) and
`jaw` (0.08%) are an order of magnitude cleaner, and `nose` never leaves its box at all (0 of
12,500). The spout hallucination is what made this visible; the tongue is what makes it worth having.

Usage — wraps DLC's own inference, no fork::

    from wfield_local import dlc_prior
    with dlc_prior.masking(dlc_prior.boxes()):
        deeplabcut.analyze_videos(cfg, [video], ...)

``offset`` is for when inference runs on a CROP: the boxes are in full-frame coordinates and the
heatmap is in crop coordinates, so the crop's top-left has to be subtracted. Getting this wrong
fails silently in the worst way — the mask lands somewhere else in the frame and quietly deletes
real peaks — so `masking` takes the offset explicitly rather than trying to infer it.
"""
from __future__ import annotations

import contextlib

import numpy as np

from wfield_local import config, dlc_train


def _cfg() -> dict:
    return ((config.defaults().get("dlc") or {}).get("prior") or {})


def pad() -> float:
    """Padding in px added on every side of the labelled envelope.

    Covers three things the labelled set under-samples, all of which are real: the spout IN TRANSIT
    between trials (no labelled frame is in that interval), cross-animal and cross-session variation
    in head position, and the plain fact that 24 frames per session is a thin sample of the extremes.
    Default 60 px, which on the measured envelope leaves every labelled point at least 78 px from a
    box edge while still excluding the x≈105 fur hallucination by a wide margin.
    """
    return float(_cfg().get("pad", 60))


def boxes(rv=None, cam: str | None = None, padding: float | None = None,
          frame: tuple[int, int] = (680, 680)) -> dict[str, tuple[float, float, float, float]]:
    """``bodypart -> (x0, x1, y0, y1)`` in FULL-FRAME pixels, from the human labels plus padding.

    Clipped to the frame, because a box that runs off the sensor is not wrong but is misleading to
    read in a log. The `tongue` legitimately reaches the bottom edge at full protrusion (measured
    y=674 of 680), so the lower clip is load-bearing rather than defensive.
    """
    p = pad() if padding is None else float(padding)
    proj = dlc_train.train_project(rv)
    lab = dlc_train.labels(proj, cam or dlc_train.cam())
    out: dict[str, tuple[float, float, float, float]] = {}
    for bp in dlc_train.parts():
        a = dlc_train._xy(lab, bp)
        a = a[~np.isnan(a[:, 0])]
        if not len(a):
            raise SystemExit(f"no labelled points for {bp} — cannot build a prior for it")
        out[bp] = (max(0.0, a[:, 0].min() - p), min(frame[0], a[:, 0].max() + p),
                   max(0.0, a[:, 1].min() - p), min(frame[1], a[:, 1].max() + p))
    return out


@contextlib.contextmanager
def masking(box: dict[str, tuple[float, float, float, float]],
            bodyparts: list[str] | None = None, offset: tuple[float, float] = (0.0, 0.0)):
    """Patch DLC's ``HeatmapPredictor`` so every heatmap is masked to ``box`` before the argmax.

    A context manager and a CLASS patch rather than an argument, because `analyze_videos` builds
    its own model internally and never hands the caller the predictor. The patch is removed on exit
    including on an exception, so a failed run cannot leave a global monkeypatch behind for the next
    one — which would be a silent, sticky change to results.

    ``bodyparts`` must be in the model's HEAD-CHANNEL order. It defaults to `dlc.train.bodyparts`,
    which is that order by construction (`dlc_train.conversion` builds the head from this list), but
    it is exposed because a model trained from a different list would otherwise be masked channel-by-
    wrong-channel — a failure that produces plausible output and no error.
    """
    import torch
    from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import HeatmapPredictor

    bps = list(bodyparts or dlc_train.parts())
    missing = [b for b in bps if b not in box]
    if missing:
        raise SystemExit(f"no prior box for {', '.join(missing)} — refusing to mask partially")

    original = HeatmapPredictor.forward
    ox, oy = float(offset[0]), float(offset[1])
    # The mask depends only on (heatmap shape, stride, device) -- all fixed for a whole video -- so
    # it is built once and reused rather than rebuilt per frame.
    #
    # ITS MEASURED BENEFIT IS BELOW NOISE, and that is worth stating rather than implying otherwise.
    # Caching moved a real run 83.9 -> 85.9 frames/s (~2%), and repeat timings of the SAME condition
    # on this box vary by more than that (full frame + no prior measured 62.3 and 43.6 frames/s in
    # two runs an hour apart). So this is tidiness with a plausible mechanism, not a demonstrated
    # optimisation; do not quote a speed-up for it. The prior's own overhead is likewise not
    # resolvable at this precision -- one attribution run had it apparently FASTER than no prior,
    # which is only variance.
    cache: dict = {}

    def patched(self, stride, outputs):
        hm = outputs["heatmap"]
        if hm.shape[1] != len(bps):
            # Channel count and bodypart list disagree: masking would apply each box to the wrong
            # part. Refuse rather than guess -- the output would look fine.
            raise SystemExit(f"prior has {len(bps)} bodyparts, heatmap has {hm.shape[1]} channels")
        h, w = hm.shape[-2:]
        s = float(stride)
        key = (h, w, s, str(hm.device), hm.dtype)
        if key not in cache:
            # Cell centres in FULL-FRAME pixels. `+ offset` converts crop coords back to frame
            # coords; the boxes are always in frame coords so there is one place this can be wrong,
            # and it is here.
            gy, gx = torch.meshgrid(
                torch.arange(h, device=hm.device, dtype=hm.dtype) * s + oy,
                torch.arange(w, device=hm.device, dtype=hm.dtype) * s + ox, indexing="ij")
            drop = torch.zeros((len(bps), h, w), device=hm.device, dtype=torch.bool)
            for k, bp in enumerate(bps):
                x0, x1, y0, y1 = box[bp]
                ok = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
                if not bool(ok.any()):
                    raise SystemExit(f"prior box for {bp} covers no heatmap cell at stride {s}")
                drop[k] = ~ok
            cache[key] = drop
        hm = hm.clone()                      # never mutate the head's own output in place
        hm.masked_fill_(cache[key].unsqueeze(0), -1e4)   # a floor, not -inf: sigmoid(-1e4) is 0
        outputs = dict(outputs)
        outputs["heatmap"] = hm
        return original(self, stride, outputs)

    HeatmapPredictor.forward = patched
    try:
        yield box
    finally:
        HeatmapPredictor.forward = original

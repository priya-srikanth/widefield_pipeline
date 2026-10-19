"""Held-out JAW fill error vs gap length (Priya 10-19: is 24 ms -- set for the fast tongue -- right for the slow jaw?).

    python -m scripts.pose3d_jaw_gap_check      # dlc env

PS92 0807 + 0818: hide one camera on both-camera jaw frames in blocks of 24 / 48 / 100 / 200 ms, fill with no gap limit
(Akima, 3 mm ray gate), compare to the truth; plus how many real one-camera jaw frames each limit would cover.
"""
import numpy as np

from scripts.apparatus_mouth import _rig
from scripts.pose3d import _cgroup
from wfield_local import o2_inference as oi
from wfield_local import pose3d as P3
from wfield_local import spout_frame as SF
from wfield_local.paths import PathResolver

rv = PathResolver()
rng = np.random.default_rng(0)
cg0 = _cgroup(rv).subset_cameras_names(["cam1", "cam4"])
for d in ("20260807", "20260818"):
    c1 = oi.session_windows(rv, "lp_cam1_ctx_prod_20261019", "PS92", d, "cam1")
    c4 = oi.session_windows(rv, "lp_r4_prod_20261015", "PS92", d, "cam4")
    m1, m4 = SF.ruler_medians(c1[1], c1[0]), SF.ruler_medians(c4[1], c4[0])
    cg, _ = P3.pnp_group(cg0, {"cam1": m1, "cam4": m4}, _rig(rv, "PS92", d)["geometry"])
    i1, i4 = P3.pair(c1[0], c4[0])
    trial, t = c1[0].trial_id.to_numpy()[i1], c1[0].t_ms.to_numpy(float)[i1]
    a, _ = P3.cleaned_xy(c1[1], c1[0], "jaw", "cam1", SF.from_medians_lr(m1).origin)
    b, _ = P3.cleaned_xy(c4[1], c4[0], "jaw", "cam4", SF.from_medians(m4).origin)
    a, b = a[i1], b[i4]
    X, _ = P3.triangulate(cg, a, b)
    both = np.isfinite(X).all(1)
    # how long are the real one-camera jaw gaps? (runs of one-camera frames between both-camera frames, same trial)
    one = (np.isfinite(a).all(1) ^ np.isfinite(b).all(1))
    runs, i = [], 0
    while i < len(one):
        if one[i]:
            j = i
            while j < len(one) and one[j] and trial[j] == trial[i]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    runs = np.array(runs)
    frames_in = {L: int(runs[runs * 4 <= L].sum()) for L in (24, 48, 100, 200, 1e9)}
    print(f"PS92 {d}: one-camera jaw frames {runs.sum()}; share in gaps <= 24 / 48 / 100 / 200 ms / any: "
          + " / ".join(f"{frames_in[k] / runs.sum():.0%}" for k in (24, 48, 100, 200, 1e9)))
    idx = np.flatnonzero(both)
    for L in (6, 12, 25, 50):
        ok = [s for s in idx[::7] if s + L + 3 < len(X) and s - 3 >= 0 and both[s - 3:s + L + 3].all()
              and (trial[s - 3:s + L + 3] == trial[s]).all()]
        if len(ok) < 50:
            print(f"   gap {4 * L} ms: too few long both-camera jaw runs ({len(ok)})")
            continue
        starts = rng.choice(ok, min(2000, len(ok)), replace=False)
        m = np.zeros(len(X), bool)
        for s0 in starts:
            m[s0 + 1:s0 + 1 + L] = True
        for hide, name in ((1, "cam1 only"), (0, "cam4 only")):
            a2, b2, Xh = a.copy(), b.copy(), X.copy()
            (b2 if hide else a2)[m] = np.nan
            Xh[m] = np.nan
            F, _ = P3.fill_single_view(Xh, a2, b2, cg, trial, t, max_gap_ms=1e9, interp="akima")
            e = np.linalg.norm(F[m] - X[m], axis=1)
            e = e[np.isfinite(e)]
            print(f"   gap {4 * L:3d} ms {name}: n={len(e):6d}  median {np.median(e):.3f} mm  p90 {np.percentile(e, 90):.2f}")

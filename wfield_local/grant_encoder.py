"""GRANT FIGURES — ENCODER

The encoder: gain versus shape, and how much of the pre-stroke code survives.

Split out of `grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS FROM THE
CALL GRAPH, not from the names: every function here is reached from this family's entry points and
from no other family's. Anything shared with a sibling lives in `grant_kit` -- which is why this
imports from there and never from `grant_figures`, a direction that would be a cycle.

Entry points, registered in `grant_figures.JOBS`:
  - `fig_encoder_gain_shape`
  - `fig_coding_retained`
  - `fig_frozen_vs_within`
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wfield_local import config
from wfield_local.grant_kit import (  # noqa: F401
    _BUNDLE_CACHE,
    ANIMALS,
    BOOT_CACHE_VERSION,
    CONF_LABELS,
    N_BOOT_DELTA,
    N_BOOT_RDM,
    N_LOO_DRAW,
    POS,
    POS_SHORT,
    WINDOWS,
    _anchor,
    _best_match,
    _block_boot,
    _block_index,
    _boot_cached,
    _cd_labels,
    _class_note,
    _class_select,
    _collect_7,
    _colw,
    _corr_matrix,
    _day,
    _delta_cis,
    _delta_diag_ci,
    _delta_diag_one,
    _delta_grid,
    _diag,
    _digest,
    _excludes_zero,
    _feed,
    _fig_root,
    _fit_bottom,
    _fit_header,
    _footer,
    _impaired,
    _matrices_pattern,
    _mats_pattern,
    _means,
    _nanmean_stack,
    _out,
    _overlaps,
    _pct3,
    _pooled_bundle,
    _position_metrics,
    _pre_reference,
    _runs_to_blocks,
    _save,
    _seed,
    _session_trials,
    _sessions,
    _sg_labels,
    _short,
    _suptitle,
    _twinned,
    _txt,
    _variants,
    _windows,
    coverage_note,
    only,
    pos_style,
    set_only,
)


def fig_encoder_gain_shape(out_dir, min_trials=10):
    """11: the FROZEN ENCODER -- and the amplitude-versus-tuning question answered directly.

    Priya, 2026-08-26: *"then consider what encoder analyses make the most sense. I like the
    pre-stroke then post-stroke by session analysis structure."* This is the forward model, in that
    structure, and it is the first encoder figure in the grant set.

    WHY AN ENCODER EARNS ITS PLACE HERE. The decoder asks whether position can be READ OUT of
    cortex; it answers with one number per session and, because it pools across components by
    construction, it cannot say what changed. The encoder asks whether the position -> activity
    MAPPING still holds, and its residual is a per-position, per-component object. More to the
    point, it is the only framing in which the question figures 6, 7, 8 and 8b have circled for a
    fortnight -- did the code MOVE, or did it merely get SMALLER? -- becomes two separately
    estimated numbers instead of two readings of one.

    A frozen encoder with a one-hot position design trained on pre-stroke sessions only predicts,
    for a trial at position q, the pre-stroke mean pattern at q. Fitting ONE gain for the session
    splits its failure (`_enc_terms`):

    LEFT -- how well the frozen model transfers, without rescaling (`raw`) and with (`gain`). The
    SHADED GAP between them is what rescaling recovers. Read `gain` first: a high `gain` with a gap
    means the code is intact and smaller, a low `gain` means the tuning itself changed and no
    rescaling saves it. A LARGE GAP IS NOT BY ITSELF AN AMPLITUDE RESULT -- an unrelated code also
    recovers a lot, because the best gain collapses towards zero and predicting nothing beats
    predicting something wrong.

    MIDDLE -- the fitted gain. 1.0 is no amplitude change; below it the whole position code is
    weaker, above it stronger. This is the quantity every correlation-based panel here is blind to
    by construction and every distance-based panel is dominated by.

    RIGHT -- per position, what is left after the session gain is removed: a genuine tuning change,
    localised. A boxed cell is one whose change from the pre-stroke ceiling excludes zero.

    THE PRE COLUMN IS THE CEILING, leave-one-session-out, and it is not 1.0: a held-out pre-stroke
    day does not reproduce the others exactly either. Read every post column against it.

    NO MOVEMENT REGRESSORS (no DLC yet). A position -> activity encoder attributes to POSITION
    anything that co-varies with it, including how differently the animal moves to reach each spout,
    so a post-stroke change in movement would appear here as a change in tuning. The `lick` class,
    the pre-cue window (which contains no lick at all) and the `working` class each bound that
    differently; a difference that holds across all three is not a movement artefact, and one that
    appears only in the lick window probably is.
    """
    made = []
    for _disp, align, wname in _windows():
        for v in _variants(align):
            tab, days = _enc_tables(align, v, min_trials)
            if not tab or not days:
                continue
            cis, _cd = _enc_ci(align, v, min_trials)
            fig, axes = plt.subplots(
                len(ANIMALS), 3, squeeze=False,
                figsize=(8.4 + 0.62 * len(days), 2.5 * len(ANIMALS) + 2.0),
                gridspec_kw={"width_ratios": [1.5, 1.0, 1.3], "hspace": 0.42, "wspace": 0.30})
            drew = False
            cols = ["PRE"] + list(days)
            xs = np.arange(len(cols))
            lab_c = ["PRE"] + [f"d{d}" for d in days]
            for ri, an in enumerate(ANIMALS):
                rec = tab.get(an)
                if not rec:
                    for ci in range(3):
                        axes[ri][ci].axis("off")
                    continue
                crec = cis.get(an) or {}
                drew = True

                def _tr(idx, rec=rec, cols=cols):
                    return np.array([rec[c][idx] if c in rec else np.nan for c in cols], float)

                def _err(key, crec=crec, cols=cols):
                    """(lower, upper) bar lengths from the stored percentile interval."""
                    lo = np.full(len(cols), np.nan)
                    hi = np.full(len(cols), np.nan)
                    for j, c in enumerate(cols):
                        iv = (crec.get(c) or {}).get(key)
                        if iv:
                            lo[j], hi[j] = iv[0], iv[1]
                    return lo, hi

                # ---------------- left: transfer, with and without a refitted gain
                ax = axes[ri][0]
                raw, gain = _tr(0), _tr(2)
                m = np.isfinite(raw) & np.isfinite(gain)
                if m.any():
                    # THE SHADED GAP is what rescaling recovers. Drawn under the traces so neither
                    # line is obscured by it.
                    ax.fill_between(xs[m], raw[m], gain[m], color="#f0a202", alpha=0.28, lw=0,
                                    label="recovered by rescaling" if ri == 0 else None)
                for key, y, col, mk, nm in ((("raw"), raw, "#b2182b", "o", "frozen, as fitted"),
                                            (("gain"), gain, "#2166ac", "s", "after one gain")):
                    lo, hi = _err(key)
                    ok = np.isfinite(y)
                    e = np.vstack([np.where(np.isfinite(lo), y - lo, np.nan),
                                   np.where(np.isfinite(hi), hi - y, np.nan)])
                    ax.errorbar(xs[ok], y[ok], yerr=np.abs(e[:, ok]), fmt=mk + "-", color=col,
                                ms=4.5, lw=1.4, elinewidth=1.0, capsize=2.5,
                                label=nm if ri == 0 else None)
                ax.axhline(0, color="k", lw=0.8)
                # THE LIMIT COMES FROM THE INTERVAL BOUNDS, NOT THE POINTS. Scaling to the points
                # alone cut PS93 day 3's lower bar off at the axis floor: the estimate is -1.63 and
                # the interval reaches -2.42, so the figure drew a bar that stopped where the axis
                # did and understated the uncertainty exactly where it was largest.
                lo_r, _hr = _err("raw")
                lo_g, _hg = _err("gain")
                floor = np.nanmin(np.concatenate([raw, gain, lo_r, lo_g, [0.0]]))
                ax.set_ylim(max(-3.5, floor - 0.12) if np.isfinite(floor) else -0.5, 1.10)
                ax.set_ylabel(f"{an}\nvariance explained", fontsize=10.5, fontweight="bold")
                if ri == 0:
                    ax.set_title("does the position->activity map transfer?", fontsize=10.5)

                # ---------------- middle: the fitted gain
                ax1 = axes[ri][1]
                a = _tr(1)
                lo, hi = _err("a")
                ok = np.isfinite(a)
                e = np.vstack([np.where(np.isfinite(lo), a - lo, np.nan),
                               np.where(np.isfinite(hi), hi - a, np.nan)])
                ax1.errorbar(xs[ok], a[ok], yerr=np.abs(e[:, ok]), fmt="D-", color="#4d4d4d",
                             ms=4, lw=1.3, elinewidth=1.0, capsize=2.5)
                ax1.axhline(1.0, color="#1a9850", lw=1.2, ls="--")
                if np.isfinite(a[0]):
                    ax1.axhline(a[0], color="0.55", lw=1.0, ls=(0, (1, 2)))
                # Same rule here: the bars, not the diamonds, decide the limits.
                _top = np.nanmax(np.concatenate([a, hi, [1.05]])) if ok.any() else 1.6
                _bot = np.nanmin(np.concatenate([a, lo, [0.0]])) if ok.any() else 0.0
                ax1.set_ylim(min(-0.1, _bot - 0.08), max(1.6, _top + 0.08))
                ax1.set_ylabel("fitted gain", fontsize=9.5)
                if ri == 0:
                    ax1.set_title("amplitude of the whole\nposition code (1 = unchanged)",
                                  fontsize=10.5)

                # ---------------- right: what is left per position after the gain
                ax2 = axes[ri][2]
                G = np.full((len(CONF_LABELS), len(cols)), np.nan)
                for j, c in enumerate(cols):
                    for i, q in enumerate(CONF_LABELS):
                        if c in rec and q in rec[c][3]:
                            G[i, j] = rec[c][3][q]
                imh = ax2.imshow(np.ma.masked_invalid(G), vmin=-1, vmax=1, cmap="RdBu_r",
                                 aspect="auto")
                for i in range(len(CONF_LABELS)):
                    for j in range(len(cols)):
                        if np.isfinite(G[i, j]):
                            _txt(ax2, j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=7)
                        if j == 0:
                            continue
                        iv = ((crec.get(cols[j]) or {}).get("dpos") or {}).get(CONF_LABELS[i])
                        if _excludes_zero(iv):
                            ax2.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                        edgecolor="k", lw=1.6, zorder=5))
                ax2.set_yticks(range(len(CONF_LABELS)))
                ax2.set_yticklabels(_short(CONF_LABELS), fontsize=9)
                if ri == 0:
                    ax2.set_title("tuning left per position\n(gain already removed)", fontsize=10.5)

                last = ri == len(ANIMALS) - 1
                for axx in (ax, ax1):
                    axx.set_xticks(xs)
                    axx.set_xticklabels(lab_c if last else [], fontsize=9, rotation=45,
                                        ha="right" if last else "center")
                    axx.grid(alpha=0.25, lw=0.5)
                ax2.set_xticks(xs)
                ax2.set_xticklabels(lab_c if last else [], fontsize=9, rotation=45,
                                    ha="right" if last else "center")
                if last:
                    for axx in (ax, ax1, ax2):
                        axx.set_xlabel("days from lesion", fontsize=10)
            if not drew:
                plt.close(fig)
                continue
            h, lb = axes[0][0].get_legend_handles_labels()
            if h:
                fig.legend(h, lb, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
                           bbox_to_anchor=(0.5, 0.012))
            cls = _class_note(v)
            # LAY THE PANELS OUT INTO WHAT THE HEADER ACTUALLY LEFT. Calling `tight_layout` FIRST
            # and then `_suptitle` -- which compresses every axes into [0, top] afterwards -- left
            # a tenth of the figure blank between the header and the first panel title, because
            # tight_layout had spread the panels over the full height and the compression then
            # shrank them away from it. `_suptitle` returns that `top`, so the layout can simply
            # target it.
            top = _suptitle(fig,
                      f"FROZEN ENCODER: did the position code MOVE, or just get SMALLER? -- "
                      f"{wname} window\n"
                      f"Post-stroke class: {cls}.  A one-hot position encoder trained on pre-stroke "
                      f"sessions ONLY predicts each position's pre-stroke mean pattern; one gain "
                      f"fitted per session splits its failure in two.\n"
                      f"READ THE GAIN FIRST. High 'after one gain' with a wide shaded gap = the "
                      f"code is intact and WEAKER. Low 'after one gain' = the tuning itself "
                      f"changed, and no rescaling saves it. A wide gap alone is not an amplitude "
                      f"result: an unrelated code also recovers a lot, by collapsing the gain "
                      f"towards zero.\n"
                      f"PRE column = leave-one-session-out and is NOT 1.0. Boxed cell = change from "
                      f"it excludes zero (95% block bootstrap, sessions held fixed).  A BLANK CELL "
                      f"is a position with too few trials that session, NOT a zero.  NO MOVEMENT "
                      f"REGRESSORS yet: a post-stroke change in how the animal moves would appear "
                      f"here as a change in tuning.")
            fig.tight_layout(rect=(0, 0.075, 1, top))
            # THE SCALE FOR THE RIGHT-HAND PANEL, attached to that column only so it lands at the
            # figure's right edge and cannot sit between panels. `--compact` drops the in-cell
            # numbers, and this is then the only thing telling red from blue. AFTER `tight_layout`,
            # for the reason spelled out in figure 8b: a colour bar made before it does not move
            # when the panels do.
            fig.colorbar(imh, ax=axes[:, 2].tolist(), fraction=0.02, pad=0.03,
                         label="tuning left (R^2 after the session gain)")
            _footer(fig)
            p = _out(out_dir, f"grant_11_encoder_gain_shape_{align}_{v}")
            _save(fig, p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            made.append(p)
    return made
def _enc_terms(m_by_q, p_by_q):
    """Split a FROZEN ENCODER's failure into an amplitude part and a tuning part.

    A frozen encoder with a one-hot position design and a pre-stroke-only training set predicts, for
    a trial at position q, the pre-stroke mean pattern p_q -- ridge on a one-hot design IS the
    per-position mean, shrunk. Its residual on a post-stroke session mixes the two hypotheses the
    rest of this figure set has spent a fortnight unable to separate: the response may be the same
    shape but SMALLER, or it may genuinely have changed shape. ONE gain fitted for the whole session
    splits them:

        raw   = 1 - sum|m - p|^2 / sum|m|^2      what the frozen encoder actually achieves
        a     = sum m.p / sum p.p                the single best gain for this session
        gain  = 1 - sum|m - a p|^2 / sum|m|^2    what it would achieve if allowed to rescale

    ``1 - gain`` is what survives rescaling and is a genuine change in TUNING. ``gain - raw`` is the
    part of the failure that rescaling recovers -- **which is an amplitude change only when `gain`
    itself is high.** A session whose code is simply gone also recovers a lot, because the best gain
    collapses towards zero and predicting nothing beats predicting an unrelated pattern: unrelated
    patterns give raw = -1.37, gain = 0.01, a difference of 1.38 that is not an amplitude story at
    all. READ `gain` FIRST, then `a`:

        gain high, a far from 1   -> same code, smaller (or larger). Pure amplitude.
        gain high, a near 1       -> nothing changed.
        gain low                  -> the tuning changed, whatever `a` says.

    THE GAIN IS ONE NUMBER PER SESSION, deliberately: a per-position gain would absorb the
    position-specific amplitude loss that IS the deficit, and the decomposition would say nothing.

    Patterns are centred on the session's own mean across positions first, so a session-wide shift
    in F0 or SNR -- which carries no position information and which the encoder is not being asked
    to predict -- is charged to neither term.

    Returns ``(raw, a, gain, {position: shape r2 after the gain})`` or NaNs when fewer than two
    positions are shared.
    """
    qs = [q for q in CONF_LABELS if q in m_by_q and q in p_by_q]
    nan = (np.nan, np.nan, np.nan, {})
    if len(qs) < 2:
        return nan
    M = np.stack([m_by_q[q] for q in qs])
    P = np.stack([p_by_q[q] for q in qs])
    M = M - M.mean(0)
    P = P - P.mean(0)
    tot = float((M ** 2).sum())
    pp = float((P ** 2).sum())
    if tot <= 1e-12 or pp <= 1e-12:
        return nan
    raw = 1.0 - float(((M - P) ** 2).sum()) / tot
    a = float((M * P).sum()) / pp
    gain = 1.0 - float(((M - a * P) ** 2).sum()) / tot
    per = {}
    for k, q in enumerate(qs):
        d = float((M[k] ** 2).sum())
        if d > 1e-12:
            per[q] = 1.0 - float(((M[k] - a * P[k]) ** 2).sum()) / d
    return raw, a, gain, per
def _enc_scores(src, ref):
    """`_enc_terms` on the MEAN patterns of a scored set against a reference set."""
    return _enc_terms(_means(src), _means(ref))
@lru_cache(maxsize=6)
def _enc_tables(align, variant, min_trials=10):
    """{animal: {"PRE"|day: (raw, a, gain, per-position)}} for the encoder figure, plus days.

    The PRE entry is LEAVE-ONE-SESSION-OUT and averaged over the held-out sessions -- the same
    construction as every other pre-stroke column here. Without it the reference would contain the
    session being scored and the encoder would read as near-perfect by construction.
    """
    store, days = _collect_7(align, variant, min_trials)
    out = {}
    for an, (pre_by_sess, by_day) in store.items():
        if len(pre_by_sess) < 2 or not by_day:
            continue
        rec = {}
        loo = [_enc_scores(pat, _pre_reference(pre_by_sess, exclude=s))
               for s, pat in pre_by_sess.items()]
        loo = [t for t in loo if np.isfinite(t[0])]
        if loo:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per = {q: float(np.nanmean([t[3][q] for t in loo if q in t[3]]))
                       for q in CONF_LABELS if any(q in t[3] for t in loo)}
                rec["PRE"] = (float(np.nanmean([t[0] for t in loo])),
                              float(np.nanmean([t[1] for t in loo])),
                              float(np.nanmean([t[2] for t in loo])), per)
        full = _pre_reference(pre_by_sess)
        for d, pat in by_day.items():
            t = _enc_scores(pat, full)
            if np.isfinite(t[0]):
                rec[d] = t
        if len(rec) > 1:
            out[an] = rec
    return out, days
@lru_cache(maxsize=6)
def _enc_ci(align, variant, min_trials=10, n_boot=N_BOOT_RDM, n_loo=N_LOO_DRAW):
    """Block-bootstrap intervals for the encoder figure, built exactly like `_rdm_ci`.

    Every draw resamples the scheduler's position blocks within each session, holds the SESSIONS
    fixed, and scores the day and the leave-one-out pre-stroke ceiling in the SAME draw, so the
    delta is taken draw by draw and the two share their noise.

    Returns ``({animal: {"PRE"|day: rec}}, days)`` where each ``rec`` carries ``raw``, ``a``,
    ``gain`` and ``pos`` intervals, and a post-stroke ``rec`` also carries ``d*`` versions -- the
    change from the pre-stroke ceiling, which is the only quantity the figure makes a claim about.
    """
    x_store, days = _collect_7(align, variant, min_trials)
    b_store, _ = _collect_7(align, variant, min_trials, "blk")
    out = {}
    for an in ANIMALS:
        if an not in x_store or an not in b_store:
            continue
        (pre_x, day_x), (pre_b, day_b) = x_store[an], b_store[an]
        if len(pre_x) < 2 or not day_x:
            continue
        rng = np.random.default_rng(_seed(an, align, variant, "encci"))
        keys3 = ("raw", "a", "gain")
        base = {k: [] for k in keys3}
        base["pos"] = []
        acc = {d: {k: [] for k in list(keys3) + ["d" + k for k in keys3] + ["pos", "dpos"]}
               for d in day_x}
        try:
            for _ in range(n_boot):
                drawn = {s: _block_boot(pre_x[s], pre_b[s], rng) for s in sorted(pre_x)}
                drawn = {s: v for s, v in drawn.items() if v}
                if len(drawn) < 2:
                    continue

                def _pool(exclude=None, drawn=drawn):
                    a = {}
                    for s, Z in drawn.items():
                        if s == exclude:
                            continue
                        for q, z in Z.items():
                            a.setdefault(q, []).append(z)
                    return {q: np.vstack(v) for q, v in a.items()}

                held = list(drawn)
                if len(held) > n_loo:
                    held = [held[k] for k in rng.choice(len(held), n_loo, replace=False)]
                got = [_enc_scores(drawn[s], _pool(exclude=s)) for s in held]
                got = [t for t in got if np.isfinite(t[0])]
                if not got:
                    continue
                ceil = [float(np.mean([t[i] for t in got])) for i in range(3)]
                cper = {q: float(np.mean([t[3][q] for t in got if q in t[3]]))
                        for q in CONF_LABELS if any(q in t[3] for t in got)}
                for k, v in zip(keys3, ceil):
                    base[k].append(v)
                base["pos"].append(cper)
                full = _pool()
                for d in day_x:
                    dr = _block_boot(day_x[d], day_b[d], rng)
                    if not dr:
                        continue
                    t = _enc_scores(dr, full)
                    if not np.isfinite(t[0]):
                        continue
                    for ki, (k, v) in enumerate(zip(keys3, t[:3])):
                        acc[d][k].append(v)
                        acc[d]["d" + k].append(v - ceil[ki])
                    acc[d]["pos"].append(t[3])
                    acc[d]["dpos"].append({q: t[3][q] - cper[q] for q in t[3] if q in cper})
        except Exception as ex:                                          # noqa: BLE001
            print(f"  !! enc CI {an} {align}/{variant}: {type(ex).__name__} {str(ex)[:80]}",
                  flush=True)
            continue

        # THE PLOTTED ESTIMATE SUPPLIES THE LOCATION, the bootstrap the width -- see `_anchor`.
        # `_enc_tables` is lru_cached and computes these with the very same `_enc_scores`, so the
        # band and the point on top of it cannot come from two different definitions.
        obs_all, _od = _enc_tables(align, variant, min_trials)
        orec = obs_all.get(an) or {}

        def _theta(col, key, orec=orec):
            t = orec.get(col)
            if t is None:
                return None
            if key in ("raw", "a", "gain"):
                return t[("raw", "a", "gain").index(key)]
            if key.startswith("d") and key[1:] in ("raw", "a", "gain"):
                b0 = orec.get("PRE")
                i = ("raw", "a", "gain").index(key[1:])
                return None if b0 is None else t[i] - b0[i]
            return None

        def _theta_pos(col, q, delta, orec=orec):
            t = orec.get(col)
            if t is None or q not in t[3]:
                return None
            if not delta:
                return t[3][q]
            b0 = orec.get("PRE")
            return None if b0 is None or q not in b0[3] else t[3][q] - b0[3][q]

        def _pack(store_, want, col, n_boot=n_boot):
            rec = {}
            for k in want:
                v = np.array([x for x in store_.get(k, []) if np.isfinite(x)])
                if len(v) >= n_boot // 4:
                    # `a` IS AN UNBOUNDED GAIN; the R^2 terms cannot exceed 1 and their intervals
                    # must not either, or a session at ceiling advertises an impossible upper limit.
                    hi = None if k in ("a", "da") else 1.0
                    rec[k] = _anchor(_pct3(v), _theta(col, k), hi=hi)
            for pk in ("pos", "dpos"):
                dicts = store_.get(pk) or []
                if not dicts:
                    continue
                got = {}
                for q in CONF_LABELS:
                    v = np.array([dd[q] for dd in dicts if q in dd and np.isfinite(dd[q])])
                    if len(v) < n_boot // 4:
                        continue
                    got[q] = _anchor(_pct3(v), _theta_pos(col, q, pk == "dpos"),
                                     hi=None if pk == "dpos" else 1.0)
                if got:
                    rec[pk] = got
            return rec

        rec = {}
        b = _pack(base, keys3, "PRE")
        if b:
            rec["PRE"] = b
        for d, s in acc.items():
            r = _pack(s, list(keys3) + ["d" + k for k in keys3], d)
            if r:
                rec[d] = r
        if len(rec) > 1:
            out[an] = rec
    return out, days
def fig_coding_retained(out_dir, meth="dom_orth"):
    """Two lines per panel, not six.

    THE SIX-LINE VERSION DID NOT WORK and the reason is structural, not cosmetic. It read the
    `poststroke_lick` class, so at an IMPAIRED position -- where the animal barely licks -- there is
    no cell to plot, and the positions the figure exists to describe were the ones missing from it.
    Twelve panels of six overlapping traces also had no legible message.

    So: the impaired positions are shown from MISS-WHILE-WORKING trials (the only trials they have)
    and the preserved positions from LICK trials, averaged within each group, SEM across positions.
    The two lines therefore come from different trial classes ON PURPOSE, which is stated on the
    figure -- an impaired position has no lick trials to average, and pretending otherwise is what
    produced the empty panels.
    """
    src = _fig_root() / "coding_direction.json"
    if not src.exists():
        return None
    data = json.loads(src.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(len(WINDOWS), len(ANIMALS), figsize=(10.4, 6.6),
                             sharey="row", squeeze=False)
    for ri, (disp, _align, wname) in enumerate(WINDOWS):
        for ci, an in enumerate(ANIMALS):
            ax = axes[ri][ci]
            ax.axhline(1.0, color="tab:green", ls=":", lw=1.4)
            ax.axhline(0.0, color="k", lw=0.8)
            res = (data.get(disp) or {}).get(an)
            imp = _impaired(an)
            if res and meth in res.get("methods", {}):
                by_cls = res["methods"][meth].get("cross_by_session", {})
                for group, cls, col, mk, lbl in (
                        (imp, "poststroke_miss_working", "#b2182b", "o",
                         "IMPAIRED positions (miss trials)"),
                        (set(POS) - imp, "poststroke_lick", "#2166ac", "s",
                         "positions still licked (lick trials)")):
                    cs = by_cls.get(cls, {})
                    xs, ys, es = [], [], []
                    for lab in sorted(cs):
                        vals = [(cs[lab].get(p) or {}).get(p) or {} for p in group]
                        vals = [c["mean"] for c in vals
                                if c.get("mean") is not None and (c.get("n") or 0) >= 10]
                        if not vals:
                            continue
                        xs.append(_day(an, lab.split("_")[-1]))
                        ys.append(float(np.mean(vals)))
                        es.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                                  if len(vals) > 1 else 0.0)
                    if xs:
                        order = np.argsort(xs)
                        ax.errorbar(np.array(xs)[order], np.array(ys)[order],
                                    yerr=np.array(es)[order], color=col, marker=mk, ms=5, lw=1.8,
                                    capsize=3, elinewidth=0.9,
                                    label=lbl if (ri == 0 and ci == 0) else None)
                        ax.set_xticks(sorted({int(v) for v in xs}))
            if ri == 0:
                ax.set_title(f"{an}\nimpaired: {', '.join(sorted(imp)) or 'none'}", fontsize=10,
                             fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{wname}\ncoding retained")
            if ri == len(WINDOWS) - 1:
                ax.set_xlabel("days from lesion")
            ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=11, loc="best")
    _suptitle(fig, "How much of each position's PRE-STROKE code survives, over days after the lesion.\n"
                 "1.0 (green) = that position's own pre-stroke signature; 0 = indistinguishable from "
                 "the other positions. Mean over positions in each group, error bars = SEM across "
                 "positions.\nThe two groups use DIFFERENT trial classes because an impaired "
                 "position has almost no lick trials to average.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1.0))   # top reserved by _suptitle
    _footer(fig, _cd_labels())
    p = Path(out_dir) / "grant_3a_coding_retained.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p
# ------------------------------------------------------------------ 3b. frozen vs within
def _binom_ci(p, n):
    """Half-width of a normal-approx binomial 95% CI, 0 when n is 0."""
    return 1.96 * float(np.sqrt(max(p * (1 - p), 0) / n)) if n else 0.0
#: (window, section_g condition key, display name, which ARM it comes from).
#:
#: POST-LICK COMES FROM THE LICK-ONLY ARM, and that is not a workaround -- it is the only arm in
#: which the condition is defined. The ALL-trials arm includes trials with no detected lick and a
#: lick-aligned window cannot be built for a trial with no lick, so `poststroke_section_g` skips it
#: there (`if align == "lick" and arm_all: continue`) and computes it for lick-only, where every
#: trial has a lick by construction. It has been computed all along, in all 24 session records, with
#: permutation nulls and the pre-stroke band -- I asserted otherwise on 2026-08-24 after checking
#: `arms["all"]` alone and generalising from one arm to the analysis.
#:
#: THE COST, which is why the row is drawn with its own chance line: the lick-only arm scores each
#: session on ITS OWN preserved positions, so the class count and therefore chance differ between
#: sessions -- 4-way at 0.25 on one day and 6-way at 0.167 on another. Accuracies in that row are
#: NOT comparable across sessions or with the two rows above it.
FROZEN_WINDOWS = (("ENL", "pre-cue", "ENL (pre-cue)", "all"),
                  ("cue", "post-cue", "post-cue", "all"),
                  ("lick", "post-lick", "post-lick  [lick-only arm]", "lickonly"))
def fig_frozen_vs_within(out_dir):
    sg = _fig_root() / "section_g.json"
    if not sg.exists():
        return None
    G = json.loads(sg.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(len(FROZEN_WINDOWS), len(ANIMALS), figsize=(10.4, 7.2),
                             sharey="row", squeeze=False)
    for ri, (_disp, gkey, wname, armkey) in enumerate(FROZEN_WINDOWS):
        for ci, an in enumerate(ANIMALS):
            ax = axes[ri][ci]
            fx, fy, fe, wx, wy, we, band = [], [], [], [], [], [], None
            chance_x, chance_y = [], []
            for sess in sorted(k for k in G if k.startswith(an)):
                mmdd = sess.split("_")[-1]
                if config.session_phase(an, mmdd) != "post":
                    continue
                day = _day(an, mmdd)
                arm = (G[sess].get("arms") or {}).get(armkey) or {}
                if arm.get("chance"):
                    chance_x.append(day); chance_y.append(arm["chance"])
                cell = arm.get(gkey) or {}
                n = cell.get("n") or 0
                if cell.get("accuracy") is not None:
                    fx.append(day); fy.append(cell["accuracy"])
                    fe.append(_binom_ci(cell["accuracy"], n))
                # WITHIN-SESSION LIVES IN section_g TOO, under "<cond> within-session" -- but as a
                # POOLED block carrying a per_session LIST across every session, not a scalar for
                # this one. Reading it as a scalar silently produced nothing; reading it from
                # poststroke_grid.json (the other obvious source) gives only day 1 and 2, which is
                # one or two points per animal and no trajectory at all.
                wblk = arm.get(f"{gkey} within-session") or {}
                if wblk and band is None:
                    band = wblk.get("within_pre_band")
                # EACH SESSION'S BLOCK CARRIES ONLY ITS OWN POST ROW -- the rest of `per_session` is
                # that animal's pre-stroke sessions, which is what the band is built from. Taking
                # the list from the first session and stopping (the obvious read) yields exactly one
                # green point per animal, which is what the first version of this figure showed.
                for row in wblk.get("per_session", []):
                    if row.get("post") and row.get("label") == sess \
                            and row.get("within_accuracy") is not None:
                        wx.append(day); wy.append(row["within_accuracy"])
                        we.append(_binom_ci(row["within_accuracy"], row.get("n") or n))
            if wx:
                order = np.argsort(wx)
                wx = list(np.array(wx)[order]); wy = list(np.array(wy)[order])
                we = list(np.array(we)[order])
            if band:
                ax.axhspan(band["min"], band["max"], color="tab:blue", alpha=0.15, zorder=1,
                           label="pre-stroke range" if (ri == 0 and ci == 0) else None)
            if fx:
                ax.errorbar(fx, fy, yerr=fe, color="tab:red", marker="o", ms=5, lw=1.6, capsize=3,
                            label="FROZEN pre-stroke decoder" if (ri == 0 and ci == 0) else None)
            if wx:
                ax.errorbar(wx, wy, yerr=we, color="tab:green", marker="s", ms=5, lw=1.6, capsize=3,
                            ls="--",
                            label="trained on that session" if (ri == 0 and ci == 0) else None)
            # CHANCE IS PER SESSION IN THE LICK-ONLY ARM. One flat 1/6 line would be wrong on
            # every 4-position session, and drawing it anyway is how a 4-way 0.5 gets read as
            # twice chance when it is exactly twice a DIFFERENT chance.
            if armkey == "lickonly" and chance_x:
                o = np.argsort(chance_x)
                ax.step(np.array(chance_x)[o], np.array(chance_y)[o], where="mid", color="k",
                        ls=":", lw=1.1)
            else:
                ax.axhline(1 / 6, color="k", ls=":", lw=1.0)
            ax.set_ylim(0, 1.02)
            if fx or wx:
                ax.set_xticks(sorted({int(v) for v in list(fx) + list(wx)}))
            if ri == 0:
                ax.set_title(an, fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{wname}\naccuracy")
            if ri == len(FROZEN_WINDOWS) - 1:
                ax.set_xlabel("days from lesion")
            ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=9.5, loc="lower left")
    _suptitle(fig, "Does the OLD code still read out, and is position information still there?\n"
                 "RED = frozen pre-stroke decoder.  GREEN = decoder trained on that session.  "
                 "Band = pre-stroke range.  Bars = binomial 95% CIs.\n"
                 "Top two rows: all trials, 6 positions, chance 1/6. Bottom row: LICK-ONLY arm "
                 "(a trial with no lick has no lick-aligned window), so chance is per session "
                 "(dotted step) and those panels are NOT comparable across sessions.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1.0))   # top reserved by _suptitle
    _footer(fig, _sg_labels())
    p = Path(out_dir) / "grant_3b_frozen_vs_within.png"
    _save(fig, p, dpi=200)
    plt.close(fig)
    return p

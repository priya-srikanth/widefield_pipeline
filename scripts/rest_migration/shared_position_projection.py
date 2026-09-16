"""`15s` — HOW MUCH OF A POSITION'S TASK MAP IS ALREADY PRESENT IN ITS OWN REST?

Scoped in `docs/STATUS_2026-09-13.md` ("SCOPED, NOT BUILT"), built 2026-09-16 after the docked
frozen decoder arm, which is the order that scope specifies: the frozen arm establishes whether the
pre-stroke rest code survives at all, and this measurement assumes a position on that.

THE QUANTITY, and why it is a PROJECTION rather than a difference map:

    shared_p = <rest_p - restw , trial_p - restw> / ||trial_p - restw||^2

per position, per window, per session. It is the fraction of that position's task-evoked map that
its own rest already contains -- a regression coefficient, so `shared = 1` would mean the task map
adds nothing the rest map did not already have, and `shared = 0` means they are orthogonal.

WHY NOT "REFERENCE EACH POSITION TO ITS OWN REST". That was the original proposal and it is a trap
the repo has already rejected twice. The algebra settles it:

    (trial_p - restw) - (trial_p - rest_p)  =  rest_p - restw

The difference between the two referencings does not involve the trials AT ALL -- it is exactly the
per-position deviation of rest from the neutral baseline, which `epoch_15x` already plots. So
"do both and compare the maps" adds nothing; that comparison IS 15x. And as a REFERENCE, per-position
rest subtracts the between-trial position signal from itself (the persistence trace, +0.0754 across
4/4 animals) and returns ~zero by construction. The per-position baseline is admissible as a
MEASUREMENT and never as the subtrahend -- which is what this file is.

THE FIVE DESIGN POINTS FROM THE SCOPE, none of them optional
------------------------------------------------------------
1. BOTH WINDOWS, because the prediction differs. If the pre-cue readout is PERSISTENCE, `shared`
   should be HIGH at pre-cue and LOWER at cue. If it is high at BOTH, the "pre-cue position
   information" claim is weaker than currently stated, and that matters more than a confirmation.
2. THE CIRCULARITY GUARD, which invalidates the whole thing if skipped. `rest_p` and `trial_p` are
   disjoint FRAMES but share the session's slow drift, and shared drift alone produces a positive
   projection. So `rest_p` is built from ODD position-blocks and `trial_p` from EVEN ones: a drift
   common to both can no longer manufacture the effect, because the two are drawn from interleaved
   and non-adjacent stretches of the session. The split is stated on the figure.
3. NULL = CIRCULAR SHIFT of the position labels over time-ordered blocks, the construction finding
   11 uses. It keeps block-time structure INSIDE the null. A label shuffle is not acceptable here
   for the same reason it was not there -- it destroys the block structure and gives an
   optimistically low null.
4. THE TRIAL MAP IS REFERENCED TO `restw`, NEVER to `rest_p`. Referencing it to its own rest would
   make the projection circular by construction: the quantity would appear on both sides.
5. THE NAME CANNOT IMPLY A REFERENCE VARIANT. `epoch_15s_shared_position_*`, never `_PERPOSref_` --
   a filename implying a fourth reference would invite exactly the per-position referencing the
   scope rejects.

WHAT IT CAN ANSWER that nothing else here can: whether the SUSTAINED component shrinks after the
lesion while the EVOKED one survives, or the reverse. That bears directly on the intention readout.

RUN:  python -m scripts.rest_migration.shared_position_projection [--align cue precue]
                                                                 [--perm 200] [--animals PS92 ...]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

MAP_SHAPE = (540, 640)


def _frame_samples(mc, fmdir, regime, pco):
    """DAQ sample index of every BLUE frame -- same mapping the rest arm uses."""
    if regime == "B":
        fm = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_frame_map.npz"))
        summ = sorted(glob.glob(f"{fmdir or mc}/*cleanpairs_summary.json"))
        if not fm or not summ:
            return None
        with open(summ[0]) as fh:
            off = int(json.load(fh)["chosen_exposure_offset"])
        z = np.load(fm[0])
        return pco[np.clip(z["original_frame_index_ch0"] + off, 0, len(pco) - 1)]
    return pco[np.arange(len(pco) // 2) * 2]


def _epoch_dir():
    """The shared `grant_figures/epoch` directory every other epoch figure writes to.

    NOT a local scratch path. These outputs were landing in `E:/cue_lick/rest_migration/`, which is
    this box's disk -- so every rest-arm result was invisible from the other machine and from the
    deck (Priya, 2026-09-16: "our new rest figures should join all our other figures"). Resolved
    through `PathResolver` exactly as `rest_position_vs_drift` does for `epoch_15x`, so it is
    correct on either box rather than correct on the one it was written on.
    """
    from wfield_local.paths import PathResolver
    return Path(PathResolver().root("labcams")) / "grant_figures" / "epoch"


def projection(rest_p, trial_p, restw):
    """`shared_p`: the fraction of `trial_p - restw` that `rest_p - restw` already accounts for.

    A REGRESSION COEFFICIENT, not a correlation, and the choice is deliberate. A correlation would
    be scale-free and would call a rest map that has the right SHAPE but a hundredth of the
    amplitude a perfect match; the question here is how much of the task map is already THERE, which
    is a question about magnitude as well as shape. The cosine is reported alongside so shape and
    magnitude can be told apart -- a high cosine with a low coefficient is "same pattern, much
    weaker", which is a different finding from "unrelated".
    """
    a = np.asarray(rest_p, float).ravel() - np.asarray(restw, float).ravel()
    b = np.asarray(trial_p, float).ravel() - np.asarray(restw, float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan"), float("nan")
    a, b = a[m], b[m]
    nb = float(b @ b)
    if nb <= 0:
        return float("nan"), float("nan")
    coef = float(a @ b) / nb
    na = float(np.sqrt(a @ a))
    cos = float(a @ b) / (na * np.sqrt(nb)) if na > 0 else float("nan")
    return coef, cos


def projection_components(rest_c, trial_c, restw_c, G):
    """`projection` done in COMPONENT space -- mathematically identical, ~1000x cheaper.

    Every map here is ``u @ c`` for a length-K component vector, so an inner product over pixels is
    a quadratic form in components:

        <u(a-w), u(b-w)>  =  (a-w)^T (u^T u) (b-w)

    with ``G = u^T u`` a KxK matrix computed ONCE per session. The null rebuilds a trial map for
    every position on every permutation; doing that in pixel space is a (345600, K) matmul each
    time and made a full two-window cohort run about four hours. This is the same number --
    `tests/test_shared_position_projection.py` asserts the two agree -- and it is what makes the
    permutation count affordable rather than something to economise on.
    """
    a = np.asarray(rest_c, float).ravel() - np.asarray(restw_c, float).ravel()
    b = np.asarray(trial_c, float).ravel() - np.asarray(restw_c, float).ravel()
    nb = float(b @ G @ b)
    if not np.isfinite(nb) or nb <= 0:
        return float("nan"), float("nan")
    num = float(a @ G @ b)
    na = float(a @ G @ a)
    cos = num / float(np.sqrt(na * nb)) if na > 0 else float("nan")
    return num / nb, cos


def _plot(rows, out, variant_name, aligns):
    """`epoch_15s_shared_position_*` -- one row per window, positions on x, epochs as series.

    THE NULL IS DRAWN, not summarised in a caption. `shared` is a regression coefficient with no
    natural zero-point for "unrelated" once drift and block structure are in play, so a bar without
    its circular-shift null cannot be read at all. The null band is the 5th-95th percentile of the
    per-cell nulls pooled within (window, epoch).

    PER-ANIMAL POINTS OVER THE BARS, for the reason this family has needed them before: a cohort
    mean in the rest arm has already hidden a reversal once (the withdrawn acute dip, where PS95
    went the other way). Four points per bar is not clutter here, it is the check.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import config as _cfg
    from wfield_local import epoch_figures as ef
    from wfield_local.grant_figures import CONF_LABELS

    colors = _cfg.animal_color()
    labels = [x.title() for x in ef.anatomical_labels(CONF_LABELS, short=False)]
    EPO = ["pre", "acute", "subacute", "chronic"]
    shades = {"pre": "0.25", "acute": "#b2182b", "subacute": "#ef8a62", "chronic": "#2166ac"}
    fig, axes = plt.subplots(len(aligns), 1, figsize=(13.0, 4.6 * len(aligns)), squeeze=False)
    x = np.arange(len(CONF_LABELS))
    w = 0.2
    for r, align in enumerate(aligns):
        ax = axes[r][0]
        sub = [q for q in rows if q["align"] == align]
        for k, e in enumerate(EPO):
            v = [q for q in sub if q["epoch"] == e]
            if not v:
                continue
            means, pts = [], []
            for q in CONF_LABELS:
                cells = [t for t in v if t["position"] == q and np.isfinite(t["shared"])]
                means.append(np.mean([t["shared"] for t in cells]) if cells else np.nan)
                per_an = {}
                for t in cells:
                    per_an.setdefault(t["animal"], []).append(t["shared"])
                pts.append({a_: float(np.mean(b)) for a_, b in per_an.items()})
            off = (k - 1.5) * w
            ax.bar(x + off, means, w, color=shades[e], label=e, alpha=0.85)
            for i, d in enumerate(pts):
                for an, val in d.items():
                    ax.scatter(x[i] + off, val, s=9, color=colors.get(an, "k"),
                               zorder=3, linewidths=0)
            nl = [t["null_mean"] for t in v if np.isfinite(t["null_mean"])]
            if nl:
                ax.axhspan(float(np.percentile(nl, 5)), float(np.percentile(nl, 95)),
                           color=shades[e], alpha=0.07, zorder=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("shared = <rest$_p$-restw, trial$_p$-restw> / ||trial$_p$-restw||$^2$")
        ax.set_title(f"{align} window", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        if r == 0:
            ax.legend(fontsize=8, frameon=False, ncol=4)
    fig.suptitle(
        "15s -- how much of each position's TASK map is already present in its own REST\n"
        f"rest variant {variant_name}; REST from ODD position-blocks, TRIALS from EVEN so shared "
        "drift cannot manufacture it; shaded band = circular-shift null (5-95 pct); "
        "dots = per-animal means", fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    p = out / f"epoch_15s_shared_position_{variant_name}.png"
    fig.savefig(p, dpi=150)
    fig.savefig(p.with_suffix(".svg"))
    plt.close(fig)
    return p


def block_parity(blocks):
    """ODD / EVEN masks over the session's blocks IN TIME ORDER.

    Parity of the block's ORDINAL POSITION in the session, not of its id: ids come from
    `block_ids` and are not guaranteed to be consecutive, so `id % 2` would split unevenly and, worse,
    could correlate with position if a position's blocks happened to land on one parity.
    """
    blocks = np.asarray(blocks)
    order = {b: i for i, b in enumerate(sorted(set(blocks.tolist())))}
    idx = np.array([order[b] for b in blocks])
    return (idx % 2 == 1), (idx % 2 == 0)


def _session_terms(session, align, post_s, variant, rng, perm, gate=True):
    """``{position: (rest_p, trial_p)}`` plus `restw`, with the odd/even circularity guard applied."""
    import h5py

    from wfield_local import daq_io, joint_basis
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.block_ids import block_ids, block_size_max_for
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES, _load_cue_events
    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import trial_features_cached
    from wfield_local.position_reference_maps import MIN_TRIALS_PER_CLASS, session_restw_svt
    from wfield_local.quiet_periods import quiet_dir

    code_of = {nm: int(c) for c, nm in POSITION_NAMES.items()}
    u, v = joint_basis._load_session(session["mc"])
    V = np.asarray(v)
    K = u.shape[1]

    # ---- the TRIAL side, with block ids -------------------------------------------------------
    X, y, g, _Xn, _yn, _r, _ie, _in = trial_features_cached(
        session, _args("locanmf", align, post_s), signal=V,
        feat_region=np.arange(V.shape[0]), signal_key=f"svt:rank{V.shape[0]}", with_indices=True)
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
    if not len(y):
        return None, "no trials"
    n_bins = X.shape[1] // K
    _odd_t, even_t = block_parity(g)

    # COMPONENT VECTORS, not pixel maps. `_cvec` is the length-K summary a map would be built from;
    # `u` is only needed through G = u^T u. Bins are averaged exactly as figure 14 does.
    G = np.asarray(u.T @ u, dtype=np.float64)

    def _cvec(feat):
        return np.asarray(feat, float).reshape(n_bins, K).mean(0)

    # ---- the REST side, per position, from the SAME session ------------------------------------
    qs = sorted(glob.glob(f"{quiet_dir(session['mc'])}/*quiet_sample.npy"))
    if not qs:
        return None, "no rest mask"
    rest = np.load(qs[0]).astype(bool)
    with h5py.File(session["h5"], "r") as f:
        dn = [x.decode() for x in f["digital/channel_names"][:]]
        packed = f["digital/packed_samples"][:, 0]
    pco = daq_io.rising_edges((packed >> dn.index("pco_exposure")) & 1)
    ts = daq_io.rising_edges((packed >> dn.index("trial_start")) & 1)
    cue = _load_cue_events(session["h5"])
    codes = np.asarray(classify_cues_with_backup(session, cue))
    cs = np.asarray(cue["cue_samples"], np.int64)
    fs_samp = _frame_samples(session["mc"], session.get("fmdir"), session.get("regime"), pco)
    if fs_samp is None:
        return None, "no frame map"
    blk_c = block_ids(np.asarray(codes), block_size_max_for(session))
    engaged = None
    if gate:
        from wfield_local.rest_engagement import engaged_by_cue
        engaged, gate_note = engaged_by_cue(session, cs, codes)
        if "UNGATED" in gate_note:
            print(f"  !! {session['label']}: {gate_note}", flush=True)
    odd_c, _even_c = block_parity(blk_c)

    T = V.shape[1]
    f_of = np.clip(fs_samp, 0, rest.shape[0] - 1)
    pad = np.concatenate([[0], rest.view(np.int8), [0]])
    dif = np.diff(pad)
    rest_frames = {}
    for aa, bb in zip(np.flatnonzero(dif > 0), np.flatnonzero(dif < 0)):
        prev = np.searchsorted(cs, aa, "right") - 1
        nxt = np.searchsorted(ts, bb, "left")
        if prev < 0 or nxt >= len(ts):
            continue
        nc = np.searchsorted(cs, ts[nxt], "left")
        if nc >= len(codes) or prev >= len(codes):
            continue
        if codes[prev] != codes[nc] or codes[prev] < 0:
            continue
        # THE ENGAGEMENT GATE (2026-09-16). Same omission as every other rest analysis: claimed in
        # prose, never applied. See `wfield_local/rest_engagement.py`.
        if engaged is not None and not (engaged[prev] and engaged[nc]):
            continue
        # THE GUARD: rest from ODD blocks only, trials from EVEN only.
        if not odd_c[prev]:
            continue
        fr = np.flatnonzero((f_of >= aa) & (f_of < bb))
        fr = fr[fr < T]
        if fr.size:
            rest_frames.setdefault(int(codes[prev]), []).append(fr)

    # docked=FALSE, matching `position_reference_maps` line 486 -- the production call. The
    # `restdock05` MASK is already docked, so `docked=True` would apply the term a second time and
    # 15s would be built on a different rest window from the 15r family it is compared with.
    restw, used_codes = session_restw_svt(session, V)
    if restw is None:
        return None, f"no restw ({len(used_codes)} positions)"
    restw_c = np.asarray(restw, float).ravel()[:K]

    out = {}
    for q in CONF_LABELS:
        c = code_of.get(q)
        if c is None or c not in rest_frames:
            continue
        sel = (y == c) & even_t
        if int(sel.sum()) < MIN_TRIALS_PER_CLASS:
            continue
        fr = np.concatenate(rest_frames[c])
        if fr.size < 20:
            continue
        rest_c = np.asarray(V[:, fr].mean(1), float)
        trial_c = _cvec(X[sel].mean(0))
        out[q] = (rest_c, trial_c, int(sel.sum()), int(fr.size))

    if not out:
        return None, "no position had both halves"

    # ---- the CIRCULAR-SHIFT NULL ---------------------------------------------------------------
    # Shift the position labels over TIME-ORDERED blocks, then rebuild the trial maps under the
    # shifted assignment and re-project each position's REAL rest map onto them. Block structure and
    # drift stay inside the null; only the position correspondence is broken.
    nulls = {q: [] for q in out}
    ublk = sorted(set(g.tolist()))
    lab_of = {b: int(y[np.flatnonzero(g == b)[0]]) for b in ublk}
    for _ in range(int(perm)):
        k = int(rng.integers(1, max(2, len(ublk))))
        rolled = np.roll([lab_of[b] for b in ublk], k)
        m = dict(zip(ublk, rolled))
        y_s = np.array([m[b] for b in g])
        for q, (rest_c, _tc, _nt, _nf) in out.items():
            c = code_of[q]
            sel = (y_s == c) & even_t
            if int(sel.sum()) < MIN_TRIALS_PER_CLASS:
                continue
            co, _cs2 = projection_components(rest_c, _cvec(X[sel].mean(0)), restw_c, G)
            if np.isfinite(co):
                nulls[q].append(co)
    return (out, restw_c, G, nulls), None


def _qv():
    from wfield_local.quiet_periods import quiet_variant
    return quiet_variant()


def main() -> int:
    from wfield_local import config, epochs
    from wfield_local.grant_figures import CONF_LABELS
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.quiet_periods import quiet_variant

    ap = argparse.ArgumentParser()
    ap.add_argument("--align", nargs="+", default=["cue", "precue"])
    ap.add_argument("--variant", default="working")
    ap.add_argument("--post-s", type=float, default=2.0)
    ap.add_argument("--perm", type=int, default=50)
    ap.add_argument("--animals", nargs="*", default=None)
    ap.add_argument("--no-engagement-gate", action="store_true",
                    help="keep the pre-2026-09-16 ungated behaviour, for measuring "
                         "the size of the correction only")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure from the existing CSV without re-measuring")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to the shared "
                         "grant_figures/epoch where every other epoch figure lives")
    a = ap.parse_args()
    a.out = a.out or _epoch_dir()

    # REDRAW WITHOUT RE-MEASURING. The measurement is ~35 min at 200 permutations and the figure is
    # seconds; forcing a full re-run to change an axis label is how a caption fix comes to be worth
    # postponing. The CSV is the measurement of record, so replotting from it cannot disagree with
    # the numbers -- it reads the same file a reader would check.
    if a.replot:
        src = a.out / f"epoch_15s_shared_position_{(_qv() or 'retired')}.csv"
        if not src.exists():
            print(f"!! --replot: {src} does not exist; run the measurement first")
            return 1
        rows = []
        with open(src, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for k in ("shared", "cosine", "null_mean", "p"):
                    r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
                rows.append(r)
        aligns = [x for x in a.align if any(r["align"] == x for r in rows)]
        fp = _plot(rows, a.out, _qv() or "retired", aligns)
        print(f"replotted {len(rows)} cells from {src}\nwrote {fp}")
        return 0

    variant_name = quiet_variant() or "retired"
    t0 = time.time()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [s for s in SESSIONS if s["label"] in want and s.get("h5")
            and (not a.animals or s["label"].split("_")[0] in a.animals)]
    print(f"15s SHARED-POSITION PROJECTION -- rest variant {variant_name}, "
          f"{len(todo)} sessions, windows {a.align}\n"
          f"GUARD: rest from ODD position-blocks, trials from EVEN -- shared drift cannot "
          f"manufacture the effect\n")

    rows, skipped = [], []
    for align in a.align:
        rng = np.random.default_rng(0)
        for s in todo:
            ep = epochs.epoch_of(s["label"])
            if ep is None:
                skipped.append(f"{s['label']}: no epoch")
                continue
            try:
                got, why = _session_terms(s, align, a.post_s, a.variant, rng, a.perm,
                                          gate=not a.no_engagement_gate)
            except Exception as ex:                                    # noqa: BLE001
                skipped.append(f"{s['label']} [{align}]: {type(ex).__name__} {str(ex)[:60]}")
                continue
            if got is None:
                skipped.append(f"{s['label']} [{align}]: {why}")
                continue
            terms, restw_c, G, nulls = got
            shown = {}
            for q, (rest_c, trial_c, n_tr, n_fr) in terms.items():
                coef, cos = projection_components(rest_c, trial_c, restw_c, G)
                shown[q] = coef
                nl = nulls.get(q, [])
                rows.append({"align": align, "animal": s["label"].split("_")[0],
                             "label": s["label"], "epoch": ep, "position": q,
                             "shared": coef, "cosine": cos,
                             "null_mean": float(np.mean(nl)) if nl else float("nan"),
                             "n_perm": len(nl),
                             "p": ((1 + sum(1 for x in nl if x >= coef)) / (1 + len(nl)))
                                  if nl else float("nan"),
                             "n_trials_even": n_tr, "n_rest_frames_odd": n_fr})
            print(f"  .. {s['label']} [{ep}/{align}] "
                  + " ".join(f"{q}={c:.2f}" for q, c in shown.items()), flush=True)

    if not rows:
        print("NOTHING MEASURED -- a failed run, not a null result.")
        for x in skipped[:12]:
            print("   skipped:", x)
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    cp = a.out / f"epoch_15s_shared_position_{variant_name}.csv"
    with open(cp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'=' * 92}\nSHARED-POSITION PROJECTION -- fraction of the task map already present "
          f"in that position's REST\n{'=' * 92}")
    for align in a.align:
        sub = [r for r in rows if r["align"] == align]
        if not sub:
            continue
        print(f"\n--- {align} ---")
        print(f"{'epoch':<10}" + "".join(f"{q:>13}" for q in CONF_LABELS) + f"{'null':>9}")
        for e in ("pre", "acute", "subacute", "chronic"):
            v = [r for r in sub if r["epoch"] == e]
            if not v:
                continue
            line = f"{e:<10}"
            for q in CONF_LABELS:
                vals = [r["shared"] for r in v if r["position"] == q and np.isfinite(r["shared"])]
                line += f"{np.mean(vals):>13.3f}" if vals else f"{'--':>13}"
            nl = [r["null_mean"] for r in v if np.isfinite(r["null_mean"])]
            line += f"{np.mean(nl):>9.3f}" if nl else f"{'--':>9}"
            print(line)

    print("\nshared = <rest_p - restw, trial_p - restw> / ||trial_p - restw||^2, a REGRESSION")
    print("coefficient: 1 = the task map adds nothing its own rest did not already have, 0 =")
    print("orthogonal. REST comes from ODD position-blocks and TRIALS from EVEN, so a drift common")
    print("to both cannot produce a positive value. Null = circular shift of the block->position")
    print("map, which keeps block-time structure inside the null.")
    fp = _plot(rows, a.out, variant_name, a.align)
    print(f"\nwrote {cp}\nwrote {fp}")
    print(f"\n{len(rows)} cells; {len(skipped)} skipped")
    for x in skipped[:10]:
        print("   skipped:", x)
    print(f"[done in {time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

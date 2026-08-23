"""Pre- vs post-stroke activity maps on ONE COMMON COLOUR SCALE.

Priya, 2026-08-19: the preprocessing decks show much larger amplitude bars post-stroke, and asked
whether any current figure bears that out clearly. None does, and one actively hides it.

WHY THE EXISTING MAPS CANNOT SHOW IT. `plot_spout_trial_averages` sets its colour limit from a
percentile of THAT SESSION's own maps, so every session is renormalised to fill the same colour range.
A session whose responses are three times larger looks identical; only the number on the colourbar
changes. That is the right default for reading one session's spatial pattern and exactly wrong for
comparing amplitude across sessions -- the observation had to be made by reading colourbar values,
which is the figure failing at its job.

Here every panel shares one symmetric vmin/vmax computed across ALL panels in the figure, so a 2-3x
amplitude difference appears as a 2-3x difference in colour saturation.

THE AMPLITUDE RISE IS NOT A dF/F DENOMINATOR ARTEFACT, which was the obvious worry: baseline F is
unchanged post-stroke (PS94 pre/post ratio 1.01, PS95 1.02, PS92 0.99), so the rise is in the
numerator. Verified before building this, because a figure that dramatises an artefact is worse than
no figure.

Maps are reconstructed on the ATLAS grid (U_atlas @ window-mean SVT), so pixels are comparable across
sessions and animals. The window is the same one the decoders use.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from wfield_local import config                                      # noqa: E402
from wfield_local.paths import PathResolver                          # noqa: E402
from wfield_local.plot_lick_aligned_averages import (DISPLAY_ORDER,   # noqa: E402
                                                     POSITION_NAMES)

MIN_TRIALS = 8


def _position_maps(s, align="cue", post_s=2.0):
    """{position -> (H, W) mean window map} on the atlas grid, plus the trial count per position."""
    from wfield_local.behavior_position import classify_cues_with_backup
    from wfield_local.locanmf_crossanimal_dff import _frames
    from wfield_local.locanmf_position_decoder import _load_cue_events, _load_daq_events

    ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
    if not ad:
        return None, None
    U = np.load(f"{ad[0]}/U_atlas.npy")                      # (H, W, k)
    SVT = np.load(config.svtcorr_path(s["mc"]))              # (k, T)
    cue = _load_cue_events(s["h5"])
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, _csmp = _frames(s, cue, lk)
    codes = np.asarray(classify_cues_with_backup(s, cue))
    if align == "cue":
        ref = cue_f
    else:
        # lick_f is indexed by LICK EVENT, not by trial -- indexing it with a trial number walked off
        # the end ("index 535 out of bounds for axis 0 with size 535"). The lick-aligned reference is
        # the FIRST lick of each trial, so it has to be built per trial.
        cs = np.asarray(cue["cue_samples"], float)
        ls = np.asarray(lk["lick_samples"], float)
        ref = np.full(len(cue_f), -1, dtype=int)
        for k, c in enumerate(cs):
            m = (ls >= c) & (ls < c + 3.5 * 5000.0)
            if not m.any():
                continue
            first = ls[m][0]
            j = int(np.argmin(np.abs(np.asarray(_csmp, float) - first)))
            ref[k] = j
    n = int(round(post_s * 30.0))                            # ~30 Hz per channel
    maps, counts = {}, {}
    for c in DISPLAY_ORDER:
        sel = [k for k in range(len(ref))
               if codes[k] == c and ref[k] >= 0 and ref[k] + n < SVT.shape[1]]
        counts[POSITION_NAMES[c]] = len(sel)
        if len(sel) < MIN_TRIALS:
            continue
        # average SVT over each trial's window FIRST -- a k-vector per position -- then project once
        v = np.mean([SVT[:, f:f + n].mean(axis=1) for f in (ref[k] for k in sel)], axis=0)
        maps[POSITION_NAMES[c]] = (U @ v).astype(np.float32)
    return maps, counts


def animal_maps(animal, align="cue", max_post=None, post_s=2.0):
    """(pre-stroke mean map per position, {post label -> maps}, counts) for one animal.

    ``post_s`` is the WINDOW LENGTH, and it was previously fixed at 2 s because `_position_maps`
    accepted the argument and no caller passed it. The preprocessing decks show the post-lick maps
    at **150 ms**, which is the window a reader actually compares across days -- and those had no
    common-scale version at all, so every cross-day comparison of them was being made between two
    independently-chosen colour limits (Priya, 2026-08-22: "scale is also wildly different"). PS94
    is +-0.02425 on 8/14 and +-0.08854 on 8/17, a factor of 3.65.
    """
    pre_dates = set(config.curated_dates())
    pre_maps, post = {}, {}
    for s in config.load_sessions():
        if s["label"][:4] != animal:
            continue
        date = s["label"].split("_")[-1]
        phase = config.session_phase(animal, date)
        if phase == "pre" and date in pre_dates:
            m, _c = _position_maps(s, align, post_s)
            if m:
                for p, arr in m.items():
                    pre_maps.setdefault(p, []).append(arr)
        elif phase == "post":
            m, c = _position_maps(s, align, post_s)
            if m:
                post[s["label"]] = (m, c)
    pre_mean = {p: np.mean(v, axis=0) for p, v in pre_maps.items() if len(v) >= 3}
    # EVERY post-stroke session by default. This was capped at the first TWO, which silently dropped
    # 0819 onward from every figure -- exactly the "no silent caps" failure: the figure looked
    # complete, and a reader comparing it against a per-session deck would find days that simply were
    # not there (Priya, 2026-08-23: "ensure the shared axis SVD maps generate for all post-stroke
    # dates"). If a cap is ever passed again, say what it dropped.
    ordered = sorted(post.items())
    if max_post is not None and len(ordered) > max_post:
        dropped = [lab for lab, _ in ordered[max_post:]]
        print(f"[fixed_scale_maps] {animal}: --max-post {max_post} DROPS {len(dropped)} "
              f"post-stroke session(s): {', '.join(dropped)}", flush=True)
        ordered = ordered[:max_post]
    return pre_mean, dict(ordered)


#: post-stroke sessions per FIGURE. Rows are sessions, so with every post-stroke day shown the
#: figure grows without bound -- at 18 it is 19 rows on one slide and each map is unreadable, the
#: same failure as the G8f strip. Chunking keeps a map the same size however long the cohort runs.
MAX_POST_PER_FIG = 4


def plot(animal, pre_mean, post, out_dir, align, post_s):
    """Chunked across figures, with the PRE-STROKE row repeated at the top of every part.

    The pre-stroke row is the reference the whole figure exists to compare against, so a part
    without it is unreadable on its own (Priya, 2026-08-23). It costs one row per part and makes
    each slide self-contained.

    THE COLOUR LIMIT IS COMPUTED ACROSS EVERY PART, not per part. Re-deriving it per chunk would
    give each slide its own scale, which is precisely the auto-scaling defect this whole module
    exists to fix -- and it would be invisible, because each part would look internally consistent.
    """
    if not pre_mean or not post:
        return None
    POS = [p for p in (POSITION_NAMES[c] for c in DISPLAY_ORDER) if p in pre_mean]
    pre_row = ("PRE-stroke mean", pre_mean, None)
    post_rows = [(lab, m, c) for lab, (m, c) in post.items()]
    # ONE colour limit for every panel of every PART -- the entire point of the figure
    allv = np.concatenate([np.asarray(m[p]).ravel()
                           for _t, m, _c in [pre_row] + post_rows for p in POS if p in m])
    lim = float(np.nanpercentile(np.abs(allv), 99.5))
    chunks = [post_rows[i:i + MAX_POST_PER_FIG]
              for i in range(0, len(post_rows), MAX_POST_PER_FIG)] or [[]]
    written = []
    for ci, chunk in enumerate(chunks, 1):
        written.append(_plot_part(animal, [pre_row] + chunk, POS, lim, out_dir, align, post_s,
                                  ci, len(chunks)))
    # a stale part from a run when the cohort was larger would silently show old sessions
    tag = align if abs(post_s - 2.0) < 1e-9 else f"{align}{round(post_s * 1000)}ms"
    for extra in sorted(Path(out_dir).glob(f"fixed_scale_maps_{animal}_{tag}__p*.png")):
        if int(extra.stem.rsplit("__p", 1)[1]) > len(chunks):
            print(f"[fixed_scale_maps] removing stale {extra.name}", flush=True)
            extra.unlink()
    return written[0]


def _plot_part(animal, cols, POS, lim, out_dir, align, post_s, part, n_parts):
    """One figure: the pre-stroke row plus up to MAX_POST_PER_FIG post-stroke sessions."""
    # SESSIONS AS ROWS, POSITIONS AS COLUMNS. The transpose of this was 6 rows x 2-3 columns, i.e.
    # 17-32 inches tall against 7-10 wide. Placed on a 13.3 x 7.5 in slide at full width it ran far
    # off the bottom and the lower positions were never visible at all; scaled to fit, it was
    # unreadable. This orientation is also the one every other figure in the deck uses, so positions
    # line up across slides.
    fig, axes = plt.subplots(len(cols), len(POS), figsize=(2.7 * len(POS), 2.5 * len(cols)),
                             squeeze=False)
    for r, (tit, m, cnt) in enumerate(cols):
        for k, p_ in enumerate(POS):
            ax = axes[r][k]
            ax.set_axis_off()
            if r == 0:
                ax.set_title(p_, fontsize=9)
            if k == 0:
                ax.text(-0.07, 0.5, tit, transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=8)
            if p_ not in m:
                ax.text(0.5, 0.5, "not attempted", ha="center", va="center", fontsize=8,
                        color="firebrick", transform=ax.transAxes)
                continue
            im = ax.imshow(m[p_], cmap="RdBu_r", vmin=-lim, vmax=lim)
            peak = float(np.nanmax(np.abs(m[p_])))
            # n ON EVERY PANEL (Priya, 2026-08-19). PS95 far_center on 8/17 is a mean over TEN
            # trials sitting beside panels averaging 110+, and on a common colour scale an
            # undersampled mean shows as saturated blobs that read as a large effect. Red when the
            # count is within 3x of the inclusion floor, so the reader sees it without doing sums.
            lab = f"peak {peak:.3f}"
            low = False
            if cnt:
                lab += f"  n={cnt[p_]}"
                low = cnt[p_] < 3 * MIN_TRIALS
            ax.text(0.02, 0.02, lab, transform=ax.transAxes, fontsize=6.5,
                    color=("firebrick" if low else "k"),
                    fontweight=("bold" if low else "normal"),
                    bbox=dict(fc="white", alpha=0.65, lw=0))
    # DEDICATED AXES on the right. `ax=axes` shrinks every panel to make room and still let the
    # bar overlap them once the grid is wide (Priya, 2026-08-20, slides 181-188).
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.68])
    fig.colorbar(im, cax=cax)
    fig.suptitle(
        f"{animal} - {align}-aligned activity maps, {round(post_s * 1000)} ms window, on ONE COMMON "
        + (f"[part {part} of {n_parts}, PRE row repeated; the colour limit is shared "
           f"across ALL parts] " if n_parts > 1 else "") + 
        f"COLOUR SCALE (+-{lim:.3f}).\n"
        "The per-session maps in the preprocessing deck are auto-scaled, so every session fills the "
        "same colour range and an amplitude difference is invisible -- only the colourbar NUMBER "
        "changes. Here all panels share one limit, so a 2-3x larger response looks 2-3x more "
        "saturated. Baseline F is unchanged post-stroke (ratio ~1.0), so this is a numerator effect, "
        "not a dF/F denominator artefact. n IS ON EVERY PANEL, in RED where it is under 3x "
        "the inclusion floor -- a mean over ten trials beside means over a hundred looks "
        "like a large effect and is mostly noise.", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 0.90, 0.93))
    # the window goes in the FILENAME whenever it is not the historical 2 s, so a 150 ms
    # figure can never overwrite the 2 s one or be mistaken for it
    _tag = align if abs(post_s - 2.0) < 1e-9 else f"{align}{round(post_s * 1000)}ms"
    # part 1 keeps the historical filename so no existing deck reference dangles
    q = Path(out_dir) / (f"fixed_scale_maps_{animal}_{_tag}.png" if part == 1
                         else f"fixed_scale_maps_{animal}_{_tag}__p{part}.png")
    fig.savefig(q, dpi=140)
    plt.close(fig)
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--align", nargs="+", default=["cue", "lick"])
    ap.add_argument("--post-s", type=float, default=2.0,
                    help="window length in seconds (default 2.0; the preprocessing "
                         "decks' post-lick maps are 0.15)")
    ap.add_argument("--max-post", type=int, default=None,
                    help="cap the number of post-stroke sessions shown (default: ALL). A cap is "
                         "reported, never silent.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    animals = args.animals or sorted({l.split("_")[0] for l in config.phase_labels("post")})
    for align in args.align:
        for a in animals:
            print(f"[fixed_scale_maps] {a} {align} ...", flush=True)
            try:
                pre, post = animal_maps(a, align, max_post=args.max_post, post_s=args.post_s)
            except Exception as ex:                                  # noqa: BLE001
                print(f"  {a}: skip ({str(ex)[:70]})", flush=True)
                continue
            if not pre or not post:
                print(f"  {a}: {len(pre)} pre positions, {len(post)} post sessions -> skip")
                continue
            p = plot(a, pre, post, out, align, post_s=args.post_s)
            print(f"  wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

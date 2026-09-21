"""GRANT FIGURES — CONFUSION

Confusion matrices: pre-stroke, pre-vs-post, per session, and the delta.

Split out of `grant_figures` on 2026-09-21, one module per figure family. THE GROUPING IS FROM THE
CALL GRAPH, not from the names: every function here is reached from this family's entry points and
from no other family's. Anything shared with a sibling lives in `grant_kit` -- which is why this
imports from there and never from `grant_figures`, a direction that would be a cycle.

Entry points, registered in `grant_figures.JOBS`:
  - `fig_confusion_prestroke`
  - `fig_confusion_pre_post`
  - `fig_confusion_pre_post_working`
  - `fig_confusion_per_session`
  - `fig_confusion_delta`
"""
from __future__ import annotations

import json
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


def fig_confusion_prestroke(out_dir):
    """4: mean PRE-STROKE leave-one-session-out confusion, 2x2 animals, one file per window.

    Counts are SUMMED over the held-out pre-stroke sessions and then row-normalised, so each row is
    P(predicted | true) over the whole baseline -- not the mean of per-session rates, which would
    weight a 200-trial session the same as a 500-trial one.
    """
    made = []
    for _disp, align, wname in _windows():
        f = _fig_root() / f"joint_xsession_decoder_{align}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.0), squeeze=False)
        drew = False
        for k, an in enumerate(ANIMALS):
            ax = axes[k // 2][k % 2]
            r = d.get(an)
            pre = {lab for lab in config.phase_labels("pre") if lab.startswith(an)}
            mats = [np.array(m, float) for lab, m in ((r or {}).get("confusion") or {}).items()
                    if lab in pre]
            if not mats:
                ax.axis("off")
                continue
            C = np.sum(mats, axis=0)
            row = C.sum(1, keepdims=True)
            M = np.divide(C, row, out=np.zeros_like(C), where=row > 0)
            acc = float(np.trace(C) / C.sum()) if C.sum() else float("nan")
            im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
            for i in range(len(CONF_LABELS)):
                for j in range(len(CONF_LABELS)):
                    if M[i, j] >= 0.01:
                        _txt(ax, j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                                color="white" if M[i, j] < 0.6 else "black")
            ax.set_xticks(range(len(CONF_LABELS)))
            ax.set_xticklabels(_short(CONF_LABELS), rotation=90, ha="center", fontsize=10)
            ax.set_yticks(range(len(CONF_LABELS)))
            ax.set_yticklabels(_short(CONF_LABELS), fontsize=10)
            ax.set_title(f"{an} — {acc:.2f} correct ({len(mats)} held-out sessions)",
                         fontsize=10, fontweight="bold")
            # X-LABEL ON THE BOTTOM ROW ONLY -- on the top row it lands on the row below's title.
            if k // 2 == 1:
                ax.set_xlabel("predicted")
            if k % 2 == 0:
                ax.set_ylabel("true")
            drew = True
        if not drew:
            plt.close(fig)
            continue
        fig.colorbar(im, ax=axes, fraction=0.035, pad=0.09, label="P(predicted | true)")
        _suptitle(fig, f"Pre-stroke cross-session decoding — {wname} window\n"
                     "Frozen leave-one-session-out in the shared LocaNMF basis: every trial scored "
                     "by a decoder that never saw its session.\n"
                     "Counts summed over held-out sessions, then row-normalised. Chance = 0.17.",
                     fontsize=10)
        p = Path(out_dir) / f"grant_4_confusion_prestroke_{align}.png"
        _save(fig, p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made[0] if len(made) == 1 else (made or None)
def fig_confusion_pre_post(out_dir):
    """5: pre-stroke lick / pre-stroke NO-LICK control / post-stroke, one row per animal.

    THE MIDDLE PANEL IS WHY THIS IS THREE PANELS AND NOT TWO. Post-stroke the impaired positions are
    almost entirely no-lick trials, so an honest pre-vs-post pair on the ALL-trials arm compares
    pre-stroke LICK rows against post-stroke NON-LICK rows and confounds the lesion with the absence
    of a movement. `pre_nolick` is the matched control that already exists in `section_g`:
    PRE-stroke no-lick trials scored by a decoder trained on the OTHER pre-stroke sessions' engaged
    trials, so it differs from the post panel in PHASE ALONE. Read left-to-right: what the code
    looks like normally, what it looks like without a lick but without a lesion, and what it looks
    like after the lesion.

    Counts are reconstructed from the stored row-normalised matrices (matrix * n_per_true_position),
    summed across post-stroke sessions, and re-normalised. The `pre` and `pre_nolick` panels are the
    POOLED pre-stroke reference and are byte-identical in every session record -- summing them across
    sessions would multiply the same matrix by the session count and change nothing except to imply
    a sample size that does not exist.
    """
    sg = _fig_root() / "section_g.json"
    if not sg.exists():
        return None
    G = json.loads(sg.read_text(encoding="utf-8"))
    # TWO LINES, because width is the scarce dimension. A one-line title plus the accuracy is wider
    # than a third of a 9in figure at this font, and neighbouring titles collided. Height is free:
    # the header reservation now measures panel titles rather than assuming their size.
    PANELS = (("pre", "PRE-stroke\nLICK trials"),
              ("pre_nolick", "PRE-stroke, NO-LICK\n(matched control)"),
              ("post", "POST-stroke\nALL trials"))
    made = []
    for gkey, wname in (("pre-cue", "ENL (pre-cue)"), ("post-cue", "post-cue")):
        fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(9.0, 12.4), squeeze=False,
                                 gridspec_kw={"hspace": 0.42})
        drew = False
        for ri, an in enumerate(ANIMALS):
            sessions = sorted(k for k in G if k.startswith(an)
                              and config.session_phase(an, k.split("_")[-1]) == "post")
            if not sessions:
                for ci in range(3):
                    axes[ri][ci].axis("off")
                continue
            blocks = [((G[s].get("arms") or {}).get("all") or {}).get("confusion", {}).get(gkey)
                      for s in sessions]
            blocks = [b for b in blocks if b]
            if not blocks:
                for ci in range(3):
                    axes[ri][ci].axis("off")
                continue
            for ci, (key, ptitle) in enumerate(PANELS):
                ax = axes[ri][ci]
                use = blocks if key == "post" else blocks[:1]
                C = None
                for b in use:
                    d = b.get(key)
                    if not d:
                        continue
                    M = np.array(d["matrix"], float)
                    n = np.array(d["n_per_true_position"], float)
                    C = (np.nan_to_num(M) * n[:, None]) if C is None \
                        else C + np.nan_to_num(M) * n[:, None]
                if C is None:
                    ax.axis("off")
                    continue
                row = C.sum(1, keepdims=True)
                P = np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)
                acc = float(np.nansum(np.diag(C)) / C.sum()) if C.sum() else float("nan")
                im = ax.imshow(np.ma.masked_invalid(P), vmin=0, vmax=1, cmap="magma")
                for i in range(len(CONF_LABELS)):
                    if row[i, 0] == 0:
                        ax.text(2.5, i, "no trials", ha="center", va="center", fontsize=8,
                                color="firebrick", fontweight="bold")
                        continue
                    for j in range(len(CONF_LABELS)):
                        if P[i, j] >= 0.02:
                            _txt(ax, j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                                    color="white" if P[i, j] < 0.6 else "black")
                ax.set_xticks(range(len(CONF_LABELS)))
                ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                   rotation=90, ha="center", fontsize=9.5)
                ax.set_yticks(range(len(CONF_LABELS)))
                ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9.5)
                # THE ANIMAL MOVES TO THE Y LABEL, where every other figure in this module puts it.
                # In the title it competed for the panel's WIDTH, which is the scarce dimension --
                # the first column's title then reached its neighbour's.
                if ci == 0:
                    ax.set_ylabel(an, fontsize=11, fontweight="bold")
                ax.set_title(f"{ptitle}  ({acc:.2f})", fontsize=10,
                             fontweight="bold" if ci == 0 else "normal")
                drew = True
        if not drew:
            plt.close(fig)
            continue
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.07, label="P(predicted | true)")
        _suptitle(fig, f"The frozen pre-stroke decoder before and after the lesion — {wname} window\n"
                     "Rows = TRUE spout position, columns = predicted. MIDDLE PANEL IS THE MATCHED "
                     "CONTROL: pre-stroke NO-LICK trials scored by a decoder trained on the other "
                     "pre-stroke sessions,\nso it differs from the post panel in PHASE alone rather "
                     "than in phase and the absence of a movement together. "
                     "Post-stroke sessions pooled. Chance = 0.17.", fontsize=9.5)
        _footer(fig, _sg_labels())
        p = Path(out_dir) / f"grant_5_confusion_pre_post_{gkey.replace('-', '')}.png"
        _save(fig, p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made[0] if len(made) == 1 else (made or None)
class _Stored(Exception):
    """Raised to skip the LocaNMF recompute when the stored per-class confusions covered it."""
def fig_confusion_pre_post_working(out_dir):
    """5b: figure 5 with the post-stroke TERMINAL QUIT PERIOD removed.

    Priya, 2026-08-25: the same comparison on engaged post-stroke trials, without the "stopped"
    ones. Recomputed rather than filtered, because `section_g.json` stores confusions already summed
    over trials and a summed matrix cannot be un-summed.

    WHICH TRIALS. Post-stroke = lick trials PLUS miss-while-working, i.e. everything except the
    terminal non-recovering collapse. That is the `poststroke_all_working` population the coding
    directions already use. Dropping the no-lick trials entirely would be a different figure -- it
    would empty the impaired rows, which is the whole reason the all-trials arm exists.

    THIS IS NOT THE GATE `POSTSTROKE_ENGAGEMENT_FILTERING = False` FORBIDS, and the distinction
    matters. That flag rejects the ROLLING reference-rate gate, on the ground that a local dip is
    indistinguishable from a run of motor failures, so splitting trials on it would label the effect
    as the confound. `engagement_gate` requires a NON-RECOVERING FINAL collapse -- PS94_0817's rate
    dips at trial ~420 and is back near 0.95 by 480, and that session is correctly NOT called
    disengaged. Removing a terminal quit period is a much weaker claim than adjudicating individual
    trials, and it is the same construct the miss-vs-stopped split rests on throughout.

    STILL NOT VALIDATED, and the figure says so: nothing in the spout data proves the terminal run is
    satiety rather than a late motor collapse. It is reported BESIDE figure 5, never instead of it.
    """
    from wfield_local.locanmf_frozen_decoder import _pipe

    PANELS = (("pre", "PRE-stroke\nLICK trials"),
              ("post_all", "POST-stroke\nALL trials"),
              ("post_working", "POST-stroke\nquit period REMOVED"))

    def _from_json(an, disp):
        """The three panels as SUMS of the stored per-class confusions, or None to recompute.

        Added 2026-08-26. This figure used to redo the whole LocaNMF pooling -- >10 min of network
        reads -- because `section_g.json` stores a confusion already summed over trials and "a summed
        matrix cannot be un-summed". `coding_direction.json` now stores one matrix PER CLASS, so the
        populations this figure needs are additions:

            post_working = poststroke_lick + poststroke_miss_working
            post_all     = post_working    + poststroke_stopped

        Returning None (missing file, missing block, missing class) falls through to the recompute
        path, so this is a shortcut and never a new source of truth.
        """
        f = _fig_root() / "coding_direction.json"
        if not f.exists():
            return None
        try:
            rec = ((json.loads(f.read_text(encoding="utf-8")).get(disp) or {}).get(an) or {})
            c = rec.get("confusions")
            if not c or c.get("prestroke_lick") is None:
                return None
            lick, work, stop = (c.get("poststroke_lick"), c.get("poststroke_miss_working"),
                                c.get("poststroke_stopped"))
            if lick is None or work is None:
                return None
            L, W = np.array(lick, float), np.array(work, float)
            S = np.zeros_like(L) if stop is None else np.array(stop, float)
            return {"pre": np.array(c["prestroke_lick"], float),
                    "post_working": L + W, "post_all": L + W + S}
        except Exception:                                             # noqa: BLE001
            return None
    made = []
    for _disp, align, wname in (("ENL", "precue", "ENL (pre-cue)"), ("cue", "cue", "post-cue")):
        fig, axes = plt.subplots(len(ANIMALS), 3, figsize=(9.0, 12.4), squeeze=False,
                                 gridspec_kw={"hspace": 0.42})
        drew = False
        stored = []                # animals served from coding_direction.json rather than recomputed
        for ri, an in enumerate(ANIMALS):
            # STORED CLASSES FIRST; the LocaNMF recompute is the fallback, not the default.
            mats = _from_json(an, _disp)
            if mats is not None:
                stored.append(an)
            try:
                if mats is not None:
                    raise _Stored          # skip the pooling; panels are drawn below either way
                mats = {}
                from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES
                # THE SHARED POOLING, not a fourth recipe for it -- see `_pooled_bundle`.
                # The call stays INSIDE the try, after `raise _Stored`: hoisting it above
                # would force the pooling even when the stored classes already serve, which
                # is the whole reason they are checked first.
                bd = _pooled_bundle(an, align)
                XE, YE, GE = bd["XE"], bd["YE"], bd["GE"]
                XU, YU, GU = bd["XU"], bd["YU"], bd["GU"]
                pre_i, e_pre, not_eng = bd["pre_i"], bd["e_pre"], bd["not_eng"]
                u_pre = np.isin(GU, list(pre_i)) if len(GU) else np.zeros(0, bool)
                clf = _pipe().fit(XE[e_pre], YE[e_pre])
                name = np.vectorize(lambda v: POSITION_NAMES.get(int(v), str(v)))

                def conf(X, y, clf=clf, name=name):
                    """Counts matrix in CONF_LABELS order. `clf`/`name` bound as defaults because
                    they are loop variables and a late-binding closure would silently score every
                    animal with the LAST animal's decoder."""
                    if not len(y):
                        return None
                    pred = name(clf.predict(X))
                    true = name(y)
                    M = np.zeros((len(CONF_LABELS), len(CONF_LABELS)), float)
                    for t, q in zip(true, pred):
                        if t in CONF_LABELS and q in CONF_LABELS:
                            M[CONF_LABELS.index(t), CONF_LABELS.index(q)] += 1
                    return M

                # THE PRE PANEL MUST BE LEAVE-ONE-SESSION-OUT. Scoring the training trials with the
                # decoder fitted on them gave 0.89-0.99 here against 0.45-0.66 for the same animals
                # in figure 5, and a reader comparing post 0.48 to an in-sample 0.97 would read a
                # collapse that is mostly overfitting. The POST panels need no such care: those
                # trials are held out by construction.
                Cpre = None
                for i in sorted(pre_i):
                    tr = e_pre & (GE != i)
                    te = e_pre & (GE == i)
                    if te.sum() < 5 or len(np.unique(YE[tr])) < 2:
                        continue
                    c1 = conf(XE[te], YE[te], clf=_pipe().fit(XE[tr], YE[tr]))
                    Cpre = c1 if Cpre is None else Cpre + c1
                mats["pre"] = Cpre
                pe, pu = ~e_pre, (~u_pre if len(u_pre) else np.zeros(0, bool))
                Xa = np.vstack([XE[pe]] + ([XU[pu]] if len(u_pre) and pu.any() else []))
                ya = np.concatenate([YE[pe]] + ([YU[pu]] if len(u_pre) and pu.any() else []))
                mats["post_all"] = conf(Xa, ya)
                pw = pu & ~not_eng if len(u_pre) else np.zeros(0, bool)
                Xw = np.vstack([XE[pe]] + ([XU[pw]] if len(u_pre) and pw.any() else []))
                yw = np.concatenate([YE[pe]] + ([YU[pw]] if len(u_pre) and pw.any() else []))
                mats["post_working"] = conf(Xw, yw)
            except _Stored:
                pass                       # the stored per-class matrices are already in `mats`
            except Exception as ex:                                       # noqa: BLE001
                print(f"  !! 5b {an} {align}: {type(ex).__name__} {str(ex)[:90]}", flush=True)
                mats = mats or {}
            for ci, (key, ptitle) in enumerate(PANELS):
                ax = axes[ri][ci]
                C = mats.get(key)
                if C is None:
                    ax.axis("off")
                    continue
                row = C.sum(1, keepdims=True)
                P = np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)
                acc = float(np.trace(C) / C.sum()) if C.sum() else float("nan")
                im = ax.imshow(np.ma.masked_invalid(P), vmin=0, vmax=1, cmap="magma")
                for i in range(len(CONF_LABELS)):
                    if row[i, 0] == 0:
                        ax.text(2.5, i, "no trials", ha="center", va="center", fontsize=8,
                                color="firebrick", fontweight="bold")
                        continue
                    for j in range(len(CONF_LABELS)):
                        if P[i, j] >= 0.02:
                            _txt(ax, j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                                    color="white" if P[i, j] < 0.6 else "black")
                ax.set_xticks(range(len(CONF_LABELS)))
                ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                                   rotation=90, ha="center", fontsize=9.5)
                ax.set_yticks(range(len(CONF_LABELS)))
                ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9.5)
                # Same treatment as figure 5: the animal on the Y LABEL, not in the title where it
                # competes for the panel's width. This figure's titles are the longest in the module
                # ("quit period REMOVED") and the trial count made them longer still, so n moves onto
                # its own line rather than extending the widest one.
                if ci == 0:
                    ax.set_ylabel(an, fontsize=11, fontweight="bold")
                ax.set_title(f"{ptitle}\n({acc:.2f}, n={int(C.sum())})", fontsize=9.5,
                             fontweight="bold" if ci == 0 else "normal")
                drew = True
        if not drew:
            plt.close(fig)
            continue
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.07, label="P(predicted | true)")
        _suptitle(fig, f"Frozen pre-stroke decoder, with and without the terminal quit period — "
                     f"{wname} window\n"
                     "Rows = TRUE spout position, columns = predicted. RIGHT panel drops the "
                     "post-stroke trials after a NON-RECOVERING collapse in responding at the "
                     "positions the animal can still reach;\nlick and miss-while-working trials are "
                     "kept, so the impaired rows still have trials. Post-stroke sessions pooled. "
                     "Chance = 0.17.\nThe quit period is not independently validated as satiety "
                     "rather than a late motor collapse — read this beside figure 5, not instead "
                     "of it.", fontsize=9.5)
        # SAY WHERE THE NUMBERS CAME FROM, and whether that source lags. A figure served from
        # coding_direction.json is only as current as the last position_coding_directions run, and
        # this module already has the scar: on 2026-08-25 the JSONs predated both 8/24 sessions while
        # the config had them, so the footer printed a reassuring "6, 6, 6, 6" on exactly the three
        # stale figures. Reading a stored artifact silently would re-create that.
        if stored:
            lag = sorted(set(config.phase_labels("post")) - set(_cd_labels()))
            print(f"  [5b] {align}: {len(stored)}/{len(ANIMALS)} animal(s) from stored per-class "
                  f"confusions ({', '.join(stored)})"
                  + (f" -- coding_direction.json LAGS the config by {len(lag)} session(s): "
                     f"{', '.join(lag)}; re-run position_coding_directions" if lag else
                     " -- coding_direction.json is current"), flush=True)
        _footer(fig)
        p = Path(out_dir) / f"grant_5b_confusion_working_{align}.png"
        _save(fig, p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
    return made[0] if len(made) == 1 else (made or None)
def fig_confusion_per_session(out_dir):
    """5c: figure 5b unpooled -- one column per post-stroke SESSION, animals aligned by day.

    Priya, 2026-08-25. 5b pools every post-stroke day into one matrix, which is the same objection
    that produced the per-block coding-direction view: pooling averages a recovery and a collapse
    into "no change". PS94's far_R row and PS95's whole matrix move a lot across days and the pooled
    panel cannot show it.

    COLUMNS ARE DAYS FROM LESION, not session index, so a column means the same thing in every row
    even though the animals were lesioned on different dates and PS93 has one fewer post-stroke
    session than the others. A missing session is a blank cell rather than a shift.

    Post-stroke trials are LICK + MISS-WHILE-WORKING with the terminal quit period removed, as in
    5b, and the pre-stroke column is leave-one-session-out for the reason recorded there.
    """
    made = []
    for _disp, align, wname in _windows():
        # THE LICK WINDOW ADMITS ONLY THE LICK CLASS: a miss trial has no lick to align to, so a
        # "miss, lick-aligned" panel is undefined rather than weak. Every other figure in this
        # module already draws both classes for pre-cue and post-cue; 5c and 5d did not, which is
        # why the LICK-ONLY reading of the frozen decoder had no per-session panel at all.
        for variant in _variants(align):
            per_animal, days = _collect_5c(align, variant)
            if not days:
                continue
            p = _draw_5c(per_animal, days, out_dir, align, wname, variant)
            if p:
                made.append(p)
    return made[0] if len(made) == 1 else (made or None)
#: A class this session cannot TRAIN on. Below either floor the within-session refit has no chance
#: of learning the position, and its prediction says something about trial counts rather than about
#: the code -- so those trials are marked unavailable (-1) instead of scored.
#:
#: THE SHARE FLOOR IS THE LOAD-BEARING ONE, and an absolute count is not a substitute for it. What
#: stops a regularised multinomial predicting a class is the PRIOR against it, not the number of
#: examples: at 4% of a session's trials in a six-way problem the model is right to almost never
#: emit that label, and the frozen decoder -- carrying a balanced pre-stroke prior -- is not. A
#: count-only floor of 10 was tried first and made the artefact WORSE (-0.42 -> -0.47), because the
#: two sessions that cleared it were the two with the largest absolute counts and still only 2.6%
#: and 4.3% shares. Set at a third of uniform for six positions.
#:
#: THIS BITES IN EXACTLY ONE PLACE, and it is behavioural rather than technical: the lick-aligned
#: arm conditions on a DETECTED LICK, and acutely the far-contralateral spout is the one the animal
#: does not lick. Every acute session holds far-contra at 0.0-4.3% of its trials against a
#: pre-stroke 16.5%, so the whole cell is gated out and draws nothing -- the same principle figure
#: 10b already applies by scoring an absent position as nothing rather than as a wrong match.
#: Missing data must not become evidence, least of all evidence in the direction that says the code
#: is gone.
MIN_REFIT_CLASS = 10
MIN_REFIT_SHARE = 1.0 / 18.0
#: Sentinel written into the refit column for a trial whose class the session could not train on.
#: Both arms drop these trials together, so the refit-minus-frozen contrast stays paired.
REFIT_UNAVAILABLE = -1
def _matched_frozen(pipe_of, Xp, Yp, Bp, n_target, rng):
    """The frozen estimator fitted on a SIZE-MATCHED random subset of pre-stroke blocks.

    WHY THE COMPARISON NEEDS THIS. The frozen arm trains on ten pre-stroke sessions (~4,500-5,500
    trials) and the refit arm on four fifths of one (~400), so at pre-stroke -- where no lesion has
    happened -- refitting COSTS 0.073 accuracy post-cue and 0.138 pre-cue. That gap is training-set
    size and nothing else, and it is why the raw gap bars cannot be read directly. Matching the
    training-set size leaves WHICH SESSIONS the data came from as the only difference between arms,
    which is the comparison the figure is meant to make.

    WHOLE BLOCKS, NOT RANDOM TRIALS. Blocks are the scheduler's ~6-trial runs at one position and
    they are the unit everything else here resamples. Sampling loose trials would hand the matched
    model a training set with less within-block correlation than the refit model's, which is a
    second difference reintroduced while removing the first.

    Returns ``(fitted, n_used)`` or None when the subset cannot carry two classes.
    """
    ub = rng.permutation(np.unique(Bp))
    keep = np.zeros(len(Yp), bool)
    for b in ub:
        keep |= (Bp == b)
        if keep.sum() >= n_target:
            break
    if keep.sum() < 2 or len(np.unique(Yp[keep])) < 2:
        return None
    return pipe_of().fit(Xp[keep], Yp[keep]), int(keep.sum())
def _refit_pred(pipe_of, X, y, blk, *, max_splits=5):
    """Within-session block-CV prediction on ONE session: what a decoder that SAW this session gets.

    The counterpart to the frozen decoder, and the control that separates the two readings of a
    frozen-decoder failure: a code that is GONE cannot be decoded by any model, while a code that
    has MOVED is decodable within the session and unreadable only by the pre-stroke model. Same
    estimator, same trials, same block grouping -- the only difference is what the model was fitted
    on.

    Returns None rather than guessing when the session cannot support the split (one class, fewer
    than two blocks, or a fold that leaves a training set with one class). A silent fallback to
    something else would put a different kind of number in the same bar.
    """
    from sklearn.model_selection import GroupKFold, cross_val_predict

    b = np.asarray(blk)
    ng = int(np.unique(b).size)
    if ng < 2 or len(np.unique(y)) < 2:
        return None
    try:
        pred = np.asarray(cross_val_predict(pipe_of(), X, y, cv=GroupKFold(min(max_splits, ng)),
                                            groups=b))
    except Exception:                                                 # noqa: BLE001
        return None
    y = np.asarray(y)
    cls, cnt = np.unique(y, return_counts=True)
    thin = cls[(cnt < MIN_REFIT_CLASS) | (cnt / max(len(y), 1) < MIN_REFIT_SHARE)]
    if thin.size:
        pred = pred.copy()
        pred[np.isin(y, thin)] = REFIT_UNAVAILABLE
    return pred
@lru_cache(maxsize=24)
def _collect_5c(align, variant="working", mode="frozen"):
    """{animal: (pre-stroke LOSO record, {day: record})} plus the sorted day list.

    A RECORD IS (y_true, y_pred, blocks), not a counts matrix. Counts are one reduction of it and a
    bootstrap interval is another; keeping the trial-level predictions means the panel's accuracy and
    its interval come from the same object rather than from two passes that can disagree.

    ``mode`` selects WHICH MODEL PRODUCED ``y_pred``, on trial sets that are identical in all three:

      ``frozen``  -- the pre-stroke model, frozen (the default, and what every existing caller gets)
      ``refit``   -- a within-session block-CV decoder fitted on the session being scored
      ``paired``  -- ``y_pred`` is an (n, 2) array: column 0 frozen, column 1 refit

    ``paired`` exists so the refit-minus-frozen contrast is PAIRED at the trial level. Collecting
    the two arms separately and subtracting their pooled accuracies would compare two bootstraps
    that resampled different blocks, and the difference of two intervals is not an interval on the
    difference.

    IN ``refit``/``paired`` THE PRE ENTRY IS A LIST, one record per pre-stroke session, not the
    single concatenated leave-one-session-out record ``frozen`` returns. The refit arm's unit is a
    session -- each pre-stroke session is scored by a model fitted on ITSELF -- so the session level
    is real there and has to be resampled. `epoch_figures._sessions_of` accepts both forms for
    exactly this reason. ``frozen`` is left byte-identical so no existing figure moves.

    ``variant`` selects the post-stroke trial class, as everywhere else in this module:
      ``working`` -- lick PLUS miss-while-working, i.e. all but the terminal quit period
      ``lick``    -- trials with a detected lick only

    THE LICK-ALIGNED WINDOW ADMITS ONLY ``lick``. A trial with no detected lick has no lick to align
    to, so a "miss trial, lick-aligned" panel is not a weak result but an undefined one -- the same
    guard `crossed_confusion` carries as ``include_nolick=False``. The caller is responsible for not
    asking; this raises rather than quietly returning the wrong population.
    """
    if align == "lick" and variant != "lick":
        raise ValueError("the lick-aligned window has no miss trials to align: use variant='lick'")
    if mode not in ("frozen", "refit", "paired", "paired_matched"):
        raise ValueError(f"mode must be frozen/refit/paired/paired_matched, got {mode!r}")

    from wfield_local.locanmf_frozen_decoder import _pipe

    per_animal, all_days = {}, set()
    for an in ANIMALS:
        try:
            # THE SHARED POOLING -- see `_pooled_bundle`, which derives these block ids by
            # the same two rules this site used to repeat: the scheduler's own for the
            # engaged arm, runs of one position for the undetected arm that carries none.
            #
            # This function is lru-cached on (align, variant), so rebuilding the pooling
            # here cost one rebuild PER VARIANT -- five keys x four animals, where twelve
            # memoised bundles serve the entire render.
            bd = _pooled_bundle(an, align)
            XE, YE, GE = bd["XE"], bd["YE"], bd["GE"]
            XU, YU, GU = bd["XU"], bd["YU"], bd["GU"]
            kept, pre_i, e_pre = bd["kept"], bd["pre_i"], bd["e_pre"]
            not_eng, BE_all, BU_all = bd["not_eng"], bd["BE"], bd["BU"]

            clf = _pipe().fit(XE[e_pre], YE[e_pre])

            def rec(X, y, blk, model, pool=None):
                """One record, with `mode` deciding whose prediction fills column(s) of y_pred.

                ``pool`` is the pre-stroke (X, y, blocks) the matched frozen model may draw from --
                for a post-stroke day that is every pre-stroke session, and for the pre column it is
                every pre-stroke session BUT the one being scored, so the matched arm inherits the
                same leave-one-session-out discipline as the unmatched one.
                """
                y, blk = np.asarray(y), np.asarray(blk)
                if mode == "paired_matched":
                    if pool is None:
                        return None
                    k = min(5, int(np.unique(blk).size))
                    # what the refit model gets: (k-1)/k of this session's trials
                    n_target = int(round(len(y) * max(k - 1, 1) / max(k, 1)))
                    # SEEDED PER SCORED SESSION, not per animal. One seed per animal makes
                    # `rng.permutation` return the same block ORDER every time, so every session of
                    # that animal is scored by very nearly the SAME matched model -- one draw
                    # presented as many, and one unlucky subset would bias the whole animal.
                    got = _matched_frozen(_pipe, *pool, n_target, np.random.default_rng(
                        abs(_seed(an, align, variant)) + 1000003 * len(y) + int(blk[0])))
                    if got is None:
                        return None
                    model = got[0]
                fz = np.asarray(model.predict(X))
                if mode == "frozen":
                    return (y, fz, blk)
                rf = _refit_pred(_pipe, X, y, blk)
                if rf is None:
                    return None
                return (y, rf, blk) if mode == "refit" else (y, np.column_stack([fz, rf]), blk)

            # PRE: leave-one-session-out among pre-stroke, concatenated over held-out sessions.
            #
            # THE FROZEN ARM'S PRE BASELINE IS LOSO, and the refit arm's counterpart on the same
            # session is a within-session fit -- so the pre column of the paired figure is not a
            # null contrast but the TRAINING-SET-SIZE effect measured on its own: ten sessions of
            # training data against one, with no lesion involved. Every post-stroke gap has to be
            # read against it, which is why pre is drawn rather than assumed to be zero.
            pre_y, pre_p, pre_b, pre_recs = [], [], [], []
            for i in sorted(pre_i):
                tr = e_pre & (GE != i)
                # THE TRAINING SET IS ALWAYS THE ENGAGED PRE-STROKE ANIMAL -- that is what "frozen"
                # means, and it does not depend on the class being SCORED. What the class changes is
                # the trials the held-out pre-stroke session contributes, and for `stopped` that is
                # its quit period rather than its licking trials.
                #
                # THIS SITE WAS THE ONE `_class_select` DID NOT COVER, and the symptom was a
                # stopped-arm figure whose pre bars read 0.85-0.92 -- the engaged accuracy -- beside
                # post-stroke stopped bars near chance, i.e. an enormous apparent effect that was
                # mostly the two panels being different kinds of trial. Caught 2026-09-11 by the
                # subtitle reporting 44 pre sessions when only 15 pre-stroke sessions have any
                # stopped trials at all.
                # THE PRE PANEL IS THE LICKING SET FOR `lick` AND `working`, and `stopped` is the
                # ONLY class that may take unengaged rows here.
                #
                # REGRESSION I INTRODUCED AND THEN CAUGHT, 2026-09-11. Routing every variant
                # through `_class_select` looked like the tidy thing to do, and for `working` it
                # quietly ADDED the miss-while-working trials to a panel that had always been
                # licking-only: the pooled pre-stroke set went 21,017 -> 22,076 trials and its
                # accuracy 0.89 -> 0.859, moving every pre-stroke position number in the deck. It
                # surfaced only because a NEW figure's pre bar disagreed with the number printed on
                # an OLD one, which is luck, not a guard -- hence
                # `test_pre_panel_is_licking_only_for_lick_and_working`.
                #
                # `_collect_7` states the rule the other collectors follow: a pre-stroke animal is
                # not missing, so `working` adds nothing at pre except a different KIND of trial,
                # and the reference then differs from itself.
                if variant == "stopped":
                    me_te, mu_te = _class_select(variant, e_pre & (GE == i),
                                                 (GU == i) if len(GU) else np.zeros(0, bool),
                                                 not_eng)
                else:
                    me_te = e_pre & (GE == i)
                    mu_te = np.zeros(len(GU), bool)
                Xte = np.vstack([XE[me_te]] + ([XU[mu_te]] if mu_te.any() else []))
                yte = np.concatenate([YE[me_te]] + ([YU[mu_te]] if mu_te.any() else []))
                bte = np.concatenate([BE_all[me_te]] + ([BU_all[mu_te]] if mu_te.any() else []))
                if len(yte) < 5 or len(np.unique(YE[tr])) < 2:
                    continue
                r = rec(Xte, yte, bte, _pipe().fit(XE[tr], YE[tr]),
                        pool=(XE[tr], YE[tr], BE_all[tr]))
                if r is None:
                    continue
                pre_recs.append(r)
                pre_y.append(r[0]); pre_p.append(r[1]); pre_b.append(r[2])
            if mode == "frozen":
                Cpre = ((np.concatenate(pre_y), np.concatenate(pre_p), np.concatenate(pre_b))
                        if pre_y else None)
            else:
                Cpre = pre_recs or None

            by_day = {}
            for i, lab in enumerate(kept):
                if i in pre_i:
                    continue
                day = _day(an, lab.split("_")[-1])
                me, mu = _class_select(variant, (GE == i), (GU == i), not_eng)
                Xs = np.vstack([XE[me]] + ([XU[mu]] if mu.any() else []))
                ys = np.concatenate([YE[me]] + ([YU[mu]] if mu.any() else []))
                bs = np.concatenate([BE_all[me]] + ([BU_all[mu]] if mu.any() else []))
                if not len(ys):
                    continue
                r = rec(Xs, ys, bs, clf, pool=(XE[e_pre], YE[e_pre], BE_all[e_pre]))
                if r is None:
                    continue
                by_day[day] = r
                all_days.add(day)
            per_animal[an] = (Cpre, by_day)
        except Exception as ex:                                       # noqa: BLE001
            print(f"  !! 5c {an} {align}/{variant}/{mode}: {type(ex).__name__} {str(ex)[:90]}",
                  flush=True)
    return per_animal, sorted(all_days)
def _counts(record):
    """Confusion counts in CONF_LABELS order from a (y_true, y_pred, blocks) record."""
    from wfield_local.locanmf_cue_lick_analysis import POSITION_NAMES

    if record is None:
        return None
    y, p, _b = record
    M = np.zeros((len(CONF_LABELS), len(CONF_LABELS)), float)
    for t, q in zip(y, p):
        tn = POSITION_NAMES.get(int(t), str(t))
        qn = POSITION_NAMES.get(int(q), str(q))
        if tn in CONF_LABELS and qn in CONF_LABELS:
            M[CONF_LABELS.index(tn), CONF_LABELS.index(qn)] += 1
    return M
def _acc_ci(record, n_boot=400):
    """(accuracy, lo, hi, n_blocks) by cluster bootstrap over the scheduler's own trial blocks.

    Reuses `decode_ci.bootstrap_recall` rather than adding a second implementation of the same
    resampling -- it already resamples BLOCKS, which is the unit these figures use everywhere else,
    and it already reports n_effective, the honest sample size.
    """
    if record is None:
        return (float("nan"),) * 3 + (0,)
    from wfield_local.decode_ci import bootstrap_recall

    y, p, b = record
    if not len(y):
        return (float("nan"),) * 3 + (0,)
    try:
        out = bootstrap_recall(y, p, blocks=b, n_boot=n_boot)
        lo, hi = out["accuracy_ci"]
        return float(out["accuracy"]), float(lo), float(hi), int(out["n_effective"])
    except Exception as exc:                                          # noqa: BLE001
        print(f"  [5c] accuracy CI unavailable ({type(exc).__name__})", flush=True)
        acc = float((np.asarray(y) == np.asarray(p)).mean())
        return acc, float("nan"), float("nan"), 0
def _draw_5c(per_animal, days, out_dir, align, wname, variant="working"):
    """The absolute rendering of 5c. Split from the collector so 5d reuses the same numbers."""
    ncol = 1 + len(days)
    # Taller with more room between rows than the width alone would need: at the wider `_colw`
    # the two-line panel titles ("day N" over the accuracy and its interval) reach into the axes
    # above. See `_colw` -- width and vertical space are opposing knobs here.
    fig, axes = plt.subplots(len(ANIMALS), ncol, figsize=(_colw() * ncol + 1.2, 9.5),
                             gridspec_kw={"hspace": 0.60}, squeeze=False)
    im = None
    for ri, an in enumerate(ANIMALS):
        got = per_animal.get(an)
        for ci in range(ncol):
            ax = axes[ri][ci]
            record = None if not got else (got[0] if ci == 0 else got[1].get(days[ci - 1]))
            C = _counts(record)
            if C is None or not C.sum():
                ax.axis("off")
                continue
            row = C.sum(1, keepdims=True)
            P = np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)
            # ACCURACY AND ITS INTERVAL FROM THE SAME RECORD. Both are reductions of the same
            # (y_true, y_pred, blocks), so the number and its uncertainty cannot come from
            # different populations.
            acc, lo, hi, _nb = _acc_ci(record)
            im = ax.imshow(np.ma.masked_invalid(P), vmin=0, vmax=1, cmap="magma")
            ax.set_xticks(range(len(CONF_LABELS)))
            ax.set_yticks(range(len(CONF_LABELS)))
            ax.set_xticklabels(_short(CONF_LABELS) if ri == len(ANIMALS) - 1 else [],
                               rotation=90, fontsize=9)
            ax.set_yticklabels(_short(CONF_LABELS) if ci == 0 else [], fontsize=9)
            head = "PRE" if ci == 0 else f"day {days[ci - 1]}"
            band = f"\n[{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) and np.isfinite(hi) else ""
            ax.set_title(f"{head}\n{acc:.2f}{band}", fontsize=8.5,
                         fontweight="bold" if ci == 0 else "normal")
            if ci == 0:
                ax.set_ylabel(f"{an}\ntrue position", fontsize=11, fontweight="bold")
    if im is None:
        plt.close(fig)
        return None
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02, label="P(predicted | true)")
    cls = _class_note(variant)
    _suptitle(fig, f"Frozen pre-stroke decoder, session by session — {wname} window\n"
                   f"Post-stroke class: {cls}. Columns are DAYS FROM LESION so they mean the same "
                   f"thing in every row; a blank cell is a session that animal does not have.\n"
                   f"Rows = TRUE spout position, columns within a panel = predicted. "
                   f"Chance = 0.17.\n"
                   f"Below each accuracy: its 95% CLUSTER-BOOTSTRAP interval, resampling the "
                   f"scheduler's own ~6-trial position blocks. Blocks, not trials -- trials next to "
                   f"each other in time are not independent, and a trial-level interval would be "
                   f"several times too tight.", fontsize=9.5)
    _footer(fig)
    p = _out(out_dir, f"grant_5c_confusion_per_session_{align}_{variant}")
    _save(fig, p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p
def fig_confusion_delta(out_dir):
    """5d: figure 5c as DIFFERENCES from the pre-stroke reference.

    The frozen decoder's per-session confusion minus the leave-one-session-out pre-stroke confusion,
    cell by cell, in row-normalised probability. Zero means the decoder makes the same errors in the
    same proportions it made before the lesion.

    WHY THIS IS THE MOST INFORMATIVE OF THE THREE. The pre-stroke confusion is far from uniform --
    close positions are confusable with each other and far ones are not -- so an absolute
    post-stroke cell of 0.3 means different things in different places. Subtracting removes the
    baseline structure and leaves only what the lesion did: a negative diagonal cell is recall lost
    at that position, and the positive cell in the same ROW says where those trials went instead.
    """
    made = []
    for _disp, align, wname in _windows():
        # Same window/class coverage as 5c, and the same guard: a miss trial has no lick to align to.
        for variant in _variants(align):
            per_animal, days = _collect_5c(align, variant)
            if not days or not per_animal:
                continue

            def _norm(C):
                row = C.sum(1, keepdims=True)
                return np.divide(C, row, out=np.full_like(C, np.nan), where=row > 0)

            mats = {}
            for an, (Cpre, by_day) in per_animal.items():
                Cp = _counts(Cpre)
                if Cp is None or not Cp.sum():
                    continue
                d = {"PRE": _norm(Cp)}
                for day, record in by_day.items():
                    C = _counts(record)
                    if C is not None and C.sum():
                        d[day] = _norm(C)
                mats[an] = d
            cls = _class_note(variant)
            p = _delta_grid(
                mats, days, out_dir, f"grant_5d_confusion_delta_{align}_{variant}.png",
                title=(f"Frozen pre-stroke decoder, CHANGE FROM PRE-STROKE — {wname} window\n"
                       f"Post-stroke class: {cls}. Column 1 is the pre-stroke confusion "
                       f"(leave-one-session-out);\n"
                       f"every later column is THAT DAY MINUS IT. Rows = TRUE spout position, columns "
                       f"within a panel = predicted. ZERO = the same errors in the same proportions.\n"
                       f"A negative DIAGONAL cell is recall lost at that position; the positive cell in "
                       f"the SAME ROW says where those trials went instead. The number above each panel "
                       f"is the change in overall accuracy."),
                abs_label="pre-stroke P(pred | true)",
                delta_label="change in P(predicted | true)",
                vmin=0, vmax=1, cmap="magma", dmax=0.6, summary=_diag,
                ylab="true position", figh=9.0)
            if p:
                made.append(p)
    return made

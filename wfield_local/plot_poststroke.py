"""Figures for deck Section G — POST-STROKE, read from the saved JSON rather than recomputed.

ORDER IS THE ARGUMENT. Behaviour comes first (G1), because on 8/17 both animals stopped attempting
the far positions -- PS94 has ZERO engaged trials at far_center and far_R -- and every decoding
number after it is uninterpretable without that. Putting decoding first is exactly the error that
produced a "PS94 neural deficit" headline which was mostly trial composition.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402

POS = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]


def fig_behaviour(counts, out):
    """G1: per-position engaged / undetected counts, pre vs post. The primary finding."""
    ans = sorted(counts)
    fig, axes = plt.subplots(2, len(ans), figsize=(5.2 * len(ans), 6.4), squeeze=False)
    x = np.arange(len(POS))
    w = 0.38
    for k, an in enumerate(ans):
        c = counts[an]
        for r, arm in enumerate(("engaged", "undetected")):
            ax = axes[r][k]
            ax.bar(x - w / 2, [c["pre"][arm].get(p, 0) for p in POS], w,
                   label="pre-stroke (per-session mean)", color="tab:blue",
                   edgecolor="k", linewidth=0.4)
            ax.bar(x + w / 2, [c["post"][arm].get(p, 0) for p in POS], w,
                   label="post-stroke 8/17", color="tab:red", edgecolor="k", linewidth=0.4)
            for xi, p in zip(x + w / 2, POS):
                if c["post"][arm].get(p, 0) == 0:
                    ax.text(xi, 1, "0", ha="center", va="bottom", fontsize=8,
                            color="firebrick", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(POS, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{an} — {arm} trials", fontsize=10)
            ax.set_ylabel("trials")
            if r == 0 and k == 0:
                ax.legend(fontsize=7)
    fig.suptitle("POST-STROKE BEHAVIOUR FIRST: which positions the animal still attempts. "
                 "Zero engaged trials at a position means no decoding number for it can exist.",
                 fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = Path(out) / "poststroke_G1_behaviour.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_matched(matched, out):
    """G2: position-matched decoding in each condition against the pre-stroke LOSO band."""
    ans = sorted(matched)
    conds = ["post-cue", "post-lick", "pre-cue"]
    fig, axes = plt.subplots(1, len(ans), figsize=(4.6 * len(ans), 4.4), squeeze=False)
    for k, an in enumerate(ans):
        ax = axes[0][k]
        m = matched[an]
        for i, cname in enumerate(conds):
            r = m.get(cname)
            if not r:
                continue
            b = r["pre_band"]
            ax.add_patch(plt.Rectangle((i - 0.34, b["min"]), 0.68, b["max"] - b["min"],
                                       color="tab:blue", alpha=0.20, zorder=1))
            ax.plot([i - 0.34, i + 0.34], [b["mean"]] * 2, color="tab:blue", lw=2, zorder=2)
            ax.plot([i], [r["accuracy"]], "o", ms=11, color="tab:red", zorder=3,
                    markeredgecolor="k")
            if r.get("below_every_pre_session"):
                ax.text(i, max(r["accuracy"] - 0.055, 0.02), "below all", ha="center",
                        fontsize=7.5, color="firebrick", fontweight="bold")
        ax.axhline(0.25, color="k", ls=":", lw=1)
        ax.text(len(conds) - 0.5, 0.26, "chance (4-way)", fontsize=7, ha="right")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{an} — position-matched", fontsize=10)
        ax.set_ylabel("accuracy")
    fig.suptitle("Frozen PRE-stroke decoder on POST-stroke trials, matched to the positions the "
                 "animal still attempts. Band = pre-stroke leave-one-session-out range under the "
                 "SAME restriction. 4-way: NOT comparable to 6-way numbers elsewhere in this deck.",
                 fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = Path(out) / "poststroke_G2_matched.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_identity(ident, out):
    """G4: do post-stroke no-lick trials look like pre-stroke licking? WITH the control."""
    ans = sorted(ident)
    fig, ax = plt.subplots(figsize=(3.2 * len(ans) + 4.0, 4.8))
    x = np.arange(len(ans))
    w = 0.2
    keys = [("scale_pre_engaged", "pre ENGAGED", "tab:blue"),
            ("scale_pre_undetected", "pre UNDETECTED", "tab:grey"),
            ("CONTROL_post_engaged_frac_engaged_like", "post ENGAGED (control)", "tab:green"),
            ("post_undetected_frac_classified_ENGAGED_like", "post UNDETECTED", "tab:red")]
    for j, (k, lab, col) in enumerate(keys):
        ax.bar(x + (j - 1.5) * w, [ident[a].get(k, np.nan) for a in ans], w, label=lab,
               color=col, edgecolor="k", linewidth=0.4)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    for i, a in enumerate(ans):
        d = ident[a]
        ok = d.get("boundary_still_discriminates_post")
        ax.text(i, 1.02, ("control PASSES — read the red bar"
                          if ok else "CONTROL FAILS — do NOT read the red bar"),
                ha="center", fontsize=8.5, fontweight="bold",
                color=("darkgreen" if ok else "firebrick"))
        ax.text(i, -0.12, f"pre-separability {d.get('pre_separability_cv', float('nan')):.2f}"
                          f"   n post-undetected = {d.get('n_post_undetected', 0)}",
                ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ans)
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("fraction classified ENGAGED-like (licking)")
    # OUTSIDE the axes: at loc="lower center" it covered the pre-UNDETECTED bars, which are the
    # baseline the whole comparison is against.
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              frameon=False)
    fig.suptitle("Do post-stroke NO-LICK trials look like pre-stroke LICKING trials?  "
                 "Boundary trained on PRE-stroke engaged-vs-undetected, position-balanced so it "
                 "cannot simply answer 'far'.  READ THE CONTROL FIRST: post ENGAGED (green) must sit "
                 "ABOVE post UNDETECTED (red), or the boundary is tracking 'post-stroke' rather than "
                 "licking and the answer means nothing.", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    p = Path(out) / "poststroke_G4_identity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_similarity(sim, out):
    """G5: per-position pattern correlation between phases — same code or different code?"""
    ans = sorted(sim)
    fig, ax = plt.subplots(figsize=(3.0 * len(ans) + 4.4, 4.2))
    x = np.arange(len(POS))
    w = 0.8 / max(len(ans), 1)
    for k, an in enumerate(ans):
        vals = [sim[an].get(p, {}).get("r", np.nan) for p in POS]
        ax.bar(x + (k - (len(ans) - 1) / 2) * w, vals, w, label=an, edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="k", lw=1)
    # An empty column reads as r = 0, i.e. "no similarity", when it means "the animal stopped
    # attempting this position so there is no post-stroke pattern to correlate against".
    for xi, q in zip(x, POS):
        if all(sim[a].get(q, {}).get("r") is None or q not in sim[a] for a in ans):
            ax.text(xi, 0.04, "not attempted", ha="center", va="bottom", fontsize=7,
                    rotation=90, color="firebrick", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(POS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("r (pre-stroke vs post-stroke mean pattern)")
    ax.set_ylim(-0.6, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Is it the same code at reduced strength, or a different code?\n"
                 "Per-position correlation of the mean activity pattern; decoding accuracy alone "
                 "cannot separate those.", fontsize=9)
    fig.tight_layout()
    p = Path(out) / "poststroke_G5_similarity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_confusion(conf, out):
    """G3: crossed confusion — frozen pre-stroke decoder, pre vs post, full 6x6."""
    ans = sorted(conf)
    fig, axes = plt.subplots(len(ans), 2, figsize=(10.4, 4.8 * len(ans)), squeeze=False)
    for r, an in enumerate(ans):
        for c, phase in enumerate(("pre", "post")):
            ax = axes[r][c]
            d = conf[an][phase]
            M = np.array(d["matrix"], dtype=float)
            im = ax.imshow(np.ma.masked_invalid(M), vmin=0, vmax=1, cmap="magma")
            ax.set_xticks(range(len(POS)))
            ax.set_xticklabels(POS, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(POS)))
            ax.set_yticklabels([f"{p} (n={n})" for p, n in
                                zip(POS, d["n_per_true_position"])], fontsize=7)
            for i in range(len(POS)):
                if d["n_per_true_position"][i] == 0:
                    ax.text(len(POS) / 2 - 0.5, i, "no trials attempted", ha="center",
                            va="center", fontsize=8, color="firebrick", fontweight="bold")
            ttl = "PRE (leave-one-session-out)" if phase == "pre" else "POST (frozen pre-stroke model)"
            ax.set_title(f"{an} — {ttl}", fontsize=9)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Crossed confusion: the frozen PRE-stroke decoder applied to POST-stroke trials. "
                 "The diagonal is per-position recall; the OFF-diagonal is where the errors go, "
                 "which a scalar accuracy discards.", fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = Path(out) / "poststroke_G3_confusion.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p



CONDS = ["post-cue", "post-lick", "pre-cue WITH lick", "pre-cue NO lick"]
MIN_N = 10        # below this a per-position recall is marked, not trusted


def fig_per_position(pp, out):
    """G2b: per-position recall pre vs post in all four decoder conditions (Priya, 2026-08-17).

    One row per condition, one column per animal; pre-stroke and post-stroke bars side by side at
    every position. A position the animal stopped attempting gets an explicit "n/a" mark rather than
    a zero bar -- the distinction the scalar summaries kept losing, and the one that turned a "PS94
    neural deficit" headline into a statement about trial composition.
    """
    ans = sorted(pp)
    fig, axes = plt.subplots(len(CONDS), len(ans), figsize=(6.0 * len(ans), 3.1 * len(CONDS)),
                             squeeze=False, sharey=True)
    x = np.arange(len(POS))
    w = 0.38
    for r, cond in enumerate(CONDS):
        for k, an in enumerate(ans):
            ax = axes[r][k]
            d = pp[an].get(cond)
            if not d:
                ax.text(0.5, 0.5, cond + ": insufficient trials", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="firebrick")
                ax.set_xticks([])
                continue
            for j, (ph, col) in enumerate((("pre", "tab:blue"), ("post", "tab:red"))):
                vals = [d[ph].get(q, {}).get("recall", np.nan) for q in POS]
                ns = [d[ph].get(q, {}).get("n", 0) for q in POS]
                xs = x + (j - 0.5) * w
                ax.bar(xs, [v if v == v else 0 for v in vals], w, color=col,
                       edgecolor="k", linewidth=0.4,
                       label=("pre-stroke (LOSO)" if ph == "pre" else "post-stroke (frozen)"))
                # A recall computed on a handful of trials is not a number a reader should weigh the
                # same as one computed on a hundred, and the extreme values are exactly where n is
                # smallest: PS95 far_R post-stroke reads 1.00 off ONE trial.
                for xi, n, v in zip(xs, ns, vals):
                    if n == 0:
                        ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=6.5,
                                rotation=90, color="firebrick", fontweight="bold")
                    elif n < MIN_N:
                        ax.bar(xi, v if v == v else 0, w, color="none", edgecolor="firebrick",
                               linewidth=1.1, hatch="////", zorder=3)
                        ax.text(xi, (v if v == v else 0) + 0.03, f"n={n}", ha="center",
                                va="bottom", fontsize=6, color="firebrick", fontweight="bold")
            ax.axhline(1 / 6, color="k", ls=":", lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels(POS if r == len(CONDS) - 1 else [], rotation=45, ha="right",
                               fontsize=7)
            ax.set_ylim(0, 1.05)
            if k == 0:
                ax.set_ylabel(cond + "\nrecall", fontsize=8)
            ax.set_title(f"{an} — {cond}  (balanced {d['pre_balanced']:.2f} -> "
                         f"{d['post_balanced']:.2f})", fontsize=8.5)
            if r == 0 and k == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Per-position recall in FOUR conditions. Training is ALWAYS on pre-stroke ENGAGED "
                 "trials;\n'with/without lick' is the RESPONSE lick, i.e. engaged vs undetected "
                 "trials. Dotted line = 1/6.\n'n/a' = position not attempted (NOT zero recall);   "
                 f"RED HATCHED = fewer than {MIN_N} trials, do not weigh these",
                 fontsize=9.5, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    q = Path(out) / "poststroke_G2b_per_position.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q


def fig_nolick_readout(rd, out):
    """G6: was a PLAN formed on post-stroke no-lick trials? IMPAIRED vs PRESERVED positions.

    Replaces the working/disengaged split, retired 2026-08-18: "disengaged" has no valid post-stroke
    construction, so that figure compared against a class that was never established. This contrast
    splits on the true spout position, which is measured rather than inferred, and needs no
    engagement label anywhere in it.
    """
    ans = sorted(rd)
    fig, axes = plt.subplots(1, len(ans), figsize=(5.4 * len(ans) + 1.0, 5.6), squeeze=False)
    for k, an in enumerate(ans):
        ax = axes[0][k]
        xt, lab, i = [], [], 0.0
        for al in ("precue", "cue"):
            r = rd[an].get(al, {})
            for arm, col in (("preserved", "tab:green"), ("impaired", "tab:red")):
                a = r.get(arm, {})
                if "balanced_accuracy" in a:
                    ax.bar(i, a["balanced_accuracy"], 0.66, color=col, edgecolor="k", linewidth=0.5)
                    ax.plot([i - 0.33, i + 0.33], [a["bal_null_mean"]] * 2, color="k", lw=2.2, zorder=3)
                    # Star on the CORRECTED p, matching the verdict under the panel. Starring
                    # the raw p put a "*" on PS95 cue/impaired while the caption on the same
                    # panel said the effect was not supported -- the reader believes the star.
                    praw = a["bal_p"]
                    pc_ = a.get("bal_p_bonferroni_x8", min(1.0, praw * 8))
                    star = ("***" if pc_ < 0.001 else "**" if pc_ < 0.01 else
                            "*" if pc_ < 0.05 else "n.s.")
                    ax.text(i, a["balanced_accuracy"] + 0.02,
                            f"{star}\nn={a['n']}\np={praw:.3f} raw",
                            ha="center", fontsize=6.5)
                else:
                    ax.text(i, 0.04, "n=" + str(a.get("n", 0)) + "\ntoo few", ha="center",
                            fontsize=7, color="dimgrey")
                xt.append(i)
                lab.append(al + "\n" + arm)
                i += 1
            i += 0.4
        ax.set_xticks(xt)
        ax.set_xticklabels(lab, fontsize=7.5)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("balanced accuracy (6-way)")
        ax.set_title(an, fontsize=11, fontweight="bold")
        # The verdict wrapped UNDER the panel. Untreated, two of these sentences overlapped into an
        # unreadable overlay -- and the verdict is the figure's conclusion, so losing it loses the
        # slide. Only the pre-cue verdict is shown: post-cue on a no-lick trial is ambiguous because
        # the spout is physically present (see impaired_nolick_readout).
        v = rd[an].get("precue", {}).get("verdict", "")
        v = v.split(". CAVEAT")[0]
        ax.set_xlabel("PRE-CUE verdict: " + textwrap.fill(v, 62), fontsize=7, labelpad=8)
    fig.suptitle("Post-stroke NO-LICK trials: was the position represented anyway?\n"
                 "Frozen pre-stroke decoder; BLACK RULE = that arm's own permutation null. Above the "
                 "null at IMPAIRED positions = plan formed, movement failed. CAVEAT: 'no lick "
                 "detected' is not 'no tongue protrusion' -- DLC replaces this inference with a "
                 "measurement. Stars are Bonferroni-corrected for the 8 tests in a pass; the raw p "
                 "is printed under each bar.", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    q = Path(out) / "poststroke_G6_nolick_readout.png"
    fig.savefig(q, dpi=150)
    plt.close(fig)
    return q

"""Figures for deck Section G — POST-STROKE, read from the saved JSON rather than recomputed.

ORDER IS THE ARGUMENT. Behaviour comes first (G1), because on 8/17 both animals stopped attempting
the far positions -- PS94 has ZERO engaged trials at far_center and far_R -- and every decoding
number after it is uninterpretable without that. Putting decoding first is exactly the error that
produced a "PS94 neural deficit" headline which was mostly trial composition.
"""
from __future__ import annotations

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
                 fontsize=10)
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
                 fontsize=9)
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
        ax.text(i, 1.05, ("control PASSES" if ok else "CONTROL FAILS — not interpretable"),
                ha="center", fontsize=8.5, fontweight="bold",
                color=("darkgreen" if ok else "firebrick"))
        ax.text(i, -0.12, f"pre-separability {d.get('pre_separability_cv', float('nan')):.2f}"
                          f"   n post-undetected = {d.get('n_post_undetected', 0)}",
                ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ans)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("fraction classified ENGAGED-like (licking)")
    ax.legend(fontsize=7.5, ncol=2, loc="lower center")
    ax.set_title("Do post-stroke NO-LICK trials look like pre-stroke LICKING trials?\n"
                 "Boundary trained on PRE-stroke engaged-vs-undetected, position-balanced. The "
                 "CONTROL (post ENGAGED) must sit ABOVE post UNDETECTED, or the boundary is "
                 "tracking 'post-stroke' rather than licking and the answer means nothing.",
                 fontsize=9)
    fig.tight_layout()
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
                 "which a scalar accuracy discards.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = Path(out) / "poststroke_G3_confusion.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
